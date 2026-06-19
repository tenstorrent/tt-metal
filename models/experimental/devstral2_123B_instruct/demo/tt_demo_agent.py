# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Text-only Tenstorrent agent demo for Devstral-2-123B (Ministral3).

Mirrors ``models/experimental/devstral2_small/demo/tt_demo_agent.py``:

  * session KV prefix cache: system + tool rules are prefilled once; later turns
    only prefill new suffix tokens at ``start_pos = kv_seq_len`` (prefix validation
    with fallback to full re-prefill on mismatch),
  * chunked paged prefill (``kv_block_size`` tokens per dispatch), then
  * traced decode (default): one capture per generation, ``execute_trace`` for later
    tokens; persistent ``(token, current_pos)`` device buffers updated each step.
  * decode trace **2CQ** (default, ``DEVSTRAL2_DECODE_TRACE_2CQ=1``): CQ1 for input H2D,
    CQ0 for forward/trace replay. Use ``--no-decode-trace-2cq`` or ``DEVSTRAL2_DECODE_TRACE_2CQ=0``.

Sampling: decode uses **on-device argmax** by default (force-argmax over the column-parallel
``lm_head`` logits via ``SamplingGenerator``), run **after** the decode trace replay — never
folded into the trace, so its CCL/argmax buffers do not share L1 with the 123B decode graph.
Only one ``int32`` token is read back per step instead of the full sharded vocab row. Set
``DEVSTRAL2_ONDEVICE_SAMPLING=0`` to fall back to CPU sampling (argmax/multinomial), which also
honours ``--temperature`` / ``--top-p``. With on-device sampling, the prefill first token uses
the same device argmax path as decode (last prompt row sliced from the final prefill chunk).

Run::

    python models/experimental/devstral2_123B_instruct/demo/tt_demo_agent.py [--mesh-device T3K] [--num-layers 88]

Use ``--verbose`` to print per-turn KV/prefill/decode-trace progress (off by default).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.devstral2_123B_instruct.demo.decode_trace_2cq import (
    DecodeTrace2CQ,
    decode_trace_2cq_enabled,
    signal_decode_step_done,
    stage_decode_inputs,
)
from models.experimental.devstral2_123B_instruct.demo.on_device_sampling import (
    OnDeviceSampler,
    build_sampler,
    on_device_sampling_enabled,
    quiet_per_token_sampling_logs,
    sample_first_token_from_prefill_logits,
    supports_on_device_sampling,
)
from models.experimental.devstral2_123B_instruct.demo.text_demo import (
    _current_pos_to_tt,
    _input_ids_host,
    _input_ids_to_tt,
    _logits_to_torch,
)
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    model_prefill_weight_keys,
    require_hf_weights,
    require_text_config,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_model import (
    TtMinistral3ForCausalLM,
)
from models.experimental.devstral2_123B_instruct.tt.weight_loading import DEVSTRAL2_LARGE_REPO_ID
from models.tt_transformers.tt.ccl import TT_CCL


DEFAULT_SYSTEM_PROMPT = (
    "You are Devstral Agent, a concise and practical coding assistant. "
    "Help with debugging, implementation, and code review in a local repository."
)
DEFAULT_AGENT_RULES = """
Tools: read_file, write_file, search_replace, grep, todo, ask_user_question, load_skill,
inspect_codebase.

When you need a tool, output ONLY the tool call — no preamble, no markdown code fences, no plain-text
explanation on that turn. Prefer a single-line Devstral native call, e.g.
  read_file{"path":"src/main.py","offset":1,"limit":80}
or XML:
  <tool_call>
  {"name":"read_file","arguments":{"path":"src/main.py","offset":1,"limit":80}}
  </tool_call>

Rules:
- Exactly one tool call per tool turn.
- Always close all JSON braces.
- Use ``path``, not ``file_path``, for file tools.
- Shell, web search, and web fetch are not available; use read_file/write_file/search_replace/grep/inspect_codebase.
- After ``<tool_result>...</tool_result>``: another tool call (same format) or a plain-text answer.
(Closing ``</tool_call>`` is optional.)
"""

_KNOWN_TOOL_NAMES = frozenset(
    {
        "read_file",
        "write_file",
        "search_replace",
        "grep",
        "todo",
        "ask_user_question",
        "load_skill",
        "inspect_codebase",
    }
)


@dataclass
class ChatConfig:
    model_id: str
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    device: str
    system_prompt: str
    workspace_root: str
    max_tool_calls_per_turn: int


@dataclass
class AgentState:
    todos: List[Dict[str, str]] = field(default_factory=list)


def _extract_balanced_json_object(text: str, start: int) -> Optional[Tuple[str, int]]:
    """Return (json substring, index after closing brace) when ``text[start]`` is ``{``.

    Stops at ``</tool_call>`` if present. When the model truncates before the outer
    ``}`` when the model truncates mid-object), append missing closing braces so
    ``json.loads`` can still succeed.
    """
    if start >= len(text) or text[start] != "{":
        return None
    close_tag = text.find("</tool_call>", start)
    limit = close_tag if close_tag != -1 else len(text)
    depth = 0
    in_string = False
    escape = False
    for i in range(start, limit):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], i + 1
    if depth > 0 and not in_string:
        repaired = text[start:limit].strip() + ("}" * depth)
        return repaired, limit
    return None


def _normalize_tool_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Map Devstral-native argument names to what ``execute_tool_call`` expects."""
    out = dict(args)
    if name in ("write_file", "read_file", "search_replace") and "path" not in out and "file_path" in out:
        out["path"] = out.pop("file_path")
    return out


def _coerce_tool_payload(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize XML or native Devstral payloads to ``{name, arguments}``."""
    if "name" in payload:
        tool_name = str(payload["name"])
        raw_args = payload.get("arguments", payload)
    else:
        tool_name = name
        raw_args = payload
    if not isinstance(raw_args, dict):
        raw_args = {}
    else:
        raw_args = dict(raw_args)
        raw_args.pop("name", None)
    return {"name": tool_name, "arguments": _normalize_tool_args(tool_name, raw_args)}


def _parse_xml_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse ``<tool_call>{...}</tool_call>`` (closing tag optional)."""
    marker = "<tool_call>"
    idx = text.find(marker)
    if idx == -1:
        return None
    pos = idx + len(marker)
    while pos < len(text) and text[pos] in " \t\n\r":
        pos += 1
    extracted = _extract_balanced_json_object(text, pos)
    if extracted is None:
        return None
    json_str, _ = extracted
    try:
        payload = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "name" not in payload:
        return None
    return _coerce_tool_payload(str(payload["name"]), payload)


def _parse_devstral_native_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse ``write_file{...}`` / ``grep{...}`` style calls (no XML wrapper)."""
    best_idx: Optional[int] = None
    best: Optional[Dict[str, Any]] = None
    for tool in _KNOWN_TOOL_NAMES:
        needle = f"{tool}{{"
        idx = text.find(needle)
        if idx == -1:
            continue
        pos = idx + len(tool)
        extracted = _extract_balanced_json_object(text, pos)
        if extracted is None:
            continue
        json_str, _ = extracted
        try:
            payload = json.loads(json_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if best_idx is None or idx < best_idx:
            best_idx = idx
            best = _coerce_tool_payload(tool, payload)
    return best


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse the first tool call from assistant text.

    Supports:
      * ``<tool_call>{"name":...,"arguments":{...}}</tool_call>`` (demo protocol)
      * Devstral native ``tool_name{...}`` (what the HF model often emits)

    Brace-balanced JSON extraction handles truncated ``</tool_call>`` and long
    ``write_file`` content strings.
    """
    return _parse_xml_tool_call(text) or _parse_devstral_native_tool_call(text)


def _log_unparsed_tool_output(assistant_text: str) -> None:
    """Log raw model output when a tool call was started but JSON parsing failed."""
    preview_len = 2000
    text = assistant_text.strip()
    if len(text) <= preview_len:
        body = text
    else:
        body = f"{text[:preview_len]}\n... [{len(text) - preview_len} chars truncated]"
    logger.warning(
        "Tool call parse failed (model output looked like a tool but JSON was incomplete). "
        f"Raw assistant output ({len(assistant_text)} chars):\n{body}"
    )


def _limit_text(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _resolve_workspace_path(workspace_root: str, raw_path: str) -> Path:
    root = Path(workspace_root).resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Path escapes workspace_root.")
    return candidate


def _tool_path_arg(args: Dict[str, Any]) -> str:
    return str(args.get("path") or args.get("file_path") or "")


def tool_read_file(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    try:
        path = _resolve_workspace_path(config.workspace_root, _tool_path_arg(args))
        offset = int(args.get("offset", 1))
        limit = int(args.get("limit", 200))
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max(offset - 1, 0)
        end = min(start + max(limit, 1), len(lines))
        rendered = "\n".join(f"{idx + 1}|{lines[idx]}" for idx in range(start, end))
        return {"ok": True, "path": str(path), "output": _limit_text(rendered)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_write_file(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    try:
        path = _resolve_workspace_path(config.workspace_root, _tool_path_arg(args))
        content = str(args.get("content", ""))
        append = bool(args.get("append", False))
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": str(path), "bytes_written": len(content.encode("utf-8"))}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_search_replace(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    try:
        path = _resolve_workspace_path(config.workspace_root, _tool_path_arg(args))
        search = str(args.get("search", ""))
        replace = str(args.get("replace", ""))
        max_replacements = int(args.get("max_replacements", 0))
        text = path.read_text(encoding="utf-8")
        if max_replacements <= 0:
            count = text.count(search)
            updated = text.replace(search, replace)
        else:
            updated = text.replace(search, replace, max_replacements)
            count = min(text.count(search), max_replacements)
        path.write_text(updated, encoding="utf-8")
        return {"ok": True, "path": str(path), "replacements": count}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_grep(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    pattern = str(args.get("pattern", ""))
    rel_path = str(args.get("path", "."))
    glob_pattern = str(args.get("glob", "*"))
    case_insensitive = bool(args.get("case_insensitive", False))
    max_results = int(args.get("max_results", 200))
    if not pattern:
        return {"ok": False, "error": "pattern is required"}
    try:
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return {"ok": False, "error": f"invalid pattern: {exc}"}
    try:
        root = Path(config.workspace_root).resolve()
        target = _resolve_workspace_path(config.workspace_root, rel_path)
        paths = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
        hits: List[str] = []
        for path in paths:
            if ".git" in path.parts:
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if not fnmatch.fnmatch(str(rel), glob_pattern) and not fnmatch.fnmatch(path.name, glob_pattern):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                if regex.search(line):
                    hits.append(f"{path}:{lineno}:{line}")
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break
        return {"ok": True, "output": _limit_text("\n".join(hits))}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_todo(args: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    action = str(args.get("action", "list")).lower()
    if action == "list":
        return {"ok": True, "todos": state.todos}
    if action == "clear":
        state.todos = []
        return {"ok": True, "todos": state.todos}
    if action == "add":
        item = {
            "id": str(args.get("id", f"todo-{len(state.todos) + 1}")),
            "content": str(args.get("content", "")),
            "status": str(args.get("status", "pending")),
        }
        state.todos.append(item)
        return {"ok": True, "todos": state.todos}
    if action == "update":
        todo_id = str(args.get("id", ""))
        for item in state.todos:
            if item["id"] == todo_id:
                if "content" in args:
                    item["content"] = str(args["content"])
                if "status" in args:
                    item["status"] = str(args["status"])
                return {"ok": True, "todos": state.todos}
        return {"ok": False, "error": f"Todo id not found: {todo_id}"}
    if action == "set":
        raw_items = args.get("items", [])
        if not isinstance(raw_items, list):
            return {"ok": False, "error": "items must be a list"}
        parsed: List[Dict[str, str]] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            parsed.append(
                {
                    "id": str(raw.get("id", f"todo-{len(parsed) + 1}")),
                    "content": str(raw.get("content", "")),
                    "status": str(raw.get("status", "pending")),
                }
            )
        state.todos = parsed
        return {"ok": True, "todos": state.todos}
    return {"ok": False, "error": f"Unsupported todo action: {action}"}


def tool_ask_user_question(args: Dict[str, Any]) -> Dict[str, Any]:
    question = str(args.get("question", "Please provide more detail:")).strip()
    answer = input(f"[Agent question] {question}\nYour answer: ").strip()
    return {"ok": True, "question": question, "answer": answer}


def tool_load_skill(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    try:
        path = _resolve_workspace_path(config.workspace_root, str(args.get("path", "")))
        content = path.read_text(encoding="utf-8")
        return {"ok": True, "path": str(path), "content": _limit_text(content)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_inspect_codebase(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    try:
        base = _resolve_workspace_path(config.workspace_root, str(args.get("path", ".")))
        max_entries = int(args.get("max_entries", 300))
        entries: List[str] = []
        for p in sorted(base.rglob("*")):
            if ".git" in p.parts:
                continue
            entries.append(str(p.relative_to(config.workspace_root)))
            if len(entries) >= max_entries:
                break
        return {"ok": True, "path": str(base), "entries": entries}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def execute_tool_call(tool_call: Dict[str, Any], config: ChatConfig, state: AgentState) -> Dict[str, Any]:
    name = str(tool_call.get("name", ""))
    args = tool_call.get("arguments", {})
    if not isinstance(args, dict):
        return {"ok": False, "error": "arguments must be an object"}

    if name == "read_file":
        return tool_read_file(args, config)
    if name == "write_file":
        return tool_write_file(args, config)
    if name == "search_replace":
        return tool_search_replace(args, config)
    if name == "grep":
        return tool_grep(args, config)
    if name == "todo":
        return tool_todo(args, state)
    if name == "ask_user_question":
        return tool_ask_user_question(args)
    if name == "load_skill":
        return tool_load_skill(args, config)
    if name == "inspect_codebase":
        return tool_inspect_codebase(args, config)
    return {"ok": False, "error": f"Unknown tool: {name}"}


_TRACE_REGION_SIZE = 100_000_000

# ``--mesh-device`` name → (rows, cols); matches ``text_demo.py``.

_MESH_SHAPES: Dict[str, Tuple[int, int]] = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "P150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
}


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _round_down(value: int, multiple: int) -> int:
    return (value // multiple) * multiple


def _longest_common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    """Compare two 1-D token-id tensors; return shared prefix length."""
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n == 0:
        return 0
    diff = (a[:n] != b[:n]).nonzero(as_tuple=False)
    if diff.numel() == 0:
        return n
    return int(diff[0].item())


def _prefill_start_for_kv_len(kv_seq_len: int, block_size: int) -> int:
    """``start_pos`` for the next incremental prefill (may rewind to block boundary)."""
    if kv_seq_len <= 0:
        return 0
    return _round_down(kv_seq_len, block_size)


@dataclass
class TTAgentConfig(ChatConfig):
    """Extends the base chat config with TT-runtime + tokenizer knobs."""

    mesh_device_name: str = "T3K"
    num_layers: Optional[int] = None
    max_seq_len: Optional[int] = None
    max_context_tokens: int = 512
    seed: Optional[int] = None
    prefill_activations_dram: bool = False
    use_kv_cache: bool = True
    trace_decode: bool = True
    trace_decode_2cq: bool = True
    verbose_runtime: bool = False


def _log_runtime_verbose(config: TTAgentConfig, msg: str, *args) -> None:
    """Log KV/prefill/decode progress during chat when ``--verbose`` is set."""
    if config.verbose_runtime:
        logger.info(msg, *args)


@dataclass
class TtAgentRuntime:
    mesh_device: ttnn.MeshDevice
    tokenizer: Any
    args: Devstral2Args
    model: TtMinistral3ForCausalLM
    tt_ccl: TT_CCL
    pad_token_id: int
    eos_ids: List[int]
    cfg: TTAgentConfig
    # Persistent decode input buffers (trace binds to these addresses).
    decode_tok_dev: Optional[ttnn.Tensor] = None
    decode_pos_dev: Optional[ttnn.Tensor] = None
    decode_trace_id: Optional[int] = None
    decode_trace_logits: Optional[ttnn.Tensor] = None
    # Persistent prefill chunk buffer (one ``kv_block_size`` frame).
    prefill_chunk_dev: Optional[ttnn.Tensor] = None
    # Incremental KV session state (device K/V tensors persist in the model).
    kv_seq_len: int = 0
    cached_token_ids: Optional[torch.Tensor] = None
    system_prefix_warmed: bool = False
    decode_2cq: Optional[DecodeTrace2CQ] = None
    # On-device argmax sampler (decode only, run post-trace). None = host sampling.
    sampler: Optional[OnDeviceSampler] = None


def _set_fabric_1d_or_warn(enabled: bool) -> bool:
    """Best-effort enable of the FABRIC_1D config used by the pytest demo. Returns True if set."""
    if not enabled:
        return False
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not set FabricConfig.FABRIC_1D ({exc}); continuing without fabric override.")
        return False


def _open_mesh(mesh_name: str, *, num_command_queues: int = 1) -> Tuple[ttnn.MeshDevice, bool]:
    shape = _MESH_SHAPES.get(mesh_name)
    if shape is None:
        raise ValueError(f"Unknown --mesh-device {mesh_name!r}. Supported: {sorted(_MESH_SHAPES.keys())}")
    if num_command_queues not in (1, 2):
        raise ValueError(f"num_command_queues must be 1 or 2, got {num_command_queues}")
    rows, cols = shape
    is_multichip = rows * cols > 1
    fabric_set = _set_fabric_1d_or_warn(enabled=is_multichip)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(rows, cols),
        l1_small_size=DEVSTRAL2_LARGE_L1_SMALL_SIZE,
        trace_region_size=_TRACE_REGION_SIZE,
        num_command_queues=num_command_queues,
    )
    return mesh_device, fabric_set


def _close_mesh(mesh_device: ttnn.MeshDevice, fabric_was_set: bool) -> None:
    try:
        ttnn.close_mesh_device(mesh_device)
    finally:
        if fabric_was_set:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:  # noqa: BLE001
                pass


def _resolve_eos_ids(tokenizer) -> List[int]:
    out: List[int] = []
    eid = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eid, int):
        out.append(eid)
    elif isinstance(eid, list):
        out.extend(int(x) for x in eid if isinstance(x, int))
    # Some Mistral tokenizers also use [/INST]-style stops; if defined and an int, include it.
    extra = getattr(tokenizer, "additional_eos_token_ids", None)
    if isinstance(extra, list):
        out.extend(int(x) for x in extra if isinstance(x, int))
    return sorted(set(out))


def _agent_system_content(config: TTAgentConfig) -> str:
    return f"{config.system_prompt}\n\n{DEFAULT_AGENT_RULES}".strip()


def _count_agent_system_tokens(tokenizer: Any, config: TTAgentConfig) -> int:
    """Token length of the fixed system message (dominates the agent prompt budget)."""
    text = _agent_system_content(config)
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "system", "content": text}],
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        return int(encoded["input_ids"].shape[-1])
    return int(tokenizer(text, return_tensors="pt")["input_ids"].shape[-1])


def _resolve_agent_max_seq_len(tokenizer: Any, config: TTAgentConfig, mesh_device: ttnn.MeshDevice) -> int:
    """Size KV + RoPE from system + max_context + max_new + margin (mirrors text_demo)."""
    _ = mesh_device  # BH-specific cap removed; prefill activations are routed through DRAM (mem_config).
    system_tokens = _count_agent_system_tokens(tokenizer, config)
    reserve = system_tokens + int(config.max_new_tokens) + 256
    need = reserve + int(config.max_context_tokens)
    if config.max_seq_len is not None:
        max_seq_len = int(config.max_seq_len)
    else:
        max_seq_len = max(512, _round_up(need, 512))
    if max_seq_len < need:
        logger.warning(f"--max-seq-len {max_seq_len} < need {need}; bumping to {_round_up(need, 512)}.")
        max_seq_len = _round_up(need, 512)

    user_budget = max_seq_len - reserve
    if config.max_context_tokens > user_budget:
        logger.warning(
            f"Clamping --max-context-tokens {config.max_context_tokens} -> {user_budget} "
            f"(system ~{system_tokens} tokens, max_seq_len={max_seq_len})."
        )
        config.max_context_tokens = max(32, user_budget)

    logger.info(
        f"Agent KV/RoPE budget: max_seq_len={max_seq_len} "
        f"(system ~{system_tokens}, user/turn cap {config.max_context_tokens})."
    )
    return max_seq_len


def load_tt_runtime(config: TTAgentConfig) -> TtAgentRuntime:
    """Open mesh, build TT model + tokenizer, allocate persistent decode buffers."""
    quiet_per_token_sampling_logs()
    if config.trace_decode_2cq and not config.trace_decode:
        raise ValueError("2CQ decode requires traced decode (do not pass --no-decode-trace).")

    logger.info(f"Loading tokenizer for {config.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None) or 0
    pad_token_id = int(pad_token_id if not isinstance(pad_token_id, list) else pad_token_id[0])

    text_cfg = require_text_config()
    full_layers = int(text_cfg.num_hidden_layers)
    num_layers = config.num_layers or full_layers
    if not (1 <= num_layers <= full_layers):
        raise ValueError(f"--num-layers must be in [1, {full_layers}], got {num_layers}")
    if num_layers != full_layers:
        logger.warning(f"Using --num-layers {num_layers} (full depth is {full_layers}); quality will be reduced.")

    num_command_queues = 2 if config.trace_decode_2cq else 1
    mesh_device, fabric_was_set = _open_mesh(config.mesh_device_name, num_command_queues=num_command_queues)
    try:
        max_seq_len = _resolve_agent_max_seq_len(tokenizer, config, mesh_device)
        args = Devstral2Args.from_hf_config(
            text_cfg,
            mesh_shape=tuple(mesh_device.shape),
            max_seq_len=max_seq_len,
            max_batch_size=1,
            prefill_activations_dram=config.prefill_activations_dram,
        )
        if config.prefill_activations_dram:
            logger.info("Prefill activations: DRAM (decode stays L1).")
        if config.trace_decode:
            logger.info(
                f"Decode trace enabled (trace_region_size={_TRACE_REGION_SIZE}); "
                "persistent decode_tok/pos buffers per step."
            )
        if config.trace_decode_2cq:
            logger.info("Decode trace 2CQ enabled (CQ1=input H2D, CQ0=forward/trace).")

        base_keys = model_prefill_weight_keys(num_layers)
        want_lm_head = not args.tie_word_embeddings
        try:
            weight_keys = base_keys + (["lm_head.weight"] if want_lm_head else [])
            state_dict = require_hf_weights(weight_keys)
        except Exception:
            if want_lm_head:
                logger.warning("lm_head.weight unavailable on the Hub; falling back to tied embeddings.")
                state_dict = require_hf_weights(base_keys)
            else:
                raise

        tt_ccl = TT_CCL(mesh_device)
        logger.info(f"Building TT model with {num_layers}/{full_layers} layers (max_seq_len={max_seq_len})...")
        model = TtMinistral3ForCausalLM(args, mesh_device, state_dict, tt_ccl, num_layers=num_layers)

        # Persistent decode-step buffers (host copy + device tensor each decode step).
        decode_tok_dev = _input_ids_to_tt(torch.zeros((1, 1), dtype=torch.long), mesh_device)
        decode_pos_dev = _current_pos_to_tt(torch.tensor([0], dtype=torch.long), mesh_device)
        decode_2cq = None
        if config.trace_decode_2cq:
            decode_2cq = DecodeTrace2CQ.create(mesh_device, decode_tok_dev, decode_pos_dev)
        block_size = int(args.kv_block_size)
        prefill_chunk_dev = _input_ids_to_tt(
            torch.full((1, block_size), int(pad_token_id), dtype=torch.long),
            mesh_device,
        )

        # On-device argmax sampling for decode (default on; post-trace, not folded in the trace).
        sampler: Optional[OnDeviceSampler] = None
        if on_device_sampling_enabled():
            if supports_on_device_sampling(args, mesh_device):
                sampler = build_sampler(args, mesh_device, tt_ccl, temperature=0.0)  # greedy → force-argmax
                logger.info("On-device argmax sampling for decode (post-trace; not folded into the decode trace).")
            else:
                logger.warning(
                    f"On-device sampling unsupported on mesh {tuple(mesh_device.shape)} "
                    f"(vocab {args.vocab_size} / {args.num_devices} devices); using host sampling."
                )

        runtime = TtAgentRuntime(
            mesh_device=mesh_device,
            tokenizer=tokenizer,
            args=args,
            model=model,
            tt_ccl=tt_ccl,
            pad_token_id=pad_token_id,
            eos_ids=_resolve_eos_ids(tokenizer),
            cfg=config,
            decode_tok_dev=decode_tok_dev,
            decode_pos_dev=decode_pos_dev,
            prefill_chunk_dev=prefill_chunk_dev,
            decode_2cq=decode_2cq,
            sampler=sampler,
        )
        # Prefill and decode compile on demand; TTNN caches programs after first use.
        return runtime
    except Exception:
        _close_mesh(mesh_device, fabric_was_set)
        raise


def _tokenize_messages(rt: TtAgentRuntime, messages: List[Dict[str, str]]) -> torch.LongTensor:
    return _tokenize_messages_impl(rt, messages, add_generation_prompt=True)


def _tokenize_messages_no_gen(rt: TtAgentRuntime, messages: List[Dict[str, str]]) -> torch.LongTensor:
    return _tokenize_messages_impl(rt, messages, add_generation_prompt=False)


def _tokenize_messages_impl(
    rt: TtAgentRuntime,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> torch.LongTensor:
    if getattr(rt.tokenizer, "chat_template", None):
        encoded = rt.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        ids = encoded["input_ids"]
    else:
        flat = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
        if add_generation_prompt:
            flat += "\nassistant:"
        ids = rt.tokenizer(flat, return_tensors="pt")["input_ids"]
    return ids[0].to(torch.long).unsqueeze(0)  # (1, T)


def _tokenize_system_prefix(rt: TtAgentRuntime, config: TTAgentConfig) -> torch.LongTensor:
    """Tokenize the fixed system + tool-rules message (no generation prompt suffix)."""
    messages = [{"role": "system", "content": _agent_system_content(config)}]
    if getattr(rt.tokenizer, "chat_template", None):
        encoded = rt.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        return encoded["input_ids"].to(torch.long)
    text = _agent_system_content(config)
    return rt.tokenizer(text, return_tensors="pt")["input_ids"].to(torch.long)


def _release_decode_trace(rt: TtAgentRuntime) -> None:
    """Release decode trace (required before prefill; Metal corrupts buffers if trace stays active)."""
    if rt.decode_trace_id is not None:
        ttnn.release_trace(rt.mesh_device, rt.decode_trace_id)
        rt.decode_trace_id = None
        rt.decode_trace_logits = None


def _reset_kv_session(rt: TtAgentRuntime) -> None:
    """Clear host-side KV session state (device cache is overwritten on next prefill)."""
    _release_decode_trace(rt)
    rt.kv_seq_len = 0
    rt.cached_token_ids = None
    rt.system_prefix_warmed = False


def _ensure_prefill_chunk_dev(rt: TtAgentRuntime, block_size: int, pad_id: int) -> None:
    if rt.prefill_chunk_dev is None:
        rt.prefill_chunk_dev = _input_ids_to_tt(
            torch.full((1, block_size), int(pad_id), dtype=torch.long),
            rt.mesh_device,
        )


def _resolve_incremental_prefill_start(
    rt: TtAgentRuntime,
    input_ids: torch.Tensor,
    prompt_len: int,
    block_size: int,
) -> int:
    """Return ``start_pos`` for chunked prefill; 0 means full prompt prefill."""
    if not rt.cfg.use_kv_cache or rt.kv_seq_len <= 0:
        return 0

    cached = rt.cached_token_ids
    if cached is None:
        return 0

    cache_len = int(cached.shape[0])
    if cache_len > prompt_len:
        logger.warning(f"Cached prompt ({cache_len} tokens) longer than new prompt ({prompt_len}); full re-prefill.")
        _reset_kv_session(rt)
        _warm_system_kv_prefix(rt, rt.cfg)
        return 0

    new_prefix = input_ids[0, :cache_len]
    if not torch.equal(cached, new_prefix):
        common = _longest_common_prefix_len(cached, new_prefix)
        logger.warning(
            f"KV prefix token mismatch at {common}/{cache_len}; "
            f"re-prefill from block boundary (device KV may diverge after decode)."
        )
        if common <= 0:
            _reset_kv_session(rt)
            _warm_system_kv_prefix(rt, rt.cfg)
            return 0
        rt.kv_seq_len = common
        rt.cached_token_ids = cached[:common].clone()
        return _prefill_start_for_kv_len(common, block_size)

    start = _prefill_start_for_kv_len(rt.kv_seq_len, block_size)
    if start >= prompt_len:
        logger.warning(f"Prompt length {prompt_len} <= cached KV start {start}; full re-prefill.")
        _reset_kv_session(rt)
        _warm_system_kv_prefix(rt, rt.cfg)
        return 0
    return start


def _warm_system_kv_prefix(rt: TtAgentRuntime, config: TTAgentConfig) -> None:
    """Prefill the static system + tool-rules prefix once per session."""
    if not config.use_kv_cache or rt.system_prefix_warmed:
        return

    system_ids = _tokenize_system_prefix(rt, config)
    system_len = int(system_ids.shape[1])
    block_size = int(rt.args.kv_block_size)
    pad_id = rt.pad_token_id
    num_chunks = max(1, (system_len + block_size - 1) // block_size)
    padded_len = num_chunks * block_size

    padded = torch.full((padded_len,), int(pad_id), dtype=torch.long)
    padded[:system_len] = system_ids[0]
    padded = padded.unsqueeze(0)

    _ensure_prefill_chunk_dev(rt, block_size, pad_id)
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * block_size
        chunk_tokens = padded[:, chunk_start : chunk_start + block_size]
        ttnn.copy_host_to_device_tensor(
            _input_ids_host(chunk_tokens, rt.mesh_device),
            rt.prefill_chunk_dev,
        )
        tt_out = rt.model(rt.prefill_chunk_dev, mode="prefill", start_pos=chunk_start)
        tt_out.deallocate(True)
    ttnn.synchronize_device(rt.mesh_device)

    rt.kv_seq_len = system_len
    rt.cached_token_ids = system_ids[0].clone()
    rt.system_prefix_warmed = True
    _log_runtime_verbose(
        config,
        f"KV session: prefilled static system prefix ({system_len} tokens, {num_chunks} chunk(s)).",
    )


def _sync_kv_cache_from_messages(rt: TtAgentRuntime, messages: List[Dict[str, str]]) -> None:
    """Update host cache metadata from the current chat (no generation-prompt suffix)."""
    if not rt.cfg.use_kv_cache:
        return
    prompt_ids = _tokenize_messages_no_gen(rt, messages)
    rt.cached_token_ids = prompt_ids[0].clone()
    rt.kv_seq_len = int(prompt_ids.shape[1])


def _chunked_prefill(
    rt: TtAgentRuntime,
    input_ids_padded: torch.Tensor,
    *,
    prompt_len: int,
    prefill_start: int,
    block_size: int,
    num_chunks: int,
    config: TTAgentConfig,
) -> int:
    """Run chunked prefill from ``prefill_start``; return first sampled token."""
    first_chunk = prefill_start // block_size
    next_token: Optional[int] = None
    for chunk_idx in range(first_chunk, num_chunks):
        chunk_start = chunk_idx * block_size
        chunk_tokens = input_ids_padded[:, chunk_start : chunk_start + block_size]
        ttnn.copy_host_to_device_tensor(
            _input_ids_host(chunk_tokens, rt.mesh_device),
            rt.prefill_chunk_dev,
        )
        tt_out = rt.model(rt.prefill_chunk_dev, mode="prefill", start_pos=chunk_start)
        if chunk_idx == num_chunks - 1:
            ttnn.synchronize_device(rt.mesh_device)
            local_pos = (prompt_len - 1) - chunk_start
            if rt.sampler is not None:
                next_token = sample_first_token_from_prefill_logits(rt.sampler, tt_out, local_pos=local_pos)
            else:
                logits_torch = _logits_to_torch(tt_out, rt.mesh_device, rt.args.vocab_size)
                next_token = _sample_next(
                    logits_torch[local_pos],
                    do_sample=bool(config.do_sample),
                    temperature=float(config.temperature),
                    top_p=float(config.top_p),
                )
        tt_out.deallocate(True)
    assert next_token is not None, "chunked prefill produced no logits"
    return next_token


def _sample_next(logits_row: torch.Tensor, do_sample: bool, temperature: float, top_p: float) -> int:
    """CPU-side sampling of a single token id from a logits row (1-D over vocab)."""
    row = logits_row.float().view(-1)
    if not do_sample or temperature <= 0:
        return int(row.argmax(dim=-1).item())
    probs = torch.softmax(row / max(float(temperature), 1e-6), dim=-1)
    if 0.0 < float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= float(top_p)
        keep[0] = True  # always keep top-1
        filtered = torch.zeros_like(probs)
        filtered[sorted_idx[keep]] = sorted_probs[keep]
        probs = filtered / filtered.sum()
    return int(torch.multinomial(probs, num_samples=1).item())


def _sample_decode_token(rt: TtAgentRuntime, logits_tt: ttnn.Tensor, config: TTAgentConfig) -> int:
    """Next decode token: on-device argmax (one int32 D2H, post-trace) when enabled, else host sample.

    ``logits_tt`` is the raw decode logits (logical 1 row); the on-device path pads to the sampling
    batch and runs ``SamplingGenerator`` as its own program — it is never folded into the decode trace.
    """
    if rt.sampler is not None:
        rt.sampler.sample_into_buffer(logits_tt)
        return rt.sampler.tt_read_token(0)
    logits_torch = _logits_to_torch(logits_tt, rt.mesh_device, rt.args.vocab_size)
    return _sample_next(
        logits_torch[0],
        do_sample=bool(config.do_sample),
        temperature=float(config.temperature),
        top_p=float(config.top_p),
    )


def _write_decode_inputs(rt: TtAgentRuntime, token_id: int, pos: int) -> None:
    """Refresh the persistent ``(token, current_pos)`` device buffers for one decode step."""
    stage_decode_inputs(
        rt.decode_2cq,
        rt.mesh_device,
        rt.decode_tok_dev,
        rt.decode_pos_dev,
        token_id,
        pos,
    )


def _decode_one_token(rt: TtAgentRuntime, token_id: int, pos: int, config: TTAgentConfig) -> int:
    """Single untraced decode step (host buffer refresh + forward + sample)."""
    _write_decode_inputs(rt, token_id, pos)
    tt_out = rt.model(rt.decode_tok_dev, mode="decode", current_pos=rt.decode_pos_dev)
    signal_decode_step_done(rt.decode_2cq)
    ttnn.synchronize_device(rt.mesh_device)
    next_token = _sample_decode_token(rt, tt_out, config)
    tt_out.deallocate(True)
    return next_token


def _run_traced_decode_loop(
    rt: TtAgentRuntime,
    *,
    next_token: int,
    current_pos: int,
    config: TTAgentConfig,
    max_extra_tokens: int,
) -> List[int]:
    """Traced decode: compile once, capture once, then ``execute_trace`` for remaining steps.

    Uses persistent ``decode_tok_dev`` / ``decode_pos_dev`` (host→device copy each step).
    When ``trace_decode_2cq`` is set, input H2D uses CQ1 and forward/trace uses CQ0.
    Trace is released before the next prefill in ``generate_assistant_text_tt``.
    """
    if rt.decode_2cq is not None:
        return _run_traced_decode_loop_2cq(
            rt,
            next_token=next_token,
            current_pos=current_pos,
            config=config,
            max_extra_tokens=max_extra_tokens,
        )

    extra: List[int] = []
    pos = current_pos
    tok = next_token

    for iteration in range(max_extra_tokens):
        if tok in rt.eos_ids:
            break

        _write_decode_inputs(rt, tok, pos)
        if iteration == 0:
            # Untraced compile pass (kernel cache only).
            tt_out = rt.model(rt.decode_tok_dev, mode="decode", current_pos=rt.decode_pos_dev)
            tt_out.deallocate(True)
            ttnn.synchronize_device(rt.mesh_device)

            rt.decode_trace_id = ttnn.begin_trace_capture(rt.mesh_device, cq_id=0)
            rt.decode_trace_logits = rt.model(rt.decode_tok_dev, mode="decode", current_pos=rt.decode_pos_dev)
            ttnn.end_trace_capture(rt.mesh_device, rt.decode_trace_id, cq_id=0)
            ttnn.synchronize_device(rt.mesh_device)
            if max_extra_tokens > 1:
                _log_runtime_verbose(
                    config,
                    "Decode trace captured (replay for subsequent tokens this generation).",
                )
        else:
            ttnn.execute_trace(rt.mesh_device, rt.decode_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(rt.mesh_device)

        # Sampling runs here, after the trace replay — never folded into the captured trace.
        tok = _sample_decode_token(rt, rt.decode_trace_logits, config)
        extra.append(tok)
        pos += 1

    return extra


def _run_traced_decode_loop_2cq(
    rt: TtAgentRuntime,
    *,
    next_token: int,
    current_pos: int,
    config: TTAgentConfig,
    max_extra_tokens: int,
) -> List[int]:
    """Traced decode with CQ1 input staging and CQ0 forward/trace (see ``DecodeTrace2CQ``)."""
    pipe = rt.decode_2cq
    assert pipe is not None

    extra: List[int] = []
    pos = current_pos
    tok = next_token

    for iteration in range(max_extra_tokens):
        if tok in rt.eos_ids:
            break

        stage_decode_inputs(pipe, rt.mesh_device, rt.decode_tok_dev, rt.decode_pos_dev, tok, pos)

        if iteration == 0:
            tt_out = rt.model(rt.decode_tok_dev, mode="decode", current_pos=rt.decode_pos_dev)
            tt_out.deallocate(True)
            ttnn.synchronize_device(rt.mesh_device)
            signal_decode_step_done(pipe)

            stage_decode_inputs(pipe, rt.mesh_device, rt.decode_tok_dev, rt.decode_pos_dev, tok, pos)

            rt.decode_trace_id = ttnn.begin_trace_capture(rt.mesh_device, cq_id=0)
            rt.decode_trace_logits = rt.model(rt.decode_tok_dev, mode="decode", current_pos=rt.decode_pos_dev)
            ttnn.end_trace_capture(rt.mesh_device, rt.decode_trace_id, cq_id=0)
            ttnn.synchronize_device(rt.mesh_device)
            signal_decode_step_done(pipe)
            if max_extra_tokens > 1:
                _log_runtime_verbose(
                    config,
                    "Decode trace captured with 2CQ (CQ1=H2D, CQ0=trace replay).",
                )
        else:
            ttnn.execute_trace(rt.mesh_device, rt.decode_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(rt.mesh_device)
            signal_decode_step_done(pipe)

        # Sampling runs here, after the trace replay — never folded into the captured trace.
        tok = _sample_decode_token(rt, rt.decode_trace_logits, config)
        extra.append(tok)
        pos += 1

    return extra


def generate_assistant_text_tt(
    rt: TtAgentRuntime,
    messages: List[Dict[str, str]],
    config: TTAgentConfig,
) -> str:
    """One turn: incremental chunked prefill + traced or untraced decode.

    When ``use_kv_cache`` is enabled, reuses device KV for the shared prefix (system
    prefix warmed at session start; conversation extended across tool sub-calls).
    Only token suffixes not already in ``cached_token_ids`` are prefilled.

    Decode trace uses persistent ``(token, current_pos)`` buffers each step. The trace is
    released before prefill (Metal requirement) and re-captured once per generation.
    """
    _release_decode_trace(rt)

    input_ids = _tokenize_messages(rt, messages)
    prompt_len = int(input_ids.shape[1])
    if prompt_len + config.max_new_tokens > int(rt.args.max_seq_len):
        raise RuntimeError(
            f"prompt({prompt_len}) + max_new_tokens({config.max_new_tokens}) exceeds "
            f"max_seq_len={rt.args.max_seq_len}. Use /clear or restart with larger --max-seq-len."
        )

    block_size = int(rt.args.kv_block_size)
    num_chunks = max(1, (prompt_len + block_size - 1) // block_size)
    padded_prompt_len = num_chunks * block_size
    pad_id = rt.pad_token_id
    input_ids_padded = torch.full((padded_prompt_len,), int(pad_id), dtype=torch.long)
    input_ids_padded[:prompt_len] = input_ids[0]
    input_ids_padded = input_ids_padded.unsqueeze(0)  # (1, padded_prompt_len)

    if config.seed is not None:
        torch.manual_seed(int(config.seed))

    _ensure_prefill_chunk_dev(rt, block_size, pad_id)
    prefill_start = _resolve_incremental_prefill_start(rt, input_ids, prompt_len, block_size)
    if prefill_start > 0:
        _log_runtime_verbose(
            config,
            f"Incremental prefill: reusing {prefill_start} cached tokens, "
            f"prefilling {prompt_len - prefill_start} new "
            f"({num_chunks - prefill_start // block_size} chunk(s)).",
        )
    elif rt.cfg.use_kv_cache and rt.system_prefix_warmed:
        _log_runtime_verbose(config, f"Full prompt prefill ({num_chunks} chunk(s)).")

    next_token = _chunked_prefill(
        rt,
        input_ids_padded,
        prompt_len=prompt_len,
        prefill_start=prefill_start,
        block_size=block_size,
        num_chunks=num_chunks,
        config=config,
    )

    generated: List[int] = [next_token]
    current_pos = prompt_len

    if next_token in rt.eos_ids or config.max_new_tokens <= 1:
        return rt.tokenizer.decode(generated, skip_special_tokens=True).strip()

    max_extra = config.max_new_tokens - 1
    if config.trace_decode:
        generated.extend(
            _run_traced_decode_loop(
                rt,
                next_token=next_token,
                current_pos=current_pos,
                config=config,
                max_extra_tokens=max_extra,
            )
        )
    else:
        for _ in range(max_extra):
            next_token = _decode_one_token(rt, next_token, current_pos, config)
            if next_token in rt.eos_ids:
                break
            generated.append(next_token)
            current_pos += 1

    return rt.tokenizer.decode(generated, skip_special_tokens=True).strip()


def _append_assistant_message(
    rt: TtAgentRuntime,
    messages: List[Dict[str, str]],
    content: str,
) -> None:
    """Append assistant turn and refresh KV cache metadata for the next incremental prefill."""
    messages.append({"role": "assistant", "content": content})
    _sync_kv_cache_from_messages(rt, messages)


def chat_loop_tt(rt: TtAgentRuntime, config: TTAgentConfig) -> None:
    system_content = _agent_system_content(config)
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    state = AgentState()

    print("\n--- Devstral2 Large Agent Demo (Tenstorrent, text-only) ---")
    print("Type 'quit' or 'exit' to stop. '/clear' resets chat + tools.\n")

    _warm_system_kv_prefix(rt, config)

    py_cfg = ChatConfig(
        model_id=config.model_id,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
        device="cpu",
        system_prompt=config.system_prompt,
        workspace_root=config.workspace_root,
        max_tool_calls_per_turn=config.max_tool_calls_per_turn,
    )

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_content}]
            state = AgentState()
            _reset_kv_session(rt)
            _warm_system_kv_prefix(rt, config)
            print("Conversation history and todo state cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})
        final_response: Optional[str] = None
        last_tool_result: Optional[Dict[str, Any]] = None
        last_tool_name: Optional[str] = None
        last_tool_signature: Optional[str] = None
        repeated_tool_calls = 0

        for _ in range(config.max_tool_calls_per_turn):
            try:
                assistant_text = generate_assistant_text_tt(rt, messages, config)
            except RuntimeError as exc:
                final_response = f"[TT generation error] {exc}"
                _append_assistant_message(rt, messages, final_response)
                break

            tool_call = parse_tool_call(assistant_text)

            if tool_call is None:
                if "<tool_call>" in assistant_text or any(f"{tool}{{" in assistant_text for tool in _KNOWN_TOOL_NAMES):
                    _log_unparsed_tool_output(assistant_text)
                    final_response = (
                        "I tried to run a tool but the tool-call JSON was incomplete. "
                        "Please try again with a shorter command or use /clear."
                    )
                else:
                    final_response = assistant_text
                _append_assistant_message(rt, messages, final_response)
                break

            _append_assistant_message(rt, messages, assistant_text)
            call_signature = json.dumps(tool_call, sort_keys=True, ensure_ascii=True)
            if call_signature == last_tool_signature:
                repeated_tool_calls += 1
            else:
                repeated_tool_calls = 0
            last_tool_signature = call_signature

            tool_result = execute_tool_call(tool_call, py_cfg, state)
            last_tool_result = tool_result
            last_tool_name = str(tool_call.get("name", ""))
            result_content = f"<tool_result>\n{json.dumps(tool_result, ensure_ascii=True)}\n</tool_result>"
            messages.append({"role": "user", "content": result_content})

            if repeated_tool_calls >= 1:
                if tool_result.get("ok", False):
                    output = str(tool_result.get("output", "")).strip()
                    final_response = output if output else f"Completed `{last_tool_name}` successfully."
                else:
                    err = str(tool_result.get("error", "")).strip()
                    out = str(tool_result.get("output", "")).strip()
                    details = err if err else out
                    final_response = f"`{last_tool_name}` failed." + (f" Details: {details}" if details else "")
                _append_assistant_message(rt, messages, final_response)
                break

        if final_response is None:
            if last_tool_result is not None:
                if last_tool_result.get("ok", False):
                    output = str(last_tool_result.get("output", "")).strip()
                    final_response = output if output else "Tool execution completed successfully."
                else:
                    err = str(last_tool_result.get("error", "")).strip()
                    out = str(last_tool_result.get("output", "")).strip()
                    details = err if err else out
                    final_response = "Tool execution did not complete cleanly." + (
                        f" Details: {details}" if details else ""
                    )
            else:
                final_response = (
                    "I reached the per-turn tool-call limit before producing a final answer. "
                    "Try narrowing the request."
                )
            _append_assistant_message(rt, messages, final_response)

        print(f"\nAssistant: {final_response}\n")


def parse_tt_args() -> TTAgentConfig:
    p = argparse.ArgumentParser(description="Interactive Devstral-2-123B agent on Tenstorrent (text-only).")
    p.add_argument("--model", default=DEVSTRAL2_LARGE_REPO_ID, help=f"HF model id (default {DEVSTRAL2_LARGE_REPO_ID})")

    # Agent (CLI parity with the small variant).
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens per model call (tool turns with long write_file content may need more)",
    )
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--no-sample", action="store_true", help="Greedy decoding (argmax)")
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--workspace-root", default=str(Path.cwd()))
    p.add_argument("--max-tool-calls-per-turn", type=int, default=6)

    # TT runtime
    p.add_argument(
        "--mesh-device",
        default=os.environ.get("MESH_DEVICE", "T3K"),
        choices=sorted(_MESH_SHAPES.keys()),
        help="Mesh shape preset (default T3K = 1x8).",
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Decoder layer count (default: full model depth). Lower values reduce quality but speed bring-up.",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=262144,
        metavar="S",
        help="KV/RoPE cap (default 262144; must be >= prompt + max-new, multiple of 128).",
    )
    p.add_argument(
        "--max-context-tokens",
        type=int,
        default=512,
        metavar="N",
        help="Max total prompt tokens per turn (includes the fixed system/tool rules message).",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--prefill-dram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Route prefill activations through DRAM (default: on). Use --no-prefill-dram for L1 prefill.",
    )
    p.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable incremental KV prefix caching (full prefill every call).",
    )
    p.add_argument(
        "--no-decode-trace",
        action="store_true",
        help="Disable traced decode (full model dispatch every token).",
    )
    p.add_argument(
        "--no-decode-trace-2cq",
        action="store_true",
        help="Disable 2CQ decode (single command queue; CQ0 does input H2D + trace).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log KV warm-up, incremental prefill, and decode-trace capture during chat (default: off).",
    )

    a = p.parse_args()

    return TTAgentConfig(
        model_id=a.model,
        max_new_tokens=a.max_new_tokens,
        temperature=a.temperature,
        top_p=a.top_p,
        do_sample=not a.no_sample,
        device="cpu",
        system_prompt=a.system_prompt,
        workspace_root=str(Path(a.workspace_root).resolve()),
        max_tool_calls_per_turn=a.max_tool_calls_per_turn,
        mesh_device_name=a.mesh_device,
        num_layers=a.num_layers,
        max_seq_len=a.max_seq_len,
        max_context_tokens=a.max_context_tokens,
        seed=a.seed,
        prefill_activations_dram=a.prefill_dram,
        use_kv_cache=not a.no_kv_cache,
        trace_decode=not a.no_decode_trace,
        trace_decode_2cq=decode_trace_2cq_enabled() and not a.no_decode_trace_2cq,
        verbose_runtime=a.verbose,
    )


def main() -> None:
    config = parse_tt_args()
    rt = load_tt_runtime(config)
    fabric_set = (rt.mesh_device.shape[0] * rt.mesh_device.shape[1]) > 1
    try:
        chat_loop_tt(rt, config)
    finally:
        try:
            if rt.decode_tok_dev is not None:
                rt.decode_tok_dev.deallocate(True)
        except Exception:  # noqa: BLE001
            pass
        try:
            if rt.decode_pos_dev is not None:
                rt.decode_pos_dev.deallocate(True)
        except Exception:  # noqa: BLE001
            pass
        try:
            if rt.prefill_chunk_dev is not None:
                rt.prefill_chunk_dev.deallocate(True)
        except Exception:  # noqa: BLE001
            pass
        try:
            if rt.sampler is not None:
                rt.sampler.deallocate()
        except Exception:  # noqa: BLE001
            pass
        _release_decode_trace(rt)
        _close_mesh(rt.mesh_device, fabric_set)


if __name__ == "__main__":
    main()
