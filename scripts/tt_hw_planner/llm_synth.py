from __future__ import annotations


from .discovery import safe_relative_to_root, BRINGUP_ROOT
import ast
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .bringup import REPO_ROOT
from .bringup_loop import (
    COMPONENT_SUBMODULE_HINTS,
    _emit_pcc_template,
    _run_pytest_target,
    _safe_id,
    find_demo_dir,
)
from .family_backends import DEFAULT_TEMPLATE_PYTEST_EXCLUDE_K


_TTNN_CHEATSHEET = """
TTNN op surface available for synthesis. Use ONLY these. If a feature you
need is missing, leave a `# TODO(ttnn-gap): ...` comment and skip it rather
than inventing an op.

# Tensor lifecycle
ttnn.from_torch(t: torch.Tensor, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None) -> ttnn.Tensor
ttnn.to_torch(t: ttnn.Tensor) -> torch.Tensor
ttnn.to_device(t, device)
ttnn.from_device(t)
ttnn.to_layout(t, layout)
ttnn.deallocate(t)

# Layouts and dtypes
ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32, ttnn.uint32, ttnn.int32

# Linear algebra
ttnn.matmul(a, b, *, transpose_a=False, transpose_b=False, dtype=None, memory_config=None) -> ttnn.Tensor
ttnn.linear(input, weight, *, bias=None, transpose_b=True, dtype=None) -> ttnn.Tensor
ttnn.add(a, b), ttnn.sub(a, b), ttnn.mul(a, b), ttnn.div(a, b)

# Activations
ttnn.relu(x), ttnn.gelu(x), ttnn.silu(x), ttnn.sigmoid(x), ttnn.tanh(x)
ttnn.softmax(x, dim=-1)

# Norms
ttnn.layer_norm(x, *, weight=None, bias=None, epsilon=1e-5) -> ttnn.Tensor
ttnn.rms_norm(x, *, weight=None, epsilon=1e-6) -> ttnn.Tensor
ttnn.group_norm(x, *, num_groups, weight=None, bias=None, epsilon=1e-5)

# Shape ops
ttnn.reshape(x, shape: tuple) -> ttnn.Tensor
ttnn.permute(x, dims: tuple) -> ttnn.Tensor
ttnn.transpose(x, dim0, dim1) -> ttnn.Tensor
ttnn.concat(tensors: list, dim) -> ttnn.Tensor
ttnn.split(x, sections, dim)
ttnn.unsqueeze(x, dim), ttnn.squeeze(x, dim)
ttnn.repeat(x, shape), ttnn.repeat_interleave(x, repeats, dim)
ttnn.tile(x, dims)

# Convs / pools (vision)
ttnn.conv2d(input, weight, *, bias=None, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, batch_size, input_height, input_width)
ttnn.max_pool2d(x, *, batch_size, input_h, input_w, channels, kernel_size, stride, padding=(0,0))
ttnn.avg_pool2d(x, *, batch_size, input_h, input_w, channels, kernel_size, stride, padding=(0,0))
ttnn.upsample(x, scale_factor)  # nearest only
ttnn.interpolate(x, size=None, scale_factor=None, mode="nearest")

# Fused transformer
ttnn.transformer.scaled_dot_product_attention(q, k, v, *, is_causal=False, attn_mask=None, scale=None) -> ttnn.Tensor
ttnn.transformer.attention_softmax(scores, *, scale=None, attention_mask=None)

# Convention: every TTNN op returns a new ttnn.Tensor on-device; chain them.
# Convention: weight tensors are loaded once in __init__ via ttnn.from_torch
#             from torch_module.state_dict(); do NOT re-upload per call.

# COMMON MISTAKES — ttnn.Tensor has a DIFFERENT API than torch.Tensor.
# Calling torch methods on a ttnn.Tensor raises AttributeError at runtime.
# These are typical hallucinations the agent makes after reading the torch
# reference; substitute the listed ttnn replacement instead.
ttnn_t.float()              # WRONG  -> ttnn.typecast(t, ttnn.float32)
                            #          OR ttnn.to_torch(t).float() if you need torch
ttnn_t.to(device)           # WRONG  -> ttnn.to_device(t, device)
ttnn_t.cpu()                # WRONG  -> ttnn.to_torch(t).cpu()
ttnn_t.numpy()              # WRONG  -> ttnn.to_torch(t).numpy()
ttnn_t.detach()             # WRONG  -> ttnn tensors are already detached
ttnn_t.contiguous()         # WRONG  -> ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
ttnn_t + 0.5                # WRONG  -> ttnn.add(t, 0.5)
ttnn_t * 2.0                # WRONG  -> ttnn.mul(t, 2.0)
ttnn_t / 2.0                # WRONG  -> ttnn.div(t, 2.0)
ttnn_t - other              # WRONG  -> ttnn.sub(t, other)
ttnn_t @ other              # WRONG  -> ttnn.matmul(t, other)
torch.cat([ttnn_t, ...])    # WRONG  -> ttnn.concat([t, ...], dim=...)
torch.matmul(ttnn_t, ...)   # WRONG  -> ttnn.matmul(t, ...)
torch.nn.functional.relu(ttnn_t)  # WRONG  -> ttnn.relu(t)
""".strip()


_PROJECT_CONVENTIONS = """
Project conventions for synthesized TTNN modules:

1. File MUST start with this exact SPDX header (no modifications):
       # SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
       #
       # SPDX-License-Identifier: Apache-2.0

2. Imports: only `import ttnn`, `import torch` (for weight prep only), and
   standard library. Do NOT import transformers in the synthesized body —
   weight loading happens via the torch_module passed into __init__.

3. Public surface MUST export TWO names that downstream code depends on:
     class <ComponentClassName>:
         def __init__(self, device, torch_module):  ...
         def __call__(self, x): ...            # may take *args, **kwargs
     def build(device, torch_module):
         return <ComponentClassName>(device, torch_module)
   And the original module-level function name MUST still exist as a thin
   wrapper that lazily constructs an instance using a default device. Use
   this template at the end of the file:
       _instance = None
       def <component_safe>(*args, **kwargs):
           global _instance
           if _instance is None:
               raise RuntimeError(
                   "Synthesized TTNN module requires `build(device, torch_module)`. "
                   "Call it from the PCC test's `_build_ttnn_port`."
               )
           return _instance(*args, **kwargs)

4. Weight loading: inside __init__, walk torch_module.state_dict() and
   convert each tensor you need via:
       self.w_<name> = ttnn.from_torch(state_dict["<key>"], dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=device)
   Cache shapes/scales as plain Python attributes when possible.

5. Forward: every intermediate is a ttnn.Tensor. Do NOT call .cpu(), .numpy(),
   or .item(). Do NOT roundtrip through torch in the hot path.

6. If a feature in the HF reference cannot be expressed with the cheatsheet
   ops, raise NotImplementedError("ttnn-gap: <what's missing>") in the body
   so the bring-up loop surfaces it as a real PCC failure, and approximate
   with the closest available op for the rest of the file. Do NOT emit a
   comment for this — raise, don't narrate.

7. NO COMMENTS in the body. Specifically:
       - Do NOT add docstrings for classes, methods, or modules.
       - Do NOT add inline `# ...` comments that narrate what the code does.
       - Do NOT add section banners or `# ===`-style separators.
   The only comments allowed in the file are the 3-line SPDX header from
   rule 1. The code must be self-explanatory; if a step needs explaining,
   pick a clearer variable / method name instead of a comment. If the
   bring-up loop wants context, it reads the prompt — not the source.

8. Output FORMAT: use the Write tool to write the file to the exact
   `WRITE THIS FILE:` path shown in the prompt header. Do NOT paste the
   file contents into chat; do NOT respond with the Python source as
   chat text. The bring-up loop reads files from disk (under the demo's
   `_synth_responses/` directory), not from your chat output. After you
   call Write, your chat message can be just a 1-line confirmation like
   "Wrote `_synth_responses/<component>.py`". No code fences, no prose.
""".strip()


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.2
    timeout_s: int = 180


class LLMError(RuntimeError):
    pass


def resolve_llm_config(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> LLMConfig:
    chosen = (provider or os.environ.get("TT_HW_PLANNER_LLM_PROVIDER") or "").strip().lower()
    if not chosen:
        if os.environ.get("OPENAI_API_KEY"):
            chosen = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            chosen = "anthropic"
        else:
            raise LLMError(
                "No LLM provider configured. Set one of:\n"
                "  OPENAI_API_KEY=...   (works with OpenAI, OpenRouter, vLLM, Ollama)\n"
                "  ANTHROPIC_API_KEY=...\n"
                "Or pass `--llm-provider openai|anthropic` explicitly."
            )

    if chosen == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        return LLMConfig(
            provider="openai",
            api_key=api_key,
            base_url=(endpoint or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
            model=model or os.environ.get("OPENAI_MODEL") or "gpt-4o",
        )

    if chosen == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        return LLMConfig(
            provider="anthropic",
            api_key=api_key,
            base_url=(endpoint or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/"),
            model=model or os.environ.get("ANTHROPIC_MODEL") or "claude-3-7-sonnet-latest",
        )

    raise LLMError(f"Unknown LLM provider {chosen!r}. Use 'openai' or 'anthropic'.")


def _http_post_json(url: str, headers: Dict[str, str], payload: dict, timeout_s: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise LLMError(f"HTTP {exc.code} from {url}: {exc.reason}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Network error contacting {url}: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Non-JSON response from {url}: {raw[:300]}") from exc


def call_llm(cfg: LLMConfig, *, system: str, user: str) -> str:
    if cfg.provider == "openai":
        url = f"{cfg.base_url}/chat/completions"
        headers = {"authorization": f"Bearer {cfg.api_key}"}
        payload = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = _http_post_json(url, headers, payload, cfg.timeout_s)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI response shape: {data!r}") from exc

    if cfg.provider == "anthropic":
        url = f"{cfg.base_url}/v1/messages"
        headers = {
            "x-api-key": cfg.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = _http_post_json(url, headers, payload, cfg.timeout_s)
        try:
            chunks = data["content"]
            return "".join(c.get("text", "") for c in chunks if c.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected Anthropic response shape: {data!r}") from exc

    raise LLMError(f"Unsupported provider {cfg.provider!r}")


def invoke_llm_cli_one_shot(
    prompt: str,
    *,
    agent_bin: str = "claude",
    model: str = "sonnet",
    timeout_s: int = 180,
) -> str:
    """Single-shot LLM invocation via a CLI binary (default: ``claude``).

    Uses ``--print --output-format text`` to avoid the stream-json
    complexity of the iterative agent. Returns the raw stdout.
    Raises :class:`RuntimeError` on missing binary, timeout, or non-zero
    exit. Used by :mod:`auto_onboard` and :mod:`meta_plan`.
    """
    cmd = [
        agent_bin,
        "--print",
        "--output-format",
        "text",
        "--model",
        model,
        prompt,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"agent binary not found: {agent_bin!r}. Install the Claude " f"CLI or pass --agent-bin <path>."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"LLM timed out after {timeout_s}s; consider a higher " f"--timeout-s or a smaller prompt"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"LLM call failed (exit {proc.returncode}): " f"stderr={proc.stderr[:500]!r}")
    return proc.stdout or ""


def extract_json_from_llm_output(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from LLM output.

    The prompt asks for JSON-only, but models sometimes wrap the result
    in markdown fences or prepend a brief explanation. This helper
    strips fences, finds the outermost ``{...}`` block, and tries
    :func:`json.loads` first then :func:`ast.literal_eval` (some
    models emit single-quoted Python dicts). Returns ``None`` if no
    valid JSON-like block can be parsed.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            return None


@dataclass
class HFContext:
    resolved_path: str
    class_name: str
    class_source: str
    param_shapes: List[Tuple[str, List[int], str]]
    buffer_shapes: List[Tuple[str, List[int], str]]
    forward_signature: str
    source_origin: str = "local"
    upstream_url: Optional[str] = None
    loader_error: Optional[str] = None


_HF_RAW_BASE = "https://raw.githubusercontent.com/huggingface/transformers"


def _http_get_text(url: str, timeout_s: int = 15) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError):
        return None


def _fetch_hf_config_model_type(model_id: str) -> Optional[str]:
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    text = _http_get_text(url, timeout_s=10)
    if not text:
        return None
    try:
        return json.loads(text).get("model_type")
    except (json.JSONDecodeError, AttributeError):
        return None


def _fetch_upstream_modeling_source(
    *,
    model_id: str,
    hint_model_type: Optional[str] = None,
    max_chars: int = 80_000,
    branch: str = "main",
) -> Optional[Tuple[str, str, str]]:
    candidates: List[str] = []

    cfg_mt = _fetch_hf_config_model_type(model_id)
    if cfg_mt:
        candidates.append(cfg_mt)

    if hint_model_type and hint_model_type not in candidates:
        candidates.append(hint_model_type)

    for mt in list(candidates):
        for suffix in ("_video", "_audio", "_bnb", "_4bit", "_8bit"):
            if mt.endswith(suffix):
                stripped = mt[: -len(suffix)]
                if stripped and stripped not in candidates:
                    candidates.append(stripped)

    for mt in candidates:
        url = f"{_HF_RAW_BASE}/{branch}/src/transformers/models/{mt}/modeling_{mt}.py"
        text = _http_get_text(url)
        if text and "class " in text:
            if len(text) > max_chars:
                head = text[: max_chars // 2]
                tail = text[-max_chars // 2 :]
                text = f"{head}\n# ... [{len(text) - max_chars} chars elided] ...\n{tail}"
            return (mt, url, text)
    return None


_LOADER_MT_RE = re.compile(r"model[_ ]type[`'\"]?\s+[`'\"](\w+)[`'\"]")


def _extract_model_type_from_error(message: str) -> Optional[str]:
    m = _LOADER_MT_RE.search(message)
    return m.group(1) if m else None


def extract_hf_context(
    *,
    model_id: str,
    component_name: str,
    candidate_paths: List[str],
    max_source_chars: int = 12000,
    fetch_upstream: bool = True,
) -> HFContext:
    import inspect

    try:
        import transformers
    except ImportError as exc:
        raise LLMError(
            "transformers is not importable from the active env. " "Run `source python_env/bin/activate` first."
        ) from exc

    try:
        model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
    except (ValueError, KeyError, OSError, ImportError) as exc:
        installed = getattr(transformers, "__version__", "unknown")
        if not fetch_upstream:
            raise LLMError(
                f"transformers (v{installed}) could not load `{model_id}`: {exc}\n"
                f"--no-fetch-upstream was set, so no fallback was attempted."
            ) from exc

        hint_mt = _extract_model_type_from_error(str(exc))
        fetched = _fetch_upstream_modeling_source(model_id=model_id, hint_model_type=hint_mt)
        if fetched is None:
            raise LLMError(
                f"transformers (v{installed}) could not load `{model_id}`: {exc}\n"
                f"Upstream-source fallback also failed (no modeling file "
                f"found on huggingface/transformers main). The model may "
                f"live in a custom repo; consider `pip install --upgrade "
                f"transformers` or fetching the source manually."
            ) from exc

        mt_used, url_used, src_text = fetched
        if len(src_text) > max_source_chars:
            head = src_text[: max_source_chars // 2]
            tail = src_text[-max_source_chars // 2 :]
            src_text = f"{head}\n# ... [{len(src_text) - max_source_chars} chars elided] ...\n{tail}"

        return HFContext(
            resolved_path="(local load failed; upstream source fetched)",
            class_name=f"upstream::models/{mt_used}/modeling_{mt_used}.py",
            class_source=src_text,
            param_shapes=[],
            buffer_shapes=[],
            forward_signature="(local load failed; see candidate paths to locate the right class in the upstream file)",
            source_origin="upstream-github",
            upstream_url=url_used,
            loader_error=str(exc),
        )

    resolved = None
    resolved_path = None
    for path in candidate_paths:
        try:
            resolved = _resolve(model, path)
            resolved_path = path
            break
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
    if resolved is None:
        raise LLMError(
            f"Could not resolve any candidate path for `{component_name}` " f"of {model_id}. Tried: {candidate_paths}"
        )

    cls = type(resolved)
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        source = f"# inspect.getsource failed for {cls.__module__}.{cls.__name__}"

    if len(source) > max_source_chars:
        head = source[: max_source_chars // 2]
        tail = source[-max_source_chars // 2 :]
        source = f"{head}\n# ... [{len(source) - max_source_chars} chars elided] ...\n{tail}"

    params: List[Tuple[str, List[int], str]] = []
    for name, p in resolved.named_parameters():
        params.append((name, list(p.shape), str(p.dtype)))
    bufs: List[Tuple[str, List[int], str]] = []
    for name, b in resolved.named_buffers():
        bufs.append((name, list(b.shape), str(b.dtype)))

    fwd_sig = ""
    try:
        fwd_sig = f"forward{inspect.signature(cls.forward)}"
    except (ValueError, TypeError):
        fwd_sig = "forward(self, *args, **kwargs)"

    return HFContext(
        resolved_path=resolved_path,
        class_name=f"{cls.__module__}.{cls.__name__}",
        class_source=source,
        param_shapes=params,
        buffer_shapes=bufs,
        forward_signature=fwd_sig,
    )


def _resolve(obj, dotted: str):
    """Backwards-compat shim. See :func:`module_tree.resolve_dotted`."""
    from .module_tree import resolve_dotted

    return resolve_dotted(obj, dotted)


def _read_sibling_example(repo_root: Path, sibling_hint: Optional[str]) -> Optional[str]:
    if not sibling_hint:
        return None
    path = repo_root / sibling_hint
    if not path.is_file():
        return None
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError):
        return None
    if len(text) > 6000:
        text = text[:6000] + "\n# ... [sibling example truncated] ...\n"
    return text


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"[^A-Za-z0-9]+", name) if part)


def build_prompts(
    *,
    component_name: str,
    component_safe: str,
    model_id: str,
    hf_ctx: HFContext,
    new_shape: dict,
    sibling_example: Optional[str],
    previous_attempt: Optional[str] = None,
    previous_failure: Optional[str] = None,
    manifest_path: Optional[Path] = None,
    opplan_path: Optional[Path] = None,
) -> Tuple[str, str]:
    system = (
        "You are an expert TTNN engineer porting PyTorch modules from "
        "HuggingFace transformers to the Tenstorrent TTNN runtime. You "
        "produce Python code that runs on Tenstorrent Wormhole/Blackhole "
        "hardware. You follow project conventions exactly. You never "
        "invent ops that are not in the provided cheatsheet."
    )

    class_camel = _camel(component_safe) or "SynthesizedModule"

    if hf_ctx.source_origin == "upstream-github":
        param_table = (
            "    (local transformers loader failed; parameter shapes are not\n"
            "     available. Infer them from the HF source below, or load the\n"
            "     `state_dict()` of the corresponding submodule yourself.)"
        )
        buf_table = "    (same: not available without a working local loader)"
    else:
        param_table = (
            "\n".join(f"    {name}  shape={shape}  dtype={dt}" for name, shape, dt in hf_ctx.param_shapes)
            or "    (no learnable parameters)"
        )
        buf_table = (
            "\n".join(f"    {name}  shape={shape}  dtype={dt}" for name, shape, dt in hf_ctx.buffer_shapes)
            or "    (no buffers)"
        )

    origin_block = ""
    if hf_ctx.source_origin == "upstream-github":
        origin_block = (
            f"\n\nNOTE ON HF SOURCE PROVENANCE\n"
            f"----------------------------\n"
            f"The local transformers loader failed for this model:\n"
            f"  {hf_ctx.loader_error}\n"
            f"The HF source below was instead fetched directly from the\n"
            f"upstream `huggingface/transformers` repo at:\n"
            f"  {hf_ctx.upstream_url}\n"
            f"It contains the entire `modeling_<arch>.py` file (possibly\n"
            f"truncated). Locate the specific `nn.Module` for `{component_name}`\n"
            f"by matching the candidate access paths against the file's class\n"
            f"definitions, then port only that class (plus any helper classes\n"
            f"it uses).\n"
        )

    sibling_block = (
        f"\n\nSIBLING TTNN EXAMPLE (style reference from a related demo)\n"
        f"-----------------------------------------------------------\n"
        f"{sibling_example}\n"
        if sibling_example
        else ""
    )

    # Run the ttnn constraint catalog against this component's captured
    # input metadata + opplan. If any ttnn shape/dtype constraint will
    # bite, surface the matching recipe so the LLM can apply it upfront
    # instead of rediscovering the workaround across iterations.
    constraint_block = ""
    try:
        from .constraints import check_component, format_constraint_hints

        violations = check_component(
            component_name=component_name,
            hf_class_name=hf_ctx.class_name,
            manifest_path=manifest_path,
            opplan_path=opplan_path,
        )
        constraint_block = format_constraint_hints(violations)
    except Exception:
        # Constraint plumbing is advisory — never block prompt construction.
        constraint_block = ""

    retry_block = ""
    if previous_attempt and previous_failure:
        retry_block = (
            f"\n\nPREVIOUS ATTEMPT FAILED PCC TEST\n"
            f"--------------------------------\n"
            f"Your previous response (the .py file) was:\n"
            f"```\n{previous_attempt[:6000]}\n```\n"
            f"\nThe PCC test produced this failure tail:\n"
            f"```\n{previous_failure[:2000]}\n```\n"
            f"\nFix the issue and produce a corrected complete file. Pay "
            f"special attention to: tensor shapes (PCC failures often mean "
            f"a reshape/permute is wrong), weight key mappings (compare your "
            f"state_dict keys to the parameter table above), and the order "
            f"of operations in the HF forward()."
        )

    user = textwrap.dedent(
        f"""
        TASK
        ====
        Port the PyTorch class below to TTNN. Produce the complete contents
        of `_stubs/{component_safe}.py` and nothing else.

        TARGET
        ------
        model_id           : {model_id}
        component name     : {component_name}
        class to expose    : {class_camel}
        module-level fn    : {component_safe}
        HF class           : {hf_ctx.class_name}
        HF submodule path  : {hf_ctx.resolved_path}
        forward signature  : {hf_ctx.forward_signature}
        new-model shape    : {json.dumps(new_shape or {{}})}
        {origin_block}
        HF PyTorch SOURCE (this is the reference you are porting)
        ---------------------------------------------------------
        {hf_ctx.class_source}

        PARAMETER TABLE (HF state_dict keys + shapes; use these to load weights)
        ----------------------------------------------------------------------
        {param_table}

        BUFFER TABLE (non-trainable; usually causal masks / positional embeddings)
        ------------------------------------------------------------------------
        {buf_table}

        TTNN CHEATSHEET (the only ops you may use)
        ------------------------------------------
        {_TTNN_CHEATSHEET}

        PROJECT CONVENTIONS (mandatory)
        -------------------------------
        {_PROJECT_CONVENTIONS}
        {sibling_block}{constraint_block}{retry_block}

        OUTPUT
        ======
        Use the Write tool to write the complete file contents to the
        `WRITE THIS FILE:` path shown in the section header above. The
        file content must start with the 3-line SPDX header and contain
        only Python source (no markdown fences). Do NOT paste the file
        contents into chat; the bring-up loop reads files from disk.
        """
    ).strip()

    return system, user


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip() + "\n"


def _looks_like_python(body: str) -> bool:
    try:
        compile(body, "<synth>", "exec")
        return True
    except SyntaxError:
        return False


def _detect_call_signature_collisions(body: str) -> List[str]:
    """Detect `__call__` (or `forward`) method signatures that risk
    `TypeError: got multiple values for argument X` at invocation.

    Common pattern from v12: agent writes
        def __call__(self, hidden_states, *args, **kwargs):
    The test harness invokes `module(*pos, hidden_states=val)` where `pos`
    already contains a positional value at `hidden_states`'s index. Python
    raises TypeError because the same argument was passed twice.

    Risk = signature has `*args` AND a named parameter (other than self)
    whose name matches common tensor-input arg names that the test scaffold
    or HF reference might also pass as kwarg. Returns one message per
    offending method.
    """
    import ast

    try:
        tree = ast.parse(body)
    except SyntaxError:
        return []
    common_input_names = {
        "x",
        "hidden_states",
        "features",
        "input",
        "inputs",
        "image_embeddings",
        "image_positional_embeddings",
        "pixel_values",
    }
    issues: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in ("__call__", "forward"):
            continue
        sig = node.args
        if sig.vararg is None:
            continue
        named = [a.arg for a in sig.args[1:]]
        risky = [n for n in named if n in common_input_names]
        if risky:
            issues.append(
                f"`{node.name}({', '.join(named)}, *args, ...)` -- "
                f"`{risky[0]}` will collide if the test passes it both "
                f"positionally (in *args) and as keyword."
            )
    return issues


def _detect_self_inheriting_classes(body: str) -> List[str]:
    """Return class names that inherit from `_stub_mod.<same_name>` (or any
    submodule attribute matching the class's own name). This pattern is a
    runtime recursion trap: the agent reuses `_stub_mod.X` as a base for the
    new `class X`, but once the new stub is written, `_stub_mod` resolves
    to itself and the class becomes its own ancestor. v11 saw
    `class EncoderStack(_stub_mod.EncoderStack)` cause RecursionError on
    every PCC test call.
    """
    import ast

    try:
        tree = ast.parse(body)
    except SyntaxError:
        return []
    bad: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == node.name:
                bad.append(node.name)
                break
    return bad


@dataclass
class SynthAttempt:
    attempt: int
    body: str
    test_status: str
    test_summary: str
    response_chars: int


@dataclass
class SynthResult:
    component: str
    stub_path: str
    test_path: str
    attempts: List[SynthAttempt] = field(default_factory=list)
    final_status: str = "not-run"
    backup_path: Optional[str] = None
    audit_log: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def _audit_log_dir(demo_dir: Path) -> Path:
    d = demo_dir / "_synth_log"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_audit(
    *,
    demo_dir: Path,
    component_safe: str,
    cfg: LLMConfig,
    attempts: List[Tuple[str, str, str, str]],
) -> Path:
    stamp = _utc_stamp()
    path = _audit_log_dir(demo_dir) / f"{component_safe}__{stamp}.md"
    lines = [
        f"# Synthesis audit log",
        f"",
        f"- component: `{component_safe}`",
        f"- provider:  `{cfg.provider}`",
        f"- model:     `{cfg.model}`",
        f"- endpoint:  `{cfg.base_url}`",
        f"- timestamp: `{stamp}`",
        f"- attempts:  {len(attempts)}",
        f"",
    ]
    for i, (system, user, response, summary) in enumerate(attempts, start=1):
        lines.append(f"## Attempt {i}")
        lines.append("")
        lines.append("### System prompt")
        lines.append("```")
        lines.append(system)
        lines.append("```")
        lines.append("")
        lines.append("### User prompt (truncated to 8 KiB)")
        lines.append("```")
        lines.append(user[:8192])
        lines.append("```")
        lines.append("")
        lines.append("### Response (truncated to 16 KiB)")
        lines.append("```python")
        lines.append(response[:16384])
        lines.append("```")
        lines.append("")
        lines.append("### Test summary")
        lines.append("```")
        lines.append(summary or "(not run)")
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines))
    return path


_MACHINE_BANNER = (
    "# >>> MACHINE-GENERATED by `tt_hw_planner bringup --synthesize` <<<\n"
    "# Review required before trusting on hardware. This body was produced\n"
    "# by an LLM from the HF reference source and may contain subtle bugs.\n"
    "# Backup of the previous stub is at <stub>.before_synth.py.bak.\n"
)


def _wrap_for_banner(body: str) -> str:
    lines = body.splitlines(keepends=True)
    insert_at = 0
    if (
        len(lines) >= 3
        and lines[0].startswith("# SPDX-FileCopyrightText")
        and "SPDX-License-Identifier" in (lines[2] if len(lines) > 2 else "")
    ):
        insert_at = 3
    return "".join(lines[:insert_at]) + _MACHINE_BANNER + "".join(lines[insert_at:])


def synthesize_component(
    *,
    model_id: str,
    component_name: str,
    cfg: LLMConfig,
    repo_root: Path = REPO_ROOT,
    run_tests: bool = False,
    max_retries: int = 2,
    dry_run: bool = False,
    exclude_k: str = DEFAULT_TEMPLATE_PYTEST_EXCLUDE_K,
    fetch_upstream: bool = True,
) -> SynthResult:
    if repo_root == REPO_ROOT:
        from .discovery import BRINGUP_ROOT as _BRINGUP_ROOT

        repo_root = _BRINGUP_ROOT()
    demo_dir = find_demo_dir(model_id, repo_root=repo_root)
    if demo_dir is None:
        raise LLMError(
            f"No scaffolded demo folder found for {model_id!r}. " f"Run `scaffold {model_id} --apply` first."
        )

    status_path = demo_dir / "bringup_status.json"
    data = json.loads(status_path.read_text())
    comp = next(
        (c for c in data.get("components", []) if c.get("name") == component_name),
        None,
    )
    if comp is None:
        raise LLMError(
            f"Component {component_name!r} not found in bringup_status.json "
            f"for {model_id}. Available: "
            f"{[c['name'] for c in data.get('components', [])]}"
        )
    if comp.get("status") != "NEW":
        raise LLMError(
            f"Component {component_name!r} has status {comp.get('status')!r}, "
            f"not NEW. Synthesis only runs on NEW components."
        )

    safe = _safe_id(component_name)
    stub_path = demo_dir / "_stubs" / f"{safe}.py"
    manifest_path = demo_dir / "_captured" / safe / "manifest.json"
    opplan_path = stub_path.with_suffix(".opplan.json")
    new_shape = comp.get("new_shape") or {}
    sibling_hint = comp.get("sibling_tt_file")
    candidate_paths = COMPONENT_SUBMODULE_HINTS.get(component_name, [])

    hf_ctx = extract_hf_context(
        model_id=model_id,
        component_name=component_name,
        candidate_paths=candidate_paths,
        fetch_upstream=fetch_upstream,
    )
    sibling_example = _read_sibling_example(repo_root, sibling_hint)

    test_path, _gen, _existed = _emit_pcc_template(
        demo_dir=demo_dir,
        component_name=component_name,
        model_id=model_id,
        hf_reference=comp.get("hf_reference") or "",
        new_shape=new_shape,
        repo_root=repo_root,
    )

    result = SynthResult(
        component=component_name,
        stub_path=str(safe_relative_to_root(stub_path)),
        test_path=str(safe_relative_to_root(test_path)),
    )

    if stub_path.exists():
        backup_path = stub_path.with_suffix(".py.bak")
        if not backup_path.exists():
            shutil.copy2(stub_path, backup_path)
        result.backup_path = str(safe_relative_to_root(backup_path))

    if dry_run:
        system, user = build_prompts(
            component_name=component_name,
            component_safe=safe,
            model_id=model_id,
            hf_ctx=hf_ctx,
            new_shape=new_shape,
            sibling_example=sibling_example,
            manifest_path=manifest_path,
            opplan_path=opplan_path,
        )
        result.notes.append("dry-run: prompts assembled, no LLM call made")
        audit_path = _write_audit(
            demo_dir=demo_dir,
            component_safe=safe,
            cfg=cfg,
            attempts=[(system, user, "(dry-run; LLM not called)", "(dry-run)")],
        )
        result.audit_log = str(safe_relative_to_root(audit_path))
        result.final_status = "dry-run"
        return result

    audit_records: List[Tuple[str, str, str, str]] = []
    previous_attempt: Optional[str] = None
    previous_failure: Optional[str] = None

    total_calls = max_retries + 1
    for attempt_idx in range(1, total_calls + 1):
        system, user = build_prompts(
            component_name=component_name,
            component_safe=safe,
            model_id=model_id,
            hf_ctx=hf_ctx,
            new_shape=new_shape,
            sibling_example=sibling_example,
            previous_attempt=previous_attempt,
            previous_failure=previous_failure,
            manifest_path=manifest_path,
            opplan_path=opplan_path,
        )
        response = call_llm(cfg, system=system, user=user)
        body = _strip_fences(response)

        attempt = SynthAttempt(
            attempt=attempt_idx,
            body=body,
            test_status="not-run",
            test_summary="",
            response_chars=len(response),
        )

        if not _looks_like_python(body):
            attempt.test_status = "syntax-error"
            attempt.test_summary = "LLM response did not parse as Python."
            audit_records.append((system, user, response, attempt.test_summary))
            result.attempts.append(attempt)
            previous_attempt = body
            previous_failure = "Your previous response did not parse as Python. Re-emit a valid file."
            if attempt_idx < total_calls:
                continue
            break

        stub_path.parent.mkdir(parents=True, exist_ok=True)
        stub_path.write_text(_wrap_for_banner(body))

        if not run_tests:
            attempt.test_status = "not-run"
            attempt.test_summary = "(--run-tests not set; skipping PCC validation)"
            audit_records.append((system, user, response, attempt.test_summary))
            result.attempts.append(attempt)
            break

        status, summary = _run_pytest_target(test_path, exclude_k=exclude_k)
        attempt.test_status = status
        attempt.test_summary = summary
        audit_records.append((system, user, response, summary))
        result.attempts.append(attempt)

        if status == "passed":
            break

        previous_attempt = body
        previous_failure = summary

    audit_path = _write_audit(
        demo_dir=demo_dir,
        component_safe=safe,
        cfg=cfg,
        attempts=audit_records,
    )
    result.audit_log = str(safe_relative_to_root(audit_path))
    result.final_status = result.attempts[-1].test_status if result.attempts else "no-call"

    if result.final_status == "passed":
        result.notes.append("PCC test passed; stub will be auto-removed on next `bringup --run-tests` pass.")
    elif result.final_status == "not-run":
        result.notes.append("Stub was written; run `bringup --run-tests --component <name>` to validate.")
    elif result.final_status == "failed":
        result.notes.append(
            f"All {len(result.attempts)} synthesis attempts failed PCC. The "
            f"last attempt is on disk; original stub is at {result.backup_path}."
        )
    elif result.final_status == "syntax-error":
        result.notes.append("LLM produced non-parseable output every attempt. " "Try a stronger model via --llm-model.")

    return result


@dataclass
class SynthTargets:
    model_id: str
    demo_dir: str
    reuse: List[str] = field(default_factory=list)
    adapt: List[str] = field(default_factory=list)
    new: List[str] = field(default_factory=list)
    refused: List[Tuple[str, str]] = field(default_factory=list)


def list_synth_targets(
    *,
    model_id: str,
    only: Optional[List[str]] = None,
    repo_root: Path = REPO_ROOT,
) -> SynthTargets:
    if repo_root == REPO_ROOT:
        from .discovery import BRINGUP_ROOT as _BRINGUP_ROOT

        repo_root = _BRINGUP_ROOT()
    demo_dir = find_demo_dir(model_id, repo_root=repo_root)
    if demo_dir is None:
        raise LLMError(
            f"No scaffolded demo folder found for {model_id!r}. "
            f"Run `python -m scripts.tt_hw_planner scaffold {model_id} --apply` "
            f"first — synthesis only operates on the bring-up plan that "
            f"scaffold+bringup produced."
        )
    status_path = demo_dir / "bringup_status.json"
    if not status_path.exists():
        raise LLMError(
            f"bringup_status.json missing at {status_path}. The demo folder "
            f"exists but the bring-up plan was not generated. Re-run "
            f"`scaffold {model_id} --apply` to refresh."
        )
    data = json.loads(status_path.read_text())
    components = data.get("components", [])

    targets = SynthTargets(model_id=model_id, demo_dir=str(safe_relative_to_root(demo_dir)))
    requested = set(only) if only else None

    for comp in components:
        name = comp.get("name", "")
        status = comp.get("status")
        bucket = {"REUSE": targets.reuse, "ADAPT": targets.adapt, "NEW": targets.new}.get(status)
        if bucket is not None:
            bucket.append(name)
        if requested and name in requested and status != "NEW":
            targets.refused.append((name, f"status is {status!r}, not NEW — synthesis is gated on NEW only"))

    if requested:
        targets.new = [n for n in targets.new if n in requested]
        unknown = requested - set(c["name"] for c in components)
        for u in sorted(unknown):
            targets.refused.append((u, "not present in bringup_status.json — re-run `scaffold --apply`"))

    return targets


def render_synth_targets(t: SynthTargets) -> str:
    lines = [
        f"Bring-up plan for {t.model_id}",
        f"  demo dir: {t.demo_dir}",
        f"  REUSE  ({len(t.reuse):2d}): {', '.join(t.reuse) if t.reuse else '(none)'}",
        f"  ADAPT  ({len(t.adapt):2d}): {', '.join(t.adapt) if t.adapt else '(none)'}",
        f"  NEW    ({len(t.new):2d}): {', '.join(t.new) if t.new else '(none)'}",
        "",
        "LLM synthesis is gated on NEW only. REUSE and ADAPT components are",
        "NEVER touched by the LLM — they reuse / adapt existing tt-metal code.",
    ]
    if t.refused:
        lines.append("")
        lines.append("Refused (will NOT be synthesized):")
        for name, reason in t.refused:
            lines.append(f"  - {name}: {reason}")
    if not t.new:
        lines.append("")
        lines.append("Nothing for the LLM to synthesize — bring-up plan reports no NEW gaps.")
    return "\n".join(lines)


_BYO_PROMPTS_DIRNAME = "_synth_prompts"
_BYO_RESPONSES_DIRNAME = "_synth_responses"

_ADAPT_RESPONSE_SUFFIX = ".adapt.py"
_DEMO_RESPONSE_NAME = "demo.py"


def _byo_prompt_dir(demo_dir: Path) -> Path:
    d = demo_dir / _BYO_PROMPTS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _adapt_target_path_in_demo(
    *,
    sibling_tt_path: str,
    sibling_demo_dir: str,
    new_demo_dir: Path,
) -> Optional[Path]:
    sibling_demo_norm = sibling_demo_dir.rstrip("/")
    if not sibling_tt_path.startswith(sibling_demo_norm + "/"):
        return None
    rel = sibling_tt_path[len(sibling_demo_norm) + 1 :]
    sibling_base = Path(sibling_demo_norm).name
    new_base = new_demo_dir.name
    rel_renamed = rel.replace(sibling_base, new_base)
    return new_demo_dir / rel_renamed


def _collect_adapt_targets(
    *,
    demo_dir: Path,
    status_data: dict,
) -> List[Tuple[str, Path, Path, dict, dict]]:
    sibling_demo = status_data.get("backend", {}).get("demo_path", "")
    out: List[Tuple[str, Path, Path, dict, dict]] = []
    for comp in status_data.get("components", []):
        if comp.get("status") != "ADAPT":
            continue
        sibling_tt = comp.get("tt_reuse_target") or ""
        if not sibling_tt or not sibling_demo:
            continue
        new_tt = _adapt_target_path_in_demo(
            sibling_tt_path=sibling_tt,
            sibling_demo_dir=sibling_demo,
            new_demo_dir=demo_dir,
        )
        if new_tt is None or not new_tt.exists():
            continue
        out.append(
            (
                comp["name"],
                BRINGUP_ROOT() / sibling_tt,
                new_tt,
                comp.get("sibling_shape") or {},
                comp.get("new_shape") or {},
            )
        )
    return out


def _format_prompt_doc(
    *,
    component_name: str,
    component_safe: str,
    model_id: str,
    system: str,
    user: str,
    stub_rel: str,
) -> str:
    return (
        f"# TTNN port: `{component_name}` of `{model_id}`\n\n"
        f"**WRITE THIS FILE:** `_synth_responses/{component_safe}.py` "
        f"(under the demo dir). Use your Write tool.\n\n"
        f"Or, if you're a human reading this: paste the entire `## Prompt` "
        f"block below into your chat assistant and let it use its Write "
        f"tool. Then run:\n\n"
        f"```bash\n"
        f"python -m scripts.tt_hw_planner bringup {model_id} \\\n"
        f"    --apply-response {component_name} _synth_responses/{component_safe}.py\n"
        f"python -m scripts.tt_hw_planner bringup {model_id} \\\n"
        f"    --run-tests --component {component_name}\n"
        f"```\n\n"
        f"The response will overwrite `{stub_rel}` after a syntax check + backup.\n\n"
        f"---\n\n"
        f"## Prompt\n\n"
        f"### System\n\n"
        f"{system}\n\n"
        f"### Task\n\n"
        f"{user}\n"
    )


def _render_fallback_prompt(
    *,
    component_name: str,
    component_safe: str,
    model_id: str,
    new_shape: dict,
    candidate_paths: List[str],
    hf_reference: str,
    error_message: str,
    stub_rel: str,
) -> str:
    class_camel = (
        "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", component_safe) if p) or "SynthesizedModule"
    )
    paths_block = "\n".join(f"  - {p}" for p in candidate_paths) or "  (no candidate paths registered)"

    return (
        f"# TTNN port: `{component_name}` of `{model_id}`\n\n"
        f"**This env's `transformers` could not load the HF model**, so the "
        f"upstream source could not be auto-inlined. The prompt below is "
        f"still self-contained — you just need to paste the relevant "
        f"`nn.Module` source into the `## HF source` section manually.\n\n"
        f"Loader error (for context):\n"
        f"```\n{error_message}\n```\n\n"
        f"---\n\n"
        f"## How to use this prompt\n\n"
        f"1. Visit the upstream HF transformers source for `{model_id}`:\n"
        f"   - {hf_reference or '(no path resolved; search the transformers GitHub repo for the modeling file)'}\n"
        f"2. Find the `nn.Module` class that corresponds to `{component_name}` — likely one of these access paths inside the model:\n"
        f"{paths_block}\n"
        f"3. Copy the class source (plus any helper classes it calls) into the `## HF source` block below.\n"
        f"4. Paste the entire `## Prompt` section into your chat assistant.\n"
        f"5. Save the response and apply it:\n\n"
        f"```bash\n"
        f"python -m scripts.tt_hw_planner bringup {model_id} \\\n"
        f"    --apply-response {component_name} _synth_responses/{component_safe}.py\n"
        f"python -m scripts.tt_hw_planner bringup {model_id} \\\n"
        f"    --run-tests --component {component_name}\n"
        f"```\n\n"
        f"The response will overwrite `{stub_rel}` after a syntax check + backup.\n\n"
        f"---\n\n"
        f"## Prompt\n\n"
        f"### System\n\n"
        f"You are an expert TTNN engineer porting PyTorch modules from "
        f"HuggingFace transformers to the Tenstorrent TTNN runtime. You "
        f"produce Python code that runs on Tenstorrent Wormhole/Blackhole "
        f"hardware. You follow project conventions exactly. You never "
        f"invent ops that are not in the provided cheatsheet.\n\n"
        f"### Task\n\n"
        f"Port the PyTorch class below to TTNN. Produce the complete contents "
        f"of `_stubs/{component_safe}.py` and nothing else.\n\n"
        f"TARGET\n"
        f"------\n"
        f"model_id        : {model_id}\n"
        f"component name  : {component_name}\n"
        f"class to expose : {class_camel}\n"
        f"module-level fn : {component_safe}\n"
        f"HF reference    : {hf_reference}\n"
        f"new-model shape : {json.dumps(new_shape or {})}\n"
        f"candidate paths inside the HF model:\n"
        f"{paths_block}\n\n"
        f"## HF source\n\n"
        f"```python\n"
        f"# PASTE THE UPSTREAM HF SOURCE HERE.\n"
        f"#\n"
        f"# Open the file at `{hf_reference or '(see candidate paths above)'}` in the\n"
        f"# transformers GitHub repo, copy the nn.Module class that corresponds\n"
        f"# to `{component_name}`, and paste it between these fences. Include\n"
        f"# any helper classes that the class' __init__ / forward references.\n"
        f"```\n\n"
        f"## TTNN op cheatsheet (the only ops you may use)\n\n"
        f"```\n"
        f"{_TTNN_CHEATSHEET}\n"
        f"```\n\n"
        f"## Project conventions (mandatory)\n\n"
        f"```\n"
        f"{_PROJECT_CONVENTIONS}\n"
        f"```\n\n"
        f"## Output\n\n"
        f"Use the Write tool to write the complete file contents to the "
        f"`WRITE THIS FILE:` path in the section header above. The "
        f"content must start with the 3-line SPDX header and contain "
        f"only Python source (no markdown fences). Do NOT paste the file "
        f"into chat; the bring-up loop reads it from disk.\n"
    )


def _render_adapt_prompt(
    *,
    component_name: str,
    component_safe: str,
    model_id: str,
    cloned_file_rel: str,
    cloned_file_content: str,
    sibling_shape: dict,
    new_shape: dict,
    sibling_hf_id: str,
    target_file_in_responses: str,
) -> str:
    diff_lines: List[str] = []
    all_keys = sorted(set(sibling_shape.keys()) | set(new_shape.keys()))
    for k in all_keys:
        sv = sibling_shape.get(k, "(absent)")
        nv = new_shape.get(k, "(absent)")
        if sv != nv:
            diff_lines.append(f"  - {k}: {sv!r}  ->  {nv!r}")
    diff_block = (
        "\n".join(diff_lines)
        if diff_lines
        else "  (no scalar shape diff detected; check tensor reshapes against the target HF config)"
    )

    return (
        f"# ADAPT: `{component_name}` for `{model_id}`\n\n"
        f"## What you are doing\n\n"
        f"There is already a working TTNN implementation for this component "
        f"in tt-metal — the sibling model's port (`{sibling_hf_id}`). "
        f"`scaffold` has cloned that file into the new demo dir and "
        f"mechanically renamed the sibling base to `{component_safe}`'s "
        f"new base. The algorithm is correct; only the **shape constants** "
        f"need to be substituted to match the new model's config.\n\n"
        f"You are NOT writing this from scratch. You are NOT changing the "
        f"algorithm. You are NOT replacing TTNN ops. You only edit the "
        f"shape constants that depended on the sibling's HF config, and "
        f"any tensor dimensions / loop bounds / dtype choices that depend "
        f"on those constants.\n\n"
        f"## Shape diff (sibling -> new model)\n\n"
        f"```\n{diff_block}\n```\n\n"
        f"## Cloned file content (currently has SIBLING shapes baked in)\n\n"
        f"`{cloned_file_rel}`:\n\n"
        f"```python\n{cloned_file_content}\n```\n\n"
        f"## Output\n\n"
        f"Write the complete adapted Python file to disk at:\n\n"
        f"  `{target_file_in_responses}`\n\n"
        f"Use the same imports, the same class / function structure, the "
        f"same TTNN ops. Only the numeric constants and any dimension "
        f"calculations derived from them should change. If the new model "
        f"adds a config field the sibling didn't have, surface it as a "
        f"clearly-named module attribute; do not silently ignore it.\n\n"
        f"## No-comments rule (mandatory)\n\n"
        f"Do NOT add docstrings, narrative `# ...` comments, or section "
        f"banners while editing. Preserve only the SPDX header at the top "
        f"and the comments that were already in the cloned sibling file. "
        f"If your change deletes a comment that's now wrong (because the "
        f"shape changed), delete it — do NOT add a new comment in its "
        f"place. Pick clearer names instead.\n\n"
        f"After it lands, the user will run "
        f"`bringup {model_id} --apply-all-responses` to install your file.\n"
    )


def _render_demo_prompt(
    *,
    model_id: str,
    model_type: str,
    category: str,
    new_demo_dir_rel: str,
    target_file_in_responses: str,
    new_components: List[str],
    adapt_components: List[str],
    new_shape: dict,
) -> str:
    category_hints = {
        "CNN": (
            "vision — load an image with `PIL.Image.open`, run it through the "
            "model's HF image processor (`AutoImageProcessor.from_pretrained`), "
            "then call the TTNN encoder / decoder. Output: segmentation mask "
            "or class probs depending on the model head."
        ),
        "Image": ("vision — same as CNN. Image processor + encoder + classification head."),
        "Video": (
            "video — load N frames, batch through the encoder, apply temporal "
            "head. Use the model's HF video processor if available."
        ),
        "STT": (
            "speech-to-text — load audio with `torchaudio.load`, run through "
            "the HF feature extractor, then encoder + decoder, then "
            "tokenizer.decode."
        ),
        "TTS": ("text-to-speech — tokenize text, run through encoder/decoder, " "produce mel-spectrogram or waveform."),
        "Embed": ("embedding — tokenize text, forward, pool last-hidden-state."),
        "NLP": ("text — tokenize, forward, sample / argmax."),
        "LLM": (
            "language model — use tt_transformers' `Generator` for the "
            "decode loop; this demo skeleton is mostly stub since LLM "
            "demos already exist in `models/tt_transformers/demo/`."
        ),
        "VLM": (
            "vision-language — image processor + tokenizer, run the vision " "tower then the language model, decode."
        ),
    }
    hint = category_hints.get(category, "Generic — fill in load / forward / postprocess for this architecture.")
    new_block = (
        "\n".join(f"  - `{c}` -> import from `_stubs/{_safe_id(c)}.py`" for c in new_components)
        or "  (no NEW components)"
    )
    adapt_block = (
        "\n".join(f"  - `{c}` -> import from `tt/...` (cloned + adapted)" for c in adapt_components)
        or "  (no ADAPT components)"
    )

    return (
        f"# DEMO entry: `demo.py` for `{model_id}`\n\n"
        f"## What you are doing\n\n"
        f"Write the top-level entry script that loads a real input, runs "
        f"the model on TT hardware end-to-end, and prints / saves the "
        f"output. Use the TTNN modules the user already has under "
        f"`{new_demo_dir_rel}/_stubs/` (NEW components) and "
        f"`{new_demo_dir_rel}/tt/` (ADAPT components). Do NOT write any "
        f"new module bodies in this file — only the pipeline.\n\n"
        f"Category guidance ({category}): {hint}\n\n"
        f"## Components available to wire\n\n"
        f"NEW (TTNN ports the user generated separately):\n"
        f"{new_block}\n\n"
        f"ADAPT (sibling-cloned files, shape-adjusted):\n"
        f"{adapt_block}\n\n"
        f"## Model config snapshot\n\n"
        f"```json\n{json.dumps(new_shape, indent=2)}\n```\n\n"
        f"## Required structure of `demo.py`\n\n"
        f"The file must contain, in this order:\n"
        f"  1. The 3-line SPDX header (rule 1 of project conventions).\n"
        f"  2. Imports: argparse, pytest, torch, ttnn, the HF processor /\n"
        f"     tokenizer appropriate for the category, and the TTNN modules\n"
        f"     from `{new_demo_dir_rel}/_stubs/` and `{new_demo_dir_rel}/tt/`.\n"
        f"  3. A `test_demo(device_params, device)` pytest entry decorated\n"
        f"     with `@pytest.mark.parametrize('device_params', [{{}}], indirect=True)`\n"
        f"     that performs: load real input -> preprocess -> instantiate\n"
        f"     TTNN modules with weights from the HF state_dict -> forward\n"
        f"     on device -> postprocess -> assert sanity (output shape /\n"
        f"     value range).\n"
        f"  4. An `if __name__ == '__main__':` block exposing the same\n"
        f"     pipeline as a standalone CLI (`--input <path>`).\n\n"
        f"## No-comments rule (mandatory)\n\n"
        f"Do NOT add docstrings, inline `# ...` comments, or section\n"
        f"banners. The only comments allowed are the 3-line SPDX header\n"
        f"at the top. Use clearly named helpers (`_preprocess`, `_build_modules`,\n"
        f"`_run_inference`, `_postprocess`) so the pipeline reads itself.\n\n"
        f"## Output\n\n"
        f"Write the complete file to disk at:\n\n"
        f"  `{target_file_in_responses}`\n\n"
        f"Make it runnable: `pytest {new_demo_dir_rel}/demo.py -svv` should "
        f"open the device, run the model end-to-end on a small input, and "
        f"return a sane shape. Use real HF assets (processor / tokenizer) "
        f"so the user can swap in their own input without modifying the "
        f"file. After it lands, the user will run "
        f"`bringup {model_id} --apply-all-responses` to install it.\n"
    )


def emit_prompts(
    *,
    model_id: str,
    repo_root: Path = REPO_ROOT,
    only: Optional[List[str]] = None,
    fetch_upstream: bool = True,
    include_adapt: bool = True,
    include_demo: bool = True,
) -> List[Tuple[str, Path]]:
    if repo_root == REPO_ROOT:
        from .discovery import BRINGUP_ROOT as _BRINGUP_ROOT

        repo_root = _BRINGUP_ROOT()
    targets = list_synth_targets(model_id=model_id, only=only, repo_root=repo_root)

    demo_dir = repo_root / targets.demo_dir
    status_path = demo_dir / "bringup_status.json"
    data = json.loads(status_path.read_text())
    by_name = {c["name"]: c for c in data.get("components", [])}
    if not targets.new and not (include_adapt and targets.adapt) and not include_demo:
        return []

    out_dir = _byo_prompt_dir(demo_dir)
    written: List[Tuple[str, Path]] = []

    for name in targets.new:
        comp = by_name[name]
        safe = _safe_id(name)
        new_shape = comp.get("new_shape") or {}
        sibling_hint = comp.get("sibling_tt_file")
        candidate_paths = COMPONENT_SUBMODULE_HINTS.get(name, [])

        try:
            hf_ctx = extract_hf_context(
                model_id=model_id,
                component_name=name,
                candidate_paths=candidate_paths,
                fetch_upstream=fetch_upstream,
            )
        except LLMError as exc:
            doc = _render_fallback_prompt(
                component_name=name,
                component_safe=safe,
                model_id=model_id,
                new_shape=new_shape,
                candidate_paths=candidate_paths,
                hf_reference=comp.get("hf_reference") or "",
                error_message=str(exc),
                stub_rel=str(safe_relative_to_root(demo_dir / "_stubs" / f"{safe}.py")),
            )
            out_path = out_dir / f"{safe}.prompt.md"
            out_path.write_text(doc)
            written.append((name, out_path))
            continue

        sibling_example = _read_sibling_example(repo_root, sibling_hint)
        manifest_path = demo_dir / "_captured" / safe / "manifest.json"
        opplan_path = demo_dir / "_stubs" / f"{safe}.opplan.json"
        system, user = build_prompts(
            component_name=name,
            component_safe=safe,
            model_id=model_id,
            hf_ctx=hf_ctx,
            new_shape=new_shape,
            sibling_example=sibling_example,
            manifest_path=manifest_path,
            opplan_path=opplan_path,
        )
        stub_rel = safe_relative_to_root(demo_dir / "_stubs" / f"{safe}.py")
        doc = _format_prompt_doc(
            component_name=name,
            component_safe=safe,
            model_id=model_id,
            system=system,
            user=user,
            stub_rel=str(stub_rel),
        )
        out_path = out_dir / f"{safe}.prompt.md"
        out_path.write_text(doc)
        written.append((name, out_path))

    if include_adapt:
        adapts = _collect_adapt_targets(demo_dir=demo_dir, status_data=data)
        for comp_name, sibling_tt_path, new_tt_path, sib_shape, new_shape in adapts:
            if only and comp_name not in only:
                continue
            safe = _safe_id(comp_name)
            try:
                cloned_content = new_tt_path.read_text()
            except OSError:
                continue
            target_in_responses = (
                f"{safe_relative_to_root(demo_dir / _BYO_RESPONSES_DIRNAME)}/{safe}{_ADAPT_RESPONSE_SUFFIX}"
            )
            doc = _render_adapt_prompt(
                component_name=comp_name,
                component_safe=safe,
                model_id=model_id,
                cloned_file_rel=str(safe_relative_to_root(new_tt_path)),
                cloned_file_content=cloned_content,
                sibling_shape=sib_shape,
                new_shape=new_shape,
                sibling_hf_id=data.get("sibling_hf_id", ""),
                target_file_in_responses=target_in_responses,
            )
            out_path = out_dir / f"{safe}.adapt.prompt.md"
            out_path.write_text(doc)
            written.append((f"{comp_name} (adapt)", out_path))

    if include_demo:
        category = data.get("category") or _infer_category_from_demo(demo_dir)
        new_components = [c["name"] for c in data.get("components", []) if c.get("status") == "NEW"]
        adapt_components = [c["name"] for c in data.get("components", []) if c.get("status") == "ADAPT"]
        combined_shape: dict = {}
        for c in data.get("components", []):
            combined_shape.update(c.get("new_shape") or {})
        target_in_responses = f"{safe_relative_to_root(demo_dir / _BYO_RESPONSES_DIRNAME)}/{_DEMO_RESPONSE_NAME}"
        doc = _render_demo_prompt(
            model_id=model_id,
            model_type=data.get("new_model_type", ""),
            category=category,
            new_demo_dir_rel=str(safe_relative_to_root(demo_dir)),
            target_file_in_responses=target_in_responses,
            new_components=new_components,
            adapt_components=adapt_components,
            new_shape=combined_shape,
        )
        out_path = out_dir / "demo.prompt.md"
        out_path.write_text(doc)
        written.append(("demo (entry script)", out_path))

    return written


def _infer_category_from_demo(demo_dir: Path) -> str:
    parts = [p.lower() for p in demo_dir.parts]
    if "vision" in parts:
        return "CNN"
    if "audio" in parts or "speech" in parts:
        return "STT"
    if "video" in parts:
        return "Video"
    if "multimodal" in parts:
        return "VLM"
    if "tt_transformers" in parts or "llama" in parts or "qwen" in parts or "mistral" in parts:
        return "LLM"
    return "NLP"


_HANDOFF_DIRNAME = "_handoff"


def _handoff_dir(demo_dir: Path) -> Path:
    d = demo_dir / _HANDOFF_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _handoff_safe_model_id(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_") or "model"


def build_handoff_master(
    *,
    model_id: str,
    repo_root: Path = REPO_ROOT,
    only: Optional[List[str]] = None,
    fetch_upstream: bool = True,
) -> Optional[Path]:
    if repo_root == REPO_ROOT:
        from .discovery import BRINGUP_ROOT as _BRINGUP_ROOT

        repo_root = _BRINGUP_ROOT()
    single_new_mode = bool(only) and len(only) == 1
    written = emit_prompts(
        model_id=model_id,
        repo_root=repo_root,
        only=only,
        fetch_upstream=fetch_upstream,
    )
    if not written:
        return None

    targets = list_synth_targets(model_id=model_id, only=only, repo_root=repo_root)
    demo_dir = repo_root / targets.demo_dir
    status_data = json.loads((demo_dir / "bringup_status.json").read_text())
    if single_new_mode:
        adapt_targets: List[Tuple[str, Path, Path, dict, dict]] = []
    else:
        adapt_targets = _collect_adapt_targets(demo_dir=demo_dir, status_data=status_data)
    out_dir = _handoff_dir(demo_dir)
    out_path = out_dir / f"{_handoff_safe_model_id(model_id)}__handoff.md"

    responses_rel = safe_relative_to_root(demo_dir / _BYO_RESPONSES_DIRNAME)
    apply_cmd = f"python -m scripts.tt_hw_planner bringup {model_id} --apply-all-responses"

    section_index: List[str] = []
    counter = 0
    for name in targets.new:
        counter += 1
        section_index.append(f"  {counter}. NEW   `{name}` -> write to `{responses_rel}/{_safe_id(name)}.py`")
    for comp_name, _sib, _new_tt, _ss, _ns in adapt_targets:
        counter += 1
        section_index.append(
            f"  {counter}. ADAPT `{comp_name}` -> write to `{responses_rel}/{_safe_id(comp_name)}{_ADAPT_RESPONSE_SUFFIX}`"
        )
    if not single_new_mode:
        counter += 1
        section_index.append(f"  {counter}. DEMO  top-level entry -> write to `{responses_rel}/{_DEMO_RESPONSE_NAME}`")

    if single_new_mode:
        types_block = (
            f"This is a focused per-iteration call: write the ONE NEW component\n"
            f"file listed above. The ADAPT sibling clones and DEMO entry point\n"
            f"are already on disk from earlier scaffolding — DO NOT touch them.\n"
        )
    else:
        types_block = (
            f"The three section types and what they mean:\n"
            f"  - NEW:   write a TTNN module from scratch (HF source + cheatsheet + conventions in the section).\n"
            f"  - ADAPT: a sibling tt-metal file is already cloned for you; edit shape constants ONLY.\n"
            f"  - DEMO:  the top-level entry script that wires everything end-to-end on TT hardware.\n"
        )

    header = (
        f"# Bring-up handoff: write {counter} file(s) for `{model_id}`\n"
        f"\n"
        f"## You are the chat agent. Do this:\n"
        f"\n"
        f"For EACH section below, write **one Python file** to disk at the\n"
        f"path shown. Do NOT paste code into chat — write the file using your\n"
        f"file-writing tool. After all files are written, list the paths so\n"
        f"the user can verify.\n"
        f"\n"
        f"Files to write:\n"
        f"\n" + "\n".join(section_index) + "\n"
        f"\n" + types_block + f"\n"
        f"## House rule for every file you write (mandatory)\n"
        f"\n"
        f"NO COMMENTS in any file you write through this tool. Specifically:\n"
        f"  - No docstrings on modules, classes, or functions.\n"
        f"  - No inline `# ...` comments that narrate what the code does.\n"
        f"  - No section banners or `# ===`-style separators.\n"
        f"The ONLY comments allowed are the 3-line SPDX header at the top of\n"
        f"each file. If a step needs explaining, pick a clearer name for the\n"
        f"variable / method / helper instead of adding a comment. The code\n"
        f"must read itself.\n"
        f"\n"
        f"When done, the user will run:\n"
        f"\n"
        f"    {apply_cmd}\n"
        f"\n"
        f"which scans `{responses_rel}/` and installs every `.py` you wrote\n"
        f"into its correct destination after a syntax check + backup.\n"
        f"\n"
        f"---\n\n"
    )

    sections: List[str] = []

    for name in targets.new:
        safe = _safe_id(name)
        prompt_path = demo_dir / _BYO_PROMPTS_DIRNAME / f"{safe}.prompt.md"
        if not prompt_path.exists():
            sections.append(
                f"## NEW `{name}`\n\n"
                f"(prompt file missing at {safe_relative_to_root(prompt_path)}; "
                f"re-run `--emit-prompts`)\n\n---\n"
            )
            continue
        body = prompt_path.read_text()
        if "\n## Prompt\n" in body:
            body = body.split("\n## Prompt\n", 1)[1]
            body = "## Prompt\n" + body
        target_file = f"{responses_rel}/{safe}.py"
        sections.append(f"## NEW `{name}`\n\n" f"**WRITE THIS FILE:** `{target_file}`\n\n" f"{body}\n\n---\n")

    for comp_name, _sib, _new_tt, _ss, _ns in adapt_targets:
        safe = _safe_id(comp_name)
        prompt_path = demo_dir / _BYO_PROMPTS_DIRNAME / f"{safe}.adapt.prompt.md"
        if not prompt_path.exists():
            sections.append(
                f"## ADAPT `{comp_name}`\n\n"
                f"(prompt file missing at {safe_relative_to_root(prompt_path)}; "
                f"re-run `--emit-prompts`)\n\n---\n"
            )
            continue
        body = prompt_path.read_text()
        target_file = f"{responses_rel}/{safe}{_ADAPT_RESPONSE_SUFFIX}"
        sections.append(f"## ADAPT `{comp_name}`\n\n" f"**WRITE THIS FILE:** `{target_file}`\n\n" f"{body}\n\n---\n")

    demo_prompt_path = demo_dir / _BYO_PROMPTS_DIRNAME / "demo.prompt.md"
    if demo_prompt_path.exists() and not single_new_mode:
        target_file = f"{responses_rel}/{_DEMO_RESPONSE_NAME}"
        sections.append(
            f"## DEMO (top-level entry)\n\n"
            f"**WRITE THIS FILE:** `{target_file}`\n\n"
            f"{demo_prompt_path.read_text()}\n\n---\n"
        )

    out_path.write_text(header + "\n".join(sections))
    return out_path


def apply_all_responses(
    *,
    model_id: str,
    repo_root: Path = REPO_ROOT,
) -> List["ApplyResponseResult"]:
    if repo_root == REPO_ROOT:
        from .discovery import BRINGUP_ROOT as _BRINGUP_ROOT

        repo_root = _BRINGUP_ROOT()
    demo_dir = find_demo_dir(model_id, repo_root=repo_root)
    if demo_dir is None:
        raise LLMError(
            f"No scaffolded demo folder found for {model_id!r}. " f"Run `scaffold {model_id} --apply` first."
        )

    responses_dir = demo_dir / _BYO_RESPONSES_DIRNAME
    if not responses_dir.is_dir():
        raise LLMError(
            f"No `{_BYO_RESPONSES_DIRNAME}/` folder at {safe_relative_to_root(responses_dir)}. "
            f"Run `--handoff-to-chat` first, then have the chat agent write "
            f"response files there."
        )

    status_path = demo_dir / "bringup_status.json"
    data = json.loads(status_path.read_text())
    new_components = {c["name"] for c in data.get("components", []) if c.get("status") == "NEW"}
    new_safe_to_name = {_safe_id(n): n for n in new_components}
    adapt_targets = _collect_adapt_targets(demo_dir=demo_dir, status_data=data)
    adapt_safe_to = {_safe_id(name): (name, dst) for (name, _sib, dst, _ss, _ns) in adapt_targets}

    results: List[ApplyResponseResult] = []
    for fp in sorted(responses_dir.iterdir()):
        if not fp.is_file():
            continue
        fname = fp.name

        if fname == _DEMO_RESPONSE_NAME:
            results.append(
                _install_demo_response(
                    model_id=model_id,
                    demo_dir=demo_dir,
                    response_path=fp,
                    repo_root=repo_root,
                )
            )
            continue

        if fname.endswith(_ADAPT_RESPONSE_SUFFIX):
            safe = fname[: -len(_ADAPT_RESPONSE_SUFFIX)]
            adapt = adapt_safe_to.get(safe)
            if adapt is None:
                results.append(
                    ApplyResponseResult(
                        component=safe,
                        stub_path="",
                        backup_path=None,
                        response_chars=fp.stat().st_size,
                        status="skipped:not-adapt",
                        note=(f"`{fname}` does not match any ADAPT component in the " f"current bring-up plan."),
                    )
                )
                continue
            comp_name, dst = adapt
            results.append(
                _install_adapt_response(
                    model_id=model_id,
                    component_name=comp_name,
                    response_path=fp,
                    destination=dst,
                    repo_root=repo_root,
                )
            )
            continue

        if not fname.endswith(".py"):
            continue
        safe = fp.stem
        component_name = new_safe_to_name.get(safe)
        if component_name is None:
            results.append(
                ApplyResponseResult(
                    component=safe,
                    stub_path="",
                    backup_path=None,
                    response_chars=fp.stat().st_size,
                    status="skipped:not-new",
                    note=(
                        f"`{fname}` does not match any NEW component in the "
                        f"current bring-up plan (probably already-closed or "
                        f"never registered). Left untouched."
                    ),
                )
            )
            continue
        try:
            r = apply_response(
                model_id=model_id,
                component_name=component_name,
                response_path=fp,
                repo_root=repo_root,
            )
        except LLMError as exc:
            r = ApplyResponseResult(
                component=component_name,
                stub_path="",
                backup_path=None,
                response_chars=fp.stat().st_size,
                status="error",
                note=str(exc),
            )
        results.append(r)
    return results


def _install_adapt_response(
    *,
    model_id: str,
    component_name: str,
    response_path: Path,
    destination: Path,
    repo_root: Path,
) -> "ApplyResponseResult":
    raw = response_path.read_text()
    body = _strip_fences(raw)
    if not body.strip():
        return ApplyResponseResult(
            component=component_name,
            stub_path="",
            backup_path=None,
            response_chars=len(raw),
            status="empty",
            note="ADAPT response is empty after stripping markdown fences.",
        )
    if not _looks_like_python(body):
        return ApplyResponseResult(
            component=component_name,
            stub_path="",
            backup_path=None,
            response_chars=len(raw),
            status="syntax-error",
            note="ADAPT response did not parse as Python — inspect + fix + retry.",
        )
    backup_rel: Optional[str] = None
    if destination.exists():
        backup = destination.with_suffix(destination.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(destination, backup)
        backup_rel = str(safe_relative_to_root(backup))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_wrap_for_banner(body))
    return ApplyResponseResult(
        component=f"{component_name} (adapt)",
        stub_path=str(safe_relative_to_root(destination)),
        backup_path=backup_rel,
        response_chars=len(raw),
        status="applied",
        note=f"adapted tt-file installed for {component_name}",
    )


def _install_demo_response(
    *,
    model_id: str,
    demo_dir: Path,
    response_path: Path,
    repo_root: Path,
) -> "ApplyResponseResult":
    raw = response_path.read_text()
    body = _strip_fences(raw)
    if not body.strip():
        return ApplyResponseResult(
            component="demo",
            stub_path="",
            backup_path=None,
            response_chars=len(raw),
            status="empty",
            note="demo response is empty after stripping markdown fences.",
        )
    if not _looks_like_python(body):
        return ApplyResponseResult(
            component="demo",
            stub_path="",
            backup_path=None,
            response_chars=len(raw),
            status="syntax-error",
            note="demo response did not parse as Python — inspect + fix + retry.",
        )
    # Align to tt-metal repo standard: demos live at <demo_dir>/demo/demo.py
    # (matches qwen3_vl/demo/demo.py, bert/demo/demo.py, etc.).
    demo_subdir = demo_dir / "demo"
    demo_subdir.mkdir(parents=True, exist_ok=True)
    destination = demo_subdir / "demo.py"
    backup_rel: Optional[str] = None
    if destination.exists():
        backup = destination.with_suffix(".py.bak")
        if not backup.exists():
            shutil.copy2(destination, backup)
        backup_rel = str(safe_relative_to_root(backup))
    destination.write_text(_wrap_for_banner(body))
    return ApplyResponseResult(
        component="demo (entry script)",
        stub_path=str(safe_relative_to_root(destination)),
        backup_path=backup_rel,
        response_chars=len(raw),
        status="applied",
        note=f"top-level demo installed at {safe_relative_to_root(destination)}",
    )


@dataclass
class ApplyResponseResult:
    component: str
    stub_path: str
    backup_path: Optional[str]
    response_chars: int
    status: str
    note: str = ""


def apply_response(
    *,
    model_id: str,
    component_name: str,
    response_path: Path,
    repo_root: Path = REPO_ROOT,
) -> ApplyResponseResult:
    if repo_root == REPO_ROOT:
        from .discovery import BRINGUP_ROOT as _BRINGUP_ROOT

        repo_root = _BRINGUP_ROOT()
    demo_dir = find_demo_dir(model_id, repo_root=repo_root)
    if demo_dir is None:
        raise LLMError(
            f"No scaffolded demo folder found for {model_id!r}. " f"Run `scaffold {model_id} --apply` first."
        )

    status_path = demo_dir / "bringup_status.json"
    data = json.loads(status_path.read_text())
    comp = next(
        (c for c in data.get("components", []) if c.get("name") == component_name),
        None,
    )
    if comp is None:
        raise LLMError(
            f"Component {component_name!r} not found in bringup_status.json. "
            f"Available: {[c['name'] for c in data.get('components', [])]}"
        )
    if comp.get("status") != "NEW":
        return ApplyResponseResult(
            component=component_name,
            stub_path="",
            backup_path=None,
            response_chars=0,
            status="not-new",
            note=(
                f"Component status is {comp.get('status')!r}; apply-response " f"is gated on NEW only. Nothing changed."
            ),
        )

    if not response_path.is_file():
        raise LLMError(f"Response file does not exist: {response_path}")

    raw = response_path.read_text()
    body = _strip_fences(raw)
    if not body.strip():
        return ApplyResponseResult(
            component=component_name,
            stub_path="",
            backup_path=None,
            response_chars=len(raw),
            status="empty",
            note="Response file is empty after stripping markdown fences.",
        )

    if not _looks_like_python(body):
        return ApplyResponseResult(
            component=component_name,
            stub_path="",
            backup_path=None,
            response_chars=len(raw),
            status="syntax-error",
            note=(
                "Response did not parse as Python. Inspect the file, fix the "
                "syntax (likely a stray fence or prose paragraph), and retry."
            ),
        )

    self_inheriting = _detect_self_inheriting_classes(body)
    if self_inheriting:
        return ApplyResponseResult(
            component=component_name,
            stub_path="",
            backup_path=None,
            response_chars=len(raw),
            status="recursion-trap",
            note=(
                f"Response defines class(es) {self_inheriting} that inherit "
                f"from `<module>.<same_name>`. Once the stub is written, the "
                f"base resolves back to the new class itself, causing "
                f"RecursionError on every call. Rewrite the class to inherit "
                f"directly from `object` (or `torch.nn.Module` if needed) "
                f"and copy any logic from the existing stub explicitly."
            ),
        )

    signature_risks = _detect_call_signature_collisions(body)
    if signature_risks:
        return ApplyResponseResult(
            component=component_name,
            stub_path="",
            backup_path=None,
            response_chars=len(raw),
            status="signature-collision",
            note=(
                f"Response has a collision-prone method signature: "
                f"{signature_risks[0]} Tests may invoke this with the named "
                f"argument both positionally and as a keyword, raising "
                f"`TypeError: got multiple values for argument`. Either drop "
                f"`*args` from the signature, or remove the named parameter "
                f"and read it from `**kwargs` (e.g. "
                f"`val = kwargs.pop('hidden_states')`)."
            ),
        )

    safe = _safe_id(component_name)
    stub_path = demo_dir / "_stubs" / f"{safe}.py"
    backup_rel: Optional[str] = None
    if stub_path.exists():
        backup = stub_path.with_suffix(".py.bak")
        if not backup.exists():
            shutil.copy2(stub_path, backup)
        backup_rel = str(safe_relative_to_root(backup))

    stub_path.parent.mkdir(parents=True, exist_ok=True)
    stub_path.write_text(_wrap_for_banner(body))

    return ApplyResponseResult(
        component=component_name,
        stub_path=str(safe_relative_to_root(stub_path)),
        backup_path=backup_rel,
        response_chars=len(raw),
        status="applied",
        note=(f"Run `bringup {model_id} --run-tests --component {component_name}` " f"to validate via PCC."),
    )


def synthesize_all_new(
    *,
    model_id: str,
    cfg: LLMConfig,
    repo_root: Path = REPO_ROOT,
    run_tests: bool = False,
    max_retries: int = 2,
    dry_run: bool = False,
    only: Optional[List[str]] = None,
    fetch_upstream: bool = True,
) -> List[SynthResult]:
    if repo_root == REPO_ROOT:
        from .discovery import BRINGUP_ROOT as _BRINGUP_ROOT

        repo_root = _BRINGUP_ROOT()
    demo_dir = find_demo_dir(model_id, repo_root=repo_root)
    if demo_dir is None:
        raise LLMError(
            f"No scaffolded demo folder found for {model_id!r}. " f"Run `scaffold {model_id} --apply` first."
        )
    data = json.loads((demo_dir / "bringup_status.json").read_text())
    targets = [
        c["name"]
        for c in data.get("components", [])
        if c.get("status") == "NEW" and (only is None or c["name"] in only)
    ]
    out: List[SynthResult] = []
    for name in targets:
        try:
            res = synthesize_component(
                model_id=model_id,
                component_name=name,
                cfg=cfg,
                repo_root=repo_root,
                run_tests=run_tests,
                max_retries=max_retries,
                dry_run=dry_run,
                fetch_upstream=fetch_upstream,
            )
        except LLMError as exc:
            res = SynthResult(
                component=name,
                stub_path="",
                test_path="",
                final_status="error",
                notes=[f"synthesis raised: {exc}"],
            )
        out.append(res)
    return out


def render_synth_results(results: List[SynthResult], *, model_id: str) -> str:
    if not results:
        return f"No NEW components to synthesize for {model_id}."
    lines = [f"LLM synthesis for {model_id}", "=" * 60, ""]
    for r in results:
        lines.append(f"  component:    {r.component}")
        lines.append(f"  status:       {r.final_status}")
        lines.append(f"  stub:         {r.stub_path}")
        lines.append(f"  test:         {r.test_path}")
        if r.backup_path:
            lines.append(f"  backup:       {r.backup_path}")
        if r.audit_log:
            lines.append(f"  audit log:    {r.audit_log}")
        for i, a in enumerate(r.attempts, start=1):
            lines.append(f"    attempt {i}: {a.test_status}  ({a.response_chars} chars)")
            if a.test_status in ("failed", "syntax-error") and a.test_summary:
                tail = a.test_summary.splitlines()[-1]
                lines.append(f"        last: {tail[:160]}")
        for note in r.notes:
            lines.append(f"  note: {note}")
        lines.append("")
    return "\n".join(lines)


def render_synth_json(results: List[SynthResult]) -> str:
    def to_dict(r: SynthResult) -> dict:
        return {
            "component": r.component,
            "stub_path": r.stub_path,
            "test_path": r.test_path,
            "final_status": r.final_status,
            "backup_path": r.backup_path,
            "audit_log": r.audit_log,
            "attempts": [
                {
                    "attempt": a.attempt,
                    "test_status": a.test_status,
                    "response_chars": a.response_chars,
                    "test_summary": a.test_summary,
                }
                for a in r.attempts
            ],
            "notes": r.notes,
        }

    return json.dumps([to_dict(r) for r in results], indent=2)
