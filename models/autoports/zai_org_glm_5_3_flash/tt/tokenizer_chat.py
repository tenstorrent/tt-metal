# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tokenizer, chat template and stop-token set for zai-org/GLM-5.3-Flash (`glm5_next`), host side.

Everything is read from the checkpoint snapshot (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`generation_config.json`); nothing is hard-coded except the constants that are verified against those files at load.
The implementation needs only `tokenizers` and `jinja2` (no `transformers`), so it runs in the TT python env
(transformers 5.12, which predates `glm5_next`) and in the vLLM venv alike. The chat template is rendered with the
same jinja2 environment as `transformers.utils.chat_template_utils._cached_compile_jinja_template` (sandboxed,
`trim_blocks`, `lstrip_blocks`, `loopcontrols`, transformers' `tojson` / `raise_exception` / `strftime_now`), and the
rendered text is tokenized with `add_special_tokens=False`, exactly as `PreTrainedTokenizerBase.apply_chat_template`.
Token-for-token parity with transformers 5.19 is tested in `tests/test_tokenizer_chat.py` (golden prompts and 50
random conversations).

Template modes (`chat_template.jinja` of revision eb9eb208eb0d):
  * `reasoning_effort` in {"low", "high"} renders `<|system|>Reasoning Effort: Low|High`; anything else (default)
    renders `Reasoning Effort: Max`. The header is always emitted, before any user/system message.
  * The generation prompt is always `<|assistant|><think>`: the template has NO no-thinking mode (there is no
    `enable_thinking` switch; the vocabulary has a `/nothink` token, 154851, that the template never emits).
  * `clear_thinking` (default False): reasoning of assistant turns before the last user message is replaced by
    `<think></think>`; otherwise every assistant turn keeps `<think>{reasoning}</think>` (from `reasoning_content`,
    or split out of `content` at `</think>`).
  * `tools`: a `# Tools` system block with JSON schemas; assistant `tool_calls` render as
    `<tool_call>{name}<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`; `tool` messages render inside one
    `<|observation|>` block as `<tool_response>...</tool_response>` (sorted by tool_call_id when unambiguous).
  * image / video / audio content parts render placeholder tokens (vision is out of milestone-1 scope).

Stop tokens: `generation_config.eos_token_id` = [154820 `<|endoftext|>`, 154827 `<|user|>`, 154829 `<|observation|>`]
(`<|user|>` ends an assistant turn, `<|observation|>` hands over to a tool); pad = 154820. Sampling defaults from the
checkpoint: temperature 1.0, top_p 0.95, no top_k.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HF_MODEL_ID = "zai-org/GLM-5.3-Flash"
HF_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
MODEL_VOCAB_SIZE = 154880  # lm_head rows (text_config.vocab_size); the tokenizer uses ids 0..154855
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
REASONING_EFFORTS = ("low", "high", "max")
# Verified against generation_config.json / tokenizer.json at load time (a mismatch raises).
EXPECTED_STOP_TOKENS = {154820: "<|endoftext|>", 154827: "<|user|>", 154829: "<|observation|>"}
CONTINUE_FINAL_MESSAGE_TAG = "CONTINUE_FINAL_MESSAGE_TAG "

TEMPLATE_MODES = {
    "reasoning_effort": {
        "values": ["low", "high", "max"],
        "default": "max",
        "rendered": "<|system|>Reasoning Effort: {Low|High|Max} (always present, first after [gMASK]<sop>)",
        "invalid_values": "any value other than 'low'/'high' renders Max",
    },
    "thinking": {
        "generation_prompt": "<|assistant|><think>",
        "no_thinking_mode": False,
        "note": "the template always opens <think>; no enable_thinking switch. Vocabulary token /nothink (154851) "
        "exists but the template never emits it.",
    },
    "clear_thinking": {
        "default": False,
        "effect": "assistant turns before the last user message render <think></think> instead of their reasoning",
    },
    "tools": {
        "system_block": "# Tools ... <tools>{json schemas}</tools> + <tool_call> format instructions",
        "assistant_tool_call": "<tool_call>{name}<arg_key>{k}</arg_key><arg_value>{v}</arg_value>...</tool_call>",
        "tool_response": "<|observation|><tool_response>{content}</tool_response>... (one block per run of tool msgs)",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_snapshot(path: str | os.PathLike | None = None) -> Path:
    """Checkpoint directory holding the tokenizer files: explicit path, else $GLM53_HF_SNAPSHOT, else the HF hub
    cache layout under $HF_HOME / ~/.cache/huggingface or /home/alex/mpi-shfs/llm-wksp/models."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    if os.environ.get("GLM53_HF_SNAPSHOT"):
        candidates.append(Path(os.environ["GLM53_HF_SNAPSHOT"]))
    hub_roots = [
        Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/home/alex/mpi-shfs/llm-wksp/models"),
    ]
    for root in hub_roots:
        if root is not None:
            candidates.append(root / "models--zai-org--GLM-5.3-Flash" / "snapshots" / HF_REVISION)
    for c in candidates:
        if all((c / f).exists() for f in TOKENIZER_FILES):
            return c
    raise FileNotFoundError(f"no GLM-5.3-Flash snapshot with {TOKENIZER_FILES} among {[str(c) for c in candidates]}")


def _compile_template(source: str):
    """jinja2 environment identical to transformers' chat-template compiler (minus its `{% generation %}` tracker
    extension, which this template does not use)."""
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env.from_string(source)


@dataclass
class ParsedAssistant:
    """Assistant output split the way the template writes it back: reasoning, visible content, tool calls."""

    reasoning_content: str | None
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


class Glm53Tokenizer:
    """HF-equivalent tokenizer + chat template + stop tokens. Exposes the subset of the transformers tokenizer API the
    generator, the readiness runners and the vLLM adapter use (`encode`, `decode`, `apply_chat_template`,
    `eos_token_id`, `pad_token_id`, `convert_ids_to_tokens`, `__call__`)."""

    def __init__(self, snapshot: str | os.PathLike | None = None):
        from tokenizers import Tokenizer

        self.snapshot = resolve_snapshot(snapshot)
        self.files_sha256 = {f: _sha256(self.snapshot / f) for f in TOKENIZER_FILES}
        self._tok = Tokenizer.from_file(str(self.snapshot / "tokenizer.json"))
        self.config = json.loads((self.snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.generation_config = json.loads((self.snapshot / "generation_config.json").read_text(encoding="utf-8"))
        self.chat_template = (self.snapshot / "chat_template.jinja").read_text(encoding="utf-8")
        self._template = _compile_template(self.chat_template)

        self.eos_token = self.config.get("eos_token")
        self.pad_token = self.config.get("pad_token")
        self.eos_token_id = self.token_to_id(self.eos_token)
        self.pad_token_id = self.token_to_id(self.pad_token)
        self.bos_token_id = None  # the template writes [gMASK]<sop>; encode() adds nothing
        eos = self.generation_config.get("eos_token_id")
        self.stop_token_ids: tuple[int, ...] = tuple(eos if isinstance(eos, list) else [eos])
        names = {i: self.id_to_token(i) for i in self.stop_token_ids}
        if names != EXPECTED_STOP_TOKENS:
            raise ValueError(f"stop tokens {names} differ from the audited set {EXPECTED_STOP_TOKENS}")
        if self.generation_config.get("pad_token_id") != self.pad_token_id:
            raise ValueError("generation_config pad_token_id != tokenizer pad token")
        self.generation_defaults = {
            k: self.generation_config[k] for k in ("temperature", "top_p", "top_k") if k in self.generation_config
        }
        self.special_token_ids = frozenset(
            int(i) for i, t in self._tok.get_added_tokens_decoder().items() if getattr(t, "special", False)
        )
        self.vocab_size = self._tok.get_vocab_size(with_added_tokens=True)
        if self.vocab_size > MODEL_VOCAB_SIZE:
            raise ValueError(f"tokenizer vocab {self.vocab_size} > model vocab {MODEL_VOCAB_SIZE}")
        self.model_vocab_size = MODEL_VOCAB_SIZE
        self.special_tokens_map = {k: self.config[k] for k in ("eos_token", "pad_token") if self.config.get(k)}

    # ------------------------------------------------------------------------------------------------ token level
    def token_to_id(self, token: str) -> int:
        i = self._tok.token_to_id(token)
        if i is None:
            raise KeyError(token)
        return i

    def id_to_token(self, token_id: int) -> str | None:
        return self._tok.id_to_token(int(token_id))

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.id_to_token(ids)
        return [self.id_to_token(i) for i in ids]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Plain text -> ids. `add_special_tokens` is accepted for API compatibility; the checkpoint's ByteLevel
        post-processor adds no tokens either way (asserted on every golden prompt in the parity tests, against
        transformers with add_special_tokens=True as well)."""
        return list(self._tok.encode(text, add_special_tokens=add_special_tokens).ids)

    def __call__(self, text: str | Sequence[str], add_special_tokens: bool = False, **_):
        if isinstance(text, str):
            return {"input_ids": self.encode(text, add_special_tokens)}
        return {"input_ids": [self.encode(t, add_special_tokens) for t in text]}

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        ids = [int(i) for i in ids]
        bad = [i for i in ids if not 0 <= i < self.vocab_size]
        if bad:
            # ids in [tokenizer vocab, 154880) are lm_head padding rows that a correct model never samples
            raise ValueError(f"token ids {bad[:8]} outside the tokenizer vocabulary [0, {self.vocab_size})")
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def is_stop_token(self, token_id: int) -> bool:
        return int(token_id) in self.stop_token_ids

    # ------------------------------------------------------------------------------------------------ chat level
    def render_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool | str = False,
        reasoning_effort: str | None = None,
        clear_thinking: bool | None = None,
        documents=None,
        **template_kwargs: Any,
    ) -> str:
        """messages -> rendered prompt text (transformers `apply_chat_template(tokenize=False)` semantics).
        `reasoning_effort` / `clear_thinking` are only passed to the template when not None (template defaults)."""
        if not messages:
            raise ValueError("Cannot apply chat template to an empty conversation. Provide at least one message.")
        if continue_final_message and add_generation_prompt:
            raise ValueError("continue_final_message and add_generation_prompt are not compatible")
        if tools is not None:
            for t in tools:
                if not isinstance(t, Mapping):
                    raise TypeError("tools must be JSON-schema dicts (callables are converted by transformers only)")
            tools = list(tools)
        kwargs = {**self.special_tokens_map, **template_kwargs}
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if clear_thinking is not None:
            kwargs["clear_thinking"] = clear_thinking
        chat = list(messages)
        final_message = None
        if continue_final_message:
            chat = deepcopy(chat)
            key = continue_final_message if isinstance(continue_final_message, str) else "content"
            final_message = chat[-1].get(key)
            if final_message is None:
                raise ValueError(f'continue_final_message is set but the final message has no "{key}" to continue!')
            if key not in self.chat_template:
                raise ValueError(f'continue_final_message is set to "{key}" but this is not a template field')
            if isinstance(final_message, (list, tuple)):
                for block in reversed(final_message):
                    if "text" in block:
                        final_message = block["text"]
                        block["text"] = block["text"] + CONTINUE_FINAL_MESSAGE_TAG
                        break
                else:
                    raise ValueError("continue_final_message: no text block in the final message")
            else:
                chat[-1][key] = chat[-1][key] + CONTINUE_FINAL_MESSAGE_TAG
        rendered = self._template.render(
            messages=chat, tools=tools, documents=documents, add_generation_prompt=add_generation_prompt, **kwargs
        )
        if continue_final_message:
            tag = CONTINUE_FINAL_MESSAGE_TAG
            if final_message.strip() not in rendered or tag.strip() not in rendered:
                raise ValueError("continue_final_message: the final message does not survive the chat template")
            loc = rendered.rindex(tag.strip())
            rendered = rendered[:loc] if rendered[loc : loc + len(tag)] == tag else rendered[:loc].rstrip()
        return rendered

    def apply_chat_template(
        self, messages: Sequence[Mapping[str, Any]], *, tokenize: bool = True, **kwargs: Any
    ) -> list[int] | str:
        """transformers-compatible entry point: returns ids (`tokenize=True`, as `return_dict=False`) or text.
        Unlike transformers' default, `add_generation_prompt` defaults to True here (the serving case)."""
        text = self.render_chat(messages, **kwargs)
        return self.encode(text) if tokenize else text

    def encode_chat(self, messages: Sequence[Mapping[str, Any]], **kwargs: Any) -> list[int]:
        return self.encode(self.render_chat(messages, **kwargs))

    # ------------------------------------------------------------------------------------------------ output side
    def parse_assistant(self, generated: str | Sequence[int]) -> ParsedAssistant:
        """Split generated assistant text (what follows the `<|assistant|><think>` generation prompt) into reasoning,
        content and tool calls, inverting the template's assistant rendering. Stop tokens are dropped. Argument values
        are JSON-decoded when they parse: the template writes string values raw and other values as JSON, so the
        string "3" and the number 3 render identically and come back as 3."""
        if not isinstance(generated, str):
            ids = [int(i) for i in generated]
            while ids and self.is_stop_token(ids[-1]):
                ids.pop()
            generated = self.decode(ids, skip_special_tokens=False)
        text = generated
        reasoning = None
        if "</think>" in text:
            head, text = text.split("</think>", 1)
            reasoning = head.split("<think>")[-1]
        calls = []
        for body in _TOOL_CALL_RE.findall(text):
            name = body.split("<arg_key>", 1)[0].strip()
            args = {}
            for k, v in _ARG_RE.findall(body):
                try:
                    args[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    args[k] = v
            calls.append({"name": name, "arguments": args})
        content = _TOOL_CALL_RE.sub("", text).strip()
        return ParsedAssistant(reasoning_content=reasoning, content=content, tool_calls=calls)

    # ------------------------------------------------------------------------------------------------ provenance
    def describe(self) -> dict[str, Any]:
        """Prompt-format / tokenizer metadata for evidence files ($qualitative-check prompt-format record)."""
        return {
            "hf_model": HF_MODEL_ID,
            "hf_revision": HF_REVISION,
            "snapshot": str(self.snapshot),
            "files_sha256": self.files_sha256,
            "tokenizer_impl": "tt/tokenizer_chat.py (tokenizers + jinja2; transformers-equivalent rendering)",
            "tokenizer_class_hf": self.config.get("tokenizer_class"),
            "chat_template_present": bool(self.chat_template.strip()),
            "prompt_mode": "chat",
            "tokenizer_vocab_size": self.vocab_size,
            "model_vocab_size": self.model_vocab_size,
            "stop_token_ids": list(self.stop_token_ids),
            "stop_tokens": {i: self.id_to_token(i) for i in self.stop_token_ids},
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "generation_defaults": self.generation_defaults,
            "template_modes": TEMPLATE_MODES,
        }


def load_tokenizer(snapshot: str | os.PathLike | None = None) -> Glm53Tokenizer:
    return Glm53Tokenizer(snapshot)
