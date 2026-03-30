# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import NamedTuple, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.demos.speculative_deepseek_r1_broad.base_runtime import DecodeState
from models.demos.speculative_deepseek_r1_broad.base_verification import PathVerification, verify_paths_from_decode_state
from models.demos.speculative_deepseek_r1_broad.config import EagleConfig, PathProposal
from models.demos.speculative_deepseek_r1_broad.default_paths import NEXTN_HF_REPO_ID


class _ModelStubConfig:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)


class _ModelStub:
    def __init__(self, vocab_size: int) -> None:
        self.config = _ModelStubConfig(vocab_size=vocab_size)


class TracePosSoftmaxStats(NamedTuple):
    """Softmax of stored base logits at trace step ``pos`` for the greedy record token."""

    pos: int
    record_greedy_token_id: int
    p_record_greedy: float
    p_max: float
    argmax_token_id: int


@dataclass(frozen=True)
class TraceBundle:
    model_id: str
    prompt: str
    prompt_token_ids: list[int]
    step_next_tokens: list[int]
    step_last_hidden: torch.Tensor  # [num_steps, hidden_size]
    step_multi_layer_hidden: torch.Tensor | None  # [num_steps, 3*hidden_size]
    topk_tokens: torch.Tensor | None  # [num_steps, k]
    topk_scores: torch.Tensor | None  # [num_steps, k]
    # MTP reference only: full vocab logits before each step's greedy next token [num_steps, vocab_size]
    step_next_logits: torch.Tensor | None = None
    # When set, :class:`TraceReplayBaseAdapter` loads the tokenizer from this Hub id instead of
    # ``model_id`` (unless ``tokenizer_local_dir`` or ``tokenizer_hub_model_id`` is passed).
    # MTP reference bundles set this to ``NEXTN_HF_REPO_ID`` so we never touch the full R1 repo
    # for tokenizer-only loads.
    tokenizer_hub_id: str | None = None
    # --- MTP-only (None for collect_base traces) ---
    # How ``prompt_token_ids`` was chosen when loading the MTP reference.
    mtp_prefix_source: str | None = None
    # Copy of ``start_tokens`` from the file [batch_size] (authoritative per-row first context id
    # when no long prefill is stored). None for collect traces.
    mtp_batch_start_tokens: tuple[int, ...] | None = None
    # Optional scalar from ``metadata`` (e.g. test harness default); can differ from per-row tensor.
    mtp_metadata_start_token_id: int | None = None


def trace_bundle_from_collect_payload(payload: dict) -> TraceBundle:
    """Build TraceBundle from a collect_base_trace-style dict (already loaded)."""
    if not isinstance(payload, dict):
        raise ValueError(f"Trace payload is not a dict: {type(payload)}")
    step_last_hidden = payload["step_last_hidden"].cpu()
    if step_last_hidden.ndim != 2:
        raise ValueError(f"Expected step_last_hidden rank 2, got shape={tuple(step_last_hidden.shape)}")
    multi = payload.get("step_multi_layer_hidden")
    if multi is not None:
        multi = multi.cpu()
    thub = payload.get("tokenizer_hub_id")
    return TraceBundle(
        model_id=str(payload["model_id"]),
        prompt=str(payload["prompt"]),
        prompt_token_ids=[int(x) for x in payload["prompt_token_ids"]],
        step_next_tokens=[int(x) for x in payload["step_next_tokens"]],
        step_last_hidden=step_last_hidden,
        step_multi_layer_hidden=multi,
        topk_tokens=payload.get("topk_tokens", None),
        topk_scores=payload.get("topk_scores", None),
        step_next_logits=None,
        tokenizer_hub_id=str(thub) if thub is not None else None,
    )


def load_trace_bundle(path: str | Path) -> TraceBundle:
    payload = torch.load(str(path), map_location="cpu")
    return trace_bundle_from_collect_payload(payload)


def _is_mtp_reference_payload(payload: dict) -> bool:
    return isinstance(payload, dict) and "hidden_states" in payload and "next_tokens" in payload


def _coerce_token_id_sequence(
    raw: object,
    batch_index: int,
    batch_size: int,
) -> list[int] | None:
    """Turn optional payload/metadata field into a flat list of token ids for one batch row."""
    if raw is None:
        return None
    if not isinstance(raw, torch.Tensor) and hasattr(raw, "__array__"):
        try:
            raw = torch.as_tensor(raw)
        except (TypeError, ValueError):
            return None
    if isinstance(raw, torch.Tensor):
        t = raw.long().cpu()
        if t.ndim == 1:
            return [int(t[i].item()) for i in range(t.shape[0])]
        if t.ndim == 2:
            if t.shape[0] == batch_size:
                return [int(t[batch_index, j].item()) for j in range(t.shape[1])]
            if t.shape[0] == 1:
                return [int(t[0, j].item()) for j in range(t.shape[1])]
        return None
    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        if isinstance(raw[0], (list, tuple)):
            if batch_index < len(raw):
                return [int(x) for x in raw[batch_index]]
            return None
        return [int(x) for x in raw]
    return None


_PREFILL_ID_KEYS = (
    "prefill_token_ids",
    "prompt_token_ids",
    "prefix_token_ids",
    "context_token_ids",
    "full_prompt_token_ids",
    "all_input_ids",
    "cached_input_ids",
    "input_token_ids",
    "input_ids",
    "token_ids",
    "sequence",
    # Some dumps use a scalar or 1-element sequence as the whole prefill.
    "start_token",
)


def _mtp_prefill_token_ids(
    payload: dict,
    batch_index: int,
    batch_size: int,
    *,
    num_steps: int,
    metadata: dict,
) -> list[int] | None:
    """Resolve full prefill token ids from common MTP / HF dump layouts."""

    def try_keys(src: dict | None) -> list[int] | None:
        if not isinstance(src, dict):
            return None
        for key in _PREFILL_ID_KEYS:
            raw = src.get(key)
            if key == "start_token" and isinstance(raw, int) and not isinstance(raw, bool):
                return [int(raw)]
            seq = _coerce_token_id_sequence(raw, batch_index, batch_size)
            if seq:
                return seq
        return None

    # Optional explicit prefill length + one flat id tensor (prefill || prefill+decode).
    for nk in ("num_prefill_tokens", "prompt_len", "prefill_len", "num_prompt_tokens"):
        raw_n = None
        if isinstance(metadata, dict):
            raw_n = metadata.get(nk)
        if raw_n is None and nk in payload:
            raw_n = payload.get(nk)
        if raw_n is None:
            continue
        try:
            n_pre = int(raw_n)
        except (TypeError, ValueError):
            continue
        if n_pre <= 0:
            continue
        seq = try_keys(payload) or try_keys(metadata)
        if seq is None:
            for v in payload.values():
                if isinstance(v, dict):
                    seq = try_keys(v)
                    if seq:
                        break
        if seq is None:
            continue
        if len(seq) == n_pre + num_steps:
            return seq[:n_pre]
        if len(seq) >= n_pre:
            return seq[:n_pre]

    for src in (payload, metadata):
        got = try_keys(src if isinstance(src, dict) else None)
        if got:
            return got

    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, dict):
                got = try_keys(v)
                if got:
                    return got
    return None


def load_mtp_reference_bundle(
    path: str | Path | None = None,
    payload: dict | None = None,
    batch_index: int = 0,
) -> TraceBundle:
    """Load a DeepSeek v3 MTP reference .pt file and convert it to a TraceBundle.

    Expects the format produced by models/demos/deepseek_v3/tests/test_mtp.py
    (test_generate_mtp_reference_io): hidden_states [num_steps, batch, hidden_size],
    next_tokens [num_steps, batch], start_tokens [batch], and optional metadata.

    Optional **prefill** (so replay prefix matches the real input and ``decode`` is not empty):
    tries ``prefill_token_ids``, ``prompt_token_ids``, ``input_ids``, ``start_token`` (scalar),
    ``token_ids``, ``sequence``, etc. on the payload, ``metadata``, or one-level nested dicts
    (HF-style ``inputs`` blobs).
    If ``metadata`` sets ``num_prefill_tokens`` / ``prompt_len`` / ``prefill_len`` and a long
    ``input_ids`` vector equals ``prefill + num_steps`` tokens, only the prefill prefix is used.
    If nothing matches, **``prompt_token_ids = [start_tokens[batch_index]]``** — the per-row
    tensor is the MTP I/O contract (do not substitute ``metadata.start_token_id`` for row > 0).
    Uses the given batch_index (default 0) to build a single-sequence TraceBundle.

    Provide either path (file path to load) or payload (already-loaded dict).
    """
    if payload is None:
        if path is None:
            raise ValueError("Provide either path or payload")
        payload = torch.load(str(path), map_location="cpu")
    if not _is_mtp_reference_payload(payload):
        raise ValueError(
            "File is not an MTP reference payload: expected dict with 'hidden_states' and 'next_tokens'"
        )
    hidden_states = payload["hidden_states"].cpu().float()
    next_tokens = payload["next_tokens"].cpu()
    start_tokens = payload["start_tokens"].cpu()
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected hidden_states rank 3 [steps, batch, hidden], got {tuple(hidden_states.shape)}")
    if next_tokens.ndim != 2:
        raise ValueError(f"Expected next_tokens rank 2 [steps, batch], got {tuple(next_tokens.shape)}")
    num_steps, batch_size, hidden_size = hidden_states.shape
    if batch_index < 0 or batch_index >= batch_size:
        raise ValueError(f"batch_index must be in [0, {batch_size}), got {batch_index}")
    if next_tokens.shape[0] != num_steps or next_tokens.shape[1] != batch_size:
        raise ValueError("next_tokens shape does not match hidden_states")
    if start_tokens.shape[0] != batch_size:
        raise ValueError("start_tokens length does not match batch_size")

    metadata = payload.get("metadata") or {}
    model_id = str(metadata.get("model_id", "deepseek-ai/DeepSeek-R1-0528"))
    mtp_batch_start_tokens = tuple(int(start_tokens[i].item()) for i in range(batch_size))
    raw_meta_st = metadata.get("start_token_id")
    try:
        mtp_metadata_start_token_id = int(raw_meta_st) if raw_meta_st is not None else None
    except (TypeError, ValueError):
        mtp_metadata_start_token_id = None

    prefill = _mtp_prefill_token_ids(
        payload, batch_index, batch_size, num_steps=num_steps, metadata=metadata
    )
    if prefill is not None:
        prompt_token_ids = prefill
        mtp_prefix_source = "explicit_prefill"
    else:
        # Per-row tensor is the I/O contract for batched MTP references (hidden/next_tokens align).
        prompt_token_ids = [int(start_tokens[batch_index].item())]
        mtp_prefix_source = "start_tokens_tensor"
    step_next_tokens = [int(next_tokens[i, batch_index].item()) for i in range(num_steps)]
    step_last_hidden = hidden_states[:, batch_index, :].clone()

    step_next_logits: torch.Tensor | None = None
    raw_logits = payload.get("logits")
    if raw_logits is not None:
        lt = raw_logits.cpu()
        if lt.ndim == 3:
            if lt.shape[0] != num_steps or lt.shape[1] != batch_size:
                raise ValueError(
                    f"logits shape {tuple(lt.shape)} does not match hidden_states "
                    f"steps/batch ({num_steps}, {batch_size})"
                )
            step_next_logits = lt[:, batch_index, :].float().clone()
        elif lt.ndim == 2:
            if lt.shape[0] != num_steps:
                raise ValueError(f"logits rank-2 expected first dim {num_steps}, got {lt.shape[0]}")
            step_next_logits = lt.float().clone()
        else:
            raise ValueError(f"Expected logits rank 2 or 3, got shape {tuple(lt.shape)}")

    return TraceBundle(
        model_id=model_id,
        prompt="",  # No natural-language prompt in MTP reference; see mtp_* fields + prompt_token_ids
        prompt_token_ids=prompt_token_ids,
        step_next_tokens=step_next_tokens,
        step_last_hidden=step_last_hidden,
        step_multi_layer_hidden=None,
        topk_tokens=None,
        topk_scores=None,
        step_next_logits=step_next_logits,
        tokenizer_hub_id=NEXTN_HF_REPO_ID,
        mtp_prefix_source=mtp_prefix_source,
        mtp_batch_start_tokens=mtp_batch_start_tokens,
        mtp_metadata_start_token_id=mtp_metadata_start_token_id,
    )


def format_mtp_prefix_banner(trace: TraceBundle, *, batch_index: int) -> str:
    """Human-readable explanation of MTP record prefill (for script stdout)."""
    if trace.mtp_prefix_source is None:
        return ""
    lines: list[str] = []
    if trace.mtp_prefix_source == "explicit_prefill":
        lines.append(
            "MTP record: prefix_token_ids from explicit prefill fields in the .pt "
            "(prompt_token_ids / input_ids / prefill_token_ids / …)."
        )
    else:
        lines.append(
            "MTP record: no long prefill in file — replay prefix is **start_tokens[batch_index]** "
            f"(one context token per batch row). Row {batch_index} → prefill ids {trace.prompt_token_ids!r}."
        )
        if trace.mtp_metadata_start_token_id is not None:
            lines.append(
                f"  Note: metadata.start_token_id={trace.mtp_metadata_start_token_id} is a global scalar from "
                "the test harness; it is **not** used instead of per-row start_tokens[] (would misalign hiddens "
                "for batch_index>0). Pick --batch-index to match the row you care about."
            )
    st = trace.mtp_batch_start_tokens
    if st:
        if len(st) <= 24:
            lines.append(f"  start_tokens (all rows, len={len(st)}): {st!r}")
        else:
            lines.append(
                f"  start_tokens: len={len(st)} head={st[:12]!r} … tail={st[-6:]!r} "
                f"(row {batch_index}={st[batch_index]})"
            )
    lines.append(
        "  Engine uses prefix_token_ids=trace.prompt_token_ids (above) so draft KV/positions match the record."
    )
    return "\n".join(lines)


def load_trace_or_mtp_reference(path: str | Path, batch_index: int = 0) -> TraceBundle:
    """Load a trace file in either collect_base_trace format or DeepSeek v3 MTP reference format.

    Auto-detects by payload keys: if 'hidden_states' and 'next_tokens' are present,
    treats as MTP reference; otherwise uses load_trace_bundle (trace from collect_base_trace_gpu).
    """
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Trace file is not a dict: {type(payload)}")
    if _is_mtp_reference_payload(payload):
        return load_mtp_reference_bundle(payload=payload, batch_index=batch_index)
    return trace_bundle_from_collect_payload(payload)


class TraceReplayBaseAdapter:
    """Replay-only base adapter backed by collected decode trace data.

    This supports EAGLE-style control flow on CPU without running the base model.
    Accuracy is exact only on the recorded greedy trajectory.

    Pass ``tokenizer_local_dir`` to load the tokenizer **only from disk** (no Hub). Typical
    layout: a Hugging Face snapshot folder that contains ``tokenizer.json`` / ``tokenizer_config.json``
    alongside ``config.json`` (same path as ``--local-deepseek-snapshot`` in record-only runs).

    For MTP reference traces, :attr:`TraceBundle.tokenizer_hub_id` defaults to
    ``lmsys/DeepSeek-R1-NextN`` so tokenizer resolution does not use ``deepseek-ai/DeepSeek-R1-0528``
    (full-weight repo). Override with ``tokenizer_hub_model_id`` or ``tokenizer_local_dir`` if needed.
    """

    def __init__(
        self,
        trace: TraceBundle,
        *,
        tokenizer_local_dir: str | Path | None = None,
        tokenizer_hub_model_id: str | None = None,
        tokenizer_trust_remote_code: bool = False,
    ) -> None:
        self.trace = trace
        self.device = torch.device("cpu")
        if tokenizer_local_dir is not None:
            tdir = str(Path(tokenizer_local_dir).expanduser().resolve())
            self.tokenizer = AutoTokenizer.from_pretrained(
                tdir,
                local_files_only=True,
                trust_remote_code=tokenizer_trust_remote_code,
            )
        else:
            hub_id = tokenizer_hub_model_id if tokenizer_hub_model_id is not None else trace.tokenizer_hub_id
            if hub_id is None:
                hub_id = trace.model_id
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(hub_id),
                trust_remote_code=tokenizer_trust_remote_code,
            )
        vocab_size = int(getattr(self.tokenizer, "vocab_size", 0))
        if vocab_size <= 0:
            vocab_size = max(max(trace.prompt_token_ids, default=0), max(trace.step_next_tokens, default=0)) + 1
        if trace.step_next_logits is not None:
            vocab_size = max(vocab_size, int(trace.step_next_logits.shape[-1]))
        self.model = _ModelStub(vocab_size=vocab_size)
        self._hidden_size = int(trace.step_last_hidden.shape[-1])
        self._logged_missing_logits_for_confidence = False

    @staticmethod
    def format_prefix_display(tokenizer: object, token_ids: Sequence[int]) -> str:
        """Human-readable prefix: decode, or ``convert_ids_to_tokens`` if decode is blank."""
        ids = [int(x) for x in token_ids]
        if not ids:
            return "(empty prefix)"
        for skip_special in (True, False):
            text = tokenizer.decode(ids, skip_special_tokens=skip_special)
            if text.strip():
                return text
        try:
            parts = tokenizer.convert_ids_to_tokens(ids)
            return " ".join(parts) if parts else repr(ids[:16])
        except Exception:
            return repr(ids[:32]) + ("…" if len(ids) > 32 else "")

    def encode_prompt(self, prompt: str) -> list[int]:
        encoded = self.tokenizer(prompt, return_tensors=None, add_special_tokens=True)
        input_ids = encoded["input_ids"]
        if len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return [int(token_id) for token_id in input_ids]

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def _state_meta(self, pos: int, valid: bool) -> dict[str, int | bool]:
        return {"pos": int(pos), "valid": bool(valid)}

    def _build_logits(self, token_id: int) -> torch.Tensor:
        logits = torch.full((1, int(self.model.config.vocab_size)), fill_value=-1e9, dtype=torch.float32)
        if 0 <= token_id < logits.shape[-1]:
            logits[0, token_id] = 0.0
        return logits

    def _state_from(self, pos: int, valid: bool) -> DecodeState:
        multi = None
        if pos < len(self.trace.step_next_tokens):
            token_id = int(self.trace.step_next_tokens[pos])
            hidden = self.trace.step_last_hidden[pos].unsqueeze(0).float().clone()
            if (
                self.trace.step_next_logits is not None
                and pos < self.trace.step_next_logits.shape[0]
            ):
                logits = self.trace.step_next_logits[pos].unsqueeze(0).float().clone()
            else:
                logits = self._build_logits(token_id) if valid else self._build_logits(0)
            if self.trace.step_multi_layer_hidden is not None and pos < self.trace.step_multi_layer_hidden.shape[0]:
                multi = self.trace.step_multi_layer_hidden[pos].unsqueeze(0).float().clone()
        else:
            hidden = torch.zeros((1, self._hidden_size), dtype=torch.float32)
            logits = self._build_logits(0)
        return DecodeState(
            past_key_values=self._state_meta(pos=pos, valid=valid),
            next_token_logits=logits,
            last_hidden_state=hidden,
            multi_layer_hidden=multi,
        )

    def forward_prefill(self, prefix_token_ids: Sequence[int]) -> DecodeState:
        expected = self.trace.prompt_token_ids
        given = [int(x) for x in prefix_token_ids]
        if given != expected:
            raise ValueError("Trace replay requires prompt tokens to match the collected trace prompt.")
        return self.create_decode_state(prefix_token_ids)

    def forward_decode(self, state: DecodeState, token_id: int) -> DecodeState:
        return self.advance_decode_state(state, token_id)

    def create_decode_state(self, prefix_token_ids: Sequence[int]) -> DecodeState:
        expected = self.trace.prompt_token_ids
        given = [int(x) for x in prefix_token_ids]
        if given != expected:
            raise ValueError("Trace replay requires prompt tokens to match the collected trace prompt.")
        return self._state_from(pos=0, valid=True)

    def _decode_state_for_committed(self, committed: Sequence[int]) -> DecodeState:
        """Decode state at trace position ``len(committed) - len(prompt)`` (greedy-prefix alignment)."""
        prompt = list(self.trace.prompt_token_ids)
        given = [int(x) for x in committed]
        if len(given) < len(prompt) or given[: len(prompt)] != prompt:
            raise ValueError("Trace replay: committed sequence must begin with trace.prompt_token_ids")
        pos = len(given) - len(prompt)
        max_pos = len(self.trace.step_next_tokens)
        if pos > max_pos:
            return self._state_from(pos=max_pos, valid=False)
        valid = True
        for i in range(pos):
            if given[len(prompt) + i] != int(self.trace.step_next_tokens[i]):
                valid = False
                break
        return self._state_from(pos=pos, valid=valid)

    def decode_state_next_token(self, state: DecodeState) -> int:
        meta = state.past_key_values if isinstance(state.past_key_values, dict) else {}
        pos = int(meta.get("pos", 0))
        valid = bool(meta.get("valid", False))
        if not valid or pos >= len(self.trace.step_next_tokens):
            return -1
        return int(self.trace.step_next_tokens[pos])

    def clone_decode_state(self, state: DecodeState) -> DecodeState:
        meta = state.past_key_values if isinstance(state.past_key_values, dict) else {}
        return DecodeState(
            past_key_values={"pos": int(meta.get("pos", 0)), "valid": bool(meta.get("valid", False))},
            next_token_logits=state.next_token_logits.clone(),
            last_hidden_state=state.last_hidden_state.clone(),
            multi_layer_hidden=state.multi_layer_hidden.clone() if state.multi_layer_hidden is not None else None,
        )

    def advance_decode_state(self, state: DecodeState, token_id: int) -> DecodeState:
        meta = state.past_key_values if isinstance(state.past_key_values, dict) else {}
        pos = int(meta.get("pos", 0))
        valid = bool(meta.get("valid", False))
        expected = int(self.trace.step_next_tokens[pos]) if pos < len(self.trace.step_next_tokens) else -1
        next_valid = valid and (token_id == expected)
        next_pos = min(pos + 1, len(self.trace.step_next_tokens))
        return self._state_from(pos=next_pos, valid=next_valid)

    def verify_paths_from_decode_state(
        self,
        decode_state: DecodeState,
        proposed_paths: Sequence[Sequence[int]],
        *,
        acceptance_mode: str = "argmax",
        rng: random.Random | None = None,
        draft_probs_per_path: Sequence[Sequence[float]] | None = None,
        return_base_argmax: bool = False,
    ) -> PathVerification:
        return verify_paths_from_decode_state(
            decode_state,
            proposed_paths,
            clone_decode_state=self.clone_decode_state,
            advance_decode_state=self.advance_decode_state,
            acceptance_mode=acceptance_mode,
            rng=rng,
            draft_probs_per_path=draft_probs_per_path,
            return_base_argmax=return_base_argmax,
        )

    def softmax_stats_at_trace_pos(self, pos: int) -> TracePosSoftmaxStats | None:
        """Probability of the **record's** greedy token at trace index ``pos`` (needs real logits in trace)."""
        if pos < 0 or pos >= len(self.trace.step_next_tokens):
            return None
        tid = int(self.trace.step_next_tokens[pos])
        logits_1d = self._state_from(pos=pos, valid=True).next_token_logits.reshape(-1)
        p = F.softmax(logits_1d.float(), dim=-1)
        if tid < 0 or tid >= p.shape[-1]:
            return TracePosSoftmaxStats(pos, tid, float("nan"), float("nan"), int(torch.argmax(p, dim=-1).item()))
        return TracePosSoftmaxStats(
            pos=pos,
            record_greedy_token_id=tid,
            p_record_greedy=float(p[tid].item()),
            p_max=float(p.max().item()),
            argmax_token_id=int(torch.argmax(p, dim=-1).item()),
        )

    @staticmethod
    def format_base_token_confidence(logits_1d: torch.Tensor, token_id: int) -> str:
        """Softmax stats for one token id (for correlation with speculative acceptance)."""
        p = F.softmax(logits_1d.float().reshape(-1), dim=-1)
        v = p.shape[-1]
        if token_id < 0 or token_id >= v:
            return "p(drafted)=NA p_max=NA argmax=NA (token_id out of vocab)"
        pt = float(p[token_id].item())
        pmax = float(p.max().item())
        argm = int(torch.argmax(p, dim=-1).item())
        match = "Y" if argm == token_id else "N"
        ent = float((-(p * (p.clamp_min(1e-30)).log())).sum().item())
        return (
            f"p(drafted)={pt:.6e} p_max={pmax:.6e} entropy={ent:.4f} "
            f"argmax={argm} argmax_eq_drafted={match}"
        )

    def log_base_confidence_round_preamble(self, decode_state: DecodeState, round_idx: int) -> None:
        """One line before verify: record greedy next token vs stored logits (needs step_next_logits)."""
        if self.trace.step_next_logits is None:
            if not self._logged_missing_logits_for_confidence:
                print(
                    "[base_conf] Trace has no step_next_logits; verification still uses synthetic one-hot logits. "
                    "Use an MTP reference .pt that includes `logits` for real base confidence.",
                    flush=True,
                )
                self._logged_missing_logits_for_confidence = True
            return
        meta = decode_state.past_key_values if isinstance(decode_state.past_key_values, dict) else {}
        pos = int(meta.get("pos", 0))
        valid = bool(meta.get("valid", False))
        logits_1d = decode_state.next_token_logits.reshape(-1)
        rec_next = self.decode_state_next_token(decode_state) if valid else -1
        line = self.format_base_token_confidence(logits_1d, rec_next) if rec_next >= 0 else "invalid_state"
        print(
            f"[base_conf] round={round_idx} pos={pos} record_greedy_next={rec_next} pre_verify {line}",
            flush=True,
        )

    def log_round_replay_detail(
        self,
        *,
        round_idx: int,
        decode_state: DecodeState,
        paths: list[list[int]],
        draft_probs_per_path: list[list[float]] | None,
        verification: PathVerification,
        best_path_idx: int,
        best_accepted_len: int,
        acceptance_mode: str,
    ) -> None:
        """Summarize one speculative round (counts, not full token lists)."""
        meta = decode_state.past_key_values if isinstance(decode_state.past_key_values, dict) else {}
        pos = int(meta.get("pos", 0))
        valid = bool(meta.get("valid", False))

        has_logits = self.trace.step_next_logits is not None
        logits_source = "record_logits" if has_logits else "synthetic_one_hot"

        rec_next = self.decode_state_next_token(decode_state) if valid else -1
        lg = decode_state.next_token_logits.reshape(-1)
        conf_line = self.format_base_token_confidence(lg, rec_next) if rec_next >= 0 else "(no record next)"

        b0 = verification.base_argmax_pos0
        total_proposed = sum(len(p) for p in paths)
        acc_lens = list(verification.accepted_prefix_lengths)
        print(
            f"\n=== spec_round round={round_idx} trace_pos={pos} valid={valid} verify={acceptance_mode} ===\n"
            f"  record_greedy_next@pos={rec_next} | logits_source={logits_source}\n"
            f"  base(replay_logits) for that token: {conf_line}\n"
            f"  base_argmax_pos0={b0} | record_eq_argmax={'Y' if rec_next >= 0 and b0 == rec_next else 'N'}\n"
            f"  draft_paths={len(paths)} total_proposed_token_slots={total_proposed} "
            f"accepted_prefix_lens={acc_lens}",
            flush=True,
        )

        sel = (
            f"path[{best_path_idx}] accepted={best_accepted_len}"
            if best_path_idx >= 0
            else "none"
        )
        print(f"  SELECTED: {sel}\n", flush=True)

    def verify_paths_batched_single_pass(
        self,
        prefix_token_ids: Sequence[int],
        proposed_paths: Sequence[Sequence[int]],
        *,
        acceptance_mode: str = "argmax",
        rng: random.Random | None = None,
        draft_probs_per_path: Sequence[Sequence[float]] | None = None,
        return_base_argmax: bool = False,
        per_path_forward: bool = False,
    ) -> PathVerification:
        del per_path_forward  # trace replay has no batched transformer forward
        state = self._decode_state_for_committed(prefix_token_ids)
        return self.verify_paths_from_decode_state(
            state,
            proposed_paths,
            acceptance_mode=acceptance_mode,
            rng=rng,
            draft_probs_per_path=draft_probs_per_path,
            return_base_argmax=return_base_argmax,
        )

    def verify_paths_flattened_tree(
        self,
        prefix_token_ids: Sequence[int],
        proposed_paths: Sequence[Sequence[int]],
        *,
        acceptance_mode: str = "argmax",
        rng: random.Random | None = None,
        draft_probs_per_path: Sequence[Sequence[float]] | None = None,
        return_base_argmax: bool = False,
    ) -> PathVerification:
        """No joint attention forward on trace; same acceptance as :meth:`verify_paths_from_decode_state`."""
        state = self._decode_state_for_committed(prefix_token_ids)
        return self.verify_paths_from_decode_state(
            state,
            proposed_paths,
            acceptance_mode=acceptance_mode,
            rng=rng,
            draft_probs_per_path=draft_probs_per_path,
            return_base_argmax=return_base_argmax,
        )


class RecordMTPDraftAdapter:
    """MTP-style draft adapter that proposes paths from a recorded trajectory (no live draft model).

    Uses the record's step_next_tokens: at position pos, proposes a single path of length
    depth from the record, path = [step_next_tokens[pos], ..., step_next_tokens[pos+depth-1]].
    This allows testing the speculative loop with MTP at depth 2+ on CPU using only the
    record file (speculative_deepseek_r1_broad record-based MTP pipeline).
    """

    def __init__(self, trace: TraceBundle) -> None:
        self.trace = trace

    def propose_paths(
        self,
        prefix_token_ids: Sequence[int],
        cfg: EagleConfig,
        *,
        decode_state: DecodeState | None = None,
        base_adapter: object = None,
    ) -> PathProposal:
        if decode_state is None:
            return PathProposal(paths=[], draft_probs_per_path=None)
        meta = decode_state.past_key_values if isinstance(decode_state.past_key_values, dict) else {}
        pos = int(meta.get("pos", 0))
        if pos < 0 or pos >= len(self.trace.step_next_tokens):
            return PathProposal(paths=[], draft_probs_per_path=None)
        depth = max(1, int(cfg.depth))
        end = min(pos + depth, len(self.trace.step_next_tokens))
        if end <= pos:
            return PathProposal(paths=[], draft_probs_per_path=None)
        path = [int(self.trace.step_next_tokens[i]) for i in range(pos, end)]
        return PathProposal(paths=[path], draft_probs_per_path=None)
