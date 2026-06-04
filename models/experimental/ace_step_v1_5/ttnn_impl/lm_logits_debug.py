# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Debug helpers for experimental 5 Hz LM logits PCC bring-up.

Enable with ``ACE_STEP_DEBUG_LM_LOGITS=1`` (optional ``ACE_STEP_DEBUG_LM_LOGITS_HF=1`` to
also run a HuggingFace reference forward on the host for staged PCC).

Typical staged tags (prefill):

- ``prefill.qwen_params`` — ``seq_len``, padded prefill len, ``get_last_token``, tile offset
- ``prefill.raw_tt_tile`` — full ``[1,1,32,V]`` tile row vs HF last-token logits
- ``prefill.tile_row{i}`` — PCC for each row ``i`` in the 32-wide tile (find wrong row pick)
- ``prefill.after_offset_pick`` — row selected via ``_prefill_last_token_offset_in_tile``
- ``prefill.bridge_expanded_last`` — final ``[1,S,V]`` tensor at position ``S-1`` (what sampling uses)

Decode:

- ``decode.raw_tt`` — TTNN decode logits vs HF single-step logits
- ``decode.bridge_trimmed`` — after batch-pad trim along sequence axis

Set ``ACE_STEP_DEBUG_LM_LOGITS_LAYER_PCC=1`` to print HF per-layer hidden-state PCC at the
last real token (host only; does not instrument TTNN layers).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch


def _env_truthy(name: str) -> bool:
    v = (os.environ.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on", "y")


def ace_step_debug_lm_logits_enabled() -> bool:
    return _env_truthy("ACE_STEP_DEBUG_LM_LOGITS")


def ace_step_debug_lm_logits_hf_enabled() -> bool:
    return ace_step_debug_lm_logits_enabled() and _env_truthy("ACE_STEP_DEBUG_LM_LOGITS_HF")


def ace_step_debug_lm_logits_layer_pcc_enabled() -> bool:
    return ace_step_debug_lm_logits_enabled() and _env_truthy("ACE_STEP_DEBUG_LM_LOGITS_LAYER_PCC")


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = np.asarray(a.detach().float().cpu().numpy(), dtype=np.float64).reshape(-1)
    bb = np.asarray(b.detach().float().cpu().numpy(), dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0:
        return 1.0
    if aa.size != bb.size:
        n = min(aa.size, bb.size)
        aa, bb = aa[:n], bb[:n]
    if np.array_equal(aa, bb):
        return 1.0
    r = float(np.corrcoef(aa, bb)[0, 1])
    if not np.isfinite(r):
        return 1.0
    return r


def log_lm_tensor_stats(tag: str, t: torch.Tensor, *, extra: str = "") -> None:
    if not ace_step_debug_lm_logits_enabled():
        return
    x = t.detach().float().cpu().reshape(-1)
    m = torch.isfinite(x)
    if not bool(m.any()):
        print(f"[ace_step_lm_logits_debug] {tag} shape={tuple(t.shape)} NO_FINITE {extra}", flush=True)
        return
    xf = x[m]
    suffix = f" {extra}" if extra else ""
    print(
        f"[ace_step_lm_logits_debug] {tag} shape={tuple(t.shape)} "
        f"min={float(xf.min()):.6g} max={float(xf.max()):.6g} "
        f"mean={float(xf.mean()):.6g} std={float(xf.std(unbiased=False)):.6g} "
        f"n_finite={int(xf.numel())}{suffix}",
        flush=True,
    )


def log_lm_pcc(tag: str, ref: torch.Tensor, got: torch.Tensor, *, min_pcc: float = 0.99) -> float:
    if not ace_step_debug_lm_logits_enabled():
        return 1.0
    r = _pearson(ref, got)
    status = "PASS" if r >= min_pcc else "FAIL"
    print(
        f"[ace_step_lm_logits_debug] {tag}: PCC={r:.8f} ({status} vs {min_pcc}) "
        f"shapes ref={tuple(ref.shape)} got={tuple(got.shape)}",
        flush=True,
    )
    return r


def log_lm_topk_match(tag: str, ref: torch.Tensor, got: torch.Tensor, k: int = 5) -> None:
    if not ace_step_debug_lm_logits_enabled():
        return
    ref1 = ref.detach().float().cpu().reshape(-1)
    got1 = got.detach().float().cpu().reshape(-1)
    v = min(ref1.numel(), got1.numel())
    ref1, got1 = ref1[:v], got1[:v]
    ref_top = torch.topk(ref1, min(k, v)).indices.tolist()
    got_top = torch.topk(got1, min(k, v)).indices.tolist()
    argmax_match = int(torch.argmax(ref1).item()) == int(torch.argmax(got1).item())
    print(
        f"[ace_step_lm_logits_debug] {tag}: argmax_match={argmax_match} " f"ref_top{k}={ref_top} got_top{k}={got_top}",
        flush=True,
    )


def _hf_last_logits(
    lm_dir: str,
    input_ids: torch.Tensor,
    *,
    past_key_values: Any = None,
) -> torch.Tensor:
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        str(lm_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    with torch.inference_mode():
        out = hf(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
    del hf
    return out.logits[:, -1:, :].float().cpu()


def debug_hf_reference_last_logits(
    lm_dir: str,
    input_ids: torch.Tensor,
    *,
    past_key_values: Any = None,
) -> torch.Tensor:
    """Host HF forward; last-position logits ``[1, 1, V]`` (float32 CPU)."""
    return _hf_last_logits(lm_dir, input_ids, past_key_values=past_key_values)


def debug_hf_layer_hidden_pcc(
    lm_dir: str,
    input_ids: torch.Tensor,
    *,
    min_pcc: float = 0.99,
) -> None:
    """Print HF-only layer index / hidden dim stats (baseline for TTNN layer bring-up)."""
    if not ace_step_debug_lm_logits_layer_pcc_enabled():
        return
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        str(lm_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    with torch.inference_mode():
        out = hf(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    last_idx = int(input_ids.shape[1]) - 1
    print(
        f"[ace_step_lm_logits_debug] hf_layer_hidden: n_layers={len(hs)} last_token_idx={last_idx}",
        flush=True,
    )
    prev = None
    for i, h in enumerate(hs):
        row = h[0, last_idx, :].float().cpu()
        log_lm_tensor_stats(f"hf_layer_hidden.{i}", row)
        if prev is not None:
            log_lm_pcc(f"hf_layer_hidden.{i-1}_vs_{i}_autocorr", prev, row, min_pcc=min_pcc)
        prev = row
    del hf


def debug_compare_prefill_logits_stages(
    *,
    hf_ref_last: torch.Tensor,
    logits_tt_torch: torch.Tensor,
    last_token_offset_in_tile: Optional[int],
    seq_log: int,
    vocab: int,
    lm_dir: Optional[str] = None,
    input_ids: Optional[torch.Tensor] = None,
    qwen_params: Optional[dict[str, Any]] = None,
) -> None:
    if not ace_step_debug_lm_logits_enabled():
        return

    ref = hf_ref_last.detach().float().cpu()
    if ref.dim() == 3:
        ref_last = ref[:, -1:, :vocab]
    else:
        ref_last = ref.view(1, 1, -1)[:, :, :vocab]

    if qwen_params:
        parts = " ".join(f"{k}={v!r}" for k, v in qwen_params.items())
        print(f"[ace_step_lm_logits_debug] prefill.qwen_params {parts}", flush=True)
        padded = qwen_params.get("prefill_seq_len")
        seq_len = qwen_params.get("seq_len")
        if padded is not None and seq_len is not None and int(padded) != int(seq_len):
            print(
                f"[ace_step_lm_logits_debug] prefill.reference_note: TTNN prefill runs "
                f"{int(padded)} tokens (padded); HF PCC tests usually pass only {int(seq_len)} "
                f"real tokens — last-position logits can still match (~0.91 PCC) because causal "
                f"mask isolates position {int(seq_len)-1}, but decode KV may diverge.",
                flush=True,
            )

    got = logits_tt_torch.detach().float().cpu()
    log_lm_tensor_stats("prefill.raw_tt_logits", got)

    # Normalise to [1, S_tile, V]
    if got.shape[-1] != vocab:
        if got.shape[-1] > vocab:
            got = got[..., :vocab]
        elif int(got.shape[-2]) == vocab:
            got = got.transpose(-1, -2)
    tile = got.reshape(1, -1, vocab)
    s_tile = int(tile.shape[1])

    log_lm_pcc("prefill.raw_tt_flat_vs_hf", ref_last, tile.reshape(1, 1, -1)[:, :, :vocab], min_pcc=0.99)

    for i in range(min(32, s_tile)):
        row = tile[:, i : i + 1, :]
        log_lm_pcc(f"prefill.tile_row{i}", ref_last, row, min_pcc=0.99)

    offset = last_token_offset_in_tile
    if offset is not None and 0 <= int(offset) < s_tile:
        picked = tile[:, int(offset) : int(offset) + 1, :]
        log_lm_pcc("prefill.after_offset_pick", ref_last, picked, min_pcc=0.99)
        log_lm_topk_match("prefill.after_offset_pick", ref_last, picked)
    else:
        picked = tile[:, -1:, :]
        log_lm_pcc("prefill.after_offset_pick_fallback_last_row", ref_last, picked, min_pcc=0.99)
        log_lm_topk_match("prefill.after_offset_pick_fallback_last_row", ref_last, picked)

    if seq_log > 1:
        expanded = torch.zeros(1, seq_log, vocab, dtype=picked.dtype)
        expanded[:, -1, :] = picked[0, 0, :]
        log_lm_pcc("prefill.bridge_expanded_last", ref_last, expanded[:, -1:, :], min_pcc=0.99)

    if ace_step_debug_lm_logits_hf_enabled() and lm_dir and input_ids is not None:
        hf = debug_hf_reference_last_logits(lm_dir, input_ids)
        log_lm_pcc("prefill.hf_rerun_last", ref_last, hf, min_pcc=0.99)


def debug_compare_decode_logits_stages(
    *,
    hf_ref_last: torch.Tensor,
    logits_tt_torch: torch.Tensor,
    seq_log: int,
    vocab: int,
    qwen_params: Optional[dict[str, Any]] = None,
) -> None:
    if not ace_step_debug_lm_logits_enabled():
        return

    if qwen_params:
        parts = " ".join(f"{k}={v!r}" for k, v in qwen_params.items())
        print(f"[ace_step_lm_logits_debug] decode.qwen_params {parts}", flush=True)

    ref = hf_ref_last.detach().float().cpu()
    if ref.dim() == 3:
        ref_last = ref[:, -1:, :vocab]
    else:
        ref_last = ref.view(1, 1, -1)[:, :, :vocab]

    got = logits_tt_torch.detach().float().cpu()
    log_lm_tensor_stats("decode.raw_tt_logits", got)

    if got.shape[-1] != vocab:
        if got.shape[-1] > vocab:
            got = got[..., :vocab]
        elif int(got.shape[-2]) == vocab:
            got = got.transpose(-1, -2)
    got = got.reshape(-1, vocab).view(1, -1, vocab)
    s_out = int(got.shape[1])
    if s_out > seq_log:
        trimmed = got[:, :seq_log, :]
    else:
        trimmed = got

    log_lm_pcc("decode.raw_tt_vs_hf", ref_last, trimmed[:, -1:, :], min_pcc=0.99)
    log_lm_topk_match("decode.raw_tt_vs_hf", ref_last, trimmed[:, -1:, :])
    if s_out > seq_log:
        log_lm_pcc("decode.bridge_trimmed_vs_hf", ref_last, trimmed[:, -1:, :], min_pcc=0.99)


__all__ = [
    "ace_step_debug_lm_logits_enabled",
    "ace_step_debug_lm_logits_hf_enabled",
    "ace_step_debug_lm_logits_layer_pcc_enabled",
    "debug_compare_decode_logits_stages",
    "debug_compare_prefill_logits_stages",
    "debug_hf_layer_hidden_pcc",
    "debug_hf_reference_last_logits",
    "log_lm_pcc",
    "log_lm_tensor_stats",
    "log_lm_topk_match",
]
