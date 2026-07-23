# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Chunked KV prefill and optional trace capture for AR recaption prefix processing.

from __future__ import annotations

import os
import time

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import COMPUTE_CQ


def prefill_chunk_size() -> int:
    """Tokens per prefill chunk; 0 disables chunking."""
    return int(os.environ.get("HY_RECAPTION_PREFILL_CHUNK", "1024"))


def recaption_trace_prefill_enabled() -> bool:
    """Prefix prefill cannot be trace-captured: KV ``replace()`` writes DRAM during prefill.

    TT raises ``Writes are not supported during trace capture`` if ``begin_trace_capture``
    wraps a full prefill forward. Decode trace is OK (fixed KV buffers + ``paged_update_cache``).
    Chunked eager prefill remains available via ``HY_RECAPTION_PREFILL_CHUNK``.
    """
    from models.experimental.hunyuan_image_3_0.tt.trace_config import hy_trace_enabled

    if not hy_trace_enabled():
        return False
    if os.environ.get("HY_RECAPTION_TRACE_PREFILL", "0") != "0":
        print(
            "[recaption] HY_RECAPTION_TRACE_PREFILL=1 ignored: prefill uses KV replace() "
            "writes that are illegal during trace capture; using eager/chunked prefill",
            flush=True,
        )
    return False


def run_kv_prefill(tracer, *, use_trace_prefill: bool) -> ttnn.Tensor:
    """Fill KV cache for ``tracer.prefix_len`` tokens; return last-token logits on device."""
    chunk = prefill_chunk_size()
    if chunk > 0 and tracer.prefix_len > chunk:
        return _chunked_kv_prefill(tracer)
    if use_trace_prefill:
        return _traced_single_prefill(tracer)
    return _single_prefill_step(tracer, start=0, end=tracer.prefix_len, return_logits=True)


def _upload_mask(tracer, query_start: int, query_end: int, total_len: int) -> ttnn.Tensor | None:
    # Text-only: SDPA ``is_causal`` handles prefill — skip host S×S build/upload.
    if not tracer.attn_slices:
        return None
    if query_start == 0 and query_end == total_len:
        mask_bool = build_attention_mask(total_len, tracer.attn_slices, bsz=1)
        mask_add = to_additive(mask_bool, dtype=torch.bfloat16).reshape(1, 1, total_len, total_len)
    else:
        # mask_bool is [bsz, 1, S, S]; query rows live on dim 2, not dim 1.
        mask_bool = build_attention_mask(total_len, tracer.attn_slices, bsz=1)
        mask_add = to_additive(mask_bool, dtype=torch.bfloat16)
        mask_add = mask_add[:, :, query_start:query_end, :].reshape(1, 1, query_end - query_start, total_len)
    if tracer.replicate_to_mesh is not None:
        return tracer.replicate_to_mesh(mask_add)
    return ttnn.from_torch(
        mask_add,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=tracer.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _single_prefill_step(tracer, *, start: int, end: int, return_logits: bool) -> ttnn.Tensor | None:
    chunk_len = end - start
    hidden_tt = tracer.prefix_hidden_slice(start, end)
    if tracer._cos_full is None or tracer._sin_full is None:
        tracer._cos_full, tracer._sin_full = tracer.model.layers[0].self_attn.rope.prepare_cos_sin(
            tracer.max_cache_len, image_infos=tracer.image_infos
        )
    cos_tt = ttnn.slice(tracer._cos_full, [0, 0, start, 0], [1, 1, end, tracer._cos_full.shape[-1]])
    sin_tt = ttnn.slice(tracer._sin_full, [0, 0, start, 0], [1, 1, end, tracer._sin_full.shape[-1]])
    mask_tt = _upload_mask(tracer, start, end, end)
    hidden = tracer.model.forward(
        inputs_embeds=hidden_tt,
        seq_len=chunk_len,
        image_infos=tracer.image_infos,
        attention_mask=mask_tt,
        kv_cache=tracer.kv_cache,
        use_cache=True,
        decode_step=False,
        cos_sin=(cos_tt, sin_tt),
    )
    logits_tt = None
    if return_logits:
        logits_tt = tracer.lm_head(hidden, last_token_only=True)
    ttnn.deallocate(mask_tt)
    ttnn.deallocate(hidden_tt)
    ttnn.deallocate(cos_tt)
    ttnn.deallocate(sin_tt)
    ttnn.deallocate(hidden)
    tracer.kv_cache.seq_len = end
    return logits_tt


def _chunked_kv_prefill(tracer) -> ttnn.Tensor:
    chunk = prefill_chunk_size()
    n_chunks = (tracer.prefix_len + chunk - 1) // chunk
    t0 = time.perf_counter()
    print(
        f"[recaption] chunked KV prefill prefix_len={tracer.prefix_len} " f"chunk_size={chunk} chunks={n_chunks}",
        flush=True,
    )
    logits_tt = None
    start = 0
    for idx in range(n_chunks):
        end = min(start + chunk, tracer.prefix_len)
        is_last = end == tracer.prefix_len
        t_chunk = time.perf_counter()
        logits_tt = _single_prefill_step(tracer, start=start, end=end, return_logits=is_last)
        print(
            f"[recaption] prefill chunk {idx + 1}/{n_chunks} "
            f"tokens [{start}:{end}] took {(time.perf_counter() - t_chunk) * 1000:.2f} ms",
            flush=True,
        )
        start = end
    print(
        f"[recaption] chunked prefill done in {(time.perf_counter() - t0) * 1000:.2f} ms",
        flush=True,
    )
    return logits_tt


def _traced_single_prefill(tracer) -> ttnn.Tensor:
    t0 = time.perf_counter()
    logits_w = _single_prefill_step(tracer, start=0, end=tracer.prefix_len, return_logits=True)
    ttnn.deallocate(logits_w)
    ttnn.synchronize_device(tracer.device)
    if tracer.dual_cq is not None:
        tracer.dual_cq.fence_compute_before_forward()
    print("[recaption] trace prefill: warmup done, using begin_trace_capture", flush=True)
    tracer.prefill_trace_id = ttnn.begin_trace_capture(tracer.device, cq_id=COMPUTE_CQ)
    logits_tt = _single_prefill_step(tracer, start=0, end=tracer.prefix_len, return_logits=True)
    ttnn.end_trace_capture(tracer.device, tracer.prefill_trace_id, cq_id=COMPUTE_CQ)
    ttnn.synchronize_device(tracer.device)
    print(
        f"[recaption] trace prefill captured trace_id={tracer.prefill_trace_id} "
        f"took {(time.perf_counter() - t0) * 1000:.2f} ms",
        flush=True,
    )
    return logits_tt
