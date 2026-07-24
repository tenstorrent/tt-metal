# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the Generator contract: pack the (cos,sin) rope pair into one
tensor (copy_host_to_device cannot carry a nested tuple), and the shared
short/long prefill dispatch (so the wrapper and demo define the seam once)."""
import torch

import ttnn


def pack_rope_host(cos_host, sin_host):
    """HOST path (decode): cos,sin are HOST ttnn tensors (from rope.get_cos_sin_host),
    so ttnn.concat (a device op) can't be used — pack via torch instead."""
    cos_t = ttnn.to_torch(cos_host)
    sin_t = ttnn.to_torch(sin_host)
    packed = torch.cat([cos_t, sin_t], dim=0)
    return ttnn.from_torch(packed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)


def unpack_rope(packed):
    """Split a pack_rope_host() tensor back into (cos, sin). Works on a
    device tensor (slicing dim 0); called inside ttnn_decode_forward."""
    n = packed.shape[0] // 2
    return packed[0:n], packed[n : 2 * n]


def prefill_dispatch(model, tokens, page_table, prompt_lens, use_trace, vision_tokens=None):
    """All prefill is model-owned. traced -> chunk-outer trace; non-traced -> paged.
    Both fill the paged KV cache + finalize GDN state, so decode continues correctly.

    The traced path has a single entry, prefill_traced_chunked, for EVERY input length up to
    128k: it replays the captured 2048-token chunk trace for each full chunk, then runs the
    final partial chunk (or a whole short prompt, when there are no full chunks) through the
    masked fixed-bucket path. The masked path runs at one of a few pre-warmed bucket lengths, so
    it never compiles a new program at request time and can't clobber the parked trace — the
    short-prompt / long-tail hang fix. Defining the short/long seam inside prefill_traced_chunked
    (num_full==0 -> masked bucket) keeps it in one place.

    NOTE (vLLM block allocation): the masked path writes K/V for the full bucket, so the
    page_table must map enough blocks to cover the rounded-up bucket length (<= 2048 -> 32
    blocks of 64), not just the real prompt length.

    vision_tokens (multimodal): the image embeddings to splice into the text embeddings. The
    traced path splices them with a fixed-shape ttnn.where over persistent buffers (trace-safe —
    compiled at warmup, updated per request by copy_host_to_device), so multimodal works WITH a
    captured trace (single device). The non-traced path uses the on-device scatter in
    prefill_paged, which is safe only because no trace is parked there.
    """
    T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
    if use_trace:
        return model.prefill_traced_chunked(tokens, page_table, actual_len=T, vision_tokens=vision_tokens)
    # The single-device paged path derives its sequence length from tokens.shape and returns logits
    # for the last token, so a bucket-padded buffer would prefill the padding and read out the pad
    # boundary instead of prompt_lens[0]-1. Clip to the real length T first (the TP path clips the
    # same way, and the traced path passes actual_len=T, vision_tokens=vision_tokens).
    if tokens.shape[1] > T:
        tokens = tokens[:, :T]
    return model.prefill_paged(tokens, page_table, valid_len=T, vision_tokens=vision_tokens)


def prime_decode_trace(generator, model, tokens, current_pos, page_table):
    """Capture the Generator decode trace WITHOUT corrupting GDN recurrent state.

    Used by the text_demo decode loop to capture the decode trace at the real post-prefill
    position/state. (The vLLM serving path instead uses the standard pos-0 warmup capture from the
    inherited WarmupForwardMixin, which is position-general because the model re-zeros GDN state at
    the start of every sequence via model._reset_gdn_state_for_new_sequence.)

    The stock Generator
    decode-trace capture runs the forward twice (a compile run + the capture run) on this first
    token before any real replay. For ordinary models that's harmless (re-writing the same paged KV
    slot is idempotent), but GDN's recurrent state is a running accumulation, so those extra passes
    advance it non-idempotently. Snapshot the DeltaNet state, drive one ``decode_forward`` with
    ``enable_trace=True`` (which performs the capture), then restore the snapshot — so the
    subsequent traced decode loop replays from the correct post-prefill state.

    Call once after prefill, before the decode loop. Inputs match ``decode_forward``:
    ``tokens`` [B,1], ``current_pos`` a [B] tensor, ``page_table`` host tensor.
    """
    saved = model._save_deltanet_states()
    generator.decode_forward(
        tokens, current_pos, page_table=page_table, kv_cache=None, enable_trace=True, read_from_device=True
    )
    model._restore_deltanet_states(saved, model.mesh_device)
