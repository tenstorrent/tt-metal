# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TDD: chunk-outer per-chunk-replay prefill must match the non-traced reference.

Background: the chunk-seq GDN kernel (now the only prefill path) is correct only at
<=16 sub-chunks (2048 tokens) per call, and a single whole-sequence trace of the
chunk-seq prefill exceeds tt-metal's 4 GiB uint32 trace-size ceiling at 128K. The
fix is to capture ONE 2048-token chunk's full-layer forward and replay it per chunk,
DMA-advancing the per-chunk inputs (token slice, chunk_start_idx, page table, RoPE).

For a causal transformer, processing layer-outer over the whole sequence and
chunk-outer (each chunk through all layers, KV/recurrent state carried across chunks)
are mathematically equivalent. This test pins that equivalence: the new chunk-outer
replay path (prefill_traced_chunked) must match the trusted non-traced layer-outer
reference (prefill_paged), with chunk-seq enabled, across multiple chunks.

Run:
  pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_trace_chunked.py -v -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn

# HF_MODEL (hub name or local path) is the single source of truth.
CHECKPOINT_DIR = os.environ.get("HF_MODEL", "/local/ttuser/atupe/Qwen9b")
os.environ.setdefault("HF_MODEL", CHECKPOINT_DIR)
DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]
BLOCK_SIZE = 64
MAX_NUM_BLOCKS = 1280  # 1280 * 64 = 81920 token capacity (covers the >64k isolation case)


def compute_pcc(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)).item()


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "actual_len",
    [6144, 5000, 73728],
    ids=["full_3chunks", "partial_last_chunk", "nopad_past_64k"],
)
def test_chunked_replay_matches_reference(device, actual_len):
    """Chunk-outer per-chunk-replay prefill == non-traced layer-outer reference (chunk-seq).

    `full_3chunks`: actual_len is a multiple of chunk_size (no padding).
    `partial_last_chunk`: actual_len falls inside the last chunk — the replay processes a
    padded bucket but must extract the next-token logit at position actual_len-1, unaffected
    by the padding (causal).
    """
    device.enable_program_cache()

    from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model

    # 4 layers (pattern G,G,G,F) exercises both chunk-seq GDN and paged attention.
    model = Qwen35Model.from_pretrained(
        device,
        max_batch_size=1,
        max_seq_len=MAX_NUM_BLOCKS * BLOCK_SIZE,
        n_layers=4,
    )

    chunk_size = 2048
    bucket = ((actual_len + chunk_size - 1) // chunk_size) * chunk_size  # pad up to a multiple of chunk_size
    torch.manual_seed(0)
    real_tokens = torch.randint(0, 2000, (1, actual_len), dtype=torch.long)
    page_table = torch.arange(MAX_NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    kv_shape = [MAX_NUM_BLOCKS, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]

    # ---- Reference: non-traced, layer-outer paged prefill on the REAL (unpadded) tokens ----
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    ref_logits = model.prefill_paged(real_tokens, page_table)
    ref = ttnn.to_torch(ref_logits).squeeze().float()
    assert not torch.isnan(ref).any(), "reference logits contain NaN"
    ref_tok = int(ref.argmax())

    # ---- New: chunk-outer per-chunk-replay traced prefill on the PADDED bucket ----
    pad = bucket - actual_len
    padded = torch.cat([real_tokens, real_tokens[:, -1:].expand(1, pad)], dim=1) if pad else real_tokens
    model.capture_prefill_trace_chunked(device, page_table, chunk_size=chunk_size)
    test_logits = model.prefill_traced_chunked(padded, page_table, actual_len=actual_len)
    test = ttnn.to_torch(test_logits).squeeze().float()
    assert not torch.isnan(test).any(), "traced-chunked logits contain NaN"
    test_tok = int(test.argmax())

    pcc = compute_pcc(ref, test)
    logger.info(f"actual_len={actual_len} bucket={bucket} ref_tok={ref_tok} test_tok={test_tok} pcc={pcc:.6f}")
    assert test_tok == ref_tok, f"next-token argmax mismatch: ref={ref_tok} test={test_tok} (pcc={pcc:.4f})"
    assert pcc > 0.99, f"prefill-logits PCC {pcc:.6f} < 0.99"
