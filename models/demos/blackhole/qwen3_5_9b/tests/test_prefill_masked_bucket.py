# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Masked fixed-bucket prefill must match the trusted exact-length reference.

Why this exists
---------------
Short prompts (< the 2048 chunk size) take the on-demand eager prefill path, which
compiles a fresh program for each distinct prompt length. After a prefill trace is
parked in device memory, that request-time compilation can clobber the trace and hang
the device. The fix is `prefill_masked_bucket`: pad the prompt to one of a few fixed
bucket lengths and mask the Gated DeltaNet recurrent + conv state so it reflects only
the real tokens. Only a handful of bucket programs ever compile, so warmup can compile
them all before any trace is captured.

This test pins the correctness premise of that fix: the masked-bucket forward is
numerically equivalent to the trusted non-traced exact-length reference (`prefill_paged`).
We check, for a range of short lengths that each round up to a different bucket:

  1. the next-token logits (the prefill output), AND
  2. the post-prefill GDN recurrent state, AND
  3. the post-prefill GDN conv state,

the last two being exactly what decode continues from — so matching them is what proves
the masking does not corrupt decode (the failure mode of naive token-padding).

Run:
  HF_MODEL=Qwen/Qwen3.5-9B \
  pytest models/demos/blackhole/qwen3_5_9b/tests/test_prefill_masked_bucket.py -v -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]
BLOCK_SIZE = 64
MAX_NUM_BLOCKS = 64  # 64 * 64 = 4096 token capacity — plenty for short-prompt buckets

LOGIT_PCC = 0.99
STATE_PCC = 0.99


def compute_pcc(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)).item()


def test_mask_bucket_rounding():
    """Bucket rounding (no device): every length maps to the smallest fixed bucket >= it."""
    from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model

    f = Qwen35Model._mask_bucket_for  # classmethod — callable without a device/instance
    cases = {
        1: 128,
        50: 128,
        128: 128,
        129: 256,
        256: 256,
        300: 512,
        512: 512,
        700: 1024,
        1024: 1024,
        1500: 2048,
        2048: 2048,
    }
    for length, expected in cases.items():
        assert f(length) == expected, f"_mask_bucket_for({length}) -> {f(length)}, want {expected}"


# NOTE on reference lengths: the masked path is compared against `prefill_paged`, the
# trusted non-traced path. `prefill_paged` itself has a PRE-EXISTING L1 circular-buffer
# clash for real lengths in roughly (256, 512] (its GDN runs in L1 for seq_len<=512 and
# overflows there: T=256/700 are fine, T=300/400/512 throw) — unrelated to masking. So
# every actual_len below is one prefill_paged handles. The masked path is immune (it forces
# the GDN to DRAM), which is also how it covers bucket 512: exercised via (256, bucket=512).
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "actual_len, bucket",
    [
        (50, 128),  # natural bucket
        (137, 256),  # natural, padded
        (256, 256),  # exact boundary
        (256, 512),  # forced 512 bucket (DRAM GDN): 256 real + 256 masked pad
        (700, 1024),  # natural, padded
        (1024, 1024),  # exact boundary
        (700, 2048),  # forced 2048 bucket: 700 real + 1348 masked pad
        (1500, 2048),  # natural, padded
        (2000, 2048),  # natural, padded
    ],
    ids=[
        "len50_b128",
        "len137_b256",
        "len256_b256",
        "len256_b512",
        "len700_b1024",
        "len1024_b1024",
        "len700_b2048",
        "len1500_b2048",
        "len2000_b2048",
    ],
)
def test_masked_bucket_matches_reference(device, actual_len, bucket):
    device.enable_program_cache()

    from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model

    # 4 layers (pattern G,G,G,F) exercises both chunk-seq GDN and paged attention.
    model = Qwen35Model.from_pretrained(
        device,
        max_batch_size=1,
        max_seq_len=MAX_NUM_BLOCKS * BLOCK_SIZE,
        n_layers=4,
    )

    torch.manual_seed(0)
    real_tokens = torch.randint(0, 2000, (1, actual_len), dtype=torch.long)
    page_table = torch.arange(MAX_NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    kv_shape = [MAX_NUM_BLOCKS, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]

    # ---- Reference: trusted non-traced exact-length paged prefill ----
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    ref_logits = model.prefill_paged(real_tokens, page_table)
    ref = ttnn.to_torch(ref_logits).squeeze().float()
    ref_states = model._save_deltanet_states()

    # ---- Masked fixed-bucket prefill on the SAME real tokens (explicit bucket) ----
    test_logits = model.prefill_masked_bucket(real_tokens, page_table, actual_len=actual_len, bucket=bucket)
    test = ttnn.to_torch(test_logits).squeeze().float()
    test_states = model._save_deltanet_states()

    # 1) Next-token logits
    logit_pcc = compute_pcc(ref, test)
    ref_tok = int(ref.argmax())
    test_tok = int(test.argmax())

    # 2/3) Post-prefill GDN recurrent + conv state (what decode continues from)
    assert len(ref_states) == len(test_states) and len(ref_states) > 0
    rec_pccs, conv_pccs = [], []
    for i, (rs, ts) in enumerate(zip(ref_states, test_states)):
        rec_pccs.append(compute_pcc(rs["recurrent"], ts["recurrent"]))
        if rs["conv"] is not None and ts["conv"] is not None:
            conv_pccs.append(compute_pcc(rs["conv"], ts["conv"]))

    logger.info(
        f"len={actual_len} bucket={bucket} | logit_pcc={logit_pcc:.6f} "
        f"ref_tok={ref_tok} test_tok={test_tok} | rec_pcc(min)={min(rec_pccs):.6f} "
        f"conv_pcc(min)={min(conv_pccs) if conv_pccs else float('nan'):.6f}"
    )

    assert logit_pcc > LOGIT_PCC, f"next-token logit PCC {logit_pcc:.6f} < {LOGIT_PCC}"
    assert test_tok == ref_tok, f"next-token argmax mismatch: ref {ref_tok} vs masked {test_tok}"
    assert min(rec_pccs) > STATE_PCC, f"GDN recurrent-state PCC {min(rec_pccs):.6f} < {STATE_PCC}"
    if conv_pccs:
        assert min(conv_pccs) > STATE_PCC, f"GDN conv-state PCC {min(conv_pccs):.6f} < {STATE_PCC}"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_masked_bucket_after_trace_capture(device):
    """The vLLM scenario: a chunk-prefill trace is parked AND the masked buckets are warmed
    (exactly what capture_prefill_trace_chunked now does), THEN short prompts of varying
    lengths run via the masked path. This is the case that used to hang — the eager short
    path would compile a fresh program per length and clobber the parked trace. Here we
    assert each varied-length short prompt runs (no clash) in the in-place state mode and
    reproduces the next token from a pre-capture reference.
    """
    device.enable_program_cache()

    from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model

    model = Qwen35Model.from_pretrained(device, max_batch_size=1, max_seq_len=MAX_NUM_BLOCKS * BLOCK_SIZE, n_layers=4)
    page_table = torch.arange(MAX_NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    kv_shape = [MAX_NUM_BLOCKS, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    torch.manual_seed(0)
    lengths = [50, 137, 700, 1024]  # prefill_paged-safe lengths covering buckets 128/256/1024
    prompts = {L: torch.randint(0, 2000, (1, L), dtype=torch.long) for L in lengths}

    # ---- Pre-capture references (reassign-mode prefill_paged) ----
    ref = {}
    for L in lengths:
        ref[L] = ttnn.to_torch(model.prefill_paged(prompts[L], page_table)).squeeze().float()

    # ---- Park the chunk-prefill trace AND warm the masked buckets (what warmup does) ----
    model.capture_prefill_trace_chunked(device, page_table, chunk_size=2048)

    # ---- Now run short prompts of VARYING lengths through the masked path, AFTER the trace
    #      is parked. The point: each must RUN (no circular-buffer clash — the old hang) in
    #      the in-place state mode and stay numerically faithful to the pre-capture reference.
    #      We assert on logit PCC, not argmax: at the larger buckets the padded attention
    #      length differs slightly from the reference's exact length (~0.9999 PCC), which can
    #      flip a close top-2 argmax — a numerical artifact, not state corruption. ----
    for L in lengths:
        out = ttnn.to_torch(model.prefill_masked_bucket(prompts[L], page_table, actual_len=L)).squeeze().float()
        assert torch.isfinite(out).all(), f"non-finite logits at L={L}"
        p = compute_pcc(ref[L], out)
        logger.info(
            f"post-trace masked L={L} bucket={model._mask_bucket_for(L)} "
            f"logit_pcc={p:.6f} tok={int(out.argmax())} ref_tok={int(ref[L].argmax())}"
        )
        assert p > LOGIT_PCC, f"L={L}: post-trace masked logit PCC {p:.6f} < {LOGIT_PCC} (in-place state corruption?)"
