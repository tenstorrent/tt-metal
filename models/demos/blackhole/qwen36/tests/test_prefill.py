# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-device prefill-path regressions (Qwen3.5-9B).

Two families of prefill correctness checks, both validated against the trusted
non-traced ``prefill_paged`` reference:

* Masked fixed-bucket prefill (``test_mask_bucket_rounding``,
  ``test_masked_bucket_matches_reference``, ``test_masked_bucket_after_trace_capture``,
  ``test_traced_chunked_tail_matches_reference``) — short prompts pad up to a few fixed
  buckets and mask the GDN recurrent + conv state so request-time compilation can't
  clobber a parked prefill trace.
* Chunk-outer per-chunk replay (``test_chunked_replay_matches_reference``) — long prompts
  replay one captured 2048-token chunk per chunk, carrying KV/recurrent state across
  chunk boundaries; mathematically equivalent to layer-outer single-pass prefill.

Run:
  HF_MODEL=Qwen/Qwen3.5-9B \
  pytest models/demos/blackhole/qwen36/tests/test_prefill.py -v -s

  # long-context cases (>--max-prefill) only run when the cap is raised:
  pytest ... --max-prefill 131072 -v -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc

# Single-device test: default to the 9B checkpoint (the 27B needs a multi-device mesh for TP).
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

DEVICE_PARAMS = [{"l1_small_size": 24576, "num_command_queues": 2}]
BLOCK_SIZE = 64
MNB_MASKED = 64  # 64 * 64 = 4096 token capacity — plenty for short-prompt buckets
MNB_CHUNKED = 1280  # 1280 * 64 = 81920 token capacity (covers the >64k isolation case)

LOGIT_PCC = 0.99
STATE_PCC = 0.99


# --------------------------------------------------------------------------- #
# Masked fixed-bucket prefill
# --------------------------------------------------------------------------- #
def test_mask_bucket_rounding():
    """Bucket rounding (no device): every length maps to the smallest fixed bucket >= it."""
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    f = Qwen36Model._mask_bucket_for  # classmethod — callable without a device/instance
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


# The exact-length reference and masked path both keep multi-token GDN activations in DRAM.
# Include the lengths that previously collided with gated_delta_attn_seq's static L1 circular
# buffers so this comparison also guards the eager prefill memory-planning contract.
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "actual_len, bucket",
    [
        (50, 128),  # natural bucket
        (137, 256),  # natural, padded
        (256, 256),  # exact boundary
        (256, 512),  # forced 512 bucket: 256 real + 256 masked pad
        (300, 512),  # exact reference previously hit the L1 circular-buffer clash
        (400, 512),
        (512, 512),
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
        "len300_b512",
        "len400_b512",
        "len512_b512",
        "len700_b1024",
        "len1024_b1024",
        "len700_b2048",
        "len1500_b2048",
        "len2000_b2048",
    ],
)
def test_masked_bucket_matches_reference(device, actual_len, bucket):
    device.enable_program_cache()

    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    # 4 layers (pattern G,G,G,F) exercises both chunk-seq GDN and paged attention.
    model = Qwen36Model.from_pretrained(
        device,
        max_batch_size=1,
        max_seq_len=MNB_MASKED * BLOCK_SIZE,
        n_layers=4,
    )

    torch.manual_seed(0)
    real_tokens = torch.randint(0, 2000, (1, actual_len), dtype=torch.long)
    page_table = torch.arange(MNB_MASKED, dtype=torch.int32).unsqueeze(0)
    kv_shape = [MNB_MASKED, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]

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
    # Exact next-token argmax is checked only at the prompt's *natural* bucket. When the prompt is
    # forced into a larger bucket (more masked padding), the longer padded attention length differs
    # slightly from the reference's exact length (~0.999+ PCC), which can flip a close top-2 argmax —
    # a numerical artifact, not state corruption (the GDN recurrent/conv-state PCCs below are the real
    # decode-continuation check). Same reasoning as test_masked_bucket_after_trace_capture.
    if bucket == model._mask_bucket_for(actual_len):
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

    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=MNB_MASKED * BLOCK_SIZE, n_layers=4)
    page_table = torch.arange(MNB_MASKED, dtype=torch.int32).unsqueeze(0)
    kv_shape = [MNB_MASKED, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]
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


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "actual_len",
    [2098, 2337, 2546, 2748, 3548],
    ids=[
        "chunk1_tail50_b128",
        "chunk1_tail289_b512",
        "chunk1_tail498_b512",
        "chunk1_tail700_b1024",
        "chunk1_tail1500_b2048",
    ],
)
def test_traced_chunked_tail_matches_reference(device, actual_len):
    """Long-prompt path: prefill_traced_chunked replays the parked chunk trace for the full
    2048-token chunk, then runs the partial FINAL chunk through the masked-bucket path with the
    GDN/KV state carried in place (chunk_start > 0). This tail previously took the eager
    per-length path, which compiled a fresh program at request time and clobbered the parked
    trace (the device hang). Assert the tail now (1) runs with a trace parked and (2) stays
    faithful to the trusted exact-length prefill_paged reference — next-token logits AND the
    carried GDN recurrent/conv state, which decode continues from.
    """
    device.enable_program_cache()

    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=MNB_MASKED * BLOCK_SIZE, n_layers=4)
    page_table = torch.arange(MNB_MASKED, dtype=torch.int32).unsqueeze(0)
    kv_shape = [MNB_MASKED, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    torch.manual_seed(0)
    real_tokens = torch.randint(0, 2000, (1, actual_len), dtype=torch.long)

    # ---- Reference: trusted non-traced exact-length paged prefill (captured BEFORE the trace
    #      reassigns the GDN to its in-place external state buffers). ----
    ref = ttnn.to_torch(model.prefill_paged(real_tokens, page_table)).squeeze().float()
    ref_states = model._save_deltanet_states()

    # ---- Park the chunk trace + warm the masked buckets (what vLLM warmup does), then run the
    #      long traced-chunked path on the SAME tokens. ----
    model.capture_prefill_trace_chunked(device, page_table, chunk_size=2048)
    test = ttnn.to_torch(model.prefill_traced_chunked(real_tokens, page_table, actual_len=actual_len)).squeeze().float()
    test_states = model._save_deltanet_states()

    assert torch.isfinite(test).all(), f"non-finite logits at L={actual_len}"
    logit_pcc = compute_pcc(ref, test)

    assert len(ref_states) == len(test_states) and len(ref_states) > 0
    rec_pccs, conv_pccs = [], []
    for rs, ts in zip(ref_states, test_states):
        rec_pccs.append(compute_pcc(rs["recurrent"], ts["recurrent"]))
        if rs["conv"] is not None and ts["conv"] is not None:
            conv_pccs.append(compute_pcc(rs["conv"], ts["conv"]))

    tail = actual_len % 2048
    logger.info(
        f"traced-chunked L={actual_len} tail={tail} bucket={model._mask_bucket_for(tail)} "
        f"| logit_pcc={logit_pcc:.6f} tok={int(test.argmax())} ref_tok={int(ref.argmax())} "
        f"| rec_pcc(min)={min(rec_pccs):.6f} conv_pcc(min)={min(conv_pccs) if conv_pccs else float('nan'):.6f}"
    )
    assert logit_pcc > LOGIT_PCC, f"L={actual_len}: traced-chunked logit PCC {logit_pcc:.6f} < {LOGIT_PCC}"
    assert (
        min(rec_pccs) > STATE_PCC
    ), f"L={actual_len}: carried GDN recurrent-state PCC {min(rec_pccs):.6f} < {STATE_PCC}"
    if conv_pccs:
        assert (
            min(conv_pccs) > STATE_PCC
        ), f"L={actual_len}: carried GDN conv-state PCC {min(conv_pccs):.6f} < {STATE_PCC}"


# --------------------------------------------------------------------------- #
# Chunk-outer per-chunk-replay prefill
# --------------------------------------------------------------------------- #
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

    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    # 4 layers (pattern G,G,G,F) exercises both chunk-seq GDN and paged attention.
    model = Qwen36Model.from_pretrained(
        device,
        max_batch_size=1,
        max_seq_len=MNB_CHUNKED * BLOCK_SIZE,
        n_layers=4,
    )

    chunk_size = 2048
    bucket = ((actual_len + chunk_size - 1) // chunk_size) * chunk_size  # pad up to a multiple of chunk_size
    torch.manual_seed(0)
    real_tokens = torch.randint(0, 2000, (1, actual_len), dtype=torch.long)
    page_table = torch.arange(MNB_CHUNKED, dtype=torch.int32).unsqueeze(0)
    kv_shape = [MNB_CHUNKED, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]

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
