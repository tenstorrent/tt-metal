# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP full-model contract validation (Qwen3.5/3.6).

The vLLM-contract path (allocate_kv_caches + prefill_paged + decode contract) must
produce the SAME next tokens as the validated bespoke prefill_tp/decode_tp path
(B=1). Truncated to 8 layers (covers full-attn layers 3 & 7 + GDN) for speed. The
bespoke TP path (concat KV) is the oracle; the contract path (paged KV + Generator
decode chain incl. the rope_tp seam) must match it.

* ``test_model_tp_contract``            — short prompt: per-step logits PCC for the
  paged+decode chain, plus the masked fixed-bucket prefill path.
* ``test_model_tp_long_prefill``        — >2048 chunk-outer eager prefill (cross-chunk
  GDN/conv state carry).
* ``test_model_tp_long_prefill_traced`` — >2048 prefill via the captured chunk-outer
  trace (the path vLLM serves at long ISL).

Run:
  MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
    pytest -svq models/demos/blackhole/qwen36/tests/test_model_tp.py
"""
import math

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@torch.no_grad()
@parametrize_mesh_tp()
def test_model_tp_contract(mesh_device, reset_seeds, ensure_gc):
    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) contract path"
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=512, n_layers=8)
    args = model.args
    vocab = args.vocab_size
    T, N_DEC = 128, 3
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (T,)).tolist()
    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    # ---- bespoke oracle (concat KV): record per-step logits + the token fed at each step ----
    model.reset_tp()
    ref_logits = [model.prefill_tp(torch.tensor([prompt], dtype=torch.long), valid_len=T)]  # [vocab]
    ref = [int(torch.argmax(ref_logits[0]))]
    pos = T
    for _ in range(N_DEC):
        lg = model.decode_tp(ref[-1], pos)  # feed bespoke's own argmax
        ref_logits.append(lg)
        ref.append(int(torch.argmax(lg)))
        pos += 1
    logger.info(f"bespoke tokens: {ref}")

    # ---- vLLM contract path (paged KV), teacher-forced with the SAME tokens ----
    block_size = 64
    num_blocks = T // block_size + 8
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    kv_shape = (num_blocks, args.n_local_kv_heads, block_size, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    c_logits_dev = model.prefill_paged(torch.tensor([prompt], dtype=torch.long), page_table, valid_len=T)
    c_logits = [ttnn.to_torch(c_logits_dev, mesh_composer=comp0).reshape(-1, vocab)[0].float()]
    pos = T
    for i in range(N_DEC):
        dev = model.prepare_inputs_decode(
            torch.tensor([[ref[i]]], dtype=torch.int32), torch.tensor([pos], dtype=torch.int32), page_table
        )
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        c_logits.append(model.process_output_decode(out, 1).reshape(-1)[:vocab].float())
        pos += 1
    logger.info(f"contract argmax: {[int(torch.argmax(l)) for l in c_logits]}")

    # Per-step logits PCC with identical inputs isolates per-step correctness from
    # greedy-argmax-tie compounding (bf8/chunked-SDPA numerics differ at the 1e-4 level).
    pccs = []
    for i, (r, c) in enumerate(zip(ref_logits, c_logits)):
        _, pcc = comp_pcc(r.reshape(-1), c.reshape(-1), 0.99)
        pccs.append(float(pcc))
        logger.info(f"step {i} ({'prefill' if i == 0 else 'decode'}) logits PCC = {pcc}")
    assert all(p >= 0.99 for p in pccs), f"per-step logits PCC below 0.99: {pccs}"
    logger.info("PASSED: TP contract path matches bespoke path per-step (B=1)")

    # ---- masked fixed-bucket prefill (prefill_traced_chunked, the path vLLM serves) ----
    # T=128 has num_full==0, so prefill_traced_chunked routes entirely through the masked
    # bucket (no chunk trace). Its prefill logit must match the bespoke prefill too.
    mb_dev = model.prefill_traced_chunked(torch.tensor([prompt], dtype=torch.long), page_table, actual_len=T)
    mb_logits = ttnn.to_torch(mb_dev, mesh_composer=comp0).reshape(-1, vocab)[0].float()
    _, mb_pcc = comp_pcc(ref_logits[0].reshape(-1), mb_logits.reshape(-1), 0.99)
    logger.info(f"masked-bucket prefill logits PCC = {mb_pcc}")
    assert float(mb_pcc) >= 0.99, f"masked-bucket prefill PCC below 0.99: {mb_pcc}"
    logger.info("PASSED: TP masked-bucket prefill matches bespoke prefill (B=1)")


@torch.no_grad()
@parametrize_mesh_tp()
def test_model_tp_long_prefill(mesh_device, reset_seeds, ensure_gc):
    """TP long-prompt (>2048) prefill: the chunk-outer eager path must carry GDN recurrent +
    conv state across the chunk boundary to match the bespoke single-pass prefill. T=2304 =>
    one full 2048 chunk + a 256 tail, so this exercises the cross-chunk carry."""
    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) contract path"
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=4096, n_layers=8)
    args = model.args
    vocab = args.vocab_size
    T = 2304  # 1 full 2048 chunk + 256 tail
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (T,)).tolist()
    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    # bespoke oracle: single-pass prefill over the whole sequence.
    model.reset_tp()
    ref = model.prefill_tp(torch.tensor([prompt], dtype=torch.long), valid_len=T).reshape(-1).float()

    # contract path: chunk-outer eager prefill (full 2048 chunk carries state into the 256 tail).
    block_size = 64
    num_blocks = (T // block_size) + 8
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    kv_shape = (num_blocks, args.n_local_kv_heads, block_size, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    c_dev = model.prefill_traced_chunked(torch.tensor([prompt], dtype=torch.long), page_table, actual_len=T)
    c_logits = ttnn.to_torch(c_dev, mesh_composer=comp0).reshape(-1, vocab)[0].float()

    _, pcc = comp_pcc(ref.reshape(-1), c_logits.reshape(-1), 0.99)
    logger.info(f"long-prompt (T={T}) chunk-outer prefill logits PCC = {pcc}")
    assert float(pcc) >= 0.99, f"long-prompt chunk-outer prefill PCC below 0.99: {pcc}"
    logger.info("PASSED: TP chunk-outer eager prefill matches bespoke single-pass (B=1, >2048)")


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("T", [4096, 4352], ids=["exact_2chunks", "2chunks_plus_tail"])
def test_model_tp_long_prefill_traced(mesh_device, T, reset_seeds, ensure_gc):
    """TP long-prompt (>2048) prefill via the CAPTURED chunk-outer trace — the path vLLM serves
    at long ISL (and the fix for the eager path's >7872-token crash). Replays the per-chunk trace
    for each full 2048 chunk, carrying GDN recurrent/conv + paged-KV state in place, then the
    masked tail. Must match the bespoke single-pass prefill_tp. T=4096 (= 2*2048, no tail) hits the
    exact-multiple branch (_masked_bucket_logits_tp on the last chunk's hidden); T=4352 (= 2*2048 +
    256) exercises the multi-chunk replay + masked tail.

    The oracle prefill_tp runs BEFORE any trace is parked (it compiles per-length, so it must not
    run with the trace captured); the traced replay + masked buckets are all pre-warmed by
    capture_prefill_trace_chunked, so they never compile at request time."""
    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) contract path"
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=8192, n_layers=8)
    args = model.args
    vocab = args.vocab_size
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (T,)).tolist()
    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    # bespoke oracle: single-pass prefill over the whole sequence (no trace parked yet).
    model.reset_tp()
    ref = model.prefill_tp(torch.tensor([prompt], dtype=torch.long), valid_len=T).reshape(-1).float()

    # contract path: capture the chunk-outer trace (TP fork), then replay it per full chunk + tail.
    block_size = 64
    num_blocks = math.ceil(((T // block_size) + 8) / 32) * 32  # 32-aligned for the flexible SDPA
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    kv_shape = (num_blocks, args.n_local_kv_heads, block_size, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    model.capture_prefill_trace_chunked(mesh_device, page_table, chunk_size=2048)
    assert model._chunked_trace_id is not None, "TP chunk-outer trace was not captured"

    c_dev = model.prefill_traced_chunked(torch.tensor([prompt], dtype=torch.long), page_table, actual_len=T)
    c_logits = ttnn.to_torch(c_dev, mesh_composer=comp0).reshape(-1, vocab)[0].float()

    _, pcc = comp_pcc(ref.reshape(-1), c_logits.reshape(-1), 0.99)
    logger.info(f"traced chunk-outer prefill (T={T}) logits PCC = {pcc}")
    assert float(pcc) >= 0.99, f"traced chunk-outer prefill PCC below 0.99 at T={T}: {pcc}"
    logger.info(f"PASSED: TP traced chunk-outer prefill matches bespoke single-pass (B=1, T={T})")


@parametrize_mesh_tp()
def test_prefill_warmup_no_recompile(mesh_device, reset_seeds, ensure_gc):
    """After capture_prefill_trace_chunked parks the trace, a request-time masked-bucket prefill
    must reuse warmed programs only -- a post-park compile clobbers the trace (#48536).

    paged_fill_cache is keyed per fill width, so sweep EVERY width (1..32, across mask buckets
    128..2048) with program-cache misses disallowed and assert the cache does not grow. A sample
    would miss a regression that skips one width in _warmup_paged_fill_widths; here it fails
    op-named instead of hanging in serving. 8 layers suffices: warmed programs are layer-independent
    and layers 0..7 include the full-attention layers (3, 7) that own the fill path.
    """
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048, n_layers=8)
    args = model.args

    block_size = 64
    num_blocks = math.ceil((2048 // block_size + 8) / 32) * 32  # 32-aligned for the flexible SDPA
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    kv_shape = (num_blocks, args.n_local_kv_heads, block_size, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    # Warmup + park the per-chunk prefill trace.
    model.capture_prefill_trace_chunked(mesh_device, page_table, chunk_size=2048)

    # Request-time masked-bucket prefills must reuse warmed programs only.
    before = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        # One length per fill width 1..32: (width-1)*64 + 33 lands mid-block, so ceil(len/64) ==
        # width and the length stays masked below its auto-selected mask bucket.
        block_size = 64
        for width in range(1, 2048 // block_size + 1):
            actual_len = (width - 1) * block_size + 33
            model.prefill_masked_bucket(
                torch.zeros(1, actual_len, dtype=torch.int32), page_table, actual_len=actual_len
            )
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    after = mesh_device.num_program_cache_entries()

    logger.info(f"prefill-warmup no-recompile: program cache before={before} after={after} delta={after - before}")
    assert after == before, (
        f"{after - before} program(s) compiled after the trace was parked -> warmup missed a "
        f"fill-width-dependent program (add it to _warmup_paged_fill_widths); at request time this "
        f"clobbers the parked trace (hang)."
    )
    logger.info("PASSED: masked-bucket warmup pre-compiles every fill width (no post-capture recompile)")
