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
from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold, parametrize_mesh_tp
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
    masked tail. T=4096 (= 2*2048, no tail) hits the exact-multiple branch (_masked_bucket_logits_tp
    on the last chunk's hidden); T=4352 (= 2*2048 + 256) exercises the multi-chunk replay + masked
    tail.

    Reference = the EAGER chunk-outer prefill (prefill_traced_chunked before any trace is parked),
    which test_model_tp_long_prefill validates as equivalent to the single-pass prefill_tp oracle at
    T<=2304. We use it here instead of prefill_tp because prefill_tp does the whole sequence in one
    GDN pass, and the batched GDN chunk-prefill's L1 footprint scales with T/128 sub-chunks and
    overflows at T>=4096 (production never single-passes >2048). Both the eager reference and the
    traced replay re-zero GDN state at the start, so running them back-to-back on one model is safe."""
    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) contract path"
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=8192, n_layers=8)
    args = model.args
    vocab = args.vocab_size
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (T,)).tolist()
    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    block_size = 64
    num_blocks = math.ceil(((T // block_size) + 8) / 32) * 32  # 32-aligned for the flexible SDPA
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    kv_shape = (num_blocks, args.n_local_kv_heads, block_size, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    # Reference: eager chunk-outer prefill (no trace parked yet -> _prefill_chunked_eager_tp).
    assert model._chunked_trace_id is None, "eager reference must run before the trace is captured"
    ref_dev = model.prefill_traced_chunked(torch.tensor([prompt], dtype=torch.long), page_table, actual_len=T)
    ref = ttnn.to_torch(ref_dev, mesh_composer=comp0).reshape(-1, vocab)[0].float()

    # Contract path: capture the chunk-outer trace, then replay it per full chunk + tail.
    model.capture_prefill_trace_chunked(mesh_device, page_table, chunk_size=2048)
    assert model._chunked_trace_id is not None, "TP chunk-outer trace was not captured"

    c_dev = model.prefill_traced_chunked(torch.tensor([prompt], dtype=torch.long), page_table, actual_len=T)
    c_logits = ttnn.to_torch(c_dev, mesh_composer=comp0).reshape(-1, vocab)[0].float()

    _, pcc = comp_pcc(ref.reshape(-1), c_logits.reshape(-1), 0.99)
    logger.info(f"traced vs eager chunk-outer prefill (T={T}) logits PCC = {pcc}")
    assert float(pcc) >= 0.99, f"traced chunk-outer prefill PCC below 0.99 at T={T}: {pcc}"
    logger.info(f"PASSED: TP traced chunk-outer prefill matches eager chunk-outer (B=1, T={T})")


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


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("B", [8, 32], ids=["B8", "B32"])
def test_model_tp_decode_batched(mesh_device, B, reset_seeds, ensure_gc):
    """Batched per-user decode contract (TP): higher-batch serving acceptance test.

    B users with distinct prompt lengths share one paged KV cache + batched GDN state
    (prefill_paged_peruser); batched decode steps at diverging positions must match B
    independent B=1 bespoke runs (the concat-path oracle). PCC is compared per user, not
    flattened, so a single contaminated row is not hidden. Sub-0.99 is expected SDPA-decode
    batch-variance, not a bug.
    """
    import gc

    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) contract path"
    N_DEC = 3
    torch.manual_seed(0)

    # ---- B=1 bespoke oracle (concat KV); collect per-user logits + argmax chains, then free
    # before allocating the batched model so only one model is resident at a time. ----
    model1 = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=512, n_layers=8)
    vocab = model1.args.vocab_size
    prompt_lens = [128 + 32 * (u % 4) for u in range(B)]  # {128,160,192,224}, distinct lengths
    prompts = [torch.randint(0, vocab, (prompt_lens[u],)).tolist() for u in range(B)]

    ref_logits = []  # ref_logits[u] = [prefill_logit, dec0, dec1, ...]
    fed = []  # fed[u] = bespoke argmax chain, teacher-forced into the batched path
    for u in range(B):
        model1.reset_tp()
        lg0 = model1.prefill_tp(torch.tensor([prompts[u]], dtype=torch.long), valid_len=prompt_lens[u])
        chain, toks, pos = [lg0.float()], [int(torch.argmax(lg0))], prompt_lens[u]
        for _ in range(N_DEC):
            lg = model1.decode_tp(toks[-1], pos)
            chain.append(lg.float())
            toks.append(int(torch.argmax(lg)))
            pos += 1
        ref_logits.append(chain)
        fed.append(toks)
    del model1
    gc.collect()

    # ---- batched path: per-user paged prefill + batched per-user-position decode ----
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=512, n_layers=8)
    args = model.args
    block_size, bpu = 64, 8  # 8 blocks/user covers up to 512 tokens (>> max prompt + N_DEC)
    num_blocks = B * bpu
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])  # [B, bpu]
    kv_shape = (num_blocks, args.n_local_kv_heads, block_size, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)
    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    token_list = [torch.tensor([prompts[u]], dtype=torch.long) for u in range(B)]
    pf_logits = model.prefill_paged_peruser(token_list, page_table, valid_lens=prompt_lens)
    pf_torch = [ttnn.to_torch(pf_logits[u], mesh_composer=comp0).reshape(-1, vocab)[0].float() for u in range(B)]

    dec_torch = [[] for _ in range(B)]
    pos = list(prompt_lens)
    for step in range(N_DEC):
        tokens_step = torch.tensor([[fed[u][step]] for u in range(B)], dtype=torch.int32)  # [B, 1]
        pos_t = torch.tensor(pos, dtype=torch.int32)  # [B] — per-user diverging positions
        dev = model.prepare_inputs_decode(tokens_step, pos_t, page_table)
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        logits_step = model.process_output_decode(out, B)  # [B, 1, vocab]
        for u in range(B):
            dec_torch[u].append(logits_step[u, 0, :vocab].float())
        pos = [p + 1 for p in pos]

    # ---- per-user, per-step logits PCC (paged-batched vs concat-B1 oracle) ----
    thr = 0.97
    worst = (1.0, -1, -1)
    for u in range(B):
        steps = [pf_torch[u]] + dec_torch[u]
        for s, (r, c) in enumerate(zip(ref_logits[u], steps)):
            _, pcc = comp_pcc(r.reshape(-1), c.reshape(-1), thr)
            if float(pcc) < worst[0]:
                worst = (float(pcc), u, s)
            assert float(pcc) >= thr, f"user {u} step {s} (len={prompt_lens[u]}) logits PCC {pcc} < {thr}"
    logger.info(
        f"PASSED: batched per-user decode (B={B}) worst logits PCC = {worst[0]:.5f} @ user{worst[1]} step{worst[2]}"
    )


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("B", [8], ids=["B8"])
def test_model_tp_prefill_paged_slots(mesh_device, B, reset_seeds, ensure_gc):
    """vLLM continuous-batching prefill contract (TP): per-slot prefill acceptance test.

    Mirrors test_model_tp_decode_batched but drives the online-serving path the vLLM wrapper uses:
    the batched prefill warmup (capture_prefill_trace_chunked(capture_chunk_trace=False): masked
    buckets warmed against a B=1 GDN scratch, no chunk trace parked), then prefill_paged_slots
    (each user prefilled B=1 into its empty_slots[u] via write_slot, preserving the other rows).
    Batched decode at diverging positions must match B independent B=1 bespoke runs, per user.
    """
    import gc

    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) contract path"
    N_DEC = 3
    torch.manual_seed(0)

    # ---- B=1 bespoke oracle (concat KV) ----
    model1 = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=512, n_layers=8)
    vocab = model1.args.vocab_size
    prompt_lens = [128 + 32 * (u % 4) for u in range(B)]  # {128,160,192,224}, distinct lengths
    prompts = [torch.randint(0, vocab, (prompt_lens[u],)).tolist() for u in range(B)]
    ref_logits, fed = [], []
    for u in range(B):
        model1.reset_tp()
        lg0 = model1.prefill_tp(torch.tensor([prompts[u]], dtype=torch.long), valid_len=prompt_lens[u])
        chain, toks, pos = [lg0.float()], [int(torch.argmax(lg0))], prompt_lens[u]
        for _ in range(N_DEC):
            lg = model1.decode_tp(toks[-1], pos)
            chain.append(lg.float())
            toks.append(int(torch.argmax(lg)))
            pos += 1
        ref_logits.append(chain)
        fed.append(toks)
    del model1
    gc.collect()

    # ---- vLLM path: batched warmup + per-slot prefill + batched decode ----
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=512, n_layers=8)
    args = model.args
    block_size, bpu = 64, 8
    num_blocks = B * bpu
    page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)])  # [B, bpu]
    kv_shape = (num_blocks, args.n_local_kv_heads, block_size, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)

    # Batched prefill warmup exactly as qwen36_vllm.warmup_model_prefill: bind a B=1 scratch, warm
    # the masked buckets (no chunk trace), restore the batched decode buffers.
    warmup_pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    prev = model._alloc_gdn_scratch_b1()
    try:
        model.capture_prefill_trace_chunked(mesh_device, warmup_pt, chunk_size=2048, capture_chunk_trace=False)
    finally:
        model._restore_gdn_batched(prev)

    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    token_list = [torch.tensor([prompts[u]], dtype=torch.long) for u in range(B)]
    # Per-slot prefill into slots 0..B-1 (the default empty_slots order the plugin uses for a fresh batch).
    pf_host = model.prefill_paged_slots(token_list, page_table, list(range(B)), valid_lens=prompt_lens)
    pf_torch = [pf_host[u].reshape(-1, vocab)[0].float() for u in range(B)]

    dec_torch = [[] for _ in range(B)]
    pos = list(prompt_lens)
    for step in range(N_DEC):
        tokens_step = torch.tensor([[fed[u][step]] for u in range(B)], dtype=torch.int32)  # [B, 1]
        pos_t = torch.tensor(pos, dtype=torch.int32)
        dev = model.prepare_inputs_decode(tokens_step, pos_t, page_table)
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        logits_step = model.process_output_decode(out, B)  # [B, 1, vocab]
        for u in range(B):
            dec_torch[u].append(logits_step[u, 0, :vocab].float())
        pos = [p + 1 for p in pos]

    thr = 0.97
    worst = (1.0, -1, -1)
    for u in range(B):
        steps = [pf_torch[u]] + dec_torch[u]
        for s, (r, c) in enumerate(zip(ref_logits[u], steps)):
            _, pcc = comp_pcc(r.reshape(-1), c.reshape(-1), thr)
            if float(pcc) < worst[0]:
                worst = (float(pcc), u, s)
            assert float(pcc) >= thr, f"user {u} step {s} (len={prompt_lens[u]}) logits PCC {pcc} < {thr}"
    logger.info(
        f"PASSED: vLLM per-slot prefill + batched decode (B={B}) worst logits PCC = "
        f"{worst[0]:.5f} @ user{worst[1]} step{worst[2]}"
    )


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("B", [8, 32], ids=["B8", "B32"])
def test_model_tp_prefill_traced_bucket(mesh_device, B, reset_seeds, ensure_gc, request):
    """Traced batched short-prompt prefill (TP): traced-bucket-prefill acceptance test.

    Validates the traced bucket prefill (capture_prefill_trace_bucket +
    prefill_traced_bucket_batched) against the eager path it replaces (prefill_paged_peruser).
    B users are prefilled both ways on the same batched model (free + re-allocate between runs);
    per-user prefill logits, post-prefill GDN recurrent state, and a few decode steps must all
    match, so the traced path never regresses vs eager.
    """
    import gc

    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) traced bucket prefill path"
    N_DEC = 2
    torch.manual_seed(0)

    # All prompts are exactly the bucket length (128) — the only length the traced path serves
    # (full bucket, valid_len=None, numerically identical to eager valid_len=128 by GDN full-chunk
    # equivalence). Sub-bucket prompts would corrupt the GDN decode state through the recurrence,
    # so the caller routes them to eager prefill instead; there is no sub-bucket traced case.
    # Distinct content per user still exercises per-user page-table routing.
    bucket = 128
    prompt_lens = [bucket] * B
    block_size = 64
    bpu = max(4, -(-(bucket + N_DEC + 4) // block_size))  # covers prompt + decode, >=4

    def _build_model():
        model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=B, max_seq_len=bpu * block_size, n_layers=8)
        args = model.args
        num_blocks = B * bpu
        page_table = torch.stack(
            [torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(B)]
        )  # [B, bpu]
        kv_shape = (num_blocks, args.n_local_kv_heads, block_size, args.head_dim)
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)
        return model, args, page_table

    vocab_ref = None
    prompts = None

    # ---- eager reference: prefill_paged_peruser (the path being replaced) ----
    model, args, page_table = _build_model()
    vocab_ref = args.vocab_size
    prompts = [torch.randint(0, vocab_ref, (prompt_lens[u],)).tolist() for u in range(B)]
    token_list = [torch.tensor([prompts[u]], dtype=torch.long) for u in range(B)]
    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    eager_pf = model.prefill_paged_peruser(token_list, page_table, valid_lens=prompt_lens)
    eager_pf_torch = [
        ttnn.to_torch(eager_pf[u], mesh_composer=comp0).reshape(-1, vocab_ref)[0].float() for u in range(B)
    ]
    # Snapshot the eager-assembled batched GDN recurrent state per user (the decode seed).
    eager_gdn = [
        ttnn.to_torch(la.attention.rec_state, mesh_composer=comp0).float()
        for la in model.layers
        if not la.is_full_attention
    ]
    # A few eager decode steps for a per-user decode-correctness baseline. Both paths decode from the
    # SAME greedy token sequence (fed_ref, eager's argmax): when a user's top-2 prefill logits are
    # near-tied, the tiny eager-vs-traced numerical delta can flip the argmax, so feeding each path
    # its own argmax would decode divergent continuations and make the decode-PCC comparison
    # meaningless (a false failure). Production greedily decodes its own argmax per request and both
    # paths are individually correct; sharing the seed here isolates decode-compute equivalence.
    eager_dec = [[] for _ in range(B)]
    pos = list(prompt_lens)
    fed_ref = [[int(torch.argmax(eager_pf_torch[u]))] for u in range(B)]
    for step in range(N_DEC):
        tokens_step = torch.tensor([[fed_ref[u][step]] for u in range(B)], dtype=torch.int32)
        pos_t = torch.tensor(pos, dtype=torch.int32)
        dev = model.prepare_inputs_decode(tokens_step, pos_t, page_table)
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        logits_step = model.process_output_decode(out, B)
        for u in range(B):
            eager_dec[u].append(logits_step[u, 0, :vocab_ref].float())
            fed_ref[u].append(int(torch.argmax(logits_step[u, 0, :vocab_ref])))
        pos = [p + 1 for p in pos]
    model.free_kv_caches()
    del model
    gc.collect()

    # ---- traced bucket prefill (the new path) ----
    model, args, page_table = _build_model()
    token_list = [torch.tensor([prompts[u]], dtype=torch.long) for u in range(B)]
    model.capture_prefill_trace_bucket(mesh_device, page_table[0:1].contiguous(), bucket=bucket)
    traced_pf = model.prefill_traced_bucket_batched(token_list, page_table, valid_lens=prompt_lens)
    ttnn.synchronize_device(mesh_device)
    traced_pf_torch = [
        ttnn.to_torch(traced_pf[u], mesh_composer=comp0).reshape(-1, vocab_ref)[0].float() for u in range(B)
    ]
    traced_gdn = [
        ttnn.to_torch(la.attention.rec_state, mesh_composer=comp0).float()
        for la in model.layers
        if not la.is_full_attention
    ]
    # release_prefill_trace_bucket restores the batched GDN bindings the decode trace reads.
    model.release_prefill_trace_bucket()
    # Decode the SAME token sequence eager used (fed_ref) — see the note at the eager decode loop.
    traced_dec = [[] for _ in range(B)]
    pos = list(prompt_lens)
    for step in range(N_DEC):
        tokens_step = torch.tensor([[fed_ref[u][step]] for u in range(B)], dtype=torch.int32)
        pos_t = torch.tensor(pos, dtype=torch.int32)
        dev = model.prepare_inputs_decode(tokens_step, pos_t, page_table)
        out, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        logits_step = model.process_output_decode(out, B)
        for u in range(B):
            traced_dec[u].append(logits_step[u, 0, :vocab_ref].float())
        pos = [p + 1 for p in pos]
    model.free_kv_caches()
    del model
    gc.collect()

    # ---- per-user prefill logits + GDN state + decode PCC (traced vs eager) ----
    thr = get_pcc_threshold(request, default=0.99)
    worst = (1.0, -1, "")
    for u in range(B):
        _, pcc_pf = comp_pcc(eager_pf_torch[u].reshape(-1), traced_pf_torch[u].reshape(-1), thr)
        if float(pcc_pf) < worst[0]:
            worst = (float(pcc_pf), u, "prefill")
        assert float(pcc_pf) >= thr, f"user {u} prefill logits PCC {pcc_pf} < {thr}"
        for s in range(N_DEC):
            _, pcc_d = comp_pcc(eager_dec[u][s].reshape(-1), traced_dec[u][s].reshape(-1), thr)
            if float(pcc_d) < worst[0]:
                worst = (float(pcc_d), u, f"decode{s}")
            assert float(pcc_d) >= thr, f"user {u} decode{s} logits PCC {pcc_d} < {thr}"
    for li, (er, tr) in enumerate(zip(eager_gdn, traced_gdn)):
        for u in range(B):
            _, pcc_g = comp_pcc(er[u].reshape(-1), tr[u].reshape(-1), thr)
            if float(pcc_g) < worst[0]:
                worst = (float(pcc_g), u, f"gdn{li}")
            assert float(pcc_g) >= thr, f"gdn layer {li} user {u} rec-state PCC {pcc_g} < {thr}"
    logger.info(
        f"PASSED: traced bucket prefill (B={B}) matches eager per-user prefill; "
        f"worst PCC = {worst[0]:.5f} @ user{worst[1]} {worst[2]}"
    )


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("B", [8, 32], ids=["B8", "B32"])
@pytest.mark.parametrize("seqlen", [2048, 4096, "mixed"], ids=["isl2048", "isl4096", "mixed"])
def test_model_tp_prefill_chunked_batched(mesh_device, B, seqlen, reset_seeds, ensure_gc, request):
    """Batched per-user long prefill (TP): chunked-batched-prefill acceptance test.

    Validates prefill_chunked_peruser (batched long-prompt path) against the proven B=1
    chunk-outer path (prefill_traced_chunked, itself validated vs the bespoke oracle by
    test_model_tp_long_prefill), isolating the batch scratch-swap + state assembly +
    batched-decode routing as the only delta.

    Regimes: isl2048 (1 full chunk), isl4096 (2 full chunks), and "mixed" (lengths cycled
    over {128, 1024, 2048, 4096}, exercising short-via-masked-bucket + long-via-chunked in one
    batch with diverging decode positions). Per-user prefill logits, post-prefill GDN recurrent
    state, and a few decode steps must all match the B=1 reference.
    """
    import gc

    nd = mesh_device.get_num_devices()
    assert nd > 1, "this test exercises the TP (num_devices>1) batched chunked prefill path"
    N_DEC = 2
    block_size = 64
    torch.manual_seed(0)

    if seqlen == "mixed":
        cyc = [128, 1024, 2048, 4096]
        prompt_lens = [cyc[u % len(cyc)] for u in range(B)]
    else:
        prompt_lens = [int(seqlen)] * B
    max_len = max(prompt_lens)
    # bpu covers the longest prompt + decode; %8 keeps the flexible-SDPA page-table stick 32B-aligned.
    bpu = max(8, -(-(max_len + N_DEC + 4) // block_size))
    bpu = ((bpu + 7) // 8) * 8

    def _build(batch):
        model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=batch, max_seq_len=bpu * block_size, n_layers=8)
        args = model.args
        num_blocks = batch * bpu
        page_table = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(batch)])
        kv_shape = (num_blocks, args.n_local_kv_heads, block_size, args.head_dim)
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=batch)
        return model, args, page_table

    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    # ---- per-user B=1 reference: prefill_traced_chunked (eager) + B=1 decode ----
    # Shared B=1 model; users reuse blocks [0, bpu) sequentially. prefill_traced_chunked resets
    # the GDN state at sequence start, so users don't bleed into each other.
    omodel, args, opt = _build(1)
    vocab = args.vocab_size
    prompts = [torch.randint(0, vocab, (prompt_lens[u],)).tolist() for u in range(B)]
    oracle_pf = []
    oracle_rec = []  # per user: list over GDN layers of device-0 rec_state shard [Nv,Dk,Dv]
    oracle_dec = [[] for _ in range(B)]
    for u in range(B):
        toks = torch.tensor([prompts[u]], dtype=torch.long)
        lg = omodel.prefill_traced_chunked(toks, opt, actual_len=prompt_lens[u])
        ttnn.synchronize_device(mesh_device)
        oracle_pf.append(ttnn.to_torch(lg, mesh_composer=comp0).reshape(-1, vocab)[0].float())
        oracle_rec.append(
            [
                ttnn.to_torch(la.attention.rec_state, mesh_composer=comp0)[0].float()  # device-0 shard
                for la in omodel.layers
                if not la.is_full_attention
            ]
        )
        pos = prompt_lens[u]
        fed = int(torch.argmax(oracle_pf[u]))
        for s in range(N_DEC):
            dev = omodel.prepare_inputs_decode(
                torch.tensor([[fed]], dtype=torch.int32), torch.tensor([pos], dtype=torch.int32), opt
            )
            out, _ = omodel.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
            ls = omodel.process_output_decode(out, 1)
            oracle_dec[u].append(ls[0, 0, :vocab].float())
            fed = int(torch.argmax(ls[0, 0, :vocab]))
            pos += 1
    n_gdn = len(oracle_rec[0])
    omodel.free_kv_caches()
    del omodel
    gc.collect()

    # ---- batched path: prefill_chunked_peruser + batched decode at diverging positions ----
    bmodel, args, bpt = _build(B)
    token_list = [torch.tensor([prompts[u]], dtype=torch.long) for u in range(B)]
    bpf = bmodel.prefill_chunked_peruser(token_list, bpt, valid_lens=prompt_lens)
    ttnn.synchronize_device(mesh_device)
    batched_pf = [ttnn.to_torch(bpf[u], mesh_composer=comp0).reshape(-1, vocab)[0].float() for u in range(B)]
    # Assembled batched rec_state per GDN layer: [N*B, Nv, Dk, Dv] (device-major). device-0 user u = row u.
    batched_rec = [
        ttnn.to_torch(la.attention.rec_state, mesh_composer=comp0).float()
        for la in bmodel.layers
        if not la.is_full_attention
    ]
    batched_dec = [[] for _ in range(B)]
    pos = list(prompt_lens)
    fed = [int(torch.argmax(batched_pf[u])) for u in range(B)]
    for s in range(N_DEC):
        tokens_step = torch.tensor([[fed[u]] for u in range(B)], dtype=torch.int32)
        pos_t = torch.tensor(pos, dtype=torch.int32)
        dev = bmodel.prepare_inputs_decode(tokens_step, pos_t, bpt)
        out, _ = bmodel.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
        ls = bmodel.process_output_decode(out, B)
        for u in range(B):
            batched_dec[u].append(ls[u, 0, :vocab].float())
            fed[u] = int(torch.argmax(ls[u, 0, :vocab]))
        pos = [p + 1 for p in pos]
    bmodel.free_kv_caches()
    del bmodel
    gc.collect()

    # ---- per-user prefill logits + GDN state + decode PCC (batched vs B=1 reference) ----
    thr = get_pcc_threshold(request, default=0.97)
    worst = (1.0, -1, "")
    for u in range(B):
        _, pcc_pf = comp_pcc(oracle_pf[u].reshape(-1), batched_pf[u].reshape(-1), thr)
        if float(pcc_pf) < worst[0]:
            worst = (float(pcc_pf), u, "prefill")
        assert float(pcc_pf) >= thr, f"user {u} (len {prompt_lens[u]}) prefill logits PCC {pcc_pf} < {thr}"
        for s in range(N_DEC):
            _, pcc_d = comp_pcc(oracle_dec[u][s].reshape(-1), batched_dec[u][s].reshape(-1), thr)
            if float(pcc_d) < worst[0]:
                worst = (float(pcc_d), u, f"decode{s}")
            assert float(pcc_d) >= thr, f"user {u} decode{s} logits PCC {pcc_d} < {thr}"
    for li in range(n_gdn):
        for u in range(B):
            _, pcc_g = comp_pcc(oracle_rec[u][li].reshape(-1), batched_rec[li][u].reshape(-1), thr)
            if float(pcc_g) < worst[0]:
                worst = (float(pcc_g), u, f"gdn{li}")
            assert float(pcc_g) >= thr, f"gdn layer {li} user {u} rec-state PCC {pcc_g} < {thr}"
    logger.info(
        f"PASSED: batched chunked prefill (B={B}, {seqlen}) matches B=1 chunk-outer reference; "
        f"worst PCC = {worst[0]:.5f} @ user{worst[1]} {worst[2]}"
    )
