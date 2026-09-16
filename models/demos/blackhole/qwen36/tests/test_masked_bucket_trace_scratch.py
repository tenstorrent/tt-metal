# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device acceptance tests for the traced masked-bucket prefill (QWEN36_PREFILL_BUCKET_TRACE).

The masked bucket serves every prompt shorter than one 2048 chunk and the tail of every long
prompt. Tracing it must be numerically indistinguishable from the eager masked path it replaces
(the traced body always applies the GDN masks, which is exact: an all-ones mask is the identity),
must not compile anything at request time (a post-park compile clobbers the parked traces -> hang,
#48536), and must carry GDN state correctly out of the chunk trace into the tail.

Structure (the gate is read in Qwen36Model.__init__, so it is set before the model is built):
the EAGER references are computed first, while ``model._mb_traces`` is still empty; then
``capture_prefill_trace_chunked`` parks the chunk trace and captures the bucket traces; the same
calls then dispatch to ``_replay_prefill_bucket_trace_tp``.

Run (P300x2 / P150x4 mesh):
  MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B QWEN36_PREFILL_BUCKET_TRACE=128 \
    pytest -svq models/demos/blackhole/qwen36/tests/test_masked_bucket_trace_scratch.py
"""
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.masked_bucket_trace import parse_bucket_trace_gate
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

# Bucket 128 ships first; the orchestrator can widen the list from the environment.
os.environ.setdefault("QWEN36_PREFILL_BUCKET_TRACE", "128")
GATE = parse_bucket_trace_gate(os.environ["QWEN36_PREFILL_BUCKET_TRACE"], Qwen36Model._PREFILL_MASK_BUCKETS)
BLOCK_SIZE = 64
PCC_MIN = 0.999


def parametrize_mesh_tp(max_tp=8, trace_bytes=1073741824):
    """test_factory.parametrize_mesh_tp plus an explicit 1 GiB trace region (the b8 model spec's
    size): these tests park the chunk trace AND one trace per gated bucket, so the budget is the
    thing under test — an overflow surfaces as a TT_FATAL at end_trace_capture. Mirrors
    test_decode_bucketing._parametrize_traced."""
    from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

    shape = _resolve_mesh_shape(max_tp)

    def decorator(fn):
        fn = pytest.mark.parametrize(
            "device_params",
            [
                {
                    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                    "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE,
                    "trace_region_size": trace_bytes,
                }
            ],
            indirect=True,
        )(fn)
        fn = pytest.mark.parametrize("mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True)(
            fn
        )
        return fn

    return decorator


def _blocks_32_aligned(tokens, spare=8):
    return math.ceil((tokens // BLOCK_SIZE + spare) / 32) * 32


def _logits(dev_tensor, mesh_device, vocab):
    """First mesh replica's logits row as float32 torch [vocab]."""
    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    return ttnn.to_torch(dev_tensor, mesh_composer=comp0).reshape(-1, vocab)[0].float()


@torch.no_grad()
@parametrize_mesh_tp()
def test_masked_bucket_trace_matches_eager(mesh_device, reset_seeds, ensure_gc):
    """Bucket 128, actual_len in (1, 33, 100, 128): the traced replay must match the eager masked
    prefill in logits (PCC >= 0.999, same top-1) and must not compile a single program after the
    traces are parked."""
    assert mesh_device.get_num_devices() > 1, "the traced masked bucket is the TP path"
    assert 128 in GATE, f"set QWEN36_PREFILL_BUCKET_TRACE to include 128 (got {GATE})"

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048, n_layers=8)
    assert model._mb_trace_buckets == GATE, f"gate not picked up: {model._mb_trace_buckets} != {GATE}"
    args = model.args
    vocab = args.vocab_size

    num_blocks = _blocks_32_aligned(2048)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    # One SPARE block past the page table: the fixed-width fill page table points its pad entries
    # at the last physical block, which no request may own (model._pad_kv_block).
    kv_shape = (num_blocks + 1, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    assert model._pad_kv_block == num_blocks, "scratch block must be the spare block past the page table"

    lens = [n for n in (1, 33, 100, 128) if n <= 128]
    torch.manual_seed(0)
    prompts = {n: torch.randint(0, vocab, (1, n), dtype=torch.long) for n in lens}

    # ---- eager references (no trace parked yet -> the int-valid_len masked path) ----
    assert not model._mb_traces, "eager reference must run before the bucket traces are captured"
    ref = {}
    for n in lens:
        dev = model.prefill_masked_bucket(prompts[n], page_table, actual_len=n, bucket=128)
        ref[n] = _logits(dev, mesh_device, vocab)

    # ---- capture the chunk trace + the gated bucket traces ----
    model.capture_prefill_trace_chunked(mesh_device, page_table, chunk_size=2048)
    assert 128 in model._mb_traces, "bucket-128 masked prefill trace was not captured"
    logger.info(f"captured bucket traces: {sorted(model._mb_traces)}")
    used = Qwen36Model._trace_region_bytes(mesh_device)
    if used is not None:  # Phase B budget input: chunk + gated buckets must leave room for decode
        logger.info(f"TRACE region after chunk + bucket traces: {used / (1 << 20):.1f} MiB of 1024 MiB")

    # ---- traced replays ----
    pccs = {}
    for n in lens:
        t0 = time.perf_counter()
        dev = model.prefill_masked_bucket(prompts[n], page_table, actual_len=n, bucket=128)
        dt_ms = (time.perf_counter() - t0) * 1e3
        got = _logits(dev, mesh_device, vocab)
        assert torch.isfinite(got).all(), f"non-finite traced logits at actual_len={n}"
        _, pcc = comp_pcc(ref[n].reshape(-1), got.reshape(-1), PCC_MIN)
        pccs[n] = float(pcc)
        top_ref, top_got = int(torch.argmax(ref[n])), int(torch.argmax(got))
        logger.info(
            f"bucket=128 actual_len={n}: PCC={pcc} top1 eager={top_ref} traced={top_got} "
            f"traced masked call {dt_ms:.1f} ms (8 layers)"
        )
        assert top_ref == top_got, f"top-1 token differs at actual_len={n}: {top_ref} vs {top_got}"
    assert all(p >= PCC_MIN for p in pccs.values()), f"traced masked bucket PCC below {PCC_MIN}: {pccs}"

    # ---- no post-park compiles: a program-cache miss here is the second-request hang (#48536) ----
    before = mesh_device.num_program_cache_entries()
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        for n in lens:
            model.prefill_masked_bucket(prompts[n], page_table, actual_len=n, bucket=128)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    after = mesh_device.num_program_cache_entries()
    assert after == before, f"{after - before} program(s) compiled during a traced bucket replay"
    logger.info("PASSED: traced masked bucket 128 matches eager and compiles nothing at request time")


@torch.no_grad()
@parametrize_mesh_tp()
def test_masked_bucket_trace_long_prompt_tail(mesh_device, reset_seeds, ensure_gc):
    """Long prompt: two 2048 chunk-trace replays followed by a TRACED masked tail at
    chunk_start=4096. Exercises cos/sin + chunk_start + full-page-table refresh at a non-zero
    position and the GDN state handoff from the chunk trace into the bucket trace."""
    assert mesh_device.get_num_devices() > 1, "the traced masked bucket is the TP path"
    tail = 256 if 256 in GATE else 128
    assert tail in GATE, f"set QWEN36_PREFILL_BUCKET_TRACE to include 128 or 256 (got {GATE})"
    T = 2 * 2048 + tail

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=8192, n_layers=8)
    args = model.args
    vocab = args.vocab_size
    torch.manual_seed(0)
    prompt = torch.randint(0, vocab, (1, T), dtype=torch.long)

    num_blocks = _blocks_32_aligned(T)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
    kv_shape = (num_blocks + 1, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    # Reference: the fully eager chunk-outer prefill (no traces parked yet), the same oracle
    # test_model_tp_long_prefill_traced uses.
    assert model._chunked_trace_id is None and not model._mb_traces
    ref = _logits(model.prefill_traced_chunked(prompt, page_table, actual_len=T), mesh_device, vocab)

    model.capture_prefill_trace_chunked(mesh_device, page_table, chunk_size=2048)
    assert model._chunked_trace_id is not None and tail in model._mb_traces

    t0 = time.perf_counter()
    got = _logits(model.prefill_traced_chunked(prompt, page_table, actual_len=T), mesh_device, vocab)
    dt_ms = (time.perf_counter() - t0) * 1e3

    _, pcc = comp_pcc(ref.reshape(-1), got.reshape(-1), 0.99)
    logger.info(f"T={T} (2 chunk replays + traced {tail}-token tail): PCC={pcc}, whole prefill {dt_ms:.1f} ms")
    assert float(pcc) >= 0.99, f"traced tail prefill PCC below 0.99 at T={T}: {pcc}"
    logger.info("PASSED: traced masked tail matches the eager chunk-outer prefill")


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize(
    "owned, actual_len",
    [([5, 6], 100), ([5], 1)],
    ids=["two_real_blocks", "one_real_block_plus_scratch"],
)
def test_masked_bucket_trace_kv_integrity(mesh_device, owned, actual_len, reset_seeds, ensure_gc):
    """The fixed-width fill page table writes K/V for the WHOLE bucket, including pad rows. Only
    the request's own blocks (and, when its page-table row is zero-padded, the scratch block) may
    change; in particular block 0 must never be written by another request's padding."""
    assert mesh_device.get_num_devices() > 1, "the traced masked bucket is the TP path"
    assert 128 in GATE, f"set QWEN36_PREFILL_BUCKET_TRACE to include 128 (got {GATE})"

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048, n_layers=8)
    args = model.args
    num_blocks = _blocks_32_aligned(2048)
    kv_shape = (num_blocks + 1, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim)
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    # A vLLM-shaped row: the request owns only `owned`, everything past it is zero padding (which
    # fill_pt_row must redirect to the scratch block rather than to block 0).
    row = torch.zeros(1, num_blocks, dtype=torch.int32)
    row[0, : len(owned)] = torch.tensor(owned, dtype=torch.int32)
    full_pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)  # for warmup/capture
    model.capture_prefill_trace_chunked(mesh_device, full_pt, chunk_size=2048)
    assert 128 in model._mb_traces

    comp0 = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    k_cache = model._paged_kv_caches[0][0]
    before = ttnn.to_torch(k_cache, mesh_composer=comp0).clone()
    torch.manual_seed(0)
    model.prefill_masked_bucket(
        torch.randint(0, args.vocab_size, (1, actual_len), dtype=torch.long),
        row,
        actual_len=actual_len,
        bucket=128,
    )
    after = ttnn.to_torch(k_cache, mesh_composer=comp0)

    nblk = before.shape[0] // mesh_device.get_num_devices()
    changed = sorted({int(b) % nblk for b in (before != after).any(dim=-1).any(dim=-1).any(dim=-1).nonzero()})
    allowed = set(owned) | {model._pad_kv_block}
    logger.info(f"actual_len={actual_len}: KV blocks changed {changed}, allowed {sorted(allowed)}")
    assert 0 not in changed, "block 0 was written by a padded fill (page-table pad entry aliased it)"
    assert set(changed) <= allowed, f"unexpected KV blocks written: {sorted(set(changed) - allowed)}"
    logger.info("PASSED: traced masked bucket writes only the request's blocks and the scratch block")
