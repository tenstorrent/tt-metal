# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sparse_sdpa_msa (MSA block-sparse prefill) basic post-commit smoke.

This file keeps fast Blackhole coverage for the core contract: CPU reference sanity, registration, native
n_kv=1 execution, native GQA execution, sentinel masking, q dtype, and GQA runtime-T cache hits. Broader
shape/cache/determinism coverage lives in:
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sparse_sdpa_msa.py
"""

import pytest
import torch

import ttnn

from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.sdpa.sparse_sdpa_msa_test_utils import (
    BLK_KV,
    SENTINEL,
    dense_grouped_kv_attention,
    make_msa_inputs,
    pcc,
    run_op_msa_native,
    sparse_attention_ref_msa,
)

_D = 128
REFERENCE_PCC = 0.99999  # torch reference vs equivalent torch reference
DEVICE_PCC = 0.99  # TTNN device output vs torch reference
FP8_Q_DEVICE_PCC = 0.985


# CPU reference checks.


@pytest.mark.parametrize("H,n_kv", [(16, 1), (64, 4)], ids=["mha", "gqa"])
def test_golden_all_blocks_selected_equals_dense(H, n_kv):
    d, S, nblk = _D, 8, 4
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=nblk, d=d, causal=False, seed=1)
    scale = d**-0.5
    ref = sparse_attention_ref_msa(q, k, v, indices, scale)
    dense = dense_grouped_kv_attention(q, k, v, scale)
    assert pcc(ref, dense) > REFERENCE_PCC, f"sparse(all)=dense mismatch, pcc={pcc(ref, dense)}"


def test_golden_sentinel_tail_is_truncation():
    d, H, n_kv, S, nblk = _D, 16, 1, 6, 8
    T = nblk * BLK_KV
    scale = d**-0.5
    gen = torch.Generator().manual_seed(3)
    q = torch.randn(1, H, S, d, generator=gen)
    k = torch.randn(1, n_kv, T, d, generator=gen)
    v = torch.randn(1, n_kv, T, d, generator=gen)
    chosen = torch.tensor([1, 3, 5], dtype=torch.int32)
    full = torch.full((1, n_kv, S, 6), SENTINEL, dtype=torch.int32)
    tight = torch.empty((1, n_kv, S, 3), dtype=torch.int32)
    for s in range(S):
        full[0, 0, s, :3] = chosen
        tight[0, 0, s] = chosen
    out_pad = sparse_attention_ref_msa(q, k, v, full, scale)
    out_tight = sparse_attention_ref_msa(q, k, v, tight, scale)
    assert pcc(out_pad, out_tight) > REFERENCE_PCC


def test_op_is_registered():
    assert hasattr(ttnn.transformer, "sparse_sdpa_msa"), "ttnn.transformer.sparse_sdpa_msa not registered"


pytestmark = pytest.mark.use_module_device


@run_for_blackhole()
def test_msa_native_pcc_random_selection(device):
    d, H, S, topk, nblk = _D, 16, 8, 16, 16
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=1)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device)
    assert pcc(out, gold) > DEVICE_PCC


@run_for_blackhole()
def test_msa_native_gqa_pcc_random_selection(device):
    d, H, n_kv, S, topk, nblk = _D, 64, 4, 33, 16, 16
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk, d, causal=False, seed=21)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat8_b)
    assert pcc(out, gold) > DEVICE_PCC


@run_for_blackhole()
def test_msa_native_gqa_group_isolation(device):
    d, H, n_kv, S, topk, nblk = _D, 64, 4, 16, 16, 16
    T = nblk * BLK_KV
    q, k, _, indices = make_msa_inputs(H, n_kv, S, T, topk, d, causal=False, seed=23)
    v = torch.empty(1, n_kv, T, d)
    for g in range(n_kv):
        v[:, g].fill_(float(g + 1))
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device)
    assert torch.max(torch.abs(out.float() - gold.float())).item() < 0.1


@run_for_blackhole()
def test_msa_native_pcc_sentinel_tail(device):
    d, H, S, topk, nblk = _D, 16, 12, 16, 32
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=5)
    indices[0, 0, 0, 3:] = SENTINEL
    indices[0, 0, 1, 2:] = SENTINEL
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device)
    assert pcc(out, gold) > DEVICE_PCC


@run_for_blackhole()
@pytest.mark.parametrize("n_active", [1, 16], ids=["1chunk", "16chunk"])
def test_msa_native_fp32_dest_vs_bf16_dest_pcc_bisect(device, n_active):
    """Bisect the fp32-CB garbage: 1 active chunk (no online-softmax accumulation / salad) vs 16 chunks.
    topk must be 64B-aligned (mult of 16), so we force n_active via sentinels. If 1chunk is clean but
    16chunk is garbage, the fp32 packer L1-accumulate (running sum/out across chunks) is the bug."""
    d, H, S, topk, nblk = _D, 64, 8, 16, 32
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=11)
    if n_active < topk:
        indices[0, 0, :, n_active:] = SENTINEL  # keep only the first n_active blocks valid
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    fp32_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
        fp32_dest_acc_en=True, packer_l1_acc=False,
    )
    out_bf16 = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat16)
    out_fp32 = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat16, compute_kernel_config=fp32_cfg)
    print(f"\n[bisect topk={topk}] bf16={pcc(out_bf16, gold):.6f}  fp32={pcc(out_fp32, gold):.6f}")


@run_for_blackhole()
def test_msa_native_fp32_dest_vs_bf16_dest_pcc(device):
    """A/B the fp32-DEST accumulation path (fp32_dest_acc_en=True, enabled by the mm_init fix) vs the
    default bf16-DEST path, both against the fp32 reference. Many active chunks (topk=nblk, non-causal)
    so the online-softmax combine accumulates over the full block set — where DEST precision shows.

    fp32 DEST must not be WORSE than bf16 DEST; we log both PCCs + the delta to quantify the recovery
    (the op-level signal for the depth-compounding KV-PCC gap seen end-to-end)."""
    d, H, S, topk, nblk = _D, 64, 8, 32, 32  # 32 active chunks/query; bf16 q + bf16 k/v isolates DEST precision
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=11)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)

    fp32_dest_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    out_bf16 = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat16)
    out_fp32 = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat16, compute_kernel_config=fp32_dest_cfg)

    pcc_bf16 = pcc(out_bf16, gold)
    pcc_fp32 = pcc(out_fp32, gold)
    print(f"\n[fp32-DEST A/B] bf16-DEST PCC={pcc_bf16:.6f}  fp32-DEST PCC={pcc_fp32:.6f}  delta={pcc_fp32 - pcc_bf16:+.6f}")
    assert pcc_bf16 > DEVICE_PCC, f"baseline bf16-DEST regressed: {pcc_bf16}"
    assert pcc_fp32 >= pcc_bf16 - 1e-4, f"fp32-DEST worse than bf16-DEST: {pcc_fp32} < {pcc_bf16}"


@run_for_blackhole()
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16, ttnn.fp8_e4m3], ids=["q_bf16", "q_fp8"])
def test_msa_native_q_dtype(device, q_dtype):
    d, H, S, topk, nblk = _D, 32, 33, 16, 16
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk, d, causal=False, seed=11)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
    out = run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat8_b, q_dtype=q_dtype)
    thresh = DEVICE_PCC if q_dtype == ttnn.bfloat16 else FP8_Q_DEVICE_PCC
    assert pcc(out, gold) > thresh


def _rm(t, device, dtype):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _tile(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _msa_op(device, q, k_tt, v_tt, indices):
    out = ttnn.transformer.sparse_sdpa_msa(
        _rm(q.to(torch.bfloat16), device, ttnn.bfloat16),
        k_tt,
        v_tt,
        _rm(indices.to(torch.int32), device, ttnn.uint32),
        scale=_D**-0.5,
        block_size=BLK_KV,
    )
    return ttnn.to_torch(out)


@run_for_blackhole()
def test_msa_gqa_kv_len_no_recompile(device):
    H, n_kv, S, topk = 64, 4, 8, 16
    device.clear_program_cache()
    for T in (2048, 4096):
        q, k_full, v_full, indices = make_msa_inputs(H, n_kv, S, T, topk, _D, causal=False, seed=T)
        out = _msa_op(device, q, _tile(k_full, device), _tile(v_full, device), indices)
        gold = sparse_attention_ref_msa(q, k_full, v_full, indices, _D**-0.5)
        assert pcc(out, gold) > DEVICE_PCC, f"GQA T={T}"
    n = device.num_program_cache_entries()
    assert n == 1, f"changing GQA K/V length recompiled: {n} entries (expected 1)"


@run_for_blackhole()
def test_msa_bad_n_kv_rejected_on_hit(device, expect_error):
    H, S, T, topk = 32, 8, 2048, 16
    device.clear_program_cache()
    q, k_full, v_full, indices = make_msa_inputs(H, 1, S, T, topk, _D, causal=False, seed=41)
    _msa_op(device, q, _tile(k_full, device), _tile(v_full, device), indices)
    assert device.num_program_cache_entries() == 1

    q_bad, k_bad, v_bad, _ = make_msa_inputs(H, 2, S, T, topk, _D, causal=False, seed=42)
    with expect_error(RuntimeError, "indices must be"):
        _msa_op(device, q_bad, _tile(k_bad, device), _tile(v_bad, device), indices)


# ---- Causal (diagonal-block token-level mask) coverage ----
# Block selection alone is causal only at block granularity; a query's own (diagonal) block holds future
# tokens that must be masked. These exercise the chunk_start_idx path that enables the token-level mask.


@pytest.mark.parametrize("H,n_kv", [(16, 1), (64, 4)], ids=["mha", "gqa"])
def test_golden_all_blocks_causal_equals_dense_causal(H, n_kv):
    # With every causally-visible block selected, the masked reference must equal full causal attention.
    d, S, nblk = _D, 320, 4  # S spans blocks 0..2; the diagonal block is partially filled.
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=nblk, d=d, causal=True, seed=7)
    scale = d**-0.5
    causal_ref = sparse_attention_ref_msa(q, k, v, indices, scale, causal=True)
    dense_causal = dense_grouped_kv_attention(q, k, v, scale, causal=True)
    assert (
        pcc(causal_ref, dense_causal) > REFERENCE_PCC
    ), f"sparse(causal) != dense(causal), pcc={pcc(causal_ref, dense_causal)}"
    # And the legacy block-only reference is NOT causal -> the fixed reference genuinely tests something new.
    block_only = sparse_attention_ref_msa(q, k, v, indices, scale, causal=False)
    assert pcc(block_only, dense_causal) < 0.95, "block-only reference unexpectedly matches causal"


@run_for_blackhole()
def test_msa_native_causal_pcc(device):
    # S spans multiple blocks with the diagonal block partially filled, so the token-level mask matters.
    # nblk/topk=16 keeps TOPK*4 64B-aligned; causal selection still picks only blocks 0..local (rest sentinel).
    d, H, n_kv, S, nblk = _D, 64, 4, 320, 16
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=nblk, d=d, causal=True, seed=31)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5, causal=True)
    # chunk_start_idx=0 enables the token-level diagonal-block causal mask.
    out = run_op_msa_native(q, k, v, indices, device, chunk_start_idx=0)
    assert pcc(out, gold) > DEVICE_PCC, f"causal MSA pcc={pcc(out, gold)}"


@run_for_blackhole()
def test_msa_native_causal_fp8_q_rejected(device, expect_error):
    # fp8 q is silently inaccurate with the causal mask: fp8-specific, and not fp32-DEST-related -- bf16 q with
    # fp32_dest_acc_en forced on passes -- but the root cause is not yet identified. The op rejects the combo
    # rather than returning wrong scores; bf16 q is the supported causal path.
    d, H, n_kv, S, nblk = _D, 64, 4, 320, 16
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=nblk, d=d, causal=True, seed=31)
    with expect_error(RuntimeError, "fp8_e4m3 q is not supported"):
        run_op_msa_native(q, k, v, indices, device, kv_dtype=ttnn.bfloat8_b, q_dtype=ttnn.fp8_e4m3, chunk_start_idx=0)


@run_for_blackhole()
def test_msa_native_causal_vs_noncausal_differs(device):
    # Sanity that the mask is actually applied: with the diagonal block partially filled, the causal output
    # must differ from the block-only (no chunk_start_idx) output.
    d, H, S, nblk = _D, 16, 320, 16  # nblk/topk=16 -> TOPK*4 is 64B-aligned
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, 1, S, T, topk=nblk, d=d, causal=True, seed=9)
    out_causal = run_op_msa_native(q, k, v, indices, device, chunk_start_idx=0)
    out_block_only = run_op_msa_native(q, k, v, indices, device)  # chunk_start_idx=None -> legacy path
    assert pcc(out_causal, out_block_only) < 0.999, "causal mask had no effect vs block-only"


@run_for_blackhole()
def test_msa_replay_dump(device):
    """Replay the captured worst L31 galaxy shard on a single device to reproduce the diag-mask divergence
    and DPRINT the masked scores. Gated on M3_REPLAY_DUMP=1. Run with:
      TT_METAL_DPRINT_CORES=0,0 TT_METAL_DPRINT_RISCVS=TR0 M3_REPLAY_DUMP=1 pytest ... -k test_msa_replay_dump -s
    """
    import os

    if os.getenv("M3_REPLAY_DUMP") != "1":
        pytest.skip("set M3_REPLAY_DUMP=1 to run the captured-shard replay")
    blob = torch.load("/tmp/m3_opdump/L31_worst.pt")
    q, k, v = blob["q"].float(), blob["k"].float(), blob["v"].float()  # [1,16,640,128], [1,1,5120,128]x2
    block_ids = blob["block_ids"].to(torch.int64)  # [1,1,640,16], 0xFFFFFFFF sentinel
    cs, scale = int(blob["chunk_start_idx"]), float(blob["scale"])
    Hql = q.shape[1]
    print(f"\n[replay] shard r={blob['rank_r']} c={blob['col_c']} chunk_start={cs} saved_pcc={blob['shard_pcc']:.5f}")

    q_t = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_t = ttnn.from_torch(k.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_t = ttnn.from_torch(v.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # int64 block_ids -> uint32 bit pattern (sentinel 0xFFFFFFFF)
    bids_t = ttnn.from_torch((block_ids & 0xFFFFFFFF).to(torch.int64), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out = ttnn.transformer.sparse_sdpa_msa(q_t, k_t, v_t, bids_t, scale=scale, block_size=BLK_KV, chunk_start_idx=cs, cluster_axis=None)
    dev = ttnn.to_torch(out)[:, :Hql].float()

    # host fp32 causal reference on the SAME inputs (indices as -1 sentinel for the ref)
    idx_ref = block_ids.clone()
    idx_ref[idx_ref == 0xFFFFFFFF] = -1
    ref = sparse_attention_ref_msa(q, k, v, idx_ref.to(torch.int32), scale, causal=True, chunk_start_idx=cs)
    print(f"[replay] device vs host-fp32 causal PCC = {pcc(dev, ref):.5f}  (galaxy shard was {blob['shard_pcc']:.5f})")
    import collections as _collections

    _S = dev.shape[2]
    _buckets = _collections.defaultdict(list)
    for _s in range(_S):
        _off = (cs + _s) % BLK_KV
        _buckets[_off // 16].append(pcc(dev[:, :, _s : _s + 1, :], ref[:, :, _s : _s + 1, :]))
    print("[replay] per-query PCC by diagonal-block offset bucket:")
    for _b in sorted(_buckets):
        _v = _buckets[_b]
        print(f"  off[{_b*16:>3}-{_b*16+15:>3}] n={len(_v):>3} mean={sum(_v)/len(_v):.4f} min={min(_v):.4f}")
    print("[replay] exact-offset PCC (127 writes NO -inf but still runs the diag_chunk machinery):")
    for _target in (0, 32, 64, 96, 127):
        _vals = [pcc(dev[:, :, s : s + 1, :], ref[:, :, s : s + 1, :]) for s in range(_S) if (cs + s) % BLK_KV == _target]
        if _vals:
            print(f"  offset=={_target:>3}: n={len(_vals)} mean={sum(_vals)/len(_vals):.4f} min={min(_vals):.4f}")
    # characterize the offset-0 query (s where (cs+s)%128==0): magnitude vs direction, per head
    _s0 = next(s for s in range(_S) if (cs + s) % BLK_KV == 0)
    _d0, _r0 = dev[0, :, _s0, :], ref[0, :, _s0, :]  # [Hq, hd]
    print(f"[replay] offset-0 query s={_s0} (pos={cs+_s0}):")
    print(f"  dev norm={_d0.norm():.3f} ref norm={_r0.norm():.3f} ratio={_d0.norm()/_r0.norm():.3f}")
    print(f"  head0 dev[:6]={[round(x,3) for x in _d0[0,:6].tolist()]}")
    print(f"  head0 ref[:6]={[round(x,3) for x in _r0[0,:6].tolist()]}")
    _hp = [pcc(_d0[h:h+1], _r0[h:h+1]) for h in range(_d0.shape[0])]
    print(f"  per-head PCC: min={min(_hp):.3f} max={max(_hp):.3f} mean={sum(_hp)/len(_hp):.3f}")
