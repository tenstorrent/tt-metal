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
def test_msa_native_causal_multiband_pcc(device):
    # Query heads per KV group padded to > dst_size tile-rows force the compute kernel to process the heads
    # in more than one DEST band (qg>0). The token-level causal mask must land on every band, not just the
    # first. bf16 DEST holds 8 tiles, so H_logical must exceed 256: H=288, n_kv=1 -> Sqt=9 -> 3 bands.
    d, H, n_kv, S, nblk = _D, 288, 1, 320, 16
    T = nblk * BLK_KV
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=nblk, d=d, causal=True, seed=31)
    gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5, causal=True)
    out = run_op_msa_native(q, k, v, indices, device, chunk_start_idx=0)
    assert pcc(out, gold) > DEVICE_PCC, f"multi-band causal MSA pcc={pcc(out, gold)}"


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


# ---- block-cyclic remap: block-ids are NATURAL positions but the K/V cache is stored block-cyclic across SP
# ---- shards. sp is DERIVED from the mesh (block_cyclic_sp_axis = the striped axis). On ONE device the mesh is
# ---- 1x1 so sp=1 and the block-granular remap reduces to identity (shard=0, BC_SLAB_STRIDE_GAP=0 => phys=block)
# ---- — enough to smoke the whole BC path (API, chunk_local cross-check, BC_ENABLE reader+writer branch, T
# ---- hashing). The sp>1 PERMUTATION arithmetic needs a real SP mesh; that multi-device coverage is deferred to
# ---- a nightly multidevice file (a mesh_device fixture can't share this single-device file), mirroring
# ---- test_sparse_sdpa's block-cyclic coverage. ----
@run_for_blackhole()
def test_msa_native_block_cyclic_sp1_identity(device):
    """sp=1 (from the 1x1 device-mesh): block-cyclic == natural, so the op must reproduce the natural golden
    while exercising the BC_ENABLE path, across cache sizes T. T is hashed for this path (BC_SHARD_STRIDE_GAP is
    a compile-time define), so each distinct T is a DISTINCT program — asserted below."""
    H, n_kv, S, d, topk = 32, 1, 128, _D, 16  # S == chunk_local (mult of block_size); TOPK*4 must be 64B-aligned
    Ts = (2048, 4096)  # nblk = T/128 >= topk; distinct T -> distinct program (T hashed for the BC path)
    device.clear_program_cache()
    for T in Ts:
        nblk = T // BLK_KV
        q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk=min(topk, nblk), d=d, causal=False, seed=T)
        gold = sparse_attention_ref_msa(q, k, v, indices, d**-0.5)
        out = run_op_msa_native(q, k, v, indices, device, block_cyclic_sp_axis=0, block_cyclic_chunk_local=S)
        p = pcc(out, gold)
        assert p >= DEVICE_PCC, f"PCC {p:.5f} (sp=1 identity block-cyclic, T={T})"
    n = device.num_program_cache_entries()
    assert n == len(Ts), f"block-cyclic should hash cache size T: got {n} entries (expected {len(Ts)})"


@run_for_blackhole()
def test_msa_native_block_cyclic_sp1_bit_exact(device):
    """The remap is pure addressing, not arithmetic: at sp=1 invP is the identity, so the block-cyclic path reads
    the exact same tiles in the same order as the plain path and must produce a BIT-IDENTICAL result (stronger
    than PCC — proves the BC_ENABLE branch and the extra phys_block computation perturb nothing)."""
    H, n_kv, S, d, topk, T = 32, 1, 128, _D, 16, 2048
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk, d, causal=False, seed=7)
    plain = run_op_msa_native(q, k, v, indices, device)
    bc = run_op_msa_native(q, k, v, indices, device, block_cyclic_sp_axis=0, block_cyclic_chunk_local=S)
    assert torch.equal(
        plain, bc
    ), f"block-cyclic sp=1 must be bit-identical to non-BC; max|delta|={(plain - bc).abs().max().item()}"


@run_for_blackhole()
def test_msa_native_block_cyclic_chunk_local_rejected(device, expect_error):
    """The chunk_local cross-check rejects a value that is neither q_isl nor tp*q_isl. On a single device
    sp=1/tp=1, so the only legal chunk_local is q_isl (= S); any other value must raise before dispatch."""
    H, n_kv, S, d, topk = 32, 1, 128, _D, 2
    T = 256
    q, k, v, indices = make_msa_inputs(H, n_kv, S, T, topk, d, causal=False, seed=1)
    with expect_error(RuntimeError, "block_cyclic_chunk_local"):
        run_op_msa_native(q, k, v, indices, device, block_cyclic_sp_axis=0, block_cyclic_chunk_local=S - 32)
