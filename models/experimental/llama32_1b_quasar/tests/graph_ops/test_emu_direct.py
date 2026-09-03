# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# HAND-AUTHORED (not generated). Emulator smoke tests, with torch references, for
# uplifted ops that the graph-capture harness cannot exercise directly:
#
#   * tilize — reached only inside input construction (from_torch(layout=TILE,
#     device=...) -> ttnn.tilize), which graph_case.build_tensor routes around on
#     Quasar because the mainline op was legacy (Gen1-only CreateKernel). Now that
#     tilize is a Metal 2.0 factory (#54805) with a Gen2 hw_config, this checks the
#     on-device path so the host-tilize workaround can be retired.
#   * scaled_dot_product_attention (experimental.quasar fork, #54468) — the captured
#     prefill case is [1,32,1024,64] with an 8x8 grid; graph_case also has no SDPA
#     golden. Scaled to one head, seq 128, grid 1x1, compared against torch.
#   * paged_scaled_dot_product_attention_decode (experimental.quasar fork, #54249) —
#     decode geometry (32 q heads / 8 kv heads / paged bf8 cache) on a 1x1 grid,
#     compared against torch. Mirrors tests/ops but sized for the 2-node emulator.
#
# K/V are bfloat16 here, not the captured bfloat8_b: Quasar has no Bfp8_b
# (tt::is_data_format_supported -> is_supported_quasar lists MxFp8 instead), and
# ValidateProgramSpec rejects any DFB carrying it before the program is built.
#
#   pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_emu_direct.py -m emulator
# ---------------------------------------------------------------------------
"""Emulator-sized direct tests (with references) for tilize and the SDPA forks."""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

pytestmark = pytest.mark.emulator

HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5


def _to_device_tiled(x, mesh, dtype=ttnn.bfloat16, memory_config=None):
    """Host-tilize, upload interleaved, then (optionally) relay out to a sharded config.

    Same shape as graph_case.build_tensor: the sharded placement goes through the
    experimental.quasar.to_memory_config fork on Quasar and mainline elsewhere.
    """
    tt = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh)
    )
    tt = ttnn.to_device(tt, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if memory_config is not None and memory_config != ttnn.DRAM_MEMORY_CONFIG:
        to_mc = ttnn.experimental.quasar.to_memory_config if G._is_quasar(mesh) else ttnn.to_memory_config
        tt = to_mc(tt, memory_config)
    return tt


def _hs_l1_1core(shard_shape):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            list(shard_shape),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


# =============================================================================
# tilize (mainline, Metal 2.0 default factory)
# =============================================================================

_TILIZE_SHAPES = [(1, 1, 32, 64), (1, 1, 64, 128)]


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _TILIZE_SHAPES, ids=["x".join(map(str, s)) for s in _TILIZE_SHAPES])
def test_tilize_on_device(ttnn_mesh_device, reset_seeds, shape):
    """Explicit ttnn.tilize of a row-major interleaved DRAM tensor already on device."""
    mesh = ttnn_mesh_device
    x = U.torch_rand(shape)
    rm = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh)
    )
    rm = ttnn.to_device(rm, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.tilize(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert out.layout == ttnn.TILE_LAYOUT
    U.assert_pcc(x, out, pcc=0.9999, mesh_device=mesh)


@U.with_default_mesh()
@pytest.mark.parametrize("shape", _TILIZE_SHAPES[:1], ids=["x".join(map(str, s)) for s in _TILIZE_SHAPES[:1]])
def test_from_torch_tile_on_device(ttnn_mesh_device, reset_seeds, shape):
    """from_torch(layout=TILE, device=...) — the exact call graph_case.build_tensor avoids on Quasar."""
    mesh = ttnn_mesh_device
    x = U.torch_rand(shape)
    out = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )
    assert out.layout == ttnn.TILE_LAYOUT
    U.assert_pcc(x, out, pcc=0.9999, mesh_device=mesh)


# =============================================================================
# SDPA prefill (experimental.quasar fork)
# =============================================================================

_SDPA_SEQ = 128
_SDPA_CHUNK = 64


def _torch_sdpa_causal(q, k, v, scale):
    n_rep = q.shape[1] // k.shape[1]
    k_rep = k.repeat_interleave(n_rep, dim=1)
    v_rep = v.repeat_interleave(n_rep, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), k_rep.float(), v_rep.float(), is_causal=True, scale=scale
    )


# Confirmed on emu-quasar-1x3 (2026-09-03): the fork's compute kernel (compute_streaming.hpp)
# fails to JIT on Quasar — it uses the WH/BH-only experimental custom compute APIs
# (mm_no_mop_init_short / mm_no_mop_reinit_short / matmul_block_no_mop from matmul_custom.h,
# sub_bcast_cols_init_short_custom / sub_tiles_bcast_cols_custom from sdpa_sub_custom.h,
# exp_packthread_tile_init, log_tile_init) and raw WH LLK internals (t6_semaphore_wait_on_zero,
# ckernel::semaphore::PACK_DONE) that have no Quasar implementation.
_SDPA_PREFILL_BLOCKER = "sdpa quasar fork compute kernel uses WH/BH-only custom LLK APIs; no Quasar JIT"
_SDPA_DECODE_BLOCKER = "sdpa_decode quasar fork: multi-bound DFB 'out_o' rejected on Gen2"


def _quasar_xfail(mesh, reason, fn):
    """Quasar-only strict xfail: WH/BH must pass; on Quasar the known blocker must still fire."""
    if not G._is_quasar(mesh):
        return fn()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — TT_FATAL / JIT failure surface as RuntimeError
        pytest.xfail(f"{reason}: {str(exc).splitlines()[0][:160]}")
    pytest.fail(f"XPASS on Quasar — blocker fixed, drop the xfail: {reason}")


@U.with_default_mesh()
@pytest.mark.parametrize("nh,nkv", [(1, 1), (2, 1)], ids=["1q1kv", "2q1kv_gqa"])
def test_sdpa_prefill_fork(ttnn_mesh_device, reset_seeds, nh, nkv):
    _quasar_xfail(ttnn_mesh_device, _SDPA_PREFILL_BLOCKER, lambda: _sdpa_prefill_fork(ttnn_mesh_device, nh, nkv))


def _sdpa_prefill_fork(mesh, nh, nkv):
    q_t = U.torch_rand((1, nh, _SDPA_SEQ, HEAD_DIM))
    k_t = U.torch_rand((1, nkv, _SDPA_SEQ, HEAD_DIM))
    v_t = U.torch_rand((1, nkv, _SDPA_SEQ, HEAD_DIM))

    q = _to_device_tiled(q_t, mesh)
    k = _to_device_tiled(k_t, mesh)  # captured dtype is bfloat8_b — unsupported on Quasar, see header
    v = _to_device_tiled(v_t, mesh)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        q_chunk_size=_SDPA_CHUNK,
        k_chunk_size=_SDPA_CHUNK,
        exp_approx_mode=False,
        max_cores_per_head_batch=16,
    )
    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=SCALE, program_config=program_config
    )
    assert tuple(out.shape) == (1, nh, _SDPA_SEQ, HEAD_DIM)
    ref = _torch_sdpa_causal(q_t, k_t, v_t, SCALE)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)


# =============================================================================
# paged SDPA decode (experimental.quasar fork)
# =============================================================================

_DEC_NH, _DEC_NKV = 32, 8
_DEC_PAGES, _DEC_BLOCK = 16, 32  # 512-token cache, identity page table
_DEC_CUR_POS = 8
_DEC_K_CHUNK = 32  # k_chunk_size=0 in the capture -> op default; 32 keeps the padded length explicit


def _torch_decode_ref(q, k, v, cur_pos, scale, padded_len):
    b, nh = q.shape[1], q.shape[2]
    nkv = k.shape[1]
    q_s = q.permute(1, 2, 0, 3).float()  # [b, nh, 1, d]
    k_s = k[:, :, :padded_len, :].repeat_interleave(nh // nkv, dim=1).float()
    v_s = v[:, :, :padded_len, :].repeat_interleave(nh // nkv, dim=1).float()
    mask = torch.zeros((b, nh, 1, padded_len))
    mask[:, :, :, cur_pos + 1 :] = torch.finfo(torch.float32).min
    out = torch.nn.functional.scaled_dot_product_attention(q_s, k_s, v_s, mask, scale=scale, is_causal=False)
    return out.squeeze(2).unsqueeze(0)  # [1, b, nh, d]


# Confirmed on emu-quasar-1x3 (2026-09-03): DFB 'out_o' sets allow_instance_multi_binding
# (tree-reduction: writer P+C, compute P+C), which ValidateProgramSpec rejects on Gen2
# (program_spec.cpp:1288) even when the grid has a single core and no reduction happens.
@U.with_default_mesh()
def test_sdpa_decode_fork(ttnn_mesh_device, reset_seeds):
    _quasar_xfail(ttnn_mesh_device, _SDPA_DECODE_BLOCKER, lambda: _sdpa_decode_fork(ttnn_mesh_device))


def _sdpa_decode_fork(mesh):
    seq = _DEC_PAGES * _DEC_BLOCK
    q_t = U.torch_rand((1, 1, _DEC_NH, HEAD_DIM))
    k_t = U.torch_rand((1, _DEC_NKV, seq, HEAD_DIM))
    v_t = U.torch_rand((1, _DEC_NKV, seq, HEAD_DIM))
    # Paged layout [pages, nkv, block, d] with an identity page table.
    paged_k = (
        k_t.reshape(1, _DEC_NKV, _DEC_PAGES, _DEC_BLOCK, HEAD_DIM)
        .transpose(1, 2)
        .reshape(_DEC_PAGES, _DEC_NKV, _DEC_BLOCK, HEAD_DIM)
    )
    paged_v = (
        v_t.reshape(1, _DEC_NKV, _DEC_PAGES, _DEC_BLOCK, HEAD_DIM)
        .transpose(1, 2)
        .reshape(_DEC_PAGES, _DEC_NKV, _DEC_BLOCK, HEAD_DIM)
    )
    page_table = torch.arange(_DEC_PAGES, dtype=torch.int32).reshape(1, _DEC_PAGES)
    cur_pos = torch.tensor([_DEC_CUR_POS], dtype=torch.int32)

    q = _to_device_tiled(q_t, mesh, memory_config=_hs_l1_1core([_DEC_NH, HEAD_DIM]))  # captured: HS L1, 1 core
    k = _to_device_tiled(paged_k, mesh)  # captured cache dtype is bfloat8_b — unsupported on Quasar
    v = _to_device_tiled(paged_v, mesh)
    page_table_tt = ttnn.to_device(
        ttnn.from_torch(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
        ),
        mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cur_pos_tt = ttnn.to_device(
        ttnn.from_torch(
            cur_pos,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
        ),
        mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        q_chunk_size=0,
        k_chunk_size=_DEC_K_CHUNK,
        exp_approx_mode=False,
        max_cores_per_head_batch=16,
    )
    out = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode(
        q,
        k,
        v,
        page_table_tt,
        cur_pos_tensor=cur_pos_tt,
        scale=SCALE,
        program_config=program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert tuple(out.shape) == (1, 1, _DEC_NH, HEAD_DIM)
    padded_len = ((_DEC_CUR_POS + 1 + _DEC_K_CHUNK - 1) // _DEC_K_CHUNK) * _DEC_K_CHUNK
    ref = _torch_decode_ref(q_t, k_t, v_t, _DEC_CUR_POS, SCALE, padded_len)
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
