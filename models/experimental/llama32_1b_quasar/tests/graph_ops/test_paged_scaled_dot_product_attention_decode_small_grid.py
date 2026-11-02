# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# MANUAL companion to the generated
# test_paged_scaled_dot_product_attention_decode.py — do NOT regenerate this file
# from a capture.
#
# The captured case pins SDPAProgramConfig.compute_with_storage_grid_size = [8, 8]
# (64 cores). sdpa_decode_program_factory.cpp:194 then FATALs on any device whose
# compute grid has fewer than 64 cores ("Cores available (64) exceeds grid size").
#
# Nothing about this problem actually needs 64 cores: it is a paged decode with
# batch B = page_table.padded_shape[0] = 1 (factory line 124-126), 8 KV heads, non-
# MLA. The only core minimums are `num_cores_available <= device grid` (line 194)
# and `num_cores_available >= B` = >= 1 (line 199); the core-allocation math
# (lines 202-211) scales to any core count (2 cores -> 4 KV heads/core; 32 cores ->
# 4 cores/KV head). So we reuse the captured case verbatim and only shrink the grid.
#
# Each variant SKIPS (rather than FATALs) when the device compute grid is smaller
# than the variant's grid rectangle, so the file is safe to run on any device.
# ---------------------------------------------------------------------------
"""Small-grid variants (2 and 32 cores) of ``paged_scaled_dot_product_attention_decode``."""

import copy

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.graph_ops import graph_case as G

_OP = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode

# The captured signature (see the generated test), reused verbatim except for the
# program config's compute grid, which each variant overrides below.
_BASE_CASE = {
    "op": "ttnn.transformer.paged_scaled_dot_product_attention_decode",
    "count": 1,
    "args": [
        {
            "k": "t",
            "shape": [1, 1, 32, 64],
            "dtype": "BFLOAT16",
            "layout": "TILE",
            "mem": {
                "layout": "HEIGHT_SHARDED",
                "buffer": "L1",
                "shard": {"grid": [[0, 0, 0, 0]], "shape": [32, 64], "orientation": "ROW_MAJOR"},
            },
        },
        {
            "k": "t",
            "shape": [128, 8, 32, 64],
            "dtype": "BFLOAT16",
            "layout": "TILE",
            "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
        },
        {
            "k": "t",
            "shape": [128, 8, 32, 64],
            "dtype": "BFLOAT16",
            "layout": "TILE",
            "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
        },
    ],
    "kwargs": {
        "page_table_tensor": {
            "k": "t",
            "shape": [1, 128],
            "dtype": "INT32",
            "layout": "ROW_MAJOR",
            "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
        },
        "cur_pos_tensor": {
            "k": "t",
            "shape": [1],
            "dtype": "INT32",
            "layout": "ROW_MAJOR",
            "mem": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None},
        },
        "scale": {"k": "lit", "v": 0.125},
        "sliding_window_size": {"k": "lit", "v": None},
        "program_config": {
            "kind": "SDPAProgramConfig",
            "fields": {
                "compute_with_storage_grid_size": [8, 8],  # overridden per variant
                "sub_core_grids": None,
                "q_chunk_size": 0,
                "k_chunk_size": 0,
                "exp_approx_mode": True,
                "max_cores_per_head_batch": 16,
            },
            "k": "cfg",
        },
        "memory_config": {"layout": "INTERLEAVED", "buffer": "DRAM", "shard": None, "k": "mem"},
    },
    "outs": [
        {
            "dtype": "BFLOAT16",
            "k": "t",
            "layout": "TILE",
            "mem": {"buffer": "DRAM", "layout": "INTERLEAVED", "shard": None},
            "shape": [1, 1, 32, 64],
        },
    ],
}


def _case_with_grid(case_id, grid_xy):
    case = copy.deepcopy(_BASE_CASE)
    case["id"] = case_id
    case["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"] = list(grid_xy)
    return case


CASES = [
    # 2core: the STOCK op is designed for 1 KV-head/core, so with 8 KV-heads it cannot run on 2 cores -- forcing
    # >1 KV-head/core hangs on Quasar (workers stall at WFW on the KV-mcast/DFB handshake). The supported path is
    # the per-kv-head split (1 KV-head/core -- see test_..._split_small_grid below and
    # _install_quasar_device_sdpa_split). run=False so it does NOT execute (a hang, not a catchable failure).
    # Un-xfail once the op packs multiple heads/core.
    pytest.param(
        _case_with_grid("2core_2x1_bf16", (2, 1)),
        id="2core_2x1_bf16",
        marks=pytest.mark.xfail(
            reason="stock paged decode SDPA is designed for 1 KV-head/core; with 8 KV-heads it cannot run on 2 cores (use the split)",
            run=False,
        ),
    ),
    pytest.param(_case_with_grid("32core_8x4_bf16", (8, 4)), id="32core_8x4_bf16"),
]


@G.with_default_mesh()
@pytest.mark.parametrize("case", CASES)
def test_paged_scaled_dot_product_attention_decode_small_grid(ttnn_mesh_device, reset_seeds, case):
    grid_x, grid_y = case["kwargs"]["program_config"]["fields"]["compute_with_storage_grid_size"]
    dev = ttnn_mesh_device.compute_with_storage_grid_size()
    if dev.x < grid_x or dev.y < grid_y:
        pytest.skip(f"case needs a {grid_x}x{grid_y} grid rectangle; device compute grid is {dev.x}x{dev.y}")
    G.run_case(_OP, case, ttnn_mesh_device)


# ---------------------------------------------------------------------------
# Per-KV-head SPLIT variant: the no-hang path for a grid with FEWER cores than KV heads.
# The stock 2core case above packs 8 KV-heads / 2 cores = 4/core and hangs on Quasar. Splitting into N_KV
# single-KV-head calls (full-q sharded + 1 KV-head each, max_cores_per_head_batch=1) maps exactly 1 KV-head per
# core -- the only config that doesn't hang -- then keeps each head's qpk output rows and concatenates. Mirrors
# _install_quasar_device_sdpa_split (the e2e fix) and debug_ops::test_paged_sdpa_decode_split. KV cache shrunk to
# 8 blocks (from the captured 128) to keep the sim fast; the geometry (8 KV heads, block_size, head_dim, grid)
# is preserved.
# ---------------------------------------------------------------------------
_SPLIT_N_Q = 32  # q heads
_SPLIT_N_KV = 8  # kv heads
_SPLIT_HD = 64  # head_dim
_SPLIT_BS = 32  # paged block size (tile height)
_SPLIT_BLOCKS = 8  # small cache for sim speed
_SPLIT_SCALE = _SPLIT_HD**-0.5


def _split_tile_bf16_dram(t, mesh):
    """bf16 TILE DRAM-interleaved via RM upload + quasar.tilize (from_torch(TILE) hangs on the sim)."""
    rm = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )
    qt = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    return (qt or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


def _split_q_height_sharded(t, mesh):
    """Full 32-head q -> bf16 TILE HEIGHT_SHARDED on one core (the validated no-hang q layout)."""
    qt = _split_tile_bf16_dram(t, mesh)
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard = ttnn.ShardSpec(crs, (_SPLIT_BS, _SPLIT_HD), ttnn.ShardOrientation.ROW_MAJOR)
    memcfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
    qi2s = getattr(getattr(ttnn.experimental, "quasar", None), "interleaved_to_sharded", None)
    return (qi2s or ttnn.interleaved_to_sharded)(qt, memcfg)


def _split_int32_rm_dram(t, mesh):
    return ttnn.from_torch(
        t,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )


def _split_torch_gqa_decode(q, keys, values, cur_pos, scale):
    """Reference: identity page table (block b -> positions [b*BS,(b+1)*BS)); causal decode over 0..cur_pos;
    GQA q-head h attends kv-head h // (nq//nkv). Returns [nq, hd]."""
    nb, nkv, bs, hd = keys.shape
    nq = q.shape[2]
    qpk = nq // nkv
    seqlen = cur_pos + 1
    k = keys.permute(1, 0, 2, 3).reshape(nkv, nb * bs, hd)[:, :seqlen, :].float()
    v = values.permute(1, 0, 2, 3).reshape(nkv, nb * bs, hd)[:, :seqlen, :].float()
    out = torch.zeros(nq, hd)
    for h in range(nq):
        kvh = h // qpk
        w = torch.softmax((k[kvh] @ q[0, 0, h].float()) * scale, dim=0)
        out[h] = w @ v[kvh]
    return out


def _split_pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@G.with_default_mesh()
@pytest.mark.parametrize("num_cores", [1, 2], ids=["1core", "2core"])
def test_paged_scaled_dot_product_attention_decode_split_small_grid(ttnn_mesh_device, reset_seeds, num_cores):
    """Per-KV-head split of paged decode SDPA (1 KV-head/core -> no hang). Runs on 1 and 2 cores; validates
    finiteness AND PCC vs a torch GQA reference. This is the passing 2-core decode-SDPA coverage that the stock
    2core case (xfail above) cannot provide."""
    mesh = ttnn_mesh_device
    dev = mesh.compute_with_storage_grid_size()
    if int(dev.x) < num_cores:
        pytest.skip(f"split needs >= {num_cores} cores in a row; device is {dev.x}x{dev.y}")

    nq, nkv, hd = _SPLIT_N_Q, _SPLIT_N_KV, _SPLIT_HD
    qpk = nq // nkv
    cur_pos = 200

    torch.manual_seed(0)
    q = torch.randn(1, 1, nq, hd, dtype=torch.bfloat16)
    keys = torch.randn(_SPLIT_BLOCKS, nkv, _SPLIT_BS, hd, dtype=torch.bfloat16)
    values = torch.randn(_SPLIT_BLOCKS, nkv, _SPLIT_BS, hd, dtype=torch.bfloat16)
    page_table = torch.arange(_SPLIT_BLOCKS, dtype=torch.int32).reshape(1, _SPLIT_BLOCKS)
    cur_pos_t = torch.full((1,), cur_pos, dtype=torch.int32)

    q_t = _split_q_height_sharded(q, mesh)  # full 32-head q, height-sharded (validated no-hang layout)
    k_t = _split_tile_bf16_dram(keys, mesh)
    v_t = _split_tile_bf16_dram(values, mesh)
    pt_t = _split_int32_rm_dram(page_table, mesh)
    cp_t = _split_int32_rm_dram(cur_pos_t, mesh)

    kept = []
    for h in range(nkv):
        # one KV-head: slice k/v on the (non-tiled) kv-head axis (dim 1)
        k_h = ttnn.slice(k_t, [0, h, 0, 0], [_SPLIT_BLOCKS, h + 1, _SPLIT_BS, hd])
        v_h = ttnn.slice(v_t, [0, h, 0, 0], [_SPLIT_BLOCKS, h + 1, _SPLIT_BS, hd])
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(num_cores, 1),
            exp_approx_mode=True,
            q_chunk_size=0,
            k_chunk_size=_SPLIT_BS,
            max_cores_per_head_batch=1,
        )
        cc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        out_h = _OP(
            q_t,
            k_h,
            v_h,
            page_table_tensor=pt_t,
            cur_pos_tensor=cp_t,
            scale=_SPLIT_SCALE,
            program_config=pc,
            compute_kernel_config=cc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(mesh)
        o_h = ttnn.to_torch(out_h).reshape(-1, hd)  # [nq, hd]; every q-head vs this kv-head
        kept.append(o_h[h * qpk : (h + 1) * qpk])  # keep only the qpk heads belonging to kv-head h

    o = torch.cat(kept, dim=0)  # [nq, hd]
    ref = _split_torch_gqa_decode(q, keys, values, cur_pos, _SPLIT_SCALE)
    pcc = _split_pcc(o, ref)
    assert torch.isfinite(o).all(), "split decode SDPA produced non-finite output"
    assert pcc > 0.99, f"split decode SDPA PCC too low: {pcc}"
