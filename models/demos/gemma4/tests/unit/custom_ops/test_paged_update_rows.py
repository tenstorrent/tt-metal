# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PagedUpdateRows (custom op) vs the per-position ``paged_update_cache`` loop of the packed verify
(``tt/attention/decode.py::packed_kv_write_loop``): random packed K / V rows written through both paths into
identical caches, byte equality of the whole caches (the touched tile rows and every untouched byte), bf16 and
bfp8 caches, positions inside one tile row, across a tile edge, across a block edge, around the ring modulo, a
skipped (-1) position, and the program-cache repeat.

    pytest models/demos/gemma4/tests/unit/custom_ops/test_paged_update_rows.py
"""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid, core_range_set, row_major_cores
from models.demos.gemma4.custom_ops.paged_update_rows import PagedUpdateRows
from models.demos.gemma4.tt.attention.decode import packed_kv_write_loop, packed_kv_write_rows

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
CI = os.environ.get("CI") == "true"
BLOCK = 64  # the served vLLM block
WIDTH = 8  # page-table blocks per row (each row its own physical blocks)
RING_BLOCKS = 4  # the bounded ring of the ring_wrap scenario
# (B, P, nkv, head_dim): the serving records (1,5) / (8,3) / (16,1) at the 26B's TP4 head counts (1 global / 2
# sliding, D 256), the dense D 512 global layers, the B=32 P=1 one-job form and an 8-head D 128 form that puts
# several units on a core (256 units)
SHAPES = [
    (1, 6, 1, 256),
    (8, 4, 2, 256),
    (16, 2, 2, 256),
    (1, 6, 2, 512),
    (16, 2, 1, 512),
    (32, 1, 2, 256),
    (5, 5, 8, 128),
]
SCENARIOS = ["tile_mid", "tile_edge", "block_edge", "ring_wrap", "skip"]
DTYPES = [pytest.param(ttnn.bfloat16, id="bf16"), pytest.param(ttnn.bfloat8_b, id="bfp8")]


def _positions(scenario, B, P):
    """``([P, B] int32 write positions, ring modulo)`` of a scenario: row b's positions are ``c_b + p``."""
    modulo = RING_BLOCKS * BLOCK if scenario == "ring_wrap" else None
    idx = torch.zeros(P, B, dtype=torch.int32)
    for b in range(B):
        k = b % 3  # the row's virtual block
        if scenario == "tile_mid":
            c = BLOCK * k + (32 - P) // 2  # inside the block's first tile row
        elif scenario == "tile_edge":
            c = BLOCK * k + 32 - (P + 1) // 2  # across the two tile rows of one block
        elif scenario == "block_edge":
            c = BLOCK * k + BLOCK - (P + 1) // 2  # across a block boundary (two physical blocks)
        elif scenario == "ring_wrap":
            c = modulo * (1 + b % 2) + modulo - (P + 1) // 2  # absolute positions past the ring: wrap to its start
        else:
            c = BLOCK * k + 5
        for p in range(P):
            idx[p, b] = c + p
        if scenario == "skip":
            idx[b % P, b] = -1  # the stock's skip sentinel for one position of every row
    return idx, modulo


def _cases():
    out = []
    for shape in SHAPES:
        for sc in SCENARIOS:
            in_ci = sc in ("tile_edge", "ring_wrap") or shape == (8, 4, 2, 256)
            marks = [] if in_ci or not CI else [pytest.mark.skip(reason="CI subset")]
            out.append(pytest.param(shape, sc, id=f"B{shape[0]}P{shape[1]}H{shape[2]}D{shape[3]}-{sc}", marks=marks))
    return out


class _Case:
    """Two identical paged caches (K and V), their page table and the P write-position tensors on the device."""

    def __init__(self, mesh_device, B, P, nkv, D, dtype, scenario, seed=0):
        torch.manual_seed(seed)
        self.device, self.B, self.P, self.nkv, self.D, self.dtype = mesh_device, B, P, nkv, D, dtype
        self.blocks = B * WIDTH + 1
        self.k_host = torch.randn(self.blocks, nkv, BLOCK, D).to(torch.bfloat16)
        self.v_host = torch.randn(self.blocks, nkv, BLOCK, D).to(torch.bfloat16)
        g = torch.Generator().manual_seed(seed + 1)
        self.pt_host = torch.stack([b * WIDTH + torch.randperm(WIDTH, generator=g) for b in range(B)]).to(torch.int32)
        self.idx_host, self.modulo = _positions(scenario, B, P)
        self.pt = ttnn.from_torch(self.pt_host, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
        self.idx = [
            ttnn.from_torch(
                self.idx_host[p].clone(), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
            )
            for p in range(P)
        ]
        self.k_src = torch.randn(1, nkv, B * P, D).to(torch.bfloat16)
        self.v_src = torch.randn(1, nkv, B * P, D).to(torch.bfloat16)

    def caches(self):
        return tuple(
            ttnn.from_torch(h, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
            for h in (self.k_host, self.v_host)
        )

    def sources(self):
        """The prefill-style K / V the packed verify hands the write (bf16 TILE on L1, as after RoPE)."""
        return tuple(
            ttnn.from_torch(
                h, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            for h in (self.k_src, self.v_src)
        )

    def sharded_mem(self):
        """The loop's one-user-per-core layout (``nlp_create_qkv_heads_decode``'s output spec: B cores, [32, D])."""
        cores = core_range_set(row_major_cores(compute_grid(self.device), self.B))
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, [32, self.D], ttnn.ShardOrientation.ROW_MAJOR),
        )

    def run_loop(self):
        k, v = self.caches()
        ks, vs = self.sources()
        packed_kv_write_loop(
            k, v, ks, vs, self.idx, self.pt, self.B, self.P, self.nkv, self.D, BLOCK, self.sharded_mem(), self.modulo
        )
        return k, v

    def run_op(self):
        k, v = self.caches()
        ks, vs = self.sources()
        packed_kv_write_rows(k, v, ks, vs, self.idx, self.pt, self.B, self.nkv, BLOCK, self.modulo)
        return k, v

    def golden(self):
        return tuple(
            PagedUpdateRows.golden(
                h,
                s,
                self.idx_host,
                self.pt_host,
                num_kv_heads=self.nkv,
                block_size=BLOCK,
                cache_position_modulo=self.modulo,
            )
            for h, s in ((self.k_host, self.k_src), (self.v_host, self.v_src))
        )


def _read(t):
    return ttnn.to_torch(t).float()


@pytest.mark.parametrize("shape, scenario", _cases())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_paged_update_rows_matches_loop(mesh_device, device_params, shape, scenario, dtype):
    B, P, nkv, D = shape
    case = _Case(mesh_device, B, P, nkv, D, dtype, scenario)
    assert (
        PagedUpdateRows.unsupported_reason(
            case.caches()[0],
            case.idx,
            case.pt,
            batch=B,
            packed_p=P,
            num_kv_heads=nkv,
            head_dim=D,
            block_size=BLOCK,
            modulo=case.modulo,
        )
        is None
    )
    k_ref, v_ref = case.run_loop()
    k_out, v_out = case.run_op()
    for name, ref, out in (("K", k_ref, k_out), ("V", v_ref, v_out)):
        r, o = _read(ref), _read(out)
        assert torch.equal(o, r), f"{name}: op differs from the loop at {int((o != r).sum())} elements"
    if dtype == ttnn.bfloat16:
        for name, out, gold in zip("KV", (k_out, v_out), case.golden()):
            assert torch.equal(_read(out), gold.float()), f"{name}: op differs from the torch golden"
    # the program-cache hit: fresh buffers, the same program
    k2, v2 = case.run_op()
    assert torch.equal(_read(k2), _read(k_out)) and torch.equal(_read(v2), _read(v_out))


@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_paged_update_rows_on_the_wormhole_grid(mesh_device, device_params, monkeypatch):
    """The 256-unit shape on the 8x8 split (4 units per core) of the same chip: the same bytes as the loop."""
    monkeypatch.setenv("GEMMA4_TEST_GRID", "8x8")
    case = _Case(mesh_device, 32, 1, 8, 128, ttnn.bfloat8_b, "block_edge")
    k_ref, v_ref = case.run_loop()
    k_out, v_out = case.run_op()
    assert torch.equal(_read(k_out), _read(k_ref)) and torch.equal(_read(v_out), _read(v_ref))
