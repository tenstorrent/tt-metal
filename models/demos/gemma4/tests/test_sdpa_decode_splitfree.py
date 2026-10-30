# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device checks of ``custom_ops/sdpa_decode_splitfree`` (the split-independent paged decode attention): accuracy
against a torch reference over random paged caches and page tables, the bitwise identity of a row across the
padded batches and the grids, and the long-context timing against the stock op pinned (3 cores per head, the
Galaxy catalog's value) and unpinned.

Gated like ``tests/unit/test_attention.py::test_decode_sdpa_core_split_long_context``: ``GEMMA4_SDPA_TEST_CACHE``
must be set (the S of the timing table, e.g. 262144), so the long device run never starts by accident:

    TT_VISIBLE_DEVICES=<chip> GEMMA4_SDPA_TEST_CACHE=262144 pytest models/demos/gemma4/tests/test_sdpa_decode_splitfree.py
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.custom_ops.sdpa_decode_splitfree import SdpaDecodeSplitfree

SINGLE_CHIP = [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip")]
CACHE_ENV = "GEMMA4_SDPA_TEST_CACHE"
BLOCK = 64
PCC_MIN = 0.999

gated = pytest.mark.skipif(
    not os.environ.get(CACHE_ENV),
    reason=f"long-context device checks of sdpa_decode_splitfree: set {CACHE_ENV}=<tokens> to run them",
)


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


class _Case:
    """One paged cache of ``S`` tokens on the device (random data, bfloat8_b), its torch mirror and the scratch."""

    def __init__(self, mesh_device, S, H_q, H_kv, D, dtype=ttnn.bfloat8_b, seed=0):
        torch.manual_seed(seed)
        self.device, self.S, self.H_q, self.H_kv, self.D = mesh_device, S, H_q, H_kv, D
        self.num_blocks = S // BLOCK
        self.k_cache = ttnn.from_torch(
            torch.randn(self.num_blocks, H_kv, BLOCK, D, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.v_cache = ttnn.from_torch(
            torch.randn(self.num_blocks, H_kv, BLOCK, D, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # the reference reads what the device reads (the bfp8 rounding)
        self.k_ref = ttnn.to_torch(self.k_cache).float()
        self.v_ref = ttnn.to_torch(self.v_cache).float()
        tiles = SdpaDecodeSplitfree.scratch_tiles(
            mesh_device, 32, [(H_kv, H_q // H_kv, D, 1024), (H_kv, H_q // H_kv, D, None)], S, kv_dtype=dtype
        )
        self.scratch = SdpaDecodeSplitfree.allocate_scratch(mesh_device, tiles)

    def inputs(self, B, cur_pos, tables, q):
        """Device tensors of one call: ``q [B, H_q, D]`` (the TILE layout pads the heads to 32 rows, as the
        model's Q arrives), ``cur_pos [B]``, ``tables [B, blocks]``."""
        tt_q = ttnn.from_torch(
            q.reshape(1, B, self.H_q, self.D), device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tt_pos = ttnn.from_torch(
            cur_pos.to(torch.int32), device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
        )
        tt_tab = ttnn.from_torch(
            tables.to(torch.int32), device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
        )
        return tt_q, tt_pos, tt_tab

    def run(self, tt_q, tt_pos, tt_tab, **kw):
        return SdpaDecodeSplitfree.op(tt_q, self.k_cache, self.v_cache, tt_pos, tt_tab, self.scratch, scale=1.0, **kw)

    def golden(self, q, cur_pos, tables, **kw):
        return SdpaDecodeSplitfree.golden(
            q.reshape(1, -1, self.H_q, self.D), self.k_ref, self.v_ref, tables, cur_pos, scale=1.0, **kw
        )

    def random_tables(self, B, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.stack([torch.randperm(self.num_blocks, generator=g) for _ in range(B)])


def _to_rows(out, B, H_q):
    return ttnn.to_torch(out)[0, :B, :H_q, :].contiguous()


@gated
@pytest.mark.parametrize("S", [4096, 131072, 262144])
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_matches_torch(mesh_device, device_params, S, reset_seeds):
    """(a) PCC > 0.999 against torch at S in {4K, 128K, 256K} for B in {1, 8, 32}: random caches, random page
    tables per row, random live positions (row 0 at S-1), some padding rows; one kv head, 4 q heads, head_dim 512
    (the Galaxy TP8 global layer)."""
    case = _Case(mesh_device, S, H_q=4, H_kv=1, D=512)
    for B in (1, 8, 32):
        q = torch.randn(B, 4, 512, dtype=torch.bfloat16)
        cur_pos = torch.randint(S // 2, S, (B,))
        cur_pos[0] = S - 1
        if B > 1:
            cur_pos[1] = 0  # a one-token row
            cur_pos[B - 1] = -1  # a padding row
        tables = case.random_tables(B, seed=B)
        out = case.run(*case.inputs(B, cur_pos, tables))
        got = _to_rows(out, B, 4)
        ref = case.golden(q, cur_pos, tables)[0]
        live = [b for b in range(B) if cur_pos[b] >= 0]
        pcc = _pcc(got[live], ref[live])
        logger.info(f"S={S} B={B}: pcc {pcc:.6f} over {len(live)} live rows")
        assert pcc > PCC_MIN, f"S={S} B={B}: pcc {pcc:.5f}"
        out.deallocate(True)


@gated
@pytest.mark.parametrize(
    "H_q, H_kv, window, modulo",
    [(4, 2, None, None), (4, 2, 1024, None), (2, 1, 1024, 2048)],
    ids=["gqa", "gqa-window", "window-ring"],
)
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_gqa_and_window(mesh_device, device_params, H_q, H_kv, window, modulo, reset_seeds):
    """Two KV groups (the face-line output path), the sliding window (the window mask, one chunk per unit) and
    the bounded ring (the virtual tile wrap before the page-table lookup) at S = 4096, head_dim 256."""
    S = 4096
    case = _Case(mesh_device, S, H_q=H_q, H_kv=H_kv, D=256)
    B = 8
    q = torch.randn(B, H_q, 256, dtype=torch.bfloat16)
    cur_pos = torch.tensor([S - 1, 0, 1023, 1024, 1025, 2500, -1, 3000])
    if modulo:
        # the ring holds modulo tokens: every row's positions wrap into its first modulo/BLOCK blocks
        tables = case.random_tables(B, seed=7)
    else:
        tables = case.random_tables(B, seed=7)
    kw = dict(num_kv_heads=H_kv, block_size=BLOCK, sliding_window=window, cache_position_modulo=modulo)
    out = case.run(*case.inputs(B, cur_pos, tables), **kw)
    got = _to_rows(out, B, H_q)
    ref = case.golden(q, cur_pos, tables, **kw)[0]
    live = [b for b in range(B) if cur_pos[b] >= 0]
    pcc = _pcc(got[live], ref[live])
    logger.info(f"H_q={H_q} H_kv={H_kv} window={window} ring={modulo}: pcc {pcc:.6f}")
    assert pcc > PCC_MIN, f"pcc {pcc:.5f}"


@gated
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_row_bits_independent_of_batch_and_grid(mesh_device, device_params, reset_seeds, monkeypatch):
    """(b) row 0's output bits are identical between the B=1 run and the B=8 / B=32 runs that carry the same row
    (other rows random and live), and between the device grid and a forced 8x8 grid."""
    S = int(os.environ.get(CACHE_ENV, "131072"))
    case = _Case(mesh_device, S, H_q=4, H_kv=1, D=512)
    q0 = torch.randn(4, 512, dtype=torch.bfloat16)
    table0 = case.random_tables(1, seed=11)[0]
    bits = {}
    for B in (1, 8, 32):
        q = torch.randn(B, 4, 512, dtype=torch.bfloat16)
        q[0] = q0
        cur_pos = torch.randint(0, S, (B,))
        cur_pos[0] = S - 1
        tables = case.random_tables(B, seed=100 + B)
        tables[0] = table0
        out = case.run(*case.inputs(B, cur_pos, tables))
        bits[B] = _to_rows(out, B, 4).view(torch.int16).clone()[0]
        out.deallocate(True)
    monkeypatch.setenv("GEMMA4_TEST_GRID", "8x8")
    out = case.run(*case.inputs(1, torch.tensor([S - 1]), table0.unsqueeze(0), q0.unsqueeze(0)))
    bits["8x8"] = _to_rows(out, 1, 4).view(torch.int16).clone()[0]
    monkeypatch.delenv("GEMMA4_TEST_GRID")
    ref = case.golden(q0.unsqueeze(0), torch.tensor([S - 1]), table0.unsqueeze(0))[0, 0]
    assert _pcc(bits[1].view(torch.bfloat16), ref) > PCC_MIN
    for key in (8, 32, "8x8"):
        assert torch.equal(bits[key], bits[1]), f"row 0 differs between the B=1 run and {key}"


@gated
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP, indirect=True)
def test_splitfree_timing_vs_stock(mesh_device, device_params, reset_seeds):
    """(c) ms per op at S = GEMMA4_SDPA_TEST_CACHE for the padded batches 1 / 8 / 32 with row 0 live at S-1 and
    the other rows padding (the lone-user scenario the pin punishes): the custom op against the stock op at the
    caps 3 (the Galaxy pin), 16 (the op default) and 32. Asserted: the stock op at cap 3 gives row 0 the same
    bytes in every padded batch (the pin's purpose, as the earlier core-split harness checked), the custom op
    does so without any pin, and -- loosely -- the custom op at B=1 runs within 2x of the stock at cap 16."""
    S = int(os.environ[CACHE_ENV])
    case = _Case(mesh_device, S, H_q=4, H_kv=1, D=512)
    grid = mesh_device.compute_with_storage_grid_size()
    q0 = torch.randn(4, 512, dtype=torch.bfloat16)
    table = torch.arange(case.num_blocks, dtype=torch.int32)
    times, bits = {}, {}
    reps = 10
    for B in (1, 8, 32):
        q = torch.zeros(B, 4, 512, dtype=torch.bfloat16)
        q[0] = q0
        cur_pos = torch.full((B,), -1)
        cur_pos[0] = S - 1
        tables = table.unsqueeze(0).repeat(B, 1)
        tt_q, tt_pos, tt_tab = case.inputs(B, cur_pos, tables)

        def custom():
            return case.run(tt_q, tt_pos, tt_tab)

        def stock(cap):
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                q_chunk_size=32,
                k_chunk_size=64,
                exp_approx_mode=False,
                max_cores_per_head_batch=cap,
            )
            return ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tt_q,
                case.k_cache,
                case.v_cache,
                cur_pos_tensor=tt_pos,
                page_table_tensor=tt_tab,
                scale=1.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=pc,
            )

        runs = {
            "custom": custom,
            "stock cap 3": lambda: stock(3),
            "stock cap 16": lambda: stock(16),
            "stock cap 32": lambda: stock(32),
        }
        outs = {}
        for name, fn in runs.items():
            o = fn()
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(reps):
                r = fn()
                r.deallocate(True)
            ttnn.synchronize_device(mesh_device)
            times[(B, name)] = (time.perf_counter() - t0) * 1e3 / reps
            outs[name] = _to_rows(o, B, 4)[0].contiguous()
            o.deallocate(True)
        for name in runs:
            logger.info(f"  S={S} B={B:2d} {name:13s}: {times[(B, name)]:7.2f} ms/op")
        assert _pcc(outs["custom"], outs["stock cap 16"]) > PCC_MIN
        bits[B] = {name: outs[name].view(torch.int16).clone() for name in ("custom", "stock cap 3")}
    for B in (8, 32):
        assert torch.equal(bits[B]["stock cap 3"], bits[1]["stock cap 3"]), f"stock cap 3: row 0 differs at B={B}"
        assert torch.equal(bits[B]["custom"], bits[1]["custom"]), f"custom: row 0 differs at B={B}"
    for B in (1, 8, 32):
        logger.info(
            f"S={S} B={B}: custom {times[(B, 'custom')]:.2f} ms vs stock cap 3 {times[(B, 'stock cap 3')]:.2f} / "
            f"cap 16 {times[(B, 'stock cap 16')]:.2f} / cap 32 {times[(B, 'stock cap 32')]:.2f} ms"
        )
    assert (
        times[(1, "custom")] <= 2.0 * times[(1, "stock cap 16")]
    ), f"custom B=1 {times[(1, 'custom')]:.2f} ms vs stock cap 16 {times[(1, 'stock cap 16')]:.2f} ms"
