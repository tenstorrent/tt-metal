# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight placement: correct on anything, coalesced whenever a run can be proven.

`nd_shard_n_tiles()` is the ONE place the op learns a weight's layout, and everything downstream
is a run length. The hazard it guards is specific: an unrecognised placement is SILENTLY CORRECT
and just slower, so a detection bug shows up as a perf number attributed to the wrong path, never
as a failure. Every case here therefore asserts the reader's own predicate as well as the output.

The interesting case is `shard_tall` — the preferred N extent at four tile-rows instead of one.
That is NOT the placement `weight_memory_configs` recommends, but it is still coalescible
(`page_offset_within_shard` is `(k % SH) * SHARD_W + (n % SHARD_W)`, so for a fixed k consecutive n
stay contiguous at any height), and the detection must earn the fast path for it rather than
falling back to one transaction per tile.
"""

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu, weight_memory_configs
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import nd_shard_n_tiles

TILE = 32
EMB, HIDDEN, CAPACITY, COUNT = 6144, 2048, 1024, 256
NUM_GLOBAL_EXPERTS, NUM_LOCAL_EXPERTS, LOCAL_EXPERT_ID, GLOBAL_EXPERT_ID = 256, 8, 3, 137
GRID = (11, 8)
PCC_GATE = 0.975


def _pcc(a, b):
    a, b = a.flatten().to(torch.float32), b.flatten().to(torch.float32)
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def _configs(device, kind):
    if kind == "preferred":
        return weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID)
    if kind == "tall":
        return weight_memory_configs(device, EMB, HIDDEN, core_grid=GRID, shard_height_tiles=4)
    if kind == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    raise ValueError(kind)


@pytest.mark.parametrize("kind, coalesced", [("preferred", True), ("tall", True), ("interleaved", False)])
def test_placement(device, kind, coalesced):
    torch.manual_seed(42)
    x = torch.randn((1, 1, CAPACITY, EMB), dtype=torch.float32)
    x[:, :, COUNT:, :] = 100.0
    xb = x.to(torch.bfloat16)
    wg, wu = (torch.randn((EMB, HIDDEN), dtype=torch.bfloat16) for _ in range(2))
    wd = torch.randn((HIDDEN, EMB), dtype=torch.bfloat16)

    gu_mc, dn_mc = _configs(device, kind)
    d = lambda t, dt, l, mc: ttnn.from_torch(t, dtype=dt, layout=l, device=device, memory_config=mc)
    tt_x = d(xb, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
    tt_w = [d(w, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, mc) for w, mc in ((wg, gu_mc), (wu, gu_mc), (wd, dn_mc))]

    # The READER's own predicate, not the config we asked for: an unrecognised placement is
    # silently correct and slower, so this is the only thing that distinguishes the two paths.
    widths = [nd_shard_n_tiles(w) for w in tt_w]
    if coalesced:
        assert all(w > 0 for w in widths), f"{kind}: expected a coalescible run, reader sees {widths}"
    else:
        assert all(w == 0 for w in widths), f"{kind}: expected the per-tile stream, reader sees {widths}"

    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = COUNT
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    u = lambda t: d(t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)

    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], u(counts), u(idx), LOCAL_EXPERT_ID, core_grid=GRID)
    got = ttnn.to_torch(out)[0, 0, :COUNT]

    xs = xb[0, 0, :COUNT].to(torch.float32)
    h = torch.nn.functional.silu(xs @ wg.to(torch.float32)) * (xs @ wu.to(torch.float32))
    pcc = _pcc(got, h @ wd.to(torch.float32))
    assert torch.isfinite(got).all(), f"{kind}: non-finite output"
    assert pcc >= PCC_GATE, f"{kind}: pcc {pcc:.6f} < {PCC_GATE}"
    print(f"[placement] {kind:>11} widths={widths} pcc={pcc:.6f}")


def test_preferred_placement_is_a_pure_function_of_k_n_and_grid(device):
    """It must not depend on the runtime token count — that is device-resident and unknowable here.

    It DOES depend on the grid, because the shard width IS the per-core N slice, so a caller that
    shards for one grid and runs on another gets a correct but uncoalesced stream.
    """
    a = weight_memory_configs(device, EMB, HIDDEN, core_grid=(11, 8))
    b = weight_memory_configs(device, EMB, HIDDEN, core_grid=(11, 8))
    assert [str(m) for m in a] == [str(m) for m in b], "not deterministic for one (K, N, grid)"

    wide = weight_memory_configs(device, EMB, HIDDEN, core_grid=(8, 8))
    assert str(wide[0]) != str(a[0]), "the gate/up shard width must follow the COLUMN count"


def test_preferred_placement_defaults_to_the_full_device_grid(device):
    grid = device.compute_with_storage_grid_size()
    default = weight_memory_configs(device, EMB, HIDDEN)
    explicit = weight_memory_configs(device, EMB, HIDDEN, core_grid=(int(grid.x), int(grid.y)))
    assert [str(memory_config) for memory_config in default] == [str(memory_config) for memory_config in explicit]
