# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bit parity of Laguna's routed-expert generic_op (tt/routed_expert.py) with the C++ op it ports,
ttnn.experimental.deepseek_prefill.unified_routed_expert_moe_stacked (only on branches that still carry that op; the
test skips once the shared ttnn changes are reverted).

Same Laguna-S inputs as test_routed_expert.py (1 x D mesh, 64 local experts, 3072 / 1024, top-k 10, BFP4 stacked
weights, row-major bf16 dispatch buffer, per-chip routing with zero-count experts, counts that are not multiples of
32 and one expert at full capacity), over the dispatch buckets that select each row-group plan (128 .. 2048: 1-row
groups, 4096: 2-row, 8192: 5-row on a 10-row grid), both weight layouts Laguna can hand the op (column pages, the
default; DRAM interleaved with TT_LAGUNA_COLPAGE=0), and l1_end 0 or bound by a resident L1 tensor.

What is compared: the op writes, per local expert, every tile-row of its region below ceil(count / 32); rows of the
last tile-row past count come from padding sticks the reader never reads (stale L1, different in two programs), and
rows outside every region are never written (the output is an uninitialized allocation). So every row a token owns,
[region[g], region[g] + counts[g]) for each local expert g, must be bit-identical (torch.equal on the bf16 bit
patterns); nothing else is defined.
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.laguna.tests.test_routed_expert import (
    LOCAL_EXPERTS,
    build_dispatch,
    build_weights,
    chip_to_torch,
    expert_rows,
    global_expert_idx_table,
    open_mesh,
)
from models.demos.laguna.tt.l1_budget import lowest_occupied_l1_address
from models.demos.laguna.tt.routed_expert import routed_expert_plan, routed_expert_stacked

_CPP_OP = getattr(getattr(ttnn.experimental, "deepseek_prefill", None), "unified_routed_expert_moe_stacked", None)
pytestmark = pytest.mark.skipif(
    _CPP_OP is None,
    reason="ttnn.experimental.deepseek_prefill.unified_routed_expert_moe_stacked is not in this ttnn build (the shared "
    "op's Laguna stacked mode was reverted); nothing to compare the port against",
)

SEQ_LENS = [128, 512, 1024, 2048, 4096, 8192]


@pytest.fixture(scope="module")
def mesh():
    m = open_mesh()
    yield m
    ttnn.close_mesh_device(m)


@pytest.fixture(scope="module")
def weights(mesh):
    """Both weight layouts, built once (the same values: one seed)."""
    built = {}

    def get(column_pages):
        if column_pages not in built:
            for other in list(built):  # keep one layout resident at a time
                gu, dn, _ = built.pop(other)
                ttnn.deallocate(gu)
                ttnn.deallocate(dn)
            built[column_pages] = build_weights(mesh, column_pages=column_pages)
        return built[column_pages]

    yield get
    for gu, dn, _ in built.values():
        ttnn.deallocate(gu)
        ttnn.deallocate(dn)


def _resident_l1(mesh, plan0):
    """An interleaved L1 tensor sized so the CBs' l1_end budget falls ~96 KB below the unconstrained plan's CB
    footprint, which forces a narrower gate/up K-block (or a smaller per_core_M) in both programs."""
    from models.demos.laguna.tt.routed_expert import _device_info

    _, _, l1_size, l1_base, _ = _device_info(mesh)
    l1_banks = int(ttnn.get_memory_view(mesh, ttnn.BufferType.L1).num_banks)
    target_end = l1_base + 48 * 1024 + plan0["cb_bytes"] - 96 * 1024
    per_bank = l1_size - target_end
    tiles = l1_banks * -(-per_bank // 2048)
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * tiles]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.L1_MEMORY_CONFIG
    )


# One layout at a time (the weights fixture keeps one resident): column pages first, then interleaved.
CASES = [
    pytest.param(cp, seq_len, bound, id=f"{'colpage' if cp else 'dram'}-{seq_len}-{'bound' if bound else 'l1end0'}")
    for cp in (True, False)
    for seq_len in SEQ_LENS
    for bound in (False, True)
]


@pytest.mark.parametrize("column_pages, seq_len, bind_l1_end", CASES)
def test_routed_expert_parity(mesh, weights, seq_len, column_pages, bind_l1_end, monkeypatch):
    # The C++ factory honours TT_ROUTED_EXPERT_ROW_GRID_Y; the port dropped it, so it must be unset for parity.
    monkeypatch.delenv("TT_ROUTED_EXPERT_ROW_GRID_Y", raising=False)
    D = mesh.get_num_devices()
    gate_up, down, _ = weights(column_pages)
    idx, idx_host = global_expert_idx_table(mesh)
    dev, host = build_dispatch(mesh, seq_len, idx_host, seed=seq_len)
    args = (dev["x"], dev["region"], dev["counts"], idx, gate_up, down, seq_len)

    resident = None
    l1_end = 0
    plan0 = routed_expert_plan(*args, l1_end=0)
    if bind_l1_end:
        assert lowest_occupied_l1_address(mesh) is None, "another L1 buffer is resident; l1_end would be ambiguous"
        resident = _resident_l1(mesh, plan0)
        l1_end = lowest_occupied_l1_address(mesh) or 0
        assert l1_end, "the resident L1 tensor did not register as allocated L1"
    plan = routed_expert_plan(*args, l1_end=l1_end)
    if bind_l1_end:
        assert (plan["per_core_M"], plan["in0_block_w_gu"]) != (plan0["per_core_M"], plan0["in0_block_w_gu"]), (
            f"l1_end {l1_end} did not change the CB plan ({plan['per_core_M']}, {plan['in0_block_w_gu']}); the "
            "bound case would test nothing"
        )
    try:
        ref = _CPP_OP(*args, compute_kernel_config=None, l1_end=l1_end)
        new = routed_expert_stacked(*args, l1_end=l1_end)
        assert list(new.shape) == list(ref.shape) and new.dtype == ref.dtype and new.layout == ref.layout
        assert new.memory_config() == ref.memory_config()
        checked = 0
        for d in range(D):
            ref_chip, new_chip = chip_to_torch(ref, d), chip_to_torch(new, d)
            for e in range(LOCAL_EXPERTS):
                gid = int(idx_host[d, e])
                s, n = expert_rows(host, d, gid)
                if n == 0:
                    continue
                # compare bit patterns (bf8 values are exact in bf16), so a NaN at the same place also matches
                a, b = ref_chip[s : s + n].view(torch.int16), new_chip[s : s + n].view(torch.int16)
                if not torch.equal(a, b):
                    diff = (a != b).any(dim=1).nonzero().flatten()
                    raise AssertionError(
                        f"seq_len {seq_len} chip {d} local expert {e} (global {gid}, rows {s}..{s + n}): "
                        f"{diff.numel()} of {n} rows differ, first at row {s + int(diff[0])}; plan row_group "
                        f"{plan['row_group']} per_core_M {plan['per_core_M']} in0_block_w_gu {plan['in0_block_w_gu']}"
                    )
                checked += n
        print(
            f"seq_len {seq_len} colpage {column_pages} l1_end {l1_end}: row_group {plan['row_group']} per_core_M "
            f"{plan['per_core_M']} in0_block_w_gu {plan['in0_block_w_gu']}: {checked} token rows bit-identical"
        )
        ttnn.deallocate(ref)
        ttnn.deallocate(new)
    finally:
        if resident is not None:
            ttnn.deallocate(resident)
        for t in (dev["x"], dev["counts"], dev["region"], idx):
            ttnn.deallocate(t)
