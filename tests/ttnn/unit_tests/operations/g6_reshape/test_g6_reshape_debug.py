# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_reshape debug checks that the acceptance test cannot see by construction."""

import torch
import ttnn

from ttnn.operations.g6_reshape import describe_plan, g6_reshape

L1_OP_BUDGET = 1347776


def _to_device(x, device):
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _bits_equal(got, expected):
    return bool((got.contiguous().view(torch.int16) == expected.contiguous().view(torch.int16)).all())


def test_memo_hit_uses_live_addresses(device):
    """Two live inputs and two live outputs of one signature: a descriptor-cache hit must not replay the
    first call's addresses."""
    in_shape, out_shape = (16, 1, 64), [1, 1, 16, 64]
    torch.manual_seed(0)
    xa = torch.randn(in_shape).to(torch.bfloat16)
    xb = torch.randn(in_shape).to(torch.bfloat16)
    ta, tb = _to_device(xa, device), _to_device(xb, device)
    assert ta.buffer_address() != tb.buffer_address()
    ya = g6_reshape(ta, out_shape)
    yb = g6_reshape(tb, out_shape)
    assert ya.buffer_address() != yb.buffer_address()
    assert _bits_equal(ttnn.to_torch(ya), xa.reshape(out_shape))
    assert _bits_equal(ttnn.to_torch(yb), xb.reshape(out_shape))


def test_position_encoding_element_run(device):
    """Each value encodes its linear index, so any misrouted element shows up as a wrong integer."""
    in_shape, out_shape = (1, 185), [1, 1, 185, 1]
    x = torch.arange(185, dtype=torch.float32).reshape(in_shape).to(torch.bfloat16)
    y = ttnn.to_torch(g6_reshape(_to_device(x, device), out_shape))
    assert _bits_equal(y, x.reshape(out_shape))


def test_plan_report_production(device):
    """Plan report for the production shapes; printed for inspection with -s."""
    cases = [
        ((1, 16, 6, 256), [1, 1, 16, 1536]),
        ((16, 1, 64), [1, 1, 16, 64]),
        ((1, 1, 16, 4608), [16, 1, 4608]),
        ((16, 1, 5120), [1, 1, 16, 5120]),
        ((2, 1, 5120), [1, 1, 2, 5120]),
        ((1, 1, 6, 256), [1, 1, 1536]),
        ((1, 185), [1, 1, 185, 1]),
    ]
    for in_shape, out_shape in cases:
        plan = describe_plan(_to_device(torch.zeros(in_shape, dtype=torch.bfloat16), device), out_shape)
        movers = [r for r in plan["roles"] if r["atom_count"] > 0]
        # A core may split its atoms' face-column halves over its movers (Perf 1): count half-atoms.
        assert sum(r["atom_count"] * (r["half_hi"] - r["half_lo"]) for r in plan["roles"]) == 2 * plan["atoms_total"]
        print(
            in_shape,
            "->",
            out_shape,
            {
                k: plan[k]
                for k in (
                    "regime",
                    "atoms_total",
                    "num_cores",
                    "block_atoms",
                    "src_face_rows_bound",
                    "l1_bytes_per_core",
                )
            },
            "active movers",
            len(movers),
        )
        assert plan["l1_bytes_per_core"] <= L1_OP_BUDGET
