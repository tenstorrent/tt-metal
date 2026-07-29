# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Static contracts for batch-32 dense-expert prefill sweep plumbing."""

import math
from dataclasses import replace

import pytest

from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import (
    OptimizationConfig,
    _dense_expert_program,
    _needs_packed_dense_expert_weights,
    _use_packed_dense_experts,
)


def _split_m1024_policy():
    return replace(
        OptimizationConfig(),
        prefill_packed_dense_experts=False,
        dense_expert_prefill_gate_up_grid=(8, 8),
        dense_expert_prefill_gate_up_in0_block_w=8,
        dense_expert_prefill_gate_up_per_core_m=4,
        dense_expert_prefill_gate_up_per_core_n=3,
        dense_expert_prefill_gate_up_out_block_h=4,
        dense_expert_prefill_gate_up_out_block_w=3,
        dense_expert_prefill_gate_up_subblock_h=2,
        dense_expert_prefill_gate_up_subblock_w=3,
        dense_expert_prefill_down_grid=(10, 8),
        dense_expert_prefill_down_in0_block_w=6,
        dense_expert_prefill_down_per_core_m=4,
        dense_expert_prefill_down_per_core_n=7,
        dense_expert_prefill_down_out_block_h=4,
        dense_expert_prefill_down_out_block_w=7,
        dense_expert_prefill_down_subblock_h=1,
        dense_expert_prefill_down_subblock_w=7,
    )


def test_dense_expert_prefill_defaults_select_packed_80_w8_6_only_for_prefill():
    policy = OptimizationConfig()
    gate = _dense_expert_program(policy, down=False, prefill=True)
    down = _dense_expert_program(policy, down=True, prefill=True)

    assert (gate.compute_with_storage_grid_size.x, gate.compute_with_storage_grid_size.y) == (10, 8)
    assert (gate.in0_block_w, gate.per_core_M, gate.per_core_N) == (8, 4, 5)
    assert (gate.out_block_h, gate.out_block_w) == (4, 5)
    assert (gate.out_subblock_h, gate.out_subblock_w) == (1, 5)
    assert (down.compute_with_storage_grid_size.x, down.compute_with_storage_grid_size.y) == (10, 8)
    assert (down.in0_block_w, down.per_core_M, down.per_core_N) == (6, 4, 7)
    assert (down.out_block_h, down.out_block_w) == (4, 7)
    assert (down.out_subblock_h, down.out_subblock_w) == (1, 7)
    assert _use_packed_dense_experts(policy, phase="prefill")
    assert not _use_packed_dense_experts(policy, phase="decode")
    assert _needs_packed_dense_expert_weights(policy)
    assert _dense_expert_program(policy, down=False) is None
    assert _dense_expert_program(policy, down=True) is None


def test_dense_expert_prefill_without_override_preserves_legacy_opt_in_program():
    policy = replace(
        OptimizationConfig(),
        dense_expert_prefill_gate_up_grid=(0, 0),
        dense_expert_gate_up_in0_block_w=8,
        dense_expert_gate_up_per_core_m=4,
        dense_expert_gate_up_per_core_n=3,
        dense_expert_gate_up_subblock_h=2,
        dense_expert_gate_up_subblock_w=3,
    )
    prefill = _dense_expert_program(policy, down=False, prefill=True)
    legacy = _dense_expert_program(policy, down=False)
    assert repr(prefill) == repr(legacy)


def test_dense_expert_prefill_m1024_split_programs_are_phase_specific():
    policy = _split_m1024_policy()
    gate = _dense_expert_program(policy, down=False, prefill=True)
    down = _dense_expert_program(policy, down=True, prefill=True)

    assert (gate.compute_with_storage_grid_size.x, gate.compute_with_storage_grid_size.y) == (8, 8)
    assert (gate.in0_block_w, gate.per_core_M, gate.per_core_N) == (8, 4, 3)
    assert (gate.out_block_h, gate.out_block_w) == (4, 3)
    assert (gate.out_subblock_h, gate.out_subblock_w) == (2, 3)
    assert (down.compute_with_storage_grid_size.x, down.compute_with_storage_grid_size.y) == (10, 8)
    assert (down.in0_block_w, down.per_core_M, down.per_core_N) == (6, 4, 7)
    assert (down.out_block_h, down.out_block_w) == (4, 7)
    assert (down.out_subblock_h, down.out_subblock_w) == (1, 7)

    # Decode retains its separately swept reuse-program contract.
    assert _dense_expert_program(policy, down=False, prefill=False) is None
    assert _dense_expert_program(policy, down=True, prefill=False) is None
    assert not _use_packed_dense_experts(policy, phase="prefill")
    assert not _use_packed_dense_experts(policy, phase="decode")


def test_dense_expert_prefill_rejects_partial_geometry(expect_error):
    policy = replace(OptimizationConfig(), dense_expert_prefill_gate_up_in0_block_w=0)
    with expect_error(ValueError, "incomplete dense_expert_prefill_gate_up geometry"):
        _dense_expert_program(policy, down=False, prefill=True)


def test_legacy_global_packed_candidate_still_packs_both_phases():
    policy = replace(
        OptimizationConfig(),
        prefill_packed_dense_experts=False,
        packed_dense_experts=True,
    )
    assert _use_packed_dense_experts(policy, phase="prefill")
    assert _use_packed_dense_experts(policy, phase="decode")
    assert _needs_packed_dense_expert_weights(policy)


@pytest.mark.parametrize(
    "candidate,packed,gate,down",
    (
        (
            "split_active_w8_6",
            False,
            ((8, 8), False, 8, 4, 3, 4, 3, 2, 3, 24),
            ((10, 8), False, 6, 4, 7, 4, 7, 1, 7, 64),
        ),
        (
            "split_88_w8_6",
            False,
            ((11, 8), True, 8, 3, 3, 3, 3, 1, 3, 24),
            ((11, 8), False, 6, 4, 6, 4, 6, 2, 3, 64),
        ),
        (
            "packed_80_w8_6",
            True,
            ((10, 8), False, 8, 4, 5, 4, 5, 1, 5, 48),
            ((10, 8), False, 6, 4, 7, 4, 7, 1, 7, 64),
        ),
    ),
)
def test_dense_expert_prefill_candidate_geometry_is_statically_legal(candidate, packed, gate, down):
    del candidate
    fields = {}
    for prefix, spec, k_tiles in (
        ("dense_expert_prefill_gate_up", gate, 64),
        ("dense_expert_prefill_down", down, 24),
    ):
        grid, transpose, in0_block_w, per_core_m, per_core_n, out_h, out_w, sub_h, sub_w, n_tiles = spec
        fields.update(
            {
                f"{prefix}_grid": grid,
                f"{prefix}_transpose_mcast": transpose,
                f"{prefix}_in0_block_w": in0_block_w,
                f"{prefix}_per_core_m": per_core_m,
                f"{prefix}_per_core_n": per_core_n,
                f"{prefix}_out_block_h": out_h,
                f"{prefix}_out_block_w": out_w,
                f"{prefix}_subblock_h": sub_h,
                f"{prefix}_subblock_w": sub_w,
            }
        )

        assert k_tiles % in0_block_w == 0
        assert per_core_m % out_h == 0
        assert per_core_n % out_w == 0
        assert out_h % sub_h == 0
        assert out_w % sub_w == 0
        assert sub_h * sub_w <= 8
        m_blocks = math.ceil(32 / per_core_m)
        n_blocks = math.ceil(n_tiles / per_core_n)
        required_grid = (m_blocks, n_blocks) if transpose else (n_blocks, m_blocks)
        assert required_grid[0] <= grid[0]
        assert required_grid[1] <= grid[1]

    policy = replace(
        OptimizationConfig(),
        prefill_packed_dense_experts=packed,
        packed_dense_experts=False,
        **fields,
    )
    assert _dense_expert_program(policy, down=False, prefill=True) is not None
    assert _dense_expert_program(policy, down=True, prefill=True) is not None
    assert _use_packed_dense_experts(policy, phase="prefill") is packed
    assert not _use_packed_dense_experts(policy, phase="decode")
