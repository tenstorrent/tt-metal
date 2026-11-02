# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Precision and core-grid policy. No device required."""

import pytest

import ttnn
from models.experimental.modernbert.tt.model_config import (
    _GELU_APPROX,
    ACTIVATIONS_DTYPE,
    LINEAR_WEIGHTS_DTYPE,
    WEIGHTS_DTYPE,
    compute_kernel_config,
    mlp_shard_plan,
    mlp_up_projection_program_config,
    qkv_matmul_program_config,
    select_down_projection_grid,
)


class _Device:
    """Stands in for a device so grid-dependent policy is testable off-hardware."""

    def __init__(self, x=8, y=8):
        self._grid = type("G", (), {"x": x, "y": y})()

    def compute_with_storage_grid_size(self):
        return self._grid


def test_down_projection_grid_on_full_grid():
    grid = select_down_projection_grid(8, 8)
    assert grid is not None
    assert (grid.y, grid.x) == (8, 8)


@pytest.mark.parametrize("gx,gy", [(7, 8), (8, 7), (4, 8), (1, 1)])
def test_down_projection_grid_falls_back_on_small_grid(gx, gy):
    """Only 8x8 was tuned; anything smaller must defer to ttnn's own choice."""
    assert select_down_projection_grid(gx, gy) is None


def test_larger_grid_still_uses_tuned_8x8():
    """A bigger device should not silently get an untuned grid."""
    grid = select_down_projection_grid(16, 16)
    assert (grid.y, grid.x) == (8, 8)


def test_matmul_weights_are_bfloat8_b():
    """The measured choice: -8.9% at b1s256 for 2.6e-3 of PCC."""
    assert LINEAR_WEIGHTS_DTYPE == ttnn.bfloat8_b


def test_embedding_and_norm_weights_stay_bfloat16():
    """Not a tuning choice - bfloat8_b cannot represent either tensor."""
    assert WEIGHTS_DTYPE == ttnn.bfloat16


def test_mask_dtype_is_bfloat16():
    """ACTIVATIONS_DTYPE is the attention mask dtype, despite the name."""
    assert ACTIVATIONS_DTYPE == ttnn.bfloat16


def test_fp32_accumulation_stays_enabled():
    assert compute_kernel_config().fp32_dest_acc_en is True


def test_fidelity_is_hifi3_not_hifi4():
    """tt-metal warns that HiFi4 with fp32 accumulation can be less accurate than
    HiFi3 on Wormhole due to a hardware bug, and end-to-end measurement agrees at
    seq 256. HiFi4 also runs slower. HiFi2 is not a safe substitute: it drops
    end-to-end PCC to 0.99436858 at seq 256 against HiFi3's 0.99844078.
    """
    assert compute_kernel_config().math_fidelity == ttnn.MathFidelity.HiFi3


@pytest.mark.parametrize(
    "batch,seq_len,expected_in0_block_w",
    [(1, 256, 8), (1, 512, 8), (1, 768, 8), (2, 256, 8), (4, 256, 8), (8, 256, 8), (1, 1024, 8)],
)
def test_qkv_program_config_keys_on_batch_times_sequence(batch, seq_len, expected_in0_block_w):
    """The matmul flattens batch into M and enforces
    num_blocks_y = (batch * M_tiles) / per_core_M <= grid rows, so the config must
    be derived from batch * seq_len. Batch 2 at seq 256 must therefore match the
    batch 1 seq 512 configuration.
    """

    cfg = qkv_matmul_program_config(_Device(), batch, seq_len, 768)
    assert cfg is not None
    assert cfg.in0_block_w == expected_in0_block_w
    assert cfg.per_core_M == (batch * seq_len) // 32 // 8
    assert cfg.per_core_N == (3 * 768) // 32 // 8
    assert cfg.out_subblock_h * cfg.out_subblock_w <= 4


@pytest.mark.parametrize("batch,seq_len", [(1, 128), (1, 300), (1, 1280), (1, 1536), (16, 256)])
def test_qkv_program_config_declines_unmeasured_shapes(batch, seq_len):
    """Genuinely unmeasured shapes still fall back to ttnn's automatic choice."""

    assert qkv_matmul_program_config(_Device(), batch, seq_len, 768) is None


def test_qkv_program_config_declines_small_grid():
    assert qkv_matmul_program_config(_Device(x=7, y=8), 1, 256, 768) is None


def test_fused_gelu_matches_the_declared_precision_policy():
    """param0 is the template argument to gelu_tile<N>(): 0 is the exact erf form,
    1 the tanh approximation. The model ships the approximation - see the GELU
    section of the model_config docstring - because the exact erf is
    SFPU-throughput-bound and costs 11% of the pass for 1.66e-03 of MLM PCC.
    """
    cfg = mlp_up_projection_program_config(_Device(), 1, 256, 768, 1152, fuse_gelu=True)
    assert cfg is not None
    act = cfg.fused_activation
    assert act is not None
    assert act.op_type == ttnn.UnaryOpType.GELU
    # UnaryWithParam exposes only op_type as an attribute; the parameter is
    # reachable only through repr, which renders it as params=[0] or params=[1].
    expected = f"params=[{int(_GELU_APPROX)}]"
    assert expected in repr(act), f"fused GELU must match _GELU_APPROX, got {act!r}"


def test_gate_half_carries_no_activation():
    """GeGLU activates only the first half. Applying the activation to the gate as
    well is silent - the shapes are identical and the output is merely wrong.
    """
    cfg = mlp_up_projection_program_config(_Device(), 1, 256, 768, 1152, fuse_gelu=False)
    assert cfg is not None
    assert cfg.fused_activation is None


@pytest.mark.parametrize("batch,seq_len,expected_subblock_w", [(1, 256, 2), (1, 512, 2), (1, 768, 3), (2, 256, 2)])
def test_mlp_program_config_geometry(batch, seq_len, expected_subblock_w):
    """1152 is 36 tiles and 36 is not divisible by 8, so the grid must be 6 wide,
    not 8, for per_core_N to be an integer.
    """
    cfg = mlp_up_projection_program_config(_Device(), batch, seq_len, 768, 1152, fuse_gelu=True)
    assert cfg is not None
    assert cfg.per_core_M == (batch * seq_len) // 32 // 8
    assert cfg.per_core_N == 1152 // 32 // 6
    assert cfg.out_subblock_w == expected_subblock_w
    assert cfg.out_subblock_h * cfg.out_subblock_w <= 4


@pytest.mark.parametrize("batch,seq_len", [(1, 128), (1, 300), (4, 256)])
def test_mlp_program_config_declines_unmeasured_shapes(batch, seq_len):
    assert mlp_up_projection_program_config(_Device(), batch, seq_len, 768, 1152, fuse_gelu=True) is None


def test_mlp_program_config_declines_narrow_grid():
    """A 4-wide grid cannot express per_core_N for 36 tiles across 6 columns."""
    assert mlp_up_projection_program_config(_Device(x=4, y=8), 1, 256, 768, 1152, fuse_gelu=True) is None


def test_mlp_shard_plan_declines_below_the_threshold():
    """Sharding loses badly when each core holds only a few tiles: at b1s256 the
    activation is 8x24 tiles over 48 cores, and the block measured 58.8% slower
    than interleaved. The gate exists to keep those shapes off the sharded path.
    """
    for batch, seq in ((1, 256), (1, 512), (2, 256)):
        assert mlp_shard_plan(_Device(), batch, seq, 768, 1152) is None, f"b{batch}s{seq} should not shard"


@pytest.mark.parametrize("batch,seq", [(1, 768), (4, 256), (8, 256)])
def test_mlp_shard_plan_engages_at_or_above_the_threshold(batch, seq):
    plan = mlp_shard_plan(_Device(), batch, seq, 768, 1152)
    assert plan is not None
    m_t = (batch * seq) // 32
    # one 6-wide grid for the whole block: 1152 is 36 tiles and 36/8 is not an
    # integer, which the allocator rejects as 72 shards on 64 banks
    for cfg in (plan.act_matmul, plan.gate_matmul, plan.down_matmul):
        grid = cfg.compute_with_storage_grid_size
        assert (grid.x, grid.y) == (6, 8)
        assert cfg.per_core_M == m_t // 8
        assert cfg.out_subblock_h * cfg.out_subblock_w <= 4


def test_mlp_shard_plan_activates_only_the_act_half():
    """The fused GELU belongs to Wi_act alone; the gate half must stay linear."""
    plan = mlp_shard_plan(_Device(), 4, 256, 768, 1152)
    assert f"params=[{int(_GELU_APPROX)}]" in repr(
        plan.act_matmul.fused_activation
    ), "sharded act matmul must match _GELU_APPROX, like the interleaved one"
    assert plan.gate_matmul.fused_activation is None
    assert plan.down_matmul.fused_activation is None


def test_mlp_shard_plan_declines_narrow_grid():
    assert mlp_shard_plan(_Device(x=4, y=8), 4, 256, 768, 1152) is None
