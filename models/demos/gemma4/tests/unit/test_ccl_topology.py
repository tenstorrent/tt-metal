# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for Gemma4 CCL topology / async / L1 env knobs."""

from pathlib import Path

import pytest

import ttnn
from models.demos.gemma4.tt.attention.operations import (
    PREFILL_SDPA_HARD_MAX,
    PREFILL_SDPA_MAX_SEQ,
    prefill_short_lived_memcfg,
    prefill_tensor_memcfg,
    prefill_tilize_memcfg,
)
from models.demos.gemma4.tt.ccl import ccl_async_enabled, default_ccl_topology
from models.demos.gemma4.tt.dram_sharded import can_dram_shard


@pytest.mark.parametrize(
    "env,expected",
    [
        ("ring", ttnn.Topology.Ring),
        ("linear", ttnn.Topology.Linear),
        ("LINE", ttnn.Topology.Linear),
    ],
)
def test_ccl_topology_env_override(monkeypatch, env, expected):
    monkeypatch.setenv("GEMMA4_CCL_TOPOLOGY", env)
    assert default_ccl_topology() == expected


class _FakeMesh:
    def __init__(self, n):
        self._n = n

    def get_num_devices(self):
        return self._n


def test_ccl_topology_linear_on_4_device_mesh(monkeypatch):
    """QB2 / P300x2 opened as 1x4: Ring drops 12B full-model PCC below 0.94."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert default_ccl_topology(_FakeMesh(4)) == ttnn.Topology.Linear


def test_ccl_topology_ring_on_bh_8_device_mesh(monkeypatch):
    """LoudBox P150x8: Ring remains the TTFT-swept default on Blackhole."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert default_ccl_topology(_FakeMesh(8)) == ttnn.Topology.Ring


def test_ccl_topology_ring_on_wh_8_device_mesh_dense(monkeypatch):
    """WH T3K 1x8 dense (31B): Ring is the decode-swept default (is_moe=False)."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    assert default_ccl_topology(_FakeMesh(8), is_moe=False) == ttnn.Topology.Ring


def test_ccl_topology_linear_on_wh_8_device_mesh_moe(monkeypatch):
    """WH T3K 1x8 MoE (26B-A4B): Ring drops full-model PCC; keep Linear."""
    monkeypatch.delenv("GEMMA4_CCL_TOPOLOGY", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: False)
    assert default_ccl_topology(_FakeMesh(8), is_moe=True) == ttnn.Topology.Linear


def test_ccl_topology_env_override_beats_device_count(monkeypatch):
    monkeypatch.setenv("GEMMA4_CCL_TOPOLOGY", "ring")
    assert default_ccl_topology(_FakeMesh(4)) == ttnn.Topology.Ring
    monkeypatch.setenv("GEMMA4_CCL_TOPOLOGY", "linear")
    monkeypatch.setattr("models.demos.gemma4.tt.ccl.is_blackhole", lambda: True)
    assert default_ccl_topology(_FakeMesh(8)) == ttnn.Topology.Linear


def test_ccl_async_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_CCL_ASYNC", raising=False)
    monkeypatch.delenv("GEMMA4_CCL_ASYNC_PREFILL", raising=False)
    assert ccl_async_enabled() is False
    assert ccl_async_enabled(32) is False
    assert ccl_async_enabled(2048) is True
    monkeypatch.setenv("GEMMA4_CCL_ASYNC_PREFILL", "0")
    assert ccl_async_enabled(2048) is False
    monkeypatch.delenv("GEMMA4_CCL_ASYNC_PREFILL", raising=False)
    monkeypatch.setenv("GEMMA4_CCL_ASYNC", "1")
    assert ccl_async_enabled() is True
    assert ccl_async_enabled(32) is True
    monkeypatch.setenv("GEMMA4_CCL_ASYNC", "0")
    assert ccl_async_enabled(2048) is False


def test_fabric_router_clamps_wh_packet_bytes(monkeypatch):
    """WH Fabric max payload is 7616; oversized env overrides must not FATAL open."""
    from models.common.utility_functions import is_blackhole
    from models.demos.gemma4.tt.ccl import fabric_router_config_from_env

    if is_blackhole():
        pytest.skip("WH-only packet ceiling")
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-31B-it")
    monkeypatch.setenv("GEMMA4_CCL_PACKET_BYTES", "8192")
    router = fabric_router_config_from_env()
    assert router is not None
    assert router.max_packet_payload_size_bytes == 7616


def test_prefill_l1_act_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_PREFILL_L1_ACT", raising=False)
    assert prefill_short_lived_memcfg() == ttnn.DRAM_MEMORY_CONFIG
    monkeypatch.setenv("GEMMA4_PREFILL_L1_ACT", "1")
    assert prefill_short_lived_memcfg() == ttnn.L1_MEMORY_CONFIG


def test_prefill_tensor_memcfg_size_budget(monkeypatch):
    """Short activations → L1; over budget / disabled → DRAM."""
    monkeypatch.delenv("GEMMA4_PREFILL_L1_TENSOR_MAX_BYTES", raising=False)
    # 128 x 5376 BF16 ≈ 1.3 MiB < 4 MiB default
    assert prefill_tilize_memcfg(128, 5376) == ttnn.L1_MEMORY_CONFIG
    # RoPE slice 128 x 256 BF16 ≈ 64 KiB
    assert prefill_tensor_memcfg(128 * 256) == ttnn.L1_MEMORY_CONFIG
    # 512 x 5376 BF16 ≈ 5.3 MiB > 4 MiB
    assert prefill_tilize_memcfg(512, 5376) == ttnn.DRAM_MEMORY_CONFIG
    monkeypatch.setenv("GEMMA4_PREFILL_L1_TENSOR_MAX_BYTES", "0")
    assert prefill_tilize_memcfg(128, 5376) == ttnn.DRAM_MEMORY_CONFIG


def test_prefill_sdpa_max_seq_clamped_to_hard_max():
    """Env override must not raise the non-chunked SDPA path past 2^15."""
    assert PREFILL_SDPA_MAX_SEQ <= PREFILL_SDPA_HARD_MAX


def test_shared_mlp_down_shard_unguarded_at_tp8(monkeypatch):
    """Unpadded intermediate=2112 @ TP=8 → down_k=264 is not DRAM-shardable.

    SharedMLP now pads to 288/device before sharding; this guards the raw shape.
    """
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    assert can_dram_shard(2816, 528)  # gate_up n at tp=8 (unpadded half*2)
    assert not can_dram_shard(264, 2816)  # raw down_k
    assert can_dram_shard(288, 2816)  # padded down_k used by SharedMLP


def test_dram_shard_disabled_off_blackhole(monkeypatch):
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: False)
    assert not can_dram_shard(2816, 528)


def test_dram_shard_31b_gate_up_fits_with_in0_cap(monkeypatch):
    """31B fused gate_up @ TP=4 previously overflowed L1 at in0_block_w=6; cap=2 fits."""
    monkeypatch.setattr("models.demos.gemma4.tt.dram_sharded.is_blackhole", lambda: True)
    import ttnn

    # hidden=5376, gu_n=2*21504/4=10752
    assert can_dram_shard(5376, 10752, dtype=ttnn.bfloat16)
    assert can_dram_shard(5376, 10752, dtype=ttnn.bfloat8_b)


def test_prefill_progcfg_in0_block_w_divides_kt():
    """26B padded down_proj K=288 → Kt=9; in0_block_w must divide Kt."""
    from models.demos.gemma4.tt.dram_sharded import prefill_progcfg

    pc = prefill_progcfg(m=512, k=288, n=2816)
    k_tiles = (288 + 31) // 32
    assert k_tiles % pc.in0_block_w == 0


def test_gate_up_prefill_progcfg_1d_matches_sweep_winner():
    """31B fused gate+up @ TP=8 → 1d_c42_bw4 (test_gate_up_matmul_sweep overall winner)."""
    import ttnn
    from models.demos.gemma4.tt.dram_sharded import interleaved_gate_up_prefill_config, prefill_progcfg_1d

    # M=128 K=5376 N=5376 (2 * intermediate/TP)
    pc = prefill_progcfg_1d(m=128, k=5376, n=5376)
    assert pc is not None
    assert pc.in0_block_w == 4
    assert pc.per_core_M == 4  # 128/32
    assert pc.per_core_N == 4  # 168 N-tiles / 42 cores
    grid = pc.compute_with_storage_grid_size
    assert grid.x * grid.y == 42

    prog, out_mc, ckc = interleaved_gate_up_prefill_config(128, 5376, 5376)
    assert prog is not None
    assert out_mc == ttnn.L1_MEMORY_CONFIG
    assert ckc.math_fidelity == ttnn.MathFidelity.HiFi4
    # Decode / long-context: no override
    assert interleaved_gate_up_prefill_config(32, 5376, 5376) == (None, None, None)
    assert interleaved_gate_up_prefill_config(2048, 5376, 5376) == (None, None, None)


def test_down_proj_prefill_progcfg_1d_matches_sweep_winner():
    """31B SharedMLP down_proj @ TP=8 → 1d_c42_bw4 (test_down_proj_matmul_sweep overall winner)."""
    import ttnn
    from models.demos.gemma4.tt.dram_sharded import interleaved_down_proj_prefill_config, prefill_progcfg_1d

    # M=128 K=2688 N=5376 (intermediate/TP, hidden)
    pc = prefill_progcfg_1d(m=128, k=2688, n=5376)
    assert pc is not None
    assert pc.in0_block_w == 4
    assert pc.per_core_M == 4
    assert pc.per_core_N == 4
    grid = pc.compute_with_storage_grid_size
    assert grid.x * grid.y == 42

    prog, out_mc, ckc = interleaved_down_proj_prefill_config(128, 2688, 5376)
    assert prog is not None
    assert out_mc == ttnn.L1_MEMORY_CONFIG
    assert ckc.math_fidelity == ttnn.MathFidelity.HiFi4
    assert interleaved_down_proj_prefill_config(32, 2688, 5376) == (None, None, None)
    assert interleaved_down_proj_prefill_config(2048, 2688, 5376) == (None, None, None)


def test_prefill_matmul_lofi_env(monkeypatch):
    """LoFi tall prefill is opt-in: off unless GEMMA4_PREFILL_MATMUL_LOFI=1.

    Default-off is a correctness guardrail, not a preference -- see
    ``prefill_matmul_lofi_enabled``. Do not relax this assertion.
    """
    from models.demos.gemma4.tt.dram_sharded import _PREFILL_CUTOFF, prefill_matmul_lofi_enabled

    monkeypatch.delenv("GEMMA4_PREFILL_MATMUL_LOFI", raising=False)
    assert not prefill_matmul_lofi_enabled(_PREFILL_CUTOFF)
    assert not prefill_matmul_lofi_enabled(_PREFILL_CUTOFF * 2)
    monkeypatch.setenv("GEMMA4_PREFILL_MATMUL_LOFI", "1")
    assert not prefill_matmul_lofi_enabled(_PREFILL_CUTOFF)
    assert prefill_matmul_lofi_enabled(_PREFILL_CUTOFF * 2)
    monkeypatch.setenv("GEMMA4_PREFILL_MATMUL_LOFI", "0")
    assert not prefill_matmul_lofi_enabled(_PREFILL_CUTOFF * 2)


def test_o_proj_prefill_config_l1_in0_block_sharded_out(monkeypatch):
    """31B attention o_proj @ TP=8 → 2d_8x8_bw4 + L1 block-sharded out + HiFi2.

    M=128 K=1024 (num_local_heads*head_dim) N=5376 (hidden). The block-sharded
    output needs 8 shard columns for Nt=168, so the program-config grid must be the
    full ``prefill_grid_default()`` width — not the 7 columns ``_best_prefill_cols``
    prefers — or the matmul TT_FATALs on a shard grid wider than its compute grid.
    """
    import ttnn
    from models.demos.gemma4.tt import dram_sharded
    from models.demos.gemma4.tt.dram_sharded import interleaved_o_proj_prefill_config, prefill_grid_default

    monkeypatch.setattr(dram_sharded, "_OPROJ_TUNED", True)
    prog, out_mc, ckc = interleaved_o_proj_prefill_config(128, 1024, 5376)
    assert prog is not None
    grid_x, grid_y = prefill_grid_default()
    assert prog.compute_with_storage_grid_size.x == grid_x
    assert prog.compute_with_storage_grid_size.y == grid_y
    assert prog.in0_block_w == 4  # Kt=32 / 8 cols
    assert prog.per_core_M == 1
    assert prog.per_core_N == 21  # 168 N-tiles / 8 cols
    assert ckc.math_fidelity == ttnn.MathFidelity.HiFi2

    # Output is L1 block-sharded, one per_core_M x per_core_N block per core.
    assert out_mc.is_sharded()
    assert out_mc.buffer_type == ttnn.BufferType.L1
    assert out_mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    assert tuple(out_mc.shard_spec.shape) == (32, 672)
    box = out_mc.shard_spec.grid.bounding_box().grid_size()
    assert (box.x, box.y) == (8, 4)  # 168 col-tiles / 8, 4 row-tiles / 4
    assert box.x <= grid_x and box.y <= grid_y

    # Decode (M<=32) and long context keep shipped auto.
    assert interleaved_o_proj_prefill_config(32, 1024, 5376) == (None, None, None)
    assert interleaved_o_proj_prefill_config(2048, 1024, 5376) == (None, None, None)


def test_o_proj_tuned_path_is_opt_in(monkeypatch):
    """Default is shipped auto: the interleave-back before CCL costs more than the
    tuned matmul saves (test_o_proj_wired_config_vs_auto: 0.56x auto)."""
    import os

    from models.demos.gemma4.tt import dram_sharded

    if os.environ.get("GEMMA4_OPROJ_TUNED", "0") == "0":
        assert dram_sharded._OPROJ_TUNED is False
    # Deliberately re-assert the disabled behavior even when the env opted in, so
    # this test states the contract rather than the ambient environment.
    monkeypatch.setattr(dram_sharded, "_OPROJ_TUNED", False)
    assert dram_sharded.interleaved_o_proj_prefill_config(128, 1024, 5376) == (None, None, None)


def test_o_proj_input_memcfg_prefers_l1_in_tuned_band(monkeypatch):
    """concat_heads lands in L1 when short prefill o_proj can keep in0 on L1."""
    import ttnn
    from models.demos.gemma4.tt.attention.operations import o_proj_input_memcfg

    monkeypatch.delenv("GEMMA4_PREFILL_L1_TENSOR_MAX_BYTES", raising=False)
    # SDPA output [1, num_local_heads=4, seq, head_dim=256] → concat K=1024.
    assert o_proj_input_memcfg(_FakeTensor([1, 4, 128, 256]), 5376) == ttnn.L1_MEMORY_CONFIG
    # Decode-height and past-cutoff rows fall back to the caller's default.
    assert o_proj_input_memcfg(_FakeTensor([1, 4, 32, 256]), 5376, ttnn.DRAM_MEMORY_CONFIG) == ttnn.DRAM_MEMORY_CONFIG
    assert o_proj_input_memcfg(_FakeTensor([1, 4, 2048, 256]), 5376, ttnn.DRAM_MEMORY_CONFIG) == (
        ttnn.DRAM_MEMORY_CONFIG
    )
    # Batched prefill counts B*S rows, not S.
    assert o_proj_input_memcfg(_FakeTensor([8, 4, 512, 256]), 5376, ttnn.DRAM_MEMORY_CONFIG) == (
        ttnn.DRAM_MEMORY_CONFIG
    )


def test_should_hoist_prefill_matmul_in0_band_and_budget(monkeypatch):
    from models.demos.gemma4.tt.attention.operations import should_hoist_prefill_matmul_in0
    from models.demos.gemma4.tt.dram_sharded import in_prefill_l1_matmul_band

    monkeypatch.delenv("GEMMA4_PREFILL_L1_TENSOR_MAX_BYTES", raising=False)
    assert in_prefill_l1_matmul_band(128)
    assert not in_prefill_l1_matmul_band(32)
    assert should_hoist_prefill_matmul_in0(128, 5376, object())
    assert should_hoist_prefill_matmul_in0(128, 1024, None)
    assert should_hoist_prefill_matmul_in0(32, 5376, object())  # lm_head decode / last-token
    assert not should_hoist_prefill_matmul_in0(32, 5376, None)
    monkeypatch.setenv("GEMMA4_PREFILL_L1_TENSOR_MAX_BYTES", "0")
    assert not should_hoist_prefill_matmul_in0(128, 5376, object())


class _FakeComputeGrid:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeMeshDevice:
    def __init__(self, gx=8, gy=8):
        self._grid = _FakeComputeGrid(gx, gy)

    def compute_with_storage_grid_size(self):
        return self._grid


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class _FakeCclManager:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device


def test_width_shard_spec_decode_matches_prefill_tile():
    """Decode (height=32) and prefill (height=128) share one layout builder."""
    import ttnn
    from models.demos.gemma4.tt.rms_norm import (
        _SHARDED_NORM_MAX_HEIGHT,
        decode_width_shard_memcfg,
        decode_width_shard_spec,
        width_shard_input_memcfg,
        width_shard_spec,
    )

    mesh = _FakeMeshDevice()
    dim = 5376
    decode_mem = decode_width_shard_memcfg(mesh, dim)
    prefill_mem = width_shard_input_memcfg(mesh, dim, 128)
    assert decode_mem is not None
    assert prefill_mem is not None
    assert decode_mem != prefill_mem  # different shard heights

    decode_spec = decode_width_shard_spec(mesh, dim)
    height128_spec = width_shard_spec(mesh, dim, 128)
    assert decode_spec[0] == width_shard_spec(mesh, dim, ttnn.TILE_SIZE)[0]
    assert height128_spec[1].block_h == 4  # 128 / 32
    assert width_shard_input_memcfg(mesh, dim, _SHARDED_NORM_MAX_HEIGHT + 32) is None


def test_prefill_l1_gather_memcfg_matches_norm_layout(monkeypatch):
    """CCL all-gather layout must match RMSNorm._build_sharded_cfg for ISL<=1024."""
    from models.demos.gemma4.tt.ccl import _decode_l1_gather_memcfg
    from models.demos.gemma4.tt.rms_norm import RMSNorm, width_shard_input_memcfg

    monkeypatch.delenv("GEMMA4_CCL_L1_GATHER", raising=False)
    monkeypatch.delenv("GEMMA4_SHARDED_NORM", raising=False)
    mesh = _FakeMeshDevice()
    mgr = _FakeCclManager(mesh)
    tensor = _FakeTensor([1, 1, 128, 5376])
    norm_cfg = width_shard_input_memcfg(mesh, 5376, 128)
    assert _decode_l1_gather_memcfg(tensor, mgr) == norm_cfg

    monkeypatch.setenv("GEMMA4_CCL_L1_GATHER", "0")
    assert _decode_l1_gather_memcfg(tensor, mgr) is None

    # Long prefill stays on DRAM gather
    monkeypatch.delenv("GEMMA4_CCL_L1_GATHER", raising=False)
    assert _decode_l1_gather_memcfg(_FakeTensor([1, 1, 2048, 5376]), mgr) is None

    # Interleaved-LN experiment: AG→width_shard would be wasted
    monkeypatch.setenv("GEMMA4_SHARDED_NORM", "0")
    assert _decode_l1_gather_memcfg(tensor, mgr) is None
    monkeypatch.delenv("GEMMA4_SHARDED_NORM", raising=False)

    # RMSNorm builder agrees at the same (dim, height)
    norm = RMSNorm.__new__(RMSNorm)
    norm.mesh_device = mesh
    assert norm._build_sharded_cfg(5376, 128)[0] == norm_cfg


def test_batched_prefill_l1_gather_uses_physical_height(monkeypatch):
    """Batched prefill is [B, 1, S, H]; shard height must be B*S, not S."""
    from models.demos.gemma4.tt.ccl import _short_seq_l1_gather_memcfg
    from models.demos.gemma4.tt.rms_norm import (
        _SHARDED_NORM_MAX_HEIGHT,
        activation_physical_height,
        width_shard_input_memcfg,
    )

    monkeypatch.delenv("GEMMA4_CCL_L1_GATHER", raising=False)
    monkeypatch.delenv("GEMMA4_SHARDED_NORM", raising=False)
    mesh = _FakeMeshDevice()
    mgr = _FakeCclManager(mesh)

    assert activation_physical_height([1, 1, 96, 3840]) == 96
    assert activation_physical_height([2, 1, 96, 3840]) == 192
    assert activation_physical_height([4, 1, 512, 3840]) == 2048

    batched = _FakeTensor([2, 1, 96, 3840])
    expected = width_shard_input_memcfg(mesh, 3840, 192)
    assert expected is not None
    assert _short_seq_l1_gather_memcfg(batched, mgr) == expected
    # Must not emit the B=1 (height=96) spec — that is the TT_FATAL.
    assert _short_seq_l1_gather_memcfg(batched, mgr) != width_shard_input_memcfg(mesh, 3840, 96)

    # B*S above the sharded-norm cutoff stays DRAM (same as long B=1 prefill).
    assert 4 * 512 > _SHARDED_NORM_MAX_HEIGHT
    assert _short_seq_l1_gather_memcfg(_FakeTensor([4, 1, 512, 3840]), mgr) is None


def test_gate_up_1d_progcfg_matches_ln_width_shard():
    """LN width-shard grid (max cores) must be usable as gate_up 1D mcast_in0 grid."""
    from models.demos.gemma4.tt.dram_sharded import (
        prefill_progcfg_1d_for_width_sharded_in0,
        width_shard_core_count,
        width_shard_matches_1d_progcfg,
    )
    from models.demos.gemma4.tt.rms_norm import width_shard_spec

    mesh = _FakeMeshDevice()
    spec = width_shard_spec(mesh, 5376, 128)
    assert spec is not None
    memcfg, _ = spec
    assert width_shard_core_count(memcfg) == 56  # max divisor of 168 tiles on 8x8
    pc = prefill_progcfg_1d_for_width_sharded_in0(128, 5376, 5376, memcfg)
    assert pc is not None, "1D gate_up must accept LN's 56-core width shard (no S2I)"
    assert width_shard_matches_1d_progcfg(memcfg, pc)
    assert pc.fuse_batch, "ttnn requires fuse_batch when in0 is sharded"
    # 5376/56=96 → 3 K-tiles/core; in0_block_w must divide 3
    assert pc.in0_block_w in (1, 3)
    assert (3 % pc.in0_block_w) == 0


def test_sharded_norm_env_kill_switches(monkeypatch):
    """GEMMA4_SHARDED_NORM / GEMMA4_NORM_KEEP_SHARDED gate the AR→LN island."""
    from models.demos.gemma4.tt.rms_norm import norm_keep_sharded_enabled, sharded_norm_enabled

    monkeypatch.delenv("GEMMA4_SHARDED_NORM", raising=False)
    monkeypatch.delenv("GEMMA4_NORM_KEEP_SHARDED", raising=False)
    assert sharded_norm_enabled()
    assert norm_keep_sharded_enabled()

    monkeypatch.setenv("GEMMA4_SHARDED_NORM", "0")
    assert not sharded_norm_enabled()
    monkeypatch.setenv("GEMMA4_NORM_KEEP_SHARDED", "false")
    assert not norm_keep_sharded_enabled()


def test_prefill_mlp_island_gates(monkeypatch):
    """Short dense prefill keeps the LN island; MoE / batch / long seq / env opt out."""
    from models.demos.gemma4.tt.rms_norm import (
        _PREFILL_ISLAND_MAX_HEIGHT,
        _SHARDED_NORM_MAX_HEIGHT,
        prefill_mlp_island_enabled,
    )

    monkeypatch.delenv("GEMMA4_PREFILL_ISLAND", raising=False)
    monkeypatch.delenv("GEMMA4_SHARDED_NORM", raising=False)
    monkeypatch.delenv("GEMMA4_NORM_KEEP_SHARDED", raising=False)
    assert prefill_mlp_island_enabled(96)
    assert prefill_mlp_island_enabled(_PREFILL_ISLAND_MAX_HEIGHT)
    assert not prefill_mlp_island_enabled(_PREFILL_ISLAND_MAX_HEIGHT + 32)
    assert not prefill_mlp_island_enabled(_SHARDED_NORM_MAX_HEIGHT)
    assert not prefill_mlp_island_enabled(96, batch_size=2)
    assert not prefill_mlp_island_enabled(96, enable_moe=True)
    monkeypatch.setenv("GEMMA4_PREFILL_ISLAND", "0")
    assert not prefill_mlp_island_enabled(96)


def test_ccl_sync_rs_height_aware_defaults(monkeypatch):
    """Decode/short prefill stay w=1,c=1; T3K chunk height 2048 switches to w=2,c=2."""
    from models.demos.gemma4.tt.ccl import ccl_sync_rs_chunks, ccl_sync_rs_workers

    monkeypatch.delenv("GEMMA4_CCL_SYNC_RS_WORKERS", raising=False)
    monkeypatch.delenv("GEMMA4_CCL_SYNC_RS_CHUNKS", raising=False)
    assert ccl_sync_rs_workers() == 1
    assert ccl_sync_rs_chunks() == 1
    assert ccl_sync_rs_workers(32) == 1
    assert ccl_sync_rs_workers(96) == 1
    assert ccl_sync_rs_workers(1024) == 1
    assert ccl_sync_rs_workers(2048) == 2
    assert ccl_sync_rs_chunks(2048) == 2
    monkeypatch.setenv("GEMMA4_CCL_SYNC_RS_WORKERS", "1")
    monkeypatch.setenv("GEMMA4_CCL_SYNC_RS_CHUNKS", "1")
    assert ccl_sync_rs_workers(2048) == 1
    assert ccl_sync_rs_chunks(2048) == 1


def test_weight_cache_path_qualified_by_mesh(tmp_path, monkeypatch):
    """TP=4 on 1x4 vs 2x4 must not share tensorbin directories when mesh dirs are used."""
    import ttnn
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    monkeypatch.delenv("GEMMA4_WEIGHT_CACHE_MESH_ONLY", raising=False)
    args = Gemma4ModelArgs()
    args.model_cache_path = tmp_path
    # Empty caches → write into mesh-qualified paths (cold start).
    p_1x4 = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 4))
    p_2x4 = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(2, 4))
    p_1x1 = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 1))
    assert "mesh1x4" in str(p_1x4)
    assert "mesh2x4" in str(p_2x4)
    assert p_1x4 != p_2x4
    assert "mesh" not in p_1x1.name


def test_weight_cache_path_reuses_legacy_when_mesh_empty(tmp_path, monkeypatch):
    """CI MLPerf: empty mesh dir + warm legacy → reuse legacy (avoid cold 31B rebuild)."""
    import ttnn
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    monkeypatch.delenv("GEMMA4_WEIGHT_CACHE_MESH_ONLY", raising=False)
    legacy = tmp_path / "tensor_cache_bf16"
    legacy.mkdir()
    (legacy / "embed.tensorbin").write_text("x")
    args = Gemma4ModelArgs()
    args.model_cache_path = tmp_path
    assert args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 4)) == legacy


def test_weight_cache_path_mesh_only_ignores_legacy(tmp_path, monkeypatch):
    import ttnn
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    monkeypatch.setenv("GEMMA4_WEIGHT_CACHE_MESH_ONLY", "1")
    legacy = tmp_path / "tensor_cache_bf16"
    legacy.mkdir()
    (legacy / "embed.tensorbin").write_text("x")
    args = Gemma4ModelArgs()
    args.model_cache_path = tmp_path
    p = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 4))
    assert "mesh1x4" in str(p)
    assert p != legacy


def test_weight_cache_path_ro_mount_falls_back_writable(tmp_path, monkeypatch):
    """CI MLPerf :ro — mkdir(tensor_cache_*) must not raise Errno 30; mirror under TT_METAL_HOME."""
    import errno

    import ttnn
    from models.demos.gemma4.tt import model_config as mc
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    monkeypatch.delenv("GEMMA4_WEIGHT_CACHE_MESH_ONLY", raising=False)
    monkeypatch.setenv("TT_METAL_HOME", str(tmp_path / "metal_home"))
    ro_root = tmp_path / "mlperf_ro" / "google--gemma-4-12B-it"
    ro_root.mkdir(parents=True)

    real_mkdir = Path.mkdir

    def _ro_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        if "mlperf_ro" in self.parts:
            raise OSError(errno.EROFS, "Read-only file system", str(self))
        return real_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "mkdir", _ro_mkdir)
    args = Gemma4ModelArgs()
    args.model_cache_path = ro_root
    p = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 1))
    assert p.is_dir()
    assert "gemma4_tt_cache" in p.parts
    assert p.name == "tensor_cache_bf16"
    # Multi-device cold path on RO also mirrors (no warm legacy).
    p4 = args.weight_cache_path(ttnn.bfloat16, mesh_shape=(1, 4))
    assert p4.is_dir()
    assert "mesh1x4" in p4.name
    assert "gemma4_tt_cache" in p4.parts
    # Sanity: helper used by resolve path.
    assert mc._ensure_cache_dir(ro_root / "nested").is_dir()
