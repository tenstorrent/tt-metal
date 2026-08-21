# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the tuned Gemma4 prefill matmul program configs.

Every check here is a legality or gating assertion on what ``tt/dram_sharded.py``
hands ``ttnn.linear`` for prefill — subblock/grid legality, which M band a tuned
config fires in, and whether an opt-in knob is really opt-in. The configs are
pure functions of (m, k, n) and the compute grid, so no device is opened.
"""

import pytest

import ttnn

from .host_config_fakes import _FakeMeshDevice, _FakeTensor

# SharedMLP's two tuned prefill matmuls, at the shapes the shipped variants
# actually run (intermediate_size/TP from each checkpoint's config, TP=8):
#   gate_up  in0=[M, hidden]          weight=[hidden, 2*inter/tp]
#   down     in0=[M, inter/tp]        weight=[inter/tp, hidden]
_MLP_PREFILL_SHAPES = [
    ("31B gate_up", "gate_up", 5376, 5376),
    ("31B down_proj", "down", 2688, 5376),
    ("12B gate_up", "gate_up", 3840, 3840),
    ("12B down_proj", "down", 1920, 3840),
]


def _mlp_prefill_config(which, m, k, n):
    from models.demos.gemma4.tt.dram_sharded import (
        interleaved_down_proj_prefill_config,
        interleaved_gate_up_prefill_config,
    )

    fn = interleaved_gate_up_prefill_config if which == "gate_up" else interleaved_down_proj_prefill_config
    return fn(m, k, n)


@pytest.mark.parametrize("label,which,k,n", _MLP_PREFILL_SHAPES, ids=[c[0] for c in _MLP_PREFILL_SHAPES])
def test_mlp_prefill_config_is_wired_and_legal(label, which, k, n):
    """SharedMLP's tuned prefill matmuls must stay wired, and stay legal.

    This is deliberately NOT a pin on the swept numbers (core count,
    ``in0_block_w``, ``per_core_N``) — those were removed with the sweep-winner
    tests, and re-adding them just recreates a test that fails on any retune.
    What it does hold is the part the win rests on and the part ttnn will
    ``TT_FATAL`` on:

    * **Wired at all.** Returning all-``None`` in the band silently drops both
      matmuls back to ``ttnn.linear``'s auto choice, which is the regression this
      config exists to avoid. Nothing else in the suite would notice.
    * **1D mcast family.** ``mcast_in0`` with every core holding all of M is the
      shape of the optimization; a 2D config here is a different thing wearing
      the same name.
    * **Legality.** ``in0_block_w`` must divide Kt, ``per_core_N * cores`` must
      cover Nt exactly (a short cover drops output tiles), and the output
      subblock must fit DST.
    * **The two CKC guardrails** from ``_prefill_hifi4_ckc``: pinning a program
      config without an explicit CKC silently selects LoFi, and dest-acc on this
      band dropped last-token PCC.
    """
    from models.demos.gemma4.tt.dram_sharded import _PREFILL_CUTOFF, TILE_SIZE

    m = _PREFILL_CUTOFF  # top of the band, where SharedMLP actually calls it
    pc, out_memcfg, ckc = _mlp_prefill_config(which, m, k, n)

    assert pc is not None, f"{label}: tuned prefill config went missing — falls back to ttnn auto"
    assert out_memcfg is not None and ckc is not None, f"{label}: partial config"

    assert isinstance(pc, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig), f"{label}: not the 1D mcast family"
    assert pc.mcast_in0, f"{label}: 1D config must multicast in0"

    grid = pc.compute_with_storage_grid_size
    cores = grid.x * grid.y
    k_tiles = (k + TILE_SIZE - 1) // TILE_SIZE
    n_tiles = (n + TILE_SIZE - 1) // TILE_SIZE

    assert pc.per_core_M == (m + TILE_SIZE - 1) // TILE_SIZE, f"{label}: every core must hold all of M"
    assert k_tiles % pc.in0_block_w == 0, f"{label}: in0_block_w={pc.in0_block_w} must divide Kt={k_tiles}"
    assert pc.per_core_N * cores == n_tiles, f"{label}: per_core_N*{cores} != Nt={n_tiles} (N not covered)"
    assert pc.out_subblock_h * pc.out_subblock_w <= 4, f"{label}: output subblock exceeds DST"

    assert ckc.math_fidelity != ttnn.MathFidelity.LoFi, f"{label}: explicit CKC must not fall back to LoFi"
    assert ckc.fp32_dest_acc_en is False, f"{label}: dest-acc on this band drops last-token PCC"


@pytest.mark.parametrize("label,which,k,n", _MLP_PREFILL_SHAPES, ids=[c[0] for c in _MLP_PREFILL_SHAPES])
def test_mlp_prefill_config_band_gated(label, which, k, n):
    """Outside ``TILE < M <= _PREFILL_CUTOFF`` both configs must decline.

    Decode (M=32) keeps the auto/DRAM-sharded path, and above the cutoff the 2D
    kernel's CBs scale with ``per_core_M`` — pinning this config at long context
    blows L1. Both edges are load-bearing, so assert them rather than the middle.
    """
    from models.demos.gemma4.tt.dram_sharded import _PREFILL_CUTOFF, TILE_SIZE

    assert _mlp_prefill_config(which, TILE_SIZE, k, n) == (None, None, None), f"{label}: fired at decode M=32"
    assert _mlp_prefill_config(which, _PREFILL_CUTOFF + TILE_SIZE, k, n) == (
        None,
        None,
        None,
    ), f"{label}: fired above the cutoff"


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
    tuned matmul saves."""
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
