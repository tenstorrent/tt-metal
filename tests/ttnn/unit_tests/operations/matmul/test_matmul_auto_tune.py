# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the host-side matmul auto-tune helpers in ``ttnn.matmul_auto_tune``."""
import pytest

import ttnn

at = ttnn.matmul_auto_tune


# ----- dst_capacity ---------------------------------------------------------


def test_dst_capacity_full_sync_bf16():
    assert at.dst_capacity(fp32_dest_acc_en=False, dst_full_sync_en=True) == 8


def test_dst_capacity_full_sync_fp32_halves():
    assert at.dst_capacity(fp32_dest_acc_en=True, dst_full_sync_en=True) == 4


def test_dst_capacity_half_sync_doubles():
    assert at.dst_capacity(fp32_dest_acc_en=False, dst_full_sync_en=False) == 16
    assert at.dst_capacity(fp32_dest_acc_en=True, dst_full_sync_en=False) == 8


# ----- needs_row_major ------------------------------------------------------


@pytest.mark.parametrize(
    "h,w,per_core_n,expected",
    [
        (1, 1, 8, False),  # h==1 → legacy-OK
        (1, 8, 8, False),  # h==1 → legacy-OK
        (4, 8, 8, False),  # w==per_core_N → legacy-OK
        (2, 4, 8, True),  # h>1 and w<per_core_N → needs rmo
        (4, 2, 8, True),
    ],
)
def test_needs_row_major(h, w, per_core_n, expected):
    assert at.needs_row_major(h, w, per_core_n) is expected


# ----- subblock_options -----------------------------------------------------


def test_subblock_options_picks_volume_8_first():
    opts = at.subblock_options(per_core_M=4, per_core_N=8, fp32_dest_acc_en=False)
    # First option should be max volume (8), with h=1 fast path preferred
    assert opts[0] == (1, 8)
    # Volume 8 options come before volume 4
    v8_count = sum(1 for h, w in opts if h * w == 8)
    assert v8_count >= 3  # (1,8), (2,4), (4,2)


def test_subblock_options_legacy_writer_filters_rmo():
    rmo_opts = at.subblock_options(per_core_M=4, per_core_N=8, fp32_dest_acc_en=False, require_legacy_writer=False)
    legacy_opts = at.subblock_options(per_core_M=4, per_core_N=8, fp32_dest_acc_en=False, require_legacy_writer=True)
    # Legacy filter must be a subset
    assert set(legacy_opts).issubset(set(rmo_opts))
    # All legacy opts must satisfy h==1 OR w==per_core_N
    for h, w in legacy_opts:
        assert h == 1 or w == 8


def test_subblock_options_respects_divisibility():
    opts = at.subblock_options(per_core_M=3, per_core_N=5, fp32_dest_acc_en=False)
    for h, w in opts:
        assert 3 % h == 0
        assert 5 % w == 0


def test_subblock_options_fp32_caps_at_4():
    opts = at.subblock_options(per_core_M=8, per_core_N=8, fp32_dest_acc_en=True)
    assert all(h * w <= 4 for h, w in opts)
    assert opts[0] in [(1, 4), (4, 1), (2, 2)]


# ----- largest_subblock -----------------------------------------------------


@pytest.mark.parametrize(
    "pm,pn,fp32,want",
    [
        (4, 8, False, (1, 8)),  # volume 8, fast path (h=1)
        (4, 8, True, (1, 4)),  # cap=4, fast path
        (1, 8, False, (1, 8)),  # h=1 forced
        (8, 1, False, (8, 1)),  # w=1 forced
        (16, 20, False, (1, 5)),  # legacy_best=(1, 5), rmo_best=(2, 4) v=8 — picks rmo
    ],
)
def test_largest_subblock(pm, pn, fp32, want):
    # By default require_legacy_writer=False, so rmo picks are allowed
    h, w = at.largest_subblock(pm, pn, fp32)
    assert h * w >= want[0] * want[1], f"got ({h},{w}) v={h*w} < expected v={want[0]*want[1]}"


def test_largest_subblock_legacy_only():
    # M=16, N=20: rmo can do (2, 4)+rmo for v=8 but legacy max is (1, 5)
    h, w = at.largest_subblock(16, 20, False, require_legacy_writer=True)
    assert (h, w) == (1, 5)


def test_largest_subblock_no_legal_pair_returns_1_1():
    h, w = at.largest_subblock(0, 0, False)
    assert (h, w) == (1, 1)


# ----- upgrade_subblock -----------------------------------------------------


def test_upgrade_subblock_2d_mcast_volume_1_to_8():
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=4,
        per_core_N=8,
        out_subblock_h=1,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    out = at.upgrade_subblock(cfg, fp32_dest_acc_en=False)
    assert out is cfg  # mutates and returns same object
    assert cfg.out_subblock_h == 1
    assert cfg.out_subblock_w == 8
    assert cfg.tile_pack_row_major is False  # h=1 doesn't need rmo


def test_upgrade_subblock_enables_rmo_when_needed():
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=4,
        per_core_N=20,
        out_subblock_h=1,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    out = at.upgrade_subblock(cfg, fp32_dest_acc_en=False)
    # M=4 N=20, cap=8: best rmo (2, 4) v=8; legacy (1, 5) v=5
    assert (cfg.out_subblock_h, cfg.out_subblock_w) == (2, 4)
    assert cfg.tile_pack_row_major is True


def test_upgrade_subblock_legacy_only_skips_rmo():
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=4,
        per_core_N=20,
        out_subblock_h=1,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    at.upgrade_subblock(cfg, fp32_dest_acc_en=False, require_legacy_writer=True)
    # Legacy max for M=4 N=20: (1, 5) v=5
    assert (cfg.out_subblock_h, cfg.out_subblock_w) == (1, 5)
    assert cfg.tile_pack_row_major is False


def test_upgrade_subblock_reads_compute_kernel_config():
    """When compute_kernel_config is passed, its fp32_dest_acc_en and dst_full_sync_en
    attributes should drive the DST-capacity decision without requiring the caller to
    re-state the flags. ttnn's default WormholeComputeKernelConfig uses
    dst_full_sync_en=False (half-sync), which doubles the DST tile budget."""
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=4,
        per_core_N=8,
        out_subblock_h=1,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    # fp32_dest_acc_en=True halves cap; default dst_full_sync_en=False doubles it.
    # Net cap = 16 / 2 = 8 → can pick (1, 8).
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    at.upgrade_subblock(cfg, compute_kernel_config=ckc)
    assert (cfg.out_subblock_h, cfg.out_subblock_w) == (1, 8)


def test_upgrade_subblock_1d_mcast():
    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        in0_block_w=2,
        per_core_M=4,
        per_core_N=8,
        out_subblock_h=1,
        out_subblock_w=2,
        mcast_in0=False,
        fuse_batch=True,
        fused_activation=None,
    )
    at.upgrade_subblock(cfg, fp32_dest_acc_en=False)
    assert (cfg.out_subblock_h, cfg.out_subblock_w) == (1, 8)


# ----- estimate_l1_per_core -------------------------------------------------


def test_estimate_l1_basic_fits_bh():
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=4,
        per_core_N=8,
        out_subblock_h=1,
        out_subblock_w=8,
        transpose_mcast=False,
        fused_activation=None,
    )
    fp = at.estimate_l1_per_core(cfg, in0_tile_bytes=2048, in1_tile_bytes=2048, out_tile_bytes=2048)
    assert fp["fits_bh"] is True
    assert fp["fits_wh"] is True
    # out_buf = 4*8*2048 = 64 KB
    assert fp["out_buf_bytes"] == 4 * 8 * 2048
    # in0_buf = 2 * 4 * 2 * 2048 = 32 KB
    assert fp["in0_buf_bytes"] == 2 * 4 * 2 * 2048
    # in1_buf = 2 * 8 * 2 * 2048 = 64 KB
    assert fp["in1_buf_bytes"] == 2 * 8 * 2 * 2048


def test_estimate_l1_rmo_doubles_output():
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=4,
        per_core_N=8,
        out_subblock_h=2,
        out_subblock_w=4,
        transpose_mcast=False,
        fused_activation=None,
        tile_pack_row_major=True,
    )
    fp = at.estimate_l1_per_core(cfg)
    # rmo ⇒ interm_buf same size as out_buf
    assert fp["interm_buf_bytes"] == fp["out_buf_bytes"]


def test_estimate_l1_fuse_bias_adds_interm():
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=4,
        per_core_N=8,
        out_subblock_h=1,
        out_subblock_w=8,
        transpose_mcast=False,
        fused_activation=None,
    )
    fp_no_bias = at.estimate_l1_per_core(cfg)
    fp_bias = at.estimate_l1_per_core(cfg, fuse_bias=True)
    assert fp_bias["interm_buf_bytes"] == fp_no_bias["out_buf_bytes"]
    assert fp_bias["estimated_bytes"] > fp_no_bias["estimated_bytes"]


def test_estimate_l1_oversized_does_not_fit():
    # Huge per_core dimensions → won't fit in any L1
    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=8,
        per_core_M=64,
        per_core_N=64,
        out_subblock_h=1,
        out_subblock_w=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    fp = at.estimate_l1_per_core(cfg)
    assert fp["fits_bh"] is False
    assert fp["fits_wh"] is False
    assert fp["headroom_bh"] < 0


# ----- end-to-end: upgraded config still runs through ttnn.matmul -----------


def test_upgraded_config_runs_through_matmul(device):
    """Upgrade a config from (1, 1) → (1, 8) and verify ttnn.matmul accepts it
    and produces correct results. Catches regressions where mutating a config
    breaks the matmul op (e.g. if the program_config's other invariants depend
    on the old subblock values via out_block_h/out_block_w internal caches).
    """
    import torch

    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        per_core_M=4,
        per_core_N=8,
        out_subblock_h=1,
        out_subblock_w=1,  # Deliberately suboptimal — cap=8 supports up to (1, 8)
        transpose_mcast=False,
        fused_activation=None,
    )

    # Apply auto-upgrade
    at.upgrade_subblock(cfg, fp32_dest_acc_en=False)
    assert (cfg.out_subblock_h, cfg.out_subblock_w) == (1, 8)

    # Build inputs that match the program_config dims:
    # M_tiles = 8 rows * per_core_M = 32 → M = 1024
    # N_tiles = 8 cols * per_core_N = 64 → N = 2048
    # K must be a multiple of in0_block_w=2 tiles: pick K_tiles=64 → K=2048
    M, K, N = 1024, 2048, 2048
    a_t = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    b_t = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    out_tt = ttnn.matmul(a, b, program_config=cfg, compute_kernel_config=compute_kernel_config)
    out_torch = ttnn.to_torch(out_tt)
    expected = (a_t.float() @ b_t.float()).bfloat16()

    # PCC tolerance matches the typical bf16+HiFi2 matmul threshold used elsewhere
    # in the matmul test suite (see test_matmul.py).
    from tests.ttnn.utils_for_testing import assert_with_pcc

    assert_with_pcc(expected, out_torch, pcc=0.99)
    ttnn.deallocate(out_tt)
