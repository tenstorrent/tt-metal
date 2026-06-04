# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Guard tests for VAE ``1×1`` conv im2col matmul program-config selection."""

from __future__ import annotations

import os

import pytest

from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_device_profiler_enabled,
    ace_step_enable_tracy_profiler_env,
    ace_step_profiler_flush_every_layer,
    ace_step_vae_conv1d_im2col_matmul_program_config,
    ace_step_vae_k1_height_sharded_eligible,
    ace_step_vae_k1_height_sharded_program_config,
    ace_step_vae_k1_mid_m_matmul_program_config,
    ace_step_vae_k1_prefer_conv1d_l1,
    ace_step_vae_large_m_matmul_program_config,
    ace_step_vae_sharded_matmul_enabled,
)


class _FakeGridDevice:
    def compute_with_storage_grid_size(self):
        return type("G", (), {"x": 11, "y": 10})()


@pytest.mark.parametrize(
    "m_dim,k_dim,n_dim",
    [
        (7679, 128, 128),
        (61440, 1024, 1024),
        (7679, 512, 512),
    ],
)
def test_conv1d_im2col_matmul_skips_unsafe_or_small_shapes(m_dim, k_dim, n_dim):
    assert (
        ace_step_vae_conv1d_im2col_matmul_program_config(
            _FakeGridDevice(),
            m_dim=m_dim,
            k_dim=k_dim,
            n_dim=n_dim,
        )
        is None
    )


def test_conv1d_im2col_matmul_returns_config_for_tracy_buckets(monkeypatch):
    monkeypatch.delenv("TT_METAL_DEVICE_PROFILER", raising=False)
    pc = ace_step_vae_conv1d_im2col_matmul_program_config(
        _FakeGridDevice(),
        m_dim=7680,
        k_dim=256,
        n_dim=256,
    )
    assert pc is not None
    assert pc.mcast_in0 is False
    assert pc.per_core_M == 3  # ceil(7680/32 / 110) — Tracy-safe bucket


def test_conv1d_im2col_matmul_wide_channels_at_large_m(monkeypatch):
    monkeypatch.delenv("ACE_STEP_VAE_LARGE_M_MATMUL", raising=False)
    pc = ace_step_vae_conv1d_im2col_matmul_program_config(
        _FakeGridDevice(),
        m_dim=7680,
        k_dim=512,
        n_dim=512,
    )
    assert pc is not None
    assert pc.mcast_in0 is False


def test_large_m_helper_tiers():
    dev = _FakeGridDevice()
    pc_mid = ace_step_vae_large_m_matmul_program_config(dev, m_dim=7680, k_dim=256, n_dim=256)
    assert pc_mid is not None
    assert pc_mid.mcast_in0 is False
    assert pc_mid.out_subblock_h == 1
    assert pc_mid.per_core_M == 3
    assert ace_step_vae_large_m_matmul_program_config(dev, m_dim=61440, k_dim=128, n_dim=128) is None


def test_conv1d_im2col_matmul_opt_out(monkeypatch):
    monkeypatch.setenv("ACE_STEP_VAE_LARGE_M_MATMUL", "0")
    monkeypatch.delenv("TT_METAL_DEVICE_PROFILER", raising=False)
    assert (
        ace_step_vae_conv1d_im2col_matmul_program_config(
            _FakeGridDevice(),
            m_dim=5000,
            k_dim=128,
            n_dim=128,
        )
        is None
    )
    # Large M still returns a clamped config (avoid conv1d 640-core probe) even when opt-out.
    pc = ace_step_vae_conv1d_im2col_matmul_program_config(
        _FakeGridDevice(),
        m_dim=61440,
        k_dim=128,
        n_dim=128,
    )
    assert pc is None  # non-profiler L1 cap; profiler run uses cap 20 instead


def test_conv1d_im2col_matmul_on_under_device_profiler(monkeypatch):
    monkeypatch.delenv("ACE_STEP_VAE_LARGE_M_MATMUL", raising=False)
    monkeypatch.setenv("TT_METAL_DEVICE_PROFILER", "1")
    pc = ace_step_vae_conv1d_im2col_matmul_program_config(
        _FakeGridDevice(),
        m_dim=7680,
        k_dim=256,
        n_dim=256,
    )
    assert pc is not None
    assert pc.compute_with_storage_grid_size == (11, 10)


def test_conv1d_im2col_matmul_full_grid_config(monkeypatch):
    monkeypatch.delenv("TT_METAL_DEVICE_PROFILER", raising=False)
    pc = ace_step_vae_conv1d_im2col_matmul_program_config(
        _FakeGridDevice(),
        m_dim=7680,
        k_dim=256,
        n_dim=256,
    )
    assert pc is not None
    assert pc.compute_with_storage_grid_size == (11, 10)


def test_enable_tracy_profiler_env_sets_ttnn_op_profiler(monkeypatch):
    monkeypatch.delenv("TTNN_OP_PROFILER", raising=False)
    monkeypatch.setenv("TT_METAL_DEVICE_PROFILER", "1")
    ace_step_enable_tracy_profiler_env()
    assert os.environ.get("TTNN_OP_PROFILER") == "1"


def test_profiler_flush_every_layer_defaults(monkeypatch):
    monkeypatch.delenv("ACE_STEP_PROFILER_FLUSH_EVERY_LAYER", raising=False)
    monkeypatch.delenv("TTNN_OP_PROFILER", raising=False)
    monkeypatch.delenv("TT_METAL_DEVICE_PROFILER", raising=False)
    assert ace_step_profiler_flush_every_layer() == 0

    monkeypatch.setenv("TT_METAL_DEVICE_PROFILER", "1")
    assert ace_step_profiler_flush_every_layer() == 1

    monkeypatch.setenv("ACE_STEP_PROFILER_FLUSH_EVERY_LAYER", "0")
    assert ace_step_profiler_flush_every_layer() == 0


def test_device_profiler_enabled(monkeypatch):
    monkeypatch.delenv("TTNN_OP_PROFILER", raising=False)
    monkeypatch.delenv("TT_METAL_DEVICE_PROFILER", raising=False)
    assert ace_step_device_profiler_enabled() is False

    monkeypatch.setenv("TT_METAL_DEVICE_PROFILER", "1")
    assert ace_step_device_profiler_enabled() is True


def test_sharded_matmul_opt_in(monkeypatch):
    monkeypatch.delenv("ACE_STEP_VAE_SHARDED_MATMUL", raising=False)
    assert ace_step_vae_sharded_matmul_enabled() is False
    monkeypatch.setenv("ACE_STEP_VAE_SHARDED_MATMUL", "1")
    assert ace_step_vae_sharded_matmul_enabled() is True
    monkeypatch.setenv("ACE_STEP_VAE_SHARDED_MATMUL", "0")
    assert ace_step_vae_sharded_matmul_enabled() is False


@pytest.mark.parametrize(
    "m_dim,k_dim,n_dim,eligible",
    [
        (1920, 512, 512, True),
        (320, 1024, 1024, False),  # M < 512 — separate bucket, not in v1 path
        (511, 512, 512, False),
        (7680, 512, 512, False),
        (1920, 256, 256, False),
    ],
)
def test_k1_height_sharded_eligibility(m_dim, k_dim, n_dim, eligible):
    assert ace_step_vae_k1_height_sharded_eligible(m_dim=m_dim, k_dim=k_dim, n_dim=n_dim) is eligible


def test_k1_mid_m_matmul_program_config_1920(monkeypatch):
    monkeypatch.delenv("TT_METAL_DEVICE_PROFILER", raising=False)
    pc = ace_step_vae_k1_mid_m_matmul_program_config(
        _FakeGridDevice(),
        m_dim=1920,
        k_dim=512,
        n_dim=512,
    )
    assert pc is not None
    assert pc.mcast_in0 is False
    assert pc.per_core_M == 1  # ceil(60 M-tiles / 110 cores)
    assert pc.compute_with_storage_grid_size == (11, 10)


def test_k1_height_sharded_program_config_alias_1920(monkeypatch):
    monkeypatch.delenv("TT_METAL_DEVICE_PROFILER", raising=False)
    pc = ace_step_vae_k1_height_sharded_program_config(
        _FakeGridDevice(),
        m_dim=1920,
        k_dim=512,
        n_dim=512,
    )
    assert pc is not None
    assert pc.mcast_in0 is False
    assert pc.compute_with_storage_grid_size == (11, 10)


def test_k1_prefer_conv1d_l1(monkeypatch):
    monkeypatch.delenv("ACE_STEP_VAE_SHARDED_MATMUL", raising=False)
    assert ace_step_vae_k1_prefer_conv1d_l1(m_dim=1920, k_dim=512, n_dim=512) is False
    monkeypatch.setenv("ACE_STEP_VAE_SHARDED_MATMUL", "1")
    assert ace_step_vae_k1_prefer_conv1d_l1(m_dim=1920, k_dim=512, n_dim=512) is True
    assert ace_step_vae_k1_prefer_conv1d_l1(m_dim=256, k_dim=128, n_dim=128) is False
