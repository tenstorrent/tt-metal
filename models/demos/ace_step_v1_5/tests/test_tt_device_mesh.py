# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from models.demos.ace_step_v1_5.tt_device import (
    ace_step_cfg_data_parallel_available,
    ace_step_cfg_data_parallel_requested,
    ace_step_dit_rope_max_seq_len,
    ace_step_mesh_is_2d,
    ace_step_mesh_perf_log_default,
    ace_step_mesh_shape,
    ace_step_mesh_use_adg,
    ace_step_mesh_use_host_latent_sampler,
    ace_step_mesh_use_host_temb_precompute,
    ace_step_mesh_use_sequential_cfg,
    ace_step_needs_split_device,
    ace_step_replicate_mesh_mapper,
    ace_step_resolve_vae_tiling,
    resolve_ace_step_mesh_sku,
)
from models.demos.ace_step_v1_5.ttnn_impl.dit_sampling_ttnn import _host_gaussian_latents_f32


def test_resolve_mesh_sku_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("MESH_DEVICE", "P150")
    assert resolve_ace_step_mesh_sku(cli_value="BH_QB") == "BH_QB"


def test_mesh_shape_bh_qb():
    assert ace_step_mesh_shape("BH_QB") == (2, 2)
    assert ace_step_mesh_shape("BH_LB") == (2, 4)
    assert ace_step_mesh_shape(None) == (1, 1)


def test_split_preprocess_only_for_multi_device():
    assert not ace_step_needs_split_device(None)
    assert not ace_step_needs_split_device("P150")
    assert ace_step_needs_split_device("BH_QB")
    assert ace_step_needs_split_device("BH_LB")


def test_cfg_dp_requested_from_env(monkeypatch):
    monkeypatch.delenv("ACE_STEP_CFG_DATA_PARALLEL", raising=False)
    assert not ace_step_cfg_data_parallel_requested()
    monkeypatch.setenv("ACE_STEP_CFG_DATA_PARALLEL", "1")
    assert ace_step_cfg_data_parallel_requested()


class _FakeMesh:
    def __init__(self, n: int):
        self._n = n

    def get_num_devices(self):
        return self._n


def test_cfg_dp_available_requires_even_multi_device():
    assert not ace_step_cfg_data_parallel_available(_FakeMesh(1), do_cfg=True, requested=True)
    assert not ace_step_cfg_data_parallel_available(_FakeMesh(4), do_cfg=False, requested=True)
    assert ace_step_cfg_data_parallel_available(_FakeMesh(4), do_cfg=True, requested=True)
    assert not ace_step_cfg_data_parallel_available(_FakeMesh(3), do_cfg=True, requested=True)


def test_unknown_mesh_sku_raises():
    with pytest.raises(ValueError, match="Unknown ACE-Step mesh SKU"):
        resolve_ace_step_mesh_sku(cli_value="NOT_A_SKU")


class _FakeMesh2d:
    shape = (2, 2)

    def get_num_devices(self):
        return 4


def test_mesh_is_2d():
    assert ace_step_mesh_is_2d(_FakeMesh2d())
    assert not ace_step_mesh_is_2d(_FakeMesh(4))


def test_replicate_mesh_mapper_none_for_single_device():
    assert ace_step_replicate_mesh_mapper(None) is None


def test_dit_rope_max_seq_capped_for_duration():
    # 15 s @ 25 Hz = 375 frames; patch 2 -> 188 patches + 128 margin = 316 (below 4096)
    cap = ace_step_dit_rope_max_seq_len(expected_input_length=375, patch_size=2, hf_max=4096)
    assert cap < 4096
    assert cap >= 188


def test_host_gaussian_latents_reproducible():
    a = _host_gaussian_latents_f32((1, 16, 64), seed=7)
    b = _host_gaussian_latents_f32((1, 16, 64), seed=7)
    c = _host_gaussian_latents_f32((1, 16, 64), seed=8)
    assert a.shape == (1, 16, 64)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_host_latent_sampler_on_multi_device_only():
    assert not ace_step_mesh_use_host_latent_sampler(_FakeMesh(1), use_trace=False)
    assert ace_step_mesh_use_host_latent_sampler(_FakeMesh(4), use_trace=False)
    assert not ace_step_mesh_use_host_latent_sampler(_FakeMesh(4), use_trace=True)


def test_sequential_cfg_on_multi_device_only():
    assert not ace_step_mesh_use_sequential_cfg(_FakeMesh(1), do_cfg=True)
    assert ace_step_mesh_use_sequential_cfg(_FakeMesh(4), do_cfg=True)
    assert not ace_step_mesh_use_sequential_cfg(_FakeMesh(4), do_cfg=False)


def test_host_temb_precompute_on_multi_device_only():
    assert not ace_step_mesh_use_host_temb_precompute(_FakeMesh(1))
    assert ace_step_mesh_use_host_temb_precompute(_FakeMesh(4))


def test_mesh_use_adg_defaults():
    assert ace_step_mesh_use_adg(mesh_sku=None, variant="acestep-v15-base", cli_use_adg=None)
    assert not ace_step_mesh_use_adg(mesh_sku="BH_QB", variant="acestep-v15-base", cli_use_adg=None)
    assert ace_step_mesh_use_adg(mesh_sku="BH_QB", variant="acestep-v15-base", cli_use_adg=True)
    assert not ace_step_mesh_use_adg(mesh_sku="BH_QB", variant="acestep-v15-turbo", cli_use_adg=None)


def test_resolve_vae_tiling_mesh_long_clip():
    chunk, overlap = ace_step_resolve_vae_tiling(frames=375, mesh_sku="BH_QB", chunk_cli=32, overlap_cli=4)
    assert chunk == 32
    assert overlap >= 8


def test_mesh_perf_log_default():
    assert ace_step_mesh_perf_log_default(mesh_sku="BH_QB")
    assert not ace_step_mesh_perf_log_default(mesh_sku="P150")
    assert not ace_step_mesh_perf_log_default(mesh_sku=None)
