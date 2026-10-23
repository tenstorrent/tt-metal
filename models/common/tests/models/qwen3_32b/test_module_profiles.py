# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure semantic snapshots for Qwen3-32B WH/BH module composition."""

import inspect
import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.models.qwen3_32b import weight_utils
from models.common.models.qwen3_32b.model import (
    QWEN3_32B_ACCURACY,
    QWEN3_32B_INTERMEDIATE_SIZE,
    QWEN3_32B_PERFORMANCE,
    Qwen3_32B,
    _qwen3_attention_config,
    _qwen3_ccl_topology,
    _qwen3_lm_head_config,
    _qwen3_mlp_config,
    _qwen3_rmsnorm_config,
    _resolve_qwen3_32b_sku_overlay,
)
from models.common.modules.attention.attention_1d import Attention1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1DConfig
from models.common.modules.mlp.mlp_1d import MLP1DConfig
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1DConfig
from models.common.modules.rope.rope_1d import Rope1DConfig, _resolve_rope_config


class _FakeMesh:
    def __init__(self, dram_grid_width=8, *, arch=ttnn.device.Arch.BLACKHOLE, num_devices=4):
        self.dram_grid_width = dram_grid_width
        self._arch = arch
        self._num_devices = num_devices

    def arch(self):
        return self._arch

    def get_num_devices(self):
        return self._num_devices

    def compute_with_storage_grid_size(self):
        return ttnn.CoreCoord(8, 8)

    def dram_grid_size(self):
        return ttnn.CoreCoord(self.dram_grid_width, 1)


def _kernel_semantics(config):
    return (
        config.math_fidelity,
        config.math_approx_mode,
        config.fp32_dest_acc_en,
        config.packer_l1_acc,
        config.dst_full_sync_en,
    )


def _attention_slots(profile):
    return (
        profile.attn_decode_qkv_kernel,
        profile.attn_decode_sdpa_kernel,
        profile.attn_decode_wo_kernel,
        profile.attn_prefill_qkv_kernel,
        profile.attn_prefill_sdpa_kernel,
        profile.attn_prefill_wo_kernel,
    )


def _mlp_slots(profile):
    return (
        profile.mlp_prefill_ff1_ff3_kernel,
        profile.mlp_prefill_ff2_kernel,
        profile.mlp_decode_ff1_ff3_kernel,
        profile.mlp_decode_ff2_kernel,
    )


def test_accuracy_profile_explicitly_locks_all_attention_and_mlp_slots():
    assert [_kernel_semantics(config) for config in _attention_slots(QWEN3_32B_ACCURACY)] == [
        (ttnn.MathFidelity.HiFi4, False, True, True, False)
    ] * 6
    assert [_kernel_semantics(config) for config in _mlp_slots(QWEN3_32B_ACCURACY)] == [
        (ttnn.MathFidelity.HiFi2, False, False, True, False)
    ] * 4


def test_performance_profile_matches_tttv1_six_slot_attention_and_four_slot_mlp_table():
    hifi2_fp32_approx = (ttnn.MathFidelity.HiFi2, True, True, True, False)
    hifi4_fp32 = (ttnn.MathFidelity.HiFi4, False, True, True, False)
    assert [_kernel_semantics(config) for config in _attention_slots(QWEN3_32B_PERFORMANCE)] == [
        hifi2_fp32_approx,
        hifi2_fp32_approx,
        hifi2_fp32_approx,
        hifi2_fp32_approx,
        hifi4_fp32,
        hifi2_fp32_approx,
    ]
    assert [_kernel_semantics(config) for config in _mlp_slots(QWEN3_32B_PERFORMANCE)] == [
        (ttnn.MathFidelity.LoFi, False, False, True, False),
        (ttnn.MathFidelity.HiFi2, False, False, True, False),
        (ttnn.MathFidelity.LoFi, False, False, True, False),
        (ttnn.MathFidelity.HiFi2, False, False, True, False),
    ]


def test_wormhole_t3k_overlay_preserves_baseline(monkeypatch):
    monkeypatch.delenv("DISABLE_MINIMAL_MATMUL", raising=False)
    overlay = _resolve_qwen3_32b_sku_overlay(
        arch=ttnn.device.Arch.WORMHOLE_B0,
        cluster_type=ttnn.cluster.ClusterType.T3K,
        num_dev=8,
        # Real Wormhole reports 12 physical DRAM cores, while the approved
        # Qwen T3K recipe intentionally shards over 8.
        mesh_device=_FakeMesh(dram_grid_width=12),
    )

    assert overlay.architecture == "wormhole"
    assert overlay.topology == ttnn.Topology.Ring
    assert overlay.dram_shard_grid_width == 8
    assert overlay.mlp_prefill_len_cutoff == 1024
    assert overlay.mlp_prefill_grid == (8, 8)
    assert overlay.attention_prefill_qkv_grid == (8, 8)
    assert overlay.attention_decode_create_qkv_head_grid is None
    assert overlay.lm_head_core_grid is None
    assert overlay.lm_head_max_columns_per_device == 8192
    assert overlay.distributed_rmsnorm_min_dim_exclusive is None
    assert overlay.prefill_minimal_matmul is True
    assert overlay.disable_batched_prefill is False


@pytest.mark.parametrize(
    "cluster_type",
    [ttnn.cluster.ClusterType.P150_X4, ttnn.cluster.ClusterType.P300_X2],
)
def test_blackhole_four_die_overlay_and_lm_splits(cluster_type, monkeypatch):
    monkeypatch.delenv("DISABLE_MINIMAL_MATMUL", raising=False)
    overlay = _resolve_qwen3_32b_sku_overlay(
        arch=ttnn.device.Arch.BLACKHOLE,
        cluster_type=cluster_type,
        num_dev=4,
        mesh_device=_FakeMesh(),
    )

    assert overlay.architecture == "blackhole"
    assert overlay.topology == ttnn.Topology.Ring
    assert overlay.dram_shard_grid_width == 8
    assert overlay.mlp_prefill_len_cutoff == 512
    assert overlay.mlp_prefill_grid == (8, 5)
    assert overlay.attention_prefill_qkv_grid == (8, 4)
    assert (overlay.attention_decode_create_qkv_head_grid.x, overlay.attention_decode_create_qkv_head_grid.y) == (
        8,
        4,
    )
    assert (overlay.lm_head_core_grid.x, overlay.lm_head_core_grid.y) == (8, 5)
    assert overlay.distributed_rmsnorm_min_dim_exclusive == 4096
    assert overlay.prefill_minimal_matmul is True
    assert overlay.disable_batched_prefill is True
    assert weight_utils.lm_head_split_sizes(151936, 4, overlay.lm_head_max_columns_per_device) == [4008] * 9 + [1912]


@pytest.mark.parametrize(
    "cluster_type",
    [ttnn.cluster.ClusterType.P150_X4, ttnn.cluster.ClusterType.P300_X2],
)
def test_blackhole_four_die_ccl_recipe_is_model_owned_ring(cluster_type, monkeypatch):
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: cluster_type)
    assert _qwen3_ccl_topology(_FakeMesh()) == ttnn.Topology.Ring


def test_qwen_ccl_recipe_rejects_unadmitted_bh_cluster(monkeypatch, expect_error):
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: ttnn.cluster.ClusterType.P150_X8)
    with expect_error(ValueError, "P150_X4/P300_X2"):
        _qwen3_ccl_topology(_FakeMesh())


def test_rope_uses_attention_decode_transformation_grid():
    source = inspect.getsource(Qwen3_32B.from_pretrained)

    assert "core_grid=sku.attention_decode_transformation_grid" in source


def test_blackhole_rope_resolves_to_attention_row_major_8x4_lane_grid(monkeypatch):
    monkeypatch.delenv("DISABLE_MINIMAL_MATMUL", raising=False)
    overlay = _resolve_qwen3_32b_sku_overlay(
        arch=ttnn.device.Arch.BLACKHOLE,
        cluster_type=ttnn.cluster.ClusterType.P150_X4,
        num_dev=4,
        mesh_device=_FakeMesh(),
    )
    table = LazyWeight(torch.zeros(1, 1, 128, 128))
    resolved = _resolve_rope_config(
        Rope1DConfig(
            cos_matrix=table,
            sin_matrix=table,
            max_batch_size=32,
            head_dim=128,
            device=object(),
            core_grid=overlay.attention_decode_transformation_grid,
        )
    )
    expected = ttnn.num_cores_to_corerangeset(32, ttnn.CoreCoord(8, 8), row_wise=True)

    assert resolved.batch_grid == expected
    assert resolved.decode_trans_mat_mem_config.shard_spec.grid == expected
    assert resolved.cos_sin_shard_mem_config.shard_spec.grid == expected


@pytest.mark.parametrize(
    "arch,num_devices",
    [(ttnn.device.Arch.WORMHOLE_B0, 4), (ttnn.device.Arch.BLACKHOLE, 8), (None, 4)],
)
def test_unsupported_architecture_sku_pairs_fail_closed(arch, num_devices, expect_error):
    cluster_type = (
        ttnn.cluster.ClusterType.T3K if arch == ttnn.device.Arch.WORMHOLE_B0 else ttnn.cluster.ClusterType.P150_X4
    )
    with expect_error(ValueError, "supports Wormhole T3K.*BlackHole P150_X4/P300_X2"):
        _resolve_qwen3_32b_sku_overlay(
            arch=arch, cluster_type=cluster_type, num_dev=num_devices, mesh_device=_FakeMesh()
        )


def test_blackhole_submesh_is_not_treated_as_physical_p150x4(expect_error):
    with expect_error(ValueError, "supports Wormhole T3K.*BlackHole P150_X4/P300_X2"):
        _resolve_qwen3_32b_sku_overlay(
            arch=ttnn.device.Arch.BLACKHOLE,
            cluster_type=ttnn.cluster.ClusterType.P150_X8,
            num_dev=4,
            mesh_device=_FakeMesh(),
        )


@pytest.mark.parametrize("arch,num_devices", [(ttnn.device.Arch.WORMHOLE_B0, 8), (ttnn.device.Arch.BLACKHOLE, 4)])
def test_model_helpers_write_explicit_recipes_on_common_configs(arch, num_devices, monkeypatch):
    monkeypatch.delenv("DISABLE_MINIMAL_MATMUL", raising=False)
    cluster_type = (
        ttnn.cluster.ClusterType.T3K if arch == ttnn.device.Arch.WORMHOLE_B0 else ttnn.cluster.ClusterType.P150_X4
    )
    overlay = _resolve_qwen3_32b_sku_overlay(
        arch=arch, cluster_type=cluster_type, num_dev=num_devices, mesh_device=_FakeMesh()
    )
    common_rms = RMSNorm1DConfig(weight=object())
    common_lm = LMHead1DConfig(output_weights=[])
    common_mlp = MLP1DConfig(
        w1=object(), w2=object(), w3=object(), prefill_w2_minimal_matmul=overlay.prefill_minimal_matmul
    )
    common_attention = Attention1DConfig(
        wqkv=object(), wo=object(), prefill_qkv_minimal_matmul=overlay.prefill_minimal_matmul
    )

    rms = _qwen3_rmsnorm_config(common_rms)
    lm_head = _qwen3_lm_head_config(common_lm)
    mlp = _qwen3_mlp_config(
        common_mlp,
        sku=overlay,
        precision=QWEN3_32B_ACCURACY,
    )
    attention = _qwen3_attention_config(
        common_attention,
        sku=overlay,
        precision=QWEN3_32B_ACCURACY,
    )

    assert isinstance(rms, RMSNorm1DConfig) and rms is not common_rms
    assert isinstance(lm_head, LMHead1DConfig) and lm_head is not common_lm
    assert isinstance(mlp, MLP1DConfig) and mlp is not common_mlp
    assert isinstance(attention, Attention1DConfig) and attention is not common_attention
    assert common_rms.compute_kernel_config is None
    assert common_lm.compute_kernel_config is None
    assert common_mlp.ff1_3_compute_kernel_cfg is None
    assert common_attention.li_qkv_decode_compute_kernel_cfg is None
    assert mlp.prefill_w2_minimal_matmul is True
    assert attention.prefill_qkv_minimal_matmul is True
    assert mlp.prefill_ff1_ff3_grid == overlay.mlp_prefill_grid
    assert mlp.prefill_ff2_grid == overlay.mlp_prefill_grid
    assert mlp.prefill_dram_shard_grid_width == overlay.dram_shard_grid_width
    assert attention.prefill_qkv_grid == overlay.attention_prefill_qkv_grid
    assert attention.dram_shard_grid_width == overlay.dram_shard_grid_width
    assert _kernel_semantics(rms.compute_kernel_config) == (
        ttnn.MathFidelity.HiFi2,
        False,
        True,
        True,
        False,
    )
    assert _kernel_semantics(lm_head.compute_kernel_config) == (
        ttnn.MathFidelity.HiFi2,
        False,
        False,
        True,
        False,
    )
    assert [_kernel_semantics(slot) for slot in _mlp_slots(QWEN3_32B_ACCURACY)] == [
        _kernel_semantics(mlp.ff1_3_compute_kernel_cfg),
        _kernel_semantics(mlp.ff2_compute_kernel_cfg),
        _kernel_semantics(mlp.decode_ff1_3_compute_kernel_cfg),
        _kernel_semantics(mlp.decode_ff2_compute_kernel_cfg),
    ]


def test_checked_in_qwen_config_retains_intermediate_size_25600():
    config_path = Path(__file__).parents[4] / "tt_transformers/model_params/Qwen3-32B/config.json"
    checked_in = json.loads(config_path.read_text())

    assert QWEN3_32B_INTERMEDIATE_SIZE == 25600
    assert checked_in["intermediate_size"] == QWEN3_32B_INTERMEDIATE_SIZE
