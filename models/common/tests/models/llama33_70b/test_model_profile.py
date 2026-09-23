# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pure semantic snapshots for the Llama-3.3-70B architecture/SKU profile."""

import inspect
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.models.llama33_70b.model import (
    LLAMA33_70B_ACCURACY,
    LLAMA33_70B_BH_TP4_CLUSTER_TYPES,
    LLAMA33_70B_PERFORMANCE,
    Llama33_70BLayerWeights,
    Llama33_70BModelParameters,
    Llama33_70BPagedAttentionConfig,
    _build_decoder_layer,
    _llama33_70b_ccl_topology,
    _resolve_llama33_70b_profile,
    build_llama33_70b_transformer_1d_config,
)
from models.common.modules.attention.attention_1d import Attention1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp.mlp_1d import MLP1DConfig
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1DConfig
from models.common.modules.rope.rope_1d import Rope1DConfig, _resolve_rope_config


def _semantics(config):
    return (
        config.math_fidelity,
        config.math_approx_mode,
        config.fp32_dest_acc_en,
        config.packer_l1_acc,
    )


def _cluster_type(arch):
    return ttnn.cluster.ClusterType.T3K if arch == ttnn.device.Arch.WORMHOLE_B0 else ttnn.cluster.ClusterType.P150_X4


@pytest.mark.parametrize(
    ("arch", "cluster_type", "devices", "expected_attention", "cutoff", "qkv_grid", "lm_columns"),
    [
        (
            ttnn.device.Arch.WORMHOLE_B0,
            ttnn.cluster.ClusterType.T3K,
            8,
            (ttnn.MathFidelity.HiFi2, False, False, True),
            1024,
            (8, 8),
            8192,
        ),
        (
            ttnn.device.Arch.BLACKHOLE,
            ttnn.cluster.ClusterType.P150_X4,
            4,
            (ttnn.MathFidelity.HiFi2, True, True, True),
            512,
            (8, 10),
            4008,
        ),
        (
            ttnn.device.Arch.BLACKHOLE,
            ttnn.cluster.ClusterType.P300_X2,
            4,
            (ttnn.MathFidelity.HiFi2, True, True, True),
            512,
            (8, 10),
            4008,
        ),
    ],
)
def test_accuracy_profile_semantic_snapshot(
    arch, cluster_type, devices, expected_attention, cutoff, qkv_grid, lm_columns
):
    profile = _resolve_llama33_70b_profile(
        arch=arch,
        cluster_type=cluster_type,
        num_devices=devices,
        dram_width=8,
        precision=LLAMA33_70B_ACCURACY,
    )

    ordinary_slots = (
        profile.model.li_qkv_decode,
        profile.model.sdpa_decode,
        profile.model.li_o_decode,
        profile.model.li_qkv_prefill,
        profile.model.li_o_prefill,
    )
    assert all(_semantics(slot) == expected_attention for slot in ordinary_slots)
    assert _semantics(profile.model.sdpa_prefill) == (ttnn.MathFidelity.HiFi4, False, True, True)
    assert _semantics(profile.model.prefill_ff1_ff3) == (ttnn.MathFidelity.HiFi2, False, False, True)
    assert _semantics(profile.model.prefill_ff2) == (ttnn.MathFidelity.HiFi2, False, False, True)
    assert _semantics(profile.model.rmsnorm) == (ttnn.MathFidelity.HiFi2, False, True, True)
    assert _semantics(profile.model.lm_head) == (ttnn.MathFidelity.HiFi2, False, False, True)
    assert profile.sku.mlp_prefill_len_cutoff == cutoff
    assert profile.sku.prefill_qkv_grid == qkv_grid
    assert profile.sku.lm_head_max_columns_per_device == lm_columns
    assert profile.sku.prefill_minimal_matmul


@pytest.mark.parametrize("cluster_type", LLAMA33_70B_BH_TP4_CLUSTER_TYPES)
def test_performance_profile_makes_all_four_mlp_slots_explicit(cluster_type):
    profile = _resolve_llama33_70b_profile(
        arch=ttnn.device.Arch.BLACKHOLE,
        cluster_type=cluster_type,
        num_devices=4,
        dram_width=8,
        precision=LLAMA33_70B_PERFORMANCE,
    )

    assert _semantics(profile.model.prefill_ff1_ff3) == (ttnn.MathFidelity.LoFi, False, False, True)
    assert _semantics(profile.model.decode_ff1_ff3) == (ttnn.MathFidelity.LoFi, False, False, True)
    assert _semantics(profile.model.prefill_ff2) == (ttnn.MathFidelity.HiFi2, False, False, True)
    assert _semantics(profile.model.decode_ff2) == (ttnn.MathFidelity.HiFi2, False, False, True)


def test_rope_uses_attention_decode_transformation_grid():
    source = inspect.getsource(build_llama33_70b_transformer_1d_config)

    assert "core_grid=profile.sku.decode_transformation_core_grid" in source


def test_blackhole_rope_resolves_to_attention_row_major_8x4_lane_grid():
    profile = _resolve_llama33_70b_profile(
        arch=ttnn.device.Arch.BLACKHOLE,
        cluster_type=ttnn.cluster.ClusterType.P150_X4,
        num_devices=4,
        dram_width=8,
        precision=LLAMA33_70B_ACCURACY,
    )
    table = LazyWeight(torch.zeros(1, 1, 128, 128))
    resolved = _resolve_rope_config(
        Rope1DConfig(
            cos_matrix=table,
            sin_matrix=table,
            max_batch_size=32,
            head_dim=128,
            device=object(),
            core_grid=profile.sku.decode_transformation_core_grid,
        )
    )
    expected = ttnn.num_cores_to_corerangeset(32, ttnn.CoreCoord(8, 8), row_wise=True)

    assert resolved.batch_grid == expected
    assert resolved.decode_trans_mat_mem_config.shard_spec.grid == expected
    assert resolved.cos_sin_shard_mem_config.shard_spec.grid == expected


@pytest.mark.parametrize(
    ("arch", "devices"),
    [
        (ttnn.device.Arch.WORMHOLE_B0, 8),
        (ttnn.device.Arch.BLACKHOLE, 4),
    ],
)
def test_decoder_builder_writes_explicit_recipes_on_common_configs(monkeypatch, arch, devices):
    profile = _resolve_llama33_70b_profile(
        arch=arch,
        cluster_type=_cluster_type(arch),
        num_devices=devices,
        dram_width=8,
        precision=LLAMA33_70B_ACCURACY,
    )
    mesh = SimpleNamespace(get_num_devices=lambda: devices)
    params = Llama33_70BModelParameters(
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        hidden_dim=28672,
        vocab_size=128256,
        rms_norm_eps=1e-5,
        max_batch_size=32,
        max_seq_len=4096,
    )
    tensor = torch.zeros(32, 32)
    weights = Llama33_70BLayerWeights(tensor, tensor, tensor, tensor, tensor, tensor, tensor)
    monkeypatch.setattr(
        "models.common.models.llama33_70b.model._post_attn_norm_decode_configs",
        lambda **_: (SimpleNamespace(), ttnn.DRAM_MEMORY_CONFIG),
    )

    block = _build_decoder_layer(
        idx=0,
        weights=weights,
        mcfg=params,
        mesh_device=mesh,
        tt_ccl=SimpleNamespace(),
        topology=ttnn.Topology.Ring,
        num_dev=devices,
        precision=LLAMA33_70B_ACCURACY,
        paged_attention_config=Llama33_70BPagedAttentionConfig(block_size=32, max_num_blocks=1),
        cache_path=None,
        profile=profile,
        decode_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
    )

    assert isinstance(block.attention_config, Attention1DConfig)
    assert isinstance(block.mlp_config, MLP1DConfig)
    assert isinstance(block.attention_norm_config, RMSNorm1DConfig)
    assert isinstance(block.ff_norm_config, RMSNorm1DConfig)
    assert block.attention_config.prefill_qkv_minimal_matmul
    assert block.mlp_config.prefill_w2_minimal_matmul
    assert block.attention_norm_config.prefill_distributed
    assert block.mlp_config.prefill_len_cutoff == profile.sku.mlp_prefill_len_cutoff
    assert block.attention_config.prefill_qkv_grid == profile.sku.prefill_qkv_grid
    assert _semantics(block.attention_config.sdpa_prefill_compute_kernel_cfg) == _semantics(profile.model.sdpa_prefill)
    assert _semantics(block.mlp_config.decode_ff2_compute_kernel_cfg) == _semantics(profile.model.decode_ff2)
    assert _semantics(block.attention_norm_config.compute_kernel_config) == _semantics(profile.model.rmsnorm)


def test_paged_attention_mutation_uses_common_block_contract():
    paged = Llama33_70BPagedAttentionConfig(block_size=32, max_num_blocks=1)
    common = SimpleNamespace(
        use_vllm_paged_kv_cache=True,
        paged_attention_config=paged,
        kv_cache=None,
    )
    live = SimpleNamespace(
        config=SimpleNamespace(
            use_vllm_paged_kv_cache=True,
            paged_attention_config=paged,
            kv_cache=None,
        ),
        kv_cache=None,
    )
    model = SimpleNamespace(
        config=SimpleNamespace(block_configs=(SimpleNamespace(attention_config=common),)),
        layers=(SimpleNamespace(attention=live),),
    )

    from models.common.models.llama33_70b.model import Llama33_70BTransformer1D

    Llama33_70BTransformer1D.configure_paged_attention(model, block_size=16, max_num_blocks=200)

    assert common.paged_attention_config.block_size == 16
    assert common.paged_attention_config.max_num_blocks == 200
    assert live.config.paged_attention_config.block_size == 16


def test_blackhole_profile_rejects_non_p150x4_geometry(expect_error):
    with expect_error(ValueError, "physical cluster"):
        _resolve_llama33_70b_profile(
            arch=ttnn.device.Arch.BLACKHOLE,
            cluster_type=ttnn.cluster.ClusterType.P150_X8,
            num_devices=4,
            dram_width=8,
            precision=LLAMA33_70B_ACCURACY,
        )
    with expect_error(ValueError, "requires 4 devices"):
        _resolve_llama33_70b_profile(
            arch=ttnn.device.Arch.BLACKHOLE,
            cluster_type=ttnn.cluster.ClusterType.P150_X4,
            num_devices=8,
            dram_width=8,
            precision=LLAMA33_70B_ACCURACY,
        )
    with expect_error(ValueError, "DRAM width 8"):
        _resolve_llama33_70b_profile(
            arch=ttnn.device.Arch.BLACKHOLE,
            cluster_type=ttnn.cluster.ClusterType.P150_X4,
            num_devices=4,
            dram_width=7,
            precision=LLAMA33_70B_ACCURACY,
        )


@pytest.mark.parametrize("cluster_type", LLAMA33_70B_BH_TP4_CLUSTER_TYPES)
def test_blackhole_four_die_products_use_exact_logical_tp4_ring(cluster_type, monkeypatch):
    mesh = SimpleNamespace(
        arch=lambda: ttnn.device.Arch.BLACKHOLE,
        get_num_devices=lambda: 4,
        shape=(1, 4),
    )
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: cluster_type)

    assert _llama33_70b_ccl_topology(mesh) == ttnn.Topology.Ring


@pytest.mark.parametrize(
    ("cluster_type", "num_devices", "mesh_shape"),
    [
        (ttnn.cluster.ClusterType.P150_X8, 4, (1, 4)),
        (ttnn.cluster.ClusterType.P150_X4, 8, (1, 8)),
        (ttnn.cluster.ClusterType.P300_X2, 4, (2, 2)),
    ],
)
def test_blackhole_ccl_rejects_product_count_and_logical_shape_mismatches(
    cluster_type, num_devices, mesh_shape, monkeypatch, expect_error
):
    mesh = SimpleNamespace(
        arch=lambda: ttnn.device.Arch.BLACKHOLE,
        get_num_devices=lambda: num_devices,
        shape=mesh_shape,
    )
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: cluster_type)

    with expect_error(ValueError, "P150_X4/P300_X2.*4-device.*\\(1, 4\\).*Ring"):
        _llama33_70b_ccl_topology(mesh)
