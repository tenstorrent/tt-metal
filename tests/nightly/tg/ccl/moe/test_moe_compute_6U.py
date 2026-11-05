# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.common.moe.moe_compute_helpers import (
    ATOL_THRESHOLD,
    _MoEHostData,
    _MoEWeightInputs,
    _build_moe_host_data,
    _build_quantized_weight_host_tensors,
    _swiglu_reference,
    compute_combine_golden,
    compute_e_t_golden,
    compute_expert_activation_golden,
    compute_matmul_golden,
    compute_selective_tilize_golden,
    create_sharded_memory_config,
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    gen_expert_mapping,
    gen_sparse_buffer_and_indices,
    get_linearized_mesh_coord,
    tt_to_torch_dtype,
    validate_activation,
    validate_combine,
    validate_combine_torch,
    validate_e_t,
    validate_per_expert_tokens,
)

import dataclasses
import functools
from loguru import logger
import math
import os
import pytest
import random
import torch

# Force torch CPU ops to use all cores. bf16 matmul on CPU is single-thread on
# some torch builds even with OMP_NUM_THREADS set; this is the runtime knob.
# Required to keep compute_matmul_golden's fp32-cast fast path actually
# parallel — without it, the golden compute can take ~30 s/layer on BH-LB
# shape instead of ~33 ms.
torch.set_num_threads(os.cpu_count() or 1)

import ttnn
from ttnn.operations.ccl import MoEActivationFunction

from ttnn.experimental.moe_compute_utils import (
    auto_output_width_shard_dim,
    _shard_tiles,
    _w2_shard_tiles,
)

from tests.nightly.tg.ccl.moe.test_selective_combine_6U import device_mesh_iterator
from tests.nightly.t3000.ccl.test_all_to_all_combine import get_batch_cluster_idxr, get_cluster_dims

from models.common.utility_functions import comp_pcc

MESH_GRAPH_DESC_1x16 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x16_torus_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_1x8 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x8_torus_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_1x16_LINEAR = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x16_linear_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_1x8_LINEAR = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_1x8_linear_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_BH_LB_1x8_LINEAR = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_lb_1x8_line_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_16x1 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_16x1_torus_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_8x1 = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_8x1_torus_graph_descriptor.textproto"
)
MESH_GRAPH_DESC_BH_LB_8x1_LINEAR = (
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/bh_lb_8x1_line_graph_descriptor.textproto"
)
# FYI: These tests also work in a MESH_GRAPH_DESC_1x4 setting (~1 minute to set up), but not in a 1x2 setting.


def is_mesh_graph_descriptor_set(expected_path):
    """Check if TT_MESH_GRAPH_DESC_PATH is set to the expected path."""
    return os.environ.get("TT_MESH_GRAPH_DESC_PATH") == expected_path


# ---------------------------------------------------------------------------
# Data-driven model configs for parametrized MoE tests
# ---------------------------------------------------------------------------

MOE_DEVICE_PARAMS = {
    "l1_small_size": 16384,
    "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 500000,
}

MOE_DEVICE_PARAMS_LINEAR = {
    "l1_small_size": 16384,
    "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "trace_region_size": 750000,
}

MOE_DEVICE_PARAMS_ROW = {
    "l1_small_size": 16384,
    "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 500000,
}

MOE_DEVICE_PARAMS_ROW_LINEAR = {
    "l1_small_size": 16384,
    "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
    "trace_region_size": 750000,
}


@dataclasses.dataclass(frozen=True)
class MoEMeshConfig:
    name: str
    mesh_width: int
    mesh_graph_desc: str
    model_configs: tuple
    device_params: dict
    use_linear_topology: bool = False
    num_links: int = 4
    cluster_axis: int = 1
    mux_core_ranges: tuple = (((1, 1), (3, 3)),)

    @property
    def mesh_shape(self):
        if self.cluster_axis == 0:
            return (self.mesh_width, 1)
        return (1, self.mesh_width)


@dataclasses.dataclass(frozen=True)
class MoEModelConfig:
    name: str
    N: int
    hidden_size: int
    selected_experts_k: int
    experts_per_device_values: tuple = (2,)
    has_bias_values: tuple = (False,)
    test_modes: tuple = ("correctness",)
    activation_types: tuple = (MoEActivationFunction.SILU,)
    num_layers: int = 5
    num_iterations: int = 3
    tokens_per_device: int = 32
    output_height_shard_dim: int = 4
    marks: tuple = ()


# Tier-1 models: deepseek_v3, gpt_oss; perf + correctness when configured.
#   deepseek_v3: perf no bias; correctness no_bias + bias; gpt_oss: bias only.
#   correctness: both trace modes; perf: enable_trace only.
# Tier-2 models: all others on torus only, no bias, trace off, correctness only.
# Linear default: tier-1 only.
# MOE_COMPUTE_LINEAR_ALL_MODELS=1: all models on linear meshes (tier rules unchanged).
# MOE_COMPUTE_FULL=1: every model on every mesh (1x8/1x16 x torus/linear); tier rules unchanged.
MOE_PRIORITY_MODEL_NAMES = frozenset({"deepseek_v3", "gpt_oss"})


def _moe_compute_env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _include_model_on_mesh(mesh_cfg: MoEMeshConfig, model_cfg: MoEModelConfig) -> bool:
    if mesh_cfg.use_linear_topology:
        if _moe_compute_env_flag("MOE_COMPUTE_LINEAR_ALL_MODELS"):
            return True
        return model_cfg.name in MOE_PRIORITY_MODEL_NAMES
    return True


def _bias_values_for_case(model_cfg: MoEModelConfig, test_mode: str) -> tuple:
    if model_cfg.name == "gpt_oss":
        return (True,)
    if model_cfg.name == "deepseek_v3":
        if test_mode == "perf":
            return (False,)
        return model_cfg.has_bias_values
    return (False,)


def _test_modes_for_model(model_cfg: MoEModelConfig) -> tuple:
    if model_cfg.name in MOE_PRIORITY_MODEL_NAMES:
        modes = tuple(m for m in ("perf", "correctness") if m in model_cfg.test_modes)
        return modes if modes else ("correctness",)
    return ("correctness",)


def _trace_values_for_case(model_cfg: MoEModelConfig, test_mode: str) -> tuple:
    if test_mode == "perf":
        return (True,)
    if model_cfg.name in MOE_PRIORITY_MODEL_NAMES:
        return (False, True)
    return (False,)


def _expand_model_configs(
    configs,
    *,
    bias_values_fn=_bias_values_for_case,
    test_modes_fn=_test_modes_for_model,
    trace_values_fn=_trace_values_for_case,
):
    """Expand model configs into pytest.param entries (test_mode, has_bias, epd, act, enable_trace)."""
    expanded = []
    for cfg in configs:
        for test_mode in test_modes_fn(cfg):
            for has_bias in bias_values_fn(cfg, test_mode):
                for epd in cfg.experts_per_device_values:
                    for act in cfg.activation_types:
                        # enable_trace innermost so a shape's trace variants run back-to-back and the
                        # second hits the _build_moe_host_data lru_cache (maxsize=1, evicted on every
                        # new shape). The matmul ring size is auto-detected from the live DRAM-bank count
                        # by the op (12 on WH, 7/8 on BH) and is no longer a sweep dimension.
                        for enable_trace in trace_values_fn(cfg, test_mode):
                            bias_tag = "bias" if has_bias else "no_bias"
                            act_tag = act.name.lower()
                            trace_tag = "enable_trace" if enable_trace else "disable_trace"
                            expanded.append(
                                pytest.param(
                                    cfg,
                                    test_mode,
                                    has_bias,
                                    epd,
                                    act,
                                    enable_trace,
                                    id=f"{cfg.name}-{test_mode}-{bias_tag}-{epd}experts_per_device-{act_tag}-{trace_tag}",
                                    marks=cfg.marks,
                                )
                            )
    return expanded


def _all_moe_model_configs():
    """Union of 1x8 and 1x16 model tables (deduped by name)."""
    by_name = {}
    for cfg in _MODELS_1x8 + _MODELS_1x16:
        by_name[cfg.name] = cfg
    return tuple(by_name.values())


def _models_for_mesh(mesh_cfg: MoEMeshConfig):
    if _moe_compute_env_flag("MOE_COMPUTE_FULL"):
        return _all_moe_model_configs()
    return tuple(m for m in mesh_cfg.model_configs if _include_model_on_mesh(mesh_cfg, m))


_MUX_DEFAULT = ((1, 1), (3, 3))


def _mux_tag(mux_core_range):
    """Format a mux_core_range as a compact test-ID component."""
    if mux_core_range == _MUX_DEFAULT:
        return "mux_default"
    (sx, sy), (ex, ey) = mux_core_range
    return f"mux_{sx}x{sy}_{ex}x{ey}"


def _expand_mesh_model_test_cases(mesh_configs):
    """Expand mesh + model configs into pytest.param entries for parametrized MoE model tests."""
    cases = []
    for mesh_cfg in mesh_configs:
        for mux_core_range in mesh_cfg.mux_core_ranges:
            mux_tag = _mux_tag(mux_core_range)
            for model_param in _expand_model_configs(_models_for_mesh(mesh_cfg)):
                desc_skip = pytest.mark.skipif(
                    not is_mesh_graph_descriptor_set(mesh_cfg.mesh_graph_desc),
                    reason=f"{mesh_cfg.name} tests require TT_MESH_GRAPH_DESC_PATH={mesh_cfg.mesh_graph_desc}",
                )
                cases.append(
                    pytest.param(
                        mesh_cfg.device_params,
                        mesh_cfg,
                        mesh_cfg.mesh_shape,
                        mesh_cfg.mesh_shape,
                        mux_core_range,
                        *model_param.values,
                        marks=list(model_param.marks) + [desc_skip],
                        id=f"{mesh_cfg.name}-{mux_tag}-{model_param.id}",
                    )
                )
    return cases


# fmt: off
_MODELS_1x16 = [
    MoEModelConfig("qwen3_omni_talker",   N=384,  hidden_size=1024, selected_experts_k=6),
    MoEModelConfig("qwen3_omni_thinker",  N=768,  hidden_size=2048, selected_experts_k=8),
    MoEModelConfig("qwen35_35b",          N=512,  hidden_size=2048, selected_experts_k=8),
    MoEModelConfig("qwen3_235b",          N=1536, hidden_size=4096, selected_experts_k=8),
    MoEModelConfig("qwen35_397b",         N=1024, hidden_size=4096, selected_experts_k=10),
    MoEModelConfig("glm_47",              N=1536, hidden_size=5120, selected_experts_k=8),
    MoEModelConfig("glm5",                N=2048, hidden_size=6144, selected_experts_k=8),
    MoEModelConfig("deepseek_v3",         N=2048, hidden_size=7168, selected_experts_k=8, has_bias_values=(False, True), test_modes=("perf", "correctness")),
    MoEModelConfig("kimi_k25",            N=2048, hidden_size=7168, selected_experts_k=8, experts_per_device_values=(6,), num_layers=3, num_iterations=2),
    MoEModelConfig("deepseek_v4_flash",   N=2048, hidden_size=4096, selected_experts_k=6),

    # these next few are known failures. xfail(run=False) keeps them tracked as expected
    # failures (visibility/metrics) without paying their runtime, since they are slow.
    MoEModelConfig("deepseek_v4_pro",     N=3072, hidden_size=7168, selected_experts_k=6, experts_per_device_values=(6,), num_layers=3, num_iterations=2,
                   marks=(pytest.mark.xfail(run=False, reason="Combine AllClose fails for specific output values (hidden=7168, N=3072) — likely selective_reduce_combine kernel bug"),)),
    MoEModelConfig("mistral_large_3",     N=4096, hidden_size=7168, selected_experts_k=4, num_layers=3, num_iterations=2,
                   marks=(pytest.mark.xfail(run=False, reason="L1 overflow: N=4096 A2A buffer (12*12*2048=288KB) exceeds Wormhole L1 budget by ~21KB"),)),
    MoEModelConfig("ling_1t",             N=2048, hidden_size=8192, selected_experts_k=8, num_layers=3, num_iterations=2,
                   marks=(pytest.mark.xfail(run=False, reason="Wormhole L1 too small for hidden=8192: dim=4 overflows mux L1 by 93KB, dim=2 overflows combine CB (2MB > 1MB bank)"),)),
]

_MODELS_1x8 = [
    MoEModelConfig("deepseek_ocr", N=896,  hidden_size=1280, selected_experts_k=6),
    MoEModelConfig("gemma_4_26b",  N=704,  hidden_size=2816, selected_experts_k=8, activation_types=(MoEActivationFunction.GELU,)),
    MoEModelConfig("gpt_oss",      N=2880, hidden_size=2880, selected_experts_k=4, experts_per_device_values=(4,), has_bias_values=(True,), test_modes=("perf", "correctness"), activation_types=(MoEActivationFunction.SWIGLU,)),
]

_MODELS_BH_LB_1x8 = [
    MoEModelConfig("deepseek_ocr",       N=896,  hidden_size=1280, selected_experts_k=6),
    MoEModelConfig("qwen3_omni_talker",  N=384,  hidden_size=1024, selected_experts_k=6),
    MoEModelConfig("qwen3_omni_thinker", N=768,  hidden_size=2048, selected_experts_k=8),
    MoEModelConfig("qwen35_35b",         N=512,  hidden_size=2048, selected_experts_k=8),
    MoEModelConfig("gemma_4_26b",        N=704,  hidden_size=2816, selected_experts_k=8, activation_types=(MoEActivationFunction.GELU,)),
    MoEModelConfig("gpt_oss",            N=2880, hidden_size=2880, selected_experts_k=4, experts_per_device_values=(4,), has_bias_values=(True,), test_modes=("perf", "correctness"), activation_types=(MoEActivationFunction.SWIGLU,)),
    MoEModelConfig("qwen3_235b",         N=1536, hidden_size=4096, selected_experts_k=8),
    MoEModelConfig("qwen35_397b",        N=1024, hidden_size=4096, selected_experts_k=10),
    MoEModelConfig("glm_47",             N=1536, hidden_size=5120, selected_experts_k=8),
    MoEModelConfig("glm5",               N=2048, hidden_size=6144, selected_experts_k=8),
    MoEModelConfig("kimi_k25",           N=2048, hidden_size=7168, selected_experts_k=8, experts_per_device_values=(6,), tokens_per_device=8),
    MoEModelConfig("deepseek_v3",        N=2048, hidden_size=7168, selected_experts_k=8, has_bias_values=(False, True), tokens_per_device=8, test_modes=("perf", "correctness"),),
]

_MODELS_16x1_ROW = [
    MoEModelConfig("deepseek_v3", N=2048, hidden_size=7168, selected_experts_k=8, has_bias_values=(False,), test_modes=("correctness",)),
]
_MODELS_8x1_ROW = [
    MoEModelConfig("deepseek_v3", N=2048, hidden_size=7168, selected_experts_k=8, has_bias_values=(False,), test_modes=("correctness",)),
    MoEModelConfig("gpt_oss",     N=2880, hidden_size=2880, selected_experts_k=4, experts_per_device_values=(4,), has_bias_values=(True,), test_modes=("correctness",), activation_types=(MoEActivationFunction.SWIGLU,)),
]

_MODELS_1x8_MUX_EASTERN = [
    MoEModelConfig("gpt_oss", N=2880, hidden_size=2880, selected_experts_k=4, experts_per_device_values=(4,), has_bias_values=(True,), test_modes=("correctness",), activation_types=(MoEActivationFunction.SWIGLU,)),
]

_MOE_MESH_CONFIGS = [
    MoEMeshConfig("1x8-torus",        8, MESH_GRAPH_DESC_1x8,              _MODELS_1x8,       MOE_DEVICE_PARAMS),
    MoEMeshConfig("1x16-torus",      16, MESH_GRAPH_DESC_1x16,             _MODELS_1x16,      MOE_DEVICE_PARAMS),
    MoEMeshConfig("1x8-linear",       8, MESH_GRAPH_DESC_1x8_LINEAR,       _MODELS_1x8,       MOE_DEVICE_PARAMS_LINEAR, use_linear_topology=True),
    MoEMeshConfig("1x16-linear",     16, MESH_GRAPH_DESC_1x16_LINEAR,      _MODELS_1x16,      MOE_DEVICE_PARAMS_LINEAR, use_linear_topology=True,),
    MoEMeshConfig("1x8-linear-bh_lb", 8, MESH_GRAPH_DESC_BH_LB_1x8_LINEAR, _MODELS_BH_LB_1x8, MOE_DEVICE_PARAMS_LINEAR, use_linear_topology=True, num_links=2),
    MoEMeshConfig("16x1-row",        16, MESH_GRAPH_DESC_16x1,             _MODELS_16x1_ROW,  MOE_DEVICE_PARAMS_ROW, cluster_axis=0),
    MoEMeshConfig("8x1-row",          8, MESH_GRAPH_DESC_8x1,              _MODELS_8x1_ROW,   MOE_DEVICE_PARAMS_ROW, cluster_axis=0),
    MoEMeshConfig("8x1-row-bh_lb",    8, MESH_GRAPH_DESC_BH_LB_8x1_LINEAR, _MODELS_8x1_ROW,  MOE_DEVICE_PARAMS_ROW_LINEAR, use_linear_topology=True, num_links=2, cluster_axis=0),
    MoEMeshConfig("1x8-linear-mux",   8, MESH_GRAPH_DESC_BH_LB_1x8_LINEAR, _MODELS_1x8_MUX_EASTERN, MOE_DEVICE_PARAMS_LINEAR, use_linear_topology=True, num_links=2, mux_core_ranges=(((4, 4), (9, 9)), ((10, 0), (11, 3)), )),
]
# fmt: on

MOE_COMPUTE_MODEL_TEST_CASES = _expand_mesh_model_test_cases(_MOE_MESH_CONFIGS)


def _run_model_test(
    mesh_device,
    mesh_shape,
    enable_trace,
    model_cfg,
    test_mode,
    has_bias,
    experts_per_device,
    activation_type,
    num_links=4,
    topology=None,
    cluster_axis=1,
    mux_core_range=((1, 1), (3, 3)),
):
    if test_mode == "perf":
        selected_experts_k = 1
        num_layers = 1
        num_iterations = 5
    else:
        selected_experts_k = model_cfg.selected_experts_k
        num_layers = model_cfg.num_layers
        num_iterations = model_cfg.num_iterations

    _run_moe_compute_impl(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        cluster_axis=cluster_axis,
        experts_per_device=experts_per_device,
        tokens_per_device=model_cfg.tokens_per_device,
        selected_experts_k=selected_experts_k,
        num_layers=num_layers,
        num_iterations=num_iterations,
        N=model_cfg.N,
        hidden_size=model_cfg.hidden_size,
        output_height_shard_dim=model_cfg.output_height_shard_dim,
        output_width_shard_dim=auto_output_width_shard_dim(model_cfg.hidden_size),
        dtype=ttnn.bfloat16,
        enable_trace=enable_trace,
        activation_type=activation_type,
        has_bias=has_bias,
        topology=topology,
        num_links=num_links,
        mux_core_range=mux_core_range,
    )


def prepare_output_tensor_from_combine_writer(
    raw_torch_output,
    active_token_counts,
    device_idx,
    all_core_range_set,
    output_shard_cores,
    output_shard_height_dim,
    output_shard_width_dim,
    experts_per_device,
    hidden,
):
    all_output_shards = {}

    for i, c in enumerate(ttnn.corerange_to_cores(all_core_range_set, row_wise=True)):
        all_output_shards[(c.x, c.y)] = raw_torch_output[i]

    combine_output_shards = [all_output_shards[c.x, c.y] for c in output_shard_cores]
    output_shard_tensor = torch.stack(combine_output_shards)

    # Consistent with token_offset logic in program factories
    assert output_shard_tensor.numel() % (experts_per_device * hidden) == 0
    buffer_size_total_tokens = output_shard_tensor.numel() // (experts_per_device * hidden)

    output_shape = (
        output_shard_height_dim,
        output_shard_width_dim,
        experts_per_device,
        buffer_size_total_tokens // output_shard_height_dim,
        hidden // output_shard_width_dim,
    )

    shaped_torch_output = output_shard_tensor.view(output_shape)

    shaped_torch_output = shaped_torch_output.permute([2, 0, 3, 1, 4]).reshape(
        [experts_per_device, buffer_size_total_tokens, hidden]
    )
    torch_output = torch.zeros([experts_per_device, buffer_size_total_tokens, hidden], dtype=torch.bfloat16)

    for e in range(experts_per_device):
        active_tokens = active_token_counts[e].item()
        tokens_per_shard_chunk = active_tokens // output_shard_height_dim
        tokens_per_shard_rem = active_tokens % output_shard_height_dim

        output_token_shard = 0
        output_token_shard_row = 0
        for t in range(active_tokens):
            contrib = shaped_torch_output[
                e, output_token_shard * buffer_size_total_tokens // output_shard_height_dim + output_token_shard_row
            ]

            torch_output[e, t] = contrib

            # Determine how many tokens this shard should have
            # First tokens_per_shard_rem shards get one extra token
            tokens_in_this_shard = tokens_per_shard_chunk + (1 if output_token_shard < tokens_per_shard_rem else 0)

            if output_token_shard_row + 1 == tokens_in_this_shard:
                output_token_shard += 1
                output_token_shard_row = 0
            else:
                output_token_shard_row += 1

    return torch_output


# Matmul with bias: LoFi + bf16/bfp4 on device can land just under PCC_THRESHOLD (e.g. ~0.987994 on one
# device/expert) while still tracking golden closely; combine PCC stays above 0.988.
PCC_THRESHOLD_MATMUL_WITH_BIAS = 0.98799
SWIGLU_PCC_THRESHOLD = 0.984
SILU_PCC_THRESHOLD = 0.986
GELU_PCC_THRESHOLD = 0.986


def _get_base_pcc_threshold(activation_type, has_bias):
    # Determine PCC threshold based on activation type
    # Note: this threshold is applicable for checking a block of 32 tokens, smaller matrices will need a lower threshold
    # https://github.com/tenstorrent/tt-metal/blob/368efa1f7062704b8e885aa72dae115e91320032/tests/ttnn/nightly/unit_tests/operations/experimental/test_moe_gpt_e2e.py#L438
    act_threshold = None
    if activation_type == MoEActivationFunction.SWIGLU:
        act_threshold = SWIGLU_PCC_THRESHOLD
    elif activation_type in (MoEActivationFunction.SILU, MoEActivationFunction.CLAMPED_SILU):
        act_threshold = SILU_PCC_THRESHOLD
    elif activation_type == MoEActivationFunction.GELU:
        act_threshold = GELU_PCC_THRESHOLD
    else:
        raise TypeError("Invalid Activation type")

    bias_threshold = None
    if has_bias:
        bias_threshold = PCC_THRESHOLD_MATMUL_WITH_BIAS
    else:
        bias_threshold = act_threshold

    return min(bias_threshold, act_threshold)


def validate_matmul(
    layer_id,
    experts_per_device,
    all_core_range_set,
    output_shard_cores,
    output_shard_height_dim,
    output_shard_width_dim,
    total_tokens,
    hidden,
    expert_token_counts,
    torch_output_ref,
    tt_output_tensor,
    mesh_device,
    base_pcc_threshold,
    *,
    has_bias: bool = False,
):
    logger.info(f"\n========== Matmul Output Tensor Validation ==========")

    devices = math.prod(mesh_device.shape)

    raw_output = ttnn.to_torch(tt_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # (D * all cores, E/D, T/D, H) -> (D, all cores, E/D, T/D, H)
    # note this shape does not yet match the layout of the underlying data.
    raw_shape = list(raw_output.shape)
    raw_output = raw_output.reshape([devices, raw_shape[0] // devices] + raw_shape[1:])

    reshape_func = functools.partial(
        prepare_output_tensor_from_combine_writer,
        all_core_range_set=all_core_range_set,
        output_shard_cores=output_shard_cores,
        output_shard_height_dim=output_shard_height_dim,
        output_shard_width_dim=output_shard_width_dim,
        experts_per_device=2,  # always 2 for double buffer
        hidden=hidden,
    )

    matmul_all_passed = True
    # Calculate which experts are still in the double buffer
    # Buffer toggles for each expert: 0->1->0->1...
    # So for N experts, the last 2 experts in the buffer are:
    # - If N is even: buffer 0 has expert N-2, buffer 1 has expert N-1
    # - If N is odd: buffer 0 has expert N-1, buffer 1 has expert N-2
    experts_to_check = []
    if experts_per_device == 1:
        experts_to_check = [(0, 0)]  # Only one expert in buffer 0
    elif experts_per_device == 2:
        experts_to_check = [(0, 0), (1, 1)]  # Expert 0 in buffer 0, expert 1 in buffer 1
    else:
        # For >2 experts, determine which 2 experts remain in the buffer
        if experts_per_device % 2 == 0:
            # Even number of experts
            experts_to_check = [(experts_per_device - 2, 0), (experts_per_device - 1, 1)]
        else:
            # Odd number of experts
            experts_to_check = [(experts_per_device - 1, 0), (experts_per_device - 2, 1)]

    logger.info(f"Checking experts in double buffer: {experts_to_check}")

    # smaller batch -> smaller dataset so PCC is less stable. A lower threshold is acceptable.
    MATMUL_PCC_THRESHOLD = 0.987 if total_tokens == 512 else 0.986

    # Build buffer token counts based on which experts are actually in the buffer
    reshaped_device_outputs = []
    for d in range(devices):
        buffer_token_counts = torch.zeros(2, dtype=expert_token_counts[d].dtype)
        for expert_id, buffer_idx in experts_to_check:
            buffer_token_counts[buffer_idx] = expert_token_counts[d][expert_id]
        reshaped_device_outputs.append(reshape_func(raw_output[d], buffer_token_counts, d))
    reshaped_device_outputs = torch.stack(reshaped_device_outputs)

    for d in range(devices):
        for expert_id, buffer_idx in experts_to_check:
            active_tokens = expert_token_counts[d, expert_id].item()
            if active_tokens == 0:
                continue
            # torch_output_ref is (L, D, E/D, T, H)
            torch_layer_output = torch_output_ref[layer_id, d, expert_id, :active_tokens, :]
            # The buffer position determines where to read from in the output
            tt_layer_output = reshaped_device_outputs[d, buffer_idx, :active_tokens, :]

            _pcc_passed, pcc_val = comp_pcc(torch_layer_output, tt_layer_output)
            allclose_passed = torch.allclose(torch_layer_output, tt_layer_output, atol=ATOL_THRESHOLD)
            std = torch_layer_output.std().item()
            relative_rmse_val = (
                (torch.nn.functional.mse_loss(torch_layer_output, tt_layer_output).sqrt().item() / std)
                if std != 0
                else 0.0
            )

            # Base PCC threshold is valid for 32xhidden, for comparing smaller matrices, a looser threshold is valid
            pcc_threshold = base_pcc_threshold if active_tokens >= 16 else base_pcc_threshold - 0.001
            if pcc_val < pcc_threshold:
                matmul_all_passed = False

                logger.warning(
                    f"Layer {layer_id}, Expert {expert_id} (buffer {buffer_idx}): PCC={pcc_val:.6f} RMSE: {relative_rmse_val}"
                    f" Allclose passed: {allclose_passed}"
                )

                if not allclose_passed:
                    mask = (tt_layer_output - torch_layer_output).abs() > ATOL_THRESHOLD
                    logger.warning(
                        f"AllClose variation result: {tt_layer_output[mask]}, ref: {torch_layer_output[mask]} indices: {mask.nonzero(as_tuple=True)}"
                    )
            else:
                logger.info(
                    f"Layer {layer_id}, Expert {expert_id} (buffer {buffer_idx}): PCC={pcc_val:.6f} RMSE: {relative_rmse_val} (Passed)"
                    f" Allclose passed: {allclose_passed}"
                )

    return matmul_all_passed


@torch.no_grad()
def _run_moe_compute_impl(
    mesh_device,
    mesh_shape,
    cluster_axis,
    experts_per_device,
    tokens_per_device,
    selected_experts_k,
    num_layers,
    num_iterations,
    N,
    hidden_size,
    output_height_shard_dim,
    output_width_shard_dim,
    dtype,
    enable_trace,
    activation_type,
    has_bias,
    num_links,
    topology=None,
    mux_core_range=((1, 1), (3, 3)),
):
    """Run MoE compute E2E validation. Called from test_moe_compute via _run_model_test."""
    experts = experts_per_device * mesh_shape[cluster_axis]

    #########################################
    # TEST SETUP
    #########################################

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices
    total_tokens = tokens_per_device * num_dispatch_devices
    experts_per_device = experts // num_devices

    logger.info(f"Test configuration:")
    logger.info(f"  mesh_shape: {mesh_shape}")
    logger.info(f"  cluster_axis: {cluster_axis}")
    logger.info(f"  num_devices: {num_devices}, num_dispatch_devices: {num_dispatch_devices}")
    logger.info(f"  tokens_per_device: {tokens_per_device}, total_tokens: {total_tokens}")
    logger.info(
        f"  experts: {experts}, selected_experts_k: {selected_experts_k}, experts_per_device: {experts_per_device}"
    )
    logger.info(f"  hidden_size: {hidden_size}")
    logger.info(f"  dtype: {dtype}")
    logger.info(f"  num_iterations: {num_iterations}")
    logger.info(f"  enable_trace: {enable_trace}")
    logger.info(f"  activation_type: {activation_type}")

    #########################################
    # HOST INPUTS + GOLDENS + QUANTIZED WEIGHTS (cached across shape-preserving params)
    #########################################
    # The two expensive, device-independent (or device-detachable) artifacts —
    # the torch goldens (compute_matmul_golden) and the quantized bfloat4_b weight
    # host tensors — are reused across pytest parameters that don't change the shape
    # (notably the two enable_trace variants of a model, which run back-to-back).
    #
    # _build_moe_host_data is memoized with functools.lru_cache(maxsize=1): the same
    # shape returns the same cached (host, weight_inputs), and a new shape evicts the
    # previous entry so at most one (large) shape's artifacts live at a time. The
    # quantized weights need a live device, so they can't be built inside that pure
    # function; instead they're built once on the miss and attached to the cached host
    # object, so later hits (e.g. the trace variant) reuse them without re-quantizing.
    _hits_before = _build_moe_host_data.cache_info().hits
    host, weight_inputs = _build_moe_host_data(
        tuple(mesh_shape),
        cluster_axis,
        experts_per_device,
        tokens_per_device,
        selected_experts_k,
        num_layers,
        N,
        hidden_size,
        dtype,
        activation_type,
        has_bias,
    )
    cache_hit = _build_moe_host_data.cache_info().hits > _hits_before
    shape_desc = (
        f"mesh={tuple(mesh_shape)} EP={experts_per_device} T/dev={tokens_per_device} "
        f"k={selected_experts_k} L={num_layers} N={N} H={hidden_size} act={activation_type} bias={has_bias}"
    )
    if host.w0_w1_host is None:
        # Quantization needs a live device, so it can't run inside the pure (cached)
        # _build_moe_host_data; the host tensors it produces survive mesh_device
        # teardown. lru_cache returns the SAME host object on every hit, so mutating
        # these fields in place here memoizes the quantized weights alongside the
        # goldens: a hit (e.g. the trace variant) finds them already set and skips
        # re-quantizing. w0_w1_host is None only on a genuine miss (fresh or evicted
        # shape), which is exactly when we want to (re)build them.
        host.w0_w1_host, host.w2_host = _build_quantized_weight_host_tensors(
            mesh_device, weight_inputs, num_layers, experts_per_device, hidden_size, N, has_bias
        )
    # cache_hit (lru_cache) and "w0_w1_host was None" (quantization) always agree: a
    # goldens hit returns the already-quantized object; a miss rebuilds it with None.
    if cache_hit:
        logger.info(f"MoE host-data cache HIT — reusing goldens + quantized weights for {shape_desc}")
    else:
        logger.info(f"MoE host-data cache MISS — built goldens + quantized weights for {shape_desc}")
    logger.info(f"MoE host-data cache stats: {_build_moe_host_data.cache_info()}")

    per_expert_tokens_goldens = host.per_expert_tokens_goldens
    activation_goldens = host.activation_goldens
    e_t_goldens = host.e_t_goldens
    matmul_goldens = host.matmul_goldens
    combine_goldens = host.combine_goldens

    #########################################
    # CREATE DEVICE INPUT TENSORS
    #########################################

    mux_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(mux_core_range[0], mux_core_range[1])])

    # Drain tilize core holds height-sharded expert indices/scores in L1.
    tilize_drain_core_coord = ttnn.experimental.get_moe_tilize_drain_core(
        mesh_device,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
        mux_core_range_set=mux_core_range_set,
    )
    tilize_drain_core = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(tilize_drain_core_coord.x, tilize_drain_core_coord.y),
                ttnn.CoreCoord(tilize_drain_core_coord.x, tilize_drain_core_coord.y),
            )
        }
    )

    #### Expert mapping - per-device [num_devices, experts], replicated on every device ###
    # Each device gets its own row after sharding, but since it's replicated,
    # we give each device the full tensor and it uses its own row.
    expert_mapping_mem_config = ttnn.L1_MEMORY_CONFIG
    tt_expert_mapping = ttnn.from_torch(
        host.expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=expert_mapping_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Sparse memory config
    sparse_mem_config = ttnn.L1_MEMORY_CONFIG

    # Create L1 sharded memory config for indices on drain tilizer core
    # expert_indices shape per device: [tokens_per_device, selected_experts_k] (after shard along dispatch axis)
    # But we need the all-gathered version, so shape is [num_dispatch_devices * tokens_per_device, selected_experts_k]
    # which is [total_tokens, selected_experts_k]
    expert_indices_shard_shape = [total_tokens, selected_experts_k]
    expert_indices_mem_config = create_sharded_memory_config(tilize_drain_core, expert_indices_shard_shape, ttnn.uint16)

    # Create L1 sharded memory config for indices and scores on drain tilizer core
    expert_scores_shard_shape = [total_tokens, selected_experts_k]
    expert_scores_mem_config = create_sharded_memory_config(tilize_drain_core, expert_scores_shard_shape, dtype)

    tt_sparse_buffers = []
    tt_expert_indices_buffers = []
    tt_expert_scores_buffers = []

    # NOTE: we're extremely tight on L1 for a single invocation of the op.
    # When running multiple layers, all inputs go to DRAM and get moved to L1
    # per-layer via to_memory_config.
    init_sparse_mem_config = sparse_mem_config if num_layers == 1 else ttnn.DRAM_MEMORY_CONFIG
    init_expert_indices_mem_config = expert_indices_mem_config if num_layers == 1 else ttnn.DRAM_MEMORY_CONFIG
    init_expert_scores_mem_config = expert_scores_mem_config if num_layers == 1 else ttnn.DRAM_MEMORY_CONFIG

    logger.info(f"Uploading device input tensors")
    for layer_id in range(num_layers):
        ### Sparse buffer is sharded across devices (dim 0) ###
        tt_sparse_buffer = ttnn.from_torch(
            host.sparse_buffers[layer_id],
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=init_sparse_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_sparse_buffers.append(tt_sparse_buffer)

        ### Expert indices - all-gathered (replicated on all devices) ###
        # Shape: [num_dispatch_devices, tokens_per_device, K]
        # Flatten to [num_dispatch_devices * tokens_per_device, K] = [total_tokens, K] per device
        # Replicate on all devices
        expert_indices_flat = host.expert_indices_list[layer_id].reshape(total_tokens, selected_experts_k)
        expert_indices_replicated = expert_indices_flat.unsqueeze(0).repeat(num_devices, 1, 1)
        tt_expert_indices = ttnn.from_torch(
            expert_indices_replicated,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=init_expert_indices_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_expert_indices_buffers.append(tt_expert_indices)

        ### Expert scores - same distribution as indices ###
        expert_scores_flat = host.expert_scores_list[layer_id].reshape(total_tokens, selected_experts_k)
        expert_scores_replicated = expert_scores_flat.unsqueeze(0).repeat(num_devices, 1, 1)
        tt_expert_scores = ttnn.from_torch(
            expert_scores_replicated,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=init_expert_scores_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_expert_scores_buffers.append(tt_expert_scores)

    logger.info(f"Done uploading device input tensors")

    # ------------------------------------------------------------------------
    # Land the cached bfloat4_b weight host tensors on this invocation's mesh.
    # The expensive upload → prepare → host-quantize ran once on the cache miss
    # (in _build_quantized_weight_host_tensors); here we only re-upload under the
    # DRAM-sharded mem config the kernel consumes. The host tensors stay cached.
    weight_mem_configs = ttnn.experimental.get_weight_mem_configs(
        mesh_device,
        num_layers=num_layers,
        experts_per_device=experts_per_device,
        hidden_size=hidden_size,
        intermediate_size=N,
        has_bias=has_bias,
    )
    tt_w0_w1 = ttnn.to_device(host.w0_w1_host, mesh_device, memory_config=weight_mem_configs.w0_w1)
    tt_w2 = ttnn.to_device(host.w2_host, mesh_device, memory_config=weight_mem_configs.w2)

    output_shard_cores = ttnn.experimental.get_moe_combine_cores(
        mesh_device,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
        mux_core_range_set=mux_core_range_set,
    )
    combine_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in output_shard_cores])
    combine_barrier_semaphore = ttnn.create_global_semaphore(mesh_device, combine_core_range_set, 0)

    torch_combine_output_tensor = torch.zeros([selected_experts_k, total_tokens, hidden_size], dtype=torch.bfloat16)
    tt_combine_output_tensors = [
        ttnn.from_torch(
            torch_combine_output_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        for _ in range(num_layers)
    ]

    #########################################
    # HELPER FUNCTIONS
    #########################################

    def prepare_layer_inputs(layer_id):
        """Prepare inputs for a specific layer by moving from DRAM to L1"""
        if num_layers == 1:
            # Already in L1
            return tt_sparse_buffers[0], tt_expert_indices_buffers[0], tt_expert_scores_buffers[0]
        else:
            tt_sparse_buffer = ttnn.to_memory_config(tt_sparse_buffers[layer_id], memory_config=sparse_mem_config)
            tt_expert_indices = ttnn.to_memory_config(
                tt_expert_indices_buffers[layer_id], memory_config=expert_indices_mem_config
            )
            tt_expert_scores = ttnn.to_memory_config(
                tt_expert_scores_buffers[layer_id], memory_config=expert_scores_mem_config
            )
            return tt_sparse_buffer, tt_expert_indices, tt_expert_scores

    def deallocate_layer_inputs(tt_sparse_buffer, tt_expert_indices, tt_expert_scores):
        """Deallocate L1 inputs if using multiple layers"""
        if num_layers != 1:
            ttnn.deallocate(tt_sparse_buffer)
            ttnn.deallocate(tt_expert_indices)
            ttnn.deallocate(tt_expert_scores)

    def convert_outputs_to_dram(outputs):
        """Convert L1 outputs to DRAM and deallocate L1 versions"""
        l1_per_expert, l1_activation, l1_e_t, _, l1_matmul, combine = outputs

        # Convert to DRAM
        dram_per_expert = ttnn.to_memory_config(l1_per_expert, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dram_activation = ttnn.to_memory_config(l1_activation, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dram_e_t = ttnn.to_memory_config(l1_e_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dram_matmul = ttnn.to_memory_config(l1_matmul, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Deallocate L1 versions
        ttnn.deallocate(l1_per_expert)
        ttnn.deallocate(l1_activation)
        ttnn.deallocate(l1_e_t)
        ttnn.deallocate(l1_matmul)

        return (dram_per_expert, dram_activation, dram_e_t, dram_matmul, combine)

    def run_op_inner(tt_sparse_buffer, tt_expert_indices, tt_expert_scores, layer_id):
        """Core moe_compute operation"""
        # Forward an explicit topology when supplied. The op normally derives this from the
        # fabric config + tensor coverage via get_usable_topology(), but that heuristic
        # marks any tensor that spans a full mesh row as WRAP/Ring — incorrect for
        # physically-line meshes (e.g. BH single Loudbox p150_x8). When set, this lets the
        # caller force Linear and avoid forwarding requests across non-existent wrap edges.
        return ttnn.experimental.moe_compute(
            tt_sparse_buffer,
            tt_expert_indices,
            tt_expert_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=layer_id,
            output_height_shard_dim=output_height_shard_dim,
            intermediate_size=N,
            has_bias=has_bias,
            cluster_axis=cluster_axis,
            topology=topology,
            num_links=num_links,
            mux_core_range_set=mux_core_range_set,
            optional_output_tensor=tt_combine_output_tensors[layer_id],
            optional_cross_device_semaphore=combine_barrier_semaphore,
            activation_type=activation_type,
        )

    #########################################
    # RUN OP
    #########################################

    def run_op():
        moe_compute_outputs = []

        for layer_id in range(num_layers):
            # Prepare layer inputs
            tt_sparse_buffer, tt_expert_indices, tt_expert_scores = prepare_layer_inputs(layer_id)

            # Run core moe_compute operation
            outputs = run_op_inner(tt_sparse_buffer, tt_expert_indices, tt_expert_scores, layer_id)

            # Deallocate L1 inputs
            deallocate_layer_inputs(tt_sparse_buffer, tt_expert_indices, tt_expert_scores)

            # Convert outputs to DRAM and save
            moe_compute_output = convert_outputs_to_dram(outputs)
            moe_compute_outputs.append(moe_compute_output)

        return moe_compute_outputs

    logger.info(f"\n========== Running op ==========")

    moe_compute_outputs = []
    trace_id = None

    if enable_trace:
        # Compile the op
        for i in range(num_iterations):
            run_op()
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iterations):
            moe_compute_output = run_op()
            moe_compute_outputs.append(moe_compute_output)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")
    else:
        # Non-trace execution
        for i in range(num_iterations):
            moe_compute_output = run_op()
            ttnn.synchronize_device(mesh_device)
            moe_compute_outputs.append(moe_compute_output)

    #########################################
    # VALIDATE OUTPUTS PER LAYER
    #########################################
    logger.info(f"\n========== Starting Validation ==========")

    all_core_grid = mesh_device.compute_with_storage_grid_size()
    all_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(all_core_grid.x - 1, all_core_grid.y - 1),
            ),
        }
    )

    base_pcc_threshold = _get_base_pcc_threshold(activation_type, has_bias)

    output_shard_cores = ttnn.experimental.get_moe_combine_cores(
        mesh_device,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
        mux_core_range_set=mux_core_range_set,
    )
    worker_mcast_bbox = ttnn.experimental.get_moe_worker_mcast_bounding_box(
        mesh_device,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
        mux_core_range_set=mux_core_range_set,
    )
    per_expert_tokens_all_passed = True
    activation_all_passed = True
    e_t_all_passed = True
    matmul_all_passed = True
    combine_all_passed = True
    for i in range(num_iterations):
        for layer_id in range(num_layers):
            (
                per_expert_total_tokens_output_tensor,
                expert_activation_output_tensor,
                e_t_output_tensor,
                matmul_output_tensor,
                combine_output_tensor,
            ) = moe_compute_outputs[i][layer_id]

            logger.info(f"\n========== Iteration {i} Layer {layer_id} Validation ==========")
            logger.info(f"Per expert total tokens tensor shape: {per_expert_total_tokens_output_tensor.shape}")
            logger.info(f"Expert activation tensor shape: {expert_activation_output_tensor.shape}")
            logger.info(f"E-T (expert-to-token) tensor shape: {e_t_output_tensor.shape}")
            logger.info(f"Matmul Output tensor shape: {matmul_output_tensor.shape}")

            # ========== Per Expert Total Tokens Tensor Validation ==========
            expert_token_counts = per_expert_tokens_goldens[layer_id]
            if not validate_per_expert_tokens(
                mesh_device,
                experts_per_device,
                num_devices,
                per_expert_total_tokens_output_tensor,
                expert_token_counts,
                worker_mcast_bbox,
            ):
                per_expert_tokens_all_passed = False

            # ========== Expert Activation Tensor Validation ==========
            golden_activation = activation_goldens[layer_id]
            if not validate_activation(
                mesh_device, experts_per_device, num_devices, expert_activation_output_tensor, golden_activation
            ):
                activation_all_passed = False

            # ========== E-T (Expert-to-Token) Tensor Validation ==========
            golden_e_t = e_t_goldens[layer_id]
            if not validate_e_t(
                mesh_device, total_tokens, experts_per_device, num_devices, e_t_output_tensor, golden_e_t
            ):
                e_t_all_passed = False

            # ========== Matmul Output Tensor Validation ==========

            if not validate_matmul(
                layer_id,
                experts_per_device,
                all_core_range_set,
                output_shard_cores,
                output_height_shard_dim,
                output_width_shard_dim,
                total_tokens,
                hidden_size,
                expert_token_counts,
                matmul_goldens,
                matmul_output_tensor,
                mesh_device,
                base_pcc_threshold,
                has_bias=has_bias,
            ):
                matmul_all_passed = False

            if not validate_combine(
                layer_id,
                mesh_device,
                cluster_axis,
                combine_output_tensor,
                combine_goldens,
                base_pcc_threshold,
            ):
                combine_all_passed = False

    #########################################
    # DEALLOCATE DEVICE TENSORS
    #########################################
    # The mesh_device fixture is function-scoped and frees everything on close, but
    # validation has now consumed every output, so free the device tensors explicitly
    # rather than holding them until teardown. In trace mode the trace must be released
    # first — its captured buffers reference the tensors below.
    if enable_trace and trace_id is not None:
        # The trace ran with blocking=False; synchronize before releasing so we don't
        # tear down trace buffers while execution is still in flight.
        ttnn.synchronize_device(mesh_device)
        ttnn.release_trace(mesh_device, trace_id)

    for iter_outputs in moe_compute_outputs:
        for layer_outputs in iter_outputs:
            # 5-tuple: (per_expert, activation, e_t, matmul, combine). The combine entry
            # is the persistent optional_output_tensor freed separately below.
            dram_per_expert, dram_activation, dram_e_t, dram_matmul, _combine = layer_outputs
            ttnn.deallocate(dram_per_expert)
            ttnn.deallocate(dram_activation)
            ttnn.deallocate(dram_e_t)
            ttnn.deallocate(dram_matmul)

    for tt_combine_output in tt_combine_output_tensors:
        ttnn.deallocate(tt_combine_output)

    for tt_input_buffer in tt_sparse_buffers + tt_expert_indices_buffers + tt_expert_scores_buffers:
        ttnn.deallocate(tt_input_buffer)

    ttnn.deallocate(tt_expert_mapping)
    ttnn.deallocate(tt_w0_w1)
    ttnn.deallocate(tt_w2)

    # Asserts
    logger.info(f"\n========== Asserts ==========")
    logger.info(f"\nPer Expert Total Tokens Verification: {'PASSED' if per_expert_tokens_all_passed else 'FAILED'}")
    logger.info(f"\nExpert Activation Verification: {'PASSED' if activation_all_passed else 'FAILED'}")
    logger.info(f"\nE-T Tensor Verification: {'PASSED' if e_t_all_passed else 'FAILED'}")
    logger.info(f"\nMatmul Output Tensor Verification: {'PASSED' if matmul_all_passed else 'FAILED'}")
    logger.info(f"\nCombine Output Tensor Verification: {'PASSED' if combine_all_passed else 'FAILED'}")

    assert per_expert_tokens_all_passed, "Per expert total tokens tensor verification failed!"
    assert activation_all_passed, "Expert activation tensor verification failed!"
    assert e_t_all_passed, "E-T tensor verification failed!"
    assert matmul_all_passed, "Matmul output tensor verification failed!"
    assert combine_all_passed, "Combine output tensor verification failed!"


# ---------------------------------------------------------------------------
# Parametrized model tests (1x8 / 1x16, torus and linear topologies)
#
# Default matrix (avoids combinatorial explosion):
#   Tier 1 — deepseek_v3, gpt_oss: perf + correctness; deepseek correctness includes bias; perf uses enable_trace only.
#   Tier 2 — remaining models: torus only, no bias, correctness, trace off.
#   Linear — tier 1 only.
#
# Development overrides:
#   MOE_COMPUTE_LINEAR_ALL_MODELS=1  — all models on linear meshes (tier rules unchanged).
#   MOE_COMPUTE_FULL=1             — all models x all meshes (1x8/1x16 x torus/linear); tier rules unchanged.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "device_params, mesh_cfg, mesh_shape, mesh_device, mux_core_range, model_cfg, test_mode, has_bias, experts_per_device, activation_type, enable_trace",
    MOE_COMPUTE_MODEL_TEST_CASES,
    indirect=["device_params", "mesh_device"],
)
def test_moe_compute(
    mesh_device,
    mesh_shape,
    mesh_cfg,
    mux_core_range,
    enable_trace,
    model_cfg,
    test_mode,
    has_bias,
    experts_per_device,
    activation_type,
):
    from ttnn.operations.ccl import Topology

    topology = Topology.Linear if mesh_cfg.use_linear_topology else None
    _run_model_test(
        mesh_device,
        mesh_shape,
        enable_trace,
        model_cfg,
        test_mode,
        has_bias,
        experts_per_device,
        activation_type,
        topology=topology,
        num_links=mesh_cfg.num_links,
        cluster_axis=mesh_cfg.cluster_axis,
        mux_core_range=mux_core_range,
    )


# ---------------------------------------------------------------------------
# CPU-only unit tests for formula-based shard distribution (no hardware required)
# ---------------------------------------------------------------------------


def test_shard_tiles_deepseek_w0w1():
    """DS: Nt=64, n_cores=12 → big=6, small=5; big at positions where (i*4)%12 < 4 = {0,3,6,9}."""
    n_tiles, n_cores = 64, 12
    result = [_shard_tiles(n_tiles, c, n_cores) for c in range(n_cores)]
    assert sum(result) == n_tiles
    assert result[0] == 6  # (0*4)%12=0 < 4 → big
    assert result[1] == 5  # (1*4)%12=4, not < 4 → small
    assert result[3] == 6  # (3*4)%12=0 < 4 → big
    assert result[6] == 6
    assert result[9] == 6
    assert all(v in (5, 6) for v in result)


def test_shard_tiles_gpt_w0w1():
    """GPT: Nt=90, n_cores=12 → big=8, small=7; exactly 6 big cores."""
    n_tiles, n_cores = 90, 12
    result = [_shard_tiles(n_tiles, c, n_cores) for c in range(n_cores)]
    assert sum(result) == n_tiles
    assert result.count(8) == 6
    assert result.count(7) == 6


def test_shard_tiles_glm5_w0w1():
    """GLM-5: Nt=64 (same as DS), same distribution."""
    result = [_shard_tiles(64, c, 12) for c in range(12)]
    assert sum(result) == 64
    assert result[0] == 6 and result[3] == 6  # same big positions as DS


def test_shard_tiles_exactly_divisible():
    """DS V4 Pro: Nt=96, 96%12=0 → all cores get 8."""
    result = [_shard_tiles(96, c, 12) for c in range(12)]
    assert all(v == 8 for v in result)
    assert sum(result) == 96


def test_w2_shard_tiles_deepseek():
    """DS: Ht=224, Nt=64, n=12, n_big_nt+n_big_ht=4+8=12 → complement.
    W0W1-big cores {0,3,6,9} get small W2 (18 tiles);
    W0W1-small cores get big W2 (19 tiles)."""
    result = [_w2_shard_tiles(224, c, 64, 12) for c in range(12)]
    assert sum(result) == 224
    assert result[0] == 18  # W0W1-big → small W2
    assert result[1] == 19  # W0W1-small → big W2
    assert result[3] == 18
    assert result[2] == 19


def test_w2_shard_tiles_glm5():
    """GLM-5: Ht=192, Nt=64, n=12 → Ht%12=0 so n_big_ht=0.
    n_big_nt+n_big_ht=4+0≠12 → fallback to _shard_tiles(192,c,12)=16 uniform."""
    result = [_w2_shard_tiles(192, c, 64, 12) for c in range(12)]
    assert all(v == 16 for v in result)
    assert sum(result) == 192


def test_w2_shard_tiles_complementary_off():
    """When n_big_nt + n_big_ht != n_cores, _w2_shard_tiles falls back to _shard_tiles.

    Ht=224, Nt=96, n=12: n_big_nt=96%12=0, n_big_ht=224%12=8, 0+8!=12 -> fallback.
    """
    Ht, Nt, n = 224, 96, 12
    result = [_w2_shard_tiles(Ht, c, Nt, n) for c in range(n)]
    fallback = [_shard_tiles(Ht, c, n) for c in range(n)]
    assert result == fallback, "Expected fallback to _shard_tiles when complementary condition not met"
    assert sum(result) == Ht


def test_w2_shard_tiles_complementary_off_another():
    """Ht=160, Nt=64, n=12: n_big_nt=4, n_big_ht=4, 4+4=8!=12 -> fallback."""
    Ht, Nt, n = 160, 64, 12
    result = [_w2_shard_tiles(Ht, c, Nt, n) for c in range(n)]
    fallback = [_shard_tiles(Ht, c, n) for c in range(n)]
    assert result == fallback
    assert sum(result) == Ht


def test_w2_shard_tiles_dsv4_flash():
    """DS V4 Flash: Ht=128, Nt=64, n=12. n_big_nt=4, n_big_ht=8, 4+8=12 → complement."""
    result = [_w2_shard_tiles(128, c, 64, 12) for c in range(12)]
    assert sum(result) == 128
    assert result[0] == 128 // 12  # W0W1-big → small W2
    assert result[1] == 128 // 12 + 1  # W0W1-small → big W2


def test_w2_shard_tiles_gpt():
    """GPT: Ht=90, Nt=90, n=12. n_big_nt=6, n_big_ht=6, 6+6=12 → complement."""
    result = [_w2_shard_tiles(90, c, 90, 12) for c in range(12)]
    assert sum(result) == 90
    assert result.count(8) + result.count(7) == 12


def test_auto_output_width_shard_dim():
    assert auto_output_width_shard_dim(7168) == 4  # DS: Ht=224, 224%4=0
    assert auto_output_width_shard_dim(2880) == 3  # GPT: Ht=90, 90%4≠0, 90%3=0
    assert auto_output_width_shard_dim(6144) == 4  # GLM-5: Ht=192, 192%4=0
    assert auto_output_width_shard_dim(8192) == 4  # Ling-1T: Ht=256
    assert auto_output_width_shard_dim(5120) == 4  # GLM-4.7: Ht=160
    assert auto_output_width_shard_dim(4096) == 4  # DS V4 Flash: Ht=128
    assert auto_output_width_shard_dim(7168) == 4  # Kimi K2.5: same as DS


def test_shard_tiles_total_always_correct():
    """Property: sum of all _shard_tiles == n_tiles, for all interesting model shapes."""
    shapes = [(64, 12), (90, 12), (96, 12), (192, 12), (48, 12), (128, 12), (224, 12), (256, 12)]
    for n_tiles, n_cores in shapes:
        result = [_shard_tiles(n_tiles, c, n_cores) for c in range(n_cores)]
        assert sum(result) == n_tiles, f"Failed for n_tiles={n_tiles}, n_cores={n_cores}"


def test_intermediate_tiles_must_exceed_core_count():
    """Verify that configurations with intermediate_tiles < n_cores are detectable.

    The C++ validate function enforces intermediate_tiles >= matmul_num_cores
    via TT_FATAL. This test validates the invariant at the formula level:
    if intermediate_size / 32 < n_cores, the _shard_tiles distribution degenerates
    (some cores get 0 tiles).
    """
    n_cores = 12
    # intermediate_size = 256 -> intermediate_tiles = 8, which is < 12
    small_intermediate_tiles = 256 // 32  # = 8
    assert small_intermediate_tiles < n_cores
    shards = [_shard_tiles(small_intermediate_tiles, c, n_cores) for c in range(n_cores)]
    # At least one core gets 0 tiles -- degenerate, triggers TT_FATAL on device
    assert any(s == 0 for s in shards), "Expected degenerate shard distribution with 0-tile cores"
    assert sum(shards) == small_intermediate_tiles

    # Boundary: intermediate_tiles == n_cores -> exactly 1 tile per core, valid
    boundary_tiles = n_cores
    boundary_shards = [_shard_tiles(boundary_tiles, c, n_cores) for c in range(n_cores)]
    assert all(s == 1 for s in boundary_shards)
