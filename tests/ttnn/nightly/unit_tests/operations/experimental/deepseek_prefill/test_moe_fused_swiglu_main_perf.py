# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi K2.6 / GLM 5.1 / Kimi K3 SiTU 5K profiler sweep for the fused routed-expert block.

Run with ``scripts/run_safe_pytest.sh --profile``.  The profiler CSV preserves
one row per dispatch, ordered by the ``COUNTS``/``REPS`` loops below.
"""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_helpers import weight_memory_configs

USE_PYTHON_DESCRIPTOR = os.environ.get("MOE_FUSED_SWIGLU_PYTHON_DESCRIPTOR", "0") == "1"
TRANSPOSE_GRID = os.environ.get("MOE_FUSED_SWIGLU_TRANSPOSE_GRID", "0") == "1"
if USE_PYTHON_DESCRIPTOR:
    from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import (
        create_program_descriptor,
        make_mailbox,
        weight_memory_configs as descriptor_weight_memory_configs,
    )

TILE = 32
EXPERTS_PER_TOKEN = 8
SEQ_LEN_PER_CHIP = 640
CAPACITY = EXPERTS_PER_TOKEN * SEQ_LEN_PER_CHIP
GRID_OVERRIDE = os.environ.get("MOE_FUSED_SWIGLU_GRID")
if GRID_OVERRIDE is None:
    GRID_CASES = (
        (ttnn.CoreCoord(11, 8), "8x11"),
        (ttnn.CoreCoord(12, 8), "8x12"),
    )
else:
    grid_x, grid_y = (int(value) for value in GRID_OVERRIDE.split("x"))
    GRID_CASES = ((ttnn.CoreCoord(grid_x, grid_y), f"{grid_y}x{grid_x}"),)
DISPATCH_CORE_AXIS = getattr(ttnn.DispatchCoreAxis, os.environ.get("MOE_FUSED_SWIGLU_DISPATCH_AXIS", "ROW").upper())
DEVICE_PARAMS = {"dispatch_core_axis": DISPATCH_CORE_AXIS}
if DISPATCH_CORE_AXIS == ttnn.DispatchCoreAxis.ROW:
    DEVICE_PARAMS.update(
        fabric_config=ttnn.FabricConfig.FABRIC_1D,
        fabric_tensix_config=ttnn.FabricTensixConfig.MUX,
    )
COUNTS = tuple(
    int(value) for value in os.environ.get("MOE_FUSED_SWIGLU_COUNTS", "0,64,128,256,512,1024,2048,4096,5120").split(",")
)
REPS = int(os.environ.get("MOE_FUSED_SWIGLU_PERF_REPS", "9"))
# Kimi K2.6 can route all 8 tokens from each of its 640 tokens on this chip to
# one expert, so a 5K dispatch group requires a 5120-row expert region.
GLOBAL_EXPERT_ID = 137
LOCAL_EXPERT_ID = 3


class KimiK3SituConfig:
    """Requested Kimi K3 routed-expert shape; kept local to avoid coupling the kernel sweep to a model package."""

    EMB_SIZE = 3584
    MOE_INTERMEDIATE_SIZE = 3072
    NUM_ROUTED_EXPERTS = 384


class KimiK3SiluControlConfig(KimiK3SituConfig):
    """Same geometry as Kimi K3, retaining SiLU to isolate the SiTU SFPU cost."""


MODEL_CASES = (
    (KimiK26Config, "kimi-k26"),
    (GLM51Config, "glm-51"),
    (KimiK3SiluControlConfig, "kimi-k3-silu-control"),
    (KimiK3SituConfig, "kimi-k3-situ"),
)
INPUT_CASES = (
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "row-major-bf16"),
    (ttnn.bfloat8_b, ttnn.TILE_LAYOUT, "tile-bf8"),
)
WEIGHT_LAYOUT_CASES = (
    (ttnn.TensorMemoryLayout.INTERLEAVED, "interleaved"),
    (ttnn.TensorMemoryLayout.ND_SHARDED, "nd-sharded"),
)
if os.environ.get("MOE_FUSED_SWIGLU_TUNING_ONLY", "0") == "1":
    default_tuning_model = (
        "k3-situ" if os.environ.get("MOE_FUSED_SWIGLU_TUNING_ACTIVATION", "silu") == "situ" else "k3-silu"
    )
    tuning_models = {"kimi": 0, "glm": 1, "k3-silu": 2, "k3-situ": 3}
    tuning_model_name = os.environ.get("MOE_FUSED_SWIGLU_TUNING_MODEL", default_tuning_model)
    if tuning_model_name not in tuning_models:
        raise ValueError(f"unknown MOE_FUSED_SWIGLU_TUNING_MODEL={tuning_model_name!r}")
    tuning_model = tuning_models[tuning_model_name]
    MODEL_CASES = MODEL_CASES[tuning_model : tuning_model + 1]
    tuning_inputs = {"rm-bf16": 0, "tile-bfp8": 1}
    tuning_input_name = os.environ.get("MOE_FUSED_SWIGLU_TUNING_INPUT", "rm-bf16")
    if tuning_input_name not in tuning_inputs:
        raise ValueError(f"unknown MOE_FUSED_SWIGLU_TUNING_INPUT={tuning_input_name!r}")
    tuning_input = tuning_inputs[tuning_input_name]
    INPUT_CASES = INPUT_CASES[tuning_input : tuning_input + 1]
    WEIGHT_LAYOUT_CASES = WEIGHT_LAYOUT_CASES[1:]


def to_device(tensor, dtype, layout, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="moe_fused_swiglu is Blackhole-only")
@pytest.mark.parametrize(
    "model_config",
    [case[0] for case in MODEL_CASES],
    ids=[case[1] for case in MODEL_CASES],
)
@pytest.mark.parametrize(
    "input_dtype,input_layout",
    [(case[0], case[1]) for case in INPUT_CASES],
    ids=[case[2] for case in INPUT_CASES],
)
@pytest.mark.parametrize(
    "weight_memory_layout",
    [case[0] for case in WEIGHT_LAYOUT_CASES],
    ids=[case[1] for case in WEIGHT_LAYOUT_CASES],
)
@pytest.mark.parametrize("core_grid", [case[0] for case in GRID_CASES], ids=[case[1] for case in GRID_CASES])
def test_5k_cpp_binding_perf_matrix(
    mesh_device, model_config, input_dtype, input_layout, weight_memory_layout, core_grid
):
    device = mesh_device
    available_grid = device.compute_with_storage_grid_size()
    if core_grid.x > available_grid.x or core_grid.y > available_grid.y:
        pytest.skip(
            f"requested {core_grid.y}x{core_grid.x} grid exceeds available "
            f"{available_grid.y}x{available_grid.x} grid for {DISPATCH_CORE_AXIS} dispatch"
        )
    torch.manual_seed(42)
    k = model_config.EMB_SIZE
    n = model_config.MOE_INTERMEDIATE_SIZE
    num_routed_experts = model_config.NUM_ROUTED_EXPERTS
    x = to_device(
        torch.randn((1, 1, CAPACITY, k), dtype=torch.bfloat16),
        input_dtype,
        input_layout,
        device,
    )
    if weight_memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED:
        if USE_PYTHON_DESCRIPTOR:
            gate_up_memory_config, down_memory_config = descriptor_weight_memory_configs(
                device, k, n, core_grid=core_grid, transpose_grid=TRANSPOSE_GRID
            )
        else:
            gate_up_memory_config, down_memory_config = weight_memory_configs(device, k, n, core_grid=core_grid)
    else:
        gate_up_memory_config = down_memory_config = ttnn.DRAM_MEMORY_CONFIG
    weights = [
        to_device(host, ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device, memory_config)
        for host, memory_config in zip(
            (
                torch.randn((k, n), dtype=torch.bfloat16),
                torch.randn((k, n), dtype=torch.bfloat16),
                torch.randn((n, k), dtype=torch.bfloat16),
            ),
            (gate_up_memory_config, gate_up_memory_config, down_memory_config),
        )
    ]
    ids = torch.tensor([(11 + 37 * local) % num_routed_experts for local in range(8)], dtype=torch.int32)
    ids[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    ids = to_device(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)
    count_tensors = {}
    for count in COUNTS:
        host = torch.zeros(num_routed_experts, dtype=torch.int32)
        host[GLOBAL_EXPERT_ID] = count
        count_tensors[count] = to_device(host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    config = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    op = ttnn.experimental.deepseek_prefill.moe_fused_swiglu
    activation = (
        ttnn.RoutedExpertActivation.SituGlu if model_config is KimiK3SituConfig else ttnn.RoutedExpertActivation.Silu
    )
    if USE_PYTHON_DESCRIPTOR:
        output = ttnn.empty(
            x.shape,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mailbox = make_mailbox(device, core_grid.x * core_grid.y)
        descriptor_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        )
        descriptors = {
            count: create_program_descriptor(
                x,
                *weights,
                count_tensor,
                ids,
                output,
                mailbox,
                local_expert_id=LOCAL_EXPERT_ID,
                input_m_tiles=CAPACITY // TILE,
                compute_kernel_config=descriptor_config,
                core_grid=core_grid,
                transpose_grid=TRANSPOSE_GRID,
                situ_glu=activation == ttnn.RoutedExpertActivation.SituGlu,
            )
            for count, count_tensor in count_tensors.items()
        }

        def run_op(count):
            ttnn.generic_op([x, *weights, count_tensors[count], ids, output, mailbox], descriptors[count])

    else:

        def run_op(count):
            op(
                x,
                *weights,
                count_tensors[count],
                ids,
                LOCAL_EXPERT_ID,
                input_m_tiles=CAPACITY // TILE,
                compute_kernel_config=config,
                core_grid=core_grid,
                activation=activation,
            )

    # Compile/cache warmup is deliberately excluded from the reported matrix.
    run_op(512 if 512 in count_tensors else COUNTS[0])
    ttnn.ReadDeviceProfiler(device)

    for count in COUNTS:
        for _ in range(REPS):
            run_op(count)
            ttnn.ReadDeviceProfiler(device)
