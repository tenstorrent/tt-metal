# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP as ReferenceExpert
from models.demos.deepseek_v3.tt.experts import Experts as TTExperts
from models.demos.deepseek_v3.utils.config_helpers import even_int_div, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    dequantize_state_dict,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)

TEST_CHECK_ITERS = 100
CI_ACTIVE = os.getenv("CI") == "true"
_CI_SKIP_MARK = pytest.mark.skipif(
    CI_ACTIVE,
    reason="CI runs traced coverage only.",
)
SPARSITY_BLOCK_SIZE = 128


class DeepseekV3MoEExperts(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.experts = nn.ModuleList(
            [
                ReferenceExpert(config, intermediate_size=config.moe_intermediate_size).eval()
                for i in range(config.n_routed_experts)
            ]
        )

    def forward(self, hidden_states):
        outputs = []
        for expert in self.experts:
            outputs.append(expert(hidden_states))

        return torch.cat(outputs, dim=0)


def create_combined_state_dict(module_path: str, model_path: Path, state_dict: dict[str, torch.Tensor]) -> dict:
    """
    Create a combined state_dict from multiple experts state_dicts.
    """
    parts = module_path.split(".")
    base_path = ".".join(parts[:-1])
    s, e = module_path.split(".")[-1].split("-")
    s, e = int(s), int(e)
    out_state_dict = {}
    for i in range(s, e + 1):
        module_path_i = f"{base_path}.{i}"
        state_dict_i = sub_state_dict(state_dict, module_path_i + ".")
        for k, v in state_dict_i.items():
            k_ = f"{base_path.split('.')[-1]}.{i}.{k}"
            out_state_dict[k_] = v

    return out_state_dict


@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 128),
    ]
    + [
        ("prefill", seq_len)
        if seq_len == 128
        else pytest.param(
            "prefill",
            seq_len,
            marks=pytest.mark.skipif(
                CI_ACTIVE,
                reason=(
                    f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
                ),
            ),
        )
        for seq_len in PREFILL_SEQ_LENS
    ],
)
@pytest.mark.parametrize(
    "trace_mode",
    [
        pytest.param(False, marks=_CI_SKIP_MARK, id="eager"),
        pytest.param(True, id="tracing"),
    ],
)
@pytest.mark.parametrize(
    "use_real_weights",
    [
        pytest.param(True, id="real_weights"),
        pytest.param(False, marks=_CI_SKIP_MARK, id="random_weights"),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 10485760},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "module_path",
    ["model.layers.3.mlp.experts.0-255"],
)
def test_forward_pass(
    mode: str,
    seq_len: int,
    trace_mode: bool,
    use_real_weights: bool,
    hf_config: Any,
    cache_path: Path,
    mesh_device: Any,
    module_path: str,
    model_path: Path,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict: dict[str, torch.Tensor],
):
    if trace_mode and mode != "decode":
        pytest.skip("Tracing is only supported for decode mode.")

    batch_size = 1
    num_experts_per_device = even_int_div(hf_config.n_routed_experts, mesh_device.get_num_devices())

    reference_model = DeepseekV3MoEExperts(hf_config).eval()
    torch_input = torch.randn(batch_size, 1, seq_len, hf_config.hidden_size)

    if not use_real_weights:
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(), block_shape=hf_config.quantization_config["weight_block_size"]
        )
    else:
        state_dict = create_combined_state_dict(module_path, model_path, state_dict)
        reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config))

    weight_cache_root = cache_path if use_real_weights else cache_path / "random_weights"
    weight_config = get_test_weight_config(
        TTExperts,
        hf_config,
        (state_dict,),
        weight_cache_root,
        mesh_device,
        force_recalculate_weight_config or not use_real_weights,
    )
    model_config = get_model_config(TTExperts, mode, hf_config, mesh_device)
    model_state = TTExperts.create_state(hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state)

    tt_input = ttnn.from_torch(
        torch_input.repeat(1, run_config["num_experts_per_device"], 1, 1),  # repeat activations per expert
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    def check_outputs(tt_output: ttnn.Tensor) -> None:
        expected_output_memory_config = run_config["output_memory_config"]
        actual_output_memory_config = tt_output.memory_config()
        assert (
            actual_output_memory_config == expected_output_memory_config
        ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

        TARGET_CHUNK_SIZE = 2048
        CHUNK_SIZE_SEQ = ((TARGET_CHUNK_SIZE + SPARSITY_BLOCK_SIZE - 1) // SPARSITY_BLOCK_SIZE) * SPARSITY_BLOCK_SIZE
        num_chunks = (seq_len + CHUNK_SIZE_SEQ - 1) // CHUNK_SIZE_SEQ

        from models.common.utility_functions import comp_pcc

        min_pcc = 0.98
        passed = True

        for chunk_idx in range(num_chunks):
            start_seq = chunk_idx * CHUNK_SIZE_SEQ
            end_seq = min(start_seq + CHUNK_SIZE_SEQ, seq_len)
            chunk_seq_len = end_seq - start_seq

            chunk_input = torch_input[:, :, start_seq:end_seq, :]
            chunk_ref_output = reference_model(chunk_input)

            tt_output_chunk = ttnn.slice(
                tt_output,
                [0, 0, start_seq, 0],
                [1, num_experts_per_device, end_seq, hf_config.hidden_size],
            )

            tt_output_chunk_torch = ttnn.to_torch(
                tt_output_chunk,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
            )

            ttnn.deallocate(tt_output_chunk)

            tt_output_chunk_torch = tt_output_chunk_torch.reshape(1, -1, chunk_seq_len, hf_config.hidden_size)
            tt_output_chunk_torch = tt_output_chunk_torch[0].unsqueeze(1)

            if chunk_ref_output.shape != tt_output_chunk_torch.shape:
                chunk_ref_output = chunk_ref_output.unsqueeze(0)

            chunk_passed, chunk_pcc = comp_pcc(tt_output_chunk_torch, chunk_ref_output, pcc=0.98)

            min_pcc = min(min_pcc, chunk_pcc)
            if not chunk_passed:
                passed = False

            del chunk_ref_output
            del tt_output_chunk_torch
            del chunk_input

        assert passed, f"PCC check failed! Min PCC: {min_pcc:.6f} < 0.98"

    if trace_mode:
        # Iteration 0: eager compile run (not traced)
        tt_output = run_module_forward(TTExperts, mode, tt_input, run_config)
        ttnn.synchronize_device(mesh_device)
        check_outputs(tt_output)
        ttnn.deallocate(tt_output)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = run_module_forward(TTExperts, mode, tt_input, run_config)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        for _ in range(TEST_CHECK_ITERS - 1):
            ttnn.execute_trace(mesh_device, trace_id, blocking=True)
        ttnn.synchronize_device(mesh_device)

        check_outputs(trace_output)
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(trace_output)
    else:
        for iter_idx in range(TEST_CHECK_ITERS):
            tt_output = run_module_forward(TTExperts, mode, tt_input, run_config)
            ttnn.synchronize_device(mesh_device)
            if iter_idx in (0, TEST_CHECK_ITERS - 1):
                check_outputs(tt_output)
            ttnn.deallocate(tt_output)

    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])
