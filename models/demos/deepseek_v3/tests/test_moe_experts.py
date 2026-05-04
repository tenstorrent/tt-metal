# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP as ReferenceExpert
from models.demos.deepseek_v3.tests.pytest_utils import DEFAULT_PREFILL_SEQ_LEN
from models.demos.deepseek_v3.tt.experts import Experts as TTExperts
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, even_int_div, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config, run_module_forward


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
    container_name = base_path.split(".")[-1]
    s, e = module_path.split(".")[-1].split("-")
    s, e = int(s), int(e)
    stacked_prefix = ".".join(parts[:-2] + ["experts_stacked"]) + "."
    stacked_state_dict = sub_state_dict(state_dict, stacked_prefix)
    stacked_projection_names = ("gate_proj.weight", "down_proj.weight", "up_proj.weight")
    present_stacked_projection_names = tuple(name for name in stacked_projection_names if name in stacked_state_dict)

    out_state_dict = {}
    if present_stacked_projection_names:
        if present_stacked_projection_names != stacked_projection_names:
            missing_projection_names = sorted(set(stacked_projection_names) - set(present_stacked_projection_names))
            raise ValueError(
                "Checkpoint mixes stacked and legacy expert weights in reference model setup: "
                f"missing stacked projections {missing_projection_names} under '{stacked_prefix}'"
            )

        for projection_name in stacked_projection_names:
            stacked_weight = stacked_state_dict[projection_name]
            if stacked_weight.ndim != 3:
                raise ValueError(
                    f"Expected stacked expert weight '{stacked_prefix}{projection_name}' to have rank 3, "
                    f"got {stacked_weight.ndim}"
                )
            if stacked_weight.shape[0] <= e:
                raise ValueError(
                    f"Expected stacked expert weight '{stacked_prefix}{projection_name}' to contain expert {e}, "
                    f"got {stacked_weight.shape[0]} experts"
                )
            for expert_idx in range(s, e + 1):
                out_state_dict[f"{container_name}.{expert_idx}.{projection_name}"] = stacked_weight[expert_idx]
        return out_state_dict

    for i in range(s, e + 1):
        module_path_i = f"{base_path}.{i}"
        state_dict_i = sub_state_dict(state_dict, module_path_i + ".")
        for k, v in state_dict_i.items():
            k_ = f"{container_name}.{i}.{k}"
            out_state_dict[k_] = v

    return out_state_dict


_max_seq_len_env = os.getenv("DEEPSEEK_MAX_SEQ_LEN_OVERRIDE")
_prefill_seq_len = int(_max_seq_len_env) if _max_seq_len_env is not None else DEFAULT_PREFILL_SEQ_LEN


@pytest.mark.parametrize(
    "mode, batch_size_per_row, seq_len",
    [
        ("decode", USERS_PER_ROW, 1),
        ("prefill", 1, _prefill_seq_len),
    ],
)
@pytest.mark.parametrize(
    "weight_type",
    ["random", "real"],
)
@pytest.mark.parametrize(
    "module_path",
    ["model.layers.3.mlp.experts.0-255"],
)
def test_forward_pass(
    mode: str,
    batch_size_per_row: int,
    seq_len: int,
    hf_config: Any,
    cache_path: Path,
    mesh_device: Any,
    weight_type: str,
    module_path: str,
    model_path: Path,
    force_recalculate_weight_config,
    set_deterministic_env,
    state_dict: dict[str, torch.Tensor],
):
    num_tokens = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len
    num_experts_per_device = even_int_div(hf_config.n_routed_experts, mesh_device.get_num_devices())

    reference_model = DeepseekV3MoEExperts(hf_config).eval().to(torch.bfloat16)
    torch_input = torch.randn(1, 1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    if weight_type == "random":
        tt_state_dict = reference_model.state_dict()
    else:
        assert weight_type == "real"
        reference_state_dict = create_combined_state_dict(module_path, model_path, state_dict)
        tt_state_dict = sub_state_dict(state_dict, ".".join(module_path.split(".")[:-2]) + ".")
        reference_model.load_state_dict(reference_state_dict)

    weight_config = get_test_weight_config(
        TTExperts,
        hf_config,
        (tt_state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
        test_name="test_moe_experts",
        real_weights=weight_type == "real",
        layer_id=module_path,
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
    tt_output = run_module_forward(TTExperts, mode, tt_input, run_config)
    expected_output_memory_config = run_config["output_memory_config"]

    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    TARGET_CHUNK_SIZE = 2048
    num_chunks = (seq_len + TARGET_CHUNK_SIZE - 1) // TARGET_CHUNK_SIZE

    from models.common.utility_functions import comp_pcc

    required_pcc = 0.975  # slightly lower after moving to quantize-then-transpose
    min_pcc = float("inf")
    passed = True

    for chunk_idx in range(num_chunks):
        start_seq = chunk_idx * TARGET_CHUNK_SIZE
        end_seq = min(start_seq + TARGET_CHUNK_SIZE, seq_len)
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

        chunk_passed, chunk_pcc = comp_pcc(tt_output_chunk_torch, chunk_ref_output, pcc=required_pcc)

        min_pcc = min(min_pcc, chunk_pcc)
        if not chunk_passed:
            passed = False

        del chunk_ref_output
        del tt_output_chunk_torch
        del chunk_input

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    assert passed, f"PCC check failed! Min PCC: {min_pcc:.6f} < {required_pcc}"


def test_convert_weights_rejects_partial_stacked_expert_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class _FakeMeshDevice:
        shape = (1, 1)

        def get_num_devices(self) -> int:
            return 1

    class _ViewableStateDict(dict):
        def view_with_prefix(self, prefix: str, num_layers: int | None = None):
            return {k[len(prefix) :]: v for k, v in self.items() if k.startswith(prefix)}

    state_dict = _ViewableStateDict(
        {
            "experts_stacked.gate_proj.weight": torch.arange(12, dtype=torch.bfloat16).reshape(2, 2, 3),
            "experts.0.gate_proj.weight": torch.ones((2, 3), dtype=torch.bfloat16),
            "experts.1.gate_proj.weight": torch.full((2, 3), 2.0, dtype=torch.bfloat16),
            "experts.0.down_proj.weight": torch.ones((2, 3), dtype=torch.bfloat16),
            "experts.1.down_proj.weight": torch.full((2, 3), 2.0, dtype=torch.bfloat16),
            "experts.0.up_proj.weight": torch.ones((2, 3), dtype=torch.bfloat16),
            "experts.1.up_proj.weight": torch.full((2, 3), 2.0, dtype=torch.bfloat16),
        }
    )

    monkeypatch.setattr(
        "models.demos.deepseek_v3.tt.experts.get_dequantized_tensor",
        lambda state_dict, key, dtype=None: state_dict[key],
    )
    monkeypatch.setattr(
        "models.demos.deepseek_v3.tt.experts.shard_and_save",
        lambda path, tensor, *args, **kwargs: tensor,
    )
    monkeypatch.setattr(TTExperts, "_warned_legacy_expert_checkpoint", False)

    with pytest.raises(ValueError, match="mixes stacked and legacy expert weights"):
        TTExperts.convert_weights(
            SimpleNamespace(n_routed_experts=2),
            (state_dict,),
            tmp_path,
            _FakeMeshDevice(),
        )


def test_create_combined_state_dict_uses_stacked_expert_weights():
    class _ViewableStateDict(dict):
        def view_with_prefix(self, prefix: str, num_layers: int | None = None):
            return {k[len(prefix) :]: v for k, v in self.items() if k.startswith(prefix)}

    state_dict = _ViewableStateDict(
        {
            "model.layers.3.mlp.experts_stacked.gate_proj.weight": torch.arange(24, dtype=torch.bfloat16).reshape(
                3, 2, 4
            ),
            "model.layers.3.mlp.experts_stacked.down_proj.weight": torch.arange(24, 48, dtype=torch.bfloat16).reshape(
                3, 2, 4
            ),
            "model.layers.3.mlp.experts_stacked.up_proj.weight": torch.arange(48, 72, dtype=torch.bfloat16).reshape(
                3, 2, 4
            ),
        }
    )

    combined_state_dict = create_combined_state_dict("model.layers.3.mlp.experts.1-2", Path("."), state_dict)

    assert set(combined_state_dict) == {
        "experts.1.gate_proj.weight",
        "experts.1.down_proj.weight",
        "experts.1.up_proj.weight",
        "experts.2.gate_proj.weight",
        "experts.2.down_proj.weight",
        "experts.2.up_proj.weight",
    }
    assert torch.equal(
        combined_state_dict["experts.1.gate_proj.weight"],
        state_dict["model.layers.3.mlp.experts_stacked.gate_proj.weight"][1],
    )
    assert torch.equal(
        combined_state_dict["experts.2.down_proj.weight"],
        state_dict["model.layers.3.mlp.experts_stacked.down_proj.weight"][2],
    )
    assert torch.equal(
        combined_state_dict["experts.2.up_proj.weight"],
        state_dict["model.layers.3.mlp.experts_stacked.up_proj.weight"][2],
    )


if __name__ == "__main__":
    pytest.main([__file__])
