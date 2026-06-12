# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import math
import os
import types

import pytest
import torch
from loguru import logger

import ttnn
from models.common.modules.moe.tt_moe_gate import TTMoEGate
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.deepseek_v3.tests.pytest_utils import DEFAULT_PREFILL_SEQ_LEN
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, get_dequantized_tensor, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_test_weight_config,
    load_reference_io,
    load_reference_io_tensors_for_module,
    run_module_forward,
)
from tests.ttnn.utils_for_testing import comp_pcc


def load_real_moe_input(mode: str, module_path: str, num_tokens: int) -> torch.Tensor:
    if mode == "prefill":
        torch_input, _ = load_reference_io_tensors_for_module(mode, module_path, num_tokens, 1)
        return torch_input.squeeze(0).to(torch.bfloat16)

    reference_io = load_reference_io(mode, module_path)
    assert all(len(logs) <= 1 for logs in reference_io), f"Expected a non-range module, got {module_path}"
    assert all(len(logs) > 0 for logs in reference_io), f"Some logs for module {module_path} {mode} were not generated."

    io_module_paths, torch_args, _, _ = zip(*[logs[0] for logs in reference_io])
    (torch_inputs,) = zip(*torch_args)
    assert set(io_module_paths) == {module_path}

    torch_input = torch.concat(torch_inputs, dim=1).unsqueeze(0)

    if torch_input.shape[2] < num_tokens:
        repeats = math.ceil(num_tokens / torch_input.shape[2])
        torch_input = torch_input.repeat(1, 1, repeats, 1)

    return torch_input[:, :, :num_tokens, :].squeeze(0).to(torch.bfloat16)


def generate_reference_io(
    mode: str,
    num_tokens: int,
    reference_model: ReferenceMoEGate,
    checkpoint_state_dict: dict[str, torch.Tensor],
    module_path: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    moe_state_dict = {
        name[5:]: tensor
        for name, tensor in sub_state_dict(checkpoint_state_dict, module_path + ".").items()
        if name.startswith("gate.")
    }
    if not moe_state_dict:
        pytest.skip(f"Checkpoint does not contain routed MoE weights under '{module_path}'")

    state_dict_out = moe_state_dict
    reference_model.load_state_dict(state_dict_out)
    torch_input = load_real_moe_input(mode, module_path, num_tokens)

    reference_model.eval()
    reference_model.to(torch.bfloat16)
    with torch.no_grad():
        reference_topk_indices, reference_topk_weights = reference_model(torch_input)

    return state_dict_out, torch_input, reference_topk_indices, reference_topk_weights


_max_seq_len_env = os.getenv("DEEPSEEK_MAX_SEQ_LEN_OVERRIDE")
_prefill_seq_len = int(_max_seq_len_env) if _max_seq_len_env is not None else DEFAULT_PREFILL_SEQ_LEN


@pytest.mark.parametrize(
    "mode,batch_size_per_row,seq_len",
    [
        ("decode", USERS_PER_ROW, 1),
        ("prefill", 1, _prefill_seq_len),
    ],
)
def test_forward_pass(
    mode,
    batch_size_per_row,
    seq_len,
    hf_config,
    request,
    cache_path,
    mesh_device,
    set_deterministic_env,
    force_recalculate_weight_config,
):
    """Test forward pass against reference model."""

    module_path = "model.layers.3.mlp"
    reference_model = ReferenceMoEGate(hf_config)
    checkpoint_state_dict = request.getfixturevalue("state_dict")
    num_tokens = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len
    state_dict, torch_input, reference_topk_indices, reference_topk_weights = generate_reference_io(
        mode=mode,
        num_tokens=num_tokens,
        reference_model=reference_model,
        checkpoint_state_dict=checkpoint_state_dict,
        module_path=module_path,
    )

    weight_config = get_test_weight_config(
        MoEGate,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=force_recalculate_weight_config,
        test_name="test_moe_gate",
        real_weights=True,
        layer_id=module_path,
    )

    # Generate appropriate config using utility function
    model_config = get_model_config(MoEGate, mode, hf_config, mesh_device)

    # Create a new model state
    model_state = MoEGate.create_shared_state(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass using utility function
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_topk_weights, tt_topk_indices = run_module_forward(MoEGate, mode, tt_input, run_config)

    # Verify output memory config matches expected
    expected_output_memory_config = run_config["output_memory_config"]
    actual_topk_weights_memory_config = tt_topk_weights.memory_config()
    assert (
        actual_topk_weights_memory_config == expected_output_memory_config
    ), f"TopK experts weights memory config mismatch: expected {expected_output_memory_config}, got {actual_topk_weights_memory_config}"

    actual_topk_indices_memory_config = tt_topk_indices.memory_config()
    assert (
        actual_topk_indices_memory_config == expected_output_memory_config
    ), f"TopK experts indices memory config mismatch: expected {expected_output_memory_config}, got {actual_topk_indices_memory_config}"

    # Convert output back to torch
    tt_topk_weights_torch = ttnn.to_torch(
        tt_topk_weights,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)
    tt_topk_indices_torch = ttnn.to_torch(
        tt_topk_indices,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
    )[0].squeeze(0)

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_topk_indices)

    # Compare outputs
    logger.info(f"Mode: {mode}, Seq len: {seq_len}")

    # sort reference
    ref_weights = reference_topk_weights.to(torch.bfloat16)
    ref_indices = reference_topk_indices.to(torch.int32)

    ref_sorted_weights, ref_sort_idx = torch.sort(ref_weights, dim=-1, descending=True, stable=True)
    ref_sorted_indices = torch.gather(ref_indices, -1, ref_sort_idx)

    # sort tt
    tt_weights = tt_topk_weights_torch.to(torch.bfloat16)
    tt_indices = tt_topk_indices_torch.to(torch.int32)

    tt_sorted_weights, tt_sort_idx = torch.sort(tt_weights, dim=-1, descending=True, stable=True)
    tt_sorted_indices = torch.gather(tt_indices, -1, tt_sort_idx)

    # compare
    topk_weights_pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_sorted_weights, tt_sorted_weights, topk_weights_pcc_required)

    topk_indices_accuracy_required = 0.91 if mode == "decode" else 0.84
    accuracy = tt_sorted_indices.eq(ref_sorted_indices).float().mean()

    logger.info(f"TopK experts weights PCC: {pcc_message}")
    logger.info(f"TopK experts indices accuracy: {accuracy}")
    assert (
        passing
    ), f"TopK experts weights output does not meet PCC requirement {topk_weights_pcc_required}: {pcc_message}"

    assert accuracy >= topk_indices_accuracy_required, f"TopK experts indices output does not match: {accuracy}"


def test_tt_moe_gate_real_weights(
    hf_config,
    request,
    mesh_device,
    set_deterministic_env,
):
    """Real-weights test for the COMMON ``TTMoEGate`` (the n_group=8 grouped path), modeled on
    ``test_forward_pass`` above but routing through ``models.common.modules.moe.tt_moe_gate.TTMoEGate``
    instead of the deepseek-specific ``MoEGate`` module.

    Why it lives here (not in the standalone ``test_tt_moe_gate.py``): the grouped 8→4 group selection
    needs REAL router weights to be meaningful. With random weights the 256 logits are ~iid, the 16
    groups are near-tied, and bf16 rounding flips which groups win → device vs golden diverge (PCC ~0.87,
    overlap ~0.71). Real DeepSeek router weights separate the groups, so the device selection matches the
    HF reference — letting us hold the same strict bar as ``test_forward_pass`` (weights PCC 0.99, index
    accuracy 0.91). ``test_tt_moe_gate.py`` skips deepseek_v3 and points here.

    Loads the real gate weight + score-correction bias from the checkpoint and compares against the HF
    ``ReferenceMoEGate`` (bf16), reusing ``generate_reference_io``.
    """
    mode = "decode"
    module_path = "model.layers.3.mlp"
    num_tokens = USERS_PER_ROW  # TTMoEGate replicates (no batch sharding), so one row's worth is enough.

    reference_model = ReferenceMoEGate(hf_config)
    checkpoint_state_dict = request.getfixturevalue("state_dict")
    gate_state_dict, torch_input, reference_topk_indices, reference_topk_weights = generate_reference_io(
        mode=mode,
        num_tokens=num_tokens,
        reference_model=reference_model,
        checkpoint_state_dict=checkpoint_state_dict,
        module_path=module_path,
    )

    # Real router weights → the common TTMoEGate. HF stores the gate as [n_experts, hidden]; TTMoEGate
    # wants [hidden, n_experts]. The score-correction bias is mean-centered (as in moe_gate.py's
    # convert_weights) so `sigmoid + bias` stays small enough for the op's bf16 selection precision —
    # a constant shift changes neither the group ranking nor the (unbiased) output weights.
    gate_weight = get_dequantized_tensor(gate_state_dict, "weight")  # [n_experts, hidden]
    bias = get_dequantized_tensor(gate_state_dict, "e_score_correction_bias", dtype=torch.float32)  # [n_experts]
    bias = bias - bias.mean()

    # Config-driven entry point (mirrors TTMoEDecode). deepseek's deep (7168) + tie-sensitive gate needs
    # HiFi2 + fp32 accumulation, else the default matmul flips near-tied experts → index accuracy drops
    # (~0.83 vs ~0.90). Matches models/common/modules/moe/configs/deepseek_v3.yaml. The real
    # e_score_correction_bias is passed as torch_gate_bias below (so the config flag isn't needed here).
    gate_config = TTMoEGateConfig(
        num_routed_experts=hf_config.n_routed_experts,
        select_experts_k=hf_config.num_experts_per_tok,
        hidden_size=hf_config.hidden_size,
        batch_per_device=num_tokens,
        n_group=hf_config.n_group,
        score_func="sigmoid",
        routed_scaling_factor=hf_config.routed_scaling_factor,
        gate_matmul_compute={
            "math_fidelity": "HiFi2",
            "math_approx_mode": True,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": True,
        },
        # tuned 2D-mcast program config for the gate matmul (batch×7168 @ 7168×256) — same values as
        # deepseek's own MoEGate decode gate_proj and deepseek_v3.yaml. compute_with_storage_grid_size is
        # device-derived (auto-filled by TTMoEGate), so it is omitted here.
        gate_matmul_program_config={
            "in0_block_w": 32,
            "out_subblock_h": 1,
            "out_subblock_w": 2,
            "out_block_h": 1,
            "out_block_w": 2,
            "per_core_M": 1,
            "per_core_N": 2,
            "transpose_mcast": False,
            "fused_activation": None,
        },
    )
    gate = TTMoEGate(
        mesh_device,
        gate_config,
        torch_gate_weight=gate_weight.t().contiguous(),
        torch_gate_bias=bias,
    )

    tt_x = ttnn.from_torch(
        torch_input.reshape(1, 1, num_tokens, hf_config.hidden_size),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_topk_weights, tt_topk_indices = gate.forward(tt_x)

    # TTMoEGate replicates across the mesh (bare from_torch, no shard mapper), so every device holds the
    # full [1,1,num_tokens,k] result — read back one replica.
    dev_weights = ttnn.to_torch(tt_topk_weights, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0].reshape(
        num_tokens, -1
    )
    dev_indices = ttnn.to_torch(tt_topk_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0].reshape(
        num_tokens, -1
    )

    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_topk_indices)

    # Same comparison as test_forward_pass: sort weights desc, gather indices to that order, PCC the
    # sorted weights + position-wise index accuracy.
    ref_sorted_weights, ref_sort_idx = torch.sort(
        reference_topk_weights.to(torch.bfloat16), dim=-1, descending=True, stable=True
    )
    ref_sorted_indices = torch.gather(reference_topk_indices.to(torch.int32), -1, ref_sort_idx)

    tt_sorted_weights, tt_sort_idx = torch.sort(dev_weights.to(torch.bfloat16), dim=-1, descending=True, stable=True)
    tt_sorted_indices = torch.gather(dev_indices.to(torch.int32), -1, tt_sort_idx)

    topk_weights_pcc_required = 0.99
    passing, pcc_message = comp_pcc(ref_sorted_weights, tt_sorted_weights, topk_weights_pcc_required)
    topk_indices_accuracy_required = 0.90
    accuracy = tt_sorted_indices.eq(ref_sorted_indices).float().mean()

    logger.info(f"TTMoEGate real-weights — TopK experts weights PCC: {pcc_message}")
    logger.info(f"TTMoEGate real-weights — TopK experts indices accuracy: {accuracy}")
    assert passing, f"TTMoEGate TopK weights PCC < {topk_weights_pcc_required}: {pcc_message}"
    assert (
        accuracy >= topk_indices_accuracy_required
    ), f"TTMoEGate TopK indices accuracy {accuracy} < {topk_indices_accuracy_required}"


def test_linear_fallback_op_uses_hf_oriented_gate_weights(monkeypatch: pytest.MonkeyPatch):
    class _FakeTTTensor:
        def __init__(self, payload: torch.Tensor):
            self.payload = payload
            self.shape = tuple(payload.shape)

    torch_input_payload = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16)
    torch_weight_payload = torch.tensor([[[[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]]], dtype=torch.bfloat16)
    captured: dict[str, torch.Tensor] = {}

    monkeypatch.setattr(ttnn, "ConcatMesh2dToTensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(ttnn, "ShardTensor2dMesh", lambda *args, **kwargs: None)
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor, **kwargs: tensor.payload)

    def fake_from_torch(tensor, **kwargs):
        captured["output"] = tensor
        return _FakeTTTensor(tensor)

    monkeypatch.setattr(ttnn, "from_torch", fake_from_torch)

    mesh_device = types.SimpleNamespace(shape=(1, 1))
    output = MoEGate.linear_fallback_op(
        _FakeTTTensor(torch_input_payload),
        _FakeTTTensor(torch_weight_payload),
        mesh_device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        transpose_b=True,
    )

    expected = torch.nn.functional.linear(torch_input_payload[0], torch_weight_payload[0, 0]).unsqueeze(0).unsqueeze(0)
    assert torch.equal(captured["output"], expected)
    assert output.shape == tuple(expected.shape)


if __name__ == "__main__":
    pytest.main([__file__])
