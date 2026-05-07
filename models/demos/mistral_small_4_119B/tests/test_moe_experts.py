# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.mistral_small_4_119B.tt.moe.experts import TtMistral4Experts
from tests.ttnn.utils_for_testing import comp_pcc

PCC_REQUIRED_EXPERTS = 0.975  # Align with DeepSeek experts test tolerance after quantize/transpose path.
TARGET_CHUNK_SIZE = 2048


def _tiny_mistral4_config():
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=4096,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
    )


@pytest.mark.parametrize(
    "mode,batch_size_per_row,seq_len",
    [
        ("decode", 8, 1),
        ("prefill", 1, 64),
    ],
)
def test_forward_pass(mode, batch_size_per_row, seq_len, mesh_device):
    """Test Mistral4 experts forward against HF experts output."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4NaiveMoe

    hf_config = _tiny_mistral4_config()
    num_tokens = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len

    reference_model = Mistral4NaiveMoe(hf_config).eval().to(torch.float32)
    state_dict = reference_model.state_dict()
    weight_config = TtMistral4Experts.convert_weights(
        hf_config,
        (state_dict,),
        Path("/tmp/mistral_small4_moe_experts_weight_cache"),
        mesh_device,
    )
    model_config = (
        TtMistral4Experts.decode_model_config(hf_config, mesh_device)
        if mode == "decode"
        else TtMistral4Experts.prefill_model_config(hf_config, mesh_device)
    )
    model_state = TtMistral4Experts.create_state(hf_config, mesh_device=mesh_device, ccl=None)

    run_config = dict(model_config)
    run_config.update(weight_config)
    run_config.update(model_state)
    run_config["mesh_device"] = mesh_device

    torch_input = torch.randn(1, 1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)
    topk_indices = torch.randint(
        0,
        hf_config.n_routed_experts,
        (num_tokens, hf_config.num_experts_per_tok),
        dtype=torch.int64,
    )
    topk_weights = torch.rand((num_tokens, hf_config.num_experts_per_tok), dtype=torch.float32)
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

    with torch.no_grad():
        ref_out = reference_model(
            torch_input.reshape(-1, hf_config.hidden_size).to(torch.float32), topk_indices, topk_weights
        )

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_topk_indices = ttnn.from_torch(
        topk_indices.view(1, 1, num_tokens, -1).to(torch.int32),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_topk_weights = ttnn.from_torch(
        topk_weights.view(1, 1, num_tokens, -1).to(torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = (
        TtMistral4Experts.forward_decode(tt_input, tt_topk_indices, tt_topk_weights, run_config)
        if mode == "decode"
        else TtMistral4Experts.forward_prefill(tt_input, tt_topk_indices, tt_topk_weights, run_config)
    )

    assert tt_output.memory_config() == run_config["output_memory_config"]

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )[0, 0]

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_topk_indices)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_output)

    ref_out_flat = ref_out.to(torch.float32).reshape(-1, hf_config.hidden_size)
    tt_out_flat = tt_output_torch.to(torch.float32).reshape(-1, hf_config.hidden_size)
    rows = min(ref_out_flat.shape[0], tt_out_flat.shape[0])
    logger.info(f"Mode: {mode}, Seq len: {seq_len}, compare_rows={rows}, chunk_size={TARGET_CHUNK_SIZE}")

    num_chunks = (rows + TARGET_CHUNK_SIZE - 1) // TARGET_CHUNK_SIZE
    min_pcc = float("inf")
    all_passed = True

    for chunk_idx in range(num_chunks):
        start = chunk_idx * TARGET_CHUNK_SIZE
        end = min(start + TARGET_CHUNK_SIZE, rows)
        chunk_passed, chunk_pcc = comp_pcc(
            ref_out_flat[start:end],
            tt_out_flat[start:end],
            PCC_REQUIRED_EXPERTS,
        )
        min_pcc = min(min_pcc, chunk_pcc)
        all_passed = all_passed and chunk_passed

    status = "PASS" if all_passed else "FAIL"
    logger.info(
        f"[{status}] experts forward mode={mode} seq_len={seq_len} | min_chunk_pcc={min_pcc} | required>={PCC_REQUIRED_EXPERTS}"
    )
    assert all_passed, f"Experts output mismatch: min_chunk_pcc={min_pcc} below required {PCC_REQUIRED_EXPERTS}"


if __name__ == "__main__":
    pytest.main([__file__])
