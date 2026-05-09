# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import types

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.mistral_small_4_119B.tt.moe.moe_gate import TtMistral4MoEGate
from tests.ttnn.utils_for_testing import comp_pcc

TOPK_WEIGHTS_PCC_REQUIRED = 0.99
TOPK_INDICES_MATCH_RATE_REQUIRED = 0.90

_PREFILL_SEQ_LEN = int(os.environ.get("TT_MOE_TEST_PREFILL_TOKENS", "16"))


def _activations_mesh_mapper(mesh_device: ttnn.Device):
    if mesh_device.get_num_devices() == 1:
        return None
    return ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))


def _tiny_mistral4_config():
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    cfg = Mistral4Config(
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
    if getattr(cfg, "_experts_implementation", None) in (None, ""):
        cfg._experts_implementation = "grouped_mm"
    return cfg


def _route_tokens_to_experts(cfg, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = router_logits.softmax(-1)
    group_scores = probs.view(-1, cfg.n_group, cfg.n_routed_experts // cfg.n_group).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=cfg.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, cfg.n_group, cfg.n_routed_experts // cfg.n_group)
        .reshape(-1, cfg.n_routed_experts)
    )
    scores_for_choice = probs.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(scores_for_choice, k=cfg.num_experts_per_tok, dim=-1, sorted=False)[1]
    topk_weights = probs.gather(1, topk_indices)
    if cfg.norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator
    topk_weights = topk_weights * cfg.routed_scaling_factor
    return topk_weights, topk_indices


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "mode,batch_size_per_row,seq_len",
    [
        ("decode", 1, 1),
        ("prefill", 1, _PREFILL_SEQ_LEN),
    ],
)
@pytest.mark.parametrize(
    "topk_fallback,use_bitonic_sort",
    [
        (True, True),
    ],
)
def test_forward_pass(
    mode,
    batch_size_per_row,
    seq_len,
    topk_fallback,
    use_bitonic_sort,
    mesh_device,
    tmp_path,
):
    """Test Mistral4 gate forward against HF routing math."""

    from transformers.models.mistral4.modeling_mistral4 import Mistral4TopkRouter

    hf_config = _tiny_mistral4_config()
    num_tokens = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else seq_len

    reference_model = Mistral4TopkRouter(hf_config).eval()
    hf_state_dict = reference_model.state_dict()
    weight_config = TtMistral4MoEGate.convert_weights(
        hf_config,
        (hf_state_dict,),
        tmp_path / "moe_gate_weights",
        mesh_device,
    )

    model_config = (
        TtMistral4MoEGate.decode_model_config(
            hf_config, mesh_device, topk_fallback=topk_fallback, use_bitonic_sort=use_bitonic_sort
        )
        if mode == "decode"
        else TtMistral4MoEGate.prefill_model_config(
            hf_config, mesh_device, topk_fallback=topk_fallback, use_bitonic_sort=use_bitonic_sort
        )
    )
    model_state = TtMistral4MoEGate.create_state(hf_config, mesh_device=mesh_device, ccl=None)

    run_config = dict(model_config)
    run_config.update(weight_config)
    run_config.update(model_state)
    run_config["mesh_device"] = mesh_device

    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)
    router_logits_ref = reference_model(torch_input.to(torch.float32))
    reference_topk_weights, reference_topk_indices = _route_tokens_to_experts(hf_config, router_logits_ref)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=_activations_mesh_mapper(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_topk_weights, tt_topk_indices = (
        TtMistral4MoEGate.forward_decode(tt_input, run_config)
        if mode == "decode"
        else TtMistral4MoEGate.forward_prefill(tt_input, run_config)
    )

    expected_output_memory_config = run_config["output_memory_config"]
    assert tt_topk_weights.memory_config() == expected_output_memory_config
    assert tt_topk_indices.memory_config() == expected_output_memory_config

    if mesh_device.get_num_devices() == 1:
        tt_topk_weights_torch = ttnn.to_torch(tt_topk_weights)[0].squeeze(0)
        tt_topk_indices_torch = ttnn.to_torch(tt_topk_indices)[0].squeeze(0)
    else:
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape))
        tt_topk_weights_torch = ttnn.to_torch(tt_topk_weights, mesh_composer=composer)[0].squeeze(0)
        tt_topk_indices_torch = ttnn.to_torch(tt_topk_indices, mesh_composer=composer)[0].squeeze(0)

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_topk_indices)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")

    passing, pcc_message = comp_pcc(
        reference_topk_weights.to(torch.float32),
        tt_topk_weights_torch.to(torch.float32),
        TOPK_WEIGHTS_PCC_REQUIRED,
    )
    status = "PASS" if passing else "FAIL"
    logger.info(
        f"[{status}] moe_gate topk_weights mode={mode} seq_len={seq_len} | PCC={pcc_message} | required>={TOPK_WEIGHTS_PCC_REQUIRED}"
    )
    assert (
        passing
    ), f"TopK experts weights output does not meet PCC requirement {TOPK_WEIGHTS_PCC_REQUIRED}: {pcc_message}"

    # Stable sort to avoid top-k tie-order differences.
    reference_topk_indices = torch.sort(reference_topk_indices.to(torch.int32), dim=-1, stable=True)[0]
    tt_topk_indices_torch = torch.sort(tt_topk_indices_torch.to(torch.int32), dim=-1, stable=True)[0]
    indices_match = reference_topk_indices == tt_topk_indices_torch
    topk_indices_match_rate = indices_match.float().mean().item()
    logger.info(
        f"[{'PASS' if topk_indices_match_rate >= TOPK_INDICES_MATCH_RATE_REQUIRED else 'FAIL'}] "
        f"moe_gate topk_indices mode={mode} seq_len={seq_len} | "
        f"match_rate={topk_indices_match_rate:.6f} | required>={TOPK_INDICES_MATCH_RATE_REQUIRED:.2f}"
    )
    assert (
        topk_indices_match_rate >= TOPK_INDICES_MATCH_RATE_REQUIRED
    ), f"TopK experts indices output match rate {topk_indices_match_rate:.6f} is below required {TOPK_INDICES_MATCH_RATE_REQUIRED:.2f}"


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
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda *args, **kwargs: None)
    monkeypatch.setattr(ttnn, "to_torch", lambda tensor, **kwargs: tensor.payload)
    monkeypatch.setattr(ttnn, "get_device_tensors", lambda tensor: [tensor])

    def fake_from_torch(tensor, **kwargs):
        captured["output"] = tensor
        return _FakeTTTensor(tensor)

    monkeypatch.setattr(ttnn, "from_torch", fake_from_torch)

    mesh_device = types.SimpleNamespace(shape=(1, 1), get_num_devices=lambda: 1)
    output = TtMistral4MoEGate.linear_fallback_op(
        _FakeTTTensor(torch_input_payload),
        _FakeTTTensor(torch_weight_payload),
        mesh_device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        transpose_b=True,
    )

    expected = torch.nn.functional.linear(torch_input_payload[0], torch_weight_payload[0, 0]).unsqueeze(0)
    assert torch.equal(captured["output"], expected)
    assert output.shape == tuple(expected.shape)


if __name__ == "__main__":
    pytest.main([__file__])
