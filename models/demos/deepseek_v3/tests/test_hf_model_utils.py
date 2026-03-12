# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch

from models.demos.deepseek_v3.utils.hf_model_utils import prepare_model_state_dict, save_dequantized_hf_checkpoint
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict, load_state_dict


def _write_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    index = {"metadata": {}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))


def _dequantize_reference_tensor(
    tensor: torch.Tensor,
    inv_scale: torch.Tensor,
    block_shape: tuple[int, ...],
) -> torch.Tensor:
    original_shape = tuple(tensor.shape)
    padded_shape = tuple(inv_scale.shape[i] * block_shape[i] for i in range(tensor.ndim))
    original_slices = tuple(slice(0, size) for size in original_shape)

    out = tensor.float()
    if padded_shape != original_shape:
        padded = torch.zeros(padded_shape, dtype=out.dtype)
        padded[original_slices] = out
        out = padded

    interleaved_shape: list[int] = []
    scale_broadcast_shape: list[int] = []
    for dim, block_dim in enumerate(block_shape):
        blocks = inv_scale.shape[dim]
        interleaved_shape.extend([blocks, block_dim])
        scale_broadcast_shape.extend([blocks, 1])

    out_view = out.reshape(*interleaved_shape)
    out_view.mul_(inv_scale.float().reshape(*scale_broadcast_shape))
    return out_view.reshape(*padded_shape)[original_slices]


def _create_quantized_checkpoint(model_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"quantization_config": {"weight_block_size": [2, 2]}}))
    (model_dir / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "DeepSeekTokenizer"}))

    quantized_weight = torch.tensor(
        [
            [1.0, -2.0, 3.0],
            [4.0, -5.0, 6.0],
            [7.0, -8.0, 9.0],
        ],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    inverse_scale = torch.tensor(
        [
            [0.5, 1.0],
            [2.0, 4.0],
        ],
        dtype=torch.float32,
    )
    plain_weight = torch.tensor([[0.25, -0.5], [1.5, -2.0]], dtype=torch.float32)
    integer_weight = torch.tensor([1, 2, 3], dtype=torch.int32)

    shard1 = model_dir / "model-00001-of-00002.safetensors"
    shard2 = model_dir / "model-00002-of-00002.safetensors"
    safetensors.torch.save_file(
        {
            "w_quant": quantized_weight,
            "w_quant_scale_inv": inverse_scale,
            "w_plain": plain_weight,
        },
        str(shard1),
    )
    safetensors.torch.save_file({"w_int": integer_weight}, str(shard2))
    _write_index(
        model_dir,
        {
            "w_quant": shard1.name,
            "w_quant_scale_inv": shard1.name,
            "w_plain": shard1.name,
            "w_int": shard2.name,
        },
    )
    return quantized_weight, inverse_scale, plain_weight, integer_weight


def test_save_dequantized_hf_checkpoint_exports_bf16_weights(tmp_path: Path):
    source_dir = tmp_path / "deepseek-source"
    output_dir = tmp_path / "deepseek-source-dequantized"
    quantized_weight, inverse_scale, plain_weight, integer_weight = _create_quantized_checkpoint(source_dir)

    saved_path = save_dequantized_hf_checkpoint(source_dir, output_model_path=output_dir)

    assert saved_path == output_dir.resolve()
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "tokenizer_config.json").is_file()

    output_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert set(output_index["weight_map"]) == {"w_quant", "w_plain", "w_int"}
    assert "w_quant_scale_inv" not in output_index["weight_map"]

    state_dict = load_state_dict(output_dir, "")
    try:
        expected_quantized = _dequantize_reference_tensor(quantized_weight, inverse_scale, (2, 2)).to(torch.bfloat16)

        assert "w_quant_scale_inv" not in state_dict
        assert state_dict["w_quant"].dtype == torch.bfloat16
        assert torch.equal(state_dict["w_quant"], expected_quantized)
        assert state_dict["w_plain"].dtype == torch.bfloat16
        assert torch.equal(state_dict["w_plain"], plain_weight.to(torch.bfloat16))
        assert torch.equal(state_dict["w_int"], integer_weight)
    finally:
        state_dict.close()


def test_dequantize_state_dict_compat_shim_handles_quantized_inputs():
    quantized_weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    inverse_scale = torch.tensor([[0.5]], dtype=torch.float32)
    plain_weight = torch.tensor([1.25, -0.75], dtype=torch.float32)
    hf_config = SimpleNamespace(quantization_config={"weight_block_size": [2, 2]})

    dequantized = dequantize_state_dict(
        {
            "layer.weight": quantized_weight,
            "layer.weight_scale_inv": inverse_scale,
            "plain.weight": plain_weight,
        },
        hf_config,
    )

    assert set(dequantized) == {"layer.weight", "plain.weight"}
    assert dequantized["layer.weight"].dtype == torch.bfloat16
    assert torch.equal(
        dequantized["layer.weight"],
        _dequantize_reference_tensor(quantized_weight, inverse_scale, (2, 2)).to(torch.bfloat16),
    )
    assert dequantized["plain.weight"].dtype == torch.bfloat16
    assert torch.equal(dequantized["plain.weight"], plain_weight.to(torch.bfloat16))


def test_prepare_model_state_dict_returns_lazy_state_dict_for_hf_weights(tmp_path: Path):
    model_dir = tmp_path / "deepseek-dequantized"
    model_dir.mkdir(parents=True, exist_ok=True)

    shard = model_dir / "model-00001-of-00001.safetensors"
    lm_head = torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.bfloat16)
    safetensors.torch.save_file(
        {
            "model.embed_tokens.weight": torch.tensor([[0.5, 0.25], [-0.5, -0.25]], dtype=torch.bfloat16),
            "model.norm.weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
            "lm_head.weight": lm_head,
        },
        str(shard),
    )
    _write_index(
        model_dir,
        {
            "model.embed_tokens.weight": shard.name,
            "model.norm.weight": shard.name,
            "lm_head.weight": shard.name,
        },
    )

    state_dict = prepare_model_state_dict(SimpleNamespace(), random_weights=False, model_path=str(model_dir))

    assert isinstance(state_dict, LazyStateDict)
    assert "lm_head.weight" in state_dict
    assert torch.equal(state_dict["lm_head.weight"], lm_head)


def test_prepare_model_state_dict_rejects_quantized_hf_weights(tmp_path: Path):
    model_dir = tmp_path / "deepseek-quantized"
    model_dir.mkdir(parents=True, exist_ok=True)

    shard = model_dir / "model-00001-of-00001.safetensors"
    safetensors.torch.save_file(
        {
            "model.embed_tokens.weight": torch.tensor([[1.0]], dtype=torch.bfloat16),
            "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.tensor([[0.5]], dtype=torch.float32),
            "lm_head.weight": torch.tensor([[2.0]], dtype=torch.bfloat16),
        },
        str(shard),
    )
    _write_index(
        model_dir,
        {
            "model.embed_tokens.weight": shard.name,
            "model.layers.0.self_attn.q_proj.weight_scale_inv": shard.name,
            "lm_head.weight": shard.name,
        },
    )

    with pytest.raises(RuntimeError, match="Detected quantized HF tensors"):
        prepare_model_state_dict(SimpleNamespace(), random_weights=False, model_path=str(model_dir))
