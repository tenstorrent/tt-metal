# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import models.demos.deepseek_v3.scripts.dequantize_hf_checkpoint as dequant_script
from models.demos.deepseek_v3.scripts.dequantize_hf_checkpoint import convert_checkpoint


def _write_config(model_dir: Path, block_shape: tuple[int, int]) -> None:
    (model_dir / "config.json").write_text(
        json.dumps({"quantization_config": {"weight_block_size": list(block_shape)}}),
        encoding="utf-8",
    )


def _write_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def _write_tiny_checkpoint(model_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_config(model_dir, block_shape=(2, 2))

    quantized_weight = torch.tensor(
        [[0.5, -1.0, 1.25], [2.0, -2.5, 3.0]],
        dtype=torch.float8_e4m3fn,
    )
    scale_inv = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    normal_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    shard_1 = "model-00001-of-00002.safetensors"
    shard_2 = "model-00002-of-00002.safetensors"

    save_file(
        {
            "model.layers.0.mlp.gate_proj.weight": quantized_weight,
            "model.norm.weight": normal_weight,
        },
        str(model_dir / shard_1),
    )
    save_file(
        {"model.layers.0.mlp.gate_proj.weight_scale_inv": scale_inv},
        str(model_dir / shard_2),
    )

    _write_index(
        model_dir,
        {
            "model.layers.0.mlp.gate_proj.weight": shard_1,
            "model.norm.weight": shard_1,
            "model.layers.0.mlp.gate_proj.weight_scale_inv": shard_2,
        },
    )

    # Auxiliary file must be copied to output.
    (model_dir / "tokenizer.json").write_text('{"ok": true}', encoding="utf-8")

    return quantized_weight, scale_inv, normal_weight


def _expected_dequantized(weight: torch.Tensor, scale_inv: torch.Tensor, block_shape: tuple[int, int]) -> torch.Tensor:
    expanded = scale_inv
    for dim, block_dim in enumerate(block_shape):
        expanded = expanded.repeat_interleave(block_dim, dim=dim)
    return weight.float() * expanded[: weight.shape[0], : weight.shape[1]].float()


def _load_tensor_from_output(output_dir: Path, index: dict, key: str) -> torch.Tensor:
    shard_name = index["weight_map"][key]
    return load_file(str(output_dir / shard_name))[key]


def test_convert_checkpoint_dequantizes_and_drops_scale_inv(tmp_path: Path):
    input_dir = tmp_path / "input_ckpt"
    output_dir = tmp_path / "output_ckpt"
    quantized_weight, scale_inv, normal_weight = _write_tiny_checkpoint(input_dir)

    convert_checkpoint(
        input_dir=input_dir,
        output_dir=output_dir,
        out_dtype=torch.bfloat16,
        keep_scale_inv=False,
        max_output_shard_size_bytes=1024 * 1024,
        num_workers=2,
    )

    index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))

    assert "model.layers.0.mlp.gate_proj.weight_scale_inv" not in index["weight_map"]

    expected = _expected_dequantized(quantized_weight, scale_inv, (2, 2)).to(torch.bfloat16)
    out_dequant = _load_tensor_from_output(output_dir, index, "model.layers.0.mlp.gate_proj.weight")
    out_normal = _load_tensor_from_output(output_dir, index, "model.norm.weight")
    torch.testing.assert_close(out_dequant, expected, rtol=0, atol=0)
    torch.testing.assert_close(out_normal, normal_weight, rtol=0, atol=0)
    assert out_dequant.dtype == torch.bfloat16

    # Auxiliary files are copied.
    assert (output_dir / "tokenizer.json").is_file()


def test_convert_checkpoint_keeps_scale_inv_when_requested(tmp_path: Path):
    input_dir = tmp_path / "input_ckpt"
    output_dir = tmp_path / "output_ckpt"
    _quantized_weight, scale_inv, _normal_weight = _write_tiny_checkpoint(input_dir)

    convert_checkpoint(
        input_dir=input_dir,
        output_dir=output_dir,
        out_dtype=torch.bfloat16,
        keep_scale_inv=True,
        max_output_shard_size_bytes=1024 * 1024,
    )

    index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert "model.layers.0.mlp.gate_proj.weight_scale_inv" in index["weight_map"]

    output_scale = _load_tensor_from_output(output_dir, index, "model.layers.0.mlp.gate_proj.weight_scale_inv")
    torch.testing.assert_close(output_scale, scale_inv, rtol=0, atol=0)


def test_convert_checkpoint_missing_scale_inv_raises(tmp_path: Path):
    input_dir = tmp_path / "input_missing_scale"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_config(input_dir, block_shape=(2, 2))

    quantized_weight = torch.tensor([[1.0, 0.5], [-1.0, 2.0]], dtype=torch.float8_e4m3fn)
    shard_1 = "model-00001-of-00001.safetensors"
    save_file({"model.layers.0.self_attn.q_a_proj.weight": quantized_weight}, str(input_dir / shard_1))
    _write_index(
        input_dir,
        {
            "model.layers.0.self_attn.q_a_proj.weight": shard_1,
        },
    )

    output_fail = tmp_path / "output_fail"
    with pytest.raises(KeyError, match="Missing inverse-scale tensor"):
        convert_checkpoint(
            input_dir=input_dir,
            output_dir=output_fail,
            out_dtype=torch.bfloat16,
            keep_scale_inv=False,
            max_output_shard_size_bytes=1024 * 1024,
        )


def test_convert_checkpoint_orphan_scale_inv_raises(tmp_path: Path):
    input_dir = tmp_path / "input_orphan_scale"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_config(input_dir, block_shape=(2, 2))

    scale = torch.ones((1, 1), dtype=torch.float32)
    shard = "model-00001-of-00001.safetensors"
    save_file({"model.layers.0.mlp.gate_proj.weight_scale_inv": scale}, str(input_dir / shard))
    _write_index(
        input_dir,
        {
            "model.layers.0.mlp.gate_proj.weight_scale_inv": shard,
        },
    )

    with pytest.raises(ValueError, match="scale tensor without matching base tensor"):
        convert_checkpoint(
            input_dir=input_dir,
            output_dir=tmp_path / "output_orphan_scale",
            out_dtype=torch.bfloat16,
            keep_scale_inv=False,
            max_output_shard_size_bytes=1024 * 1024,
        )


def test_convert_checkpoint_non_fp8_tensor_with_scale_raises(tmp_path: Path):
    input_dir = tmp_path / "input_non_fp8_scale"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_config(input_dir, block_shape=(2, 2))

    shard = "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.q_a_proj.weight": torch.ones((2, 2), dtype=torch.float32),
            "model.layers.0.self_attn.q_a_proj.weight_scale_inv": torch.ones((1, 1), dtype=torch.float32),
        },
        str(input_dir / shard),
    )
    _write_index(
        input_dir,
        {
            "model.layers.0.self_attn.q_a_proj.weight": shard,
            "model.layers.0.self_attn.q_a_proj.weight_scale_inv": shard,
        },
    )

    with pytest.raises(ValueError, match="expected FP8 payload"):
        convert_checkpoint(
            input_dir=input_dir,
            output_dir=tmp_path / "output_non_fp8_scale",
            out_dtype=torch.bfloat16,
            keep_scale_inv=False,
            max_output_shard_size_bytes=1024 * 1024,
        )


def test_convert_checkpoint_index_key_missing_in_shard_raises(tmp_path: Path):
    input_dir = tmp_path / "input_bad_index"
    input_dir.mkdir(parents=True, exist_ok=True)
    _write_config(input_dir, block_shape=(2, 2))

    shard = "model-00001-of-00001.safetensors"
    save_file({"model.norm.weight": torch.ones((2, 2), dtype=torch.float32)}, str(input_dir / shard))
    _write_index(
        input_dir,
        {
            "model.layers.0.mlp.gate_proj.weight": shard,  # not present in shard payload
        },
    )

    with pytest.raises(KeyError, match="missing in the referenced shard"):
        convert_checkpoint(
            input_dir=input_dir,
            output_dir=tmp_path / "output_bad_index",
            out_dtype=torch.bfloat16,
            keep_scale_inv=False,
            max_output_shard_size_bytes=1024 * 1024,
        )


def test_convert_checkpoint_same_input_and_output_raises(tmp_path: Path):
    input_dir = tmp_path / "same_io"
    _write_tiny_checkpoint(input_dir)

    with pytest.raises(ValueError, match="Input and output directories must be different"):
        convert_checkpoint(
            input_dir=input_dir,
            output_dir=input_dir,
            out_dtype=torch.bfloat16,
            keep_scale_inv=False,
            max_output_shard_size_bytes=1024 * 1024,
        )


def test_convert_checkpoint_num_workers_must_be_positive(tmp_path: Path):
    input_dir = tmp_path / "input_ckpt"
    output_dir = tmp_path / "output_ckpt"
    _write_tiny_checkpoint(input_dir)

    with pytest.raises(ValueError, match="num_workers must be > 0"):
        convert_checkpoint(
            input_dir=input_dir,
            output_dir=output_dir,
            out_dtype=torch.bfloat16,
            keep_scale_inv=False,
            max_output_shard_size_bytes=1024 * 1024,
            num_workers=0,
        )


def test_convert_checkpoint_resume_reuses_existing_tmp_outputs(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input_ckpt"
    output_dir = tmp_path / "output_ckpt"
    quantized_weight, scale_inv, _normal_weight = _write_tiny_checkpoint(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_obj = dequant_script.load_index(input_dir / "model.safetensors.index.json")
    weight_map = index_obj["weight_map"]
    keys_by_file = dequant_script.build_keys_by_file(weight_map)
    shard_names = sorted(keys_by_file.keys())
    shard_1_name = shard_names[0]
    shard_1_keys = keys_by_file[shard_1_name]
    block_shape = dequant_script.load_block_shape(input_dir / "config.json")

    shard_1_result = dequant_script.process_source_shard(
        shard_idx=1,
        shard_name=shard_1_name,
        keys=shard_1_keys,
        input_dir=input_dir,
        output_dir=output_dir,
        weight_map=weight_map,
        block_shape=block_shape,
        out_dtype=torch.bfloat16,
        keep_scale_inv=False,
        max_output_shard_size_bytes=1024 * 1024,
    )
    dequant_script.write_source_shard_manifest(
        output_dir=output_dir,
        result=shard_1_result,
        source_keys=shard_1_keys,
        out_dtype=torch.bfloat16,
        keep_scale_inv=False,
    )

    original_process_source_shard = dequant_script.process_source_shard

    def _process_source_shard_with_guard(*args, **kwargs):
        shard_idx = kwargs.get("shard_idx")
        if shard_idx is None and args:
            shard_idx = args[0]
        if shard_idx == 1:
            raise AssertionError("Shard 1 should have been resumed, not regenerated.")
        return original_process_source_shard(*args, **kwargs)

    monkeypatch.setattr(dequant_script, "process_source_shard", _process_source_shard_with_guard)

    convert_checkpoint(
        input_dir=input_dir,
        output_dir=output_dir,
        out_dtype=torch.bfloat16,
        keep_scale_inv=False,
        max_output_shard_size_bytes=1024 * 1024,
        num_workers=1,
        resume=True,
    )

    index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    expected = _expected_dequantized(quantized_weight, scale_inv, (2, 2)).to(torch.bfloat16)
    out_dequant = _load_tensor_from_output(output_dir, index, "model.layers.0.mlp.gate_proj.weight")
    torch.testing.assert_close(out_dequant, expected, rtol=0, atol=0)

    assert not dequant_script.source_shard_manifest_path(output_dir, 1).exists()


def test_convert_checkpoint_resume_recovers_legacy_tmp_outputs_without_manifest(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "input_ckpt"
    output_dir = tmp_path / "output_ckpt"
    quantized_weight, scale_inv, _normal_weight = _write_tiny_checkpoint(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index_obj = dequant_script.load_index(input_dir / "model.safetensors.index.json")
    weight_map = index_obj["weight_map"]
    keys_by_file = dequant_script.build_keys_by_file(weight_map)
    shard_names = sorted(keys_by_file.keys())
    shard_1_name = shard_names[0]
    shard_1_keys = keys_by_file[shard_1_name]
    block_shape = dequant_script.load_block_shape(input_dir / "config.json")

    # Simulate a legacy crashed run that wrote tmp output shards but no resume manifest.
    dequant_script.process_source_shard(
        shard_idx=1,
        shard_name=shard_1_name,
        keys=shard_1_keys,
        input_dir=input_dir,
        output_dir=output_dir,
        weight_map=weight_map,
        block_shape=block_shape,
        out_dtype=torch.bfloat16,
        keep_scale_inv=False,
        max_output_shard_size_bytes=1024 * 1024,
    )

    original_process_source_shard = dequant_script.process_source_shard

    def _process_source_shard_with_guard(*args, **kwargs):
        shard_idx = kwargs.get("shard_idx")
        if shard_idx is None and args:
            shard_idx = args[0]
        if shard_idx == 1:
            raise AssertionError("Shard 1 should have been recovered from legacy tmp files.")
        return original_process_source_shard(*args, **kwargs)

    monkeypatch.setattr(dequant_script, "process_source_shard", _process_source_shard_with_guard)

    convert_checkpoint(
        input_dir=input_dir,
        output_dir=output_dir,
        out_dtype=torch.bfloat16,
        keep_scale_inv=False,
        max_output_shard_size_bytes=1024 * 1024,
        num_workers=1,
        resume=True,
    )

    index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    expected = _expected_dequantized(quantized_weight, scale_inv, (2, 2)).to(torch.bfloat16)
    out_dequant = _load_tensor_from_output(output_dir, index, "model.layers.0.mlp.gate_proj.weight")
    torch.testing.assert_close(out_dequant, expected, rtol=0, atol=0)


def test_convert_checkpoint_resume_ignores_corrupt_legacy_tmp_files(tmp_path: Path):
    input_dir = tmp_path / "input_ckpt"
    output_dir = tmp_path / "output_ckpt"
    quantized_weight, scale_inv, _normal_weight = _write_tiny_checkpoint(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate a crashed writer that left a truncated/corrupt tmp safetensors file.
    (output_dir / ".tmp-s00001-00001.safetensors").write_bytes(b"not-a-valid-safetensors-file")

    convert_checkpoint(
        input_dir=input_dir,
        output_dir=output_dir,
        out_dtype=torch.bfloat16,
        keep_scale_inv=False,
        max_output_shard_size_bytes=1024 * 1024,
        num_workers=1,
        resume=True,
    )

    index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    expected = _expected_dequantized(quantized_weight, scale_inv, (2, 2)).to(torch.bfloat16)
    out_dequant = _load_tensor_from_output(output_dir, index, "model.layers.0.mlp.gate_proj.weight")
    torch.testing.assert_close(out_dequant, expected, rtol=0, atol=0)
