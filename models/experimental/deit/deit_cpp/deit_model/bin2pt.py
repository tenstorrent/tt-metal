# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import torch
from transformers import DeiTForImageClassificationWithTeacher


MODEL_NAME = "/data/hf_cache/Deit/deit-tiny/deit-tiny"
OUTPUT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = OUTPUT_DIR / "weights"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def _reshape_patch_projection_weight(weight: torch.Tensor) -> torch.Tensor:
    padded = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, 1))
    return padded.permute(2, 3, 1, 0).reshape(
        1, 1, padded.shape[2] * padded.shape[3] * padded.shape[1], padded.shape[0]
    )


def _reshape_patch_projection_bias(bias: torch.Tensor) -> torch.Tensor:
    return bias.reshape(1, 1, 1, bias.shape[0])


def _reshape_linear_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.transpose(-2, -1)


def _reshape_linear_bias(bias: torch.Tensor) -> torch.Tensor:
    return bias.reshape(1, 1, 1, bias.shape[0])


def _reshape_layernorm_parameter(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.reshape(1, 1, parameter.shape[0] // 32, 32)


def _reshape_token(token: torch.Tensor) -> torch.Tensor:
    return token


def _reshape_position_embeddings(position_embeddings: torch.Tensor) -> torch.Tensor:
    return position_embeddings


def _tensor_to_file(tensor: torch.Tensor, relative_path: Path) -> dict:
    tensor = tensor.detach().to(torch.float32).contiguous()
    output_path = WEIGHTS_DIR / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tensor.numpy().tobytes())
    return {"file": relative_path.as_posix(), "shape": list(tensor.shape)}


def _export_manifest() -> None:
    print(f"Loading DeiT teacher model from Hugging Face Hub: {MODEL_NAME}")
    model = DeiTForImageClassificationWithTeacher.from_pretrained(MODEL_NAME)
    model.eval()

    state_dict = model.state_dict()
    manifest = {"model_name": MODEL_NAME, "weights_dir": "weights", "tensors": {}}

    special_tensors = {
        "deit.embeddings.cls_token": _reshape_token(state_dict["deit.embeddings.cls_token"]),
        "deit.embeddings.distillation_token": _reshape_token(state_dict["deit.embeddings.distillation_token"]),
        "deit.embeddings.position_embeddings": _reshape_position_embeddings(
            state_dict["deit.embeddings.position_embeddings"]
        ),
        "deit.embeddings.patch_embeddings.projection.weight": _reshape_patch_projection_weight(
            state_dict["deit.embeddings.patch_embeddings.projection.weight"]
        ),
        "deit.embeddings.patch_embeddings.projection.bias": _reshape_patch_projection_bias(
            state_dict["deit.embeddings.patch_embeddings.projection.bias"]
        ),
        "deit.layernorm.weight": _reshape_layernorm_parameter(state_dict["deit.layernorm.weight"]),
        "deit.layernorm.bias": _reshape_layernorm_parameter(state_dict["deit.layernorm.bias"]),
        "cls_classifier.weight": _reshape_linear_weight(state_dict["cls_classifier.weight"]),
        "cls_classifier.bias": _reshape_linear_bias(state_dict["cls_classifier.bias"]),
        "distillation_classifier.weight": _reshape_linear_weight(state_dict["distillation_classifier.weight"]),
        "distillation_classifier.bias": _reshape_linear_bias(state_dict["distillation_classifier.bias"]),
    }

    for tensor_name, tensor in special_tensors.items():
        relative_path = Path(*tensor_name.split(".")).with_suffix(".bin")
        manifest["tensors"][tensor_name] = _tensor_to_file(tensor, relative_path)

    for layer_idx in range(model.config.num_hidden_layers):
        prefix = f"deit.encoder.layer.{layer_idx}"

        query_weight = state_dict[f"{prefix}.attention.attention.query.weight"]
        key_weight = state_dict[f"{prefix}.attention.attention.key.weight"]
        value_weight = state_dict[f"{prefix}.attention.attention.value.weight"]
        qkv_weight = torch.cat(
            [
                _reshape_linear_weight(query_weight),
                _reshape_linear_weight(key_weight),
                _reshape_linear_weight(value_weight),
            ],
            dim=-1,
        )

        query_bias = state_dict[f"{prefix}.attention.attention.query.bias"]
        key_bias = state_dict[f"{prefix}.attention.attention.key.bias"]
        value_bias = state_dict[f"{prefix}.attention.attention.value.bias"]
        qkv_bias = _reshape_linear_bias(torch.cat([query_bias, key_bias, value_bias], dim=0))

        layer_tensors = {
            f"{prefix}.layernorm_before.weight": _reshape_layernorm_parameter(
                state_dict[f"{prefix}.layernorm_before.weight"]
            ),
            f"{prefix}.layernorm_before.bias": _reshape_layernorm_parameter(
                state_dict[f"{prefix}.layernorm_before.bias"]
            ),
            f"{prefix}.attention.attention.qkv.weight": qkv_weight,
            f"{prefix}.attention.attention.qkv.bias": qkv_bias,
            f"{prefix}.attention.output.dense.weight": _reshape_linear_weight(
                state_dict[f"{prefix}.attention.output.dense.weight"]
            ),
            f"{prefix}.attention.output.dense.bias": _reshape_linear_bias(
                state_dict[f"{prefix}.attention.output.dense.bias"]
            ),
            f"{prefix}.layernorm_after.weight": _reshape_layernorm_parameter(
                state_dict[f"{prefix}.layernorm_after.weight"]
            ),
            f"{prefix}.layernorm_after.bias": _reshape_layernorm_parameter(
                state_dict[f"{prefix}.layernorm_after.bias"]
            ),
            f"{prefix}.intermediate.dense.weight": _reshape_linear_weight(
                state_dict[f"{prefix}.intermediate.dense.weight"]
            ),
            f"{prefix}.intermediate.dense.bias": _reshape_linear_bias(state_dict[f"{prefix}.intermediate.dense.bias"]),
            f"{prefix}.output.dense.weight": _reshape_linear_weight(state_dict[f"{prefix}.output.dense.weight"]),
            f"{prefix}.output.dense.bias": _reshape_linear_bias(state_dict[f"{prefix}.output.dense.bias"]),
        }

        for tensor_name, tensor in layer_tensors.items():
            relative_path = Path(*tensor_name.split(".")).with_suffix(".bin")
            manifest["tensors"][tensor_name] = _tensor_to_file(tensor, relative_path)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Exported DeiT manifest to {MANIFEST_PATH}")
    print(f"Exported {len(manifest['tensors'])} tensor blobs to {WEIGHTS_DIR}")


if __name__ == "__main__":
    _export_manifest()
