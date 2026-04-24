# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.fp4 import EXPERT_FP4_BLOCK_SIZE, EXPERT_WEIGHT_ABI
from models.demos.deepseek_v4_flash.key_mapping import expert_packed_key, normalize_hf_key
from models.demos.deepseek_v4_flash.manifest import MODEL_NAME, TT_MANIFEST_SCHEMA_VERSION, write_tt_manifest


MODEL_INDEX_FILENAME = "model.safetensors.index.json"
NON_EXPERT_DIR = "non_expert"
EXPERT_DIR = "experts"
METADATA_FILENAME = "metadata.safetensors"
_COPIED_CONFIG_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")


def convert_hf_checkpoint(
    source_model_path: str | Path, output_model_path: str | Path, *, overwrite: bool = False
) -> Path:
    source_model_path = Path(source_model_path).resolve()
    output_model_path = Path(output_model_path).resolve()
    if not source_model_path.is_dir():
        raise FileNotFoundError(f"Source model path does not exist: {source_model_path}")
    if output_model_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {output_model_path}")
        if output_model_path.is_dir():
            shutil.rmtree(output_model_path)
        else:
            output_model_path.unlink()

    config = DeepSeekV4FlashConfig.from_model_path(source_model_path)
    output_model_path.mkdir(parents=True)
    (output_model_path / NON_EXPERT_DIR).mkdir()
    (output_model_path / EXPERT_DIR).mkdir()

    copied_files = _copy_config_and_tokenizer_files(source_model_path, output_model_path)
    weight_map = _load_weight_map(source_model_path)
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard_name in weight_map.items():
        shard_to_keys.setdefault(shard_name, []).append(key)

    non_expert_artifacts: list[str] = []
    expert_artifacts: list[str] = []
    counts = {"non_expert_tensors": 0, "expert_tensors": 0}
    for shard_index, shard_name in enumerate(sorted(shard_to_keys)):
        shard_path = source_model_path / shard_name
        _raise_if_lfs_pointer(shard_path)
        non_expert_tensors: dict[str, torch.Tensor] = {}
        expert_tensors: dict[str, torch.Tensor] = {}
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for source_key in sorted(shard_to_keys[shard_name]):
                mapped = normalize_hf_key(source_key)
                tensor = handle.get_tensor(source_key)
                if mapped.category == "expert":
                    output_key = (
                        expert_packed_key(mapped.canonical)
                        if mapped.tensor_kind == "weight"
                        else mapped.canonical
                    )
                    expert_tensors[output_key] = _canonicalize_expert_tensor(tensor, mapped.tensor_kind)
                    counts["expert_tensors"] += 1
                else:
                    non_expert_tensors[mapped.canonical] = tensor.contiguous()
                    counts["non_expert_tensors"] += 1

        if non_expert_tensors:
            artifact = f"{NON_EXPERT_DIR}/shard-{shard_index:05d}.safetensors"
            save_file(non_expert_tensors, str(output_model_path / artifact))
            non_expert_artifacts.append(artifact)
        if expert_tensors:
            artifact = f"{EXPERT_DIR}/shard-{shard_index:05d}.safetensors"
            save_file(expert_tensors, str(output_model_path / artifact))
            expert_artifacts.append(artifact)

    metadata_artifact = METADATA_FILENAME
    save_file(_metadata_tensors(config), str(output_model_path / metadata_artifact))

    manifest = _build_manifest(
        source_model_path=source_model_path,
        config=config,
        copied_files=copied_files,
        non_expert_artifacts=non_expert_artifacts,
        expert_artifacts=expert_artifacts,
        metadata_artifact=metadata_artifact,
        counts=counts,
    )
    write_tt_manifest(output_model_path, manifest)
    return output_model_path


def _canonicalize_expert_tensor(tensor: torch.Tensor, tensor_kind: str | None) -> torch.Tensor:
    if tensor_kind == "weight":
        if tensor.dtype not in (torch.uint8, torch.int8):
            raise TypeError(f"Expected packed FP4 expert weight bytes, got {tensor.dtype}")
        return tensor.to(torch.uint8).contiguous()
    if tensor_kind == "scale":
        return tensor.contiguous()
    raise ValueError(f"Unsupported expert tensor kind {tensor_kind!r}")


def _copy_config_and_tokenizer_files(source_model_path: Path, output_model_path: Path) -> list[str]:
    copied_files: list[str] = []
    for filename in _COPIED_CONFIG_FILES:
        source = source_model_path / filename
        if source.is_file():
            shutil.copy2(source, output_model_path / filename)
            copied_files.append(filename)
    inference_config = source_model_path / "inference" / "config.json"
    if inference_config.is_file():
        shutil.copy2(inference_config, output_model_path / "inference_config.json")
        copied_files.append("inference_config.json")
    return copied_files


def _load_weight_map(source_model_path: Path) -> dict[str, str]:
    index_path = source_model_path / MODEL_INDEX_FILENAME
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Expected non-empty weight_map in {index_path}")
        return {str(key): str(value) for key, value in weight_map.items()}

    weight_map: dict[str, str] = {}
    for safetensors_path in sorted(source_model_path.glob("*.safetensors")):
        _raise_if_lfs_pointer(safetensors_path)
        with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = safetensors_path.name
    if not weight_map:
        raise FileNotFoundError(f"No safetensors weights found in {source_model_path}")
    return weight_map


def _raise_if_lfs_pointer(path: Path) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(64)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        raise RuntimeError(f"{path} is a Git LFS pointer. Fetch real checkpoint shards before conversion.")


def _metadata_tensors(config: DeepSeekV4FlashConfig) -> dict[str, torch.Tensor]:
    return {
        "schema_version": torch.tensor([TT_MANIFEST_SCHEMA_VERSION], dtype=torch.int32),
        "compress_ratios": torch.tensor(config.compress_ratios, dtype=torch.int32),
        "num_hash_layers": torch.tensor([config.num_hash_layers], dtype=torch.int32),
        "num_experts_per_tok": torch.tensor([config.num_experts_per_tok], dtype=torch.int32),
        "expert_fp4_block_size": torch.tensor([EXPERT_FP4_BLOCK_SIZE], dtype=torch.int32),
        "compress_rope_theta": torch.tensor([config.compress_rope_theta], dtype=torch.float32),
    }


def _build_manifest(
    *,
    source_model_path: Path,
    config: DeepSeekV4FlashConfig,
    copied_files: list[str],
    non_expert_artifacts: list[str],
    expert_artifacts: list[str],
    metadata_artifact: str,
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": TT_MANIFEST_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "source": {
            "path": str(source_model_path),
            "repo_id": "deepseek-ai/DeepSeek-V4-Flash",
        },
        "config": config.to_manifest_dict(),
        "artifacts": {
            "copied_files": copied_files,
            "non_expert_safetensors": non_expert_artifacts,
            "expert_safetensors": expert_artifacts,
            "metadata_safetensors": metadata_artifact,
        },
        "expert_format": {
            "abi": EXPERT_WEIGHT_ABI,
            "block_size": EXPERT_FP4_BLOCK_SIZE,
            "packed_order": "low_nibble_first",
            "scale_axis": "input_blocks",
        },
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a DeepSeek V4 Flash HF checkpoint to TT-preprocessed files.")
    parser.add_argument("--source-model-path", required=True, type=Path)
    parser.add_argument("--output-model-path", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_hf_checkpoint(args.source_model_path, args.output_model_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
