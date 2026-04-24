# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from models.demos.deepseek_v4_flash.fp4 import EXPERT_WEIGHT_ABI, dequantize_fp4_packed
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest


@dataclass(frozen=True)
class PackedExpertWeight:
    layer: int
    expert: int
    projection: str
    weight_packed: torch.Tensor
    scale: torch.Tensor
    abi: str
    block_size: int

    def dequantize(self, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return dequantize_fp4_packed(self.weight_packed, self.scale, block_size=self.block_size, dtype=dtype)


def load_packed_expert_weight(
    preprocessed_path: str | Path,
    *,
    layer: int,
    expert: int,
    projection: str,
) -> PackedExpertWeight:
    preprocessed_path = Path(preprocessed_path)
    manifest = load_tt_manifest(preprocessed_path)
    expert_format = manifest["expert_format"]
    abi = expert_format["abi"]
    if abi != EXPERT_WEIGHT_ABI:
        raise ValueError(f"Unsupported expert ABI {abi!r}")
    block_size = int(expert_format["block_size"])
    weight_key = f"layers.{layer}.ffn.experts.{expert}.{projection}.weight_packed"
    scale_key = f"layers.{layer}.ffn.experts.{expert}.{projection}.scale"

    for artifact in manifest["artifacts"]["expert_safetensors"]:
        artifact_path = preprocessed_path / artifact
        with safe_open(artifact_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if weight_key not in keys and scale_key not in keys:
                continue
            if weight_key not in keys or scale_key not in keys:
                raise ValueError(f"Expert artifact {artifact_path} has incomplete pair for {weight_key}")
            return PackedExpertWeight(
                layer=layer,
                expert=expert,
                projection=projection,
                weight_packed=handle.get_tensor(weight_key),
                scale=handle.get_tensor(scale_key),
                abi=abi,
                block_size=block_size,
            )
    raise KeyError(f"Could not find packed expert weight {weight_key} in {preprocessed_path}")
