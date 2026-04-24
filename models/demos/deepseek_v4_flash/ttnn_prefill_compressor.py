# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.cpu_reference import rms_norm
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest


@dataclass(frozen=True)
class PrefillCompressorWeights:
    wkv: torch.Tensor
    wgate: torch.Tensor
    ape: torch.Tensor
    norm_weight: torch.Tensor


class TtPrefillCompressor(LightweightModule):
    """Single-device TTNN projection path for DeepSeek V4 Flash compressor prefill.

    The ratio-window reshape, softmax, and reduction still run on host. That keeps
    this first prefill slice small while exercising converted compressor weights
    and TTNN linear projections on the target device.
    """

    def __init__(
        self,
        *,
        device,
        weights: PrefillCompressorWeights,
        compress_ratio: int,
        head_dim: int,
        norm_eps: float = 1e-6,
        overlap: bool | None = None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        if overlap is None:
            overlap = compress_ratio == 4
        _validate_compressor_weights(
            weights,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            overlap=overlap,
        )

        self.device = device
        self.dtype = dtype
        self.memory_config = memory_config
        self.compress_ratio = int(compress_ratio)
        self.head_dim = int(head_dim)
        self.norm_eps = float(norm_eps)
        self.overlap = bool(overlap)
        self.projected_dim = int(weights.wkv.shape[0])
        self.ape = weights.ape.float().contiguous()
        self.norm_weight = weights.norm_weight.float().contiguous()
        self.wkv = _to_tt_linear_weight(weights.wkv, device=device, dtype=dtype, memory_config=memory_config)
        self.wgate = _to_tt_linear_weight(weights.wgate, device=device, dtype=dtype, memory_config=memory_config)

    @classmethod
    def from_preprocessed(
        cls,
        preprocessed_path: str | Path,
        *,
        device,
        layer: int,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        overlap: bool | None = None,
    ) -> "TtPrefillCompressor":
        preprocessed_path = Path(preprocessed_path)
        manifest = load_tt_manifest(preprocessed_path)
        config = manifest["config"]
        compress_ratio = int(config["compress_ratios"][layer])
        if compress_ratio == 0:
            raise ValueError(f"Layer {layer} has compress_ratio=0 and no prefill compressor")
        return cls(
            device=device,
            weights=load_prefill_compressor_weights(preprocessed_path, manifest=manifest, layer=layer),
            compress_ratio=compress_ratio,
            head_dim=int(config["head_dim"]),
            norm_eps=float(config["rms_norm_eps"]),
            overlap=overlap,
            dtype=dtype,
            memory_config=memory_config,
        )

    def forward(self, hidden_states):
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [batch, 1, seq_len, hidden], got {input_shape}")
        seq_len = int(input_shape[2])
        if seq_len < self.compress_ratio:
            raise ValueError(
                f"Prefill compressor requires seq_len >= compress_ratio; got {seq_len} < {self.compress_ratio}"
            )

        kv = ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config)
        score = ttnn.linear(hidden_states, self.wgate, memory_config=self.memory_config)
        kv_host = _ttnn_prefill_to_torch_3d(kv)
        score_host = _ttnn_prefill_to_torch_3d(score)
        compressed = _compress_projected_prefill(
            kv_host,
            score_host,
            ape=self.ape,
            norm_weight=self.norm_weight,
            compress_ratio=self.compress_ratio,
            head_dim=self.head_dim,
            norm_eps=self.norm_eps,
            overlap=self.overlap,
        )
        return ttnn.from_torch(
            compressed.unsqueeze(1).to(torch.bfloat16),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.memory_config,
        )


def load_prefill_compressor_weights(
    preprocessed_path: str | Path,
    *,
    manifest: dict | None = None,
    layer: int,
) -> PrefillCompressorWeights:
    preprocessed_path = Path(preprocessed_path)
    if manifest is None:
        manifest = load_tt_manifest(preprocessed_path)
    prefix = f"layers.{layer}.attn.compressor"
    keys = {
        "wkv": f"{prefix}.wkv.weight",
        "wgate": f"{prefix}.wgate.weight",
        "ape": f"{prefix}.ape",
        "norm_weight": f"{prefix}.norm.weight",
    }
    loaded: dict[str, torch.Tensor] = {}
    for artifact in manifest["artifacts"]["non_expert_safetensors"]:
        with safe_open(preprocessed_path / artifact, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name, key in keys.items():
                if name not in loaded and key in available:
                    loaded[name] = handle.get_tensor(key).contiguous()
        if len(loaded) == len(keys):
            return PrefillCompressorWeights(
                wkv=loaded["wkv"],
                wgate=loaded["wgate"],
                ape=loaded["ape"],
                norm_weight=loaded["norm_weight"],
            )

    missing = sorted(key for name, key in keys.items() if name not in loaded)
    raise KeyError(f"Missing prefill compressor weights for layer {layer}: {missing}")


def _validate_compressor_weights(
    weights: PrefillCompressorWeights,
    *,
    compress_ratio: int,
    head_dim: int,
    overlap: bool,
) -> None:
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    expected_projected_dim = 2 * head_dim if overlap else head_dim
    if weights.wkv.ndim != 2:
        raise ValueError(f"Expected wkv to be rank 2, got shape {tuple(weights.wkv.shape)}")
    if weights.wgate.shape != weights.wkv.shape:
        raise ValueError(f"Expected wgate shape {tuple(weights.wkv.shape)}, got {tuple(weights.wgate.shape)}")
    if weights.wkv.shape[0] != expected_projected_dim:
        raise ValueError(
            f"Expected projected dim {expected_projected_dim} for head_dim={head_dim}, overlap={overlap}, "
            f"got {weights.wkv.shape[0]}"
        )
    if tuple(weights.ape.shape) != (compress_ratio, expected_projected_dim):
        raise ValueError(
            f"Expected ape shape {(compress_ratio, expected_projected_dim)}, got {tuple(weights.ape.shape)}"
        )
    if tuple(weights.norm_weight.shape) != (head_dim,):
        raise ValueError(f"Expected norm_weight shape {(head_dim,)}, got {tuple(weights.norm_weight.shape)}")


def _compress_projected_prefill(
    kv: torch.Tensor,
    score: torch.Tensor,
    *,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    compress_ratio: int,
    head_dim: int,
    norm_eps: float,
    overlap: bool,
) -> torch.Tensor:
    batch_size, seq_len, _ = kv.shape
    cutoff = seq_len - (seq_len % compress_ratio)
    kv = kv.float()[:, :cutoff].unflatten(1, (-1, compress_ratio))
    score = score.float()[:, :cutoff].unflatten(1, (-1, compress_ratio)) + ape.float()
    if overlap:
        kv = _overlap_transform(kv, head_dim, value=0)
        score = _overlap_transform(score, head_dim, value=float("-inf"))
    pooled = (kv * score.softmax(dim=2)).sum(dim=2)
    expected_shape = (batch_size, cutoff // compress_ratio, head_dim)
    if tuple(pooled.shape) != expected_shape:
        raise ValueError(f"Expected compressed shape {expected_shape}, got {tuple(pooled.shape)}")
    return rms_norm(pooled, norm_weight, norm_eps)


def _overlap_transform(tensor: torch.Tensor, head_dim: int, *, value: float) -> torch.Tensor:
    batch_size, seq_blocks, ratio, _ = tensor.shape
    out = tensor.new_full((batch_size, seq_blocks, 2 * ratio, head_dim), value)
    out[:, :, ratio:] = tensor[:, :, :, head_dim:]
    out[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
    return out


def _to_tt_linear_weight(
    weight: torch.Tensor,
    *,
    device,
    dtype,
    memory_config,
):
    torch_weight = weight.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    return ttnn.from_torch(
        torch_weight,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )


def _ttnn_prefill_to_torch_3d(tensor) -> torch.Tensor:
    torch_tensor = ttnn.to_torch(tensor)
    if torch_tensor.ndim != 4 or torch_tensor.shape[1] != 1:
        raise ValueError(
            f"Expected TTNN prefill tensor shape [batch, 1, seq_len, dim], got {tuple(torch_tensor.shape)}"
        )
    return torch_tensor[:, 0].contiguous()
