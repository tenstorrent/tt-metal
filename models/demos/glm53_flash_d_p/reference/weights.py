# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access and FP8 dequantization for GLM-5.3-Flash (shared by the CPU reference and the HF oracle).

Storage (quantization_config: fp8 e4m3, weight_block_size [128, 128]):
    FP8 e4m3 + float32 ``<name>_scale_inv`` [ceil(out/128), ceil(in/128)]: dense MLP (layers 0-2), shared expert,
        routed experts, DSA q_a_proj / q_b_proj / kv_a_proj_with_mqa / o_proj. W = fp8(w) * scale_inv[r//128, c//128].
    BF16: KDA projections, convs (q/k/v_conv1d [8192, 1, 4] each), norms, router weight, indexer, kv_b_proj, mHC fn,
        embedding, lm_head. FP32: A_log, dt_bias, e_score_correction_bias, mHC base and scale.
Every tensor is read through model.safetensors.index.json (the subset trim renames shards).
"""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open

FP8_BLOCK = 128
PREFIX = "model.language_model."


class WeightLoader:
    """Lazy safetensors accessor keyed by checkpoint tensor name; never reads model.visual.* or the MTP layer."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles = {}

    def get(self, name: str) -> torch.Tensor:
        assert not name.startswith("model.visual"), name
        fname = self.weight_map[name]
        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.model_path, fname), framework="pt")
        return self._handles[fname].get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def layer(self, i: int, name: str) -> torch.Tensor:
        return self.get(f"{PREFIX}layers.{i}.{name}")

    def is_fp8(self, name: str) -> bool:
        return self.has(name + "_scale_inv")

    def weight(self, name: str, dtype=torch.float32) -> torch.Tensor:
        """A weight as the model uses it: FP8 blocks dequantized, anything else cast."""
        if self.is_fp8(name):
            return fp8_block_dequant(self.get(name), self.get(name + "_scale_inv"), dtype)
        return self.get(name).to(dtype)


def fp8_block_dequant(w: torch.Tensor, scale_inv: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """W = fp8(w) * scale_inv[row // 128, col // 128], the product in fp32 (exact: e4m3 x fp32), then cast."""
    rows, cols = w.shape
    s = scale_inv.float()
    if rows % FP8_BLOCK == 0 and cols % FP8_BLOCK == 0:
        wf = w.float().view(rows // FP8_BLOCK, FP8_BLOCK, cols // FP8_BLOCK, FP8_BLOCK)
        return (wf * s[:, None, :, None]).view(rows, cols).to(dtype)
    s = s.repeat_interleave(FP8_BLOCK, 0)[:rows].repeat_interleave(FP8_BLOCK, 1)[:, :cols]
    return (w.float() * s).to(dtype)


class PackedExpert:
    """One routed expert in its checkpoint form (fp8 + block scales), dequantized on use."""

    __slots__ = ("loader", "prefix")

    def __init__(self, loader: WeightLoader, layer: int, expert: int):
        self.loader = loader
        self.prefix = f"{PREFIX}layers.{layer}.mlp.experts.{expert}."

    def weights(self, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(gate [I, H], up [I, H], down [H, I])."""
        return tuple(
            self.loader.weight(self.prefix + f"{n}.weight", dtype) for n in ("gate_proj", "up_proj", "down_proj")
        )
