# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SwiGLU MLP for Dots vision: gate/up/down + SiLU + mul fully in TTNN.
"""

from __future__ import annotations

import torch

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt._ttnn_import import get_ttnn
from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs


def _pk(sd: dict, *keys: str) -> str | None:
    for k in keys:
        if k in sd:
            return k
    return None


class VisionMLPTT(LightweightModule):
    def __init__(
        self,
        mesh_device,
        model_args: DotsVisionModelArgs,
        state_dict: dict,
        layer_num: int,
        weight_cache_path=None,
        dtype=None,
    ):
        super().__init__()
        ttnn = get_ttnn()
        if dtype is None:
            dtype = ttnn.bfloat16 if ttnn is not None else torch.bfloat16
        self.mesh_device = mesh_device
        self.model_args = model_args
        self.layer_num = layer_num
        self.dtype = dtype
        self._load_weights(state_dict, weight_cache_path, dtype)

    def _load_weights(self, state_dict: dict, weight_cache_path, dtype):
        base_prefix = self.model_args.get_state_dict_prefix("MLP", self.layer_num)
        # HF checkpoints have varied naming for the MLP module: feed_forward / mlp.
        prefixes = []
        if base_prefix:
            prefixes.append(base_prefix)
            prefixes.append(base_prefix.replace("feed_forward", "mlp"))
        else:
            prefixes.append("")
        prefixes = [p + "." if p and not p.endswith(".") else p for p in prefixes]
        ttnn = get_ttnn()
        mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None) if ttnn is not None else None
        mesh = self.mesh_device
        wcache = weight_cache_path

        def as_tt(w, name: str):
            if ttnn is None or mesh is None:
                return w.clone() if hasattr(w, "clone") else w
            # TTNN `linear` expects weights shaped [in_features, out_features].
            if hasattr(w, "dim") and callable(w.dim) and w.dim() == 2:
                w = torch.transpose(w, -2, -1).contiguous()
            return ttnn.as_tensor(
                w,
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mc,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if hasattr(ttnn, "ReplicateTensorToMesh") else None,
                cache_file_name=(wcache / f"layer_{self.layer_num}_mlp_{name}" if wcache else None),
            )

        # Pick a prefix that exists.
        chosen = None
        for prefix in prefixes:
            if _pk(state_dict, f"{prefix}fc1.weight", f"{prefix}gate_proj.weight", f"{prefix}w1.weight") is not None:
                chosen = prefix
                break
        if chosen is None:
            # Backstop: search for layer id in keys (blocks/layers).
            layer_tags = (f"blocks.{self.layer_num}.", f"layers.{self.layer_num}.", f"layer.{self.layer_num}.")
            for lt in layer_tags:
                for k in state_dict.keys():
                    if lt in k and (
                        k.endswith("fc1.weight") or k.endswith("gate_proj.weight") or k.endswith("w1.weight")
                    ):
                        chosen = (
                            k.split("fc1")[0]
                            if "fc1" in k
                            else (k.split("gate_proj")[0] if "gate_proj" in k else k.split("w1")[0])
                        )
                        break
                if chosen is not None:
                    break
        if chosen is None:
            self.w1 = self.w2 = self.w3 = None
            self.b1 = self.b2 = self.b3 = None
            return
        prefix = chosen

        # Dots: fc1 (gate), fc3 (up), fc2 (down) — SwiGLU: y = w2( silu(w1(x)) * w3(x) )
        w1k = _pk(
            state_dict, f"{prefix}fc1.weight", f"{prefix}w1.weight", f"{prefix}c_fc.weight", f"{prefix}gate_proj.weight"
        )
        w3k = _pk(state_dict, f"{prefix}fc3.weight", f"{prefix}w3.weight", f"{prefix}up_proj.weight")
        w2k = _pk(
            state_dict,
            f"{prefix}fc2.weight",
            f"{prefix}w2.weight",
            f"{prefix}down_proj.weight",
            f"{prefix}c_proj.weight",
        )
        b1k = _pk(state_dict, f"{prefix}fc1.bias", f"{prefix}w1.bias", f"{prefix}gate_proj.bias")
        b3k = _pk(state_dict, f"{prefix}fc3.bias", f"{prefix}w3.bias", f"{prefix}up_proj.bias")
        b2k = _pk(state_dict, f"{prefix}fc2.bias", f"{prefix}w2.bias", f"{prefix}down_proj.bias")
        self.w1 = as_tt(state_dict[w1k], "w1") if w1k else None
        self.b1 = as_tt(state_dict[b1k], "b1") if b1k else None
        self.w3 = as_tt(state_dict[w3k], "w3") if w3k else None
        self.b3 = as_tt(state_dict[b3k], "b3") if b3k else None
        self.w2 = as_tt(state_dict[w2k], "w2") if w2k else None
        self.b2 = as_tt(state_dict[b2k], "b2") if b2k else None

    def forward(self, x):
        ttnn = get_ttnn()
        if ttnn is None:
            raise RuntimeError("VisionMLPTT requires ttnn")
        if self.w1 is None or self.w2 is None or self.w3 is None:
            raise ValueError("VisionMLPTT weights not loaded")
        if not isinstance(x, ttnn.Tensor):
            raise TypeError(f"Expected ttnn.Tensor, got {type(x)}")

        h1 = ttnn.linear(x, self.w1, bias=self.b1, memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None))
        h3 = ttnn.linear(x, self.w3, bias=self.b3, memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None))
        act = ttnn.silu(h1)
        m = ttnn.mul(act, h3)
        return ttnn.linear(m, self.w2, bias=self.b2, memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None))


def create_vision_mlp(mesh_device, model_args, state_dict, layer_num, weight_cache_path=None, dtype=None):
    return VisionMLPTT(
        mesh_device=mesh_device,
        model_args=model_args,
        state_dict=state_dict,
        layer_num=layer_num,
        weight_cache_path=weight_cache_path,
        dtype=dtype,
    )
