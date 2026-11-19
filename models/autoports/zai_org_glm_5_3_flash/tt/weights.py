# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Weight sources for the functional decoder.

Two interchangeable sources produce the same per-layer tensors (same names, layouts and dtypes as the pre-packed tile
cache):

* `TileCacheSource(cache_dir, manifest_path)` — the pre-packed bfp4/bfp8/bf16 tile cache (one `ttnn.dump_tensor`
  file per tensor, described by a JSON manifest). Every 2-D linear is stored `[in, out]`; conv taps `[4, C]`; routed experts one
  file per expert (`gate_up_proj [4096, 4096] = [gate | up]`, `down_proj [2048, 4096]`); kv_b pre-split into
  `W_uk [64, 256, 512]` / `W_uv [64, 512, 256]`.
* `TorchStateDictSource` — a host state dict in the HF post-conversion naming of one decoder layer (module names of
  upstream `Glm5NextTextDecoderLayer`), converted at setup time to the cache layout and dtype map. Used
  for synthetic-weight tests and the `FunctionalDecoder.from_state_dict` boundary.

All host->device conversion happens here, at construction time; the runtime forward paths never touch torch.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.precision import WEIGHT_DTYPES

DRAM = ttnn.DRAM_MEMORY_CONFIG

# dtype of every per-layer tensor (suffix after "layers.<L>."), mirroring weights/glm53_cache_spec.py.
_BFP8_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "mlp.shared_experts.gate_proj.weight",
    "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
)
_FP32_SUFFIXES = (
    "self_attn.A_log",
    "self_attn.dt_bias",
    "mlp.gate.e_score_correction_bias",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_scale",
)


def cache_dtype(suffix: str):
    if ".experts." in suffix:
        return WEIGHT_DTYPES["routed_expert"]
    if suffix in _BFP8_SUFFIXES:
        return WEIGHT_DTYPES["projection_bfp8"]
    if suffix in _FP32_SUFFIXES:
        return WEIGHT_DTYPES["fp32"]
    return WEIGHT_DTYPES["bf16"]


class WeightSource:
    """Per-layer tensor provider. `device(name)` -> device tensor in cache layout/dtype; `host(name)` -> torch."""

    def device(self, layer_idx: int, name: str, mesh_device) -> ttnn.Tensor:
        raise NotImplementedError

    def host(self, layer_idx: int, name: str) -> torch.Tensor:
        raise NotImplementedError

    def experts(self, layer_idx: int, name: str, num_experts: int, mesh_device) -> ttnn.Tensor:
        """Routed experts stacked to [1, E, in, out] (sparse_matmul weight layout)."""
        raise NotImplementedError


_MANIFESTS: dict = {}


def _manifest(path: str) -> dict:
    """Parsed weight-cache manifest (16 MB JSON, 0.5-2 s to parse), shared by every source of the process: the model
    builds one source per layer (and the NFS metadata round trips of the checks cost up to 0.5 s each on a busy share).
    Parsed once per process and path; callers treat it as read-only."""
    if path not in _MANIFESTS:
        with open(path) as f:
            _MANIFESTS[path] = json.load(f)
    return _MANIFESTS[path]


class TileCacheSource(WeightSource):
    def __init__(self, cache_dir: str, manifest_path: str):
        if manifest_path not in _MANIFESTS or not cache_dir:  # checked once per process and path (see _manifest)
            if not cache_dir or not os.path.isdir(cache_dir):
                raise FileNotFoundError(f"weight cache dir {cache_dir!r} not found")
            if not manifest_path or not os.path.isfile(manifest_path):
                raise FileNotFoundError(f"weight cache manifest {manifest_path!r} not found")
        self.cache_dir = cache_dir
        manifest = _manifest(manifest_path)
        self.tensors = manifest["tensors"]
        self.tt_metal_sha = manifest.get("tt_metal_sha")

    def _entry(self, layer_idx: int, name: str) -> dict:
        key = f"layers.{layer_idx}.{name}"
        if key not in self.tensors:
            raise KeyError(f"{key} not in the weight-cache manifest")
        return self.tensors[key]

    def device(self, layer_idx, name, mesh_device):
        e = self._entry(layer_idx, name)
        t = ttnn.load_tensor(os.path.join(self.cache_dir, e["path"]), device=mesh_device)
        expected = cache_dtype(name)
        if t.dtype != expected:
            raise ValueError(f"layers.{layer_idx}.{name}: cache dtype {t.dtype} != dtype map {expected}")
        return t

    def host(self, layer_idx, name):
        e = self._entry(layer_idx, name)
        t = ttnn.load_tensor(os.path.join(self.cache_dir, e["path"]))
        return ttnn.to_torch(t).reshape(e["shape"])

    def experts(self, layer_idx, name, num_experts, mesh_device):
        per_expert = []
        groups = []
        for e in range(num_experts):
            t = self.device(layer_idx, f"mlp.experts.{e}.{name}", mesh_device)
            per_expert.append(ttnn.reshape(t, (1, 1, t.shape[-2], t.shape[-1])))
            if len(per_expert) == 32 or e == num_experts - 1:
                groups.append(ttnn.concat(per_expert, dim=1, memory_config=DRAM))
                for p in per_expert:
                    ttnn.deallocate(p)
                per_expert = []
        stacked = ttnn.concat(groups, dim=1, memory_config=DRAM)
        for g in groups:
            ttnn.deallocate(g)
        return stacked


def reference_to_cache_layout(sd: Mapping[str, torch.Tensor], cfg) -> dict[str, torch.Tensor]:
    """Reference/HF post-conversion decoder-layer state dict -> tensors in the tile-cache layout (host, unquantized).

    Linear weights [out, in] -> [in, out]; fused conv [3C, 1, 4] -> per-q/k/v taps [4, C]; forget gate params moved
    under self_attn; hc_{attn,ffn}.{fn,base,scale} -> hc_{attn,ffn}_{fn,base,scale} (fn transposed to [16384, 24]);
    kv_b -> W_uk / W_uv; routed experts stay stacked (`mlp.experts.gate_up_proj` [E, H, 2I], `down_proj` [E, I, H]).
    """
    out: dict[str, torch.Tensor] = {}
    qkv = cfg.linear_num_heads * cfg.linear_head_dim
    for k, v in sd.items():
        if k.startswith("attn_hc.") or k.startswith("ffn_hc."):
            which = "attn" if k.startswith("attn_hc.") else "ffn"
            leaf = k.split(".")[-1]
            out[f"hc_{which}_{leaf}"] = v.t().contiguous() if leaf == "fn" else v
        elif k == "self_attn.conv1d.weight":
            taps = v.squeeze(1).t().contiguous()  # [4, 3C]
            out["self_attn.q_conv1d.weight"] = taps[:, :qkv].contiguous()
            out["self_attn.k_conv1d.weight"] = taps[:, qkv : 2 * qkv].contiguous()
            out["self_attn.v_conv1d.weight"] = taps[:, 2 * qkv :].contiguous()
        elif k.startswith("self_attn.forget_gate."):
            leaf = k[len("self_attn.forget_gate.") :]
            out[f"self_attn.{leaf}"] = v.t().contiguous() if leaf.endswith("proj.weight") else v
        elif k == "self_attn.kv_b_proj.weight":
            h = cfg.num_attention_heads
            w = v.view(h, cfg.qk_nope_head_dim + cfg.v_head_dim, cfg.kv_lora_rank)
            out["self_attn.kv_b_proj.W_uk"] = w[:, : cfg.qk_nope_head_dim, :].contiguous()
            out["self_attn.kv_b_proj.W_uv"] = w[:, cfg.qk_nope_head_dim :, :].mT.contiguous()
        elif k == "mlp.experts.gate_up_proj":
            out[k] = v.mT.contiguous()  # [E, 2I, H] -> [E, H, 2I] = [gate | up] columns
        elif k == "mlp.experts.down_proj":
            out[k] = v.mT.contiguous()  # [E, H, I] -> [E, I, H]
        elif v.dim() == 2 and k.endswith(".weight") and "conv" not in k:
            out[k] = v.t().contiguous()
        else:
            out[k] = v
    return out


class TorchStateDictSource(WeightSource):
    """Host state dicts per layer in the reference naming; converted to the cache layout/dtypes at load."""

    def __init__(self, state_dicts: Mapping[int, Mapping[str, torch.Tensor]], cfg):
        self.layers = {i: reference_to_cache_layout(sd, cfg) for i, sd in state_dicts.items()}

    def _get(self, layer_idx, name):
        if name not in self.layers[layer_idx]:
            raise KeyError(f"layer {layer_idx}: {name} missing from the provided state dict")
        return self.layers[layer_idx][name]

    def device(self, layer_idx, name, mesh_device):
        t = self._get(layer_idx, name)
        dtype = cache_dtype(name)
        return ttnn.from_torch(
            t.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )

    def host(self, layer_idx, name):
        return self._get(layer_idx, name)

    def experts(self, layer_idx, name, num_experts, mesh_device):
        t = self._get(layer_idx, f"mlp.experts.{name}")
        assert t.shape[0] == num_experts
        return ttnn.from_torch(
            t.unsqueeze(0).to(torch.bfloat16),
            dtype=WEIGHT_DTYPES["routed_expert"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
        )
