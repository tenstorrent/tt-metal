# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash checkpoint reader for the prefill: model.py's own tensor names, dequantised on the host.

Checkpoint formats (``model.safetensors.index.json``):

* fp8 Linear weights (``attn.wq_a/wq_b/wkv/wo_a/wo_b``, ``ffn.shared_experts.w1/w2/w3``, ``attn.indexer.wq_b``,
  ``engram.wkv``): ``float8_e4m3fn`` ``[N, K]`` + an ``e8m0`` scale per 32 x 32 block (``fp8_block_size = 32``; V4-Flash
  used 128 x 128) -> ``w * scale`` (exact in fp32 / bf16, the scale is a power of two).
* fp4 routed experts (``ffn.experts.E.w1/w2/w3``): ``int8`` ``[N, K/2]``, two e2m1 values per byte (element ``2i`` is
  the LOW nibble) + an ``e8m0`` scale per 32 along K.
* everything else as stored (bf16 / fp32): norms, ``attn_sink``, ``hc_*``, ``ffn.gate.weight / bias``, the compressor's
  ``wkv / wgate`` (model.py promotes the ratio-2 ones to fp32), ``attn.indexer.wk / weights_proj``, ``embed``.

Nothing here imports ttnn: the module tests compare against ``inference/model.py`` on the host first.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import torch
from safetensors import safe_open

from .config import DEFAULT_MODEL_DIR

FP4_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    u = packed.contiguous().view(torch.uint8)
    return torch.stack([FP4_E2M1[(u & 0x0F).long()], FP4_E2M1[((u >> 4) & 0x0F).long()]], dim=-1).flatten(-2)


class V41Checkpoint:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.dir = model_dir
        self.weight_map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self._handles: dict = {}

    def _file(self, name: str):
        fn = self.weight_map[name]
        h = self._handles.get(fn)
        if h is None:
            h = self._handles[fn] = safe_open(os.path.join(self.dir, fn), "pt")
        return h

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def raw(self, name: str) -> torch.Tensor:
        return self._file(name).get_tensor(name)

    def linear(self, prefix: str) -> torch.Tensor:
        """``prefix.weight`` dequantised to fp32 ``[N, K]`` (fp8 + block scale, fp4 + per-32 scale, or as stored)."""
        w = self.raw(f"{prefix}.weight")
        if not self.has(f"{prefix}.scale"):
            return w.float()
        s = self.raw(f"{prefix}.scale").float()
        if w.dtype in (torch.int8, torch.uint8) or str(w.dtype).startswith("torch.float4"):
            v = unpack_fp4(w)
            return v * s.repeat_interleave(32, dim=1)[:, : v.shape[1]]
        bn, bk = -(-w.shape[0] // s.shape[0]), -(-w.shape[1] // s.shape[1])
        return w.float() * s.repeat_interleave(bn, 0)[: w.shape[0]].repeat_interleave(bk, 1)[:, : w.shape[1]]

    def layer(self, layer: int, *, experts: bool = False) -> dict:
        """Every tensor of ``layers.<layer>.`` (model.py names minus the prefix), Linear weights dequantised; the routed
        experts only with ``experts=True`` (``ffn.experts.E.w{1,2,3}`` -> fp32 ``[N, K]``)."""
        pre = f"layers.{layer}."
        out = {}
        for name in self.weight_map:
            if not name.startswith(pre) or name.endswith(".scale"):
                continue
            short = name[len(pre) :]
            if ".experts." in short and not experts:
                continue
            if short.endswith(".weight") and self.has(name[: -len(".weight")] + ".scale"):
                out[short] = self.linear(name[: -len(".weight")])
            else:
                out[short] = self.raw(name)
        return out

    def expert(self, layer: int, e: int) -> dict:
        p = f"layers.{layer}.ffn.experts.{e}"
        return {w: self.linear(f"{p}.{w}") for w in ("w1", "w2", "w3")}


@lru_cache(maxsize=2)
def checkpoint(model_dir: str = DEFAULT_MODEL_DIR) -> V41Checkpoint:
    return V41Checkpoint(model_dir)
