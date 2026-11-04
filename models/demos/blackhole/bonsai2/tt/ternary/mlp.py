# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bonsai's MLP in its own ternary values for one-token decode: down(silu(gate(x')) * up(x'))', each projection a
ternary decode linear (decode_linear3.py) on the rotated activation (rotate.py), weights straight from the GGUF.

The device holds the dense model (for prefill) and the ternary MLPs only one at a time (34 GB of DRAM), so the
records are packed on the host at build (prepare_ternary_mlp) and swapped in for the dense gate/up/down after the
prefill (swap_in_ternary_mlp)."""

import os
import time
from pathlib import Path

import numpy as np

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3, pack_records_inline
from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import BonsaiGGUF
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation

STEMS = ("ffn_gate", "ffn_up", "ffn_down")


class TernaryMLP:
    def __init__(self, device, packed: dict, rotations: dict):
        def lin(stem):
            rec, ppr, K, N = packed[stem]
            return TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N)

        self.gate, self.up, self.down = (lin(s) for s in STEMS)
        self.rot_in, self.rot_hidden = rotations[self.gate.K], rotations[self.down.K]

    def forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [..., dim] bf16 TILE, one token (one tile row) -> the same shape."""
        lead = tuple(list(x.shape)[:-1])
        xr = self.rot_in(x)
        g = self.gate(xr, lead=lead)
        u = self.up(xr, lead=lead)
        ttnn.deallocate(xr)
        h = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        hr = self.rot_hidden(h)
        ttnn.deallocate(h)
        y = self.down(hr, lead=lead)
        ttnn.deallocate(hr)
        return y


def prepare_ternary_mlp(model, gguf_path) -> None:
    """Pack every layer's gate/up/down records on the host (nothing on the device yet)."""
    t0 = time.time()
    gguf = BonsaiGGUF(gguf_path)
    packed = {}
    cache = Path(os.environ.get("BONSAI_TERNARY_CACHE", "/local/ttuser/jerrywang/bonsai/ternary_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    for layer in model.layers:
        per = {}
        for stem in STEMS:
            f = cache / f"blk.{layer.layer_num}.{stem}.decode_inline.npz"
            if f.exists():
                z = np.load(f)
                per[stem] = (z["rec"], int(z["ppr"]), int(z["K"]), int(z["N"]))
                continue
            t, s = gguf.ternary(f"blk.{layer.layer_num}.{stem}.weight")
            rec, ppr = pack_records_inline(t, s)
            np.savez(f, rec=rec, ppr=ppr, K=t.shape[0], N=t.shape[1])
            per[stem] = (rec, ppr, t.shape[0], t.shape[1])
        packed[layer.layer_num] = per
    model._ternary_mlp = (gguf.signs, packed)
    print(f"BONSAI_TERNARY_MLP_PACK_S={time.time() - t0:.1f} layers={len(model.layers)}", flush=True)


def swap_in_ternary_mlp(model) -> None:
    """Free the dense gate/up/down and give every layer's MLP its TernaryMLP (Qwen36MLP.forward uses it for
    one-chip decode). Prefill after this would need the dense weights back."""
    if getattr(model, "_ternary_mlp", None) is None:
        return
    t0 = time.time()
    signs, packed = model._ternary_mlp
    device = model.mesh_device
    for layer in model.layers:
        w = layer.feed_forward.weights
        for t in (w.w1, w.w2, w.w3):
            ttnn.deallocate(t)
    rotations = {width: DecodeRotation(device, s) for width, s in signs.items()}
    for layer in model.layers:
        layer.feed_forward.ternary = TernaryMLP(device, packed[layer.layer_num], rotations)
    model._ternary_mlp = None
    if os.environ.get("BONSAI_TERNARY_GDN") == "1":
        from models.demos.blackhole.bonsai2.tt.ternary.gdn import swap_in_ternary_gdn
        from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import BonsaiGGUF

        swap_in_ternary_gdn(
            model,
            BonsaiGGUF(
                os.environ.get("BONSAI_GGUF", "/local/ttuser/jerrywang/bonsai/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
            ),
            os.environ.get("BONSAI_TERNARY_CACHE", "/local/ttuser/jerrywang/bonsai/ternary_cache"),
        )
    print(f"BONSAI_TERNARY_MLP_SWAP_S={time.time() - t0:.1f}", flush=True)
