# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bonsai's MLP in its own ternary values for one-token decode: down(silu(gate(x')) * up(x'))', each projection a
ternary decode linear (decode_linear3.py) on the rotated activation (rotate.py), weights straight from the GGUF.

The device holds the dense model (for prefill) and the ternary MLPs only one at a time (34 GB of DRAM), so the
records are packed on the host at build (prepare_ternary_mlp) and swapped in for the dense gate/up/down after the
prefill (swap_in_ternary_mlp). With BONSAI_TERNARY_PREFILL=1 the swap happens at build and the prefill runs on the same
records (forward_prefill, bonsai2/tt/tprefill): the dense gate/up/down are never used."""

import os
import time
from pathlib import Path

import numpy as np

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary import prerot
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import (
    TernaryDecodeLinear3,
    pack_glu_records,
    pack_records_inline,
)
from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import BonsaiGGUF
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation

STEMS = ("ffn_gate", "ffn_up", "ffn_down")
ROT_GLU = os.environ.get("BONSAI_ROT_GLU", "1") == "1"


PREFILL = os.environ.get("BONSAI_TERNARY_PREFILL") == "1"


def _take_pf():
    """Under the weight stream, the next GDN step's pending input prefetch: the down projection's writer runs it."""
    from models.demos.blackhole.bonsai2.tt.stream import decode as stream_decode

    return stream_decode.STATE.take_pending() if stream_decode.active() else None


class TernaryMLP:
    def __init__(self, device, packed: dict, rotations: dict, prefill_rotations: dict = None):
        def lin(stem):
            rec, ppr, K, N = packed[stem]
            return TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N)

        self.gate, self.up, self.down = (lin(s) for s in STEMS)
        self.rot_in, self.rot_hidden = rotations[self.gate.K], rotations[self.down.K]
        self.gateup = None
        if os.environ.get("BONSAI_GLU", "1") == "1":
            # gate and up in one ternary pass that also applies silu(gate) * up
            (rg, ppr, K, N), (ru, _, _, _) = packed["ffn_gate"], packed["ffn_up"]
            self._gu_host = pack_glu_records(rg, ru, ppr)
            self.gateup = TernaryDecodeLinear3.from_packed(device, self._gu_host, ppr, K, 2 * N)
            self.gateup.glu = True
            for t in (self.gate.w, self.up.w):
                ttnn.deallocate(t)
        self.pre_gateup = self.pre_down = None
        if prefill_rotations is not None:
            # exact ternary prefill on the same records (bonsai2/tt/tprefill)
            from models.demos.blackhole.bonsai2.tt.tprefill.linear import prefill_linear

            assert self.gateup is not None, "the exact prefill MLP uses the (gate, up) pair records"
            self.pre_gateup = prefill_linear(self.gateup, self._gu_host)
            self.pre_down = prefill_linear(self.down, packed["ffn_down"][0])
            self.prot_in, self.prot_hidden = prefill_rotations[self.gate.K], prefill_rotations[self.down.K]
        self._gu_host = None

    def forward_prefill(self, x: ttnn.Tensor, residual=None) -> ttnn.Tensor:
        """x [..., T, dim] bf16 TILE, T a multiple of 32 -> the same shape: down(up * silu(gate)) in the exact ternary
        weights (+ residual, added in the down projection's output stage)."""
        xr = prerot.take(x)  # the layer's fused add + norm + rotation (tprefill/norm_rot.py), else rotate here
        if xr is None:
            xr = self.prot_in(x)
        h = self.pre_gateup(xr)
        ttnn.deallocate(xr)
        hr = self.prot_hidden(h)
        ttnn.deallocate(h)
        y = self.pre_down(hr, residual=residual) if residual is not None else self.pre_down(hr)
        ttnn.deallocate(hr)
        return y

    def forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x [..., dim] bf16 TILE, one token (one tile row) -> the same shape."""
        lead = tuple(list(x.shape)[:-1])
        nrot = prerot.take_norm(x)
        if nrot is not None and self.gateup is not None:
            # the layer's norm + rotation inside the gate/up op (x is the un-normalised residual stream)
            h = self.gateup.rotated(nrot, x, lead=lead, out_l1=True)
            y = self.down.rotated(self.rot_hidden, h, lead=lead, residual=prerot.take_residual(), host_pf=_take_pf())
            ttnn.deallocate(h)
            return y
        xr = prerot.take(x)
        if xr is None:
            xr = self.rot_in(x)
        if self.gateup is not None:
            fuse = os.environ.get("BONSAI_ROT_FUSE", "1") == "1"
            h = self.gateup(xr, lead=lead, out_l1=fuse)
            ttnn.deallocate(xr)
            if fuse:  # the hidden rotation inside the down projection's op (decode_linear3 rotated())
                y = self.down.rotated(
                    self.rot_hidden, h, lead=lead, residual=prerot.take_residual(), host_pf=_take_pf()
                )
                ttnn.deallocate(h)
                return y
            hr = self.rot_hidden(h)
            ttnn.deallocate(h)
            y = self.down(hr, lead=lead, residual=prerot.take_residual())  # the layer's residual add, if registered
            ttnn.deallocate(hr)
            return y
        g = self.gate(xr, lead=lead)
        u = self.up(xr, lead=lead)
        ttnn.deallocate(xr)
        if ROT_GLU:
            hr = self.rot_hidden(g, u=u)  # rotates silu(g) * u in one op
        else:
            h = ttnn.mul(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            hr = self.rot_hidden(h)
            ttnn.deallocate(h)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
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


def swap_in_ternary_mlp(model, mlp_only=False) -> None:
    """Free the dense gate/up/down and give every layer's MLP its TernaryMLP (Qwen36MLP.forward uses it for
    one-token decode, and for prefill with BONSAI_TERNARY_PREFILL=1); then (unless mlp_only) the other decode swaps.
    BONSAI_TERNARY_PREFILL=1 calls it with mlp_only=True at build, before the prefill, and again after it."""
    t0 = time.time()
    device = model.mesh_device
    if getattr(model, "_ternary_mlp", None) is not None:
        signs, packed = model._ternary_mlp
        for layer in model.layers:
            w = layer.feed_forward.weights
            for t in (w.w1, w.w2, w.w3):
                ttnn.deallocate(t)
        rotations = {width: DecodeRotation(device, s) for width, s in signs.items()}
        prefill_rotations = None
        if PREFILL:
            from models.demos.blackhole.bonsai2.tt.tprefill.rotate import PrefillRotation

            prefill_rotations = {width: PrefillRotation(device, s) for width, s in signs.items()}
        for layer in model.layers:
            layer.feed_forward.ternary = TernaryMLP(device, packed[layer.layer_num], rotations, prefill_rotations)
        model._ternary_mlp = None
        model._ternary_signs, model._ternary_rotations = signs, rotations
        model._prefill_rotations = prefill_rotations
        if PREFILL and os.environ.get("BONSAI_PREFILL_NROT", "1") == "1":
            # prefill norms fused with the residual add and the 5120-wide rotation (tprefill/norm_rot.py)
            from models.demos.blackhole.bonsai2.tt.tprefill.norm_rot import PrefillNormRot

            for layer in model.layers:
                for norm, attr in (
                    (layer.attention_norm, "_bonsai_pre_attn_nrot"),
                    (layer.ffn_norm, "_bonsai_pre_ff_nrot"),
                ):
                    w = ttnn.to_torch(norm.weight).float().reshape(-1)  # includes the unit offset
                    setattr(layer, attr, PrefillNormRot(device, signs[5120], w, norm.eps))
        print(f"BONSAI_TERNARY_MLP_SWAP_MLP_S={time.time() - t0:.1f}", flush=True)
    if mlp_only or not hasattr(model, "_ternary_signs") or getattr(model, "_ternary_rest_done", False):
        return
    model._ternary_rest_done = True
    signs, rotations = model._ternary_signs, model._ternary_rotations
    if os.environ.get("BONSAI_TERNARY_GDN") == "1":
        from models.demos.blackhole.bonsai2.tt.ternary.gdn import swap_in_ternary_gdn
        from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import BonsaiGGUF

        swap_in_ternary_gdn(
            model,
            BonsaiGGUF(
                os.environ.get("BONSAI_GGUF", "/local/ttuser/jerrywang/bonsai/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
            ),
            os.environ.get("BONSAI_TERNARY_CACHE", "/local/ttuser/jerrywang/bonsai/ternary_cache"),
            prefill_rotations=getattr(model, "_prefill_rotations", None),
        )
    if os.environ.get("BONSAI_TERNARY_ATTN") == "1":
        from models.demos.blackhole.bonsai2.tt.ternary.attention import swap_in_ternary_attention

        swap_in_ternary_attention(
            model,
            BonsaiGGUF(
                os.environ.get("BONSAI_GGUF", "/local/ttuser/jerrywang/bonsai/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
            ),
            rotations,
            os.environ.get("BONSAI_TERNARY_CACHE", "/local/ttuser/jerrywang/bonsai/ternary_cache"),
            prefill_rotations=getattr(model, "_prefill_rotations", None),
        )
    if os.environ.get("BONSAI_NORM_ROT", "1") == "1":
        # decode norms fused with the 5120-wide rotation (rotate.NormRotation): every layer, both norms (=0: off)
        from models.demos.blackhole.bonsai2.tt.ternary.rotate import NormRotation

        for layer in model.layers:
            for norm, attr in ((layer.attention_norm, "_bonsai_attn_nrot"), (layer.ffn_norm, "_bonsai_ff_nrot")):
                w = ttnn.to_torch(norm.weight).float().reshape(-1)
                setattr(layer, attr, NormRotation(device, signs[5120], w, norm.eps))
    if os.environ.get("BONSAI_TERNARY_HEAD") == "1":
        model.ternary_lm_head = TernaryLMHead(device, pack_lm_head(), rotations[5120])
    print(f"BONSAI_TERNARY_MLP_SWAP_S={time.time() - t0:.1f}", flush=True)


class TernaryLMHead:
    """The output projection (Bonsai's lm_head, ternary like every linear) for one-token decode."""

    def __init__(self, device, packed, rotation: DecodeRotation):
        rec, ppr, K, N = packed
        self.lin = TernaryDecodeLinear3.from_packed(device, rec, ppr, K, N)
        self.rot = rotation

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        lead = tuple(list(x.shape)[:-1])
        xr = self.rot(x)
        y = self.lin(xr, lead=lead)
        ttnn.deallocate(xr)
        return y


def pack_lm_head():
    cache = Path(os.environ.get("BONSAI_TERNARY_CACHE", "/local/ttuser/jerrywang/bonsai/ternary_cache"))
    f = cache / "output.decode_inline.npz"
    if f.exists():
        z = np.load(f)
        return z["rec"], int(z["ppr"]), int(z["K"]), int(z["N"])
    gguf = BonsaiGGUF(
        os.environ.get("BONSAI_GGUF", "/local/ttuser/jerrywang/bonsai/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
    )
    t, s = gguf.ternary("output.weight")
    rec, ppr = pack_records_inline(t, s)
    np.savez(f, rec=rec, ppr=ppr, K=t.shape[0], N=t.shape[1])
    return rec, ppr, t.shape[0], t.shape[1]
