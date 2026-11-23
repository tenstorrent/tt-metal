# Layout of the published GPTQ package adapted from Lottolabs' Inkling-2-Mini P150 runtime
# (runtime/inkling2/tt/moe_layout.py, https://huggingface.co/Lottolabs/Inkling-2-Mini-TT-Mixed-BFP4-BFP8-P150),
# Copyright Lottolabs, Licensed under the Apache License, Version 2.0.
# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net> (changes: inverse layout, store for our converter)
# SPDX-License-Identifier: Apache-2.0

"""GPTQ-calibrated BFP4 expert values as a ``store`` for ``convert.convert_experts`` (host only, opt-in).

Two sources, both giving per layer HF-orientation tensors ``gate_up_proj`` [E, 2I, H] (gate rows first) and
``down_proj`` [E, H, I] whose values are BFP4-exact in the cache's W^T tile orientation (one exponent per 16
consecutive outputs at one input), so ``from_torch(bfloat4_b)`` stores them bit for bit:

``lotto:DIR``  the ``native/gptq`` package of Lottolabs' repo, read as data only (``ttnn.load_tensor``, never their
    code). Expert tensors there are 8-bank WIDTH_SHARDED W^T tile streams (``k_aligned=False``, 3 down columns):
      gate_up [E*H, 2I]: bank b holds, per expert e, pairs j in [start(e, b), start(e, b) + pairs(e, b)) (36 pairs
        over 8 banks, 5 / 4 by the parity of e + b); a pair is K tiles of (gate, up); stream e, j, k, {gate, up}.
      down [E*I, H]: bank b owns output tiles [9b, 9b + 9) in 3 groups of 3; stream e, group, k, column.
      Tile L of a bank stream sits at shard tile (L // 9, L % 9).
    NVFP4 layers 0-19 hold the factored values (code * block scale): the per-expert factors f13 = f2 = amax / 2688
    (``factors`` [f13 | f2]) multiply the gate/up accumulators and the routing weight in their kernels. Our cache
    folds them into the weights, which moves every value off the BFP4 grid: layers 0-19 are re-rounded (RTN of the
    GPTQ values), so only the MXFP8 layers 20-23 (factor 1) map exactly. ``--exact-only`` refuses 0-19.
``codes:DIR``  our own calibration (convert/gptq_calib.py, ``L{nn}.pt``: ``gate_up`` / ``down`` dicts of ``q4`` uint8
    [E, out, in / 2], two 4-bit two's complement mantissas -7..7 per byte, low nibble = even input, and uint8 ``e``
    [E, out / 16, in], value ``q * 2^(e - 129)``; or ``q8`` int8 [E, out, in] mantissas -127..127, value
    ``q * 2^(e - 133)``, stored as BFP8), already in the baked domain: every layer exact.
"""

from __future__ import annotations

from pathlib import Path

import torch

BANKS = 8
TILE = 32
DOWN_COLS = 3
E, H, I = 128, 2304, 1152
TAG = "moe.nf.b8c3"


def _gu_pairs(e: int, b: int, it: int) -> int:
    base, odd = it // BANKS, int(it % BANKS != 0)
    return base + (odd and (e + b) % 2 == 0)


def _gu_start(e: int, b: int, it: int) -> int:
    base, odd = it // BANKS, int(it % BANKS != 0)
    return b * base + odd * ((b + (1 if e % 2 == 0 else 0)) // 2)


def _bank_streams(t: torch.Tensor, width: int) -> list[torch.Tensor]:
    """Stored [R, BANKS * width * 32] -> per bank its tile stream [L, 32, 32] (tile L at shard (L // w, L % w))."""
    out = []
    for b in range(BANKS):
        s = t[:, b * width * TILE : (b + 1) * width * TILE]
        out.append(s.reshape(-1, TILE, width, TILE).permute(0, 2, 1, 3).reshape(-1, TILE, TILE))
    return out


def unpack_gate_up(t: torch.Tensor) -> torch.Tensor:
    """Stored gate_up [E*H, 2I] -> W^T [E, H, 2I] (gate columns first)."""
    it, kt = I // TILE, H // TILE
    out = torch.empty(E, it, kt, 2, TILE, TILE, dtype=t.dtype)  # e, j, k, {g, u}, 32k, 32n
    for b, s in enumerate(_bank_streams(t, 2 * it // BANKS)):
        pos = 0
        for e in range(E):
            j0, n = _gu_start(e, b, it), _gu_pairs(e, b, it)
            out[e, j0 : j0 + n] = s[pos : pos + n * kt * 2].reshape(n, kt, 2, TILE, TILE)
            pos += n * kt * 2
        if pos != s.shape[0]:
            raise ValueError(f"gate_up bank {b}: {s.shape[0]} tiles stored, layout uses {pos}")
    # [e, j, k, g, 32k, 32n] -> [e, k*32, g, j*32] -> [E, H, 2I]
    return out.permute(0, 2, 4, 3, 1, 5).reshape(E, H, 2 * I)


def unpack_down(t: torch.Tensor) -> torch.Tensor:
    """Stored down [E*I, H] -> W^T [E, I, H]."""
    nt, kt = H // TILE, I // TILE
    per_bank = nt // BANKS
    groups = per_bank // DOWN_COLS
    out = torch.empty(E, BANKS, groups, kt, DOWN_COLS, TILE, TILE, dtype=t.dtype)  # e, b, g, k, c, 32k, 32n
    for b, s in enumerate(_bank_streams(t, per_bank)):
        out[:, b] = s.reshape(E, groups, kt, DOWN_COLS, TILE, TILE)
    # n tile = b * 9 + g * 3 + c -> [e, k*32, n*32]
    return out.permute(0, 3, 5, 1, 2, 4, 6).reshape(E, I, H)


def _load(path: Path) -> torch.Tensor:
    import ttnn

    return ttnn.to_torch(ttnn.load_tensor(str(path))).to(torch.bfloat16)


def lotto_layer(pkg: Path, layer: int) -> dict:
    """Package layer -> {"gate_up_t": W^T [E, H, 2I], "down_t": W^T [E, I, H], "f13", "f2"} (factored values)."""
    d = Path(pkg) / "tensors"
    f = _load(d / f"L{layer}.{TAG}.factors.fp32.rm.tensorbin").float().reshape(-1)
    if f.numel() != 2 * E:
        raise ValueError(f"L{layer}: factors {f.numel()} values, want {2 * E}")
    gu = _load(d / f"L{layer}.{TAG}.gate_up.bfp4.tile.tensorbin")
    if tuple(gu.shape[-2:]) != (E * H, 2 * I):
        raise ValueError(f"L{layer}: gate_up shape {tuple(gu.shape)}")
    gate_up_t = unpack_gate_up(gu.reshape(E * H, 2 * I))
    del gu
    dn = _load(d / f"L{layer}.{TAG}.down.bfp4.tile.tensorbin")
    if tuple(dn.shape[-2:]) != (E * I, H):
        raise ValueError(f"L{layer}: down shape {tuple(dn.shape)}")
    down_t = unpack_down(dn.reshape(E * I, H))
    return {"gate_up_t": gate_up_t, "down_t": down_t, "f13": f[:E], "f2": f[E:]}


def unpack4(b: torch.Tensor) -> torch.Tensor:
    """uint8 [..., C / 2] -> int8 [..., C] (inverse of gptq_calib.pack4)."""
    n = torch.stack([b & 0xF, b >> 4], dim=-1).flatten(-2).to(torch.int8)
    return n - 16 * (n >= 8).to(torch.int8)


def decode_codes(q: torch.Tensor, e: torch.Tensor, mbits: int = 3) -> torch.Tensor:
    """int8 q [..., R, C], uint8 e [..., R / 16, C] -> float32 q * 2^(e - 126 - mbits) (BFP4: 3, BFP8: 7)."""
    return torch.ldexp(q.float(), e.to(torch.int32).repeat_interleave(16, dim=-2) - 126 - mbits)


def _code_values(z: dict) -> torch.Tensor:
    """One tensor of a ``codes:`` layer file: ``q4`` (BFP4) or ``q8`` (int8 mantissas -127..127, BFP8)."""
    if "q8" in z:
        return decode_codes(z["q8"], z["e"], 7)
    return decode_codes(unpack4(z["q4"]), z["e"])


def layer_dtypes(source: str, layer: int) -> dict[str, str]:
    """Stored dtype of each expert tensor for ``layer``: BFP8 for ``q8`` codes, else BFP4."""
    kind, _, root = source.partition(":")
    if kind != "codes":
        return {"experts.w_gate_up": "BFP4", "experts.w_down": "BFP4"}
    z = torch.load(Path(root) / f"L{layer:02d}.pt", weights_only=True, mmap=True)
    return {
        n: ("BFP8" if "q8" in z[k] else "BFP4")
        for n, k in (("experts.w_gate_up", "gate_up"), ("experts.w_down", "down"))
    }


class Store:
    """``convert_experts`` store over one layer: ``experts(key, s, e)`` -> bf16 [e - s, out, in]."""

    def __init__(self, root: str, layer: int, gate_up: torch.Tensor, down: torch.Tensor):
        self.root = root
        p = f"model.language_model.layers.{layer}.mlp.experts."
        self.t = {p + "gate_up_proj": gate_up, p + "down_proj": down}

    def experts(self, key: str, start: int, end: int) -> torch.Tensor:
        return self.t[key][start:end]


def layer_store(source: str, layer: int, exact_only: bool = False) -> tuple[Store, bool]:
    """(store, exact) for ``lotto:DIR`` or ``codes:DIR``; exact = every value lands on the BFP4 grid unchanged."""
    kind, _, root = source.partition(":")
    if kind == "lotto":
        z = lotto_layer(Path(root), layer)
        exact = bool((z["f13"] == 1).all() and (z["f2"] == 1).all())
        if exact_only and not exact:
            raise ValueError(f"L{layer}: package values are factored (NVFP4 layer); not BFP4-exact once folded")
        # fold the per-expert factors in fp32 the way our dequant does (w = values * amax / 2688), then bf16
        # to HF orientation [E, out, in] (convert_experts transposes back to W^T)
        gu = (z["gate_up_t"].float() * z["f13"][:, None, None]).to(torch.bfloat16).transpose(1, 2)
        dn = (z["down_t"].float() * z["f2"][:, None, None]).to(torch.bfloat16).transpose(1, 2)
    elif kind == "codes":
        z = torch.load(Path(root) / f"L{layer:02d}.pt", weights_only=True)
        gu = _code_values(z["gate_up"]).to(torch.bfloat16)
        dn = _code_values(z["down"]).to(torch.bfloat16)
        exact = True
    else:
        raise ValueError(f"expert source {source!r}: want lotto:DIR or codes:DIR")
    return Store(source, layer, gu.contiguous(), dn.contiguous()), exact
