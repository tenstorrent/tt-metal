# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify ``gguf_ternary`` against the dense HF checkpoint converted from the same GGUF.

The dense checkpoint in ``~/avan/models/bonsai2-dense`` was produced by the PR's
``convert_gguf_to_hf.py`` from ``Ternary-Bonsai-2-27B-PQ2_0.gguf``: it dequantizes
``W'``, unfolds the Hadamard rotation and applies the llama.cpp -> HF row fixups.

So: read a tensor with ``gguf_ternary``, unrotate the same way, and the result must
match the stored bf16 tensor.  Any mismatch in the 2-bit decode, the scale
placement, the Hadamard block order or the sign vector shows up as a cosine well
below 1.  This is the harness's self-check on the reader, independent of the
converter's own code paths (which this does not import).

Usage::

    python check_reader.py --gguf FILE.gguf --dense DIR [--tensors a,b,c]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:  # importable as a package module
    from . import gguf_ternary as gt
    from .hadamard import BlockHadamard
except ImportError:  # run as a script from this directory
    import gguf_ternary as gt

    from hadamard import BlockHadamard


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


class Dense:
    """Reader for the converted bf16 HF checkpoint."""

    def __init__(self, d):
        from safetensors import safe_open

        self.d = Path(d)
        self.wm = json.load(open(self.d / "model.safetensors.index.json"))["weight_map"]
        self.h: dict[str, object] = {}
        self._open = safe_open

    def get(self, k) -> np.ndarray:
        fn = self.wm[k]
        if fn not in self.h:
            self.h[fn] = self._open(str(self.d / fn), "pt")
        return self.h[fn].get_tensor(k).float().numpy()


# GGUF stem -> HF suffix, mirroring convert_gguf_to_hf.py's MAP/GLOBAL.
_STEM = {
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}
_GLOBAL = {
    "output.weight": "lm_head.weight",
    "token_embd.weight": "model.language_model.embed_tokens.weight",
}


def hf_name(gguf_name: str) -> str:
    if gguf_name in _GLOBAL:
        return _GLOBAL[gguf_name]
    _, layer, stem = gguf_name.split(".", 2)
    return f"model.language_model.layers.{layer}.{_STEM[stem]}"


def row_order(gguf_name: str, kv: dict, num_rows: int) -> np.ndarray | None:
    """GGUF -> HF row permutation for a tensor, or ``None`` for the identity.

    llama.cpp tiles the GDN value heads as ``[nv // nk, nk, unit]`` where HF
    wants ``[nk, nv // nk, unit]``.  The converter applies this only when the
    two head counts differ, and only to the tensors that carry v-head rows:
    ``attn_qkv`` (after the q/k prefix) and ``attn_gate``.  Scales move with
    their rows, so this applies to the code rows, not to the activations.
    """
    nv = int(kv["qwen35.ssm.time_step_rank"])
    nk = int(kv["qwen35.ssm.group_count"])
    if nv == nk or not gguf_name.startswith("blk."):
        return None
    stem = gguf_name.split(".", 2)[2]  # e.g. "attn_qkv.weight"
    hd = int(kv["qwen35.ssm.inner_size"]) // nv
    hk = int(kv["qwen35.ssm.state_size"])
    idx = gt.v_head_index(nv, nk, hd)
    if stem == "attn_qkv.weight":
        qk = 2 * nk * hk
        return np.concatenate([np.arange(qk), qk + idx])
    if stem == "attn_gate.weight":
        return gt.v_head_index(nv, nk, hd)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dense", required=True)
    ap.add_argument(
        "--tensors",
        default="blk.0.ffn_gate.weight,blk.0.attn_qkv.weight,blk.0.ssm_out.weight,blk.3.attn_q.weight,blk.3.attn_output.weight",
    )
    ap.add_argument("--tol", type=float, default=0.999)
    args = ap.parse_args()

    gguf = gt.read_gguf(args.gguf)
    had = BlockHadamard.from_metadata(gguf.kv)
    dense = Dense(args.dense)
    folded = set(gguf.kv["prism.hadamard.weight_names"]) | set(gguf.kv.get("prism.hadamard.inverse_weight_names", []))

    print(f"{args.gguf}")
    print(f"  group={had.block} sign widths={list(had.signs)}")

    rc = 0
    for name in args.tensors.split(","):
        t = gt.read_ternary(gguf, name)
        w = t.dequant()
        if name in folded:
            w = had.unrotate(w)
        order = row_order(name, gguf.kv, w.shape[0])
        if order is not None:
            w = w[order]
        ref = dense.get(hf_name(name))
        c = cosine(w, ref)
        ok = c >= args.tol
        rc |= 0 if ok else 1
        print(
            f"  {name:28s} type={t.ggml_type:3d} G={t.group:3d} shape={t.codes.shape} "
            f"{'perm' if order is not None else '    '} cos={c:.7f} {'ok' if ok else 'FAIL'}"
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
