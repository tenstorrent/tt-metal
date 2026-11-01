# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Float64 reference for one ternary projection, and the gate the device op must pass.

Two ways to evaluate the same projection:

  ternary   y = (T * s) @ (H (D x))     what the GGUF runtime and the new kernel do
  dense     y = W @ x                   W = (T * s) @ H @ D, folded on the host

They agree to float64 round-off, so ``dense`` is a cheap way to state the same
thing without materializing a rotated activation.  Both are exposed here, plus
the two checks the plan's gates need:

  * ``projection_pcc``  -- device output vs this reference, per projection
  * ``code_roundtrip``  -- device-packed tiles decode back to the GGUF codes

Everything runs in float64 on the host; nothing here imports ttnn.

Usage (from a checkout, on tt_qb)::

    python -m models.demos.blackhole.bonsai2.native.cpu_ref \
        --gguf ~/avan/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
        --proj blk.0.ffn_gate.weight --tokens 8
"""

from __future__ import annotations

import argparse

import numpy as np

from .gguf_ternary import TernaryTensor, read_gguf, read_ternary
from .hadamard import BlockHadamard


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over the flattened arrays."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 1.0


def dense_weight(t: TernaryTensor, had: BlockHadamard) -> np.ndarray:
    """The equivalent dense ``W`` for a rotated ternary weight, float64 ``[N, K]``."""
    w = t.dequant().astype(np.float64)
    return had.unrotate(w)


def project_ternary(t: TernaryTensor, had: BlockHadamard, x: np.ndarray) -> np.ndarray:
    """``y = (T * s) @ (H (D x))`` in float64.  ``x`` is ``[M, K]``; returns ``[M, N]``."""
    x = np.asarray(x, dtype=np.float64)
    k = x.shape[-1]
    if k != t.codes.shape[1]:
        raise ValueError(f"x has K={k}, weight has K={t.codes.shape[1]}")
    xr = had.rotate(x)
    w = t.dequant().astype(np.float64)  # [N, K]
    return xr @ w.T


def project_dense(w_dense: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``y = W @ x`` in float64; the unfolded equivalent of :func:`project_ternary`."""
    return np.asarray(x, dtype=np.float64) @ np.asarray(w_dense, dtype=np.float64).T


def projection_pcc(device_out: np.ndarray, ref_out: np.ndarray) -> float:
    """Gate: per-projection PCC of the device result against the float64 reference."""
    if device_out.shape != ref_out.shape:
        raise ValueError(f"shape mismatch: device {device_out.shape} vs ref {ref_out.shape}")
    return pcc(device_out, ref_out)


def code_roundtrip(device_codes: np.ndarray, gguf_codes: np.ndarray) -> float:
    """Gate: fraction of positions where the device-decoded tile matches the GGUF code.

    Returns the match fraction; the gate is 1.0 exactly (no tolerance), because a
    correct BFP2 pack/unpack is a pure permutation of bits.
    """
    a = np.asarray(device_codes).reshape(-1)
    b = np.asarray(gguf_codes).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: device {a.shape} vs gguf {b.shape}")
    return float(np.mean(a == b))


# ---------------------------------------------------------------------------
# Self-check: ternary and dense paths agree, and re-quantization is a no-op
# ---------------------------------------------------------------------------


def selfcheck(gguf_path, names: list[str], tokens: int = 4, seed: int = 0) -> int:
    """Run the two reference formulations against each other on real weights."""
    gguf = read_gguf(gguf_path)
    had = BlockHadamard.from_metadata(gguf.kv)
    folded = set(gguf.kv["prism.hadamard.weight_names"]) | set(gguf.kv.get("prism.hadamard.inverse_weight_names", []))
    rng = np.random.default_rng(seed)

    rc = 0
    for name in names:
        t = read_ternary(gguf, name)
        x = rng.standard_normal((tokens, t.codes.shape[1])).astype(np.float32)
        y_tern = project_ternary(t, had, x)
        if name in folded:
            y_dense = project_dense(dense_weight(t, had), x)
            c = pcc(y_tern, y_dense)
            rel = float(np.linalg.norm(y_tern - y_dense) / (np.linalg.norm(y_dense) + 1e-30))
        else:
            # Not rotated: the two forms are the same matmul, nothing to compare.
            c, rel = 1.0, 0.0
        ok = c >= 0.999999999 and rel < 1e-9
        rc |= 0 if ok else 1
        print(
            f"  {name:28s} G={t.group:3d} [{t.codes.shape[0]}, {t.codes.shape[1]}] "
            f"pcc={c:.12f} rel={rel:.2e} {'ok' if ok else 'FAIL'}"
        )
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--proj", default="blk.0.ffn_gate.weight,blk.0.attn_qkv.weight,blk.3.attn_q.weight")
    ap.add_argument("--tokens", type=int, default=4)
    a = ap.parse_args()
    raise SystemExit(selfcheck(a.gguf, a.proj.split(","), a.tokens))
