# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-op PCC / fingerprint logging for llama32_1b bring-up — the llama analog of resnet's
RESNET_PCC_LOG.

Entirely env-gated (zero overhead when off). Three cooperating pieces, mirroring the resnet scheme:

  * ``log_op(name, tensor)`` — called inline at each seam in the model. When ``LLAMA_PCC_LOG=1`` it reads
    the device tensor back, logs a ``[PCCLOG]`` fingerprint, and — if a golden with the same NAME was
    registered — a ``[GOLDENPCC]`` line with the exact device-vs-golden PCC. Returns the tensor untouched
    so call sites read ``x = log_op("name", x)``.
  * ``set_golden_intermediates(dict)`` — the test captures torch reference intermediates (via forward
    hooks on the HF model) keyed by the SAME names and installs them here before the device run.
  * ``LLAMA_PCC_DUMP=<dir>`` — optional; dumps each op's readback to ``<dir>/op<NNN>_<name>.pt`` so exact
    PCC can be computed offline / across arches (dump on WH, load+compare on Quasar).

Stable string NAMES (not running indices) are the join key: arch-gated ops shift indices between WH and
Quasar, so keying by name lets a WH log and a Quasar log be diffed op-for-op.
"""

from __future__ import annotations

import os
import re

from loguru import logger

_GOLDEN: dict = {}
_OP_IDX = 0

_DIVERGE_PCC = 0.98  # marker threshold for the "<<< DIVERGES" annotation (diagnostic only)


def is_enabled() -> bool:
    return os.environ.get("LLAMA_PCC_LOG") == "1"


def set_golden_intermediates(golden: dict) -> None:
    """Install the name->torch-tensor golden dict captured by the test (call before the device run)."""
    global _GOLDEN
    _GOLDEN = dict(golden) if golden else {}


def reset_op_log() -> None:
    """Reset the per-run op index (call at the start of each device forward)."""
    global _OP_IDX
    _OP_IDX = 0


def _pcc(dev, gold):
    """Inf/NaN/padding-robust Pearson correlation. Returns (pcc, finite_frac).

    Reshapes both to 2D, slices the device output down to the golden's logical [rows, cols] (dropping tile
    / head padding), masks non-finite and >1e30 garbage lanes, computes float64 correlation.
    """
    import torch

    d = dev.reshape(-1, dev.shape[-1]).float()
    g = gold.reshape(-1, gold.shape[-1]).float()
    rows = min(d.shape[0], g.shape[0])
    cols = min(d.shape[1], g.shape[1])
    d = d[:rows, :cols].reshape(-1).double()
    g = g[:rows, :cols].reshape(-1).double()
    finite = torch.isfinite(d) & torch.isfinite(g) & (d.abs() < 1e30) & (g.abs() < 1e30)
    finite_frac = float(finite.float().mean()) if finite.numel() else 0.0
    d = d[finite]
    g = g[finite]
    if d.numel() < 2:
        return 0.0, finite_frac
    d = d - d.mean()
    g = g - g.mean()
    denom = d.norm() * g.norm()
    if float(denom) == 0.0:
        # Both constant (e.g. all-zero) -> perfectly correlated iff equal.
        return (1.0 if torch.allclose(dev.reshape(-1)[:1], gold.reshape(-1)[:1]) else 0.0), finite_frac
    pcc = float((d * g).sum() / denom)
    return pcc, finite_frac


def _readback(t, mesh_device=None):
    """Best-effort device->torch readback that tolerates a mesh device."""
    import ttnn

    try:
        num = mesh_device.get_num_devices() if mesh_device is not None else 1
    except Exception:
        num = 1
    if num > 1:
        try:
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
        except Exception:
            pass
    return ttnn.to_torch(t).float()


def log_op(name, t, mesh_device=None):
    """Log a fingerprint (and golden PCC when available) for one device op. Returns ``t`` unchanged."""
    if not is_enabled():
        return t
    global _OP_IDX
    idx = _OP_IDX
    _OP_IDX += 1
    try:
        import torch

        tt = _readback(t, mesh_device)
        flat = tt.reshape(-1)
        n = flat.numel()
        nan = int(torch.isnan(flat).sum())
        inf = int(torch.isinf(flat).sum())
        finite = flat[torch.isfinite(flat)]
        stats = (
            f"mean={finite.mean().item():.4e} std={finite.std().item():.4e} "
            f"min={finite.min().item():.4e} max={finite.max().item():.4e} absmean={finite.abs().mean().item():.4e}"
            if finite.numel()
            else "mean=nan"
        )
        first8 = [round(float(x), 4) for x in flat[:8].tolist()]
        logger.info(
            f"[PCCLOG] op{idx:03d} {name} shape={tuple(tt.shape)} n={n} {stats} nan={nan} inf={inf} first8={first8}"
        )

        g = _GOLDEN.get(name)
        if g is not None:
            pcc, finite_frac = _pcc(tt, g)
            marker = "  <<< DIVERGES" if not (pcc >= _DIVERGE_PCC) else ""
            logger.info(f"[GOLDENPCC] op{idx:03d} {name} pcc={pcc:.6f} finite={finite_frac:.6f}{marker}")

        dump_dir = os.environ.get("LLAMA_PCC_DUMP")
        if dump_dir:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
            base = os.path.realpath(dump_dir)
            path = os.path.realpath(os.path.join(base, f"op{idx:03d}_{safe}.pt"))
            if path.startswith(base + os.sep):
                os.makedirs(base, exist_ok=True)
                torch.save(tt, path)
    except Exception as e:  # logging must never break the model run
        logger.warning(f"[PCCLOG] op{idx:03d} {name} readback failed: {e}")
    return t
