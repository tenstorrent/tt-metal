# SPDX-License-Identifier: MIT
"""33-layer device hidden-NRMSE mini-check vs the CPU fp32 twin.

Go/no-go gate for the bf16 smoke rerun after the fp32-residual-stream fix:
runs the fixed TT backend on the exact public smoke input (short case) and
compares hidden/logits against the fp32 reference implementation
(reference_layers; matched the harness oracle to 2.7e-5,.
using the harness metric (per-row RMS error / RMS reference, max over rows).

Gate: hidden max-row NRMSE <= ~0.02 (sim prediction 1.94e-2, CPU bf16
attribution study fa9696c0) agrees with the simulation; device full-bf16
measured 5.40e-2 on the same input.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/work")
from tt.esm2.config import Esm2TTConfig
from tt.esm2.loader import load_canonical_weights
from tt.esm2.reference_layers import Esm2Model


def row_nrmse(actual: np.ndarray, expected: np.ndarray):
    axes = tuple(range(1, actual.ndim))
    err = np.sqrt(np.mean((actual - expected) ** 2, axis=axes)) / np.maximum(
        1e-12, np.sqrt(np.mean(expected**2, axis=axes))
    )
    return float(err.max()), float(err.mean())


def main():
    d = np.load("/input/inputs.npz")
    ids = d["short__input_ids"]
    am = d["short__attention_mask"]
    with open("/weights/config.json") as f:
        cfg = Esm2TTConfig.from_dict(json.load(f))
    w = load_canonical_weights("/weights", cfg)

    ref = Esm2Model(cfg, weights=w).eval()
    with torch.no_grad():
        logits_ref, hidden_ref = ref(torch.from_numpy(ids), torch.from_numpy(am))
    logits_ref, hidden_ref = logits_ref.numpy(), hidden_ref.numpy()
    print(f"twin done shapes logits={logits_ref.shape} hidden={hidden_ref.shape}", flush=True)

    from tt.esm2.ttnn_backend import TtnnEsm2

    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        tt = TtnnEsm2(cfg, w, device=device, precision="bf16").build()
        out = tt.forward(ids, am)
        out2 = tt.forward(ids, am)  # determinism spot check
        det = float(np.max(np.abs(out["logits"] - out2["logits"])))
        hm, ha = row_nrmse(out["hidden"], hidden_ref)
        lm, la = row_nrmse(out["logits"], logits_ref)
        print(f"cast_fn={getattr(tt._cast_fn, '__name__', None)}", flush=True)
        print(f"hidden  NRMSE max={hm:.4e} mean={ha:.4e}  (device pre-fix 5.40e-2, sim fix 1.94e-2)", flush=True)
        print(f"logits  NRMSE max={lm:.4e} mean={la:.4e}  (pre-fix 1.62e-2)", flush=True)
        print(f"determinism max|dlogits|={det:.1e}", flush=True)
        print("MINI_CHECK_" + ("AGREE" if hm <= 0.024 else "DISAGREE"), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
