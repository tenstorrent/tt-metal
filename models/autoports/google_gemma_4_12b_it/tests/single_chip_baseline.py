# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run ``mc_cases`` scenarios on the single-chip ``OptimizedDecoder`` (1x1 mesh, no fabric) and store the outputs.

    python -m models.autoports.google_gemma_4_12b_it.tests.single_chip_baseline --cases lengths_L0,traced_L0_b1

Outputs go to ``mc_cases.BASELINE_DIR/<case>.pt`` with the hash of the code they ran on; a stored baseline whose
hash differs from the current code is recomputed. ``test_multichip_decoder.py`` calls this in a subprocess before
it opens the QB2 mesh.
"""

import argparse
import hashlib
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests import mc_cases
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import text_config

HERE = Path(__file__).resolve().parent
CODE = [
    HERE.parent / "tt" / "functional_decoder.py",
    HERE.parent / "tt" / "fused_decoder.py",
    HERE.parent / "tt" / "optimized_decoder.py",
    HERE / "mc_cases.py",
    HERE / "harness.py",
    HERE / "hf_reference.py",
]


def code_hash():
    h = hashlib.sha256()
    for p in CODE:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def baseline_path(case):
    return mc_cases.BASELINE_DIR / f"{case}.pt"


def load(case):
    """Stored single-chip outputs of ``case`` if they match the current code, else None."""
    p = baseline_path(case)
    if not p.exists():
        return None
    data = torch.load(p)
    return data["outputs"] if data.get("code") == code_hash() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = text_config()
    reg = mc_cases.case_registry(cfg)
    todo = [c for c in a.cases.split(",") if c and (a.force or load(c) is None)]
    if not todo:
        print("BASELINE all cached")
        return
    torch.set_num_threads(16)
    mc_cases.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    be = mc_cases.SingleChip(dev)
    try:
        for case in todo:
            t0 = time.time()
            out = reg[case](be, False)
            torch.save({"code": code_hash(), "outputs": out}, baseline_path(case))
            print(f"BASELINE {case}: {len(out)} outputs in {time.time() - t0:.1f} s", flush=True)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
