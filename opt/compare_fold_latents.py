"""Pairwise latent comparison for the LTX_FOLD_GATED_RESIDUAL 48-layer A/B.

Reads the LTX_DUMP_LATENTS files and reports PCC / CCC / RMSE-over-sigma per pair,
video and audio separately, via check.py::assert_quality. Bit-exactness is reported
separately: assert_quality's PCC prints 100.0000% well before bit-equality, so it
cannot by itself establish the run-to-run noise floor.
"""

import sys
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.tt_dit.utils.check import assert_quality  # noqa: E402

DUMPS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/smarton/tmp/ltx_fold48")
GEN = sys.argv[2] if len(sys.argv) > 2 else "gen1"

logger.remove()
logger.add(sys.stdout, format="{message}", level="INFO")


def load(name: str) -> dict:
    return torch.load(DUMPS / f"{name}.{GEN}.pt", map_location="cpu")


def bitexact(a: torch.Tensor, b: torch.Tensor) -> str:
    n = int((a != b).sum())
    if n == 0:
        return "BIT-EXACT (0 of %d elements differ)" % a.numel()
    return (
        f"NOT bit-exact: {n}/{a.numel()} ({100 * n / a.numel():.2f}%) elems differ, "
        f"max|Δ| = {(a - b).abs().max().item():.4g}"
    )


def compare(label: str, name_a: str, name_b: str) -> None:
    a, b = load(name_a), load(name_b)
    print(f"\n{'=' * 78}\n{label}\n  {name_b}.{GEN} {b.get('flags')}\n  {name_a}.{GEN} {a.get('flags')}\n{'=' * 78}")
    for key in ("video", "audio"):
        ta, tb = a[key], b[key]
        print(f"-- {key}: shape {tuple(ta.shape)}")
        print(f"   {bitexact(ta, tb)}")
        print("   ", end="")
        assert_quality(ta, tb)


PAIRS = [
    ("NOISE FLOOR       A' vs A   (fold=0 twice, same seed, SEPARATE RUNS)", "Aprime", "A"),
    ("FOLD @ 48 LAYERS  B  vs A   (fold=1 vs fold=0)  <-- the answer", "B", "A"),
    ("ROUNDING CONTROL  C  vs A   (addcmul-split vs fold=0)  <-- the yardstick", "C", "A"),
    ("cross-check       C  vs B   (control vs fold)", "C", "B"),
]

for label, x, y in PAIRS:
    compare(label, x, y)

print(f"\n{'#' * 78}\n# WITHIN-PROCESS: gen0 (trace capture) vs gen1 (steady-state replay)\n{'#' * 78}")
for name in ("A", "Aprime", "B", "C"):
    g0 = torch.load(DUMPS / f"{name}.gen0.pt", map_location="cpu")
    g1 = torch.load(DUMPS / f"{name}.gen1.pt", map_location="cpu")
    print(f"-- {name}: video {bitexact(g0['video'], g1['video'])} | audio {bitexact(g0['audio'], g1['audio'])}")
