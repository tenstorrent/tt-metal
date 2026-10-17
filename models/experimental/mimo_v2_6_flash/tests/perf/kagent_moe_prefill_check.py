# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for MiMo MoE prefill changes (``TtMiMoMoE.forward``, one layer on Galaxy row 0, BFP4 experts).

1. Prefill on the golden prompts' real tokens at this layer (96) and on normalized random tokens (32, 1056 and
   4096: one tile row, an odd number of tile rows, a full 4K chunk) must match the base implementation's saved
   output on the same inputs: PCC >= 0.9995 overall and >= 0.999 on every token, so one dropped or mis-weighted
   expert of one token fails it, and every value must be finite.
2. A second call on the 4096 tokens must give a bitwise identical output (a race between cores shows up here).

Decode is not checked here (tests/test_moe_decode_tp8.py checks ``forward_decode`` against ``forward`` on the same
tokens), and prefill changes stay out of moe.py.

``--write-base`` records the base outputs; they live outside the repository. Prints
MOE_PREFILL_CORRECTNESS_PASS after every check passed.
"""

import argparse
import os

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common

BASE = os.environ.get("MIMO_MOE_PREFILL_BASE", os.path.join(common.GOLDEN_DIR, "moe_prefill_layer1_bfp4_base_v2.pt"))
RANDOM_TOKENS = (32, 1056, 4096)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def row_pcc(a, b):
    """Smallest PCC over rows of two [rows, hidden] tensors."""
    return min(pcc(x, y) for x, y in zip(a, b))


def random_tokens(n, hidden, seed):
    """Normalized random rows (the scale of a post-attention RMSNorm output with unit gamma)."""
    g = torch.Generator().manual_seed(seed)
    return (
        (torch.nn.functional.normalize(torch.randn(n, hidden, generator=g), dim=-1) * hidden**0.5).bfloat16().float()
    )


def run_prefill(row, moe, rows):
    """TtMiMoMoE.forward on ``rows`` [T, hidden] (T a multiple of 32) -> ([T, hidden] of chip 0, all finite)."""
    x = ttnn.from_torch(
        rows.reshape(1, 1, *rows.shape).bfloat16(),
        device=row,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(row),
    )
    out = moe.forward(x)
    chips = [ttnn.to_torch(t).float()[0, 0] for t in ttnn.get_device_tensors(out)]
    x.deallocate(True)
    out.deallocate(True)
    finite = all(bool(torch.isfinite(c).all()) for c in chips)
    same = all(torch.equal(c, chips[0]) for c in chips[1:])  # the all-reduce leaves every chip the same sum
    return chips[0], finite and same


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--write-base", action="store_true")
    args = parser.parse_args()

    mesh, row = common.open_row()
    ok = True
    try:
        cfg = common.config()
        real = common.golden_moe_inputs(cfg, common.checkpoint(cfg))
        real = real[: real.shape[0] // 32 * 32]
        cases = {f"real{real.shape[0]}": real}
        for i, n in enumerate(RANDOM_TOKENS):
            cases[f"random{n}"] = random_tokens(n, cfg.hidden_size, seed=10 + i)
        moe = common.build_moe(row, cfg, "model_bfp4experts", ttnn.bfloat4_b)

        got, finite = {}, {}
        for name, rows in cases.items():
            got[name], finite[name] = run_prefill(row, moe, rows)
        again, _ = run_prefill(row, moe, cases[f"random{RANDOM_TOKENS[-1]}"])

        if args.write_base:
            torch.save(got, BASE)
            print(f"wrote base outputs {BASE}: {sorted(got)}", flush=True)
            return
        base = torch.load(BASE)
        for name in got:
            p, r = pcc(got[name], base[name]), row_pcc(got[name], base[name])
            passed = p >= 0.9995 and r >= 0.999 and finite[name]
            print(
                f"{name}: PCC vs base {p:.6f} (worst row {r:.6f}), finite and identical on every chip {finite[name]}"
                f" -> {'ok' if passed else 'FAIL'}",
                flush=True,
            )
            ok &= passed
        same = torch.equal(again, got[f"random{RANDOM_TOKENS[-1]}"])
        print(f"random{RANDOM_TOKENS[-1]} again: bitwise identical {same}", flush=True)
        ok &= same
    finally:
        common.close(mesh, row)
    if ok:
        print("MOE_PREFILL_CORRECTNESS_PASS", flush=True)
    else:
        raise RuntimeError("MoE prefill correctness check failed")


if __name__ == "__main__":
    main()
