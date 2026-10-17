# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for MiMo MoE prefill changes (``TtMiMoMoE.forward``, one layer on Galaxy row 0, BFP4 experts).

1. Prefill on the golden prompts' real tokens at this layer (96) and on normalized random tokens (32, 1056 and
   4096: one tile row, an odd number of tile rows, a full 4K chunk) must match the base implementation's saved
   output on the same inputs: PCC >= 0.9995 overall and >= 0.999 on every token, so one dropped or mis-weighted
   expert of one token fails it, and every value must be finite.
2. Decode (``forward_decode``, 4 users, 8 steps) shares the expert tensors and must still match its saved base
   output: PCC >= 0.9998 overall and >= 0.9995 on every (step, user) row.

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
DECODE_STEPS = 8


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


def run_decode(row, moe, rows):
    """forward_decode over DECODE_STEPS x 4 users -> ([steps * users, hidden], all finite)."""
    outs, finite = [], True
    for s in range(DECODE_STEPS):
        x, mask = common.device_inputs(row, rows[s * common.USERS : (s + 1) * common.USERS])
        out = moe.forward_decode(x, mask)
        full = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0]
        finite &= bool(torch.isfinite(full).all())
        outs.append(full[: common.USERS])
        for t in (x, mask, out):
            t.deallocate(True)
    return torch.cat(outs), finite


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
        got["decode"], finite["decode"] = run_decode(row, moe, real[: DECODE_STEPS * common.USERS])

        if args.write_base:
            torch.save(got, BASE)
            print(f"wrote base outputs {BASE}: {sorted(got)}", flush=True)
            return
        base = torch.load(BASE)
        for name in got:
            p, r = pcc(got[name], base[name]), row_pcc(got[name], base[name])
            need = (0.9998, 0.9995) if name == "decode" else (0.9995, 0.999)
            passed = p >= need[0] and r >= need[1] and finite[name]
            print(
                f"{name}: PCC vs base {p:.6f} (worst row {r:.6f}), finite and identical on every chip {finite[name]}"
                f" -> {'ok' if passed else 'FAIL'}",
                flush=True,
            )
            ok &= passed
    finally:
        common.close(mesh, row)
    if ok:
        print("MOE_PREFILL_CORRECTNESS_PASS", flush=True)
    else:
        raise RuntimeError("MoE prefill correctness check failed")


if __name__ == "__main__":
    main()
