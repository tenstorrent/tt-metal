# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reads the probe files of tests/test_probe.py (no device, no ttnn):

    python analysis.py [directory] [experiment-prefix]

For each file: how the device's sums compare with exact arithmetic on the stored operands and with the operand
models of model.py; for the weight reads of the block formats, the table "stored mantissa field -> what came back"."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

try:
    from models.experimental.hy4_preview.fused.fidelity import probes as PR
except ImportError:  # run as a script from this directory
    import probes as PR

DEFAULT_DIR = "generated/hy4_preview/logs/mk9_probe"
FIELD_BITS = {"bfp8": 7, "bfp4": 3}


def load(directory=DEFAULT_DIR, prefix=""):
    out = {}
    for path in sorted(Path(directory).glob(f"{prefix}*.pt")):
        d = torch.load(path)
        if "fidelity" in d:  # the files of tests/test_dense.py carry their form in the name
            out[(d["name"], d["fidelity"])] = d
    return out


def weight_view(d):
    """A wread file -> (stored rows, device rows, K index of each row, column), every one [n, 32 ct]: the device
    row is the output row divided by the row's x value."""
    x, out = d["x"], d["out"]
    runs, rows, _ = x.shape
    idx = x.abs().argmax(dim=2)
    value = x.gather(2, idx[..., None])
    stored = d["stored"][0][idx.reshape(-1)]
    seen = (out / value).reshape(runs * rows, -1)
    cols = torch.arange(seen.shape[1])[None].expand_as(seen)
    return stored, seen, idx.reshape(-1, 1).expand_as(seen), cols


def field(v, block_e, bits):
    """|v| in units of the block's mantissa step."""
    return (v.double().abs() / 2.0 ** (block_e - bits + 1)).round().long()


def summarize_wread(d):
    stored, seen, k, col = weight_view(d)
    lines = [
        f"equal to stored {float((seen == stored).float().mean()):.4f}",
    ]
    for bits in (3, 4, 5, 6, 7):
        lines.append(
            f"cut{bits} {float((seen == PR.trunc_sig(stored, bits)).float().mean()):.4f} "
            f"round{bits} {float((seen == PR.round_sig(stored, bits)).float().mean()):.4f}"
        )
    rel = ((seen.double() - stored.double()).pow(2).sum().sqrt() / stored.double().pow(2).sum().sqrt()).item()
    gain = ((seen.double() * stored.double()).sum() / stored.double().pow(2).sum()).item()
    lines.append(f"rel rms {rel:.3e} gain {gain:.5f}")
    return "; ".join(lines)


def field_table(d):
    """Block formats: stored mantissa field -> Counter of the device's field."""
    bits = FIELD_BITS[d["fmt"]]
    stored, seen, k, col = weight_view(d)
    e = PR.block_exponent(d["stored"][0])[k[:, 0]]
    fs, fd = field(stored, e, bits), field(seen, e, bits)
    table = defaultdict(Counter)
    for a, b in zip(fs.flatten().tolist(), fd.flatten().tolist()):
        table[a][b] += 1
    return table


def summarize_sums(d):
    x = d["x"].to(torch.bfloat16).float() if d["x_fmt"] == "bf16" else d["x"]
    exact = x.double() @ d["stored"].double()
    out = d["out"].double()
    nz = exact != 0
    equal = float((out == exact).float().mean())
    rel = ((out - exact).pow(2).sum().sqrt() / exact.pow(2).sum().sqrt()).item()
    gain = ((out * exact).sum() / exact.pow(2).sum()).item()
    ratio = out[nz] / exact[nz]
    return (
        f"equal {equal:.4f} rel rms {rel:.3e} gain {gain:.6f} ratio min {ratio.min().item():.5f} "
        f"max {ratio.max().item():.5f}"
    )


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""
    for (name, fidelity), d in load(directory, prefix).items():
        print(f"{name} {fidelity}: {summarize_sums(d)}")
        if name.startswith("wread"):
            print(f"    weights seen: {summarize_wread(d)}")


if __name__ == "__main__":
    main()
