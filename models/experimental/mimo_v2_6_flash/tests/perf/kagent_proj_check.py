# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for the MiMo decode projections (tt/proj_stream.py) on Galaxy row 0.

Four input steps through qkv, o and router. Each output must keep the model's format (qkv [1, 1, 32, 1920] bf16
TILE in L1, o [1, 1, 32, 4096] bf16 TILE in L1, router [1, 1, 32, 256] fp32 TILE in DRAM, all interleaved), match
the base implementation's saved outputs (``--write-base``) on the user rows 0-3 with PCC >= 0.99999 per row (same
fidelities and fp32 accumulation; only the accumulation order may differ), and be finite in every row.

Prints PROJ_DECODE_CORRECTNESS_PASS after every check passed.
"""

import argparse
import os

import torch

from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common_paths

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import proj_decode_common as proj_common

BASE = os.environ.get("MIMO_PROJ_DECODE_BASE", os.path.join(common_paths.GOLDEN_DIR, "proj_decode_base.pt"))
STEPS = 4
FORMATS = {
    "qkv": ((1, 1, 32, proj_common.QKV_WIDTH), ttnn.bfloat16, ttnn.BufferType.L1),
    "o": ((1, 1, 32, proj_common.HIDDEN), ttnn.bfloat16, ttnn.BufferType.L1),
    "router": ((1, 1, 32, proj_common.EXPERTS), ttnn.float32, ttnn.BufferType.DRAM),
}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--write-base", action="store_true")
    args = parser.parse_args()
    mesh, row = proj_common.open_row()
    got = {k: [] for k in FORMATS}
    ok_format, finite = True, True
    try:
        plan, proj, weights = proj_common.build(row)
        for step in range(STEPS):
            x_h, x_o = proj_common.inputs(row, step)
            for name, t in zip(FORMATS, proj_common.run(proj, weights, x_h, x_o)):
                shape, dtype, buf = FORMATS[name]
                mc = t.memory_config()
                fmt = (
                    tuple(t.shape) == shape
                    and t.dtype == dtype
                    and t.layout == ttnn.TILE_LAYOUT
                    and mc.buffer_type == buf
                    and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
                )
                if not fmt:
                    print(f"{name}: wrong output format {tuple(t.shape)} {t.dtype} {t.layout} {mc}", flush=True)
                ok_format &= fmt
                full = plan.to_torch_replicated(t).float()[0, 0]
                finite &= bool(torch.isfinite(full).all())
                got[name].append(full[: proj_common.USERS])
                t.deallocate(True)
            x_h.deallocate(True)
            x_o.deallocate(True)
    finally:
        proj_common.close(mesh, row)
    got = {k: torch.cat(v) for k, v in got.items()}
    if args.write_base:
        torch.save(got, BASE)
        print(f"wrote base outputs {BASE}", flush=True)
        return
    base = torch.load(BASE)
    ok = ok_format and finite
    for name in FORMATS:
        worst = min(pcc(a, b) for a, b in zip(got[name], base[name]))
        print(f"{name}: worst row PCC vs base {worst:.7f}", flush=True)
        ok &= worst >= 0.99999
    print(f"formats ok {ok_format}, finite {finite}", flush=True)
    if ok:
        print("PROJ_DECODE_CORRECTNESS_PASS", flush=True)
    else:
        raise RuntimeError("decode projection correctness check failed")


if __name__ == "__main__":
    main()
