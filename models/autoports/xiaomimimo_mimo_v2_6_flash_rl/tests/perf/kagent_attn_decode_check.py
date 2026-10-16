# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for MiMo attention decode changes (Galaxy row 0, 4 users, layers 1 (SWA) and 5 (global)).

For each layer: prefill the users (input norm + attention prefill + residual), then DECODE_STEPS decode steps
of TtMiMoDecoderLayer.attention_decode with fresh inputs; the KV cache grows across the steps. The attention
contribution (output minus input) must match the base implementation's saved outputs (``--write-base``):
PCC >= 0.9998 over all decode rows, >= 0.9995 on every (step, user) row and on every prefilled user; every row
of the [32, hidden] decode output (the padding rows too) must be finite.

Prints ATTN_DECODE_CORRECTNESS_PASS after every check passed.
"""

import argparse
import os

import torch

from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.perf import attn_decode_common as attn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.perf import moe_decode_common as common

BASE = os.environ.get("MIMO_ATTN_DECODE_BASE", "/home/zni/models/mimo-golden/attn_decode_layers1_5_base.pt")
DECODE_STEPS = 4


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run_layer(row, cfg, idx):
    state = attn.build(row, cfg, idx)
    pre_out = attn.prefill(state, cfg, seed=idx)
    g = torch.Generator().manual_seed(idx)
    pre_x = [
        (torch.randn(n, cfg.hidden_size, generator=g) * attn.HIDDEN_STD).bfloat16().float() for n in attn.PROMPT_LENS
    ]
    pre_delta = [o - x for o, x in zip(pre_out, pre_x)]
    deltas, finite = [], True
    for step in range(DECODE_STEPS):
        x_tt, rot, cur_pos, x = attn.decode_inputs(state, cfg, step, seed=1000 * idx + step)
        out = attn.decode(state, x_tt, rot, cur_pos)
        full = state["plan"].to_torch_replicated(out).float()[0, 0]
        finite &= bool(torch.isfinite(full).all())
        deltas.append(full[: attn.USERS] - x[: attn.USERS])
        out.deallocate(True)
        x_tt.deallocate(True)
    return pre_delta, torch.cat(deltas), finite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--write-base", action="store_true", help="record this code's outputs as the base")
    args = parser.parse_args()

    cfg = common.config()
    mesh, row = common.open_row()
    results = {}
    try:
        for idx in attn.LAYERS:
            results[idx] = run_layer(row, cfg, idx)
    finally:
        common.close(mesh, row)

    if args.write_base:
        torch.save({idx: (p, d) for idx, (p, d, _) in results.items()}, BASE)
        print(f"wrote base outputs {BASE}", flush=True)
        return
    base = torch.load(BASE)
    ok = True
    for idx, (pre, dec, finite) in results.items():
        base_pre, base_dec = base[idx]
        p_all = pcc(dec, base_dec)
        p_row = min(pcc(a, b) for a, b in zip(dec, base_dec))
        p_pre = min(pcc(a, b) for a, b in zip(pre, base_pre))
        print(
            f"layer {idx}: decode delta PCC {p_all:.6f} (worst row {p_row:.6f}), prefill worst user {p_pre:.6f}, "
            f"finite {finite}",
            flush=True,
        )
        ok &= p_all >= 0.9998 and p_row >= 0.9995 and p_pre >= 0.9995 and finite
    if ok:
        print("ATTN_DECODE_CORRECTNESS_PASS", flush=True)
    else:
        raise RuntimeError("attention decode correctness check failed")


if __name__ == "__main__":
    main()
