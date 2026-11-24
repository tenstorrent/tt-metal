# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""K-GPTQ: GPTQ rounding of gpt-oss-20b's MXFP4 routed experts onto the K-grouped bfloat4_b grid.

This builds GPT_OSS_DECODE_SPLIT_ROUND_DIR (kgptq_experts/layer_<L>.pt), the expert weights the gpt-oss-20b release
settings read (models/demos/gpt_oss/__init__.py) in the fused decode streams and the routed prefill experts.

The grid. A bfloat4_b block is 16 values that share one exponent; each value is a sign and a 3-bit magnitude in steps
of 2^(e - 2), e the exponent of the block's largest value. tools/gptq_experts.py groups the 16 values along the
output dimension N (one input row, 16 output columns: the normal tile layout). The release stores the expert tiles
transposed within themselves (ttnn.Tile transpose_tile, unified_routed_expert_moe weights_tile_transposed), so each
block is 16 consecutive input rows k .. k + 15 of one output column (k a multiple of 16). This tool rounds onto that
grid. Values on it survive a transposed-tile bfloat4_b pack unchanged; a normal pack re-rounds them.

The rounding. GPTQ walks the input rows in order, rounds each row, and pushes the row's rounding error onto the rows
not yet rounded through the inverse Hessian (H = X^T X of real expert inputs), so the errors cancel in the expert's
output. With K-grouped blocks the 16 rows of a block share each column's step, so the step is fixed when the walk
reaches the block's first row, from the block's current (error-updated) values: the block max's step 2^(e - 2), or
the next finer step 2^(e - 3) (the max clips to 7 steps) when that rounds the block closer. The device derives the
same step from the stored values. Rows are walked in natural order: act order would mix the rows of a block.

Inputs. The calibration files are the same as tools/gptq_experts.py's (`collect`: each layer's MoE inputs and routed
experts from the Hugging Face reference on CPU). The published weights (hf://jerrywangTT/gpt-oss-20b-qb2-kgptq)
were calibrated on 180,224 tokens in 4,096-token chunks (gpt-oss-20b's own AIME 2025 reasoning traces, plus some code
and documentation text; see that repo's README). Different calibration text gives different (equally valid) bytes.

  # 1. calibration (tools/gptq_experts.py; ~70 s per 4096 tokens on CPU)
  python models/demos/gpt_oss/tools/gptq_experts.py collect calib.pt texts/*.txt
  # 2. rounding: one layer_<L>.pt per layer, the format GPT_OSS_DECODE_SPLIT_ROUND_DIR reads
  python models/demos/gpt_oss/tools/gptq_k.py round kgptq_experts/ calib.pt [calib2.pt ...]
  # 3. check a directory (yours or the published one): on the K-grouped grid, and expert output error vs MXFP4
  python models/demos/gpt_oss/tools/gptq_k.py check kgptq_experts/ --calib held_out.pt
  # then
  export GPT_OSS_DECODE_SPLIT_ROUND_DIR=$PWD/kgptq_experts

Output keys (per layer, checkpoint channel order): gate_up_proj [E, H, 2I] (gate / up interleaved) and down_proj
[E, I, H] in bfloat16 (every value is exact in bfloat16), gate_up_proj_bias and down_proj_bias unchanged (fp32).
Use 8 threads (THREADS=8): at 16 the per-row ops spend most of their time synchronizing.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gptq_experts import Checkpoint, _snapshot, swiglu  # noqa: E402

GROUP = 16  # values per bfloat4_b block (one exponent)
MAG = 7  # largest magnitude in steps (3 bits)


def _block_steps(blk):
    """blk [..., GROUP, N] (16 consecutive input rows) -> [..., 1, N] the step each column's block rounds with: the
    block max's step 2^(e - 2), or 2^(e - 3) when that rounds the block closer (squared error over the 16 values)."""
    m = blk.abs().amax(-2, keepdim=True)
    e = torch.floor(torch.log2(m.clamp_min(1e-30)))
    s0, s1 = torch.pow(2.0, e - 2), torch.pow(2.0, e - 3)
    err0 = ((torch.clamp(torch.round(blk / s0), -MAG, MAG) * s0 - blk) ** 2).sum(-2, keepdim=True)
    err1 = ((torch.clamp(torch.round(blk / s1), -MAG, MAG) * s1 - blk) ** 2).sum(-2, keepdim=True)
    step = torch.where(err1 < err0, s1, s0)
    return torch.where(m > 0, step, torch.ones_like(step))  # an all-zero block stays zero whatever the step


def k_grouped_rne(w):
    """w [..., K, N] -> the K-grouped bfloat4_b grid, round half to even: what a transposed-tile bfloat4_b pack does
    (blocks of 16 consecutive rows along K, the step from each block's max)."""
    K, N = w.shape[-2:]
    b = w.reshape(*w.shape[:-2], K // GROUP, GROUP, N)
    m = b.abs().amax(-2, keepdim=True)
    step = torch.pow(2.0, torch.floor(torch.log2(m.clamp_min(1e-30))) - 2)
    q = torch.clamp(torch.round(b / step), -MAG, MAG) * step
    return torch.where(m > 0, q, b).reshape(w.shape)


def gptq_k(W, H, damp=0.01, block=128):
    """W [E, K, N] (K: input rows), H [E, K, K] = X^T X of the calibration inputs -> W rounded onto the K-grouped
    bfloat4_b grid. Natural row order; each block's steps fixed at its first row; each row's error pushed onto the rows
    after it (lazy batch updates every `block` rows)."""
    E, K, N = W.shape
    assert K % GROUP == 0 and block % GROUP == 0, (K, block)
    W, H = W.clone().float(), H.clone().float()
    dead = torch.diagonal(H, dim1=1, dim2=2) == 0
    H += torch.diag_embed(dead.float())
    W.masked_fill_(dead.unsqueeze(-1), 0)
    H += damp * torch.diagonal(H, dim1=1, dim2=2).mean(-1)[:, None, None] * torch.eye(K)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H.double())), upper=True).float()
    Q = torch.zeros_like(W)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1, Err, Hi = W[:, i1:i2].clone(), torch.zeros(E, i2 - i1, N), Hinv[:, i1:i2, i1:i2]
        step = None
        for i in range(i2 - i1):
            if i % GROUP == 0:
                step = _block_steps(W1[:, i : i + GROUP])[:, 0]  # [E, N]
            q = torch.clamp(torch.round(W1[:, i] / step), -MAG, MAG) * step
            Q[:, i1 + i] = q
            err = (W1[:, i] - q) / Hi[:, i, i].unsqueeze(-1)
            W1[:, i:] -= Hi[:, i, i:].unsqueeze(-1) * err.unsqueeze(1)
            Err[:, i] = err
        W[:, i2:] -= torch.bmm(Hinv[:, i1:i2, i2:].transpose(1, 2), Err)
    return Q


def expert_hessians(weights, x, routed):
    """Per expert: H_gate_up = X^T X over the tokens routed to it, H_down = A^T A of its SwiGLU outputs A (computed
    with the checkpoint's gate / up weights). x [T, H] MoE inputs, routed [T, top_k] expert ids."""
    gu, gub = weights["gate_up_proj"], weights["gate_up_proj_bias"]
    E, H, I2 = gu.shape
    Hg, Hd = torch.zeros(E, H, H), torch.zeros(E, I2 // 2, I2 // 2)
    counts = []
    for e in range(E):
        X = x[(routed == e).any(-1)]
        A = swiglu(X @ gu[e] + gub[e])
        Hg[e], Hd[e] = X.t() @ X, A.t() @ A
        counts.append(X.shape[0])
    return Hg, Hd, counts


def round_layer(weights, x, routed, damp=0.01):
    """One layer's experts onto the K-grouped grid: gate_up [E, H, 2I] grouped along H, down [E, I, H] along I. Gate
    and up columns never share a block here (a block is one column), so gate_up rounds as one matrix."""
    Hg, Hd, counts = expert_hessians(weights, x, routed)
    out = dict(weights)
    out["gate_up_proj"] = gptq_k(weights["gate_up_proj"], Hg, damp)
    out["down_proj"] = gptq_k(weights["down_proj"], Hd, damp)
    for name in ("gate_up_proj", "down_proj"):
        w = out[name]
        if not torch.equal(k_grouped_rne(w), w):  # a transposed-tile pack would round it again
            raise RuntimeError(f"{name} is not on the K-grouped bfloat4_b grid after GPTQ")
        if not torch.equal(w.bfloat16().float(), w):
            raise RuntimeError(f"{name} is not exact in bfloat16")
    return out, min(counts)


def _load_calib(files, i):
    parts = [torch.load(f) for f in files]
    return torch.cat([p[i][0] for p in parts]).float(), torch.cat([p[i][1] for p in parts])


def round_all(out_dir, calib_files, layers, damp):
    os.makedirs(out_dir, exist_ok=True)
    parts = [torch.load(f) for f in calib_files]
    ckpt = Checkpoint()
    for i in layers:
        t = time.time()
        x = torch.cat([p[i][0] for p in parts]).float()
        routed = torch.cat([p[i][1] for p in parts])
        out, fewest = round_layer(ckpt.layer(i), x, routed, damp)
        torch.save({k: v.bfloat16() if k in ("gate_up_proj", "down_proj") else v for k, v in out.items()},
                   f"{out_dir}/layer_{i}.pt")  # fmt: skip
        print(f"layer {i}: {time.time() - t:.0f} s ({x.shape[0]} tokens, fewest per expert {fewest})", flush=True)


def _expert_out(w, x, routed, e, gate_up=None, down=None):
    gu = w["gate_up_proj"][e].float() if gate_up is None else gate_up
    dn = w["down_proj"][e].float() if down is None else down
    X = x[(routed == e).any(-1)]
    return X, swiglu(X @ gu + w["gate_up_proj_bias"][e].float()) @ dn + w["down_proj_bias"][e].float()


def check(round_dir, layers, calib_files):
    """Each layer file: keys and shapes as the checkpoint's, on the K-grouped grid, exact in bfloat16; with calibration
    inputs (ideally held out) also the expert output error against the MXFP4 checkpoint, next to plain round-to-nearest
    onto the same grid (what loading the checkpoint without the rounded weights gives)."""
    ckpt = Checkpoint()
    for i in layers:
        f = Path(round_dir) / f"layer_{i}.pt"
        w, ref = torch.load(f, mmap=True), ckpt.layer(i)
        assert set(w) == set(ref), f"{f}: keys {sorted(w)} != {sorted(ref)}"
        line = [f"layer {i}:"]
        for k in ref:
            assert w[k].shape == ref[k].shape, f"{f}: {k} {tuple(w[k].shape)} != {tuple(ref[k].shape)}"
        for k in ("gate_up_proj", "down_proj"):
            v = w[k].float()
            on = torch.equal(k_grouped_rne(v), v)
            moved = (v - ref[k]).abs().gt(0).float().mean().item()
            line.append(f"{k} on grid {on}, {moved:.1%} of values differ from MXFP4")
        if calib_files:
            x, routed = _load_calib(calib_files, i)
            num_g = den_g = num_r = 0.0
            for e in range(ref["gate_up_proj"].shape[0]):
                _, y_ref = _expert_out(ref, x, routed, e)
                _, y = _expert_out(w, x, routed, e)
                _, y_rtn = _expert_out(
                    ref,
                    x,
                    routed,
                    e,
                    gate_up=k_grouped_rne(ref["gate_up_proj"][e]),
                    down=k_grouped_rne(ref["down_proj"][e]),
                )
                num_g += ((y - y_ref) ** 2).sum().item()
                num_r += ((y_rtn - y_ref) ** 2).sum().item()
                den_g += (y_ref**2).sum().item()
            line.append(
                f"expert output rel. error {(num_g / den_g) ** 0.5:.2%} (round-to-nearest {(num_r / den_g) ** 0.5:.2%})"
            )
        print("  ".join(line), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("round", help="calibration file(s) -> layer_<L>.pt")
    r.add_argument("out_dir")
    r.add_argument("calib", nargs="+")
    r.add_argument("--layers", default="all")
    r.add_argument("--damp", type=float, default=0.01)
    c = sub.add_parser("check", help="verify a directory of layer_<L>.pt")
    c.add_argument("round_dir")
    c.add_argument("--layers", default="all")
    c.add_argument("--calib", nargs="*", default=[])
    a = p.parse_args()
    n = json.loads((_snapshot() / "config.json").read_text())["num_hidden_layers"]
    layers = range(n) if a.layers == "all" else [int(v) for v in a.layers.split(",")]
    if a.cmd == "round":
        round_all(a.out_dir, a.calib, layers, a.damp)
    else:
        check(a.round_dir, layers, a.calib)
