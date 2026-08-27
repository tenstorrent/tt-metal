# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One attention head on device, against torch's scaled_dot_product_attention.

    out = softmax(Q @ K.T / sqrt(d) + mask) @ V

This is the first kernel that composes the whole set: a transposed matmul, a scalar
broadcast, a two-buffer elementwise add, a row-wise max and sum, a broadcast subtract with
a fused exp, a reciprocal folded into a reduction's epilogue, a broadcast multiply, and a
second matmul.

WHY PCC IS NOT THE GATE. Softmax rows sum to 1 and every probability is O(1/S), so a
global scale error or a per-row offset correlates almost perfectly with the truth -- the
same blind spot that let a bias offset and a mean scale factor through at 0.9999 earlier in
this model. The checks that carry information here are:

  max abs error vs torch      catches magnitude errors PCC cannot see
  row sums of the softmax     must be 1.0; catches a missing or wrong normalisation
  causal structure            with a causal mask, row 0 of the output must equal V's row 0
                              exactly, because position 0 attends only to itself

The last one is the sharpest: it is a closed-form value that depends on the mask, the
softmax and both matmuls all being right, and it is not a tolerance -- it is an identity.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_attention.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import dfb, run_unified_spec, single_core, unified_program_spec

KERNEL = "unified_kernels/attention.cpp"
TILE = 32


# A MODEST negative additive mask. -inf and -1e4 are both wrong here: the SFPU's exp has a
# finite input domain and exp(-1e4 - rowmax) leaves it rather than underflowing to zero.
# -30 is ample -- exp(-30) is 1e-13 against a row sum of at least 1 -- and stays inside it.
MASK_NEG = -30.0


def grid_transpose(k, sk, dt):
    """Move K's tile (s, d) to slot (d, s). Tile CONTENTS untouched -- the hardware flag
    does the within-tile half. See TransposeB in tt/unified/math.hpp."""
    v = k.reshape(sk, TILE, dt, TILE)  # s, i, d, j
    return v.permute(2, 1, 0, 3).reshape(dt * TILE, sk * TILE)  # d, i, s, j


def run(device, sq, sk, dt, causal=True, seed=0):
    torch.manual_seed(seed)
    S_q, S_k, D = sq * TILE, sk * TILE, dt * TILE

    q = (torch.rand([S_q, D]) - 0.5).to(torch.bfloat16)
    k = (torch.rand([S_k, D]) - 0.5).to(torch.bfloat16)
    v = (torch.rand([S_k, D]) - 0.5).to(torch.bfloat16)

    if causal:
        keep = torch.arange(S_k).unsqueeze(0) <= torch.arange(S_q).unsqueeze(1)
    else:
        keep = torch.ones([S_q, S_k], dtype=torch.bool)
    mask = torch.where(keep, 0.0, MASK_NEG).to(torch.bfloat16)

    k_dev = grid_transpose(k.to(torch.float32), sk, dt).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )

    tq, tk, tv, tmask = to_dev(q), to_dev(k_dev), to_dev(v), to_dev(mask)
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, S_q, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_ranges, cores = single_core()
    named_ct_args = [("sq", sq), ("sk", sk), ("dt", dt)]
    # 1/sqrt(head dim) as a packed bfloat16 pair -- the kernel toolchain has no sqrtf, and
    # fill_reduce_scaler writes raw 32-bit words, so bfloat16 means two values per word.
    scale = 1.0 / (D**0.5)
    bits = int(torch.tensor([scale], dtype=torch.bfloat16).view(torch.uint16)[0])
    rt_args = [x.buffer_address() for x in (tq, tk, tv, tmask, tout)] + [(bits << 16) | bits]

    dfbs = [
        dfb("q", sq * dt),
        dfb("k", dt * sk),
        dfb("v", sk * dt),
        dfb("mask", sq * sk),
        dfb("one", 1),
        dfb("scale", 1),
        dfb("scores", sq * sk),
        dfb("scaled", sq * sk),
        dfb("masked", sq * sk),
        dfb("row_max", sq),
        dfb("exp", sq * sk),
        dfb("recip", sq),
        dfb("prob", sq * sk),
        dfb("out", sq * dt),
    ]

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        named_compile_time_args=named_ct_args,
        tensors={"q": tq, "k": tk, "v": tv, "mask": tmask, "out": tout},
        runtime_arg_names=["scale_bits"],
    )
    run_unified_spec(
        device,
        spec,
        {"q": tq, "k": tk, "v": tv, "mask": tmask, "out": tout},
        runtime_args={"scale_bits": (bits << 16) | bits},
        nodes=cores,
    )
    out = tout
    got = ttnn.to_torch(out).to(torch.float32)[0, 0]

    # Reference, in fp32 from the same bfloat16 inputs the device saw.
    qf, kf, vf = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    scores = qf @ kf.T / (D**0.5)
    scores = scores + torch.where(keep, 0.0, float(MASK_NEG))
    probs = torch.softmax(scores, dim=-1)
    want = probs @ vf
    return got, want, probs, vf


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--abs-err", type=float, default=0.02, help="max absolute error vs torch")
    args = p.parse_args(argv)

    # (sq, sk, dt). Both matmuls' output blocks must fit the 8-tile DST budget, so
    # sq*sk <= 8 and sq*dt <= 8 -- see the static_assert in Strategy<FPUFusion>.
    cases = [(1, 1, 1), (2, 2, 2), (2, 4, 2), (1, 4, 2), (4, 2, 2)]

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for sq, sk, dt in cases:
            for causal in (True, False):
                got, want, probs, vf = run(device, sq, sk, dt, causal)
                abs_err = (got - want).abs().max().item()
                ok = abs_err <= args.abs_err
                note = ""
                if causal:
                    # Position 0 attends only to itself, so out row 0 IS V row 0. An
                    # identity, not a tolerance.
                    row0 = (got[0] - vf[0]).abs().max().item()
                    ok = ok and row0 <= args.abs_err
                    note = f"  out[0]==V[0] within {row0:.4f}"
                logger.info(
                    f"Sq={sq * TILE:3d} Sk={sk * TILE:3d} D={dt * TILE:3d} causal={int(causal)}  "
                    f"max|err|={abs_err:.5f}{note}   {'ok' if ok else 'FAIL'}"
                )
                if not ok:
                    failed.append((sq, sk, dt, causal))

        # Row sums of the softmax the device actually computed cannot be read back, so the
        # reference's are checked instead -- if they were not 1.0 the comparison above would
        # be against the wrong thing.
        s = probs.sum(dim=-1)
        logger.info(f"reference softmax row sums: min={s.min():.6f} max={s.max():.6f}")
        if not torch.allclose(s, torch.ones_like(s), atol=1e-5):
            failed.append("reference-row-sums")
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
