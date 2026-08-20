# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Flash attention on device: K and V streamed in chunks, with an online softmax.

THE GATE IS CHUNK INVARIANCE. The same Q, K, V and mask, run at 1, 2 and 4 chunks, must give
the same answer -- and all of them must match torch. That is the entire claim of the online
algorithm: the result does not depend on how the sequence was carved up. A correction that is
wrong, applied in the wrong order, or skipped shows up as a difference BETWEEN chunk counts
even when each individual run looks plausible on its own.

Matching torch alone would not catch it: a single-chunk run never rescales anything, so it
passes with the correction machinery entirely broken.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_flash.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
from unified_harness import make_cb, single_core, unified_program

KERNEL = "unified_kernels/flash_attention.cpp"
TILE = 32

CB = dict(
    q=0,
    k=1,
    v=2,
    mask=3,
    one=4,
    scale=5,
    scores=6,
    scaled=7,
    masked=8,
    rowmax=9,
    prob=10,
    probscaled=11,
    rowsum=12,
    pv=13,
    oscaled=14,
    corrold=15,
    out=16,
    corrnew=17,
    m=18,
    l=19,
    o=20,
    recipl=21,
    mnow=22,
)

MASK_NEG = -30.0  # exp's domain is finite; -1e4 leaves it (see the non-flash attention test)


def bf16_pair(v):
    bits = int(torch.tensor([v], dtype=torch.bfloat16).view(torch.uint16)[0])
    return (bits << 16) | bits


def grid_transpose(k, sk, dt):
    """K's tile (s, d) to slot (d, s); contents untouched. See TransposeB."""
    v = k.reshape(sk, TILE, dt, TILE)
    return v.permute(2, 1, 0, 3).reshape(dt * TILE, sk * TILE)


def run(device, sq, sk_total, dt, num_chunks, causal, seed=0, fidelity=None):
    assert sk_total % num_chunks == 0
    sk = sk_total // num_chunks
    S_q, S_k, D = sq * TILE, sk_total * TILE, dt * TILE

    torch.manual_seed(seed)
    # Q carries the 1/sqrt(d) scale: folding it in on the host costs one multiply here and
    # saves a broadcast pass per chunk on device.
    q_raw = torch.rand([S_q, D]) - 0.5
    q = (q_raw / (D**0.5)).to(torch.bfloat16)
    k = torch.rand([S_k, D]) - 0.5
    v = (torch.rand([S_k, D]) - 0.5).to(torch.bfloat16)

    # The keys are RAMPED along the sequence so later ones score much higher. Without this the
    # test is vacuous: with uniform random inputs every chunk's row maximum is nearly the same,
    # so exp(m_old - m_new) is already 1 and forcing the correction to 1 changes nothing. This
    # measured 0.005 against a 0.02 tolerance -- a broken rescale sailed through.
    #
    # The ramp is a function of the key's position in the FULL sequence, not of the chunk it
    # lands in, so every chunk count sees the same problem and invariance still means something.
    ramp = 1.0 + 20.0 * (torch.arange(S_k, dtype=torch.float32) / S_k)
    k = (k * ramp.unsqueeze(1)).to(torch.bfloat16)

    if causal:
        # K may carry context the queries did not produce, so query i sees keys up to
        # i + (S_k - S_q) -- ordinary prefill-with-history.
        off = S_k - S_q
        keep = torch.arange(S_k).unsqueeze(0) <= (torch.arange(S_q) + off).unsqueeze(1)
    else:
        keep = torch.ones([S_q, S_k], dtype=torch.bool)
    mask = torch.where(keep, 0.0, MASK_NEG)

    # Chunk-major layouts, so each chunk's slice is contiguous for the kernel.
    k_dev = torch.cat(
        [grid_transpose(k[j * sk * TILE : (j + 1) * sk * TILE].to(torch.float32), sk, dt) for j in range(num_chunks)],
        dim=0,
    ).to(torch.bfloat16)
    v_dev = v
    mask_dev = torch.cat([mask[:, j * sk * TILE : (j + 1) * sk * TILE] for j in range(num_chunks)], dim=0).to(
        torch.bfloat16
    )

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )

    tq, tk, tv, tmask = to_dev(q), to_dev(k_dev), to_dev(v_dev), to_dev(mask_dev)
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, S_q, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_ranges, cores = single_core()
    ct_args = [sq, sk, dt, num_chunks]
    for t in (tq, tk, tv, tmask, tout):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt_args = [t.buffer_address() for t in (tq, tk, tv, tmask, tout)]

    scores_pages, out_pages, vec_pages = sq * sk, sq * dt, sq
    cbs = [
        make_cb(CB["q"], core_ranges, num_pages=sq * dt),
        make_cb(CB["k"], core_ranges, num_pages=dt * sk),
        make_cb(CB["v"], core_ranges, num_pages=sk * dt),
        make_cb(CB["mask"], core_ranges, num_pages=scores_pages),
        make_cb(CB["one"], core_ranges, num_pages=1),
        make_cb(CB["scores"], core_ranges, num_pages=scores_pages),
        make_cb(CB["masked"], core_ranges, num_pages=scores_pages),
        make_cb(CB["rowmax"], core_ranges, num_pages=vec_pages),
        make_cb(CB["prob"], core_ranges, num_pages=scores_pages),
        make_cb(CB["rowsum"], core_ranges, num_pages=vec_pages),
        make_cb(CB["pv"], core_ranges, num_pages=out_pages),
        make_cb(CB["oscaled"], core_ranges, num_pages=out_pages),
        make_cb(CB["corrold"], core_ranges, num_pages=vec_pages),
        make_cb(CB["recipl"], core_ranges, num_pages=vec_pages),
        make_cb(CB["mnow"], core_ranges, num_pages=vec_pages),
        make_cb(CB["out"], core_ranges, num_pages=out_pages),
        # State: TWICE the block, so store() can reserve while the old value is still resident.
        make_cb(CB["m"], core_ranges, num_pages=2 * vec_pages),
        make_cb(CB["l"], core_ranges, num_pages=2 * vec_pages),
        make_cb(CB["o"], core_ranges, num_pages=2 * out_pages),
    ]

    program = unified_program(
        kernel_source=KERNEL,
        core_ranges=core_ranges,
        cores=cores,
        cbs=cbs,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        **(fidelity or {}),
    )
    out = ttnn.generic_op([tq, tk, tv, tmask, tout], program)
    got = ttnn.to_torch(out).to(torch.float32)[0, 0]

    # The reference divides the scores, where the device pre-divided Q: mathematically the same,
    # and the rounding difference shows up in the error rather than being hidden.
    qf, kf, vf = q_raw.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    scores = qf @ kf.T / (D**0.5) + mask
    want = torch.softmax(scores, dim=-1) @ vf
    return got, want


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--abs-err", type=float, default=0.03)
    p.add_argument("--invariance", type=float, default=0.02, help="max spread between chunk counts")
    args = p.parse_args(argv)

    sq, sk_total, dt = 2, 4, 2  # 64 queries, 128 keys, head dim 64
    chunk_counts = [1, 2, 4]

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for causal in (True, False):
            results = {}
            for nc in chunk_counts:
                got, want = run(device, sq, sk_total, dt, nc, causal)
                e = (got - want).abs().max().item()
                ok = e <= args.abs_err
                logger.info(
                    f"causal={int(causal)} chunks={nc} (Sk/chunk={sk_total // nc}t)  "
                    f"max|err|={e:.5f}  {'ok' if ok else 'FAIL'}"
                )
                results[nc] = got
                if not ok:
                    failed.append((causal, nc))

            # The real gate. A single chunk never rescales anything, so agreement with torch
            # alone would pass with the correction machinery broken.
            base = results[chunk_counts[0]]
            for nc in chunk_counts[1:]:
                spread = (results[nc] - base).abs().max().item()
                ok = spread <= args.invariance
                logger.info(
                    f"causal={int(causal)} INVARIANCE {chunk_counts[0]} vs {nc} chunks: "
                    f"max|diff|={spread:.5f}  {'ok' if ok else 'FAIL'}"
                )
                if not ok:
                    failed.append((causal, "invariance", nc))
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
