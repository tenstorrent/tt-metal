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
    colones=23,
)

MASK_NEG = -30.0  # exp's domain is finite; -1e4 leaves it (see the non-flash attention test)


def bf16_pair(v):
    bits = int(torch.tensor([v], dtype=torch.bfloat16).view(torch.uint16)[0])
    return (bits << 16) | bits


def grid_transpose(k, sk, dt):
    """K's tile (s, d) to slot (d, s); contents untouched. See TransposeB."""
    v = k.reshape(sk, TILE, dt, TILE)
    return v.permute(2, 1, 0, 3).reshape(dt * TILE, sk * TILE)


def run(device, sq, sk_total, dt, num_chunks, causal, seed=0, fidelity=None, stream_buffering=2, num_q=1):
    """One launch, `num_q` query chunks of `sq` tiles each, against `sk_total` key tiles.

    num_q=1 is the old shape: one query chunk against the whole key range, which for a
    causal case is the LAST chunk of a prefill (it sees everything). num_q>1 walks the
    query dimension inside the kernel, so a whole head is one launch -- and the causal
    walk then skips key chunks that lie entirely above the diagonal instead of masking
    them.
    """
    assert sk_total % num_chunks == 0
    sk = sk_total // num_chunks
    # The kernel's causal walk needs whole chunks for every query chunk's key range.
    k_offset = sk_total - num_q * sq
    assert k_offset >= 0, "the key range must cover every query chunk"
    if causal:
        # Same condition the kernel static_asserts: sk divides the first query chunk's key
        # range, and divides sq too once there is more than one chunk to step by.
        assert (k_offset + sq) % sk == 0, "sk must divide k_offset + sq"
        assert num_q == 1 or sq % sk == 0, "sk must divide sq when there are several query chunks"
    S_q, S_k, D = num_q * sq * TILE, sk_total * TILE, dt * TILE

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

    # Chunk-major layouts, so each chunk's slice is contiguous for the kernel. K and V are
    # laid out ONCE over the whole key range: every query chunk reads a prefix of the same
    # chunks, so no per-query-chunk copy is needed.
    k_dev = torch.cat(
        [grid_transpose(k[j * sk * TILE : (j + 1) * sk * TILE].to(torch.float32), sk, dt) for j in range(num_chunks)],
        dim=0,
    ).to(torch.bfloat16)
    v_dev = v

    # The mask, in exactly the order the kernel consumes it: for each query chunk, one
    # block per key chunk it actually visits. The kernel walks a flat counter, so this
    # sequence IS the indexing -- there is no (i, j) arithmetic on either side.
    mask_blocks = []
    for i in range(num_q):
        visited = (k_offset + (i + 1) * sq) // sk if causal else num_chunks
        rows = slice(i * sq * TILE, (i + 1) * sq * TILE)
        for j in range(visited):
            mask_blocks.append(mask[rows, j * sk * TILE : (j + 1) * sk * TILE])
    mask_dev = torch.cat(mask_blocks, dim=0).to(torch.bfloat16)

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )

    # A column of ones, sk tiles tall, so the kernel's row sum can be a matmul against
    # it. Ones in column 0 only, which reproduces the reduction's contract exactly: the
    # sum lands in column 0 and the rest of the tile stays zero.
    col_ones = torch.zeros([sk * TILE, TILE])
    col_ones[:, 0] = 1.0
    tq, tk, tv, tmask = to_dev(q), to_dev(k_dev), to_dev(v_dev), to_dev(mask_dev)
    tcolones = to_dev(col_ones.to(torch.bfloat16))
    tout = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, S_q, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, dram)

    core_ranges, cores = single_core()
    ct_args = [sq, sk, dt, num_q, k_offset, sk_total]
    for t in (tq, tk, tv, tmask, tcolones, tout):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt_args = [t.buffer_address() for t in (tq, tk, tv, tmask, tcolones, tout)]

    scores_pages, out_pages, vec_pages = sq * sk, sq * dt, sq
    cbs = [
        make_cb(CB["q"], core_ranges, num_pages=sq * dt),
        # Streamed per chunk, so these are the only CBs where a second block lets the
        # reader run ahead of the compute. Worth a measured 5-6%, for L1 pages and
        # nothing else -- these shipped single-buffered, which is what a single/double
        # sweep turned up. The state CBs below are a different matter: their 2x is an
        # aliasing requirement, and halving THEM deadlocks.
        make_cb(CB["k"], core_ranges, num_pages=stream_buffering * dt * sk),
        make_cb(CB["v"], core_ranges, num_pages=stream_buffering * sk * dt),
        make_cb(CB["mask"], core_ranges, num_pages=stream_buffering * scores_pages),
        make_cb(CB["one"], core_ranges, num_pages=1),
        make_cb(CB["colones"], core_ranges, num_pages=sk),
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
        defines=None if causal else [("FLASH_NONCAUSAL", "1")],
        **(fidelity or {}),
    )
    out = ttnn.generic_op([tq, tk, tv, tmask, tcolones, tout], program)
    # Free the inputs: run() is called many times in one device session, and without this
    # the accumulated allocations exhaust DRAM partway through the sweep rather than at a
    # boundary that would make the cause obvious.
    for t in (tq, tk, tv, tmask, tcolones):
        ttnn.deallocate(t)
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

        # A whole causal head in ONE launch, which is what the q-loop is for. Checked
        # against the full S x S reference rather than a single query chunk, so this
        # covers the causal walk's chunk-skipping as well as the arithmetic: query chunk
        # i visits only the key chunks at or below its diagonal, and a bound that was one
        # chunk out either reads a masked block or misses a needed one.
        for sq_q, nq, sk_q in ((2, 4, 2), (2, 8, 1), (4, 4, 4)):
            got, want = run(device, sq_q, nq * sq_q, dt, nq * sq_q // sk_q, True, num_q=nq)
            e = (got - want).abs().max().item()
            ok = e <= args.abs_err
            logger.info(
                f"q-loop sq={sq_q} nq={nq} sk={sk_q} (S={nq * sq_q * TILE}): "
                f"max|err|={e:.5f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"qloop-{sq_q}-{nq}-{sk_q}")
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
