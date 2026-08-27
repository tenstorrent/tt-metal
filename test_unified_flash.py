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
from unified_harness import core_block, dfb, run_unified_spec, single_core, split_evenly, unified_program_spec

KERNEL = "unified_kernels/flash_attention.cpp"
TILE = 32


MASK_NEG = -30.0  # exp's domain is finite; -1e4 leaves it (see the non-flash attention test)


def bf16_pair(v):
    bits = int(torch.tensor([v], dtype=torch.bfloat16).view(torch.uint16)[0])
    return (bits << 16) | bits


def grid_transpose(k, sk, dt):
    """K's tile (s, d) to slot (d, s); contents untouched. See TransposeB."""
    v = k.reshape(sk, TILE, dt, TILE)
    return v.permute(2, 1, 0, 3).reshape(dt * TILE, sk * TILE)


def _launch(
    device,
    tq,
    tk,
    tv,
    tmask,
    tcolones,
    tout,
    sq,
    sk,
    dt,
    num_q,
    k_offset,
    sk_total,
    n_heads,
    n_kv_heads,
    causal,
    cores,
    stream_buffering,
    fidelity,
):
    """The launch itself, shared by run() and run_preloaded() so both take the same path."""
    # Heads are partitioned across cores and share nothing: a core reads its own heads'
    # queries and their KV heads, and writes its own output blocks. No core touches
    # another's data, so there is no communication here at all -- only a range each.
    #
    # Capped at n_heads so no core is allocated a share of zero. The kernel would survive
    # one (its loop simply would not run), but an idle core is a pointless allocation and
    # the path is not worth having untested.
    ncores = min(cores, n_heads)
    core_ranges, core_list = core_block(ncores)
    shares = split_evenly(n_heads, ncores)

    named_ct_args = [
        ("sq", sq),
        ("sk", sk),
        ("dt", dt),
        ("num_q_chunks", num_q),
        ("k_offset", k_offset),
        ("k_tiles", sk_total),
        ("n_heads", n_heads),
        ("n_kv_heads", n_kv_heads),
    ]

    scores_pages, out_pages, vec_pages = sq * sk, sq * dt, sq
    dfbs = [
        dfb("q", sq * dt),
        # Streamed per chunk, so these are the only CBs where a second block lets the
        # reader run ahead of the compute. Worth a measured 5-6%, for L1 pages and
        # nothing else -- these shipped single-buffered, which is what a single/double
        # sweep turned up. The state CBs below are a different matter: their 2x is an
        # aliasing requirement, and halving THEM deadlocks.
        dfb("k", stream_buffering * dt * sk),
        dfb("v", stream_buffering * sk * dt),
        dfb("mask", stream_buffering * scores_pages),
        dfb("one", 1),
        dfb("col_ones", sk),
        dfb("masked", scores_pages),
        dfb("row_max", vec_pages),
        dfb("prob", scores_pages),
        dfb("row_sum", vec_pages),
        dfb("p_v", out_pages),
        dfb("o_scaled", out_pages),
        dfb("corr_old", vec_pages),
        dfb("recip_l", vec_pages),
        dfb("m_now", vec_pages),
        dfb("out", out_pages),
        # State: TWICE the block, so store() can reserve while the old value is still resident.
        dfb("m", 2 * vec_pages),
        dfb("l", 2 * vec_pages),
        dfb("o", 2 * out_pages),
    ]

    tensors = {"q": tq, "k": tk, "v": tv, "mask": tmask, "colones": tcolones, "out": tout}
    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        tensors=tensors,
        named_compile_time_args=named_ct_args,
        runtime_arg_names=["head_begin", "head_count"],
        defines=None if causal else [("FLASH_NONCAUSAL", "1")],
        name="flash_attention",
        **(fidelity or {}),
    )
    run_unified_spec(
        device,
        spec,
        tensors,
        # Same everywhere except the head range, which is this core's share.
        runtime_args={
            "head_begin": {c: b for c, (b, _) in zip(core_list, shares)},
            "head_count": {c: n for c, (_, n) in zip(core_list, shares)},
        },
    )
    out = tout
    # Free the inputs: run() is called many times in one device session, and without this
    # the accumulated allocations exhaust DRAM partway through the sweep rather than at a
    # boundary that would make the cause obvious.
    for t in (tq, tk, tv, tmask, tcolones):
        ttnn.deallocate(t)
    return ttnn.to_torch(out).to(torch.float32)[0, 0]


def run(
    device,
    sq,
    sk_total,
    dt,
    num_chunks,
    causal,
    seed=0,
    fidelity=None,
    stream_buffering=2,
    num_q=1,
    n_heads=1,
    n_kv_heads=1,
    cores=1,
):
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
    assert n_heads % n_kv_heads == 0, "every KV head must serve the same number of query heads"
    kv_group = n_heads // n_kv_heads
    S_q, S_k, D = num_q * sq * TILE, sk_total * TILE, dt * TILE

    torch.manual_seed(seed)
    # GQA: n_heads query heads over n_kv_heads key/value heads, query head h reading KV head
    # h // kv_group -- the same mapping the kernel does, written here independently so the
    # test would catch the kernel disagreeing with it rather than sharing the mistake.
    #
    # Each head gets its OWN random data. Identical heads would make a head-indexing bug
    # invisible: reading the wrong head would return the right answer.
    #
    # Q carries the 1/sqrt(d) scale: folding it in on the host costs one multiply here and
    # saves a broadcast pass per chunk on device.
    q_raws = [torch.rand([S_q, D]) - 0.5 for _ in range(n_heads)]

    # The keys are RAMPED along the sequence so later ones score much higher. Without this the
    # test is vacuous: with uniform random inputs every chunk's row maximum is nearly the same,
    # so exp(m_old - m_new) is already 1 and forcing the correction to 1 changes nothing. This
    # measured 0.005 against a 0.02 tolerance -- a broken rescale sailed through.
    #
    # The ramp is a function of the key's position in the FULL sequence, not of the chunk it
    # lands in, so every chunk count sees the same problem and invariance still means something.
    ramp = 1.0 + 20.0 * (torch.arange(S_k, dtype=torch.float32) / S_k)
    ks = [((torch.rand([S_k, D]) - 0.5) * ramp.unsqueeze(1)).to(torch.bfloat16) for _ in range(n_kv_heads)]
    vs = [(torch.rand([S_k, D]) - 0.5).to(torch.bfloat16) for _ in range(n_kv_heads)]

    if causal:
        # K may carry context the queries did not produce, so query i sees keys up to
        # i + (S_k - S_q) -- ordinary prefill-with-history.
        off = S_k - S_q
        keep = torch.arange(S_k).unsqueeze(0) <= (torch.arange(S_q) + off).unsqueeze(1)
    else:
        keep = torch.ones([S_q, S_k], dtype=torch.bool)
    mask = torch.where(keep, 0.0, MASK_NEG)

    # Head-major over chunk-major, which is the layout the kernel's two strides assume: a
    # head's blocks are contiguous, and within a head each chunk's slice is contiguous. K and
    # V are laid out ONCE over the whole key range per head: every query chunk reads a prefix
    # of the same chunks, so no per-query-chunk copy is needed.
    q = torch.cat([(qr / (D**0.5)).to(torch.bfloat16) for qr in q_raws], dim=0)
    k_dev = torch.cat(
        [
            grid_transpose(kh[j * sk * TILE : (j + 1) * sk * TILE].to(torch.float32), sk, dt)
            for kh in ks
            for j in range(num_chunks)
        ],
        dim=0,
    ).to(torch.bfloat16)
    v_dev = torch.cat(vs, dim=0)

    # The mask, in exactly the order the kernel consumes it: for each query chunk, one
    # block per key chunk it actually visits. The kernel walks a flat counter, so this
    # sequence IS the indexing -- there is no (i, j) arithmetic on either side. ONE copy
    # serves every head, because a causal mask does not depend on the head and the kernel
    # restarts its counter per head.
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
    # Pre-filled with NaN rather than allocated uninitialised, so a block NOTHING writes is
    # unmistakable. This is not hygiene, it is load-bearing: run() is called many times per
    # session and the allocator hands back the same address, so an uninitialised output
    # holds the PREVIOUS run's correct values. A partition that dropped a head passed at
    # 8 heads over 8 cores and over 4 -- the stale block read as right -- and was only
    # caught at 4 over 2, where the preceding run had left something different there. NaN
    # propagates into the error, so a missed write now fails everywhere.
    # ONE [S_q, d_model] activation tensor, head h in columns [h*D, (h+1)*D) -- the heads
    # concatenated by the writer as it stores them, not stacked head-major. That is the
    # layout the output projection and everything after it wants.
    tout = to_dev(torch.full([S_q, n_heads * D], float("nan")).to(torch.bfloat16))

    got = _launch(
        device,
        tq,
        tk,
        tv,
        tmask,
        tcolones,
        tout,
        sq,
        sk,
        dt,
        num_q,
        k_offset,
        sk_total,
        n_heads,
        n_kv_heads,
        causal,
        cores,
        stream_buffering,
        fidelity,
    )

    # The reference divides the scores, where the device pre-divided Q: mathematically the same,
    # and the rounding difference shows up in the error rather than being hidden. One head at a
    # time, through the same h // kv_group mapping, stacked in the same head-major order.
    wants = []
    for h in range(n_heads):
        qf = q_raws[h].to(torch.float32)
        kf = ks[h // kv_group].to(torch.float32)
        vf = vs[h // kv_group].to(torch.float32)
        scores = qf @ kf.T / (D**0.5) + mask
        wants.append(torch.softmax(scores, dim=-1) @ vf)
    # Concatenated along the COLUMNS, matching the layout the kernel writes.
    return got, torch.cat(wants, dim=1)


def run_preloaded(device, q_blocks, k_blocks, v_blocks, sq, sk, dt, n_heads, n_kv_heads):
    """The attention kernel on operands SOMEONE ELSE produced, for composing a layer.

    q_blocks / k_blocks / v_blocks are already in the per-(head, chunk) block layout the
    kernel reads -- run() builds those from random data, this takes them as given. One query
    chunk against one key chunk, causal, which is the shape a single-chunk prefill has.

    The mask and the column of ones are built here rather than passed: they are functions of
    the shape, not of the data, so a caller has nothing to say about them.
    """
    S_q, S_k = sq * TILE, sk * TILE
    keep = torch.arange(S_k).unsqueeze(0) <= (torch.arange(S_q) + (S_k - S_q)).unsqueeze(1)
    mask = torch.where(keep, 0.0, MASK_NEG)
    col_ones = torch.zeros([sk * TILE, TILE])
    col_ones[:, 0] = 1.0

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )

    tq, tk, tv = to_dev(q_blocks), to_dev(k_blocks), to_dev(v_blocks)
    tmask, tcolones = to_dev(mask), to_dev(col_ones)
    tout = to_dev(torch.full([S_q, n_heads * dt * TILE], float("nan")))
    return _launch(
        device,
        tq,
        tk,
        tv,
        tmask,
        tcolones,
        tout,
        sq,
        sk,
        dt,
        1,
        0,
        sk,
        n_heads,
        n_kv_heads,
        True,
        1,
        2,
        None,
    )


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

        # GQA: several query heads per KV head, in one launch. Every head gets its own
        # random data, so reading the wrong one is a wrong answer rather than a coincidence
        # -- with identical heads a broken mapping would return the right result. Sabotaging
        # the kernel's h // kv_group to h % n_kv_heads takes these to 0.46 error, so they do
        # bite; the n_kv=1 rows are the exception and cannot, since with one KV head every
        # mapping selects it.
        for n_heads, n_kv in ((2, 1), (2, 2), (4, 1), (4, 2), (4, 4), (8, 2)):
            got, want = run(device, sq, sk_total, dt, 2, True, n_heads=n_heads, n_kv_heads=n_kv)
            e = (got - want).abs().max().item()
            ok = e <= args.abs_err and tuple(got.shape) == (sq * TILE, n_heads * dt * TILE)
            logger.info(
                f"GQA n_heads={n_heads} n_kv={n_kv} (group {n_heads // n_kv}): "
                f"max|err|={e:.5f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"gqa-{n_heads}-{n_kv}")

        # Heads partitioned across cores. Uneven splits are the interesting ones: 8 over 3
        # is 3, 3, 2 and 6 over 4 is 2, 2, 1, 1, so a core's range is not simply its index.
        for n_heads, n_kv, ncores in ((4, 2, 2), (4, 2, 4), (8, 2, 8), (8, 1, 3), (6, 3, 4), (16, 4, 8)):
            got, want = run(device, sq, sk_total, dt, 2, True, n_heads=n_heads, n_kv_heads=n_kv, cores=ncores)
            e = (got - want).abs().max().item()
            ok = e <= args.abs_err
            logger.info(
                f"multicore heads={n_heads} kv={n_kv} cores={ncores}: max|err|={e:.5f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"multicore-{n_heads}-{n_kv}-{ncores}")

        # The partition must not change the answer, so every core count has to agree
        # EXACTLY -- heads are independent, so this is not a tolerance question. Note what
        # this can and cannot catch: a partition that MISSES a head shows up here, since
        # nothing writes that output block, but one that OVERLAPS does not, because two
        # cores computing the same head both write the same correct values. Over-coverage
        # shows up as a missing speedup instead, which is why the scaling numbers in
        # unified_llama_prefill.md are part of the evidence and not just decoration.
        ref = run(device, sq, sk_total, dt, 2, True, n_heads=8, n_kv_heads=2, cores=1)[0]
        for ncores in (2, 4, 8):
            other = run(device, sq, sk_total, dt, 2, True, n_heads=8, n_kv_heads=2, cores=ncores)[0]
            spread = (other - ref).abs().max().item()
            ok = spread == 0.0
            logger.info(f"PARTITION INVARIANCE 1 vs {ncores} cores: max|diff|={spread:.6f}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"partition-invariance-{ncores}")
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
