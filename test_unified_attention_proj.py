# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Head concat and the output projection, on device, fed by the real attention kernel.

The projection is an accumulating matmul whose k-blocks are the HEADS:

    concat(O_0 .. O_{H-1}) @ Wo  ==  sum over h of  O_h @ Wo_h

so there is no concat pass and no rearranging of anything. The attention kernel writes
head h's query chunk i at block h * num_q_chunks + i, and the projection's k-loop reads
exactly those blocks. See unified_kernels/attention_proj.cpp.

Two things are checked, and the second is the one that matters:

  proj      the projection alone, on random heads -- shape coverage, cheap to sweep.
  chained   the projection on the ACTUAL output of unified_kernels/flash_attention.cpp,
            against a torch reference that starts from Q, K and V. Nothing but the layout
            contract connects the two kernels, so this is what would catch the two of them
            disagreeing about where head h's chunk i lives.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python test_unified_attention_proj.py
"""

import argparse
import sys

import torch
from loguru import logger

import ttnn
import test_unified_flash as flash
from unified_harness import core_block, make_cb, split_evenly, unified_program

KERNEL = "unified_kernels/attention_proj.cpp"
CB_IN0, CB_IN1, CB_OUT, CB_ACC = 0, 1, 16, 24
TILE = 32


def project(device, attn_torch, wo_torch, sq, dt, num_q, n_heads, cores=1, fidelity=None):
    """attn_torch is head-major [n_heads * num_q * sq * TILE, dt * TILE]; wo is square."""
    dm = n_heads * dt
    assert attn_torch.shape == (n_heads * num_q * sq * TILE, dt * TILE), attn_torch.shape
    assert wo_torch.shape == (dm * TILE, dm * TILE), wo_torch.shape

    dram = ttnn.DRAM_MEMORY_CONFIG

    def to_dev(t):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram
        )

    tattn, two = to_dev(attn_torch.to(torch.bfloat16)), to_dev(wo_torch.to(torch.bfloat16))
    # NaN-filled, not uninitialised: the allocator reuses addresses across runs, so an
    # output block nothing writes would otherwise hold a previous run's correct values and
    # a missed chunk would pass. This is the same hole that hid a dropped head in the
    # flash harness until it was fixed there.
    tout = to_dev(torch.full([num_q * sq * TILE, dm * TILE], float("nan")).to(torch.bfloat16))

    # Query chunks are the unit of work here, not heads: a head is a k-block of this
    # matmul, so splitting heads across cores would leave partial sums to reduce. Chunks
    # are independent rows of the output.
    ncores = min(cores, num_q)
    core_ranges, core_list = core_block(ncores)
    shares = split_evenly(num_q, ncores)

    ct_args = [sq, dt, dm, num_q, n_heads]
    for t in (tattn, two, tout):
        ct_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    addrs = [t.buffer_address() for t in (tattn, two, tout)]
    rt_args = {c: addrs + [begin, count] for c, (begin, count) in zip(core_list, shares)}

    cbs = [
        make_cb(CB_IN0, core_ranges, num_pages=sq * dt),
        make_cb(CB_IN1, core_ranges, num_pages=dt * dm),
        make_cb(CB_OUT, core_ranges, num_pages=sq * dm),
        # The running total, a separate CB from the output: the accumulator re-reads it
        # every k-block, and one CB with two poppers is the bug its api.h note warns about.
        make_cb(CB_ACC, core_ranges, num_pages=sq * dm),
    ]

    program = unified_program(
        kernel_source=KERNEL,
        core_ranges=core_ranges,
        cores=core_list,
        cbs=cbs,
        compile_time_args=ct_args,
        runtime_args=rt_args,
        **(fidelity or {}),
    )
    logger.info(f"projection: sq={sq} dt={dt} dm={dm} num_q={num_q} n_heads={n_heads} cores={ncores}")
    out = ttnn.generic_op([tattn, two, tout], program)
    for t in (tattn, two):
        ttnn.deallocate(t)
    return ttnn.to_torch(out).to(torch.float32)[0, 0]


def reference(attn_torch, wo_torch, sq, dt, num_q, n_heads):
    """The concat this kernel does not do, done explicitly."""
    S_q = num_q * sq * TILE
    heads = [attn_torch[h * S_q : (h + 1) * S_q].to(torch.float32) for h in range(n_heads)]
    return torch.cat(heads, dim=1) @ wo_torch.to(torch.float32)


def run_proj(device, sq, dt, num_q, n_heads, cores=1, seed=0, fidelity=None):
    """The projection alone, on random head outputs."""
    torch.manual_seed(seed)
    dm = n_heads * dt
    # Per-head distinct data, so reading the wrong head's block is a wrong answer rather
    # than a coincidence -- the same reason the flash harness randomises per head.
    attn = (torch.rand([n_heads * num_q * sq * TILE, dt * TILE]) - 0.5).to(torch.bfloat16)
    wo = ((torch.rand([dm * TILE, dm * TILE]) - 0.5) / (dm * TILE) ** 0.5).to(torch.bfloat16)
    got = project(device, attn, wo, sq, dt, num_q, n_heads, cores=cores, fidelity=fidelity)
    return got, reference(attn, wo, sq, dt, num_q, n_heads)


def run_chained(device, sq, sk_total, dt, num_chunks, num_q, n_heads, n_kv_heads, cores=1, seed=0):
    """Attention on device, then the projection on device, against a reference from Q/K/V."""
    torch.manual_seed(seed + 1)
    dm = n_heads * dt
    wo = ((torch.rand([dm * TILE, dm * TILE]) - 0.5) / (dm * TILE) ** 0.5).to(torch.bfloat16)

    # The attention kernel's own output and its own reference, head-major.
    attn_got, attn_want = flash.run(
        device, sq, sk_total, dt, num_chunks, True, num_q=num_q, n_heads=n_heads, n_kv_heads=n_kv_heads, seed=seed
    )
    got = project(device, attn_got, wo, sq, dt, num_q, n_heads, cores=cores)
    # From the attention REFERENCE, not from the device's attention output: the error then
    # covers both kernels rather than treating the first one's output as ground truth.
    return got, reference(attn_want, wo, sq, dt, num_q, n_heads)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pcc", type=float, default=0.99)
    p.add_argument("--abs-err", type=float, default=0.05)
    args = p.parse_args(argv)

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        # The projection alone. n_heads is the k-block count and dm grows with it, so the
        # output block is sq * dm tiles -- 16 at sq=2 with four 64-wide heads, which only
        # the subblocked accumulating path can express at all.
        for sq, dt, num_q, n_heads in ((1, 2, 1, 2), (2, 2, 1, 4), (2, 2, 2, 4), (1, 4, 2, 2), (2, 1, 2, 8)):
            got, want = run_proj(device, sq, dt, num_q, n_heads)
            pcc = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
            ok = pcc >= args.pcc
            logger.info(
                f"proj sq={sq} dt={dt} num_q={num_q} heads={n_heads} (out {sq}x{n_heads * dt}t): "
                f"pcc={pcc:.6f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"proj-{sq}-{dt}-{num_q}-{n_heads}")

        # Query chunks partitioned across cores.
        for num_q, ncores in ((2, 2), (4, 2), (4, 4), (3, 2)):
            got, want = run_proj(device, 2, 2, num_q, 4, cores=ncores)
            pcc = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
            ok = pcc >= args.pcc
            logger.info(f"proj multicore num_q={num_q} cores={ncores}: pcc={pcc:.6f}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"proj-mc-{num_q}-{ncores}")

        # The real thing: both kernels, one after the other, against a reference that
        # starts from Q, K and V.
        for n_heads, n_kv in ((2, 1), (4, 2), (4, 4)):
            got, want = run_chained(device, 2, 4, 2, 2, 1, n_heads, n_kv)
            e = (got - want).abs().max().item()
            ok = e <= args.abs_err
            logger.info(f"chained attention+proj heads={n_heads} kv={n_kv}: max|err|={e:.5f}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"chained-{n_heads}-{n_kv}")

        # And with the query loop on, so both kernels are walking chunks.
        got, want = run_chained(device, 2, 8, 2, 4, 4, 4, 2)
        e = (got - want).abs().max().item()
        ok = e <= args.abs_err
        logger.info(f"chained q-loop num_q=4 heads=4 kv=2: max|err|={e:.5f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failed.append("chained-qloop")
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
