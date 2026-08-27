# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The output projection, on device, fed by the real attention kernel.

The attention kernel writes its heads already concatenated -- head h's query chunk as an
[sq, dt] rectangle at columns [h*dt, +dt) of one [S_q, d_model] tensor -- so the projection
is an ordinary [S_q, d_model] @ Wo matmul with no concat and no k-loop over heads. The
earlier design left the output head-major and recovered the concat as a k-loop, which cost
30% of the projection; see unified_kernels/matmul_blocked.cpp.

Two things are checked, and the second is the one that matters:

  proj      the projection alone, on a random activation -- shape coverage, cheap to sweep.
  chained   the projection on the ACTUAL output of unified_kernels/flash_attention.cpp,
            against a torch reference that starts from Q, K and V. Nothing but the layout
            contract connects the two kernels, so this is what would catch them disagreeing
            about where head h's columns live.

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
from unified_harness import core_block, dfb, run_unified_spec, split_evenly, unified_program_spec

KERNEL = "unified_kernels/matmul_blocked.cpp"
TILE = 32


def project(device, attn_torch, wo_torch, sq, dt, num_q, n_heads, cores=1, fidelity=None, kt=None, nt=None, acc="l1"):
    """attn_torch is the concatenated [num_q * sq * TILE, d_model] activation; wo is square."""
    dm = n_heads * dt
    assert attn_torch.shape == (num_q * sq * TILE, dm * TILE), attn_torch.shape
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

    # kt is the k-block width in tiles. Default: the whole of d_model in one block, which is
    # the single-shot path -- correct and fastest while Wo fits in L1 (dm*dm tiles). It does
    # not fit for long: 64 tiles at d_model 256, 4096 (8MB) at 2048, so a real d_model has to
    # pass a smaller kt and let the matmul accumulate over the blocks.
    kt = dm if kt is None else kt
    nt = dm if nt is None else nt
    assert dm % kt == 0, "the k-block width must divide d_model"
    assert dm % nt == 0, "the output-column block width must divide d_model"
    # The unit of work across cores is one OUTPUT BLOCK -- an (m, n) tile -- not a query
    # chunk. Heads are a k-block of this matmul, so splitting THOSE would leave partial sums
    # to reduce; different m or different n write disjoint output and need nothing.
    nunits = num_q * (dm // nt)
    ncores = min(cores, nunits)
    core_ranges, core_list = core_block(ncores)
    shares = split_evenly(nunits, ncores)

    named_ct_args = [("mt", sq), ("ktot", dm), ("ntot", dm), ("kt", kt), ("nt", nt)]

    dfbs = [
        dfb("in", sq * kt),
        dfb("wo", kt * nt),
        dfb("out", sq * nt),
        dfb("acc", sq * nt),
    ]

    spec = unified_program_spec(
        kernel_source=KERNEL,
        nodes=core_ranges,
        dfbs=dfbs,
        named_compile_time_args=named_ct_args,
        tensors={"attn": tattn, "wo": two, "out": tout},
        runtime_arg_names=["block_begin", "block_count"],
        defines=[("MMB_ACC_DST", "1")] if acc == "dst" else None,
        **(fidelity or {}),
    )
    logger.info(
        f"projection: sq={sq} dm={dm} kt={kt} nt={nt} (kb={dm // kt} nb={dm // nt}) num_q={num_q} cores={ncores}"
    )
    run_unified_spec(
        device,
        spec,
        {"attn": tattn, "wo": two, "out": tout},
        runtime_args={
            # Per core: its slice of the block range. Named, so a launcher that supplied
            # one and not the other is an error from metal rather than a garbage bound.
            "block_begin": {c: b for c, (b, _) in zip(core_list, shares)},
            "block_count": {c: n for c, (_, n) in zip(core_list, shares)},
        },
    )
    out = tout
    for t in (tattn, two):
        ttnn.deallocate(t)
    return ttnn.to_torch(out).to(torch.float32)[0, 0]


def reference(attn_torch, wo_torch, sq, dt, num_q, n_heads):
    """Just the matmul: the concat already happened, in the attention kernel's store."""
    return attn_torch.to(torch.float32) @ wo_torch.to(torch.float32)


def run_proj(device, sq, dt, num_q, n_heads, cores=1, seed=0, fidelity=None, kt=None, nt=None, acc="l1"):
    """The projection alone, on a random activation."""
    torch.manual_seed(seed)
    dm = n_heads * dt
    attn = (torch.rand([num_q * sq * TILE, dm * TILE]) - 0.5).to(torch.bfloat16)
    wo = ((torch.rand([dm * TILE, dm * TILE]) - 0.5) / (dm * TILE) ** 0.5).to(torch.bfloat16)
    got = project(device, attn, wo, sq, dt, num_q, n_heads, cores=cores, fidelity=fidelity, kt=kt, nt=nt, acc=acc)
    return got, reference(attn, wo, sq, dt, num_q, n_heads)


def run_chained(device, sq, sk_total, dt, num_chunks, num_q, n_heads, n_kv_heads, cores=1, seed=0):
    """Attention on device, then the projection on device, against a reference from Q/K/V."""
    torch.manual_seed(seed + 1)
    dm = n_heads * dt
    wo = ((torch.rand([dm * TILE, dm * TILE]) - 0.5) / (dm * TILE) ** 0.5).to(torch.bfloat16)

    # The attention kernel's own output and its own reference, both [S_q, d_model].
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
        # The projection alone. The output block is sq * dm tiles -- 16 at sq=2 with four
        # 64-wide heads -- which the single-shot path walks in subblocks; before subblocking
        # it could not have been expressed at all.
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

        # k-blocked, which is what lets d_model exceed what L1 can hold. Every kt must give
        # the same answer: the blocking is a decomposition, not an approximation.
        base = None
        for kt in (8, 4, 2, 1):
            got, want = run_proj(device, 2, 2, 2, 4, kt=kt)  # dm = 8
            pcc = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
            spread = 0.0 if base is None else (got - base).abs().max().item()
            if base is None:
                base = got
            ok = pcc >= args.pcc and spread <= 0.02
            logger.info(f"proj kt={kt} (kb={8 // kt}): pcc={pcc:.6f} vs-kt8={spread:.5f}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"proj-kt-{kt}")

        # Blocking BOTH dimensions, at dm=8. Every (kt, nt) is a decomposition of the same
        # matmul, so all of them have to agree.
        for kt, nt in ((8, 8), (4, 8), (8, 4), (4, 4), (2, 2), (1, 1)):
            got, want = run_proj(device, 2, 2, 2, 4, kt=kt, nt=nt)
            pcc = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
            ok = pcc >= args.pcc
            logger.info(f"proj 2D kt={kt} nt={nt}: pcc={pcc:.6f}  {'ok' if ok else 'FAIL'}")
            if not ok:
                failed.append(f"proj-2d-{kt}-{nt}")

        # d_model 2048, llama-3.2-1B's width: 32 heads of head_dim 64. Wo is 4096 tiles,
        # 8MB, so this shape exists only because both dimensions are blocked. sq=8/nt=16 is
        # the configuration that measures fastest (3711us against ttnn.matmul's 3963us on
        # the same shape); sq=2/nt=64 is the K-only shape it replaced, kept as a comparison.
        for sq, kt, nt, acc in ((8, 8, 16, "l1"), (16, 8, 8, "l1"), (2, 2, 64, "l1"), (2, 2, 64, "dst")):
            got, want = run_proj(device, sq, 2, 16 // sq, 32, kt=kt, nt=nt, acc=acc)
            pcc = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
            ok = pcc >= args.pcc
            logger.info(
                f"proj d_model=2048 sq={sq} kt={kt} nt={nt} (kb={64 // kt} nb={64 // nt}) acc={acc}: "
                f"pcc={pcc:.6f}  {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(f"proj-2048-{sq}-{kt}-{nt}-{acc}")

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
