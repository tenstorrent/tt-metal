# SPDX-License-Identifier: Apache-2.0
"""Matched 1M-position prefix, circular-cache and full-stack DRAM reservation checks.

Every prefix key/value is computed by the tested TTNN decoder. Complete decoder
checks sample long-context chunks and tails; this is not full-model execution.
"""

import argparse
import gc
import hashlib
import json
import os
import time

import torch

import ttnn

from .multichip_coverage import OUT, ROOT, Harness
from .optimized_coverage import runtime_audit


def memory(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return dict(
        banks=v.num_banks,
        allocated_per_bank=v.total_bytes_allocated_per_bank,
        available_per_bank=v.total_bytes_per_bank,
        largest_free_per_bank=v.largest_contiguous_bytes_free_per_bank,
    )


def run(a):
    torch.set_num_threads(8)
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") != "1"
    capacity = 1048576
    largest_chunk = int(os.environ.get("MC_CONTEXT_CHUNK", "4096"))
    ring_tokens = int(os.environ.get("MC_CONTEXT_RING", "4608"))
    h = Harness(
        a.layer, a.baseline, capacity, ring=not a.baseline, ring_tokens=ring_tokens, reference_name="context_baseline"
    )
    reservations = []
    before = memory(h.mesh)
    if not a.baseline:
        # Reserve the entire documented stack budget in addition to the real
        # layer already loaded. This deliberately double-counts this layer's
        # weights/cache and keeps the 2GiB activation/trace allowance untouched.
        plan = json.loads((OUT / "memory_capacity_plan.json").read_text())
        reserve = plan["planned_bytes_per_device"] - plan["reserved_trace_activation_ccl_bytes"]
        chunk = 128 * 1024 * 1024
        for _ in range((reserve + chunk - 1) // chunk):
            reservations.append(
                ttnn.empty(
                    (1, 1, 1024, 65536),
                    device=h.mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
    reserved = memory(h.mesh)
    start = time.monotonic()
    try:
        pages = h.pages(19023)
        table = h.integer(pages)
        seed_in = h.inp(h.input(4096)[None])
        c, s = h.angles(torch.arange(4096))
        seed_cos, seed_sin = h.tt(c), h.tt(s)
        seed_pages = h.integer(pages[:, :128])

        def seed():
            q, k, v = h.model._qkv(seed_in, seed_cos, seed_sin)
            ttnn.experimental.paged_fill_cache(h.cache[0], ttnn.typecast(k, h.cache[0].dtype), seed_pages)
            ttnn.experimental.paged_fill_cache(h.cache[1], ttnn.typecast(v, h.cache[1].dtype), seed_pages)

        def prefix(begin, end):
            for off in range(begin, end, 4096):
                h.copy(h.input(4096, off)[None], seed_in, dim=3 if h.sharded else None)
                c, s = h.angles(torch.arange(off, off + 4096))
                h.copy(c, seed_cos)
                h.copy(s, seed_sin)
                h.copy(pages[:, off // 32 : (off + 4096) // 32], seed_pages)
                with runtime_audit():
                    seed()
            ttnn.synchronize_device(h.mesh)
            print("PREFIX_READY", end, flush=True)

        # Warm the largest chunk with all capacity reservations alive.
        plan = h.model.prepare_prefill(page_table_host=pages, seq_len=largest_chunk)
        warm_input = h.inp(h.input(largest_chunk)[None])
        with runtime_audit():
            out = h.model.prefill_forward(warm_input, kv_cache=h.cache, plan=plan)
        del out, plan, warm_input
        # Sequential progression makes ring residency explicit, including
        # wraps inside4096-token fills and an unaligned continuation.
        prefix(0, 65536)

        def prefill(pos, count):
            inp = h.inp(h.input(count, pos)[None])
            plan = h.model.prepare_prefill(page_table_host=pages, seq_len=count, start_pos=pos)
            with runtime_audit():
                out = h.model.prefill_forward(inp, kv_cache=h.cache, plan=plan)
            h.check(f"prefill_{pos}_{count}", h.read(out))
            del out, inp, plan
            h.cache_check(f"cache_{pos}_{count}", pages, pos + count)

        prefill(65536, 129)
        # Refill starting65536 to restore an exact recorded-input prefix.
        prefix(65536, capacity - 8192)
        if largest_chunk == 8192 and not a.baseline:
            # Concatenate contiguous accepted reference outputs, without changing
            # their values or rerunning a candidate as its own reference.
            label = f"prefill_{capacity - 8192}_8192"
            h.reference[label] = torch.cat(
                [
                    h.reference[f"prefill_{capacity - 8192}_4096"],
                    h.reference[f"prefill_{capacity - 4096}_4079"],
                    h.reference[f"prefill_{capacity - 17}_17"],
                ],
                dim=2,
            )
            for role in ("k", "v"):
                h.reference[f"cache_{capacity - 8192}_8192_{role}"] = h.reference[f"cache_{capacity - 17}_17_{role}"]
            prefill(capacity - 8192, 8192)
        prefill(capacity - 8192, 4096)
        prefill(capacity - 4096, 4079)
        prefill(capacity - 17, 17)
        # Capture at the final valid position; mutate a previously captured
        # position and input to test replay runtime-argument ownership.
        inp = h.inp(h.input(1, capacity - 1)[None])
        c, s = h.angles(torch.tensor([capacity - 1]))
        kw = dict(
            kv_cache=h.cache,
            page_table=table,
            current_pos=h.integer(torch.tensor([capacity - 1], dtype=torch.int32)),
            cos=h.tt(c),
            sin=h.tt(s),
        )
        for _ in range(2):
            with runtime_audit():
                out = h.model.decode_forward(inp, **kw)
            del out
        gc.collect()
        ttnn.synchronize_device(h.mesh)
        tid = ttnn.begin_trace_capture(h.mesh, cq_id=0)
        with runtime_audit():
            out = h.model.decode_forward(inp, **kw)
        ttnn.end_trace_capture(h.mesh, tid, cq_id=0)
        try:
            for repeat in range(3):
                h.copy(h.input(1, capacity - 1 + repeat)[None], inp, dim=3 if h.sharded else None)
                ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                value = h.read(out)
                h.check(f"decode_end_{repeat}", value)
                ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                assert torch.equal(value, h.read(out))
        finally:
            ttnn.release_trace(h.mesh, tid)
        peak = memory(h.mesh)
        if a.baseline:
            torch.save(h.outputs, OUT / f"context_baseline_{a.layer}.pt")
        result = dict(
            layer=a.layer,
            baseline=a.baseline,
            capacity=capacity,
            physical_cache_tokens=h.blocks * 32,
            prefix="all rows computed with tested TTNN QKV/norm/RoPE",
            rows=h.rows,
            reservation_buffers=len(reservations),
            reservation_bytes=len(reservations) * 128 * 1024 * 1024,
            memory_before=before,
            memory_reserved=reserved,
            memory_end=peak,
            tracking=1,
            skip_program_cache=0,
            trace_replays=6,
            largest_physical_chunk=largest_chunk,
            runtime_audit=True,
            seconds=time.monotonic() - start,
            source_sha256=hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        )
        (OUT / f"{a.tag}_{a.layer}.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        h.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--tag", default="context")
    run(p.parse_args())
