# SPDX-License-Identifier: Apache-2.0
"""Three independent circular-cache owners with differing absolute positions."""

import argparse
import gc
import hashlib
import json
import os

import torch

import ttnn

from .multichip_coverage import OUT, ROOT, Harness
from .optimized_coverage import runtime_audit


def run(a):
    torch.set_num_threads(8)
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    h = Harness(0, a.baseline, ring=not a.baseline, reference_name="ring_batch_baseline")
    try:
        batch = 3
        blocks = 512 if a.baseline else 256
        h.cache = tuple(
            h.tt(torch.zeros(batch * blocks, 4, 32, 128, dtype=torch.bfloat16), ttnn.bfloat8_b, dim=1) for _ in range(2)
        )
        gen = torch.Generator().manual_seed(989)
        pages = torch.stack(
            [
                (torch.randperm(blocks, generator=gen, dtype=torch.int32) + r * blocks)[torch.arange(512) % blocks]
                for r in range(batch)
            ]
        )
        # Every key needed by the three sliding windows comes from tested QKV.
        c, s = h.angles(torch.arange(8192, 9216))
        cos, sin = h.tt(c), h.tt(s)
        for r in range(batch):
            inp = h.inp(h.input(1024, 8192 + r * 137)[None])
            part = h.integer(pages[r : r + 1, 256:288])
            with runtime_audit():
                q, k, v = h.model._qkv(inp, cos, sin)
                ttnn.experimental.paged_fill_cache(h.cache[0], ttnn.typecast(k, ttnn.bfloat8_b), part)
                ttnn.experimental.paged_fill_cache(h.cache[1], ttnn.typecast(v, ttnn.bfloat8_b), part)
            del inp, part, q, k, v
        positions = torch.tensor([9001, 9117, 9215], dtype=torch.int32)
        c, s = h.angles(positions)
        inp = h.inp(h.input(1, 901, batch).reshape(1, 1, batch, 2560))
        kw = dict(
            kv_cache=h.cache, page_table=h.integer(pages), current_pos=h.integer(positions), cos=h.tt(c), sin=h.tt(s)
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
            for step in range(3):
                pos = positions + step
                h.copy(pos, kw["current_pos"])
                c, s = h.angles(pos)
                h.copy(c, kw["cos"])
                h.copy(s, kw["sin"])
                h.copy(h.input(1, 901 + step, batch).reshape(1, 1, batch, 2560), inp)
                ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                actual = h.read(out)
                h.check(f"trace_b3_ring_{step}", actual)
                ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                assert torch.equal(actual, h.read(out))
        finally:
            ttnn.release_trace(h.mesh, tid)
        for role, cache in zip(("k", "v"), h.cache):
            cpu = h.read(cache, cache=True)
            for r, end in enumerate((positions + 3).tolist()):
                pos = torch.arange(end - 513, end)
                logical = cpu[pages[r, pos // 32].long(), :, pos % 32, :]
                h.check(f"cache_{role}_owner{r}", logical)
        if a.baseline:
            torch.save(h.outputs, OUT / "ring_batch_baseline_0.pt")
        result = dict(
            baseline=a.baseline,
            batch=3,
            positions=positions.tolist(),
            logical_capacity=16384,
            physical_tokens_per_owner=blocks * 32,
            trace_replays=6,
            rows=h.rows,
            runtime_audit=True,
            tracking=1,
            skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
            watcher=os.environ.get("TT_METAL_WATCHER"),
            watcher_disable_eth=os.environ.get("TT_METAL_WATCHER_DISABLE_ETH"),
            source_sha256=hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
        )
        (OUT / f"{a.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        h.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--tag", default="ring_batch")
    run(p.parse_args())
