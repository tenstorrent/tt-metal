# SPDX-License-Identifier: Apache-2.0
"""Two representative decoder kinds composed directly, without boundary conversion."""

import argparse
import gc
import json
import os

import torch

import ttnn

from .multichip_coverage import OUT, Harness
from .optimized_coverage import runtime_audit
from .reference import load_weights


def run(a):
    torch.set_num_threads(8)
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    h = Harness(0, a.baseline, reference_name="stack_baseline")
    try:
        options = {} if a.baseline else {"policy": h.model.policy, "collective_workspace": h.model.collective_workspace}
        second = type(h.model).from_state_dict(
            load_weights(4), hf_config=h.cfg, layer_idx=4, mesh_device=h.mesh, **options
        )
        if not a.baseline:
            assert second.collective_workspace is h.model.collective_workspace
            assert second.ar_buffers[0] is h.model.ar_buffers[0]
        second_cache = tuple(
            h.tt(torch.zeros(h.blocks, 4, 32, 128, dtype=torch.bfloat16), ttnn.bfloat8_b, dim=1) for _ in range(2)
        )
        for length, batch in ((33, 3), (129, 1), (4097, 1)):
            pages = h.pages(101, batch=batch)
            inp = h.inp(h.input(length, batch=batch)[None])
            p0 = h.model.prepare_prefill(page_table_host=pages, seq_len=length)
            p4 = second.prepare_prefill(page_table_host=pages, seq_len=length)
            with runtime_audit():
                middle = h.model.prefill_forward(inp, kv_cache=h.cache, plan=p0)
                out = second.prefill_forward(middle, kv_cache=second_cache, plan=p4)
            h.check(f"prefill_{length}_b{batch}", h.read(out))
            del inp, middle, out, p0, p4
        inp = h.inp(h.input(1, 4097)[None])
        c, s = h.angles(torch.tensor([4097]))
        kw = dict(
            page_table=h.integer(pages),
            current_pos=h.integer(torch.tensor([4097], dtype=torch.int32)),
            cos=h.tt(c),
            sin=h.tt(s),
        )

        def decode():
            middle = h.model.decode_forward(inp, kv_cache=h.cache, **kw)
            return second.decode_forward(middle, kv_cache=second_cache, **kw)

        for _ in range(2):
            with runtime_audit():
                out = decode()
            del out
        gc.collect()
        ttnn.synchronize_device(h.mesh)
        tid = ttnn.begin_trace_capture(h.mesh, cq_id=0)
        with runtime_audit():
            out = decode()
        ttnn.end_trace_capture(h.mesh, tid, cq_id=0)
        try:
            for step in range(5):
                pos = 4097 + step
                h.copy(h.input(1, pos)[None], inp, dim=3 if h.sharded else None)
                h.copy(torch.tensor([pos], dtype=torch.int32), kw["current_pos"])
                c, s = h.angles(torch.tensor([pos]))
                h.copy(c, kw["cos"])
                h.copy(s, kw["sin"])
                ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                value = h.read(out)
                h.check(f"decode_{pos}", value)
                ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                assert torch.equal(value, h.read(out))
        finally:
            ttnn.release_trace(h.mesh, tid)
        if a.baseline:
            torch.save(h.outputs, OUT / "stack_baseline_0.pt")
        (OUT / f"{a.tag}.json").write_text(
            json.dumps(
                dict(
                    baseline=a.baseline,
                    rows=h.rows,
                    layers=[0, 4],
                    layout="replicated2560" if not h.sharded else "sharded640",
                    boundary_conversions=0,
                    shared_collective_workspace=not a.baseline,
                    runtime_audit=True,
                    tracking=1,
                    trace_replays=10,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        h.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--tag", default="stack")
    run(p.parse_args())
