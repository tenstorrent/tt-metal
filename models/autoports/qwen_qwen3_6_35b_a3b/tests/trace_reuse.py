# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prepared eager prefill interleaved with one live, refreshed decode trace."""

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.decoder_factory import Decoder as FunctionalDecoder
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, layer, pcc, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_decoder import hf_forward
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only


def run(args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    if args.output:
        record(args.output)
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    torch.set_num_threads(8)
    torch.manual_seed(76)
    c = config()
    sd = weights(args.layer)
    hf = layer(sd, args.layer)
    requests = []
    for n, start in ((31, 0), (32, 0), (33, 0), (255, 0), (257, 0), (7, 0), (50, 33), (49, 32)):
        x = torch.randn(1, n, 2048, dtype=torch.bfloat16) * 0.05
        xd = torch.randn(1, 1, 2048, dtype=torch.bfloat16) * 0.05
        cache = DynamicCache(config=c)
        hf_forward(hf, x, cache, 0)
        ref = hf_forward(hf, xd, cache, n)
        requests.append(dict(n=n, start=start, x=x, xd=xd, ref=ref))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        dec = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        state = dec.new_recurrent_state() if dec.kind == "linear_attention" else dec.new_kv_state(context_length=512)
        keys = ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v")
        zero = {k: ttnn.clone(state[k]) for k in keys}

        def reset():
            for k in keys:
                ttnn.copy(zero[k], state[k])

        def upload(x):
            return ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        td = upload(requests[0]["xd"])
        pos = dec.prepare_positions(start_pos=31, seq_len=1) if dec.kind == "full_attention" else None
        for r in requests:
            start = r["start"]
            r["inputs"] = [upload(r["x"])] if not start else [upload(r["x"][:, :start]), upload(r["x"][:, start:])]
            r["plans"] = (
                [dec.prepare_prefill(seq_len=r["n"])]
                if not start
                else [dec.prepare_prefill(seq_len=start), dec.prepare_prefill(seq_len=r["n"] - start, start_pos=start)]
            )
            r["pages"] = torch.randperm(16).int().reshape(1, -1)

        def prefill(r):
            for tx, plan in zip(r["inputs"], r["plans"]):
                with device_only():
                    out = dec.prefill_forward(tx, state=state, plan=plan)
                del out

        for r in requests:
            reset()
            prefill(r)
        # Compile restores and the identical decode signature before capture.
        reset()
        with device_only():
            warm = dec.decode_forward(td, state=state, positions=pos)
        del warm
        reset()
        clone_warm = ttnn.clone(td)
        del clone_warm
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            out = dec.decode_forward(td, state=state, positions=pos)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        from ttnn.tools import trace_allocation_tracker

        trace_allocation_tracker.acknowledge_corruptible(out)
        # Positive control: a live, non-exempt allocation must be rejected.
        hazard = ttnn.clone(td)
        try:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        except RuntimeError as exc:
            assert "allocat" in str(exc).lower(), str(exc)
            print("TRACKER_POSITIVE_CONTROL_REJECTED", str(exc).splitlines()[0], flush=True)
        else:
            raise AssertionError("allocation tracker failed to reject live hazard")
        del hazard
        gc.collect()

        def refresh(dest, t, dtype, layout):
            host = ttnn.from_torch(t, dtype=dtype, layout=layout)
            ttnn.copy_host_to_device_tensor(host, dest)

        rows = []
        outputs = []
        order = list(range(len(requests))) + [0]
        for idx in order:
            r = requests[idx]
            reset()
            if pos is not None:
                refresh(state["page_table"], r["pages"], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            prefill(r)
            refresh(td, r["xd"], ttnn.bfloat16, ttnn.TILE_LAYOUT)
            if pos is not None:
                inv = 1 / (c.rope_parameters["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64))
                a = r["n"] * inv
                a = torch.cat([a, a]).reshape(1, 1, 1, 64)
                refresh(pos["cos"], a.cos(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                refresh(pos["sin"], a.sin(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                refresh(
                    pos["current_pos"], torch.tensor([r["n"]], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                )
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            got = ttnn.to_torch(out).reshape(1, 1, 2048).clone()
            score = pcc(r["ref"], got)
            assert score >= 0.995, (idx, score)
            outputs.append(got)
            rows.append(dict(request=idx, length=r["n"], resume_start=r["start"], trace_id=str(trace), pcc=score))
            print("REUSE", rows[-1], flush=True)
        assert torch.equal(outputs[0], outputs[-1])
        assert not torch.equal(outputs[0], outputs[1])
        ttnn.release_trace(mesh, trace)
        Path(args.output).write_text(
            json.dumps(
                dict(
                    layer=args.layer,
                    tracking=True,
                    program_cache_included=True,
                    positive_control="rejected",
                    captures=1,
                    request_releases=0,
                    teardown_releases=1,
                    results=rows,
                    repeated_request_bit_equal=True,
                    changed_input_changed_output=True,
                    acknowledged_buffers=[
                        "trace output: read before next replay; only one live trace; no surviving aliases"
                    ],
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())
