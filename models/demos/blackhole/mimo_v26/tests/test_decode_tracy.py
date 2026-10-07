# SPDX-License-Identifier: Apache-2.0
"""Tracy / device-profiler capture of the REAL full (48-layer) traced decode step with real GSM8K state after prefill (one process, U users per mesh row, several U via Generator.reconfigure).
Run as: python -m tracy -r -p -v --no-web-server -o <dir> --op-support-count N -m models.demos.blackhole.mimo_v26.tests.test_decode_tracy   (or via launch.py, see bringup/mimo_v26/tools/tracy_report.py docstring)
Per U: prefill 4U prompts, set inputs from the prefill first tokens (pos = prompt length), Decoder.run() once (eager compile + trace capture + 1 replay), TR_WARM extra replays, flush the device profiler,
then TR_REPS measured replays each bracketed by tracy signposts 'U<U>_rep<i>_start' / '_end' and followed by ttnn.ReadDeviceProfiler. Host wall of each replay (execute_trace blocking) is printed (TRACY_HOST json lines).
Writes <TR_OUT>/segments.json = [{U, rep, trace_id, host_ms}].
Env: TR_US (1,8) TR_LAYERS (48) TR_REPS (2) TR_WARM (2) TR_OUT"""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get


def sp(header):
    try:
        from tracy import signpost

        signpost(header=header)
    except Exception as e:  # noqa
        print("SIGNPOST_FAIL", header, e, flush=True)


def main():
    Us = [int(v) for v in E("TR_US", "1,8").split(",")]
    NL = int(E("TR_LAYERS", "48"))
    reps, warm = int(E("TR_REPS", "2")), int(E("TR_WARM", "2"))
    out_dir = E("TR_OUT", E("TT_METAL_PROFILER_DIR", "/mnt/tt-data/ssinghal/mimo-logs/tracy_out"))
    os.makedirs(out_dir, exist_ok=True)
    tok = tokenizer()
    items = gsm_prompts(4 * max(Us), tok)
    S, max_pos = 256, 512
    mesh = open_mesh_full()
    segs = []
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=KVLayout(users=Us[0], max_pos=max_pos, chunk=S), log=lambda s: None)
        gen = Generator(model, max_pos, trace=True, keep_logits=False, U=Us[0])
        for k, U in enumerate(Us):
            if k > 0:
                gen.reconfigure(U, KVLayout(users=U, max_pos=max_pos, chunk=S))
            prompts = [it["prompt_ids"] for it in items[: 4 * U]]
            first, _ = gen.prefill(prompts, S)
            rows = 4
            cur = torch.zeros(rows, U, dtype=torch.long)
            pos = torch.zeros(rows, U, dtype=torch.long)
            for i, (r, s) in enumerate(gen.place(len(prompts))):
                cur[r, s] = first[r, s]
                pos[r, s] = len(prompts[i])
            dec = gen.dec
            dec.set_inputs(cur, pos, torch.zeros(rows, U, dtype=torch.bool))
            sp(f"U{U}_compile_capture_begin")
            # same sequence as Decoder.run() on first call, but with a profiler flush after each phase: one full decode is ~5000 ops and the per-core profiler DRAM buffer holds
            # TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT programs, so eager warmup + capture + replay in one read window overflows it.
            dec._forward()  # eager warmup/compile
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            os.environ["MIMO_CHECK_CAP"] = "0"
            dec._tr = ttnn.begin_trace_capture(mesh, cq_id=0)
            dec.out = dec._forward()
            ttnn.end_trace_capture(mesh, dec._tr, cq_id=0)
            ttnn.synchronize_device(mesh)
            sp(f"U{U}_compile_capture_end")
            ttnn.ReadDeviceProfiler(mesh)
            tr = dec._tr
            for _ in range(warm):
                ttnn.execute_trace(mesh, tr, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(mesh)
            for i in range(reps):
                sp(f"U{U}_rep{i}_start")
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tr, cq_id=0, blocking=True)
                ms = (time.perf_counter() - t0) * 1e3
                sp(f"U{U}_rep{i}_end")
                t1 = time.perf_counter()
                ttnn.ReadDeviceProfiler(mesh)
                rd = (time.perf_counter() - t1) * 1e3
                segs.append(dict(U=U, rep=i, trace_id=int(tr), host_ms=ms, read_profiler_ms=rd))
                print("TRACY_HOST", json.dumps(segs[-1]), flush=True)
            json.dump(segs, open(os.path.join(out_dir, "segments.json"), "w"), indent=1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
