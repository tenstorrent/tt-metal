# SPDX-License-Identifier: Apache-2.0
"""Wall-time decomposition of the traced decode step at 48 layers for U users per mesh row (one session, Generator.reconfigure between U):
 trace_blocking   = execute_trace(blocking=True) only (device time + trace dispatch + one host sync)
 trace_pipelined  = N back-to-back non-blocking execute_trace + one sync, per step (device throughput, no host sync per step)
 set_inputs       = host->device input copies + sync
 readback         = candidate readback (to_torch of vals/idx + host argmax)
 run_full         = Decoder.run() (blocking trace + readback)
Env: DEC_US (1,8,32) DEC_LAYERS (48) DEC_REPS (40)."""
import json
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, dram, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/dec_wall.jsonl")


def med(f, n):
    ts = []
    for _ in range(n + 3):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    ts = sorted(ts[3:])
    return round(ts[len(ts) // 2] * 1e3, 3)


def main():
    Us = [int(v) for v in E("DEC_US", "1,8,32").split(",")]
    NL = int(E("DEC_LAYERS", "48"))
    reps = int(E("DEC_REPS", "40"))
    tok = tokenizer()
    items = gsm_prompts(4 * max(Us), tok)
    S, max_pos = 256, 512
    mesh = open_mesh_full()
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=KVLayout(users=Us[0], max_pos=max_pos, chunk=S), log=lambda s: None)
        gen = Generator(model, max_pos, trace=True, keep_logits=False, U=Us[0])
        for k, U in enumerate(Us):
            if k > 0:
                gen.reconfigure(U, KVLayout(users=U, max_pos=max_pos, chunk=S))
            r = gen.generate([it["prompt_ids"] for it in items[: 4 * U]], 24, S=S)
            dec = gen.dec
            # re-set inputs of a mid-generation state: use the last step's inputs (still on device)
            tr = dec._tr
            res = {}
            res["trace_blocking_ms"] = med(lambda: ttnn.execute_trace(mesh, tr, cq_id=0, blocking=True), reps)
            n = 30
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.execute_trace(mesh, tr, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            res["trace_pipelined_ms"] = round((time.perf_counter() - t0) / n * 1e3, 3)
            res["run_full_ms"] = med(lambda: dec.run(), reps)
            ttnn.execute_trace(mesh, tr, cq_id=0, blocking=True)
            comp = ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=tuple(mesh.shape))

            def readback():
                vals, idx, _, _ = dec.out
                ttnn.to_torch(vals, mesh_composer=comp)
                ttnn.to_torch(idx, mesh_composer=comp)

            res["readback_ms"] = med(readback, reps)
            cur = torch.zeros(4, U, dtype=torch.long)
            pos = torch.full((4, U), 100, dtype=torch.long)
            dead = torch.zeros(4, U, dtype=torch.bool)
            res["set_inputs_ms"] = med(lambda: dec.set_inputs(cur, pos, dead), reps)
            line = json.dumps(dict(case="wall", U=U, B=4 * U, layers=NL, **res, dram=dram(mesh)))
            print("WALL_RESULT " + line, flush=True)
            open(OUT, "a").write(line + "\n")
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
