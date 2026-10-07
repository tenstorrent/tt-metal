# SPDX-License-Identifier: Apache-2.0
"""Device-profiler op tables of ONE traced decode step, per layer type, at U users per mesh row.
Builds the first PROF_NL layers (default 12: L0 dense+global, L1-4/6-10 SWA+MoE, L5/L11 global+MoE), prefills 4U real GSM8K prompts, then for each profiled layer runs the layers before it
eagerly to get the real hidden state and captures THAT layer's decode as its own trace (embedding and head are separate traces), replays each trace PROF_REPS times and reads the device profiler
after every trace. Needs TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 TT_METAL_PROFILER_DIR=<dir>. Writes <dir>/segments.json = {segment: {trace_id, layer, kind}}.
Env: PROF_U (8) PROF_NL (12) PROF_LAYERS (0,1,2,3,5,11) PROF_REPS (2) MIMO_DEAD etc."""
import json
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.decode import HIDDEN, Decoder  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.layer import EPS, _k_norm  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get


def main():
    U = int(E("PROF_U", "8"))
    NL = int(E("PROF_NL", "12"))
    layers = [int(v) for v in E("PROF_LAYERS", "0,1,2,3,5,11").split(",")]
    reps = int(E("PROF_REPS", "2"))
    out_dir = E("TT_METAL_PROFILER_DIR", "/tmp/prof")
    tok = tokenizer()
    items = gsm_prompts(4 * U, tok)
    S, max_pos = 256, 512
    mesh = open_mesh_full()
    segs = {}
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=KVLayout(users=U, max_pos=max_pos, chunk=S), log=lambda s: None)
        gen = Generator(model, max_pos, trace=True, keep_logits=False, U=U)
        prompts = [it["prompt_ids"] for it in items]
        first, _ = gen.prefill(prompts, S)
        rows = 4
        cur = torch.zeros(rows, U, dtype=torch.long)
        pos = torch.zeros(rows, U, dtype=torch.long)
        for i, (r, s) in enumerate(gen.place(len(prompts))):
            cur[r, s] = first[r, s]
            pos[r, s] = len(prompts[i])
        dec = Decoder(model, trace=False, U=U)
        dec.set_inputs(cur, pos, torch.zeros(rows, U, dtype=torch.bool))
        live = dec.live.t
        os.environ["MIMO_CHECK_CAP"] = "0"

        def embed():
            e = ttnn.embedding(dec.tok, model.embed, layout=ttnn.TILE_LAYOUT)
            return ttnn.reshape(e, (1, 1, U, HIDDEN))

        def capture(name, fn, meta):
            ttnn.synchronize_device(mesh)
            o = fn()  # compile / warm
            ttnn.synchronize_device(mesh)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            o = fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)  # flush the compile-run records so they do not mix with the replays
            for _ in range(reps):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(mesh)
            segs[name] = dict(trace_id=int(tid), **meta)
            print("PROF_SEGMENT", name, segs[name], flush=True)
            ttnn.release_trace(mesh, tid)
            return o

        # embedding
        capture("embed", embed, dict(kind="embed"))
        h = embed()
        done = 0
        for L in layers:
            while done < L:  # advance the real hidden state eagerly through the layers before L
                h2 = model.layers[done].decode(h, dec.pos["cur"], dec.pos["idx"], live=live)
                ttnn.deallocate(h)
                h = h2
                done += 1
            lyr = model.layers[L]
            hin = ttnn.clone(h)
            kind = ("dense" if L == 0 else "moe") + "_" + ("swa" if lyr.swa else "global")
            capture(
                f"layer{L}",
                lambda: lyr.decode(hin, dec.pos["cur"], dec.pos["idx"], live=live),
                dict(kind=kind, layer=L),
            )

        # head: norm + lm_head + max + argmax on the last hidden state
        def head():
            x = ttnn.rms_norm(h, epsilon=EPS, weight=model.w_norm, compute_kernel_config=_k_norm())
            lg = ttnn.linear(x, model.lm_head, dtype=ttnn.bfloat16)
            v = ttnn.max(lg, dim=3, keepdim=True)
            i = ttnn.argmax(ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT), dim=3, keepdim=True)
            return v, i

        capture("head", head, dict(kind="head"))
        os.makedirs(out_dir, exist_ok=True)
        json.dump(segs, open(os.path.join(out_dir, "segments.json"), "w"), indent=1)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
