# SPDX-License-Identifier: Apache-2.0
"""Generic PAIRED in-process A/B of traced decode-step variants at 48 layers (one process, all traces alive, alternating rounds of blocking replays, host-timed), for U in AB_U.
Variants are declared in VARIANTS below: each is (name, env dict applied while it is built, kind) where kind 'layers' = embedding + decoder layers only (the race agent's measurement),
'full' = Decoder._forward (embedding + layers + final norm + lm_head + max + argmax). Also records the host load average per round.
Env: DEC_LAYERS(48) AB_U("1,8,32") AB_ROUNDS(5) AB_REPS(30) AB_VARIANTS(comma list of names) DEC_OUT."""
import json
import os
import statistics
import sys
import time

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_common import gsm_prompts  # noqa: E402
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.reference.prompts import tokenizer  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.decode import HIDDEN  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.generator import Generator  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402

E = os.environ.get
OUT = E("DEC_OUT", "/tmp/dec_ab2.jsonl")
VARIANTS = {
    "layers_exact": ({"MIMO_ROUTER": "exact"}, "layers"),
    "layers_fast": ({"MIMO_ROUTER": "fast"}, "layers"),
    "full_exact": ({"MIMO_ROUTER": "exact"}, "full"),
    "full_fast": ({"MIMO_ROUTER": "fast"}, "full"),
    "full_nohoist": ({"MIMO_ROPE_HOIST": "0"}, "full"),
    "full_hoist": ({"MIMO_ROPE_HOIST": "1"}, "full"),
    "base": ({"MIMO_NORM": "sharded", "MIMO_ROPE_HOIST": "0", "MIMO_ROUTER": "exact"}, "full"),
    "hoist": ({"MIMO_NORM": "sharded", "MIMO_ROPE_HOIST": "1", "MIMO_ROUTER": "exact"}, "full"),
    "hoist_fast": ({"MIMO_NORM": "sharded", "MIMO_ROPE_HOIST": "1", "MIMO_ROUTER": "fast"}, "full"),
    "hf_nomm1d": ({"MIMO_ROUTER": "fast", "MIMO_QKV_CORES": "0", "MIMO_WO_CORES": "0"}, "full"),
    "hf_mm1d": ({"MIMO_ROUTER": "fast", "MIMO_QKV_CORES": "60", "MIMO_WO_CORES": "32"}, "full"),
    "full_plainnorm": ({"MIMO_NORM": "plain"}, "full"),
    "full_shardednorm": ({"MIMO_NORM": "sharded"}, "full"),
}


def main():
    NL = int(E("DEC_LAYERS", "48"))
    Us = [int(v) for v in E("AB_U", "1,8,32").split(",")]
    ROUNDS, REPS = int(E("AB_ROUNDS", "5")), int(E("AB_REPS", "30"))
    names = E("AB_VARIANTS", "layers_exact,layers_fast,full_exact").split(",")
    tok = tokenizer()
    items = gsm_prompts(128, tok)
    S, max_pos = 256, 512
    mesh = open_mesh_full()
    try:
        model = MiMoModel(mesh, n_layers=NL, layout=KVLayout(users=Us[0], max_pos=max_pos, chunk=S), log=lambda s: None)
        gen = Generator(model, max_pos, trace=True, keep_logits=False, U=Us[0])
        for k, U in enumerate(Us):
            if k > 0:
                gen.reconfigure(U, KVLayout(users=U, max_pos=max_pos, chunk=S))
            gen.generate([it["prompt_ids"] for it in items[: 4 * U]], 6, S=S)
            gen.dec.release()
            dec = gen.dec
            os.environ["MIMO_CHECK_CAP"] = "0"
            dec._forward()  # builds every layer's MiMoMoEDecode once
            ttnn.synchronize_device(mesh)
            traces = {}
            for name in names:
                env, kind = VARIANTS[name]
                old = {k2: os.environ.get(k2) for k2 in env}
                os.environ.update(env)
                for (
                    lyr
                ) in (
                    model.layers
                ):  # the router mode is a runtime attribute of the (single, shared) MiMoMoEDecode: toggle it, never rebuild (two sets of persistent buffers hang)
                    if getattr(lyr, "moe_dec", None) is not None:
                        lyr.moe_dec.router = env.get("MIMO_ROUTER", "exact")

                def fwd(kind=kind):
                    if kind == "full":
                        return dec._forward()
                    e = ttnn.embedding(dec.tok, model.embed, layout=ttnn.TILE_LAYOUT)
                    h = ttnn.reshape(e, (1, 1, dec.U, HIDDEN))
                    for lyr in model.layers:
                        h = lyr.decode(h, dec.pos["cur"], dec.pos["idx"], live=dec.live.t)
                    return h

                fwd()
                ttnn.synchronize_device(mesh)
                os.environ["MIMO_CHECK_CAP"] = "0"
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                fwd()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                traces[name] = tid
                for k2, v in old.items():
                    if v is None:
                        os.environ.pop(k2, None)
                    else:
                        os.environ[k2] = v
            tms = {n: [] for n in names}
            load = []
            for rd in range(ROUNDS):
                order = names if rd % 2 == 0 else names[::-1]
                load.append(os.getloadavg()[0])
                for n in order:
                    ts = []
                    for _ in range(REPS + 2):
                        t0 = time.perf_counter()
                        ttnn.execute_trace(mesh, traces[n], cq_id=0, blocking=True)
                        ts.append(time.perf_counter() - t0)
                    tms[n].append(statistics.median(ts[2:]) * 1e3)
            res = {n: round(statistics.median(v), 3) for n, v in tms.items()}
            line = json.dumps(
                dict(
                    case="ab",
                    U=U,
                    layers=NL,
                    median_ms=res,
                    per_round={n: [round(x, 2) for x in v] for n, v in tms.items()},
                    host_load1=[round(x, 1) for x in load],
                )
            )
            print("AB_RESULT " + line, flush=True)
            open(OUT, "a").write(line + "\n")
            for t in traces.values():
                ttnn.release_trace(mesh, t)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
