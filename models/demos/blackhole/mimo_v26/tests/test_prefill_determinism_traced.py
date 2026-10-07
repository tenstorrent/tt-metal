# SPDX-License-Identifier: Apache-2.0
"""Race hunt for the PREFILL column reduces (attention o_proj all_reduce, dense FFN, MoE reduce in tt/layer.py/attention.py): the traced chunked-prefill step of PD_LAYERS layers (default 6) is
replayed PD_REPS times on identical inputs (real GSM8K text, one user per mesh row) and the output hidden state [rows, cols, CH, 4096] of every replay is compared bitwise with replay 0.
KV writes of a repeated chunk are idempotent. No device allocation after trace capture (it hangs the next replay). Prints PD_GLITCH lines and a PD_RESULT summary."""
import json
import os
import socket
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import close_mesh, open_mesh_full  # noqa: E402

from models.demos.blackhole.mimo_v26.tt.kv_pool import KVLayout  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.model import MiMoModel  # noqa: E402
from models.demos.blackhole.mimo_v26.tt.prefill import ChunkedPrefill  # noqa: E402

E = os.environ.get
GSM = "/mnt/tt-data/ssinghal/mimo-dumps/gsm8k/gsm8k_512.json"


def main():
    NL, CH, R = int(E("PD_LAYERS", "6")), int(E("PD_CH", "512")), int(E("PD_REPS", "1000"))
    stream = [t for it in json.load(open(GSM))["items"] for t in it["prompt_ids"]]
    mesh = open_mesh_full(trace_mb=int(E("PD_TRACE_MB", "1024")))
    try:
        lay = KVLayout(users=1, max_pos=CH, chunk=CH)
        model = MiMoModel(mesh, n_layers=NL, layout=lay, log=lambda s: None)
        pf = ChunkedPrefill(model, trace=True)
        toks = torch.stack([torch.tensor(stream[r * 997 : r * 997 + CH]) for r in range(4)])
        pf.set_step(toks, [0, 0, 0, 0], [0, 0, 0, 0])
        out = pf.run_step()  # eager warm-up + capture + first replay
        comp = ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))

        def grab():
            return ttnn.to_torch(out, mesh_composer=comp).reshape(4, 8, CH, 4096).view(torch.int16)

        pf.run_step()
        ref = grab()
        bad = 0
        t0 = time.time()
        for rep in range(R):
            pf.run_step()
            cur = grab()
            d = cur != ref
            if d.any():
                cnt = d.sum(dim=(2, 3))
                for r, c in cnt.nonzero().tolist():
                    bad += 1
                    rows = d[r, c].any(-1).nonzero().flatten()
                    print(
                        "PD_GLITCH", rep, (r, c), int(cnt[r, c]), "rows", int(rows.min()), int(rows.max()), flush=True
                    )
            if rep % 100 == 0:
                print("PD_PROGRESS", rep, round(time.time() - t0), flush=True)
        s = json.dumps(
            dict(
                case="prefill_determinism_traced",
                host=socket.gethostname(),
                layers=NL,
                ch=CH,
                reps=R,
                device_events=bad,
                expert_op=E("MIMO_EXPERT_OP"),
            )
        )
        print("PD_RESULT " + s, flush=True)
        open(E("PD_OUT", "/tmp/pd.jsonl"), "a").write(s + "\n")
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
