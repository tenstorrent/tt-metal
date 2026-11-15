"""pf-integrate bisect: per-layer taps (x4 after every layer, first TAPR token rows of the tapped chunks) of the 78-layer prefill, MODE=rep (replicated-cache chain) or MODE=pf (direct-to-pool), whole mesh.
Env: SRC, LEN (prompt tokens per user), TAPCH (comma chunk indices to tap, default last two), TAPR (rows, 32), TAGNAME. Output $OUT/pf-integrate/bis_<TAG>.pt: {chunk: {layer: [4 users][4 streams, TAPR, 6144] bf16}}; also ids-free."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

OUT = os.environ.get("OUT", "/out")
MODE = os.environ.get("MODE", "pf")
TAG = os.environ.get("TAGNAME", MODE)
PF, PAD = 512, 120002
TAPR = int(os.environ.get("TAPR", "32"))
os.environ["HY4_TAP_ROWS"] = str(TAPR)


def main():
    d = torch.load(os.environ["SRC"])
    LEN = int(os.environ["LEN"])
    prompts = [list(p[:LEN]) for p in d["ids"]]
    nch = -(-LEN // PF)
    tapch = [int(x) for x in os.environ.get("TAPCH", f"{nch-2},{nch-1}").split(",")]
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "700e6"))))
    try:
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as _pd
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest

        links = hy4_fabric_selftest(md)
        _pd._POOL_AG_LINKS = links.pool_ag
        import models.demos.blackhole.hy4_preview.tt.hy4_pf_attention as _pa

        _pa.LINKS = links.pool_ag
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=os.environ.get("HY4_CKPT", "/ckpt"))
        ch.add_layers(0, 78, log=lambda s: None)
        ch.build_head()
        dec = Hy4PoolDecoder(ch, nch + 2, mode="topk")
        out, ibl = {}, {}
        for c in range(nch):
            ids = torch.full((4, PF), PAD, dtype=torch.long)
            for r in range(4):
                seg = prompts[r][c * PF : (c + 1) * PF]
                ids[r, : len(seg)] = torch.tensor(seg)
            taps = {} if c in tapch else None
            x4, ibl = ch.prefill(ids, start=c * PF, ids_by_layer=ibl, taps=taps)
            ttnn.synchronize_device(md)
            if taps is not None:
                out[c] = {}
                for L, t in taps.items():
                    devs = ttnn.get_device_tensors(t)
                    out[c][L] = [ttnn.to_torch(devs[r * 8]).bfloat16().reshape(4, TAPR, -1) for r in range(4)]
                    ttnn.deallocate(t)
                print("RES " + json.dumps(dict(event="tapped", c=c, layers=len(out[c]))), flush=True)
            ttnn.deallocate(x4)
        os.makedirs(f"{OUT}/pf-integrate", exist_ok=True)
        torch.save(out, f"{OUT}/pf-integrate/bis_{TAG}.pt")
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
