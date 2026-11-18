"""debug (1x8 slot, NL=1, HY4_NO_MOE=1): stock chain layer 0 on real-ish inputs, chunks 0..3, taps of every intermediate: which one is the first non-finite for the odd chunks."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt import hy4_chain as HC
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def main():
    full, md = open_row()
    try:
        mc, ccl = mesh_1x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        ch = HC.Hy4Chain(md, mc, ccl, ckpt="/ckpt", kv_fmt=os.environ.get("HY4_KV_FMT", "bf16"))
        ch.add_layers(0, 1, log=lambda s: None)
        emb = ch.rd.get("model.embed_tokens.weight")
        g = torch.Generator().manual_seed(3)
        ly = ch.layers[0]
        for rep in range(2):
            ly.attn.reset()
            ibl = {}
            for c in range(4):
                ids = torch.randint(1000, 100000, (HC.CHUNK,), generator=g)
                x = upload(md, emb[ids].float().reshape(1, 1, HC.CHUNK, 6144).repeat(1, 4, 1, 1), ttnn.float32)
                taps = {}
                x4 = ly.prefill_chunk(x, c * HC.CHUNK, ibl, taps=taps)
                ttnn.synchronize_device(md)
                fin = {k: bool(all(torch.isfinite(t).all() for t in v)) for k, v in taps.items()}
                xo = ttnn.to_torch(ttnn.get_device_tensors(x4)[0]).float()
                print(
                    "RES "
                    + json.dumps(
                        {"rep": rep, "chunk": c, "out_finite": bool(torch.isfinite(xo).all()), "taps_finite": fin}
                    ),
                    flush=True,
                )
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
