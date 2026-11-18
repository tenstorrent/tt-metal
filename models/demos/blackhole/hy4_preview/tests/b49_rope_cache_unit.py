"""B49 unit (1x8 slot): HY4_B49_ROPE_CACHE=1 Hy4Attention.rope_tabs (prefill chunk tables shared by all layers) vs the uncached tables: bit-equal for layers of both kinds (full / shared indexer) in
the orders full->shared, shared->full (idx tables added on demand), several starts, and a decode-size call (n=1) is never cached. RES lines."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt import hy4_attention as HA
from models.demos.blackhole.hy4_preview.tt import hy4_chain as HC
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def snap(t):
    return None if t is None else ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def main():
    full, md = open_row()
    try:
        mc, ccl = mesh_1x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        ch = HC.Hy4Chain(md, mc, ccl, ckpt=os.environ.get("HY4_CKPT", "/ckpt"), kv_fmt="bf16")
        ch.add_layers(0, 3, log=lambda s: None)
        la = [ly.attn for ly in ch.layers]  # L0 full, L1 full, L2 shared
        res(event="kinds", has_indexer=[a.has_indexer for a in la])
        bad = 0
        for start in (0, 512, 65536, 1048064):
            os.environ["HY4_B49_ROPE_CACHE"] = "0"
            ref = [[snap(t) for t in a.rope_tabs(start, 512)] for a in la]
            os.environ["HY4_B49_ROPE_CACHE"] = "1"
            HA._ROPE_TABS.clear()
            for order in ((2, 0, 1), (0, 2, 1), (1, 2, 0)):
                for k in order:
                    got = [snap(t) for t in la[k].rope_tabs(start, 512)]
                    for g, r in zip(got, ref[k]):
                        if (g is None) != (r is None) or (g is not None and not torch.equal(g, r)):
                            bad += 1
            n_before = len(HA._ROPE_TABS)
            la[0].rope_tabs(start, 1)
            res(event="start", start=start, cached_entries=n_before, entries_after_decode_call=len(HA._ROPE_TABS))
        res(event="rope_cache", mismatches=bad, ok=bad == 0)
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
