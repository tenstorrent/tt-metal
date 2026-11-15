"""mtp-device probe: does the Hy4 chain layer 78 (MTP) build on a 1x8 slot and does its MoE decode block run at T rows? Prints RES lines."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dm_block import close_row, open_row
from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID, Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def main():
    full, md = open_row()
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean

        ccl_lean.make_lean(mc)
        ccl_lean.make_lean_sems(ccl)
        t0 = time.time()
        ch = Hy4Chain(md, mc, ccl, os.environ["HY4_CKPT"])
        ch.add_layers(78, 79)
        ly = ch.mtp_layers[0]
        res(event="built", sec=round(time.time() - t0, 1), dram=dram_view(md))
        for T in [int(x) for x in os.environ.get("TS", "1,4").split(",")]:
            blk = ly.blk if T == 1 else ly.blk.for_batch(T)
            x = torch.randn(1, 1, T, HID).bfloat16()
            tt_gate = upload(md, x, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            tt_tok = upload(md, x.reshape(T, 1, 1, HID), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
            for call in range(2):
                part = ly.shared_dec.forward(tt_gate) if ly.shared_dec is not None else ly.shared.forward(tt_gate)
                out = blk.forward(tt_gate, tt_tok, fold=part)
                ttnn.synchronize_device(md)
                o = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
                res(
                    event="moe",
                    T=T,
                    call=call,
                    shape=list(o.shape),
                    absmean=float(o.abs().mean()),
                    finite=bool(torch.isfinite(o).all()),
                )
    finally:
        close_row(full, md)


if __name__ == "__main__":
    main()
