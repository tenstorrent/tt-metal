# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""reconcile task: device-profiler pass over the real traced pool decode of SEL layers (embedding + layers + head), trace replay, per-op device start/end on every chip.
Run under TT_METAL_DEVICE_PROFILER=1 (TT_METAL_PROFILER_CPP_POST_PROCESS=1 for op names). Env: NLMAX (3), SEL ('1,2'), MODE (dense), CTX (2048), WARM (2), PROF_REPLAYS (1)."""
import json
import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.op_table_recorder import OpRecorder
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tests.recon_bisect import make_step
from models.demos.blackhole.hy4_preview.tests.recon_slope import res
from models.demos.blackhole.hy4_preview.tt.hy4_chain import CHUNK, Hy4Chain
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CKPT = os.environ.get("HY4_CKPT", "/ckpt")
OUT = os.environ.get("OUT", "/out")
NLMAX = int(os.environ.get("NLMAX", "3"))
SEL = [int(x) for x in os.environ.get("SEL", "1,2").split(",") if x]
MODE = os.environ.get("MODE", "dense")
CTX = int(os.environ.get("CTX", "2048"))
PAD = 120002


def main():
    prompts = torch.load(os.environ.get("PROMPTS", f"{OUT}/decode_prompts.pt"))
    lens = [len(p) for p in prompts]
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "700e6"))))
    try:
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as _pd
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest

        links = hy4_fabric_selftest(md)
        _pd._POOL_AG_LINKS = links.pool_ag
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=CKPT)
        ch.add_layers(0, NLMAX, log=lambda s: None)
        ch.build_head()
        pages = max(CTX // 512, 8)
        dec = Hy4PoolDecoder(ch, pages, mode="dense")
        full = list(ch.layers)
        if MODE == "dense":
            ids = torch.full((4, CHUNK), PAD, dtype=torch.long)
            for r in range(4):
                ids[r, : lens[r]] = torch.tensor(prompts[r])
            x4, _ = ch.prefill(ids)
            ttnn.synchronize_device(md)
            ttnn.deallocate(x4)
            dec.handoff(CHUNK)
            ch.reset()
            pos = list(lens)
        else:
            dec.fill_synthetic(CTX // 512 * 512, torch.Generator().manual_seed(11))
            dec.mode = "topk"
            dec.ctx.set_bucket(CTX, "topk")
            pos = [CTX - 1] * 4
        dec.set_inputs([1000, 2000, 3000, 4000], pos)
        if (
            os.environ.get("STOCK_STEP", "0") == "1"
        ):  # the product step() (honours HY4_IHC_FUSED); layers = the first len(SEL)+1 built layers
            ch.layers = full[: max(SEL) + 1]
            step = dec.step
        else:
            step = make_step(dec, full, SEL, int(os.environ.get("CUT_A", "5")), int(os.environ.get("CUT_M", "4")))
        rec = OpRecorder()
        rec.install()
        dec.release()
        lg = step()
        ttnn.synchronize_device(md)
        ttnn.deallocate(lg)
        dec.trace = ttnn.begin_trace_capture(md, cq_id=0)
        rec.enabled = True
        dec.out = step()
        rec.enabled = False
        ttnn.end_trace_capture(md, dec.trace, cq_id=0)
        ttnn.synchronize_device(md)
        tag = os.environ.get("TAGNAME", "p")
        json.dump(dict(rows=rec.rows, zero=rec.zero, sel=SEL, mode=MODE), open(f"/rout/{tag}_ops.json", "w"))
        res(event="recorded", nops=len(rec.rows))
        ttnn.execute_trace(md, dec.trace, cq_id=0, blocking=True)
        ttnn.ReadDeviceProfiler(md)
        for _ in range(int(os.environ.get("PROF_REPLAYS", "1"))):
            ttnn.execute_trace(md, dec.trace, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(md)
        res(event="prof_done")
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
