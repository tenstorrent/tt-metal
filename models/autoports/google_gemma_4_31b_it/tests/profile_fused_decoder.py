"""Warmed 4096-input / 128-decode layer workload, with explicit setup boundaries."""

import argparse
import gc

import torch
from tracy import signpost
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.run_decoder import weights
from models.autoports.google_gemma_4_31b_it.tt.fused_decoder import HF_MODEL, HF_REVISION, FusedDecoder


def run(layer, batch):
    torch.set_num_threads(16)
    torch.manual_seed(419)
    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    kind = cfg.layer_types[layer]
    sd = weights(layer)
    x = torch.randn(batch, 4224, cfg.hidden_size, dtype=torch.bfloat16)
    cos, sin = Gemma4TextRotaryEmbedding(cfg)(x, torch.arange(4224).expand(batch, -1), layer_type=kind)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)

    def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)

    try:
        model = FusedDecoder.from_state_dict(sd, hf_config=cfg, layer_idx=layer, mesh_device=mesh)
        pages = torch.randperm(batch * 132).reshape(batch, 132).to(torch.int32)
        page = tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cache = model.allocate_cache(num_blocks=batch * 132)
        px = tt(x[:, :4096].unsqueeze(1))
        pc = tt(cos[:, :4096].unsqueeze(1))
        ps = tt(sin[:, :4096].unsqueeze(1))
        dx = tt(x[:, 4096].reshape(1, 1, batch, -1))
        dc = tt(cos[:, 4096].reshape(1, batch, 1, -1))
        ds = tt(sin[:, 4096].reshape(1, batch, 1, -1))
        cp = tt(torch.full((batch,), 4096, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        host = []
        for i in range(4096, 4224):
            host.append(
                (
                    tt(x[:, i].reshape(1, 1, batch, -1), device=None),
                    tt(cos[:, i].reshape(1, batch, 1, -1), device=None),
                    tt(sin[:, i].reshape(1, batch, 1, -1), device=None),
                    tt(torch.full((batch,), i, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, device=None),
                )
            )
        prekw = dict(cos=pc, sin=ps, kv_cache=cache, page_table=page)
        dekw = dict(cos=dc, sin=ds, kv_cache=cache, page_table=page, current_pos=cp)
        warm = model.prefill_forward(px, **prekw)
        del warm
        warm = model.decode_forward(dx, **dekw)
        del warm
        # Warmed prefill restores prefix state before recording the decode trace.
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_PREFILL")
        out = model.prefill_forward(px, **prekw)
        ttnn.synchronize_device(mesh)
        signpost("PERF_PREFILL_END")
        ttnn.ReadDeviceProfiler(mesh)
        del out
        gc.collect()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = model.decode_forward(dx, **dekw)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_DECODE")
        for step, row in enumerate(host):
            for source, dest in zip(row, (dx, dc, ds, cp)):
                ttnn.copy_host_to_device_tensor(source, dest)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            if (step + 1) % 8 == 0:
                # Drain outside whole-layer replay windows to retain every device marker.
                ttnn.synchronize_device(mesh)
                ttnn.ReadDeviceProfiler(mesh)
        ttnn.synchronize_device(mesh)
        signpost("PERF_DECODE_END")
        ttnn.release_trace(mesh, tid)
        print(f"PROFILE_COMPLETE {kind} batch={batch} input=4096 decode_steps=128", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    run(args.layer, args.batch)
