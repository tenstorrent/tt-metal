"""Warmed 4096-input / 128-decode layer workload, with explicit setup boundaries."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from tracy import signpost
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.run_decoder import weights
from models.autoports.google_gemma_4_31b_it.tt.optimized_decoder import HF_MODEL, HF_REVISION, OptimizedDecoder


def run(layer, batch, precision=None, attention_precision=None, geometry=None, result=None):
    torch.set_num_threads(16)
    torch.manual_seed(419)
    cfg = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    kind = cfg.layer_types[layer]
    sd = weights(layer)
    input_path = Path(f"bringup/artifacts/optimized_decoder_layer{layer}_inputs262144.pt")
    x = torch.load(input_path, weights_only=True, mmap=True)[:, :4224].expand(batch, -1, -1).contiguous()
    assert tuple(x.shape) == (batch, 4224, cfg.hidden_size)
    cos, sin = Gemma4TextRotaryEmbedding(cfg)(x, torch.arange(4224).expand(batch, -1), layer_type=kind)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)

    def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)

    tid = None
    try:
        cls = OptimizedDecoder
        extra = {}
        if geometry is not None:
            from models.autoports.google_gemma_4_31b_it.tests.optimized_attention_candidate import AttentionCandidate

            cls = AttentionCandidate
            if geometry.get("lifetime"):
                from models.autoports.google_gemma_4_31b_it.tests.optimized_lifetime_candidate import LifetimeCandidate

                cls = LifetimeCandidate
            if geometry.get("tune_prefill"):
                from models.autoports.google_gemma_4_31b_it.tests.optimized_prefill_candidate import PrefillCandidate

                cls = PrefillCandidate
            if (
                geometry.get("head_l1")
                or geometry.get("keep_full_q")
                or geometry.get("direct_qkv")
                or geometry.get("attn_act8")
            ):
                from models.autoports.google_gemma_4_31b_it.tests.optimized_topology_candidate import TopologyCandidate

                cls = TopologyCandidate
            if geometry.get("output_padding"):
                from models.autoports.google_gemma_4_31b_it.tests.optimized_padding_candidate import PaddingCandidate

                cls = PaddingCandidate
            if "final_sdpa_fp32" in geometry:
                from models.autoports.google_gemma_4_31b_it.tests.optimized_sdpa_candidate import SDPAPrefillCandidate

                cls = SDPAPrefillCandidate
            extra["geometry"] = geometry
        policy = {
            key: value
            for key, value in dict(precision=precision, attention_precision=attention_precision).items()
            if value is not None
        }
        model = cls.from_state_dict(sd, hf_config=cfg, layer_idx=layer, mesh_device=mesh, **policy, **extra)
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
        pre_start = time.perf_counter()
        out = model.prefill_forward(px, **prekw)
        ttnn.synchronize_device(mesh)
        prefill_host_us = (time.perf_counter() - pre_start) * 1e6
        signpost("PERF_PREFILL_END")
        ttnn.ReadDeviceProfiler(mesh)
        del out
        gc.collect()
        from models.autoports.google_gemma_4_31b_it.tests.optimized_trace import capture

        tid, output = capture(mesh, lambda: model.decode_forward(dx, **dekw))
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        signpost("PERF_DECODE")
        start = time.perf_counter()
        group_start = start
        active_seconds = 0.0
        drain_seconds = 0.0
        for step, row in enumerate(host):
            for source, dest in zip(row, (dx, dc, ds, cp)):
                ttnn.copy_host_to_device_tensor(source, dest)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            if (step + 1) % 8 == 0:
                # Drain outside whole-layer replay windows to retain every device marker.
                ttnn.synchronize_device(mesh)
                active_seconds += time.perf_counter() - group_start
                drain_start = time.perf_counter()
                ttnn.ReadDeviceProfiler(mesh)
                drain_seconds += time.perf_counter() - drain_start
                group_start = time.perf_counter()
        ttnn.synchronize_device(mesh)
        total_seconds = time.perf_counter() - start
        signpost("PERF_DECODE_END")
        if result:
            Path(result).write_text(
                json.dumps(
                    dict(
                        prefill_host_us=prefill_host_us,
                        decode_host_us_per_token_without_profiler_drains=active_seconds * 1e6 / 128,
                        decode_host_us_per_token_including_profiler_drains=total_seconds * 1e6 / 128,
                        profiler_drain_us=drain_seconds * 1e6,
                        scope="Same profiled progressing 4096/128 layer run; includes harness-owned teacher-forcing input copies and trace submissions; no token-out generator.",
                    ),
                    indent=2,
                )
                + "\n"
            )
        ttnn.release_trace(mesh, tid)
        tid = None
        print(f"PROFILE_COMPLETE {kind} batch={batch} input=4096 decode_steps=128", flush=True)
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--attention-precision", default=None)
    parser.add_argument("--geometry")
    parser.add_argument("--geometry-file")
    parser.add_argument("--result")
    args = parser.parse_args()
    run(
        args.layer,
        args.batch,
        args.precision,
        args.attention_precision,
        (
            json.loads(Path(args.geometry_file).read_text())
            if args.geometry_file
            else json.loads(args.geometry)
            if args.geometry
            else None
        ),
        args.result,
    )
