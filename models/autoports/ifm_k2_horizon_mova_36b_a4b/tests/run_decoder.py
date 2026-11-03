# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer-only real checkpoint validation; host work is outside decoder calls."""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.functional_decoder import FunctionalDecoder

MODEL = "IFM/K2-Horizon-MoVA-36B-A4B"
REVISION = "e9ce717642b2074082875be233c02c729ba47960"
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--IFM--K2-Horizon-MoVA-36B-A4B/snapshots" / REVISION


def reference(layer, synthetic=False):
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "sdpa"
    cls = get_class_from_dynamic_module(
        "modeling_k2_horizon.K2HorizonDecoderLayer", MODEL, revision=REVISION, local_files_only=True
    )
    with torch.device("meta"):
        model = cls(config, layer)
    sd = {}
    stats_path = ROOT / f"doc/functional_decoder/weight_stats_layer_{layer}.json"
    if synthetic:
        for stat in json.loads(stats_path.read_text()):
            sd[stat["name"]] = (torch.randn(stat["shape"]) * stat["std"] + stat["mean"]).bfloat16()
    else:
        index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = f"model.layers.{layer}."
        for shard in sorted({f for k, f in index.items() if k.startswith(prefix)}):
            with safe_open(SNAPSHOT / shard, framework="pt") as f:
                for k in f.keys():
                    if k.startswith(prefix):
                        sd[k.removeprefix(prefix)] = f.get_tensor(k)
    model.load_state_dict(sd, assign=True)
    model.eval()
    rope = importlib.import_module(cls.__module__).K2HorizonRotaryEmbedding(config)
    if not synthetic:
        stats = [
            {
                "name": k,
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "mean": v.float().mean().item(),
                "std": v.float().std().item(),
            }
            for k, v in sd.items()
        ]
        stats_path.write_text(json.dumps(stats, indent=2))
    return config, model, rope, sd


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack((a, b)))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    c, hf, rope, sd = reference(args.layer)
    b, s = args.batch, args.seq
    x = (torch.randn(b, s + 1, c.hidden_size) * 0.1).bfloat16()
    cache = DynamicCache(config=c)
    with torch.no_grad():
        cs = rope(x, torch.arange(s).unsqueeze(0).expand(b, -1))
        mask = torch.full((s, s), float("-inf")).triu(1)
        y = hf(x[:, :s], position_embeddings=cs, attention_mask=mask[None, None], past_key_values=cache)
        dcs = rope(x, torch.full((b, 1), s))
        yd = hf(x[:, s:], position_embeddings=dcs, past_key_values=cache)
    print("HF_READY", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(v, device=mesh, dtype=dtype, layout=layout)

        def cpu(v):
            return ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

        model = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        blocks = (args.context + 31) // 32
        pages = torch.randperm(b * blocks).reshape(b, blocks).int()
        table = tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kv = [tt(torch.zeros(b * blocks, 8, 32, 128, dtype=torch.bfloat16)) for _ in range(2)]
        tx = tt(x[:, :s].unsqueeze(0))
        cos, sin = [tt(v.unsqueeze(1)) for v in cs]
        dx = tt(x[:, s:].reshape(1, 1, b, c.hidden_size))
        dc, ds = [tt(v.reshape(1, b, 1, 128)) for v in dcs]
        pos = tt(torch.full((b,), s, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kwargs = dict(cos=dc, sin=ds, page_table=table, current_pos=pos, kv_cache=kv)
        print("TT_READY", flush=True)
        out = model.prefill_forward(tx, cos=cos, sin=sin, page_table=table, kv_cache=kv)
        pre = pcc(y, cpu(out).reshape(b, s, -1))
        print("PREFILL_PCC", pre, flush=True)
        assert pre >= 0.995
        outd = model.decode_forward(dx, **kwargs)
        dec = pcc(yd, cpu(outd).reshape(b, 1, -1))
        print("DECODE_PCC", dec, flush=True)
        assert dec >= 0.995
        if args.profile:
            from tracy import signpost

            warm = model.prefill_forward(tx, cos=cos, sin=sin, page_table=table, kv_cache=kv)
            warm.deallocate(True)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL")
            t0 = time.perf_counter()
            measured = model.prefill_forward(tx, cos=cos, sin=sin, page_table=table, kv_cache=kv)
            ttnn.synchronize_device(mesh)
            prefill_ms = (time.perf_counter() - t0) * 1000
            signpost("PERF_PREFILL_END")
            measured.deallocate(True)
        if args.trace:
            outd.deallocate(True)
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            outd = model.decode_forward(dx, **kwargs)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            for i in range(3):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                traced = pcc(yd, cpu(outd).reshape(b, 1, -1))
                print("TRACE_PCC", i, traced, flush=True)
                assert traced >= 0.995
            if args.profile:
                ttnn.synchronize_device(mesh)
                signpost("PERF_DECODE")
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                decode_ms = (time.perf_counter() - t0) * 1000
                signpost("PERF_DECODE_END")
                print("WARM_WALL_MS", prefill_ms, decode_ms, flush=True)
                (ROOT / f"doc/functional_decoder/perf_layer{args.layer}.json").write_text(
                    json.dumps(
                        dict(
                            layer=args.layer,
                            batch=b,
                            prefill_length=s,
                            decode_position=s,
                            warmed_prefill_wall_ms=prefill_ms,
                            traced_warmed_decode_wall_ms=decode_ms,
                            measured_prefill_calls=1,
                            measured_decode_replays=1,
                            blocking=False,
                            prefill_pcc=pre,
                            traced_decode_pcc=traced,
                            environment={
                                key: os.environ.get(key)
                                for key in (
                                    "TT_METAL_TRACE_ALLOC_TRACKING",
                                    "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                                    "TT_METAL_WATCHER",
                                    "TTNN_CONFIG_OVERRIDES",
                                )
                            },
                            runner_argv=sys.argv,
                            decode_window_host_counters=dict(
                                trace_replays=1,
                                input_refreshes=0,
                                current_position_rope_refreshes=0,
                                page_table_refreshes=0,
                                synchronizations=1,
                                readbacks=0,
                                positions_advanced=False,
                            ),
                        ),
                        indent=2,
                    )
                )
            ttnn.release_trace(mesh, tid)
        print("PASS", vars(args), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
