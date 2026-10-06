# SPDX-License-Identifier: Apache-2.0
"""Matched real-weight single-chip/TP4 decoder validation and warmed timing."""

import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tracy import signpost

import ttnn

from ..tt.multichip_decoder import MeshPolicy, MultichipDecoder
from ..tt.optimized_decoder import OptimizedDecoder
from .optimized_coverage import runtime_audit
from .reference import config, load_weights
from .run_decoder import pcc

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "doc/multichip_decoder"
OUT = Path(os.environ.get("MC_ARTIFACT_DIR", str(REFERENCE)))


def run(a):
    torch.set_num_threads(8)
    torch.manual_seed(1890 + a.layer)
    cfg = config()
    cfg.max_position_embeddings = 1048576
    weights = load_weights(a.layer)
    if not a.baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if a.baseline else 4),
        physical_device_ids=[0] if a.baseline else [],
        trace_region_size=10000000,
    )
    try:
        opts = {} if a.baseline else {"policy": MeshPolicy(**json.loads(os.environ.get("MC_POLICY", "{}")))}
        model = (OptimizedDecoder if a.baseline else MultichipDecoder).from_state_dict(
            weights, hf_config=cfg, layer_idx=a.layer, mesh_device=mesh, **opts
        )
        del weights
        gc.collect()

        def tt(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, dim=None):
            mapper = (
                None
                if a.baseline
                else (ttnn.ReplicateTensorToMesh(mesh) if dim is None else ttnn.ShardTensorToMesh(mesh, dim=dim))
            )
            return ttnn.from_torch(
                x.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                mesh_mapper=mapper,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def integer(x):
            return tt(x, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        sharded = not a.baseline and model.hidden_width == 640

        def read(x):
            if sharded:
                return torch.cat([ttnn.to_torch(v) for v in ttnn.get_device_tensors(x)], dim=-1)
            return ttnn.to_torch(x if a.baseline else ttnn.get_device_tensors(x)[0])

        def angles(pos, length):
            phase = torch.arange(pos, pos + length).float()[:, None] / 10000.0 ** (
                torch.arange(0, 128, 2).float() / 128
            )
            phase = torch.cat([phase, phase], -1)[None, None]
            return tt(phase.cos().bfloat16()), tt(phase.sin().bfloat16())

        capacity = max(1024, (a.tokens + 512) // 512 * 512)
        blocks = capacity // 32
        cache = tuple(
            tt(torch.zeros(blocks, 4, 32, 128, dtype=torch.bfloat16), ttnn.bfloat8_b, dim=1) for _ in range(2)
        )
        pages = torch.randperm(blocks, dtype=torch.int32)[None]
        plan = model.prepare_prefill(page_table_host=pages, seq_len=a.tokens)
        recorded = torch.load(ROOT / f"doc/optimized_decoder/recorded_inputs/layer_{a.layer}.pt", weights_only=True)
        x = recorded[:, torch.arange(a.tokens + 1) % recorded.shape[1]].clone()
        inp = tt(x[:, : a.tokens][None], dim=3 if sharded else None)
        dec_in = tt(x[:, a.tokens :][None], dim=3 if sharded else None)
        c, s = angles(a.tokens, 1)
        kw = dict(
            kv_cache=cache,
            page_table=integer(pages),
            current_pos=integer(torch.tensor([a.tokens], dtype=torch.int32)),
            cos=c,
            sin=s,
        )

        def prefill():
            return model.prefill_forward(inp, kv_cache=cache, plan=plan)

        def decode():
            return model.decode_forward(dec_in, **kw)

        def drain_profiler():
            if os.environ.get("MC_PROFILE_DRAIN") == "1":
                ttnn.ReadDeviceProfiler(mesh)

        print("SETUP_DONE", flush=True)
        drain_profiler()
        for _ in range(2):
            with runtime_audit():
                pre = prefill()
                dec = decode()
                del pre, dec
            drain_profiler()
        gc.collect()
        ttnn.synchronize_device(mesh)
        print("WARMUP_DONE", flush=True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        dec = decode()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        drain_profiler()
        try:
            signpost("PERF_PREFILL")
            begin = time.monotonic()
            for _ in range(a.repetitions):
                pre = prefill()
                del pre
            ttnn.synchronize_device(mesh)
            pre_ms = (time.monotonic() - begin) * 1000 / a.repetitions
            signpost("PERF_PREFILL_END")
            drain_profiler()
            pre = prefill()
            pre_cpu = read(pre)
            del pre
            drain_profiler()
            signpost("PERF_DECODE")
            begin = time.monotonic()
            for _ in range(a.repetitions):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            dec_ms = (time.monotonic() - begin) * 1000 / a.repetitions
            signpost("PERF_DECODE_END")
            outputs = {"prefill": pre_cpu, "decode": read(dec)}
            rows = {}
            reference = REFERENCE / f"baseline_{a.layer}_{a.tokens}.pt"
            if a.baseline:
                torch.save(outputs, reference)
            else:
                expected = torch.load(reference, weights_only=True)
                for mode in outputs:
                    rows[mode] = pcc(expected[mode], outputs[mode])
                    print(mode, rows[mode], flush=True)
                # Each rank must contain the same reduced result.
                for rank, value in enumerate([] if sharded else ttnn.get_device_tensors(dec)):
                    rows[f"replica_{rank}"] = pcc(outputs["decode"], ttnn.to_torch(value))
            result = dict(
                layer=a.layer,
                tokens=a.tokens,
                mesh_shape=list(mesh.shape),
                model_class=type(model).__name__,
                physical_chunks=[[entry[2] for entry in chunks] for chunks in plan["slots"]],
                cache_dtype=str(cache[0].dtype),
                decode_trace=True,
                trace_blocking=False,
                warmup_iterations=2,
                baseline=a.baseline,
                pcc=rows,
                latency_ms=dict(prefill=pre_ms, decode=dec_ms),
                repetitions=a.repetitions,
                policy=asdict(model.policy),
                tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
                source_sha256=hashlib.sha256(
                    Path(__import__(model.__module__, fromlist=[""]).__file__).read_bytes()
                ).hexdigest(),
            )
            (OUT / f"{a.tag}_{a.layer}_{a.tokens}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            for name, score in rows.items():
                assert score >= (0.99999 if name.startswith("replica_") else 0.995), (name, score)
        finally:
            ttnn.release_trace(mesh, tid)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--tag", default="initial")
    run(parser.parse_args())
