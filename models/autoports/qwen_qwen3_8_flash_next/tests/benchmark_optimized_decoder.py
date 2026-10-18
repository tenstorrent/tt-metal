"""Serialized real-weight, identical-input fusion candidate measurements."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import (
    CHECKPOINT,
    no_runtime_host_conversion,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["fused", "optimized"], default="optimized")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--prefill-length", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--decode-position", type=int, default=0)
    parser.add_argument("--advance-position", action="store_true", help="Append a token and advance device position on each replay")
    args = parser.parse_args()
    if not 0 <= args.decode_position <= args.prefill_length:
        raise ValueError("The benchmark must prefill the decode context")
    if args.advance_position and args.decode_position != args.prefill_length:
        raise ValueError("Advancing decode must append to the complete prefilled prefix")
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    dest = root / "doc/optimized_decoder" / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    source = root / "tt" / (args.impl + "_decoder.py")
    source_snapshot = source.read_bytes()
    (dest / "source.py.txt").write_bytes(source_snapshot)
    (dest / "runner.py.txt").write_bytes(Path(__file__).read_bytes())
    if os.environ.get("QWEN38_ADVICE_VARIANT"):
        (dest / "advice_runner.py.txt").write_bytes((root / "tests/benchmark_optimized_advice.py").read_bytes())
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    state = CheckpointState(CHECKPOINT, f"model.language_model.layers.{args.layer}.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        model = (FusedDecoder if args.impl == "fused" else OptimizedDecoder).from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=args.layer,
            mesh_device=mesh,
            batch_size=1,
            chunk_size=args.chunk_size,
            max_requests=3,
            cache_dir=Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"),
        )
        model.prepare([1, args.prefill_length])
        identities = {
            name: type(getattr(model, name)).__module__ + "." + type(getattr(model, name)).__qualname__
            for name in ("attention", "mlp", "ple", "attn_hyper", "mlp_hyper")
            if hasattr(model, name)
        }

        native_calls = []
        original_native = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe
        def measured_native(*call_args, **call_kwargs):
            # Actual runtime argument evidence for this composite op; no tensor readback.
            if not native_calls:
                record = {}
                for name, tensors in zip(("gate", "up", "down"), call_args[4:7]):
                    record[name] = dict(count=len(tensors),
                        dtypes=sorted({str(t.dtype) for t in tensors}),
                        shapes=sorted({tuple(t.shape) for t in tensors}))
                native_calls.append(record)
            return original_native(*call_args, **call_kwargs)
        ttnn.experimental.deepseek_prefill.unified_routed_expert_moe = measured_native

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)

        rng = torch.Generator().manual_seed(2026)
        x = (
            torch.randn(1, args.prefill_length, 2560, generator=rng).repeat(1, 1, 4) * 0.05
            + torch.randn(1, args.prefill_length, 10240, generator=rng) * 0.005
        ).bfloat16()
        tokens = torch.randint(cfg.vocab_size, (1, args.prefill_length), generator=rng, dtype=torch.int32)
        xd, dd = upload(x.unsqueeze(0)), upload(x[:, :1].unsqueeze(0))
        pd = upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        decode_position = torch.zeros(1,32,dtype=torch.int32)
        decode_position[0,0] = args.decode_position
        dp = upload(decode_position,ttnn.uint32,ttnn.ROW_MAJOR_LAYOUT)
        slots = torch.zeros(1, 32, dtype=torch.int32)
        slots[0, 0] = 1
        sd = upload(slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        page = (
            upload(
                torch.randperm(cfg.max_position_embeddings // 32, generator=rng).reshape(1, -1).int(),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            if args.layer == 3
            else None
        )
        pt = upload(tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if args.layer == 1 else None
        dt = upload(tokens[:, :1], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if args.layer == 1 else None
        common = dict(current_positions=pd, slots=sd, page_table=page)

        def prefill():
            with no_runtime_host_conversion():
                return model.prefill_forward(xd, tokens=pt, **common)

        def decode():
            with no_runtime_host_conversion():
                result = model.decode_forward(dd, tokens=dt, **dict(common,current_positions=dp))
                if args.advance_position:
                    ttnn.plus_one(dp)
                return result

        def drain():
            if args.profile:
                ttnn.ReadDeviceProfiler(mesh)

        def reset_decode_position():
            if args.advance_position:
                host = ttnn.from_torch(decode_position, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                ttnn.copy_host_to_device_tensor(host, dp)

        for _ in range(2):
            reset_decode_position()
            prefill()
            decode()
            ttnn.synchronize_device(mesh)
            drain()
        if args.profile:
            from tracy import signpost
        else:
            signpost = lambda _: None
        timings = []
        prefill_host = None
        for _ in range(1 if args.profile else 5):
            signpost("PERF_PREFILL")
            before = time.perf_counter()
            out = prefill()
            ttnn.synchronize_device(mesh)
            timings.append((time.perf_counter() - before) * 1000)
            signpost("PERF_PREFILL_END")
            drain()
            prefill_host = ttnn.to_torch(out)
            del out
        reset_decode_position()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        position_after_capture = int(ttnn.to_torch(dp)[0, 0])
        assert position_after_capture in (args.decode_position, args.decode_position + int(args.advance_position)), position_after_capture
        drain()
        for _ in range(2):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            drain()
        samples = []
        for _ in range(1 if args.profile else 5):
            signpost("PERF_DECODE")
            before = time.perf_counter()
            for _ in range(10):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - before) * 1000 / 10)
            signpost("PERF_DECODE_END")
            drain()
        final_position = int(ttnn.to_torch(dp)[0, 0])
        if args.advance_position:
            assert final_position == position_after_capture + (12 if args.profile else 52), (position_after_capture, final_position)
        saved = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/optimization_outputs")
        saved.mkdir(parents=True, exist_ok=True)
        output_path = saved / (args.tag + ".pt")
        torch.save({"prefill": prefill_host, "decode": ttnn.to_torch(out)}, output_path)
        result = dict(
            impl=args.impl,
            layer=args.layer,
            tag=args.tag,
            seed=2026,
            policy=vars(args),
            advice_variant=os.environ.get("QWEN38_ADVICE_VARIANT"),
            runtime_classes=identities,
            native_expert_runtime_arguments=native_calls,
            batch=1,
            prefill_length=args.prefill_length,
            chunk_size=args.chunk_size,
            position=args.decode_position,
            advance_position=args.advance_position,
            position_after_capture=position_after_capture,
            final_position=final_position,
            decode_replays=50 if not args.profile else 10,
            prefill_host_ms=timings,
            traced_decode_ms=samples,
            median_prefill_host_ms=statistics.median(timings),
            median_traced_decode_ms=statistics.median(samples),
            profiler=args.profile,
            tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
            outputs=str(output_path),
            source_sha256=hashlib.sha256(source_snapshot).hexdigest(),
        )
        (dest / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if "original_native" in locals():
            ttnn.experimental.deepseek_prefill.unified_routed_expert_moe = original_native
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
