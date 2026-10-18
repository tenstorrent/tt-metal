"""Serialized real-weight, identical-input fusion candidate measurements."""

import argparse
import hashlib
import json
import os
import statistics
import sys
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
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder, _hidden_parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["single", "mesh"], default="mesh")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--residual-layout", choices=["hidden", "stream", "replicated"], default="hidden")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-prefill", action="store_true", help="Measure caller-owned fixed-shape prefill replay")
    parser.add_argument(
        "--check-prefill-trace", action="store_true", help="Verify changed inputs and slots on the same prefill trace"
    )
    parser.add_argument("--recorded", action="store_true")
    parser.add_argument("--recorded-tag", default="recorded_inputs")
    parser.add_argument("--prefill-length", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--decode-position", type=int, default=0)
    parser.add_argument(
        "--advance-position", action="store_true", help="Append a token and advance device position on each replay"
    )
    args = parser.parse_args()
    if args.check_prefill_trace and not args.trace_prefill:
        parser.error("--check-prefill-trace requires --trace-prefill")
    if args.profile and args.trace_prefill:
        parser.error(
            "The profiler runner currently supports one decode trace; prefill replay evidence uses separate unprofiled runs"
        )
    if not 0 <= args.decode_position <= args.prefill_length:
        raise ValueError("The benchmark must prefill the decode context")
    if args.advance_position and args.decode_position != args.prefill_length:
        raise ValueError("Advancing decode must append to the complete prefilled prefix")
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    dest = root / os.environ.get("QWEN38_STAGE_DIR", "doc/multichip_decoder") / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    source = root / "tt" / ("multichip_decoder.py" if args.impl == "mesh" else "optimized_decoder.py")
    source = Path(globals().get("SOURCE_OVERRIDE", source))
    source_snapshot = source.read_bytes()
    (dest / "source.py.txt").write_bytes(source_snapshot)
    (dest / "runner.py.txt").write_bytes(Path(__file__).read_bytes())
    invoked = Path(sys.argv[0])
    if invoked.is_file() and invoked.suffix == ".py":
        (dest / "invoked_runner.py.txt").write_bytes(invoked.read_bytes())
    if os.environ.get("QWEN38_ADVICE_VARIANT"):
        (dest / "advice_runner.py.txt").write_bytes((root / "tests/benchmark_optimized_advice.py").read_bytes())
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    state = CheckpointState(CHECKPOINT, f"model.language_model.layers.{args.layer}.")
    if args.impl == "mesh":
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4 if args.impl == "mesh" else 1), trace_region_size=0)
    trace = None
    try:
        model = (MultichipDecoder if args.impl == "mesh" else OptimizedDecoder).from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=args.layer,
            mesh_device=mesh,
            batch_size=1,
            chunk_size=args.chunk_size,
            max_requests=2,
            cache_dir=Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"),
        )
        model.prepare([1, args.prefill_length])
        identities = {
            name: type(getattr(model, name)).__module__ + "." + type(getattr(model, name)).__qualname__
            for name in ("attention", "mlp", "ple", "attn_hyper", "mlp_hyper")
            if hasattr(model, name)
        }

        class_sources = {}
        for name in ("attention", "mlp", "ple", "attn_hyper", "mlp_hyper"):
            module = sys.modules.get(type(getattr(model, name, None)).__module__)
            path = getattr(module, "__file__", None)
            if path and Path(path).suffix == ".py":
                data = Path(path).read_bytes()
                class_sources[path] = hashlib.sha256(data).hexdigest()
                (dest / (Path(path).stem + ".py.txt")).write_bytes(data)
        native_calls = []
        original_native = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe

        def measured_native(*call_args, **call_kwargs):
            # Actual runtime argument evidence for this composite op; no tensor readback.
            if not native_calls:
                record = {}
                for name, tensors in zip(("gate", "up", "down"), call_args[4:7]):
                    record[name] = dict(
                        count=len(tensors),
                        dtypes=sorted({str(t.dtype) for t in tensors}),
                        shapes=sorted({tuple(t.shape) for t in tensors}),
                    )
                native_calls.append(record)
            return original_native(*call_args, **call_kwargs)

        ttnn.experimental.deepseek_prefill.unified_routed_expert_moe = measured_native

        def host_output(t):
            if args.impl == "single":
                return ttnn.to_torch(t)
            if args.residual_layout == "replicated":
                return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
            if args.residual_layout == "stream":
                return torch.cat([ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)], -1)
            values = [ttnn.to_torch(x).reshape(1, 1, t.shape[2], 4, 640) for x in ttnn.get_device_tensors(t)]
            return torch.cat(values, -1).reshape(1, 1, t.shape[2], 10240)

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            if args.impl == "mesh":
                if dtype == ttnn.bfloat16 and x.shape[-1] == 10240:
                    if args.residual_layout == "replicated":
                        return model.ops.upload(x, dtype, layout)
                    return model.ops.shards(
                        list(x.chunk(4, -1)) if args.residual_layout == "stream" else _hidden_parts(x), dtype, layout
                    )
                return model.ops.upload(x, dtype, layout)
            return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)

        rng = torch.Generator().manual_seed(2026)
        x = (
            torch.randn(1, args.prefill_length, 2560, generator=rng).repeat(1, 1, 4) * 0.05
            + torch.randn(1, args.prefill_length, 10240, generator=rng) * 0.005
        ).bfloat16()
        tokens = torch.randint(cfg.vocab_size, (1, args.prefill_length), generator=rng, dtype=torch.int32)
        decode_x, decode_tokens = x[:, :1], tokens[:, :1]
        if args.recorded:
            fixture_path = (
                Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip")
                / args.recorded_tag
                / f"layer{args.layer}.pt"
            )
            fixture = torch.load(fixture_path, weights_only=True)
            x, decode_x = fixture["inputs"][:2]
            tokens, decode_tokens = fixture["tokens"][:2]
            if args.prefill_length > x.shape[1]:
                raise ValueError("Requested prefix exceeds recorded fixture")
            x, tokens = x[:, : args.prefill_length], tokens[:, : args.prefill_length]
        xd, dd = upload(x.unsqueeze(0)), upload(decode_x.unsqueeze(0))
        pd = upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        decode_position = torch.zeros(1, 32, dtype=torch.int32)
        decode_position[0, 0] = args.decode_position
        dp = upload(decode_position, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        slots = torch.zeros(1, 32, dtype=torch.int32)
        slots[0, 0] = 1
        sd = upload(slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        page_host = torch.randperm(cfg.max_position_embeddings // 32, generator=rng).reshape(1, -1).int()
        page = (
            upload(
                page_host,
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            if args.layer == 3
            else None
        )
        pt = upload(tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if args.layer == 1 else None
        dt = upload(decode_tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if args.layer == 1 else None
        common = dict(current_positions=pd, slots=sd, page_table=page)

        def prefill():
            with no_runtime_host_conversion():
                return model.prefill_forward(xd, tokens=pt, **common)

        def decode():
            with no_runtime_host_conversion():
                result = model.decode_forward(dd, tokens=dt, **dict(common, current_positions=dp))
                if args.advance_position:
                    ttnn.plus_one(dp)
                return result

        def drain():
            if args.profile:
                ttnn.ReadDeviceProfiler(mesh)

        def reset_decode_position():
            if args.advance_position:
                host = ttnn.from_torch(
                    decode_position,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if args.impl == "mesh" else None,
                )
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
        eager_prefill = prefill
        prefill_capture_ms = None
        prefill_trace_checks = []
        if args.trace_prefill:
            if args.check_prefill_trace:
                # Compile singleton seed signatures before a live prefill trace.
                seeded = model.prefill_forward(dd, tokens=dt, **common)
                ttnn.synchronize_device(mesh)
                del seeded
                eager_prefill()
                ttnn.synchronize_device(mesh)
            before = time.perf_counter()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            captured_prefill = eager_prefill()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.synchronize_device(mesh)
            prefill_capture_ms = (time.perf_counter() - before) * 1000

            def prefill():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                return captured_prefill

            for _ in range(2):
                prefill()
                ttnn.synchronize_device(mesh)
        for _ in range(1 if args.profile else 5):
            signpost("PERF_PREFILL")
            before = time.perf_counter()
            out = prefill()
            ttnn.synchronize_device(mesh)
            timings.append((time.perf_counter() - before) * 1000)
            signpost("PERF_PREFILL_END")
            drain()
            prefill_host = host_output(out)
            del out
        if args.trace_prefill:
            if args.check_prefill_trace:

                def copy_input(value, device, residual=False):
                    mapper = None
                    if args.impl == "mesh":
                        if residual:
                            assert args.residual_layout == "hidden"
                            value = torch.cat(_hidden_parts(value), dim=0)
                            mapper = ttnn.ShardTensorToMesh(mesh, dim=0)
                        else:
                            mapper = ttnn.ReplicateTensorToMesh(mesh)
                    host = ttnn.from_torch(value, dtype=device.dtype, layout=device.layout, mesh_mapper=mapper)
                    ttnn.copy_host_to_device_tensor(host, device)

                for changed in (False, True, False):
                    copy_input(x.flip(1).unsqueeze(0) if changed else x.unsqueeze(0), xd, residual=True)
                    if pt is not None:
                        copy_input(tokens.flip(1) if changed else tokens, pt)
                    new_slots = slots.clone()
                    if changed:
                        new_slots[0, 0] = 0
                    copy_input(new_slots, sd)
                    if page is not None:
                        copy_input(page_host.flip(1) if changed else page_host, page)

                    def seed_and_position():
                        position = torch.zeros(1, 32, dtype=torch.int32)
                        copy_input(position, pd)
                        if changed:
                            seeded = model.prefill_forward(dd, tokens=dt, **common)
                            ttnn.synchronize_device(mesh)
                            del seeded
                            position[0, 0] = 1
                            copy_input(position, pd)
                        appended = position.clone()
                        appended[0, 0] += args.prefill_length
                        copy_input(appended, dp)

                    seed_and_position()
                    expected = eager_prefill()
                    ttnn.synchronize_device(mesh)
                    expected_host = host_output(expected)
                    del expected
                    expected_decode = decode()
                    ttnn.synchronize_device(mesh)
                    expected_decode_host = host_output(expected_decode)
                    del expected_decode
                    if not changed and prefill_trace_checks:
                        assert torch.equal(first_a, expected_host), "Repeated A changed"
                    elif not changed:
                        first_a = expected_host.clone()
                    else:
                        assert not torch.equal(first_a, expected_host), "Changed inputs had no effect"
                    for repeat in range(2):
                        seed_and_position()
                        replayed = prefill()
                        ttnn.synchronize_device(mesh)
                        actual = host_output(replayed)
                        assert torch.equal(expected_host, actual), ("prefill replay mismatch", changed, repeat)
                        appended = decode()
                        ttnn.synchronize_device(mesh)
                        assert torch.equal(expected_decode_host, host_output(appended)), (
                            "appended decode mismatch",
                            changed,
                            repeat,
                        )
                        del appended
                        prefill_trace_checks.append(
                            dict(changed=changed, repeat=repeat, bitwise_equal=True, appended_decode_equal=True)
                        )
                    del replayed
            ttnn.release_trace(mesh, trace)
            trace = None
            del captured_prefill
            prefill = eager_prefill
            if args.check_prefill_trace:
                prefill()
                ttnn.synchronize_device(mesh)
        reset_decode_position()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        position_after_capture = int(
            (ttnn.to_torch(ttnn.get_device_tensors(dp)[0]) if args.impl == "mesh" else ttnn.to_torch(dp))[0, 0]
        )
        assert position_after_capture in (
            args.decode_position,
            args.decode_position + int(args.advance_position),
        ), position_after_capture
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
        final_position = int(
            (ttnn.to_torch(ttnn.get_device_tensors(dp)[0]) if args.impl == "mesh" else ttnn.to_torch(dp))[0, 0]
        )
        if args.advance_position:
            assert final_position == position_after_capture + (12 if args.profile else 52), (
                position_after_capture,
                final_position,
            )
        saved = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip_benchmark_outputs")
        saved.mkdir(parents=True, exist_ok=True)
        output_path = saved / (args.tag + ".pt")
        torch.save({"prefill": prefill_host, "decode": host_output(out)}, output_path)
        result = dict(
            impl=args.impl,
            layer=args.layer,
            tag=args.tag,
            seed=2026,
            policy=vars(args),
            candidate=globals().get("CANDIDATE"),
            command=[sys.executable, *sys.argv],
            advice_variant=os.environ.get("QWEN38_ADVICE_VARIANT"),
            runtime_classes=identities,
            class_sources=class_sources,
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
            prefill_capture_ms=prefill_capture_ms,
            prefill_trace_checks=prefill_trace_checks,
            prefill_trace_captures=int(args.trace_prefill),
            prefill_trace_releases=int(args.trace_prefill),
            prefill_trace_replays=(7 + len(prefill_trace_checks)) if args.trace_prefill else 0,
            watcher=os.environ.get("TT_METAL_WATCHER"),
            skip_program_cache_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
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
