# SPDX-License-Identifier: Apache-2.0
"""Isolated exact-byte GDN reader experiment; never serving instrumentation.

Run with TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0,
exclusive TP4 ownership, no live profiler/watcher. Captures never coexist. Explicit
synchronization and readbacks are standalone diagnostic boundaries only.
"""

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
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as decoder
from models.autoports.qwen_qwen3_8_flash_next.tt import precision
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState


def candidate_reader(source, prefill):
    """Keep tile scratch coordinates and all arithmetic byte-for-byte unchanged."""
    for which, name in ((1, "q"), (2, "k"), (3, "v")):
        old = f"if (which=={which}) noc.async_read({name},CoreLocalMem<uint32_t>(scratch),4096,{{.page_id=page}},{{}});"
        row = "h%32" if which == 3 else "h/3"
        new = f"""if (which=={which}) {{
                    const uint32_t row={row};
                    for(uint32_t face=0;face<2;++face) {{
                        const uint32_t offset=off(row,face*16)*4;
                        noc.async_read({name},CoreLocalMem<uint32_t>(scratch+offset),64,
                            {{.page_id=page,.offset_bytes=offset}},{{}});
                    }}
                }}"""
        if source.count(old) != 1:
            raise ValueError(f"Expected one unmodified {name} reader")
        source = source.replace(old, new)
    page = "(b*st+token/32)+h/32" if prefill else "b+h/32"
    row = "token%32" if prefill else "0"
    for which, name in ((4, "decay"), (5, "beta")):
        condition = "if(which==4)" if which == 4 else "else"
        old = f"{condition} noc.async_read({name},CoreLocalMem<uint32_t>(scratch),4096,{{.page_id={page}}},{{}});"
        new = f"""{condition} {{
                const uint32_t offset=(off({row},h%32)*4/32)*32;
                noc.async_read({name},CoreLocalMem<uint32_t>(scratch+offset),32,
                    {{.page_id={page},.offset_bytes=offset}},{{}});
            }}"""
        if source.count(old) != 1:
            raise ValueError(f"Expected one unmodified {name} reader")
        source = source.replace(old, new)
    return source


def broadcast_reader(source):
    old = """for(uint32_t c=0;c<16;++c) {
                            dst[tile*1024+off(r,c)]=value;
                            dst[tile*1024+off(r,c+16)]=value;
                        }"""
    if source.count(old) != 1:
        raise ValueError("Expected one Q/K column replication loop")
    source = source.replace(old, "dst[tile*1024+off(r,0)]=value;")
    old = "for(uint32_t i=0;i<1024;++i) dst[i]=value;"
    if source.count(old) != 1:
        raise ValueError("Expected one decay/beta replication loop")
    return source.replace(old, "dst[0]=value;")


def broadcast_compute(source):
    old = """    copy_init(cb);
    copy_tile(cb,tile,dst);"""
    if source.count(old) != 1:
        raise ValueError("Expected one unmodified load helper")
    source = '#include "api/compute/bcast.h"\n' + source
    return source.replace(
        old,
        """    if(cb==1 || cb==2) {
        unary_bcast_init<BroadcastType::COL>(cb);
        unary_bcast<BroadcastType::COL>(cb,tile,dst);
        unary_bcast_uninit<BroadcastType::COL>(cb);
    } else if(cb==4 || cb==5) {
        unary_bcast_init<BroadcastType::SCALAR>(cb);
        unary_bcast<BroadcastType::SCALAR>(cb,tile,dst);
        unary_bcast_uninit<BroadcastType::SCALAR>(cb);
    } else {
        copy_init(cb);
        copy_tile(cb,tile,dst);
    }""",
    )


def host(value):
    return [ttnn.to_torch(x).clone() for x in ttnn.get_device_tensors(value)]


def snapshot(model, output):
    return {
        name: host(value)
        for name, value in {
            "output": output,
            "recurrent_state": model.recurrent_state,
            "conv_state": model.conv_state,
            "recurrent_bank": model.recurrent_bank,
            "conv_bank": model.conv_bank,
        }.items()
    }


def compare(reference, actual):
    result = {}
    for name in reference:
        result[name] = []
        for ref, got in zip(reference[name], actual[name]):
            result[name].append(
                {
                    "exact": torch.equal(ref, got),
                    "finite": bool(torch.isfinite(got).all()),
                    "max_abs": float((ref.float() - got.float()).abs().max()),
                    "shape": list(got.shape),
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--cases", nargs="+", default=["1x128", "1x256", "32x1"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--candidate", choices=("row-fragment", "broadcast"), default="row-fragment")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("positive repeats and warmup required")
    tracking = {
        "native_enabled": bool(ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()),
        "python_enabled": bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
        "skip_program_cache_environment": os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        "replay_wrapper_module": ttnn.execute_trace.__module__,
        "exemptions": [],
    }
    if not tracking["native_enabled"] or not tracking["python_enabled"]:
        raise RuntimeError("Set TT_METAL_TRACE_ALLOC_TRACKING=1 before importing ttnn; actual tracking is disabled")
    if tracking["skip_program_cache_environment"] != "0":
        raise RuntimeError("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 required; no cache exemptions")
    for key in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_WATCHER_DUMP_ALL"):
        if os.environ.get(key, "0") not in ("", "0"):
            raise RuntimeError(f"Disable {key} for this standalone diagnostic")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    original = (decoder._LOCAL_GDN_READER, decoder._LOCAL_PREFILL_READER, decoder._LOCAL_DECODE_COMPUTE)
    candidate = (
        (broadcast_reader(original[0]), broadcast_reader(original[1]), broadcast_compute(original[2]))
        if args.candidate == "broadcast"
        else (candidate_reader(original[0], False), candidate_reader(original[1], True), original[2])
    )
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    if cfg.layer_types[args.layer] != "linear_attention":
        raise ValueError("Choose a GDN layer")
    policy = precision.for_layer(precision.load_policy(), args.layer)
    report = {
        "status": "started",
        "scope": "isolated real-weight GDN attention, not full decoder/model benchmark",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "precision": policy,
        "trace_tracking": tracking,
        "sources": {},
        "results": [],
        "operand_bytes_per_job": {"baseline": 45056, "candidate": 45056 if args.candidate == "broadcast" else 1216},
        "qk_decay_beta_reader_stores_per_job": {
            "baseline": 10240,
            "candidate": 258 if args.candidate == "broadcast" else 10240,
        },
        "timing": "host execute_trace through standalone synchronize_device; state reset/readbacks excluded",
    }
    for path in (Path(__file__), Path(decoder.__file__), Path(precision.__file__)):
        report["sources"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())
    for label, sources in (("baseline", original), ("candidate", candidate)):
        for mode, source in zip(("decode", "prefill", "compute"), sources):
            (args.output / f"{label}_{mode}.cpp").write_text(source)

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=268435456)
    try:
        report["device_ids"] = list(mesh.get_device_ids())
        for case in args.cases:
            batch, seq = map(int, case.split("x"))
            if (batch, seq) not in ((1, 128), (1, 256), (32, 1)):
                raise ValueError(f"Unsupported case {case}")
            ops = decoder.MeshOps(mesh)
            ops.precision_ccl = policy["ccl_dtype"]
            state = CheckpointState(args.checkpoint, f"model.language_model.layers.{args.layer}.linear_attn.")
            model = decoder.ParallelGatedDeltaNet(
                state,
                cfg,
                ops,
                batch_size=batch,
                max_requests=batch,
                chunk_size=seq,
                allocate_column_weight=False,
            )
            precision.cast_attr(model, "packed_weight", policy["weight_groups"]["gdn_input"])
            precision.cast_attr(model, "out_weight", policy["weight_groups"]["gdn_output"])
            model.projection_compute = precision.compute(mesh, policy["compute_fidelities"]["gdn_input"])
            model.output_projection.row_weight = model.out_weight
            model.output_projection.compute = precision.compute(mesh, policy["compute_fidelities"]["gdn_output"])
            # B1 prefill and B32 decode both use row-parallel output in production.
            rng = torch.Generator().manual_seed(2026101608 + batch + seq)
            input_host = torch.randn(batch, seq, 2560, generator=rng).bfloat16() * 0.2
            recurrent_host = [torch.randn(batch, 12, 128, 128, generator=rng) * 0.01 for _ in range(4)]
            conv_host = [torch.randn(batch, 3, 2560, generator=rng).bfloat16() * 0.03 for _ in range(4)]
            value = ops.upload(input_host)
            initial_recurrent = ops.shards(recurrent_host, ttnn.float32)
            initial_conv = ops.shards(conv_host)
            slot_host = torch.zeros(batch, 32, dtype=torch.int32)
            position_host = slot_host.clone()
            slot_host[:, 0] = torch.arange(batch - 1, -1, -1)
            position_host[:, 0] = torch.arange(batch) + 17
            slots = ops.upload(slot_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            positions = ops.upload(position_host, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            torch.save(
                {
                    "input": input_host,
                    "initial_recurrent": recurrent_host,
                    "initial_conv": conv_host,
                    "slots": slot_host,
                    "positions": position_host,
                },
                args.output / f"{case}_inputs.pt",
            )

            def reset():
                ttnn.copy(initial_recurrent, model.recurrent_bank)
                ttnn.copy(initial_conv, model.conv_bank)

            def call():
                fn = model.decode_forward if seq == 1 else model.prefill_forward
                return fn(value, current_positions=positions, slots=slots)

            # Compile both variants before any capture. Temporary outputs are
            # destroyed before capture, preserving allocation-tracker lifetimes.
            for sources in (original, candidate):
                decoder._LOCAL_GDN_READER, decoder._LOCAL_PREFILL_READER, decoder._LOCAL_DECODE_COMPUTE = sources
                reset()
                output = call()
                ttnn.synchronize_device(mesh)
                del output
            row = {"case": case, "variants": {}, "comparisons": {}}
            report["results"].append(row)
            reference = None
            for label, sources in (("baseline", original), ("candidate", candidate)):
                decoder._LOCAL_GDN_READER, decoder._LOCAL_PREFILL_READER, decoder._LOCAL_DECODE_COMPUTE = sources
                trace = None
                output = None
                try:
                    reset()
                    ttnn.synchronize_device(mesh)
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    output = call()
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    times = []
                    for repeat in range(args.warmup + args.repeats):
                        reset()
                        ttnn.synchronize_device(mesh)
                        begin = time.perf_counter()
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        elapsed = time.perf_counter() - begin
                        if repeat >= args.warmup:
                            times.append(elapsed)
                            actual = snapshot(model, output)
                            torch.save(actual, args.output / f"{case}_{label}_{repeat - args.warmup}.pt")
                            if reference is None:
                                reference = actual
                            metrics = compare(reference, actual)
                            row["comparisons"][f"{label}_{repeat - args.warmup}"] = metrics
                            save()
                            if not all(x["exact"] and x["finite"] for chips in metrics.values() for x in chips):
                                raise AssertionError(f"Output/state mismatch: {case}/{label}/{repeat}")
                    row["variants"][label] = {"seconds": times, "median_seconds": statistics.median(times)}
                finally:
                    if trace is not None:
                        ttnn.release_trace(mesh, trace)
                    del output
                save()
            row["speedup"] = (
                row["variants"]["baseline"]["median_seconds"] / row["variants"]["candidate"]["median_seconds"]
            )
            print("GDN_ROW_FRAGMENT", json.dumps(row), flush=True)
            del model, ops, value, initial_recurrent, initial_conv, slots, positions
            save()
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        decoder._LOCAL_GDN_READER, decoder._LOCAL_PREFILL_READER, decoder._LOCAL_DECODE_COMPUTE = original
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
