# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parent-serialized exact local convolution probe, or hooked full-layer runner.

Imports TTNN only after argument parsing in a requested device entrypoint.
Never resets devices; run under the parent's hardware/evidence wrapper.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def hashes():
    paths = (
        Path(__file__),
        ROOT / "tt/recurrent_next_candidate.py",
        ROOT / "tt/multichip_decoder.py",
        ROOT / "tt/optimized_decoder.py",
    )
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def full_layer(args, forwarded):
    from unittest.mock import patch

    from models.autoports.qwen_qwen3_6_27b.tests import run_multichip_decoder as runner
    from models.autoports.qwen_qwen3_6_27b.tt.recurrent_next_candidate import local_conv

    assert not args.poison_dead_padding, "Use focused poison mode; full-layer timing must remain unpoisoned"
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    assert "--baseline" not in forwarded, "Hook is for TP4 candidate only"
    assert "--legacy-decode-layout" not in forwarded
    original = runner.MultichipDecoder.from_state_dict
    instances = []
    state = {"calls": 0, "legacy_calls": 0}

    def create(cls, *positional, **keywords):
        model = original(*positional, **keywords)
        assert model.kind == "linear_attention", "Choose a GDN layer"
        assert model.policy.get("flat_decode") and model.policy.get("batched_conv")
        original_conv = model._local_batched_conv

        def hooked(token, history):
            if tuple(token.shape) != (1, history.shape[0], 2560):
                # Legacy-shaped prefill tails retain the existing implementation.
                state["legacy_calls"] += 1
                return original_conv(token, history)
            state["calls"] += 1
            return local_conv(token, history, model.taps_rm, cores=args.cores)

        model._local_batched_conv = hooked
        instances.append(model)
        return model

    failure = None
    try:
        with (
            patch.object(runner.MultichipDecoder, "from_state_dict", classmethod(create)),
            patch.object(sys, "argv", [sys.argv[0], *forwarded, "--output", str(args.output)]),
        ):
            runner.main()
        assert instances and state["calls"] > 0, "Hook not exercised; B1 requires batched_conv_b1=True"
    except BaseException:
        failure = traceback.format_exc()
        raise
    finally:
        evidence = {
            "kind": "local_conv_instance_hook",
            "calls_during_eager_or_capture": state["calls"],
            "legacy_prefill_calls": state["legacy_calls"],
            "cores": args.cores,
            "source_sha256": hashes(),
            "failure": failure,
            "forwarded_arguments": forwarded,
        }
        if args.output.exists():
            report = json.loads(args.output.read_text())
            report["conv_candidate"] = evidence
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"status": "fail", "conv_candidate": evidence}, indent=2) + "\n")


def focused(args):
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder
    from models.autoports.qwen_qwen3_6_27b.tt.recurrent_next_candidate import local_conv

    torch.set_num_threads(8)
    report = {
        "status": "running",
        "arguments": vars(args) | {"output": str(args.output)},
        "source_sha256": hashes(),
        "cases": [],
        "fixture": "seeded synthetic BF16 tokens/taps; nonzero distinct rank/user/channel history",
        "scope": "untilize + convolution + SiLU + raw history update",
        "environment": {
            k: os.environ.get(k)
            for k in ("TT_METAL_WATCHER", "TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_DEVICE_PROFILER")
        },
        "timing": "alternating order, queued traces with endpoint synchronization; disabled in poison mode",
        "trace_allocation_lifetime": "all fixtures and candidate output preallocated; both ops warmed before either capture",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def upload(values, layout):
        return ttnn.from_torch(
            torch.cat(values, -1).contiguous(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )

    def read(tensor):
        return [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(tensor)]

    def check(got, expected, batch, label, entry):
        result = []
        assert len(got) == len(expected) == 4
        for rank, (a, b) in enumerate(zip(got, expected)):
            assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
            a, b = a.reshape(batch, -1), b.reshape(batch, -1)
            for user in range(batch):
                result.append(
                    {
                        "rank": rank,
                        "user": user,
                        "bit_mismatches": int((a[user].view(torch.int16) != b[user].view(torch.int16)).sum()),
                        "finite": bool(torch.isfinite(a[user]).all() and torch.isfinite(b[user]).all()),
                    }
                )
        entry[label] = result
        save()
        assert all(r["finite"] and r["bit_mismatches"] == 0 for r in result), label

    def run_case(batch, cores, mode):
        entry = {"batch": batch, "cores": cores, "mode": mode, "eager": [], "trace": []}
        report["cases"].append(entry)
        save()
        print(f"CONV begin B{batch} cores={cores} mode={mode} poison={args.poison_dead_padding}", flush=True)
        generators = [torch.Generator().manual_seed(20261113 + batch * 97 + rank) for rank in range(4)]
        users = torch.arange(batch).reshape(batch, 1, 1) * 0.0078125
        channels = (torch.arange(2560).reshape(1, 1, 2560) % 31) * 0.001953125
        initial = [
            (torch.randn(batch, 3, 2560, generator=g) * 0.25 + users + channels + rank * 0.03125).bfloat16()
            for rank, g in enumerate(generators)
        ]
        if mode == "zero_history":
            initial = [x.zero_() for x in initial]
        initial_t = upload(initial, ttnn.ROW_MAJOR_LAYOUT)
        histories = [upload(initial, ttnn.ROW_MAJOR_LAYOUT) for _ in range(2)]
        taps = [
            upload(
                [(torch.randn(1, 1, 2560, generator=g) * 0.125 + (tap - 1.5) * 0.03125).bfloat16() for g in generators],
                ttnn.ROW_MAJOR_LAYOUT,
            )
            for tap in range(4)
        ]
        bank = []
        for step in range(args.steps):
            tokens = [
                (
                    torch.randn(1, batch, 2560, generator=g) * 0.25
                    + users.reshape(1, batch, 1)
                    + channels
                    + rank * 0.0625
                    + step * 0.03125
                ).bfloat16()
                for rank, g in enumerate(generators)
            ]
            if mode == "zero_qkv":
                tokens = [x.zero_() for x in tokens]
            bank.append(upload(tokens, ttnn.TILE_LAYOUT))
        token = ttnn.empty_like(bank[0])
        # Capture 2 must not leave any new live allocation behind for trace 1.
        # The helper's existing output parameter lets us own this buffer before
        # either capture; its untilize temporary dies when local_conv returns.
        candidate_output = ttnn.empty(
            (batch, 1, 2560),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        owner = SimpleNamespace(taps_rm=taps, policy={"local_fused_conv": False})
        traces, trace_outputs = [], []
        allocations = [initial_t, *histories, *taps, *bank, token, candidate_output]

        def invoke(index):
            if index == 0:
                return MultichipDecoder._local_batched_conv(owner, token, histories[0])
            return local_conv(
                token,
                histories[1],
                taps,
                output=candidate_output,
                cores=cores,
                poison_dead_padding=args.poison_dead_padding,
            )

        def reset():
            for history in histories:
                ttnn.copy(initial_t, history)

        try:
            eager = []
            for step in range(args.steps):
                ttnn.copy(bank[step], token)
                outputs = [invoke(index) for index in range(2)]
                values, states = [read(o) for o in outputs], [read(h) for h in histories]
                record = {"step": step}
                entry["eager"].append(record)
                check(values[1], values[0], batch, "output", record)
                check(states[1], states[0], batch, "history", record)
                eager.append((values[0], states[0]))
                ttnn.deallocate(outputs[0])
                del outputs
            reset()
            ttnn.copy(bank[0], token)
            for index in range(2):
                for _ in range(args.warmup):
                    warm = invoke(index)
                    ttnn.synchronize_device(mesh)
                    if index == 0:
                        ttnn.deallocate(warm)
                    del warm
            # Complete all eager/program-cache allocations before trace 1.
            # Capture 2 creates only helper-local temporaries, released before
            # replay, and writes the candidate output allocated above. No
            # allocation-tracker acknowledgement or corruptible scope is used.
            ttnn.synchronize_device(mesh)
            for index in range(2):
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = invoke(index)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                traces.append(tid)
                trace_outputs.append(out)
            reset()
            for step in range(args.steps):
                ttnn.copy(bank[step], token)
                for tid in traces:
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                record = {"step": step}
                entry["trace"].append(record)
                for index, label in enumerate(("baseline", "candidate")):
                    check(read(trace_outputs[index]), eager[step][0], batch, label + "_output_vs_eager", record)
                    check(read(histories[index]), eager[step][1], batch, label + "_history_vs_eager", record)
            if not args.poison_dead_padding:
                reset()
                ttnn.copy(bank[0], token)
                times = [[], []]
                for tid in traces:
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                for sample in range(args.timing_samples):
                    for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter_ns()
                        for _ in range(args.replays_per_sample):
                            ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        times[index].append((time.perf_counter_ns() - start) / 1000 / args.replays_per_sample)
                entry["trace_us"] = {
                    label: {"samples": samples, "median": statistics.median(samples)}
                    for label, samples in zip(("baseline", "candidate"), times)
                }
                check(read(trace_outputs[1]), read(trace_outputs[0]), batch, "timed_output", entry)
                check(read(histories[1]), read(histories[0]), batch, "timed_history", entry)
            entry["pass"] = True
            save()
            print(f"CONV pass B{batch} cores={cores} mode={mode}", flush=True)
        finally:
            for tid in traces:
                ttnn.release_trace(mesh, tid)
            # Candidate output is already owned by allocations, while the
            # baseline output is created during the first capture.
            for tensor in [*trace_outputs[:1], *allocations]:
                ttnn.deallocate(tensor)

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 * 1024 * 1024)
        cases = [(batch, args.cores) for batch in args.batches]
        if args.low_cores and (3, args.low_cores) not in cases:
            cases.append((3, args.low_cores))
        for batch, cores in cases:
            for mode in args.modes:
                run_case(batch, cores, mode)
        report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--full-layer", action="store_true", help="Forward unknown args to run_multichip_decoder")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 3, 17, 31, 32])
    parser.add_argument("--modes", nargs="+", choices=("normal", "zero_qkv", "zero_history"), default=["normal"])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cores", type=int)
    parser.add_argument("--low-cores", type=int, default=1, help="Additional B3 case; zero disables")
    parser.add_argument("--poison-dead-padding", action="store_true", help="BF16 NaNs in dead tail half; no timing")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing-samples", type=int, default=5)
    parser.add_argument("--replays-per-sample", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    assert all(1 <= b <= 32 for b in args.batches)
    assert args.steps >= 2 and args.warmup >= 1 and args.timing_samples >= 1 and args.replays_per_sample >= 1
    assert args.cores is None or args.cores >= 1
    assert args.low_cores >= 0
    if args.full_layer:
        full_layer(args, forwarded)
    else:
        assert not forwarded, forwarded
        focused(args)


if __name__ == "__main__":
    main()
