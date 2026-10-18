# SPDX-License-Identifier: Apache-2.0
"""TEST ONLY: current RM versus full-sync register-resident FP32 recurrence.

No HTTP, model scores, profiler, or production mutation. Root owns device runs.
Isolated recurrence timings include unchanged normalized Q/K TILE-to-RM bridges.
Whole-GDN timings provide the relevant denominator before any gain projection.
"""

import argparse
import gc
import hashlib
import inspect
import os
import statistics
import subprocess
import time
import traceback
import types
from pathlib import Path

import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_gdn_resident_candidate as candidate
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_view_probe as common
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_gdn_qk_rm_probe import Step
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.precision import apply_layer, for_layer, load_policy
from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_view import _decoder

VARIANTS = ("current_rm", "half_zero", "half_nan")


def bits(tensor):
    return tensor.contiguous().view(torch.uint8)


def digest(tensor):
    return hashlib.sha256(bits(tensor).numpy().tobytes()).hexdigest()


def compare(left, right):
    assert len(left) == len(right) == 4
    return [
        dict(
            chip=i,
            exact_bits=torch.equal(bits(a), bits(b)),
            finite=bool(torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()),
            max_abs=float((a.double() - b.double()).abs().max()),
        )
        for i, (a, b) in enumerate(zip(left, right))
    ]


def require_exact(rows):
    assert all(row["exact_bits"] and row["finite"] for row in rows), rows


def select(view, name):
    view.attention._half_column_poison = 0x7FC12345 if name == "half_nan" else 0
    view.attention._recurrence = types.MethodType(
        candidate.gdn_method(candidate.baseline._rm_recurrence if name == "current_rm" else candidate.RECURRENCE),
        view.attention,
    )


def measure(mesh, forward, reset, read, repeats):
    """One live trace; no device allocation from read/reset while trace is live."""
    trace = None
    try:
        reset(-1)
        forward()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        timings = []
        for replay in range(repeats):
            reset(replay)
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            timings.append(time.perf_counter() - begin)
        ttnn.release_trace(mesh, trace)
        trace = None
        # Host copies/state diagnostics occur after trace release.
        raw = read()
        return dict(seconds=timings, median_seconds=statistics.median(timings), trace_released=True), raw
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)


def state_record(owner):
    """All public slots, including high slots above the serving32, are checked."""
    result, raw = {}, {}
    for name in ("recurrent_bank", "conv_bank"):
        value = getattr(owner.attention, name)
        rows, active = [], []
        for chip, tensor in enumerate(common.read_shards(value)):
            other = torch.cat((tensor[: common.SLOT], tensor[common.SLOT + 1 :]))
            rows.append(
                dict(
                    chip=chip,
                    full_sha256=digest(tensor),
                    inactive_sha256=digest(other),
                    inactive_canary_exact=bool(torch.all(other == 0.125)),
                )
            )
            active.append(tensor[common.SLOT].clone())
        result[name] = dict(shape=list(value.shape), rows=rows)
        raw[name] = active
    return result, raw


def run(args, cfg, policy, report, save):
    mesh = None
    variants = VARIANTS[:1] if args.baseline_only else VARIANTS
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region)
        grid = mesh.compute_with_storage_grid_size()
        report["grid"] = dict(x=grid.x, y=grid.y)
        assert grid.x * grid.y >= 96
        owner = MultichipDecoder.from_state_dict(
            CheckpointState(args.checkpoint),
            hf_config=cfg,
            layer_idx=0,
            mesh_device=mesh,
            batch_size=32,
            max_requests=64,
            chunk_size=128,
            attention_pool_tokens=common.PUBLIC_TOKENS + 32 * 512,
            cache_dir=args.weight_cache,
        )
        apply_layer(owner, for_layer(policy, 0))
        owner.prepare([1])
        view = _decoder(owner, {}, [1, 128, 512])
        assert view.attention.prefill_qk_rm_enabled and view.attention.recurrence_broadcast
        assert view.attention.recurrent_bank is owner.attention.recurrent_bank
        report["actual_precision"] = dict(
            packed_weight=str(owner.attention.packed_weight.dtype),
            output_weight=str(owner.attention.output_projection.row_weight.dtype),
            state=str(owner.attention.recurrent_bank.dtype),
            compute={
                k: str(getattr(owner.attention.compute, k))
                for k in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc", "dst_full_sync_en")
            },
        )
        rng = torch.Generator().manual_seed(81032)
        hidden = (torch.randn(4, 1, 1024, 2560, generator=rng) * 0.05).bfloat16()
        tokens = torch.randint(0, cfg.vocab_size, (1024,), generator=rng).int()
        torch.save(dict(hidden=hidden, tokens=tokens), args.output / "inputs.pt")
        common.SLOT = 19
        lengths = [int(v) for v in args.lengths.split(",")]
        assert lengths and all(n in (128, 256, 512) for n in lengths)
        steps = {n: Step(view, n, hidden, tokens) for n in lengths}
        # Persistent outputs and host buffers exist before the first capture.
        isolated_outputs = {
            n: ttnn.empty(
                (1, 12, n, 128),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for n in lengths
        }
        banks = {
            name: torch.full(
                tuple(getattr(owner.attention, name).shape),
                0.125,
                dtype=torch.float32 if name == "recurrent_bank" else torch.bfloat16,
            )
            for name in ("recurrent_bank", "conv_bank")
        }
        normalized = {}
        for n, step in steps.items():
            common.reset(owner, view)
            step.refresh(0)
            captured = []

            def retain(model, q, k, v, g, beta):
                captured.append((q, k, v, g, beta))
                return candidate.baseline._rm_recurrence(model, q, k, v, g, beta)

            view.attention._recurrence = types.MethodType(candidate.gdn_method(retain), view.attention)
            step.forward()
            assert len(captured) == 1
            normalized[n] = captured[0]
            report.setdefault("operands", {})[str(n)] = [common.tensor_identity(t) for t in captured[0]]
        # Retained normalized operands came from selected real weights/normalization.
        normalized_host = {n: [common.read_shards(t) for t in values] for n, values in normalized.items()}
        torch.save(normalized_host, args.output / "normalized_operands.pt")
        initial = (torch.randn(1, 12, 128, 128, generator=rng) * 0.01).float()
        report["isolated"] = []
        report["whole_gdn"] = []
        for n in lengths:
            q, k, v, g, beta = normalized[n]
            for trial in (0, 1):
                case = dict(sequence=n, trial=trial, variants={})
                report["isolated"].append(case)
                raw = {}
                # Changed q bits and initial state between trials expose stale replay inputs.
                q_host = torch.cat(normalized_host[n][0], dim=0) * (-1 if trial else 1)
                state_host = initial * (2 if trial else 1)

                def reset_isolated(replay):
                    common.host_copy(state_host * (1 + 0.125 * max(0, replay)), view.attention.recurrent_state, mesh)
                    common.host_copy(q_host * (-1 if replay % 2 else 1), q, mesh, sharded=True)

                for name in variants:
                    select(view, name)
                    fn = candidate.baseline._rm_recurrence if name == "current_rm" else candidate.RECURRENCE

                    def forward_isolated():
                        output, final = fn(view.attention, q, k, v, g, beta)
                        assert final is view.attention.recurrent_state
                        ttnn.copy(output, isolated_outputs[n])

                    timing, result = measure(
                        mesh,
                        forward_isolated,
                        reset_isolated,
                        lambda: dict(
                            output=common.read_shards(isolated_outputs[n]),
                            state=common.read_shards(view.attention.recurrent_state),
                        ),
                        args.repeats,
                    )
                    case["variants"][name] = timing
                    raw[name] = result
                    save()
                path = args.output / f"isolated_s{n}_trial{trial}.pt"
                torch.save(raw, path)
                case["raw"] = dict(path=str(path), sha256=common.sha(path))
                case["comparisons"] = {
                    name: {key: compare(raw["current_rm"][key], raw[name][key]) for key in ("output", "state")}
                    for name in variants[1:]
                }
                save()
                for group in case["comparisons"].values():
                    for rows in group.values():
                        require_exact(rows)
                case["speedup"] = (
                    None
                    if args.baseline_only
                    else case["variants"]["current_rm"]["median_seconds"]
                    / case["variants"]["half_zero"]["median_seconds"]
                )
            # Whole chain includes real normalization, Q/K bridges, conv and projection.
            step = steps[n]
            for start in (0, 128):
                case = dict(sequence=n, position=start, slot=common.SLOT, variants={})
                report["whole_gdn"].append(case)
                raw = {}

                def reset_whole(replay):
                    for key, value in banks.items():
                        common.host_copy(value, getattr(owner.attention, key), mesh)
                    step.refresh(start)

                for name in variants:
                    select(view, name)

                    def read_whole():
                        state, selected = state_record(owner)
                        return dict(output=common.read_shards(step.output), active_state=selected, public_state=state)

                    timing, result = measure(mesh, step.forward, reset_whole, read_whole, args.repeats)
                    case["variants"][name] = timing
                    raw[name] = result
                    save()
                path = args.output / f"whole_s{n}_position{start}.pt"
                torch.save(raw, path)
                case["raw"] = dict(path=str(path), sha256=common.sha(path))
                case["comparisons"] = {
                    name: dict(
                        output=compare(raw["current_rm"]["output"], raw[name]["output"]),
                        active_state={
                            key: compare(values, raw[name]["active_state"][key])
                            for key, values in raw["current_rm"]["active_state"].items()
                        },
                        public_state_equal=raw["current_rm"]["public_state"] == raw[name]["public_state"],
                    )
                    for name in variants[1:]
                }
                case["public_state"] = {name: values["public_state"] for name, values in raw.items()}
                save()
                for group in case["comparisons"].values():
                    require_exact(group["output"])
                    for rows in group["active_state"].values():
                        require_exact(rows)
                    assert group["public_state_equal"], group
                assert all(
                    row["inactive_canary_exact"]
                    for values in raw.values()
                    for bank in values["public_state"].values()
                    for row in bank["rows"]
                )
                case["speedup"] = (
                    None
                    if args.baseline_only
                    else case["variants"]["current_rm"]["median_seconds"]
                    / case["variants"]["half_zero"]["median_seconds"]
                )
                case["isolated_fraction_of_whole_baseline"] = (
                    report["isolated"][-2]["variants"]["current_rm"]["median_seconds"]
                    / case["variants"]["current_rm"]["median_seconds"]
                )
        report["memory_after_measurements"] = common.memory_view(mesh)
        report["status"] = (
            "baseline_measurements_only" if args.baseline_only else "passed_exact_output_state_and_replay"
        )
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            report["mesh_closed"] = True
            save()
        gc.collect()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path(f"/home/vkovacevic/qwen38-flash/hf-local/{common.REVISION}"))
    p.add_argument(
        "--weight-cache", default="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"
    )
    p.add_argument("--lengths", default="128,512")
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="Measure current recurrence share without compiling/running the candidate",
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--trace-region", type=int, default=400000000)
    args = p.parse_args()
    assert args.repeats >= 2
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    native_tracking = ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()
    assert native_tracking and trace_allocation_tracker.TRACE_ALLOC_TRACKING
    assert all(os.environ.get(n, "0") in ("", "0") for n in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER"))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cfg = Qwen4ExpConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    policy = load_policy(common.ROOT / "doc/datatype_sweep/selected_precision_config.json")
    assert cfg.max_position_embeddings == common.CONTEXT and policy["config_id"] == "combined_head4_gdn4_lofi"
    paths = [
        Path(__file__),
        Path(candidate.__file__),
        candidate.BODY_PATH,
        *[candidate.REPO / name for name in candidate.NATIVE_PINS],
        *[Path(__file__).with_name(name) for name in candidate.PINS],
        candidate.HEADER_PATH,
        Path(common.__file__),
        Path(inspect.getfile(Step)),
        Path(candidate.baseline.__file__),
        common.ROOT / "tt/multichip_decoder.py",
        common.ROOT / "tt/prefill_view.py",
    ]
    report = dict(
        status="running",
        sources={str(path): common.sha(path) for path in paths},
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        context=common.CONTEXT,
        policy=policy,
        trace_tracking=dict(
            native=bool(native_tracking),
            python=bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
            skip_program_cache=False,
            exemptions=[],
        ),
        candidate="full_sync_register_resident_half_columns",
        qualified_half_source_pins=candidate.PINS,
        native_dst_contract_pins=candidate.NATIVE_PINS,
        workers=dict(current_rm=48, half_columns=96),
        scope="single real layer0 GDN; no model logits/quality, no serving/roofline measurements; all64 public slots checked",
        timing="warmed trace submission through standalone completion synchronization, resets/readback excluded; isolated includes unchanged Q/K RM conversions and output copy; whole includes output copy",
        op_sequences=dict(
            current_rm="Q/K TILE→RM;48jobs;full4face FP32 recurrence and Accurate K128 SUM",
            half="same bridges/full CBs;96jobs;C-only faces0/2 binary;identical K128 order;SUM groups0/1;same row broadcast;stateDST0..3;fullsync no-clear;outputpack/token+finalstate;disjoint publicface writes",
        ),
    )

    def save():
        common.dump(args.output / "summary.json", report)

    for path in paths:
        (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())
    for name, source in (
        ("candidate_reader.cpp", candidate.READER),
        ("candidate_compute.cpp", candidate.COMPUTE),
        ("baseline_reader.cpp", candidate.baseline._RM_QK_READER),
        ("baseline_compute.cpp", candidate.baseline._BROADCAST_DECODE_COMPUTE),
    ):
        (args.output / name).write_text(source)
    save()
    try:
        run(args, cfg, policy, report, save)
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
