# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test-only B32 compensated input norm versus 32 production singleton calls.

CPU mode never imports TTNN. Acceptance uses allocation tracking; benchmark is a
separate untracked run with synchronized wall timing and no profiler. No model,
server, precision policy, selected context, or production implementation changes.
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
KEY = "input_layernorm.weight"


def finite_tilize_bits(value):
    """Measured BF16 transport rule, restricted to this probe's finite inputs.

    Native padded concat untilizes and retilizes each group (concat.cpp's
    build_untilize_rm_retilize_concat). The independent all65536-pattern
    hidden-row-transport capture observes exp==0 -> +0; normal finite words
    are unchanged. This is an exact input-transport oracle, not a tolerance
    or arithmetic change. Every downstream stage still uses raw singleton bits.
    """
    import torch

    assert value.dtype == torch.bfloat16 and torch.isfinite(value).all()
    words = value.contiguous().view(torch.int16)
    return torch.where((words & 0x7F80) == 0, 0, words).to(torch.int16).view(torch.bfloat16)


def validate_transport_evidence():
    """Host-only cross-check against saved independent measurements, all ranks."""
    import torch

    path = ROOT / "readiness_vllm/optimized_hidden_row_transport/result_maps.pt"
    mapping = torch.load(path, weights_only=True, map_location="cpu")
    counts = []
    for rank in range(4):
        # This evidence stores unsigned BF16 bit patterns in INT32 tensors.
        assert mapping["uploaded"].dtype == mapping["old"].dtype == torch.int32
        uploaded = mapping["uploaded"][rank].reshape(-1).to(torch.int16).view(torch.bfloat16)
        observed = mapping["old"][rank].reshape(-1).to(torch.int16).view(torch.bfloat16)
        assert torch.unique(uploaded.view(torch.int16)).numel() == 65536
        finite = torch.isfinite(uploaded)
        assert int(finite.sum()) == 65280
        expected = finite_tilize_bits(uploaded[finite])
        assert torch.equal(expected.view(torch.int16), observed[finite].contiguous().view(torch.int16))
        counts.append(int((expected.view(torch.int16) != uploaded[finite].contiguous().view(torch.int16)).sum()))
    assert counts == [255] * 4  # 254 signed subnormal patterns and negative zero.
    return dict(
        artifact=str(path),
        finite_patterns_per_rank=65280,
        ranks=4,
        finite_changed_patterns_per_rank=counts,
        rule="BF16 exp==0 -> +0; all finite normal bits unchanged",
    )


def fixtures(args):
    import torch
    from safetensors import safe_open

    payload = torch.load(args.activations, map_location="cpu", weights_only=True)
    assert payload["metadata"]["checkpoint_revision"] == REVISION
    layer = payload["layers"][str(args.layer)]
    # Preserve individual real rows, but choose distinct values across recorded
    # steps/prefill positions: layer0 embeddings repeat for identical tokens.
    selected, coordinates, seen = [], [], set()
    for kind in ("xd", "x"):
        tensor = layer[kind]
        for index, row in enumerate(tensor.reshape(-1, 2048)):
            bits = row.contiguous().view(torch.int16).numpy().tobytes()
            if bits not in seen:
                seen.add(bits)
                selected.append(row.clone())
                coordinates.append(dict(key=kind, flattened_row=index, original_shape=list(tensor.shape)))
            if len(selected) == 64:
                break
        if len(selected) == 64:
            break
    assert len(selected) == 64, "Need 64 distinct recorded rows; do not silently substitute random inputs"
    cases = [
        ("recorded_a", torch.stack(selected[:32]).unsqueeze(0)),
        ("recorded_b", torch.stack(selected[32:]).unsqueeze(0)),
    ]
    # Finite edge rows test zero, signed zero/subnormals, alternating signs,
    # powers of two, and near-BF16-rounding boundaries. Transport is observed.
    edge = cases[0][1].clone()
    edge[0, 0] = 0
    edge[0, 1] = -0.0
    edge[0, 2] = torch.arange(2048, dtype=torch.int16).view(torch.bfloat16)
    edge[0, 3] = torch.tensor([1.0, -1.0], dtype=torch.bfloat16).repeat(1024)
    edge[0, 4] = torch.tensor([2.0**-12, 2.0**12], dtype=torch.bfloat16).repeat(1024)
    edge[0, 5] = torch.tensor([0x3F7F, 0x3F80, 0x3F81, 0x3F82], dtype=torch.int16).view(torch.bfloat16).repeat(512)
    cases.append(("finite_edges", edge))
    config = json.loads((args.snapshot / "config.json").read_text())["text_config"]
    assert config["layer_types"][args.layer] == "linear_attention"
    assert config["hidden_size"] == 2048 and config["rms_norm_eps"] == 1e-6
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    name = f"model.language_model.layers.{args.layer}.{KEY}"
    with safe_open(args.snapshot / index[name], framework="pt", device="cpu") as handle:
        # Production upload adds 1 in FP32; gamma remains FP32 on device.
        gamma = (handle.get_tensor(name).float() + 1).reshape(1, 1, 2048)
    assert gamma.dtype == torch.float32 and all(torch.isfinite(x).all() for _, x in cases)
    assert all(x.dtype == torch.bfloat16 and tuple(x.shape) == (1, 32, 2048) for _, x in cases)
    return (
        cases,
        gamma,
        dict(
            activations=str(args.activations),
            layer=args.layer,
            revision=REVISION,
            checkpoint=str(args.snapshot),
            weight=name,
            rows=coordinates,
            recorded_rows_distinct=64,
            edge_rows_derived=True,
        ),
    )


def cpu_mapping():
    import torch

    index = torch.arange(32 * 2048).reshape(1, 32, 2048)
    candidate = index.reshape(1, 32, 64, 32).permute(0, 2, 1, 3).reshape(4, 16, 32, 32)
    oracle = torch.cat([index[:, i : i + 1].reshape(4, 16, 1, 32) for i in range(32)], dim=2)
    assert torch.equal(candidate, oracle)
    assert not torch.equal(index.reshape(4, 16, 32, 32), oracle)
    return dict(
        pass_mapping=True,
        words=32 * 2048,
        missing_permute_negative_control=True,
        reduction="Existing fast_reduce_nc dims=[0,1], native dim1 then dim0; defaults unchanged",
        device_arithmetic_verified=False,
    )


def normalization(t, dec, x, epilogue, *, batched, debug):
    """Test candidate, or diagnostic copy of the existing singleton sequence."""
    rows = x.shape[-2]
    assert rows == (32 if batched else 1)
    memory = x.memory_config()
    wide = t.to_memory_config(x, t.DRAM_MEMORY_CONFIG)
    square = t.multiply(wide, wide, dtype=t.float32, fast_and_approximate_mode=False)
    if batched:
        squares = t.reshape(square, (1, rows, 64, 32))
        squares = t.permute(squares, (0, 2, 1, 3))
        squares = t.reshape(squares, (4, 16, rows, 32))
    else:
        squares = t.reshape(square, (4, 16, 1, 32))
    reduced = t.empty(
        (1, 1, rows, 32),
        dtype=t.float32,
        layout=t.TILE_LAYOUT,
        device=dec.mesh_device,
        memory_config=t.DRAM_MEMORY_CONFIG,
    )
    # Do not split/reassociate this call or change fp32_intermediate_stages.
    reduced = t.experimental.fast_reduce_nc(squares, dims=[0, 1], output=reduced, compute_kernel_config=dec.compute)
    groups = t.reshape(reduced, (rows, 1, 4, 8))
    total = dec.norm_terminal_scan(groups)
    inverse = t.reshape(epilogue(total), (1, rows, 1))
    normalized = t.multiply(wide, inverse, dtype=t.float32, fast_and_approximate_mode=False)
    output = t.to_memory_config(
        t.multiply(normalized, dec.w[KEY], dtype=t.bfloat16, fast_and_approximate_mode=False), memory
    )
    stages = (
        dict(
            wide=wide,
            square=square,
            squares=squares,
            groups=groups,
            total=total,
            inverse=inverse,
            normalized=normalized,
            output=output,
        )
        if debug
        else {}
    )
    return output, stages


def device_probe(args, cases, gamma, result):
    import torch

    assert not os.environ.get("TT_METAL_WATCHER")
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") == "0"
    tracked = args.mode == "acceptance"
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1" if tracked else "0")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == str(int(tracked))
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    from ttnn.tools import trace_allocation_tracker as tracker

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import NormTerminalScan
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_compensated_norm import (
        CompensatedNormEpilogue,
        normalized_singleton,
    )
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import _bounded_concat

    t.set_fabric_config(t.FabricConfig.FABRIC_1D)
    mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=args.trace_region_size)
    traces, captured = {}, {}
    try:
        mesh.enable_program_cache()

        def upload(x, dtype, device=False):
            kwargs = dict(dtype=dtype, layout=t.TILE_LAYOUT, mesh_mapper=t.ReplicateTensorToMesh(mesh))
            if device:
                kwargs.update(device=mesh, memory_config=t.DRAM_MEMORY_CONFIG)
            return t.from_torch(x.contiguous(), **kwargs)

        inputs = tuple(upload(cases[0][1][:, i : i + 1], t.bfloat16, True) for i in range(32))
        hosts = {name: tuple(upload(x[:, i : i + 1], t.bfloat16) for i in range(32)) for name, x in cases}
        dec = SimpleNamespace(
            kind="linear_attention",
            mesh_device=mesh,
            w={KEY: upload(gamma, t.float32, True)},
            compute=t.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=t.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
            norm_terminal_scan=NormTerminalScan(t, mesh),
        )
        epilogue = CompensatedNormEpilogue(t, mesh, 1e-6)

        def refresh(name):
            for host, destination in zip(hosts[name], inputs):
                t.copy_host_to_device_tensor(host, destination)

        def call(name, debug=False):
            if name == "reference":
                return tuple(normalized_singleton(dec, x, KEY, epilogue) for x in inputs), {}
            packed = _bounded_concat(inputs, dim=1, fan_in=16)
            value, stages = normalization(t, dec, packed, epilogue, batched=True, debug=debug)
            # Match the established row-major slice/tilize transport boundary.
            row_major = t.to_layout(value, t.ROW_MAJOR_LAYOUT)
            outputs = tuple(t.to_layout(row_major[:, i : i + 1, :], t.TILE_LAYOUT) for i in range(32))
            return outputs, stages

        def physical(value):
            return [t.from_device(local).to_torch_with_padded_shape().clone() for local in t.get_device_tensors(value)]

        def logical(value):
            return [t.to_torch(local).clone() for local in t.get_device_tensors(value)]

        def bits(x):
            return x.contiguous().view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)

        def exact(a, b, label):
            assert a.dtype == b.dtype and a.shape == b.shape, (label, a.shape, b.shape)
            unequal = bits(a) != bits(b)
            if unequal.any():
                raise AssertionError(
                    f"{label}: {int(unequal.sum())} unequal words; first={unequal.nonzero()[0].tolist()}"
                )

        def snapshot(output):
            return [physical(value) for value in output]

        def check_outputs(a, b, label):
            for row, (aa, bb) in enumerate(zip(a, b)):
                assert len(aa) == len(bb) == 4
                for rank, (x, y) in enumerate(zip(aa, bb)):
                    exact(x, y, f"{label}/row{row}/rank{rank}/physical")
                    assert torch.isfinite(x).all(), f"Nonfinite physical output {label}/{row}/{rank}"

        def stage_snapshot(stages):
            return {key: logical(value) for key, value in stages.items()}

        def check_stages(observed, expected, label):
            transport = []
            for key in axes:
                for rank in range(4):
                    target = expected[key][rank]
                    if key == "wide":
                        # Packing introduces the known tilize boundary before
                        # multiplication. Validate its complete bit mapping;
                        # never canonicalize square or later arithmetic stages.
                        target = finite_tilize_bits(target)
                        transport.append(
                            dict(rank=rank, changed_words=int((bits(target) != bits(expected[key][rank])).sum()))
                        )
                    exact(observed[key][rank], target, f"{label}/{key}/rank{rank}")
            result.setdefault("input_transport_checks", []).append(
                dict(label=label, exact_mapping=True, per_rank=transport)
            )

        axes = dict(wide=1, square=1, squares=2, groups=0, total=0, inverse=1, normalized=1, output=1)
        expected, expected_stages = {}, {}
        warmed_entries = None
        for name, _ in cases:
            refresh(name)
            originals, _ = call("reference")
            expected[name] = snapshot(originals)
            del originals
            diagnostics = []
            for row, x in enumerate(inputs):
                output, stages = normalization(t, dec, x, epilogue, batched=False, debug=True)
                for rank, value in enumerate(physical(output)):
                    exact(value, expected[name][row][rank], f"{name}/diagnostic_vs_production/{row}/{rank}")
                diagnostics.append(stage_snapshot(stages))
                del output, stages
            expected_stages[name] = {
                key: [torch.cat([d[key][rank] for d in diagnostics], dim=axis) for rank in range(4)]
                for key, axis in axes.items()
            }
            output, stages = call("candidate", debug=True)
            check_outputs(snapshot(output), expected[name], name + "/eager")
            observed = stage_snapshot(stages)
            check_stages(observed, expected_stages[name], name)
            del output, stages
            current_entries = mesh.num_program_cache_entries()
            if warmed_entries is not None:
                assert current_entries == warmed_entries, "Changed inputs unexpectedly compiled a new program"
            warmed_entries = current_entries
            result["checks"].append(
                dict(
                    case=name, eager_exact=True, downstream_intermediate_exact=True, input_transport_exact=True, ranks=4
                )
            )
        assert any(
            not torch.equal(bits(a), bits(b)) for a, b in zip(expected["recorded_a"][0], expected["recorded_b"][0])
        ), "Changed-row stale-input control is vacuous"
        # Prewarm measured (non-debug) lifetimes and every read/copy before capture.
        for _ in range(2):
            for variant in ("reference", "candidate"):
                output, _ = call(variant)
                snapshot(output)
                del output
        gc.collect()
        t.synchronize_device(mesh)
        entries = mesh.num_program_cache_entries()
        result["program_cache_entries_before_capture"] = entries
        mesh.set_program_cache_misses_allowed(False)
        for variant in ("reference", "candidate"):
            trace = t.begin_trace_capture(mesh, cq_id=0)
            traces[variant] = trace
            try:
                captured[variant] = call(variant, debug=tracked and variant == "candidate")
            finally:
                t.end_trace_capture(mesh, trace, cq_id=0)
            # Exact captured outputs only: fully regenerated before every read,
            # consumed to CPU before another trace; input/gamma allocations predate traces.
            output, stages = captured[variant]
            for value in (*output, *stages.values()):
                tracker.acknowledge_corruptible(value)
            del output, stages, value
        result["acknowledgment_lifetime"] = (
            "Only exact captured output/stage tensors acknowledged; aliases share backing buffers. "
            "Each is fully written before use and consumed to CPU before the next trace. "
            "All inputs/gamma predate both traces; refresh inputs before every variant replay. No broad scopes."
        )
        if tracked:
            poison = t.empty_like(inputs[0])
            try:
                t.execute_trace(mesh, traces["reference"], cq_id=0, blocking=False)
            except RuntimeError as exc:
                assert "allocat" in str(exc).lower(), str(exc)
                result["allocation_tracker_positive_control"] = str(exc)[:600]
            else:
                raise AssertionError("Allocation tracker failed to reject a deliberately live allocation")
            del poison
            gc.collect()
        # A -> B -> B -> edge -> A tests changed, unchanged and return-to-old input,
        # alternating trace identities without capturing again or eager execution.
        for name in ("recorded_a", "recorded_b", "recorded_b", "finite_edges", "recorded_a"):
            for variant in ("reference", "candidate"):
                refresh(name)
                t.execute_trace(mesh, traces[variant], cq_id=0, blocking=False)
                t.synchronize_device(mesh)
                output, stages = captured[variant]
                check_outputs(snapshot(output), expected[name], f"{name}/trace/{variant}")
                if stages:
                    observed = stage_snapshot(stages)
                    check_stages(observed, expected_stages[name], f"{name}/trace/{variant}")
                del output, stages
                result["checks"].append(dict(case=name, variant=variant, replay_exact=True))
        assert mesh.num_program_cache_entries() == entries, "Late program compilation"
        if args.mode == "benchmark":
            timings = {key: [] for key in traces}
            refresh("recorded_a")
            for sample in range(args.samples):
                order = ("reference", "candidate") if sample % 2 == 0 else ("candidate", "reference")
                for variant in order:
                    t.synchronize_device(mesh)
                    begin = time.perf_counter_ns()
                    for _ in range(args.repeats):
                        t.execute_trace(mesh, traces[variant], cq_id=0, blocking=False)
                    t.synchronize_device(mesh)
                    timings[variant].append((time.perf_counter_ns() - begin) / 1e6 / args.repeats)
            result["benchmark_ms_per_batch32"] = {
                key: dict(samples_ms=value, median_ms=statistics.median(value)) for key, value in timings.items()
            }
            result["benchmark_speedup"] = statistics.median(timings["reference"]) / statistics.median(
                timings["candidate"]
            )
            result["benchmark_scope"] = (
                "Serial synchronized elapsed trace replay, includes candidate pack/split; excludes refresh/readback. "
                "No profiler/tracker. Repeats are fixed-input microtest, not model latency or a floor claim."
            )
        result.update(hardware_verified=True, tracking=tracked, captures=len(traces), status="PASS")
    finally:
        for trace in traces.values():
            t.release_trace(mesh, trace)
        t.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "acceptance", "benchmark"), default="cpu")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--activations", type=Path, default=ROOT / "doc/multichip_decoder/natural_b32.pt")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--trace-region-size", type=int, default=128 * 1024 * 1024)
    args = parser.parse_args()
    assert 1 <= args.samples <= 20 and 1 <= args.repeats <= 100
    if args.output and args.output.exists():
        parser.error("Use a fresh output path to preserve evidence")
    if args.mode != "cpu" and args.output is None:
        parser.error("Device modes require --output for preserved evidence")
    import torch

    torch.set_num_threads(4)
    result = dict(
        status="RUNNING",
        hardware_verified=False,
        mode=args.mode,
        selected_context=262144,
        precision_policy_changed=False,
        checks=[],
        command=sys.argv,
        limitations=[
            "Only one requested linear layer's input norm gamma; not all30 layer weights.",
            "Recorded HF rows are distinct natural activations, selected across positions; not serving trace captures.",
            "Logical intermediate bits and all physical BF16 output bits on four replicas are checked; unused intermediate padding is unspecified.",
            "No attention, recurrence, sampling, full-model or serving/floor acceptance claim.",
        ],
    )
    try:
        result["cpu_mapping"] = cpu_mapping()
        result["transport_evidence"] = validate_transport_evidence()
        cases, gamma, result["fixtures"] = fixtures(args)
        assert "ttnn" not in sys.modules, "Host validation unexpectedly imported TTNN"
        if args.mode == "cpu":
            result["status"] = "PASS"
        else:
            device_probe(args, cases, gamma, result)
    except BaseException as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps({key: result.get(key) for key in ("status", "mode", "hardware_verified", "error")}), flush=True
        )


if __name__ == "__main__":
    main()
