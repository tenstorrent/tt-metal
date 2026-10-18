# SPDX-License-Identifier: Apache-2.0
"""Exclusive TP4 diagnostic: batch only the existing QSA score mask and TopK.

No production monkeypatches, profiler or altered gather/indexer. Exact ordered
indices are required, including ties and different runtime prefix bounds.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import statistics
import subprocess
import time
import traceback
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_prefill_view_probe import host_copy, read_shards
from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as decoder
from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder, paged_qsa, precision
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedQSA
from models.autoports.qwen_qwen3_8_flash_next.tt.state_ops import zero_inplace

CONTEXT = 262144
BLOCKS = CONTEXT // 4


def make_row_mask():
    source = optimized_decoder._QSA_MASK_SOURCE
    start = source.index("    if(worker==0) {")
    stop = source.index("    constexpr uint32_t chunks=", start)
    source = (
        source[:start]
        + """    if(worker<B) {
        noc.async_read(positions,CoreLocalMem<uint32_t>(base),128,{.page_id=worker},{});
        noc.async_read_barrier();invalidate_l1_cache();
        const uint32_t end=(pos[0]+S)/4;
        const uint32_t last=end>512?end:512;
        ASSERT(last<=K);
        auto result=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base+640);
        for(uint32_t i=0;i<8;++i) { result[i]=0; }
        result[0]=last;
        noc.async_write(CoreLocalMem<uint32_t>(base+640),bound,32,{}, {.page_id=worker});
        noc.async_write_barrier();
    }
"""
        + source[stop:]
    )
    method = inspect.getsource(OptimizedQSA._mask_scores)
    import textwrap

    method = textwrap.dedent(method)
    old = "ttnn.empty((1,), dtype=ttnn.uint32"
    assert method.count(old) == 1
    method = method.replace(old, "ttnn.empty((self.batch_size, 1), dtype=ttnn.uint32")
    namespace = dict(optimized_decoder.__dict__, _QSA_MASK_SOURCE=source)
    exec(compile(method, "<isolated_qsa_row_mask>", "exec"), namespace)
    return namespace["_mask_scores"], source


ROW_MASK, ROW_MASK_SOURCE = make_row_mask()


def geometry(value):
    return dict(
        shape=list(value.shape),
        padded_shape=list(value.padded_shape),
        dtype=str(value.dtype),
        layout=str(value.layout),
        memory_config=str(value.memory_config()),
    )


def make_candidate(baseline_path=None):
    """Exercise production against checksum-pinned pre-integration selection."""
    root = Path(__file__).resolve().parents[4]
    relative = "models/autoports/qwen_qwen3_8_flash_next/tt/paged_qsa.py"
    raw = (
        baseline_path.read_bytes()
        if baseline_path
        else subprocess.check_output(
            ["git", "-C", str(root), "show", "baf47af3b4166cfe6bde0f14e0afddce6db2210c:" + relative]
        )
    )
    assert hashlib.sha256(raw).hexdigest() == "485c618481c801e1c33bfa484f4b5ffa5c1c181ef468a4e867f61938ca56c7e0"
    source = raw.decode()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "select_groups")
    baseline = ast.get_source_segment(source, node) + "\n"
    namespace = dict(paged_qsa.__dict__)
    exec(compile(baseline, "<pinned_original_qsa_selection>", "exec"), namespace)
    return paged_qsa.select_groups, namespace["select_groups"], baseline, inspect.getsource(paged_qsa.select_groups)


def baseline_tail(context, rows, positions):
    one = SimpleNamespace(device=context.device, batch_size=1, blocks=BLOCKS)
    selected = []
    for lane, scores in enumerate(rows):
        bound = OptimizedQSA._mask_scores(one, scores, positions[lane : lane + 1], 1)
        selected.append(ttnn.experimental.topk_large_indices(scores, k=512, valid_length_tensor=bound))
    return ttnn.concat(selected, dim=0)


def candidate_tail(context, rows, positions):
    scores = ttnn.concat(rows, dim=0)
    bound = paged_qsa.mask_scores_per_row(context, scores, positions, 1)
    return ttnn.experimental.topk_large_indices(scores, k=512, valid_length_tensor=bound)


def positions_for(profile, batch):
    if profile == "mixed":
        values = [0, 3, 127, 2047, 4095, 8191, 131071, 262143]
        # Each tested batch includes different bounds and the full-context end.
        return [values[(i * 7) % len(values)] for i in range(batch)]
    return [dict(short=127, k4=4095, k8=8191, full=262143)[profile]] * batch


def compare(reference, actual):
    result = {}
    for name, shards in actual.items():
        assert len(shards) == len(reference[name]) == 4
        result[name] = []
        for ref, got in zip(reference[name], shards):
            # PyTorch CPU equal/isfinite do not support Float8_e4m3fn. Compare
            # stored bytes directly and widen only the diagnostic finite check.
            exact = (
                ref.dtype == got.dtype
                and ref.shape == got.shape
                and torch.equal(ref.contiguous().view(torch.uint8), got.contiguous().view(torch.uint8))
            )
            finite = bool(torch.isfinite(got.float()).all())
            row = dict(exact=exact, finite=finite, shape=list(got.shape), dtype=str(got.dtype))
            row["pcc"] = 1.0 if exact and finite else None
            row["pcc_basis"] = "exact finite byte equality" if row["pcc"] == 1.0 else "not claimed"
            row["sha256"] = hashlib.sha256(got.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            if name == "indices":
                row["sentinels"] = int(((got.to(torch.int64) & 0xFFFFFFFF) == 0xFFFFFFFF).sum())
            result[name].append(row)
    return result


def measure(mesh, call, reset, snapshot, args, label, reference, save):
    trace = output = None
    result = dict(seconds=[], comparisons=[])
    try:
        reset()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = call()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.synchronize_device(mesh)
        for repeat in range(args.warmup + args.repeats):
            reset()
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - start
            if repeat < args.warmup:
                continue
            actual = snapshot(output)
            # Preserve every measured snapshot before diagnostic code can fail.
            torch.save(actual, args.output / f"{label}_{repeat - args.warmup}.pt")
            if reference is None:
                reference = actual
            metrics = compare(reference, actual)
            exact = all(x["exact"] and x["finite"] for rows in metrics.values() for x in rows)
            result["seconds"].append(elapsed)
            result["comparisons"].append(metrics)
            save()
            if not exact:
                (args.output / f"{label}_FAILURE.json").write_text(json.dumps(metrics, indent=2))
                raise AssertionError(f"Ordered index/output/cache mismatch: {label}")
        result["median_seconds"] = statistics.median(result["seconds"])
        return result, reference
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        del output


def run_pair(mesh, calls, reset, snapshot, args, row, save, bind=lambda _: None):
    # Both eager variants warm before either trace; persistent inputs preexist.
    for variant, call in calls.items():
        bind(variant)
        reset()
        output = call()
        ttnn.synchronize_device(mesh)
        # Diagnostic cache slices have program-cache-owned metadata buffers.
        # Warm those reads before capture, so their persistent owners cannot
        # first allocate inside a trace's reusable scratch address range.
        snapshot(output)
        ttnn.synchronize_device(mesh)
        del output
    reference = None
    for variant, call in calls.items():
        bind(variant)
        result, reference = measure(mesh, call, reset, snapshot, args, row["label"] + "_" + variant, reference, save)
        row[variant] = result
        save()
    row["median_speedup"] = row["baseline"]["median_seconds"] / row["candidate"]["median_seconds"]
    print(json.dumps({k: row[k] for k in ("label", "median_speedup")}), flush=True)
    save()


def native_metadata_checks(mesh, ops, args, report, save):
    """B4->32->8 cache reuse, new metadata addresses and changed traced values."""
    report["native_metadata"] = []
    for batch in (4, 32, 8):
        values = ((torch.arange(BLOCKS).float() * 17) % 13 - 6).bfloat16()
        host_scores = values.reshape(1, 1, 1, BLOCKS).repeat(batch, 1, 1, 1)
        scores = ops.upload(host_scores, layout=ttnn.ROW_MAJOR_LAYOUT)
        rows = [ops.upload(host_scores[i : i + 1], layout=ttnn.ROW_MAJOR_LAYOUT) for i in range(batch)]
        scalars = [
            ops.upload(torch.tensor([512], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) for _ in range(batch)
        ]
        vectors = [
            ops.upload(torch.full((batch, 1), 512, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            for _ in range(2)
        ]
        assert vectors[0].buffer_address() != vectors[1].buffer_address()
        bounds = [
            [512, 513, 1024, 65536, 32768, 2048, 4096, 16384] * (batch // 8 + 1),
            [65536, 512, 511, 513, 1023, 1024, 2047, 2048] * (batch // 8 + 1),
        ]
        bounds = [x[:batch] for x in bounds]

        def reset(which):
            for scalar, value in zip(scalars, bounds[which]):
                host_copy(torch.tensor([value], dtype=torch.int32), scalar, mesh)
            for vector in vectors:
                host_copy(torch.tensor(bounds[which], dtype=torch.int32).reshape(batch, 1), vector, mesh)

        def baseline():
            return ttnn.concat(
                [
                    ttnn.experimental.topk_large_indices(row, k=512, valid_length_tensor=bound)
                    for row, bound in zip(rows, scalars)
                ],
                dim=0,
            )

        calls = {"scalar_baseline": baseline}
        for address, vector in enumerate(vectors):
            calls[f"vector_address_{address}"] = lambda vector=vector: ttnn.experimental.topk_large_indices(
                scores, k=512, valid_length_tensor=vector
            )
        # All persistent allocations and every eager kernel warm precede traces.
        reset(0)
        for call in calls.values():
            output = call()
            ttnn.synchronize_device(mesh)
            del output
        row = dict(
            batch=batch,
            metadata_addresses=[x.buffer_address() for x in vectors],
            metadata_geometry=geometry(vectors[0]),
            bounds=bounds,
            variants={},
        )
        report["native_metadata"].append(row)
        references = {}
        for name, call in calls.items():
            trace = output = None
            records = []
            try:
                reset(0)
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                output = call()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                ttnn.synchronize_device(mesh)
                for replay, which in enumerate((0, 1, 0)):
                    reset(which)
                    ttnn.synchronize_device(mesh)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    actual = {"indices": read_shards(output)}
                    if name == "scalar_baseline" and which not in references:
                        references[which] = actual
                    metrics = compare(references[which], actual)
                    records.append(dict(bound_case=which, metrics=metrics))
                    torch.save(actual, args.output / f"native_b{batch}_{name}_replay{replay}.pt")
                    assert all(x["exact"] for group in metrics.values() for x in group), (batch, name, which)
                row["variants"][name] = records
                save()
            finally:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                del output

        invalid_shape = ops.upload(torch.full((batch,), 512, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        wrong_count = ops.upload(torch.full((batch + 1, 1), 512, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        core = ttnn.CoreCoord(0, 0)
        one_core = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        row["guards"] = []
        for name, metadata, kwargs, expected in (
            ("flat_vector", invalid_shape, {}, "shape [rows, 1]"),
            ("wrong_rows", wrong_count, {}, "shape [rows, 1]"),
            ("multiple_rows_per_core", vectors[0], {"sub_core_grids": one_core}, "at most one input row"),
        ):
            try:
                output = ttnn.experimental.topk_large_indices(scores, k=512, valid_length_tensor=metadata, **kwargs)
            except RuntimeError as error:
                assert expected in str(error), str(error)
                row["guards"].append(dict(name=name, rejected=True, error=str(error)))
            else:
                del output
                raise AssertionError(f"Missing per-row metadata guard: {name}")
        save()


def component(mesh, ops, batch, args, report, save):
    context = SimpleNamespace(device=mesh, batch_size=batch, blocks=BLOCKS)
    positions = ops.upload(torch.zeros(batch, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    rows = [ops.upload(torch.zeros(1, 1, 1, BLOCKS).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT) for _ in range(batch)]
    for profile in ("mixed", "short", "k4", "k8", "full"):
        host_positions = torch.zeros(batch, 32, dtype=torch.int32)
        host_positions[:, 0] = torch.tensor(positions_for(profile, batch))
        host_copy(host_positions, positions, mesh)
        for pattern in ("all_ties", "repeated_ties", "distinct_bf16"):
            for poison in (False, True):
                inputs = []
                for lane, pos in enumerate(host_positions[:, 0].tolist()):
                    values = torch.arange(BLOCKS).float()
                    if pattern == "all_ties":
                        values.zero_()
                    elif pattern == "repeated_ties":
                        values = (values * 17 + lane * 7).remainder(13) - 6
                    else:
                        values = torch.sin(values * 0.013 + lane * 0.1) * 8
                    values = values.bfloat16()
                    values[(pos + 1) // 4 :] = float("nan") if poison else 123.0
                    inputs.append(values.reshape(1, 1, 1, BLOCKS))

                def reset():
                    for source, destination in zip(inputs, rows):
                        host_copy(source, destination, mesh)

                def snapshot(output):
                    shards = read_shards(output)
                    for shard in shards:
                        ids = shard.reshape(batch, 512).to(torch.int64) & 0xFFFFFFFF
                        for lane, pos in enumerate(host_positions[:, 0].tolist()):
                            valid = ids[lane][ids[lane] != 0xFFFFFFFF]
                            complete = (pos + 1) // 4
                            assert valid.numel() == min(512, complete)
                            assert valid.unique().numel() == valid.numel()
                            assert bool((valid < complete).all())
                    return {"indices": shards}

                row = dict(
                    label=f"tail_b{batch}_{profile}_{pattern}_{'nan' if poison else 'finite'}",
                    positions=host_positions[:, 0].tolist(),
                    score_row=geometry(rows[0]),
                    future_padding="NaN" if poison else "finite 123",
                )
                report["component"].append(row)
                run_pair(
                    mesh,
                    {
                        "baseline": lambda: baseline_tail(context, rows, positions),
                        "candidate": lambda: candidate_tail(context, rows, positions),
                    },
                    reset,
                    snapshot,
                    args,
                    row,
                    save,
                )


def complete_qsa(mesh, ops, batch, cfg, policy, candidate, baseline, args, report, save):
    state = CheckpointState(args.checkpoint, f"model.language_model.layers.{args.layer}.self_attn.")
    model = decoder.ParallelQSA.build(
        state, cfg, ops, args.layer, batch_size=batch, max_requests=batch, attention_pool_tokens=CONTEXT + batch * 32
    )
    weights, fidelity = policy["weight_groups"], policy["compute_fidelities"]
    precision.cast_attr(model, "packed_projection", weights[policy["qsa_execution"]["batched_decode_weight_group"]])
    model.prefill_projection_compute = precision.compute(mesh, fidelity["qsa_prefill"])
    model.projection_compute = precision.compute(mesh, fidelity["qsa_decode"])
    model.packed_singleton_compute = precision.compute(
        mesh, fidelity[policy["qsa_execution"]["singleton_packed_fidelity_group"]]
    )
    precision.cast_attr(model, "wo", weights["qsa_output"])
    model.output_projection.row_weight = model.wo
    model.output_projection.compute = precision.compute(mesh, fidelity["qsa_output"])
    model.route_sequence = 1
    # Match B32 owner's compact views, which inherit this prepared prefix cap.
    model.indexer_prefix_cap = True
    if model.indexer_prefix_bound is None:
        model.indexer_prefix_bound = ops.upload(torch.zeros(1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    rng = torch.Generator().manual_seed(2026101707 + batch)
    hidden = ops.upload((torch.randn(1, batch, 1, 2560, generator=rng) * 0.2).bfloat16())
    positions = ops.upload(torch.zeros(batch, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    table = ops.upload(torch.zeros(batch, CONTEXT // 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def bind(variant):
        fn = baseline if variant == "baseline" else candidate

        def select(self, *values):
            output = fn(self, *values)
            self._probe_selected = output
            return output

        model._select_groups = types.MethodType(select, model)

    def call():
        output = model.decode_forward(hidden, current_pos=positions, page_table=table)
        selected = model._probe_selected
        del model._probe_selected
        return output, selected

    def reset():
        for name in ("packed_cache", "raw_key_cache", "pooled_cache", "pooled_keys"):
            zero_inplace(mesh, getattr(model, name))

    def snapshot(output):
        result = {"output": read_shards(output[0]), "indices": read_shards(output[1])}
        # All written pages and one untouched past-page canary; this is not a
        # claim to compare every unused byte of the full physical cache.
        for name, value in (
            ("raw_tails", model.raw_key_cache[CONTEXT // 32 : CONTEXT // 32 + batch]),
            ("packed_tails", model.packed_cache[:, :, CONTEXT : CONTEXT + batch * 32]),
            ("pooled_tails", model.pooled_cache[:, :, BLOCKS : BLOCKS + batch * 8]),
            ("raw_canary", model.raw_key_cache[:1]),
            ("packed_canary", model.packed_cache[:, :, :32]),
            ("pooled_canary", model.pooled_cache[:, :, :8]),
        ):
            result[name] = read_shards(value)
        return result

    for profile in ("k4", "k8", "full", "mixed"):
        values = positions_for(profile, batch)
        host_positions = torch.zeros(batch, 32, dtype=torch.int32)
        host_positions[:, 0] = torch.tensor(values)
        host_table = torch.stack(
            [(torch.arange(CONTEXT // 32) + 37 * lane) % (CONTEXT // 32) for lane in range(batch)]
        ).int()
        for lane, pos in enumerate(values):
            host_table[lane, pos // 32] = CONTEXT // 32 + lane
        host_copy(host_positions, positions, mesh)
        host_copy(host_table, table, mesh)
        row = dict(
            label=f"full_qsa_b{batch}_{profile}",
            positions=values,
            context_capacity=CONTEXT,
            page_table_sha256=hashlib.sha256(host_table.numpy().tobytes()).hexdigest(),
            score_rows_retained_bytes_per_asic=batch * BLOCKS * 2,
            geometry={
                name: geometry(value)
                for name, value in dict(
                    hidden=hidden,
                    positions=positions,
                    page_table=table,
                    pooled_keys=model.pooled_keys,
                    packed_cache=model.packed_cache,
                    packed_projection=model.packed_projection,
                ).items()
            },
        )
        report["complete_qsa"].append(row)
        run_pair(mesh, {"baseline": call, "candidate": call}, reset, snapshot, args, row, save, bind)
        assert model._qsa_batched_selection
        row["production_selection_geometry"] = dict(model._qsa_batched_selection_geometry)
        save()
    del model._select_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    assert args.repeats > 0 and args.warmup > 0 and all(b in (4, 8, 16, 32) for b in args.batches)
    tracking = dict(
        native=bool(ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()),
        python=bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
        skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
        exemptions=[],
    )
    assert tracking["native"] and tracking["python"] and tracking["skip_program_cache"] == "0"
    assert all(
        os.environ.get(k, "0") in ("", "0")
        for k in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_WATCHER_DUMP_ALL")
    )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    candidate, baseline, baseline_source, candidate_source = make_candidate(args.baseline_source)
    report = dict(
        status="started",
        tracking=tracking,
        component=[],
        complete_qsa=[],
        sources={},
        scope="Standalone ordered selection and real-weight complete QSA tensor contract, not text quality or full-model speed",
        cache_fixture="Zero previous logical KV/raw/pooled state; real current hidden input/weights; distinct writable physical tail pages and permuted shared read-only past pages",
        timing_boundary="Warmed execute_trace through actual standalone completion synchronization; reset and host readbacks excluded",
        op_sequence=[
            dict(
                baseline="B gather/prefix/indexer/last-row BF16 conversions", candidate="Identical producer operations"
            ),
            dict(
                baseline="B one-row mask + one-row TopK; concatenate B selections",
                candidate="Concatenate B final score rows; one batched mask + same native batched TopK",
            ),
        ],
    )
    native = Path(__file__).resolve().parents[4] / "ttnn/cpp/ttnn/operations/experimental/topk_large_indices"
    for path in (
        Path(__file__),
        Path(paged_qsa.__file__),
        Path(decoder.__file__),
        Path(precision.__file__),
        native / "device/topk_large_indices_device_operation.cpp",
        native / "device/topk_large_indices_program_factory.cpp",
        native / "device/kernels/reader.cpp",
        native / "device/kernels/compute.cpp",
        native / "device/kernels/writer.cpp",
        Path(paged_qsa.__file__).with_name("kernels") / "qsa_mask_per_row.cpp",
    ):
        report["sources"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())
    (args.output / "baseline_selection.py.txt").write_text(baseline_source)
    (args.output / "candidate_selection.py.txt").write_text(candidate_source)
    (args.output / "candidate_row_mask.cpp.txt").write_text(ROW_MASK_SOURCE)

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    mesh = None
    save()
    try:
        cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
        # Installed Qwen4ExpTextConfig canonicalizes the checkpoint's raw
        # "full_attention" declaration to the native indexed-attention type.
        assert cfg.layer_types[args.layer] == "indexed_attention", cfg.layer_types[args.layer]
        report["layer_type"] = cfg.layer_types[args.layer]
        report["config_class"] = type(cfg).__name__
        policy = precision.for_layer(precision.load_policy(), args.layer)
        report["precision"] = policy
        ttnn.CONFIG.throw_exception_on_fallback = True
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=268435456)
        grid = mesh.compute_with_storage_grid_size()
        report.update(device_ids=list(mesh.get_device_ids()), grid=[grid.x, grid.y])
        native_metadata_checks(mesh, decoder.MeshOps(mesh), args, report, save)
        for batch in args.batches:
            ops = decoder.MeshOps(mesh)
            ops.precision_ccl = policy["ccl_dtype"]
            component(mesh, ops, batch, args, report, save)
            complete_qsa(mesh, ops, batch, cfg, policy, candidate, baseline, args, report, save)
        report["status"] = "passed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["closed_mesh"] = mesh is not None
        save()


if __name__ == "__main__":
    main()
