# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Matched pure SDPA: real Wan QKV, no timed preparation or communication.

Grid, chunks and reader barrier are independently controlled. Numerical recipes
and input buffer depths are unchanged. Stock receives logical-length K/V so its
native padding mask excludes exactly the same eight keys as the private kernels.
"""

import json
import os
from pathlib import Path
import statistics
import time

import pytest
import torch
import ttnn

from attention import kernel
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.utils import cache
from test_perf_breakdown import FinishedProbe, measure, counters
from test_suite import PROMPTS, device_params


def cb_bytes(variant, qc, kc):
    fp32, fast, _, _ = kernel.recipe(variant)
    qt, kt = qc // 32, kc // 32
    state = 4096 if fp32 else 2048
    kv = 576 if variant == "G" else 1088 if variant in "EF" else 2048
    slots, stride = (1 if fp32 else 2), (2 if fast else 1)
    return (
        2 * qt * 4 * 2048
        + 2 * kt * 4 * slots * kv
        + 2 * 2048
        + state
        + qt * kt * state
        + 2 * qt * 4 * stride * state
        + 2 * qt * 2048
        + 2 * qt * stride * state
        + qt * state
        + (8 if fp32 else 16) * 2048
        + 2 * 2048
    )


def queued_measure(device, fn, count=10):
    """Ten queued replays between profiler drains; report amortized device time."""
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    output = fn()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    ttnn.ReadDeviceProfiler(device)
    samples = []
    try:
        for sample in range(4):
            start = time.perf_counter_ns()
            for _ in range(count):
                ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            host_ms = (time.perf_counter_ns() - start) / 1e6 / count
            ttnn.ReadDeviceProfiler(device)
            chips = counters()
            if sample:
                samples.append(
                    dict(host_ms=host_ms, device_ms=max(x["span_ms"] for x in chips.values()) / count, chips=chips)
                )
    finally:
        ttnn.release_trace(device, trace)
    return dict(
        replays_per_sample=count,
        samples=samples,
        device_ms=statistics.median(x["device_ms"] for x in samples),
        host_ms=statistics.median(x["host_ms"] for x in samples),
    )


def wall_measure(device, fn, warmup=100, count=20, samples=5):
    """Uninstrumented sustained replay, including amortized dispatch/sync overhead."""
    warmup = int(os.environ.get("WAN_SDPA_WARMUP", str(warmup)))
    eager = fn()
    ttnn.synchronize_device(device)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    output = fn()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    records = []
    try:
        for _ in range(warmup):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        for _ in range(samples):
            start = time.perf_counter_ns()
            for _ in range(count):
                ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            records.append((time.perf_counter_ns() - start) / 1e6 / count)
    finally:
        ttnn.release_trace(device, trace)
    return dict(
        metric="Uninstrumented synchronized wall time per queued SDPA replay",
        warmup=warmup,
        replays_per_sample=count,
        samples_ms=records,
        wall_ms=statistics.median(records),
    )


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.timeout(3600)
def test_sdpa_tuning(mesh_device, monkeypatch):
    torch.set_num_threads(16)
    target = Path(os.environ["WAN_SDPA_REPORT"])
    assert not target.exists() or os.environ.get("WAN_SDPA_RESUME") == "1"
    report = dict(
        status="running",
        scope="Real high-noise Wan block 20; SP4/TP2; all eight chips",
        timed="Pure SDPA, prepared/gathered inputs; no CCL or preprocessing",
        reference="Float64 SDPA on original BF16 inputs, chip 0, all 20 heads, 32 spaced query rows",
        rows=[],
        skips=[],
    )
    if target.exists():
        report = json.loads(target.read_text())
    report["status"] = "running"
    timing_source = os.environ.get("WAN_SDPA_TIMING_ONLY_SOURCE")
    if timing_source:
        assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") == "0"
        monkeypatch.setattr(ttnn, "ReadDeviceProfiler", lambda *args, **kwargs: None)
        warmups = int(os.environ.get("WAN_SDPA_WARMUP", "100"))
        report["timed"] = f"Uninstrumented sustained pure SDPA; {warmups} warmups, five batches of 20 replays"
    l1_limit = int(os.environ.get("WAN_SDPA_CB_LIMIT", "1441792"))
    # A follow-up can admit a near-capacity configuration, still subject to the
    # runtime's exact allocation check. Keep historical skips in the report.
    report["cb_limit"] = l1_limit

    def save():
        target.write_text(json.dumps(report, indent=2) + "\n")

    original_load = cache.load_model

    def cached_load(model=None, **kwargs):
        def miss():
            raise RuntimeError("Converted-weight cache miss")

        kwargs["get_torch_state_dict"] = miss
        return original_load(kwargs.pop("tt_model", model), **kwargs)

    monkeypatch.setattr(cache, "load_model", cached_load)
    pipeline = WanPipeline(
        device=mesh_device,
        config=WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=os.environ["WAN_CHECKPOINT"],
            height=480,
            width=832,
            num_frames=81,
        ),
        run_warmup=False,
    )
    block = pipeline.transformer_states[0].model.blocks[20]
    raw_forward = block.forward
    for state in pipeline.transformer_states:
        for current_block in state.model.blocks:
            original = current_block.forward

            def drained(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                ttnn.ReadDeviceProfiler(mesh_device)
                return result

            current_block.forward = drained

    def probe(*args, **kwargs):
        ring = ttnn.transformer.ring_joint_scaled_dot_product_attention
        saved = {}

        def capture(*a, **kw):
            saved["qkv"] = tuple(ttnn.clone(x) for x in a[:3])
            return ring(*a, **kw)

        with monkeypatch.context() as patch:
            patch.setattr(ttnn.transformer, "ring_joint_scaled_dot_product_attention", capture)
            raw_forward(*args, **kwargs)
        ttnn.ReadDeviceProfiler(mesh_device)
        q, local_k, local_v = saved["qkv"]
        hw = mesh_device.compute_with_storage_grid_size()
        full = hw.x * hw.y
        old = hw.x * (hw.y - 1)
        k, v = (
            block.attn1.ccl_manager.all_gather(x, dim=2, mesh_axis=1, use_hyperparams=True) for x in (local_k, local_v)
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        report.update(
            q_shape=list(q.shape),
            kv_shape=list(k.shape),
            logical_k=32760,
            hardware_grid=[hw.x, hw.y],
            old_core_budget=old,
            full_core_budget=full,
        )
        host = [ttnn.to_torch(ttnn.get_device_tensors(x)[0]).clone() for x in (q, k, v)]
        indices = torch.linspace(0, q.shape[2] - 1, 32).long()
        ref = torch.nn.functional.scaled_dot_product_attention(
            host[0][:, :, indices, :].double(), host[1][:, :, :32760, :].double(), host[2][:, :, :32760, :].double()
        )
        input_path = target.with_suffix(".inputs.pt")
        if input_path.exists():
            previous = torch.load(input_path, weights_only=True)
            assert all(torch.equal(previous[name], value) for name, value in zip(("q", "k", "v"), host))
            assert torch.equal(previous["reference"], ref)
            del previous
        else:
            torch.save(dict(q=host[0], k=host[1], v=host[2], indices=indices, reference=ref), input_path)
        del host
        # Slice is outside every timed region. Logical 32760, physical 32768:
        # stock factory selects its lightweight native partial-K mask, not dense/windowed masking.
        sk, sv = (ttnn.slice(x, (0, 0, 0, 0), (1, 20, 32760, 128)) for x in (k, v))
        prepared = {x: (q, k, v) for x in "ABCD"}
        for variant in "EFG":
            pq = kernel.prepare(mesh_device, q, variant, is_q=True, cores=full)
            pk, pv = (kernel.prepare(mesh_device, x, variant, is_q=False, cores=full) for x in (local_k, local_v))
            gk, gv = (block.attn1.ccl_manager.all_gather(x, dim=2, mesh_axis=1, use_hyperparams=True) for x in (pk, pv))
            prepared[variant] = (pq, gk, gv)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        baseline_outputs = {}

        def run(variant, qc, kc, budget, barrier=2, phase="sweep", repeat=False):
            key = (variant, qc, kc, budget, barrier)
            if not repeat and any(tuple(row["key"]) == key for row in report["rows"]):
                return
            if any(
                tuple(row["key"]) == key
                and (
                    "Conservative" not in row["reason"]
                    or cb_bytes("A" if variant == "stock" else variant, qc, kc) > l1_limit
                )
                for row in report["skips"]
            ):
                return
            row = dict(
                key=key,
                variant=variant,
                q_chunk=qc,
                k_chunk=kc,
                core_budget=budget,
                active_cores=(min(budget // 20, 8192 // qc) * 20 if variant != "stock" else None),
                reader_barrier=barrier if variant != "stock" else None,
                phase=phase,
            )
            if variant == "stock":
                if cb_bytes("A", qc, kc) > l1_limit:
                    report["skips"].append(dict(**row, reason="Conservative stock L1 capacity guard"))
                    save()
                    return
                grid = hw if budget == full else ttnn.CoreCoord(hw.x, hw.y - 1)
                program = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid, q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
                )

                def fn():
                    return ttnn.transformer.scaled_dot_product_attention(
                        q,
                        sk,
                        sv,
                        is_causal=False,
                        program_config=program,
                        compute_kernel_config=block.attn1.sdpa_compute_kernel_config,
                    )

            else:
                row["cb_bytes"] = cb_bytes(variant, qc, kc)
                # Conservative guard: leave 128 KiB for firmware, kernels and the small-L1 region.
                if row["cb_bytes"] > l1_limit:
                    report["skips"].append(dict(**row, reason="Conservative L1 capacity guard"))
                    save()
                    return

                def fn():
                    return kernel.attention(
                        mesh_device,
                        *prepared[variant],
                        variant,
                        logical_k=32760,
                        max_cores=budget,
                        q_chunk_size=qc,
                        k_chunk_size=kc,
                        reader_barrier_tiles=barrier,
                    )

            print("SDPA_START", row, flush=True)
            try:
                output = fn()
            except RuntimeError as exc:
                if "Statically allocated circular buffers" not in str(exc):
                    raise
                report["skips"].append(dict(**row, reason=str(exc).split("backtrace:")[0]))
                save()
                return
            ttnn.synchronize_device(mesh_device)
            actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0])[:, :, indices, :].double()
            assert torch.isfinite(actual).all(), row
            row["l2_pct"] = ((actual - ref).norm() / ref.norm() * 100).item()
            row["pcc"] = torch.corrcoef(torch.stack((actual.flatten(), ref.flatten())))[0, 1].item()
            if variant != "stock" and barrier != 2:
                control = kernel.attention(
                    mesh_device,
                    *prepared[variant],
                    variant,
                    logical_k=32760,
                    max_cores=budget,
                    q_chunk_size=qc,
                    k_chunk_size=kc,
                    reader_barrier_tiles=2,
                )
                original = ttnn.to_torch(ttnn.get_device_tensors(control)[0])[:, :, indices, :].double()
                row["reader_sample_exact"] = torch.equal(original, actual)
                assert row["reader_sample_exact"], "Reader barrier changed sampled output"
            if qc == 256 and kc == 512 and barrier == 2:
                if variant in baseline_outputs:
                    row["grid_sample_exact"] = torch.equal(actual, baseline_outputs[variant])
                    assert row["grid_sample_exact"], "Grid-only change affected sampled output"
                else:
                    baseline_outputs[variant] = actual.clone()
            ttnn.ReadDeviceProfiler(mesh_device)
            row["perf"] = (
                wall_measure(mesh_device, fn) if timing_source else measure(mesh_device, fn, warmup=12, samples=5)
            )
            if phase == "best_recheck":
                row["queued_perf"] = queued_measure(mesh_device, fn)
            latency = row["perf"].get("device_ms", row["perf"].get("wall_ms"))
            row["tflops_per_chip"] = (4 * 20 * 8192 * 32760 * 128) / (latency * 1e9)
            report["rows"].append(row)
            save()
            print(
                "SDPA_RESULT",
                {k: v for k, v in row.items() if k not in ("perf", "queued_perf")},
                "wall_ms" if timing_source else "device_ms",
                latency,
                flush=True,
            )

        choices = tuple(os.environ.get("WAN_SDPA_VARIANTS", "stock A D C B E F G").split())
        if timing_source:
            source = json.loads(Path(timing_source).read_text())
            assert source["status"] == "completed"
            for variant in choices:
                best = min((r for r in source["rows"] if r["variant"] == variant), key=lambda r: r["perf"]["device_ms"])
                if not os.environ.get("WAN_SDPA_BEST_ONLY"):
                    run(variant, 256, 512, old, phase="wall_baseline")
                run(variant, best["q_chunk"], best["k_chunk"], best["core_budget"], best["key"][4], phase="wall_best")
                if variant in "DG" and os.environ.get("WAN_SDPA_TUNED_GRID_CHECK"):
                    run(variant, best["q_chunk"], best["k_chunk"], old, best["key"][4], phase="wall_tuned_100_cores")
            run("stock", 256, 256, full, phase="wall_model_stock_chunks")
            raise FinishedProbe
        for variant in choices:
            run(variant, 256, 512, old, phase="grid_control")
            run(variant, 256, 512, full, phase="grid_control")
        chunks = [
            (128, 256),
            (128, 512),
            (128, 1024),
            (256, 128),
            (256, 256),
            (256, 512),
            (256, 1024),
            (512, 128),
            (512, 256),
        ]
        for qc, kc in chunks:
            for variant in choices:
                run(variant, qc, kc, full)
        for variant in (x for x in choices if x != "stock"):
            best = min((r for r in report["rows"] if r["variant"] == variant), key=lambda r: r["perf"]["device_ms"])
            for barrier in (8, 0):
                run(variant, best["q_chunk"], best["k_chunk"], full, barrier, phase="reader_control")
        # Re-check each best point and its original point in reversed variant order.
        for variant in reversed(choices):
            best = min((r for r in report["rows"] if r["variant"] == variant), key=lambda r: r["perf"]["device_ms"])
            run(
                variant,
                best["q_chunk"],
                best["k_chunk"],
                best["core_budget"],
                best["key"][4],
                phase="best_recheck",
                repeat=True,
            )
            run(variant, 256, 512, old, phase="baseline_recheck", repeat=True)
        raise FinishedProbe

    block.forward = probe
    with pytest.raises(FinishedProbe), torch.no_grad():
        pipeline(prompts=[PROMPTS[0]], num_inference_steps=2, seed=42, output_type="uint8", traced=False)
    report["status"] = "completed"
    save()
