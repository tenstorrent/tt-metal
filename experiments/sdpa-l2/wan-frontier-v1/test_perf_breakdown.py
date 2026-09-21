# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-counter decomposition on shared real Wan block-20 inputs."""

import functools
import json
import os
from pathlib import Path
import statistics
import time

import pytest
import torch
import ttnn

from attention import WanAttentionAdapter, kernel
from diagnostics import map_tensors
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.utils import cache
from test_suite import PROMPTS, device_params


def counters():
    result = {}
    for chip, programs in ttnn.get_latest_programs_perf_data().items():
        entries = []
        for program in programs:
            value = program.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
            if value is not None and value.duration and value.end_timestamp > value.start_timestamp:
                uid = program.program_execution_uid
                entries.append(
                    dict(
                        start=value.start_timestamp,
                        end=value.end_timestamp,
                        ns=value.duration,
                        runtime=uid.runtime_id,
                        trace=uid.trace_id,
                        counter=uid.trace_id_counter,
                    )
                )
        assert entries, (chip, "No device counter samples")
        # Timestamps are device cycles, whereas duration is converted to ns.
        longest = max(entries, key=lambda e: e["ns"])
        ns_per_tick = longest["ns"] / (longest["end"] - longest["start"])
        intervals = sorted((e["start"], e["end"]) for e in entries)
        start, end = intervals[0]
        union = 0
        for left, right in intervals[1:]:
            if left > end:
                union += end - start
                start, end = left, right
            else:
                end = max(end, right)
        union += end - start
        span = max(e["end"] for e in entries) - min(e["start"] for e in entries)
        result[str(chip)] = dict(
            span_ms=span * ns_per_tick / 1e6,
            busy_union_ms=union * ns_per_tick / 1e6,
            sum_program_ms=sum(e["ns"] for e in entries) / 1e6,
            ns_per_tick=ns_per_tick,
            programs=entries,
        )
    assert len(result) == 8, result.keys()
    return result


def measure(device, fn, *, warmup=5, samples=3, verify=False):
    eager = fn()
    ttnn.synchronize_device(device)
    reference = ttnn.to_torch(ttnn.get_device_tensors(eager)[0]).clone() if verify else None
    ttnn.ReadDeviceProfiler(device)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    output = fn()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    ttnn.ReadDeviceProfiler(device)
    records = []
    try:
        for _ in range(warmup):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(device)
        for _ in range(samples):
            start = time.perf_counter_ns()
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            host_ms = (time.perf_counter_ns() - start) / 1e6
            ttnn.ReadDeviceProfiler(device)
            chips = counters()
            records.append(dict(host_ms=host_ms, chips=chips, device_ms=max(c["span_ms"] for c in chips.values())))
        exact = None
        if verify:
            actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0])
            exact = torch.equal(actual, reference)
            assert exact, "Full block trace replay differs"
    finally:
        ttnn.release_trace(device, trace)
    return dict(
        device_ms=statistics.median(r["device_ms"] for r in records),
        host_ms=statistics.median(r["host_ms"] for r in records),
        replay_exact=exact,
        samples=records,
    )


class FinishedProbe(Exception):
    pass


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.timeout(1800)
def test_block_breakdown(mesh_device, monkeypatch):
    torch.set_num_threads(16)
    target = Path(os.environ["WAN_PERF_REPORT"])
    assert not target.exists()
    report = dict(
        status="running",
        scope="High-noise block 20, same stock-pilot inputs for every choice",
        metric="Max per-chip device first-to-last kernel span; not sum of overlapping kernels",
        timestamps="Device cycles converted using longest program's ns/cycle ratio",
        rows=[],
    )
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
    # Drain profiler data between bring-up blocks to keep device buffers bounded.
    for state in pipeline.transformer_states:
        for current_block in state.model.blocks:
            original = current_block.forward

            def drained(*args, _original=original, **kwargs):
                result = _original(*args, **kwargs)
                ttnn.ReadDeviceProfiler(mesh_device)
                return result

            current_block.forward = drained

    def probe(*args, **kwargs):
        args = map_tensors(args, ttnn.clone)
        kwargs = map_tensors(kwargs, ttnn.clone)
        ring = ttnn.transformer.ring_joint_scaled_dot_product_attention
        saved = {}

        def capture(*a, **kw):
            saved["qkv"] = tuple(ttnn.clone(x) for x in a[:3])
            saved["ring_args"] = saved["qkv"] + a[3:]
            saved["ring_kwargs"] = kw
            return ring(*a, **kw)

        with monkeypatch.context() as patch:
            patch.setattr(ttnn.transformer, "ring_joint_scaled_dot_product_attention", capture)
            raw_forward(*args, **kwargs)
        ttnn.ReadDeviceProfiler(mesh_device)
        qkv = saved["qkv"]
        report["qkv_shape"] = list(qkv[0].shape)
        report["logical_n"] = 32760
        hw = mesh_device.compute_with_storage_grid_size()
        cores = hw.x * (hw.y - 1)
        order = ("stock", "D", "C", "B", "E", "F", "G")
        # Sustained warmup is counter-instrumented and dumped each replay.
        measure(mesh_device, lambda: raw_forward(*args, **kwargs), warmup=100, samples=1)
        for round_index, choices in enumerate((order, tuple(reversed(order)))):
            for variant in choices:
                if hasattr(block.attn1, "_attention_override"):
                    del block.attn1._attention_override
                adapter = None if variant == "stock" else WanAttentionAdapter(variant)
                if adapter:
                    block.attn1._attention_override = functools.partial(adapter.run, "probe")
                row = dict(round=round_index, variant=variant, components={})
                components = row["components"]
                components["full_block"] = measure(
                    mesh_device, lambda: raw_forward(*args, **kwargs), warmup=12, verify=True
                )
                if variant == "stock":
                    components["attention_path"] = measure(
                        mesh_device, lambda: ring(*saved["ring_args"], **saved["ring_kwargs"])[0]
                    )
                else:
                    components["attention_path"] = measure(
                        mesh_device, lambda: adapter.run("probe", block.attn1, *qkv, 32760)
                    )

                    def prep_q():
                        return kernel.prepare(mesh_device, qkv[0], variant, is_q=True, cores=cores)

                    def prep_kv():
                        return tuple(kernel.prepare(mesh_device, x, variant, is_q=False, cores=cores) for x in qkv[1:])

                    pq, (pk, pv) = prep_q(), prep_kv()

                    def gather():
                        return tuple(
                            block.attn1.ccl_manager.all_gather(x, dim=2, mesh_axis=1, use_hyperparams=True)
                            for x in (pk, pv)
                        )

                    gk, gv = gather()
                    if variant in "EFG":
                        components["prepare_q"] = measure(mesh_device, prep_q)
                        components["prepare_kv"] = measure(mesh_device, prep_kv)
                    components["kv_all_gather"] = measure(mesh_device, gather)
                    components["sdpa"] = measure(
                        mesh_device,
                        lambda: kernel.attention(mesh_device, pq, gk, gv, variant, logical_k=32760, max_cores=cores),
                    )
                    del pq, pk, pv, gk, gv
                report["rows"].append(row)
                target.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    "WAN_PERF",
                    round_index,
                    variant,
                    {k: round(v["device_ms"], 4) for k, v in components.items()},
                    flush=True,
                )
        raise FinishedProbe

    block.forward = probe
    with pytest.raises(FinishedProbe), torch.no_grad():
        pipeline(prompts=[PROMPTS[0]], num_inference_steps=2, seed=42, output_type="uint8", traced=False)
    assert len(report["rows"]) == 14
    report["status"] = "completed"
    target.write_text(json.dumps(report, indent=2) + "\n")
