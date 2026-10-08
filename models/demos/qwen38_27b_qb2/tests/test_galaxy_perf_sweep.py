# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in full-model native sweep; all hardware access must be serialized."""

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.tests.sweep_recovery import is_dram_allocation_error, resume_measurements
from models.demos.qwen38_27b_qb2.tests.sweep_report import render, save_report, summarize
from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric


def dram_allocation(generators):
    """Host allocator snapshots outside timing windows; no allocator block-table dump."""
    fields = (
        "num_banks",
        "total_bytes_per_bank",
        "total_bytes_allocated_per_bank",
        "total_bytes_free_per_bank",
        "largest_contiguous_bytes_free_per_bank",
    )
    return [
        {
            "replica": replica,
            "device_ids": list(gen.mesh.get_device_ids()),
            **{key: int(getattr(view, key)) for key in fields},
        }
        for replica, gen in enumerate(generators)
        for view in [ttnn.get_memory_view(gen.mesh, ttnn.BufferType.DRAM)]
    ]


def drop_request_buffers(gen):
    """Release previous geometry before allocating a larger KV pool."""
    ttnn.synchronize_device(gen.mesh)
    gen._release_traces()
    for name in (
        "cache",
        "positions",
        "page_table",
        "page_host",
        "rope_indices",
        "logits",
        "prefill_sample_input",
        "token_history",
        "history_cursor",
        "_prefill_sampling_cache",
        "_recurrent_reset_warmed",
    ):
        setattr(gen, name, None)
    gen.history_capacity = gen.history_count = 0
    gen.prefill_signatures.clear()
    gen.remaining_steps = gen.active_slots = None
    gc.collect()


def run_batch(generators, prompts, output_tokens):
    batch, length = prompts.shape
    slots = tuple(range(batch))
    before = [gen.counters.copy() for gen in generators]
    first_tokens, ttfts, prefill_durations = [], [], []
    started = time.perf_counter()
    for gen in generators:
        gen.set_sampling_params(top_k=1, seed=0)
        gen.reset_recurrent_slots(list(slots))
        gen._reset_history()
        prefill_started = time.perf_counter()
        logits = gen.prefill_forward(
            prompts,
            page_table=gen.page_table,
            kv_cache=gen.cache,
            prompt_lens=[length] * batch,
        )
        ttnn.synchronize_device(gen.mesh)
        prefill_durations.append(time.perf_counter() - prefill_started)
        gen.sample_prefill(logits)
        del logits
        first = gen._read_tokens()[:batch]
        first_tokens.append(first)
        ttfts.extend([time.perf_counter() - started] * batch)
    decode_started = time.perf_counter()
    for step in range(output_tokens - 1):
        for gen, first in zip(generators, first_tokens):
            initial = (
                dict(tokens=first, start_pos=torch.full((batch,), length), active_slots=slots) if step == 0 else {}
            )
            gen.decode_forward(
                page_table=gen.page_table,
                kv_cache=gen.cache,
                read_from_device=False,
                record_history=True,
                **initial,
            )
    histories = [gen._read_history()[:, :batch] for gen in generators]
    finished = time.perf_counter()
    token_hashes, counters = [], []
    for gen, first, history, initial_counters in zip(generators, first_tokens, histories, before):
        tokens = torch.cat([first[None], history], dim=0).T
        assert tuple(tokens.shape) == (batch, output_tokens)
        assert ((tokens >= 0) & (tokens < gen.model.config.vocab_size)).all()
        token_hashes.append(hashlib.sha256(tokens.contiguous().numpy().tobytes()).hexdigest())
        counters.append(dict(gen.counters - initial_counters))
    return dict(
        ttft_s=ttfts,
        prefill_s=sum(prefill_durations),
        prefill_s_per_replica=prefill_durations,
        decode_s=finished - decode_started,
        elapsed_s=finished - started,
        output_sha256_per_replica=token_hashes,
        counters_per_replica=counters,
        trace_captures=sum(row.get("trace_captures", 0) for row in counters),
    )


@pytest.mark.skipif(os.getenv("QWEN_GALAXY_SWEEP") != "1", reason="explicit allocated-Galaxy performance sweep")
def test_galaxy_perf_sweep():
    directory = Path(os.environ["QWEN_SWEEP_RESULTS"])
    report = json.loads((directory / "sweep.json").read_text())
    assert report["state"] == "queued", "Use a new receipt directory; do not overwrite partial runs"
    assert report["replicas"] in (1, 8) and report["output_tokens"] == 128
    torch.set_num_threads(8)
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    generators = []
    active_cell = None
    try:
        report["state"] = "loading"
        report["configuration"] = {
            "topology": "linear",
            "num_links": 2,
            "native_layers": 64,
            "environment": {key: value for key, value in os.environ.items() if key.startswith("QWEN_")},
        }
        source = Path(__file__).resolve().parents[1]
        report["source_sha256"] = model_source_hashes(source)
        save_report(report, directory)
        render(report, directory)
        for replica in range(report["replicas"]):
            mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(replica, 0))
            tick = time.perf_counter()
            gen = build_generator(source, mesh, topology=ttnn.Topology.Linear)
            generators.append(gen)
            assert len(gen.model.layers) == 64
            report.setdefault("setup_s_per_replica", []).append(time.perf_counter() - tick)
        report["precision"] = generators[0].model.precision
        report["dram_after_model_load"] = dram_allocation(generators)
        resume_measurements(report, json.loads(os.environ.get("QWEN_SWEEP_RESUME_FROM", "[]")))
        report["state"] = "running"
        passage = (
            "The scientific method tests explanations against observations. "
            "Describe an experiment, its controls, and the evidence needed to evaluate the result. "
        )
        base = generators[0].tokenizer.encode(passage, add_special_tokens=False)
        for cell in report["cells"]:
            if cell["status"] in ("capacity_guard", "implementation_guard", "oom"):
                continue
            length, batch = cell["input_tokens"], cell["batch_per_replica"]
            ids = (base * (length // len(base) + 1))[:length]
            prompt_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if cell["status"] == "completed":
                assert cell["prompt_sha256"] == prompt_hash, "Resume prompt tokens differ"
                continue
            active_cell = cell
            cell["status"] = "running"
            cell["prompt_sha256"] = prompt_hash
            save_report(report, directory)
            prompts = torch.tensor([ids] * batch, dtype=torch.int64)
            print(f"SWEEP_CELL_BEGIN isl={length} batch={batch} replicas={len(generators)}", flush=True)
            for gen in generators:
                drop_request_buffers(gen)
            cell["dram_before_cache"] = dram_allocation(generators)
            for gen in generators:
                gen._ensure_cache(batch, length + report["output_tokens"] - 1)
                gen._ensure_history(report["output_tokens"] - 1)
                ttnn.synchronize_device(gen.mesh)
            cell["dram_after_cache"] = dram_allocation(generators)
            save_report(report, directory)
            cell["warmup"] = run_batch(generators, prompts, report["output_tokens"])
            cell["dram_after_warmup"] = dram_allocation(generators)
            reference = cell["warmup"]["output_sha256_per_replica"]
            assert len(set(reference)) == 1, "Independent replicas produced different greedy outputs"
            cell["samples"] = []
            for repeat in range(report["measured_runs"]):
                sample = run_batch(generators, prompts, report["output_tokens"])
                cell["samples"].append(sample)
                save_report(report, directory)
                assert sample["output_sha256_per_replica"] == reference, "Greedy outputs changed between repeats"
                assert sample["trace_captures"] == 0, "Warm measurement recaptured a trace"
                print(f"SWEEP_REPEAT_COMPLETE isl={length} batch={batch} repeat={repeat}", flush=True)
            cell["summary"] = summarize(
                cell["samples"],
                concurrency=cell["concurrency"],
                output_tokens=report["output_tokens"],
                input_tokens=length,
            )
            cell["status"] = "completed"
            cell["dram_after_measurement"] = dram_allocation(generators)
            save_report(report, directory)
            render(report, directory)
            print(f"SWEEP_CELL_COMPLETE {json.dumps(cell['summary'])}", flush=True)
        report["state"] = (
            "completed_with_oom" if any(cell["status"] == "oom" for cell in report["cells"]) else "completed"
        )
    except BaseException as error:
        allocation_failure = active_cell is not None and is_dram_allocation_error(error)
        report["state"] = "allocation_failed" if allocation_failure else "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error))
        if active_cell is not None and active_cell["status"] == "running":
            active_cell["status"] = "oom" if allocation_failure else "failed"
            active_cell["error"] = report["error"]
            active_cell["reason"] = report["error"]["message"]
            if allocation_failure:
                active_cell["dram_after_allocation_failure"] = dram_allocation(generators)
        for cell in report["cells"]:
            if cell["status"] == "queued":
                cell["status"] = "not_run"
        raise
    finally:
        try:
            save_report(report, directory)
            render(report, directory)
        finally:
            try:
                try:
                    for gen in generators:
                        ttnn.synchronize_device(gen.mesh)
                        gen.close()
                finally:
                    ttnn.close_mesh_device(parent)
                report["cleanup_completed"] = True
            except BaseException as error:
                report["cleanup_completed"] = False
                report["cleanup_error"] = dict(type=type(error).__name__, message=str(error))
                raise
            finally:
                save_report(report, directory)
