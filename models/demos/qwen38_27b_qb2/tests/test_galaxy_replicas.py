# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""G0: independent TP4 replicas, exact tokens and concurrent/isolated decode time.

Prefill and trace warmup run serially. Timed decode enqueues one step on each
submesh without readbacks, so all replicas can execute simultaneously. Each
replica is compared with its own single-active-replica timing using the same
measurement path. Host completion times are upper bounds, not device timestamps.
This test does not measure serving TTFT or high-context/high-batch throughput.
"""

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.qwen38_27b_qb2.demo.galaxy_serving import model_source_hashes
from models.demos.qwen38_27b_qb2.tests.galaxy_prompt import qualification_prompt
from models.demos.qwen38_27b_qb2.tests.replica_timing import completion_times
from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric
from models.demos.qwen38_27b_qb2.tt.model import checkpoint_path


def decode_window(generators, prompt, output_tokens, read_order):
    first_tokens = []
    for gen in generators:
        # Warmup already reserved cache/history and captured the graphs at the
        # full output length. G=1 resets request state and reuses that geometry.
        first_tokens.append(gen.generate(prompt, 1)[0])
        assert gen.trace is not None and gen.trace_records_history
        assert gen.history_count == 0 and gen.history_capacity >= output_tokens - 1
        ttnn.synchronize_device(gen.mesh)
    counters = [gen.counters.copy() for gen in generators]
    started = time.perf_counter()
    for _ in range(output_tokens - 1):
        for gen in generators:
            gen.decode_forward(
                page_table=gen.page_table,
                kv_cache=gen.cache,
                read_from_device=False,
                record_history=True,
            )
    enqueue_s = time.perf_counter() - started
    completed = [
        timestamp - started
        for timestamp in completion_times([gen.mesh for gen in generators], ttnn.synchronize_device, read_order)
    ]
    rows = []
    for i, gen in enumerate(generators):
        delta = gen.counters - counters[i]
        assert delta["trace_captures"] == 0, "Timed window must reuse warmed traces"
        assert delta["model_replays"] == delta["sampling_replays"] == output_tokens - 1
        assert delta["token_readbacks"] == delta["full_logits_readbacks"] == 0
        rows.append(
            dict(
                output_tokens=[first_tokens[i]] + gen._read_history()[:, 0].tolist(),
                decode_s=completed[i],
                tpot_ms=completed[i] * 1000 / (output_tokens - 1),
                counters=dict(delta),
            )
        )
    return dict(
        replicas=rows,
        enqueue_s=enqueue_s,
        wall_s=max(completed),
        completion_read_order=read_order,
        completion_method="independent_host_waiters",
    )


@pytest.mark.skipif(
    os.getenv("QWEN_GALAXY_REPLICAS", "1") == "1", reason="explicit multi-replica allocated-Galaxy test"
)
def test_concurrent_galaxy_replicas():
    count = int(os.environ["QWEN_GALAXY_REPLICAS"])
    assert 2 <= count <= 8
    output_tokens, repeats = 128, 5
    output = Path(os.environ["QWEN_GALAXY_RECEIPT"])
    prompt = qualification_prompt(AutoTokenizer.from_pretrained(checkpoint_path(), local_files_only=True))
    source_sha256 = model_source_hashes(Path(__file__).resolve().parents[1])
    torch.set_num_threads(8)
    assert ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY
    configure_fabric(topology=ttnn.Topology.Linear)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), trace_region_size=200000000)
    generators = []
    report = dict(
        parent_mesh=[8, 4],
        replica_mesh=[1, 4],
        topology="linear",
        replicas_requested=count,
        replicas_executed=0,
        state="loading",
        output_tokens=output_tokens,
        repeats=repeats,
        warmup=[],
        isolated=[],
        concurrent=[],
        timing_scope="traced decode plus device sampling; synchronized host completion; no prefill/readback",
        completion_method="independent_host_waiters",
        passed=False,
        prompt_tokens=prompt,
        loaded=[],
        source_sha256=source_sha256,
    )
    try:
        for i in range(count):
            mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(i, 0))
            started = time.perf_counter()
            gen = build_generator(Path(__file__).resolve().parents[1], mesh, topology=ttnn.Topology.Linear)
            generators.append(gen)
            assert len(gen.model.layers) == 64
            if "precision" not in report:
                report["precision"] = gen.model.precision
            assert gen.model.precision == report["precision"], "Replica precision must be identical"
            report["loaded"].append(
                dict(replica=i, setup_s=time.perf_counter() - started, device_ids=list(mesh.get_device_ids()))
            )
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"GALAXY_REPLICA_LOADED replica={i} seconds={time.perf_counter() - started:.3f}", flush=True)
        assert len({id(gen.model.ccl) for gen in generators}) == count
        report["state"] = "warming"
        reference = None
        for i, gen in enumerate(generators):
            tokens = gen.generate(prompt, output_tokens)
            if reference is None:
                reference = tokens
            assert tokens == reference, f"Replica {i} differs from replica 0 during warmup"
            report["warmup"].append(dict(replica=i, perf=dict(gen.last_perf)))
            report["replicas_executed"] = i + 1
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"GALAXY_REPLICA_WARMED replica={i}", flush=True)
        assert len({id(gen.cache) for gen in generators}) == count
        report["reference_tokens"] = reference
        report["text"] = generators[0].tokenizer.decode(reference)
        report["state"] = "isolated"
        for i, gen in enumerate(generators):
            samples = [decode_window([gen], prompt, output_tokens, [0]) for _ in range(repeats)]
            report["isolated"].append(dict(replica=i, samples=samples))
            assert all(row["replicas"][0]["output_tokens"] == reference for row in samples)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"GALAXY_ISOLATED_COMPLETE replica={i}", flush=True)
        report["state"] = "concurrent"
        for repeat in range(repeats):
            order = [(repeat + i) % count for i in range(count)]
            result = decode_window(generators, prompt, output_tokens, order)
            report["concurrent"].append(result)
            assert all(row["output_tokens"] == reference for row in result["replicas"])
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"GALAXY_CONCURRENT_COMPLETE repeat={repeat} seconds={result['wall_s']:.3f}", flush=True)
        comparisons = []
        for i in range(count):
            isolated = statistics.median(row["replicas"][0]["tpot_ms"] for row in report["isolated"][i]["samples"])
            concurrent = statistics.median(row["replicas"][i]["tpot_ms"] for row in report["concurrent"])
            comparisons.append(
                dict(replica=i, isolated_tpot_ms=isolated, concurrent_tpot_ms=concurrent, ratio=concurrent / isolated)
            )
        report["comparisons"] = comparisons
        report["aggregate_decode_tokens_per_s"] = (
            count * (output_tokens - 1) / statistics.median(row["wall_s"] for row in report["concurrent"])
        )
        assert all(row["ratio"] <= 1.03 for row in comparisons), comparisons
        report["passed"] = True
        report["state"] = "completed"
    except BaseException as error:
        report["state"] = "failed"
        report["error"] = dict(type=type(error).__name__, message=str(error)[:1000])
        raise
    finally:
        # Preserve measured failures as well as passes before releasing devices.
        try:
            output.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            try:
                for gen in generators:
                    gen.close()
            finally:
                ttnn.close_mesh_device(parent)
