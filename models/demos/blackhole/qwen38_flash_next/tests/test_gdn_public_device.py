# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qualification of public GDN after merged owner prerequisite 96cc4a7937.

Forced phased, forced fused, and automatic public dispatch must agree with
the independent recurrence and retain trace/address reuse. Saved old-arithmetic
outputs are optional comparison evidence and do not change the numeric gate.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from ttnn.operations.transformer_golden import recurrent_gated_delta_rule
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, pcc
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_public_adapter import chunk_public

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")


def host(tensor):
    replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(replicas[0], other) for other in replicas[1:])
    return replicas[0]


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 8_000_000}], indirect=True)
@pytest.mark.parametrize(
    "rows,nonzero_state,committed_rows",
    [(rows, nonzero, None) for rows in (32, 128, 2048) for nonzero in (False, True)]
    + [(32, True, committed) for committed in (0, 1, 5, 31)],
)
def test_gdn_merged_public_paths(mesh_device, rows, nonzero_state, committed_rows, record_property):
    heads, dim, chunk = 12, 128, 32
    owned, traces, observations = [], [], []

    def upload(value, dtype):
        tensor = ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        owned.append(tensor)
        return tensor

    ii, jj = torch.arange(chunk)[:, None], torch.arange(chunk)[None, :]
    low_i, low_j = ii < 16, jj < 16
    masks = torch.cat([(low_i & low_j).float(), (~low_i & ~low_j).float(), (~low_i & low_j).float()], 1)
    constants = tuple(
        upload(x[None, None], ttnn.float32)
        for x in (
            torch.eye(chunk),
            torch.ones(chunk, chunk).tril(),
            torch.ones(chunk, chunk),
            masks,
        )
    )
    configs = {
        "phased": ttnn.ChunkGdnPhasedProgramConfig(),
        "fused": ttnn.ChunkGdnFusedProgramConfig(),
        "default": None,
    }
    baseline_dir = (
        Path(os.environ["QWEN38_GDN_BASELINE_OUTPUT"]) if os.environ.get("QWEN38_GDN_BASELINE_OUTPUT") else None
    )
    cached = None
    addresses = []
    try:
        for seed in (20260929, 20260930):
            rng = torch.Generator().manual_seed(seed)

            def random(shape):
                return torch.randn(shape, generator=rng)

            # Rounded producer bytes define the independent recurrence inputs.
            # Both source q scale folds remain in the query; no normalization occurs in the adapter.
            q = (F.normalize(random((1, rows, heads, dim)), dim=-1) * dim**-1).bfloat16()
            k = F.normalize(random((1, rows, heads, dim)), dim=-1).bfloat16()
            v = random((1, rows, heads, dim)).bfloat16()
            g = -F.softplus(random((1, rows, heads))) * 0.5
            beta = torch.sigmoid(random((1, rows, heads)))
            if committed_rows is not None:
                # The MTP commit masks beta and log-decay, making the suffix an identity update.
                beta[:, committed_rows:] = 0
                g[:, committed_rows:] = 0
            state = random((1, heads, dim, dim)) * 0.1 if nonzero_state else torch.zeros(1, heads, dim, dim)
            q_c, k_c = [
                upload(x.permute(0, 2, 1, 3).reshape(heads, rows // chunk, chunk, dim), ttnn.bfloat16) for x in (q, k)
            ]
            g_c, beta_c = [
                upload(x.permute(0, 2, 1).reshape(heads, rows // chunk, chunk, 1), ttnn.float32) for x in (g, beta)
            ]
            value = upload(v.reshape(1, 1, rows, heads * dim), ttnn.bfloat16)
            initial = upload(state, ttnn.float32)
            addresses.append(q_c.buffer_address())
            assert len(set(addresses)) == len(addresses), "keep prior seed inputs alive to force fresh addresses"
            args = (q_c, k_c, value, beta_c, g_c, initial, constants)

            def invoke(kind):
                return chunk_public(*args, rows_total=rows, program_config=configs[kind])

            outputs, actual_by_kind, path_cache = {}, {}, {}
            for kind in configs:
                before_path = mesh_device.num_program_cache_entries()
                outputs[kind] = invoke(kind)
                owned.extend(outputs[kind])
                actual_by_kind[kind] = tuple(host(t) for t in outputs[kind])
                path_cache[kind] = dict(before=before_path, after=mesh_device.num_program_cache_entries())
            if seed == 20260929:
                assert (
                    path_cache["fused"]["after"] > path_cache["phased"]["after"]
                ), "forced fused must compile a distinct program"
            assert (
                path_cache["default"]["after"] == path_cache["default"]["before"]
            ), "default must reuse one of the explicitly warmed configurations"
            actual = actual_by_kind["phased"]
            reference = recurrent_gated_delta_rule(
                q.float(),
                k.float(),
                v.float(),
                beta,
                g,
                scale=1.0,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm=False,
            )
            expected = (reference[0].permute(0, 2, 1, 3).reshape(heads, rows, dim), reference[1])
            correlations = {kind: [pcc(a, b) for a, b in zip(pair, expected)] for kind, pair in actual_by_kind.items()}
            old_comparison = None
            if baseline_dir is not None:
                baseline_path = (
                    baseline_dir / f"rows{rows}-state{int(nonzero_state)}-commit{committed_rows}-seed{seed}.pt"
                )
                baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
                old_pair = (baseline["output"], baseline["state"])
                old_comparison = [
                    dict(
                        bitwise_equal=torch.equal(a, b),
                        pcc=pcc(a, b),
                        max_abs=float((a - b).abs().max()),
                        changed_elements=int((a != b).sum()),
                        old_sha256=digest(b),
                        new_sha256=digest(a),
                    )
                    for a, b in zip(actual, old_pair)
                ]
            diagnostic = dict(
                seed=seed,
                rows=rows,
                nonzero_state=nonzero_state,
                committed_rows=committed_rows,
                public_configs={kind: repr(config) for kind, config in configs.items()},
                pcc=correlations,
                old_arithmetic_comparison=old_comparison,
                path_program_cache=path_cache,
                output_sha256={kind: [digest(x) for x in pair] for kind, pair in actual_by_kind.items()},
            )
            print("GDN_MERGED_BEFORE_GATES=" + json.dumps(diagnostic, sort_keys=True), flush=True)
            for kind, pair in actual_by_kind.items():
                assert all(torch.isfinite(t).all() for t in pair), kind
                assert min(correlations[kind]) >= 0.99999, (kind, correlations[kind])
                assert all(
                    torch.equal(a, b) for a, b in zip(pair, actual)
                ), f"{kind} must be bitwise identical to explicit phased"
            assert torch.equal(host(initial), state), "public op must not mutate the persistent input state"
            before = mesh_device.num_program_cache_entries()
            if cached is not None:
                assert before == cached
            cached = before
            trace_ids, captured = {}, {}
            for kind in configs:
                tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                try:
                    captured[kind] = invoke(kind)
                finally:
                    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
                trace_ids[kind] = tid
                traces.append(tid)
                owned.extend(captured[kind])
                # These two tensors are intentional trace destinations. Inputs and
                # constants stay tracked; no broad allocation scope is exempted.
                for destination in captured[kind]:
                    TraceAllocationTracker.acknowledge_corruptible(destination)
                for _ in range(3):
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                    for observed, wanted in zip(captured[kind], actual):
                        assert torch.equal(host(observed), wanted)

            # Five alternating warmups, then five alternating batches of 100 replays.
            # Host wall time includes trace submission and final synchronization.
            timings = {kind: [] for kind in trace_ids}
            for _ in range(5):
                for tid in trace_ids.values():
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
            for repeat in range(5):
                order = ("phased", "fused", "default") if repeat % 2 == 0 else ("default", "fused", "phased")
                for kind in order:
                    start = time.perf_counter()
                    for _ in range(100):
                        ttnn.execute_trace(mesh_device, trace_ids[kind], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    timings[kind].append((time.perf_counter() - start) / 100)
            assert mesh_device.num_program_cache_entries() == cached
            diagnostic.update(
                trace_seconds_per_call=timings, cache_entries=cached, fresh_input_addresses=list(addresses)
            )
            observations.append(diagnostic)
            for tid in trace_ids.values():
                ttnn.release_trace(mesh_device, tid)
                traces.remove(tid)
        record_property("gdn_merged_public_paths", json.dumps(observations, sort_keys=True))
        print("GDN_MERGED_PUBLIC_PATHS=" + json.dumps(observations, sort_keys=True))
    finally:
        for tid in traces:
            ttnn.release_trace(mesh_device, tid)
        for tensor in owned:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
