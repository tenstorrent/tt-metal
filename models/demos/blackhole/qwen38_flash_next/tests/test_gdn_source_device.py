# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Independent reference, frozen source words and trace controls for source GDN chunks."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from ttnn.operations.transformer_golden import recurrent_gated_delta_rule
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, pcc
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_source_chunk import chunk_source, chunk_token_major

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held Blackhole four-die mesh"
)


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
    [(rows, nonzero, None) for rows in (32, 128, 2048, 4096) for nonzero in (False, True)]
    + [(32, True, committed) for committed in (0, 1, 5, 31)],
)
@pytest.mark.parametrize("entry", ["head_major", "token_major"])
def test_source_chunk_synthetic(mesh_device, rows, nonzero_state, committed_rows, entry, record_property):
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

            if entry == "token_major":
                qt = upload(q * 8, ttnn.bfloat16)
                kt = upload(k, ttnn.bfloat16)
                vt = ttnn.reshape(value, (1, rows, heads * dim))
                gt = upload(g, ttnn.float32)
                bt = upload(beta, ttnn.float32)

            def invoke(kind):
                if entry == "head_major":
                    return chunk_source(*args, rows_total=rows)
                # A power-of-two scale makes the interface fold exactly testable
                # from the same admitted source producer words.
                return chunk_token_major(qt, kt, vt, gt, bt, initial, constants, rows_total=rows, scale=0.125)

            outputs = {"sequence": invoke("sequence")}
            owned.extend(outputs["sequence"])
            actual = tuple(host(t) for t in outputs["sequence"])
            input_hashes = {
                name: digest(x) for name, x in zip(("q", "k", "v", "beta", "g", "initial"), (q, k, v, beta, g, state))
            }
            key = f"rows{rows}-state{int(nonzero_state)}-commit{committed_rows}-seed{seed}"
            source = json.loads((Path(__file__).parent / "data/source_chunk_words.json").read_text())
            expected_row = source["cases"][key]
            assert expected_row["input_sha256"] == input_hashes, "Fixture input generation changed"
            assert expected_row["output_sha256"] == [digest(t) for t in actual], (
                "Source sequence words differ",
                key,
                entry,
            )
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
            correlations = [pcc(a, b) for a, b in zip(actual, expected)]
            assert min(correlations) >= 0.99999
            assert torch.equal(host(initial), state), "persistent initial state changed"
            before = mesh_device.num_program_cache_entries()
            if cached is not None:
                assert before == cached
            cached = before
            trace_ids, captured = {}, {}
            for kind in ("sequence",):
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

            timings = {}  # Diagnostic numerical/replay run; no timing comparison.
            assert torch.equal(host(initial), state), "trace changed persistent initial state"
            assert mesh_device.num_program_cache_entries() == cached
            observations.append(
                dict(
                    key=key,
                    input_sha256=input_hashes,
                    seed=seed,
                    rows=rows,
                    nonzero_state=nonzero_state,
                    committed_rows=committed_rows,
                    pcc=correlations,
                    output_sha256=[digest(x) for x in actual],
                    trace_seconds_per_call=timings,
                    cache_entries=cached,
                    fresh_input_addresses=list(addresses),
                )
            )
            for tid in trace_ids.values():
                ttnn.release_trace(mesh_device, tid)
                traces.remove(tid)
        record_property("source_chunk_sequence", json.dumps(observations, sort_keys=True))
        print("GDN_SOURCE_SEQUENCE=" + json.dumps(observations, sort_keys=True))
    finally:
        for tid in traces:
            ttnn.release_trace(mesh_device, tid)
        for tensor in owned:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
