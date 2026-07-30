# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Like-for-like functional/fused warmed latency and fused Tracy entry points."""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _page_table,
    _positions,
    _synthetic_state,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.fused_decoder import FusedDecoder


def _decoder(decoder_type, config, mesh_device, batch):
    return decoder_type.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
    )


def _measure(call, mesh_device, *, iterations=20):
    iterations = int(os.environ.get("FUSED_PROFILE_ITERATIONS", iterations))
    call()
    ttnn.synchronize_device(mesh_device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        ttnn.synchronize_device(mesh_device)
        samples.append(1000 * (time.perf_counter() - start))
    return statistics.mean(samples), min(samples)


def _profile_order():
    configured_order = os.environ.get("FUSED_PROFILE_ORDER")
    if os.environ.get("FUSED_PROFILE_ONLY") and configured_order is None:
        return ("fused",)
    order = tuple(part.strip() for part in (configured_order or "functional,fused").split(","))
    if sorted(order) != ["functional", "fused"]:
        raise ValueError("FUSED_PROFILE_ORDER must be 'functional,fused' or 'fused,functional'")
    return order


def _time_trace(mesh_device, trace_id):
    start = time.perf_counter()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    return 1000 * (time.perf_counter() - start)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_profile_warmed_prefill_before_after(mesh_device, batch):
    config = _config()
    hidden = torch.randn(batch, 128, config.hidden_size, generator=torch.Generator().manual_seed(8000 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_prefill(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    results = {}
    candidates = (
        (("fused", FusedDecoder),)
        if os.environ.get("FUSED_PROFILE_ONLY")
        else (
            ("functional", FunctionalDecoder),
            ("fused", FusedDecoder),
        )
    )
    for name, decoder_type in candidates:
        decoder = _decoder(decoder_type, config, mesh_device, batch)
        key_cache, value_cache = decoder.create_paged_kv_cache()

        def prefill():
            return decoder.prefill_forward(
                tt_hidden, key_cache=key_cache, value_cache=value_cache, page_table=page_table
            )

        if name == "fused":
            signpost(f"FUSED_PREFILL_B{batch}")
        results[name] = _measure(prefill, mesh_device)
        if name == "fused":
            signpost(f"FUSED_PREFILL_B{batch}_END")
    print(f"PERF_RESULTS mode=prefill batch={batch} sequence=128 values={results}")
    # Prefill is reported at both required batches.  Traced decode, below, is
    # the selection gate because it is the stage's serving-critical workload.


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_profile_traced_decode_before_after(mesh_device, batch):
    selected_batch = os.environ.get("FUSED_PROFILE_BATCH")
    if selected_batch is not None and batch != int(selected_batch):
        pytest.skip(f"FUSED_PROFILE_BATCH={selected_batch}")

    config = _config()
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(8100 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    positions = _positions([0] * batch, mesh_device)
    order = _profile_order()
    decoder_types = {"functional": FunctionalDecoder, "fused": FusedDecoder}
    trace_ids = {}
    trace_resources = {}
    try:
        for name in order:
            decoder = _decoder(decoder_types[name], config, mesh_device, batch)
            key_cache, value_cache = decoder.create_paged_kv_cache()
            # Keep trace-owned tensors alive through all replays.
            trace_resources[name] = (decoder, key_cache, value_cache)

            def decode():
                return decoder.decode_forward(
                    tt_hidden,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    page_table=page_table,
                    current_positions=positions,
                    use_long_rope=False,
                )

            decode()
            ttnn.synchronize_device(mesh_device)
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            decode()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            trace_ids[name] = trace_id

        warmup = int(os.environ.get("FUSED_PROFILE_WARMUP", "1"))
        for _ in range(warmup):
            for name in order:
                ttnn.execute_trace(mesh_device, trace_ids[name], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)

        iterations = int(os.environ.get("FUSED_PROFILE_ITERATIONS", "100"))
        paired_samples = []
        signpost_name = (
            f"FUSED_DECODE_B{batch}" if order == ("fused",) else f"PAIRED_DECODE_B{batch}_{'_'.join(order).upper()}"
        )
        signpost(signpost_name)
        for pair_index in range(iterations):
            sample = {"pair": pair_index}
            for name in order:
                sample[f"{name}_ms"] = _time_trace(mesh_device, trace_ids[name])
            if len(order) == 2:
                sample["fused_minus_functional_ms"] = sample["fused_ms"] - sample["functional_ms"]
            paired_samples.append(sample)
        signpost(f"{signpost_name}_END")
    finally:
        for trace_id in trace_ids.values():
            ttnn.release_trace(mesh_device, trace_id)

    results = {
        name: (
            statistics.mean(sample[f"{name}_ms"] for sample in paired_samples),
            min(sample[f"{name}_ms"] for sample in paired_samples),
        )
        for name in order
    }
    print(f"PERF_RESULTS mode=decode batch={batch} context=128 traced=true values={results}")
    json_path = os.environ.get("FUSED_PROFILE_JSON")
    if json_path:
        if len(order) != 2:
            raise ValueError("FUSED_PROFILE_JSON requires both functional and fused candidates")
        output = {
            "schema_version": 1,
            "mode": "traced_decode",
            "batch": batch,
            "context": 128,
            "order": list(order),
            "warmup_pairs": warmup,
            "sample_pairs": iterations,
            "samples": paired_samples,
        }
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n")
