# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Candidate harness for the optimized Phi-3.5 decoder."""

import os
import statistics
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.fused_decoder_perf import _time_trace
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _page_table,
    _positions,
    _synthetic_state,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.fused_decoder import FusedDecoder
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizationPolicy, OptimizedDecoder


def _decoder(decoder_type, config, mesh_device, batch, **kwargs):
    return decoder_type.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
        **kwargs,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [int(value) for value in os.environ.get("OPT_PERF_BATCHES", "1,32").split(",")])
def test_candidate_traced_decode(mesh_device, batch):
    config = _config()
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(9100 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    positions = _positions([0] * batch, mesh_device)
    candidates = {
        "fused": (FusedDecoder, {}),
        "optimized": (
            OptimizedDecoder,
            {
                "optimization_policy": OptimizationPolicy(
                    attention_weight_dtype=ttnn.bfloat4_b,
                    mlp_gate_up_weight_dtype=ttnn.bfloat4_b,
                    mlp_down_weight_dtype=ttnn.bfloat4_b,
                )
            },
        ),
    }
    traces = {}
    resources = {}
    decodes = {}
    try:
        for name, (decoder_type, kwargs) in candidates.items():
            decoder = _decoder(decoder_type, config, mesh_device, batch, **kwargs)
            key_cache, value_cache = decoder.create_paged_kv_cache()
            resources[name] = (decoder, key_cache, value_cache)

            def decode(d=decoder, k=key_cache, v=value_cache):
                return d.decode_forward(
                    tt_hidden,
                    key_cache=k,
                    value_cache=v,
                    page_table=page_table,
                    current_positions=positions,
                    use_long_rope=False,
                )

            decode()
            ttnn.synchronize_device(mesh_device)
            decodes[name] = decode
        # All candidate tensors and programs must exist before any trace is
        # captured. Allocating the second candidate while the first trace is
        # alive can corrupt trace-owned buffers.
        for name, decode in decodes.items():
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            decode()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            traces[name] = trace_id
        for trace_id in traces.values():
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
        samples = {name: [] for name in traces}
        signpost(f"PERF_DECODE_B{batch}")
        for _ in range(int(os.environ.get("OPT_PROFILE_ITERATIONS", "100"))):
            for name, trace_id in traces.items():
                samples[name].append(_time_trace(mesh_device, trace_id))
        signpost(f"PERF_DECODE_B{batch}_END")
        print(
            "OPT_CANDIDATE "
            f"batch={batch} "
            + " ".join(f"{name}_mean_ms={statistics.mean(values):.6f}" for name, values in samples.items())
        )
    finally:
        for trace_id in traces.values():
            ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [int(value) for value in os.environ.get("OPT_PERF_BATCHES", "1,32").split(",")])
def test_candidate_warmed_prefill(mesh_device, batch):
    config = _config()
    hidden = torch.randn(batch, 128, config.hidden_size, generator=torch.Generator().manual_seed(9500 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_prefill(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    results = {}
    decoder_types = {"fused": FusedDecoder, "optimized": OptimizedDecoder}
    candidate_names = os.environ.get("OPT_PREFILL_CANDIDATES", "fused,optimized").split(",")
    for name in candidate_names:
        decoder_type = decoder_types[name]
        decoder = _decoder(decoder_type, config, mesh_device, batch)
        key_cache, value_cache = decoder.create_paged_kv_cache()

        def prefill():
            return decoder.prefill_forward(
                tt_hidden, key_cache=key_cache, value_cache=value_cache, page_table=page_table
            )

        prefill()
        ttnn.synchronize_device(mesh_device)
        samples = []
        if name == "optimized":
            signpost(f"PERF_PREFILL_B{batch}")
        for _ in range(int(os.environ.get("OPT_PREFILL_ITERATIONS", "20"))):
            start = time.perf_counter()
            prefill()
            ttnn.synchronize_device(mesh_device)
            samples.append(1000 * (time.perf_counter() - start))
        if name == "optimized":
            signpost(f"PERF_PREFILL_B{batch}_END")
        results[name] = statistics.mean(samples)
    print(f"OPT_PREFILL batch={batch} sequence=128 values={results}")
