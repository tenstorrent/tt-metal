# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Same-harness functional/optimized Phi decoder performance and policy sweeps."""

from __future__ import annotations

import os
import time

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _page_table,
    _positions,
    _real_state,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizationPolicy, OptimizedDecoder


def _dtype(name):
    return {
        "bf16": ttnn.bfloat16,
        "bfp8": ttnn.bfloat8_b,
        "bfp4": ttnn.bfloat4_b,
    }[name]


def _fidelity(name):
    return {
        "lofi": ttnn.MathFidelity.LoFi,
        "hifi2": ttnn.MathFidelity.HiFi2,
        "hifi4": ttnn.MathFidelity.HiFi4,
    }[name]


def _policy():
    return OptimizationPolicy(
        attention_weight_dtype=_dtype(os.environ.get("PHI_ATTN_DTYPE", "bfp4")),
        gate_up_weight_dtype=_dtype(os.environ.get("PHI_GATE_UP_DTYPE", "bfp4")),
        down_weight_dtype=_dtype(os.environ.get("PHI_DOWN_DTYPE", "bfp4")),
        kv_cache_dtype=_dtype(os.environ.get("PHI_KV_DTYPE", "bfp8")),
        attention_math_fidelity=_fidelity(os.environ.get("PHI_ATTN_FIDELITY", "lofi")),
        gate_up_math_fidelity=_fidelity(os.environ.get("PHI_GATE_UP_FIDELITY", "lofi")),
        down_math_fidelity=_fidelity(os.environ.get("PHI_DOWN_FIDELITY", "lofi")),
        decode_core_count=int(os.environ.get("PHI_DECODE_CORES", "16")),
        in0_block_w_qkv=int(os.environ.get("PHI_QKV_BLOCK_W", "6")),
        in0_block_w_o=int(os.environ.get("PHI_O_BLOCK_W", "6")),
        in0_block_w_gate_up=int(os.environ.get("PHI_GATE_UP_BLOCK_W", "6")),
        in0_block_w_down=int(os.environ.get("PHI_DOWN_BLOCK_W", "16")),
        use_explicit_prefill_programs=os.environ.get("PHI_EXPLICIT_PREFILL", "0") == "1",
        use_explicit_decode_sdpa=os.environ.get("PHI_EXPLICIT_SDPA", "1") == "1",
        split_decode_qkv=os.environ.get("PHI_SPLIT_QKV", "0") == "1",
        split_decode_gate_up=os.environ.get("PHI_SPLIT_GATE_UP", "1") == "1",
    )


def _decoder(config, mesh_device, *, batch, max_context):
    implementation = os.environ.get("PHI_DECODER_IMPL", "optimized")
    decoder_cls = FunctionalDecoder if implementation == "functional" else OptimizedDecoder
    kwargs = {}
    if decoder_cls is OptimizedDecoder:
        kwargs["optimization_policy"] = _policy()
    return implementation, decoder_cls.from_state_dict(
        _real_state(),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=max_context,
        **kwargs,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_profile_warmed_prefill(mesh_device, batch):
    config = _config()
    implementation, decoder = _decoder(config, mesh_device, batch=batch, max_context=128)
    hidden = torch.randn(batch, 128, config.hidden_size, generator=torch.Generator().manual_seed(101 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_prefill(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()

    def run():
        return decoder.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache, page_table=page_table)

    run()
    ttnn.synchronize_device(mesh_device)
    signpost(f"PERF_PREFILL_B{batch}")
    start = time.perf_counter()
    output = run()
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    signpost(f"PERF_PREFILL_B{batch}_END")
    assert tuple(output.shape) == (1, batch, 128, config.hidden_size)
    print(f"PERF_RESULT impl={implementation} mode=prefill batch={batch} sequence=128 warmed_ms={elapsed_ms:.6f}")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_profile_traced_decode(mesh_device, batch):
    config = _config()
    implementation, decoder = _decoder(config, mesh_device, batch=batch, max_context=128)
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(200 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    current_positions = _positions([33] * batch, mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()

    def run():
        return decoder.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            use_long_rope=False,
        )

    run()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = run()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    samples = []
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        signpost(f"PERF_DECODE_B{batch}")
        for _ in range(20):
            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append(1000 * (time.perf_counter() - start))
        signpost(f"PERF_DECODE_B{batch}_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    assert tuple(output.shape) == (1, 1, batch, config.hidden_size)
    print(
        f"PERF_RESULT impl={implementation} mode=decode batch={batch} context=128 trace_replays=20 "
        f"mean_ms={sum(samples) / len(samples):.6f} min_ms={min(samples):.6f}"
    )
