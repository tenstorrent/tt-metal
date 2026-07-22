# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reproducible Stage 02 MoE topology candidate benchmark.

This is evidence tooling, not a pytest test or a shipped runtime entry point.
It keeps the exact sparse and split/wide candidates available for reruns.
"""

from __future__ import annotations

import argparse
import gc
import time

import torch

import models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder as functional_test
import ttnn
from models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder import (
    _assert_pcc,
    _config,
    _empty_caches,
    _hf_layer,
    _real_state,
    _reference_layer,
    _synthetic_state,
    _to_host,
    _tt_tensor,
)
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import EMITTED_PREFILL_SEQUENCE, REPRESENTATIVE_LAYER
from models.autoports.openai_gpt_oss_20b.tt.fused_decoder import FusedDecoder


def _mean_ms(samples):
    return sum(samples) / len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=("auto", "wide", "split", "sparse", "sparse_split"),
        required=True,
    )
    parser.add_argument("--decode-policy", choices=("auto", "wide", "split", "sparse", "sparse_split"))
    parser.add_argument("--real-layer", type=int, choices=(12, 13))
    parser.add_argument("--prefill-iterations", type=int, default=5)
    parser.add_argument("--decode-iterations", type=int, default=100)
    args = parser.parse_args()

    previous_layer = functional_test.LAYER_IDX
    layer_idx = args.real_layer if args.real_layer is not None else REPRESENTATIVE_LAYER
    functional_test.LAYER_IDX = layer_idx
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=128_000_000)
    try:
        config = _config()
        state = _real_state() if args.real_layer is not None else _synthetic_state(config)
        decoder = FusedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer_idx,
            mesh_device=mesh,
        )
        decoder.moe_policy = args.policy
        reference_layer = _hf_layer(state, config)
        seed = {12: 9090, 13: 9249}.get(args.real_layer, 7711)
        generator = torch.Generator().manual_seed(seed)
        prefill_host = torch.randn(
            (1, 1, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16 if args.real_layer is None else torch.float32,
        ).to(torch.bfloat16)
        decode_host = torch.randn(
            (1, 1, 1, config.hidden_size),
            generator=generator,
            dtype=torch.bfloat16 if args.real_layer is None else torch.float32,
        ).to(torch.bfloat16)
        reference_prefill, _, _, reference_cache = _reference_layer(reference_layer, prefill_host, config)
        reference_decode, _, _, _ = _reference_layer(
            reference_layer,
            decode_host,
            config,
            start_pos=EMITTED_PREFILL_SEQUENCE,
            cache=reference_cache,
        )
        prefill_input = _tt_tensor(prefill_host, mesh)
        decode_input = _tt_tensor(decode_host, mesh)
        key_cache, value_cache = _empty_caches(config, mesh)

        output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
        ttnn.synchronize_device(mesh)
        _assert_pcc(reference_prefill, _to_host(output), 0.99, f"{args.policy} prefill")
        output.deallocate(True)
        prefill_ms = []
        for _ in range(args.prefill_iterations):
            start = time.perf_counter()
            output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
            ttnn.synchronize_device(mesh)
            prefill_ms.append((time.perf_counter() - start) * 1000)
            output.deallocate(True)

        decode_policy = args.decode_policy or args.policy
        decoder.moe_policy = decode_policy
        output = decoder.decode_forward(
            decode_input,
            key_cache,
            value_cache,
            current_pos=EMITTED_PREFILL_SEQUENCE,
        )
        ttnn.synchronize_device(mesh)
        _assert_pcc(reference_decode, _to_host(output), 0.99, f"{decode_policy} decode")
        output.deallocate(True)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_output = decoder.decode_forward(
            decode_input,
            key_cache,
            value_cache,
            current_pos=EMITTED_PREFILL_SEQUENCE,
        )
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        decode_ms = []
        for _ in range(args.decode_iterations):
            start = time.perf_counter()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            decode_ms.append((time.perf_counter() - start) * 1000)
        print(
            "CANDIDATE_RESULT "
            f"prefill_policy={args.policy} decode_policy={decode_policy} layer={layer_idx} "
            f"prefill_mean_ms={_mean_ms(prefill_ms):.9f} "
            f"prefill_min_ms={min(prefill_ms):.9f} "
            f"decode_traced_mean_ms={_mean_ms(decode_ms):.9f} "
            f"decode_traced_min_ms={min(decode_ms):.9f} "
            f"prefill_iterations={len(prefill_ms)} decode_iterations={len(decode_ms)}"
        )
        ttnn.release_trace(mesh, trace_id)
        trace_output.deallocate(True)
        del decoder, reference_layer, state
        gc.collect()
    finally:
        ttnn.close_mesh_device(mesh)
        functional_test.LAYER_IDX = previous_layer


if __name__ == "__main__":
    main()
