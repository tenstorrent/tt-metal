# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-weight warmed performance harness for the GPT-OSS decoder stage.

This is intentionally a small single-layer harness.  It times non-traced
prefill and traced decode separately and emits Tracy signposts when launched
through ``python -m tracy``.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from tracy import signpost

import ttnn
from models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder import (
    EMITTED_PREFILL_SEQUENCE,
    LAYER_IDX,
    _config,
    _decode_mask,
    _position_tensor,
    _real_state_dict,
    _to_tt,
)
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import FunctionalDecoder


def _decoder_class(name: str):
    if name == "functional":
        return FunctionalDecoder
    if name == "optimized":
        from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import OptimizedDecoder

        return OptimizedDecoder
    raise ValueError(f"unknown decoder {name!r}")


def _time_prefill(decoder, mesh_device, hidden, *, warmups: int, iterations: int) -> float:
    tt_hidden = _to_tt(hidden, mesh_device)
    key_cache, value_cache = decoder.create_kv_cache()
    for _ in range(warmups):
        output = decoder.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache)
        ttnn.synchronize_device(mesh_device)
        output.deallocate(True)

    samples = []
    for index in range(iterations):
        if index == 0:
            signpost("PERF_PREFILL")
        start = time.perf_counter()
        output = decoder.prefill_forward(tt_hidden, key_cache=key_cache, value_cache=value_cache)
        ttnn.synchronize_device(mesh_device)
        samples.append(time.perf_counter() - start)
        if index == 0:
            signpost("PERF_PREFILL_END")
        output.deallocate(True)
    return 1000.0 * sum(samples) / len(samples)


def _time_traced_decode(
    decoder,
    mesh_device,
    hidden,
    *,
    cache_position: int,
    config,
    warmups: int,
    iterations: int,
) -> tuple[float, float]:
    tt_hidden = _to_tt(hidden, mesh_device)
    key_cache, value_cache = decoder.create_kv_cache()
    position_tensor = _position_tensor(cache_position, mesh_device)
    attention_mask = _decode_mask(cache_position, config, decoder.max_cache_len, mesh_device)

    def decode():
        return decoder.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            cache_position_tensor=position_tensor,
            attention_mask=attention_mask,
        )

    compile_output = decode()
    ttnn.synchronize_device(mesh_device)
    compile_output.deallocate(True)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    try:
        for _ in range(warmups):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        samples = []
        for index in range(iterations):
            if index == 0:
                signpost("PERF_DECODE")
            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append(time.perf_counter() - start)
            if index == 0:
                signpost("PERF_DECODE_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    # Read only after the measured window.  This is a harness boundary, not a
    # runtime-path host fallback.
    output = ttnn.to_torch(ttnn.get_device_tensors(trace_output)[0]).float()
    if not torch.isfinite(output).all():
        raise AssertionError("traced decode produced non-finite output")
    samples_ms = [1000.0 * sample for sample in samples]
    return sum(samples_ms) / len(samples_ms), min(samples_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", choices=("functional", "optimized"), required=True)
    parser.add_argument("--candidate", default="default")
    parser.add_argument("--prefill-warmups", type=int, default=2)
    parser.add_argument("--prefill-iterations", type=int, default=5)
    parser.add_argument("--decode-warmups", type=int, default=5)
    parser.add_argument("--decode-iterations", type=int, default=20)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    config = _config()
    state = _real_state_dict()
    decoder_type = _decoder_class(args.decoder)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=128 * 1024 * 1024)
    try:
        kwargs = {}
        if args.decoder == "optimized":
            kwargs["candidate"] = args.candidate
        decoder = decoder_type.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LAYER_IDX,
            mesh_device=mesh_device,
            max_cache_len=288,
            **kwargs,
        )
        generator = torch.Generator().manual_seed(20260725)
        prefill_hidden = torch.randn(
            1,
            EMITTED_PREFILL_SEQUENCE,
            config.hidden_size,
            generator=generator,
            dtype=torch.bfloat16,
        )
        decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator, dtype=torch.bfloat16)

        prefill_ms = _time_prefill(
            decoder,
            mesh_device,
            prefill_hidden,
            warmups=args.prefill_warmups,
            iterations=args.prefill_iterations,
        )
        decode_mean_ms, decode_min_ms = _time_traced_decode(
            decoder,
            mesh_device,
            decode_hidden,
            cache_position=EMITTED_PREFILL_SEQUENCE,
            config=config,
            warmups=args.decode_warmups,
            iterations=args.decode_iterations,
        )
        result = {
            "decoder": args.decoder,
            "candidate": args.candidate,
            "device": str(mesh_device.arch()),
            "mesh": list(mesh_device.shape),
            "prefill_seq_len": EMITTED_PREFILL_SEQUENCE,
            "prefill_warmed_mean_ms": prefill_ms,
            "decode_cache_position": EMITTED_PREFILL_SEQUENCE,
            "decode_traced_warmed_mean_ms": decode_mean_ms,
            "decode_traced_warmed_min_ms": decode_min_ms,
            "prefill_iterations": args.prefill_iterations,
            "decode_iterations": args.decode_iterations,
        }
        print("PERF_RESULT " + json.dumps(result, sort_keys=True))
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as output_file:
                json.dump(result, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
