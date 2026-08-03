# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed functional-decoder latency and Tracy-signpost harness."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tracy import signpost
from transformers import AutoConfig

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _decode_inputs,
    _page_table,
    _synthetic_state,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import MODEL_ID, FunctionalDecoder


def _prefill(decoder, mesh_device, config, *, sequence, warmups, iterations):
    generator = torch.Generator().manual_seed(17001 + sequence)
    hidden = (torch.randn(1, decoder.batch, sequence, config.hidden_size, generator=generator) * 0.02).to(
        torch.bfloat16
    )
    hidden = _to_tt(hidden, mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = (sequence + decoder.page_size - 1) // decoder.page_size
    table = _to_tt(
        _page_table(decoder.batch, blocks),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    for _ in range(warmups):
        decoder.prefill_forward(hidden, **kwargs)
        ttnn.synchronize_device(mesh_device)
    samples = []
    for index in range(iterations):
        if index == 0:
            signpost("PERF_PREFILL")
        start = time.perf_counter()
        decoder.prefill_forward(hidden, **kwargs)
        ttnn.synchronize_device(mesh_device)
        samples.append(1000 * (time.perf_counter() - start))
        if index == 0:
            signpost("PERF_PREFILL_END")
    return {"mean_ms": sum(samples) / len(samples), "min_ms": min(samples), "samples_ms": samples}


def _decode(decoder, mesh_device, config, *, warmups, iterations):
    generator = torch.Generator().manual_seed(18000 + decoder.batch)
    hidden = (torch.randn(1, decoder.batch, 1, config.hidden_size, generator=generator) * 0.02).to(torch.bfloat16)
    hidden = _to_tt(hidden, mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    table = _to_tt(
        _page_table(decoder.batch, 1),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0] * decoder.batch)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    decoder.decode_forward(hidden, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = decoder.decode_forward(hidden, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
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
            samples.append(1000 * (time.perf_counter() - start))
            if index == 0:
                signpost("PERF_DECODE_END")
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    if not torch.isfinite(ttnn.to_torch(output).float()).all():
        raise AssertionError("traced decode produced non-finite output")
    return {"mean_ms": sum(samples) / len(samples), "min_ms": min(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0, choices=(0, 1, 4))
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION)
    state = _synthetic_state(config, args.layer)
    max_cache_len = args.sequence if args.mode == "prefill" else 32
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = FunctionalDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh_device,
            batch=args.batch,
            max_cache_len=max_cache_len,
        )
        if args.mode == "prefill":
            result = _prefill(
                decoder,
                mesh_device,
                config,
                sequence=args.sequence,
                warmups=args.warmups,
                iterations=args.iterations,
            )
        else:
            result = _decode(
                decoder,
                mesh_device,
                config,
                warmups=args.warmups,
                iterations=args.iterations,
            )
    finally:
        ttnn.close_mesh_device(mesh_device)
    result.update(
        {
            "mode": args.mode,
            "batch": args.batch,
            "sequence": args.sequence,
            "layer": args.layer,
            "model_revision": REAL_REVISION,
        }
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
