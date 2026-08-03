# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Explicit long-context probes kept outside the fast pytest suite."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
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


def _decode_probe(decoder, mesh_device, config, context):
    hidden = torch.zeros(1, decoder.batch, 1, config.hidden_size, dtype=torch.bfloat16)
    hidden_tt = _to_tt(hidden, mesh_device)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = math.ceil(context / decoder.page_size)
    page_table = _to_tt(
        _page_table(decoder.batch, blocks),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    positions = [context - 1] * decoder.batch
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, positions)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = 1000 * (time.perf_counter() - start)
        finite = bool(torch.isfinite(ttnn.to_torch(output).float()).all())
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    if not finite:
        raise AssertionError("long-context traced decode returned non-finite output")
    return {"trace_replay_ms": elapsed_ms, "finite": finite}


def _prefill_probe(decoder, mesh_device, config, sequence, *, warmed):
    hidden = ttnn.zeros(
        (1, decoder.batch, sequence, config.hidden_size),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = math.ceil(sequence / decoder.page_size)
    page_table = _to_tt(
        _page_table(decoder.batch, blocks),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    if warmed:
        decoder.prefill_forward(hidden, **kwargs)
        ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    output = decoder.prefill_forward(hidden, **kwargs)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = 1000 * (time.perf_counter() - start)
    finite = bool(torch.isfinite(ttnn.to_torch(output).float()).all())
    if not finite:
        raise AssertionError("long-context prefill returned non-finite output")
    return {("warmed_ms" if warmed else "single_pass_ms"): elapsed_ms, "finite": finite}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0, choices=(0, 1, 4))
    parser.add_argument("--warmed", action="store_true", help="Run one warmup before the measured prefill pass")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = FunctionalDecoder.from_state_dict(
            _synthetic_state(config, args.layer),
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh_device,
            batch=args.batch,
            max_cache_len=args.context,
        )
        if args.mode == "decode":
            result = _decode_probe(decoder, mesh_device, config, args.context)
        else:
            result = _prefill_probe(decoder, mesh_device, config, args.context, warmed=args.warmed)
    finally:
        ttnn.close_mesh_device(mesh_device)

    result.update(
        {
            "mode": args.mode,
            "batch": args.batch,
            "context": args.context,
            "layer": args.layer,
            "layer_kind": {
                0: "dense_full_forced_rope",
                1: "sliding_rope_moe",
                4: "full_no_rope_moe",
            }[args.layer],
            "model_revision": REAL_REVISION,
            "kv_cache_bytes": args.batch * args.context * 2 * 4 * 128 * 2,
        }
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
