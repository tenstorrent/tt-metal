#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does the precision policy lose accuracy as the context grows?

Weight quantisation error is length-independent, but a `full_attention` layer attends over
the *whole* paged prefix, so the error in the cached K/V compounds with the number of keys
in the softmax while a `sliding_attention` layer only ever sees its 2048-token window. This
probe separates the two: for each policy and each length it measures the prefill PCC of the
**last** query block, which is the block with the longest prefix and therefore the worst
case, against the host reference.

Writes ``doc/optimized_decoder/pcc/length_dependence.json``.

    python models/autoports/meta_models_muse_glimmer_30b/scripts/probe_length_dependence.py

Environment: ``POLICIES``, ``KINDS``, ``LENGTHS``, ``WEIGHTS`` (``synthetic``/``real``),
``OUT``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.reference import hf_reference as R
from models.autoports.meta_models_muse_glimmer_30b.tests import decoder_test_utils as U
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import OptimizedDecoder

MODEL_DIR = Path(__file__).resolve().parents[1]
WEIGHT_STATS = MODEL_DIR / "doc" / "functional_decoder" / "weight_stats.json"
OUT = Path(os.environ.get("OUT", MODEL_DIR / "doc" / "optimized_decoder" / "pcc" / "length_dependence.json"))
POLICY_NAMES = os.environ.get("POLICIES", "bfp8_all_lofi,bfp8_attn_bfp4_mlp,bfp4_all").split(",")
KINDS = os.environ.get("KINDS", "full_nope,sliding_rope").split(",")
LENGTHS = [int(v) for v in os.environ.get("LENGTHS", "1000,8192,32768,131072").split(",")]
WEIGHTS = os.environ.get("WEIGHTS", "synthetic")
BLOCK_SIZE = 64
Q_CHUNK = 8192


def main() -> int:
    R.configure_host_threads()
    text_config = R.load_text_config()
    kinds = R.layer_kinds(text_config)
    stats = json.loads(WEIGHT_STATS.read_text())
    rows = []

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        for kind_id in KINDS:
            kind = kinds[kind_id]
            if WEIGHTS == "real":
                state_dict = R.load_real_layer_state_dict(kind.layer_idx)
            else:
                state_dict = R.synthetic_layer_state_dict(stats["layers"][str(kind.layer_idx)]["tensors"], seed=7)
            reference = R.ReferenceDecoderLayer(text_config, kind.layer_idx, state_dict)
            decoders = {
                name: OptimizedDecoder.from_state_dict(
                    state_dict,
                    hf_config=text_config,
                    layer_idx=kind.layer_idx,
                    mesh_device=mesh,
                    precision=name,
                    block_size=BLOCK_SIZE,
                )
                for name in POLICY_NAMES
            }
            for length in LENGTHS:
                if WEIGHTS == "real":
                    hidden = R.stacked_layer_input(text_config, kind.layer_idx, R.real_token_ids(length))
                else:
                    hidden = R.unit_rms_hidden_states((1, length, text_config.hidden_size), seed=97)

                # Only the last query block is compared: it has the longest prefix, so it is
                # the worst case for any error that compounds over the cached K/V.
                q_chunk = Q_CHUNK if length >= Q_CHUNK else 32 * max(1, (length // 32) // 4)
                last_start = ((length - 1) // q_chunk) * q_chunk
                golden = {}

                def on_chunk(start, end, out_chunk, _golden=golden, _limit=length):
                    if out_chunk is not None:
                        _golden[start] = (min(end, _limit), out_chunk)

                reference.prefill(
                    hidden,
                    q_chunk=q_chunk,
                    q_chunk_filter=lambda start, end, _s=last_start: start == _s,
                    on_chunk=on_chunk,
                    backend="sdpa",
                )
                end, ref_block = golden[last_start]

                for name, decoder in decoders.items():
                    blocks = decoder.blocks_per_seq(length + 2)
                    page_table = U.build_page_table(batch=1, blocks_per_seq=blocks, total_blocks=blocks, seed=97)
                    kv_cache = decoder.allocate_kv_cache(
                        batch_size=1, max_seq_len=blocks * BLOCK_SIZE, num_blocks=blocks
                    )
                    page_table_tt = U.page_table_to_device(page_table, mesh)
                    hidden_tt = U.prefill_input(hidden, mesh)
                    out = decoder.prefill_forward(hidden_tt, kv_cache=kv_cache, page_table=page_table_tt)
                    device_block = ttnn.slice(out, [0, 0, last_start, 0], [1, 1, end, text_config.hidden_size])
                    pcc = U.pcc(ref_block[:, : end - last_start], U.prefill_output(device_block))
                    device_block.deallocate(True)
                    out.deallocate(True)
                    hidden_tt.deallocate(True)
                    for tensor in (kv_cache[0], kv_cache[1], page_table_tt):
                        tensor.deallocate(True)
                    rows.append(
                        {
                            "weights": WEIGHTS,
                            "kind": kind_id,
                            "policy": name,
                            "seq_len": length,
                            "query_block": [last_start, end],
                            "last_block_prefill_pcc": pcc,
                        }
                    )
                    print(
                        f"LEN {WEIGHTS:9s} {kind_id:12s} {name:22s} len={length:>6d} "
                        f"block=[{last_start},{end}) pcc={pcc:.6f}",
                        flush=True,
                    )
            for decoder in decoders.values():
                del decoder
    finally:
        ttnn.close_mesh_device(mesh)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT.read_text())["records"] if OUT.is_file() else []
    key = lambda r: (r["weights"], r["kind"], r["policy"], r["seq_len"])  # noqa: E731
    merged = {key(r): r for r in existing}
    for row in rows:
        merged[key(row)] = row
    OUT.write_text(json.dumps({"records": sorted(merged.values(), key=key)}, indent=2) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
