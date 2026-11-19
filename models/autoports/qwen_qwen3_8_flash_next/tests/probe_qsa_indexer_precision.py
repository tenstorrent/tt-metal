# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only indexer precision controls around a long-context cutoff.

Reconstructs original 32-row projection chunks containing every HF candidate
within a score margin of the top-512 cutoff. Hybrid controls retain exact HF
keys outside examined chunks: this is not context acceptance evidence.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_context_window import metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--window-capture", type=Path, required=True)
    parser.add_argument(
        "--source-cache", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_cpu_prefill_262144.pt")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--row", type=int, default=3)
    parser.add_argument("--score-margin", type=float, default=0.1)
    args = parser.parse_args()
    torch.set_num_threads(8)
    fixture = torch.load(args.fixture, weights_only=True, mmap=True, map_location="cpu")
    previous = torch.load(args.window_capture, weights_only=True, mmap=True, map_location="cpu")
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    config, hf, rope, weights, provenance = load_reference(args.snapshot)
    assert fixture["provenance"] == provenance
    assert source["key"] == fixture["source_key"]
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    row = args.row
    position = fixture["prefix"] + row
    complete = (position + 1) // 4
    reference_keys = fixture["compressed_after"][:, :complete]
    queries = {
        "hf": fixture["projections"]["index_query"][0, :, row].float(),
        "tt_window": previous["index_query"][0, :, row].float(),
    }
    reference_scores = (queries["hf"] @ reference_keys[0].float().T).relu().sum(0) / math.sqrt(128)
    cutoff = reference_scores.topk(512).values[-1]
    candidates = torch.where((reference_scores - cutoff).abs() <= args.score_margin)[0]
    candidates = torch.unique(torch.cat((candidates, torch.tensor([position // 4]))))
    chunks = (candidates.div(8, rounding_mode="floor") * 32).unique().tolist()
    print(
        json.dumps({"candidate_blocks": len(candidates), "projection_chunks": len(chunks), "cutoff": cutoff.item()}),
        flush=True,
    )
    report = {
        "scope": "Diagnostic only: sampled prefix indexer reconstruction and exact-HF injection controls",
        "position": position,
        "score_margin": args.score_margin,
        "candidate_blocks": candidates.tolist(),
        "projection_chunk_starts": chunks,
        "unchanged_keys_outside_sample": "exact HF; no full TT prefix claim",
        "provenance": provenance,
    }
    capture = {"reference_scores": reference_scores, "chunk_starts": chunks, "variants": {}}
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa import QSA
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa_chunked import ChunkedQSA

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        loader = WeightLoader(weights, mesh)
        model = ChunkedQSA(QSA(loader, config))
        a = model.attention
        policies = ("native", "exact_raw", "exact_pool", "exact_norm", "precise_rope")
        hybrid = {name: reference_keys.clone() for name in policies}
        captured_stages = {
            name: []
            for name in (
                "raw",
                "pool",
                "norm",
                "rope",
                "exact_raw_rope",
                "exact_pool_rope",
                "exact_norm_rope",
                "precise_rope",
            )
        }
        expected_stages = {name: [] for name in ("raw", "pool", "norm", "rope")}

        def precise_rope(value, positions):
            count = value.shape[-2]
            indices = ttnn.reshape(
                ttnn.to_layout(ttnn.typecast(positions, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, count)
            )
            cos = ttnn.reshape(
                ttnn.to_layout(ttnn.embedding(indices, a.cos, layout=ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT),
                (1, 1, count, 64),
            )
            sin = ttnn.reshape(
                ttnn.to_layout(ttnn.embedding(indices, a.sin, layout=ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT),
                (1, 1, count, 64),
            )
            rotated = ttnn.concat((ttnn.neg(value[..., 32:64]), value[..., :32]), -1)
            first = ttnn.typecast(
                ttnn.multiply(ttnn.typecast(value[..., :64], ttnn.float32), ttnn.typecast(cos, ttnn.float32)),
                ttnn.bfloat16,
            )
            second = ttnn.typecast(
                ttnn.multiply(ttnn.typecast(rotated, ttnn.float32), ttnn.typecast(sin, ttnn.float32)), ttnn.bfloat16
            )
            result = ttnn.typecast(
                ttnn.add(ttnn.typecast(first, ttnn.float32), ttnn.typecast(second, ttnn.float32)), ttnn.bfloat16
            )
            return ttnn.concat((result, value[..., 64:]), -1)

        for number, start in enumerate(chunks):
            hidden = source["hidden"][:, start : start + 32]
            raw = hf.indexer.index_qk_proj(hidden)[..., 512:]
            previous_raw = (
                hf.indexer.index_qk_proj(source["hidden"][:, start - 3 : start])[..., 512:]
                if start
                else torch.zeros(1, 3, 128, dtype=torch.bfloat16)
            )
            history = torch.cat((previous_raw, raw), 1)
            pool = history.float().unfold(1, 4, 1).mean(-1).bfloat16()
            normalized = hf.indexer.k_layernorm(pool)
            positions = torch.arange(start, start + 32).div(4, rounding_mode="floor").reshape(1, -1) * 4
            cos, sin = rope(hidden, positions)
            rotated = apply_rotary_pos_emb(normalized.unsqueeze(1), cos=cos, sin=sin)
            x = loader.tensor(hidden.unsqueeze(0))
            raw_tt = a.linear(x, a.index_qk, a.compute)[..., 512:]
            pos = loader.tensor(positions.reshape(1, 1, 32, 1).float(), ttnn.float32)
            previous_tt = loader.tensor(previous_raw.unsqueeze(0))
            exact = {
                "raw": loader.tensor(raw.unsqueeze(0)),
                "pool": loader.tensor(pool.unsqueeze(0)),
                "norm": loader.tensor(normalized.unsqueeze(0)),
            }
            for policy in policies:
                input_raw = exact["raw"] if policy == "exact_raw" else raw_tt
                history_tt = ttnn.typecast(ttnn.concat((previous_tt, input_raw), 2), ttnn.float32)
                pooled = ttnn.add(history_tt[:, :, :32], history_tt[:, :, 1:33])
                pooled = ttnn.add(pooled, history_tt[:, :, 2:34])
                pooled = ttnn.add(pooled, history_tt[:, :, 3:35])
                pooled = ttnn.typecast(ttnn.multiply(pooled, 0.25), ttnn.bfloat16)
                if policy == "exact_pool":
                    pooled = exact["pool"]
                normed = a.rms(pooled, a.index_k_norm, a.eps)
                if policy == "exact_norm":
                    normed = exact["norm"]
                result = precise_rope(normed, pos) if policy == "precise_rope" else model._rope(normed, pos)
                output = ttnn.to_torch(result)
                if policy == "native":
                    captured_stages["raw"].append(ttnn.to_torch(raw_tt))
                    captured_stages["pool"].append(ttnn.to_torch(pooled))
                    captured_stages["norm"].append(ttnn.to_torch(normed))
                    captured_stages["rope"].append(output)
                else:
                    captured_stages[policy + "_rope" if policy != "precise_rope" else policy].append(output)
                first_block, block_end = start // 4, min(start // 4 + 8, complete)
                hybrid[policy][:, first_block:block_end] = output[:, 0, 3::4][:, : block_end - first_block]
            for name, value in (
                ("raw", raw.unsqueeze(0)),
                ("pool", pool.unsqueeze(0)),
                ("norm", normalized.unsqueeze(0)),
                ("rope", rotated),
            ):
                expected_stages[name].append(value)
            if number % 32 == 0 or number == len(chunks) - 1:
                print(json.dumps({"completed_chunks": number + 1, "total_chunks": len(chunks)}), flush=True)
        report["stage_metrics"] = {}
        for name, values in captured_stages.items():
            target = name if name in expected_stages else "rope"
            expected = torch.cat(expected_stages[target], 2)
            actual = torch.cat(values, 2)
            report["stage_metrics"][name] = metrics(expected, actual)
            capture[name] = actual
        report["hybrid_selections"] = {}
        for policy, keys in hybrid.items():
            capture["variants"][policy] = keys
            report["hybrid_selections"][policy] = {}
            for query_name, query in queries.items():
                scores = (query @ keys[0].float().T).relu().sum(0) / math.sqrt(128)
                top = scores.topk(513)
                self_score = scores[position // 4]
                report["hybrid_selections"][policy][query_name] = {
                    "self_score": self_score.item(),
                    "strictly_higher_blocks": int((scores > self_score).sum()),
                    "self_selected": bool((top.indices[:512] == position // 4).any()),
                    "score_512": top.values[511].item(),
                    "score_513": top.values[512].item(),
                    "changed_top512_blocks": sorted(
                        set(top.indices[:512].tolist()) ^ set(reference_scores.topk(512).indices.tolist())
                    ),
                }
        torch.save(capture, args.capture)
        report["capture_path"] = str(args.capture)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["hybrid_selections"], indent=2), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
