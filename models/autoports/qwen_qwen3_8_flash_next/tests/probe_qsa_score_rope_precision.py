# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact-input indexer RoPE/dot/head-sum controls; diagnostic only.

All injections happen at setup. Test-local patches affect indexer RoPE only
and the four-head score reduction; model runtime source is not changed.
"""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_projection_precision import compact_metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/ttuser/qwen38-flash"))
    parser.add_argument("--window-capture", type=Path, help="Defaults to the original full-TT-indexer capture")
    args = parser.parse_args()
    torch.set_num_threads(8)
    window_path = args.window_capture or args.root / "qsa_context_tt_index_prefix_222272.pt"
    window = torch.load(window_path, weights_only=True, mmap=True)
    fixture = torch.load(args.root / "qsa_context_window_222272.pt", weights_only=True, mmap=True)
    projected = torch.load(args.root / "qsa_projection_sfpu.pt", weights_only=True, mmap=True)
    config, hf, rope, weights, provenance = load_reference(args.snapshot)
    assert provenance == fixture["provenance"]
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    keys = torch.zeros(1, 1, 65536, 128, dtype=torch.bfloat16)
    keys[:, :, :55568] = window["tt_compressed_prefix"]
    keys[:, :, 55568:55576] = window["compressed_window"]
    query = window["index_query"]
    cpu_dot = query.float() @ keys.float().transpose(-1, -2)
    cpu_score = cpu_dot.relu().sum(1, keepdim=True) / math.sqrt(128)
    positions = torch.arange(222272, 222304).reshape(1, 1, 32, 1).float()
    qk = projected["sfpu_tree"]
    query_norm = hf.indexer.q_layernorm(qk[..., :512].reshape(1, 32, 4, 128)).transpose(1, 2)
    raw = torch.cat((fixture["prefix_state"]["raw_history"], qk[..., 512:]), 1)
    key_norm = hf.indexer.k_layernorm(raw.float().unfold(1, 4, 1).mean(-1).bfloat16()).unsqueeze(1)
    report = {
        "scope": "Exact-input diagnostic controls, not decoder acceptance",
        "provenance": provenance,
        "window_capture": str(window_path),
        "rope": {},
        "scores": {},
    }
    capture = {"query": query, "keys": keys, "cpu_dot": cpu_dot, "cpu_score": cpu_score}

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa import QSA
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa_chunked import ChunkedQSA

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        loader = WeightLoader(weights, mesh)
        model = ChunkedQSA(QSA(loader, config))
        original_rope = ChunkedQSA._rope

        def precise_indexer_rope(self, value, pos):
            if value.shape[-1] != 128:
                return original_rope(self, value, pos)
            count = value.shape[-2]
            indices = ttnn.reshape(ttnn.to_layout(ttnn.typecast(pos, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, count))
            cos = ttnn.reshape(
                ttnn.to_layout(
                    ttnn.embedding(indices, self.attention.cos, layout=ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT
                ),
                (1, 1, count, 64),
            )
            sin = ttnn.reshape(
                ttnn.to_layout(
                    ttnn.embedding(indices, self.attention.sin, layout=ttnn.ROW_MAJOR_LAYOUT), ttnn.TILE_LAYOUT
                ),
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

        for name, normed, pos in (
            ("query", query_norm, positions),
            ("key", key_norm, positions.div(4, rounding_mode="floor") * 4),
        ):
            cos, sin = rope(fixture["hidden"], pos.reshape(1, 32).long())
            expected = apply_rotary_pos_emb(normed, cos=cos, sin=sin)
            value, pos_tt = loader.tensor(normed), loader.tensor(pos, ttnn.float32)
            native = ttnn.to_torch(model._rope(value, pos_tt))
            with mock.patch.object(ChunkedQSA, "_rope", precise_indexer_rope):
                precise = ttnn.to_torch(model._rope(value, pos_tt))
            report["rope"][name] = {
                "native": compact_metrics(expected, native),
                "precise": compact_metrics(expected, precise),
            }
            capture["rope_" + name] = {"expected": expected, "native": native, "precise": precise}

        query_tt, pos_tt = loader.tensor(query), loader.tensor(positions, ttnn.float32)
        exact_dot = loader.tensor(cpu_dot, ttnn.float32)
        state = SimpleNamespace(
            capacity=262144,
            compressed_keys=loader.tensor(keys),
            block_numbers=loader.tensor(torch.arange(65536).float().reshape(1, 1, 1, -1), ttnn.float32),
        )
        original_matmul, original_sum, original_topk = ttnn.matmul, ttnn.sum, ttnn.topk
        for policy in ("native", "pairwise_heads", "exact_dot_native_heads", "exact_dot_pairwise_heads"):
            seen = {}

            def matmul(left, right, **kwargs):
                result = exact_dot if policy.startswith("exact_dot") else original_matmul(left, right, **kwargs)
                seen["dot"] = result
                return result

            def sum_heads(value, *pargs, **kwargs):
                if "pairwise" in policy and value.shape[1] == 4 and kwargs.get("dim") == 1:
                    return ttnn.add(ttnn.add(value[:, :1], value[:, 1:2]), ttnn.add(value[:, 2:3], value[:, 3:4]))
                return original_sum(value, *pargs, **kwargs)

            def topk(value, *pargs, **kwargs):
                seen["scores"] = value
                return original_topk(value, *pargs, **kwargs)

            with (
                mock.patch.object(ttnn, "matmul", matmul),
                mock.patch.object(ttnn, "sum", sum_heads),
                mock.patch.object(ttnn, "topk", topk),
            ):
                ids = model._selected_ids(query_tt, state=state, positions=pos_tt)
            dot = ttnn.to_torch(seen["dot"])
            scores = ttnn.to_torch(seen["scores"])
            same_dot_cpu_score = dot.relu().sum(1, keepdim=True) / math.sqrt(128)
            rows = []
            for row in range(32):
                complete = (222272 + row + 1) // 4
                actual = scores[0, 0, row, :complete]
                expected = cpu_score[0, 0, row, :complete]
                same_dot = same_dot_cpu_score[0, 0, row, :complete]
                rows.append(
                    {
                        "position": 222272 + row,
                        "vs_exact_operand_scores": compact_metrics(expected[None], actual[None]),
                        "vs_CPU_sum_of_same_TT_dots": compact_metrics(same_dot[None], actual[None]),
                        "changed_top512_blocks": sorted(
                            set(expected.topk(512).indices.tolist()) ^ set(actual.topk(512).indices.tolist())
                        ),
                    }
                )
            report["scores"][policy] = {"dot_vs_CPU": compact_metrics(cpu_dot, dot), "rows": rows}
            capture[policy] = {"dot": dot, "scores": scores, "selected_ids": ttnn.to_torch(ids)}
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            torch.save(capture, args.capture)
            print(
                json.dumps(
                    {
                        "policy": policy,
                        "dot_max_error": report["scores"][policy]["dot_vs_CPU"]["maximum_absolute_error"],
                        "same_dot_score_max_error": max(
                            item["vs_CPU_sum_of_same_TT_dots"]["maximum_absolute_error"] for item in rows
                        ),
                    }
                ),
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
