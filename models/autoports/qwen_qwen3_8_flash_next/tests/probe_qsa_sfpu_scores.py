# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic exact-input native/SFPU QSA score dots with bounded scratch.

Defaults to exact HF index queries and compressed keys for the known 32-row
window. Every context key remains eligible. SFPU mode processes one head and
one static key block at a time; it never changes selection policy or scores.
This is an optional test candidate, not the model runtime implementation.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_projection_precision import compact_metrics


@torch.inference_mode()
def attention_output_control(fixture, scores, model):
    """CPU HF attention with policy-selected IDs; not a device output gate."""
    from transformers.models.qwen4_exp.modeling_qwen4_exp import repeat_kv

    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_context_window import metrics

    keys = torch.cat((fixture["prefix_state"]["keys"], fixture["projections"]["key"]), 2)
    values = torch.cat((fixture["prefix_state"]["values"], fixture["projections"]["value"]), 2)
    outputs, changes, selected_ids = [], [], []
    for row in range(32):
        position = fixture["prefix"] + row
        complete = (position + 1) // 4
        blocks = scores[0, 0, row, :complete].topk(512).indices
        selected = (blocks[:, None] * 4 + torch.arange(4)[None]).flatten()
        selected = torch.cat((selected, torch.arange(complete * 4, position + 1)))
        selected_ids.append(selected)
        changes.append(len(set(selected.tolist()) ^ set(fixture["selected_ids"][row].tolist())))
        query = fixture["projections"]["query"][:, :, row : row + 1]
        k, v = repeat_kv(keys[:, :, selected], 12), repeat_kv(values[:, :, selected], 12)
        probabilities = torch.softmax((query @ k.transpose(-1, -2)) * (256**-0.5), -1, dtype=torch.float32).bfloat16()
        result = probabilities @ v
        result *= fixture["projections"]["gate"][:, :, row : row + 1].sigmoid()
        outputs.append(model.o_proj(result.reshape(1, 1, 6144)))
    actual = torch.cat(outputs, 1)
    measured = metrics(fixture["saved_expected"], actual)
    return {
        "output_accuracy": {key: value for key, value in measured.items() if not isinstance(value, list)},
        "per_row_pcc": measured["row_pcc_fp64"],
        "changed_token_set_sizes": changes,
        "passed_window_row_gate": min(measured["row_pcc_fp64"]) >= 0.995,
    }, {"output": actual, "selected_ids": selected_ids}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_context_window_222272.pt"))
    parser.add_argument("--window-capture", type=Path, help="Use captured TT queries/keys instead of exact HF operands")
    parser.add_argument("--policies", default="native,sfpu")
    parser.add_argument("--key-block", type=int, default=8192)
    parser.add_argument("--snapshot", type=Path, help="Also evaluate CPU HF attention on policy-selected IDs")
    args = parser.parse_args()
    if args.key_block < 32 or 65536 % args.key_block or args.key_block % 32:
        parser.error("key-block must be tile-aligned and divide 65536")
    policies = args.policies.split(",")
    assert set(policies) <= {"native", "sfpu"}
    torch.set_num_threads(8)
    fixture = torch.load(args.fixture, weights_only=True, mmap=True)
    prefix = fixture["prefix"]
    assert prefix % 32 == 0 and prefix >= 2048 and prefix + 32 <= 262144
    keys = torch.zeros(1, 1, 65536, 128, dtype=torch.bfloat16)
    if args.window_capture:
        window = torch.load(args.window_capture, weights_only=True, mmap=True)
        keys[:, :, : prefix // 4] = window["tt_compressed_prefix"]
        keys[:, :, prefix // 4 : (prefix + 32) // 4] = window["compressed_window"]
        query = window["index_query"]
    else:
        keys[:, :, : (prefix + 32) // 4] = fixture["compressed_after"].unsqueeze(1)
        query = fixture["projections"]["index_query"]
    assert query.shape == (1, 4, 32, 128) and fixture["rows"] == 32
    cpu_dot = query.float() @ keys.float().transpose(-1, -2)
    cpu_score = cpu_dot.relu().sum(1, keepdim=True) / math.sqrt(128)
    report = {
        "scope": "Exact-input score arithmetic diagnostic, not context acceptance",
        "operands": str(args.window_capture) if args.window_capture else "Exact HF query and compressed keys",
        "fixture": str(args.fixture),
        "prefix": prefix,
        "provenance": fixture["provenance"],
        "key_block": args.key_block,
        "largest_product_tensor_bytes": 32 * args.key_block * 128 * 4,
        "policies": {},
    }
    capture = {"query": query, "keys": keys, "cpu_dot": cpu_dot, "cpu_score": cpu_score}
    hf_model = None
    if args.snapshot:
        from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference

        _, hf_model, _, _, provenance = load_reference(args.snapshot)
        assert provenance == fixture["provenance"]
        report["cpu_attention_control"], capture["cpu_attention_control"] = attention_output_control(
            fixture, cpu_score, hf_model
        )
        assert report["cpu_attention_control"]["passed_window_row_gate"]

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        loader = WeightLoader({}, mesh)
        query_tt, keys_tt = loader.tensor(query, ttnn.float32), loader.tensor(keys, ttnn.float32)

        def project(policy):
            if policy == "native":
                return ttnn.matmul(
                    query_tt,
                    ttnn.transpose(keys_tt, -1, -2),
                    compute_kernel_config=loader.compute,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            heads = []
            for head in range(4):
                q = ttnn.reshape(query_tt[:, head : head + 1], (1, 32, 1, 128))
                blocks = []
                for start in range(0, 65536, args.key_block):
                    products = ttnn.multiply(q, keys_tt[:, :, start : start + args.key_block])
                    width = 128
                    while width > 1:
                        half = width // 2
                        products = ttnn.add(products[..., :half], products[..., half:width])
                        width = half
                    blocks.append(ttnn.reshape(products, (1, 1, 32, args.key_block)))
                heads.append(ttnn.concat(blocks, dim=-1))
            return ttnn.concat(heads, dim=1)

        for policy in policies:
            start = time.perf_counter()
            with runtime_audit():
                dot_tt = project(policy)
            ttnn.synchronize_device(mesh)
            initial_seconds = time.perf_counter() - start
            ttnn.deallocate(dot_tt)
            start = time.perf_counter()
            with runtime_audit():
                dot_tt = project(policy)
                score_tt = ttnn.multiply(ttnn.sum(ttnn.relu(dot_tt), dim=1, keepdim=True), 128**-0.5)
            ttnn.synchronize_device(mesh)
            warmed_seconds = time.perf_counter() - start
            dot, score = ttnn.to_torch(dot_tt), ttnn.to_torch(score_tt)
            item = {
                "dot_accuracy": compact_metrics(cpu_dot, dot),
                "initial_dot_seconds": initial_seconds,
                "warmed_dot_and_scores_seconds": warmed_seconds,
                "runtime_audit": "clean",
                "rows": [],
            }
            for row in range(32):
                position = prefix + row
                complete = (position + 1) // 4
                expected, actual = cpu_score[0, 0, row, :complete], score[0, 0, row, :complete]
                actual_ids = actual.topk(512).indices
                r = {
                    "position": position,
                    "score_accuracy": compact_metrics(expected[None], actual[None]),
                    "changed_top512_blocks": sorted(
                        set(expected.topk(512).indices.tolist()) ^ set(actual_ids.tolist())
                    ),
                }
                if position % 4 == 3:
                    block = position // 4
                    r.update(
                        self_selected=bool((actual_ids == block).any()),
                        self_score=float(actual[block]),
                        self_rank=int((actual > actual[block]).sum()),
                        cpu_self_score=float(expected[block]),
                        cpu_self_rank=int((expected > expected[block]).sum()),
                    )
                item["rows"].append(r)
            report["policies"][policy] = item
            capture[policy] = {"dot": dot, "score": score}
            if hf_model is not None:
                (
                    item["attention_output_control"],
                    capture[policy]["attention_output_control"],
                ) = attention_output_control(fixture, score, hf_model)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            torch.save(capture, args.capture)
            print(
                json.dumps(
                    {
                        "policy": policy,
                        "warmed_seconds": warmed_seconds,
                        "dot_max_error": item["dot_accuracy"]["maximum_absolute_error"],
                        "prefix": prefix,
                        "minimum_attention_row_pcc": (
                            item["attention_output_control"]["output_accuracy"]["minimum_row_pcc_fp64"]
                            if hf_model is not None
                            else None
                        ),
                        "changed_topk_rows": [r["position"] for r in item["rows"] if r["changed_top512_blocks"]],
                    }
                ),
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
