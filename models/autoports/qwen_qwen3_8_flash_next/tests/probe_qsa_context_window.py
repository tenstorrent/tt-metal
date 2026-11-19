# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic QSA window with exact HF prefix injected during setup only.

This intentionally does NOT establish end-to-end prefill or context acceptance.
CPU mode reconstructs the prefix using the original 32-row HF projection/norm/
RoPE shapes, without evaluating prefix attention. Device mode records selection,
gather, and attention intermediates for an isolated continuation window.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest import mock

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_prefill import StreamingHFQSA


def metrics(expected, actual):
    actual = actual.reshape(expected.shape)
    a, b = expected.double().reshape(-1, expected.shape[-1]), actual.double().reshape(-1, expected.shape[-1])
    ac, bc = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    pcc = (ac * bc).sum(-1) / (ac.square().sum(-1) * bc.square().sum(-1)).sqrt()
    af, bf = a.flatten() - a.mean(), b.flatten() - b.mean()
    whole = (af * bf).sum() / (af.square().sum() * bf.square().sum()).sqrt()
    finite = pcc[torch.isfinite(pcc)]
    return {
        "whole_pcc_fp64": whole.item() if torch.isfinite(whole) else None,
        "minimum_row_pcc_fp64": finite.min().item() if finite.numel() else None,
        "row_pcc_fp64": [v.item() if torch.isfinite(v) else None for v in pcc],
        "invalid_pcc_rows": int((~torch.isfinite(pcc)).sum()),
        "maximum_absolute_error": (a - b).abs().max().item(),
        "expected_row_std": a.std(-1).tolist(),
        "actual_row_std": b.std(-1).tolist(),
        "actual_nonfinite": int((~torch.isfinite(b)).sum()),
        "bf16_equal_fraction": (expected.bfloat16() == actual.bfloat16()).float().mean().item(),
    }


@torch.inference_mode()
def window_projections(model, rope, hidden, start):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    count = hidden.shape[1]
    cos, sin = rope(hidden, torch.arange(start, start + count).reshape(1, -1))
    query, gate = model.q_proj(hidden).reshape(1, count, 24, 512).chunk(2, -1)
    query = apply_rotary_pos_emb(model.q_norm(query).transpose(1, 2), cos=cos, sin=sin)
    key = apply_rotary_pos_emb(
        model.k_norm(model.k_proj(hidden).reshape(1, count, 2, 256)).transpose(1, 2), cos=cos, sin=sin
    )
    value = model.v_proj(hidden).reshape(1, count, 2, 256).transpose(1, 2)
    qk = model.indexer.index_qk_proj(hidden)
    index_query = apply_rotary_pos_emb(
        model.indexer.q_layernorm(qk[..., :512].reshape(1, count, 4, 128)).transpose(1, 2), cos=cos, sin=sin
    )
    return {"query": query, "gate": gate.transpose(1, 2), "key": key, "value": value, "index_query": index_query}


@torch.inference_mode()
def prepare(args, model, rope, provenance):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    key = source["key"]
    assert key["capacity"] == key["length"] == 262144 and key["decode_steps"] == 0 and key["seed"] == 1979
    assert key["reference_sha256"] == provenance["reference_sha256"]
    assert key["config_sha256"] == provenance["config_sha256"]
    assert key["snapshot"] == str(args.snapshot.resolve())
    reference = StreamingHFQSA(model, rope, args.capacity)
    begin_time = time.monotonic()
    for begin in range(0, args.prefix, 32):
        end = min(begin + 32, args.prefix)
        hidden = source["hidden"][:, begin:end]
        count = end - begin
        positions = torch.arange(begin, end).reshape(1, -1)
        cos, sin = rope(hidden, positions)
        group_cos, group_sin = rope(hidden, positions.div(4, rounding_mode="floor") * 4)
        keys = model.k_norm(model.k_proj(hidden).reshape(1, count, 2, 256)).transpose(1, 2)
        reference.k[:, :, begin:end] = apply_rotary_pos_emb(keys, cos=cos, sin=sin)
        reference.v[:, :, begin:end] = model.v_proj(hidden).reshape(1, count, 2, 256).transpose(1, 2)
        raw = model.indexer.index_qk_proj(hidden)[..., 512:]
        history = torch.cat((reference.history, raw), 1)
        pooled = history.float().unfold(1, 4, 1).mean(-1).bfloat16()
        pooled = model.indexer.k_layernorm(pooled)
        pooled = apply_rotary_pos_emb(pooled.unsqueeze(1), cos=group_cos, sin=group_sin).squeeze(1)
        for row, position in enumerate(range(begin, end)):
            if position % 4 == 3:
                reference.compressed[:, position // 4] = pooled[:, row]
        reference.history = history[:, -3:].clone()
        if end % 16384 == 0 or end == args.prefix:
            print(json.dumps({"prefix_projection_rows": end, "seconds": time.monotonic() - begin_time}), flush=True)
    reference.position = args.prefix
    prefix = {
        "keys": reference.k[:, :, : args.prefix].clone(),
        "values": reference.v[:, :, : args.prefix].clone(),
        "compressed": reference.compressed[:, : args.prefix // 4].clone(),
        "raw_history": reference.history.clone(),
    }
    hidden = source["hidden"][:, args.prefix : args.prefix + args.rows].clone()
    saved_expected = source["expected"][:, args.prefix : args.prefix + args.rows].clone()
    continued, selections = reference.forward_chunk(hidden)
    rebuilt_match = metrics(saved_expected, continued)
    assert rebuilt_match["minimum_row_pcc_fp64"] >= 0.99999, rebuilt_match
    fixture = {
        "version": 1,
        "scope": "DIAGNOSTIC ONLY: exact HF prefix setup injection, not prefill acceptance",
        "source_key": key,
        "source_path": str(args.source_cache),
        "provenance": provenance,
        "prefix": args.prefix,
        "capacity": args.capacity,
        "rows": args.rows,
        "prefix_state": prefix,
        "hidden": hidden,
        "saved_expected": saved_expected,
        "continued_expected": continued,
        "selected_ids": selections,
        "projections": window_projections(model, rope, hidden, args.prefix),
        "compressed_after": reference.compressed[:, : (args.prefix + args.rows) // 4].clone(),
        "rebuilt_reference_vs_saved": rebuilt_match,
        "preparation_seconds": time.monotonic() - begin_time,
    }
    temporary = args.fixture.with_name(args.fixture.name + f".tmp-{os.getpid()}")
    torch.save(fixture, temporary)
    os.replace(temporary, args.fixture)
    assert "ttnn" not in sys.modules
    return fixture


@torch.inference_mode()
def device_run(args, config, weights, hf_model, fixture):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import repeat_kv

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa import QSA
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa_chunked import ChunkedQSA

    prefix, count = fixture["prefix"], fixture["rows"]
    capture = {}
    report = {"scope": fixture["scope"], "prefix": prefix, "rows": count, "capacity": args.capacity}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        loader = WeightLoader(weights, mesh)
        model = ChunkedQSA(QSA(loader, config))
        permutation = (
            torch.randperm(args.capacity // 32, generator=torch.Generator().manual_seed(1980)).reshape(1, -1).int()
        )
        if permutation[0, 0] == 0:
            permutation = permutation.roll(1, dims=1)
        state = model.attention.make_state(capacity=args.capacity, page_table=permutation)
        report["page_table_sha256"] = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
        capture["page_table"] = permutation

        def upload(value, destination):
            host = ttnn.from_torch(value.contiguous(), dtype=destination.dtype, layout=destination.layout)
            ttnn.copy_host_to_device_tensor(host, destination)

        for name, source_name in (("k_cache", "keys"), ("v_cache", "values")):
            logical = torch.zeros(1, 2, args.capacity, 256, dtype=torch.bfloat16)
            logical[:, :, :prefix] = fixture["prefix_state"][source_name]
            pages = logical.reshape(2, args.capacity // 32, 32, 256).permute(1, 0, 2, 3).contiguous()
            physical = torch.empty_like(pages)
            physical[permutation.flatten().long()] = pages
            upload(physical, getattr(state, name))
        compressed = torch.zeros(1, 1, args.capacity // 4, 128, dtype=torch.bfloat16)
        compressed[:, :, : prefix // 4] = fixture["prefix_state"]["compressed"].unsqueeze(1)
        upload(compressed, state.compressed_keys)
        upload(fixture["prefix_state"]["raw_history"].unsqueeze(1), state.raw_history)
        upload(torch.tensor([prefix], dtype=torch.int32), state.position)
        if args.tt_prefix_indexer:
            from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk

            source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
            assert source["key"] == fixture["source_key"]
            prefix_hidden = loader.tensor(source["hidden"][:, :prefix].unsqueeze(0))
            raw_history = loader.tensor(torch.zeros(1, 1, 3, 128, dtype=torch.bfloat16))
            offsets = loader.tensor(torch.arange(32).float().reshape(1, 1, 32, 1), ttnn.float32)
            start_time = time.monotonic()
            for start in range(0, prefix, 32):
                raw = model.attention.project_index_qk(prefix_hidden[:, :, start : start + 32])[..., 512:]
                history = ttnn.concat((raw_history, raw), 2)
                history_fp32 = ttnn.typecast(history, ttnn.float32)
                pooled = ttnn.add(history_fp32[:, :, :32], history_fp32[:, :, 1:33])
                pooled = ttnn.add(pooled, history_fp32[:, :, 2:34])
                pooled = ttnn.add(pooled, history_fp32[:, :, 3:35])
                pooled = ttnn.typecast(ttnn.multiply(pooled, 0.25), ttnn.bfloat16)
                positions = ttnn.add(offsets, start)
                group_starts = ttnn.multiply(ttnn.floor(ttnn.multiply(positions, 0.25)), 4)
                pooled = model._rope(
                    model.attention.rms(pooled, model.attention.index_k_norm, model.attention.eps), group_starts
                )
                completed = ttnn.concat([pooled[:, :, row : row + 1] for row in range(3, 32, 4)], 2)
                write_token_chunk(completed, state.compressed_keys, start // 4)
                ttnn.copy(history[:, :, -3:], raw_history)
                if (start + 32) % 16384 == 0 or start + 32 == prefix:
                    ttnn.synchronize_device(mesh)
                    print(
                        json.dumps({"tt_indexer_prefix_rows": start + 32, "seconds": time.monotonic() - start_time}),
                        flush=True,
                    )
            ttnn.copy(raw_history, state.raw_history)
            ttnn.deallocate(prefix_hidden)
            capture["tt_compressed_prefix"] = ttnn.to_torch(state.compressed_keys[:, :, : prefix // 4])
            report["prefix_indexer_mode"] = "TT reconstructed from original hidden states; diagnostic group writes"
            report["prefix_projection_policy"] = type(model.attention.project_index_qk).__name__
            report["scope"] = "DIAGNOSTIC ONLY: HF K/V prefix injection with TT-reconstructed indexer prefix"
            report["prefix_indexer_wall_seconds"] = time.monotonic() - start_time
        else:
            report["prefix_indexer_mode"] = "exact HF setup injection"
        x = loader.tensor(fixture["hidden"].unsqueeze(0))
        original_rope, original_selection, original_gather = model._rope, model._selected_ids, model.gather
        original_attention = ttnn.transformer.scaled_dot_product_attention
        original_linear = model.attention.linear
        original_topk = ttnn.topk

        def rope(value, positions):
            result = original_rope(value, positions)
            heads = value.shape[1]
            name = {24: "query", 2: "key", 4: "index_query", 1: "pooled"}[heads]
            capture[name] = ttnn.to_torch(result)
            capture[name + "_positions"] = ttnn.to_torch(positions)
            return result

        def selection(query, *, state, positions):
            result = original_selection(query, state=state, positions=positions)
            capture["selected_ids"] = ttnn.to_torch(result).long()
            return result

        def gather(*values, **kwargs):
            keys, values = original_gather(*values, **kwargs)
            capture["selected_keys"] = ttnn.to_torch(keys)
            capture["selected_values"] = ttnn.to_torch(values)
            return keys, values

        def attention(*values, **kwargs):
            result = original_attention(*values, **kwargs)
            capture["attention_output"] = ttnn.to_torch(result)
            return result

        def linear(value, weight, compute):
            output = original_linear(value, weight, compute)
            if weight is model.attention.q:
                capture["q_gate_projection"] = ttnn.to_torch(output)
            elif weight is model.attention.v:
                capture["value_projection"] = ttnn.to_torch(output)
            elif weight is model.attention.out:
                capture["gated_attention"] = ttnn.to_torch(value)
            return output

        def topk(scores, *values, **kwargs):
            result = original_topk(scores, *values, **kwargs)
            capture["indexer_scores"] = ttnn.to_torch(scores).reshape(count, -1)
            capture["indexer_topk_blocks"] = ttnn.to_torch(result[1]).reshape(count, -1).long()
            return result

        with (
            mock.patch.object(model, "_rope", rope),
            mock.patch.object(model, "_selected_ids", selection),
            mock.patch.object(model, "gather", gather),
            mock.patch.object(model.attention, "linear", linear),
            mock.patch.object(ttnn.transformer, "scaled_dot_product_attention", attention),
            mock.patch.object(ttnn, "topk", topk),
        ):
            output, ids = model.forward_chunk(x, state=state, return_selected_ids=True)
        capture["actual"] = ttnn.to_torch(output).reshape(fixture["saved_expected"].shape)
        capture["compressed_window"] = ttnn.to_torch(state.compressed_keys[:, :, prefix // 4 : (prefix + count) // 4])
        capture["state_position"] = ttnn.to_torch(state.position)
        torch.save(capture, args.capture)
        report["actual_vs_saved_reference"] = metrics(fixture["saved_expected"], capture["actual"])
        report["rebuilt_reference_vs_saved"] = fixture["rebuilt_reference_vs_saved"]
        report["projection_metrics"] = {
            name: metrics(fixture["projections"][name], capture[name]) for name in ("query", "key", "index_query")
        }
        report["actual_query_positions"] = capture["query_positions"].reshape(-1).tolist()
        keys = torch.cat((fixture["prefix_state"]["keys"], fixture["projections"]["key"]), 2)
        values = torch.cat((fixture["prefix_state"]["values"], fixture["projections"]["value"]), 2)
        actual_inserted_keys = torch.cat((fixture["prefix_state"]["keys"], capture["key"]), 2)
        actual_inserted_values = torch.cat(
            (fixture["prefix_state"]["values"], capture["value_projection"].reshape(1, count, 2, 256).transpose(1, 2)),
            2,
        )
        oracle_selected_outputs, tt_operand_attention = [], []
        rows = []
        for row in range(count):
            ids = capture["selected_ids"][row]
            valid = ids != 0xFFFFFFFF
            selected = ids[valid]
            expected_ids = fixture["selected_ids"][row]
            actual_set, expected_set = set(selected.tolist()), set(expected_ids.tolist())
            lost, extra = sorted(expected_set - actual_set), sorted(actual_set - expected_set)
            rows.append(
                {
                    "position": prefix + row,
                    "intersection": len(actual_set & expected_set),
                    "expected_count": len(expected_set),
                    "actual_count": len(actual_set),
                    "duplicates": len(selected) - len(actual_set),
                    "missing_ids": lost,
                    "extra_ids": extra,
                    "future_ids": selected[selected > prefix + row].tolist(),
                }
            )
            query = fixture["projections"]["query"][:, :, row : row + 1]
            cpu_topk = capture["indexer_scores"][row].topk(512).indices
            native_topk = capture["indexer_topk_blocks"][row]
            rows[-1]["topk_changed_blocks_vs_cpu_same_scores"] = sorted(
                set(cpu_topk.tolist()) ^ set(native_topk.tolist())
            )
            rows[-1]["topk_min_selected_score"] = capture["indexer_scores"][row, native_topk].min().item()
            rows[-1]["cpu_512th_score"] = capture["indexer_scores"][row, cpu_topk].min().item()
            oracle_k, oracle_v = repeat_kv(keys[:, :, selected], 12), repeat_kv(values[:, :, selected], 12)
            probabilities = torch.softmax(
                (query @ oracle_k.transpose(-1, -2)) * (256**-0.5), -1, dtype=torch.float32
            ).bfloat16()
            result = probabilities @ oracle_v
            result *= fixture["projections"]["gate"][:, :, row : row + 1].sigmoid()
            oracle_selected_outputs.append(hf_model.o_proj(result.reshape(1, 1, 6144)))
            actual_q = capture["query"][:, :, row : row + 1]
            actual_k = repeat_kv(capture["selected_keys"][row : row + 1, :, valid], 12)
            actual_v = repeat_kv(capture["selected_values"][row : row + 1, :, valid], 12)
            probability = torch.softmax(
                (actual_q @ actual_k.transpose(-1, -2)) * (256**-0.5), -1, dtype=torch.float32
            ).bfloat16()
            tt_operand_attention.append(probability @ actual_v)
            rows[-1]["gather_key_max_abs_error_against_hf"] = (
                (capture["selected_keys"][row, :, valid].float() - keys[0, :, selected].float()).abs().max().item()
            )
            rows[-1]["gather_value_max_abs_error_against_hf"] = (
                (capture["selected_values"][row, :, valid].float() - values[0, :, selected].float()).abs().max().item()
            )
            rows[-1]["gather_key_max_abs_error_against_inserted"] = (
                (capture["selected_keys"][row, :, valid].float() - actual_inserted_keys[0, :, selected].float())
                .abs()
                .max()
                .item()
            )
            rows[-1]["gather_value_max_abs_error_against_inserted"] = (
                (capture["selected_values"][row, :, valid].float() - actual_inserted_values[0, :, selected].float())
                .abs()
                .max()
                .item()
            )
        capture["hf_attention_on_tt_selected_ids"] = torch.cat(oracle_selected_outputs, 1)
        capture["cpu_attention_on_tt_operands"] = torch.cat(tt_operand_attention, 2)
        report["selections"] = rows
        report["hf_attention_on_tt_selected_ids_vs_saved"] = metrics(
            fixture["saved_expected"], capture["hf_attention_on_tt_selected_ids"]
        )
        actual_attention = capture["attention_output"].permute(2, 1, 0, 3)
        report["device_attention_vs_cpu_on_same_tt_operands"] = metrics(
            capture["cpu_attention_on_tt_operands"], actual_attention
        )
        torch.save(capture, args.capture)
        report["capture_path"] = str(args.capture)
        return report
    finally:
        ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--source-cache", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_cpu_prefill_262144.pt")
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--prefix", type=int, default=222272)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=262144)
    parser.add_argument("--device", action="store_true")
    parser.add_argument(
        "--tt-prefix-indexer",
        action="store_true",
        help="Rebuild prefix compressed keys on TT, skipping prefix attention; diagnostic only",
    )
    args = parser.parse_args()
    assert args.prefix % 32 == 0 and 1 <= args.rows <= 32 and args.prefix + args.rows <= args.capacity == 262144
    torch.set_num_threads(8)
    config, model, rope, weights, provenance = load_reference(args.snapshot)
    if args.fixture.exists():
        fixture = torch.load(args.fixture, weights_only=True, mmap=True, map_location="cpu")
        assert (fixture["prefix"], fixture["rows"], fixture["capacity"]) == (args.prefix, args.rows, args.capacity)
        assert fixture["provenance"] == provenance
    else:
        fixture = prepare(args, model, rope, provenance)
    if args.device:
        if args.capture is None:
            parser.error("--device requires --capture external tensor artifact path")
        report = device_run(args, config, weights, model, fixture)
    else:
        report = {
            name: fixture[name]
            for name in (
                "scope",
                "source_key",
                "provenance",
                "prefix",
                "capacity",
                "rows",
                "rebuilt_reference_vs_saved",
                "preparation_seconds",
            )
        }
        report["fixture_path"] = str(args.fixture)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("selections", "projection_metrics")}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
