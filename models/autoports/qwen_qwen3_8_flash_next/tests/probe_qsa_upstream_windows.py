# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced actual-TT upstream regression at selected long-context windows.

All state starts empty on TT and is populated from original hidden inputs with
the runtime projection/norm/RoPE arithmetic. Skipped queries cannot affect K/V
or indexer state, so earlier attention scores and outputs are not computed.
Complete32-row K/V pages use native paged_fill_cache; eight completed compressed
keys use the model-local write_token_chunk. These diagnostic writes differ
from runtime's per-token updates and do NOT establish full-prefill acceptance.
No HF cache, key, query, score, or selected ID is injected into TT.

At every target window the unchanged runtime forward_chunk computes all32
outputs, selections, and state updates. Readback/captures happen only after
that audited pass. One prefix build serves all requested windows.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_context_window import metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.qsa_state_fixture import runtime_source_hashes


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/ttuser/qwen38-flash"))
    parser.add_argument(
        "--source-cache", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_cpu_prefill_262144.pt")
    )
    parser.add_argument("--windows", default="165856,205504,215424,222272")
    args = parser.parse_args()
    windows = sorted(set(int(value) for value in args.windows.split(",")))
    if not windows or any(start % 32 or not 0 <= start <= 262112 for start in windows):
        parser.error("Windows must be32-row aligned and fit the advertised context")
    torch.set_num_threads(8)
    config, _, _, weights, provenance = load_reference(args.snapshot)
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    source_key = source["key"]
    assert source_key["length"] == source_key["capacity"] == 262144 and source_key["decode_steps"] == 0
    assert source_key["seed"] == 1979 and source_key["snapshot"] == str(args.snapshot.resolve())
    for name in ("reference_sha256", "config_sha256", "transformers"):
        assert source_key[name] == provenance[name]
    oracle_ids = {}
    for start in windows:
        if start + 32 <= 2048:
            oracle_ids[start] = [torch.arange(start + row + 1) for row in range(32)]
        else:
            fixture = torch.load(args.root / f"qsa_score_window_{start}.pt", weights_only=True, mmap=True)
            assert fixture["source_key"] == source_key and fixture["provenance"] == provenance
            assert fixture["prefix"] == start and fixture["rows"] == 32
            oracle_ids[start] = fixture["selected_ids"]
            del fixture
    report = {
        "scope": __doc__,
        "provenance": provenance,
        "source_key": source_key,
        "runtime_source_sha256": runtime_source_hashes(),
        "windows": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa import QSA
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa_chunked import ChunkedQSA

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        loader = WeightLoader(weights, mesh)
        model = ChunkedQSA(QSA(loader, config))
        a = model.attention
        permutation = torch.randperm(8192, generator=torch.Generator().manual_seed(1980)).reshape(1, -1).int()
        if permutation[0, 0] == 0:
            permutation = permutation.roll(1, dims=1)
        report["page_table_sha256"] = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
        state = a.make_state(capacity=262144, page_table=permutation)
        hidden = loader.tensor(source["hidden"][:, : windows[-1] + 32].unsqueeze(0))
        offsets = model.query_offsets

        def populate(x, start):
            """Actual runtime upstream math; diagnostic whole-page cache writes."""
            positions = ttnn.add(offsets, start)
            key = ttnn.permute(ttnn.reshape(a.linear(x, a.k, a.compute), (1, 32, 2, 256)), (0, 2, 1, 3))
            key = model._rope(a.rms(key, a.k_norm, a.eps), positions)
            value = ttnn.permute(ttnn.reshape(a.linear(x, a.v, a.compute), (1, 32, 2, 256)), (0, 2, 1, 3))
            page = state.page_table[:, start // 32 : start // 32 + 1]
            ttnn.experimental.paged_fill_cache(state.k_cache, key, page, batch_idx=0)
            ttnn.experimental.paged_fill_cache(state.v_cache, value, page, batch_idx=0)
            raw = a.project_index_qk(x)[..., 512:]
            history = ttnn.concat((state.raw_history, raw), 2)
            history_fp32 = ttnn.typecast(history, ttnn.float32)
            pooled = ttnn.add(history_fp32[:, :, :32], history_fp32[:, :, 1:33])
            pooled = ttnn.add(pooled, history_fp32[:, :, 2:34])
            pooled = ttnn.add(pooled, history_fp32[:, :, 3:35])
            pooled = ttnn.typecast(ttnn.multiply(pooled, 0.25), ttnn.bfloat16)
            group_starts = ttnn.multiply(ttnn.floor(ttnn.multiply(positions, 0.25)), 4)
            pooled = model._rope(a.rms(pooled, a.index_k_norm, a.eps), group_starts)
            complete = ttnn.concat([pooled[:, :, row : row + 1] for row in range(3, 32, 4)], 2)
            write_token_chunk(complete, state.compressed_keys, start // 4)
            ttnn.copy(history[:, :, -3:], state.raw_history)
            model.advance(state, 32)

        started = time.perf_counter()
        for start in range(0, windows[-1] + 32, 32):
            x = hidden[:, :, start : start + 32]
            if start not in oracle_ids:
                with runtime_audit():
                    populate(x, start)
            else:
                assert int(ttnn.to_torch(state.position).reshape(-1)[0]) == start
                operands = {}
                original_scores = a.indexer_scores

                def capture_scores(query, keys):
                    result = original_scores(query, keys)
                    operands.update(query=query, score=result)
                    return result

                a.indexer_scores = capture_scores
                begin_window = time.perf_counter()
                try:
                    with runtime_audit():
                        output, ids = model.forward_chunk(x, state=state, return_selected_ids=True)
                        model.advance(state, 32)
                finally:
                    a.indexer_scores = original_scores
                ttnn.synchronize_device(mesh)
                window_seconds = time.perf_counter() - begin_window
                actual = ttnn.to_torch(output).reshape(1, 32, 2560)
                selected = ttnn.to_torch(ids).long()
                capture = {
                    "actual": actual,
                    "selected_ids": selected,
                    "index_query": ttnn.to_torch(operands["query"]),
                    "index_scores": ttnn.to_torch(operands["score"]),
                    "compressed_keys": ttnn.to_torch(state.compressed_keys),
                    "raw_history": ttnn.to_torch(state.raw_history),
                    "page_table": permutation,
                    "position": start,
                    "source_key": source_key,
                    "runtime_source_sha256": report["runtime_source_sha256"],
                }
                measured = metrics(source["expected"][:, start : start + 32], actual)
                selection = []
                for row in range(32):
                    valid = selected[row][selected[row] != 0xFFFFFFFF]
                    expected_set, actual_set = set(oracle_ids[start][row].tolist()), set(valid.tolist())
                    selection.append(
                        {
                            "position": start + row,
                            "missing_ids": sorted(expected_set - actual_set),
                            "extra_ids": sorted(actual_set - expected_set),
                            "duplicates": len(valid) - len(actual_set),
                            "future_ids": valid[valid > start + row].tolist(),
                        }
                    )
                capture_path = args.root / f"{args.output.stem}_{start}.pt"
                torch.save(capture, capture_path)
                report["windows"][str(start)] = {
                    "actual_vs_HF": measured,
                    "selection": selection,
                    "window_seconds": window_seconds,
                    "prefix_elapsed_seconds": time.perf_counter() - started,
                    "capture": str(capture_path),
                    "passed": measured["minimum_row_pcc_fp64"] >= 0.995 and measured["actual_nonfinite"] == 0,
                }
                print(json.dumps({"window": start, **report["windows"][str(start)]}), flush=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                ttnn.deallocate(output)
                ttnn.deallocate(ids)
                del capture, operands
            if (start + 32) % 16384 == 0:
                ttnn.synchronize_device(mesh)
                print(json.dumps({"TT_prefix_rows": start + 32, "seconds": time.perf_counter() - started}), flush=True)
        ttnn.synchronize_device(mesh)
        report["final_position"] = int(ttnn.to_torch(state.position).reshape(-1)[0])
        assert report["final_position"] == windows[-1] + 32
        report["runtime_fallback_audit"] = "clean"
        report["passed"] = all(window["passed"] for window in report["windows"].values())
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
