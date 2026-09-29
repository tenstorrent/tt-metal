# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recorded-activation batched candidate checks, including trace/KV refresh."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.tsu_batch_candidate import install_shared_batch
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32, choices=(2, 3, 8, 32))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=(
            "shared",
            "attention",
            "attention_rowsdpa",
            "norms",
            "runtime",
            "experts",
            "experts_vector_mix",
            "experts_index_union",
        ),
        default="shared",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    report = {"batch": args.batch, "rows": [], "fixtures": {}, "candidate": args.candidate, "passed": False}
    report["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            Path(__file__).resolve(),
            Path(__file__).with_name("tsu_batch_candidate.py"),
            *sorted((root / "tt").glob("*.py")),
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1000000000)
    gen = None
    try:
        gen = build_generator(None, mesh, max_seq_len=262144, layer_indices=(0, 5))
        # The control must remain serialized even after the production default
        # enables batching; otherwise runtime comparisons test the same path.
        for decoder in gen.model.layers:
            decoder.batched_shared_decode = False
        report["baseline_batched_shared_decode"] = False
        for layer_number, decoder in enumerate(gen.model.layers):
            index = gen.model.layer_indices[layer_number]
            path = root / f"doc/optimized_decoder/actual_text_layer{index}_4096_128.pt"
            report["fixtures"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            fixture = torch.load(path, weights_only=True)["prefill"]
            lengths = [(31, 63, 127)[slot % 3] for slot in range(args.batch)]
            token_values = torch.cat(
                [fixture[:, slot * 64 + length : slot * 64 + length + 1] for slot, length in enumerate(lengths)],
                dim=1,
            ).unsqueeze(0)
            expected = []
            original = decoder.decode_forward
            for candidate in (False, True):
                caches, table = gen.model.allocate_cache(slots=32, context=256)
                cache = caches[layer_number]
                torch.manual_seed(915)
                table = torch.randperm(table.numel(), dtype=torch.int32).reshape(table.shape)[: args.batch]
                page_table = gen.model.upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                for slot, length in enumerate(lengths):
                    hidden = gen.model.upload(fixture[:, slot * 64 : slot * 64 + length].unsqueeze(0))
                    result = decoder.prefill_forward(
                        hidden,
                        rope_mats=gen.model.rope_prefill[gen.model.config.layer_types[index]],
                        page_table=page_table,
                        kv_cache=cache,
                        user_id=slot,
                    )
                    del hidden, result
                if candidate:
                    if args.candidate == "runtime":
                        decoder.batched_shared_decode = True
                    else:
                        install_shared_batch(
                            SimpleNamespace(layers=[decoder]),
                            batch_attention=args.candidate.startswith("attention"),
                            batch_norms=args.candidate == "norms",
                            row_sdpa=args.candidate == "attention_rowsdpa",
                            batch_experts=args.candidate.startswith("experts"),
                            vector_mix=args.candidate in ("experts_vector_mix", "experts_index_union"),
                            indexed_union=args.candidate == "experts_index_union",
                        )
                tokens = gen.model.upload(token_values)
                positions = gen.model.upload(torch.tensor([lengths]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                cache_positions = gen.model.upload(torch.tensor(lengths), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

                def forward():
                    return decoder.decode_forward(
                        tokens,
                        rope_mats=gen.model.rope_decode[gen.model.config.layer_types[index]],
                        current_pos=positions,
                        cache_pos=cache_positions,
                        page_table=page_table,
                        kv_cache=cache,
                    )

                warmed = forward()
                del warmed
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                output = forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                try:
                    for step in range(3):
                        values = token_values if step == 0 else token_values.flip(2) * (1 + step / 100)
                        gen._copy(values, tokens, "test_input_refreshes")
                        gen._copy(torch.tensor([lengths]) + step, positions, "test_position_refreshes")
                        gen._copy(torch.tensor(lengths) + step, cache_positions, "test_cache_position_refreshes")
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        ranks = [ttnn.to_torch(t).float().clone() for t in ttnn.get_device_tensors(output)]
                        assert all(torch.equal(ranks[0], rank) for rank in ranks[1:])
                        hashes = [
                            hashlib.sha256(ttnn.to_torch(rank).float().numpy().tobytes()).hexdigest()
                            for tensor in cache
                            for rank in ttnn.get_device_tensors(tensor)
                        ]
                        if not candidate:
                            expected.append((ranks[0], hashes))
                            continue
                        reference, expected_hashes = expected[step]
                        row = {
                            "layer": index,
                            "step": step,
                            "exact_output": torch.equal(reference, ranks[0]),
                            "pcc": float(torch.corrcoef(torch.stack((reference.flatten(), ranks[0].flatten())))[0, 1]),
                            "max_abs": float((reference - ranks[0]).abs().max()),
                            "all_rank_cache_exact": hashes == expected_hashes,
                        }
                        report["rows"].append(row)
                        save()
                        assert row["pcc"] >= 0.999 and row["all_rank_cache_exact"], row
                finally:
                    ttnn.release_trace(mesh, trace)
                    del output, tokens, positions, cache_positions, page_table, cache, caches
            decoder.decode_forward = original
        report["passed"] = True
        save()
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
