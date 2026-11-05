# SPDX-License-Identifier: Apache-2.0
"""Correctness probe: identical rotated Q and cached K/V across SDPA partitions.

Uses reduced real layers 0/5; no profiling or serving-performance measurement.
"""

import argparse
import bz2
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import Gemma4Generator
from .probe_prefill_geometry import compare


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        for chip in ttnn.get_device_tensors(tensor):
            host = ttnn.to_torch(chip)
            digest.update(host.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def query_rows(tensor, begin, end):
    """Return [query rows, all chip/head features] without device-side slicing."""
    chips = []
    for chip in ttnn.get_device_tensors(tensor):
        host = ttnn.to_torch(chip).float()
        chips.append(host.movedim(2, 0).reshape(host.shape[2], -1)[begin:end].clone())
    return torch.cat(chips, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="0,5")
    parser.add_argument("--q128-k32", action="store_true", help="Also compare aligned Q128 with fixed K32")
    parser.add_argument(
        "--text-file", type=Path, default=Path("models/tt_transformers/tests/tale-of-two-cities.txt.bz2")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.output is None:
        filename = "attention_partition_q128_k32.json" if args.q128_k32 else "attention_partition.json"
        args.output = Path("models/autoports/google_gemma_4_31b_it/readiness_vllm") / filename
    indices = tuple(int(value) for value in args.layers.split(","))
    torch.set_num_threads(8)
    text = bz2.open(args.text_file, "rt").read() if args.text_file.suffix == ".bz2" else args.text_file.read_text()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = Gemma4Generator(mesh, layer_indices=indices, batch_size=1, max_seq_len=4096)
        gen.prepare_decode()
        ids = gen.tokenizer.encode(text, add_special_tokens=True)[:1024]
        assert len(ids) == 1024
        tokens = torch.tensor(ids, dtype=torch.int32)
        queries = {}

        def capture_qkv(original, index):
            def wrapped(x, cos, sin, *, decode):
                result = original(x, cos, sin, decode=decode)
                if not decode:
                    assert result[0].shape[2] == 1024
                    queries[index] = ttnn.clone(result[0], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                return result

            return wrapped

        patches = [
            patch.object(layer, "_qkv", capture_qkv(layer._qkv, index))
            for index, layer in zip(indices, gen.model.layers)
        ]
        for context in patches:
            context.start()
        try:
            gen.reset()
            for start, valid, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=0, end=1024):
                assert start == 0 and valid == 1024
                del hidden
        finally:
            for context in reversed(patches):
                context.stop()

        records = []
        for ordinal, (index, layer, cache) in enumerate(zip(indices, gen.model.layers, gen.kv_cache)):
            q = queries[index]
            assert tuple(q.shape) == (1, layer.nq, 1024, layer.head_dim)
            q_slice = ttnn.slice(q, (0, 0, 32, 0), (1, layer.nq, 64, layer.head_dim))
            identical_q = torch.equal(query_rows(q, 32, 64), query_rows(q_slice, 0, 32))
            assert identical_q, "Query slice must preserve the exact supplied Q values"
            table = gen.prefill.attention[ordinal]
            offset = gen.prefill.offsets[ordinal]
            cache_before = tensor_digest(cache)
            outputs = {}
            signatures = {}
            arms = [
                ("full_q128_k128", q, 0, 128, 128),
                ("full_q32_k32", q, 0, 32, 32),
                ("slice32_q32_k32", q_slice, 32, 32, 32),
            ]
            if args.q128_k32:
                arms.append(("full_q128_k32", q, 0, 128, 32))
            for name, query, start, q_chunk, k_chunk in arms:
                gen.prefill.copy(torch.tensor([start], dtype=torch.int32), offset)
                attention = ttnn.transformer.chunked_scaled_dot_product_attention(
                    query,
                    *cache,
                    table,
                    chunk_start_idx_tensor=offset,
                    scale=1.0,
                    sliding_window_size=layer.sliding_window,
                    program_config=ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=(8, 8), q_chunk_size=q_chunk, k_chunk_size=k_chunk
                    ),
                    compute_kernel_config=layer.prefill_sdpa_kernel,
                )
                outputs[name] = query_rows(attention, 32 if start == 0 else 0, 64 if start == 0 else 32)
                signatures[name] = {
                    "query_shape": list(query.shape),
                    "query_dtype": str(query.dtype),
                    "kv_dtype": str(cache[0].dtype),
                    "page_table_shape": list(table.shape),
                    "page_table_base_position": 0,
                    "chunk_start_idx_tensor": start,
                    "q_chunk_size": q_chunk,
                    "k_chunk_size": k_chunk,
                    "sliding_window_size": layer.sliding_window,
                    "compute_kernel_config": str(layer.prefill_sdpa_kernel),
                }
                del attention
            gen.prefill.copy(torch.tensor([0], dtype=torch.int32), offset)
            cache_after = tensor_digest(cache)
            records.append(
                {
                    "layer_index": index,
                    "layer_type": layer.layer_type,
                    "absolute_query_range": [32, 64],
                    "query_slice_bit_equal": identical_q,
                    "cache_unchanged": cache_before == cache_after,
                    "cache_digest_before": cache_before,
                    "cache_digest_after": cache_after,
                    "signatures": signatures,
                    "full128_vs_full32": compare(outputs["full_q128_k128"], outputs["full_q32_k32"]),
                    "full128_vs_slice32": compare(outputs["full_q128_k128"], outputs["slice32_q32_k32"]),
                    "full32_vs_slice32": compare(outputs["full_q32_k32"], outputs["slice32_q32_k32"]),
                }
            )
            if args.q128_k32:
                records[-1].update(
                    full128_k32_vs_full32=compare(outputs["full_q128_k32"], outputs["full_q32_k32"]),
                    full128_k32_vs_slice32=compare(outputs["full_q128_k32"], outputs["slice32_q32_k32"]),
                    full128_k128_vs_full128_k32=compare(outputs["full_q128_k128"], outputs["full_q128_k32"]),
                )
            del q_slice
        report = {
            "scope": "reduced_real_layers",
            "layer_indices": list(indices),
            "source_text": str(args.text_file),
            "prompt_tokens": 1024,
            "precision_config": gen.model.precision_config,
            "records": records,
            "interpretation": [
                "Every arm uses identical supplied rotated queries and the same unchanged K/V cache and table.",
                "full128_vs_full32 isolates Q/K partition selection at the same absolute origin.",
                "full32_vs_slice32 holds partition sizes fixed while changing query extent and chunk offset.",
                "Attention is compared before output projection, residual addition or MLP.",
            ],
            "limitations": ["Correctness-only host readback and probe allocations; no performance/profiler claim."],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        assert all(row["cache_unchanged"] for row in records), "Attention-only probe mutated K/V"
        print(json.dumps({"output": str(args.output), "layers": list(indices)}))
    finally:
        if gen is not None:
            gen.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
