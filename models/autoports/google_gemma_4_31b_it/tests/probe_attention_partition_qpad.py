# SPDX-License-Identifier: Apache-2.0
"""Optional control: compare offset Q32 with the same slice zero-padded into Q128.

Uses real P19 prefill Q and the same cached K/V. Candidate only; no profiling.
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


def validate_p19_policy(policy):
    """Reject accidental use of the selected/default policy for the P19 probe."""
    expected = {
        "config_id": "stage12_bf16_hifi4_kv_down_bfp8_p19",
        "weight_groups": {
            "attention": "bf16",
            "mlp": "bf16",
            "down": "bfp8_lofi",
            "lm_head": "bf16",
        },
        "compute_fidelities": {
            "attention": "HiFi4",
            "mlp": "HiFi4",
            "down": "LoFi",
            "lm_head": "HiFi4",
        },
        "activation_dtype": "bfloat16",
        "residual_dtype": "bfloat16",
        "ccl_dtype": {"prefill": "bfloat16", "decode": "bfloat16"},
        "kv_cache_dtype": "bfloat8_b",
        "sdpa_compute_fidelity": "HiFi4",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"P19 policy mismatch for {key}: {policy.get(key)!r} != {value!r}")


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


def compare_bands(reference, candidate, *, absolute_start, band_size=32):
    rows = []
    for begin in range(0, reference.shape[0], band_size):
        end = min(reference.shape[0], begin + band_size)
        rows.append(
            {
                "absolute_query_range": [absolute_start + begin, absolute_start + end],
                **compare(reference[begin:end], candidate[begin:end]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="0,5")
    parser.add_argument("--q128-k32", action="store_true", help="Also compare aligned Q128 with fixed K32")
    parser.add_argument("--precision-config", type=Path, default=None)
    parser.add_argument("--require-p19-policy", action="store_true")
    parser.add_argument("--compare-start", type=int, default=32)
    parser.add_argument("--compare-length", type=int, default=32)
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
    precision_config = None
    precision_config_sha256 = None
    if args.precision_config is not None:
        precision_bytes = args.precision_config.read_bytes()
        precision_config_sha256 = hashlib.sha256(precision_bytes).hexdigest()
        precision_config = json.loads(precision_bytes)
    if args.require_p19_policy:
        if precision_config is None:
            parser.error("--require-p19-policy requires --precision-config")
        validate_p19_policy(precision_config)
    if (
        args.compare_start < 0
        or args.compare_start % 32
        or args.compare_length <= 0
        or args.compare_length % 32
        or args.compare_start + args.compare_length > 1024
    ):
        parser.error("comparison query range must be 32-token aligned and lie within the 1024-token source prefill")
    torch.set_num_threads(8)
    text = bz2.open(args.text_file, "rt").read() if args.text_file.suffix == ".bz2" else args.text_file.read_text()
    text_file_sha256 = hashlib.sha256(args.text_file.read_bytes()).hexdigest()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = Gemma4Generator(
            mesh,
            layer_indices=indices,
            batch_size=1,
            max_seq_len=4096,
            precision_config=precision_config,
        )
        gen.prepare_decode()
        ids = gen.tokenizer.encode(text, add_special_tokens=True)[:1024]
        assert len(ids) == 1024
        prompt_ids_sha256 = hashlib.sha256(torch.tensor(ids, dtype=torch.int64).numpy().tobytes()).hexdigest()
        if args.require_p19_policy:
            assert gen.model.precision_config == precision_config
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
            compare_end = args.compare_start + args.compare_length
            q_slice = ttnn.slice(q, (0, 0, args.compare_start, 0), (1, layer.nq, compare_end, layer.head_dim))
            identical_q = torch.equal(
                query_rows(q, args.compare_start, compare_end), query_rows(q_slice, 0, args.compare_length)
            )
            assert identical_q, "Query slice must preserve the exact supplied Q values"
            table = gen.prefill.attention[ordinal]
            offset = gen.prefill.offsets[ordinal]
            cache_before = tensor_digest(cache)
            table_before = tensor_digest([table])
            outputs = {}
            signatures = {}
            zero_prefix = ttnn.zeros(
                (1, layer.nq, args.compare_start, layer.head_dim),
                dtype=q.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            pad_tail = (-(args.compare_start + args.compare_length)) % 128
            zero_tail = (
                ttnn.zeros(
                    (1, layer.nq, pad_tail, layer.head_dim),
                    dtype=q.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if pad_tail
                else None
            )
            padded_parts = [zero_prefix, q_slice]
            if zero_tail is not None:
                padded_parts.append(zero_tail)
            padded_offset_q = ttnn.concat(padded_parts, dim=2)
            assert padded_offset_q.shape[2] % 128 == 0
            assert padded_offset_q.shape[2] == 384
            arms = [
                ("full_q128_k128", q, 0, 128, 128),
                ("full_q32_k32", q, 0, 32, 32),
                ("offset_slice_q32_k32", q_slice, args.compare_start, 32, 32),
                ("zero_padded_offset_q128_k32", padded_offset_q, 0, 128, 32),
                ("full_q128_k32", q, 0, 128, 32),
            ]
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
                outputs[name] = query_rows(
                    attention,
                    args.compare_start if start == 0 else 0,
                    compare_end if start == 0 else args.compare_length,
                )
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
            table_after = tensor_digest([table])
            records.append(
                {
                    "layer_index": index,
                    "layer_type": layer.layer_type,
                    "absolute_query_range": [args.compare_start, compare_end],
                    "query_slice_bit_equal": identical_q,
                    "cache_unchanged": cache_before == cache_after,
                    "cache_digest_before": cache_before,
                    "cache_digest_after": cache_after,
                    "page_table_unchanged": table_before == table_after,
                    "page_table_digest_before": table_before,
                    "page_table_digest_after": table_after,
                    "query_digest": tensor_digest([q]),
                    "signatures": signatures,
                    "full128_k128_vs_full32_k32": compare(outputs["full_q128_k128"], outputs["full_q32_k32"]),
                    "full128_k128_vs_offset_slice_q32_k32": compare(
                        outputs["full_q128_k128"], outputs["offset_slice_q32_k32"]
                    ),
                    "full32_vs_offset_slice_q32_k32": compare(outputs["full_q32_k32"], outputs["offset_slice_q32_k32"]),
                    "full_q128_k32_vs_zero_padded_offset_q128_k32": compare(
                        outputs["full_q128_k32"], outputs["zero_padded_offset_q128_k32"]
                    ),
                    "offset_slice_q32_k32_vs_zero_padded_offset_q128_k32": compare(
                        outputs["offset_slice_q32_k32"], outputs["zero_padded_offset_q128_k32"]
                    ),
                    "full128_k128_vs_full32_k32_bands32": compare_bands(
                        outputs["full_q128_k128"], outputs["full_q32_k32"], absolute_start=args.compare_start
                    ),
                    "full128_k128_vs_offset_slice_q32_k32_bands32": compare_bands(
                        outputs["full_q128_k128"], outputs["offset_slice_q32_k32"], absolute_start=args.compare_start
                    ),
                    "full32_vs_offset_slice_q32_k32_bands32": compare_bands(
                        outputs["full_q32_k32"], outputs["offset_slice_q32_k32"], absolute_start=args.compare_start
                    ),
                }
            )
            if args.q128_k32:
                records[-1].update(
                    full128_k32_vs_full32_k32=compare(outputs["full_q128_k32"], outputs["full_q32_k32"]),
                    full128_k32_vs_offset_slice_q32_k32=compare(
                        outputs["full_q128_k32"], outputs["offset_slice_q32_k32"]
                    ),
                    full128_k32_vs_full32_k32_bands32=compare_bands(
                        outputs["full_q128_k32"], outputs["full_q32_k32"], absolute_start=args.compare_start
                    ),
                    full128_k32_vs_offset_slice_q32_k32_bands32=compare_bands(
                        outputs["full_q128_k32"], outputs["offset_slice_q32_k32"], absolute_start=args.compare_start
                    ),
                    full128_k32_vs_zero_padded_offset_q128_k32_bands32=compare_bands(
                        outputs["full_q128_k32"],
                        outputs["zero_padded_offset_q128_k32"],
                        absolute_start=args.compare_start,
                    ),
                    offset_slice_q32_k32_vs_zero_padded_offset_q128_k32_bands32=compare_bands(
                        outputs["offset_slice_q32_k32"],
                        outputs["zero_padded_offset_q128_k32"],
                        absolute_start=args.compare_start,
                    ),
                    full128_k128_vs_full128_k32=compare(outputs["full_q128_k128"], outputs["full_q128_k32"]),
                )
            del q_slice, padded_offset_q, zero_prefix
            if zero_tail is not None:
                del zero_tail
        report = {
            "scope": "reduced_real_layers",
            "reference_revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "layer_indices": list(indices),
            "source_text": str(args.text_file),
            "source_text_sha256": text_file_sha256,
            "prompt_tokens": 1024,
            "compared_query_range": [args.compare_start, args.compare_start + args.compare_length],
            "comparison_band_size": 32,
            "prompt_ids_sha256": prompt_ids_sha256,
            "precision_config_path": str(args.precision_config) if args.precision_config else None,
            "precision_config_sha256": precision_config_sha256,
            "precision_policy_required": args.require_p19_policy,
            "p19_policy_validation": "passed" if args.require_p19_policy else "not_requested",
            "precision_config": gen.model.precision_config,
            "records": records,
            "interpretation": [
                "Every arm uses identical supplied rotated queries and the same unchanged K/V cache and table.",
                "full128_k32_vs_full32_k32 isolates Q partition size at the same absolute origin, holding K32 fixed.",
                "full128_k128_vs_full32_k32 changes both Q and K partition sizes and is not an isolated Q comparison.",
                "full32_vs_offset_slice_q32_k32 holds partition sizes fixed while changing query extent and absolute chunk offset.",
                "zero_padded_offset_q128_k32 inserts 32 leading zero query rows, the exact original Q[32:288] slice, then tail padding to 384 rows; outputs are read only for rows32:288. This tests whether offset-Q32 output agrees with the same Q values placed in a Q128 arm at absolute positions.",
                "Attention is compared before output projection, residual addition or MLP.",
            ],
            "limitations": [
                "Correctness-only host readback and probe allocations; no performance/profiler claim.",
                "A 1024-token natural-text P19 layer capture tests partition numerics, not end-to-end resumed-prefill behavior or the source failing prompt.",
                "Zero padding is an explicit control input and is not a normal model query; this arm tests masking/partition position handling, not a production padding policy.",
            ],
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
