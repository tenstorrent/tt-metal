# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real QSA index_qk projection precision candidates, diagnostic only.

Uses identical original 32-row chunks from the cutoff investigation and real
BF16 checkpoint weights. Reports full640, query512 and raw-key128 errors.
No model runtime changes, selection overrides, or acceptance changes.
"""

import argparse
import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_context_window import metrics


def compact_metrics(expected, actual):
    return {key: value for key, value in metrics(expected, actual).items() if not isinstance(value, list)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--source-cache", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_cpu_prefill_262144.pt")
    )
    parser.add_argument("--critical-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--policies", default="native,fp32_output,fp32_operands,split512,split256")
    parser.add_argument("--chunk-starts", help="Comma-separated original chunk starts; defaults to all critical chunks")
    parser.add_argument(
        "--capture-raw", action="store_true", help="Capture pre-cast FP32 values and operand roundtrips"
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    investigation = json.loads(args.critical_report.read_text())
    starts = investigation["projection_chunk_starts"]
    if args.chunk_starts:
        starts = [int(value) for value in args.chunk_starts.split(",")]
        assert all(start % 32 == 0 and start in investigation["projection_chunk_starts"] for start in starts)
    config, hf, _rope, weights, provenance = load_reference(args.snapshot)
    assert provenance == investigation["provenance"]
    policies = args.policies.split(",")
    assert set(policies) <= {
        "native",
        "fp32_output",
        "fp32_operands",
        "split512",
        "split256",
        "split128",
        "hifi3",
        "weight_split2",
        "weight_split3",
        "weight_split4",
        "sfpu_tree",
    }
    expected = [hf.indexer.index_qk_proj(source["hidden"][:, start : start + 32]) for start in starts]
    expected = torch.cat(expected, 1)
    high_precision_expected = torch.cat(
        [
            torch.nn.functional.linear(
                source["hidden"][:, start : start + 32].float(), hf.indexer.index_qk_proj.weight.float()
            ).bfloat16()
            for start in starts
        ],
        1,
    )
    report = {
        "scope": "Projection-only precision candidates on original real-weight 32-row chunks; diagnostic",
        "provenance": provenance,
        "chunk_starts": starts,
        "hf_bf16_vs_fp32_linear_rounded": compact_metrics(expected, high_precision_expected),
        "candidates": {},
    }
    capture = {"expected": expected, "hf_fp32_rounded": high_precision_expected, "chunk_starts": starts}
    if args.capture_raw:
        capture["hf_fp64"] = torch.cat(
            [
                torch.nn.functional.linear(
                    source["hidden"][:, start : start + 32].double(), hf.indexer.index_qk_proj.weight.double()
                )
                for start in starts
            ],
            1,
        )
        report["hf_bf16_vs_fp64_rounded"] = compact_metrics(expected, capture["hf_fp64"].bfloat16())
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader, linear

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        loader = WeightLoader(weights, mesh)
        matrix = loader.matrix("self_attn.indexer.index_qk_proj.weight")
        matrix32 = loader.matrix("self_attn.indexer.index_qk_proj.weight", ttnn.float32)
        hifi3 = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        decomposed_weights = {}
        sfpu_weights = []
        if "sfpu_tree" in policies:
            sfpu_weights = [
                loader.tensor(hf.indexer.index_qk_proj.weight[start : start + 32].reshape(1, 1, 32, 2560), ttnn.float32)
                for start in range(0, 640, 32)
            ]
            report["sfpu_tree_policy"] = (
                "FP32 elementwise products broadcast over 32 tokens and 32 output columns; "
                "pairwise FP32 adds over K, with odd tails added after the balanced part; "
                "no FPU matmul or FPU sum reduction; final BF16 cast only"
            )
        for bits in (2, 3, 4):
            if f"weight_split{bits}" not in policies:
                continue
            original = hf.indexer.index_qk_proj.weight.float().contiguous()
            high = (original.view(torch.int32) & (-1 << (23 - bits))).view(torch.float32)
            low = original - high
            assert torch.equal(high.bfloat16().float() + low.bfloat16().float(), original)
            decomposed_weights[bits] = (loader.tensor(high.T), loader.tensor(low.T))
        if decomposed_weights:
            report["weight_decomposition"] = {
                "method": "Retain top N explicit mantissa bits in high; low is exact BF16 residual; high+low equals original weight",
                "bits": list(decomposed_weights),
                "runtime": "Two FP32-output HiFi4 matmuls, SFPU FP32 add, one final BF16 cast",
            }
        if args.capture_raw:
            capture["weight_roundtrip"] = ttnn.to_torch(matrix)
            capture["weight32_roundtrip"] = ttnn.to_torch(matrix32)
            report["operand_roundtrip"] = {
                "weight_bf16_equal": torch.equal(capture["weight_roundtrip"], hf.indexer.index_qk_proj.weight.T),
                "weight_fp32_equal": torch.equal(
                    capture["weight32_roundtrip"], hf.indexer.index_qk_proj.weight.T.float()
                ),
                "input_equal": [],
            }
        split_weights = {
            size: [matrix[start : start + size, :] for start in range(0, 2560, size)]
            for size in (128, 256, 512)
            if f"split{size}" in policies
        }
        actual = {name: [] for name in policies}
        raw_actual = {name: [] for name in policies if name != "native"}
        for number, start in enumerate(starts):
            x = loader.tensor(source["hidden"][:, start : start + 32].unsqueeze(0))
            if args.capture_raw:
                report["operand_roundtrip"]["input_equal"].append(
                    torch.equal(ttnn.to_torch(x).reshape(1, 32, 2560), source["hidden"][:, start : start + 32])
                )
            for policy in policies:
                if policy == "native":
                    result = linear(x, matrix, loader.compute)
                elif policy == "fp32_operands":
                    result = ttnn.linear(
                        ttnn.typecast(x, ttnn.float32),
                        matrix32,
                        dtype=ttnn.float32,
                        compute_kernel_config=loader.compute,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                elif policy == "fp32_output":
                    result = ttnn.linear(
                        x,
                        matrix,
                        dtype=ttnn.float32,
                        compute_kernel_config=loader.compute,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                elif policy == "hifi3":
                    result = ttnn.linear(
                        x,
                        matrix,
                        dtype=ttnn.float32,
                        compute_kernel_config=hifi3,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                elif policy == "sfpu_tree":
                    expanded_x = ttnn.reshape(ttnn.typecast(x, ttnn.float32), (1, 32, 1, 2560))
                    columns = []
                    for weight in sfpu_weights:
                        products = ttnn.multiply(expanded_x, weight)
                        tails = []
                        width = 2560
                        while width > 1:
                            half = width // 2
                            if width % 2:
                                tails.append(products[..., -1:])
                            products = ttnn.add(products[..., :half], products[..., half : 2 * half])
                            width = half
                        for tail in tails:
                            products = ttnn.add(products, tail)
                        columns.append(ttnn.reshape(products, (1, 1, 32, 32)))
                    result = ttnn.concat(columns, dim=-1)
                elif policy.startswith("weight_split"):
                    high, low = decomposed_weights[int(policy.removeprefix("weight_split"))]
                    high_result = ttnn.linear(
                        x,
                        high,
                        dtype=ttnn.float32,
                        compute_kernel_config=loader.compute,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    low_result = ttnn.linear(
                        x,
                        low,
                        dtype=ttnn.float32,
                        compute_kernel_config=loader.compute,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    result = ttnn.add(high_result, low_result)
                else:
                    size = int(policy.removeprefix("split"))
                    result = None
                    for offset, weight in zip(range(0, 2560, size), split_weights[size]):
                        part = ttnn.linear(
                            x[..., offset : offset + size],
                            weight,
                            dtype=ttnn.float32,
                            compute_kernel_config=loader.compute,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        result = part if result is None else ttnn.add(result, part)
                if policy != "native":
                    if args.capture_raw:
                        raw_actual[policy].append(ttnn.to_torch(result).reshape(1, 32, 640))
                    result = ttnn.typecast(result, ttnn.bfloat16)
                actual[policy].append(ttnn.to_torch(result).reshape(1, 32, 640))
            if number % 32 == 0 or number == len(starts) - 1:
                print(json.dumps({"completed_chunks": number + 1, "total_chunks": len(starts)}), flush=True)
        for name, values in actual.items():
            joined = torch.cat(values, 1)
            capture[name] = joined
            report["candidates"][name] = {
                "full640": compact_metrics(expected, joined),
                "query512": compact_metrics(expected[..., :512], joined[..., :512]),
                "raw128": compact_metrics(expected[..., 512:], joined[..., 512:]),
            }
            if args.capture_raw and name != "native":
                raw = torch.cat(raw_actual[name], 1)
                capture[name + "_fp32"] = raw
                difference = raw.double() - capture["hf_fp64"]
                report["candidates"][name]["precast_fp32"] = {
                    "versus_fp64": compact_metrics(capture["hf_fp64"], raw),
                    "cpu_cast_vs_hf": compact_metrics(expected, raw.bfloat16()),
                    "device_cast_vs_cpu_cast_equal_fraction": float((joined == raw.bfloat16()).float().mean()),
                    "mean_error": float(difference.mean()),
                    "mean_absolute_error": float(difference.abs().mean()),
                    "rms_error": float(difference.square().mean().sqrt()),
                }
        torch.save(capture, args.capture)
        report["capture_path"] = str(args.capture)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    name: {
                        part: {
                            key: value
                            for key, value in metric.items()
                            if key not in ("row_pcc_fp64", "expected_row_std", "actual_row_std")
                        }
                        for part, metric in values.items()
                    }
                    for name, values in report["candidates"].items()
                },
                indent=2,
            ),
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
