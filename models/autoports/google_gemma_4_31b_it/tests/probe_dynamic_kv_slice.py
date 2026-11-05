# SPDX-License-Identifier: Apache-2.0
"""Validate device-indexed BFP8 page extraction and partial-page blending.

Synthetic cache values exercise real TP4 sliding/global local-head geometries.
This is a correctness probe, not a model or performance measurement.
"""

import argparse
import json
from pathlib import Path

import torch

import ttnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/autoports/google_gemma_4_31b_it/readiness_vllm/dynamic_kv_slice.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(349)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    mesh.enable_program_cache()
    results = []
    try:

        def upload(value, dtype, layout):
            return ttnn.from_torch(
                value,
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def copy(value, target):
            host = ttnn.from_torch(
                value, dtype=target.dtype, layout=target.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
            )
            ttnn.copy_host_to_device_tensor(host, target)

        def read(tensor):
            return [ttnn.to_torch(chip).clone() for chip in ttnn.get_device_tensors(tensor)]

        # Tensor-args slice reads one source tile from each index buffer. These
        # 2048-byte buffers cover the 1088-byte BFP8 tile transaction explicitly.
        starts = upload(torch.zeros(512, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ends = upload(torch.zeros(512, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        pages = 17
        page_order = (11, 2, 16, 0, 7, 11, 1, 16)
        prefix_order = (1, 4, 31, 5, 16, 1, 29, 31)
        for kind, heads, width in (("sliding_attention", 4, 256), ("full_attention", 1, 512)):
            shape = (pages, heads, 32, width)
            source_host = torch.randn(shape, dtype=torch.bfloat16)
            source = upload(source_host, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
            baseline = read(source)
            replacement = upload(
                torch.randn(1, heads, 32, width, dtype=torch.bfloat16), ttnn.bfloat8_b, ttnn.TILE_LAYOUT
            )
            replacement_reference = read(replacement)
            replacement_bf16 = ttnn.typecast(replacement, ttnn.bfloat16)
            mask = upload(torch.zeros(1, heads, 32, width, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            extract_cache_count = None
            blend_cache_count = None
            cases = []
            for page, prefix in zip(page_order, prefix_order):
                start_values = torch.zeros(512, dtype=torch.int32)
                end_values = torch.zeros(512, dtype=torch.int32)
                start_values[:4] = torch.tensor([page, 0, 0, 0])
                end_values[:4] = torch.tensor([page + 1, heads, 32, width])
                copy(start_values, starts)
                copy(end_values, ends)
                before = mesh.num_program_cache_entries()
                extracted = ttnn.slice(source, starts, ends, slice_dim=0, num_devices=pages)
                actual = read(extracted)
                after_extract = mesh.num_program_cache_entries()
                assert after_extract > 0, "Program-cache reuse gate requires an enabled, populated cache"
                exact_extract = all(
                    torch.equal(value, reference[page : page + 1]) for value, reference in zip(actual, baseline)
                )
                # Count changes after the complete previous iteration, so first
                # blend warmup cannot look like an extraction cache miss.
                extraction_reused = extract_cache_count is None or after_extract == before
                extract_cache_count = after_extract
                mask_values = torch.zeros(1, heads, 32, width, dtype=torch.bfloat16)
                mask_values[:, :, :prefix, :] = 1
                copy(mask_values, mask)
                before_blend = mesh.num_program_cache_entries()
                old_bf16 = ttnn.typecast(extracted, ttnn.bfloat16)
                blended = ttnn.typecast(ttnn.where(mask, old_bf16, replacement_bf16), ttnn.bfloat8_b)
                blended_host = read(blended)
                after_blend = mesh.num_program_cache_entries()
                blending_reused = blend_cache_count is None or after_blend == before_blend
                blend_cache_count = after_blend
                prefix_exact = all(
                    torch.equal(value[:, :, :prefix], old[:, :, :prefix]) for value, old in zip(blended_host, actual)
                )
                suffix_exact = all(
                    torch.equal(value[:, :, prefix:], new[:, :, prefix:])
                    for value, new in zip(blended_host, replacement_reference)
                )
                cases.append(
                    {
                        "physical_page": page,
                        "prefix_rows": prefix,
                        "page_exact": exact_extract,
                        "prefix_exact": prefix_exact,
                        "suffix_exact": suffix_exact,
                        "extraction_program_reused": extraction_reused,
                        "blend_program_reused": blending_reused,
                        "program_count_before": before,
                        "program_count_after_extraction": after_extract,
                        "program_count_after_blend": after_blend,
                    }
                )
                del extracted, old_bf16, blended
            source_unchanged = all(torch.equal(a, b) for a, b in zip(baseline, read(source)))
            passed = source_unchanged and all(
                all(
                    case[key]
                    for key in (
                        "page_exact",
                        "prefix_exact",
                        "suffix_exact",
                        "extraction_program_reused",
                        "blend_program_reused",
                    )
                )
                for case in cases
            )
            results.append(
                {
                    "layer_type": kind,
                    "local_cache_shape": list(shape),
                    "source_dtype": "bfloat8_b",
                    "source_unchanged": source_unchanged,
                    "passed": passed,
                    "cases": cases,
                }
            )
            del source, replacement, replacement_bf16, mask
        report = {
            "scope": "synthetic_TP4_cache_operation_correctness",
            "passed": all(result["passed"] for result in results),
            "mesh": [1, 4],
            "index_buffer_elements": 512,
            "dynamic_slice": "ttnn.slice(cache, starts, ends, slice_dim=0, num_devices=cache.shape[0])",
            "blend": "BFP8 pages -> BF16 exact unpack -> where(runtime mask, old, new) -> BFP8",
            "results": results,
            "limitations": [
                "No model accuracy or performance claim.",
                "No writeback is performed; paged_fill_cache integration and prefix ownership require separate testing.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
        assert report["passed"], "Dynamic extraction/blend correctness or cache-reuse check failed"
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
