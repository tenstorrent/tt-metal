# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Primitive correctness gate for runtime SDPA offsets aligned to 32, not 512.

Run under an external bounded timeout. No traces, profiling, model quality, or
performance claim. Replicated inputs on a 1x4 mesh exercise the serving device
geometry; numerical comparisons read the first replica. The reference uses the
actual quantized device KV values, then applies absolute-position causality.
"""

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

import torch

import ttnn

CAPACITY = 2048
PAGE_SIZE = 32
HEADS = 6
HEAD_DIM = 256


def eligible(start, length):
    """Host guard only: an arbitrary start must first peel to a tile boundary."""
    return start >= 0 and start % PAGE_SIZE == 0 and length > 0 and start + ((length + 31) // 32) * 32 <= CAPACITY


def pcc_rows(reference, actual):
    x, y = reference.double(), actual.double()
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)
    denominator = x.square().sum(dim=-1).sqrt() * y.square().sum(dim=-1).sqrt()
    # Random projected values should not be constant; treat any constant row as
    # failing rather than silently excluding it from the minimum-row check.
    return torch.where(denominator > 0, (x * y).sum(dim=-1) / denominator, torch.zeros_like(denominator))


def metrics(reference, actual):
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    pcc = float(pcc_rows(reference.flatten()[None], actual.flatten()[None])[0]) if finite else 0.0
    minimum = float(pcc_rows(reference, actual).min()) if finite else 0.0
    return {
        "finite": finite,
        "pcc": pcc,
        "minimum_head_row_pcc": minimum,
        "max_abs_error": float((reference - actual).abs().max()) if finite else None,
        "passed": finite and pcc >= 0.995 and minimum >= 0.99,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(8161)
    cases = [
        (start, length)
        for start in (0, 32, 480, 512, 992, 2016)
        for length in (1, 31, 32, 33)
        if eligible(start, length)
    ]
    cases += [(0, 512), (32, 480), (512, 512)]
    physical_lengths = sorted({((length + 31) // 32) * 32 for _, length in cases})
    result = {
        "command": shlex.join([sys.executable, *sys.argv]),
        "cwd": os.getcwd(),
        "status": "running",
        "passed": False,
        "mesh": [1, 4],
        "comparison_replica": 0,
        "query_heads": HEADS,
        "kv_heads": 1,
        "head_dim": HEAD_DIM,
        "capacity": CAPACITY,
        "page_size": PAGE_SIZE,
        "query_dtype": "bfloat16",
        "kv_dtype": "bfloat8_b",
        "reference": "FP32 causal SDPA using first-replica quantized K/V readback and BF16 Q",
        "sdpa_config": {"grid": [8, 4], "q_chunk_size": 32, "k_chunk_size": 512, "exp_approx_mode": False},
        "compute_config": {
            "math_fidelity": "HiFi4",
            "math_approx_mode": False,
            "fp32_dest_acc_en": True,
            "packer_l1_acc": True,
        },
        "physical_query_lengths": physical_lengths,
        "thresholds": {"pcc": 0.995, "minimum_head_row_pcc": 0.99},
        "negative_control": {"start": 481, "length": 32, "host_eligible": eligible(481, 32), "sent_to_device": False},
        "skipped_invalid_cases": [{"start": 2016, "length": 33, "reason": "logical end exceeds capacity"}],
        "traces": False,
        "profiling": False,
        "performance_claim": False,
        "cases": [],
    }
    assert not result["negative_control"]["host_eligible"]
    mesh = None
    started = time.monotonic()

    def write_report():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    write_report()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        def upload(value, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper)

        def refresh(value, destination, dtype, layout):
            host = ttnn.from_torch(value, dtype=dtype, layout=layout, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(host, destination)

        def read_first(value):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).float()

        # page_table maps virtual pages to physical pages. Populate physical KV
        # with distinguishable random pages and read the quantized values back.
        page_table = torch.randperm(CAPACITY // PAGE_SIZE).to(torch.int32)[None]
        shape = (CAPACITY // PAGE_SIZE, 1, PAGE_SIZE, HEAD_DIM)
        key = upload(torch.randn(shape), ttnn.bfloat8_b)
        value = upload(torch.randn(shape), ttnn.bfloat8_b)
        table = upload(page_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        reference_k, reference_v = [
            read_first(tensor)[page_table[0].long()].transpose(0, 1).reshape(1, 1, CAPACITY, HEAD_DIM)
            for tensor in (key, value)
        ]
        result["page_table"] = page_table[0].tolist()
        starts = [upload(torch.tensor([0], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for _ in range(2)]
        queries = {
            length: upload(torch.zeros((1, HEADS, length, HEAD_DIM), dtype=torch.bfloat16), ttnn.bfloat16)
            for length in physical_lengths
        }
        result["persistent_start_addresses"] = [int(tensor.buffer_address()) for tensor in starts]
        result["persistent_query_addresses"] = {
            str(length): int(tensor.buffer_address()) for length, tensor in queries.items()
        }
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            q_chunk_size=32,
            k_chunk_size=512,
            exp_approx_mode=False,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def invoke(query, start_tensor):
            return ttnn.transformer.chunked_scaled_dot_product_attention(
                query,
                key,
                value,
                table,
                chunk_start_idx_tensor=start_tensor,
                program_config=program_config,
                compute_kernel_config=compute_config,
            )

        # Prepare each physical signature once, before the no-growth checks.
        for length in physical_lengths:
            warm_output = invoke(queries[length], starts[0])
            read_first(warm_output)
            del warm_output
        baseline_programs = int(mesh.num_program_cache_entries())
        result["program_cache_after_warmup"] = baseline_programs
        write_report()

        # Pass 1 updates the same start buffer across all offsets. Pass 2 changes
        # Q values, reverses case order, and alternates two same-spec start buffers
        # to exercise cached-program descriptor rebinding as well as runtime reads.
        for repetition in range(2):
            order = cases if repetition == 0 else list(reversed(cases))
            for index, (start, logical_length) in enumerate(order):
                assert eligible(start, logical_length)
                physical_length = ((logical_length + 31) // 32) * 32
                seed = 8161 + 1000 * repetition + index
                rng = torch.Generator().manual_seed(seed)
                q = torch.randn((1, HEADS, physical_length, HEAD_DIM), generator=rng).to(torch.bfloat16)
                q[:, :, logical_length:] = 0
                query_tensor = queries[physical_length]
                start_slot = index % 2 if repetition else 0
                start_tensor = starts[start_slot]
                refresh(q, query_tensor, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                refresh(torch.tensor([start], dtype=torch.int32), start_tensor, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                output = invoke(query_tensor, start_tensor)
                actual = read_first(output)[:, :, :logical_length]
                del output
                assert int(read_first(start_tensor)[0]) == start
                assert torch.equal(read_first(query_tensor), q.float())
                # SDPA's is_causal=True assumes Q starts at zero; the explicit
                # absolute mask below is essential for a continuation reference.
                q_positions = torch.arange(start, start + logical_length)[:, None]
                visible = torch.arange(CAPACITY)[None, :] <= q_positions
                scores = q[:, :, :logical_length].float() @ reference_k.transpose(-2, -1) * HEAD_DIM**-0.5
                expected = scores.masked_fill(~visible, float("-inf")).softmax(dim=-1) @ reference_v
                row = {
                    "repetition": repetition,
                    "start": start,
                    "logical_length": logical_length,
                    "physical_length": physical_length,
                    "query_seed": seed,
                    "start_buffer_slot": start_slot,
                    **metrics(expected, actual),
                    "program_cache_entries": int(mesh.num_program_cache_entries()),
                }
                row["program_cache_stable"] = row["program_cache_entries"] == baseline_programs
                row["passed"] = row["passed"] and row["program_cache_stable"]
                result["cases"].append(row)
                write_report()
                assert row["passed"], f"SDPA tile-start gate failed: {row}"
        result["passed"] = True
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["exception"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds_including_setup"] = time.monotonic() - started
        write_report()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    print(
        f"SDPA_TILE_START_PROBE passed={result['passed']} cases={len(result['cases'])} output={args.output}", flush=True
    )


if __name__ == "__main__":
    main()
