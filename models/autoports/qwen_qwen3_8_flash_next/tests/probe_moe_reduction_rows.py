# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-input routed reduction diagnostic; no weights, cache, or model required.

Serialize hardware externally and restore the accepted mesh descriptor after
lane-env before running::

  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
    timeout 180 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_moe_reduction_rows \
    --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/moe_reduction_rows_v1.json

--cpu-only validates fixture construction/reference math without importing TTNN.
Optional --fixture accepts a torch-save dictionary with `combined` BF16
[4,R,10,2560], `weights` BF16 [R,10], `ids` integer [R,10], R>=2, and optional
`provenance`. Capture these immediately before MeshMoe._reduce; combined's
leading axis is mesh rank, not a model batch. Only rows0/1 are used. Captured
nonowned values are preserved; the reference masks them before arithmetic.

Calls actual MeshMoe._reduce and _reduce_single, never replacement device math.
The object has only their required metadata, avoiding expert weight/workspace
allocation. Same-native-path invariance is asserted; B1 versus native and CPU
differences are diagnostic, not an invented accuracy gate or precision change.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]


def validate_fixture(fixture):
    values, weights, ids = (fixture[key] for key in ("combined", "weights", "ids"))
    assert values.ndim == 4 and values.shape[0] == 4 and values.shape[2:] == (10, 2560)
    assert values.shape[1] >= 2 and weights.shape == ids.shape == (values.shape[1], 10)
    assert values.dtype == weights.dtype == torch.bfloat16
    assert ids.dtype in (torch.int16, torch.int32, torch.int64, torch.uint16)
    ids = ids.to(torch.int32)
    assert ((ids >= 0) & (ids < 512)).all() and torch.isfinite(weights).all()
    for rank in range(4):
        owned = ((ids >= rank * 128) & (ids < (rank + 1) * 128)).unsqueeze(-1)
        assert torch.isfinite(torch.where(owned, values[rank], 0)).all()


def synthetic_fixture(pattern, seed):
    rng = torch.Generator().manual_seed(seed)
    owners = {
        "zero_local_rank0": [1] * 10,
        "one_local_rank0": [0, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "multiple_each_rank": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
    }[pattern]
    base = torch.tensor([128 * owner + index for index, owner in enumerate(owners)], dtype=torch.int32)
    ids = torch.stack((base, base.roll(3)))
    weights = torch.softmax(torch.randn(2, 10, generator=rng), dim=-1).to(torch.bfloat16)
    values = torch.randn(4, 2, 10, 2560, generator=rng).to(torch.bfloat16)
    # Alternate signs and modest expert scales exercise cancellation without
    # introducing overflow or synthetic nonfinite data absent from combine.
    scales = torch.tensor([1, -1, 0.5, -0.5, 2, -2, 1, -1, 0.25, -0.25])
    values = (values.float() * scales[None, None, :, None]).to(torch.bfloat16)
    for rank in range(4):
        owned = ((ids >= rank * 128) & (ids < (rank + 1) * 128)).unsqueeze(-1)
        values[rank] = torch.where(owned, values[rank], 0)
    result = {
        "combined": values,
        "weights": weights,
        "ids": ids,
        "provenance": {"kind": "synthetic_real_reduction_shape", "pattern": pattern, "seed": seed},
    }
    validate_fixture(result)
    return result


def cpu_reference(fixture, rows):
    ids, weights = fixture["ids"].to(torch.int32)[rows], fixture["weights"][rows].float()
    result = []
    for rank in range(4):
        owned = ((ids >= rank * 128) & (ids < (rank + 1) * 128)).unsqueeze(-1)
        values = torch.where(owned, fixture["combined"][rank, rows].float(), 0)
        result.append((values * weights.unsqueeze(-1)).sum(dim=1))
    return torch.stack(result)


def cpu_bf16_candidates(fixture, rows):
    """Finite-data hypotheses, not an asserted bit-accurate FPU/packer model.

    HiFi4 masks follow Blackhole FidelityMasking._apply_fidelity_masking in
    tt_metal/tt-llk/tests/python_tests/helpers/golden_generators.py: SrcA
    11111000000 / 00000111110, SrcB 11111110000 / 00000001111. For BF16
    sources these separate the low3/low1 mantissa bits, respectively. Both
    separately rounded and fused phase accumulation are retained as hypotheses.
    The truncation model is a negative control, not claimed hardware behavior.
    These models omit subnormal/overflow/stochastic semantics.
    """
    ids = fixture["ids"].to(torch.int32)[rows]
    weights = fixture["weights"][rows].float().unsqueeze(-1)

    def rne(x):
        return x.to(torch.bfloat16).float()

    def trunc(x):
        return (x.contiguous().view(torch.int32) & ~0xFFFF).view(torch.float32)

    def high(x, drop):
        return (x.contiguous().view(torch.int32) & ~((1 << (16 + drop)) - 1)).view(torch.float32)

    variants = {
        name: []
        for name in (
            "product_rne_serial_rne",
            "hifi4_separate_phase_rne_serial_rne",
            "hifi4_fused_phase_rne_serial_rne",
            "product_trunc_serial_trunc",
        )
    }
    for rank in range(4):
        owned = ((ids >= rank * 128) & (ids < (rank + 1) * 128)).unsqueeze(-1)
        a = torch.where(owned, fixture["combined"][rank, rows].float(), 0)
        b = weights
        ahi, bhi = high(a, 3), high(b, 1)
        alo, blo = a - ahi, b - bhi
        phases = (ahi * bhi, alo * bhi, ahi * blo, alo * blo)
        separate, fused = rne(phases[0]), rne(phases[0])
        for phase in phases[1:]:
            separate = rne(separate + rne(phase))
            fused = rne(fused + phase)
        products = {
            "product_rne_serial_rne": rne(a * b),
            "hifi4_separate_phase_rne_serial_rne": separate,
            "hifi4_fused_phase_rne_serial_rne": fused,
            "product_trunc_serial_trunc": trunc(a * b),
        }
        for name, product in products.items():
            quantize = trunc if name == "product_trunc_serial_trunc" else rne
            acc = torch.zeros(len(rows), 2560)
            for expert in range(10):
                acc = quantize(acc + product[:, expert])
            variants[name].append(acc.to(torch.bfloat16))
    return {name: torch.stack(parts) for name, parts in variants.items()}


def compare(expected, actual):
    assert expected.shape == actual.shape, (expected.shape, actual.shape)
    left, right = expected.double().reshape(-1), actual.double().reshape(-1)
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return {"exact": False, "finite": False, "shape": list(expected.shape)}
    delta = (left - right).abs()
    a, b = left - left.mean(), right - right.mean()
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return {
        "exact": bool(torch.equal(expected, actual)),
        "finite": True,
        "shape": list(expected.shape),
        "different_elements": int((left != right).sum()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "pcc": float(torch.dot(a, b) / denominator) if denominator else None,
        "relative_l2": float(torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(left).clamp_min(1e-30)),
    }


def make_reducer(tt, mesh):
    from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshLoader, MeshMoe

    loader = MeshLoader({}, mesh)
    moe = MeshMoe.__new__(MeshMoe)
    moe.compute = loader.compute  # Exact accepted HiFi4/FP32 DEST n=1 config.
    moe.workspace = SimpleNamespace(
        combine_zeros=tt.zeros((320, 2560), device=mesh, dtype=tt.bfloat16, layout=tt.ROW_MAJOR_LAYOUT),
        score_zeros=tt.zeros((32, 10), device=mesh, dtype=tt.bfloat16, layout=tt.ROW_MAJOR_LAYOUT),
        id_zeros=tt.from_torch(
            torch.zeros(32, 10, dtype=torch.int32), device=mesh, dtype=tt.uint16, layout=tt.ROW_MAJOR_LAYOUT
        ),
    )
    for name, offset in (("rank_lo", 0), ("rank_hi", 128)):
        setattr(
            moe.workspace,
            name,
            loader.shards(
                [torch.full((1, 1, 1, 1), rank * 128 + offset, dtype=torch.float32) for rank in range(4)],
                dim=0,
                dtype=tt.float32,
            ),
        )
    table = torch.full((4, 512), -1, dtype=torch.int32)
    for rank in range(4):
        table[rank, rank * 128 : (rank + 1) * 128] = 0
    moe.reduce_table = loader.shards(list(table.split(1)), dim=0, dtype=tt.int32, layout=tt.ROW_MAJOR_LAYOUT)
    return moe


def run_fixture(tt, mesh, moe, fixture, tensor_outputs=None):
    def upload(rows):
        n = len(rows)
        values = fixture["combined"][:, rows].reshape(4, 1, n, 10, 2560)
        combined = tt.from_torch(
            values.contiguous(),
            device=mesh,
            dtype=tt.bfloat16,
            layout=tt.ROW_MAJOR_LAYOUT,
            mesh_mapper=tt.ShardTensorToMesh(mesh, dim=0),
        )
        weights = tt.from_torch(
            fixture["weights"][rows].reshape(1, 1, n, 10).contiguous(),
            device=mesh,
            dtype=tt.bfloat16,
            layout=tt.ROW_MAJOR_LAYOUT,
            mesh_mapper=tt.ReplicateTensorToMesh(mesh),
        )
        ids = tt.from_torch(
            fixture["ids"][rows].to(torch.int32).reshape(1, 1, n, 10).contiguous(),
            device=mesh,
            dtype=tt.uint16,
            layout=tt.ROW_MAJOR_LAYOUT,
            mesh_mapper=tt.ReplicateTensorToMesh(mesh),
        )
        return combined, weights, ids

    def read(result, n):
        return torch.stack([tt.to_torch(rank).reshape(n, 2560).clone() for rank in tt.get_device_tensors(result)])

    def run(rows, repeat=False):
        tensors = upload(rows)
        output = read(moe._reduce(*tensors, len(rows)), len(rows))
        # Reuse the exact resident inputs, not another upload, for replay control.
        again = read(moe._reduce(*tensors, len(rows)), len(rows)) if repeat else None
        return output, again

    singles = [run([row], repeat=True) for row in (0, 1)]
    single = torch.cat([pair[0] for pair in singles], dim=1)
    single_repeated = torch.cat([pair[1] for pair in singles], dim=1)
    mixed, repeated = run([0, 1], repeat=True)
    swapped, _ = run([1, 0])
    duplicated, _ = run([0, 0])
    rows32 = [0, 1] * 16
    full, full_repeated = run(rows32, repeat=True)
    shifted, _ = run([1, 0] * 16)
    reference = cpu_reference(fixture, [0, 1])
    candidates = cpu_bf16_candidates(fixture, [0, 1])
    if tensor_outputs is not None:
        tensor_outputs.append({"fixture": fixture, "single": single, "native_batch": mixed, "native_full32": full})
    invariants = {
        "single_repeat": compare(single, single_repeated),
        "native_b2_repeat": compare(mixed, repeated),
        "native_b32_repeat": compare(full, full_repeated),
        "native_b2_permutation": compare(mixed, swapped[:, [1, 0]]),
        "native_b2_duplicate_rows": compare(mixed[:, [0, 0]], duplicated),
        "native_b2_vs_all32_positions": compare(mixed[:, rows32], full),
        "native_b32_row_relocation": compare(full, shifted[:, [i ^ 1 for i in range(32)]]),
    }
    return {
        "provenance": fixture.get("provenance", {}),
        "local_experts_per_rank": [
            [
                int(((fixture["ids"][row].int() >= rank * 128) & (fixture["ids"][row].int() < (rank + 1) * 128)).sum())
                for row in (0, 1)
            ]
            for rank in range(4)
        ],
        "invariants": invariants,
        "all_native_invariants_exact": all(item["exact"] for item in invariants.values()),
        "single_vs_native_batch": compare(single, mixed),
        "single_vs_cpu_fp32": compare(reference, single),
        "native_batch_vs_cpu_fp32": compare(reference, mixed),
        "single_vs_cpu_rounded_bf16": compare(reference.to(torch.bfloat16), single),
        "native_batch_vs_cpu_rounded_bf16": compare(reference.to(torch.bfloat16), mixed),
        "native_batch_vs_cpu_bf16_candidates": {
            name: compare(candidate, mixed) for name, candidate in candidates.items()
        },
        "per_rank_native_vs_cpu_bf16_candidates": {
            name: [compare(candidate[rank], mixed[rank]) for rank in range(4)] for name, candidate in candidates.items()
        },
        "per_rank_single_vs_native": [compare(single[rank], mixed[rank]) for rank in range(4)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--seed", type=int, default=162)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--tensor-output", type=Path, help="Optional device results + fixtures for offline CPU analysis"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {
        "completed": False,
        "cpu_only": args.cpu_only,
        "scope": "same-input MoE reduction diagnostic; not full-model acceptance",
        "reference_note": "FP32 mathematical reference; not a simulation of native BF16 packer rounding.",
    }
    mesh, started, tensor_outputs = None, time.monotonic(), []
    try:
        fixtures = (
            [torch.load(args.fixture, weights_only=True, map_location="cpu")]
            if args.fixture
            else [
                synthetic_fixture(pattern, args.seed + index)
                for index, pattern in enumerate(("zero_local_rank0", "one_local_rank0", "multiple_each_rank"))
            ]
        )
        for fixture in fixtures:
            validate_fixture(fixture)
            reference = cpu_reference(fixture, [0, 1])
            assert torch.equal(reference[:, [1, 0]], cpu_reference(fixture, [1, 0]))
            assert torch.equal(reference[:, [0, 0]], cpu_reference(fixture, [0, 0]))
            candidates = cpu_bf16_candidates(fixture, [0, 1])
            swapped = cpu_bf16_candidates(fixture, [1, 0])
            assert all(torch.equal(value[:, [1, 0]], swapped[name]) for name, value in candidates.items())
        if args.cpu_only:
            report["fixture_contracts_and_cpu_reference"] = "passed"
            report["fixtures"] = [fixture.get("provenance", {}) for fixture in fixtures]
        else:
            assert os.environ.get("TT_MESH_GRAPH_DESC_PATH"), "Restore the accepted mesh descriptor after lane-env"
            assert not os.environ.get("TT_METAL_DEVICE_PROFILER", "").startswith("1")
            assert int(os.environ.get("TT_METAL_PROFILE_PERF_COUNTERS") or "0") == 0
            assert not os.environ.get("TT_METAL_DEVICE_PROFILER_STREAMING", "").startswith("1")
            import ttnn as tt

            tt.set_fabric_config(tt.FabricConfig.FABRIC_1D_RING)
            mesh = tt.open_mesh_device(tt.MeshShape(1, 4), l1_small_size=32768)
            moe = make_reducer(tt, mesh)
            report["cases"] = []
            for index, fixture in enumerate(fixtures):
                print("MOE_REDUCE_PHASE", index, fixture.get("provenance", {}), flush=True)
                report["cases"].append(
                    run_fixture(tt, mesh, moe, fixture, tensor_outputs if args.tensor_output else None)
                )
            report["all_native_invariants_exact"] = all(case["all_native_invariants_exact"] for case in report["cases"])
            assert report["all_native_invariants_exact"], "Native same-path row/replay invariance failed; inspect JSON"
        report["completed"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        report["source_sha256"] = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
        native = ROOT.parents[2] / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device"
        report["native_source_sha256"] = {
            name: hashlib.sha256((native / name).read_bytes()).hexdigest()
            for name in (
                "post_combine_reduce_program_factory.cpp",
                "kernels/deepseek_moe_post_combine_reduce_compute.cpp",
                "kernels/deepseek_moe_post_combine_reduce_reader.cpp",
                "kernels/deepseek_moe_post_combine_reduce_writer.cpp",
            )
        }
        if args.fixture:
            report["fixture_sha256"] = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
        if args.tensor_output and tensor_outputs:
            args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor_outputs, args.tensor_output)
            report["tensor_output"] = str(args.tensor_output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            tt.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
