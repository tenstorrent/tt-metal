# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact UINT16-code to compact-decay-index probe; deliberately performs no gather."""

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
CHAINS = {
    "finite": [("BITWISE_AND", 32767), ("MAXIMUM", 13056), ("MINIMUM", 17839), ("SUB_UNARY_SFPU", 13056)],
    "nan": [("BITWISE_AND", 32767), ("MAXIMUM", 32640), ("SUB_UNARY_SFPU", 32640)],
    "sign": [("BITWISE_AND", 32768), ("RIGHT_SHIFT", 2)],
}


def expected_indices(raw):
    raw = raw.to(torch.int64)
    mag = raw & 32767
    magnitude = mag.clamp(13056, 17839) - 13056 + mag.clamp(min=32640) - 32640
    return {"uint32_cast": raw, "magnitude_and_nan": magnitude, "complete_index": magnitude + ((raw & 32768) >> 2)}


def cases():
    magnitudes = [0, 1, 127, 128, 13055, 13056, 13057, 16884, 17838, 17839, 17840, 32639, 32640, 32641, 32703, 32767]
    signs = torch.tensor(magnitudes + [n + 32768 for n in magnitudes], dtype=torch.int64)
    return [
        ("singleton_boundaries", signs.reshape(1, 1, 32)),
        ("singleton_interleaved_signs", signs.reshape(2, 16).T.contiguous().reshape(1, 1, 32)),
        ("singleton_all_positive_nan", torch.arange(32641, 32673).reshape(1, 1, 32)),
        ("singleton_all_negative_nan", torch.arange(65504, 65536).reshape(1, 1, 32)),
        ("all_65536_codes", torch.arange(65536).reshape(1, 2048, 32)),
    ]


def compare(expected, actual, raw):
    actual = actual.to(torch.int64).reshape(expected.shape)
    mismatch = actual != expected
    positions = mismatch.flatten().nonzero().flatten()[:16]
    return {
        "elements": expected.numel(),
        "mismatches": int(mismatch.sum()),
        "max_abs_error": int((actual - expected).abs().max()),
        "actual_min": int(actual.min()),
        "actual_max": int(actual.max()),
        "passed": not bool(mismatch.any()),
        "first_mismatches": [
            {
                "flat_index": int(i),
                "raw_code": int(raw.flatten()[i]),
                "expected": int(expected.flatten()[i]),
                "actual": int(actual.flatten()[i]),
            }
            for i in positions
        ],
    }


def run(args):
    torch.set_num_threads(8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    sources = [
        "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp",
        "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp",
        "ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp",
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/unary_device_operation.cpp",
    ]
    params = {"physical_device_ids": [args.physical_device_id], "l1_small_size": args.l1_small_size}
    result = {
        "status": "running",
        "scope": "UINT16 upload, UINT32 typecast and two fused integer adds; no gather",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fused_source_sha256": hashlib.sha256((ROOT / "tt/fused_decoder.py").read_bytes()).hexdigest(),
        "native_source_sha256": {s: hashlib.sha256((REPO / s).read_bytes()).hexdigest() for s in sources},
        "device_open_parameters": params,
        "chains": CHAINS,
        "compact_table": {"lo": 13056, "hi": 17839, "sign_stride": 8192, "entries_per_head": 16384},
        "cases": [],
    }
    proof = ROOT / "doc/fused_decoder/endurance/gate_lut_plateau_cpu.json"
    if proof.exists():
        result["cpu_table_proof_sha256"] = hashlib.sha256(proof.read_bytes()).hexdigest()
    provenance = Path(str(output) + ".provenance.json")
    prov = json.loads(provenance.read_text())
    prov["device_open_parameters"] = params
    provenance.write_text(json.dumps(prov, indent=2) + "\n")

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    mesh = None
    stage = "import_ttnn"
    save()
    try:
        import ttnn

        stage = "open_device"
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **params)
        chains = {
            name: [ttnn.UnaryWithParam(getattr(ttnn.UnaryOpType, op), float(value)) for op, value in chain]
            for name, chain in CHAINS.items()
        }
        for name, raw in cases():
            row = {"name": name, "shape": list(raw.shape), "checks": {}}
            result["cases"].append(row)
            reference = expected_indices(raw)
            stage = name + ":upload_uint16"
            device_raw = ttnn.from_torch(
                raw.to(torch.int32).contiguous(),
                device=mesh,
                dtype=ttnn.uint16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            row["checks"]["upload_uint16"] = compare(raw, ttnn.to_torch(device_raw), raw)
            stage = name + ":uint32_cast"
            wide = ttnn.typecast(device_raw, ttnn.uint32)
            row["checks"]["uint32_cast"] = compare(reference["uint32_cast"], ttnn.to_torch(wide), raw)
            stage = name + ":magnitude_and_nan"
            magnitude = ttnn.add(
                wide,
                wide,
                dtype=ttnn.uint32,
                input_tensor_a_activations=chains["finite"],
                input_tensor_b_activations=chains["nan"],
            )
            row["checks"]["magnitude_and_nan"] = compare(reference["magnitude_and_nan"], ttnn.to_torch(magnitude), raw)
            stage = name + ":complete_index"
            mapped = ttnn.add(magnitude, wide, dtype=ttnn.uint32, input_tensor_b_activations=chains["sign"])
            row["checks"]["complete_index"] = compare(reference["complete_index"], ttnn.to_torch(mapped), raw)
            row["status"] = "passed" if all(v["passed"] for v in row["checks"].values()) else "numerical_failure"
            ttnn.synchronize_device(mesh)
            del mapped, magnitude, wide, device_raw
            save()
            print("COMPACT_DECAY_INDEX", name, row["status"], flush=True)
        result["status"] = "passed" if all(r["status"] == "passed" for r in result["cases"]) else "numerical_failure"
    except Exception:
        result.update(status="failed", failure_stage=stage, error=traceback.format_exc())
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
                result["device_closed"] = True
        except Exception:
            result.update(status="failed", close_error=traceback.format_exc())
            raise
        finally:
            save()
    return result["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-device-id", type=int, choices=[0], default=0)
    parser.add_argument("--l1-small-size", type=int, default=32768)
    args = parser.parse_args()
    if args.l1_small_size < 0:
        parser.error("l1-small-size must be nonnegative")
    raise SystemExit(0 if run(args) else 1)
