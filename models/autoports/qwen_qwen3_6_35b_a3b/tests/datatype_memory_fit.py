# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-capacity allocation and short preparation; never a long-context run.

This checks a P1 reference's memory eligibility. No performance claim is made.
Use an external timeout and an idle assigned mesh.
"""

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefill-lengths", default="32,33,161", help="Preparation geometries; include2048 for MoE peak-chunk coverage"
    )
    args = parser.parse_args()
    lengths = [int(value) for value in args.prefill_lengths.split(",")]
    assert lengths and all(0 < value <= 8192 for value in lengths)
    assert not args.output.exists(), "Preserve each allocation experiment"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import build_generator

    result = dict(
        status="running",
        command=sys.argv,
        context=262144,
        hardware="4 Blackhole p300c",
        mesh=[1, 4],
        purpose="Full-capacity allocation plus short preparation; no long prefill or throughput claim",
        precision_config=str(args.precision_config),
        precision_sha256=hashlib.sha256(args.precision_config.read_bytes()).hexdigest(),
        prefill_preparation_lengths=lengths,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*.py")
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    mesh = generator = None
    save()
    try:
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        generator = build_generator(ROOT, mesh, max_seq_len=262144, precision_config=str(args.precision_config))
        result["runtime_precision"] = generator.model.precision_summary()
        result["loaded_census"] = generator.model.census("precision_reference_full_capacity_loaded")
        save()
        generator.prepare(prefill_cases=tuple((length, 0) for length in lengths))
        result["prepared_census"] = generator.model.census("precision_reference_full_capacity_prepared")
        census = result["prepared_census"]
        result["free_bytes_per_device"] = census["banks"] * census["total_bytes_free_per_bank"]
        result["trace_ids"] = {key: int(value) for key, value in generator.traces.items()}
        result["runtime_counters"] = dict(generator.counters)
        result["status"] = "pass"
        save()
        print(json.dumps({k: result[k] for k in ("status", "free_bytes_per_device")}), flush=True)
    except BaseException:
        result["status"] = "runtime_error"
        result["error"] = traceback.format_exc()
        save()
        raise
    finally:
        if generator is not None:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        result["device_closed"] = mesh is not None
        save()


if __name__ == "__main__":
    main()
