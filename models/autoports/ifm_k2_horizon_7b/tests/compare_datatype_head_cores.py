"""ABBA full-model check of a small precision-locked head-layout difference."""

import json
import statistics
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_check, run_teacher_forcing

import ttnn

from ..tt.generator import build_generator

MODEL = Path("models/autoports/ifm_k2_horizon_7b")
OUTPUT = MODEL / "doc/datatype_sweep/head_cores_full36.json"


def main():
    torch.set_num_threads(16)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    result = {
        "layers": 36,
        "order": [16, 8, 8, 16],
        "records": [],
        "scope": "geometry control for the same selected precision policy; coherent direct head input, no intermediate layout",
    }
    try:
        gen = build_generator(MODEL, mesh)
        result["precision_config"] = gen.model.precision_config
        reference = MODEL / "readiness_aime24_chat.refpt"
        prefill = {}
        for cores in result["order"]:
            gen._release_traces()
            memory = ttnn.create_sharded_memory_config(
                (32, 4096 // cores),
                core_grid=ttnn.num_cores_to_corerangeset(cores, mesh.compute_with_storage_grid_size(), True),
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            gen.model.head_input_memory = gen.model.head_decode.config.input_memcfg = memory
            if cores not in prefill:
                with patch.object(run_prefill_check, "_import_build_generator", return_value=lambda **kw: gen):
                    prefill[cores] = run_prefill_check.run_prefill_check(
                        model_dir=MODEL, reference_path=reference, mesh_device=mesh
                    )
            record = {"head_input_cores": cores, "prefill": prefill[cores], "runs": []}
            for iteration in range(10):
                with patch.object(run_teacher_forcing, "_import_build_generator", return_value=lambda **kw: gen):
                    scores = run_teacher_forcing.run_teacher_forcing(
                        model_dir=MODEL, reference_path=reference, mesh_device=mesh
                    )
                perf = gen.last_perf.copy()
                counters = perf["steady_state_counters"]
                assert counters["model_replays"] == counters["sampling_replays"] == 99
                assert counters["token_refreshes"] == counters["token_readbacks"] == 99
                assert all(
                    r["top1"] >= 0.90 and r["top5"] >= 0.98 and r["top100"] == 1 for r in [*scores, *prefill[cores]]
                )
                if iteration:
                    record["runs"].append({"scores": scores, "perf": perf})
            record["median_tps"] = statistics.median(
                r["perf"]["decode_tokens_per_second_per_user"] for r in record["runs"]
            )
            record["capacity_tokens"] = gen.capacity
            result["records"].append(record)
            OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
            print("HEAD_CORES", cores, record["median_tps"], flush=True)
        result["medians"] = {
            str(cores): statistics.median(
                r["perf"]["decode_tokens_per_second_per_user"]
                for block in result["records"]
                if block["head_input_cores"] == cores
                for r in block["runs"]
            )
            for cores in [8, 16]
        }
        result["pass"] = True
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
