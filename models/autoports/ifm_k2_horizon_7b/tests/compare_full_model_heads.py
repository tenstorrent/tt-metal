"""Same all36 stack, precision and workload; compare two real LM-head layouts."""

import gc
import json
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_check, run_teacher_forcing

import ttnn

from ..tt.generator import K2Generator
from ..tt.model import K2Model
from .benchmark_full_model import run as benchmark
from .run_qualitative_extended import run as quality

DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/full_model")
OUT = DOC / "head_full_model"
MODEL_DIR = DOC.parents[1]


def main():
    torch.set_num_threads(8)
    OUT.mkdir(exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(mesh, head_split_size=32768, head_workers=2, head_k=2)
        heads = {"baseline": gen.model.head_decode}
        # Use the production constructor's exact packing and program contract.
        # Only retain its terminal helper. The extra diagnostic layer, embedding,
        # prefill head and RoPE storage are freed before tracing or measurement.
        candidate = K2Model(mesh, override_num_layers=1, head_split_size=16384, head_workers=1, head_k=2)
        heads["candidate"] = candidate.head_decode
        del candidate
        gc.collect()
        ttnn.synchronize_device(mesh)
        result = {
            "layers": gen.model.num_layers,
            "shared_stack": True,
            "policy": "BFP8/HiFi2/FP32 head; inherited decoder and common split sampler unchanged",
            "head_configs": {
                "baseline": {"split_size": 32768, "k_block": 2, "workers_per_bank": 2},
                "candidate": {"split_size": 16384, "k_block": 2, "workers_per_bank": 1},
            },
            "order": ["baseline", "candidate", "candidate", "baseline"],
            "measurements": [],
            "accuracy": {},
        }
        outputs = {}
        for index, name in enumerate(result["order"]):
            # Invalidate before replacing immutable trace resources. Both head
            # weight sets already exist; request setup recompiles/captures normally.
            gen._release_traces()
            gen.model.head_decode = heads[name]
            print("HEAD_VARIANT", name, index, flush=True)
            perf = benchmark(gen, OUT / f"{name}_{index}_performance.json")
            result["measurements"].append({"name": name, "order_index": index, "performance": perf})
            if name not in outputs:
                result["accuracy"][name] = {}
                for label, runner, method in [
                    ("prefill", run_prefill_check, "run_prefill_check"),
                    ("teacher", run_teacher_forcing, "run_teacher_forcing"),
                ]:
                    with patch.object(runner, "_import_build_generator", return_value=lambda **kw: gen):
                        result["accuracy"][name][label] = getattr(runner, method)(
                            model_dir=MODEL_DIR,
                            reference_path=MODEL_DIR / "readiness_aime24_chat.refpt",
                            mesh_device=mesh,
                        )
                outputs[name] = quality(gen, output_name=f"head_full_model/quality_{name}.json")
            (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        result["quality_token_equivalence"] = {
            a["id"]: a["prompt_token_ids"] == b["prompt_token_ids"] and a["tt_token_ids"] == b["tt_token_ids"]
            for a, b in zip(outputs["baseline"], outputs["candidate"], strict=True)
        }
        result["complete"] = True
        (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        print("HEAD_COMPARISON_COMPLETE", json.dumps(result["quality_token_equivalence"]), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
