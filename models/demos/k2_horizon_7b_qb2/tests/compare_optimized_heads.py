"""Precision-locked full-stack ABBA comparison of legal terminal geometries."""

import gc
import json
import statistics
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_check, run_teacher_forcing

import ttnn

from ..tt.full_model_policy import stage6_precision_policy
from ..tt.generator import K2Generator
from ..tt.model import K2Model

OUT = Path("models/demos/k2_horizon_7b_qb2/doc/optimized_full_model")
MODEL_DIR = OUT.parents[1]


def main():
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        gen = K2Generator(
            mesh, head_k=2, head_split_size=16384, decoder_policies={i: stage6_precision_policy(i) for i in range(36)}
        )
        heads = {"k2_s16384": gen.model.head_decode}
        candidate = K2Model(mesh, override_num_layers=1, head_k=8, head_split_size=8192)
        heads["k8_s8192"] = candidate.head_decode
        del candidate
        gc.collect()
        prompt = gen.tokenizer.encode("The sky appears blue because sunlight scatters in the atmosphere. " * 20)[:128]
        result = {
            "shared_stack": True,
            "layers": 36,
            "input_cores": 16,
            "readers_per_bank": 1,
            "weight_dtype": "BFP8",
            "fidelity": "HiFi2",
            "fp32_accumulation": True,
            "order": ["k2_s16384", "k8_s8192", "k8_s8192", "k2_s16384"],
            "records": [],
            "accuracy": {},
        }
        for name in result["order"]:
            gen._release_traces()
            gen.model.head_decode = heads[name]
            gen.generate(prompt, 128)
            runs = []
            for _ in range(3):
                tokens = gen.generate(prompt, 128)
                runs.append(gen.last_perf.copy())
            result["records"].append(
                {
                    "name": name,
                    "runs": runs,
                    "tokens": tokens,
                    "median_decode_ms": statistics.median(r["decode_seconds"] * 1000 / 127 for r in runs),
                    "median_ttft_ms": statistics.median(r["ttft_seconds"] * 1000 for r in runs),
                    "capacity_tokens": gen.capacity,
                }
            )
            (OUT / "head_full36_abba.json").write_text(json.dumps(result, indent=2) + "\n")
            print("HEAD", name, result["records"][-1]["median_decode_ms"], flush=True)
        for name, head in heads.items():
            gen._release_traces()
            gen.model.head_decode = head
            result["accuracy"][name] = {}
            for label, runner, method in [
                ("prefill", run_prefill_check, "run_prefill_check"),
                ("teacher", run_teacher_forcing, "run_teacher_forcing"),
            ]:
                with patch.object(runner, "_import_build_generator", return_value=lambda **kw: gen):
                    scores = getattr(runner, method)(
                        model_dir=MODEL_DIR, reference_path=MODEL_DIR / "readiness_aime24_chat.refpt", mesh_device=mesh
                    )
                result["accuracy"][name][label] = scores
                (OUT / "head_full36_abba.json").write_text(json.dumps(result, indent=2) + "\n")
                assert all(row["top5"] >= 0.98 and row["top100"] == 1 for row in scores)
        result["complete"] = True
        (OUT / "head_full36_abba.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
