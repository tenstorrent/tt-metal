# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-layer readiness, buffered generation and same-workload warmed timing."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator
from models.common.readiness_check.run_prefill_check import _run_one_entry_prefill
from models.common.readiness_check.run_teacher_forcing import _run_one_entry
from models.common.readiness_check.schema import load_reference
from models.common.readiness_check.teacher_forcing import TokenAccuracy

ROOT = Path("models/autoports/google_gemma_4_26b_a4b_it")


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc/optimized_full_model")
    parser.add_argument("--performance-only", action="store_true")
    args = parser.parse_args()
    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        gen = build_generator(None, mesh, max_seq_len=8192)
        write(dest / "precision_runtime.json", gen.model.precision_summary())
        if not args.performance_only:
            refpath = ROOT / "readiness_aime24_chat.refpt"
            reference = load_reference(refpath)
            report = {"reference_sha256": hashlib.sha256(refpath.read_bytes()).hexdigest()}
            report["prefill"] = [
                _run_one_entry_prefill(generator=gen, entry=e, reference=reference) for e in reference.entries
            ]
            write(dest / "readiness.json", report)
            acc = TokenAccuracy(refpath)
            report["decode"] = [_run_one_entry(generator=gen, acc=acc, entry_idx=i) for i in range(acc.num_entries)]
            report["decode_metrics"] = dict(gen.metrics)
            write(dest / "readiness.json", report)
            assert all(
                r["top1"] >= 0.90 and r["top5"] >= 0.98 and r["top100"] == 1
                for p in ("prefill", "decode")
                for r in report[p]
            )
            controls = json.loads((ROOT / "doc/full_model/qualitative_hf.json").read_text())
            rows = []
            for row in controls["prompts"]:
                streaming = gen.generate(row["prompt_tokens"], 128)
                buffered = gen.generate(row["prompt_tokens"], 128, stop_on_eos=False)
                assert buffered[: len(streaming)] == streaming, row["id"]
                rows.append(
                    {
                        **row,
                        "tt_tokens": streaming,
                        "tt_completion": gen.tokenizer.decode(streaming, skip_special_tokens=False),
                        "buffered_tokens": buffered,
                        "buffered_matches_streaming_prefix": True,
                        "buffered_metrics": dict(gen.metrics),
                    }
                )
                write(dest / "qualitative_tt.json", {**controls, "prompts": rows})
                print("QUALITATIVE", row["id"], rows[-1]["tt_completion"], flush=True)
            prior_ar = ROOT / "doc/full_model/autoregressive"
            ar = dest / "autoregressive"
            ar.mkdir(exist_ok=True)
            metadata = json.loads((prior_ar / "autoregressive_meta.json").read_text())
            tokens = gen.generate(metadata["prompt_token_ids"], 128, stop_on_eos=False)
            metadata["tt"] = {"token_ids": tokens, "num_tokens": len(tokens)}
            metadata["hf_control_source"] = str(prior_ar)
            metadata["tt_metrics"] = dict(gen.metrics)
            metadata["prompt_file"] = str(ar / "prompt.txt")
            write(ar / "autoregressive_meta.json", metadata)
            for name in ("prompt.txt", "prompt_format.json", "hf_completion.txt"):
                (ar / name).write_text((prior_ar / name).read_text())
            (ar / "tt_completion.txt").write_text(gen.tokenizer.decode(tokens, skip_special_tokens=True))
        prompt = gen.tokenizer.encode("This is a document about numbers and arithmetic. " * 1024)[:4096]
        assert len(prompt) == 4096
        results = {}
        sequences = {}
        for buffered in (False, True):
            label = "streaming" if not buffered else "buffered"
            gen.generate(prompt, 128, stop_on_eos=False, buffer_tokens=buffered)
            samples = []
            for _ in range(3):
                tokens = gen.generate(prompt, 128, stop_on_eos=False, buffer_tokens=buffered)
                samples.append({**gen.metrics, "concurrency": 1})
                if label in sequences:
                    assert tokens == sequences[label]
                sequences[label] = tokens
            results[label] = {
                "samples": samples,
                "median_ttft_ms": statistics.median(r["ttft_ms"] for r in samples),
                "median_decode_tps": statistics.median(r["decode_tps"] for r in samples),
            }
            print("PERFORMANCE", label, results[label], flush=True)
            write(dest / "performance_comparison.json", results)
        assert sequences["streaming"] == sequences["buffered"]
        write(dest / "precision_runtime.json", gen.model.precision_summary())
        final = results["buffered"]
        write(
            dest / "performance.json",
            {
                "source": "autoregressive",
                "input_tokens": 4096,
                "output_tokens": 128,
                "batch": 1,
                "concurrency": 1,
                "ttft_ms": final["median_ttft_ms"],
                "decode_tps": final["median_decode_tps"],
                "aggregation": "median of three warmed requests",
                "tokens": sequences["buffered"],
                "streaming_token_equivalence": True,
                "counters": final["samples"][-1]["counters"],
                "samples": "performance_comparison.json",
                "reduced_probe": False,
            },
        )
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
