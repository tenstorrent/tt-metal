# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One full-model policy, standard readiness scoring, audited traced decode."""

import argparse
import copy
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-kwargs-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from readiness_check.run_prefill_check import _run_one_entry_prefill
    from readiness_check.run_teacher_forcing import _run_one_entry
    from readiness_check.schema import load_reference
    from readiness_check.teacher_forcing import TokenAccuracy

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator

    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    kwargs = json.loads(args.build_kwargs_file.read_text())
    result = {
        "status": "running",
        "build_kwargs": kwargs,
        "command": [
            sys.executable,
            "-m",
            "models.autoports.ornith_ai_ornith_1_5_9b.tests.datatype_evaluate",
            *sys.argv[1:],
        ],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "tt").glob("*.py")
        },
        "hardware": "4 x P300c Blackhole, chips 0-3",
        "mesh": [1, 4],
        "fabric": "FABRIC_1D",
        "measurement_regime": "serial; warmed traced teacher forcing; explicit forced token uploads and caller readback",
        "references": {},
    }
    gen = mesh = None
    started = time.time()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=256 * 1024 * 1024)
        gen = build_generator(root, mesh, **kwargs)
        result["runtime_policy"] = gen.model.precision_summary()
        result["config_id"] = gen.model.precision_policy["id"]
        result["context"] = gen.max_seq_len
        for name, path in (
            ("aime24", root / "readiness_aime24_chat.refpt"),
            ("natural_text", root / "doc/datatype_sweep/natural_text.refpt"),
        ):
            reference = load_reference(path)
            entry = reference.entries[0]
            row = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            row["prefill"] = _run_one_entry_prefill(generator=gen, entry=entry, reference=reference)
            samples = []
            # The warmup follows the same logical request; captures and compilation
            # are excluded from all ranking samples, identically for every policy.
            for repeat in range(3):
                gen.reset()
                acc = TokenAccuracy(reference)
                score = _run_one_entry(generator=gen, acc=acc, entry_idx=0)
                score["generator_metrics"] = copy.deepcopy(gen.last_generation_metrics)
                if repeat:
                    delta = score["generator_metrics"]["counter_delta"]
                    if delta["model_replays"] != entry.num_generated - 1 or delta["captures"] != 0:
                        raise RuntimeError(f"Decode timing is not warmed trace replay: {delta}")
                    samples.append(score)
                else:
                    row["warmup"] = score
            row["teacher_forcing_samples"] = samples
            row["teacher_forcing"] = dict(samples[0])
            for key in ("ttft_ms", "decode_t/s/u"):
                row["teacher_forcing"][key] = statistics.median(s[key] for s in samples)
            for phase in ("prefill", "teacher_forcing"):
                s = row[phase]
                s["top1_binomial_95_half_width"] = 1.96 * math.sqrt(s["top1"] * (1 - s["top1"]) / s["total"])
                s["accuracy_pass"] = s["top1"] >= 0.90 and s["top5"] >= 0.98 and s["top100"] >= 1.0
            result["references"][name] = row
            print(
                "SCORE",
                name,
                row["prefill"]["top1"],
                row["teacher_forcing"]["top1"],
                row["teacher_forcing"]["decode_t/s/u"],
                flush=True,
            )
        result["memory_prepared"] = gen.model.memory_view()
        result["trace_counters"] = dict(gen.counters)
        result["status"] = (
            "pass"
            if all(r[p]["accuracy_pass"] for r in result["references"].values() for p in ("prefill", "teacher_forcing"))
            else "accuracy_fail"
        )
    except Exception as error:
        result["status"] = "runtime_error"
        result["error"] = repr(error)
        raise
    finally:
        result["wall_seconds"] = time.time() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if gen is not None:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
