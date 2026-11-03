# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized full-model precision evaluation with explicit trace evidence."""

import argparse
import gzip
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_check as prefill
from readiness_check import run_teacher_forcing as teacher

import ttnn

from ..tt.generator import build_generator
from ..tt.precision import runtime_summary
from .full_model_performance import memory

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(8)
    report = dict(
        command=shlex.join([sys.executable, "-u", "-m", __spec__.name, *sys.argv[1:]]),
        hardware="4x Blackhole p300c",
        mesh=[1, 4],
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        config_path=args.config,
        layers=2 if args.smoke else 48,
        context=524288,
        reference=str(ROOT / "readiness_aime24_chat.refpt"),
        reference_sha256=hashlib.sha256((ROOT / "readiness_aime24_chat.refpt").read_bytes()).hexdigest(),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*.py")
        },
        creation_source_sha256=hashlib.sha256(
            Path("ttnn/cpp/ttnn/operations/creation/creation.cpp").read_bytes()
        ).hexdigest(),
        runtime_library_sha256=hashlib.sha256(Path("build_Release/lib/_ttnncpp.so").read_bytes()).hexdigest(),
        tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        measurement_regime="warmed traced teacher forcing; B1, AIME24 chat 156 prompt/100 generated, full524288 allocation, explicit HF-token feedback and token readback",
        status="running",
    )
    source_dir = DOC / "sources"
    source_dir.mkdir(exist_ok=True)
    for relative, digest in report["source_sha256"].items():
        archive = source_dir / (digest + ".py.gz")
        if not archive.exists():
            archive.write_bytes(gzip.compress((ROOT / relative).read_bytes(), mtime=0))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    gen = None
    try:
        gen = build_generator(ROOT, mesh, precision_config=args.config, override_num_layers=report["layers"])
        report["policy"] = gen.model.precision
        report["runtime"] = runtime_summary(gen.model)
        report["cache_dtypes"] = sorted({str(t.dtype) for pair in gen.kv_cache for t in pair})
        report["constructed_memory"] = memory(mesh)
        save()
        gen.capture()
        report["captured_memory"] = memory(mesh)
        report["model_trace"] = str(gen.model_trace)
        save()
        factory = lambda **_: gen
        if args.smoke:
            prompt = json.loads((ROOT / "doc/full_model/reference_metadata.json").read_text())["prompt_tokens"]
            report["nonaligned"] = []
            for length in (31, 32, 33, 127, 128, 129, 156):
                ids = prompt[:length]
                a = gen.generate(ids, 3, stop_on_eos=False)
                b = gen.generate(ids, 3, stop_on_eos=False)
                assert a == b, (length, a, b)
                report["nonaligned"].append(dict(length=length, tokens=a, repeat_equal=True))
            report["status"] = "smoke_pass"
        else:
            with patch.object(gen, "teardown", lambda: None):
                with patch.object(prefill, "_import_build_generator", lambda _: factory):
                    report["prefill"] = prefill.run_prefill_check(
                        model_dir=ROOT, reference_path=ROOT / "readiness_aime24_chat.refpt", mesh_device=mesh
                    )
                with patch.object(teacher, "_import_build_generator", lambda _: factory):
                    teacher.run_teacher_forcing(
                        model_dir=ROOT, reference_path=ROOT / "readiness_aime24_chat.refpt", mesh_device=mesh
                    )
                    report["teacher_forcing"] = []
                    for _ in range(args.repeats):
                        gen.reset()
                        before = dict(gen.counters)
                        tid = gen.model_trace
                        rows = teacher.run_teacher_forcing(
                            model_dir=ROOT, reference_path=ROOT / "readiness_aime24_chat.refpt", mesh_device=mesh
                        )
                        delta = {k: v - before.get(k, 0) for k, v in gen.counters.items()}
                        assert gen.model_trace == tid and delta["model_replays"] == 99, delta
                        report["teacher_forcing"].append(dict(rows=rows, counters=delta, trace_id=str(tid)))
                        save()
            rows = [r["rows"][0] for r in report["teacher_forcing"]]
            assert len({(r["top1"], r["top5"], r["top100"]) for r in rows}) == 1
            report.update({k: rows[0][k] for k in ("top1", "top5", "top100", "total")})
            report["ttft_ms"] = statistics.median(r["ttft_ms"] for r in rows)
            report["decode_t/s/u"] = statistics.median(r["decode_t/s/u"] for r in rows)
            report["trace_verified"] = True
            report["status"] = (
                "pass"
                if all(r["top1"] >= 0.90 and r["top5"] >= 0.98 and r["top100"] == 1 for r in rows + report["prefill"])
                else "accuracy_fail"
            )
        report["counters"] = dict(gen.counters)
        save()
    except Exception:
        report["status"] = "runtime_fail"
        report["error"] = traceback.format_exc()
        save()
        raise
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
