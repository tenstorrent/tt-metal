# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model AIME accuracy after nonaligned split prefill; not a timing benchmark."""

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_teacher_forcing as teacher

import ttnn

from ..tt.generator import build_generator

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    report = dict(
        status="running",
        command=shlex.join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]),
        hardware="4x Blackhole p300c",
        mesh=[1, 4],
        context=524288,
        layers=48,
        reference_sha256=hashlib.sha256((ROOT / "readiness_aime24_chat.refpt").read_bytes()).hexdigest(),
        regime="tracked traced teacher forcing after split prefill; correctness only, not Pareto ranking",
        rows=[],
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*.py")
        },
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        assert not os.environ.get("K2_PRECISION_CONFIG")
        gen = build_generator(ROOT, mesh, max_seq_len=524288)
        report["policy"] = gen.model.precision
        gen.capture()
        original = gen.prefill_forward
        for split in (33, 129):

            def split_prefill(tokens, **kw):
                assert kw.get("enable_trace") is True and kw["prompt_lens"] == [156]
                prefix_kw = {**kw, "prompt_lens": [split], "sampling_mode": "host"}
                original(tokens, **prefix_kw)
                return original(tokens, **{**kw, "start_pos": [split]})

            before = gen.counters.copy()
            with (
                patch.object(gen, "prefill_forward", split_prefill),
                patch.object(gen, "teardown", lambda: None),
                patch.object(teacher, "_import_build_generator", lambda _: lambda **_: gen),
            ):
                stats = teacher.run_teacher_forcing(
                    model_dir=ROOT, reference_path=ROOT / "readiness_aime24_chat.refpt", mesh_device=mesh
                )
            delta = {k: gen.counters[k] - before.get(k, 0) for k in gen.counters}
            report["rows"].append(dict(split=split, end=156, stats=stats, counters=delta))
            save()
            assert delta["model_replays"] == 99 and delta["sampling_replays"] == 100 and delta["token_readbacks"] == 100
            assert all(
                r["total"] == 100 and r["top1"] >= 0.90 and r["top5"] >= 0.98 and r["top100"] == 1 for r in stats
            ), stats
        report["status"] = "pass"
        save()
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
