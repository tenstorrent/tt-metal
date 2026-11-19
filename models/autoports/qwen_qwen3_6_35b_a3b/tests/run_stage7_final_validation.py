# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serial final correctness gates; no performance/profile/long-context jobs.

Source lane-env.sh and activate python_env first. --dry-run touches no devices.
Every failure stops the queue; outputs still require a fresh qualitative review.
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
MODULE = "models.autoports.qwen_qwen3_6_35b_a3b.tests"


def aime_worker(out):
    import torch

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_batched_teacher import load_fixture
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel

    out.mkdir(parents=True, exist_ok=False)
    entry, reference = load_fixture(ROOT / "readiness_aime24_chat.refpt")
    ids = entry.prompt_tokens[0].tolist()
    assert ids == json.loads((ROOT / "doc/full_model/hf_reference/aime24_prompt_tokens.json").read_text())
    assert len(ids) == 161
    torch.set_num_threads(8)
    t.set_fabric_config(t.FabricConfig.FABRIC_1D)
    mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        model = QwenModel(mesh, max_seq_len=512)
        assert list(model.layer_indices) == list(range(40))
        gen = QwenGenerator(model, host_sampling=False)
        generated = gen.generate(ids, 100, temperature=0.0, top_k=1, top_p=1.0, seed=0)
        hf = entry.generated_tokens[0].tolist()
        record = dict(
            layers=list(model.layer_indices),
            mesh=[1, 4],
            reference=reference,
            prompt_mode="chat",
            prompt_token_ids=ids,
            host_sampling=gen.host_sampling,
            generation=dict(max_new_tokens=100, temperature=0.0, top_k=1, top_p=1.0, seed=0),
            tt_token_ids=generated,
            hf_token_ids=hf,
            counters=dict(gen.counters),
            trace_ids={k: int(v) for k, v in gen.traces.items()},
            quality_verdict="pending independent reading; execution is not quality acceptance",
        )
        (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        (out / "tt_completion.txt").write_text(gen.tokenizer.decode(generated, skip_special_tokens=False))
        (out / "hf_completion.txt").write_text(gen.tokenizer.decode(hf, skip_special_tokens=False))
    finally:
        if gen is not None:
            gen.close()
        t.close_mesh_device(mesh)


def commands(out):
    py = [sys.executable, "-u"]

    def module(name, *args):
        return [*py, "-m", f"{MODULE}.{name}", *map(str, args)]

    def entry(name, seconds, command, hardware=True):
        return dict(
            name=name,
            seconds=seconds,
            command=command,
            hardware=hardware,
            watcher=10 if name == "replay_all40" else None,
        )

    return [
        entry(
            "positive_control",
            1800,
            module(
                "run_full_model_probe", "--generate", "--positive-control", "--output", out / "positive_control.json"
            ),
        ),
        entry(
            "prefill_accuracy",
            1800,
            module(
                "run_full_model_readiness",
                "--mode",
                "prefill",
                "--context",
                512,
                "--output",
                out / "prefill_accuracy.json",
            ),
        ),
        entry(
            "teacher_accuracy",
            1800,
            module(
                "run_full_model_readiness",
                "--mode",
                "teacher",
                "--context",
                512,
                "--output",
                out / "teacher_accuracy.json",
            ),
        ),
        entry(
            "aime_autoregressive",
            1800,
            [*py, str(Path(__file__).resolve()), "--aime-worker", str(out / "aime_autoregressive")],
        ),
        entry(
            "qualitative",
            1800,
            [
                *py,
                str(ROOT / "doc/full_model/qualitative/run_tt_controls.py"),
                "--mode",
                "suite",
                "--hf-controls",
                str(ROOT / "doc/full_model/qualitative/hf_greedy_outputs.json"),
                "--output-dir",
                str(out / "qualitative"),
                "--max-new-tokens",
                "128",
                "--context",
                "1024",
            ],
        ),
        entry(
            "degenerate_output_check",
            120,
            [
                *py,
                "-m",
                "readiness_check.check_degenerate_output",
                str(out / "qualitative"),
                "--hf-model",
                "Qwen/Qwen3.6-35B-A3B",
                "--scope",
                "autoregressive",
                "--missing-artifacts",
                "critical",
                "--json",
                str(out / "qualitative/degenerate_output_check.json"),
            ],
            False,
        ),
        entry(
            "b32",
            3600,
            module(
                "run_full_model_probe",
                "--all-layers",
                "--generate",
                "--batch-size",
                32,
                "--batch-probe",
                "--context",
                1024,
                "--output",
                out / "b32.json",
            ),
        ),
        entry(
            "replay_all40",
            5400,
            module(
                "run_generator_replay_probe",
                "--all-layers",
                "--section",
                "all",
                "--context",
                2048,
                "--steps",
                4,
                "--output",
                out / "replay_all40.json",
            ),
        ),
        entry(
            "serving_commands_all40",
            3600,
            module(
                "run_serving_commands_probe",
                "--all-layers",
                "--context",
                1024,
                "--output",
                out / "serving_commands_all40.json",
            ),
        ),
        entry(
            "sampler_acceptance",
            900,
            module(
                "run_sampler_contract_probe",
                "--mode",
                "acceptance",
                "--iterations",
                100,
                "--output",
                out / "sampler_acceptance.json",
            ),
        ),
    ]


def source_hashes():
    paths = set((ROOT / "tt").glob("*.py")) | set((ROOT / "tests").glob("*.py"))
    paths.update((REPO / "models/common/sampling").rglob("*.py"))
    paths.add(ROOT / "doc/full_model/qualitative/run_tt_controls.py")
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aime-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.aime_worker:
        aime_worker(args.aime_worker)
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    out = args.output_dir.resolve()
    jobs = commands(out)
    if args.dry_run:
        for job in jobs:
            prefix = ["env", "TT_METAL_WATCHER=10"] if job["watcher"] else []
            print(shlex.join([*prefix, "timeout", str(job["seconds"]), *job["command"]]))
        return
    assert os.environ.get("TT_MODEL_BRINGUP_ROOT"), "Source lane-env.sh before running"
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        HF_HUB_OFFLINE="1",
        TT_METAL_TRACE_ALLOC_TRACKING="1",
        TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
    )
    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TTNN_OP_PROFILER", "TT_METAL_PROFILER_TRACE_TRACKING"):
        env.pop(key, None)
    frozen = source_hashes()
    record = dict(
        passed=False,
        source_sha256=frozen,
        jobs=[],
        scope="serial correctness execution only; qualitative review/perf/profile/context/review still required",
        environment={
            key: env.get(key)
            for key in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_MODEL_BRINGUP_ROOT",
            )
        },
    )
    (out / "source_provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    try:
        for job in jobs:
            assert source_hashes() == frozen, "Source changed during final validation; stop and use fresh evidence"
            if job["hardware"]:
                devices = sorted(str(p) for p in Path("/dev/tenstorrent").glob("*"))
                assert devices, "No assigned TT device nodes found"
                with (out / "logs" / f"{job['name']}_ownership.log").open("w") as log:
                    idle = subprocess.run(["timeout", "30", "fuser", *devices], stdout=log, stderr=log)
                assert idle.returncode == 1, "Device holder or ownership inspection error; stop without reset"
                if not record["jobs"]:
                    with (out / "logs/health.log").open("w") as log:
                        subprocess.run(
                            ["timeout", "60", "tt-smi", "-ls", "--local"], stdout=log, stderr=log, check=True
                        )
            command = ["timeout", str(job["seconds"]), *job["command"]]
            row = dict(job, started_unix=time.time(), bounded_command=command)
            record["jobs"].append(row)
            print(f"START {job['name']}: {shlex.join(command)}", flush=True)
            with (out / "commands.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            job_env = env.copy()
            if job["watcher"]:
                job_env["TT_METAL_WATCHER"] = str(job["watcher"])
            try:
                with (out / "logs" / f"{job['name']}.log").open("w") as log:
                    completed = subprocess.run(command, cwd=REPO, env=job_env, stdout=log, stderr=log)
            finally:
                if job["watcher"]:
                    watcher = REPO / "generated/watcher"
                    row["watcher_artifacts_present"] = watcher.is_dir()
                    if watcher.is_dir():
                        shutil.copytree(watcher, out / "watcher_replay")
                        row["watcher_artifacts"] = "watcher_replay"
            row.update(returncode=completed.returncode, elapsed_seconds=time.time() - row["started_unix"])
            assert (
                completed.returncode == 0
            ), f"{job['name']} failed/timeout; inspect log and device evidence before any next job"
            if job["watcher"]:
                assert row["watcher_artifacts_present"], "Watched replay produced no watcher artifact directory"
            if job["name"] == "positive_control":
                assert json.loads((out / "positive_control.json").read_text())["positive_control_rejected"] is True
            assert source_hashes() == frozen, "Source changed during a gate; artifact cannot qualify frozen source"
            print(f"DONE {job['name']} ({row['elapsed_seconds']:.1f}s)", flush=True)
        record["passed"] = True
    finally:
        (out / "execution_summary.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
