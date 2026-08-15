# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model evaluation of one precision candidate.

One process, one candidate, one 52-layer build.  The candidate's artifact is
installed as ``tt.precision_config.SELECTED_PRECISION_CONFIG_PATH`` for the life
of the process, so ``build_generator(model_dir, mesh_device)`` with **no knobs at
all** -- which is what the readiness runners call -- constructs it.  A candidate
is therefore measured through the same default path the selected config will
ship on, not through a special harness argument.

What it records, per candidate:

* the **realised** precision policy read off the built model, and the diff
  against the requested artifact (``check_propagation``).  A candidate whose
  policy did not propagate is not a measurement;
* ``run_prefill_check`` and ``run_teacher_forcing`` top-1 / top-5 / top-100
  against the main AIME24 chat-template reference at 100 generated tokens;
* teacher-forcing TTFT and **traced** decode t/s/u, repeated ``--rounds`` times.
  The runner drives ``generate(..., enable_trace=True)``, and the generator's own
  trace-replay counter is recorded per round, so "traced" is a counter rather
  than a claim;
* the traced logits-only decode replay, as a low-variance cross-check on the
  teacher-forcing ranking metric (teacher forcing restages a token per step, so
  its spread is ~3 % where the replay's is ~0.2 %);
* the per-device DRAM budget at this cache dtype, which is what the context
  contract is recomputed from.

Usage::

    python doc/datatype_sweep/bench/sweep.py --config c05-kv4 --rounds 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt import precision_config as pc  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    build_generator,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.model import dram_capacity_bytes  # noqa: E402
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    close_multichip_mesh,
    open_multichip_mesh,
)

CONFIG_DIR = ROOT / "doc/datatype_sweep/configs"
OUT = ROOT / "doc/datatype_sweep"
RUNS = OUT / "runs"
REFERENCE = "readiness_aime24_chat.refpt"


def say(*args) -> None:
    print(*args, flush=True)


def capture_stdout(fn, *args, **kwargs):
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        value = fn(*args, **kwargs)
    text = buffer.getvalue()
    print(text, end="", flush=True)
    return value, text


def register_hf_shims(model_id: str) -> None:
    """The readiness runners need the port's HF auto-class registration."""
    sys.path.insert(0, str(ROOT / "doc/full_model/bench"))
    import importlib.util

    path = ROOT / "doc/full_model/bench/readiness_cli.py"
    spec = importlib.util.spec_from_file_location("muse_glimmer_readiness_cli", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "register_hf_shims"):
        module.register_hf_shims(model_id)


def environment() -> dict:
    def git(*args) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # pragma: no cover
            return ""

    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "host": platform.node(),
        "python": platform.python_version(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="config id under doc/datatype_sweep/configs/")
    parser.add_argument("--rounds", type=int, default=5, help="teacher-forcing repeats")
    parser.add_argument("--replays", type=int, default=64, help="decode-trace replays per logits-only round")
    parser.add_argument("--max-seq-len", type=int, default=131072)
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    config_path = CONFIG_DIR / f"{args.config}.json"
    config = pc.load_precision_config(config_path)
    RUNS.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    command = (
        f"python doc/datatype_sweep/bench/sweep.py --config {args.config} "
        f"--rounds {args.rounds} --replays {args.replays}"
    )
    result: dict = {
        "config_id": config["config_id"],
        "config_path": str(config_path.relative_to(REPO)),
        "description": config.get("description", ""),
        "reference": args.reference,
        "command": command,
        "environment": environment(),
        "max_seq_len": args.max_seq_len,
        "max_batch_size": args.max_batch_size,
        "status": "started",
    }

    from models.autoports.meta_models_muse_glimmer_30b.tt.generator import HF_MODEL_ID

    register_hf_shims(HF_MODEL_ID)

    mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    generator = None
    try:
        result["hardware"] = {
            "num_devices": mesh.get_num_devices(),
            "mesh_shape": list(mesh.shape),
            "arch": str(mesh.arch()),
            "compute_with_storage_grid_size": [
                mesh.compute_with_storage_grid_size().x,
                mesh.compute_with_storage_grid_size().y,
            ],
            "per_device_dram_capacity_bytes": dram_capacity_bytes(mesh),
        }
        # Point the *module-level* selected-config path at this candidate rather
        # than passing ``precision_config=`` to one call.  Two reasons, and both
        # matter for the evidence:
        #
        # * the readiness runners load ``tt/generator.py`` by path and call
        #   ``build_generator(model_dir, mesh_device)`` themselves, with no knobs.
        #   Only the module-level default reaches that call, so this is what makes
        #   the accuracy numbers come from the candidate rather than from whatever
        #   ``selected_precision_config.json`` happens to hold;
        # * it means the candidate is measured through the *default* construction
        #   path, which is the path the selected config will ship on.
        #
        # The generator cache then hands the runners the build made here, so the
        # 52-layer stack is packed once.
        pc.SELECTED_PRECISION_CONFIG_PATH = config_path
        started = time.perf_counter()
        generator = build_generator(
            ROOT,
            mesh,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
        )
        result["build_seconds"] = round(time.perf_counter() - started, 1)
        say(f"SWEEP built {args.config} in {result['build_seconds']}s")

        report = generator.capability_report()
        realised = report["precision_policy"]
        if realised["selected_config_id"] != config["config_id"]:
            raise RuntimeError(
                f"the default construction path built {realised['selected_config_id']!r}, not "
                f"{config['config_id']!r}: the candidate did not reach build_generator"
            )
        result["capacity"] = report
        result["realised_precision"] = realised
        result["propagation_problems"] = pc.check_propagation(config, realised)
        if result["propagation_problems"]:
            for problem in result["propagation_problems"]:
                say(f"SWEEP PROPAGATION {problem}")
            raise RuntimeError(
                f"{args.config}: the built model does not match the requested precision artifact; "
                "a candidate whose policy did not propagate is not a measurement"
            )
        say(f"SWEEP {args.config} propagation OK ({realised['policy_name']})")

        # ---- accuracy: prefill check
        from models.common.readiness_check.run_prefill_check import run_prefill_check

        stats, text = capture_stdout(
            run_prefill_check,
            model_dir=ROOT,
            reference_path=ROOT / args.reference,
            mesh_device=mesh,
        )
        result["prefill_check"] = {"per_entry": stats, "output": text}
        (OUT / f"logs/prefill_check_{args.config}.txt").write_text(text)

        # ---- accuracy + traced decode: teacher forcing, repeated
        from models.common.readiness_check.run_teacher_forcing import run_teacher_forcing

        rounds = []
        for round_idx in range(args.rounds):
            generator.reset()
            generator.reset_counters()
            stats, text = capture_stdout(
                run_teacher_forcing,
                model_dir=ROOT,
                reference_path=ROOT / args.reference,
                mesh_device=mesh,
            )
            counters = dict(generator.counters)
            rounds.append({"round": round_idx, "per_entry": stats, "counters": counters, "output": text})
            entry = stats[0]
            say(
                f"SWEEP {args.config} teacher round {round_idx}: top1={entry['top1']:.3f} "
                f"top5={entry['top5']:.3f} top100={entry['top100']:.3f} "
                f"decode={entry['decode_t/s/u']:.2f} t/s/u ttft={entry['ttft_ms']:.2f} ms "
                f"trace_replays={counters.get('trace_replays')}"
            )
        (OUT / f"logs/teacher_forcing_{args.config}.txt").write_text("\n\n".join(r["output"] for r in rounds))
        result["teacher_forcing_rounds"] = rounds

        decode = [r["per_entry"][0]["decode_t/s/u"] for r in rounds]
        ttft = [r["per_entry"][0]["ttft_ms"] for r in rounds]
        decode_sorted = sorted(decode)
        result["teacher_forcing"] = {
            "top1": rounds[0]["per_entry"][0]["top1"],
            "top5": rounds[0]["per_entry"][0]["top5"],
            "top100": rounds[0]["per_entry"][0]["top100"],
            "total_tokens": rounds[0]["per_entry"][0]["total"],
            "decode_tok_s_u_rounds": decode,
            "decode_tok_s_u_median": decode_sorted[len(decode_sorted) // 2],
            "decode_tok_s_u_max": max(decode),
            "decode_tok_s_u_min": min(decode),
            "ttft_ms_rounds": ttft,
            "ttft_ms_min": min(ttft),
            "ttft_ms_median": sorted(ttft)[len(ttft) // 2],
            # The runner asserts ``enable_trace=True`` is an explicit keyword and
            # calls it; this is the generator's own count of trace replays for the
            # 100-token entry, so the "traced" claim is a measured counter.
            "trace_replays_per_round": [r["counters"].get("trace_replays") for r in rounds],
            "traced": all((r["counters"].get("trace_replays") or 0) >= 99 for r in rounds),
        }
        # Accuracy must not move between rounds; if it does, the run is not
        # deterministic and the accuracy number means nothing.
        result["teacher_forcing"]["top1_rounds"] = [r["per_entry"][0]["top1"] for r in rounds]
        result["teacher_forcing"]["accuracy_stable"] = (
            len({(r["per_entry"][0]["top1"], r["per_entry"][0]["top5"], r["per_entry"][0]["top100"]) for r in rounds})
            == 1
        )

        # ---- traced logits-only decode replay: the low-variance cross-check.
        vocab = generator.model.config.vocab_size
        torch.manual_seed(17)
        prompt = [int(t) for t in torch.randint(0, vocab, (128,)).tolist()]
        generator.reset()
        generator.generate(prompt_token_ids=prompt, max_new_tokens=2, enable_trace=True)
        logits_only = []
        for _ in range(3):
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            for _ in range(args.replays):
                ttnn.execute_trace(mesh, generator._trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            logits_only.append((time.perf_counter() - started) / args.replays * 1e3)
        result["traced_logits_only"] = {
            "ms_per_token_min": min(logits_only),
            "tok_s_u": 1e3 / min(logits_only),
            "rounds_ms": logits_only,
            "replays_per_round": args.replays,
        }
        say(
            f"SWEEP {args.config} traced logits-only {min(logits_only):.3f} ms/token "
            f"-> {1e3/min(logits_only):.2f} t/s/u"
        )
        result["status"] = "ok"
        say("SWEEP_OK")
        return 0
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        say(f"SWEEP {args.config} ERROR {result['error']}")
        raise
    finally:
        path = RUNS / (args.out or f"{args.config}.json")
        path.write_text(json.dumps(result, indent=2, default=str) + "\n")
        say(f"SWEEP summary -> {path}")
        if generator is not None:
            generator.teardown()
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
