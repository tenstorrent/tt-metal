#!/usr/bin/env python3
"""
Run profiling experiments

Generates model/training configs and mesh descriptors, resets the board
before each experiment, runs the profiler, and saves all logs.

Usage:
    python run_experiments.py                                    # run all (default: llama_8b)
    python run_experiments.py --name my_model --model-template path/to/model.yaml --training-template path/to/training.yaml
    python run_experiments.py --phases 1 2                       # only Phase 1 & 2
    python run_experiments.py --experiments p1_b1_blk2_tp1_ddp1_ga1_default
    python run_experiments.py --dry-run                          # print plan, no execution
    python run_experiments.py --skip-reset                       # skip board reset
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path

# =============================================================================
# Paths & defaults
# =============================================================================

TT_METAL_HOME = Path(os.environ.get("TT_METAL_HOME", "/data/philei/tt-metal"))
TT_TRAIN_HOME = TT_METAL_HOME / "tt-train"
BASE_DIR = TT_TRAIN_HOME / "tools" / "profiling"
NANO_GPT_BIN = (
    TT_METAL_HOME
    / "build"
    / "tt-train"
    / "sources"
    / "examples"
    / "nano_gpt"
    / "nano_gpt"
)

DEFAULT_NAME = "llama_8b"
DEFAULT_MODEL_TEMPLATE = TT_TRAIN_HOME / "configs" / "model_configs" / "llama8b.yaml"
DEFAULT_TRAINING_TEMPLATE = (
    TT_TRAIN_HOME
    / "configs"
    / "training_configs"
    / "llama8b"
    / "training_shakespeare_llama_8b_galaxy.yaml"
)

# Overridable at runtime via CLI args
MODEL_TEMPLATE = DEFAULT_MODEL_TEMPLATE
TRAINING_TEMPLATE = DEFAULT_TRAINING_TEMPLATE
EXPERIMENT_NAME = DEFAULT_NAME

EXPERIMENT_TIMEOUT_S = 2400  # 40 min per experiment
DEFAULT_MAX_STEPS = 8

# =============================================================================
# Custom YAML dumper — lists render in flow style: [4, 8]
# =============================================================================


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(
    list,
    lambda d, data: d.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)

# =============================================================================
# Experiment definitions
#
# Names are auto-generated: p{phase}_b{batch}_blk{blocks}_tp{tp}_ddp{ddp}_ga{ga}_{runner}
# =============================================================================


def _name(exp):
    rt = (
        "memeff"
        if exp.get("runner_type", "default") == "memory_efficient"
        else "default"
    )
    prof = exp.get("profiler", True)
    suffix = "_noprof" if prof is False else "_naive" if prof == "naive" else ""
    return (
        f"p{exp['phase']}"
        f"_b{exp['local_batch']}"
        f"_blk{exp['num_blocks']}"
        f"_tp{exp.get('tp', 1)}"
        f"_ddp{exp.get('ddp', 1)}"
        f"_ga{exp.get('grad_accum', 1)}"
        f"_{rt}"
        f"{suffix}"
    )


def _exp(**kwargs):
    exp = dict(**kwargs)
    exp["name"] = _name(exp)
    return exp


# Each experiment section lists ALL runs it needs (including overlaps).
# Duplicates are removed at the end — only unique (name) entries are kept.

_ALL_EXPERIMENTS = [
    # =================================================================
    # Phase 1 — Single-device baselines (TP=1, DDP=1)
    #   Profiled: fwd/bwd/opt at 2, 4, 8 blocks → extrapolate to 32
    #   Checkpointing: compare default vs memeff at 8 blocks
    # =================================================================
    _exp(phase="1", local_batch=1, num_blocks=2, runner_type="default"),
    _exp(phase="1", local_batch=1, num_blocks=4, runner_type="default"),
    _exp(phase="1", local_batch=1, num_blocks=8, runner_type="default"),
    _exp(phase="1", local_batch=1, num_blocks=8, runner_type="memory_efficient"),
    # No profiler - to compare tracy to naive profiler, and estimate host overhead
    _exp(
        phase="1", local_batch=1, num_blocks=2, runner_type="default", profiler="naive"
    ),
    _exp(
        phase="1", local_batch=1, num_blocks=4, runner_type="default", profiler="naive"
    ),
    _exp(
        phase="1", local_batch=1, num_blocks=8, runner_type="default", profiler="naive"
    ),
    # =================================================================
    # Phase 2 — TP characterization (differential method)
    #   Profiled: TP=2,4,8 at 2, 4, 8 blocks, batch=1
    #   Derives: tp_perf_perc, tp_ccl_fwd/bwd per block
    # =================================================================
    # TP=2
    _exp(phase="2", local_batch=1, num_blocks=2, tp=2, runner_type="default"),
    _exp(phase="2", local_batch=1, num_blocks=4, tp=2, runner_type="default"),
    _exp(phase="2", local_batch=1, num_blocks=8, tp=2, runner_type="default"),
    # TP=4
    _exp(phase="2", local_batch=1, num_blocks=2, tp=4, runner_type="default"),
    _exp(phase="2", local_batch=1, num_blocks=4, tp=4, runner_type="default"),
    _exp(phase="2", local_batch=1, num_blocks=8, tp=4, runner_type="default"),
    # TP=8
    _exp(phase="2", local_batch=1, num_blocks=2, tp=8, runner_type="default"),
    _exp(phase="2", local_batch=1, num_blocks=4, tp=8, runner_type="default"),
    _exp(phase="2", local_batch=1, num_blocks=8, tp=8, runner_type="default"),
    # =================================================================
    # Phase 3 — DDP characterization (naive profiler, TP=1 only)
    #   DDP = 2, 4, 8, 32. Clean gradient_sync from naive profiler.
    #   DDP scaling with TP is verified in Phase 6.
    # =================================================================
    _exp(
        phase="3",
        local_batch=1,
        num_blocks=8,
        ddp=2,
        runner_type="default",
        profiler="naive",
    ),
    _exp(
        phase="3",
        local_batch=1,
        num_blocks=8,
        ddp=4,
        runner_type="default",
        profiler="naive",
    ),
    _exp(
        phase="3",
        local_batch=1,
        num_blocks=8,
        ddp=8,
        runner_type="default",
        profiler="naive",
    ),
    _exp(
        phase="3",
        local_batch=1,
        num_blocks=8,
        ddp=32,
        runner_type="default",
        profiler="naive",
    ),
    # =================================================================
    # Phase 5 — Scaling verification
    # =================================================================
    # 5.1 + 5.5: Compute + optimizer vs batch size
    #   TP=1, DDP=1, 4 blocks, memory_efficient, naive profiler
    #   batch = 1, 2, 4, 8, 16
    _exp(
        phase="5",
        local_batch=1,
        num_blocks=4,
        runner_type="memory_efficient",
        profiler="naive",
    ),
    _exp(
        phase="5",
        local_batch=2,
        num_blocks=4,
        runner_type="memory_efficient",
        profiler="naive",
    ),
    _exp(
        phase="5",
        local_batch=4,
        num_blocks=4,
        runner_type="memory_efficient",
        profiler="naive",
    ),
    _exp(
        phase="5",
        local_batch=8,
        num_blocks=4,
        runner_type="memory_efficient",
        profiler="naive",
    ),
    _exp(
        phase="5",
        local_batch=16,
        num_blocks=4,
        runner_type="memory_efficient",
        profiler="naive",
    ),
    # 5.2: Compute vs num_blocks
    #   TP=1, DDP=1, batch=1, memory_efficient, naive profiler
    #   blocks = 2, 4, 8, 16
    _exp(
        phase="5",
        local_batch=1,
        num_blocks=2,
        runner_type="memory_efficient",
        profiler="naive",
    ),
    _exp(
        phase="5",
        local_batch=1,
        num_blocks=4,
        runner_type="memory_efficient",
        profiler="naive",
    ),  # dup of 5.1 batch=1
    _exp(
        phase="5",
        local_batch=1,
        num_blocks=8,
        runner_type="memory_efficient",
        profiler="naive",
    ),
    _exp(
        phase="5",
        local_batch=1,
        num_blocks=16,
        runner_type="memory_efficient",
        profiler="naive",
    ),
    # =================================================================
    # Phase 6 — End-to-end validation (naive profiler)
    #   TP+DDP: validates that DDP sync scales with TP as predicted.
    #   Grad_accum sweep: verifies compute scales linearly, sync stays constant.
    # =================================================================
    _exp(
        phase="6",
        local_batch=1,
        num_blocks=8,
        tp=8,
        ddp=4,
        grad_accum=1,
        runnder_type="default",
        profiler="naive",
    ),
    # _exp(phase="6", local_batch=1, num_blocks=8, tp=8, ddp=4, grad_accum=2, runnder_type="default", profiler="naive"),
    # _exp(phase="6", local_batch=1, num_blocks=8, tp=8, ddp=4, grad_accum=4, runnder_type="default", profiler="naive"),
    _exp(
        phase="6",
        local_batch=1,
        num_blocks=8,
        tp=4,
        ddp=8,
        grad_accum=1,
        runnder_type="default",
        profiler="naive",
    ),
    # _exp(phase="6", local_batch=1, num_blocks=8, tp=4, ddp=8, grad_accum=2, runnder_type="default", profiler="naive"),
    # _exp(phase="6", local_batch=1, num_blocks=8, tp=4, ddp=8, grad_accum=4, runnder_type="default", profiler="naive"),
    _exp(
        phase="6",
        local_batch=1,
        num_blocks=32,
        tp=8,
        ddp=4,
        grad_accum=1,
        runnder_type="default",
        profiler="naive",
    ),
    _exp(
        phase="6",
        local_batch=1,
        num_blocks=32,
        tp=4,
        ddp=8,
        grad_accum=1,
        runnder_type="default",
        profiler="naive",
    ),
]

# Deduplicate by name (last occurrence wins)
_seen = {}
for _e in _ALL_EXPERIMENTS:
    _seen[_e["name"]] = _e
EXPERIMENTS = list(_seen.values())

# =============================================================================
# Helpers
# =============================================================================


def mesh_shape(exp):
    """Return [row, col] mesh dims.

    TP/DDP-only → first dim; TP+DDP → [ddp, tp].
    """
    tp = exp.get("tp", 1)
    ddp = exp.get("ddp", 1)
    if tp > 1 and ddp > 1:
        return [ddp, tp]
    if tp > 1:
        return [tp, 1]
    if ddp > 1:
        return [ddp, 1]
    return [1, 1]


def config_batch_size(exp):
    """Config batch_size = local_batch * ddp (total across DDP ranks)."""
    return exp["local_batch"] * exp.get("ddp", 1)


# =============================================================================
# Config generation
# =============================================================================


def generate_model_config(num_blocks, runner_type, out_path):
    with open(MODEL_TEMPLATE) as f:
        cfg = yaml.safe_load(f)
    cfg["transformer_config"]["num_blocks"] = num_blocks
    cfg["transformer_config"]["runner_type"] = runner_type
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, Dumper=_Dumper, default_flow_style=False, sort_keys=False)


def generate_training_config(exp, model_cfg_path, out_path):
    with open(TRAINING_TEMPLATE) as f:
        cfg = yaml.safe_load(f)

    tp = exp.get("tp", 1)
    ddp = exp.get("ddp", 1)
    ms = mesh_shape(exp)

    tc = cfg["training_config"]
    tc["batch_size"] = config_batch_size(exp)
    tc["max_steps"] = MAX_STEPS
    tc["model_config"] = model_cfg_path
    tc["gradient_accumulation_steps"] = exp.get("grad_accum", 1)
    tc["model_save_interval"] = 10000  # no checkpoint saves during profiling

    dc = cfg["device_config"]
    dc["enable_tp"] = tp > 1
    dc["enable_ddp"] = ddp > 1
    dc["mesh_shape"] = ms

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, Dumper=_Dumper, default_flow_style=False, sort_keys=False)


TEXTPROTO_TEMPLATE = """\
mesh_descriptors {{
  name: "M0"
  arch: BLACKHOLE
  device_topology {{ dims: [ {d0}, {d1} ] }}
  host_topology   {{ dims: [ 1, 1 ] }}
  channels {{ count: 2 policy: RELAXED }}
}}

pinnings {{
  logical_fabric_node_id {{
    mesh_id: 0
    chip_id: 0
  }}
  physical_asic_position {{
    tray_id: 1
    asic_location: 1
  }}
}}

top_level_instance {{ mesh {{ mesh_descriptor: "M0" mesh_id: 0 }} }}
"""


def generate_textproto(dims, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(TEXTPROTO_TEMPLATE.format(d0=dims[0], d1=dims[1]))


# =============================================================================
# Pre-generate all configs for a set of experiments
# =============================================================================


def generate_all_configs(experiments, run_dir):
    """Generate all model configs, training configs, and mesh descriptors.

    Returns (model_cfgs, textprotos) dicts mapping keys to absolute paths.
    """
    model_cfgs = {}
    textprotos = {}

    # Unique model configs keyed by (num_blocks, runner_type)
    for exp in experiments:
        key = (exp["num_blocks"], exp.get("runner_type", "default"))
        if key not in model_cfgs:
            fname = f"{EXPERIMENT_NAME}_{key[0]}blocks_{key[1]}.yaml"
            path = run_dir / "model_configs" / fname
            generate_model_config(key[0], key[1], path)
            model_cfgs[key] = path

    # Unique mesh descriptors keyed by (dim0, dim1)
    for exp in experiments:
        ms = mesh_shape(exp)
        key = tuple(ms)
        if key not in textprotos:
            fname = f"mesh_{key[0]}x{key[1]}.textproto"
            path = run_dir / "mesh_descriptors" / fname
            generate_textproto(ms, path)
            textprotos[key] = path

    # Per-experiment training configs
    for exp in experiments:
        mkey = (exp["num_blocks"], exp.get("runner_type", "default"))
        model_path = str(model_cfgs[mkey].resolve())
        out = run_dir / exp["name"] / "training_config.yaml"
        generate_training_config(exp, model_path, out)

    return model_cfgs, textprotos


# =============================================================================
# Execution
# =============================================================================


def reset_board():
    print("  Resetting board (tt-smi -glx_reset) ...")
    result = subprocess.run(["tt-smi", "-glx_reset"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: board reset returned {result.returncode}")
        if result.stderr.strip():
            print(f"  stderr: {result.stderr.strip()[:200]}")


def run_experiment(exp, run_dir, textprotos, skip_reset=False):
    name = exp["name"]
    exp_dir = run_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    ms = mesh_shape(exp)
    total_devices = ms[0] * ms[1]
    training_cfg = exp_dir / "training_config.yaml"
    training_cfg_abs = str(training_cfg.resolve())

    # Board reset
    if not skip_reset:
        reset_board()

    # Environment — set or unset TT_MESH_GRAPH_DESC_PATH
    env = os.environ.copy()
    if total_devices > 1:
        tp_path = textprotos[tuple[int, ...](ms)]
        env["TT_MESH_GRAPH_DESC_PATH"] = str(tp_path.resolve())
    else:
        env.pop("TT_MESH_GRAPH_DESC_PATH", None)

    profiler_mode = exp.get("profiler", True)
    if profiler_mode is True:
        cmd = (
            "env -u TT_METAL_DPRINT_CORES "
            "TT_METAL_WATCHER_NOINLINE=1 "
            "TT_METAL_WATCHER_DEBUG_DELAY=10 "
            "TT_METAL_READ_DEBUG_DELAY_CORES=0,0 "
            "TT_METAL_WRITE_DEBUG_DELAY_CORES=0,0 "
            "TT_METAL_READ_DEBUG_DELAY_RISCVS=BR "
            "TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR "
            "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 "
            f'python3 -m tracy -r -v -p "{NANO_GPT_BIN} --config {training_cfg_abs}"'
        )
    elif profiler_mode == "naive":
        cmd = f"TTML_NAIVE_PROFILER=1 {NANO_GPT_BIN} --config {training_cfg_abs}"
    else:
        cmd = f"{NANO_GPT_BIN} --config {training_cfg_abs}"

    log_out = exp_dir / "stdout.log"
    log_err = exp_dir / "stderr.log"
    meta = {
        "name": name,
        "command": cmd,
        "profiler": profiler_mode,
        "mesh_shape": ms,
        "total_devices": total_devices,
        "env_TT_MESH_GRAPH_DESC_PATH": env.get("TT_MESH_GRAPH_DESC_PATH", "<unset>"),
        "experiment": exp,
    }
    with open(exp_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  CMD: {cmd}")
    print(f"  Logs: {log_out}")

    t0 = time.time()
    try:
        with open(log_out, "w") as fo, open(log_err, "w") as fe:
            result = subprocess.run(
                cmd,
                shell=True,
                env=env,
                stdout=fo,
                stderr=fe,
                cwd=str(TT_METAL_HOME),
                timeout=EXPERIMENT_TIMEOUT_S,
            )
        rc = result.returncode
    except subprocess.TimeoutExpired:
        rc = -999
        print(f"  TIMEOUT after {EXPERIMENT_TIMEOUT_S}s")
    except Exception as exc:
        rc = -1
        print(f"  EXCEPTION: {exc}")

    elapsed = time.time() - t0

    # Try to grab the latest profiler report
    profiler_src = TT_TRAIN_HOME / "generated" / "profiler" / "reports"
    if profiler_src.exists():
        reports = sorted(profiler_src.iterdir(), key=os.path.getmtime, reverse=True)
        if reports:
            dst = exp_dir / "profiler_report"
            if dst.exists():
                shutil.rmtree(dst)
            try:
                shutil.copytree(reports[0], dst)
                print(f"  Profiler report copied to {dst}")
            except Exception as exc:
                print(f"  WARNING: could not copy profiler report: {exc}")

    status = "OK" if rc == 0 else f"FAILED(rc={rc})"
    print(f"  Result: {status}  elapsed={elapsed:.0f}s")

    return {
        "name": name,
        "returncode": rc,
        "elapsed_s": round(elapsed, 1),
        "log_out": str(log_out),
        "log_err": str(log_err),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    global MODEL_TEMPLATE, TRAINING_TEMPLATE, EXPERIMENT_NAME, NANO_GPT_BIN, MAX_STEPS

    parser = argparse.ArgumentParser(description="Run profiling experiments")
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help=f"Experiment set name (used in filenames and logs, default: {DEFAULT_NAME})",
    )
    parser.add_argument(
        "--model-template",
        type=str,
        default=None,
        help=f"Model config YAML template (default: {DEFAULT_MODEL_TEMPLATE})",
    )
    parser.add_argument(
        "--training-template",
        type=str,
        default=None,
        help=f"Training config YAML template (default: {DEFAULT_TRAINING_TEMPLATE})",
    )
    parser.add_argument(
        "--binary",
        type=str,
        default=None,
        help=f"Training binary path (default: {NANO_GPT_BIN})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Number of training steps per experiment (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--phases", nargs="+", help="Run only these phases (e.g. 1 2 5)"
    )
    parser.add_argument(
        "--experiments", nargs="+", help="Run only these experiments by name"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print plan without executing",
    )
    parser.add_argument(
        "--list-names",
        action="store_true",
        help="Print experiment names (one per line) and exit. Respects --phases filter.",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip tt-smi board reset between experiments",
    )
    parser.add_argument(
        "--run-dir", type=str, help="Output directory (default: timestamped)"
    )
    args = parser.parse_args()

    # Apply overrides
    EXPERIMENT_NAME = args.name
    MAX_STEPS = args.max_steps
    if args.model_template:
        MODEL_TEMPLATE = Path(args.model_template)
        if not MODEL_TEMPLATE.exists():
            print(f"Model template not found: {MODEL_TEMPLATE}")
            sys.exit(1)
    if args.training_template:
        TRAINING_TEMPLATE = Path(args.training_template)
        if not TRAINING_TEMPLATE.exists():
            print(f"Training template not found: {TRAINING_TEMPLATE}")
            sys.exit(1)
    if args.binary:
        NANO_GPT_BIN = Path(args.binary)
        if not NANO_GPT_BIN.exists():
            print(f"Binary not found: {NANO_GPT_BIN}")
            sys.exit(1)

    # Filter experiments
    exps = list(EXPERIMENTS)
    if args.phases:
        exps = [e for e in exps if e["phase"] in args.phases]
    if args.experiments:
        exps = [e for e in exps if e["name"] in args.experiments]
    if not exps:
        print("No experiments matched the filter.")
        sys.exit(1)

    if args.list_names:
        for e in exps:
            print(e["name"])
        return

    # Run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = BASE_DIR / "experiments" / f"{EXPERIMENT_NAME}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Header
    print("=" * 72)
    print(f"  Profiling Experiment Runner: {EXPERIMENT_NAME}")
    print(f"  Model template    : {MODEL_TEMPLATE}")
    print(f"  Training template : {TRAINING_TEMPLATE}")
    print(f"  Binary            : {NANO_GPT_BIN}")
    print(f"  Run directory     : {run_dir}")
    print(f"  Experiments       : {len(exps)}")
    print(f"  Max steps/exp     : {MAX_STEPS}")
    print(f"  Board reset       : {'OFF' if args.skip_reset else 'ON'}")
    print("=" * 72)

    # Print plan
    for i, exp in enumerate(exps, 1):
        tp = exp.get("tp", 1)
        ddp = exp.get("ddp", 1)
        ga = exp.get("grad_accum", 1)
        rt = exp.get("runner_type", "default")[:6]
        ms = mesh_shape(exp)
        print(
            f"  {i:2d}. {exp['name']:<40s}  "
            f"blk={exp['num_blocks']} lb={exp['local_batch']} "
            f"TP={tp} DDP={ddp} GA={ga} "
            f"mesh={ms} {rt}"
        )
    print()

    # Generate all configs up-front
    print("Generating configs ...")
    _, textprotos = generate_all_configs(exps, run_dir)
    with open(run_dir / "experiments.json", "w") as f:
        json.dump(exps, f, indent=2)
    print(f"  Configs written to {run_dir}\n")

    if args.dry_run:
        print("Dry run — no experiments executed.")
        return

    # Execute
    results = []
    for i, exp in enumerate(exps, 1):
        print(f"\n{'=' * 72}")
        print(f"  [{i}/{len(exps)}] {exp['name']}")
        print(f"{'=' * 72}")
        r = run_experiment(exp, run_dir, textprotos, skip_reset=args.skip_reset)
        results.append(r)

    # Summary
    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")

    ok = sum(1 for r in results if r["returncode"] == 0)
    fail = len(results) - ok
    total_s = sum(r["elapsed_s"] for r in results)

    for r in results:
        tag = " OK " if r["returncode"] == 0 else "FAIL"
        print(f"  [{tag}] {r['name']:<40s} {r['elapsed_s']:>8.1f}s")

    print(
        f"\n  {ok} passed, {fail} failed, "
        f"total elapsed: {total_s:.0f}s ({total_s / 60:.1f}m)"
    )
    print(f"  Results: {run_dir}")

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main()
