#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Perf + quality matrix for Wan2.2 (T2V, I2V) and MiniMax-H3 (T2VA, FL2VA, Ref2VA), same on any machine.

    run_matrix.py probe                      # what machine is this, what would run
    run_matrix.py plan                       # the resolved test node ids, nothing executed
    run_matrix.py run   [--only a,b] [--tdp 130] [--repeat 1] [--out DIR]
    run_matrix.py score --out DIR            # VBench + CLIP + frame sanity on the clips a run kept
    run_matrix.py setup-scoring              # one-off: install vbench + decord into ~/dit-perf-pydeps
    run_matrix.py check-tdp [--tdp 130]      # open the devices and require every chip to read back that limit

Each case gets its own `tt-smi` reset, a wall-clock limit (a hang is recorded as HUNG and the matrix goes
on), and the TDP limit pinned and verified from the runtime's own log. Results land in <out>/results.json,
with the machine description in <out>/machine.json, so two machines' runs can sit side by side.
Run it from the checkout you want to measure; PYTHONPATH and TT_METAL_HOME follow that checkout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the checkout this script lives in
TESTS = ROOT / "models/tt_dit/tests/models"
WAN_PERF = TESTS / "wan2_2/test_performance_wan.py"
H3_PERF = TESTS / "minimax_h3/test_performance_minimax_h3.py"
PYDEPS = Path.home() / "dit-perf-pydeps"
PYDEPS_RUN = (
    Path.home() / "dit-perf-pydeps-run"
)  # imageio_ffmpeg only: Wan's VBench export needs it, nothing else may shadow the env
FFMPEG_BIN = (
    Path.home() / "dit-perf-bin"
)  # holds an `ffmpeg` symlink; the tests mux their mp4 with whatever ffmpeg is on PATH
H3_DURATION = "15s"
H3_REF_SETS = ("image", "video_with_sound", "mixed")

# (arch, chip count) -> the mesh token in each test's parametrize id. A machine not listed here is reported
# as unsupported rather than guessed at.
WAN_MESH = {
    ("blackhole", 32): "ring_bh_4x8_sp1tp0",
    ("wormhole", 32): "wh_4x8_sp1tp0",
    ("blackhole", 128): "bh_4x32sp1tp0",
    ("blackhole", 8): "bh_2x4_sp1tp0",
    ("wormhole", 8): "2x4_sp0tp1",
    ("wormhole", 4): "2x2_sp0tp1",
}
H3_MESH = {
    ("blackhole", 32): "4x8",
    ("wormhole", 32): "4x8_WH",
    ("blackhole", 128): "4x32_TRACED",
}
# Weights each H3 task needs, to fail fast instead of letting the test skip itself.
H3_REQUIRED = {"t2va": ("transformer",), "fl2va": ("transformer",), "ref2va": ("transformer_ref",)}
H3_WEIGHT_CANDIDATES = (
    "/mnt/weka/model-weights/video/MiniMaxAI/MiniMax-H3",
    str(Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots"),
)


@dataclass
class Case:
    id: str
    family: str  # "wan" | "h3"
    task: str
    test_file: Path
    func: str
    tokens: tuple[str, ...]  # id components the parametrize id must contain (mesh token added per machine)
    mesh_table: dict
    timeout_s: int
    env: dict = field(default_factory=dict)
    prompt: str = ""


def build_cases(args) -> list[Case]:
    cases = []
    for model in ("t2v", "i2v"):
        cases.append(
            Case(
                f"wan-{model}-720p",
                "wan",
                model,
                WAN_PERF,
                "test_pipeline_performance",
                (model, "resolution_720p"),
                WAN_MESH,
                3 * 3600,
                env={"RUN_VBENCH": "0" if args.no_vbench else "1"},
            )
        )
    cases.append(
        Case("h3-t2va-15s", "h3", "t2va", H3_PERF, "test_t2va_performance", (f"16x9_{H3_DURATION}",), H3_MESH, 2 * 3600)
    )
    cases.append(
        Case("h3-fl2va-15s", "h3", "fl2va", H3_PERF, "test_fl2va_performance", (H3_DURATION,), H3_MESH, 2 * 3600)
    )
    for ref_set in H3_REF_SETS:
        cases.append(
            Case(
                f"h3-ref2va-{ref_set}-15s",
                "h3",
                "ref2va",
                H3_PERF,
                "test_ref2va_performance",
                (H3_DURATION, ref_set),
                H3_MESH,
                3 * 3600,
            )
        )
    if args.only:
        wanted = args.only.split(",")
        cases = [c for c in cases if any(w in c.id for w in wanted)]
    return cases


# ------------------------------------------------------------------ environment


def run_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(
        TT_METAL_HOME=str(ROOT),
        PYTHONPATH=os.pathsep.join(
            [str(ROOT), str(ROOT / "ttnn"), str(ROOT / "tools")] + ([str(PYDEPS_RUN)] if PYDEPS_RUN.is_dir() else [])
        ),
        TT_DIT_CACHE_DIR=env.get("TT_DIT_CACHE_DIR", str(Path.home() / ".cache/tt-dit")),
        HF_HUB_OFFLINE="1",
    )
    if FFMPEG_BIN.is_dir():
        env["PATH"] = f"{FFMPEG_BIN}{os.pathsep}{env.get('PATH', '')}"
    env.update(extra or {})
    return env


def python_exe() -> str:
    """The checkout's own python_env if it has one, else a sibling checkout's, else the current python."""
    for cand in (ROOT / "python_env/bin/python", Path.home() / "tt-metal/python_env/bin/python"):
        if cand.is_file():
            return str(cand)
    return sys.executable


def sh(cmd, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe(args) -> dict:
    """Machine facts. --arch/--chips override the live query (used for dry runs off-box)."""
    info = {"host": socket.gethostname(), "time": datetime.now().isoformat(timespec="seconds")}
    info["git_sha"] = sh(["git", "-C", str(ROOT), "rev-parse", "--short=11", "HEAD"]).stdout.strip()
    info["git_branch"] = sh(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    info["git_dirty_files"] = len(
        sh(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]).stdout.splitlines()
    )
    snippet = (
        "import json, ttnn;"
        "from models.common.utility_functions import is_blackhole;"
        "n = ttnn.GetNumAvailableDevices();"
        "print('PROBE ' + json.dumps({'arch': 'blackhole' if is_blackhole() else 'wormhole', 'chips': n,"
        " 'pcie_chips': ttnn.GetNumPCIeDevices()}))"
    )
    if args.arch and args.chips:
        info.update(arch=args.arch, chips=args.chips, pcie_chips=args.chips, probed="overridden")
    else:
        r = sh([python_exe(), "-c", snippet], env=run_env(), cwd=ROOT)
        line = next((l for l in r.stdout.splitlines() if l.startswith("PROBE ")), None)
        if line:
            info.update(json.loads(line[6:]), probed="live")
        else:
            info.update(arch=None, chips=None, probed="failed", probe_error=r.stderr[-500:])
    info["dram_banks"] = args.dram_banks  # filled in from the first test's log when not given (see parse_dram)
    return info


def verify_tdp(args, machine: dict) -> dict:
    """Open the devices with the TDP limit set and require every chip to read back exactly `--tdp` watts.

    The runtime sets the limit on each chip it drives over PCIe when it builds the cluster and logs the
    value firmware reports afterwards, so this reads what firmware holds, not what was asked for. Exits
    non-zero on any other value, on a chip that did not report, or on a limit the runtime left alone.
    """
    snippet = "import ttnn; d = ttnn.open_device(device_id=0); ttnn.close_device(d)"
    r = sh([python_exe(), "-c", snippet], env=run_env({"TT_METAL_TDP_LIMIT_WATTS": str(args.tdp)}), cwd=ROOT)
    got = parse_tdp(r.stdout + r.stderr)
    expected = machine.get("pcie_chips") or machine.get("chips")
    problems = []
    if got["watts"] != [args.tdp]:
        problems.append(f"firmware reports {got['watts'] or 'no limit'} W, wanted {args.tdp} W")
    if got["chips_reported"] != expected:
        problems.append(f"{got['chips_reported']} of {expected} PCIe chips reported a limit")
    if problems:
        sys.exit(
            "TDP check FAILED: " + "; ".join(problems) + ". Reset the machine and check chip_limits.tdp_limit "
            "(a bh-mod override can hold the board at 75 W); no test was run.\n" + (r.stdout + r.stderr)[-1500:]
        )
    print(f"TDP check OK: {got['chips_reported']} chips at {args.tdp} W", flush=True)
    return got


def find_h3_weights() -> str | None:
    cands = [os.environ.get("MINIMAX_H3_MODEL_PATH")] + list(H3_WEIGHT_CANDIDATES)
    for cand in filter(None, cands):
        p = Path(cand)
        snaps = sorted(p.glob("*")) if p.name == "snapshots" else [p]
        for snap in snaps:
            if list((snap / "transformer").glob("*.safetensors")):
                return str(snap)
    return None


# ------------------------------------------------------------------ test node resolution


def resolve_node(case: Case, machine: dict) -> tuple[str | None, str]:
    """The single pytest node id for this case on this machine, or (None, reason)."""
    mesh = case.mesh_table.get((machine["arch"], machine["chips"]))
    if mesh is None:
        return None, f"no {case.family} mesh configuration for {machine['arch']} x{machine['chips']}"
    r = sh(
        [python_exe(), "-m", "pytest", "--collect-only", "-q", f"{case.test_file}::{case.func}"],
        env=run_env(),
        cwd=ROOT,
    )
    want = set(case.tokens) | {mesh}
    hits = []
    for pid in re.findall(rf"\b{re.escape(case.func)}\[(.+)\]", r.stdout):  # `<Function f[id]>` or `file::f[id]`
        if want <= set(pid.split("-")):
            hits.append(f"{case.test_file}::{case.func}[{pid}]")
    if len(hits) != 1:
        return None, f"{len(hits)} collected ids match {sorted(want)} (expected 1)"
    return hits[0], ""


# ------------------------------------------------------------------ result parsing

H3_STAGES = {
    "Text Encoding": "text_encode",
    "VAE Encode": "vae_encode",
    "Denoising (total)": "denoise",
    "Denoising (per step)": "denoise_per_step",
    "VAE Decoding": "vae_decode",
    "Audio Decoding": "audio_decode",
    "Total Pipeline": "total",
}
WAN_STAGES = {
    "Text Encoding": "text_encode",
    "Image Encoding (total)": "image_encode",
    "VAE Encoder only": "vae_encode",
    "Denoising": "denoise",
    "VAE Decoding": "vae_decode",
    "Total Pipeline": "total",
}
H3_RE = re.compile(r"^\s*(" + "|".join(map(re.escape, H3_STAGES)) + r")\s*\|\s*([\d.]+)s", re.M)
WAN_RE = re.compile(
    r"^\s*(" + "|".join(map(re.escape, WAN_STAGES)) + r")\s*\| Mean:\s*([\d.]+)s \| Std:\s*([\d.]+)s"
    r" \| Min:\s*([\d.]+)s \| Max:\s*([\d.]+)s",
    re.M,
)


def parse_perf(case: Case, log: str) -> dict:
    out: dict = {}
    if case.family == "h3":
        for name, sec in H3_RE.findall(log):
            out[H3_STAGES[name]] = {"mean": float(sec)}
    else:
        for name, mean, std, lo, hi in WAN_RE.findall(log):
            out[WAN_STAGES[name.strip()]] = {"mean": float(mean), "std": float(std), "min": float(lo), "max": float(hi)}
    for key, pat in (
        ("image_size", r"Image Size: (\S+)"),
        ("num_frames", r"Num Frames: (\d+)"),
        ("steps", r"Inference Steps: (\d+)"),
        ("mesh", r"Mesh Shape: (.+)"),
    ):
        m = re.search(pat, log)
        if m:
            out[key] = m.group(1).strip()
    return out


def parse_tdp(log: str) -> dict:
    """Per-chip limits as the runtime applied them, from its own log lines."""
    now = re.findall(r"firmware TDP limit on chip (\d+) is now (\d+) W", log)
    kept = re.findall(r"leaving the firmware TDP limit on chip (\d+) as it is: (\d+)", log)
    watts = sorted({int(w) for _, w in now + kept})
    return {"chips_reported": len({c for c, _ in now + kept}), "watts": watts}


# Wan's perf test asserts its stage times against limits tuned for a full-DRAM device. A run that only
# trips those finished and measured fine, so it is reported, not discarded.
PERF_LIMIT_RE = re.compile(r"outside \[[\d.]+, [\d.]+\]s")


def status_of(code: int | None, log: str) -> tuple[str, str]:
    if code is None:
        return "HUNG", "wall-clock limit hit; process group killed"
    if code == 0 and re.search(r"\b1 passed\b", log):
        return "PASS", ""
    if code == 0 and "skipped" in log:
        reason = re.findall(r"SKIPPED \[\d+\] [^:]+:\d+: (.*)", log)
        return "SKIPPED", reason[-1] if reason else "skipped"
    if code == 5:
        return "NO_TESTS", "pytest collected nothing"
    err = re.findall(r"^E\s+(.*)$", log, re.M)
    if err and PERF_LIMIT_RE.search(err[-1]):
        return "PERF_LIMIT", err[-1][:300]
    return "FAIL", (err[-1] if err else f"exit {code}")[:300]


# ------------------------------------------------------------------ running


def reset_devices(machine: dict, outdir: Path) -> bool:
    cmd = ["tt-smi", "-glx_reset"] if machine["chips"] and machine["chips"] >= 32 else ["tt-smi", "-r"]
    r = sh(cmd)
    (outdir / "reset.log").write_text(r.stdout + r.stderr)
    return r.returncode == 0


def run_case(case: Case, node: str, machine: dict, args, out: Path, rep: int) -> dict:
    d = out / (case.id if args.repeat == 1 else f"{case.id}.r{rep}")
    d.mkdir(parents=True, exist_ok=True)
    row = {"case": case.id, "family": case.family, "task": case.task, "node": node, "rep": rep}
    if not args.no_reset:
        row["reset_ok"] = reset_devices(machine, d)
    extra = dict(case.env, TT_METAL_TDP_LIMIT_WATTS=str(args.tdp))
    if case.family == "h3":
        extra["MINIMAX_H3_MODEL_PATH"] = machine["h3_weights"]
    if case.family == "wan" and case.env.get("RUN_VBENCH") == "1":
        extra["VBENCH_EXPORT_DIR"] = str(d / "vbench")
    started = time.time()
    log_path = d / "pytest.log"
    cmd = [python_exe(), "-m", "pytest", "-s", "--timeout", str(case.timeout_s), "--basetemp", str(d / "pytest"), node]
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, env=run_env(extra), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = proc.wait(timeout=case.timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = None
    text = log_path.read_text(errors="replace")
    row["status"], row["detail"] = status_of(code, text)
    row["wall_s"] = round(time.time() - started, 1)
    row["perf"] = parse_perf(case, text)
    row["tdp"] = parse_tdp(text)
    row["tdp_requested"] = args.tdp
    row["tdp_verified"] = row["tdp"]["watts"] == [args.tdp] and row["tdp"]["chips_reported"] == (
        machine.get("pcie_chips") or machine["chips"]
    )
    if not row["tdp_verified"]:
        # The numbers were measured at the wrong power: keep them, but never as a result.
        row[
            "detail"
        ] = f"INVALID: firmware TDP {row['tdp']['watts']} W on {row['tdp']['chips_reported']} chips, wanted {args.tdp} W. {row['detail']}".strip()
        row["status"] = "INVALID_TDP"
    row["clips"] = collect_clips(case, d, started)
    return row


def collect_clips(case: Case, d: Path, since: float) -> list[str]:
    """Keep the clip(s) the case produced next to its log, for scoring and for looking at."""
    kept = []
    if case.family == "wan":
        for p in sorted((d / "vbench").glob("p*_s*.mp4")):
            kept.append(str(p.relative_to(d.parent)))
        return kept
    folder = {"t2va": "h3_t2va_artifacts", "fl2va": "h3_fl2va_perf_artifacts", "ref2va": "h3_ref2va_perf_artifacts"}[
        case.task
    ]
    ref_set = case.tokens[-1] if case.task == "ref2va" else ""
    for p in (Path.home() / folder).glob(f"{case.task}_*{H3_DURATION}.mp4"):
        if p.stat().st_mtime >= since and ref_set in p.name and (case.task != "t2va" or "16x9" in p.name):
            dst = d / p.name
            shutil.copy2(p, dst)
            kept.append(str(dst.relative_to(d.parent)))
    return kept


def cmd_plan(args, quiet=False) -> tuple[dict, list]:
    machine = probe(args)
    machine["h3_weights"] = find_h3_weights()
    plan = []
    for case in build_cases(args):
        if machine["arch"] is None:
            node, why = None, "machine probe failed"
        elif case.family == "h3" and not machine["h3_weights"]:
            node, why = None, "no complete MiniMax-H3 weights found (set MINIMAX_H3_MODEL_PATH)"
        else:
            node, why = resolve_node(case, machine)
        plan.append((case, node, why))
    if not quiet:
        print(json.dumps(machine, indent=2))
        for case, node, why in plan:
            print(f"{'RUN ' if node else 'SKIP'} {case.id:28} {node or why}")
    return machine, plan


def cmd_run(args):
    machine, plan = cmd_plan(args, quiet=True)
    out = Path(args.out or Path.home() / "dit-perf-runs" / f"{machine['host']}-{datetime.now():%Y%m%d-%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)
    machine["tdp_requested"] = args.tdp
    if machine["arch"] is None:
        sys.exit("machine probe failed; cannot verify TDP")
    machine["tdp_preflight"] = verify_tdp(args, machine)
    (out / "machine.json").write_text(json.dumps(machine, indent=2))
    results = []
    for case, node, why in plan:
        if node is None:
            results.append({"case": case.id, "status": "SKIPPED", "detail": why})
            print(f"SKIP {case.id}: {why}", flush=True)
        else:
            for rep in range(args.repeat):
                print(f"RUN  {case.id} (rep {rep + 1}/{args.repeat}) {node}", flush=True)
                row = run_case(case, node, machine, args, out, rep)
                results.append(row)
                print(
                    f"     -> {row['status']} {row['wall_s']}s tdp_verified={row['tdp_verified']} {row['detail']}",
                    flush=True,
                )
                if row["status"] == "INVALID_TDP" and not args.keep_going:
                    (out / "results.json").write_text(json.dumps(results, indent=2))
                    sys.exit(f"stopping: {case.id} ran at the wrong TDP (use --keep-going to continue past this)")
        (out / "results.json").write_text(json.dumps(results, indent=2))  # after every case: a crash keeps what ran
    print(f"results: {out}/results.json")
    return out


UNZIP_SHIM = """#!/usr/bin/env python3
# Stand-in for `unzip -d DIR FILE` (all VBench calls), for hosts without unzip.
import sys, zipfile
args = sys.argv[1:]
dest = args[args.index("-d") + 1] if "-d" in args else "."
archive = [a for i, a in enumerate(args) if not a.startswith("-") and (i == 0 or args[i - 1] != "-d")][0]
zipfile.ZipFile(archive).extractall(dest)
"""


def install_unzip_shim():
    if shutil.which("unzip", path=os.environ.get("PATH")) and not (FFMPEG_BIN / "unzip").exists():
        return  # the host has a real one
    FFMPEG_BIN.mkdir(exist_ok=True)
    (FFMPEG_BIN / "unzip").write_text(UNZIP_SHIM)
    (FFMPEG_BIN / "unzip").chmod(0o755)


def cmd_setup_scoring(_args):
    req = ROOT / "models/tt_dit/utils/vbench-requirements.txt"
    for pkgs in (["-r", str(req)], ["decord", "imageio-ffmpeg"]):
        r = subprocess.run(
            ["uv", "pip", "install", "--python", python_exe(), "--no-deps", "--target", str(PYDEPS), *pkgs]
        )
        if r.returncode:
            sys.exit(r.returncode)
    probe_exe = subprocess.run(
        [python_exe(), "-c", "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
        capture_output=True,
        text=True,
        env=run_env({"PYTHONPATH": str(PYDEPS)}),
        cwd=ROOT,
    )
    exe = probe_exe.stdout.strip().splitlines()[-1] if probe_exe.returncode == 0 else None
    if exe:
        FFMPEG_BIN.mkdir(exist_ok=True)
        (FFMPEG_BIN / "ffmpeg").unlink(missing_ok=True)
        (FFMPEG_BIN / "ffmpeg").symlink_to(exe)
    install_unzip_shim()
    print(f"installed into {PYDEPS}; ffmpeg -> {exe}; `score` puts the deps on PYTHONPATH for the scorer only")


def cmd_score(args):
    out = Path(args.out)
    env = run_env({"PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "ttnn"), str(ROOT / "tools"), str(PYDEPS)])})
    scorer = Path(__file__).with_name("score_clips.py")
    sys.exit(subprocess.run([python_exe(), str(scorer), str(out)], env=env, cwd=ROOT).returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["probe", "plan", "run", "score", "setup-scoring", "check-tdp"])
    ap.add_argument("--only", help="comma list of substrings of case ids, e.g. wan-t2v,h3-fl2va")
    ap.add_argument(
        "--tdp", type=int, default=130, help="TT_METAL_TDP_LIMIT_WATTS for every case (default 130, the CI value)"
    )
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="continue the matrix after a case ran at the wrong TDP")
    ap.add_argument("--no-vbench", action="store_true", help="skip Wan's four extra VBench gate clips")
    ap.add_argument("--arch", choices=["blackhole", "wormhole"], help="override the probe (with --chips)")
    ap.add_argument("--chips", type=int)
    ap.add_argument("--dram-banks", type=int)
    args = ap.parse_args()
    {
        "probe": lambda a: print(json.dumps(probe(a), indent=2)),
        "plan": cmd_plan,
        "run": cmd_run,
        "score": cmd_score,
        "setup-scoring": cmd_setup_scoring,
        "check-tdp": lambda a: verify_tdp(a, probe(a)),
    }[args.mode](args)


if __name__ == "__main__":
    main()
