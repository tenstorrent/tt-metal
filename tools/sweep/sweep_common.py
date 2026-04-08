# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for LLM util-report sweeps (see tools/sweep/README.md)."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

SIMPLE_TEXT_DEMO = "models/tt_transformers/demo/simple_text_demo.py"
SAMPLE_PROMPTS_DIR = "models/tt_transformers/demo/sample_prompts"

# Preset name -> (prompt filename under sample_prompts/, default max_seq_len).
# max_seq_len must be large enough for the full prompt to avoid clipping.
# Can be overridden globally via --max-seq-len on the sweep CLI.
SEQLEN_PRESETS: dict[str, tuple[str, int]] = {
    "128": ("input_data_questions_prefill_128.json", 2048),
    "256": ("input_data_questions_prefill_256.json", 2048),
    "1k": ("input_data_long_1k.json", 2048),
    "2k": ("input_data_long_2k.json", 4096),
    "4k": ("input_data_long_4k.json", 8192),
    "8k": ("input_data_long_8k.json", 8192),
    "16k": ("input_data_long_16k.json", 32768),
    "32k": ("input_data_long_32k.json", 65536),
    "64k": ("input_data_long_64k.json", 131072),
    "128k": ("input_data_long_128k.json", 132 * 1024),
}


def tt_metal_home() -> Path:
    env = os.environ.get("TT_METAL_HOME")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def apply_profiler_env() -> None:
    os.environ["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"] = "100000"
    os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
    os.environ["TT_METAL_PROFILER_SYNC"] = "1"


def check_environment(ttm: Path) -> None:
    """Verify venv and tt-npe env are active. Call once at sweep startup."""
    errors: list[str] = []

    venv = os.environ.get("VIRTUAL_ENV", "")
    expected_venv = str(ttm / "python_env")
    if not venv or not venv.startswith(expected_venv):
        errors.append(
            f"tt-metal venv is not active (VIRTUAL_ENV={venv!r}).\n"
            f"  Run: source {ttm / 'python_env' / 'bin' / 'activate'}"
        )

    npe_setup = ttm.parent / "tt-npe" / "ENV_SETUP"
    npe_bin = ttm.parent / "tt-npe" / "install" / "bin"
    if not npe_setup.is_file():
        errors.append(f"tt-npe ENV_SETUP not found at {npe_setup}.\n" f"  Expected tt-npe as a sibling of tt-metal.")
    elif str(npe_bin.resolve()) not in os.environ.get("PATH", ""):
        errors.append(
            f"tt-npe ENV_SETUP exists but does not appear to be sourced "
            f"({npe_bin} not in PATH).\n"
            f"  Run: source {npe_setup}"
        )

    if errors:
        msg = "Environment not ready:\n" + "\n".join(f"  - {e}" for e in errors)
        msg += (
            f"\n\nBefore running this script:\n"
            f"  cd {ttm}\n"
            f"  source python_env/bin/activate\n"
            f"  source {npe_setup}"
        )
        raise SystemExit(msg)


def normalize_seqlen_key(key: str) -> str:
    k = key.strip().lower()
    if k.endswith("k") and k[:-1].isdigit():
        return f"{k[:-1]}k"
    return k


def resolve_seqlen_with_max(spec: str, max_seq_len_override: int | None) -> tuple[str, int, str]:
    """
    Returns (label_tag, max_seq_len, input_prompts path relative to TT_METAL_HOME).

    `spec` is either a preset key (256, 1k, …) or a repo-relative path to a JSON file.
    Custom paths require ``max_seq_len_override``.
    """
    s = spec.strip()
    if s.endswith(".json") or "/" in s or "\\" in s:
        if max_seq_len_override is None:
            raise ValueError(
                f"Custom prompt path '{spec}' requires --max-seq-len. " f"Presets: {', '.join(sorted(SEQLEN_PRESETS))}"
            )
        rel = s.replace("\\", "/")
        label = Path(rel).stem.replace(".", "_")
        return label, max_seq_len_override, rel
    key = normalize_seqlen_key(s)
    if key not in SEQLEN_PRESETS:
        raise ValueError(f"Unknown seqlen preset '{spec}'. Known: {', '.join(sorted(SEQLEN_PRESETS))}")
    fname, default_msl = SEQLEN_PRESETS[key]
    rel = f"{SAMPLE_PROMPTS_DIR}/{fname}"
    msl = max_seq_len_override if max_seq_len_override is not None else default_msl
    return key, msl, rel


def build_pytest_argv(
    *,
    mode: str,
    batch_size: int,
    num_layers: int,
    input_prompts: str,
    max_seq_len: int,
    max_generated_tokens: int | None,
    instruct: bool | None,
    enable_trace: bool | None,
    extra_args: list[str],
) -> list[str]:
    argv: list[str] = [
        "pytest",
        SIMPLE_TEXT_DEMO,
        "-k",
        "performance and batch-1",
        "--timeout=0",
        "--input_prompts",
        input_prompts,
        "--max_seq_len",
        str(max_seq_len),
        "--batch_size",
        str(batch_size),
        "--num_layers",
        str(num_layers),
        "--mode",
        mode,
    ]
    if max_generated_tokens is not None:
        argv.extend(["--max_generated_tokens", str(max_generated_tokens)])
    if instruct is not None:
        argv.extend(["--instruct", "1" if instruct else "0"])
    if enable_trace is not None:
        argv.extend(["--enable_trace", "true" if enable_trace else "false"])
    argv.extend(extra_args)
    return argv


def pytest_command_string(argv: list[str]) -> str:
    return shlex.join(argv)


def run_collect_only(ttm: Path, argv: list[str]) -> None:
    proc = subprocess.run(argv, cwd=ttm, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"pytest --collect-only failed with code {proc.returncode}")


def run_gen_util_report(
    ttm: Path,
    output_dir: Path,
    pytest_cmd: str,
    *,
    steady_state: bool,
    single_model_iteration: bool,
    noc_timeout: int | None = None,
) -> None:
    gen_script = ttm / "tools" / "hw_debug" / "gen_util_report.py"
    cmd = [
        sys.executable,
        str(gen_script),
        "-o",
        str(output_dir),
        "-c",
        pytest_cmd,
    ]
    if steady_state:
        cmd.append("--steady-state")
    if single_model_iteration:
        cmd.append("--single-model-iteration")
    if noc_timeout is not None:
        cmd.extend(["--noc-timeout", str(noc_timeout)])
    subprocess.run(cmd, cwd=ttm, check=True)


def parse_comma_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_comma_strs(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def grid_points(seqlens: Iterable[str], batch_sizes: Iterable[int]) -> list[tuple[str, int]]:
    return [(sq, bs) for sq in seqlens for bs in batch_sizes]
