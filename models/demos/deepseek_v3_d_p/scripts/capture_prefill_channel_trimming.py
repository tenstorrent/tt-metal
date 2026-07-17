# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Capture and apply a fabric channel-trimming profile for the DeepSeek V3
disaggregated-prefill (deepseek_v3_d_p) FABRIC_2D path.

Channel trimming is an env-var driven tt-metal fabric feature: a *capture* pass
records which EDM sender/receiver channels each ETH router actually uses, writes
a YAML profile, and an *apply* pass replays that profile so unused channels are
turned off (reducing per-router work and enabling cheaper fabric fast paths).

Because the trimming env vars are read once at process startup and because
capture is mutually exclusive with apply (a TT_FATAL enforces this in
tt_metal/fabric/fabric_builder_context.cpp), each pass runs the prefill test in
its own pytest subprocess:

  1. capture: run the prefill test with
       TT_METAL_ENABLE_CHANNEL_TRIMMING_CAPTURE=1
       TT_METAL_LOGS_PATH=<capture_dir>
     which writes <capture_dir>/generated/reports/channel_trimming_capture.yaml
     at fabric teardown, then promote that YAML to the checked-in profile path.
  2. apply: run the same prefill test with
       TT_METAL_FABRIC_TRIMMING_PROFILE=<checked-in profile>
     and confirm it still passes (PCC gate) with the trimmed fabric.

This mirrors the --channel-trim flow in
models/demos/deepseek_v3_b1/scripts/ccl_trace_perf_matrix.py, focused on the
d_p prefill instead of the CCL micro-op benchmarks.

Examples:
  # Capture + apply on the 8x4 (BH Galaxy) FABRIC_2D prefill transformer:
  python models/demos/deepseek_v3_d_p/scripts/capture_prefill_channel_trimming.py --mesh 8x4

  # Capture only (produce + promote the profile YAML):
  python .../capture_prefill_channel_trimming.py --mesh 8x4 --mode capture

  # Apply an already checked-in profile and validate PCC:
  python .../capture_prefill_channel_trimming.py --mesh 8x4 --mode apply
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Repo root is five parents up: .../tt-metal/models/demos/deepseek_v3_d_p/scripts/<file>
REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = Path(__file__).resolve().parents[1]  # models/demos/deepseek_v3_d_p

# Env vars consumed by the tt-metal fabric builder (see tt_metal/llrt/rtoptions.{hpp,cpp}).
TT_METAL_ENABLE_CHANNEL_TRIMMING_CAPTURE_ENV = "TT_METAL_ENABLE_CHANNEL_TRIMMING_CAPTURE"
TT_METAL_FABRIC_TRIMMING_PROFILE_ENV = "TT_METAL_FABRIC_TRIMMING_PROFILE"
TT_METAL_FABRIC_TRIMMING_OVERRIDE_ENV = "TT_METAL_FABRIC_TRIMMING_OVERRIDE"
TT_METAL_FABRIC_TRIMMING_PRESERVE_VC0_FORWARDING_ENV = "TT_METAL_FABRIC_TRIMMING_PRESERVE_VC0_FORWARDING"
TT_METAL_LOGS_PATH_ENV = "TT_METAL_LOGS_PATH"

# The exporter always writes here under the active logs dir
# (tt_metal/fabric/channel_trimming_export.cpp).
CHANNEL_TRIMMING_CAPTURE_RELPATH = Path("generated") / "reports" / "channel_trimming_capture.yaml"

# Checked-in profiles live under the model dir; one per mesh/SKU.
PROFILE_DIR = MODEL_DIR / "fabric_profiles"

# Prefill tests that expose FABRIC_2D variants (ids fabric2d-mesh-<mesh>).
TEST_TARGETS = {
    "transformer": "models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py",
    "block": "models/demos/deepseek_v3_d_p/tests/test_prefill_block.py",
}
SUPPORTED_MESHES = ("4x2", "2x4", "8x4")


def default_profile_path(mesh: str) -> Path:
    return PROFILE_DIR / f"channel_trimming_prefill_fabric2d_{mesh}.yaml"


def clear_channel_trimming_env(env: dict) -> None:
    """Remove every trimming env var so a pass starts from a clean state."""
    env.pop(TT_METAL_ENABLE_CHANNEL_TRIMMING_CAPTURE_ENV, None)
    env.pop(TT_METAL_FABRIC_TRIMMING_PROFILE_ENV, None)
    env.pop(TT_METAL_FABRIC_TRIMMING_OVERRIDE_ENV, None)
    env.pop(TT_METAL_FABRIC_TRIMMING_PRESERVE_VC0_FORWARDING_ENV, None)


def build_pytest_cmd(test_target: str, mesh: str, extra_k: str | None) -> list:
    """pytest command selecting the FABRIC_2D variant for `mesh` via -k."""
    k_expr = f"fabric2d-mesh-{mesh}"
    if extra_k:
        k_expr = f"({k_expr}) and ({extra_k})"
    return [sys.executable, "-m", "pytest", "-svv", test_target, "-k", k_expr]


def run_pytest(cmd: list, env: dict) -> None:
    print(f"\n[channel-trim] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, env=env, cwd=REPO_ROOT, check=True)


def count_selected_tests(test_target: str, mesh: str, extra_k: str | None, env: dict) -> int | None:
    """Best-effort count of tests the selector resolves to (via --collect-only).

    A capture pass MUST resolve to exactly one test: the profile YAML is re-exported at
    every fabric teardown, so running >1 test silently overwrites it with only the last
    test's channel usage — and a profile that doesn't match the applied workload can
    deadlock the trimmed fabric. Returns the count, or None if collection couldn't be
    parsed (caller then proceeds with a warning). Note `-k` matches substrings, so e.g.
    'balanced' also selects 'non_balanced' — pin the id (e.g. 'and not non_balanced').
    """
    cmd = [c for c in build_pytest_cmd(test_target, mesh, extra_k) if c != "-svv"] + ["--collect-only", "-q"]
    result = subprocess.run(cmd, env=env, cwd=REPO_ROOT, capture_output=True, text=True)
    ids = [ln.strip() for ln in result.stdout.splitlines() if "::" in ln and ln.rstrip().endswith("]")]
    if not ids:
        # Collection can misbehave under this heavy conftest (it opens the device during
        # collection); a 0 here is unreliable, so treat it as "unknown" rather than fatal.
        print(
            "[channel-trim] warning: could not determine selected-test count from --collect-only "
            f"(rc={result.returncode}); proceeding. If capture produces no YAML, check the selector.",
            flush=True,
        )
        return None
    return len(ids)


PRESERVE_VC0_FORWARDING_FIELD = "preserve_vc0_forwarding: true\n"


def _ensure_preserve_vc0_forwarding(profile_out: Path) -> None:
    """Prepend `preserve_vc0_forwarding: true` to the promoted profile if not present.

    The DeepSeek prefill MoE dispatch/combine collectives are data-dependent, multi-hop
    all-to-all: a single capture cannot cover every VC0 forwarding path, so trimming VC0
    from one run deadlocks other routings. The field tells the fabric builder to keep VC0
    topology-complete (all VC0 sender/receiver channels serviced, fast-path collapse
    suppressed) while still trimming VC1/VC2. Making the committed profile self-contained
    means no separate override file is needed to run it safely.
    """
    text = profile_out.read_text()
    if "preserve_vc0_forwarding" in text:
        return
    profile_out.write_text(PRESERVE_VC0_FORWARDING_FIELD + text)


def run_capture(
    test_target: str,
    mesh: str,
    extra_k: str | None,
    capture_dir: Path,
    profile_out: Path,
    preserve_vc0_forwarding: bool = True,
) -> Path:
    """Run the capture pass and promote the produced YAML to `profile_out`."""
    capture_dir.mkdir(parents=True, exist_ok=True)
    capture_yaml = capture_dir / CHANNEL_TRIMMING_CAPTURE_RELPATH

    env = os.environ.copy()
    clear_channel_trimming_env(env)
    env[TT_METAL_ENABLE_CHANNEL_TRIMMING_CAPTURE_ENV] = "1"
    env[TT_METAL_LOGS_PATH_ENV] = str(capture_dir)
    if preserve_vc0_forwarding:
        # Have the exporter stamp preserve_vc0_forwarding into the raw capture YAML too, so the
        # intermediate file is self-contained (not just the promoted copy the harness rewrites).
        env[TT_METAL_FABRIC_TRIMMING_PRESERVE_VC0_FORWARDING_ENV] = "1"

    n = count_selected_tests(test_target, mesh, extra_k, env)
    if n is not None and n != 1:
        raise RuntimeError(
            f"Capture selector resolved to {n} tests, but capture requires exactly 1 "
            "(the profile is overwritten at each fabric teardown, so it would only reflect "
            "the last test — and a mismatched profile can deadlock the trimmed fabric).\n"
            "Narrow the selector with -k. Remember -k matches substrings, e.g. 'balanced' also "
            "matches 'non_balanced' (add 'and not non_balanced'). Use a deterministic workload "
            "(real weights + fixed prompt), not random-routing smoke tests."
        )

    run_pytest(build_pytest_cmd(test_target, mesh, extra_k), env)

    if not capture_yaml.exists():
        raise RuntimeError(
            "Channel trimming capture did not produce the expected YAML: "
            f"{capture_yaml}\n"
            "Check that the run actually opened a FABRIC_2D mesh and reached fabric teardown."
        )

    profile_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(capture_yaml, profile_out)
    if preserve_vc0_forwarding:
        _ensure_preserve_vc0_forwarding(profile_out)
        print("[channel-trim] profile marked preserve_vc0_forwarding: true", flush=True)
    print(f"[channel-trim] captured profile promoted to: {profile_out}", flush=True)
    return profile_out


def run_apply(test_target: str, mesh: str, extra_k: str | None, profile: Path, override: Path | None = None) -> None:
    """Run the prefill test with the checked-in profile (and optional override) applied."""
    if not profile.exists():
        raise RuntimeError(
            f"Profile not found: {profile}\n"
            "Run with --mode capture first, or pass --profile pointing at an existing YAML."
        )
    env = os.environ.copy()
    clear_channel_trimming_env(env)
    env[TT_METAL_FABRIC_TRIMMING_PROFILE_ENV] = str(profile)
    print(f"[channel-trim] applying profile: {profile}", flush=True)
    if override is not None:
        if not override.exists():
            raise RuntimeError(f"Override not found: {override}")
        env[TT_METAL_FABRIC_TRIMMING_OVERRIDE_ENV] = str(override)
        print(f"[channel-trim] applying override: {override}", flush=True)
    run_pytest(build_pytest_cmd(test_target, mesh, extra_k), env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--test",
        choices=sorted(TEST_TARGETS),
        default="transformer",
        help="Which prefill test to drive. 'transformer' = full model (representative profile); "
        "'block' = single-decoder-block smoke (faster). Default: transformer.",
    )
    parser.add_argument(
        "--mesh",
        choices=SUPPORTED_MESHES,
        default="8x4",
        help="FABRIC_2D mesh variant to select (via -k fabric2d-mesh-<mesh>). Default: 8x4.",
    )
    parser.add_argument(
        "--mode",
        choices=("capture", "apply", "both"),
        default="both",
        help="capture: produce+promote the profile. apply: run with the profile. both: capture then apply.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Checked-in profile path (default: fabric_profiles/channel_trimming_prefill_fabric2d_<mesh>.yaml).",
    )
    parser.add_argument(
        "--no-preserve-vc0-forwarding",
        dest="preserve_vc0_forwarding",
        action="store_false",
        help="Do NOT stamp preserve_vc0_forwarding into the captured profile. Default is to stamp it, "
        "because MoE dispatch/combine are data-dependent all-to-all and trimming VC0 deadlocks them.",
    )
    parser.set_defaults(preserve_vc0_forwarding=True)
    parser.add_argument(
        "--override",
        default=None,
        help="Optional TT_METAL_FABRIC_TRIMMING_OVERRIDE YAML applied on top of the profile during "
        "apply (e.g. tt_metal/fabric/config/channel_trimming_overrides/enable_vc0_all_channels.yaml "
        "to keep VC0 forwarding topology-complete for all-to-all MoE dispatch/combine). Usually "
        "unnecessary now that the profile carries preserve_vc0_forwarding by default.",
    )
    parser.add_argument(
        "--capture-dir",
        default=None,
        help="Where the capture pass writes its logs/YAML "
        "(default: generated/channel_trim/<test>/<mesh> under the repo root).",
    )
    parser.add_argument(
        "-k",
        dest="extra_k",
        default=None,
        help="Extra pytest -k expression, ANDed with the fabric2d-mesh-<mesh> selector "
        "(e.g. 'pretrained' or 'dense').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_target = TEST_TARGETS[args.test]
    profile = Path(args.profile).resolve() if args.profile else default_profile_path(args.mesh)
    capture_dir = (
        Path(args.capture_dir).resolve()
        if args.capture_dir
        else REPO_ROOT / "generated" / "channel_trim" / args.test / args.mesh
    )

    override = Path(args.override).resolve() if args.override else None

    if args.mode in ("capture", "both"):
        run_capture(test_target, args.mesh, args.extra_k, capture_dir, profile, args.preserve_vc0_forwarding)
    if args.mode in ("apply", "both"):
        run_apply(test_target, args.mesh, args.extra_k, profile, override)

    print("[channel-trim] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
