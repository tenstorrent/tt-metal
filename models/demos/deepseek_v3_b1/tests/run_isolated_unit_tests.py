# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run DeepSeek unit tests in pytest subprocesses isolated by fabric topology.

This runner avoids in-process fabric topology changes such as FABRIC_2D ->
FABRIC_2D_TORUS_X. Each bucket runs in a fresh Python process. Buckets reset and
cluster-validate before pytest by default; pure-local buckets opt out.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
UNIT_TEST_DIR = "models/demos/deepseek_v3_b1/tests/unit_tests"
SMOKE_TEST = (
    f"{UNIT_TEST_DIR}/per_core_allocation/test_matmul_expert.py::"
    "test_hybrid_expert_multi_device_2sram_2dram[blackhole-True]"
)
RESET_VALIDATE_SMOKE_ATTEMPTS = 3


@dataclass(frozen=True)
class Bucket:
    name: str
    timeout: int
    targets: tuple[str, ...]
    pytest_args: tuple[str, ...] = ()
    # Run each collected nodeid in its own pytest subprocess so a timeout cannot poison later cases.
    case_isolated: bool = False
    reset_before: bool = True
    slow_dispatch: bool = True
    env: tuple[tuple[str, str], ...] = ()


BUCKETS: tuple[Bucket, ...] = (
    Bucket(
        name="pure_local_matmul_expert",
        timeout=600,
        reset_before=False,
        targets=(
            f"{UNIT_TEST_DIR}/per_core_allocation/test_matmul_expert.py",
            f"{UNIT_TEST_DIR}/test_dram_matmul_custom_compressed.py",
            f"{UNIT_TEST_DIR}/test_dram_streaming_matmul.py",
            f"{UNIT_TEST_DIR}/test_matmul.py",
            f"{UNIT_TEST_DIR}/test_matmul_custom_compressed.py",
            f"{UNIT_TEST_DIR}/test_moe_routed_expert.py",
            f"{UNIT_TEST_DIR}/test_shared_expert.py",
        ),
    ),
    Bucket(
        name="pure_local_single_device_remainder",
        timeout=600,
        reset_before=False,
        targets=(
            f"{UNIT_TEST_DIR}/per_core_allocation/test_compressed_tensor.py",
            f"{UNIT_TEST_DIR}/per_core_allocation/test_compressed_tensor_multi_device.py",
            f"{UNIT_TEST_DIR}/test_compact_io.py",
            f"{UNIT_TEST_DIR}/test_compressed_tensor.py",
            f"{UNIT_TEST_DIR}/test_create_q_heads.py",
            f"{UNIT_TEST_DIR}/test_deepseek_moe_gate.py",
            f"{UNIT_TEST_DIR}/test_demo.py",
            f"{UNIT_TEST_DIR}/test_eltwise_add.py",
            f"{UNIT_TEST_DIR}/test_flash_mla.py",
            f"{UNIT_TEST_DIR}/test_gated_local_reduce.py",
            f"{UNIT_TEST_DIR}/test_gated_local_reduce_down_proj.py",
            f"{UNIT_TEST_DIR}/test_gather.py",
            f"{UNIT_TEST_DIR}/test_kn_sliced_matmul.py",
            f"{UNIT_TEST_DIR}/test_kv_cache_branch.py",
            f"{UNIT_TEST_DIR}/test_local_reduce.py",
            f"{UNIT_TEST_DIR}/test_mcast.py",
            f"{UNIT_TEST_DIR}/test_mcast_112_core_down.py",
            f"{UNIT_TEST_DIR}/test_rmsnorm.py",
            f"{UNIT_TEST_DIR}/test_rope.py",
            f"{UNIT_TEST_DIR}/test_sdpa.py",
            f"{UNIT_TEST_DIR}/test_sdpa_tail.py",
            f"{UNIT_TEST_DIR}/test_tensor_cache.py",
            f"{UNIT_TEST_DIR}/test_tilize_8x32.py",
            f"{UNIT_TEST_DIR}/test_upload.py",
        ),
    ),
    Bucket(
        name="persistent_loop",
        timeout=120,
        targets=(f"{UNIT_TEST_DIR}/test_persistent_loop.py",),
    ),
    Bucket(
        name="fabric_2d",
        timeout=480,
        targets=(
            f"{UNIT_TEST_DIR}/test_attention_block.py",
            f"{UNIT_TEST_DIR}/test_ccl_all_gather.py",
            f"{UNIT_TEST_DIR}/test_ccl_all_reduce.py",
            f"{UNIT_TEST_DIR}/test_ccl_broadcast.py::test_ccl_broadcast",
            f"{UNIT_TEST_DIR}/test_ccl_broadcast.py::test_ccl_broadcast_loop",
            f"{UNIT_TEST_DIR}/test_ccl_broadcast.py::test_ccl_broadcast_host_iter_stamped_chunks",
            f"{UNIT_TEST_DIR}/test_ccl_broadcast.py::test_ccl_broadcast_remainder_chunk",
            f"{UNIT_TEST_DIR}/test_ccl_broadcast.py::test_ccl_broadcast_auto_chunk",
            f"{UNIT_TEST_DIR}/test_ccl_broadcast.py::test_ccl_broadcast_torus_8x4_functional_fabric_2d",
            f"{UNIT_TEST_DIR}/test_dram_zero_fill.py",
            f"{UNIT_TEST_DIR}/test_lm_head_sampling.py",
            f"{UNIT_TEST_DIR}/test_model.py",
            f"{UNIT_TEST_DIR}/test_moe_mlp.py",
            f"{UNIT_TEST_DIR}/test_pipeline_stage_sync.py",
            f"{UNIT_TEST_DIR}/test_post_sdpa.py",
            f"{UNIT_TEST_DIR}/test_pre_sdpa.py",
            f"{UNIT_TEST_DIR}/test_prepare_weights.py",
            f"{UNIT_TEST_DIR}/test_reduce_to_one_b1.py::test_reduce_to_one_2d",
            f"{UNIT_TEST_DIR}/test_sampling.py",
            f"{UNIT_TEST_DIR}/test_sdpa_reduce_to_all.py",
            f"{UNIT_TEST_DIR}/test_broadcast_rms.py",
            f"{UNIT_TEST_DIR}/test_multi_host_pipeline.py::test_passthrough_pipeline_block",
        ),
    ),
    Bucket(
        name="prepare_weights_hybrid_allocator",
        timeout=480,
        env=(("TT_METAL_ALLOCATOR_MODE_HYBRID", "1"),),
        targets=(
            f"{UNIT_TEST_DIR}/test_prepare_weights.py::test_prepare_attention_weights_dense_4x2",
            f"{UNIT_TEST_DIR}/test_prepare_weights.py::test_prepare_attention_weights_moe_4x2",
            f"{UNIT_TEST_DIR}/test_prepare_weights.py::test_prepare_dense_layer_single_layer_4x2",
            f"{UNIT_TEST_DIR}/test_prepare_weights.py::test_prepare_moe_layer_single_layer_4x2",
            f"{UNIT_TEST_DIR}/test_prepare_weights.py::test_prepare_attention_weights_with_cache_dense_4x2",
            f"{UNIT_TEST_DIR}/test_prepare_weights.py::test_prepare_attention_weights_with_cache_moe_4x2",
            f"{UNIT_TEST_DIR}/test_prepare_weights.py::test_prepare_dense_layer_weights_with_cache_4x2",
            f"{UNIT_TEST_DIR}/test_prepare_weights.py::test_prepare_moe_layer_weights_with_cache_4x2",
        ),
    ),
    Bucket(
        name="fabric_1d",
        timeout=60,
        targets=(f"{UNIT_TEST_DIR}/test_reduce_to_one_b1.py::test_reduce_to_one_1d",),
    ),
    Bucket(
        name="fabric_2d_torus_x",
        timeout=60,
        targets=(
            f"{UNIT_TEST_DIR}/test_ccl_broadcast.py::" "test_ccl_broadcast_torus_8x4_functional_fabric_2d_torus_x",
            f"{UNIT_TEST_DIR}/test_reduce_to_all_b1.py",
        ),
    ),
    Bucket(
        name="fabric_2d_torus_y",
        timeout=120,
        targets=(
            f"{UNIT_TEST_DIR}/test_multi_host_pipeline.py",
            f"{UNIT_TEST_DIR}/test_moe_15_stages.py",
            f"{UNIT_TEST_DIR}/test_bcast_moe_reduce_pipeline.py",
            f"{UNIT_TEST_DIR}/test_broadcast_rms_two_stage_pipeline.py",
            f"--deselect={UNIT_TEST_DIR}/test_multi_host_pipeline.py::test_passthrough_pipeline_block",
        ),
    ),
    Bucket(
        name="fabric_2d_torus_y_xy_skip_only",
        timeout=60,
        targets=(
            f"{UNIT_TEST_DIR}/test_ccl_broadcast.py::" "test_ccl_broadcast_torus_8x4_functional_fabric_2d_torus_y",
            f"{UNIT_TEST_DIR}/test_ccl_broadcast.py::" "test_ccl_broadcast_torus_8x4_functional_fabric_2d_torus_xy",
        ),
    ),
    Bucket(
        name="decoder_integration",
        timeout=600,
        # TODO: Remove the `not mtp_layer_61` exclusion after the MTP decoder cases are burned down.
        pytest_args=("-k", "(unrigged_all_experts or rigged_groups8) and random_weights and not mtp_layer_61"),
        case_isolated=True,
        targets=(f"{UNIT_TEST_DIR}/test_decoder_block.py",),
    ),
    Bucket(
        name="host_io_risk",
        timeout=120,  # 2 minutes
        targets=(
            f"{UNIT_TEST_DIR}/test_host_io.py",
            f"{UNIT_TEST_DIR}/test_broadcast_rms_single_device.py",
        ),
    ),
    # Keep fast-dispatch buckets visually separated from slow-dispatch buckets.
    Bucket(
        name="fast_dispatch",
        timeout=180,
        slow_dispatch=False,
        targets=(
            f"{UNIT_TEST_DIR}/test_ccl_all_gather.py::"
            "test_ccl_all_gather[blackhole-True-device_params0-30-15-1-output_shape0]",
            f"{UNIT_TEST_DIR}/test_ccl_all_reduce.py::"
            "test_ccl_all_reduce[blackhole-True-True-device_params0-30-15-2-0-"
            "DataType.BFLOAT16-Layout.TILE-2-output_shape0-input_shard_shape0-"
            "TensorMemoryLayout.WIDTH_SHARDED]",
            f"{UNIT_TEST_DIR}/test_ccl_broadcast.py::"
            "test_ccl_broadcast[blackhole-True-device_params0-1-30-15-"
            "DataType.BFLOAT16-Layout.TILE-4-2-1-0-output_shape0-input_shard_shape0-"
            "TensorMemoryLayout.WIDTH_SHARDED]",
            f"{UNIT_TEST_DIR}/test_dram_zero_fill.py::"
            "test_dram_zero_fill[blackhole-True-128-1-131072-device_params0]",
            f"{UNIT_TEST_DIR}/test_dram_zero_fill.py::"
            "test_dram_zero_fill[blackhole-True-128-32-131072-device_params0]",
            f"{UNIT_TEST_DIR}/test_dram_zero_fill.py::"
            "test_dram_zero_fill[blackhole-True-128-64-131072-device_params0]",
            f"{UNIT_TEST_DIR}/test_dram_zero_fill.py::"
            "test_dram_zero_fill[blackhole-True-256-1-131072-device_params0]",
            f"{UNIT_TEST_DIR}/test_dram_zero_fill.py::"
            "test_dram_zero_fill[blackhole-True-256-32-131072-device_params0]",
            f"{UNIT_TEST_DIR}/test_dram_zero_fill.py::"
            "test_dram_zero_fill[blackhole-True-256-64-131072-device_params0]",
        ),
    ),
    Bucket(
        name="lm_head_sampling_perf",
        timeout=240,
        slow_dispatch=False,
        env=(("RUN_LM_HEAD_SAMPLING_PERF", "1"),),
        targets=(f"{UNIT_TEST_DIR}/test_lm_head_sampling.py::test_perf",),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    bucket_names = [bucket.name for bucket in BUCKETS]
    parser.add_argument(
        "--bucket",
        action="append",
        choices=bucket_names,
        help="Bucket to run. May be repeated. Defaults to all buckets in isolation order.",
    )
    parser.add_argument(
        "--bucket-except",
        action="append",
        choices=bucket_names,
        default=[],
        help="Bucket to skip. May be repeated. Applied after --bucket selection.",
    )
    parser.add_argument("--list-buckets", action="store_true", help="List buckets and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--no-reset", action="store_true", help="Skip reset/validation/smoke before fabric buckets.")
    parser.add_argument(
        "--smoke-after-reset",
        action="store_true",
        help="Run the smoke pytest after reset/validation before each reset-enabled bucket.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_false",
        dest="continue_on_failure",
        default=True,
        help="Stop after the first bucket failure.",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument appended to every pytest invocation. May be repeated.",
    )
    return parser.parse_args()


def make_env(*, slow_dispatch: bool) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(REPO_ROOT), str(REPO_ROOT / "ttnn"), str(REPO_ROOT / "tools")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["TT_METAL_ALLOCATOR_MODE_HYBRID"] = "1"
    if slow_dispatch:
        env["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
    else:
        env.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
    return env


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(f"+ {quote_command(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


class CollectionError(RuntimeError):
    pass


def collect_nodeids(command: list[str], *, env: dict[str, str], dry_run: bool) -> list[str]:
    print(f"+ {quote_command(command)}", flush=True)
    if dry_run:
        return []

    result = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    result.check_returncode()
    nodeids = [line.strip() for line in result.stdout.splitlines() if ".py::" in line]
    if not nodeids:
        raise CollectionError(
            "pytest collection completed but produced no nodeids; refusing to mark the bucket as passed"
        )
    return nodeids


def reset_validate_smoke(*, env: dict[str, str], dry_run: bool, run_smoke: bool) -> None:
    for attempt in range(1, RESET_VALIDATE_SMOKE_ATTEMPTS + 1):
        print(
            f"Reset/validation/smoke attempt {attempt}/{RESET_VALIDATE_SMOKE_ATTEMPTS}",
            flush=True,
        )
        try:
            run_command(["tt-smi", "-glx_reset_auto"], env=env, dry_run=dry_run)
            run_command(
                [
                    str(REPO_ROOT / "build/tools/scaleout/run_cluster_validation"),
                    "--cabling-descriptor-path",
                    "tools/tests/scaleout/cabling_descriptors/bh_galaxy_xy_torus.textproto",
                    "--hard-fail",
                    "--send-traffic",
                    "--num-iterations",
                    "1",
                ],
                env=env,
                dry_run=dry_run,
            )
            if run_smoke:
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        SMOKE_TEST,
                    ],
                    env=env,
                    dry_run=dry_run,
                )
            return
        except subprocess.CalledProcessError as exc:
            if attempt == RESET_VALIDATE_SMOKE_ATTEMPTS:
                print(
                    f"Reset/validation/smoke failed after {RESET_VALIDATE_SMOKE_ATTEMPTS} attempts",
                    flush=True,
                )
                raise
            print(
                f"Reset/validation/smoke attempt {attempt} failed with exit code {exc.returncode}; retrying",
                flush=True,
            )


def run_case_isolated_bucket(
    bucket: Bucket,
    *,
    args: argparse.Namespace,
    env: dict[str, str],
) -> list[str]:
    nodeids = collect_nodeids(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            # pytest.ini adds -vvs, which makes collect-only print a tree instead of nodeids.
            "-qqq",
            *args.pytest_arg,
            *bucket.pytest_args,
            *bucket.targets,
        ],
        env=env,
        dry_run=args.dry_run,
    )

    failures: list[str] = []
    for case_idx, nodeid in enumerate(nodeids, start=1):
        print(f"\n=== {bucket.name} case {case_idx}/{len(nodeids)} ===", flush=True)
        try:
            run_command(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    f"--timeout={bucket.timeout}",
                    *args.pytest_arg,
                    *bucket.pytest_args,
                    nodeid,
                ],
                env=env,
                dry_run=args.dry_run,
            )
        except subprocess.CalledProcessError as exc:
            failures.append(f"{bucket.name}::{nodeid} exited with {exc.returncode}")
            if not args.continue_on_failure:
                break
            if bucket.reset_before and not args.no_reset and case_idx < len(nodeids):
                reset_validate_smoke(env=env, dry_run=args.dry_run, run_smoke=args.smoke_after_reset)

    return failures


def selected_buckets(names: list[str] | None, excluded_names: list[str]) -> tuple[Bucket, ...]:
    selected = BUCKETS
    if names:
        requested = set(names)
        selected = tuple(bucket for bucket in BUCKETS if bucket.name in requested)

    excluded = set(excluded_names)
    return tuple(bucket for bucket in selected if bucket.name not in excluded)


def print_buckets() -> None:
    for bucket in BUCKETS:
        reset = "reset" if bucket.reset_before else "no-reset"
        dispatch = "slow-dispatch" if bucket.slow_dispatch else "fast-dispatch"
        isolation = "case-isolated" if bucket.case_isolated else "bucket-isolated"
        print(
            f"{bucket.name}: timeout={bucket.timeout}, {reset}, {dispatch}, {isolation}, targets={len(bucket.targets)}"
        )


def main() -> int:
    args = parse_args()
    if args.list_buckets:
        print_buckets()
        return 0

    failures: list[str] = []
    for bucket in selected_buckets(args.bucket, args.bucket_except):
        print(f"\n=== {bucket.name} ===", flush=True)
        env = make_env(slow_dispatch=bucket.slow_dispatch)
        env.update(dict(bucket.env))
        try:
            if bucket.reset_before and not args.no_reset:
                reset_validate_smoke(env=env, dry_run=args.dry_run, run_smoke=args.smoke_after_reset)
            if bucket.case_isolated:
                failures.extend(run_case_isolated_bucket(bucket, args=args, env=env))
                if failures and not args.continue_on_failure:
                    break
                continue
            run_command(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    *(("-x",) if not args.continue_on_failure else ()),
                    f"--timeout={bucket.timeout}",
                    *args.pytest_arg,
                    *bucket.pytest_args,
                    *bucket.targets,
                ],
                env=env,
                dry_run=args.dry_run,
            )
        except CollectionError as exc:
            failures.append(f"{bucket.name} collection failed: {exc}")
            if not args.continue_on_failure:
                break
        except subprocess.CalledProcessError as exc:
            failures.append(f"{bucket.name} exited with {exc.returncode}")
            if not args.continue_on_failure:
                break

    if failures:
        print("\nFailures:", flush=True)
        for failure in failures:
            print(f"- {failure}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
