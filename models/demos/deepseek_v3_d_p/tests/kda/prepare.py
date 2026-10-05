# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only preparation of KDA weight caches and CPU references; opens no device.

Run from the tt-metal checkout root, in the same checkout (and ttnn model cache) as the device tests:

    python -m models.demos.deepseek_v3_d_p.tests.kda.prepare --case kimi_k3-synthetic-mesh2x4-tpaxis1-T1280
    python -m models.demos.deepseek_v3_d_p.tests.kda.prepare --all --weights synthetic
    python -m models.demos.deepseek_v3_d_p.tests.kda.prepare --list

Case names come from ``tests/kda/cases.py::KDA_CASES``; a device test that misses a cache names the command
for its case. Real-weight cases need ``KIMI_K3_CKPT``. Device tests then run load-only (the default
``KDA_CACHE_MISS=fail``), e.g.

    scripts/run_safe_pytest.sh <exact test ids> -vv

Host tilization initializes the TT-Metal runtime, so this command runs it against a mock cluster
(``TT_METAL_MOCK_CLUSTER_DESC_PATH``, defaulting to the LoudBox 8xP150 descriptor) and verifies on exit that
it holds no device handle.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_MOCK_CLUSTER = (
    _REPOSITORY_ROOT
    / "tt_metal/third_party/tt-cluster-descriptors/blackhole/blackhole_8xP150_cluster_desc/blackhole_8xP150.yaml"
)
# Must precede the first TT-Metal runtime initialization (any TILE-layout host conversion).
os.environ.setdefault("TT_METAL_MOCK_CLUSTER_DESC_PATH", str(_DEFAULT_MOCK_CLUSTER))

import argparse  # noqa: E402
import time  # noqa: E402

from loguru import logger  # noqa: E402

from models.demos.deepseek_v3_d_p.tests.kda.cases import (  # noqa: E402
    KDA_CASES,
    KDACaseSpec,
    build_kda_case,
    kda_weight_cache_dir,
)
from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import prepare_cpu_references  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights  # noqa: E402


def _device_handles() -> list[str]:
    handles = []
    for descriptor in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            continue
        if "tenstorrent" in target:
            handles.append(target)
    return handles


def prepare_case(spec: KDACaseSpec, checkpoint_dir: Path | None) -> None:
    """Write the case's weight cache for its mesh placement and every chained CPU reference."""
    case = build_kda_case(spec, checkpoint_dir)
    cache_dir = kda_weight_cache_dir(case.weights, spec.mesh_shape, spec.tensor_parallel_axis)
    prefix = f"layer_{case.weights.layer_idx}.kda"
    start = time.perf_counter()
    if KDAWeights.check_cache_complete(
        cache_dir, prefix, case.config, spec.mesh_shape, tensor_parallel_axis=spec.tensor_parallel_axis
    ):
        logger.info(f"KDA prepare {spec.name}: weight cache hit {cache_dir}")
    else:
        logger.info(f"KDA prepare {spec.name}: weight cache miss, writing {cache_dir}")
        KDAWeights.build_ttnn_cache(
            case.weights.load_state_dict(),
            cache_dir,
            prefix,
            case.config,
            spec.mesh_shape,
            tensor_parallel_axis=spec.tensor_parallel_axis,
        )
        logger.info(f"KDA prepare {spec.name}: weight cache written in {time.perf_counter() - start:.1f} s")
    references = prepare_cpu_references(case)
    hits = sum(reference.cache_hit for reference in references)
    logger.info(
        f"KDA prepare {spec.name}: CPU references {hits}/{len(references)} hits, "
        f"{sum(reference.seconds for reference in references):.1f} s"
    )


def _selected_specs(arguments: argparse.Namespace) -> list[KDACaseSpec]:
    if arguments.all:
        specs = list(KDA_CASES.values())
    else:
        unknown = [name for name in arguments.case if name not in KDA_CASES]
        if unknown:
            raise SystemExit(f"unknown KDA case(s) {unknown}; see --list")
        specs = [KDA_CASES[name] for name in arguments.case]
    if arguments.weights is not None:
        specs = [spec for spec in specs if spec.weights == arguments.weights]
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--case", action="append", default=[], help="registered case name (repeatable)")
    selection.add_argument("--all", action="store_true", help="every registered case")
    selection.add_argument("--list", action="store_true", help="print registered case names and exit")
    parser.add_argument("--weights", choices=("synthetic", "real"), help="restrict to one weight source")
    arguments = parser.parse_args()
    if arguments.list:
        print("\n".join(KDA_CASES))
        return

    specs = _selected_specs(arguments)
    checkpoint = os.environ.get("KIMI_K3_CKPT")
    if any(spec.weights == "real" for spec in specs) and checkpoint is None:
        raise SystemExit("real-weight cases need KIMI_K3_CKPT")
    logger.info(f"KDA prepare: mock cluster {os.environ['TT_METAL_MOCK_CLUSTER_DESC_PATH']}")
    for index, spec in enumerate(specs, start=1):
        logger.info(f"KDA prepare [{index}/{len(specs)}] {spec.name} start")
        prepare_case(spec, Path(checkpoint) if checkpoint else None)
        logger.info(f"KDA prepare [{index}/{len(specs)}] {spec.name} done")
    handles = _device_handles()
    if handles:
        raise SystemExit(f"KDA prepare opened device handles {handles}; it must run without a device")
    logger.info("KDA prepare: done; no device handle was opened")


if __name__ == "__main__":
    main()
