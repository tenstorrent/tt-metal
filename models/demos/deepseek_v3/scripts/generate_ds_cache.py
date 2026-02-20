# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape

optimal_topology = (
    ttnn.FabricConfig.FABRIC_1D_RING if (os.getenv("USE_TORUS_MODE") is not None) else ttnn.FabricConfig.FABRIC_1D
)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Generate DeepSeek-V3 TT-NN cache")
    p.add_argument("--model-path", type=str, required=True, help="Path to local HF DeepSeek-V3 model (safetensors)")
    p.add_argument("--cache-dir", type=str, required=True, help="Destination cache directory")
    p.add_argument("--temp-cache-dir", type=str, required=True, help="Temporary cache directory to be written to")
    p.add_argument(
        "--archive-dir",
        type=str,
        help="Directory to archive overwritten cache files into a timestamped subdirectory. "
        "If not provided, existing files in cache-dir will be overwritten without backup.",
    )
    p.add_argument("--override-num-layers", type=int, help="Override the number of layers in the model.")
    return p


def move_cache(temp_cache_dir: Path, cache_dir: Path, archive_dir: Optional[Path]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Moving generated cache from '{temp_cache_dir}' to '{cache_dir}'")
    for src in temp_cache_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(temp_cache_dir)
        dst = cache_dir / rel
        if dst.exists() and archive_dir is not None:
            # moves from cache_dir to archive_dir
            archive_dst = archive_dir / timestamp / rel
            archive_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(archive_dst))

        # moves from temp_cache_dir to cache_dir
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    shutil.rmtree(str(temp_cache_dir))


def main() -> None:
    args = create_parser().parse_args()

    model_path = Path(args.model_path)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temp_cache_dir = Path(args.temp_cache_dir)
    temp_cache_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = Path(args.archive_dir) if args.archive_dir is not None else None

    if archive_dir is None:
        logger.warning(
            "No --archive-dir specified. Any existing files in cache-dir that conflict with "
            "newly generated cache files will be overwritten and lost."
        )

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        raise ValueError("Environment variable $MESH_DEVICE is not set. Please set it to DUAL, QUAD, or TG.")
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    logger.info(f"Selected MESH_DEVICE: '{requested_system_name}' - mesh shape will be set to: {mesh_shape}")
    fabric_config = optimal_topology
    logger.info(f"Setting fabric config to {fabric_config} for demo run")
    ttnn.set_fabric_config(fabric_config, ttnn.FabricReliabilityMode.RELAXED_INIT)

    logger.info(f"Opening mesh device with shape {mesh_shape}")
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    try:
        logger.info(f"Generating cache in {temp_cache_dir}")
        gen = DeepseekGenerator(
            mesh_device=mesh_device,
            model_path=model_path,
            cache_dir=temp_cache_dir,
            override_num_layers=args.override_num_layers,
        )
        logger.info(f"Cache was generated in {temp_cache_dir}")
    finally:
        ttnn.synchronize_device(mesh_device)
        for submesh in mesh_device.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    if ttnn.distributed_context_is_initialized():
        logger.info("Waiting for all hosts to finish cache generation and validation before moving files")
        ttnn.distributed_context_barrier()

    if (not ttnn.distributed_context_is_initialized()) or (int(ttnn.distributed_context_get_rank()) == 0):
        move_cache(temp_cache_dir, cache_dir, archive_dir)

    logger.info("Cache generation complete.")


if __name__ == "__main__":
    main()
