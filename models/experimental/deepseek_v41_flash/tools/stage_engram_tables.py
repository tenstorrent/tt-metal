# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Keep the Engram n-gram tables in RAM between processes.

Every model, test and demo that uses Engram otherwise copies both layers' tables (~203 GB) from the
checkpoint into its own memory on start-up: ~15 s with the shards in the page cache, ~40 s from NVMe,
and twice the RAM while the page cache still holds them. This copies the checkpoint files holding the
    10|tables into ``STAGE_DIR`` (``/dev/shm/deepseek_v41_engram``, or ``$DEEPSEEK_V41_ENGRAM_STAGE_DIR``), a
tmpfs, where they stay resident until removed or the host reboots. ``load_engram_table`` then maps the
copies instead, so every process shares the one copy.

``demo/server.py`` and the full-model tests stage the tables themselves before building, so this tool is
for staging ahead of time -- once per host before a benchmark run, say -- or for dropping the copies. It
runs on the host only and opens no device.

The copies hold ~203 GB of RAM for as long as they exist, used or not: remove them with ``--remove``
when done.

    python models/experimental/deepseek_v41_flash/tools/stage_engram_tables.py
    python models/experimental/deepseek_v41_flash/tools/stage_engram_tables.py --remove
"""

import argparse
import os
from pathlib import Path

from loguru import logger

from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    STAGE_DIR,
    engram_shards,
    stage_engram_tables,
    staged_copy,
)
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--remove", action="store_true", help="delete the staged copies instead")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    loader = DeepseekV4WeightLoader(args.model_dir)
    config = load_config(loader.snapshot_dir)
    shards = engram_shards(loader, config.engram_layer_ids)

    if args.remove:
        for shard in shards:
            copy = STAGE_DIR / shard.name
            if copy.exists():
                copy.unlink()
                logger.info(f"removed {copy}")
        return

    staged = sum(staged_copy(shard) is not None for shard in shards)
    logger.info(f"Engram layers {config.engram_layer_ids}: {staged} of {len(shards)} files staged")
    # A build leaves the staging to its own flag; this tool stages whenever it is asked to.
    os.environ["DEEPSEEK_V41_ENGRAM_STAGE"] = "1"
    _, copied = stage_engram_tables(loader, config.engram_layer_ids, threads=args.threads)
    if not copied:
        logger.info(f"nothing to copy: the tables are already in {STAGE_DIR}")


if __name__ == "__main__":
    main()
