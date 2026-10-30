# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Keep the Engram n-gram tables in RAM between processes.

Every model, test and demo that uses Engram otherwise copies both layers' tables (~189 GiB) from the
checkpoint into its own memory on start-up: ~15 s with the shards in the page cache, ~40 s from NVMe,
and twice the RAM while the page cache still holds them. This copies the checkpoint files holding the
tables into ``STAGE_DIR`` (``/dev/shm/deepseek_v41_engram``, or ``$DEEPSEEK_V41_ENGRAM_STAGE_DIR``), a
tmpfs, where they stay resident until removed or the host reboots. ``load_engram_table`` then maps the
copies instead, so every process shares the one copy.

The copies hold ~203 GB of RAM for as long as they exist, used or not: remove them with ``--remove``
when done. It runs on the host only.

    python models/experimental/deepseek_v41_flash/tools/stage_engram_tables.py
    python models/experimental/deepseek_v41_flash/tools/stage_engram_tables.py --remove
"""

import argparse
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.engram_lookup import STAGE_DIR, staged_copy
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader


def _copy(src: Path, dst: Path, threads: int, chunk: int = 1 << 30) -> None:
    """``src`` copied to ``dst`` by ``threads`` ``sendfile`` copies of ``chunk`` bytes."""
    size = src.stat().st_size
    with open(dst, "wb") as f:
        f.truncate(size)

    def copy(begin):
        end = min(begin + chunk, size)
        with open(src, "rb", buffering=0) as fin, open(dst, "r+b", buffering=0) as fout:
            fout.seek(begin)
            while begin < end:
                n = os.sendfile(fout.fileno(), fin.fileno(), begin, end - begin)
                assert n, f"short read of {src} at {begin}"
                begin += n

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(copy, range(0, size, chunk)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--remove", action="store_true", help="delete the staged copies instead")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    loader = DeepseekV4WeightLoader(args.model_dir)
    config = load_config(loader.snapshot_dir)
    shards = sorted(
        {
            Path(loader.shard_of(f"layers.{i}.engram.embed.{part}", translate=False)).resolve()
            for i in config.engram_layer_ids
            for part in ("weight", "scale")
        }
    )

    if args.remove:
        for shard in shards:
            copy = STAGE_DIR / shard.name
            if copy.exists():
                copy.unlink()
                logger.info(f"removed {copy}")
        return

    todo = [shard for shard in shards if staged_copy(shard) is None]
    logger.info(f"Engram layers {config.engram_layer_ids}: {len(shards) - len(todo)} of {len(shards)} files staged")
    if not todo:
        return
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    need = sum(shard.stat().st_size for shard in todo)
    free = shutil.disk_usage(STAGE_DIR).free
    if free < need:
        raise SystemExit(f"{STAGE_DIR} has {free / 1e9:.0f} GB free; staging needs {need / 1e9:.0f} GB")
    for shard in todo:
        dst = STAGE_DIR / shard.name
        partial = dst.with_name(f".{dst.name}.partial")
        start = time.monotonic()
        try:
            _copy(shard, partial, args.threads)
            partial.chmod(0o644)
            os.replace(partial, dst)
        finally:
            partial.unlink(missing_ok=True)
        seconds = time.monotonic() - start
        size = shard.stat().st_size
        logger.info(f"staged {shard.name}: {size / 1e9:.0f} GB in {seconds:.0f}s ({size / 1e9 / seconds:.1f} GB/s)")


if __name__ == "__main__":
    main()
