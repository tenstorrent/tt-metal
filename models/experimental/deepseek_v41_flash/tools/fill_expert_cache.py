# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Fill the weight cache with decoder layers' routed experts, spread over worker processes.

The routed experts are nearly all of a layer's weights (384 x 3 x 5120 x 2304 values, against ~0.3 B
for the attention, shared expert, router, hyper-connections and Engram ``wkv``), and
``DeepSeekV41PreloadedExperts`` converts them one after another. This script writes the same
``layers.N.ffn.experts.E.{gate_up,down}_fused.tp4`` files ahead of time, so the TP4 decoder-layer
test loads every expert from the cache. The layer's other weights are small and are still converted
on the test's first run.

It runs on the host only. Packing bfloat4_b / bfloat8_b reads the L1 alignment from the HAL, which
otherwise brings up the real cluster in every worker, so the workers run tt-metal in mock mode
(``TT_METAL_MOCK_CLUSTER_DESC_PATH``, a Blackhole P150 unless already set) and never touch the
card. Experts already cached are skipped. Each file is written to a scratch directory inside the
cache and renamed into place, so an interrupted run leaves no truncated entries.

The files are the ones experts built for a 1x4 mesh read, one per expert and part holding all four
ranks' shards. ``--tp-size N`` writes a 1xN mesh's instead, and ``--tp-size 1`` the single-chip
files ``test_moe.py`` and the TP1 decoder-layer test read.

    python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --layers 20 21 24
    python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --layers 1 2 --tp-size 1
    python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --all-layers
    python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --dspark
"""

import argparse
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    dspark_expert_provider,
    expert_provider,
    load_config,
)
from models.experimental.deepseek_v41_flash.tt.decode.moe import fused_experts_config, padded_expert_provider
from models.experimental.deepseek_v4_flash.tt.decode.moe import (
    _fused_nd_dram_config,
    _interleave_gate_up,
    _interleave_gate_up_tp,
    _load_fused_weight,
    _swiglu_cols_per_core,
)
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

_PARTS = ("gate_up_fused", "down_fused")
_MOCK_CLUSTER_DESC = (
    Path(__file__).resolve().parents[4]
    / "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
)


def _tp_tag(tp_size: int) -> str:
    return f".tp{tp_size}" if tp_size > 1 else ""


class _ExpertWriter:
    """Writes routed experts into the cache files ``DeepSeekV41PreloadedExperts`` reads, converted
    the way its constructor converts them at ``tp_size``."""

    def __init__(self, snapshot: Path, dtype: ttnn.DataType, cache_dir: str, scratch: str, tp_size: int):
        self.config = load_config(snapshot)
        self.loader = DeepseekV4WeightLoader(snapshot)
        self.dtype = dtype
        self.cache = WeightCache(cache_dir)
        self.scratch = scratch
        self.tp_size = tp_size
        fused_config, sys_cfg = fused_experts_config(self.config, tp_size=tp_size)
        self.padded = fused_config.moe_intermediate_size
        local = self.padded // tp_size
        hidden, num_cores, banks = self.config.hidden_size, sys_cfg.moe.fused_num_cores, sys_cfg.moe.fused_dram_banks
        self.swiglu_cols = _swiglu_cols_per_core(local, num_cores)
        self.gate_up_nd = _fused_nd_dram_config(hidden, 2 * local, 2 * self.swiglu_cols, banks)
        self.down_nd = _fused_nd_dram_config(local, hidden, hidden // num_cores, banks)
        self.gate_up_mapper = self.down_mapper = None
        if tp_size > 1:
            # The shards ``ShardTensorToMesh`` cuts for a 1xTP mesh, built without one.
            self.gate_up_mapper, self.down_mapper = (
                ttnn.create_mesh_mapper(
                    ttnn.MeshShape(1, tp_size),
                    ttnn.MeshMapperConfig([ttnn.PlacementShard(dim)], ttnn.MeshShape([tp_size])),
                )
                for dim in (1, 0)
            )

    def __call__(self, group: str, e: int) -> None:
        """Convert expert ``e`` of ``group``: ``layers.N`` (decoder layer N) or ``mtp.S`` (DSpark stage S)."""
        kind, index = group.split(".")
        source = (expert_provider if kind == "layers" else dspark_expert_provider)(self.loader, int(index))
        provider = padded_expert_provider(source, self.config.moe_intermediate_size, self.padded)
        gate_up, down = provider(e)
        if self.tp_size > 1:
            gate_up = _interleave_gate_up_tp(gate_up, self.tp_size, self.swiglu_cols)
        else:
            gate_up = _interleave_gate_up(gate_up.t().contiguous(), self.swiglu_cols)
        tag = _tp_tag(self.tp_size)
        self._write(group, f"experts.{e}.gate_up_fused{tag}", gate_up, self.gate_up_nd, self.gate_up_mapper)
        self._write(group, f"experts.{e}.down_fused{tag}", down.t().contiguous(), self.down_nd, self.down_mapper)

    def _write(self, group: str, name: str, weight: torch.Tensor, nd_config: ttnn.MemoryConfig, mapper) -> None:
        scratch_file = os.path.join(self.scratch, f"{group}.ffn.{name}")
        _load_fused_weight(weight, None, nd_config, cache_file_name=scratch_file, dtype=self.dtype, mesh_mapper=mapper)
        suffix = f"_dtype_{self.dtype.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin"
        os.replace(scratch_file + suffix, self.cache.sub(f"{group}.ffn").file(name) + suffix)


_writer = None


def _init_worker(snapshot: Path, dtype_name: str, cache_dir: str, scratch: str, threads: int, tp_size: int) -> None:
    global _writer
    torch.set_num_threads(threads)
    _writer = _ExpertWriter(snapshot, getattr(ttnn, dtype_name), cache_dir, scratch, tp_size)


def _fill(group: str, e: int) -> None:
    _writer(group, e)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    layers = parser.add_mutually_exclusive_group(required=True)
    layers.add_argument("--layers", type=int, nargs="+", help="decoder layer ids")
    layers.add_argument("--all-layers", action="store_true", help="every decoder layer of the checkpoint")
    layers.add_argument("--dspark", action="store_true", help="the DSpark drafter's stages (mtp.*)")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count()), help="worker processes")
    parser.add_argument("--dtype", default="bfloat4_b", help="ttnn dtype of the experts, as the test builds them")
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tp-size", type=int, default=4, help="TP ranks of the 1xN mesh the experts are built for")
    parser.add_argument("--force", action="store_true", help="rewrite experts that are already cached")
    args = parser.parse_args()

    snapshot = resolve_snapshot_dir(args.model_dir)
    config = load_config(snapshot)
    if args.dspark:
        groups = {f"mtp.{s}": config.dspark_n_routed_experts for s in range(config.num_nextn_predict_layers)}
    else:
        if args.all_layers:
            args.layers = list(range(config.num_hidden_layers))
        bad = [i for i in args.layers if not 0 <= i < config.num_hidden_layers]
        if bad:
            parser.error(f"layers {bad} outside [0, {config.num_hidden_layers})")
        groups = {f"layers.{i}": config.num_local_experts for i in args.layers}
    dtype = getattr(ttnn, args.dtype)
    cache = WeightCache(args.cache_dir)
    tag = _tp_tag(args.tp_size)
    todo = [
        (group, e)
        for group, count in groups.items()
        for e in range(count)
        if args.force or not all(cache.sub(f"{group}.ffn").hit(f"experts.{e}.{p}{tag}", dtype) for p in _PARTS)
    ]
    total = sum(groups.values())
    logger.info(f"{len(todo)} of {total} experts to convert into {args.cache_dir}, {args.workers} workers")
    if not todo:
        return

    # Before the workers spawn, so each one starts with it.
    os.environ.setdefault("TT_METAL_MOCK_CLUSTER_DESC_PATH", str(_MOCK_CLUSTER_DESC))
    os.makedirs(args.cache_dir, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix=".fill_expert_cache.", dir=args.cache_dir)
    start = time.monotonic()
    try:
        with ProcessPoolExecutor(
            args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(
                snapshot,
                args.dtype,
                args.cache_dir,
                scratch,
                max(1, os.cpu_count() // args.workers),
                args.tp_size,
            ),
        ) as pool:
            futures = [pool.submit(_fill, group, e) for group, e in todo]
            try:
                for n, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if n % 64 == 0 or n == len(todo):
                        logger.info(f"{n}/{len(todo)} experts, {time.monotonic() - start:.0f} s")
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
