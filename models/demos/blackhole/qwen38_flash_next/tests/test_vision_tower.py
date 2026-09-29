# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The vision tower on the device against ``vision_reference.py`` (the CPU oracle proven against Transformers).

Per pinned synthetic grid (``tests/fixtures/vision_fixture.py``): the op pins at head_dim 96 (block 0's and block 26's
attention alone and MLP alone, the MLP with the fused GELU-tanh), every block alone on the reference's input, the whole
tower block by block (the accumulated BF16 residual stream) and at the merger output (PCC, max abs error, NMSE), the
MEASURED resident DRAM bytes, the allocator peak during a forward (sampled after every transient buffer), the cold
(first call) and warm wall times, the steady state of the allocator over repeated forwards.  Grids too large for the
CPU reference (16,384 and 65,536 patches, the stock maximum) run the time / memory part only.

Thresholds are measured floors, not tuned targets.  The one-die records of 2026-09-29 (bitwise reproducible, the
same values on the four-die line) read: attention or MLP alone PCC 0.999973 .. 1.000000; every block alone 0.999897 ..
1.000000 (the minimum is block 0 on the padded 396-patch grid); the accumulated stream through block 8 >= 0.99958;
merger features 0.998154 (1024 patches), 0.997557 (4096), 0.996807 (960, non-square), 0.997176 (396, padded).  The
floors sit one decade of the observed spread under those minima: 0.9998 for anything alone, 0.999 for the early stream,
0.995 for the features.  (The Wormhole port of the same tower gates a block alone at 0.99 and the whole model at
0.91-class.)  Every measured value is printed as a table and written to ``$QWEN38_VISION_RECORD_DIR/<grid>-<mesh>.json``
when that directory is set.

Run (a held die or line; the no-device sets are masked):
  QWEN38_FUSED_DEVICE_TEST=1 QWEN38_VISION_MESH=1x1 pytest tests/test_vision_tower.py -s
Env: QWEN38_CHECKPOINT (or MODEL_WEIGHTS_DIR); QWEN38_VISION_MESH 1x1 (one die) or 1x4 (the line); QWEN38_VISION_GRIDS
restricts the fixtures (comma separated; default: every pinned fixture).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.tests.fixtures import vision_fixture
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS, checkpoint_root, pcc
from models.demos.blackhole.qwen38_flash_next.ttnn.vision import VisionTower
from models.demos.blackhole.qwen38_flash_next.vision_reference import VisionTowerConfig, VisionTowerReference

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1",
    reason="a device test: set QWEN38_FUSED_DEVICE_TEST=1 on a held die or line (the no-device sets are masked)",
)

MESH = tuple(int(v) for v in os.environ.get("QWEN38_VISION_MESH", "1x1").split("x"))
GRIDS = tuple(
    os.environ.get("QWEN38_VISION_GRIDS", ",".join(spec.name for spec in vision_fixture.PINNED_FIXTURES)).split(",")
)
SINGLE_DEVICE_PARAMS = {"l1_small_size": DEVICE_PARAMS["l1_small_size"], "trace_region_size": 0}
PARAMS = DEVICE_PARAMS if MESH == (1, 4) else SINGLE_DEVICE_PARAMS

ALONE_PCC = 0.9998  # a block, an attention or an MLP alone on the reference's input (measured minimum 0.999897)
EARLY_STREAM_PCC = 0.999  # the accumulated stream through block 8 (measured minimum 0.99958)
FEATURES_PCC = 0.995  # the merger features (measured minimum 0.996807, the non-square grid)
PIN_BLOCKS = (0, 26)  # op pins: the first block and the one with the largest residual values


def _record_dir() -> Path | None:
    value = os.environ.get("QWEN38_VISION_RECORD_DIR")
    return Path(value) if value else None


def _dram_view(mesh_device) -> dict[str, int]:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return {
        "num_banks": int(view.num_banks),
        "allocated_per_bank": int(view.total_bytes_allocated_per_bank),
        "free_per_bank": int(view.total_bytes_free_per_bank),
        "largest_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
    }


def _error_row(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | str]:
    actual, expected = actual.to(torch.float32), expected.to(torch.float32)
    diff = (actual - expected).abs()
    return {
        "stage": name,
        "pcc": pcc(actual, expected),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "nmse": float((diff**2).sum() / (expected**2).sum().clamp_min(1e-30)),
        "ref_abs_max": float(expected.abs().max()),
    }


def _print_rows(title: str, rows: list[dict]) -> None:
    logger.info(title)
    logger.info(f"{'stage':>14s} {'pcc':>10s} {'max|err|':>10s} {'mean|err|':>10s} {'nmse':>10s} {'|ref|max':>10s}")
    for row in rows:
        logger.info(
            f"{row['stage']:>14s} {row['pcc']:10.6f} {row['max_abs_err']:10.4g} {row['mean_abs_err']:10.4g} "
            f"{row['nmse']:10.3e} {row['ref_abs_max']:10.4g}"
        )


class _Peak:
    """The allocator's lowest free bytes per bank seen by a probe, and where."""

    def __init__(self, mesh_device, baseline: dict[str, int]):
        self.mesh_device = mesh_device
        self.baseline = baseline
        self.min_free = baseline["free_per_bank"]
        self.min_largest = baseline["largest_free_per_bank"]
        self.stage = "start"

    def __call__(self, stage: str) -> None:
        view = _dram_view(self.mesh_device)
        if view["free_per_bank"] < self.min_free:
            self.min_free, self.min_largest, self.stage = view["free_per_bank"], view["largest_free_per_bank"], stage

    def record(self, rows: int) -> dict[str, float | str]:
        per_bank = self.baseline["free_per_bank"] - self.min_free
        return {
            "peak_activation_bytes_per_bank": per_bank,
            "peak_activation_bytes_per_die": per_bank * self.baseline["num_banks"],
            "peak_activation_bytes_per_patch_per_die": per_bank * self.baseline["num_banks"] / rows,
            "min_largest_contiguous_free_per_bank": self.min_largest,
            "stage": self.stage,
        }


@pytest.fixture(scope="module")
def oracle():
    checkpoint = Qwen38Checkpoint(checkpoint_root())
    config = VisionTowerConfig.from_checkpoint(checkpoint.root)
    state = checkpoint.vision_state_dict()
    return config, state, VisionTowerReference(state, config)


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("device_params", [PARAMS], indirect=True)
@pytest.mark.parametrize("grid", GRIDS)
def test_vision_tower(mesh_device, grid, oracle):
    config, state, reference = oracle
    spec = vision_fixture.fixture(grid)
    patches, grid_thw = vision_fixture.pixel_patches(spec, config)
    load = os.getloadavg()[0]
    record: dict = {
        "grid": grid,
        "grid_thw": grid_thw.tolist(),
        "patches": spec.patches,
        "tokens": spec.merged_tokens,
        "mesh": list(MESH),
        "load_1min": load,
        "reference": spec.reference,
    }

    trace = None
    if spec.reference:
        started = time.perf_counter()
        trace = reference.forward_trace(patches, grid_thw)
        record["reference_seconds"] = time.perf_counter() - started

    tower = VisionTower(mesh_device, state, config)
    before = _dram_view(mesh_device)
    started = time.perf_counter()
    modeled = tower.load()
    ttnn.synchronize_device(mesh_device)
    record["load_seconds"] = time.perf_counter() - started
    after = _dram_view(mesh_device)
    measured_per_bank = after["allocated_per_bank"] - before["allocated_per_bank"]
    record["resident_bytes_modeled_per_die"] = modeled
    record["resident_bytes_measured_per_bank"] = measured_per_bank
    record["resident_bytes_measured_per_die"] = measured_per_bank * after["num_banks"]
    record["dram_free_per_bank_after_load"] = after["free_per_bank"]
    record["dram_largest_free_per_bank_after_load"] = after["largest_free_per_bank"]
    logger.info(
        f"{grid} ({spec.patches} patches, {spec.merged_tokens} tokens): weights resident: MODELED {modeled / 1e6:.1f} MB per die, "
        f"MEASURED {measured_per_bank * after['num_banks'] / 1e6:.1f} MB per die ({measured_per_bank / 1e6:.1f} MB per bank x "
        f"{after['num_banks']}), load {record['load_seconds']:.1f} s"
    )
    pin_rows: list[dict] = []
    alone_rows: list[dict] = []
    stream_rows: list[dict] = []
    merger_row: dict | None = None
    try:
        # 1. Cold: the first forward at this shape (program compile from the JIT cache, host work), untraced, with the
        #    allocator sampled after every transient buffer: the peak of record.
        cold_peak = _Peak(mesh_device, after)
        output = tower.run_image(patches, grid_thw, probe=cold_peak)
        record["timing_cold"] = output.timing
        features = tower.features_to_torch(output)
        assert tuple(features.shape) == (spec.merged_tokens, config.out_hidden_size)
        assert torch.isfinite(features).all(), "the features have a non-finite value"
        ttnn.deallocate(output.features)
        record["activation_peak"] = cold_peak.record(output.rows)
        free_after_cold = _dram_view(mesh_device)["free_per_bank"]

        if spec.reference:
            # 2. Op pins at head_dim 96: attention alone and the MLP alone (fused GELU-tanh) on two blocks.
            block_inputs = [trace.embed] + trace.blocks[:-1]
            cos, sin, segments = reference.positions(grid_thw)
            for index in PIN_BLOCKS:
                normed1 = reference.block_norm(index, 1, block_inputs[index])
                attention_ref = reference.attention(index, normed1, cos, sin, segments)
                pin_rows.append(
                    _error_row(f"attn{index}", tower.run_attention(index, normed1, grid_thw), attention_ref)
                )
                normed2 = reference.block_norm(index, 2, block_inputs[index] + attention_ref)
                pin_rows.append(_error_row(f"mlp{index}", tower.run_mlp(index, normed2), reference.mlp(index, normed2)))
            _print_rows(f"{grid}: attention and MLP alone (op pins at head_dim 96, fused GELU-tanh)", pin_rows)
            record["op_pins"] = pin_rows

            # 3. Every block alone on the reference's input (the block's own numerics, BF16 in and out).
            for index in range(config.depth):
                actual = tower.run_block(index, block_inputs[index], grid_thw)
                alone_rows.append(_error_row(f"block{index}", actual, trace.blocks[index]))
            _print_rows(f"{grid}: each block alone on the reference input", alone_rows)
            record["blocks_alone"] = alone_rows

            # 4. The whole tower with every block output kept: the accumulated BF16 residual stream and the merger.
            traced = tower.run_image(patches, grid_thw, trace=True)
            record["timing_traced"] = traced.timing
            stream_rows.append(
                _error_row("embed", tower.replicated_to_torch(traced.trace["embed"], spec.patches), trace.embed)
            )
            for index, tensor in enumerate(traced.trace["blocks"]):
                stream_rows.append(
                    _error_row(f"block{index}", tower.replicated_to_torch(tensor, spec.patches), trace.blocks[index])
                )
            features_traced = tower.features_to_torch(traced)
            merger_row = _error_row("features", features_traced, trace.features)
            _print_rows(f"{grid}: the accumulated stream and the merger", stream_rows + [merger_row])
            record["stream"] = stream_rows
            record["features"] = merger_row
            record["traced_equals_cold_bitwise"] = bool(torch.equal(features_traced, features))
            for tensor in [traced.trace["embed"], *traced.trace["blocks"], traced.features]:
                ttnn.deallocate(tensor)

        # 5. Warm, twice: weights resident and programs compiled; the second run proves the allocator's steady state.
        warm, free_after_warm = [], []
        for _ in range(2):
            output = tower.run_image(patches, grid_thw)
            warm.append(output.timing)
            features_warm = tower.features_to_torch(output)
            ttnn.deallocate(output.features)
            free_after_warm.append(_dram_view(mesh_device)["free_per_bank"])
        record["timing_warm"] = warm
        record["warm_equals_cold_bitwise"] = bool(torch.equal(features_warm, features))
        record["free_per_bank_after_cold_and_warm_runs"] = [free_after_cold, *free_after_warm]
        peak = record["activation_peak"]
        logger.info(
            f"{grid}: cold {record['timing_cold']['total_s']:.3f} s (device {record['timing_cold']['device_s']:.3f}, host prepare "
            f"{record['timing_cold']['host_prepare_s']:.3f}, upload {record['timing_cold']['upload_s']:.3f}); warm "
            f"{[round(t['total_s'], 3) for t in warm]} s (device {[round(t['device_s'], 3) for t in warm]}); peak activations "
            f"{peak['peak_activation_bytes_per_bank'] / 1e6:.1f} MB per bank = {peak['peak_activation_bytes_per_patch_per_die'] / 1e3:.1f} KB "
            f"per patch per die at {peak['stage']} (largest contiguous free there {peak['min_largest_contiguous_free_per_bank'] / 1e6:.0f} MB per bank); "
            f"1-min load {load:.1f}"
        )
    finally:
        tower.free()
        ttnn.synchronize_device(mesh_device)
        record["dram_free_per_bank_after_free"] = _dram_view(mesh_device)["free_per_bank"]
        record["bytes_per_bank_held_after_free"] = before["free_per_bank"] - record["dram_free_per_bank_after_free"]
        logger.info(
            f"{grid}: DRAM per bank held after free(): {record['bytes_per_bank_held_after_free']} B (one-time op buffers)"
        )
        directory = _record_dir()
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"{grid}-{MESH[0]}x{MESH[1]}.json").write_text(json.dumps(record, indent=2, sort_keys=True))

    runs = record["free_per_bank_after_cold_and_warm_runs"]
    assert runs[1] == runs[2], f"DRAM grows per forward: {runs}"
    assert record["warm_equals_cold_bitwise"], "the warm forward differs from the cold one"
    if not spec.reference:
        return
    assert record["traced_equals_cold_bitwise"], "keeping the block outputs changed the features"
    weak = [row for row in pin_rows + alone_rows if row["pcc"] < ALONE_PCC]
    assert not weak, f"under PCC {ALONE_PCC} alone: {[(r['stage'], round(r['pcc'], 4)) for r in weak]}"
    weak_early = [row for row in stream_rows[1:10] if row["pcc"] < EARLY_STREAM_PCC]
    assert (
        not weak_early
    ), f"accumulated stream under PCC {EARLY_STREAM_PCC} before block 9: {[(r['stage'], round(r['pcc'], 4)) for r in weak_early]}"
    assert merger_row["pcc"] >= FEATURES_PCC, f"merger features PCC {merger_row['pcc']:.4f} under {FEATURES_PCC}"
    assert math.isfinite(merger_row["nmse"])
