# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused-stage whole-trace protocol with audited grouped-GR worker coverage.

No Tracy events are synthesized. Monotonic timestamps establish host ordering
only; elapsed performance comes exclusively from native device CSV endpoints.
"""

import hashlib
import json
import struct
import time
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_trace_profiler_canary import ROOT, layout_from_library


def prepare_protocol(mesh, model):
    import ttnn

    mappings = [line for line in Path("/proc/self/maps").read_text().splitlines() if "libtt_metal.so" in line]
    libraries = {Path(line.split()[-1]).resolve() for line in mappings}
    if len(libraries) != 1 or "blackhole" not in str(mesh.arch()).lower():
        raise ValueError("Expected one loaded Metal library and Blackhole worker layout")
    layout = layout_from_library(next(iter(libraries)))
    from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedPreciseProjection
    from models.autoports.qwen_qwen3_8_flash_next.tt.precise_residual_projection import PreciseResidualProjection

    down = model.attn_mix.precise_down
    if (down.input_width, down.output_width) != (10240, 320) or not down.weights:
        raise ValueError("Independent GR worker-coverage dimensions no longer apply")
    weight_shapes = [list(weight.shape) for weight in down.weights]
    rows = [shape[-2] for shape in weight_shapes]
    for weight, shape, count in zip(down.weights, weight_shapes, rows):
        if (
            shape != [1, 1, count, 10240]
            or list(weight.padded_shape) != shape
            or count < 32
            or count % 32
            or weight.dtype != ttnn.float32
            or weight.layout != ttnn.TILE_LAYOUT
            or weight.memory_config() != ttnn.DRAM_MEMORY_CONFIG
            or list(weight.device().get_device_ids()) != list(mesh.get_device_ids())
        ):
            raise ValueError("GR coverage requires complete FP32 TILE DRAM rows on this mesh")
    if sum(rows) != 320:
        raise ValueError("GR coverage requires the complete320-row matrix")
    if type(down) is PreciseResidualProjection:
        if down.block != 32 or rows != [32] * 10:
            raise ValueError("Functional GR geometry differs from the audited implementation")
    elif type(down) is FusedPreciseProjection:
        if (down.parent_block, down.prefill_block, down.chunk_prefill) != (32, 32, False) or rows != [320]:
            raise ValueError("Fused GR geometry differs from the selected grouped implementation")
    else:
        raise ValueError(f"Unaudited GR projection class: {type(down).__name__}")
    grid = mesh.compute_with_storage_grid_size()
    if not 0 < grid.x * grid.y <= 320:
        raise ValueError("GR projection does not prove full worker-grid coverage")
    chips = list(mesh.get_device_ids())
    if len(chips) != 1:
        raise ValueError("Single-device protocol only")
    cores = []
    for y in range(grid.y):
        for x in range(grid.x):
            physical = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            cores.append({"logical": [x, y], "translated": [physical.x, physical.y]})
    files = [
        ROOT / "models/autoports/qwen_qwen3_8_flash_next/tt/precise_residual_projection.py",
        ROOT / "models/autoports/qwen_qwen3_8_flash_next/tt/decoder_components.py",
        ROOT / "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp",
        ROOT / "tt_metal/common/work_split.cpp",
        ROOT / "models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py",
        ROOT / "models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py",
        ROOT / "models/autoports/qwen_qwen3_8_flash_next/tests/profile_fused_trace_decoder.py",
        ROOT / "models/autoports/qwen_qwen3_8_flash_next/tests/run_fused_probe.py",
        ROOT / "ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp",
        ROOT / "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp",
        ROOT / "tt_metal/distributed/mesh_device.cpp",
        ROOT / "tt_metal/impl/sub_device/sub_device_manager_tracker.cpp",
        Path(__file__),
    ]
    return {
        "authority": "harness_trace_protocol_v1",
        "marker_clock": "time.monotonic_ns; ordering only, not a performance measurement",
        "device": chips[0],
        "worker_grid": [grid.x, grid.y],
        "known_cores": cores,
        "coverage_projection": {
            "class": type(down).__name__,
            "weight_shapes": weight_shapes,
            "weight_rows": rows,
            "input_family": "Public TILE [1,1,Q,10240], logical Q=1..32; decode Q=1",
            "prefill_block": 32,
            "conservative_minimum_product_tiles": 320,
            "grid_scope": "Fixture opens the default full-grid sub-device; no custom manager or sub-grid override",
        },
        "coverage_proof": "Each public pass invokes GR down K10240/N320. Functional products use B32 rows; fused Q1 uses its grouped B320 rows and fused Q>1 uses B32 slices. FP32 product [1,Q,B,10240] has320*Q*(B/32) physical tiles, at least320. The interleaved BinaryNg CoreRangeSet split assigns positive work to every worker in the default full grid of at most320cores. Inventory precedes capture and is independent of observed endpoints.",
        "coverage_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
        "loaded_mappings": mappings,
        "layout": layout,
        "events": [],
        "predrain": [],
    }


def record_event(protocol, name, **fields):
    protocol["events"].append({"name": name, "monotonic_ns": time.monotonic_ns(), **fields})


def sample_predrain(protocol, ordinal):
    """Read only: caller has completed/synchronized replay; no model operation."""
    import ttnn

    layout = protocol["layout"]
    samples = []
    for core in protocol["known_cores"]:
        x, y = core["translated"]
        control = list(
            struct.unpack(
                "<" + "I" * layout["control_words"],
                ttnn._ttnn.cluster.read_from_core(
                    protocol["device"], x, y, layout["control_address"], 4 * layout["control_words"]
                ),
            )
        )
        markers = list(
            struct.unpack(
                "<12I", ttnn._ttnn.cluster.read_from_core(protocol["device"], x, y, layout["aggregate_address"], 48)
            )
        )
        samples.append({**core, "control": control, "markers": markers})
    protocol["predrain"].append({"ordinal": ordinal, "sampled_monotonic_ns": time.monotonic_ns(), "cores": samples})


def save_partial(report, path):
    Path(path).write_text(json.dumps(report, indent=2) + "\n")
