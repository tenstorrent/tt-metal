# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test-only explicit whole-trace protocol and independent worker inventory.

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
    down = model.attn_mix.precise_down
    if (down.input_width, down.output_width, down.block) != (10240, 320, 32):
        raise ValueError("Independent GR worker-coverage proof no longer applies")
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
        Path(__file__),
    ]
    return {
        "authority": "harness_trace_protocol_v1",
        "marker_clock": "time.monotonic_ns; ordering only, not a performance measurement",
        "device": chips[0],
        "worker_grid": [grid.x, grid.y],
        "known_cores": cores,
        "coverage_proof": "Every public layer invokes GR down [320,10240]; its FP32 product [1,Q,32,10240] has at least320 tiles. BinaryNg interleaved split assigns nonzero work to every worker in the <=320-core grid. Inventory is established before capture, independently of observed endpoints.",
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
