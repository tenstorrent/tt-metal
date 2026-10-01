# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Source FP32 words and real-norm numerical controls under trace/address reuse.

The inherited three-iteration inverse does not meet 1e-4 over every mantissa.
That earlier wide-domain failure remains recorded. Wide fixtures test exact
source words; the independently accurate norm mantissa keeps the 1e-4 gate.
"""
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

KERNELS = Path(__file__).parent / "kernels/source_sfpu_rounding"
FIXTURES = Path(__file__).parent / "data/source_sfpu_rounding"
pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held Blackhole four-die mesh"
)


def sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
@pytest.mark.parametrize("mode", ["reciprocal_source_words", "reciprocal_norm_scaling", "softplus_source_words"])
def test_source_rounding_regression(mesh_device, mode, tmp_path):
    mesh_device.enable_program_cache()
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    out = tmp_path
    owned, traces, addresses, results = [], [], [], []
    cached = None
    try:
        for seed in (20260930, 20261001):
            softplus = mode == "softplus_source_words"
            if mode == "reciprocal_norm_scaling":
                generator = torch.Generator().manual_seed(seed)
                powers = torch.randint(-12, 13, (1, 1, 32, 32), generator=generator)
                signs = torch.randint(0, 2, powers.shape, generator=generator) * 2 - 1
                powers.flatten()[0], signs.flatten()[0] = 0, 1
                base = torch.tensor([0x3A834422], dtype=torch.int32).view(torch.float32)[0]
                source_result = torch.tensor([0x4479A001], dtype=torch.int32).view(torch.float32)[0]
                value = (base * torch.exp2(powers.float()) * signs).float()
                wanted = (source_result * torch.exp2(-powers.float()) * signs).float()
            else:
                report = json.loads((FIXTURES / "manifest.json").read_text())
                row = next(
                    r
                    for r in report["fixtures"]
                    if r["operator"] == ("softplus" if softplus else "reciprocal") and r["seed"] == seed
                )
                fixture = FIXTURES / row["file"]
                assert hashlib.sha256(fixture.read_bytes()).hexdigest() == row["artifact_sha256"]
                saved = torch.load(fixture, weights_only=True)
                value, wanted = saved["input"], saved["output"]
                assert sha(wanted) == row["output_sha256"] and sha(value) == row["input_sha256"]
            source = ttnn.from_torch(
                value,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            output = fp.allocate(value.shape, ttnn.float32, ttnn.TILE_LAYOUT, mesh_device)
            owned.extend((source, output))
            addresses.append(source.buffer_address())
            assert len(set(addresses)) == len(addresses)
            reader = fp.reader_kernel(
                str(KERNELS / "reader.cpp"), cores, fp.accessor_args(source), [(core, [source.buffer_address(), 1])]
            )
            writer = fp.writer_kernel(
                str(KERNELS / "writer.cpp"), cores, fp.accessor_args(output), [(core, [output.buffer_address(), 1])]
            )

            def descriptor(name):
                compute = fp.compute_kernel(
                    str(KERNELS / name),
                    cores,
                    [],
                    [(core, [1])],
                    defines=(("INP_FLOAT32", "1"),),
                    fp32_dest=True,
                    unpack_to_dest_fp32=(0,),
                )
                return fp.program_descriptor(
                    [reader, writer, compute], [fp.cb_descriptor(i, ttnn.float32, 4096, 2, cores) for i in (0, 1)]
                )

            fixed = descriptor("softplus.cpp" if softplus else "reciprocal.cpp")
            negative = None
            if seed == 20260930 and mode != "reciprocal_source_words":
                fp.run_program(
                    [source, output], descriptor("softplus-control.cpp" if softplus else "reciprocal-control.cpp")
                )
                values = [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(output)]
                assert all(torch.equal(values[0], v) for v in values[1:])
                negative = {
                    "changed_words": int((values[0] != wanted).sum()),
                    "first_bits": int(values[0].view(torch.int32).flatten()[0]),
                }
                assert negative["changed_words"] > 0, "Unfixed control must fail exact source regression"

            def execute():
                fp.run_program([source, output], fixed)

            def compare():
                values = [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(output)]
                assert len(values) == 4 and all(torch.equal(values[0], v) for v in values[1:])
                actual = values[0]
                assert torch.isfinite(actual).all() and torch.equal(actual, wanted)
                reference = torch.nn.functional.softplus(value.double()) if softplus else value.double().reciprocal()
                if softplus:
                    torch.testing.assert_close(actual.double(), reference, rtol=1e-3, atol=1e-5)
                elif mode == "reciprocal_norm_scaling":
                    torch.testing.assert_close(actual.double(), reference, rtol=1e-4, atol=0)
                # Wide inverse's existing 1e-4 failure is measured, never reported as a pass.
                relative = (actual.double() - reference).abs() / reference.abs()
                return actual, float(relative.max()), int((relative > 1e-4).sum())

            execute()
            expected, maximum, wide_failed_words = compare()
            count = mesh_device.num_program_cache_entries()
            if cached is not None:
                assert count == cached
            cached = count
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            try:
                execute()
            finally:
                ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            traces.append(tid)
            TraceAllocationTracker.acknowledge_corruptible(output)
            for _ in range(3):
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                assert torch.equal(compare()[0], expected)
            assert mesh_device.num_program_cache_entries() == cached
            ttnn.release_trace(mesh_device, tid)
            traces.remove(tid)
            results.append(
                {
                    "seed": seed,
                    "source_words_exact": True,
                    "replicas": 4,
                    "elements": expected.numel(),
                    "first_bits": int(expected.view(torch.int32).flatten()[0]),
                    "negative_control": negative,
                    "max_independent_relative_error": maximum,
                    "inverse_wide_1e4_failed_words": wide_failed_words if mode == "reciprocal_source_words" else None,
                    "trace_replays_exact": 3,
                    "program_cache_entries": cached,
                    "output_sha256": sha(expected),
                    "input_address": source.buffer_address(),
                }
            )
    finally:
        (out / (mode + "-report.json")).write_text(
            json.dumps(
                {
                    "mode": mode,
                    "passed": len(results) == 2,
                    "results": results,
                    "scope": "Exact source words, original real-norm 1e-4 gate, softplus independent1e-3/1e-5, trace3 and freshaddresses. Earlier wide inverse1e-4 remains failed.",
                },
                indent=2,
            )
            + "\n"
        )
        for tid in traces:
            ttnn.release_trace(mesh_device, tid)
        for tensor in reversed(owned):
            ttnn.deallocate(tensor)
