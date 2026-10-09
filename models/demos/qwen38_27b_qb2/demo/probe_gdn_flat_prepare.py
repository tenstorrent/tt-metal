# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Simulator screen of direct GDN preparation against the existing adapter ops."""

import argparse
import hashlib
import os
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save


def run(args):
    assert not args.output.exists(), "Preserve prior experiment receipts"
    simulator = Path(os.environ["TT_METAL_SIMULATOR"])
    assert (
        hashlib.sha256(simulator.read_bytes()).hexdigest()
        == "d01be2a094f9f0f6a00e771defeae2311cb82f051b8af89c2df509e78b8432b7"
    )
    assert os.environ["TT_METAL_SLOW_DISPATCH_MODE"] == "1"
    assert os.environ["TT_METAL_DISABLE_SFPLOADMACRO"] == "1"
    report = dict(
        state="importing",
        passed=False,
        cleanup_completed=False,
        cases=[],
        physical_devices_accessed=False,
        hardware_timing_claim=False,
        promoted_to_model=False,
        scope="Direct tiled-input packing plus existing FP32 normalization; native exp stays external",
    )
    save(args.output, report)
    import torch

    import ttnn
    from models.demos.qwen38_27b_qb2.tt.gdn_step.flat_prepare import prepare
    from models.demos.qwen38_27b_qb2.tt.gdn_step.shared_qk import prepare as normalize

    torch.set_num_threads(1)
    assert ttnn.GetNumAvailableDevices() == 1, "Require one virtual chip"
    mesh = None

    def sha(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def download(tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()

    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

        def upload(value, dtype, memory, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout,
                memory_config=memory,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        for batch, time_rows, placement in ((1, 32, "dram"), (16, 1, "dram"), (32, 32, "l1")):
            memory = ttnn.DRAM_MEMORY_CONFIG if placement == "dram" else ttnn.L1_MEMORY_CONFIG
            report.update(state="running", active_case=[batch, time_rows, placement])
            save(args.output, report)
            operands, expected, keepalive = [], [], []
            for allocation in range(2):
                rng = torch.Generator().manual_seed(610090 + batch * 100 + allocation)
                hosts = [torch.randn(batch, time_rows, width, generator=rng).bfloat16() for width in (512, 512, 1536)]
                for value in hosts[:2]:
                    value[0, 0, :128] = 0
                    value[0, 0, 128:256] *= 1e-5
                inputs = [upload(value, ttnn.bfloat16, memory) for value in hosts]
                # Actual exp stays unchanged; feed its exact result to both paths.
                log_decay = upload(-torch.rand(batch, 32, 12, generator=rng), ttnn.float32, memory)
                decay = ttnn.exp(log_decay[:, :1, :])
                beta = upload(torch.rand(batch, 32, 12, generator=rng).bfloat16(), ttnn.bfloat16, memory)
                inputs += [decay, beta]
                shapes = [(batch * 4, 128)] * 2 + [(batch * 12, 128), (batch * 12, 8)]
                outputs = [
                    upload(
                        torch.full(shape, float("nan")), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
                    )
                    for shape in shapes
                ]

                def native_vector(tensor, count):
                    row = ttnn.to_layout(tensor[:, :1, :], ttnn.ROW_MAJOR_LAYOUT)
                    row = ttnn.reshape(row, [batch, count, 128])
                    value = ttnn.typecast(ttnn.to_layout(row, ttnn.TILE_LAYOUT), ttnn.float32)
                    value = ttnn.reshape(ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT), [batch * count, 128])
                    return ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)

                raw_q, raw_k, raw_v = [native_vector(value, count) for value, count in zip(inputs[:3], (4, 4, 12))]
                packed = ttnn.concat([decay, ttnn.typecast(beta[:, :1, :], ttnn.float32)], dim=1)
                packed = ttnn.to_layout(ttnn.permute(packed, [0, 2, 1]), ttnn.ROW_MAJOR_LAYOUT)
                packed = ttnn.pad(ttnn.reshape(packed, [batch * 12, 2]), [(0, 0), (0, 6)], 0.0)
                packed = ttnn.to_memory_config(packed, ttnn.DRAM_MEMORY_CONFIG)
                nq, nk = [
                    upload(
                        torch.full(shapes[0], float("nan")),
                        ttnn.float32,
                        ttnn.DRAM_MEMORY_CONFIG,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                    for _ in range(2)
                ]
                normalize(raw_q, raw_k, nq, nk)
                reference = [download(tensor) for tensor in (nq, nk, raw_v, packed)]
                assert torch.equal(reference[2], hosts[2][:, 0].reshape(batch * 12, 128).float())
                assert torch.count_nonzero(reference[3][:, 2:]).item() == 0
                expected.append(reference)
                operands.append((*inputs, *outputs))
                keepalive.extend((log_decay, raw_q, raw_k, raw_v, packed, nq, nk))
            for allocation in (0, 1, 0):
                tensors = operands[allocation]
                before = [sha(download(tensor)) for tensor in tensors[:5]]
                addresses = [tensor.buffer_address() for tensor in tensors]
                prepare(*tensors)
                actual = [download(tensor) for tensor in tensors[5:]]
                row = dict(
                    batch=batch,
                    time_rows=time_rows,
                    placement=placement,
                    allocation=allocation,
                    bit_identical=[
                        torch.equal(value, reference) for value, reference in zip(actual, expected[allocation])
                    ],
                    finite=all(torch.isfinite(value).all().item() for value in actual),
                    max_abs_errors=[
                        float((value - reference).abs().max()) for value, reference in zip(actual, expected[allocation])
                    ],
                    output_sha256=[sha(value) for value in actual],
                    reference_sha256=[sha(value) for value in expected[allocation]],
                    inputs_unchanged=before == [sha(download(tensor)) for tensor in tensors[:5]],
                    addresses_unchanged=addresses == [tensor.buffer_address() for tensor in tensors],
                )
                row["passed"] = (
                    all(row["bit_identical"])
                    and row["finite"]
                    and row["inputs_unchanged"]
                    and row["addresses_unchanged"]
                )
                report["cases"].append(row)
                save(args.output, report)
                assert row["passed"], row
            del operands, expected, keepalive, inputs, outputs, tensors
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", passed=False, error=type(error).__name__, detail=str(error)[:3000])
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            report["cleanup_completed"] = True
        save(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
