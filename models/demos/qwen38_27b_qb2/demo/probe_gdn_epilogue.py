# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU simulator screen of a fused GDN epilogue against actual native operations."""

import argparse
import hashlib
import os
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save


def run(args):
    if args.output.exists():
        raise ValueError("Preserve prior simulator receipts")
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
        physical_devices_accessed=False,
        hardware_timing_claim=False,
        promoted_to_model=False,
        cases=[],
        scope="Synthetic single-token epilogue vs actual native norm plus multiply; not full-model or hardware qualification",
        compiler_fallback="Simulator uses explicit instructions instead of SFPLOADMACRO; hardware instruction parity unqualified",
    )
    save(args.output, report)
    import torch

    import ttnn
    from models.demos.qwen38_27b_qb2.tt.gdn_epilogue.op import epilogue

    def sha(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def metrics(value, reference):
        value, reference = value.reshape(-1, 128).double(), reference.reshape(-1, 128).double()
        centered = [v - v.mean(-1, keepdim=True) for v in (value, reference)]
        norms = [v.norm(dim=-1) for v in centered]
        nonzero = norms[1] > 1e-10
        pcc = (centered[0] * centered[1]).sum(-1) / (norms[0] * norms[1]).clamp_min(1e-30)
        relative = (value - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-10)
        return dict(
            min_pcc=float(pcc[nonzero].min()),
            max_relative_rms=float(relative.max()),
            max_abs_error=float((value - reference).abs().max()),
            bit_identical=torch.equal(value, reference),
        )

    torch.set_num_threads(1)
    assert ttnn.GetNumAvailableDevices() == 1, "Require exactly one virtual chip"
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

        def upload(value, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def download(value):
            return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).float()

        for batch in (1, 16, 32):
            operands = []
            expected = []
            expected_normalized = []
            reference_storage = []
            for allocation in range(2):
                rng = torch.Generator().manual_seed(20261009 + batch * 100 + allocation)
                raw_host = torch.randn(batch * 12, 128, generator=rng)
                raw_host[0] = 0
                raw_host[1] *= 1e-5
                gate_host = torch.randn(batch, 1, 12 * 128, generator=rng).bfloat16()
                gate_host.reshape(-1)[:8] = torch.tensor([-30, -10, -1, 0, 1, 10, 30, 0.0001]).bfloat16()
                weight_host = torch.randn(128, generator=rng).bfloat16()
                raw = upload(raw_host, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)
                gate = upload(gate_host, ttnn.bfloat16)
                weight = upload(weight_host, ttnn.bfloat16)
                output = upload(torch.full_like(gate_host, float("nan")), ttnn.bfloat16)
                # Reconstruct exactly the existing adapter and epilogue boundary.
                head_major = ttnn.to_layout(ttnn.reshape(raw, [batch * 12, 1, 128]), ttnn.TILE_LAYOUT)
                head_major = ttnn.reshape(head_major, [batch * 12, 32, 128], head_major.padded_shape)
                padded_gate = ttnn.pad(gate, [(0, 0), (0, 31), (0, 0)], 0.0)
                normalized = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                    head_major,
                    padded_gate,
                    weight,
                    12,
                    epsilon=1e-6,
                    output_dtype=ttnn.bfloat16,
                )
                native = ttnn.mul(normalized, padded_gate)
                expected.append(download(native)[:, :1, :])
                expected_normalized.append(download(normalized)[:, :1, :])
                # Padding may return a view of an already-padded tiled buffer.
                # Keep reference intermediates alive; explicit deallocation of
                # an alias can invalidate the candidate's original gate input.
                report.setdefault("reference_buffers", []).append(
                    dict(
                        batch=batch,
                        allocation=allocation,
                        padded_gate_aliases_gate=padded_gate.buffer_address() == gate.buffer_address(),
                    )
                )
                reference_storage.extend((head_major, padded_gate, normalized, native))
                operands.append((raw, gate, weight, output))
            # Alternate live allocations and return to the first. This exercises
            # program-cache argument rebinding, multiple worker waves and reuse.
            for allocation in (0, 1, 0):
                report.update(state="running", active_case=[batch, allocation])
                save(args.output, report)
                tensors = operands[allocation]
                before = [sha(download(tensor)) for tensor in tensors[:3]]
                epilogue(*tensors, multiply_z=False)
                normalized_metrics = metrics(download(tensors[-1]), expected_normalized[allocation])
                epilogue(*tensors)
                value = download(tensors[-1])
                assert torch.isfinite(value).all(), "Output still contains poison or nonfinite arithmetic"
                measured = metrics(value, expected[allocation])
                measured["normalized_before_z"] = normalized_metrics
                measured.update(
                    batch=batch,
                    allocation=allocation,
                    input_hashes=before,
                    output_sha256=sha(value),
                    native_sha256=sha(expected[allocation]),
                )
                measured["inputs_unchanged"] = before == [sha(download(tensor)) for tensor in tensors[:3]]
                padded = ttnn.reshape(tensors[-1], [batch, 32, 12 * 128], tensors[-1].padded_shape)
                measured["padding_zero"] = torch.count_nonzero(download(padded)[:, 1:]).item() == 0
                measured["passed"] = (
                    measured["min_pcc"] >= 0.99999
                    and measured["max_relative_rms"] <= 0.001
                    and measured["inputs_unchanged"]
                    and measured["padding_zero"]
                )
                report["cases"].append(measured)
                save(args.output, report)
                assert measured["passed"], measured
            for tensors in operands:
                for tensor in tensors:
                    ttnn.deallocate(tensor)
            del reference_storage
        report.update(state="completed", passed=True)
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:2000])
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            report["cleanup_completed"] = True
        save(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())
