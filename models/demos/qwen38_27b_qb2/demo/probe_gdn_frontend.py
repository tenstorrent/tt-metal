# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only simulator screen of compact convolution and its preparation reader."""

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
        scope="Synthetic BF16 convolution taps/activations, exact native arithmetic and FP32 preparation on one virtual chip",
    )
    save(args.output, report)
    import torch

    import ttnn
    from models.demos.qwen38_27b_qb2.tt.decode_conv import make_actual_start
    from models.demos.qwen38_27b_qb2.tt.gdn_frontend.op import WIDTHS, convolution
    from models.demos.qwen38_27b_qb2.tt.gdn_step.flat_prepare import prepare

    torch.set_num_threads(1)
    assert ttnn.GetNumAvailableDevices() == 1, "Require one virtual chip"
    mesh = None

    def sha(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    def download(tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()

    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

        def upload(value, memory, *, fp32=False, row=False):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=ttnn.float32 if fp32 else ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT if row else ttnn.TILE_LAYOUT,
                memory_config=memory,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        actual_start = make_actual_start(mesh)
        for batch, compact, placement in ((16, False, "dram"), (16, True, "dram"), (32, True, "l1")):
            memory = ttnn.DRAM_MEMORY_CONFIG if placement == "dram" else ttnn.L1_MEMORY_CONFIG
            report.update(state="running", active_case=[batch, compact, placement], active_phase="allocate_operands")
            save(args.output, report)
            rng = torch.Generator().manual_seed(101010 + batch)
            taps = [upload(torch.randn(1, 1, sum(WIDTHS), generator=rng).bfloat16(), memory) for _ in range(4)]
            allocations = []
            for index in range(2):
                width = 4160 if compact else sum(WIDTHS)
                host_input = torch.randn(batch, 1, width, generator=rng).bfloat16()
                host_history = torch.randn(batch, 3, sum(WIDTHS), generator=rng).bfloat16()
                qkv = upload(host_input.reshape(1, batch, width) if compact else host_input, memory)
                history = upload(host_history, memory, row=True)
                outputs = [upload(torch.full((1, batch, w), float("nan")).bfloat16(), memory) for w in WIDTHS]
                decay = upload(torch.rand(batch, 1, 12, generator=rng), memory, fp32=True)
                beta = upload(torch.rand(batch, 1, 12, generator=rng).bfloat16(), memory)
                prepared, control_prepared = [
                    [
                        upload(torch.full(shape, float("nan")), ttnn.DRAM_MEMORY_CONFIG, fp32=True, row=True)
                        for shape in ((batch * 4, 128), (batch * 4, 128), (batch * 12, 128), (batch * 12, 8))
                    ]
                    for _ in range(2)
                ]
                allocations.append(
                    dict(
                        qkv=qkv,
                        host_input=host_input,
                        host_history=host_history,
                        history=history,
                        outputs=outputs,
                        decay=decay,
                        beta=beta,
                        prepared=prepared,
                        control_prepared=control_prepared,
                    )
                )
            for index in (0, 1, 0):
                a = allocations[index]
                # Independent native layout path, with the same four-tap order
                # and rounding. The second call on allocation zero sees shifted
                # history and copied new inputs at the same buffer addresses.
                if index == 0 and a.get("used"):
                    for key in ("qkv", "decay", "beta"):
                        ttnn.copy(allocations[1][key], a[key])
                    a["host_input"] = allocations[1]["host_input"]
                immutable = [a["qkv"], a["decay"], a["beta"], *taps]
                before = [sha(download(t)) for t in immutable]
                addresses = [t.buffer_address() for t in (a["history"], *a["outputs"], *a["prepared"])]
                report.update(active_allocation=index, active_phase="compact_convolution")
                save(args.output, report)
                convolution(a["qkv"], a["history"], taps, a["outputs"], compact_input=compact)
                ttnn.synchronize_device(mesh)
                report.update(active_phase="compact_preparation")
                save(args.output, report)
                prepare(*a["outputs"], a["decay"], a["beta"], *a["prepared"], compact_qkv=True)
                ttnn.synchronize_device(mesh)
                report.update(active_phase="native_reference_input_upload")
                save(args.output, report)
                # Keep reference layout plumbing independent of device layout
                # kernels: upload the same convolution windows from the host
                # and select retained output rows after download. This changes
                # neither convolution's arithmetic nor the physical TP4 test.
                joined_host = torch.cat([a["host_history"], a["host_input"][:, :, : sum(WIDTHS)]], dim=1)
                joined = upload(joined_host.reshape(1, batch * 4, sum(WIDTHS)), memory, row=True)
                native_history = upload(a["host_history"][:1], memory, row=True)
                report.update(active_phase="native_convolution")
                save(args.output, report)
                native_outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
                    joined,
                    native_history,
                    *taps,
                    *WIDTHS,
                    program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
                    actual_start=actual_start,
                    predecessor_carry=native_history,
                )
                ttnn.synchronize_device(mesh)
                native_values = [download(t).reshape(batch, 4, w)[:, 3] for t, w in zip(native_outputs, WIDTHS)]
                report.update(active_phase="native_reference_preparation")
                save(args.output, report)
                reference = [upload(value[:, None, :].bfloat16(), memory) for value in native_values]
                prepare(*reference, a["decay"], a["beta"], *a["control_prepared"])
                ttnn.synchronize_device(mesh)
                report.update(active_phase="verify")
                save(args.output, report)
                values = [download(t)[0] for t in a["outputs"]] + [download(t) for t in a["prepared"]]
                expected = native_values + [download(t) for t in a["control_prepared"]]
                a["host_history"] = torch.cat([a["host_history"][:, 1:], a["host_input"][:, :, : sum(WIDTHS)]], dim=1)
                histories = [download(a["history"])]
                row = dict(
                    batch=batch,
                    compact_input=compact,
                    placement=placement,
                    allocation=index,
                    changed_input=a.get("used", False),
                    bit_identical=[torch.equal(v, e) for v, e in zip(values, expected)],
                    max_abs_errors=[float((v - e).abs().max()) for v, e in zip(values, expected)],
                    finite=all(torch.isfinite(v).all().item() for v in values),
                    output_sha256=[sha(v) for v in values],
                    reference_sha256=[sha(v) for v in expected],
                    history_matches_host=all(torch.equal(v, a["host_history"].float()) for v in histories),
                    inputs_unchanged=before == [sha(download(t)) for t in immutable],
                    addresses_unchanged=addresses
                    == [t.buffer_address() for t in (a["history"], *a["outputs"], *a["prepared"])],
                )
                row["passed"] = all(row["bit_identical"]) and all(
                    row[k] for k in ("finite", "history_matches_host", "inputs_unchanged", "addresses_unchanged")
                )
                report["cases"].append(row)
                save(args.output, report)
                assert row["passed"], row
                a["used"] = True
            del allocations, a, immutable, reference, joined, native_history, native_outputs
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
