# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU simulator screen of register-resident FP32 GDN against the current kernel."""

import argparse
import hashlib
import os
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import save


def run(args):
    if args.output.exists():
        raise ValueError("Preserve previous simulator attempts")
    simulator = Path(os.environ["TT_METAL_SIMULATOR"])
    if hashlib.sha256(simulator.read_bytes()).hexdigest() != (
        "d01be2a094f9f0f6a00e771defeae2311cb82f051b8af89c2df509e78b8432b7"
    ):
        raise ValueError("Require the task-owned pinned CPU simulator")
    assert os.environ["TT_METAL_SLOW_DISPATCH_MODE"] == "1"
    assert os.environ["TT_METAL_DISABLE_SFPLOADMACRO"] == "1"
    report = dict(
        state="importing",
        passed=False,
        cleanup_completed=False,
        physical_devices_accessed=False,
        hardware_timing_claim=False,
        promoted_to_model=False,
        candidate_first=args.candidate_first,
        scope="FP32 recurrence only; simulator arithmetic/rebinding check, not a hardware or long-horizon pass",
        compiler_fallback="Explicit instructions replace SFPLOADMACRO; hardware instruction parity unqualified",
        checks=[],
    )
    save(args.output, report)
    import torch

    import ttnn
    from models.demos.qwen38_27b_qb2.tests.test_gdn_step_candidate import accuracy, reference, stimulus
    from models.demos.qwen38_27b_qb2.tt.gdn_step.op import step

    torch.set_num_threads(1)
    assert ttnn.GetNumAvailableDevices() == 1, "Require one virtual chip"
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

        def upload(value, *, tiled=False):
            return ttnn.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT if tiled else ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def download(value):
            ranks = ttnn.get_device_tensors(value)
            assert len(ranks) == 1
            return ttnn.to_torch(ranks[0]).float()

        allocations = []
        # Three value heads share one Q/K head, matching the deployed TP4 ratio.
        # Both independent sets stay live throughout program-cache rebinding.
        for index in range(2):
            q, k, v, gates, state = stimulus(3, 861000 + index)
            q, k = q[:1], k[:1]
            if index:
                v = torch.einsum("hk,hkv->hv", k.expand(3, -1), state * gates[:, 0, None, None]) + 1e-6 * v
            inputs = [upload(value) for value in (q, k, v, gates)]
            sessions = [(upload(state, tiled=True), upload(torch.full((3, 128), float("nan")))) for _ in range(2)]
            allocations.append(dict(inputs=inputs, sessions=sessions, host=(q, k, v, gates), expected=state.clone()))
        addresses = [
            [t.buffer_address() for t in a["inputs"] + [t for s in a["sessions"] for t in s]] for a in allocations
        ]
        for iteration, index in enumerate((0, 1, 0, 1)):
            report.update(state="running", active_allocation=index, active_iteration=iteration)
            save(args.output, report)
            allocation = allocations[index]
            q, k, v, gates = allocation["host"]
            expected, expected_output = reference(allocation["expected"], q.expand(3, -1), k.expand(3, -1), v, gates)
            allocation["expected"] = expected
            results = {}
            for resident in (True, False) if args.candidate_first else (False, True):
                state, output = allocation["sessions"][int(resident)]
                report["active_resident"] = resident
                save(args.output, report)
                step(
                    *allocation["inputs"],
                    state,
                    output,
                    value_splits=4,
                    input_buffer_items=2,
                    qk_head_repeat=3,
                    resident_state=resident,
                )
                ttnn.synchronize_device(mesh)
                results[resident] = (download(state), download(output))
            checks = dict(
                allocation=index,
                iteration=iteration,
                state_bit_identical=torch.equal(results[False][0], results[True][0]),
                output_bit_identical=torch.equal(results[False][1], results[True][1]),
                state_reference=accuracy(results[True][0], expected),
                output_reference=accuracy(results[True][1], expected_output),
                inputs_unchanged=all(
                    torch.equal(download(tensor), host)
                    for tensor, host in zip(allocation["inputs"], allocation["host"])
                ),
            )
            report["checks"].append(checks)
            save(args.output, report)
            assert all(
                checks[key] for key in ("state_bit_identical", "output_bit_identical", "inputs_unchanged")
            ), checks
            assert checks["state_reference"]["passed"] and checks["output_reference"]["passed"], checks
            print("GDN_RESIDENT_SIM_CHECK", index, iteration, "PASS", flush=True)
        assert addresses == [
            [t.buffer_address() for t in a["inputs"] + [t for s in a["sessions"] for t in s]] for a in allocations
        ]
        report.update(state="completed", passed=True, persistent_addresses_unchanged=True)
    except BaseException as error:
        report.update(state="failed", error=type(error).__name__, detail=str(error)[:2000])
        raise
    finally:
        try:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)
                report["cleanup_completed"] = True
        finally:
            save(args.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-first", action="store_true", help="Attempt candidate compilation before control")
    run(parser.parse_args())
