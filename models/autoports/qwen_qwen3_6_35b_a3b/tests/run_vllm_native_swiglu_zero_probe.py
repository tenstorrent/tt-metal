# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native fused SwiGLU -> native zero_pages transition, not full-model acceptance.

Run under a finite external GNU timeout after all other device owners exit.
No performance claim: checkpoints read and synchronize all four chips.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
REFERENCE = "tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_moe_fused_swiglu.py"
NATIVE = "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu/device"


def timeout_ancestor():
    """Require an actual finite timeout wrapper, not a caller assertion."""
    pid = os.getppid()
    while pid > 1:
        proc = Path(f"/proc/{pid}")
        argv = [part.decode() for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
        if argv and Path(argv[0]).name == "timeout":
            index = 1
            while index < len(argv) and argv[index].startswith("-"):
                if argv[index] == "--":
                    index += 1
                    break
                index += 2 if argv[index] in ("-s", "--signal", "-k", "--kill-after") else 1
            duration = argv[index] if index < len(argv) else ""
            if re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)[smhd]?", duration) and float(duration.rstrip("smhd")) > 0:
                return dict(pid=pid, duration=duration, argv=argv)
            raise RuntimeError("GNU timeout ancestor has no recognized finite positive duration")
        status = (proc / "status").read_text().splitlines()
        pid = int(next(line.split()[1] for line in status if line.startswith("PPid:")))
    raise RuntimeError("Launch this hardware probe under a finite external GNU timeout")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000, help="Replays per count fixture")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--capacity", type=int, default=2240, help="Dispatch rows; model decode capacity is 2240")
    parser.add_argument("--active-expert", type=int, default=0, choices=range(64))
    parser.add_argument("--trace-region-size", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 1_000_000 or not 1 <= args.progress_every <= args.iterations:
        parser.error("Require 1 <= progress-every <= iterations <= 1000000")
    if not 32 <= args.capacity <= 2240 or args.capacity % 32 or args.trace_region_size <= 0:
        parser.error("Capacity must be a multiple of 32 in [32,2240]; trace region must be positive")
    return args


def main():
    args = parse_args()
    timeout = timeout_ancestor()
    if args.output.exists():
        raise ValueError("Choose a fresh output artifact")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    if os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] != "0":
        raise ValueError("Program-cache allocations must be included in trace tracking")

    # All imports that could initialize the device runtime are after validation.
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_zeroing_probe import (
        NATIVE_HELPER,
        PARTITIONS,
        SHAPE,
        make_program,
    )
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.precision import device_compute, device_dtype, resolve_precision

    precision = resolve_precision()
    if precision["weight_groups"]["expert_ep"] != "bfp8":
        raise ValueError("This discriminator requires the selected BFP8 expert policy")
    result = dict(
        status="RUNNING",
        experiment="native moe_fused_swiglu immediately followed by native combine zero_pages",
        command=sys.argv,
        external_timeout=timeout,
        mesh=[1, 4],
        core_grid=[8, 8],
        capacity=args.capacity,
        embedding=2048,
        expert_hidden=512,
        local_experts=64,
        global_counts_length=256,
        active_expert=args.active_expert,
        input_m_tiles=1,
        precision_config=precision,
        tracking=os.environ["TT_METAL_TRACE_ALLOC_TRACKING"],
        skip_program_cache=False,
        activation=dict(dtype="BFLOAT16", layout="ROW_MAJOR", memory="DRAM_INTERLEAVED"),
        expert_output=dict(dtype="BFLOAT16", layout="TILE", memory="DRAM_INTERLEAVED"),
        zero_output=dict(shape=list(SHAPE), dtype="BFLOAT16", layout="ROW_MAJOR", partitions=PARTITIONS),
        weight_storage="Three distinct shared tensors; each of 64 read-only expert entries reuses those tensors",
        counts_fixtures={
            "all_zero": [0] * 256,
            "one_nonempty": [32 if i == args.active_expert else 0 for i in range(256)],
        },
        iterations_per_fixture=args.iterations,
        completed={"all_zero": 0, "one_nonempty": 0},
        captures=0,
        releases=0,
        replays=0,
        checkpoints=[],
        sources={
            path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            for path in (
                REFERENCE,
                NATIVE_HELPER,
                f"{NATIVE}/kernels/moe_fused_swiglu_reader.cpp",
                f"{NATIVE}/kernels/moe_fused_swiglu_writer.cpp",
                "models/autoports/qwen_qwen3_6_35b_a3b/tests/run_vllm_zeroing_probe.py",
            )
        },
        limitations=[
            "No dispatch, native combine routing/semaphores, model, sampling or serving cache is exercised.",
            "Expert IDs/counts/weights are replicated; this is not expert-sharded model traffic.",
            "Sparse diagonal weights are synthetic, with a small analytic numerical oracle; only one expert is nonempty.",
            "Checkpoint poisoning/readbacks add NoC work and synchronization; this is not a performance measurement.",
            "Only the last replay of each checkpoint batch is observed; intermediate output is not independently checked.",
            "The actual native fused kernel and zero helper execute consecutively inside one retained CQ0 trace.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh = trace = None
    capture_open = False
    owned = []
    try:
        torch.set_num_threads(4)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)

        def upload(host, dtype, layout):
            tensor = ttnn.from_torch(
                host.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            owned.append(tensor)
            return tensor

        def bf16(host, layout=ttnn.ROW_MAJOR_LAYOUT):
            return upload(host, ttnn.bfloat16, layout)

        def uint32(values):
            return upload(torch.tensor(values, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

        host_x = torch.zeros(args.capacity, 2048, dtype=torch.bfloat16)
        host_x[:32] = ((torch.arange(32 * 2048).reshape(32, 2048) % 17) - 8).to(torch.bfloat16) / 4
        x = bf16(host_x)
        gate_host = torch.zeros(2048, 512)
        gate_host[torch.arange(512), torch.arange(512)] = 0.125
        down_host = gate_host.T.contiguous()
        weight_dtype = device_dtype(precision["weight_groups"]["expert_ep"])
        gate = upload(gate_host, weight_dtype, ttnn.TILE_LAYOUT)
        up = upload(gate_host, weight_dtype, ttnn.TILE_LAYOUT)
        down = upload(down_host, weight_dtype, ttnn.TILE_LAYOUT)
        weights = ([gate] * 64, [up] * 64, [down] * 64)
        ids, offsets = uint32(list(range(64))), uint32([0] * 256)
        fixtures = {name: uint32(values) for name, values in result["counts_fixtures"].items()}
        counts = uint32([0] * 256)
        expert_poison = bf16(torch.full((args.capacity, 2048), 7.0), ttnn.TILE_LAYOUT)
        expert_output = bf16(torch.full((args.capacity, 2048), 7.0), ttnn.TILE_LAYOUT)
        zero_poison, zero_output = bf16(torch.ones(SHAPE)), bf16(torch.ones(SHAPE))
        program = make_program(ttnn, zero_output)
        compute = device_compute(mesh, precision["compute_fidelities"]["expert_ep"])
        z = host_x[:32, :512].float() * 0.125
        golden = torch.zeros(32, 2048)
        golden[:, :512] = torch.nn.functional.silu(z) * z * 0.125
        references = {}

        def execute():
            ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                x,
                *weights,
                counts,
                ids,
                input_m_tiles=1,
                compute_kernel_config=compute,
                core_grid=ttnn.CoreCoord(8, 8),
                output=expert_output,
                expert_region_offsets=offsets,
                read_x_at_offset=True,
                stochastic_rounding=False,
            )
            ttnn.generic_op([expert_output, zero_output], program)

        def check_fixture(name, *, establish=False):
            expert_chips = ttnn.get_device_tensors(expert_output)
            zero_chips = ttnn.get_device_tensors(zero_output)
            assert len(expert_chips) == len(zero_chips) == 4
            max_error = 0.0
            for chip, (expert, zero) in enumerate(zip(expert_chips, zero_chips)):
                actual = ttnn.to_torch(expert).reshape(args.capacity, 2048).float()
                zeros = ttnn.to_torch(zero)
                assert torch.count_nonzero(zeros).item() == 0, (name, chip, "zero-helper output")
                assert torch.isfinite(actual).all(), (name, chip, "nonfinite expert output")
                if name == "all_zero":
                    assert torch.equal(actual, torch.full_like(actual, 7)), (name, chip, "skip wrote output")
                else:
                    error = float((actual[:32] - golden).abs().max())
                    max_error = max(max_error, error)
                    torch.testing.assert_close(actual[:32], golden, rtol=0.1, atol=0.001)
                    assert torch.count_nonzero(actual[:32]).item() > 0, "nonempty fixture executed no useful output"
                    assert torch.equal(actual[32:], torch.full_like(actual[32:], 7)), "writes beyond active rows"
                    if establish:
                        references[chip] = actual[:32].clone()
                    else:
                        assert torch.equal(actual[:32], references[chip]), (name, chip, "replay differs from eager")
            return dict(exact_zero_chips=4, expert_checked_chips=4, maximum_analytic_absolute_error=max_error)

        def prime(name):
            ttnn.copy(fixtures[name], counts)
            ttnn.copy(expert_poison, expert_output)
            ttnn.copy(zero_poison, zero_output)

        # All buffer allocations, both device-side branches, and observation
        # programs are warmed before the single retained capture.
        for name in fixtures:
            prime(name)
            execute()
            result.setdefault("eager_checks", {})[name] = check_fixture(name, establish=True)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        capture_open = True
        execute()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        capture_open = False
        result.update(captures=1, trace_id=str(trace))
        entries = int(mesh.num_program_cache_entries())
        result["prepared_program_entries"] = entries
        result["buffer_addresses"] = dict(
            activations=x.buffer_address(),
            counts=counts.buffer_address(),
            expert_output=expert_output.buffer_address(),
            zero_output=zero_output.buffer_address(),
            weights=[gate.buffer_address(), up.buffer_address(), down.buffer_address()],
        )
        save()
        for first in range(0, args.iterations, args.progress_every):
            last = min(first + args.progress_every, args.iterations)
            for name in fixtures:
                prime(name)
                # Prove each checkpoint starts with nonzero helper destination.
                for chip in ttnn.get_device_tensors(zero_output):
                    poison = ttnn.to_torch(chip)
                    assert torch.equal(poison, torch.ones_like(poison)), "poison upload failed"
                result["pending"] = dict(fixture=name, start=first + 1, end=last)
                save()
                print("NATIVE_SWIGLU_ZERO_BATCH", name, first + 1, last, flush=True)
                for _ in range(first, last):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    result["replays"] += 1
                checks = check_fixture(name)
                assert int(mesh.num_program_cache_entries()) == entries, "program-cache growth after capture"
                result["completed"][name] = last
                result["checkpoints"].append(dict(fixture=name, iteration=last, program_entries=entries, **checks))
                result.pop("pending")
                save()
                print("NATIVE_SWIGLU_ZERO_PASS", name, last, flush=True)
        result["status"] = "PASS"
    except BaseException:
        result.update(status="FAIL", error=traceback.format_exc())
        save()
        raise
    finally:
        try:
            if mesh is not None:
                if capture_open:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                    result["releases"] += 1
                for tensor in reversed(owned):
                    ttnn.deallocate(tensor)
                ttnn.close_mesh_device(mesh)
                result["mesh_closed"] = True
        except BaseException:
            result.update(status="FAIL", cleanup_error=traceback.format_exc())
            raise
        finally:
            save()
    print(json.dumps(dict(status=result["status"], completed=result["completed"], output=str(args.output))))


if __name__ == "__main__":
    main()
