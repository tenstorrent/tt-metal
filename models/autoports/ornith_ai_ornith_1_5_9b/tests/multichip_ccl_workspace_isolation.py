# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate TP4 collective workspace ownership with fixed inputs and buffers.

Run each operation/order/cache combination in a fresh process. Persistent
buffers also disable the CCL startup barrier in the current kernels, so compare
fixed and automatic buffers before attributing their difference to allocation.
This is an eager correctness diagnostic, not a performance measurement.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import pcc
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipCollectives


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("rs", "ag"), required=True)
    parser.add_argument("--order", choices=("ABA", "BAB"), default="ABA")
    parser.add_argument("--cache", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--buffers", choices=("fixed", "automatic"), default="fixed")
    parser.add_argument("--bank", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ping-pong", action="store_true", help="Cycle each manager's banks instead of holding --bank")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--memory", choices=("l1", "dram"), default="l1")
    parser.add_argument(
        "--vary-input", action="store_true", help="Replace input values in place before each later call"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    if args.rows < 1 or args.rows % 32:
        parser.error("--rows must be a positive multiple of 32")
    return args


def tensor_addresses(value):
    return [int(shard.buffer_address()) for shard in ttnn.get_device_tensors(value)]


def semaphore_address(value):
    address = ttnn.get_global_semaphore_address(value)
    return list(map(int, address)) if isinstance(address, (tuple, list)) else int(address)


def main():
    args = parse_args()
    torch.set_num_threads(8)
    generator = torch.Generator().manual_seed(927)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        arguments=vars(args),
        command=[sys.executable, *sys.argv],
        diagnostic="rank-distinct BF16 input; independent RS or AG; eager synchronized calls",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        decoder_sha256=hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "tt/multichip_decoder.py").read_bytes()
        ).hexdigest(),
        devices=[0, 1, 2, 3],
        mesh=[1, 4],
        records=[],
        completed=False,
    )

    def save():
        path.write_text(json.dumps(result, indent=2) + "\n")

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        # Use the production ownership/allocation contract, including both AR
        # workspaces even though this experiment exercises RS and AG alone.
        owners = {name: MultichipCollectives(mesh) for name in "AB"}
        managers = {name: workspace.ccl for name, workspace in owners.items()}
        memory = ttnn.L1_MEMORY_CONFIG if args.memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
        width = 4096 if args.operation == "rs" else 1024
        # Multiples of 1/4 with a small range sum exactly in BF16 for TP4. This
        # separates corruption from tolerance/rounding differences in RS.
        host_shards = [
            (torch.randint(-8, 9, (1, 1, args.rows, width), generator=generator).float() / 4).bfloat16()
            for _ in range(4)
        ]
        shard_dim = 0 if args.operation == "rs" else 3
        host_input = torch.cat(host_shards, dim=shard_dim)
        inp = ttnn.from_torch(
            host_input,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=shard_dim),
        )
        if args.operation == "rs":
            total = torch.stack([value.float() for value in host_shards]).sum(dim=0).bfloat16()
            expected = list(torch.chunk(total, 4, dim=3))
        else:
            expected = [torch.cat(host_shards, dim=3)] * 4

        def zeros(shape):
            return ttnn.zeros(shape, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=memory)

        persistent = None
        if args.buffers == "fixed":
            persistent = (
                [zeros((2, 1, args.rows, 4096)), zeros((1, 1, args.rows, 1024))]
                if args.operation == "rs"
                else zeros((1, 1, args.rows, 4096))
            )
        ttnn.synchronize_device(mesh)
        result["input_addresses"] = tensor_addresses(inp)
        result["persistent_addresses"] = (
            [tensor_addresses(value) for value in persistent]
            if isinstance(persistent, list)
            else tensor_addresses(persistent) if persistent is not None else None
        )
        result["startup_barrier_enabled"] = args.buffers == "automatic"
        result["owner_allocations"] = {
            name: dict(
                rs=[semaphore_address(value) for value in manager.rs_ping_pong_semaphores],
                ag=[semaphore_address(value) for value in manager.ag_ping_pong_semaphores],
                barriers=[semaphore_address(value) for value in manager.barrier_semaphore],
                unused_ar_buffers=[tensor_addresses(value) for value in owners[name].ar_buffers],
            )
            for name, manager in managers.items()
        }
        # Remove setup-op history. Enabled calls then reuse their first CCL
        # entry; disabled calls rebuild with all user tensors still alive.
        mesh.disable_and_clear_program_cache()
        if args.cache == "enabled":
            mesh.enable_program_cache()
        save()
        first_outputs = None
        for index, name in enumerate(args.order * args.cycles):
            if args.vary_input and index:
                host_shards = [
                    (torch.randint(-8, 9, (1, 1, args.rows, width), generator=generator).float() / 4).bfloat16()
                    for _ in range(4)
                ]
                host_tensor = ttnn.from_torch(
                    torch.cat(host_shards, dim=shard_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=shard_dim),
                )
                ttnn.copy_host_to_device_tensor(host_tensor, inp)
                # Finish the mesh-wide upload before remote consumers can run;
                # this synchronization does not initialize unallocated semaphores.
                ttnn.synchronize_device(mesh)
                if args.operation == "rs":
                    total = torch.stack([value.float() for value in host_shards]).sum(dim=0).bfloat16()
                    expected = list(torch.chunk(total, 4, dim=3))
                else:
                    expected = [torch.cat(host_shards, dim=3)] * 4
                assert tensor_addresses(inp) == result["input_addresses"], "Input allocation changed"
            manager = managers[name]
            if args.ping_pong:
                bank = manager.rs_ping_pong_idx if args.operation == "rs" else manager.ag_ping_pong_idx
                semaphores = (
                    manager.get_rs_ping_pong_semaphore()
                    if args.operation == "rs"
                    else manager.get_ag_ping_pong_semaphore()
                )
                barrier = manager.get_barrier_semaphore()
            else:
                bank = args.bank
                count = 3 if args.operation == "rs" else 2
                pool = manager.rs_ping_pong_semaphores if args.operation == "rs" else manager.ag_ping_pong_semaphores
                semaphores = pool[bank * count : (bank + 1) * count]
                barrier = manager.barrier_semaphore[bank]
            row = dict(
                index=index,
                owner=name,
                bank=bank,
                semaphores=[semaphore_address(value) for value in semaphores],
                barrier=semaphore_address(barrier),
                cache_entries_before=mesh.num_program_cache_entries(),
                input_generation=index if args.vary_input else 0,
                input_addresses=tensor_addresses(inp),
            )
            result["records"].append(row)
            save()
            print("CCL_CALL", json.dumps(row), flush=True)
            common = dict(
                dim=3,
                cluster_axis=1,
                topology=manager.topology,
                num_links=manager.num_links,
                multi_device_global_semaphore=semaphores,
                barrier_semaphore=barrier,
                memory_config=memory,
            )
            try:
                if args.operation == "rs":
                    out = ttnn.experimental.reduce_scatter_minimal_async(
                        inp, persistent_output_buffers=persistent, **common
                    )
                else:
                    out = ttnn.experimental.all_gather_async(
                        inp, mesh_device=mesh, persistent_output_tensor=persistent, **common
                    )
                ttnn.synchronize_device(mesh)
                actual = [ttnn.to_torch(value).clone() for value in ttnn.get_device_tensors(out)]
                finite = [bool(torch.isfinite(value).all()) for value in actual]
                row.update(
                    output_addresses=tensor_addresses(out),
                    finite=finite,
                    pcc=[
                        pcc(reference, value) if ok else None for reference, value, ok in zip(expected, actual, finite)
                    ],
                    max_error=[
                        float((reference.float() - value.float()).abs().max()) if ok else None
                        for reference, value, ok in zip(expected, actual, finite)
                    ],
                    exact=[torch.equal(reference, value) for reference, value in zip(expected, actual)],
                    same_as_first=(
                        [torch.equal(reference, value) for reference, value in zip(first_outputs, actual)]
                        if first_outputs is not None
                        else None
                    ),
                    cache_entries_after=mesh.num_program_cache_entries(),
                )
                if args.operation == "ag":
                    row["replicas_exact"] = [torch.equal(actual[0], value) for value in actual]
                row["passed"] = all(row["exact"])
                if first_outputs is None:
                    first_outputs = actual
                del out
            except Exception as error:
                row.update(passed=False, error=repr(error))
                save()
                raise
            save()
            print("CCL_RESULT", json.dumps(row), flush=True)
        result["completed"] = True
        result["passed"] = all(row["passed"] for row in result["records"])
        save()
        assert result["passed"], "CCL isolation failed; inspect per-rank exactness and owner transition records"
        print("CCL_WORKSPACE_ISOLATION_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
