"""Model-free, one-device corruption check. Exit 1 preserves a fault; exit 2 is a setup/error failure.

Example: python device_read_repro_v86.py --fixture 55041-read-fixture-v86.pt.gz
         --device 11 --iterations 30000 --output /tmp/read-check-unique-name

The output directory must not exist. Only this process's tensors are accessed.
The default uses the stock reduction. Explicit NoC routes additionally need
noc_reduce_v63.py alongside this file, and use the unchanged repository kernels.
"""

import argparse
import faulthandler
import gzip
import hashlib
import json
import time
import traceback
from pathlib import Path

import torch
import ttnn


def digest(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def read_raw(tensor):
    host = ttnn.from_device(tensor)
    return torch.utils.dlpack.from_dlpack(host.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))).clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--device", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--route", choices=["original", "noc0", "noc1"], default="original")
    parser.add_argument("--input-memory", choices=["dram", "l1"], default="dram")
    parser.add_argument(
        "--keep-going", action="store_true", help="Retain every fault and finish the requested iterations."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.iterations > 0
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        stage="starting",
        device=args.device,
        iterations=0,
        requested_iterations=args.iterations,
        route=args.route,
        input_memory=args.input_memory,
        faults=[],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
    )

    def checkpoint(stage):
        report["stage"] = stage
        temp = args.output / "report.tmp"
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(args.output / "report.json")
        print(json.dumps(dict(stage=stage, iterations=report["iterations"], faults=len(report["faults"]))), flush=True)

    mesh = source = output = None
    torch.set_num_threads(4)
    faulthandler.enable()
    faulthandler.dump_traceback_later(300, repeat=True)
    try:
        with gzip.open(args.fixture, "rb") as stream:
            fixture = torch.load(stream, map_location="cpu", weights_only=False)
        values, source_raw, expected = fixture["source_values"], fixture["source_raw"], fixture["expected_raw"]
        assert tuple(values.shape) == (8, 1, 128, 2048)
        assert source_raw.numel() == 2048 * 1088 and expected.numel() == 256 * 1088
        report["fixture_provenance"] = fixture["provenance"]
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            physical_device_ids=[args.device],
            dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL),
            worker_l1_size=1344544,
            trace_region_size=0,
        )
        assert list(mesh.get_device_ids()) == [args.device]
        mesh.enable_program_cache()
        source = ttnn.from_torch(
            values,
            device=mesh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if args.input_memory == "dram" else ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        assert torch.equal(read_raw(source), source_raw), "Packed source upload differs from the fixture"
        report.update(
            input_address=source.buffer_address(), source_sha256=digest(source_raw), expected_sha256=digest(expected)
        )
        programs = {}
        started = time.monotonic()
        for iteration in range(args.iterations):
            if args.route == "original":
                output = ttnn.experimental.fast_reduce_nc(
                    source, dims=[0], output=None, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            else:
                from noc_reduce_v63 import program

                output = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([1, 1, 128, 2048]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
                )
                key = output.buffer_address()
                if key not in programs:
                    programs[key] = program(source, output, mesh, int(args.route[-1]))
                ttnn.generic_op([source, output], programs[key])
            if args.input_memory == "dram":
                assert output.buffer_address() != source.buffer_address()
            actual = read_raw(output)
            report["iterations"] += 1
            if not torch.equal(actual, expected):
                current_source = read_raw(source)
                rereads = [read_raw(output) for _ in range(3)]
                offsets = (actual != expected).nonzero().flatten().tolist()
                item = dict(
                    iteration=iteration,
                    changed_bytes=len(offsets),
                    examples=[
                        dict(
                            offset=o,
                            expected=int(expected[o]),
                            actual=int(actual[o]),
                            xor=int(expected[o]) ^ int(actual[o]),
                        )
                        for o in offsets[:16]
                    ],
                    source_unchanged=torch.equal(current_source, source_raw),
                    rereads_equal=[torch.equal(value, actual) for value in rereads],
                    output_address=output.buffer_address(),
                )
                with gzip.open(args.output / f"fault-{iteration:06d}.pt.gz", "wb", compresslevel=1) as stream:
                    torch.save(
                        dict(source=current_source, output=actual, expected=expected, rereads=rereads, report=item),
                        stream,
                    )
                report["faults"].append(item)
                print(json.dumps(item), flush=True)
                checkpoint("fault captured")
                if not args.keep_going:
                    break
            ttnn.deallocate(output)
            output = None
            if report["iterations"] % 10000 == 0:
                checkpoint("running")
        report["source_unchanged_end"] = torch.equal(read_raw(source), source_raw)
        assert report["source_unchanged_end"], "Source changed during the test"
        report["seconds"] = time.monotonic() - started
        checkpoint("complete")
        return 1 if report["faults"] else 0
    except BaseException:
        report["exception"] = traceback.format_exc()
        checkpoint("exception")
        raise
    finally:
        if output is not None:
            ttnn.deallocate(output)
        if source is not None:
            ttnn.deallocate(source)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
