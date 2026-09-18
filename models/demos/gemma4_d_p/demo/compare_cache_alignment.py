# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare aligned/unaligned tensorbin uploads, including exact device readback."""

import argparse
import hashlib
import json
import statistics
import struct
import time
from pathlib import Path

import torch

import ttnn

PAIR_DIR = Path("/tmp/gemma_tensorbin_alignment")


def file_info(path):
    with path.open("rb") as stream:
        offset = 8 + struct.unpack("<Q", stream.read(8))[0]
        stream.seek(offset)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "payload_offset": offset, "payload_sha256": digest}


def check_readback(device_tensor, host):
    readback = ttnn.from_device(device_tensor)
    expected = ttnn.get_device_tensors(host)
    actual = ttnn.get_device_tensors(readback)
    assert len(expected) == len(actual) == 32
    for rank, (reference, result) in enumerate(zip(expected, actual)):
        assert torch.equal(ttnn.to_torch(reference), ttnn.to_torch(result)), f"Readback mismatch on rank {rank}"
    print("READBACK_MATCH all_32_shards", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unaligned", type=Path, default=PAIR_DIR / "down_proj_unaligned.tensorbin")
    parser.add_argument("--aligned", type=Path, default=PAIR_DIR / "down_proj_aligned.tensorbin")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", type=Path, default=PAIR_DIR / "comparison.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    paths = {"unaligned": args.unaligned, "aligned": args.aligned}
    info = {name: file_info(path) for name, path in paths.items()}
    assert info["unaligned"]["payload_sha256"] == info["aligned"]["payload_sha256"]
    assert info["unaligned"]["payload_offset"] % 16 != 0
    assert info["aligned"]["payload_offset"] % 16 == 0
    samples = {name: [] for name in paths}
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    try:
        hosts = {name: ttnn.load_tensor(path) for name, path in paths.items()}
        a, b = hosts.values()
        assert a.shape == b.shape and a.dtype == b.dtype and a.layout == b.layout
        # Keep mappings alive to compare repeated uploads with a warm pin cache.
        # Round zero warms up and checks correctness; alternate order afterwards.
        for iteration in range(args.repeats + 1):
            order = list(paths) if iteration % 2 == 0 else list(reversed(paths))
            for name in order:
                print(f"BEGIN_UPLOAD {name} iteration={iteration} warmup={iteration == 0}", flush=True)
                ttnn.synchronize_device(mesh_device)
                start = time.perf_counter()
                device_tensor = ttnn.to_device(hosts[name], mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.synchronize_device(mesh_device)
                elapsed = (time.perf_counter() - start) * 1000
                print(f"END_UPLOAD {name} elapsed_ms={elapsed:.3f}", flush=True)
                try:
                    if iteration == 0:
                        check_readback(device_tensor, hosts[name])
                    else:
                        samples[name].append(elapsed)
                finally:
                    device_tensor.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh_device)
    assert all(file_info(path) == info[name] for name, path in paths.items())
    result = {
        "files": info,
        "samples_ms": samples,
        "median_ms": {name: statistics.median(values) for name, values in samples.items()},
        "readback": "exact match on all 32 shards for both files",
        "timing": "synchronized upload including allocation/pinning/dispatch/logging; excludes load and readback",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
