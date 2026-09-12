# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Upload existing tensorbins without modifying them; capture stdout/stderr for analysis."""

import argparse
import hashlib
import struct
import time
from pathlib import Path

import ttnn

CACHE_ROOT = Path("/mnt/models/huggingface/tt_cache/gemma4_d_p/google--gemma-4-31B-it/tensor_cache_bf16_mesh8x4")
DEFAULT_FILES = [
    CACHE_ROOT / "layer_59/mlp/down_proj.weight_tp4_bfp8_dtype_BFLOAT8_B_layout_TILE.tensorbin",
    CACHE_ROOT / "final_norm/weight_dtype_BFLOAT16_layout_ROW_MAJOR.tensorbin",
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, default=DEFAULT_FILES)
    parser.add_argument("--cache-dir", type=Path, help="Recursively upload every tensorbin in this directory")
    args = parser.parse_args()
    if args.cache_dir is not None:
        args.files = sorted(args.cache_dir.rglob("*.tensorbin"))
    if not args.files:
        parser.error("No tensorbins found")
    print(f"FILES={len(args.files)}; hashing source files before upload", flush=True)
    originals = {path: digest(path) for path in args.files}
    print(f"TTNN_MODULE={ttnn.__file__}", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    try:
        print(f"MESH_OPEN devices={mesh_device.get_num_devices()}", flush=True)
        for index, path in enumerate(args.files):
            with path.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
            offset = 8 + header_size
            print(
                f"BEGIN_TENSOR {index} path={path} payload_offset={offset} "
                f"mod16={offset % 16} mod64={offset % 64} payload_bytes={path.stat().st_size - offset}",
                flush=True,
            )
            host = ttnn.load_tensor(path)
            print(f"HOST shape={host.shape} dtype={host.dtype} layout={host.layout}", flush=True)
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            device_tensor = ttnn.to_device(host, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.synchronize_device(mesh_device)
            print(f"UPLOAD_DONE {index} elapsed_ms={(time.perf_counter() - start) * 1000:.3f}", flush=True)
            device_tensor.deallocate(True)
            del device_tensor, host
            print(f"END_TENSOR {index}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh_device)
        for path, expected in originals.items():
            assert digest(path) == expected, f"File contents changed: {path}"
        print("SOURCE_SHA256_UNCHANGED", flush=True)


if __name__ == "__main__":
    main()
