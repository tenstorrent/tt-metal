# SPDX-License-Identifier: Apache-2.0
"""Setup-only reproducer for checkpoint mmap versus owned transfer buffers."""
import argparse
import time

import torch

import ttnn

from ..tt.checkpoint import SNAPSHOT, load_weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clone", action="store_true")
    p.add_argument("--owned-same-alignment", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(8)
    value = load_weights("model.embed_tokens.")["model.embed_tokens.weight"]
    if a.clone:
        value = value.clone()
    if a.owned_same_alignment:
        residue = value.data_ptr() % 4096
        storage = torch.empty(value.numel() + 4096, dtype=value.dtype)
        offset = ((residue - storage.data_ptr()) % 4096) // value.element_size()
        owned = storage[offset : offset + value.numel()].reshape(value.shape)
        owned.copy_(value)
        value = owned
    print(
        "HOST_READY",
        str(SNAPSHOT),
        tuple(value.shape),
        value.dtype,
        a.clone,
        "pointer",
        value.data_ptr(),
        "alignment",
        {n: value.data_ptr() % n for n in (16, 32, 64, 4096)},
        flush=True,
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        tick = time.monotonic()
        print("TRANSFER_BEGIN", flush=True)
        weight = ttnn.from_torch(
            value,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)
        print("TRANSFER_DONE", time.monotonic() - tick, flush=True)
        ids = ttnn.from_torch(
            torch.tensor([[0, 42, 127999]], dtype=torch.int32),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        output = ttnn.embedding(ids, weight, layout=ttnn.TILE_LAYOUT)
        for tensor in ttnn.get_device_tensors(output):
            actual = ttnn.to_torch(tensor)
            assert torch.equal(actual, value[[0, 42, 127999]][None]), (
                (actual - value[[0, 42, 127999]][None]).abs().max()
            )
        print("EMBEDDING_TRANSFER_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
