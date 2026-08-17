# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""How much DRAM a 1x1 mesh actually hands out, by asking until it refuses.

`doc/context_contract.json` records `usable_dram_bytes`, and the capability argument (batch 32 at
the full advertised context fits) rests on it, so it should be a recorded measurement rather than a
number quoted from a work log. This allocates 512 MiB DRAM tensors until the bank manager raises,
prints the total, and frees everything.

Not a pytest: it deliberately drives the allocator to failure, and it needs the whole device to
itself.

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/probe_dram_capacity.py
"""

import torch

import ttnn

CHUNK_BYTES = 512 * 1024 * 1024
GIB = 1024**3


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    held = []
    try:
        # bf16, so element count is bytes / 2; shaped 2D and tile-aligned so nothing pads
        elems = CHUNK_BYTES // 2
        rows, cols = elems // 1024, 1024
        block = torch.zeros(1, 1, rows, cols, dtype=torch.bfloat16)
        while True:
            try:
                held.append(
                    ttnn.from_torch(
                        block,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
            except RuntimeError as exc:
                first = str(exc).splitlines()[0]
                total = len(held) * CHUNK_BYTES
                print(f"CAP allocated {len(held)} x 512 MiB = {total} bytes = {total / GIB:.2f} GiB")
                print(f"CAP refused at chunk {len(held) + 1}: {first}")
                break
    finally:
        for tensor in held:
            ttnn.deallocate(tensor)
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
