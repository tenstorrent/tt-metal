# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Can a fixed-capacity drafter K/V cache be written with ``fill_cache`` without recompiling?

The persistent anchor cache rests on three mechanical claims, and all three are cheap to
settle without loading the 30B target:

1. **Which binding exists** -- ``ttnn.fill_cache`` or ``ttnn.kv_cache.fill_cache``.
2. **A varying tile-aligned ``update_idx`` does not recompile.**  This is the whole design:
   ``UpdateKVCacheOperation::compute_program_hash`` hashes only ``(op_type, cache, input)``, so
   the destination row is a runtime argument re-patched on every program-cache hit.  If that is
   wrong, the anchor cache reintroduces exactly the shape churn F6 measured at 82x and the
   design is dead.
3. **The undocumented work-split limit.**  ``fill_cache``'s program factory computes
   ``num_blocks_of_work = heads * rows / 32`` and advances ``Wt`` tiles per block without
   accounting for head boundaries, so a core assigned blocks that cross one writes to the wrong
   tiles.  At one tile row (``rows == 32``) that is 8 blocks and safe; the prompt-sized fill is
   the case that must be chunked.

Graded on bit-equality, because a cache write has no numerics of its own to lose.
"""

from __future__ import annotations

import argparse

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import close_generator_mesh, open_generator_mesh

KV_HEADS = 8
HEAD_DIM = 128
TILE = 32


def _fill_cache_fn():
    """Return the live ``fill_cache`` binding, whichever module it is registered under."""
    for owner, name in ((ttnn, "fill_cache"), (getattr(ttnn, "kv_cache", None), "fill_cache")):
        if owner is not None and hasattr(owner, name):
            return getattr(owner, name), f"{'ttnn' if owner is ttnn else 'ttnn.kv_cache'}.{name}"
    raise AttributeError("no fill_cache binding found on ttnn or ttnn.kv_cache")


def _program_cache_entries(mesh) -> int:
    for attr in ("num_program_cache_entries",):
        fn = getattr(mesh, attr, None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:  # noqa: BLE001 - probe only
                pass
    total = 0
    try:
        for dev in mesh.get_devices():
            fn = getattr(dev, "num_program_cache_entries", None)
            if callable(fn):
                total += int(fn())
    except Exception:  # noqa: BLE001
        return -1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=544, help="cache rows; a multiple of 32")
    parser.add_argument("--offsets", type=int, nargs="*", default=[0, 32, 64, 96, 128, 160, 192])
    args = parser.parse_args()

    fill_cache, binding_name = _fill_cache_fn()
    print(f"binding: {binding_name}")

    mesh = open_generator_mesh()
    try:
        grid = mesh.compute_with_storage_grid_size()
        cores = grid.x * grid.y
        print(f"grid {grid.x}x{grid.y} = {cores} cores; fill of R rows needs 8*R/32 <= {cores}")

        cap = args.capacity
        host_cache = torch.zeros((1, KV_HEADS, cap, HEAD_DIM), dtype=torch.bfloat16)
        cache = ttnn.from_torch(
            host_cache,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        torch.manual_seed(0)
        reference = torch.zeros_like(host_cache)
        before = _program_cache_entries(mesh)
        counts = [before]

        for offset in args.offsets:
            rows = TILE
            block = torch.randn((1, KV_HEADS, rows, HEAD_DIM), dtype=torch.float32).to(torch.bfloat16)
            tt_block = ttnn.from_torch(
                block,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            fill_cache(cache, tt_block, 0, update_idx=offset)
            ttnn.deallocate(tt_block)
            reference[:, :, offset : offset + rows, :] = block
            counts.append(_program_cache_entries(mesh))

        ttnn.synchronize_device(mesh)
        got = ttnn.to_torch(ttnn.get_device_tensors(cache)[0]).reshape(1, KV_HEADS, cap, HEAD_DIM)

        exact = bool(torch.equal(got.to(torch.float32), reference.to(torch.float32)))
        finite = bool(torch.isfinite(got.to(torch.float32)).all())
        tail_start = max(args.offsets) + TILE
        tail_zero = bool((got[:, :, tail_start:, :].to(torch.float32) == 0).all())

        print(f"program cache entries: {counts}")
        print(
            f"  grew during the offset sequence: {counts[1] != counts[-1]}  "
            f"(first write {counts[0]} -> {counts[1]}, then {counts[1]} -> {counts[-1]})"
        )
        print(f"bit-exact against reference : {exact}")
        print(f"all finite                  : {finite}")
        print(f"untouched tail still zero   : {tail_zero}")

        if not exact:
            diff = (got.to(torch.float32) - reference.to(torch.float32)).abs()
            bad = torch.nonzero(diff.sum(dim=-1)[0, 0])
            print(f"  first mismatching rows (head 0): {bad.flatten()[:12].tolist()}")
            print(f"  max abs diff: {float(diff.max())}")

        ttnn.deallocate(cache)
    finally:
        close_generator_mesh(mesh)


if __name__ == "__main__":
    main()
