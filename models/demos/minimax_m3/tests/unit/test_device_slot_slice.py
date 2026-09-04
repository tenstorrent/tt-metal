"""Byte-parity: device-valued slot select vs host-int slice.

The chunked-prefill cache read (attention/prefill.py) slices one (user,layer) slot out of the
packed cache with a host-int begin index, which bakes the slot into any captured trace. Request-mode
tracing needs the slot to vary per replay, i.e. read from a device tensor. ttnn.slice's device-tensor
path is a partition-select (slice_dim split into num_devices equal parts, one picked by the device
begin) that only reshapes slice_dim, so it cannot also bound the row dim -> slot via the device
partition-slice on dim 0, then the existing host-int slice bounds the row dim. This proves that
two-step path is bit-identical to the single host-int slice for an arbitrary slot.
"""

import torch

import ttnn


def _run(mesh):
    n_slots, max_rows, head_dim = 12, 256, 128
    torch.manual_seed(0)
    packed = torch.randn(n_slots, 1, max_rows, head_dim)
    dev = ttnn.from_torch(
        packed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    failures = []
    for slot in (0, 1, 5, n_slots - 1):
        for n_rows in (32, 128, max_rows):
            ref = ttnn.to_torch(ttnn.slice(dev, (slot, 0, 0, 0), (slot + 1, 1, n_rows, head_dim)))

            start = ttnn.from_torch(
                torch.tensor([slot, 0, 0, 0], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            end = ttnn.from_torch(
                torch.tensor([slot + 1, 1, max_rows, head_dim], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            slot_full = ttnn.slice(dev, starts=start, ends=end, slice_dim=0, num_devices=n_slots)  # [1,1,max_rows,hd]
            got = ttnn.to_torch(ttnn.slice(slot_full, (0, 0, 0, 0), (1, 1, n_rows, head_dim)))

            max_abs = (ref - got).abs().max().item()
            tag = f"slot={slot:2d} n_rows={n_rows:3d} shape={tuple(got.shape)} max_abs_diff={max_abs:.3e}"
            print(("PASS " if max_abs == 0.0 else "FAIL ") + tag, flush=True)
            if max_abs != 0.0:
                failures.append(tag)
    return failures


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        failures = _run(mesh)
    finally:
        ttnn.close_mesh_device(mesh)
    if failures:
        raise SystemExit(f"{len(failures)} byte-parity FAILURES: {failures}")
    print("ALL PASS: device slot-slice == host-int slice (byte-parity)", flush=True)


if __name__ == "__main__":
    main()
