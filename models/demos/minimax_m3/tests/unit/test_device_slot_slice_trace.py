"""Trace-replayability of the device-valued slot slice.

Request-mode tracing needs one captured trace to serve any user: the slot begin rides a persistent
device tensor, and updating that tensor between replays must re-target the slot WITHOUT recapture.
The slice reader NoC-reads the start tensor at kernel runtime, so execute_trace should observe fresh
contents. This proves it: capture the slice over a persistent start = slot A, replay -> slot A;
copy_host_to_device the start = slot B, replay the SAME trace -> slot B.
"""

import torch

import ttnn


def main():
    n_slots, rows, head_dim = 12, 256, 128
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000)
    try:
        # Distinct per-slot contents so a mis-targeted slot is caught (slot k filled with value k).
        packed = torch.arange(n_slots, dtype=torch.float32).view(n_slots, 1, 1, 1).expand(n_slots, 1, rows, head_dim)
        dev = ttnn.from_torch(
            packed.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def host_ref(slot):
            return ttnn.to_torch(ttnn.slice(dev, (slot, 0, 0, 0), (slot + 1, 1, rows, head_dim)))

        slot_a, slot_b = 3, 9
        start = ttnn.from_torch(
            torch.tensor([slot_a, 0, 0, 0], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        end = ttnn.from_torch(
            torch.tensor([slot_a + 1, 1, rows, head_dim], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def sliced():
            return ttnn.slice(dev, starts=start, ends=end, slice_dim=0, num_devices=n_slots)

        sliced()  # warm compile before capture
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = sliced()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)

        def replay_and_check(expect_slot, tag):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            got = ttnn.to_torch(out)
            ref = host_ref(expect_slot)
            max_abs = (got - ref).abs().max().item()
            val = got.flatten()[0].item()
            print(
                f"{'PASS' if max_abs == 0.0 else 'FAIL'} {tag}: slot={expect_slot} first_val={val:.1f} max_abs_diff={max_abs:.3e}",
                flush=True,
            )
            return max_abs == 0.0

        ok = replay_and_check(slot_a, "replay@capture-slot")

        # Re-target: overwrite the persistent start in place; the SAME trace must now read slot B.
        host_start = ttnn.from_torch(
            torch.tensor([slot_b, 0, 0, 0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(host_start, start)
        ok &= replay_and_check(slot_b, "replay@updated-slot")

        # Back to A to rule out a one-way latch.
        host_start_a = ttnn.from_torch(
            torch.tensor([slot_a, 0, 0, 0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        ttnn.copy_host_to_device_tensor(host_start_a, start)
        ok &= replay_and_check(slot_a, "replay@back-to-A")

        ttnn.release_trace(mesh, tid)
    finally:
        ttnn.close_mesh_device(mesh)
    if not ok:
        raise SystemExit("FAIL: persistent start tensor did not re-target the traced slice")
    print("ALL PASS: one trace re-targets the slot via the persistent start tensor (request-mode ready)", flush=True)


if __name__ == "__main__":
    main()
