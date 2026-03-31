# MoE Dispatch — Debug Log

## Bug: go_sem signaled to self instead of next EP device

**Date:** 2026-03-31

**Symptom:** Only devices 0 and 3 (column 0 and column 3 of TP row 0) printed `ALL DONE`. All others hung after `fabric open, go_sem wait`.

```
0:(x=0,y=0):BR: SENDER[0]: ALL DONE
3:(x=0,y=0):BR: SENDER[0]: ALL DONE
# all other devices stuck at "fabric open, go_sem wait"
```

**Root cause:** In `moe_dispatch_program_factory.cpp`, `next_sender_noc_x/y` RT args were set to `sender_phys.x/y` — this device's own sender core — instead of the sender core on the next EP device. Every device was signaling its own `go_sem`, which had no effect on the waiting device next in the EP chain.

**Why device 0 completed:** Column 0 has `go_sem` pre-initialized to `E` (all turns pre-granted), so it never waits and always runs to completion.

**Why device 3 completed:** Column 3 is `is_last_ep_device=1`, so it skips the go_sem signal step. If all its `expert_n_rows[e]` happen to be zero (no tokens assigned), it loops through all experts without sending anything and exits cleanly.

**Fix (`moe_dispatch_program_factory.cpp`):**
```cpp
// Before:
sender_rt.push_back(sender_phys.x);  // wrong — this device's own core
sender_rt.push_back(sender_phys.y);

// After:
uint32_t next_sender_x = 0, next_sender_y = 0;
if (!is_last && forward_coord.has_value()) {
    IDevice* next_device = mesh_device->get_device(forward_coord.value());
    auto next_sender_phys = next_device->worker_core_from_logical_core(sender_core);
    next_sender_x = next_sender_phys.x;
    next_sender_y = next_sender_phys.y;
}
sender_rt.push_back(next_sender_x);
sender_rt.push_back(next_sender_y);
```
