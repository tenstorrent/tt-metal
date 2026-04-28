# MoE subgroups 4×2 hang — resume notes (2026-04-27)

Branch: `ianastasijevic/moe_expert_dispatch_subgroups`

## TL;DR

`test_ttnn_moe_subgroups.py` 4×2 hangs in `CombineDeviceOperation`
(triage Op Id 125) on 6 of 8 chips (`0, 2, 3, 4, 6, 7`). The two chips that
finished cleanly are mesh row 1 of subgroup 0: chips 1 and 5. Subgroup 1
(chips 2, 3, 6, 7) is fully stuck; subgroup 0 is half stuck (row 0 only).

## What passes / what hangs (4×2 mesh, two 2×2 subgroups along axis 0)

| Test | 8×1 | 4×2 |
|---|:-:|:-:|
| `test_subgroup_gather_histograms.py` | ✅ | ✅ |
| `test_prefill_dispatch_subgroups.py` | ✅ | ✅ |
| `test_prefill_combine_subgroups.py` (torch dispatch → TTNN combine) | ✅ | ✅ |
| `test_ttnn_dispatch_combine_subgroups.py` (TTNN dispatch → TTNN combine, no MLPs) | ✅ | ✅ |
| `test_ttnn_moe_subgroups.py` (full TtMoe: gate → dispatch → MLPs → untilize → **combine** → reduce) | ✅ | ❌ hangs |

The simple `dispatch → combine` chain on 4×2 passes; the full TtMoe forward
on 4×2 hangs at combine. The diff between the two is the expert-MLP +
`UntilizeDeviceOperation` (op id 124) sitting between dispatch and combine.

## Key triage finding

From `triage_logs/tt-triage_moe_subgroups_4x2.log` (`dump_running_operations.py`
section):

```
Op Id 125 = CombineDeviceOperation
  Tensor[0]: logical_shape [1, 1, 16, 800, 7168] BFLOAT16 INTERLEAVED DRAM
  Tensor[1]: logical_shape [1, 1, 16, 800, 5]    INT32    INTERLEAVED DRAM
  Tensor[2]: logical_shape [1, 64]               UINT32   INTERLEAVED DRAM
  Prev Op = 124 UntilizeDeviceOperation, shape [1, 16, 800, 7168] BFLOAT8_B
  Device Cnt: 6 — Devices: 0, 2, 3, 4, 6, ... (the 6th is almost certainly 7)
  Cores: 0:1-2, 2:1-2, 3:1-2, 4:1-2, 6:1-2, ...
```

Stuck chips: `{0, 2, 3, 4, 6, 7}` → physical mesh layout (row, col):
- (0,0) chip 0 — SG0 row 0
- (0,1) chip 4 — SG0 row 0
- (1,0) chip 1 — SG0 row 1 — **finished**
- (1,1) chip 5 — SG0 row 1 — **finished**
- (2,0) chip 2 — SG1 row 0
- (2,1) chip 6 — SG1 row 0
- (3,0) chip 3 — SG1 row 1
- (3,1) chip 7 — SG1 row 1

So SG0's mesh-row-1 finished combine; SG0's mesh-row-0 didn't, and all of
SG1 didn't.

## What's healthy

Host-side subgroup partitioning is correct on 4×2. From the
`combine create_mesh_workload:` log_info we added:

```
mesh_shape=MeshShape([4, 2])  num_dispatch_subgroups=2  subgroup_axis=0
tensor_coords.num_ranges=1
tensor_coords={[(0,0)..(3,1)]}                ← single range, full mesh
subgroups={[(0,0)..(1,1)], [(2,0)..(3,1)]}    ← two clean 2×2 subgroups
SG0 init_barrier_sem.address=0x17ffc0
SG1 init_barrier_sem.address=0x17ff80
```

All 8 `combine create_at` log lines fire (one per chip), each with
`subgroup_num_rows=2 subgroup_num_cols=2 subgroup_num_devices=4` and a
clean local `linearized_subgroup_coord` 0..3.

`get_neighbors_in_range` with axis=0 correctly opens N/S only — every chip
has exactly one column-mate as its neighbor.

Dispatch (op before combine) ran on the same mesh view and finished cleanly.

## What we don't know yet

Which barrier each stuck chip is parked on. The DPRINTs we added in
`writer_combine.cpp` and `reader_combine.cpp` will tell us, but
`feature_DPRINT_enabled=false` in the triage run, so they didn't fire.

Suggested re-run with DPRINTs on:

```bash
TT_METAL_DPRINT_CORES='(0,0)-(11,9)' \
TT_METAL_DPRINT_RISCVS='BR' \
pytest -svx \
  "models/demos/deepseek_v3_d_p/tests/pcc/test_ttnn_moe_subgroups.py::test_ttnn_moe_subgroups[blackhole-subgroups-2x2x2-mesh-4x2-1link-1600-7168-2048-64-8-2-GateComputeMode.HOST_ALL]" \
  > /tmp/moe_4x2_dprint.log 2>&1 &
```

Each chip's last `Combine writer[N]:` line tells which barrier it's at:
- only `init send begin` printed → chip never reached its neighbor
- `init send done; waiting` but no `init wait satisfied` → init exchange deadlock
- `init wait satisfied` but no `exit send begin` → stuck in fabric data loop
- `exit send done; waiting` but no `exit wait satisfied` → exit semaphore deadlock

## Working hypotheses (in order of likelihood)

1. **Fabric exit-semaphore deadlock specific to the moe-path data shape.**
   Combine on the moe path consumes a tilized→untilized BFLOAT16 tensor
   `[1,1,16,800,7168]` — 16 experts × 800 max-tokens × 7168 hidden, ~50× the
   payload of the standalone `test_ttnn_dispatch_combine_subgroups` chain
   (which uses `experts_per_chip=4, max_tokens=128`). The data loop is much
   longer; chips with little/no tokens (1, 5) reach the exit semaphore quickly
   and might call `close_direction_connections(...)` before peers send the
   exit packet. Worth checking `close_direction_connections` semantics.

2. **`UntilizeDeviceOperation` (op 124) leaves fabric or L1 state that
   poisons combine.** Untilize is run on a 6-device subset (Device Cnt: 6
   in the running-ops table for op 125 — but op 124 isn't visible, would
   need re-triage to see). If untilize touches L1 small / DRAM in ways that
   shift the addresses combine sees, addresses may diverge between subgroup
   peers.

3. **A long combine data loop hits a fabric mux flow-control issue that's
   8×1 vs 4×2 specific** (different physical eth chans on 4×2). Subgroup 1
   being 100% stuck while only half of subgroup 0 hangs would fit if the
   physical eth-chan layout differs between the two subgroups.

## Files in this directory

- `RESUME_NOTES.md` — this file
- `tt-triage_moe_subgroups_4x2.log` — `tools/tt-triage.py -vv` output during the hang
- `pytest_moe_subgroups_4x2.log` — pytest output of the 4×2 hang (last log line:
  `[TtMoe.forward] combined_output shape: Shape([1, 1, 1600, 8, 7168])`)
- `pytest_moe_subgroups_8x1.log` — pytest output of the 8×1 PASSING moe run
- `pytest_dispatch_combine_4x2_chain.log` — passing 4×2 dispatch→combine chain (no MLPs)
- `pytest_dispatch_combine_8x1_chain.log` — passing 8×1 dispatch→combine chain

## Diagnostic instrumentation already added (unstaged in branch)

Search for `log_info` and raw `DPRINT` additions in:
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/combine_program_factory.cpp`
  — top-of-`create_mesh_workload` logs `tensor_coords`, subgroup ranges,
  per-subgroup `init_barrier_sem.address`.
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/dispatch_program_factory.cpp`
  — same pattern, plus `final_barrier_sem.address`.
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp`
  — raw DPRINTs around init send/wait, exit send/wait, and exit.
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp`
  — DPRINTs around the writer-barrier wait.
- `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/writer_dispatch.cpp`
  — same init/exit DPRINT pattern as writer_combine.

The `log_info` lines are always-on. The `DPRINT` lines need
`TT_METAL_DPRINT_CORES` + `TT_METAL_DPRINT_RISCVS=BR` to fire.

## Next actions when resuming

1. Re-run 4×2 moe with DPRINTs on (command above). Hang should reproduce.
2. Inspect `/tmp/moe_4x2_dprint.log` for the last `Combine writer[N]` /
   `Combine reader[N]` line per chip — that tells exactly where they died.
3. If the hang is at the *exit* semaphore wait while peers already exited,
   the fix is likely in the exit-exchange ordering (don't close fabric
   connections until all peers have acked exit).
