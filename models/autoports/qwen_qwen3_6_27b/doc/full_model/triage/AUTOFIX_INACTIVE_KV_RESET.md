# AutoFix: inactive paged KV and slot reuse

Date: 2026-08-13

Status: incomplete hardware gate. The proposed code is unproven and has not
been refuted. Do not use this report to claim full-model stage completion.

## Proposed repair

- Full-attention decode keeps real positions for RoPE and paged SDPA, but
  passes an active-mask-derived `INT32 -1` to `paged_update_cache` for inactive
  users. Both paged-update dataflow kernels implement `-1` as `skip_update`.
- `reset_slots` clears only selected slots' linear conv/recurrent state,
  invalidates those slots logically, and rejects decode until a new prefill.
- Refill uses a separate cache-fill page table whose inactive rows contain
  `-1`; `paged_fill_cache` skips those entries while attention reads keep the
  ordinary page table. This avoids extra physical KV blocks or reduced context.
- Generator host uploads are converted to the requested dtype and contiguous.

Focused probes were added at `tests/full_attention_inactive_kv.py` and extended
in `tests/full_model_mixed_slots.py`. Python compilation, `git diff --check`,
and the two public-contract tests passed. Device assertions did not execute
because the command queue stalled during model setup.

## Command-queue evidence

The first B2 four-layer wrapper command was:

```bash
TT_METAL_HOME=/home/mvasiljevic/tt-metal \
python models/autoports/qwen_qwen3_6_27b/tests/full_model_mixed_slots.py
```

After more than twelve minutes, GDB showed the main thread blocked in
`FDMeshCommandQueue::finish_nolock -> enqueue_write_shard_to_sub_grid ->
enqueue_write_mesh_buffer`. The terminal had printed `Pinned source memory
start address ... must be aligned 64 B`. A fresh one-layer official-weight
repro before recovery blocked in `to_device -> enqueue_write_tensor ->
enqueue_write_shards -> finish_nolock`, before decode.

After the first explicit reset, the focused repro instead blocked in
`CreateGlobalSemaphore -> GlobalSemaphore::setup_buffer/reset_semaphore_value
-> enqueue_write_mesh_buffer -> FDMeshCommandQueue::finish_nolock`.

After the second bounded recovery, independent mesh and global-semaphore
smokes passed. Nevertheless, the unchanged official-weight repro again blocked
in `enqueue_write_mesh_buffer -> FDMeshCommandQueue::finish_nolock` during
model setup, before `ttnn.where`, paged cache update, or a cache assertion.
The cache hypothesis is therefore unproven, not refuted.

`tools/tt-triage.py` was attempted during the first live hang. It could not
collect valid running-operation or RISC evidence because the installed UMD
binding rejected its `noc_read(..., memoryview)` call as an incompatible
signature. The GDB stacks above are the usable live-hang evidence.

## Recovery history

First safe recovery:

```bash
tt-smi -r 0 1 2 3
tt-smi -s
```

It exited successfully. All four p300c boards reported `dram_status=true`,
live heartbeats, and zero uncorrected GDDR errors. The subsequent repro stalled
at global-semaphore creation.

Second bounded recovery:

```bash
timeout 60 tt-smi -ls --local
timeout 180 tt-smi -r
timeout 60 tt-smi -ls --local
```

The reset reported `Resetting all PCI devices: [0, 1, 2, 3]`, reinitialized
the boards, and the post-list showed all four p300c devices as resettable.
Logs are `/tmp/qwen_second_recovery_pre_list.log`,
`/tmp/qwen_second_recovery_reset.log`, and
`/tmp/qwen_second_recovery_post_list.log`.

Fresh-process smokes then passed:

- `MeshShape(1,4)`, trace region zero: `MESH_OPEN_OK`/`MESH_CLOSE_OK` in
  `/tmp/qwen_mesh_open_close_smoke.log`.
- `FABRIC_1D_RING` plus `ttnn.create_global_semaphore`:
  `GLOBAL_SEMAPHORE_OK`/`FABRIC_CLOSE_OK` in
  `/tmp/qwen_global_semaphore_smoke.log`.

The unchanged focused repro was then run with a 600-second bound:

```bash
timeout 600 bash -lc \
  'TT_METAL_HOME=/home/mvasiljevic/tt-metal python \
   models/autoports/qwen_qwen3_6_27b/tests/full_attention_inactive_kv.py'
```

Log: `/tmp/full_attention_inactive_kv_second_reset.log`. It stalled during a
model-setup mesh-buffer write and was terminated after the live GDB capture.
Post-failure `tt-smi` showed all four boards with `dram_status=true`, live
heartbeat `384`, and `GDDR_UNCORR_ERRS=0`; see
`/tmp/qwen_second_recovery_post_failure_health.log`.

## Required next action

Two safe board-reset attempts are exhausted. An operator must reboot the host
before further TT hardware work. After reboot, rerun unchanged
`full_attention_inactive_kv.py` first. Only an exact inactive-KV preservation
pass plus active-KV change permits proceeding to the B2 reset/refill probe.
Do not claim the repair proven, stage completion, clean rereview, or commit.

## Resumed source-only remediation

The independent follow-up audit at `CACHE_FIX_SOURCE_AUDIT.md` verified the two
kernel `-1` sentinels but found an unfenced trace-release race after nonblocking
token-out replay. That source defect is repaired: trace teardown now fences the
mesh before releasing trace buffers/snapshots, and whole-cache reset fences its
zeroing kernels before returning. The active-mask predicate is now typecast to
INT32 before producing INT32 update positions, avoiding the uncovered mixed
BF16/INT32 ternary variant. Static compilation, public-contract tests, and diff
checks pass. These source results do not replace the required post-reboot TP4
cache/reset/trace lifecycle proof.

---

## OPERATOR CORRECTION (supersedes the reboot requirement above)

**Do not wait for a host reboot. No reboot is required and none will happen.**
The "operator must reboot the host" conclusion earlier in this file was tested
and is not supported. Read
[`BLOCKER_REDIAGNOSIS.md`](BLOCKER_REDIAGNOSIS.md) before acting on anything in
this document.

Measured on this host after that conclusion was written:

- A fresh `MeshShape(1,4)` open + `FABRIC_1D_RING` + `create_global_semaphore` +
  close passes cleanly and repeatedly; all four p300c enumerate. Two `tt-smi -r`
  cycles each restored a working mesh.
- The model-setup stall is not intrinsic. The `from_torch` call it stalled in
  (`mlp.gate_proj.weight`, `[5120,17408]`, BFP4, sharded dim -1) takes **0.88 s**
  timed in isolation on healthy devices. After a board reset,
  `from_state_dict` completes in about 2 minutes.
- On freshly reset devices the repro reaches the **first decode all-reduce** and
  hangs there at 100 % CPU: `_all_reduce` (`multichip_decoder.py:441`) via
  `_tp_linear` -> `_full_attention_decode`, identical across stack samples at
  +1/+2/+3 minutes. Batch 2, one inactive row, one full-attention layer.

**That collective hang is the actual bug to fix.** Stages 04 and 05 ran the same
ring all-reduce successfully at batch 1 and batch 32, so the trigger is the
batch-2 / inactive-row configuration rather than the collective in general.

**The trap that produced the reboot conclusion:** killing a process while it is
inside a device operation leaves the devices in a state where the *next* run
hangs earlier — in setup rather than in the collective — which reads as hardware
degrading run over run. `tt-smi -r` clears it. After any such kill, reset and
re-verify with a mesh smoke before drawing conclusions from the next run.

Suggested first cuts on the real bug are in `BLOCKER_REDIAGNOSIS.md`. Also noted
there: the Watcher run cannot start at all because instrumentation makes
ACTIVE_ETH fabric firmware 27,920 bytes against a 25,600-byte kernel-config
limit — try `TT_METAL_WATCHER_DISABLE_ETH=1`, which stage 04 used successfully,
before declaring Watcher evidence unobtainable.
