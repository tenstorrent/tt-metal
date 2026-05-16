<!--
SUMMARY: Step-by-step correctness audit of the deterministic hang in MeshDevice1x4FabricFixture.TestGenericOpAllGather
KEYWORDS: allgather, hang, deadlock, dispatch, fabric, MUX, generic_op, event-recording, done-signal, K-accounting
SOURCE: Code analysis of branch nsexton/0-racecondition-hunt worktree at /workspace/group/worktrees/racecondition-hunt/
SCOPE: Test setup, sub-device allocation, event recording, worker done-signal accounting, MUX termination, root cause
USE WHEN: Investigating hangs in generic_op allgather tests, dispatch K-value mismatches, fabric MUX lifecycle issues
-->

# AllGather Hang Audit: `MeshDevice1x4FabricFixture.TestGenericOpAllGather`

**Test file**: `tests/ttnn/unit_tests/gtests/test_generic_op.cpp` lines 1042-1383
**Branch**: `nsexton/0-racecondition-hunt`
**Symptom**: Deterministic hang every CI run at/inside `ttnn::generic_op()`, 25-minute timeout

---

## 1. Test Setup: What Programs Run Where

The test creates a **MeshProgramDescriptor** with 4 programs (one per device in a 1x4 linear mesh). Each program runs:

- **Reader kernel** on `worker_cores = {(1,0), (3,0)}` — DataMovement, Reader config (BRISC)
- **Writer kernel** on `worker_cores = {(1,0), (3,0)}` — DataMovement, Writer config (NCRISC), `USE_WORKER_MUX` defined
- **Forward MUX kernel** on core `(0,0)` — conditionally, only if `has_forward` (RISCV_0)
- **Backward MUX kernel** on core `(2,0)` — conditionally, only if `has_backward` (RISCV_0)

Per-device layout (Linear topology, `gather_dim=0`):

```
Device  ring_idx  has_fwd  has_bwd  fwd_tgt  bwd_tgt  Programmed cores        MUX cores
------  --------  -------  -------  -------  -------  ---------------------   ----------
  0        0       true    false      3        0      {0,0},{1,0},{3,0}       {0,0} fwd
  1        1       true    true       2        1      {0,0},{1,0},{2,0},{3,0} {0,0} fwd + {2,0} bwd
  2        2       true    true       1        2      {0,0},{1,0},{2,0},{3,0} {0,0} fwd + {2,0} bwd
  3        3       false   true       0        3      {1,0},{2,0},{3,0}       {2,0} bwd
```

Key compile-time args differ per device: `num_targets_forward`, `num_targets_backward`, `my_chip_id`.

Each core {1,0} runs the **forward direction** (direction=0) reader+writer pair.
Each core {3,0} runs the **backward direction** (direction=1) reader+writer pair.

---

## 2. Sub-Device Allocation

The test uses the **default sub-device manager**. This means:
- There is a single sub-device: `SubDeviceId{0}`
- It contains ALL TENSIX worker cores on each device
- `determine_sub_device_ids()` always returns `{SubDeviceId{0}}`

The prior fix (commit `8d4a40e56b`) that scoped `enqueue_record_event_to_host()` to workload sub-device IDs is a **no-op**: with the default sub-device manager, both the fixed and unfixed paths yield `{SubDeviceId{0}}`.

---

## 3. Event Recording: What K Does DISPATCH_D Wait For

**K computation** (in `enqueue_mesh_workload`, line 287-294):

```
num_workers = mesh_device_->num_worker_cores(TENSIX, sub_device_id)
```

With the default sub-device manager, this is ALL TENSIX cores on one physical device. On a Wormhole chip this is typically 64 (8x8 grid). K is computed ONCE for the entire mesh workload.

**Event wait** (in `issue_record_event_commands`, line 102-139):

For each sub_device_id in the event's scope (just `{0}`), DISPATCH_D issues a `CQ_DISPATCH_CMD_WAIT` for `expected_num_workers_completed[0]` = previous + num_TENSIX_cores.

Since this is the first workload (previous=0): **K = num_TENSIX_cores**.

Each device's DISPATCH_D independently waits for K done signals from its local TENSIX grid.

---

## 4. Worker Done-Signal Accounting: Can K Be Reached?

When the go signal is multicast to all TENSIX cores in the sub-device:
- **Unprogrammed cores** (majority): receive go signal, see no pending kernels, immediately send done signal
- **Programmed cores** (3 or 4 per device): execute their kernels, then send done signal when ALL riscs on the core complete

For K to be reached, every programmed kernel on every device must exit. The analysis below traces kernel termination:

### Forward direction (direction=0) data flow:

Linear pipeline: Device 0 → Device 1 → Device 2 → Device 3
- Device 0 fwd writer: sends local slice via fabric (1 hop) to device 1. No forwarding. Terminates MUX. Exits.
- Device 1 fwd writer: sends local slice + forwards 1 slice from device 0. Terminates MUX. Exits.
- Device 2 fwd writer: sends local slice + forwards 2 slices. Terminates MUX. Exits.
- Device 3 fwd writer: no fabric send (num_targets_forward=0, mux_connection_valid=false). Drains CB. Exits.
- Device 3 fwd reader: waits for 3 semaphore increments from device 2's sends. Exits when received.

### Backward direction (direction=1) data flow:

Linear pipeline: Device 3 → Device 2 → Device 1 → Device 0
- Device 3 bwd writer: sends local slice 1 hop backward + writes local DRAM. No forwarding. Terminates MUX. Exits.
- Device 2 bwd writer: sends local slice + forwards 1 slice. Terminates MUX. Exits.
- Device 1 bwd writer: sends local slice + forwards 2 slices. Terminates MUX. Exits.
- Device 0 bwd writer: no fabric send (num_targets_backward=0). Writes local DRAM only. No MUX. Exits.
- Device 0 bwd reader: waits for 3 semaphore increments from device 1's sends. Exits when received.

### CB analysis:

Each core has its own CB instance (separate cores for fwd/bwd). Reader is producer, writer is consumer. CB size = 3 × `num_tiles_to_write_per_packet` (triple buffer). No cross-core or cross-direction CB sharing. **No CB deadlock possible.**

### Cross-direction analysis:

Forward and backward directions use different cores, different MUXes, different fabric links. No resource contention between directions. **No cross-direction deadlock possible.**

---

## 5. MUX Termination Race Analysis

Every MUX that exists in a program has exactly one writer that terminates it:

```
Device 0: fwd MUX on (0,0) ← terminated by fwd writer on (1,0) ✓
Device 1: fwd MUX on (0,0) ← terminated by fwd writer on (1,0) ✓
          bwd MUX on (2,0) ← terminated by bwd writer on (3,0) ✓
Device 2: (same as device 1) ✓
Device 3: bwd MUX on (2,0) ← terminated by bwd writer on (3,0) ✓
```

- `num_mux_clients = 1` (from CT arg 22)
- Termination master waits for `num_mux_clients - 1 = 0` semaphore → immediate
- Then sends `fabric_endpoint_terminate()` to MUX → graceful drain → MUX exits

For writers with `mux_connection_valid=false` (device 0 backward, device 3 forward), there is no corresponding MUX in their program, so no termination needed and `fabric_direction_connection = nullptr` is never dereferenced (guarded by `constexpr` branches).

**No MUX termination race exists.**

---

## 6. Root Cause Hypothesis

### What code analysis rules OUT:

1. **CB deadlock**: No circular dependency. Linear pipeline, triple-buffered.
2. **MUX termination race**: All MUXes properly terminated by their writers.
3. **Sub-device K mismatch**: The scoping fix is a no-op with the default sub-device manager.
4. **Cross-direction deadlock**: Separate cores, MUXes, and fabric links per direction.
5. **Nullptr dereference**: All fabric_direction_connection usage is guarded by `constexpr` checks.
6. **Semaphore addressing**: Global semaphores have the same address on all devices; virtual core coords match.

### Remaining hypotheses (ranked by likelihood):

**Hypothesis A (MOST LIKELY): MUX `fabric_connection.open()` blocks indefinitely**

The MUX kernel's startup sequence (tt_fabric_mux.cpp line 136-222):
1. Sets status to STARTED
2. Clears memory, builds fabric_connection
3. Calls `fabric_connection.open()` — handshake with the EDM on an ethernet core
4. Sets status to READY_FOR_TRAFFIC

The writer's startup (minimal_default_writer.cpp line 189-193):
1. Calls `wait_for_fabric_endpoint_ready()` — polls MUX status for READY_FOR_TRAFFIC
2. Calls `fabric_client_connect()` — handshake with MUX

If `fabric_connection.open()` in step 3 blocks (because the EDM channel isn't expecting this TENSIX-based MUX, or the control plane returns stale connection info), the MUX never reaches READY_FOR_TRAFFIC, and the writer spins forever at `wait_for_fabric_endpoint_ready()`. The writer is a programmed core, so it never sends its done signal, and K is never reached.

The `FabricMuxConfig` in the test does NOT call `set_wait_for_fabric_endpoint_ready(true)` (default=false), so the MUX does not check if the EDM is ready before attempting the handshake. If the EDM's connection slot for this MUX isn't properly initialized, the NOC read in `open_start()` could read stale/zero data, and the handshake might spin or proceed with garbage state.

**Hypothesis B: Fabric packet loss / misrouting**

If the `append_fabric_connection_rt_args()` call generates incorrect routing info for the MUX-to-EDM path, fabric packets (data or semaphore increments) could be dropped or misrouted. A reader waiting for a semaphore increment that never arrives would hang forever.

This is less likely than Hypothesis A because it would be a systemic fabric issue affecting many tests, not just this one.

**Hypothesis C: EDM backpressure causing starvation**

With `num_buffers_per_channel = 1` in the MUX, the MUX can only hold one packet at a time. If the EDM is slow to consume packets (e.g., due to ethernet link congestion), the MUX's `wait_for_empty_write_slot()` blocks. However, this is backpressure (not deadlock) and should eventually resolve — unless there's a credit-based deadlock in the EDM layer.

---

## Recommended Investigation Path

1. **Runtime debugging**: Attach a debugger or add print statements to the MUX kernel to verify:
   - Does the MUX reach `fabric_connection.open()`?
   - Does it complete `open()` and set READY_FOR_TRAFFIC?
   - Does the writer's `wait_for_fabric_endpoint_ready()` eventually see READY_FOR_TRAFFIC?

2. **Verify EDM connection setup**: Check that `append_fabric_connection_rt_args()` returns valid EDM channel info for each MUX. Specifically verify the EDM noc_x, noc_y, and channel handshake addresses are correct for a TENSIX-based MUX (vs the typical ETH-based EDM).

3. **Test with `wait_for_fabric_endpoint_ready=true`**: Call `mux_config.set_wait_for_fabric_endpoint_ready(true)` in the test. This would make the MUX wait for the EDM to signal READY before opening its connection. If this fixes the hang, it confirms Hypothesis A.

4. **Check `generic_op` program dispatch order**: Verify that the MeshWorkload dispatch correctly handles 4 heterogeneous programs. Specifically, ensure the go signal is sent to each device after its program commands, not before.

5. **Add watcher/inspector breakpoints**: Use the tt-metal watcher to check which cores are stuck and in what state after a timeout.
