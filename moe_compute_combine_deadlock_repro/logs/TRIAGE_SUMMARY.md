# tt-triage of the fused moe_compute combine hang — 2026-06-23

Captured with the **host process alive** (smoke parked at the post-`moe_compute`
`synchronize_device`) and tt-triage run **in parallel** against the live runtime.

- Container: `tt-xla-ird-mvasiljev`; venv `/home/mvasiljev/tt-xla/venv`.
- Runtime tt-metal: `.../tt-mlir/.../tt-metal/src/tt-metal` @ commit
  `68e82deb155ec50633cbb505d33a5a014cf678e2` (short `68e82deb155`, dated 2026-05-29; ".so"
  compiled locally 2026-06-23 13:18). This is the model's runtime build, NOT the workspace build-tree.
- Config: mesh (4,8), `cluster_axis=0`, COL dispatch, `FABRIC_1D_RING`, GLM dims
  (160 experts, 5/dev, H=5120, N=1536, K=8, bf4), `SMOKE_PINPOINT=1`.
- Raw logs: `livehang_smoke_20260623_162834.txt`, `livehang_triage_20260623_162834.txt`.

## Host signal
`moe_compute ok: returned 6 combine (8, 16, 5120)` then the immediate
`synchronize_device` never returns (no `SMOKE SYNC ... OK`). Reproduced 2/2 (random inputs).

## tt-triage device signal (authoritative)
- `dump_op_window`: **op 9 = `MoEComputeDeviceOperation` is the only op still `RUNNING`**,
  on devices **16, 20, 24, 28**. Ops 1-8 (tilize/typecast/untilize/`AllToAllDispatchMetadata`/
  `Full`) all completed -> the device executed everything before the fused combine and hangs
  only inside it.
- **Same 4 devices every run** (4/4 captures: random seed, seed 1234, seed 4242, and after a
  full container+galaxy reset). Topology-determined, not input-determined.
- Device 24 worker BRISC cores (logical (5,y)/(6,y)) report **"Failed to halt"** = wedged.

## Where the 4 stuck devices sit in the mesh (probe_mesh.py)
Physical device IDs are scrambled by galaxy wiring (NOT row-major), so the raw numbers are
misleading. Enumerating the opened (4,8) mesh gives:

```
R\C   C0  C1  C2  C3  C4  C5  C6  C7
R0    24  25  26  27   3   2   1   0
R1    28  29  30  31   7   6   5   4
R2    20  21  22  23  15  14  13  12
R3    16  17  18  19  11  10   9   8
```

The stuck devices are **column 0, all four rows**: 24->(R0,C0), 28->(R1,C0), 20->(R2,C0),
16->(R3,C0). Two equivalent readings:
- **dispatch view** (`cluster_axis=0` ring = a column): the 4 stuck = **one whole dispatch
  ring** = one CCL plane (col 0).
- **replicate view** (the barrier axis `replicate_axis` = axis 1 = the 8 columns; a replicate
  group = a row): the stuck device is the **column-0 member of *every* one of the 4 rows** —
  i.e. in each replicate group, exactly the col-0 "tail" is left waiting; the other 7 columns
  drained.

## Stuck-core callstacks (the deadlock)
Distinct parked frames inside the fused op:

| #cores | frame |
|---|---|
| 24 | `selective_reduce_combine/.../writer.cpp:352` -> `noc_semaphore_wait()` |
| 32 | `selective_reduce_combine/.../writer.cpp:270` (per-expert consume loop) |
| 24 | `moe_compute/.../dm1.cpp:359` |
| 24 | `moe_compute/.../dm0.cpp:397` |
| 24 | `moe_compute/.../compute.cpp:346/386/405` |

**Root frame:** `writer.cpp:352`

```cpp
// sync core only, AFTER fabric_multicast_bidirectional_atomic_inc_ring_1d<...,replicate_axis,...>
noc_semaphore_wait(semaphore_ptr, replicate_group_devices);   // <-- parked here
```

The barrier runs **along `replicate_axis`** (for `cluster_axis=0`, that is axis 1 = the 8
columns; `replicate_group_devices` = 8). The combine **sync cores** finish, fire the ring
atomic-inc around their replicate group (row), then wait for the global semaphore to reach
`replicate_group_devices`. In **each** of the 4 rows, only the **column-0 device** is left
parked here — its replicate ring never delivers the full increment count. Meanwhile non-sync
combine cores are mid-loop at `writer.cpp:270` and the matmul cores (dm0/dm1/compute) wait on
their producer/consumer semaphores -> circular wait, op never drains. This is a **ring-barrier
participant/expected-count mismatch**, reproducible to the same column-0 tails every run.

## Axis comparison (same build, same harness)
Parametrized the harness by `SMOKE_CLUSTER_AXIS` (derives ring size, tokens/dev, replicate
group, epilogue axis):

| dispatch ring | replicate ring (barrier axis) | result |
|---|---|---|
| `cluster_axis=0` — 4-device col ring | axis 1 = **8 columns** | **HANG** at `writer.cpp:352` |
| `cluster_axis=1` — 8-device col ring | axis 0 = **4 rows** | **PASS** (combine drains + full epilogue) |

So the hang correlates with the **8-wide replicate ring barrier** (axis-0 dispatch); the
4-wide replicate ring (axis-1 dispatch) drains end-to-end. Caveat: holding global batch=64
couples ring size with tokens/dev (16 vs 8); a tokens/dev-matched control is the one remaining
isolation experiment.

## Robustness of the repro
- **Input-independent:** identical signature for random inputs, seed 1234, seed 4242.
- **Reset-independent:** still hangs after **container restart + `tt-smi -glx_reset_auto`
  immediately before** the run (same 16/20/24/28, dev24 fail-to-halt, 12 cores at 352).
- **`tt-smi -r` is insufficient** for the galaxy fabric (intermittent `topology_mapper`
  "target node not mapped" at `open_mesh_device`); `glx_reset_auto` retrains ethernet -> use it.

## Corroborates AutoDebug Finding #2
`writer.cpp` final barrier is topology-compiled: Ring waits `replicate_group_devices` (N),
Linear waits `N-1`. The fused path resolves topology internally
(`moe_compute_device_operation.cpp`), and `topology` is omitted from
`SelectiveReduceCombineParams::attributes()` (program-cache key). A Ring/Linear mismatch at
this exact `noc_semaphore_wait` matches the parked state observed here.

## tt-triage caveats
- `tt-exalens` is 0.3.13 (required 0.3.19); ran with `--skip-version-check`. `dump_op_mesh`
  and `device_info` errored on missing UMD attrs (`get_chip_pci_bdfs`, `get_tray_id`), so the
  mesh-coord->device-id grid above was recovered separately via `probe_mesh.py` — unrelated to the hang.
- Earlier first attempt got no callstacks because the smoke was killed by `timeout`
  (Inspector needs the live process). Re-run kept it alive -> full callstacks (102 ELFs).
