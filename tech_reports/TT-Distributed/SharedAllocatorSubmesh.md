# Shared-allocator submeshes

## Problem

`MeshDevice::create_submesh` gives the submesh its own allocator on the same chips as the parent. The parent and the
submesh therefore hand out the same addresses: a tensor written through the submesh can land on top of a live parent
tensor (or a kernel binary) on the shared chips. Using a parent and an overlapping submesh in one process, for example
a 4-chip model and a 2-chip model on the same chips, is not safe today.

## API

```cpp
std::shared_ptr<MeshDevice> MeshDevice::create_submesh(
    const MeshShape& submesh_shape,
    const std::optional<MeshCoordinate>& offset = std::nullopt,
    bool share_allocator = false);
```

```python
submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 2), offset=ttnn.MeshCoordinate(0, 0), share_allocator=True)
```

With `share_allocator=True` the submesh places every buffer through the parent's allocator. Nothing changes for
submeshes created without it.

## What it guarantees

- Buffers of the parent and of every sharing submesh never overlap, in DRAM and in L1. The parent's allocator works in
  lock step on all of its chips, so a submesh buffer also reserves its range on the parent's other chips. That wastes
  space on those chips but needs no new allocator.
- A submesh program's circular buffers are checked against the L1 buffers of the whole family (the submesh reports the
  parent's lowest occupied L1 address), so a clash fails at enqueue time, as it does inside one mesh.
- Switching between overlapping meshes is explicit. A command queue of a sharing submesh (or of its parent or a
  sibling) that goes from idle to in use while an overlapping mesh of the family still has a command queue in use
  raises a `TT_FATAL` that names both meshes and says to call `quiesce_devices()`. Disjoint siblings may run at the same
  time. The check only involves sharing submeshes, so existing code that uses overlapping meshes without
  `share_allocator` behaves as before.
- `quiesce_devices()` now also clears each command queue's record of what the prefetcher caches. The prefetcher is per
  chip but that record is per mesh, so after another mesh used the chip (a submesh read was enough), the first mesh's
  next workload replayed a stale cache entry and hung. The record does not depend on the allocator, so this applies to
  overlapping submeshes without `share_allocator` too; the hang was reproduced and fixed with a sharing submesh.
- Closing the parent first closes its sharing submeshes, because they free their buffers through the parent's
  allocator. Submesh buffers that outlive the parent are dropped without being freed.

## What it does not do

- No concurrency. Overlapping meshes still run one after another, with a `quiesce_devices()` between phases. Only
  disjoint submeshes run at the same time.
- No traces on a sharing submesh in v1: `begin_mesh_trace` raises. Trace buffers and their DRAM accounting are per
  mesh, and sharing them with the parent is follow-up work. The parent can still capture and replay traces.
- No sub-device managers or command lists on a sharing submesh (they would need their own allocators).
- No HYBRID allocator mode, no remote-only parents.
- The overlap check assumes one thread drives the overlapping meshes.

## Why a fatal error instead of an automatic quiesce

An automatic quiesce would hide a full drain of both meshes, and a resend of program binaries, inside a call that looks
asynchronous, at a place the caller did not choose. A serving runtime wants to pick its switch points (between decode
steps, not inside one) and to know what each switch costs. The error makes a missing switch visible at the first
wrong enqueue instead of as a hang or corrupted data later.

## Cost of a switch

Measured on one Blackhole P300 board (a (1, 2) parent and a (1, 1) sharing submesh on its second chip), 200 samples
each, medians (`test_switch_cost` in `tests/ttnn/unit_tests/base_functionality/test_shared_allocator_submesh.py`):

| What | Median | p90 |
|---|---|---|
| `synchronize_device` after one small op (the finish every phase boundary needs anyway) | 13.9 us | 15.1 us |
| `quiesce_devices()` with nothing pending | 21.1 us | 43.3 us |
| `quiesce_devices()` right after enqueueing one op | 23.3 us | 24.8 us |
| A step of 16 small ops on the parent, after a parent step, ending in a finish | 289.0 us | 322.9 us |
| The same step right after a submesh step, ending in a quiesce | 302.4 us | 339.0 us |

So a switch costs about 13 us over a back-to-back step (13.4 us here, 13.6 us in a second run): the quiesce itself plus resending the two programs' binaries
that the prefetcher no longer caches. The resend grows with the number of programs a step runs; it was not measured
for a full model. Against a decode step of tens of milliseconds this is well under 1%.

## How a serving runtime uses it

1. Open the parent mesh over all chips and create the smaller model's submesh with `share_allocator=True`.
2. Load both models. Their weights and caches cannot overlap.
3. Run in phases: a decode step (or a batch of steps) of the parent model, `quiesce_devices()`, a step of the submesh
   model, `quiesce_devices()`, and so on. The scheduler decides how many steps each phase gets.
4. Capture traces only for the parent model in v1. The submesh model runs untraced until traces are supported on
   sharing submeshes.
