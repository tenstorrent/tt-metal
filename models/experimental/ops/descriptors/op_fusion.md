# Sequential Fusion Architecture

## Overview

Sequential fusion combines multiple operations into a single fused kernel that
runs as one program dispatch. Instead of launching each op as a separate
host-to-device round-trip, all ops execute back-to-back within a single
long-running kernel on each core. Intermediate results stay in L1 circular
buffers (CBs) -- no DRAM round-trips between fused phases.

The fusion tree is a standard tree of `OpNode` objects. Each node holds one
operation (`OpDescriptor`). Parent-to-child edges encode sequential ordering
(parent runs before child). Sibling nodes run in parallel on disjoint core
subsets. A linear chain is a tree with branching factor 1.


## Glossary

### Node (`OpNode`)

An `OpNode` holds a single `OpDescriptor` and an optional list of children.
The node's core range is derived from its op's `ProgramDescriptor` kernels
(no separate `core_range` field). Every node in the tree -- root, internal,
and leaf -- has exactly one op.

### Root

The root is the topmost node of the tree. It is a regular `OpNode` like any
other -- no special treatment. The root's op runs first, before any children.

### Leaf

A leaf is a terminal node with no children. Leaves are the endpoints of the
tree. Each root-to-leaf path produces one fused kernel binary.

### Internal Node

An internal node has one or more children. Its op runs before its children's
ops. When an internal node has multiple children, those children run in
parallel on disjoint core subsets.

### Segment

A segment is a contiguous portion of a root-to-leaf path where the barrier
scope (set of participating cores) remains constant. Consecutive nodes with
the same core range are grouped into one segment. Each segment has its own
`arrive`/`release` `GlobalSemaphore` pair for cross-core synchronization.

For example, in a tree where the root and one intermediate node share the same
core range (0-7), but the leaf uses cores 0-3, the path has two segments:
segment 0 covers cores 0-7 (root + intermediate phases), segment 1 covers
cores 0-3 (leaf phase).


## Tree Model

```
root: op0 (cores 0-7)
  node A: op1 (cores 0-3)
    leaf A1: op2 (cores 0-1)
    leaf A2: op3 (cores 2-3)
  leaf B: op4 (cores 4-7)
```

A linear chain is simply a tree with branching factor 1:

```
root: op0 (cores 0-3)
  node: op1 (cores 0-3)
    leaf: op2 (cores 0-3)
```

### Topology Rules

- **Sibling disjointness**: Children of the same parent must have
  non-overlapping core ranges. Each core runs exactly one kernel binary.
- **Child subset of parent**: A child's core range must be a subset of its
  parent's range.
- **Partial coverage allowed**: Children don't need to fully tile their parent.
  Uncovered cores don't participate in child phases.

Topology is validated by `_validate_topology()` before any device allocation.


## Path Tracing and Kernel Generation

### One Kernel Binary Per Root-to-Leaf Path

Each root-to-leaf path through the tree produces one fused kernel binary. The
binary contains all phases from root to leaf, concatenated. It runs on the
leaf's core range for the entire execution.

For the example tree above, three paths are traced:

```
Path 1: [(cores 0-7, [op0]), (cores 0-3, [op1]), (cores 0-1, [op2])]
Path 2: [(cores 0-7, [op0]), (cores 0-3, [op1]), (cores 2-3, [op3])]
Path 3: [(cores 0-7, [op0]), (cores 4-7, [op4])]
```

Consecutive nodes with the same core range are grouped into a single segment.
Internal nodes use their *effective leaf range* (union of descendant leaf core
ranges) as the barrier scope, not the node's own declared range. This ensures
barrier scope matches the cores that actually participate.

### Fused Kernel Structure

Each path's fused kernel binary contains three RISC-specific kernels:

- **RISCV_0 (Reader/BRISC)**: Reads data from DRAM/L1 into input CBs.
  Coordinates all inter-phase cleanup. Runs the cross-core NOC barrier.
- **Compute (TRISC)**: Processes tiles from input CBs, writes to output CBs.
- **RISCV_1 (Writer/NCRISC)**: Writes output CB data to DRAM/L1.

All three run concurrently on each core. Each contains all phases for the
path, with barrier synchronization code between phases.

```c++
// Example: fused reader kernel for a 2-phase path
void kernel_main() {
    phase0_reader();          // root op

    // === inter-phase barrier ===
    noc_async_full_barrier();
    noc_semaphore_wait_min(__compute_done, 1);
    noc_semaphore_wait_min(__writer_done, 1);
    __cb_reset_to_empty();
    __barrier_seg0(0, arrive, release);

    phase1_reader();          // child op
}
```

### Self-Contained Build Output

All paths from a tree share barrier semaphore addresses. The builder merges
all path ProgramDescriptors internally via `merge_program_descriptors()` and
returns a single `OpDescriptor`. This makes the result self-contained --
dispatching it via `composite.launch([result])` dispatches all paths as one
program. There is no way to accidentally dispatch a subset.


## CB Pool Allocator

The CB pool allocator (`CBPoolAllocator` in `cb_allocator.py`) is responsible
for mapping each fused phase's original CB indices to a shared set of hardware
CB slots (0-31). Without remapping, two phases that both use CB 0 for different
purposes (different data formats, page sizes) would corrupt each other's data.
With remapping, compatible CBs share a slot and incompatible CBs get separate
slots.


### The Problem

Each unfused op uses its own set of CB indices, assigned by its C++ factory.
A LayerNorm might use CBs `{0, 1, 2, 3, 4, 5, 24, 25}`. A matmul might use
CBs `{0, 1, 4, 5}`. When fused into a two-phase kernel:

- CB 0 in the LN (BFloat16, page_size=2048) and CB 0 in the matmul
  (BFloat16, page_size=2048) are compatible — same format, same page size.
  They can share hardware slot 0.
- CB 5 in the LN (BFloat16, page_size=2048) and CB 5 in the matmul
  (Float32, page_size=4096) are incompatible — different format and page
  size. They need separate hardware slots.

The allocator produces a **remap table** per phase: `{orig_cb_index: hw_slot}`.
The remap is applied to named compile-time args (so kernel code references
the correct slot) and to the merged `CBDescriptor` list (so hardware is
configured correctly).

The device has exactly 32 CB hardware slots. Exceeding this is a hard error.


### Compatibility Key (`CBPoolKey`)

Two CBs can share a hardware slot if and only if they have the same
`CBPoolKey`:

```python
@dataclass(frozen=True)
class CBPoolKey:
    data_format: Any    # tt::DataFormat (BFloat16, Float32, etc.)
    page_size: int      # Bytes per tile page
    has_buffer: bool    # True if backed by an L1 Buffer allocation
    unpack_to_dest_mode: Any  # Default or UnpackToDestFp32
```

**Why `has_buffer` matters**: Some CBs are backed by pre-allocated L1 buffers
(e.g., sharded input tensors). Their L1 address is fixed by the tensor
allocation, not by the CB configuration. Sharing a buffer-backed CB slot with
a non-buffer CB would cause the non-buffer phase to read/write at the buffer's
fixed address instead of the CB's normal FIFO address. Keeping `has_buffer` in
the key prevents this.

**Why `unpack_to_dest_mode` matters**: The unpack hardware mode
(`UnpackToDestMode`) is configured per CB slot in the compute kernel's
`unpack_to_dest_mode` vector (32 entries, one per slot). If two phases share a
slot but disagree on the mode, one phase gets the wrong unpack behavior. The
allocator avoids this by making the mode part of the compatibility key.


### Allocation Algorithm

#### Phase-by-Phase Processing

The allocator processes phases sequentially. For each phase, it allocates
slots for that phase's CBs. The key invariant:

- **Within a phase**: Every CB gets its own slot, even if two CBs in the
  same phase have identical `CBPoolKey`s. They hold different data
  concurrently.
- **Across phases**: CBs with matching keys can share a slot. Only one
  phase runs at a time, so the slot is reused.

```python
def allocate_phase(self, phase_idx, cb_info, phantom_cb_indices):
    remap = {}
    slots_used_this_phase = set()

    # 1. Reserve phantom CBs (identity mapping)
    # 2. Allocate non-aliased CBs (two-pass: identity first, then remaining)
    # 3. Allocate aliased CBs (reuse existing alias group or fresh slots)

    self.phase_remaps.append(remap)
```

#### Two-Pass Allocation (Non-Aliased CBs)

Non-aliased CBs (the common case — one `CBFormatDescriptor` per
`CBDescriptor`) are allocated in two passes:

**Pass 1 — Identity matches**: CBs whose original index already has a
compatible slot from a prior phase, where that slot was created from the same
original index. These are allocated first to maximize index stability:

```python
def _partition_by_identity(self, cb_info):
    identity_cbs = []
    remaining_cbs = []
    for orig_idx, info in sorted(cb_info.items()):
        key = info.pool_key
        has_identity = False
        if key in self._config_to_slots:
            for candidate_idx in self._config_to_slots[key]:
                if self._slot_to_orig_index.get(candidate_idx) == orig_idx:
                    has_identity = True
                    break
        if has_identity:
            identity_cbs.append((orig_idx, info, key))
        else:
            remaining_cbs.append((orig_idx, info, key))
    return identity_cbs, remaining_cbs
```

**Pass 2 — Remaining CBs**: For each CB, search for any compatible slot not
yet used this phase. If found, reuse it. If not, allocate a fresh slot.

**Why identity-first matters**: Without this ordering, a non-identity-matched
CB could claim a slot that another CB needs for identity matching, forcing the
second CB to a new slot unnecessarily. This wastes slots and increases the risk
of hitting the 32-slot limit. More importantly, identity mapping keeps CB
indices stable across phases, which matters for cross-group consistency in
branching trees (see Forced Remaps below).

#### Slot Reuse vs. Fresh Allocation

When searching for a reusable slot, the allocator prefers identity matches:

```python
def _find_reusable_slot(self, key, orig_idx, slots_used_this_phase):
    if key not in self._config_to_slots:
        return None
    # First: identity match (same original CB index created this slot)
    for candidate_idx in self._config_to_slots[key]:
        if candidate_idx not in slots_used_this_phase:
            if self._slot_to_orig_index.get(candidate_idx) == orig_idx:
                return candidate_idx
    # Second: any compatible slot not used this phase
    for candidate_idx in self._config_to_slots[key]:
        if candidate_idx not in slots_used_this_phase:
            return candidate_idx
    return None
```

When allocating a fresh slot, the allocator prefers identity mapping (placing
the CB at its original hardware index):

```python
def _allocate_new_slot(self, key, info, orig_idx, phase_idx):
    if orig_idx not in self._allocated_indices and orig_idx < self.max_slots:
        slot_idx = orig_idx      # Prefer: CB 5 → slot 5
    else:
        slot_idx = self._alloc_index()  # Fallback: next free slot
    ...
```

When reusing a slot, only `total_size` is updated (to the max across phases).
The `source_cb` and `source_fmt` references are kept from the first allocating
phase — this is important for `build_merged_cb_descriptors`, which uses these
references to construct the merged CB descriptor.


### Alias Groups

Some ops use **aliased CBs** — a single `CBDescriptor` with multiple
`CBFormatDescriptor` entries. For example, matmul's output uses one
`CBDescriptor` with two format descriptors at indices 4 and 5 (`c_out` and
`c_intermed0`). These share the same L1 allocation; the hardware uses the
same memory region but interprets it differently depending on which CB index
is referenced.

The allocator must preserve this aliasing relationship. If phase 0's matmul
has CB 4 and CB 5 aliased, and phase 1 reuses slot 4 for one purpose and
slot 5 for another (independently), the merged CBDescriptor would force both
slots into a single L1 allocation — corrupting the phase that expected them
to be independent.

**Detection**: Each `CBInfo` carries an `alias_group` field set to the
`CBDescriptor`'s position in the program's `cbs` list. CBs with the same
`alias_group` share a `CBDescriptor` and thus share L1.

**Allocation rules**:

1. Aliased CBs are allocated separately from non-aliased CBs.
2. The allocator first tries to reuse an **existing alias group** from a prior
   phase — a set of slots that were previously allocated together as an alias
   group. Reuse requires: same number of members, each member slot compatible
   with a current CB, no member slot used this phase.
3. Matching uses permutation search (trying all orderings of the existing
   group's slots against the current phase's aliased CBs), since the CBs may
   appear in a different order:

```python
def _match_alias_members(self, members, group_slots, cb_info):
    member_keys = [(orig_idx, cb_info[orig_idx].pool_key) for orig_idx in members]
    for perm in itertools.permutations(group_slots):
        result = []
        valid = True
        for (orig_idx, key), slot_idx in zip(member_keys, perm):
            slot = self._slots.get(slot_idx)
            if slot is None or slot.config != key:
                valid = False
                break
            result.append((orig_idx, slot_idx))
        if valid:
            return result
    return None
```

4. If no existing group matches, all members get fresh slots, and the new
   group is recorded for future phases to reuse.

**Why permutation search is safe**: Alias groups have 2-3 members in practice
(e.g., matmul's `c_out`/`c_intermed0`). The permutation count is trivial
(2! = 2, 3! = 6).


### Phantom CBs

C++ op factories sometimes create named compile-time args for CB indices
(e.g., `("cb_bias", 18)`) even when no `CBDescriptor` exists for that index.
This happens when the op's configuration doesn't use a particular code path
(e.g., bias is absent), but the factory still emits the named arg.

**The risk**: Without knowledge of phantom index 18, the allocator might
assign a real CB to slot 18. The kernel code would then have two CBs at the
same hardware slot — one real (allocated by the pool) and one phantom
(referenced in a dead code path). If the dead code path isn't perfectly dead
(e.g., a branch that reads the CB index but doesn't access it), the collision
could cause incorrect behavior.

**The fix**: Before allocating a phase, `_get_phantom_cb_indices` scans all
kernel named compile-time args for `cb_`-prefixed entries whose values don't
appear in the phase's `cb_info`. These phantom indices get identity-mapped
reservations (`remap[18] = 18`).

```python
def _get_phantom_cb_indices(phase):
    real_cb_indices = set(phase.cb_info.keys())
    phantom = set()
    for kernel_desc in phase.op_descriptor.descriptor.kernels:
        for name, value in kernel_desc.named_compile_time_args:
            if _is_cb_named_arg(name, value) and value not in real_cb_indices:
                phantom.add(value)
    return phantom
```

**Phantom CBs do NOT block slot reuse**: They are added to
`_allocated_indices` (preventing the pool from assigning that index to a
*new* slot) but NOT to `slots_used_this_phase` (allowing an existing slot at
that index to be reused by a real CB in the same phase). This is safe because
the phantom's code path is dead at runtime.

**Phantom CBs are NOT added to `_slots`**: This means they are excluded from
per-phase CB reset arrays, `build_merged_cb_descriptors`, and the
`unpack_to_dest_mode` vector. No hardware configuration is emitted for them.


### GlobalCB Remote Indices

`GlobalCircularBuffer`-backed CBs have a dual-index model: a local
`format_descriptor` (pool-allocated normally) and a `remote_format_descriptor`
(managed by the GlobalCB firmware, not by stream registers).

Remote indices must be **reserved** to prevent collisions but must NOT be:
- Pool-allocated (no `CBSlot` created)
- Remapped (no entry in `phase_remaps`)
- Included in inter-phase CB reset (remote CBs use L1-based tracking,
  not stream registers — resetting them would corrupt the GlobalCB state)

```python
def reserve_index(self, index):
    self._allocated_indices.add(index)
```

This is called before phase allocation:

```python
for phase in phases:
    for remote_idx in _extract_remote_cb_indices(phase.op_descriptor.descriptor):
        pool.reserve_index(remote_idx)
```


### Forced Remaps (Cross-Group Consistency)

In branching trees, an op may appear in multiple groups (e.g., block-sharded
LN where the mcast sender is in group 0 and the receiver is in group 1). Each
group builds its own `CBPoolAllocator` independently. If the LN's CBs get
different slot assignments in each group, multicast writes from one group's
cores would hit the wrong L1 address on the other group's cores.

**Solution**: Before building any group, `_compute_shared_cb_remaps()` in
`graph.py` identifies ops that appear in multiple groups (by Python object
identity). It builds a single **reference pool** from all shared ops and
records the resulting remap for each:

```python
# Build reference pool from all shared ops (first occurrence order)
ref_pool = CBPoolAllocator(max_slots=32)
for phase_idx, pi in enumerate(ref_phase_infos):
    phantom_indices = _get_phantom_cb_indices(pi)
    ref_pool.allocate_phase(phase_idx, pi.cb_info, phantom_indices)

# Record: op identity → reference remap
ref_remaps = {}
for ref_idx, op in enumerate(shared_ops_ordered):
    ref_remaps[id(op)] = ref_pool.get_remap(ref_idx)
```

Each group then receives a `forced_phase_remaps` dict. When the group's
allocator encounters a forced phase, it calls `force_phase_remap()` instead
of `allocate_phase()`:

```python
def force_phase_remap(self, phase_idx, cb_info, forced_remap):
    for orig_idx, slot_idx in forced_remap.items():
        self._allocated_indices.add(slot_idx)
        info = cb_info.get(orig_idx)
        if info is None:
            continue  # Phantom — just reserve
        # Register slot as if freshly allocated
        ...
    self.phase_remaps.append(dict(forced_remap))
```

This replays the reference allocation, ensuring all groups assign the same
slots to shared ops. Subsequent non-shared phases allocate normally, aware of
the forced slots.

**L1 address equalization**: Even with matching slot indices, non-shared
phases can cause different `total_size` values for the same slot across groups
(the max-across-phases differs per group). Since CB L1 addresses are allocated
sequentially by slot index, a size difference at any slot shifts all
subsequent addresses. `_equalize_cb_sizes()` runs after all groups are built,
padding each slot's `total_size` to the cross-group maximum.


### Merged CB Descriptors

After allocation, `build_merged_cb_descriptors()` constructs the final
`CBDescriptor` list for the fused kernel. This is non-trivial because:

1. **Alias groups must be preserved**: Slots that share an L1 allocation
   (from aliased CBs) must be emitted as a single `CBDescriptor` with
   multiple `format_descriptors`. The method uses `_compute_slot_alias_groups`
   to determine which slots belong together.

2. **New CBDescriptors are constructed**: The method never emits original
   CBDescriptor objects directly. It creates new `ttnn.CBDescriptor()` objects
   per alias group, setting `total_size` to the max across all member slots,
   `core_ranges` from the representative slot, and `format_descriptors` from
   each member's `source_fmt`.

3. **Buffer-backed slots**: If any member of an alias group is buffer-backed
   (has an L1 Buffer allocation), the merged CBDescriptor inherits the buffer
   via `set_buffer_from_cb()`. The buffer source is taken from the earliest
   phase that has a buffer-backed CB in the group, matching the rebind logic
   which computes address diffs relative to phase 0.

4. **Mutation contract**: The method mutates `source_fmt.buffer_index` on the
   original `CBFormatDescriptor` C++ objects (setting them to the remapped
   slot index). Callers must bracket the build with `_save_cb_state()` /
   `_restore_cb_state()` to revert these mutations when building multiple
   groups from the same ops.


### CB Address Rebinding

When a buffer-backed CB is remapped to a slot that had a different buffer
address in the previous phase, the CB's L1 FIFO pointers must be updated
at runtime. `_compute_rebind_info()` compares each phase's buffer addresses
against the previous phase's and emits `(slot_idx, address, size)` tuples
for slots that changed.

At runtime, the barrier's `reset()` function applies rebinds between phases:

```cpp
template <size_t N>
void rebind_cbs(const std::array<uint32_t, N>& slots, uint32_t rt_start) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t slot = slots[i];
        uint32_t addr = get_arg_val<uint32_t>(rebind_rt_offset + rt_start + i * 2);
        uint32_t size = get_arg_val<uint32_t>(rebind_rt_offset + rt_start + i * 2 + 1);
        get_local_cb_interface(slot).fifo_rd_ptr = addr >> cb_addr_shift;
        get_local_cb_interface(slot).fifo_wr_ptr = addr >> cb_addr_shift;
        get_local_cb_interface(slot).fifo_size = size >> cb_addr_shift;
        get_local_cb_interface(slot).fifo_limit = (addr + size) >> cb_addr_shift;
    }
}
```

The addresses and sizes are passed as runtime args (not compile-time args)
because buffer addresses are determined by tensor allocation, which varies
across executions.


### CB State Save/Restore

`build_merged_cb_descriptors()` mutates `buffer_index`, `total_size`, and
`core_ranges` on original C++ `CBDescriptor` objects. Python's `deepcopy`
cannot pickle these C++ bindings, so the only option is in-place mutation
with save/restore:

```python
saved = _save_cb_state(program_descriptors)
try:
    # ... build fused descriptor (mutates CBDescriptors) ...
finally:
    _restore_cb_state(saved)
    _verify_cb_restore(saved)
```

This is critical for branching trees where the same stem op's CBDescriptors
are used by multiple group builds. Without restore, the second group would
see the first group's mutated `buffer_index` values.


## Circular Buffer Management

### Output-to-Input Chaining

Phase N's output CB becomes Phase N+1's input CB. The output data stays in L1;
no DRAM round-trip is needed.

### CB State Reset Between Phases

After each phase, all CB state must be reset to empty before the next phase can
use them. This is complex because four RISC processors independently track CB
state:

| RISC | Tracks | Reset Action |
|------|--------|-------------|
| BRISC | `tiles_acked`, `tiles_received` (via stream registers), `fifo_rd_ptr`, `fifo_wr_ptr` | Equalize stream registers (per-tile increment loop), reset pointers to CB start |
| TRISC0 (unpack) | `tiles_acked` (local copy), `fifo_rd_ptr` | Sync from stream register via `reg_read`, reset pointer |
| TRISC2 (pack) | `tiles_received` (local copy), `fifo_wr_ptr`, `fifo_wr_tile_ptr` | Sync from stream register via `reg_read`, reset pointer and tile pointer |
| NCRISC | `fifo_rd_ptr`, `fifo_wr_ptr` | Reset pointers to CB start (reads stream registers directly) |

**Ordering**: BRISC reset runs first (it owns the stream registers). Then the
cross-core barrier fires. After the barrier, compute and writer resync their
local copies from the (now-equalized) stream registers.

**Per-tile increment**: The stream controller requires per-tile increments when
equalizing `tiles_acked` to match `tiles_received`. Bulk increments
(`acked += N` where N > 1) hang the hardware. The code uses a `for` loop that
increments by 1 each iteration.

### CB State Save/Restore

`_build_fused_descriptor()` mutates `buffer_index`, `total_size`, and
`core_ranges` in-place on original `CBDescriptor` C++ objects (Python's
`deepcopy` cannot pickle them). To prevent corruption when building multiple
paths that share ops (e.g., root ops), `_save_cb_state()` /
`_restore_cb_state()` snapshot and restore these fields around each path build.

### Phantom CB Handling

C++ op factories may create named compile-time args for CB indices (e.g.,
`cb_ex=18`) even when no corresponding `CBDescriptor` exists. These "phantom
CBs" get identity mappings during pool allocation to prevent collisions.


## Synchronization Protocol

### Two-Level Barrier

Phase synchronization uses a two-level protocol:

1. **Local RISC sync** (per-core, L1 flags): Reader waits for compute and
   writer to finish the current phase before resetting CBs.
2. **Cross-core NOC barrier** (across cores, GlobalSemaphore): All cores in the
   barrier scope must complete before any core proceeds to the next phase.

Both levels use monotonically increasing counters -- never reset during kernel
execution.

### Local RISC Synchronization

Each core has two L1 flags allocated via `GlobalSemaphore`:

- `compute_done`: Compute writes `phase_idx + 1` after completing a phase.
- `writer_done`: Writer writes `phase_idx + 1` after completing a phase.

The reader waits for both before proceeding with CB reset.

### Cross-Core NOC Barrier

After local RISC sync and CB reset, the reader executes a cross-core barrier.
One designated core ("core 0") acts as the coordinator. `arrive` and `release`
are `GlobalSemaphore` L1 words. `arrive` accumulates on core 0 via NOC atomic
increments. `release` is multicast from core 0 to all cores in the bounding
box. Both are monotonic.

### Multi-Segment Barriers

When a path transitions between tree segments (e.g., wider scope to narrower),
the barrier scope changes. Each segment has its own `arrive`/`release`
`GlobalSemaphore` pair.

A `MultiBarrierSpec` maps each phase transition to a barrier segment. Paths
that share a tree segment MUST use identical `arrive`/`release` L1 addresses.
A `segment_cache` keyed by `frozenset(core_ranges)` ensures this.


## Code Generation

Fusion generates a single C++ source file per RISC type (reader, compute,
writer) per fused kernel binary. Each generated file follows the same
structure and is compiled by the JIT build system. This section defines every
part of the generated file and describes how original kernel sources are
decomposed, transformed, and reassembled.


### Terminology

**Original kernel source**: The `.cpp` file from a single unfused op (e.g.
`dataflow/reader_unary_interleaved.cpp`). These files follow a standard
layout:

```c++
#include <header.h>             // ← includes
#define SOME_MACRO 42           // ← source defines

namespace N { ... }             // ← pre-main: shared block
void helper() { ... }           // ← pre-main: phase-specific block
constexpr int X = 1;            // ← pre-main: phase-specific block

void kernel_main() {            // ← kernel body
    uint32_t addr = get_arg_val<uint32_t>(0);
    helper();
}
```

**Includes**: `#include` directives at the top of the file. Collected,
deduplicated, and emitted once in the fused output.

**Source defines**: `#define` directives that appear in the source file before
`kernel_main()` (distinct from the per-kernel defines injected by the C++
build system via command-line flags). Collected, deduplicated, and emitted once
in the fused output.

**Pre-main code**: All top-level declarations and definitions between the
includes/defines and `kernel_main()`. This is where helper functions, global
variables, namespace blocks, `using` declarations, `typedef`s, struct
definitions, and preprocessor conditional blocks (`#ifdef`/`#if`/`#ifndef`)
live. Pre-main code is categorized into **shared** blocks (deduplicated) and
**phase-specific** blocks (prefixed per-phase).

**Kernel body**: The contents of `kernel_main()` — everything between its
opening and closing braces, not including the braces themselves.

**Phase function**: A `FORCE_INLINE` wrapper that contains one phase's
transformed kernel body. Named `phaseN_reader()`, `phaseN_writer()`, or
`phaseN_compute()`.

**Generated `kernel_main()`**: The outer dispatch function that calls phase
functions in sequence with inter-phase barrier code between them.


### Source Parsing (`cpp_parser` Module)

All C++ source decomposition uses tree-sitter AST parsing via the
`cpp_parser` module. No regex-based parsing is used for structural
decomposition (regex is only used for targeted compile-time arg rewriting).

#### `extract_kernel_body(source)`

Finds the `kernel_main` function definition in the AST, extracts the body
node, and returns its inner text (without outer braces).

#### `categorize_pre_main(source)`

Walks all top-level AST children before `kernel_main()` and classifies each
into a `PreMainBlock` with a semantic `kind`:

| Kind | AST Node Type | Examples |
|------|--------------|---------|
| `"function"` | `function_definition`, `template_declaration` containing function | `FORCE_INLINE void helper() { ... }`, `template<typename T> void f() { ... }` |
| `"variable"` | `declaration` (non-function) | `constexpr uint32_t X = 1;`, `static int buf[4];` |
| `"namespace"` | `namespace_definition` | `namespace MATH { ... }` |
| `"using"` | `using_declaration`, `alias_declaration`, `type_alias_declaration` | `using uint32_t = ...;` |
| `"namespace_alias"` | `namespace_alias_definition` | `namespace ckernel = ...;` |
| `"struct"` | `struct_specifier`, `class_specifier`, `enum_specifier` | `struct Foo { ... };` |
| `"preproc_block"` | `preproc_ifdef`, `preproc_if`, `preproc_ifndef` | `#ifdef FP32 ... #endif` |
| `"other"` | everything else | Rare: forward declarations, etc. |

For `preproc_block`, it also extracts `inner_names`: the function and variable
names defined inside the conditional block, so they can be prefixed.

#### `inline_local_includes(source, kernel_dir)`

Resolves `#include "local.h"` (quoted, not angle-bracket) by reading the
referenced file relative to `kernel_dir` and inlining its contents in place.
Strips `#pragma once` from inlined content. Recursive for nested local
includes.

This is necessary because fused kernels are emitted as `SOURCE_CODE` type
(source string, not file path). The JIT compiler doesn't know the original
op's kernel directory, so local includes would fail to resolve.

#### `replace_in_code_only(source, old_name, new_name)`

Word-boundary replacement that skips string literals, comments, and other
non-code tokens. Uses tree-sitter to identify non-code byte ranges, then
applies regex replacement only in code regions. This prevents false matches
like renaming `helper` inside a string `"helper"`.

#### `collect_includes(sources)` / `collect_defines(sources)`

Collect and deduplicate `#include` and `#define` lines across multiple source
files. `collect_defines` uses tree-sitter to find the `kernel_main` line
number and only collects defines before that point.

#### `normalize_block(block)`

Collapses whitespace and removes blank lines for content comparison during
deduplication.


### Pre-Main Categorization and Isolation

`_collect_all_pre_main_code()` takes all phases' parsed sources and sorts
every pre-main block into either **shared** or **phase-specific**:

**Shared** (deduplicated, emitted once at file scope):
- Namespace blocks: deduplicated by signature (text before `{`). First
  occurrence wins.
- Everything else that isn't a function, variable, or preproc block:
  `using`, `namespace_alias`, `struct`, `typedef`, `template` (non-function).
  Deduplicated by normalized content.

**Phase-specific** (prefixed, emitted per-phase):
- Free function definitions: the function name is prefixed with `phaseN_`.
  Example: `void helper()` becomes `void phase0_helper()`.
- Global/static variables: the variable name is prefixed.
  Example: `constexpr uint32_t X = 1;` becomes
  `constexpr uint32_t phase0_X = 1;`.
- Preprocessor conditional blocks: the block text is preserved as-is, but
  all inner function/variable names are prefixed.

ALL phases get prefixed, including phase 0. This prevents:
- Silent first-wins drops when two phases define the same function with
  different bodies (C++ one-definition rule).
- Redefinition errors between inlined header code and phase code.

The returned `phase_names` dict records which original names were prefixed
per phase. These same names are then prefixed in the kernel body (see
Phase Body Transformation below).


### Preprocessor Define Handling

Preprocessor defines injected by the C++ build system (not from source files)
are classified by `_categorize_phase_defines()`:

**Uniform defines**: Same `(name, value)` pair in ALL phases. Emitted once
as `#define` at the top of the generated file. Example: `#define REDUCE_OP 0`
when all phases use the same reduction operation.

**Varying defines**: Present in only some phases, or with different values
across phases. Emitted as `#define`/`#undef` pairs scoped to each phase's
pre-main code and phase function body. The C++ preprocessor resolves `#ifdef`
blocks within each phase's scope according to that phase's define state.

**MUST_MATCH defines**: `REDUCE_OP`, `REDUCE_DIM`, `BCAST_LLKOP`,
`BCAST_DIM`. These are consumed by LLK headers at include time and must be
identical across all phases. Validated at fusion time; mismatches raise an
error. Treated as uniform after validation.

The `#define`/`#undef` scoping pattern:

```c++
// Varying define: SOME_FLAG is 1 in phase 0, absent in phase 1

#define SOME_FLAG 1
// phase 0 pre-main code (helper functions see SOME_FLAG=1)
#undef SOME_FLAG

// phase 1 pre-main code (helper functions see SOME_FLAG undefined)

#define SOME_FLAG 1
FORCE_INLINE void phase0_reader() {
    // kernel body: #ifdef SOME_FLAG is true here
}
#undef SOME_FLAG

FORCE_INLINE void phase1_reader() {
    // kernel body: #ifdef SOME_FLAG is false here
}
```

This replaces any need for source-level `#ifdef` resolution. The compiler's
preprocessor handles it naturally.


### Phase Body Transformation

Each phase's kernel body goes through `_transform_phase_source()`, which
applies three transformations in order:

#### 1. Named compile-time arg prefixing

Phase 0 keeps original names. Phase N > 0 gets `phaseN_` prefix:

```
get_named_compile_time_arg_val("blk")
→ get_named_compile_time_arg_val("phase1_blk")
```

Uses regex on the string argument inside the function call.

#### 2. Positional compile-time arg offsetting

Positional `get_compile_time_arg_val(N)` indices are offset by the
cumulative count of prior phases' compile-time args. Also handles
`TensorAccessorArgs<N>` which expands to `get_compile_time_arg_val`:

```
get_compile_time_arg_val(5)
→ get_compile_time_arg_val(15)        // offset = 10

TensorAccessorArgs<0>
→ TensorAccessorArgs<10>              // same offset
```

Uses regex on the numeric argument.

#### 3. Phase name prefixing

All names from the `phase_names` dict (functions and globals that were
prefixed in pre-main) are also prefixed in the kernel body. Uses
`replace_in_code_only()` to avoid replacing inside strings/comments:

```
helper();         → phase0_helper();
X + 1             → phase0_X + 1
```


### Runtime Arg Offsetting

Runtime args (`get_arg_val<T>(idx)`) are set by the host at dispatch time
(tensor addresses, tile counts, semaphore addresses). They cannot be
substituted as compile-time constants. During fusion, each phase's runtime
args are concatenated per-core into a single flat array. Phase N's args start
at an offset equal to the cumulative count of all prior phases' args.

The offsetting uses a **wrapper + `#define` redirect** pattern instead of
source-level rewriting:

**Step 1**: A wrapper function is emitted at file scope (before any
`#define` redirect is active), so its body references the real `get_arg_val`:

```c++
template <typename T>
FORCE_INLINE T __phase1_get_arg_val(int arg_idx) {
    return get_arg_val<T>(arg_idx + 10);  // offset for phase 1
}
```

**Step 2**: A `#define` redirects all `get_arg_val` tokens to the wrapper
within the scope of each phase's pre-main code and phase function:

```c++
#define get_arg_val __phase1_get_arg_val
FORCE_INLINE void phase1_reader() {
    uint32_t addr = get_arg_val<uint32_t>(0);  // reads arg[10]
}
#undef get_arg_val
```

This catches ALL `get_arg_val` calls in scope — including calls in helper
functions defined in pre-main, calls in inlined header code, calls with
variable indices like `get_arg_val<uint32_t>(arg_idx++)`, and calls produced
by macro expansion. The wrapper is `FORCE_INLINE` with a constant offset, so
it compiles to a single add-immediate instruction.

Phase 0 uses no wrapper (offset is 0; it calls the real `get_arg_val`
directly).


### Generated File Structure

The three RISC-specific generators (`_generate_fused_riscv0_source`,
`_generate_fused_riscv1_source`, `_generate_fused_compute_source`) all emit
files with the same layered structure. Below is the complete layout for a
3-phase reader kernel. Writer and compute kernels follow the same pattern
with RISC-specific barrier behavior (described in the next section).

```c++
// =====================================================================
// Section 1: License Header
// =====================================================================
// SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Auto-generated fused reader kernel - 3 phases


// =====================================================================
// Section 2: Uniform Defines
// =====================================================================
// Defines identical across all phases. Emitted once.
#define REDUCE_OP 0
#define REDUCE_DIM 0


// =====================================================================
// Section 3: Source Defines
// =====================================================================
// #define lines collected from original source files (before kernel_main).
#define BIT_SET(x, b) ((x) | (1 << (b)))


// =====================================================================
// Section 4: Includes
// =====================================================================
// Deduplicated, sorted #include lines from all phases.
#include <cstdint>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.h"


// =====================================================================
// Section 5: Shared Pre-Main
// =====================================================================
// Deduplicated namespace blocks, using declarations, struct definitions.
namespace MATH {
    // ...
}

using uint32_t = unsigned int;


// =====================================================================
// Section 6: Runtime Arg Wrapper Functions
// =====================================================================
// Emitted at file scope BEFORE any #define redirect, so the wrapper
// body references the real get_arg_val.
template <typename T>
FORCE_INLINE T __phase1_get_arg_val(int arg_idx) {
    return get_arg_val<T>(arg_idx + 5);
}

template <typename T>
FORCE_INLINE T __phase2_get_arg_val(int arg_idx) {
    return get_arg_val<T>(arg_idx + 11);
}


// =====================================================================
// Section 7: Per-Phase Pre-Main Code
// =====================================================================
// Each phase's helper functions and globals, prefixed with phaseN_.
// Wrapped with that phase's varying #define/#undef and RT arg redirect.

// -- Phase 0 (no RT arg redirect needed; offset is 0) --
constexpr uint32_t phase0_TILE_SIZE = 2048;

FORCE_INLINE void phase0_read_tiles(uint32_t addr) {
    uint32_t n = get_arg_val<uint32_t>(3);  // real get_arg_val
    // ...
}

// -- Phase 1 --
#define get_arg_val __phase1_get_arg_val
#define SOME_VARYING_FLAG 1
constexpr uint32_t phase1_TILE_SIZE = 4096;

FORCE_INLINE void phase1_read_tiles(uint32_t addr) {
    uint32_t n = get_arg_val<uint32_t>(0);  // → __phase1_get_arg_val(0)
    // ...                                     → reads arg[5]
}
#undef SOME_VARYING_FLAG
#undef get_arg_val

// -- Phase 2 --
#define get_arg_val __phase2_get_arg_val
constexpr uint32_t phase2_TILE_SIZE = 2048;

FORCE_INLINE void phase2_read_tiles(uint32_t addr) {
    uint32_t n = get_arg_val<uint32_t>(0);  // → reads arg[11]
    // ...
}
#undef get_arg_val


// =====================================================================
// Section 8: Infrastructure Functions (reader/BRISC only, multi-phase)
// =====================================================================

// Named compile-time args for barrier configuration
constexpr uint32_t __barrier_rt_offset =
    get_named_compile_time_arg_val("barrier_rt_offset");
constexpr uint32_t __seg0_num_cores =
    get_named_compile_time_arg_val("seg0_num_cores");
// ... more per-segment constants (core0 coords, mcast bounds)

// CB reset: equalize stream registers + reset BRISC local pointers
FORCE_INLINE void __cb_reset_to_empty() {
    // For each CB in sweep list:
    {
        uint16_t remaining = *get_cb_tiles_received_ptr(cb)
                           - *get_cb_tiles_acked_ptr(cb);
        volatile tt_reg_ptr uint32_t* acked_ptr = ...;
        for (uint16_t i = 0; i < remaining; i++) {
            acked_ptr[0] += 1;       // per-tile increment (bulk hangs HW)
        }
        uint32_t fifo_start = cb.fifo_limit - cb.fifo_size;
        cb.fifo_rd_ptr = fifo_start;
        cb.fifo_wr_ptr = fifo_start;
    }
}

// Per-segment NOC barrier
FORCE_INLINE void __barrier_seg0(uint32_t call_idx,
    volatile tt_l1_ptr uint32_t* arrive,
    volatile tt_l1_ptr uint32_t* release) {
    if constexpr (__seg0_num_cores > 1) {
        noc_semaphore_inc(core0_arrive_noc_addr, 1);      // all cores
        if (is_core_0) {
            noc_semaphore_wait_min(arrive, N*(call_idx+1)); // wait all
            *release = call_idx + 1;                        // set release
            noc_semaphore_set_multicast_loopback_src(...);  // broadcast
            noc_async_write_barrier();
        } else {
            noc_semaphore_wait_min(release, call_idx + 1);  // wait
        }
    } else {
        *release = call_idx + 1;  // single-core: no NOC needed
    }
}


// =====================================================================
// Section 9: Phase Functions
// =====================================================================
// Each phase's kernel body wrapped in a FORCE_INLINE function.
// Varying defines and RT arg redirect are scoped per-function.

// Phase 0 reader (no redirect)
FORCE_INLINE void phase0_reader() {
    // Transformed kernel body:
    //   - Named CT args: original names (phase 0 keeps them)
    //   - Positional CT args: original indices
    //   - RT args: original get_arg_val (offset = 0)
    //   - Helper calls: phase0_read_tiles(...)
    uint32_t addr = get_arg_val<uint32_t>(0);       // reads arg[0]
    phase0_read_tiles(addr);
}

// Phase 1 reader (with redirect + varying defines)
#define get_arg_val __phase1_get_arg_val
FORCE_INLINE void phase1_reader() {
    #define SOME_VARYING_FLAG 1
    // Transformed kernel body:
    //   - Named CT args: "phase1_blk", "phase1_cb_in", etc.
    //   - Positional CT args: offset by phase 0's count
    //   - RT args: redirected to __phase1_get_arg_val
    //   - Helper calls: phase1_read_tiles(...)
    uint32_t addr = get_arg_val<uint32_t>(0);       // reads arg[5]
    phase1_read_tiles(addr);
    #undef SOME_VARYING_FLAG
}
#undef get_arg_val

// Phase 2 reader (with redirect)
#define get_arg_val __phase2_get_arg_val
FORCE_INLINE void phase2_reader() {
    uint32_t addr = get_arg_val<uint32_t>(0);       // reads arg[11]
    phase2_read_tiles(addr);
}
#undef get_arg_val


// =====================================================================
// Section 10: Generated kernel_main()
// =====================================================================
void kernel_main() {
    // Read barrier L1 flag addresses from concatenated runtime args
    const uint32_t __compute_done_addr =
        get_arg_val<uint32_t>(__barrier_rt_offset);
    const uint32_t __writer_done_addr =
        get_arg_val<uint32_t>(__barrier_rt_offset + 1);
    volatile tt_l1_ptr uint32_t* __compute_done = ...;
    volatile tt_l1_ptr uint32_t* __writer_done = ...;

    // Per-segment arrive/release pointers
    volatile tt_l1_ptr uint32_t* __seg0_arrive = ...;
    volatile tt_l1_ptr uint32_t* __seg0_release = ...;

    // ---- Phase 0 ----
    phase0_reader();

    // ==== Barrier: Phase 0 → Phase 1 ====
    noc_async_full_barrier();                  // drain NOC queue
    noc_semaphore_wait_min(__compute_done, 1); // wait compute done
    noc_semaphore_wait_min(__writer_done, 1);  // wait writer done
    __cb_reset_to_empty();                     // equalize streams, reset ptrs
    // Reset op semaphores to initial values
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        get_semaphore(sem_id)) = initial_val;
    // Rebind CB addresses for phase 1
    {
        constexpr uint32_t new_addr =
            get_named_compile_time_arg_val("phase1_cb0_rebind_addr");
        constexpr uint32_t new_size =
            get_named_compile_time_arg_val("phase1_cb0_rebind_size");
        get_local_cb_interface(0).fifo_rd_ptr = new_addr;
        get_local_cb_interface(0).fifo_wr_ptr = new_addr;
        get_local_cb_interface(0).fifo_size = new_size;
        get_local_cb_interface(0).fifo_limit = new_addr + new_size;
    }
    __barrier_seg0(0, __seg0_arrive, __seg0_release);  // cross-core sync

    // ---- Phase 1 ----
    phase1_reader();

    // ==== Barrier: Phase 1 → Phase 2 ====
    noc_async_full_barrier();
    noc_semaphore_wait_min(__compute_done, 2);
    noc_semaphore_wait_min(__writer_done, 2);
    __cb_reset_to_empty();
    // ... rebind, barrier ...

    // ---- Phase 2 ----
    phase2_reader();
}
```


### RISC-Specific Barrier Behavior

The three RISC types play different roles in inter-phase synchronization.
All three are generated with the same sectional structure (sections 1-7, 9
above), but sections 8 and 10 differ:

#### Reader / BRISC (`_generate_fused_riscv0_source`)

The reader is the **barrier coordinator**. Between phases it:

1. Drains the NOC queue (`noc_async_full_barrier()`)
2. Waits for local compute and writer to finish
   (`noc_semaphore_wait_min` on `compute_done` and `writer_done`)
3. Resets CB state (`__cb_reset_to_empty()`) — equalizes stream registers
   via per-tile increment loop, resets `fifo_rd_ptr`/`fifo_wr_ptr` to CB
   start
4. Resets op semaphores to initial values
5. Rebinds CB addresses for the next phase (if CB pool remapped addresses)
6. Runs the cross-core NOC barrier (`__barrier_segN()`)

#### Writer / NCRISC (`_generate_fused_riscv1_source`)

Between phases the writer:

1. Drains NOC writes (`noc_async_write_barrier()`)
2. Signals done by writing `phase_idx + 1` to `writer_done` L1 flag
3. Spins on `global_release` (`while (*release < N)`) — plain volatile read,
   no NOC APIs
4. Resyncs NCRISC local CB pointers (`__resync_ncrisc_cb_state()`) — resets
   `fifo_rd_ptr`/`fifo_wr_ptr` to CB start
5. Rebinds CB addresses for the next phase

The NCRISC resync function only resets local pointers. It does not touch
stream registers (BRISC owns those).

#### Compute / TRISC (`_generate_fused_compute_source`)

Between phases compute:

1. Signals done by writing `phase_idx + 1` to `compute_done` L1 flag
2. Spins on `global_release` (`while (*release < N)`)
3. Resyncs compute-side CB state (`__resync_cb_state_after_sweep()`)

The compute resync is split by TRISC instance using `#ifdef`:

- **TRISC0 (unpack)**: Reads `tiles_acked` from stream register via
  `reg_read()`, stores to local `tiles_acked`. Resets `fifo_rd_ptr` to CB
  start.
- **TRISC1 (math)**: No CB resync needed (no cb_interface on TRISC1).
  Rebind code is guarded with `#ifndef TRISC_MATH`.
- **TRISC2 (pack)**: Reads `tiles_received` from stream register via
  `reg_read()`, stores to local `tiles_received`. Resets `fifo_wr_ptr` to
  CB start and `fifo_wr_tile_ptr` to 0.

### CB Rebinding

When the CB pool allocator remaps a CB to a different physical slot between
phases, the L1 address and buffer size change. Rebind code is emitted in
`kernel_main()` between the CB reset and the global barrier. It updates
`fifo_rd_ptr`, `fifo_wr_ptr`, `fifo_size`, and `fifo_limit` on the
`local_cb_interface` struct.

For compute kernels, rebind addresses are right-shifted by 4 (`>> 4`)
because TRISC addresses use 16-byte units. This shift is guarded with
`#ifndef TRISC_MATH` since TRISC1 has no `cb_interface`.

The rebind addresses and sizes are passed as named compile-time args:
`phaseN_cbM_rebind_addr` and `phaseN_cbM_rebind_size`.


### Code Generation Pipeline Summary

For each fused kernel binary (one per root-to-leaf path), the code
generation pipeline runs independently for each RISC type:

```
1. Read original sources     Read each phase's .cpp file, inline local
                             #include "..." directives

2. Collect & dedup           Gather #include lines, source #define lines
                             across all phases. Deduplicate.

3. Categorize defines        Split build-system defines into uniform
                             (same in all phases) vs varying (differ).
                             Validate MUST_MATCH defines.

4. Categorize pre-main       Parse each phase's pre-main with tree-sitter.
                             Split into shared (namespace, using) and
                             phase-specific (functions, variables, preproc
                             blocks). Prefix phase-specific names with
                             phaseN_.

5. Transform phase bodies    For each phase's kernel body:
                             - Prefix named CT args (phase N>0)
                             - Offset positional CT args
                             - Prefix phase-specific function/global names

6. Emit sections 1-7         License, uniform defines, source defines,
                             includes, shared pre-main, RT arg wrappers,
                             per-phase pre-main with #define/#undef scoping

7. Emit section 8            RISC-specific infrastructure functions
                             (CB reset, barrier, CB resync)

8. Emit section 9            Phase functions with per-phase #define/#undef
                             scoping and RT arg redirect

9. Emit section 10           Generated kernel_main() with phase calls and
                             inter-phase barrier orchestration
```


## Public API

### `Sequential`

Fuses a sequence of ops into a single dispatch. Items can be `OpDescriptor`,
`Sequential`, or `Parallel` objects. Nested `Sequential` items are flattened.

`build()` returns a `FusedOp` — a self-contained result that holds the fused
`ProgramDescriptor`, input/output tensors, and `GlobalSemaphore` references.
Device is auto-extracted from the input tensors (or can be passed explicitly).

```python
# Linear chain
fused = Sequential(op0, op1, op2).build()
composite.launch([fused])

# Incremental construction
s = Sequential(op0)
s.add(op1).add(op2)
fused = s.build()

# Composition via nesting (flattened automatically)
stem = Sequential(op0, op1)
full = Sequential(stem, op2).build()  # equivalent to op0 -> op1 -> op2
```

### `Parallel`

Items that run in parallel on disjoint core subsets. Requires at least 2 items.

```python
# As part of a Sequential (branching tree)
fused = Sequential(stem_op, Parallel(branch_a, branch_b)).build()
composite.launch([fused])
# fused.output_tensors[0] = branch_a output
# fused.output_tensors[1] = branch_b output

# Standalone (independent ops merged into one dispatch)
fused = Parallel(op_a, op_b).build()
composite.launch([fused])
```

### Nested splits

`Parallel` items can contain `Sequential` chains, which can themselves
contain further `Parallel` splits:

```python
fused = Sequential(
    stem,
    Parallel(
        Sequential(op_a, Parallel(op_a1, op_a2)),
        op_b,
    ),
).build()
```

### `FusedOp` vs `OpDescriptor`

- **`OpDescriptor`**: Simple `(descriptor, input_tensors, output_tensors)` triple.
  Used as input to `Sequential`/`Parallel`.
- **`FusedOp`**: Result of `build()`. Same public fields plus `semaphores`
  (keeps `GlobalSemaphore` refs alive). **Cannot be nested** in
  `Sequential`/`Parallel` — `_resolve()` rejects it with a `TypeError`.

### Rules

- All three types (`OpDescriptor`, `Sequential`, `Parallel`) are
  interchangeable as items. `FusedOp` is **not** accepted as an item.
- Both `Sequential` and `Parallel` have `.add()` for incremental building
  (returns `self` for chaining).
- Items after a `Parallel` in a `Sequential` raise an error at build time
  (the tree diverges and can't rejoin).
- `build()` should only be called once on the outermost container.

### `composite.launch(op_descriptors)`

Merges and dispatches multiple `OpDescriptor` or `FusedOp` objects as a
single program.

### Implementation details

`OpNode`, `OpGraphBuilder`, and `build_op_graph` are internal implementation
details used by `Sequential` and `Parallel`. They remain exported for
backward compatibility but are not part of the primary interface.


## Constraints and Limitations

- **fp32_dest_acc_en must match across all phases**: `DST_ACCUM_MODE` is a
  compile-time constant. Cannot change mid-kernel.
- **All phases must have all three kernel types** (reader, compute, writer).
- **32 CB slot limit**: Multi-phase fusion with many phantom CBs may exhaust
  the 32-slot hardware limit.
- **C++ binding objects cannot be deepcopied**: The save/restore mechanism
  works around this by saving individual field values.
- **MUST_MATCH defines must be consistent**: `REDUCE_OP`, `REDUCE_DIM`,
  `BCAST_LLKOP`, `BCAST_DIM` must have identical values across all fused
  phases.

### Multicast Ops as Stem in Branching Trees

Ops that perform inter-core communication (NOC multicast, semaphore-based
reductions) require special handling as stem phases in branching trees. This
specifically affects sharded LayerNorm, which multicasts partial statistics
(mean, variance) across cores to compute global normalization parameters.
Linear chains are unaffected (single binary, single CB pool).

#### The Challenge

In a branching tree, each root-to-leaf path produces a separate fused kernel
binary. The stem phase appears in every binary, but each binary only runs on
its leaf's core subset:

```
Stem: sharded LN (cores 0-3)
  Branch A: RMS (cores 0-1)  → Binary A runs on cores 0-1
  Branch B: RMS (cores 2-3)  → Binary B runs on cores 2-3
```

Each binary has its own `CBPoolAllocator` instance. Without coordination, two
problems arise:

1. **CB index divergence**: Different phantom CBs or reserved indices across
   binaries can cause the same original CB to land at different hardware
   slots. A multicast targeting a CB address on a remote core would hit the
   wrong slot.

2. **CB L1 address divergence**: Even with matching slot indices, different
   `total_size` values (from different branch ops' max-across-phases) shift
   L1 addresses for subsequent slots. A multicast to a fixed L1 offset would
   land at the wrong address.

#### Solution: Forced Remaps + Size Equalization

Both problems are addressed by two mechanisms (see CB Pool Allocator section):

- **`_compute_shared_cb_remaps()`** builds a single reference
  `CBPoolAllocator` from all shared ops and forces each group to replay the
  reference allocation via `force_phase_remap()`. This guarantees identical
  CB slot assignments across all binaries for shared phases.

- **`_equalize_cb_sizes()`** pads each slot's `total_size` to the cross-group
  maximum after all groups are built. This guarantees identical L1 layouts
  across all binaries.

Together, these ensure that every core — regardless of which binary it
runs — sees the same CB hardware indices and the same L1 buffer addresses for
shared (stem) phases. Multicast writes land at the correct location.


## File Map

| File | Purpose |
|------|---------|
| `models/experimental/ops/descriptors/sequential.py` | All fusion logic: `OpGraphBuilder`, `OpNode`, CB pool allocation, kernel code generation, barrier protocol |
| `models/experimental/ops/descriptors/cpp_parser.py` | Tree-sitter based C++ source parsing |
| `models/experimental/ops/descriptors/op_descriptor.py` | `OpDescriptor` namedtuple |
| `models/experimental/ops/descriptors/composite.py` | `launch()`: merges and dispatches multiple OpDescriptors |
| `tests/ttnn/unit_tests/operations/fused/test_sequential.py` | Device tests (require hardware) |
| `tests/ttnn/unit_tests/operations/fused/test_sequential_standalone.py` | Standalone tests (mock ttnn, no hardware) |
