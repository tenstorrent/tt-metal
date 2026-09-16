# Descriptor Rebuild Inside override_runtime_arguments

## Description

`override_runtime_arguments()` runs on **every program-cache hit**. Calling
`create_descriptor()` (or `apply_descriptor_runtime_args()`) from inside it
therefore pays the full cache-*miss* host cost on every *hit*:
`split_work_to_cores`, `CoreRangeSet` construction, arch queries for the compute
config, `TensorAccessorArgs`, kernel-source strings, compile-time-arg vectors, a
freshly heap-allocated runtime-arg vector for every core, and then a walk over
every kernel x core x arg and every CB — all to update two or three scalars.

It is described internally as *the single most expensive mistake on the
descriptor path*, and it is strictly slower than writing the changed slots in
place. It is the same cliff that produced the ResNet50 (20x) and resnet-T3K
(760%) performance regressions.

It is tempting precisely because it looks *safe*: re-running the source of truth
seems to guarantee the re-applied args cannot drift from the built layout. The
correct way to get that guarantee for free is to single-source the work split —
put the core split and per-core layout in one helper called by **both**
`create_descriptor` and the override — so arg indices are structurally unable to
drift, at zero host cost.

There is a mechanical guard, `scripts/detect_override_rebuild.py`, run as the
`detect-override-rebuild` pre-commit hook, with a grandfather list in
`scripts/detect_override_rebuild_baseline.txt`. Seven ops remain on that
baseline: `ccl/mesh_partition`, `data_movement/move`, `data_movement/slice`
(rm_sharded), `data_movement/tilize`, `eltwise/unary`, `moreh/moreh_adam`, and
`moreh/moreh_adamw`. Each baseline line is a live per-dispatch cost, and
deleting a line is the fix.

**Why this rule exists on top of that guard:** the script only scans text
*inside* `override_runtime_arguments` bodies. Factoring the rebuild into a
shared helper — `refresh_args(...)`, `rebuild_and_apply(...)`, a lambda, a
base-class method — hides it completely from the textual scan while costing
exactly the same at runtime. Detecting that requires following the call into the
helper and recognising that the helper does full-rebuild work. That is the
central job of this rule.

## What to Look For

1. **Rebuild hidden behind a helper call (the primary case)**: an
   `override_runtime_arguments()` whose own body is short and clean, but which
   calls a helper that itself invokes `create_descriptor()`,
   `apply_descriptor_runtime_args()`, `split_work_to_cores()`, or otherwise
   reconstructs the descriptor / core layout. Trace every function the override
   calls — one level is rarely enough. The pre-commit guard cannot see through
   this; you can. Treat "the override body looks fine" as the beginning of the
   check, not the end.

2. **Direct rebuild in the override body**: a literal
   `create_descriptor(...)` / `apply_descriptor_runtime_args(...)` call inside
   `override_runtime_arguments`. Only acceptable if the file+symbol is already
   on `scripts/detect_override_rebuild_baseline.txt`, and even then a diff that
   touches that op should be reducing the cost, not entrenching it.

3. **Work-split logic duplicated instead of shared**: the override recomputing
   `split_work_to_cores`, core ranges, or per-core tile counts — even inline
   rather than via `create_descriptor`. This is both the host cost and a drift
   risk. The fix is one shared helper called by the factory and the override.

4. **The no-op early return trap**: an override that returns early when it has
   no hash-excluded scalars to write (a prefill path, for example) also skips
   the **address and CB patching**. `override_runtime_arguments` *supersedes*
   the framework's own binding patching, so if the override returns, nothing
   else refreshes those addresses and they freeze at the first miss. Gate only
   the scalar writes; never gate the address and CB patching. This bit
   `rotary_embedding`'s prefill path.

5. **Matching CBs by position instead of by `CBIndex`**: refreshing a CB address
   via `desc.cbs[7]` rather than by looking up its `CBIndex`. Positions shift
   between descriptor variants — rotary's output CB is `cbs[7]` in the
   single-tile variant and `cbs[8]` in the multi-tile one — so a positional
   match silently patches the wrong buffer.

6. **Incomplete refresh coverage**: because the override supersedes framework
   patching, anything it forgets is frozen at the first miss. Enumerate every
   bound buffer address, every globally-allocated CB base address, and every
   hash-excluded scalar. Values derived purely from *hashed* inputs may be
   skipped — a hit means they are identical — but trace each value to its
   inputs rather than skipping on "looks static". Cover every core the
   descriptor emplaced args for, including zero-work / no-op cores.

7. **In-place and aliased tensors**: `resolve_bindings` bails to *empty*
   bindings when the same buffer appears twice within the **input** region (the
   `matmul(X, X)` ambiguity). An output aliasing an input is safe and keeps the
   fast path — but an op that carries its output inside `tensor_args` as an
   optional `output_tensor` lands in the input region and looks ambiguous,
   silently losing all bindings. Check the
   `allow_inplace_output_tensor_alias` handling before assuming the fast path
   holds. This is the SDXL in-place silu / MorehAdamW class of bug.

8. **Multi-factory ops whose override does not mirror `select_program_factory`**:
   the selection logic must live in one shared helper too, or the override can
   patch against the wrong factory's layout.

## Bad Code Examples

```cpp
// BUG: the canonical anti-pattern — a full descriptor rebuild on every cache
// hit, just to refresh a couple of addresses.
void XDeviceOperation::override_runtime_arguments(
    Program& program, const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args, tensor_return_value_t& out, MeshCoordinate coord) {
    auto desc = XProgramFactory::create_descriptor(attrs, tensor_args, out);  // FULL REBUILD
    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
}
```

```cpp
// BUG: same cost, invisible to scripts/detect_override_rebuild.py because the
// rebuild is one level down in a helper. The override body looks surgical.
static void refresh_args(Program& program, const operation_attributes_t& attrs,
                         const tensor_args_t& tensor_args, tensor_return_value_t& out) {
    auto desc = XProgramFactory::create_descriptor(attrs, tensor_args, out);
    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
}

void XDeviceOperation::override_runtime_arguments(
    Program& program, const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args, tensor_return_value_t& out, MeshCoordinate coord) {
    refresh_args(program, attrs, tensor_args, out);   // hidden full rebuild
}
```

```cpp
// BUG: no-op early return. The prefill path has no scalars to write, so the
// override returns — and thereby also skips the address patching it alone is
// responsible for. The addresses freeze at the first cache miss.
void XDeviceOperation::override_runtime_arguments(...) {
    if (!attrs.is_decode) {
        return;   // also skipped every buffer address and CB refresh
    }
    write_update_idx(program, attrs.update_idx);
    patch_addresses(program, tensor_args);
}
```

```cpp
// BUG: CB matched by position. desc.cbs[7] is the output CB in the
// single-tile descriptor variant but cbs[8] in the multi-tile one, so this
// patches an unrelated buffer half the time.
desc.cbs[7].buffer = out.buffer();
```

```cpp
// BUG: the override recomputes the work split itself. Cheaper than a full
// create_descriptor, still the expensive part, and now the split logic exists
// in two places that can drift apart.
void XDeviceOperation::override_runtime_arguments(...) {
    auto [num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_1, tiles_per_core_2] =
        tt::tt_metal::split_work_to_cores(compute_grid, num_tiles);
    for (const auto& core : corerange_to_cores(all_cores)) { /* ... */ }
}
```

## Good Code Examples

```cpp
// GOOD: the work split lives in ONE helper used by both create_descriptor and
// the override, so arg indices cannot drift — and the override itself only
// re-derives and writes per-dispatch state in place.
static WorkSplit compute_work_split(const operation_attributes_t& attrs, const tensor_args_t& args);

ProgramDescriptor XProgramFactory::create_descriptor(...) {
    const auto split = compute_work_split(attrs, tensor_args);
    // ... build kernels/CBs from `split` ...
}

void XDeviceOperation::override_runtime_arguments(
    Program& program, const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args, tensor_return_value_t& out, MeshCoordinate coord) {
    const auto split = compute_work_split(attrs, tensor_args);   // cheap, no descriptor build
    auto& reader_args = GetRuntimeArgs(program, shared.reader_kernel_id);
    for (const auto& core : split.cores) {
        auto& args = reader_args[core.x][core.y];
        args[shared.src_addr_idx] = tensor_args.input.buffer()->address();
        args[shared.dst_addr_idx] = out.buffer()->address();
        args[shared.update_idx_slot] = attrs.update_idx;
    }
}
```

```cpp
// GOOD: only the scalar write is gated. Addresses and CBs are refreshed
// unconditionally, because nothing else will do it.
void XDeviceOperation::override_runtime_arguments(...) {
    patch_addresses(program, tensor_args, out);   // always
    patch_cb_addresses(program, tensor_args);     // always
    if (attrs.is_decode) {
        write_update_idx(program, attrs.update_idx);
    }
}
```

```cpp
// GOOD: CB looked up by CBIndex, so it survives descriptor variants that
// reorder the cbs vector.
auto it = std::find_if(desc.cbs.begin(), desc.cbs.end(),
                       [](const auto& cb) { return cb.index == CBIndex::c_16; });
TT_FATAL(it != desc.cbs.end(), "output CB c_16 missing from descriptor");
it->buffer = out.buffer();
```
