# Metal 2.0 host-API gaps blocking the TTNN op migration — for Audrey

**From:** Diego (TTNN op migration)
**Context:** Migrating a batch of TTNN ops to the Metal 2.0 `ProgramSpec`/`create_program_spec` path
(rand, matmul, transpose, pad, tilize, untilize_with_unpadding, slice, reshard, binary_ng, conv2d,
halo, pool). 13 factories across 9 ops are ported and PCC-correct; the items below are the metal-side
(host-API / validator / kernel-codegen) gaps that block the rest. Everything else remaining is TTNN-side
work on our end.

All references are against the branch state I built/validated on (WH B0).

---

## P0 — the one that gates *production*: in-place run-args update

**`ProgramRunArgsView` / `GetProgramRunArgsView` is declared but not implemented**
`tt_metal/api/tt-metalium/experimental/metal2_host_api/program.hpp:82` ("Sketch only; not yet implemented").

**Why it matters.** With the spec path, the only correct cache-hit options today are (a) re-run the
factory + `SetProgramRunArgs` (we do this for ops with dynamic run-args like rand's seed), or (b)
`UpdateTensorArgs` (tensors only). Both rebuild per-core, string-keyed `ProgramRunArgs` every dispatch.
Measured host-cost regression vs the legacy descriptor path, 1000 cache-hit calls, identical op calls,
spec-factory-confirmed-live:

| op | descriptor | spec | ratio |
|---|---|---|---|
| transpose WH | 18.6 µs | 85.0 µs | **4.6×** |
| transpose CN | 18.8 µs | 69.1 µs | 3.7× |
| rand 512² | 48 µs | 81 µs | 1.7× |
| pad / untilize | ~19–37 µs | ~40–76 µs | ~2.1× |
| slice / binary_ng | ~27–48 µs | ~43–74 µs | ~1.6× |
| matmul (device-bound) | 84.9 µs | 87.7 µs | ~1.0× |

The cost is the per-core string-keyed run-args rebuild + re-apply. Profiling rand's hit path: ~30 µs
rebuilding run-args that are a pure function of the cache key, <1 µs of genuinely-dynamic values. An
in-place poke (`ProgramRunArgsView`: overwrite the changed ints at resolved byte offsets, no Table
rebuild, no re-resolution) collapses this toward the descriptor path's ~5 µs. **This is the difference
between "migration is host-perf-neutral" and "migration ships a 1.5–4.6× host regression on light ops."**

Related: a relaxation-aware **`std::hash<ProgramSpec>`** (spec-as-cache-key) would also help — we measured
the current key over-fragmenting 7 cache entries where 2 suffice (same-volume shapes). I prototyped a
ttnn-side spec hash for measurement; a canonical one belongs in tt_metal.

---

## P1 — small, unblocks two ops immediately

**Borrowed-DFB memory check rejects `L1_SMALL`**
`tt_metal/impl/metal2_host_api/program_spec.cpp:1265`:
```cpp
TT_FATAL(tensor_spec.memory_config().buffer_type() == tt::tt_metal::BufferType::L1, ...);
```
conv2d and halo back their borrowed config DFBs with `L1_SMALL` buffers (`sliding_window.cpp` config
tensors). The validator only accepts `BufferType::L1`. **Suggested fix:** accept `{L1, L1_SMALL}`
(both are L1-resident). I have a local one-line prototype (relaxed to a two-way check) — needs your
review/blessing since it's your validator, and confirmation that `AttachBorrowedDFBBuffers`
(`program_run_args.cpp`) handles an `L1_SMALL` backing buffer's per-bank sizing. **Blocks:** conv2d
(L1 path), halo.

---

## P1 — `TensorAccessor` binding can't express runtime base-offset / per-shard page-size

`tt_metal/hw/inc/api/tensor/tensor_accessor.h`: the binding-token ctor (`TensorAccessor(ta::name)`,
~`:99-110`, dynamic twin ~`:390-396`) hard-seats `bank_base_address` from the injected CRTA and
`aligned_page_size` from the spec's static `AlignedPageSize` (`tensor_accessor_args.h:44`). The flexible
3-arg ctor `TensorAccessor(args, base, page_size)` (~`:82-87`) is unreachable through `ta::`.

**Use case (slice ROW_MAJOR, sharded path):** the legacy kernel (i) bakes a host-computed offset into the
accessor base (`start_addr + begins_bytes - misalignment`) then hands the *accessor object* to
`noc_async_read_sharded` (which derives NoC addresses internally), and (ii) passes a per-shard page size
(`shard_W * elem_size`) read back via `get_aligned_page_size()`. Neither knob is reachable via the typed
binding, and the recipe's "host-computed offset" pattern only covers kernels that use a *raw* base+offset,
not ones that hand the accessor to `noc_async_*_sharded`. **Suggested:** let a `ta::` binding optionally
carry a runtime page-size and/or a host-computed base offset (route to the 3-arg ctor). **Blocks:**
slice RM (and likely other sharded RM data-movement ops).

---

## P2 — same-core multi-producer DFB (broadcast)

`tt_metal/impl/metal2_host_api/program_spec.cpp` local-DFB validation (~`:1137-1153`, `TT_FATAL` ~`:1143`;
self-loop exception ~`:1182-1197`) enforces pairwise-disjoint producers per node.

**Use case (halo):** the split-reader writes one output CB from **both** RISCV_0 and RISCV_1 on the
**same** core — two PRODUCER bindings on one DFB, same node, same kernel-kind. Not a self-loop, so the
self-loop exception doesn't apply. This is the broadcast/same-core multi-endpoint case your stacked metal
prereq (PR #47037) targets. **Blocks:** halo. **Suggested:** the broadcast DFB model from #47037, or an
explicit "multiple same-kind producers on identical node coverage" allowance.

---

## P2 — buffer-address binding (address through a CTA to a DM kernel)

conv2d and pool, on their **DRAM-config** paths, thread `buffer->address()` (+ `page_size`) through
compile-time args into a data-movement kernel (e.g. conv `conv_reader_common.hpp` `load_config_tensor_if_in_dram`;
pool reader CTAs). Metal 2.0 disallows raw addresses through CTAs/RTAs, and a borrowed DFB requires L1
(so a DRAM-resident config can't borrow). There is no sanctioned way to hand a DM kernel a DRAM buffer's
base address. **Suggested:** a host-API `BufferParameter`/address-binding, or a DRAM-backed borrowed DFB.
**Blocks:** conv2d (DRAM-config path), pool (DRAM-config path). (Both ops' L1 paths are unblocked once
P1-L1_SMALL lands + our TTNN op-owned fix.)

---

## P3 — GlobalCircularBuffer DFB binding

matmul `MatmulMultiCoreReuseMcast1DProgramFactory` uses a `GlobalCircularBuffer` (`global_cb`) path with
no corresponding DFB binding in the host API. **Blocks:** matmul Mcast1D (one of several matmul mcast
factories). Lower priority — the 2D mcast factory has no such gap (just large), and we have MultiCore +
ReuseOptimized ported.

---

## Design feedback on op-owned tensors (FYI, not a blocker for you)

I built the ttnn-side `ProgramArtifacts::op_owned_tensors` channel + cache-hit policy (parked tensors,
reuse-not-realloc on hit). Two notes from the first real consumers vs the documented design:
- The docs describe op-owned tensors as **empty scratch** via `MeshTensor::allocate_on_device`
  (fully-replicated, "untested at runtime"). conv2d/pool need **host-populated** config tensors and
  **sharded** placement. We can handle this TTNN-side (carry a populated `ttnn::Tensor`), but the
  documented `allocate_on_device(... fully_replicated ...)` recipe should be widened or annotated.
- `MeshTensor` being non-copyable + `Tensor::mesh_tensor()` returning `const&` means there's no clean
  `Tensor`→owned-`MeshTensor` move; we'll add a ttnn extraction helper. Flagging in case you'd prefer a
  blessed API.

---

## Priority summary
1. **`ProgramRunArgsView`** — gates production (removes the host regression). Biggest item.
2. **L1_SMALL borrowed-DFB relax** — one line, unblocks conv2d + halo (L1). Prototype ready for review.
3. **`ta::` runtime base-offset / page-size** — unblocks slice RM (+ sharded RM family).
4. **Same-core multi-producer DFB (broadcast, #47037)** — unblocks halo.
5. **Buffer-address binding / DRAM-borrowed-DFB** — unblocks conv2d + pool DRAM paths.
6. **GlobalCB DFB binding** — unblocks matmul Mcast1D.

Everything else remaining (multi-program concept for matmul MeshWorkload, host-populated/sharded op-owned
tensors, binary_ng's kernel-path fan-out, the remaining per-op default factories) is TTNN-side and on us.
