# Quasar Uplift — Field Notes (extension of the Metal 2.0 op-porting recipe)

**Audience:** engineers driving AI coding agents to get an already-Metal-2.0 op running correctly on the **Quasar** (Gen2) emulator, while keeping it green on **Wormhole (WH)** and **Blackhole (BH)** (Gen1).

> ⚠️ **This recipe makes the agent edit your op's real source files in place.** It is *not* read-only. A GREEN, already-M2 op yields only a `QUASAR_UPLIFT_REPORT.md`; an op that actually needs an uplift gets **in-place edits to its kernels/factories** (plus the report). Protect yourself with **git, not caution about editing** — the workflow (from `READ_ME_FIRST.md`) is: **work on a fresh branch** (never commit directly to `main` or the recipe branch), **one op per workspace**, leave the report **uncommitted for review**, then **review the diff + report and run BH/WH parity + Quasar tests before merging** (delete the report first). If you only want an assessment, ask the agent to run the **audit** and produce the report *without* applying changes.

**This document is an extension, not a standalone recipe.** The authoritative op-porting process lives on branch `akertesz/op-porting-recipe`, under `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`:
- Orientation (humans): `human/READ_ME_FIRST.md`, `human/intro_to_metal_2.0.md`
- Metal 2.0 pre-port audit: `ai/audit/metal2_audit.md` → `METAL2_PREPORT_AUDIT.md` (GREEN/RED) + `METAL2_PORT_BRIEF.md`
- Basic Metal 2.0 port: `ai/port/metal2_port.md`
- Post-port style: `ai/post_port/style/sync_free_dfbs.md` (sync-free DFBs → `Scratchpad`/`LocalTensorAccessor`)
- Post-port semantic: `ai/post_port/semantic/{dm_self_loop_dfbs,gen2_hardware_configs}.md`
- **Quasar-uplift gate:** `ai/audit/quasar_audit.md` — a young scaffold that literally says *"more checks land here."* **This document is that extension** — the field-tested Gen2 gotchas the uplift hits, feeding back into that audit.
- Shared: `ai/shared/{migration_guide,port_patterns,ttnn_factory}.md`

**Two hard rules inherited from the canonical recipe — do not violate them:**
1. **Port/uplift the op IN PLACE — in its existing directory and namespace, in the mainline (`main`) code path** (i.e. the production op, reached via a reviewed PR — *not* a direct commit to `main`, and *not* the `experimental/quasar/` fork). Do **not** copy the op into `ttnn/cpp/ttnn/operations/experimental/quasar/`, and do **not** invent a `ttnn::prim::qsr` / `::experimental::quasar::` namespace. Those `experimental/quasar/*` ops are **deliberately hacky, non-production copies** made to unblock early bring-up; `metal2_port.md` forbids citing, forking, or copying anything from that tree. This doc distills their *lessons* — never their structure.
2. **A Metal 2.0 port makes no functional change.** The Quasar uplift adds only Gen2-specific fixes, each `#ifdef ARCH_QUASAR`-guarded (or behaviorally identical) so **WH/BH take the original path unchanged.** Prove it with the control tests in §9.

**Where this fits in the flow:** Gen1 Metal 2.0 audit → in-place Metal 2.0 port → post-port style fixups → **Quasar-uplift audit (`quasar_audit.md`)** → **Quasar uplift (this doc)**. If the op is already Metal 2.0 on Gen1, you start at the Quasar-uplift audit — and the uplift is often a no-op (see §2).

---

## 1. Workflow (per op)

The base Metal 2.0 port is the canonical recipe's job; follow `metal2_port.md`. What this doc adds is the **Quasar-uplift pass** that runs after it, **in place**:

1. **Confirm the Gen1 Metal 2.0 port is done and green on WH/BH.** The uplift assumes an in-place Metal 2.0 op (factory is `create_program_artifacts` → `ProgramArtifacts`; kernels use `dfb::`/`args::`/`tensor::`/`scratch::`, `get_entry_size()`). If the op is still on the legacy `descriptor` concept, stop and run the Metal 2.0 port first.
2. **Run the Quasar-uplift audit** (`ai/audit/quasar_audit.md`): device-side CB/DFB redesign debt, non-zero-init semaphores, and the growing list of Gen1-legal-but-Gen2-unsupported constructs — plus everything in §7–§12 here.
3. **Build BH → WH → Quasar.** BH/WH must stay unchanged (parity check). Fix Quasar build skew reactively (§8.1).
4. **Bring up on Quasar**, chasing runtime issues (§8.2–§8.5) with the tooling in §9. Apply fixes **in place**, each `ARCH_QUASAR`-guarded.
5. **Guard, don't fork.** Any Quasar fix that would change WH/BH goes behind `ARCH_QUASAR`; a shared kernel that can't convert with its peers gets a `_metal2` fork *alongside the original* (the only sanctioned out-of-op write — see `metal2_port.md` "Porting a shared kernel"), never a copy into `experimental/quasar/`.
6. **Write `QUASAR_UPLIFT_REPORT.md`** (below) and check the **RED-stop conditions**; then §10 for the done checklist.

### Deliverable: `QUASAR_UPLIFT_REPORT.md`

Write it in the op directory, leave it uncommitted for review, and delete it before merge (matching the canonical `METAL2_*` reports). It must state:
- **Status: GREEN or RED** (see below). A "no changes needed" GREEN is a valid, common result for an already-Metal-2.0 op — say so plainly; **do not manufacture changes**.
- **Every file changed**, each with a one-line reason. (The op's namespace and directory never change — flag it loudly if anything tempted you to move or rename.)
- **Which §7–§8 gotchas applied**, and which you considered but did not need (and why — e.g. "no device run yet, so applied only reactively").
- **Deferred / follow-up items**: missing-feature flags for the runtime/LLK team (with the exact symptom), anything that belongs in a separate PR.
- **Parity claim**: confirmation that WH/BH keep the original path (every Quasar change guarded).

### RED status — STOP the uplift and report (do not force it through)

Any one of these means the op is not ready for an in-place Quasar uplift right now. **A RED result is a *success* of the audit — it stops a bad port.** Record the reason in the report and stop:
- **Not Metal 2.0 on Gen1 yet** — factory still `create_descriptor`/`ProgramDescriptor`. Do the Metal 2.0 port first (`ai/port/metal2_port.md`); this doc starts from an already-M2 op.
- **A required capability is missing from the sanctioned Quasar API** — e.g. the DFB ring-rewind `evil_set_*` is Gen1-only (§7). Flag the missing feature; do **not** hand-roll an op-level interface.
- **A construct needs an owner decision** — a non-zero-init semaphore dependency; a DM self-loop / CB redesign the op leans on; an open HW bug on the op's path (e.g. block-sharded fused conv `0x19`).
- **The only apparent fix changes WH/BH un-guarded**, or would require copying into `experimental/quasar/` or a special namespace — both forbidden. Stop rather than do either.
- **An LLK the op needs is a stub / unported** — hand to the LLK team.

---

## 2. In place, no special namespace (and the "already-M2 → often a no-op" reality)

> **The whole point of this reframe:** the uplift edits the op **where it already lives** — `ttnn/cpp/ttnn/operations/<family>/<op>/` — with the op's **existing namespace**. No `experimental/quasar/` clone, no `::qsr`. This matches the device-2.0 / host-2.0 / Metal-2.0 API ports, which all land in place on `main`.

- **If the op is already Metal 2.0 on Gen1, the Quasar uplift is often a no-op.** Run the Quasar-uplift audit; if it's clean, the op may build and pass on Quasar with zero changes, or need only one or two `ARCH_QUASAR` guards. The §7–§8 fixes are **reactive** — apply one only when its symptom actually fires, never pre-emptively.
- **What the uplift may touch (in place):** an `#ifdef ARCH_QUASAR` guard around a Gen1-only construct; a Gen2 `hw_config` variant (`gen2_hardware_configs.md`); a sync-free/self-loop DFB converted to `Scratchpad`/`LocalTensorAccessor` (`sync_free_dfbs.md` / `dm_self_loop_dfbs.md`); removal of a non-zero-init semaphore dependency (`quasar_audit.md`).
- **Shared kernels** bound by factories that don't all convert together: create/reuse a `_metal2` fork **beside the original** + a pointer comment in the legacy file (per `metal2_port.md`). This is the *only* write outside the op's directory the recipe sanctions.
- **Do not import anything from `experimental/quasar/`** — not a name, not a construct, not as "evidence it's portable." Those copies carry idioms this recipe forbids (stale `circular_buffer.h` includes, `cb_*` naming, pre-settled hw-config shapes). If you land in one, close it and re-derive from the canonical recipe.

---

## 3. Mental model of the Metal 2.0 runtime (recap)

| Legacy concept | Metal 2.0 |
|---|---|
| `create_descriptor` → `ProgramDescriptor` | `create_program_artifacts` → `ProgramArtifacts` |
| CB index `tt::CBIndex::c_0` | named **DFB** `dfb::in0` + `DataflowBuffer` object |
| buffer-address RTA + `TensorAccessorArgs<N>()` | bound **tensor parameter** `tensor::src` + `TensorAccessor(tensor::src)` |
| positional `get_arg_val<uint32_t>(i)` | named `get_arg(args::name)` |
| sync-free / DM self-loop scratch CB | **Scratchpad** / **LocalTensorAccessor** (Gen1 self-loop is legal; **Gen2 rejects a DM self-loop** — a uplift item, see §6) |
| kernel `opt_level` (absent → O2 DM / O3 compute) | `KernelSpec::compiler_options.opt_level` — **carry the resolved legacy value verbatim** (§4) |

Tokens (`dfb::`, `args::`, `tensor::`, `scratch::`) come from the `KernelSpec` bindings via `#include "experimental/kernel_args.h"`. This is all base-recipe material — see `metal2_port.md` / `migration_guide.md`.

---

## 4. Host-factory settings that bite on the uplift

Full factory reference is in `metal2_port.md`; below are the settings most likely to differ or break during a Quasar uplift.

- **`opt_level` is a base-port concern, not an uplift edit.** Its resolution rule (absent → O2 on DM / O3 on compute; explicit values, even `Os`/`O0`, carried verbatim and never "corrected") belongs to the base Metal 2.0 port — see `metal2_port.md` "Compiler options". The uplift does **not** change `opt_level`. If you notice the base port set the wrong level (e.g. a compute kernel left the field absent so it defaulted below the legacy O3), flag it *to the base port* — don't fix it here. Wrong `opt_level` compiles and passes tests but silently shifts perf/precision.
- **Hardware config: carry values, not names.** `hw_config` is `to_compute_hardware_config(arch, compute_kernel_config)` (compute) / `create_*_datamovement_config(arch)` (DM). Fields were renamed for clarity — carry the legacy op's *resolved* values across, diff before/after. Adding the Gen2 variant is the `gen2_hardware_configs.md` post-port pass.
- **`unpack_modes` required for Float32 DFB consumers when `enable_32_bit_dest = true`.** Every Float32-formatted DFB a compute kernel consumes needs an explicit `unpack_modes` entry (legacy `Default` → `UnpackMode::UnpackToSrc`; legacy `UnpackToDestFp32` → `UnpackMode::UnpackToDest`). Symptom: spec-validator fails at build naming the FP32 DFB. (`migration_guide.md` → `unpack_modes`.)
- **Do NOT set `disable_dfb_implicit_sync_for_all` / `disable_implicit_sync_for`.** See §7 — implicit sync is the Gen2 default and disabling it is currently **not an option**.
- **Every DFB needs a valid `data_format_metadata`** — `Invalid` throws in `tile_size()`. Quasar has **Int32, not UInt32** (and no uint16) — see §7.
- **Designated-initializer field order** must match `KernelSpec` declaration order or `-Werror=reorder-init-list` fires.

---

## 5. Kernel-side recap

Base transformations (CB index → `dfb::`, positional arg → `get_arg(args::…)`, address-RTA + `TensorAccessorArgs` → `TensorAccessor(tensor::…)`) are the canonical port's job. The one Quasar-specific kernel rule worth repeating:

> **Use `DataflowBuffer::get_entry_size()`, never `get_local_cb_interface().fifo_page_size`.** On Quasar `fifo_page_size` is **stale**; reading it is a top cause of value-inflation / wrong-output bugs (§8.3).

NoC: `Noc noc; noc.async_write(cb, s, nbytes, {.offset_bytes=..}, {.page_id=.., .offset_bytes=..}); noc.async_write_barrier();` — and see §11 for the Gen2 NoC/multicast constraints.

---

## 6. Scratchpad, self-loop & borrowed DFBs (mostly canonical now)

`Scratchpad` / `LocalTensorAccessor` conversion is a **canonical post-port pass** — follow `sync_free_dfbs.md` (style) and `dm_self_loop_dfbs.md` (semantic). The Quasar-relevant facts:

- **A DM self-loop DFB is legal on Gen1 but rejected on Gen2.** Converting sync-free / DM self-loop CBs to `Scratchpad` (a node-local L1 page, no producer/consumer credits) or `LocalTensorAccessor` is the uplift fix. A *compute* self-loop is fine on both. **To classify:** read every binding kernel's `dfb_bindings` for that DFB — one kernel bound *both* `PRODUCER` and `CONSUMER` is a self-loop (compute → leave it; **DM → the Gen2 fix**); two distinct kernels is a normal FIFO. `sync_free_dfbs.md` is explicit: a self-loop is **not** the same as sync-free, and the pointer getters don't tell you which (conv2d's `ACT_TILIZED` is sync-free height-sharded, a real FIFO otherwise).
- **Borrowed-memory DFBs** (`DataflowBufferSpec::borrowed_from = <TensorParameter>`) overlay a resident shard; the backing L1 resolves at runtime. Real capacity = per-shard bytes / entry_size; over-pushing asserts in `dataflow_buffer.inl`.
- **Local self-read/self-copy (src==dst L1)** on the emulator can spin on `can_post` or silently drop the read — use a direct L1→L1 RISC copy, not a NoC loopback.

---

## 7. Quasar programming quirks (must-know)

- **`disable_dfb_implicit_sync_for_all` / `disable_implicit_sync_for` is NOT an option right now.** Implicit sync (the runtime handling FIFO sync via ISR when you pass a DFB straight to `Noc::async_read`/`async_write`) is the **Gen2 default**. Earlier bring-up disabled it wholesale to dodge a tile-counter **double-count** bug; that was never meant for a general port, and the underlying bug(s) should now be fixed. **Do not disable implicit sync.** If you still see a double-count / credit stall (§8.2), treat it as a regression to **report**, not to paper over by disabling — flag it and check with the runtime/LLK owners.
- **`compute_kernel_hw_startup(...)` exactly ONCE** at `main()` start. A per-block `hw_configure` corrupts engine state. Init state is in flux: not all inits are *short* yet, and you usually can't tell short vs long by name (only names containing `short` are guaranteed); a `hw_startup` left mid-kernel (e.g. by search-and-replace) is worse on Quasar than WH/BH — keep it at `main()` start only.
- **Re-init on every DFB-id change.** Buffer descriptors live in the init; if a kernel switches the DFB it operates on, re-run the relevant `*_init`.
- **Tilize needs pack config before `tilize_in`** — `tilize_init` omits pack `hw_configure`/init/dest_init → stale BD base → `PACR0_TILE_INC`. Do pack `hw_configure + init + dest_init` first. It also needs `llk_math_pack_sync_init` before it, or it inherits matmul's stale MATH_PACK phase (`0x19`).
- **No wide-tilize chunking / no DEST wrap.** SyncHalf DEST is a 2-bank ping-pong that never wraps; a single continuous loop is correct — manual chunking + re-init *causes* `0x19`.
- **Quasar has Int32, no uint16/uint32** device format. If a kernel has a uint16/uint32-specific code path, `#ifdef ARCH_QUASAR` it out; an op that merely *forwards* a `DataType` (no format branch of its own — e.g. typecast) has nothing to guard, and the limitation then lives at the format/LLK layer — flag it rather than editing the op. **RM shard width must be 16-byte aligned** (bf16 ⇒ multiple of 8 elements); validate it and round output widths up.
- **Non-zero-init semaphores don't port to Quasar** — a semaphore created with a non-zero initial value is WH/BH-only (slated for deprecation). Remove the dependency (op-owner change). (`quasar_audit.md` check 2.)
- **DPRINT-on changes timing** and can mask/expose HW races — see §8 and §9.
- **NoC / multicast differs** — no independent NOC0/NOC1, top-left-only mcast, degenerate-grid clamps: see §11.
- **Don't invent Quasar-only device interfaces.** If a sanctioned Metal 2.0 DFB/kernel API is missing on Quasar (e.g. the ring-rewind `evil_set_read_ptr`/`evil_set_write_ptr` is `#ifndef ARCH_QUASAR`, so absent on Gen2), **flag it as a missing-feature for the runtime team** rather than hand-rolling an op-level equivalent (e.g. poking `g_dfb_interface` with bespoke `SNAPSHOT`/`RESTORE` macros). A hand-rolled device interface is exactly the kind of thing reviewers reject; the fix belongs in the DFB API so both archs share one whitelisted mechanism. (This is a `quasar_audit.md`-style "missing Metal 2.0 feature" blocker.)

---

## 8. Known pitfalls & fixes (grep the symptom signatures)

> Signatures are kept verbatim (`0x19`, `0x10000`, `PACR0_TILE_INC`, …) so you can grep the tt-metal / tt-llk source for them. The Fix column is the one-line action.

### 8.1 Build / JIT errors
| Symptom | Root cause | Fix |
|---|---|---|
| DM kernel build fails including `common.hpp` (ckernel symbols) | shared data_movement `common.hpp` pulls compute-only ckernel into DM TU | `#if !ARCH_QUASAR` guard + `uintptr_t` casts (~20 DM kernels) |
| `'REDUCE_OP' was not declared` `-Wtemplate-body` compiling **plain tilize** | `llk_unpack_tilizeA_B_init` uses `REDUCE_OP` (a `-D` macro) as a free non-dependent name; plain tilize TU doesn't define it | `#ifdef REDUCE_OP … #endif` guard (now in `main`) |
| `-Werror=narrowing` in matmul CCL reader header | `hetergeneous_data_structs.hpp:191` | `static_cast<>` |
| `-Werror=int-to-pointer-cast` on `(void*)(uint32_t)` L1 address | Quasar pointers are 64-bit; a bare int→ptr cast is rejected | cast through `uintptr_t`: `(void*)(uintptr_t)addr` |
| matmul factory fails: `blank.cpp` missing | Metal2 tree lacks the trivial noop kernels | create `matmul/device/kernels/{dataflow,compute}/blank.cpp`; repoint `NOOP_*_KERNEL_PATH` |
| `log_debug` many-arg build fail (`TT_LOG_FOR_EACH_AGAIN`) | old tt-logger caps macro args | keep `log_debug` ≤ 12 args |
| `R_RISCV_HI20 has no matching LO12` at load (source uses `%` and `/`) | tt-2xx crt0 emits two `lui %hi` for `__tdata_lma`; relaxation sinks the 2nd, XIPify orphans it | pin once with `asm volatile` |
| halo build: `flush_l2_cache_range` undeclared on WH/BH | it's tt-2xx (Quasar)-only (declared in `internal/tt-2xx/risc_common.h`) | `#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)` guard in `halo_gather.cpp` |
| halo: `MEM_ZEROS_BASE` undefined on Quasar | Quasar memory map lacks the WH/BH zeros region | zero via the runtime `async_write_zeros()` (no `MEM_ZEROS_BASE` on Quasar) |
| reduce JIT fails: `sfpu_reduce` unresolved | `reduce_helpers_compute.inl` WH/BH-only SFPU path unguarded | `#ifndef ARCH_QUASAR` guard (3 sites) |
| `Semaphore::get_l1_addr()` is private | a kernel reads the raw semaphore L1 value | use the public `wait`/`wait_min`/`set`; drop the raw-pointer poll (it's redundant after `wait_min`) |
| arch reports `"invalid"` | `get_string_lowercase` missing QUASAR case | add the case |

### 8.2 Hangs / deadlocks
> **`disable_dfb_implicit_sync_*` is no longer the fix.** The first three rows are the historical implicit-sync **double-count** symptoms that were once worked around by disabling implicit sync. That workaround is retired (§7); the underlying bug should be fixed. If you still hit these, **report the regression** — do not disable implicit sync.

| Symptom | Root cause | Fix |
|---|---|---|
| `to_layout`/reader/writer hangs, credits stall | implicit-sync tile-counter double-count (historical) | report regression; do **not** re-disable implicit sync |
| explicit CB ops + hang, 16-bit tile counter overflow | explicit push/pop **and** implicit-sync ISR both bump the counter (historical) | report regression; do **not** re-disable implicit sync |
| fold-pad hang, DFB credit stall on `cb_out0` | implicit sync double-count on the out CB (historical) | report regression; do **not** re-disable implicit sync |
| `"Not done phys cores"` s2i hang | 8-bit DFB capacity truncation | 16-bit capacity fix (in `main`) |
| unpacker traps right after `wait_front`→`pop_front` | bare `pop_front` after `wait_front` needs a TDMA op between | intervening IDMA op (≥1 `copy_tile` between `tile_regs_*`); see §8.5 |
| conv tilize `0x19` | `tilize_init` omits `llk_math_pack_sync_init` (stale MATH_PACK) | add `pack_sync_init` pre-tilize |
| WH/BH conv fast_tilize hang | packer never reconfigured for tilize output when `packer_l1_acc` forced false | ungate `pack_reconfig_data_format` (3 sites) |
| WH conv SFPU-relu hang (LLK_ASSERTS only) | MATH↔PACK deadlock via `are_packers_configured_correctly` | reproduce with `TT_METAL_LLK_ASSERTS` unset — the assert path itself is what deadlocks |
| compute `0x19` with early/varying-t + wait-heavy waypoint | it's a MOP **timeout**, not a fault (slow reader) | raise `CSR_TIMEOUT_COUNT` in `ckernel_template.h` |
| block-sharded fused conv `0x19`/`0x119` at first DEST bank0 reuse | matmul recycle-handshake deadlock | **(still open)** route off fused path → split matmul |
| K-spill matmul `0x10000` (wrong-Neo addr) | HW TDMA→SYNC tile-counter race | HW-team issue; interim = **DPRINT on** |
| maxpool single-core pool-reduce hang | pool-reduce DEST sync on the resnet path | serialize the reduce DEST sync; a 32-core repro is a separate halo edge-shard artifact (evenly-sharded pool unaffected) |

### 8.3 Wrong output / PCC
| Symptom | Root cause | Fix |
|---|---|---|
| value inflation (~2×, above input max) / wrong pad | `fifo_page_size` read stale from `get_local_cb_interface` | use `DataflowBuffer::get_entry_size()` (fixed 13 sites) |
| global 7×7 avgpool = golden × **1.1504** (divisor ~42.67) | wrong reduce scalar in `pool_sum` (`generic_pools.cpp:1080`), Quasar-only | fix scalar |
| tilize PCC≈0 (batched UNP_DEST) | DEST-dvalid CTRL masks never armed on tt-metal path | `set_up_dest_dvalid_per_thread` in `tilize_init` UTD branch |
| tilize `0x19`/PCC≈0 (compute-API) | missing per-tile FPU dvalid clear | `llk_math_set_dvalid<FPU>` per tile in `tilize_block` |
| PACK→DM PCC 0.897, stale single-buffered slot | sim ignores `packer_wr_done_wait_mask` | producer `TTI_STALLWAIT(PACK)` before `push_back` |
| conv wrong output (in-place matmul partials, aliases output) | each K-block must re-accumulate into the same L1, so the DFB ring position needs rewinding between blocks | WH/BH: sanctioned DFB `get_*_ptr` snapshot + `evil_set_read_ptr`/`evil_set_write_ptr` restore (whitelist). Quasar: `evil_set_*` is Gen1-only (`#ifndef ARCH_QUASAR`) — **do NOT hand-roll a `g_dfb_interface` snapshot/restore in the op**; flag the missing Quasar DFB rewind API for the runtime team (§7). Scratchpad/LocalTensorAccessor do **not** apply here (the packer writes to a DFB and the buffer aliases the output tensor) |
| stale-L2-on-reused-L1-read | missing L2 invalidate | resnet path clean; ~7 off-path fixes (slice/transpose/tilize) |
| reduce reuses a STALE kernel on a branch behind `main` | buggy default program hash | custom `compute_program_hash` folding scaler/post_mul |

### 8.4 Init / LLK correctness
| Symptom | Root cause | Fix |
|---|---|---|
| unported/stub LLKs silently used | audit found `negative_tile`, `fast_tilize`, `sfpu_reduce`, `copy_tile_to_dst_init_short_with_dt` gaps | verify each LLK is ported before relying on it |
| conv `PACR0_TILE_INC` fault | `tilize_init` leaves stale pack BD base | pack `hw_configure+init+dest_init` before `tilize_in` |
| reduce/buffer-desc validator rejects `face_r_dim=9` | non-pow2 face rows from 3×3 | LLK limitation — avoid or pad |
| conv CB tile 2×2-face (z=4) with `face_r_dim!=16` asserts (`ckernel_trisc_common.h:121`) | tiny-tile geometry | dump `l1_addr` to confirm; avoid z=4 tiny tiles with non-16 face rows |
| 2D-mcast matmul `"No core at (0,1)"` on 2×1 grid | mcast corner not clamped for single-row/col | clamp +1 mcast corner when a dim has 1 core |

*(craq-sim emulator LLK gaps you may hit: `UNPACR0_STRIDE` unimplemented — stub it and file a ticket; partial-face `UNPACR_FACE` (`face_r<16`) unimplemented — mirror the `UNPACR_TILE` path with `face_rows=y_dim`; multicast corner `start>end` on NOC1 reverse rectangles — swap-normalize both corners.)*

### 8.5 More hardware bugs (field experience)
| Symptom | Root cause | Fix |
|---|---|---|
| unpacker/packer traps on `reserve_back`→`push_back` (write-side twin of `wait_front`→`pop_front`) | HW bug: a CB push/pop issued immediately after the matching reserve/wait needs an intervening TDMA op | put an IDMA op between them (in compute, ≥1 `copy_tile` between the `tile_regs_*` calls); a DPRINT in between only "fixes" it as a timing hack |
| intra-tensix DFB tile-counter aliasing (candidates: GroupNorm, LayerNorm, SDPA) | DM↔tensix DFBs use tile-counter indices 0–15 and ARE remapped; intra-tensix DFBs use 16–31 and were NOT, so non-remapped HW aliases via `index % 16` | remap ALL DFBs/tile counters (each takes 2 indices, e.g. 16&17, 18&19) — same tile-counter family as the K-spill `0x10000` |

---

## 9. Testing protocol

**Order:** BH → WH → Quasar emulator. BH/WH are the parity check that the Metal 2.0 port is a no-op refactor; a regression there means you changed behavior, not just enabled Gen2.

- **User runs all builds/tests.** Don't run or ask to run them — hand the exact command.
- **Control tests to prove WH/BH unaffected:** run the mainline op at the same geometry; if it passes and your build hangs, the LLK is fine and the bug is in your driving/plumbing.
- **Auditing without a device run?** You can't assert parity by test. Argue it structurally in the report — a zero, or fully `ARCH_QUASAR`-guarded, diff ⇒ no WH/BH behavior change — and hand the human the exact test command to run for confirmation.
- **Force JIT** when kernels change: `TT_METAL_FORCE_JIT_COMPILE=1`.
- **DPRINT:** requires `unset TT_METAL_LLK_ASSERTS` **and** the format form `DPRINT("fmt {}", args)`. DPRINT changes timing, so a "DPRINT makes it pass" result is a signal, not a fix.
- **`LLK_ASSERTS`:** a pass with `TT_METAL_LLK_ASSERTS` unset is not proof — several hangs only assert with it *on*. Run both ways, and start with asserts **on** (the LLK team added asserts for unsupported cases: pow2-only, 32×32-tile-only, unimplemented paths).
- **Slow dispatch / per-op serialization:** `enable_fast_runtime_mode:false` = slow dispatch; add `enable_logging:true` for one-op-at-a-time. Faults under it are genuine per-op bugs.
- **Available debuggers depend on the target.** On real WH/BH: `tt-triage` `stack.txt` (`dump_callstacks.py` = per-RISC kernel stacks, authoritative for hangs; `dump_lightweight_asserts.py` = pass/fail) + `generated/watcher/watcher.log`. **On the Quasar emulator `tt-triage`, `tt-exalens`, and device-side gdb are NOT available** — use DPRINT, `log_debug()`, WATCHER, LLK + lightweight asserts, and **host-side gdb**.
- **Debug tooling can hurt on the emulator:** DPRINT/WATCHER can push a kernel over the size limit (build fails) or hang; the DPRINT ring buffer and the NoC sanitizer are unreliable there; and debug slows execution enough to more easily trip the MOP timeout (§8.2). Always run **both** with debug env on and off.
- **Per-op golden-PCC logging:** gate a numeric-fingerprint log behind an env var (e.g. `RESNET_PCC_LOG=1`) after every value-producing op to localize the first op that diverges from golden. Keep it env-gated and off by default.
- **Strip DIAG before committing:** remove `[#… DEBUG]` / `DIAG` / stray `log_*` scaffolding. Keep only (a) functional workarounds (document why) and (b) env-gated instrumentation.

---

## 10. Definition of done

- [ ] Op is uplifted **in place** in its existing directory + namespace — **nothing copied into `experimental/quasar/`, no `::qsr`**.
- [ ] Factory is `create_program_artifacts`/`ProgramArtifacts`; kernels use `dfb::`/`args::`/`tensor::`/`scratch::`; no `CBIndex::c_`, positional `get_arg_val`, or address-RTA `TensorAccessorArgs`.
- [ ] Each kernel's `opt_level` **matches the resolved legacy value** (absent → O2 DM / O3 compute; explicit values carried verbatim).
- [ ] Every DFB has a valid `data_format_metadata`; kernels read sizes via `get_entry_size()`.
- [ ] Sync-free / DM self-loop DFBs converted to `Scratchpad`/`LocalTensorAccessor` (per `sync_free_dfbs.md`); sharded borrows capacity-checked.
- [ ] **No `disable_dfb_implicit_sync_for_all` / `disable_implicit_sync_for`** — rely on the Gen2 implicit-sync default; any double-count is reported, not worked around.
- [ ] No non-zero-init semaphore dependency.
- [ ] **BH and WH pass** the op's existing test suite unchanged (no PCC/behavior/perf regression).
- [ ] Quasar builds and runs; every Quasar-specific change is `ARCH_QUASAR`-guarded so WH/BH keep the original path.
- [ ] No DIAG/debug leftovers; functional workarounds documented.
- [ ] Genuine missing core-LLK deps (not fixable inside the op) flagged for a dedicated PR, not silently bundled — feed them back into `quasar_audit.md`.
- [ ] `QUASAR_UPLIFT_REPORT.md` written with GREEN/RED status, every changed file + reason, and the WH/BH parity claim; RED-stop conditions checked.

---

## 11. NoC & multicast on Quasar

Quasar does **not** expose the independent **NOC0 / NOC1** pair WH/BH code assumes, so directional NoC tricks (reverse-direction transfers on "the other" NoC, or multicasting bottom-right → top-left) don't translate. Normalize everything to one forward direction from the top-left.

- **Multicast originates top-left, forward.** You cannot mcast bottom-right → top-left like WH/BH. A reverse rectangle (`start > end`, the NOC1-style backward mcast) must be **swap-normalized** so `start ≤ end` on *both* corners (fixed the craq-sim `start>end` mcast assert).
- **Clamp the mcast corner on degenerate grids.** The emulator grid is tiny (e.g. height 2 × width 1) while much code assumes ≥ 7×8. The 2D-mcast corner `grid_start + {1,1}` names a nonexistent core on a 1-wide/1-tall grid → `No core at (0,1)`. Clamp the `+1` per dimension so it only steps into a dim spanning > 1 core (see §8.4).
- **Pad the W-dim tail, not only H.** A writer that padded only the H-dim tail left the N/W-dim tail block writing the full (unpadded) width → OOB / wrong output. Pad both tails.
- **Zero-fill via `async_write_zeros()`** — there is no `MEM_ZEROS_BASE` region on Quasar.

---

## 12. Quasar architecture background (why several of the above happen)

- **DFB credits are HW tile counters, behind a remapper.** DM↔tensix DFBs use counter indices **0–15** and are remapped; intra-tensix DFBs use **16–31**. Non-remapped HW aliases via `index % 16`, so intra-tensix counters can collide — hence the "remap all DFBs" workaround (each DFB takes 2 indices). This is the family behind the K-spill `0x10000` and the §8.5 intra-tensix aliasing.
- **Consumer allocation (arena).** Producer adds N tiles → each of C consumers sees N/C (N must be a multiple of C); a consumer's decrement hits **both** its own and the producer's counter. (Group allocation — N/4 per consumer, decrement own only — isn't used on the resnet path.)
- **Implicit sync** (Gen2): the runtime handles FIFO sync via ISR when a DFB is passed straight to `Noc::async_read`/`async_write` — the default, and (per §7) the one you should rely on. On Gen1 it's a no-op and the explicit `reserve/push` / `wait/pop` FIFO pattern remains.
- **DM caches aren't auto-coherent with L1 producers.** A reused L1 slot can read stale from L2 → explicit `invalidate_l2_cache_range` (being moved into runtime code).
- **Emulator variants differ.** The functional simulator (`libttsim.so`) and craq-sim aren't identical (e.g. `UNPACR0_STRIDE` is stubbed on craq-sim), and both can diverge from real HW — reproduce a suspected sim artifact on more than one target before concluding it's real.

---

## 13. References

**Canonical Metal 2.0 op-porting recipe** (branch `akertesz/op-porting-recipe`): `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` — start at `human/READ_ME_FIRST.md`; Quasar-uplift gate at `ai/audit/quasar_audit.md`.

**Tenstorrent Confluence:**
- Tensix NEO High-Level Specification — https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/84508873/Tensix+NEO+High+Level+Specification
- Tensix NEO / Quasar Errata — https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/1802436609/Tensix+NEO+Quasar+Errata
- Overlay Tile Counter Interrupt Protocol — https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/408289306/Overlay+Tile+Counter+Interrupt+Protocol
- Quasar — Programming Quirks — https://tenstorrent.atlassian.net/wiki/spaces/LLK/pages/2316533761/Quasar+-+Programming+Quirks
- Resnet OP bring-up — https://tenstorrent.atlassian.net/wiki/spaces/LLK/pages/2608463913/Resnet+OP+bringup
- Tile Counter Remapping Block — https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/1401028761/Tile+Counter+Remapping+Block
- Tensix Formats — https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/237174853/Tensix+Formats

---

*Finding the fix site:* the symptom signatures above (`0x19`, `0x10000`, `face_r_dim=9`, `"Not done phys cores"`, `PACR0_TILE_INC`) are kept verbatim so you can grep the tt-metal / tt-llk source for the exact fault site and the guarding `#ifdef ARCH_QUASAR`.
