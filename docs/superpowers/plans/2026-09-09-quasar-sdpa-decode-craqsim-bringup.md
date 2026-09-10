# Quasar SDPA-decode craq-sim Bring-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended for Phase 1) or superpowers:executing-plans (recommended for Phase 2) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get `ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode` to compile for Quasar and pass its `batch1` unit test (correct PCC) on the craq-sim (`libttsim.so`) simulator.

**Architecture:** The op is already Metal-2.0-ported and DFB-audited; this is a bring-up, not a rewrite. The gating compile blocker is that the decode compute kernel `#include`s a custom fused-SFPU LLK (`ckernel_sfpu_sdpa.h`) that has no Quasar implementation and is raw SFPI microcode (hard to port). We instead **swap the fused SFPU calls for generic Quasar SFPU tile ops** (which already exist), fold in a stashed draft that fixes the semaphore API + three Gen2 DFB-structure blockers, cherry-pick a `wait_front`/`read_tile_value` prerequisite branch, tighten dtype validation to Quasar's formats, then iteratively debug the runtime to correct PCC on craq-sim.

**Tech Stack:** C++ TTNN device op (`DeviceOperation`/`ProgramSpec`/`ProgramArtifacts`, Metal 2.0 DataflowBuffers), JIT compute/dataflow kernels, Quasar SFPU LLK (`tt_llk_quasar`), craq-sim (`libttsim.so`) under UMD, pytest.

**Spec / technical references (travel with this plan):**
- `quasar_sdpa_migration_audit.md` (repo root) — external migration audit. **Authoritative except two corrections, both verified 2026-09-09:** (1) its blocker #1 "port `ckernel_sfpu_sdpa.h`" is *superseded* — we swap to generic SFPU instead of porting; (2) its claim that `reduce_custom`/`matmul_custom`/`sub_bcast_custom` LLK backing needs porting is *stale* — those already exist under `tt_metal/hw/ckernels/quasar/metal/llk_api/experimental/`.
- Memory `project_quasar_sdpa_sim_bringup.md` — consolidated brainstorm decisions.
- GitHub issue #54630 — the umbrella Quasar-uplift thread (vsureshTT's emulator runs + the `pack_init` runtime root-cause).

## Global Constraints

Every task's requirements implicitly include these:

- **Quasar dtypes: `BFLOAT16` / `FLOAT32` / `INT32` only.** No `BFLOAT8_B`, `BFLOAT4_B`, `UINT16`, `UINT32` — they pass host validation today but fail later at DFB creation on Quasar.
- **Keep WH/BH green.** The fork is still validated on Wormhole n150. Every Quasar-specific change is guarded (`#ifdef ARCH_QUASAR` in kernels; `device->arch() == tt::ARCH::QUASAR` on host) so the WH/BH code path is textually unchanged.
- **Fork-local edits.** Prefer editing files under `ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/`. Do **not** modify shared `tt_metal/hw/inc/...` headers or the restored main-tree `transformer/sdpa_decode` unless a task explicitly says so.
- **One test at a time on device/sim.** Never batch multiple pytest files in one invocation (device-hang isolation).
- **`build_metal.sh` returns 0 even on a `ttnn -Werror` failure** — after every build, `grep -c "error:" <build log>` and treat non-zero as failure. Do not trust the exit code alone.
- **Commit after each green step.** Small, scoped commits; `[Bug fix]`/`[Feature]`/`[Cleanup]` category prefixes per CONTRIBUTING.
- **Sentinel test** (the bar): `models/experimental/llama32_1b_quasar/tests/ops/test_scaled_dot_product_attention_decode.py::test_scaled_dot_product_attention_decode[batch1-ttnn_mesh_device0]` passing with correct PCC on craq-sim.

### Standard commands (referenced by tasks)

**craq-sim run (`$SIMRUN <test-id>`):**
```bash
cd /localdev/gchoudhary/tt-metal && source python_env/bin/activate && \
export TT_METAL_HOME=/localdev/gchoudhary/tt-metal PYTHONPATH=/localdev/gchoudhary/tt-metal && \
export TT_METAL_SIMULATOR=/localdev/gchoudhary/sim/libttsim.so \
       TT_UMD_SIMULATOR_PATH=/localdev/gchoudhary/sim TT_SIMULATOR_LOCALHOST=1 && \
export CHIP_ARCH=quasar ARCH_NAME=quasar TT_METAL_SLOW_DISPATCH_MODE=1 && \
export TT_METAL_CACHE=/localdev/gchoudhary/.qsr-sdpa-cache && rm -rf /localdev/gchoudhary/.qsr-sdpa-cache && \
pytest "<test-id>" -vv --timeout=600 -p no:xdist
```
Use a PRIVATE `TT_METAL_CACHE` (not the shared `/tmp/craq-ttnn-ops-cache`, which races peer runs); clear it each run since kernels JIT from source. `-p no:xdist` avoids CoordinateTranslationError on Quasar's 8x4 layout. Sim sanity (once): `strings "$TT_METAL_SIMULATOR" | grep libttsim_pci_mem_wr_bytes` must hit.

**Build (`$QBUILD`)** — per the op owner, ALWAYS `./build_metal.sh --build-all`. It runs the install/copy step itself, so the freshly built libs land in the runtime locations; a targeted `ninja`/`cmake --build` compiles but does NOT install, silently leaving a stale lib (this cost several confusing iterations).
```bash
cd /localdev/gchoudhary/tt-metal && \
./build_metal.sh --build-all 2>&1 | tee /tmp/qbuild.log ; \
grep -c "error:" /tmp/qbuild.log   # must print 0 (build_metal.sh exit code is unreliable)
```
`CHIP_ARCH=quasar` is a RUN-time var, NOT a build flag. Device kernels JIT-compile from source at RUN time and are cached, so a kernel-only edit needs only a JIT-cache clear + rerun (no rebuild); only HOST C++ (`*_device_operation.cpp`, `*_program_factory.cpp`) needs `$QBUILD`.

---

## Execution progress (updated 2026-09-10)

**Phase 1 — COMPLETE.** The op compiles cleanly for Quasar and loads on craq-sim.
- Task 1 ✅ cherry-picked `wait_front` + `read_tile_value` (test-only commit dropped; the 3 stale DFB test files later reverted to main — do NOT re-patch `test_dataflow_buffer_apis.cpp`, it's correct on current main).
- Task 2 ✅ stash applied + vetted (semaphore-API migration, `out_o`/`out_worker` split, `col_identity` drop, `intermed_out` scratchpad).
- Task 3 ✅ SFPU swap — survived a Critical fix round: apply exp scale via `mul_unary_tile` + unscaled `exp_tile_init<approx>()`, NOT `exp_tile_init<…, scale_fp32>()` (Quasar `exp_init` static_asserts on non-default scale).
- Task 4 ✅ dtype validation gated to bf16/Int32 on Quasar (robustness; sentinel is bf16/non-paged, so not load-bearing).
- Task 5 ✅ build. **Op-owner directive: ALWAYS `./build_metal.sh --build-all`** (see `$QBUILD`). A targeted `ninja` build compiles but skips the install step, silently leaving a stale runtime lib — this masqueraded as phantom bugs (e.g. an `out_o` multi-binding error from pre-fix code).

**Phase 2 — IN PROGRESS.** The plan's anticipated failure catalog (Tasks 6–11: `pack_init`/all-zero/tree-reduction) did NOT materialize. Instead the first real JIT compiles surfaced a chain of pre-existing Quasar LLK-compat gaps, each fixed + committed:
1. `datacopy_init` — Quasar LLK has no `PackMode` template arg (arch-branch).
2. `log_tile` — SFPU log not wired on Quasar; guarded in the sink-only `log_block`.
3. `fp32_dest_acc_en` `reduce_max` — Quasar can't do 32-bit-DEST block `reduce_max`; forced bf16 DEST on Quasar. ⚠️ **May affect PCC — unverified; the sentinel has not yet reached a numerics result.**
4. semaphore typing — `read_k` templated on the token-deduced type; call sites keep bare `Semaphore s(sem::name)` (`Semaphore<>` forces LOCAL_NONATOMIC and static_asserts).
5. code size — compute `opt_level` O3→`Os` (trisc0 code region overflow 0x6924 > 0x6000).

**CURRENT BLOCKER (architectural — awaiting op-owner decision):** Quasar caps intra-Tensix DFBs at **8** (16 tile counters, 2 per DFB), but flash-decode declares **11** compute intermediates — `qk_im, out_im, out_accumulate_im, max_1, max_2, sum_1, sum_2, exp_max_diff, prev_sum_2, exp_max_diff_2, out_accumulate_im_2` — all allocated unconditionally, all core online-softmax double-buffers (NOT sink/streaming-gated; verified). Reducing to ≤8 needs a real refactor: merge `_1`/`_2` paired buffers into single DFBs with 2 entries (counter cost is per-DFB, so 3 merges → 11→8), or convert some to scratchpads. The sentinel has NOT yet produced a PCC number.

**Note:** Tasks 6–11 below are the *originally anticipated* Phase-2 work and did not occur as written; the real Phase-2 work is the chain above. The SDD ledger `.superpowers/sdd/2026-09-09-quasar-sdpa-decode-craqsim-bringup/progress.md` has the blow-by-blow. Commit SHAs move on rebase — match commits by subject line.

---

## File Structure

Files this plan creates or modifies, by responsibility:

| File | Phase | Responsibility / change |
|---|---|---|
| `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h` | 1 | (cherry-pick) `wait_front` block-fix surface |
| `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl` | 1 | (cherry-pick) Quasar `read_tile_value` + `wait_front` impl |
| `sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp` | 1 | (stash) semaphore-API migration (`sem::k_mcast` object form) |
| `sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp` | 1 | (stash) semaphore-API migration, `col_identity` removal, `intermed_out`→Scratchpad, `out_worker` |
| `sdpa_decode/device/kernels/dataflow/dataflow_common.hpp` | 1 | (stash) `read_k` takes `Semaphore*` instead of id |
| `sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp` | 1,2 | (stash) `out_o`→`out_worker` routing, `read_tile_value` free-fn; (P2) SFPU→MATH |
| `sdpa_decode/device/kernels/compute/compute_common.hpp` | 1,2 | **SFPU swap** (guard the LLK include + ARCH_QUASAR generic-SFPU helper branches); (P2) `pack_init` + SFPU→MATH |
| `sdpa_decode/device/sdpa_decode_program_factory.cpp` | 1 | (stash) `out_o`/`out_worker` split, `col_identity` drop, `intermed_out` ScratchpadSpec |
| `sdpa_decode/device/sdpa_decode_device_operation.cpp` | 1 | arch-aware dtype validation (bf16/Int32 on Quasar) |
| `models/experimental/llama32_1b_quasar/tests/ops/op_utils.py` | 1 | (stash) Quasar host-tilize upload workaround |

---

## PHASE 1 — Compile for Quasar

Deterministic edits; subagent-friendly. Phase 1's gate is a clean Quasar build (`$QBUILD` prints `0` errors) with the sentinel test at least *loading and dispatching* (may still fail PCC/hang — that's Phase 2).

### Task 1: Cherry-pick the `wait_front` / `read_tile_value` prerequisite

**Files:**
- Modify (via cherry-pick): `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h`, `tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl`

**Interfaces:**
- Produces: a Quasar `wait_front()` that blocks the RISCV core until data lands (prevents premature L1 peeks), and a Quasar `ckernel::read_tile_value(dfb, tile_idx, elem_idx)` free function. Task 6's `sdpa_flash_decode.cpp` change consumes `read_tile_value`.

- [ ] **Step 1: Confirm the three commits and their files**

Run:
```bash
git log --oneline origin/main..origin/skotarac/fix_wait_front_qsr
git diff --stat origin/main...origin/skotarac/fix_wait_front_qsr
```
Expected: 3 commits (`14113fb` wait_front block-fix, `b46ecd8`/`83dbff7` read_tile_value); files include `dataflow_buffer.h` + `tt-2xx/dataflow_buffer.inl` (+ test kernels).

- [ ] **Step 2: Cherry-pick the two source commits, dropping test-only hunks on conflict**

```bash
git cherry-pick 83dbff78c3e b46ecd8be4b 14113fb269c
# if a test file conflicts:  git checkout --theirs <path> || git rm <path> ; git add -A ; git cherry-pick --continue
```
Expected: `dataflow_buffer.h` and `tt-2xx/dataflow_buffer.inl` land. Resolve conflicts in favor of the incoming Quasar impl; test-kernel hunks (`test_dataflow_buffer_apis.cpp`, `dfb_producer.cpp`) may be dropped.

- [ ] **Step 3: Verify the Quasar `read_tile_value` symbol is present**

Run: `grep -n "read_tile_value" tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl`
Expected: a Quasar definition exists.

- [ ] **Step 4: Commit**

```bash
git add tt_metal/hw/inc/api/dataflow/dataflow_buffer.h tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer.inl
git commit -m "[Bug fix] Quasar: wait_front RISCV-block + read_tile_value (cherry-pick skotarac/fix_wait_front_qsr)"
```

### Task 2: Apply and vet the stashed host/structural draft

**Files:**
- Modify: `reader_decode_all.cpp`, `writer_decode_all.cpp`, `dataflow_common.hpp`, `sdpa_flash_decode.cpp`, `compute_common.hpp`, `sdpa_decode_program_factory.cpp`, `op_utils.py` (all via `git stash apply stash@{0}`)

**Interfaces:**
- Produces: semaphore API in object form (`Semaphore obj(sem::name)`); a directed `out_o`(writer→compute) / `out_worker`(compute→writer) DFB pair replacing the `out_o` multi-binding; `col_identity` removed; `intermed_out` as `ScratchpadSpec{unique_id="intermed_out"}` with `writer_scratch` binding. Task 6+ (runtime) depend on these being structurally correct.

- [ ] **Step 1: Apply the stash (same base as HEAD → clean)**

Run: `git stash apply stash@{0}`
Expected: applies with no conflict (7 files modified).

- [ ] **Step 2: Vet — no legacy semaphore tokens remain in decode kernels**

Run: `grep -rn "get_semaphore\|Semaphore<>" ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/kernels/`
Expected: **zero** hits (all migrated to `Semaphore obj(sem::name)`).

- [ ] **Step 3: Vet — the `out_o`/`out_worker` split is consistent host↔kernel**

Run:
```bash
grep -n "out_worker\|out_o" ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp
grep -n "dfb_out_worker\|dfb_out_o\|out_worker\|out_o" ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/kernels/dataflow/writer_decode_all.cpp
```
Expected (confirm by reading): factory binds `out_o` = writer-PRODUCER/compute-CONSUMER and `out_worker` = compute-PRODUCER/writer-CONSUMER (no `/*multi=*/true`); compute `move_block(...dfb_out_worker...)`; writer reads `dfb::out_worker`. Flag if any endpoint is doubly-bound (would re-introduce the Gen2 self-loop).

- [ ] **Step 4: Vet — `intermed_out` scratchpad + `col_identity` removal**

Run:
```bash
grep -n "INTERMED_OUT_SCRATCH\|ScratchpadSpec\|writer_scratch\|col_identity" ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp
```
Expected: `ScratchpadSpec{.unique_id = INTERMED_OUT_SCRATCH, ...}` registered on `spec.scratchpads`; `.scratchpad_bindings = std::move(writer_scratch)` on the writer kernel; **no** remaining `col_identity` DFB/bindings. Confirm the writer kernel uses `Scratchpad<uint8_t> intermed_scratch(scratch::intermed_out)` and `get_base_address()`.

- [ ] **Step 5: Commit (structural draft only — not yet compiled)**

```bash
git add -A
git commit -m "[Feature] Quasar sdpa_decode: semaphore-API migration + Gen2 DFB structure (out_worker split, col_identity drop, intermed_out scratchpad)"
```

### Task 3: Swap the fused SFPU calls to generic Quasar SFPU

The compile blocker. The decode compute kernel includes `experimental/llk_sfpu/ckernel_sfpu_sdpa.h` (no Quasar impl) and calls fused first-column/scaled primitives. On Quasar we (a) guard the include and (b) route each fused call to generic SFPU tile ops the fork's helpers already have building blocks for. WH/BH keep the fast path.

**Files:**
- Modify: `sdpa_decode/device/kernels/compute/compute_common.hpp` (the helper bodies; the include at line 36)

**Interfaces:**
- Consumes: generic SFPU/eltwise already `#include`d by this file — `exp_tile`/`exp_tile_init` (`api/compute/eltwise_unary/exp.h`), `recip_tile`/`recip_tile_init` (`.../recip.h`), `sigmoid` (`api/compute/eltwise_unary` / SFPU), `sub_tiles`/`add_tiles`/`mul_tiles` (`eltwise_binary.h`), `binop_with_scalar` (`.../binop_with_scalar.h`), `bcast` (`bcast.h`).
- Produces: on Quasar, the same softmax numerics via generic ops. The fused entry-point *names* the kernel calls (`exp_block`, `reduce_c`, `sub_exp_block`, ...) keep the same signatures — only their bodies branch — so `sdpa_flash_decode.cpp` call sites are unchanged.

- [ ] **Step 1: Read the current fused-helper bodies + the reference math to derive exact replacements**

Read, in the decode `compute_common.hpp`, the bodies of the helpers that wrap: `exp_tile_first_column` (×4 uses), `recip_tile_first_column` (×4), `sub_exp_block` / `sub_exp_block_bcast_cols_inplace` / `sub_exp_add_tile` (softmax), `sigmoid_sub`. Cross-reference the WH math in `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h` (`calculate_exponential_first_column` = `exp((x)*scale)` with packer-ReLU clamp of negatives; `calculate_recip_first_column` = `1/x`). Note each helper's `scale` handling and whether only column 0 is read downstream (it is — "first_column" variants are a compute-saving optimization, so a full-tile generic op is numerically safe).

- [ ] **Step 2: Guard the LLK include so Quasar never pulls the missing header**

In `compute_common.hpp` (line ~36):
```cpp
#ifndef ARCH_QUASAR
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
#endif
```

- [ ] **Step 3: Add ARCH_QUASAR branches in each fused helper (worked patterns)**

Apply this transformation per helper (WH/BH branch unchanged; Quasar branch uses generic ops). Representative patterns:

`recip_tile_first_column`-based helper → generic reciprocal:
```cpp
#ifdef ARCH_QUASAR
    recip_tile_init();
    recip_tile(dst_idx);          // full-tile 1/x; only col 0 is consumed downstream
#else
    recip_tile_first_column(dst_idx);   // WH/BH fused fast path
#endif
```

`exp_tile_first_column<approx, scale>`-based helper → scale then generic exp:
```cpp
#ifdef ARCH_QUASAR
    // out = exp(in * scale); scale folded via mul_unary_tile (packed bf16 const `scale_bf16`)
    binop_with_scalar_tile_init();
    mul_unary_tile(dst_idx, scale_bf16);
    exp_tile_init<true /*approx*/>();
    exp_tile(dst_idx);
#else
    exp_tile_first_column<true /*approx*/, scale_bf16, false>(dst_idx);
#endif
```

`sub_exp_block<scale>(a_dfb, b_dfb, out_dfb, n)` → `out = exp((a-b)*scale)`:
```cpp
#ifdef ARCH_QUASAR
    sub_tiles(a_dfb, b_dfb, /*a_idx*/0, /*b_idx*/0, /*dst*/0);   // a-b into dst
    binop_with_scalar_tile_init();
    mul_unary_tile(0, scale_bf16);
    exp_tile_init<true>();
    exp_tile(0);
    pack_tile(0, out_dfb);
#else
    /* existing fused sub_exp_block body */
#endif
```

`sigmoid_sub` → generic sigmoid + subtract (attention-sink path):
```cpp
#ifdef ARCH_QUASAR
    sigmoid_tile_init();
    sigmoid_tile(dst_idx);
    sub_tiles(/* ... as in the fused body ... */);
#else
    sigmoid_sub(/* existing */);
#endif
```
Repeat for every enumerated site from Step 1. Keep each helper's outer signature (name, params) identical so `sdpa_flash_decode.cpp` is untouched.

- [ ] **Step 4: Verify no unguarded reference to the fused primitives remains on the Quasar path**

Run:
```bash
grep -n "first_column\|sub_exp\|sigmoid_sub\|ckernel_sfpu_sdpa" ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp
```
Expected: every remaining hit is inside a `#else`/`#ifndef ARCH_QUASAR` block. The include is guarded.

- [ ] **Step 5: Commit**

```bash
git add ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp
git commit -m "[Feature] Quasar sdpa_decode: swap fused SFPU softmax primitives for generic SFPU tile ops"
```

### Task 4: Arch-aware dtype validation (Quasar formats)

**Files:**
- Modify: `sdpa_decode/device/sdpa_decode_device_operation.cpp` (lines ~41-42 Q/K/V, ~144-145 mask, ~225-226 sharded page-table)

**Interfaces:**
- Produces: on Quasar, validation rejects `BFLOAT8_B`/`BFLOAT4_B`/`UINT16`/`UINT32` with a clear message, so a mis-typed input fails at host with a readable error instead of a cryptic DFB-creation crash.

- [ ] **Step 1: Confirm the sentinel test's actual input dtypes**

Read `models/experimental/llama32_1b_quasar/tests/ops/test_scaled_dot_product_attention_decode.py` (the `batch1` case) and note the Q/K/V, mask, and page-table dtypes it constructs. If it uses a Bfp8 KV cache or a non-Int32 page table, the test config itself must be bf16/Int32 for Quasar (adjust the test's Quasar branch, mirroring the `op_utils.py` arch check) — this is required for the test to *pass*, not just compile.

- [ ] **Step 2: Add an arch guard to the Q/K/V and mask dtype asserts**

At `sdpa_decode_device_operation.cpp:41-42` and `:144-145`, replace the accept-list with an arch-aware one:
```cpp
const bool is_quasar = /* device / arch handle */ ->arch() == tt::ARCH::QUASAR;
TT_FATAL(
    input_tensor.dtype() == DataType::BFLOAT16 ||
    (!is_quasar && (input_tensor.dtype() == DataType::BFLOAT8_B || input_tensor.dtype() == DataType::BFLOAT4_B)),
    "Quasar SDPA-decode supports only BFLOAT16 for Q/K/V; got {}", input_tensor.dtype());
```
Apply the analogous guard to the mask assert. (Obtain the arch handle from the existing device/compute-config already in scope in `validate`.)

- [ ] **Step 3: Build (host change) and confirm it compiles**

Run: `$QBUILD`
Expected: `grep -c "error:"` prints `0`.

- [ ] **Step 4: Commit**

```bash
git add ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp
git commit -m "[Bug fix] Quasar sdpa_decode: restrict dtype validation to bf16/Int32"
```

### Task 5: Full Quasar build — Phase 1 gate

- [ ] **Step 1: Build everything for Quasar**

Run: `$QBUILD`
Expected: `grep -c "error:" /tmp/qbuild.log` prints `0`. If non-zero, read the first error, localize to the task that introduced it (semaphore token → Task 2; SFPU symbol → Task 3; dtype/arch handle → Task 4; `read_tile_value` → Task 1), fix, rebuild.

- [ ] **Step 2: Confirm the kernels JIT-compile at dispatch (sentinel loads)**

Run: `$SIMRUN "models/experimental/llama32_1b_quasar/tests/ops/test_scaled_dot_product_attention_decode.py::test_scaled_dot_product_attention_decode[batch1-ttnn_mesh_device0]"`
Expected: the test gets past program creation and JIT kernel build (no compile/link error, no missing-symbol). It **may** still hang or fail PCC — that is Phase 2. Capture the exact failure text into `/tmp/sim_run_0.log` for Task 6.

- [ ] **Step 3: Commit any build-fix deltas** (if Step 1 required fixes)

---

## PHASE 2 — Drive `batch1` decode to correct PCC on craq-sim

Empirical and iterative; best executed **inline** (interactive debugging), not one-shot subagents. Use superpowers:systematic-debugging for each failure. The "test" each cycle is `$SIMRUN <sentinel>`; the loop is run → classify → apply the matching known-fix → re-run → commit.

### Task 6: First-run triage (diagnostic gate)

**Interfaces:**
- Produces: a classified failure signature that routes to Tasks 7–10.

- [ ] **Step 1: Run the sentinel and classify the failure**

Run: `$SIMRUN "<sentinel test-id>"` (see Global Constraints for the full id)
Classify the result against this catalog and proceed to the matching task:

| Signature | Likely cause | Go to |
|---|---|---|
| Output all-zero / PCC ≈ 0 (no hang, no assert) | packer target not re-armed after output switch | Task 7 |
| Hang / dispatch stall | CB/DFB producer-consumer mismatch, or SFPU on wrong thread | Task 8, then re-triage |
| PCC low-but-nonzero | SFPU-swap numeric parity (scale/approx) or tree-reduction rewrite | Task 3 (revisit) / Task 10 |
| Watcher NoC assert / bad-address | bare packer pair, or DFB-getter→NoC uncached alias | Task 9 |
| Correct PCC | done — go to Phase 3 | Task 11 |

- [ ] **Step 2: Record the signature** in `/tmp/sim_run_0.log` and note which task(s) apply.

### Task 7: `pack_init` after each output-DFB switch

vsureshTT's verified root cause of all-zero output on Quasar (rms_norm, paged_update_cache): `pack_reconfig_data_format` only reprograms the format gasket; on Quasar the packer's L1 destination is a descriptor set by `pack_init`, so after switching the output DFB you must re-arm it. Credits still balance → no assert, silent zeros. SDPA switches output DFBs constantly (qk_im, out_accumulate_im, out_m/l, out_worker...). **This is documented in-tree:** `tt_metal/hw/inc/api/compute/reconfig_data_format.h:758` — `NOTE(ARCH_QUASAR): ... Call pack_init(new_cb_id) before pack_tile`.

**Files:** Modify `sdpa_decode/device/kernels/compute/compute_common.hpp` and/or `sdpa_flash_decode.cpp` at each output-DFB switch.

- [ ] **Step 1: Enumerate output-DFB switch sites**

Run: `grep -n "pack_reconfig_data_format\|pack_tile\|reserve_back" ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp` — list the points where packing target changes.

- [ ] **Step 2: Insert a guarded `pack_init` after each switch**

Pattern (apply at every enumerated site):
```cpp
pack_reconfig_data_format(out_dfb);
#ifdef ARCH_QUASAR
    pack_init(out_dfb);   // Quasar: re-arm packer L1 descriptor (pack_reconfig alone is insufficient)
#endif
```

- [ ] **Step 3: Re-run the sentinel**

Run: `$SIMRUN "<sentinel>"`
Expected: output is no longer all-zero (PCC moves off ~0). If still zero, check that *every* switch site was covered and that `tilize_init`/`copy_init` paths also re-arm pack.

- [ ] **Step 4: Commit**
```bash
git commit -am "[Bug fix] Quasar sdpa_decode: re-arm packer with pack_init after output-DFB switch"
```

### Task 8: Complete the SFPU→MATH pass

On Quasar, SFPU ops must run on the MATH thread (no dedicated trisc3 yet); the fork's stash only wrapped `log_block`. Any SFPU init/op that currently dispatches to PACK must be `MATH(())`-wrapped on Quasar.

**Files:** Modify `compute_common.hpp`, `sdpa_flash_decode.cpp`.

- [ ] **Step 1: Find SFPU calls not already MATH-guarded**

Run: `grep -n "exp_tile\|recip_tile\|sigmoid\|log_tile\|reduce_tile" ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp | grep -v "MATH("`

- [ ] **Step 2: Wrap the SFPU init/op on the Quasar path**

Pattern:
```cpp
#ifdef ARCH_QUASAR
    MATH((exp_tile_init<true>()));
    MATH((exp_tile(dst_idx)));
#else
    exp_tile_init<true>();
    exp_tile(dst_idx);
#endif
```
(Where a call is already inside a `MATH(())` or the generic tile-op already forces MATH internally, leave it.)

- [ ] **Step 3: Re-run + commit**
Run: `$SIMRUN "<sentinel>"` → expected: no hang attributable to SFPU-thread mismatch. `git commit -am "[Bug fix] Quasar sdpa_decode: run SFPU ops on MATH thread"`

### Task 9: Bare packer-pair mitigation + DFB-getter→NoC aliases

**Files:** Modify `reader_decode_all.cpp` (:227-234), and any writer/reader site feeding a DFB pointer-getter to a NoC API.

- [ ] **Step 1: Insert `dummy_pack` in the bare packer pair**

At `reader_decode_all.cpp:227-234` the `dfb_q_rm_buf.reserve_back → push_back` and `dfb_q.reserve_back → push_back` pairs have nothing between them (TEN-4746). Add the mitigation used elsewhere in the fork:
```cpp
dfb_q_rm_buf.reserve_back(n);
#ifdef ARCH_QUASAR
    dummy_pack(dfb::q_rm);   // TEN-4746: order push after the (absent) pack
#endif
dfb_q_rm_buf.push_back(n);
```

- [ ] **Step 2: If Task 6 showed a NoC bad-address assert,** confirm any `get_read_ptr()`/`get_write_ptr()` fed to a peer NoC op uses the uncached alias on Quasar (the DFB getter already applies `MEM_L1_UNCACHED_BASE` under `ARCH_QUASAR && COMPILE_FOR_DM` per `project_metal2_port_gotchas`); fix any raw-pointer-math site that bypasses the getter.

- [ ] **Step 3: Re-run + commit.**

### Task 10: Verify the tree-reduction rewrites are numerically correct

The stash's riskiest change: `out_o`→`out_o`/`out_worker` split and `intermed_out`→Scratchpad reshaped the multi-core flash-decode reduction. If PCC is nonzero but wrong, suspect these.

- [ ] **Step 1: Sanity-check single-core first**

The `batch1` case may map to few cores. Confirm whether the reduction (`reduce_core`/`output_core` roles, `send_at_round`) is even exercised. If `batch1` is single-core, the tree-reduction path is inert and PCC error lies elsewhere (revisit Task 3 numerics).

- [ ] **Step 2: If multi-core reduction is active,** trace one round: compute writes O→`out_worker`; writer sends `out_worker`→parent's `intermed_out` scratchpad at `block_offset`; parent reads back via `intermed_base + block_offset`. Verify `block_offset`, `tile_bytes_intermed` (now `get_tile_size(dfb_out_m)`), and `reducer_sem` step encoding match the pre-split semantics. Add `TT_METAL_DPRINT_CORES` on the reducer core to compare O/M/L against a WH reference run if needed.

- [ ] **Step 3: Fix any mismatch, re-run, commit.**

### Task 11: Iterate to PCC

- [ ] **Step 1: Loop** — `$SIMRUN "<sentinel>"` → classify (Task 6 catalog) → apply matching fix → commit — until PCC passes. Escalate to the user if a failure needs API-owner input (e.g. a `page_table`-sharded STOP the test unexpectedly hits, or a Quasar LLK gap outside SDPA).

- [ ] **Step 2: Confirm the sentinel passes**

Run: `$SIMRUN "<sentinel>"`
Expected: `1 passed`, PCC above the test's threshold.

---

## PHASE 3 — Validate & package

### Task 12: Sibling cases, doc, decision

- [ ] **Step 1: Run adjacent batch cases on the same path**

Run: `$SIMRUN "models/experimental/llama32_1b_quasar/tests/ops/test_scaled_dot_product_attention_decode.py"` (whole file; still one file = allowed). Note which parametrizations pass; any that hit the deferred `page_table`-sharded STOP are expected-fail, documented.

- [ ] **Step 2: Write a short bring-up report** in the op root (`QUASAR_CRAQSIM_BRINGUP.md`): what changed per phase, what passes, deferred items (`page_table`-sharded STOP; prefill SFPU swap as the cheap follow-on; emulator validation pending the other machine), and the WH n150 regression check status.

- [ ] **Step 3: WH regression check** (keep-WH-green constraint)

Run the fork's own WH sentinel on n150 (arch auto-detected per `project_env_setup`): `scripts/run_safe_pytest.sh models/experimental/llama32_1b_quasar/tests/ops/test_scaled_dot_product_attention_decode.py`
Expected: still passing on WH — the ARCH_QUASAR guards left the WH path unchanged.

- [ ] **Step 4: Decide commit/PR with the user** — draft branch is `gchoudhary/quasar/get-sdpa-sdpa_decode-functional-on-sim-or-emu`. Do not open a PR without explicit go-ahead.

---

## Self-Review notes

- **Spec coverage:** audit blocker #1 (SFPU LLK) → Task 3 (swap, not port); #2 (out_o multi-binding, decode self-loops) → Task 2 (stash) + verified; #3 (formats) → Task 4; #4 (runtime reduce/matmul/transpose) → Tasks 6–11 iterative; #5 (bare packer pair, DFB→NoC aliases) → Task 9. Semaphore-API build breakage (not in audit) → Task 2. `wait_front`/`read_tile_value` prereq (not in audit) → Task 1. Prefill's streaming `evil_set_write_ptr` and `WINDOWED` STOP are **out of scope** (decode-first; prefill is Phase-3 follow-on).
- **Type consistency:** `out_worker`/`out_o` roles, `intermed_out` scratchpad `unique_id`, and `read_tile_value` signature are used identically across Tasks 1, 2, 6, 10.
- **Known open decision:** Task 3 interception is fork-local helper bodies (chosen for WH-green + fork-hygiene); if execution finds the helpers too entangled, the fallback is an ARCH_QUASAR branch in the shared `api/compute/experimental/sdpa.h` front-end — a shared-header change requiring extra care.
