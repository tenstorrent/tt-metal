# PR #47994 Review Analysis — "[feature] Scratchpad for metal 2.0"

- **Branch:** `riverwu/scratchpad` → `main`
- **State:** OPEN (explicitly WIP)
- **URL:** https://github.com/tenstorrent/tt-metal/pull/47994

## Context

This PR adds a **Program-scope Scratchpad** resource to the experimental Metal 2.0 host API.
A `ScratchpadSpec` declares a per-node L1 (SRAM) region with Program lifetime; kernels bind it
via `KernelSpec::ScratchpadBinding`, and the device side accesses it through a new
`Scratchpad<T>` accessor whose base address is injected at enqueue time into a CRTA slot.

The CRTA buffer layout grows from **three** sections to **four**:
`[named | tensor bindings | scratchpads | varargs]`.

The PR is WIP: actual L1 allocation, name registration, and address resolution are stubbed
(`get_scratchpad_address` returns the sentinel `0xCAFE0000`). Only the spec / validation /
codegen / CRTA-injection plumbing is functional and exercised by tests.

**Comment sources:** 9 inline comments from `riverwuTT`, 2 inline from Copilot, plus Copilot's
review summary which flagged 3 additional gaps that fall outside the diff hunks (so they could
not be posted inline). **14 actionable items total.**

---

## 🔴 Category A — Correctness bugs (highest priority)

### A1 — `UpdateProgramRunArgs` vararg base omits scratchpad section
- **File:** [program_run_args.cpp:1194](tt_metal/impl/metal2_host_api/program_run_args.cpp#L1194)
- **Source:** Copilot (inline)
- **Detail:** `UpdateProgramRunArgs` computes the common-vararg base as
  `named CRTAs + tensor_binding_section_words`, but omits the new scratchpad section.
  `SetProgramRunArgs` now assembles the CRTA buffer as
  `[named | tensor bindings | scratchpads | varargs]`, and the JIT-generated
  `get_common_vararg` uses `crta_layout.vararg_section_offset` which *includes* the scratchpad
  words (set in `program_spec.cpp`). Consequently, for any kernel that has **both** scratchpad
  bindings and common varargs, the partial update writes the varargs one (or more) words too
  early — overwriting the scratchpad base-address slot and leaving the device reading varargs
  from the wrong offset.
- **Fix sketch:** Add the scratchpad section width
  (`kernel->scratchpad_binding_handles().size()`, one word each) to the vararg base computation
  on the Update path so it matches the Set path / `vararg_section_offset`.

### A2 — Emulator `get_common_vararg` base omits scratchpad words (same bug, emulation path)
- **File:** [emulated_program_runner.cpp:801](tt_metal/impl/emulation/emulated_program_runner.cpp#L801)
- **Source:** Copilot review summary (out-of-diff)
- **Detail:** The emulator's `get_common_vararg` base still assumes the old three-section layout,
  mirroring the A1 bug. Should be fixed together with A1 — same root cause.

### A3 — `Kernel::compute_hash()` does not hash scratchpad binding handles
- **File:** [kernel.cpp:552](tt_metal/impl/kernels/kernel.cpp#L552)
- **Source:** Copilot review summary (out-of-diff)
- **Detail:** `compute_hash()` hashes tensor binding handles but not scratchpad handles. Two
  kernels that differ *only* in their scratchpad bindings can therefore collide on the JIT cache
  key, so one would reuse the other's compiled artifact.
- **Fix sketch:** Fold `scratchpad_binding_handles()` into the hash alongside the tensor binding
  handles.

---

## 🟡 Category B — API / design decisions (need a deliberate call)

### B1 — Enforce single scratchpad binding; add AdvancedOption for multi-binding
- **File:** [scratchpad_spec.hpp:28](tt_metal/api/tt-metalium/experimental/metal2_host_api/scratchpad_spec.hpp#L28)
- **Source:** riverwuTT
- **Detail:** Restrict to a single binding for now, and expose multi-binding behind an
  AdvancedOption (matches the Metal 2.0 experimental "new knobs go on AdvancedOptions" pattern).

### B2 — Device `Scratchpad<T>` should be opaque, not a POD ✅ DONE
- **File:** [scratchpad.h:11](tt_metal/hw/inc/api/scratchpad.h#L11)
- **Source:** riverwuTT
- **Detail:** Other device-side accessors behave like opaque classes rather than exposing their
  internals as plain data. The comment landed on `scratchpad.h:19` — i.e. on the
  **`ScratchpadAccessor` binding token** (the POD with public `crta_offset` / `size_in_bytes`),
  not on `Scratchpad<T>` itself (which was already a class with private members).
- **Resolution:** Reshaped `ScratchpadAccessor` into an opaque class matching the codebase's
  accessor-token convention (`DFBAccessor`, `TensorAccessorBindingToken`):
  - Data members are now private (`crta_offset_`, `size_in_bytes_`).
  - Public surface is a single `explicit constexpr ScratchpadAccessor(uint32_t, uint32_t) noexcept`.
  - `Scratchpad<T>` is declared a `friend` so it reads the internals without them being public.
  - Doc comment rewritten to mirror `DFBAccessor`'s opaque-handle style (handle line →
    "user will never directly interact with this type" → host/kernel flow → usage example →
    constexpr note).
- **No codegen change required:** the two emit sites
  ([genfiles.cpp:196](tt_metal/jit_build/genfiles.cpp#L196),
  [emulated_program_runner.cpp:780](tt_metal/impl/emulation/emulated_program_runner.cpp#L780))
  produce `constexpr ScratchpadAccessor name{crta_offset, size_in_bytes}`, which is
  direct-list-initialization and calls the new 2-arg explicit constexpr ctor unchanged.
- **Testing:**
  - Build: `./build_metal.sh -c -e --build-tests` (target `unit_tests_api`).
  - Spec/codegen coverage (confirms the generated `constexpr ScratchpadAccessor name{...}` still
    compiles against the opaque class):
    `./build/test/tt_metal/unit_tests_api --gtest_filter='ScratchpadSpecTest.*'`
  - Device dispatch round-trip (local 4-chip Wormhole), validates the friend-access path
    end-to-end via `Scratchpad<int32_t> pad(scratch::scratch)`:
    `./build/test/tt_metal/unit_tests_api --gtest_filter='ProgramSpecHWTest.ScratchpadAccessorDispatch'`
  - Pass criterion: filters build with no aggregate-init breakage and the HW dispatch test reports
    the expected scratchpad readback.

### B3 — Consider size as a template parameter (à la `std::span`)
- **File:** [scratchpad.h:21](tt_metal/hw/inc/api/scratchpad.h#L21)
- **Source:** riverwuTT
- **Detail:** Inject the region size as a template parameter so it is part of the type, similar
  to a fixed-extent `std::span<T, N>`.

### B4 — "Is this desired?" — open question on current behavior
- **File:** [scratchpad.h:38](tt_metal/hw/inc/api/scratchpad.h#L38)
- **Source:** riverwuTT
- **Detail:** Self-flagged question about whether the behavior at this line is intended. Needs a
  decision before resolving.

---

## 🟢 Category C — Scope cleanup (pull out of this PR)

### C1 — Remove `constexpr` changes from `CoreLocalMem` ✅ DONE
- **File:** [core_local_mem.h:1](tt_metal/hw/inc/api/core_local_mem.h#L1)
- **Source:** riverwuTT
- **Detail:** The `constexpr` additions should be removed because:
  1. They are outside the scope of this PR.
  2. They bake in the assumption that `CoreLocalMem` stores a plain `uint` address rather than a
     `T*` (where the reinterpret cast would live) — a design question that should not be settled
     in a drive-by change.
  They were only added to quickly verify `scratchpad_spec` during development.
- **Resolution:** Reverted the header to main (`git checkout main -- core_local_mem.h`); it now
  has zero diff vs main (dropped `constexpr` on the address ctor, copy ctor/assign, `get_address()`,
  the comparison operators, and `operator bool`).
- **Knock-on cleanup (in `scratchpad.h`):** removing the `CoreLocalMem` constexpr path forced
  dropping `constexpr` from the `Scratchpad` members that route through it — the
  `Scratchpad(pointer, size_type)` ctor, `size()`, `size_in_bytes()`, `get_base_addr()`, and
  `begin()/end()`. `ScratchpadAccessor`'s `explicit constexpr` ctor is retained (it only sets two
  `uint32_t`s and the codegen requires a constexpr token). A `// constexpr note:` after `private:`
  records which members could regain `constexpr` if `CoreLocalMem<T>` ever gains constexpr
  construction/copy and a constexpr `get_address()`.
- **Knock-on cleanup (test):** the `static_assert` constexpr block embedded in
  `ScratchpadConfiguredEmptyKernelCompiles`' kernel source
  ([test_program_spec.cpp ~3842](tests/tt_metal/tt_metal/api/metal2_host_api/test_program_spec.cpp#L3842))
  exercised the removed constexpr path and no longer compiles; the block was stripped (test
  otherwise intact). Verified via grep that this was the only other branch site relying on the
  reverted constexpr.

---

## 🔵 Category D — Test cleanup

### D1 — Tests should follow the testing file structure
- **File:** [test_program_spec.cpp:3824](tests/tt_metal/tt_metal/api/metal2_host_api/test_program_spec.cpp#L3824)
- **Source:** riverwuTT
- **Detail:** Relocate / restructure these tests to conform to the established test file layout.

### D2 — Remove the mock-device test that doesn't work
- **File:** [test_program_spec.cpp:3880](tests/tt_metal/tt_metal/api/metal2_host_api/test_program_spec.cpp#L3880)
- **Source:** riverwuTT
- **Detail:** This test does not function because the underlying device is a mock device; remove it.

### D3 — Stale test name `KernelCrtaLayout_AllThreeSectionsConsistent`
- **Source:** Copilot review summary (out-of-diff)
- **Detail:** The test name references "ThreeSections", but the CRTA layout now has **four**
  sections. Rename to reflect the current layout.

---

## ❓ Category F — Unresolved / needs investigation

### F1 — "???" left at own code
- **File:** [program_spec.cpp:2341](tt_metal/impl/metal2_host_api/program_spec.cpp#L2341)
- **Source:** riverwuTT
- **Detail:** A self-flagged "???" indicating confusion / a suspected issue. Needs investigation
  to determine what is wrong and what the intended behavior should be.

---

## Suggested order of attack

1. **Category A (bugs)** — fix A1 + A2 together (same root cause), then A3.
2. **C1** — revert out-of-scope `constexpr` changes.
3. **Category D (tests)** — restructure (D1), remove dead test (D2), rename stale test (D3).
4. **Category E (nits)** — fast, low-risk.
5. **Category B + F** — design calls / investigation that may warrant discussion before coding.
