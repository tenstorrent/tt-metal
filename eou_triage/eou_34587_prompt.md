# Issue tt-metal#34587 — Pack tilize/untilize bools → `PackMode` enum

> **Audience:** implementor (human or agent) taking on this refactor.
> **Author:** mentor handoff. Read this top-to-bottom once before touching code.
> **Links:**
> - Issue: https://github.com/tenstorrent/tenstorrent/tt-metal/issues/34587 (use the live issue as the source of truth; the text below is a guide, not a substitute)
> - Attached plan on the issue: `PACK_MODE_REFACTORING_PLAN.md` — **read it, but do not follow it blindly.** It has stale paths and at least one arch-specific assumption that doesn't hold. I call those out below.
> - Workspace repo clone: `/localdev/ncvetkovic/tt-metal/` (primary), `/localdev/ncvetkovic/temp/tt-metal/` (worktree — use this for the actual work so `main` stays clean)
> - Workspace `CLAUDE.md` has the canonical build/test/debug commands. When in doubt, trust `CLAUDE.md` over the attached plan.

---

## 1. What the issue is really about

Today, internal LLK pack functions are templated on **two booleans** — `untilize` and `tilize` — that select one of three modes of the Tensix packer. Example signature on Blackhole:

```cpp
template <bool untilize = false, bool zero_output = false, bool tilize = false>
inline void _llk_pack_mop_config_(...);
```

There are a few problems with this:

1. **The two bools encode three states, and the fourth — `untilize=true, tilize=true` — is nonsense.** Nothing in the type system prevents a caller from passing it. Today we rely on humans not writing it.
2. **Callsites like `_llk_pack_init_<false, false, false>(fmt)` are unreadable** — the reader has to remember the position of each bool. `_llk_pack_init_<PackMode::Default, false>(fmt)` reads for itself.
3. **The ODR-style enum-class concern** that is already pushing us to convert `DataCopyType` ([tt-llk#1148](https://github.com/tenstorrent/tt-llk/issues/1148)) applies here too: a typed enum scopes its values and is self-documenting at the callsite.

The refactor replaces the two bools with one `enum class PackMode { Default, Untilize, Tilize }` inside the internal LLK layer (the `_llk_*` functions with leading underscore, living in `tt_metal/tt-llk/<arch>/`). **The public compute kernel API stays on bools** — the wrapper layer (`tt_metal/hw/ckernels/<arch>/metal/llk_api/`) converts bool → enum before calling into LLK.

Why keep the public API on bools? Because changing it is a much bigger blast radius (every op kernel + every test kernel), and not what this issue is asking for. Keep the scope tight.

---

## 2. Architectural layers — memorize this picture

```
┌─────────────────────────────────────────────────────────────┐
│ Compute Kernel API                                          │
│   tt_metal/include/compute_kernel_api/                      │  ← STAYS ON BOOLS
│   e.g. llk_pack_hw_configure_disaggregated<DST_ACCUM, false>│    (untouched)
└───────────────────────┬─────────────────────────────────────┘
                        │ calls
┌───────────────────────▼─────────────────────────────────────┐
│ LLK API wrapper (per arch)                                  │
│   tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api│  ← TRANSLATES
│   bool untilize, bool tilize  ⟶  constexpr PackMode mode    │    bool → PackMode
└───────────────────────┬─────────────────────────────────────┘
                        │ calls
┌───────────────────────▼─────────────────────────────────────┐
│ Internal LLK                                                │
│   tt_metal/tt-llk/{blackhole,wormhole_b0}/llk_lib/          │  ← USES PackMode
│   tt_metal/tt-llk/{blackhole,wormhole_b0}/common/inc/       │    (the refactor
│   template <PackMode mode = PackMode::Default>              │     target)
└─────────────────────────────────────────────────────────────┘
```

The rule in one line: **if the function name starts with `_llk_` (underscore), it takes `PackMode`. If it starts with `llk_` (no underscore), it keeps booleans.**

---

## 3. Important corrections to the attached plan

I have verified these against the current repo on 2026-04-23. Trust this section over the attachment.

### 3.1 Path correction: `tt_metal/tt-llk/`, not `tt_metal/third_party/tt_llk/`

The attachment consistently uses `tt_metal/third_party/tt_llk/`. The submodule has since moved. Correct paths:

| Attachment says | Use instead |
|---|---|
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/ckernel_defs.h` | `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_defs.h` |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_pack.h` | `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h` |
| `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h` | `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h` |
| …and the same swap for `tt_llk_wormhole_b0/` | ditto |

### 3.2 Wormhole B0 does **not** have a `tilize` bool

This is the biggest ambiguity. On Blackhole you see pairs like:

```cpp
// tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h
template <bool untilize = false, bool tilize = false>                 // L20
template <bool untilize = false, bool zero_output = false, bool tilize = false>  // L72, L365, L381
template <bool is_fp32_dest_acc_en, bool untilize = false, bool tilize = false>  // L347
```

On Wormhole B0 the same functions have **only `untilize`** — no `tilize` flag exists:

```cpp
// tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h
template <bool untilize = false>                            // L73, L228
template <bool untilize = false, bool zero_output = false>  // L108, L244, L264
template <DstSync Dst, bool is_fp32_dest_acc_en, bool untilize = false>  // L289
```

The attachment's "Phase 7: repeat phases 1–6 for WH" glosses over this. Two options — **please raise this in the issue before implementing** and pick one with @ncvetkovicTT:

- **Option A (my recommendation):** Add `PackMode` on WH too, but only `Default` and `Untilize` are legal values. Add a `static_assert(mode != PackMode::Tilize, "Tilize pack mode not supported on Wormhole B0")` in the WH helpers. Pro: same type lives in both arches, compute API stays architecture-agnostic, wrapper conversion logic is identical. Con: enum carries a value that some arches can't handle (but `static_assert` catches it at compile time).
- **Option B:** Leave WH on a single bool. Con: inconsistent naming across arches; compute API has to branch per-arch — not what we want.

Go with A unless there's a strong reason not to.

### 3.3 Quasar is effectively a stub — skip it

`tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_pack.h` is 75 lines and has zero occurrences of `untilize`/`tilize`. Don't refactor it as part of this issue. Quasar uninits/inits are tracked separately under [tt-metal#35819](https://github.com/tenstorrent/tt-metal/issues/35819).

### 3.4 Build command

Attachment says `make -j8 ARCH_NAME=blackhole`. That's wrong. Per workspace `CLAUDE.md`:

```bash
cd /localdev/ncvetkovic/temp/tt-metal
./create_venv.sh          # first time only
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
./build_metal.sh --build-tests
```

Arch is selected via environment, not make args.

### 3.5 Submodule workflow

Because `tt-llk` is a submodule, changes to files under `tt_metal/tt-llk/...` require a commit inside that submodule repo first, then the super-repo (`tt-metal`) updates its pointer. From workspace `CLAUDE.md`:

> 1. Make changes in tt_llk submodule first
> 2. Commit in tt_llk on the matching branch (only when explicitly asked)
> 3. Update main repo to point to new tt_llk commit
> 4. Commit in main repo (only when explicitly asked)

Do **not** commit or push without explicit approval.

---

## 4. Scope — what to change, what to leave alone

### In scope (change these)

1. **Add the enum and helpers:**
   - `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_defs.h` — add `enum class PackMode` alongside the existing `enum class` types (see existing ones around lines 54, 243, 252, 259 for placement convention)
   - `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h` — same enum (see 3.2 for the Tilize-on-WH question)
   - Optionally `pack_mode_helpers` namespace as the attachment suggests, but keep it minimal — just `validate_pack_mode<mode>()` is enough. The `is_untilize`/`is_tilize` helpers in the attachment are fine but not necessary if you just compare with `if constexpr (mode == PackMode::Untilize)` at the call sites.

2. **Refactor internal LLK signatures to take `PackMode mode` instead of two bools:**
   - **BH** — `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h` (5 signatures at lines 20, 72, 347, 365, 381)
   - **BH** — `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cpack_common.h` (2 signatures at lines 300, 568)
   - **WH** — `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h` (6 signatures visible, only one bool to convert — see 3.2)
   - **WH** — `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/cpack_common.h` (grep for the same patterns)

3. **Update the LLK API wrapper to translate bool → PackMode** and call internals with the enum:
   - `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_api.h` (grep shows ≥12 relevant templates)
   - `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_api.h`
   - Conversion pattern:
     ```cpp
     template <bool untilize = false, bool zero_output = false, bool tilize = false>
     void llk_pack_init(...) {
         constexpr PackMode mode = untilize ? PackMode::Untilize
                                 : tilize   ? PackMode::Tilize
                                            : PackMode::Default;
         _llk_pack_init_<mode, zero_output>(...);
     }
     ```
     Keep the wrapper's public bool signature unchanged so callers in `compute_kernel_api/` don't care.

4. **Internal `_llk_*` callsites** that are not part of the public compute API. This is a small set — mostly:
   - `tt_metal/tt-llk/tests/sources/*.cpp` (standalone LLK tests)
   - Any other place under `tt_metal/tt-llk/` that directly calls `_llk_pack_*<bool, bool, ...>`

   Run `grep -rn "_llk_pack.*<" tt_metal/tt-llk/` and audit each hit.

### Out of scope (do **not** touch)

- `tt_metal/include/compute_kernel_api/**` — the public compute API stays on bools.
- `tt_metal/tests/tt_metal/**/test_kernels/**` — test kernels use the public API, so they stay on bools.
- TTNN-level code (`ttnn/**`).
- `tt_llk_quasar/**` — see 3.3.
- Any experimental SDPA work under `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/experimental/` (note: experimental lives under the *old* path that's still active — verify whether that's been moved as well and flag it if not).

**The compile-time invariant you're aiming for:** after the refactor, `grep -rn "template.*bool untilize.*bool tilize" tt_metal/tt-llk/` returns zero. The same grep under `tt_metal/hw/ckernels/*/metal/llk_api/` should still return hits (wrappers keep bools). The same grep under `tt_metal/include/compute_kernel_api/` should still return hits (public API keeps bools).

---

## 5. Suggested execution order

Work **one architecture at a time** and commit at each phase boundary. Keep WH unchanged while BH is in flight so bisecting regressions stays simple.

1. **Phase A — Blackhole only:**
   1. Add `PackMode` to BH `ckernel_defs.h`.
   2. Refactor BH `cpack_common.h` (`set_packer_strides`, `configure_pack`).
   3. Refactor BH `llk_pack.h` internal functions (5 signatures).
   4. Update BH `llk_pack_api.h` wrapper to do bool→PackMode translation.
   5. Update any internal BH callsites under `tt_metal/tt-llk/` (tests + helpers).
   6. **Build:** `./build_metal.sh --build-tests` (BH should be default; if not, set the appropriate arch env var).
   7. **Smoke test:** run a subset of LLK tests per `CLAUDE.md` (§ Running Tests → tt-llk standalone tests).

2. **Phase B — Wormhole B0:**
   1. Decide the Option A / B question from §3.2 **before** writing any WH code.
   2. Mirror phase A on WH (remember: only `untilize` bool exists; `Tilize` enum value is BH-only if Option A).
   3. Build & smoke-test on WH.

3. **Phase C — Verify both arches in one pass:**
   1. Invariant greps from §4 "Out of scope" all pass.
   2. Full builds for both arches succeed with `--build-tests`.
   3. Targeted regression: run the nightly SDPA accuracy+determinism tests on BH (per workspace `CLAUDE.md`):
      ```bash
      rm -rf ~/.cache/tt-metal-cache/
      pytest -x tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy
      pytest -x tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism
      ```
      These exercise the pack path heavily and are the fastest high-signal gate.
   4. If a hang occurs: `pkill -9 -f pytest && tt-smi -r`.

4. **Phase D — CI gating:** push the branch and let APC (All Post Commit) + BPC (Blackhole Post Commit) run. These are the ground truth. Any failure blocks merge.

---

## 6. Things that will trip you up (common pitfalls)

1. **Template parameter default ordering.** If you put `PackMode mode` after a parameter that has a default (e.g. `bool zero_output = false`), C++ requires `mode` also have a default. Keep defaults at the end:
   ```cpp
   // good
   template <PackMode mode = PackMode::Default, bool zero_output = false>
   // bad — won't compile
   template <bool zero_output = false, PackMode mode>
   ```

2. **`if constexpr (untilize)` → `if constexpr (mode == PackMode::Untilize)`.** Straightforward. But watch for `if constexpr (untilize || tilize)` and `if constexpr (untilize ^ tilize)` patterns — e.g. `cpack_common.h:311` uses `(untilize ^ tilize)`. Translate each one faithfully; the XOR becomes `(mode == PackMode::Untilize || mode == PackMode::Tilize)` since `Default` is the only value where both were false.

3. **Forgetting to include the enum.** If the compiler says `PackMode not declared`, the file doesn't transitively include `ckernel_defs.h`. Add the include.

4. **Enum leakage.** `enum class` is the whole point — don't introduce `using enum PackMode;` at namespace scope, that defeats the readability win. Fully-qualify at callsites.

5. **"zero_output" is a separate bool, not part of PackMode.** It's orthogonal. Leave it alone.

6. **Compute-API backward compat.** Before merging, verify by greppping that **no file under `tt_metal/include/compute_kernel_api/` was modified.** If one was, undo that change unless you have a specific reason.

7. **Submodule pointer.** After editing anything under `tt_metal/tt-llk/`, `git status` in the super-repo will show the submodule is dirty. That's expected. Don't force-push or reset the submodule to escape this — follow §3.5.

8. **Don't skip pre-commit hooks.** `clang-format` + `black` run automatically. If they fail, fix the underlying issue, don't use `--no-verify`.

---

## 7. Success criteria (Definition of Done)

The issue can be closed when **all** of the following hold:

- [ ] `enum class PackMode { Default, Untilize, Tilize };` exists in both BH and WH `ckernel_defs.h` (with the Option A / B decision from §3.2 resolved and documented).
- [ ] All internal `_llk_pack_*` function templates in `tt_metal/tt-llk/{blackhole,wormhole_b0}/` take `PackMode mode` instead of `bool untilize` + `bool tilize`.
- [ ] `grep -rn "template.*bool untilize.*bool tilize" tt_metal/tt-llk/` returns zero hits.
- [ ] `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_pack_api.h` wrappers translate bool → `PackMode` and keep their public bool signatures unchanged.
- [ ] No file under `tt_metal/include/compute_kernel_api/` is modified.
- [ ] No file under `tt_metal/tests/tt_metal/**/test_kernels/**` is modified (only internal `tt_metal/tt-llk/tests/` callsites are).
- [ ] `./build_metal.sh --build-tests` succeeds for both BH and WH.
- [ ] Blackhole SDPA nightly accuracy + determinism pytest tests from workspace `CLAUDE.md` pass.
- [ ] APC + BPC CI are green on the PR branch.
- [ ] No new warnings introduced (`-Wall -Wextra` posture of the codebase; clang-format clean).
- [ ] Commit story is clear — one commit (or small, cohesive series) per phase from §5, with messages that reference `tt-metal#34587`.

---

## 8. Open questions to raise on the issue before coding

Post these on [tt-metal#34587](https://github.com/tenstorrent/tt-metal/issues/34587) and wait for sign-off from @ncvetkovicTT before starting Phase A:

1. **WH `Tilize` value:** Option A (shared enum with compile-time `static_assert` on WH) or Option B (WH stays on single bool)? See §3.2.
2. **`pack_mode_helpers` namespace:** keep it (matches attachment) or inline the comparisons (simpler)? Not a blocker, but worth confirming before sprinkling a new namespace across the codebase.
3. **Experimental LLK pack code** under `.../llk_lib/experimental/` for BH — include in scope or exclude? (SDPA perf work is in flight there; co-locating a refactor there could create merge pain.)

---

## 9. Reference commands cheat-sheet

```bash
# Repo (use the worktree, keep main clean)
cd /localdev/ncvetkovic/temp/tt-metal

# Env
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)

# Discovery
grep -rn "inline void _llk_pack" tt_metal/tt-llk/tt_llk_blackhole/
grep -rn "_llk_pack.*<.*false" tt_metal/tt-llk/ tt_metal/hw/ckernels/

# Build
./build_metal.sh --build-tests

# Invariant check after refactor (should all be empty on tt-llk/)
grep -rn "template.*bool untilize.*bool tilize" tt_metal/tt-llk/
grep -rn "bool untilize.*bool tilize" tt_metal/tt-llk/tt_llk_blackhole/llk_lib/
grep -rn "bool untilize.*bool tilize" tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/

# Smoke tests (BH)
rm -rf ~/.cache/tt-metal-cache/
pytest -x tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_accuracy
pytest -x tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py::test_sdpa_determinism

# On hang
pkill -9 -f pytest && tt-smi -r
```

---

## 10. If you get stuck

- Re-read the attached `PACK_MODE_REFACTORING_PLAN.md` with §3 corrections in mind — it has useful concrete find/replace examples for the conditional-translation mechanics.
- The [tt-llk#1148](https://github.com/tenstorrent/tt-llk/issues/1148) `DataCopyType` enum-class refactor is the closest cousin; cross-reference once its PR lands.
- Ping @ncvetkovicTT on the issue with a specific question — "I'm blocked on X because Y" is much more useful than "I'm stuck."
