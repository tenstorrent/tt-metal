# Issue tt-llk#1148 — Make `DataCopyType` an `enum class`

> **Audience:** implementor (human or agent) taking on this refactor.
> **Author:** mentor handoff. Read top-to-bottom once before touching code.
> **Issue:** https://github.com/tenstorrent/tt-llk/issues/1148
> **Repo note:** `tt_metal/tt-llk/` is a **plain directory** inside the `tt-metal` monorepo —
> not a submodule. Edit files in-place. The PR goes to `tenstorrent/tt-metal`.

---

## 1. What the issue is really about

`DataCopyType` in Blackhole and Wormhole B0 is declared as a plain (unscoped) `enum`:

```cpp
// tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h (line 45)
// tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h (line 45)
enum DataCopyType
{
    A2D,
    B2D,
};
```

Plain enums inject their values (`A2D`, `B2D`) into the **enclosing namespace** — in this case
`ckernel`. That causes two concrete problems:

1. **ODR (One Definition Rule) violation risk.** If two translation units include different
   headers that each define `A2D` (or a symbol that collides with it), the linker gets confused.
   The tt-llk team was burned by exactly this pattern with another enum — hence the sub-issue series.

2. **Unqualified usage leaks everywhere.** Code can write `if constexpr (type == A2D)` instead
   of `if constexpr (type == DataCopyType::A2D)`. That's what's happening in the current BH/WH
   datacopy implementation files. With an `enum class` it won't compile — forcing clean,
   self-documenting callsites.

**The fix:** Convert BH and WH `DataCopyType` to `enum class`, then qualify every bare `A2D`/`B2D`
reference across the repo.

**Quasar already has this right** — it's the reference implementation:

```cpp
// tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h (line 26)
enum class DataCopyType : std::uint8_t
{
    A2D,
    B2D,
};
```

Match that declaration exactly for BH and WH.

---

## 2. Architectural layers involved

```
┌───────────────────────────────────────────────────────────────┐
│ TTNN compute kernels                                          │
│   ttnn/cpp/ttnn/operations/*/kernels/compute/*.cpp/.hpp       │  ← bare A2D/B2D → fix
└───────────────────┬───────────────────────────────────────────┘
                    │
┌───────────────────▼───────────────────────────────────────────┐
│ Compute API layer                                             │
│   tt_metal/hw/inc/api/compute/*.h                             │  ← bare A2D/B2D → fix
└───────────────────┬───────────────────────────────────────────┘
                    │
┌───────────────────▼───────────────────────────────────────────┐
│ LLK API wrapper (per arch)                                    │
│   tt_metal/hw/ckernels/{bh,wh_b0}/metal/llk_api/             │  ← uses DataCopyType as
│   llk_math_unary_datacopy_api.h                               │    template param — no change
└───────────────────┬───────────────────────────────────────────┘
                    │
┌───────────────────▼───────────────────────────────────────────┐
│ Internal LLK (the root cause)                                 │
│   tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/            │  ← enum → enum class
│   llk_lib/llk_defs.h                                          │    + qualify A2D/B2D usages
│   llk_lib/llk_math_eltwise_unary_datacopy.h                   │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Complete scope — every file that needs changing

Run these greps yourself before starting to verify nothing has moved:

```bash
# Files with the enum definition to convert
grep -rn "enum DataCopyType" tt_metal/tt-llk/

# Files with bare A2D/B2D in executable code (not comments, not Quasar)
grep -rn "\bA2D\b\|\bB2D\b" tt_metal/ ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "DataCopyType::\|ckernel::DataCopyType\|MOVB2D\|p_movb2d\|//\|\.py\|\.md\|tt_llk_quasar"
```

As of the time this prompt was written, the files that need editing are:

### 3.1 LLK internal layer (root cause — fix these first)

| File | What to change |
|---|---|
| `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h:45` | `enum DataCopyType` → `enum class DataCopyType : std::uint8_t` |
| `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h:45` | same |
| `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h` | 7 occurrences of bare `A2D`/`B2D` at lines 189, 193, 224, 297, 316, 393, 408 |
| `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_datacopy.h` | 6 occurrences at lines 160, 164, 188, 271, 290, 378, 382 |

Pattern: `type == A2D` → `type == DataCopyType::A2D`, `type == B2D` → `type == DataCopyType::B2D`.
These files are inside `namespace ckernel` so `DataCopyType::` (without `ckernel::`) is correct.

### 3.2 LLK API wrapper layer (no signature changes needed — just verify)

- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_unary_datacopy_api.h`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_unary_datacopy_api.h`

These use `DataCopyType type` as a template parameter type — that's fine with `enum class`.
The callsite values like `DataCopyType::A2D` in these files already work. **Verify they compile
but don't edit them unless the compiler forces you to.**

### 3.3 Compute API layer (bare A2D/B2D in executable code)

These files are **outside** `namespace ckernel`. Some already use `DataCopyType::A2D`, some
use `ckernel::DataCopyType::A2D`, and some use bare `A2D`. Only the bare ones break.
Check each file with `grep -n "\bA2D\b\|\bB2D\b"` and fix:

| File | Notes |
|---|---|
| `tt_metal/hw/inc/api/compute/bcast.h` | A2D/B2D as template args AND as a runtime `auto` variable at line 102 |
| `tt_metal/hw/inc/api/compute/pack_untilize.h` | A2D as template arg |
| `tt_metal/hw/inc/api/compute/tile_move_copy.h` | A2D as template arg |
| `tt_metal/hw/inc/api/compute/tilize.h` | A2D in a multi-line template argument list — check carefully |
| `tt_metal/hw/inc/api/compute/transpose_wh.h` | A2D as template arg |
| `tt_metal/hw/inc/api/compute/untilize.h` | A2D as template arg |
| `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h` | A2D as template arg |

For all template argument uses: `<A2D, ...>` → `<ckernel::DataCopyType::A2D, ...>` (use the
fully-qualified name since these files may not have `using namespace ckernel`; check each file).

For the runtime variable in `bcast.h` (line 102):
```cpp
// before
const auto data_copy_type = (new_bcast_type == BroadcastType::NONE) ? A2D : B2D;
const bool enable_unpack_to_dest = data_copy_type == A2D;
// after
const auto data_copy_type = (new_bcast_type == BroadcastType::NONE)
    ? ckernel::DataCopyType::A2D : ckernel::DataCopyType::B2D;
const bool enable_unpack_to_dest = data_copy_type == ckernel::DataCopyType::A2D;
```

### 3.4 Test kernels

These use the public `llk_*` API (not the `_llk_*` internals) but still pass bare `A2D`/`B2D`
as template arguments. They will break at compile time and must be fixed:

| File |
|---|
| `tests/tt_metal/tt_metal/test_kernels/compute/3T/matmul_large_block_zm/zm_3m_math.cpp` |
| `tests/tt_metal/tt_metal/test_kernels/compute/3T/untilize_A_and_eltwise_binary/chlkc_math.cpp` |
| `tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp` |
| `tests/tt_metal/tt_metal/test_kernels/compute/untilA_elwbin_3m.cpp` |

Pattern: `llk_math_eltwise_unary_datacopy_init<A2D, ...>` → `...<DataCopyType::A2D, ...>` or
`<ckernel::DataCopyType::A2D, ...>` depending on what namespace is in scope.

### 3.5 TTNN compute kernels

| File | Notes |
|---|---|
| `ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/ssm_eltwise_mul.cpp` | Two occurrences — bare `A2D` |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp` | One occurrence in a template list |

### 3.6 Model-specific kernel

| File | Notes |
|---|---|
| `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/custom_tilize.h` | One bare `A2D` |

### 3.7 Out of scope

- **`tt_metal/tt-llk/tt_llk_quasar/`** — already `enum class`; do not touch.
- **Comments** mentioning "A2D" or "B2D" in prose — leave them as-is.
- **Python test files** — `DataCopyType.A2D` in Python is a completely separate Python enum, unaffected.
- **Other enums in `llk_defs.h`** (e.g. `EltwiseBinaryType`, `ReduceDim`) — out of scope for this issue; do not touch.

---

## 4. Branch and worktree setup

`tt_metal/tt-llk/` is a plain directory in the `tt-metal` repo — no submodule, no separate repo
to push to. All work happens in a single branch on `tenstorrent/tt-metal`.

```bash
cd /localdev/ncvetkovic/work/tt-metal

# Pull latest main
git fetch origin main

# Create a clean branch from tip of main
git checkout -b ncvetkovic/1148-datacopytype-enum-class FETCH_HEAD

# Verify: the branch should be exactly 0 commits ahead before you start editing
git log --oneline origin/main..HEAD   # should be empty
```

Never base this branch on another in-flight feature branch. If you need a worktree:

```bash
git worktree add ../tt-metal-1148 ncvetkovic/1148-datacopytype-enum-class
```

---

## 5. Suggested execution order

Work in phases so each phase is independently buildable:

### Phase A — LLK layer (BH + WH)

1. Edit `tt_llk_blackhole/llk_lib/llk_defs.h`: `enum DataCopyType` → `enum class DataCopyType : std::uint8_t`
2. Edit `tt_llk_wormhole_b0/llk_lib/llk_defs.h`: same
3. Edit `tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h`: qualify all bare `A2D`/`B2D`
4. Edit `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_datacopy.h`: same
5. Build attempt — expect failures upstream; collect them before fixing them

### Phase B — Propagate upward (fix all build errors)

Work through the build errors one layer at a time:
- `tt_metal/hw/inc/api/compute/` files
- Test kernels
- TTNN kernels
- Model kernels

For each file: `grep -n "\bA2D\b\|\bB2D\b"` first, then fix, then re-grep to confirm zero bare names.

### Phase C — Verify

```bash
# Build (arch env selects BH/WH; run both)
./build_metal.sh --build-tests

# Invariant: should return zero hits (excluding comments and Quasar)
grep -rn "\bA2D\b\|\bB2D\b" tt_metal/ ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "DataCopyType::\|ckernel::DataCopyType\|MOVB2D\|p_movb2d\|//\|\.py\|\.md\|tt_llk_quasar"
# Expected output: empty

# LLK standalone tests
cd tt_metal/tt-llk/tests
pytest --compile-producer -n 8 -x python_tests/
pytest --compile-consumer -x python_tests/
cd -
```

If a test hang occurs: `pkill -9 -f pytest && tt-smi -r`

---

## 6. Commit message

```
refactor(llk): convert DataCopyType from plain enum to enum class

Plain enum DataCopyType in Blackhole and Wormhole B0 injected A2D and B2D
into the ckernel namespace, enabling unqualified usage that risks ODR violations
and makes callsites unreadable. Convert to enum class (matching Quasar) and
qualify all bare A2D/B2D references across the LLK, compute API, test kernel,
and TTNN layers.

Resolves tt-llk#1148
```

---

## 7. PR creation

```bash
cd /localdev/ncvetkovic/work/tt-metal
git push origin ncvetkovic/1148-datacopytype-enum-class

gh pr create \
  --base main \
  --head ncvetkovic/1148-datacopytype-enum-class \
  --title "refactor(llk): convert DataCopyType from plain enum to enum class" \
  --body "$(cat <<'PRBODY'
### Summary

`DataCopyType` in Blackhole and Wormhole B0 was a plain (unscoped) `enum`, which injected its
enumerators `A2D` and `B2D` directly into the `ckernel` namespace. This allowed — and in practice
caused — callsites throughout the LLK implementation, compute API, and compute kernel files to
write bare `A2D` and `B2D` without any qualifying type. Beyond readability, this is an ODR hazard:
if any two translation units include headers that both define a symbol named `A2D`, the linker
sees ambiguity. The tt-llk team was burned by exactly this pattern with another enum previously.

This PR converts `DataCopyType` to `enum class DataCopyType : std::uint8_t` in the Blackhole and
Wormhole B0 `llk_defs.h` files, matching the Quasar architecture which already had the correct
declaration. All bare `A2D` and `B2D` references across the LLK internals, compute API layer,
test kernels, and TTNN compute kernels are updated to use the fully-qualified
`DataCopyType::A2D` / `ckernel::DataCopyType::A2D` form.

Resolves [tt-llk#1148](https://github.com/tenstorrent/tt-llk/issues/1148)

### What's changed

**LLK internal layer (root cause)**
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h` — `enum DataCopyType` → `enum class DataCopyType : std::uint8_t`
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h` — same
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_datacopy.h` — 7 occurrences of bare `A2D`/`B2D` in `if constexpr` guards qualified to `DataCopyType::A2D`/`DataCopyType::B2D`
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_datacopy.h` — 6 occurrences, same treatment

**Compute API layer** (`tt_metal/hw/inc/api/compute/`)
- `bcast.h`, `pack_untilize.h`, `tile_move_copy.h`, `tilize.h`, `transpose_wh.h`, `untilize.h`, `eltwise_unary/eltwise_unary.h` — bare `A2D`/`B2D` template arguments and runtime comparisons qualified to `ckernel::DataCopyType::A2D`/`B2D`

**Test kernels** (`tests/tt_metal/tt_metal/test_kernels/compute/`)
- `3T/matmul_large_block_zm/zm_3m_math.cpp`, `3T/untilize_A_and_eltwise_binary/chlkc_math.cpp`, `eltwise_copy_3m.cpp`, `untilA_elwbin_3m.cpp` — bare `A2D` template arguments qualified

**TTNN compute kernels**
- `ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul/device/kernels/ssm_eltwise_mul.cpp`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`

**Model kernel**
- `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/custom_tilize.h`

### What's intentionally unchanged

- **`tt_metal/tt-llk/tt_llk_quasar/`** — already `enum class DataCopyType : std::uint8_t`; no changes needed.
- **LLK API wrapper files** (`tt_metal/hw/ckernels/*/metal/llk_api/llk_math_unary_datacopy_api.h`) — use `DataCopyType` as a template type parameter which is compatible with both `enum` and `enum class`; verified they compile without changes.
- **Other enums in `llk_defs.h`** (e.g. `EltwiseBinaryType`, `ReduceDim`) — out of scope; left as plain `enum` for a future clean-up issue.
- **Python test infrastructure** — `DataCopyType.A2D` in Python is an independent Python enum object, completely unaffected.
- **Prose comments** mentioning "A2D" or "B2D" — unchanged; only executable code references were updated.

### Notes for reviewers

- The reference for the target declaration is `tt_llk_quasar/llk_lib/llk_defs.h:26` — the Quasar arch already had this right.
- The internal `llk_math_eltwise_unary_datacopy.h` files for BH and WH use `using namespace ckernel`, so `DataCopyType::A2D` (without `ckernel::`) is correct there. Files in `tt_metal/hw/inc/api/compute/` are outside that namespace and need `ckernel::DataCopyType::A2D`.
- The `bcast.h` change at the runtime variable is slightly non-obvious: `auto` resolves correctly (both sides of the ternary are now `ckernel::DataCopyType`), but the bare `A2D` comparison on the next line also needed qualification.
- Compile-time invariant verifying completeness:
  ```bash
  grep -rn "\bA2D\b\|\bB2D\b" tt_metal/ ttnn/ tests/ models/ \
    --include="*.h" --include="*.cpp" --include="*.hpp" \
    | grep -v "DataCopyType::\|ckernel::DataCopyType\|MOVB2D\|p_movb2d\|//\|\.py\|\.md\|tt_llk_quasar"
  # Expected: zero hits
  ```

### Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [x] Refactoring

### Checklist

- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.md) Complied with assert doc (if applicable)
PRBODY
)"
```

---

## 8. Definition of Done

- [ ] `grep -rn "enum DataCopyType" tt_metal/tt-llk/` returns zero hits (all converted to `enum class`).
- [ ] The invariant grep from section 5 Phase C returns zero hits.
- [ ] `./build_metal.sh --build-tests` succeeds for both Blackhole and Wormhole B0.
- [ ] LLK standalone test suite passes.
- [ ] PR open with description matching section 7 (no placeholder text remaining).
- [ ] No files under `tt_metal/tt-llk/tt_llk_quasar/` were modified.
- [ ] No other enums in `llk_defs.h` were touched.
- [ ] No new compiler warnings introduced.

---

## 9. Common pitfalls

1. **Namespace mismatch.** Files inside `namespace ckernel { using namespace ckernel; }` blocks use
   `DataCopyType::A2D`. Files outside `namespace ckernel` need `ckernel::DataCopyType::A2D`.
   Check the namespace context before picking which prefix to use.

2. **`auto` in `bcast.h`.** The `auto` on the runtime variable is fine — `auto` will deduce
   `ckernel::DataCopyType` once both sides of the ternary use qualified names. Do not change
   `auto` to an explicit type unless the compiler demands it.

3. **Forgetting the `: std::uint8_t` underlying type.** The Quasar definition has it
   (`enum class DataCopyType : std::uint8_t`). Match it exactly in BH and WH.

4. **Grepping only tt-llk.** The bare names appear in `tt_metal/hw/`, `tests/`, `ttnn/`, and
   `models/` too. Always run the full-repo invariant grep before declaring done.

5. **Editing Quasar.** It's already correct. Touching it risks introducing a regression.

6. **Stacking on the wrong base.** Always branch from `FETCH_HEAD` of `origin/main`, not from any
   local branch that may carry unrelated eou_triage or ai-agents commits.
