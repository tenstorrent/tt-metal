# Issue tt-metal#43036 — Make `PoolType` an `enum class` (Blackhole + Wormhole B0)

> **Audience:** implementor (human or agent) taking on this refactor.
> **Author:** mentor handoff. Read top-to-bottom once before touching code.
> **Issue:** https://github.com/tenstorrent/tt-metal/issues/43036
> **Repo note:** `tt_metal/tt-llk/` is a **plain directory** inside the `tt-metal` monorepo —
> not a submodule. Edit files in-place. The PR goes to `tenstorrent/tt-metal`.

---

## 1. What the issue is really about

`PoolType` in Blackhole and Wormhole B0 is declared as a plain (unscoped) `enum`:

```cpp
// tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h (line 37)
// tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h (line 37)
enum PoolType
{
    SUM,
    AVG,
    MAX,
    MIN,
};
```

Plain enums inject `SUM`, `AVG`, `MAX`, and `MIN` into the `ckernel` namespace. `MAX` and `MIN` in particular are extremely generic names that are very likely to collide with standard library macros or other definitions — making this the highest-ODR-risk enum in the list.

**The fix:** Convert BH and WH `PoolType` to `enum class`, then qualify every bare `SUM`/`AVG`/`MAX`/`MIN` reference **that is being used as a `PoolType`**.

**Quasar has this right**, but with **fewer enumerators**:

```cpp
// tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h
enum class PoolType : std::uint8_t
{
    SUM,
    AVG,
    MAX,
};
```

**CRITICAL arch difference:** Quasar does **not** have `MIN`. Blackhole and Wormhole B0 do. **Do not remove `MIN` from the BH/WH enum.** The target declaration for BH and WH is:

```cpp
enum class PoolType : std::uint8_t
{
    SUM,
    AVG,
    MAX,
    MIN,
};
```

Only the `enum` → `enum class : std::uint8_t` header changes; the enumerator list stays the same.

---

## 2. Scope — find every bare callsite before editing

**Warning:** `SUM`, `AVG`, `MAX`, and `MIN` are extremely common identifiers. A naive full-repo grep will return thousands of false positives from unrelated code (C++ `std::max`, Python test utilities, etc.). Narrow the search:

```bash
cd /localdev/ncvetkovic/work/tt-metal

# Files with the enum definition to convert (these are the ground truth)
grep -rn "enum PoolType" tt_metal/tt-llk/

# Find callsites — ONLY in LLK and compute paths where PoolType is actually used
# Narrow to the directories you know use it; avoid broad matches
grep -rn "\bSUM\b\|\bAVG\b\|\bMAX\b\|\bMIN\b" \
  tt_metal/tt-llk/ tt_metal/hw/ \
  --include="*.h" --include="*.cpp" \
  | grep -v "PoolType::\|ckernel::PoolType\|//\|\.py\|\.md\|tt_llk_quasar\|std::max\|std::min"
```

If the above still has noise, restrict to specific subdirectories:
```bash
grep -rn "\bSUM\b\|\bAVG\b\|\bMAX\b\|\bMIN\b" \
  tt_metal/tt-llk/tt_llk_blackhole/llk_lib/ \
  tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/ \
  tt_metal/hw/ckernels/ \
  tt_metal/hw/inc/api/compute/ \
  --include="*.h" \
  | grep -v "PoolType::\|ckernel::PoolType\|//\|std::"
```

Study the output carefully — not every hit of `MAX` or `MIN` is a `PoolType` enumerator. Look at the surrounding context to confirm it is a `PoolType` usage before editing.

Expected files to check:
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_reduce.h` — `PoolType` as template param; `if constexpr (type == SUM)` etc.
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_reduce.h` — same
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_reduce_api.h` — wrapper
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_reduce_api.h` — wrapper
- `tt_metal/hw/inc/api/compute/reduce.h` — public compute API

---

## 3. Qualification rules

Files **inside** `namespace ckernel` (or TU with `using namespace ckernel` in scope):
```cpp
// before
if constexpr (pool_type == MAX)
// after
if constexpr (pool_type == PoolType::MAX)
```

Files **outside** `namespace ckernel`:
```cpp
// after
if constexpr (pool_type == ckernel::PoolType::MAX)
```

Before choosing the prefix, check:
```bash
grep -n "namespace ckernel\|using namespace ckernel" <file>
```

**Do not edit** any occurrence of `MAX` or `MIN` that is not a `PoolType` enumerator — e.g. `std::max`, `#define MAX(a,b)`, Python test assertions.

---

## 4. Branch and worktree setup

```bash
cd /localdev/ncvetkovic/work/tt-metal
git fetch origin main
git checkout -b ncvetkovic/43036-pooltype-enum-class FETCH_HEAD
git log --oneline origin/main..HEAD   # should be empty before edits
```

---

## 5. Suggested execution order

**Phase A — LLK layer (BH + WH)**
1. Edit `tt_llk_blackhole/llk_lib/llk_defs.h`: `enum PoolType` → `enum class PoolType : std::uint8_t` (keep `MIN`!)
2. Edit `tt_llk_wormhole_b0/llk_lib/llk_defs.h`: same
3. Edit `llk_math_reduce.h` for both arches: qualify bare `SUM`/`AVG`/`MAX`/`MIN` *as PoolType*

**Phase B — Propagate upward**
4. Fix wrapper files: `tt_metal/hw/ckernels/*/metal/llk_api/llk_math_reduce_api.h`
5. Fix compute API: `tt_metal/hw/inc/api/compute/reduce.h` (if any bare names)
6. Fix any TTNN / test / model kernel files found by the targeted grep

**Phase C — Verify**
```bash
./build_metal.sh --build-tests
```

For the invariant grep, use the same targeted version from section 2 (not a broad `MAX` grep):
```bash
# Should return zero hits after qualification
grep -rn "\bSUM\b\|\bAVG\b\|\bMAX\b\|\bMIN\b" \
  tt_metal/tt-llk/tt_llk_blackhole/llk_lib/ \
  tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/ \
  tt_metal/hw/ckernels/ \
  tt_metal/hw/inc/api/compute/ \
  --include="*.h" \
  | grep -v "PoolType::\|ckernel::PoolType\|//\|std::\|tt_llk_quasar"
# Expected: empty (or only non-PoolType hits that are correctly left alone)
```

---

## 6. Commit message

```
refactor(llk): convert PoolType from plain enum to enum class

Plain enum PoolType in Blackhole and Wormhole B0 injected SUM, AVG, MAX, and MIN
into the ckernel namespace. MAX and MIN are high-ODR-risk names that collide with
common macros and standard library identifiers. Convert to enum class (using Quasar
as the declaration pattern) and qualify all PoolType enumerator references. BH/WH
retain the MIN enumerator which Quasar does not support.

Resolves tt-metal#43036
```

---

## 7. PR creation

```bash
git push origin ncvetkovic/43036-pooltype-enum-class

gh pr create \
  --base main \
  --head ncvetkovic/43036-pooltype-enum-class \
  --title "refactor(llk): convert PoolType from plain enum to enum class" \
  --body "$(cat <<'PRBODY'
### Summary

`PoolType` in Blackhole and Wormhole B0 was a plain (unscoped) `enum`, injecting `SUM`, `AVG`,
`MAX`, and `MIN` into the `ckernel` namespace. `MAX` and `MIN` are extremely generic identifiers
that risk colliding with `#define MAX(a,b)`, `std::max`, and other common definitions — making
this the highest-ODR-risk enum in the plain-enum-to-enum-class series.

This PR converts `PoolType` to `enum class PoolType : std::uint8_t`, using Quasar as the
declaration pattern. All `PoolType` enumerator references in LLK internals, wrapper layers,
and compute API are qualified. Note: `MIN` is retained in the BH/WH enum since Quasar does
not support that pool type.

Closes tt-metal#43036 (PoolType sub-task)

### What's changed

- **`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h`** — `enum PoolType` → `enum class PoolType : std::uint8_t` (enumerator list unchanged, `MIN` kept)
- **`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h`** — same
- **`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_reduce.h`** — bare PoolType enumerators qualified
- **`tt_metal/hw/ckernels/*/metal/llk_api/llk_math_reduce_api.h`** — bare enumerators qualified
- **`tt_metal/hw/inc/api/compute/reduce.h`** — bare enumerators qualified (if any)

### What's intentionally unchanged

- **`tt_metal/tt-llk/tt_llk_quasar/`** — already `enum class PoolType`; no changes needed.
- **`MIN` enumerator** — kept in BH/WH because Quasar omits it for arch reasons. No value change.
- **`std::max`, `std::min`, `#define MAX/MIN`** — unrelated; not touched.
- **Other enums in `llk_defs.h`** — out of scope; separate sub-tasks of tt-metal#43036.

### Architecture-specific notes

- **Quasar** supports `{SUM, AVG, MAX}`. **Blackhole and Wormhole B0** additionally support `MIN`.

### Notes for reviewers

- `MAX` and `MIN` require careful grep scoping to avoid false positives from unrelated code.
  The targeted grep in the PR description is more reliable than a broad full-repo match.
- Do not remove `MIN` from BH/WH even though Quasar lacks it.
- Compile-time verification (targeted):
  ```bash
  grep -rn "\bSUM\b\|\bAVG\b\|\bMAX\b\|\bMIN\b" \
    tt_metal/tt-llk/tt_llk_blackhole/llk_lib/ \
    tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/ \
    tt_metal/hw/ckernels/ tt_metal/hw/inc/api/compute/ \
    --include="*.h" \
    | grep -v "PoolType::\|ckernel::PoolType\|//\|std::\|tt_llk_quasar"
  # Expected: zero PoolType hits (non-PoolType MAX/MIN occurrences OK to remain)
  ```

### Type of change

- [x] Refactoring

### Checklist

- [ ] [All post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/all-post-commit-workflows.yaml) CI passes
- [ ] [Blackhole Post commit](https://github.com/tenstorrent/tt-metal/actions/workflows/blackhole-post-commit.yaml) CI passes (if applicable)
- [ ] [Assert validation](https://github.com/tenstorrent/tt-llk/blob/main/docs/Introduction_to_asserts.h.md) Complied with assert doc (if applicable)
PRBODY
)"
```

---

## 8. Definition of Done

- [ ] `grep -rn "enum PoolType" tt_metal/tt-llk/` returns zero hits.
- [ ] Targeted invariant grep returns zero PoolType-unqualified hits.
- [ ] `./build_metal.sh --build-tests` succeeds (no `error: 'MAX' was not declared` type errors).
- [ ] PR open with description above (no placeholder text).
- [ ] `MIN` enumerator is still present in BH/WH `llk_defs.h`.
- [ ] No `std::max` / `std::min` / `#define MAX` were accidentally modified.
- [ ] No files under `tt_metal/tt-llk/tt_llk_quasar/` were modified.
- [ ] No other enums in `llk_defs.h` were touched.

---

## 9. Common pitfalls

1. **DO NOT remove `MIN`.** It exists in BH/WH but not Quasar. Only the `enum` → `enum class` header changes.

2. **MAX/MIN false positives.** `std::max`, `std::min`, `#define MAX`, and `#define MIN` are common and must not be touched. Read the context around every grep hit before editing.

3. **Broad grep gives misleading counts.** The ~245 callsite count in the backlog was a naive full-repo grep. The actual `PoolType` callsite count is much lower — use the targeted grep.

4. **Namespace mismatch.** LLK internal files use `PoolType::MAX`; wrapper/API files need `ckernel::PoolType::MAX`.

5. **Editing Quasar.** It's already correct; don't touch it.
