# Issue tt-metal#43036 — Make `BroadcastType` an `enum class` (Blackhole + Wormhole B0)

> **Audience:** implementor (human or agent) taking on this refactor.
> **Author:** mentor handoff. Read top-to-bottom once before touching code.
> **Issue:** https://github.com/tenstorrent/tt-metal/issues/43036
> **Repo note:** `tt_metal/tt-llk/` is a **plain directory** inside the `tt-metal` monorepo —
> not a submodule. Edit files in-place. The PR goes to `tenstorrent/tt-metal`.

---

## 1. What the issue is really about

`BroadcastType` in Blackhole and Wormhole B0 is declared as a plain (unscoped) `enum`:

```cpp
// tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h (line 73)
// tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h (line 73)
enum BroadcastType
{
    NONE   = 0x0, // A - None || B - None
    COL    = 0x1, // A - None || B - Col Broadcast
    ROW    = 0x2, // A - None || B - Row Broadcast
    SCALAR = 0x3, // A - None || B - Scalar Broadcast
};
```

Plain enums inject `NONE`, `COL`, `ROW`, and `SCALAR` into the `ckernel` namespace. `NONE` in particular is an extraordinarily common identifier — a collision with it is virtually guaranteed in a codebase this size. This is the **largest** refactor in the enum class series by callsite count.

**The fix:** Convert BH and WH `BroadcastType` to `enum class`, then qualify every bare `NONE`/`COL`/`ROW`/`SCALAR` reference **that is a `BroadcastType` enumerator**.

**Quasar already has this right** — it is the reference implementation:

```cpp
// tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h
// Broadcasts only occur on SrcB
enum class BroadcastType : std::uint8_t
{
    NONE,
    COL,
    ROW,
    SCALAR,
};
```

The BH/WH declaration has explicit hex values (`= 0x0`, `= 0x1`, `= 0x2`, `= 0x3`). Keep them — they are the same values Quasar's implicit 0/1/2/3 produce, but keeping them explicit documents the HW encoding and avoids surprise if someone reorders the list:

```cpp
enum class BroadcastType : std::uint8_t
{
    NONE   = 0x0,
    COL    = 0x1,
    ROW    = 0x2,
    SCALAR = 0x3,
};
```

Preserve the inline comments (`// A - None || B - None` etc.) — they document HW semantics.

---

## 2. Scope — find every bare callsite before editing

**Warning:** `NONE`, `COL`, `ROW`, and `SCALAR` are extremely common identifiers. A naive full-repo grep will return enormous noise. Use targeted scoping:

```bash
cd /localdev/ncvetkovic/work/tt-metal

# Files with the enum definition to convert
grep -rn "enum BroadcastType" tt_metal/tt-llk/

# Targeted grep — restrict to LLK/compute directories first
grep -rn "\bNONE\b\|\bCOL\b\|\bROW\b\|\bSCALAR\b" \
  tt_metal/tt-llk/tt_llk_blackhole/llk_lib/ \
  tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/ \
  tt_metal/hw/ckernels/ \
  tt_metal/hw/inc/api/compute/ \
  --include="*.h" \
  | grep -v "BroadcastType::\|ckernel::BroadcastType\|//\|tt_llk_quasar"

# Then expand to TTNN and tests if needed
grep -rn "\bNONE\b\|\bCOL\b\|\bROW\b\|\bSCALAR\b" \
  ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "BroadcastType::\|ckernel::BroadcastType\|//\|\.py\|\.md\|tt_llk_quasar\|nullptr\|None\b"
```

Study the output carefully. `NONE` in particular will match unrelated code (`enum class Foo { NONE, ... }` in other files, `BroadcastType::NONE` already qualified, etc.). Only edit occurrences where the surrounding context makes clear this is a `BroadcastType` enumerator.

Expected files to check:
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary_sfpu.h` — `if constexpr (bcast_type == NONE)` guards
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h` — bcast type dispatch
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB.h` — bcast unpack dispatch
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary*.h` — same patterns
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_AB.h` — same
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_eltwise_binary_api.h` — wrapper
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_eltwise_binary_api.h` — wrapper
- `tt_metal/hw/inc/api/compute/bcast.h` — public compute API (already fixed DataCopyType here; check for bare BroadcastType enumerators)
- `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` — if it exists
- TTNN compute kernels for bcast/matmul/binary operations

**Note:** `bcast.h` already uses `BroadcastType::NONE` in the `constexpr auto data_copy_type` expression (fixed as part of the DataCopyType refactor). Verify those lines are already qualified before assuming they are bare.

---

## 3. Qualification rules

Files **inside** `namespace ckernel` (or TU with `using namespace ckernel` in scope):
```cpp
// before
if constexpr (bcast_type == NONE)
// after
if constexpr (bcast_type == BroadcastType::NONE)
```

Files **outside** `namespace ckernel`:
```cpp
// after
if constexpr (bcast_type == ckernel::BroadcastType::NONE)
```

Before choosing the prefix, check:
```bash
grep -n "namespace ckernel\|using namespace ckernel" <file>
```

**Do not edit** occurrences of `NONE`, `ROW`, `COL`, `SCALAR` that belong to other enums or are unrelated to `BroadcastType`.

---

## 4. Branch and worktree setup

```bash
cd /localdev/ncvetkovic/work/tt-metal
git fetch origin main
git checkout -b ncvetkovic/43036-broadcasttype-enum-class FETCH_HEAD
git log --oneline origin/main..HEAD   # should be empty before edits
```

---

## 5. Suggested execution order

**Phase A — LLK layer (BH + WH)**
1. Edit `tt_llk_blackhole/llk_lib/llk_defs.h`: `enum BroadcastType` → `enum class BroadcastType : std::uint8_t` (keep hex values and comments)
2. Edit `tt_llk_wormhole_b0/llk_lib/llk_defs.h`: same
3. Edit `llk_math_eltwise_binary*.h` and `llk_unpack_AB.h` for both arches: qualify bare enumerators

**Phase B — Build and collect errors**
4. Attempt `./build_metal.sh --build-tests` and collect all compilation errors from bare enum names
5. Fix each file the compiler flags: wrapper files, compute API, TTNN kernels, test kernels

**Phase C — Verify**
```bash
./build_metal.sh --build-tests

# Targeted invariant (LLK + compute paths)
grep -rn "\bNONE\b\|\bCOL\b\|\bROW\b\|\bSCALAR\b" \
  tt_metal/tt-llk/tt_llk_blackhole/llk_lib/ \
  tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/ \
  tt_metal/hw/ckernels/ \
  tt_metal/hw/inc/api/compute/ \
  --include="*.h" \
  | grep -v "BroadcastType::\|ckernel::BroadcastType\|//\|tt_llk_quasar"
# Expected: empty
```

**Strategy tip:** For this refactor, letting the compiler guide you is more reliable than chasing grep hits. After Phase A, run the build and fix every compile error before doing another grep sweep.

---

## 6. Commit message

```
refactor(llk): convert BroadcastType from plain enum to enum class

Plain enum BroadcastType in Blackhole and Wormhole B0 injected NONE, COL,
ROW, and SCALAR into the ckernel namespace. NONE in particular collides with
identifiers from numerous other scopes. Convert to enum class (matching Quasar)
and qualify all BroadcastType enumerator references across LLK internals,
wrapper layers, compute API, and TTNN compute kernels.

Resolves tt-metal#43036
```

---

## 7. PR creation

```bash
git push origin ncvetkovic/43036-broadcasttype-enum-class

gh pr create \
  --base main \
  --head ncvetkovic/43036-broadcasttype-enum-class \
  --title "refactor(llk): convert BroadcastType from plain enum to enum class" \
  --body "$(cat <<'PRBODY'
### Summary

`BroadcastType` in Blackhole and Wormhole B0 was a plain (unscoped) `enum`, injecting
`NONE`, `COL`, `ROW`, and `SCALAR` into the `ckernel` namespace. `NONE` is the most
collision-prone identifier in the entire `llk_defs.h` — it appears in countless other
enums and definitions across the codebase. This is the largest single refactor in the
plain-enum-to-enum-class series by callsite count.

This PR converts `BroadcastType` to `enum class BroadcastType : std::uint8_t`, matching
Quasar. The explicit HW-encoding values (`= 0x0` through `= 0x3`) and inline comments are
preserved. All `BroadcastType` enumerator references across LLK internals, wrapper layers,
compute API, and TTNN compute kernels are updated to use the fully-qualified form.

Closes tt-metal#43036 (BroadcastType sub-task)

### What's changed

- **`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h`** — `enum BroadcastType` → `enum class BroadcastType : std::uint8_t` (hex values and comments preserved)
- **`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h`** — same
- **`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_binary*.h`** — bare enumerators qualified
- **`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_unpack_AB.h`** — bare enumerators qualified
- **`tt_metal/hw/ckernels/*/metal/llk_api/llk_math_eltwise_binary_api.h`** — bare enumerators qualified
- **`tt_metal/hw/inc/api/compute/bcast.h`** — bare enumerators qualified (if any remain after DataCopyType refactor)
- Additional TTNN / test / model kernel files as found by compiler errors

### What's intentionally unchanged

- **`tt_metal/tt-llk/tt_llk_quasar/`** — already `enum class BroadcastType`; no changes needed.
- **Other enums' `NONE` enumerators** — not touched; only `BroadcastType::NONE` is in scope.
- **Other enums in `llk_defs.h`** — out of scope; separate sub-tasks of tt-metal#43036.
- **Hex values** — retained (`0x0`, `0x1`, `0x2`, `0x3`) for HW documentation.

### Notes for reviewers

- This is the highest-callsite refactor in the series. The compiler is your best guide after
  Phase A — let build errors drive the remaining file list rather than relying solely on grep.
- `NONE` appears in many other contexts (other enums, `EltwiseBinaryReuseDestType::NONE` which
  is already an `enum class`, etc.) — verify each hit is actually a `BroadcastType` context.
- The `bcast.h` file already had `BroadcastType::NONE` from the DataCopyType refactor (the
  `constexpr auto data_copy_type` comparison). Verify those lines are already qualified.
- Targeted compile-time invariant (LLK + compute paths only):
  ```bash
  grep -rn "\bNONE\b\|\bCOL\b\|\bROW\b\|\bSCALAR\b" \
    tt_metal/tt-llk/tt_llk_blackhole/llk_lib/ \
    tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/ \
    tt_metal/hw/ckernels/ tt_metal/hw/inc/api/compute/ \
    --include="*.h" \
    | grep -v "BroadcastType::\|ckernel::BroadcastType\|//\|tt_llk_quasar"
  # Expected: zero hits from BroadcastType contexts
  ```

### Type of change

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

- [ ] `grep -rn "enum BroadcastType" tt_metal/tt-llk/` returns zero hits.
- [ ] Targeted invariant grep returns zero BroadcastType-unqualified hits in LLK/compute paths.
- [ ] `./build_metal.sh --build-tests` succeeds with zero new compiler errors.
- [ ] PR open with description above (no placeholder text).
- [ ] Hex values and inline comments in BH/WH `llk_defs.h` are preserved.
- [ ] No files under `tt_metal/tt-llk/tt_llk_quasar/` were modified.
- [ ] No other enums' `NONE`/`ROW`/`COL`/`SCALAR` enumerators were accidentally qualified under a wrong type.
- [ ] No other enums in `llk_defs.h` were touched.

---

## 9. Common pitfalls

1. **`NONE` appears everywhere.** Read the surrounding context before every edit. `EltwiseBinaryReuseDestType::NONE` is already an enum class and must not be touched. Other enums in other files may also have `NONE` — leave those alone.

2. **Compiler-first approach.** After Phase A, run the build immediately. The compiler errors pinpoint every file that needs editing. This is more reliable than grep for a high-generic-name enum.

3. **Preserved hex values.** Don't strip the `= 0x0` etc. — they are intentional HW documentation.

4. **Preserved comments.** The `// A - None || B - None` etc. inline comments in `llk_defs.h` document HW semantics — preserve them.

5. **Namespace mismatch.** LLK internal files use `BroadcastType::NONE`; wrapper/API files need `ckernel::BroadcastType::NONE`.

6. **`bcast.h` already partially fixed.** The DataCopyType refactor touched `bcast.h` and already used `BroadcastType::NONE` in the `constexpr auto data_copy_type` line. Verify which lines are already qualified before editing.

7. **Editing Quasar.** It's already correct; don't touch it.
