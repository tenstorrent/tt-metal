# Issue tt-metal#43036 — Make `ReduceDim` an `enum class` (Blackhole + Wormhole B0)

> **Audience:** implementor (human or agent) taking on this refactor.
> **Author:** mentor handoff. Read top-to-bottom once before touching code.
> **Issue:** https://github.com/tenstorrent/tt-metal/issues/43036
> **Repo note:** `tt_metal/tt-llk/` is a **plain directory** inside the `tt-metal` monorepo —
> not a submodule. Edit files in-place. The PR goes to `tenstorrent/tt-metal`.

---

## 1. What the issue is really about

`ReduceDim` in Blackhole and Wormhole B0 is declared as a plain (unscoped) `enum`:

```cpp
// tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h (line 24)
// tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h (line 24)
enum ReduceDim
{
    REDUCE_ROW,
    REDUCE_COL,
    REDUCE_SCALAR,
};
```

Plain enums inject `REDUCE_ROW`, `REDUCE_COL`, and `REDUCE_SCALAR` into the `ckernel` namespace. These names are generic enough to collide with other definitions in a large codebase, creating ODR risk. Unqualified callsites are also harder to grep and understand.

**The fix:** Convert BH and WH `ReduceDim` to `enum class`, then qualify every bare `REDUCE_ROW`/`REDUCE_COL`/`REDUCE_SCALAR` reference.

**Quasar already has this right** — it is the reference implementation:

```cpp
// tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h
enum class ReduceDim : std::uint8_t
{
    REDUCE_ROW,
    REDUCE_COL,
    REDUCE_SCALAR,
};
```

Match that declaration exactly for BH and WH:

```cpp
enum class ReduceDim : std::uint8_t
{
    REDUCE_ROW,
    REDUCE_COL,
    REDUCE_SCALAR,
};
```

---

## 2. Scope — find every bare callsite before editing

This is a **high-callsite-count refactor** (~115 bare hits). Run the grep first and study the output before editing anything.

```bash
cd /localdev/ncvetkovic/work/tt-metal

# Files with the enum definition to convert
grep -rn "enum ReduceDim" tt_metal/tt-llk/

# Files with bare enumerators in executable code
grep -rn "\bREDUCE_ROW\b\|\bREDUCE_COL\b\|\bREDUCE_SCALAR\b" \
  tt_metal/ ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "ReduceDim::\|ckernel::ReduceDim\|//\|\.py\|\.md\|tt_llk_quasar"
```

Expected files to check (non-exhaustive — verify with the grep):
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_reduce.h` — template params and `if constexpr` guards
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_reduce.h` — same
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_reduce.h` — same
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_unpack_reduce.h` — same
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_reduce_api.h` — wrapper
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_reduce_api.h` — wrapper
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_reduce_api.h` — wrapper
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_unpack_reduce_api.h` — wrapper
- `tt_metal/hw/inc/api/compute/reduce.h` — public compute API
- Various TTNN and test kernel files that call reduce operations

---

## 3. Qualification rules

Files **inside** `namespace ckernel` (or TU with `using namespace ckernel` in scope):
```cpp
// before
if constexpr (reduce_dim == REDUCE_ROW)
// after
if constexpr (reduce_dim == ReduceDim::REDUCE_ROW)
```

Files **outside** `namespace ckernel`:
```cpp
// after
if constexpr (reduce_dim == ckernel::ReduceDim::REDUCE_ROW)
```

Before choosing the prefix, check:
```bash
grep -n "namespace ckernel\|using namespace ckernel" <file>
```

**Expect a mix:** the LLK internal files are inside `namespace ckernel`; the wrapper files (`llk_api/`) and compute API files are outside. Check each file individually — don't assume.

---

## 4. Branch and worktree setup

```bash
cd /localdev/ncvetkovic/work/tt-metal
git fetch origin main
git checkout -b ncvetkovic/43036-reducedim-enum-class FETCH_HEAD
git log --oneline origin/main..HEAD   # should be empty before edits
```

---

## 5. Suggested execution order

**Phase A — LLK layer (BH + WH)**
1. Edit `tt_llk_blackhole/llk_lib/llk_defs.h`: `enum ReduceDim` → `enum class ReduceDim : std::uint8_t`
2. Edit `tt_llk_wormhole_b0/llk_lib/llk_defs.h`: same
3. Edit `llk_math_reduce.h` and `llk_unpack_reduce.h` for both arches: qualify bare enumerators

**Phase B — Propagate upward**
4. Fix wrapper files: `tt_metal/hw/ckernels/*/metal/llk_api/llk_math_reduce_api.h` and `llk_unpack_reduce_api.h`
5. Fix compute API: `tt_metal/hw/inc/api/compute/reduce.h` (if any bare names present)
6. Fix any TTNN / test / model kernel files found by the grep

**Phase C — Verify**
```bash
./build_metal.sh --build-tests

# Invariant: should return zero hits
grep -rn "\bREDUCE_ROW\b\|\bREDUCE_COL\b\|\bREDUCE_SCALAR\b" \
  tt_metal/ ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "ReduceDim::\|ckernel::ReduceDim\|//\|\.py\|\.md\|tt_llk_quasar"
# Expected: empty
```

---

## 6. Commit message

```
refactor(llk): convert ReduceDim from plain enum to enum class

Plain enum ReduceDim in Blackhole and Wormhole B0 injected REDUCE_ROW, REDUCE_COL,
and REDUCE_SCALAR into the ckernel namespace, enabling unqualified usage across
reduce math and unpack layers. Convert to enum class (matching Quasar) and qualify
all bare enumerator references across LLK internals, wrappers, and compute API.

Resolves tt-metal#43036
```

---

## 7. PR creation

```bash
git push origin ncvetkovic/43036-reducedim-enum-class

gh pr create \
  --base main \
  --head ncvetkovic/43036-reducedim-enum-class \
  --title "refactor(llk): convert ReduceDim from plain enum to enum class" \
  --body "$(cat <<'PRBODY'
### Summary

`ReduceDim` in Blackhole and Wormhole B0 was a plain (unscoped) `enum`, which injected
`REDUCE_ROW`, `REDUCE_COL`, and `REDUCE_SCALAR` directly into the `ckernel` namespace.
The LLK math and unpack reduce implementations used these names without any qualifying type,
making callsites harder to grep and creating ODR risk.

This PR converts `ReduceDim` to `enum class ReduceDim : std::uint8_t` in the Blackhole and
Wormhole B0 `llk_defs.h` files, matching Quasar. All bare enumerator references across LLK
internals, wrapper layers, and compute API are updated to use the fully-qualified form.

Closes tt-metal#43036 (ReduceDim sub-task)

### What's changed

- **`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h`** — `enum ReduceDim` → `enum class ReduceDim : std::uint8_t`
- **`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h`** — same
- **`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_reduce.h`** — bare enumerators qualified
- **`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_unpack_reduce.h`** — bare enumerators qualified
- **`tt_metal/hw/ckernels/*/metal/llk_api/llk_*reduce_api.h`** — bare enumerators qualified
- **`tt_metal/hw/inc/api/compute/reduce.h`** — bare enumerators qualified (if any)
- Additional TTNN / test kernel files as found by the invariant grep

### What's intentionally unchanged

- **`tt_metal/tt-llk/tt_llk_quasar/`** — already `enum class ReduceDim`; no changes needed.
- **Other enums in `llk_defs.h`** — out of scope; separate sub-tasks of tt-metal#43036.

### Notes for reviewers

- High callsite count (~115): run the invariant grep to confirm all are covered.
- LLK internal files are inside `namespace ckernel` (use `ReduceDim::REDUCE_ROW`);
  wrapper and compute API files are outside (use `ckernel::ReduceDim::REDUCE_ROW`).
- Compile-time invariant:
  ```bash
  grep -rn "\bREDUCE_ROW\b\|\bREDUCE_COL\b\|\bREDUCE_SCALAR\b" \
    tt_metal/ ttnn/ tests/ models/ \
    --include="*.h" --include="*.cpp" --include="*.hpp" \
    | grep -v "ReduceDim::\|ckernel::ReduceDim\|//\|\.py\|\.md\|tt_llk_quasar"
  # Expected: zero hits
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

- [ ] `grep -rn "enum ReduceDim" tt_metal/tt-llk/` returns zero hits (all converted to `enum class`).
- [ ] Invariant grep from section 5 returns zero hits.
- [ ] `./build_metal.sh --build-tests` succeeds.
- [ ] PR open with description above (no placeholder text).
- [ ] No files under `tt_metal/tt-llk/tt_llk_quasar/` were modified.
- [ ] No other enums in `llk_defs.h` were touched.

---

## 9. Common pitfalls

1. **Mixed namespace context.** LLK internal files use `ReduceDim::REDUCE_ROW`; wrapper/API files need `ckernel::ReduceDim::REDUCE_ROW`. Check each file individually.

2. **High callsite count.** Do not assume the grep output from the summary is complete — run it fresh; files may have changed.

3. **Forgetting the `: std::uint8_t`.** Match the Quasar declaration exactly.

4. **Grepping only tt-llk.** Run the full-repo invariant grep before declaring done.

5. **Editing Quasar.** It's already correct; don't touch it.
