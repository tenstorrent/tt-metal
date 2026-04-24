# Issue tt-metal#43036 — Make `DstSync` an `enum class` (Blackhole + Wormhole B0)

> **Audience:** implementor (human or agent) taking on this refactor.
> **Author:** mentor handoff. Read top-to-bottom once before touching code.
> **Issue:** https://github.com/tenstorrent/tt-metal/issues/43036
> **Repo note:** `tt_metal/tt-llk/` is a **plain directory** inside the `tt-metal` monorepo —
> not a submodule. Edit files in-place. The PR goes to `tenstorrent/tt-metal`.

---

## 1. What the issue is really about

`DstSync` in Blackhole and Wormhole B0 is declared as a plain (unscoped) `enum`:

```cpp
// tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h (line 67)
// tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h (line 67)
enum DstSync
{
    SyncHalf = 0,
    SyncFull = 1,
};
```

Plain enums inject `SyncHalf` and `SyncFull` into the `ckernel` namespace, enabling bare unqualified usage throughout the codebase. This is an ODR hazard and makes callsites harder to read.

**The fix:** Convert BH and WH `DstSync` to `enum class`, then qualify every bare `SyncHalf`/`SyncFull` reference.

**Quasar already has this right** — it is the reference implementation:

```cpp
// tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h
enum class DstSync : std::uint8_t
{
    SyncHalf,
    SyncFull,
};
```

Match that declaration exactly for BH and WH. Keep the explicit `= 0` / `= 1` values since they appear in the current definitions (they're harmless and document HW encoding):

```cpp
enum class DstSync : std::uint8_t
{
    SyncHalf = 0,
    SyncFull = 1,
};
```

---

## 2. Scope — find every bare callsite before editing

```bash
cd /localdev/ncvetkovic/work/tt-metal

# Files with the enum definition to convert
grep -rn "enum DstSync" tt_metal/tt-llk/

# Files with bare SyncHalf/SyncFull in executable code
grep -rn "\bSyncHalf\b\|\bSyncFull\b" \
  tt_metal/ ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "DstSync::\|ckernel::DstSync\|//\|\.py\|\.md\|tt_llk_quasar"
```

As of the time this prompt was written, approximately **11 bare callsites** exist.

Expected files to check:
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_*.h` — `DstSync` as template parameter; `if constexpr (dst_sync == SyncHalf)`
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_*.h` — same pattern
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_*.h` — wrapper layer
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_*.h` — wrapper layer
- `tt_metal/hw/inc/api/compute/*.h` — public compute API (may be out of scope; verify)

Run the grep first and collect all hits before editing anything.

---

## 3. Qualification rules

Files **inside** `namespace ckernel` (or in a TU with `using namespace ckernel` in scope):
```cpp
// before
if constexpr (dst_sync == SyncHalf)
// after
if constexpr (dst_sync == DstSync::SyncHalf)
```

Files **outside** `namespace ckernel` (no `using namespace ckernel`):
```cpp
// after
if constexpr (dst_sync == ckernel::DstSync::SyncHalf)
```

Before choosing the prefix, always check:
```bash
grep -n "namespace ckernel\|using namespace ckernel" <file>
```

---

## 4. Branch and worktree setup

```bash
cd /localdev/ncvetkovic/work/tt-metal
git fetch origin main
git checkout -b ncvetkovic/43036-dstsync-enum-class FETCH_HEAD
git log --oneline origin/main..HEAD   # should be empty before edits
```

---

## 5. Suggested execution order

**Phase A — LLK layer (BH + WH)**
1. Edit `tt_llk_blackhole/llk_lib/llk_defs.h`: `enum DstSync` → `enum class DstSync : std::uint8_t`
2. Edit `tt_llk_wormhole_b0/llk_lib/llk_defs.h`: same
3. Edit all `llk_math_*.h` files in both arch dirs: qualify bare `SyncHalf`/`SyncFull`

**Phase B — Propagate upward**
4. Fix wrapper files in `tt_metal/hw/ckernels/*/metal/llk_api/`
5. Fix compute API files in `tt_metal/hw/inc/api/compute/` (if any)
6. Fix any TTNN / model kernel files found by the grep

**Phase C — Verify**
```bash
./build_metal.sh --build-tests

# Invariant: should return zero hits
grep -rn "\bSyncHalf\b\|\bSyncFull\b" \
  tt_metal/ ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "DstSync::\|ckernel::DstSync\|//\|\.py\|\.md\|tt_llk_quasar"
# Expected: empty
```

---

## 6. Commit message

```
refactor(llk): convert DstSync from plain enum to enum class

Plain enum DstSync in Blackhole and Wormhole B0 injected SyncHalf and SyncFull
into the ckernel namespace, enabling unqualified usage that risks ODR violations.
Convert to enum class (matching Quasar) and qualify all bare enumerator references
across LLK internals, the compute API, and wrapper layers.

Resolves tt-metal#43036
```

---

## 7. PR creation

```bash
git push origin ncvetkovic/43036-dstsync-enum-class

gh pr create \
  --base main \
  --head ncvetkovic/43036-dstsync-enum-class \
  --title "refactor(llk): convert DstSync from plain enum to enum class" \
  --body "$(cat <<'PRBODY'
### Summary

`DstSync` in Blackhole and Wormhole B0 was a plain (unscoped) `enum`, which injected
`SyncHalf` and `SyncFull` into the `ckernel` namespace. This allowed unqualified usage
throughout the LLK implementation and wrapper layers — an ODR hazard and readability issue.

This PR converts `DstSync` to `enum class DstSync : std::uint8_t` in the Blackhole and
Wormhole B0 `llk_defs.h` files, matching Quasar which already had the correct declaration.
All bare `SyncHalf` and `SyncFull` references across LLK internals, compute API, and wrapper
layers are updated to use fully-qualified `DstSync::SyncHalf`/`ckernel::DstSync::SyncHalf`.

Closes tt-metal#43036 (DstSync sub-task)

### What's changed

- **`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h`** — `enum DstSync` → `enum class DstSync : std::uint8_t`
- **`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h`** — same
- **`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_*.h`** — bare `SyncHalf`/`SyncFull` qualified
- **`tt_metal/hw/ckernels/*/metal/llk_api/llk_math_*.h`** — bare enumerators qualified (if any)
- **`tt_metal/hw/inc/api/compute/*.h`** — bare enumerators qualified (if any)

### What's intentionally unchanged

- **`tt_metal/tt-llk/tt_llk_quasar/`** — already `enum class DstSync`; no changes needed.
- **Other enums in `llk_defs.h`** — out of scope; separate sub-tasks of tt-metal#43036.
- **Python test infrastructure** — unaffected.

### Notes for reviewers

- The reference declaration is `tt_llk_quasar/llk_lib/llk_defs.h`. BH/WH keep the explicit
  `= 0` / `= 1` values to document HW encoding.
- Files inside `namespace ckernel` (or with `using namespace ckernel`) use `DstSync::SyncHalf`;
  files outside need `ckernel::DstSync::SyncHalf`.
- Compile-time invariant:
  ```bash
  grep -rn "\bSyncHalf\b\|\bSyncFull\b" tt_metal/ ttnn/ tests/ models/ \
    --include="*.h" --include="*.cpp" --include="*.hpp" \
    | grep -v "DstSync::\|ckernel::DstSync\|//\|\.py\|\.md\|tt_llk_quasar"
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

- [ ] `grep -rn "enum DstSync" tt_metal/tt-llk/` returns zero hits (all converted to `enum class`).
- [ ] Invariant grep from section 5 returns zero hits.
- [ ] `./build_metal.sh --build-tests` succeeds.
- [ ] PR open with description above (no placeholder text).
- [ ] No files under `tt_metal/tt-llk/tt_llk_quasar/` were modified.
- [ ] No other enums in `llk_defs.h` were touched.

---

## 9. Common pitfalls

1. **Namespace mismatch.** Check `grep -n "namespace ckernel\|using namespace ckernel" <file>` before choosing the prefix.

2. **Forgetting the `: std::uint8_t`.** Match the Quasar declaration exactly.

3. **Grepping only tt-llk.** The bare names may appear in `tt_metal/hw/`, `tests/`, `ttnn/`, and `models/` too.

4. **Editing Quasar.** It's already correct; don't touch it.
