# Issue tt-metal#43036 — Make `ReluType` an `enum class` (Blackhole + Wormhole B0)

> **Audience:** implementor (human or agent) taking on this refactor.
> **Author:** mentor handoff. Read top-to-bottom once before touching code.
> **Issue:** https://github.com/tenstorrent/tt-metal/issues/43036
> **Repo note:** `tt_metal/tt-llk/` is a **plain directory** inside the `tt-metal` monorepo —
> not a submodule. Edit files in-place. The PR goes to `tenstorrent/tt-metal`.

---

## 1. What the issue is really about

`ReluType` in Blackhole and Wormhole B0 is declared as a plain (unscoped) `enum`:

```cpp
// tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h (line 108)
// tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h (line 108)
enum ReluType
{
    NO_RELU,
    ZERO_RELU,
    MIN_THRESHOLD_RELU,
    MAX_THRESHOLD_RELU,
};
```

Plain enums inject all four enumerators into the `ckernel` namespace, enabling unqualified usage. This is an ODR hazard (especially for generic names like `NO_RELU`) and hurts readability at callsites.

**The fix:** Convert BH and WH `ReluType` to `enum class`, then qualify every bare enumerator reference.

**Quasar already has this right** — it is the reference implementation:

```cpp
// tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h
enum class ReluType : std::uint8_t
{
    NO_RELU = 0,
    ZERO_RELU,
    MIN_THRESHOLD_RELU,
    MAX_THRESHOLD_RELU,
};
```

Match that declaration for BH and WH (the explicit `= 0` on `NO_RELU` is harmless to add and documents the HW encoding):

```cpp
enum class ReluType : std::uint8_t
{
    NO_RELU = 0,
    ZERO_RELU,
    MIN_THRESHOLD_RELU,
    MAX_THRESHOLD_RELU,
};
```

**Note:** Quasar also defines a `ReluConfig` struct and a `GetReluMode()` helper — these are Quasar-specific additions and should **not** be added to BH/WH as part of this issue.

---

## 2. Scope — find every bare callsite before editing

```bash
cd /localdev/ncvetkovic/work/tt-metal

# Files with the enum definition to convert
grep -rn "enum ReluType" tt_metal/tt-llk/

# Files with bare enumerators in executable code
grep -rn "\bNO_RELU\b\|\bZERO_RELU\b\|\bMIN_THRESHOLD_RELU\b\|\bMAX_THRESHOLD_RELU\b" \
  tt_metal/ ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "ReluType::\|ckernel::ReluType\|//\|\.py\|\.md\|tt_llk_quasar"
```

As of the time this prompt was written, approximately **15 bare callsites** exist.

Expected files to check:
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack*.h` — `ReluType` as template param; comparisons in `if constexpr`
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack*.h` — same
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_api.h` — wrapper translating bool → `ReluType`
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_pack_api.h` — same
- `tt_metal/hw/inc/api/compute/pack*.h` — public compute API (may have bare enumerators)
- TTNN compute kernels that pass relu mode (relu-fused pack operations)

Run the grep first and collect all hits before editing anything.

---

## 3. Qualification rules

Files **inside** `namespace ckernel` (or TU with `using namespace ckernel` in scope):
```cpp
// before
if constexpr (relu_type == NO_RELU)
// after
if constexpr (relu_type == ReluType::NO_RELU)
```

Files **outside** `namespace ckernel`:
```cpp
// after
if constexpr (relu_type == ckernel::ReluType::NO_RELU)
```

Before choosing the prefix, check:
```bash
grep -n "namespace ckernel\|using namespace ckernel" <file>
```

---

## 4. Branch and worktree setup

```bash
cd /localdev/ncvetkovic/work/tt-metal
git fetch origin main
git checkout -b ncvetkovic/43036-relutype-enum-class FETCH_HEAD
git log --oneline origin/main..HEAD   # should be empty before edits
```

---

## 5. Suggested execution order

**Phase A — LLK layer (BH + WH)**
1. Edit `tt_llk_blackhole/llk_lib/llk_defs.h`: `enum ReluType` → `enum class ReluType : std::uint8_t` (with `NO_RELU = 0`)
2. Edit `tt_llk_wormhole_b0/llk_lib/llk_defs.h`: same
3. Edit all `llk_pack*.h` files in both arch dirs: qualify bare `NO_RELU`/`ZERO_RELU`/`MIN_THRESHOLD_RELU`/`MAX_THRESHOLD_RELU`

**Phase B — Propagate upward**
4. Fix wrapper files in `tt_metal/hw/ckernels/*/metal/llk_api/`
5. Fix compute API files in `tt_metal/hw/inc/api/compute/` (if any)
6. Fix any TTNN / model kernel files found by the grep

**Phase C — Verify**
```bash
./build_metal.sh --build-tests

# Invariant: should return zero hits
grep -rn "\bNO_RELU\b\|\bZERO_RELU\b\|\bMIN_THRESHOLD_RELU\b\|\bMAX_THRESHOLD_RELU\b" \
  tt_metal/ ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "ReluType::\|ckernel::ReluType\|//\|\.py\|\.md\|tt_llk_quasar"
# Expected: empty
```

---

## 6. Commit message

```
refactor(llk): convert ReluType from plain enum to enum class

Plain enum ReluType in Blackhole and Wormhole B0 injected NO_RELU, ZERO_RELU,
MIN_THRESHOLD_RELU, and MAX_THRESHOLD_RELU into the ckernel namespace, enabling
unqualified usage that risks ODR violations. Convert to enum class (matching
Quasar) and qualify all bare enumerator references across LLK internals,
the compute API, and wrapper layers.

Resolves tt-metal#43036
```

---

## 7. PR creation

```bash
git push origin ncvetkovic/43036-relutype-enum-class

gh pr create \
  --base main \
  --head ncvetkovic/43036-relutype-enum-class \
  --title "refactor(llk): convert ReluType from plain enum to enum class" \
  --body "$(cat <<'PRBODY'
### Summary

`ReluType` in Blackhole and Wormhole B0 was a plain (unscoped) `enum`, which injected its
enumerators (`NO_RELU`, `ZERO_RELU`, `MIN_THRESHOLD_RELU`, `MAX_THRESHOLD_RELU`) directly into
the `ckernel` namespace. This allowed — and in practice caused — callsites in the LLK pack
implementation to compare against bare `NO_RELU` etc. without any qualifying type. Beyond
readability, generic names like `NO_RELU` in a shared namespace are an ODR hazard.

This PR converts `ReluType` to `enum class ReluType : std::uint8_t` in the Blackhole and
Wormhole B0 `llk_defs.h` files, matching Quasar. All bare enumerator references across LLK
internals, compute API, and wrapper layers are updated to use the fully-qualified form.

Closes tt-metal#43036 (ReluType sub-task)

### What's changed

- **`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h`** — `enum ReluType` → `enum class ReluType : std::uint8_t`
- **`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h`** — same
- **`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_pack*.h`** — bare enumerators qualified
- **`tt_metal/hw/ckernels/*/metal/llk_api/llk_pack_api.h`** — bare enumerators qualified (if any)
- **`tt_metal/hw/inc/api/compute/pack*.h`** — bare enumerators qualified (if any)

### What's intentionally unchanged

- **`tt_metal/tt-llk/tt_llk_quasar/`** — already `enum class ReluType`; no changes needed.
- **Quasar-specific `ReluConfig` struct** — not added to BH/WH; out of scope.
- **Other enums in `llk_defs.h`** — out of scope; separate sub-tasks of tt-metal#43036.

### Notes for reviewers

- The reference declaration is `tt_llk_quasar/llk_lib/llk_defs.h`. Added `NO_RELU = 0` to
  document the HW encoding (value 0 matches Quasar; harmless).
- Compile-time invariant:
  ```bash
  grep -rn "\bNO_RELU\b\|\bZERO_RELU\b\|\bMIN_THRESHOLD_RELU\b\|\bMAX_THRESHOLD_RELU\b" \
    tt_metal/ ttnn/ tests/ models/ \
    --include="*.h" --include="*.cpp" --include="*.hpp" \
    | grep -v "ReluType::\|ckernel::ReluType\|//\|\.py\|\.md\|tt_llk_quasar"
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

- [ ] `grep -rn "enum ReluType" tt_metal/tt-llk/` returns zero hits (all converted to `enum class`).
- [ ] Invariant grep from section 5 returns zero hits.
- [ ] `./build_metal.sh --build-tests` succeeds.
- [ ] PR open with description above (no placeholder text).
- [ ] No files under `tt_metal/tt-llk/tt_llk_quasar/` were modified.
- [ ] `ReluConfig` struct was **not** added to BH/WH.
- [ ] No other enums in `llk_defs.h` were touched.

---

## 9. Common pitfalls

1. **Namespace mismatch.** Check `grep -n "namespace ckernel\|using namespace ckernel" <file>` before choosing the prefix.

2. **Quasar's `ReluConfig` struct.** It exists in Quasar's `llk_defs.h` right after `ReluType`. Do **not** copy it to BH/WH — it is Quasar-specific and out of scope.

3. **Forgetting the `: std::uint8_t`.** Match the Quasar declaration exactly.

4. **Grepping only tt-llk.** Run the full-repo invariant grep before declaring done.

5. **Editing Quasar.** It's already correct; don't touch it.
