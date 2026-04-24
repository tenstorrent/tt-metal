# Issue tt-metal#43036 — Make `EltwiseBinaryType` an `enum class` (Blackhole + Wormhole B0)

> **Audience:** implementor (human or agent) taking on this refactor.
> **Author:** mentor handoff. Read top-to-bottom once before touching code.
> **Issue:** https://github.com/tenstorrent/tt-metal/issues/43036
> **Repo note:** `tt_metal/tt-llk/` is a **plain directory** inside the `tt-metal` monorepo —
> not a submodule. Edit files in-place. The PR goes to `tenstorrent/tt-metal`.

---

## 1. What the issue is really about

`EltwiseBinaryType` in Blackhole and Wormhole B0 is declared as a plain (unscoped) `enum`:

```cpp
// tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h (line 51)
// tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h (line 51)
enum EltwiseBinaryType
{
    ELWMUL,
    ELWDIV,
    ELWADD,
    ELWSUB,
    ELWLESS,
};
```

Plain enums inject `ELWMUL`, `ELWDIV`, `ELWADD`, `ELWSUB`, and `ELWLESS` into the `ckernel` namespace, enabling unqualified usage across eltwise binary implementations.

**The fix:** Convert BH and WH `EltwiseBinaryType` to `enum class`, then qualify every bare enumerator reference.

**Quasar has this right**, but with **fewer enumerators** — it is the reference for the declaration pattern:

```cpp
// tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_defs.h
enum class EltwiseBinaryType : std::uint8_t
{
    ELWMUL,
    ELWADD,
    ELWSUB,
};
```

**CRITICAL arch difference:** Quasar does **not** have `ELWDIV` or `ELWLESS`. Blackhole and Wormhole B0 do. **Do not remove `ELWDIV` or `ELWLESS` from the BH/WH enum.** The target declaration for BH and WH is:

```cpp
enum class EltwiseBinaryType : std::uint8_t
{
    ELWMUL,
    ELWDIV,
    ELWADD,
    ELWSUB,
    ELWLESS,
};
```

Only the `enum` → `enum class : std::uint8_t` header changes; the enumerator list stays the same.

---

## 2. Scope — find every bare callsite before editing

This is a **high-callsite-count refactor** (~197 bare hits). Run the grep first and study the output before editing anything.

```bash
cd /localdev/ncvetkovic/work/tt-metal

# Files with the enum definition to convert
grep -rn "enum EltwiseBinaryType" tt_metal/tt-llk/

# Files with bare enumerators in executable code
grep -rn "\bELWMUL\b\|\bELWDIV\b\|\bELWADD\b\|\bELWSUB\b\|\bELWLESS\b" \
  tt_metal/ ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "EltwiseBinaryType::\|ckernel::EltwiseBinaryType\|//\|\.py\|\.md\|tt_llk_quasar"
```

Expected files to check (non-exhaustive — verify with the grep):
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary*.h` — template params and `if constexpr` guards
- `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary*.h` — same
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_eltwise_binary_api.h` — wrapper
- `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_math_eltwise_binary_api.h` — wrapper
- `tt_metal/hw/inc/api/compute/eltwise_binary.h` — public compute API
- `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` — if it exists
- Various TTNN kernels for add/mul/sub/div operations
- Test kernels for eltwise binary operations

---

## 3. Qualification rules

Files **inside** `namespace ckernel` (or TU with `using namespace ckernel` in scope):
```cpp
// before
if constexpr (op == ELWMUL)
// after
if constexpr (op == EltwiseBinaryType::ELWMUL)
```

Files **outside** `namespace ckernel`:
```cpp
// after
if constexpr (op == ckernel::EltwiseBinaryType::ELWMUL)
```

Before choosing the prefix, check:
```bash
grep -n "namespace ckernel\|using namespace ckernel" <file>
```

**Expect a mix:** the LLK internal files are inside `namespace ckernel`; wrapper and compute API files are outside.

---

## 4. Branch and worktree setup

```bash
cd /localdev/ncvetkovic/work/tt-metal
git fetch origin main
git checkout -b ncvetkovic/43036-eltwisebinarytype-enum-class FETCH_HEAD
git log --oneline origin/main..HEAD   # should be empty before edits
```

---

## 5. Suggested execution order

**Phase A — LLK layer (BH + WH)**
1. Edit `tt_llk_blackhole/llk_lib/llk_defs.h`: `enum EltwiseBinaryType` → `enum class EltwiseBinaryType : std::uint8_t` (keep `ELWDIV` and `ELWLESS`!)
2. Edit `tt_llk_wormhole_b0/llk_lib/llk_defs.h`: same
3. Edit all `llk_math_eltwise_binary*.h` files in both arch dirs: qualify bare enumerators

**Phase B — Propagate upward**
4. Fix wrapper files: `tt_metal/hw/ckernels/*/metal/llk_api/llk_math_eltwise_binary_api.h`
5. Fix compute API: `tt_metal/hw/inc/api/compute/eltwise_binary*.h` (if any bare names)
6. Fix any TTNN / test / model kernel files found by the grep

**Phase C — Verify**
```bash
./build_metal.sh --build-tests

# Invariant: should return zero hits
grep -rn "\bELWMUL\b\|\bELWDIV\b\|\bELWADD\b\|\bELWSUB\b\|\bELWLESS\b" \
  tt_metal/ ttnn/ tests/ models/ \
  --include="*.h" --include="*.cpp" --include="*.hpp" \
  | grep -v "EltwiseBinaryType::\|ckernel::EltwiseBinaryType\|//\|\.py\|\.md\|tt_llk_quasar"
# Expected: empty
```

---

## 6. Commit message

```
refactor(llk): convert EltwiseBinaryType from plain enum to enum class

Plain enum EltwiseBinaryType in Blackhole and Wormhole B0 injected ELWMUL,
ELWDIV, ELWADD, ELWSUB, and ELWLESS into the ckernel namespace, enabling
unqualified usage across eltwise binary implementations. Convert to enum class
(using Quasar as the declaration pattern) and qualify all bare enumerator
references. BH/WH retain ELWDIV and ELWLESS which Quasar does not support.

Resolves tt-metal#43036
```

---

## 7. PR creation

```bash
git push origin ncvetkovic/43036-eltwisebinarytype-enum-class

gh pr create \
  --base main \
  --head ncvetkovic/43036-eltwisebinarytype-enum-class \
  --title "refactor(llk): convert EltwiseBinaryType from plain enum to enum class" \
  --body "$(cat <<'PRBODY'
### Summary

`EltwiseBinaryType` in Blackhole and Wormhole B0 was a plain (unscoped) `enum`, which injected
`ELWMUL`, `ELWDIV`, `ELWADD`, `ELWSUB`, and `ELWLESS` directly into the `ckernel` namespace.
Unqualified callsites throughout the eltwise binary LLK implementation make it hard to grep
and create ODR risk.

This PR converts `EltwiseBinaryType` to `enum class EltwiseBinaryType : std::uint8_t`, using
Quasar as the declaration pattern. All bare enumerator references across LLK internals, wrapper
layers, and compute API are updated. Note: `ELWDIV` and `ELWLESS` are retained in the BH/WH
enum since Quasar does not support those operations.

Closes tt-metal#43036 (EltwiseBinaryType sub-task)

### What's changed

- **`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h`** — `enum EltwiseBinaryType` → `enum class EltwiseBinaryType : std::uint8_t` (enumerator list unchanged)
- **`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h`** — same
- **`tt_metal/tt-llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_binary*.h`** — bare enumerators qualified
- **`tt_metal/hw/ckernels/*/metal/llk_api/llk_math_eltwise_binary_api.h`** — bare enumerators qualified
- **`tt_metal/hw/inc/api/compute/eltwise_binary*.h`** — bare enumerators qualified (if any)
- Additional TTNN / test kernel files as found by the invariant grep

### What's intentionally unchanged

- **`tt_metal/tt-llk/tt_llk_quasar/`** — already `enum class EltwiseBinaryType`; no changes needed.
- **`ELWDIV` and `ELWLESS` enumerators** — kept in BH/WH since Quasar omits them for arch reasons; their values are not changed.
- **Other enums in `llk_defs.h`** — out of scope; separate sub-tasks of tt-metal#43036.

### Architecture-specific notes

- **Quasar** supports `{ELWMUL, ELWADD, ELWSUB}`. **Blackhole and Wormhole B0** additionally support `ELWDIV` and `ELWLESS`. The enum class conversion does not change this arch split — it only scopes the names.

### Notes for reviewers

- High callsite count (~197): run the invariant grep to confirm all are covered.
- Do not remove `ELWDIV` or `ELWLESS` from BH/WH even though Quasar lacks them.
- Compile-time invariant:
  ```bash
  grep -rn "\bELWMUL\b\|\bELWDIV\b\|\bELWADD\b\|\bELWSUB\b\|\bELWLESS\b" \
    tt_metal/ ttnn/ tests/ models/ \
    --include="*.h" --include="*.cpp" --include="*.hpp" \
    | grep -v "EltwiseBinaryType::\|ckernel::EltwiseBinaryType\|//\|\.py\|\.md\|tt_llk_quasar"
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

- [ ] `grep -rn "enum EltwiseBinaryType" tt_metal/tt-llk/` returns zero hits.
- [ ] Invariant grep from section 5 returns zero hits.
- [ ] `./build_metal.sh --build-tests` succeeds.
- [ ] PR open with description above (no placeholder text).
- [ ] `ELWDIV` and `ELWLESS` are still present in BH/WH `llk_defs.h`.
- [ ] No files under `tt_metal/tt-llk/tt_llk_quasar/` were modified.
- [ ] No other enums in `llk_defs.h` were touched.

---

## 9. Common pitfalls

1. **DO NOT remove `ELWDIV` or `ELWLESS`.** They exist in BH/WH but not Quasar. Do not "align" the enumerator list to Quasar — only the `enum` → `enum class` header changes.

2. **Mixed namespace context.** LLK internal files use `EltwiseBinaryType::ELWMUL`; wrapper/API files need `ckernel::EltwiseBinaryType::ELWMUL`. Check each file.

3. **High callsite count.** Do not assume the count from the summary is current — run a fresh grep.

4. **Forgetting the `: std::uint8_t`.** Match the Quasar declaration pattern.

5. **Editing Quasar.** It's already correct; don't touch it.
