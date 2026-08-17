---
description: 'PR review rules for LLK compute kernels and ckernels'
applyTo: 'tt_metal/hw/ckernels/**,tt_metal/tt-llk/**,tt_metal/hw/inc/api/compute/**'
excludeAgent: "cloud-agent"
---

# LLK Kernels Review

## 🔴 CRITICAL

- **No duplicated LLK functions**: do not re-implement `_llk_math_eltwise_unary_sfpu_init_`, `_llk_math_eltwise_unary_sfpu_params_`, or other existing LLK primitives in new kernel files. Use the shared implementations from common SFPU headers. Before flagging a new `_llk_*` definition, check whether it already exists under `tt_llk_{arch}/llk_lib/` or the common SFPU headers.
- **Architecture-consistent API signatures**: the **Compute API** (`tt_metal/hw/inc/api/compute/`) is the layer that must have a single signature across all architectures (WH/BH/Quasar). Use `[[maybe_unused]]` parameters and runtime asserts (`LLK_ASSERT`) on unsupported features rather than `#ifdef`-guarded signature changes. Minimize `#ifdef` count in Compute API. Note: per-arch `llk_api` and `llk_lib` files are **expected** to differ between architectures — do NOT flag legitimate per-arch divergence in those files as a violation. The arch-consistency requirement applies only to the Compute API layer.
- **Layer placement**: `TTI_*` instructions belong only inside `tt-llk`. The Compute API (`tt_metal/hw/inc/api/compute/`) must be device-agnostic — no per-arch branches, no direct `ckernel::math` calls. Per-arch divergence belongs in the per-arch `llk_api` layer at `tt_metal/hw/ckernels/{wormhole_b0,blackhole,quasar}/metal/llk_api/` (outside `tt-llk`) or in the per-arch `llk_lib` inside `tt-llk` at `tt_llk_{arch}/llk_lib/`, not in the Compute API.
- **Loop/face iteration correctness**: when generalizing unpack/math loops for non-standard tile shapes, verify that outer-loop counts are consistent with sibling branches (e.g., SCALAR, ROW, COL paths must all agree on `num_faces_r_dim` vs `num_faces`). A mismatch silently produces wrong results.
- **Dead code removal**: untested dead code (unused sort modes, alternative algorithms, unreachable paths) must be removed, not left commented out. It accumulates and misleads future readers. Bare `// TODO:` comments with no linked issue are not acceptable — link to a GitHub issue or resolve inline.
- **Reconfig escapes**: a change that reconfigures unpack/pack/DEST state must restore or fully re-set it. Leaked format or DEST-mode state corrupts the *next* kernel while the current test still passes — flag any reconfig path that does not round-trip its state.
- **CFG register read-after-write ordering**: reading a config register that may have in-flight tensix/MMIO writes needs explicit ordering — drain/sync the prior write before the read, otherwise the read races the write and silently returns a stale value. The exact placement (NOPs, temp register, sync) depends on the register and who writes it, so flag a CFG read with no visible ordering against a recent write and ask the author to confirm.
- **Comments must match the code they describe**: a comment, doc tag (`@brief`/`@param`/`@note`), or inline note must describe what the adjacent code *actually does*, not a copied-in or AI-generated guess at what it should do.

## 🟡 IMPORTANT

- **Collapse nested conditionals**: if the only difference between two branches is a single parameter value, extract that value into a variable and use it once. Do not duplicate function calls in nested if/else chains that differ by one argument.
- **No magic numbers**: hardware register offsets, stride constants, face counts, and loop bounds must have named constants or comments explaining their derivation. `18`, `30`, `0x80000` are not self-explanatory.
- **Comments for non-obvious HW config**: any manipulation of data format bits, unpacker config registers, stride overrides, or SFPU config sequences must have a comment explaining *why* — especially when the code deviates from the standard pattern (e.g., "needed due to HW bug with small SrcB transactions").
- **Replay buffer and SFPCONFIG NOPs**: explain why NOPs are inserted after SFPCONFIG or before replay execution. If they exist for pipeline hazard reasons, document it. If there is no documented reason, request a comment explaining why — do not suggest removal unilaterally, as that is an author/expert call.
- **Code size awareness**: LLK kernels run on RISCs with tight code budgets. Prefer template arguments and `constexpr` selection over runtime if-else chains. Extract duplicated logic into helper functions rather than inlining the same code twice. Use `__builtin_clz` or bit tricks cautiously.
- **Init ordering**: `math_init` must come after `hw_configure`. `_init` calls belong at the start of their respective LLK function, not scattered mid-computation.
- **TRISC macros handle dispatch**: do not add compile-time `#ifdef TRISC_UNPACK` / `#ifdef TRISC_MATH` guards around LLK calls that are already wrapped in `UNPACK((...))` / `MATH((...))` macros. The macros already handle thread dispatch.
- **`@pre`/`@post` forbidden in doc comments**: per the LLK doc style in `tt_metal/tt-llk/.claude/references/doxygen-style.md`, do not use `@pre` or `@post` annotations — they imply guarantees. Use imperative `@note` instead.
- **Coverage generators checked in**: if a coverage table or manifest is auto-generated, the generator script must live in the repo (not a `/tmp` path). Otherwise the table drifts with no way to regenerate.

## 🟢 SUGGESTION

- Use `sfpi::vConst1` and other named SFPI constants instead of raw hex or float literals.
- When removing an unused LLK API overload, check cross-arch parity (does the other arch still use it?). Reference the tracking issue.
- Prefer `switch`-case over chained if/else for runtime enum dispatch — it generates smaller code on RISC-V and the compiler can warn on missing cases. Do not apply this rule to `if constexpr` chains — `if constexpr` cannot be replaced with `switch`, and trying to do so is incorrect.
- Reduce code duplication by templating on the differing parameter rather than duplicating the entire function body.
- Use `struct name_t {` (modern C++) instead of `typedef struct { } name_t;` for new code.
- FSM/sanitizer state machines should document all valid transitions — missing transition checks are blind spots.
- Prefer randomized/format-swept inputs over a few hardcoded format values; reuse existing tile-size helpers rather than hardcoding raw tile dimensions.

## Review Checklist

This checklist is the source of truth and mirrors every rule above (🔴 CRITICAL, 🟡 IMPORTANT, 🟢 SUGGESTION). Keep it in sync when rules change.

### 🔴 CRITICAL

- [ ] No duplicated LLK primitives — uses shared common implementations; checked `tt_llk_{arch}/llk_lib/` and common SFPU headers before flagging a new `_llk_*`
- [ ] Compute API signature consistent across WH/BH/Quasar (`[[maybe_unused]]` + `LLK_ASSERT`, minimal `#ifdef`); legitimate per-arch `llk_api`/`llk_lib` divergence not flagged
- [ ] Layer placement correct — `TTI_*` only inside `tt-llk`; Compute API device-agnostic; per-arch divergence in `ckernels/{arch}/metal/llk_api/` or `tt_llk_{arch}/llk_lib/`
- [ ] Loop iteration counts match sibling branches for the same tile dimension (`num_faces_r_dim` vs `num_faces`)
- [ ] Dead/unreachable code removed (not commented out); no bare `// TODO:` without a linked issue
- [ ] Reconfig paths round-trip unpack/pack/DEST state — no leaked format/DEST-mode state
- [ ] CFG register reads are ordered against in-flight writes (drain/sync/NOPs) — no stale-value races
- [ ] Comments and doc tags describe what the code actually does (no copied-in / AI-generated guesses)

### 🟡 IMPORTANT

- [ ] Nested conditionals collapsed where only one parameter differs
- [ ] Magic numbers replaced with named constants or documented
- [ ] Non-obvious HW config changes have explanatory comments
- [ ] Replay buffer / SFPCONFIG NOPs are documented (request a comment rather than removal)
- [ ] Code size considered — template/`constexpr` over runtime if-else; duplicated logic extracted to helpers
- [ ] Init ordering correct (hw_configure before math_init); `_init` calls at start of LLK function
- [ ] No redundant TRISC ifdef guards around UNPACK/MATH macros
- [ ] No `@pre`/`@post` in doc comments — use imperative `@note`
- [ ] Auto-generated coverage tables/manifests have their generator script checked into the repo (not `/tmp`)

### 🟢 SUGGESTION

- [ ] Named SFPI constants (`sfpi::vConst1`) used instead of raw hex/float literals
- [ ] Removed LLK API overloads checked for cross-arch parity; tracking issue referenced
- [ ] `switch`-case preferred over chained if/else for runtime enum dispatch (not for `if constexpr`)
- [ ] Code duplication reduced by templating on the differing parameter
- [ ] `struct name_t {` used instead of `typedef struct { } name_t;` for new code
- [ ] FSM/sanitizer state machines document all valid transitions
- [ ] Randomized/format-swept inputs preferred; existing tile-size helpers reused over raw dimensions
