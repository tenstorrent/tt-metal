---
description: 'PR review rules for LLK compute kernels and ckernels'
applyTo: 'tt_metal/hw/ckernels/**,tt_metal/tt-llk/**'
excludeAgent: "cloud-agent"
---

# LLK Kernels Review

## 🔴 CRITICAL

- **No duplicated LLK functions**: do not re-implement `_llk_math_eltwise_unary_sfpu_init_`, `_llk_math_eltwise_unary_sfpu_params_`, or other existing LLK primitives in new kernel files. Use the shared implementations from common SFPU headers.
- **Architecture-consistent API signatures**: the Compute API must have a single signature across all architectures (WH/BH/Quasar). Use `[[maybe_unused]]` parameters and runtime asserts (`LLK_ASSERT`) on unsupported features rather than `#ifdef`-guarded signature changes. Minimize `#ifdef` count in Compute API.
- **Loop/face iteration correctness**: when generalizing unpack/math loops for non-standard tile shapes, verify that outer-loop counts are consistent with sibling branches (e.g., SCALAR, ROW, COL paths must all agree on `num_faces_r_dim` vs `num_faces`). A mismatch silently produces wrong results.
- **Dead code removal**: untested dead code (unused sort modes, alternative algorithms, unreachable paths) must be removed, not left commented out. It accumulates and misleads future readers.

## 🟡 IMPORTANT

- **Collapse nested conditionals**: if the only difference between two branches is a single parameter value, extract that value into a variable and use it once. Do not duplicate function calls in nested if/else chains that differ by one argument.
- **No magic numbers**: hardware register offsets, stride constants, face counts, and loop bounds must have named constants or comments explaining their derivation. `18`, `30`, `0x80000` are not self-explanatory.
- **Comments for non-obvious HW config**: any manipulation of data format bits, unpacker config registers, stride overrides, or SFPU config sequences must have a comment explaining *why* — especially when the code deviates from the standard pattern (e.g., "needed due to HW bug with small SrcB transactions").
- **Replay buffer and SFPCONFIG NOPs**: explain why NOPs are inserted after SFPCONFIG or before replay execution. If they exist for pipeline hazard reasons, document it. If unnecessary, remove them.
- **Code size awareness**: LLK kernels run on RISCs with tight code budgets. Prefer template arguments and `constexpr` selection over runtime if-else chains. Extract duplicated logic into helper functions rather than inlining the same code twice. Use `__builtin_clz` or bit tricks cautiously — verify they don't inflate code size on the target ISA.
- **Init ordering**: `math_init` must come after `hw_configure`. `_init` calls belong at the start of their respective LLK function, not scattered mid-computation.
- **TRISC macros handle dispatch**: do not add compile-time `#ifdef TRISC_UNPACK` / `#ifdef TRISC_MATH` guards around LLK calls that are already wrapped in `UNPACK((...))` / `MATH((...))` macros. The macros already handle thread dispatch.
- **`@pre`/`@post` forbidden in doc comments**: per tt-llk doc style, do not use `@pre` or `@post` annotations — they imply guarantees. Use imperative `@note` instead.
- **Coverage generators checked in**: if a coverage table or manifest is auto-generated, the generator script must live in the repo (not a `/tmp` path). Otherwise the table drifts with no way to regenerate.

## 🟢 SUGGESTION

- Use `sfpi::vConst1` and other named SFPI constants instead of raw hex or float literals.
- When removing an unused LLK API overload, check cross-arch parity (does the other arch still use it?). Reference the tracking issue.
- Prefer `switch`-case over chained if/else for enum dispatch — it generates smaller code on RISC-V and the compiler can warn on missing cases.
- Reduce code duplication by templating on the differing parameter rather than duplicating the entire function body.
- Use `struct name_t {` (modern C++) instead of `typedef struct { } name_t;` for new code.
- FSM/sanitizer state machines should document all valid transitions — missing transition checks are blind spots.

## Review Checklist

- [ ] No duplicated LLK primitives — uses shared common implementations
- [ ] Compute API signature consistent across WH/BH/Quasar
- [ ] Loop iteration counts match sibling branches for the same tile dimension
- [ ] Dead/unreachable code removed (not commented out)
- [ ] Nested conditionals collapsed where only one parameter differs
- [ ] Magic numbers replaced with named constants or documented
- [ ] Non-obvious HW config changes have explanatory comments
- [ ] Init ordering correct (hw_configure before math_init)
- [ ] No redundant TRISC ifdef guards around UNPACK/MATH macros
- [ ] Code size considered — no unnecessary runtime branches on RISC
