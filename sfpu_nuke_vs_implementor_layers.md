# Why Some Layers Are Nuked and Others Are Left Over

The **implementor** creates 11 layers (numbered 1-11) when building a new SFPU unary op.
The **nuke skill** removes from 12 layers (numbered differently) when tearing one down.
They don't perfectly mirror each other — and that's mostly intentional.

---

## Layer-by-Layer Mapping

> Path convention: paths under the `tt_llk` submodule are rooted at `tt_metal/third_party/tt_llk/`. Paths prefixed with `hw/` are under `tt_metal/hw/`. Paths prefixed with `ttnn/` are project-root-relative.

| # | Implementor Layer | File Path(s) | Nuke Action | Nuke # |
|---|---|---|---|---|
| 1 | SFPU Kernel | **tt_llk impl (actual math):**<br>`tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_{op}.h`<br>`tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_{op}.h`<br>**hw wrapper (calls into tt_llk):**<br>`hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_{op}.h`<br>`hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_{op}.h` | **DELETE** | 10 |
| 2 | LLK Dispatch | **hw (per-op files):**<br>`hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_{op}.h`<br>`hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_{op}.h`<br>**tt_llk (single shared file):**<br>`tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`<br>`tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h` | **DELETE** | 10 |
| 3 | Compute API Header | `hw/inc/api/compute/eltwise_unary/{op}.h` | **DELETE** | 9 |
| 4 | SFPU Include Guard | `hw/inc/api/compute/eltwise_unary/sfpu_split_includes.h` | **EDIT** (remove block) | 8 |
| 5 | SfpuType Enum | `hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu_types.h`<br>`hw/ckernels/blackhole/metal/llk_api/llk_sfpu_types.h` | **EDIT** (remove entry, unless shared) | 11 |
| 6 | UnaryOpType Enum | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_types.hpp` | **EDIT** (remove entry) | 1 |
| 7 | Op Utils — 3 functions | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp` | **EDIT** (remove cases) | 2 + 3 |
| 8 | Op Utils Header | `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp` | **EDIT** (remove case) | 4 |
| 9 | C++ API Registration | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.hpp` | **EDIT** (remove line) | 5 |
| 10 | Python Nanobind | `ttnn/cpp/ttnn/operations/eltwise/unary/unary_nanobind.cpp` | **EDIT** (remove call) | 7 |
| 11 | Python Golden Function | `ttnn/ttnn/operations/unary.py` | **LEFT IN PLACE** | — |

### Nuke-Only Layers (not created by the implementor)

| Nuke # | What It Is | File Path(s) | Why Nuke Handles It |
|---|---|---|---|
| 6 | C++ API Implementation | `ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp` | Some legacy ops use hand-written functions instead of `REGISTER_UNARY_*` macros |
| 12 | Specialized Compute Kernel | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/{op}_kernel.cpp` | Some ops (LGAMMA, MISH, etc.) don't use the generic `eltwise_sfpu.cpp` |
| — | Test files | `tests/ttnn/unit_tests/operations/eltwise/test_{op}.py` | Deleted so reimplementation can be tested cleanly |

---

## Why One Layer Is Left In Place

### Python Golden Function (`unary.py`) — NOT nuked

The golden function is a PyTorch reference implementation attached via `ttnn.attach_golden_function()`. It computes the expected output for test comparison.

Three reasons to leave it:

1. **It's the test oracle, not the implementation.** The golden function defines *what* the operation should compute (using PyTorch). The SFPU kernel defines *how* it computes on Tenstorrent hardware. Nuking the oracle means the agent can't verify its own work.

2. **It enables immediate test feedback.** After reimplementation, the agent creates a test that calls `ttnn.{op}()` and compares against the golden function. If the golden function is missing, the test infrastructure itself breaks — the agent would be debugging test scaffolding instead of kernel correctness.

3. **Writing a PyTorch one-liner is trivial.** The golden function is typically `return torch.nn.functional.elu(input, alpha=alpha)` — a single line calling PyTorch. Testing whether the agent can write this provides no signal about its ability to implement SFPU kernels. The hard part is `ckernel_sfpu_{op}.h`, not `_golden_function_{op}`.

---

## Why the Nuke Has Extra Layers

### C++ API Implementation (`unary.cpp`) — Nuke Layer 6

The implementor always uses registration macros (`REGISTER_UNARY_OPERATION`, etc.) which auto-generate the C++ API functions. But some existing operations in the codebase predate these macros and have hand-written function bodies in `unary.cpp`. The nuke must handle both patterns because it operates on the full existing codebase, not just agent-generated ops.

### Specialized Compute Kernel — Nuke Layer 11

The implementor always targets the generic `eltwise_sfpu.cpp` compute kernel, which dispatches via `SFPU_OP_CHAIN_0`. But some operations (LGAMMA, MISH, TANHSHRINK, IDENTITY, etc.) have their own dedicated compute kernel `.cpp` files and a custom path in `get_compute_kernel_path()`. The nuke must clean these up. The implementor never creates them because new agent-generated ops always use the standard dispatch path.

### Test Files

The nuke deletes test files because:
- The goal is to evaluate reimplementation end-to-end, including test creation
- Stale test files referencing a nuked enum value would cause import/compile errors
- The implementor creates tests in Mode B (separate from the implementation layers in Mode A)

---

## The `get_op_approx_mode()` Gap

One minor asymmetry: the implementor's Layer 7 includes updating `get_op_approx_mode()` in `unary_op_utils.cpp` (to return `true` for ops needing approximation). The nuke's Layers 2-3 don't explicitly mention removing this case. In practice, most operations fall through to `default: return false`, so there's nothing to remove. For operations that do have an explicit case, a leftover `case UnaryOpType::{OP}: return true;` would cause a build error once the enum value is removed in Layer 1 — so the build verification step (nuke Step 5) catches this if it happens.

---

## Summary

| Decision | Rationale |
|---|---|
| **Delete** dedicated files (SFPU kernel, LLK dispatch, compute API header) | These ARE the implementation — removing them is the whole point |
| **Edit** shared files (enums, switch cases, registrations, bindings) | Surgical removal from shared infrastructure without breaking other ops |
| **Edit** SfpuType enum (standalone) / **Skip** (shared) | Same family-aware logic as include guards — remove if unique, skip if other ops depend on it |
| **Leave** Python golden function | It's the test oracle — removing it prevents the agent from verifying correctness |
| **Delete** test files | Evaluation includes test creation; stale tests would cause errors |
| **Extra nuke layers** (custom C++ impl, specialized kernels) | The nuke handles the full zoo of existing ops; the implementor only generates standard ones |
