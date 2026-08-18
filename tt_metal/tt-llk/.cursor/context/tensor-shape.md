# TensorShape — LLK API / LIB Conversion & Coverage

Canonical guide for plumbing `ckernel::TensorShape` through LLK lib + metal wrappers, and for keeping TRISC coverage tables honest.

## What TensorShape Is

Packed 4-byte tile geometry in `common/tensor_shape.h`:

| Field | Meaning |
|-------|---------|
| `face_r_dim` | Rows per face (1/2/4/8/16) |
| `face_c_dim` | Cols per face (HW: always 16 / `MAX_FACE_C_DIM`) |
| `num_faces_r_dim` | Face grid rows (1 or 2) |
| `num_faces_c_dim` | Face grid cols (1 or 2) |

Helpers:

- `make_tensor_shape(fr, fc, nfr, nfc)` — preferred when the 2D grid is known
- `make_tensor_shape_from_legacy(face_r_dim, num_faces)` / `tensor_shape_from_num_faces(face_r_dim, num_faces)` — same arg order; legacy bridge only
- `DEFAULT_TENSOR_SHAPE` — full 32×32 (`{16,16,2,2}`)

**Legacy ambiguity:** flat `num_faces == 2` always maps to **1×2** (16×32-class), never **2×1** (32×16). Narrow tiles must use `make_tensor_shape` / metal `get_operand_tensor_shape`.

Python mirror: `tests/python_tests/helpers/tile_shape.py` (`TileShape`, `cpp_tensor_shape`).

## Coverage Model (Important)

Coverage is **TRISC-scoped**, not per-API:

| Macro | Checker header |
|-------|----------------|
| `LLK_VALIDATE_TENSOR_SHAPE_UNPACK(fn_name, ts)` | `tensor_shape_coverage_unpack.h` |
| `LLK_VALIDATE_TENSOR_SHAPE_MATH(fn_name, ts)` | `tensor_shape_coverage_math.h` |
| `LLK_VALIDATE_TENSOR_SHAPE_PACK(fn_name, ts)` | `tensor_shape_coverage_pack.h` (stub: always false until pack tables exist) |

Shared defs: `common/tensor_shape_coverage.h`.

### Myth vs reality (reviewer FAQ)

> “Every new API must be added to a list in `coverage.h`.”

**False for the current design.** There is **no** central API enum in `tensor_shape_coverage.h`. Call sites pass a **string literal** `fn_name` for DPRINT tagging only. Adding a new LLK that takes `TensorShape` means:

1. Include the matching TRISC coverage header.
2. Call `LLK_VALIDATE_TENSOR_SHAPE_{UNPACK|MATH|PACK}("_llk_foo_init_", tensor_shape)` early in init / mop_config / execute as appropriate.
3. **Do not** edit an API registry in `coverage.h`.

What *does* need updates when shapes/APIs change:

| Change | Where |
|--------|--------|
| New **observed shape** under asserts | Regenerate TRISC table via parser; add `TENSOR_SHAPE_FR*_NF*` named constant in `tensor_shape_coverage.h` if missing |
| New DPRINT `fn_name` that should feed harvest→emit unions | Add the string to `MATH_FUNCTIONS` / `UNPACK_FUNCTIONS` in `tests/python_tests/helpers/tensor_shape_coverage_parser.py` |
| New pack probes | Implement `is_pack_tensor_shape_covered` (today always `false`) |

Production builds without `ENABLE_LLK_ASSERT` / `DEBUG_PRINT_ENABLED` compile the macros to `((void)0)`.

### Discovery workflow

From `tests/python_tests`:

```bash
# Optional bootstrap from checked-in headers
python3 helpers/tensor_shape_coverage_parser.py seed

TT_LLK_DISABLE_ASSERTS=1 pytest --logging-level=DEBUG <tests>

python3 helpers/tensor_shape_coverage_parser.py harvest <label>
python3 helpers/tensor_shape_coverage_parser.py emit
python3 helpers/tensor_shape_coverage_parser.py summary
```

Default harvest state: `tests/python_tests/tensor_shape_coverage/coverage.json` (gitignored; override with `--coverage-json`).

Normal assert-enabled runs should **fail loudly** on unobserved shapes — that is intentional.

## Converting an LLK LIB Entry Point

Reference pattern: `llk_unpack_A` (WH/BH).

### Signature

Replace scattered tile args with one shape:

```cpp
// Before
inline void _llk_foo_init_(..., const std::uint32_t face_r_dim, const std::uint32_t num_faces, ...);

// After
#include "tensor_shape.h"
#include "tensor_shape_coverage_unpack.h"  // or _math.h / _pack.h

inline void _llk_foo_init_(..., const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE, ...);
```

Prefer `const ckernel::TensorShape&` only if the arch already uses references consistently; unpack_A currently passes by value (4 bytes).

### Body

```cpp
LLK_VALIDATE_TENSOR_SHAPE_UNPACK("_llk_foo_init_", tensor_shape);
const std::uint8_t face_r_dim = tensor_shape.face_r_dim;
const std::uint8_t num_faces  = tensor_shape.total_num_faces();
// Prefer num_faces_r_dim / num_faces_c_dim for new mop math; keep flat num_faces only when behavior is intentionally unchanged.
```

Keep existing unsupported-geometry `LLK_ASSERT`s. Do **not** silently generalize mop loops in the same PR as API plumbing unless tests prove the new paths (track follow-ups separately).

### Docs

- `@param tensor_shape: ...`
- Init/uninit contract: if uninit is a no-op, say so on **both** init and execute notes (do not claim “restore modified state” when uninit is empty).

### Mirror arches

Update WH and BH together. Quasar only when regressions exist / TensorShape is already the Quasar contract for that op.

## Converting the Metal LLK API Wrapper

Path: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_*_api.h`

```cpp
const ckernel::TensorShape tensor_shape = get_operand_tensor_shape(operand_id);
_llk_foo_init_<...>(..., tensor_shape, ...);
```

`get_operand_tensor_shape` lives in `llk_io/llk_operands.h` and uses real `unpack_num_faces_{r,c}_dim` — prefer it over `make_tensor_shape_from_legacy`.

Propagate per `.claude/references/metal-integration.md` (ckernels → compute API → TTNN bypasses → tests).

## Tests & Fuser

- C++ tests: `ckernel::make_tensor_shape(...)` when grid is known; avoid legacy helper for 32×16.
- Fuser emitters: `cpp_tensor_shape(tile_shape)` from `helpers/tile_shape.py`.
- Add targeted pytest cases for newly claimed shapes (see `test_unpack_A_targeted_tensor_shape_coverage`).

## Review Checklist

When reviewing or implementing TensorShape work, flag:

- [ ] New `_llk_*` taking tile geometry uses `TensorShape`, not new `face_r_dim`/`num_faces` params
- [ ] Matching `LLK_VALIDATE_TENSOR_SHAPE_*` on init (and mop_config / execute if they consume shape)
- [ ] Correct TRISC macro (unpack vs math vs pack)
- [ ] `fn_name` is a **string literal** matching the C++ symbol (for harvest)
- [ ] No new central API enum / per-function array in `coverage.h`
- [ ] New shapes either covered by regenerated TRISC tables or explicitly asserted-unsupported
- [ ] Legacy helper not used for narrow 32×16
- [ ] Metal wrapper uses `get_operand_tensor_shape` where CB metadata exists
- [ ] WH/BH (and Quasar if in scope) stay in sync
- [ ] Parser `MATH_FUNCTIONS` / `UNPACK_FUNCTIONS` updated if a new probe name should contribute to emit
- [ ] Doc notes do not claim uninit restores state when it is a no-op
