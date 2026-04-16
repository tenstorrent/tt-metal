# Compute API Cleanup — Demo Guide

**Target**: 2-week demo showing `transformer_attn_matmul.cpp` before/after all cleanups.
**Scope**: matmul + tilize + untilize APIs only (not the full 70-op cleanup).
**Issues**: [#22219](https://github.com/tenstorrent/tt-metal/issues/22219), [#22907](https://github.com/tenstorrent/tt-metal/issues/22907), [#22946](https://github.com/tenstorrent/tt-metal/issues/22946), [#33823](https://github.com/tenstorrent/tt-metal/issues/33823)

---

## 1. What these changes are trying to achieve

The Compute API — the interface kernel writers use to program Tensix cores — has
accumulated inconsistencies that make kernels harder to write, read, and maintain:

- **"Long" vs "short" inits**: Every operation has a "full init" that bundles one-time
  HW configuration with operation-specific setup, plus multiple "short init" variants
  (`_short`, `_short_with_dt`, `_short_with_both_dt`). Kernel writers must choose
  between 5+ init variants per operation and reason about which HW state each one touches.

- **Ambiguous execute API names**: The function that actually *does* the work has
  different naming patterns per operation — `matmul_tiles` vs `matmul_block` vs
  `tilize_block` vs `untilize_block` vs `pack_untilize_block`. Some names don't
  clearly convey what they do.

- **Inconsistent reconfig ceremony**: After switching between operations mid-kernel,
  the programmer must manually call the right combination of `reconfig_data_format_srca`,
  `reconfig_data_format`, `pack_reconfig_data_format`, and then a `_with_dt` init
  variant. This is error-prone and leaks HW details (the matmul srcA/srcB swap)
  into kernel code.

- **Mixed prefixes**: Matmul uses both `mm_*` and `matmul_*` prefixes for related
  functions. Init functions sometimes include `out_cb_id` and sometimes don't.

The cleanup goal is to make a kernel like `transformer_attn_matmul.cpp` go from
~100 lines of init/reconfig ceremony down to a handful of self-explanatory calls.

---

## 2. The desired programming model

Every compute kernel should follow this structure:

```
compute_kernel_hw_startup(in0, in1, out)    // once, at kernel start — op-agnostic

op_init(...)                                 // before each operation
op_block(...)                                // execute the operation
op_uninit(...)                               // after each operation — restores HW state
```

The `op_init` / `op_block` / `op_uninit` triplet is the standard lifecycle for every
operation. `op_init` sets up operation-specific HW state (unpack mode, math MOP, etc.),
`op_block` executes, and `op_uninit` restores any state that the operation modified so
the next operation starts from a clean baseline. This eliminates the need for the caller
to manually figure out what reconfig calls are needed between operations — each operation
cleans up after itself.

### Rules

| Rule | Before (broken) | After (target) |
|------|-----------------|----------------|
| HW config happens once | Buried inside each "long init" | Single `compute_kernel_hw_startup()` call |
| One init per operation | `mm_init`, `mm_init_short`, `mm_init_short_with_dt`, `mm_block_init`, `mm_block_init_short`, `mm_block_init_short_with_dt`, `mm_block_init_short_with_both_dt` (7 variants) | `matmul_init` (1 function) |
| One uninit per operation | Missing for matmul; inconsistent `_uninit` vs `_uninit_with_dt` for tilize | `matmul_uninit` / `tilize_uninit` / `untilize_uninit` — each restores the state its init modified |
| Execute API naming | `matmul_tiles`, `tilize_block`, `untilize_block` | `matmul_block`, `tilize_block`, `untilize_block` |
| Data format reconfig | Manually call `reconfig_data_format_srca(old, new)` + `pack_reconfig_data_format(old, new)` + `mm_init_short_with_dt(...)` | Handled by `op_uninit` + `op_init` — each operation restores state on exit, next operation sets its own state on entry |
| Matmul operand swap | Kernel writer must know `in0→srcB, in1→srcA` and swap args to `compute_kernel_hw_startup` | Hidden inside `matmul_init`; kernel writer passes CBs in natural order |

### Target kernel pattern

```cpp
compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);     // once — op-agnostic, standard CB order

for (...) {
    matmul_init<cb_in0, cb_in1, cb_out, transpose>();   // init matmul mode
    matmul_block(cb_in0, cb_in1, ...);                   // execute matmul
    matmul_uninit();                                     // restore state
    // ... pack ...

    untilize_init<...>();                                // init untilize mode
    untilize_block<...>(...);                            // execute untilize
    untilize_uninit<...>();                              // restore state

    tilize_init<...>();                                  // init tilize mode
    tilize_block<...>(...);                              // execute tilize
    tilize_uninit<...>();                                // restore state
}
```

With the `compute_kernel_lib` wrappers (which bundle init+block+uninit), this
simplifies further — the kernel writer just calls the wrapper and the lifecycle
is handled automatically:

```cpp
compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);

for (...) {
    compute_kernel_lib::matmul_init<cb_in0, cb_in1, cb_out, transpose>();
    matmul_block(cb_in0, cb_in1, ...);
    compute_kernel_lib::matmul_uninit();
    // ... pack ...

    compute_kernel_lib::untilize<...>(...);              // init + block + uninit in one call
    compute_kernel_lib::tilize<...>(...);                // init + block + uninit in one call
}
```

Every operation ALWAYS follows the `init → block → uninit` triplet, even when
`uninit` compiles to a no-op (as is the case for matmul). This keeps
the programming model consistent and future-proof — if matmul's uninit ever needs
to restore state, all call sites are already correct.

### What "only `op_init` / `op_block` / `op_uninit`" means concretely

**Init** — one function per operation, no `_short`/`_with_dt` variants:

| Operation | Current variants to DELETE | Replacement |
|-----------|--------------------------|-------------|
| **matmul** | `mm_init`, `mm_init_short`, `mm_init_short_with_dt`, `mm_block_init`, `mm_block_init_short`, `mm_block_init_short_with_dt`, `mm_block_init_short_with_both_dt` | `matmul_init` (in `matmul_helpers.hpp`) |
| **tilize** | `tilize_init_short_with_dt`, `fast_tilize_init_with_dt` | Already done — `compute_kernel_lib::tilize<>()` wraps init internally |
| **untilize** | (already clean) | Already done — `compute_kernel_lib::untilize<>()` wraps init internally |

**Uninit** — one function per operation, no `_with_dt` variants. Each uninit restores
the state that its corresponding init modified, so the next operation starts clean:

| Operation | Current variants to DELETE | Replacement |
|-----------|--------------------------|-------------|
| **matmul** | (none exists — matmul currently has no uninit, leaving the caller to manually reconfig) | `matmul_uninit` (compiles to no-op today, but must always be called for consistency) |
| **tilize** | `tilize_uninit_with_dt` | `tilize_uninit` — already wrapped by `compute_kernel_lib::tilize<>()` |
| **untilize** | (already clean — `untilize_uninit` exists) | Already done — wrapped by `compute_kernel_lib::untilize<>()` |

### What "only `op_block`" means concretely

| Current name | Problem | Target name |
|-------------|---------|-------------|
| `matmul_tiles` | Misleading — it processes tiles, but so does `matmul_block`. The "tiles" suffix is inconsistent with `tilize_block`/`untilize_block`. | `matmul_block` (already exists for the block variant, unify the tile variant into it with dims=1) |
| `tilize_block` | OK | Keep as-is (or use `compute_kernel_lib::tilize<>`) |
| `untilize_block` | OK | Keep as-is (or use `compute_kernel_lib::untilize<>`) |

### What "clean reconfig" means — current vs near-term vs long-term

**Long-term vision**: `op_init` and `op_uninit` track the *current* operand state
(data format, tile dimensions, strides, etc.) in a lightweight structure. When you
call `matmul_init(cb_in0, cb_in1)`, it compares the new CB metadata against the
current state and only emits register writes when something actually changed. The
kernel writer never thinks about reconfig — it happens silently inside init/uninit.

**Why we can't do that yet**: We don't have a structure that tracks current operand
state at runtime. Building one is real work and out of scope for the demo.

**Near-term (what we do now)**: Reconfig calls remain explicit, but we clean up the
naming and simplify the API surface. The kernel writer still calls reconfig between
operations, but the calls are self-explanatory and consistent.

#### Rename: `reconfig_data_format_*` → `reconfig_operand_*`

| Current name | Problem | New name |
|-------------|---------|----------|
| `reconfig_data_format_srca(old, new)` | Leaks HW register names (`srca`) into kernel code | `reconfig_operand_srca(old, new)` |
| `reconfig_data_format_srcb(old, new)` | Same | `reconfig_operand_srcb(old, new)` |
| `reconfig_data_format(old_a, new_a, old_b, new_b)` | 4-arg overload, confusing which pair is which | `reconfig_operand(old_a, new_a, old_b, new_b)` |
| `pack_reconfig_data_format(old, new)` | Inconsistent prefix vs the above | `reconfig_pack(old, new)` |

Keep the `_srca`, `_srcb`, and 2-operand versions. If the user doesn't know which
source register changed, they can always call the 2-operand version — it checks both
and only reconfigures what actually differs. Today this check is at runtime (comparing
CB format descriptors); in the future we can move it to compile time using the
JIT-generated `unpack_src_format[]` arrays, making the unused branch a dead-code
elimination.

#### What the kernel looks like now (near-term)

```cpp
compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);
matmul_init<cb_in0, cb_in1, cb_out, transpose>();

for (...) {
    matmul_block(cb_in0, cb_in1, ...);
    matmul_uninit();

    // reconfig still explicit, but with clear names
    reconfig_operand_srca(cb_intermed0, cb_in1);
    untilize_init<...>();
    untilize_block<...>(...);
    untilize_uninit<...>();

    reconfig_operand(cb_intermed2, cb_in1, cb_intermed2, cb_in0);
    reconfig_pack(out_cb, cb_intermed0);
    matmul_init<cb_in0, cb_in1, cb_out, transpose>();
}
```

#### What the kernel will look like (long-term, with state tracking)

```cpp
compute_kernel_hw_startup(cb_in0, cb_in1, cb_out);

for (...) {
    matmul_init<cb_in0, cb_in1, cb_out, transpose>();   // silently reconfigures if needed
    matmul_block(cb_in0, cb_in1, ...);
    matmul_uninit();

    untilize_init<...>();                                // silently reconfigures if needed
    untilize_block<...>(...);
    untilize_uninit();

    tilize_init<...>();                                  // silently reconfigures if needed
    tilize_block<...>(...);
    tilize_uninit();

    // no manual reconfig anywhere — init/uninit handle everything
}
```

---

## 3. How to perform the changes

### Strategy: experimental/custom shim layer — don't touch `matmul.h`

We do **not** modify `matmul.h` or any other existing Compute API headers. Changing
`matmul.h` would force changes to all ~70 kernel files that use `mm_init`, which is
out of scope for the demo.

Instead, we create a small shim header in the experimental/custom include path that
wraps the existing APIs under the new names. The demo kernel includes this shim and
uses the clean API. All other kernels are untouched.

### Files to create/modify

| File | What to do |
|------|-----------|
| **`tt_metal/hw/inc/api/compute/experimental/matmul_api.h`** | **CREATE** — shim header that implements `matmul_init`, `matmul_uninit`, `matmul_block` by calling existing `mm_init_short`, `matmul_tiles`, etc. under the hood. |
| **`ttnn/.../transformer_attn_matmul.cpp`** | **MODIFY** — the demo kernel. Include the experimental shim and use the clean API. |

That's it. Two files. Everything else stays the same.

The shim header lives alongside the existing experimental headers (e.g.,
`tt_metal/hw/inc/api/compute/experimental/sdpa_sub_custom.h`) and follows the same
pattern — a thin wrapper over existing APIs that can be promoted to the main API
once the full cleanup lands.

### Where to look for patterns

- **Tilize helper** (already done, use as template): `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` + `.inl`
- **Untilize helper** (already done): `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` + `.inl`
- **Compute kernel hw startup**: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`
- **Reconfig functions**: `tt_metal/hw/inc/api/compute/reconfig_data_format.h`
- **Existing experimental shims**: `tt_metal/hw/inc/api/compute/experimental/`
- **Existing kernel using new pattern**: `ttnn/.../sdpa_decode/.../sdpa_flash_decode.cpp` (uses `compute_kernel_hw_startup` + `mm_init_short`)

### Test

```bash
cd /localdev/ncvetkovic/tt-metal
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
source python_env/bin/activate
rm -rf ~/.cache/tt-metal-cache/

# All attn_matmul tests (covers bf16, bfp8_b, fp32, mixed precision, program cache, sharding)
pytest -x tests/ttnn/nightly/unit_tests/operations/matmul/test_attn_matmul.py -v
```

19 tests total, covering all dtype combinations and sharding configs. Already validated
on Blackhole with the intermediate refactoring (all 19 tests pass on BH).

### Files in this folder

| File | Description |
|------|-------------|
| `1_before_kernel_lib.cpp` | Original kernel — raw init/uninit/reconfig ceremony (~100 lines) |
| `2_current_with_kernel_lib.cpp` | Current main — tilize/untilize wrapped, matmul still uses old API |
| `3_with_hw_startup.cpp` | Tested intermediate — `compute_kernel_hw_startup` + explicit reconfig (all 19 tests pass on BH) |
| `4_proposed_with_matmul_init.cpp` | End goal — `compute_kernel_lib::matmul_init` hides everything |
| `matmul_helpers.hpp` | Implementation of `compute_kernel_lib::matmul_init` / `matmul_uninit` / `matmul_block` |
| `DEMO_GUIDE.md` | This file |
