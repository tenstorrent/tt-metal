# Ternary + Special + Misc Investigation Report — Phase 1

## Scope (deduped, 23 ops)

- **Ternary** (5): addcmul, addcdiv, lerp, where, mac (**mac NOT FOUND in LLK — flag as missing primitive**)
- **Special** (6): clamp, fmod, lerp, remainder, where, xlogy
- **Misc** (16): add_top_row, alt_complex_rotate90, copy_dest_values, dropout, fill, gcd_tile, heaviside, identity, lcm_tile, mask, max_pool_indices, rand, reduce, reshuffle, sfpu_int_sum, tiled_prod

## Critical Contracts

### Ternary Slot Layout

All ternary SFPU ops enforce `(idst0, idst1, idst2, odst)` order via LLK signature. No permutation. Helper must template `<In0, In1, In2, Out>` with all 4 slots.

```cpp
// addcmul / addcdiv / lerp / where signature shape:
template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_ternary_sfpu_addcmul(
    uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst,
    int vector_mode = (int)VectorMode::RC);
```

### Mask DataSlot+1 Contract (lessons §1.4)

`mask_tile()` reads mask from `idst_data + 1` unconditionally, ignoring `idst2_mask` parameter passed.

**Encode at compile time** to prevent misuse:
```cpp
template <DataFormat DF, Dst DataSlot>
struct Mask : BinaryOp<Mask<DF, DataSlot>, DataSlot,
                       static_cast<Dst>(static_cast<uint32_t>(DataSlot) + 1),
                       DataSlot> {
    static_assert(static_cast<uint32_t>(DataSlot) < 7, "Mask requires DataSlot < 7");
};
```

### FP32 Accumulation in Ternary

`addcmul` and `addcdiv` expose `is_fp32_dest_acc_en` template param matching kernel's `DST_ACCUM_MODE`.

### RNG State Management

`dropout` and `rand` both set global RNG seed → mutual exclusion within same kernel invocation. Helper must document or enforce.

### Cross-Iteration State (sfpu_int_sum, tiled_prod)

Incompatible with standard per-tile eltwise loop (`acquire → init+exec → release`). Requires custom sequencing where DEST is held across iterations.

**Out of scope for eltwise helper** — flag as `BLOCKER:CROSS_ITERATION_STATE` per lessons §3.7.

## Op Categorization

### IN SCOPE (14–16 ops)

| Op | CRTP Base | Notes |
|---|---|---|
| addcmul | TernaryOp | `<In0, In1, In2, Out>` + APPROX + fp32_dest_acc |
| addcdiv | TernaryOp | Same shape; numerically sensitive |
| lerp | TernaryOp | (start, end, weight, out) |
| where | TernaryOp | (cond, true, false, out); cond is selector |
| clamp | UnaryOp w/ 2 scalars | Already covered in unary omnibus (Family 3) |
| fmod | UnaryOp w/ scalar+recip | Reciprocal pre-computed on host |
| remainder | UnaryOp w/ scalar+recip | Same as fmod |
| fill | UnaryOp w/ scalar | Materializes scalar into tile |
| xlogy | BinaryOp | x * log(y) |
| mask | BinaryOp w/ DataFormat + hardcoded slot+1 | Per §1.4 |
| heaviside | UnaryOp w/ scalar | Cutoff value |
| gcd_tile | BinaryOp | int32 binary |
| lcm_tile | BinaryOp | int32 binary |
| dropout | UnaryOp w/ RNG seed + prob + scale | Init programs RNG |
| rand | UnaryOp w/ RNG seed | Init programs RNG |

### FIX-AND-CONTINUE (lessons §11.2 — 4-line struct addition)

- gcd_tile, lcm_tile, heaviside, alt_complex_rotate90

### OUT OF SCOPE

| Op | Reason |
|---|---|
| reduce | Different category (reduction); separate CB lifecycle. Already has `reduce_helpers_*` |
| sfpu_int_sum | Cross-iteration state; held DEST across tiles |
| tiled_prod | Cross-iteration state; multi-tile product accumulator |
| reshuffle | External L1 index dependency; caller-managed |
| add_top_row | Specialized DEST helper; low adoption, kernel-specific |
| copy_dest_values | DEST-to-DEST copy; not a CB op (helper-internal at most) |
| max_pool_indices | Pooling-specific; not generic eltwise |
| identity | Trivial copy; better expressed as raw `copy_tile` |
| mac | NOT FOUND in LLK — missing primitive, flag as gap |

## Wrapper Signatures

### Ternary (canonical)

```cpp
// addcmul
template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_ternary_sfpu_addcmul_init();

template <bool APPROXIMATE, bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_ternary_sfpu_addcmul(
    uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst,
    int vector_mode = (int)VectorMode::RC);
```

Same shape for `addcdiv`, `lerp`, `where`.

### Mask (data-format dependent)

```cpp
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mask(
    uint dst_index, DataFormat data_format, int vector_mode);
```

### Dropout / Rand (RNG-state)

```cpp
inline void llk_math_eltwise_unary_sfpu_dropout_init(uint32_t seed);

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_dropout(
    uint dst_index, int integer_dropout, int scale_factor, int vector_mode);
```

## Init State Compatibility

| Op A | Op B | Compatible? | Reason |
|---|---|---|---|
| addcmul_init | addcdiv_init | No | Both program ternary SFPU mode; different ops |
| dropout_init | rand_init | No | Both set global RNG seed |
| mask_init | (any plain) | Yes | Mask init trivial |
| fill_init | scalar ops | Yes | Both trivial |
| ternary | unary SFPU LUT op | No | Conservative — different SFPU resources |

## Compile-Time Feature Matrix

| Flag | Op | Becomes |
|---|---|---|
| APPROXIMATE | all SFPU | Template `Approx` |
| is_fp32_dest_acc_en | addcmul, addcdiv | Template bool |
| dropout integer_prob | dropout | Runtime arg |
| dropout scale_factor | dropout | Runtime arg (host pre-compute) |
| RNG seed | dropout, rand | Runtime arg or compile-time |

## Recommendations

1. **TernaryOp CRTP base**: `<Derived, In0, In1, In2, Out>` covers addcmul/addcdiv/lerp/where.
2. **Mask hardcoded slot pattern**: encode `Slot+1` at compile time per lessons §1.4.
3. **RNG documentation**: Note dropout_init / rand_init mutual exclusion at helper-level docs.
4. **Pre-compute reciprocals**: fmod/remainder host pre-compute, passed as runtime args.
5. **Mark out-of-scope ops** in feature_gap_map: sfpu_int_sum, tiled_prod (CROSS_ITERATION_STATE), reduce (OTHER_CATEGORY), reshuffle (EXTERNAL_INDEX).
6. **mac**: missing primitive — flag as gap; do NOT introduce op struct until LLK exists.

## Files Analyzed

- `tt_metal/hw/inc/api/compute/eltwise_unary/*.h` (18 headers, mask, dropout, rand, fill, etc.)
- LLK ternary headers: `llk_math_eltwise_ternary_sfpu_*.h` (addcmul, addcdiv, lerp, where)
- TTNN codegen: ternary_op_utils, unary_op_utils, binary_op_utils
- `eltwise_helper_lessons.md` (design principles, §1.4, §3.7, §11)

---

Generated: 2026-04-30
Investigation Scope: ternary + special + misc (23 ops deduped)
