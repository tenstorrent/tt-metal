# BINARY GROUP INVESTIGATION REPORT — Phase 1 (Eltwise Binary Operations)

## Executive Summary

This investigation covers 34 binary operations across two architectural tiers:
- **FPU binary ops** (4): `add_tiles`, `sub_tiles`, `mul_tiles`, `mul_tiles_bcast`
- **SFPU binary tile ops** (30): `add_binary_tile`, `sub_binary_tile`, `mul_binary_tile`, `div_binary_tile`, `rsub_binary_tile`, `power_binary_tile`, `binary_max_tile` (+int32/uint32), `binary_min_tile` (+int32/uint32), `eq/ne/gt/ge/lt/le_binary` (6 comparisons), `binary_pow`, `binary_remainder`, `binary_fmod`, `binary_shift`, `atan2_binary_tile`, `logsigmoid`, plus SFPU broadcast family (`*_bcast_col`/`*_bcast_row`).

**Catalog Issue Found**: `logsigmoid` correctly listed as binary (unary form wrapper around binary SFPU op internally; takes two operands in DEST).

## 1. Wrapper Signatures

| Op | Init Signature | Exec Signature | Template Params | Runtime Params |
|---|---|---|---|---|
| **add_tiles** | `add_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false)` | `add_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | `EltwiseBinaryType::ELWADD`, `BroadcastType::NONE`, `MathFidelity::LoFi` | icb0, icb1, itile0, itile1, idst |
| **sub_tiles** | `sub_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest = false)` | `sub_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | `EltwiseBinaryType::ELWSUB`, `BroadcastType::NONE`, `MathFidelity::LoFi` | icb0, icb1, itile0, itile1, idst |
| **mul_tiles** | `mul_tiles_init(uint32_t icb0, uint32_t icb1, uint32_t acc_to_dest = true)` | `mul_tiles(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)` | `EltwiseBinaryType::ELWMUL`, `BroadcastType::NONE`, `MATH_FIDELITY` | icb0, icb1, itile0, itile1, idst |
| **mul_tiles_bcast** | `binary_tiles_init<true, ELWMUL, BCAST_TYPE>(icb0, icb1)` | `mul_bcast_tiles(icb0, icb1, idst_a, idst_b, idst_out)` | `BroadcastType::{NONE, ROW, COL, SCALAR}` | icb0, icb1, idst_a, idst_b, idst_out |
| **add_binary_tile** | `add_binary_tile_init()` | `add_binary_tile(uint32_t idst0, uint32_t idst1, uint32_t odst)` | `APPROX` | idst0, idst1, odst |
| **sub_binary_tile** | `sub_binary_tile_init()` | `sub_binary_tile(...)` | `APPROX` | idst0, idst1, odst |
| **mul_binary_tile** | `mul_binary_tile_init()` | `mul_binary_tile(...)` | `APPROX`, `DST_ACCUM_MODE` | idst0, idst1, odst |
| **div_binary_tile** | `div_binary_tile_init()` | `div_binary_tile(...)` | `APPROX`, `DST_ACCUM_MODE` | idst0, idst1, odst |
| **rsub_binary_tile** | `rsub_binary_tile_init()` | `rsub_binary_tile(...)` | `APPROX` | idst0, idst1, odst |
| **power_binary_tile** | `power_binary_tile_init()` | `power_binary_tile(...)` | `APPROX`, `DST_ACCUM_MODE` | idst0, idst1, odst |
| **eq/ne/lt/gt/le/ge_binary_tile** | `<op>_binary_tile_init()` | `<op>_binary_tile(...)` | `APPROX` | idst0, idst1, odst |
| **binary_max_tile** | `binary_max_tile_init()` | `binary_max_tile(idst0, idst1, odst, vector_mode = VectorMode::RC)` | `APPROX` | idst0, idst1, odst, vector_mode |
| **binary_max/min_int32_tile** | `<op>_int32_tile_init()` | `<op>_int32_tile(...)` | `APPROX` | idst0, idst1, odst |
| **binary_max/min_uint32_tile** | `<op>_uint32_tile_init()` | `<op>_uint32_tile(...)` | `APPROX` | idst0, idst1, odst |
| **binary_min_tile** | `binary_min_tile_init()` | `binary_min_tile(idst0, idst1, odst, vector_mode = VectorMode::RC)` | `APPROX` | idst0, idst1, odst, vector_mode |
| **binary_pow** | `binary_pow_init()` | `binary_pow(...)` | `APPROX`, `DST_ACCUM_MODE` | idst0, idst1, odst |
| **binary_remainder** | `binary_remainder_init()` | `binary_remainder(...)` | `APPROX` | idst0, idst1, odst |
| **binary_fmod** | `binary_fmod_init()` | `binary_fmod(...)` | `APPROX` | idst0, idst1, odst |
| **binary_shift** | `binary_shift_init()` | `binary_shift(...)` | `APPROX` | idst0, idst1, odst |
| **atan2_binary_tile** | `atan2_binary_tile_init()` | `atan2_binary_tile(...)` | `APPROX` | idst0, idst1, odst |
| **logsigmoid** | `logsigmoid_tile_init()` | `logsigmoid_tile(idst_in0, idst_in1, idst_out)` | `APPROX` | idst_in0, idst_in1, idst_out |
| **binary_dest_reuse_tiles (DEST_TO_SRCA)** | `binary_dest_reuse_tiles_init<EltwiseBinaryType, DEST_TO_SRCA>(icb0)` | `binary_dest_reuse_tiles<EltwiseBinaryType, DEST_TO_SRCA>(icb_id, in_tile_index, dst_tile_index)` | `EltwiseBinaryType`, `EltwiseBinaryReuseDestType::DEST_TO_SRCA` | icb_id, in_tile_index, dst_tile_index |
| **binary_dest_reuse_tiles (DEST_TO_SRCB)** | `<...DEST_TO_SRCB>` init | `<...DEST_TO_SRCB>` exec | `EltwiseBinaryType`, `DEST_TO_SRCB` | icb_id, in_tile_index, dst_tile_index |

## 2. Init State Compatibility

| Op A Init | Op B Init | Mutually Exclusive? | Reason |
|---|---|---|---|
| `add_tiles_init` | `sub_tiles_init` | Yes | Both FPU; reconfigure math MOP to different EltwiseBinaryType |
| `add_tiles_init` | `mul_tiles_init` | Yes | ELWADD vs ELWMUL |
| `sub_tiles_init` | `mul_tiles_init` | Yes | ELWSUB vs ELWMUL |
| `add_binary_tile_init` | `mul_binary_tile_init` | Yes | SFPU; different binop_init paths |
| FPU init | SFPU init | Yes | Different hardware paths (FPU unpack AB vs SFPU dest-only) |
| `binary_max_tile_init` | `binary_min_tile_init` | Yes (dedup at caller) | Share underlying `binary_max_init()` LLK; calling both is wasted work |
| `binary_max_tile_init` | `binary_max_int32_tile_init` | Yes | Float vs INT32 paths |
| `binary_remainder_init` | `binary_fmod_init` | Likely Yes | Both remainder-family; assume exclusive |
| Standard SFPU init | `binary_dest_reuse_tiles_init` | Yes | DEST reuse uses single-operand unpack path; FPU-clash incompatible |
| `dest_reuse<ADD>` | `dest_reuse<MUL>` | No (template differs) | Same init logic, different op type at exec |

**Rule**: FPU and SFPU paths are orthogonal — never mix in a single loop.

## 3. DEST Batching Limits

| Op Class | Max Tiles Per DEST Batch | FP32 Accumulation Required? |
|---|---|---|
| FPU binary (add/sub/mul_tiles) | 8 (4 if FP32_DEST_ACC) | mul_tiles: yes if MATH_FIDELITY strict |
| SFPU binary in-DST (add/sub/mul/div/rsub_binary_tile, etc.) | 4 tiles total (2 pairs); 2 if FP32 | varies: mul/div/pow → yes |
| SFPU broadcast (`*_bcast_col`, etc.) | Same as SFPU | per-op |
| Predicates (eq/ne/lt/gt/le/ge_binary_tile) | 4 tiles | No |
| Max/Min (float + int32 + uint32) | 4 tiles | No |
| binary_pow / power_binary_tile | 4 tiles | Yes |
| binary_remainder/fmod/shift/atan2 | 4 tiles | No |
| binary_dest_reuse_tiles | 8 (4 if FP32) for output | per op type |

## 4. Call Sites (representative)

| Op | File | Pattern | Init Placement |
|---|---|---|---|
| mul_tiles | `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu.cpp` | `mul_tiles_init(cb1, cb2); for (...) mul_tiles(cb1, cb2, i, j, out_idx);` | Before tile loop |
| add_binary_tile | `binary_op_utils.cpp` macro injection | `BINOP_INIT → init(); ELTWISE_OP → tile_op` | Compile-time macro |
| mul_binary_tile | `tanh_bw/...eltwise_bw_tanh_deriv.cpp` | `mul_binary_tile_init(); for (...) mul_binary_tile(0,1,0);` | Upfront |
| binary_op_init_common | `binary_ng/.../eltwise_binary_no_bcast.cpp` | Once at kernel start | Once |
| SFPU binary | `binary_ng/.../eltwise_binary_sfpu_no_bcast.cpp` | `copy_tile_to_dst_init_short_with_dt(); copy_tile(...); BINARY_SFPU_OP(i*2, i*2+1, i*2);` | Macro BINARY_SFPU_INIT |

## 5. Init/Exec Pairing Rules

| Op Family | Pairing |
|---|---|
| FPU Binary | `binary_op_init_common(icb0, icb1, ocb)` + `binary_tiles_init<EltwiseBinaryType>(icb0, icb1)`; exec: `add_tiles(...)`, `sub_tiles(...)`, `mul_tiles(...)`. Init template type must match exec op |
| SFPU Binary | `<op>_binary_tile_init()` + `<op>_binary_tile(idst0, idst1, odst)`. Exclusive per family |
| Max-Min | Init shared underlying LLK; dedup at caller |
| SFPU Broadcast | Single `bcast_col_init()` covers SUB/ADD/MUL on that dim |
| DEST Reuse | `binary_dest_reuse_tiles_init<Op, ReuseType>(icb0)` + `binary_dest_reuse_tiles<Op, ReuseType>(icb_id, in_idx, dst_idx)`. Template type must match |

## 6. Chaining Patterns

| Pattern | Ops | Description |
|---|---|---|
| FPU → FPU sequential | add_tiles → mul_tiles | Each op's own init; unpack AB shared, math MOP reconfigured |
| SFPU in-loop | add_binary_tile → rsub_binary_tile | Reuse DEST slots without full reinit when ops don't overlap HW state |
| SFPU broadcast multi-op | sub_bcast_col + add_bcast_col + mul_bcast_col | Single bcast_col_init covers all three |
| DEST Reuse (CB+DEST → DEST) | binary_dest_reuse_tiles<ELWMUL> | Load CB + DEST (accumulator), multiply, write back to DEST. FPU-clash: per-tile reinit needed |
| Broadcast + post-activation | sub_bcast_col + relu/exp | Activation as separate SFPU call after broadcast |

## 7. Compile-Time Feature Matrix

| Flag | Affects Loop? | Classification | Should Become |
|---|---|---|---|
| `PACK_RELU` | pack stage | Adjacent operation | `OutputActivation::None/Relu` template enum |
| `FP32_DEST_ACC_EN` | DEST capacity | Loop-internal | `FP32DestAccum::On/Off` template enum |
| `BCAST_*` | unpacker, broadcast | Loop-internal | `BroadcastDim` template enum (and `BroadcastSide::LHS/RHS`) |
| `ELTWISE_OP / ELTWISE_OP_TYPE` | dispatch | Loop-internal | `BinaryOpType` enum template |
| `BINARY_OP_TYPE` | FPU vs SFPU dispatch | Loop-internal | `BinaryOpType` template |
| `BINARY_SFPU_INIT` | gate per-tile vs upfront | Loop-internal | `InitFrequency::PerTile/Upfront` policy |
| `BINARY_SFPU_OP` | dispatch SFPU op name | Loop-internal | Template param |
| `HAS_ACTIVATIONS(SIDE)` | preprocess + init placement | Loop-internal | `bool PreActivationA/B` template + activation chain |
| `MATH_FIDELITY` | FPU precision | Loop-internal | `MathFidelity::LoFi/Hi` template |
| `DST_ACCUM_MODE` | DEST packing | Loop-internal | implicit via FP32 mode |
| `APPROX` | SFPU approx level | Loop-internal | `Approx::Fast/Exact` template |

**Pattern**: every flag → template parameter with sensible default.

## 8. CB Compile-Time Analysis

| CB | Declared As | Varies at Runtime? | Recommendation |
|---|---|---|---|
| icb0 / icb1 / ocb | `constexpr uint32_t` (in test kernels); runtime in production | conservative: runtime | Runtime params at exec; can be template if always constexpr at given call site |
| cb_post_lhs / cb_post_rhs | `HAS_ACTIVATIONS(LHS) ? c_3 : c_0` | compile-time conditional | Template bool gates CB selection |

## 9. SrcA vs SrcB Reconfig Path Enumeration

Per lessons §8.1: srcA-reconfig and srcB-reconfig are separate LLK paths:

| Op | SrcA Reconfig Path | SrcB Reconfig Path |
|---|---|---|
| add_tiles / sub_tiles / mul_tiles | Yes (via `llk_unpack_AB`) | Yes (via `llk_unpack_AB`) |
| add_binary_tile / mul_binary_tile etc. (in-DEST) | N/A (no CB inputs) | N/A |
| binary_dest_reuse_tiles<CB_TO_SRCA> | Yes (single-input unpacker, CB → srca) | No (DEST → srcb hardwired) |
| binary_dest_reuse_tiles<DEST_TO_SRCB> | No (DEST → srca hardwired) | Yes (CB → srcb path) |

**Test coverage required**:
- FPU srcA-only reconfig (A dtype changes, B unchanged)
- FPU srcB-only reconfig
- FPU both reconfigured
- DEST-reuse CB_TO_SRCA with CB dtype change
- DEST-reuse DEST_TO_SRCB with CB dtype change
- Cross with broadcast dims (bcast may change unpacker path)

## 10. BroadcastDim Validation

Proposed enum:
```cpp
enum class BroadcastDim {
    NONE = 0,    // [Ht, Wt]
    ROW = 1,     // [1, Wt] — broadcast height direction
    COL = 2,     // [Ht, 1] — broadcast width direction
    SCALAR = 3   // [1, 1]
};
```

- ✓ Maps 1:1 to LLK `BroadcastType` (NONE/ROW/COL/SCALAR exist).
- ✓ Opposite-axis mapping (REDUCE_ROW → COL broadcast) accurate.
- ✗ **`mul_tiles_bcast` is NOT a separate LLK function** — broadcast is encoded as BroadcastType template param to `binary_tiles_init` and `llk_math_eltwise_binary`. Same for `add_tiles_bcast`, `sub_tiles_bcast`. Catalog should clarify.
- SFPU broadcast is more explicit (`*_bcast_col`/`*_bcast_row`) but still shares generic init per dimension.

## 11. DestReuseOp Pattern Validation

Per lessons §1.7 / §3.4:

```cpp
binary_dest_reuse_tiles_init<Op, DEST_TO_SRCA>(icb0);
binary_dest_reuse_tiles<Op, DEST_TO_SRCA>(icb_id, in_tile_idx, dst_tile_idx);
// or DEST_TO_SRCB inverse
```

**Search**:
- Found in ternary addcdiv/addcmul patterns (implicit; not explicit DEST_TO_SRCA)
- Likely used in batch_norm stage 2 (mul with accumulator); no kernel found in this scan
- Not heavily used in forward eltwise; mostly backward + fusion

**Implication**: real pattern; must support distinct call signature. Testing must include batch_norm-like mul accumulation.

## 12. Catalog Issue Notes

1. **logsigmoid**: correctly placed in binary group (architectural binary, API unary)
2. **mul_tiles_bcast**: not a distinct LLK function — mode of mul_tiles via BroadcastType template
3. **binary_pow vs power_binary**: distinct SFPU functions; both correct
4. **No missing ops** in scan

## 13. Helper Design Recommendations

### Canonical Call Shapes (4 forms)

1. **FPU Binary** (CB inputs, CB output)
2. **SFPU Binary** (DEST inputs, DEST output)
3. **SFPU Broadcast** (DEST + broadcast shape)
4. **DEST Reuse** (CB + DEST → DEST)

### Init Mutual Exclusion Summary

- Same family: exclusive per op type
- FPU vs SFPU: orthogonal, never mix
- Max-min family: share underlying init; dedup at caller
- Broadcast: one init per dim covers multiple ops on that dim
- DEST reuse: requires dedicated lighter init (incompatible with standard SFPU init)

### Feature flags → template params (default safe option)

`FP32DestAccum::Off`, `MathFidelity::LoFi`, `Approx::Fast`, `BroadcastDim::NONE`, `OutputActivation::None`, `PreActivationA/B = false`.

### Test Matrix
- num_tiles ∈ {1, 8, 64}
- fp32_dest_acc_en ∈ {false, true}
- Each srcA/srcB reconfig path independently
- Each ReuseType (CB_TO_SRCA, DEST_TO_SRCB)
- BroadcastDim ∈ {NONE, ROW, COL, SCALAR}

---

Generated: 2026-04-30
Investigation Scope: Binary group, 34 ops
Architecture: Wormhole_B0 (Blackhole identical LLK prefix sets)
