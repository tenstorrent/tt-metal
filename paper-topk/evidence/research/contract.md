# Gate 1 — The Actual Public Contract of `ttnn.topk` (as implemented), and the Differential Test Matrix a New Selector Must Pass

Date: 2026-08-16. Branch `nkapre/sorting`, Blackhole box. All file:line references are to the working tree at HEAD `22563f240c2`.
Method: read-only source audit of the composite op, device op, program factories, kernels, Blackhole LLK, ISA docs, and the shipping test/sweep suites. No device runs.

---

## 1. Routing gates — which engine actually answers a given `ttnn.topk` call

There are **three** engines behind the one public symbol, selected at two levels.

### 1.1 Composite level (host, before the device op): large-k Blackhole route

`should_route_to_topk_large_indices` — `ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:247-295`, invoked at `topk.cpp:485-501`. ALL of the following must hold, else fall through to the device op:

| Condition | Value | Line |
|---|---|---|
| `largest` | must be `true` | topk.cpp:258-260 |
| `stable` | must be `false` | topk.cpp:263-265 |
| no user `indices_tensor`, no preallocated outputs, no `sub_core_grids` | all absent | topk.cpp:267-269 |
| reduction dim | already last (`is_dim_last_idx`) | topk.cpp:271-273 |
| `k` | `64 < k <= 2048` (`large_k_route_min_k_exclusive=64` @ :236, `large_k_route_max_k=2048` @ :238) | topk.cpp:274-276 |
| dtype | `BFLOAT16` only | topk.cpp:277-279 |
| layout | `TILE` only | topk.cpp:280-282 |
| memory | not sharded | topk.cpp:283-285 |
| arch | `BLACKHOLE` only | topk.cpp:286-288 |
| width | `k_rounded_to_16 <= W <= 2^19` (=524,288; `large_k_route_max_width` @ :245) | topk.cpp:289-294 |

Routed pipeline (`run_topk_large_indices_route`, topk.cpp:301-325): TILE→ROW_MAJOR untilize → `ttnn::experimental::topk_large_indices_with_values(input_rm, k_rounded16)` → tilize values+indices → typecast indices to UINT16 iff tile-padded width ≤ 65535 (topk.cpp:315-318) → shared `post_topk_transform_tensor` slices `k_rounded16 → k`.

Underlying experimental op gates (`ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp`): `k>0 && k<=2048 && k%16==0` (:16,:25), row length `<= 2^30` (:17,:51-54), ROW_MAJOR (:32), BFLOAT16 (:33), interleaved (:34).

### 1.2 Device-op level: multi-core bitonic vs single-core insertion sort

`TopKDeviceOperation::select_program_factory` — `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp:59-115`. Multi-core (`TopKMultiCoreProgramFactory`) requires ALL of:

| Condition | Value | Line |
|---|---|---|
| padded width | `>= 8192` (`multi_core_min_width`, topk_constants.hpp:11) | topk_device_operation.cpp:66 |
| padded width | `< 65535` (strictly `< uint16_t max` — so UINT16 indices fit) | :70 |
| padded width | power of two (bitonic requirement) | :72 |
| `k` | `<= 64` | :75 |
| L1/core-grid feasibility | `verify_multi_core_cost(...)` — even split, contiguous rectangle, per-core CB cost < L1, `num_cores > 1`, split ≥ 64 (`topk_utils.cpp:86-162`) | :98-107 |

Everything else → `TopKSingleCoreProgramFactory` (:114). Combined with the pow2 requirement, the practical multi-core width envelope is **W ∈ {8192, 16384, 32768}** (65536 fails `< 65535`). The PR2 large-k route exists precisely because k > 64 always fell to single-core (158 ms at W=65536, k=512 — comment at topk.cpp:189-191).

### 1.3 Device-op validation (contract enforcement)

`validate_on_program_cache_miss` — topk_device_operation.cpp:117-272:

- 4D padded input (:125); inner padded dim ≥ 64 (:126-130, `min_dim_per_core`); combined batch dims multiple of 32 (:131-134); `k != 0` (:136).
- `stable=true` only on WORMHOLE_B0/BLACKHOLE (Quasar LLK static_asserts it off) (:141-148).
- Reduction must be on the last dim (:150-160); output not sharded (:163); input TILE layout (:166).
- Input dtype ∈ {BFLOAT16, BFLOAT8_B, FLOAT32} (:169-174).
- Optional `indices_tensor` ∈ {UINT16, UINT32}; **must be UINT32 when input is FLOAT32** (:177-187).
- Preallocated values dtype must equal input dtype; indices ∈ {UINT16, UINT32} (:190-207).
- L1 feasibility or `TT_FATAL("Not enough cores or cache size...")` (:220-271). fp32 compute buffers are full-width fp32 (no bf16 downcast), so single-core fp32 caps at **k ≤ 1728** (54 output tiles) in WH-size L1 (documented in test_topk.py:23-29; cost model at topk_utils.cpp:244-251).

### 1.4 Composite pre/post handling (shapes, k, special cases)

- `k` is rounded **up to a multiple of 32** for the device op (`get_nearest_supported_k_value`, topk.cpp:42-44, applied :470-472) and sliced back after (:119-128). The routed path rounds to a multiple of 16 instead (:290).
- `K > dim size` is a `TT_FATAL` (:368-372). PyTorch also rejects this — parity.
- `rank==0` scalar: k∈{0,1} returns a clone + index 0 (UINT16), matching torch (:396-435). Zero-volume input or `k==0`: returns 0-volume tensors (:437-458).
- Width `< 64` is host-padded to 64 with `-inf` (largest=True) / `+inf` (largest=False) (:506-519); **all** implicit tile padding is then filled with the same ±inf via `fill_implicit_tile_padding` (:522).
- Non-last `dim` handled by host transpose both ways (:475, :149-152); rank ≠ 4 by reshape/squeeze (:131-144).

### 1.5 Output index dtype

`compute_output_specs`, topk_device_operation.cpp:290-304: indices are **UINT16 iff (tile-padded width ≤ 65535 AND input dtype ≠ FLOAT32)**, else UINT32. fp32 forces UINT32 regardless of width (fp32 dest-acc loads indices as INT32, :291-295). The routed path replicates this exact boundary against the **padded** width (topk.cpp:315-318, verified in test_topk.py:466-469).

---

## 2. Value semantics — what the implementation actually does per input class

### 2.1 The comparator: SFPSWAP sign-magnitude total order

Both bitonic device paths and `topk_large_indices` compare exclusively with `SFPSWAP` (0 uses of SFPGT/SFPLE in the sort/topk headers — SORTING.md:591). Blackhole LLK: `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h` — every compare-exchange in `_bitonic_topk_phases_steps` (:648-960), `_bitonic_topk_merge` (:963-1015), `_bitonic_topk_rebuild` (:1018+) is `TTI_SFPSWAP(..., p_sfpswap::ALL_ROWS_MAX / ROWS_01_MAX / ROWS_02_MAX)`.

Per the ISA (`tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/SFPSWAP.md:3, :94-98`), SFPSWAP's min/max treats operands as **32-bit sign-magnitude integers**, giving the total order:

```
-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN
```

Additionally the swap predicate is `SignMagIsSmaller(C,D) || (C==D && C<0)` (SFPSWAP.md:49) — bit-equal negative pairs still swap, which is one mechanism behind the stable-network fragility (§2.7).

Consequences vs `torch.topk`:

| Class | ttnn.topk behavior (all engines) | torch.topk | Divergence? |
|---|---|---|---|
| `+NaN` (sign bit 0) | above `+Inf` — is top-1 for largest=True | all NaN above +Inf | agrees |
| `-NaN` (sign bit 1) | **below `-Inf`** — is bottom-1; top-1 for largest=**False** | above +Inf regardless of sign | **diverges** |
| NaN payload | preserved bit-for-bit (pipeline is bitwise; no arithmetic touches values) | payload unspecified | n/a — but ttnn is deterministic-bitwise |
| `+0` vs `-0` | **distinct**: `-0 < +0`; a boundary between them prefers `+0` for largest | numerically equal (tie) | order diverges; returned values still compare torch-equal |
| `±Inf` | ordered normally within the sign-magnitude order | same | agrees |
| Subnormals | compared **exactly** by the comparator (sign-magnitude integer compare has no flush) — but whether subnormals survive the unpack→DEST→pack datapath per dtype is **NOT verified by any test**; bf16 subnormal flushing in format conversion is a known hardware-family concern | exact | **UNVERIFIED — must be pinned** |

These divergences are pre-acknowledged as the family contract in `tt_metal/tt-llk/tests/docs/THRESHOLD_SELECT_DESIGN.md:350-372` ("Documented divergences from torch, inherited from the hardware order") and confirmed independently by the packer-order finding SORTING.md §A2 (:493-498).

### 2.2 Duplicate values at the k-th boundary (tie allocation)

- `stable=false` (default): tie order/membership among equal boundary values is **deterministic but unspecified**. The returned index set contains all strict winners plus *some* subset of boundary-equals. The stock bitonic path's tie choice falls out of the network's positional dataflow; the routed path's falls out of chunk order (topk.cpp:260-262: "tie order is deterministic but unspecified"). Tests deliberately validate via gather + uniqueness, never index equality (test_topk.py:84-96, :445-449).
- `stable=true`: **advertised as EXPERIMENTAL best-effort, not correct** — `topk_nanobind.cpp:49`: the stable bitonic network "can still return incorrect indices for tied values" (open issue tenstorrent/tt-metal#33492); every stable case in the LLK test suite is skipped. Default `stable=false` (`topk_nanobind.cpp:105`). Validation rejects `stable=true` off WH/BH (topk_device_operation.cpp:141-148). Routing: `stable=true` disqualifies the large-k route (topk.cpp:263).

### 2.3 `sorted` flag: accepted, **ignored** — output is always sorted

- Single-core factory does not even pass `sorted` to the compute kernel (named CT args at `topk_single_core_program_factory.cpp:364-365` carry only `largest` and `stable_sort`).
- Multi-core factory passes it as CT arg 13 (`topk_multi_core_program_factory.cpp:455,:496`) but both kernels only read it into an unused `constexpr` (`topk_local.cpp:108`, `topk_final.cpp:60` — zero further uses).
- The routed path always emits descending order ("satisfies both sorted=true and sorted=false callers", topk.cpp:212-213).

**De facto contract: values are always fully sorted (descending for largest=True, ascending for largest=False), regardless of `sorted`.** A new selector may exploit `sorted=false` for speed, but nothing in the ecosystem has ever observed unsorted output — treat sorted-always as the compatibility bar unless a deliberate contract change is accepted.

### 2.4 `largest=false`

Fully supported on stock paths: padding flips to `+inf` (topk.cpp:510), merge networks flip min/max (`topk_merge<true,...>` at topk_common_funcs.hpp:177; LLK `top_min` template arg, ckernel_sfpu_topk.h:963; SFPSWAP direction reversal via `TTI_SFPCONFIG(0x104,...)` at :351-356). **Disqualifies the large-k route** (topk.cpp:258) — `largest=false, k>64` rides the single-core cliff.

### 2.5 Non-pow2 W / padding, and the padding-index leak

- Indices are **generated on device** covering the full padded width: `reader_create_index_tensor.cpp:57-59` emits `generate_index_tile` for every tile `w` in `Wt` — so tile-padding columns `[W, padded_W)` carry real, in-range-of-padded-width index values, paired with ±inf padding values.
- Therefore on the stock path, if a row's true top-k reaches into `-inf` territory (fewer than k finite elements with largest=True), **returned indices can point into the padding, i.e. be ≥ logical W**. Acknowledged in-source: "the stock path is itself loose there (it can return indices pointing into its own -inf padding beyond the logical width)" (topk.cpp:215-217).
- The routed path replaces this with an explicit sentinel: `-inf` value lanes carry index **`0xFFFFFFFF`** (`0xFFFF` after UINT16 typecast) with bit-exact `-inf` values (topk.cpp:211-217; sentinel emitted by the XL tail path, topk_large_indices_device_operation.cpp:63-64; pinned by test_topk.py:524-553).
- So the two engines behind the same public symbol **disagree on -inf-lane index semantics today**. A new selector needs an explicit decision here (sentinel is the cleaner contract; padding leak is the incumbent single-core behavior).
- Non-pow2 W: single-core handles any W ≥ 1 (host pads to ≥ 64); multi-core requires pow2; routed path handles arbitrary W in `[k_16, 2^19]` (tested at W=100000, test_topk.py:502-509).

### 2.6 Dtypes

- BFLOAT8_B/BFLOAT4_B inputs are **upcast to bf16 for the sort** (intermediate transposed CBs bf16 to avoid shared-exponent loss — topk_utils.cpp:78-83, :244-251); output values re-quantize to bfp8. The bfp8+inf shared-exponent regression has a dedicated test (test_topk.py:260-340).
- FLOAT32 sorts at full 32-bit precision (exact), forces UINT32 indices, and caps single-core k (§1.3).
- Non-float inputs (uint32/int32) are rejected (topk_device_operation.cpp:169-174; test_topk.py:350-359). Note sign-magnitude ≠ two's-complement, so integer support would need a premap (THRESHOLD_SELECT_DESIGN.md §3).

### 2.7 UInt16-values-in-fp32-DEST quirk

When values are UInt16 with 32-bit DEST (sort ops), high garbage bits must be stripped before compare-swap and re-packed via SFPSTORE mode 9 (`TOPK_UINT16_FP32_DEST`, ckernel_sfpu_topk.h:133-207). Not reachable through `ttnn.topk` (float-only inputs) but is part of the LLK surface a new selector shares.

---

## 3. What existing tests actually pin vs leave unspecified

### 3.1 Pinned (has failing-test coverage today)

| Semantic | Where |
|---|---|
| Exact values (assert_equal, sorted order) on Gaussian inputs, bf16/fp32, both largest, both sorted, dims 1/2/3, 20 shapes incl. non-pow2 W (18992, 10000, 64128, 20, 22...) | tests/ttnn/unit_tests/operations/reduce/test_topk.py:99-158 (`assert_equal` :82) |
| Index validity via gather + cosine ≥ 0.99 (tolerates tie permutation by construction) | test_topk.py:84-96 |
| Vocab-size widths (151936, 128256) incl. UINT32-index regime, with/without user indices tensor | test_topk.py:212-257 |
| bfp8 + all-+inf rows (shared-exponent corruption guard), single- and multi-core | test_topk.py:260-340 |
| Rejection: uint32/int32 input; preallocated dtype mismatches; fp32+uint16 indices tensor | test_topk.py:343-405 |
| Multi-core local-writer WAR value correctness (W=8192, k=32, both largest) | test_topk.py:408-436 |
| Routed path: exact sorted values; unique, in-range, gather-consistent indices; index-dtype boundary (padded W vs 65535); k∈{96..2048} × W∈{8192..131072}; non-pow2 W=100000; single-row column-parallel | test_topk.py:455-518 |
| Routed -inf lanes: bit-exact -inf values + `0xFFFFFFFF` sentinel indices | test_topk.py:521-553 |
| Routing-fallback boundary cells: k=2049, largest=False, stable=True stay on stock path and stay correct | test_topk.py:556-575 |
| Sweeps: small-k (k=32 only) across dtypes/layouts/memcfgs/dims incl. multi-dim + RM; large_k suite = exactly the routed predicate, 168 vectors | tests/sweep_framework/sweeps/reduction/topk/topk.py:28-80 |
| stable=true returns correct *values* at one shape (fallback test) | test_topk.py:569-575 |

### 3.2 NOT pinned by any test (unspecified in practice)

1. **NaN — anything.** No test injects NaN. The `-NaN` divergence from torch, `+NaN`-as-top-1, and payload/sign preservation are all untested at the ttnn layer.
2. **±0 at the boundary.** No mixed-sign-zero input anywhere; the `-0 < +0` ordering and boundary preference are untested.
3. **Subnormals.** Zero coverage; survival through the unpack/pack datapath per dtype is unknown.
4. **Tie allocation at the k-th boundary.** Cosine/gather validation *tolerates* ties but no test constructs a duplicate mass straddling k and checks the winner-set property (`Cgt` strict winners all present + exactly `k−Cgt` equals).
5. **`sorted=false`** — swept as a parameter but the reference always uses `sorted=True` and output is compared sorted; the flag's no-op nature is unobserved, not asserted.
6. **`stable=true` index correctness** — explicitly disclaimed (nanobind :49); LLK stable tests skipped; only the values side is spot-checked.
7. **Stock-path all/mostly `-inf` rows** — the padding-index leak (indices ≥ W) has no test; only the routed sentinel variant is tested.
8. **`largest=false` with `-NaN`/`+Inf` extremes**, k=1 and k=W degenerate cells, k exactly at 64/65 (device-op multicore gate boundary) with pow2 W.
9. **fp32 NaN/±0/subnormal** anything (fp32 coverage is Gaussian-only).

---

## 4. Gate-1 differential test matrix specification

Reference oracle: `torch.topk` on float64 shadow + explicit expected-divergence lists (below), run per cell against the engine the routing actually selects (record `select_program_factory` outcome + routed-or-not per cell — RADIX §5.2 gate 1 requires pinning the selected factory).

### 4.1 Engine-selecting parameter axes (columns)

Choose cells to hit each engine and each gate boundary:

| Axis | Cells | Rationale |
|---|---|---|
| Engine cell A: single-core | W=4096 (pow2 < 8192), W=10000 (non-pow2), W=65536 (≥ u16 max), k=65 (>64) | each disqualifier independently |
| Engine cell B: multi-core | W ∈ {8192, 32768} pow2, k ∈ {1→32, 64}, bf16/fp32/bfp8 | all four §1.2 gates satisfied |
| Engine cell C: routed | bf16, W ∈ {8192, 100000, 2^19}, k ∈ {96, 2048} | §1.1 predicate |
| Boundary flips | k=64 vs 65 (multicore→single/route); k=2048 vs 2049; W=2^19 vs 2^19+32; padded-W 65504 vs 65568 (index dtype flip); W=63 vs 64 (host-pad path) | one-sided gate tests |
| dtype | bf16, bfp8_b, fp32 | fp32 forces u32 indices; bfp8 upcasts |
| largest | True / False | route eligibility flips; padding sign flips |
| sorted | True / False | assert byte-identical outputs (documents the no-op) |
| stable | False (primary) / True (values-only assertions, known-broken indices) | |
| dim | -1 and 1 (transpose path) | |
| k alignment | k ≡ 0, 1, 31 mod 32; k ≡ 0, 15 mod 16 (route rounding) | slice-back correctness |
| outputs | fresh / preallocated / user indices_tensor | preallocated+indices disqualify the route |

### 4.2 Input classes (rows) × coverage status

| # | Input class | What it pins | Current coverage |
|---|---|---|---|
| I1 | Gaussian random (seeded) | baseline exactness | ✅ test_topk.py:99-158, sweeps |
| I2 | `+NaN` (single, several, payloads 0x7FC0/0x7FFF-style bf16) | +NaN top-1 for largest=True; payload preservation | ❌ none |
| I3 | `-NaN` | **expected divergence**: ttnn bottom-1 (torch: top). Assert ttnn-order, document torch delta | ❌ none |
| I4 | mixed `+0`/`-0` mass straddling k | boundary prefers +0; values torch-equal; ±0 counted as distinct by hw order | ❌ none |
| I5 | `±Inf` mixed with finite | ordering; +Inf ties | ⚠️ bfp8 +inf only (:260-340) |
| I6 | subnormal-only and subnormal-boundary rows (bf16 & fp32) | do subnormals survive/order exactly, per dtype? UNKNOWN — outcome defines contract | ❌ none |
| I7 | duplicate mass at boundary (m copies of v straddling k; also all-equal row) | winner-set property: all `>v` present, exactly `k−Cgt` equals, unique in-range indices | ⚠️ tolerated, never constructed |
| I8 | fewer than k finite values (tail `-inf`), largest=True | stock: padding-index leak (idx ≥ W allowed?); routed: sentinel `0xFFFFFFFF` — **assert per-engine, document the split contract** | ⚠️ routed only (:521-553) |
| I9 | sorted / reverse-sorted / positional adversarial (winners at face boundaries: cols 15/16/31/32, rows 15/16; winners only in last tile) | tile/face-layout-correlated bugs | ❌ none |
| I10 | all-negative rows; alternating sign | sign-magnitude edge (matters doubly for any threshold/radix selector, RADIX §5.2 gate 2) | ❌ none |
| I11 | non-mult-32 W with winners inside `[W-31, W)` (last partial tile) | logical-tail vs padding discrimination | ⚠️ shapes exist (18992, 20, 22) with random data; not adversarial |
| I12 | k=1, k=W, k=W-1; W=64 minimum | degenerate cells | ⚠️ k small yes; k=W no |
| I13 | repeated-launch determinism (same input × 20 runs, program-cache hit path) | bit-identical outputs incl. tie choice ("deterministic but unspecified" ⇒ at least deterministic) | ❌ none |

### 4.3 Assertion tiers (per cell)

1. **Values: bit-exact** vs torch under the sign-magnitude order model (build a reference comparator: torch.topk on keys mapped through the IEEE-bits monotone map; for NaN-free, zero-uniform inputs this equals plain torch). Never PCC (branch discipline; THRESHOLD_SELECT_DESIGN.md §2).
2. **Indices: set-valid** — unique, in-range (`< W` OR documented sentinel/padding exception per engine), `gather(input, idx) == values` bit-for-bit, all strict winners present, equal-count = `k − Cgt`.
3. **Metadata:** index dtype matches §1.5 boundary; output shape/logical-shape; selected factory recorded.
4. **Divergence ledger:** each cell carries `torch_exact | torch_equal_valueset | documented_divergence(name)` — I3 (−NaN), I4 (±0 order), I8 (sentinel/padding) are the only permitted `documented_divergence` entries.

### 4.4 Priority order for implementation

The matrix is ~13 input classes × ~10 engine/param cells, but Gate 1 does not need the full cross-product. Minimum decisive set (~60 cells): {I2,I3,I4,I6,I7,I8,I10,I13} × {single-core W=10000, multi-core W=8192/k=32, routed W=100000/k=512} × largest∈{T,F} × bf16, plus fp32 for {I2,I3,I4,I6}, plus the six boundary flips in §4.1. Everything already ✅ above stays in CI as-is.

---

## 5. Conclusions relevant to the campaign

1. **The public contract is looser than torch and looser than it looks**: sorted is a no-op (always sorted), stable is officially best-effort-broken (#33492), tie sets are unspecified, and the two engines behind the symbol already disagree on -inf-lane indices (padding leak vs 0xFFFFFFFF sentinel). A new selector therefore does NOT need torch-exact NaN/±0/tie semantics — it needs **sign-magnitude value exactness + valid tie sets + per-engine-documented -inf behavior**, which is exactly the contract THRESHOLD_SELECT_DESIGN.md §2 already wrote down. The incumbent's own contract is the bar.
2. **The differential tests that gate everything are the unpinned rows I2-I4, I6-I8, I10, I13** — cheap to write (host-constructed tensors, existing gather/uniqueness assertion helpers in test_topk.py), no new infra.
3. **Subnormal datapath survival (I6) is the only genuine unknown** that could surprise a bitwise selector: the comparator is exact, but no test proves bf16 subnormals survive unpack→DEST→pack unflushed on any engine. Pin it first; the result becomes contract text either way.
4. Routing-gate boundary cells (§4.1) double as regression armor for any future routing change (a new selector will eventually claim a region of this same predicate space, per RADIX §5.2 gate 8).
