# Compact intake and output for the SFPU recurrence

Source-only candidate, 2026-11-12: `candidates/gdn_sfpu_compact_inline.py`. No device execution by this helper. Python syntax checks passed. A host-only mapping check covered B1/B3/B32, verified normalized Q head/column addresses, compact output addresses, disjoint output writes, and the FP32 bit encodings of epsilon and Q scale.

The stage owner reported that the preceding inline SFPU recurrence compiled and passed initial output/state, trace/eager, and changed-input checks. Its full decoder latency was3.7515ms versus a3.103ms native candidate that fails the long numerical gate. Those results are not measurements of this new candidate.

## Interface

```python
recurrent_sfpu_compact(q, k, v, g, beta, state,
                       normalized=False, output=None, cores=None)
```

Raw mode consumes BF16 q/k `[B,S,2048]` and V `[B,S,6144]`, where logical S is1 or32 and only row0 is decoded. G and beta remain `[B,1,48]`, each BF16 or FP32; G is the log decay, not exp(G). State is the existing FP32 `[B*48,128,128]` view. All tensors are TILE interleaved DRAM.

The exact-math control sets `normalized=True` and instead supplies FP32 q/k `[B,16,128]`; Q is already normalized and scaled, K normalized. This compact layout can come from a metadata reshape of the preceding `_l2` result `[B,1,16,128]`, without GQA repetition or per-head column expansion. V and gates retain the raw-mode interface. G exponentiation still moves inside the kernel.

Both return **compact** FP32 `[1,B*48,128]`. Feed this directly into the compact GDN head norm. Expanding it back to `[B,48,1,128]` would reintroduce the padded tensor and movement this candidate removes. Final state remains in the caller's original allocation.

## Reader mapping and movement

For flat head h, user=`h//48`, value head=`h%48`, key head=`(h%48)//3`. The reader fetches only the two16-element face-row spans from each of the four tiles in a head vector. It widens BF16 to FP32 with integer shift-left16 in local memory, preserving every input bit. There is no RISC floating-point arithmetic and no host tensor conversion.

Raw Q/K page stride is64 tiles/user and V stride192 tiles/user. Normalized compact Q/K use four tiles/user, selecting the key-head row in those tiles. Gates use two tiles/user and select the correct16-element face row and scalar. All head payload reads, including the16 state tiles, are issued before a shared read barrier. Input padding is zeroed once per core and retained across the one-head-deep reader CB rings.

At B32, the preceding expanded Q/K/V, scalar gates, and output accounted for108MiB of tensor traffic before counting preprocessing. This candidate's useful vector/gate/output payload is below3MiB. The unchanged FP32 state read/write is192MiB. These are payload estimates, not measured bandwidth or latency.

## Compute and output

Raw Q/K normalization uses SFPU square, four-tile addition fold, accurate SFPU row reduction, epsilon add, rsqrt, normalization multiply, and a **separate** Q scale multiply. The FP32 constants are epsilon=`0x358637bd` and scale=`0x3db504f3`. A local FP32 transpose converts normalized rows into columns. The normalized control performs only that transpose. Exp(G) uses `exp_tile_init<false>`/`exp_tile<false>`.

The recurrent state arithmetic remains the previous FP32 SFPU sequence. The writer emits each head as a row of the compact tensor with eight disjoint64-byte writes. Different cores may own rows in the same tile, but never overlapping64-byte regions. The final head zeroes unused padded output rows for odd batches. State and output synchronization retain the earlier rule: writer waits for output before popping updated state, so readout has completed.

CB storage is91 FP32 tiles/core,364KiB, before ordinary kernel/runtime overhead. The generic-op runtime descriptor is rebuilt each call, preserving the previously documented safe cache-hit address update behavior.

The initial source inherited a missing unpack precision setting from the first inline prototype. It now uses `_fp32_compute_config()` with `UnpackToDestFp32` on every used FP32 CB0..18. FP32 storage and FP32 destination accumulation alone otherwise select TF32 during unpack, even when every arithmetic operation is SFPU. See the FP32 unpack correction in `AUTOFIX_gdn_sfpu_inline.md` for exact source evidence. Corrected syntax checks passed; all device gates remain outstanding for this version.

## Validation order

1. Compare normalized control against the original inline recurrence with the same normalized input values, state, and gates. Check every head and both outputs/state; this isolates mapping, compact writes, transpose, and internal exp.
2. Compare raw mode to the control and direct own-input CPU recurrence, including128 repeated steps. Report any normalization delta rather than assuming the matching algebra proves bit identity.
3. Test B1, odd batches, B32, distinct per-head inputs, trace replay, state reset/allocation identity, allocation tracking, and watcher separately.
4. Measure the full decoder with compact output flowing directly to norm; include all remaining preprocessing. The prior3.103ms native result is a performance comparator, not an acceptable numerical fallback.

## BF16 read alignment correction

The first compact device probe (`compact14_probe.json`) found normalized-control state relative RMS error0.6965 for random input and0.7053 with zero initial state, while the zero-K/V case was accurate to8.8e-9. The corrected original SFPU recurrence was accurate to1.1e-8 state relative RMS. This isolates a compact intake defect rather than the core recurrence arithmetic.

Source inspection identified illegal BF16 reads. Blackhole `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379` sets DRAM read alignment to64 bytes. `tt_metal/hw/inc/internal/debug/sanitize.h:746` checks that source and destination have the same address modulo that alignment. Both face-row sources are64-byte aligned, but the initial reader placed alternate BF16 face rows at destination offsets32 modulo64. The missing alternate half-face hypothesis matches the observed approximately sqrt(0.5) error, although corrected hardware verification is still required.

The candidate now assigns each16-element face row a64-byte scratch slot, reads64 bytes, and extracts only its first16 elements. BF16 therefore overfetches the following row; FP32 reads stay unchanged. BF16 extraction uses index`(i/16)*32+i%16`; FP32 uses index`i`. Existing512-byte per-vector scratch allocations suffice. Gate reads already used64-byte-aligned destinations and require no change. Host-only checks verified all source/destination modulo64 matches for raw BF16 row0 and normalized FP32 rows0..15 and all128 extraction indices. Python AST validation passed; no device run was performed by this helper.

The raw-mode rsqrt API was also compared with external TTNN rsqrt. `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp:744` emits the same `rsqrt_tile_init()` and `rsqrt_tile<RsqrtMode::Default>()` used by the candidate's default template. `unary_program_factory.cpp:532` uses `math_approx_mode=false`, matching this candidate. `api/compute/eltwise_unary/rsqrt.h` states that `RsqrtMode::Fast` only skips negative-input handling and does not change approximation precision for nonnegative inputs. Both paths use FP32 destination mode and the same Blackhole SFPU implementation. There is no source-supported rsqrt precision mismatch to fix. Raw-mode comparison still needs to assess reduction order and, if enabled externally, the distinct HF BF16 normalization policy.
