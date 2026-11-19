# GDN existing-op performance hypotheses

Source-only AutoFix helper, 2026-11-12. No hardware was touched, and no main decoder or shared TTNN source was edited. `candidates/gdn_existing_ops.py` passed `python -m py_compile`; all performance and correctness claims below remain hypotheses pending the stage owner's measurements.

## First candidate: native fused outer-product accumulation

The direct recurrence currently materializes `k_column * delta_row`, a 96 MiB FP32 tensor at B32, then adds it to decayed state. Replace those operations with:

```python
ttnn.addcmul(decayed, k_column, delta_row, value=1.0, output_tensor=state)
```

The exact shapes are `[B,48,128,128]`, `[B,48,128,1]`, and `[B,48,1,128]`, all FP32 TILE. This is **native**, not the ternary composite fallback: `ternary_op_utils.cpp:768-834` classifies this as `ROW_COL_BCAST`, and lines272-285 explicitly map ADDCMUL/TTT/ROW_COL_BCAST to its mixed-broadcast compute/dataflow kernels. `ternary.cpp:331-380` accepts FP32 and only falls back for an invalid broadcast or a block-float subtile broadcast. The Python binding names the output argument `output_tensor` (`ternary_nanobind.cpp:214-225`).

All three inputs are separate from the destination state allocation. Earlier operations finish reading old state before this ordered write, and the output contraction reads state afterward. Thus this change preserves stable state allocation without input/output tile aliasing. It can change rounding through the fused SFPU arithmetic; run the same long-sequence gate rather than assuming algebra proves equivalence. The expected movement saving is one full temporary write and read, about192 MiB per B32 step.

## Second candidate: batched matrix-vector reductions

The two `sum(state * column_vector, dim=-2)` contractions materialize another two96 MiB FP32 intermediates. Replace each with FP32 batched matmul, `row_vector @ state`. This is a single1536-batch operation and does not have the recurrent kernel's one-core-per-head constraint.

Start with:

```python
ttnn.MatmulMultiCoreReuseProgramConfig(
    compute_with_storage_grid_size=(11, 10),
    in0_block_w=4, out_subblock_h=1, out_subblock_w=4,
    per_core_M=1, per_core_N=4,
)
```

Use input row tensors `[B,48,1,128]`, state `[B,48,128,128]`, FP32 output and DRAM memory, and HiFi4/fp32 destination accumulation/no approximation. The factory derives total batch count from outer dimensions and distributes batched output blocks across the core grid (`matmul_multicore_reuse_optimized_program_factory.cpp:97-124` and its later work split). The FP32 subblock limit is four tiles (`:91-95`), satisfied by1x4. K=4 tiles and N=4 tiles divide the proposed blocks. Tune `per_core_N=2/out_subblock_w=2` and `1/1` if splitting the value dimension gives better bandwidth.

This changes reduction order and uses matrix-engine input arithmetic rather than the functional SFPU elementwise path. FP32 tensor storage is not a promise of a full IEEE FP32 multiply. Report actual PCC. First test addcmul alone; then batched matmul alone; then their combination. The helper exposes both switches. The matmul computes padded32-row tiles, so it is not automatically faster despite eliminating DRAM intermediates.

Do not switch these tests to tiny tiles at HiFi4: Blackhole LLK `llk_math_matmul.h:738` explicitly documents a tiny-tile HiFi4 hang, and matmul validation guards the unsupported combination. A separately legalized lower-fidelity tiny-tile candidate would need its own numerical evidence; it is not the precision control here.

## Fused chunk geometry candidates

For S=1, NC=1: every head has only one producer item. The pool cannot have more producers than heads (`chunk_gdn_device_operation.cpp:198-204`), and cannot amortize extra producer scheduling over later chunks. Compare the current pool path with explicit per-head producers:

```python
ttnn.ChunkGdnFusedProgramConfig(
    num_producers=1, num_receivers=1, producer_pool=False,
    row_local=True, handoff_depth=1, unicast=True, posted=True,
    wave_heads=55,
)
```

Call the existing private `chunk_gdn_fused` exactly as the optimized path already does, with caller-supplied output and `final_state_out=state_view`, actual `num_heads` and `head_offset`. The last wave has51 heads at B32; per-head NP=1 remains legal without changing a pool-size pin. Pinning pool `num_producers=55` would be illegal for that51-head tail at NC=1.

The eleven-by-ten grid supports these source-proven limits (`fused_row_local_feasible`, `fused_pool_feasible`, and validation in `chunk_gdn_device_operation.cpp:190-259,467-505`):

| Receivers/head | Flat maximum heads | Row-local maximum heads | B32 launch count at flat maximum |
|---|---:|---:|---:|
|1|55|55|28|
|2|36|35|43|
|4|22|22|70|

Compare row-local/flat, posted/unposted unicast, and receiver splits. More receivers spread the scan's128-wide state across more cores but require more launches; that is a measured tradeoff. `handoff_depth=1` is sufficient for one chunk. Existing source provides a4us receiver kickoff hold (`chunk_gdn_fused_program_factory.cpp:77-82`) which these configs cannot remove; modifying shared kernels is outside stage scope. The helper enumerates only legal NC=1 geometries, including the36-head NV2 flat-only restriction.

## Existing composites that do not provide a direct escape

- Phased prep can distribute BH*NC work over the whole grid, but phased scan still requires BH<=110 (`chunk_gdn_phased_program_factory.cpp:149-151`), has no head-offset/supplied-output binding, and would need slab movement for B32. It is not an all1536-head scan replacement.
- Mono requires head-major inputs, no in-kernel Q/K normalization, BH<=110, head_offset=0, and disallows supplied outputs (`chunk_gdn_device_operation.cpp:145-181`). It adds layout/state movement compared with the current direct-offset fused path.
- `gated_delta_attn_seq` requires C=K=V=128 and eight explicit FP32 intermediates (`gated_delta_attn_device_operation.cpp:65-112`); it does not expose a single-token recurrence or persistent supplied output. It is not an equivalent cheap S=1 composite.
- `deepseek_moe_fast_reduce_nc_fused` fuses score multiply and sum, but supports only BF16/BFP8 activation and BF16 scores, outer-axis reduction, and expert routing metadata (`deepseek_moe_fast_reduce_nc_fused_device_operation.cpp:58-113`). Using it would require state precision and layout changes plus routing tensors, not an FP32 fused reduction replacement.
- Chunk GDN lower fidelity is explicitly rejected by `gdn_compute_config.hpp:26-39`: every legal path must use HiFi4, FP32 destination, and no approximation. This is a source contract, not a first API error.

## Gate order

For each candidate record short real-weight output/state PCC, long random and prompt-embedding sequence results, and traced full-decoder latency. Preserve failures in the table. For any winner, run repeated trace with input/state reset, allocation tracking, mixed/odd batch, and watcher separately. No candidate is selected merely because it removes an allocation or reduces op count.
