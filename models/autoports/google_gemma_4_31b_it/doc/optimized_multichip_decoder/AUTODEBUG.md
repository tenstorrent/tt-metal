# Stage05 TP4 prefill RMSNorm: valid geometry, excessive live L1 footprint

## Finding

The reported failures are host-side static-dataflow-buffer placement rejections, not evidence of a normalization arithmetic error or a device hang. The candidate's 7×8 block-sharded configuration is dimensionally legal, but its FP32 scratch requires **925,696 bytes per core** before accounting for tensor allocations and runtime reservations. `inplace=True` removes a separate output tensor allocation; it does **not** reduce these scratch buffers. The final post-feedforward norm reaches this footprint while other layer tensors remain live.

The smallest precision-preserving configuration experiment is **8×8, shard 128×672, block_h=4, block_w=21, subblock_w=3 (or 1)**. It retains all 5,376 channels, all 1,024 rows, HiFi4 and FP32 intermediates. Static scratch falls by **100,352 bytes (98 KiB) per core**, and the inplace shard falls by another 24,576 bytes. This is a source-supported legal candidate, not a hardware-validated fix or performance claim.

**Separate tail bug:** the current height-grid planner creates entirely empty trailing shards for some non-1024 row counts; shared sharding validation rejects them. Correct the grid count as shown below when generalizing the lower-memory configuration.

## Scope and provenance

Inspection only: no devices, resets, model execution, implementation edits, remote publication, or removed Gemma sources. Read `bringup/run_contract.json` and `bringup/agentic_evals/inputs.json`. Inspected only this fresh autoport and generic TTNN/Metal/tests. One independent xhigh investigator audited factory sizing and kernel geometry.

Checkout HEAD: `bdbdc9e17c3f5dbe0ac12ac1e44bc408c81f8da0`, with pre-existing uncommitted Stage05 work. At initial inspection, `tests/optimized_multichip_candidate.py` SHA256 was `df3ae1919f39ca48c6ddcb23845930f9459334dbe5018c21eb101d2ba30a2cf2`, exactly matching the norm3 manifest. The workspace changed concurrently later in the investigation; line references describe the inspected snapshot and may shift. Norm2 records a different candidate hash; its complete source snapshot is not supplied. Do not infer its exact option delta from the experiment name. Manifests do not establish a full C++ binary/source hash correspondence; the static derivation below uses the prepared tree and matches the reported endpoint with a fixed runtime/base allowance.

Paths below are relative to the repository. `A` denotes `models/autoports/google_gemma_4_31b_it`; `L` denotes `ttnn/cpp/ttnn/operations/normalization/layernorm`.

## Observations versus interpretation

| Direct observation | What it establishes |
|---|---|
| `A/doc/optimized_multichip_decoder/norm2_layer0_batch1.log:53` reports static end 1,037,440 and lowest occupied L1 address 1,020,416. Stack at lines 62–71 is initial `prefill_forward`, harness line 153, post-feedforward norm. | An overlap of 17,024 bytes at that placement check. |
| `norm3_layer0_batch1.log:55` reports the same static end and lowest occupied address 993,024. Stack at lines 64–73 is `warm_pref`, harness line 195, the same post-feedforward norm. | An overlap of 44,416 bytes in a later, different live-allocation state. |
| Norm3 reaches the warm prefill after initial prefill, decode, host reads and an explicit synchronization (`A/tests/run_multichip_decoder.py:153–195`). | This is stronger evidence than a progress print: the earlier execution reached completion boundaries. It does not prove every warm call fits or that all candidate accuracy checks passed. |
| `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3072–3099` compares static region end against `lowest_occupied_compute_l1_address`. The stack goes through `compile_and_allocate` / workload enqueue. | The failing program is rejected before its kernel executes. The printed lowest address is not a named tensor identity or proof of corruption. |

Interpretation: the same scratch footprint crosses two different live-allocation floors. This explains both failures without a speculative queue or arithmetic bug. Norm3's later failure is consistent with additional retained harness outputs and different allocation history, but the logs do not identify the exact allocation at 993,024. The address change is **not** a controlled measurement of the effect of `inplace`.

## Lowered operation and exact capacity ledger

1. `A/tests/optimized_multichip_candidate.py:54–70` admits named layernorms at width 5,376 and total padded rows ≤1,024. At 1,024 rows: `gy=8`, height=128, 7 columns of 768, row-major BLOCK sharding. Conversion allocates the sharded tensor while the original input remains live.
2. `A/tt/multichip_decoder.py:80–82` supplies Blackhole HiFi4, exact math, FP32 destination accumulation, packer L1 accumulation. Norm weights are BF16 tiled DRAM tensors (`:120–162`). No bias or fused residual is passed by the candidate. The replicated residual path presents a full 5,376-channel norm on each TP device; TP4 does not divide this norm width by four.
3. `normalization/rmsnorm/rmsnorm.cpp:22–86` forwards the explicit program and compute configurations to `prim::layer_norm` with RMSNORM. There is no operand swap or hidden dtype conversion on this path. `L/device/layernorm_device_operation.cpp:19–24` selects the sharded factory from the input. Sharded validation checks tile/block/grid consistency and divisibility (`:368–460`), not available live scratch capacity.
4. `L/device/sharded_layernorm_factory_helpers.cpp:109–136` selects Float32 intermediates whenever FP32 accumulation is enabled. BF16 input/output/gamma tiles are 2,048 bytes; FP32 intermediate tiles are 4,096 bytes. The default non-Welford, non-distributed factory selects `layernorm_sharded.cpp`, multicast sender/receiver readers and tiled-gamma `writer_unary_sharded_ln.cpp` (`:477–505`). This is an intra-device reduction, not distributed pre/post-all-gather RMSNorm.
5. The grid has 7 width blocks, 4 tile rows per shard, and 4 all-to-all workers, one tile row per worker; remaining 3 cores in each row are non-workers. `GridParams` / `WorkerDistribution` (`:169–234`) choose single-stage reduction. There are 96 input tiles per shard.
6. Output creation returns the input tensor for inplace (`L/device/layernorm_device_operation.cpp:550–560`). IN0 and OUT borrow tensor storage (`helpers.cpp:631–635,730–739`); neither adds a separate static scratch allocation. Static X and XMM remain allocated by the factory.

Factory sizing: `helpers.cpp:510–578`; declarations: `:626–719`.

| Allocation per core | 7×8 / 128×768 | 8×8 / 128×672 |
|---|---:|---:|
| X: h×w FP32 tiles | 393,216 | 344,064 |
| XMM: h×w FP32 tiles | 393,216 | 344,064 |
| EX_PARTIAL2 + EX2 + EX_GLOBAL: 3×h FP32 tiles | 49,152 | 49,152 |
| EX_EXTERNAL2: width-block count FP32 tiles | 28,672 | 32,768 |
| EX2PE: ceil(h / width-block count) FP32 tiles | 4,096 | 4,096 |
| GAMMA: w BF16 tiles | 49,152 | 43,008 |
| SCALER + EPS BF16, SCALER_GLOBAL FP32 | 8,192 | 8,192 |
| **Static scratch total** | **925,696** | **825,344** |
| **One borrowed inplace input/output shard** | **196,608** | **172,032** |
| **Distinct scratch + shard bytes** | **1,122,304** | **997,376** |

No beta, mean-of-x chain, width mask, Welford LUT, or distributed-statistics buffers are required in this case. XMM is allocated even though this particular no-residual, gamma-only RMSNorm path does not use its backing storage: `dfb_xmm_id` aliases IN0, `dfb_im_id` is X when gamma exists, and `dfb_outgamma_id` is OUT when beta is absent (`layernorm_sharded.cpp:98–105,196–200`). X is reused after the variance pass. This is a concrete generic-factory over-allocation opportunity (393,216 bytes here), but removing it requires auditing named buffer bindings and every other variant; it is a broader intervention than the caller geometry experiment.

The observed endpoint minus computed scratch is 111,744 bytes. Treat this as an inferred runtime/base allowance, not a separately measured free-space budget. With the same allowance, the 8-column endpoint is **937,088**, below even the lower reported floor by **55,936 bytes**, before crediting the smaller tensor shard. Geometry changes can change actual placement; verify actual endpoint and floor on the complete layer, including warm execution.

Time-ordered budget: existing residual/intermediates/harness allocations → sharded input conversion → output alias (or a new tensor for out-of-place) → static scratch placement/validation. The Python `rows <= 1024` condition is a size heuristic, not an admission calculation for this combined footprint. There is no evidence that the final validator is wrong or that a sharded capacity selector used a stale free-space estimate; this path unconditionally selects the sharded factory.

## Minimal discriminating changes (not applied)

### First experiment: retain precision, change only the width partition

In `_norm`, change the coupled geometry values together:

```python
mt = rows // 32
block_h = math.ceil(mt / 8)
gy = math.ceil(mt / block_h)
height = 32 * block_h
grid = ttnn.CoreGrid(x=8, y=gy)
memory = ttnn.create_sharded_memory_config(
    (height, 672), grid,
    ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)
# same weight, epsilon, self.kernel, inplace=True, and output conversion
program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
    compute_with_storage_grid_size=(8, gy),
    subblock_w=3, block_h=height // 32, block_w=21, inplace=True,
)
```

Keep the existing row-count guard, but correct height-grid selection for tails as described below. An 8-column grid must be available in the selected compute grid. `21 % 3 == 0`, `3 <= 4` satisfies FP32 half-destination capacity (`helpers.cpp:80–105`); `subblock_w=1` is a conservative alternate control with exactly the same static bytes. **Keeping subblock 4 with block_w 21 is illegal.** Reducing subblock alone on the original 24-tile shard does not save scratch: the sizing helper never uses subblock width.

Kernel-level instantiated contrast: original h=4,w=24,s=4 gives 6 subblocks per row and 96 tiles; proposed h=4,w=21,s=3 gives 7 subblocks per row and 84 tiles. Squaring reserves/pushes the whole 96/84-tile X block, reduction consumes it, normalization reuses the now-drained X block, and gamma consumes X and writes the same 96/84 output tiles (`layernorm_sharded.cpp:410–467,554–643`). Gamma waits/pops 24/21 tiles once and reuses them across 4 rows. Four reduction workers still process one tile row each, now gathering 8 rather than 7 partials. A smaller subblock affects register work, not those full-block reservations.

### Precision-only control, if needed

Keep 7×8 and use a norm-local compute config with `fp32_dest_acc_en=False`, retaining HiFi4. This changes intermediate tile storage to BF16 and reduces static scratch to 489,472 bytes. It is a useful capacity-vs-numerics discriminator, **not** an accuracy-approved recommendation. Do not mutate `self.kernel` globally: it is shared with other operations. Check norm and full decoder accuracy at existing thresholds before considering adoption.

### Lifetime-only control, if needed

`A/tt/multichip_decoder.py:372–380` retains `intermediate` after down projection, while evaluating the final norm. A separate experiment releasing this dead Python tensor handle after `_linear(intermediate, 'mlp.down_proj')` can measure whether the live floor rises. Other references or asynchronous buffer ownership may delay reclamation; this is not a proven sufficient fix. Preserve `x`, which is the live residual. Do not manually deallocate the original input to `_norm` merely to make the allocation fit.

For a robust fallback, retain the inherited interleaved norm for unsupported/high-pressure cases, or investigate norm-local row chunks while preserving all rows. Neither requires lowering the model context limit. Do not claim success from an isolated norm alone: rerun the exact initial-prefill → decode/reads → warm-prefill sequence and complete-layer timing.

## Dimensions, tails, and validation coverage

The harness label `--seq 4096` is not the norm's physical row count: `A/tt/optimized_decoder.py:516–553` splits prefill into 1,024-token chunks, rounds the final chunk to 32 tokens, passes `valid_tokens`, and restores logical length after concatenation. The candidate uses total padded rows across leading dimensions; batches whose total exceeds 1,024 fall back to the parent norm.

Both partitions cover exactly 168 width tiles: 7×24 = 8×21 = 168, so **no width padding or truncation is needed**. In particular, do not round 21 up to 24 merely to retain subblock 4. RMSNorm reduces channels independently for each token; padded token rows must remain separate from valid rows and the existing logical view/valid-token handling must remain intact.

A cheap Python arithmetic check covered all 32 possible tile-aligned total row counts from 32 through 1,024: the existing height rule equals `ceil(Mt/gy)` for the proposed 8-column grid. This verifies host geometry arithmetic only, not device execution. That check alone is insufficient: shared `normalization/shard_spec_validation.cpp:73–105`, called by `layernorm_device_operation.cpp:281`, also requires the exact number of occupied shards and forbids a wholly empty trailing shard. **The existing tail planner is a separate confirmed bug.** For 288 rows it chooses gy=8,height=64, but ceil(288/64)=5; a 7-column grid has 56 cores when only35 are required. Width repartitioning alone does not repair this.

Use the corrected formula shown above: block_h=ceil(Mt/8), gy=ceil(Mt/block_h), height=32*block_h. For288 rows this gives block_h2, gy5, height64. It satisfies both ceiling-height and exact occupied-shard-count checks for every padded row count32..1024. The original formula violates the occupied-count check for rows288,320,352,384,416,448,544,576,608,640,672,800,832,864,896. This tail issue does not cause the reported1024-row overlap, where all8 shard rows are occupied.

Do not crop actual rows to obtain even division. Row-wise RMS reduction uses a column reduction with explicit row stride (`layernorm_sharded.cpp:456–465`), so a partial final height shard does not mix padded rows into valid rows. Existing logical-length restoration must remain unchanged.

Follow-up coverage should include full 1,024-row chunks, short/tail inputs (1,31,32,33,255,256,257,1023 tokens), a 1,025-token multi-chunk case, and batches 1/8/16/32 exercising both the guard and fallback. Check numerical accuracy, original residual preservation, output logical shape, warm/cache reuse, and the actual static endpoint/live floor at the final norm. Norm-only microbenchmarks do not establish complete-layer benefit after shard/unshard costs.

Generic coverage exists in `tests/ttnn/unit_tests/operations/fused/test_rms_norm_sharded.py`. Its common helper (`sharded_test_utils.py:199–237`) defaults to out-of-place and does not pass the candidate's FP32 compute config; its basic matrix (`:295`) uses small widths and subblocks 1/2. It is useful contract evidence, not proof of this 5,376-wide inplace workload under retained layer allocations.

## Other potential issues / rejected headline explanations

- No evidence of a device queue deadlock or data corruption is needed to explain the observed allocation exception. An independent kernel-protocol bug would require separate evidence.
- `inplace=True` is not inherently unsafe here: the candidate first creates a different sharded layout from the replicated interleaved input; the output aliases that temporary. This relies on that layout-conversion ownership boundary. Reusing an already identically sharded residual would require a separate alias/lifetime audit.
- Lower HiFi fidelity alone does not halve scratch; the intermediate dtype is controlled by `fp32_dest_acc_en`.
- The unused XMM backing storage is a supported generic-factory inefficiency for this exact variant. It contributes to the footprint but is not necessary to change to test the smaller caller geometry. Do not remove XMM unconditionally: residual, LayerNorm, beta and distributed variants have different consumers/bindings.
- The precise identity of the lowest-address live tensor and the exact source delta between norm2 and norm3 remain unresolved. They do not undermine the direct scratch-vs-live-floor contradiction, but preclude attributing the 27,392-byte floor shift to one option.

## Report review

Rechecked headline claims against wrapper dispatch, validation, output allocation, factory declarations, compute consumers, both stack traces and manifest hashes. Independently recomputed 7×8 and 8×8 byte totals with a CPU-only arithmetic script. An independent review caught and corrected two draft errors: the normalized gamma input uses X rather than XMM, and checking only the layernorm ceiling-height condition misses shared occupied-shard validation. Rechecked both against the actual branches and shared validator. Confirmed subblock 3 legality from divisibility and destination limits rather than assuming a power-of-two restriction. No runtime pass, numerical equivalence measurement, performance win, or precise tensor-allocation identity is claimed.
