# AutoDebug: Qwen3.6 P2 top-k40 requires a real two-tile sampler

**Leading diagnosis:** the prepared canonical sampler has a genuine **32-retained-candidate limit**, even when its input row is 64, 128, or 256 columns wide. Qwen correctly advertises that limit and routes P2's explicit `top_k=40` request to host sampling. Raising the advertised/Python limit alone cannot implement top-k40: compute retains one tile, mask generation wraps k40 to k8, and the writer cannot traverse a second candidate tile correctly.

**Smallest coherent repair:** keep `SamplingGenerator → TTSampling → ttnn.sampling`; add an opt-in shared TTNN path for **exactly 64 input candidates**, fully sort and preserve its two tiles, and fix mask/count/face traversal together. In canonical `TTSampling`, collect 64 candidates per vocabulary shard and compact the gathered TP4 row from **256 to 64** on device before that primitive. Advertise 64 only with the completed implementation. A general wider-input top-64 merge or a separate model-only sampler is a larger intervention. Correctness and the P2 latency target remain unverified on hardware.

## Scope and evidence boundary

- Inspection date: 2026-11-16. Repository: `/home/ttuser/qwen36-27b/tt-metal`.
- Baseline revision: **`b527d5aafa7ea4e12a9eca2dfdd350ff19dfed74`**. All source line references below refer to that revision, available with `git show <revision>:<path>`; concurrent edits can move current line numbers.
- The relevant tracked files were clean at the initial inspection. During investigation another session changed the model/common sampler and TTNN sampling files. A later snapshot contained 13 modified tracked files, 123 insertions and 38 deletions. Those changes are **not this investigation's edits or validated fixes**. They include an opt-in capacity argument, a two-tile compute/writer path and canonical compaction, broadly matching the required intervention; this report does not certify that evolving patch.
- This investigator and its three independent xhigh subinvestigators did not modify implementation, run tests/builds/devices, launch another AutoDebug session, operate servers, or reset hardware. Local checks used source reads, Git revision reads/diffs, standard-library JSON parsing, integer coordinate arithmetic and offline BF16-reference calculations. The only file written by this investigation is this report. Prior reports were not edited.

Paths below are relative to the repository root. Short source names used repeatedly:

| Name | Source |
|---|---|
| Compute | `ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp` |
| Writer | `ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp` |
| Reader | `ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/reader_values_indices_tensor.cpp` |
| Factory | `ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_program_factory.cpp` |
| Validation | `ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_device_operation.cpp` |
| Common sampler | `models/common/sampling/tt_sampling.py` |
| Qwen generator | `models/autoports/qwen_qwen3_6_27b/tt/generator.py` |

## Direct observations versus interpretation

The saved `models/autoports/qwen_qwen3_6_27b/doc/eval_readiness/harness_requests/runs.json` records the reasoning request as temperature 1, top-p 0.95, top-k 40, seed 42, and `seeded_sampling_policy="request"`. Its 32-user profile records 134 host-sampling log lines and zero device-sampling lines; its plain control records zero host lines. Both complete 32 requests with 128 generated tokens each and no request errors. The stored TPOT medians give:

```text
356.79754386215996 / 83.45769281777751 = 4.2751906003583855
required ratio_max = 1.15
```

The result is also recorded as critical/fail in `doc/eval_readiness/harness_requests.json:48–54`. These are saved observations, not a new reproduction. The explicit route is supported by source: adapter `tt/generator_vllm.py:31–40` advertises `max_device_top_k=32`; Qwen generator `:171–180` constructs canonical `SamplingGenerator` with `max_top_k=32`; both configure-time `:430–431` and sampled-prefill `:840–842` validation reject stochastic k40.

**Interpretation:** the width limit explains why this workload takes the expensive host compatibility route. The saved route contrast and latency establish a strong repair target, but do not prove that host sampling alone accounts for every millisecond, or that any proposed width64 implementation will satisfy 1.15. The failing P2 execution does **not** establish corrupted on-device k40 output: that request is routed away from the unsupported primitive.

Checkpoint defaults are T1/p0.95/k20 (`runs.json:8–11`); P2 explicitly overrides k to40. Changing defaults or silently lowering k would change the workload instead of repairing it.

## Canonical lowering and the actual dimensions

The adapter delegates to Qwen's generator, whose `_sample_graph` calls canonical `SamplingGenerator._run_sampling` (`Qwen generator:361–372`; `models/common/sampling/generator.py:412–445`). That method applies requested penalties, calls `TTSampling`, and updates token history. The final sampling wrapper forwards directly to the shared primitive (`ttnn/.../sampling/sampling.cpp:11–21`); it does not choose a wider implementation based on k.

Qwen uses vocabulary248320 on one 1×4 TP replica. Its per-device logits are physically `[1,1,32,62080]`, including smaller logical batches (`Qwen generator:168–178`; common sampler `:301–304`). The common path pads the per-shard vocabulary to65536 with negative fill for top-k (`:1331–1343`), selects local candidates (`:1355–1365`), and gathers values and indices over the vocabulary axis (`:1370–1404`). Original-shard offsets correctly use62080, not the temporary padded65536 (`:639–650`, `:1433–1447`).

| Quantity | Existing max32 | Proposed max64 |
|---|---:|---:|
| Logical vocabulary per TP shard | 62080 | 62080 |
| Padded local top-k input width | 65536 | 65536 |
| Local candidates per shard | 32 | 64 |
| Gathered candidate width | 128 | 256 |
| Gathered width tiles, Wt | 4 | 8 |
| Required input to the scoped width64 primitive | — | 64 after global compaction |

This distinction is essential: **`max_top_k=64` does not make the TP4 gathered tensor width64.** Keeping local32 also cannot guarantee global top40: if one shard owns all40 largest logits, eight are discarded before the gather. The union of each shard's top64 contains a valid global top64, apart from the separately noted tie-policy limitations.

The primitive accepts BF16 TILE interleaved values, matching-shape UINT32/INT32 ROW_MAJOR token IDs, one k/p/temperature value per user, and H in1..32 (Validation `:36–93`, `:142–159`). On WH/BH it generates UInt16 **candidate positions** internally with 16-bit destination accumulation; the final ID table remains32-bit (Factory `:117–138`; Reader `:45–61`). A candidate position below256 is not a vocabulary token ID; the writer maps it through that user's ROW_MAJOR ID row (Writer `:146–150`, `:218`). Increasing retained candidate width does not require truncating global vocabulary IDs to16 bits.

## Proven 32-candidate assumptions that must be repaired together

### 1. Compute always emits and normalizes one retained tile

Compute `:469–488` explicitly sets:

```cpp
nearest32_K = 32;
logk = 5;
Kt = nearest32_K / tile_width;  // 1 for normal 32×32 tiles
```

`top_k` emits `ceil(K/32)` value and local-index tiles (`:375–415`), so only32 candidates reach masking/softmax. Mask, temperature, row reductions, exponentiation and normalized output all use `Ht*Kt` (`:492–503`); at Ht1 this is one tile regardless of input Wt.

The merge tree is also structurally a top32 algorithm. Each merge loads two value tiles plus two index tiles, but packs only the winning left value/index tile back (`:318–360`). Merely setting K64/log6 and transposing two output tiles does not produce global top64 for Wt4 or8: the second output tile is not maintained as the second half of a global64 result. The low-level merge itself clamps `k_max` to32 (`tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_topk.h:1213`; corresponding Blackhole file `:1235`). A wider K requires a different merge schedule, not a constant change.

### 2. Current mask generation turns k40 into k8

Factory `:496` passes `ids_per_batch=tile_width=32` to the writer. Writer `:107–109` invokes `generate_mask(..., ids_per_batch/32, k-1)`, so its width is one tile. The helper computes:

```text
cur_pos_in_chunk = cur_pos % (Sk_chunk_t * 32)
```

at `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp:251–278`. For k40 it uses `39 % 32 = 7`, leaving only ranks0..7 unmasked. This is a more precise failure than simply “clamps to32.” In a corrected two-tile path, `39 % 64 = 39`: tile0 remains unmasked and tile1 keeps its first8 columns.

This is an unsupported-k consequence, not proof that baseline P2 ran the wrong mask. The public binding documents k in `(0,32]` (`sampling_nanobind.cpp:110`), while host validation checks k dtype/layout/shape without reading its payload (Validation `:142–159`). The model's existing gate prevents P2 from entering this case.

### 3. The writer's two phases describe two faces of one tile

Writer `:152–166` addresses the first16 ranks in one face row, then a contiguous `k-16` span in the right face row. That works through32. For k40 the latter span has24 entries; its last8 are not that user's ranks32..39.

For a standard face-packed32×32 tile, the correct **element** offset for user `u` and sorted candidate rank `r` is:

```text
offset(u,r) = 1024*(r//32)
            + 512*(u//16)
            + 256*((r%32)//16)
            + 16*(u%16)
            + (r%16)
```

The four-face layout and column-based index generation are directly documented/implemented in `ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp:23–54`. Compute transposes sorted values and positions back before packing them to writer buffers (`:377–415`). Apply the same element coordinate to both buffers, with each buffer's own element type; use the resulting local position to index the current user's32-bit token-ID row.

| User | Current spans if k40 is traversed | Correct offsets for ranks32..39 |
|---|---|---|
| 0 | `0..15`, `256..279` | `1024..1031` |
| 15 | `240..255`, `496..519` | `1264..1271` |
| 16 | `512..527`, `768..791` | `1536..1543` |
| 31 | `752..767`, `1008..1031` | `1776..1783` |

User0's offsets272..279 are user1's ranks16..23. User15 crosses into a different face/row region. User31 runs past the sole produced tile into the unproduced second buffer slot. Thus “output is in vocabulary” is too weak an oracle: another row's local position can still map to an in-vocabulary ID in this row's ID table.

**Causal caveat:** actual traversal depends on top-p stopping. With the unchanged wrapped mask and p0.95, a cutoff within the first8 nonzero candidates can hide the bad address span. Do not claim every unsupported k40 call reads another row. The coordinate defect remains a necessary repair when real probability mass extends beyond rank31; fixing only compute or only masking exposes it.

### 4. Writer readiness and consumption also assume one tile

Writer `:141–142` waits for one local-values and one local-indices tile, and `:239–240` pops one of each. Emitting two tiles without changing both waits and pops permits reading an unpublished tile or leaving a produced tile unconsumed. Mask production, compute Kt, and writer consumption must describe the same retained width.

For the scoped W64/Ht1 path, existing capacities already fit: values/local-values/output-indices each hold two entries; mask holds four; transposed buffers hold Wt entries (Factory `:228–240`). No broad buffer-size increase is implied by this repair.

| Buffer | Existing retained32 contract | Required scoped retained64 contract |
|---|---|---|
| Reader values/generated positions | Push Wt; compute consumes all Wt | Wt2: push2, consume2 |
| Transposed values/positions | Wt tiles; lossy top32 merge tree | Full64 sort in DST; pack/push2; transpose both; pop2 |
| Sorted values | Emit1; in-place stages rotate1 | Emit2; mask/temperature/exp rotate2 |
| Top-k mask | Writer push1; compute consumes1 | Writer push2; compute consumes2 |
| Cur-max / cur-sum | One row-reduction tile each | Still one each; reduce across2 columns |
| Normalized local values | Compute push1; writer wait/pop1 | Compute push2; writer wait/pop2 |
| Sorted local indices | Compute push1; writer wait/pop1 | Compute push2; writer wait/pop2 |
| Temperature, MAX/SUM scalers, random | One entry each | One entry each, balanced cleanup |
| Final token-ID rows | Reader push num_users; writer wait/pop num_users | Same:32 rows for P2 |

Compute `:505–517` already balances cur-max, masks, temperature and scalers using the corresponding counts; a Kt change must retain that balance. For Wt2 the reader supplies exactly the two input tiles needed by local sort. The transposed buffers hold exactly2, sorted-values and index buffers hold2, and normalized output holds2; both result tiles can be published before the writer consumes them. There is no demonstrated new capacity shortage at this size.

## Recommended intervention and alternatives

### Shared TTNN: scoped full64 sort, preserving the legacy path

`tt_metal/hw/inc/api/compute/topk.h:33–65` defines local sort across64 values spanning two value tiles plus two index tiles, with ending phase `log2(K)-1`. WH's implementation derives sequence length `1 << (phase+1)` and handles64 data (`ckernel_sfpu_topk.h:1099–1123`). Thus **exactly Wt2** can use phase5/log6, keep both sorted tiles, and bypass the lossy sampler merge tree entirely.

The minimal semantic changes are:

1. Select retained capacity64 only for the explicitly supported W64 path; enforce W64 for that mode. Keep the legacy32 path/default available. A backward-compatible optional `max_top_k=64` operation attribute is one clear implementation; deriving capacity from W64 is another policy, but changes behavior/performance of existing width64 calls. Neither requires a new vLLM routing interface.
2. Fully sort both tiles, skip the legacy merge tree, transpose/pack both values and matching candidate positions, and run masking/temperature/softmax with Kt2.
3. Pass the same retained capacity to the writer. Generate two masks and wait/pop two values and position tiles.
4. Replace the two-face-span assumption in the wide mode with rank iteration using `offset(u,r)` for both nucleus accumulation and random selection. Handle p0 as existing first-positive-candidate/greedy behavior; preserve the current strict `cum_mass > p` rule unless deliberately changing that contract. Keep a valid retained-candidate fallback if floating accumulation falls short of the draw.
5. Update the public k/candidate-width contract, argument propagation, program specialization/cache identity and common caller bounds together. No host readback is needed to inspect k: model metadata already bounds admitted values, and the device reads per-user k. Do not silently wrap invalid wider k.

A full shared W128/W256→top64 implementation could remove the compaction op, but requires the correct two-tile-per-sequence merge algorithm and its own validation. It is not the smallest repair established here. Increasing K alone, retaining stale tile1, or skipping all merges at Wt8 is incorrect.

### Canonical common/model integration: compact candidates on device

After the shared primitive works, use the existing canonical path with these changes:

1. Construct with local capacity64; gather all256 value/ID pairs and add the original62080-based vocabulary offsets.
2. Apply the existing greedy tie-value adjustment **before global compaction**. It guarantees the lowest global ID among gathered tied maxima survives selection for k1 users; compacting an unboosted tied set can discard that candidate (`common sampler:962–986`, `:1450–1458`). Stochastic rows are unchanged by this boost.
3. Select global64 from that adjusted gathered row with ordinary `ttnn.topk(k=64)` and carry/gather the corresponding global token IDs. Untilize those64 IDs, then call the shared opt-in W64 sampler with the true per-user k40/p/temperature.
4. Keep original gathered tensors for any log-probability consumers, maintain explicit lifetimes, and warm/recapture changed shapes before trace reuse. Preserve existing penalty/history, persistent output-token and seed ownership.
5. Raise the adapter capability, generator construction limit, configure validation and sampled-prefill validation together. Make generic formatting capacity-aware for other callers; Qwen's direct reset path bypasses that formatter today.

Ordinary `ttnn.topk` has an independent implementation that supports64 outputs: `topk/topk.cpp:583–587` aligns k; `topk/device/kernels/compute/topk.cpp:319–342` retains both initial tiles, `:446–447` runs the local sort, and `:512–513` emits all output tiles. Two available pairing methods are:

```python
# Use normal positional top-k, then gather the matching global IDs on device.
values64, positions64 = ttnn.topk(adjusted_values256, k=64, dim=-1)
ids64 = ttnn.gather(global_ids256_tile, dim=3, index=positions64)
```

or carry the TILE global-ID tensor directly through `topk(..., indices_tensor=global_ids256_tile, stable=False)`. The latter avoids a separate gather; caller labels must remain32-bit to preserve IDs above65535. `topk_nanobind.cpp:59` documents this overload and `topk/device/topk_device_operation.cpp:239–274` checks layout/type/tile-grid compatibility. **`stable=True` with supplied labels is rejected** (`:203–210`). These alternatives are source-supported interfaces, not measured performance recommendations.

A model-only reconstruction from TTNN softmax/CDF/random/comparison/gather primitives could also stay on device within canonical `TTSampling`, but it would need a new faithful nucleus/renormalization/RNG implementation, per-user seed behavior and trace validation. It is a larger correctness surface than the scoped shared-kernel correction. Copying/adopting another sampler class or adding an adapter-specific route does not repair the selected canonical chain. Global compaction alone cannot overcome the existing primitive's retained32/mask/writer assumptions.

### Trace, seed and completion boundaries

Qwen owns its traces and directly captures canonical `_run_sampling` inside `_sample_graph` (`Qwen generator:361–372`); the explicit-seed internal-trace exclusion in `SamplingGenerator.sample()` (`models/common/sampling/generator.py:673–675`) is **not this model's route**. Qwen advances persistent seeds on device and restores inactive token/seed rows. Decode executes the decode and sample traces and can return resident tokens without a transfer (`:1090–1097`).

Keep `manual_seed` immediately before the sampler draw, after any new global top-k/gather work. Existing common code intentionally does so because intervening SFPU work can perturb PRNG state (`common sampler:1460–1477`). Preserve request seed42, absolute-position reset and inactive-slot behavior instead of changing seeded policy to obtain a faster route.

Qwen `_execute` uses `blocking=False` (`:559–561`), so trace enqueue/capture completion alone does not demonstrate successful device execution. Correctness validation must wait for sampled output or an explicit device synchronization after the queued chain, and also exercise teardown/reuse after outstanding work completes. Within the primitive, Reader NoC barriers precede buffer pushes, compute publishes values/positions, writer waits on them, then performs its output NoC write and barrier (Reader `:38–61`; Writer `:139–147`, `:243–246`). These boundaries make the two-tile readiness/count agreement material even if a failure only becomes visible at later synchronization.

The repair should add no candidate/logit readback or host RNG. Tests may read final samples for their oracle outside measured execution. Existing serving has ordinary token-delivery boundaries; this report does not claim every prefill/decode API call has zero host interaction.

## Discriminating fixed-candidate tests to run after implementation

**None of these tests was run here.** They distinguish a real top40 distribution from a truncated subset that happens to satisfy membership. The current suites cannot establish this extension:

- `tests/ttnn/unit_tests/operations/reduce/test_sampling.py:183–193` covers W64/W256 with k10/15/20/25/30. Its `:69–96` oracle uses identity IDs and original FP32 logits, only checks membership, and its p branch ignores k and admits an additional cutoff token.
- `test_sampling_distribution.py:36–38` fixes W128/k20; its stronger per-user histogram oracle (`:61–80`, `:125–144`) does not exercise ranks32..39.
- `models/common/tests/test_tt_sampling.py:39–40` fixes `MAX_TOP_K=32`.
- Use **p1**, not p0, to test full top40 support. Writer `:175–181` treats p0 as an immediate first-positive-mass cutoff; the existing k32/p0 argmax assertion at `test_sampling.py:107–124` agrees with that behavior.

### A. Direct primitive W64: support, labels, and distribution

Use32 users, k40, temperature1 and p1. Freeze candidate logits instead of regenerating model text. One fully specified fixture is:

```text
S = {0..7, 12..19, 28..35, 44..51, 56..63}  # 40 hot source columns
B = [15,16,31,32,47,48,63,0]
C = sorted(S minus set(B)) followed by B
logits[C[r]] = -r/64                 for r=0..39
logits[remaining_columns[s]] = -8-s/16
token_id[u,j] = 131072 + 1024*u + 3*j
```

The values are BF16-exact, ordered top40 is unambiguous, and sorted ranks32..39 originate at source columns around face/tile boundaries. Token IDs are nonidentity, distinct per user and below Qwen's vocabulary size, including IDs above65535. Decode samples using an explicit ID lookup.

Build the reference from the actual BF16 input: top40, FP32 softmax of those logits, BF16 probability rounding, then normalization. With this reference, ranks32..39 hold about **15.342%** mass and the least likely retained candidate about1.819%. At4096 draws per user its expected count is about74; probability of never seeing that candidate under independent reference draws is about `2.25e-33`. These are offline oracle calculations, not device measurements.

Require all of:

- Zero samples outside that user's mapped top40 IDs.
- Every retained candidate is reached, including every rank32..39.
- Per-user masses for rank groups0..15,16..31,32..39 match reference within a predeclared statistical/numeric tolerance, for example `6*sqrt(q*(1-q)/N) + 0.005`. Retain individual histograms to expose duplicated/swapped ranks. This easily separates15% missing mass from small BF16/approx-exp differences.
- Repeat a flat-row control. With64 equal logits, the shared stable position tie rule selects the first40 input positions and samples uniformly among them. Missing ranks32..39 disproves real width40 support even if all sampled IDs belong to the allowed set.

Add dominant-winner sentinels at source columns `{0,15,16,17,31,32,33,47,48,49,63}`, with a one-token nucleus. They isolate input tile/face/ID handling from distribution errors, but cannot by themselves prove support beyond sorted rank31.

### B. Pairing and row isolation

Reseed identically and change only the ID table by a known bijection: each output must undergo exactly that bijection. Independently permute each user's **(value, ID) pairs**; with unique sorted logits, identically reseeded sampling should return the same token sequence. These checks test pairing and mapping; truncation can still pass them, so keep test A.

Alternate rows whose hot columns are0..39 and24..63, with distinct per-user IDs and a strict rank order. Explicitly examine users0/14/15/16/30/31. Adjacent rows' rank16..23 local positions then map to cold columns in the current row, making the writer's cross-row face traversal observable when reached. Avoid pooling all users into one histogram: pooling can conceal row swaps or local-position substitution.

### C. Nucleus cutoffs after face16 and tile32

Using fixture A, BF16-exact p values0.4921875 and0.875 produce reference retained counts17 and34 respectively. The nearest reference CDF margins are approximately0.00977 and0.00806, comfortably away from an exact boundary. Derive/assert those margins in the test before using them as a strict membership oracle. Check every retained candidate has positive, statistically consistent frequency and every excluded candidate has zero frequency. The p0.875 case must reach ranks32..33.

Also cover the actual P2 p0.95, quantized to BF16 **0.94921875**. Its exact cutoff can be sensitive to device approximations; do not conceal missing ranks32..39 behind a broad nucleus-edge exemption. Use p1 and the margin-separated cases for decisive width tests, and diagnose the P2 boundary separately. Use mixed per-user k values `{1,16,17,31,32,33,40,48,49,63,64}` plus p0/p1 controls to cover both old and new mask/face boundaries.

### D. Canonical TP4: candidate completeness and provenance

Exercise full `SamplingGenerator → TTSampling` with physical32 rows and max64, first eager and then Qwen's traced sample graph. Put all global top40 logits on a single vocabulary shard, including source candidates whose local ranks are32..39. Move that hot set across all four shards. Require mapped global support and distribution, not just in-vocabulary outputs. This falsifies a repair that leaves local capacity32 while widening only final sampling.

Use different values/IDs for each row, and include token IDs around original shard offsets62080/124160/186240, local padding62080..65535, and the vocabulary endpoint248319. No padded token may be sampled. Verify the final primitive receives W64 while its inputs originate from gathered W256, and IDs undergo the same selection as values. Compare same **frozen** inputs across compaction implementations if choosing positional gather versus supplied-label top-k.

### E. RNG, trace replay, cache and performance acceptance

Use the same explicit core set for `manual_seed` and sampling; the canonical suite explains the required seed-to-core mapping at `test_tt_sampling.py:121–132`. Seed per-user streams once and advance draws, then reset and require exact same-implementation replay. Do not repeatedly pass the same operation seed for a histogram: that would repeat one draw; varying that compile-time seed can rebuild kernels (`test_sampling_distribution.py:109–111`). Include distinct seeds, an explicit zero seed via the supported manual-seed path, and Qwen's request42/position reset behavior. Cross-implementation/HF token-for-token seed agreement is not a prerequisite for distribution agreement.

Replay with changed logits/candidate winners, parameter reloads and inactive rows; synchronize/read final results after queued work. Verify no new host sampling/logit or candidate readback, no stale outputs, correct slot seed advancement, and no repeated trace/capture allocation on warmed replays. Retain ordinary k≤32 controls on the legacy32 path and mixed greedy/stochastic64-capacity batches. Cache specialization must distinguish the retained capacity if an explicit attribute is introduced.

Only after those correctness checks should the unchanged P2 reasoning profile run again at1 and32 users with T1/p0.95/k40/seed42 and the original control/ratio threshold. Record actual device-route counters, completed output counts and host-route absence alongside latency. P2 remains failed until the real required result passes; inspection and a width declaration are not acceptance evidence.

## Other potential issues and adjudicated non-causes

1. **Temperature convention is discrepant for T≠1 in the baseline Qwen path.** Qwen passes its nonzero user temperature directly to `reset_sampling_params` (`Qwen generator:425–445`), while compute multiplies logits by that scalar (`Compute:496`); generic formatting inverts temperature (`models/common/sampling/generator.py:845–859`). This appears to scale by T instead of1/T on Qwen's direct path. P2 uses T1, so this cannot explain its route/latency failure. Test T0.5/T2 with fixed logits separately; do not attribute the P2 failure to it.
2. **Generic formatting hard-caps32, but is not Qwen's active reset route.** `format_sampling_params` clamps k at32 (`generator.py:854–859`), including its `apply_decode_state` callers (`:318–340`). It needs a capacity-aware opt-in for general wider callers. Qwen's direct `configure_sampling → reset_sampling_params` path bypasses it, so it is not an additional present cause of Qwen P2's cap.
3. **W32 is admitted/documented although the pairwise reader/compute contract does not support it.** Validation `:79–93` accepts Wt1; Reader `:38–52` produces one input tile, while Compute `:259–263` waits for two. This is a direct separate progress concern, not a cause of P2's W128 baseline or the scoped W64 repair. Do not use W32 as the nominal passing control or claim all-equal values are required to expose the mismatch.
4. **Greedy ties have a documented candidate-completeness limitation.** The common adjustment selects the lowest global ID among gathered tied maxima; an unstable local top-k can omit the true lowest global ID when a shard has more maxima than its local candidate count (`common sampler:980–984`). Increasing to64 changes the boundary but does not remove this general limitation. Unique-logit tests isolate width correctness; a separate tied case should retain the existing declared policy.
5. **Naming the transposed source in output pack reconfiguration is not itself a width bug.** Compute `:380`/`:400` reconfigures from the transposed buffer, then packs to another buffer. Factory assigns identical BF16 formats to both value buffers and identical UInt16/Int32 formats to both index buffers (`:130–138`, `:230–237`). The formats agree on the inspected path. This discrepancy is not a headline root cause.

## Post-draft claim audit

The headline claims were revisited against the frozen baseline, rather than against the evolving worktree. Independent source reads and local arithmetic confirmed hard K32/log5, one-tile output, `ids_per_batch=32`, mask39%32→7, standard tile/face coordinates, both Qwen admission checks and TP4 gathered widths128/256. An independent post-draft kernel review confirmed the coordinates, scoped full64 sort and buffer ledger; its wording correction places the local sort before the transposed-buffer push. The low-level local-sort64 contract and separate normal-topk64 implementation support the proposed scoped intervention; they do not prove its hardware result.

The audit rejected three overclaims: the current P2 run does not exercise broken device k40; the writer overflow is conditional and can be hidden by early cutoff; and changing constants cannot turn the generic top32 merge into top64. Temperature, W32 progress, generic formatter behavior and suspicious-but-compatible packing are separated above because they do not explain the stated P2 observation. Remaining uncertainty is hardware correctness/distribution, asynchronous replay/cache behavior and whether the completed canonical device path meets the unchanged1.15 latency gate.
