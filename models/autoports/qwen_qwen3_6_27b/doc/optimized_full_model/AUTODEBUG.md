# AutoDebug: TP4 split-greedy sampler

## Status and direct observations

This was an inspection-only investigation. No implementation code was edited and no TT hardware command was run. The prescribed fresh Codex AutoDebug session could not launch repository shell commands because its workspace sandbox could not find `bubblewrap`; the Claude fallback could not authenticate because its OAuth session had expired. The findings below were therefore checked directly against the repository code and existing artifacts rather than fabricated from an incomplete fresh-agent run.

- `artifacts/baseline_split_unpadded_reduced.json`: 16 replays in 202.670 ms, 12.667 ms/token, 78.946 t/s/u.
- `artifacts/candidate_split_padded_reduced.json`: 16 replays in 217.881 ms, 13.618 ms/token, 73.435 t/s/u.
- Both artifacts prove semantic greedy output and direct overwrite of the persistent feedback token. Both report zero per-token host token, position, and page-table refreshes.
- The earlier split-sampling profile `../full_model/artifacts/profile_candidate_greedy/tt_perf_report.csv` contains a `TopKDeviceOperation` taking 9.697361 ms on **one core**, 76.42% of the profiled interval. `SamplingDeviceOperation` is only 28.406 us and the two candidate all-gathers are 16.534 us and 13.060 us. This directly identifies local TopK, not candidate gathering or final sampling, as the dominant cost.
- The force-argmax comparison in `../full_model/artifacts/profile_active_rm_crop_final` is a useful localization control, not an acceptable final implementation: its traced argmax kernel is about 45 us, while its full-vocab all-gather is about 914-916 us per replay.

## Headline finding 1: both current widths are forced onto the single-core TopK factory

The slow behavior follows exactly from the TopK selection rules.

`models/common/sampling/tt_sampling.py` passes `sub_core_grid_topk` to `ttnn.topk`, but Qwen's `SamplingArgs` never sets it, so TopK receives the normal full-device default grid. This is not why the profile uses one core. `ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp::select_program_factory` selects multicore only when all of the following are true:

1. width is at least 8192;
2. width is strictly less than `uint16_t::max()` (65535);
3. width is a power of two;
4. `k <= 64` and the cost/grid check succeeds.

The Qwen LM head owns a large contiguous local vocabulary shard on each of four devices (`tt/model.py`, `padded_vocab_size / 4`). The prior profile's single-core TopK uses UINT16 indices and the model comment records the local width as 62,080. Consequently:

- unpadded 62,080 fails the power-of-two requirement;
- padding it with `pad_logits_to_power_of_2` makes the width 65,536, which fails the strict `< 65,535` requirement and cannot use UINT16 multicore indices.

Thus the padded candidate is guaranteed to remain single-core while sorting more elements. Its regression from 12.667 to 13.618 ms/token is consistent with this; power-of-two padding cannot unlock the optimized factory for this TP4 shard geometry.

### Smallest verify/refute experiment

Run a sampler-only TopK shape sweep, with the same BF16 tiled `[1,1,32,W]` DRAM-interleaved tensor and `k=32`, under Tracy signposts:

| Width | Prediction |
|---:|---|
| 32,768 | multicore, more than one core, sharply below 9.7 ms |
| 62,080 | single-core, approximately current slow row |
| 65,536 | single-core, same or slower than 62,080 |

Repeat 32,768 with `sub_core_grids=None`, an explicit full compute grid, and two smaller legal rectangular grids. This distinguishes factory gating from grid-quality tuning. Do not spend a full-model run on this experiment.

## Headline finding 2: split each device-local vocabulary before TopK; do not tune the current monolithic call

The smallest model-level repair is a two-stage local reduction:

1. split each 62,080-wide device-local logits shard into two contiguous logical chunks;
2. pad each chunk independently to 32,768 with negative infinity;
3. run local `topk(k=32)` on both 32,768-wide chunks (the legal multicore shape);
4. merge the 64 candidate value/index pairs locally and retain the best 32;
5. perform the existing TP4 candidate all-gathers and final `ttnn.sampling` call.

This preserves the required semantics: the physical candidate tensors remain tile-shaped, runtime sampling parameters remain `k=1,p=0,temp=1` for greedy, and the same path remains top-k/top-p capable. It also preserves `tt_out_tok` feedback and avoids a full-vocab gather.

The existing `multi_step_reduction` branch in `tt_sampling.py` is not directly sufficient: it is enabled only for a 1x1 mesh, and a simple equal split of 62,080 produces two 31,040-wide non-power-of-two calls. The TP4 implementation needs independently padded power-of-two chunks and correct device-local indices. Preserve original local indices (0..62,079) through both chunk TopKs, or add the second-chunk base before the existing per-device offset. Padded candidates must carry negative-infinity values and must never produce a valid vocabulary ID.

### Smallest verify/refute experiment

Build a sampler-only A/B with identical recorded logits:

- A: monolithic width 62,080, local top32;
- B: two 32,768 physical chunks, local top32 per chunk, local merge top32;
- C: same as B with explicit candidate-index checks around the chunk boundary and the last valid local ID.

Measure every TopK row and total traced sampler replay. Verify exact equality of the final greedy token against torch for adversarial maxima at local indices 0, 32,767, 32,768, 62,079, and on each TP device. Then verify sampled top-k/top-p behavior with fixed seeds. Only after B wins sampler-only should it be integrated into the reduced token-out trace.

## Headline finding 3: the claimed padded-vocabulary mask is absent at the LM-head/sampler boundary

`tt/model.py` says padded IDs are masked before sampling, but the inspected path only zero-pads `lm_head.weight`. `_project_lm_head_tile` concatenates those projection chunks and returns them. `Qwen36Generator._sampling_logits` returns logits unchanged. The non-force path in `tt_sampling.py` only masks the extra padding introduced when rounding the entire local TopK input to a power of two; it does not mask model-level IDs in `[vocab_size, padded_vocab_size)`.

Zero-padding weights is not a sampling mask: padded logits become zero and can beat valid negative logits. This does not explain the 9.7 ms latency, but it is a correctness hole in the path being optimized and must be fixed while restructuring local TopK.

### Smallest verify/refute experiment

Use a sampler-ready tensor whose valid vocabulary logits are all negative and whose model-padded columns are zero. Assert that greedy never returns an ID `>= vocab_size`, for each TP shard and for maxima immediately before/after the true-vocab boundary. Inspect the post-mask tensor or local candidates to prove invalid values are negative infinity. Repeat through two traced feedback steps.

## Ranked secondary hypotheses

1. **Core-grid tuning matters only after legal chunking (high confidence).** `sub_core_grid_topk=None` supplies the full default grid; it does not cause the current single-core selection. After width 32,768 reaches the multicore factory, sweep legal rectangular sub-grids because `find_topk_core_config` requires a contiguous rectangle and chooses a split from the first range. Record selected core count and latency. A sub-grid-only change on 62,080/65,536 is predicted not to help.
2. **Candidate-gather geometry is not the present bottleneck (high confidence).** Existing candidate gathers total about 30 us versus 9.697 ms for TopK. Persistent buffers may still be worthwhile later, but optimizing gather first cannot close the gap.
3. **Final generic sampling is not the present bottleneck (high confidence).** `SamplingDeviceOperation` is about 28 us on 32 cores. Keep it for the top-k/top-p-capable contract unless a post-fix profile shows it becomes material.
4. **Index dtype and boundary handling can silently defeat multicore (high confidence).** A 65,536-wide physical call cannot use the current UINT16 multicore implementation. Chunk-local physical widths must stay below 65,535, and global/local offsets must be applied after TopK without wrapping invalid padded indices.
5. **The two LM-head output chunks are not the same as TopK chunks (medium confidence).** LM-head chunks are concatenated into one 62,080-wide device-local tensor before sampling. Reusing their existing unequal boundary might avoid another split, but each chunk must be independently padded to a legal power-of-two width and candidate indices adjusted. Compare this against a clean 32,768 boundary; do not assume the existing matmul split is optimal for TopK.

## Recommended focused profile sequence

1. Static shape ledger: record true vocab, globally padded vocab, per-device logical width, LM-head chunk widths, physical TopK widths, index dtype, `sub_core_grid_topk`, selected factory, and selected core count.
2. Sampler-only width/factory sweep (32,768 / 62,080 / 65,536).
3. Sampler-only two-chunk correctness and latency probe, including invalid-vocab masking.
4. Sub-core-grid sweep on the winning legal 32,768 chunk shape only.
5. Reduced one-layer token-out trace with model and sampler signposted separately. Require no generic one-core 62,080/65,536 `TopKDeviceOperation` row.
6. Full-model traced token-out rerun only after the reduced trace closes the sampler gap.

## Expected acceptance evidence

- No full-vocab all-gather or force-argmax in the selected path.
- Local TopK profiler rows use more than one core on widths below 65,535.
- No single `TopKDeviceOperation` dominates token-out decode.
- Exact greedy equality and fixed-seed top-k/top-p correctness, including TP and vocabulary-boundary cases.
- IDs `>= vocab_size` are impossible after explicit masking.
- The sampler remains a separate nonblocking trace writing directly to the persistent decode token; token, position, and unchanged page table retain zero per-token host refreshes.

## Conclusion

The current result is not evidence that split sampling is inherently slow. It is evidence that the chosen per-device width falls through to a 9.7 ms single-core TopK, and padding to 65,536 necessarily stays on that fallback. The next experiment should be a sampler-only two-chunk multicore local TopK, not another full-model run and not force-argmax acceptance.
