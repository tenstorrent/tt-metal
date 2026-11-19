# P2 native top-64 repair design

Source/CPU investigation started **2026-11-17 01:20:42 UTC**. No TT device access, server request, production edit, plugin/vLLM change, or performance claim. The prior audit established the boundary; this investigation identifies a concrete extension of the **same shared sampling operation**, rather than another adapter sampler.

## Concrete candidate

The existing LLK already has the needed primitive: `topk_local_sort(..., idir=0, i_end_phase=5)` sorts all 64 values across value DST tiles 0/1 with index tiles 2/3. Its interface explicitly supports phases through 5 (`tt_metal/hw/inc/api/compute/topk.h:42–66`). The current general `ttnn.topk` kernel uses this primitive in a tile insertion cascade for K>32 (`ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk.cpp:199–490`). Reuse that primitive in the existing sampling kernel; no new stochastic algorithm is required.

Add an opt-in scalar `max_top_k=32` argument to `ttnn.sampling`, its native attribute struct and nanobind entry. Accept only 32 or 64; require input width at least the selected width and, for this change, at least two tiles for the 64 route. Preserve the existing 32 route verbatim. The attribute must participate in the normal operation cache key and become a compute/writer compile argument. This is a TTNN API extension, **not** a plugin/vLLM API change. Do not infer the requested draw width from runtime k tensor contents or silently change existing callers.

For the 64 route, make these changes:

1. **Compute top-k retention.** Existing input loading, index creation, transpose buffers and local sorting remain. Set local-sort phase to 5, so every input pair is fully sorted over 64. In `top_k`, replace the existing merge-tree block only for K64 with the insertion cascade below. Existing single-tile merge retention is unsuitable for K64. Retain both leading tiles and transpose both into the existing values/index output buffers.
2. **Mask and softmax.** Pass the selected draw width as the writer's `ids_per_batch` rather than hard-coded `tile_width`. The existing `generate_mask` then produces exactly Kt=2 tiles for k≤64. Set compute K/logk to 64/6, making the existing masking, temperature and max/sum softmax stages consume both tiles. Their `cols` and `num_tiles` arguments already support 2; this still needs a kernel probe. Keep BF16 values and the existing compute configuration unchanged.
3. **Writer traversal.** For the 64 route wait for, address, and pop two local-values/index tiles. Iterate ranks with the tiled offset formula below; keep the existing random-digit construction and FP32 cumulative top-p/draw semantics. Preserve the original 32 writer branch to avoid unrelated behavior/performance changes.
4. **Canonical Python path.** Only after the native probe passes, have `TTSampling.forward` pass the configured supported width to `ttnn.sampling`. Construct Qwen's canonical sampler with 64; update its explicit input validation and parameter tensor bound together, followed by `max_device_top_k=64` and its serving-contract evidence. The adapter continues delegating to the generator with the same `tt_out_tok`, seed buffers, trace and feedback. Other unsupported fields remain explicit host compatibility requests. No precision configuration changes.

### Exact top-64 insertion invariant

After the existing first pair's full 64-sort, `buffer[0]` and `buffer[1]` hold the best 64 in descending order. For each subsequent transposed input tile t (32 values):

```text
for t = 2 .. Wt-1:
    carry = buffer[t]
    for level in {0, 1}:
        load values buffer[level], carry into DST0/1
        load corresponding original-position indices into DST2/3
        topk_local_sort<existing stable/tie flags>(0, 0, 5)
        pack high 32 values/indices back into buffer[level]
        pack low 32 values/indices back into buffer[t] as carry
    discard carry
```

Use the existing in-place pack operations and buffer lifetime convention, with commit/wait before reusing DST and no consumption of the full Wt intermediate until the cascade completes. Sorting the first pair is sufficient initialization; sorting later pairs first is harmless because every cascade comparison performs a full 64-sort anyway. A more elaborate optimization is unnecessary for the first discriminator.

Proof of retention: merging the current best-32 tile with the new tile leaves the best 32 in level 0. Its 32 losers plus the old second-best tile contain every remaining candidate that can enter positions 33–64. Sorting those and retaining their best 32 yields precisely top64 of the enlarged prefix. Original input-position indices remain attached, so the stable comparator can preserve the existing tie contract. This reasoning depends on the LLK primitive doing the documented stable sort; the CPU model does not prove that device implementation.

### Correct two-tile writer addressing

For row/user `u` and rank `r` in [0,64), use the element offset (not bytes):

```text
tile = r / 32; col = r % 32
offset = tile*1024 + (u/16)*512 + (col/16)*256 + (u%16)*16 + col%16
```

Both BF16 probabilities and local index tiles use this element layout; their element types still differ by architecture exactly as the current code selects. The final global token ID remains `final_indices[local_indices[offset]]`. The final-index row-major sticks already contain **all input candidates**, so their reader/layout needs no new narrowing or host mapping.

For the top-p pass, scan increasing r through k, accumulating FP32 probability until the existing strict `cum > p` threshold, and retain the crossing token. For the stochastic pass, scan that retained prefix against the same random threshold. Initial/fallback token selection and underflow handling must preserve current semantics; do not use this change to modify nucleus definitions. Masked ranks above k are never scanned.

## Buffer and compilation accounting

Current factory allocations already provide two entries each for `SAMPLING_VALUES`, `SAMPLING_LOCAL_VALS`, and `SAMPLING_OUTPUT_IND`, and four for `SAMPLING_TOPK_MASK`. For K64 these are exactly enough for two result tiles and two masks. The transposed value/index buffers have Wt entries, so they can hold accumulator0/1 and the current carry tile without a new allocation. Scalar max/sum buffers remain Ht=1. Random output remains one tile. Final global-index storage remains one W-wide stick per user.

Change the writer's local-values/index wait/pop counts from 1 to Kt only in the new route. Compute must pop Kt mask entries after its two-tile operations. The existing scalar-temperature multiply acquires two value DST tiles for Kt2, within the four slots already used by the sort on Blackhole. Existing input reader requires no change: it streams Wt value/index tiles and the same row-major global-ID table. All 1–32 user counts must still work, especially rows 15/16/31 which cross the face boundaries.

Qwen's four-shard local top64 yields 256 gathered candidates if the current per-shard path is used, hence Wt8. Index offsets are already allocated as `max_top_k * num_devices` and scale to 256. The Qwen sampler has no `GALAXY_NUM_LINKS` config, so its normal gather-link setting remains 1. This candidate changes the small candidate tensor widths and sort work, not activation/CCL dtype, KV cache, layer policy, model traces or mesh topology. Sampling trace size and measured overhead still need verification.

## Discriminating probe, before any full model rerun

Use a bounded, exclusive single-chip Blackhole job after the parent's P5 releases devices. Start with native `ttnn.sampling`, **without** weights or a model server. No profiler is needed.

- Compile separate default32 and opt-in64 variants on identical BF16 logits and UINT32 non-contiguous token IDs. Use widths64 and256, users1 and32, plus mixed k values1/16/31/32/33/40/63/64. Deliberately give every row a disjoint token-ID set. Compare eager calls and 128 changing-seed trace replays with a preallocated output tensor; no reads or parameter updates inside capture. Capture `manual_seed` with persistent seed/user-ID tensors followed by sampling, as the canonical sampler does, and update those tensors between replays. Do not change the sampling operation's scalar `seed` attribute per draw, which could produce distinct cached programs instead of testing one trace. Observe stable program count after warmup, one capture/release, exact unchanged-seed repeats and correct output-buffer identity.
- For k40/p1 use nearly uniform descending BF16 logits so every top40 candidate has meaningful mass. Across 128 seeds require all outputs to belong to that row's exact top40 and at least one output per row to have rank33–40. This rejects an unnoticed top32 draw; it does not demand an exact statistical fit. Do the analogous k64 tail check. Use an adversarial fixture where the global top64 span late input tiles, so retaining two stale local tiles cannot pass.
- For p0 the winner must be exact greedy for every k. For p0.95 and a small boundary-free p (e.g.0.5), validate membership against a BF16-aware reference with a deliberate margin around the cutoff; do not mistake BF16 rounding at an exact cutoff for a width failure. Check temperature1 and0.5 inverse-scale parameters through the shared wrapper.
- For a strict writer-only discriminator, expose a diagnostic compute output or craft fixed input ranks: row0 ranks32–39 live at offsets1024–1031, whereas the old writer would read row1 positions16–23 at offsets272–279. Disjoint per-row IDs make contamination detectable immediately.
- Once that passes, run the **same canonical `SamplingGenerator`** on a 1×4 mesh with synthetic sharded logits and cross-shard winners, manual seeds, and in-place token output. Verify exact support, active/inactive slot semantics and trace replay. Only then enable model capability64 and request an unchanged P2 reasoning profile on the real model. The final default20 path and affected canonical sampling/qualitative/performance gates would need refreshed evidence after a production change.

## Rejected shortcuts and concrete failures

| Candidate | Why it fails |
|---|---|
| Capability64 / Python clamp64 alone | Compute still retains32; writer reads other rows for k>32. |
| Change compute `nearest32_K` to64 alone | Existing merge writes back only the first value/index tile. The second output tile is not global ranks33–64; writer and mask remain wrong. |
| Fix sort and writer but keep `ids_per_batch=32` | `generate_mask` computes `(k-1) % 32`; k40 wraps39 to7, masking to an eight-candidate prefix. |
| Supply more gathered candidates only | Input width is independent of retained draw width; still reduces to32. |
| Add a separate adapter categorical draw | Breaks the mandated canonical full-model token-out path and duplicates state/seed semantics. |
| Clamp the reasoning profile to32 | Changes the exact accepted harness request; invalid P2 repair. |
| Globally force seeded requests to host | Does not improve this host-route ratio and violates the required default seeded policy. |

## CPU evidence and limits

`p2_top64_design_model.py` and `p2_top64_design_model.json` validate the insertion invariant against an independent global sort on **1280** fixtures (width64/128/256/512/1024, random and heavily tied values). They exhaustively check **2048** row/rank addresses for uniqueness and absence of cross-row overlap, and demonstrate the old k40 writer and mask errors numerically. Result: `PASS_CPU_DESIGN_MODEL_ONLY`. This is design evidence, **not** TT kernel correctness, compilation, trace or performance evidence.

This is a plausible bounded repair candidate rather than proof of success. Its main unresolved device risks are LLK stable-sort behavior under repeated in-place packing, two-tile buffer balance/softmax correctness, and added candidate-sort latency. No existing P2 failure or source capability has been relabeled.

Investigation ended **2026-11-17T01:25:47.561279+00:00**; continuously active source/CPU work for approximately **5.1 minutes** (wall interval 306s, including short local reads and the CPU model). No waiting for hardware was counted. This is not a two-hour P2 repair box and does not justify a performance waiver.

## Reviewable patch and probe, prepared after the next owner resume

Implementation preparation started **2026-11-17 02:43:00 UTC**. Only these stage-owned diagnostic artifacts and the model-owned probe were written; production native/common files and loaded libraries remain untouched.

- `native_sampling64_candidate.patch`: nine existing native/API files. Adds opt-in native `max_top_k=64`, restricted to Blackhole, while preserving the default32 compute merge and writer bodies. Implements the two-tile cascade, mask, softmax extents and writer traversal. A full intermediate-buffer pop/push handoff follows **each** in-place cascade merge so unpack cannot read the newly packed carry before its publication.
- `canonical_sampling64_wiring.patch`: separate, **unapplied** three-file proposal for the common sampler, Qwen generator and capability declaration. An explicit `device_sampling_draw_width` constructor option defaults to32 for unrelated models. Qwen would select64 and continue the existing `SamplingGenerator`/`tt_out_tok` path. Do not apply this capability change before native and canonical four-chip verification; its serving-contract evidence must be updated with actual results.
- `tests/run_vllm_sampling64_probe.py`: focused native probe with `--cpu-only`, unchanged-default32 baseline mode and opt-in64 mode. It uses one chip, persistent parameter/output buffers and one retained trace per shape. Width64 first isolates the two-tile draw, followed by width256 to exercise the cascade. Users1/32 exercise the row/face boundaries. Six dynamic profiles cover k40, full draw width, mixed k1/16/31/32/33/40/63/64, p0 greedy, a nucleus threshold with a measured numerical margin, and changed logits. Every replay is compared exactly with a separately reseeded eager call; the output is poisoned before replay. Tail assertions require ranks33–40 for k40 and ranks41+ for k63/64, without a fragile requirement to observe the single rank33 outcome for k33. The trace must survive changed seed/k/p/temperature/logit values with no additional program entries. Raw eager and replay tokens go to the result's `.tokens.raw.log` companion.

Validation already performed, without devices:

- Both patches pass `git apply --check`, including a combined check.
- Both original32 kernel branch bodies preserve their non-comment tokens exactly; `native_sampling64_patch_sources.json` records the checks, all before/after source hashes and the isolated candidate tree. It also confirms all nine production source hashes remain unchanged.
- Four changed host translation units pass `clang++-20 -fsyntax-only` using the existing build's compile flags, with candidate headers supplied from an isolated temporary tree and PCH disabled. No object/library output was produced. Full argv and results are in `native_sampling64_host_syntax.json`. The nanobind unit initially needed its unchanged local header directory added to the copied-tree include search; that setup failure and successful retry are preserved. **Device kernels have not been compiled.**
- Canonical wiring parses as Python AST. The model-owned probe passes repository pre-commit hooks (`sampling64_probe_precommit.log`). Both `sampling32_probe_cpu.json` and `sampling64_probe_cpu.json` pass all **24 CPU fixture checks**. Those results explicitly have zero captures and no hardware token artifact.

The parent can run this sequence after the current server releases its devices, using the lane environment and device-health procedure. These commands are prepared, **not executed** here:

```bash
M=models/autoports/qwen_qwen3_6_35b_a3b
D="$M/doc/vllm_integration/sampling_debug"
timeout 600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_sampling64_probe \
  --draw-width 32 --output "$M/readiness_vllm/sampling64_probe/baseline32.json"
git apply --check "$D/native_sampling64_candidate.patch"
git apply "$D/native_sampling64_candidate.patch"
timeout 1800 cmake --build build --target ttnn ttnncpp --parallel 2
timeout 600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_sampling64_probe \
  --draw-width 32 --output "$M/readiness_vllm/sampling64_probe/candidate32.json"
timeout 600 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_sampling64_probe \
  --draw-width 64 --output "$M/readiness_vllm/sampling64_probe/candidate64.json"
```

Require matching per-fixture `token_digest` values in baseline32 and candidate32, as well as successful internal checks. Both `ttnn` and `ttnncpp` are confirmed build targets. The saved host syntax-check commands compile only isolated source copies; they are not substitutes for the actual rebuild above. The canonical-wiring patch remains a later step, after the native result is understood. A successful native probe still does not establish cross-chip CCL, full-model accuracy, P2 latency, or a new serving capability by itself.

Patch/probe preparation ended **2026-11-17T02:58:51.623948+00:00**, approximately **15.9 active investigation minutes** since 02:43:00. Together with the earlier recorded 5.1-minute design session, the documented active total is approximately **21.0 minutes**, not two hours. No device-wait time was counted. See `p2_patch_preparation_time.json`.


TP4 follow-up and serving-policy omission: see `P2_MESH_SAMPLER_PROBE.md`. Prepared weight-free canonical probe, extended the unapplied canonical proposal with all three serving validator bounds, and passed CPU fixtures/policy checks. This source-only session added approximately 10.0 active minutes; cumulative tracked total 31.0 minutes, not two hours. No hardware wait counted; exact timestamps in `p2_mesh_probe_preparation_time.json`.
