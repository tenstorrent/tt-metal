# Runnable long-prefill geometry matrix and next normalization family

CPU/source preparation only. [long_prefill_geometry27.json](long_prefill_geometry27.json) is a runnable list for `tests.run_multichip_search`: **31 jobs**, comprising 14 long rows and17 short regression rows. All referenced `.pt` files exist. [CPU validation](long_prefill_geometry27_cpu.json) records policy/source hashes and checks. No device run, build, or production edit was performed.

## Controls and coverage

Controls copy the matching **regression/minimal** policy from `norm_whole26.json` exactly, for each layer/shape, with the same frozen TTNN reference. Every job retains `distributed_decode_norm="minimal"`; GDN retains `recurrent_column_grain=1`. Full attention retains its current compact-QK norm/RoPE, cache, SDPA and packed settings. Six shared controls cover L0/L3 at B32/S513, B1/S512 and B32/S33. Each candidate names its exact `comparison_to` control; the runner ignores that explanatory field but records it in provenance.

All13 configurations from matrix26 are represented at B32/S513. J0 maps to the exact current GDN control; it is not run again as an explicitly equivalent policy. The second layer adds its own control, hence14 long jobs. Deep merges preserve all unrelated control policy fields. Joint changes use supported `packed_correction_options` constructor arguments and `prefill_config` keys. Full-attention changes use `prefill_projection_configs.packed_qkvg` plus the existing global `matmul_row_cap`; no unsupported per-key row-cap keyword is invented. Decode precision stays FP32 for joint projection. Every direct-window configuration has FP32 joint output; BF16 variants retain the legal existing combine/concat route.

The17 short rows include four controls and13 representative candidates. B1/S512 covers new8×10 geometry, BF16-pack/FP32-DST, seven-column FP32/BF16-DST cross, L1 admission and QKVG geometry. B32/S33 additionally exercises windowed c640 on one1056-row flat chunk (640+416), different final projection chunks, both QKVG caps, and changed precision. There is no32-row temporal tail for S33; c1280 processes all1056 rows in one call. Caps1280/2560/5120 have equivalent no-split behavior on the smaller shapes unless grid/precision/L1 differs; avoid redundant tests there. These are regression selections, not exhaustive final stress coverage.

The manifest orders the long comparison first, followed by selected regression rows. The runner stops at a failed gate; preserve that artifact, repair/adapt or remove only the unsupported failing configuration into a documented follow-up manifest, then resume remaining work. A stopped runner does not reject unrun rows. Existing13-configuration geometry checks cover9,104 internal chunks across B1..32/S1,3,31,32,33,127,511,512,513; this proves descriptor arithmetic, not native/L1 legality.

Parent-owned serialized execution command, **not run here**:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_search models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/long_prefill_geometry27.json --resume
```

The runner records fresh source/environment provenance and disables watcher/profiler/allocation tracking for its performance run. Separate watcher/ownership checks remain required before retention. Existing control timings are not substituted for these newly scheduled controls.

## Gather-first normalization: bounded next path

Current `MultichipDecoder._norm` receives width1280 residual shards and, on B32 prefill, executes local pre-statistics → statistics AG → postnorm → full activation AG. That final width5120 gathered activation is already the projection contract. `gather_norm=True` selects activation AG first, then `OptimizedDecoder._norm` with the full actual FP32 affine weight. It keeps the inter-layer residual width1280; no gather/restore of that residual is needed. The decode-native branch is checked **before** `gather_norm`, so selected minimal decode normalization remains active.

A bounded next comparison can use existing policy hooks without source changes:

1. G0: exact current policy/control, both kinds B32/S513.
2. G1: `gather_norm=True, flat_gather_prefill=True` (the explicit flat-admission hook is required to preserve topology). With the current `explicit_sum_norm=True`, this uses FP32 square/reduce/scale/rsqrt/affine then final BF16. It is a valid numerical/placement control but material intermediate traffic may dominate.
3. G2: `gather_norm=True, flat_gather_prefill=True, sfpu_prefill_norm=True`. The existing `prefill_hidden_rmsnorm_sfpu` (`optimized_decoder.py:2493`) precedes explicit-sum routing, accepts logical `[B,S,5120]` and FP32 gamma, owns the final BF16 DRAM result, and preserves per-batch padded-row addressing. It fuses those local intermediates and consumes the gathered activation directly. Measure the full AG→norm→next projection boundary and whole layer, with no immediate restore. A slow G1 cannot reject G2.

For G1/G2, first establish B3/S33 and B32/S513 normal and recorded-text accuracy, both actual input/post gammas; then B1/S512 (existing B1 norm placement may already gather) and B32/S33 regression. Include changed inputs and queued trace replay. Public unaligned lengths remain valid; poison dead rows independently and verify valid rows finite and accurate. Preserve the layer's accepted aggregate/per-user/contribution/state gates. These normalization proposals are separate from geometry27; do not combine both before attributing their effects.

### Existing fused distributed native limitations

`fused_rms_minimal` validates logical shape `(1,1,M,N)`, **M≤32**, and padded height32 (`rms_allgather_device_operation.cpp:47–53,164`). The current `FusedDecodeNormOwner` additionally owns only two physicalB32 buffers and phase-paired views. Calling this owner on M16384 is unsupported; splitting into512 native calls is a different high-launch family and is not a meaningful direct long replacement.

DiT fused distributed RMSNorm is **not inherently limited to32 rows**: `dit_fused_distributed_rmsnorm_program_factory.cpp:300–355` sizes workers and forwarder rounds from physical tile rows / explicit `row_capacity`; native validation checks actual rows against capacity (`...device_operation.cpp:117`). Its current model wrapper, however, fixes capacity32 and two decode phases. Extending that wrapper by changing a logical view would be invalid. If G2 loses, the concrete independent distributed candidate is a **separate prefill owner** using actual DRAM `[1,1,16384,1280]`, FP32 local gamma, `per_head_norm=False`, and native-created row-capacity16384 stats, then the existing activation AG. Keep separate input/post semaphore/stat sets, correct cache-hit bindings and terminal-drain protocol; tailM32 uses a separately warmed legal descriptor or a verified smaller logical view at declared capacity. Native page size/round count and owner memory must be measured, not bounded by the decode wrapper's64KiB constant. Compare1024/4096 row chunks only if full-capacity resource or performance evidence requires adaptation; include all copies, launches and final placement. This is applicable future work, not a rejection based on the minimal kernel's row limit.

### Independent numerical oracle requirement

The norm comparison must not derive its golden output by running the candidate or by reusing its reduced statistics. Save each rank's **actual BF16 input shard**, reconstructed in mesh hidden-coordinate order, and the actual uploaded FP32 full gamma (including any already-applied model convention). In CPU FP64, compute `x * rsqrt(sum(x*x)/5120 + epsilon) * gamma`, round only the final expected output to BF16, and report per-row/per-rank PCC and absolute/bit diagnostics. Include near-zero, heterogeneous scales and real checkpoint activations; valid-row checks precede metrics and unequal constants stay failures. Save statistics/output artifacts before asserting. This oracle diagnoses norm arithmetic independently; the accepted frozen TTNN full-layer reference remains the mandatory layer gate, and existing HF-contract checks remain separate and unchanged. Mathematical FP64 and exact replay are distinct tests: changed reduction trees need unchanged numerical gates, whereas replay/rebind ownership requires exact equality to that path's eager result. No relaxed thresholds or reuse of candidate outputs as a reference.


Correction recorded with movement28: S33 projection geometry is M1056; the previous1024-row description was a prose error. The prior9,104-chunk CPU proof used `logical=b*s` and already covered B32/S33; no geometry-proof result, policy or reference changed. Gather-first proposals require the now-applied flat-admission hook; old unflattened jobs are diagnostics only.
