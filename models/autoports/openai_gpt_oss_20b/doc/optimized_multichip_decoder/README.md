# GPT-OSS 20B optimized multichip decoder

Status: complete. Final independent `$stage-review`: `clean-pass`.

This stage optimizes the completed fixed `MeshShape(1,4)` Blackhole P300c
decoder in `tt/multichip_decoder.py`.  It does not include full-model or
vLLM work.  The accepted starting point is commit `753f7207969` and the
frozen evidence under `../multichip_decoder/`.

## Starting contract

- TP4 packed attention, EP4 whole routed experts, and a 1D fabric ring.
- Exactly gate-selected top-4 expert execution through three
  `ttnn.sparse_matmul` operations; no dense all-expert fallback.
- Replicated BF16 `[1,1,S,2880]` public and inter-layer residual.
- Per-rank BFP8 paged K/V cache with two local KV heads.
- Arbitrary positive logical sequence lengths, with internal
  padding/chunking/slicing.
- Advertised context 131072.

The accepted S=128 baseline is 21.7227/1.0581 ms prefill/traced decode for
sliding layer 12 and 25.2078/1.0417 ms for full-attention layer 13.  At
S=17 it is 5.0955/0.6317 ms and 5.1102/0.6347 ms respectively.  These are
the before numbers; final numbers were recaptured from the selected default
path and are reported below.

## Initial operation-topology audit

The audit combines `tt/multichip_decoder.py` with the accepted
`../multichip_decoder/perf/decode_final_frozen.csv` and
`prefill_final_frozen.csv`.

| Region | Current measured sequence / contract | Material cost or defect | Candidate family and constraints | Initial action |
| --- | --- | --- | --- | --- |
| input residual and attention norm | replicated BF16 DRAM → 10-core sharded RMSNorm → 10-core width reshard | repeated residual-to-norm movement | carry a compatible L1 residual, or prove the replicated boundary wins through the next norm/QKV | remeasure as a coherent layer-to-layer contract |
| attention projections | one packed TP4 QKV matmul from the shared normalized activation | no repeated Q/K/V matmuls; already packed | advisor 1D layout, DRAM-sharded weight matmul, BFP4 attention weights, LoFi/HiFi2/HiFi4, larger legal K blocks | sweep on real weights and trace |
| head/cache/attention | fused QKV-head creation → fused RoPE → paged update → paged SDPA below position 127; device-only sink-correct manual attention above it | manual long path has material FP32 matmuls, but native control is incorrect | retain composite ops; tune only under the sink/cache correctness contract | compare explicit configs and dtype-compatible paths |
| output projection | local TP4 O matmul → sharded-to-interleaved L1 → generic ring all-reduce | the conversion is avoidable; generic `AllBroadcastDeviceOperation` rows are about 35% of decode | persistent-buffer minimal width-sharded L1 all-reduce; fused AG+local-O; padded RS+AG; fused matmul+RS when safe | isolate the minimal all-reduce first; retain earlier adapted-family controls |
| attention residual | replicated BF16 add in DRAM | restores replicated state immediately | carry fractured residual into distributed norm/router/next packed QKV | existing adapted candidate is the comparison floor; seek lower-cost variants rather than immediate restore |
| post-attention norm/router | replicated BF16 → sharded RMSNorm → DRAM; FP32 router linear/softmax/top-k | router matmul is small but precision-sensitive | L1 router input, row-sharded router plus reduction, BF16/BFP8 communication only if real routing remains accepted | measure with downstream selected-expert agreement |
| expert projections | separate gate and up sparse matmuls from one token activation → fused SwiGLU helper → sparse down | active sparse matmuls are about 32% of decode and 42% of prefill | packed gate/up if sparse op supports legal packing/split; role-specific geometry, activation dtype, fidelity, L1 intermediates, static `nnz` only if exact | sweep each sparse role; never replace with dense all-expert execution |
| expert output | local active-expert weighted sum → generic ring all-reduce → replicated residual add | second dominant generic `AllBroadcast` | same persistent minimal all-reduce family; BFP8 CCL payload; carried EP residual | measure whole layer and next-layer compatibility |
| persistent state | K/V and fused-AG buffers persist; default all-reduces do not provide a model-owned output/intermediate buffer | repeated CCL allocation/program path | persistent minimal AR buffer/semaphore and persistent RS/AG buffers where legal | implement and compare traced warmed decode |

Repeated same-input dense projections are already packed for QKV.  The
remaining repeated same-input projection is sparse expert gate/up; it must
be assessed with the sparse routing and split/activation costs included.
The current path uses fused QKV head creation, RoPE, paged cache operations,
SDPA, RMSNorm, and SwiGLU helpers.  No primitive attention or dense
all-expert rewrite is acceptable.

The lower-movement family will not be judged only after restoring the old
replicated residual.  The existing carried residual control already reaches
distributed RMSNorm, row-sharded router, and a persistent fused gather into
the next packed QKV; any new carried layout will be measured through an
equivalent next-layer boundary.

## Evidence

Evidence generated by this pass will live in:

- `logs/` for correctness, timing, candidate, watcher, and health logs;
- `tracy/` for raw compact operation CSVs and `tt-perf-report` tables; and
- `shard_advise/` for this pass's `report.json` and `final_ir.mlir`.

## Selected default

The final default keeps the public/inter-layer residual replicated BF16
TILE DRAM. Decode uses two direct
`ttnn.experimental.all_reduce_async` operations on width-sharded L1
partials, each with a decoder-owned persistent L1 output buffer and global
semaphore. This removes the two pre-collective sharded-to-interleaved
conversions from the hot path. The ring, TP4 attention, and EP4
gate-selected top-4 expert execution are unchanged.

The selected precision and geometry policy is:

- BF16 residuals, activations, norms, router, CCL payloads, and paged
  prefill/long-decode QKV/O weights;
- BFP8_B short-decode QKV/O copies below position 127, BFP8_B expert
  weights, and BFP8_B paged K/V;
- HiFi2 short attention projections; the established HiFi2 sliding and
  HiFi4 full long-decode policy at and above position 127;
- LoFi sparse expert matmuls with BF16 activations;
- 90 sparse cores, gate/up `in0_block_w=45`, down
  `in0_block_w=90`, and one-tile output subblocks; and
- sparse expert outputs in L1 for decode and internally padded prefill
  M<=32, with DRAM output for M>=64; and
- dynamic sparse `nnz`, because exact profiler-input routing produces
  rank-local decode counts `[1,3,0,0]`/`[2,2,0,0]` at S=17 and
  `[1,1,0,2]`/`[2,1,0,1]` at S=128 for layers 12/13.

Two 737,280-byte L1 collective buffers plus semaphore words add 1,475,440
bytes per device. Short-decode BFP8_B QKV/O copies add 7,050,240 DRAM
bytes per device. Both are fixed-size; paged-cache capacity and the
131072-token context limit are unchanged. Exact selected-API knobs and
limitations are in `logs/minimal_all_reduce_config.json`: Ring,
`cluster_axis=1`, one link, 100 workers observed, modulo-2 persistent
buffers/semaphores, and no exposed `chunks_per_sync` or
`num_buffers_per_channel` knob on this all-reduce overload.

## Final before/after performance

All values below are from the 1x4 target mesh, after warmup. Prefill uses
20 measured iterations and decode uses 500 replays of a captured trace.
The before values are the accepted completed-multichip baseline; the after
values are from the final default after the long-decode BF16 correction and
tile-bounded prefill L1-output policy.

| Sequence | Layer kind | Prefill before / after (ms) | Change | Traced decode before / after (ms) | Improvement |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | sliding, layer 12 | 5.0955 / 4.1403 | 18.75% | 0.6317 / 0.5909 | 6.46% |
| 17 | full, layer 13 | 5.1102 / 4.1539 | 18.71% | 0.6347 / 0.5224 | 17.70% |
| 128 | sliding, layer 12 | 21.7227 / 21.8127 | -0.41% | 1.0581 / 1.0390 | 1.80% |
| 128 | full, layer 13 | 25.2078 / 25.2924 | -0.34% | 1.0417 / 1.0237 | 1.73% |

The S=17 prefill improvement comes from keeping one-tile sparse expert
outputs in L1. Direct M=64 and M=128 L1 outputs were retried and hit hard
94,371,840-byte and 188,743,680-byte allocation failures, respectively.
The final path therefore selects L1 through M=32 and DRAM above it; S=128
remains within sub-half-percent noise of the incoming path. An adapted
prefill async-all-reduce retry also
measured no coherent benefit: S=17 was 5.1183/5.1312 ms versus selected
5.1154/5.1311 ms, while S=128 was 21.8239/25.3024 ms versus
21.8172/25.3053 ms. It was removed rather than adding persistent prefill
state for noise.

Authoritative final JSON:

- `logs/final_default_after_review2_layer12_seq17.json`
- `logs/final_default_after_review2_layer13_seq17.json`
- `logs/final_default_after_review2_layer12_seq128.json`
- `logs/final_default_after_review2_layer13_seq128.json`

## Correctness, context, trace, and runtime gates

Every meaningful layer kind passes the accepted real-weight multichip
thresholds on the selected default:

| Coverage | Sliding layer 12 | Full layer 13 |
| --- | --- | --- |
| S=128 prefill output PCC | 0.998366 | 0.999240 |
| S=128 following decode output PCC | 0.997958 | 0.998651 |
| S=129 prefill / following decode PCC | 0.999307 / 0.997565 | 0.997231 / 0.998090 |
| S=2048 prefill / following decode PCC | 0.998368 / 0.997833 | 0.997543 / 0.997504 |
| advertised-context final output PCC | 0.997990 | 0.999226 |
| advertised-context selected-expert agreement | exact top-4 | exact top-4 |

The S=128/129/2048 attention PCC is at least 0.999994. Positions 127-131
pass the real-weight boundary sweep for both layer kinds. A trial using
BFP8_B attention projections in long decode changed one near-tied sliding
router decision at position 130; the final path therefore owns the
short/long switch and uses BF16 QKV/O weights at position 127 and above.
This is the path used by the final performance and endpoint artifacts.

The public API still accepts non-aligned lengths. S=17, S=129, and S=2048
exercise decoder-owned internal padding, chunking, masking, and slicing.
Reverse page tables cross logical and physical page boundaries. One
captured short trace accepts mutable positions 17-19; long trace banks
match eager at positions 128-131 and 191-193. The 131072-token endpoint
allocates `[2048,2,64,64]` local K and V caches on every rank and updates
reverse-mapped physical page 0, offset 63.

The static runtime audit proves the selected branch executes exactly three
sparse matmuls, no PyTorch/CPU fallback, no dense all-expert path, and two
minimal async all-reduces. Watcher ran separately from profiling with ETH
inspection disabled, completed four representative final-path cases with
no watcher errors/asserts/hangs, and left all four P300c devices healthy.
An additional real-weight S=17 hardware run for both layer kinds passed with
`throw_exception_on_fallback=true`, so the throwing configuration was
exercised on the selected decoder rather than only by source inspection.
The ETH exclusion is an inherited platform limit, not a model waiver:
`../multichip_decoder/logs/watcher_eth_attempt_final_frozen.log` fails
before model execution because the instrumented ACTIVE_ETH program is
27,920 bytes versus a 25,600-byte kernel-config buffer. The incoming
stage's `stage_review_round2.md` accepted worker/Tensix watcher plus
CCL-heavy trace and final fabric/device health as the compensating control.
This pass reran the longer CCL/cache/boundary cases on the final source.

Primary gates:

- `logs/final_default_full_suite_after_review2.junit.xml` (34 passed,
  26 expected opt-in skips; every material skipped candidate has a
  dedicated enabled-run artifact)
- `logs/final_precision_all_lengths_after_long_bf16_fix.junit.xml`
- `logs/final_precision_after_l1_policy.junit.xml`
- `logs/final_boundary_after_long_bf16_fix.junit.xml`
- `logs/final_endpoint_after_long_bf16_fix.junit.xml`
- `logs/final_endpoint_after_review2.junit.xml`
- `logs/final_watcher_after_review2.junit.xml` and
  `logs/final_watcher_after_review2.log`
- `logs/final_static_contract_audit_after_fix.junit.xml`
- `logs/final_throw_on_fallback_hardware.junit.xml`
- `logs/final_hardware_health_after_review2.log`

## Operation-topology audit resolution

| Family | Action and evidence | Decision |
| --- | --- | --- |
| repeated same-input matmuls / packed projections | QKV remains one packed TP4 matmul. Sparse gate/up packing was implemented as one packed sparse matmul plus on-device bias/typecast/stride-2 split; PCC passed, but layer-12 decode regressed from about 0.591 to 0.853 ms. | keep packed QKV; keep separate gate/up sparse projections |
| material decode collectives | Replaced interleaved generic all-reduce with direct width-sharded L1 async all-reduce, two persistent buffers, and two semaphores. BF16 payload beat an adapted BFP8 payload including casts. | selected |
| residual/layout movement | Two coherent families avoided an immediate restore. The expert-output carried family kept H2944/4=736 local through distributed RMSNorm, row-sharded router, and persistent fused AG into real packed QKV: 0.6118/0.6086 ms versus 0.4773/0.4740 ms at 500 replays, PCC >=0.999898 and exact top-4. The review-driven attention family started with fused local O+RS, sharded the incoming residual, carried it through distributed post-attention RMSNorm and row-sharded router, gathered normalized activations only where the current EP4 sparse gate/up contract requires H2880, ran the remaining gate-selected sparse MLP, and gathered the residual only for the layer-output add. It passed exact top-4 and final PCC 0.999450/0.999677, but measured 0.6599/0.7372 ms versus 0.4488/0.5263 ms. | replicated BF16 inter-layer residual is faster overall |
| fused CCL plus matmul | Fused attended-all-gather plus local O was adapted to the selected BFP8 payload and persistent buffer; PCC was about 0.999924, but the boundary cost was 0.1932/0.1937 ms versus 0.0661/0.0661 ms. The exact decode M=1 O-projection shape was also adapted to `minimal_matmul_strided_reduce_scatter_async`: the first 4x1 grid failed its documented >=2x2 constraint, the 4x2 retry passed per-rank PCC >=0.99969, and whole-decoder PCC passed. It measured 0.7263/0.6582 ms versus 0.5911/0.5226 ms. The later deferred-gather full-boundary trial above removes immediate restoration as a confounder and remains slower on both layers. The upstream issue-46181 source gate applies only to prefill M-tiles=32 and is not used to reject decode. | reject both measured fused families |
| collective placement | Decode collectives stay after O and active-expert reduction. Carrying either fractured residual across the next norm/router/QKV was slower; adding async collectives to unchanged prefill was neutral. | selected placement |
| persistent buffers | Model-owned L1 all-reduce buffers/semaphores remove hot-path conversion/allocation. Carried RS/AG and fused AG candidates also used persistent buffers in their coherent comparisons. | selected for decode AR only |
| activation sharding / DRAM-sharded decode matmul | Width-sharded L1 QKV input/O partial and CCL payloads remain selected. Rank-local QKV `[2880,1280]` and O `[1024,2880]` were adapted to DRAM-sharded matmul and passed layer-12 PCC, but decoded at 0.6622/0.5935 ms for layers 12/13 versus 0.5911/0.5226 ms. | reject DRAM-sharded weights |
| advisor sharding | The compiler advisor QKV 1D layout passed PCC but measured about 0.651/0.653 ms. Its O recommendation matched the existing 90-core/in0=8 geometry. | reject QKV proposal; confirm O baseline |
| sparse geometry and memory | Both layer kinds were swept for compact 45-core/subblock-2, 90-core K90, swapped role K blocks, L1/DRAM decode outputs, and S=17/S=128 prefill. Decode keeps gate/up K45 and down K90 on 90 cores with L1 outputs. Prefill keeps K45/90 cores and uses L1 only for M<=32; M=64 and M=128 have physical L1 OOM evidence. `logs/sparse_policy_sweep.json` has the matrix. | selected role-specific geometry and tile-bounded memory |
| activation/CCL dtype | BFP8 CCL passed but regressed to 0.6228 ms from the BF16 selected family because of casts. BFP8 expert activations failed full-layer PCC. | BF16 |
| attention precision/fidelity | The original BFP4/LoFi family failed prefill PCC. A second decode-only BFP4 trial preserved BF16 prefill/long weights and trace/cache exactness, but short attention PCC was only 0.969133/0.968371; layer 12 was also slower at 0.6008 ms. BFP8/LoFi and BFP8/HiFi2 passed; HiFi2 was selected for short decode. Long BFP8 was retried through the real boundary and rejected on routing, so long decode stays BF16. | mixed short/long policy selected |
| expert precision/fidelity | Whole, gate/up-only, and down-only BFP4 variants failed real full-layer PCC; BFP8 activations also failed. HiFi2 experts passed but were slower than LoFi. | BFP8 weights, BF16 activations, LoFi |
| fixed sparse `nnz` | Real rank-local active counts vary and include zero. | dynamic gate-selected active experts |

No applicable optimization is deferred. Rejected material families have
hardware before/after evidence. The fused matmul+RS decode primitive was
shape- and grid-adapted through its first API failure and measured in the
whole decoder; the separate upstream prefill race is retained only as
scope-specific provenance.

## `tt-perf-report` findings

The original compact profiler capture is one warmed S=17 prefill plus one
traced decode replay per layer. After the tile-bounded L1 change, S=17 was
recaptured on the final default; a second final-default profile covers
S=128 prefill and long-position traced decode for both layer kinds.
Profiling and watcher were separate. Raw CSV paths, SHA-256 hashes,
signposts, compact tables, and retention are recorded in
`logs/final_short_profile_provenance.json` and
`logs/final_long_profile_provenance.json`; `logs/profile_provenance.json`
retains the earlier optimization-loop capture.

| Final decode region | Layer 12 | Layer 13 | Consequence |
| --- | ---: | ---: | --- |
| two minimal async all-reduces | 203.56 us, 28.97% | 141.59 us, 24.67% | still dominant, but layer-12 per-replay CCL fell about 32% from the frozen generic-AR profile |
| three active sparse matmuls | 246.20 us, 35.04% | 179.96 us, 31.36% | geometry/fidelity/dtype/packing families were swept |
| FP32 router matmul | 23.17 us, 3.30% | 23.50 us, 4.09% | router precision retained because boundary routes are near-tied |
| packed QKV matmul | 13.47 us, 1.92% | 13.73 us, 2.39% | advisor and DRAM-sharded alternatives were slower |
| local O matmul | 9.63 us, 1.37% | 9.82 us, 1.71% | fused AG+O was slower |

The final reports use EP-faithful sparse accounting. One-token decode uses
the exact critical-rank local K (S=17: 3/2; S=128: 2/2 for layers 12/13).
Prefill activity varies by token and rank, which the report's scalar
`--active-experts` option cannot represent; those tables intentionally
omit sparse DRAM/FLOP estimates and pair with exact activity histograms in
`logs/profile_ep_activity_layer*_seq*.json`. Remaining advice maps to
families actually tried here: DRAM-sharded decode attention, L1 sparse
outputs, legal larger subblocks/K blocks, and attention/expert fidelity
changes. Prefill is still dominated by three sparse matmuls and two
generic broadcasts (17.86/17.31%). The async prefill retry showed no
whole-layer gain, so the unchanged synchronous prefill path remains the
default. The report's fidelity advice was tried: HiFi2 experts were slower,
while reduced attention fidelity was accepted only where real routing
remained stable.

The authoritative final-default S=17 tables are under
`logs/final_short_profile/`; S=128 tables and CSVs are under
`logs/final_long_profile/`. The earlier compact tables under `tracy/` are
optimization-loop evidence only. `logs/performance_accounting.json`
reconciles the exact rank-local decode floors (0.1701935/0.118556 ms at
S=17 and 0.130706/0.130706 ms at S=128 for layers 12/13),
uninstrumented latency, merged device rows, overlapping asynchronous
intervals, op gaps, and signpost wall.

## Inter-layer contract for full-model bringup

Each layer accepts and returns replicated BF16 TILE DRAM
`[1,1,logical_sequence,2880]`. Internal alignment never escapes the
decoder. There is no gather, reshard, reduce-scatter, or all-reduce between
layers. Consumers should pass the returned tensor directly to the next
decoder and must not insert a collective. The measured carried-local
alternative is slower through the next layer's norm/router/QKV boundary,
so preserving this replicated contract is intentional rather than an
unexamined fallback.

## Reproduction commands

```bash
# Final 20-iteration prefill / 500-replay traced decode
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PERF_SEQ=17 \
  MULTICHIP_DECODER_PREFILL_REPEATS=20 \
  MULTICHIP_DECODER_TRACE_REPLAYS=500 \
  pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf

RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PERF_SEQ=128 \
  MULTICHIP_DECODER_PREFILL_REPEATS=20 \
  MULTICHIP_DECODER_TRACE_REPLAYS=500 \
  pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf

# Real-weight lengths, boundary, and advertised endpoint
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_precision_lengths_match_current_optimized
pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_boundary_positions_match_current_optimized
RUN_MULTICHIP_CONTEXT=1 pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_full_context_cache_allocation_and_last_page_update

# Watcher is deliberately separate from profiler
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  pytest -q -s \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_warmed_long_position_trace_replay_matches_eager \
  models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_boundary_positions_match_current_optimized
```

## Limitations

This is a fixed 1x4 Blackhole P300c decoder optimization. It intentionally
does not begin full-model, generator, or vLLM work. The host emits a
non-fatal low-`/dev/shm` warning and identifies motherboard `B850M-C`
through its PCI fallback; neither affected the serialized hardware gates.
