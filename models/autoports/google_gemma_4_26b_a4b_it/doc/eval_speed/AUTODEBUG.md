# AutoDebug: Gemma4 BFP8 control and long-context quality

## Conclusion

**The configurable-weight BFP8 experiment is genuine: no active configurable
weight was found to be widened from already-quantized BFP4.** The constructor
starts the imported layer at BF16, then creates the selected decode weights
from those tensors or directly from checkpoint values
(`A/multichip_decoder.py:846–856,892–902,994–1038`).

**It does not test removal of prefill BFP4 or low-fidelity arithmetic.** Separate
EP4 prefill weights still use BFP4 for every expert down projection and for
full-attention expert gate/up projections, with LoFi computation
(`A/multichip_decoder.py:534–580,1074–1079`). Full-attention prefill after the
first 1,024-token chunk also reads BFP8 K/V from cache, unlike the short
readiness prefill (`A/optimized_decoder.py:1387–1392,1425–1445`). These are
confirmed limits of the control, **not a demonstrated cause of Astropy's
repetition or destructive over-editing**. No concrete cache/position defect
explaining that behavior was established in the inspected C1 path.

The smallest additional weight control is to reload just those fixed EP4
prefill BFP4 groups as BFP8 from checkpoint/BF16 source, keep the existing
candidate's head/decode weights and all other settings fixed, and compare
bounded numerical outputs on an unchanged recorded long prompt. This requires
a diagnostic implementation change; the current JSON schema intentionally
rejects changing fixed fields (`A/precision_policy.py:110–113,140`). No source
changes or model/device execution were performed.

## Scope and evidence

Path abbreviations below are repository-relative:

- `A/` = `models/autoports/google_gemma_4_26b_a4b_it/tt/`
- `D/` = `models/demos/gemma4/tt/`
- `T/` = `models/autoports/google_gemma_4_26b_a4b_it/tests/`

Read `README.md` and `suite_20261002.md`. Inspected local source, JSON artifacts,
and tests only; no model/device commands, external endpoints, or source edits.
Final inspected HEAD is `5d381d145aab0a136b36c7a776acf4e4f2757682`; both focus trees
have no diff against image revision
`c9ec3469f1b875e7e5e505660c4421e5126e8dad`. All 12 autoport runtime source
hashes recorded in the retained `weight_control/readiness.json` match the
inspected files, despite that artifact recording an earlier repository commit.
The inspected matmul/SDPA/cache implementation directories also have no diff
against the image revision.

Direct artifact observations: candidate policy SHA256 is
`46389c08f1c99f068669f66e902cc34daf00e54e7c7d8014dc1539b3dd1af954`;
readiness records prefill top1 .96 and decode top1 .98 over 100 predictions.
The retained Astropy 1,200-second request telemetry contains 38 response
records with prompt lengths 1,124 through 42,452 tokens. The documented
correct-patch-then-corruption outcome is in `suite_20261002.md:81–94`.
These observations establish neither a numerical root cause nor that more
generation time should improve the result.

Local telemetry source:
`/home/mvasiljevic/gemma4-eval-speed-evidence/local_astropy_bfp8_submit_1200_seed9472/local_astropy_bfp8_submit_1200_seed9472_requests.jsonl`;
lines 2, 32, and 76 record the 1,124-, 18,772-, and 42,452-token examples below.

## 1. Weight provenance

The 93 changes count **serialized policy fields**, not 93 allocated tensors:
one head field, eight layer-type defaults, and 84 overrides. Resolving all 30
layers yields 177 changed configurable groups; all 181 configurable groups
(six per layer plus head) resolve to BFP8. Fixed policy, activations, KV, CCL, and
fidelity are unchanged. The preparation script explicitly skips `fixed`
(`../../tools/prepare_eval_weight_control.py:20–30`).

| Configurable weight | Actual BFP8 construction path |
| --- | --- |
| QKV, all layers | Checkpoint Q/K/V packed and uploaded BF16 by `D/attention/weights.py:75–103,177–185`; BF16 → BFP8 at `A/multichip_decoder.py:892–902`. The selected-policy BFP4 cast is skipped when BFP8 is requested. |
| Attention output, all layers | Checkpoint BF16 at `D/attention/weights.py:120–123,192–200`; BF16 → BFP8 at `A/multichip_decoder.py:913–952`. |
| Full-attention expert gate/up; all expert down | Imported BF16 source → concatenation/alias → selected BFP8. `D/layer.py:74–83,157–165`, `D/moe.py:43–51`, `D/experts/__init__.py:46–52`, `D/experts/weights.py:110–135`, `A/fused_decoder.py:259–262`, `A/optimized_decoder.py:104–112`. |
| Sliding expert gate/up | Additionally repacked from host checkpoint and directly uploaded at selected dtype: `A/multichip_decoder.py:1019–1038`. |
| Shared MLP gate/up and down | Original host checkpoint tensors packed and directly uploaded at selected dtype: `A/multichip_decoder.py:330–354`. |
| LM head | Original checkpoint embedding transpose directly uploaded at selected dtype: `A/model.py:92–98,159–169`. Used by both prefill and decode. |

The warning about widening BFP4 at `A/multichip_decoder.py:1019–1021` does
**not** prove that full-attention experts currently do so: the full constructor
chain starts them at BF16. `precision_summary()` checks final tensor dtypes and
bound configs, not quantization lineage (`A/multichip_decoder.py:1104–1176`);
the source trace above supplies that missing evidence. No converted-weight
cache contamination is supported: `tensor_cache_path=None` is explicit at
`:853`, and the imported cache-name helper returns `None` for that case
(`models/demos/gemma4/utils/general_utils.py:9–16`).

## 2. What long prefill actually executes

The adapter rejects nonzero scheduler prefill offsets
(`A/generator_vllm.py:247–248`), but the model still internally chunks every
layer into 1,024-token pieces (`A/multichip_decoder.py:1229–1282`). The
generator slices to the logical prompt length (`A/generator.py:447–454`).
Each full-attention tail rounds to 32 rows; each noninitial sliding tail
shorter than the 1,024-token window pads to 1,024. Cache writes cover only
`ceil(valid/32)*32`, and layer outputs are unpadded before the final real-token
logits (`A/optimized_decoder.py:1385–1392`; `A/multichip_decoder.py:1283–1297`;
`A/model.py:230–232`).

| Recorded logical context | Internal chunks | Last valid rows | Full-attention last rows | Sliding last rows |
| ---: | ---: | ---: | ---: | ---: |
| 1,124 (Astropy first request) | 2 | 100 | 128 | 1,024 |
| 1,989 | 2 | 965 | 992 | 1,024 |
| 12,028 | 12 | 764 | 768 | 1,024 |
| 18,772 (Astropy) | 19 | 340 | 352 | 1,024 |
| 21,018 | 21 | 538 | 544 | 1,024 |
| 32,155 | 32 | 411 | 416 | 1,024 |
| 42,452 (Astropy final recorded request) | 42 | 468 | 480 | 1,024 |
| 48,600 | 48 | 472 | 480 | 1,024 |

All these contexts execute the following unchanged per-layer prefill policy.
There are 25 sliding layers and five full layers (indices 5, 11, 17, 23, 29;
`models/demos/gemma4/configs/gemma-4-26B-A4B-it/config.json:36–90`).

| Stage | Sliding layers | Full-attention layers | Source |
| --- | --- | --- | --- |
| QKV projection | BFP8 weights, HiFi4, FP32 destination accumulation | Same | `A/multichip_decoder.py:127–154,880–902` |
| Attention output projection | BFP8 weights, LoFi, FP32 destination accumulation | Same | `A/multichip_decoder.py:913–943`; `A/optimized_decoder.py:1262–1275,1322–1335` |
| EP4 expert gate/up | BFP8 weights, LoFi | **BFP4 weights, LoFi** | `A/multichip_decoder.py:563–580` |
| EP4 expert down | **BFP4 weights, LoFi** | **BFP4 weights, LoFi** | Same; FP32 destination accumulation disabled at `A/optimized_decoder.py:60–65` |
| Shared MLP | BF16 weights/activations; default **HiFi2**, FP32 destination accumulation off | Same | `A/multichip_decoder.py:395–424,1591`; `D/shared_mlp.py:174,201`; C++ default below |
| SDPA | LoFi, BF16 direct K/V plus retained BF16 sliding tail | HiFi2; chunk zero direct BF16 K/V, later chunks **BFP8 paged K/V** | `A/multichip_decoder.py:931–936`; `A/optimized_decoder.py:1242–1252,1376–1445` |
| Prefill router | BF16 weights, HiFi4 projection | Same; the summary's full-layer router LoFi describes **decode** | `A/optimized_decoder.py:917–919`; `A/fused_decoder.py:420–421`; `A/routing_precision.py:58–64` |
| Attention / MoE collective payload | BF16 / **BFP8** | **BFP8** / BF16 | `A/multichip_decoder.py:767–779,1299–1318`; candidate JSON |

For shared MLP, the imported closures pass no compute config or program/grid;
both operands are BF16. `ttnn/cpp/ttnn/operations/matmul/device/
matmul_device_operation.cpp:2804–2815,2848–2860` resolves that case to HiFi2,
BF16 output, FP32 destination accumulation off, packer L1 accumulation on,
and approximate math off. Thus `library_default` is not a HiFi4 guarantee.
Prefill expert activations stay BF16: `_ExpertParallelExperts._chunk` enters
`_active_prefill` before the decode-only activation cast
(`A/multichip_decoder.py:658–670`; `A/optimized_decoder.py:243–266`).

The common head is now BFP8/LoFi in both phases (`A/model.py:94–108,197–209`),
so unchanged per-layer prefill does not mean unchanged prefill logits. Decode
also retains LoFi QKV/output/expert/shared projections, BFP8 expert input and
KV cache, and the existing CCL policy. The experiment is not an all-high-
precision control.

## 3. Cache/position adjudication and test limits

No inspected cache/position mismatch completes a causal chain to the reported
semantic behavior. In particular:

- RoPE tables are generated by the HF per-layer-type rotary implementation
  over the full context extent (`A/model.py:114–120`); no 32K RoPE cutoff is
  introduced there. Long chunks use absolute offsets (`A/multichip_decoder.py:
  1267–1279`). Crossing 32 chunks changes output concatenation grouping, not
  position arithmetic (`:1287–1297`).
- The demo's documented nonchunked-SDPA failure at 32,768 rows
  (`D/attention/prefill.py:531–540`) does not describe the autoport's actual
  inputs: direct SDPA sees at most 1,024 global or 2,048 sliding rows. Later
  global chunks use chunked SDPA (`A/optimized_decoder.py:1111–1119`), whose
  lowering carries absolute offsets into the Q offset and K extent
  (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/
  sdpa_program_factory.cpp:153–154,326–328`). No dropped concatenation group
  was found at the observed 32K crossing.
- Sliding prefill clears its tail for each layer/request, retains the previous
  chunk's BF16 K/V, and drops the temporary tail on the final chunk
  (`A/multichip_decoder.py:1257`; `A/optimized_decoder.py:1396–1423`). This is
  not evidence of cross-request history leakage.
- Full-prefill SDPA accounts for query/K-block rounding and switches to smaller
  blocks at capacity boundaries (`A/optimized_decoder.py:1095–1109`). Replacing
  BFP8 cache with BF16 also selects smaller blocks for wide heads (`:1096–1098`);
  that control therefore changes kernel geometry as well as storage precision.
- Decode positions are uint32/int32 tensors consumed then incremented on
  device (`A/generator.py:497–507,550–553`); the SDPA compute kernel reads the
  current position dynamically (`ttnn/cpp/ttnn/operations/transformer/
  sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp:144–156`). Changed
  page tables and explicit position resets are copied before replay
  (`A/generator.py:614–651`). Greedy-only eager-prefill trace reuse is excluded
  for the temperature-1 SWE requests (`A/generator.py:277–288`;
  `A/generator_vllm.py:178–188`).
- Hybrid cache views validate local head/block/width geometry
  (`A/generator_vllm.py:114–129`); the fused cache writer derives those same
  dimensions from the view (`ttnn/cpp/ttnn/operations/experimental/paged_cache/
  device/fused_update_cache/paged_tiled_fused_update_cache_program_factory.cpp:
  106–107,138,146–148`). No contradictory cache geometry was found.
- Converted-weight cache reuse cannot explain the candidate: the loader has
  no converted-weight cache path, as traced above.

The short test builds an 8,192-token-capacity model but uses only 161 prompt
tokens and 100 teacher-forced predictions (`T/run_datatype_candidate.py:74–95`;
retained readiness JSON). Its prefill accuracy call concatenates prompt and
continuation, giving **261 logical / 288 physical rows**; decode prefill uses
**161 logical / 192 physical rows** (`models/common/readiness_check/
run_prefill_check.py:91–104`; `A/optimized_decoder.py:701–710`). Neither enters
later-chunk cached full-attention prefill. Capacity allocation and short top1
agreement do not validate 12K–49K numerical quality.

Other potential issues / unresolved evidence: low-precision attention, expert
math, CCL, BF16 router-score rounding, and long autoregressive error accumulation
remain possible numerical contributors, not identified bugs. Existing tests
include long single-layer or reduced-model checks; for example
`T/long_context.py:86–113` uses a 1x1 functional/fused decoder, while
`T/check_full_prompt_lengths.py:21–27` uses only layers 0 and 5. Those are useful
contract tests but do not establish this candidate's full-stack long-SWE quality.
The successful Django/Matplotlib trajectories and short gate do not eliminate
numerical sensitivity; Astropy's over-analysis alone does not establish it.

## 4. Smallest bounded numerical follow-up (not executed)

1. Freeze one existing long request's complete rendered tokens, including
   reasoning/tool history, and its checkpoint revision. The 1,124-token
   Astropy request is the smallest observed branch check; use one retained
   18,772-token or 42,452-token request to test actual long-context quality.
   Preserve all 30 layers, C1, real positions, and actual chunk/tail lengths.
   Start with terminal prefill logits; at most 128 identical teacher-forced
   continuation tokens can extend the comparison. Do not run an agent trial
   or add task guidance.
2. Compare the current candidate against **only** fresh BFP8 EP4 prefill
   replacements: 30 down groups plus gate/up in the five full layers.
   Construct from the BF16/checkpoint values at
   `A/multichip_decoder.py:542–578`, never by casting the existing BFP4 buffers
   upward. Keep LoFi, KV, CCL, head, decode, sampling and prompt fixed. This
   isolates the remaining BFP4-weight question. The present JSON alone cannot
   implement it; any diagnostic override needs truthful construction attestation.
3. Compare aligned logits/top-k and token log-probabilities against a bounded
   same-token HF reference, if available. A difference between two TT policies
   proves sensitivity, not which is correct. Keep cache generations separate
   so each arm prefills its own complete history. Avoid materializing all
   42K-by-vocabulary logits: `A/generator.py:457–482` requests that large result
   through `prefill_logits`; the ordinary last-token path is sufficient.

If that weight-only comparison does not account for the divergence, separately
test full-attention `kv_cache_dtype=bfloat16` (already accepted by
`A/precision_policy.py:135–137`) and explicitly stronger prefill math. Those
address distinct untested boundaries; do not combine them and attribute the
result solely to BFP4 weights. A negative weight-only result would not clear
LoFi/HiFi2, BFP8 cache, CCL precision, or free-running long-generation behavior.

Validation in this investigation: local JSON/diff/hash and source-contract
checks only. After drafting, headline claims were rechecked against the
constructors, active branches, and relevant C++ lowering; an independent
provenance review found no substantive correction. No build was needed for
this report-only change. Repository-root `AUTODEBUG.md` was preserved.
