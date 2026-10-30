# AutoDebug: full-attention accuracy at 262144 tokens

## Finding

**The leading diagnosis is excessive serial BF16 online-attention accumulation, exposed by the paged-decode default of 32-token chunks and one core per KV head.** FP32 destination accumulation does not make the persistent attention state FP32. Newly available external probe results strongly support this diagnosis: changing only the core cap from 1 to 16 improves raw attention PCC from about .896 to .999, while disabling approximate exp alone barely helps. Prefill retains a BF16 output accumulator too, but has FP32 scores and sums; this is consistent with its smaller degradation. The exact contribution of each rounding point remains unproven.

No clear model-semantic, page-index, or trace-only bug was found that better predicts the reported contrasts. Do not relax the accuracy gate. The external probe identifies a promising decode configuration; validate it through the original cache-producing decoder and trace tests before adopting it.

## Scope and evidence provenance

Inspection only: no device commands, reproductions, source changes, resets, or external publication. Only this report was written. Read the repository-root `bringup/run_contract.json` and `bringup/agentic_evals/inputs.json`. Inspected the new autoport, local HF Transformers reference, and generic TTNN/LLK source; no removed Gemma implementation or Gemma-specific `tt_transformers` code was used.

Prepared checkout HEAD: `db97382f276a5aecc1cd98a7eb5ba5def783f832`. Config and weight downloads in `tests/run_decoder.py:18-24,38` use `842da3794eaa0b77d5f08bae87a17459d91ff475`. HF executable reference code is the locally installed Transformers implementation, rather than code identified by that model-weight revision. The work log identifies Transformers 5.12.1.

The historical log's traceback line numbers do not match the present harness. Thus the reported measurements are observations, while exact dispatch below is established for the prepared source. The log does not record loaded shared-library identity or effective SDPA arguments. Confirm those before treating present-source dispatch as proven historical dispatch.

### Direct observations

| Artifact/case | Prefill PCC | Decode PCC | Other evidence |
|---|---:|---:|---|
| `doc/functional_decoder/full_context.log`, 262143 prefill tokens | .9986207485 | .9902030230 | Three traced scores equal eager score; changed inputs .9935083389 |
| Same log, exact 262144-token prefill | .9981801510 | — | Last output token, also used by initial decode comparison |
| `full1025.json` | .9999918938 | .9999897480 | Real weights, batch 1 |
| `sliding_context.json`, 262143 tokens | .9999270439 | .9999141693 | Changed inputs .9999462962; batch 1 |

Long prefill PCC covers the last 32 rows, not all 262143 rows (`run_decoder.py:56,113-116`). The exact-maximum comparison covers one row (`175-184`). Prefill exceeds the .995 gate; decode and updated-input results fail it. Sliding uses a different layer, head dimension and weights, so it is evidence consistent with bounded reduction length, not a pure single-variable control.

On initial inspection `long_attention_probe.log` contained initialization only. During final review it had been updated by concurrent work: it reports `QUERY_PCC=0.999995768070221`, followed by the incomplete-program-config constructor TypeError described below. This supports accurate query production for that probe, but supplies no successful workaround, raw attention result or final layer result. The probe source was also corrected concurrently; those edits were not made by this investigation.

### New controlled evidence found during final review

`long_attention_probe3.log` and `long_attention_probe.json` subsequently became available. These runs were performed outside this investigator session. The corrected probe uploads HF K/V directly with identity paging and tests both TT-produced and HF-produced Q. Selected TT-query results:

| K chunk | Core cap | Approx exp | Attention PCC | Layer PCC |
|---:|---:|---|---:|---:|
| 32 | 1 | true | .89620256 | .99027508 |
| 32 | 1 | false | .89647281 | .99051529 |
| 32 | 16 | true | .99878120 | .99986559 |
| 32 | 16 | false | .99870139 | .99985796 |
| 64 | 1 | false | .91941082 | .99190581 |
| 64 | 16 | false | .99989325 | .99998039 |
| 128 | 16 | false | .99997258 | .99999255 |

The implicit default exactly matches explicit K32/core1/approx-true results in this probe. HF Q shows the same contrast: default attention PCC .89673215, K32/core16/approx-true .99853158, and K128/core16/approx-false .99997205. Thus TT Q error, cache-fill error and random page addressing are not necessary for the isolated failure. Exp accuracy alone is insufficient. Shortening serial recurrence is the intervention best supported by both source and these controlled results. Core splitting also changes the merge path, so these data do not separately prove numerator versus denominator error; a persistent-FP32 experiment remains discriminating. K128/core16 is a measured successful isolated configuration in this external run, not yet a full original-decoder/trace validation.

## Lowered cause chain

Paths below are relative to the repository root, except `tt/` and `tests/`, which are relative to this autoport.

### 1. Decode defaults turn a long prefix into 8192 sequential reductions

`tt/functional_decoder.py:49-51,246-254` supplies HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True`, `packer_l1_acc=True`, BF16 Q/K/V, `scale=1.0`, and **no program config**. Q is interleaved; K/V are paged DRAM caches with 32-token pages. Q has 32 heads, full-attention K/V have 4 heads of dimension 512. The temporary height-sharded K/V at `243-245` are for cache update; they do not make attention Q sharded.

`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/sdpa_decode.cpp:18-19,114-121` replaces absent config with device grid, Q/K chunk sizes 32, and `max_cores_per_head_batch=1`. In `device/sdpa_decode_program_factory.cpp:200-215`, batch 1 and 4 KV heads therefore use four compute cores, one per KV head, with no tree-reduction rounds. Each core carries the query-head group over the entire prefix. It is incorrect to infer multi-core sequence splitting merely from the available device grid.

Concrete dimensions:

| Case | Position / live prefix | K chunks at 32 tokens | Consequence |
|---|---|---:|---|
| Full short decode | 1025 / 1026 | 33 | Last chunk has 2 live tokens |
| Full maximum decode | 262143 / 262144 | 8192 | Last chunk is fully live |
| Updated batch-1 decode | 262142 / 262143 | 8192 | Last chunk has 31 live tokens |
| Sliding maximum decode | 262143 / last 1024 | 32 | Prefix before window is skipped |
| Sliding updated decode | 262142 / last 1024 | 33 intersecting chunks | First and last chunks partially masked |

For maximum full context the page table has 8192 entries; cache allocation is 8195 physical blocks because the first three are sentinels (`run_decoder.py:102-105`). The additional physical blocks are not extra logical attention tokens. Both 262143-token and 262144-token prefill use the same 262144 physical token bucket. Initial decode overwrites the last token with that same token after exact-maximum prefill, rather than appending beyond the advertised context.

### 2. FP32 DEST is followed by BF16 storage every iteration

Decode factory `device/sdpa_decode_program_factory.cpp:497-508` hardcodes `im_df=Float16_b` and `stats_df=Float16_b`. These cover score/probability intermediates, output accumulation and softmax statistics. `fp32_dest_acc_en=True` changes arithmetic inside an operation; it cannot restore information rounded away by packing persistent state into these CBs.

`device/kernels/compute/sdpa_flash_decode.cpp:479-574` performs online softmax. Conceptually, for each new chunk:

```
m_new = max(m_old, max(scores))
alpha = exp(m_old - m_new)             # scale is 1 here
l_new = chunk_sum + alpha * l_old
O_new = chunk_probability_times_V + alpha * O_old
```

The code packs/rescales/adds BF16 persistent `l` and `O` on every iteration; final reciprocal and normalization are at `687-696`. Full maximum decode has 8191 updates after initialization. Short and sliding attention have only roughly 32.

A cheap CPU-only illustrative check rounded to BF16 after every `sum += 32`: 33 additions returned 1056, 256 returned 8192, and 8192 additions still returned 8192 instead of 262144. This is not a simulation of actual logits or the device packer and does not predict PCC. It demonstrates why destination precision alone cannot guarantee a long BF16 recurrence's accuracy. Constant V could cancel matching numerator/denominator errors; actual signed, varying V need not. Per-head denominator errors also survive concatenation and output projection. A single global attention scaling would mostly cancel under post-attention RMSNorm, so that alone is insufficient to explain the layer PCC drop.

### 3. Prefill shares the numerator risk but not the BF16-sum risk

`tt/functional_decoder.py:169-190,212-223` chunks input projection/cache filling into 1024-token chunks, but attention for each query still covers its whole causal prefix. `q_chunk_size=k_chunk_size=32` does **not** bound historical context to 1024.

In `sdpa/device/sdpa_program_factory.cpp:79-81,1108-1112,1216-1223`, FP32 DEST selects legacy compute; QK and sum CBs are FP32, while output A/B, maxima and max-rescale factors remain BF16. `sdpa/device/kernels/compute/compute_common.hpp:1934-2010` packs each chunk's QK@V output, rescales/adds the previous BF16 output through L1 accumulation, and swaps persistent buffers. The final query tile at maximum context processes 8192 K chunks. Parallelizing different query tiles does not split this recurrence.

This distinction is consistent with the qualitative hierarchy: short full attention passes, long full attention degrades in both modes, and decode can degrade more because its scores and denominator are also BF16. It does not establish how much each rounding point contributes. Changing only the decode default cannot remove the separate prefill numerator issue.

### Buffer and completion checks

For the selected decode path, one K chunk is one token tile by 16 D tiles: 16 K tiles and 16 V tiles are consumed per iteration. The QK block is one tile, and the output block is 16 tiles. The first iteration initializes output accumulation; later iterations rescale/add it. QK is popped each iteration; current max/sum are moved to previous buffers; final normalization consumes the accumulated sum. The same nominal tile counts apply to partially masked last chunks. No partial-chunk producer/consumer count contradiction was identified on this path. There are no inter-core reduction children under the absent-config default.

Host outputs are read with `ttnn.to_torch` before PCC; the harness explicitly synchronizes before capture and replays with `blocking=True` (`run_decoder.py:113,213-225,247-248`). This is a completed wrong-output observation, not an inference from merely reaching the end of an enqueue-only call. Equal eager/traced scores and repeatable trace output argue against a trace-only cause; they do not prove every upstream value is correct.

## Minimal discriminating tests for the fixing session

These are proposed follow-ups, not tests executed here. The new external probe has now supplied much of steps 1-2; use its successful K128/core16/exp-false configuration in the original decoder next, rather than blindly repeating completed controls. Preserve the context and threshold.

1. **Establish effective configuration and isolate SDPA.** Record loaded TTNN library/source identity and actual factory grid/chunk/core/format values. Upload exact HF Q/K/V for the single last query, with a known page mapping. Compare raw attention to CPU FP32 attention over those same BF16 inputs before `_finish`. Repeat using TT-produced Q and then TT-produced caches. This separates attention reduction from QKV/RoPE/cache production errors without rerunning full prefill for every control.
2. **Run one-variable decode controls.** Use an explicit baseline matching `None`: device grid, `q_chunk_size=32`, `k_chunk_size=32`, `max_cores_per_head_batch=1`, `exp_approx_mode=True`. Hold all else fixed while testing (a) only exp false, (b) only K chunk 256, (c) only core cap 16. Afterward combine successful changes. Validate supported geometry/L1 capacity before larger chunks. At 262144, K256/core1 gives 1024 serial chunks; K32/core16 gives 512 local chunks plus tree merges if the grid permits 64 active cores. Shorter recurrence improving raw attention is evidence for the leading diagnosis, although changing splitting also introduces a different reduction path.
3. **Match the same query across prefill and decode.** Use HF-uploaded caches and the last 1024 queries in chunked prefill with absolute start 261120. Compare its last row with decode at 262143 using the same uploaded Q/K/V. Sweep prefill K32 versus K256 at fixed exp settings. The legacy validator requires `chunk_start_idx % k_chunk_size == 0` (`sdpa/device/sdpa_device_operation.cpp:396-402`); the last-32-row start 262112 is invalid for K256. Raw attention, per-head norm/error, and final layer PCC are all needed; the residual can hide attention error.
4. **If config controls implicate accumulation, test persistent precision directly.** In a separate fixing experiment, promote running O and denominator to FP32 with all tile-byte allocations, pack/unpack formats and helper assumptions updated coherently; isolate O versus denominator changes. Prefill already has FP32 denominator, so promote O there separately. This is a stronger causal test than increasing math fidelity, which is already HiFi4. It is not safe to blindly change one CB enum and assume helper contracts still match.
5. **Validate the long reference independently.** For one selected query, compute explicit FP32 QK, stable softmax and P@V over all prefix keys; also reproduce HF eager's BF16-logit/probability rounding. Compare these against `selected_reference`. This requires only one query row, not quadratic full-context attention.

Use sequence lengths such as 1026, 8192, 32768 and 262144 with the same query and nested KV prefixes for a reduction-length sweep. Track attention PCC, absolute/relative error and per-head norms. The selected hypothesis predicts sensitivity to recurrence length; it does not predict that different randomly generated examples have monotonically decreasing layer PCC. A combined decode K128/core8 control gives 256 local chunks plus three merge rounds at maximum context. Avoid blindly combining K512/core16: factory CB sizing predicts about 2.87 MiB per core including the page table, versus about 1.04 MiB for K128/core8; neither is a measured placement result.

## Other potential issues and test gaps

- **Historical probe failure, now corrected in source.** The initial `tests/probe_long_attention.py:52` constructed `SDPAProgramConfig(exp_approx_mode=False)` and other incomplete configs. `transformer_nanobind.cpp:41-44` requires grid, Q chunk and K chunk; the updated log confirms the constructor TypeError. All list elements were constructed before loop execution, preventing even the nominal default row from running. An explicit config also defaults to 16 cores per head (`:46`), whereas omitted config becomes one core, confounding a default-versus-"exact_exp" comparison. Final inspection found the probe source corrected at `:52-56` to specify grid, chunk, core cap and exp for controlled cases. No raw-attention or layer result was present in the failed probe log at review time. Compare explicit cases against each other; its explicit 8x8 grid can differ from the implicit device grid.
- **Exp approximation is independently enabled.** Decode factory `:221-223` and prefill factory `:105-110` default `exp_approx_mode` to true despite `math_approx_mode=False`. This warrants an isolated control, but is not proof of an exp defect or proof that exact exp fixes the recurrence. Even exact exp still packs persistent BF16 state.
- **Blackhole exp-path adjudication:** prefill `compute_common.hpp:369-407` selects the approximate full-tile exponent, but running-max correction uses a distinct first-column helper. In `tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h:196-215`, the approximate-mode branch of that correction calls accurate FP32 exp. An attractive story in which an approximate `exp(0)<1` repeatedly decays the previous accumulator is therefore unsupported. No current mixed-format contradiction was found in the inspected prefill recurrence. A future output-only FP32 change must explicitly reconfigure packing in `compute_common.hpp:739-764` for the BF16 max-difference output, or promote compatible statistics alongside output; current code relies on those formats matching.
- **Long reference is not literally HF eager.** `tests/reference.py:46-52` calls PyTorch SDPA, whereas the small-length HF control uses eager attention. Agreement is checked only at length <=4096 (`run_decoder.py:58-67`). Long-context numerical equivalence is unmeasured. This affects the absolute interpretation of the gate, but does not by itself explain why prefill and decode differ for the same selected last row.
- **Layer semantics inspected without an obvious contradiction:** full attention reuses raw K projection for V before their separate normalizations; Q/K normalization and rotation, V normalization without rotation, GQA head grouping, scale 1, four layer norms, tanh GELU, residuals and layer scalar match local HF. Masking includes the current token and excludes positions at or before `position-window`. Treat QKV precision and reference rounding as unresolved quantitative contributors, not established semantic bugs.

  HF source: `python_env/lib/python3.10/site-packages/transformers/models/gemma4/modeling_gemma4.py`, RMSNorm `202-211`, RoPE/GQA/eager attention `780-850`, attention setup and K/V construction `1185-1195,1243-1265`, decoder order `1409-1454`.
- **Batch coverage caveat:** exact-maximum prefill comparison indexes output batch 0 but compares with all selected batches (`run_decoder.py:181-182`), so this branch is not valid general batch>1 coverage. Reported maximum-context evidence is batch 1. Likewise flipping page rows is the identity operation at batch 1; the updated-input log label does not demonstrate multi-request page-row permutation correctness.
- **No established boundary overflow:** updated position 262142 and initial position 262143 both use 8192 chunks; causal masks differ by one live token. Their different PCC values concern different query inputs, not a controlled one-token accumulation experiment. Physical sentinel pages, partial final prefill padding and later cache overwrite are consistent with the intended ownership for the inspected batch-1 path.

## Claim review

Rechecked the central claims against wrapper defaults, factory formats, actual online update loops, and the newly available controlled probe. The report deliberately does not claim a proven BF16 saturation value for real logits, a trace bug, an out-of-bounds access, or a fully revision-matched original full-context binary. The smallest supported intervention is an explicit decode program config that shortens local recurrence. The stronger implementation remedy, if needed, is preserving running reduction precision or using a numerically suitable split/merge scheme. Prefill needs its own validation because it retains BF16 O independently of decode defaults.
