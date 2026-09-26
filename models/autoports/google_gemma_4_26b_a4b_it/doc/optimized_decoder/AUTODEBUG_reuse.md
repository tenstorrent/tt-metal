# AutoDebug: actual-input request-reuse decode miss

Status: the current evidence localizes the miss to the uncentered generalized
router's numerical boundary. A fresh exact-input run reproduces the failure;
centering the same composite's FP32 scores before BF16 conversion passes. No
implementation change was made by this investigator. Full-contract verification
of any selected default remains parent-owned.

This is a fresh source-only AutoDebug/AutoFix subagent investigation, using the
forked fallback because the CLI runner's bwrap environment was already reported
unavailable. No TTNN import, device execution, or hardware access was performed.
Hardware results below are the parent's recorded artifacts, inspected directly.

## Direct observations

`final_request_reuse_layer5.json`/`.log` use runtime SHA256
`373afe87025f3bd33160cd80ea0930412ceaa1e1c51ce03d84dfb42957811f6a`.
The nine lengths are 31, 32, 33, 1023, 1024, 1025, 2049, 33, 2047. Every prefill
passes. Only request index 6 fails decode: PCC **0.9948247973191795**, below the
unchanged **0.995** threshold. Its prefill PCC is 0.9990028786153986. Both later
requests pass; there is no persistent corruption demonstrated by these scores.

The input is recorded real HF layer-5 text activation data. Request 6 uses rows
768:2817 for prefill and row 2817 for decode, rebased to positions 0:2049 and
2049. `tests/request_reuse.py:47-59,94-98,107-120` establish this policy, including
a new HF cache for every request. Thus position 2049 is the request's absolute
cache/RoPE position, while 2817 is its source-fixture row.

`actual_reuse_failure_2049_1.pt` preserves that exact 2050-row window from
`actual_text_layer5_4096_128.pt`; SHA256 is
`3753fa5c85b498fe121b02a6601ffcef125c34b0b8b0861e2dc4c5127778c78b`.
All following results use that fixture and the same runtime hash. Commands and
return codes are in `reuse_control_commands.json`. Every control passes prefill
(0.999002878380969), runtime audits, and deterministic trace replay. These
focused runs do **not** enable the program-cache-miss guard.

| Result artifact | Only override from fresh defaults | Decode PCC | Result |
| --- | --- | ---: | --- |
| `reuse_control_fresh.json` | None | .9948247936074835 | Fail |
| `reuse_control_output_hifi4.json` | `output_fidelity=HiFi4` | .9948778278726624 | Fail |
| `reuse_control_shared_readers1.json` | `shared_readers=1` | .9948247936074835 | Fail |
| `reuse_control_router_center.json` | `generalized_router_center=true` | .9981819549518332 | Pass |
| `reuse_control_router_fp32.json` | `generalized_router=false` | .9981727130158773 | Pass |
| `reuse_control_expert_activation16.json` | `expert_activation_dtype=bfloat16` | .9948212067944865 | Fail |

The fresh run differs from the reused run by only 3.72e-9 PCC despite a fresh
cache, reverse page mapping, and a newly captured trace at position 2049.
Consequently prior requests, random page reassignment, and capturing at position
31 are not necessary causes of this miss. PCC equality alone is not a claim of
bitwise output identity between the two harnesses.

## Source-backed interpretation

The failing full-attention default sets generalized routing on and centering off
(`tt/optimized_decoder.py:334-337`, at the recorded hash). `GeneralizedRouter`
computes 128 scores, optionally subtracts their FP32 row maximum, and then casts
to BF16 (`:730-743`). It pads with negative infinity and runs top-8 plus softmax
through the same composite (`:745-758`), scattering probabilities to experts
(`:760-765`). Composite validation explicitly requires BF16 scores/output and
UInt16 indices:
`ttnn/cpp/ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/device/generalized_moe_gate_device_operation.cpp:65-69`.
The ordinary router keeps scores for `ttnn.topk` and softmax without that added
cast (`tt/fused_decoder.py:402-427`).

For exact arithmetic, subtracting a shared maximum preserves the ranking and
softmax: `topk(s-c)=topk(s)` and `softmax(s-c)=softmax(s)`. BF16 rounding is not
translation invariant. A common offset affects its absolute rounding grid, so
`BF16(s-c)` need not equal `BF16(s)-c`. Centering therefore changes the information
fed to the composite while preserving the intended mathematical function. It
can preserve distinctions near the largest scores that raw conversion loses.
It is not a guarantee that every rank-8/9 gap survives rounding.

The matched composite pair verifies that changing this numerical boundary is
sufficient to repair this exact failure with the same attention, KV cache,
expert weights, activation policy, and composite. The FP32-router pass independently
supports that localization, although it also changes the selection backend and
softmax implementation. The result does **not** establish an actual expert-ID
flip, a rank-8/9 tie, or a composite kernel defect. Such claims require saved
FP32/BF16 logits, selected IDs, and routing weights for this exact row. A routing
weight change without an ID change remains possible.

`AUTOFIX_actual_precision.md` contains analogous sliding-layer controls showing
raw BF16 ordinary routing failures and a centered-composite pass. That prior
evidence motivated this hypothesis; the new full-layer matched pair supplies
the evidence for the present miss. The unsuccessful output-fidelity, reader-count,
and expert-activation controls do not establish that those components have zero
numerical error; they establish that those individual changes do not repair it.

## Rounded read window and cache coverage

`NativePagedAttention` sets `k_chunk_size=0`, `fp32_dest_acc_en=True`, and passes a
device position tensor (`tt/optimized_decoder.py:909-934`). The native factory
therefore caps dynamic chunks at four tiles:
`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:386-389`.
The kernel computes tiles as `cur_pos/32+1`, caps the power-of-two chunk size at
four, and rounds full-attention reads up from `cur_pos+1`:
`.../sdpa_decode/device/kernels/rt_args_common.hpp:61-70,95-107`.
`.../kernels/dataflow/reader_decode_all.cpp:167-177,294-343` consumes that window
through the page table for K and V.

At failed position 2049:

- Logical updated page is `2049 // 32 = 64`, row `2049 % 32 = 1`.
- Valid cache length is 2050; sequence tiles are 65.
- Dynamic chunk size is `min(next_power_of_two(65),4)*32 = 128` tokens.
- Rounded exclusive read end is `ceil(2050/128)*128 = 2176`: 17 chunks,
  68 logical pages, pages 0..67. The failing token belongs to chunk 16.
- Reuse allocates 4096 tokens / 128 pages (`tests/request_reuse.py:47,77-85`).
- Fresh control allocates 3072 tokens / 96 pages via
  `(2049+max(128,1)+1023)//1024*1024`
  (`tests/run_decoder.py:102,137-143`). Both cover the 2176-token read window.

The passing position 2047 has the same 128-token dynamic chunk size and reads
2048 tokens / 16 chunks / 64 pages. The boundary adds a chunk, but does not change
the dynamic chunk size or exceed either allocation. At length 2049, prefill writes
the final tile through row 2079 and decode updates row 2049
(`tt/optimized_decoder.py:578-607,1052-1069`). Reads beyond the valid position are
part of native attention's masking contract. No allocation fix is justified by
the present evidence. The centered-router pass retains the exact same cache and
attention policy, further weakening a cache-boundary explanation.

## Precision ledger and remaining controls

The matched pair retains QKV/output BFP8 weights, direct FP32 QKV input/output,
LoFi QKV/output/SDPA, BFP8 KV cache, FP32 SDPA destination accumulation with full
sync, BFP4 expert gate/down with LoFi and BFP8 input activation, BFP4 shared
weights with two readers, FP32 residuals, batch one, 32-token pages, and the same
native cache update/read ops. Prefill gate/down remain BFP4; prefill attention is
LoFi. Only pre-BF16 router centering changes. Exact policies are recorded in each
JSON; no cache-dtype workaround was used.

The parent should promote a centered full-layer default only after rerunning the
original nine-request reuse check, actual 4096/128 headline, actual 1025/512
stress, and affected long/context contracts under the final source hash. All
positions and thresholds stay unchanged. Reprofile complete-layer performance
if this becomes the selected implementation; focused PCC results make no speed
claim.

If a later failure needs mechanism proof, the next single-variable controls are
ordinary FP32 top-k versus ordinary top-k after raw BF16 conversion, followed by
centered ordinary top-k, with otherwise identical production inputs. Capture
FP32 logits, rank-8/9 gap, IDs, and routing weights outside the scored timing;
first verify instrumented output matches the saved failing output. Existing
ordinary-route probes intercept `ttnn.topk` and cannot observe composite IDs
without adaptation. If cache evidence reappears, use the exact model geometry
and an over-allocation control before changing cache precision; then perform
same-cache higher-precision attention and exact-shape cache/page-table probes.

Report verification: source inspection, JSON parsing, recorded policy/command
comparison, and arithmetic only. No build is needed for this report-only change.
