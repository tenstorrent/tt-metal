# AutoDebug: fused layer-0 request-3 decode regression

## Finding

**The fresh owner controls localize the regression to the fused GDN QKV projection. Restoring that projection alone passes the complete batch32 gate while retaining exact convolution, direct recurrence layouts, the other fused projections, packed Z/A/B, and fused MoE. Restoring only the GDN output projection leaves the original request-3 failure unchanged. A further control retaining accurate sum but returning QKV to 256-output groups still fails with identical recorded metrics.**

The leading mechanism is the changed FP32 summation order before the QKV BF16 boundary. The functional projection uses an explicit halves/addition tree; the fused projection serially folds tiles through the accurate SFPU reduction. “Accurate FP32” does not preserve the original arithmetic order. This particular request has historical, measured sensitivity to prefix-state/projection changes and a tied expert-selection cutoff.

Confidence is **high for localization to the QKV reduction replacement**, and **moderate to high for arithmetic-order drift as the mechanism**. The successful original-projection control restores both the tree and 256-output blocking, but the additional sum/group1 control fails at that same blocking. This rules out group64 as a necessary trigger and strengthens the reduction diagnosis. A generic-sum kernel or trace-sensitive defect remains possible because the operation graph still differs. The fresh artifacts contain no device QKV/state/router captures proving the first differing value or current expert-membership change. Do not present the historical expert switch as a newly observed switch.

The smallest evidence-backed intervention is to preserve the original `PreciseGDNProjection` for QKV at `tt/fused_decoder.py:1029`, leaving the other selected fusions in place. An optimized replacement should preserve that projection's numerical contract and be validated with the same batch fixture. Changing tolerances or extending the request-16 backend exception would not address this regression.

This session performed inspection and cheap host-only evidence/static checks. It did not run TTNN, reproduce on hardware, reset devices, edit runtime/test source, or relaunch AutoDebug. The new runtime controls below were produced by the owner and read during this investigation.

## Evidence and revision ledger

References beginning `M/` mean `models/autoports/qwen_qwen3_8_flash_next/` under `/home/ttuser/qwen38-flash/tt-metal`. Artifact names in the table are under `M/doc/fused_decoder/`. Fused source line references refer to the failing SHA55c version, preserved exactly in `../sources/batch0_control_qkv_tree_v1/fused_decoder.py`; do not resolve them against a subsequently edited live file.

HEAD is `70dc6f5dd24298741e2f108a281160677e986155`. The failing runtime and all six rows below record fused SHA256 `55c66422c92ed477bcd0c644608ab69868a6216d48607755ea8268bd5f5e31ad`. Source identity matters because the fused file is untracked, so HEAD alone is insufficient.

Verified probe SHA256: `0ad6b3d4d94cf1bf395c93e54515f74dd74161ae3139e0fb1ae3ffda5ad17f87`.

| Completed artifact | QKV implementation | GDN output implementation | Request 3 decode output PCC | Contribution PCC | Full policy result |
|---|---|---|---:|---:|---|
| `final_v2_batch0_32.json` | Fused sum/group64 | Fused sum/group64 | 0.9928041697 | 0.9835285544 | Fail |
| `batch0_control_baseline_v1.json` | Functional tree | Functional tree | 0.9999905825 | 0.9999780059 | Pass |
| `batch0_control_gdn_tree_v1.json` | Functional tree | Functional tree | 0.9999909997 | 0.9999795556 | Pass |
| `batch0_control_qkv_tree_v1.json` | Functional tree | Fused sum/group64 | 0.9999909997 | 0.9999795556 | Pass |
| `batch0_control_out_tree_v1.json` | Fused sum/group64 | Functional tree | 0.9928041697 | 0.9835285544 | Fail |
| `batch0_control_qkv_sum_group1_v1.json` | Fused sum/group1 (256 outputs) | Fused sum/group64 | 0.9928041697 | 0.9835285544 | Fail |

All report exact repeat determinism. Each failing row has exactly one unaccepted request/phase: request 3 decode. Every run retains the same accepted request-16 prefill exception. A policy pass is not a claim that request 16 strictly matches literal HF.

The original failure and functional baseline have identical recorded source-hash and environment dictionaries. The four selective controls retain those runtime hashes/environment and add a diagnostic script. Their provenance files are `status=finished`; exit codes are 0 for baseline/GDN/QKV and 1 for final/output-only/sum-group1. The selective-control log hashes match their provenance. Logs explicitly record the selected projection objects, including QKV-only `(PreciseGDNProjection,10240,True)` and output-only `(PreciseGDNProjection,2560,True)`.

The original-control diagnostic, preserved in `./probe_fused_batch_control.py.txt:29-34`, returns the original object for the selected projection and constructs `FusedPreciseProjection` for the others; it does not change thresholds or HF. The QKV/output runs record diagnostic SHA `5f8f5f2e5b6b1caa0b8f848da0e752575c600d5ae3bd4ae292331d074f9d34e8`. The preserved formatted diagnostic has SHA `99f6a49532b34c1c66dff5981311db40ba4aac91742a7316ca749f7b63d80e43`; undoing only the line wrapping of `selected = ...` in memory reproduces the recorded QKV/output hash exactly. The earlier both-GDN run used diagnostic SHA `78cd43ddeb882ba0a40ec73d48c088241734399081c312e0a5dc5db1e5b95361`; its log records both GDN replacements. The group1 diagnostic is preserved byte-for-byte in `./probe_fused_batch_control_group1.py.txt`, SHA `e2f54460d5c666f52f993ba982b35c43abe66184d36c27440fd679659d2f0613`, matching its provenance; its log selects group1 only for QKV. All 32 request metric dictionaries in sum/group1 equal those in the failing sum/group64 report. Equal metrics do not prove cross-run bitwise output equality; raw outputs were not saved. No historical script revision is silently treated as byte-identical to a later live file.

During report review, the owner began a separate repair and changed the live fused source to SHA `a6f31432f825834b2c0cc0c8848e21bb4257a05cb4eba04fdf5c0ee8a46076ac`, adding `balanced_tree`/`preserve_qkv_tree`. That is not the source used by the six controls above. This report diagnoses SHA55c and does not claim validation of that subsequent implementation.

The failure's maximum absolute output error is 0.01904296875, versus 0.001953125 for functional baseline request 3 decode. Fused request 3 prefill passes at output/contribution PCC 0.9999934435/0.9999862909. Request 3 literal-HF versus GR-FP64-control output is bit-exact in both phases; that backend carry cannot explain its decode failure.

Same-fused-SHA `final_v2_conv_component`, `final_v2_layer0_hf`, `final_v2_continuation0`, and `final_v2_synthetic` provenance all finish with exit 0. The exact-convolution component reports bitwise original/replay agreement at lengths 1/2/31/32 and passes explicit BF16-midpoint/cancellation controls. These are positive contrasts, not proof of equality for every integrated intermediate.

## Actual execution: batch32 is 32 independent single-request calls

`M/tests/probe_batch.py:235-292` seeds 1981 for layer 0, constructs requests in order, and uses prompt length `1 + request % 3`. Each iteration consumes prompt randomness, decode randomness, and token-ID randomness even though layer 0 does not use the IDs. Request 3 has **one prompt token followed by one decode token**. There are 11 length-1, 11 length-2, and 10 length-3 prompts, totaling 63 prefill token rows and 32 decode rows.

`M/tt/functional_decoder.py:144-151,180-190` implements batch tuples by looping over requests and pairing each with `state[i]`. The failing device call has logical token count 1, not a 32-token reduction or recurrence. The other ten length-1 requests pass. Existing single-request suites use other fixtures/seeds (`probe_decoder.py:27`, `test_synthetic_decoder.py:92`, `probe_public_continuation.py:77`). Their passes do not test this exact request in isolation. A true batch-size dependency has not been demonstrated.

State ownership/completion ledger:

1. `probe_batch.py:438-443` allocates an independent state for each request. Layer-0 state is conv BF16 `[1,1,3,10240]` and recurrent FP32 `[1,48,128,128]` (`functional_decoder.py:91-94`).
2. Prefill outputs and both state tensors are read back before snapshots are prepared (`probe_batch.py:477-487`). No layer-0 mutable tensor was found omitted from restoration.
3. Warm decode is explicitly synchronized at line 491, then both buffers are restored before capture.
4. Each of two replays restores the snapshots, executes with `blocking=True`, and reads outputs back (`:505-514`). The failing PCC observes completed work, not merely successful enqueue.
5. **Coverage gap:** warm/eager outputs are discarded at `:489-493`; only replay is scored. Deterministic repeated replay does not prove eager/replay equality or exclude a deterministic address/cache problem.

For request 3, conv history after prefill is `[0,0,QKV(prompt)]`; decode consumes `[0,0,QKV(prompt),QKV(decode)]` and retains the last three rows. QKV therefore influences both stored convolution history and the recurrent state generated during prefill. A passing prefill output gate does not establish identical stored state.

## Where the QKV numerical contract changes

The original GDN projection stores 40 BF16 parent blocks of shape `[1,1,256,2560]` for QKV, promotes operands to FP32, forms FP32 products, repeatedly adds lower/upper halves, appends odd tails, and casts only the final output to BF16 (`M/tt/precise_gdn_projection.py:25-70`). Single-token calls use 256 output channels per block; multi-token calls slice 32 channels. Its existing docstring explicitly records the previous request-3 repair.

`FusedPreciseProjection` retains weights and final BF16 output but replaces that explicit tree with `ttnn.sum(...,dim=-1,fast_and_approximate_mode=False)` (`M/tt/fused_decoder.py:31-68`). `projection_decode_group=64` joins all 40 QKV parents into one 10240-output block. For request 3, **both prefill and decode** take the `count == 1` branch; the option's name does not limit its effect to decode mode.

Concrete geometry:

| QKV stage | Functional / restored control | Failing fused default |
|---|---|---|
| Input | BF16 `[1,1,1,2560]` | Same |
| Expanded input | FP32 `[1,1,1,2560]` | Same |
| Per-call weight/product | 40 blocks, FP32 product `[1,1,256,2560]` | One FP32 product `[1,1,10240,2560]` |
| Reduction | Halves/tree over K=2560, plus odd tail | W reduction: 320 tile rows, 80 width tiles per row |
| Product payload | 2,621,440 bytes per block | 104,857,600 bytes |
| Result | BF16 `[1,1,1,10240]` | Same logical layout/dtype |

Group64 does not change which weight column belongs to which output. Concatenation order and subsequent reshapes preserve that mapping. Its material changes are work grouping, live workspace, and the reduction implementation/order. For count 2/3, the GDN wrapper still uses 32-output slices, although the changed reduction order remains.

Lowering confirms this is **accurate SFPU, not an accidentally selected TF32/FPU reduction**. `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp:202-217` selects SFPU for FP32, `fast=False`, FP32 destination accumulation, and non-Quasar architecture. The logged device is Blackhole. The W program factory explicitly configures direct FP32 destination unpack (`reduce_op_multi_core_w_program_factory.cpp:477-488`).

The effective order differs nonetheless: `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:108-124` seeds from the first tile and adds each later tile; `:521-535` traverses width tiles serially, then `:552-555` performs the intra-tile reduction. Thus a lane folds products at offsets `j,j+32,j+64,...` before the final 32-column reduction. The original tree instead begins with product pairs separated by 1280, then 640, and so on. Correct FP32 arithmetic can produce different BF16-rounded outputs when sums approach a midpoint or undergo cancellation. No generic reduction contract promises the model's original tree order.

An independently checked, host-only scalar example demonstrates that this difference can survive the final BF16 boundary. Let K=2560, with every product zero except `p[0]=2`, `p[1]=1`, `p[2]=2^-8`, `p[32]=2^-23`, and `p[160]=-2`. These values can all be products of BF16 operands. Emulating each addition with FP32 rounding using Python `struct` gives:

| Association | Final FP32 | BF16 nearest-even |
|---|---:|---:|
| Functional halves/tails tree | 1.0039063692092896 | 1.0078125 |
| Serial tile-lane fold, then row reduction | 1.00390625 | 1.0 |

The serial lane loses `2^-23` when added to 2 before cancellation; the tree cancels positions 0 and 160 before position 32 joins. This proves numerical non-equivalence, **not** that these values occur in request 3 or that a device test was run.

The selected reduction's local buffer/phase audit found no concrete contradiction. The W reader reserves, reads, barriers, and pushes batches of four into an eight-tile input buffer (`.../reduction/generic/device/common.hpp:36-44`; `.../device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp:30-40`). Compute waits/pops one tile per iteration. A QKV width row consumes 80 tiles, exactly 20 reader batches, for both passing and failing single-token requests; per-core work consists of whole rows. Destination 0 holds the accumulator, destination 1 the next tile; the row reducer is initialized after folding, and commit/wait precedes packing (`reduce_helpers_compute.inl:508-567`). Blackhole `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h:1881-1929` resets counters and initializes row masks/replay. Its row path explicitly separates the float-add/shift dependency at `:669-679`, with later stores providing separation at `:695-703`. This audit supports the intended arithmetic path; it is not an exhaustive ISA timing proof.

The same fused wrapper also replaces four GR projections, router projection, and GDN output (`fused_decoder.py:1016-1030`). The fresh QKV-only recovery shows those other replacements can remain active for this complete fixture. Disabling all precise reductions would obscure this narrower boundary.

## Causal chain and remaining uncertainty

The source supports this chain: changed QKV result → changed BF16 convolution history/output → changed normalized Q/K/V and recurrent state → changed GDN output → changed post-attention/MLP mixed input → potentially changed BF16 router cutoff → a larger expert contribution difference. QKV is called before history construction (`fused_decoder.py:826-827`); history is copied to persistent conv state (`:860`), and recurrence state is copied back (`:892`). The precise point of first fresh divergence remains uncaptured.

Historical evidence helps prioritize the chain but is kept separate. `M/doc/functional_decoder/request3_device_projection_precision.json.xz` records request-3 HF decode experts ending in 57, with expert 386 next; both boundary logits are 1.09375. In that older native-projection failure, device expert 386 replaced 57; restoring QKV/output trees recovered the HF set and contribution PCC 0.9999782443. `request3_prefix_projection.json.xz` and `work_log.md:476-478` also record recovery from combined prefix conv/recurrent substitution, whereas either state alone did not recover the old failure. Those runs used different runtime component hashes and are not current device-route evidence.

The fresh QKV control proves a sufficient intervention, not exclusive ownership of every difference. The subsequent sum/group1 failure excludes the large output group as a necessary trigger: at the same 256-output grouping, sum fails while the original tree passes. Arithmetic order and the sum-versus-tree operation graph remain coupled. Both controls apply to prefill and decode, so they do not establish whether the decisive divergence originates in the prefix, the current token, or both. A shared downstream operation could respond differently to changed inputs; the observed recovery alone is not a proof that every consumer is correct.

## Competing explanations, adjudicated

| Candidate | Direct source/evidence support | Unverified precondition / verdict | Smallest distinguishing control |
|---|---|---|---|
| QKV reduction-order change | Known changed order; original QKV passes, sum fails at both group64 and group1; output-only restoration fails | **Leading mechanism**, but fresh per-value causality missing | Same saved QKV operands, same output grouping, compare tree vs sum before/after BF16 |
| QKV group64 geometry/workspace | Sum/group1 has the same recorded failure with 256-output groups | Demoted as a necessary cause; no index or work-count contradiction found | Grouping discriminator is completed |
| Generic-sum kernel/cache/trace behavior | Both failing sums share a kernel family unlike the passing tree | Residual alternative; no concrete low-level violation found | Compare sum/tree on identical operands eagerly and under replay |
| Prefix state divergence | QKV writes both state sources; historical state substitutions recovered this fixture | Plausible link, not an independent demonstrated state-management bug | Functional/fused prefix × functional/fused decode, restoring both buffers |
| Exact convolution arithmetic | Same-source component/replay agreement; QKV-only recovery leaves exactconv active | Demoted; no fresh mismatch in identical-input conv shown | Compare exact/original on captured failing history, only if upstream outputs match |
| Direct recurrence reshape/layout | Same recurrence kernels; QKV-only recovery retains direct layout | Demoted; no incorrect mapping identified | Same-operand recurrence output and final-state equality |
| MoE/shared/weighted reduction | Some numerical boundaries differ, but all remain enabled in the passing QKV control | Demoted as required repair; possible downstream amplification remains | Fixed-route or same-input MoE comparison after locating first divergence |
| Request-16 GR backend sensitivity | Literal versus control differs only for its documented carry; request3 control is exact | Ruled out as explanation for request3 decode | Already discriminated by matched control |

## Discriminating follow-ups for the owner

1. **Protect the narrow passing boundary.** Preserve original QKV while validating the final selected implementation. The completed QKV-only batch run already supports this intervention; do not relax the 0.995 gate. Any new fused QKV tree still needs the same full batch and existing synthetic/continuation gates.
2. **Grouping discriminator is now completed.** The owner's QKV accurate-sum/group1 control fails with the same recorded metrics as group64. Next compare original tree and accurate sum on identical saved operands, including BF16 mismatches; a grouped tree can then be evaluated as an optimization. Preserve the original arithmetic association rather than merely selecting more precision.
3. **Locate first divergence on paired operands.** Capture request-3 prefill/decode QKV inputs and outputs, conv history, normalized Q/K/V, recurrent state, GDN output, MLP mixed input, BF16 router logits, and actual selected IDs in a separate diagnostic. Compare functional/fused projections on the *same saved input*. Record coordinate, phase, dtype, and source identity; aggregate PCC is insufficient near tied logits. A fixed-route diagnostic can test how much error the selected set explains.
4. **Separate prefix from current-token effects.** Use a 2×2 matrix of functional/fused prefix producer and functional/fused decode implementation, restoring the same conv and recurrent tensors for each run. Split conv-only/recurrent-only substitutions afterward if useful. This tests state-content divergence without assuming a stale-state bug.
5. **Close the trace gap.** Compare warm/eager output and updated state against replay from identical snapshots. Isolate exact fixture3 only after generating preceding fixtures with their original RNG consumption, including token IDs. Re-running a different single-request random fixture is not that control.

These are proposed hardware diagnostics, not tests executed by this investigator.

## Other potential issues and test traps

- **`pack_gdn=False` alone does not disable packing.** At `fused_decoder.py:975-979`, any of packing/nativeconv/exactconv/layout being enabled installs `FusedGatedDeltaNet` and unconditionally concatenates Z/A/B. A packing ablation with exactconv/layout left on would still run packed projections. This is a concrete control/API coupling, not an explanation of the current default failure.
- **The existing intermediate probe is not automatically a fused probe.** `probe_batch_intermediates.py:301-317` constructs functional components directly. `run_fused_probe.py:143-147` patches `FunctionalDecoder` and asserts a decoder was constructed. Merely wrapping that intermediate probe does not select fused components. Its exact-prefix controls are useful designs, but instrumentation must target the actual fused objects/functions.
- **Weighted expert reduction has a different rounding sequence.** The functional path materializes BF16 `down * score` before expert summation; the fused reducer combines weighting and accumulation before final output packing (`fused_decoder.py:220-248`). This cannot change router IDs because it is downstream of selection. Since it remains active in the QKV-only pass, it is not a necessary repair for this fixture. Avoid declaring global bitwise equivalence from whole-output PCC.
- **The weighted reducer's counts and ownership are coherent for this case.** On the 1×1 mesh, `down` is logical `[1,512,1,2560]`, physically `[1,512,32,2560]`; indices/scores are `[1,1,1,512]`, mapping is `[1,512]` with owner 0. Each of 80 output tiles consumes 512 expert tiles in 64 groups of eight, including the final experts 504–511. Reader and compute advance the same counts; there is no zero-score skip or guessed `nnz` count. See `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/` program factory and its reader/compute kernels. The same geometry applies to passing request 0 and failing request 3.
- **Fused SiLU/multiply does retain the BF16 activation intermediate.** `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp:1197-1211` assigns the preprocessing intermediate the BF16 input format, and `kernels/compute/eltwise_utils_sfpu.hpp:27-51` packs activation into that intermediate before the multiply reloads it. Thus a claim that this selected activation fusion necessarily removes the BF16 rounding boundary is unsupported. Packed shared gate/up/route columns are sliced in the same order they were concatenated.
- **The old native-convolution diagnosis is resolved separately.** Current defaults set `fuse_gdn_conv=False`, `fuse_gdn_conv_exact=True`; the old native FPU/SiLU boundary discrepancy in `../autofix_ci/AUTODEBUG.md` does not execute here. Original RMSNorm fusion is also disabled. Neither should be revived as the headline without new contrary evidence.

## Claim review

The report was rechecked against the matched controls, selected paths, and restoration/completion boundaries, including an independent review and the subsequently completed sum/group1 control. The headline is limited to the QKV reduction replacement. Arithmetic order remains a ranked mechanism rather than a proven fresh coordinate-level root cause; group64 is no longer a necessary trigger, while kernel/trace alternatives remain explicit. Current expert switching, an intrinsically batch32 kernel defect, omitted state restoration, and recurrence corruption are not claimed. The unchanged request-16 carry and passing synthetic/single-request/continuation cases are retained as constraints rather than dismissed.
