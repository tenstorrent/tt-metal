# AutoDebug: shared layer-3 QSA prefill failures

## Finding

**The leading divergence boundary is attention gated-residual (GR) down-projection accumulation and its BF16 rounding, upstream of QSA.** The newly available, exact-fixture CPU control strongly supports this boundary: changing only HF `attn_hyper_connection.input_mix_weight_down` accumulation to FP64 reproduces the two problematic literal-HF rows, while the saved TT outputs closely match that control. Changing only MLP GR-down does not move either failure.

This is strong evidence of arithmetic sensitivity at a shared upstream boundary, **not yet proof of the first differing TT intermediate or a specific incorrect kernel instruction**. The cached device artifacts contain final metric operands, not normalized GR inputs, projection outputs, or expert IDs. No obvious logical, shape, ownership, or tail-write bug explaining these rows was established by source inspection. A BF16 rounding change amplified by downstream computation, potentially MoE routing, is the leading mechanism; the actual downstream amplification point remains unobserved on these fixtures.

**Every original literal-HF failure remains failed.** Baseline agreement, fused-policy byte equality, and agreement with a counterfactual reference do not establish acceptance. Both whole-output and contribution per-row gates fail here; this is stronger than the historical contribution-only GR discrepancy.

## Scope and evidence identity

Investigation date: 2026-11-13. This session and its three source/evidence reviewers performed inspection and host-only checks of existing artifacts. No target forward was executed, TTNN imported, device opened, implementation edited, or gate altered. The AutoDebug launcher was not invoked. This report is the only file authored by this investigation.

Current source identities:

| Source | SHA256 |
|---|---|
| `tt/fused_decoder.py` | `8f8a4e2adccf66cb43135a6d1f11d8ed65d472ce1c3ce46f8e815ffb20f145ea` |
| `tt/functional_decoder.py` | `e775e5fd6a37ebaca8faadd9d4f5414b82e44e4b57e8265fae331edec94fb2d1` |
| `tests/probe_qsa_decoder.py` | `fcbb25b2325066abbfde9c2dfc97e129c7935079ab182f07cc5e003185332da6` |
| Installed HF `modeling_qwen4_exp.py` | `e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df` |

The relevant current model/probe sources match the fresh control provenance. The installed HF source is the reference named by those reports; a separate `reference/modeling_qwen4_exp.py` copy has a different hash and was not substituted. Low-level reduction inspection below describes the prepared source, not an independently verified historical device binary.

Primary artifacts are [the expanded integrated run](../integrated_writer_layer3_hf_v1.json), the five `../qsa_tail_control_<policy>_v1.json` reports and their `_selection`/`_provenance` files, [the policy comparison](../qsa_tail_control_comparison_v1.json), and existing `/home/ttuser/qwen38-flash/qsa_tail_control_<policy>_v1.pt` host-output caches.

The parent-produced [CPU counterfactual report](../qsa_tail_backend_cpu_v1.json) was also available during this investigation. Its [provenance](../qsa_tail_backend_cpu_v1_provenance.json) records completion at `2026-11-13T14:53:41.796259+00:00`, exit 0. Its report SHA256 is `071d73536d7e6c300d8e821dbc56419a9a8f9b7d29f5ba3319c9cd3715574d3f`. Its recorded input-cache hash `f39b3f8956c002e3668fd929299323061971a513236500931769676feeb4d72a` matches `qsa_tail_control_best_writer_tail2_v1.pt`; its script hash `4969c585ecef0d96b63698c14e0886938978f542d9e1aba0fdda977d26113777` matches [the inspected CPU script][cpu-control]. These controls were read, not run by this investigation.

## Direct observations

The integrated matrix covers lengths `1,2,31,32,33,34,65,66`. Only lengths 34 and 66 fail, at prefill rows 30 and 44 respectively. All other prefill rows, all decode checks, and all recorded reset/repeatability checks pass. The [gate][gate] requires finite output, aggregate PCC and contribution PCC, and **every row's** PCC and contribution PCC to be at least 0.995.

All five fresh controls reproduce exactly those two failing coordinates:

| Policy | Length 34, row 30: output / contribution PCC | Length 66, row 44: output / contribution PCC | Raw result |
|---|---:|---:|---|
| `baseline` | 0.9936982393 / 0.9935274124 | 0.9931198359 / 0.9927526116 | failed |
| `best_exact_qkv_cleanup` (v5) | 0.9936800599 / 0.9935082793 | 0.9931327701 / 0.9927660823 | failed |
| `best_writer_v5` | same as v5 | same as v5 | failed |
| `best_tail2` | same as v5 | same as v5 | failed |
| `best_writer_tail2` | same as v5 | same as v5 | failed |

The control labels are relative to the v5 policy: writer toggles `fuse_residual_write`; tail toggles `compact_prefill_limit` from 1 to 2. Their remaining policy dictionaries agree. They are not otherwise unfused single-optimization models.

Independent CPU-only cache inspection verified:

- All eight cached metric records per policy agree exactly with the corresponding report records: two full prefills and six first-round decode outputs.
- Inputs/residuals and literal reference tensors have identical bytes across all five policies.
- All eight actual tensors have identical bytes across the four fused policies. The functional baseline differs; at the two failing rows baseline-versus-fused output PCC is 0.9999897480 and 0.9999851584.
- Aggregate prefill correlations pass, but the required row gates above fail. The lowest fresh-control decode contribution PCC is approximately 0.99835449; reset replay, all five state comparisons, and repeated prefill equality are true.

The [exporter][exporter] clones existing host metric operands after the original metric calculation; it does not feed intermediates back into the model.

### These are not established tail-length failures

The [fixture generator][fixture] seeds with `1733 + length`. Length 33 versus 34, or 65 versus 66, therefore changes the input fixture rather than adding a token to a fixed prefix. Passing neighboring lengths do not isolate a length-triggered defect.

The inherited [prefill loop][prefill] computes complete decoder chunks of at most 32 rows, then writes each chunk to the output:

| Observation | Full chunk containing the row | GR/QSA logical count | Actual short tail |
|---|---|---:|---|
| length 34, row 30 | rows 0–31, local row 30 | 32 | rows 32–33 |
| length 66, row 44 | rows 32–63, local row 12 | 32 | rows 64–65 |
| passing length 32 | rows 0–31 | 32 | none |
| passing length 65 | rows 0–31 and 32–63 | 32 | row 64 |

The [compact MoE predicate][compact] only changes the direct execution path for chunks with at most two rows. The failed full chunks use the same branch under both tail policies. The [output writer][chunk-write] writes only `T` logical rows at `offset + row`: the two-row tails target 32–33 and 64–65, not 30 or 44. This source audit plus the complete byte-identical policy controls strongly demotes the proposed writer/tail regression; it is not a universal proof that every possible corruption is impossible.

## Exact-fixture causal control now available

The CPU script keeps the HF model, weights, fixture generator, fresh cache semantics, and BF16 projection output boundary. Its only intervention is `F.linear(x.double(), weight.double()).to(x.dtype)` at selected GR projection modules. It asserts exact prompt identity and an independently regenerated, bit-exact literal output before comparing policies ([lines 30–59][cpu-control]). There is no forced expert selection or replacement device input.

The following are **saved TT versus CPU-control** metrics at the two original failing coordinates:

| HF intervention | Length 34 row 30: output / contribution PCC | Length 66 row 44: output / contribution PCC |
|---|---:|---:|
| literal rerun | 0.9936800599 / 0.9935082793 | 0.9931327701 / 0.9927660823 |
| attention down only, FP64 | **0.9999622107 / 0.9999610782** | **0.9999626875 / 0.9999607801** |
| MLP down only, FP64 | 0.9936800599 / 0.9935082793 | 0.9931327701 / 0.9927660823 |
| both down projections, FP64 | 0.9999622107 / 0.9999610782 | 0.9999626875 / 0.9999607801 |
| all four down/up projections, FP64 | **0.9936839342 / 0.9935130477** | 0.9999619722 / 0.9999602437 |

Attention-down-only also reproduces the failing rows when compared against literal HF: output/contribution PCC is 0.9937036633/0.9935326576 at length34 row30 and 0.9931012988/0.9927332401 at length66 row44. Every saved-TT/control prefill row clears 0.995 for that intervention; the output/contribution minima over all rows are 0.9986056685/0.9985680580 for length34 and 0.9970691204/0.9969674945 for length66. Thus the effect is not merely an improvement of the two reported rows with an omitted new failure elsewhere in these prefills.

**Critical limitation:** all-GR FP64 brings back the length34 discrepancy. Adding the two up-projection changes can alter downstream rounding again; this control does not distinguish attention-up from MLP-up. Also, `all_gr_fp64` does not change the separate injection projection. FP64 is neither the functional balanced FP32 tree nor the fused SFPU reduction. There is no monotonic “more precision fixes the model” result here.

This report records only prefill comparisons from the CPU control. Its helper constructs decode fixtures, but does not report counterfactual decode comparisons. The original device decode/reset evidence remains separate.

## Shared computation and lowered arithmetic

The public layer3 chain is [the inherited `_chunk` implementation][chunk]:

```text
BF16 residual [1,1,Q,10240]
  → attention GR grouped RMSNorm
  → GR down 10240→320 → /4 → SiLU → GR up 320→10240 → sigmoid
  → BF16 per-stream product, four-stream mean → mixed attention input [1,1,Q,2560]
  → QSA → attention GR residual write
  → MLP GR read → router/top10/expert computation → MLP GR residual write
```

Layer3 has indexed attention and no PLE. [TT GR read][gr] agrees logically with [installed HF GR][hf-gr]: the group dimension is four streams of 2560, both learned down/up projections and their BF16 boundaries are present, and the injection branch is separate. All compared fused policies leave the GR normalization fusion options disabled. The writer subclass inherits this same read path. No omission or transpose error was found.

There are, however, real arithmetic-order differences:

| Path | Down/up accumulation and output boundary |
|---|---|
| Literal HF | BF16 `nn.Linear` at each projection on the full HF call shape; down is followed by BF16 `/4` and SiLU, up by sigmoid. Backend reduction order is not specified by the Python formula. |
| Functional TT | [FP32 products plus an explicit half-split addition tree][functional-projection], including odd-width tails, then BF16 cast. |
| Fused TT | [FP32 products plus accurate `ttnn.sum`][fused-projection], then BF16 cast. GR replacement [does not enable `balanced_tree`][gr-replace]. `preserve_qkv_tree` is for the linear-attention GDN QKV path, not layer3 GR. |

At each failing full chunk, GR-down multiplies FP32 `[1,32,1,10240]` by FP32 `[1,1,32,10240]`, producing `[1,32,32,10240]` for each of ten output blocks. GR-up produces `[1,32,256,320]` for each of forty blocks. Passing full chunks use those same geometries. Both widths are exact multiples of 32; there is no partial reduction-width tile. The BF16 input widening and the FP32 checkpoint-weight storage mean `fuse_projection_cast` does not create a mixed-format GR multiply.

For the recorded P150 path, the current C++ [dispatch condition][reduce-dispatch] selects the accurate SFPU sum for FP32 input with FP32 destination accumulation and `fast_and_approximate_mode=False`. The [width factory][reduce-factory] derives `Wt=K/32`, selects `reduce.cpp`, and passes the accurate-mode flag. The [compute kernel][reduce-kernel] uses the per-tile wait/pop policy. The [helper][reduce-helper] seeds from the first tile, folds later tiles into the FP32 accumulator in order, then reduces within the accumulated tile: 320 width tiles for down, 10 for up. It waits/pops each input tile once and commits/waits before packing the result. This differs from the explicit functional half-tree. No violated producer/consumer count or incomplete block was found in this inspected path.

The direct-upload ownership path also checks out: a fresh functional model is constructed before fused replacement; replacement releases that projector's old weight list. Down's 320 outputs split into ten complete 32-output prefill blocks, and up's 10240 outputs split into forty complete 256-output blocks. These checks demote a shape/ownership explanation; they do not prove numerical equivalence to the HF backend.

The likely causal chain is therefore a changed BF16 value at attention GR-down, propagated through the up/gating/mixing operations, QSA and the later MLP. A discrete expert-set change would explain a small upstream difference producing an isolated larger row error, but **no current exact-fixture router capture establishes that link**. Neither normalization equality before down nor the exact first differing projection coordinate is available. The CPU intervention localizes sensitivity, not every runtime mismatch.

## What the older functional diagnostics establish

[GR same-operand rounding](../../functional_decoder/gr_same_operand_rounding_cpu.json) contains a concrete **layer0 request16 prefill**, attention-down example at `[0,0,34]`:

| Quantity | Value |
|---|---:|
| FP64 dot product | 2.1640585417408147 |
| BF16 midpoint | 2.1640625 |
| Literal HF BF16 projection | 2.171875 |
| Balanced FP32-tree raw value | 2.1640586853027344 |
| Tree/FP64 BF16 result | 2.15625 |

Only one of 640 attention-down outputs differs in that case. [The old backend control](../../functional_decoder/gr_backend_sensitivity_cpu.json) changes only attention-down accumulation to FP32, FP64, or the explicit tree. All three reproduce a row0 contribution PCC of 0.9906104207 against literal HF and replace expert 405 with 365 in the selected set, agreeing with the recorded TT set. Request16 decode has no same-operand down mismatch. This is direct prior evidence for rounding followed by routing amplification, **on different layer weights and fixtures**.

The older [GR projection diagnostic](../../functional_decoder/gr_projection_intermediates.json.xz) and [arithmetic diagnostic](../../functional_decoder/gr_arithmetic_exact_inputs.json) also explain why the current implementation already uses precise down/up projection and FP32 nonlinear/reduction work with explicit BF16 boundaries. Those old native-arithmetic problems must not be described as unfixed omissions in current source. The [GR row-split control](../../functional_decoder/gr_query_rows_control_v1.json) is a layer1 component control across Q1..32, not an exact current layer3 fixture result.

Historical QSA backend work is useful methodology, but a different mechanism: [the documented indexer investigation](../../functional_decoder/onednn_projection_order.md) concerns the `[32,2560]×[2560,640]` indexer projection, tiny top-k cutoffs at long context, and a source-derived BF16-backend accumulation order. Its K-block/shape rule cannot be transplanted to GR-down `[34 or 66,10240]×[10240,320]` without its own dispatch and same-operand proof. A prior uniform FP64 indexer control improved one long-context row and introduced another failure; the current all-GR reversal reinforces the need for narrow controls.

The functional-stage [acceptance discussion][old-acceptance] retained failures and required literal whole-output passes for its documented fresh-contribution classification. The present rows fail literal **whole-output** PCC too. That historical classification is not an applicable acceptance exception here.

## Alternative explanations and unresolved issues

| Candidate | Evidence against or missing prerequisite | Smallest discriminator / status |
|---|---|---|
| New residual writer or tail2 MoE regression | Disabling either/both leaves every fused output byte unchanged; baseline fails same rows; failed rows are full chunks. | Already strongly demoted for this matrix. No writer/tail repair is supported. |
| Old QSA sparse-cutoff selection defect | At position30 there are only 7 complete four-token groups plus 3 tail tokens; position44 has 11 groups plus 1 tail token. All fit within the 512-group budget. | A score perturbation cannot prune a valid group under the finite-score/index contracts. Selection order can still change finite-precision attention arithmetic; it is not proven bit-exact. |
| Fused GR accurate-sum bug as sole cause | Functional explicit-tree baseline fails the same rows. Shapes and tile counts are coherent; exact-fixture HF attention-down intervention reproduces the symptom without this fused kernel. | Possible extra fused differences remain, but no concrete kernel violation was found. Compare exact operands if needed. |
| MLP GR-down alone | The exact-fixture MLP-down-only FP64 control leaves both failed coordinates unchanged. | Demoted as the isolated intervention; downstream MLP/up/router sensitivity remains possible. |
| Up-projection arithmetic interaction | All-GR FP64 reverses length34 recovery, but changes two up projections at once. | Unresolved. Add attention-up and MLP-up separately to the both-down control. |
| Shared normalization, attention, router or expert error | Final-output caches contain no intermediate localization; baseline equality cannot exclude shared faults. | Capture first divergences before naming a concrete TT implementation bug. Attention-down evidence makes this a targeted next step. |
| Trace/reset or asynchronous failure | Prefill synchronizes before readback; decode replay is blocking; all reset/repeat checks pass. | No sequence-matched evidence for stale state. Passing replay equality is not HF accuracy. |

The short-context selection argument is directly supported by [functional selection][selection], [the fused heap kernel][fused-selection], and [HF selection][hf-selection]. At 7 or 11 groups the fused heap never reaches its replacement branch; each completed group is emitted once. This rules against the **old top-k membership cutoff mechanism**, not every possible attention arithmetic or cache defect.

Synchronization was traced rather than inferred from successful assertions: [prefill and warmup][sync] explicitly synchronize before readback/restoration; replay uses `blocking=True`, then reads output and state before reuse. Repeated prefill is checked through host readback after initial-state restoration. The result [combines metrics with reset checks][final-gate] and exits nonzero for these failures.

## Minimal discriminating follow-up

The parent’s planned attention-down-only FP64 intervention is already the smallest completed causal control. Preserve it and its literal comparisons. **Do not broaden the implementation change to all GR projections or use its output as a replacement oracle.**

To turn the leading boundary into a first-divergence diagnosis, the smallest useful next experiment is an observer-only capture on the same two saved prompts:

1. Record literal HF and attention-down-only FP64 normalized inputs, actual BF16 down outputs, activated inputs, up outputs, mixed attention inputs, then router logits and paired top10 IDs/weights. Record raw and BF16-rounded FP64/tree control outputs on identical normalized operands; ordinary hooks cannot expose the literal BF16 backend's internal pre-round accumulator. Preserve the original full-call shapes, all rows, prompt/oracle hashes, and original gates. A changed MoE set would identify a discrete divergence; compare routing weights and MoE outputs to establish its amplification. Unchanged sets would require following continuous downstream error instead.
2. For functional TT arithmetic, run a separate CPU control changing only attention-down to the existing [explicit `tree_projection`][cpu-tree], retaining BF16 output. This emulates the functional reducer's addition structure, **not the fused accurate sum**. Compare both the original functional-cache outputs and fused-cache outputs, with whole/contribution per-row minima. If necessary, independently compare literal BF16 projection on the full M=34/66 operands against the same operands split into M=32/2 calls; this distinguishes a call-shape/backend effect without changing values or dtype. Do not infer GR backend order from the old indexer rule.
3. If the first actual TT/HF mismatch is still needed, have the parent capture normalized input and down output at the public runtime boundaries, with global/chunk/local row identity. Compare projection outputs on **identical operands** before blaming accumulation. If normalized inputs already differ, move localization earlier. A selected row’s QSA result may also depend on changed earlier rows, so retain the full prefix rather than only the failed token.

The all-GR length34 reversal is a secondary, tightly bounded control: starting from both-down FP64, add only attention-up FP64, then independently only MLP-up FP64. Record which intermediate and route changes; neither intervention has been run by this investigator. A matched-prefix length extension is only needed if the parent continues to suspect length/tail sensitivity; the existing length-seeded fixtures cannot settle that question.

The smallest eventual implementation boundary, **if identical-input captures confirm it**, is the shared attention GR-down arithmetic policy. Exact literal-backend matching would need shape-specific, source-supported accumulation and BF16 rounding evidence across holdout inputs, not per-row corrections, forced expert IDs, or a tolerance change. No implementation patch is justified solely by final-output counterfactual agreement.

## Post-draft claim review

Each headline claim was checked again against source reachability and the full named matrix:

- **Attention-down is causally sensitive on these fixtures:** supported by a one-module intervention with exact literal/input identity and all-row comparisons. **First TT divergence occurs there:** still unproven; kept qualified.
- **Writer/tail causes these failures:** contradicted by the matched policy outputs and full-chunk coordinates; not headlined.
- **A sparse indexer cutoff changes selected membership here:** incompatible with 7/11 complete groups below budget; demoted. Reordering and other attention arithmetic remain possible.
- **A particular expert changes on row30/44:** absent from the current caches; retained only as an inferred mechanism, unlike the directly observed historical request16 route change.
- **All FP64 projection policies improve agreement:** contradicted by length34 all-GR FP64; explicitly rejected.
- **Baseline/control agreement permits acceptance:** contradicted by the unchanged required literal per-row gates and the historical exception's scope. Raw reports remain failed.

[gate]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_qsa_decoder.py:61
[fixture]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_qsa_decoder.py:114
[exporter]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_fused_qsa_controls.py:30
[cpu-control]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_fused_qsa_backend_cpu.py:30
[prefill]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py:157
[chunk]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/functional_decoder.py:123
[compact]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:988
[chunk-write]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/chunk_output.py:29
[gr]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/decoder_components.py:64
[hf-gr]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:995
[functional-projection]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/precise_residual_projection.py:44
[fused-projection]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:77
[gr-replace]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:2688
[reduce-dispatch]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op.cpp:198
[reduce-factory]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp:577
[reduce-kernel]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce.cpp:18
[reduce-helper]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:500
[old-acceptance]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/README.md:294
[selection]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/qsa_chunked.py:82
[fused-selection]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tt/fused_decoder.py:1251
[hf-selection]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:726
[sync]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_qsa_decoder.py:190
[final-gate]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_qsa_decoder.py:253
[cpu-tree]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_gr_rounding.py:22
