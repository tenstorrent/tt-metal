# AutoDebug: Qwen3.8-27B stage06 long free-running repetition

Inspection date: 2026-10-25. Checkout HEAD: `cdcf1091d55e793f68fd46acdbe80372337e1c34` plus the existing working-tree changes. Scope: `models/autoports/qwen_qwen3_8_27b` and the TTNN operations reached by its production paths.

## Finding

**The B1 quality failure is confirmed, but this inspection does not establish a kernel or model-math root cause.** The strongest localization is a greedy decision on an identical history: TT and HF produce the same first **56 generated token IDs**, then TT selects token **279** where HF selects token **1**, closing its quotation. TT subsequently remains in an exact 15-token cycle. Calling this simply “acceptable quantization drift” would be unsupported; calling the exact4096 fused projection or fused linear-attention kernel broken would also overstate the evidence.

There is one concrete semantic discrepancy to check first: **the selected TT top-k pipeline does not guarantee HF's lowest-index tie break.** It can select a different token from an equal-valued maximum set. No saved logits show whether that prerequisite holds at the observed divergence, so it is a conditional cause, not a diagnosed cause. A same-TT-logit host/device comparison at output index56 tests whether sampler ordering changes the selection; an HF logit comparison separately measures model-distribution drift.

The B1/B8/B32 contrast has several simultaneous implementation differences, including an additional one below Python: **current source predicts fused chunk-GDN prefill for B1 and phased chunk-GDN for the large batches.** That dispatch prediction is not a captured runtime kernel identity. This is not an isolated precision experiment. Stage05's strong decoder PCC evidence compares against an already low-precision TTNN baseline and cannot certify this 64-layer free-running behavior.

No TTNN/device runtime, model inference, device discovery, reset, process termination, target compilation, or implementation edit was performed. The only requested output is this report. The device-owning pipeline was left untouched. Proposed device controls below are future work, not results.

## 1. Direct observations and revision limits

Here and below, model-relative paths start at `models/autoports/qwen_qwen3_8_27b/`.

| Evidence | What it directly establishes | What it does not establish |
|---|---|---|
| `doc/full_model/benchmark_b1_perf.json`, `throughput_fixture_qualitative_control` | Current all64 B1 generates512 tokens with `stop_on_eos=true`; it never emits a reasoning closer and repeatedly quotes the research sentence. | The first layer/op where logits became different from HF. |
| Same file, three4096/128 records | Repeated runs have identical output hashes; their decoded128-token completion is a prefix of the512-token completion. The loop already occurs before the native-context capacity check. | Determinism is not proof of correct arithmetic or cache contents. |
| `doc/full_model/benchmark_hf_control_512.json` | Exact4096 input IDs, greedy BF16 HF control,512 outputs; `</think>` token248069 at generated index214, followed by coherent prose. Neither checkpoint EOS ID248044/248046 occurs before the cap. | A fully completed answer/EOS within512 tokens; the final prose is itself truncated. |
| `benchmark_hf_control.json` | Its128 output IDs equal the first128 IDs of the512 HF control. | A128-token completion is not expected to have left reasoning: HF itself has not. |
| `benchmark_b8_tracking_final.json`, `benchmark_b32_tracking_final.json` | Same4096 prompt hash;128-token continuations move beyond literal quotation toward planning; identical slots within each run agree. | Neither is a512-token quality pass, nor an exact B1/B8/B32 arithmetic comparison. |
| `prefill_accuracy.json`, `teacher_forcing_accuracy.json` | The short AIME fixture reports99/100 top1,100/100 top5/top100. | Long repeated-text accuracy; long free-running stability; solved AIME accuracy. |
| `qualitative/*`, `qualitative_verdict.md` | Short shared prompts show coherent behavior; the verdict correctly leaves stage06 incomplete for this new failure. | A waiver for the throughput prompt's quality failure. |

Static comparisons of the saved JSON give:

- HF input ID hash, using the benchmark's `sha256(json.dumps(ids).encode())` convention: `ec5606540969599d781d96ad892b0cded3f605d8f000669886cfaad9411899e2`. It matches the B1/B8/B16/B32 **4096** records. B32's32768 record is a different prompt and must stay separate.
- TT512 and HF512 match at indices0..55. At56: TT279, HF1. This is the first observed token difference, not necessarily the first numerical difference.
- TT IDs47..511 are exactly31 repetitions of `[561,3302,2015,7189,1754,21254,15060,321,11988,279,5721,1518,12945,29256,13]`. The full decoded TT512 output contains32 complete copies of the sentence.
- B8/B16 have the same first312 decoded characters as B1/HF, then close their quotation earlier; B32 differs after116 characters. Those artifacts store text/hashes rather than raw generated IDs, so these are character comparisons, not invented token alignments.

Useful raw evidence anchors: [TT512 IDs](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/full_model/benchmark_b1_perf.json:27858), [HF512 IDs](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/full_model/benchmark_hf_control_512.json:4103). In the inspected files TT output56 is at line27915; HF output56 at4160 and reasoning closer at4318.

All12 source entries in B1's `source_sha256` match the inspected files: the benchmark, ten `tt/*.py` modules, and rotary program factory. In particular:

| Item | SHA256 |
|---|---|
| Current generator | `6a24b49785f02bef6dcd10a31f618a550fa53d87576a03a71de7f474eef46c5c` |
| Multichip decoder | `46b8d305f5c10e96dc7dc1aa404e3a92b29bb8469cd6d560ac6e61fb204cd24f` |
| Rotary factory | `be0d9ed714cfd13cb8129985fe8f9a5a2b92cdbe184587cc38b6c545d72ba54c` |
| Recorded loaded C++ library | `aca153162839c6b5134ac9b446dd32d1574441a9c58fd6bc9a6ffceec17028d4` |

The last row is **recorded provenance**, from `rotary_build_provenance.json` and the benchmark, not an independently inspected live process. B8/B32 tracking artifacts record older generator `babb8b76579a384d30c1f9ce5c7a9ce80b109f3b117be9bfc01aa85998581d00`; B16 records `f89290d1538a6f1476ba16684ac1cd80874b4a30250f63b6e458cb0aa3f6fdd4`. Their recorded model/math/rotary files match current. Shared sampling code and most lowered C++ sources are not individually hashed by these artifacts. Thus current source analysis is not complete transitive build provenance, and the batch comparison should be refreshed on one frozen revision before causal attribution.

## 2. Prompt, wrapper, cache, and completion audit

**Prompt/sampling setup.** `tests/benchmark_full_model.py:41–51` and `tests/hf_benchmark_control.py:14–22` construct the same repeated-text body and apply the HF chat template, adjusting body length until the *rendered* prompt is4096 tokens. The saved hash confirms equality; no bare-prompt/chat-template mismatch is needed to explain this failure. HF explicitly uses `.float().argmax()` (`hf_benchmark_control.py:27–36`); TT explicitly uses k1/p0/temp1 (`tt/generator.py:58–64,127–132`). Although checkpoint generation defaults enable sampling, neither control uses those defaults. Neither path applies a repetition penalty.

**Actual runtime dimensions.** `Generator.physical_length` rounds non-singletons to32 (`generator.py:356–358`). This4096 prompt remains4096: one prepared chunk, all4096 validity entries true, positions0..4095, final hidden selection4095, history selection corresponding to the last three real projected rows,128 **used** KV pages of32 tokens. The B1 context262144 pool allocates8192 pages; used prefix and allocated capacity are distinct. This prompt does not exercise a masked tail. A203-token prompt instead becomes224 physical rows. Nearby4095 also becomes4096 and therefore still selects the4096 native projection; “test4095” alone does not disable that branch (`generator.py:381–393,422–453`).

**Token and position order.** Prefill supplies output y0. The generator copies final logits into persistent sampling storage, samples y0, and initializes decode positions to4096. The next model replay consumes y0 at4096, increments positions and RoPE indices, and samples y1. Subsequent steps repeat this sequence (`generator.py:114–132,538–587`). Consequently output y56 is predicted after consuming y55 at absolute position**4151**; the mutable position/RoPE buffers read4152 after that replay's increment. The first56 shared output tokens provide an exact common history for that comparison. There is no source-level duplicated-token feedback or one-step position error in this path.

**Stopping.** EOS comes from both checkpoint IDs, and normal generation checks each sampled token (`generator.py:18–21,576–583`). TT512 does not reach an EOS or a reasoning closer. HF's fixed-length mode cannot explain this difference because its512 outputs contain no EOS either. `</think>` is a generated content token, not an EOS setting.

**Reset/ownership.** The benchmark uses owned caches and the default valid map. Each `generate` calls `reset`; this is distinct from the repaired external sparse-map/preparation paths. The identical B1_128 behavior before native-context checks and its prefix agreement with the later TT512 output argue against blaming the intervening capacity check alone. They do not prove every cache value numerically correct. Existing feedback/page-map/slot probes remain evidence about their tested scopes, not the exact all64/4096 decision.

**Completion boundaries.** Model and sampling traces enqueue nonblocking on CQ0; the token `to_torch` readback at `generator.py:572` is the relevant per-step host observation. Host `DecodeAllReduceWorkspace.finish*()` checks balanced enqueue order only, not device completion. It must not be used as a fence in a diagnostic. Any state/logit capture must synchronize or read back the intended queued result before comparing it or restoring scratch/state. Current benchmark completions are actual readback tokens, not merely successful host enqueue returns.

## 3. The batch contrast in the inspected source

The full model directly constructs `MultichipDecoder` for every layer and shares one workspace (`tt/model.py:31–49`). It inherits stage05 policy instead of silently replacing it. Embedding and terminal tuple-key gathers preserve BF16 (`model.py:65–88`; `multichip_decoder.py:278–301`); the vocabulary head is BF16 with HiFi4/FP32 accumulation. B1/B8/B32 use the same checkpoint weights and nominal projection policy, but not identical algorithms or rounding boundaries. The table traces current source selection; the GDN and SDPA factory details still require runtime identity capture for a definitive experiment.

| Component | B1 at4096+decode | B8/B32 at4096+decode | Source |
|---|---|---|---|
| Prefill residual | Hidden-sharded1280/device | Same | `multichip_decoder.py:209–214,230–245` |
| Prefill out/down | Native fused MM/RS, BF8 MM output | Separate BF16-output MM, BF8 cast, RS | `multichip_decoder.py:389–412`; `fused_prefill.py:21–40`; `optimized_decoder.py:293` |
| Chunk-GDN | Automatic fused prep/scan | Automatic phased prep/scan; B32 subdivided | See section5 |
| Full-attention mix gather | L1, because B×S=4096 | DRAM, because B×S>4096 | `multichip_decoder.py:76–81,274–285` |
| Prefill RoPE | Separate user/head layout | Packed rows from B8 | `multichip_decoder.py:42–43,715–735` |
| Decode residual/norm | Replicated5120; ordinary sharded RMSNorm | Sharded1280; distributed pre/gather/post RMSNorm | `multichip_decoder.py:193–225,424–475` |
| Decode row projections | Ordinary MM, BF8 payload all-reduce, BF16 result | Native fused MM/RS with BF16 buffers/payload | `multichip_decoder.py:303–348,395–412`; `fused_decode.py:27–53` |
| Decode linear convolution | Per-user path | Four temporal rows/user packed from B8 | `multichip_decoder.py:537–583` |
| Decode RoPE | Ordinary | Packed from B4 | `multichip_decoder.py:42,715–735` |
| Decode SDPA |32 cores/head, shared LoFi/FP32-off policy |13 cores/head at B8;3 at B32 | `multichip_decoder.py:97–105,759–769`; SDPA factory below |

Same-slot equality only says rows within a given execution agree. It does not turn the right-hand column into a bit-exact B1 control. The B8/B32 behavior also does not by itself prove that B1's prefill, decode, or sampler owns the divergence.

## 4. Concrete sampler discrepancy: ties are not HF-equivalent

The canonical selected path is:

1. BF16 local logits, width248320/4=62080; pad each shard to65536 with negative-infinity-equivalent padding.
2. Local `ttnn.topk(k=32)` with default `stable=False`, no custom top-k core grid.
3. Gather four candidate lists and add **unpadded** shard offsets, obtaining global token IDs.
4. `ttnn.sampling` selects with k1/p0/temp1 and writes the persistent feedback tensor.

Sources: `models/common/modules/sampling/sampling_1d.py:60–80,400–455,594–642`; `tt/generator.py:58–64,127–132`. No incorrect62080-versus65536 vocabulary offset was found.

Only `stable=True` promises lowest-original-index ties in [topk_nanobind.cpp](ttnn/cpp/ttnn/operations/reduction/topk/topk_nanobind.cpp):60,120. The current Blackhole wide-input route is eligible for this BF16/tiled/interleaved65536/k32 shape and explicitly has deterministic but unspecified tie order ([topk.cpp](ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp):282–373). This is current-source dispatch, not a recorded kernel identity from the measured executable.

The next sort's `stable_sort=true` does **not** repair original-vocabulary ordering. The sampling reader generates new candidate-position indices separately from the supplied global token IDs; the final writer maps the chosen position back through those IDs:

- `ttnn/cpp/ttnn/operations/reduction/sampling/device/sampling_program_factory.cpp:126–128`;
- `.../kernels/dataflow/reader_values_indices_tensor.cpp:38–61`;
- `.../kernels/dataflow/writer_interleaved.cpp:203–209`.

Thus, **if** full-vocabulary host argmax is1, TT279 has the same maximum value, and local top-k puts279 first, device k1 can choose279 while host argmax chooses1. This is a real difference in semantics, even though both choices are greedy maxima. It could start an autoregressive loop without any token-feedback corruption. It does not prove a tie occurred, that current top-k actually orders this pair that way, or that choosing1 makes the full continuation coherent. A TT tie-breaking difference can coexist with TT-versus-HF numerical changes.

`tests/probe_sampling.py:18–36` verifies that the reduced real-model selected token has maximum value and separately records first-index agreement. Its synthetic cases use one unique peak (`:39–51`). It does not cover ties at the observed all64 divergence. **First test the same captured logits, not two separately generated continuations.** If TT279 is strictly larger than TT1, this candidate cannot explain that first difference. If the sampler returns a nonmaximum, that is a different sampler bug and should be localized directly.

Smallest possible intervention if a tie mismatch is proved: make the greedy tie contract deterministic by original global token ID, preserving the optimized decoder's policy and layout. A broad switch to slower generation or higher-precision weights is not implied. No implementation change is justified by this inspection alone.

## 5. Lowered audits of the main math-path candidates

### 5.1 Exact4096 native matmul/reduce-scatter

The branch guard is **physical** `[B,S]=[1,4096]`, out/down only (`multichip_decoder.py:391`). It passes A BF16, row weights BFP4, LoFi, FP32 destination disabled, requested packer-L1 accumulation, output BF8, DRAM intermediate/output, mesh1×4, dim3, two-link ring. No operand swap is introduced by the wrapper. Its explicit grid is11×8, per-core M16/N15, blockH4/W15, subblock2×3; outK48, downK17 (`fused_prefill.py:21–40`).

Concrete dimensions per device:

| Role | A | B | K tiles / blockK | Inner blocks | Per-core output traversal |
|---|---|---|---|---|---|
| Attention out, both layer kinds |4096×1536|1536×5120|48/48|1|4 M blocks ×1 N block;10 six-tile subblocks per M block|
| MLP down |4096×4352|4352×5120|136/17|8|Same|

The wrapper lowers through `experimental/ccl/matmul_reduce_scatter_async/{matmul_reduce_scatter_async.cpp,device/matmul_reduce_scatter_async_device_operation.cpp}` into `device/matmul_reduce_scatter_async_program_factory.cpp:31–132`, which combines the RS descriptor and `experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp` descriptor builder. Actual compute is the legacy `operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp`, not its `_metal2` sibling.

Static contract checks:

- `out_block_w == per_core_N ==15`; the prior undersized native-weight-reader block is not selected. The factory clamps bank-descriptor widths to the worker width (`ccl_fusion_mcast_2d.cpp:3254–3310`). Eight DRAM banks contain20 N tiles each; each full worker consumes15, the final worker10 real tiles. Discarded padded N lanes are not logical output.
- For each M block, out reads192 A tiles/720 B tiles; down reads68/255 for each of8 K blocks. Both produce60 output tiles. Compute publishes ten six-tile output subblocks and the writer drains them, dropping only padded columns on the last N worker.
- Packer-L1 accumulation is *actually disabled* for out's one K block and enabled for down's eight (`ccl_fusion_mcast_2d.cpp:1973–1984`). Down uses BF16 partials and BF8 final output. This is deliberate factory behavior, not evidence that the requested policy failed to propagate.
- The native weight reader resets its K byte offset for every M block (`reader_bmm_tile_layout_in1_sender_writer_padding.cpp:339–358`). It waits for output writes before popping subblocks and signals RS after completing that batch (`:734–773`). The older fused-AG multi-M readiness issue is a different, unselected path.
- RS uses the actual MM output tensor to build its program. Both persistent prefill buffers are correctly shaped and BF8 (`decode_workspace.py:49–56`). Its separate reduction compute config does not silently turn the LoFi matmul into HiFi4.

No new reader/CB extent contradiction was found in this selected geometry. Native and separate paths nevertheless have different K blocking, packing, and reduction schedules. Their difference should be measured on **identical input operands** before attributing an effect to either a kernel defect or benign rounding. Disabling native RS changes more than “fusion,” including BF16-then-BF8 versus direct-BF8 rounding; an improving full run alone would not isolate the cause.

### 5.2 Chunk-GDN dispatch and producer/consumer ledger

`multichip_decoder.py:563–653` produces per-rank q/k512, v1536, z1536 and12 decay/beta heads. All batches call chunk-GDN with C32, output-head-major, explicit FP32 recurrent state, HiFi4/FP32/noapprox compute. The wrapper passes no program config. `operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp:274–297` chooses fused versus phased using the core-grid/shape cost model.

For11×10 cores and12 value heads/device:

| Request | Physical S / NC | Flattened batch-heads | Selected geometry |
|---|---|---|---|
| B1 AIME203 |224 /7|12|Fused,2 scan receivers/head,4 prep producers/head,3 handoff buffers|
| B1 exact4096 |4096 /128|12|Fused,2 receivers/head,7 producers/head,3 buffers;108 cores total|
| B8 exact4096 |4096 /128|96|Phased; even one producer plus one receiver/head cannot fit110 cores|
| B32 exact4096 |4096 /128|108,108,108,60|Python batches9+9+9+5; each selects phased|

These are static dispatch predictions, not profiler observations. They were cross-checked with the pure host geometry oracle in `tests/ttnn/unit_tests/operations/transformers/test_chunk_gdn_fused_geometry.py:46–269`, extracting its definitions with AST and without importing TTNN. This is not a low-precision-versus-high-precision split. `device/chunk_gdn_compute_config.hpp:18–36` enforces the same HiFi4/FP32/noapprox math; AUTO chooses the same Blackhole/C32 SFPU forward-substitution inverse (`:48–68`). Fused-only hoisted reconfiguration is in the other Horner inverse branch, so blaming that macro for this workload is unsupported.

For failing-shape B1, Ct1/Kt4/Vt4, V split over NV2 receivers (Vtl2), NP7, no extra producer pool:

- There are12×128=1536 prep items. Producers0/1 per head own19 chunks; producers2..6 own18. Ownership is `head*7 + chunk%7`.
- Each item contains19 FP32 tiles: v-beta4, `nkd = −β·k·exp(cumulative g)`4, decayed Q4, intra1, transposed decayed K4, decay1, inverse1. Each receiver receives its two-tile V slice and the six V-independent tensors.
- V-beta's physical ring has12 tiles (six two-tile receiver slots); writer address `(chunk*2)%12` matches receiver advancement by2. Other handoff rings have three chunk slots, addressed by `chunk%3`. Two outstanding grants with a three-slot ring prevent reuse before the old chunk is popped.
- Each receiver keeps8 state tiles. State ping-pong advances chunk0 initial→s2, chunk126 s3→s2, chunk127 s2→final. The two final-state V slices are disjoint. Each chunk's two output tiles are written with full-V stride4 and receiver offset0/2.

Relevant sources under `operations/transformer/chunk_gated_delta_rule/device/`: `chunk_gdn_fused_program_factory.cpp:186–253,443–471`; `kernels/dataflow/writer_chunk_gdn_fused.cpp:264–350,380–441`; `kernels/dataflow/reader_chunk_gdn_scan.cpp:463–588`; `kernels/compute/chunk_gdn_scan.cpp:68–94`; `kernels/compute/chunk_gdn_math.hpp:1143–1178`. Data acknowledgement precedes VALID publication. The instantiated count/address ledger did not expose a missing push/pop, wrong ring extent, or overlapping final-state slice. Short203 also uses fused GDN, but with fewer chunks and different producer count: each prep producer has at most two items, whereas4096 repeatedly wraps its three-slot output ring. That is a useful test boundary, not proof of bad wrapping.

An explicit phased B1 control is useful **only as a diagnostic**, retaining C32, inverse choice, all dtypes, fidelity, inputs/state, and the rest of the decoder policy. If it differs, compare the seven prep intermediates and per-chunk state to locate the first mismatch; do not adopt disabled fusion as the final fix. Existing fused/phased bit-exact tests construct host-normalized head-shaped Q/K (`test_chunk_gdn_fused.py:102–146`), with BH12 default/repeat coverage at NC8/64 (`:298–299,825–851`). Handoff tests cover NP7/NC8 and autoNC64 (`test_chunk_gdn_handoff_checks.py:34–37`). These are source-defined tests, not results run here, and do not substitute for production raw flat Q/K plus in-kernel normalization at NC128. If needed, pin fused NV2/NP4 versus NV2/NP7 at NC128 to isolate producer scheduling.

### 5.3 Decode all-reduce and shared scratch

B1 row projections produce BF16, explicitly assign to BF8 width-sharded `[1,1,32,5120]`, then call the minimal `all_reduce_async` overload with BF16 output (`multichip_decoder.py:315–341`). `fp32_dest_acc` defaults false in `experimental/ccl/all_reduce_async/all_reduce_async_nanobind.cpp:108–136`.

Eight output cores each own20 tiles. Each scratch shard holds4×20=80 BF8 tiles. Rank offsets are fixed, not inferred from arrival order (`.../device/kernels/dataflow/worker_writer.cpp:117–145`). The reduction receiver waits for all four contributions and publishes80 tiles; compute visits four source blocks in pairs, emits20 tiles in groups8/8/4, and never reaches its unsupported odd-ring branch (`.../compute/reduction.cpp:20–67`). Input/output CB formats are separately derived in the factory (`all_reduce_async_program_factory.cpp:248–249,359–421`). No mixed-BF8/BF16 format contradiction was found.

The two out/down scratch roles are retained and alternate across complete layers (`tt/decode_workspace.py`). Their host state machine does not itself synchronize devices. The intervening collective/data dependencies are part of the reuse contract; nothing in this inspection establishes a new premature-reuse violation. A serialized fresh-buffer control could test that boundary if exact-prefix state inspection points there, but repeated deterministic sentence generation alone is not evidence of a scratch race.

### 5.4 Full attention, MLP, norms, and recurrence

Module-by-module comparison used the local HF source at `/home/mvasiljev/.cache/uv/archive-v0/l767Qehgs-M1kBL9OGkok/transformers/models/qwen3_5/modeling_qwen3_5.py` and the functional/fused/optimized/TP4 implementations. No omitted gate, incorrect residual order, or head-axis semantic mismatch was found:

- Full attention uses per-head interleaved Q/gate weights, Q/K RMSNorm with1+weight,64 rotary dimensions of256, GQA6:1 per device, scale1/16, and sigmoid gating after attention (`multichip_decoder.py:170–178,715–808`; HF:683–716).
- Per-rank full-attention projection3584 = Q1536+K256+V256+gate1536. Decode consumes Q `[1,B,6,256]` and one K/V head/device. KV is BF8 `[pages,1,32,256]` for every batch (`multichip_decoder.py:477–492,738–769`).
- MLP partitions17408 into4352 channels/device, applies SiLU(gate)×up, then reduces the down projection; norm/residual sequence matches HF (`multichip_decoder.py:494–534`; HF:720–750,778–807).
- Linear attention repeats4 local Q/K heads to12 value heads, normalizes Q/K, computes negative softplus decay and sigmoid beta, preserves convolution history, applies gated output norm/SiLU, and updates FP32 recurrence. The saved reference `doc/functional_decoder/hf_modeling_reference.py:358–411,465–477,554–610` agrees with that logical order. The fused gated norm yields norm(y)×sigmoid(z), followed by the explicit z multiply, so the apparent extra multiply is the required SiLU factor. The older direct-to-DST state precision repair remains present (`tt/kernels/fp32_recurrence.cpp:137–162,205–245`); historical long-linear or rotary diagnoses must not be reused as the current cause.

Decode SDPA's32/13/3 cores/head for B1/B8/B32 follow `operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:201–216`. At positions4096 through4151 there are17 chunks of256; B1 has at most one per participating core, B8 two, B32 six. A claim that B1 alone has the longest serial softmax update chain is false. Tree combination order can still affect numerics. The current page-capacity chunk guard, common workload helper, and inactive-child guards are consistent; the prior mapped-capacity overread does not explain these valid-map runs.

## 6. What the precision evidence can support

The production contract remains BFP4/LoFi projection weights, selected BF8 prefill inputs/CCL/KV, FP32 recurrence, BF16 norms/residuals, phase/batch-specific residual and collective strategy, and LoFi/FP16 decode SDPA. The wrapper retains it. B1 decode BF8 all-reduce and B8+ decode BF16 native RS are **already different accepted branches**; they must not be described as having identical effective precision merely because the policy name is shared.

`tests/run_multichip_decoder.py:93–110,334–345` constructs an optimized single-chip TTNN baseline and compares candidate outputs/states against it. `doc/optimized_multichip_decoder/README.md` explicitly limits its evidence to layer equivalence. A high PCC is useful for regressions but does not bound the final logit margin after64 layers or guarantee qualitative equivalence after a greedy branch.

In particular `doc/optimized_multichip_decoder/precision_attn_bf8_l3.json` changes the candidate attention weights to BF8 while retaining the BFP4 baseline. K/V PCC0.995854/0.994134 in that experiment cannot establish worse BF8 fidelity to HF. The precision sweep is not an HF end-to-end quality sweep. Conversely, the current full-model short AIME success is real evidence that broad logical structure works; it does not validate this different prompt, prefill branch, or length.

Plausible numerical sensitivity is therefore an **unproven mechanism**, not an acceptable final explanation or fallback. Missing same-history logits, margins, and layer-boundary comparisons prevent assigning it to quantization, CCL, GDN, SDPA, or the terminal head.

## 7. Candidate ledger after adversarial source checks

| Candidate | Direct contradiction / required condition | Match to observations | Smallest falsifier / intervention boundary | Status |
|---|---|---|---|---|
| Greedy tie semantics | Source lacks original-ID stable ordering; requires a tie at the relevant TT maximum | Can change279 versus1 on the shared prefix and launch a loop; batch logits may differ. Does not establish why such a tie occurs or later recovery. | Capture one full logit vector at y56; strict TT279>TT1 falsifies tie explanation there. If proved, repair greedy tie semantics only. | Concrete discrepancy, conditional cause |
| Accumulated numerical change | Requires changed logits/margins, not available in artifacts | Compatible with short-pass/long-fail and deterministic feedback; many changed batch paths can contribute | Same-history logits and earliest layer/op differences against HF and production-operand controls | Plausible, unlocalized |
| Exact4096 fused MM/RS defect | Reached only for B1 physical4096; no current geometry/CB contradiction found | Explains a relevant length/batch boundary, but does not alone explain why divergence is at56 | Same-input native versus equivalent separate projection, including precision-matched rounding controls | Diagnostic target, not proven bug |
| Long fused-GDN handoff defect | B1 uses long fused schedule; ledger currently consistent | Large batches use phased; short B1 has fewer handoffs/different producer count | Identical q/k/v/g/beta/state fused/phased and per-chunk comparisons | Diagnostic target, not proven bug |
| Cache/feedback/position fault | Would require a wrong token, state identity, page, or position on this actual run | Reduced probes pass; common first56 outputs and deterministic periodic tail do not independently rule out latent state error | Exact-history eager versus trace with readback positions/state checkpoints | No supporting contradiction found |
| Repaired rotary, EOS, external-map issue | Current evidence reaches repaired backend; EOS correctly enabled; benchmark uses owned valid map | Prior failures are different and do not explain current B1 persistence | Fresh revision-consistent confirmation if future evidence disagrees | Not a current diagnosis |

## 8. Discriminating controls, ordered by information gained

Run only under the owning pipeline's serialized hardware schedule after current work completes. Preserve original artifacts and record source/library hashes, exact IDs, actual selected programs, and explicit synchronization boundaries. Reduced controls are localization evidence; they cannot replace an all64 quality check.

1. **Shared-history y56 decision, unchanged production policy.** Reset, prefill the saved4096 HF IDs, then feed the shared y0..y55 IDs; read logits predicting y56 after consuming input position4151 (post-replay position4152). Capture the exact input token/position, all four vocabulary shards, full-vocabulary host argmax, maximum tie set, TT1/TT279 values/ranks/margins, local top-k IDs/order, and actual selected token. Compare with HF on that exact history. First determine whether selection changes for fixed TT logits, including a tied-maximum difference or a nonmaximum error; separately quantify TT/HF distribution differences, which may coexist. The existing `generate(...,host_sampling=True)` path now handles *both* initial and later tokens on host; use it as a diagnostic, not a performance result, and still compare the same saved logits before free-running histories separate.
2. **One-current-revision batch control.** Same checkpoint, generator, library, prompt IDs, output cap512, and EOS policy at B1/B8/B32; record raw output IDs for every relevant slot. Use a common valid context capacity (at least4608) to remove current262144/32768/33024 allocation differences. Preserve the selected batch-specific strategy. Teacher-force common histories for numerical comparison; do not compare logits after histories have diverged.
3. **Separate prefill state from decode behavior.** Capture last hidden/logits, per-layer convolution/recurrent state, and used K/V after4096 prefill. Compare the same slot across batches and HF; use real captured TT operands to test a suspect operator independently. Where layouts allow an exact supported state transfer, test B1 decode from the comparison prefill state and vice versa; treat state-transfer correctness as a prerequisite, not an assumption. First collect enough checkpoints to identify an early sharp discrepancy versus gradual drift.
4. **Reduced exact4096 probes.** Use real weights/activations, first layers0–3 together and the earliest discrepant layer(s), on the actual4096 prefix. Test native out/down on identical operands and fused versus phased GDN with identical C32/inverse/FP32 settings. Collect output and state, maximum/RMS errors, and selected-token effects; aggregate PCC alone misses directionally important error. A separately prepared alternate4096 chunk partition can diagnose the exact4096 branch, but changes GDN/SDPA scheduling too, so it is not a one-op proof.4095 rounds to4096;4064 or an explicit alternate chunk plan is needed to avoid that physical branch, and a changed prompt is an additional confounder.
5. **Localize decode differences only if needed.** Under the common prefix, inspect norm output, projections, convolution, GDN recurrence, Q/K before/after RoPE, used K/V, ungated/gated SDPA, out/down, residual, final norm and head. Compute host reference results from the exact captured TT operands to separate input drift from operator error. Single-axis alternate norm/CCL/SDPA geometry, higher precision, or disabled fusion is allowed only as a diagnostic; an improvement must be traced to the first differing operation and its numerical/ownership contract.
6. **Free-running closure on the intended policy.** After a minimal justified repair, reproduce exact all64 B1_4096/512 with normal EOS and coherent exit from reasoning; compare HF's continuation at equal budget. Recheck same-revision B8/B32, short AIME top-k and shared qualitative prompts, trace feedback/positions/page maps, and allocation/workspace contracts. Include one nonrepetitive long prompt and a nearby length as additional coverage, not substitutes for the failing fixture.

The final solution must retain the optimized multichip weight/activation/KV/CCL/fidelity policy, KV split, residual contract and strategy. Disabling optimized branches, adopting higher precision, forcing a larger batch, adding a repetition penalty, or extending the cap is not an accepted final fallback. No vLLM work is needed for this investigation.

## 9. Other Potential Issues and test gaps

- The fused MM/RS device operation's `compute_output_specs` derives RS specs from the input tensor rather than the MM result (`.../matmul_reduce_scatter_async_device_operation.cpp:63–72`). This looks inconsistent for non-square projections, but the executed `create_output_tensors` returns the supplied persistent output and the program factory consumes the actual MM tensor. Production provides correctly sized persistent outputs. No causal path from that spec discrepancy to this loop was established; it is not a headline cause.
- Current proofs of deterministic slots, exact trace feedback, and reduced sampling do not imply global first-index tie agreement or full-stack HF distribution equivalence.
- The apparent GDN fused-only reconfiguration change is in a nonselected Horner branch for Blackhole/C32. Its mere presence is not evidence against the active SFPU branch.
- Existing prepared-bucket4095/4096 equivalence is reduced-stack evidence, largely against the inherited TT path. It does not compare all64 exact4096 outputs with HF or certify long free-running quality.
- The generator/build provenance is unusually useful, but does not pin every shared sampler/lowered kernel source or the HF Python environment. Capture those identities for a final causal control; do not retroactively assume historical behavior from current analogous code.

## 10. Inspection and report review

Performed only cheap local/static work: source reads, git/source-hash checks, JSON token/hash/cycle comparisons, and host-only geometry/cost-model/count calculations. Three isolated xhigh investigators covered wrapper/sampling, linear attention, and full attention/precision; the primary investigation independently traced native MM/RS, minimal all-reduce, relevant sampler lowering, and the evidence.

Before finalizing, headline claims were tested against selected branches and passing contrasts. The earlier rotary issue, mapped-capacity overread, native-RS undersized block, fused-AG readiness mismatch, unsupported odd-ring sum, and nonselected GDN Horner reconfiguration were excluded as explanations for the present configuration. Missing logits were not treated as evidence for or against a numerical or tie hypothesis. The conclusion remains deliberately bounded: **a real long free-running failure, a precisely located first token divergence, a concrete conditional sampler discrepancy, and specific unfalsified math-path controls—not a proven root cause or a verified repair.**
