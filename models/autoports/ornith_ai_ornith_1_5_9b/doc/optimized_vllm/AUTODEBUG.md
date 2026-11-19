# AutoDebug: Ornith vLLM chunked-prefill shape amplification

## Leading finding

**A scheduled 512-token fragment of a 32768-token prompt executes 32768 physical rows in each linear-attention layer and 16384 physical queries in each full-attention layer.** The earliest model-owned decision is `tt/prefill_plan.py:138–141`: it derives the execution bucket from the original canonical chunk's length, rather than the fragment's consumed rows. `tt/prefill_runtime.py:286–296` materializes and executes that entire bucket. Masks enforce logical validity and mathematically neutral recurrence updates for finite dummy operands; they do not remove dummy computation or guarantee bit-identical state.

This is an intentional correctness/preparation design with a large serving cost, rather than evidence of a broken plugin scheduler or a defective accelerator kernel. Whole-request profiles previously repaired real numerical discrepancies. The optimization boundary is therefore **separating canonical numerical/boundary policy from bounded execution extent**, with validation of the resulting arithmetic. Changing only the planner's bucket is not an established safe fix.

There is also a smaller, concrete waste: intermediate fragments compute a terminal norm/head whose device result is immediately discarded. Skipping that unused terminal work is the narrowest candidate with a direct argument for preserving numerical outputs and state. It cannot remove the dominant per-layer amplification.

Confidence is high in the shape/work diagnosis. No compact implementation, speedup, final ITL result, or quality acceptance was established by this inspection. No devices were opened, no profiler was run, no processes were controlled, and no source was edited. Only this report was written.

## Scope and evidence identity

Inspection date: 2026-11-14 UTC. Checkout HEAD: `fffd4918cd09d2b58b0f1cbcdba4a8ddefc727ab`. Pinned plugin HEAD: `2a1504554aa7babb1d426dfbca8c7db8c0a795ef`, clean at inspection. The checkout has pre-existing untracked Stage 10 documentation, measurement directories and a deferred-read test; they were preserved. No applicable checkout `AGENTS.md` was found; the plugin's `AGENTS.md` was read.

**Concurrent-work note:** during the final review, edits made outside this investigation appeared in `tt/generator.py`, `tt/generator_vllm.py`, serving probes, and additional untracked artifacts. The later generator hash was `46fcc84eaca478c4ef5cff1c9e6bbbb29a7df829505c045132595d98af4a6561`; the diff defers read submission into `read_decode_output` and changes sampling-history reset ownership. The later wrapper hash was `162a226212012438daa87ffe4b6b36805770707704c9a20ebfcde419d0d19cc6`; its diff changes decode-event logging. All source line references and read-submission descriptions below refer to the initial hashes in the table. These later edits are not acceptance-reviewed here. The planner, runtime, adapters, decoder and prefill contract retained their recorded hashes, so the leading shape finding is unchanged. No concurrent edits were reverted or modified.

Unless prefixed otherwise, paths below are relative to [the model directory](models/autoports/ornith_ai_ornith_1_5_9b). `P/` means `/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin/src/vllm_tt_plugin/`. `C/` means checkout-root `ttnn/cpp/ttnn/operations/`.

The old report is preserved at [doc/optimized_vllm/prior_AUTODEBUG.md](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_vllm/prior_AUTODEBUG.md). It diagnoses an earlier trace-allocation failure at another revision; its proposals are not current implementation or numerical proof.

Source fingerprints at inspection:

| Model-relative file | SHA-256 |
|---|---|
| `tt/prefill_plan.py` | `2985bbfb67899b46815423a7f82dc867ba47e8c501f8d54cd340932a52ce1426` |
| `tt/prefill_runtime.py` | `14ac2ede2e63c346deb7ad264557e8da4f22a88f3cc547c75e56d87b7110c4dd` |
| `tt/prefill_ops.py` | `462b6f8a1a3ae2975f20048a1729f140e9b55cd9ab4b3380570933db1dba012f` |
| `tt/multichip_decoder.py` | `1a23ecb9b6c8b11919ab8b8a2f1e3d4e01e463f0f8fa55c5cac1e0da8a661140` |
| `tt/generator_vllm.py` | `438e7e659ae552efda27493c82f10f337e24740dc043df74024d3187afeb89cb` |
| `tt/generator.py` | `0bc4ee8bff8ebebcf3c5d4a1168f8acda7108692f49ade257cfe407e6342def5` |
| `tt/prefill_contract.py` | `ad8670b0361fae87605865793e411ab6844ca71ec11f8449ee2e4daa8f2b1587` |
| `doc/datatype_sweep/selected_precision_config.json` | `598eca494c69c462b1581b8968cc1d106bd2b0b311a1a0461a7dd32fc264124e` |

## Direct observations and their limits

The reviewed log is `readiness_vllm/stage10_chunk512_b32_untracked/server.log`, SHA-256 `5aa0eab9cf7c03eca3c3b8ba6c6a88ecf1801e69e845130c524137e3a750bd1d`, 1,223,507 bytes in the inspected snapshot.

| Observation | What it establishes |
|---|---|
| Log lines 28 and 42 announce at most two prefills before one decode-only step. | The configured policy is enabled. |
| Log lines 12819–12829 show successive 512-token fragments, original/request length 32768, slot 31; replay counters stay at 511 for a pair, then 512, then 513. | Original horizon reaches the model and decode interleaving occurs. |
| There are 64 horizon-32768/chunk-512 prefill-entry events. Among contiguous same-slot event pairs, 32 have unchanged model replay counters; their entry gaps are min/median/max 1.069/1.279/1.485 seconds. The other 31 have a replay-counter increment and gaps 2.682/3.249/3.852 seconds. | The reported long pauses are visible; these are host event gaps, not isolated device timings. |
| The eight horizon-4096/chunk-512 entries have four corresponding unchanged-counter gaps of 0.0921, 0.0918, 0.0941 and 0.0951 seconds. | Same scheduled size can have very different observed cost at different horizons. Position/key-history differences prevent treating this as an isolated padding benchmark. |
| Six captures and zero releases remain in the cited fragment sequence. | No recapture/trace retirement is visible there. This does not prove absence of new compilation or safe allocations. |
| `vllm_run_summary.json` records budget 512, context 262144, B32, P300x2, decode-only trace mode and async scheduling; `outputs` is empty. | Launch settings are available. This summary does not establish a passed ITL percentile gate. |

The production metadata path is intact: `P/model_runner.py:2793` reads original `num_prompt_tokens`; `:4118–4123` passes it through the opted-in capability. Ornith declares that capability (`tt/generator_vllm.py:28–32`), requires authoritative original lengths (`:183–191`), and forwards them as `request_lens` (`:204–235`). There is no evidence that the plugin replaced the original horizon with 512.

The scheduler bounds **consecutive step count**, not elapsed time: defaults are 2/1 in `P/config.py:639–648`; the policy consumes those counts in `P/scheduler.py:189–235,894–931`. Reducing scheduled tokens therefore cannot guarantee a subsecond pause while each step still executes huge physical buckets. A simple time budget would require roughly `2 * completed_prefill_step + completed_decode_and_other_work < 1000 ms`; measured completion/streaming ITL is still required.

### Completion boundaries

`ORNITH_SERVING prefill` is printed before calling the generator (`tt/generator_vllm.py:209–224`). Intermediate fragments run device work but return CPU zeros without a device-logit read (`tt/prefill_contract.py:423–439`). Replay counters advance immediately after nonblocking trace enqueue on CQ0 (`tt/generator.py:1095–1104`). They count submissions, not completions.

In the initial generator snapshot, decode output submission queues owning asynchronous host copies and a CQ0 event (`tt/generator.py:1156–1178`). Host processing synchronizes that event (`:1181–1204`). The plugin drains pending async output before a transition to prefill (`P/model_runner.py:3224–3234`; `P/async_decode.py:947–966`), and stamps completion after finalization (`P/async_decode.py:1038–1068`). Consequently some queued prefill cost can become visible in a later decode drain. Neither the 1.279-second entry-gap median nor the larger gaps can be assigned wholly to GDN, SDPA, host overhead, or decode kernels from this log. The concurrent deferred-read edit changes where the transfer is enqueued; it does not turn these prefill-entry timestamps into device-completion measurements.

## Earliest calculation and complete work chain

1. `plan_prefill(total_length=32768, start=0, end=512, kind=...)` uses original chunks anchored at zero. At `tt/prefill_plan.py:138–141`, linear chunk `[0,32768)` selects bucket 32768; full-attention chunk `[0,16384)` selects 16384. `valid_count` remains 512 (`:103–106`).
2. Runtime passes that bucket to `_execute` (`tt/prefill_runtime.py:421–423,474–482`). `_read` uploads fixed-shape row indices and gathers **height** residual rows (`:235–240`), including dummy/stale staging rows. It does not gather only valid rows.
3. Linear `refresh_profile` masks g/beta after the real end and selects real-end convolution-history indices (`tt/prefill_ops.py:172–183`). Forward still runs all projections, convolution, recurrence, output reduction and MLP at the residual height (`:81–166`).
4. Full attention likewise receives physical residual height, derives all projections/Q/K/V at that height, suppresses wholly dummy cache-page fills with `-1`, then runs SDPA and the rest of the layer over the entire height (`tt/prefill_ops.py:325–374`). `valid_count` is not passed as a query-work bound to SDPA.
5. Runtime converts/writes the entire physical result to staging (`tt/prefill_runtime.py:243–248,296`). It selects requested terminal rows only afterward (`:306–318,498`). There is no early valid-row crop that removes the expensive layer work.

Concrete host-only planner results:

| Horizon / interval | Linear consumed → physical rows | Full consumed → physical rows |
|---|---:|---:|
| 512 / `[0,512)` | 512 → 512 | 512 → 512 |
| 32768 / `[0,512)` | 512 → 32768 | 512 → 16384 |
| 32768 / `[512,1024)` | 512 → 32768 | 512 → 16384 |
| 32768 / `[16384,16896)` | 512 → 32768 | 512 → 16384, next original full chunk |
| 262144 / `[262112,262144)` | 32 → 32768 | 32 → 16384 |

For one aligned fragment this is 64 times the necessary row extent in linear layers and 32 times in full layers. Across 24 linear and eight full layers, the sum of declared layer-row extents is 917,504 versus 16,384 for 512 rows in all layers: a **56-fold row-count ratio, not a predicted latency speedup**. A BF16 width-4096 residual is 256 MiB at height32768, 128 MiB at16384, and 4 MiB at512, per chip per tensor. These dimensions affect staging traffic, dense operations and collectives as well as attention.

Nonaligned fragments can amplify this again. At horizon32768, `[68,580)` plans `[68,576)` with aligned start64/consumed512, then `[576,580)` with consumed4 and a new boundary snapshot. **Both segments inherit the same large bucket in every layer.** This is a valid continuation path and must stay valid; it is not the path needed to explain the aligned sweep.

### Lowered operations confirm the dummy work is real

| Operation | Caller and lowered contract | Instantiated work |
|---|---|---|
| QKV causal convolution | `tt/prefill_ops.py:110–120`; `C/experimental/kda/qkv_causal_conv1d_silu/qkv_causal_conv1d_silu.cpp:30` sets sequence from input logical shape. Factory `device/qkv_causal_conv1d_silu_program_factory.cpp:295–308` distributes `sequence/32 * channel_blocks`. | Q/K/V widths512/512/1024 and channel chunk128 imply16 channel blocks. Height32768 has1024 time tiles /16384 work items; height512 has16 /256. g/beta masks do not reach this scheduler. |
| Gated delta rule | `tt/prefill_ops.py:134–146` passes chunk_size32 but no valid count. `C/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp:145–148,197–200` computes `T` and `NC=ceil(T/32)` from physical input. Its automatic fused/phased choice also depends on NC (`:304–327`). Scan kernel `device/kernels/compute/chunk_gdn_scan.cpp:62–68` iterates `c < NC`. | 1024 recurrence chunks instead of16. Masked chunks still run; masks are arithmetic inputs rather than work bounds. |
| Chunked SDPA | `tt/prefill_ops.py:331–362` supplies per-chip Q `[1,4,16384,256]`, q_chunk32/k_chunk128, start tensor and page map. `C/transformer/sdpa/device/sdpa_program_factory.cpp:1187–1269,1365–1369` derives query work from Q shape. Reader `device/kernels/dataflow/reader_interleaved.cpp:432–434,563–583` traverses that query work with causal key bounds. | 512 query chunks/head, 2048 across four heads, versus16/head and64 total at512 rows. Dummy fill sentinels suppress writes, not dummy queries. |
| Dense projections / MLP | `tt/multichip_decoder.py:630–701,785–806`; `C/experimental/minimal_matmul/device/minimal_matmul_program_descriptor.cpp:321–333,373–414` derives M from physical volume and partitions it. | Full physical M is computed. A mask in the recurrent branch does not shrink dense/MLP/CCL work. |

This is not a claim that SDPA attends all262144 keys for every live query. Logical context/cache capacity, padded virtual page-table capacity and actual causal key bounds are separate. Dummy queries nevertheless have their own causal work and safe page aliases (`tt/prefill_ops.py:221–230`).

## Why whole-request profiles were required

The old trace issue and the resume-numerics issue were separate requirements solved together:

- Exact request shapes/offsets introduced new persistent program-cache allocations after decode capture. The prior report's same33 versus34 contrast and the current pre-capture catalogue explain why bounded, prepared execution signatures are needed. Current preparation is implemented; the old missing-preparation diagnosis must not be copied into the current headline.
- Shape selected numerical policy. Norm activation/CCL payload switches at512, projections switch family at512–1024, and MLP switches at128 (`tt/multichip_decoder.py:480–495,654–701,739–806`). Canonical whole-request selection stopped each scheduler fragment from independently choosing those policies.
- Nonaligned starts changed the phase of the 32-token recurrence grouping. The current solution restores saved **layer inputs**, conv history and FP32 recurrence at the preceding aligned boundary, recomputes up to31 overlap rows and suppresses their logical emission (`tt/prefill_plan.py:10–14,134–149`; `tt/prefill_runtime.py:250–284,467–485`). Whole-request shapes alone do not solve that phase problem.

Historical length513 evidence in `doc/full_model/` distinguishes these causes:

| Evidence | Start | Minimum logit PCC | Top1 agreement | Result |
|---|---:|---:|---:|---|
| Original fragmented path, summarized in `work_log.md:72–74` | 32 /68 | 0.8744 /0.7978 | — | Fail |
| Canonical profiles only, `full_resume_profiles.json:60–72` | 68 | 0.7978458573 | 0.9612402916 | Fail |
| Profiles plus aligned overlap, `full_resume_overlap.json:60–69` | 68 | 0.9999667689 | 1.0 | Pass |
| Profiles plus aligned overlap, same artifact `:29–38` | 32 | 0.9999667689 | 1.0 | Pass |

The identical-input GDN control matched outputs and final state exactly at aligned splits32/512. Split68 changed output/state, with max differences0.0061884/0.0843143 (`doc/full_model/resume_policy_probe.json:480–578`). These are historical controls, not a new hardware result or proof about every32K execution.

The old report's one-token decode prologue proposal is superseded by current aligned-boundary replay. It should not replace that mechanism. Rejected BF16 and fixed32 experiments also do not justify changing the selected policy (`doc/full_model/work_log.md:95–120`).

### Exact equality and quality acceptance are different contracts

**A claim of exact preservation requires bit equality of the relevant outputs AND state.** PCC near1 or unchanged argmax does not establish it. But the existing normalized/full-model numerical gates do not require universal bit equality against the old path:

- Linear normalized probes require PCC≥0.995 (`tests/full_model_normalized_linear.py:256–259`). Saved real outputs, conv and next decode are bit-equal in the cited boundary cases, while FP32 recurrent state is not: valid 65→128 max delta 0.0009746552; 129→511 delta 0.0015764236; 513→640 delta 0.0019459724 (`doc/full_model/normalized_linear_boundaries.json:3–168,423–503`). The neutral-padding adapter is therefore not universally state-bit-exact; these experiments do not isolate dummy recurrence as the cause.
- Accepted full resume uses mean PCC≥0.99, minimum≥0.98 and top1≥0.98 (`doc/full_model/full_resume_overlap.json:18–22`). Its observed top1=1 is not an assertion of tensor equality.
- Mixed versus solo execution under the **same** schedule has exact logits/state requirements (`tests/full_model_serving_prefill_probe.py:119–127,165–174`). Untouched pages and inactive peers have separate exact-preservation requirements. These must remain exact even for a candidate that differs numerically from the old large-profile implementation.

Do not reuse the early normalized smokes as selected-policy proof. Both standalone probes construct the decoder without `precision_policy` (`tests/full_model_normalized_linear.py:199`; `tests/full_model_normalized_attention.py:304`), selecting historical attention HiFi2/MLP LoFi defaults. Current `bfp8_all_lofi` makes all five projection groups, including the head, BFP8/LoFi, with no exceptions. Norm/GDN compute remains HiFi4, convolution retains its fixed config, terminal norm is HiFi2 and decode SDPA HiFi2 (`tt/multichip_decoder.py:137–184,228–242`; selected JSON). “All LoFi” must not be applied to every primitive.

## Minimal interventions and safety conditions

### A. Skip a provably unused intermediate terminal head

Propagate the existing authoritative `discard_logits` intent to the runtime so that, after advancing all layers and request history, it can omit `_head` and return the existing fixed host-zero placeholder. Today `_head` is unconditional (`tt/prefill_runtime.py:498`), then its device result is deleted (`tt/prefill_contract.py:423–439`). `model.logits` has no recurrent/cache/sampler mutation or CCL call (`tt/model.py:218–235`).

Keep terminal warmup in `_warm`; final fragments, sampled fragments, host diagnostics and all-logit calls still need it. Keep history bookkeeping and sampler-history behavior unchanged. This removes one32-row terminal read/norm/vocabulary projection per discarded row. Exact remaining outputs/state and unchanged sampling counters can be checked directly. Timing benefit is unmeasured and is unlikely to solve the padding problem alone.

### B. Compact execution with explicit canonical numerical policy

The smallest useful experiment is internal to the model: retain `total_length`, original caps/chunk starts, profile/history ownership and page mappings, but represent **canonical numerical profile** and **execution rows** separately. Start with512 aligned consumed rows at horizon32768. Use already prepared bounded physical heights; avoid a new plugin capability or option.

A narrow first implementation can compact only large-profile segments with consumed count at most 512 to a **fixed execution height of 512, including short tails**. Leave other segments on the canonical path while testing compatibility across transitions. This bounds the experiment and avoids accidentally selecting the below-512 BF16/short-MLP regimes for a long request's tiny suffix. The existing 512 mask/history/map plumbing already represents those valid counts; its numerical behavior under the long-request comparison still needs validation. Supporting a smaller execution-height catalogue later is a separate expansion.

For an exactness-oriented first control, retain canonical BFP8 norm/CCL casts, HiFi4 norm/GDN, selected LoFi projections, fused packed MLP and the canonical projection family while compacting to 512 rows. Keeping the projection family is a conservative isolation control, not a necessary condition for bit equality. Simply using the existing 512 adapter selects standard 2D projection kernels where canonical 32768/16384 uses `minimal_matmul`; it also changes the large-norm branch and residency. For small tails, default compact shapes additionally select BF16 norm/CCL and short/decode MLP paths. Those are real execution-policy changes, even if the JSON policy is unchanged.

Preserving explicit branches helps isolate shape effects but **does not prove exactness**. Minimal matmul changes grid transposition at M>N and per-core blocks (`C/experimental/minimal_matmul/device/minimal_matmul_program_descriptor.cpp:373–414`); GDN replans with NC. Removing masked GDN tail chunks also removes real FP32 operations: `C/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_math.hpp:1143–1169` computes and packs a matrix state update rather than copying state. Finite `g=0,beta=0` establishes a mathematical identity, not a bitwise one.

A default compact512 path is a **quality candidate**, not a proven exact replacement. It may be acceptable if full-model accuracy, resumed-prefill, scheduling invariants and prompt quality remain within their existing gates. Neither the old failed mixed-shape paths nor the old passing large-profile paths decide that question. Do not silently relabel approximate agreement as exact equivalence or relax exact batch/peer invariants.

For either candidate:

1. Preserve original prompt IDs, absolute positions and horizon. Never substitute fragment end512 for request length32768. That changes compatibility and can trigger prefix rebuilding (`tt/prefill_contract.py:342–383`). Keep context262144, actual KV allocation, supported lengths and both original layer chunk caps. No truncation or rejection of valid nonaligned lengths is a repair.
2. Keep g/beta masks and convolution history tied to **consumed real rows including overlap**, not execution height (`tt/prefill_ops.py:102–129,172–183`). Zero embedding alone is insufficient: gates/decay and old convolution history can still change state. Vary finite dummy values in testing; masking is not a sanitizer for NaNs/Infs.
3. Preserve aligned snapshot/input-tail replay. With capacity512 and start383, overlap31 leaves room for at most481 new rows in that invocation. Split remaining work at an aligned boundary; do not drop tokens or overflow512. Preserve same-tile continuation68→70, partial nonfinal tail snapshots and original cap crossings. A temporary canonical execution path for other valid shapes is a coverage strategy, not a license to reject them; test transitions and profile compatibility.
4. Maintain actual per-slot/pool ownership and page offsets. `fill_pages=-1` suppresses wholly dummy pages; a partially valid final page is written and later rewritten from the saved aligned layer input. Keep causal key history and safe padded aliases, and clamp only dummy RoPE lookup indices (`tt/prefill_ops.py:195–230`). Compact Q must not shorten the causal key prefix or replace valid positions.
5. Give differing numerical execution policies honest compatibility identity. Current `ProfileTag` contains version/kind/chunk start/bucket only. If a compact prefix can differ, existing large-profile tags alone cannot certify reuse; revise internal policy tags/version and boundary records consistently, or establish exact compatibility first. Rebuild incompatible known prefixes through the existing path. Do not advertise serving prefix caching; recurrent-state snapshots remain necessary.
6. Construct/warm every reachable policy/height/binding/helper combination **before** the first capture, using the real serving cache. Existing default preparation covers current heights, including512, but a new forced-policy512 variant can create new program signatures. `PrefillRuntime.prepare/_require_no_traces/register_cache` (`tt/prefill_runtime.py:40–84,149–199`) and `PrefillContract.prepare` (`tt/prefill_contract.py:48–74`) already own this lifecycle. Do not allocate new persistent helpers at admission or exempt program-cache allocations from tracking.
7. Retain decode/sampling trace IDs and backing identities through requests. Preparation must restore state; `_warm` synchronizes before readiness (`tt/prefill_runtime.py:320–375`), and capture follows preparation (`tt/generator_vllm.py:136–162`; `tt/generator.py:706–823`). Preserve the reviewed norm-tensor release before MLP (`tt/prefill_ops.py:163–166,371–374`), output-handle lifetimes and CQ0 ordering. Initial-snapshot teardown waits for pending device-transfer events before releasing traces (`tt/generator.py:1365–1379`); that wait need not complete host materialization.

## Discriminating reduced-layer checks

These are proposed follow-ups, not executed device experiments. Use layers 0 and 3 with real weights, explicit **selected** precision policy, TP4 and the actual served cache/page geometry. Preserve horizon 32768 for the central tests. Larger boundary/max-context cases must retain their appropriate original horizons unchanged between variants. Record effective compute objects/dtypes, not just configuration names. No profiler is needed.

### First isolate shape from fragmentation

Clone identical nonzero conv/recurrent/KV state and identical live layer inputs. Compare current32768/16384 versus compact512 on the **same ≤512 real rows**, before introducing a different request schedule. Use finite different dummy rows. Include valid1,2,31,33,68,129,383,481,511,512 and a subsequent decode. Then compare full versus fragmented execution over513/1025 rows and around16384/32768 boundaries.

| Control | Change | What it can falsify |
|---|---|---|
| A | Current canonical bucket baseline | Repeatability/fixture validity. |
| B | Default compact512 only | Whether existing small profiles suffice for quality; not an exactness claim. |
| C | Compact512 plus canonical norm/casts, minimal projection family, fused minimal MLP and CCL policy | Whether explicit shape-policy changes explain B's first discrepancy. |
| D | Only convolution extent changes | Whether causal convolution/history or its geometry introduces the first difference. |
| E | Only GDN extent changes, first baseline factory/geometry then automatic planning | Whether dummy state updates or GDN replanning matter independently of projection inputs. |
| F | Only full-attention query extent changes, same live Q/K/V and key history | Whether SDPA query scheduling changes valid results independently of projections. |

Compare the earliest intermediate that differs: input norm, packed projection, Q/K/V/g/beta, conv history, GDN output and FP32 final state, output projection/reduction, postnorm/MLP, full-layer live residual, valid K/V, terminal logits and next decode. For an exact claim record bit equality on every chip; also record max error/PCC and top1/top5 for the quality assessment. Passing only last-token argmax can hide persistent-state damage.

### Then exercise valid continuation and ownership

- Compare both implementations' own whole-request and fragmented runs at starts1,2,31,33,68,383,511,513; same-tile68→70; `[68,580)`; boundaries16384/32768; final262144 tail. Include zero and nonzero history, shuffled physical page maps/pool offsets, slot remap/reuse, inactive peers and mixed admission. Cover both kinds' different canonical tails, including linear511/full512.
- Save/restore the preceding aligned **layer input** and linear boundary state. Require no-history/wrong-boundary controls to fail. Check consumed/logical counts separately so overlap is neither omitted nor double-applied.
- Preserve exact mixed-versus-solo behavior for the same execution policy and exact untouched peer/prefix pages outside the documented overlap and partial-page write set. Compare candidate versus canonical with the explicitly declared exact or numerical contract; do not conflate those checks.
- Interleave changing prefill offsets/counts with live traced decode and all supported sampler graphs. With `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, require stable warmed program counts, no forbidden allocations, stable trace IDs and no request-time releases. Six captures alone is insufficient. Validate two outstanding async outputs and sampler RNG/history continuity.

### Promotion gates

Reduced-layer agreement is a discriminator, not the final quality gate. Run the existing full-stack resumed-prefill checks and both-reference prefill/traced-decode accuracy under `bfp8_all_lofi`. The recorded datatype acceptance is top1≥90%, top5≥98%, top100=100%; P1 ΔNLL≤0.004 and KL×1000≤5 (`doc/datatype_sweep/README.md:5–9`). Selected natural prefill is90.2344%, so there is little spare margin. Review generated text using original raw/chat prompt formats; preserve the original prompt and sampling semantics. Long-context capacity passing does not itself establish long-context quality.

Finally measure real TTFT and background-decoder ITL during 4096/32768 arrivals under identical settings. Diagnostic completion timing must explicitly wait for the queued fragment without being mixed into the uninstrumented serving comparison. Test scheduler budgets 512/1024/2048 as separate experiments: this inspected log establishes 512 behavior, not outcomes for the other sweep points. The observed >1-second same-counter entry gaps provide no evidence that changing interleave alone meets subsecond ITL.

## Competing explanations and other potential issues

| Candidate | Direct evidence / match | Missing proof / smallest intervention | Assessment |
|---|---|---|---|
| Canonical profile also dictates full execution extent | Planner/runtime contradiction with intended small scheduled work; lowered loops consume the extent; horizon dependence matches observations. | Quantitative runtime attribution and compact numerical acceptance need hardware. Separate numerical profile from execution rows. | Leading performance cause. |
| Plugin dropped horizon or failed to interleave | Source and log show original32768 and 2/1 replay-counter pattern. | Would require contradictory routed metadata/step sequence. | Refuted for cited sequence. |
| Discarded terminal work | Unconditional runtime head then immediate delete/zero replacement. | Measure savings; skip only explicit intermediate-discard heads. | Concrete secondary waste. |
| Host scheduling/logging/refresh, async drains | Present in source; entry gaps include these effects. | Completion-aligned timing needed. Preserve async ownership/order. | Possible contributor; no isolated attribution. |
| Max-context cache/page-table size itself causes all work | Context determines capacity, but planner uses original chunk count and SDPA uses Q/causal bounds. | Must distinguish physical queries from metadata/key capacity. | Incorrect as the earliest explanation. |
| GDN/SDPA/CCL low-level defect | No concrete kernel-contract violation found on this path. | Would need a state/output failure beyond intentional oversized work. | Unproven; not needed for current symptom. |

Two secondary risks should not be promoted into the headline: dummy recurrence neutrality assumes finite operands, and new policy variants can violate retained-trace preparation even when capture counts stay constant. No supplied observation proves either caused this run's latency or corruption.

## Inspection validation and report review

Executed only cheap host/static checks: parsed the relevant Python sources with `ast`, imported the pure host planner without torch/TTNN/vLLM, instantiated the concrete intervals above, and ran the existing host planner suite through `unittest` with `-I -S`: **10 tests passed in0.119 seconds**. These tests check coverage, profiles, alignment and capacity; they do not validate device numerical equivalence or performance.

The first draft was revisited against call sites, lowered work bounds and saved numerical artifacts, including three xhigh subagent reviews. Corrections included the active ordinary-matmul descriptor path, revision-specific read timing, and the scope of exact state comparisons. Claims were constrained as follows: log gaps are entry gaps, not kernel runtimes; step counters are not completion; neutral padded recurrence is not a proof of exact state; historical normalized smokes are not all-LoFi acceptance; exact mixed/peer invariants are distinct from tolerant canonical comparisons; and the old missing-preparation/prologue proposals are historical. Unproven low-level defects and host-cost attribution remain secondary. The inspected source establishes the work amplification and its ownership; hardware checks remain necessary to select and accept its replacement.
