# AutoDebug: reduced B32 greedy repeat mismatch

## Findings

**The first observed divergent tensor boundary is prefill logits, and the sampled tokens remain maximum-valued. A sampler-only nondeterminism diagnosis is not supported.** The parent's completed `repeat_greedy_probe.json` reproduces the slot-30 sequence change while recording different prefill logits. Twenty sampler-only replays on one frozen real-logit tensor return the same complete token vector. The source also implements a deterministic, unspecified tie order for the selected Blackhole top-k path.

**The common sampler has a confirmed lowest-ID tie-policy limitation, but its involvement in this particular token switch is not yet proved.** It retains only 32 candidates per vocabulary shard before choosing the lowest-ID retained maximum. A shard containing more than 32 equal maxima can lose its lowest-ID maximum. The probe records 89–96 maxima across the whole vocabulary, not counts per shard or the complete maximum sets. It therefore does not establish that token 179 remained maximal when 514 won, or that candidate truncation excluded it.

The supported causal chain is:

1. Repeated reset/prefill executions produce different sampler-input logits, including different cardinalities of slot 30's maximum set.
2. The BF16 terminal softcap produces many exact maxima at 30.0. The sampler selects a maximum from each changing input, using its retained-candidate tie policy.
3. The prefill token changes from 179 to 514. That token becomes the next decoder input, after which the observed continuations differ: `[179,177,177,177]` versus `[514,179,179,179]`.

The source-level cause of the changing prefill logits remains unresolved. Calling it benign numerical noise, cache corruption, or a CCL race would exceed the evidence. A tie-policy repair alone would not establish logit reproducibility.

## Scope and provenance

- Investigation performed on 2026-10-31, live worktree HEAD `bdb236e49bdd887c9719e7a13b35af8bdc659964`. Relevant autoport files are uncommitted; the original failure log does not embed an immutable source manifest.
- Read `bringup/run_contract.json`, `bringup/agentic_evals/inputs.json`, and the AutoDebug skill. This was the already-isolated investigator; no launcher was invoked.
- Source inspection and cheap standard-library parsing/calculation only. No model, device command, reset, or hardware reproduction was run by this investigation. Runtime results below are explicitly the parent's existing artifacts.
- No removed Gemma implementations, Git history, external implementation sources, credentials, or API/enterprise login substitutes were accessed. The pinned HF `config.json` was read from the local snapshot for revision `842da3794eaa0b77d5f08bae87a17459d91ff475`.
- This report uses the problem's specific destination. The existing `doc/full_model/AUTODEBUG.md`, concerning late allocation, is preserved.

Inspected source fingerprints (SHA-256):

| File | Hash |
|---|---|
| `tt/generator.py` | `35060a8ee1cfbde9ecf0e3078532b232e17ec5b98580d54e2d590f04dbb8d10a` |
| `tt/model.py` | `cab3a766f45ea2303b21a9ca3dfbeaf126b825592340aac9493c7e46a268a79d` |
| `models/common/sampling/tt_sampling.py` | `fa48c2d253b20a8b14d2be1d8597d30395a03070e33522f16550e95d36fd374a` |

Paths abbreviated `tt/` or `tests/` below are relative to `models/autoports/google_gemma_4_31b_it/`; other source paths are repository-relative.

## Direct observations, separate from interpretation

`generator_state_final_b32.log` fails at `tests/check_generator_state.py:53`. Only slot 30 differs between its two reported output lists. The first-token difference is already present before the three decode steps. The initial run passes the earlier feedback and position assertions at lines 34–50; those assertions concern token replication/feedback and position advancement, not logit equality.

The parent subsequently produced `tests/probe_repeat_greedy.py` and `doc/full_model/repeat_greedy_probe.{json,log}`. The wrapper at probe lines 23–27 calls the original prefill, then copies `gen.logits` to host **before returning to generation and before decode**. The sampled token and saved logits refer to that same prefill. The sampler does not overwrite `self.logits`; its adjusted candidate tensor is separate.

| Probe repeat | Slot 30 outputs | Logit elements different from previous repeat, over all 32 rows | Slot 30 max | Count equal to max | Selected value |
|---|---|---:|---:|---:|---:|
| 0 | 179,177,177,177 | — | 30.0 | 90 | 30.0 |
| 1 | 514,179,179,179 | 1,850,994 | 30.0 | 89 | 30.0 |
| 2 | 514,179,179,179 | 1,799,437 | 30.0 | 94 | 30.0 |
| 3 | 514,179,179,179 | 1,782,757 | 30.0 | 93 | 30.0 |
| 4 | 179,177,177,177 | 1,830,921 | 30.0 | 96 | 30.0 |

The differing-element counts cover the complete `[32,262144]` snapshot; they are **not slot-30 counts** and do not measure difference magnitude. The probe saves summaries, not the raw logits. The eight returned `torch.topk` IDs are an unordered subset of a much larger tie: omission of 179 from that list does not prove it fell below 30.

The frozen-input control at probe lines 40–58 performs a fresh prefill, preserves its device logits, then executes only the existing sample trace 20 times. All 20 complete vectors are identical, with slot 30 selecting 514. The assertion verifies maximum-value selection for all 31 active rows. It does not record or assert that fixed-input logits remain equal after each replay, though source inspection shows the sampler's ordinary path operates on separate candidate tensors.

The prior passing `generator_state_b32.json` has exactly the same outputs as probe repeat 0. Probe repeats 1–3 differ from that artifact only at slot 30; repeat 4 differs only at slot 21. This is compatible with an intermittent boundary sensitivity already present in the earlier passing test. It does not establish that either recent wrapper change introduced the problem. The strengthened seeded and long-tail checks are after the failing assertion and are not evidence from this failed run.

## Concrete execution and state ledger

### Request, cache, and first-token path

`tests/check_generator_state.py:18–26` constructs a `(1,4)` Blackhole mesh, layers `(0,5)`, configured B32, context 4096, with prompt lengths `31 + 17*slot` for slots 0–30 and an inactive slot 31. This is a 32-row sampler, not a B31 dispatch. Prefill executes active requests sequentially with logical batch one (`tt/generator.py:170–188`).

For slot 30:

| Quantity | Derived value and source |
|---|---|
| Logical prompt length | 541 (`31+17*30`) |
| Physical prefill length | 544, rounded to 32 (`generator.py:178–181`) |
| Chunk offset | 0; no second 1024-token chunk |
| Last valid hidden row | 540; selected tile starts at 512, embedding index 28 (`prefill.py:70–76,94–99`) |
| Full-attention pages reserved | 20: `ceil((541+4)/128)*4` (`generator.py:232`) |
| Full-attention physical ownership | Pages 320–339; padded fill uses the first 17, pages 320–336 |
| Rounded attention read extent | 640 tokens / 20 pages; reservation covers the 128-token chunk rounding |
| Sliding ownership | 64 private pages per slot, cyclic capacity 2048; no wrap at this length |

The two complete generation calls both execute `set_sampling -> reset -> reserve -> prefill` (`generator.py:405–413`). Reset enqueues multiplication of every K/V cache by zero and refreshes positions/seeds (`:429–434`). The second reservation can skip copying unchanged tables because the revision is the same tuple of counts (`:247–248,273–285`); `prefill.stage` still updates every used prepared fill/attention table, offset, token/position input, and last-row selection (`prefill.py:45–76`).

The old inactive-slot allocation of four pages versus the new zero pages changes total reserved full pages from 344 to 340. The inactive slot is **last**, so no active physical offset changes. Active slot 30 retains pages 320–339 in either calculation. The physical cache allocation itself is unchanged; this is a reservation/map change. Inactive prefill exits before staging or reading its page map (`generator.py:170–176`). Before decode it receives position -1 (`:418`). Thus there is no demonstrated page-alias mechanism for the initial divergent prefill token.

Each active prefill produces one selected terminal row. All 31 selected rows plus the inactive zero row are concatenated into `[1,1,32,65536]` per chip and copied to persistent logits (`generator.py:187–200`). Sampling is enqueued on CQ0, and `_read_tokens` performs device-to-host readback (`:152–154,200–202`). Decode traces are enqueued later on the same CQ0 (`:312–319`), using sampled token storage as their input (`:95–104`).

The original test adds synchronous token/position reads around first-generation decode only (`check_generator_state.py:31–40`). The parent's probe reproduces the output change without that asymmetrical `_replay` wrapper. The first divergence also occurs before those wrapper reads. The failure is not merely an assertion reaching the end of an unfinished asynchronous dispatch: concrete prefill tokens and then a host logit snapshot have been read. Nevertheless, chip-zero token readback alone is not a blanket all-mesh completion proof.

### Terminal logits and sampler lowering

`tt/model.py:122–138` applies final RMSNorm, the tied vocabulary head (BFP8 weights, BF16 output, HiFi2 with FP32 destination accumulation), then `cap*tanh(logits/cap)` with cap 30 from the pinned config. Each scalar multiply and the tanh return BF16 tensors here. Large positive logits can therefore collapse to exact 30-valued ties; the probe confirms that this happens.

The tanh call does not inherit the matmul's FP32 accumulator configuration: `ttnn/.../eltwise/unary/unary.cpp:33–60` derives the unary accumulator mode from its BF16 input/output, and `unary.hpp:293–298` defaults `approx=false`. The Blackhole non-FP32 tanh branch uses its polynomial and explicit nearest BF16 rounding (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_tanh.h:184–207`), with clamping in the polynomial helper. This explains a source of exact ties, not run-to-run variation by itself. No tanh state/race violation was established.

The wrapper passes no force-argmax model configuration (`generator.py:63–72`), so `TTSampling` disables that shortcut (`models/common/sampling/tt_sampling.py:351–365`). Its ordinary path is:

1. BF16 tiled, interleaved DRAM logits, local shape `[1,1,32,65536]`.
2. Local top-k with physical `k=32`, `stable=False` (`tt_sampling.py:1267–1279`).
3. Gather 32 values and indices from each of four chips, producing width 128; convert/add offsets to global integer IDs (`:1282–1362`). This uses ordinary `ttnn.all_gather` when the supplied `TT_CCL` has no `line_all_gather` (`:797–821`).
4. Select the lowest global index among retained maxima, boost only that maximum for semantic `k=1`, then call `ttnn.sampling` (`:918–1008,1372–1393`).

At width 65536 the canonical C++ router selects Blackhole `topk_large_indices`, not the ordinary multicore bitonic factory (`ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp:282–373`). Its documented tie order is deterministic but unspecified (`:296–299`). The internal LLK target snaps to 512; 65536/512 gives 128 chunks, selecting the Classic body (`operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp:35–38,116–128`). The body sorts and merges chunks in a fixed loop (`device/kernels/compute.cpp:120–141`). Token IDs 179 and 514 lie in shard zero, on opposite sides of its internal 512-element chunk boundary. This is not an index-overflow or 16-bit-token issue.

The tie adjustment's documented limitation is exact: the minimum is over **gathered candidates**, so a maximum discarded by local top-k cannot be recovered (`tt_sampling.py:936–940`). At maximum 30, its nonzero boost is ample: `30/64` yields an adjusted BF16 winner of 30.5. The index minimum and comparisons use int32, avoiding the obsolete FP32-index truncation problem described in nearby comments (`:967–990`). Those older comments are not evidence that the current code has that bug.

The formatter maps temperature zero to device `(k,p,temp)=(1,0,1)` (`models/common/sampling/generator.py:829–837`). A standard-library AST extraction of the actual formatter confirmed that scalar greedy defaults and explicit 32-slot greedy defaults produce equal parameter dataclasses. The sampling writer considers only one candidate for `k=1` (`operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp:142–155`); the changing seed does not introduce a choice between two candidates. Slot 30 uses ordinary lower-face offset `512+14*16=736`; there is no special last-active-row branch.

### Prefill collective audit

The projection output is trimmed to 5376 channels before the BFP8 reduce-scatter (`optimized_decoder.py:283–284`, `multichip_decoder.py:375–399`). At physical length 544 this is 17×168=2856 input tiles and 714 output tiles per chip. The dim-3 TP4 output width, 1344, is tile-aligned, so the ordinary ring contiguous-staging path applies.

Its reduction order is fixed, not selected by packet arrival: TP4 ring iterations forward, form designated two-term partials, then add a designated third contribution (`ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/ring_reduction.cpp:57–138`). Absolute eight-tile boundaries determine directional parity (`operations/experimental/ccl/reduce_scatter_common/kernels/common.hpp:20–25`). At the derived worker boundaries 178, 357, and 535, reader/writer staging addresses use a page index plus the within-page offset; transfers and compute use only valid tiles. Reserving/popping a full eight-entry CB chunk does not itself consume unproduced valid data here: unused entries are not packed or transferred. No contradiction was found in this boundary ledger.

RS startup also configures stochastic rounding off: `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_common.h:81` passes false for the rounding controls, and `common/inc/cunpack_common.h:839–847` writes the FPU/gasket/packer bits. The following gather performs byte movement, not reduction arithmetic. These facts argue against explaining repeat variation as an intended consequence of BFP8 rounding or arrival-order accumulation. They do not prove all collective/device execution is correct; fixed-input snapshots remain necessary to locate the defect.

## Candidate adjudication

| Candidate | Evidence and missing preconditions | Assessment / smallest intervention |
|---|---|---|
| Changing prefill logits / maximum membership | Directly observed before decode; changing max-set cardinality; all selected values remain maximal. Producer responsible and numerical magnitudes are unknown. | Leading divergent boundary. Locate the first varying intermediate; preserve exact logits and per-token values. Do not weaken the repeat assertion as a repair. |
| Local candidate truncation changes tie winner | Real source limitation and synthetic `sampler_acceptance.json` lowest-index mismatches. This run lacks per-shard counts and retained candidates, and does not show whether 179 is still maximal when 514 wins. | Plausible amplification/secondary issue. Full-vocabulary deterministic argmax or stable local tie selection can repair the tie policy, but cannot repair varying logits. |
| Sampler changes tokens for identical logits | Would require varying candidates, failed boost, or other sampler state. Twenty identical-input replays are stable, and the selected route is deterministic by design. | Not observed; do not headline. Repeating the fixed-input control across additional saved failing snapshots would test it further. |
| Scalar sampling broadcast repair changed greedy behavior | Actual formatter returns equal old/new defaults, and failure occurs before stochastic parameter tests. | Ruled out as a direct parameter-value cause for this failure. |
| Inactive four-to-zero-page reservation corrupts slot 30 | Active maps do not change; inactive row is skipped in prefill. A lower-level invalid-row side effect surviving reset would be an extra unproved condition. | No direct source contradiction. Revert only as a controlled diagnostic, not an evidence-backed fix. |
| Decode feedback/position drift is the first error | Initial outputs already differ at prefill token zero; first-generation feedback and position assertions passed. | Cannot be the initial within-request error. Prior-request decode contamination still requires a producer/reset investigation. |

## Other potential issues and limits

- **Deterministic tie order and lowest-ID parity are distinct.** `sampling_choice.md` describes deterministic greedy tie handling and explicitly disclaims lowest-ID parity for crowded ties; those statements are compatible. Read the common source's broad stable-sort/lowest-ID guarantee comments together with its retained-candidate exception. The known mismatch is lowest-global-ID parity, not evidence of non-maximal sampling or random choices on identical inputs.
- **Low precision is not a nondeterminism explanation.** Prefill projections use BFP4/BFP8 choices, and `multichip_decoder.py:378–399` performs BFP8 reduce-scatter plus gather. Those are useful producer-isolation boundaries, but fixed quantization and a fixed reduction order are deterministic. No aggregate PCC or small-delta result in these artifacts establishes that the observed variation is harmless.
- **Reset by multiplication is conditional on finite cache values.** `0*NaN` is not a zero fill. There is no observed NaN/Inf or demonstrated cache contamination here, so this is not a headline cause or a reason to claim the reset failed.
- **Prior success is token-level evidence.** Existing state JSON files do not establish bitwise logit determinism, and the synthetic sampler acceptance checks maximum-value selection rather than repeated model-state equality. Neither should be used to close the upstream issue.

## Minimal follow-up controls for the parent

These are proposed controls, not executions performed by this investigation.

1. Save the exact prefill logits for two divergent requests. For slot 30, record values at 177, 179, and 514; global lowest maximum ID; per-shard maximum counts; and each shard's local top-32 value/index pairs. If 179 is still maximal when 514 wins and was discarded locally, candidate truncation is proved. If 179 falls below the maximum, changed maximum membership already explains the switch. A top-eight excerpt cannot decide this.
2. Repeat the same reset/prefill with `max_new_tokens=1` so no generation decode updates intervene, retaining identical maps and seed parameters. If variation remains, isolate prefill alone. Separately force identical page-table refreshes each repeat to test the first-run-versus-cache-hit asymmetry. Compare embedding, layer-0 output/KV, layer-5 output/KV, selected hidden row, final norm, pre-softcap logits, and final logits in that order. Use fixed inputs at each boundary and compare per-chip values; identify the earliest divergence before changing precision or resetting hardware.
3. Where the first varying boundary includes a projection/collective, preserve its fixed input and snapshot before typecast, after typecast, after reduce-scatter, and after all-gather. Compare frozen-input repetitions. Only then rank a numeric kernel, reduction, or synchronization diagnosis.
4. Use the existing force-argmax comparison path as a tie-policy control on saved logits. Preserve each run's true maximum and lowest maximum ID; stable tokens under force-argmax would not establish stable logits. A production graph change also needs separately prepared/captured greedy and stochastic paths because this generator owns a single sample trace.

The investigation's conclusion is deliberately narrower than a completed root-cause fix: **prefill logits vary; the sampler returns maxima; the exact producer defect and whether local candidate truncation causes this particular ID switch remain open.**

## Report adjudication

After the first draft, the sampler claims received a separate source/evidence review. The wording was narrowed to the first **observed** divergent tensor boundary; deterministic tie order was separated from lowest-ID parity; and candidate truncation remained unproved for this real-logit failure. The collective and cache candidates were checked against their concrete dimensions and lowered paths and were not promoted without a source contradiction. The three source fingerprints above and the preserved late-allocation report were rechecked unchanged.
