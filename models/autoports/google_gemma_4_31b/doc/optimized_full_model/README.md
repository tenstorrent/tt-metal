# Gemma 4 31B Optimized Full Model

Stage: optimized-full-model (Stage 07)
Model: `google/gemma-4-31B` at revision `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`
Target: four Blackhole P150b devices, `MeshShape(1,4)`, TP4, `FABRIC_1D`
Implementation: `tt/model.py`, `tt/generator.py`
Status: complete; fresh independent `$stage-review` verdict `clean-pass`.

## Headline performance

| Matched full 60-layer batch-1 path, median of five | Unsharded before | Stage 07 after | Change |
| --- | ---: | ---: | ---: |
| Warmed token-out TTFT | 444.01 ms | 452.41 ms | +1.89% |
| Traced token-out, overall | 24.889 t/s/u | 24.987 t/s/u | +0.39% |
| Traced token-out, steady | 33.875 t/s/u | 34.182 t/s/u | +0.91% |

The matched source-current comparison uses the same constructed-generator protocol for both configurations: one discarded warmup followed by five recorded 149-prefill/100-output calls. The selected TTFT range is 445.42–455.09 ms versus 437.09–445.24 ms before; the 8.40 ms median cost is offset within the complete request by the decode gain. The earlier historical single samples (693.70/752.09 ms TTFT and 24.97/24.532 overall t/s/u) mixed setup/cache state and are retained only as provenance, not as the headline comparison.

Teacher forcing remains a separate measurement because it reads one prediction and writes one ground-truth token per step. Final Stage 07 teacher forcing is 841.55 ms TTFT, 23.15 decode t/s/u, and 19.54 end-to-end t/s/u. Token-out instead uses cooperating model and sampler traces, device token feedback, no per-replay host synchronization/readback, and one prefill-to-decode seed-token readback before capture.

| Teacher-forcing harness | Stage 06 before | Stage 07 after | Change |
| --- | ---: | ---: | ---: |
| TTFT | 1,793.14 ms | 841.55 ms | -53.07% |
| Traced decode | 22.79 t/s/u | 23.15 t/s/u | +1.58% |
| End-to-end | 16.30 t/s/u | 19.54 t/s/u | +19.88% |

The Stage 05 standalone optimized-layer target is unchanged:

```text
50 sliding * 0.463813 ms + 10 full * 0.5166275 ms
= 28.356925 ms/token = 35.2647 t/s/u
```

Those independently timed layer medians are a decoder-stack optimization target, not an additive physical lower bound for a captured full trace. The source-current reduced profile provides a like-regime operation-sum model: 50 sliding layers at 0.44878175 ms plus 10 full layers at 0.479633 ms, 0.025503 ms embedding/input work, and 1.879172 ms terminal plus sampler work gives 29.1400925 ms/token (34.317 t/s/u). The matched selected median is 29.254761 ms/token, leaving 0.114669 ms (0.39%) unmodeled. This is below the 10–15% gap trigger without mislabeling the difference from standalone layer medians as terminal cost.

## Optimization result

The Stage 06 terminal path used an interleaved BF16 TP-local LM-head projection. Stage 07 host-tiles the already tied BF16 embedding values, places each TP-local projection across all eight Blackhole DRAM views, and runs eight 8,192-column projections per device with `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`. The input is width-sharded into four logical shards; each split is converted to interleaved DRAM and concatenated in TP-local vocabulary order. The default is:

```text
weight/logit dtype       BF16 / BF16
math fidelity            HiFi2
DRAM views / input shards 8 / 4
local split width        8,192 columns
in0_block_w              2 K tiles
per_core_M               1 tile
```

The common `LMHead1D` implementation was reviewed first. Its optimized execution family is the one used here, but its TTTv1 factory expects a distinct `output.weight`, model-args DRAM-program/grid helpers, and `LazyWeight` ownership, while Gemma needs tied `embed_tokens.weight`, selective state loading, explicit tensor lifetime, exact TP-local ordering, and arbitrary full-logit M. The model-local implementation reproduces the common module's split DRAM-sharded linear -> sharded-to-interleaved -> concat topology and adds the required logical-M tiling.

The like-for-like reduced two-layer trace improves from 303.502 to 339.164 steady t/s/u (+11.75%). The selected four-shard/block-2 candidate measured 339.823 t/s/u; the final default repeat is within 0.2%, so default wiring reproduces the choice. `candidate_results.csv` is the compact ledger.

A repository search found no other Gemma 4 31B optimized-full-model artifact on this hardware. The strongest comparable reference is therefore the source-current Stage 06 full-model path in this checkout, plus Stage 05's optimized per-layer lower bound; both are used below rather than a weaker functional baseline.

| Candidate | Result | Evidence |
| --- | ---: | --- |
| Unsharded Stage 06 source-current baseline | 303.502 t/s/u | `reduced_token_out_baseline.json` |
| Split 4,096, block 7 | 334.004 t/s/u | `candidates/lm_head_dram8_split4096_block7_perf.json` |
| Split 4,096, block 3 | 335.032 t/s/u | `candidates/lm_head_dram8_split4096_block3_perf.json` |
| Split 8,192, 8 input shards, block 3 | 339.160 t/s/u; rejected after prompt-5 drift | `candidates/lm_head_dram8_split8192_block3_perf.json` |
| Split 8,192, 4 input shards, block 2 | 339.823 t/s/u; selected | `candidates/lm_head_dram4_split8192_block2_perf.json` |
| Split 8,192, 4 input shards, block 3 | 337.546 t/s/u; rejected | Slower; only 16.1% pre-softcap logits exactly match legacy and greedy changes 669 -> 108. |
| Split 8,192, 4 input shards, block 6 | rejected | L1 buffer at 1,351,040 clashes with static-CB end 1,381,120. |
| Split 8,192, 4 input shards, block 7 | rejected | Static CB reaches 1,581,824 bytes, above 1,572,864-byte L1. |
| Split 8,192, 4 input shards, blocks 14/21/42 | rejected | Static CB reaches 2,986,752/4,391,680/8,606,464 bytes. |
| Final selected-default repeat | 339.164 t/s/u | `reduced_token_out_final_perf.json` |
| Unsplit local vocabulary | rejected | 11,674,368-byte static CB exceeds 1,572,864-byte L1 |
| Split 4,096, block 21 | rejected | 2,294,528-byte static CB exceeds 1,572,864-byte L1 |
| Split 16,384, block 3 | rejected | L1 buffer at 1,394,048 clashes with static-CB end 1,434,368 |

Eight input shards/block 3 was the initial throughput selection, but its Fibonacci drift forced a compatible-geometry sweep. Four input shards expose 42 K tiles/shard: block 2 is selected; block 3 is slower and changes the aligned greedy winner; every larger legal divisor has an exact L1/CB blocker. Block 2 preserves the same eight physical DRAM views, restores the Stage 06 Fibonacci continuation exactly for 64/64 tokens, and is bit-identical to the legacy LM head. A broader precision frontier was deliberately not run here; `$datatype-sweep` owns that decision.

## Complete-path audit

| Subsystem | Stage 07 disposition and evidence |
| --- | --- |
| Embedding | Preserved BF16 hidden-column TP4 embedding; remains inside model trace. |
| Decoder stack | Preserved all 60 optimized multichip layers and Stage 05 program configs. |
| Residuals/norms | Preserved replicated BF16 DRAM inter-layer boundary and final RMSNorm. |
| Collectives | Preserved TP4 Linear/two-link asynchronous CCL and shared persistent L1 scratch; no replicated decoder stream. |
| KV/cache | Preserved BFP8 cache, 50 sliding physical-1,024 layers, 10 full physical-262,144 layers, explicit caller state, and fixed slots. |
| Position/RoPE/page tables | Persistent device position and RoPE advance; changed-only distributed page-table copy. |
| LM head/logits | Replaced only terminal placement/program geometry; BF16/HiFi2 and TP-local vocabulary order preserved; softcap remains device-side. |
| Greedy sampling | Preserved exact custom TP4 device sampler and lowest-token tie rule. No force-argmax, generic TopK, or full-vocabulary all-gather. |
| Top-k/top-p | Preserved the common device sampling path for non-greedy operation; not used to distort the greedy benchmark. |
| Feedback/readback | `tt_out_tok` is the next replay input; zero per-token host refresh, sync, or full-logit readback. |
| Generator orchestration | Preserved explicit cache/page-table/position/prompt-length/batch state, mixed prompts, inactive rows, reset/recreate, and serving-ready API. |
| Non-aligned prefill | Arbitrary valid prompt lengths remain public. Final norm executes once; normalized rows are projected in logical 1–32-row tiles and concatenated along sequence. Hardware covers 33 rows and readiness covers 249 rows. |

The Stage 05 selected decoder policy remains binding: attention BFP8/LoFi, MLP BFP4/LoFi, BFP8 KV cache, phase-specific CCL precision, and the replicated BF16 residual boundary. Stage 07 does not substitute any faster policy rejected by Stage 05. The existing rejection ledger for synchronous BFP8 CCL, persistent BF16-output decode CCL, BFP4 attention, 24-core MLP, Ring, fractured residuals, fused collective alternatives, larger prefill grids, and adapted block-sharded L1 prefill remains authoritative.

## Profiler conclusions

The reduced source-current token-out trace is signposted and retained losslessly as `profiler_raw_ops.csv.gz`; `tt_perf_report.csv`, `tt_perf_summary.csv`, and `tt_perf_report.txt` are the compact reports. `profiler_sha256.txt` binds all four selected artifacts. The selected block-2 window contains 155 device ops and 2.833 ms summed device work:

| Group | Device time | Share | Conclusion |
| --- | ---: | ---: | --- |
| Width-sharded matmuls | 1,935.15 us | 68.31% | Includes the selected terminal rows and preserved decoder matmuls. |
| Exact greedy local-winner kernel | 298.93 us | 10.55% | No generic TopK or force-argmax; no longer the dominant terminal blocker. |
| Async all-reduce | 50.28 us | 1.77% | Preserved persistent optimized CCL path. |
| Layout conversion: sharded-to-interleaved | 58.81 us | 2.08% | Terminal split outputs must become concat-compatible TP-local shards. |
| Layout conversion: interleaved-to-sharded | 38.89 us | 1.37% | Includes decoder and terminal input movement. |

Each selected LM-head row is BF16 x BF16 -> BF16, HiFi2, 178–179 us, and 96.4–96.6% of the report's modeled DRAM bandwidth. The decoder rows in the same report retain BFP8 attention and BFP4 MLP weights at LoFi. The report's 667.9 ms cross-device chronology gap is a merge artifact between asynchronously captured device streams, not a device-op duration or an untraced measured token boundary; the trace-counter JSON is the source for host-boundary claims.

## Correctness and qualitative evidence

The exact pinned reference is the Stage 06 `readiness_aime24_plain.refpt`: revision `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`, `GemmaTokenizer`, prompt shape `[1,149]`, continuation `[1,100]`, and top-k shape `[100,100]`. The base checkpoint exposes `chat_template=None`; per `$qualitative-check`, the control therefore uses exact plain-tokenizer completion rather than an invented chat template.

| Full 60-layer gate | Stage | Top-1 | Top-5 | Top-100 |
| --- | --- | ---: | ---: | ---: |
| Prefill, 100 positions | Stage 06 before | 91% | 100% | 100% |
| Prefill, 100 positions | Stage 07 after | 91% | 100% | 100% |
| Teacher-forced decode, 100 positions | Stage 06 before | 91% | 100% | 100% |
| Teacher-forced decode, 100 positions | Stage 07 after | 91% | 100% | 100% |

The original full-prefill run exposed a real multi-tile terminal bug: a 249-row readiness request was incorrectly fed to an M=1 DRAM-sharded matmul. `$autodebug` isolated the invalid program contract; `$autofix` changed the implementation to normalize once, project contiguous logical sequence tiles, concatenate along M, and apply softcap once. A 33-row hardware regression and the final four-shard/block-2 full 249-row readiness run now pass at 91%/100%/100% top-1/top-5/top-100. `AUTODEBUG_PREFILL_LM_HEAD.md`, `AUTOFIX_PREFILL_LM_HEAD.md`, and `run_prefill_check_pre_autofix.log` preserve the diagnosis and failure.

The 100-token autoregressive control keeps the correct first token, first differs from HF at zero-based token 1, matches 8/100 positions, and remains coherent English with zero adjacent repetitions and zero repeated trigrams. Across the six shared prompts, the selected block-2 path is token-for-token identical to the Stage 06 TT artifacts; two prompts are exact HF matches for all 64 tokens. Aggregate HF/TT agreement is 136/384 tokens with 134 shared-prefix tokens. Prompt-specific repetition in the two exact-HF corpus-completion controls mirrors HF rather than forming a TT-only loop. Mechanical metrics and the review disposition are retained in `qualitative/degenerate_output_check.json` and `qualitative/verdict.md`.

The controlled aligned A/B is stronger than trajectory agreement: all 262,144 pre- and post-softcap BF16 logits from the optimized block-2 LM head are bit-for-bit identical to the legacy interleaved LM head, with PCC 1.0, maximum delta 0, all 32 vocabulary blocks in identity order, and identical device/host greedy token 669. The rejected block-3 evidence is retained separately.

## Canonical split trace and runtime fallback audit

The measured full path uses two cooperating traces on one command queue. The model trace contains embedding, all 60 decoder layers, final norm, sharded LM head, softcap, and device position updates. The sampler trace reduces exact TP4 greedy winners into the same persistent token tensor consumed by the next model replay.

The final 100-token artifact records 99 model-trace replays, zero token host refreshes, two one-time position refreshes, two one-time RoPE refreshes, zero page-table refreshes, three setup/final synchronizations, one pre-trace sampled-token seed readback, and zero full-logit readbacks. Unchanged page tables cause no copies; the focused functional suite changes one table generation once and sees one distributed copy, then repeats the identity/generation with no copy.

- No single-chip/demo decoder branch, replicated stream, CPU decoder, or host logits occurs on the measured path.
- No per-token Python token-feedback loop, host argmax, full-vocabulary gather, or full-logit readback occurs.
- Persistent cache, token, position, RoPE, page-table, sampler, and CCL buffers remain bound through nonblocking replay.
- Explicit `host_sampling_compat=True` and readiness full-logit requests remain compatibility/test paths, not the optimized measurement.
- Both traces are released before reset, cache clear, prefill allocation, sampler-mode change, or request reuse.
- Bounds are checked before device work and replay; no silent truncation or aligned-length-only public fallback exists.

## Context contract

`doc/context_contract.json` retains the full 262,144-token HF context and records Stage 07's eight-view split placement. Weight dtype and payload are unchanged from Stage 06, so per-device accounting remains 10,908,115,456 bytes of physical model weights, 2,789,212,160 bytes of batch-1 KV, and 27,672,814,984 total accounted DRAM against 34,225,520,640 usable bytes. No context capability reduction is needed.

Production accuracy/performance evidence remains batch one. Mixed lengths and inactive fixed slots are hardware-tested at batch two/context 128. Full-context accounting admits at most batch three (974,281,336 bytes margin); batch four is physically short 1,814,930,824 bytes/device. Stage 07 does not reduce this contract.

## Device safety and commands

Hardware commands were serialized. Each run used `LD_LIBRARY_PATH=$PWD/build/lib` and `MPLCONFIGDIR=/tmp/mplconfig`. Health checks passed on four P150b devices. Watcher and profiler were never enabled together. Full Ethernet watcher instrumentation cannot fit the active ETH 25,600-byte configuration buffer because instrumentation expands the fabric program to 27,792 bytes; the scoped worker-watcher rerun used `TT_METAL_WATCHER_DISABLE_ETH=1` and passed the measured reduced path.

Primary commands and full transcripts are in `work_log.md`. Compact evidence includes:

- `run_prefill_check.log`, `run_teacher_forcing.log`, and the preserved pre-autofix failure;
- `autoregressive/`, `qualitative/`, and `token_out_no_readback.json`;
- `reduced_token_out_baseline.json`, `reduced_token_out_final_perf.json`, and `candidate_results.csv`;
- `profiler_raw_ops.csv.gz`, `tt_perf_report.csv`, `tt_perf_summary.csv`, and `tt_perf_report.txt`;
- focused JUnit XML for functional, autofix, performance, and scoped watcher runs;
- `AUTODEBUG_PREFILL_LM_HEAD.md`, `AUTOFIX_PREFILL_LM_HEAD.md`, and `stage_review.md`.

No vLLM implementation, registration, server, or integration work was started. The qualitative artifact name `vllm_qualitative_outputs.json` is only the established readiness schema.

## Limitations

- This base checkpoint/tokenizer has no chat template; instruction-like prompts can autocomplete corpus patterns instead of answering.
- Full-context cache provisioning and first-time tensor-cache construction are expensive; neither is a single-chip or host fallback.
- The selected BF16 LM head is DRAM-bound; Stage 07 optimized its legal placement and geometry without taking over `$datatype-sweep` precision selection.
- Production measurements are batch one; batch two is a mixed-slot functional test, and full-context batch three is only a physical capacity upper bound.
