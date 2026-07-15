# Gemma 4 31B Optimized-Full-Model Work Log

## 2026-07-14

Selected skills: `$multichip`, `$optimize`, `$tt-device-usage`, `$qualitative-check`, `$autodebug`, `$autofix`, and `$stage-review`. Stage scope is the complete full-model/generator path on four P150b devices; no vLLM work was authorized or performed. Hardware commands were serialized, and watcher and profiler were not combined.

### Baseline and invariant contract

The implementation starts at Stage 06 commits `cc5b46623f0` and `203c4f909d9`. The full-stack baseline is 693.70 ms token-out TTFT, 24.97 overall t/s/u, and 33.86618 steady t/s/u. The source-current reduced baseline copied into this stage is 303.502387 steady t/s/u.

The decoder stack remains exactly the selected Stage 05 policy: attention BFP8/LoFi, MLP BFP4/LoFi, BFP8 KV storage, phase-specific CCL precision, TP4 Linear/two-link asynchronous CCL, one shared persistent scratch pool, and replicated BF16 DRAM residuals between layers. The 262,144-token context and serving generator contract remain unchanged.

### Topology audit and implementation

Profiling identified the tied BF16 LM head as the dominant avoidable terminal cost. The optimized implementation:

- host-tiles each tied-weight split before device placement, avoiding an 8+ MiB device-tilizer CB;
- distributes each TP-local split across all eight Blackhole DRAM views;
- width-shards the M=1 input into four logical shards;
- projects eight 8,192-column splits/device with block width 2;
- converts outputs to interleaved DRAM and concatenates in exact TP-local vocabulary order;
- preserves BF16 weights/logits, HiFi2, device softcap, TP-local logits, and exact split greedy sampling.

The complete generator path was audited: embedding, final and layer norms, 60-layer decoder, residual layouts, CCL, cache, page tables, positions/RoPE, LM head, logits, greedy and non-greedy sampling contracts, feedback, trace replay, reset, mixed prompt lengths, fixed slots, and inactive rows. No additional host boundary was introduced.

### Candidate ledger

`candidate_results.csv` contains the compact matrix. The initial split-8,192/eight-input-shard/block-3 path measured 339.159538 steady reduced t/s/u and repeated at 339.834362, but changed the Fibonacci control at token zero. `$autofix` isolated the change to LM-head accumulation geometry. Four-input-shard/block-2 restores the Stage 06 TT continuation exactly for 64/64 tokens, measures 339.823107 t/s/u as a candidate, and repeats at 339.164348 t/s/u as the final default (+11.75% over the 303.502387 baseline). Split 4,096 block 7/block 3 measured 334.004289/335.032349. Unsplit, block 21, and split 16,384 candidates have exact L1/CB blockers in their JUnit artifacts.

The selected four-input-shard geometry has 42 hidden K tiles/shard and legal block width 2; weight placement still uses all eight physical DRAM views. Review remediation swept every larger legal divisor in the compatible geometry. Block 3 passes but is slower at 337.545868 t/s/u and changes the aligned greedy token from 669 to 108. Block 6 clashes at L1 address 1,351,040 versus static-CB end 1,381,120; block 7 reaches 1,581,824 bytes beyond 1,572,864-byte L1; blocks 14/21/42 reach 2,986,752/4,391,680/8,606,464 bytes. The faster-looking eight-input-shard/block-3 path is also rejected by the controlled qualitative A/B. No datatype frontier sweep was run.

### Multi-tile prefill autofix

The first full `run_prefill_check` failed after the decoder because the 249-row full-logit readiness request reached an M=1-only DRAM-sharded program. `$autodebug` ranked the program-contract root cause; `$autofix` implemented normalize-once, contiguous 1–32-row terminal projection tiles, M-axis concatenation, and one final softcap. It also added static tile-range coverage and a 33-row full-logit hardware regression. The reduced hardware regression and the exact full 249-row readiness run pass. See `AUTODEBUG_PREFILL_LM_HEAD.md`, `AUTOFIX_PREFILL_LM_HEAD.md`, `run_prefill_check_pre_autofix.log`, and `run_prefill_check.log`.

### Accuracy and quality

```text
prefill:          top1 91/100, top5 100/100, top100 100/100
teacher forcing:  top1 91/100, top5 100/100, top100 100/100; TTFT 841.55 ms, decode 23.15 t/s/u, e2e 19.54 t/s/u
autoregressive:   100 HF and 100 TT tokens; first token matches, first difference at zero-based token 1, 8/100 positional matches; both coherent English, zero adjacent repetitions and zero repeated trigrams
qualitative:      six 64-token prompts; 136/384 HF token matches and 134 shared-prefix tokens; two exact-HF prompts; all six TT outputs exactly match Stage 06 TT; selected-vs-legacy aligned logits are bit-identical
```

The exact HF revision is `d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3`. `GemmaTokenizer.chat_template` is absent, so all HF and TT comparisons use the same plain exact-tokenizer completion rendering. No chat format was invented.

### Performance and trace evidence

```text
Stage 05 stack lower bound = 28.356925 ms/token = 35.2647 t/s/u
Stage 06 full token-out    = 29.527980 ms/token = 33.86618 t/s/u
Matched unsharded median  = 29.520719 ms/token = 33.87453 steady t/s/u; 444.01 ms TTFT; 24.88885 overall t/s/u
Matched Stage 07 median   = 29.254761 ms/token = 34.18247 steady t/s/u; 452.41 ms TTFT; 24.98654 overall t/s/u
Stage 06 reduced           = 303.502387 t/s/u
Stage 07 reduced default   = 339.164348 t/s/u (+11.75%)
```

The final full artifact records 99 model-trace replays, zero token host refreshes, two position refreshes, two RoPE refreshes, zero page-table refreshes, three setup/final synchronizations, one prefill-to-decode seed-token readback, and zero full-logit readbacks. Token-out uses nonblocking model/sampler replay and persistent token/position/RoPE/page/cache/CCL state, with no per-replay readback.

The source-current selected block-2 profiler window contains 155 device ops and 2.833 ms device work. Width-sharded matmuls are 1,935.15 us/68.31%; exact greedy is 298.93 us/10.55%; async all-reduce is 50.28 us/1.77%. Selected LM-head rows are BF16 x BF16 -> BF16, HiFi2, 178–179 us and 96.4–96.6% of modeled DRAM bandwidth. Decoder rows retain the selected BFP8/BFP4 LoFi policies. The profile-derived full-path operation model is 29.1400925 ms versus the matched observed 29.254761 ms, leaving a 0.39% unmodeled gap. The 667.9 ms cross-device merged chronology gap is not used as token latency.

The initial review correctly rejected the single-sample TTFT/overall comparison. Matched source-current runs use one warmup and five samples per constructed generator. Selected versus unsharded medians are +1.89% TTFT, +0.39% overall throughput, and +0.91% steady throughput; ranges and every per-sample counter are in `full_token_out_matched_{baseline,selected}.json`.

### Commands

All device commands set `LD_LIBRARY_PATH=$PWD/build/lib` and `MPLCONFIGDIR=/tmp/mplconfig`.

```bash
pytest -q models/autoports/google_gemma_4_31b/tests/test_full_model_contract.py

GEMMA4_31B_FULL_MODEL_RUN_REDUCED=1 pytest -q -s \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_reduced_full_model_prefill_split_greedy_and_trace

TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
GEMMA4_31B_FULL_MODEL_RUN_REDUCED=1 pytest -q -s \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_reduced_full_model_prefill_split_greedy_and_trace

GEMMA4_31B_FULL_MODEL_RUN_PERF=1 pytest -q -s \
  models/autoports/google_gemma_4_31b/tests/test_full_model.py::test_reduced_full_model_token_out_perf_signposts

python -m models.common.readiness_check.run_prefill_check \
  --model-dir models/autoports/google_gemma_4_31b \
  --reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D

python -m models.common.readiness_check.run_teacher_forcing \
  --model-dir models/autoports/google_gemma_4_31b \
  --reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --mesh-device P150_X4 --fabric-config FABRIC_1D

python -m models.common.readiness_check.run_autoregressive \
  --model-dir models/autoports/google_gemma_4_31b \
  --hf-model /home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3 \
  --mesh-device P150_X4 --fabric-config FABRIC_1D --max-new-tokens 100 \
  --output-dir models/autoports/google_gemma_4_31b/doc/optimized_full_model/autoregressive

python models/autoports/google_gemma_4_31b/tests/run_full_model_qualitative.py \
  --model-dir models/autoports/google_gemma_4_31b \
  --hf-model /home/odjuricic/.cache/huggingface/hub/models--google--gemma-4-31B/snapshots/d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3 \
  --prompt-source models/common/readiness_check/vllm_prompts.txt \
  --output-dir models/autoports/google_gemma_4_31b/doc/optimized_full_model/qualitative \
  --max-new-tokens 64 \
  --benchmark-reference models/autoports/google_gemma_4_31b/doc/full_model/readiness_aime24_plain.refpt \
  --benchmark-tokens 100
```

### Review and commits

The first independent review returned `more-work-needed` on compatible K-block coverage, compact-profile provenance, cross-regime lower-bound arithmetic, and unmatched full-path timing. `$autofix` isolated each issue; owner-side hardware evidence closed the block frontier and matched timing; compact reports were regenerated and hashed; and the arithmetic was replaced with a like-regime operation model. A different fresh independent reviewer then returned `clean-pass`; see `stage_review.md`. The path-scoped implementation/evidence commit SHA is recorded in the follow-up checkpoint entry below.
