# Optimized full-model work log

## Environment and preserved policy

- Date: 2026-08-12 UTC; starting SHA `78b12c0b520`.
- Hardware: four Blackhole p300c, 1x4 ring, TP4, two links; `tt-smi -ls --local` and ring smoke passed.
- Profiler and Watcher were kept separate. Hardware commands were serialized.
- Preserved decoder selection: BFP4/LoFi projections, BFP8 attention/MLP/CCL/KV, BF16 L1 width-sharded residual, persistent async CCL, qkv/o/gate/down core geometry 4/4/24/8.
- Unrelated untracked `third_party/tt-metal/` content was not touched.

## Measurements and decisions

1. Baseline `results/baseline/full_model_evidence.json`: warm TTFT 24.6575 ms, model 12.4168 ms, sampler 0.7937 ms, device-only 75.687 t/s/u, caller-visible 75.418 t/s/u.
2. Real-weight LM-head candidates: 8192 = 1.357854 ms, 16384 = 1.334443 ms, 32768 = 1.318926 ms. Selected 32768; tokens were identical. Artifacts are under `results/candidates/`.
3. Current-source reduced profiling shows the selected local LM matmul at about 183 us, no LM concat, and split greedy faster than full-vocabulary alternatives. `tt-perf-report` artifacts are under `tracy/current_source/`; `perf_summary.json` reconciles reduced operation attribution with 28-layer warmed/device/caller timing.
4. A corrected 32K depth sweep exposed 0.454048 ms/layer and 12.9443 ms at 28 layers. Narrow page tables and physically small KV caches did not close the gap. A 256-row RoPE probe reduced one-layer trace from 0.6850 to 0.5167 ms.
5. Shared grow-on-demand RoPE gives 0.285563 ms/layer, 0.230304 ms intercept, R² 0.999999918, and 8.22669 ms at 28 layers, matching the selected isolated 0.286597 ms layer.
6. Final: warm TTFT 23.5617 ms; model 8.22688 ms; split sampler 0.79357 ms; device-only 110.8359 t/s/u; caller-visible 110.3839 t/s/u; teacher-forcing trace 110.8127 t/s/u.

Rejected candidates retained as evidence: 8192/16384 LM partitions; force-argmax; generic full-vocabulary sampling; narrowing only the page table; shrinking only physical KV capacity. No datatype frontier was run.

## Gates run

- `full_model_evidence.py`: pass, fallback strict, 28 layers, trace/page-table/token-feedback checks.
- `full_model_depth_sweep.py --depths 1,14,28 --iterations 64 --max-context-len 32768`: pass.
- `full_context_coverage.py`: pass at 32,767 prompt + 2 requested tokens, all layers/pages.
- `full_model_contract_coverage.py`: pass for non-aligned/mixed/inactive/reset/greedy/stochastic paths.
- `full_model_batch32.py`: pass across all 28 layers and fixed slots.
- AIME24 prefill: 92/100 top-1, 100/100 top-5, 100/100 top-100.
- AIME24 teacher forcing: 93/100 top-1, 100/100 top-5, 100/100 top-100; `results/accuracy/run_teacher_forcing.log`.
- Autoregressive 100 tokens: pass; `results/autoregressive/`.
- Full 28-layer Watcher evidence: pass through all 128 split replays with no assert, NoC error, or corruption; `results/final/full_model_watcher.json`. Watcher timings are diagnostic only and are not used as performance claims.
- Authoritative strict-fallback LM-head sweep: `results/candidates/lm_head_program_config_sweep.json`. Legal DRAM K-block widths 1 and 3 and adapted terminal layouts were measured in one process with real BFP4/LoFi weights. Width 3 wins 1.31242 versus 1.439 ms token-out with identical tokens. The adapted 8-core interleaved-weight 1D matmul is physically blocked (2,192,192 bytes/core circular buffers versus 1,572,864 bytes L1); the runnable 16-core candidate including explicit reshard is slower at 1.33289 ms, again with identical tokens.

The qualitative-check workflow determined that this exact tokenizer has no chat template, so the shared autoregressive evidence and verdict correctly use base-model completion prompts with an HF control.

Nanobind lifetime diagnostics occur after successful runner output during interpreter shutdown. They recur across passing controls, occur after device close and metric emission, and did not affect numerical results or hardware health; classified as a binding cleanup warning outside the measured path.

## Independent review

The fresh independent stage review returned `clean-pass` after four review/fix
cycles. Required work: none. Hard-check gaps: none. The final review is recorded
in `stage_review.md`; the remaining LM-head advisor label is controlled by the
strict same-run geometry/layout sweep described above.

The stage implementation checkpoint SHA is appended after the local commit.
