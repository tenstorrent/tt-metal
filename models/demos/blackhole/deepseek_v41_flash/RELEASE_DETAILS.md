# DeepSeek-V4.1-Flash on Blackhole Galaxy: release details (test logs, bisect tables, long-form findings; the short release notes are in RELEASE_NOTES.md)

## This release: image `afb78699c8b-5b6f9268-dd4fbc6`

| | |
|---|---|
| Image | `dsv41-flash-local:afb78699c8b-5b6f9268-dd4fbc6` (local build, id `de6952f6f477`, not pushed to ghcr.io; offline copy `/mnt/tt-data/ssinghal/dsv4-docker-ctx/img_afb78699c8b.tar`, 5.38 GB; size on disk TODO; verified on hosts .34, .45, .33) |
| tt-metal | `afb78699c8b15104e8e9239d2bd1545a882091d9` (branch `ssinghal/dsv4p1-rebase-1017-v3`; the image is built from this commit, the documentation commits of branch `ssinghal/dsv4p1-rebase-1017-v3-docs` come after it and change documentation only) |
| tt-inference-server | `5b6f9268ac7a36a19766425c6cc721503efe7646` (branch `ssinghal/dsv4p1-rebase-1017`) |
| vllm-tt-plugin | `dd4fbc6230610c4d12a436eaa29d45f7d2bc6c29` (branch `ssinghal/dsv4p1-replay-v3`: `15bcbd6` prefill_final plus `dd4fbc6` prefill_total) |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Stack | the Python environment is reused from the previous build (previous release: vLLM 0.31.0, torch 2.11.0+cpu, transformers 5.12.1); not re-read from this image |
| Previous release | `0.24.0-0655530-4743e01` (section "Previous release" below). Images in between, not released: `a58ff4b5c2b-acf66ceb-6066a1e` (production image of the earlier checkpoint), `81284183e99-*` (1017 checkpoint), `74727b7f7ee-5b6f9268-15bcbd6` (v2, mHC and ids sort), `7ca53f64776-5b6f9268-15bcbd6` (v2.1, adaptive-k probe fix) |
| Build | `scripts/build_local_dsv41_image.sh` with the three commits above; the compiled tt-metal is reused, `models/`, the plugin and the tt-inference-server tree are replaced. Verification started on the image built without the documentation commits. |

<!-- sources: dsv4-logs/image_v3b/TAG, MS, PS, build.sh; ttis sha = git rev-parse in wt/ttis_rebase; intermediate images: dsv4-logs/image_tests/*/results.md headers, memory production image -->

### What is new since `0.24.0-0655530-4743e01`

| Feature | Status | How to turn it off | Measured |
|---|---|---|---|
| Bounded prefill replay | **New**, on by default (`DSV41_PF_REPLAY=2560`): layers 21-39 run only on the last 2,560 tokens of a cold prompt; exact | `DSV41_PF_REPLAY=0`; live switch `DSV41_PF_REPLAY_FILE` | 4 concurrent cold prompts, batch TTFT 8k / 32k / 64k / 128k: 6.66 / 19.65 / 37.94 / 75.02 s against 8.41 / 32.07 / 64.24 / 129.60 s (-21 to -42 %); needle 64k / 131k / 600k 25.8 / 50.7 / 231.1 s |
| Fused decode indexer score | **New**, default (`DSV41_IDX_FUSED=1`): one program replaces the key-slab slice, two matmuls, slice, untilize and mask of the speculative indexer | `DSV41_IDX_FUSED=0` | 8-layer verify window at position 65,536: k = 3 23.955 -> 20.679 ms, k = 5 27.935 -> 23.648 ms; 40-layer round at 64k about -9 to -12 ms |
| Drafter | **New**, default: no layout and gather round trips in the drafter round | `DSV41_DRAFT_ROUTER=chain DSV41_DRAFT_KEEP_PAD=0 DSV41_DRAFT_X_RM=0 DSV41_DRAFT_SAMPLE_RM=0 DSV41_DRAFT_CONF_RM=0 DSV41_SAMPLE_V2=0` | drafter alone k = 3 6.32 -> 4.86 ms, k = 5 6.35 -> 4.88 ms; round -1.4 ms |
| Attention front (qk norm and heads) | **New**, default | `DSV41_ATTN_QKPOST=0` (both), `DSV41_ATTN_QKNORM=0`, `DSV41_ATTN_QKHEADS=0` | 40-layer round -1.52 / -1.37 ms (GSM8K / 64k needle), bit-identical |
| Lean CCLs | **New**, default | `DSV41_CCL_LEAN=0` | 40-layer verify round k = 3 -0.232 ms, k = 5 -0.836 ms |
| mHC blocks (T-aware, fused front) | **New** (in the 1017 images), default | `DSV41_MHC_TAWARE=0`, `DSV41_MHC_PRE=0` | whole-layer mHC chain at T = 6 (k = 5 verify) 354.4 -> 120.2 us, T = 1 (plain decode) 174.8 -> 79.7 us; 8-layer window k = 5 -2.5 ms |
| ids sort of the selected index entries | **New** (in the 1017 images), default | `DSV41_IDX_SORT=0` | exactness: plain step and verify block read the same keys in the same order (64k boundary) |
| Adaptive k probe | **Fixed** (v2.1): the scheduler re-measures the other k every 128 rounds | `DSV41_SPEC_PROBE_EVERY=0` (old behaviour) | GSM8K eval: k = 5 in 95.5 % of the rounds (0.5 % before), acceptance length 4.18 (3.32), client decode at 4 concurrent greedy requests +11.5 % |
| Host-side server prefill | **New**, default | `DSV41_PF_RETURN_LAG=-1` (wait for the last window), `DSV41_PF_OVERLAP=0` | time outside the device in a 4 x 64k prefill 10.7 -> 1.35 s at 2,048 / 512 |
| 1017 checkpoint, APC and paged prefill on, single chunk trace, per-row chunk starts, budget 16,384 / 4,096 | Defaults of the specs and the catalog | see `RELEASE_NOTES.md` | see `OPTIMIZATIONS.md` sections 1 and 2 |

<!-- sources: replay: dsv4-logs/replay_study/RESULTS_phase3.md and OPTIMIZATIONS.md 8.6; indexer, drafter, qkpost, lean CCL: OPTIMIZATIONS.md 8.1-8.4; mHC: dsv4-logs/levers/mhc/README.md; ids sort: tt/indexer.py canonical_ids; probe: OPTIMIZATIONS.md 8.5 and image_tests/7ca53f64776-5b6f9268-15bcbd6/results.md; host-side prefill: OPTIMIZATIONS.md P17; switch names checked in the code of afb78699c8b (tt/replay_cfg.py, tt/spec_paged.py, tt/mtp.py, tt/attention.py, tt/ccl_lean.py, tt/mhc.py, tt/indexer.py, tt/spec_adapt.py) -->

### Switches that restore the previous behaviour

All switches are environment variables of the server (`-e NAME=value` in the `docker run`, or `env_vars` of the runtime spec). Defaults are the values of this release.

| Switch | Default | Value that restores the previous behaviour | Effect |
|---|---|---|---|
| `DSV41_PF_REPLAY` | 2560 | `0` | all 40 layers on every prefill chunk (values below 2414 are not exact) |
| `DSV41_PF_REPLAY_FILE` | unset | file holding `0` | live switch for A/B runs of one server (needs a server booted with replay on) |
| `DSV41_IDX_FUSED` | 1 | `0` | the chain of ops in the speculative indexer |
| `DSV41_DRAFT_ROUTER` / `DSV41_DRAFT_KEEP_PAD` / `DSV41_DRAFT_X_RM` / `DSV41_DRAFT_SAMPLE_RM` / `DSV41_DRAFT_CONF_RM` / `DSV41_SAMPLE_V2` | fused / 1 / 1 / 1 / 1 / 1 | `chain` / 0 / 0 / 0 / 0 / 0 | old drafter path (set all six) |
| `DSV41_ATTN_QKPOST` (`DSV41_ATTN_QKNORM`, `DSV41_ATTN_QKHEADS`) | 1 | `0` | old attention-front ops |
| `DSV41_CCL_LEAN` | 1 | `0` | stock all-reduce and combine all-gather |
| `DSV41_SPEC_PROBE_EVERY` | 128 | `0` | the scheduler never re-measures the other k |
| `DSV41_MHC_TAWARE` / `DSV41_MHC_PRE` | 1 / 1 | `0` / `0` | padded mHC wrappers / separate programs |
| `DSV41_IDX_SORT` | 1 | `0` | unsorted selected ids |
| `DSV41_PF_RETURN_LAG` | 1 | `-1` | a prefill step waits for its last window before it returns |

<!-- sources: defaults read in the code of afb78699c8b: tt/replay_cfg.py (2560), tt/spec_paged.py (IDX_FUSED "1"), tt/mtp.py (DRAFT_ROUTER "fused", DRAFT_KEEP_PAD "1"), tt/attention.py (QKPOST "1"), tt/ccl_lean.py (CCL_LEAN "1"), tt/spec_adapt.py (PROBE_EVERY "128"), tt/mhc.py (MHC_TAWARE, MHC_PRE "1"), tt/indexer.py (IDX_SORT "1"); restore values and the DRAFT_X_RM / DRAFT_SAMPLE_RM / DRAFT_CONF_RM / SAMPLE_V2 defaults of 1: OPTIMIZATIONS.md 8.1; DSV41_PF_RETURN_LAG default 1 read in tt/dsv41_model.py (windows allowed in flight), -1 restores the wait: OPTIMIZATIONS.md P17; DSV41_PF_OVERLAP default 1 in tt/dsv41_model.py; DRAFT_X_RM / SAMPLE_RM / CONF_RM / SAMPLE_V2 default 1 in tt/mtp.py and tt/device_head.py -->

### Measurements of this release

- Merged tree, 4 users, 1M build, 40 layers, 1017 checkpoint, adaptive k with a frozen time table (3:47, 5:58 ms): see the table in `RELEASE_NOTES.md`; all four scenarios reproduce plain greedy decoding for 4 of 4 users, with the same accepted tokens per round and k mix as the v2 image (source `dsv4-logs/levers/merged/MERGED.md`).
- Bounded prefill replay, server level (host .45, catalog default B=4 / 1M build): `dsv4-logs/replay_study/RESULTS_phase3.md`. Exactness gates (all pass): last-token logits bit-identical (4 x 8k and 4 x 32k, 40 layers; 8-layer server path for equal, mixed, piecewise, staggered and short prompts), server cold = APC hit = repeat for 32k / 64k late question / 128k needle, GSM8K 8-shot first 256 questions 253 / 256 for replay on and off with identical completions, needles 64k / 131k / 600k found, API suite 23 / 23, vision tests pass. Limits: R must be 2,414 or more; the zone is per mesh row (one prefill slot per row, `DSV41_PREFILL_UP=1`, the default); a request whose total length the plugin does not pass (plugin older than `dd4fbc6`) is treated as no skip.
- Not run for replay: the B=16 / 262,144 build, full GSM8K / GPQA / tau3 with replay on the server (the verification of this image runs them, see below), 1M needle, B=8.

### Testing done on this image

Verification of `afb78699c8b-5b6f9268-dd4fbc6` (results: `dsv4-logs/image_tests/afb78699c8b-5b6f9268-dd4fbc6/results.md`; same-host baselines of the v2.1 image: `image_tests/v21_ab_for_v3b/`; switch bisect: `image_tests/bisect_v3b/`). Lanes: A (GSM8K, GPQA), B (needles, determinism, cold TTFT, stall, 1M needle), C (API, overload, APC, multi-turn, vision, tools, bench), G (tau3), R and Y (decode and greedy-completion comparisons). All lanes boot the B=4 / 1,048,576-token catalog default (`lane.sh`); B=16 / 262,144 is not part of them.

| Test | Result |
|---|---|
| Boot, catalog default (four lanes) | 1,773 / 1,217 / 1,526 / 1,787 s; verify runners k = 3 and k = 5 built; DRAM free 417.7 MiB per bank after warm-up, 416.0 MiB at the end of the build (largest free block 210.1 MiB; trace region 76.8 MiB per bank allocated, 66.2 free); 0 server errors |
| API suites (smoke, sanity, serving_api, overload, multi-turn, tools, api_again), vision (client_img, client_many, client_stress, vis_apc, repeat_image) | all pass |
| GSM8K 8-shot | 96.21 (v2.1 96.21; gate 94.9); k = 5 in 96.3 % of the eval rounds (k_hist {3: 359, 5: 9369}), acceptance length 4.20, 3.07 accepted drafts per row-round, wall 23:48 |
| GPQA | 80.30 (gate 76.8) |
| tau3 airline | 74.0 % (gate 72 %; one 50-task run, binomial sd about 6 points) |
| Needles 64k / 128k / 600k / 1M | 33 / 55 / 240 / 396 s (v2.1 54 / 104 / 431 / 718 s), all found |
| Determinism 32k / 64k late question / 128k needle, 200 tokens | cold = APC hit = repeat, 3 of 3 |
| 4 concurrent cold prompts, batch TTFT 8k / 32k / 64k / 128k (host .45) | 4.95 / 14.47 / 26.89 / 51.97 s (v2.1 6.2 / 21.74 / 43.0 / 86.03 s) |
| Stall, 3 x 32k prompts arriving while users decode, longest gap | 3.95 s (v2.1 5.35 s; gate 6.2 s) |
| APC cold / hit, 8k / 16k / 32k / 64k | 4.01 / 2.65, 7.11 / 4.27, 12.96 / 7.18, 25.11 / 13.43 s (v2.1 5.43 / 2.91, 10.76 / 5.66, 21.09 / 10.75, 42.5 / 21.66 s); hit text = cold text. The 8k and 16k rows do not meet the old gate hit <= 0.6 x cold only because the cold request is faster; every hit is faster than on v2.1 |
| Multi-turn, turn 2 against cold, 32k / 64k | 7.16 / 12.94 and 13.44 / 25.07 s |
| Bench B=4, 4 requests, ISL 1,024 / 8,192 | 1649 ms / 27.2 ms / 94.4 tok/s and 5342 ms / 28.2 ms / 56.2 tok/s |
| GSM8K-prompt decode (c = 1 / 4, sampled and greedy, host .33, same-host v2.1) | +3.6 / +3.8 / +6.2 / +0.9 % |
| Long-context decode, 768 forced tokens, high-acceptance prompts (host .33) | 64k 96.19 against 78.99 tok/s per user (+22 %), 128k 91.4 against 76.7 (+19 %); the low-acceptance prompts (forced tokens after EOS) depend on the text the model emits and are not a speed comparison |
| Replay in the server | active: the per-call prefill statistics carry `replays_a_only` (32 for a 16k-token window group, 8 for 4k) and `replay_per_chunk`. The log line `REPLAY: R=2560 ...` is printed by the demo only and does not appear in the server log |

Not run: B=16 / 262,144, full eval workflows beyond GSM8K / GPQA / tau3. AgentX at B=4: TODO (run in progress, expected 13:00-13:45 UTC; fill from its results file, no number yet).

<!-- sources: image_tests/afb78699c8b-5b6f9268-dd4fbc6/results.md, tables and "Server facts" -->

### Benchmark tables of this image

TODO: ISL x OSL tables up to 1,048,576 tokens, speculation on (`dsv4-logs/bench_1m/` runs 1 to 3; they were taken on the v2.1 image `7ca53f64776-5b6f9268-15bcbd6` according to `bench_1m/PATHS.md`: re-run on this image or label them) and plain decode (speculation off, `dsv4-logs/bench_1m_nospec/`, running). One v2.1 point already exists: 1,048,320-token prompt, 128 output tokens, TTFT 732.6 s (c = 1) and 733.7 s (c = 4), TPOT 75.6 / 134.2 ms (`bench_1m/cache_root_b/workflow_logs/reports_output/benchmarks/report_*.md`).

## Single chunk trace, per-row chunk starts, no admission window (branch `ssinghal/dsv4p1-single`)

| Item | What it is | Switch | Measured (4 users, 1,048,576 tokens, adaptive speculative k, APC on, trace region 1.2 GB) |
|---|---|---|---|
| Single prefill chunk trace | ONE chunk trace at the build's maximum context, replayed for every chunk of every prompt whatever its length; the index-key slab is position indexed (current chunk + history, valid prefix read from a device tensor by the new `kv_len_tensor` argument of `indexer_score_dsa`); replaces the per-width small contexts (`DSV41_PF_SMALL_BUCKET`) | default when read mode + bf16 pool + the `kv_len_tensor` op are present (`DSV41_PF_SINGLE=0` restores the per-S_pad traces; a missing op logs the reason and falls back) | one prefill trace, +6.0 MiB/bank; per-chunk replay 99 ms at 16k, 102 ms at 64k (8-layer probe) against 237 ms for the old 1M-wide trace; APC snapshots / restore of the slab bit-identical |
| Per-row chunk starts | every mesh row replays ITS OWN chunk position in one trace replay (RoPE tables, masks, window ids, valid length, hand-off indices, drafter taps are per row), so users that vLLM admits step by step cost the same as aligned users; without it users at different positions needed one replay per distinct position (3 per 512-token step) | default with the single trace and one prefill slot per mesh row (`DSV41_PF_PERROW=0` restores the window loop); also for image-input launches (DSV41_VISION=1: the image mask and embedding override of a window are built per user at its own chunk start; not a precondition of anything else) | 4 users at different positions: logits bit-identical to each user's own lone prefill (max diff 0.0), 7 replays instead of 16 |
| No admission window | the adapter capability `prefill_admission_window_ms` defaults to 0; nothing is held | `TT_PREFILL_ADMIT_WINDOW_MS` / `DSV41_PREFILL_ADMIT_MS` stay as experiment knobs | window-off, old window loop, budget 67,584, 4 concurrent distinct prompts: TTFT 67.8 / 180 / 380 s at 32k / 64k / 128k, stall of a running request 43.8 s |
| First step of an idle engine capped (plugin `prefill_first_step_tokens`, branch `ssinghal/dsv4p1-single` of the plugin repo) | the first prefill step of an idle engine is one chunk: requests arriving a few ms later share the next step instead of trailing a lone request that took the whole budget; nothing is held | adapter capability `prefill_first_step_tokens` (default one chunk, `DSV41_PREFILL_FIRST_STEP`); plugin `TT_PREFILL_FIRST_STEP_TOKENS` | see the table below |
| Scheduler budget of the B=4 / 1M specs | `max_num_batched_tokens` 16,384 (APC spec), `long-prefill-token-threshold` 4,096, `DSV41_VLLM_MAX_PARTIAL=4` (all four prompts may be part-way through their prefill) | in `assets/runtime_spec_b4_k5_1048576*.json` (also the 1M vision specs) | |

4 concurrent distinct cold prompts, TTFT of the last first token (s), per-row single trace, first step capped, plugin `ff71d29` + the cap:

| ISL | budget 8,192 / threshold 2,048 | budget 16,384 / threshold 4,096 (shipped) | demo (aligned, same pool) |
|---|---|---|---|
| 32k | 28.0 | 25.8 | 22.9-26.5 |
| 64k | 53.8 | 49.2 | 45.6-49.7 |
| 128k | 110.7 | 100.5 | 98.0-99.9 |
| longest decode gap of a running request while 3 x 32k prefill | 3.5 s | 6.2 s | |

The remaining difference to an aligned call is the per-step host work of a call (about 0.2-0.3 s: Engram hashing and uploads of the first window, APC snapshot reads, key export), which a larger step amortises; a larger step lengthens the decode stall.

## Previous release: image `0.24.0-0655530-4743e01` (vLLM 0.31.0)

| | |
|---|---|
| Image | `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dsv4.1-flash:0.24.0-0655530-4743e01` |
| tt-metal | `0655530d60388057829d20dee426063ea51712ee` (branch `ssinghal/dsv4p1`; the image was built from this commit, later commits change documentation only) |
| tt-inference-server | `4743e018f94f274465731ae16189301368305f57` (branch `ssinghal/dsv4p1`; the catalog carries the production defaults: APC off, budget 67,584, see below) |
| vllm-tt-plugin | `ff71d290df8e95e5e956d49ba860478db2cb034c` (branch `ssinghal/dsv4p1`) |
| Stack | vLLM 0.31.0, torch 2.11.0+cpu, transformers 5.12.1 |
| Previous release | `0.24.0-c9cfb00-ffd1a57` (vLLM 0.31.0; before it `0.24.0-799a223-ffd1a57`). Candidates in between, not released: `0.24.0-8cfe120-84b529b`, `0.24.0-1fa9922-84b529b`, `0.24.0-9bf06a9-84b529b` |
| Build | an overlay on the candidate `0.24.0-0725831-84b529b`: the compiled tt-metal (C++), the Python environment, the vLLM install and the plugin are reused unchanged; `models/` is replaced (documentation-only differences to the candidate) and the tt-inference-server tree and the catalog `model_spec.json` are regenerated at `4743e018` (the DeepSeek-V4.1-Flash entries: APC off, `max_num_batched_tokens` 67,584). The candidate `0.24.0-0725831-84b529b` has the same code and specs and a catalog with APC on. |

## What is new since `0.24.0-c9cfb00-ffd1a57`

The previous release already shipped the 262,144-token / 16-user and the 1,048,576-token / 4-user specs, image input (also at 1M), speculative decoding k = 5 at those contexts, the plain-decode fix, and prefix caching + chunked prefill as opt-in `_apc` specs. This release does not add those features; it adds the items marked "new" below and measurements that did not exist before. "Measured" names the test; numbers marked (candidate) were taken on `0.24.0-1fa9922-84b529b` or `0.24.0-9bf06a9-84b529b`, which differ from this image only in the items listed in "What changed" below (the default specs).

| Feature | Status in this release | How to enable | Measured |
|---|---|---|---|
| Speculative decoding for requests deeper than 32,768 tokens | **New**: `DSV41_VLLM_SPEC_ISL_MAX` default 32768 -> unlimited (before, deeper requests decoded one token per step) | default | single request, 128 output tokens: 1,000,028-token prompt (4 users) TPOT 98 -> 37.4 ms; 250,028-token prompt (16 users) 74 -> 31.5-31.6 ms; `SpecDecoding metrics` lines in the server log (acceptance length 6.0 on the 1M prompt) |
| Small prefill context at max context >= 524,288 (1M build) | **New**: a second prefill-chunk trace at a 16,384-token context replays for short prompts | default for 524,288 and 1M; `DSV41_PF_SMALL_BUCKET=0` turns it off | short-prompt TTFT at 4 users / 1M: 0.73 / 0.68 / 0.74 s at 1 / 2 / 4 requests against 0.90 / 0.83 / 0.87 s without it; batched, new default against `c9cfb00` (same APC-off / 67,584 settings): 4 x ISL 8,192 TTFT 9.1 s against 16.9 s, 4 x ISL 1,024 1.6 s against 2.8 s (not isolated from the other changes) |
| Shared tt_transformers sampler | **New**: default on in plain decode (`DSV41_SHARED_SAMPLER`) and in speculative verify (`DSV41_SHARED_SAMPLER_SPEC`); `top_k` above 32 is clamped to 32, "no top_k" means the top 32 of the vocabulary | default; set both to 0 for the previous sampler | sampler time per step 1.2-1.9 ms against 1.5-3.6 ms (plain), 1.85 ms against 2.3 / 8.5 ms per verify round (16 users); in the batched benchmark the two samplers differ by less than 4 % (serving bisect V1 / V2) |
| Prefix caching (APC) | Available, **off by default** (the default specs set `no-enable-prefix-caching`); already opt-in in `c9cfb00` | `assets/*_apc.json` (APC on, token budget 4,096) | 100,000-token prefix at 16 users: 112 s cold, 42 s on a hit (candidate); 599,955-token prompt at 4 users / 1M: 1030 s cold, 144 s on a hit, restore at 524,288 tokens 7.8 s (candidate); cost: batched serving 2x slower at 16 x ISL 8,192 (see Known issues) |
| APC snapshot at 524,288 tokens or more | **Fixed**: the snapshot read one 512 MiB device slice and stopped the 1M server with an out-of-DRAM error; it now reads row pieces of at most 16 MiB (`SNAP_PIECE_MIB`) | automatic | a 1,000,028-token prompt and a 599,955-token prompt (twice) complete on the APC spec (candidate) |
| Chunked prefill interleaved with decode (`max_num_batched_tokens`) | Available; already in `c9cfb00` as opt-in. In the default specs the budget is 67,584, which is practically one large prefill call; the `_apc` specs use 4,096 for real interleaving | the budget in the spec | stall of other users while a 32k prompt prefills (16 users): 32.9 s at 67,584, 9.1-9.2 s at 4,096; the same budget costs 16 x ISL 8,192 serving speed (below) |
| 4 users at 1,048,576 tokens | Spec and image input existed in `c9cfb00`; **new**: long-prompt measurements and the fixes above | `assets/runtime_spec_b4_k5_1048576*.json` | 1,000,028-token prompt completes (candidate): TTFT 1711-1781 s, TPOT 37.4 ms; needles of 100k tokens at the start and the end answer correctly (116-169 s); not measured before: any prompt above 100,020 tokens |
| 16 users at 262,144 tokens | Spec existed in `c9cfb00`; **new**: long-prompt measurement | `assets/runtime_spec_b16_k5_262144_vision.json` | 250,028-token prompt: TTFT 283-292 s, TPOT 31.5-31.6 ms; needles of 100k at start and end correct |
| Batched serving speed against `c9cfb00` | Same settings as `c9cfb00` by default (APC off, budget 67,584) | default | 16 users: 16 x ISL 1,024 within 1.3 %, 16 x ISL 8,192 +5.2 % TTFT / -5.0 % tok/s (within the repeat spread); 4 users: 42-47 % lower TTFT, 25-34 % lower TPOT at 4 requests (see Performance) |
| Image input | Not new (16 images per prompt, also at 1M in `c9cfb00`) | `_vision` specs / default 262,144 spec | 48 random images, 16 concurrent: 48/48 numbers, 48/48 colours at 16 users (46/48 colours at 4 users / 1M) |
| Index-key hand-off offset | **Fixed** for the column-sharded prefill key FIFO: the offset used the per-column length (negative for a 100,033-token prompt, decode read zero keys for the early entries) | automatic | needle at the start of a 100,033-token prompt answers correctly on 16 users and 4 users; whether answers of `c9cfb00` were affected was not tested |

Limits of this release: APC is off by default because of its cost (the catalog of the image, tt-inference-server `4743e018`, and the default runtime specs agree on this); making APC cheap (asynchronous snapshots, `ssinghal/dsv4p1-apc-fast`), packing several prompts into one prefill call (`ssinghal/dsv4p1-pf-pack`) and paged-prefill APC (`PAGED_PREFILL_APC_DESIGN.md`, branch `ssinghal/dsv4p1-paged-apc-design`) are next-release work and not part of this image. Everything not listed with a measurement above was not measured for this release.

## What changed since `0.24.0-799a223-ffd1a57`

New since `0.24.0-9bf06a9-84b529b`: the default specs run with prefix caching off and a chunked-prefill budget of 67,584 (below). New since `0.24.0-8cfe120-84b529b`: the prefix-cache snapshot reads the kv table in row pieces, speculation above 32,768 tokens, an index-offset fix (below). New since `0.24.0-c9cfb00-ffd1a57`: the small prefill context at 1,048,576 tokens, the shared sampler in the speculative verify rounds (`DSV41_SHARED_SAMPLER_SPEC`, default 1), and the catalog defaults of tt-inference-server (a launch without a runtime spec serves 16 users at 262,144 tokens with prefix caching, chunk budget 4,096 and k = 5; `--impl deepseek-v41-flash-1m` serves 4 users at 1,048,576 tokens). The other items below are in both images.

- **Prefix-cache snapshot no longer needs one large DRAM buffer (4 users / 1,048,576 tokens).** The snapshot at 524,288 tokens of a long prompt sliced the kv table of each source layer on the device as one buffer (512 MiB, 64 MiB per bank) while the largest free DRAM block of the 1M build is about 41 MiB: the server stopped with a device out-of-memory error (`TT_FATAL`) about 14 minutes into a 1,000,000-token prompt, i.e. for any prompt that reaches 524,288 tokens with APC on. The slice is now read in row pieces of at most 16 MiB (`SNAP_PIECE_MIB`, `tt/prefix_cache.py`). Prompts of 1,000,028 tokens complete; a 599,955-token prompt asked twice restores the 524,288-token snapshot in 7.6-8.7 s.
- **Speculative decoding now also runs for requests deeper than 32,768 tokens.** `DSV41_VLLM_SPEC_ISL_MAX` defaulted to 32768 (deeper requests decoded one token per step); the default is now unlimited. Single-request TPOT of a 1,000,028-token prompt (4 users) fell from 98 ms to 37.4 ms and of a 250,028-token prompt (16 users) from 74 ms to 31.5 ms; the server log shows `SpecDecoding metrics` lines during these requests.
- **Index-key hand-off offset uses the full FIFO length when the prefill key FIFO is column-sharded** (`tt/dsv41_model.py`, `ix.KL_full`). No effect on the shipped replicated layout; needed by the column-sharded prefill FIFO that is in test.
- **Plain decode is fixed.** A server without speculative decoding used the fused decode-indexer backend and answered wrongly at 16 users (GSM8K 85.67 % at 131,072 tokens, 80.59 % at 262,144). Plain decode now uses the matmul backend up to 262,144 index entries, like the speculative servers (`DSV41_IDX_SWITCH` default 262144 for every launch). See "Known issues" for how this was verified.
- **Default: prefix caching (APC) off and chunked-prefill token budget 67,584 (decision of 2026-10-19).** The default runtime specs (16 users / 262,144 tokens, 4 users at 524,288 and 1,048,576 tokens) start the server with APC off (`no-enable-prefix-caching`, no `DSV41_APC`) and `max_num_batched_tokens` 67,584, the content of the former `_noapc` specs; the user asked for "64k", the measured value used here is 67,584 (the 16-user budget of the previous release). Why: with APC on and a 4,096-token budget batched serving was 2x slower than on `0.24.0-c9cfb00-ffd1a57` at 16 x ISL 8,192 (see "Batched serving against the previous release"); with both off it is back within about 5 % of it. Trade-offs: no prefix caching (a shared prefix is prefilled again by every request), and a long prompt stalls decoding of the other users while it prefills (32.9 s worst gap for a 32k prompt at 16 users, against 9.1 s with the 4,096 budget). The previous defaults stay available as opt-in specs: `assets/runtime_spec_b16_k5_262144_vision_apc.json`, `runtime_spec_b4_k5_1048576_apc.json`, `runtime_spec_b4_k5_1048576_vision_apc.json`, `runtime_spec_b4_k5_524288_vision_apc.json` (APC on, budget 4,096: shared-prefix bursts get their first token about 3x sooner, a 32k prompt stalls decoding for 9 s, batched serving is slower). The 131,072-token specs are unchanged (APC off, budget 67,584; APC is wrong with the unsharded prefill-indexer layout of those sizes). The catalog of the image (tt-inference-server `4743e018`) carries the same defaults, so a launch without a runtime spec also starts with APC off and budget 67,584 (checked on a launch through `run.py` with the catalog, below).
- **Sampling uses the shared tt_transformers sampler**, in plain decode (`DSV41_SHARED_SAMPLER`) and in the speculative verify rounds (`DSV41_SHARED_SAMPLER_SPEC`), both default 1; 0 selects the previous full-vocabulary sampler. It takes the top 32 candidates of the vocabulary per step: `top_k` above 32 is clamped to 32 and "no top_k" (the model card's recommended sampling: temperature 1.0, top_p 0.95 or 1.0) means the top-32 of the vocabulary (logged once at the first such request); candidates are bf16 roundings of the logits. Greedy rows (temperature 0 / `top_k` 1) stay the exact argmax and seeds keep their meaning. Sampler time per step: 1.2-1.9 ms against 1.5-3.6 ms (ours, top-k / full vocabulary) in plain decode, and 1.85 ms against 2.3 / 8.5 ms per verify round at 16 users.
- **Small prefill context at 4 users / 1,048,576 tokens (default for builds with max context >= 524,288).** The traced prefill chunk used to run at the max-context size, so a short prompt paid the O(context) work of the 8 indexer layers on every chunk (0.661 s per chunk replay at 1,048,576 against 0.428 s at 262,144). The build now captures a second chunk trace at a 16,384-token context (`DSV41_PF_SMALL_SPAD`); a prompt whose first call is certainly the whole prompt replays it, longer prompts use the max-context trace (prefix-cache snapshots are keyed by the context they were taken in). Measured at 4 users and 1,048,576 tokens, short-prompt TTFT 0.73 / 0.68 / 0.74 s at concurrency 1 / 2 / 4 against 0.90 / 0.83 / 0.87 s for the same build without it (the gap to the 262,144-token build is closed); TPOT is unchanged. Fitting the DRAM needs two settings that are set automatically when the context is active: `DSV41_KEY_EXPORT_FILL=1` (tile-native key export) and `DSV41_MASK_LEAN=1` (bf16 decode score mask, bit-identical). It costs trace-region headroom: 3.0 MiB per bank free at the end of the 1M build (9.0 without), with `trace_region_size` 650000000 as in the shipped 1M specs; do not lower it. Switches: `DSV41_PF_SMALL_BUCKET=0` turns it off, `DSV41_PF_SMALL_MIN_CTX` (default 524288) is the smallest max context that enables it. It is inactive below 524,288 tokens (the 16-user 262,144-token server is unchanged). Checks at 1M / 4 users: API suite 23/23, image input, needle 100,020 tokens -> 4817, GSM8K 98-99 %.
- **Image input at 4 users and 1,048,576 tokens** now works (`runtime_spec_b4_k5_1048576_vision.json`). The Engram reduce-scatter of the prefill trace capture is split into 4 pieces when vision is on (`DSV41_PFA_ENGRAM_RS_SPLIT`), which makes the contiguous blocks it needs 4 times smaller.

## What this release adds

### 1. Long context with more users per galaxy

The budget that fits one galaxy is about **1M tokens of context per mesh row** (4 mesh rows): users x context per row.

| Serving configuration | Users | Context | Speculative decoding | Image input | Runtime spec |
|---|---|---|---|---|---|
| **Default, 16 users** (APC off, budget 67,584) | 16 | **262,144** | k = 5 | yes | `assets/runtime_spec_b16_k5_262144_vision.json` |
| Smaller context, 16 users (the previous default) | 16 | 131,072 | k = 5 | optional | `assets/runtime_spec_b16_k5_131072.json` / `_vision.json` |
| **Long context, 4 users** (APC off, budget 67,584) | 4 | **1,048,576** | k = 5 | optional | `assets/runtime_spec_b4_k5_1048576.json` / `_vision.json` |
| APC + chunked prefill, 16 users (opt-in, budget 4,096) | 16 | 262,144 | k = 5 | yes | `assets/runtime_spec_b16_k5_262144_vision_apc.json` |
| APC + chunked prefill, 4 users (opt-in, budget 4,096) | 4 | 1,048,576 | k = 5 | yes | `assets/runtime_spec_b4_k5_1048576_vision_apc.json` / `runtime_spec_b4_k5_1048576_apc.json` |

Also measured on the same code (plain decode, not shipped as specs): 8 users at 524,288 tokens, and 8 users at 1,048,576 tokens does **not** fit (out of DRAM at warm-up).

What made it fit (all in this release):
- RoPE tables compacted from 7,168 to 768 B per position per chip (bit-identical), column-sharded from 262,144 positions.
- Dense prefill masks only where the dense path runs; chunked visibility.
- The prefill indexer's key FIFO, scores and visibility mask sharded over the 8 mesh columns (from 131,072 index entries, `DSV41_PIDX_SHARD`); the kv-table FIFO shifted in place in row pieces (`DSV41_PF_FIFO_CHUNK`).
- The decode indexer ranks each user's score row on its own when the concatenated rows would not fit a core's L1.
- Server warm-up reworked for long contexts: arange tables shared across layers, the index-key hand-off in 64K-entry pieces written in place, one eager step before each trace capture (prefill, decode, spec) so no program is compiled inside a capture.
- Still replicated: the prefill kv table (every mesh column holds the single KV head; sharding it is a separate piece of work).

### 2. Speculative decoding at long context

Spec decode (k = 5, lossless) now runs at 262,144 tokens (16 users) and 1,048,576 tokens (4 users) instead of stopping at 131,584. At 1M and 4 users: 3.31 accepted drafts per round, 41.2 tok/s per user, 1.81x plain decode on the demo (short GSM8K prompts). `DSV41_SPEC_MAX_CTX` is the final word on the context cap (policy and vLLM guard); the runtime free-DRAM check when the runners are built still applies. The verify step scores the key slab in pieces (`DSV41_SPEC_SCORE_PIECE`) and uses the matmul indexer backend above 131,584 tokens (forced at build; the fused backend is 75.3 vs 43.8 ms/token at 1M).

### 3. Image input

- Up to **16 images per prompt** (was 4). The model card sets no limit; this is the server's `--limit-mm-per-prompt {"image": 16}`. A request with more is refused with an OpenAI-style HTTP 400 error:
  `{"error": {"message": "At most 16 image(s) may be provided in one prompt. (parameter=image)", "type": "BadRequestError", "param": "image", "code": 400}}` and the server keeps serving.
- An image is at most `max_image_tokens` = 1,024 tokens (31 x 31 token grid, about 1302 x 1302 pixels, 8,649 ViT patches); larger images are resized, keeping the aspect ratio.
- Works together with speculative decoding and with the long-context configurations above that list image input: verified at 16 users / 262,144 tokens and 4 users / 524,288 tokens.
- At 4 users and 1,048,576 tokens image input needed a fix of the prefill trace capture (the vision tower, about 22 MB per DRAM bank, left no contiguous 21 MB block for the 160 MiB reduce-scatter buffer). The Engram reduce-scatter is now split into 4 pieces along the hidden dimension when `DSV41_VISION=1`; the result is the same up to bf16 summation order (test `tests/test_engram_rs_split.py`, PCC at least 0.99995). Verified in this image at 4 users / 1,048,576 tokens: image checks, 100,020-token needle, GSM8K, and with APC on.

### 4. Fixes

- **Plain decode no longer depends on the other users of its mesh row.** The fused rows-layout RoPE kernel rotated every row by the angle of the first row's position, so in a ragged batch each user's compressed latent was written rotated by its first row-mate's position (logit shifts up to 3.5 on 4 layers and 9.2 on 40 when a row-mate moved by one token). The per-row path is the default now (`DSV41_ROPE_ROWS_FUSED=1` restores the old kernel, about 4.6 ms/token faster in plain decode at 16 users). Speculative decoding already used per-row RoPE and is unchanged.
- **The engine no longer dies when more requests than slots are queued** (vLLM 0.31 admits waiting requests against `max_num_active_reqs`; the scheduler's prefill-only pass only lowered `max_num_running_reqs`). Before, the 5th concurrent request of a 4-slot server, or the 17th of a 16-slot server, killed the engine on an assertion.
- Speculative verify no longer pads the score rows of a ragged batch to 32 rows per user (768 MiB at 24 rows x 262,144 entries ran the 1M build out of DRAM).

### 5. vLLM stack

vLLM 0.31.0 with the tt plugin ported to it: request-validation refusals (logprobs / structured outputs under speculation, `prompt_embeds`, `prompt_logprobs`, block-output limits) answer with an OpenAI-style HTTP 400 instead of the bare empty 500 that 0.31 returns for a `ValueError`; the scheduler follows 0.31's admission limits (fix above). Installed with the plugin's `docs/install-vllm-tt.sh`.

## Performance

Method: GSM8K 8-shot prompts (about 700 tokens), 256 output tokens, `vllm bench serve` through tt-inference-server's `benchmarks` workflow run from the image, speculative decoding k = 5, one persistent server per configuration. TPOT in ms, TTFT in s, means; "tok/s" is total output throughput (greedy). Full tables are in `GETTING_STARTED.md`.

| Configuration | Concurrency | Greedy TPOT | Greedy TTFT | Sampled TPOT | tok/s |
|---|---|---|---|---|---|
| 16 users, 131,072 tokens (not a default any more) | 1 / 2 / 4 / 8 / 16 | 28.5 / 32.2 / 38.0 / 49.1 / 70.4 | 1.03 / 1.33 / 1.38 / 1.74 / 2.40 | 28.0 / 31.8 / 37.5 / 49.6 / 68.0 | 30.8 / 53.2 / 85.5 / 125.4 / 196.6 |
| 16 users, 262,144 tokens, images on | 1 / 2 / 4 / 8 / 16 | 26.4 / 34.8 / 36.1 / 53.4 / 71.0 | 1.05 / 1.26 / 1.44 / 1.74 / 2.56 | 29.3 / 35.2 / 39.0 / 52.2 / 72.5 | 32.8 / 50.0 / 94.4 / 120.7 / 189.4 |
| 4 users, 1,048,576 tokens | 1 / 2 / 4 | 29.8 / 35.2 / 45.5 | 1.49 / 1.74 / 1.86 | 30.9 / 39.6 / 44.8 | 28.1 / 47.3 / 74.5 |
| 4 users, 524,288 tokens, images on | 1 / 2 / 4 | 31.8 / 33.9 / 40.4 | 1.26 / 1.61 / 1.79 | 31.3 / 33.3 / 43.5 | 27.3 / 48.2 / 81.2 |

Against what the previous image `0.24.0-3dce07e-233b250` shipped (16 users, 131,072 tokens: greedy TPOT 27.7 / 33.7 / 43.7 / 52.5 / 80.1 ms, TTFT 1.24 / 1.51 / 1.54 / 1.96 / 2.75 s, 168.3 tok/s at 16 requests): the default configuration of this release is the same or faster at every concurrency (TPOT 0.8 ms higher at one request, 9.7 ms lower at 16; TTFT lower throughout; throughput up to 17 % higher at 16 requests), and the 262,144-token server has the same speed within the run-to-run noise (3-4 ms). The previous image's numbers were measured on other hosts, so part of the difference can be the hosts; an A/B run of the old image on the same host was not done. The two fixes of this release do not change speculative decoding speed (the 262,144-token server measured 29.95 / 33.0 / 41.4 / 49.8 / 73.9 ms greedy on the same configuration before them).
The 1M-token and 524,288-token servers are slower than the 16-user ones at the same concurrency (every decode step and every prefill chunk scores the index entries of the whole context; TTFT +0.3-0.4 s). A 100,020-token prompt prefills in 115-117 s at 16 users / 262,144 tokens, 132-134 s at 4 users / 524,288, and 158-159 s at 4 users / 1,048,576 tokens.

### Batched serving against the previous release (new default specs)

Method: `vllm bench serve` random prompts as in the benchmark workflow (`serving_bisect` harness: warm-up, then 3 repeats with a different seed per repeat, the same prompts for every server, ISL / OSL 128), images on, k = 5, one server per row; means over the repeats, spread = (max - min) / mean of the repeats (TTFT / TPOT). Previous release = `0.24.0-c9cfb00-ffd1a57` (spec `b16_new_262k_vis` / `b4_new_1m_vis`); new = `0.24.0-0725831-84b529b` with the default specs. 16 users: both on host .45; 4 users: both on host .46.

| Config | Row | Previous release TTFT ms / TPOT ms / tok/s (spread) | `0.24.0-0725831-84b529b` | Change |
|---|---|---|---|---|
| 16 users / 262,144 | 16 x ISL 1,024 | 3588 / 164.6 / 81.0 (4 / 8 %) | 3626 / 162.8 / 82.1 (6 / 9 %) | +1.1 % / -1.1 % / +1.3 % |
| 16 users / 262,144 | 16 x ISL 8,192 | 24904 / 499.9 / 22.9 (21 / 8 %) | 26208 / 522.7 / 21.8 (7 / 3 %) | **+5.2 % / +4.6 % / -5.0 %** |
| 16 users / 262,144 | 1 x ISL 8,192 | 8547 / 47.4 / 8.8 (1 / 2 %) | 8353 / 44.1 / 9.2 (2 / 12 %) | -2.3 % / -7.1 % / +4.5 % |
| 4 users / 1,048,576 | 4 x ISL 1,024 | 2752 / 90.2 / 34.9 (5 / 3 %) | 1590 / 67.5 / 48.0 (9 / 9 %) | -42 % / -25 % / +37 % |
| 4 users / 1,048,576 | 4 x ISL 8,192 | 16853 / 183.4 / 12.6 (0 / 25 %) | 9059 / 120.9 / 20.2 (1 / 24 %) | -46 % / -34 % / +61 % |
| 4 users / 1,048,576 | 1 x ISL 8,192 | 12138 / 45.2 / 7.2 (0 / 36 %) | 6474 / 45.2 / 10.5 (1 / 26 %) | -47 % / +0.1 % / +46 % |

- **16 users: two rows are outside the 5 % line, and 16 x ISL 8,192 is not fully back: TTFT +5.2 %, TPOT +4.6 %, output tok/s -5.0 % against the previous release.** The difference is inside the spread of the previous release's own repeats (TTFT 21 %, TPOT 8 %) and inside the host-to-host difference of the same configuration (the same config on `0.24.0-9bf06a9-84b529b`, run on .45 again, gave +5.1 % / +4.5 % / -5.3 %; up to 7 % was seen between .45 and .46 earlier). The 1 x ISL 8,192 TPOT -7.1 % has a 12 % spread and is not an improvement we claim.
- **4 users: faster than the previous release** at every batched row (the 1,048,576-token prefill context now sized by the prompt chunk instead of the maximum context). The TPOT spread of the 8,192 rows is 24-36 %.
- Stall of a 32k-token prompt arriving while other users decode (16 users, new default): worst gap 32.9 s, median gap 0.097 s, TTFT of the long prompt 32.8 s (4,096-token budget: 9.1 s).

### Prefix caching and chunked prefill (opt-in specs), measured with `tools/` clients on the rebased branch

Method: 16 prompts of about 727 tokens with a shared 8-shot header; a 32k-token prompt arriving 4 s after 8 short chats. Speculative decoding k = 5.

| Configuration | APC | Token budget | Burst TTFT p50 | Worst decode stall during the 32k prompt |
|---|---|---|---|---|
| 16 users, 262,144 tokens | off | 67,584 | 5.5 s | 32.7 s |
| 16 users, 262,144 tokens | off | 4,096 | 6.2 s | 9.1 s |
| 16 users, 262,144 tokens | on | 4,096 | 2.1 s | 9.1 s |
| 16 users, 262,144 tokens, images on, in this image | on | 4,096 | 2.2 s | 9.2 s |
| 4 users, 1,048,576 tokens | on | 67,584 | 11.9 s | 47.9 s |
| 4 users, 1,048,576 tokens | off | 4,096 | 15.5 s | 12.6 s |
| 4 users, 1,048,576 tokens | on | 4,096 | 12.3 s | 12.7 s |

Long shared prefix (needle prompt, cold then cache hits, 4 users and 16 users): a 100,000-token prefix answers in 39-46 s on a hit against 107-124 s cold at 16 users, and in 59 s against 166 s at 4 users / 1,048,576 tokens; a 300,000-token prefix 69 s against 504 s at 4 users / 1,048,576 tokens. The long prompt itself takes 1-3 s longer to its first token with a small budget (34.5 s -> 35.8 s at 16 users).
The burst numbers at 4 users queue 16 requests on 4 slots.

## Accuracy

GSM8K 8-shot (`dsv41_gsm8k_cot`, all 1319 questions, tt-inference-server `evals` workflow from the image, temperature 1.0, flexible-extract, standard error about 0.6):

| Server | Score |
|---|---|
| 16 users, 131,072 tokens | 95.22 % |
| 16 users, 131,072 tokens, image input on | 95.38 % |
| 4 users, 1,048,576 tokens | 94.92 % |
| 16 users, 262,144 tokens, image input on | **93.56 % and 93.48 %** (two runs on the same server) |
| Previous image, 16 users, 131,072 tokens | 95.45 % |
| **This image**, 16 users, 262,144 tokens, image input on (default spec) | **94.09 %** |
| **This image**, 16 users, 262,144 tokens, image input on, APC on, budget 4,096 | **95.60 %** |
| Plain decode, 16 users, 131,072 tokens, matmul indexer (previous image, `DSV41_IDX_BACKEND=matmul`) | 95.22 % (85.67 % with the fused indexer) |
| Plain decode, 16 users, 262,144 tokens, matmul indexer (previous image, `DSV41_IDX_BACKEND=matmul`) | 94.47 % (80.59 % with the fused indexer) |
| **This image**, plain decode (no speculative config), 16 users, 262,144 tokens, default settings | **93.40 %** |
| **This image**, plain decode (no speculative config), 16 users, 131,072 tokens, default settings | **95.83 %** |

The 16-user 262,144-token server scores about 2 points lower than the other configurations (about 3 standard errors, reproduced on a second run). It passes the workflow's gate (within 5 % of the model card's 93.0). Checks made to find the cause: vision on or off makes no difference (131,072 tokens: 95.38 % with, 95.22 % without; 262,144 tokens: 93.5 % with, 93.18 % without and with the column-sharded RoPE tables switched off), the new code is not the cause at the default context (95.22 % at 131,072 tokens on this image), and 163,840 tokens scored 94.54 %. **The cause of this 2-point gap is not identified** (the speculative servers already run the matmul decode indexer; a plain-decode server on the fused indexer scored lower, see "Known issues"). The 262,144-token server passes every functional test and has the speed of the default configuration.

## Requalification of `0.24.0-0725831-84b529b`

Image `0.24.0-0655530-4743e01` (id `c2a19efbe0aa` as built on host .46) has the same models/ code and runtime specs as `0.24.0-0725831-84b529b`, on which the checks below were made (the default specs on 16 users / 262,144 on .45 and 4 users / 1,048,576 on .45; the 16-user serving benchmark above on .45 and the 4-user one on .46). Checked on `0.24.0-0655530-4743e01` itself: (1) the catalog `model_spec.json` inside the image: the four DeepSeek-V4.1-Flash entries have budget 67,584, `no-enable-prefix-caching` and no `DSV41_APC`; (2) a launch generated from the catalog without a runtime spec (`run.py` with `--model deepseek-ai/DeepSeek-V4.1-Flash --tt-device blackhole_galaxy`, specs `b16_cat` and `b4_cat` of the qualification logs): the server starts with `enable_prefix_caching: False` and `max_num_batched_tokens: 67584` (16 users), batched benchmark of that server, one repeat on .45: 16 x ISL 1,024 TTFT 3612 ms / TPOT 171.1 ms / 78.4 tok/s, 16 x ISL 8,192 25937 ms / 459.8 ms / 23.9 tok/s, 1 x ISL 8,192 8623 ms / 46.9 ms / 8.8 tok/s (previous release: 3588 / 164.6 / 81.0, 24904 / 499.9 / 22.9, 8547 / 47.4 / 8.8); (3) the 1M catalog entry (`--impl deepseek-v41-flash-1m`) boots on .46 with `max_model_len` 1,048,576, 4 slots, `enable_prefix_caching: False`, budget 67,584, and the serving API suite passes 23/23. The defaults changed (APC off, budget 67,584) on top of the checked candidate `0.24.0-9bf06a9-84b529b`; the code is identical. Done so far (16 users / 262,144, default spec, host .45): API 23/23, overload OK, 48/48 images (numbers) and 48/48 (both colours), 100,020-token needle at the end 4817 (116 s) and at the start 4817 (117 s), GSM8K (256 questions, 16 concurrent, sampled, same questions every run) 94.9 / 96.5 / 93.8 % in three runs (mean 95.1 %, 9 / 13 / 16 wrong; two of the runs on host .46, one on .45), against 96.5 / 97.3 % on the candidate `0.24.0-9bf06a9-84b529b` (APC on, budget 4,096) on the same host .46 and 97.7 % on .45: the new default scores about 1.8 points lower on average, which is about two standard errors of a 256-question run and is not explained; see Known issues, cold needle prompts of 2k / 8k / 32k / 100k tokens answer 4817 (TTFT 2.1 / 8.0 / 32.2 / 108 s; a differently worded question about the 2k and 8k prompts answered `4819` and `481902`, the retrieval noise seen before), a 250,028-token prompt: TTFT 283.4 s, TPOT 31.6 ms, 32k stall 32.9 s. 4 users / 1,048,576 (default spec, host .45, vision on): API 23/23, overload OK, many images OK, 48/48 images (numbers) and 46/48 (both colours), 100,020-token needle at the end 4817 (165 s) and 100,033-token needle at the start 4817 (164 s), a 599,955-token prompt (no APC, cold) answers 4817 with TTFT 1002 s and no snapshot, a 1,000,028-token prompt completes with 128 tokens: TTFT 1694 s, TPOT 36.7 ms, `SpecDecoding metrics` lines in the log. Requalification of this tag: done (the GSM8K difference is open, see Known issues). The results of the previous candidate follow.

## Release-candidate checks on the candidate `0.24.0-9bf06a9-84b529b` (APC on, budget 4,096)

Basic runs only. Each configuration was served once; spec decode k = 5, APC on, token budget 4,096, vision on. The short-prompt benchmark sweep, the stall test and the baseline comparison were run on the candidate `0.24.0-1fa9922-84b529b` (same code except the two changes above): every row within 15 % of `0.24.0-8cfe120-84b529b`.

| Check | 16 users / 262,144 | 4 users / 1,048,576 |
|---|---|---|
| Serving API (23 checks), overload, many images | 23/23, pass, pass | 23/23, pass, pass |
| 16-concurrent image stress, 48 random images (number / both colours) | 48/48, 48/48 | 48/48, 46/48 |
| 100,020-token needle at the end / 100,033-token needle at the start | 4817 / 4817 (103 s / 123 s) | 4817 / 4817 (144 s / 169 s) |
| GSM8K 8-shot, 256 questions, full concurrency | 97.7 % | not run |
| APC prefix sweep, needle at the middle, 2k-100k tokens | correct cold and on the same-prompt hits at every length; the other-question hit answered `481907` (2k) and `4819` (8k), correct from 16k up (the same 2k miss occurred on the previous candidate); at 100k: 112 s cold, 42 s hit | not run |
| Prompt of about 250,000 / 1,000,000 tokens, 128 output tokens | 250,028 tokens: TTFT 291.8 s, **TPOT 31.5 ms** | 1,000,028 tokens: TTFT 1781 s, **TPOT 37.4 ms**; completes |
| Prompt of 599,955 tokens asked cold, then twice with another closing question | not applicable | 4817 each time; TTFT 1030 s cold, 144 s and 144 s on the hits (restore at 524,288 tokens: 7.8 s) |

## Testing done on this image

Each configuration was served once and every test ran against that one server (no restarts between tests). Scripts: `tools/serving_tests/` (README there).

| Test | 16 users / 262,144 | 4 users / 1,048,576 | 4 users / 524,288 (images) |
|---|---|---|---|
| Sanity (sampled, streaming, 8 concurrent questions) | pass | pass | pass |
| 20,020-token prompt, code found | 4817, 23.5 s | 4817, 32.3 s | 4817, 27.1 s |
| 100,020-token needle at the end / at the start | 4817 / 4817 (115.7 / 117.1 s) | 4817 / 4817 (157.7 / 159.2 s) | 4817 / 4817 (132.4 / 134.0 s) |
| Image checks (1 / 2 images, 4 image + 4 text at once, streaming) | pass | not applicable | pass |
| 16 images in one prompt, one maximum-size image, 17 images -> HTTP 400 | pass | not applicable | pass |
| 16-concurrent image stress, 48 random images (number / colours) | 16/16, 48/48, 48/48 | not applicable | 16/16, 48/48, 48/48 |
| Overload: 4 x slots requests at once, two waves | 64/64 on 16 slots, both waves | 16/16 on 4 slots, both waves | 16/16 on 4 slots, both waves |
| Serving API (23 checks) | 23/23 | 23/23 | 23/23 |
| Benchmarks (greedy, sampled) | done | done | done |
| GSM8K 8-shot | 93.56 / 93.48 | 94.92 | not run |

The 131,072-token configuration (16 users, no images; no longer a default) was re-measured too: sanity pass, the 20,020-token prompt (4817), the 100,020-token needle at the end and at the start (4817 / 4817, prefill 97.3 / 99.5 s), benchmarks, and GSM8K (95.22 %; 95.38 % with image input on).
Model-level device test of the RoPE fix (`tests/test_decode_batch_sensitivity.py`, 4 layers): the logits of a user are bit-identical (24 of 24 steps) whatever the other users of its mesh row do; the 40-layer variant shows run-to-run noise in the serving environment of the test wrapper (MoE fp32 accumulation / bfp8 flags) and was not conclusive there.

## Testing done on `0.24.0-c9cfb00-ffd1a57`

Each configuration was served once and tested against that server (tools in `tools/serving_tests/`).

| Test | 16 users / 262,144, images (default spec) | 16 users / 262,144, images, APC + budget 4,096 | 4 users / 1,048,576, images, APC + budget 4,096 |
|---|---|---|---|
| Sanity, 20,020-token prompt, 100,020-token needle | pass | not run | pass |
| Image checks, 16 images in one prompt, 17 images -> HTTP 400, 16-concurrent image stress | pass | pass (vision suite) | pass |
| Prefix sweep (needle prefix, cold then hits): 8k / 32k / 100k tokens | not applicable | correct at 32k and 100k; 8k other-suffix hit wrong (cold and same-prompt hit right) | correct at 16k and 100k |
| GSM8K 8-shot, workflow (flexible-extract) | 94.09 % | 95.60 % | not run |
| GSM8K 8-shot, sampled, 256 questions, own client | not run | 98.4 % and 97.7 % | on the branch (prior to the image): 96.9 % |

Plain decode on this image (no speculative config, 16 users, default settings): GSM8K 95.83 % at 131,072 tokens and 93.40 % at 262,144 tokens (flexible-extract, 1319 questions, workflow; the previous default scored 85.67 % and 80.59 %).
Serving-API suite and overload test passed on the 4-user / 1,048,576-token servers of this image (default and APC specs); benchmark workflows were run on those servers too (results not tabulated here).

## Known issues

- **Decode is slower with a deep context.** Single-request TPOT with speculation is 31.5 ms at 250,000 tokens (16 users) and 37.4 ms at 1,000,000 tokens (4 users), against about 23 ms at 16k; earlier measurements at 32k-131k (66-76 ms) were taken with speculation switched off above 32,768 tokens and are not repeated on this image. A column-sharded decode indexer is in test (not in this image).
- **Concurrent long prompts are prefilled one after another and the decode steps of the other users wait between prefill pieces.** 16 simultaneous 250,000-token prompts took 5,035 s (first token 320 s + about 312 s per request) with a TPOT of about 10 s; 4 simultaneous 1,000,000-token prompts took 6,751 s with about 13.7 s per token.
- **Prefix caching and the 4,096-token budget (the opt-in `_apc` specs) trade first-token time for decode stalls at high concurrency; the default specs no longer use them:** against the same image without them, 16 concurrent requests wait 15-24 % longer for the first token at 1,024-16,384-token prompts (TPOT is lower for long prompts). At one request, TPOT of 8k-10k-token prompts was 46 ms with APC on against about 30 ms with it off in the default sweep; a direct test (same prompt cold and as cache hits) shows no difference between cold and hit decode, so the cause is not found.
- **Batched requests are slower than on `0.24.0-c9cfb00-ffd1a57` with prefix caching (APC) and the 4,096-token budget (chunked prefill), the `_apc` specs of this release; the default specs of this release switch both off, which gives the previous release's batched speed back (16 x ISL 8,192: +5.2 % TTFT, -5.0 % tok/s, see "Batched serving against the previous release") but also leaves the two features off.** The table below was measured on the candidate `0.24.0-9bf06a9-84b529b` (same code). Measured 2026-10-19 on 16 users / 262,144 tokens, images on, k = 5, one server per row, the same prompts for every server (`vllm bench serve` random prompts as in the benchmark workflow, a different seed per repeat so that no request hits the prefix cache, 3 repeats, means). TTFT s / TPOT ms / output tok/s:

  | Server | 16 x ISL 1,024, OSL 128 | 16 x ISL 8,192, OSL 128 |
  |---|---|---|
  | Previous image `0.24.0-c9cfb00-ffd1a57` (APC off, budget 67,584) | 3.59 / 164.6 / 81.0 | 24.9 / 500 / 22.9 |
  | This image, shipped spec (APC on, budget 4,096) | 4.08 / 198.6 / 68.4 | 49.8 / 856 / 12.7 |
  | This image, APC off only | 3.35 / 165.5 / 82.0 | 48.9 / 865 / 12.8 |
  | This image, budget 16,384 only (APC on) | 4.42 / 173.9 / 75.6 | 35.1 / 693 / 16.4 |
  | This image, budget 32,768 only (APC on) | 4.45 / 183.2 / 71.9 | 28.0 / 608 / 19.1 |
  | This image, budget 67,584 only | 4.39 / 183.8 / 72.0 | 29.2 / 535 / 20.8 |
  | This image, APC off and budget 67,584 (`assets/runtime_spec_b16_k5_262144_vision_noapc.json`), two hosts | 3.57 / 162.7-167.8 / 80.2-81.7 | 25.5-26.2 / 522-536 / 21.5-21.7 |
  | This image, APC off, budget 67,584 and the previous sampler (`DSV41_SHARED_SAMPLER=0 DSV41_SHARED_SAMPLER_SPEC=0`) | 3.56 / 161.7 / 82.5 | 23.9 / 476 / 23.8 |

  APC causes the ISL 1,024 rows: every prompt is snapshotted at 512 and 1,024 tokens inside its prefill call (a device-to-host copy of about 90 ms each, `apc_snapshot_s` 182 ms of a 1,724 ms prefill against 1,532 ms without APC), about 11 s of each 125 s run in which no decode step runs. The 4,096-token budget causes most of the ISL 8,192 rows: concurrent prompts are prefilled in pieces of at most 4,096 tokens with one or two prompts per call (about 75 prefill calls and 300 s of prefill per 32 requests, against 15-18 calls and 180 s with 67,584 tokens, where up to 9 prompts share a call). An intermediate budget with APC on recovers the ISL 8,192 rows only in part (TPOT +39 % at 16,384 and +22 % at 32,768 against +71 % at 4,096 and +7 % at 67,584; 24-25 and 18-21 prefill calls per 32 requests) and leaves the ISL 1,024 rows at +23-24 % TTFT. With both off, on the host of the previous image's run, every number is within about 5 % of the previous image (ISL 8,192: TTFT +5.1 %, TPOT +4.5 %, tok/s -5.3 %; another host measured TPOT +7 %, the same configuration differs by up to 7 % between the two hosts); with the previous sampler as well, every number is within 5 % or better (ISL 8,192 TPOT -4.8 %), so the code of this image has no batched-speed regression of its own. With the shipped settings the sampler makes no difference (`DSV41_SHARED_SAMPLER=0 DSV41_SHARED_SAMPLER_SPEC=0`: within 3 % of the shipped spec on the same host); with both settings off the previous sampler measured 11 % lower TPOT at ISL 8,192 than the shared one on the same host, inside the 18 % spread of its repeats (not established). Switches: APC = `DSV41_APC=1` with vLLM `enable-prefix-caching` (off: remove both, `no-enable-prefix-caching`); budget = vLLM `max_num_batched_tokens` (4,096 shipped, 67,584 before). Trade-off: without APC, shared-prefix requests lose their cache hits (burst first token 2.1 s -> 5.5-6.2 s, a 100,000-token prefix 107-124 s instead of 39-46 s); with the 67,584 budget a long prompt is prefilled in one go and the other users' decoding stops meanwhile (32k-token prompt: worst stall 32.7 s instead of 9.1 s). The default-sweep comparison in the item above re-sent the same seed-0 prompts at every sweep point (vLLM logged a prefix-cache hit rate of about 40 %), so its APC numbers include cache hits. Data and scripts: `dsv4-logs/serving_bisect` (README).
- **Plain decode on the previous image `0.24.0-799a223-ffd1a57` answered wrongly at 16 users (GSM8K 85.67 % at 131,072 tokens, 80.59 % at 262,144 tokens; speculation 95.2 % and 93.5 %).** Cause: servers without a speculative config used the fused decode-indexer backend (default switch 65,536 entries); speculative servers ran matmul. With the matmul backend plain decode scored 95.22 % and 94.47 %. This image makes matmul the default up to 262,144 index entries for plain decode too, and it was verified on plain-decode servers of this image (no speculative config, no environment override): **95.83 %** at 16 users / 131,072 tokens and **93.40 %** at 262,144 tokens (workflow, flexible-extract; the old default scored 85.67 % and 80.59 %). The fused backend is still in the code (it is selected above 262,144 entries and by `DSV41_IDX_BACKEND=fused`) and still answers wrongly at 4 users per mesh row; why is not understood. Ruled out: the plain-decode RoPE change (the old fused RoPE kernel scores 74.68 %), the sampler (0.02-0.04 % mismatch against a CPU reference at the same random numbers), request admission, vision.
- **16 users at 262,144 tokens score about 2 points lower on GSM8K** (93.5 % against 95.2-95.4 % at 131,072 tokens), cause not identified (see "Accuracy"). The 131,072-token configuration is no longer a default; it stays available where that accuracy matters.
- **Prefix caching (APC) at 16 users / 131,072 tokens answers worse: sampled GSM8K 89.1 % and 87.5 % with APC and the large token budget, 88.7 % and 88.3 % with APC and the 4,096 budget, against 95.3 % and 98.0 % without APC and 97.3 % with only the 4,096 budget (chunked prefill).** So APC causes it, chunked prefill does not. The 131,072-token configuration does not shard the prefill indexer, the same unsharded key layout in which the restore is wrong with `DSV41_PIDX_SHARD=0` at 262,144 tokens (next item); the shipped APC specs (262,144 and 1,048,576 tokens) use the sharded layout and score 95.6-98.4 %. The cause inside the unsharded layout is not found. There is no APC spec for 131,072 tokens in the assets; do not enable APC (`DSV41_APC=1`) with the 131,072-token specs or with `DSV41_PIDX_SHARD=0`.
- **APC restore is wrong when the prefill indexer is forced to the unsharded layout at 262,144 tokens** (`DSV41_PIDX_SHARD=0`): cache hits answer wrongly from a 16,384-token snapshot upward (cold requests are right; `DSV41_ROPE_SHARD=0` alone is fine). The same layout is why APC is wrong at 131,072 tokens (previous item). The shipped APC configurations shard the indexer (from 131,072 index entries) and are not affected; cause not found.
- **APC is switched off for prompts that contain images** (the vision guard); text prompts of a vision server are cached.
- **Needle retrieval at about 8,000 tokens fails now and then even without a cache**, and single-digit errors at 64k appear cold as well (1 of 6 cold requests, 0 of 6 hits); this is retrieval noise of the model on filler text, not an APC defect.
- **The cost of APC (snapshots inside the prefill call) and of the small budget (prompts prefilled one or two per call) is next-release work, not fixed here:** an asynchronous snapshot is on branch `ssinghal/dsv4p1-apc-fast`, packing of several prompts into one prefill call on `ssinghal/dsv4p1-pf-pack`, and the design of paged-prefill APC is in `PAGED_PREFILL_APC_DESIGN.md` on branch `ssinghal/dsv4p1-paged-apc-design`.
- **GSM8K at 16 users / 262,144 tokens is about 1.8 points lower with the new default (APC off, budget 67,584) than with the `_apc` spec in same-host repeats:** 94.9 / 96.5 / 93.8 % (mean 95.1 %) against 96.5 / 97.3 % (APC on, budget 4,096) and 97.7 %; each run is 256 sampled questions (standard error about 1.2 points). Earlier APC-off runs of the same config on `0.24.0-c9cfb00-ffd1a57` scored 94.09 % (workflow) and 93.5 % (earlier notes: 16 users at 262,144 score about 2 points below 131,072), so this is the known 262,144-token level; the difference to the `_apc` spec is not explained.
- **Catalog (changed in this image):** tt-inference-server `4743e018` sets APC off and budget 67,584 for the four DeepSeek-V4.1-Flash entries (16 users / 262,144 and 4 users / 1,048,576, text and vision), so a launch without a runtime spec matches the default specs. Earlier images (`0.24.0-c9cfb00-ffd1a57` to the candidate `0.24.0-0725831-84b529b`) shipped APC on and budget 4,096 in the catalog.
- APC and chunked prefill are opt-in: the snapshots live in host memory (budget 6 GiB, snapshot grid at powers of two of the prompt length), and a small token budget raises the first-token time of a long prompt by 1-3 s.
- **The plain-decode RoPE fix is verified bit-exact on a 4-layer model** (24 of 24 steps on 7 batch compositions, `tests/test_decode_batch_sensitivity.py`); the same test on the full 40-layer model in this tree still shows differences between runs of the identical batch composition (max logit difference 2.4-3.3, deterministic), so independence from the other users of a mesh row is not shown at 40 layers on this tree. Speculative decoding (all shipped configurations) does not use the plain-decode path.
- **4 users at 524,288 tokens with images score about 91 % on sampled GSM8K (256 questions, 8 runs: 87.5-93.0 %), against 94.5-97.7 % at 4 users with the same spec at 262,144 tokens (97.3 / 94.5), 786,432 tokens (97.7) and 1,048,576 tokens (96.9-97.7).** Not caused by images (text-only 87.5 / 91.8 %), APC (on 88.7 / 93.0, off 91.4 / 91.4), the column-sharded RoPE tables (off: 91.0 %) or the sharded prefill indexer (off: 91.4 %). The specs differ only in the context length; the cause is not identified. Two more runs make it worse, not better: **plain decode (no speculation) at 4 users / 524,288 with images scores 69.1 / 69.5 %**, and the speculative server with `DSV41_POS_BUCKETS=0` (decode always on the full-length trace) scores 69.1 / 70.7 %. So the full-length decode path at this context looks weak and the position-bucket traces hide it for short prompts; requests that fall back to plain decode (penalties, `min_p` is refused) on this configuration are affected most. **Plain decode at 4 users / 1,048,576 tokens with images scores 51.6 / 52.7 %**, and the 524,288-token plain server 69.1 / 69.5 % (70.7 / 73.4 % with the fused indexer), while plain decode at 4 users / 262,144 tokens scores 93.0 % (first pass) like the 16-user server (93.4 %): plain decode at 4 users degrades with the context size, between 262,144 and 524,288 tokens. Switching the sharded RoPE tables off (69.5 %), the sharded prefill indexer off (67.2 %) or both off (73.4 %) does not recover the 524,288-token plain server, so neither feature is the cause. The wrong answers are fluent, not garbage: in the 1,048,576-token plain run the model reasons over the wrong numbers or the wrong problem (one answer solves a few-shot exemplar instead of the question), which points at long-range attention picking the wrong span rather than a sampling or numerical-noise problem (an observation from 110 wrong answers, not a located cause). Forcing the fused decode indexer at 524,288 does not change it (70.7 %), so the indexer backend is not the cause here. The speculative servers of the same sizes are not affected (97 % at 1,048,576), so only requests that cannot be speculated (frequency / presence / repetition penalty, the plain-decode fallback) lose accuracy on the 4-user long-context specs; keep those requests to the 16-user 262,144-token spec or shorter contexts. The cause is not identified. The shipped 4-user image spec is therefore the 786,432-token one (97.7 / 97.7 / 97.3 %); the 524,288-token spec stays in the assets as not recommended.
- **A plain-decode (no speculation) 4-user / 786,432-token server with images lost its host twice under load** (hosts .45 and .47; the first time during the API test `client disconnect mid-stream`, the second two minutes into the 16-way GSM8K run; both after the 23-test API suite had passed). On .47 the kernel log shows `pciehp: Link Down / Card not present` on dozens of PCIe slots at the same second, then I/O errors on the cache NVMe, then the container's log stream ends with no Python error. So the card / drive set dropped off the bus; no model error was logged. The speculative 786,432-token server (three GSM8K passes, request controls, 23/23 API tests) and plain servers at 524,288 / 1,048,576 tokens did not do this. Cause not identified (power or thermal event under that particular load is a guess, not a measurement); not a shipped configuration.
- **BBH (3-shot CoT, lm-eval `bbh_cot_fewshot`, 131,072-token server): the strict task score is 32.6 % exact match and is an extraction artefact, not model quality.** The model answers in its own chat format (`**(C) Negation or Antonyms**`, rambling checks) instead of the exemplars' "the answer is (X)", so the task marks the answer wrong or `[invalid]` (boolean_expressions 2 %, logical_deduction_five 1 %, but 98 % on three objects and 92 % on object counting). Re-scoring the same 6,011 multiple-choice / yes-no / number answers offline with a looser last-answer extraction gives about 79 % (a rough estimate, not a benchmark number; dyck_languages and word_sorting excluded). The published 86.1 is a base-model 3-shot score, so the chat endpoint with the chat template at T=1.0 is not an equivalent setup. Do not quote either number as the BBH result. The 4-user 786,432-token vision configuration scored 97.7 / 97.7 / 97.3 % on sampled GSM8K (3 runs).
- Full GSM8K was not run on the 4-user 1,048,576-token configuration with images or APC (a 256-question sample scored 96.9 % with images and 97.7 % / 96.9 % with APC on).

## Tests still to run

- **Decode speed at deep context:** measured with single requests (see Known issues); the numbers in the sweeps of the baseline reports (`tt-serving-qualification` skill) carry the rest.
- **A prompt close to the context limit:** run on this image (4 users: 1,000,028 tokens, TTFT 1781 s, TPOT 37.4 ms; 16 users: 250,028 tokens, TTFT 292 s, TPOT 31.5 ms). The 1M prefill takes about 30 minutes, not the earlier estimate of 12.
- **Request controls under speculation, tested on 8 live servers of this image (16 users / 262,144, 4 users / 524,288 and 1,048,576, text and vision, APC on and off):** every server gave the same answers: `logprobs`, structured output, `logit_bias`, `bad_words`, `allowed_token_ids`, `min_tokens` and `min_p` are refused with HTTP 400 and an explanatory message (`n > 1` with greedy sampling also 400); frequency / presence / repetition penalty, `top_k` / `top_p`, `seed`, `stop`, `max_tokens` and streaming return 200, and a batch of 4 penalty plus 4 plain requests at once answered all 8 correctly. No request produced a 5xx. (The earlier reading of the platform code that `min_p` is served without speculation was wrong.)

## Not in this release

- Prefix caching and chunked prefill as the default: they are opt-in specs; the default specs keep APC off and the 67,584-token budget.
- fp8 prefill kv table, sharding the prefill kv table over the mesh columns.
- 8 users at 1,048,576 tokens (does not fit).
- `logprobs`, structured output, `logit_bias`, `bad_words`, `allowed_token_ids` and `min_tokens` with speculative decoding (refused with HTTP 400, because the model-owned device sampler of the speculative round does not apply them; answering without them would silently change the request).

## Upgrade notes

- New runtime specs (above): 4 users / 1,048,576 tokens with images, and the two APC + chunked-prefill specs; the 16-user default is now the 262,144-token spec with images; the 131,072-token specs and the earlier long-context specs are unchanged. Use `SPEC=...` in step 3 of `GETTING_STARTED.md`.
- Startup of the long-context servers takes about 25-40 minutes (weight cache and kernel cache populated).
- Environment variables added or changed: `DSV41_SPEC_MAX_CTX` (final word on the spec context cap), `DSV41_VLLM_SPEC_GUARD` (0 disables the vLLM-side guard), `DSV41_ROPE_ROWS_FUSED`, `DSV41_KEY_EXPORT_PIECE`, `DSV41_SPEC_SCORE_PIECE`, `DSV41_DEC_PRECAPTURE_WARM`, `DSV41_SPEC_PRECAPTURE_WARM`, `DSV41_PIDX_SHARD`, `DSV41_PF_FIFO_CHUNK`, `DSV41_PF_PRECAPTURE_WARM`; new in this image: `DSV41_APC` (1 = prefix caching, with `--enable-prefix-caching`), `DSV41_PFA_ENGRAM_RS_SPLIT` (default 4 with `DSV41_VISION=1`, else 1), and `DSV41_IDX_SWITCH` now defaults to 262144 for plain decode as well.
