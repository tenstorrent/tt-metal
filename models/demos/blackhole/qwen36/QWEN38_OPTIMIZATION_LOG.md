# qwen38 → qwen36 optimization log

Branch `atupe/qwen38-optimizations` (local, uncommitted). Plan: `QWEN38_OPTIMIZATION_PLAN.md`.
Machine: QB2 (4× Blackhole p300c, P300x2), weights `/home/runara/models/Qwen3.6-27B`, ISL 128 batch 1 unless noted.
Columns: TTFT and decode t/s/u from `text_demo.py -k "traced_128 and not traced_128k"` (`[TP 4-dev]` line; two runs), accuracy from `-k accuracy_512` (top-1 / top-5, targets 97 / 99), output = first part of the generated text for the traced_128 prompt.

Metric note (from row 1.6): rows 0-1.4 report the host decode loop's mean per-step wall time (1 / mean step time incl. host input update, device sync and token readback, first step dropped). Row 1.6 (device-resident loop, default for greedy single-user decode) reports steady-state tokens / (2nd enqueue -> final drain + history read); it is NOT step-by-step comparable. The same-build host-loop number (`QWEN36_DECODE_DEVICE_LOOP=0`) is given in row 1.6 for an apples-to-apples delta.

| # | Change | Flag (default) | TTFT | decode t/s/u | Δ vs previous | accuracy_512 | Output coherent & on-prompt? | Notes |
|---|---|---|---|---|---|---|---|---|
| 0 | Baseline (= main behavior) | QWEN36_DECODE_ALLREDUCE=0 | 0.13s / 0.12s | 27.74 / 27.74 | — | 98.44 / 100.00 | yes: fluent near-verbatim continuation of repeated prompt | traced_4k: ttft 0.46s, 27.62 t/s/u; b8: ttft 13.91s (includes cold JIT), 22.15 t/s/u/user, 177.2 tok/s aggregate |
| 1.1 | Replicated residual + all_reduce_async (2 CCL/layer instead of 4) | QWEN36_DECODE_ALLREDUCE (1) | 0.13s / 0.13s | 28.77 / 28.82 | +1.03 / +1.08 t/s/u (+3.7% / +3.9%); -1.29 / -1.35 ms/token (36.05 -> 34.76 / 34.70) | 98.44 / 100.00 | yes: fluent; one near-tie token differs (`plain taste`) | traced_4k: ttft 0.46s, 28.67 t/s/u (+1.05); b8: ttft 0.69s (warm cache), 22.73 t/s/u/user (+0.58), 181.9 tok/s aggregate (+4.7) |
| 1.3 | Lean attention decode (fused K+V update, fused q/k-norm weight, RoPE w/o transposes, sigmoid fused into gate mul; q returned to DRAM for paged SDPA) | QWEN36_ATTN_DECODE_LEAN (1) | 0.12s / 0.12s | 29.38 / 29.50 | +0.61 / +0.68 t/s/u (+2.1% / +2.4%) vs row 1.1; -0.72 / -0.80 ms/token (34.76 -> 34.04 / 34.70 -> 33.90) | 98.44 / 100.00 | yes: fluent, on-prompt; two small insertions vs the prompt (`the burgers,` and `of the mayonnaise`), text differs from 1.1 (as expected with changed numerics) | first attempt crashed at B=1 (paged SDPA TT_FATAL: Q must be DRAM, got L1), fixed by returning q in DRAM (`out_memory_config`); traced_4k: ttft 0.46s, 29.24 t/s/u (+0.57 vs 1.1); b8: ttft 0.69s, 22.93 t/s/u/user (+0.20 vs 1.1), 183.4 tok/s aggregate (+1.5); unit: lean paged PCC prefill 0.99987081 (= legacy) / decode 0.99994766 (legacy 1.0), peruser B8 & B32 min=max=1.00000 (PASS) |
| 1.2 | GDN fused decode (KDA causal conv + chunk_gated_delta_rule + sigmoid_gated_rms_norm; replaces ~69-op chain), pad/dealloc use-after-free fixed | QWEN36_GDN_FUSED_DECODE (1) | 0.13s / 0.13s | 32.38 / 32.37 | +3.00 / +2.87 t/s/u (+10.2% / +9.7%) vs row 1.3; -3.15 / -3.01 ms/token (34.04 -> 30.88 / 33.90 -> 30.89); cumulative vs baseline row 0: +4.64 / +4.63 t/s/u (+16.7%) | 98.63 / 100.00 | yes: fluent, on-prompt, no garbage; traced_128 repeats the prompt sentence through `hoisin sauce` with one small glitch (`Do you don't prefer`), traced_4k opens a coherent `<think>` analysis of the AI-history paragraph (row 1.3 opened with an empty think block, a near-tie flip) | first attempt (fused ON, `L3_gdn_fused`) FAILED: garbage output, acc 0.20, 32.37 / 32.41 was not valid; root cause ttnn.pad returning a VIEW + deallocating its source (use-after-free), fixed in `_forward_decode_fused` (no op added/removed, so speed identical to the invalid run: 32.38 / 32.37 vs 32.37 / 32.41); traced_4k: ttft 0.47s, 32.13 t/s/u (+2.89 vs 1.3; invalid run 32.18); b8: ttft 0.68s, 23.01 t/s/u/user (+0.08 vs 1.3, noise: fused path inactive at B>1), 184.1 tok/s, text identical to row 1.3 b8; accuracy 98.63 / 100.00 (row 1.3 98.44 / 100.00); unit (fused=1, fixed): G1-G6b all PASS, PCC equal to the fused=0 values within 1e-3 (G1 0.99997, G2 0.99990, G3 0.99925 / 0.99977, G4a min 0.99992, G4b 0.99992, G4c 0.99998 / 0.99989, G6a 0.99980 / 0.99986 / 0.99990 / 0.99992, G6b worst 0.99895); padded rows 1..31 of beta_p / g_p verified exactly 0 on device |
| 1.5 | DRAM-sharded multi-reader matmul for decode attention QKV (bf8, HiFi2, 2 readers/bank, 10-core [32,512] input) | QWEN36_QKV_DRAM_SHARDED (1) | 0.12s / 0.14s | 30.89 / 32.28 | -1.49 / -0.09 t/s/u vs row 1.2 (mean 31.59 vs 32.38, -0.79); +1.49 / +0.09 ms/token (30.88 -> 32.37 / 30.89 -> 30.98); supplementary warm rerun 32.16 (+0.20 ms/token); NO gain, the microbenchmark gain (~-0.25 ms/token) did not show up end to end | 98.44 / 100.00 | yes: fluent, on-prompt, no garbage; text differs from row 1.2 (at the near-tie after `different dishes.` it opens a `<think>` block instead of repeating `Do you don't prefer ...`) | **REVERTED** (decode mean 31.59 < 32.275 = row 1.2 mean 32.375 - 0.1; perf1 30.89 was the cache-miss run, but the warm runs 32.28 / 32.16 are also at or below row 1.2). traced_4k: ttft 0.46s, 32.13 t/s/u (= row 1.2, +0.00); b8: ttft 0.69s, 23.06 t/s/u/user (+0.05 vs 1.2, noise), 184.5 tok/s aggregate (+0.4); unit (lean=1): paged decode PCC 0.99994357 (prev. lean 0.99994766), prefill 0.99987081, peruser B8 & B32 min=max=1.00000 (PASS); the patch is reverted (`tt/attention/tp.py` byte-identical to the pre-1.5 file); the 16 extra `.dsh2` cache files (1.25 GB) were left on disk, nothing reads them with the flag code removed |
| 1.4 | Traced exact-bucket short prefill (prompt length exactly 128/256/512/1024 tokens) + on-device first-token argmax | QWEN36_PREFILL_BUCKET_TRACE (1) | 0.08s / 0.08s | 32.39 / 32.41 | TTFT -0.05 / -0.05 s (0.13 -> 0.08, -38%) vs row 1.2 (row 1.5 was reverted, so the previous state is row 1.2); decode +0.01 / +0.04 t/s/u (noise, decode untouched); cumulative vs baseline row 0: TTFT 0.13 / 0.12 -> 0.08 / 0.08, decode +4.65 / +4.67 t/s/u (+16.8%) | 98.63 / 100.00 (= row 1.2) | yes: first generated token identical to row 1.2 (` bringing`), the first 68 characters identical, then (near-tie at ~the 12th token) `Do you don't prefer ...` becomes `<think></think>` + a fluent on-topic answer (`As an AI, I don't have taste buds ... personal favorite in the traditional sense`), no garbage | **KEPT.** Scope: only prompts whose length is EXACTLY 128 / 256 / 512 / 1024 tokens (traced_128 and accuracy_512 prompts); other short lengths would need persistent GDN masks in `tt/gdn/` (not done). Capture line `Short-prompt prefill trace (TP, exact bucket N, on-device argmax) captured.` present in perf1 / perf2 (bucket 128) and acc (bucket 512), absent in t4k and b8; the 2048-chunk trace is skipped when the short trace is used. t4k (2642-token prompt, unaffected path): ttft 0.47s (= row 1.2), 32.08 t/s/u (-0.05 vs 1.2, noise), text identical to row 1.2; b8 (unaffected path): ttft 0.68s (= row 1.2), 23.02 t/s/u/user (+0.01), 184.2 tok/s aggregate, text identical to row 1.2 |
| 1.6 | Device-resident greedy decode loop (on-device token feedback, RoPE table lookup, in-trace position increment, deferred readback) | QWEN36_DECODE_DEVICE_LOOP (1) | 0.08s / 0.08s | 32.85 / 32.85 (device-loop steady-state metric, see Notes) | +0.46 / +0.44 t/s/u (+1.4% / +1.4%) vs row 1.4 (32.39 / 32.41); -0.43 / -0.41 ms/token (30.87 -> 30.44 / 30.85 -> 30.44). **The metric changed**: rows 0-1.4 are the host loop's mean per-step wall time (host update + sync + readback per token, first step dropped); row 1.6 is device-loop steady-state tokens / (2nd enqueue -> final drain + one history read). Apples-to-apples, same build, same session: host loop (S0, `QWEN36_DECODE_DEVICE_LOOP=0`) 32.43 (30.84 ms) -> device loop 32.85 (30.44 ms) = +0.42 t/s/u (+1.3%), -0.39 ms/token. Cumulative vs baseline row 0 (27.74, 36.05 ms): +5.11 t/s/u (+18.4%), -5.61 ms/token (device-loop metric); host-loop metric (S0) +4.69 (+16.9%), -5.21 ms | 98.63 / 100.00 (= row 1.4; accuracy_512 is teacher-forced and keeps the host loop) | yes: text identical to row 1.4 and to S0 (host loop, flag 0) = S2 (strict device loop, flag 2): 226 chars, no differing position; fluent `<think></think>` + `As an AI, I don't have taste buds ... personal favorite in the traditional sense` | **KEPT.** traced_4k (device loop, 99 steps): ttft 0.47s, 32.59 t/s/u (+0.51 vs row 1.4's host-loop 32.08; 30.69 ms/token), text identical to row 1.4 (all 459 chars); b8 (batched, old host path, no `[TP device-loop]` line): ttft 0.69s, 23.00 t/s/u/user (-0.02 vs 1.4, noise), 184.0 tok/s aggregate, text identical to row 1.4; determinism_128 with `QWEN36_DECODE_DEVICE_LOOP=2`: PASSED (both device-loop runs 32.85 t/s/u, identical text); no `falling back` / `build failed` / `not used` line in any log; host only enqueued traces (0.1 ms for all 49 enqueues) and read the token history once |

## Details

### 0 — Baseline

- Flags: `QWEN36_DECODE_ALLREDUCE=0` (everything else default; no other QWEN* env set)
- Date/time (UTC, 2026-10-02): 22:59:47 - 23:04:05; log dir `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L0_baseline/`; run via `runs/T1/measure.sh L0_baseline` (traced_128 x2, accuracy_512, traced_4k, batched_128_b8; all five pytest invocations EXIT=0, models imported from the worktree).
- Raw `[TP 4-dev]` lines:

```
perf1: [TP 4-dev] ttft=0.13s decode=27.74 tok/s
perf2: [TP 4-dev] ttft=0.12s decode=27.74 tok/s
```

- accuracy_512:

```
Top-1 token accuracy: 98.44%  Top-5 token accuracy: 100.00%
```

- traced_4k (2642-token prompt):

```
[TP 4-dev] ttft=0.46s decode=27.62 tok/s
```

- batched_128_b8:

```
[TP 4-dev B=8] ttft=13.91s per-user-decode=22.15 tok/s aggregate=177.2 tok/s
```

- traced_128 prompt (reconstructed host-side with the Qwen3.6-27B tokenizer exactly as `text_demo._get_prompt(128, ...)`: the 110-token condiment paragraph from `input_data_questions_prefill_128.json` is repeated and clipped to 128 tokens, so the prompt stops mid-way through the repeat, after `each`; same prompt for every row of this table):

```
'What is your favorite condiment? There are so many condiments to choose from, each bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy the tangy zest of salsa or the smooth and savory taste of aioli. Share what your favorite condiment is and why you love it. Does it remind you of a specific dish or meal?What is your favorite condiment? There are so many condiments to choose from, each'
```

- Generated text, traced_128 perf1 (full, 50 tokens; perf2 identical = True):

```
' bringing its unique flavor and texture to elevate different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce?\n\n'
```

- Generated text, traced_4k (first 300 chars):

```
'\n\n<think>\n\n</think>\n\nBased on the text provided, the history of artificial intelligence can be summarized through the following key milestones:\n\n1.  **Antiquity and Myth**: The origins of AI trace back to ancient myths and stories featuring artificial beings endowed with intelligence by master craft'
```

- Generated text, batched_128_b8 row 0 (first 300 chars; the test asserts all 8 rows identical):

```
' bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the perfect balance of sweet and spicy in sriracha, or perhaps something more exotic like hoisin sauce'
```

- Judgement: coherent and on-prompt. The prompt is the condiment paragraph, repeated and cut after `each`; the model continues ` bringing its unique flavor and texture to elevate different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce?` which is the repeated sentence copied almost verbatim (`elevate` for `enhance`) and then closes with a paragraph break, exactly what a model should do here. traced_4k opens with an empty think block and starts a correct, structured summary (Antiquity and Myth, Classical Philosophy, ...) of the AI-history text. The b8 row-0 text is fluent but paraphrases the sentence with a small drift (`the perfect balance of sweet and spicy in sriracha`), still sensible. Note: the b8 TTFT of 13.91s in this baseline run most likely includes first-use JIT kernel compilation of the grouped batched-prefill path (the 1.1 run, executed afterwards, shows 0.69s, presumably with the warm kernel cache), so do not compare b8 TTFT between rows 0 and 1.1.

### 1.1 — Replicated residual + all_reduce_async

- Files changed: `models/demos/blackhole/qwen36/tt/{tp_common.py,layer.py,model.py,mlp.py,attention/tp.py,gdn/tp.py}`.
- Description:
  1. Decode residual is replicated full-width ([1,1,B,dim]) in L1 on a 40-core (10x4) grid, with local sharded RMSNorms (no all-gather before each norm).
  2. One `all_reduce_async` per row-parallel output (attention/GDN out-proj, MLP down-proj) into a persistent L1 buffer replaces all-gather + reduce-scatter: 2 CCLs per layer instead of 4.
  3. The embedding is all-gathered once at model entry; prefill is unchanged and keeps the fractured residual.
- Flags: none set; `QWEN36_DECODE_ALLREDUCE` default = 1 (log line: `QWEN36_DECODE_ALLREDUCE=1: decode uses the replicated 40-core L1 residual + all_reduce_async (gdn_fp32=False, num_links=2, topology=Topology.Ring)`; no `ignored` warning); `QWEN36_DECODE_AR_GDN_FP32` default 0 (GDN out-proj partial cast to bf16 before the all-reduce)
- Date/time (UTC, 2026-10-02): 23:04:21 - 23:06:20; log dir `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L1_allreduce/`; run via `runs/T1/measure.sh L1_allreduce` (traced_128 x2, accuracy_512, traced_4k, batched_128_b8; all five pytest invocations EXIT=0, models imported from the worktree).
- Raw `[TP 4-dev]` lines:

```
perf1: [TP 4-dev] ttft=0.13s decode=28.77 tok/s
perf2: [TP 4-dev] ttft=0.13s decode=28.82 tok/s
```

- accuracy_512:

```
Top-1 token accuracy: 98.44%  Top-5 token accuracy: 100.00%
```

- traced_4k (2642-token prompt):

```
[TP 4-dev] ttft=0.46s decode=28.67 tok/s
```

- batched_128_b8:

```
[TP 4-dev B=8] ttft=0.69s per-user-decode=22.73 tok/s aggregate=181.9 tok/s
```

- traced_128 prompt: identical to the one in section 0 (128 tokens, ends with `... choose from, each`).

- Generated text, traced_128 perf1 (full, 50 tokens; perf2 identical = True):

```
' bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the plain taste of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe'
```

- Generated text, traced_4k (first 300 chars):

```
"\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User Input:**\n   - **Input Text:** A highly repetitive paragraph about the history of AI. It repeats the same core message multiple times:\n     - Began in antiquity with myths/stories of artificial beings created by craftsmen.\n     - Classical phil"
```

- Generated text, batched_128_b8 row 0 (first 300 chars; the test asserts all 8 rows identical):

```
' bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the burgers, the creamy richness of mayonnaise, on fries, the spicy kick of mustard, on sandwiches, or perhaps something more exotic like'
```

- Delta vs row 0 (baseline): decode 27.74 -> 28.77 (perf1) / 28.82 (perf2) t/s/u = +1.03 / +1.08 t/s/u (+3.7% / +3.9%), i.e. 36.05 -> 34.76 / 34.70 ms/token = -1.29 / -1.35 ms/token. traced_4k: 27.62 -> 28.67 t/s/u (+1.05, +3.8%; 36.21 -> 34.88 ms/token, -1.33 ms). b8: 22.15 -> 22.73 t/s/u per user (+0.58, +2.6%), aggregate 177.2 -> 181.9 tok/s (45.15 -> 43.99 ms/step, -1.15 ms). TTFT unchanged (0.13s / 0.13s vs 0.13s / 0.12s; prefill is untouched). accuracy_512 identical: 98.44 / 100.00 (targets 97 / 99).
- Judgement: coherent and on-prompt. traced_128 continues the repeated sentence ` bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the plain taste of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe`. It diverges from the baseline text at the near-tie token `creamy richness` -> `plain taste` (expected: the all-reduce changes the summation/rounding order vs reduce-scatter, so greedy free-running text need not be bit-identical to the baseline); the continuation stays fluent and on-topic and teacher-forced top-1/top-5 are unchanged, which points to a near-tie token flip rather than degradation (not verified at logit level). traced_4k now opens a thinking block (`Here's a thinking process: 1. Analyze User Input: ... A highly repetitive paragraph about the history of AI ...`) that correctly describes the input text (the baseline opened an empty think block and answered directly; both are valid behaviors). b8 row 0 is a fluent list elaboration (`the burgers, the creamy richness of mayonnaise, on fries, ...`), slightly odd but grammatical and on-topic. perf1 and perf2 texts are identical to each other (deterministic across runs).

### 1.3 — Lean attention decode

**Status: first attempt FAILED at B=1 (failure record kept below, not a perf result); fixed and re-measured, see "Fix" at the end of this section (table row 1.3 holds the fixed result).** Note: 1.3 was measured before 1.2 (cumulative order: 1.1 -> 1.3; 1.2 is not applied in these runs).

- Files changed (uncommitted, worktree `qwen38-optimizations`): `models/demos/blackhole/qwen36/tt/attention/tp.py` (+`_forward_decode_lean`, `_kv_key_shard_cfg`, `_decode_lean_eligible`), `models/demos/blackhole/qwen36/tt/attention/rope_tp.py` (+`apply_partial_rope_decode_single`). Ported from `models/demos/qwen38_27b_qb2` `_full_decode`.
- Op sequence per attention layer (B=1): q|k|v slice straight to L1; V kept sharded; q/k RMSNorm with fused (1+w) weight; partial RoPE via `rotary_embedding(token_index=0)` without transposes; one `paged_fused_update_cache` (K on core (1,0), V on (0,0)) instead of 2 pads + 2 `paged_update_cache`; gate applied on the flat SDPA output with the sigmoid fused into the multiply (removes gate reshape, sigmoid, concat-heads chain). Harness op count 42 -> 23 per attention layer.
- Flags: `QWEN36_ATTN_DECODE_LEAN` default "1" (L2_attn_lean: no QWEN* env set, so lean on; `QWEN36_ATTN_DECODE_LEAN_MAX_B` default 32; `QWEN36_ATTN_DECODE_LEAN_FUSED_SIGMOID` default "1"). L2b_attn_lean_nofusedsig: `QWEN36_ATTN_DECODE_LEAN_FUSED_SIGMOID=0`. `QWEN36_DECODE_ALLREDUCE` default 1 (row 1.1 behavior) in all runs.
- Date/time (UTC, 2026-10-02): L2_attn_lean 23:15:15 - 23:16:52; unit tests 23:17:14 - 23:18:05; L2b_attn_lean_nofusedsig 23:18:59 - 23:20:29. Log dirs `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L2_attn_lean/` and `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L2b_attn_lean_nofusedsig/`; run via `runs/T1/measure.sh <label>` (models imported from the worktree). Exit codes (both labels): perf1 EXIT=1, perf2 EXIT=1, acc EXIT=1, t4k EXIT=1, b8 EXIT=0. Last-80-line tails of each failing log: `<label>/{perf1,perf2,acc,t4k}_last80.txt`; full extract output `runs/T1/L2_extract.txt`, `runs/T1/L2b_extract.txt`.
- Failure (identical in perf1, perf2, accuracy_512, traced_4k, for both fused and unfused sigmoid): in `tt/attention/tp.py:786 _forward_decode_lean` -> `ttnn.transformer.paged_scaled_dot_product_attention_decode`:

```
TT_FATAL @ ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp:93: Q_memcfg.buffer_type() == tt::tt_metal::BufferType::DRAM
info: Q tensor buffer type must be DRAM when not sharded but got BufferType::L1
```

  Likely cause (read-only inspection, not verified by a fix): at B=1 `apply_partial_rope_decode_single(..., memory_config=_L1)` returns q as an L1-interleaved tensor, which the SDPA-decode validation rejects (it needs DRAM-interleaved or sharded q). The legacy path and the lean B>1 path (`apply_partial_rope_decode`) produce a DRAM q. The failing call receives q of shape [1,1,6,256] (B=1, 6 heads/device). This crash happens on the first decode call (warmup), so no perf/accuracy/text numbers exist for B=1; the failure is not related to the fused sigmoid (L2b fails identically).
- Raw `[TP 4-dev]` lines: none (perf1, perf2, traced_4k crashed before printing). accuracy_512: none (crashed).
- batched_128_b8 (B=8, passed with lean path active by default, MAX_B=32; the text differs from row 1.1, which indicates the lean numerics were used):

```
L2_attn_lean:                 [TP 4-dev B=8] ttft=0.69s per-user-decode=23.00 tok/s aggregate=184.0 tok/s
L2b_attn_lean_nofusedsig:     [TP 4-dev B=8] ttft=0.72s per-user-decode=22.79 tok/s aggregate=182.4 tok/s
```

  vs row 1.1: `[TP 4-dev B=8] ttft=0.69s per-user-decode=22.73 tok/s aggregate=181.9 tok/s` -> 23.00 t/s/u/user (+0.27, +1.2%; 43.99 -> 43.48 ms/step, -0.51 ms), aggregate 181.9 -> 184.0 tok/s (+2.1). Unfused sigmoid: 22.79 t/s/u/user, 182.4 tok/s (+0.06 vs row 1.1, below the +0.27 of the fused sigmoid; single run each, so the fused-vs-unfused gap is only indicative). TTFT 0.69s / 0.72s (prefill unchanged).
- Generated text, batched_128_b8 row 0 (first 300 chars; the test asserts all 8 rows identical; identical for fused and unfused sigmoid):

```
' bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the burgers, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin'
```

  (row 1.1 b8 text: `... the creamy richness of mayonnaise, on fries, the spicy kick of mustard, on sandwiches, or perhaps something more exotic like`.)
- Generated text, traced_128 perf1 / traced_4k: none (crashed).
- Unit tests (`tests/test_attention_tp.py`, same env as measure.sh, logs `L2_attn_lean/unit_U{1,2,3}.log`, last-80 tails `unit_U{1,3}_last80.txt`):
  - U1 `QWEN36_ATTN_DECODE_LEAN=1 -k "test_attention_tp_paged and not peruser"`: FAILED (1 failed, 6 deselected, 24.98s), same TT_FATAL (Q must be DRAM, got L1) at `tp.py:786`. No PCC line printed (the failure precedes the PCC logging).
  - U2 `QWEN36_ATTN_DECODE_LEAN=0` (legacy, same -k): PASSED (1 passed, 4.41s). PCC lines: `PREFILL paged-vs-concat PCC = 0.99987081120057` ; `DECODE  paged-vs-concat PCC = 1.0`.
  - U3 `QWEN36_ATTN_DECODE_LEAN=1 -k test_attention_tp_paged_peruser`: FAILED both params (`[B8]`, `[B32]`; 2 failed, 6.49s), same TT_FATAL at `tp.py:786`; q shape [1,1,6,256] in the failing SDPA call (B=1 decode inside the per-user test). No PCC lines printed.
  - So lean-vs-legacy PCC comparison is not available: paged lean = FAIL (crash), legacy prefill 0.99987081120057 / decode 1.0.
- Judgement: the lean decode path is not usable at B=1 in its current form (hard crash in every B=1 test and in the unit tests), so there is no TTFT/decode/accuracy/coherence data to compare with row 1.1 for traced_128, traced_4k or accuracy_512. The only produced text is the b8 row 0 continuation (`... the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin`), which is fluent, on-prompt (continues the repeated condiment sentence) and no NaN/garbage, slightly different from row 1.1's b8 text (drops `on fries` / `on sandwiches`), consistent with changed numerics but not a regression in coherence. L2c (`QWEN36_ATTN_DECODE_LEAN_MAX_B=1`) was not run: it would keep only B=1 on the lean path, which is the broken case; the task condition (b8 alone broken) did not hold. No code was edited during this measurement.

#### 1.3 Fix (re-measured as `L2_attn_lean_fix`)

**Root cause (confirmed by the fix):** at B=1 the lean path's `apply_partial_rope_decode_single(q, ..., memory_config=_L1)` did its final `ttnn.concat` into L1, so q reached `ttnn.transformer.paged_scaled_dot_product_attention_decode` as an L1-interleaved tensor, which the op rejects (`Q tensor buffer type must be DRAM when not sharded`). qwen38's implementation (`models/demos/qwen38_27b_qb2/tt/decoder.py` ~697-703) keeps the RoPE parts in L1 but lets the concat default to DRAM. B>1 was unaffected (legacy `apply_partial_rope_decode` returns a DRAM q).

**Change (2 lines of substance, uncommitted):**
- `tt/attention/rope_tp.py` `apply_partial_rope_decode_single(..., memory_config=_L1, out_memory_config=None)`: the final `ttnn.concat` uses `out_memory_config or memory_config` (the `rope_dim == hd` early return converts with `ttnn.to_memory_config` and frees the source when the configs differ; docstring updated).
- `tt/attention/tp.py` `_forward_decode_lean` (B == 1 branch): q only, `apply_partial_rope_decode_single(q, cos_tt, sin_tt, self.rope_dim, memory_config=_L1, out_memory_config=ttnn.DRAM_MEMORY_CONFIG)` with the comment that paged SDPA decode requires a DRAM-interleaved or sharded Q (as in qwen38's `_rope_decode`). k is unchanged (goes to a height shard for the fused cache update).

- Flags: none set (`flags.txt` empty; `QWEN36_ATTN_DECODE_LEAN` default 1, `..._FUSED_SIGMOID` default 1, `..._MAX_B` default 32, `QWEN36_DECODE_ALLREDUCE` default 1). The unfused-sigmoid fallback run (`L2b`) was not needed (no NaN/garbage, accuracy >= 97/99).
- Date/time (UTC, 2026-10-02): unit tests 23:23:07 - 23:23:54; `measure.sh L2_attn_lean_fix` 23:23:59 - 23:25:59 (all five pytest invocations EXIT=0, models imported from the worktree). Log dir `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L2_attn_lean_fix/`; full extract `runs/T1/L2_fix_extract.txt`.
- Raw `[TP 4-dev]` lines:

```
perf1: [TP 4-dev] ttft=0.12s decode=29.38 tok/s
perf2: [TP 4-dev] ttft=0.12s decode=29.50 tok/s
t4k:   [TP 4-dev] ttft=0.46s decode=29.24 tok/s
b8:    [TP 4-dev B=8] ttft=0.69s per-user-decode=22.93 tok/s aggregate=183.4 tok/s
acc:   Top-1 token accuracy: 98.44%  Top-5 token accuracy: 100.00%
```

- Delta vs row 1.1 (perf 28.77 / 28.82, ttft 0.13s, acc 98.44 / 100.00, t4k 28.67, b8 22.73): decode +0.61 / +0.68 t/s/u (+2.1% / +2.4%), i.e. 34.76 -> 34.04 / 34.70 -> 33.90 ms/token (-0.72 / -0.80 ms); traced_4k 28.67 -> 29.24 (+0.57, +2.0%; 34.88 -> 34.20 ms, -0.68 ms); b8 22.73 -> 22.93 t/s/u/user (+0.20, +0.9%; 43.99 -> 43.61 ms/step, -0.38 ms), aggregate 181.9 -> 183.4 tok/s (+1.5); TTFT 0.13s -> 0.12s (prefill untouched, within noise); accuracy_512 identical 98.44 / 100.00. Cumulative vs baseline row 0 (27.74): +1.64 / +1.76 t/s/u (+5.9% / +6.3%), -2.01 / -2.15 ms/token. Single run per metric (perf1/perf2 are two runs), so differences of a few hundredths t/s/u are within noise.
- Generated text, traced_128 perf1 (full as printed by the test; perf2 identical = True):

```
' bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the burgers, the creamy richness of the mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or ho'
```

- Generated text, traced_4k (first 300 chars):

```
'\n\n<think>\n\n</think>\n\nBased on the text provided, the history of artificial intelligence can be summarized through the following key milestones:\n\n1.  **Antiquity and Myth**: The origins of AI trace back to ancient myths and stories featuring artificial beings endowed with intelligence by master craft'
```

- Generated text, batched_128_b8 row 0 (first 300 chars; all 8 rows identical by the test's assertion):

```
' bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the burgers, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin'
```

- Unit tests (same env as measure.sh; logs `L2_attn_lean_fix/unit_U{1,3}.log`):
  - U1 `QWEN36_ATTN_DECODE_LEAN=1 -k "test_attention_tp_paged and not peruser"`: PASSED (1 passed, 6 deselected, 3.85s). `PREFILL paged-vs-concat PCC = 0.99987081120057` (identical to legacy U2 0.99987081120057; prefill does not use the lean decode path), `DECODE  paged-vs-concat PCC = 0.9999476649543133` (legacy U2: 1.0).
  - U3 `QWEN36_ATTN_DECODE_LEAN=1 -k test_attention_tp_paged_peruser`: PASSED both params (2 passed, 5 deselected, 33.20s): `per-user paged decode (B=8) PCC min=1.00000 max=1.00000`, `per-user paged decode (B=32) PCC min=1.00000 max=1.00000` (legacy peruser was not run in this session; the test compares against a per-user reference).
- Judgement: coherent and on-prompt, no NaN/garbage. traced_128 continues the repeated condiment sentence (` bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, ... the spicy kick of mustard, or perhaps something more exotic like sriracha or ho[isin]`). Versus the prompt ('What is your favorite condiment? ... the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce?') it inserts `the burgers,` after ketchup and `the` before mayonnaise (a small drift, still grammatical and on-topic), and it differs from row 1.1's text (`the plain taste of mayonnaise ... sauce? Maybe`) and from the baseline (`creamy richness of mayonnaise ... hoisin sauce?`); the output is cut after 50 tokens at `ho`(isin), which is why it does not reach `sauce`. The `the burgers,` insertion also appeared in the first-attempt b8 run (lean numerics), and the b8 text now matches that earlier lean b8 text up to `hoisin`. perf1 == perf2 (deterministic). traced_4k opens an empty think block and starts the same correct structured AI-history summary (Antiquity and Myth, ...) as the baseline (identical first 300 chars to row 0). Teacher-forced accuracy_512 is unchanged at 98.44 / 100.00, which indicates the lean numerics do not degrade next-token quality; the b8 text is slightly different again from row 1.1 but fluent.

### 1.2 — GDN fused decode

**Status: first attempt FAILED correctness (garbage output; failure record + root cause kept below, not a perf result); fixed and re-measured, see "1.2 Fix" at the end of this section (table row 1.2 holds the fixed result).** Measured after 1.3 (cumulative order: 1.1 -> 1.3 -> 1.2; the 1.3 lean attention fix is in these runs). The patch is applied (uncommitted) in the worktree with the default `QWEN36_GDN_FUSED_DECODE=1`. A pre-patch copy of `tt/gdn/tp.py` is at `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/gdn_tp_before_1_2.py`.

- Files changed (uncommitted, worktree `qwen38-optimizations`): `models/demos/blackhole/qwen36/tt/gdn/tp.py` (patch `scratchpad/patch_1_2/gdn_fused_decode.patch`: `gdn_fused_decode_enabled`, fp32 `dt_bias`/`neg_exp_A` and rank-1 `norm_w_1d` weights, `_conv_hist_rm` history + `_ensure_conv_hist`/`_write_conv_hist`/`_sync_conv_hist_from_states`, `snapshot_fused_decode_state`/`restore_fused_decode_state`, `_forward_decode_fused`, one info line); `models/demos/blackhole/qwen36/demo/text_demo.py` (two snapshot/restore pairs: single-user and batched) and `models/demos/blackhole/qwen36/demo/vision_demo.py` (one pair): each snapshot tuple got a third element `dn.snapshot_fused_decode_state()` and the restore calls `dn.restore_fused_decode_state(hist)` after restoring rec/conv state.
- Design: A's ~69-op GDN decode chain (shift-register conv, recurrent kernel, host-side gate math) is replaced by B's fused kernels at `max_batch_size == 1` and active width 1: `ttnn.experimental.kda.qkv_causal_conv1d_silu` (conv + SiLU + q/k/v split) reading a new persistent ROW_MAJOR bf16 history `_conv_hist_rm` `[1,3,qkv_dim_tp]` (kept in sync by every prefill/state writer: `capture_state`, `assemble_batched_state`, `write_slot`, `forward_prefill_batched`, `reset_state` zeroes it in place; it is not part of the model's per-binding state swaps because one user means the latest prefill is the decode state), `ttnn.transformer.chunk_gated_delta_rule` with chunk_size 32 over one live token padded to 32 rows (zero beta/g make the 31 padded steps identity state updates), `ttnn.experimental.kda.sigmoid_gated_rms_norm` then `mul(z)`. `a`/`g` stay fp32 (fp32 `dt_bias` / `-exp(A_log)`), the out-proj input is bf16. `conv_states` is not maintained during fused decode. Falls back to the original path for B>1, for `QWEN36_GDN_FUSED_DECODE=0`, for QWEN35_GDN_STATE_BF16 / DECODE_BF16, or when the kda op is missing.
- Flags: `QWEN36_GDN_FUSED_DECODE` default "1" (`flags.txt` empty in `L3_gdn_fused`, `QWEN36_DECODE_ALLREDUCE` default 1, `QWEN36_ATTN_DECODE_LEAN` default 1). Control run `L3b_gdn_off`: `QWEN36_GDN_FUSED_DECODE=0`.
- Info line (one per process) confirmed: perf1, perf2, acc, t4k: `[GDN] decode path: FUSED for max_batch_size == 1 (qkv_causal_conv1d_silu + chunk_gated_delta_rule + sigmoid_gated_rms_norm; QWEN36_GDN_FUSED_DECODE=0 reverts). conv_states is not maintained during decode; out-proj input is bf16.`; b8: `[GDN] decode path: ORIGINAL (shift-register conv + recurrent kernel): max_batch_size=8 (fused decode is single-user)`; L3b (all five): `... ORIGINAL ...: QWEN36_GDN_FUSED_DECODE=0`.
- Date/time (UTC, 2026-10-02): unit tests (fused=1 G1-G6, fused=0 G1-G3) 23:29:47 - 23:33:51; extra fused=0 runs of G4-G6 23:34:16 - 23:35:20; `measure.sh L3_gdn_fused` 23:35:27 - 23:37:23 (exits: perf1 0, perf2 0, acc 1, t4k 0, b8 0; models imported from the worktree); `QWEN36_GDN_FUSED_DECODE=0 measure.sh L3b_gdn_off` 23:37:38 - 23:39:33 (all five EXIT=0); diagnostics afterwards. Log dirs `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L3_gdn_fused/` (unit_*.log, unit_exits.txt, failing-test tails `unit_*_last80.txt`; the measure logs share the directory) and `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L3b_gdn_off/`; extracts `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L3_extract.txt`, `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L3b_extract.txt`; diagnostic scripts and logs `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/diag/`.
- Raw `[TP 4-dev]` lines:

```
L3_gdn_fused (fused ON):
perf1: [TP 4-dev] ttft=0.12s decode=32.37 tok/s
perf2: [TP 4-dev] ttft=0.13s decode=32.41 tok/s
t4k:   [TP 4-dev] ttft=0.47s decode=32.18 tok/s
b8:    [TP 4-dev B=8] ttft=0.69s per-user-decode=23.02 tok/s aggregate=184.2 tok/s
acc:   Top-1 token accuracy: 0.20%  Top-5 token accuracy: 0.20%   (AssertionError: top-1 token accuracy 0.20% below target 96.5%, EXIT=1)

L3b_gdn_off (QWEN36_GDN_FUSED_DECODE=0, control):
perf1: [TP 4-dev] ttft=0.13s decode=29.45 tok/s
perf2: [TP 4-dev] ttft=0.13s decode=29.44 tok/s
t4k:   [TP 4-dev] ttft=0.47s decode=29.26 tok/s
b8:    [TP 4-dev B=8] ttft=0.67s per-user-decode=23.00 tok/s aggregate=184.0 tok/s
acc:   Top-1 token accuracy: 98.44%  Top-5 token accuracy: 100.00%
```

- Delta vs row 1.3 (perf 29.38 / 29.50, ttft 0.12s, acc 98.44 / 100.00, t4k 29.24, b8 22.93): fused ON decode +2.99 / +2.91 t/s/u (+10.2% / +9.9%), 34.04 -> 30.89 / 33.90 -> 30.85 ms/token (-3.14 / -3.04 ms); traced_4k 29.24 -> 32.18 (+2.94, +10.1%; 34.20 -> 31.08 ms, -3.12 ms); b8 22.93 -> 23.02 (+0.09; the fused path is inactive at B=8, so this is noise; L3b b8 23.00); TTFT 0.12s / 0.13s (prefill untouched). Control with the flag off reproduces row 1.3 (29.45 / 29.44 vs 29.38 / 29.50; t4k 29.26 vs 29.24; b8 23.00 vs 22.93; accuracy 98.44 / 100.00 identical; texts identical), so the rest of the stack (including the demo snapshot/restore edit) is healthy and the failure is confined to the fused decode path. The +3 t/s/u is what the fused path achieves while computing wrong values (about 3.1 ms/token saved over 48 GDN layers); the post-fix cost will be a little higher if the fix adds ops.
- Generated text, traced_128 perf1 (full as printed; perf2 identical = True):

```
' bringingRh Huffcideulpabar&e一击ertonantom淫\u200b\u200b匪umpsagio疏iganoardbj\u200b\u200b�asonicurvoid�amana霖ruz vọng iffield�rocekaorrorettapresse芝堂堂耳目 HarbourinoiTzell널 Laf�ơوترده'
```

- Generated text, traced_4k (first 300 chars): `'\n\nustr末oareLANGPropTypes满月玄幻浏eganilandcacäأ酢nex猛asher蛆邦�iranagerajo�eton政CCI洋ettelakash勃erial一门朝夕返�NOخالFromClassirl勉知的بران海棠หมด|%ibro�aultILA足iha�FML�rr永康alletesyanteippleodash�ęb8[:穂洋 Hollow光辉bauer扇izontalaborfoxasesoviacow希랭毓itches�bedoouchsoleolidays为难ชimitivesama Robbinsimbajar Prov襟azed翅饮'` (the control L3b text is the normal `<think> ... Based on the text provided, the history of artificial intelligence ... 1.  **Antiquity and Myth**` summary, identical to row 1.3).
- Generated text, batched_128_b8 row 0 (first 300 chars; all 8 rows identical by the test's assertion; original path since max_batch_size=8): `' bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the burgers, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin'` (identical to row 1.3's b8 text).
- Unit tests (`tests/test_gdn_tp.py`, `tests/test_model_tp.py`, same env as measure.sh, logs `L3_gdn_fused/unit_<id>_fused{1,0}.log`; PCC lines verbatim from the logs):

| id | test | fused=1 | fused=0 (original) |
|---|---|---|---|
| G1 | `test_gdn_tp -k B1` | FAIL: `GDN TP PCC (pos0) min=-0.00000 max=-0.00000` (threshold 0.92) | PASS: `min=0.99997 max=0.99997` |
| G2 | `test_gdn_tp_prefill` | FAIL: `PREFILL vs DECODE PCC (T=128) = 0.0` (`One tensor is all zero. PCC undefined; falling back to allclose.`) | PASS: `0.9999527610992562` |
| G3 | `test_gdn_tp_fused_chunk_prefill` | FAIL: fused-chunk vs seq-adapter `0.9992457957560229` (pass, prefill only), fused-chunk prefill vs step-decode `0.0` (FAIL, same all-zero warning) | PASS: `0.9992457957560229` / `0.9999723827849271` |
| G4a | `test_gdn_tp_peruser_state -k B8` | FAIL: `PCC min=-0.00000 max=0.99228`, bad users `[(1,-0.0),(2,0.0),(3,-0.0),(6,0.0),(7,0.0)]` | PASS: `min=1.00000 max=1.00000` |
| G4b | `test_gdn_tp_write_slot_and_remap -k B8` | FAIL: `[(1,-0.0),(2,0.0),(3,-0.0),(4,0.9512048363685608),(6,0.0),(7,0.0)] (min=-0.00000)` | PASS: `worst PCC = 1.00000` |
| G4c | `test_gdn_tp_batched_prefill -k "B2 or B4"` | FAIL both: B2 `min=-0.00081 max=-0.00000`; B4 `min=nan max=nan` (bad `(1,-0.6576),(2,0.0),(3,-0.0)`) | PASS: B2 `0.99999`, B4 `0.99997` (min) |
| G5 | `test_gdn_tp_prefill_trace_replay -k random` | PASS: chunks 0-2 `trace replay == eager: True; pcc 1.0` (prefill only, does not run the fused decode) | PASS: identical |
| G6a | `test_model_tp_contract` | FAIL: step logits PCC `[0.9997960549838517 (prefill), 0.3812563615448047, 0.18210192825544344, 0.9998744213329819]` | PASS: `0.99980, 0.99986, 0.99991, 0.99991`, masked-bucket prefill `0.99980` |
| G6b | `test_model_tp_decode_batched -k B8` | FAIL: `user 0 step 1 (len=128) logits PCC 0.0635557278921496 < 0.97` | PASS: `worst logits PCC = 0.99969 @ user3 step0` |

  Notes: G4 and G6b build their B=1 reference/oracle with `max_batch_size=1`, so under fused=1 the *reference* is the broken fused path while the B>1 side runs the original path; G1-G3 and G6a compare the fused decode against a torch reference / the prefill / the bespoke oracle. G4/G5/G6 fused=0 runs are extra (the plan required fused=0 only for G1-G3). The G6a pattern (steps 1 and 2 wrong, step 3 fine) is the same layout-dependent corruption described below; it is not a conv-history handoff problem.
- Diagnostics (no code changes; scripts `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/diag/diag_gdn_fused*.py`, run through `pytest -p conftest`): (1) the model-level path and `forward_decode` return finite but enormous values at B=1 with fused=1 (absmax 1.2e35, 1.7e17, 203 on three consecutive calls, PCC vs original -0.000, -0.061, -0.506), deterministic and identical cold/warm. (2) With per-stage readbacks, every input of `ttnn.transformer.chunk_gated_delta_rule` is finite and sane (q/k/v/g/beta absmax ~1-8), but its output `o` has absmax 4.3e36 and `new_rec` 1.06e38, i.e. the recurrent state itself is corrupted by the first call. (3) Bisecting the deallocations in a verbatim copy of `_forward_decode_fused`: all deallocs -> garbage; none of the deallocs before the chunk op -> PCC vs original 0.999993 / 0.999982 / 0.999988 over three consecutive steps (so the algorithm, the conv history update and the state carry are correct); none of the deallocs after the chunk op -> still garbage; keeping only `g` and `beta` alive -> fixed (0.999993 / 0.999678 / 0.999920); keeping only the DRAM temporaries (`qkv_p`, `row_qkv`, `keep`, `cur`, `new_hist`) alive -> still garbage. (4) `ttnn.pad(t, [(0,0),(0,31),(0,0)], value=0.0, memory_config=DRAM)` on the decode tensors returns a VIEW of the same buffer: for `qkv`, `z` and `sigmoid(b)` the output has the same buffer address and is still L1 (`memory_config` ignored) because the tile-padded shape (32 rows) already covers the target logical shape. Consequences: (a) `ttnn.deallocate(g)`, `ttnn.deallocate(beta)` (and `qkv`, `z`) right after the pad free the buffer that `g_p`, `beta_p`, `qkv_p`, `z_p` alias, so later allocations overwrite them (use-after-free), which matches the layout-dependent garbage; (b) even without the free, rows 1..31 of the padded tensors are not zeroed by the pad (the identity-step assumption `beta = g = 0` on the padded rows is not guaranteed; they only happened to be zero in a fresh-memory run). [Correction from the fix verification, see "1.2 Fix": claim (b) is wrong; the pad DOES zero-fill the implicit tile padding in place (a FillPad runs), rows 1..31 of beta_p and g_p read back exactly 0 on device in every call; only claim (a), the use-after-free, was the defect.] Suggested fix (NOT applied, outside this task): build the 32-row beta/g/z/qkv inputs with a real copy that also writes zeros (for example concat with a zero block, or a ROW_MAJOR pad followed by tilize) and do not deallocate the source of a view; then re-run G1-G6 and the measurement.
- Judgement: the generated text is not coherent and not on the prompt: after the first word (` bringing`) the traced_128 continuation is random multilingual tokens, traced_4k starts with a newline and then random tokens instead of the `<think>` block and the structured AI-history summary, and teacher-forced accuracy_512 is 0.20 / 0.20 (vs 98.44 / 100.00 for row 1.3). Row 1.3's traced_128 text continued the repeated condiment sentence (`... ketchup, the burgers, the creamy richness of the mayonnaise, ... sriracha or ho`); the fused ON text shares only the first generated word with it. perf1 == perf2 (deterministic garbage). The b8 text is fine only because the original path runs at B=8. Decision for this row: not usable as is; the +3.0 t/s/u (+10%) is an upper bound for what the fused path can give, to be re-measured after the pad/dealloc defect is fixed. Next step for the caller: fix the padded-input construction and the premature deallocs in `_forward_decode_fused` (or default `QWEN36_GDN_FUSED_DECODE` to 0 meanwhile), then re-run G1-G6 and `measure.sh`.

#### 1.2 Fix (re-measured as `L3c_gdn_fused_fix`)

**Root cause (confirmed by the fix):** `ttnn.pad(t, [(0,0),(0,31),(0,0)], value=0.0, memory_config=...)` on a `[1,1,C]` TILE tensor (physical tile height 32) returns a VIEW of the SAME device buffer (the tile already holds the target rows; the pad only runs a fill of the implicit tile padding in place and ignores `memory_config`; the outputs are still L1: `same_buffer=True`, `mem=L1`, confirmed again in this session for g_p and beta_p). `_forward_decode_fused` then called `ttnn.deallocate(z / g / beta / qkv)` on the pad SOURCES, which frees the buffer the padded tensors still use (`ttnn.deallocate` is force-free), so later allocations overwrote `g_p` / `beta_p` / `z_p` before the chunk op consumed them and `chunk_gated_delta_rule` produced ~1e36 garbage. qwen38's `_delta` (`models/demos/qwen38_27b_qb2/tt/decoder.py`) does the same pads but never deallocates in the forward path.

**Change (only `_forward_decode_fused`, uncommitted; pre-fix copy of `tt/gdn/tp.py` at `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/tp_before_1_2_fix.py`, diff at `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/fix_1_2.diff`):** the four `ttnn.deallocate` calls on the pad sources (`z`, `g`, `beta`, `qkv`) are removed; each padded tensor keeps exactly ONE deallocate, after its last consumer (`z_p` after the gate `ttnn.mul`, `g_p` / `beta_p` after the chunk op in the existing `for t in (q, k, v, g_p, beta_p)` loop, `qkv_p` right after the `to_layout` that produced the new RM/DRAM `row_qkv`); a comment explains that padding to the tile height returns a view; the docstring sentence that claimed every kernel input is in DRAM was corrected (the pad views stay in their source memory space, L1; the chunk op reads L1 `g`/`beta` without a CB clash). Other view-risk ops were checked: `ttnn.typecast` always allocates (no same-dtype early return), `to_layout` TILE -> ROW_MAJOR allocates, the `slice`s have a smaller shape. No op was added or removed, only four frees.

```diff
@@ -1392,9 +1392,9 @@
         Differences from the original path (all follow the validated fused decoder, models/demos/qwen38_27b_qb2):
         a and g stay fp32 (dt_bias / -exp(A_log) are fp32 copies), and the out-proj input is bf16 (the original
         feeds fp32; the out-proj partial is therefore bf16 too). conv_states is not updated here -- only
-        _conv_hist_rm and rec_state are. Every tensor the fused kernels read is moved to DRAM before they run (the
-        chunk op's CBs clash with L1 inputs) and no L1 activation outlives the gate math. No allocation of
-        persistent state: trace-safe.
+        _conv_hist_rm and rec_state are. conv input / q / k / v are DRAM; the padded z_p / g_p / beta_p are views of
+        their (L1) sources and stay in that memory space until their last consumer. No allocation of persistent
+        state: trace-safe.
         """
         from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE

@@ -1402,6 +1402,12 @@
         _L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
         C, kd, vd = self.qkv_dim_tp, self.key_dim_tp, self.value_dim_tp
         # The kernels run on one 32-row tile; row 0 is the live token. Zero rows: identity recurrence steps.
+        # NOTE: ttnn.pad of a [1,1,C] TILE tensor up to the 32-row tile height returns a VIEW of the SAME device
+        # buffer (the tile already holds the 32 rows; pad only zero-fills the implicit tile padding in place and
+        # ignores memory_config). So the padded tensors (z_p, g_p, beta_p, qkv_p) must NEVER be freed by
+        # deallocating their SOURCE (that frees the shared buffer under the live view -> later allocations
+        # overwrite it -> garbage): the source is left alone and exactly ONE handle (the padded one) is
+        # deallocated, after the padded tensor's last consumer.
         pad_rows = [(0, 0), (0, _FUSED_CHUNK_SIZE - 1), (0, 0)]

         qkv, z, a, b = self._project_qkvzab(x, 1, out_mc=_L1)
@@ -1409,8 +1415,7 @@
         # ---- z (the out gate) and the gates, ahead of the kernels: DRAM, 32 rows, zero padded ----
         if z.dtype != ttnn.bfloat16:  # sigmoid_gated_rms_norm's gate contract
             z = ttnn.typecast(z, ttnn.bfloat16, memory_config=_L1)
-        z_p = ttnn.pad(z, pad_rows, value=0.0, memory_config=_DRAM)
-        ttnn.deallocate(z)
+        z_p = ttnn.pad(z, pad_rows, value=0.0, memory_config=_DRAM)  # view of z; freed once, after the gate mul
         # beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), a/g in fp32
         beta = ttnn.sigmoid(b, memory_config=_L1)
         ttnn.deallocate(b)
@@ -1425,17 +1430,14 @@
             memory_config=_L1,
         )
         ttnn.deallocate(a_dt)
-        g_p = ttnn.pad(g, pad_rows, value=0.0, memory_config=_DRAM)
-        ttnn.deallocate(g)
-        beta_p = ttnn.pad(beta, pad_rows, value=0.0, memory_config=_DRAM)
-        ttnn.deallocate(beta)
+        g_p = ttnn.pad(g, pad_rows, value=0.0, memory_config=_DRAM)  # view of g; freed once, after the chunk op
+        beta_p = ttnn.pad(beta, pad_rows, value=0.0, memory_config=_DRAM)  # view of beta; freed once after chunk

         # ---- causal conv1d + SiLU + q/k/v split: one program on the persistent RM history ----
         hist = self._ensure_conv_hist()
-        qkv_p = ttnn.pad(qkv, pad_rows, value=0.0, memory_config=_DRAM)
-        ttnn.deallocate(qkv)
-        row_qkv = ttnn.to_layout(qkv_p, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)
-        ttnn.deallocate(qkv_p)
+        qkv_p = ttnn.pad(qkv, pad_rows, value=0.0, memory_config=_DRAM)  # view of qkv
+        row_qkv = ttnn.to_layout(qkv_p, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)  # new (RM, DRAM) buffer
+        ttnn.deallocate(qkv_p)  # last consumer of the tile view done: frees the shared qkv buffer, once
         q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
             row_qkv,
             hist,
```

- Pad-zero verification (on device, `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/diag_fix/diag_fix_verify.py`, runs via `diag_fix/run_diag.sh`, logs `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/diag_fix/fix_hook1_n3.log`, `fix_hook0_n6.log`, `fix_hook1_n6.log`): it runs the REAL fixed `forward_decode` (fused) for N consecutive decode calls with `ttnn.pad` and `chunk_gated_delta_rule` wrapped so that `g_p` and `beta_p` are read back (`to_torch`, all 4 devices) right after each pad and right before the chunk op, and compares each call's output with the original path (QWEN36_GDN_FUSED_DECODE=0 path on the same layer, same inputs). Result: rows 1..31 of `g_p` (fp32) and `beta_p` (bf16) are exactly 0 (0 nonzero values) in every check (6 calls x 4 checks = 24 of 24 `rows1..31_all_zero=True`, also in the 3-call run), row 0 finite and nonzero (g_p absmax 4.4-5.4, beta_p absmax 0.64-0.85 over the 6 calls); `beta` is `sigmoid(b)` whose padded rows would be 0.5 if the pad did not fill them, so this proves the fill. The pad outputs are views (`src_addr == out_addr`, L1), so the pad zero fill is reliable and NO explicit zero-fill / concat replacement was needed. PCC of the fixed fused decode vs the original path over 6 consecutive calls (HOOK=0, no syncs): `0.999993 0.999982 0.999988 0.999958 0.999941 0.999949`; with the hooks identical values (the readbacks do not mask anything). Before the fix this diag setup gave absmax 1e35 and PCC ~0 / negative.
- Date/time (UTC, 2026-10-02): edit + compile 23:51; diag runs 23:52:20 - 23:53:00; fused=1 unit tests 23:53:15 - 23:54:42 (all 9 EXIT=0, every log shows the `[GDN] decode path: FUSED` line, so the fused path really ran); `measure.sh L3c_gdn_fused_fix` 23:54:50 - 23:56:50 (perf1/perf2/acc/t4k/b8 all EXIT=0, models imported from the worktree). Log dirs `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L3c_units/` (unit_*.log, unit_exits.txt) and `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L3c_gdn_fused_fix/`; extract `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L3c_extract.txt`.
- Flags: none set (`flags.txt` empty; `QWEN36_GDN_FUSED_DECODE` default 1, `QWEN36_DECODE_ALLREDUCE` default 1, `QWEN36_ATTN_DECODE_LEAN` default 1). Info lines: perf1, perf2, acc, t4k: `[GDN] decode path: FUSED ...`; b8: `[GDN] decode path: ORIGINAL ... max_batch_size=8 (fused decode is single-user)`. No Error/Traceback/clash lines in any measurement log.
- Raw `[TP 4-dev]` lines:

```
perf1: [TP 4-dev] ttft=0.13s decode=32.38 tok/s
perf2: [TP 4-dev] ttft=0.13s decode=32.37 tok/s
t4k:   [TP 4-dev] ttft=0.47s decode=32.13 tok/s
b8:    [TP 4-dev B=8] ttft=0.68s per-user-decode=23.01 tok/s aggregate=184.1 tok/s
acc:   Top-1 token accuracy: 98.63%  Top-5 token accuracy: 100.00%
```

- Delta vs row 1.3 (perf 29.38 / 29.50, ttft 0.12s / 0.12s, acc 98.44 / 100.00, t4k 29.24, b8 22.93 / 183.4 aggregate) and vs the invalid first fused run (32.37 / 32.41, t4k 32.18, b8 23.02): decode +3.00 / +2.87 t/s/u (+10.2% / +9.7%), 34.04 -> 30.88 / 33.90 -> 30.89 ms/token (-3.15 / -3.01 ms); traced_4k 29.24 -> 32.13 (+2.89, +9.9%; 34.20 -> 31.12 ms, -3.08 ms); b8 22.93 -> 23.01 t/s/u/user (+0.08, +0.3%; fused path inactive at B=8, so noise; aggregate 183.4 -> 184.1); TTFT 0.12s / 0.12s -> 0.13s / 0.13s (prefill untouched, +0.01 s is within run-to-run noise: L3b flag-off control also showed 0.13s); accuracy_512 98.44 / 100.00 -> 98.63 / 100.00 (+0.19, one more correct token of 512). Versus the invalid run: 32.38 / 32.37 vs 32.37 / 32.41 and 32.13 vs 32.18, i.e. no measurable throughput cost from the fix (it only removes frees). Cumulative vs baseline row 0 (27.74 / 27.74): +4.64 / +4.63 t/s/u (+16.7%), -5.17 / -5.16 ms/token (36.05 -> 30.88 / 36.05 -> 30.89). Single run per metric (perf1/perf2 are two runs), so differences of a few hundredths t/s/u are within noise.
- Unit tests, fused=1 with the fix vs the original path (fused=0, from the previous session, same tests; PCC lines verbatim from `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L3c_units/unit_<id>_fused1.log`):

| id | test | fused=1 fixed | fused=0 (original) | fused=1 before the fix |
|---|---|---|---|---|
| G1 | `test_gdn_tp -k B1` | PASS: `GDN TP PCC (pos0) min=0.99997 max=0.99997` | `0.99997` | FAIL (-0.00000) |
| G2 | `test_gdn_tp_prefill` | PASS: `PREFILL vs DECODE PCC (T=128) = 0.9999041660437071` | `0.99995` (0.9999527610992562) | FAIL (0.0) |
| G3 | `test_gdn_tp_fused_chunk_prefill` | PASS: fused-chunk vs seq-adapter `0.9992457957560229`; fused-chunk prefill vs step-decode `0.9997668085922082` | `0.99925` / `0.99997` | FAIL (0.0 on the second) |
| G4a | `test_gdn_tp_peruser_state -k B8` | PASS: `PCC min=0.99992 max=0.99999` | `1.0` (min=max=1.00000) | FAIL (min -0.0) |
| G4b | `test_gdn_tp_write_slot_and_remap -k B8` | PASS: `worst PCC = 0.99992` | `1.0` | FAIL (min -0.0) |
| G4c | `test_gdn_tp_batched_prefill -k "B2 or B4"` | PASS both: B2 `min=0.99998 max=0.99999`, B4 `min=0.99989 max=1.00000` | `0.99999` / `0.99997` | FAIL both |
| G5 | `test_gdn_tp_prefill_trace_replay -k random` | PASS: chunks 0-2 `trace replay == eager: True; pcc 1.0` | identical | PASS |
| G6a | `test_model_tp_contract` | PASS: step logits PCC `0.9997960549838517 (prefill), 0.999858971725654, 0.9999018940618456, 0.9999150198387337`; masked-bucket prefill `0.9997978495781239` | `0.99980, 0.99986, 0.99991, 0.99991`; masked-bucket `0.99980` | FAIL (steps 1, 2: 0.381, 0.182) |
| G6b | `test_model_tp_decode_batched -k B8` | PASS: `worst logits PCC = 0.99895 @ user6 step1` | `0.99969 @ user3 step0` | FAIL (0.0636) |

  All nine pass. G1 equals the original (0.99997); G2 / G3 / G4 / G6a are 1e-4 to 1e-3 below the original path (the fused path feeds the out-proj in bf16 and keeps `a`/`g` in fp32, as documented in the design), far above every threshold (G6b's lower 0.99895 is still the B=8 original path compared to a B=1 oracle that now runs the fused decode). G4a/G4b/G4c/G6b build their B=1 reference with `max_batch_size=1`, i.e. with the fused path itself.
- Generated text, traced_128 perf1 (full as printed by the test; perf2 identical = True):

```
" bringing its unique flavor and texture to enhance different dishes. Do you don't prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce"
```

- Generated text, traced_4k (first 300 chars; the log prints it as):

```
"\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User Input:**\n   - **Input Text:** A highly repetitive paragraph (repeated ~10 times) about the history of AI. It covers:\n     - Antiquity: myths/stories of artificial beings with intelligence\n     - Classical philosophers: described human thinking as mechanical symbol manipulation\n     - Culmination: invention of the programmable digital computer (based on mathematical reasoning)\n     - Modern progress:"
```

- Generated text, batched_128_b8 row 0 (first 300 chars; all 8 rows identical by the test's assertion; original path since max_batch_size=8): `' bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the burgers, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin'` (identical to row 1.3's b8 text).
- Judgement: no NaN / garbage; the fix restores correctness. traced_128 vs the prompt ('... Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy ...'): the model repeats the prompt's sentence (`bringing its unique flavor and texture to enhance different dishes. Do you ... prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce`) word for word and reaches `hoisin sauce` within the 50 generated tokens, with ONE insertion, `don't` (`Do you don't prefer`), a small grammatical slip; it has no `the burgers,` / extra `the` insertions that row 1.3 showed. Versus row 1.3's traced_128 text (` ... the classic taste of ketchup, the burgers, the creamy richness of the mayonnaise, ... sriracha or ho`): different (as expected with changed numerics: bf16 out-proj input, fp32 a/g), equally fluent and on-prompt, one small slip each; perf1 == perf2 (deterministic). traced_4k vs row 1.3 / baseline: row 1.3 opened with an empty think block (`<think>\n\n</think>\n\nBased on the text provided, the history of artificial intelligence ...`), the fixed fused run opens a thinking process (`<think>\nHere's a thinking process: 1. **Analyze User Input:** ... A highly repetitive paragraph (repeated ~10 times) about the history of AI ... Antiquity: myths/stories of artificial beings ... Classical philosophers ... Culmination: invention of the programmable digital computer ... Modern progress:`), which is coherent, on-topic and a correct description of the 2642-token prompt (the repeated AI-history paragraph); the two outputs diverge right after `<think>` + newline (empty think block vs `Here's a thinking process`), most likely a near-tie flip from the changed numerics (not verified with logit margins) rather than corruption (flag-off control L3b reproduced the row 1.3 text exactly, and teacher-forced accuracy is slightly higher than row 1.3: 98.63 / 100.00 vs 98.44 / 100.00, with every unit PCC >= 0.9989). b8 text unchanged from row 1.3 (original path). The decode-time gain of the fused path is real (+3.0 t/s/u, +10%) at a numeric cost that the accuracy test does not detect.

### 1.5 — DRAM-sharded multi-reader matmul for the decode attention QKV (REVERTED)

**Status: measured, all runs pass, but no throughput gain -> reverted per the decision rule (decode mean must be >= row 1.2 mean - 0.1 = 32.275; measured mean of the two official runs 31.585).** Cumulative order at this point: 1.1 -> 1.3 -> 1.2 -> 1.5; the baseline for the comparison is row 1.2 (the fixed GDN fused decode run, `L3c_gdn_fused_fix`).

- Change (patch `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/patch_1_5/qkv_dram_sharded.patch`, only `tt/attention/tp.py`, dry-run clean, `py_compile` OK, applied and later reverted with `patch -R -p1`): an extra DRAM `WIDTH_SHARDED` copy of the fused per-device QKV(+gate) weight (cache suffix `.dsh2`, bf8, 16 layers, 77,988,608 bytes each = 1.248 GB total) and, in the decode in-projection (`x.shape[-2] <= 32`, includes B=8), `to_memory_config(x, [32,512] on 10 cores)` -> `ttnn.linear` with `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=16, per_core_M=1, per_core_N=12, num_workers_per_dram_bank=2)`, HiFi2, bf16 out into L1 width-sharded -> `sharded_to_interleaved` -> the usual `ttnn.slice`s (gate forced to DRAM). Flag `QWEN36_QKV_DRAM_SHARDED` (default 1; 0 = old 1D-mcast path, no extra weight).
- Flags: none set (`flags.txt` empty; `QWEN36_QKV_DRAM_SHARDED` default 1, `QWEN36_GDN_FUSED_DECODE` default 1, `QWEN36_ATTN_DECODE_LEAN` default 1, `QWEN36_DECODE_ALLREDUCE` default 1). The decode path was not asserted by a log line (the patch has none); evidence the new matmul was active: the unit decode PCC changed (0.99994766 -> 0.99994357), the generated text of every run changed vs row 1.2, and the `.dsh2` files were created (16 files, 00:01:07 - 00:01:16 UTC, during the perf1 model load; no `can't use the DRAM-sharded layout` warning).
- Date/time (UTC, 2026-10-03): patch applied 00:00:11; unit tests 00:00:16 - 00:00:49 (`runs/T1/unit.sh L4_units`); `measure.sh L4_qkv_dsh` 00:00:54 - 00:02:56 (perf1/perf2/acc/t4k/b8 all EXIT=0, models imported from the worktree, no Error/Traceback/clash lines); supplementary `measure_perf_only.sh L4b_supp_perf3` ~00:03; revert 00:04:01. Log dirs `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L4_units/`, `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L4_qkv_dsh/`, `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L4b_supp_perf3/`; extract `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L4_extract.txt`; pre-1.5 copy of `tt/attention/tp.py` at `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/attn_tp_before_1_5.py`.
- Raw `[TP 4-dev]` lines (official runs):

```
perf1: [TP 4-dev] ttft=0.12s decode=30.89 tok/s     (cache-miss run: model load 11.2s, writes the .dsh2 files)
perf2: [TP 4-dev] ttft=0.14s decode=32.28 tok/s     (warm, model load 7.1s)
t4k:   [TP 4-dev] ttft=0.46s decode=32.13 tok/s
b8:    [TP 4-dev B=8] ttft=0.69s per-user-decode=23.06 tok/s aggregate=184.5 tok/s
acc:   Top-1 token accuracy: 98.44%  Top-5 token accuracy: 100.00%
```

  Supplementary (not part of the decision rule; extra warm rerun of `traced_128`, `L4b_supp_perf3`): `perf3: [TP 4-dev] ttft=0.12s decode=32.16 tok/s`, same text as perf1/perf2.

- Delta vs row 1.2 (perf 32.38 / 32.37, ttft 0.13s / 0.13s, acc 98.63 / 100.00, t4k 32.13 ttft 0.47s, b8 23.01 / 184.1 tok/s ttft 0.68s): decode perf1 -1.49 t/s/u (30.88 -> 32.37 ms/token, +1.49 ms), perf2 -0.09 (30.89 -> 30.98 ms, +0.09 ms), mean 31.585 vs 32.375 (-0.79); supplementary warm perf3 32.16 (-0.21 vs the row 1.2 mean of the two runs, +0.20 ms/token). traced_4k 32.13 -> 32.13 (0.00); b8 23.01 -> 23.06 t/s/u/user (+0.05, +0.2%, noise; aggregate 184.1 -> 184.5); TTFT 0.13 / 0.13 -> 0.12 / 0.14 (prefill untouched, noise); accuracy_512 98.63 -> 98.44 (one fewer correct token of 512, 98.44 equals row 0 / 1.1 / 1.3), top-5 100.00. Expected ~-0.25 ms/token (+0.7%) from the 70 -> ~55 us microbenchmark was not observed: the three warm decode measurements (32.28, 32.16, 32.13 t4k) are 0 to 0.2 ms/token SLOWER than row 1.2, and perf1's 30.89 (+1.5 ms) was an outlier of the cache-miss run (cause not investigated; the 7 s warm loads of all later runs show nothing unusual). Likely reason (a hypothesis, not profiled): the extra ops around the matmul (`to_memory_config` to the 10-core input shard, `sharded_to_interleaved` of the output, 16 full-attention layers per token) cost about what the faster matmul saves.
- Generated text, traced_128 perf1 (full as printed; perf2 and perf3 identical):

```
' bringing its unique flavor and texture to enhance different dishes.\n\n<think>\nHere\'s a thinking process:\n\n1.  **Analyze User Input:**\n   - The user asks: "What is your favorite condiment?"\n   - They provide'
```

- Generated text, traced_4k (first 300 chars, decoded):

```
\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User Input:**\n   - **Input Text:** A highly repetitive paragraph about the history of AI. It repeats the same core message multiple times:\n     - Began in antiquity with myths/stories of artificial beings.\n     - Classical philosophers described hu
```

- Generated text, batched_128_b8 row 0 (first 300 chars): `' bringing its unique flavor and texture to enhance different dishes.\n\n<think>\nHere\'s a thinking process:\n\n1.  **Analyze User Input:**\n   - The user asks: "What is your favorite condiment?"\n   - They provide'` (the b8 text also changed vs row 1.2, which still repeated `Do you prefer the classic taste of ketchup, the burgers, ...`).
- Unit tests (same env as measure.sh, `QWEN36_ATTN_DECODE_LEAN=1`; logs `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L4_units/U1.log`, `U3.log`):
  - U1 `-k "test_attention_tp_paged and not peruser"`: PASSED (1 passed, 6 deselected, 3.96s). `PREFILL paged-vs-concat PCC = 0.99987081120057` (= row 1.3's value), `DECODE  paged-vs-concat PCC = 0.9999435714498186` (row 1.3 lean: 0.9999476649543133, legacy 1.0).
  - U3 `-k test_attention_tp_paged_peruser`: PASSED both (2 passed, 5 deselected, 7.18s): `per-user paged decode (B=8) PCC min=1.00000 max=1.00000`, `(B=32) PCC min=1.00000 max=1.00000`.
- Judgement: coherent, on-prompt, no NaN/garbage in any of the five runs. traced_128: the model repeats the prompt's sentence up to `different dishes.` (as in rows 0 / 1.x), then at what is a near-tie in the rows before (`Do you don't prefer ...` in row 1.2, `Do you prefer ...` in row 0) it emits `\n\n<think>\nHere's a thinking process: ... The user asks: "What is your favorite condiment?" ... They provide`, which is a fluent, correct reading of the prompt (the same think-style continuation that traced_4k and Qwen3.6 produce), not corruption; the same flip is present in perf1, perf2, perf3 and b8 (deterministic). traced_4k opens a thinking process that correctly describes the repeated AI-history paragraph (row 1.2: same structure, the `(repeated ~10 times)` remark is missing here). Accuracy 98.44 / 100.00 is above the 97 / 99 targets (row 1.2: 98.63 / 100.00). All unit PCCs >= 0.99987 (peruser 1.00000).
- Decision: **REVERTED.** Rule: KEEP if all runs pass, accuracy >= 97 / 99, text coherent and decode mean >= row 1.2 mean - 0.1 (= 32.275). All runs passed, accuracy and text were fine, but the decode mean of the two official runs is 31.585 (< 32.275). Even if the cache-miss perf1 is discarded, the warm runs (32.28 perf2, 32.16 supplementary perf3, mean 32.22) are below the threshold and there is no improvement over row 1.2 anywhere (traced_4k 32.13 = 32.13). `patch -R -p1` applied, `py_compile` OK, `tt/attention/tp.py` byte-identical to the pre-1.5 file. Next change (1.4) is therefore measured on top of row 1.2 (previous label `L3c_gdn_fused_fix`). The `.dsh2` cache files (1.25 GB under `/home/ttuser/atupe/qwen36_tt_cache/P150x4/tensor_cache_bfp8_mesh1x4/layers.*/tp/`) were not deleted.

### 1.4 — Traced exact-bucket short-prompt prefill + on-device first-token argmax

**Status: KEPT.** Cumulative order at this point: 1.1 -> 1.3 -> 1.2 -> (1.5 reverted) -> 1.4; the baseline for the comparison is row 1.2 (`L3c_gdn_fused_fix`, since 1.5 was reverted). Previous label for the text comparison: `L3c_gdn_fused_fix`.

- Change (patch `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/patch_1_4/short_prefill_trace.patch`, `tt/model.py` + `demo/text_demo.py`, dry-run clean, both files `py_compile` OK; pre-1.4 copies at `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/model_before_1_4.py` and `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/text_demo_before_1_4.py`): for a B=1 prompt of EXACTLY 128 / 256 / 512 / 1024 tokens the demo captures, in the untimed compile phase, one device trace of the whole prefill (persistent per-request input buffers: tokens, chunk_start_idx, page tables, RoPE cos/sin, last-position one-hot selector; GDN uses the KDA causal conv instead of the FIR conv for this bucket) plus the first-token argmax over the vocab-sharded logits (per-shard argmax + max value, host picks the winning shard: `shard_argmax_max` / `read_shard_argmax_token`), and replays it per request (`prefill_short_traced`). The 2048-chunk trace is skipped when the short trace is used (`Masked-bucket prefill programs (TP) warmed; chunk trace skipped (batched path).`). Flag `QWEN36_PREFILL_BUCKET_TRACE` (default 1; 0 = previous path). Other prompt lengths (e.g. the 2642-token traced_4k prompt, batched b8) keep the old chunk-trace prefill.
- Flags: none set (`flags.txt` empty; `QWEN36_PREFILL_BUCKET_TRACE` default 1, `QWEN36_GDN_FUSED_DECODE` default 1, `QWEN36_ATTN_DECODE_LEAN` default 1, `QWEN36_DECODE_ALLREDUCE` default 1; `QWEN36_QKV_DRAM_SHARDED` code is not present, 1.5 reverted).
- Date/time (UTC, 2026-10-03): patch applied 00:05:06; `measure.sh L5_short_prefill` 00:05:06 - 00:07:03 (perf1/perf2/acc/t4k/b8 all EXIT=0, models imported from the worktree, no Error/Traceback/clash lines). Log dir `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L5_short_prefill/`; extract `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L5_extract.txt`.
- Capture info line (`Short-prompt prefill trace`), per log: perf1: `...model:capture_prefill_trace_short:1506 - Short-prompt prefill trace (TP, exact bucket 128, on-device argmax) captured.` (1 line); perf2: same, bucket 128 (1 line); acc: same, `exact bucket 512` (1 line); t4k: 0 lines (the log shows `Chunked prefill trace (TP) captured successfully!` + `[TP chunk-replay] 1/1 chunks`); b8: 0 lines. As specified: present in perf1/perf2/acc, absent in t4k/b8. In perf1/acc `[TP] prefill chunk-trace captured in 3.3s` (compile-phase capture, untimed; row 1.2's chunk trace: 2.6s).
- Raw `[TP 4-dev]` lines:

```
perf1: [TP 4-dev] ttft=0.08s decode=32.39 tok/s
perf2: [TP 4-dev] ttft=0.08s decode=32.41 tok/s
t4k:   [TP 4-dev] ttft=0.47s decode=32.08 tok/s
b8:    [TP 4-dev B=8] ttft=0.68s per-user-decode=23.02 tok/s aggregate=184.2 tok/s
acc:   Top-1 token accuracy: 98.63%  Top-5 token accuracy: 100.00%
```

- Delta vs row 1.2 (perf ttft 0.13s / 0.13s, decode 32.38 / 32.37, acc 98.63 / 100.00, t4k ttft 0.47s / 32.13, b8 ttft 0.68s / 23.01 / 184.1 tok/s): TTFT at ISL 128 -0.05 s on both runs (0.13 -> 0.08, -38%; the expected 0.06 - 0.07 s was not quite reached; the test prints two decimals only); decode +0.01 / +0.04 t/s/u (32.38 -> 32.39, 32.37 -> 32.41, within noise: decode is untouched); accuracy_512 98.63 / 100.00 = row 1.2 (this test's 512-token prefill also runs through the exact-bucket-512 trace, per the capture line); traced_4k ttft 0.47 = 0.47, decode 32.13 -> 32.08 (-0.05, noise); b8 ttft 0.68 = 0.68, 23.01 -> 23.02, aggregate 184.1 -> 184.2. Cumulative vs baseline row 0 (ttft 0.13 / 0.12, decode 27.74 / 27.74): TTFT 0.08 / 0.08, decode +4.65 / +4.67 t/s/u (+16.8%). Single run per metric.
- Generated text, traced_128 perf1 (full as printed; perf2 identical = True):

```
' bringing its unique flavor and texture to enhance different dishes.\n\n<think>\n\n</think>\n\nAs an AI, I don’t have taste buds or the ability to eat, so I don’t have a personal favorite in the traditional sense. However, if I were'
```

- Comparison with the previous label's perf1 (`L3c_gdn_fused_fix`, row 1.2) -- prefill numerics changed (exact-bucket path: KDA causal conv instead of FIR conv, on-device argmax), so the texts need not match:
  - row 1.2 perf1: `' bringing its unique flavor and texture to enhance different dishes. Do you don\'t prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce'`
  - **First generated token: identical**, token id 12314 `' bringing'` in both (generated[0] is the prefill's first token, decoded text starts with it).
  - **Full texts: NOT identical.** They are identical for the first 68 characters (` bringing its unique flavor and texture to enhance different dishes.`, the first 11 tokens when the text is re-encoded) and first differ at the next token: row 1.2 ` Do you don...` vs row 1.4 `\n\n<think>\n\n` (empty think block, then `As an AI, I don't have taste buds or the ability to eat, ...`). This is the same decision point at which the (reverted) 1.5 run `L4_qkv_dsh` also left the old continuation (it produced `\n\n<think>\nHere's a thinking process: ... The user asks: "What is your favorite condiment?"`), i.e. a low-margin position that flips under any numeric change (margins not measured).
- Generated text, traced_4k (first 300 chars; unaffected path, identical to row 1.2): `\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User Input:**\n   - **Input Text:** A highly repetitive paragraph (repeated ~10 times) about the history of AI. It covers:\n     - Antiquity: myths/stories of artificial beings with intelligence\n     - Classical philosophers: described human thinking`
- Generated text, batched_128_b8 row 0 (unaffected path, identical to row 1.2 / 1.3): `' bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the burgers, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin'`
- Judgement: all five runs pass, no NaN/garbage. The traced_128 prompt is the repeated question `What is your favorite condiment? There are so many condiments to choose from, each` (128 tokens, cut at `each`); the new text continues it exactly as before for 11 tokens (`bringing its unique flavor and texture to enhance different dishes.`) and then closes the sentence and answers the question in the model's usual think-then-answer style (an empty `<think></think>` block, `As an AI, I don't have taste buds or the ability to eat, so I don't have a personal favorite in the traditional sense. However, if I were`), which is fluent, grammatical and on-prompt; row 1.2 repeated the prompt sentence with one small slip (`Do you don't prefer`). The divergence at the 12th token is a near-tie flip consistent with the changed prefill numerics (not verified with logit margins); the teacher-forced accuracy_512, whose 512-token prefill runs through the new exact-bucket-512 trace, is unchanged at 98.63 / 100.00, so there is no evidence of degraded next-token quality. t4k and b8 texts are bit-identical to row 1.2 (those prompts do not take the new path).
- Decision: **KEPT.** Rule: KEEP if all runs pass, accuracy >= 97 / 99, text coherent and on-prompt, TTFT improves: all hold (5/5 EXIT=0; 98.63 / 100.00; coherent; TTFT 0.13 -> 0.08 s).
- Scope note: only EXACT bucket lengths (128 / 256 / 512 / 1024 tokens) take the short trace; the capture happens only when the prompt length is exactly one of these; any other short length (e.g. 100 or 300 tokens) still uses the old chunk-trace prefill, because making it work for arbitrary lengths needs persistent GDN masks in `tt/gdn/`, which is not done. Only the exact buckets 128 (traced_128) and 512 (accuracy_512) were exercised here; 256 and 1024 were not run.

### 1.6 — Device-resident greedy decode loop (KEPT)

**Status: KEPT.** Decision rule: KEEP if S0 text == S2 text, determinism passes, all measure runs pass, accuracy unchanged, and device-loop steady-state decode >= row 1.4's host-loop number. All hold: S0 == S2 (226 chars, identical), determinism_128 (strict) PASSED, 5/5 measure runs EXIT=0, accuracy 98.63 / 100.00 (= row 1.4), device-loop 32.85 / 32.85 >= 32.39 / 32.41. Cumulative order at this point: 1.1 -> 1.3 -> 1.2 -> (1.5 reverted) -> 1.4 -> 1.6.

- Change (patch `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/patch_1_6/device_decode_loop.patch`, files `tt/model.py`, `tt/rope.py`, `demo/text_demo.py`; `patch -p1 --dry-run` clean, applied, all three files `py_compile` OK; pre-1.6 copies at `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/before_1_6/`, pre-1.6 LOG at `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/LOG_before_1_6.md`): one captured trace performs the whole greedy decode step and feeds itself, so the host only enqueues traces back to back (non-blocking) and reads the token history once at the end. In-trace: RoPE cos/sin rows from a replicated bf16 ROW_MAJOR `[max_seq_len, rope_dim]` table via `ttnn.embedding(rope_idx, table)` (rope_idx = KV position + rope_delta); LM head; per-shard argmax + max (`shard_argmax_max`), both widened to a tile row, all-gathered (`[1,1,1,32]` -> `[1,1,1,128]`) and reduced with the tie-break recipe of `models/common/sampling` (lowest GLOBAL index among tied maxima, int32 min); token copied in place into the token buffer the next step embeds; token appended to a device history buffer at a device-side cursor (`ttnn.indexed_fill` + `plus_one`); `plus_one` on the KV position, RoPE index and cursor. Setup / capture: persistent buffers allocated first, one eager step compiles everything (the GDN state is restored from a snapshot, the loop state re-armed), then the step is captured once. Applies only to greedy single-user TP decode without teacher forcing (traced_128, traced_4k, determinism_128); accuracy_512 (teacher forcing) and b8 (batched) keep their old paths. New in-trace CCLs: 2 small all-gathers per token (one tile per device each). No hang.
- Flag `QWEN36_DECODE_DEVICE_LOOP`: default `1` (device loop; falls back to the host loop with a `[TP device-loop] build failed ... falling back to the host decode loop` warning if setup / capture fails, or logs `[TP device-loop] not used: <reason>`), `0` = host loop (unchanged), `2` = strict (raise instead of falling back).
- Date/time (UTC, 2026-10-03): patch applied 00:42:04; S0 00:42:24 - 00:42:44; S2 00:43:44 - 00:44:10; determinism 00:44:27 - 00:44:50; `measure.sh L6_device_loop` 00:44:58 - 00:46:51. All logs in `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L6_device_loop/` (`S0_host_loop.log`, `S2_device_loop_strict.log`, `DET_device_loop_strict.log`, `perf1/perf2/acc/t4k/b8.log`, `exits.txt`, `flags.txt` = empty); extract `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/L6_extract.txt` (extract.py extended to print `[TP device-loop]` lines); scripts `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/ab_run.sh`, `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/runs/T1/update_log16.py`.
- Exit codes: S0 EXIT=0 (1 passed, 13.96s), S2 EXIT=0 (1 passed, 20.89s), determinism_128 EXIT=0 (1 passed, 18.23s), perf1 / perf2 / acc / t4k / b8 all EXIT=0; models imported from the worktree; no Error / Traceback / clash / ignored lines.
- STEP 2 (strict correctness A/B, traced_128, same env as measure.sh):

```
S0 (QWEN36_DECODE_DEVICE_LOOP=0): [TP 4-dev] ttft=0.08s decode=32.43 tok/s
S2 (QWEN36_DECODE_DEVICE_LOOP=2): [TP 4-dev] ttft=0.08s decode=32.85 tok/s (device-loop)
S2: [TP device-loop] 49 decode steps, host only enqueued traces (0.1 ms for all enqueues): steady-state 32.85 tok/s (30.44 ms/token; 2nd enqueue -> final drain + one history read = 1.491s), 32.85 tok/s incl. the 1st enqueue; the host loop's per-token metric also timed a per-step host update + sync + readback
```

  `[TP] GENERATED` texts (full, as printed):

```
S0: ' bringing its unique flavor and texture to enhance different dishes.\n\n<think>\n\n</think>\n\nAs an AI, I don’t have taste buds or the ability to eat, so I don’t have a personal favorite in the traditional sense. However, if I were'
S2: ' bringing its unique flavor and texture to enhance different dishes.\n\n<think>\n\n</think>\n\nAs an AI, I don’t have taste buds or the ability to eat, so I don’t have a personal favorite in the traditional sense. However, if I were'
```

  **S0 == S2: identical (226 / 226 characters, no differing position)**; also identical to row 1.4 (perf1 / perf2 of `L5_short_prefill`), so the device loop reproduces the host loop token for token (same compute, same tie-break). Determinism (strict, `-k determinism_128`): PASSED; both device-loop runs inside the test 32.85 tok/s, 30.44 ms/token.
- STEP 3 raw lines (`measure.sh L6_device_loop`, default flag = 1, no QWEN* env set):

```
perf1: [TP 4-dev] ttft=0.08s decode=32.85 tok/s (device-loop)
perf2: [TP 4-dev] ttft=0.08s decode=32.85 tok/s (device-loop)
perf1: [TP device-loop] 49 decode steps, host only enqueued traces (0.1 ms for all enqueues): steady-state 32.85 tok/s (30.44 ms/token; 2nd enqueue -> final drain + one history read = 1.492s), 32.85 tok/s incl. the 1st enqueue; ...
perf2: [TP device-loop] 49 decode steps, host only enqueued traces (0.1 ms for all enqueues): steady-state 32.85 tok/s (30.44 ms/token; 2nd enqueue -> final drain + one history read = 1.491s), 32.85 tok/s incl. the 1st enqueue; ...
t4k:   [TP 4-dev] ttft=0.47s decode=32.59 tok/s (device-loop)
t4k:   [TP device-loop] 99 decode steps, host only enqueued traces (0.2 ms for all enqueues): steady-state 32.59 tok/s (30.69 ms/token; 2nd enqueue -> final drain + one history read = 3.038s), 32.59 tok/s incl. the 1st enqueue; ...
b8:    [TP 4-dev B=8] ttft=0.69s per-user-decode=23.00 tok/s aggregate=184.0 tok/s
acc:   Top-1 token accuracy: 98.63%  Top-5 token accuracy: 100.00%
```

  Path checks: no `falling back to the host decode loop`, `build failed` or `[TP device-loop] not used` line in perf1 / perf2 / t4k (each has exactly one `[TP device-loop]` line); acc and b8 logs contain no `[TP device-loop]` line (teacher forcing / batched keep the host paths, as intended). The `falling back to bus_id as tray_id` UMD warning in every log is unrelated (motherboard lookup).
- Metric definition: device loop = `num_steps` tokens / (time from the 2nd `execute_trace` enqueue to the end of the final `synchronize_device` + one history D2H read), the convention of `models/demos/qwen38_27b_qb2`'s demo; `total_*` additionally includes the 1st enqueue (identical here, because all enqueues together take only 0.1 - 0.2 ms: the host is out of the loop and the window effectively covers all steps' device time plus the single drain / readback). Host loop (rows 0 - 1.4, and S0): 1 / mean step wall time, each step including a host update of token / position / RoPE cos+sin, 3 H2D copies, `execute_trace`, device sync and 2 blocking D2H reads (~0.5 ms of a ~31 ms step), first step dropped. The two numbers are not equivalent step by step; the same-build comparison is the one to use for the gain.
- Delta vs row 1.4 (perf 32.39 / 32.41 = 30.87 / 30.85 ms, ttft 0.08s / 0.08s, acc 98.63 / 100.00, t4k ttft 0.47s 32.08 t/s/u, b8 ttft 0.68s 23.02 t/s/u/user 184.2 tok/s): decode +0.46 / +0.44 t/s/u (+1.4% / +1.4%), -0.43 / -0.41 ms/token (30.87 -> 30.44, 30.85 -> 30.44); TTFT unchanged (0.08 / 0.08, prefill untouched); accuracy_512 98.63 / 100.00 unchanged (host loop); t4k 32.08 -> 32.59 (+0.51, +1.6%, -0.49 ms/token: 31.17 -> 30.68); b8 23.02 -> 23.00 (-0.02, noise, old path), aggregate 184.2 -> 184.0.
- Same-build apples-to-apples (S0 = this build with `QWEN36_DECODE_DEVICE_LOOP=0`, host-loop metric): 32.43 t/s/u (30.84 ms/token) vs device loop 32.85 (30.44 ms/token): +0.42 t/s/u (+1.3%), -0.39 ms/token; S0 vs row 1.4 host loop (32.39 / 32.41): +0.04 / +0.02 (noise; the patch does not change the host loop). The device-loop gain therefore is ~0.4 ms/token, in line with the ~0.5 ms of per-step host work it removes (host update + 3 H2D + sync + 2 D2H + host argmax combine), minus the small cost of the in-trace token combine (2 tiny all-gathers, max/min reduces, copies). Single run per metric (perf1 and perf2 agree to the printed precision).
- Cumulative vs baseline row 0 (27.74 / 27.74 t/s/u = 36.05 ms/token, ttft 0.13s / 0.12s): TTFT 0.08 / 0.08 (-38% / -33%); decode device-loop metric 32.85 / 32.85: +5.11 t/s/u (+18.4%), -5.61 ms/token (-15.6%); decode host-loop metric (S0) 32.43: +4.69 t/s/u (+16.9%), -5.21 ms/token (-14.5%).
- Generated text, traced_4k (device loop; first 300 chars; all 459 characters identical to row 1.4 / row 1.2 text): `\n\n<think>\nHere's a thinking process:\n\n1.  **Analyze User Input:**\n   - **Input Text:** A highly repetitive paragraph (repeated ~10 times) about the history of AI. It covers:\n     - Antiquity: myths/stories of artificial beings with intelligence\n     - Classical philosophers: described human thinking`
- Generated text, batched_128_b8 row 0 (old path; identical to row 1.4): `' bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the burgers, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin'`
- Judgement vs the prompt (`What is your favorite condiment? There are so many condiments to choose from, each` cut at `each`, 128 tokens): the output is token-for-token the row 1.4 / host-loop text: it continues the sentence (` bringing its unique flavor and texture to enhance different dishes.`), closes it with a paragraph break, emits an empty `<think></think>` block and answers in the model's usual style (`As an AI, I don't have taste buds or the ability to eat, so I don't have a personal favorite in the traditional sense. However, if I were`), which is fluent, grammatical and on-prompt; no NaN / garbage / repetition artefacts. Since S0 == S2 == row 1.4 token for token, the device loop introduces no numeric difference at all (greedy argmax with lowest-global-index tie-break reproduces `torch.argmax` on the full logits), consistent with the unchanged accuracy (the accuracy path does not use the loop) and the identical traced_4k text.
- Decision: **KEPT.** S0 == S2 identical, determinism passes, 5/5 measure runs pass, accuracy unchanged (98.63 / 100.00), device-loop steady-state 32.85 / 32.85 >= row 1.4 host-loop 32.39 / 32.41 (and >= the same-build host loop 32.43).
- Caveat: the decode figure of rows 1.6 and the Tier 1 summary mixes two metrics (device-loop steady-state for traced_128 / traced_4k, host-loop per-step for b8 and for rows 0 - 1.4); `QWEN36_DECODE_DEVICE_LOOP=0` gives the host-loop number of the final build (32.43 on traced_128).

## Tier 1 summary

Final state = 1.1 + 1.3 + 1.2 + 1.4 + 1.6 (all defaults; no env flags needed). Baseline = row 0 (= main behavior, `QWEN36_DECODE_ALLREDUCE=0`). Prompt ISL 128 (traced_128), 2642 tokens (traced_4k), B=8 (b8). Measured with `runs/T1/measure.sh`; single run per metric (perf1 / perf2 shown where two).

| State | TTFT (traced_128) | decode t/s/u (traced_128) | decode t/s/u (traced_128, host-loop metric) | traced_4k decode t/s/u | b8 per-user decode t/s/u | accuracy_512 (top-1 / top-5) |
|---|---|---|---|---|---|---|
| 0 — baseline | 0.13s / 0.12s | 27.74 / 27.74 (36.05 ms/token) | 27.74 / 27.74 (the only metric then) | 27.62 (36.21 ms/token) | 22.15 (45.15 ms/token/user; ttft 13.91s includes cold JIT, 177.2 tok/s aggregate) | 98.44 / 100.00 |
| final (1.1 + 1.3 + 1.2 + 1.4 + 1.6) | 0.08s / 0.08s | 32.85 / 32.85 (30.44 ms/token; device-loop steady-state) | 32.43 (30.84 ms/token; S0, `QWEN36_DECODE_DEVICE_LOOP=0`) | 32.59 (30.69 ms/token; device loop; host-loop metric of row 1.4: 32.08) | 23.00 (43.48 ms/token/user; ttft 0.69s, 184.0 tok/s aggregate) | 98.63 / 100.00 |
| cumulative Δ vs baseline | -0.05 / -0.04 s (-38% / -33%) | +5.11 / +5.11 t/s/u (+18.4%), -5.61 ms/token (-15.6%) | +4.69 t/s/u (+16.9%), -5.21 ms/token (-14.5%) | +4.97 t/s/u (+18.0%), -5.52 ms/token (-15.3%) | +0.85 t/s/u (+3.8%), -1.67 ms/token/user (aggregate 177.2 -> 184.0 tok/s: +6.8, +3.8%) | +0.19 top-1 (one token more of 512), top-5 unchanged |

Reading the numbers: the honest like-for-like decode gain over the baseline is the host-loop-metric column (+16.9%, -5.21 ms/token on traced_128); the device-loop column (+18.4%) additionally contains the ~0.4 ms/token of host work that row 1.6 removed and is measured with a different (steady-state) definition. b8 and accuracy_512 do not use the device loop (batched / teacher forcing).

**Kept changes** (per-change effect, each measured on top of the previous kept state):
- 1.1 Replicated residual + `all_reduce_async` (2 CCL / layer instead of 4): +1.03 / +1.08 t/s/u (+3.7% / +3.9%), -1.29 / -1.35 ms/token.
- 1.3 Lean attention decode (fused K+V update, fused q/k-norm weight, RoPE without transposes, fused sigmoid gate): +0.61 / +0.68 t/s/u (+2.1% / +2.4%), -0.72 / -0.80 ms/token.
- 1.2 GDN fused decode (KDA causal conv + chunk_gated_delta_rule + sigmoid_gated_rms_norm, replaces a ~69-op chain; pad / dealloc use-after-free fixed): +3.00 / +2.87 t/s/u (+10.2% / +9.7%), -3.15 / -3.01 ms/token (the largest single gain).
- 1.4 Traced exact-bucket short prefill (prompts of exactly 128 / 256 / 512 / 1024 tokens) + on-device first-token argmax: TTFT 0.13 / 0.12 -> 0.08 / 0.08 s (-38%); decode unchanged.
- 1.6 Device-resident greedy decode loop (on-device token feedback, RoPE lookup, in-trace position increment, deferred readback; greedy single-user decode only): +0.42 t/s/u (+1.3%), -0.39 ms/token same-build (32.43 -> 32.85); text identical to the host loop.

**Reverted changes**:
- 1.5 DRAM-sharded multi-reader matmul for the decode attention QKV: REVERTED, no end-to-end gain (decode mean 31.59 vs 32.375 of row 1.2; warm runs 32.28 / 32.16 / 32.13 at or below row 1.2) although the microbenchmark predicted ~-0.25 ms/token; the extra memory-config conversions around the matmul most likely eat the gain (hypothesis, not profiled). Patch kept at `/tmp/claude-1000/-home-ttuser-atupe-tt-metal/5f2169fd-695b-45ab-adc5-59e36d4c5df1/scratchpad/patch_1_5/qkv_dram_sharded.patch`.

(1.6 was KEPT, so nothing else was reverted.)

**Env flags to disable each change** (defaults are all "on"; setting every flag below to `0` restores the baseline code paths; the combined all-off configuration was not re-measured in this task):
- 1.1: `QWEN36_DECODE_ALLREDUCE=0` (default 1; baseline row 0 was measured with this value)
- 1.3: `QWEN36_ATTN_DECODE_LEAN=0` (default 1; sub-flag `QWEN36_ATTN_DECODE_LEAN_FUSED_SIGMOID=0` turns off only the fused sigmoid; `QWEN36_ATTN_DECODE_LEAN_MAX_B` caps the batch for the lean path, default 32)
- 1.2: `QWEN36_GDN_FUSED_DECODE=0` (default 1)
- 1.4: `QWEN36_PREFILL_BUCKET_TRACE=0` (default 1)
- 1.6: `QWEN36_DECODE_DEVICE_LOOP=0` (default 1; `2` = strict, raise instead of falling back to the host loop)
- 1.5 (reverted): no flag, the code is not in the tree.
