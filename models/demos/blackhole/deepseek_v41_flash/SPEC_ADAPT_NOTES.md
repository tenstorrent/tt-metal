# Adaptive speculative verification length ("adaptive up to 5") - branch ssinghal/dsv4p1-spec-adapt (base ssinghal/dsv4p1 44878e2dd8d)

Enable: `DSV41_SPEC=<max k> DSV41_SPEC_ADAPT=1 [DSV41_SPEC_SET=1,3,5]` (default off; without ADAPT nothing changes). Default sets (`spec_model.default_ks`): U=users/mesh row=2 (B=8): {1,3,5}, U=4 (B=16): {1,3,5},
U=8 (B=32): {1,3}; T = U*(1+k) rows per mesh row must be a fast/supported mHC row count (8/12/16/24/32 ok; T=12/20 pad; T>32 needs the spec-rows chunked verify, not in this branch).

## Design
* One resident, traced `SpecRunner` per candidate k. All runners share ONE drafter (weights, rings: `DSparkDrafter.view_n(n)`, `ChunkedDrafter.view_n`) and the model pool; the drafter always proposes 5 drafts per user and a round of
  runner k verifies the first k of them (n = 1+k rows/user). DRAM: the first runner (views + drafter) costs +106.7 MiB/bank, every further runner only +0.6-0.8 MiB/bank (40 layers, B=16), trace region separate (1.9 GB fits 3 runners).
* Accept rule unchanged (plain greedy) => the stream equals plain greedy for every k (up to bf16 near-ties, see exactness). Batch-level k (rows are shared by the whole mesh row); per-user keff masking is NOT used: it only loses accepted tokens and saves no rows.
* Confidence head: `tt/mtp.py` already computed `conf` = fp32 logit; it is now quantised on device (65535*sigmoid) and appended to the round readback (no extra transfer). p_j = sigmoid(logit_j) = P(draft j accepted | drafts < j accepted)
  (reference `DSparkBlock.forward_head`: confidence of position i uses the hidden state and the Markov embedding of the token feeding position i).
* Scheduler (`AdaptiveSpec.choose`): E[tokens](k) = 1 + sum_{j<=k} S_j, S_j = mean over active users of prod_{i<=j} p_i; pick argmax_k E[tokens](k)/round_ms(k). round_ms(k): startup calibration (every trace is replayed
  3x with forced accepts when captured, `DSV41_SPEC_CALIB`, default 3) then an EMA (0.8/0.2) of the measured rounds; `DSV41_SPEC_TIMES='1:70,3:83,5:100'` overrides; `DSV41_SPEC_HYST` adds hysteresis (default 0); `DSV41_SPEC_CONF_AB=a,b` recalibrates p = sigmoid(a*logit(p)+b) (identity by default, calibration below says it is not needed).
* Policies: `DSV41_SPEC_POLICIES` = `adapt` | `k<j>` | `cycle:adapt+k1+k3+k5:W` (A/B of policies in windows of W rounds on the SAME stream inside one pass; per-policy results are `SPEC_RESULT_CYCLE` lines). `SPEC_CONF` lines carry the reliability bins.
* Tests/tools: `tests/test_spec_adapt_sched.py` (CPU scheduler test), `tools/adapt_run.sh` (host launcher with flock + hangwatch), `tools/spec_adapt_report.py <log>` (tables from the logs), prompts `demo/sample_prompts/input_data_struct_16.json` (JSON/CSV/code/repeated text; scenarios `struct_b8/16/32`).

## Hazards found (important for merging with spec-rows / spec-k)
1. **Several resident traces hang unless ALL eager compile passes run before the FIRST trace capture** (replaying trace A then trace B hung, reproduced with a pure switch test, no seeding/prefill). `SpecRunner.prepare()` (compile pass + snapshots) for every runner, then
   `capture_trace()` for every runner, then calibration (called from `seed`). Probable cause: tensors allocated after a capture land on that trace's freed scratch memory and are overwritten by its replays; this likely explains the spec-k in-process sweep hangs too.
2. **A second prefill after a spec pass makes the next spec compile pass hang** (stack: `SpecRunner.prepare` -> `synchronize_device` after the eager verify). Not caused by: trace overlap with prefill (runner built before any trace: same hang), stale spec traces (released first: same hang), poisoned prefill state
   (plain decode after the re-prefill reproduces the first plain stream 16/16), leftover spec state (second spec pass WITHOUT re-prefill works). Cause unknown. Consequence: **one spec pass per process** (sessions with several spec scenarios, or several spec passes per scenario, are not safe);
   hence policies are compared with the `cycle:` mode inside one pass, and every table cell below is one 40-layer process.
3. `hangwatch.sh` killed only the `timeout` wrapper (python survived and kept the device): now sends USR1 (faulthandler stack in the log) and kills the children. tt-triage cannot read a busy device here (MMIO timeouts); the USR1 stack is more useful.

## Results (40 layers, greedy, Engram on; plain = the same process' plain decode; `cycle` windows of 4 rounds (W=1 where marked) on the same stream; tok/s/user = tokens per round / round wall ms)
Round ms are measured walls (host feed + trace + readback). acc = accepted drafts/round. Speedup vs the plain decode of the same run.

| B | workload | plain tok/s/u | k=1 | k=3 | k=5 | ADAPT | ADAPT k-mix | note |
|---|---|---|---|---|---|---|---|---|
| 8 | GSM8K | 21.3 | 29.4 (1.38x, acc 0.83, 62 ms) | 43.5 (2.04x, 2.16, 72 ms) | **47.3 (2.22x, 3.40, 92.5 ms)** | 46.4 (2.18x) | 3:10 5:14 | |
| 8 | struct (JSON/code) | 21.0 | 29.6 (1.41x) | 50.3 (2.39x, 2.76) | 55.3 (2.63x, 4.27, 95 ms) | **57.1 (2.72x)** | 3:4 5:18 | |
| 8 | isl4k (ISL 3720) | 21.4 | 28.4 (1.32x) | 26.3 (1.23x) | 26.6 (1.24x) | 26.4 (1.23x) | 1:8 3:1 | 8 rounds/policy |
| 8 | isl64k (ISL 60453, idx matmul) | 19.6 | 19.7 (1.00x) | **33.9 (1.73x, acc 1.88)** | 22.9 (1.17x) | 25.6 (1.30x) | 1:4 3:4 | 8 rounds/policy, 8/8 exact |
| 16 | GSM8K | 20.7 | 26.0 (1.26x, 0.89, 72.8 ms) | 40.0 (1.93x, 2.40, 84.7 ms) | **43.5 (2.10x, 3.31, 98.6 ms)** | 42.7 (2.07x) | 3:6 5:25 | |
| 16 | struct | 18.1 | 22.8 (1.26x) | 37.4 (2.06x, 2.67) | **45.0 (2.48x, 4.15, 114.6 ms)** | 44.4 (2.45x) | 5:22 | |
| 16 | isl4k, W=1, 192 tok | 22.2 | **25.3 (1.14x)** | 24.4 (1.10x) | 18.2 (0.82x) | 22.7 (1.02x) | 1:13 3:13 | 26 rounds/policy |
| 16 | isl4k, W=4, 64 tok | 21.8 | 21.4 (0.98x) | 25.2 (1.16x) | 15.8 (0.73x) | 25.2 (1.16x) | 1:8 3:4 | 8 rounds/policy |
| 16 | isl64k (idx matmul) | 18.9 | 16.7 (0.89x) | **22.3 (1.18x)** | 15.1 (0.80x) | 20.0 (1.06x) | 1:12 3:4 | 16 rounds/policy, 16/16 exact |
| 32 | GSM8K | 15.0 | 17.9 (1.20x, 105 ms) | **26.0 (1.73x, acc 2.40, 130 ms)** | n/a (T=48) | 25.7 (1.71x) | 3:44 | |
| 32 | struct | 16.1 | 18.2 (1.13x) | 28.1 (1.75x, 2.62) | n/a | **28.8 (1.78x)** | 3:33 | |
| 32 | isl4k, W=4, 64 tok | 19.7 | 16.9 (0.86x) | 19.8 (1.01x) | n/a | 14.4 (0.73x) | 1:7 3:5 | 12 rounds/policy: window-position bias |
| 32 | isl4k, W=1, 192 tok | 19.6 | 17.6 (0.90x) | 15.9 (0.81x) | n/a | **18.0 (0.92x)** | 1:28 3:8 | plain (19.6) still wins |
(ISL 64k B=32: see the end of this file if it finished.)

Exactness vs the plain stream (first divergences are bf16/bfp8 near-ties, same as the fixed-k spec work): GSM8K B=8/16/32 2/8, 4/16, 10/32 identical, struct 4/8, 7/16, 18/32, isl4k B=8/16 0 identical (ALL users diverge at generated token 1 with plain gap 0.059 /
spec gap 0.119: the same near-tie of the identical prompt repeated), isl4k B=32 32/32, isl64k B=8 8/8 and B=16 16/16. "SUSPECTED REAL DIVERGENCES" (plain gap > 0.1) only on struct/GSM8K tokens 22-170 with gaps 0.18-0.53, as in the fixed-k runs. First token spec==plain in every run.

## Confidence head reliability (SPEC_CONF; conditional P(draft j ok | prefix ok): predicted / observed; n)
| run | j=1 | j=2 | j=3 | j=4 | j=5 | prefix survival j=3 / j=5 |
|---|---|---|---|---|---|---|
| GSM8K B=16 | .889/.886 (911) | .852/.845 | .843/.874 | .833/.896 | .791/.825 (240) | .628/.652 , .428/.475 |
| GSM8K B=8 | .879/.867 | .857/.862 | .824/.811 | .876/.922 | .828/.819 | .622/.615 , .488/.500 |
| GSM8K B=32 | .906/.894 (2006) | .875/.882 | .858/.880 | - | - | .678/.692 |
| struct B=16 | .949/.948 (1044) | .934/.919 | .933/.935 | .927/.918 | .927/.944 | .829/.818 , .720/.704 |
| struct B=8 | .969/.963 | .940/.935 | .944/.942 | .941/.925 | .969/.970 | .863/.854 , .787/.770 |
| isl4k B=16 (3720 tok) | .601/.556 (576) | .483/.583 | .486/.286 (112) | - | - | .136/.100 |
| isl64k B=8 | .674/.621 | .603/.769 | .425/.500 | .196/.500 (16) | - | .213/.294 |
The head is well calibrated on GSM8K and structured text (sigmoid(logit) as is; slightly conservative at depth), noisier but unbiased on long documents (small n deep in the block). No recalibration needed; `DSV41_SPEC_CONF_AB` is there if a different checkpoint needs it.

## Findings
* Fixed k=5 (T=24, B=8/16) beats the previous best fixed k=3 on GSM8K (2.10x vs 1.93x at B=16; 2.22x vs 2.04x at B=8) and clearly on structured text (2.48-2.63x vs 2.06-2.39x): the T=24/T=12 rows cost only 12-20 ms more than T=16/T=8 once the mHC mixes pad to a multiple of 8.
* ADAPT is within noise of the best fixed k on every workload where acceptance is high (never more than ~3% behind) and avoids the fixed-k=5 loss at low acceptance (isl4k/64k: k=5 0.73-0.82x, ADAPT 1.02-1.30x), but on long documents it is 10-24% below the best fixed k in 3 of 6 cells
  (isl4k B=16 -10%, isl64k B=16 -10%, isl64k B=8 -24%; a k1/k3 coin flip around the break-even; with 8-26 rounds/policy and windows of different positions the cells are noisy, the cause was not isolated). Treat ADAPT as 'safe, not optimal' on long documents.
* Where spec itself loses to plain (isl4k B=32: plain 19.6 tok/s/u, best spec 18.0 and 15.9 for k=3) a fixed or adaptive k cannot help; that needs a k=0 plain-like mode (verify 1 row + write_main, no drafting, probe every N rounds) - not implemented (the draft state would have to be refreshed on re-entry).
* B=32: only k in {1,3} fit T<=32; k=5 (T=48) needs the spec-rows chunked verify (`DSV41_SPEC_ROWS=1`): worth it for structured text (acc 4.2 vs 2.7 at B=8/16).
* B=4 is blocked by the drafter (T=5 rows assert), B=64+ only has k=1.

## Recommended defaults
B=8 and B=16: `DSV41_SPEC_ADAPT=1 DSV41_SPEC_SET=1,3,5` (ADAPT within 3% of the best fixed on GSM8K/struct, 2.1-2.7x; protects against the k=5 loss on long documents). If one fixed k is wanted: k=5 for short-context chat/code, k=3 for ISL >= 4k.
B=32: `SET=1,3` (ADAPT == k=3 on GSM8K/struct 1.7-1.8x). Run one spec pass per process (hazard 2).

## Reproduce
```
tools/adapt_run.sh <name> "DSV41_LAYERS=0-39 DSV41_SPEC=5 DSV41_SPEC_ADAPT=1 DSV41_SPEC_SET=1,3,5 DSV41_SPEC_POLICIES=cycle:adapt+k1+k3+k5:4 [DSV41_PREFILL_ROW_TOKENS=2048 DSV41_IDX_BACKEND=matmul DSV41_MAX_GEN=192]" <scenario: gsm8k_b16|struct_b16|isl4k_b16|isl64k_b8..>
python tools/spec_adapt_report.py /mnt/tt-data/ssinghal/dsv4-logs/spec_adapt_<name>_h<host>.log
```
Logs: /mnt/tt-data/ssinghal/dsv4-logs/spec_adapt_{f8,f16,f32,g16,g32}_*_h*.log.

## Which logs are final / exploratory
Final result lines (40 layers, one spec pass per process, `cycle` policy): spec_adapt_f8_gsm_h34, f8_struct_h34, f8_isl4k_h40, f8_isl64k_h33, f16_gsm_h33, f16_struct_h40, f16_isl4k_h41 (W=4, 64 tok), g16_isl4k_h40 (W=1, 192 tok, preferred over f16_isl4k),
f16_isl64k_h30, f32_gsm_h43, f32_struct_h41, f32_isl4k_h44 (W=4, 64 tok, window-position bias), g32_isl4k_h34 (W=1, 192 tok, preferred), g32_isl64k_h33 (B=32 at ISL 60k, still running when this was written: read its SPEC_RESULT_CYCLE lines; if the log has no result the run did not finish).
`f32_gsm_h43` shows "FAILED" at the end only because `-k gsm8k_b32` matches TWO pytest items (the scenario id gsm8k_b32 is listed twice in SCENARIOS, the second is `gsm8k_b32_1`): the first item completed and printed all results (adapt/k1/k3, SPEC_CONF, exactness);
the duplicate then re-ran the demo in the same process and died at `before begin_trace_capture` (TT_THROW SubDeviceManagerTracker ... remote-only MeshDevice, the same "second pass in one process" problem, hazard 2). The numbers in this file are from the first item.
Exploratory only (4-layer smokes, not for numbers): smoke4*, e1..e10 logs (hang bisection: trace switch, re-prefill, early runner build, no-re-prefill control, plain-after-spec check). Stale hung jobs of those runs were killed (the last one on .35).
