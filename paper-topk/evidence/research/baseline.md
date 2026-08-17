# Gate 1 — Matched baseline + harness pinning (threshold-select campaign)

Repo: `/home/nachiket/tt-metal`, branch `nkapre/sorting`, single Blackhole box. Date of analysis: 2026-08-16.
Scope of this report: (1) how the campaign measures, exactly; (2) the pinned baseline numbers and where they live; (3) the harness-pinning checklist with enforcement evidence; (4) the missing §5.1 baseline cell and the command that would measure it.

All paths below are absolute or repo-relative to `/home/nachiket/tt-metal`. Nothing was run; nothing in the repo was edited.

---

## 1. The measurement instrument: `_canonical_topk_sweep.py`

File: `tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py` (2001 lines, committed).

### 1.1 Architecture

- **Two layers**: `ttnn` (op-level, Tracy `DEVICE KERNEL DURATION [ns]`) and `llk` (MATH_ISOLATE cyc/vec from `tt_metal/tt-llk/perf_data/perf_topk_rebuild_xl/*.post.csv`, two-point slope; lines 16–19, 1152–1232).
- **Two modes**:
  - **Classic grid** (default): ops × (N, K, dtype) × arms. Ops: `topk, sort, topk_large_indices, moe_gate` (line 1849). Defaults: `--ns 1024,8192,32768,65534,65536,131072`, `--ks 8,16,32,64,128,256,512,1024,2048`, `--dtypes bf16,fp32` (lines 222–224). 10 measured iters, 3 warmup, 3 trials.
  - **Competition mode** (`--competition`): deterministic K×W table across five layers in FIXED run order (lines 286–292): `op` (topk_large_indices, our column-parallel), `opstock` (as-shipped proxy via rows=2 row-parallel; batch=2, line 1609), `routed` (ttnn.topk largest=True, composite sum over ALL device ops per iteration), `stocknow` (ttnn.topk largest=False, committed header = replay STORE ON), `prebranch` (same with `#define TOPK_DISABLE_REPLAY_STEP 1` armed — the only header-editing layer, run LAST, lines 269–271). Optional `blaze` layer (`--with-blaze`, single fixed cell k=2048 W=65536, lines 299–318). Roofline column is the llm_perf model (PR 671+676), explicitly ASPIRATIONAL (lines 240–256).
- **Child = same file** re-invoked as `python -m tracy -r -v <this file>` with the cell spec riding env var `CANONICAL_SWEEP_CHILD_SPEC` (lines 63–67, 1083, 1904–1907) because tracy's `shell=True` re-invocation mangles argv.

### 1.2 Per-cell discipline (docstring lines 21–43, enforced in code)

- Fresh subprocess per cell under a watchdog (`--timeout`, default 900 s): `run_cell()` lines 1080–1093. A hang in one cell cannot kill the sweep; one Tracy report per cell makes CSV attribution exact.
- Attribution: `parse_tracy_for_cell()` filters by OP CODE (`_op_code_matches`, lines 190–202, FillPad/Pad excluded), orders by GLOBAL CALL COUNT, keeps the LAST `iters` rows (drops correctness call + warmup + cache-miss row), reports **`ns_median`** (lines 883–929). Composite layers (`routed`, `stocknow`): `parse_tracy_composite()` sums per-opcode `ns` per iteration anchored on the top-k row count, warmup-aware by multiplicity (lines 932–1005). **Result JSONs store nanoseconds in field `ns_median`**; the ledger renderer divides by 1000 (`_topk_ledger_render.py:72`).
- Determinism: per-cell seed `competition_seed(k, w, layer_index)` (lines 341–344), seed_index PINNED per layer (lines 272–275) so inserting a layer never reseeds siblings.
- Correctness gates timing (see §3 checklist).
- DPRINT/Watcher env vars scrubbed from the child env (lines 1061–1064) — they share SRAM with the device profiler.
- Reports rewritten after every cell (lines 1691–1692, 1983–1986) so an abort keeps everything measured.
- Statistical gate (classic mode): speedups printed only when |delta| > 2·pooled_std, else `~1.00 (noise)` (lines 1275–1281).

### 1.3 A/B arms — the SWEEP_ARM header edit

- Mechanism: marker block inserted at the very top of `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h` (`HEADER_RELPATH`, line 170). Arms today: `baseline` = header as-committed = **replay STORE ON** (the branch committed the replay optimization default-ON; header lines 73–78 verified clean on the current tree), `disable_replay` = `#define TOPK_DISABLE_REPLAY_STEP 1` = pre-branch stock kernel (lines 160–177). The old `replay_load`/`replay_store` arm names are retired — inserting them now would measure AS baseline while claiming to be an arm (lines 166–168).
- The script recognizes BOTH marker dialects (`CANONICAL_SWEEP_ARM` and the foreign `SWEEP_ARM_BEGIN/END`) so a stray block from another session can't silently poison "baseline" (lines 769–791).
- `_verify_baseline_header()` (lines 794–805) requires exactly one guarded `#define TOPK_REPLAY_STEP_STORE 1`, one `#define TOPK_REPLAY_STEP_LOAD 1`, zero live `#define TOPK_DISABLE_REPLAY_STEP` after stripping; otherwise the run **refuses to start** (lines 817–822).
- Guarded behind `--allow-header-edit`; asserted via `git diff --name-only` that ONLY the header changed (lines 849–859); restored in `finally` (competition: 1693–1696; classic: 1990–1993); JIT kernel cache `~/.cache/tt-metal-cache` cleared between arms (lines 863–868) as the hard guarantee over trusting the dephash sidecar.

### 1.4 Exact invocations

**(a) §5.1 scope cells — BF16, K≤64, long rows N∈{8192…65536}, single row, ttnn.topk, current stock (replay ON):**

```bash
cd /home/nachiket/tt-metal && source python_env/bin/activate && \
flock /tmp/tt-device.lock \
python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
    --arms baseline --layers ttnn --ops topk \
    --ns 8192,16384,32768,65534,65536 --ks 8,16,32,64 --dtypes bf16 \
    --trials 3 --warmup 3 --timeout 900 \
    --out generated/canonical_sweep/scope51_baseline
```

Notes: classic mode uses a single row (`torch.randn((1,1,1,n))`, line 569). `--arms baseline` needs no header edit and measures today's shipped kernel. The multi-core factory fires only for pow2 N in [8192, 65535) with adjusted-k ≤ 64 (constraint model lines 209–212, 369–375, mirroring `topk_device_operation.cpp:66–75`): so 8192/16384/32768 are multi-core(pred), 65534 (non-pow2 boundary probe) and 65536 (fails the strict W<65535 gate) are single-core — the cliff is part of the baseline. Output: `generated/canonical_sweep/scope51_baseline/{canonical_sweep.csv,canonical_sweep.md,results/*.json}`.

**(b) Large-K cells — K∈{512,1024,2048}, N∈{65536, 262144}, all five competition layers:**

```bash
cd /home/nachiket/tt-metal && source python_env/bin/activate && \
flock /tmp/tt-device.lock \
python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
    --competition --allow-header-edit --with-blaze \
    --ks 512,1024,2048 --ns 65536,262144 \
    --out generated/canonical_sweep/comp_largek
```

Notes: without `--allow-header-edit` the `prebranch` layer is SKIPPED and the other four still run (lines 1669–1674). `--with-blaze` contributes only its native k=2048 W=65536 cell (needs `/home/nachiket/tt-blaze` with a `tt-metal` symlink into THIS repo; preflight lines 1400–1416). Stock layers at W·k ≥ 2²⁴ auto-drop from 5 to 3 measured iters (lines 265–267, 1588–1589). Resume with `--resume` (retries FAILED, skips MEASURED/UNSUPPORTED/WRONG, lines 1646–1654). Output: `competition_table.csv` + `competition_table.md`.

**(c) P-sweep (feeds the ledger PSWEEP_TABLE):** `--competition --layers-competition op --ks <K> --ns <W> --op-num-slices <P>` per P point, one out dir (render script docstring lines 10–13).

**(d) Ledger re-render (numbers only, prose untouched):**

```bash
python tests/ttnn/unit_tests/operations/reduction/_topk_ledger_render.py \
    --competition-dir <comp_out> --psweep-dir <psweep_out>   # default --ledger TOPK_LEDGER.html
```

Splices three marker regions (`EXEC_NUMBERS`, `COMPETITION_TABLE`, `PSWEEP_TABLE`); renders em-dash for anything missing, never invents a number (`_topk_ledger_render.py:15–27, 163–169`).

---

## 2. Pinned baseline numbers (archived data, with provenance)

### 2.1 Where the archives live

- **Committed**: `TOPK_LEDGER.html` (repo root, rendered numbers = the final competition2 run) + `_topk_ledger_render.py`. The ledger is the only committed carrier of the numbers.
- **Session-scratchpad (AT RISK — /tmp, prior session `674c5c85…`)**:
  - `…/scratchpad/sweep/competition2/competition_table.csv` + per-cell `results/*.json` — the final 24/24 competition run. Provenance verified in `results/comp_op_k512_w65536.op.t0.json`: `head_sha=4b3ebaef8e94…`, `so_mtime=1786899330`, `so_md5=890badf1…`, `max_abs_err=0.0`.
  - `…/scratchpad/sweep/ab3/canonical_sweep.csv` — the 90-cell three-arm A/B (arms: baseline=pre-branch, replay_load, replay_store; replay_store column = today's committed stock).
  - `…/scratchpad/sweep/tracy_baseline.csv` — the original op-level Tracy baseline (profiler report `2026_08_16_07_58_06`, still present under `generated/profiler/reports/`).
  - `psweep_tree2/` — P-sweep result JSONs.

### 2.2 The pinned numbers

**Stock ttnn.topk single-core, k=512 @ N=65536** (competition2 row `512,65536`):
- `stocknow` (committed header, replay ON — the number any new work must beat on the stock path): **137,691.78 µs ≈ 137.7 ms**, 1 core.
- `prebranch` (replay disabled, pre-branch kernel): **161,600.04 µs ≈ 161.6 ms**, 1 core. (The ledger's "stock ttnn.topk" column renders `prebranch_us` — `_topk_ledger_render.py:80` — so the HTML shows 161.6 ms; the 137.7 ms stocknow figure lives only in the CSV/JSONs.)
- k=2048 @ 65536: stocknow **539.4 ms**, prebranch **631.5 ms**; k=512 @ 262144: stocknow **552.2 ms**, prebranch **648.4 ms**.

**Routed ttnn.topk (largest=True → TopkLargeIndices composite)** — three eras, all same shape k=512 @ 65536:
- gather-era (PR2, commit 809cf5b): **134.0 µs** (ledger "Routed composite breakdown": xl 34.1 + tilize 59.9 + untilize 18.2 + gather 11.8 + mask 10.0).
- values-native, pre-tree: **112.4 µs** (campaign memory, 96/96 competition run).
- **current (competition2, tree merge landed 8794fbb): 93.44 µs @ 52 cores**; k=2048 @ 65536: **70.79 µs @ 26 cores**; k=512 @ 262144: **145.36 µs**.

**topk_large_indices op (our column-parallel multi-core), competition2**:
- k=2048 @ 65536: **41.87 µs @ 26 cores** (roofline gap 22.8×)
- k=512 @ 65536: **14.96 µs @ 52 cores** (gap 13.3×)
- k=512 @ 262144 (1M-ctx CSA decode shape): **31.97 µs @ 52 cores, P=64 cap, still descending** (gap 13.4×)
- opstock (as-shipped, rows=2 proxy): 356.63 / 279.18 / 1110.98 µs respectively. blaze fused cell: 24.46 µs @ 129 cores (k2048@65536 only; carries the fused-SDPA caveat).

**K≤64 cells (the §5.1 regime) — ab3 90-cell A/B, bf16, single row** (columns: baseline = pre-branch; replay_store = today's stock):

| N | k | cores | pre-branch ns | replay-STORE ns (today's stock) | store speedup |
|---|---|---|---|---|---|
| 4096 | 8 | 1 | 661,536 | 567,578 | 1.166× |
| 4096 | 32 | 1 | 661,591 | 567,614 | 1.166× |
| 4096 | 64 | 1 | 1,288,352 | 1,072,281 | 1.202× |
| 8192 | 8 | 65 | 106,609 | 106,676 | ~1.00× |
| 8192 | 32 | 65 | 106,621 | 106,704 | ~1.00× |
| 8192 | 64 | 65 | 238,963 | 238,505 | ~1.00× |
| 32768 | 8 | 65 | 171,219 | 171,261 | ~1.00× |
| 32768 | 32 | 65 | **171,205** | 171,227 | ~1.00× |
| 32768 | 64 | 65 | 309,020 | 308,509 | ~1.00× |
| 65536 | 8 | 1 | 10,955,739 | 9,492,023 | 1.154× |
| 65536 | 32 | 1 | 10,955,982 | 9,492,354 | 1.154× |
| 65536 | 64 | 1 | 20,888,514 | 17,950,597 | 1.164× |
| 131072 | 8–64 | 1 | 21.9–41.8 ms | 19.0–35.9 ms | 1.154–1.163× |

The "~171 µs @ N=32k/K=32 multi-core" figure the audit cites is confirmed twice: `tracy_baseline.csv` (TopKDeviceOperation rows, report 2026_08_16_07_58_06) and ab3 (171.2 µs, replay-insensitive — multi-core is bound elsewhere). Headline structure of the regime: **multi-core K≤64 lives at ~107–309 µs for N=8192–32768; at N=65536 the strict `W<65535` gate drops to 1 core and the cell costs 9.5–18 ms (stock today)** — a 55× cliff inside the §5.1 window.

---

## 3. Harness-pinning checklist for any new Gate-5 claim

Every item with the code that enforces it (file = `_canonical_topk_sweep.py` unless noted):

1. **Per-cell fresh subprocess + watchdog.** `run_cell()` launches `python -m tracy -r -v <script>` per cell, `timeout=args.timeout`, TimeoutExpired → status FAILED with a tt-smi hint (lines 1080–1094). One Tracy CSV per cell (docstring 24–27).
2. **Correctness gate BEFORE timing.** The first (cache-warming) call is correctness-checked; `wrong` → status **WRONG**, timing never enters the table (child lines 519–529; competition cells carry `strict: True`, line 1619). Routed/stock layers: exact value multiset vs `torch.topk` AND index self-consistency (gather input at returned indices must reproduce values), lines 584–618. `topk_large_indices`: gathered-at-indices vs torch.topk, exact under `strict` (lines 659–677). WRONG/FAILED/UNSUPPORTED never enter numeric columns (`build_competition_table`, lines 1736–1746).
3. **Provenance stamp on every cell record.** `provenance_stamp()` = git HEAD sha + md5 of `git diff --stat` + **`ttnn/ttnn/_ttnn.so` mtime and md5** (mtime-cached), lines 1019–1040, stamped at line 1077. Mid-run rebuild or dirtied tree ⇒ `PROVENANCE DRIFT` banner naming the distinct combos (lines 1762–1772). This exists because a mid-grid `./build_metal.sh` burned the campaign once (docstring 111–114).
4. **Child-script md5 tripwire.** Child stamps its own on-disk md5 into the manifest (lines 140–149, 500); orchestrator compares against launch-time md5 → `CHILD_SCRIPT_DRIFT` note (lines 1108–1113). A dispatch fall-through raises `HARNESS_BUG` → FAILED (retryable), not UNSUPPORTED (lines 745–759).
5. **Replay-STORE arm state verified, not assumed.** `_verify_baseline_header()` refuses to run if anything outside the markers arms `TOPK_REPLAY_STEP_*` (lines 794–822); leftover foreign `SWEEP_ARM` blocks are stripped (lines 769–791); a dirty header without `--allow-header-edit` aborts ("refusing to measure a mystery arm", lines 1643–1644 and 1946–1947); header restored + kernel cache cleared in `finally` (1693–1696, 1990–1993, 863–868); header-editing layer runs LAST (269–271).
6. **Profiler purity.** `TT_METAL_DPRINT*` / `TT_METAL_WATCHER*` scrubbed from the child env (lines 1061–1064). Metric is Tracy `DEVICE KERNEL DURATION [ns]`, never wall clock (lines 895–897).
7. **Device flock — CONVENTION, NOT SELF-ENFORCED.** The only `flock` inside the harness is around the LLK driver (`flock /tmp/tt-device.lock`, line 1249). The op-level sweep relies on the campaign convention that *every* device run is wrapped externally in `flock /tmp/tt-device.lock` (SORTING.md:647; `scripts/run_safe_pytest.sh:53` uses the same lock file; the archived `competition_chain.sh` flocked its pytest gate phases). **Gate-5 rule: always launch the sweep itself under `flock /tmp/tt-device.lock`** — the invocations in §1.4 do.
8. **Determinism + resume semantics.** Pinned per-layer seeds (272–275, 341–344); `--resume` skips MEASURED/UNSUPPORTED/WRONG, retries FAILED only (1646–1654).
9. **Report the harness with the number.** Per audit finding STRA-3 debate (RADIX_BUCKET_GPU.md:815): "Any Gate-5 target must name the harness, not just the number: canonical sweep, device flock, .so-mtime stamping per cell, and the replay-STORE arm state — otherwise the refreshed baseline drifts as fast as the stale one did." Also §6.1 item 5 (RADIX_BUCKET_GPU.md:620–623): the same cell measured 2.3× apart on the same day under different harnesses.

Current tree state relevant to pinning: header is clean (STORE default-ON, `ckernel_sfpu_topk.h:73–78`, no marker block); working tree carries dirt in `.github/workflows/package-and-release.yaml` and `RADIX_BUCKET_GPU.md` (the §6 audit appendix is uncommitted) — neither touches the measured path, but the `tree_diff_md5` of any new run will reflect it and must be recorded.

---

## 4. The missing §5.1 baseline cell

**What the audit says** (RADIX_BUCKET_GPU.md:620–622, also 831, 906): "the §5.1 cell itself (BF16, K≤64, long rows) has no measured baseline at all (archived stock-multicore point: ~171 µs at N=32k/K=32)"; §6.2 corrections queue row 6 (line 660): "Measure the actual §5.1 cell baseline before Gate-2 work; … pin the harness (canonical sweep, .so-mtime stamping, replay-STORE arm state)."

**What actually exists**: more than the audit credits — ab3 (§2.2 table) covers N∈{4096,8192,32768,65536,131072} × k∈{8,32,64} bf16 — but it fails Gate-1 pinning on four counts:
1. It lives in an **ephemeral /tmp session scratchpad**, not committed anywhere (the ledger only carries large-K competition rows; its A/B `details` table shows 8 representative rows).
2. Its "baseline" arm is the **pre-branch header**; today's stock is the `replay_store` column, measured under the retired 3-arm naming — a rerun with today's `--arms baseline` is the clean statement.
3. Grid gaps inside the §5.1 window: **N=16384 (multi-core) and N=65534 (the W<65535 boundary) and k=16 were never measured**.
4. It predates the values-native/tree-merge `.so` (K≤64 stock path is host-side unchanged, but the pinning discipline says re-measure and stamp rather than argue).

**The command that closes the gap** (do NOT run in this session — device-using):

```bash
cd /home/nachiket/tt-metal && source python_env/bin/activate && \
flock /tmp/tt-device.lock \
python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
    --arms baseline --layers ttnn --ops topk \
    --ns 8192,16384,32768,65534,65536 --ks 8,16,32,64 --dtypes bf16 \
    --trials 3 --warmup 3 --timeout 900 \
    --out generated/canonical_sweep/scope51_baseline
```

~20 cells × 3 trials; per-trial cost dominated by the N=65536 single-core cells (~10–18 ms/iter × 13 calls ≈ negligible; wall time is Tracy/JIT overhead, expect well under an hour). Expected shape of the result, from ab3: **~107 µs (N=8192) → ~171 µs (N=32768) multi-core for k≤32, ~239–309 µs at k=64; 9.5–18 ms single-core at N=65534/65536**. Any threshold-select Gate-5 claim in the §5.1 scope must beat the multi-core ~107–309 µs cells (not the single-core cliff) beyond 2·pooled_std, and per §5.3 the SFPU streaming arm additionally needs ≥10% repeatable complete-pass improvement.

**Recommended follow-up once measured**: commit the `scope51_baseline` CSV (or splice a K≤64 region into TOPK_LEDGER.html via a new marker) so the §5.1 baseline stops living in /tmp.

---

## 5. One-line context for the campaign

Per §5.2 as committed, **Gate 2 = candidate materialization given a known threshold** is the load-bearing go/no-go (the §6 audit's "Gate 4" refers to the pre-renumbering draft — same gate); the exact SFPU arm is threshold bisection (1 bit @ 2.0 cyc/vec CountD1, 3 bits @ 3.0 cyc/vec HistMacro+HistSum, ≥25.1-cyc rendezvous per data-dependent decision), not multi-bin radix; the strongest unmeasured alternative is the dual-RISC BF16 two-byte-digit histogram selector on BRISC/NCRISC (§5 verdict table; Gate 3 shootout arms (c)/(d)). Gate 1's deliverable — this document's §1.4 commands plus §3 checklist plus the §2 pinned numbers — is the precondition for all of it.
