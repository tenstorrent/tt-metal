# PLAN.md — Autonomous op-optimization of GLM-4.7-Flash on Wormhole LoudBox (T3K, 2×4)

> **You are an optimization agent. Work non-stop through this plan until the Definition of
> Done holds. Do not ask for permission between steps — follow the loop, log every iteration,
> and keep going. Every command below is copy-pasteable. Read "Invariants" once; they never
> change. This doc is the single source of truth.**

Model: `models/experimental/glm4_moe_lite`. Hardware: Wormhole **LoudBox / T3K**, 8 chips,
mesh **2×4** (`--mesh-rows 2 --mesh-cols 4`). Prior bring-up + fixes: see
`agent_logs/2026-07-01-wh-loudbox-bringup.md`. Baseline sweep (batch 1): see
`experiments/t3k_isl_sweep_b1/`.

---

## 0. Definition of Done (the goal condition)

Optimize **decode and prefill** for **batch ∈ {1, 8, 16, 32}** across **all ISLs**
{128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072}. There is **no latency
or throughput SLA** — so the target is **utilization**, not absolute ms:

**DONE when, for every (batch, phase) config profiled with tt-perf-report:**
1. **No `SLOW` matmuls** — every Matmul/SparseMatmul row has `Bound ∈ {FLOP, DRAM, BOTH}`
   (i.e. ≥65% FLOPs- or DRAM-utilization). SLOW (yellow) = under-utilized = must be fixed.
2. **No `HOST` ops** on the hot path (no `(torch)` fallbacks in decode/prefill).
3. **TM + DM category time minimized** — the `TM` (Tensor-Manipulation: Reshape/Transpose/
   Permute/Concat/Tilize/Untilize/Typecast/CreateHeads/ConcatHeads) and unnecessary `DM`
   (Reshard/Move/InterleavedToSharded round-trips) share of total device time is driven as low
   as practical. **Target: TM < 5% of the phase's device time; each individual TM/DM op either
   fused away or justified in the log.**
4. Every change is **correctness-verified** (greedy tokens match baseline / PCC test passes).

Start with **batch 1** (baseline exists). Then 8, 16, 32.

---

## 1. Invariants (NEVER violate — these are hard-won)

- Mesh is **2×4** (8 chips). Never a grid wider than **8** (WH Tensix grid is 8×8).
- **`GLM4_MOE_LITE_CCL_NUM_LINKS=1`** always. T3K has 1 CCL link/axis; `2` deadlocks (silent
  ~forever hang). `GLM4_MOE_LITE_CCL_TOPOLOGY=linear`.
- **`GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0`** always on WH. `=1` selects Blackhole-only 10×4
  grids → `TT_FATAL grid (10,4) must fit (8,8)`.
- **`TT_METAL_GTEST_ETH_DISPATCH=1`** always.
- **batch > 1 prefill MUST use `GLM4_MOE_LITE_BATCHED_PREFILL=0`.** `=1` (the batched-prefill path)
  crashes at batch 8/16/32 with an L1 circular-buffer clash (`program.cpp:1549`). Only batch 1 may
  use `=1`. (Verified: 8/16/32 all run cleanly with `=0`.)
- After **any** `kill -9` of a mesh process, run **`tt-smi -r`** before the next run (SIGKILL
  skips `close_mesh_device` → orphaned fabric → next `open_mesh_device` hangs at topology
  discovery). Prefer `timeout` (SIGTERM) so Python cleanup runs.
- Judge hang-vs-progress by **cache-file growth** (`~/.cache/tt-metal-cache/*.o`), NOT CPU. A
  busy-spin CCL hang shows ~100% CPU with a frozen cache.
- The TT weight cache is **mesh-shape specific**; changing mesh forces a slow regen.
- **Correctness before speed.** A wrong config can give silently-wrong logits (garbage tokens),
  not a clean error. Always verify tokens after a change.

### Standard env prefix (paste before every run)
```bash
cd /home/gtobar/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
export TT_METAL_GTEST_ETH_DISPATCH=1 \
       GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 \
       GLM4_MOE_LITE_CCL_NUM_LINKS=1 \
       GLM4_MOE_LITE_CCL_TOPOLOGY=linear
```

---

## 2. Tools (what each is for — validated commands in §3)

| Tool | Path | Purpose |
|---|---|---|
| **Run** | `scripts/debug_run_full_tt_greedy.py` | the actual model run (prefill+decode); prints prefill_s / decode ms |
| **Throughput sweep** | `scripts/run_sweep_isl_batch.py` | ISL×batch → prefill/tok-s/TTFT/decode table (regression baseline) |
| **Matmul micro-sweep** | `sweeps/run_matmul_sweep.py` | brute-force one matmul's program/mem/fidelity/dtype, reads device kernel **ns** in-process |
| **Tracy** | `python -m tracy` | per-op device profile of a real run → `ops_perf_results_*.csv` |
| **tt-perf-report** | `tt-perf-report` (in venv) | analyze the CSV → per-op **Bound** (FLOP/DRAM/SLOW/HOST) + **Category** (Compute/DM/TM) + stacked report + advice |

**tt-perf-report semantics (the guidance):**
- Matmul `Bound`: `DRAM%≥65 → DRAM` (bandwidth-bound, good), `FLOPs%≥65 → FLOP` (compute-bound,
  good), both → `BOTH`, else → `SLOW` (bad, fix it), `(torch)` → `HOST` (bad).
- Categories: **Compute** (Matmul, SparseMatmul, SDPA, LayerNorm, Reduce, Softmax, eltwise),
  **DM** (InterleavedToSharded, ShardedToInterleaved, Reshard, Move, Copy, ReduceScatter,
  PagedUpdateCache), **TM** (Reshape, Transpose, Permute, Slice, Concat, Tilize/Untilize,
  **Typecast**, CreateHeads/ConcatHeads).
- Use `--group-by category` for the compute/DM/TM stacked breakdown; `--start-signpost` /
  `--end-signpost` to isolate one region (e.g. a single decode step).

---

## 3. The optimization loop (run this for each (batch, phase))

Repeat until §0 holds for the config. Log every iteration to `agent_logs/opt_progress.md`.

### Step A — PROFILE (Tracy → CSV)
Eager run (no `--enable-trace`) so each op is an individual CSV row with clean device time.
Signposts on so we can isolate one decode step.
```bash
GLM4_MOE_LITE_SIGNPOST=1 \
GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 GLM4_MOE_LITE_FUSE_QKV_A=1 GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 \
GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1 GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 GLM4_MOE_LITE_SKIP_TYPECAST=1 \
python -m tracy -r -o /tmp/glm_prof/b<BATCH>_isl<ISL> \
  models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Summarize" --simulate-context-len <ISL> --min-cache-tokens $((<ISL>+64)) \
  --max-new-tokens 4 --batch-size <BATCH> \
  --mesh-rows 2 --mesh-cols 4 --kv-cache-dtype <bf16|bf8> --phase both
```
(batch≥32 → `--kv-cache-dtype bf8`; batch>1 → also set `GLM4_MOE_LITE_BATCHED_PREFILL=0`.)
CSV lands at `<tracy_out>/.../ops_perf_results_<ts>.csv` (the run prints the exact path;
`--tt-perf-report-command-exact-path-here` is filled in §7 after first validation).

### Step B — ANALYZE (tt-perf-report)
```bash
# whole run, per-op table + advice:
tt-perf-report <CSV> --arch wormhole
# one decode step only:
tt-perf-report <CSV> --arch wormhole --start-signpost decode-start --end-signpost decode-end
# category breakdown (Compute vs DM vs TM):
tt-perf-report <CSV> --arch wormhole --group-by category --summary-file /tmp/glm_prof/summary_b<BATCH>.csv
```
From the report, build the **worklist**, sorted by `Total %` (biggest time first):
- (a) matmuls with `Bound = SLOW`
- (b) TM ops (and redundant DM reshards) with high `Total %`
- (c) any `HOST` op on the hot path

### Step C — FIX (highest `Total %` first)
- **SLOW matmul** → find a FLOP/DRAM-bound config with the micro-sweep (§4), then wire the
  winner into the matmul's call site in `tt/linear_helpers.py` (behind an arch check — WH only).
  Also consider math fidelity/dtype: if not FLOP-bound, tt-perf-report advice suggests fidelity;
  BF4/BF8 weights lower DRAM pressure.
- **TM op (Transpose/Reshape/Typecast/CreateHeads/ConcatHeads/Concat)** → fuse or skip. Existing
  levers: `GLM4_MOE_LITE_FUSE_QKV_A`, `FUSE_SHARED_GATE_UP`, `FUSE_EXPERTS_GATE_UP`,
  `SKIP_TYPECAST`, `CONCAT_HEADS`/`NLP_CONCAT_HEADS`, `DECODE_Q_DIRECT_RESHAPE`,
  `DECODE_FUSE_QA_NORM`, `SHARDED_DECODE_NORM`. If no flag exists, do a code change in
  `tt/decoder_layer_tt.py` / `tt/attention_decode.py` to keep the tensor in its native layout
  (avoid interleaved↔sharded round-trips → those are the `DM` reshards).
- **HOST op** → move the computation on-device (find the `(torch)` fallback and replace).

### Step D — VERIFY (correctness, every time)
```bash
# same run WITHOUT profiling; confirm greedy tokens unchanged vs baseline
python models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Summarize" --simulate-context-len <ISL> --min-cache-tokens $((<ISL>+64)) \
  --max-new-tokens 8 --batch-size <BATCH> --mesh-rows 2 --mesh-cols 4 \
  --kv-cache-dtype <bf16|bf8> --phase both
```
Also run the relevant PCC test in `tests/*optional*` if the op has one. If tokens change or PCC
drops → revert the change. **Never keep a faster-but-wrong config.**

### Step E — LOG & REPEAT
Append to `agent_logs/opt_progress.md`: `(batch, isl, phase) | op | change | Bound before→after |
device_time before→after | TM% before→after | correctness PASS/FAIL`. Then go back to Step A and
re-profile. Stop the config when §0 holds; move to the next (batch, phase).

---

## 4. Matmul micro-sweep (Step C helper)

To find a FLOP/DRAM-bound config for a specific matmul without a full model run:
```bash
python models/experimental/glm4_moe_lite/sweeps/run_matmul_sweep.py \
  --mesh-rows 2 --mesh-cols 4 --phase <decode|prefill> --batch <BATCH> --targets <name>
```
Targets (verified per-device shapes, TP=4) live in `sweeps/targets.py`:
`w_q_a, w_kv_a, w_q_kv_a_fused, w_q_b_attndp, w_q_b_headpar, w_o(+all_reduce),
w_shared_gate, w_shared_up, w_shared_gate_up, w_shared_down(+all_reduce),
w_dense_gate/up/down (layer 0), w_router`. Phase-2 (sparse experts, head-parallel kv_b1/b2) are
stubs — implement them when their ops show up hot in the report.
Widen the grid by editing `SweepAxis` in `targets.py` (dtypes, fidelity, mem, prog configs).
The harness reports device kernel **ns** per config (min over iters, max over chips); pick the
config whose shape/layout the report says would be FLOP/DRAM-bound, then hard-code it in
`linear_helpers.py`.

---

## 5. Batch / ISL execution order (do NOT brute-force all 44 cells blindly)

Key insight — profile the *minimum* set, because op shapes repeat:
- **Decode** op shapes depend only on **batch** (M = batch, tile-padded to 32), NOT on ISL. So
  profile decode **once per batch** (at ISL 128 for speed); the per-op config carries to all ISLs.
  (KV-length-dependent ops — FlashMLA decode, PagedUpdateCache — scale with ISL; spot-check those
  at ISL 32768.)
- **Prefill** uses **chunked** prefill (`MAX_PREFILL_CHUNK_SIZE=128`), so per-matmul M is fixed at
  the chunk (128) regardless of ISL. Profile prefill **once per batch** at ISL 128; validate the
  chunk matmuls don't regress at long ISL.
- Therefore the real work matrix is **4 batches × {decode, prefill} = 8 profiling configs**, not 44.

Order: **batch 1 decode → batch 1 prefill → 8 → 16 → 32.** After each batch is "green", run the
full throughput sweep (§6) as a regression gate before moving on.

---

## 6. Regression gate (after each batch is optimized)
```bash
python models/experimental/glm4_moe_lite/scripts/run_sweep_isl_batch.py \
  --out-dir models/experimental/glm4_moe_lite/experiments/t3k_isl_sweep_b<BATCH> \
  --mesh-rows 2 --mesh-cols 4 --timeout 3600 \
  --isl 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 --batch <BATCH>
```
(The script now respects the env overrides in the standard prefix; it also runs with `--warmup`,
so Prefill/TTFT are clean.) Compare the new table to the pre-optimization one; decode ms must not
regress and should improve where SLOW ops were fixed.

---

## 7. Validated tool commands & baseline report (batch-1 prefill, ISL 128)

### 7.0 REQUIRED patch to make Tracy work on this multi-device model
`python -m tracy -r` asserts and dies during host↔device op merge on multi-device meshes
(`AssertionError: Device data missing: Op <id> not present in cpp_device_perf_report.csv for
device N`) — some ops legitimately record no device-perf row on some chips. **Fix applied** in
`tools/tracy/process_ops_logs.py` (~L560): the `assert candidates` was changed to a
warn-and-`continue` (skip that op on that device; it still appears from the other chips). Without
this patch NO ops_perf_results.csv is produced. Keep this patch.

### 7.1 Validated Tracy command (produces the CSV)
Profiling **prefill** works cleanly. (Decode at batch 1 emits many skip-warnings due to sparse MoE
coverage but still completes with the patch.)
```bash
tt-smi -r > /tmp/reset.log 2>&1    # if device not fresh
<STANDARD ENV PREFIX from §1> \
GLM4_MOE_LITE_SIGNPOST=1 GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1 \
GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 GLM4_MOE_LITE_SKIP_TYPECAST=1 \
python -m tracy -r -o /tmp/glm_prof/<tag> \
  models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Summarize" --simulate-context-len 128 --min-cache-tokens 256 \
  --max-new-tokens 1 --batch-size 1 --mesh-rows 2 --mesh-cols 4 --kv-cache-dtype bf16 --phase prefill
```
CSV lands at `/tmp/glm_prof/<tag>/reports/<TS>/ops_perf_results_<TS>.csv`. First run recompiles
all kernels with `-DPROFILE_KERNEL=1` (~10 min); cached afterward (~2-3 min).

### 7.2 Validated tt-perf-report commands
```bash
csv=$(find /tmp/glm_prof/<tag> -name "ops_perf_results_*.csv" | head -1)
tt-perf-report "$csv" --arch wormhole --group-by category   # Compute/DM/TM stacked split
tt-perf-report "$csv" --arch wormhole --group-by op         # per-op-type time + FLOPs%
tt-perf-report "$csv" --arch wormhole | grep -iE "SLOW"     # under-utilized matmuls (the worklist)
tt-perf-report "$csv" --arch wormhole --start-signpost decode-start --end-signpost decode-end  # one decode step
```

### 7.3 BASELINE stacked report (batch-1 prefill, ISL 128) — the starting worklist
Category split of device time:
```
66.5%  Compute   |  16.6%  TM  |  10.6%  Other(routing/CCL/reshape)  |  6.3%  DM
```
Top op types by device time:
```
17.8% SparseMatmul(MoE)  14.9% Unary  14.9% Matmul(FLOPs~24% mean → SLOW)  10.0% BinaryNg
 6.0% Concat[TM]  5.3% AllGather[CCL]  4.9% ReduceScatter[DM]  4.7% LayerNorm
 4.4% Slice[TM, 1581 ops!]  2.6% SDPA  2.1% FillPad[TM]  1.6% Gather  1.6% TopK  1.4% FastReduceNC
```
**Finding: every dense matmul is `SLOW` (5–33% util) at prefill M=128** — under-utilized, not
compute/BW bound. Shapes → GLM weights (all `LoFi`, DRAM%/FLOPs%):
| shape (K×N/dev) | weight | DRAM% | FLOPs% |
|---|---|---|---|
| 2048×1344 | fused q_kv_a | 19 | 19 |
| 2048×768 | q_a | 6 | 20 |
| 768×1280 | q_b (headpar) | 9 | 18 |
| b5×192×512 | kv_b1 | 7 | 8 |
| b5×512×256 | kv_b2 | 4 | 11 |
| 1280×2048 | w_o | 27 | 17 |
| 2048×5120 | q_b (attndp) | 33 | 34 |
| 2560×2048 | dense down (L0) | 15 | 19 |
| 384×2048 | shared down | 12 | 15 |
| 2048×64 | router | 1 | 8 (tiny, expected) |
| 2048×38720 | lm_head | SLOW | |

### 7.4 First worklist actions (highest-leverage, by category)
1. **TM 16.6% + Slice(1581)/Concat(187)/FillPad** — the largest reducible overhead. Investigate
   why 1581 Slice ops and 187 Concat: likely per-layer head-splitting / KV assembly / MoE gather.
   Fuse or eliminate (CONCAT_HEADS, NLP_CONCAT_HEADS, DECODE_Q_DIRECT_RESHAPE; code-level: keep
   native layouts). Target: TM < 5%.
2. **SLOW matmuls** — raise utilization at M=128 via program-config tuning (grid/subblock/
   in0_block_w) using `sweeps/run_matmul_sweep.py`; consider a larger prefill chunk (M) if it
   helps utilization without OOM. Start with the biggest-time ones: q_kv_a, w_o, q_b, dense-down.
3. **Other 10.6%** (AllGather/ReduceScatter/TopK/Gather/Scatter/MeshPartition) — MoE routing +
   CCL. AllGather 5.3% + ReduceScatter 4.9% are CCL; check num_links/topology are optimal (1 link
   is forced by HW, so focus on reducing collective count/size, e.g. FUSE_MLP_MOE_REDUCE).
4. **lm_head SLOW (2048×38720)** — huge N; sharded across 4 cols. Tune or accept (it's one op/step).

---

## 8. Bookkeeping & status
- Per-iteration log: `agent_logs/opt_progress.md` (create if missing; append every Step E).
- Baseline throughput (batch 1): `experiments/t3k_isl_sweep_b1/sweep_table.md`.
- When a config reaches §0, note it in `opt_progress.md` as `CONFIG (batch,phase) = DONE`.
- The goal condition auto-clears when all four batches × both phases satisfy §0.

## 9. Guardrails / do-not
- Do NOT set `num_links=2`, do NOT enable `PREFILL_MATMUL_TUNED` on WH, do NOT change the mesh
  shape, do NOT keep a change that fails correctness.
- Do NOT trust CPU% for liveness; use cache growth.
- Do NOT profile with trace for op-level work (use eager); trace hides per-op rows behind one
  replay program. Trace is only for end-to-end latency numbers (§6).
- If a run wedges: `pkill -9 -f debug_run_full_tt_greedy; tt-smi -r`, then continue.
