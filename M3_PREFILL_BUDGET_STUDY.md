# MiniMax-M3 prefill: forward-budget and KV-depth study on one Blackhole galaxy

**Audience:** an engineering agent with shell access to one Blackhole galaxy (8×4) and the tt-metal repo.
**Owner:** Slava Melnykov (MiniMax-M3 prefill).
**Goal:** collect the data needed to decide how a prefill forward should be filled, and so to maximise prefill throughput.

Read the whole document before running anything. Work in priority order (P0 → P1 → P2). Keep `results/PROGRESS.md` up to date as you go.

---

## 0. What you are asked to do

0. **Find the code (§4.0).** The packed-prefill experiment code may never have been pushed. Recover it if possible. If not, run Phase A on `main` with the in-tree tools and rebuild the packed path for Phase B.
1. **Set up and discover (§4).** Verify the harness knobs that exist, and add the few default-off knobs listed in §4.3 that are missing. Run the sanity check.
2. **Run the experiments (§6).** Record every run as one row in `results/runs.csv`, and per-op profiles in `results/ops.csv`. Use the schema in §5.
3. **Analyse (§7).** Fit the per-layer cost model, then write `results/coeffs.json`. Run the packing simulator in Appendix A with it, and produce the plots.
4. **Report (§8).** Write `results/REPORT.md` using the template in §8. Every question in §1.3 must have an answer, even if the answer is "not measurable, because…".

Stop and write to the human (in `PROGRESS.md` and your final message) if:

- 3 runs in a row hang;
- a required knob needs a kernel or C++ change;
- the §4.4 sanity numbers are off by more than 25%;
- the additivity check in E4 fails by more than 20%.

---

## 1. Context

### 1.1 Model and target deployment

- **Model: MiniMax-M3.** 60 layers.
  - Layers 0–2 are **dense**: full attention over the whole history, via `ring_joint` SDPA.
  - Layers 3–59 are **sparse** (MSA). A lightning indexer (`indexer_score_msa`) scores 128-token history blocks, top-16 blocks are selected, then `sparse_sdpa_msa` runs over them.
  - FFN is MoE (bf4 experts, EP).
- **Target prefill deployment:** 4 galaxies run as **one 8-stage pipeline**, 2 stages per galaxy.
  - Each stage is a **(4,4) sub-mesh, SP=4, TP=4, EP=16**, holding 7–8 layers.
  - Default split: `8,8,8,8,7,7,7,7`. Stage 0 = layers 0–7 = 3 dense + 5 sparse.
- **Target decode:** 16 galaxies × 64 users. Requests with few new tokens *and* a short history are appended on decode. Everything else, including small requests on a very deep history, goes to prefill. So prefill sees many **hot requests**: about 1–6k new tokens on a 60k–550k history.
- **You have one galaxy.** It contains two (4,4) sub-meshes, i.e. exactly **two stages of the target layout**.
  - We measure each stage type in isolation, like the previous pipeline study.
  - Optionally, we run a 2-stage mini pipeline (E7).
  - Cross-galaxy hops cannot be measured here; they become a parameter in the simulator.

### 1.2 What is already known

These numbers come from branch `vmelnykov/batched_prefill_experiment`. They are untraced, bf4 experts, cold cache.

| Layout | Tokens/forward | Slowest-stage ms/forward | Note |
|---|---|---|---|
| 2 stages of (4,4), SP=4, 30 layers/stage | 2048 / 5120 / 8192 / 10240 | 178 / 344 / 524 / 640 | ≈ 5.9 / 11.5 / 17.5 / 21 ms per sparse layer |
| 4 stages of (2,4), SP=2, 15 layers/stage | 2048 / 5120 / 8192 / 10240 | 143 / 277 / 430 / 517 | |
| 4 stages, packed 4×2048 vs plain 8192 | 8192 | 438 vs 430 | packing is ~free on sub-meshes |
| whole galaxy, packed 4×2k | 8192 | 1283 (vs 761 plain) | host-launch bound: 5.4× more ops |

The design facts that follow from these runs:

- **Packing works.** Several requests go into one forward as equal 2048-token segments. Token-parallel ops (norms, projections, MoE) run once on the packed tensor. Position- and slot-aware ops (RoPE, KV write, SP gathers, indexer, top-k, SDPA) run **per segment, in a loop**.
- **All segments are 2048 tokens today.** The kernels infer the KV block-cyclic period from the tensor shape. Requests are rounded up to 2048, and `n_real` carries the true length.
- **`cached_len` must be a multiple of 2048.**
- **CCL bug, likely fixed on `main`.** The SP token-id all-gather in the embedding hung on sub-meshes and on wide forwards. Root cause: `all_broadcast` sized row-major packets as payload + header. A uint32 row over 4352 B (over 1088 tokens per chip) overflowed a fabric slot. PR **#57199** (merged 2026-09-23, commit `718e1f2`) fixes it.
  - Check `git merge-base --is-ancestor 718e1f2 HEAD`. If true, wide forwards should run without workarounds; confirm with one 8192 run on (4,4).
  - If it's not in your tree, or a wide run still hangs, set `M3_REPLICATED_TOKENS=1` (only if the experiment harness is recovered) and note it.

### 1.3 Questions this study must answer

| # | Question | Main experiments |
|---|---|---|
| Q1 | Per-stage cost vs forward width, cold: floor, slope, knee, for dense and sparse layers on (4,4) SP=4 | E1 |
| Q2 | How cost grows with history depth, and which ops cause it. Split into terms ∝ new×history (compute) and ∝ history only (e.g. the `index_k` gather) | E2, E3 |
| Q3 | Is a packed forward additive, i.e. shared(W) + Σ per-segment attention? Does mixing depths in one forward cost extra? | E4 |
| Q4 | How should requests be grouped by history (fcfs / depth-bucketed / cost-balanced), and should the budget be in tokens or in ms? Which width W? | E4 + simulator |
| Q5 | Does stage 0 (the 3 dense layers) become the pipeline bottleneck at deep history? Which layer split balances an AgentX-like mix? | E2 + simulator |
| Q6 | Do padded rows inside a segment cost compute (is the 2048 rounding waste real)? | E5 |
| Q7 | (P2) SP=4 (4,4) vs SP=2 (2,4) per-chip efficiency at depth; 2-stage mini-pipeline hop cost | E6, E7 |

A prior estimate to confirm or kill in Q5: from FLOPs, one dense layer at 549k history costs about 13× a sparse layer per new token. That would make stage 0 about 5× heavier than an 8-sparse-layer stage for deep hot segments. On the whole galaxy this was hidden under the ~500 ms host floor; the sub-mesh floors are small, so it should show here.

---

## 2. Vocabulary used below

- **W**: forward width in tokens = B × 2048.
- **B**: number of segments in a forward.
- **Segment**: one request's chunk inside a forward: `(slot, cached_len = h, n_real = n)`.
- **h**: history, the tokens already in that slot's KV cache before this segment.
- **n**: real new tokens in the segment (≤ 2048); the rest of the segment is padding.
- **Layer sets:**
  - **D** = layers 0–2 (3 dense).
  - **S8** = layers 8–15 (8 sparse; target stage 1).
  - **S0** = layers 0–7 (target stage 0: 3 dense + 5 sparse).
- **History grid H:**
  - Main points: `0, 16384, 65536, 141312, 309248, 548864`. All are multiples of 2048; 141312 ≈ median and 548864 ≈ p90 of the AgentX trace history.
  - Optional: `1040384`, with `MAX_SEQ_LEN=1048576`.
- **n grid:** `256, 1024, 2048`.

---

## 3. Rules of engagement

- **Branch.** Work on a new branch `<you>/m3_prefill_budget_study`, created from `vmelnykov/batched_prefill_experiment` (or from `main` if it has been merged; say which in the report). Commit knobs, scripts and results as you go, and push the branch at the end. Never push to `main`.
- **Code changes.** Do **not** change kernels, C++ ops or model math. You may add **default-off** env knobs to Python harness and test files, plus analysis scripts. Defaults must reproduce today's behaviour exactly.
- **Machine.** Before starting, check that nobody else is using the galaxy (`tt-smi`, running processes). Run `tt-smi -glx_reset` before every new process or config.
- **Timeouts.** Give each run 20 min including weight load. If there is no log progress for 5 min after load, kill it, reset, record `status=HANG`, and continue with the next run.
- **Data.** Append-only. Never overwrite `runs.csv`; failed runs are rows too.
- **Environment.** Record the git SHA, full env, and `tt-smi` state for each run in `results/logs/<run_id>.env`.
- **Weights and cache.** Checkpoint and tilized caches have moved to weka (PR #57617):
  - checkpoint: `/mnt/weka/model-weights/llm/minimax/MiniMax-M3`
  - caches: `/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/tensor_cache_bfp8_{MeshShape}`, with `[8,4]`, `[4,4]` and `[2,4]`, plus `golden/`.

  If this host has no weka, the old NFS cache (`/data/zbaczewski/m3_pp_cache`) is very slow: over 25 min for 30 layers. Pre-warm it. Set `TT_CACHE_PATH` explicitly and record it.

---

## 4. Setup and discovery

### 4.0 Find the experiment code first; fall back if it's gone

The packed-prefill work (`segment_size`, `prefill_segments`, the `galaxy_prefill_segmented.py` harness, `run_matrix_v2.sh`, `prefill_experiments/`) lived on the local branch `vmelnykov/batched_prefill_experiment`. It may never have been pushed. The owner will share two docs that describe it exactly: `BATCHED_PREFILL_EXPERIMENT.md` and `PREFILL_THROUGHPUT_CONCLUSIONS.md`. Treat them as the spec.

**Step 1: look for it.** Do this on the host the owner names (the original work tree was under `/data/vmelnykov/tt-metal`):

```bash
git -C /data/vmelnykov/tt-metal status --short | head -50
git -C /data/vmelnykov/tt-metal branch -a --list '*batched*'
git -C /data/vmelnykov/tt-metal stash list
git -C /data/vmelnykov/tt-metal log --all --oneline -- '*galaxy_prefill_segmented*' | head
find /data /home -xdev \( -name 'galaxy_prefill_segmented.py' -o -name 'run_matrix_v2.sh' -o -type d -name 'prefill_experiments' \) 2>/dev/null
# files that were ever `git add`-ed survive as dangling blobs even if the work tree is gone:
git -C /data/vmelnykov/tt-metal fsck --lost-found && grep -l 'prefill_segments' /data/vmelnykov/tt-metal/.git/lost-found/other/* 2>/dev/null
# if the work was done with Claude Code, its transcripts contain every file it wrote:
grep -rl 'galaxy_prefill_segmented' ~/.claude/projects 2>/dev/null | head
```

If you find it, immediately commit to a backup branch and push. Leave out logs, profiles and large files:

```bash
git switch -c vmelnykov/batched_prefill_experiment_backup
git add models/demos/minimax_m3 prefill_experiments/*.sh prefill_experiments/*.py prefill_experiments/*.md
git commit -m "WIP backup: M3 batched (segmented) prefill experiment"
git push -u origin HEAD
```

Then rebase it on current `main`. Expect conflicts in `tt/attention/prefill.py`, `tt/topk.py` and the MoE files, because main has moved: high_bw gathers #55668, MoE combine #55490, new MoE ops.

**Step 2: if it can't be found, split the study.**

- **Phase A (on `main`, no packing needed): E1, E2, E3, E7.** Use the in-tree tools in §4.2. These measure the cost of one request per forward vs width and vs history depth. That is most of the cost model, and it answers Q1, Q2 and Q5.
- **Phase B (needs the packed path): E4, E5, E8.** Rebuild the segmented forward from `BATCHED_PREFILL_EXPERIMENT.md` §1–2. It is Python-only:
  - `TtPrefillRuntimeConfig.segment_size`;
  - `prefill_segments(input, kv_cache, [(slot, cached_len, n_real)])`;
  - the per-segment `_attention_core` loop in `tt/attention/prefill.py` (RoPE, KV write, gathers, indexer, top-k, SDPA per segment; everything else once on the packed tensor);
  - `TopKRouter.build_padding_config` taking a per-chip real-row count;
  - `compile()` warm-up of the used `cached_len` buckets.

  Reproduce §3 of that doc first: W1 is 4 × 2k packed vs sequential, expected ≈1.7× on the whole galaxy. That is the correctness and perf gate before E4. Rebuilding is allowed model-code work here, but it must not change the default (`segment_size` unset) path.

### 4.1 Environment

Previous runs used this setup. Verify each item still applies:

```bash
cd $TT_METAL_HOME && source python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME
export TT_MESH_GRAPH_DESC_PATH=<...>/single_bh_galaxy_mesh_graph_descriptor.textproto
tt-smi -glx_reset
```

### 4.2 Existing harness to read first

Knob names can drift. Open each file and confirm what exists **before** you plan runs.

**Tools on `main`, usable for Phase A:**

- **`models/demos/minimax_m3/scripts/run_prefill_profile.sh`** with **`tests/perf/profile_prefill.py`**: zone profiler.
  - `STAGES=1|2|4 STAGE=k` carves stage k's (8,4), (4,4) or (2,4) sub-mesh from its own tilized cache (PR #55654).
  - `LAYER_IDS=…` builds explicit layers, e.g. `0,1,2` for D, `8,…,15` for S8, `0,…,7` for S0.
  - `CHUNK=W` sets the forward width.
  - `CACHE=h` sets the history depth.
  - `SKIP_PREFIX=1` skips building the history. **Check what the KV then contains:** zeros make indexer scores degenerate, so prefer random data or validate against a real prefix.
  - `FABRIC=1d` (must be set) and `M3_FABRIC`.
  - It reports per-zone device time, and the chunk time (e.g. #57199 measured about 26 ms for layers 0+3, chunk 5120, on 4x4).
- **`tests/galaxy_prefill_kv_pcc.py`** and **`scripts/run_prefill_perf.sh`**: throughput and KV-PCC harness. Env vars are documented in `models/demos/minimax_m3/README.md`.
- **`models/demos/common/prefill/runners/run_pipeline_prefill.sh`** with `tt/runners/manifests/m3_binding_mock_migration_intragalaxy_2rank.yaml` (2 × [4,4], 30 layers per stage, trace on): a real 2-stage pipeline on one galaxy, for E7.

**Tools only on the (possibly lost) experiment branch:**

- **`models/demos/minimax_m3/tests/galaxy_prefill_segmented.py`**: packed prefill harness. Previously known env vars:
  - `REQUESTS`, `SEGMENT_SIZE` (0 = baseline runtime), `PREFILL_CHUNK_SIZE` (= W), `PACK_POLICY=breadth|depth`, `MAX_SEQ_LEN`;
  - `STAGES`, `STAGE` (carve a sub-mesh and run that stage's layers);
  - `DUMP_DIR`, `COMPARE_DIR`, `GOLDEN_PREFIX`, `PREFILL_TRACE_DIR`;
  - `M3_REPLICATED_TOKENS`, `TT_CACHE_PATH`.
- **`models/demos/minimax_m3/tests/perf/profile_prefill.py`** and **`scripts/run_prefill_profile.sh`**: zone profiler. Previously known: `CHUNK`, `CACHE`, `LAYER_IDS`, `PROFILE_SEGMENT`.
- **`parse_zone_perf.py`**: per-zone device time on the worst chip.
- **`prefill_experiments/run_matrix_v2.sh`** and **`plot_matrix.py`**: the previous matrix driver. Reuse its structure: it already tells a load timeout apart from a hang.
- **`tt/tt_prefill_runtime.py`**: `segment_size`, `prefill_segments(input, kv_cache, [(slot, cached_len, n_real)])`, and `compile()`, which warms `cached_len` buckets.

### 4.3 Knobs to add if missing

All knobs are default-off.

| Knob | Behaviour |
|---|---|
| `LAYER_RANGE=a:b` + `SUBMESH=4x4` | Run layers `[a, b)` on one (4,4) sub-mesh with SP=4, TP=4, EP=16, independent of `STAGES/STAGE`. If `STAGES/STAGE` already allows arbitrary ranges, use that and document it. |
| `SEGMENTS=h1:n1,h2:n2,...` | Explicit segment list for one packed forward. Each entry is one slot with `cached_len = h` (a multiple of 2048) and `n_real = n`. The list order is the order of the attention loop. `h1:n1+h2:n2` (joined with `+`) means consecutive chunks of one slot, depth-first. |
| `SYNTH_HISTORY=1` | Don't prefill the history. Fill K, V and `index_k` for positions `[0, h)` of each slot with **random** data (normal, scaled like real activations; *not* zeros, because zeros make indexer scores degenerate), then run only the segment. Apply this to every cache the layer set uses. |
| `WARM_BUCKETS=list` | Make `compile()` warm only the `cached_len` buckets actually used. Warming all buckets up to 1M would take far too long. |
| `RESULT_JSON=1` | Print one machine-readable line per timed iteration, e.g. `RESULT {"iter":k,"wall_ms":...}`, so the driver can parse it without regex on free text. |

### 4.4 Sanity check (must pass before E1)

Run **S8, cold, W = 2048 / 5120 / 8192 (the last with `M3_REPLICATED_TOKENS=1`), B = W/2048, all segments `h=0, n=2048`**.

Expected ≈ 8/30 of the 2-stage (4,4) numbers: **~47 / ~92 / ~140 ms** per forward, ±25%. If the numbers are far off, stop and investigate (layers, mesh, host overhead, warm-up) before spending time on the matrix.

These references are from 2026-09-10. Host-dispatch work since then made eager runs 15–19% faster (issue #50772), so results somewhat **lower** than these are expected and fine; results **higher** by more than 25% are not.

In Phase A there is no packed path yet: run the sanity check as one plain request of W tokens on S8 via the profiler (`STAGES=2 STAGE=0 LAYER_IDS=8,…,15 CHUNK=W CACHE=0`). Cross-check against #57199: layers 0+3 at chunk 5120 on 4x4 took about 26 ms.

---

## 5. Measurement protocol and data format

- **Timing.** Per configuration: 2 warm-up forwards, then ≥5 timed forwards. Report the median, min and max of wall time per forward. Where possible, batch several configurations that share a layer set and width into one process to avoid reloading weights. Reset between processes.
- **Wall time.** Wall = host-observed time per forward with a device sync at the end of the forward. Synchronise only at forward boundaries, never inside the forward.
- **Profiles.** Use a separate process with the zone profiler. Don't mix profiled and timed runs.

`results/runs.csv` has one row per configuration:

```
run_id,timestamp,git_sha,exp,layer_set,layers,n_dense,n_sparse,mesh,sp,tp,ep,W,B,
segments_json,replicated_tokens,traced,synth_history,iters,wall_ms_median,wall_ms_min,
wall_ms_max,device_ms,host_gap_pct,peak_dram_gb,status,notes
```

- `segments_json` example: `[{"slot":0,"h":141312,"n":2048},{"slot":1,"h":0,"n":2048}]`.
- `status` is one of `OK`, `HANG`, `OOM`, `ERROR`, `LOAD_TIMEOUT`.

`results/ops.csv` has one row per (profiled run, layer, zone):

```
run_id,layer,layer_type,zone,device_ms_worst_chip,device_ms_mean,ccl_bytes,notes
```

**Zones to keep:**

- attention total;
- `indexer_score_msa` + index branch, top-k, `sparse_sdpa_msa`;
- `ag_kv`, `ag_index_k`;
- `ring_joint_sdpa`;
- RoPE, KV write (`update_padded_kv_cache`);
- MLP / MoE total (router, dispatch, experts, combine);
- SP / TP collectives;
- whole-forward device time and wall time.

**File layout:**

```
results/
  PROGRESS.md  REPORT.md  runs.csv  ops.csv  coeffs.json
  logs/<run_id>.log  logs/<run_id>.env
  profiles/<run_id>/...
  sim/<name>.txt
  plots/*.png
```

---

## 6. Experiments

Priorities: **P0** is required; **P1** should be done; **P2** is if time allows.

### E1 · Cold width sweep (P0) → Q1, Q6

- **Layer sets:** D, S8, S0.
- **W:** 2048, 4096, 5120. Also 6144, 8192, 10240 with `M3_REPLICATED_TOKENS=1`.
- **Packed:** B = W/2048 separate cold requests (`h=0, n=2048`).
- **Plain:** also one plain request of W tokens on the baseline path (`SEGMENT_SIZE=0`), to see packed vs plain on this sub-mesh.
- **Output:** the ms-vs-W curve per layer set; floor = intercept, slope = ms/token; the knee. Report the host-gap % once per layer set from a profile.

### E2 · Depth sweep (P0) → Q2, Q5

- **Layer sets:** D and S8 (the full grid). Run S0 at h ∈ {0, 141312, 548864}, n = 2048 only, as a composition check.
- **Single-segment grid:** W = 2048, B = 1, n ∈ {256, 1024, 2048} × h ∈ H (18 points per layer set). Use `SYNTH_HISTORY=1`.
- **Two-segment homogeneous:** W = 4096, B = 2, both segments at the same h ∈ {0, 141312, 548864}, n = 2048.
- **What to look for:**
  - **Growth with h** at fixed n. Is it linear? What is the slope for D vs S8?
  - **Change with n** at fixed deep h. If time barely drops from n = 2048 to n = 256 at h = 548864, there is a large **history-only** term (a gather or scan that doesn't shrink with n). This is the key number for hot requests.
  - **The stage-0 ratio** S0(h) / S8(h) as h grows.
- **Validation of synthetic history** (do once): at h = 102400, compare a hot segment on synthetic history against the same segment on **real** history. Prefill a real 102,400-token prompt first; previous runs tiled `longbook_qa_eng_prefill_56320_nopad` into a 102,400 trace. If they differ by more than 5%, say so in the report and treat all synthetic-depth numbers as approximate.

### E3 · Op breakdown (P0) → Q2

Zone profile using the profiler, `LAYER_IDS=0,3` (one dense, one sparse), W = 2048, B = 1:

- h ∈ {0, 141312, 548864} × n ∈ {256, 2048};
- plus one packed W = 4096 profile with `SEGMENTS=548864:2048,0:2048`.

For each zone, report worst-chip device time and, where available, CCL bytes and achieved bandwidth. Name the top 3 zones that grow with h, for the dense layer and for the sparse layer separately.

### E4 · Packing and coupling by history (P0) → Q3, Q4

*Needs the packed path (Phase B in §4.0).* If it isn't available yet, finish Phase A and write the report with Q3/Q4 marked "pending Phase B". The simulator can still run with an additivity assumption, stated as such.

- **Layer sets:** S8 and D; also S0 for C4 and C9.
- **n:** 2048 unless noted.

**W = 4096 (native):**

| ID | `SEGMENTS` | Purpose |
|---|---|---|
| C1 | `0:2048,0:2048` | cold baseline |
| C2 | `141312:2048,141312:2048` | homogeneous median depth |
| C3 | `548864:2048,548864:2048` | homogeneous deep |
| C4 | `548864:2048,0:2048` | one deep + one cold (mixed) |
| C4r | `0:2048,548864:2048` | same as C4, loop order reversed |
| C5 | `141312:2048,0:2048` | median + cold |
| C6 | `20480:2048+22528:2048` | depth-first: two consecutive chunks of one slot |
| C7 | `548864:1024,0:2048` | deep hot segment with a small n |

**W = 8192 (`M3_REPLICATED_TOKENS=1`):**

| ID | `SEGMENTS` | Purpose |
|---|---|---|
| C8 | `0:2048` ×4 | cold baseline |
| C9 | `548864:2048,0:2048,0:2048,0:2048` | one deep in a cold forward |
| C10 | `548864:2048,548864:2048,0:2048,0:2048` | two deep |
| C11 | `141312:2048` ×4 | homogeneous median |
| C12 | `141312:2048,0:2048,40960:2048+43008:2048` | realistic mix: hot + new + two chunks of a long prompt |

**Additivity check (the core of Q3).** Let T1(h, n) be the E2 single-segment times. For each mixed composition, predict:

```
T_pred(C) = T(W, all cold) + Σ_i [T1(h_i, n_i) − T1(0, 2048)]
```

Report `(T_meas − T_pred) / T_meas`. If it is within ±10%, the cost model is additive and grouping by history does not change total work. It then only changes per-forward variance and stage balance, which the simulator handles. If it is off by more than 20%, stop and report which compositions break it.

**Coupling question.** Compare C2 + C1 (two forwards: deep with deep, cold with cold) against 2 × C5 (two mixed forwards):

- Is total time equal?
- Is the per-forward time more uniform in the mixed case?
- Does loop order matter (C4 vs C4r)?

### E5 · Padding cost (P1) → Q6

This mostly comes free from E2 at h = 0 (n = 256 / 1024 / 2048). Add one packed pair at W = 4096: `0:2048,0:256` vs `0:2048,0:2048`. Pad rows are free only if the times match.

### E6 · Layout comparison at depth (P2) → Q7

Compare (2,4) SP=2 with 15 sparse layers against (4,4) SP=4 with S8, at W ∈ {4096, 8192} × h ∈ {0, 141312, 548864}.

Normalise to **chip-µs per token-layer** = chips × stage_ms × 1000 / (W × layers). The previous cold numbers were 27 (SP=2) vs 33 (SP=4). Does the gap hold, shrink or flip at depth?

### E7 · Two-stage mini pipeline (P2, stretch) → Q7

Only if the common prefill runner (`models/demos/common/prefill`) can run M3 on 2 × (4,4). Use stage A = layers 0–7 and stage B = layers 8–15; this is a perf-only partial model.

- Throughput vs forwards in flight ∈ {1, 2, 4} on a cold stream and on a stream with 141312-deep hot segments.
- The **hop time**: from stage A finishing a forward to stage B starting it.

If it's not supported, skip it and write one line on what's missing.

### E8 · Accuracy spot-check at depth (P1)

On **real** history (a prefilled real prompt of 102,400 or 141,312 tokens), compare packed `[deep hot, cold]` against the same segments run alone:

- per-slot K / V / `index_k` PCC, as in the previous study;
- **indexer top-k overlap**: the fraction of selected blocks that match, on the sparse layers.

Packing is expected to add nothing beyond run-to-run noise.

### Optional repeat when trace capture lands

Repeat E1 (S8), E2 (S8, n = 2048, three h values) and E4 (C1, C4, C9) with trace capture on, and mark them `traced=1`.

---

## 7. Analysis

### 7.1 Fit the cost model → `results/coeffs.json`

Per layer type (dense, sparse), per forward:

```
layer_ms = a + b·W + Σ_i ( c·n_i·h_i + d·h_i + e·n_i )
stage_ms = o + Σ_layers layer_ms
```

- **Fitting:** least squares on E1 + E2 + E4. Fit dense from the D runs and sparse from the S8 runs.
- **Stage overhead o:** estimate it from the S0 composition check: `o = T_S8 − (8/5)·(T_S0 − T_D)`. It should be small and consistent across h; report it.
- **Quality:** report R², the residual distribution, and the worst 5 points.
- **Output:** write `coeffs.json` in the format of `DEFAULT_COEFFS` in Appendix A. Set `hop_ms` to 0 unless E7 measured it.

### 7.2 Derived tables

- **Equivalent-token factor** f(h) = T1(h, 2048) / T1(0, 2048), for dense, sparse, and a whole 60-layer model split.
- **History-only share at depth:** `d·h / (c·n·h + d·h)` at n = 256 and n = 2048, for h = 548864.
- **Stage-0 ratio:** S0 / S8 vs h, measured and predicted.
- **Additivity residuals** for C1–C12.

### 7.3 Simulator runs (Appendix A)

Save each run's output to `results/sim/`. Use `--coeffs results/coeffs.json --n 4000` for all.

1. `--stages 8 --widths 2048,4096,6144,8192`: fcfs vs bucket vs cost at each W. This covers the budget and grouping question, Q4.
2. Layer splits for Q5: `--split 8,8,8,8,7,7,7,7`, `--split 6,8,8,8,8,8,7,7`, `--split 5,8,8,8,8,8,8,7`, `--split 4,8,8,8,8,8,8,8`. Pick the split with the best throughput and the most even stage utilisation.
3. Decode-routing sensitivity: `--dec-max-hist 16384`, `65536`, `1048576` (the last is today's rule of only ≤1k new tokens).
4. `--paged`: what segment granularity 128 would buy vs 2048.
5. Hop sensitivity: copy `coeffs.json` with `hop_ms` = 5 and 10.

How to read the simulator output:

- **tok/s** is steady-state useful throughput.
- **fwd p50/p99** is the slowest-stage time per forward. A high p99 means one forward stalls all the others.
- **fill %** = real tokens / padded forward tokens.
- **stage utilisation** shows which stage is the bottleneck.

The traffic mix is an assumption (Appendix C); state that in the report.

### 7.4 Plots → `results/plots/`

1. Stage ms vs W, cold (D, S8, S0).
2. Layer ms vs h at n = 256 / 1024 / 2048, dense vs sparse (two panels).
3. Stacked op breakdown at h = 0 / 141312 / 548864, dense and sparse.
4. Additivity: measured vs predicted, with a y = x line.
5. Simulator: tok/s vs W per policy, and stage utilisation bars per layer split.

---

## 8. Report template (`results/REPORT.md`)

```
# M3 prefill budget study — results
Branch / SHA / date / galaxy / traced? / synthetic history validated? (Δ%)

## Answers
Q1 floor, slope, knee per layer type (table)
Q2 depth scaling per layer type; top zones that grow with h; history-only share at 549k
Q3 additivity: max |residual|; mixed-depth penalty yes/no
Q4 recommended forward budget: W today (SP=4 limit) and after the CCL fix; tokens or ms;
   grouping policy (fcfs / bucket / cost) and why, with simulator numbers
Q5 stage-0 ratio vs h; recommended layer split; utilisation per stage
Q6 padding cost: pad rows free or not; implied rounding waste on the AgentX mix
Q7 (if run) SP=4 vs SP=2 chip-µs per token-layer at depth; hop ms

## Cost model
coeffs.json, R², worst residuals, stage overhead o

## Equivalent-token table
h → f(h) for dense, sparse, whole model

## What surprised us / what to measure next
## Runs that failed (HANG/OOM) and why
```

---

## Appendix A · Packing / pipeline simulator (`m3_budget_sim.py`)

Save this as `results/m3_budget_sim.py`. Pure Python, no dependencies. It runs with placeholder coefficients so you can test it now. **Its throughput numbers mean nothing until you pass `--coeffs results/coeffs.json`.**

- **Model:** in-order pipeline (flow shop), per-layer cost model from §7.1, AgentX-like traffic (Appendix C).
- **Constraints it respects:** 2048-segment padding (`--paged` gives 128) and depth-first chunking of long prompts.
- **Scheduler:** a 64-request lookahead window.

```python
#!/usr/bin/env python3
"""Packing / pipeline simulator for the M3 prefill budget study.

Replays an AgentX-like request mix through an in-order N-stage prefill pipeline and
compares packing policies (fcfs, bucket, cost) and forward widths.
Per-layer costs come from coeffs.json (fitted from the single-galaxy measurements).
The DEFAULT_COEFFS below are PLACEHOLDERS so the script runs; replace them.

  python3 m3_budget_sim.py --coeffs coeffs.json --stages 8 --n 4000
"""
import argparse, json, math, random
from collections import deque

SEG = 2048                      # max tokens per segment (= KV period today)
PAD = 2048                      # each segment is padded to a multiple of PAD (2048 today, 128 with paged KV)
TRACE_BUCKET = 2048             # forward width is rounded up to a multiple of this (trace shapes)
DENSE_LAYERS = {0, 1, 2}        # full-attention layers in M3
N_LAYERS = 60

# Per-layer cost model, ms, for one forward on one (4,4) SP=4 stage:
#   layer_ms = a + b*W_padded + sum_i (c*n_i*h_i + d*h_i + e*n_i)
# W_padded = forward width after trace-bucket rounding, n_i = real tokens of segment i,
# h_i = its cached_len (history). a,b: per-layer fixed + per-token cost; c: new x history
# (attention / indexer compute); d: history-only (e.g. index_k gather); e: per real token.
# stage_ms = overhead + sum over the stage's layers of layer_ms
DEFAULT_COEFFS = {  # PLACEHOLDERS - replace with fitted values
    "sparse": {"a": 2.2, "b": 1.8e-3, "c": 1.0e-9, "d": 1.0e-6, "e": 0.0},
    "dense":  {"a": 2.2, "b": 1.8e-3, "c": 3.0e-8, "d": 1.0e-6, "e": 0.0},
    "stage_overhead_ms": 0.0,
    "hop_ms": 0.0,
}

# ---------------------------------------------------------------- traffic
NEW_Q = [(0, 32), (.25, 640), (.5, 1600), (.9, 5888), (.99, 50000), (.999, 100000), (1, 1_000_000)]
HIST_Q = [(0, 0), (.05, 0), (.25, 60_000), (.5, 142_000), (.75, 310_000), (.9, 549_000), (1, 1_000_000)]

def qsample(rng, qs):
    """Piecewise log-linear interpolation between published quantiles (linear next to 0)."""
    u = rng.random()
    for (p0, v0), (p1, v1) in zip(qs, qs[1:]):
        if u <= p1:
            t = (u - p0) / (p1 - p0) if p1 > p0 else 0.0
            if v0 > 0 and v1 > 0:
                return int(math.exp(math.log(v0) + t * (math.log(v1) - math.log(v0))))
            return int(v0 + t * (v1 - v0))
    return int(qs[-1][1])

def sample_requests(n, seed, max_ctx):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        new = max(1, qsample(rng, NEW_Q))
        hist = qsample(rng, HIST_Q)
        hist = (hist // SEG) * SEG                  # today: continue only from a SEG boundary
        new = min(new, max_ctx - hist)
        if new > 0:
            out.append((new, hist))
    return out

def route_to_prefill(reqs, dec_max_new, dec_max_hist):
    return [r for r in reqs if not (r[0] <= dec_max_new and r[1] <= dec_max_hist)]

def to_segments(rid, new, hist):
    segs, off = [], 0
    while off < new:
        n = min(SEG, new - off)
        segs.append({"rid": rid, "h": hist + off, "n": n})
        off += SEG
    return segs

# ---------------------------------------------------------------- cost
class CostModel:
    """stage_ms(fwd) = overhead + sum_layers(a + b*W) + sum_segs sum_layers(c*n*h + d*h + e*n)."""
    def __init__(self, C, split):
        self.C, self.stages = C, []
        start = 0
        for nl in split:
            layers = range(start, start + nl); start += nl
            kinds = [C["dense"] if L in DENSE_LAYERS else C["sparse"] for L in layers]
            self.stages.append(kinds)
        assert start == N_LAYERS, split
        self.S = len(split)
        self.A = [C.get("stage_overhead_ms", 0.0) + sum(k["a"] for k in ks) for ks in self.stages]
        self.Bw = [sum(k["b"] for k in ks) for ks in self.stages]

    def seg_vec(self, seg):
        n, h = seg["n"], seg["h"]
        return [sum(k["c"] * n * h + k["d"] * h + k["e"] * n for k in ks) for ks in self.stages]

    def fwd_vec(self, W, segsum):
        return [self.A[s] + self.Bw[s] * W + segsum[s] for s in range(self.S)]

# ---------------------------------------------------------------- packing
def bucket_of(h, edges=(16_384, 142_000, 310_000)):
    return sum(h >= e for e in edges)

def padded(n):
    return -(-n // PAD) * PAD

def pack(policy, reqs, W_max, M, lookahead=64):
    """reqs: (new, hist) in arrival order -> list of (segments, seg_cost_sum, W_forward).
    Only the next unscheduled segment of each request is eligible; a forward may take
    several consecutive segments of one request (depth-first). The scheduler looks at the
    first `lookahead` requests in the queue.
      fcfs   : fill the width in arrival order
      bucket : fill only from requests whose next segment is in the same history bucket
      cost   : like fcfs, but stop adding when the forward's slowest stage would exceed the
               cost of a full cold forward (deep segments displace tokens)"""
    streams = [deque(to_segments(i, n, h)) for i, (n, h) in enumerate(reqs)]
    for st in streams:
        for seg in st:
            seg["v"] = M.seg_vec(seg)
    active = deque(range(len(streams)))
    zero = [0.0] * M.S
    B_full = W_max // SEG
    full_cold = [sum(x) for x in zip(*([M.seg_vec({"n": SEG, "h": 0})] * B_full))]
    target = max(M.fwd_vec(W_max, full_cold))
    fwds = []
    while active:
        head = [active.popleft() for _ in range(min(len(active), lookahead))]
        window = head
        if policy == "bucket":
            b = bucket_of(streams[head[0]][0]["h"])
            window = [i for i in head if bucket_of(streams[i][0]["h"]) == b]
        fwd, ssum, used = [], zero[:], 0
        for i in window:
            st = streams[i]
            while st and used + padded(st[0]["n"]) <= W_max:
                seg = st[0]
                trial = [x + y for x, y in zip(ssum, seg["v"])]
                if policy == "cost" and fwd:
                    Wt = -(-(used + padded(seg["n"])) // TRACE_BUCKET) * TRACE_BUCKET
                    if max(M.fwd_vec(Wt, trial)) > target:
                        break
                fwd.append(st.popleft()); ssum = trial; used += padded(seg["n"])
            if used + PAD > W_max:
                break
        keep = [i for i in head if streams[i]]
        active.extendleft(reversed(keep))
        W_fwd = -(-used // TRACE_BUCKET) * TRACE_BUCKET
        fwds.append((fwd, ssum, W_fwd))
    return fwds

# ---------------------------------------------------------------- pipeline
def flowshop(fwds, M):
    """In-order pipeline: forward f enters stage s when stage s is free and f left stage s-1."""
    hop = M.C.get("hop_ms", 0.0)
    done, busy, fwd_ms = [0.0] * M.S, [0.0] * M.S, []
    for fwd, ssum, W in fwds:
        cs = M.fwd_vec(W, ssum)
        prev = 0.0
        for s in range(M.S):
            start = max(done[s], prev + (hop if s else 0.0))
            done[s] = start + cs[s]; busy[s] += cs[s]; prev = done[s]
        fwd_ms.append(max(cs))
    mk = done[-1]
    return mk, [b / mk for b in busy], fwd_ms

def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs", default=None)
    ap.add_argument("--stages", type=int, default=8)
    ap.add_argument("--split", default=None, help="comma list of layers per stage, e.g. 8,8,8,8,7,7,7,7")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--widths", default="2048,4096,6144,8192")
    ap.add_argument("--dec-max-new", type=int, default=1000)
    ap.add_argument("--dec-max-hist", type=int, default=65_536)
    ap.add_argument("--max-ctx", type=int, default=1_048_576)
    ap.add_argument("--paged", action="store_true", help="pad segments to 128 instead of 2048 (paged-KV what-if)")
    a = ap.parse_args()
    global PAD
    if a.paged:
        PAD = 128
    C = json.load(open(a.coeffs)) if a.coeffs else DEFAULT_COEFFS
    if a.split:
        split = [int(x) for x in a.split.split(",")]
    else:
        base, extra = divmod(N_LAYERS, a.stages)
        split = [base + (1 if i < extra else 0) for i in range(a.stages)]
    M = CostModel(C, split)
    reqs = route_to_prefill(sample_requests(a.n, a.seed, a.max_ctx), a.dec_max_new, a.dec_max_hist)
    real = sum(n for n, _ in reqs)
    print(f"split={split} prefill requests={len(reqs)} real tokens={real:,}")
    print(f"{'W':>6} {'policy':>7} {'tok/s':>9} {'fwd p50 ms':>10} {'fwd p99 ms':>10} {'fill %':>7}  stage utilisation")
    for W in [int(x) for x in a.widths.split(",")]:
        for pol in ("fcfs", "bucket", "cost"):
            fw = pack(pol, reqs, W, M)
            mk, util, fms = flowshop(fw, M)
            fill = real / sum(f[2] for f in fw) * 100
            print(f"{W:6d} {pol:>7} {real / mk * 1000:9.0f} {pct(fms, 50):10.1f} {pct(fms, 99):10.1f} {fill:7.1f}  "
                  + " ".join(f"{u*100:3.0f}" for u in util))

if __name__ == "__main__":
    main()
```

---

## Appendix B · Known issues and gotchas

- **SP CCL hang:** fixed on `main` by #57199 (row-major `all_broadcast` packet sizing). If your tree predates `718e1f2`, (4,4) runs, and forwards over 1088 tokens per chip, can hang; use `M3_REPLICATED_TOKENS=1` in the experiment harness, or update main.
- **Alignment:** `cached_len` and every history value must be multiples of 2048.
- **Warm-up time:** `compile()` warms every `cached_len` bucket. Restrict it to the buckets you use (`WARM_BUCKETS`), or warm-up may take longer than the runs.
- **Memory:** at h = 548864 with several deep slots, check per-chip DRAM (record `peak_dram_gb`). If a composition OOMs, record it and drop to fewer deep slots rather than shrinking h.
- **Accuracy vs width:** wide forwards drift slightly (KV PCC 0.94–0.95 vs 2k) from MoE routing flips. That is a width effect, not a packing effect, so compare packed against the same width.
- **Host overhead:** runs are untraced, so the whole galaxy is host-launch bound. Sub-meshes hide most of it, but record the host-gap % so traced runs can be compared later.
- **Synthetic history** changes which blocks the indexer selects. Validate once against real history (E2); if the gap is large, prefer real-history runs for the headline numbers.

## Appendix C · Traffic assumptions used by the simulator

These come from the AgentX coding-agent traces (internal analysis) and are request-weighted percentiles.

- **New tokens:** p25 640, p50 1.6k, p90 5.9k, p99 50k. The tail p99.9 of 100k is an assumption.
- **History:** p25 ~60k, p50 142k, p75 310k, p90 549k. About 5% cold (h = 0) is an assumption.
- **Correlation:** new tokens and history are sampled independently. Internal analysis found the median new tokens roughly flat vs history above 16k.
- **Routing to decode:** requests with ≤1000 new tokens **and** history ≤ `--dec-max-hist` go to decode; everything else goes to prefill.
- **Processing:** long requests are split into consecutive 2048-token segments, processed in order.

If the IS team provides a real histogram of new tokens and history for requests routed to prefill, replace `NEW_Q` / `HIST_Q` with it and rerun §7.3.
