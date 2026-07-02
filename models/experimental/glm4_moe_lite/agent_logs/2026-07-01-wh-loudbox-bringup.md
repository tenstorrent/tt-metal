# GLM-4.7-Flash bring-up on Wormhole LoudBox (T3K, 8 chips) — 2026-07-01

Author: agent session. Scope: get `models/experimental/glm4_moe_lite` running end-to-end
on a Wormhole **LoudBox / T3K** (8 Wormhole chips = 4× n300, `/dev/tenstorrent/0..3`),
and stand up an in-process matmul device-time **sweep** harness for later optimization.

The model/README were originally validated on **Galaxy (32-chip WH, 4×8)** and **Blackhole
QB-2 (1×4)**. T3K (2×4 / 1×4) had two real bugs that made every multi-chip run hang or crash.
Both are now fixed and the full 47-layer model runs e2e on all 8 chips.

---

## TL;DR — how to run on the LoudBox (2×4, all 8 chips)

```bash
cd /home/gtobar/tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
```

Batch-1, ISL 128, trace (the canonical single test):
```bash
TT_METAL_GTEST_ETH_DISPATCH=1 \
GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0 \
GLM4_MOE_LITE_CCL_NUM_LINKS=1 \
GLM4_MOE_LITE_CCL_TOPOLOGY=linear \
GLM4_MOE_LITE_SKIP_DEFENSIVE_CLONES=1 GLM4_MOE_LITE_FUSE_QKV_A=1 \
GLM4_MOE_LITE_FUSE_SHARED_GATE_UP=1 GLM4_MOE_LITE_BATCHED_PREFILL=1 \
GLM4_MOE_LITE_DECODE_L1_ACT=1 GLM4_MOE_LITE_EP_L1=1 \
GLM4_MOE_LITE_FUSE_MLP_MOE_REDUCE=1 GLM4_MOE_LITE_SKIP_TYPECAST=1 \
python models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py \
  --prompt "Summarize" --simulate-context-len 128 --min-cache-tokens 256 \
  --max-new-tokens 128 --batch-size 1 \
  --mesh-rows 2 --mesh-cols 4 --kv-cache-dtype bf16 \
  --phase both --enable-trace --trace-mode sampling
```

Batch-32: `--batch-size 32 --kv-cache-dtype bf8 GLM4_MOE_LITE_BATCHED_PREFILL=0`.

**Non-negotiables on T3K:** `GLM4_MOE_LITE_CCL_NUM_LINKS=1`, `GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0`,
`--mesh-rows 2 --mesh-cols 4`.

### Measured (batch-1, ISL 128, 2×4, trace-sampling)
- prefill_s ≈ 1.34 (≈ 0.53 with `--warmup`, i.e. after prefill compile is warmed)
- steady-state decode ≈ **47.5 ms/tok** (`subsequent: mean=47.5 min=46.6 max=55.6`) ≈ **21 tok/s**
- eager (no trace) steady-state ≈ 407 ms/tok — trace is ~8.5× faster once captured.

With **`--warmup`** (capture excluded from the timed run) — the true numbers:
- `warmup_prefill_s=0.791`, `warmup_decode_s=2.030` (one-time trace **capture** ≈ 2.0 s; the earlier
  15.8 s "first token" also included first-time decode-kernel compilation, which `--warmup` absorbs)
- `prefill_s=0.529`, `first token=48.4 ms` (real first **replay** — matches steady state), `tok_s=21.06`
- Reference: README Galaxy batch-1 ISL-128 ≈ 74.8 ms/tok; the 8-chip LoudBox is faster per-token here.

---

## Fixes made

### Fix 1 — CCL hang (the 43-minute "stuck" run) → `GLM4_MOE_LITE_CCL_NUM_LINKS=1`
- **Symptom:** any mesh>1 run wedged during prefill — one host thread busy-spinning ~100% CPU
  forever, no output, kernel cache frozen. Single-chip (1×1) ran fine.
- **Root cause:** T3K has **1 CCL link per axis**. See `models/common/modules/tt_ccl.py`
  `link_dict` → `"T3K": (1, 1)`. The debug/sweep path used `GLM4_MOE_LITE_CCL_NUM_LINKS=2`
  (default in `runtime_config.py` and `moe_tt.py`). The TP `all_reduce` (row-parallel `w_o`,
  `mlp_down`) reads `cfg.ccl_num_links` and, unlike the MoE sparse path, is **not** auto-capped
  to the hardware link count → it requested 2 links on 1-link hardware → the collective
  deadlocked at the first reduce in prefill (layer-0 dense `mlp_down`).
- **Fix:** run with `GLM4_MOE_LITE_CCL_NUM_LINKS=1` (topology `linear`). Fabric setup was already
  correct (`_set_default_fabric_config` sets `FABRIC_1D` for non-Galaxy before `open_mesh_device`).
- **Durability TODO:** make `num_links` auto-cap to the detected hardware links for the non-MoE
  CCL paths too (mirror `moe_tt._detect_galaxy_ccl`), so T3K doesn't need the env var.

### Fix 2 — lm_head reshape crash on 2×4 (DP mesh) → code change in `tt/model_tt.py`
- **Symptom (after Fix 1):** run reached the end of prefill then crashed:
  `RuntimeError: shape '[-1, 154880]' is invalid for input of size 77440` (154880 = vocab;
  77440 = vocab/2). Only appeared on a **2D / DP>1** mesh (2×4); 1×1 and 1×4 were fine.
- **Root cause:** the lm_head sharded the vocab across **all** mesh devices
  (`ttnn.ShardTensorToMesh(device, dim=3)`, `lm_head_tp_axis=None`,
  `vocab_per_shard = vocab // num_devices`). On a DP mesh the final hidden state lives on a
  **single DP row** (4 devices), so only that row's vocab shards get computed → logits come back
  as vocab/2 → the reshape to full vocab fails.
- **Fix (`Glm4MoeLiteDenseOnlyTT.create`, the `if tp_enabled and num_devices > 1:` block):**
  shard the vocab across the **TP axis only** (cols preferred), **replicating across the DP rows** —
  exactly how every other TP weight is sharded (`_tp_mesh_mapper` → `ShardTensor2dMesh(dims=(None,3))`):
  ```python
  mesh_rows_lh, mesh_cols_lh = int(device.shape[0]), int(device.shape[1])
  if mesh_cols_lh > 1: lh_axis, lh_tp = 1, mesh_cols_lh
  else:               lh_axis, lh_tp = 0, mesh_rows_lh
  shard_dims = (None, 3) if lh_axis == 1 else (3, None)
  lm_head_mapper = ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=[mesh_rows_lh, mesh_cols_lh])
  lm_head_sharded_vocab = True
  lm_head_tp_axis = lh_axis          # was None
  lm_head_tp_size = lh_tp            # was num_devices (8) -> now mesh_cols (4)
  lm_head_vocab_per_shard = vocab // lh_tp
  ```
  The gather sites already handle `lm_head_tp_axis==1` (gather across `mesh_cols`); the prefill
  `_extract_logits` gather (`get_device_tensors` + concat + `[..., :vocab]`) is robust to the new
  layout. Confirmed working in **both** eager and trace-sampling (on-device top-1 / sampling-allgather).
- **Note:** this is a genuine 2×4-support bug in the model, worth committing / sending to the owner.
  It forces a one-time regen of the lm_head weight cache (shard variant tag bumped to `..._v2`).

### Also relevant (Wormhole vs Blackhole)
- `GLM4_MOE_LITE_PREFILL_MATMUL_TUNED` defaults **on** and selects **Blackhole-only** matmul grids
  (e.g. a 10×4 grid for head-parallel `w_q_b`) that exceed Wormhole's 8×8 Tensix grid →
  `TT_FATAL: compute_with_storage_grid_size (10,4) must fit within device grid (8,8)`. On WH set
  `GLM4_MOE_LITE_PREFILL_MATMUL_TUNED=0` (falls back to grid-aware configs). The decode tuned dict
  and sparse-MoE grids self-adapt (`_resolve_grid`), so only the prefill flag needs disabling.

### Operational lessons (T3K device hygiene)
- After **any** hard `kill -9` of a mesh process, run `tt-smi -r` before the next run. SIGKILL skips
  `close_mesh_device`, leaving the ethernet **fabric orphaned** → the next `open_mesh_device` hangs
  at "topology discovery". (Prefer letting `timeout` send SIGTERM so Python's `finally` closes cleanly.)
- The TT weight cache under `~/.cache/ttnn/models/glm4_moe_lite/vllm` is **mesh-shape specific**
  (encodes `tp4`, per-device coords). A 2×4 cache fails to load on 1×4 (`MeshCoordinate([1,0]) not
  found in MeshShape([1,4])`) and silently regenerates — expect a slow first load when changing mesh.
- Distinguishing hang vs progress: watch **cache-file growth** (`~/.cache/tt-metal-cache/*.o`), not
  CPU. A busy-spin CCL hang shows ~100% CPU with a frozen cache — CPU alone is misleading.

---

## Open item the user asked about: `--warmup` to time the real first token

Yes — `debug_run_full_tt_greedy.py` has **`--warmup`** (line 156). With `--warmup --enable-trace`
it runs a warmup prefill **and a warmup decode step that captures the trace** *before* the timed
loop, and reports the capture cost separately as `warmup_decode_s`. So the timed run's
`first token` becomes the real first **replay** (no 15.8 s capture). Use:

```bash
... --enable-trace --trace-mode sampling --warmup
```

Then read `first token` (real replay) and `subsequent: mean=...` (steady state); capture cost is the
separate `warmup_decode_s` line. (`GLM4_MOE_LITE_PRESERVE_TRACE=1` also avoids re-capture overhead
across prefill.)

---

## For the optimization agent: the matmul device-time sweep tools

Location: `models/experimental/glm4_moe_lite/sweeps/`. Purpose: brute-force different
**program config / memory config / math fidelity / dtype** for each GLM matmul and read back the
**on-device kernel time (ns) in-process** — no `python -m tracy`, no CSV.

### Files
- `profiler_setup.py` — sets the 3 profiler env vars (`TT_METAL_DEVICE_PROFILER`,
  `TT_METAL_PROFILER_MID_RUN_DUMP`, `TT_METAL_PROFILER_CPP_POST_PROCESS`,
  `TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES`) **before ttnn imports** (only place env vars live),
  and exposes `read_latest_kernel_ns` / `dominant_kernel_ns` / `sum_kernel_ns` that call
  `ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`. Requires a build with
  `ENABLE_TRACY=ON` (present in `build_Release`).
- `targets.py` — `MatmulTarget` (full [K,N] + shard scheme + collective + dp_split) and the
  verified GLM target registry (`GLM_TARGETS_2D` = attention + shared/dense MLP + router;
  `GLM_TARGETS_PHASE2` = head-parallel kv_b1/b2 + sparse experts, not yet implemented). Plus
  `PhaseSpec` (decode/prefill → M) and `SweepAxis` (the brute-force grid).
- `harness.py` — builds the full weight, pushes it to the mesh with the model's real mesh mapper
  (so each chip runs its true per-device shard), runs the matmul eagerly, reads per-chip kernel ns,
  aggregates (max across chips for matmul, avg for collective), prints a ranked table.
- `run_matmul_sweep.py` — CLI entry (opens 2×4 mesh, fabric+dispatch like the model).

### How M (matmul rows) is modeled (matches the model)
- **decode:** M = batch (1 tok/seq, tile-padded to 32).
- **prefill:** M = batch × seq_len, capped at `prefill_chunk` (default 128 → chunked prefill; to
  actually sweep prefill M, vary `--prefill-chunk` or set 0 for unchunked).
- **DP:** attention matmuls (`dp_split=True`) see M // `dp_rows` (set `--dp-rows 2` for realistic
  2×4 attn_dp; default 1 = off).

### Verified per-device shapes (TP=4, from config.json + on-device cache)
`w_q_a [2048,768] repl` · `w_kv_a [2048,576] repl` · `w_q_b(attndp) [768,5120] repl` ·
`w_q_b(headpar) [768,1280] col` · `w_o [1280,2048] row+all_reduce` ·
`w_shared_gate/up [2048,384] col` · `w_shared_down [384,2048] row+all_reduce` ·
`w_dense_gate/up [2048,2560] col` · `w_dense_down [2560,2048] row+all_reduce` · `w_router [2048,64] repl`.

### Run it (once the mesh is free)
```bash
# validated: single matmul-only target returns real kernel ns (e.g. w_q_a m=32 -> ~32.8us)
python models/experimental/glm4_moe_lite/sweeps/run_matmul_sweep.py \
  --mesh-rows 2 --mesh-cols 4 --phase decode --batch 1,32          # decode sweep
python .../run_matmul_sweep.py --mesh-rows 2 --mesh-cols 4 --phase prefill --batch 1 --prefill-chunk 128
```
Edit `SweepAxis` in `targets.py` to widen the grid (dtypes, fidelity, mem, prog configs). Program-
config grids auto-clamp to the device grid (won't request >8 wide on WH). Invalid configs are caught
and reported as `FAILED` so the brute force keeps going.

### Status / caveats for the optimization agent
- **Validated:** the in-process readback works end-to-end (matmul-only targets, no collective).
- **Not yet validated on device:** collective (`all_reduce`/CCL) timing path, `l1_width_sharded`
  output config edge cases, and the phase-2 sparse-expert / head-parallel batched targets (stubs).
- **Reuse the model's kernel intel:** the tuned configs the model already uses live in
  `tt/linear_helpers.py` (`_DECODE_MATMUL_TUNED`, `prefill_linear_2d_bs_out`, the
  `prefill_matmul_tuned_enabled()` branches) and `tt/moe_tt.py` (sparse matmul prog cfgs). Those
  are **Blackhole-tuned**; the sweep's job is to find the Wormhole-optimal equivalents and feed them
  back into those call sites (guarded by arch, since PREFILL_MATMUL_TUNED must be off on WH today).
- Any sweep uses collectives on the row-parallel targets — remember `num_links=1` on T3K.

---

## Suggested next steps
1. Verify **batch-32** (trace, `--kv-cache-dtype bf8`, `BATCHED_PREFILL=0`) e2e on 2×4.
2. Re-time batch-1 with **`--warmup`** to report the true first-token replay latency.
3. Correctness check with a **real prompt** (drop `--simulate-context-len`) and/or the repo PCC tests
   (`tests/*optional*`) — current perf runs use repeated-prompt input so output is degenerate by design.
4. Make `num_links` auto-cap on T3K (durable Fix 1) and commit the lm_head fix (Fix 2).
5. Run the matmul sweeps to find WH-optimal program/memory/fidelity configs; wire winners into
   `linear_helpers.py` behind an arch check.
