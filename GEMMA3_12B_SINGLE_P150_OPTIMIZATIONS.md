# Gemma3-12B — Single P150 Decode Optimization Log

Automation-ready record of manual decode optimizations for **`google/gemma-3-12b-it`** on a
**single Blackhole P150** (batch-1, greedy). Written so an optimization agent (e.g. `tt_hw_planner`)
can (a) **replay the BANKED levers**, (b) **skip the REJECTED ones** (don't re-burn device time),
and (c) **not re-attempt what the stack already does**.

---

## 1. Context

| Item | Value |
|---|---|
| Model | `google/gemma-3-12b-it` (text), 48 decoder layers, dim 3840, vocab 262 400 |
| HW | 1× Blackhole P150 (board enumerates as `p300`, 1 chip visible) |
| Backbone | `models/tt_transformers` (shared) + `models/demos/multimodal/gemma3` demo |
| Config | `MESH_DEVICE=P150`, `TT_VISIBLE_DEVICES=0`, p150 mesh graph descriptor |
| Mode | `performance` (`DecodersPrecision.performance`) |
| Metric | decode tok/s/user, batch-1, ISL 128 / OSL 200, `enable_trace=True` |

### Result

| Stage | tok/s/user | Δ | Notes |
|---|---|---|---|
| performance baseline | **21.43** | — | stock `performance` mode |
| + Lever A (decode fidelity LoFi) | **24.8** | +15.7% | accuracy-gated |
| + on-device force-argmax | **27.73** | +11.8% | byte-identical to host |
| + row-major-slice argmax | **29.1** | +4.3% | byte-identical to host |
| **cumulative** | **29.1** | **+36%** | |

**Accuracy (performance mode, gated):** Top1 **84.6%**, Top5 **95.8%** (floor 82.0 / 94.0).

### Roofline / ceiling (why we stop ~29–30)
Batch-1 decode is **weight-DRAM-bandwidth-bound**. 24.8 tok/s/u ≈ 52% of DRAM BW; the 80% HW
ceiling for a dense LLM is ~**38 tok/s/u**. After the argmax wins removed host/overhead time, the
remaining time is dominated by weight-BW-bound matmuls that are already near their kernel floor.
**38 is not reachable with accuracy-safe drop-in changes at batch-1** — it needs batch>1 (amortize
weight reads) or >1 chip.

---

## 2. Accuracy gate protocol (apply to EVERY accepted lever)

A lever is only banked if **all** hold:
1. **Token-matching** vs HF greedy reference (`generate_reference_hf.py` → `gemma-3-12b-it.refpt`):
   `text_demo.py -k performance-ci-token-matching` → **Top1 ≥ 82.0, Top5 ≥ 94.0**
   (floors in `models/model_targets.yaml`, `gemma-3-12b / bh_p150`).
2. **Per-module PCC ≥ 0.99** in performance mode: `test_mlp.py`, `test_decoder.py` with
   `GATE_OPT=performance` (the tests gate on `GATE_OPT`).
3. For sampling-path changes: **on-device output == host argmax** (A/B via `FORCE_HOST_SAMPLING`;
   confirmed byte-identical).

> Note: `performance-ci-token-matching` forces **host** sampling, so it validates model quality but
> NOT the device argmax — always add the A/B check for sampling changes.

---

## 3. BANKED optimizations (replay these)

### Lever A — decode-path matmul fidelity HiFi2 → LoFi
- **Files:** `models/tt_transformers/tt/model_config.py`, `models/tt_transformers/tt/lm_head.py`
- **What:** In `ModelOptimizations.performance` (the non-Qwen `else` branch), set LoFi for the
  decode matmul op-groups; weights stay BFP8 (mantissa preserved), only math cycles drop.
  Also add a new `OpGroup.LI_LM_HEAD` so the 262k-vocab projection is config-driven.
- **Diff (model_config.py):**
  ```python
  # OpGroup enum: add
  LI_LM_HEAD = "li_lm_head"
  # performance() OpFidelity:
  "OpFidelity": {
      OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI,
      OpGroup.LI_FF2: MathFidelitySetting.LOFI,
      OpGroup.LI_QKV_DECODE: MathFidelitySetting.LOFI,
      OpGroup.LI_O_DECODE: MathFidelitySetting.LOFI,
      OpGroup.LI_LM_HEAD: MathFidelitySetting.LOFI,
  }
  # _default_settings() OpFidelity: add LI_LM_HEAD: HIFI2 (preserve default)
  ```
- **lm_head.py:** read `LI_LM_HEAD` fidelity from `args.decoders_optimizations`; keep the exact
  original hardcoded HiFi2 kernel when the config resolves to HiFi2 (so other models are unchanged),
  else use the config value.
- **Impact:** 21.43 → 24.8 tok/s/u (+15.7%). **Accuracy:** Top1 84.6 / Top5 95.8 (pass).
- **Generalization:** safe pattern for any dense model that is decode-math-bound; always re-gate.

### Lever — on-device force-argmax (enable)
- **Files:** `models/tt_transformers/tt/model_config.py`, `models/tt_transformers/tt/model.py`
- **Problem:** single device has per-device padded vocab (262k) > 64K, so the on-device sampling
  gate was false → decode fell back to **host** argmax = full-logits D2H every token (~tens of ms).
- **What:**
  - `model_config.py`: for `num_devices == 1`, set `SAMPLING_AG_CONFIG["allow_force_argmax"] = True`.
  - `model.py`: relax the gate so a single device with `allow_force_argmax` samples on device:
    ```python
    single_device = list(self.mesh_device.shape) == [1, 1]
    allow_force_argmax = self.args.model_config.get("SAMPLING_AG_CONFIG", {}).get("allow_force_argmax", False)
    self._supports_on_device_sampling = prefetcher is None and (
        self.args.vocab_size // sampling_splits <= 64 * 1024 or (single_device and allow_force_argmax)
    )
    ```
- **Impact:** 24.8 → 27.73 tok/s/u (+11.8%). **Accuracy:** byte-identical to host argmax.
- **Generalization:** applies to any single-device greedy model whose vocab exceeds the 64K split.

### Lever — row-major-slice argmax (decode fast path)
- **File:** `models/common/sampling/tt_sampling.py` (+ thread-through in
  `models/common/sampling/generator.py` and `models/tt_transformers/tt/generator.py`)
- **Problem:** the force-argmax path untilized + argmax'd the **full 32 tile-padded rows** over the
  262k vocab, even at batch-1 (only 1 real row). (Ref: gpt-oss `gtobarTT/gpt_oss_opt`, ~8ms→~0.2ms.)
- **What:** DECODE-only, slice logits to the real user rows → `ROW_MAJOR` → argmax. If a wider
  feedback buffer is preallocated (decode token buffer `[1,1,1,32]`), argmax the active rows then
  `ttnn.pad` to the buffer width and `ttnn.copy` in place (trace-safe). PREFILL keeps the original
  full untilize+argmax (its 32 rows are sequence positions — it indexes the true last-token row).
  - Thread `active_rows` through `sample → capture_trace/_run_sampling → tt_sampling.forward`.
  - Decode passes `active_rows = tt_sampling.decode_active_rows` (= real unpadded batch); prefill
    passes `None`.
- **Impact:** 27.9 → 29.1 tok/s/user (+4.3%). **Accuracy:** byte-identical; gate Top1 84.6/95.8.
- **Note:** smaller than gpt-oss's win because our path already untilized to row-major; the saving
  here is only the 32→1 row reduction (~2ms).

### Supporting infra (not perf, but required to gate)
- `models/model_targets.yaml`: added `gemma-3-12b / bh_p150` Top1/Top5 floors (82.0 / 94.0).
- `test_mlp.py`, `test_decoder.py`: `GATE_OPT=performance` exercises perf-mode fidelity for PCC.
- `generate_reference_hf.py` → produced `gemma-3-12b-it.refpt` (golden greedy reference).

---

## 4. REJECTED — do NOT re-attempt (accuracy or negligible)

| Lever | Result | Verdict |
|---|---|---|
| FF2 weights BFP8→BFP4 (whole model) | +4% perf, **Top1 81.6** (< 82 floor) | REJECT (accuracy) |
| WQKV BFP8→BFP4 | **Top1 79.0** | REJECT (accuracy) |
| WO BFP8→BFP4 | ~1.5% upside, same risk profile | SKIP (not worth) |
| Mixed-depth FF2 BFP4 (middle 36/48 layers) | **+0.2 tok/s/u** (29.1→29.3) | REJECT (negligible post-argmax) |
| Width-sharded QKV (Llama PR #50666 style) | decode warmup **hang** | REJECT — gemma3 GQA 16/8 + QK-norm differ from Llama |
| lm_head logits output L1→DRAM (gpt-oss note) | no-op (29.0≈29.1) | SKIP — our sampling path doesn't hold logits in L1 |

**Rule for the agent:** BFP4 is only safe on **FF1/FF3** for gemma3-12b. FF2, QKV, WO must stay BFP8.

---

## 5. ALREADY PRESENT in `tt_transformers` — do NOT reinvent

These gpt-oss (`gtobarTT/gpt_oss_opt`) levers are already implemented in our backbone; porting them
is a no-op. (gpt-oss's demo was an immature codebase; `tt_transformers` already has the mature form.)

| gpt-oss lever | Status in tt_transformers |
|---|---|
| lm_head matmul program-config | Already DRAM-sharded + tuned via `get_lm_head_program_config` (gpt-oss started from a configless `ttnn.matmul`) |
| decode RMSNorm parallelization | Already `LayerNormShardedMultiCoreProgramConfig` across **30–40 cores** (attn grid 10×3, mlp/ff grid 8×5) — better than gpt-oss's 9-core fix |
| matmul dtype/block sweeps | Swept (`sweep_llama_mm_v2.py`, gemma3 shapes): decode matmuls already near DRAM-BW floor; standalone sweep is NOT a faithful in-model proxy |
| MoE / sparse_matmul / expert-grid | N/A — gemma3 is a **dense** LLM |

### 5a. Full per-commit audit of `gtobarTT/gpt_oss_opt` (PR #51400, 24 commits, 21→56 tok/s on gpt-oss)

Every commit checked against gemma3-12b (dense, single P150, batch-1). **Result: nothing new to port.**
gpt-oss's gains came from (a) MoE-specific work (inapplicable to a dense model) and (b) bringing an
immature demo up to the optimization level `tt_transformers` (gemma3's backbone) already has.

| # | commit | gpt-oss gain | applies to gemma3-12b? |
|---|---|---|---|
| 1 | matmul program-config sweep **tool** | — | Tooling only. We already swept gemma3 shapes (`sweep_llama_mm_v2.py`); decode matmuls near DRAM-BW floor. |
| 2 | tune decode MoE **expert grids** | +43% | **N/A — MoE** |
| 3 | tune prefill MoE **expert grids** | TTFT −54% | **N/A — MoE** |
| 4 | fuse decode MoE **gate+up sparse_matmul** | −6.6% dev | **N/A — MoE** (dense w1/w3 already separate tuned linears) |
| 5 | parallelize decode **RMSNorm** 9 cores | −2.5% dev | **ALREADY** — gemma3 decode norm sharded 30–40 cores (`get_norm_config`→`create_sharded_norm_config`) |
| 6 | RMSNorm sharding for multi-tile-row | −1.9% | **ALREADY** (same) |
| 7 | keep sharded RMSNorm out in **L1** | −0.4% | **ALREADY** (sharded_output_config in L1) |
| 8 | pin MoE **router** weight+bias in L1 | −0.1% | **N/A — MoE router** |
| 9 | **on-device argmax** greedy decode | +40% | **DONE BY US** (`allow_force_argmax` + gate relax): +11.8% |
| 10 | faster argmax via **row-major slice** | +31% | **DONE BY US** (decode-only slice in `tt_sampling.py`): +4.3% |
| 11 | multi-core config for 1-core **router** matmul | +5% | **N/A — MoE router** |
| 12 | fuse **o_proj bias**-add + bf8 cast | — | **N/A** — gemma3 attention has no o_proj bias |
| 13 | fuse **qkv bias** into decode linear | — | **N/A** — gemma3 has no qkv bias (uses QK-norm) |
| 14 | generalize moe_sparse_matmul_sweep CLI | — | **N/A — MoE tooling** |
| 15 | retune decode **expert** sparse_matmul in0_block_w=45 | +11% | **N/A — MoE** |
| 16 | **lm_head** matmul program config | +9% | **ALREADY** — gemma3 lm_head DRAM-sharded+tuned (`get_lm_head_program_config`); gpt-oss started from a config-less `ttnn.matmul`. **A/B TESTED** the 1D-mcast factory (env-gated interleaved full-width weight + `MatmulMultiCoreReuseMultiCast1DProgramConfig`, per_core_N=64, 129 cores): **28.27 vs 28.8 tok/s/u baseline = ~2% REGRESSION** (coherent output). dram-sharded is weight-stationary/BW-optimal; mcast broadcasts the activation and dumps a 262400-wide logits tensor to DRAM. **REVERTED.** |
| 17 | custom JiT fused MoE **expert-reduce** kernel | — | **N/A — MoE** |
| 18 | parallelize custom fused **MoE reduce** | — | **N/A — MoE** |
| 19 | batch activation reads in **MoE-reduce reader** | — | **N/A — MoE kernel** |
| 20 | emit decode **gate/up sparse_matmul** bf16 | +3% | **N/A — MoE** (we tested MLP FF2 bf16/bfp4 on gemma3: no gain / accuracy fail — see §4) |
| 21 | emit decode **down sparse_matmul** bf16 | +1.7% | **N/A — MoE** |
| 22 | **qkv output L1_INTERLEAVED** @TP=1 | +3.2% | **ALREADY** — gemma3 `attention.py:645` does `sharded_to_interleaved(..., L1_MEMORY_CONFIG)` before `nlp_create_qkv_heads_decode` (gpt-oss used DRAM) |
| 23 | **trim lm_head to real vocab** when sampling disabled | +3% | **N/A** — gemma3 has on-device sampling **enabled** single-chip (this branch is the disabled path); saving ~256 padding cols ≈ 0.1% anyway |
| 24 | **SwiGLU alpha-fold** → fused silu | +2.2% | **N/A** — MoE-expert SwiGLU with `alpha` scale; gemma3 MLP uses gelu-tanh (no alpha to fold) |

**Bottom line:** the portable techniques (lm_head PC, qkv→L1, sharded RMSNorm, on-device argmax) are
already in gemma3; the remaining ~18 commits are MoE-specific or bias fusions gemma3 doesn't need.
Single-chip batch-1 stays **~28.8 tok/s/u**; the real latency lever remains **≥2 chips** (see §5b/§6).

---

## 5b. NON-PRECISION / structural levers investigated (batch-1 single P150)

Precision is exhausted (§3/§4). These are the *structural / dataflow* levers explored to attack the
~8.7 ms/token non-matmul overhead and the weight-BW inefficiency. Summary: **the one lever that
materially moves batch-1 (the DRAM prefetcher) is hardware-blocked on a single chip.**

| Lever | Finding | Verdict |
|---|---|---|
| **DRAM prefetcher** (overlap weight DRAM→L1 with compute; the lever that makes Llama-8B fast) | **Does NOT fit single-chip L1.** `is_prefetcher_supported` math for 12b (dim 3840 × hidden 15360): FF weight block = **1.04 MB/core at ring 80** (best case) vs **0.85 MB** L1 budget. Fails at every ring size (16/64/80). Even **Llama-3.1-8B does not fit at 1 device** — only at ≥2. The prefetcher is fundamentally a **multi-device** technique (shards weight blocks into per-chip L1). | **BLOCKED single-chip (HW).** Viable on ≥2 chips (12b: 0.52 MB/core at ring 80, 2 devices). |
| `force_fixed_decode_k_chunk` (pin SDPA-decode k_chunk=256 vs auto; enabled for 4b/27b, not 12b) | 28.82 vs 28.78 tok/s/u — noise. SDPA is a negligible fraction at batch-1 / short context. | REJECT (no batch-1 benefit) |
| Sliding-window attention KV-read capping (gemma3 local:global, window=1024) | At the customer's ISL 128 / OSL 200 the context (≤328) is **< 1024**, so no layer reads beyond the window regardless. No reads to cap. | N/A at target context |
| Reshard / op-graph pruning (cut `to_memory_config` churn in the norm→matmul→norm chain) | Remaining single-chip opportunity in principle (~8.7 ms/token non-matmul), but requires a reliable per-op decode profile to target safely. **Full-model tracy is unreliable for 12b** (DRAM marker-buffer overflow / "device data missing"), so blind edits here risk correctness for uncertain gain. gemma3's extra norms (QK-norm + pre/post-attn + pre/post-FF) are architecture-required and already sharded across 30–40 cores. | NOT attempted (needs profile; high risk / low confidence) |
| **Core-grid tuning** of decode matmuls (P150 has a 10×12 tensix grid vs the Wormhole-era 8×8 caps) | **A/B TESTED.** MLP FF1/FF3/FF2 grid comes from `find_grid_k_n` (drives `num_cores`→`per_core_N` in `dram_matmul_config`), hardcoded `max 8×8`. Expanding to `10×12` (env `TT_GEMMA3_MLP_GRID`) does raise it to **CoreGrid(12,10)=120 cores**, but the run **fails allocation**: `TT_FATAL bank_manager.cpp:430: num_shards <= num_compute_banks` (120 > P150 usable compute banks). The divisor constraint (`c \| gcd(K_tiles,N_tiles)=120`) offers only **40 (current, works) or 120 (fails)** — nothing in between. Attention QKV decode is **already** Blackhole-tuned (`CoreCoord(8,10)`); `attn_input_grid` uses `find_grid` (already 10×12); SDPA-decode `(8,8)` is **under-subscribed at batch-1** (16 q-heads × 1 batch ⇒ idle cores, more won't help). And the big FF/QKV/O matmuls are **DRAM-sharded ⇒ weight-BW-bound**: adding compute cores doesn't cut DRAM read time. | REJECT (MLP 120 fails bank alloc; others already maxed or BW-bound) |

### The real path to 38 tok/s/user for a single user = **≥2 chips (tensor-parallel)**
Splitting 12b across 2 chips halves per-device weight bytes/token. This is the correct config for a
*latency* (single-user) target. **Multi-chip now works on this box after a proper `tt-smi` reset**
(see below); earlier `unordered_map::at` failures were a transient bad device state, NOT a permanent
limitation.

### This box is actually 2× P300 (4 chips, healthy eth ring) — CORRECTED
- Real topology (from `build/tools/umd/topology` + `system_health`): **two P300 boards** — chips
  `{0,1}` = one P300, `{2,3}` = the other — wired in a **4-chip ethernet ring** 0↔1↔2↔3↔0. All eth
  links **UP, retrain 0**. (Earlier "4× P150, no ethernet" conclusion was wrong — I misread the UMD
  "remote chip ids {}" line, which only means no *off-host* chips.)
- **The `unordered_map::at` on every multi-chip call was a bad device STATE.** A proper device reset
  cleared it: afterwards `get_num_devices()==4` and 2-chip / 4-chip meshes open cleanly.
- **2-chip P300** (chips {0,1}, stock `p300` descriptor, `MeshShape(1,2)`) opens and runs.
- **4-chip** needs a `channels count: 2` descriptor: stock `p150_x4` assumes **4 eth links/hop** and
  fails (`control_plane.cpp:1168: Expected 4 eth links from chip 3 to chip 0`); this box has **2
  links/hop**. A custom 2×2 / channels=2 descriptor opens all 4. (Not needed for the 2-chip goal.)

### 2-chip P300 result (gemma3-12b, batch-1)
- **Baseline (no prefetcher): 28.76 tok/s/user** — essentially flat vs single-chip's ~29. At batch-1,
  tensor-parallel splits the matmuls but adds ethernet CCL (all-gather) overhead that roughly cancels
  the gain, and on-device argmax is disabled in multi-chip (host round-trip returns).

### DRAM prefetcher on 2-chip P300 — attempted, structurally blocked for gemma3-12b
Wiring added (env-gated `TT_GEMMA3_PREFETCHER=1`, default off): registered `gemma-3-12b` in
`VERIFIED_MODEL_CONFIGS`, threaded `prefetcher` through gemma3 `ModelArgs`/`create_tt_model`. Support
math OK: at 2 devices it selects ring_size 64 (`receiver_cores=8`, 557 KB/core < 850 KB). Blockers hit:
1. **WO ring-shard used `dim` instead of `n_heads·head_dim`** → shard-grid-fit fatal. gemma3 has
   `n_heads·head_dim = 16·256 = 4096 ≠ dim = 3840`. **FIXED** (`get_sharded_wo_ring_mem_config` now uses
   `n_heads·head_dim`; no-op for Llama where they're equal) + WO ring tensor forced to a 2D mesh mapper
   (`ShardTensor2dMesh(dims=(2,3))`) since fused-AGMM is off at `num_devices != 8`. This got the model
   through construction + prefill on 2-chip.
2. **`fd_mesh_command_queue.cpp:389: sub_device_ids.size() == 1`** in decode (both traced AND eager),
   at `model.py:796` — gemma3's `rope_local` op (created with `prefetcher=None`, `model.py:86`) ran on
   cores outside the prefetcher's worker sub-device. **FIXED** (`model.py:86` now passes
   `prefetcher=prefetcher`; `HfRotarySetup.get_rot_mats` then builds cos/sin/trans_mat on
   `prefetcher.all_worker_cores_range_set`). Strict no-op when `prefetcher is None` (all single-chip /
   non-prefetcher paths unchanged). Decode now advances past rope to warmup.
3. **`tensor_layout.cpp:168: Physical shard shape (32,120) must be tile {32,32} sized`** at the decode
   **embedding** (`embedding.py:45` ← `model.py:_transform_decode_inputs_device`) — a **structural
   tile-alignment cascade**, not a single bug. The prefetcher's decode shard configs are tuned for
   Llama's power-of-2 dims; gemma3's `dim=3840` (=**120 tiles**) / `hidden=15360` (=**480 tiles**) don't fit:
   - **Residual** (`get_residual_mem_config`, `dim//cluster_shape[1](=2)//worker_cores`): 1920 = **60 tiles**,
     and `dynamic_worker_core_grid` only yields **multiples of 8** (8→240, 16→**120**, 32→60, 48→40 elems) —
     **none** is a multiple of 32. 60 has no multiple-of-8 divisor ⇒ **cannot** tile-align via the existing helper.
   - **MLP input** (`get_mlp_input_mem_config`, `dim//ring_size`): aligns only at ring ∈ {8,24,40}, but the
     auto-selected **ring=64** gives 60 (misaligned) and doesn't divide the weight tile-counts (480/64=7.5).
   - Ring that divides gcd(120,480)=120 **and** is `receiver_cores·8`: only {8,24,40,120}. A working config
     would need ring **retuned to 40** (rc=5) AND the residual/attn/lm_head grids rewritten off
     `dynamic_worker_core_grid` — a **multi-front, gemma3-specific rewrite** of the prefetcher sharding math,
     with L1-fit re-derivation at the new ring. NOT attempted (throughput-only; see below).

**Why not worth chasing for batch-1:** the prefetcher hides **DRAM-bandwidth** latency (a *throughput*
lever). At batch-1 on 2×P150 we sit at 28.76 tok/s/u vs a DRAM roofline of ~80+ (12b bfp8 ≈ 6 GB/chip
÷ ~512 GB/s) — i.e. **~3× under roofline ⇒ latency/dispatch-bound, not BW-bound**. Even a fully working
prefetcher would not move batch-1 meaningfully; it pays off at higher batch (multi-user throughput).

**Agent takeaway:** For a single-user *latency* target, single-chip (~29 tok/s/u, on-device argmax) is
the best config; 2-chip is flat at batch-1. The DRAM prefetcher is a *throughput* lever — only pursue it
for multi-user/high-batch. Progress banked toward a working prefetcher: (a) WO `n_heads·head_dim` ring
fix, (b) **rope global+local now confine to the worker sub-device** (`model.py:86`). Remaining =
the tile-alignment cascade (blocker #3): retune ring 64→40 and rewrite residual/mlp/attn/lm_head shard
grids for gemma3's 120/480-tile dims. Only do this for a **multi-user/high-batch throughput** goal —
it yields **zero** batch-1 latency benefit (batch-1 is dispatch-bound at ~3× under the DRAM roofline).

---

## 6. Reproduce (env + commands)

```bash
cd <repo>
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
export TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150 HF_MODEL=google/gemma-3-12b-it
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

NODE=models/demos/multimodal/gemma3/demo/text_demo.py::test_demo_text
# perf (batch-1 tok/s/user):
pytest "$NODE[blackhole-mesh_device0-device_params0-performance-batch-1]" -s -q
# accuracy (token-matching, host sampling):
pytest "$NODE[blackhole-mesh_device0-device_params0-performance-ci-token-matching]" -s -q
# per-module PCC in perf mode:
GATE_OPT=performance pytest models/demos/multimodal/gemma3/tests/test_mlp.py     -k "512 or 256" -s -q
GATE_OPT=performance pytest models/demos/multimodal/gemma3/tests/test_decoder.py -k "512 or 256" -s -q
```

---

## 7. Guidance for the automation agent

1. **Gate every candidate** with §2 (token-match + PCC + argmax A/B). Reject on any failure.
2. **Start from BANKED (§3)**; they compose. Re-measure after each.
3. **Never propose §4 levers** for gemma3-12b (BFP4 outside FF1/FF3, width-sharded QKV, L1 logits).
4. **Never re-port §5 levers** — already in the backbone.
5. **Pre-flight `trace_replay`** on the actual perf harness before entering an optimize/reset loop;
   fall back to **eager** if trace isn't available (gemma3 text demo DOES support trace).
6. Batch-1 is BW-bound near ceiling — for larger gains, evaluate **batch>1** or **>1 chip** instead
   of more single-chip precision/kernel micro-opts.
