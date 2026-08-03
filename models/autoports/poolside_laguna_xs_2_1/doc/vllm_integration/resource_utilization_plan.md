# Laguna-XS-2.1 — spend the idle silicon

## Context

A vllm-bench sweep on tt-quietbox (4× p300c, P150x4 mesh, all three shipped decode opts default
ON) shows the machine is almost entirely idle while serving:

| resource | measured | headroom |
|---|---|---|
| FLOPs (tracy, time-weighted) | **1.6–1.9 % of peak** | ~50× |
| DRAM bandwidth (short ctx) | 448 MB/token/dev → 11–14 GB/s of ~512 GB/s = **2.2–2.9 %** | ~35× |
| Tensix cores per op (time-weighted) | **48–60 of 120** | ~2× |
| DRAM capacity | 10.7 GB of ~32 GB per ASIC (**~31 %**) | ~23.5 GB/dev free |
| Board power | 324 W of 500 W serving vs **50 W idle**, and **flat vs batch** (C1 297 W, C8 291 W, C16 298 W) | not the limit |
| AICLK / temps | 1346 of 1350 MHz serving (800 MHz idle), 64–70 °C vs 90 °C throttle, 0 trips | not throttling |

Nothing throttles and nothing saturates. Decode is latency/dispatch-bound, which the batch-flat
t/s/u confirms directly: 25.3 (C=1) → 23.8 (C=8) → 24.5 (C=16) — 16× the work for the same watts
and the same per-user speed.

**Read the power numbers carefully.** Idle (device open, trace resident, no requests) is 50 W at
800 MHz, so serving really does draw ~274 W of *dynamic* power — this is not a story about static
leakage. The diagnostic point is that those 274 W are **already being spent at batch 1** and do
not rise when the useful work grows 16×. The energy is going into per-step overhead — dispatch,
NOC traffic, and cores spinning at 1350 MHz — rather than into arithmetic. That is why power is a
poor utilisation proxy here and why the FLOPs/DRAM/core-occupancy columns above are the ones to
trust.

**Goal (user-selected priority): raise single-user decode t/s/u first**, then convert the
remaining idle capacity into concurrency. Target 25 → 35+ t/s/u at batch 1, no accuracy or
determinism regression.

The existing docs (`dispatch_gap_analysis.md`, `performance_plan.md`) argue "reduce op COUNT".
This plan keeps that and adds two axes they miss: **a reproducible multi-minute decode stall**,
and **the ops that do run occupy a fraction of the core array**.

---

## Measured op inventory (source of W2)

`doc/optimized_multichip_decoder/tracy/layer4/decode_perf_report.csv` — one decode layer,
4-device merge. `mean cores` is time-weighted, out of 120:

| op | % device time | mean cores | headroom |
|---|---|---|---|
| SparseMatmul (gate+up, packed) | 12.4 % | **24** | 5× |
| SparseMatmul (down) | 12.3 % | 51 | 2.4× |
| BinaryNg | 11.1 % | 107 | — |
| **ReduceScatter** | 9.3 % | **8** | link-bound |
| SdpaDecode | 8.5 % | 109 | — |
| Unary | 7.7 % | 103 | — |
| Matmul 32×2048×256 (router) | 4.6 % | **9** | 13× |
| FillPad | 4.3 % | 68 | op-count target |
| **AllGather** | 4.2 % | **8** | link-bound |
| Matmul 32×2048×2048 (o_proj) | 4.1 % | **12** | 10× |
| Slice | 3.4 % | 60 | op-count target |
| ReshapeView | 3.3 % | 20 | op-count target |
| **LayerNorm** | 2.6 % | **1** | 100× |
| Matmul 32×1536×2048 (qkv) | 2.3 % | **12** | 10× |

> **Caveat (why W0 exists):** this capture is dated Jul 22 and contains no `RotaryEmbedding` op,
> i.e. it predates the shipped fused-RoPE/fused-reduce work. The core counts are structural
> (they come from program configs, not opt flags), but every percentage must be re-measured.

### Root cause of the low core counts

`tt/optimized_decoder.py:271`

```python
def _decode_shard_cores(k: int, n: int, max_cores: int = 32) -> int:
```

Every dense decode matmul goes through `_dram_mm` (`tt/optimized_decoder.py:713-730`), which
calls this with the **default cap of 32** — against 120 cores. The divisibility constraints
(`kt % c == 0`, `kt // c >= 2`) push the actual count to the 9–12 observed.

The in-family reference it was adapted from does better: `models/common/modules/mlp/mlp_1d.py:639`
`_find_grid_k_n(..., max_rows=8, max_cols=8)` → **up to 64 cores**, requiring the count to divide
both K-tiles and N-tiles. Reuse that helper rather than writing a new one.

---

## Workstreams

Ordered for single-user decode first. Each step is gated by a fast probe before the next.

### W0 — Re-baseline (do first; half a day)

1. Re-capture single-layer tracy on **current** code with shipped opts ON. Confirm op mix, the
   `Cores` column, and the device-time/gap split.
2. Record the batch-1 decode baseline with `vllm bench` from `.venv_benchmarks_vllm`.
3. Capture the accuracy baseline with `TT_LAGUNA_DECODE_SDPA_PC=0` to neutralise the known
   confound (`doc/vllm_integration/decode_sdpa_pc_finding.md`).

**Gate:** if re-captured core counts are already high, W2 is void — say so and move to W3.

### W1 — Root-cause the multi-minute decode stall (biggest observed single-user effect)

Caught live during this sweep. `ISL=1024 / OSL=1024 / C=1` ran twice on the same config:

| run | t/s/u | E2EL | median ITL | p99 ITL | mean ITL |
|---|---|---|---|---|---|
| 21:34 | **25.28** | 41.2 s | — | — | 39.5 ms |
| 22:10 | **1.92** | **533.9 s** | 43.1 ms | 53.0 ms | **521.2 ms** |

Median 43 ms and p99 53 ms are *healthy* — decode itself is fine. The mean is 12× the p99, so
essentially all 533 s is **one stall of ~8 minutes** (or a few tens of seconds each) beyond the
p99 boundary.

**A second instance confirms it is not a one-off.** `ISL=32768 / OSL=128 / C=16` shows the same
shape: median 46.5 ms, p99 52.3 ms, **mean 178.7 ms** (ratio 3.4), t/s/u 5.60 against 21.50 for
the same ISL at C=8. Both affected points involve **C=16**, and the 1024-token case ran
immediately after the C=16 points — so the trigger correlates with a large or changing batch
shape. Every unaffected point has mean ≈ median.

Ruled out: it is **not** lazy trace capture — the Tier-3.2 warning at
`tt/generator_vllm.py:1040` never fired (0 occurrences in the server log). The sweep driver is
serial, so it is not overlapping bench processes.

**Leading hypothesis — program recompile / buffer allocation under the resident decode trace.**
Supporting evidence:
- `allocator.cpp:123` *"Allocating device buffers is unsafe due to the existence of an active
  trace"* fired at 21:58:54 in this server's log.
- `warmup_model_prefill` warms only **row-count 1** page tables (`tt/generator_vllm.py:1206`,
  `:1213`), but the plugin passes `[num_reqs, w]`. Every unseen batch size allocates a fresh
  device buffer under the resident trace.
- The commit note for the chunked-prefill work records exactly this magnitude: *"chunked 4096 is
  3.2 s standalone but **11+ min** at serving until the wide-page-table programs recompile."*
- The bad run followed the C=16 points, i.e. a batch-shape transition.

**Actions:** reproduce deterministically (run C=16 then C=1 back-to-back); instrument
`decode_forward`/`_prefill_pt_grouped` to log every allocation and every shape-key miss; then fix
by pre-warming the full `(1..max_num_seqs, w)` page-table shape set, or by pinning the prefill
page table to a fixed `(max_num_seqs, w)` with unused rows zeroed.

This is first because a 13× intermittent regression outweighs every tuning item below, and
because it will silently poison the measurements those items depend on.

### W2 — Widen the core grids

**W2a — dense decode matmuls (router, qkv, o_proj, shared/dense MLP).** Replace
`_decode_shard_cores` (`tt/optimized_decoder.py:271-287`) with the `mlp_1d.py` selection logic
(`_find_grid_k_n`, `_dram_shard_core_grid_k_n`), raising the cap 32 → 64. The DRAM-sharded weight
memcfg (`_dram_weight_memcfg`, `:297`) is keyed to `dram_cores = mesh_device.dram_grid_size().x`
and must stay consistent with the new compute grid — a weight-prep change at load time.

*Honest risk:* these are single-tile-M matmuls (M=32). If they are launch-bound rather than
compute-bound, more cores buys nothing and the reduction overhead may cost. Probe one matmul
standalone before touching the model.

**W2b — LayerNorm on 1 core** (`tt/optimized_decoder.py:1124-1127`, `core_grid=one_core`).
2.6 % of decode device time on a single Tensix core. Cheapest item on the list.

**W2c — SparseMatmul gate+up on 24 cores.** 12.4 % of decode *and* 73 % of prefill device time —
the largest single consumer, and it pays twice. Grid comes from `_sparse_pc`
(`tt/optimized_multichip_decoder.py`).

**W2d — CCL (ReduceScatter 9.3 % + AllGather 4.2 % on 8 cores).** Fabric-link-bound, not
core-bound, so the knob is `num_links` (currently 2) and payload dtype, not the grid.
`fused_moe_analysis.md` §3 already proved the deepseek fused reduce-scatter is unavailable at
Laguna's 1×4 / DP=1 shape. Lowest expected value here; do last.

### W3 — Cut decode op count (co-equal with W2; may be bigger)

Audited against current source, **not** the docs — `dispatch_gap_analysis.md` line numbers are
stale (doc 07-30 21:38, `tt/*.py` edited 07-31 21:41). One sliding+MoE layer (30 of 40) issues
**≈84 ttnn calls**; full-attention layers ≈90; **≈3.3–3.4 k python-level ttnn calls per step**.
**Half of every layer (42 of 84 ops) is pure layout/data movement**: 18 reshape,
10 to_memory_config, 6 sharded_to_interleaved, 5 slice, 1 permute, 1 to_layout, 1 zeros_like.

| item | verdict | note |
|---|---|---|
| 2a fused RoPE | **DONE**, default ON | via `rotary_embedding_hf`; *added* layout tax — see W3c |
| 2b `sharded_to_interleaved` | **NOT DONE** — now **6**/layer, was 5 | zero work done |
| 2c `_split_qkv` | **NOT DONE** | 3 slice + 3 reshape |
| 2d `_per_head_norm` | **NOT DONE** | 4 reshapes/layer |
| 2e `_gate` | **NOT DONE** | 3 reshapes, one undoing the previous |
| 2f MoE layout cluster | **PARTIAL — ~1 of 14 ops** | only the combine reduce fused |
| §3 `reset_batch` | **NOT DONE** — leave it, see W3d | |

**W3a — the attention micro-cluster (highest value, lowest risk).** `_split_qkv` (6 ops,
`tt/optimized_decoder.py:833-843`) + `_per_head_norm` ×2 (6 ops, `:679-683`) + `_gate` (6 ops,
`:702-710`) + the attn reshape at `tt/multichip_decoder.py:555` = **19 ops/layer ≈ 760 ops/step**,
all pure address math.

The replacements already exist in this repo and are already used **in prefill**:
`ttnn.experimental.nlp_create_qkv_heads` (`tt/optimized_decoder.py:854-860`, whose own comment
reads "one op replaces 3× slice + 3× reshape + 3× permute") and `nlp_concat_heads` (`:887`).
They were never applied to decode. Reuse, don't invent. Free bonus: `multichip_decoder.py:555`
reshapes `attn` and `_gate`'s first line reshapes it straight back — a 2-op trim, no semantic
change.

**W3b — the MoE block (42 of 84 ops/layer, ×39 layers).** `OptimizedMultichipDecoder._moe`
(`tt/optimized_multichip_decoder.py:113-174`). The router prologue alone is **13 ops on a
`[1,1,B,256]` tensor** (`:123-132`), ending in the `ttnn.to_layout(union, ROW_MAJOR)` that appears
as the 16.7–16.9 µs FillPad/Untilize. The `wv` reshape→permute→reshape triple (`:161-163`) is a
clean hand-fusion target. Only `deepseek_moe_fast_reduce_nc` landed; `fused_moe_analysis.md` §3
proved the `_fused` score-weighted and reduce-scatter variants are unavailable at this mesh shape,
so this is hand-fusing, not op adoption.

**W3c — RoPE's leftover layout tax.** `_fused_rope_decode` (`tt/optimized_decoder.py:1177-1195`)
wraps the fused op in shard/unshard: 2 `sharded_to_interleaved` at `:1192`, then a re-shard at
`:1096`. Keeping q/k **sharded** through `_shard_kv` and SDPA removes the round-trip. On the 10
full-attention layers, partial rotary (`partial_rotary_factor = 0.5`) adds 2 slice + 1 concat per
q/k.

**W3d — `reset_batch`: explicitly de-prioritised, do not revert.** Still hard-coded `True` (live
plugin `model_runner.py:963-970`) with 3 `copy_host_to_device_tensor` per step
(`tt/generator_vllm.py:1064-1071`). But the A/B has **already been run** —
`doc/vllm_integration/reset_batch_probe.tsv` gives 6.14/6.03 t/s/u changed-only vs ~6.0 baseline,
**within noise** — while `batched_decode_corruption.md` records the cliff it guards (22–29/100
corrupt turns → 0/100). Measured ~zero gain against a known correctness cliff.

> Related hazard, one-line fix: an *uninstalled* plugin tree at
> `/home/ttuser/dispatch/vllm/plugins/vllm-tt-plugin/.../model_runner.py:940` still carries the
> unsafe changed-only refresh. If anything re-points the install there, the corruption returns
> silently.

### W4 — Prefill: stop pinning every TTFT to the slowest request

Measured: 8×16k → TTFT 104 s of a 110 s E2E; 16×16k → TTFT 166 s with **identical** aggregate
throughput; 8×32k → TTFT 186 s. Pure serialization.

Confirmed causes:
- Scheduler-level chunked prefill is hard-disabled (`platform.py:481-483`, asserted at
  `model_runner.py:160-161`); batches are all-prefill-or-all-decode (`scheduler.py:32-36`), and
  the scheduler is **prefill-first, decode-starving** (`scheduler.py:66-99`).
- The model loops requests **serially** inside one `execute_model`
  (`tt/generator_vllm.py:686-703`) — nothing returns until the last prefill finishes.
- Prefill is **eager, never traced**: `enable_trace` is accepted at `tt/generator_vllm.py:655`
  and silently ignored.

**W4a (zero code, do immediately):** lower `--max-num-batched-tokens` from 131072 to ~16384–32768
so the scheduler admits 1–2 prefills per step. Total prefill time is unchanged — the device is
serial regardless — but TTFT staggers (~13 s, 26 s, …) instead of uniformly 104 s.

**W4b:** the `(N, w)` page-table warmup gap — same fix as W1, and likely the same root cause.

**W4c (large, optional):** true scheduler-level chunked prefill. The hard part exists — the
suffix-prefill path takes `start_pos` and reads the paged prefix via
`chunked_scaled_dot_product_attention`, and the `start_pos % 64 == 0` constraint is free at
block_size=64. The cost is the mixed-batch work in the plugin (removing the all-prefill/all-decode
split and `assert not any_running` at `model_runner.py:920`). Gated on the blocker below.

### W5 — Fix the silently-dead hybrid KV cache, then convert free DRAM into concurrency

**W5a — the hybrid KV spec hook never fires. This is a live bug, and `hybrid_kv_status.md`'s
"STATUS: ACTIVE" is wrong.**

`_HYBRID_KV_CACHE_GROUPS_ENABLED = True` (`tt/generator_vllm.py:105`) and `get_kv_cache_spec`
(`:181-243`) correctly emit `SlidingWindowSpec(512)` for the 30 sliding layers — but vLLM never
calls them. Root cause, live plugin `worker.py:194-201`:

```python
arch = next((a for a in self.model_config.architectures if a.startswith("TT")), None)
```

`platform.py:541-544` prefixes **`hf_config.architectures`** in place, but
`model_config.architectures` is a separate, already-materialised list that never gets the prefix.
The scan finds nothing → falls back to `model_config.architecture` → `TTTransformersMoEForCausalLM`
→ resolves to vLLM's *generic* Transformers impl, which has no `get_kv_cache_spec` → the dummy
single-layer spec at `worker.py:220-256` is used instead.

Proof from the live log, independent of reading the code: `Overriding num_gpu_blocks=70368744177664`
is `2^64 / page_size / **1**`, so `group_size == 1`; and `GPU KV cache size: 197,632 tokens` is
`3088 // **1** * 64`, so `len(kv_cache_groups) == 1`. Had the hybrid spec been used, the 10 full +
30 sliding layers would have formed **4 groups of 10** and the log would read **49,408 tokens**.

Consequences today:
- **All 40 layers carry a full-length KV cache** — `(3088, 2, 64, 128)` BFP8 per layer — instead of
  10. KV is therefore **4.30 GB/device, not the 1.08 GB the hybrid design intends** (a 4× overspend
  that buys nothing).
- The `+ sliding_window * max_batch * 8` headroom term is *still* added to the pool
  (`worker.py:576-578`), so we pay the sliding-window block tax and get none of its savings.
- `_layer_to_group_idx` stays `None` (`model_runner.py:383-384`), so `page_tables_per_layer` is
  never passed; Laguna's `_prefill_pt_grouped` / `_decode_pt_grouped_alloc` are dead code in this
  configuration.
- It is also the one place bandwidth stops being free: at 131k context, KV reads are
  **2.85 GB/token/device** against 448 MB of weights — ~79 GB/s at 36 ms/token, ~15 % of roofline.
  Fixing the hook cuts that ~4×.

Fix: scan `hf_config.architectures` (or prefix `model_config.architectures` too). One line.

**W5b — raise the pool. Coupled to W5a; do not ship either alone.**

Fixing W5a *without* raising the block count would split the same pool 4 ways and **cut usable
full-attention capacity from 197,632 to 49,408 tokens** — a serving regression.

The pool is pure policy, with no memory profiling anywhere. `determine_available_memory`
(`worker.py:269-273`) returns the literal constant `1 << 64` and overrides the block count:

```
get_max_tokens_all_users → min(max_model_len, ADVERTISED_MAX_CONTEXT)   = 131,072
  + block_size * max_batch          (64 * 16)                           = 132,096
  + sliding_window * max_batch * 8  (512 * 16 * 8, _MAX_SLIDING_GROUPS_HEURISTIC)
                                                                        = 197,632
  ceil(197,632 / 64)                                                    =   3,088
```

The "1.51×" is an accident of those two additive constants, not a capacity calculation.

Practical notes for the implementation:
- **`--num-gpu-blocks-override` on the CLI does not work** — the worker *overwrites*
  `cache_config.num_gpu_blocks_override` in `determine_available_memory`. Change
  `ADVERTISED_MAX_CONTEXT` / `get_max_tokens_all_users` (`tt/generator_vllm.py:59-64`, `:245-254`)
  or the plugin term instead.
- **No trace-shape risk from the pool itself**: page-table width is
  `min(cdiv(max_model_len, block_size), num_blocks) = min(2048, N)` (`model_runner.py:368-371`),
  already saturated at 2048. Growing `N` changes only the KV tensors' leading dimension.
- **It is a restart, not live surgery.** KV buffers are re-allocated and the decode trace
  re-captured; both are already correctly sequenced at init, and `allocate_kv_cache_per_layer`
  defensively clears `self._decode` (`tt/generator_vllm.py:341-342`). There is no supported path
  to grow the pool on a running server.
- **Nothing checks DRAM.** Because available memory is faked, `_check_enough_kv_cache_memory` can
  never fire — over-allocating fails as a raw TT-NN DRAM OOM at allocation time, not as a clean
  vLLM error. Size it by hand: **1.393 MB per block per device** today, or 0.348 MB once W5a lands.

Because t/s/u is batch-flat, this is close to free *aggregate* throughput — but it is aggregate,
not single-user, so it ranks last under the chosen priority. W5a, however, is a correctness-grade
bug and should be fixed regardless of where the throughput work lands.

---

## Blocker to resolve before any plugin work

The **live** plugin is
`/home/ttuser/.local/lib/model-bringup/tt-metal/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/`
— confirmed by the server log (`Using custom scheduler class vllm_tt_plugin.scheduler.TTScheduler`).
It is **not** `tt-inference-server/tt-vllm-plugin`, which is not even importable. `TODO.md` records
that the live tree carries uncommitted working-tree edits (reset_batch fix, hybrid-KV plumbing,
suffix-only prefill, sliding-window gate). Any W4/W5 plugin change lands on an unversioned diff —
commit or branch it first.

---

## Measurement protocol (non-negotiable)

- Benchmark **only** with `vllm bench serve` from `.venv_benchmarks_vllm`. Custom clients are
  banned — they produced a phantom 21→6 t/s/u "regression" before.
- Report **ISL / OSL / E2EL** plus **t/s/u** and **agg tok/s**. Never ms/tok.
- **Always check mean-vs-p99 ITL**, not just the mean — that is the only reason the W1 stall was
  visible rather than being recorded as "decode got slow".
- Iterate with a fast 1-point probe + single-layer tracy; full sweep only at milestones.
- Keep `power_watch.log` + `sweep_power_report.log` running — rising core occupancy should raise
  watts. A t/s/u gain with no power movement deserves suspicion.
- The server takes ~16 min from launch to open routes (585 s init + warmup). Do not start a bench
  before the routes are logged, or it fails with ConnectionRefused and records a zero row — which
  is exactly what invalidated two points of the current sweep.

## Guard rails

- Every step re-validates greedy-token equality against the W0 baseline; bit-exactness is the bar
  the prior opts were held to.
- Accuracy checks run with `TT_LAGUNA_DECODE_SDPA_PC=0`.
- Each change stays behind a `TT_LAGUNA_*` env flag so it reverts without a rebuild.
- Do not re-litigate bfp8 experts — already disproven as an accuracy win.
- Hard-killing a FABRIC_1D_RING server dirties eth cores: `tt-smi -r all` before reopening.

## Verification

1. `pytest models/autoports/poolside_laguna_xs_2_1/tests/` — offline shape/bucket invariants.
2. Standalone single-layer PCC vs the HF reference for any op whose config changed.
3. Full-model greedy generation, token-diffed against the W0 baseline at 4k and 128k.
4. `vllm bench` batch-1 t/s/u at 1k/16k/32k/128k (the single-user headline), then the concurrency
   ladder C=1/8/16 for aggregate — with the C=16→C=1 transition included specifically to prove
   the W1 stall is gone.
5. Re-capture tracy and confirm the `Cores` column actually moved — verify the mechanism, not
   just the outcome.

## Provenance

Written 2026-07-31 from live telemetry taken during the Stage-2 k64 vllm-bench sweep
(`power_watch.log`, `sweep_power_report.log`, both in this directory), the tracy captures under
`doc/optimized_multichip_decoder/tracy/` and `doc/functional_decoder/tracy/`, and a source audit
of `tt/*.py` at commit e3aa655acd5.
