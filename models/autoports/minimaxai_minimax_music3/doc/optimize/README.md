# Stage 07 — optimize on one chip (latency, precision, tracing)

Work log for the optimized MiniMax-Music3 pipeline: the `dtype_policy="optimized"` preset of `tt/pipeline.py` (now the
default), measured before / after with one harness on the stage-06 golden 10 s clip and a free-running 10 s song.

* implementation: [`../../tt/llm.py`](../../tt/llm.py) (`DTYPE_POLICIES`: bfp8 attention / MLP / LM-head weights, bfp8 KV
  cache, LoFi decode matmuls + LM head; the sweep policies), [`../../tt/depth_decoder.py`](../../tt/depth_decoder.py)
  (`weight_dtype`, explicit 1D-mcast matmul program configs), [`../../tt/flow_transformer.py`](../../tt/flow_transformer.py)
  (`weight_dtype`, matmul fidelity presets, 256-wide SDPA chunks, `DiTStepTrace`: one trace per window shape),
  [`../../tt/denoiser.py`](../../tt/denoiser.py) (traced Euler loop, persistent 200-frame-window shape),
  [`../../tt/ar_generator.py`](../../tt/ar_generator.py) (CFG + top-50 on the 16389-value logits window, candidate-only
  multinomial), [`../../tt/pipeline.py`](../../tt/pipeline.py) (`POLICY_PRESETS`, per-component overrides, vocoder
  overlapped with the next window's denoising, `policy_report`), [`../../tt/vocoder.py`](../../tt/vocoder.py) (the
  TTNN vocoder port: correct, slower than the host, not enabled - see "Vocoder")
* tests: [`../../tests/test_optimized.py`](../../tests/test_optimized.py) (gate); the stage 02-06 tests rerun by the gate
* scripts: [`../../scripts/measure_perf.py`](../../scripts/measure_perf.py) (the before / after harness),
  [`../../scripts/dtype_sweep.sh`](../../scripts/dtype_sweep.sh) + [`dtype_sweep_table.py`](../../scripts/dtype_sweep_table.py),
  [`../../scripts/matmul_config_sweep.py`](../../scripts/matmul_config_sweep.py), [`dit_grid_sweep.py`](../../scripts/dit_grid_sweep.py),
  [`dit_step_timing.py`](../../scripts/dit_step_timing.py), [`probe_sampling_ops.py`](../../scripts/probe_sampling_ops.py),
  [`probe_topk_timing.py`](../../scripts/probe_topk_timing.py), [`probe_depth_heads.py`](../../scripts/probe_depth_heads.py),
  [`vocoder_device_check.py`](../../scripts/vocoder_device_check.py), [`vocoder_trace_probe.py`](../../scripts/vocoder_trace_probe.py),
  [`perf_report_by_op.py`](../../scripts/perf_report_by_op.py); the stage 02-05 Tracy collectors gained `MM3_PERF_OUT` and
  dtype / layer-count / policy knobs (`MM3_LLM_POLICY`, `MM3_LLM_NUM_LAYERS`, `MM3_DEPTH_WEIGHT_DTYPE`, `MM3_DIT_*`)
* measured numbers: [`perf.json`](perf.json) (gate: `before` / `after`), [`perf_runs/*.json`](perf_runs/) (every harness
  run, one file per configuration), [`perf_summary.json`](perf_summary.json), [`dtype_sweep.md`](dtype_sweep.md) /
  [`dtype_sweep.json`](dtype_sweep.json), [`sweeps/*.json`](sweeps/) (op-level micro-sweeps), [`dit/*.json`](dit/),
  [`vocoder/*.json`](vocoder/), [`tracy/*/perf_report.{txt,csv,summary.txt}`](tracy/) (reduced-layer device profiles),
  [`pcc/results.json`](pcc/results.json) (written by the gate tests), [`../pipeline/qualitative/golden_seed7_60s_optimized.json`](../pipeline/qualitative/golden_seed7_60s_optimized.json)
* local-only (gitignored): `generated/perf_*.wav`, `generated/golden_seed7_60s_optimized.wav`, `generated/*.log`,
  `generated/watcher_optimized/`, the raw Tracy `ops.csv.gz` dumps under `doc/optimize/tracy/*/`

Hardware: one chip of the P300x2 host `qbge-devex-02` (`TT_METAL_VISIBLE_DEVICES=0`, `tt-smi -s` board id
`000004613193411b`, p300c reported as P150, 11x10 = 110 worker cores, 12 DRAM banks visible to the DRAM-sharded matmul),
1x1 mesh, program cache on, trace region **200 MB** (90 MB in stages 02-06), no fabric. Software: ttnn from `~/tt-metal` at
`e946955cc15` (shared prebuilt binary; `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` has no
`num_workers_per_dram_bank` in this build), model code in worktree branch `jashan/minimax-music3`, stage 06 at `b8bac93ba28`.
Golden: the stage-01 fp32 diffusers run (seed 7, 10 s, 30 steps, 250 frames, 2 windows). All timings are host wall on an
otherwise idle host (the load average during the runs is the pipeline process itself).

## Headline: before / after with the same harness (`perf.json`, `perf_runs/before.json`, `perf_runs/after.json`)

Free-running 10 s song (seed 7, 30 steps, 250 frames, windows 689 + 516 latents), idle host, warm weight caches:

| | functional (stage 06) | **optimized (stage 07 default)** | change |
|---|---|---|---|
| AR frames/s (realtime 25) | 13.32 | **21.66** | **1.63x** |
| per frame: LLM step / depth loop / host | 37.95 / 33.12 / 3.86 ms | 21.74 / 23.96 / 0.37 ms | -43 % / -28 % / -90 % |
| DiT per window, 30 steps (689 / 516 latents) | 3.51 / 2.77 s | **2.51 / 2.39 s** | **-28 %** |
| host vocoder per window (measured inside its thread) | 6.77 / 4.81 s (sequential) | 8.22 / 4.81 s (overlapped with the next window's denoising) | DiT + vocoder phase 15.5 s instead of 3.51 + 2.77 + 6.77 + 4.81 = 17.9 s |
| total, 10 s clip | 37.1 s | **27.2 s** | **-27 %** |
| 60 s golden-prompt song (`../pipeline/qualitative/`) | 257.6 s | 200.3 s (measured before the DiT full-grid configs, with 2.9 s windows) | -22 % |
| resident DRAM after load | 21.74 GB | **15.13 GB** | -6.6 GB |
| load with warm caches (incl. AR + DiT warm-up) | 9.3 s | 13.3 s | + the DiT trace capture |

Golden replay (teacher-forced codes + golden noises, 30 steps; the path-matched precision evidence):

| | bar | functional | **optimized** |
|---|---|---|---|
| frame-hidden PCC vs golden (per-frame minimum) | >= 0.98 | 0.99941 (0.99512) | **0.99903 (0.97895)** |
| latent PCC window 0 / 1 | >= 0.98 | 0.99951 / 0.99923 | **0.99903 / 0.99868** |
| stitched wav vs golden: log-mel RMS / wav PCC | <= 2.0 dB | 0.754 dB / 0.99875 | **1.267 dB / 0.99855** |
| AR frames/s (teacher-forced, no sampling) | | 14.3 | 22.5 |

The free-running optimized clip: 250 frames, 178 distinct semantic codes, most common 4.4 % (bar 30 %), longest run 6,
repeated 4-grams 0.8 %, RMS -20.5 dBFS, peak 0.87, no silent second, spectral flatness 0.0055 (`perf_runs/after.json`).
Roofline accounting for the AR frame (device-bound): LLM 21.7 ms wall vs 16.0 ms at the 512 GB/s DRAM peak for its 8.2 GB
(the matmuls run at 85-89 % of DRAM bandwidth, `tracy/llm_decode_1layer_optimized_lofi`); depth 24.0 ms wall vs 20.7 ms
device time (`tracy/depth_bfp8_configs_traced`; 1.14 GB of bfp8 weights per step x 7 = 4.0 GB -> 7.8 ms at the peak, the
matmuls reach 56-68 %, the 2-core head split / merge adds 4.2 ms); host 0.4 ms. `perf_summary.json` holds the same figures.

## Baseline (stage 06 code, `dtype_policy="functional"`, this stage's harness: `perf_runs/before.json`)

| | golden replay 10 s | free-running 10 s (seed 7) |
|---|---|---|
| AR frames/s (LLM step + depth loop + host per frame) | 14.3 | **13.3** (38.0 + 33.1 + 3.9 ms) |
| DiT per window, 30 steps (689 latents / 516 latents) | 3.50 s / 2.81 s | **3.51 s** / 2.77 s |
| host vocoder per window (fp32, 12 threads) | 6.79 s / 4.93 s | 6.77 s / 5.11 s |
| total | 35.9 s | **37.1 s** |
| frame-hidden PCC vs golden (min per frame) / latents w0 / w1 / log-mel RMS / wav PCC | 0.99941 (0.99512) / 0.99951 / 0.99923 / 0.754 dB / 0.99875 | — |

Identical to the stage-06 numbers to every printed digit (same code, same seeds), so the two harnesses agree.

## What was done, in measurement order

### 1. Operation-topology audit (Tracy, reduced-layer variants; `tracy/`)

| path | device time | dominant items (functional policy) |
|---|---|---|
| backbone decode step, 1 layer + norm + LM head (`llm_decode_1layer_functional`, x36 layers = the 37 ms step) | 2.25 ms | LM head 12 x [32 x 4096 x 16032] bf16 DRAM-sharded on **12 cores**: 1.61 ms at 95 % DRAM; per layer 0.42 ms of DRAM-sharded matmuls at 53-59 % DRAM (compute-bound at HiFi2 on 12 cores) |
| depth frame, traced (stage-03 report) | 29.5 ms | 170 matmuls with bf16 weights at 74-78 % DRAM (22.4 ms), `nlp_create_qkv_heads` 3.2 ms on 2 cores, `nlp_concat_heads` 1.1 ms |
| DiT forward, 36 blocks (stage-05 report) | 106.4 ms | matmuls 62 ms (39-41 % of peak FLOPs, `in0_block_w = 1`), SDPA 13.6 ms (128-wide chunks), eltwise adds / mul 9.8 ms, unfused SiLU 6.7 ms, LayerNorm 4.9 ms, RoPE 4.3 ms, head split / merge 4.8 ms |
| DiT Euler step, host wall (`dit/step_timing_bfp8_l36_T689.json`) | eager 113.6 ms, traced 113.5 ms, replay-only 112.9 ms | the step is device-bound; tracing alone removes nothing measurable |
| host vocoder | 6.8 s per 689-latent window (torch fp32, 12 threads) | the largest per-window item of the baseline |

The AR loop is device-bound too (stage 04: device time = 94-97 % of wall). So the levers were bytes (bfp8 weights and KV
cache), math passes (fidelity), program configs, and hiding the host vocoder - not dispatch.

### 2. Backbone (`tt/llm.py`)

* `optimized` policy: WQKV / WO / FF1-FF3 / FF2 **bfp8**, KV cache **bfp8**, LM head **bfp8** (`Transformer(dtype=...)` reaches
  the LM head only; the embedding is always bf16 in tt_transformers), norms / activations bf16. tt_transformers' own
  fill / update path handles the bfp8 cache (`paged_fill_cache` receives typecast K / V, `paged_update_cache` bf16).
  Result (`perf_runs/opt_try1.json`, HiFi2 everywhere): LLM step 38.0 -> 33.1 ms; frame-hidden PCC 0.99922 (min 0.99392).
* Profile of that step (`tracy/llm_decode_1layer_optimized`): the LM head and MLP matmuls now run at **53 % DRAM / 53 % FLOPs**
  on 12 cores - they became compute-bound at HiFi2 once the weights halved (the bf16 LM head had been at 95 % DRAM).
  This build has no `num_workers_per_dram_bank` (OPT-015), so the fix is the fidelity: **LoFi for the decode matmul groups
  and the LM head** (`_LOFI_DECODE`; the LM head's compute config is overridden after construction because
  `tt_transformers.LMHead` hard-codes HiFi2). Profile (`tracy/llm_decode_1layer_optimized_lofi`): LM head 1.76 ms at 87 % DRAM,
  MLP matmuls 85-89 % DRAM, per step 2.58 ms for 1 layer + head. Model level: **LLM step 33.1 -> 21.8 ms** (teacher-forced,
  `perf_runs/opt_lofi.json`); frame-hidden PCC 0.99903 (min per frame 0.97895), latents 0.99927 / 0.99897, log-mel 0.784 dB.
  Roofline of the step: 7.3 GB decoder weights + 0.82 GB LM head + KV reads at ~350 positions (0.1 GB) = 8.2 GB; at the
  512 GB/s DRAM peak 16.0 ms; the matmuls reach 85-89 % of it, the step 372 GB/s effective.
* Micro-sweep of the same shapes as interleaved 1D-mcast matmuls on 100 cores (`sweeps/llm.json`): [32 x 4096 x 12288]
  150 us (356 GB/s) vs the DRAM-sharded 185 us in the model - a further ~3 ms/step would need replacing tt_transformers'
  decode matmul program configs and weight memory configs (open lead, not done: it changes the residual / sharded
  activation contract of `models/tt_transformers`).

### 3. Depth decoder (`tt/depth_decoder.py`)

* **bfp8 transformer weights** (`weight_dtype`; embedding, position embedding, norms and the one-hot selectors stay bf16,
  activations bf16): device time per frame 29.5 -> 23.3 ms (`tracy/depth_bfp8_traced`), but the matmuls fell from 74-78 %
  to 40-67 % DRAM utilisation with the default program (`in0_block_w = 2`).
* Program-config micro-sweep (`sweeps/depth.json`, `sweeps/depth12k.json`): a 1D-mcast config on 100 cores with
  `in0_block_w = 4` gives [64 x 4096 x 4096] 85 -> 59 us, [64 x 4096 x 6144] 102 -> 92 us, [64 x 6144 x 4096] 124 -> 86 us,
  [32 x 4096 x 4096] 76 -> 59 us; for N = 12288 (wqkv) and 7168 (heads) the default already reaches 320-340 GB/s and the 1D
  configs tie or lose, so those keep the default. LoFi and fp16 accumulation change nothing here (bandwidth-bound). Applied
  in `_program_config` (N <= 6144 only): device time per frame **20.7 ms** (`tracy/depth_bfp8_configs_traced`), wall
  22.3 ms (33.1 in the baseline).
* Head split / merge (`sweeps/depth_heads.json`): `nlp_create_qkv_heads` 113 us and `nlp_concat_heads` 38 us run on 2 cores
  (one tile row per batch row); the alternatives - slice + reshape + `permute` (159 us / 57 us) and
  `split_query_key_value_and_split_heads` / `concatenate_heads` (113 us / 38 us, the same kernels) - are not faster. Kept;
  4.2 ms per frame (20 %) remain in these two ops.

### 4. AR sampling (`tt/ar_generator.py`)

The stage prompt asks for CFG + top-k on device. Measured (`probe_topk_timing.py`, `probe_sampling_ops.py`): the ops are
numerically fine (`ttnn.topk` k = 64 over the 16416-wide window matches torch, `ttnn.gather` of both CFG rows at the
conditional row's indices is exact) but **`topk` costs 5.06 ms and `gather` 5.13 ms** (10.2 ms traced) for the 16416-wide
window, 0.27 + 2.57 ms for an 8192-wide one, while reading the whole `[32, 16416]` bf16 window back costs **0.18 ms**.
On-device top-k is therefore rejected with evidence (it would add ~10 ms to a 45 ms frame). Instead the host path was
restructured: the mask, CFG (1.5), conditional top-50 restriction (ties kept) and the sampling top-50 now run on the
16389-value window (`_guided_window`, identical candidate sets and probabilities to the 200k-wide reference arithmetic) and
the final draw is a **multinomial over the <= 50 candidates** (`sample_top_k_candidates`, the "final 50-way multinomial on
host" the prompt allows; the depth steps use the same function over their 1024 logits). Host time per frame 3.9 -> ~1 ms
(free-running). Consequence, recorded: the random-number consumption differs from diffusers' `_sample_top_k` (one draw over
<= 50 categories instead of one over 200000 / 1024 categories), so free-running sample paths for a given seed differ from
the stage-04/06 ones while the distribution is identical (`test_candidate_multinomial_matches_reference_distribution`:
4000 seeded draws per width, empirical vs reference probabilities within 0.03, every draw inside the candidate set).

### 5. DiT (`tt/flow_transformer.py`, `tt/denoiser.py`)

* **bfp8 block weights** (`weight_dtype`; norms, biases and the folded in / out projections stay bf16): no speed change at
  HiFi2 with fp32 accumulation (2.91-2.94 s per window for bfp8 and bf16 alike, `dtype_sweep.md`) - the matmuls are
  compute-bound; kept for the 2.4 GB of DRAM it frees (DiT 4.95 -> 2.5 GB).
* **Trace per window shape** (`DiTStepTrace`): persistent latent / timestep-row / condition-projection / velocity buffers,
  a compile run, then the capture of the whole forward (latent projection + condition + timestep token -> 36 blocks -> `w_out`);
  per step the host writes 393 KB + 128 KB and reads 393 KB. Replay equals the eager forward to PCC 1.000000
  (`dit/step_timing_bfp8_l36_T689.json`; `test_dit_trace_matches_eager` also checks determinism and that changed inputs
  change the output). The 200-frame-window shape is allocated in `load` before the AR traces exist and captured in `warm`;
  other shapes (the last window of a song) are captured on first use and released with their RoPE / mask / selector tensors
  at the end of the song (`clear_caches(keep_latents=...)`), keeping the stage-04 trace-lifetime rule. Wall per step is
  unchanged (113 ms; the step is device-bound), so the trace is kept for the ~3 ms of dispatch gaps and the host-side
  headroom it gives the overlapped vocoder, not for a measured speedup.
* **Matmul fidelity** (`sweeps/dit.json`, `sweeps/dit_grid.json`, model level `perf_runs/opt_dit_*.json`): fp32 accumulation
  halves the destination-register capacity; HiFi2 with **fp16 accumulation** runs [1536 x 2048 x 8192] in 340 us instead of
  457 (default program) and the full window in **2.90 s instead of 3.46 s**; latents 0.99893 / 0.99916, log-mel 0.949 dB
  (inside the stage-06 bf16-vocoder control band of 1.115 dB). LoFi: 2.74 s but latents 0.99811 / 0.99788 and log-mel
  1.468 dB (outside the control band) - rejected.
* **Program configs** (`sweeps/dit.json`, `sweeps/dit_grid.json`): explicit 2D-mcast configs on 8x8 / 8x6 grids lose to the op
  default, which already spreads the matmul over the full 11x10 grid with padded per-core blocks but `in0_block_w = 1`.
  The same full-grid layout with **`in0_block_w = 8`**, the largest legal subblock (8 tiles with fp16 accumulation) and fp16
  accumulation wins every block shape: [1536 x 2048 x 8192] 335 -> 297 us, [1536 x 8192 x 2048] 382 -> 249 us,
  [1536 x 2048 x 6144] 259 -> 223 us, [1536 x 2048 x 2048] 115 -> 82 us (`_matmul_config`, shapes with >= 10 tile rows;
  the fp32-accumulating stage-05 configuration keeps the op defaults because its 4-tile subblock cap was not swept on the
  full grid). **SiLU**: `a * silu(g)` as a multiply with the SiLU as input activation costs 226 us instead of 364 us for
  the separate unary + multiply (`silu_mode = "multiply"`; fusing it into the g matmul's program config is equivalent,
  82.4 vs 81.8 ms per step; the fused SiLU is numerically exact - `perf_runs/after_k8_unary.json` equals the fused run to
  every digit). Whole 36-block step (`dit/step_timing_*.json`, T 689, traced): fp16 accumulation with default configs
  96.4 ms, + full-grid configs (`in0_block_w = 8`) 85.6 ms, + fused SiLU **81.8 ms** (113.6 ms in the stage-05
  configuration; the traced output equals the eager one to PCC 1.000000 in every variant). **Precision vs block size**
  with fp16 accumulation, golden replay (`perf_runs/after_k{2,4,8}.json`, sweep row `sweep_mlp-bfp8_kv-bfp8_dit-bfp8` for
  the default): `in0_block_w` 1 (default program) 2.91 s per window / log-mel 0.982 dB; 2: 2.58 s / 1.256 dB; 4: 2.54 s /
  1.267 dB; 8: 2.48 s / 1.345 dB (latents 0.9986-0.9990 throughout). Chosen **`in0_block_w = 4`**: 13 % faster than the
  default program for +0.29 dB, while 8 buys another 2 % for +0.08 dB (`MM3_DIT_IN0_BLOCK_W` overrides).
* **SDPA** (`sweeps/sdpa.json`, S_pad 768, B 2 x 32 heads x 64): 256-wide q / k chunks 323 us vs 392 us for the 128-wide
  stage-05 chunks (HiFi4 kept; HiFi2 saves 2 %, not taken); used whenever `S_pad % 256 == 0`, else 128.
* Not fused: the residual adds (BinaryNg, ~60 us each) and the (now SiLU-carrying) multiply run at ~440 GB/s, i.e.
  bandwidth-bound; LayerNorm (2 x 70 us per block, 48 cores), RoPE (2 x 60 us) and the head split / merge (140 us) stay as
  in stage 05.

### 6. Vocoder (`tt/vocoder.py`, `vocoder/*.json`)

Ported on `models/tt_dit/layers/audio_ops.py` (`Conv1dViaConv3d` for every conv incl. the dilated 7-taps,
`ConvTranspose1dViaConv3d` for the 8 / 8 / 4 / 2 upsamplers - the reference's `padding = ceil(stride / 2)` equals the class's
`(k - stride) // 2` for these even strides, asserted - and `Snake`), fp32, structure mirroring the torch module so the folded
state dict loads by name. **Accuracy on the golden windows: wav PCC 0.999984 / 0.999982, log-mel 0.16 / 0.17 dB, max abs
error 0.035** (bar: PCC >= 0.99) - the DAC padding convention is right. **Time: 16.8 s / 12.5 s per window (warm) vs 7.3 s /
5.6 s on the host** (`vocoder/device_check_fp32.json`). Tracy (`tracy/vocoder_fp32`): 16.7 s of device time, **16.06 s in the
31 `conv3d` ops** (fp32 conv3d with the conservative `C_in_block = C_out_block = 32, T_out_block = 1` blocking over
T up to 352768 samples) - the port is device-bound in conv3d, not dispatch-bound. bf16 fails to compile its conv3d program
(circular buffers exceed L1 with the default blocking, `generated/vocoder_check_bf16.log`); tracing the fp32 graph fails
because tt_dit's zero-pad / zero-stuff helpers create tensors with host writes inside the graph
(`Writes are not supported during trace capture`, `vocoder_trace_probe.py`). Decision: **keep the host fp32 vocoder** (the
prompt's fallback) and record the evidence; the open lead is a `ttnn.conv2d`-based 1D conv (height-sharded, tuned blocking) or
conv3d blocking tables for these (C, T) shapes.

What was done instead: the host vocoder of window k now runs in a worker thread **while the device denoises window k + 1**
(`MiniMaxMusic3Pipeline.generate`; the worker never touches the device, results are consumed in order, the audio is
bit-identical to the sequential path - `test_same_seed_determinism` still passes). On the 60 s golden-prompt song the
DiT + vocoder phase went from 50 + 94 = 144 s to 130 s; the vocoder itself slows from 6.7 to 9.1 s per window under the
contention (the DiT loop's host thread polls for read-backs), so the saving is smaller than the overlap could give.

### 7. Datatype sweep (`scripts/dtype_sweep.sh`, `dtype_sweep.md`)

{LLM MLP bfp4 / bfp8 / bf16} x {KV bfp8 / bf16} x {DiT bfp8 / bf16}, bfp8 attention + LM head and LoFi decode in every
cell (the winning fidelity above), one process per cell, scored on the golden 10 s replay. Control band: latent PCC >= 0.98
per window and log-mel RMS <= 2.0 dB (the stage-06 bars), preferring cells within the 1.115 dB torch-bf16-vocoder control.
The table is in [`dtype_sweep.md`](dtype_sweep.md) (8 of the 12 cells ran; the DiT rows there are the fp16-accumulation
default-config variant, before the full-grid configs and the fused SiLU). Reading: **bfp4 MLP** stays inside the 2.0 dB bar
but **outside the bf16 control band** (frame-hidden PCC 0.9905 with a per-frame minimum of 0.914-0.918, latents 0.9924-0.9957,
log-mel 1.18-1.23 dB) for +1 % AR speed - the LoFi decode step is bound by its *other* bytes (LM head, attention, KV), so halving
the MLP bytes buys little; **bf16 MLP does not run on P150**: the prefill MLP matmul's circular buffers grow to 1.82 MB > 1.5 MB
L1 (`generated/sweep_mlp-bf16_*.log`, the same limit stage 02 met with fp32-accumulating bfp8; tt_transformers sizes that
program for bfp8 weights) - all four bf16-MLP cells are infeasible, not slow; **KV bf16 vs bfp8** and **DiT bf16 vs bfp8** are
speed ties (the KV cache is small at 350 positions; the DiT is compute-bound) with bf16 slightly more accurate (frame-hidden
minimum 0.984 vs 0.979 with bf16 KV). Chosen: **MLP bfp8, KV bfp8, DiT bfp8** = the fastest cell within the control band,
with bfp8 KV for the long-context case (10240 positions: 1.5 GB of KV reads per step in bf16, half in bfp8) and bfp8 DiT for
the 2.4 GB of DRAM. This is `POLICY_PRESETS["optimized"]`, the pipeline default; `"functional"` and every sweep cell remain
selectable (`load(dtype_policy=..., llm_policy=..., dit_dtype=..., dit_fidelity=..., depth_dtype=..., vocoder=...)`).

## Commands

```bash
source ~/mm3-bringup/common.sh && cd $MM3_WT
# before / after (the gate reads doc/optimize/perf.json)
with_hw_lock timeout 3000 $MM3_PY $MM3_MODEL_DIR/scripts/measure_perf.py --label before --policy functional --write-perf-json before --trace-region 90000000
with_hw_lock timeout 3000 $MM3_PY $MM3_MODEL_DIR/scripts/measure_perf.py --label after --policy optimized --write-perf-json after
# datatype sweep (12 processes) and its table
nohup bash $MM3_MODEL_DIR/scripts/dtype_sweep.sh > $MM3_MODEL_DIR/generated/dtype_sweep.log 2>&1 &
$MM3_PY $MM3_MODEL_DIR/scripts/dtype_sweep_table.py
# op-level sweeps / probes
with_hw_lock timeout 1800 $MM3_PY $MM3_MODEL_DIR/scripts/matmul_config_sweep.py --group depth   # depth12k | dit | sdpa | llm
with_hw_lock timeout 1200 $MM3_PY $MM3_MODEL_DIR/scripts/dit_grid_sweep.py
with_hw_lock timeout 1200 $MM3_PY $MM3_MODEL_DIR/scripts/dit_step_timing.py --weight-dtype bfp8
with_hw_lock timeout 600  $MM3_PY $MM3_MODEL_DIR/scripts/probe_topk_timing.py
with_hw_lock timeout 1800 $MM3_PY $MM3_MODEL_DIR/scripts/vocoder_device_check.py --dtype fp32
# reduced-layer device profiles (tt-perf-report; one run at a time, never with the watcher)
MM3_LLM_POLICY=optimized MM3_LLM_NUM_LAYERS=1 MM3_PERF_OUT=$MM3_MODEL_DIR/doc/optimize/tracy/llm_decode_1layer_optimized_lofi with_hw_lock bash $MM3_MODEL_DIR/scripts/collect_llm_perf.sh decode
MM3_DEPTH_WEIGHT_DTYPE=bfp8 MM3_PERF_OUT=$MM3_MODEL_DIR/doc/optimize/tracy/depth_bfp8_configs_traced with_hw_lock bash $MM3_MODEL_DIR/scripts/collect_depth_perf.sh traced
MM3_DIT_WEIGHT_DTYPE=bfp8 MM3_DIT_FIDELITY=hifi2_fp16 MM3_DIT_NUM_LAYERS=1 MM3_DIT_TRACED=1 MM3_PERF_OUT=$MM3_MODEL_DIR/doc/optimize/tracy/dit_1layer_bfp8_fp16acc_traced with_hw_lock bash $MM3_MODEL_DIR/scripts/collect_dit_perf.sh
$MM3_PY $MM3_MODEL_DIR/scripts/perf_report_by_op.py $MM3_MODEL_DIR/doc/optimize/tracy/<run>/perf_report.csv
# qualitative 60 s song, watcher, gate
with_hw_lock timeout 3000 $MM3_PY $MM3_MODEL_DIR/scripts/generate_song.py --preset golden --seed 7 --duration 60 --name golden_seed7_60s_optimized
TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 TT_METAL_LOGS_PATH=$MM3_MODEL_DIR/generated/watcher_optimized with_hw_lock timeout 3600 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_optimized.py -k "golden_replay or short_song or dit_trace" -q -p no:cacheprovider
bash ~/mm3-bringup/checks/07.sh
```

Note on the gate script: `~/mm3-bringup/checks/07.sh` as delivered called `bash "$G"tests/test_optimized.py` and
`bash "$G""$t"` (no space after the helper path), which can only fail with "No such file or directory"; the two spaces were
added (original kept as `07.sh.orig`) before the recorded gate run. Nothing else in the checks was changed.

## Evidence

### Gate tests (`tests/test_optimized.py`, then the stage 02-06 test files; `~/mm3-bringup/checks/07.sh`, `generated/gate07_final.log`)

| test | bar | measured (`pcc/results.json`) |
|---|---|---|
| `test_candidate_multinomial_matches_reference_distribution` (host) | every draw inside the reference candidate set; empirical vs reference probabilities within 0.03 over 4000 seeded draws (widths 1024 and 16389) | pass |
| `test_perf_json_before_after` (host) | `after` faster than `before` on AR frames/s and DiT chunk time; `after` accuracy within the bars | pass (numbers in the headline table) |
| `test_policy_report` | the loaded components report bfp8 attention / KV / LM head, bfp8 depth and DiT weights, traced DiT | pass |
| `test_golden_replay_optimized` | frame-hidden PCC >= 0.98, latent PCC >= 0.98 per window, log-mel RMS <= 2.0 dB | see headline table |
| `test_dit_trace_matches_eager` | traced step vs eager forward PCC >= 0.9999, deterministic replay, changed inputs change the output | PCC 1.000000, max abs 0.0 |
| `test_short_song_runs` | 2 s / 6-step song end to end | pass |
| stage 02-06 files (`test_llm`, `test_depth_decoder`, `test_ar_generator`, `test_flow_transformer`, `test_pipeline`) | their own bars; `test_pipeline` now runs the optimized default (golden replay frame-hidden PCC >= 0.99 overall, latents >= 0.98, log-mel <= 2.0 dB) | all green in the gate run (the stage 02-05 fixtures build their components in the stage 02-05 configurations, so those bars are unchanged) |

### Watcher

`TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1` over `test_golden_replay_optimized`, `test_dit_trace_matches_eager` and
`test_short_song_runs` (the full optimized pipeline: load, warm-up, the teacher-forced 10 s replay through the LoFi backbone
trace, the bfp8 depth traces, the traced fp16-accumulation DiT, the overlapped vocoder): **3 passed in 164.7 s**
(`generated/watcher_optimized.log`; the replay reports the same PCC / log-mel as the plain run to every printed digit);
`generated/watcher_optimized/generated/watcher/watcher.log` has 18649 lines and **zero** matches of
`exception | assert | sanitiz | overflow | fault | hang | stall | error`. Watcher and profiler runs were never combined.

### Qualitative: the 60 s golden-prompt song under the optimized policy (`../pipeline/qualitative/golden_seed7_60s_optimized.json`)

Same prompt, seed 7, 30 steps as the stage-06 song (`golden_seed7_60s.json`): 1500 frames (`max_frames`), 14 windows,
60.07 s of audio, **total 200.3 s vs 257.6 s** (AR 70.1 s at 21.4 frames/s vs 113.5 s at 13.2; DiT 43.5 s vs 49.9 s -
before the full-grid configs and the fused SiLU; DiT + overlapped vocoder phase 130 s vs 50 + 94 s). Statistics side by
side (stage 06 -> stage 07): RMS -17.5 -> -17.6 dBFS, no silent second, stereo correlation 0.73 -> 0.60, spectral flatness
0.0051 -> 0.0040 (white noise 1.0, the golden fp32 clip 0.023), band energy < 250 Hz / 250-2 kHz / 2-8 kHz / > 8 kHz
0.50 / 0.43 / 0.06 / 0.01 -> 0.57 / 0.41 / 0.01 / 0.005, adjacent-second log-mel correlation mean 0.87 -> 0.88 (max 0.99 -> 0.99);
codes: 768 -> 753 distinct semantic codes, most common 1.7 % -> 2.0 % (bar 30 %), adjacent repeat 11.8 % -> 17.2 %,
longest run 5 -> 8 frames, repeated 4-grams 1.4 % -> 2.5 %, no full-frame repeat. The optimized song is a different
sample path (candidate multinomial, bf16 / LoFi logits), a little more repetitive and darker (less energy above 2 kHz)
than the stage-06 song but far from degenerate; the path-matched precision evidence is the golden replay (headline table).
Listening is not possible in this headless run.

## Decisions taken without anyone to ask

1. **CFG + top-k stays on the host** (measured: on-device `topk` + `gather` 10 ms vs a 0.18 ms read-back), restructured to
   the logits window with a candidate-only multinomial - the prompt's explicit fallback. Random-number consumption
   differs from the reference; the distribution is identical (tested).
2. **LoFi for the backbone decode matmuls and LM head** (the only way to make the 12-core DRAM-sharded matmuls
   bandwidth-bound in this build): frame-hidden PCC 0.99903 overall with a per-frame minimum of 0.979 (one frame; 0.984
   with a bf16 KV cache, 0.994 at HiFi2) against the prompt's 0.98 bar on the (overall) frame-hidden PCC. `optimized_hifi2`
   keeps the HiFi2 variant selectable.
3. **DiT: HiFi2 with fp16 accumulation, full-grid configs with `in0_block_w = 4`, fused SiLU, 256-wide SDPA chunks**; LoFi and
   `in0_block_w = 8` rejected / not chosen on the log-mel evidence above. bfp8 block weights kept although they do not
   speed the compute-bound matmuls (2.4 GB of DRAM).
4. **Vocoder stays on the host** (TTNN port 2.3x slower, device-bound in fp32 conv3d; the port and its evidence are kept
   in the tree, `vocoder="device"` loads it), overlapped with the next window's denoising instead.
5. **bf16 MLP cells of the sweep are infeasible on P150** (prefill circular buffers exceed L1), recorded as such rather
   than as slow.
6. **Trace region 200 MB** (context contract): backbone + depth + one 36-block DiT trace per window shape; the trace-
   lifetime rule of stage 04 is kept by allocating the 200-frame-window DiT buffers before the AR traces and releasing
   per-song shapes at the end of each song.
7. The **stage-04 AR test bars are unchanged** (they run the functional backbone); the optimized policy is tested by
   `test_optimized.py` and by `test_pipeline.py` (whose golden-replay bar of 0.99 overall frame-hidden PCC the optimized
   pipeline meets at 0.99903).

## Open risks / hand-off

* Realtime is 25 frames/s; the AR loop runs at 21.7 (LLM 21.7 ms at the DRAM roofline of its bytes, depth 23.9 ms of which
  4.2 ms are the 2-core head split / merge, host 0.4 ms). Next levers: interleaved 1D-mcast decode matmuls instead of the
  12-core DRAM-sharded ones (needs tt_transformers config surgery; 150 vs 185 us per MLP matmul measured),
  `num_workers_per_dram_bank` on a newer tt-metal, a faster head split for 2 tile rows.
* The per-frame frame-hidden PCC minimum of 0.979 under LoFi (decision 2) and the 1.27 dB log-mel of the replay (0.75 dB
  in stage 06) are the precision cost of this stage; both are inside the stage-06 bars but outside the 1.115 dB
  bf16-vocoder control the stage-06 log used as a reference point.
* Free-running sample paths changed (candidate multinomial + LoFi logits): the 60 s song is a little more repetitive
  (longest semantic run 8 frames, 4-gram repeats 2.5 %); the end-token behaviour was re-verified only through the stage-04
  test file in the gate (functional backbone) - the optimized backbone's end-token statistics are not measured here.
* The overlapped host vocoder runs 35 % slower under contention with the DiT loop's polling thread (9.1 vs 6.7 s per
  window); pinning threads or a device vocoder would recover it. AR / DiT window pipelining (start denoising window k while
  the AR emits later frames) is a further ~40 s on a 60 s song but needs every DiT-phase temporary to be a persistent
  buffer allocated before the AR traces (trace-lifetime rule) - not done.
* The TTNN vocoder needs a conv2d-based 1D conv or conv3d blocking tables for (C, T) = (1536..96, 689..352768) to be
  competitive; its trace needs device-side padding instead of `ttnn.zeros`.
* The first request of a new window length still compiles the DiT programs and captures a trace (window 1 of the golden
  replay: 4.9-5.6 s instead of 2.4 s in every fresh process); the 200-frame window is warmed in `load`.
