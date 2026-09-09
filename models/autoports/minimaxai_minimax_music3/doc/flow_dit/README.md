# Stage 05 — flow-matching DiT, condition encoder, Euler scheduler and chunk denoiser on one Blackhole chip

Work log for MiniMax-Music3's diffusion stage (`MiniMaxMusic3Transformer1DModel`,
`MiniMaxMusic3ConditionEncoder`, `FlowMatchEulerDiscreteScheduler`, `MiniMaxMusic3ChunkDenoiseStep`) in TTNN.

* implementation: [`../../tt/flow_transformer.py`](../../tt/flow_transformer.py) (`FlowTransformer`),
  [`../../tt/condition_encoder.py`](../../tt/condition_encoder.py) (`ConditionEncoder`),
  [`../../tt/scheduler.py`](../../tt/scheduler.py) (`FlowMatchEulerScheduler`),
  [`../../tt/denoiser.py`](../../tt/denoiser.py) (`ChunkDenoiser`, `chunk_starts_for`)
* host reference: [`../../reference/flow_transformer_ref.py`](../../reference/flow_transformer_ref.py) (the
  diffusers DiT / condition encoder / `TimestepEmbedding` vendored as plain torch, Apache-2.0 header kept,
  plus safetensors loaders)
* tests: [`../../tests/test_flow_transformer.py`](../../tests/test_flow_transformer.py) (gate),
  [`../../tests/test_flow_transformer_perf.py`](../../tests/test_flow_transformer_perf.py) (Tracy-signposted, `-m slow`)
* scripts: [`../../scripts/dump_scheduler_triples.py`](../../scripts/dump_scheduler_triples.py) (ref venv),
  [`../../scripts/dump_ref_trajectory.py`](../../scripts/dump_ref_trajectory.py) (CPU, per-step fp32 trajectory),
  [`../../scripts/probe_dit_ops.py`](../../scripts/probe_dit_ops.py) (op semantics probe),
  [`../../scripts/collect_dit_perf.sh`](../../scripts/collect_dit_perf.sh) (Tracy + tt-perf-report)
* measured numbers: [`pcc/results.json`](pcc/results.json) (every PCC / timing the gate writes, including the
  per-step drift log), [`tracy/forward/perf_report.{txt,csv,summary.txt}`](tracy/forward/),
  [`pcc/scheduler_triples.pt`](pcc/scheduler_triples.pt) (test input dumped from diffusers, 0.5 MB, committed)
* local-only (gitignored under `generated/`): `ref_trajectory.pt` (21 MB fp32 per-step reference latents; the
  gate falls back to logging drift vs the golden final latent when it is missing), `tt_dit_cache/` (4.6 GB
  converted bf16 weights), `gate05*.log`, `tracy/forward/{pytest.log,ops.csv.gz}`.

## What was built

### `FlowTransformer` (`tt/flow_transformer.py`)

`FlowTransformer.from_pretrained(mesh_device)` loads `transformer/diffusion_pytorch_model-0000{1,2}-of-00002.safetensors`
(fp32, 2.4 G parameters -> bf16 on device, ~4.6 GB) and exposes

| method | contract |
|---|---|
| `forward(latents [2, 128, T], timestep [2], condition [2, T, 2048])` | predicted velocity `[2, 128, T]` (host fp32); row 0 = conditional, row 1 = the zero-condition CFG row, both in ONE pass; any `1 <= T <= 9000` |
| `prepare_condition(condition [2, T, 2048])` | the folded input projection of the condition, `[1, 1, 2 * S_pad, 2048]` on device; constant over a chunk's 30 steps, so `forward(..., cond_proj=...)` skips it |
| `embed_inputs` / `blocks` / `project_out` | the three phases of `forward`, exposed for stage 07 (tracing) |
| `release()` | frees weights and the cached RoPE / mask / selector tensors |

Layout: both batch rows share one tile-aligned activation `[1, 1, 2 * S_pad, 2048]` (bf16, DRAM interleaved),
`S = T + 1` (timestep token at position 0 + `T` latent frames), `S_pad = ceil(S / 128) * 128` (128 = SDPA q/k chunk).
Rows past `S` are zero on input, hidden as attention keys by an additive `-1e9` mask `[1, 1, S_pad, S_pad]` and
sliced away on the host. Head ops see the same buffer as `[2, 1, S_pad, 2048]` through zero-copy
`ttnn.experimental.view` (tile-aligned row split). Per block: `layer_norm` -> fused `[2048, 6144]` QKV matmul ->
`nlp_create_qkv_heads` -> partial RoPE -> `scaled_dot_product_attention` (non-causal, mask, HiFi4) ->
`nlp_concat_heads` -> `to_out` -> residual -> `layer_norm` -> two `[2048, 8192]` matmuls (`a`, `silu(g)`) ->
`a * g` -> `[8192, 2048]` -> residual. Matmuls HiFi2 with fp32 accumulation, LayerNorm HiFi4 fp32.

Algebraic folds, done in fp32 on the host at load time (`fold_input_weights`, `fold_output_weights`):

* `cat(latents, zeros, condition^T) -> preprocess_conv (1x1) + residual -> proj_in` equals
  `x @ ((I + Wc^T) Wp^T)`; the 128 zero channels contribute nothing, so the folded `[2304, 2048]` matrix
  splits into `w_in_latent [128, 2048]` (per step) and `w_in_condition [2048, 2048]` (once per chunk).
* `proj_out -> postprocess_conv (1x1) + residual` equals `h @ (Wo^T (I + Wpost^T)) = w_out [2048, 128]`.
* The timestep embedding (Fourier features -> linear -> silu -> linear, 1.3 M parameters) runs on the host in
  fp32 and is scattered into row `b * S_pad` with a one-hot `[2 * S_pad, 32] @ [32, 2048]` matmul (the
  depth-decoder selector trick), so a step moves only the latents (393 KB) and 32 embedding rows to the device.

Partial RoPE (32 of 64 dims, theta 1e4, rotate-half pairing inside the 32 rotary dims, position 0 = timestep
token) uses `ttnn.experimental.rotary_embedding_llama` in prefill mode with cos/sin tables that are `1 / 0` on
dims 32..63 and a custom 32x32 transformation matrix `x @ M = cat(-x[16:], x[:16])`. The op applies `M` tile by
tile, so the first 32-wide tile of each head rotates exactly like the reference and the second is untouched
(`scripts/probe_dit_ops.py`: PCC 0.999996 vs torch, no weight permutation needed). Batch 2 is folded into the head
dimension (`[1, 64, S_pad, 64]` view) because the prefill kernel takes batch 1.

Weights live in `models.tt_dit` `Parameter`s inside a `Module` tree so `models.tt_dit.utils.cache.load_model`
caches the converted tensors under `TT_DIT_CACHE_DIR/minimax-music3/transformer_l36/CP1_0_TP1_0_SP1_0_mesh1x1_bf16`
(`tests/conftest.py` defaults `TT_DIT_CACHE_DIR` to `generated/tt_dit_cache`): 7.1 s from safetensors on the
first run, 0.7 s from the cache afterwards (the fp32 safetensors are still read for the host-side timestep embedder).

### `ConditionEncoder` (`tt/condition_encoder.py`)

`[1, F, 8 * 4096]` frame hiddens of one window (`F <= 200`) -> `[1, L, 2048]`, `L = int(F * 44100 / 24000 * 960 / 512)`:
softmax layer mix and `layer_scale` on the host (fp32, a 6.5 M-element weighted sum), `Conv1d(4096, 2048, k=3, pad=1)`
as ONE `[F_pad, 12288] @ [12288, 2048] + b` matmul on device over the 3-tap unfold (bf16, HiFi2 fp32-acc), nearest
resampling as a host `index_select` whose index map is taken from `F.interpolate(mode="nearest")` itself on an index
ramp (bit-exact rounding vs the reference). Output is host fp32 because the chunk loop splices and carries windows of it.
**Decision (nobody to ask):** the prompt allows host torch here; the only heavy op (25 M-parameter conv) is on device,
the rest stays on the host on purpose (documented above). 68 ms for the 200-frame window including transfers.

### `FlowMatchEulerScheduler` (`tt/scheduler.py`)

diffusers' `FlowMatchEulerDiscreteScheduler` restricted to the checkpoint config (`num_train_timesteps=1`, `shift=1`,
`invert_sigmas=True`, nothing else on) and the pipeline's `set_timesteps(sigmas=linspace(1, 1/N, N))`: float32
`sigmas = 1 - linspace(1, 1/N, N)`, `timesteps = sigmas[:-1]`, terminal sigma 1.0; `step` = `x + (sigma_next - sigma) v`
in float32, cast back to the velocity dtype. Verified bit-exact (see evidence).

### `ChunkDenoiser` (`tt/denoiser.py`)

`denoise_chunk(frame_hiddens_chunk, previous_latent, previous_condition, noise, steps)` follows
`MiniMaxMusic3ChunkDenoiseStep` block by block: condition splice over `overlap = min(prev_len, L)`, `noise_prompt`
snapshot, per-step overlap blend `(1 - (1 - 1e-6) t) noise_prompt + t previous_latent`, one DiT pass for both CFG
rows, `v = v_u + 1.7 (v_c - v_u)`, Euler step, overlap restore, carry of `latents[..., L-344 : L-172]` and the same
condition window (clamped like the reference for short windows). `denoise(frame_hiddens, noises)` runs all windows
with `chunk_starts_for` = `before_denoise.py`'s formula (`[0]` up to 200 frames, else `range(0, F - 100, 100)`).
Latents, CFG combine and Euler update stay fp32 on the host as in the reference; the DiT input is bf16.

## Hardware and software identity

* board: `tt-smi -s` board id `000004613193411b` (p300c, chip 0 = PCI `0000:01:00.0`, `TT_METAL_VISIBLE_DEVICES=0`,
  reported as P150 by tt-metal; compute grid 11x10 = 110 worker cores), 1x1 mesh, program cache on, no fabric.
* tt-metal: worktree `~/tt-metal-mm3` branch `jashan/minimax-music3`, shared prebuilt binary from `~/tt-metal`
  (base `e946955cc15`); stage-05 code commits `8ba0e4e3cb1` (model code) and the follow-up commits listed by
  `git log --oneline -- models/autoports/minimaxai_minimax_music3/tt/flow_transformer.py`.
* reference: diffusers `040c7cde626504d14caf63b13b8b25b6a9f62120` (ref venv, only used to dump the scheduler
  triples); the DiT reference in the tests is the vendored module in fp32 inside `~/tt-metal/python_env`
  (torch 2.11 cpu). Golden: `~/mm3-bringup/reference/{chunks,frame_hiddens,sigmas}.pt` (fp32, seed 7, 30 steps).
* weights: `MiniMaxAI/MiniMax-Music3` snapshot `fbdf52fbaaca799592917417eb05f1899f1255ec`.

## Commands

```bash
source ~/mm3-bringup/common.sh && cd $MM3_WT
# gate (also: ~/mm3-bringup/checks/05.sh)
with_hw_lock timeout 3600 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_flow_transformer.py -m "not slow" -x -q
# chained two-window run with our own carry (slow marker)
with_hw_lock timeout 3600 $MM3_PY -m pytest $MM3_MODEL_DIR/tests/test_flow_transformer.py -m slow -q
# evidence inputs
$MM3_REF_PY $MM3_MODEL_DIR/scripts/dump_scheduler_triples.py          # doc/flow_dit/pcc/scheduler_triples.pt
$MM3_PY $MM3_MODEL_DIR/scripts/dump_ref_trajectory.py                 # generated/ref_trajectory.pt (~10 min CPU)
# device perf report for one forward
with_hw_lock $MM3_MODEL_DIR/scripts/collect_dit_perf.sh               # doc/flow_dit/tracy/forward/
```

## Evidence

All numbers below are from this stage's runs on the board above; the JSON in `pcc/results.json` is what the
gate wrote on its final run (`generated/gate05_final.log`).

### Correctness

| test | bar | measured |
|---|---|---|
| scheduler sigmas / timesteps vs `sigmas.pt` (both chunks) and vs diffusers for N = 30, 8, 1 | exact | exact (`torch.equal`) |
| scheduler `step` vs 9 dumped diffusers steps (`scheduler_triples.pt`) | max abs err < 1e-5 | 0.0 (bit-exact) |
| condition encoder vs golden `conditions_raw[0]` (200 frames -> 689 latents) | PCC >= 0.999 | PCC 0.999995, max abs err 0.034 (ref rms 0.39) |
| condition encoder vs golden `conditions_raw[1]` (150 frames -> 516 latents) | PCC >= 0.999 | PCC 0.999995, max abs err 0.032 |
| DiT forward, T = 689 (S_pad 768), both rows, t = 0.5, mid-trajectory latent, vs fp32 torch | PCC >= 0.99 | cond 0.99984, uncond 0.99984 (max abs err 0.47, ref rms 2.11; latent = ref_trajectory step 14) |
| DiT forward, T = 100 (S_pad 128) vs fp32 torch | PCC >= 0.99 | 0.99978 |
| DiT forward, T = 37 (S = 38, not a tile multiple, S_pad 128) vs fp32 torch | PCC >= 0.99 | 0.99965 |
| `denoise_chunk` chunk 0 (golden noise, 30 steps) vs golden `latents[0]` | PCC >= 0.98 | PCC 0.99962, max abs err 0.56 |
| `denoise_chunk` chunk 1 (golden noise + golden carry from chunk 0) vs golden `latents[1]` | PCC >= 0.98 | PCC 0.99953 (0.99943 excluding the 172 restored overlap latents), condition splice `torch.equal` to golden `conditions[1]` |
| both chunks chained with OUR carry (`-m slow`, `test_denoise_all_chunks_chained`) | PCC >= 0.98 | PCC 0.99962 / 0.99932 |

Layer-wise: one block gives PCC 0.99997 vs torch (dev run, `generated/dev_dit_1layer.log`); the drop to 0.9998
over 36 blocks is the bf16 residual stream, uniformly spread (no single bad op). The mid-trajectory latent for
the forward test is step 14 of the fp32 reference trajectory when `generated/ref_trajectory.pt` exists (final run),
otherwise the linear blend `0.5 noise + 0.5 golden_final` (first run; PCC identical to 5 digits).

Per-step drift (chunk tests, `pcc/results.json::denoise_chunk{0,1}.drift`): PCC of our latents after every Euler
step against the fp32 reference trajectory started from the same noise/carry. Chunk 0: step 0 1.00000, step 9 0.99983, step 19 0.99966, step 29 0.99962 (min 0.99962). Chunk 1: step 0 1.00000, step 9 0.99979, step 19 0.99959, step 29 0.99954 (min 0.99954). The PCC vs the reference trajectory starts at 1.0 (the first update is small against the noise) and drifts slowly and roughly linearly down to 0.9996 / 0.9995 at step 29: the bf16 per-forward error (PCC 0.9998) accumulates additively over the 30 Euler steps, with no blow-up. Chained run with our own carry: PCC 0.99962 / 0.99932.

### Performance (eager, no trace, DRAM interleaved, host <-> device latents / velocity every call)

| what | measured |
|---|---|
| one DiT forward, B = 2, T = 689 (S_pad 768), warmed, condition projection precomputed | 116 ms median over 5 (min 113 ms); Tracy device time 106.4 ms + 3.3 ms gaps |
| `prepare_condition` (once per chunk) | 2.4 ms |
| one full 30-step chunk 0 (`denoise_chunk`, includes the condition encoder, 30 forwards, host CFG/Euler) | 4.2 s (141 ms per step) |
| one full 30-step chunk 1 (516 latents, S_pad 640) | 3.5 s |
| weight load | 7.1 s from safetensors, 0.7 s from the `TT_DIT_CACHE_DIR` cache |

Tracy device report for one warmed forward (`tracy/forward/perf_report.summary.txt`, 617 device ops between the
`PERF_DIT_FORWARD` signposts): device time 106.4 ms, op-to-op gaps 3.3 ms in total.

| op class | ops | device ms | note |
|---|---|---|---|
| Matmul | 183 | 62.3 | 38-41 % of peak FLOPs on the big ones (`[1536 x 2048 x 8192]` 418 us, `[1536 x 8192 x 2048]` 428 us); tt-perf-report flags "in0 in DRAM" |
| SDPA | 36 | 13.6 | 380 us each, S_pad 768, 32 heads x 2 rows |
| BinaryNg (residual adds, `a * g`, cond add) | 146 | 9.8 | |
| Unary (silu) | 36 | 6.6 | `ttnn.linear(activation="silu")` did NOT fuse: a separate `UnaryDeviceOperation` runs per block |
| LayerNorm | 72 | 4.9 | runs on 48 cores |
| RotaryEmbeddingLlama | 72 | 4.3 | |
| NlpCreateHeads / NLPConcatHeads | 36 + 36 | 3.4 + 1.3 | |

Optimization leads for stage 07 (not done here, correctness first): fuse silu into the matmul (or `mul` with a
fused activation), L1/sharded activations for the block matmuls, trace the 30-step loop (host gaps + transfers
are ~6 ms of a ~112 ms step), LayerNorm on more cores, keep latents on device between steps.

## Decisions taken without anyone to ask

1. **Folded 1x1 convs and split input projection** (exact algebra in fp32; the zero channel block is dropped;
   condition projection hoisted out of the step loop). Verified against the unfolded torch reference at PCC 0.9998.
2. **Timestep embedder on the host** (fp32, 1.3 M parameters, per step) rather than on device: the prompt allows
   fp32 for timestep params and this keeps the tiny matmuls off the device; the token is scattered on device with a
   one-hot matmul so the step's host->device traffic stays small.
3. **Partial RoPE through `rotary_embedding_llama` with a custom rotate-half transformation matrix** instead of
   `tt_dit`'s elementwise `_apply_rope` (`alt_complex_rotate90` pairs adjacent dims, which would need a q/k weight
   column permutation). Probe-verified, no permutation.
4. **`S_pad` = multiple of 128** (the SDPA chunk) rather than 32: at most 127 wasted rows; T = 37 and T = 100 are
   exercised by the gate.
5. **Condition encoder returns host fp32** (splice + carry happen on the host, as in the reference); the DiT folds
   it once per chunk. Only the conv matmul is on device.
6. **Scheduler is host torch** (one axpy per step over `[1, 128, T]`), bit-exact with diffusers.
7. **Chunk 1's gate test uses the golden carry** so its PCC measures chunk 1 alone; the chained run (our carry) is
   the `slow` test and is also recorded.
8. **`*.pt` un-ignored for `doc/flow_dit/pcc/scheduler_triples.pt`** (0.5 MB test input from the ref venv; the
   repo-level `.gitignore` drops `*.pt`).

## Open risks

* bf16 residual stream: 0.9998 per forward and 0.9996 per chunk today; longer windows are capped at 200 frames by
  the pipeline so the sequence length cannot grow, but the vocoder stage will show whether 0.9995 latent PCC is
  audibly clean (stage 06 qualitative check).
* `ttnn.linear(activation="silu")` runs the activation as a separate op on this build (Tracy); harmless for
  correctness, 6.6 ms per forward.
* Only one board / one golden clip; T values covered: 37, 100, 516, 689 (S_pad 128 / 640 / 768). `MAX_LATENTS`
  9000 is asserted but the largest window the pipeline produces is 689.
* Host CPU contention skews eager timings (a concurrent CPU job inflated one forward from 112 ms to 210 ms during
  the first gate run); the numbers above are from a quiet host.
