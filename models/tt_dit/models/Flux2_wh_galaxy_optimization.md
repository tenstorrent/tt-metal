# Flux2 — Wormhole Galaxy Optimization Notes

WH-Galaxy (4x8) inference tuning for Flux2 at 1024x1024, layered on top of the
fused-kernel base branch (`sadesoye/flux2_1024`). This document tracks the
optimization journey, the current state, and the plan to land on `main`.

Config under test: `wh_glx_ring_sp0tp1_fsdp` (mesh 4x8, sp=4 on axis 0, tp=8 on
axis 1, Ring topology, FSDP weight sharding), 1024x1024, 50 denoising steps.

## Performance journey (1024x1024, total pipeline)

| Stage | Total | Notes |
|-------|-------|-------|
| WH baseline (pre-fused-base) | 20.38s | original WH branch |
| After rebase onto fused-kernel base | 22.70s | new fused kernels changed matmul shapes → regression |
| Re-tune AGMM + RS+MM, port bf8 FSDP MLP gather | ~18.0s | WH-specific block sweeps |
| + minor RS+MM tuning | 17.59s | flat, committed |
| + ring-SDPA chunk tuning (res 4608) | 17.11s | quality-neutral, -0.48s |
| + bf8 QKV weight gather | **16.05s** | -1.06s, visually identical |

## Key findings / optimizations

- **The workload is bandwidth-bound by the per-step FSDP weight gather, not
  compute.** Evidence: dropping matmuls to LoFi fidelity gave ~2% and was
  unstable (reverted); halving QKV gather *bytes* gave ~6.7%. The lever that
  matters is reducing gather bytes, not faster math.
- **Ring-SDPA chunk tuning (res 4608):** the single-block sequence
  (spatial+prompt) was silently falling back to the untuned default `(128,512)`.
  A sweep found `(256,512)` is the L1-bounded optimum for WH. Chunk size only
  changes tiling, so this is bit-neutral for output. (`q>=320` / `k>=768` exceed
  the 1.5MB L1 limit.) Knob: `TT_RING_SDPA_CHUNK="q,k"` for future
  single-block-resolution tuning.
- **bf8 QKV weight gather:** `to_qkv` / `add_qkv_proj` gather their FSDP-sharded
  weights in bfloat8_b (`fsdp_gather_bf8=True`), halving their gather traffic.
  These use `minimal_matmul_split` (bf8-compatible). Image quality is visually
  identical (PSNR 32–48 dB vs the bf16 output across the 3 test prompts).
- **`to_out` stays bf16:** its fused AG+matmul+addcmul op requires the weight
  tile size to match the activation tile size, so a bf8 weight against bf16
  activations is rejected (`ternary_a_tile_size == in1_tile_size`). Pushing
  further (toward ~15s) requires **bf8 activations** so the tiles match — the
  proven Wan2.2 recipe ("bf8 weights+activations"), but with more quality risk.

## Dead ends (don't re-try)

- **Matmul LoFi fidelity:** ~2% gain (bandwidth-bound) and deadlocked the fabric.
- **FSDP AllGather prefetch/overlap:** no gain — the SP axis is already saturated
  by ring-joint SDPA + weight gathers sharing the same semaphore pool.
- **2048x2048 on this base:** L1 OOM in `to_qkv` AGMM; needs its own higher-res
  block-size tuning (1024 blockings don't scale).

## How to run

```bash
export TT_DIT_CACHE_DIR=/your/cache/path
huggingface-cli login   # FLUX.2-dev is gated

# Performance (saves output PNGs flux2_4x8_1024x1024_{0,1,2}.png to cwd):
pytest "models/tt_dit/tests/models/flux2/test_performance_flux2.py::test_flux2_performance" \
  -k "1024x1024 and wh_glx_ring_sp0tp1_fsdp" -s
```

Sweep tooling: `models/tt_dit/utils/sweep_mm_block_sizes.py` (AGMM blocks),
`models/tt_dit/utils/sweep_rsmm_block_sizes.py` (RS+MM blocks).

## Quality verification

- Visual: all 3 prompts (cat / portrait / alien-world) are artifact-free and
  indistinguishable from the bf16-QKV output; worst case (busy alien-world
  scene) is 32 dB, the other two 46–48 dB.
- A strict PCC-vs-torch-golden check could not be run on this WH galaxy: the
  `test_transformer` mesh configs that engage FSDP+bf8 fail device init here
  (2x4 sub-mesh ethernet handshake timeout; 4x8 ring uses Blackhole-only fabric
  params). Getting a ground-truth number needs a torch/diffusers reference run.

## Merge-to-main plan

Our work sits on top of the fused-kernel base (`sadesoye/flux2_1024`), which is
not on `main` yet and is still actively changing. Plan, per discussion with the
base owner:

1. Base owner lands the Flux2 fused-kernel work on `main` as a series of smaller,
   reviewable PRs (it touches many kernels).
2. Once the base is on `main`, rebase this WH branch on top and open a follow-up
   PR with the WH tuning. Changes are mostly additive/low-risk: config tables
   (AGMM / RS+MM block sizes), the ring-SDPA `(4608)` chunk entry, and per-layer
   `fsdp_gather_bf8` flags.
3. **Open design question — bf8:** decide whether bf8 stays as per-layer
   `fsdp_gather_bf8` flags or folds into a unified precision/quant config, and
   whether to extend to bf8 activations (unblocks `to_out`, path toward ~15s).
