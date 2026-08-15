# LTX-2.5 DiffVAE: mesh sharding and decode performance

The 2.5 model card ships a **diffusion** video decoder, not the convolutional one. Until this
branch the pipeline decoded with the conv VAE, and the DiffVAE could not decode a full clip at
all: replicated across the mesh it ran out of memory somewhere in stage 5.

This branch makes it decode 145 frames of 1088x1920, split across a 4x8 Blackhole galaxy, and
takes that decode from 43.6 s to 19.2 s.

| | before | after |
|---|---|---|
| 145f @ 1088x1920 | OOM | **19.2 s** (same in-pipeline) |
| stage-5 context per chip | 9.69 GB | 0.405 GB |
| `context_and_x` per chip | 19.39 GB | 0.81 GB |
| pcc vs upstream capture | 0.9987 (replicated, crop) | **0.9997** (split, crop) |

Everything below is measured on the full 4x8; the code asserts
`require_exact_physical_num_devices`, so nothing smaller is comparable.

## Why it OOMed

Stage 5 attends over a context volume at patch resolution. At 145 frames that grid is
`(145, 272, 480)` with 256 channels:

| frames | sites | context | `context_and_x` | q/k/v row tables |
|---|---|---|---|---|
| 25 | 3.26 M | 1.67 GB | 3.34 GB | 5.01 GB |
| 145 | 18.93 M | 9.69 GB | 19.39 GB | **29.08 GB** |

Every chip built all of it. 5.8x the frames is 5.8x everything, which is why 25 frames fit and
145 did not.

## The split

H over cluster axis 0, W over axis 1, T left whole — a mesh has two axes, and T is the one that
divides worst (145 frames over 32 devices is 4.5 frames against a 10-frame halo).

Neighborhood attention needs its neighbours, so each device holds a halo. `neighbor_pad_async`
(merged as #52514) delivers it: neighbour rows where a neighbour exists, `padding_mode` at the
volume border.

Four things had to be true for that to work.

**Window bounds are global.** `window_bounds` implements NATTEN's constant-size *inward-shifted*
window — index 0 attends `[0, K)`, not a truncated `[0, K//2]`. A shard that computes bounds on
its own extent treats all 31 interior seams as volume edges and is wrong at every one of them,
silently. `plan_na3d_sharded` computes bounds on the global axis and rebases them into the local
buffer. `test_local_bounds_planning_is_wrong` asserts the mistake still fails.

**The halo follows from the same bounds.** `required_halo` derives it rather than assuming
`k//2`, which under-requests when a shard is narrower than the kernel — a 4-wide shard with an
11-kernel needs 7 on one side.

**Every device needs one dispatch shape.** ttnn runs one program across the mesh, but an edge
shard's windows clamp where an interior shard's slide, so their key spans and grouping diverge —
production stage-5 wanted six distinct shapes. `uniform_spans` holds the key count fixed and
masks the surplus; `plan_na3d_mesh` gives every device the mesh-wide union of geometries, padded
with repeated tiles that are attended and discarded (`TileGroup.real_tiles` keeps them out of the
restore). Sorting that union is also what makes group *i* mean the same thing everywhere, which
`build_mesh_device_plan` relies on when it pairs groups by index.

**One axis at a time.** `neighbor_pad_async`'s fused 2D mode stages corner sticks in an L1 buffer
sized by every outer dimension a core owns; at 145 frames that is 3.8 MB against a 1.5 MB L1 and
stage 5 cannot dispatch. Two sequential 1D passes allocate none, and corners stay correct because
the second pass runs after every device already holds the first pass's halo.

## What did not need changing

**RoPE.** Attention scores depend on the *difference* of two positions, so building tables over
the local buffer shifts every q and k by the same constant and it cancels. The tests place shards
at different offsets so a global-position assumption would fail.

**The upsamples.** A pixel shuffle only redistributes a token's own channels, so a shard's extent
scales by the stride and its boundaries stay where its neighbours' are. No reshard between
stages.

**Where the split starts** is bounded by divisibility, not memory: the latent grid is 34 wide and
a mesh of 4 does not divide it, but it is also the only grid small enough to leave replicated.
`MeshShardConfig.enter_stage` picks the transition; from the first upsample on, everything
divides. Entry is `ttnn.mesh_partition`, the inverse of an all-gather.

## Decode performance

Phase split at 145f, warm, on the second call:

| phase | initial | now |
|---|---|---|
| latent + noise (host) | 4.08 s | ~0.1 s |
| noise packing + upload | 3.96 s | ~0 (cached) |
| **device** | **18.69 s** | **18.69 s** |
| unpack + download | 16.72 s | **0.64 s** |
| total | 43.6 s | **19.2 s** |

Host work fell from 57% of the decode to 4%. Three changes, in order of what they were worth:

1. **`fast_device_to_host` instead of a mesh composer — 16.7 s to 0.6 s.** The composer reads 32
   shards serially and leaves a full-volume cast on host; the async-DMA path reads them
   concurrently, zero-copies, and casts in flight. The conv decoder had been using it all along.
2. **Depth-to-space on device.** `unpack_pixels_device` does the 7-dim permute, so the transfer
   carries finished pixels rather than tile-padded patch channels and both full-volume host
   permutes disappear. The module docstring claimed ttnn could not express this; the conv decoder
   disproves it in its own `decode_device`.
3. **Seeded noise cached.** A fixed seed makes `torch.randn` produce the same tensor every decode,
   which was then repacked and re-uploaded per call.

### Tracing does not help

| | eager | traced |
|---|---|---|
| 145f decode | **19.2 s** | 19.9 s |

Measured both ways in one run. At this size the device half is real compute across 24 attention
blocks, so there is no dispatch gap for a capture to close, and replay still pays to copy its
inputs into the captured buffers. Tracing is off by default; `LTX25_DIFFVAE_TRACED=1` opts in,
and it is worth 1.38x at crop scale where ops are small enough for dispatch to dominate.

Capturing DiffVAE *inside the pipeline*, alongside the DiT and audio traces, wedged the device.
A bisect showed the DiT traces and DiffVAE's eager fabric ops coexist fine, so the fault is in
DiffVAE's own capture. Not pursued, because replay is worth nothing here anyway.

## Against the conv decoder

At 145f: DiffVAE **19.2 s**, conv **0.94 s** eager / 0.58 s traced. About 20x, down from ~46x.

The gap is now entirely `decode_device`. A profile of one NA block at production shard shape puts
it at **38% layout movement** (reshape, permute, tilize, untilize) against **27% SDPA** — the same
shape of problem as the download above, one level down: the time is in moving data between
layouts, not in the arithmetic. Attention is not the bottleneck.

Two candidates, both unstarted:
- Gather straight into the layout SDPA wants, or fold the head permute into the gather indices,
  so `PermuteDeviceOperation` disappears.
- `DEFAULT_SCORE_BUDGET` per stage — tile-aware work is minimised near 2^18 for stage 5 and 2^14
  for the deterministic stages, worth ~2.1x on the SDPA portion. Waste *rises* with the budget,
  and dispatch count is free (12 vs 120 groups differ by under 3% at fixed tile count).
  **But the budget trades compute against memory, not for free**, and the memory side is the
  steeper one. A group's gathered key tensor is `tiles x keys-per-tile x width` and carries no
  query factor, so halving the tile size barely shrinks the key span — the `+k-1` halo dominates
  it — while doubling the tile count. At the production stage-5 shard:

  | budget | tile | tiles | keys | largest gathered buffer |
  |---|---|---|---|---|
  | 2^18 | (3,5,4) | 14994 | 2730 | **11.99 GB** |
  | 2^22 | (5,9,8) | 3190 | 5130 | 3.47 GB |
  | 2^26 | (10,17,15) | 540 | 13500 | **1.44 GB** |

  `LTX25_DIFFVAE_SCORE_BUDGET` sets it. Lower it only for a decode that has to share the device —
  and see below for why that is rarely the right lever.

## In the pipeline

Standalone the decode owns the mesh. In the pipeline it does not, and a 145-frame clip OOMed on
the first four attempts. The score budget looks like the knob for that and is the wrong one: it
buys headroom with arithmetic, and at 2^26 the same decode costs 392 s against 132 s at 2^22
(both cold — the warm figure is below).

The real cause was that the DiffVAE was absent from the co-resident exclusion graph.
`_register_coresident_exclusions` is gated on `dynamic_load` and names the *conv* decoder as the
thing that evicts the active transformer — "LTX-22B + LTX-VAE don't both fit on BH LB" is stated
in that function. The DiffVAE needs the eviction more, since its attention buffers are gigabytes
where the conv decoder's are megabytes, and it inherited none of it.

`LTX25DistilledPipeline._prepare_vae` now pages the transformer, upsampler and Gemma out before
the decode, which is where the conv path already does the same thing:

```
DiffVAE decode: evicted transformer[0], upsampler, gemma — DRAM 20.43 -> 4.20 GB per device
```

**16.2 GB per device**, against the 1.44 GB the decode had been failing to allocate. Everything
reloads from the disk cache at the top of the next generation. It is skipped under tracing: a
captured DiT trace holds pointers into those weight buffers.

The conv decoder's weights are also never loaded when `LTX25_DIFFVAE=1` — 0.81 GB whose only
forward is the `decode_latents` this pipeline overrides. The encoder is a separate reload and is
left alone, so i2v conditioning is unaffected.

With that, a 6.04 s clip at 1088x1920 (145 frames, T2V + audio) end to end:

| phase | warm |
|---|---|
| stage 1 denoise | 16.1 s |
| stage 2 denoise | 8.6 s |
| **DiffVAE decode** | **19.2 s** |
| total (compute) | **46.1 s** |

The decode matches the 19.2 s it takes standalone, so nothing is lost to sharing the device. A
cold decode is 132 s; the difference is JIT, and `RUN_WARMUP=1` moves it off the measured clip.

## Using it

```
LTX25_DIFFVAE=1                 route video decode through the DiffVAE
LTX25_DIFFVAE_PATH=...          required where the 2.5 cache holds only the conv file
LTX25_DIFFVAE_ENTER_STAGE=1     first stage that runs split (divisibility-bound)
LTX25_DIFFVAE_SEED=0            stage 5 predicts x0 from noise, so the noise is an input
LTX25_DIFFVAE_SCORE_BUDGET=...  NA3D tile size; trades dispatched work against peak memory
LTX_YUV_EXPORT=0                required and asserted
```

`LTX_YUV_EXPORT=0` is not optional: the conv decoder's YUV path converts on device from a still
sharded tensor, and DiffVAE returns host pixels. Refused loudly rather than approximated.

## Tests

| file | what it holds |
|---|---|
| `tests/unit/test_na3d_sharded.py` | planner on host: reassembly parity, halo formula, one shape per mesh, and the seam bug as a negative control |
| `tests/unit/test_na3d_mesh.py` | attention across the 4x8 with halos over fabric; `replicate` and `zeros` must agree, which is what proves the border fill is never attended |
| `tests/models/vae/test_diffvae_sharded_block.py` | one NA block, split vs replicated, same module instance |
| `tests/models/vae/test_diffvae_sharded_stages.py` | deterministic stages through two upsamples |
| `tests/models/vae/test_diffvae_sharded_stage5.py` | the diffusion stack |
| `tests/models/vae/test_diffvae_decoder_sharded.py` | whole decoder vs upstream's capture, real weights |
| `tests/models/vae/test_diffvae_decoder_145f.py` | 145f completes, and the phase split |
| `tests/models/vae/test_diffvae_traced.py` | capture and replay, eager vs traced |

## Known gaps

- No upstream capture at 145f, so correctness there rests on crop-scale parity plus shape and
  finiteness. `capture_stages.py` can produce one if the intermediates are worth the disk.
- DiffVAE tracing wedges the device inside the pipeline. Off by default; not diagnosed.
- The 0.64 s download could go the way the conv path went, encoding YUV on device and never
  moving RGB. 3% of the decode, so not before the attention work.
