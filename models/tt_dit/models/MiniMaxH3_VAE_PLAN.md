# MiniMax-H3 VAEs on BH Galaxy 4x8

**Scope changed 2026-08-03.** This plan was a full FL2VA bringup (DiT + encoder +
VAEs + pipeline). It is now **VAE-only**: the video VAE (encode and decode) and the
audio VAE (encode and decode), each with comprehensive unit tests and measured
performance. The DiT and text-encoder work already done is preserved and parked;
see "Parked work" at the end.

Branch: **`kevinmi/minimax-h3-vae`, cut from `origin/cglagovich/minimax-h3`.**
That branch owns the canonical folder structure and the pinned diffusers reference,
and this work conforms to it rather than inventing a layout.

## Context

MiniMax-H3 generates video and its 32 kHz stereo soundtrack jointly. Two VAEs sit
either side of the denoiser:

- **Visual VAE** (`AutoencoderKLMiniMaxH3`, 10.4 GB, 560 tensors, **all fp32**):
  f16t4d24 — 16x spatial, 4x temporal, 24 latent channels. Causal **CNN encoder**;
  a separately trained **36-layer ViT decoder**.
- **Audio VAE** (`AutoencoderKLMiniMaxH3Audio`, 0.6 GB): 32 kHz to a **40 Hz**
  latent at 32 channels. One encoder/decoder shared by both channels, each processed
  independently then recombined, so stereo is simply batch 2. DAC-style encoder,
  BigVGAN vocoder.

Nothing in `models/tt_dit` implements either.

## References, in priority order

1. **diffusers [PR #14355](https://github.com/huggingface/diffusers/pull/14355)**,
   pinned at commit `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` per
   `MiniMaxH3.md` on `cglagovich/minimax-h3`. Provides `AutoencoderKLMiniMaxH3` and
   `AutoencoderKLMiniMaxH3Audio` as **directly importable reference classes** —
   every unit test compares against these, not against a hand-written port.
2. sglang [PR #33275](https://github.com/sgl-project/sglang/pull/33275) — the
   serving-grade reference, mirrored at `<scratchpad>/h3ref/`. Use for the pipeline
   contract and as a cross-check.
3. The raw checkpoint under `FL2VA/video_vae/source/` — the authority on
   architecture, since the outer `config.json` hides it behind `source_path`.

Install the reference (note the `uv` caveat in `MiniMaxH3.md` — a bare `pip install`
in a `uv` venv silently installs to `~/.local` and has no effect):

```bash
uv pip install --python /home/kevinmi/tt-metal/python_env/bin/python --no-deps \
  "diffusers @ git+https://github.com/huggingface/diffusers@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
```

`--no-deps` keeps the resolver from pulling a newer numpy/Pillow/huggingface-hub
into the environment ttnn was built against. Re-check `import ttnn` afterwards.

## Folder structure (from `cglagovich/minimax-h3`, follow exactly)

```
models/tt_dit/
├── models/
│   ├── MiniMaxH3.md                  # owned by cglagovich/minimax-h3
│   ├── vae/minimax_h3/               # visual VAE
│   └── audio_vae/minimax_h3/         # audio VAE
└── tests/models/minimax_h3/          # tests, laid out like ltx/ and wan2_2/
```

No `__init__.py` anywhere — `tt_dit` uses namespace packages.

---

## 1. Visual VAE architecture

From `FL2VA/video_vae/source/config.json`:

**Encoder — CNN, causal.** `ch=128`, `ch_mult=[1,2,2,4,4,8]` (128 -> 1024),
`num_res_blocks=2`, `space_down=[2,2,2,2,1,1]` (16x),
`time_down=[1,2,2,1,1,1]` (4x), `use_3d_conv=true`, `causal_encoder=true`,
`padding_mode="reflect"`, `use_t_isolated_gn=true`, `z_channels=24`,
`pixel_norm_type="imagenet"`. GroupNorm is 32 groups, eps 1e-6, affine.

Verified against the checkpoint: `conv_in(3->128)`, 6 levels x 2 resnets, **3**
`nin_shortcut` (levels 1/3/5) and **4** `downsample` (levels 0-3), `norm_out`,
`conv_out(1024->48)` so `double_z=true`, then a 1x1x1 `quant_conv(48->48)`.

**Decoder — 36-layer ViT, non-causal.** `use_vit_decoder=true`, `num_layers=36`,
`heads=32`, `dim_head=64` (inner 2048), `norm_type="rms_norm"` `norm_affine=true`,
`qk_norm_type="rms_norm"` with **`qk_norm_affine=false`** (no parameters at all),
`ffn_use_gated=true` + `ffn_activation_fn="silu"` (SwiGLU), 3D RoPE with
`rope_dim_ratio=0.75` and **`rope_theta=100.0`** (not 10000),
`space_up=[1,2,2,2,2,1]`, `time_up=null`, plus `post_quant_conv(24->24)`.

The decoder being a transformer is good news: it reuses `layers/linear.py`,
`RMSNorm` and the attention machinery. `vae_ltx.py` and `vae_wan2_1.py` are both
CNN-only, so what they lack is not what this needs.

### The single-frame collapse (already proven)

For **T=1** the causal encoder collapses to a 2D CNN: `BaseConv3d` front-pads the
temporal axis with zeros, so a 3-tap conv sees `[0, 0, x]` and reduces to
`weight[:, :, -1] * x`. Measured per conv shape: rel err 1.5e-07, **bit-exact for
k1**, and 1.13e-07 after chaining twelve — fp32 accumulation order, and it does not
compound. `TemporalIsolatedGroupNorm` at T=1 is likewise plain `GroupNorm`.

This is the keyframe path (FL2VA conditioning encodes single frames), and it is
worth keeping as a fast path with its own test. It is **not** a substitute for the
T>1 encoder, which this plan now also covers.

---

## 2. Deliverables

### 2a. Visual VAE encoder, T=1 (keyframe fast path)
Exists as WIP: `models/vae/minimax_h3/vae_minimax_h3.py`. Host collapse gate passes
(5 tests); the device half is unvalidated. Built on `Conv2dViaConv3d` so it stays on
the `ttnn.experimental.conv3d` path. Runs unsharded — one frame, once per request.

### 2b. Visual VAE encoder, T>1
The general path: restore the temporal taps and the causal halo. Chunked over T at
`vae_clip_length=17` with `vae_token_drop=3`, so `17n+5` pixel frames give `5n+2`
latent frames. Use `ContextParallelConv3d` / the WAN `WanCausalConv3d` pattern with
`VaeHWParallelConfig`.

### 2c. Visual VAE ViT decoder
36 layers, inner 2048, 3D RoPE at theta 100.0, SwiGLU, parameterless qk-norms. This
is the compute-heavy half — 57 latent frames at the canonical working point — and
where sharding earns its keep.

### 2d. Audio VAE decode
DAC `dec_in_proj` then BigVGAN: `upsample_rates [5,5,2,2,2,2]`, SnakeBeta, AMPBlock1,
weight-normed Conv1d. Assess reuse of `models/audio_vae/vocoder_ltx.py` and
`bwe_ltx.py`, which are the same family.

### 2e. Audio VAE encode
Snake1d, ResidualUnit, `AttnProjection` with CausalAttention, `mean_proj`/`logs_proj`.

---

## 3. Known gaps

**Reflect padding.** H3 pads reflect; `neighbor_pad_async` accepts only `zeros` and
`replicate` (`neighbor_pad_async_nanobind.cpp:33`). Narrower than it looks: at
interior shard boundaries the halo is the neighbour's real data, so the mode is
irrelevant and `replicate` is exact there. They differ only in the outermost pixel
at the **global image edges** — after a replicate pad the layout is `[x0, x0, x1,…]`
where reflect wants `[x1, x0, x1,…]`. Fix with a per-device edge-mask blend (a
per-device mask is sharded *data*, so it stays SPMD-uniform). Extending the C++ op
is the alternative and touches a primitive WAN and LTX both depend on. Gate it
explicitly — a 1-pixel border error passes PCC and reads as a faint vignette.

**Reflect index arithmetic** is already verified against torch on host, including the
corner produced by padding H then W sequentially.

---

## 4. Test plan

Structure follows `tests/models/ltx/` and `tests/models/wan2_2/`: one file per
component, `assert_quality(pcc=…, relative_rmse=…)` from `utils/check.py`, and the
mesh parametrization convention

```python
@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params"),
    [pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, id="bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
```
then `mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))`.

**Single-device tests must pass `{}` for `device_params`, not `ring_params`** —
requesting a fabric ring with no partner times out the ethernet handshake before any
kernel runs (observed: job 226).

| File | Covers |
|---|---|
| `test_vae_conv_minimax_h3.py` | one conv per shape the encoder uses, reflect edges included; the T=1 collapse |
| `test_vae_encoder_minimax_h3.py` | ResnetBlock, Downsample, full encoder moments; T=1 and T>1 |
| `test_vae_decoder_minimax_h3.py` | ViT block, 3D RoPE at theta 100, full decoder |
| `test_vae_minimax_h3.py` | end-to-end encode/decode roundtrip PSNR against `AutoencoderKLMiniMaxH3` |
| `test_audio_vae_minimax_h3.py` | BigVGAN decode, DAC encode, stereo-as-batch-2 recombine |
| `test_performance_vae_minimax_h3.py` | per-component device time vs an `expected_metrics` dict, as `tests/models/wan2_2/test_performance_wan.py` does |

Every correctness test compares against the pinned diffusers class. Thresholds start
at the existing convention (`pcc >= 0.999` per component, `0.99` end-to-end) and
tighten where the component is exactly reproducible.

Beyond PCC, the VAEs need reconstruction gates that PCC does not capture:
**roundtrip PSNR/SSIM** on real frames, and for audio a **mel-spectrogram distance**
plus a check that the two stereo channels are correlated but not identical (they
share an encoder, so an accidental broadcast would look plausible).

## 5. Performance

The user asked for good perf on all of them, so each component carries a measured
number, not just a PCC. Order: correctness first, then profile, then optimize.

- Encoder at T=1 is ~3.7 TFLOP for one frame and runs once per request — not worth
  sharding. Establish it as a baseline and move on.
- **The ViT decoder is the target.** 36 layers over 57 latent frames is the bulk of
  VAE time. Shard it, and profile before choosing between HW-parallel and
  sequence-parallel over latent tokens.
- Audio VAE is small but BigVGAN's transposed convs are awkward; measure before
  assuming it is free.
- Report before/after tables in the PR, per the goal's PR requirements.

## 6. Verification workflow

Device runs go through tt-device-mcp in background (`tt_device_job_run_bg` + poll),
never blocking-wait — the Galaxy is shared. **Never pipe a device job to `tail -N`**:
it buffers until exit and the log looks empty. Redirect to a file under
`/home/kevinmi/`. `tt-smi -glx_reset` has standing permission; `tt-smi -r` is
forbidden (it dropped all 32 chips off PCIe on CPLD < 1.16).

## 7. Parked work (done, preserved on `kevinmi/minimax-h3`)

Not part of this scope, but green and worth keeping:

| M | Gate | Evidence |
|---|---|---|
| 1 | disk + weights | 163 GB reclaimed, no weights deleted; FL2VA 81/81 files |
| 2 | packing / conditioning / scheduler bit-exact vs diffusers | 71 passed |
| 3 | AdaLN precompute parity | 0 mismatches on real checkpoint; 1.416 GB table vs 26 GB of weights |
| 4 | DiT weight load at TP=4/SP=8 | 4 passed on the real 4x8 mesh, 90.7 s |

Its findings that still matter to the VAEs: pixels are ImageNet-normalized, not
`[-1,1]`; the keyframe posterior is *sampled* under seed 42 and the latent is
**rounded through float16** before normalization (7.5e-4 effect, load-bearing); and
the whole VAE checkpoint is fp32.
