# MiniMax-H3 on BH Galaxy 4x8 — execution state

**Scope: `t2va` end to end.** Prompt in, video plus synchronized audio out, with every
component gated at the production working point. Superseded the VAE-only scope on
2026-08-04 (amendment 72); the sections below that predate it are kept because their
measurements and dead ends still hold, but read amendment 72 onward for what is current.

**Status: t2va runs end to end and every tier is green** (amendments 78, 80). Artifacts at
`~/h3_t2va_artifacts/t2va.mp4`. Fully-warm latency 63-74 s, VAE decode 3.8 s
(amendments 81-84).

Working point, everywhere: **1344x768, 124 frames @ 24 fps, 50 scheduler steps -> 49
forwards**, mesh 4x8, TP=4 axis 0 / SP=8 axis 1, ring, 2 links.

Plan: `~/.claude/plans/serialized-orbiting-rain.md`. Re-read it and this file every
iteration. (The earlier VAE plan it supersedes was
`models/tt_dit/models/MiniMaxH3_VAE_PLAN.md`, since removed.)

Branch: **`kevinmi/minimax-h3-t2va`, cut from the dangling `0c4ce3596b5`** — the only
commit holding both the tuned DiT and the VAE work (amendment 72). Its parent is
`gh/cglagovich-minimax-h3` (`b85be88d6d3`), which owns the canonical folder structure and
the pinned diffusers reference. Conform to it; do not invent a layout.

References in priority order: diffusers PR #14355 pinned at
`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` (gives `AutoencoderKLMiniMaxH3` and
`AutoencoderKLMiniMaxH3Audio` as importable reference classes — compare against
these, not a hand-written port), then sglang PR #33275, then the raw checkpoint
under `FL2VA/video_vae/source/`.

Tests follow `tests/models/ltx/` and `tests/models/wan2_2/`.

## Branch base — read before any git operation

**Superseded 2026-08-03 by the move to the bash-driven 4x8 BH Galaxy.** The
uncommitted rjsdpa WIP described below belonged to the *previous* machine and is
not present here. The rule it motivated still applies, to different files.

On this machine (`/data/kevinmi/tt-metal`, checked out in place at `8c1cfe3f811`)
the unrelated WIP is the **prodia** work:

```
models/tt_dit/internal-prodia/     (untracked, 768 K)
models/tt_dit/prodia               (untracked symlink -> internal-prodia)
recover-logs/                      (untracked, 4.9 MB)
```

None of these has a path on the H3 branch, so they survive checkout. Backed up to
`~/h3_prodia_wip_backup.tgz`; pre-checkout state in `~/h3_pre_checkout_{head,status,
stashes}.txt` and `~/h3_pre_checkout_umd.diff`. Two more things were handled:

- `tt_metal/third_party/tt-cluster-descriptors` was an **untracked directory** here
  but is a **tracked submodule** on the branch, so checkout would have refused.
  Moved to `~/tt-cluster-descriptors.pre-h3`; the submodule now sits at `7b2176e2`.
- `tt_metal/third_party/umd` was modified (recorded `9058c8434d2`, working tree
  `7b539098e37`); the branch pins `ef7aa4b9dace`, which `git submodule update`
  installed. The `7b539098e37` pointer belonged to the prodia work and is recorded
  in `~/h3_pre_checkout_umd.diff`.
- The previous build (`build_Release`, 4.6 GB, compiled at `dac5c2b0af5`, **3066
  commits behind** this branch) was moved to `build_Release.prodia-dac5c2b` rather
  than deleted, so the prodia work stays restorable. A **full rebuild is required**
  on this branch; `ccache` was cold.

**Never `git add -A`, never reset, never rebase over this tree.** Stage H3 paths
explicitly. Three pre-existing stashes must not be disturbed.

**`origin` is SSH and publickey auth is denied on this host.** Use HTTPS with gh's
credential helper:
`git config credential.helper '!gh auth git-credential'`, then
`git fetch https://github.com/tenstorrent/tt-metal.git <branch>:<branch>`.

**`python_env/bin/python` was a dangling symlink** — the uv interpreter had been
relocated to `/data/kevinmi/uv_pythons/cpython-3.10.19-linux-x86_64-gnu`. Repaired
by symlinking that back under `~/.local/share/uv/python/`. `site-packages` was
intact (torch 2.11.0+cpu).

## Current milestone

**M9 complete — t2va is green end to end.** See amendments 78 and 80 for the gate
evidence and 81-84 for latency. The table below is the original M1-M12 map with its
outcomes filled in; the narrative from amendment 72 onward is authoritative where they
disagree.

**Milestone order changed by user directive (2026-08-03): the VAEs come next**,
ahead of the DiT forward and the text encoder. Video VAE decode + encode and
audio VAE decode + encode, then back to M5/M6/M7. Good call independently: the
ViT decoder was the plan's largest unknown, and the VAEs gate any visible output.

Revised order: **M8 (VAEs) -> M5 (attention PCC) -> M6 (full DiT forward) ->
M7 (Qwen3-VL) -> M9 (e2e) -> M10 (trace) -> M11 (canonical) -> M12 (perf + A/B)**.

## Gate evidence

| M | Gate | Status | Evidence |
|---|---|---|---|
| 1 | 177 GB reclaimed, FL2VA downloaded, `TT_DIT_CACHE_DIR` set | **PASS** | reclaim 273 -> 436 GB free (`~/h3_reclaim.log`); FL2VA **81/81 files, 0 `.incomplete`**, 135 GiB (transformer 62G, text_encoder 63G, video_vae 9.8G, audio_vae 578M) at snapshot `73372e6cf53e414edd3ab03e357717fb0602e758`; checkpoint keys/shapes/dtypes verified (below) |
| 2a | packing bit-exact vs diffusers `minimax-h3` | **PASS** | 48/48 exact checks vs reference across both working points, keyframe and t2va; then `test_packing_minimax_h3.py` **38 passed, 2 skipped** (skips = diffusers branch not installed) |
| 2b | conditioning + scheduler bit-exact | **PASS** | scheduler: 16/16 exact vs reference incl. full 49-eval rollouts at shift 12.0 and 3.0; conditioning: noise stream + generator advance + fp16 recipe exact. Suite `models/tt_dit/tests/models/minimax_h3/`: **71 passed, 3 skipped** |
| 3 | AdaLN precompute parity | **PASS** | real checkpoint: 196 (step, layer) projections + all 49 final-layer rows, **0 mismatches**, built in 6.8 s (`~/h3_adaln_build.log`, table at `~/h3_adaln_table.pt`). `test_adaln_precompute_minimax_h3.py` 13 passed; suite now **84 passed, 3 skipped** |
| 4 | weight load at TP=4/SP=8, shapes/dtypes/fixups | **PASS** | job 225 on the real 4x8 mesh: **4 passed in 90.7 s**, device closed cleanly. Shards read back off the mesh confirm both fixups landed; strict load consumed every key with none missing or unexpected |
| 5 | attention block PCC | **PASS** | 2-layer real weights at the **production** shape (512/414/37296): video PCC 99.9979 %, audio 99.9974 % (amendment 76) |
| 6 | full 50-layer forward | **PASS** | Real checkpoint at the production packed length 38400, 4800 rows/device; finite, correct shapes; no reference is computable at this size (amendment 77) |
| 7 | Qwen3-VL enc (50 layers, unnormalized) | **PASS** | PCC 99.9892 %, RMSE/sigma 1.5 % at a 512-token prompt vs HF `hidden_states[50]` (amendments 74, 76). Vision tower **out of scope**: t2va has no keyframe |
| 8 | video VAE + audio VAE PCC and roundtrip | **PASS** | Pre-existing, re-gated throughout this work: 9 + 16 tests, PCC 99.9977-99.9986 % |
| 9 | e2e t2va @ **1344x768**, all tiers | **PASS** | Tiers 4/5 in amendment 78, tier 6 (VBench + CLIP) in amendment 80. Not 960x544: the working point moved to the shape the perf log is tuned for |
| 10 | trace + residency + warmup | not started | — |
| 11 | canonical 1344x768/8 s, quality gates 3/5/6 | not started | — |
| 12 | profile; **12b TP=8/SP=4 A/B**, adopt measured winner | not started | — |

## Measurements recorded

From M2, on host (no device):

- Canonical 1344x768 / 192f / first+last keyframe: `used=61109`, padded 61440
  (0.54% waste @ `k_chunk=512`), M=3 unique timesteps, 6 global AdaLN runs,
  per-SP-shard runs `[6,1,1,1,1,1,1,2]` mean 1.75.
- Bringup 960x544 / 124f / first+last: `used=21301`, padded 24576 (**13.3%**
  waste @ `k_chunk=512`), 22528 (5.45%) @ `k_chunk=256`.
- `t2va` canonical: M=2, 3 global runs, mean 1.38 runs/shard.
- Shape algebra confirmed: 192f -> 57 latent frames, 320 audio latents;
  124f -> 37 latent frames, 207 audio latents.

## Checkpoint verification (M1, header-only reads, no device)

`FL2VA/transformer`: **535 tensors, 66.3 GB**. Every key name the plan assumed is
present and nothing else is: `blocks.{0..49}.{norm1,norm2,attn.{qkv_proj,out_proj,
q_norm,k_norm},mlp.{fc1,fc2},adaln_proj.linear}`, `token_refiner.blocks.{0,1}.*`
(no adaln, as expected) + `token_refiner.final_norm`, `final_layer.{norm,
adaln_proj.linear,video_out,audio_out}`, `{video,audio}_patch_proj`,
`condition_proj`, `time_embedder.proj_{in,out}`, `rope.inv_freq`.

Shapes confirm the arithmetic: `qkv_proj [21504, 5376]`, `out_proj [5376, 7168]`,
`fc1 [28672, 5376]`, `fc2 [5376, 14336]`, `adaln_proj.linear [96768, 2688]`
(260.1M params/block, the 13B/40% figure), `final_layer.adaln_proj.linear
[10752, 2688]`, `video_out [96, 5376]`, `audio_out [32, 5376]`.

Dtype census: **522 BF16 + 13 F32**, and the 13 fp32 tensors are exactly the set
the plan named — `{video,audio}_patch_proj.{weight,bias}`,
`time_embedder.proj_{in,out}.{weight,bias}`,
`final_layer.{video,audio}_out.{weight,bias}`, `rope.inv_freq`.

## AdaLN precompute (M3) measurements

- 49 evaluations, 2-3 distinct timesteps per step, **146 table rows**.
- `block_params [50, 438, 6, 5376]` bf16, `final_shift`/`final_scale` `[146, 5376]`.
- **1.416 GB total -> 0.354 GB/device at TP=4**, against 26.0 GB of `adaln_proj`
  weights (6.50 GB/device). **18x** fewer resident bytes; the 13B parameters never
  reach the device, taking the DiT from 16.6 to ~10.0 GB/device.
- Build reads the 26 GB once, streaming one block at a time: 6.8 s warm page cache.
- Rounding-order sensitivity quantified: activating in bf16 instead of fp32 shifts
  modulation by **7.8e-3**.

## Video VAE architecture (read from `FL2VA/video_vae/source/config.json`)

The real arch, which the outer `config.json` hides behind `source_path`:

**Encoder — CNN, causal.** `ch=128`, `ch_mult=[1,2,2,4,4,8]` (so 128 -> 1024),
`num_res_blocks=2`, `space_down=[2,2,2,2,1,1]` (16x), `time_down=[1,2,2,1,1,1]`
(4x), `use_3d_conv=true`, `causal_encoder=true`, `padding_mode="reflect"`,
`use_t_isolated_gn=true` (per-time-slice GroupNorm), `z_channels=24`.
`pixel_norm_type="imagenet"` — independent confirmation of amendment 8.

**Decoder — a 36-layer ViT, non-causal.** `use_vit_decoder=true`,
`causal_decoder=false`, `num_layers=36`, `heads=32`, `dim_head=64` (inner 2048),
`norm_type="rms_norm"` with `norm_affine=true`, `qk_norm_type="rms_norm"` with
**`qk_norm_affine=false`** (no weight at all), `ffn_use_gated=true` +
`ffn_activation_fn="silu"` (SwiGLU), 3D RoPE with `rope_dim_ratio=0.75` and
**`rope_theta=100.0`**, `space_up=[1,2,2,2,2,1]`, `time_up=null`.

This *lowers* the §9 "ViT-decoder primitives missing" risk. The decoder is a
transformer, so it reuses `layers/linear.py`, `RMSNorm` and the attention
machinery rather than needing new conv primitives; what `vae_ltx.py` /
`vae_wan2_1.py` lack is not needed. The encoder is the part that wants the
existing conv3d/halo infrastructure. Two genuine deltas to watch:
`qk_norm_affine=false` means the qk norms carry no parameters, and `rope_theta`
is 100.0 rather than the usual 10000.

## M8a design finding: the keyframe encoder is a 2D CNN

FL2VA conditioning only ever encodes **single frames** (T=1), and `BaseConv3d`
front-pads causally with **zeros**. So a 3-tap temporal conv sees `[0, 0, x]` and
collapses to `weight[:, :, -1] * x` — every temporal tap but the last is
multiplied by zero. Verified against the reference `BaseConv3d` for all four shapes
the encoder uses:

| conv | rel err vs 2D collapse |
|---|---|
| k3 pad1 reflect stride1 (`conv_in`, resnet `conv1`/`conv2`) | 1.51e-07 |
| k3 pad(1,0,0) stride(1,2,2) (`downsample`, space only) | 1.55e-07 |
| k3 pad(1,0,0) stride(2,2,2) (`downsample`, space+time) | 1.47e-07 |
| k1 pad0 (`nin_shortcut`, `quant_conv`) | **0.0, bit-exact** |

Chaining 12 of them — the encoder's resnet depth — leaves rel err at 1.13e-07, so
it is fp32 accumulation order and does not compound.

**Consequence: no conv3d, no temporal halo exchange, and no causal machinery for
the keyframe path.** M8a is a 2D CNN built on `layers/conv2d.py`, with the kernel
sliced to `weight[:, :, -1]` as a `_prepare_torch_state` fixup. `time_down` becomes
irrelevant (stride 2 over a 3-deep zero-padded axis still selects the one real
frame). Likewise `TemporalIsolatedGroupNorm` at T=1 is plain
`GroupNorm(num_groups=32, eps=1e-6, affine=True)`.

The T>1 encoder path is only needed for Ref2VA video references, which are out of
FL2VA scope; it can come later and will need the halo machinery.

### Implementation: conv3d mocking conv2d, per the WAN/LTX pattern

Do **not** use `layers/conv2d.py`. Reuse `WanConv2d`
(`models/vae/vae_wan2_1.py:733`, "a conv2d implemented with conv3d"), which is
already HW-parallel over `VaeHWParallelConfig`, does the H/W halo through
`neighbor_pad_persistent_buffer`, carries `logical_h`/`logical_w` masking, and
holds weights in the `prepare_conv3d_weights` layout. Its `_prepare_torch_state`
does `state["weight"].unsqueeze(2)` on a 4D conv2d weight, so H3 slots in by
slicing `weight[:, :, -1]` first — the same fixup the 2D collapse already needs.
Staying on the conv3d path also means the T>1 encoder is an extension later rather
than a rewrite. (`Conv2dViaConv3d` in `layers/audio_ops.py:407` is the same idea
but single-device, so it is precedent rather than the vehicle.)

**Open gap — reflect padding.** H3 sets `padding_mode="reflect"`;
`neighbor_pad_async` accepts only `zeros` and `replicate` (per its nanobind doc,
`neighbor_pad_async_nanobind.cpp:33`). This is narrower than it looks: at interior
shard boundaries the halo is the neighbour's real data, so the mode is irrelevant
and `replicate` is already exact there. The two differ only at the **global image
edges**, and only in the outermost 1 pixel: after a replicate pad the row layout is
`[x0, x0, x1, ...]` where reflect wants `[x1, x0, x1, ...]`.

Plan: pad with `replicate`, then blend the two global edge rows/cols to their
reflect values using a per-device edge mask. A per-device mask is sharded *data*,
so it stays SPMD-uniform in program structure — unlike the §5.2 modulation problem,
which needed differing op counts. Extending the C++ op with a reflect mode is the
alternative and touches a primitive WAN and LTX both depend on; prefer the blend.
Gate it explicitly: a 1-pixel border error is exactly the kind of thing that passes
PCC and looks like a subtle vignette.

Encoder structure confirmed against the checkpoint (560 tensors, **all fp32**):
`conv_in(3->128)`, 6 levels x 2 resnets with channels `[128,256,256,512,512,1024]`,
**3** `nin_shortcut` (levels 1/3/5, where in != out) and **4** `downsample` (levels
0-3) — exactly matching the derivation from `ch_mult`/`space_down`/`time_down` —
then `norm_out(1024)` + silu, `conv_out(1024->48)` (so `double_z=true`), and a
1x1x1 `quant_conv(48->48)`. Only `GroupNorm` needs care on TP: 32 groups / 4
devices = 8, and 128/256/512/1024 all divide by 4.

## §9 verdicts

| Risk | Verdict |
|---|---|
| 4096-row SP alignment waste | **CONFIRMED at bringup canvas**: 13.3% @ `k_chunk=512`, 5.45% @ 256. Canonical is only 0.54%, so this is a bringup-only concern. Re-measure device-side at M6 before changing the default. |
| Vision tower is required scope | **CONFIRMED** — a keyframe feeds both the VLM (as `<Picture i>:` + a vision block whose rows are video-tagged) and the video VAE. Plan and M7 updated; not optional. |

## Amendments to plan assumptions

1. **`last` keyframe anchor overshoots the final frame by 5.0 rotary units.**
   `origin + span(n) - 5/3` subtracts the shortest per-frame span (the `1` of
   `(1,4,4,4,4)`), not the final frame's actual span (a `4` when
   `(latent_t-1) % 5 != 0`). Reads as an off-by-one; it is not. Pinned by
   `test_keyframe_anchor_times`.
2. **diffusers builds the packed sequence with no padding at all**, stating
   padding "cannot influence a live row" — which is why its transformer needs no
   attention mask. Independently validates the plan's `logical_n = used` argument.
3. **Padding-waste estimates corrected** (see Measurements): canonical 0.54% not
   ~2%; bringup 13.3% not 19%, and `k_chunk=256` gives 5.45% not 0.02%.
4. **Text encoder is 50 of 64 layers**, `lm_head` unused, final
   `language_model.norm` **not** applied (~25.2B params / ~50 GB, not 62-66 GB).
5. **QKV is per-head interleaved** — confirmed, plan assumption held. Raw rows are
   `[head0_q(128), head0_k(128), head0_v(128), head1_q, ...]` = 56 x 384 = 21504.
   Reorder to `[q_all; k_all; v_all]` before use; keep fused for
   `ColParallelLinear(chunks=3)`. Source: `reorder_interleaved_qkv` in the
   diffusers `scripts/convert_minimax_h3_to_diffusers.py`, which says the
   reference applies exactly this at load time. No transposes anywhere in the
   conversion.
6. **`fc1` halves must be swapped — this was NOT in the plan and would have
   silently corrupted every FFN.** The checkpoint stores `[gate; value]` and the
   reference computes `fc2(silu(gate) * value)`, but tt_dit's swiglu is
   `t, gate = ttnn.chunk(t, 2, -1); t * ttnn.silu(gate)`
   (`models/tt_dit/layers/linear.py:441-443`), i.e. it wants `[value; gate]`.
   Swap halves in `_prepare_torch_state`, *then* let `permute_for_swiglu`
   (`linear.py:157-164`) do the cross-device interleave — that helper does not
   swap halves, it only redistributes an already-correct order. Plan §5.4 updated.

7. **RNG draw order resolved in favour of diffusers.** The two references genuinely
   differ: diffusers draws once per condition at that condition's own latent shape
   off the request generator, as the request's *first* draws (before video, then
   audio); sglang re-seeds `manual_seed(seed)` per condition and draws at
   `target_latent_t + cond_frames` before slicing. diffusers adopted as the
   HF/MiniMax-authored path. `noise_aug * clean + (1 - noise_aug) * noise` is
   identical in both, so only the stream differed.
8. **Keyframe pixels use ImageNet statistics, not `[-1, 1]`** —
   `mean (0.485, 0.456, 0.406)`, `std (0.229, 0.224, 0.225)`, applied to
   `pixels/255`. Not what the plan assumed. Verified: normalized values fall
   outside `[-1, 1]`.
9. **The sampled latent is rounded through float16 before normalization**, keeping
   ~11 bits. Measured effect vs fp32-exact: **maxdiff 7.5e-4** — well above noise,
   so the released model's conditioning cannot be reproduced without it. Guarded
   by `test_float16_round_trip_is_load_bearing`.
10. **The scheduler is rectified-flow Euler with `eta = 0`, not ancestral**, despite
    sglang naming its file `scheduling_minimax_h3_euler_ancestral.py`. Three
    inverted conventions, all verified: `t = 1 - sigma` with `t = 1` meaning
    *clean*; `x0 = x_t + sigma * v` (**plus**, not the usual flow-match minus); and
    `num_inference_steps` counts sigma grid points including the terminal 0, so
    **50 steps = 49 model evaluations**. Plan §5 wording corrected.
11. **`x0` recovers sigma from the timestep while the Euler ratio reads the sigma
    grid**, deliberately kept apart because `1 - (1 - sigma)` is not exact in fp32
    below sigma 0.5. Do not unify them.
12. **A second copy of the noise-aug formula drifted by 2.4e-7** because it computed
    `1 - t` in Python double instead of fp32. Removed; callers must use
    `MiniMaxH3Scheduler.scale_noise`, which is what the reference calls.
    `test_noise_augmentation_has_no_second_implementation` prevents its return.

### 2026-08-03 — amendments from the move to the bash-driven 4x8 BH Galaxy

Machine: host with 32 `tt-galaxy-bh` chips, fw bundle 19.8.1.0. Workspace
`/data/kevinmi/tt-metal`, checked out **in place** (user decision) at
`8c1cfe3f811`. All of the below is measured on this machine, not assumed.

13. **The weights are NOT at `/data/cglavioch/minmax-h3`, and no raw `FL2VA`
    snapshot exists on this host.** The only copy is
    **`/data/cglagovich/MiniMax-H3-diffusers/`** — a **diffusers-converted** repo
    (`vae/` 10.4 GB in 3 shards + `audio_vae/` 605 MB, world-readable). Globbing
    `/data/*/min*max*` finds nothing else. Consequence: `FL2VA/video_vae/source/
    config.json` is **not available** as the architecture authority. It is not
    needed — `vae/config.json` carries the architecture inline and **matches every
    value this file previously recorded from `source/config.json`** (ch_mult,
    space/time_down, 36 ViT layers, rope_theta 100.0, qk_norm affine-free,
    reflect padding, clip_length 17, token_drop 3). It is also the only format
    `AutoencoderKLMiniMaxH3.from_pretrained` accepts. Do not re-download.

14. **The checkpoint uses diffusers key names, so the WIP encoder's
    `_prepare_torch_state` loads nothing.** 703 tensors, **all F32**. Actual names:
    `encoder.down_blocks.{0..5}.resnets.{0,1}.{norm1,norm2,conv1,conv2}`,
    `...resnets.0.conv_shortcut` (levels **1/3/5** only), `encoder.down_blocks.
    {0..3}.downsamplers.0.conv`, `encoder.{conv_in,conv_out,norm_out}`,
    `quant_conv`/`post_quant_conv`, and `decoder.transformer_blocks.{0..35}.*`.
    The level/shortcut/downsample **counts** confirm the earlier derivation
    exactly — only the names differ from the assumed `down.<L>.block.<i>` /
    `downsample` / `nin_shortcut`. Sizes: encoder **180.3 M** params (0.67 GiB
    fp32), decoder **2.424 B** (9.03 GiB fp32 / 4.51 GiB bf16).

15. **Spatial tiling is ON by default and it reshapes the perf story.**
    `AutoencoderKLMiniMaxH3.__init__` sets `use_tiling=True`,
    `tile_sample_min_{height,width}=256`, `tile_sample_min_overlap_*=64`. Both
    `_encode_clip` and `_decode_clip` split into 256x256 **pixel** tiles, run the
    model **independently per tile**, and cross-fade overlaps. Derived geometry
    (verified): `frame_pre_padding=3`, `tokens_chunk_size=5`, `token_overlap=2`,
    `frame_overlap=5`; `_decode_clip` always receives **7** latent frames.
    Therefore one decoder call is **1792 patches + 5 suffix = 1797 tokens**, not a
    230k-token sequence. Measured budgets:

    | Working point | Tiles | Latent T | Decoder calls | Decode | Encoder calls | Encode (T>1) |
    |---|---|---|---|---|---|---|
    | 1344x768 / 192f | 4x7=28 | 57 | 308 | ~10.6 TFLOP ea, **3261 TFLOP** | 336 | **3009 TFLOP** |
    | 960x544 / 124f | 3x5=15 | 37 | 105 | 1112 TFLOP | 120 | 1074 TFLOP |

    **This contradicts the plan's §5** ("the ViT decoder is the bulk of VAE time";
    "encoder not worth sharding"). True for the FL2VA **keyframe** path (T=1,
    ~18 TFLOP tiled, once per request). **False for the T>1 clip encoder**, which
    is within 10% of the decoder. Tiling also *costs* work: a tiled 1344x768
    keyframe is ~18 TFLOP vs ~10 untiled, from overlap — but parity requires it.

16. **Both models have exactly one activation shape, always.** `_split_tiles` only
    emits a non-256 length when the tile spans the whole axis, and the last decode
    chunk lands exactly on `z_len_padded`. So encoder is always
    `(1,3,17,256,256) -> (1,48,5,16,16)` and decoder always
    `(1,24,7,16,16) -> (1,3,28,256,256)`. One conv3d blocking, one matmul config,
    one SDPA program config — and **both models are single-shape traceable**. At
    ~540 ops per decoder call x 308 calls, trace capture is the whole perf story.

17. **Parallelization: data-parallel over independent (tile, chunk) work units,
    weights replicated — NOT `VaeHWParallelConfig` H/W sharding.** Tiles are
    computed independently and only blended at the end, so keep each unit whole on
    one device. Consequences: **the reflect-padding gap is void** (reflect becomes a
    local slice-and-concat; no `neighbor_pad_async` reflect mode needed, no
    edge-mask blend, no C++ op change — this retires the risk §3 and the M8a
    section below both treat as central), and GroupNorm stays local. DP-32
    replicated costs **0 CCLs**; TP4xDP8 would add 72 all-gathers per decoder call
    on 1797x2048 and make the matmuls skinny. The **only** argument for TP is
    tooling: `Module.save` -> `ttnn.dump_tensor` writes all 32 device shards, so a
    replicated 4.51 GiB parameter set becomes **~144 GiB on disk** — verify with a
    1-layer decoder and `du -sh`, and prefer `load_model(create_cache=False)`.

18. **`normalization.GroupNorm` already implements the per-frame GroupNorm
    exactly — no new code and no H/W stats reduction needed.** Its `forward`
    reshapes `(B,H,W,C) -> (B,1,H*W,C)`, so `ttnn.group_norm` pools over
    `C_group x H x W` **per batch row**; feeding `(T,H,W,C)` with `mesh_axis=None`
    reproduces `MiniMaxH3VideoGroupNorm` (T folded into batch). Do **not** use
    `GroupNorm3D` — its `dims=3` pools *across* T, which is wrong here. Two
    caveats: the kernel requires **bf16** input, and its core grid must satisfy
    `Ht % nvr == 0` with `Ht = T*ceil(H*W/32)`, so the four validators at
    `vae_mochi.py:368-460` (`_valid_multicast`, `_valid_norm_grid`,
    `_safe_num_out_blocks`, `_run_norm`) are **load-bearing**. Swept in pure
    Python over all 12 encoder GroupNorm sites (T in {17,9,5,1},
    HW in {65536,16384,4096,1024,256}) x nvc in {1,2,4,8}: **a valid grid exists
    everywhere and the `nvr=1` fallback is never needed**, but the default
    `CoreGrid(8,8)` is invalid at T=9 (needs gy=4) and T=5 (needs gy=5). So use
    the validator, never the default grid.

19. **`Conv2dViaConv3d` cannot load a conv with unaligned input channels.**
    `__init__` sets `self.in_channels = aligned_channels(in_channels)` (3 -> 32) and
    sizes `Parameter` as `d = kh*kw*32 = 288`, but `_prepare_torch_state` calls
    `prepare_conv3d_weight_state` **without** `unpadded_in=`/`in_channels=`, so the
    prepared weight keeps `d = kh*kw*3 = 27` and the shape check rejects it.
    `encoder.conv_in` (3->128) hits this exactly. **`LTXCausalConv3d` does not have
    this bug** — it inlines the C_in pad (`vae_ltx.py:178-181`, comment "encoder
    conv_in: 48 -> 64") and pairs it with `conv_pad_in_channels` on the
    activations (`vae_ltx.py:1286`). Consequence: build the H3 conv on the
    `LTXCausalConv3d` shape (with `parallel_config` removed), not on
    `Conv2dViaConv3d`.

20. **Decoder details missing from the plan.** (a) `scale1`/`scale2` are per-channel
    **LayerScale** multipliers: `h = h + attn(norm1(h))*scale1`, then
    `h = h + ff(norm2(h))*scale2`. (b) The cls token is `torch.zeros_like(
    hidden[:, :1, :])` — a runtime zero, **not** a parameter; the sequence is
    `[patches, register_tokens(4), zero_cls(1)]` and all 5 suffix rows take
    **zero** position ids. (c) The reference runs `norm1`/`norm2` and `norm_q`/
    `norm_k` in **fp32** regardless of compute dtype. (d) `norm1`/`norm2` are
    RMSNorm eps **1e-5** weight-only (so `bias=False`); `norm_q`/`norm_k` are
    RMSNorm over `dim_head=64` with `elementwise_affine=False` (**no parameters**);
    `norm_out` is **LayerNorm** (weight+bias) at eps 1e-5.

21. **RoPE rotates 48 of 64 head channels, pairing (i, i+24) — so
    `rotary_embedding_llama` cannot be used directly.** `inv_freq = 1/theta**
    arange(0, 1, 2*3/48)` gives 8 freqs; `angles = 2*pi*pos*inv_freq` flattens to
    24 and `.tile(2)` doubles to 48, so `chunk(2)` pairs *i* with *i+24* and
    channels [48,64) pass through. A 64-wide NeoX op pairs *i* with *i+32* — wrong
    pairing, and it would rotate the passthrough channels. 48 is also not
    tile-aligned (48 % 32 = 16). Best plan: `ttnn.alt_complex_rotate90`
    (interleaved GPT-J) plus a **load-time lane permute** of the q/k weight rows
    per head, `perm = [0,24,1,25,...,23,47] + [48..63]`, with cos/sin built 64-wide
    carrying interleaved-duplicated angles on lanes 0-47 and **cos=1, sin=0 on
    lanes 48-63** so the passthrough falls out of `x*cos + rot90(x)*sin` with no
    slicing. Q and K take the same permute so `QK^T` is invariant; V untouched; no
    un-permute. The affine-free q/k RMS norms are permutation-invariant. Template:
    `attention_ltx.py:_permute_qk:226`.

22. **The swiglu half-swap of amendment 6 does NOT apply to the VAE decoder — do
    not port it.** Pinned `diffusers/models/activations.py:143-146` `SwiGLU.forward`
    is `hidden, gate = proj(x).chunk(2,-1); hidden * activation(gate)`, i.e. first
    half = **value**, second = **gate** — identical to tt_dit's
    `t, gate = ttnn.chunk(t,2,-1); t * ttnn.silu(gate)`. Amendment 6's swap came
    from the **raw** MiniMax checkpoint layout, not the diffusers-converted one.
    Applying it here would silently corrupt every decoder FFN. Assert it with a
    CPU-only test either way.

23. **Both quant convs can be folded away algebraically.** `encoder.conv_out`
    (1024->48 k3) is followed by `quant_conv` (48->48 k1) with no nonlinearity ->
    fold into one 1024->48 k3. `post_quant_conv` (24->24 k1) is followed by
    `decoder.proj_in` (Linear 24->2048) -> fold into one 24->2048 linear. This
    removes most of the awkward-channel risk. Gate behind a flag for A/B. Relatedly,
    the `max(32, out)` page-size risk is **smaller than feared**: `vae_ltx.py`
    ships a working 1024->48 conv3d on this branch, so that failure mode is
    specific to the degenerate `(T,1,1)` conv1d shape. Do not build an
    `_AlignedOutConv2d`.

24. **Audio VAE: the decoder has 7 upsample stages, not 6.** `config.json` gives
    `decoder_rates=[5,5,2,2,2,2,2]` and `decoder_kernel_sizes=[9,9,4,4,4,4,4]`; the
    checkpoint has `decoder.ups.{0..6}` and **21** AMP blocks (7 stages x 3 kernel
    sizes) = 63+63 convs and **126** anti-aliased SnakeBeta activations. Plan §2d's
    `[5,5,2,2,2,2]` is wrong. 1087 tensors, all F32. The VAE is **mono**
    (`conv_post` emits 1 channel); stereo is batch 2.

25. **Audio VAE corrections and gaps.** (a) `pre_block.attn.proj.weight` is
    **[32,32]** (the reference's `nn.Linear(out_dim,out_dim)`); the **[32,2048]**
    tensor is `pre_block.proj.weight`, the residual bypass. (b) **The Kaiser-vs-Hann
    concern is void**: `audio_resample.Activation1d` already defaults
    `window="kaiser"`, `_make_kaiser_sinc_kernel_1d` matches the reference
    arithmetically, and `LowPassFilter1d`/`UpSample1d._prepare_torch_state`
    **already absorb a checkpoint `filter` buffer of shape (1,1,k)** — and the H3
    key paths line up with the tt_dit child names exactly. Load the checkpoint
    filters verbatim. (c) **Real gap instead**: `Conv1dViaConv3d`'s `"zeros"`
    padding is wrong for **all five** encoder strided convs — the reference uses
    `padding=ceil(stride/2)` with `kernel=2*stride`, while the base derives
    `eff_k//2`, giving `L/s + 1` instead of `L/s` for s in {2,4,5}. Needs an
    additive `padding` kwarg. The dilated k=7 residual convs are already exact.
    (d) **Weight norm is unsupported anywhere in tt_dit** (zero `weight_g` hits);
    fuse torch-side with the uniform `dim=0` rule
    `w = weight_g * weight_v / weight_v.flatten(1).norm(dim=1).view(-1,1,1)`, noting
    that for `ConvTranspose1d` axis 0 is **`Cin`** (weight is `(in,out,k)`) — do not
    transpose in the converter. Getting that axis wrong is the most likely
    silent-wrongness bug in the audio port.

### 2026-08-03 — M8 results on the BH Galaxy

Scope narrowed by user directive to four product configs: **768P and 1440P, at 5 s and
10 s**. Because tiling is on, all four reduce to **two device shapes** -- encoder
`(17,256,256) -> (5,16,16)` and decoder `(7,16,16) -> 1797 tokens` -- so only those are
tuned and tested. Resolution changes the tile count (768P 4x7=28, 1440P 8x13=104) and
duration the clip/chunk count (5 s = 124 frames = 37 latent frames; 10 s = 243 = 72).

| Gate | Result |
|---|---|
| visual conv / downsample / resnet, both temporal-tap modes | **27 passed, 0 failed** in 92 s. Convs at PCC 100.0000%, resnets 99.99% |
| visual full encoder, keyframe + 17-frame clip at `(*,256,256)` | **PASS**, pcc 99.98-99.99% vs `MiniMaxH3VideoEncoder3d` |
| visual encode **e2e** on the real checkpoint | **7 passed** in 262 s: tiled `_encode_clip` (keyframe + clip), `_encode` chunking with `token_drop`, and `_split_tiles` geometry for all four configs |
| decoder RoPE tables | **bit-exact** (zero difference) vs `MiniMaxH3VideoRotaryPosEmbed` |
| decoder permuted-rot90 rotation | **exactly** equals the reference rotation; pass-through lanes and suffix rows preserved |
| decoder attention @ 1797 tokens | pcc 99.9844% |
| decoder transformer block (LayerScale + SwiGLU) | pcc 99.9997% |
| audio checkpoint converter | **6 passed**, incl. real-checkpoint fusion vs `remove_weight_norm` across >100 convs at <1e-6 |
| audio encoder + decoder strict load of the real checkpoint | **0 missing / 0 unexpected** on both, 913 converted keys |

Measured numbers worth keeping:

- `conv3d` at the shipping tile with the fallback `(32,32,1,1,1)` blocking: `conv_in`
  **9 ms** at `(1,256,256)`, **305 ms** at `(17,256,256)`, resnet conv **33 ms**. Slow but
  not a correctness blocker, which corrects an earlier claim that it was.
- All 22 encoder conv shapes **miss** `_FP32_BLOCKINGS`. Stubs are now seeded (see
  `conv_minimax_h3.py`); `bruteforce_conv3d_sweep.py` is for the perf pass.
- Per-frame GroupNorm: pcc >= 0.99985 at all 16 encoder sites. Only three need explicit
  `num_out_blocks` (`(256,17,128,128)`: 4, `(128,17,128,128)`: 4, `(128,17,256,256)`: 16).

### 2026-08-03 — audio VAE findings

26. **`ttnn.conv1d` needs `l1_small_size` allocated at device open.** Without it every
    depthwise filter call fails in `bank_manager.cpp` regardless of shard layout and
    regardless of size -- which reads as an op/shape problem and is not one. Surveyed
    HEIGHT/WIDTH/BLOCK/auto sharding at `(T=51,C=512)`, `(T=2081,C=512)` and
    `(T=331211,C=8)`: **all four layouts fail without it and all succeed with
    `l1_small_size=32768`**, including the 10 s decode tail. The LTX audio tests already
    pass `{"l1_small_size": 32768}` in `device_params`; any H3 audio test must too.
    `depthwise_tap_filter` also now has a shift-multiply-add fallback for the case where
    the conv1d DRAM slicer finds no valid configuration, but conv1d is the primary path.

27. **`ttnn.transformer.scaled_dot_product_attention` is bf16-only**
    (`sdpa_device_operation.cpp:43`). The audio encoder's `pre_block` runs fp32 like the
    rest of the audio path, so q/k/v are cast to bf16 for that one op and the result cast
    back -- the same shape of compromise the visual encoder makes for GroupNorm. Without
    it the encoder fails outright.

28. **A bare `Snake` returns TILE, and the next conv asserts ROW_MAJOR.** LTX only ever
    uses `Snake`/`SnakeBeta` inside `Activation1d`, whose resamplers own the layout; H3's
    DAC encoder applies a bare Snake *between* convolutions, which LTX never does. Handled
    by `_snake_row_major`. Note `ttnn.snake_beta` exists as a **single fused op**
    (`x + sin^2(alpha*x)/beta`, TILE, fp32 or bf16) and is the better vehicle than the
    five-op composition in `audio_ops.Snake` -- H3's Snake1d is exactly `snake_beta` with
    `beta = alpha + eps`. Worth adopting in the perf pass.

29. **`register_conv3d_configs` does not reach the fp32 path.** It updates
    `_DEFAULT_BLOCKINGS`, but `get_conv3d_config` short-circuits to `_FP32_BLOCKINGS`
    when the weights are fp32, so the registration is silently ignored for an fp32 model.
    Both tables now get seeded. Also: `C_out_block` must be a multiple of 32 **and**
    divide the padded output channels evenly -- 96 is legal against 384 (as in the WAN
    entries these stubs are shaped after) but not against 128 -- and `C_in_block=256`
    with a 32-pixel H/W block overflows L1 in fp32 (measured 2786176 B against 1572864 B).

30. **Audio encode is correct**: pcc **0.998** against `AutoencoderKLMiniMaxH3Audio.encode`
    on the real checkpoint (posterior mean, 32 latent frames). The DAC trunk, the
    `pre_block` causal attention with its fused qkv and mean-pooled heads, and the
    posterior heads all match.

31. **Exactly one conv1d shape needs the MAC fallback**, not the general case.
    Instrumenting the fallback showed a single distinct shape falling back --
    `(T_pad=1041, C=512, K=7, stride=1)`, hit 36 times (once per AMP-block conv at
    decoder stage 0). Probed in isolation with `l1_small_size=32768`: **HEIGHT, WIDTH,
    BLOCK and auto sharding all fail** at that shape, so it is shape-inherent in
    `ttnn.conv1d` rather than memory pressure from the surrounding graph. K=7 is a
    sub-tap vector of the 12-tap Kaiser filter at ratio 2. Every other depthwise filter
    in the decoder uses the real conv1d op. A targeted fix (or an upstream conv1d issue)
    would remove the last fallback; the MAC path is correct meanwhile -- verified that
    `ttnn.slice`'s `slice_step` supports strided fp32 slicing, which is what it relies on.

32. **The last audio failure was a matmul, not a convolution.** Three rounds of conv3d
    blocking edits left the overflow byte count identical at 1979264 B, which was the tell:
    the traceback frames point at `linear.py:98`, i.e. the `qkv` projection inside the
    audio encoder's `CausalAttention`. At a 10 s clip that is 2048 -> 6144 over 405 latent
    frames at batch 2, and the default `(8, 8, 8)` matmul blocking overshoots L1.
    `default_block_size=(4, 4, 4)` fixes it. **Lesson: read the traceback frames before
    tuning blockings** -- an unchanging byte count across parameter edits means the op
    being tuned is not the op that is failing.

    With that fixed the whole audio suite passes, and the encode gates come in at
    RMSE/sigma **4.6-5.2%**, i.e. essentially at the original 5% bar rather than needing
    the relaxed one. So the earlier 10.6% was not accumulation after all -- it was this
    matmul's blocking. The three components measured clean along the way
    (MAC bit-exact, `Activation1d` 0.17%, no SDPA in the decoder) were correct findings,
    but the accumulation *explanation* built on them was wrong.

### 2026-08-03 — M8f baselines (single device, untuned)

Roundtrip quality: **visual PSNR 40.46 dB**, **audio PSNR 29.89 dB** (both vs the
reference's own round trip). Suite runs in 153 s, so it is cheap to re-run per iteration.

| Baseline | Per invocation |
|---|---|
| visual encoder, clip tile `(17,256,256)` | **3.544 s** |
| visual encoder, keyframe tile `(1,256,256)` | 0.264 s |
| visual decoder, `(7,16,16)` = 1797 tokens | **0.755 s** -> **14.0 TFLOP/s** effective |
| audio encode, 5 s stereo | 0.416 s |
| audio decode, 5 s stereo | 6.185 s |

Projected single-device sequential totals:

| Config | Encode | Decode |
|---|---|---|
| 768P 5 s | **793.8 s** (224 inv) | 148.0 s (196 inv) |
| 768P 10 s | 1488.4 s (420) | 296.0 s (392) |
| 1440P 5 s | 2948.4 s (832) | 549.6 s (728) |
| 1440P 10 s | 5528.2 s (1560) | 1099.3 s (1456) |

**The encoder dominates, not the decoder -- 5.4x it at 768P 5 s.** That inverts the
plan's §5 assumption *and* the FLOP-based expectation: the two halves are within 10% on
FLOPs (3009 vs 3261 TFLOP) but the decoder achieves **14.0 TFLOP/s** against the
encoder's **~2.3 TFLOP/s**, so the encoder is ~6x less efficient per FLOP. That is the
untuned conv3d blocking stubs, which is where the largest absolute win sits.

Neither number includes the data-parallel mapping: with tiles spread over the 32-chip
mesh these divide by up to 32 (768P 5 s decode ~4.6 s, encode ~24.8 s).

### Next: encoder perf via H/W parallelism + neighbor_pad (user directive)

**Order settled: H/W parallelism BEFORE the blocking sweep.** `get_conv3d_config` keys on
`h_factor`, `w_factor`, `T`, `H`, `W`, so sharding changes every conv's per-device extent --
sweeping first would tune shapes that are about to be discarded. Sweep the shapes we ship.

**Done so far:** `reflect_edge_correction` and `edge_mask_pair` in `conv_minimax_h3.py`.
The index mapping is verified on host: with pad `p`, the reflect value for `padded[p-1-j]`
is `padded[p+1+j]`, so `[0,0,1,2,3,4,5,5] -> [1,0,1,2,3,4,5,4]` exactly. The blend is
mask-driven so every device runs identical ops.

`MiniMaxH3CausalConv3d` now takes `parallel_config` + `ccl_manager`: the external/internal
pad split (halo on a sharded axis, local slice-and-concat on a replicated one, never both),
a `_halo_pad` that issues one fused `neighbor_pad_persistent_buffer` for both axes with
`padding_mode="replicate"` and then applies the reflect correction per axis, and the real
`h_factor`/`w_factor` fed to `get_conv3d_config` so the blocking lookup keys on the sharded
extent. **Unsharded path verified unregressed: 20/20 conv tests still pass.** The sharded
path has not run yet -- it needs the encoder wiring below before it can be exercised.

**Still to do:** switch `MiniMaxH3FrameGroupNorm` to the all-gather/norm/re-partition shape from
`latent_upsampler_ltx.py:46-82`; thread `logical_h`/`logical_w` through the encoder for the
mesh-factor pad crop; then sweep.

Two obstacles, both with an existing in-tree precedent, and one measurement question to
settle first.

**1. `neighbor_pad_async` has no `reflect` mode** (`neighbor_pad_async_nanobind.cpp:33`
offers only `zeros`/`replicate`), and H3 pads reflect. Narrower than it looks: at interior
shard boundaries the halo *is* the neighbour's real data, so the mode is irrelevant and
`replicate` is already exact there. They differ only in the outermost pixel at the two
**global** image edges -- replicate gives `[x0, x0, x1, ...]` where reflect wants
`[x1, x0, x1, ...]`. Plan: pad `replicate` via `neighbor_pad_persistent_buffer`, then blend
the global edge rows/cols to their reflect values with a per-device edge mask. A per-device
mask is sharded *data*, so program structure stays SPMD-uniform. Gate it explicitly: a
1-pixel border error passes PCC and reads as a faint vignette. `WanCausalConv3d`
(`vae_wan2_1.py:253`) and `LTXCausalConv3d` (`vae_ltx.py:42`) are the templates; LTX's
external/internal padding split is the cleaner of the two.

**2. GroupNorm reduces over C, H *and* W within a frame**, so H/W sharding needs
cross-device stats. `normalization.GroupNorm` shards channels, not H/W, so it cannot do it
-- but the **LTX latent upsampler already solves this**: `latent_upsampler_ltx.py:46-82`
all-gathers H/W, runs GroupNorm on the full (cropped) extent, and re-partitions, handling
the ROW_MAJOR/TILE transitions and the mesh-factor pad crop. Reuse that shape rather than
inventing a distributed GroupNorm.

**Measurement question to settle before building it.** The encoder's problem is
*efficiency*, not parallelism: 2.3 TFLOP/s against the decoder's 14.0 on the same silicon.
H/W sharding spreads one tile over more chips, but each chip still runs at 2.3 TFLOP/s, and
DP-over-tiles already saturates the mesh (768P 5 s has 224 work units for 32 devices; 1440P
has 832). So H/W parallelism buys **per-tile latency and lower peak memory per device**, not
throughput, until the blockings are tuned. Sweeping `_FP32_BLOCKINGS` with
`bruteforce_conv3d_sweep.py` is what addresses the 6x gap. Worth doing the sweep first, or
at least measuring both, so the H/W work is evaluated against a tuned single-device number
rather than an untuned one.

## Hangs / resets

None yet. No device work has run.

`tt-smi -glx_reset` has standing permission for this goal. `tt-smi -r` remains
forbidden — it dropped all chips off PCIe on CPLD < 1.16.

## Failed attempts

- Three assertions of mine were wrong, not the port; in every case parity against
  the reference had already passed, which localized the fault to the test:
  `last`-anchor equality with the final frame (amendment 1), and twice an exact
  comparison against a Python float literal that is not fp32-representable
  (`0.7`, then `[1.0, 0.6, 0.3, 0.0]`). Lesson: compare fp32 tensors to fp32
  tensors, never to Python literals.
- `pytest.raises` is rejected by the `prefer-expect-error` pre-commit hook; the
  repo requires the `expect_error` fixture from `conftest.py:881`.
- `pre-commit` needs `python_env/bin` on `PATH` or the commit aborts with
  "`pre-commit` not found". black/autoflake reformat on first run and abort the
  commit; re-add and re-commit. autoflake will strip an import the module no
  longer uses even if a *test* reads it through the module — import it from its
  real home instead.
- `Module` is an ABC with `forward` abstract, so a parameter-owning container
  cannot be instantiated for a load-only gate without declaring one.
- tt_dit's plain `RMSNorm` defaults to **`bias=True`**; every H3 norm is
  weight-only, so all of them need `bias=False`. `DistributedRMSNorm` asserts
  `not bias` and is unaffected.
- The broker appends `tt-metal` to `workspace`, so pass `workspace=/home/kevinmi`,
  not the repo root.
- Piping a device job to `tail -N` makes its log empty until exit — the known
  buffering trap, hit again. Stream to a file, or read the broker's job output.

## Device runs

| Job | What | Result |
|---|---|---|
| 222 | M4 first attempt | 3 failed: `Module` ABC needs `forward`. 35 s, clean close |
| 224 | M4 second attempt | 3 failed: `RMSNorm` bias keys missing. 90 s, clean close |
| 225 | M4 | **4 passed, 90.7 s**, clean close, JIT cache 100% hits |

Pre-run state for 225: device idle, `device_degraded` empty, no compute processes
holding the mesh (broker MCP servers only), host `g03blx02`, TT-KMD 2.9.0.
No hangs, no resets needed.

## Next step

**M8a — video VAE encoder on device.** Design is settled (above); this is now
implementation, not research. Write
`models/tt_dit/models/vae/vae_minimax_h3.py` with:

1. An H3 conv wrapping `WanConv2d`, whose `_prepare_torch_state` slices
   `weight[:, :, -1]` before delegating.
2. The reflect-edge blend over `replicate` neighbour padding.
3. `ResnetBlock` = GroupNorm(32, eps=1e-6, affine) + silu + conv, twice, plus a
   `nin_shortcut` k1 when in != out (levels 1/3/5).
4. `Downsample` = asymmetric spatial pre-pad `(0,1,0,1)` then stride-2 conv
   (levels 0-3).
5. `conv_in(3->128)`, 6 levels x 2 resnets over `[128,256,256,512,512,1024]`,
   `norm_out` + silu, `conv_out(1024->48)`, `quant_conv(48->48)`.
6. `VaeHWParallelConfig(height_parallel=(4,0), width_parallel=(8,1))`, fp32
   throughout — the whole VAE checkpoint is fp32.

Gates, in order: (a) a single H3 conv vs the reference `BaseConv3d` at T=1,
including the reflect edges — catches the padding gap directly; (b) one
`ResnetBlock`; (c) the full encoder's moments vs the reference `_encode_clip`,
`pcc >= 0.999`; (d) end-to-end against the already-gated
`conditioning.encode_keyframes`, with the seed-42 posterior sample drawn on host
and passed in.

Then **M8b** the 36-layer ViT decoder, **M8c** the audio VAE (BigVGAN decode
first, DAC encode second), before returning to M5.

### Revised next step — 2026-08-03, this machine

The design above still holds in outline, but amendments 13-25 change five things
in it, so read those first. Concretely, superseding the six numbered points above:

- Reference every gate against the **pinned diffusers classes on the real
  diffusers-converted checkpoint** at `/data/cglagovich/MiniMax-H3-diffusers`
  (amendment 13), not against `BaseConv3d` (which does not exist in this
  reference; the class is `MiniMaxH3VideoCausalConv3d`) and not against a
  hand-written 2D reference on synthetic weights. The existing five tests in
  `test_vae_encoder_minimax_h3.py` are a valid *collapse proof* and should be
  kept, but they are not the plan's stated gate.
- Point 1 changes: build the conv on the **`LTXCausalConv3d` shape with
  `parallel_config` removed**, not on `WanConv2d`/`Conv2dViaConv3d` — the latter
  cannot load `conv_in` at all (amendment 19).
- Point 2 is **deleted**: there is no reflect-edge blend and no halo, because
  each tile stays whole on one device (amendment 17).
- Point 3 changes: `GroupNorm` is `normalization.GroupNorm` fed `(T,H,W,C)` with
  `mesh_axis=None`, wrapped in the Mochi grid validators (amendment 18).
- Point 6 changes: `VaeHWParallelConfig` is **not** used. DP-32 over work units
  with replicated weights (amendment 17).
- The module tree must **mirror the diffusers key names one-for-one**, which makes
  most `_prepare_torch_state` overrides unnecessary (amendment 14).
- Keep the WIP file and refactor it: `_reflect_pad_hw` and
  `_reflect_pad_asymmetric` are correct and reusable at T>1, and the
  `weight[:, :, -1]` slice is **exact** (not an approximation) because
  `temporal_padding = kernel_t - 1 = 2` front-pads zeros. Parameterize the conv by
  `temporal_taps in {1,3}` so M8b extends it rather than forking.

**Immediate first action, before any model code: the DP conv3d probe.** Nothing in
tt_dit does data-parallel conv3d — every existing VAE replicates data and shards
H/W. On a 2-device submesh, run one conv3d and one `GroupNorm` with **different**
random inputs per device, then `ttnn.get_device_tensors` and assert each shard
independently against its own torch reference. If any mesh broadcast assumes
replicated inputs, the DP-over-tiles design in amendment 17 does not work and must
be reconsidered before anything is built on it.

Environment prerequisites for this machine are in "Branch base" above. Gate 0 is:
`ttnn` plus both H3 reference classes importing in one interpreter, and the
host-only suite green.

---

## Amendment 33 (2026-08-03) — the DP-over-work-units probe passed, bit-exact

The probe demanded at the end of amendment 17 (and never run until now) is green.
`models/tt_dit/tests/models/minimax_h3/test_vae_data_parallel_minimax_h3.py`, 4x8 mesh,
32 **distinct** random `(1,17,256,256,32)` units, one per device, weights replicated:

| check | result |
|---|---|
| per-device shard shape | `(1, 17, 256, 256, 32)` |
| max abs spread across the 32 replicas | `0.0` |
| DP vs replicated, units 0 / 7 / 31 | max abs diff `0.0`, PCC `100.0000 %` |

Two things this settles:

1. **The encoder is pure SPMD.** No op in the stack assumes a replicated input; a unit's
   result does not depend on what the other 31 devices hold. The gate is reference-free
   on purpose -- it re-runs selected units *replicated* and requires the answer to be
   unchanged -- so it isolates independence from parity, which is already gated per-unit
   against diffusers in `test_vae_encoder_minimax_h3.py`.
2. **`ttnn.ShardTensorToMesh(dim=0)` hands each device the LOCAL shape**, not the global
   32-unit one. That is load-bearing and now asserted in the test: conv3d blockings and
   the `GroupNorm3D` core grid are both chosen from `x.shape` at construction, so a
   global shape would have sized every kernel for 32x the work.

Wired into `MiniMaxH3Vae` as `_run_encoder_units`, which runs units in mesh-sized waves.
`encode` now collects `(clip, tile)` units across **all** clips before batching: 768P is
a 4x7 grid, so batching per clip would leave 28 units against 32 devices and waste an
eighth of the mesh on every clip. The final short wave is padded by repeating a unit
rather than shrinking the program, because a shorter wave is a different shape and would
build a second set of blockings and grids.

Projection at 768P/5s: 336 units / 32 devices = 11 waves. Against the amendment-32
baseline of 3.544 s per tile, **793.8 s -> ~40 s**, and the 32-unit wave measured about
the same wall clock as a single-unit run.

## Amendment 34 (2026-08-03) — the cherry-picked fused distributed GroupNorm cannot serve H3

`828d9e6ebbf` was cherry-picked (as `a4e6530e96e`) to supply the cross-device spatial
statistic that H/W sharding needs. Gated at the encoder's real norm sites in
`test_vae_distributed_norm_minimax_h3.py`, it fails hard:

```
TT_FATAL: v1 supports batch N==1 only (shape [N, 1, H*W, C]); got N=5.
          GroupNorm stats must not fold across batch.
```

This is **not a removable guard**. `dit_fused_distributed_groupnorm_device_operation.cpp:36-42`:
"v1 folds the spatial extent as `physical_volume()/C`, which spans all batches -- that is
the wrong statistic for N>1 ... per-batch looping is deferred to a later version."

H3's norm is **per-frame** (T folded into the batch axis, amendment 21), so `N = T` is
17, 9 or 5 at every site -- never 1. Using v1 as written would mean one invocation per
frame: 17 fabric all-gathers at the shallow sites, times 13 norm sites, per work unit.

Second, independent limitation: the op takes a single `cluster_axis`, so it cannot reduce
over both mesh axes. **2D H/W sharding is out of its reach regardless of the batch fix.**

The cherry-pick is kept (it is additive, and a later version may lift the restriction),
but the H/W path needs its own norm. See amendment 35.

## Amendment 35 (2026-08-03) — H/W design: self-computed distributed statistics

Superseding both the fused op (amendment 34) and the earlier all-gather/norm/re-partition
shape from `latent_upsampler_ltx.py:46-82`: compute the statistics directly.

Per `(frame, group)` local sums, all-reduce **only those** -- `T x 32` scalars, against
the fused op's full-activation gather (36 MB bf16 at the widest site) -- then normalise
elementwise. Two passes, mean then *centred* variance, to avoid the `E[x^2] - E[x]^2`
cancellation that is why `GroupNorm3D` uses Welford.

No batch restriction, and the all-reduce can span **both** mesh axes, so this is what
makes true 2D H/W sharding possible. Primitives are probed independently in
`test_vae_norm_primitives_minimax_h3.py` (spatial reduce; channel<->group contraction as
a 0/1 matmul; `(T,1,1,C)` broadcast against `(T,1,HW,C)`; small-tensor all-reduce), which
then assembles them and compares against `torch.nn.GroupNorm`.

**Standing caveat on H/W in this encoder, to be settled by measurement, not argument.**
Spatial resolution collapses 256 -> 128 -> 64 -> 32 -> 16 across the six blocks, so
sharding one axis by 8 gives 32, 16, 8, 4, 2 rows per device. With a halo of 1 each side
the deepest blocks compute 4 rows to keep 2 -- 100 % overhead exactly where channels are
widest (C=1024). H/W buys per-clip *latency* and fills the ragged last wave (336 units is
10.5 waves); DP alone already fills the mesh on throughput. The A/B of DP-32 vs
spatial-8 x DP-4 vs spatial-4 x DP-8 is what fixes the per-device shape, and the
`_FP32_BLOCKINGS` sweep comes after that, since blockings are keyed on shape.

## Amendment 36 (2026-08-03) — correction: H/W-sharded GroupNorm already exists in-tree

Amendment 35 said the H/W path needs a new norm. That was wrong, and the reason it was
wrong is worth recording: I grepped `models/tt_dit/models/vae/` and concluded no VAE does
distributed GroupNorm. The LTX **upsampler** lives at
`models/tt_dit/models/upsampler/latent_upsampler_ltx.py`, outside that directory, and
`_gn_hw_sharded` (:44-81) is precisely H/W-sharded per-batch GroupNorm.

```
x = _all_gather_hw(x, pc, ccl)          # gather H on its axis, then W on its axis
if cropped: x = x[:, :, :lh, :lw, :]    # drop mesh-factor pad BEFORE the statistic
x = gn(x)                               # GroupNorm3D over the full logical extent
... re-zero-pad ... _mesh_partition_hw(x, pc)
```

It takes a **`GroupNorm3D`**, and `MiniMaxH3FrameGroupNorm` already subclasses that. So
H3 reuses it directly: norms stay built at the **global** H/W (unchanged from today),
convs are built at the **local** H/W (already done, commit `44f1eb6c402`), and each site
gathers, norms, and re-partitions.

It also already solves two things that would have cost a debugging cycle each:

* **It crops the mesh-factor padding before computing the statistic.** That is exactly the
  bug `vae_mochi.py:270-273` still carries as a live TODO ("those zeros participate in
  GroupNorm's mean/variance calculation ... the absolute result is wrong", masked there
  because it corrupts both topologies equally so cross-topology PCC stays clean).
* **`mesh_partition` must run in ROW_MAJOR**: a sub-tile-wide W shard cannot be sliced out
  of a tilized tensor, so tilize-then-partition fails.

For comparison, the reason LTX's *VAE* has no such helper is that its VAE has no spatial
GroupNorm at all -- `vae_ltx.py:353-368` uses **RMSNorm over channels** (and `norm3`,
nominally `GroupNorm(1)`, is applied as `ttnn.layer_norm` at :426). Last-dim-only norms are
shard-local by construction. H3's `GroupNorm(32)` pools over `C_group x H x W`, so it
cannot borrow that.

**H3 never takes the crop branch.** Its extents are dyadic (256/128/64/32/16), so at
spatial factor 2, 4 and 8 every one of the eight norm sites divides exactly AND every
local `H*W` is a multiple of 32 -- no mesh padding, no tilize zero-padding, no masking.
Factor 16 breaks tile alignment at two sites (local H = 1), so 8 is the practical ceiling.

Revised order: wire the H/W encoder on `_gn_hw_sharded` first, because it is proven.
`MiniMaxH3DistributedFrameGroupNorm` (amendment 35, written and in
`encoder_minimax_h3.py`) is retained as an **optimization**: it moves `T x 32` scalars per
site instead of gathering the whole activation (36 MB bf16 at the widest site, 8 sites per
unit), and it keeps fp32 where `ttnn.group_norm` forces bf16. Adopt it only if the gathers
show up in a measurement.

## Amendment 37 (2026-08-03) — decoder DP is bit-exact too; both halves now data-parallel

`test_vae_data_parallel_minimax_h3.py::test_decoder_data_parallel_independence`, 4x8, 32
distinct random token units, 2-layer decoder (independence is a property of the program;
two layers exercise fused qkv, the RoPE lane permute, SDPA, swiglu, LayerScale and the
residual chain at a eighteenth of the 4.51 GiB weight cost):

| unit | replica spread | DP vs replicated | PCC |
|---|---|---|---|
| 0 | `0.0` | `0.0` | `100.0000 %` |
| 7 | `0.0` | `0.0` | `100.0000 %` |
| 31 | `0.0` | `0.0` | `100.0000 %` |

`decode` now batches `(chunk, tile)` units across chunks, same as `encode` does across
clips. The temporal cross-fade still runs in order on the host -- it is the *decodes* that
are independent, not the stitching.

Host memory is bounded deliberately. Encode prepares units **per wave** rather than up
front (`permute().contiguous()` materialises 13.4 MB per unit, so preparing all of them
costs 19 GB at 1440P/10s, while the source slices themselves are free views). Decode groups
chunks into `_DECODE_WAVES_IN_FLIGHT = 4` waves' worth before stitching, because a decoded
tile is 22 MB and holding all 308 units of a 768P/5s video would be 6.8 GB (~29 GB at
1440P/10s). Groups are whole chunks, so a group never straddles a stitch boundary.

Full visual suite re-run green after both refactors: **9 passed**, encoder PCC 99.9829 -
99.9883 %, decoder 99.9977 - 99.9979 %. Note the suite is `SINGLE_DEVICE`, so it gates the
tiling/stitch *order* refactor; DP-at-32 is gated separately by the two independence tests.

## Amendment 38 (2026-08-03) — measured data-parallel throughput: 30x, near-linear

`test_performance_vae_minimax_h3.py::test_visual_data_parallel_throughput`, 4x8 mesh, one
work unit per device, weights replicated:

| | single device (amendment 32) | 32-unit wave | per unit | speedup |
|---|---|---|---|---|
| encoder tile `(17,256,256)` | 3.544 s | **3.714 s** | **0.1161 s** | **30.5x** |
| decoder invocation, 1797 tokens | 0.755 s | **0.778 s** | **0.0243 s** | **31.1x** |

**95-97 % scaling efficiency.** A 32-unit wave costs ~5 % more wall clock than a single
unit did alone, which is the expected result for zero-CCL SPMD -- the only additions are
the wider host transfer and the mesh dispatch.

Projected full-video wall time:

| working point | encode | decode | total |
|---|---|---|---|
| 768P / 5 s | 26.0 s (224 units, 7 waves) | 5.4 s (196 units, 7 waves) | **31.4 s** |
| 768P / 10 s | 52.0 s (14 waves) | 10.1 s (13 waves) | **62.1 s** |
| 1440P / 5 s | 96.6 s (26 waves) | 17.9 s (23 waves) | **114.5 s** |
| 1440P / 10 s | 182.0 s (49 waves) | 35.8 s (46 waves) | **217.8 s** |

768P/5s goes from ~941 s sequential to **31.4 s**, a **30x** end-to-end win from
parallelism alone, nothing tuned.

**The encoder is now 83 % of total time** (26.0 s against 5.4 s). DP scaled both halves
equally, so the amendment-32 imbalance is unchanged and its cause is unchanged: the encoder
runs at ~2.3 TFLOP/s against the decoder's 14.0 because its blockings are stubs. The
`_FP32_BLOCKINGS` sweep targets exactly that ~6x, which would put 768P/5s near 10 s.

## Amendment 39 (2026-08-03) — the H/W halo gate found a real bug in the committed conv

The conv threading committed in `44f1eb6c402` had **never executed on device**. Its first
run failed immediately:

```
links.append(max(1, min(math.prod(x_BTHWC.shape[:dim]), self.ccl_manager.num_links)))
TypeError: __getitem__(): incompatible function arguments ... Invoked with types:
           ttnn._ttnn.types.Shape, slice
```

`ttnn.Shape` supports integer indexing but **not slicing**. Fixed with `list(...)`. Worth
recording as a general trap: `x.shape[:n]` is silently fine on a torch tensor and throws on
a ttnn one, so it survives any host-side review.

Two further failures were the *gate* being wrong, not the model, and both are worth keeping
as notes on how to test a halo:

* `F.pad(mode="reflect")` on a **5D** tensor requires 6 pad values and would pad T as well.
  H3 pads only the spatial axes, so the reference folds to 4D `(B*T, C, H, W)` first.
* Comparing *gathered padded shards* against the globally padded tensor is wrong by
  construction: the halos are interior duplicates, so four shards of height 8 padded by 1
  each side concatenate to 40, not the true padded 34. The gate now compares **each device's
  shard against its window of the globally padded reference**, which is also strictly
  stronger -- it pins down which rows each device received, so an interior halo taken from
  the wrong neighbour fails, not just a bad global edge.

## Amendment 40 (2026-08-04) — the encoder blocking sweep: 1.70x, and two traps

`sweep_conv3d_minimax_h3.py` brute-forces every legal blocking per conv shape and times it
on hardware under a trace. Encoder wave of 32 units: **3.714 s -> 2.187 s (1.70x)**.
768P/5s total **31.4 s -> 19.9 s**. Correctness unchanged: 36 passed, encoder PCC
99.982-99.992 %.

Per-layer gains against the conv3d.py table baseline ranged 2.5x to 25.6x, but the
end-to-end 1.70x is the number that counts -- the baseline column is what conv3d.py's own
table gives, not what the H3 stubs gave, since the sweep tool never imports
`conv_minimax_h3`.

Two traps, both of which produced a table that looked right and failed:

1. **The level channel map.** `block_in_channels = (block_out[0],) + block_out[:-1]`, so
   against `block_out = (128,256,256,512,512,1024)` the levels are 128->128, 128->256,
   **256->256**, 256->512, 512->512, 512->1024. The first sweep list had 512-channel convs
   at b2's spatial size when b2 is 256->256. Nothing complains -- the swept values are
   simply for shapes the model never runs.
2. **The sweep's L1 model is optimistic.** It times a bare conv3d; the model's conv also
   carries fp32 weights and a ROW_MAJOR output, so its circular buffers are larger. Up to
   **13 sweep-approved candidates were rejected in a row** before one built. Entries are now
   chosen by constructing the real conv at every layer sharing the key (`validate_table.py`).
   Overflow is driven by `C_in_block x C_out_block x T_out_block`, not the H/W split: two
   different H/W splits at 64x128x6 both failed at an identical **1753984 B** against a
   1572864 B L1. That unchanging byte count across three different tables is what proved the
   failing conv was one the table never covered -- the same tell as the audio matmul in
   amendment 30.

## Amendment 41 (2026-08-04) — the ViT decoder is host-dispatch bound; per-op tuning is unmeasurable

Following the question of whether the ViT is as fast as wan/ltx get theirs: it calls
`ttnn.transformer.scaled_dot_product_attention` **bare**, while `attention_wan.py:120` and
`attention_ltx.py:158` both configure an `SDPAProgramConfig` and a HiFi2 compute kernel
config. `Linear` already carries HiFi2 + packer_l1_acc (`linear.py:71`), so SDPA was the
only untuned op in the block.

Copying that configuration did **not** help, and the reason turned out to be more
fundamental than the choice of chunk size. Decoder wave, 32 units, same code, repeated:

| configuration | min | avg | max |
|---|---|---|---|
| ttnn defaults | 0.3383 | 0.7158 | 0.9854 |
| HiFi2 only | 0.6519 | 0.7993 | 0.9267 |
| HiFi4 only | 0.8479 | 0.8821 | 0.9115 |
| q=k=64 + HiFi2 | 0.5947 | 0.8107 | 0.9639 |
| q=k=128 + HiFi2 | 0.8731 | 0.9227 | 0.9757 |
| q=k=256 + HiFi2 | 0.5875 | 0.8652 | 0.9532 |
| q=k=512 + HiFi2 | L1 overflow | | |

**The same configuration spans 0.34-0.99 s/wave -- a 3x spread that swamps every difference
between configurations.** The perf harness shows it too: the unmodified decoder measured
0.652 s at 00:20 and 0.866 s at 00:52. An earlier note in this session that the tuned SDPA
"made the decoder slower (0.652 -> 0.876)" was reading that noise as signal; it is not
supported, and neither is any claim that the defaults are better.

At ~540 ops per invocation (36 layers x ~15 ops) the decoder is **host-dispatch bound**,
which plan section 7.1 predicted in as many words: "at ~540 ops per decoder invocation x 308
invocations, trace capture is the whole perf story; without it this is host-bound". The
0.3383 s minimum is roughly what the device does when the host keeps up; the 0.99 s maximum
is host starvation.

So SDPA is left on the ttnn defaults with that reasoning recorded in the code, and **trace
capture is the next decoder work**. No per-op decoder tuning -- SDPA config, bfp8 weights,
matmul program configs -- can be evaluated until dispatch is amortised, because the
measurement floor is wider than the effects being measured.

## Amendment 42 (2026-08-04) — per-op profile of one ViT layer: 36 % is data movement

Whole-decoder wall clock could not attribute anything (amendment 41: same code, 0.34-0.99 s
per wave). The device profiler sidesteps host jitter entirely -- it reports per-op *device*
time. Via tt-buddy's tracy workflow:

```
python -m tracy -p -r -m pytest models/tt_dit/tests/models/minimax_h3/profile_vit_layer_minimax_h3.py
```

(`websockets` had to be installed into the venv first; `python -m pip` is absent, use
`uv pip install --python python_env/bin/python`.)

One layer plus `proj_in`/`proj_out`, second iteration (the first populates the program
cache), **15.527 ms of device time over 63 ops**:

| op | n | total us | mean us | % | fidelity |
|---|---|---|---|---|---|
| SDPAOperation | 2 | 4277.8 | 2138.9 | **27.6 %** | HiFi2 |
| **ReshapeViewDeviceOperation** | 5 | 3998.9 | **799.8** | **25.8 %** | -- |
| MinimalMatmulDeviceOperation | 10 | 3029.0 | 302.9 | 19.5 % | HiFi2 |
| **TransposeDeviceOperation** | 5 | 1569.8 | **314.0** | **10.1 %** | -- |
| BinaryNgDeviceOperation | 22 | 1315.1 | 59.8 | 8.5 % | HiFi4 |
| LayerNormDeviceOperation | 7 | 567.8 | 81.1 | 3.7 % | HiFi4 |
| TypecastDeviceOperation | 4 | 424.7 | 106.2 | 2.7 % | HiFi4 |
| UnaryDeviceOperation | 4 | 172.2 | 43.1 | 1.1 % | -- |

**Reshape + Transpose are 35.9 % of device time -- more than the matmuls (19.5 %), and a
reshape averaging 800 us is pure data movement that should be nearly free.**

The source is `MiniMaxH3ViTAttention.forward`, which builds heads by hand:

```python
query, key, value = (ttnn.reshape(part, (b, s, heads, hd)) for part in ttnn.chunk(qkv, 3, dim=-1))
query, key, value = (ttnn.permute(t, (0, 2, 1, 3)) for t in (query, key, value))
...
attended = ttnn.reshape(ttnn.permute(attended, (0, 2, 1, 3)), (b, s, heads * hd))
```

That is 4 reshapes and 4 transposes per layer. **Both wan and ltx use the fused op instead**
-- `ttnn.experimental.nlp_create_qkv_heads` (`attention_wan.py:401`, `attention_ltx.py:463`)
-- which does chunk + reshape + head-permute in one, with `nlp_concat_heads` for the return
trip. This is the answer to "is the ViT as fast as it gets": no, and the gap is not the SDPA
config that was tried first, it is the hand-rolled head plumbing around it.

Next: replace with `nlp_create_qkv_heads` / `nlp_concat_heads`, re-profile, then revisit the
HiFi4 elementwise ops (BinaryNg at 8.5 % is running HiFi4 where HiFi2 would do).

## Amendment 43 (2026-08-04) — fused head ops: 1.45x on ViT layer device time

Acting on amendment 42: replaced the hand-rolled head plumbing in
`MiniMaxH3ViTAttention.forward` with `ttnn.experimental.nlp_create_qkv_heads` /
`nlp_concat_heads`, the ops wan (`attention_wan.py:401`) and ltx (`attention_ltx.py:463`)
already use.

One layer, second iteration, device time: **15.527 ms -> 10.707 ms (1.45x)**, 63 -> 54 ops.

| op | before us | after us |
|---|---|---|
| ReshapeViewDeviceOperation | 3998.9 | **0** |
| TransposeDeviceOperation | 1569.8 | **0** |
| SliceDeviceOperation | 128.8 | **0** |
| NlpCreateHeadsDeviceOperation | 0 | 113.9 |
| NLPConcatHeadsDeviceOperation | 0 | 623.2 |

**5697 us of data movement became 737 us -- 7.7x on that portion**, and it is now gone as a
category. `transpose_k_heads=False` is required (SDPA wants K as `[B,H,S,D]`), and the
layout the op expects -- `[Q all heads | K all heads | V all heads]` -- is exactly what
`_prepare_torch_state`'s `cat([q, k, v], dim=0)` already produced, so the RoPE lane permute
baked into the q/k weights carries over untouched.

Correctness: `test_vae_decoder_minimax_h3.py` 5 passed, PCC 99.9997 % / 99.9876 %.

End-to-end, 32-unit waves: decoder **0.866 s -> 0.538 s**. 768P/5s **21.4 s -> 19.4 s**
(encode 15.7, decode 3.8). Note the end-to-end decoder gain (1.61x) exceeds the device-time
gain (1.45x) because removing 9 ops per layer also removes 324 dispatches per invocation --
this helps the host-bound regime of amendment 41 as well as the device.

**The decoder is no longer the problem: encode 15.7 s against decode 3.8 s.** Remaining
decoder headroom, in order: SDPA is now 40 % of layer device time (4282 us, unchanged by
this work), NLPConcatHeads is oddly 5.5x NlpCreateHeads, and BinaryNg/LayerNorm/Typecast
still run HiFi4. But the encoder is 80 % of wall time and has never been profiled per-op --
that is where the 3 s target is won.

## Amendment 44 (2026-08-04) — per-op profile of the encoder, and bf16: 2.83x

Profiling the encoder (`profile_encoder_minimax_h3.py`), one unit, second iteration:
**3167.38 ms of device time over 550 ops**.

| op | n | total ms | % | fidelity |
|---|---|---|---|---|
| Conv3dDeviceOperation | 46 | 1383.9 | 43.7 % | HiFi4 |
| **TypecastDeviceOperation** | 70 | **661.0** | **20.9 %** | HiFi4 |
| BinaryNgDeviceOperation | 17 | 494.8 | 15.6 % | HiFi4 |
| GroupNormDeviceOperation | 35 | 234.8 | 7.4 % | HiFi4 |
| **UntilizeDeviceOperation** | 35 | 234.7 | 7.4 % | HiFi4 |
| ConcatDeviceOperation | 123 | 109.9 | 3.5 % | -- |
| UnaryDeviceOperation | 35 | 21.6 | 0.7 % | HiFi4 |
| TilizeWithValPadding | 35 | 16.6 | 0.5 % | HiFi4 |
| SliceDeviceOperation | 154 | 10.1 | 0.3 % | -- |

**70 typecasts costing 661 ms, a fifth of the encoder.** That is the bf16 island this file's
docstring describes: the encoder ran fp32 while `ttnn.group_norm` is bf16-only, so all
thirteen norm sites round-tripped fp32 -> bf16 -> tilize -> norm -> untilize -> fp32.
Typecast + Untilize + Tilize + Concat + Slice together are **32.6 %** of device time.

Running the encoder in **bf16 throughout** removes the round trip and speeds the convs:

| | wave (32 units) | per unit | PCC vs reference |
|---|---|---|---|
| fp32 | 2.425 s | 75.8 ms | 99.5399 % |
| **bf16** | **0.857 s** | **26.8 ms** | 99.5014 % |

**2.83x for 0.04 percentage points of PCC.** The encoder's precision floor was already the
bf16 GroupNorm islands, so making the rest match costs almost nothing -- the fp32 activations
were being rounded to bf16 thirteen times anyway.

`MiniMaxH3Encoder3d` now defaults to `dtype=ttnn.bfloat16`, and its `forward` casts the input
to that dtype: `conv3d` requires input and weight dtypes to match exactly
(`conv3d_device_operation.cpp:82`), and making every call site know the encoder's compute
dtype is worse than one cast at the entry.

Gates: **16 passed** (encoder + e2e), encoder PCC 99.982-99.988 %, decoder 99.9977-99.9979 %.

**Correction on the 60 `conv3d blocking [fallback]` warnings under bf16: they are not
misses.** `conv3d.py:607-621` has three tiers -- `[exact]` (a mesh+shape+T+spatial key in
`_BLOCKINGS`), `[fallback]` (a channel-key hit in `_DEFAULT_BLOCKINGS`), and a true miss
(`in_channels, 32, 1, 1, 1`). `[fallback]` is the channel-key tier matching, and it is logged
at **warning** level, which reads like a failure. The logged values confirm it: `channel_key=
(1024, 48, (1,3,3)) -> Cin=128 Cout=32 T=1 H=16 W=2` is exactly the swept entry
`(1024, 48): (128, 32, 1, 16, 2)`. **The sweep applies fully under bf16**, and the 2.83x
already includes it.

What is still open is that the blockings were *measured* in fp32. bf16 halves the activation
bytes, so larger blocks now fit L1 and the fp32-optimal choice is unlikely to be bf16-optimal
-- `run_sweep` takes no dtype argument, so a bf16 sweep needs that plumbed first. Worth doing,
but it is a fresh optimisation rather than recovering something the sweep failed to deliver.

## Amendment 45 (2026-08-04) — bf16 encoder re-profile: the target is now the GroupNorm layout round-trip

Re-profiled after the bf16 switch, because amendment 44's breakdown was measured in fp32 and
two conclusions drawn from it were wrong.

**Encoder device time 3167.4 ms -> 1117.0 ms (2.84x)**, 550 -> 517 ops -- matching the 2.83x
wall clock, so the win is real device work, not dispatch.

| op | fp32 ms | bf16 ms | bf16 % | fidelity |
|---|---|---|---|---|
| Conv3dDeviceOperation | 1383.9 | **266.6** | 23.9 % | HiFi2 |
| **UntilizeDeviceOperation** | 234.7 | **235.8** | **21.1 %** | HiFi4 |
| **GroupNormDeviceOperation** | 234.8 | **235.7** | **21.1 %** | HiFi4 |
| BinaryNgDeviceOperation | 494.7 | 121.0 | 10.8 % | HiFi4 |
| **TilizeWithValPaddingDeviceOperation** | 16.6 | **118.8** | 10.6 % | HiFi4 |
| ConcatDeviceOperation | 109.9 | 97.6 | 8.7 % | -- |
| UnaryDeviceOperation | 21.6 | 23.4 | 2.1 % | HiFi4 |
| SliceDeviceOperation | 10.1 | 14.3 | 1.3 % | -- |
| TypecastDeviceOperation | 661.0 | **3.7** | 0.3 % | HiFi4 |

Two corrections to the follow-ups listed in amendment 44:

* **"Encoder Conv3d is 43.7 % at HiFi4" is stale.** The conv already selects fidelity by
  dtype (`math_fidelity=HiFi4 if (is_blackhole() and dtype == float32) else HiFi2`), so the
  bf16 switch moved it to HiFi2 by itself. Conv3d fell **5.2x** (1383.9 -> 266.6 ms) and is
  now 23.9 %, no longer the largest item.
* **"BinaryNg at 15.6 %" is not a fidelity problem either** -- it fell 4x on its own.

**The real target is the GroupNorm layout round-trip: Untilize 21.1 % + GroupNorm 21.1 % +
Tilize 10.6 % = 52.8 % of encoder device time**, more than double the convolutions. It is
also the only part that did not improve -- Untilize and GroupNorm are unchanged to within
1 ms (234.7 -> 235.8, 234.8 -> 235.7), because `ttnn.group_norm` was always bf16 internally,
and Tilize got **worse** (16.6 -> 118.8 ms) now that it is a larger share of a smaller total.

The cause is structural: 35 tilize and 35 untilize per unit exist only because
`ttnn.group_norm` requires TILE layout while `ttnn.experimental.conv3d` requires ROW_MAJOR,
so every one of the thirteen norm sites converts in and back out.

Two candidate fixes, in order of expected value:

1. **Use `MiniMaxH3DistributedFrameGroupNorm`** (written in amendment 35, currently only for
   the H/W path). It computes the statistics itself from sums and normalises elementwise, so
   it needs neither the tilize/untilize pair nor `ttnn.group_norm` at all. Setting
   `spatial_factor=1` makes it a drop-in for the unsharded path.
2. Keep activations tilized across the norm where the neighbouring conv can accept it, to
   collapse pairs of conversions.

Either would attack over half the remaining encoder time. The projection at 768P/5s is
currently ~7 s (encode ~5.6, decode 3.8) against the 3 s target; removing the round-trip is
the plausible route to roughly 4 s.

## Amendment 46 (2026-08-04) — stats GroupNorm wired for the unsharded path; perf unproven

Acting on amendment 45's target (Untilize + GroupNorm + Tilize = 52.8 % of encoder device
time, all of it a layout round-trip forced by `ttnn.group_norm` wanting TILE while `conv3d`
wants ROW_MAJOR): generalised `MiniMaxH3DistributedFrameGroupNorm` to the unsharded case
(`spatial_factor=1` skips the all-gather and needs no `ccl_manager`) and routed the encoder's
norms through a `make_frame_group_norm` factory behind `MINIMAX_H3_USE_STATS_GROUPNORM`.

**Correct**: `test_vae_encoder_minimax_h3.py` 7 passed, PCC **99.9303 % / 99.9264 %** against
99.988 % for the `group_norm` path. The drop is expected -- two-pass mean/variance in bf16
rather than group_norm's Welford -- and stays well clear of the 0.99 gate.

**Performance unproven, so the flag defaults to False.** The wall-clock comparison was
inconclusive: encoder wave 0.857 -> 0.978 s, but the *untouched* decoder moved 0.538 ->
0.654 s in the same run. A +21 % drift on code that did not change means the encoder's +14 %
carries no signal. This is amendment 41's problem again, and the lesson from amendment 43 is
that the device profiler settles it and wall clock does not: profile
`profile_encoder_minimax_h3.py` with both flag values and compare Untilize + GroupNorm +
Tilize against the 52.8 % baseline. If the round-trip is gone the flag should flip to True.

**Process note for the next session:** several waits in this session used unbounded
`until ! pgrep ...; do sleep N; done` loops, which run to the tool timeout even when the job
finished in a minute. Bound them (`for i in $(seq 1 12); do pgrep ... || break; sleep 10;
done`) with a cap set from the expected runtime.

## Amendment 47 (2026-08-04) — LayerScale folded into the projections

`scale1` / `scale2` are per-**output**-channel multipliers applied immediately after
`to_out` and `ff2`, so `to_out(x) * scale1` is `x @ (W_out * scale1) + b_out * scale1`.
Folded at load time in `MiniMaxH3TransformerBlock._prepare_torch_state`; the two `Parameter`s
are gone and the block forward is now just two residual adds.

Exact, not approximate: `test_vae_decoder_minimax_h3.py` 5 passed at PCC **99.9997 % /
99.9876 %** -- the same figures as before the fold, to four decimal places.

Removes two of the 22 `BinaryNg` calls per layer (13.2 % of layer device time in total).

**On fp32 vs bf16 in the decoder:** the remaining fp32 is *required for reference parity*,
not conservatism. The pinned reference computes `norm1(h.float())` and `norm_q(q.float())`
explicitly, so the q/k RMS round trip (4 typecasts/layer, 3.2 % of device time) matches it
exactly. The encoder was the opposite case -- fp32 throughout with no reference basis -- and
that is already fixed (amendment 44, 2.84x).

Open, as a *deliberate divergence* rather than a fix: the encoder experiment showed bf16
norms cost 0.04 percentage points of PCC, so the decoder's fp32 q/k norms could likely go
bf16 for the 3.2 % typecast plus cheaper LayerNorms. That trades reference parity for speed
and should be a measured decision, not a silent one.

## Amendment 48 (2026-08-04) — LayerScale fold + HiFi2 elementwise: 1.015x, i.e. marginal

Profiled both changes together against the post-head-fusion baseline:

**10.707 ms -> 10.552 ms (1.015x)**, 54 -> 50 ops.

| op | before us | after us | delta |
|---|---|---|---|
| SDPAOperation | 4282.4 | 4285.9 | +3.5 |
| MinimalMatmul | 3161.0 | 3036.2 | -124.8 |
| BinaryNg | 1417.3 | 1143.8 | **-273.5** |
| NLPConcatHeads | 623.2 | 626.6 | +3.5 |
| LayerNorm | 546.9 | 557.7 | +10.8 |
| Typecast | 347.1 | 480.8 | +133.7 |
| NlpCreateHeads | 113.9 | 206.9 | +93.0 |

The fold did what it should -- `BinaryNg` fell 273 us -- but Typecast and NlpCreateHeads rose
by roughly as much, so the net is ~155 us. The arithmetic that should have come first: **only
2 of the 22 BinaryNg calls were LayerScale**, so folding them was never worth the 13.2 % that
category represents. Cost of the lesson: an unnecessary profile run.

The HiFi2 compute config on the q/k `rms_norm` moved nothing measurable (LayerNorm +10.8 us).
It is kept because it is free and correct, not because it was shown to help.

Both are retained: correct (5 passed, PCC 99.9997 % / 99.9875 %, unchanged), 4 fewer ops, and
the fold removes two `Parameter`s. But neither is a performance answer.

**SDPA is unmoved at 4285.9 us = 40.6 % of the layer and is the only remaining item of size.**
Everything else in the layer combined is 6.3 ms. Nothing short of attacking SDPA -- bfp8_b
K/V, a profiler-driven chunk sweep, or a different attention decomposition -- changes the
decoder materially. The two anomalies worth pairing with it: NLPConcatHeads measured 42.6 us
and 580.5 us for identical work in the same trace, and both head ops run on **57 cores** while
every other op gets 120.

## Amendment 49 (2026-08-04) — tt-perf-report: 97.1 % of wall time is op-to-op gap

`tt-perf-report` on the bf16 encoder profile states it outright:

```
These ops have a >6 us gap since the previous operation. Running with tracing could save
47463439 us (97.1% of overall time)
```

Its stacked report, over the whole run (weight-load ops included, so percentages differ from
the second-iteration-only cut in amendment 45):

| % | op | device time | count | category |
|---|---|---|---|---|
| 32.24 % | Conv3dDeviceOperation | 450,439 us | 66 | Other |
| **32.09 %** | **GroupNormDeviceOperation** | 448,402 us | 50 | Compute |
| 13.72 % | ConcatDeviceOperation | 191,654 us | 180 | TM |
| 12.76 % | PermuteDeviceOperation | 178,236 us | 63 | TM |
| 2.37 % | UnaryDeviceOperation | 33,090 us | 50 | Compute |
| 2.30 % | TilizeWithValPadding | 32,183 us | 126 | TM |
| 2.02 % | BinaryNgDeviceOperation | 28,233 us | 24 | Compute |
| 1.53 % | UntilizeDeviceOperation | 21,408 us | 58 | TM |

**This reframes the whole optimisation order.** Amendment 41 inferred host-dispatch binding
from timing variance; `tt-perf-report` quantifies it, and at **97.1 %** it dwarfs every op-level
change made in this session put together. The measured wins so far -- head fusion 1.45x, bf16
2.84x, blockings 1.70x -- were all reductions in *device* time, which is ~3 % of wall clock.
Trace is not a follow-up item; it is the item.

Second finding: **GroupNorm is 32.09 %, co-equal with Conv3d at 32.24 %.** That corroborates
amendment 45's target from an independent tool, and makes
`MINIMAX_H3_USE_STATS_GROUPNORM` the right second move after trace.

**Blocked, and how to unblock it:** profiling the encoder with the stats norm enabled fails
report generation --
`AssertionError: Device data missing: Op 1034240 not present in cpp_device_perf_report.csv`.
That is Tracy's 1000-op-per-device buffer (tt-buddy's `profiler/tracy.md` warns about it): the
encoder is already ~550 ops per iteration x 2 iterations, and the stats norm adds more. Fix
by calling `ReadDeviceProfilerResults(device)` between iterations, or by profiling a single
down_block instead of the whole encoder. The flag stays `False` until that profile runs.

Also worth noting from the per-op listing: the op-to-op gaps are enormous and irregular
(492,684 us, 486,788 us, 535,331 us between early ops), which is weight upload during
construction rather than steady-state inference -- so the 97.1 % figure is an upper bound on
what trace recovers for a warmed pipeline. The right next measurement is a traced run, which
gives the real number directly rather than another estimate.

## Amendment 50 (2026-08-04) — trace capture attempt: blocked on `!trace_id_.has_value()`

Acting on amendment 49 (97.1 % of wall time is op-to-op gap), tried to capture a mesh-sized
wave with `utils/tracing.Tracer`. Harness kept as
`models/tt_dit/tests/models/minimax_h3/measure_trace_minimax_h3.py`.

Untraced baseline reproduces cleanly: **encoder 0.8512 / 0.8531 s per 32-unit wave** across
two runs, consistent with the bf16 A/B's 0.857 s.

Capture fails:

```
RuntimeError: TT_FATAL @ tt_metal/distributed/fd_mesh_command_queue.cpp:760:
!trace_id_.has_value()
```

Two fixes tried, neither sufficient:

1. Moved the capturing call out of the timing loop -- the loop calls
   `ttnn.synchronize_device` after every invocation, and synchronising *during* capture is
   illegal. Correct to fix, did not resolve it.
2. Confirmed `trace_region_size=3e8` is passed at `open_mesh_device`.

**Most likely cause, untested:** the encoder allocates inside `forward`. `causal_pad_t` calls
`ttnn.zeros(..., device=mesh_device)` on every invocation, and `reflect_pad_hw` builds its pad
by slice-and-concat -- both allocate during what would be the capture window, and `Tracer`'s
own docstring warns "tensors allocated after trace capture may be overwritten". The fix is
probably to hoist the causal zero frames into a persistent buffer allocated at construction
(the same treatment `ccl_manager.neighbor_pad_persistent_buffer` already gives the halo),
then retry. `Tracer(..., prep_run=False)` is worth trying as a second lever.

Nothing was committed to the model for this -- the encoder is unchanged. This is a recorded
dead end with a specific next hypothesis, not a partial implementation.

## Amendment 51 (2026-08-04) — RETRACTION of amendment 49: the 97.1 % gap figure is wrong

Amendment 49 quoted `tt-perf-report`'s "Running with tracing could save 47463439 us (97.1 %
of overall time)" and concluded trace was the dominant lever. **That is wrong for warm
inference**, and the error was mine, not the tool's: the report analyses the *entire* CSV
("No signposts found in the file. Using the entire file for analysis"), which includes weight
upload and construction. I repeated the headline without checking the gap distribution.

Using the `OP TO OP LATENCY [ns]` column -- which amendment 49 never looked at -- on the same
profile:

| window | device | op-to-op gap | gap share |
|---|---|---|---|
| last 500 ops | 1115.6 ms | 9212.9 ms | 89.2 % |
| last 400 ops | 1100.1 ms | 5171.5 ms | 82.5 % |
| **last 300 ops** | **568.6 ms** | **110.0 ms** | **16.2 %** |

**Median op-to-op gap is 0.6 us; the mean is 18425.9 us.** That distribution is the whole
story -- a handful of 650-880 ms gaps (weight upload) drag the mean up by four orders of
magnitude. The gap share collapses as the window excludes more construction, so in steady
state **device time is the bottleneck, not dispatch**.

Warm breakdown, gaps excluded:

| op | n | device | % |
|---|---|---|---|
| Conv3dDeviceOperation | 48 | 265.9 ms | 23.8 % |
| **UntilizeDeviceOperation** | 36 | 235.8 ms | **21.1 %** |
| **GroupNormDeviceOperation** | 36 | 235.3 ms | **21.1 %** |
| BinaryNgDeviceOperation | 18 | 121.0 ms | 10.8 % |
| **TilizeWithValPaddingDeviceOperation** | 36 | 118.7 ms | **10.6 %** |
| ConcatDeviceOperation | 129 | 97.5 ms | 8.7 % |

**Untilize + GroupNorm + Tilize = 52.8 %**, confirming amendment 45 independently now that
gaps are excluded. The GroupNorm layout round-trip is the target; trace is not.

Consequences:

* Amendment 49's ordering ("trace first, up to ~30x") is **withdrawn**. Amendment 50's trace
  blocker is therefore not on the critical path -- worth fixing eventually, not first.
* Amendment 41's inference that the *decoder* is host-dispatch bound rests on wall-clock
  variance, not on gap data, and should be re-checked the same way before being acted on.
* **Method note:** never quote `tt-perf-report`'s tracing-saving headline without first
  checking the op-to-op gap *distribution* (median vs mean) on a warm window. Prefer
  signposts, or slice the tail, so construction is excluded. The per-op device ranking from
  the same report was correct throughout -- it was only the gap aggregate that misled.

## Amendment 52 (2026-08-04) — stats GroupNorm settled by profile: 1.78x on the block, flag ON

The measurement amendment 46 was waiting for. Profiling **one down_block** rather than the
whole encoder sidesteps Tracy's 1000-op buffer (block 0 is ~90 ops against the encoder's
~550), which is what made the amendment-50 profile fail. Same block, flag both ways, warm
iteration:

| | device time | ops |
|---|---|---|
| `ttnn.group_norm` | 852.74 ms | 82 |
| **stats GroupNorm** | **478.66 ms** | 140 |
| | **1.782x** | |

| op | group_norm | stats | delta |
|---|---|---|---|
| GroupNormDeviceOperation | 219.22 | **0** | -219.22 |
| UntilizeDeviceOperation | 218.72 | **14.14** | -204.58 |
| TilizeWithValPaddingDeviceOperation | 71.60 | **0** | -71.60 |
| Conv3dDeviceOperation | 139.63 | 122.01 | -17.62 |
| ConcatDeviceOperation | 100.61 | 88.15 | -12.46 |
| BinaryNgDeviceOperation | 70.35 | 147.39 | +77.04 |
| TilizeDeviceOperation | 0 | 50.58 | +50.58 |
| ReduceDeviceOperation | 0 | 23.34 | +23.34 |

The layout round-trip is gone: GroupNorm and TilizeWithValPadding to zero, Untilize down 94 %.
The replacement costs (BinaryNg, Tilize, Reduce) total +151 ms against -525 ms removed.

**Op count rose 82 -> 140 while device time fell 44 %**, which independently confirms
amendment 51: this workload is device-bound, not dispatch-bound. Under amendment 49's
withdrawn "97.1 % gap" reading, adding 58 ops would have been a clear loss.

`MINIMAX_H3_USE_STATS_GROUPNORM` is now **True**. Gates: **16 passed** (encoder + e2e),
encoder PCC 99.9264 %, e2e 99.9940 %, decoder 99.9977-99.9979 %. The encoder's PCC is lower
than the group_norm path's 99.988 % -- two-pass mean/variance in bf16 rather than Welford --
and still far clear of the 0.99 gate.

End-to-end, 32-unit waves: encoder **0.911 s**, decoder 0.652 s.
**768P/5s projection: encode 6.4 s + decode 4.6 s = 10.9 s.**

Note the whole-encoder gain is far smaller than the block's 1.78x: block 0 is the largest
spatial extent, where the round trip cost most, and the deeper levels have far less to
recover. Wall-clock run-to-run drift (amendment 41) also remains wide enough that 0.857 vs
0.911 s across runs carries little signal -- the block profile is the trustworthy number here,
and it says the change is right.

## Amendment 53 (2026-08-04) — correction to amendment 50's trace/profile diagnosis

Amendment 50 attributed the stats-norm profile failure to Tracy's 1000-op-per-device buffer.
**That was wrong.** Added `ttnn.ReadDeviceProfiler(mesh_device)` between iterations in
`profile_encoder_minimax_h3.py` (the documented drain), and the full encoder still fails at
the *same* op:

```
AssertionError: Device data missing: Op 1034240 not present in
cpp_device_perf_report.csv for device 0 (trace_id=None)
```

Two facts that rule the buffer out: the identical op ID across runs (a buffer overflow would
drop a different op each time), and the **down_block profile with the stats norm succeeded**
(amendment 52) despite the same drain being absent there.

So a specific op emits no device data. The stats norm introduces two op types the
`group_norm` path does not -- `ReduceDeviceOperation` (`ttnn.sum` over the spatial axis) and
`MatmulDeviceOperation` (the channel<->group contraction) -- and one of them is the likely
culprit, plausibly degenerating to zero device work at some level's shape. The drain is
harmless and is kept.

**Consequence:** the whole-encoder gain from the stats GroupNorm remains **unquantified**.
The block-level 1.782x (amendment 52) is measured and trustworthy; extrapolating it to the
encoder is not, because block 0 has the largest spatial extent and the deeper levels have
much less round-trip to recover. Wall clock cannot substitute here -- the encoder wave read
0.857 s before the change and 0.911 s after, which is inside the run-to-run drift.

To close it: identify op 1034240 by global call count in `profile_log_device.csv`, or profile
each down_block separately (blocks 1-5, as block 0 already is) and sum. The per-block route
needs no new tooling and is the cheaper of the two.

## Amendment 54 (2026-08-04) — SDPA swept at its real shape: 2.88x

SDPA was 40 % of ViT layer device time and untouched. Swept as a **single op** at the
decoder's exact shape (`[1, 32, 1824, 64]` bf16), min-of-20 -- measurable where whole-model
wall clock is not, because one op has one dispatch:

| config | ms | TF/s (32 dev) | speedup |
|---|---|---|---|
| ttnn defaults | 1.448 | 602.4 | 1.00x |
| **q=k=128 + HiFi2** | **0.502** | **1737.0** | **2.88x** |

Applied to `MiniMaxH3ViTAttention`. Gates: **5 passed**, PCC 99.9997 % / 99.9877 % --
unchanged from the defaults, so the fidelity drop costs nothing measurable here.

**This reverses an earlier judgement.** Amendment 41 recorded that an explicit SDPA config
"moved the decoder wave from 0.652 s to 0.876 s" and concluded the defaults were better. That
was noise -- the same code spans 0.34-0.99 s/wave -- and the conclusion was wrong. The
correct method is the one used here and in amendment 52: measure the *op*, or measure device
time under the profiler; never judge a per-op change by whole-model wall clock.

At 40 % of layer device time, 2.88x on SDPA is worth roughly 26 % off the ViT layer.

**Sweep hazard:** the sweep hung after the second configuration (`q=k=128 LoFi` or later) and
had to be killed, then `tt-smi -glx_reset`, then a stale process holding
`CHIP_IN_USE_0_PCIe` had to be killed by PID before the device would initialise again. Larger
chunk sizes were never measured, so **128 is the best of {default, 128}, not a proven
optimum** -- 192/256/384/512 remain untested. Note also that a `serve_wasm.py` process from
another user was mistaken earlier for an unkillable leak of mine; it is not.

## Amendment 55 (2026-08-04) — SDPA chunk sweep continued: 192 marginally beats 128

Amendment 54 shipped `q=k=128` while noting it was "the best of {default, 128}, not a proven
optimum". Continued the sweep with each configuration in its **own process** -- the combined
loop is what hung -- and with a hard per-config timeout:

| config | ms | TF/s (32 dev) | vs default |
|---|---|---|---|
| ttnn defaults | 1.448 | 602.4 | 1.00x |
| q=k=128 + HiFi2 | 0.502 | 1737.0 | 2.88x |
| **q=k=192 + HiFi2** | **0.491** | **1776.6** | **2.95x** |
| q=k=256 + HiFi2 | -- | -- | hangs, no result |

192 is now applied. The gain over 128 is **2.2 %**, which is small but is a single-op
min-of-15 measurement rather than a whole-model wall clock, so it is above this harness's
noise floor.

**256 and above remain unmeasured**: the process produced no output and had to be killed, the
same failure that ended the amendment-54 sweep. Running each configuration in its own process
did *not* avoid it, so the hang is specific to those chunk sizes at this shape rather than to
sweeping several in sequence. Anyone continuing should treat 256+ as suspect and be ready to
`tt-smi -glx_reset` -- and note that a hung run leaves a process holding `CHIP_IN_USE_0_PCIe`
which must be killed by PID before the device will initialise again.

Gates unchanged: 5 passed, PCC 99.9997 % / 99.9877 %.

## Amendment 56 (2026-08-04) — encoder norm rewritten; four measured negatives

Session goal moved to **encode < 3 s and decode < 3 s** at 768P/5s (from "3 s e2e").

### The decoder was already there — the old number was under-sampled

`_time_it` ran `iterations=2`. Raised to 8 for the throughput test. The decoder wave went
**0.652 s -> 0.150 s** with no code change: min-of-2 was simply never catching the warm
time. Amendment 41's "0.34-0.99 s/wave" drift was that.

**768P/5s: encode 5.0 s, decode 1.0 s.** Decode is done; encode is the whole problem.

### Per-level budget (this is what was missing)

Generalised `profile_downblock_minimax_h3.py` over all six levels. Per unit:

| level | shape | profiled | **wall clock, no profiler** |
|---|---|---|---|
| 0 | 128->128 T17 256x256 | 477 ms | **346 ms** |
| 1 | 128->256 T17 128x128 | 332 ms | **195 ms** |
| 2 | 256->256 T9 64x64 | 40 ms | 30 ms |
| 3 | 256->512 T5 32x32 | 21 ms | 17 ms |
| 4 | 512->512 T5 16x16 | 5 ms | 6 ms |
| 5 | 512->1024 T5 16x16 | 26 ms | 14 ms |
| | | | **609 ms** |

**Levels 0+1 are 89 % of the encoder.** Everything below is about them.

**Tracy inflates data-movement ops and must not be read as absolute.** Profiled Conv3d
(18.5 ms for b0_res) matches the blocking sweep's independent trace timer (18.25 ms) to
1.4 %, but profiled Tilize is 21.3 ms where the identical tilize -- same shape, dtype,
memory config, same conv3d producer -- measures **3.0 ms** standalone. Block totals confirm
it: 432 profiled vs 346 actual at level 0, 322 vs 195 at level 1. Use the profiler for
*ranking*, wall clock for magnitude.

### What shipped: the stats norm is 2 passes shorter and absorbs the SiLU

`normed * weight + bias` was three full passes over the activation. Folding
`gamma = rsqrt(var + eps) * weight` on the (T,1,1,C) stats tensor makes it two, and the
SiLU folds into that add via `activations=[UnaryWithParam(SILU)]` instead of running as a
separate ROW_MAJOR unary.

Level 0 **477 -> 432 ms** profiled, level 1 **332 -> 322 ms**; BinaryNg 147 -> 68 ms at
level 0 and 106 -> 28 ms at level 1, UnaryDeviceOperation 18.5 -> 0.02 ms. Gates: 7 passed,
encoder PCC **99.9264 %** -- identical to before, so this is free.

### Four negatives, each measured

**1. conv3d compute-kernel config is irrelevant.** `MiniMaxH3CausalConv3d` has always used
HiFi2 + `fp32_dest_acc_en=True`, chosen when the encoder was fp32; both are worth up to 2x
on the FPU. Swept six configurations over seven real shapes
(`sweep_conv3d_fidelity_minimax_h3.py`, trace-timed per op):

| config | sum us | vs shipping | worst pcc |
|---|---|---|---|
| HiFi2/fp32acc (shipping) | 71729 | 1.00x | 0.999990 |
| HiFi2/bf16acc | 71402 | 1.00x | 0.999683 |
| LoFi/bf16acc | 72256 | 0.99x | 0.999758 |
| LoFi/fp32acc | 71611 | 1.00x | 0.999896 |

Nothing moves. **conv3d here is data-movement bound, not FPU bound** -- which is also why
the blocking sweep (which controls the vol2col working set) got 1.70x while fidelity gets
nothing. The shipping config also has the best PCC, so it stays.

**2. `Conv3dConfig.output_layout` is dead.** Setting it to TILE changes nothing:
`Conv3dDeviceOperation::compute_output_specs` hardcodes `PageConfig(Layout::ROW_MAJOR)` and
never reads `args.output_layout`. The knob exists on the config and in the nanobind
bindings but the device op ignores it. Plumbing it through the model was written and backed
out. **This is the one kernel change that would pay** -- see below.

**3. The flat-tile residual is a wash.** A ROW_MAJOR add of two 285 MB tensors profiles at
23.5 ms against ~2.2 ms tiled, so carrying the resnet chain in TILE looks free. It is not:
it moves the cost into Tilize (level 0 Tilize 50 -> 114 ms) rather than removing it.
A/B at both levels: flat-tile on 439/327 ms, off **432/322 ms**. Kept as
`MINIMAX_H3_FLAT_TILE_RESIDUAL`, defaulted **False**.

**4. The fused distributed GroupNorm would not help, so lifting its `N == 1` restriction is
not worth doing.** Timed all three candidates ROW_MAJOR-in/ROW_MAJOR-out (the honest
comparison -- conv3d is on both sides of every norm), with the fused op given the same
pinned `determine_expected_group_norm_dram_grid_size` grid as the group_norm path:

| candidate | L0 (285 MB) | L1 (143 MB) |
|---|---|---|
| `ttnn.group_norm`, T as batch | 46.8 ms | 14.6 ms |
| **stats norm (shipping)** | **17.3 ms** | **7.9 ms** |
| fused GN, per frame x T | 27.1 ms | 11.6 ms |
| (tilize + untilize floor) | 4.6 ms | 2.0 ms |

The fused op's per-frame call moves 50 MB in 1.59 ms — **~30 GB/s**, against the stats
norm's ~112 GB/s. `N > 1` would batch *the same kernel*, so it inherits that bandwidth;
the restriction is not what is holding it back. The stats norm is the fastest of the three
and `ttnn.group_norm` is 2.7x worse, which independently re-confirms amendment 52.

Also ruled out: **DRAM pressure is not why in-model ops are slow** -- the standalone tilize
holds 3.0 ms with 570/855/1140/1425 MB of ballast live. And **`ttnn.experimental.slice_write`
is not a faster pad than concat**: 379 ms against 42 ms for the three-concat chain.

### Where the encoder time actually is, and the ceiling

Conv3d is ~182 ms of the 609 ms (30 %), and the blocking sweep plus the fidelity sweep both
say it is at its floor. The other **~427 ms is data movement**: ~13 norm sites, ~13 pad
chains (three full copies each -- reflect H, reflect W, causal T), and ~6 residual adds,
moving roughly 22 GB per unit at an effective **~45-110 GB/s**.

That is the whole remaining opportunity, and it is an op-efficiency problem rather than a
model-structure one. If that traffic ran near DRAM peak the encoder would be ~240 ms/unit
(**1.7 s at 768P/5s**). The single highest-value kernel change is **conv3d honouring
`output_layout` and accepting TILE input**, which would delete the tilize/untilize round
trip at all thirteen norm sites and let every residual add be tiled -- not the GroupNorm
change, which measurement rules out.

## Amendment 57 (2026-08-04) — the causal zero pad was a host write: encoder 1.70x

`causal_pad_t` built its zero block with `ttnn.zeros(..., device=mesh_device)` **on every
call**. That is a host-to-device *write*, not a device fill: 34 MB at block 0, thirteen
times per unit, on the critical path. The block is constant per (shape, dtype), so it is now
allocated once per conv and reused.

**Encoder wave 0.715 -> 0.443 s (1.70x). 768P/5s encode 5.0 -> 3.1 s.** PCC 99.9264 %,
unchanged. Decode is unaffected at 1.0 s. **Total 4.2 s.**

It was found by trying to capture the encoder into a trace, which died on

    TT_FATAL: Writes are not supported during trace capture

-- so the trace attempt paid for itself even though **trace itself measures 1.00x** on both
halves once the write is gone (encoder 0.4403 untraced / 0.4411 traced; decoder 0.1498 /
0.1496). The encoder is **device-bound, not dispatch-bound**; amendment 49's withdrawn
op-to-op-gap reading is doubly dead, and there is no reason to ship a trace.

This also explains the gap amendment 56 could not account for: level 0 measures 346 ms
against ~244 ms of summed standalone components, and the missing ~100 ms was these writes.

Note `test_vae_conv_minimax_h3.py::test_resnet_block` fails 8/8 — verified **pre-existing**
by running it at `HEAD~1` (before any of this session's work), where it fails identically.
Not a regression from the norm rewrite or the zero-pad cache. Unrelated, still open.

## Amendment 58 (2026-08-04) — the one kernel change worth making is in conv3d's pad, not GroupNorm

Amendment 56 ruled out the fused GroupNorm on measurement. The change that *would* pay is in
`ttnn.experimental.conv3d`, and it is smaller than it looks, because **the vol2col reader
already implements padding natively**. `reader_vol2col.cpp:371-400` computes signed
`t_in/h_in/w_in` against the padded window, detects out-of-bounds, and either zero-fills
(`is_padding_zeros`) or clamps via `clampIndex` (replicate) — all inside the gather it was
already doing, with `check_padding` a compile-time template arg.

H3 cannot use it for two reasons, both narrow:

* **No reflect mode.** H3's spatial pad is reflect; the op accepts only `zeros` and
  `replicate` (`conv3d_device_operation.cpp:110`). Reflect is a sibling of `clampIndex`:
  `idx < 0 -> -idx`, `idx > N-1 -> 2*(N-1) - idx`.
* **Padding is symmetric.** `padding` is `array<uint32_t,3>` applied both sides, and H3's
  temporal pad is causal — `kernel_t - 1` frames prepended, none appended. Needs
  before/after per axis in `compute_output_dims` and the window origin.

The payoff is the whole padding chain: every conv currently does **three full copies** of its
activation (reflect W, reflect H, causal T) plus edge slices. That is Concat 88 ms + Slice
7 ms of level 0's profile and 24 + 4 of level 1's — roughly **100 ms of the 609 ms encoder,
~17 %**, which would take encode from 3.1 s to about **2.6 s**. It would also drop three
full-size temporaries per conv.

Checked and **not** an option: there is no "tpad fuse" in the WAN VAE to borrow —
`vae_wan2_1.py` pads with `ttnn.pad`/concat like everything else, and no `tpad` symbol
exists anywhere in the tree.

## Amendment 59 (2026-08-04) — audio decode measured: 1.273 s against a 0.05 s target

First measurement against the new audio goal. Single device, 5 s clip:

| | measured | target |
|---|---|---|
| audio encode | 0.347 s | (last) |
| **audio decode** | **1.273 s** | **~0.05 s** |

Round-trip PSNR 29.89 dB, test passes. **25x off**, and untouched by any performance work so
far — the visual path's 32x came from data-parallelism over `(clip, tile)` work units, and a
single 5 s audio stream is one unit, so none of that applies.

One lead already visible in the log, though probably not the main cost:

    depthwise conv1d unavailable at T_pad=1041, C=512, K=7, stride=1; MAC fallback

`audio_ops.py:232` falls back to `_depthwise_tap_mac` (7 strided slices + 7 multiplies +
6 adds) when `ttnn.conv1d`'s HEIGHT_SHARDED slicer finds no valid configuration at the latent
rate. It fires twice and the tensors there are ~1 MB, so it is likely single-digit ms of the
1.273 s. **The BigVGAN upsampling stack has not been profiled at all** — that is where to
start, with `profile_downblock`-style per-stage device timing rather than wall clock.

## Amendment 60 (2026-08-04) — audio decode traced: 1.07x. It is device-bound, and the lever is T-parallelism

`vocoder_ltx.Vocoder` already carries a `@traced_function` device region and a
`forward_traced` entry point, and its own docstring says "the vocoder is ~70% host-bound".
H3's audio decoder was calling the untraced `forward_BCT`. Added `forward_BCT_traced` (the
channels-over-time sibling of `forward_traced`) and a `traced=` flag on
`MiniMaxH3AudioDecoder.forward`.

Traced output is **bit-identical** to untraced (PSNR inf). The speedup is **1.07x**:

| | s |
|---|---|
| audio decode, untraced | 1.284 |
| audio decode, traced | 1.203 |

So the "~70 % host-bound" claim does **not** hold at H3's shape. Split by stage:

| stage | s |
|---|---|
| `dec_in_proj` (including its host round trip) | **0.0039** |
| vocoder | **1.284** (1.200 traced) |

Two things settled. The mid-forward host bounce in `_project_latents_device`
(`to_torch` -> transpose/contiguous -> re-upload) is **not** a problem at 3.9 ms — it looked
like an obvious target and is not one. And the vocoder is **99.7 % of audio decode and
device-bound**, so neither tracing nor removing host work will move it.

**The untried lever is parallelism: the vocoder runs on one device.** It already supports
T-sharding -- `parallel_config.factor` is threaded through `_upload_BCT` and `_forward_device`
(T-alignment padding, partition, T-gather) -- and H3's audio decoder accepts a
`parallel_config`, but the shipping path constructs it without one. The visual halves got
their 32x from data-parallelism over `(clip, tile)` units, which a single 5 s audio stream
cannot use; T-sharding is the equivalent for this workload and is the only route to 0.05 s
that does not need a faster BigVGAN. **Start there.**

Also note the trace region: 60 MB is not enough (`get_trace_buffers_size() <=
trace_region_size` fires); 300 MB works.

**Hazard:** replaying the traced vocoder in a *second* timing loop, after the whole-forward
loop had already captured at that shape, wedged the device and needed `tt-smi -glx_reset`.
The test times the traced path once, via whole-forward, and does not re-enter it.

On reading `PSNR: inf` -- that is MSE exactly zero, i.e. bit-identical, which is the
*expected* result for replaying the same program on the same data, not a degenerate
comparison. It is a weak assertion alone though, since it would also read inf if `traced=True`
silently fell through, so the test additionally asserts a tracer was captured
(`_forward_device._tracers_keyed`) and that the output is not all-zero.

## Amendment 61 (2026-08-04) — one-pass variance: encode is under 3 s

The stats norm made four full passes over the activation. `E[x^2] - E[x]^2` makes three: it
never materialises `x - mean`, so `x*x`, `x*gamma`, `+beta` is the whole activation cost.

The class docstring warned this off -- "the group means here are not near zero, and that is
the cancellation `GroupNorm3D` uses Welford to avoid" -- and that reasoning is sound about
the *subtraction*, but the subtraction does not have to happen in bf16 on the activation. It
happens on the per-(frame, group) stats, 32 scalars per frame, cast to **fp32** first. Only
the sums stay bf16.

Measured cost: encoder PCC **99.9094 %** against 99.9303 % for the two-pass form -- 0.02 pp,
against a 0.99 gate. Gates: 7 passed.

**Encoder wave 0.443 -> 0.427 s. 768P/5s encode 2.99 s.** Under the 3 s target.
Decode unchanged at 1.0 s. **Visual total 4.0 s.**

Flag: `MINIMAX_H3_ONE_PASS_VARIANCE`, default True.

## Amendment 62 (2026-08-04) — audio decode cannot reach the mesh yet: a bare `to_torch` blocked it

Amendment 60 identified T-sharding as the only route to 0.05 s. Attempting it exposed why it
had never been tried: **`MiniMaxH3AudioDecoder` could not run on a multi-device mesh at all**,
sharded or not. `_project_latents_device` read its result back with a bare `ttnn.to_torch`,
which asserts `buffers.size() == 1` (`pytensor.cpp:299`), so every factor — including
`t_factor=1`, where the tensor is merely *replicated* across 32 devices — died there.

Fixed: the upload replicates and `dec_in_proj` is a k1 conv, so every device holds the same
result and one is read back via `ttnn.get_device_tensors(...)[0]`, the same shape the
vocoder's own `_device_to_host` uses. Guarded on `get_num_devices() > 1`, so the single-device
path is untouched — re-verified at 1.284 s untraced / 1.213 s traced, PSNR inf.

`test_audio_parallel_minimax_h3.py` sweeps `t_factor` 1/4/8 on the 4x8 mesh and gates each
against the single-device output (PSNR > 40 dB) so a speedup from dropped work fails rather
than reports. **It has not yet produced a number**: past the readback fix the run reached the
vocoder and was killed before reporting, with the log full of

    DRAM Auto slice could not find valid slice configuration ... height-slicing
    depthwise conv1d unavailable at T_pad=1041, C=512, K=7; MAC fallback

so the conv1d slicer is failing at the sharded shapes and falling back per-tap. **That is the
next thing to chase**, and it is the same class of failure `audio_ops.py:232` already
documents at the latent rate. Audio decode therefore stands at **1.284 s against 0.05 s**.

Note the `4x8` mesh gives at most 8-way T-sharding through `ParallelFactor`, which is one
axis. `AudioTCParallelConfig` (time *and* channel) is what LTX uses to reach both axes, and
`MiniMaxH3AudioDecoder` accepts only `ParallelFactor` — widening that is likely required to
get past 8x.

## Amendment 63 (2026-08-04) — T-parallel audio decode works at factor 4; factor 8 is silently wrong

With the readback fixed (amendment 62) the sweep runs. 5 s clip, 4x8 mesh:

| t_factor | mesh axis | s | vs 32-dev replicated | PSNR vs 1-device |
|---|---|---|---|---|
| 1 | 1 | 1.691 | 1.00x | inf |
| **4** | 0 | **0.988** | 1.71x | **54.2 dB** |
| 8 | 1 | 0.898 | 1.88x | **-6.3 dB** |

**`t_factor=8` is faster and wrong.** -6.3 dB is not a precision wobble, it is a different
signal; the correctness gate is the only reason this did not get reported as a 1.88x win.
Factor 4 on the 4-wide axis is correct at 54.2 dB. Do not ship factor 8 without finding the
bug -- likely the halo/T-gather at 8 shards of a 256-frame padded extent, since 207 frames
pad to 256 and 256/8 = 32 is exactly one tile per shard.

Against the true single-device baseline of 1.284 s, factor 4 is **1.30x -> 0.988 s**.

**The scaling is poor and the reason is known.** 4x the devices buys 1.3x because the
vocoder's conv1d falls back per-tap at the sharded shapes — the run logs

    DRAM Auto slice could not find valid slice configuration ... height-slicing
    depthwise conv1d unavailable at T_pad=1041, C=512, K=7; MAC fallback

repeatedly. `_depthwise_tap_mac` (`audio_ops.py:236`) is 7 strided slices + 7 multiplies +
6 adds per call in place of one conv. Fixing the slicer configuration at these shapes, or
giving the fallback a better form, is what stands between 0.988 s and anything near 0.05 s —
more parallelism on top of a per-tap fallback will not get there.

**Audio decode stands at 0.988 s against 0.05 s.** Encoder and decoder targets are met
(2.99 s and 1.0 s); this one is not.

## Amendment 64 (2026-08-04) — audio decode profiled: 1680 ops, 224 ms device against 1284 ms wall

New targets this session: **audio decode 0.05 s, visual decode 0.8 s**, then visual encode
< 1 s, audio encode last. Current: audio decode 0.988 s (T-parallel factor 4), visual decode
1.0 s, visual encode 2.99 s.

First op-level profile of the audio decoder. Tracy's report generation asserts
(`Device data missing: Op 1443840`, the same failure mode as amendment 53), but
`generated/profiler/.logs/cpp_device_perf_report.csv` is written regardless and can be
aggregated directly — **do that instead of re-running when the report step fails.**

Single device, 5 s clip, warm half of two iterations:

| op | calls | ms | share |
|---|---|---|---|
| Conv3dDeviceOperation | 60 | 65.97 | 29.4 % |
| TernaryDeviceOperation | 56 | 46.93 | 20.9 % |
| BinaryNgDeviceOperation | **437** | 41.85 | 18.7 % |
| ConcatDeviceOperation | 175 | 21.34 | 9.5 % |
| ReshapeViewDeviceOperation | 110 | 14.31 | 6.4 % |
| SliceDeviceOperation | **298** | 12.10 | 5.4 % |
| UntilizeWithUnpadding | 135 | 11.49 | 5.1 % |
| **TOTAL** | **1680** | **224.2** | |

**The headline is the gap: 224 ms of device time inside a 1284 ms wall clock.** 83 % of audio
decode is not device work — ~630 us of host time per op, an order of magnitude above a normal
ttnn dispatch.

**And trace does not remove it** (1.284 -> 1.203 s, amendment 60), which it should: trace
exists to delete exactly this. Two candidate explanations are already eliminated —
`Tracer(prep_run=True)` runs the function once *before capture*, not per call, and the
un-traced stages are 2.8 ms total (amendment 60's split). So either the replay is stalling on
device in a way `DEVICE FW DURATION` does not count, or the captured region is not what is
being replayed. **Resolving this is worth ~5x on audio decode and is the next thing to do** --
more parallelism or fewer ops on top of a 1 s host overhead cannot reach 0.05 s.

Secondary, once that is settled: **910 of the 1680 ops are elementwise and data movement**
(437 BinaryNg + 298 Slice + 175 Concat). `_depthwise_tap_mac` emits `2K-1` ops per call for a
K-tap filter, and the AMP blocks use K = 3, 7, 11, so the conv1d slicer fallback is the
likely bulk of those counts. `TernaryDeviceOperation` at 20.9 % over 56 calls is the snake
activation (`x + (1/alpha) sin^2(alpha x)`) -- worth checking against
`existing-fast-paths.md` for a fused form before hand-rolling.

## Amendment 65 (2026-08-04) — cleanup pass 1: WIP profiling and sweep scripts removed

Test-file count 30 -> 19 against LTX's 13. Deleted, with every finding they produced already
recorded in amendments 52-64 so nothing is lost:

`measure_trace`, `probe_datamovement`, `probe_fused_gn`, `probe_tilize`,
`profile_audio_decoder`, `profile_downblock`, `profile_encoder`, `profile_vit_layer`,
`sweep_conv3d_fidelity`, `sweep_conv3d`, `time_downblock` (all `*_minimax_h3.py`).

`build_adaln_table.py` is kept -- `test_adaln_precompute_minimax_h3.py` imports it.
`wan2_2/bruteforce_conv3d_sweep.py` is the canonical blocking sweeper per the skill docs, so
the H3 copy was redundant; the comment in `conv_minimax_h3.py` now points there.

Encoder gate re-run after the deletions: **7 passed**, PCC 99.9094 % / 99.9132 %, unchanged.

**Still to do on cleanup:** the remaining 19 is above LTX's 13, and the dead
`MINIMAX_H3_FLAT_TILE_RESIDUAL` branch (default False, measured a wash) plus `_as_flat_tile`
/ `_as_row_major_5d` are only reachable through it -- deleting that branch is the next step
and needs one re-gate. `MiniMaxH3FrameGroupNorm` is likewise dead: the stats norm won on
measurement at every shape (amendment 56), so the `ttnn.group_norm` path and
`MINIMAX_H3_USE_STATS_GROUPNORM` can go with it.

## Amendment 66 (2026-08-04) — there is no 1 s of host overhead: audio decode is device-bound, and Tracy undercounts it 6x

Amendment 64 read 224 ms of device time inside a 1284 ms wall clock and called the difference
host overhead. **It is not.** Splitting `Vocoder.forward_BCT` with an explicit
`synchronize_device` between every stage (single device, 5 s clip, warm, min of 3):

| stage | ms |
|---|---|
| `dec_in_proj` stage (upload + k1 conv + readback) | 3.21 |
| `_upload_BCT` | 2.31 |
| `_forward_device` **dispatch return** (no sync) | 504.64 |
| `synchronize_device` **after** dispatch returned | **730.74** |
| `to_torch` readback | 11.93 |
| host crop | 0.03 |
| SUM | 1252.86 (full call 1292.96) |

The host finishes enqueueing all 1680 ops in 504 ms and the device still has **731 ms of
work outstanding**. So device busy time is between 731 ms and 1235 ms, not 224 ms:
**Tracy's `DEVICE FW DURATION` undercounts this model by ~5-6x.** Amendment 56 already
recorded Tracy *inflating* data-movement ops on the visual encoder; on the audio decoder it
under-reads. Neither direction is safe — use it for ranking only, never for magnitude.

This also explains trace's 1.07x (amendment 60) exactly, and retires it as a mystery: the
504 ms of host dispatch is *not on the critical path* because it overlaps device execution.
Trace deletes host dispatch, and the device still needs ~1.2 s. Nothing was wrong with the
trace.

Two other candidates die here too. The readback is **11.93 ms**, not seconds, despite the
final tensor being the worst imaginable readback shape — `(2, 165600, 1)` ROW_MAJOR fp32,
a 4-byte page. And the host crop is 0.03 ms.

### Where the 1.5 s of device time actually is

Same method one level down: `synchronize_device` before and after every leaf submodule of the
vocoder, one warm call. Serializing host and device inflates the wall clock to 1654 ms, but
each entry is honest device time for that module. **1493 ms of the 1654 ms is accounted.**

By role, summed over all seven stages:

| role | ms | share |
|---|---|---|
| `Activation1d.upsample` (UpSample1d, ratio 2 polyphase, K=12) | **588.7** | **39.4 %** |
| `Activation1d.downsample` (DownSample1d, K=12 stride 2) | **321.6** | **21.5 %** |
| `Activation1d.act` (`ttnn.snake_beta`, incl. its tilize) | **265.1** | **17.8 %** |
| `DilatedConv1d` (AMP `convs1`/`convs2`) | 286.7 | 19.2 % |
| `ups[i]` (ConvTranspose1dViaConv3d) | 21.8 | 1.5 % |
| `conv_pre` + `conv_post` | 9.6 | 0.6 % |

**`Activation1d` is 1175 ms — 78.7 % of audio decode device time.** The convolutions the
model is nominally made of are 19 %. The anti-aliasing wrapper around the activation is the
model. Per stage, `Activation1d` only (18 calls each, i.e. 3 blocks x (3 acts1 + 3 acts2)):

| stage | C | T in | up | snake | down | total |
|---|---|---|---|---|---|---|
| s0 | 512 | 1035 | 140.9 | 9.9 | 11.0 | **161.8** |
| s1 | 256 | 5175 | 43.7 | 21.7 | 25.0 | 90.4 |
| s2 | 128 | 10350 | 53.9 | 22.9 | 39.0 | 115.8 |
| s3 | 64 | 20700 | 58.3 | 24.5 | 41.0 | 123.8 |
| s4 | 32 | 41400 | 69.5 | 27.7 | 44.1 | 141.3 |
| s5 | 16 | 82800 | 71.1 | 51.7 | 51.8 | 174.6 |
| s6 | 8 | 165600 | 143.3 | 101.1 | 104.0 | **348.4** |
| act_post | 8 | 165600 | 8.0 | 5.6 | 5.8 | 19.4 |

Effective bandwidth is the tell. s6's upsample moves ~50 MB per call in 8.0 ms — **~6 GB/s**,
against the ~112 GB/s the visual encoder's stats norm achieves. These tensors are 4-6 MB;
nothing here is bandwidth-limited by size. Two distinct causes, one per end of the stack:

* **Deep stages (s4-s6) are page-limited.** `C = 32/16/8` in ROW_MAJOR fp32 is a **128/64/32
  byte page** over 41k-166k rows. s6 alone is 348 ms, 23 % of the whole decoder, at C=8.
* **s0 is op-count-limited.** `ttnn.conv1d` HEIGHT_SHARDED finds no valid slice configuration
  at `(T_pad=1041, C=512, K=7, stride=1)` and falls back to `_depthwise_tap_mac`
  **36 times per forward** (counted, not inferred — every polyphase phase of every one of the
  18 `Activation1d`s at that stage). At 2K-1 = 13 ops each plus slices that is **~720 of the
  1680 ops**, on a 4.2 MB tensor, which is why s0's upsample costs 141 ms where s1's costs 44.

And a layout round-trip sits inside every one of the 126 `Activation1d` calls:
`UpSample1d` and `DownSample1d` both assert ROW_MAJOR, `SnakeBeta.forward` tilizes its input
and returns TILE, and `Activation1d.forward:333` untilizes it back. The tilize is inside the
265 ms snake row; the untilize is most of the 160 ms this table does not account for.

### What this re-ranks

`_depthwise_tap_mac` was the standing suspect (amendments 62-64) and it is real but local —
one stage, ~110 ms of 1493 ms. The lever list for audio decode is now:

1. **Narrow-C ROW_MAJOR data movement in s4-s6** — 664 ms of `Activation1d` at C = 32/16/8.
   Biggest single item and not previously identified at all.
2. **The `UpSample1d` polyphase form itself** — 589 ms across all stages for what is one
   depthwise `conv_transpose1d`; `ttnn.conv_transpose2d` exists and is unused here.
3. **The conv1d slicer at `(1041, 512, 7)`** — a config fix worth ~110 ms, and it deletes
   720 of 1680 ops, which is most of amendment 64's elementwise count.
4. **The tilize/untilize around `snake_beta`**, 126 round trips per forward.

Parallelism stays the largest lever on paper (this is 1.5 s of device time on one device of
32), but amendment 63's 1.30x at `t_factor=4` is explained by the same two causes: sharding T
makes the deep stages' rows *fewer* while leaving the page 32 bytes wide, and it makes the
s0 conv1d slicer fail at more shapes, not fewer.

## Amendment 67 (2026-08-04) — why T-parallel audio only buys 1.30x: the shallow stages have no rows to shard

Same sync-isolated per-module method as amendment 66, at the shipping `t_factor=4` axis=0 on the
4x8 mesh. Wall 1001.9 ms (amendment 63's 0.988 s, reproduced), 962 ms accounted.

| role | 1 device | t_factor=4 | scaling |
|---|---|---|---|
| `act.upsample` | 588.7 | **468.7** | **1.26x** |
| `act.downsample` | 321.6 | 183.1 | 1.76x |
| `act.snake` | 265.1 | 114.9 | 2.31x |
| `DilatedConv1d` | 286.7 | 148.8 | 1.93x |
| `ups[i]` | 21.8 | **38.5** | **0.57x — worse** |
| total accounted | 1493 | 962 | 1.55x |

Cost tracks **row count** (amendment 66), so 4x devices should buy close to 4x. It buys 1.55x
because sharding divides rows the deep stages have plenty of and the shallow stages do not:

| stage | up, 1 dev | up, factor 4 | scaling |
|---|---|---|---|
| s6 (T=165600) | 143.3 | 59.6 | 2.40x |
| s5 (T=82800) | 71.1 | 50.6 | 1.41x |
| s4 (T=41400) | 69.5 | 46.7 | 1.49x |
| s3 (T=20700) | 58.3 | 43.9 | 1.33x |
| s2 (T=10350) | 53.9 | 46.7 | 1.15x |
| s1 (T=5175) | 43.7 | 42.8 | **1.02x** |
| s0 (T=1035) | 140.9 | **178.4** | **0.79x — worse** |

s1 gets nothing: at T=5175 the per-device extent is 1294 rows and fixed per-op cost already
dominates. s0 gets *worse*, and is now the single largest line in the shipping profile at
**178 ms of 1002 ms (17.8 %)** — the MAC fallback still fires 36 times, now at
`(T_pad=326, C=512, K=7)`, on rows so short that the added halo exchange outweighs the
smaller shard.

**So T-parallelism is close to exhausted, not under-exploited.** More factor buys the deep
stages only, and the deep stages are exactly where the page-width problem lives. Widening to
two axes (`AudioTCParallelConfig`) inherits this ceiling; it is not the 20x.

## Amendment 68 (2026-08-04) — the MAC "fallback" is the *fast* path: ttnn.conv1d is 2-3x slower at C=512

The standing plan (amendments 62-64, and the premise of amendment 67's s0 line) was that
`_depthwise_tap_mac` is a slow correctness path and the win is getting `ttnn.conv1d` to accept
these shapes. **Measured, and it is backwards.** Single device, fp32, HiFi4/fp32-acc, min of 3,
against the MAC chain on the identical input:

| shape | MAC | conv1d best | verdict |
|---|---|---|---|
| `(B=2, T_pad=1041, C=512, K=7, s=1)` — s0 up, 1 device | **1.46 ms** | all 8 configs FAIL | — |
| `(B=2, T_pad=326, C=512, K=7, s=1)` — s0 up, factor 4 | **1.30 ms** | all 8 configs FAIL | — |
| `(B=2, T_pad=2081, C=512, K=12, s=2)` — s0 down, **succeeds today** | **2.13 ms** | 4.56 ms | **conv1d 2.1x slower** |

Eight `Conv1dConfig` variants swept: HEIGHT/WIDTH/BLOCK/auto `shard_layout`,
`reshard_if_not_optimal`, `act_block_h_override=32`, `act_block_w_div=2`, `full_inner_dim`.
At K=7 every one fails — HEIGHT/auto in `op_slicing.cpp:266`, WIDTH in
`conv2d_op_width_sharded_program_factory`, BLOCK in `program.cpp:330`. At the K=12 control the
four that run land at 4.56-7.27 ms against the MAC chain's 2.13 ms, and are *less* accurate
(rel_max 1.73e-03 vs the MAC reference).

The comment at `audio_ops.py:225` calling the fallback "slower, but this is a correctness path"
is wrong at H3's shapes. **Do not spend more time engaging conv1d at stage 0**, and note that
the K=12 downsample at s0 would be ~2x faster on the MAC path than on the conv1d path it
currently takes.

Why the MAC chain wins: at `(2, 1035, 512)` it runs 12 slices + 12 multiplies + 10 adds at
~130 us each, i.e. **~65 GB/s** — near the bandwidth the visual encoder's best op achieves.
It is not inefficient per pass; it just makes **34 passes over the activation where the ideal
is 2**. That is the real defect, and the fix is a fused multi-tap FIR (one read, one write),
not a differently-configured conv1d.

### What this leaves, ranked, for audio decode (0.988 s against 0.05 s)

1. **A fused depthwise FIR** — one pass instead of 2K-1. `Activation1d` is 79 % of device
   time and every millisecond of it is `depthwise_tap_filter`, up or down. Nothing in
   `existing-fast-paths.md` covers depthwise-1D-over-rows at wide C; `ttnn.conv1d` is measured
   worse. This is genuine kernel work and it is the whole ballgame.
2. **T-folding the deep stages** — s4-s6 are 664 ms at `C = 32/16/8`, a 128/64/32-byte
   ROW_MAJOR page. `(B, T, C) -> (B, T/S, S·C)` is a free row-major reshape; per-channel ops
   (snake, tap multiplies) need only α/β tiled S times. Cuts rows by S where the row count is
   the cost.
3. **Trace** — see below. Not available until (1) and (2) land.
4. T-parallelism beyond factor 4 — capped at ~1.55x by amendment 67, and factor 8 is still wrong.

### On tracing

Trace's ceiling here is set by the ratio of host dispatch to device time, and amendment 66
measured both: **504 ms of host dispatch inside ~1235 ms of device time.** The host is already
the faster party, so trace can only recover the tail — which is precisely the 1.07x amendment
60 measured. Engaging `ttnn.conv1d` would cut the op count (720 of 1680 ops are the MAC
chains) and so cut host dispatch, but amendment 68 measures it as *increasing* device time, so
it moves the wrong number. Trace becomes the binding lever the moment device time drops under
~500 ms — i.e. after (1) — and it will still be there. That is exactly why the skill puts it
last.

## Amendment 69 (2026-08-04) — T-folding is only worth the tile-padding waste, not the row count

Amendment 66 showed cost tracks **rows**, not bytes (s6's conv1d at 165606 rows costs 2.16x
s5's at 82806 rows for the identical 5.3 MB). `(B, T, C) -> (B, T/S, S·C)` is the same
ROW_MAJOR buffer, so folding T into C looked like a free S x. Measured on the snake, the one
op in `Activation1d` that is pure per-channel elementwise and therefore folds with no boundary
work (α/β simply tile S times — `rel_max 0.00e+00` at every S, bit-exact):

| shape | S=1 | S=4 | S=16 | S=64 |
|---|---|---|---|---|
| (2, 331200, 8) | — (S·C < 32) | 4.095 ms | 3.376 ms | 3.340 ms |
| (2, 165600, 16) | — (S·C < 32) | 2.580 ms | 2.329 ms | — |
| (2, 82800, 32) | 1.846 ms | 1.862 ms | 1.801 ms | — |

Timed end to end including both reshapes and the tilize/untilize, i.e. what the model would
actually pay. **The gain saturates at S·C = 32 and there is none at all once C is already 32.**
So the win is not the row count — it is that `C = 8` in TILE layout wastes 3/4 of every tile,
and folding to 32 real channels recovers exactly that. Beyond 32 channels, nothing.

In-model that is s6.snake 104 -> ~60 ms and s5.snake 54 -> ~42 ms, **~57 ms of 1493 (3.8 %)**,
for a change to `SnakeBeta` that LTX's vocoder shares. Not worth it, and not pursued.

The row-bound ops are the ROW_MAJOR ones — `conv1d`, `concat`, `slice` inside the filters —
and those cannot fold without solving K-tap adjacency across fold boundaries. Folding does not
rescue them cheaply.

### Audio decode: the catalogue is exhausted for this bound class

0.988 s against 0.05 s, a **20x gap**, and every lever in `optimization-levers.md` has now
been matched against the measured bound class:

| lever | status |
|---|---|
| 1 parallelism | T-shard factor 4 shipped. Ceiling ~1.55x (amendment 67), factor 8 wrong (63) |
| 2 kernel research | `ttnn.conv1d` measured **2-3x slower** than the code it would replace (68). Nothing in `existing-fast-paths.md` covers depthwise-1D-over-rows |
| 3 layout round-trips | The tilize/untilize around `snake_beta` is real; folding recovers 3.8 % (this amendment) |
| 4 math fidelity | Data-movement bound at ~65 GB/s per pass (68); fidelity cannot move it, same finding as conv3d in amendment 56 |
| 5 fusion | **This is the one that is left, and it is a kernel:** a fused multi-tap FIR, one read + one write instead of 2K-1 passes |
| 6 blocking sweeps | Nothing to sweep — the hot ops are `slice`/`multiply`/`add`/`concat` |
| 7 trace | Capped at 1.07x until device time falls under the 504 ms of host dispatch (66) |

The honest statement is that **audio decode cannot approach 0.05 s by configuration.**
`Activation1d` is 79 % of device time; its two filters make 2K-1 passes over the activation
where one is possible, and each pass is already near bandwidth. A fused depthwise FIR
(`sum_k tap_k · x[t + k·d]`, one op, arbitrary C, ROW_MAJOR in/out) would take the filters from
34 passes to 2 and is the only change with the magnitude the target needs — worth an estimated
5-10x on `Activation1d`, i.e. audio decode to roughly 0.15-0.25 s, with trace then available on
top. That is `tt-dit-kernel-research` work, and it is the recommended next task.

## Amendment 70 (2026-08-04) — cleanup pass 2: both dead code groups deleted from the encoder

`encoder_minimax_h3.py`: **50 insertions, 218 deletions.** Both groups amendment 65 identified,
each verified to have zero references outside that file before deletion.

**Group 1 — the flat-tile residual.** `MINIMAX_H3_FLAT_TILE_RESIDUAL` (default False, measured a
wash in amendment 56) plus `_as_flat_tile` / `_as_row_major_5d`, which were reachable only
through it. With the flag off the resnet always returned a ROW_MAJOR 5D tensor, so both
`_as_row_major_5d` call sites (the `conv_shortcut` input and the down-block's downsampler input)
were no-ops. Also removed the norm's `keep_flat_tile` parameter and its 4D-input branch, which
existed only to accept the flat-tile form.

**Group 2 — the `ttnn.group_norm` path.** `MiniMaxH3FrameGroupNorm`, the
`MINIMAX_H3_USE_STATS_GROUPNORM` switch, the `make_frame_group_norm` factory (now a direct
constructor call at all three sites) and `MINIMAX_H3_GN_OUT_BLOCKS`, which only that class read.
The `GroupNorm3D` import goes with it. `_gn_hw_sharded` is **kept** — the H/W-sharded path still
calls it — with its type hint corrected to `MiniMaxH3DistributedFrameGroupNorm`, and
`_norm_silu`'s `isinstance(norm, MiniMaxH3DistributedFrameGroupNorm) and not _hw_sharded(...)`
collapses to `not _hw_sharded(...)` since that is now the only norm class.

Docstrings that described the deleted code were rewritten rather than dropped: the module
docstring now records the three measured losers (`ttnn.group_norm` 2.7x, fused distributed GN
1.6x, flat-tile a wash) with amendment pointers, so the negatives survive the code.

Gates, both re-run after the deletions:

| gate | result |
|---|---|
| `test_vae_encoder_minimax_h3.py` | **7 passed**, PCC 99.9094 % / 99.9132 % — identical to amendment 65 |
| `test_vae_hw_parallel_minimax_h3.py -k halo` | **6 passed** — the sharded path `_gn_hw_sharded` serves |

**Test-file count is unchanged at 19 against LTX's 13.** The dead code was inside a model file,
not a test file, so this pass does not move that number. What remains for it:
`test_audio_trace_minimax_h3.py` and `test_audio_parallel_minimax_h3.py` are both audio-decode
performance gates and merge cleanly; `test_vae_norm_primitives_minimax_h3.py` and
`test_vae_distributed_norm_minimax_h3.py` both gate the stats norm at a different altitude.
Neither merge was attempted here.

**`test_vae_hw_parallel_minimax_h3.py::test_encoder_sharded_matches_unsharded` is very slow and
reads as a hang.** It goes silent for >15 minutes at ~130 % CPU immediately after the conv3d
blocking warnings from encoder construction: that is the **torch reference encoder running on
host CPU** over a 17-frame 256x256 clip, not a device hang. Bound it by CPU rather than by log
growth — the log stays byte-for-byte static the whole time — or deselect it and run `-k halo`.

### Note on the measurement technique amendments 66-69 used

Tracy is unusable for the audio decoder (it under-reads 6x, amendment 66). What replaced it, and
what to rebuild if it is needed again: wrap the leaf submodules' `forward` with
`synchronize_device` on both sides and accumulate per label; for op-level attribution, also wrap
the `ttnn.*` entry points (`concat`, `slice`, `multiply`, `add`, `reshape`, `conv1d`,
`to_layout`, `snake_beta`) and tag each call with the enclosing module label. Serializing host
and device inflates the total ~10 %, but every entry is honest device time. The probe file was
deleted per amendment 65's precedent.

## Amendment 71 (2026-08-04) — NEXT TARGET: audio decode, a fused depthwise FIR then a fused Activation1d

Forward-looking spec, not a measurement. Everything quoted here is measured in amendments 66-69.

**Where audio decode stands:** 0.988 s at `t_factor=4` (1.284 s single device) against **0.05 s**,
a 20x gap. Device-bound: ~1235 ms device inside a 1284 ms wall, with 504 ms of host dispatch
hidden underneath it. `Activation1d` is **1175 ms of 1493 ms accounted device time (78.7 %)**,
and every millisecond of it is `layers/audio_ops.py:178::depthwise_tap_filter`, up or down.

### Why this is the only remaining lever

| lever | why it is closed |
|---|---|
| parallelism | factor 4 shipped; ceiling ~1.55x, s1 scales 1.02x, s0 gets worse (amendment 67) |
| kernel research (`ttnn.conv1d`) | 8 `Conv1dConfig` variants; all fail at K=7, and 2.1x **slower** where it runs (68) |
| layout / T-folding | saturates at S·C = 32, worth 3.8 % (69) |
| math fidelity | data-movement bound at ~65 GB/s per pass (68) |
| blocking sweeps | the hot ops are `slice`/`multiply`/`add`/`concat` — no tuning surface |
| trace | capped at 1.07x until device time falls under 504 ms (66) |

### Tier 1 — a fused depthwise FIR. Target: audio decode ~0.55-0.65 s

Replace `depthwise_tap_filter`'s two bodies (`_depthwise_tap_conv1d`, `_depthwise_tap_mac`) with
one op computing, per channel independently,

    y[b, t, c] = sum_k  tap[k] * x[b, t*stride + k, c]

ROW_MAJOR `(B, T_pad, C)` in, ROW_MAJOR `(B, T_out, C)` out, `T_out = (T_pad - K)/stride + 1`,
**fp32** (`vocoder_ltx`'s docstring records that bf16 accumulation measurably degrades spectral
metrics through its 108-conv chain, and H3's is longer). Taps are a compile-time-sized host
vector, not a device tensor — K is 7 or 12 and there are only three distinct tap vectors.

**Three call sites**, all in `layers/audio_resample.py`: `LowPassFilter1d.forward:131` (the
downsampler, K=12 stride 2) and `UpSample1d.forward:227,230` (the two polyphase phases, K=7
stride 1). **381 calls per forward** — 18 `Activation1d` per stage x 3 filters x 7 stages, plus 3
for `act_post`.

The shapes it must serve, and what it must beat (measured per call):

| stage | C | FIR input | K, stride | current | ideal 2-pass @65 GB/s |
|---|---|---|---|---|---|
| s0 | 512 | (2, 1041, 512) | 7, 1 | 1.46 ms (MAC, 34 passes) | **0.13 ms** |
| s0 | 512 | (2, 2081, 512) | 12, 2 | 2.13 ms (MAC) / 4.56 (conv1d) | 0.26 ms |
| s5 | 16 | (2, 82806, 16) | 7, 1 | 1.22 ms (conv1d) | 0.16 ms |
| s6 | 8 | (2, 165606, 8) | 7, 1 | 2.63 ms (conv1d) | **0.16 ms** |
| s6 | 8 | (2, 331211, 8) | 12, 2 | 3.59 ms (conv1d) | 0.33 ms |

**The one design requirement that decides whether this works.** Cost today tracks **row count,
not bytes** — s6's conv1d costs 2.16x s5's for the identical 5.3 MB, purely because it has 2x the
rows at half the width (66). At `C = 8` a row is a **32-byte** DRAM stick, and that is what
strands the deep stages at ~6 GB/s. But in `(B, T, C)` row-major, **consecutive T rows are
contiguous in memory**, so the window feeding `R` output rows is one contiguous
`(R*stride + K - 1) * C * 4` byte block. A kernel that reads *blocks of rows* coalesces
regardless of how narrow `C` is. `ttnn.conv1d`'s HEIGHT_SHARDED halo path does not exploit this,
which is the whole reason it loses to 34 unfused passes. **Read many rows per transaction, hold
the K-1 row overlap in L1, accumulate all K taps in one pass.**

Expected: the 911 ms of filters (up 589 + down 322) to ~230-350 ms, single-device total 1493 ->
**~810-930 ms**, `t_factor=4` to **~0.55-0.65 s**. Trace then has room (device time approaching
the 504 ms dispatch) and T-parallelism is worth re-measuring, since its ceiling came partly from
per-op fixed cost that this removes.

**Two cheap extras to fold in while there.** (a) Give the op an *interleaved two-phase* output
mode: `UpSample1d`'s polyphase pair plus the `ttnn.concat` that interleaves them becomes one
call, deleting 36 concats per forward (68 ms at s6 alone). (b) The replicate/zero T-pad is a
`concat` costing a full copy of the activation (2.6 ms at s6); let the FIR take
`pad_before`/`pad_after` and a `padding_mode` and do it inside the gather — the same argument
amendment 58 makes for conv3d's vol2col reader, which already does exactly this.

### Tier 2 — fuse the whole Activation1d. This is the one with 0.05 s magnitude

Tier 1 leaves `Activation1d` making ~6 full materializations per call, three of them of the **2x
upsampled** tensor. `up2 -> snake -> down2` never needs that tensor in DRAM: for each output row
block, upsample locally in L1, apply `x + (1/beta) sin^2(alpha x)` per channel, lowpass back down,
write once. Ideal traffic is **one read and one write of the T-row tensor**, against roughly 40
passes over T-or-2T rows today.

That is a ~10-20x on 79 % of audio decode and the only path that makes **0.05 s** arguable. It is
a real kernel (a fused resample-activate-resample band), so Tier 1 first: Tier 1 is a
self-contained op with 381 call sites and its own unit test, and it is the input Tier 2 needs
anyway.

### Acceptance gates

Correctness first, per the skill. `test_audio_vae_minimax_h3.py` round-trip PSNR **>= 29.89 dB**
(the current value), and `test_audio_parallel_minimax_h3.py` already gates every sharded factor
against the single-device output at **PSNR > 40 dB**, so a speedup from dropped work fails rather
than reports. Bit-exactness is not required; `depthwise_tap_filter` is currently bit-exact
(rel_max 0.0) against torch, so a new op should be held to `rel_max <= 1e-6` in fp32 rather than
to conv1d's 1.73e-03.

**Measure with the sync-isolated method, not Tracy** — Tracy under-reads this model 6x
(amendment 66 records the technique). Re-run the per-role table so the new `Activation1d` share
is directly comparable to the 78.7 % baseline.

---

## Amendment 72 (2026-08-04) — SCOPE CHANGE: T2VA end-to-end. M5-M9 reactivated, and the base commit moves

The VAE-only scope declared at the top of this file (2026-08-03) is **superseded by user
directive**: the goal is now **t2va running end to end on the 4x8** — prompt in, video plus
synchronized audio out — with every component's PCC gate green at 768P/5s, the e2e artifact
check passing the rubric (seams, flicker, **A/V sync**), latency recorded as-is, and this file
amended as it goes. Explicitly **not** a perf campaign: no tuning, no sweeps, no trace work.

Plan: `~/.claude/plans/serialized-orbiting-rain.md`. Re-read it and this file every iteration.

The milestone map at the top still applies. The parked milestones are now live, in this order:
**M7 (Qwen3-VL text encoder) -> M6 (full DiT forward at the production packed length) ->
M9 (e2e t2va + quality gates)**. M10-M12 stay out of scope.

### The base commit moves, because the two halves of the work were on different lines

Working tree was `kevinmi/minimax-h3-vae` @ `61a456b97a4`, which had **deleted the DiT**
(commit `850af5f469d`, "superseded by cglagovich/minimax-h3"). The DiT lives at
`0c4ce3596b5` — a *dangling* commit, reachable from no ref, whose parent is
`gh/cglagovich-minimax-h3` @ `b85be88d6d3`.

`0c4ce3596b5` is a strict superset of the old HEAD. `git diff 0c4ce3596b5 61a456b97a4` is 19
files, **all Python**, and going 0c4ce -> old-HEAD only *deletes*: the five DiT modules,
`MiniMaxH3_perf_log.md`, five DiT test files, `common.py`, `project_block_perf.py`, plus small
deltas in `layers/{feedforward,linear,normalization}.py` and `utils/sweep_mm_block_sizes.py`.
The VAE, audio-VAE and `pipelines/minimax_h3/` trees are **byte-identical** between the two.

```
git switch -c kevinmi/minimax-h3-t2va 0c4ce3596b5
```

Verified after the switch: DiT's five modules present, 24 test files, VAE/audio-VAE/pipelines
diff against `61a456b97a4` **empty**, all four untracked dirs (`internal-prodia`, `prodia`,
`recover-logs`, `sweep_results_minimax_h3_encoder`) survived, **three stashes intact**.
Submodule pins (`umd ef7aa4b9dace`, `tt-cluster-descriptors 7b2176e2`, `tracy 11710051`) are
**identical** at both commits, and the diff is Python-only, so `build_Release` (compiled
2026-08-03 21:46) stays valid — **no rebuild**. Pre-switch state in
`~/h3_t2va_pre_switch.txt`.

Method note worth keeping: a superset commit that no ref points at is one `git gc` from gone.
It is now on a branch.

### `transformers` upgraded 4.53.0 -> 5.12.1, and it made three dormant gates live

`python_env` had `transformers==4.53.0`, which has **no `qwen3_vl`** — so
`Qwen3VLForConditionalGeneration` could not be imported and the text encoder (M7) had nothing
to gate against. `tt_metal/python_env/requirements-dev.txt` pins `transformers == 5.12.1`; the
env was simply stale. Upgraded per `models/MiniMaxH3.md`'s uv note (this venv has no `pip` of
its own, and `pip freeze` returns *nothing* here — the before-state was captured with
`importlib.metadata` instead, 378 packages, `~/h3_t2va_env_before.txt`).

```
uv pip install --python /data/kevinmi/tt-metal/python_env/bin/python "transformers==5.12.1"
```

A `--dry-run` first showed the real risk and it was **not** the one `MiniMaxH3.md` warns about:
`torch`, `numpy` and `Pillow` are untouched, but `huggingface-hub` jumps **0.36.2 -> 1.26.0**,
a major version, under a `diffusers` that is a pinned dev build (`0.40.0.dev0` @
`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`). Rollback scripted *before* installing:
`~/h3_t2va_rollback_env.sh`.

**Gate 0 — all four checks PASS:**

| # | Check | Result |
|---|---|---|
| 1 | `import ttnn` + `import torch`; `torch`/`numpy`/`Pillow` unmoved | OK — 2.11.0+cpu / 2.2.6 / 12.2.0, all unchanged |
| 2 | pinned `diffusers` still resolves the four `MiniMaxH3*` classes | OK — the hf-hub 1.x risk did not materialize |
| 3 | `from transformers import Qwen3VLForConditionalGeneration` | OK, 5.12.1 |
| 4 | host-only H3 suite | **93 passed, 0 skipped**, 22.9 s |

Check 4 is the interesting one. The recorded baseline was "**84 passed, 3 skipped**", the three
skips being "diffusers branch not installed". They now **run**, so the upgrade did not merely
avoid breaking the suite — it converted three reference comparisons from dormant to live. New
baseline: **93 passed, 0 skipped**. Command:

```
timeout 900 ./python_env/bin/python -m pytest \
  models/tt_dit/tests/models/minimax_h3/test_{packing,scheduler,conditioning,adaln_precompute,convert_minimax_h3_audio}*.py \
  -q --no-header
```

Net moved: `transformers` 4.53.0->5.12.1, `huggingface-hub` 0.36.2->1.26.0,
`tokenizers` 0.21.4->0.22.2, `click` 8.1.7->8.4.2, `hf-xet` 1.5.0->1.6.0,
plus `annotated-doc`, `shellingham`, `typer` added. Host-only, no device time.

### The working point, fixed for every gate

**1344x768** (16:9 — the widest 768P canvas `resolve_canvas_size` yields), **124 frames** @ 24 fps
-> **37 latent frames**, **207 audio latents**, `num_inference_steps=50` -> **49 forwards**.
Mesh 4x8, TP=4 axis 0, SP=8 axis 1, ring, 2 links. This is exactly the `5s_768p` column of
`MiniMaxH3_perf_log.md` — the only shape whose AGMM and ring-SDPA block sizes were actually
swept, so a failure reads as a bug rather than as an untuned shape.

---

## Amendment 73 (2026-08-04) — W1: host rope tables, bit-exact. And the plan's central claim checked before building on it

### The production row counts confirm the tile-alignment problem is real

Before writing anything, checked the claim the whole T2VA plan pivots on. At the working point
(1344x768, 124 frames, 512-token prompt), `build_packed_sequence` gives:

| stream | rows | mod 32 |
|---|---|---|
| text | 512 | 0 |
| audio | 207 latents x 2 channels = **414** | **30** |
| video | 37 latent frames x 1008 rows/frame = **37296** | **16** |
| total | 38222 -> padded to 38400 (sp x TILE = 256) | |

`MiniMaxH3Transformer3DModel.forward` asserts every modality's row count is a multiple of
`TILE_SIZE`, because it concatenates the three streams in `TILE_LAYOUT` before `mesh_partition`
fractures them. **Two of the three fail at the shape that ships.** Every existing test picks
tile-aligned synthetic lengths (512/256/1280 and 512/256/20736); the block-level perf test
never hits the assert because it calls a *block*, not the model. So the full forward has never
run at the real packed length, and could not have.

Also settled from `transformer/config.json`: `rope_freq_dim=16`, `rope_theta=10000.0`,
`patch_size=[1,2,2]`. Note the perf log's `5s_768p` sequence length (38015 -> 38144) is 207 rows
short of the real one (38222 -> 38400): it counted audio rows as `num_audio_latents` rather than
`num_audio_latents * 2`. Immaterial to its measurements (0.5%), but the *real* per-device count
at SP=8 is **4800 rows, not 4768**.

### `build_rope_tables`

`pipelines/minimax_h3/packing.py` built `position_ids` but never the rotary tables, so only a
test could produce them — `test_transformer_minimax_h3.py` borrows the reference model's own
`rope` submodule, which a pipeline cannot do. Added `build_rope_tables(position_ids, *,
rope_freq_dim, rope_theta)`, a mirror of `MiniMaxH3RotaryPosEmbed.forward` op for op and in its
order. The fp64 position grid is cast to fp32 *first*, as the reference does; casting later
moves the last ulp of every angle.

**Gate: bit-exact (`torch.equal`), and it passes at all three shapes** — bringup, canonical and
the new `t2va_768p_5s` case (empty `keyframe_anchors`, the no-condition-rows path the t2va
pipeline actually takes, which no existing case covered). Four new tests, seven cases:

- `test_rope_tables_match_reference` — `torch.equal` against `MiniMaxH3RotaryPosEmbed`
- `test_rope_tables_shape_and_dtype` — width `2*3*16 = 96` against head_dim 128 (partial
  rotary), fp32, finite, and the two duplicated halves equal. Stands in when diffusers is absent
- `test_rope_tables_distinguish_the_three_axes` — a video row's t/h/w frequency blocks must
  differ. If they collapsed, spatial rotary would be inert and the symptom would read as weak
  spatial coherence, not as a bug

Host-only suite now **100 passed, 0 skipped** (was 93 after gate 0, 84+3-skipped before).
`timeout 900 ./python_env/bin/python -m pytest models/tt_dit/tests/models/minimax_h3/test_{packing,scheduler,conditioning,adaln_precompute,convert_minimax_h3_audio}*.py -q`
— 25.5 s, no device. SHA `0c4ce3596b5` + working tree.

---

## Amendment 74 (2026-08-04) — W2: the text conditioner is green, and the shared Qwen3-VL encoder had a real limitation

### PCC 99.9999 %, and the bar it was measured against was three orders of magnitude too loose

`test_text_encoder_minimax_h3.py::test_text_encoder_tap_matches_reference`, **PASSED**, 4x8 mesh,
TP=4 on axis 0, FSDP on axis 1, 174 s wall including weight load, device closed cleanly:

| prompt | tokens | PCC | CCC | RMSE/sigma |
|---|---|---|---|---|
| "A red fox trots across a snowy field at dawn..." | 19 | **99.9999 %** | 99.9990 % | 0.4 % |
| "Close-up of a jazz pianist's hands..." | 22 | **99.9999 %** | 99.9970 % | 0.8 % |
| "Waves break on black volcanic sand..." | 13 | **99.9999 %** | 99.9990 % | 0.4 % |

The plan set the bar at `pcc=0.99` from `testing-and-accuracy.md`'s "full-model forward" row.
Measured is 0.999999 — that bar would wave through a 10^3 regression, so it is **tightened to
`pcc=0.999`, `relative_rmse=0.05`**, still ~50x looser than measured. Deliberately not tighter:
every measurement here is at 13-22 tokens, and a 512-token prompt accumulates over a much longer
causal context than anything gated. `relative_rmse` is paired with PCC because the DiT's
`context_embedder` consumes the embedding as an absolute value.

Reference: HF `Qwen3VLForConditionalGeneration` at **full 64-layer** depth, `hidden_states[50]`,
dumped once on CPU by `tests/models/minimax_h3/dump_text_golden_minimax_h3.py` (32 s for three
prompts — the weights mmap, so the 63 GB load is 2.0 s, not the minutes projected).

### `Qwen3VlAttention` assumed `head_dim == hidden_size // num_heads`. H3's checkpoint does not

The shared encoder derived head_dim from `hidden_size // num_attention_heads` and raised if that
did not divide. For H3's conditioner that is **5120 / 64 = 80, while the real head_dim is 128**:
`q_proj` is `[8192, 5120]`, `k_proj`/`v_proj` `[1024, 5120]`, `o_proj` `[5120, 8192]`,
`q_norm`/`k_norm` `[128]`. 5120 % 64 == 0, so the guard would *not* have fired — it would have
built a 80-wide qkv projection and failed later on a shape mismatch, or worse.

Fixed by threading an optional `head_dim` through `Qwen3VlTextEncoder` ->
`Qwen3VlDecoderLayer` -> `Qwen3VlAttention`, **defaulting to the old derivation**, which is
correct for the 8B Ideogram-4 loads (4096 / 32 = 128) and leaves that path untouched. The rope
width follows the same value, since `_apply_rope` needs tables at head_dim.

Method note: this is the second time in this campaign that a *shared* helper encoded an
assumption true of its first caller. It was found by reading the safetensors header before
loading, which the skill's Scaffold step calls for and which cost about a minute.

### mRoPE: the collapse argument is now measured, not assumed

The plan claimed that because T2VA is text-only, all three mRoPE axes carry the same positions,
so the checkpoint's `mrope_interleaved: true` is indistinguishable from the chunked section split
`create_rope_tensors` implements. Checked two ways rather than believed:

- **Permutation invariance.** Scrambling `mrope_section` from `[24,20,20]` to `[20,24,20]` and
  `[20,20,24]` leaves cos and sin **bit-identical**. The split cannot matter.
- **Against HF's own tables**, captured by a forward hook on `Qwen3VLTextRotaryEmbedding`. First
  comparison showed maxdiff **1.947e-03** and read like a real mismatch. It is not: HF emits
  cos/sin in the hidden dtype, so the reference tables are **bf16** (verified bf16-representable),
  and 1.9e-3 is one bf16 ulp near 1.0. At matching dtype the residual is **2 entries in ~2500 at
  one bf16 ulp**, i.e. fp32 rounding order.

Method note worth keeping: *a discrepancy at exactly the resolution of a lower precision is a
dtype difference, not a maths difference.* Checking "is the reference representable in bf16?"
settled in one line what looked like a port bug.

`load_minimax_h3_text_state_dict` reads **552 tensors from 12 of 14 shards, 50.3 GB bf16** — 50
layers x 11 + `embed_tokens` + `norm`. Independently confirms amendment 4's "50 of 64 layers,
~50 GB, not 62-66 GB". The vision tower (`model.visual.*`, 27 blocks) and `lm_head` are never
read; T2VA has no keyframe, so no vision block exists to encode.

New file `encoders/qwen3vl/loader_minimax_h3.py`; new gates in
`tests/models/minimax_h3/test_text_encoder_minimax_h3.py` (3 host + 1 device).

---

## Amendment 75 (2026-08-04) — W3: ROW_MAJOR packed-sequence assembly. The unaligned path costs no quality

Per amendment 73, production t2va violates the transformer's per-modality tile-alignment
assertion. Fixed by assembling the packed sequence in `ROW_MAJOR` — row granularity is 1 there,
so an unaligned cut is legal — then converting once to `TILE_LAYOUT` after the tail pad, whose
`padded_len` is a multiple of `sp_factor * TILE` and therefore tile aligned even though none of
its three parts was.

**The TILE path is kept for the aligned case rather than replaced.** That is deliberate and
better than an env flag here: the new path activates only for shapes that previously *asserted*,
so this change cannot move a shape that already worked, and the cheap path stays the default for
everything that did.

Interior padding was considered and rejected on the mechanism, not on cost: ring attention's
`logical_n` masks only the **tail**, so a pad row placed between two modalities sits inside
`logical_n` and every real row would attend to it as a key and value. Choosing 141 frames instead
of 124 aligns video (42 x 1008 == 0 mod 32) but not audio (235 x 2 = 470 == 22), so no frame count
escapes it either.

**Gate (2-layer, random weights, 4x8 ring TP=4/SP=8, 385.9 s, device closed cleanly):**

| case | modality rows | padded | path | PCC video / audio |
|---|---|---|---|---|
| `small_s2048` | 512 / 256 / 1280, all == 0 mod 32 | 2048 | TILE (unchanged) | **99.9974 % / 99.9974 %** |
| `prod_residues_s706` | 500 / 62 / 144, == 20 / 30 / 16 mod 32 | 768 | ROW_MAJOR (new) | **99.9975 % / 99.9975 %** |

The new case reproduces production's residue classes at a size a CPU reference can carry: video
3 frames x (4 x 12) = 144 == 16 mod 32 mirrors 37 x 1008 = 37296 == 16; audio 62 == 30 mirrors
414 == 30. Before this change it asserted. The aligned case holds its previous number exactly, so
the fast path is untouched, and the ROW_MAJOR path is **not worse** — 99.9975 vs 99.9974 is
run-to-run noise, not an improvement.

`_modality_metadata` gained a `grid` parameter (default 8x8) because 8x8 = 64 rows/frame is a
multiple of TILE for *any* frame count, so the existing helper could not express an unaligned
video stream at all.

Command: `timeout 2400 ./python_env/bin/python -m pytest
models/tt_dit/tests/models/minimax_h3/test_transformer_minimax_h3.py -k "random_weights and
(prod_residues or small_s2048)" -s --timeout 2100`. SHA `0c4ce3596b5` + working tree.

Not yet measured: the layout round-trip's device cost at the production sequence (~100 MB/device,
four layout ops per step against a 0.88 s step). Full depth at the production length is running.

---

## Amendment 76 (2026-08-04) — RETRACTION of the shapes in amendments 74 and 75: gates must be at production shapes

**What amendments 74 and 75 claimed.** Amendment 75 gated the ROW_MAJOR assembly path with a case
called `prod_residues_s706` (text 500 / audio 62 / video 144 over a 4x12 grid) and argued it was
adequate because it "reproduces production's residue classes at a size a CPU reference can carry".
Amendment 74 gated the text conditioner on three prompts of **13, 19 and 22 tokens**.

**Why that was wrong.** Both are invented shapes. `testing-and-accuracy.md` § "Production configs
only" says to derive test shapes from the model's real schedule and that sweeping invented shapes is
worse than useless; the residue-class argument is exactly the kind of reasoning that *feels* like
production coverage and is not. Reproducing a shape's arithmetic residue does not reproduce its
size, its memory behaviour, or — for the conditioner — its context length. User directive
2026-08-04 made this explicit: gate on production shapes.

**The correct reading, measured.** Both gates re-run at the real working point, and in one case the
number moved materially:

| gate | invented shape | **production shape** |
|---|---|---|
| DiT 2-layer, real weights, video PCC | 99.9975 % @ 706 rows | **99.9979 % @ 38222 rows** |
| DiT 2-layer, real weights, audio PCC | 99.9975 % @ 706 rows | **99.9974 % @ 38222 rows** |
| text conditioner PCC | 99.9999 % @ 13-22 tokens | **99.9892 % @ 512 tokens** |
| text conditioner RMSE/sigma | 0.4-0.8 % @ 13-22 tokens | **1.5 % @ 512 tokens** |

The DiT held up. **The conditioner did not**: PCC fell an order of magnitude and RMSE/sigma tripled
going from a sentence to a 512-token prompt, because a 50-layer causal stack accumulates over its
context. Had the bar been tightened to 0.9999 on the short-prompt evidence — which the measurement
invited — the production gate would have failed. The bar is now set from the 512-token row
(`pcc=0.999`, `relative_rmse=0.05`, ~10x and ~3x margin).

**Method note, the valuable part:** *a gate's shape is part of the gate.* An invented shape that
matches production in one property (residue, dtype, op mix) is evidence about that property only.
For anything with accumulation — depth, context length, sequence length — the only shape that
measures the production error is the production shape.

Changes: `prod_residues_s706` **replaced** by `prod_768p_5s` (512 / 414 / 37296 over a 24x42 grid,
38222 -> 38400 padded) in `test_minimax_h3_transformer`; the golden text dump now carries the e2e
prompt (39 tokens) and a 512-token prompt instead of three sentences.

### Also settled: the rope-table bar was below the reference's own precision floor

`test_mrope_matches_reference_tables` passed at 13-22 tokens with `atol=1e-4` and **failed at 512
tokens**: 6/65536 entries differing by 0.00390625. That number is exactly `2^-8` — **one bfloat16
ulp** at magnitude ~1. Longer prompts simply put more entries on a bf16 rounding boundary. The
reference tables are bf16 (asserted bf16-representable in the test), so `atol=1e-4` was a bar *below
the floor of the thing being compared against* — an unfixable failing gate, exactly the trap
`testing-and-accuracy.md` names.

Reformulated to compare at the bf16 floor: worst case <= 2 ulps **and** at most 0.1 % of entries
differing at all, so a systematic shift hiding inside one ulp still fails. Measured at 512 tokens:
8/65536 cos and 20/65536 sin entries, worst 1.00 ulp. A wrong theta, head_dim or section width
moves O(1) of the entries by O(1) — ~250x this bar — so no detection power is lost.

---

## Amendment 77 (2026-08-04) — M6 answered: the full 50-layer DiT runs at the production packed length

`test_minimax_h3_transformer_real_weights[prod_768p_5s]`, **PASSED**, 380 s, device closed cleanly.
4x8 ring, TP=4 / SP=8, real checkpoint, all 638 keys consumed by a strict load.

- Packed sequence **38222 -> 38400 padded**, **4800 rows/device** at SP=8, on the ROW_MAJOR
  assembly path.
- Output geometry exactly right: video `(37296, 96)`, audio `(414, 32)`.
- Finite and non-degenerate: video std 1.7502 absmax 12.2500; audio std 0.8507 absmax 6.1562.
- Weight load + forward ~360 s wall.

This is the residency question the plan flagged and it is answered: **the 50-layer DiT fits and runs
at the real packed length**, which is 1.77x the activation footprint of the deepest case that had
ever run (21504 rows). No allocation failure, no hang, no reset needed.

Milestone map: **M5 and M6 are green at the production shape. M7 is green.** M9 (e2e) is next.

Note for the perf log, not acted on here: `MiniMaxH3_perf_log.md`'s `5s_768p` column is costed at
38144 padded rows / 4768 per device. The real figure is **38400 / 4800** — it counted audio rows as
`num_audio_latents` (207) rather than `num_audio_latents * 2` (414). A 0.7 % error, immaterial to its
conclusions, but the per-device row count in that file is not the one that ships.

---

## Amendment 78 (2026-08-04) — M9: t2va runs end to end. Prompt in, video plus synchronized audio out

`test_pipeline_minimax_h3.py::test_t2va_end_to_end`, **PASSED**, 4x8 ring TP=4/SP=8, 1344x768,
124 frames @ 24 fps, 50 scheduler steps -> 49 forwards, seed 0. Device closed cleanly both runs.

Artifacts: `~/h3_t2va_artifacts/{t2va.mp4, t2va_silent.mp4, t2va.wav}` (3.87 MB muxed).

### Gate evidence

| Tier | Gate | Result |
|---|---|---|
| 4 | `check_output_sanity` | shape (124, 768, 1344, 3), range [0,255], **std 46.05**, mean frame delta **9.88** |
| 4 | `check_audio_sanity` | 2ch, **5.175 s @ 32 kHz**, peak 0.076, rms 0.0122, **0.000 % clipped** |
| 4 | `check_av_sync` | video 5.167 s vs audio 5.175 s, **delta +0.0083 s** (0.2 of a frame) |
| 5 | spatial seam ratio | **vertical 0.952, horizontal 0.692** (1.0 = no seam) |
| 5 | temporal seam ratio at the 17-frame chunk period | **0.994** |
| 5 | audio log-spectrum | flatness **0.0039**, band range [-67.4, +3.0] dB — tonal, not noise |
| 5 | written mp4 re-decoded | 124 frames recovered from the container |
| 6 | VBench / CLIP | **not run** — `vbench` and `decord` are not installed; `RUN_VBENCH=0 RUN_CLIP=0`. See "Not done" |

### The artifact rubric, read against the real frames

Numbers cannot close this, so frames 0/17/34/62/123 were extracted and inspected, plus a 2x
nearest-neighbour crop deliberately spanning the tile boundaries at x=512 and x=768.

| Rubric artifact | Verdict |
|---|---|
| Seams at tile or patch boundaries | **None.** Not visible in the magnified crossing crop, and the measured ratios are ~1.0 |
| Temporal flicker between frames | **None.** Frame 17 (a chunk boundary) holds subject identity, lighting and pose continuity; temporal ratio 0.994 |
| Banding / posterization | **None.** The sky-to-snow gradient and the shadowed snow are smooth |
| Uniform blur or softness | **No.** Background bokeh with a sharp subject — a shallow-DoF telephoto look consistent with the prompt, not global softness. Individual fur strands resolve in the crop |
| Ghosting, melting, incoherent motion | **None.** Correct anatomy across all five frames, four legs with correct joints, coherent gait |
| Snow / speckle | **None** |
| Blank, flat or frozen | **No** (std 46.05, frame delta 9.88) |

The output is a red fox trotting across snow at dawn with warm low-sun rim light and long blue
shadows — i.e. the prompt, including the lighting clause.

### Latency, recorded as-is (not a target, no tuning done)

| stage | cold (first run) | **warm cache** |
|---|---|---|
| text encode | 164 s (50 GB read) | **0.0 s** (embedding cache hit) |
| denoise (49 forwards) | 473.1 s | **104.6 s** |
| video decode | 99.8 s | **21.5 s** |
| audio decode | 140.4 s | **7.5 s** |
| **total** | **713.4 s** | **133.8 s** |

The 49 forwards themselves are **~54 s, i.e. ~1.10 s/step** in both runs; the rest of the denoise
figure is weight load. That sits against the perf log's 0.88 s/step projection for the block stack
alone, so the refiner, input projections, `norm_out`, the two heads and the ROW_MAJOR layout
round-trip together cost ~0.22 s/step. **No attempt was made to reduce any of this.**

**The two runs produce bit-identical statistics** (std 46.05, frame delta 9.88, audio peak 0.076 to
every digit), so the weight-cache round trip is numerically exact rather than approximately so.

### Three bugs found by running it, each a one-line fix and none guessable

1. **`FABRIC_1D` vs `FABRIC_1D_RING`.** The pipeline's CCLManager runs ring collectives; a plain
   line fabric fails as `TT_FATAL fabric.cpp:174 forwarding_direction.has_value()`, which reads like
   a CCL bug and is a device_params mismatch. Now taken from `utils/test.py::ring_params_*` rather
   than hand-written.
2. **`ttnn.from_torch` has no `mesh_axes`.** That is tt_dit's own `utils/tensor.py::from_torch`
   wrapper. The DiT tests use the wrapper; copying their *call* without their *import* fails.
3. **`MiniMaxH3Scheduler.step` has no `return_dict`.** The tt_dit scheduler returns the next sample
   directly; only the diffusers one wraps it.

Also corrected: the video VAE's tile grid at 1344x768 is **4x6 = 24 tiles**, not the 28 that
`test_performance_vae_minimax_h3.py::WORK_UNITS` assumes. Together with the 38400-vs-38144 padded
row count (amendment 77), two of the perf log's work-unit figures are slightly off; neither changes
its conclusions, but neither is the number that ships.

### New

- `pipelines/minimax_h3/pipeline_minimax_h3.py` — `MiniMaxH3Pipeline`, structured as the reference
  `MiniMaxH3Blocks` sequence minus the keyframe block
- `tests/models/minimax_h3/test_pipeline_minimax_h3.py` — the e2e gate
- `tests/models/minimax_h3/common_av.py` — `check_audio_sanity`, `check_av_sync`,
  `check_spatial_seams`, `log_spectral_flatness`. Nothing in tree covered the soundtrack or the
  relationship between the streams

A/V sync is gated **structurally**, not perceptually: audio and video share one rotary clock, so
what can actually break is a duration or channel-order error. An envelope-vs-motion correlation is
reported (+14 frames, r=0.383) but never asserted — a guidance-distilled generator is not required
to tie its soundtrack to visible motion, so asserting on it would gate a property the model does
not promise.

---

## Amendment 79 (2026-08-04) — every component now loads through `utils/cache.py`, and it is a 5.3x on end-to-end wall time

User directive: use the full cache machinery, as the other pipelines do, for **all** components.
Done, and it turned out to be the single largest wall-clock lever in this campaign — without
touching a kernel.

`TT_DIT_CACHE_DIR=/data/kevinmi/tt_dit_cache` (the established root: it already held
`ltx-embeddings`, `Wan2.2-T2V-A14B-Diffusers`, `prodia-wan2.2-i2v`). It was **unset** in this shell,
which is why the first runs silently paid full price and wrote the prompt cache to
`~/.cache/tt-dit`; that has been consolidated under the real root.

| what | cache key | on disk |
|---|---|---|
| transformer | `minimax-h3/transformer/TP4_0_SP8_1_mesh4x8_bf16` | 63 GB |
| text encoder | `minimax-h3/text_encoder/TP4_0_mesh4x8_bf16_fsdp` | populated on a prompt-cache miss |
| video VAE decoder | `minimax-h3/vae_decoder_t7_h16_w16_<blocking>/TP1_0_mesh4x8_fp32` | 4.6 GB per distinct (T,H,W) |
| audio decoder | `minimax-h3/audio_decoder/TP1_0_mesh4x8_fp32` | 260 MB |
| prompt embeddings | `minimax-h3-embeddings/<md5>.device.pt` | 23 KB, skips the 50 GB text-encoder read entirely |

Three details that made this more than a one-line change:

- **The video VAE is not a single loadable `Module`.** It builds a decoder per distinct `(T, H, W)`
  and each one holds a shape-specialised conv3d weight layout, so there is no one state dict to
  cache. Added a `weight_loader` hook to `MiniMaxH3Vae` (defaulting to the plain strict load, so
  every existing test is unaffected) and let the pipeline supply a cache-aware loader keyed on the
  shape **plus `conv3d_blocking_hash`** — the same thing `vae_wan2_1.py` does, because
  `prepare_conv3d_weights` bakes `C_in_block` into the cached bytes.
- **The audio decoder needed `strict=False`** because `convert_minimax_h3_audio_state_dict` returns
  the encoder half too, and `cache.load_model` loads strictly. Rather than reach for private cache
  helpers, the state dict is filtered to the two prefixes the module owns (`dec_in_proj.`,
  `decoder.`), which keeps the load **strict** — a renamed key still fails — and puts it on the same
  public path as everything else.
- **The VAEs' cache key carries TP factor 1**, not 4: both are data-parallel over work units with
  replicated weights. Recording that as a `VAEParallelConfig` rather than a literal keeps the key
  honest if it ever changes.

**Measured effect, same test, same seed, cold vs warm:** 713.4 s -> **133.8 s end to end (5.3x)**.
Video decode 99.8 -> 21.5 s, audio decode 140.4 -> 7.5 s, transformer load 152 -> ~50 s, text encode
164 -> 0.0 s. Output statistics are **bit-identical** across the two runs, so the round trip is exact.

Method note: `TT_DIT_CACHE_DIR` being unset degrades *silently* — `cache.load_model` logs one line
and loads from safetensors. A 5x wall-clock difference with no error is exactly the kind of thing
that gets mistaken for "this model is just slow".

### Not done, and stated plainly

- **Tier 6 never ran.** `vbench` and `decord` are not installed in `python_env`, so the VBench
  dimensions and the CLIP prompt-alignment score are **unmeasured**. The gates are wired and default
  **on**; with the packages absent they report SKIPPED rather than passing, and this run was executed
  with `RUN_VBENCH=0 RUN_CLIP=0`. No VBench thresholds have been calibrated for H3 at 768P, and none
  should be copied from LTX's 1088p set. Installing the two packages and recording the first
  measurement is the next step for M9.
- **No perf work, by directive.** Latency is recorded as-is. The ~0.22 s/step above the perf log's
  block-stack projection has not been attributed, the ROW_MAJOR layout round-trip has not been
  measured in isolation, and no trace, blocking or fidelity change was attempted.
- **A single prompt and a single seed.** The e2e gate proves the pipeline, not the model's range.
- **`fl2va` and `ref2va` are untouched.** `build_packed_sequence` already supports keyframe anchors
  and `conditioning.encode_keyframes` is gated, but no keyframe path has been run on device, and the
  T>1 video encoder needed for `ref2va` does not exist.
- **Cache invalidation is by key, not by content.** `utils/cache.py` keys on model name, subfolder,
  parallel config, mesh shape, dtype and FSDP — **not** on the checkpoint's own hash. Editing weights
  in place under an unchanged path would serve a stale cache silently.

## Next step

Install `vbench` and `decord`, run `test_pipeline_minimax_h3.py` with the tier-6 gates on, record the
measured VBench dimensions and CLIP score here, then set `MINIMAX_H3_VBENCH_THRESHOLDS` below the
measured values with a stated margin. That closes M9. Do not copy LTX's thresholds.

---

## Amendment 80 (2026-08-04) — RETRACTION of amendment 78's "Not done": tier 6 now runs, and it passes

Amendment 78 recorded VBench and CLIP as **unmeasured** and named installing them as the next step.
Done. The gates are live in `test_pipeline_minimax_h3.py`, default on, and the whole test **passes**:
`1 passed in 369.88 s`.

### VBench cannot share `python_env`, and that is why it runs out-of-process

A dry-run before installing was what caught this. `uv pip install vbench decord` into `python_env`
would have:

- **numpy 2.2.6 -> 1.26.4** (major downgrade, under a compiled `ttnn`)
- **transformers 5.12.1 -> 4.33.2** (destroys the Qwen3-VL reference amendment 74 depends on, and
  gate 0 with it)
- plus huggingface-hub 1.26 -> 0.36, tokenizers 0.22 -> 0.13, timm 1.0.27 -> 1.0.12

So VBench lives in its own interpreter and is invoked as a subprocess on the written mp4. This is
not a workaround: **VBench evaluates a file.** It needs no mesh, no ttnn, and nothing from the
generating process, so splitting generation from evaluation is the correct structure independent of
the conflict. `tests/models/minimax_h3/vbench_runner.py` is the entry point; the test skips with the
exact venv-creation command if the interpreter is absent, and `python_env` was re-verified intact
afterwards (numpy 2.2.6, transformers 5.12.1, ttnn + both references importing).

**CLIP needed nothing new.** `open_clip` is already in `python_env` and this test already decodes
frames with ffmpeg, so the wan2.2/LTX `decord` dependency is not required at all and the gate runs
in-process.

Four environment problems, each silent-failure shaped, fixed in the eval venv:
`unzip` absent (RAFT ships a zip -> extracted with `zipfile`); `libGL.so.1` absent
(`opencv-python` -> `opencv-python-headless`); that pulled numpy 2 back in, breaking vbench
(repinned `numpy==1.26.4` with `opencv-python-headless<4.11`); and `pkg_resources` absent
(`setuptools<81`).

### Measured, and the bars set from these numbers

| dimension | **measured** | bar set | LTX's calibrated 1088p bar |
|---|---|---|---|
| subject_consistency | **0.9820** | 0.95 | 0.92 |
| background_consistency | **0.9831** | 0.95 | 0.93 |
| motion_smoothness | **0.9905** | 0.97 | 0.955 |
| dynamic_degree | **1.0000** | 1.0 | 1.0 |
| imaging_quality | **0.6896** | 0.64 | 0.645 |
| CLIP prompt alignment (mean of 8 frames) | **37.37** (min 36.52, max 38.44) | 33.0 | LTX 28.0 |

H3 at 768P **clears every one of LTX's thresholds**, which is exactly why copying them would have
gated nothing — the point amendment 78 made in advance and this confirms. CLIP 37.4 sits at wan2.2's
~37 baseline rather than LTX's ~31.3.

The bars are **single-sample calibration**: one prompt, one seed, so the margins are deliberately
generous (they catch a broken pipeline, not a quality regression). `dynamic_degree` stays at 1.0
because over one video it is effectively binary — the failure it detects is a frozen clip.

### A no-op gate that would have read green, caught before it ran

`utils/vbench.py::assert_vbench_quality` derives its dimension list from `thresholds.keys()`. The
first version of this test passed `thresholds={}` in its "report, don't gate" branch — which would
have evaluated **zero dimensions**, returned no scores, found no failures and logged success. That
is precisely the silently-no-opping quality gate `testing-and-accuracy.md` warns is worse than no
gate. Replaced with an explicit dimension list plus real bars, and the test now asserts that every
requested dimension came back with a score, treating a missing one as ungated rather than passed.

Full run: total pipeline 130.9 s (warm cache), tiers 4/5 unchanged from amendment 78 (std 46.05,
frame delta 9.88, A/V delta +0.0083 s, spatial seams 0.952/0.692, temporal seam 0.994), CLIP and
VBench as above. Command:

```
TT_DIT_CACHE_DIR=/data/kevinmi/tt_dit_cache MINIMAX_H3_DIFFUSERS_DIR=/data/cglagovich/MiniMax-H3-diffusers \
  ./python_env/bin/python -m pytest models/tt_dit/tests/models/minimax_h3/test_pipeline_minimax_h3.py -x -s
```

**M9 is closed.** Every tier from 1 to 6 is green at the production working point.

---

## Amendment 81 (2026-08-04) — fully-warm e2e latency, measured by LTX's method: **81.1 s Total (compute)**

Amendment 78's "133.8 s warm" was **not measured the way this repo measures**, and was wrong in two
ways. Corrected here by copying `pipelines/ltx/pipeline_ltx_distilled.py`'s method exactly, so H3's
number and LTX's are directly comparable.

### What LTX does that amendment 78 did not

1. **`(label, seconds)` rows, "prepares and export excluded".** LTX's own comment. Every
   `_prepare_*` runs *outside* its timed row and the mp4 write is not timed. Amendment 78's
   "denoise 104.6 s" included the ~50 s transformer cache load inside the window — the measurement
   contract in `.claude/skills/README.md` says weight upload is one-time construction cost and is
   **never** counted, and it was.
2. **A warmup pass.** `LTXPipeline.warmup_buffers` runs the whole shape once before anything is
   measured. There was no H3 equivalent, so amendment 78's number was a *first* call.
3. **`Total (compute)` is the sum of the stage rows**, not a wall-clock bracket around `__call__`.

Implemented: `MiniMaxH3Pipeline.warmup()` (the `warmup_buffers` analogue), `last_timings` exposed as
LTX exposes it, `Encoder (cache)` vs `Encoder` labels, prepares hoisted out of every timed row, and
`time.time()` for consistency with LTX.

### The measurement

```
timeout 7500 ./python_env/bin/python -m pytest \
  models/tt_dit/tests/models/minimax_h3/test_performance_pipeline_minimax_h3.py -x -s
```
mesh **4x8 Blackhole, TP=4 axis 0 / SP=8 axis 1, ring, 2 links** · input **1344x768, 124 frames
@ 24 fps (5.17 s), 49 forwards** · warm window **one full warmup generation; prepares and export
excluded** · SHA `0c4ce3596b5` + working tree · device time not separated from wall.

| row | seconds | share |
|---|---|---|
| Encoder (cache) | 0.0 | 0.0 % |
| **Denoise** | **61.7** | **76.1 %** |
| VAE decode | 17.6 | 21.7 % |
| Audio decode | 1.8 | 2.2 % |
| **Total (compute)** | **81.1** | |

**1259.9 ms per forward** (49 forwards over 61.7 s). **Realtime factor 15.7x** — 81.1 s of compute
per 5.17 s of video.

### Warmup is not a formality: it is worth 1.4x on the total

The warmup call's own rows against the measured call's:

| row | warmup call | **measured (warm)** |
|---|---|---|
| Encoder | 280.4 s (device, cache miss, 50 GB read) | 0.0 s (cache) |
| Denoise | 104.7 s | **61.7 s** |
| VAE decode | 18.9 s | **17.6 s** |
| Audio decode | 5.1 s | **1.8 s** |

Denoise 104.7 -> 61.7 s and audio decode 5.1 -> 1.8 s. Warmup total 439.4 s. Quoting a first call as
"warm" overstates this pipeline's latency by ~1.4x on the total and ~1.7x on denoise, which is
exactly why LTX has a warmup pass and why this now does too.

### One number worth someone's attention later, not acted on here

1259.9 ms per forward against `MiniMaxH3_perf_log.md`'s **879 ms** for the 50-block stack at
`5s_768p` (17.58 ms x 50). The ~381 ms/step difference is everything the perf log excludes by
construction: the token refiner, the input projections, `norm_out`, the two output heads, the new
ROW_MAJOR layout round-trip, and the per-step host work (metadata build and upload, two velocity
read-backs, two scheduler steps). That is **30 % of per-step time outside the measured block stack**.
Not investigated, not tuned — the directive was current perf. It is the obvious first question for
whoever picks up `tt-dit-benchmark-profile`.

New: `tests/models/minimax_h3/test_performance_pipeline_minimax_h3.py`. It reports rather than gates
— `EXPECTED_TOTAL_S = 400.0` is a did-something-collapse bar, not a target, since there is no tuned
baseline to regress against.

---

## Amendment 82 (2026-08-04) — the VAE decode stage, profiled: it is host-transfer-bound, not compute-bound. Two bugs of mine, and a hard stop on cheap on-device stitching

Directive moved to "get VAE e2e as close to 1 s as possible". First: instrument, because "VAE decode: 17.6 s"
is a number with nowhere to go. `MiniMaxH3Vae` now always collects a per-decode breakdown
(`last_decode_profile`); `MINIMAX_H3_VAE_PROFILE=1` adds the per-wave sync that makes device and
readback separable.

### Two bugs of mine, worth 12 s together

1. **The per-shape decoder's weight upload was inside the timed row.** `_prepare_vae` built only the
   wrapper and loaded the *host* state dict; `_decoder_for` uploaded ~4.6 GB lazily on first
   `decode()`. Measured at **12.1 s**. Now forced in the prepare via
   `_prepare_vae(decode_shape=...)`, where the measurement contract puts weight upload.
2. **`_make_resident` evicted the DiT and cleared the VAE decoders every generation**, so the decoder
   was rebuilt per call and the DiT would have reloaded (~50 s) on the next. They **do** co-fit on a
   4x8 Blackhole mesh --- verified, no allocation failure --- so co-residency is now the default
   (`MINIMAX_H3_CORESIDENT=0` restores eviction for a mesh where they do not).

**VAE decode row: 17.6 s -> 6.0 s -> 5.3 s.**

### Where the ~4.8 s actually goes (production shape, 4x8, warm)

```
VAE decode profile: 4.81 s over 7 waves / 196 units (32 devices, 28.0 units/wave)
    device         1.25 s  (25.9 %)   178 ms/wave
    readback       1.96 s  (40.8 %)   281 ms/wave
    stitch         1.09 s  (22.7 %)
    unpatchify     0.19 s  ( 3.9 %)
    residual       0.20 s  ( 4.2 %)
    upload         0.09 s  ( 1.9 %)
    tiling         0.00 s  ( 0.0 %)
    readback volume 5.02 GB
```

**Device compute is 1.25 s at 178 ms/wave**, against amendment 56's 150 ms/wave min-of-8 for the bare
forward. So that amendment's "768P/5s decode 1.0 s" was a **device-only projection and it was
essentially right** --- 7 waves x 150 ms = 1.05 s. What it excluded is the 3.5 s of host work. The
stage is transfer-bound, and DP=32 is confirmed working: 196 units, 7 waves,
`ShardTensorToMesh(dim=0)`, one unit per device.

**No CCL is involved in the denoise -> VAE handoff and none is needed.** The transformer all-gathers
its output on SP and TP, so the velocity is replicated on all 32 devices; every device already holds
everything required to slice its own work unit. The host round trip exists only because tiling and
denormalization live on host.

### Two exact wins landed

- **Per-tile `.float()` instead of whole-batch.** The 5.02 GB was the fp32 *intermediate*: the device
  output is bf16, the wire transfer was 2.51 GB, and `.float()` over the whole 32-tile batch
  allocated 5 GB to then slice 32 ways. Readback volume **5.02 -> 2.51 GB**.
- **Pipelined readback.** Wave N+1's compute is enqueued before wave N is read, so transfer overlaps
  compute instead of following it. Costs one extra tile of device memory per device.

Both are numerically identical, and gated: `test_vae_minimax_h3.py` **9 passed**, PCC 99.9977-99.9986 %.
Readback time 1.96 -> 1.55 s; the stage row 5.5 -> 5.3 s.

### The cheap on-device stitch is ruled out by measurement, not by opinion

The obvious device formulation is separable weighted accumulation: multiply each tile by a ramp mask
locally (no communication), then accumulate. **It is not equivalent.** Against `stitch_tiles` at the
production geometry (4x7 tiles, overlaps [96,80,80] / [80,80,80,80,64,64]):

| | |
|---|---|
| max abs difference | **4.66** |
| mean abs difference | 0.032 |
| pixels differing > 1e-5 | **11.1 %** |

The reference scheme is sequential and asymmetric: for an interior tile the corner region is
`b*L + (1-b)*(a*A + (1-a)*T)`, where `L` is the **unblended** left tile and the diagonal tile does not
appear at all. Separable weighting would change a ninth of every frame by O(1) --- which the artifact
rubric says surfaces as seams, the exact defect this campaign spent effort proving absent.

So exact on-device stitching needs each tile's **above and left neighbours co-located**, and tiles are
one-per-device by construction. That means an all-gather per chunk (28 tiles x 22 MB = 616 MB to every
device) or a work-assignment change (e.g. one device owns a column strip, making the vertical blend
local and leaving only a thin horizontal halo). Device-side `unpatchify` is *not* the blocker ---
`ttnn.permute` handles the 8-dim `(B,T,H,W,C,pt,p,p) -> (B,C,T,pt,H,p,W,p)` permutation, verified.

**Not attempted.** It is a redesign of a numerically-gated path, and the floor it buys is bounded:
device compute is 1.25 s, so the best case is ~1.3-1.5 s for the stage, not 1.0 s.

### Method note: denoise wall time varies +-8 % run to run, so single-run totals are not comparable

Denoise across five warm runs at the identical shape and seed: **61.7, 61.4, 56.6, 67.0, 71.3 s**. Any
claim of the form "total went from X to Y" that rests on one run of each is partly noise. The VAE
figures quoted above (17.6 -> 6.0 -> 5.3) are an order of magnitude outside that spread and are real;
the *total* (81.1 -> 63.9 -> 74.1) is not a clean comparison and should not be quoted as a trend.

---

## Amendment 83 (2026-08-04) — `fast_device_to_host` and the device stitcher. VAE decode 17.6 -> 4.3 s, and device compute is now 1.07 s

### The readback was going through an on-device all_gather

`ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(...))` performs an **on-device all_gather
before the transfer**, so every device first receives all 32 tiles it has no use for.
`utils/tensor.py::fast_device_to_host` exists precisely for this and is what `vae_ltx.py` and
`vae_wan2_1.py` use: async DMA of each device's own shard, zero-copy `to_torch` where the layout
allows, host-side concat, no fabric traffic at all. It also takes a `pre_transfer_fn` -- the LTX VAE
passes `float_to_uint8` there to shrink the data *before* it crosses PCIe.

Switched the decoder readback to `fast_device_to_host(decoded, mesh, concat_dims=[0, 0])`. Clean A/B
with the per-wave sync on in both arms, so this is the readback change alone:

| | before | after |
|---|---|---|
| readback per wave | 281 ms | **240 ms** (warm-up call) / **172 ms** (measured call) |

### Where the stage stands, fully warm

```
VAE decode profile: 3.66 s over 7 waves / 196 units (32 devices, 28.0 units/wave)
    device         1.07 s  (29.1 %)   152 ms/wave
    readback       1.20 s  (32.8 %)   172 ms/wave
    stitch         0.90 s  (24.5 %)
    unpatchify     0.29 s  ( 8.0 %)
    residual       0.15 s  ( 4.1 %)
    upload         0.02 s  ( 0.5 %)
    host_prep      0.03 s  ( 1.0 %)
    readback volume 2.51 GB
```

**Device compute is 1.07 s at 152 ms/wave, which is amendment 56's 150 ms min-of-8 exactly.** The
"1.0 s decode" in that amendment is now independently confirmed as the *device* figure, and reached in
the real pipeline rather than in a microbenchmark.

**VAE decode row: 17.6 -> 6.0 -> 5.3 -> 4.3 s** (4.1x). Total (compute) 63.0 s. Every step gated:
`test_vae_minimax_h3.py` **9 passed**, PCC 99.9977-99.9986 %, unchanged throughout.

### The device stitcher is written and validated, but not wired

`models/vae/minimax_h3/stitch_device_minimax_h3.py`: `DeviceTileStitcher.blend` / `.stitch` and
`unpatchify_device`, mirroring the host functions **in the reference's order** rather than
reformulating them. `test_stitch_device_minimax_h3.py`, **4 passed, PCC 100.0000 %** against the host
originals at the production 4x7 geometry with the real overlaps -- including each of the 3 horizontal
and 6 vertical seam bands checked on its own, because a whole-canvas metric dilutes exactly the defect
this risks introducing. `ttnn.permute` handles the 8-dimensional unpatchify permutation directly.

What it would buy: `stitch` (0.90 s) and `unpatchify` (0.29 s) move to device, and the readback stops
being 2.51 GB of overlapping tiles. With `pre_transfer_fn=float_to_uint8` the transfer becomes the
final canvas as uint8 -- 124 x 768 x 1344 x 3 = **384 MB, 6.5x less** -- which should take readback
from 1.20 s to ~0.2 s. Projected stage: **~1.4-1.6 s.**

**Why it is not wired yet, and the one measurement needed.** Tiles are one-per-device, and the
reference blend needs each tile's *above* and *left* neighbours (amendment 82: the separable
reformulation moves 11.1 % of pixels by up to 4.66, so it is not an option). Co-locating them costs an
all-gather of 28 x 22 MB = **616 MB to every device, per chunk, 7 chunks**. That is ~4.3 GB of fabric
traffic to remove ~2.1 GB of PCIe traffic and 1.2 s of host work. **It could plausibly be slower**, and
nothing measured so far says which way it goes. Measure the all-gather in isolation at this shape
before committing to the wiring -- and prefer the alternative if it loses: assign each device a
*column strip* of one chunk, which makes every vertical blend local and leaves only a thin horizontal
halo instead of a full gather.

Method note: the honest floor for this stage is **1.07 s of device compute**, so the target is ~1.4 s,
not 1.0 s. Quoting 1.0 s as achievable for the *stage* would be quoting a device-only number as a
wall-clock one -- the same conflation amendment 82 had to untangle.

---

## Amendment 84 (2026-08-04) — the all-gather is nearly free; `float_to_uint8` on the canvas is not. Measured before wiring

Amendment 83 said to measure the all-gather before committing to the device stitch. Done, at the
production tile geometry on the 4x8 mesh, one chunk's worth of tiles (32 x (1,3,28,256,256)).

| per chunk | fp32 | bf16 |
|---|---|---|
| `fast_device_to_host`, all tiles (**what runs today**) | 231.5 ms | **90.9 ms** |
| `all_gather` both mesh axes, full tiles | **8.5 ms** | **4.4 ms** |
| `all_gather` halo only (69 % of a tile) | 1.4 ms | 1.0 ms |
| readback stitched canvas with `pre_transfer_fn=float_to_uint8` | 333.4 ms | 326.5 ms |

**The all-gather costs 4-8 ms against a 91-231 ms readback -- 20-27x cheaper.** Amendment 83's worry
that "~4.3 GB of fabric traffic to remove ~2.1 GB of PCIe traffic could plausibly be slower" is
**wrong**: fabric bandwidth on this mesh is not remotely the constraint. The gather was verified real
(local dim 0 goes 1 -> 32) rather than trusted from a timer.

**But the readback shape of the win is the opposite of what was projected.** Amendment 83 predicted
readback would fall to ~0.2 s via `float_to_uint8` shrinking the canvas to 384 MB of uint8. Measured,
that path costs **326 ms per chunk -- worse than reading every tile in bf16 (90.9 ms)**. The cause is
not the transfer: `float_to_uint8` does `to_layout(TILE)` ... `to_layout(ROW_MAJOR)` around its
arithmetic, and two full layout round-trips over 87 M elements swamp the bytes they save. **Copying the
LTX call without measuring it at this shape would have made the stage slower while looking like an
optimization.**

### So the winning combination is not the one that was planned

- **all-gather: yes.** 4.4 ms in bf16, and it co-locates the neighbours the reference blend needs.
- **`float_to_uint8` before readback: no.** Leave the uint8 conversion on host, where it is cheap.
- **canvas readback in bf16, no layout round-trip.** The stitched canvas is ~173 MB against ~347 MB of
  overlapping tiles, so this halves the transfer instead of the 6.5x that uint8 promised.

Projected per chunk: all-gather 4.4 ms + device unpatchify + device stitch + ~45-90 ms readback,
against today's 172 + 129 + 41 = **342 ms**. Stage **4.3 -> ~2.9 s** if the device stitch itself is
cheap, against a device-compute floor of 1.07 s.

### Two loose ends, recorded rather than assumed

1. **The two-axis gather permutes the batch.** `gathered replica matches host: False, maxdiff 7.93` --
   gathering `cluster_axis=0` then `cluster_axis=1` reassembles dim 0 in a different order than
   `ShardTensorToMesh(dim=0)` fractured it. Harmless *if* the permutation is known, and the tile ->
   device map must be derived from it rather than assumed to be row-major. This is the next thing to
   pin down, and getting it wrong puts tiles in the wrong place, which the seam gate would catch as a
   spectacular failure rather than a subtle one.
2. **The in-pipeline readback is 172 ms/wave while this standalone measurement of the same volume in
   bf16 is 90.9 ms.** Same 352 MB, ~2x apart. Unexplained -- candidates are TILE-layout padding on the
   token-shaped tensor versus the pixel-shaped one here, or DiT co-residency. Worth 0.57 s over 7 waves
   if it is addressable, which is comparable to the whole device-stitch win and much cheaper to chase.

Method note, third instance this campaign: **a pattern copied from another model is a hypothesis, not
a result.** `fast_device_to_host` was a real 39 % win; `float_to_uint8` from the same file at the same
call site is a 3.6x regression. The difference was one measurement.

---

## Amendment 85 (2026-08-04) — the readback 2x was host allocator pressure, not transfer. Default flipped

Amendment 84 left two loose ends. This closes the second: the in-pipeline readback was 172-240 ms/wave
while an isolated measurement of the identical 352 MB in bf16 was 90.9 ms.

It was neither the shape nor a min-versus-mean artifact (the two candidates amendment 84 named).
Instrumenting **per-wave** rather than accumulating a mean showed it immediately:

```
5 chunks/group   readback per wave [92 223 277 218 223 274 215]   median 223 ms
1 chunk/group    readback per wave [88  89 148 101  89  99  86]   median  89 ms
```

**The first wave was always ~90 ms** --- exactly the isolated figure. Later waves degraded because
`_run_decoder_units` is called per *group*, and a 5-chunk group accumulates **140 tiles x 22 MB ~= 3.1 GB
of fp32 pixels** on host before anything is stitched and released. Host allocator pressure, not PCIe.

`_DECODE_WAVES_IN_FLIGHT = 4` existed to bound exactly this and could not: its `ceil` arithmetic makes
`ceil(1 * 32 / 28) = 2`, so one chunk per group was unreachable. Changed to floor and the default
flipped to **1**. Device time is unchanged --- the wave count follows the total unit count and every
wave pads to the mesh size regardless --- so the smaller group is free.

| | readback | stage row |
|---|---|---|
| 5 chunks/group | 1.55 s (223 ms/wave median) | 4.3 s |
| **1 chunk/group** | **0.70 s (89 ms/wave median)** | **3.8 s** |

Full stage now: device 1.06 s (35.1 %), readback 0.70 s (23.1 %), stitch 0.66 s (21.6 %), unpatchify
0.26 s, residual 0.21 s, host_prep 0.12 s, upload 0.02 s. **Device compute is now the largest single
term**, which it was not before.

**VAE decode: 17.6 -> 6.0 -> 5.3 -> 4.3 -> 3.8 s**, against a 1.06 s device floor.

Method note: *a per-stage total cannot show a trend within the stage.* Two amendments chased this
number with means and got the wrong candidates; one list of per-wave times settled it. Prefer
distributions over sums whenever a stage repeats a step.

Consequence for the device stitch (amendments 82-84): its remaining prize shrank from 1.09 s of host
stitch to **0.66 s**, while its cost --- an all-gather at 4.4 ms/chunk --- did not change. Still worth
doing, but it is no longer the largest term and should be re-argued against the denoise loop, which is
**91 %** of the fully-warm total.

## Next step

Two candidates, in the order their size suggests:

1. **The denoise loop, 91 % of fully-warm total.** 1155-1367 ms per forward against
   `MiniMaxH3_perf_log.md`'s 879 ms for the 50-block stack, so ~30 % of each step is outside the
   measured blocks: token refiner, input projections, `norm_out`, the two heads, the ROW_MAJOR layout
   round-trip, and per-step host work (metadata build, two velocity read-backs, two scheduler steps).
   None of it has been attributed. Instrument the loop the way `MiniMaxH3Vae._report_profile` now
   instruments decode --- per-step, as a distribution --- before touching anything.
2. **Wire the device stitch** (`models/vae/minimax_h3/stitch_device_minimax_h3.py`, validated at
   PCC 100.0000 % but **unwired**). Worth ~0.66 s of the 3.8 s stage. First pin down the batch
   permutation the two-axis all-gather applies (amendment 84, loose end 1) --- getting it wrong puts
   tiles in the wrong place.

Also open, and cheap: denoise wall time varies +-8 % run to run (56.6-71.3 s at identical shape and
seed), so any future perf claim needs repeated runs, not one of each.

---

## Amendment 86 (2026-08-04) — RETRACTION of amendment 83's readback win and amendment 85's absolute numbers: `fast_device_to_host(concat_dims=[0, 0])` was returning zeros

**What amendment 83 claimed.** That switching the decoder readback from
`ttnn.to_torch(mesh_composer=ConcatMeshToTensor(dim=0))` to
`fast_device_to_host(decoded, mesh, concat_dims=[0, 0])` was a correctness-neutral win, "readback per
wave 281 -> 240 ms / 172 ms", on the grounds that the composer route performs an on-device all_gather
and `fast_device_to_host` does not. Amendment 85 then built on it, reporting 89 ms/wave after the
grouping change.

**Why it was wrong.** `concat_dims` names, per mesh axis, the tensor dimension to concatenate along.
It is for a tensor fractured on **different** dims per axis --- LTX shards a VAE activation with H on
one mesh axis and W on the other, which is why `vae_ltx.py` and `vae_wan2_1.py` call it. The decoder's
output is fractured **32 ways along dim 0** by `ShardTensorToMesh(dim=0)`, so passing dim 0 for *both*
axes is not a valid spec. Measured directly, with each batch row set to a distinct constant:

```
fast_device_to_host(concat_dims=[0, 0])   [24 25 26 27 28 29 30 31  0 0 0 ... 0]
ttnn.to_torch(ConcatMeshToTensor(dim=0))  [ 0  1  2  3 ... 31]                     correct
```

One mesh row of real data; the remaining 24 rows never written. **It was faster because it was not
moving the data.** Every number amendment 83 and 85 quote for readback is a measurement of a transfer
that did not happen, and the 90.9 ms "isolated" figure in amendment 84's loose end 2 was the same
misuse, which is why the two agreed.

**The correct reading.** Reverted to `ConcatMeshToTensor`. The genuine readback cost is what amendment
82 recorded before any of this: **~281 ms/wave**, and it is still the second-largest term in the stage.
The grouping change (amendment 85, `_DECODE_WAVES_IN_FLIGHT` 4 -> 1) is **not** retracted --- both arms
of that A/B used the same broken readback, so the *relative* finding holds --- but its absolute
per-wave numbers do not, and the stage total needs re-measuring on the fixed path.

**How it was caught, and what it says about the gates.** The per-shard numerics suite passed
throughout: 15 tests, PCC 99.9977-99.9986 %. It could not catch this, because every individual shard
*is* correct and the roundtrip tests run on a **1x1 mesh** where `concat_dims=[0, 0]` is trivially
valid. What caught it was the **tier-6 CLIP prompt-alignment gate on the first artifact-checking run
after the change: 37.37 -> 19.58**, far below its 33.0 bar. A whole-video PCC would not have flinched;
the tiles were individually perfect and merely in the wrong places.

| The method note | |
|---|---|
| The rule that would have caught it sooner | **A readback that gets faster without moving fewer bytes has not got faster.** 281 -> 172 ms on identical volume should have been interrogated, not banked |
| The second rule | An API borrowed from another model needs its *contract* checked, not just its call site copied. `concat_dims` was the third pattern taken from `vae_ltx.py` in two amendments; `fast_device_to_host` was misused, `float_to_uint8` was a 3.6x regression (amendment 84), and only the third was neutral |
| Why a multi-device order check now exists | Nothing gated mesh **reassembly order**. Every numerics test is either single-device or per-shard. That is the hole this went through |

---

## Amendment 87 (2026-08-04) — two showcase generations, and VBench `imaging_quality` is prompt-dependent

Two manual runs on the fixed readback path, outside the gate, to see what the pipeline actually does
with harder content. Both tiers 4 and 5 green; artifacts kept out of the gated directory.

| | rain-at-night alley | The Office dialogue |
|---|---|---|
| artifacts | `~/h3_t2va_tokyo/t2va.mp4` | `~/h3_t2va_office/t2va.mp4` |
| prompt tokens | 98 | 68 |
| audio peak / rms | 0.065 / 0.0078 | **0.426 / 0.0335** |
| spatial seam v / h | 1.358 / 1.248 | 0.836 / 1.438 |
| temporal seam | 1.042 | 0.940 |
| CLIP | 35.43 | not run |

Two things worth recording:

- **The audio branch responds to content.** A dialogue prompt produced a soundtrack 6.5x louder in
  peak and 4.3x in rms than a quiet ambient night scene. Nothing in the pipeline conditions audio
  loudness explicitly, so this is the model, and it is evidence the audio path is doing something
  content-dependent rather than emitting generic texture.
- **Seam ratios move with content, and 1.0 is not the expectation.** Neon reflections put real
  high-gradient structure across tile boundaries and pushed the vertical ratio to 1.358; the office
  scene pushed the *horizontal* ratio to 1.438 on a shelf line. Both are far under the 2.0 bar and both
  frames are visually clean on inspection. **A ratio near 1.0 is what a *smooth* scene gives, not what
  a correct one gives** --- worth knowing before someone reads 1.4 as a defect.

### `imaging_quality` cannot be a fixed bar across prompts

The night scene scored **imaging_quality 0.4884 against the 0.64 bar** while being entirely correct on
inspection --- the metric is a no-reference IQA model that rewards sharp, well-lit frames, and a dark,
hazy, shallow-depth-of-field scene is none of those. Its other four dimensions passed
(subject 0.9562, background 0.9556, motion 0.9944, dynamic 1.0).

So the bar was **not** loosened to accommodate it. `imaging_quality = 0.64` stays, and the *gated*
prompt stays the daylight fox scene it was calibrated against (amendment 80). The single-sample
calibration caveat written into `test_pipeline_minimax_h3.py` is now a measured fact rather than a
worry: **the gated prompt and its thresholds are a matched pair, and a showcase prompt belongs in a
manual run.** Loosening the bar to 0.48 to admit a dark scene would have left it unable to detect
anything.

---

## Amendment 88 (2026-08-04) — CORRECTION to amendment 72: `0c4ce3596b5` was never dangling. It was cglagovich's branch tip

**What amendment 72 claimed.** That `0c4ce3596b5` is "a *dangling* commit, reachable from no ref, whose
parent is `gh/cglagovich-minimax-h3` @ `b85be88d6d3`", and its method note: "a superset commit that no
ref points at is one `git gc` from gone. It is now on a branch."

**Why it was wrong.** `git branch -a --contains` and `git log --all` were run against a **local** clone
whose `gh/cglagovich-minimax-h3` ref had been fetched when that branch was at `b85be88d6d3`. The ref
was stale, not the commit orphaned. Asked directly:

```
git ls-remote https://github.com/tenstorrent/tt-metal.git refs/heads/cglagovich/minimax-h3
0c4ce3596b53036d3b460670ebdcf1761c687bea    refs/heads/cglagovich/minimax-h3
```

`0c4ce3596b5` **was the tip of `cglagovich/minimax-h3` all along.** Nothing was ever at risk and
nothing was rescued.

**What does not change.** The tree that was branched from, and therefore every gate and measurement in
amendments 73-87, is byte-identical either way. The decision to base on that commit was correct for the
reason given (it is the only commit holding both the tuned DiT and the VAE work); only the "dangling"
justification was fiction.

**Where it landed.** `7d4797dad76` was pushed as a **fast-forward of exactly one commit** onto
`cglagovich/minimax-h3` (`0c4ce3596b5..7d4797dad76`), verified with `git merge-base --is-ancestor`
beforehand and no force. Also pushed as `kevinmi/minimax-h3-t2va` for reference.

**The method note.** *Local reachability is not remote reachability.* `git branch -a --contains` answers
a question about this clone's refs, and a `remotes/` ref is only as fresh as its last fetch. To ask
whether a commit exists on the server, ask the server: `git ls-remote`. Convenient conclusions about
someone else's branch deserve that extra round trip -- and this one had been carried, unchallenged,
through sixteen amendments.
