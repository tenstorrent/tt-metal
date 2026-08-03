# MiniMax-H3 VAEs on BH Galaxy 4x8 — execution state

**Scope changed 2026-08-03 to VAE-only.** Video VAE (encode + decode) and audio VAE
(encode + decode), each with comprehensive unit tests and measured performance. The
DiT / text-encoder work is parked, green, and preserved — see "Parked work".

Plan: `models/tt_dit/models/MiniMaxH3_VAE_PLAN.md` (in-tree copy of
`~/.claude/plans/elegant-wibbling-brooks.md`). Re-read it and this file every
iteration.

Branch: **`kevinmi/minimax-h3-vae`, cut from `origin/cglagovich/minimax-h3`**
(`42ecb2e0339`), which owns the canonical folder structure and the pinned diffusers
reference. Conform to it; do not invent a layout.

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

**M4 complete** — first successful device milestone. Host contracts (M2, M3) and
the DiT weight load (M4) are all green.

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
| 5 | attention block PCC >= 0.9995 | not started | — |
| 6 | full 50-layer forward PCC >= 0.99 @ 960x544 | not started | — |
| 7 | Qwen3-VL enc (50 layers, unnormalized) + vision tower PCC | not started | — |
| 8 | video VAE + audio VAE PCC and roundtrip | not started | — |
| 9 | e2e FL2VA @ 960x544, quality gates 1/2/4/7 | not started | — |
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
