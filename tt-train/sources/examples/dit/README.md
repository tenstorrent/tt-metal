# DiT: Diffusion Transformer training on tt-train

Class-conditional image-generation training (DDPM objective, DiT architecture)
implemented entirely in Python on `ttml` — the first non-LLM training example
in tt-train. Trains on CIFAR-10 in pixel space and scales from one chip to a
32-chip Galaxy via data parallelism.

Measured on Blackhole (p150): DiT-S (32.6M params, patch 4) trains at
~1,050–1,300 img/s on a single chip (batch 64–256); the patch-2 variant reaches
~2,380 img/s at global batch 2,048 across a 32-chip Galaxy (DDP).

## Layout

| File | Purpose |
|---|---|
| `dit_ttml.py` | The model: DiT blocks (adaLN-zero), non-causal attention, custom `SliceLastDim` autograd op |
| `diffusion.py` | Host-side (numpy) DDPM schedule, patchify/unpatchify, batch construction |
| `train_dit_cifar.py` | Config-driven trainer: EMA, cosine LR, DDP, checkpointing |
| `sample_from_ckpt.py` | Offline DDIM sampling (with CFG) from a saved checkpoint |
| `test_device_primitives.py` | On-device gate: every primitive the model relies on, cheapest first |
| `reference_torch.py`, `test_reference.py` | PyTorch golden model (CPU) mirroring the ttml implementation 1:1 |
| `edm.py` | EDM (Karras 2022) recipe, host side: preconditioning, batch builders (token + image space), Heun sampler, SongUNet block plan |
| `edm_ops.py` | Convolutional autograd Functions for ttml: `Conv3x3Im2col`, `GroupNormMoreh`, `AvgPool2x2`/`UpsampleNearest2` (adjoint pair), `Permute`, `ConcatChannels`, `Scale` |
| `edm_unet.py` | DDPM++ SongUNet (EDM CIFAR-10 config, 56.7M params) on ttml, mirroring `reference_unet.py` 1:1 |
| `reference_unet.py`, `test_reference_unet.py` | PyTorch golden SongUNet + CPU validation (im2col ROW_ORDER, param count, shape walk, EDM overfit) |
| `test_edm_primitives.py` | On-device gate for the UNet primitives (run before training the UNet) |

## Run

```bash
# gate the device primitives first
python test_device_primitives.py

# single chip, small model (sanity: ~4 min at ~1.4k img/s)
python train_dit_cifar.py -c ${TT_METAL_RUNTIME_ROOT}/tt-train/configs/training_configs/training_cifar10_dit_tiny.yaml

# single chip, DiT-S patch-2 with EMA + cosine (the "quality" recipe)
python train_dit_cifar.py -c .../training_cifar10_dit_s_p2_v2.yaml

# 32-chip Galaxy DDP (set the mesh graph descriptor first)
export TT_MESH_GRAPH_DESC_PATH=${TT_METAL_RUNTIME_ROOT}/tt-train/configs/mgd/bh_galaxy_32_1_ring_ring.textproto
python train_dit_cifar.py -c .../training_cifar10_dit_s_p2_galaxy.yaml

# sample a checkpoint
python sample_from_ckpt.py -c <training_config.yaml> --ckpt runs/dit/<run>/ckpt_ema_XXXXXX.npz
```

CIFAR-10 downloads automatically to `training_config.data_path` on first run.
`--overfit-batch --max-steps 300` trains on one fixed batch (bring-up sanity:
loss must collapse).

## SongUNet (convolutional EDM backbone, Phase 1)

The EDM DDPM++ "SongUNet" (Karras 2022 CIFAR-10 config) is implemented on
ttml with **composite im2col convolutions** — native `ttnn.conv2d` re-preps
weights host-side on every call when weights change, which is unusable for
training. Trainable 3x3 conv weights are stored flattened as
`[1,1,9*C_in,C_out]` matrices with `ROW_ORDER = (kh, kw, c_in)` (see
`reference_unet.py` for the full spec and the documented deviations from
official EDM). Activations flow as `[B,1,H*W,C]` channels-last tokens so 1x1
convs, attention, and all conditioning broadcasts reuse the validated DiT
machinery; GroupNorm wraps the `ttnn.moreh.group_norm` kernel pair in
NHWC<->NCHW permutes; avgpool/nearest-upsample are implemented as an exact
adjoint pair. Bring-up order:

```bash
python test_reference_unet.py     # CPU (torch): golden model, 56.7M params, overfit
python test_edm_primitives.py    # device gate: conv/GN/pool/attn fwd+bwd parity
```

**Native conv forward (Phase 2):** `EDM_NATIVE_CONV=1` (or
`smoke_unet_train.py --native-conv`) routes the 3x3 conv *forward* through
native `ttnn.conv2d`, which consumes our flat `[1,1,9Cin,Cout]` TILE weight
in place (passes the device-weight shape sniff, so per-step weight prep is
skipped entirely; height-sharded is pinned so the expected layout matches
ROW_ORDER at every shape). Backward stays the im2col composite. Gated by the
`native conv2d probe`/`parity` checks in `test_edm_primitives.py`; only
convs with `Cin % 32 == 0` take the native path (`conv_in` keeps im2col).

## Design notes (framework workarounds, kept intentionally visible)

- **Patchify lives on the host.** There is no autograd permute/transpose, only
  `reshape` — so images are patchified in the dataloader (leaf tensors need no
  grad) and the loss is computed in patch-token space, which is mathematically
  identical. Unpatchify happens only at sampling time, also host-side.
- **Labels are one-hot through a bias-free linear**, not `Embedding`:
  `ttnn::embedding_backward` requires the index tensor's last dim to be a
  multiple of TILE_WIDTH, which a single per-image label cannot satisfy.
- **Attention uses the composite SDPA with `mask=None`.** The fused SDPA kernel
  interprets a missing mask as *causal*; full attention via an explicit
  all-ones Arbitrary mask is numerically correct but showed no speedup at
  these sequence lengths.
- **adaLN modulation is n linears sharing one SiLU**, with the scale branch's
  bias initialized to 1 so the linear emits `(1 + scale)` directly. A fused
  D→6D linear + `ttnn.slice`/`concat` split (see `SliceLastDim`) is provided
  and autograd-correct, but measured ~6.5× *slower* — the training step is
  dispatch/op-count sensitive; prefer fewer, larger ops.
- **Checkpoint reads use `PreferredPrecision.NATIVE`.** The default
  `to_numpy()` precision=FULL path caches its first fp32 view and never
  refreshes it (#41657), which silently freezes checkpoints/EMA at their
  first-read values.
- **DDP** shards the global batch over the `dp` mesh axis (`axis_mapper`),
  averages gradients with `ttml.sync_gradients` after backward, and reads
  losses/params through a concat composer (first replica kept). In-loop
  sampling is disabled under DDP; sample offline from checkpoints.
