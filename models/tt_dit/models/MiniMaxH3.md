# MiniMax-H3

## Folder structure

Bringup is split by component. Each component has its own folder for team members to slot work into:

```
models/tt_dit/
├── models/
│   ├── MiniMaxH3.md                  # this file
│   ├── transformers/minimax_h3/      # denoising transformer (block, attention, RoPE)
│   ├── vae/minimax_h3/               # video VAE (AutoencoderKLMiniMaxH3)
│   └── audio_vae/minimax_h3/         # audio VAE (AutoencoderKLMiniMaxH3Audio)
└── encoders/qwen3vl/                 # text encoder — already present, shared with other models
```

There is no `__init__.py`: `tt_dit` uses namespace packages, matching `transformers/wan2_2/` and
`transformers/ltx/`. Import as e.g.
`from ....models.transformers.minimax_h3.attention_minimax_h3 import ...`.

Tests follow the layout of the other models, under `models/tt_dit/tests/models/minimax_h3/`.

## Text conditioner

MiniMax-H3 conditions on **one** hidden state of its Qwen3-VL conditioner:
`hidden_states[50]` — the *unnormalized* output of decoder layer 50 of 64, mid-stack. The
language-model head is never used and the final norm is never applied. `Qwen3VlTextEncoder` serves
this with `activation_layers=(50,)`; note that `activation_layers=None` returns the *normalized*
final state, which diffusers is explicit is **not** the conditioning H3 expects.

Verified against the released weights on a Blackhole Galaxy at PCC 99.9993% (both TP=8 axes) —
`tests/models/minimax_h3/test_text_encoder_minimax_h3.py`.

Two conditioner facts that break naive assumptions:

- **`head_dim` is 128, not `hidden_size // num_heads`.** 5120 / 64 = 80, and the derivation fails
  *silently* because 5120 % 64 == 0. The q/k/v inner dimension is 8192, wider than the 5120 residual
  stream. Always pass `head_dim` from config. (Qwen3-VL-8B, which Ideogram4 uses, happens to satisfy
  the derivation, which is why this went unnoticed.)
- **`rope_scaling.mrope_interleaved` is true.** The chunked and interleaved rotary layouts coincide
  exactly while all three M-RoPE axes share a position — i.e. for `t2va`, where the flag is a no-op.
  A vision run makes them diverge. `create_rope_tensors(..., interleaved=True)` and
  `mrope_position_ids()` cover that; see `tests/encoders/qwen3vl/test_qwen3vl_mrope.py`.

FSDP is not used: it was required on a Wormhole 2x4, where TP=4 puts 14.9 GiB of weights on a 12 GiB
chip, but a Blackhole chip is 31.9 GiB so even TP=4 fits. Without it the weights are replicated
across the non-TP axis — load bandwidth, not capacity.

## Vision tower — not yet ported

`t2va` needs no vision at all: with no `pixel_values`, `Qwen3VLModel` never runs its vision tower and
never injects deepstack features, so the conditioner reduces to a plain text decoder. `fl2va` and
`ref2va` do feed vision, at **four** depths — the embedding scatter at `<|image_pad|>` positions, plus
additive deepstack injection at decoder layers 0/1/2 from vision layers `[8, 16, 24]`.

Note the demos port at `models/demos/qwen3_vl/` is built on `LightweightModule` /
`tt_transformers`, not `tt_dit`. It is an algorithm reference, not reusable code.

Measured facts for whoever picks this up (vision config: depth 27, hidden 1152, 16 heads, patch 16,
`spatial_merge_size` 2, `out_hidden_size` 5120):

- **`head_dim` is 72, and padding to 96 is mandatory.** ttnn SDPA hard-fails unpadded with
  `TT_FATAL logical_shape[3] == legacy_shape[3]`; padded it reaches PCC 0.9997 at seq_len 128/1024/4032.
  Costs 1.33x on attention. Pad the projection weights once at load time, as the demos port does.
- **`scale` must be passed explicitly** as `72 ** -0.5`. Padding to 96 would otherwise change the
  softmax temperature via SDPA's default — wrong output, not a crash.
- **`fl2va` needs no variable-length attention.** `cu_seqlens = repeat_interleave(h*w, t).cumsum()`, so
  one image is one block: a 768x1344 keyframe is grid `[1, 48, 84]` = 4032 patches with
  `cu_seqlens = [0, 4032]`, i.e. plain full attention. Only `ref2va` (up to 9 images and 3 videos, one
  block per *frame*) needs block-diagonal masking.
- **`fast_pos_embed_interpolate` is the common path, not an edge case.**
  `num_position_embeddings` is 2304 = 48², while a 16:9 keyframe is 4032 patches.

Sequence lengths for the canvases `resolve_canvas_size` produces:

| canvas | `grid_thw` | vision patches | LLM tokens |
|---|---|---|---|
| 768x1344 (16:9, max area) | `[1, 48, 84]` | 4032 | 1008 |
| 768x1024 (4:3) | `[1, 48, 64]` | 3072 | 768 |
| 768x768 (1:1) | `[1, 48, 48]` | 2304 | 576 |

## Setup

MiniMax-H3 support is not in a released `diffusers` yet. Bringup is pinned to a specific commit of
the `diffusers` main repository, which provides the reference `MiniMaxH3Transformer3DModel`,
`AutoencoderKLMiniMaxH3`, `AutoencoderKLMiniMaxH3Audio` and `MiniMaxH3Scheduler`.

Install it into the environment you run the tests from:

```bash
pip install "diffusers @ git+https://github.com/huggingface/diffusers@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
```

Verify the reference classes resolve:

```bash
python -c "from diffusers import MiniMaxH3Transformer3DModel; print('ok')"
```

### If the environment was created by `uv`

A `uv`-created virtualenv (such as `container_python_env` in the dev container) has no `pip` of its
own. After activating it, a bare `pip install` silently resolves to the system `pip` and installs to
`~/.local`, where the venv will not see it — it reports success and has no effect. Install through
`uv` against the interpreter instead:

```bash
uv pip install --python <venv>/bin/python --no-deps \
  "diffusers @ git+https://github.com/huggingface/diffusers@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
```

`--no-deps` keeps the resolver from pulling a newer `numpy` / `Pillow` / `huggingface-hub` into an
environment that `ttnn` was built against. The pinned commit's dependencies are already satisfied by
an environment that had any recent `diffusers` installed. Re-check `import ttnn` after installing.
