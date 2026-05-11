# PI0.5 (pi0_5) — Tenstorrent

PI0.5 builds directly on top of the existing `models/experimental/pi0` model.
Rather than fork the whole pipeline, this package only contains the
**deltas** from PI0; everything else (SigLIP, VLM blocks, weight loader,
prefix embedding, denoising loop) is imported and reused.

## What's different from PI0

The PI0.5 architecture (per the openpi reference and pi0.5 blog post) differs
from PI0 along two axes that affect the inference graph:

| Component        | PI0                                              | PI0.5                                                         |
| ---------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| Suffix tokens    | `[state_token, action_0, …, action_{H-1}]`       | `[action_0, …, action_{H-1}]` (state is part of lang tokens)  |
| Time injection   | concat(action, sincos(t)) → 2-layer MLP, fused   | sincos(t) → MLP → `adarms_cond` (used by adaRMSNorm)          |
| Expert RMSNorm   | Plain RMSNorm                                    | adaRMSNorm: `normed * (1 + scale) + shift`, with gated residual|
| `max_token_len`  | 48                                               | 200                                                           |

Everything else (SigLIP-27, Gemma-2B VLM, Gemma-300M expert, flow-matching
denoising with Euler integration, KV-cache prefill of the prefix) is identical
and reused via subclass-over-fork.

```
prefix (images + lang(state)) ──► VLM 18× ──► KV cache
                                                  │
noisy_actions ──action_in_proj──► suffix          │   for t in [1.0 → 0.0]:
                                       └──►       │      AdaRMS Expert 18×  (uses adarms_cond)
sincos(t) ── time_mlp ──► adarms_cond ────────────┘             │
                                                                ▼
                                                  action_out_proj → velocity
                                                  x_t ← x_t + dt · velocity
```

## Directory layout

```
pi0_5/
├── common/
│   └── configs.py              # Pi0_5ModelConfig (pi05=True, max_token_len=200)
├── reference/                  # PyTorch reference
│   ├── torch_suffix.py         # sincos-MLP → adarms_cond, no state token
│   ├── torch_gemma.py          # AdaRMSGemmaBlock + ada_rms_norm
│   ├── torch_paligemma.py      # forward_expert(..., adarms_cond)
│   └── torch_pi0_5_model.py    # Pi0_5Model.sample_actions()
├── tt/                         # TTNN implementation
│   ├── ttnn_suffix.py
│   ├── ttnn_gemma.py
│   ├── ttnn_paligemma.py
│   └── ttnn_pi0_5_model.py
└── tests/pcc/                  # PCC / smoke tests
    ├── test_pcc_suffix.py
    └── test_pcc_adarms_gemma.py
```

## Inference (mirrors openpi's `sample_actions`)

```python
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model
from models.experimental.pi0.common.weight_loader import PI0WeightLoader

loader = PI0WeightLoader("/path/to/pi0_5_weights")
model  = Pi0_5Model(Pi0_5ModelConfig(), loader)

actions = model.sample_actions(
    images=[img_tensor],            # list of (B, 3, 224, 224)
    img_masks=[mask],
    lang_tokens=tok_ids,            # (B, L) — includes discretized state
    lang_masks=tok_mask,
    state=None,                     # ignored on the pi0.5 path
)  # → (B, action_horizon, action_dim)
```

TTNN flow is identical (`Pi0_5ModelTTNN.sample_actions(...)`); device tensors
are constructed inside the model and the denoise loop stays on the device.

## Weights

The expert checkpoint must contain the adaRMS modulation tensors per layer:

```
model.layers.{i}.pre_attention_adaln.weight   # (3 * width, width)
model.layers.{i}.pre_attention_adaln.bias     # (3 * width,)            optional
model.layers.{i}.pre_ffw_adaln.weight         # (3 * width, width)
model.layers.{i}.pre_ffw_adaln.bias           # (3 * width,)            optional
```

and the suffix checkpoint must contain `time_mlp_in.{weight,bias}` /
`time_mlp_out.{weight,bias}` (in addition to `action_in_proj` and
`action_out_proj`). `state_proj` and `action_time_mlp_*` from PI0 are
**not** used.

If your checkpoint uses different names (e.g. an unnamed `Dense` from flax),
add a rename pass in `PI0WeightLoader.categorize_weights` or in
`Pi0_5PaliGemmaBackboneTTNN._inject_adarms_weights`.

## Tests

```bash
# Reference smoke tests (no device required)
python_env/bin/python -m pytest models/experimental/pi0_5/tests/pcc -v
```

## License

SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
