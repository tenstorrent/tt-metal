# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Architectural diff: our `Pi0_5Model` (pytorch reference) vs openpi's `PI0Pytorch`.

Both models are loaded with the SAME upstream openpi pi05_libero weights
(converted to PyTorch safetensors at /storage/sdawle/pi05_weights/pi05_libero_upstream).
We feed both the same canned observation + same noise + same timestep schedule
and diff:
  - the final action chunk
  - intermediate activations along the way (suffix embedding, prefix embedding,
    expert output) if a more granular bisect is needed

If both produce nearly-identical actions: our model architecture is correct,
and the LIBERO-rollout 0/4 failure must be in the adapter preprocessing
(image conventions, prompt tokenization, normalization, etc.).

If they differ: walk back to find where the layer activations split.

This is the *unambiguous* test that resolves the "is our Pi0_5Model =
openpi PI0Pytorch?" question, completing the Phase 3 investigation.

Usage:
    PYTHONPATH=/home/tt-admin/sdawle/pi0/tt-metal:/storage/sdawle/openpi/src \\
        python_env/bin/python models/experimental/pi0_5/tests/pcc/test_pi05_upstream_vs_ours_activation_diff.py
"""

import sys
from pathlib import Path

import torch

# Bypass lerobot's transformers-replace check (same as libero_rollout.py)
import types as _types

_fake = _types.ModuleType("transformers.models.siglip.check")
_fake.check_whether_transformers_replace_is_installed_correctly = lambda: True
sys.modules["transformers.models.siglip.check"] = _fake

REPO_ROOT = "/home/tt-admin/sdawle/pi0/tt-metal"
OPENPI_SRC = "/storage/sdawle/openpi/src"
for p in (REPO_ROOT, OPENPI_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

UPSTREAM_CKPT = Path("/storage/sdawle/pi05_weights/pi05_libero_upstream")


def make_canned_observation(device: str = "cpu"):
    """Build a deterministic Observation that both models can consume.

    Same images (all zeros = mid-gray in [-1,1]), same tokenized prompt
    ("pick up the bowl"), same state (zeros), same image masks. The exact
    content doesn't matter — only that both models see byte-identical input.
    """
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor()
    sp.load("/storage/sdawle/pi05_weights/paligemma_tokenizer.model")
    # Match openpi's tokenizer.py:33 — `bos + <desc> + "\n"`, no Task/Action scaffold.
    text = "pick up the bowl"
    tokens = sp.encode(text, add_bos=True) + sp.encode("\n")
    max_len = 200  # upstream pi05_libero training max_token_len
    if len(tokens) < max_len:
        padding = [0] * (max_len - len(tokens))
        mask = [True] * len(tokens) + [False] * (max_len - len(tokens))
        tokens = tokens + padding
    else:
        tokens = tokens[:max_len]
        mask = [True] * max_len

    # Images: 3 cameras × (B=1, C=3, H=224, W=224) in [-1, 1].
    # All zeros (mid-gray). Right_wrist is the masked-out empty slot.
    img = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device)
    images = {
        "base_0_rgb": img.clone(),
        "left_wrist_0_rgb": img.clone(),
        "right_wrist_0_rgb": torch.zeros_like(img),
    }
    image_masks = {
        "base_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
    }

    state = torch.zeros(1, 8, dtype=torch.float32, device=device)
    lang_tokens = torch.tensor([tokens], dtype=torch.int64, device=device)
    lang_masks = torch.tensor([mask], dtype=torch.bool, device=device)

    return {
        "images": images,
        "image_masks": image_masks,
        "state": state,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
    }


def run_openpi_model(obs_dict):
    """Build openpi's PI0Pytorch, load upstream weights, run sample_actions."""
    print("=" * 70)
    print(" [openpi PI0Pytorch] loading + forward")
    print("=" * 70)
    from openpi.models import pi0_config as _pi0_config
    from openpi.models_pytorch import pi0_pytorch
    from safetensors.torch import load_file

    cfg = _pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        dtype="bfloat16",  # match our pytorch reference precision (bf16, not openpi's default fp32)
        pytorch_compile_mode=None,  # disable compile so we can introspect
    )
    print(
        f"  config: pi05={cfg.pi05}, action_horizon={cfg.action_horizon}, "
        f"discrete_state_input={cfg.discrete_state_input}, dtype={cfg.dtype}"
    )

    model = pi0_pytorch.PI0Pytorch(cfg)
    sd = load_file(str(UPSTREAM_CKPT / "model.safetensors"))
    # PI0Pytorch state_dict keys are 1:1 with the converted safetensors.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  missing keys:    {len(missing)}  (first few: {missing[:3]})")
    print(f"  unexpected keys: {len(unexpected)}  (first few: {unexpected[:3]})")
    model.eval()

    # Build openpi Observation
    from openpi.models.model import Observation

    obs = Observation(
        images=obs_dict["images"],
        image_masks=obs_dict["image_masks"],
        state=obs_dict["state"],
        tokenized_prompt=obs_dict["lang_tokens"].to(torch.int32),
        tokenized_prompt_mask=obs_dict["lang_masks"],
        token_ar_mask=None,
        token_loss_mask=None,
    )

    # Force deterministic noise to match our model
    torch.manual_seed(0)
    bsize = obs.state.shape[0]
    noise = torch.normal(
        mean=0.0,
        std=1.0,
        size=(bsize, cfg.action_horizon, cfg.action_dim),
        device=obs.state.device,
    )

    with torch.no_grad():
        actions = model.sample_actions(device="cpu", observation=obs, noise=noise, num_steps=10)
    print(f"  output shape: {tuple(actions.shape)}")
    print(f"  output[0, :, :7] (first 10 actions × 7 real dims):")
    print(actions[0, :, :7].numpy())
    return actions, noise


def run_our_model(obs_dict, openpi_noise):
    """Build our Pi0_5Model, load upstream weights via our loader, run sample_actions."""
    print()
    print("=" * 70)
    print(" [our Pi0_5Model] loading + forward")
    print("=" * 70)
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model

    cfg = Pi0_5ModelConfig(action_horizon=10)
    print(
        f"  config: pi05={cfg.pi05}, action_horizon={cfg.action_horizon}, "
        f"max_token_len={cfg.max_token_len}, discrete_state_input={cfg.discrete_state_input}"
    )

    loader = Pi0_5WeightLoader(str(UPSTREAM_CKPT))
    model = Pi0_5Model(cfg, loader)

    # Our sample_actions takes a list of image tensors not a dict.
    images = [
        obs_dict["images"]["base_0_rgb"],
        obs_dict["images"]["left_wrist_0_rgb"],
        obs_dict["images"]["right_wrist_0_rgb"],
    ]
    img_masks = [
        obs_dict["image_masks"]["base_0_rgb"],
        obs_dict["image_masks"]["left_wrist_0_rgb"],
        obs_dict["image_masks"]["right_wrist_0_rgb"],
    ]

    # Inject the same noise openpi used so trajectories start from identical x_0
    # (we'll hack this in via a monkey-patch on the denoising's sample_noise).
    saved = model.denoising.sample_noise
    model.denoising.sample_noise = lambda bs, device=None, dtype=torch.float32: openpi_noise.clone()
    try:
        with torch.no_grad():
            actions = model.sample_actions(
                images=images,
                img_masks=img_masks,
                lang_tokens=obs_dict["lang_tokens"].to(torch.int32),
                lang_masks=obs_dict["lang_masks"],
                state=None,  # pi05 ignores state
            )
    finally:
        model.denoising.sample_noise = saved
    print(f"  output shape: {tuple(actions.shape)}")
    print(f"  output[0, :, :7]:")
    print(actions[0, :, :7].numpy())
    return actions


def bisect_intermediates(obs):
    """Bisect: diff prefix_embs, suffix_embs, velocity@step0 between the two models.
    Helps narrow down WHERE the architectural divergence happens.
    """
    print("\n" + "=" * 70)
    print(" BISECT: per-stage activation comparison")
    print("=" * 70)
    # ---- openpi side ----
    from openpi.models import pi0_config as _pi0_config
    from openpi.models_pytorch import pi0_pytorch
    from openpi.models.model import Observation
    from safetensors.torch import load_file as _sft

    cfg_op = _pi0_config.Pi0Config(
        pi05=True, action_horizon=10, discrete_state_input=False, dtype="bfloat16", pytorch_compile_mode=None
    )
    model_op = pi0_pytorch.PI0Pytorch(cfg_op)
    model_op.load_state_dict(_sft(str(UPSTREAM_CKPT / "model.safetensors")), strict=False)
    model_op.eval()
    obs_op = Observation(
        images=obs["images"],
        image_masks=obs["image_masks"],
        state=obs["state"],
        tokenized_prompt=obs["lang_tokens"].to(torch.int32),
        tokenized_prompt_mask=obs["lang_masks"],
        token_ar_mask=None,
        token_loss_mask=None,
    )
    images_op, img_masks_op, lang_tokens_op, lang_masks_op, state_op = model_op._preprocess_observation(
        obs_op, train=False
    )
    with torch.no_grad():
        prefix_embs_op, prefix_pad_masks_op, _ = model_op.embed_prefix(
            images_op, img_masks_op, lang_tokens_op, lang_masks_op
        )
    print(
        f"  openpi prefix_embs: shape={tuple(prefix_embs_op.shape)} "
        f"mean={prefix_embs_op.mean().item():+.5f} std={prefix_embs_op.std().item():.5f} "
        f"norm/tok={prefix_embs_op.norm(dim=-1).mean().item():.4f}"
    )

    # suffix_embs at t=1.0 with deterministic x_0 (= openpi's noise)
    torch.manual_seed(0)
    x_0 = torch.normal(mean=0.0, std=1.0, size=(1, 10, 32))
    t1 = torch.tensor([1.0])
    with torch.no_grad():
        suffix_embs_op, suffix_pad_op, suffix_att_op, adarms_cond_op = model_op.embed_suffix(state_op, x_0, t1)
    print(
        f"  openpi suffix_embs: shape={tuple(suffix_embs_op.shape)} "
        f"mean={suffix_embs_op.mean().item():+.5f} std={suffix_embs_op.std().item():.5f}"
    )
    print(
        f"  openpi adarms_cond: shape={tuple(adarms_cond_op.shape)} "
        f"mean={adarms_cond_op.mean().item():+.5f} std={adarms_cond_op.std().item():.5f}"
    )

    # ---- our side ----
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model

    cfg_ours = Pi0_5ModelConfig(action_horizon=10)
    loader = Pi0_5WeightLoader(str(UPSTREAM_CKPT))
    model_ours = Pi0_5Model(cfg_ours, loader)
    images_list = [obs["images"]["base_0_rgb"], obs["images"]["left_wrist_0_rgb"], obs["images"]["right_wrist_0_rgb"]]
    img_masks_list = [
        obs["image_masks"]["base_0_rgb"],
        obs["image_masks"]["left_wrist_0_rgb"],
        obs["image_masks"]["right_wrist_0_rgb"],
    ]
    with torch.no_grad():
        prefix_embs_ours, _, _ = model_ours.embed_prefix(
            images_list, img_masks_list, obs["lang_tokens"].to(torch.int32), obs["lang_masks"]
        )
    print(
        f"  ours   prefix_embs: shape={tuple(prefix_embs_ours.shape)} "
        f"mean={prefix_embs_ours.mean().item():+.5f} std={prefix_embs_ours.std().item():.5f} "
        f"norm/tok={prefix_embs_ours.norm(dim=-1).mean().item():.4f}"
    )

    with torch.no_grad():
        suffix_embs_ours, _, _, adarms_cond_ours = model_ours.suffix_embedding.embed_suffix(state_op, x_0, t1)
    print(
        f"  ours   suffix_embs: shape={tuple(suffix_embs_ours.shape)} "
        f"mean={suffix_embs_ours.mean().item():+.5f} std={suffix_embs_ours.std().item():.5f}"
    )
    print(
        f"  ours   adarms_cond: shape={tuple(adarms_cond_ours.shape)} "
        f"mean={adarms_cond_ours.mean().item():+.5f} std={adarms_cond_ours.std().item():.5f}"
    )

    # ---- comparisons ----
    print()
    if prefix_embs_op.shape == prefix_embs_ours.shape:
        d = (prefix_embs_op - prefix_embs_ours).abs()
        print(f"  prefix_embs   diff:  max={d.max().item():.4e}  mean={d.mean().item():.4e}")
        # localize: per-token diff. Prefix layout = 3×256 img + 200 lang = 968.
        per_tok_max = d.max(dim=-1).values[0]  # (968,)
        print(f"  per-token-max diff histogram:")
        regions = [
            ("img cam0 (0:256)", 0, 256),
            ("img cam1 (256:512)", 256, 512),
            ("img cam2 (512:768)", 512, 768),
            ("lang   (768:968)", 768, 968),
        ]
        for name, lo, hi in regions:
            seg = per_tok_max[lo:hi]
            print(
                f"    {name}:  max={seg.max().item():.4e}  mean={seg.mean().item():.4e}  "
                f"#tokens-with-diff>0.01={(seg > 0.01).sum().item()}"
            )
    else:
        print(
            f"  prefix_embs   SHAPE MISMATCH: openpi {tuple(prefix_embs_op.shape)} vs ours {tuple(prefix_embs_ours.shape)}"
        )
    if suffix_embs_op.shape == suffix_embs_ours.shape:
        d = (suffix_embs_op - suffix_embs_ours).abs()
        print(f"  suffix_embs   diff:  max={d.max().item():.4e}  mean={d.mean().item():.4e}")
    else:
        print(
            f"  suffix_embs   SHAPE MISMATCH: openpi {tuple(suffix_embs_op.shape)} vs ours {tuple(suffix_embs_ours.shape)}"
        )
    if adarms_cond_op is not None and adarms_cond_ours is not None and adarms_cond_op.shape == adarms_cond_ours.shape:
        d = (adarms_cond_op - adarms_cond_ours).abs()
        print(f"  adarms_cond   diff:  max={d.max().item():.4e}  mean={d.mean().item():.4e}")
    elif adarms_cond_op is None and adarms_cond_ours is None:
        print("  adarms_cond:  both None (impossible for pi05 — bug?)")
    else:
        print(f"  adarms_cond   ONE-NONE mismatch: openpi={adarms_cond_op is None} ours={adarms_cond_ours is None}")


def main():
    if not UPSTREAM_CKPT.exists():
        print(f"❌ Upstream checkpoint not found at {UPSTREAM_CKPT}")
        return 1
    print(f"📁 Using upstream checkpoint: {UPSTREAM_CKPT}")

    obs = make_canned_observation()
    print(
        f"📦 Canned obs: state.shape={tuple(obs['state'].shape)}, "
        f"lang_tokens.shape={tuple(obs['lang_tokens'].shape)}"
    )

    actions_openpi, noise = run_openpi_model(obs)
    actions_ours = run_our_model(obs, noise)
    bisect_intermediates(obs)

    # Compare
    print()
    print("=" * 70)
    print(" DIFF")
    print("=" * 70)
    a = actions_openpi[0, :, :7].float()
    b = actions_ours[0, :, :7].float()
    delta = (a - b).abs()
    print(f"  shape match? openpi {tuple(a.shape)} vs ours {tuple(b.shape)}")
    print(f"  max abs diff: {delta.max().item():.6f}")
    print(f"  mean abs diff: {delta.mean().item():.6f}")
    print(f"  cosine sim per timestep:")
    for t in range(a.shape[0]):
        cos = torch.nn.functional.cosine_similarity(a[t : t + 1], b[t : t + 1]).item()
        print(f"    t={t:2d}: cos={cos:.4f}  ||a||={a[t].norm().item():.3f}  ||b||={b[t].norm().item():.3f}")

    if delta.max().item() < 0.01:
        print("\n✅ MATCH (< 1% absolute diff) — architectures equivalent.")
        print("   Bug is in adapter pipeline (image preprocessing, prompt, normalizer).")
        return 0
    print("\n❌ DIFFER — architectural delta confirmed.")
    print("   Next: walk forward pass layer-by-layer to find where activations split.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
