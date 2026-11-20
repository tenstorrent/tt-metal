# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Alpamayo 2 Edge configuration, read from the checkpoint's config.json."""

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    hidden: int = 1152
    layers: int = 27
    heads: int = 16
    mlp: int = 4304
    patch: int = 16
    merge: int = 2
    pos_grid: int = 16
    eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"


@dataclass(frozen=True)
class ProjectorConfig:
    hidden: int = 11520
    out: int = 2048
    # Hard-coded in the upstream patch merger rather than read from the config.
    norm_eps: float = 1e-6


@dataclass(frozen=True)
class TextConfig:
    hidden: int
    layers: int
    mlp: int
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    eps: float = 1e-5
    vocab: int = 135168
    rope_theta: float = 1e8
    mrope_section: tuple = (24, 20, 20)


@dataclass(frozen=True)
class ActionConfig:
    # Fields without defaults come from config.json; only values the checkpoint does not carry have defaults.
    n_waypoints: int
    fourier_feats: int
    max_freq: float
    in_proj_hidden: int
    in_proj_layers: int
    int_method: str
    inference_guidance_weight: float
    use_classifier_free_guidance: bool
    dt: float
    accel_mean: float
    accel_std: float
    accel_bounds: tuple
    curvature_mean: float
    curvature_std: float
    curvature_bounds: tuple
    a_lambda: float
    a_ridge: float
    kappa_lambda: float
    kappa_ridge: float
    theta_lambda: float
    theta_ridge: float
    v_lambda: float
    v_ridge: float
    slip_gradient: float
    expert_non_causal_attention: bool
    action_dim: int = 2
    steps: int = 10
    # Upstream `ActionInProj`'s RMSNorm default; the checkpoint does not carry it.
    in_proj_norm_eps: float = 1e-5


@dataclass(frozen=True)
class TokenIds:
    future_start: int
    future_end: int
    future_pad: int
    future_id0: int
    history_start: int
    history_end: int
    history_pad: int
    history_id0: int
    history_vocab: int
    future_vocab: int
    image: int
    video: int
    vision_start: int
    vision_end: int
    eos: int
    pad: int


@dataclass(frozen=True)
class Alpamayo2EdgeConfig:
    vision: VisionConfig
    projector: ProjectorConfig
    text: TextConfig
    expert: TextConfig
    action: ActionConfig
    tokens: TokenIds
    model_dir: Path = field(default=Path("."))


def num_layers_override():
    """Depth limit shared by vision, LM and expert (LM == expert), for fast bring-up runs."""
    value = os.environ.get("ALPAMAYO_NUM_LAYERS")
    if value is None or value == "":
        return None
    layers = int(value)
    if layers < 1:
        raise ValueError(f"ALPAMAYO_NUM_LAYERS must be >= 1, got {layers}")
    return layers


# The snapshot every measured number in README.md was taken at, so a new upload on the hub cannot silently move them.
# A commit hash: a branch or tag would follow the hub.
MODEL_REVISION = "b92b6aa326bd3e2235db28f248ad5a30b5890e64"


def resolve_model_dir() -> Path:
    """The `nvidia/Alpamayo2-Edge` snapshot at `MODEL_REVISION`, or at `ALPAMAYO_REVISION` when that is set."""
    from huggingface_hub import snapshot_download

    revision = os.environ.get("ALPAMAYO_REVISION") or MODEL_REVISION
    return Path(snapshot_download("nvidia/Alpamayo2-Edge", revision=revision))


def _text_config(raw: dict, layers: int) -> TextConfig:
    return TextConfig(
        hidden=raw["hidden_size"],
        layers=layers,
        mlp=raw["intermediate_size"],
        heads=raw["num_attention_heads"],
        kv_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        eps=raw["rms_norm_eps"],
        vocab=raw["vocab_size"],
        rope_theta=float(raw["rope_parameters"]["rope_theta"]),
        mrope_section=tuple(raw["rope_parameters"]["mrope_section"]),
    )


def _action_config(raw: dict) -> ActionConfig:
    expert = raw["expert_config"]
    in_proj = expert["action_in_proj_cfg"]
    space = {k: v for k, v in expert["action_space_cfg"].items() if k != "_target_"}
    space = {k: tuple(v) if isinstance(v, list) else v for k, v in space.items()}
    known = {f.name for f in fields(ActionConfig)}
    unknown = set(space) - known
    if unknown:
        raise ValueError(f"unhandled action_space_cfg fields: {sorted(unknown)}")
    return ActionConfig(
        fourier_feats=in_proj["num_fourier_feats"],
        max_freq=in_proj["max_freq"],
        in_proj_hidden=in_proj["hidden_size"],
        in_proj_layers=in_proj["num_enc_layers"],
        int_method=expert["diffusion_cfg"]["int_method"],
        inference_guidance_weight=expert["diffusion_cfg"]["inference_guidance_weight"],
        use_classifier_free_guidance=expert["diffusion_cfg"]["use_classifier_free_guidance"],
        # Upstream's default when the checkpoint leaves it out.
        expert_non_causal_attention=expert.get("expert_non_causal_attention", True),
        **space,
    )


def validate_supported(cfg: Alpamayo2EdgeConfig) -> None:
    """Rejects checkpoint settings that neither the reference nor the TT path implements.

    Raises:
        ValueError: The flow is guided or not Euler, the expert attends causally, the vision MLP is not tanh GELU,
            or the expert cannot read the LM's KV cache layer for layer.
    """
    acfg = cfg.action
    if acfg.use_classifier_free_guidance or acfg.int_method != "euler":
        raise ValueError("Alpamayo2Edge inference is unguided Euler flow matching")
    if not acfg.expert_non_causal_attention:
        raise ValueError("only the non-causal action expert is implemented")
    if cfg.vision.hidden_act != "gelu_pytorch_tanh":
        raise ValueError(
            f"only the tanh-approximated GELU is implemented in the vision MLP, got {cfg.vision.hidden_act}"
        )
    for name in ("layers", "kv_heads", "head_dim"):
        if getattr(cfg.text, name) != getattr(cfg.expert, name):
            raise ValueError(f"LM and expert must share KV-cache geometry ({name})")


def load_config(model_dir: Path) -> Alpamayo2EdgeConfig:
    """The checkpoint's config with the `ALPAMAYO_NUM_LAYERS` override, checked by `validate_supported`."""
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        raw = json.load(f)

    vlm = raw["vlm_config"]
    vis = vlm["vision_config"]
    llm = vlm["text_config"]
    expert = raw["expert_config"]
    expert_dims = expert["expert_update_cfg"]
    ids = raw["traj_ids"]

    override = num_layers_override()
    pick = lambda full: full if override is None else override
    vision_layers = pick(vis["num_hidden_layers"])
    text_layers = pick(llm["num_hidden_layers"])
    expert_layers = pick(expert_dims["num_hidden_layers"])

    # The expert's llm_config is overridden by expert_update_cfg for the fields it lists.
    expert_raw = {**expert["llm_config"], **{k: v for k, v in expert_dims.items() if k != "num_hidden_layers"}}

    cfg = Alpamayo2EdgeConfig(
        vision=VisionConfig(
            hidden=vis["hidden_size"],
            layers=vision_layers,
            heads=vis["num_attention_heads"],
            mlp=vis["intermediate_size"],
            patch=vis["patch_size"],
            merge=vis["spatial_merge_size"],
            pos_grid=int(vis["num_patches"] ** 0.5),
            eps=vis["layer_norm_eps"],
            hidden_act=vis["hidden_act"],
        ),
        projector=ProjectorConfig(hidden=vlm["projector_hidden_size"], out=llm["hidden_size"]),
        text=_text_config(llm, text_layers),
        expert=_text_config(expert_raw, expert_layers),
        action=_action_config(raw),
        tokens=TokenIds(
            future_start=ids["future_start"],
            future_end=ids["future_end"],
            future_pad=ids["future_pad"],
            future_id0=ids["future_id0"],
            history_start=ids["history_start"],
            history_end=ids["history_end"],
            history_pad=ids["history_pad"],
            history_id0=ids["history_id0"],
            history_vocab=raw["history_vocab_size"],
            future_vocab=raw["future_vocab_size"],
            image=vlm["image_token_id"],
            video=vlm["video_token_id"],
            vision_start=vlm["vision_start_token_id"],
            vision_end=vlm["vision_end_token_id"],
            eos=llm["eos_token_id"],
            pad=llm["pad_token_id"],
        ),
        model_dir=model_dir,
    )
    validate_supported(cfg)
    return cfg
