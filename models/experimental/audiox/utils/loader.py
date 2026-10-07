# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pretrained-checkpoint loader for the AudioX (HKUSTAudio/AudioX) bringup.

Maps the published checkpoint's state_dict layout onto our reference modules.
Two structural differences require a remap:

1. **Outer wrapper prefixes.** Upstream wraps the DiT, conditioners, and
   autoencoder under a single ``ConditionedDiffusionModelWrapper`` so the
   checkpoint paths look like ``model.transformer...``,
   ``conditioner.conditioners.text_prompt...``,
   ``pretransform.model.decoder.layers.<i>...``. We load each piece into a
   standalone module, so we strip the prefix.

2. **Sequential vs named children for Oobleck.** Upstream packs each block in
   ``nn.Sequential`` (``layers.0``, ``layers.1``, …); we use named attributes
   (``in_conv``, ``blocks.0.upsample``, ``out_conv``, …) so the decoder code
   reads cleanly. The remap walks the upstream indexed structure and emits
   our names.

DiT, conditioners (T5/CLIP/AudioAutoencoder), and the SA temporal
transformer all use names that already match upstream verbatim — no
position-by-index remapping needed for those, just prefix stripping.
"""

import os
import typing as tp

import torch


def _strip_prefix(state_dict: tp.Mapping[str, torch.Tensor], prefix: str) -> tp.Dict[str, torch.Tensor]:
    """Return a new dict with ``prefix`` removed from keys; drop entries that don't start with it."""
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _resolve_oobleck_decoder_block_key(upstream_key: str, n_blocks: int) -> tp.Optional[str]:
    """Map a single ``layers.<i>...`` upstream key to our named layout.

    Upstream OobleckDecoder.layers (Sequential) layout:
        layers.0                    -> in_conv (WNConv1d)
        layers.1..n_blocks          -> DecoderBlock j = i-1
            layers.<i>.layers.0     -> blocks.j.act (snake)
            layers.<i>.layers.1     -> blocks.j.upsample (WNConvTranspose1d)
            layers.<i>.layers.2..4  -> blocks.j.res{1,2,3} (ResidualUnit)
                .layers.0           -> act1 (snake)
                .layers.1           -> conv1 (WNConv1d, k=7, dilated)
                .layers.2           -> act2 (snake)
                .layers.3           -> conv2 (WNConv1d, k=1)
        layers.<n+1>                -> out_act (snake)
        layers.<n+2>                -> out_conv (WNConv1d, no bias)
        layers.<n+3>                -> Tanh / Identity (no params; skip)
    """
    if not upstream_key.startswith("layers."):
        return None

    parts = upstream_key.split(".")
    idx = int(parts[1])
    rest = ".".join(parts[2:])

    if idx == 0:
        return f"in_conv.{rest}"
    if idx == n_blocks + 1:
        return f"out_act.{rest}"
    if idx == n_blocks + 2:
        return f"out_conv.{rest}"
    if not (1 <= idx <= n_blocks):
        return None  # tanh/identity tail or unexpected

    # DecoderBlock j = idx - 1; ``rest`` starts with "layers.<j>."
    block_idx = idx - 1
    if not rest.startswith("layers."):
        return None
    block_parts = rest.split(".")
    j = int(block_parts[1])
    block_rest = ".".join(block_parts[2:])

    if j == 0:
        return f"blocks.{block_idx}.act.{block_rest}"
    if j == 1:
        return f"blocks.{block_idx}.upsample.{block_rest}"
    if not (2 <= j <= 4):
        return None

    res_idx = j - 1  # 1, 2, 3
    # ResidualUnit: ``block_rest`` starts with "layers.<k>."
    if not block_rest.startswith("layers."):
        return None
    res_parts = block_rest.split(".")
    k = int(res_parts[1])
    res_rest = ".".join(res_parts[2:])

    inner_names = {0: "act1", 1: "conv1", 2: "act2", 3: "conv2"}
    inner = inner_names.get(k)
    if inner is None:
        return None

    return f"blocks.{block_idx}.res{res_idx}.{inner}.{res_rest}"


def _resolve_oobleck_encoder_block_key(upstream_key: str, n_blocks: int) -> tp.Optional[str]:
    if not upstream_key.startswith("layers."):
        return None

    parts = upstream_key.split(".")
    idx = int(parts[1])
    rest = ".".join(parts[2:])

    if idx == 0:
        return f"in_conv.{rest}"
    if idx == n_blocks + 1:
        return f"out_act.{rest}"
    if idx == n_blocks + 2:
        return f"out_conv.{rest}"
    if not (1 <= idx <= n_blocks):
        return None

    block_idx = idx - 1
    if not rest.startswith("layers."):
        return None
    block_parts = rest.split(".")
    j = int(block_parts[1])
    block_rest = ".".join(block_parts[2:])

    if 0 <= j <= 2:
        res_idx = j + 1
        if not block_rest.startswith("layers."):
            return None
        res_parts = block_rest.split(".")
        k = int(res_parts[1])
        res_rest = ".".join(res_parts[2:])
        inner_names = {0: "act1", 1: "conv1", 2: "act2", 3: "conv2"}
        inner = inner_names.get(k)
        if inner is None:
            return None
        return f"blocks.{block_idx}.res{res_idx}.{inner}.{res_rest}"

    if j == 3:
        return f"blocks.{block_idx}.act.{block_rest}"
    if j == 4:
        return f"blocks.{block_idx}.downsample.{block_rest}"
    return None


def remap_oobleck_decoder_state_dict(
    state_dict: tp.Mapping[str, torch.Tensor],
    prefix: str = "pretransform.model.decoder.",
    n_blocks: int = 5,
) -> tp.Dict[str, torch.Tensor]:
    """Remap upstream OobleckDecoder keys to our named-children layout.

    ``n_blocks`` is the number of DecoderBlocks (= ``len(c_mults)`` as passed
    in to OobleckDecoder; upstream prepends a ``1`` internally but our
    n_blocks counts the user-facing DecoderBlock count). For the AudioX HF
    config that's ``5``."""
    sub = _strip_prefix(state_dict, prefix)
    out: tp.Dict[str, torch.Tensor] = {}
    for k, v in sub.items():
        new_key = _resolve_oobleck_decoder_block_key(k, n_blocks)
        if new_key is not None:
            out[new_key] = v
    return out


def remap_oobleck_encoder_state_dict(
    state_dict: tp.Mapping[str, torch.Tensor],
    prefix: str = "pretransform.model.encoder.",
    n_blocks: int = 5,
) -> tp.Dict[str, torch.Tensor]:
    sub = _strip_prefix(state_dict, prefix)
    out: tp.Dict[str, torch.Tensor] = {}
    for k, v in sub.items():
        new_key = _resolve_oobleck_encoder_block_key(k, n_blocks)
        if new_key is not None:
            out[new_key] = v
    return out


def remap_dit_state_dict(
    state_dict: tp.Mapping[str, torch.Tensor],
    prefix: str = "model.model.",
) -> tp.Dict[str, torch.Tensor]:
    """Strip the wrapper prefixes off the DiT keys.

    Upstream nests the DiT under two ``self.model`` levels — the outer
    ``ConditionedDiffusionModelWrapper.model`` is a ``DiTWrapper`` and that
    wrapper holds ``self.model = DiffusionTransformer(...)``. So checkpoint
    paths look like ``model.model.transformer.layers.<i>...``.

    Our reference uses the same names as upstream's DiffusionTransformer
    (``timestep_features``, ``to_timestep_embed.{0,2}``, ``to_cond_embed``,
    ``transformer.layers.<i>.{pre_norm, self_attn, cross_attn, ...}``,
    ``preprocess_conv``, ``postprocess_conv``) so the remap is just a strip."""
    return _strip_prefix(state_dict, prefix)


def remap_conditioner_state_dict(
    state_dict: tp.Mapping[str, torch.Tensor],
    conditioner_id: str,
    prefix: str = "conditioner.conditioners.",
) -> tp.Dict[str, torch.Tensor]:
    """Pull a single conditioner's keys out of the multi-conditioner block.

    Our T5/CLIP/AudioAutoencoder conditioners use the same parameter names as
    upstream (``proj_out``, ``empty_visual_feat``, ``Temp_pos_embedding``,
    ``Temp_transformer.blocks.<i>...``, etc.) so this is also a prefix strip."""
    return _strip_prefix(state_dict, prefix + conditioner_id + ".")


def load_audiox_checkpoint(path: tp.Union[str, os.PathLike]) -> tp.Dict[str, torch.Tensor]:
    """Load a HKUSTAudio/AudioX checkpoint into a flat state_dict.

    Supports both ``.safetensors`` (preferred) and PyTorch ``.pt``/``.bin``
    files. Strips a top-level ``state_dict`` wrapper if present. Returns the
    raw upstream state dict; pair with the ``remap_*`` helpers above to load
    individual modules."""
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)

    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    return obj


def load_into(
    module: torch.nn.Module,
    state_dict: tp.Mapping[str, torch.Tensor],
    *,
    label: str = "module",
    strict: bool = False,
) -> tp.Tuple[tp.List[str], tp.List[str]]:
    """Wrap ``load_state_dict`` so the caller gets a clear missing/unexpected
    report per module — easier to debug remap mistakes than the default
    behavior (raise on any miss)."""
    incompatible = module.load_state_dict(state_dict, strict=strict)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        print(f"[audiox loader] {label}: " f"{len(missing)} missing, {len(unexpected)} unexpected")
    return missing, unexpected
