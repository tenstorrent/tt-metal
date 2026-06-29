# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint I/O helpers for DiffusionGemma device generation (#47464).

The text-only TT generation path consumes the DiffusionGemma decoder weights:
the Gemma4-compatible backbone plus the decoder self-conditioning module. It
does not need the encoder/vision trees for the text-first prompt->text bring-up,
so this module loads only the required safetensors keys by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

from safetensors.torch import safe_open

from models.experimental.diffusion_gemma.weight_mapping import DG_DECODER_PREFIX

TEXT_GENERATION_PREFIXES = (DG_DECODER_PREFIX,)


class CheckpointInputs(NamedTuple):
    """Host objects needed by ``tt.generate.generate_text_from_checkpoint_state``."""

    tokenizer: object
    state_dict: dict


def _as_prefix_tuple(prefixes: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(prefixes, str):
        return (prefixes,)
    return tuple(prefixes)


def load_text_generation_state_dict(
    checkpoint_dir: str | Path,
    *,
    prefixes: tuple[str, ...] | list[str] | str = TEXT_GENERATION_PREFIXES,
    device: str = "cpu",
) -> dict:
    """Load the raw DiffusionGemma text-generation weights from a HF checkpoint.

    ``generate_text_from_checkpoint_state`` expects raw DiffusionGemma key names
    so its logits builder can remap the decoder backbone and self-conditioning
    weights. By default this helper loads only ``model.decoder.*`` and skips
    encoder / vision / multimodal weights.
    """

    checkpoint_dir = Path(checkpoint_dir)
    prefixes = _as_prefix_tuple(prefixes)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    state_dict = {}

    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        file_to_keys: dict[str, list[str]] = {}
        for key, filename in weight_map.items():
            if key.startswith(prefixes):
                file_to_keys.setdefault(filename, []).append(key)
        if not file_to_keys:
            raise ValueError(f"checkpoint has no weights matching prefixes {prefixes}")
        for filename, keys in sorted(file_to_keys.items()):
            shard_path = checkpoint_dir / filename
            if not shard_path.exists():
                raise FileNotFoundError(f"checkpoint shard not found: {shard_path}")
            with safe_open(shard_path, framework="pt", device=device) as f:
                for key in keys:
                    state_dict[key] = f.get_tensor(key)
        return state_dict

    safetensors_path = checkpoint_dir / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Neither model.safetensors.index.json nor model.safetensors found in {checkpoint_dir}")
    with safe_open(safetensors_path, framework="pt", device=device) as f:
        for key in f.keys():
            if key.startswith(prefixes):
                state_dict[key] = f.get_tensor(key)
    if not state_dict:
        raise ValueError(f"checkpoint has no weights matching prefixes {prefixes}")
    return state_dict


def load_tokenizer(checkpoint_dir: str | Path, **tokenizer_kwargs):
    """Load the HuggingFace tokenizer for a DiffusionGemma checkpoint."""

    from transformers import AutoTokenizer

    kwargs = {"trust_remote_code": True}
    kwargs.update(tokenizer_kwargs)
    return AutoTokenizer.from_pretrained(checkpoint_dir, **kwargs)


def load_checkpoint_inputs(
    checkpoint_dir: str | Path,
    *,
    tokenizer_kwargs: dict | None = None,
    state_prefixes: tuple[str, ...] | list[str] | str = TEXT_GENERATION_PREFIXES,
    device: str = "cpu",
) -> CheckpointInputs:
    """Load tokenizer + text-generation state for the prompt-to-text entrypoint."""

    tokenizer = load_tokenizer(checkpoint_dir, **(tokenizer_kwargs or {}))
    state_dict = load_text_generation_state_dict(checkpoint_dir, prefixes=state_prefixes, device=device)
    return CheckpointInputs(tokenizer=tokenizer, state_dict=state_dict)


def generate_text_from_checkpoint_dir(
    tt_model,
    checkpoint_dir: str | Path,
    prompt,
    *,
    tokenizer_kwargs: dict | None = None,
    state_prefixes: tuple[str, ...] | list[str] | str = TEXT_GENERATION_PREFIXES,
    state_device: str = "cpu",
    checkpoint_loader=load_checkpoint_inputs,
    generate_fn=None,
    **generate_kwargs,
):
    """Load a HF checkpoint directory and run the TT prompt-to-text entrypoint.

    This is the thin runnable glue for #47464: callers provide an already-built
    TT Gemma4/DiffusionGemma model and a checkpoint directory; this helper loads
    the host tokenizer + raw decoder state and delegates to
    ``generate_text_from_checkpoint_state``.
    """

    inputs = checkpoint_loader(
        checkpoint_dir,
        tokenizer_kwargs=tokenizer_kwargs,
        state_prefixes=state_prefixes,
        device=state_device,
    )
    if generate_fn is None:
        from models.experimental.diffusion_gemma.tt.generate import generate_text_from_checkpoint_state

        generate_fn = generate_text_from_checkpoint_state
    return generate_fn(
        tt_model,
        inputs.tokenizer,
        prompt,
        dg_state_dict=inputs.state_dict,
        **generate_kwargs,
    )
