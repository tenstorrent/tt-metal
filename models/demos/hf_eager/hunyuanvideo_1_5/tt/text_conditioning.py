# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prompt pre-encoding and persistent embedding cache for HunyuanVideo-1.5.

Host Qwen conditioning is measured at ~12.2 s per generation on the reference
64-thread EPYC host (two 1108-token Qwen2.5-VL forwards, one positive and one
negative), against a ~0.01 s cache hit. byT5 contributes ~0.1 s and only when
the prompt carries quoted glyph text; without quotes the diffusers pipeline
emits a zero tensor and never calls the encoder at all.

The cache is opt-in through ``HY_PROMPT_CACHE=1``. Because a stale or
mis-attributed hit would silently feed the DiT conditioning for a different
prompt, the key covers every input that can move an embedding, and the artifact
carries its own descriptor so a hash collision or a hand-edited file fails
loudly instead of being consumed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

_MEMORY_CACHE: dict[str, dict[str, torch.Tensor]] = {}

CACHE_SCHEMA = 2

POSITIVE_NAMES = ("prompt_embeds", "prompt_embeds_mask", "prompt_embeds_2", "prompt_embeds_mask_2")
NEGATIVE_NAMES = (
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
    "negative_prompt_embeds_2",
    "negative_prompt_embeds_mask_2",
)

# Environment that can move a conditioning tensor. Placement flags change which
# encoder implementation runs; the zero-pad flags change the padded positions the
# maskless fused joint-attention kernel later consumes; the CFG policy is keyed
# defensively so a future coupling between policy and encode cannot go unnoticed.
_KEYED_ENV = (
    "HY_TT_QWEN",
    "HY_TT_QWEN_SHARED",
    "HY_TT_BYT5",
    "HY_QWEN_ZERO_PAD",
    "HY_BYT5_ZERO_PAD",
    "HY_CFG_PADDING_POLICY",
    "HY_MESH",
)


class PromptCacheError(RuntimeError):
    """A persisted artifact does not match the key it was found under."""


def _model_id(model) -> str:
    config = getattr(model, "config", None)
    name = str(getattr(config, "_name_or_path", "") or "unknown")
    return f"{type(model).__qualname__}:{name}"


def _tokenizer_id(tokenizer) -> str:
    name = str(getattr(tokenizer, "name_or_path", "") or "unknown")
    vocab = getattr(tokenizer, "vocab_size", None)
    return f"{type(tokenizer).__qualname__}:{name}:{vocab}"


def _dtype_id(module) -> str:
    return str(getattr(module, "dtype", None))


def _placement(model) -> dict:
    """Where this encoder ran, and under which parallel decomposition.

    Both TT adapters keep their mesh in ``self.__dict__["_device"]`` and proxy
    every other attribute to the wrapped host encoder, so reading ``__dict__``
    directly distinguishes host from device without tripping that proxy. Only
    the mesh *shape* is keyed: tensor-parallel fracture follows the shape, while
    the concrete chip ids do not change any numeric result.
    """
    state = getattr(model, "__dict__", {})
    device = state.get("_device")
    if device is None:
        return {"where": "host"}
    shape = getattr(device, "shape", None)
    entry = {"where": "device", "mesh_shape": list(shape) if shape is not None else None}
    if "_zero_padding" in state:
        entry["zero_padding"] = bool(state["_zero_padding"])
    return entry


def _library_versions() -> dict:
    import diffusers
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "diffusers": diffusers.__version__,
    }


def _guidance_enabled(guider) -> bool:
    return bool(getattr(guider, "_enabled", True))


def cache_descriptor(pipe, prompt, negative_prompt) -> dict:
    """Every input that can change the returned conditioning tensors.

    ``encode_prompt`` casts the positive tuple with ``self.text_encoder.dtype``
    (the caller omits ``dtype=``) and the negative tuple with the transformer
    dtype, so both are recorded; they are equal in the production configuration
    but the key must not assume it.
    """
    return {
        "schema": CACHE_SCHEMA,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "pipeline": type(pipe).__qualname__,
        "qwen": _model_id(pipe.text_encoder),
        "byt5": _model_id(pipe.text_encoder_2),
        "tokenizer": _tokenizer_id(pipe.tokenizer),
        "tokenizer_2": _tokenizer_id(pipe.tokenizer_2),
        "qwen_max_length": int(pipe.tokenizer_max_length),
        "byt5_max_length": int(pipe.tokenizer_2_max_length),
        "prompt_template": str(pipe.system_message),
        "crop_start": int(pipe.prompt_template_encode_start_idx),
        "transformer_dtype": _dtype_id(pipe.transformer),
        "text_encoder_dtype": _dtype_id(pipe.text_encoder),
        "text_encoder_2_dtype": _dtype_id(pipe.text_encoder_2),
        "conditions": int(pipe.guider.num_conditions),
        "guidance_enabled": _guidance_enabled(pipe.guider),
        "placement": {"qwen": _placement(pipe.text_encoder), "byt5": _placement(pipe.text_encoder_2)},
        "env": {name: os.environ.get(name) for name in _KEYED_ENV},
        "libraries": _library_versions(),
    }


def _serialize(descriptor: dict) -> str:
    return json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _cache_path(key: str, cache_dir: str | os.PathLike | None) -> Path:
    root = Path(
        cache_dir or os.environ.get("HY_PROMPT_CACHE_DIR") or os.environ.get("TT_DIT_CACHE_DIR", "~/.cache/tt-dit")
    )
    return root.expanduser() / "hunyuanvideo-1.5-embeddings" / f"{key}.pt"


def conditioning_fingerprint(values: dict[str, torch.Tensor]) -> str:
    """A byte-exact digest of a conditioning tuple, for equality assertions."""
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _expected_names(descriptor: dict) -> tuple[str, ...]:
    if descriptor["guidance_enabled"] and descriptor["conditions"] > 1:
        return POSITIVE_NAMES + NEGATIVE_NAMES
    return POSITIVE_NAMES


def _validate(artifact, payload: str, path: Path) -> dict[str, torch.Tensor]:
    """Accept a persisted artifact only if it names the key it was stored under."""
    if not isinstance(artifact, dict) or "descriptor" not in artifact or "tensors" not in artifact:
        raise PromptCacheError(f"prompt cache artifact {path} predates schema {CACHE_SCHEMA}; delete it and re-encode")
    if artifact["descriptor"] != payload:
        raise PromptCacheError(f"prompt cache artifact {path} was written for a different conditioning descriptor")
    values = artifact["tensors"]
    expected = _expected_names(json.loads(payload))
    if tuple(sorted(values)) != tuple(sorted(expected)):
        raise PromptCacheError(f"prompt cache artifact {path} holds {sorted(values)}, expected {sorted(expected)}")
    return values


def _encode(pipe, prompt, negative_prompt, descriptor: dict) -> dict[str, torch.Tensor]:
    batch_size = len(prompt) if isinstance(prompt, list) else 1
    positive = pipe.encode_prompt(
        prompt=prompt,
        device=pipe._execution_device,
        batch_size=batch_size,
        num_videos_per_prompt=1,
    )
    values = dict(zip(POSITIVE_NAMES, positive))
    if descriptor["guidance_enabled"] and descriptor["conditions"] > 1:
        negative = pipe.encode_prompt(
            prompt=negative_prompt,
            device=pipe._execution_device,
            dtype=pipe.transformer.dtype,
            batch_size=batch_size,
            num_videos_per_prompt=1,
        )
        values.update(zip(NEGATIVE_NAMES, negative))
    return {name: tensor.detach().cpu().contiguous() for name, tensor in values.items()}


@torch.no_grad()
def encode_prompt_pair(
    pipe,
    prompt: str | list[str],
    negative_prompt: str | list[str] | None,
    *,
    use_cache: bool = True,
    cache_dir: str | os.PathLike | None = None,
    verify: bool | None = None,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Encode positive/negative Qwen+byT5 conditioning before loading the DiT.

    Returns kwargs accepted by the diffusers pipeline and a cache-hit flag.
    Cached tensors are CPU tensors, so the artifact is independent of TT mesh
    addresses and can be reused by warm serving processes.

    ``verify`` (or ``HY_PROMPT_CACHE_VERIFY=1``) re-encodes on a hit and fails
    unless the result is byte-identical to the cached tuple. It costs a full
    encode, so it is a validation gate rather than a serving mode.
    """
    descriptor = cache_descriptor(pipe, prompt, negative_prompt)
    payload = _serialize(descriptor)
    key = hashlib.sha256(payload.encode()).hexdigest()
    path = _cache_path(key, cache_dir)
    if verify is None:
        verify = os.environ.get("HY_PROMPT_CACHE_VERIFY", "0") == "1"

    values = None
    if use_cache and key in _MEMORY_CACHE:
        values = _MEMORY_CACHE[key]
    elif use_cache and path.is_file():
        values = _validate(torch.load(path, map_location="cpu", weights_only=True), payload, path)
        _MEMORY_CACHE[key] = values

    if values is not None:
        if verify:
            fresh = _encode(pipe, prompt, negative_prompt, descriptor)
            if conditioning_fingerprint(fresh) != conditioning_fingerprint(values):
                raise PromptCacheError(
                    f"prompt cache hit at {path} is not byte-identical to a fresh encode of the same key"
                )
        return {name: tensor.clone() for name, tensor in values.items()}, True

    values = _encode(pipe, prompt, negative_prompt, descriptor)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"descriptor": payload, "tensors": values}, temporary)
        os.replace(temporary, path)
        _MEMORY_CACHE[key] = values
    return {name: tensor.clone() for name, tensor in values.items()}, False
