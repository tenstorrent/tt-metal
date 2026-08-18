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
from numbers import Integral
from pathlib import Path
from typing import NamedTuple

import torch
from loguru import logger
from safetensors.torch import safe_open

from models.experimental.diffusion_gemma.weight_mapping import (
    DG_DECODER_PREFIX,
    check_encoder_layer_scalar_tie,
    decoder_layer_scalar_key,
    encoder_layer_scalar_key,
)

TEXT_GENERATION_PREFIXES = (DG_DECODER_PREFIX,)
STATE_SNAPSHOT_ALLOW_PATTERNS = ("model.safetensors", "model.safetensors.index.json", "*.safetensors")


class CheckpointInputs(NamedTuple):
    """Host objects needed by ``tt.generate.generate_text_from_checkpoint_state``."""

    tokenizer: object
    state_dict: dict


class CheckpointModelInputs(NamedTuple):
    """Loaded checkpoint inputs plus a Gemma4-backed TT model."""

    tokenizer: object
    state_dict: dict
    model_args: object
    tt_model: object
    tt_kv_cache: object


def default_backbone_config_dir() -> Path:
    """Return the in-repo Gemma4 26B-A4B config used by the TT backbone."""

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "models/demos/gemma4/configs/gemma-4-26B-A4B-it"


def _as_prefix_tuple(prefixes: tuple[str, ...] | list[str] | str) -> tuple[str, ...]:
    if isinstance(prefixes, str):
        return (prefixes,)
    return tuple(prefixes)


def _validate_num_layers(num_layers: int | None) -> int | None:
    if num_layers is None:
        return None
    if isinstance(num_layers, bool) or not isinstance(num_layers, Integral):
        raise ValueError("num_layers must be an integer")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    return int(num_layers)


def text_generation_prefixes_for_layers(num_layers: int | None = None) -> tuple[str, ...]:
    """Return raw DiffusionGemma prefixes needed for text generation.

    ``num_layers=None`` means the full decoder. Passing an integer creates a
    smoke-load prefix set for the first N decoder layers while keeping shared
    weights required by embedding, final norm, lm-head tying, and self-conditioning.
    """

    num_layers = _validate_num_layers(num_layers)
    if num_layers is None:
        return TEXT_GENERATION_PREFIXES
    return (
        f"{DG_DECODER_PREFIX}embed_tokens.",
        f"{DG_DECODER_PREFIX}norm.",
        f"{DG_DECODER_PREFIX}self_conditioning.",
        *(f"{DG_DECODER_PREFIX}layers.{layer_idx}." for layer_idx in range(num_layers)),
    )


def resolve_checkpoint_dir(
    checkpoint_dir: str | Path,
    *,
    local_files_only: bool | None = None,
    allow_patterns: tuple[str, ...] | list[str] | str | None = STATE_SNAPSHOT_ALLOW_PATTERNS,
) -> Path:
    """Return a local checkpoint directory for either a path or HF model id."""

    path = Path(checkpoint_dir)
    if path.exists():
        return path

    if path.is_absolute():
        raise FileNotFoundError(f"checkpoint directory not found: {path}")

    kwargs = {}
    if local_files_only is not None:
        kwargs["local_files_only"] = local_files_only
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(_as_prefix_tuple(allow_patterns))
    return Path(_snapshot_download_with_retry(str(checkpoint_dir), **kwargs))


def _snapshot_download_with_retry(repo_id: str, *, attempts: int = 5, **kwargs):
    """``snapshot_download`` with backoff, because one transient 5xx must not cost an hour.

    This is the FIRST thing that runs when a server starts on a host with a cold weight cache, and
    it raises straight out of ``initialize_vllm_model`` -> ``load_model`` -> ``_init_executor``,
    where an exception is fatal to the vLLM EngineCore -- so a single transient CDN error partway
    through a ~50 GB fetch would kill the engine before any DiffusionGemma code has run.

    Retries are on transport/server errors only. A missing repo, a gated repo or bad credentials fail
    immediately: retrying those just burns the same hour more slowly.
    """
    import time

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

    fatal = (RepositoryNotFoundError, GatedRepoError, EntryNotFoundError, FileNotFoundError)
    delay = 5.0
    for attempt in range(1, attempts + 1):
        try:
            return snapshot_download(repo_id, **kwargs)
        except fatal:
            raise
        except Exception as exc:  # noqa: BLE001 - transport errors are not one exception type
            if attempt == attempts:
                raise RuntimeError(
                    f"checkpoint download for {repo_id!r} failed after {attempts} attempts; "
                    f"last error: {type(exc).__name__}: {exc}"
                ) from exc
            logger.warning(
                f"[DiffusionGemma] checkpoint download attempt {attempt}/{attempts} failed "
                f"({type(exc).__name__}: {exc}); retrying in {delay:.0f}s. Already-fetched files are "
                f"kept, so a retry resumes rather than restarts."
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def load_text_generation_state_dict(
    checkpoint_dir: str | Path,
    *,
    prefixes: tuple[str, ...] | list[str] | str = TEXT_GENERATION_PREFIXES,
    device: str = "cpu",
    local_files_only: bool | None = None,
) -> dict:
    """Load the raw DiffusionGemma text-generation weights from a HF checkpoint.

    ``generate_text_from_checkpoint_state`` expects raw DiffusionGemma key names
    so its logits builder can remap the decoder backbone and self-conditioning
    weights. By default this helper loads only ``model.decoder.*`` and skips
    encoder / vision / multimodal weights.
    """

    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir, local_files_only=local_files_only)
    validate_encoder_layer_scalar_tie(checkpoint_dir, device=device, local_files_only=local_files_only)
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


def validate_encoder_layer_scalar_tie(
    checkpoint_dir: str | Path,
    *,
    device: str = "cpu",
    local_files_only: bool | None = None,
) -> int:
    """Fail loud when the encoder and decoder ``layer_scalar`` copies diverge.

    The loader drops the whole ``model.encoder.*`` tree, which is only sound
    because the conversion script clones ``layer_scalar`` rather than training a
    separate encoder copy. ``layer_scalar`` multiplies the entire layer output, so
    a divergent checkpoint would put a compounding per-layer error into every
    prefill and commit KV write while the decoder-only denoise pass stayed
    correct — a failure mode a precision sweep cannot move. Returns the number of
    layers checked (0 when the checkpoint carries no encoder copy at all).
    """

    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir, local_files_only=local_files_only)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"no safetensors checkpoint found in {checkpoint_dir}")
        with safe_open(safetensors_path, framework="pt", device=device) as f:
            weight_map = {key: "model.safetensors" for key in f.keys()}

    layers = sorted(
        layer
        for layer in range(len(weight_map))
        if encoder_layer_scalar_key(layer) in weight_map and decoder_layer_scalar_key(layer) in weight_map
    )
    if not layers:
        return 0

    handles: dict[str, object] = {}
    try:

        def get_tensor(key: str):
            filename = weight_map.get(key)
            if filename is None:
                return None
            if filename not in handles:
                handles[filename] = safe_open(checkpoint_dir / filename, framework="pt", device=device)
            return handles[filename].get_tensor(key)

        divergent = check_encoder_layer_scalar_tie(get_tensor, layers)
    finally:
        for handle in handles.values():
            close = getattr(handle, "__exit__", None)
            if close is not None:
                close(None, None, None)

    if divergent:
        detail = ", ".join(f"layer {layer}: max|diff|={diff:g}" for layer, diff in divergent[:8])
        raise ValueError(
            f"{len(divergent)} of {len(layers)} DiffusionGemma layers have an encoder layer_scalar that "
            f"differs from the decoder copy ({detail}). This loader drops model.encoder.* and applies the "
            "decoder scalar on the prefill and commit passes, which would be wrong for this checkpoint. "
            "Load and apply the encoder scalar on the causal passes before using it."
        )
    return len(layers)


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

    tokenizer_kwargs = dict(tokenizer_kwargs or {})
    tokenizer = load_tokenizer(checkpoint_dir, **tokenizer_kwargs)
    state_dict = load_text_generation_state_dict(
        checkpoint_dir,
        prefixes=state_prefixes,
        device=device,
        local_files_only=tokenizer_kwargs.get("local_files_only"),
    )
    return CheckpointInputs(tokenizer=tokenizer, state_dict=state_dict)


def build_tt_model_from_checkpoint_inputs(
    mesh_device,
    checkpoint_inputs: CheckpointInputs,
    *,
    backbone_config_dir: str | Path | None = None,
    remap_fn=None,
    create_model_fn=None,
    **model_kwargs,
) -> CheckpointModelInputs:
    """Build the reused Gemma4 TT backbone from loaded DiffusionGemma inputs."""

    if "num_layers" in model_kwargs:
        model_kwargs = dict(model_kwargs)
        model_kwargs["num_layers"] = _validate_num_layers(model_kwargs["num_layers"])

    # The served KV layout is the ONLY build layout: bounded sliding windows, full
    # span on the global-attention layers. A caller asking for a KV cache without its
    # own paged config gets exactly what serving runs (this is also what lets a 256K
    # max_seq_len fit at all — a contiguous 30-layer full-length cache does not).
    # Callers that pass their own paged_attention_config keep full control; the
    # low-level contiguous writers survive only as unit-test internals.
    attach_hybrid_kv = False
    if (
        model_kwargs.get("create_kv_cache", True)  # the backbone default builds a KV cache
        and "paged_attention_config" not in model_kwargs  # caller owns its layout
        and model_kwargs.get("max_seq_len")
        and int(model_kwargs.get("max_batch_size", 1)) == 1
    ):
        from models.experimental.diffusion_gemma.tt.hybrid_kv import (
            model_owned_hybrid_kv_model_kwargs,
        )

        model_kwargs = dict(model_kwargs)
        # The hybrid layout supersedes the legacy bounded-sliding knob.
        model_kwargs.pop("bounded_sliding_kv_cache", None)
        model_kwargs.update(model_owned_hybrid_kv_model_kwargs(max_seq_len=int(model_kwargs["max_seq_len"])))
        attach_hybrid_kv = True

    if remap_fn is None:
        from models.experimental.diffusion_gemma.weight_mapping import remap_state_dict

        remap_fn = remap_state_dict
    if create_model_fn is None:
        # DG-local builder honours the experts-dtype knob (DG_EXPERTS_BFP8 / DG_EXPERTS_DTYPE);
        # with no knob set it delegates to the shared create_tt_model unchanged (#47475).
        from models.experimental.diffusion_gemma.tt.precision_build import create_tt_model_dg

        create_model_fn = create_tt_model_dg

    backbone_state, _self_conditioning_state, _ignored = remap_fn(checkpoint_inputs.state_dict)
    model_args, tt_model, tt_kv_cache, _loaded_state = create_model_fn(
        mesh_device,
        state_dict=backbone_state,
        model_path=str(backbone_config_dir or default_backbone_config_dir()),
        **model_kwargs,
    )
    # HF stores Gemma4TextScaledWordEmbedding.embed_scale as a BF16 buffer.
    # Keep the DiffusionGemma-owned model instance on that exact scalar rather
    # than multiplying embeddings by the unrounded Python sqrt(hidden_size).
    tt_model.embed_scale = torch.tensor(tt_model.hidden_size**0.5, dtype=torch.bfloat16).item()
    if attach_hybrid_kv:
        from models.experimental.diffusion_gemma.tt.hybrid_kv import attach_model_owned_hybrid_kv

        # Page tables and full-layer views land on the model itself; consumers
        # fall back to them when no explicit tables are passed (generate.py).
        attach_model_owned_hybrid_kv(tt_model, max_seq_len=int(model_kwargs["max_seq_len"]))
    return CheckpointModelInputs(
        tokenizer=checkpoint_inputs.tokenizer,
        state_dict=checkpoint_inputs.state_dict,
        model_args=model_args,
        tt_model=tt_model,
        tt_kv_cache=tt_kv_cache,
    )


def build_tt_model_from_checkpoint_dir(
    mesh_device,
    checkpoint_dir: str | Path,
    *,
    tokenizer_kwargs: dict | None = None,
    state_prefixes: tuple[str, ...] | list[str] | str = TEXT_GENERATION_PREFIXES,
    state_device: str = "cpu",
    checkpoint_loader=load_checkpoint_inputs,
    **model_kwargs,
) -> CheckpointModelInputs:
    """Load a DiffusionGemma checkpoint directory and build the TT text model."""

    if state_prefixes == TEXT_GENERATION_PREFIXES and model_kwargs.get("num_layers") is not None:
        state_prefixes = text_generation_prefixes_for_layers(model_kwargs["num_layers"])
    inputs = checkpoint_loader(
        checkpoint_dir,
        tokenizer_kwargs=tokenizer_kwargs,
        state_prefixes=state_prefixes,
        device=state_device,
    )
    return build_tt_model_from_checkpoint_inputs(mesh_device, inputs, **model_kwargs)


def generate_text_from_checkpoint_model_inputs(
    checkpoint_model_inputs: CheckpointModelInputs,
    prompt,
    *,
    generate_fn=None,
    **generate_kwargs,
):
    """Run prompt-to-text generation from a prebuilt checkpoint/model bundle."""

    if generate_fn is None:
        from models.experimental.diffusion_gemma.tt.generate import generate_text_from_checkpoint_state

        generate_fn = generate_text_from_checkpoint_state
    return generate_fn(
        checkpoint_model_inputs.tt_model,
        checkpoint_model_inputs.tokenizer,
        prompt,
        dg_state_dict=checkpoint_model_inputs.state_dict,
        **generate_kwargs,
    )
