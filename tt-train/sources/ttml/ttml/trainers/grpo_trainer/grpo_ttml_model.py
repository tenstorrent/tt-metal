# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Setup and weight export for the ttml Llama and Qwen3 models that GRPO trains.

:func:`setup_ttml_model` opens the device and builds the model and tokenizer from
a HuggingFace ID or local directory. :class:`GRPOTrainer` calls it and owns the
result; rollout samplers receive the model and tokenizer from the trainer.

:func:`weights_ref_hf_dict` exports a ttml model's parameters as an HF-keyed dict of
on-device ``ttnn.Tensor`` handles, the wire format consumed by tt-transformers'
``Transformer.update_weights(hf_state_dict, hf_rope=False)``.

Qwen3: both stacks store Q/K rows in the same interleaved (Meta-permute-equivalent)
layout:

  - ttml's ``load_weights_from_hf`` runs :func:`unpermute_proj_rows` on
    ``q_proj`` / ``k_proj`` and :func:`unpermute_norm_weights` on ``q_norm`` /
    ``k_norm`` (see ``ttml.models.qwen3.weights``).
  - tt-transformers' ``convert_hf_qkv_to_meta_format`` runs :func:`reverse_permute`
    on ``q_proj`` / ``k_proj`` and :func:`reverse_permute_1d` on ``q_norm`` /
    ``k_norm`` (see ``models/tt_transformers/tt/load_checkpoints.py``).

Both produce the same interleaved rows ``[r0, i0, r1, i1, ...]`` per head. So a
ttml Qwen3 parameter can be handed straight to tt-transformers'
``Transformer.update_weights(hf_rope=False)`` under its HF key -- no additional
permutation. The only shape rewrite is splitting ttml's fused ``kv_proj`` back
into HF's separate ``k_proj`` / ``v_proj``.

The Qwen3 dict is layout-identical to the Llama one (bf16, TILE, DRAM-interleaved,
replicated, HF Linear shapes wrapped in two leading unit dims), so the same weight
transport accepts it without changes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import torch
import ttnn

import ttml
from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.common.utils import build_mesh
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
from ttml.models.qwen3 import Qwen3, create_qwen3_config_from_hf
from ttml.models.qwen3.weights import load_weights_from_hf
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .llama_composite_kv import LlamaCompositeKV

_SUPPORTED_KINDS = ("llama", "qwen3")


def load_checkpoint(model: Any, checkpoint_path: str, dp_mapper: Any = None) -> None:
    from safetensors.numpy import load_file
    import ml_dtypes

    checkpoint = load_file(checkpoint_path)
    parameters = model.parameters()
    loaded, missing = 0, []

    for name, param in parameters.items():
        if name in checkpoint:
            arr = checkpoint[name].astype(ml_dtypes.bfloat16)
            if arr.ndim == 1:
                arr = arr.reshape(1, 1, 1, -1)
            elif arr.ndim == 2:
                arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
            restored = ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, dp_mapper)
            param.assign(restored)
            loaded += 1
        else:
            missing.append(name)

    print(f"Loaded {loaded}/{len(parameters)} parameters from {checkpoint_path}")
    if missing:
        print(f"Warning: {len(missing)} parameters not found in checkpoint:")
        for n in missing:
            print(f"  - {n}")


def load_hf_state_dict(model_source: str) -> dict:
    """Return a HuggingFace float state-dict for ``model_source``."""
    if os.path.isdir(model_source):
        path = model_source
    else:
        path = snapshot_download(
            repo_id=model_source,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        )
    hf_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, trust_remote_code=True)
    state_dict = hf_model.state_dict()
    del hf_model
    return state_dict


# --------------------------------------------------------------
# Device + model construction
# --------------------------------------------------------------


def setup_ttml_model(
    transformer_config: TransformerConfig, device_config: DeviceConfig, model_source: str
) -> Tuple[Any, Any]:
    """Open the device and build the ttml model and tokenizer.

    The model family comes from ``transformer_config.model_type`` (``"llama"`` or
    ``"qwen3"``). Llama reads its architecture from ``transformer_config``; Qwen3
    only reads ``max_sequence_length`` and ``runner_type`` and takes the
    architecture from the HF config of ``model_source``.

    Returns:
        ``(model, tokenizer)``.
    """
    model_kind = transformer_config.model_type
    if model_kind not in _SUPPORTED_KINDS:
        raise ValueError(f"setup_ttml_model: model_type must be one of {_SUPPORTED_KINDS}, got {model_kind!r}")

    mesh_device = open_device(model_kind, device_config)
    if model_kind == "llama":
        return _build_llama(transformer_config, device_config, model_source, mesh_device)
    return _build_qwen3(transformer_config, device_config, model_source)


def open_device(model_kind: str, device_config: DeviceConfig) -> Any:
    """Open the device and return the ``ttnn.MeshDevice``.

    Llama opens the ``AutoContext`` device directly (as
    ``LlamaGRPOCompleter``); Qwen3 opens a named mesh so an ``"fsdp"`` axis
    exists (as ``Qwen3GRPOCompleter``). Tests may override this to reuse an
    already-open device.
    """
    if model_kind == "llama":
        if device_config.total_devices() > 1:
            ttml.core.distributed.enable_fabric(device_config.total_devices())
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
        return autograd_ctx.get_device()

    mesh = build_mesh(device_config)
    device_ids = tuple(device_config.device_ids) if device_config.device_ids else None
    ttml.open_device_mesh(mesh, device_ids)
    return ttml.autograd.AutoContext.get_instance().get_device()


def _build_llama(
    tf_config: TransformerConfig, device_config: DeviceConfig, model_source: str, mesh_device: Any
) -> Tuple[Any, Any]:
    autograd_ctx = ttml.autograd.AutoContext.get_instance()

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    tf_config.vocab_size = len(tokenizer)

    rope_scaling = LlamaRopeScalingConfig(
        scaling_factor=getattr(tf_config, "scaling_factor", 0.0) or 0.0,
        high_freq_factor=getattr(tf_config, "high_freq_factor", 4.0) or 4.0,
        low_freq_factor=getattr(tf_config, "low_freq_factor", 1.0) or 1.0,
        original_context_length=getattr(tf_config, "original_context_length", 0) or 0,
    )

    runner_type = RunnerType.from_string(str(tf_config.runner_type))
    weight_tying = WeightTyingType.Disabled
    if tf_config.weight_tying:
        weight_tying = WeightTyingType.from_string(str(tf_config.weight_tying))

    llama_cfg = LlamaConfig(
        hidden_size=tf_config.embedding_dim,
        intermediate_size=tf_config.intermediate_dim,
        num_hidden_layers=tf_config.num_blocks,
        num_attention_heads=tf_config.num_heads,
        num_key_value_heads=tf_config.num_groups,
        vocab_size=len(tokenizer),
        max_position_embeddings=tf_config.max_sequence_length,
        rope_theta=tf_config.theta or 10000.0,
        attention_dropout=tf_config.dropout_prob,
        mlp_dropout=tf_config.dropout_prob,
        runner_type=runner_type,
        weight_tying=weight_tying,
        rope_scaling=rope_scaling,
    )

    tt_model = LlamaCompositeKV(llama_cfg)

    if device_config.enable_ddp:
        # NOTE: TP is intentionally disabled here. cross_entropy_loss assumes full-vocab logits.
        autograd_ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=False))

    ddp_enabled = (
        autograd_ctx.is_parallelism_context_initialized() and autograd_ctx.get_parallelism_context().is_ddp_enabled()
    )
    dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(mesh_device, 0) if ddp_enabled else None

    local_safetensors = os.path.isdir(model_source) and any(f == "model.safetensors" for f in os.listdir(model_source))
    if local_safetensors:
        logging.info("Loading model from local safetensors: %s", model_source)
        load_checkpoint(tt_model, model_source, dp_mapper=dp_mapper)
    else:
        logging.info("Downloading model from HuggingFace: %s", model_source)
        model_repo_path = snapshot_download(
            repo_id=model_source,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        )
        load_from_safetensors(tt_model, model_repo_path, llama_cfg)

    return tt_model, tokenizer


def _build_qwen3(tf_config: TransformerConfig, device_config: DeviceConfig, model_source: str) -> Tuple[Any, Any]:
    mesh = ttml.mesh()
    fsdp_enabled = bool(device_config.enable_fsdp) and mesh.has_axis("fsdp") and mesh.axis_size("fsdp") > 1

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)

    max_seq_len = int(getattr(tf_config, "max_sequence_length", 2048) or 2048)
    hf_config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
    runner_type = RunnerType.from_string(str(tf_config.runner_type))
    qwen_config = create_qwen3_config_from_hf(hf_config, max_seq_len, runner_type=runner_type)
    tie = bool(getattr(hf_config, "tie_word_embeddings", False))

    logging.info(
        "Building ttml Qwen3 model (hidden=%d, layers=%d)", qwen_config.hidden_size, qwen_config.num_hidden_layers
    )

    if fsdp_enabled and bool(device_config.lazy_parameter_init):
        with ttml.lazy_init():
            tt_model = Qwen3(qwen_config)
        for block in tt_model.blocks:
            ttml.fsdp.fully_shard(block, reshard_after_forward=True)
        ttml.fsdp.fully_shard(tt_model, reshard_after_forward=True)
        ttml.materialize_module(tt_model)

        hf_state_dict = load_hf_state_dict(model_source)
        load_weights_from_hf(tt_model, hf_state_dict, qwen_config, tie_word_embeddings=tie, sharded=True)
        del hf_state_dict
    else:
        tt_model = Qwen3(qwen_config)

        hf_state_dict = load_hf_state_dict(model_source)
        load_weights_from_hf(tt_model, hf_state_dict, qwen_config, tie_word_embeddings=tie)
        del hf_state_dict

        if fsdp_enabled:
            for block in tt_model.blocks:
                ttml.fsdp.fully_shard(block, reshard_after_forward=True)
            ttml.fsdp.fully_shard(tt_model, reshard_after_forward=True)

    return tt_model, tokenizer


# --------------------------------------------------------------
# Weight export
# --------------------------------------------------------------


def weights_ref_hf_dict(model) -> dict[str, ttnn.Tensor]:
    """HF-keyed on-device export of a ttml Llama or Qwen3 for tt-transformers'
    ``Transformer.update_weights(hf_state_dict, hf_rope=False)``."""
    if isinstance(model, Qwen3):
        return _qwen3_weights_ref_hf_dict(model)
    if isinstance(model, Llama):
        return _llama_weights_ref_hf_dict(model)
    raise TypeError(f"weights_ref_hf_dict supports ttml Llama and Qwen3 models, got {type(model).__name__}")


def _llama_weights_ref_hf_dict(model) -> dict[str, ttnn.Tensor]:
    """Export this ttml model's parameters as an HF-keyed dict of on-device
    ``ttnn.Tensor`` handles, shaped for tt-transformers'
    ``Transformer.update_weights(hf_state_dict, hf_rope=False)`` (HF
    safetensors dot-keys; HF shapes wrapped in two leading unit dims;
    bf16, TILE, DRAM-interleaved, replicated).

    Q/K row order: both ttml and TTT store Meta-permuted rows for
    Llama-3.2-1B, so the consumer uses ``hf_rope=False`` (no permutation).

    Tied embeddings: with ``weight_tying=Enabled``, ``embed_tokens`` and
    ``lm_head`` point at the same handle; safe because the consumer
    ``ttnn.copy``s into a separate destination and never aliases the source.

    Most values are live handles into ttml's parameter store; do not mutate
    ttml's parameters between this call and ``update_weights``. The K/V split
    is the exception: ttml fuses K and V into one ``kv_linear/weight``
    (K rows first, then V), so we expose them via two ``ttnn.slice`` calls
    (newly allocated, ~64 MB total for Llama-3.2-1B-Instruct).

    Single-device assumption: parameters must be replicated across the mesh
    (no DDP/TP shard mapper). The grpo single-device config satisfies this;
    DDP/TP would need a host-side per-parameter concat first.
    """
    cfg = model.config
    assert cfg.weight_tying == WeightTyingType.Enabled, (
        "weights_ref_hf_dict requires weight_tying=Enabled (Llama-3.2-1B/-Instruct "
        f"tie embed_tokens and lm_head). Got weight_tying={cfg.weight_tying!r}."
    )

    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    H = cfg.hidden_size
    head_dim = H // n_heads
    kv_dim = n_kv * head_dim

    params = model.parameters()

    def get(name: str) -> ttnn.Tensor:
        if name not in params:
            raise RuntimeError(
                f"ttml parameter {name!r} not found; available keys (first 10): " f"{sorted(params.keys())[:10]}"
            )
        return params[name].get_value()

    out: dict[str, ttnn.Tensor] = {}

    # Tied: same handle exposed under both HF keys.
    fc = get("Llama/fc/weight")
    out["model.embed_tokens.weight"] = fc
    out["lm_head.weight"] = fc
    out["model.norm.weight"] = get("Llama/ln_fc/gamma")

    for i in range(len(model.blocks)):
        p = f"Llama/blocks/{i}"

        out[f"model.layers.{i}.input_layernorm.weight"] = get(f"{p}/attention_norm/gamma")
        out[f"model.layers.{i}.post_attention_layernorm.weight"] = get(f"{p}/mlp_norm/gamma")

        out[f"model.layers.{i}.self_attn.q_proj.weight"] = get(f"{p}/attention/q_linear/weight")
        out[f"model.layers.{i}.self_attn.o_proj.weight"] = get(f"{p}/attention/out_linear/weight")

        kv = get(f"{p}/attention/kv_linear/weight")
        kv_shape = tuple(kv.shape)
        assert kv_shape == (1, 1, 2 * kv_dim, H), (
            f"kv_linear shape mismatch at layer {i}: got {kv_shape}, " f"expected (1, 1, {2 * kv_dim}, {H})"
        )
        out[f"model.layers.{i}.self_attn.k_proj.weight"] = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, kv_dim, H])
        out[f"model.layers.{i}.self_attn.v_proj.weight"] = ttnn.slice(kv, [0, 0, kv_dim, 0], [1, 1, 2 * kv_dim, H])

        out[f"model.layers.{i}.mlp.gate_proj.weight"] = get(f"{p}/mlp/w1/weight")
        out[f"model.layers.{i}.mlp.up_proj.weight"] = get(f"{p}/mlp/w3/weight")
        out[f"model.layers.{i}.mlp.down_proj.weight"] = get(f"{p}/mlp/w2/weight")

    return out


def _qwen3_weights_ref_hf_dict(qwen3_model: Any, tie_word_embeddings: Optional[bool] = None) -> dict[str, ttnn.Tensor]:
    """Export a ttml Qwen3 model's parameters as an HF-keyed dict of on-device
    ``ttnn.Tensor`` handles, shaped for tt-transformers'
    ``Transformer.update_weights(hf_state_dict, hf_rope=False)``.

    Row layouts (already matching between ttml and tt-transformers):

    - ``q_proj`` / ``k_proj``: interleaved ``[r0, i0, r1, i1, ...]`` per head
      -- ttml's ``unpermute_proj_rows`` produces the same output as
      tt-transformers' ``reverse_permute``.
    - ``q_norm`` / ``k_norm`` (Qwen3 QK-Norm gammas): interleaved on head_dim --
      ttml's ``unpermute_norm_weights`` matches tt-transformers'
      ``reverse_permute_1d``.

    Fused ``kv_proj`` (ttml, K rows then V rows) is split via two ``ttnn.slice``
    calls into HF's ``k_proj`` (rows ``[0, kv_dim)``) and ``v_proj`` (rows
    ``[kv_dim, 2*kv_dim)``). Those two slices are newly allocated (a copy of
    ``kv_proj``'s data on each call); every other exported handle is a live
    view into the ttml parameter store -- do not mutate ttml's parameters
    between this call and ``update_weights``.

    Tied embeddings (Qwen3-0.6B): ``model.embed_tokens.weight`` and
    ``lm_head.weight`` expose the same underlying ``fc/weight`` handle; the
    consumer ``ttnn.copy``s into a separate destination and never aliases.

    Single-device / replicated only: with DDP the params stay replicated on the
    mesh, which is fine; a TP shard mapper on the parameters is not.

    Args:
        qwen3_model: A ttml ``Qwen3`` instance (from ``ttml.models.qwen3``).
        tie_word_embeddings: Override for the tying flag. If ``None``, read
            from ``qwen3_model.config.weight_tying``.

    Returns:
        HF-keyed dict of live on-device ``ttnn.Tensor`` handles.
    """
    cfg = qwen3_model.config

    if tie_word_embeddings is None:
        tie = cfg.weight_tying == WeightTyingType.Enabled
    else:
        tie = bool(tie_word_embeddings)

    n_kv = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    kv_dim = n_kv * head_dim
    H = cfg.hidden_size

    params = qwen3_model.parameters()
    # ttml's ``load_weights_from_hf`` reads the root prefix the same way; keep parity.
    any_key = next(iter(params))
    root_prefix = any_key.split("/")[0]

    def get(name: str) -> ttnn.Tensor:
        if name not in params:
            raise RuntimeError(
                f"ttml parameter {name!r} not found; available keys (first 10): {sorted(params.keys())[:10]}"
            )
        return params[name].get_value()

    out: dict[str, ttnn.Tensor] = {}

    fc = get(f"{root_prefix}/fc/weight")
    if tie:
        out["model.embed_tokens.weight"] = fc
        out["lm_head.weight"] = fc
    else:
        out["model.embed_tokens.weight"] = get(f"{root_prefix}/tok_emb/weight")
        out["lm_head.weight"] = fc

    out["model.norm.weight"] = get(f"{root_prefix}/ln_fc/weight")

    for i in range(cfg.num_hidden_layers):
        tp = f"{root_prefix}/blocks/{i}"
        hp = f"model.layers.{i}"

        out[f"{hp}.self_attn.q_proj.weight"] = get(f"{tp}/self_attn/q_proj/weight")
        out[f"{hp}.self_attn.o_proj.weight"] = get(f"{tp}/self_attn/o_proj/weight")
        out[f"{hp}.self_attn.q_norm.weight"] = get(f"{tp}/self_attn/q_norm/weight")
        out[f"{hp}.self_attn.k_norm.weight"] = get(f"{tp}/self_attn/k_norm/weight")

        kv = get(f"{tp}/self_attn/kv_proj/weight")
        kv_shape = tuple(kv.shape)
        assert kv_shape == (
            1,
            1,
            2 * kv_dim,
            H,
        ), f"kv_proj shape mismatch at layer {i}: got {kv_shape}, expected (1, 1, {2 * kv_dim}, {H})"
        out[f"{hp}.self_attn.k_proj.weight"] = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, kv_dim, H])
        out[f"{hp}.self_attn.v_proj.weight"] = ttnn.slice(kv, [0, 0, kv_dim, 0], [1, 1, 2 * kv_dim, H])

        out[f"{hp}.input_layernorm.weight"] = get(f"{tp}/input_layernorm/weight")
        out[f"{hp}.post_attention_layernorm.weight"] = get(f"{tp}/post_attention_layernorm/weight")

        out[f"{hp}.mlp.gate_proj.weight"] = get(f"{tp}/mlp/gate_proj/weight")
        out[f"{hp}.mlp.up_proj.weight"] = get(f"{tp}/mlp/up_proj/weight")
        out[f"{hp}.mlp.down_proj.weight"] = get(f"{tp}/mlp/down_proj/weight")

    return out
