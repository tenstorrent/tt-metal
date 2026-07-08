# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DG-local model construction with a MoE experts weight-dtype knob (#47475 datatype sweep).

DiffusionGemma's MoE expert weights (gate/up/down proj) are ~88.6% of the on-device weight
DRAM (~11.6 GiB/chip) and a large fraction of the compute-bound denoise step. The shared
Gemma4 backbone builds every module at the model-wide default dtype (``ttnn.bfloat16``)
unless ``models/demos/gemma4/precision_overrides.json`` carries an override; there is no
entry for ``gemma-4-26B-A4B-it``, so DiffusionGemma's experts currently load as bf16.

This module lets DiffusionGemma flip ONLY the expert gate/up/down weights to
``ttnn.bfloat8_b`` via an env knob (``DG_EXPERTS_BFP8=1`` or ``DG_EXPERTS_DTYPE=bfp8``)
WITHOUT editing the shared backbone (the hard no-shared-edits rule). For the default (no
override) case it delegates to the shared ``create_tt_model`` unchanged; only when an
override is requested does it replicate ``create_tt_model``'s body so it can pass a
``Gemma4Precision`` with an ``experts`` override into ``Gemma4Model`` (the copy-shared-into-DG
convention: we may not edit the shared constructor, so we reproduce it here).

Decision-path precision is preserved: the override touches ONLY the ``experts`` module. The
router (``MoEBlock`` keeps ``router_dtype=bf16``), attention, shared MLP, embedding, lm_head,
the KV-cache, and every DiffusionGemma decision op (final logit softcap, softmax->
probability, entropy, Gumbel-max argmax, entropy-budget accept/renoise) stay bf16/fp32.

Expert cache filenames already carry the dtype suffix (``_bfp8`` / ``_bf16`` +
``_dtype_BFLOAT8_B`` / ``_dtype_BFLOAT16``), so bf16 and bfp8 expert caches coexist; the
first bfp8 build regenerates the expert tensors from the checkpoint.
"""

from __future__ import annotations

import os

import ttnn

_DTYPE_BY_NAME = {
    "bf16": ttnn.bfloat16,
    "bfloat16": ttnn.bfloat16,
    "bfp8": ttnn.bfloat8_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "fp32": ttnn.float32,
    "float32": ttnn.float32,
}


def dg_experts_dtype_override():
    """Return the ttnn dtype requested for the MoE experts by the DG env knobs, else ``None``.

    ``DG_EXPERTS_DTYPE`` (``bf16`` | ``bfp8`` | ``fp32``) takes precedence; ``DG_EXPERTS_BFP8=1``
    is shorthand for ``bfp8``. Returns ``None`` when neither is set (default bf16 behaviour).
    """
    val = os.getenv("DG_EXPERTS_DTYPE")
    if not val and os.getenv("DG_EXPERTS_BFP8") == "1":
        val = "bfp8"
    if not val:
        return None
    key = val.strip().lower()
    if key not in _DTYPE_BY_NAME:
        raise ValueError(f"DG_EXPERTS_DTYPE={val!r} unknown; expected one of {sorted(_DTYPE_BY_NAME)}")
    return _DTYPE_BY_NAME[key]


def create_tt_model_dg(mesh_device, **kwargs):
    """Build the Gemma4 backbone for DiffusionGemma, honouring the DG experts-dtype knob.

    With no experts override this delegates to the shared ``create_tt_model`` unchanged (zero
    behaviour drift). With an override it replicates ``create_tt_model`` and injects a
    ``Gemma4Precision`` that overrides ONLY the ``experts`` module dtype. Returns the same
    ``(model_args, model, tt_kv_cache, state_dict)`` tuple as ``create_tt_model``.
    """
    from models.demos.gemma4.tt.common import create_tt_model

    override = dg_experts_dtype_override()
    if override is None:
        return create_tt_model(mesh_device, **kwargs)

    # --- replicated from models/demos/gemma4/tt/common.create_tt_model (keep in sync) ---
    from loguru import logger

    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.model import Gemma4Model
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
    from models.demos.gemma4.tt.precision import KNOWN_MODULES, Gemma4Precision

    max_batch_size = kwargs.get("max_batch_size", 1)
    max_seq_len = kwargs.get("max_seq_len", 8192)
    dtype = kwargs.get("dtype", ttnn.bfloat16)
    state_dict = kwargs.get("state_dict", None)
    num_layers = kwargs.get("num_layers", None)
    mesh_config = kwargs.get("mesh_config", None)
    paged_attention_config = kwargs.get("paged_attention_config", None)
    create_kv_cache = kwargs.get("create_kv_cache", True)
    model_path = kwargs.get("model_path", None)
    bounded_sliding_kv_cache = kwargs.get("bounded_sliding_kv_cache", False)

    model_path = (
        model_path
        or os.getenv("HF_MODEL")
        or os.getenv("GEMMA4_MODEL_PATH", "/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it")
    )

    hf_config = Gemma4ModelArgs.load_hf_config(model_path)
    model_args = Gemma4ModelArgs.from_hf_config(hf_config)
    model_args.model_cache_path = model_args.resolve_model_cache_path(model_path)
    hf_text_config = getattr(hf_config, "text_config", hf_config)
    model_args._hf_text_config = hf_text_config

    if num_layers is not None:
        model_args.num_hidden_layers = num_layers

    if mesh_config is None:
        is_mesh = hasattr(mesh_device, "shape")
        num_devices = mesh_device.get_num_devices() if is_mesh else 1
        if is_mesh and num_devices > 1:
            mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        else:
            mesh_config = MeshConfig((1, 1), decode=ModeConfig(tp=1))

    is_mesh = hasattr(mesh_device, "shape")
    num_devices = mesh_device.get_num_devices() if is_mesh else 1
    ccl_manager = CCLManager(mesh_device) if (is_mesh and num_devices > 1) else None

    if state_dict is None:
        state_dict = Gemma4ModelArgs.load_state_dict(model_path, dummy_weights=False)

    tensor_cache_path = str(model_args.weight_cache_path(dtype))

    mesh_shape = tuple(mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1)
    precision = Gemma4Precision.load(model_path, mesh_shape)

    # DG-local experts-dtype override (the ONLY change vs create_tt_model). Preserve any
    # existing JSON overrides via the public getter, then force ``experts``.
    _sentinel = object()
    merged = {}
    for module_name in KNOWN_MODULES:
        value = precision.get(module_name, _sentinel)
        if value is not _sentinel:
            merged[module_name] = value
    merged["experts"] = override
    precision = Gemma4Precision(merged)
    logger.info(f"[dg-precision] MoE experts dtype override -> {override} (decision path stays bf16/fp32)")

    model = Gemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        dtype=dtype,
        tensor_cache_path=tensor_cache_path,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len,
        max_local_batch_size=max_batch_size,
        num_layers=num_layers,
        paged_attention_config=paged_attention_config,
        create_kv_cache=create_kv_cache,
        precision=precision,
        bounded_sliding_kv_cache=bounded_sliding_kv_cache,
    )

    return model_args, model, model.tt_kv_cache, state_dict
