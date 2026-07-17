# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.utils.hf_model_utils import dequantize_state_dict as _fp8_dequantize_state_dict

# Wormhole B0 worker-L1 override for ring MLA tests. On Blackhole we use the platform default.
WH_WORKER_L1_SIZE = 1344544


def print_buffers(device, name, buffer_type):
    buffers = ttnn._ttnn.reports.get_buffers(device)
    filtered_buffers = [buf for buf in buffers if buf.buffer_type == buffer_type]
    for i, buf in enumerate(filtered_buffers):
        logger.warning(
            f"{buffer_type} [{name}] Buffer {i}: addr={buf.address}, "
            f"size={buf.max_size_per_bank}, layout={buf.buffer_layout}"
        )


def print_l1_buffers(device, name):
    print_buffers(device, name, ttnn.BufferType.L1)


def print_l1_small_buffers(device, name):
    print_buffers(device, name, ttnn.BufferType.L1_SMALL)


def adjust_shapes_for_testing(config, mesh_device):
    """Scale TP dimension for smaller meshes. sp_dim (per-device seq len) is always correct."""
    _, n_tp_devices = mesh_device.shape
    if n_tp_devices != 4:
        config.dim = config.dim // (4 // n_tp_devices)


def get_input_mem_config(config, mesh_shape):
    shard_height = (config.sp_dim + config.num_cores - 1) // config.num_cores
    shard_height = ((shard_height + 31) // 32) * 32
    shard_width = (config.dim + mesh_shape[1] - 1) // mesh_shape[1]
    return ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=config.core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def save_intermediate_output(
    tensor: torch.Tensor,
    name: str,
    test_params: dict,
    output_dir: Optional[Union[str, Path]] = None,
):
    """
    Save intermediate output tensor to timestamped .pt file.

    Args:
        tensor: Output tensor to save
        name: Name/type of output (e.g., "norm", "lm_head")
        test_params: Dict with all test parameters (mesh_shape, isl_total, num_layers, etc.)
        output_dir: Output directory (default: /tmp/{name}_outputs or {NAME}_OUTPUT_DIR env var)
    """
    # Get output directory
    if output_dir is None:
        env_var = f"{name.upper()}_OUTPUT_DIR"
        default_dir = f"/tmp/{name}_outputs"
        output_dir = Path(os.getenv(env_var, default_dir))
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp and params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize string values
    input_source = test_params.get("input_source", "unknown").replace("/", "_").replace(".", "_")
    gate_mode = test_params.get("gate_fallback_mode")
    gate_str = gate_mode.value if hasattr(gate_mode, "value") else str(gate_mode)

    # Build filename
    mesh_shape = test_params["mesh_shape"]
    filename = (
        f"{name}_{timestamp}_"
        f"mesh{mesh_shape[0]}x{mesh_shape[1]}_"
        f"isl{test_params['isl_total']}_"
        f"L{test_params['num_layers']}_"
        f"e{test_params['n_routed_experts']}_"
        f"dbcf{test_params['dispatch_buffer_capacity_factor']}_"
        f"gate{gate_str}_"
        f"{'pretrained' if test_params.get('use_pretrained') else 'random'}_"
        f"{input_source}.pt"
    )

    save_path = output_dir / filename

    # Save tensor with metadata
    torch.save(
        {
            f"{name}_output": tensor,
            "metadata": test_params,
        },
        save_path,
    )

    logger.info(f"✓ Saved {name} output to: {save_path}")
    logger.info(f"  Shape: {tensor.shape}, Mean: {tensor.mean():.6f}, Std: {tensor.std():.6f}")

    return save_path


def detect_language_model_prefix(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Detect the multimodal LM-submodule key prefix of a checkpoint.

    Multimodal checkpoints (e.g. Kimi K2.6's ``KimiK25ForConditionalGeneration``) nest the
    language model under ``language_model.`` alongside ``vision_tower.``; text-only or
    already-dequantized checkpoints use bare ``model.`` keys. Returns the prefix to prepend so
    the rest of the pipeline can address weights with bare ``model.``/``lm_head.`` keys.
    """
    for probe in ("language_model.model.embed_tokens.weight", "language_model.model.norm.weight"):
        if probe in state_dict:
            return "language_model."
    return ""


def _get_quantization_config_dict(hf_config: Any) -> dict | None:
    """Return the ``quantization_config`` dict from an hf_config."""
    qc = getattr(hf_config, "quantization_config", None)
    if qc is None:
        text_cfg = getattr(hf_config, "text_config", None)
        if text_cfg is not None:
            qc = getattr(text_cfg, "quantization_config", None)
    return qc


def _find_int4_weight_group(quantization_config: Any) -> dict | None:
    """Return the ``weights`` config of the first pack-quantized group with ``num_bits == 4``, else ``None``.

    ``is_pack_quantized_int4`` and ``_pack_quant_params`` both resolve the authoritative INT4 group
    through this helper so they can never disagree on *which* config group is being read -- the INT4
    group is not necessarily ``group_0`` (e.g. a checkpoint may put a different precision in ``group_0``).
    """
    if not isinstance(quantization_config, dict):
        return None
    if quantization_config.get("quant_method") not in (None, "compressed-tensors"):
        return None
    for group in (quantization_config.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        if weights.get("num_bits") == 4:
            return weights
    return None


def is_pack_quantized_int4(quantization_config: Any) -> bool:
    """True for compressed-tensors pack-quantized INT4 (a config group with num_bits == 4)."""
    return _find_int4_weight_group(quantization_config) is not None


def _pack_quant_params(quantization_config: dict) -> tuple[int, int, bool]:
    """Extract and validate ``(num_bits, group_size, symmetric)`` for the pack-quantized weight group."""
    weights_cfg = _find_int4_weight_group(quantization_config)
    if weights_cfg is None:
        raise ValueError("quantization_config has no pack-quantized INT4 weight group (num_bits == 4).")
    num_bits = int(weights_cfg["num_bits"])
    group_size = int(weights_cfg["group_size"])
    symmetric = bool(weights_cfg.get("symmetric", True))
    weight_type = weights_cfg.get("type", "int")
    strategy = weights_cfg.get("strategy", "group")
    actorder = weights_cfg.get("actorder")
    if weight_type != "int" or strategy != "group" or actorder or not symmetric:
        raise NotImplementedError(
            "Pure-torch pack-quant dequant supports symmetric group-wise int weights without "
            f"activation reordering; got type={weight_type!r}, strategy={strategy!r}, "
            f"actorder={actorder!r}, symmetric={symmetric}."
        )
    return num_bits, group_size, symmetric


def _dequantize_packed_int4_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    shape: torch.Tensor | Sequence[int],
    *,
    group_size: int,
    num_bits: int = 4,
    symmetric: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize one compressed-tensors pack-quantized INT weight to ``dtype`` (pure torch).

    Self-contained INT counterpart to the shared fp8 ``dequantize_weight_tensor``, so the INT4
    path used by Kimi K2.6's routed experts needs no external quant library.

    Layout:
      * ``packed`` is an int32 tensor holding ``32 // num_bits`` little-endian ``num_bits``-wide
        lanes per word along the input dim: input column ``c`` lives in word ``c // pack_factor``,
        lane ``c % pack_factor`` (lane ``j`` occupies bits ``[num_bits*j, num_bits*(j+1))``).
      * Symmetric int weights are stored offset-binary: ``stored = value + 2**(num_bits-1)``.
      * ``scale`` carries one fp scale per ``group_size`` contiguous input columns.
      * ``shape`` is the logical ``[out_features, in_features]``.
    """
    if 32 % num_bits != 0:
        raise ValueError(f"num_bits must divide 32 for int32 packing, got num_bits={num_bits}.")
    if not symmetric:
        raise NotImplementedError("Only symmetric INT pack-quantization is supported.")

    out_features, in_features = int(shape[0]), int(shape[1])
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1

    # Unpack little-endian num_bits-wide lanes from each int32 word along the input dim.
    words = packed.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(pack_factor, dtype=torch.int64, device=words.device) * num_bits
    lanes = (words.unsqueeze(-1) >> shifts) & mask  # [out_features, n_words, pack_factor]
    levels = lanes.reshape(out_features, -1)[:, :in_features].to(torch.float32)
    levels -= float(1 << (num_bits - 1))  # offset-binary storage -> signed level

    # Broadcast the per-group scale across each contiguous block of ``group_size`` columns.
    scale_full = scale.to(torch.float32).repeat_interleave(group_size, dim=1)[:, :in_features]
    return (levels * scale_full).to(dtype).contiguous()


def _dequantize_pack_quantized_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    quantization_config: dict,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Dequantize a (sub-)state_dict whose quantized tensors are pack-quantized INT4 triplets.

    ``X.weight_packed`` anchors emit ``X.weight``; ``_scale``/``_shape`` companions are
    consumed with their anchor. Non-quantized tensors pass through with their source dtype
    preserved -- notably the fp32 ``e_score_correction_bias`` that feeds the router top-k,
    which must not be downcast to bf16.
    """
    params = None  # (num_bits, group_size, symmetric); parsed lazily on the first packed anchor
    out: dict[str, torch.Tensor] = {}
    n_dequantized = 0  # count of pack-quantized weights actually unpacked (for the summary log)
    for name in sorted(state_dict.keys()):
        if name.endswith(("_scale", "_shape")):
            continue  # consumed alongside the matching _packed anchor
        if name.endswith("_packed"):
            base = name[: -len("_packed")]  # "...gate_proj.weight"
            scale_name = base + "_scale"
            shape_name = base + "_shape"
            if scale_name not in state_dict or shape_name not in state_dict:
                raise ValueError(f"INT4 anchor '{name}' is missing companion(s) '{scale_name}'/'{shape_name}'.")
            if params is None:
                params = _pack_quant_params(quantization_config)
            num_bits, group_size, symmetric = params
            out[base] = _dequantize_packed_int4_weight(
                state_dict[name],
                state_dict[scale_name],
                state_dict[shape_name],
                group_size=group_size,
                num_bits=num_bits,
                symmetric=symmetric,
                dtype=dtype,
            )
            n_dequantized += 1
            continue
        tensor = state_dict[name]
        if tensor is None:
            raise ValueError(f"Expected tensor {name} to exist in state_dict but it was None")
        out[name] = tensor.contiguous().clone()  # passthrough: preserve source dtype
    if n_dequantized:
        num_bits, group_size, _ = params
        logger.info(
            f"compressed-tensors INT4: dequantized {n_dequantized} packed weight tensor(s) "
            f"(num_bits={num_bits}, group_size={group_size})"
        )
    return out


def dequantize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    hf_config: Any,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Dequantize an HF (sub-)state_dict for the d_p pipeline.

    Kimi K2.6's compressed-tensors pack-quantized INT4 weights are dequantized locally; every
    other checkpoint (DeepSeek fp8 block-wise) is delegated unchanged to the shared deepseek_v3
    dequantizer.
    """
    quant_cfg = _get_quantization_config_dict(hf_config)
    if is_pack_quantized_int4(quant_cfg):
        return _dequantize_pack_quantized_state_dict(state_dict, quant_cfg, dtype=dtype)
    return _fp8_dequantize_state_dict(state_dict, hf_config, dtype)
