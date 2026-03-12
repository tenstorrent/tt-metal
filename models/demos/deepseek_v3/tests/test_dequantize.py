# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Sequence

import pytest
import torch

from models.demos.deepseek_v3.utils.hf_model_utils import load_weight_from_weights_dict
from models.demos.deepseek_v3.utils.test_utils import load_state_dict

REFERENCE_WEIGHT_KEYS = [
    # Embedding + LM head (full model endpoints)
    "model.embed_tokens.weight",
    "lm_head.weight",
    # Dense decoder block weights
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    # MoE decoder block weights
    "model.layers.3.mlp.experts.0.gate_proj.weight",
]

OLD_PIPELINE_DEQUANTIZED_REFERENCE_WEIGHT_KEYS = [
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.3.mlp.experts.0.gate_proj.weight",
]

OLD_PIPELINE_DEQUANTIZED_DTYPE = torch.bfloat16
FULL_DEQUANT_COMPARE_ENV_VAR = "DEEPSEEK_V3_COMPARE_ALL_DEQUANTIZED_WEIGHTS"
QUANTIZED_MODEL_PATH_ENV_VAR = "DEEPSEEK_V3_QUANTIZED_HF_MODEL"
FULL_DEQUANT_COMPARE_TIMEOUT_SECONDS = 7200
PRINT_TENSORS_ENV_VAR = "DEEPSEEK_V3_DEQUANT_PRINT_TENSORS"
MAX_DIFF_INDICES_TO_PRINT = 20
MAX_TENSOR_ELEMENTS_TO_PRINT = 100


def _dequantize_reference_tensor(
    tensor: torch.Tensor,
    inv_scale: torch.Tensor,
    block_shape: Sequence[int],
) -> torch.Tensor:
    """Reference-only old dequantization path used to validate saved dequantized checkpoints."""
    if tensor.ndim != inv_scale.ndim:
        raise ValueError(f"Tensor and inverse scale must have same ndim, got {tensor.ndim} and {inv_scale.ndim}")
    if len(block_shape) != tensor.ndim:
        raise ValueError(
            f"Block shape rank mismatch, got len(block_shape)={len(block_shape)} and tensor.ndim={tensor.ndim}"
        )
    if any(inv_scale.shape[i] * block_shape[i] < tensor.shape[i] for i in range(tensor.ndim)):
        raise ValueError(
            "Inverse scale shape does not cover tensor shape: "
            f"tensor={tuple(tensor.shape)}, inv_scale={tuple(inv_scale.shape)}, block_shape={tuple(block_shape)}"
        )

    original_shape = tuple(tensor.shape)
    padded_shape = tuple(inv_scale.shape[i] * block_shape[i] for i in range(tensor.ndim))
    original_slices = tuple(slice(0, size) for size in original_shape)

    out = tensor.float()
    out = out.clone() if out.data_ptr() == tensor.data_ptr() else out

    if padded_shape != original_shape:
        padded = torch.zeros(padded_shape, dtype=out.dtype)
        padded[original_slices] = out
        out = padded

    interleaved_shape: list[int] = []
    scale_broadcast_shape: list[int] = []
    for dim, block_dim in enumerate(block_shape):
        blocks = inv_scale.shape[dim]
        interleaved_shape.extend([blocks, block_dim])
        scale_broadcast_shape.extend([blocks, 1])

    out_view = out.reshape(*interleaved_shape)
    out_view.mul_(inv_scale.float().reshape(*scale_broadcast_shape))
    out = out_view.reshape(*padded_shape)
    return out[original_slices]


def _print_tensor_comparison(
    stored: torch.Tensor,
    expected: torch.Tensor,
    weight_name: str,
    *,
    stored_label: str = "stored",
    expected_label: str = "expected",
) -> None:
    """Print tensor comparison details for debugging. Enable with DEEPSEEK_V3_DEQUANT_PRINT_TENSORS=1."""
    if os.getenv(PRINT_TENSORS_ENV_VAR) != "1":
        return

    def _tensor_stats(t: torch.Tensor, label: str) -> str:
        t_flat = t.flatten().float()
        stats = (
            f"  {label}: shape={tuple(t.shape)} dtype={t.dtype}\n"
            f"    min={t_flat.min().item():.6g} max={t_flat.max().item():.6g} "
            f"mean={t_flat.mean().item():.6g} std={t_flat.std().item():.6g}"
        )
        return stats

    print(f"\n--- Tensor comparison for '{weight_name}' ---")
    print(_tensor_stats(stored, stored_label))
    print(_tensor_stats(expected, expected_label))

    n_elements = stored.numel()
    if n_elements <= MAX_TENSOR_ELEMENTS_TO_PRINT:
        print(f"\n  {stored_label} (full):\n{stored}")
        print(f"\n  {expected_label} (full):\n{expected}")
    else:
        sample_size = min(5, n_elements)
        print(f"\n  {stored_label} (first {sample_size} elements): {stored.flatten()[:sample_size].tolist()}")
        print(f"  {expected_label} (first {sample_size} elements): {expected.flatten()[:sample_size].tolist()}")

    diff_mask = stored != expected
    n_diffs = diff_mask.sum().item()
    print(f"\n  Mismatched elements: {n_diffs} / {n_elements}")
    if n_diffs > 0:
        diff_indices = torch.nonzero(diff_mask.flatten(), as_tuple=False).flatten()
        n_show = min(MAX_DIFF_INDICES_TO_PRINT, len(diff_indices))
        for i in range(n_show):
            idx = diff_indices[i].item()
            print(f"    idx={idx}: stored={stored.flatten()[idx].item()} expected={expected.flatten()[idx].item()}")
        if n_diffs > n_show:
            print(f"    ... and {n_diffs - n_show} more")
    print("---\n")


def _resolve_quantized_model_path(model_path: Path) -> Path | None:
    explicit_path = os.getenv(QUANTIZED_MODEL_PATH_ENV_VAR)
    if explicit_path:
        candidate = Path(explicit_path).resolve()
        return candidate if candidate.is_dir() else None

    model_path = Path(model_path).resolve()
    if model_path.name.endswith("-dequantized"):
        candidate = model_path.with_name(model_path.name.removesuffix("-dequantized"))
        return candidate if candidate.is_dir() else None

    return None


def _iter_quantized_weight_names(state_dict):
    for weight_name in state_dict.keys():
        if weight_name.endswith("_scale_inv"):
            continue
        if f"{weight_name}_scale_inv" in state_dict:
            yield weight_name


def _materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # Force a plain dense tensor so exact comparisons do not trip over mmap/storage wrappers.
    return tensor.detach().clone(memory_format=torch.contiguous_format)


def _assert_saved_weight_matches_old_pipeline(
    *,
    dequantized_state_dict,
    quantized_state_dict,
    hf_config,
    weight_name: str,
):
    assert weight_name in quantized_state_dict, f"Original quantized checkpoint is missing '{weight_name}'"
    assert (
        f"{weight_name}_scale_inv" in quantized_state_dict
    ), f"Original quantized checkpoint is missing '{weight_name}_scale_inv'"
    assert weight_name in dequantized_state_dict, f"Saved dequantized checkpoint is missing '{weight_name}'"
    assert (
        f"{weight_name}_scale_inv" not in dequantized_state_dict
    ), f"Saved dequantized checkpoint unexpectedly still contains '{weight_name}_scale_inv'"

    expected_weight = _dequantize_reference_tensor(
        quantized_state_dict[weight_name],
        quantized_state_dict[f"{weight_name}_scale_inv"],
        hf_config.quantization_config["weight_block_size"],
    ).to(OLD_PIPELINE_DEQUANTIZED_DTYPE)
    stored_weight = dequantized_state_dict[weight_name]

    assert stored_weight.dtype == OLD_PIPELINE_DEQUANTIZED_DTYPE, (
        f"Expected saved weight '{weight_name}' to have dtype {OLD_PIPELINE_DEQUANTIZED_DTYPE}, "
        f"got {stored_weight.dtype}"
    )
    stored_weight = _materialize_tensor(stored_weight)
    expected_weight = _materialize_tensor(expected_weight)

    assert stored_weight.shape == expected_weight.shape, (
        f"Saved dequantized weight '{weight_name}' shape mismatch: "
        f"stored={tuple(stored_weight.shape)} expected={tuple(expected_weight.shape)}"
    )
    if not torch.equal(stored_weight, expected_weight):
        _print_tensor_comparison(
            stored_weight,
            expected_weight,
            weight_name,
            stored_label="stored (dequantized checkpoint)",
            expected_label="expected (old pipeline)",
        )
        raise AssertionError(f"Saved dequantized weight '{weight_name}' does not exactly match the old pipeline output")


@pytest.mark.parametrize("weight_name", REFERENCE_WEIGHT_KEYS)
def test_loaded_dequantized_weight_matches_reference_tensor(state_dict, weight_name):
    if weight_name not in state_dict:
        pytest.skip(f"Checkpoint does not contain '{weight_name}'")

    # The dequantized checkpoint should not have fp8 scales for these weights.
    assert f"{weight_name}_scale_inv" not in state_dict

    reference_weight = state_dict[weight_name]
    assert reference_weight.dtype != torch.float8_e4m3fn

    load_weight = load_weight_from_weights_dict(state_dict)
    target_tensor = torch.empty_like(reference_weight)
    loaded_tensor = load_weight(weight_name, target_tensor)

    assert loaded_tensor is target_tensor
    try:
        torch.testing.assert_close(loaded_tensor, reference_weight, rtol=0.0, atol=0.0)
    except AssertionError:
        _print_tensor_comparison(
            loaded_tensor,
            reference_weight,
            weight_name,
            stored_label="loaded",
            expected_label="reference",
        )
        raise


def test_loaded_dequantized_weight_matches_reference_with_target_dtype_cast(state_dict):
    weight_name = "model.layers.0.mlp.down_proj.weight"
    if weight_name not in state_dict:
        pytest.skip(f"Checkpoint does not contain '{weight_name}'")

    reference_weight = state_dict[weight_name]
    assert reference_weight.dtype != torch.float8_e4m3fn

    load_weight = load_weight_from_weights_dict(state_dict)
    target_tensor = torch.empty(reference_weight.shape, dtype=torch.float32)
    loaded_tensor = load_weight(weight_name, target_tensor)

    assert loaded_tensor is target_tensor
    assert loaded_tensor.dtype == torch.float32
    reference_fp32 = reference_weight.to(torch.float32)
    try:
        torch.testing.assert_close(loaded_tensor, reference_fp32, rtol=0.0, atol=0.0)
    except AssertionError:
        _print_tensor_comparison(
            loaded_tensor,
            reference_fp32,
            weight_name,
            stored_label="loaded",
            expected_label="reference",
        )
        raise


def test_dequantized_checkpoint_has_all_original_weights(model_path):
    """
    Verify that every weight from the original quantized checkpoint is saved in the
    dequantized checkpoint. Uses index JSON only (no tensor loading).
    """
    quantized_model_path = _resolve_quantized_model_path(model_path)
    if quantized_model_path is None:
        pytest.skip(
            f"Could not resolve the original quantized checkpoint. Set {QUANTIZED_MODEL_PATH_ENV_VAR} "
            "or run with DEEPSEEK_V3_HF_MODEL pointing at a '-dequantized' checkpoint."
        )

    deq_path = Path(model_path).resolve()
    orig_index_path = quantized_model_path / "model.safetensors.index.json"
    deq_index_path = deq_path / "model.safetensors.index.json"
    assert orig_index_path.is_file(), f"Original index not found: {orig_index_path}"
    assert deq_index_path.is_file(), f"Dequantized index not found: {deq_index_path}"

    orig_index = json.loads(orig_index_path.read_text())
    deq_index = json.loads(deq_index_path.read_text())

    orig_keys = set(orig_index["weight_map"].keys())
    deq_keys = set(deq_index["weight_map"].keys())

    # Expected in dequantized = all original keys except _scale_inv (folded into weight)
    orig_weight_keys = {k for k in orig_keys if not k.endswith("_scale_inv")}

    missing = orig_weight_keys - deq_keys
    extra = deq_keys - orig_weight_keys

    assert not missing, (
        f"Dequantized checkpoint is missing {len(missing)} weights from the original: "
        f"{sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}"
    )
    assert not extra, (
        f"Dequantized checkpoint has {len(extra)} unexpected keys not in the original: "
        f"{sorted(extra)[:20]}{'...' if len(extra) > 20 else ''}"
    )


@pytest.mark.parametrize("weight_name", OLD_PIPELINE_DEQUANTIZED_REFERENCE_WEIGHT_KEYS)
def test_saved_dequantized_reference_weights_match_old_pipeline(model_path, hf_config, state_dict, weight_name):
    quantized_model_path = _resolve_quantized_model_path(model_path)
    if quantized_model_path is None:
        pytest.skip(
            f"Could not resolve the original quantized checkpoint. Set {QUANTIZED_MODEL_PATH_ENV_VAR} "
            "or run with DEEPSEEK_V3_HF_MODEL pointing at a '-dequantized' checkpoint."
        )

    quantized_state_dict = load_state_dict(quantized_model_path, "")
    try:
        _assert_saved_weight_matches_old_pipeline(
            dequantized_state_dict=state_dict,
            quantized_state_dict=quantized_state_dict,
            hf_config=hf_config,
            weight_name=weight_name,
        )
    finally:
        quantized_state_dict.clear_cache()
        quantized_state_dict.close()
        state_dict.clear_cache()


@pytest.mark.timeout(FULL_DEQUANT_COMPARE_TIMEOUT_SECONDS)
def test_all_saved_dequantized_weights_match_old_pipeline(model_path, hf_config, state_dict):
    if os.getenv(FULL_DEQUANT_COMPARE_ENV_VAR) != "1":
        pytest.skip(f"Set {FULL_DEQUANT_COMPARE_ENV_VAR}=1 to run the full element-by-element checkpoint comparison.")

    quantized_model_path = _resolve_quantized_model_path(model_path)
    if quantized_model_path is None:
        pytest.skip(
            f"Could not resolve the original quantized checkpoint. Set {QUANTIZED_MODEL_PATH_ENV_VAR} "
            "or run with DEEPSEEK_V3_HF_MODEL pointing at a '-dequantized' checkpoint."
        )

    quantized_state_dict = load_state_dict(quantized_model_path, "")
    try:
        quantized_weight_names = list(_iter_quantized_weight_names(quantized_state_dict))
        assert quantized_weight_names, "Did not find any quantized weights with matching *_scale_inv tensors"

        for weight_name in quantized_weight_names:
            _assert_saved_weight_matches_old_pipeline(
                dequantized_state_dict=state_dict,
                quantized_state_dict=quantized_state_dict,
                hf_config=hf_config,
                weight_name=weight_name,
            )
            # Keep the exhaustive check memory-bounded while preserving mmap'd file handles.
            quantized_state_dict.clear_cache()
            state_dict.clear_cache()
    finally:
        quantized_state_dict.close()
