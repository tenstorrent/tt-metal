# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the LMHead1D module (1D mesh topology: N150, N300, T3K).

This test suite verifies:
1. Unit tests for config dataclass (no device needed)
2. LMHead1D matches torch.nn.Linear reference for logit computation
3. from_model_args backward compatibility
"""

import math
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig
from models.common.tensor_utils import TILE_SIZE
from models.common.utility_functions import comp_allclose, comp_pcc

# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_lm_head_1d_config_creation():
    """Test LMHead1DConfig dataclass creation."""
    from unittest.mock import MagicMock

    config = LMHead1DConfig(
        output_weights=[MagicMock()],
        mesh_device=MagicMock(),
        dim=4096,
    )
    assert config.dim == 4096
    assert config.lm_head_dtype == ttnn.bfloat8_b
    assert config.max_batch_size == 32


def test_lm_head_1d_config_defaults():
    """Test LMHead1DConfig default values."""
    from unittest.mock import MagicMock

    config = LMHead1DConfig(output_weights=[MagicMock()])
    assert config.mesh_device is None
    assert config.program_configs is None
    assert config.compute_kernel_config is None
    assert config.lm_head_dtype == ttnn.bfloat8_b
    assert config.output_memcfg is None
    assert config.input_memcfg is None
    assert config.weights_memcfgs is None


def test_lm_head_1d_config_is_resolved_all_fields():
    """Test is_resolved() when all fields are set."""
    from unittest.mock import MagicMock

    mock_device = MagicMock()
    mock_device.get_num_devices.return_value = 1

    config = LMHead1DConfig(
        output_weights=[MagicMock()],
        mesh_device=mock_device,
        dim=4096,
        program_configs=[None],
        compute_kernel_config=MagicMock(),
        output_memcfg=MagicMock(),
        input_memcfg=MagicMock(),
        weights_memcfgs=[MagicMock()],
    )
    assert config.is_resolved()


def test_compute_kernel_config_hifi2():
    """Test _compute_kernel_config_hifi2 returns valid config."""
    from models.common.modules.lm_head.lm_head_1d import _compute_kernel_config_hifi2

    cfg = _compute_kernel_config_hifi2()
    assert cfg.math_fidelity == ttnn.MathFidelity.HiFi2
    assert cfg.packer_l1_acc is True
    assert cfg.fp32_dest_acc_en is False


def test_create_dram_sharded_mem_config():
    """Test _create_dram_sharded_mem_config produces valid MemoryConfig."""
    from models.common.modules.lm_head.lm_head_1d import _create_dram_sharded_mem_config

    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})
    mc = _create_dram_sharded_mem_config(k=4096, n=16032, dram_grid=dram_grid, dram_cores=12)
    assert mc.is_sharded()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert mc.buffer_type == ttnn.BufferType.DRAM


def test_from_model_args_rejects_galaxy():
    """Test from_model_args raises for Galaxy devices."""
    from unittest.mock import MagicMock

    mock_args = MagicMock()
    mock_args.is_galaxy = True

    with pytest.raises(ValueError, match="Galaxy"):
        LMHead1D.from_model_args(
            mesh_device=MagicMock(),
            args=mock_args,
            state_dict={},
            state_dict_prefix="",
            weight_cache_path="",
            max_columns_per_device=32000,
        )


# ============================================================================
# Weight helpers
# ============================================================================

_CACHED_LM_WEIGHTS: dict[str, torch.Tensor] = {}


def _get_or_init_lm_weight(key: str, dim: int, vocab_size: int) -> torch.Tensor:
    if key not in _CACHED_LM_WEIGHTS:
        logger.info(f"\033[33m[cache miss]\033[0m Initializing LM head weight for {key}")
        _CACHED_LM_WEIGHTS[key] = torch.randn(vocab_size, dim, dtype=torch.bfloat16)
    else:
        logger.info(f"\033[32m[cache hit]\033[0m Reusing cached LM head weight for {key}")
    return _CACHED_LM_WEIGHTS[key]


def _prepare_lm_head_weights(
    weight: torch.Tensor, vocab_size: int, dim: int, num_devices: int, max_columns_per_device: int
) -> list[torch.Tensor]:
    """Split LM head weight into chunks matching TTTv1 logic (non-TG path)."""
    padded_vocab_size = math.ceil(vocab_size / 32) * 32
    size_per_device = padded_vocab_size // num_devices
    num_splits = math.ceil(size_per_device / max_columns_per_device)
    split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
    split_sizes.append(size_per_device - sum(split_sizes))

    # Transpose to (dim, vocab) and pad
    torch_w = weight.T  # (dim, vocab_size)
    if vocab_size < padded_vocab_size:
        torch_w = torch.cat([torch_w, torch.zeros(dim, padded_vocab_size - vocab_size, dtype=torch_w.dtype)], dim=-1)

    splits = []
    for i, split_size in enumerate(split_sizes):
        device_splits = []
        for dev in range(num_devices):
            start = dev * size_per_device + sum(split_sizes[:i])
            end = start + split_size
            device_splits.append(torch_w[:, start:end])
        splits.append(torch.cat(device_splits, dim=-1))

    return splits


# ============================================================================
# Model names from HF to cover in tests
# ============================================================================

LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_11B = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
QWEN2_7B = "Qwen/Qwen2-7B-Instruct"
QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN25_72B = "Qwen/Qwen2.5-72B-Instruct"
QWEN25_CODER_32B = "Qwen/Qwen2.5-Coder-32B-Instruct"
QWEN3_32B = "Qwen/Qwen3-32B"
DEEPSEEK_R1_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


_slow = pytest.mark.slow


def _list_test_cases() -> list[pytest.param]:
    # max_columns_per_device derived from TTTv1: 668 * lm_head_core_grid.num_cores
    # For simplicity, use the actual split counts from CSV
    # fmt: off
    return [
        # === Fast tests ===
        # 1x1 Llama-3.1-8B: 3 splits (dim=4096, padded_vocab=128256)
        pytest.param((1, 1), 4096, 128256, 42752, LLAMA_8B, 0.999, id="1x1-8B"),
        # 1x2 Llama-3.1-8B: 2 splits
        pytest.param((1, 2), 4096, 128256, 42752, LLAMA_8B, 0.999, id="1x2-8B"),
        # 1x8 Llama-3.1-8B: 1 split
        pytest.param((1, 8), 4096, 128256, 16032, LLAMA_8B, 0.999, id="1x8-8B"),
        # 1x8 Llama-3.3-70B: 1 split (dim=8192)
        pytest.param((1, 8), 8192, 128256, 16032, LLAMA_70B, 0.999, id="1x8-70B"),

        # === Slow tests ===
        # 1x1 Llama-3.2-1B: 3 splits (dim=2048)
        pytest.param((1, 1), 2048, 128256, 42752, LLAMA_1B, 0.999, id="1x1-1B", marks=_slow),
        # 1x1 Llama-3.2-3B: 4 splits (dim=3072)
        pytest.param((1, 1), 3072, 128256, 32064, LLAMA_3B, 0.999, id="1x1-3B", marks=_slow),
        # 1x1 Mistral-7B: 1 split (dim=4096, vocab=32768)
        pytest.param((1, 1), 4096, 32768, 32768, MISTRAL_7B, 0.999, id="1x1-Mistral-7B", marks=_slow),
        # 1x2 Llama-3.2-1B: 2 splits
        pytest.param((1, 2), 2048, 128256, 42752, LLAMA_1B, 0.999, id="1x2-1B", marks=_slow),
        # 1x2 Llama-3.2-3B: 2 splits
        pytest.param((1, 2), 3072, 128256, 32064, LLAMA_3B, 0.999, id="1x2-3B", marks=_slow),
        # 1x2 Llama-3.2-11B: 2 splits
        pytest.param((1, 2), 4096, 128256, 42752, LLAMA_11B, 0.999, id="1x2-11B", marks=_slow),
        # 1x2 Mistral-7B: 1 split
        pytest.param((1, 2), 4096, 32768, 16384, MISTRAL_7B, 0.999, id="1x2-Mistral-7B", marks=_slow),
        # 1x2 Qwen2-7B: 3 splits (dim=3584, vocab=152064)
        pytest.param((1, 2), 3584, 152064, 37408, QWEN2_7B, 0.999, id="1x2-Qwen2-7B", marks=_slow),
        # 1x2 DeepSeek-R1-14B: 3 splits (dim=5120, vocab=152064)
        pytest.param((1, 2), 5120, 152064, 26720, DEEPSEEK_R1_14B, 0.999, id="1x2-DeepSeek-R1-14B", marks=_slow),
        # 1x2 Qwen2.5-7B: 3 splits
        pytest.param((1, 2), 3584, 152064, 37408, QWEN25_7B, 0.999, id="1x2-Qwen2.5-7B", marks=_slow),
        # 1x8 Llama-3.2-1B: 1 split
        pytest.param((1, 8), 2048, 128256, 16032, LLAMA_1B, 0.999, id="1x8-1B", marks=_slow),
        # 1x8 Llama-3.2-3B: 1 split
        pytest.param((1, 8), 3072, 128256, 16032, LLAMA_3B, 0.999, id="1x8-3B", marks=_slow),
        # 1x8 Llama-3.2-11B: 1 split
        pytest.param((1, 8), 4096, 128256, 16032, LLAMA_11B, 0.999, id="1x8-11B", marks=_slow),
        # 1x8 Qwen2.5-72B: 1 split (dim=8192, vocab=152064)
        pytest.param((1, 8), 8192, 152064, 19008, QWEN25_72B, 0.999, id="1x8-Qwen2.5-72B", marks=_slow),
        # 1x8 Qwen2.5-Coder-32B: 1 split (dim=5120)
        pytest.param((1, 8), 5120, 152064, 19008, QWEN25_CODER_32B, 0.999, id="1x8-Qwen2.5-Coder-32B", marks=_slow),
        # 1x8 Qwen3-32B: 1 split (dim=5120, vocab=151936)
        pytest.param((1, 8), 5120, 151936, 18992, QWEN3_32B, 0.999, id="1x8-Qwen3-32B", marks=_slow),
        # 1x8 Mistral-7B: 1 split
        pytest.param((1, 8), 4096, 32768, 4096, MISTRAL_7B, 0.999, id="1x8-Mistral-7B", marks=_slow),
    ]
    # fmt: on


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape,dim,vocab_size,max_col_per_dev,hf_model_name,pcc",
    _list_test_cases(),
)
def test_lm_head_1d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    mesh_shape,
    dim,
    vocab_size,
    max_col_per_dev,
    hf_model_name,
    pcc,
):
    """
    Test LMHead1D output shape and basic numerical correctness.

    Uses random weights split per TTTv1 logic, verifies output is non-zero
    and has correct shape.
    """
    seed = 42
    torch.manual_seed(seed)
    batch_rows = 32  # tile_padded_batch_rows for batch_size=1
    num_devices = ttnn_mesh_device.get_num_devices()

    # Get reference weight
    key = f"{hf_model_name}_{vocab_size}_{dim}"
    full_weight = _get_or_init_lm_weight(key, dim, vocab_size)

    # Reference: torch.nn.Linear (no bias), in bfloat16
    ref_linear = torch.nn.Linear(dim, vocab_size, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_linear.weight.copy_(full_weight)

    # Reference output
    torch_input = torch.randn(1, 1, batch_rows, dim, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_output = ref_linear(torch_input)  # [1, 1, 32, vocab_size]

    # Split weights for TT model
    weight_splits = _prepare_lm_head_weights(full_weight, vocab_size, dim, num_devices, max_col_per_dev)

    # Create LazyWeights (cache-backed for faster repeated runs)
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/lm_head_1d"))
    lazy_weights = []
    for i, split in enumerate(weight_splits):
        lazy_weights.append(
            LazyWeight(source=split, dtype=ttnn.bfloat8_b, cache_dir_weight_name=(cache_dir, f"w_split_{i}"))
        )

    tt_model = LMHead1D(output_weights=lazy_weights)

    # Run TT model
    tt_input = LazyWeight(source=torch_input, dtype=ttnn.bfloat16)
    tt_output = tt_model.forward(tt_input)
    tt_output_torch = to_torch_auto_compose(tt_output)
    ttnn.SetDefaultDevice(None)

    # Shape checks
    assert tt_output_torch.shape[-2] == batch_rows, f"Expected batch_rows={batch_rows}, got {tt_output_torch.shape[-2]}"
    assert tt_output_torch.shape[-1] >= vocab_size, (
        f"Expected vocab cols>={vocab_size}, got {tt_output_torch.shape[-1]}. " f"num_devices={num_devices}"
    )

    # PCC against torch reference (trim to actual vocab_size, ignore padding zeros)
    ref_trimmed = ref_output[..., :vocab_size]
    tt_trimmed = tt_output_torch[..., :vocab_size]

    passing, pcc_message = comp_pcc(ref_trimmed, tt_trimmed, pcc)
    logger.info(comp_allclose(ref_trimmed, tt_trimmed))
    logger.info(f"LMHead1D vs reference: {pcc_message}")
    assert passing, f"LMHead1D output does not meet PCC {pcc}: {pcc_message}."
    logger.info(f"LMHead1D: PASSED for {hf_model_name} (mesh={mesh_shape}, devices={num_devices})")


# ============================================================================
# from_model_args backward compatibility test
# ============================================================================


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
def test_lm_head_1d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice):
    """
    Test LMHead1D.from_model_args produces valid output.
    """
    from models.tt_transformers.tt.model_config import ModelArgs

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=1, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    if model_args.is_galaxy:
        pytest.skip("LMHead1D test only runs on non-TG devices")

    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", None)

    def topology_aware_cache_path():
        return model_args.model_cache_path / f"tensor_cache_bfp8_{ttnn_mesh_device.shape}"

    max_columns = getattr(model_args, "max_columns_per_device_lm_head", 128256 // 4)

    tt_model = LMHead1D.from_model_args(
        mesh_device=ttnn_mesh_device,
        args=model_args,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=topology_aware_cache_path(),
        max_columns_per_device=max_columns,
        dtype=ttnn.bfloat8_b,
    )

    # Create input in the correct memory config for LM head (width-sharded matching lm_head_core_grid)
    batch_rows = TILE_SIZE * math.ceil(model_args.max_batch_size / TILE_SIZE)
    torch_input = torch.randn(1, 1, batch_rows, model_args.dim, dtype=torch.bfloat16)

    def _nearest_32(x):
        return math.ceil(x / 32) * 32

    input_memcfg = ttnn.create_sharded_memory_config(
        (
            batch_rows,
            _nearest_32(model_args.dim // model_args.lm_head_core_grid.num_cores),
        ),
        model_args.lm_head_core_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(
        torch_input,
        device=ttnn_mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_memcfg,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
    )

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = to_torch_auto_compose(tt_output)

    # Shape checks
    num_devices = ttnn_mesh_device.get_num_devices()
    assert tt_output_torch.shape[-2] == batch_rows, f"Expected batch_rows={batch_rows}, got {tt_output_torch.shape[-2]}"
    assert tt_output_torch.shape[-1] >= model_args.vocab_size, (
        f"Expected vocab cols>={model_args.vocab_size}, got {tt_output_torch.shape[-1]}. " f"num_devices={num_devices}"
    )

    # PCC against torch reference
    ref_weight = state_dict[f"{state_dict_prefix}output.weight"]
    ref_linear = torch.nn.Linear(model_args.dim, model_args.vocab_size, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_linear.weight.copy_(ref_weight.to(torch.bfloat16))
        ref_output = ref_linear(torch_input)

    ref_trimmed = ref_output[..., : model_args.vocab_size]
    tt_trimmed = tt_output_torch[..., : model_args.vocab_size]

    pcc_required = 0.999
    passing, pcc_message = comp_pcc(ref_trimmed, tt_trimmed, pcc_required)
    logger.info(comp_allclose(ref_trimmed, tt_trimmed))
    logger.info(f"LMHead1D (from_model_args) vs reference: {pcc_message}")
    assert passing, f"LMHead1D from_model_args PCC {pcc_required} not met: {pcc_message}."
    logger.info(f"LMHead1D.from_model_args: PASSED for {model_args.model_name}")
