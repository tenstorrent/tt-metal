# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""BSPM vs uniform-BFP4 output comparison for the deepseek_v3 demo (5 layers).

Runs the same decode forward pass twice through the deepseek_v3 RowBatchedModel:

  1. Baseline  — weights converted via the standard pipeline (uniform bfloat4_b).
  2. BSPM      — expert weights pre-quantized per-tile according to BitSculpt BSPM
                 assignments, then converted via the same standard pipeline.

The BSPM pre-quantization step simulates what CompressedTensor does on Blackhole
but without changing the kernel: each 32×32 expert-weight tile is quantized at its
BSPM-assigned precision (bfp8/bfp4/bfp2/zero) and dequantized back to float before
the standard bfloat4_b conversion runs.  The kernel thus sees a tensor whose values
are already at BSPM quality.

Environment variables
---------------------
DEEPSEEK_V3_HF_MODEL  : path to the local HF model dir (read by conftest).
BSPM_RESULTS_DIR      : BitSculpt results root, e.g. /localdev/.../bit_sculpt/results
BSPM_MODEL_NAME       : sub-directory under BSPM_RESULTS_DIR, e.g. deepseek-r1-0528
BSPM_VARIANT          : variant letter, default "B"
BSPM_BUDGET           : b/e budget as float, default 3.5

Run
---
    MESH_DEVICE=TG \
    DEEPSEEK_V3_HF_MODEL=/proj_sw/... \
    BSPM_RESULTS_DIR=/path/to/bit_sculpt/results \
    BSPM_MODEL_NAME=deepseek-r1-0528 \
    pytest models/demos/deepseek_v3/tests/test_bspm_demo.py -v
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel, get_fabric_config
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.hf_model_utils import default_stacked_dequantized_model_path
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    paged_caches_from_torch,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_LAYERS = 5  # Override: run only 5 layers instead of 61
MAX_SEQ_LEN = 128  # Trim KV cache tables for faster setup
BSPM_DEMO_WEIGHT_CACHE_VERSION = "v4"

# PCC thresholds
PCC_UNIFORM_VS_BSPM = 0.93  # direct comparison: uniform bfp4 vs BSPM pre-quantized
PCC_BSPM_VS_REF = 0.95  # BSPM vs float reference (lower than baseline due to aggressive tiles)
PCC_BASELINE_VS_REF = 0.97  # baseline vs float reference (same as test_model.py)

_STACKED_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts_stacked\.(?P<projection>\w+)\.weight$"
)


# ---------------------------------------------------------------------------
# Lazy state-dict wrapper that allows per-key overrides without materializing
# the full underlying mapping
# ---------------------------------------------------------------------------


class _OverrideStateDict(Mapping[str, torch.Tensor]):
    """Thin wrapper around a base Mapping that returns overridden values for
    specific keys and delegates all other key accesses to the base mapping.

    This avoids materializing the full (potentially multi-GB) state dict when
    only a small subset of keys (MoE expert weights) need to be modified.
    """

    def __init__(self, base, overrides: dict, *, _base_prefix: str = ""):
        self._base = base
        self._overrides = overrides
        self._base_prefix = _base_prefix

    def _full_key(self, key: str) -> str:
        return f"{self._base_prefix}{key}"

    def _get_base_stacked_tensor(self, key: str) -> torch.Tensor:
        view_with_prefix = getattr(self._base, "view_with_prefix", None)
        if callable(view_with_prefix) and key.startswith("experts_stacked."):
            return view_with_prefix("experts_stacked.")[key[len("experts_stacked.") :]]
        return self._base[key]

    def _maybe_get_stacked_override(self, key: str, full_key: str) -> torch.Tensor | None:
        match = _STACKED_EXPERT_KEY_RE.match(full_key)
        if match is None:
            return None

        try:
            stacked_tensor = self._get_base_stacked_tensor(key)
        except KeyError:
            return None
        override_slices: list[torch.Tensor] = []
        has_override = False
        for expert_idx in range(stacked_tensor.shape[0]):
            expert_full_key = (
                f"model.layers.{match.group('layer')}.mlp.experts.{expert_idx}.{match.group('projection')}.weight"
            )
            override_tensor = self._overrides.get(expert_full_key)
            if override_tensor is None:
                override_slices.append(stacked_tensor[expert_idx])
                continue
            has_override = True
            override_slices.append(override_tensor)

        if not has_override:
            return None
        return torch.stack(override_slices).contiguous()

    def __getitem__(self, key):
        full_key = self._full_key(key)
        if full_key in self._overrides:
            return self._overrides[full_key]

        stacked_override = self._maybe_get_stacked_override(key, full_key)
        if stacked_override is not None:
            return stacked_override

        return self._base[key]

    def __contains__(self, key):
        full_key = self._full_key(key)
        return full_key in self._overrides or key in self._base

    def __iter__(self):
        return iter(self._base)

    def __len__(self):
        return len(self._base)

    def keys(self):
        return self._base.keys()

    def items(self):
        for k in self._base:
            yield k, self[k]

    def values(self):
        for k in self._base:
            yield self[k]

    def view_with_prefix(self, prefix: str, num_layers: int | None = None) -> "_OverrideStateDict":
        view_with_prefix = getattr(self._base, "view_with_prefix", None)
        if callable(view_with_prefix):
            base_view = view_with_prefix(prefix, num_layers)
        else:
            base_view = sub_state_dict(self._base, prefix, num_layers)
        return _OverrideStateDict(base_view, self._overrides, _base_prefix=self._full_key(prefix))


class _FakeMeshDevice:
    shape = (1, 1)

    def get_num_devices(self) -> int:
        return 1


def _write_index(model_dir: Path, weight_map: dict[str, str]) -> None:
    index = {"metadata": {}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        pytest.fail(f"Expected checkpoint index at {index_path}, but it was not found.")
    return json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]


def _validate_stacked_checkpoint_for_bspm_demo(model_dir: Path, hf_config, *, num_layers: int = NUM_LAYERS) -> None:
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _validate_stacked_dequantized_checkpoint

    try:
        _validate_stacked_dequantized_checkpoint(model_dir, hf_config, num_layers=num_layers)
    except ValueError as e:
        pytest.fail(str(e))


def _resolve_stacked_checkpoint_for_bspm_demo(model_path: Path, hf_config, *, num_layers: int = NUM_LAYERS) -> Path:
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _resolve_stacked_dequantized_model_path

    try:
        resolved_path = _resolve_stacked_dequantized_model_path(model_path, hf_config)
    except FileNotFoundError:
        expected_path = default_stacked_dequantized_model_path(model_path)
        pytest.skip(
            f"Stacked dequantized checkpoint was not found at {expected_path}. "
            "Run save_dequantized_hf_checkpoint() first to create one."
        )
    except ValueError as e:
        pytest.fail(str(e))

    _validate_stacked_checkpoint_for_bspm_demo(resolved_path, hf_config, num_layers=num_layers)
    return resolved_path


def _load_required_bspm_codes(
    bspm_results_dir: Path,
    bspm_model_name: str,
    hf_config,
    *,
    variant: str,
    budget: float,
    num_layers: int,
    load_bspm_for_layer,
) -> dict[int, np.ndarray]:
    first_k_dense = getattr(hf_config, "first_k_dense_replace", 3)
    required_experts = hf_config.n_routed_experts
    codes_by_layer: dict[int, np.ndarray] = {}
    missing_files: list[Path] = []
    partial_layers: list[tuple[int, int]] = []

    for layer_idx in range(first_k_dense, num_layers):
        bspm_file = (
            bspm_results_dir
            / bspm_model_name
            / f"layer_{layer_idx}"
            / "precision_eval"
            / f"precision_map_{variant}_{budget:.1f}.bspm"
        )
        if not bspm_file.exists():
            missing_files.append(bspm_file)
            continue

        bspm_codes = load_bspm_for_layer(str(bspm_file), expected_n_experts=required_experts)["codes"]
        if bspm_codes.shape[0] != required_experts:
            partial_layers.append((layer_idx, bspm_codes.shape[0]))
            continue
        codes_by_layer[layer_idx] = bspm_codes

    if missing_files:
        sample_missing = ", ".join(str(path) for path in missing_files[:2])
        extra = "" if len(missing_files) <= 2 else f" and {len(missing_files) - 2} more"
        pytest.fail(f"Missing BSPM files required for this comparison: {sample_missing}{extra}.")

    if partial_layers:
        sample_partial = ", ".join(
            f"layer {layer_idx} covers {covered}/{required_experts} experts"
            for layer_idx, covered in partial_layers[:2]
        )
        extra = "" if len(partial_layers) <= 2 else f" and {len(partial_layers) - 2} more layers"
        pytest.fail(f"Incomplete BSPM coverage for this comparison: {sample_partial}{extra}.")

    return codes_by_layer


def _write_stacked_expert_checkpoint(
    model_dir: Path, layer_idx: int = 3, num_experts: int = 2
) -> dict[str, torch.Tensor]:
    model_dir.mkdir(parents=True, exist_ok=True)
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        f"model.layers.{layer_idx}.mlp.experts_stacked.{proj_name}.weight": (
            torch.arange(num_experts * 32 * 32, dtype=torch.float32).reshape(num_experts, 32, 32) + offset
        ).to(torch.bfloat16)
        for offset, proj_name in enumerate(("gate_proj", "down_proj", "up_proj"), start=1)
    }
    safetensors.torch.save_file(tensors, str(shard))
    _write_index(model_dir, {key: shard.name for key in tensors})
    return tensors


def _assert_experts_convert_weights_uses_stacked_inputs(
    monkeypatch: pytest.MonkeyPatch,
    state_dict,
    expected_tensors: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    from models.demos.deepseek_v3.tt import experts as experts_module

    requested_names: list[str] = []
    sharded_tensors: dict[str, torch.Tensor] = {}

    def fake_get_dequantized_tensor(state_dict, name, dtype):
        requested_names.append(name)
        return state_dict[name].to(dtype)

    def fake_shard_and_save(path: Path, tensor: torch.Tensor, **kwargs):
        sharded_tensors[path.name] = tensor.clone()
        return tensor

    monkeypatch.setattr(experts_module, "get_dequantized_tensor", fake_get_dequantized_tensor)
    monkeypatch.setattr(experts_module, "shard_and_save", fake_shard_and_save)
    monkeypatch.setattr(experts_module.Experts, "_warned_legacy_expert_checkpoint", False)

    experts_module.Experts.convert_weights(
        hf_config=types.SimpleNamespace(n_routed_experts=2),
        state_dicts=(state_dict,),
        output_path=output_path,
        mesh_device=_FakeMeshDevice(),
    )

    assert requested_names == ["gate_proj.weight", "down_proj.weight", "up_proj.weight"]
    assert torch.equal(sharded_tensors["w1_experts.input_tensor_b"], expected_tensors["gate_proj"].unsqueeze(0))
    assert torch.equal(sharded_tensors["w2_experts.input_tensor_b"], expected_tensors["down_proj"].unsqueeze(0))
    assert torch.equal(sharded_tensors["w3_experts.input_tensor_b"], expected_tensors["up_proj"].unsqueeze(0))


def test_bspm_state_dict_applies_overrides_to_stacked_expert_weights():
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict

    stacked_key = "model.layers.3.mlp.experts_stacked.gate_proj.weight"
    stacked_weight = torch.arange(2 * 32 * 32, dtype=torch.bfloat16).reshape(2, 32, 32)
    state_dict = {stacked_key: stacked_weight}
    hf_config = types.SimpleNamespace(first_k_dense_replace=3)

    wrapped_state_dict = _BsprnStateDict(
        state_dict,
        hf_config,
        Path("/tmp"),
        "unused",
        "B",
        3.5,
    )
    wrapped_state_dict._codes_cache[3] = np.array([[[0], [1], [1]], [[1], [1], [1]]], dtype=np.int32)
    wrapped_state_dict._qdq = lambda tensor, mant_bits: tensor + mant_bits

    result = wrapped_state_dict[stacked_key]

    assert torch.equal(result[0], stacked_weight[0] + 7)
    assert torch.equal(result[1], stacked_weight[1])


def test_bspm_state_dict_leaves_uncovered_stacked_experts_unmodified():
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict

    stacked_key = "model.layers.3.mlp.experts_stacked.gate_proj.weight"
    stacked_weight = torch.arange(3 * 32 * 32, dtype=torch.bfloat16).reshape(3, 32, 32)
    state_dict = {stacked_key: stacked_weight}
    hf_config = types.SimpleNamespace(first_k_dense_replace=3)

    wrapped_state_dict = _BsprnStateDict(
        state_dict,
        hf_config,
        Path("/tmp"),
        "unused",
        "B",
        3.5,
    )
    wrapped_state_dict._codes_cache[3] = np.array([[[0], [1], [1]], [[0], [1], [1]]], dtype=np.int32)
    wrapped_state_dict._qdq = lambda tensor, mant_bits: tensor + mant_bits

    result = wrapped_state_dict[stacked_key]

    assert torch.equal(result[0], stacked_weight[0] + 7)
    assert torch.equal(result[1], stacked_weight[1] + 7)
    assert torch.equal(result[2], stacked_weight[2])


def test_bspm_state_dict_applies_overrides_to_legacy_expert_weights():
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict

    weight_key = "model.layers.3.mlp.experts.0.gate_proj.weight"
    weight = torch.arange(32 * 32, dtype=torch.bfloat16).reshape(32, 32)
    state_dict = {weight_key: weight}
    hf_config = types.SimpleNamespace(first_k_dense_replace=3)

    wrapped_state_dict = _BsprnStateDict(
        state_dict,
        hf_config,
        Path("/tmp"),
        "unused",
        "B",
        3.5,
    )
    wrapped_state_dict._codes_cache[3] = np.array([[[0], [1], [1]]], dtype=np.int32)
    wrapped_state_dict._qdq = lambda tensor, mant_bits: tensor + mant_bits

    result = wrapped_state_dict[weight_key]

    assert torch.equal(result, weight + 7)


def test_bspm_state_dict_leaves_uncovered_legacy_expert_weights_unmodified():
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict

    weight_key = "model.layers.3.mlp.experts.1.gate_proj.weight"
    weight = torch.arange(32 * 32, dtype=torch.bfloat16).reshape(32, 32)
    state_dict = {weight_key: weight}
    hf_config = types.SimpleNamespace(first_k_dense_replace=3)

    wrapped_state_dict = _BsprnStateDict(
        state_dict,
        hf_config,
        Path("/tmp"),
        "unused",
        "B",
        3.5,
    )
    wrapped_state_dict._codes_cache[3] = np.array([[[0], [1], [1]]], dtype=np.int32)
    wrapped_state_dict._qdq = lambda tensor, mant_bits: tensor + mant_bits

    result = wrapped_state_dict[weight_key]

    assert torch.equal(result, weight)


def test_bspm_state_dict_validate_bspm_files_rejects_missing_files_when_required(tmp_path: Path):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict

    results_dir = tmp_path / "results"
    present_file = results_dir / "demo-model" / "layer_3" / "precision_eval" / "precision_map_B_3.5.bspm"
    present_file.parent.mkdir(parents=True, exist_ok=True)
    present_file.write_bytes(b"present")

    wrapped_state_dict = _BsprnStateDict(
        {},
        types.SimpleNamespace(first_k_dense_replace=3, n_routed_experts=2),
        results_dir,
        "demo-model",
        "B",
        3.5,
        _load_bspm=lambda _path, expected_n_experts=None: {"codes": np.zeros((2, 3, 1), dtype=np.int32)},
    )

    with pytest.raises(ValueError, match="Missing BSPM files required for export"):
        wrapped_state_dict.validate_bspm_files(
            types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=5, n_routed_experts=2),
            require_complete=True,
        )


def test_bspm_state_dict_validate_bspm_files_rejects_partial_coverage_when_required(tmp_path: Path):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict

    results_dir = tmp_path / "results"
    for layer_idx in (3, 4):
        bspm_file = results_dir / "demo-model" / f"layer_{layer_idx}" / "precision_eval" / "precision_map_B_3.5.bspm"
        bspm_file.parent.mkdir(parents=True, exist_ok=True)
        bspm_file.write_bytes(b"present")

    wrapped_state_dict = _BsprnStateDict(
        {},
        types.SimpleNamespace(first_k_dense_replace=3, n_routed_experts=2),
        results_dir,
        "demo-model",
        "B",
        3.5,
        _load_bspm=lambda _path, expected_n_experts=None: {"codes": np.zeros((1, 3, 1), dtype=np.int32)},
    )

    with pytest.raises(ValueError, match="Incomplete BSPM coverage"):
        wrapped_state_dict.validate_bspm_files(
            types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=5, n_routed_experts=2),
            require_complete=True,
        )


def test_bspm_state_dict_validate_bspm_files_rejects_excess_coverage_when_required(tmp_path: Path):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict

    results_dir = tmp_path / "results"
    for layer_idx in (3, 4):
        bspm_file = results_dir / "demo-model" / f"layer_{layer_idx}" / "precision_eval" / "precision_map_B_3.5.bspm"
        bspm_file.parent.mkdir(parents=True, exist_ok=True)
        bspm_file.write_bytes(b"present")

    wrapped_state_dict = _BsprnStateDict(
        {},
        types.SimpleNamespace(first_k_dense_replace=3, n_routed_experts=2),
        results_dir,
        "demo-model",
        "B",
        3.5,
        _load_bspm=lambda _path, expected_n_experts=None: {"codes": np.zeros((3, 3, 1), dtype=np.int32)},
    )

    with pytest.raises(ValueError, match="Incomplete BSPM coverage"):
        wrapped_state_dict.validate_bspm_files(
            types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=5, n_routed_experts=2),
            require_complete=True,
        )


def test_bspm_state_dict_validate_bspm_files_can_skip_codes_cache_population(tmp_path: Path):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict

    results_dir = tmp_path / "results"
    for layer_idx in (3, 4):
        bspm_file = results_dir / "demo-model" / f"layer_{layer_idx}" / "precision_eval" / "precision_map_B_3.5.bspm"
        bspm_file.parent.mkdir(parents=True, exist_ok=True)
        bspm_file.write_bytes(b"present")

    load_calls: list[tuple[str, int | None]] = []

    def fake_load_bspm(path, expected_n_experts=None):
        load_calls.append((path, expected_n_experts))
        return {"codes": np.zeros((2, 3, 1), dtype=np.int32)}

    wrapped_state_dict = _BsprnStateDict(
        {},
        types.SimpleNamespace(first_k_dense_replace=3, n_routed_experts=2),
        results_dir,
        "demo-model",
        "B",
        3.5,
        _load_bspm=fake_load_bspm,
    )

    missing = wrapped_state_dict.validate_bspm_files(
        types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=5, n_routed_experts=2),
        require_complete=True,
        cache_results=False,
    )

    assert missing == 0
    assert len(load_calls) == 2
    assert wrapped_state_dict._codes_cache == {}


def test_resolve_stacked_dequantized_model_path_requires_stacked_checkpoint(tmp_path: Path):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _resolve_stacked_dequantized_model_path

    source_model_path = tmp_path / "DeepSeek-R1-0528"
    source_model_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Stacked dequantized DeepSeek checkpoint not found"):
        _resolve_stacked_dequantized_model_path(
            source_model_path,
            types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=4, n_routed_experts=2),
        )


def test_resolve_stacked_dequantized_model_path_rejects_mislabeled_legacy_checkpoint(tmp_path: Path):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _resolve_stacked_dequantized_model_path

    stacked_path = tmp_path / "DeepSeek-R1-0528-dequantized-stacked"
    stacked_path.mkdir(parents=True, exist_ok=True)
    shard = stacked_path / "model-00001-of-00001.safetensors"
    legacy_tensors = {
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.down_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.up_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(legacy_tensors, str(shard))
    _write_index(stacked_path, {key: shard.name for key in legacy_tensors})

    with pytest.raises(ValueError, match="not a complete stacked DeepSeek checkpoint"):
        _resolve_stacked_dequantized_model_path(
            stacked_path,
            types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=4, n_routed_experts=2),
        )


def test_resolve_stacked_dequantized_model_path_accepts_custom_stacked_checkpoint_dir(tmp_path: Path):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _resolve_stacked_dequantized_model_path

    custom_stacked_path = tmp_path / "custom-stacked-export"
    _write_stacked_expert_checkpoint(custom_stacked_path)

    resolved = _resolve_stacked_dequantized_model_path(
        custom_stacked_path,
        types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=4, n_routed_experts=2),
    )

    assert resolved == custom_stacked_path.resolve()


def test_resolve_stacked_dequantized_model_path_rejects_truncated_stacked_checkpoint(tmp_path: Path):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _resolve_stacked_dequantized_model_path

    truncated_stacked_path = tmp_path / "custom-stacked-export"
    _write_stacked_expert_checkpoint(truncated_stacked_path, num_experts=1)

    with pytest.raises(ValueError, match="has invalid stacked expert tensors"):
        _resolve_stacked_dequantized_model_path(
            truncated_stacked_path,
            types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=4, n_routed_experts=2),
        )


def test_resolve_stacked_dequantized_model_path_rejects_invalid_explicit_dequantized_checkpoint(tmp_path: Path):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _resolve_stacked_dequantized_model_path

    legacy_path = tmp_path / "DeepSeek-R1-0528-dequantized"
    legacy_path.mkdir(parents=True, exist_ok=True)
    shard = legacy_path / "model-00001-of-00001.safetensors"
    legacy_tensors = {
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.down_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.up_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(legacy_tensors, str(shard))
    _write_index(legacy_path, {key: shard.name for key in legacy_tensors})
    _write_stacked_expert_checkpoint(tmp_path / "DeepSeek-R1-0528-dequantized-stacked")

    with pytest.raises(ValueError, match="not a complete stacked DeepSeek checkpoint"):
        _resolve_stacked_dequantized_model_path(
            legacy_path,
            types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=4, n_routed_experts=2),
        )


def test_resolve_stacked_dequantized_model_path_rejects_invalid_explicit_custom_checkpoint_dir(tmp_path: Path):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _resolve_stacked_dequantized_model_path

    custom_path = tmp_path / "custom-stacked-export"
    custom_path.mkdir(parents=True, exist_ok=True)
    shard = custom_path / "model-00001-of-00001.safetensors"
    legacy_tensors = {
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.down_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.up_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(legacy_tensors, str(shard))
    _write_index(custom_path, {key: shard.name for key in legacy_tensors})
    _write_stacked_expert_checkpoint(tmp_path / "custom-stacked-export-dequantized-stacked")

    with pytest.raises(ValueError, match="not a complete stacked DeepSeek checkpoint"):
        _resolve_stacked_dequantized_model_path(
            custom_path,
            types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=4, n_routed_experts=2),
        )


def test_resolve_stacked_checkpoint_for_bspm_demo_uses_explicit_custom_stacked_path(tmp_path: Path):
    custom_stacked_path = tmp_path / "custom-stacked-export"
    _write_stacked_expert_checkpoint(custom_stacked_path)

    resolved = _resolve_stacked_checkpoint_for_bspm_demo(
        custom_stacked_path,
        types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=4, n_routed_experts=2),
        num_layers=4,
    )

    assert resolved == custom_stacked_path.resolve()


def test_resolve_stacked_checkpoint_for_bspm_demo_rejects_invalid_explicit_dequantized_path(tmp_path: Path):
    legacy_path = tmp_path / "DeepSeek-R1-0528-dequantized"
    legacy_path.mkdir(parents=True, exist_ok=True)
    shard = legacy_path / "model-00001-of-00001.safetensors"
    legacy_tensors = {
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.down_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.up_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(legacy_tensors, str(shard))
    _write_index(legacy_path, {key: shard.name for key in legacy_tensors})
    _write_stacked_expert_checkpoint(tmp_path / "DeepSeek-R1-0528-dequantized-stacked")

    with pytest.raises(pytest.fail.Exception, match="not a complete stacked DeepSeek checkpoint"):
        _resolve_stacked_checkpoint_for_bspm_demo(
            legacy_path,
            types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=4, n_routed_experts=2),
            num_layers=4,
        )


def test_resolve_stacked_checkpoint_for_bspm_demo_rejects_invalid_explicit_custom_path(tmp_path: Path):
    custom_path = tmp_path / "custom-stacked-export"
    custom_path.mkdir(parents=True, exist_ok=True)
    shard = custom_path / "model-00001-of-00001.safetensors"
    legacy_tensors = {
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.down_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.up_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(legacy_tensors, str(shard))
    _write_index(custom_path, {key: shard.name for key in legacy_tensors})
    _write_stacked_expert_checkpoint(tmp_path / "custom-stacked-export-dequantized-stacked")

    with pytest.raises(pytest.fail.Exception, match="not a complete stacked DeepSeek checkpoint"):
        _resolve_stacked_checkpoint_for_bspm_demo(
            custom_path,
            types.SimpleNamespace(first_k_dense_replace=3, num_hidden_layers=4, n_routed_experts=2),
            num_layers=4,
        )


def test_override_state_dict_prefix_view_preserves_stacked_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_dir = tmp_path / "model"
    checkpoint_tensors = _write_stacked_expert_checkpoint(model_dir)
    state_dict = LazyStateDict(model_dir)
    overrides: dict[str, torch.Tensor] = {}
    expected_tensors = {}
    for proj_name in ("gate_proj", "down_proj", "up_proj"):
        stacked_key = f"model.layers.3.mlp.experts_stacked.{proj_name}.weight"
        stacked_tensor = checkpoint_tensors[stacked_key].clone()
        overrides[f"model.layers.3.mlp.experts.0.{proj_name}.weight"] = stacked_tensor[0] + 11
        overrides[f"model.layers.3.mlp.experts.1.{proj_name}.weight"] = stacked_tensor[1] + 13
        stacked_tensor[0] = stacked_tensor[0] + 11
        stacked_tensor[1] = stacked_tensor[1] + 13
        expected_tensors[proj_name] = stacked_tensor

    wrapped_state_dict = _OverrideStateDict(state_dict, overrides)
    mlp_state_dict = sub_state_dict(wrapped_state_dict, "model.layers.3.mlp.")

    _assert_experts_convert_weights_uses_stacked_inputs(
        monkeypatch,
        mlp_state_dict,
        expected_tensors,
        tmp_path / "override_out",
    )


def test_bspm_state_dict_prefix_view_preserves_stacked_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict

    model_dir = tmp_path / "model"
    checkpoint_tensors = _write_stacked_expert_checkpoint(model_dir)
    wrapped_state_dict = _BsprnStateDict(
        LazyStateDict(model_dir),
        types.SimpleNamespace(first_k_dense_replace=3),
        Path("/tmp"),
        "unused",
        "B",
        3.5,
        _require_complete=True,
    )
    wrapped_state_dict._codes_cache[3] = np.array(
        [
            [[0], [0], [0]],
            [[1], [1], [1]],
        ],
        dtype=np.int32,
    )
    wrapped_state_dict._qdq = lambda tensor, mant_bits: tensor + mant_bits

    mlp_state_dict = sub_state_dict(wrapped_state_dict, "model.layers.3.mlp.")
    expected_tensors = {
        proj_name: checkpoint_tensors[f"model.layers.3.mlp.experts_stacked.{proj_name}.weight"].clone()
        for proj_name in ("gate_proj", "down_proj", "up_proj")
    }
    for tensor in expected_tensors.values():
        tensor[0] = tensor[0] + 7

    _assert_experts_convert_weights_uses_stacked_inputs(
        monkeypatch,
        mlp_state_dict,
        expected_tensors,
        tmp_path / "bspm_out",
    )


def test_experts_convert_weights_rejects_truncated_stacked_checkpoint(tmp_path: Path):
    model_dir = tmp_path / "model"
    _write_stacked_expert_checkpoint(model_dir, num_experts=1)
    mlp_state_dict = sub_state_dict(LazyStateDict(model_dir), "model.layers.3.mlp.")

    with pytest.raises(
        ValueError,
        match="Expected stacked expert weight 'experts_stacked.gate_proj.weight' to contain 2 experts, got 1",
    ):
        from models.demos.deepseek_v3.tt.experts import Experts

        Experts.convert_weights(
            hf_config=types.SimpleNamespace(n_routed_experts=2),
            state_dicts=(mlp_state_dict,),
            output_path=tmp_path / "out",
            mesh_device=_FakeMeshDevice(),
        )


def test_validate_stacked_checkpoint_for_bspm_demo_rejects_legacy_checkpoint(tmp_path: Path):
    model_dir = tmp_path / "legacy-model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shard = model_dir / "model-00001-of-00001.safetensors"
    legacy_tensors = {
        "model.layers.3.mlp.experts.0.gate_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.down_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
        "model.layers.3.mlp.experts.0.up_proj.weight": torch.ones((32, 32), dtype=torch.bfloat16),
    }
    safetensors.torch.save_file(legacy_tensors, str(shard))
    _write_index(model_dir, {key: shard.name for key in legacy_tensors})

    with pytest.raises(pytest.fail.Exception, match="not a complete stacked DeepSeek checkpoint"):
        _validate_stacked_checkpoint_for_bspm_demo(
            model_dir,
            types.SimpleNamespace(first_k_dense_replace=3, n_routed_experts=2),
            num_layers=5,
        )


def test_load_required_bspm_codes_rejects_missing_layer_files(tmp_path: Path):
    results_dir = tmp_path / "results"
    present_file = results_dir / "demo-model" / "layer_3" / "precision_eval" / "precision_map_B_3.5.bspm"
    present_file.parent.mkdir(parents=True, exist_ok=True)
    present_file.write_bytes(b"present")

    def fake_loader(_path: str, expected_n_experts: int | None = None) -> dict[str, np.ndarray]:
        return {"codes": np.zeros((2, 3, 1), dtype=np.int32)}

    with pytest.raises(pytest.fail.Exception, match="Missing BSPM files required for this comparison"):
        _load_required_bspm_codes(
            results_dir,
            "demo-model",
            types.SimpleNamespace(first_k_dense_replace=3, n_routed_experts=2),
            variant="B",
            budget=3.5,
            num_layers=5,
            load_bspm_for_layer=fake_loader,
        )


def test_load_required_bspm_codes_rejects_partial_expert_coverage(tmp_path: Path):
    results_dir = tmp_path / "results"
    for layer_idx in (3, 4):
        bspm_file = results_dir / "demo-model" / f"layer_{layer_idx}" / "precision_eval" / "precision_map_B_3.5.bspm"
        bspm_file.parent.mkdir(parents=True, exist_ok=True)
        bspm_file.write_bytes(b"present")

    def fake_loader(_path: str, expected_n_experts: int | None = None) -> dict[str, np.ndarray]:
        return {"codes": np.zeros((1, 3, 1), dtype=np.int32)}

    with pytest.raises(pytest.fail.Exception, match="Incomplete BSPM coverage for this comparison"):
        _load_required_bspm_codes(
            results_dir,
            "demo-model",
            types.SimpleNamespace(first_k_dense_replace=3, n_routed_experts=2),
            variant="B",
            budget=3.5,
            num_layers=5,
            load_bspm_for_layer=fake_loader,
        )


def test_load_required_bspm_codes_rejects_excess_expert_coverage(tmp_path: Path):
    results_dir = tmp_path / "results"
    for layer_idx in (3, 4):
        bspm_file = results_dir / "demo-model" / f"layer_{layer_idx}" / "precision_eval" / "precision_map_B_3.5.bspm"
        bspm_file.parent.mkdir(parents=True, exist_ok=True)
        bspm_file.write_bytes(b"present")

    def fake_loader(_path: str, expected_n_experts: int | None = None) -> dict[str, np.ndarray]:
        return {"codes": np.zeros((3, 3, 1), dtype=np.int32)}

    with pytest.raises(pytest.fail.Exception, match="Incomplete BSPM coverage for this comparison"):
        _load_required_bspm_codes(
            results_dir,
            "demo-model",
            types.SimpleNamespace(first_k_dense_replace=3, n_routed_experts=2),
            variant="B",
            budget=3.5,
            num_layers=5,
            load_bspm_for_layer=fake_loader,
        )


# ---------------------------------------------------------------------------
# BSPM preprocessing helper
# ---------------------------------------------------------------------------


def _preprocess_experts_with_bspm(
    state_dict,
    hf_config,
    bspm_results_dir: Path,
    bspm_model_name: str,
    variant: str = "B",
    budget: float = 3.5,
    num_layers: int = NUM_LAYERS,
) -> dict:
    """Return the production lazy BSPM wrapper after validating required inputs."""
    from models.demos.deepseek_v3.scripts.convert_bspm_weights import _BsprnStateDict
    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_layer

    bspm_codes_by_layer = _load_required_bspm_codes(
        bspm_results_dir,
        bspm_model_name,
        hf_config,
        variant=variant,
        budget=budget,
        num_layers=num_layers,
        load_bspm_for_layer=load_bspm_for_layer,
    )
    wrapped_state_dict = _BsprnStateDict(
        state_dict,
        hf_config,
        bspm_results_dir,
        bspm_model_name,
        variant,
        budget,
        _require_complete=True,
    )
    wrapped_state_dict._codes_cache.update(bspm_codes_by_layer)
    logger.info(f"BSPM pre-quantization prepared for {len(bspm_codes_by_layer)} MoE layers")
    return wrapped_state_dict


# ---------------------------------------------------------------------------
# Shared decode-step runner
# ---------------------------------------------------------------------------


def _run_one_decode_step(
    hf_config,
    mesh_device,
    ccl,
    paged_config,
    weight_config,
    torch_input: torch.Tensor,  # (seq=1, batch)
    position_ids: torch.Tensor,  # (batch,)
    torch_page_table: torch.Tensor,
) -> torch.Tensor:
    """Run one RowBatchedModel decode forward step and return logits on CPU."""
    dp_factor = mesh_device.shape[1]
    batches_per_device = USERS_PER_ROW // dp_factor

    # Empty KV caches (position 0, no prior context)
    cache_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
    batch_size = int(position_ids.shape[0])
    empty_caches = tuple(
        torch.zeros((batch_size, 1, 0, cache_dim), dtype=torch.bfloat16) for _ in range(hf_config.num_hidden_layers)
    )

    mapping = torch_page_table
    paged_input_caches, _ = paged_caches_from_torch(
        empty_caches,
        tuple(mesh_device.shape),
        paged_config,
        user_id=None,
        mappings=tuple(mapping for _ in range(hf_config.num_hidden_layers)),
    )

    tt_page_tables = tuple(
        MLA2D.create_page_table(page_table=mapping, paged_config=paged_config, mesh_device=mesh_device)
        for _ in range(hf_config.num_hidden_layers)
    )

    model_config = get_model_config(RowBatchedModel, "decode", hf_config, mesh_device, batch_size_per_row=USERS_PER_ROW)
    model_state = RowBatchedModel.create_state(hf_config, paged_config, mesh_device, ccl, paged_input_caches)
    model_shared_state = RowBatchedModel.create_shared_state(hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    position_ids_tt = ttnn.from_torch(
        position_ids,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        dtype=ttnn.int32,
    )
    rope_tensors = get_rope_tensors(hf_config, USERS_PER_ROW, 1, position_ids, mesh_device)

    tt_output = RowBatchedModel.forward_decode(tt_input, position_ids_tt, run_config, rope_tensors, tt_page_tables)
    ttnn.synchronize_device(mesh_device)

    logits = (
        ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        )
        .cpu()
        .float()
    )

    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)
    ttnn.deallocate(position_ids_tt)
    return logits


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": get_fabric_config()}],
    indirect=True,
)
def test_bspm_vs_uniform_5layers_decode(
    device_params,
    mesh_device,
    hf_config,
    model_path,
    cache_path,
    ccl,
    force_recalculate_weight_config,
):
    """Run the deepseek_v3 demo (5 layers, 1 decode step) with uniform bfloat4_b
    weights and with BSPM pre-quantized weights; compare output logits.

    Both runs use identical random inputs and page tables so the comparison
    isolates the effect of BSPM-assigned per-tile precision.

    Expected result
    ---------------
    - Baseline PCC vs BSPM  > 0.93  (small quality degradation from mixed precision)
    - If BSPM is well-calibrated, most token predictions stay the same; heavily
      compressed expert tiles contribute minimal cross-attention weight so the
      output distribution shifts only slightly.
    """
    # ── 5-layer config ──────────────────────────────────────────────────────
    hf_config_5 = deepcopy(hf_config)
    hf_config_5.num_hidden_layers = NUM_LAYERS
    hf_config_5.max_seq_len = MAX_SEQ_LEN

    # Load from the dequantized (BF16) checkpoint — the raw HF checkpoint for
    # DeepSeek-R1-0528 uses float8_e4m3fn; convert_weights requires BF16 tensors.
    deq_path = _resolve_stacked_checkpoint_for_bspm_demo(model_path, hf_config_5, num_layers=NUM_LAYERS)
    deq_state_dict_5 = sub_state_dict(LazyStateDict(deq_path), "", NUM_LAYERS)
    baseline_weight_cache_identity = f"bspm_demo_savedweight_{BSPM_DEMO_WEIGHT_CACHE_VERSION}_{deq_path.resolve()}"
    rebuild_savedweight_cache = True

    # ── Shared random decode inputs ─────────────────────────────────────────
    dp_factor = mesh_device.shape[1]
    batch_size = USERS_PER_ROW * mesh_device.shape[0]
    paged_config = MLA2D.get_valid_paged_config(hf_config_5.max_seq_len, USERS_PER_ROW, dp_factor)

    # Random decode input: position 0, one token per user
    position_ids = torch.zeros(batch_size, dtype=torch.long)
    torch_input = torch.randint(0, hf_config_5.vocab_size - 1, (batch_size, 1), dtype=torch.long).T  # (1, batch)

    # Shared page table (same structure for both runs)
    batches_per_device = USERS_PER_ROW // dp_factor
    blocks_per_batch = paged_config.max_num_blocks // batches_per_device
    torch_page_table = torch.arange(paged_config.max_num_blocks, dtype=torch.int32).reshape(
        batches_per_device, blocks_per_batch
    )

    # ── Run 1: uniform bfloat4_b (baseline) ─────────────────────────────────
    logger.info("=== Run 1: uniform bfloat4_b (baseline) ===")
    # Keep the expensive Galaxy decode comparison on the historical SavedWeight
    # path; the default direct conversion path is covered by focused unit tests.
    weight_cfg_uniform = get_test_weight_config(
        RowBatchedModel,
        hf_config_5,
        (deq_state_dict_5,),
        cache_path,
        mesh_device,
        rebuild_savedweight_cache or force_recalculate_weight_config,
        test_name="test_bspm_demo",
        layer_id=f"uniform_{NUM_LAYERS}layers",
        real_weights=True,
        prefer_legacy_weight_cache=True,
        cache_identity=baseline_weight_cache_identity,
    )
    logits_uniform = _run_one_decode_step(
        hf_config_5,
        mesh_device,
        ccl,
        paged_config,
        weight_cfg_uniform,
        torch_input,
        position_ids,
        torch_page_table,
    )
    logger.info(f"Baseline logits shape: {logits_uniform.shape}")

    # ── Run 2: BSPM pre-quantized ────────────────────────────────────────────
    bspm_results_dir_env = os.environ.get("BSPM_RESULTS_DIR")
    bspm_model_name = os.environ.get("BSPM_MODEL_NAME", "")
    bspm_variant = os.environ.get("BSPM_VARIANT", "B")
    bspm_budget = float(os.environ.get("BSPM_BUDGET", "3.5"))

    if not bspm_results_dir_env or not bspm_model_name:
        pytest.skip(
            "BSPM_RESULTS_DIR and BSPM_MODEL_NAME must be set to compare BSPM weights. "
            "Baseline run completed successfully."
        )
    bspm_results_dir = Path(bspm_results_dir_env).expanduser()
    if not bspm_results_dir.exists():
        pytest.fail(f"BSPM_RESULTS_DIR={bspm_results_dir} does not exist.")
    bspm_results_dir = bspm_results_dir.resolve()
    bspm_weight_cache_identity = f"{baseline_weight_cache_identity}_{bspm_results_dir}"

    logger.info(f"=== Run 2: BSPM pre-quantized " f"({bspm_model_name}, variant {bspm_variant}, {bspm_budget} b/e) ===")
    bspm_state_dict = _preprocess_experts_with_bspm(
        deq_state_dict_5,
        hf_config_5,
        bspm_results_dir,
        bspm_model_name,
        variant=bspm_variant,
        budget=bspm_budget,
        num_layers=NUM_LAYERS,
    )
    weight_cfg_bspm = get_test_weight_config(
        RowBatchedModel,
        hf_config_5,
        (bspm_state_dict,),
        cache_path,
        mesh_device,
        rebuild_savedweight_cache or force_recalculate_weight_config,
        test_name="test_bspm_demo",
        layer_id=f"bspm_{NUM_LAYERS}layers_{bspm_model_name}_{bspm_variant}_{bspm_budget}",
        real_weights=True,
        prefer_legacy_weight_cache=True,
        cache_identity=bspm_weight_cache_identity,
    )
    logits_bspm = _run_one_decode_step(
        hf_config_5,
        mesh_device,
        ccl,
        paged_config,
        weight_cfg_bspm,
        torch_input,
        position_ids,
        torch_page_table,
    )
    logger.info(f"BSPM logits shape: {logits_bspm.shape}")

    # ── Compare ──────────────────────────────────────────────────────────────
    passing, pcc_msg = comp_pcc(logits_uniform, logits_bspm, PCC_UNIFORM_VS_BSPM)
    logger.info(f"\n{'='*60}")
    logger.info(f"Uniform bfp4  vs  BSPM ({bspm_variant} {bspm_budget} b/e):  {pcc_msg}")

    # Token-level agreement (greedy argmax)
    tokens_uniform = logits_uniform.argmax(dim=-1)  # (1, batch)
    tokens_bspm = logits_bspm.argmax(dim=-1)
    match_rate = (tokens_uniform == tokens_bspm).float().mean().item()
    logger.info(f"Top-1 token match rate (greedy): {match_rate:.3f}")
    logger.info(f"{'='*60}")

    assert passing, (
        f"BSPM logits PCC too low vs uniform baseline: {pcc_msg}. "
        f"Expected PCC > {PCC_UNIFORM_VS_BSPM}. "
        f"This may indicate the BSPM allocation is too aggressive for tt-metal's quantizer "
        f"or that the tile orientation in _preprocess_experts_with_bspm() is incorrect."
    )
    logger.info(
        f"PASSED: BSPM ({bspm_variant} {bspm_budget} b/e) output matches uniform baseline "
        f"with PCC > {PCC_UNIFORM_VS_BSPM}. Top-1 token match: {match_rate:.3f}"
    )
