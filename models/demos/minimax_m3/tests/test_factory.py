# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import json
import os

import pytest
import torch
from transformers import AutoConfig

import ttnn

from ..config import MeshConfig
from ..tt.ccl import CCLManager
from ..tt.model_config import ModelArgs
from ..utils.general_utils import get_default_num_links

# Bundled MiniMax-M3 config.json (dims only — no model code is vendored). HF-reference
# tests that need the actual modeling classes load them from a real checkpoint via
# HF_MODEL (see requires_hf_reference below).
_CONFIG_JSON = os.path.join(os.path.dirname(__file__), "..", "configs", "MiniMax-M3", "config.json")


def minimax_config_dims() -> dict:
    """Load MiniMax-M3 dimension constants from the bundled config.json (no HF needed)."""
    with open(_CONFIG_JSON) as f:
        return json.load(f)


_HF_MODEL = os.getenv("HF_MODEL")
# Tests that compare against the HuggingFace reference need the model's modeling code,
# which ships WITH the checkpoint (loaded at runtime via trust_remote_code) — we do NOT
# vendor it into the repo. Point HF_MODEL at a downloaded MiniMax-M3 checkpoint to run them.
requires_hf_reference = pytest.mark.skipif(
    not (_HF_MODEL and os.path.isdir(_HF_MODEL)),
    reason="set HF_MODEL to a downloaded MiniMax-M3 checkpoint (carries modeling_minimax_m3.py) to run HF-reference tests",
)


def hf_model_path() -> str:
    return _HF_MODEL


class TestFactory:
    # Common test configurations
    MESH_SHAPES = {"1x4": (1, 4), "4x8": (4, 8), "1x8": (1, 8), "4x4": (4, 4), "2x4": (2, 4), "1x1": (1, 1)}

    BATCH_SEQ_CONFIGS = [
        (1, 1),  # Single token
        (1, 32),  # Small sequence
        (1, 128),  # Medium sequence
    ]

    @staticmethod
    def setup_test(mesh_device, use_real_weights=True, dtype=ttnn.bfloat8_b):
        """Universal test setup - replaces all the duplicated setup code"""

        # Use mesh_device as-is (already created by conftest.py fixture)
        mesh_shape = mesh_device.shape

        # Setup ModelArgs (no import-time loading)
        model_args = ModelArgs(mesh_device=mesh_device, dummy_weights=not use_real_weights)

        # Setup mesh config using actual mesh shape

        mesh_config = MeshConfig(mesh_shape, tp=mesh_shape[1])

        # Setup CCL
        ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device))

        config = AutoConfig.from_pretrained(model_args.model_path, trust_remote_code=True)

        # Only cache tensors to disk when using real weights; with dummy (random) weights
        # there is nothing to persist and the model_path may not be a writable local directory.
        tensor_cache_path = model_args.weight_cache_path(dtype) if use_real_weights else None

        return {
            "mesh_device": mesh_device,
            "model_args": model_args,
            "mesh_config": mesh_config,
            "ccl_manager": ccl_manager,
            "config": config,
            "dtype": dtype,
            "tensor_cache_path": tensor_cache_path,
        }


def parametrize_mesh_with_fabric(mesh_shapes=None, linear_fabric=False):
    """Universal mesh + fabric parametrization for minimax_m3 tests.

    Generates a paired ``(mesh_device, device_params)`` parametrize. Each
    case opens a mesh of the requested shape directly (no submesh carving
    in test bodies) and configures the appropriate fabric for that shape.
    Each parametrize case has a single id like ``1x1`` / ``1x2`` / ``1x4`` /
    ``1x8`` / ``4x8``, so ``pytest -k 1x2`` (or ``-k 4x8``) filters cleanly
    without dual-id confusion from a separate inner ``mesh_shape`` parametrize.

    Auto-filters to the shapes that fit on the current system. Default
    shapes: (1,1) single card, (1,2) 2xP150, (1,4) QuietBox 2 (2xP300),
    (1,8) LoudBox / T3K, (4,8) Galaxy. Pass an explicit ``mesh_shapes`` list
    to override (useful for tests that only make sense at one TP factor).

    Fabric: ``(1,1)`` disables fabric (no inter-chip topology to ring
    around). Multi-device shapes use ``FABRIC_1D_RING`` — minimax_m3's CCL
    operations (reduce_scatter, all_gather, all_reduce) all use the ring
    topology.

    When ``CI=true`` is set in the environment, only the largest mesh shape
    that fits on the current system is parametrized. This lets one yaml
    entry target multiple SKUs without per-SKU ``-k "1xN"`` filters: each
    runner picks the largest mesh its device count supports. Manual / non-CI
    invocations are unchanged (all-shapes-that-fit, ``-k`` available for fast
    iteration).

    Usage:
        @parametrize_mesh_with_fabric()              # all shapes that fit
        @parametrize_mesh_with_fabric([(1, 8)])      # 1x8 only

        pytest -k 1x1   # single card             (manual / non-CI)
        pytest -k 1x2   # 2xP150                  (manual / non-CI)
        pytest -k 1x4   # QuietBox 2 (2xP300)     (manual / non-CI)
        pytest -k 1x8   # LoudBox / T3K           (manual / non-CI)
        pytest -k 4x8   # Galaxy                  (manual / non-CI)
    """
    num_devices = ttnn.get_num_devices()
    if mesh_shapes is None:
        all_shapes = [(1, 1), (1, 2), (1, 4), (1, 8), (4, 8)]
        mesh_shapes = [s for s in all_shapes if s[0] * s[1] <= num_devices]
    else:
        # User-provided shapes: still filter to those that fit, so an explicit
        # mesh_shapes=[(1,8)] decorator gracefully skips on smaller systems.
        mesh_shapes = [s for s in mesh_shapes if s[0] * s[1] <= num_devices]

    # CI mode: pick only the largest fitting shape so that one yaml entry can
    # target multiple SKUs and let each runner select the appropriate mesh.
    if os.getenv("CI") == "true" and len(mesh_shapes) > 1:
        mesh_shapes = [max(mesh_shapes, key=lambda s: s[0] * s[1])]

    if not mesh_shapes:
        params = [
            pytest.param(
                (1, 1),
                {"fabric_config": None, "trace_region_size": 100000000},
                id="1x1",
                marks=pytest.mark.skip(reason="No supported minimax_m3 mesh shape fits on this system"),
            )
        ]
    else:
        # Multi-device fabric: default is FABRIC_1D_RING (torus), but a plain-MESH
        # Galaxy (no wrap-around links) can only do FABRIC_1D — pass linear_fabric=True
        # there (and use ttnn.Topology.Linear in the CCLManager). See galaxy_mesh_smoke.py.
        multidev_fabric = ttnn.FabricConfig.FABRIC_1D if linear_fabric else ttnn.FabricConfig.FABRIC_1D_RING
        params = [
            pytest.param(
                shape,
                {
                    "fabric_config": (None if shape == (1, 1) else multidev_fabric),
                    "trace_region_size": 100000000,
                },
                id=f"{shape[0]}x{shape[1]}",
            )
            for shape in mesh_shapes
        ]

    def decorator(func):
        return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)(func)

    return decorator


def parametrize_batch_seq(configs=None, ids=None):
    """Universal batch/seq parametrization"""
    configs = configs or [(1, 1), (1, 32)]
    ids = ids or [
        f"prefill_{seq_len//1024 if seq_len > 1024 else seq_len}" + ("k" if seq_len > 1024 else "")
        if seq_len > 1
        else "decode_mode"
        for batch_size, seq_len in configs
    ]
    return pytest.mark.parametrize("batch_size, seq_len", configs, ids=ids)


def parametrize_weights(use_real=False):
    """Universal weight parametrization"""
    return pytest.mark.parametrize("use_real_weights", [use_real], ids=["real" if use_real else "random"])


# Test helper functions
def compare_tensors(tt_tensor, torch_tensor, mesh_device, pcc_threshold=0.99):
    """Universal tensor comparison - handles both TT tensors and already-converted torch tensors"""
    from models.common.utility_functions import comp_pcc

    # Check if tt_tensor is already a torch tensor
    if isinstance(tt_tensor, torch.Tensor):
        # Already converted, use directly
        tt_torch = tt_tensor
    else:
        # Convert TT tensor to torch
        mesh_shape = mesh_device.shape
        tt_torch = ttnn.to_torch(
            tt_tensor, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_shape, dims=(-2, -1))
        )

    return comp_pcc(torch_tensor, tt_torch, pcc_threshold)
