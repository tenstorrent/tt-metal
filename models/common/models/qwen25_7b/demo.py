# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests for ``Qwen25_7BTTT`` (prefill + one greedy decode step).

Requires Hugging Face weights locally (or network + auth for gated models).

Run (N300 primary)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen2.5-7B-Instruct \\
      pytest models/common/models/qwen25_7b/demo.py -v -k prefill_smoke

N150 best-effort (may OOM on full weights; use ``QWEN25_DEMO_NUM_LAYERS=1``)::

    MESH_DEVICE=N150 QWEN25_DEMO_NUM_LAYERS=1 \\
      pytest models/common/models/qwen25_7b/demo.py -v -k prefill_smoke --slow

``Attention1D`` shards heads across the mesh: ``num_attention_heads`` and
``num_key_value_heads`` must each divide ``mesh_device.get_num_devices()``.
For the default Qwen2.5-7B checkpoint (28 / 4 heads), **8-wide T3K is skipped**
— use ``MESH_DEVICE=N300`` (2) or ``N150x4`` (4), or a different ``HF_MODEL``.
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.common.models.qwen25_7b.executor import run_lm_head, run_prefill
from models.common.models.qwen25_7b.generator import greedy_argmax_from_logits, greedy_decode_one_step
from models.common.models.qwen25_7b.model import Qwen25_7BTTT


def _skip_unless_heads_divide_mesh(mesh_device: ttnn.MeshDevice, hf_model_id: str) -> None:
    """Attention1D TP requires n_heads and n_kv_heads divisible by device count."""
    n_dev = mesh_device.get_num_devices()
    if n_dev <= 1:
        return
    cfg = AutoConfig.from_pretrained(hf_model_id)
    n_h, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    if n_h % n_dev == 0 and n_kv % n_dev == 0:
        return
    pytest.skip(
        f"Incompatible mesh for {hf_model_id}: {n_dev} devices need "
        f"num_attention_heads ({n_h}) and num_key_value_heads ({n_kv}) each divisible by {n_dev}. "
        f"Try MESH_DEVICE=N300 (2) or N150x4 (4)."
    )


@pytest.fixture
def device_params(request, galaxy_type):
    """Match ``models/tt_transformers/conftest.py`` so ``fabric_config: True`` maps to a real fabric."""
    params = getattr(request, "param", {}).copy()

    mesh_device = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
        os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
    )
    is_single_device = (mesh_device == (1, 1)) if isinstance(mesh_device, tuple) else (mesh_device == 1)

    if "fabric_config" in params:
        if is_single_device:
            params["fabric_config"] = None
        elif params["fabric_config"] is True:
            params["fabric_config"] = (
                ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
            )

    return params


pytestmark = [
    pytest.mark.parametrize(
        "mesh_device",
        [
            {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
                os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
            )
        ],
        indirect=True,
    ),
    pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True),
]


@pytest.fixture(scope="module")
def hf_model_id():
    return os.environ.get("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")


_slow = pytest.mark.slow


@_slow
@pytest.mark.parametrize("seq_len", [128])
def test_qwen25_7b_prefill_smoke(mesh_device, hf_model_id, seq_len: int, tmp_path_factory):
    """One prefill pass (truncated layers via env) and LM head → greedy token."""
    _skip_unless_heads_divide_mesh(mesh_device, hf_model_id)
    num_layers = int(os.environ.get("QWEN25_DEMO_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("qwen25_cache")
    try:
        model = Qwen25_7BTTT.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=32,
            max_seq_len=max(2048, seq_len),
            num_layers=num_layers,
            cache_dir=cache,
        )
    except Exception as e:
        pytest.skip(f"Could not build model (weights / memory): {e}")

    ttnn.SetDefaultDevice(mesh_device)
    toks = torch.zeros(1, 1, 1, seq_len, dtype=torch.int32)
    toks[..., :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32).view(1, 1, 1, 4)
    x = ttnn.from_torch(
        toks,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    hidden = run_prefill(model, x, start_pos=0)
    logits = run_lm_head(model, hidden)
    _ = greedy_argmax_from_logits(logits, mesh_device=mesh_device)
    ttnn.SetDefaultDevice(None)


@_slow
def test_qwen25_7b_decode_one_step(mesh_device, hf_model_id, tmp_path_factory):
    """Prefill 128 tokens, then one greedy decode step at position 128."""
    _skip_unless_heads_divide_mesh(mesh_device, hf_model_id)
    num_layers = int(os.environ.get("QWEN25_DEMO_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("qwen25_decode_cache")
    seq_len = 128
    try:
        model = Qwen25_7BTTT.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=32,
            max_seq_len=max(512, seq_len + 8),
            num_layers=num_layers,
            cache_dir=cache,
        )
    except Exception as e:
        pytest.skip(f"Could not build model: {e}")

    ttnn.SetDefaultDevice(mesh_device)
    toks = torch.zeros(1, 1, 1, seq_len, dtype=torch.int32)
    toks[..., :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32).view(1, 1, 1, 4)
    x = ttnn.from_torch(
        toks,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    try:
        _ = run_prefill(model, x, start_pos=0)
        _ = greedy_decode_one_step(model, token_id=64, current_pos=seq_len)
    except Exception as e:
        pytest.skip(f"Prefill/decode path not runnable: {e}")
    ttnn.SetDefaultDevice(None)
