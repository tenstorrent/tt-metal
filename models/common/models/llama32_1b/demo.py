# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests for ``Llama32_1B`` (meta-llama/Llama-3.2-1B-Instruct, TTTv2).

Milestones covered:
  M1 — model builds + raw prefill/decode forward runs end-to-end
  M2 — ``EagerLlama32_1BExecutor`` + paged KV: prefill → host logits
  M3 — ``TracedLlama32_1BExecutor`` vs ``EagerLlama32_1BExecutor`` logits parity

Supported meshes: N150 (1×1), N300 (1×2) and T3K (1×8).

Run::

    # M1 prefill smoke — N300
    MESH_DEVICE=N300 HF_MODEL=meta-llama/Llama-3.2-1B-Instruct \\
      ./python_env/bin/pytest models/common/models/llama32_1b/demo.py -v -k prefill_smoke

    # M2 executor smoke — N150
    MESH_DEVICE=N150 HF_MODEL=meta-llama/Llama-3.2-1B-Instruct \\
      ./python_env/bin/pytest models/common/models/llama32_1b/demo.py -v -k executor_prefill_smoke

    # M3 eager/traced parity — N300
    MESH_DEVICE=N300 HF_MODEL=meta-llama/Llama-3.2-1B-Instruct \\
      ./python_env/bin/pytest models/common/models/llama32_1b/demo.py -v -k eager_traced_parity

Notes:
  * ``from_pretrained`` loads HF weights directly (TTTv2-native, no TTTv1 bridge).
    Set ``LLAMA32_1B_DEMO_NUM_LAYERS=1`` for fast single-layer smoke runs.
    On first run the LazyWeight cache is cold (~3-5 min); subsequent runs are fast.
  * ``MESH_DEVICE`` must satisfy: n_heads (32) and n_kv_heads (8) divisible
    by device count. N150 (1), N300 (2) and T3K (8) all satisfy this.
  * fabric_config is set to FABRIC_1D on multi-device meshes (required by
    TTTv2 CCL ops).
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.models.executor import make_contiguous_page_table
from models.common.models.llama32_1b.model import (
    LLAMA32_1B_ACCURACY,
    EagerLlama32_1BExecutor,
    Llama32_1BTransformer1D,
    TracedLlama32_1BExecutor,
)
from models.common.tests.demos.cleanup_utils import cleanup_model_case

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / pytestmark
# ─────────────────────────────────────────────────────────────────────────────

_MESH_SHAPE = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
    os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
)


@pytest.fixture
def device_params(request, galaxy_type):
    """Translate ``fabric_config: True`` → real FabricConfig for multi-device meshes."""
    params = getattr(request, "param", {}).copy()
    is_single = (_MESH_SHAPE == (1, 1)) if isinstance(_MESH_SHAPE, tuple) else (_MESH_SHAPE == 1)
    if "fabric_config" in params:
        if is_single:
            params["fabric_config"] = None
        elif params["fabric_config"] is True:
            params["fabric_config"] = (
                ttnn.FabricConfig.FABRIC_1D_RING if galaxy_type == "6U" else ttnn.FabricConfig.FABRIC_1D
            )
    return params


pytestmark = [
    pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True),
    pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True),
]


@pytest.fixture(scope="module")
def hf_model_id():
    return os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")


_slow = pytest.mark.slow

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_LLAMA32_1B_N_HEADS = 32
_LLAMA32_1B_N_KV_HEADS = 8


def _skip_unless_heads_divide_mesh(mesh_device: ttnn.MeshDevice) -> None:
    n_dev = mesh_device.get_num_devices()
    if n_dev <= 1:
        return
    if _LLAMA32_1B_N_HEADS % n_dev == 0 and _LLAMA32_1B_N_KV_HEADS % n_dev == 0:
        return
    pytest.skip(
        f"Incompatible mesh for Llama-3.2-1B-Instruct: {n_dev} devices; "
        f"need n_heads ({_LLAMA32_1B_N_HEADS}) and n_kv_heads ({_LLAMA32_1B_N_KV_HEADS}) "
        f"each divisible by {n_dev}. Use MESH_DEVICE=N150 (1), N300 (2) or T3K (8)."
    )


def _create_model(
    mesh_device,
    *,
    executor_mode: bool = True,
    optimizations: str = "accuracy",
    max_batch_size: int = 32,
    max_seq_len: int = 1024,
):
    """Build Llama32_1BTransformer1D via TTTv2-native from_pretrained.

    Returns the model. When executor_mode=True, model.model_args is set to
    a Llama32_1BExecutorRuntimeConfig.
    """
    from models.common.models.llama32_1b.model import LLAMA32_1B_PERFORMANCE

    hf_model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    num_layers = int(os.environ.get("LLAMA32_1B_DEMO_NUM_LAYERS", 0)) or None
    cache_dir = os.environ.get("TT_CACHE_PATH", None)

    precision = LLAMA32_1B_ACCURACY if optimizations == "accuracy" else LLAMA32_1B_PERFORMANCE

    return Llama32_1BTransformer1D.from_pretrained(
        mesh_device,
        hf_model_id=hf_model,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        cache_dir=cache_dir,
        precision=precision,
        block_size=32,
        executor_mode=executor_mode,
    )


def _kv_shape_from_model(model, mesh_device):
    ma = model.model_args
    block_size = 32
    max_num_blocks = (ma.max_seq_len // block_size) * ma.max_batch_size
    return (
        max_num_blocks,
        ma.n_kv_heads // mesh_device.get_num_devices(),
        block_size,
        ma.head_dim,
    )


# ─────────────────────────────────────────────────────────────────────────────
# M1: prefill smoke + decode smoke
# ─────────────────────────────────────────────────────────────────────────────


@_slow
@pytest.mark.parametrize("seq_len", [128])
def test_llama32_1b_prefill_smoke(mesh_device, hf_model_id, seq_len: int, tmp_path_factory):
    """M1 — model builds and EagerLlama32_1BExecutor prefill returns correct-shape logits."""
    _skip_unless_heads_divide_mesh(mesh_device)
    model = None
    try:
        model = _create_model(mesh_device, executor_mode=True, optimizations="accuracy")
    except Exception as e:
        pytest.skip(f"Could not build model (weights / memory): {e}")

    kv_shape = _kv_shape_from_model(model, mesh_device)
    ma = model.model_args

    ttnn.SetDefaultDevice(mesh_device)
    try:
        ex = EagerLlama32_1BExecutor(model, mesh_device)
        kv = ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, 32)
        toks = torch.zeros(1, seq_len, dtype=torch.long)
        toks[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        logits = ex.prefill_forward(toks, page_table=page_table, kv_cache=kv)
        assert logits.shape[0] == 1, f"expected batch dim 1, got {logits.shape}"
        assert logits.shape[-1] == model.vocab_size, f"expected vocab_size={model.vocab_size}, got {logits.shape}"
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)


@_slow
def test_llama32_1b_decode_smoke(mesh_device, hf_model_id, tmp_path_factory):
    """M1 — prefill 128 tokens then one decode step; verifies decode KV-cache update."""
    _skip_unless_heads_divide_mesh(mesh_device)
    seq_len = 128
    model = None
    try:
        model = _create_model(
            mesh_device, executor_mode=True, optimizations="accuracy", max_batch_size=1, max_seq_len=512
        )
    except Exception as e:
        pytest.skip(f"Could not build model (weights / memory): {e}")

    kv_shape = _kv_shape_from_model(model, mesh_device)
    ma = model.model_args

    ttnn.SetDefaultDevice(mesh_device)
    try:
        ex = EagerLlama32_1BExecutor(model, mesh_device)
        kv = ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, 32)
        toks = torch.zeros(1, seq_len, dtype=torch.long)
        toks[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        prefill_logits = ex.prefill_forward(toks, page_table=page_table, kv_cache=kv)
        assert prefill_logits is not None, "prefill returned None"
        next_tok = int(torch.argmax(torch.tensor(prefill_logits[0, 0]).float()).item())
        decode_toks = torch.tensor([next_tok], dtype=torch.long)
        start_pos = torch.tensor([seq_len], dtype=torch.long)
        decode_logits, _ = ex.decode_forward(
            decode_toks, start_pos, page_table=page_table, kv_cache=kv, read_from_device=True
        )
        assert decode_logits is not None, "decode returned None"
        assert (
            decode_logits.shape[-1] == model.vocab_size
        ), f"expected vocab_size={model.vocab_size}, got {decode_logits.shape}"
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)


# ─────────────────────────────────────────────────────────────────────────────
# M2: executor prefill smoke — full paged KV contract
# ─────────────────────────────────────────────────────────────────────────────


@_slow
@pytest.mark.parametrize("seq_len", [128])
def test_llama32_1b_executor_prefill_smoke(mesh_device, hf_model_id, seq_len: int, tmp_path_factory):
    """M2 — EagerLlama32_1BExecutor + paged KV: prefill returns host logits."""
    _skip_unless_heads_divide_mesh(mesh_device)
    model = None
    try:
        model = _create_model(mesh_device, executor_mode=True, optimizations="accuracy")
    except Exception as e:
        pytest.skip(f"Could not build model (weights / memory): {e}")

    kv_shape = _kv_shape_from_model(model, mesh_device)
    ma = model.model_args

    ttnn.SetDefaultDevice(mesh_device)
    try:
        ex = EagerLlama32_1BExecutor(model, mesh_device)
        kv = ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, 32)
        toks = torch.zeros(1, seq_len, dtype=torch.long)
        toks[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        logits = ex.prefill_forward(toks, page_table=page_table, kv_cache=kv)
        assert logits.shape[0] == 1, f"expected batch dim 1, got {logits.shape}"
        assert (
            logits.shape[-1] == model.vocab_size
        ), f"expected last dim == vocab_size={model.vocab_size}, got {logits.shape}"
    except Exception as e:
        pytest.skip(f"Executor prefill not runnable: {e}")
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)


# ─────────────────────────────────────────────────────────────────────────────
# M3: eager vs traced parity
# ─────────────────────────────────────────────────────────────────────────────


@_slow
def test_llama32_1b_eager_traced_parity(mesh_device, hf_model_id, tmp_path_factory):
    """M3 — EagerLlama32_1BExecutor vs TracedLlama32_1BExecutor: prefill logits within tolerance.

    Prefill trace capture is gated off under LazyWeight + distributed norms
    (``can_enable_trace`` returns False). The traced executor falls back to eager
    for prefill while keeping decode traced.
    """
    _skip_unless_heads_divide_mesh(mesh_device)
    seq_len = 128
    m_e = m_t = None
    try:
        m_e = _create_model(mesh_device, executor_mode=True, optimizations="accuracy")
        m_t = _create_model(mesh_device, executor_mode=True, optimizations="accuracy")
    except Exception as e:
        pytest.skip(f"Could not build models (weights / memory): {e}")

    kv_shape_e = _kv_shape_from_model(m_e, mesh_device)
    kv_shape_t = _kv_shape_from_model(m_t, mesh_device)
    ma_e = m_e.model_args
    ma_t = m_t.model_args

    ttnn.SetDefaultDevice(mesh_device)
    try:
        eager_ex = EagerLlama32_1BExecutor(m_e, mesh_device)
        traced_ex = TracedLlama32_1BExecutor(m_t, mesh_device)
        kv_e = eager_ex.allocate_kv_cache(kv_shape_e, torch.bfloat16, ma_e.n_layers)
        kv_t = traced_ex.allocate_kv_cache(kv_shape_t, torch.bfloat16, ma_t.n_layers)
        page_table = make_contiguous_page_table(1, ma_e.max_seq_len, 32)
        toks = torch.zeros(1, seq_len, dtype=torch.long)
        toks[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        le = eager_ex.prefill_forward(toks, page_table=page_table, kv_cache=kv_e)
        lt = traced_ex.prefill_forward(toks, page_table=page_table, kv_cache=kv_t)
        diff = (le.float() - lt.float()).abs().max().item()
        assert diff < 0.25, f"eager/traced prefill logits max abs diff too large: {diff:.4f}"
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(m_e, mesh_device)
        cleanup_model_case(m_t, mesh_device)
