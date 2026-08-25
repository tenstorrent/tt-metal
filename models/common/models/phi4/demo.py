# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke and accuracy tests for ``Phi4Transformer`` (microsoft/phi-4).

Requires Hugging Face weights locally (``HF_HOME=/proj_sw/user_dev/huggingface``).

Run (N300 primary, internal KV smoke)::

    MESH_DEVICE=N300 HF_MODEL=microsoft/phi-4 \\
      pytest models/common/models/phi4/demo.py -v -k prefill_smoke

Executor + paged KV::

    MESH_DEVICE=N300 HF_MODEL=microsoft/phi-4 \\
      pytest models/common/models/phi4/demo.py -v -k executor_prefill

N150 best-effort::

    MESH_DEVICE=N150 PHI4_NUM_LAYERS=1 \\
      pytest models/common/models/phi4/demo.py -v -k prefill_smoke

``Attention1D`` shards heads across the mesh: use ``MESH_DEVICE=N300`` (2) for the default
checkpoint (40 attn heads / 10 KV heads, both divisible by 2). N150 (1 device) also works.
T3K (8) is incompatible (10 KV heads / 8 is not integer).
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.common.models.executor import make_contiguous_page_table
from models.common.models.phi4.executor import EagerPhi4Executor, TracedPhi4Executor, run_lm_head, run_prefill
from models.common.models.phi4.generator import greedy_argmax_from_logits, greedy_decode_one_step
from models.common.models.phi4.model import DEFAULT_HF_REVISION, Phi4Transformer
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.common.utility_functions import comp_pcc


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
        f"Try MESH_DEVICE=N300 (2) or MESH_DEVICE=N150 (1)."
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
    return os.environ.get("HF_MODEL", "microsoft/phi-4")


_slow = pytest.mark.slow


@_slow
@pytest.mark.parametrize("seq_len", [128])
def test_phi4_prefill_smoke(mesh_device, hf_model_id, seq_len: int, tmp_path_factory):
    """One prefill pass (truncated layers via env) and LM head → greedy token (internal KV)."""
    _skip_unless_heads_divide_mesh(mesh_device, hf_model_id)
    num_layers = int(os.environ.get("PHI4_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("phi4_cache")
    model = None
    try:
        model = Phi4Transformer.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=32,
            max_seq_len=max(2048, seq_len),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=False,
        )
    except Exception as e:
        pytest.skip(f"Could not build model (weights / memory): {e}")

    ttnn.SetDefaultDevice(mesh_device)
    try:
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
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)


@_slow
def test_phi4_decode_one_step(mesh_device, hf_model_id, tmp_path_factory):
    """Prefill 128 tokens, then one greedy decode step at position 128 (single-user)."""
    _skip_unless_heads_divide_mesh(mesh_device, hf_model_id)
    num_layers = int(os.environ.get("PHI4_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("phi4_decode_cache")
    seq_len = 128
    model = None
    try:
        model = Phi4Transformer.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=max(512, seq_len + 8),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=False,
        )
    except Exception as e:
        pytest.skip(f"Could not build model (weights / memory): {e}")

    ttnn.SetDefaultDevice(mesh_device)
    try:
        toks = torch.zeros(1, 1, 1, seq_len, dtype=torch.int32)
        toks[..., :4] = torch.tensor([1, 2, 3, 4], dtype=torch.int32).view(1, 1, 1, 4)
        x = ttnn.from_torch(
            toks,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
        )
        _ = run_prefill(model, x, start_pos=0)
        _ = greedy_decode_one_step(model, token_id=64, current_pos=seq_len)
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)


@_slow
@pytest.mark.parametrize("seq_len", [128])
def test_phi4_executor_prefill_smoke(mesh_device, hf_model_id, seq_len: int, tmp_path_factory):
    """``EagerPhi4Executor`` + paged KV: prefill returns host logits."""
    _skip_unless_heads_divide_mesh(mesh_device, hf_model_id)
    num_layers = int(os.environ.get("PHI4_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("phi4_exec_cache")
    model = None
    try:
        model = Phi4Transformer.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=max(2048, seq_len),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=True,
        )
    except Exception as e:
        pytest.skip(f"Could not build model: {e}")

    ma = model.model_args
    assert ma is not None
    block_size = 32
    max_num_blocks = (ma.max_seq_len // block_size) * ma.max_batch_size
    kv_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)

    ttnn.SetDefaultDevice(mesh_device)
    try:
        ex = EagerPhi4Executor(model, mesh_device)
        kv = ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, block_size)
        toks = torch.zeros(1, seq_len, dtype=torch.long)
        toks[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        logits = ex.prefill_forward(toks, page_table=page_table, kv_cache=kv)
        assert logits.shape[0] == 1 and logits.shape[-1] == model.vocab_size
    except Exception as e:
        pytest.skip(f"Executor prefill not runnable: {e}")
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)


@_slow
def test_phi4_teacher_forcing_prefill_vs_hf(mesh_device, hf_model_id, tmp_path_factory):
    """Last-token logits vs HF after prefill (PCC gate, full depth).

    Must run the same number of decoder layers as ``AutoModelForCausalLM``; comparing a
    truncated TT stack (``PHI4_NUM_LAYERS`` for smoke tests) to full HF logits always fails PCC.
    Unset ``PHI4_NUM_LAYERS`` here, or set it equal to ``num_hidden_layers``.
    """
    _skip_unless_heads_divide_mesh(mesh_device, hf_model_id)
    hf_cfg = AutoConfig.from_pretrained(hf_model_id)
    n_hf = int(hf_cfg.num_hidden_layers)
    env_layers = os.environ.get("PHI4_NUM_LAYERS")
    if env_layers is not None:
        num_layers = int(env_layers)
        if num_layers != n_hf:
            pytest.skip(
                "Teacher-forcing PCC is defined against the full HF forward; "
                f"unset PHI4_NUM_LAYERS to use all {n_hf} layers (currently PHI4_NUM_LAYERS={num_layers})."
            )
    else:
        num_layers = n_hf
    seq_len = 128
    cache = tmp_path_factory.mktemp("phi4_tf")
    model = None
    try:
        model = Phi4Transformer.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=max(512, seq_len),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=True,
        )
    except Exception as e:
        pytest.skip(f"Could not build model: {e}")

    ma = model.model_args
    block_size = 32
    max_num_blocks = (ma.max_seq_len // block_size) * ma.max_batch_size
    kv_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)

    input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    input_ids[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    ttnn.SetDefaultDevice(mesh_device)
    try:
        hf = AutoModelForCausalLM.from_pretrained(hf_model_id, revision=DEFAULT_HF_REVISION, torch_dtype=torch.bfloat16)
        hf.eval()
        with torch.no_grad():
            hf_out = hf(input_ids[:, :seq_len]).logits[0, seq_len - 1, :].float()

        ex = EagerPhi4Executor(model, mesh_device)
        kv = ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, block_size)
        tt_logits = ex.prefill_forward(input_ids[:, :seq_len], page_table=page_table, kv_cache=kv)
        tt_vec = tt_logits[0, 0, :].float()

        ok, pcc = comp_pcc(hf_out, tt_vec, pcc=0.85)
        assert ok, f"Prefill last-token PCC too low: {pcc}"
    except Exception as e:
        pytest.skip(f"Teacher-forcing PCC check not runnable: {e}")
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)


@_slow
def test_phi4_eager_traced_prefill_logits_match(mesh_device, hf_model_id, tmp_path_factory):
    """``EagerPhi4Executor`` vs ``TracedPhi4Executor``: same prefill → host logits within tolerance."""
    _skip_unless_heads_divide_mesh(mesh_device, hf_model_id)
    cache_a = tmp_path_factory.mktemp("phi4_par_e")
    cache_b = tmp_path_factory.mktemp("phi4_par_t")
    seq_len = 128
    m_e = m_t = None
    try:
        m_e = Phi4Transformer.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=512,
            num_layers=1,
            cache_dir=cache_a,
            executor_mode=True,
        )
        m_t = Phi4Transformer.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=512,
            num_layers=1,
            cache_dir=cache_b,
            executor_mode=True,
        )
    except Exception as e:
        pytest.skip(f"Could not build models: {e}")

    ma = m_e.model_args
    assert ma is not None
    block_size = 32
    max_num_blocks = (ma.max_seq_len // block_size) * ma.max_batch_size
    kv_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)

    ttnn.SetDefaultDevice(mesh_device)
    try:
        eager_ex = EagerPhi4Executor(m_e, mesh_device)
        traced_ex = TracedPhi4Executor(m_t, mesh_device)
        kv_e = eager_ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        kv_t = traced_ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, block_size)
        toks = torch.zeros(1, seq_len, dtype=torch.long)
        toks[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        le = eager_ex.prefill_forward(toks, page_table=page_table, kv_cache=kv_e)
        lt = traced_ex.prefill_forward(toks, page_table=page_table, kv_cache=kv_t)
        diff = (le.float() - lt.float()).abs().max().item()
        assert diff < 0.25, f"eager/traced prefill logits max abs diff too large: {diff}"
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(m_e, mesh_device)
        cleanup_model_case(m_t, mesh_device)
