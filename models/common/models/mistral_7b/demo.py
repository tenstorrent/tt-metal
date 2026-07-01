# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke and accuracy tests for ``Mistral7B`` (Mistral-7B-Instruct-v0.3, TTTv2).

Requires Hugging Face weights locally (or network + auth). The HF repo is gated;
on this host the weights are cached under ``HF_HOME=/proj_sw/user_dev/huggingface``.

Run (N300 primary, internal KV smoke)::

    MESH_DEVICE=N300 HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \\
      ./python_env/bin/pytest models/common/models/mistral_7b/demo.py -v -k prefill_smoke

N150 best-effort::

    MESH_DEVICE=N150 MISTRAL_DEMO_NUM_LAYERS=1 \\
      ./python_env/bin/pytest models/common/models/mistral_7b/demo.py -v -k prefill_smoke

``Attention1D`` shards heads across the mesh: use ``MESH_DEVICE=N300`` (2) or ``N150`` (1)
for this 7B checkpoint (32 / 8 heads). T3K (8) is compatible too but not in scope here.
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.models.executor import make_contiguous_page_table
from models.common.models.mistral_7b.executor import (
    EagerMistralExecutor,
    TracedMistralExecutor,
    run_lm_head,
    run_prefill,
)
from models.common.models.mistral_7b.model import Mistral7B
from models.common.tests.demos.cleanup_utils import cleanup_model_case
from models.common.utility_functions import comp_pcc


def _skip_unless_heads_divide_mesh(mesh_device: ttnn.MeshDevice) -> None:
    """Attention1D TP requires n_heads and n_kv_heads divisible by device count."""
    n_dev = mesh_device.get_num_devices()
    if n_dev <= 1:
        return
    # Mistral-7B-Instruct-v0.3 has 32 attention heads and 8 KV heads.
    if 32 % n_dev == 0 and 8 % n_dev == 0:
        return
    pytest.skip(
        f"Incompatible mesh for Mistral-7B-Instruct-v0.3: {n_dev} devices need "
        f"num_attention_heads (32) and num_key_value_heads (8) each divisible by {n_dev}. "
        f"Use MESH_DEVICE=N150 (1), N300 (2), or T3K (8)."
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
    return os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")


_slow = pytest.mark.slow


def _greedy_argmax_from_logits(logits: ttnn.Tensor, *, mesh_device: ttnn.MeshDevice) -> int:
    lt = to_torch_auto_compose(logits, device=mesh_device).float()
    if lt.dim() == 4:
        lt = lt[0, 0, 0]
    elif lt.dim() == 3:
        lt = lt[0, 0]
    return int(torch.argmax(lt).item())


def _greedy_decode_one_step(model: Mistral7B, token_id: int, *, current_pos: int) -> int:
    tid = torch.tensor([[[[token_id]]]], dtype=torch.int32)
    x = ttnn.from_torch(
        tid,
        device=model.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(model.mesh_device),
    )
    h = model.decode_from_token_ids(x, current_pos=current_pos)
    logits = model.lm_logits(h)
    return _greedy_argmax_from_logits(logits, mesh_device=model.mesh_device)


@_slow
@pytest.mark.parametrize("seq_len", [128])
def test_mistral_7b_prefill_smoke(mesh_device, hf_model_id, seq_len: int, tmp_path_factory):
    """One prefill pass (truncated layers via env) and LM head → greedy token (internal KV)."""
    _skip_unless_heads_divide_mesh(mesh_device)
    num_layers = int(os.environ.get("MISTRAL_DEMO_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("mistral_cache")
    model = None
    try:
        model = Mistral7B.from_pretrained(
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
        _ = _greedy_argmax_from_logits(logits, mesh_device=mesh_device)
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)


@_slow
def test_mistral_7b_decode_one_step(mesh_device, hf_model_id, tmp_path_factory):
    """Prefill 128 tokens, then one greedy decode step at position 128.

    Single-user (``max_batch_size=1``): the legacy ``decode_from_token_ids`` path only
    fills one KV slot per call, so the non-paged ``paged_update_cache`` validation
    (``num_indices == batch_size``) requires the model be built at batch=1. Multi-user
    decode goes through the paged executor (covered by M2).
    """
    _skip_unless_heads_divide_mesh(mesh_device)
    num_layers = int(os.environ.get("MISTRAL_DEMO_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("mistral_decode_cache")
    seq_len = 128
    model = None
    try:
        model = Mistral7B.from_pretrained(
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
        _ = _greedy_decode_one_step(model, token_id=64, current_pos=seq_len)
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)


@_slow
@pytest.mark.parametrize("seq_len", [128])
def test_mistral_7b_executor_prefill_smoke(mesh_device, hf_model_id, seq_len: int, tmp_path_factory):
    """``EagerMistralExecutor`` + paged KV: prefill returns host logits.

    Exercises the full executor contract (``set_kv_cache``, ``embed_prefill``,
    ``rope_setup``, ``page_table`` flow, ``post_process_prefill_output``,
    ``gather_and_untilize_logits``). Batch is 1 here for fast iteration; the engine
    supports batch up to ``model.model_args.max_batch_size``.
    """
    _skip_unless_heads_divide_mesh(mesh_device)
    num_layers = int(os.environ.get("MISTRAL_DEMO_NUM_LAYERS", "1"))
    cache = tmp_path_factory.mktemp("mistral_exec_cache")
    model = None
    try:
        model = Mistral7B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=max(2048, seq_len),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=True,
        )
    except Exception as e:
        pytest.skip(f"Could not build model (weights / memory): {e}")

    ma = model.model_args
    assert ma is not None, "executor_mode=True must populate model.model_args"
    block_size = 32
    max_num_blocks = (ma.max_seq_len // block_size) * ma.max_batch_size
    kv_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)

    ttnn.SetDefaultDevice(mesh_device)
    try:
        ex = EagerMistralExecutor(model, mesh_device)
        kv = ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, block_size)
        toks = torch.zeros(1, seq_len, dtype=torch.long)
        toks[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        logits = ex.prefill_forward(toks, page_table=page_table, kv_cache=kv)
        assert logits.shape[0] == 1, f"expected batch dim 1, got shape {logits.shape}"
        assert (
            logits.shape[-1] == model.vocab_size
        ), f"expected last dim == vocab_size={model.vocab_size}, got shape {logits.shape}"
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)


@_slow
def test_mistral_7b_eager_traced_prefill_logits_match(mesh_device, hf_model_id, tmp_path_factory):
    """``EagerMistralExecutor`` vs ``TracedMistralExecutor``: same prefill → host logits within tolerance.

    Per ``Mistral7BExecutorRuntimeConfig.can_enable_trace``, traced prefill is enabled for
    the supported seq lens (N150 ``[128]``; N300/T3K ``[128, 1024]``), so at ``seq_len=128``
    this exercises the real traced prefill capture/replay path (plus the decode-time trace
    paths); larger seq lens fall back to eager capture under the hood.
    """
    _skip_unless_heads_divide_mesh(mesh_device)
    cache_e = tmp_path_factory.mktemp("mistral_par_e")
    cache_t = tmp_path_factory.mktemp("mistral_par_t")
    seq_len = 128
    m_e = m_t = None
    try:
        m_e = Mistral7B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=512,
            num_layers=1,
            cache_dir=cache_e,
            executor_mode=True,
        )
        m_t = Mistral7B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=512,
            num_layers=1,
            cache_dir=cache_t,
            executor_mode=True,
        )
    except Exception as e:
        pytest.skip(f"Could not build models (weights / memory): {e}")

    ma = m_e.model_args
    assert ma is not None
    block_size = 32
    max_num_blocks = (ma.max_seq_len // block_size) * ma.max_batch_size
    kv_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)

    ttnn.SetDefaultDevice(mesh_device)
    try:
        eager_ex = EagerMistralExecutor(m_e, mesh_device)
        traced_ex = TracedMistralExecutor(m_t, mesh_device)
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


@_slow
def test_mistral_7b_teacher_forcing_prefill_vs_hf(mesh_device, hf_model_id, tmp_path_factory):
    """Last-token logits PCC vs HF after a 128-token prefill (full depth).

    The PCC gate is defined against the full HF forward; comparing a truncated TT stack
    (``MISTRAL_DEMO_NUM_LAYERS`` for smoke tests) to full HF logits always fails. Unset
    ``MISTRAL_DEMO_NUM_LAYERS`` here, or set it equal to ``num_hidden_layers``.

    Starting PCC threshold is 0.85 (same as the Qwen 2.5-7B port at this stage); raise
    only after confirming the floor is robust across mesh and HF revision.
    """
    _skip_unless_heads_divide_mesh(mesh_device)
    hf_cfg = AutoConfig.from_pretrained(hf_model_id)
    n_hf = int(hf_cfg.num_hidden_layers)
    env_layers = os.environ.get("MISTRAL_DEMO_NUM_LAYERS")
    if env_layers is not None:
        num_layers = int(env_layers)
        if num_layers != n_hf:
            pytest.skip(
                "Teacher-forcing PCC is defined against the full HF forward; "
                f"unset MISTRAL_DEMO_NUM_LAYERS to use all {n_hf} layers "
                f"(currently MISTRAL_DEMO_NUM_LAYERS={num_layers})."
            )
    else:
        num_layers = n_hf

    seq_len = 128
    cache = tmp_path_factory.mktemp("mistral_tf")
    model = None
    try:
        model = Mistral7B.from_pretrained(
            mesh_device,
            hf_model_id,
            max_batch_size=1,
            max_seq_len=max(512, seq_len),
            num_layers=num_layers,
            cache_dir=cache,
            executor_mode=True,
        )
    except Exception as e:
        pytest.skip(f"Could not build model (weights / memory): {e}")

    ma = model.model_args
    assert ma is not None
    block_size = 32
    max_num_blocks = (ma.max_seq_len // block_size) * ma.max_batch_size
    kv_shape = (max_num_blocks, ma.n_kv_heads // mesh_device.get_num_devices(), block_size, ma.head_dim)

    input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    input_ids[0, :4] = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    ttnn.SetDefaultDevice(mesh_device)
    try:
        hf = AutoModelForCausalLM.from_pretrained(hf_model_id, torch_dtype=torch.bfloat16)
        hf.eval()
        with torch.no_grad():
            hf_out = hf(input_ids[:, :seq_len]).logits[0, seq_len - 1, :].float()
        del hf

        ex = EagerMistralExecutor(model, mesh_device)
        kv = ex.allocate_kv_cache(kv_shape, torch.bfloat16, ma.n_layers)
        page_table = make_contiguous_page_table(1, ma.max_seq_len, block_size)
        tt_logits = ex.prefill_forward(input_ids[:, :seq_len], page_table=page_table, kv_cache=kv)
        tt_vec = tt_logits[0, 0, :].float()

        ok, pcc = comp_pcc(hf_out, tt_vec, pcc=0.85)
        print(f"\nMistral-7B-Instruct-v0.3 teacher-forcing last-token PCC vs HF: {pcc}")
        assert ok, f"Prefill last-token PCC too low: {pcc}"
    finally:
        ttnn.SetDefaultDevice(None)
        cleanup_model_case(model, mesh_device)
