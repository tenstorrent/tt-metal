# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for Gemma4 lm_head + softcap + all-gather pipeline.

Loads the real lm_head weight (= embed_tokens.weight.T, tied), runs a random
hidden state through the TT path (column-parallel linear → softcap → all-gather)
and compares against a torch reference.

Decode-shaped LM head (batch dim = 32, issue #44953 / 31B bringup) lives in
``test_lm_head_decode_batch32`` and uses the same ``MatmulMultiCoreReuseMultiCast1DProgramConfig``
as ``Gemma4Model`` (``_get_lm_head_program_config``).

``test_lm_head_bfp8_weight_decode_batch32`` pins the shipped bfp8 weight path
(``precision_overrides.json`` → ``lm_head: bfp8``) for the DRAM-bound
32×5376×32768 decode matmul.

    pytest -k "1x4"   # Blackhole quietbox (TP=4, column-parallel lm_head)
    pytest -k "1x8"   # T3K (TP=8, column-parallel lm_head)
    HF_MODEL=google/gemma-4-31B-it pytest .../test_lm_head.py -k "decode_batch32"
"""


import os

import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager, ccl_allgather
from models.demos.gemma4.tt.dram_sharded import lm_head_decode_config
from models.demos.gemma4.tt.model import _get_lm_head_program_config

from ...tests.test_factory import (
    TestFactory,
    _get_model_path,
    compare_tensors,
    get_pcc_threshold,
    load_real_substate,
    parametrize_mesh_with_fabric,
    real_weight_index,
)


def _real_lm_head_weight():
    """The checkpoint's trained lm_head weight, ``[vocab, hidden]``.

    Reads the one tensor out of the safetensors shards rather than building the
    HF model to reach ``embed_tokens``. Two reasons, both load-bearing:

    * The unified multimodal releases (12B) cannot be instantiated by
      ``AutoModelForCausalLM`` at all — the checkpoint's ``embed_vision`` shapes
      contradict the adapted text ``config.json`` — so a full-model load left
      these tests unrunnable on that variant even though every text weight,
      this one included, reads fine.
    * Materializing a 31B model to CPU to clone a single tensor costs ~59 GB and
      minutes; the shard read costs one ``get_tensor``.

    Prefers an explicit ``lm_head.weight`` when a checkpoint carries one, and
    otherwise falls back to the tied ``embed_tokens.weight``
    (``tie_word_embeddings=True`` on every shipped Gemma4 variant).
    """
    from safetensors import safe_open

    untied = sorted(key for key in real_weight_index() if key.endswith("lm_head.weight"))
    if untied:
        key = untied[0]
        with safe_open(real_weight_index()[key], framework="pt") as f:
            weight = f.get_tensor(key)
        return weight.to(torch.bfloat16) if weight.dtype == torch.float32 else weight
    return load_real_substate("embed_tokens")["weight"]


def _lm_head_weight_dtype(mesh_device):
    """Weight dtype for the TT lm_head under test — same source as the demo.

    These tests construct the lm_head tensor directly, so they must resolve
    ``precision_overrides.json`` the same way ``Gemma4Model`` does via
    ``Gemma4Precision.load``. Without that, a bfp8 regression in the shipped
    config would not fail here.

    ``GEMMA4_LM_HEAD_WEIGHT_DTYPE=bf16|bfp8`` forces a dtype for sweeps.
    """
    from models.demos.gemma4.tt.precision import _DTYPE_BY_NAME, Gemma4Precision

    forced = os.getenv("GEMMA4_LM_HEAD_WEIGHT_DTYPE")
    if forced is not None:
        if forced not in _DTYPE_BY_NAME:
            raise ValueError(f"GEMMA4_LM_HEAD_WEIGHT_DTYPE={forced!r} — expected one of {sorted(_DTYPE_BY_NAME)}")
        return _DTYPE_BY_NAME[forced]

    mesh_shape = tuple(mesh_device.shape) if hasattr(mesh_device, "shape") else (1, 1)
    precision = Gemma4Precision.load(_get_model_path(), mesh_shape, hf_config=TestFactory.create_hf_config())
    return precision.get("lm_head", ttnn.bfloat16)


def _run_lm_head_decode_batch32(mesh_device, *, lm_head_dtype, use_decode_config):
    """Shared decode-batch32 lm_head path; returns (tt_logits, ref_logits)."""
    model_path = _get_model_path()
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(hf_config, "text_config", hf_config)
    hidden_size = text_config.hidden_size
    cap = getattr(text_config, "final_logit_softcapping", 0.0) or 0.0
    batch = 32

    logger.info(f"Reading the real lm_head weight from {model_path}...")
    embed_weight = _real_lm_head_weight()  # [vocab, hidden]

    x_torch = torch.randn(1, 1, batch, hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_logits = torch.matmul(x_torch.float(), embed_weight.float().T.contiguous())
        if cap > 0:
            ref_logits = torch.tanh(ref_logits / cap) * cap

    tp = mesh_device.shape[1]
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1)

    lm_head_torch = embed_weight.T.unsqueeze(0).unsqueeze(0).contiguous()
    lm_mapper = mesh_config.column_parallel(mesh_device)
    lm_head_tt = ttnn.as_tensor(
        lm_head_torch,
        device=mesh_device,
        dtype=lm_head_dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=lm_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    if use_decode_config:
        program_config, out_memcfg, compute_kernel_config = lm_head_decode_config(
            mesh_device,
            m=x_torch.shape[2],
            k=hidden_size,
            n=lm_head_tt.shape[-1],
        )
        logits_tt = ttnn.linear(
            x_tt,
            lm_head_tt,
            program_config=program_config,
            memory_config=out_memcfg,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        lm_head_pc = _get_lm_head_program_config(
            mesh_device,
            m=x_torch.shape[2],
            k=hidden_size,
            n=lm_head_tt.shape[-1],
        )
        logits_tt = ttnn.linear(x_tt, lm_head_tt, program_config=lm_head_pc)

    if cap > 0:
        logits_tt = ttnn.mul(logits_tt, 1.0 / cap)
        logits_tt = ttnn.tanh(logits_tt)
        logits_tt = ttnn.mul(logits_tt, cap)
    logits_tt = ccl_allgather(logits_tt, mesh_config, ccl_manager)

    tt_logits_torch = ttnn.to_torch(ttnn.get_device_tensors(logits_tt)[0]).float()
    return tt_logits_torch, ref_logits


@parametrize_mesh_with_fabric()
def test_lm_head(mesh_device, reset_seeds, request):
    """LM head pipeline (linear + softcap + all-gather) vs torch reference."""
    model_path = _get_model_path()
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(hf_config, "text_config", hf_config)
    hidden_size = text_config.hidden_size
    cap = getattr(text_config, "final_logit_softcapping", 0.0) or 0.0

    logger.info(f"Reading the real lm_head weight from {model_path}...")
    embed_weight = _real_lm_head_weight()  # [vocab, hidden]

    seq_len = 32

    # Random hidden state at unit RMS — matches what hits lm_head after final norm.
    x_torch = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)

    # Torch reference: matmul (in fp32 for accuracy) + softcap.
    with torch.no_grad():
        ref_logits = torch.matmul(x_torch.float(), embed_weight.float().T.contiguous())
        if cap > 0:
            ref_logits = torch.tanh(ref_logits / cap) * cap

    # ── TT path ──────────────────────────────────────────────────────
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    # Column-parallel: shard vocab dim across TP devices (matches model.py:216).
    lm_head_torch = embed_weight.T.unsqueeze(0).unsqueeze(0).contiguous()  # [1,1,hidden,vocab]
    if tp > 1:
        lm_mapper = mesh_config.column_parallel(mesh_device)
    else:
        lm_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

    lm_head_tt = ttnn.as_tensor(
        lm_head_torch,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=lm_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    logits_tt = ttnn.linear(x_tt, lm_head_tt)
    if cap > 0:
        logits_tt = ttnn.mul(logits_tt, 1.0 / cap)
        logits_tt = ttnn.tanh(logits_tt)
        logits_tt = ttnn.mul(logits_tt, cap)
    if tp > 1:
        logits_tt = ccl_allgather(logits_tt, mesh_config, ccl_manager)

    if is_mesh:
        tt_logits_torch = ttnn.to_torch(ttnn.get_device_tensors(logits_tt)[0]).float()
    else:
        tt_logits_torch = ttnn.to_torch(logits_tt).float()

    passing, pcc_msg = compare_tensors(tt_logits_torch, ref_logits, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"LM head PCC too low: {pcc_msg}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4), (1, 8)])
def test_lm_head_decode_batch32(mesh_device, reset_seeds, request):
    """LM head at decode contract [1,1,32,H] x [H,V/TP] with shipped decode config.

    Matches ``Gemma4Model._apply_lm_head`` (``lm_head_decode_config`` + weight dtype
    from ``precision_overrides.json``). PCC threshold ≥ 0.999 is keyed under
    ``gemma-4-31B-it`` / ``1x4`` and ``1x8`` in ``pcc_thresholds.json``.
    """
    lm_head_dtype = _lm_head_weight_dtype(mesh_device)
    tt_logits_torch, ref_logits = _run_lm_head_decode_batch32(
        mesh_device,
        lm_head_dtype=lm_head_dtype,
        use_decode_config=True,
    )

    passing, pcc_msg = compare_tensors(tt_logits_torch, ref_logits, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"LM head decode-batch32 PCC too low: {pcc_msg}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 8)])
def test_lm_head_bfp8_weight_decode_batch32(mesh_device, reset_seeds, request):
    """LM head decode matmul with bfp8 weight — DRAM relief for 32×5376×32768 @ TP=8.

    With bf16 weights this matmul is DRAM-bound and a large share of the decode
    step. bfp8 halves weight bandwidth; HiFi4 activations preserve logit quality
    vs the bf16 torch reference.
    """
    tt_logits_torch, ref_logits = _run_lm_head_decode_batch32(
        mesh_device,
        lm_head_dtype=ttnn.bfloat8_b,
        use_decode_config=True,
    )

    passing, pcc_msg = compare_tensors(tt_logits_torch, ref_logits, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"LM head bfp8 decode-batch32 PCC too low: {pcc_msg}"
