# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for Gemma4 lm_head + softcap + all-gather pipeline.

Loads the real lm_head weight (= embed_tokens.weight.T, tied), runs a random
hidden state through the TT path (column-parallel linear → softcap → all-gather)
and compares against a torch reference.

    pytest -k "1x8"   # T3K (TP=8, column-parallel lm_head)
"""


import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager, ccl_allgather

from ...tests.test_factory import _get_model_path, compare_tensors, get_pcc_threshold, parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric()
def test_lm_head(mesh_device, reset_seeds, request):
    """LM head pipeline (linear + softcap + all-gather) vs torch reference."""
    model_path = _get_model_path()
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(hf_config, "text_config", hf_config)
    hidden_size = text_config.hidden_size
    cap = getattr(text_config, "final_logit_softcapping", 0.0) or 0.0

    # Load just the embedding (which is tied to lm_head). full HF model
    # construction is expensive but matches what test_full_model does.
    logger.info(f"Loading HF reference weights from {model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    embed_module = getattr(getattr(hf_model.model, "language_model", hf_model.model), "embed_tokens")
    embed_weight = embed_module.weight.data.clone()  # [vocab, hidden]
    del hf_model
    import gc

    gc.collect()

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
