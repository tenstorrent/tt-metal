# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Module-by-module PCC: HuggingFace ``Molmo2ForConditionalGeneration`` (text stack) vs TT ``TextModel``.

**Separate** from ``test_pcc_e2e.py`` (end-to-end logits only): this test walks the decoder
with the same layout as HF ``output_hidden_states`` and reports Pearson correlation at
each stage so you can see **where** TTNN drifts (embed, block 0..34, after block 35 is TT-only
extra, final norm, logits).

Usage::

    pytest models/demos/molmo2/tests/test_hf_ttnn_text_module_pcc.py -v -s

Env::

    MOLMO2_PCC_ATOL=0.0          # optional, passed to comp_pcc if that API is used
    HF_MODEL=allenai/Molmo2-8B   # same as other molmo2 tests
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors
from models.demos.molmo2.tt.text_model import TextModel


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().contiguous()
    b = b.float().contiguous()
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")
    # ``pcc=0.0`` so we always get the measured correlation in the return tuple
    _ok, pcc_val = comp_pcc(a, b, pcc=0.0)
    return float(pcc_val)


def _mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_shape = ttnn.MeshShape(1, 8)
    return ttnn.open_mesh_device(mesh_shape)


@pytest.fixture(scope="module")
def mesh_device():
    d = _mesh()
    yield d
    ttnn.close_mesh_device(d)


@pytest.fixture(scope="module")
def hf_text_model():
    from transformers import AutoModelForImageTextToText

    model_id = os.environ.get("HF_MODEL", "allenai/Molmo2-8B")
    m = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        os.environ.get("HF_MODEL", "allenai/Molmo2-8B"),
        trust_remote_code=True,
    )


def test_molmo2_text_hidden_states_module_pcc(mesh_device, hf_text_model, tokenizer):
    """
    Compare HF ``ForConditionalGeneration.model`` forward hidden states to
    ``TextModel.forward_collect_hidden_states_torch``.

    HF (``modeling_molmo2``) appends at the **start** of each of 36 block iterations, then
    one more after ``ln_f`` → 37 tensors: ``[0]`` = embed, ``[i]`` = after block ``i-1`` for
    ``1 <= i <= 35``, ``[36]`` = after ``ln_f`` (``last_hidden_state``).

    TT returns 38 tensors: ``[0]`` = embed, ``[k]`` = after block ``k-1`` for ``1<=k<=35``,
    ``[36]`` = after block 35, ``[37]`` = after ``ln_f``.

    We compare ``HF[i]`` to ``TT[i]`` for ``i=0..35`` and ``HF[36]`` to ``TT[37]``.
    """
    prompt = os.environ.get("MOLMO2_PCC_TEXT_PROMPT", "What is the capital of France?")
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    with torch.no_grad():
        o = hf_text_model.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    hf_h = o.hidden_states
    assert len(hf_h) == 37, f"expected 37 HF hidden states, got {len(hf_h)}"

    state_dict = load_state_dict_from_safetensors(os.environ.get("HF_MODEL", "allenai/Molmo2-8B"))
    ttnn_text = TextModel(
        mesh_device=mesh_device,
        state_dict=state_dict,
        dtype=ttnn.bfloat8_b,
    )
    tt_h = ttnn_text.forward_collect_hidden_states_torch(input_ids)
    assert len(tt_h) == 38, f"expected 38 TT hiddens, got {len(tt_h)}"

    # Align shapes to [B, S, H]
    for i, t in enumerate(tt_h):
        assert t.dim() == 3, f"tt_h[{i}] has shape {t.shape}"

    rows: list[tuple[str, float]] = []

    for i in range(36):
        a = hf_h[i].float()
        b = tt_h[i].float()
        p = _pcc(a, b)
        name = "inputs_embed" if i == 0 else f"after block {i - 1}"
        rows.append((name, p))
        logger.info(f"  PCC {name}: {p:.5f}")
        if p < 0.90:
            logger.warning(f"LOW PCC < 0.90 at stage '{name}': pcc={p:.5f}")

    # After ln_f: HF[36] vs TT[37]
    p = _pcc(hf_h[36].float(), tt_h[37].float())
    rows.append(("after ln_f (last_hidden)", p))
    logger.info(f"  PCC after ln_f: {p:.5f}")
    if p < 0.90:
        logger.warning(f"LOW PCC after ln_f: pcc={p:.5f}")

    # Logits: last position
    is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    input_ids_ttnn = ttnn.from_torch(
        input_ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    h = ttnn_text.embed_tokens(input_ids_ttnn)
    ttnn.deallocate(input_ids_ttnn)
    logits, _ = ttnn_text.forward(h)
    if is_mesh:
        t_logits = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    else:
        t_logits = ttnn.to_torch(logits)
    t_logits = t_logits.float()
    if t_logits.dim() == 4 and t_logits.shape[0] == 1 and t_logits.shape[1] == 1:
        t_logits = t_logits.squeeze(0)
    with torch.no_grad():
        hf_logits = hf_text_model(input_ids).logits.float()
    p_log = _pcc(hf_logits[0, -1, :], t_logits[0, -1, :])
    logger.info(f"  PCC last-position logits: {p_log:.5f}")
    rows.append(("logits (last pos)", p_log))

    if p_log < 0.85:
        logger.warning("LOW logits PCC on last position — inspect rows above for first drop.")

    # Single assertion: embedding must be strong (sanity)
    assert rows[0][1] > 0.98, f"Embedding PCC should be > 0.98, got {rows[0][1]}"
