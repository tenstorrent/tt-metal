# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest
import torch

import ttnn

# Add reference repo to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../ref/CosyVoice")))
from hyperpyyaml import load_hyperpyyaml

from models.demos.wormhole.cosy_voice.tt.cosyvoice_llm import CosyVoice3LM
from models.demos.wormhole.cosy_voice.tt.model_config import CosyVoiceModelConfig
from models.utility_functions import comp_pcc


def get_reference_llm(model_dir):
    yaml_path = f"{model_dir}/cosyvoice3.yaml"
    if not os.path.exists(yaml_path):
        pytest.skip(f"Reference weights not found at {model_dir}")

    with open(yaml_path, "r") as f:
        # CosyVoice3 specific overrides
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": f"{model_dir}/CosyVoice-BlankEN"})

    llm = configs["llm"]
    # Load weights
    state_dict = torch.load(f"{model_dir}/llm.pt", map_location="cpu", weights_only=True)
    llm.load_state_dict(state_dict)
    llm.eval()
    return llm


@pytest.fixture(scope="module")
def reference_llm():
    model_dir = "/root/tt-metal/models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"
    return get_reference_llm(model_dir)


@pytest.fixture(scope="module")
def cosyvoice_llm_ttnn(mesh_device):
    model_dir = "/root/tt-metal/models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"
    if not os.path.exists(f"{model_dir}/llm.pt"):
        pytest.skip(f"Weights not found at {model_dir}")

    config = CosyVoiceModelConfig(mesh_device=mesh_device, max_batch_size=1, weights_dir=model_dir)
    llm = CosyVoice3LM(config, mesh_device, dtype=ttnn.bfloat8_b)
    return llm


def test_cosyvoice_llm_prefill_pcc(reference_llm, cosyvoice_llm_ttnn):
    """
    Test that the TTNN implementation's prefill logits match the PyTorch reference logits.
    """
    # 1. Setup inputs
    prompt_text = torch.tensor([[100, 151646, 101, 102]], dtype=torch.int32)
    prompt_speech_token = torch.tensor([[10, 11, 12]], dtype=torch.int32)

    # 2. Run Reference (PyTorch)
    # The reference formats the embeddings using its internal embedding layer
    # SOS + text + TASK_ID + speech
    sos_emb = reference_llm.speech_embedding(torch.tensor([[reference_llm.sos]])).unsqueeze(0)  # [1, 1, dim]
    task_id_emb = reference_llm.speech_embedding(torch.tensor([[reference_llm.task_id]])).unsqueeze(0)

    # Text uses the LLM's embed_tokens
    text_emb = reference_llm.llm.model.embed_tokens(prompt_text)
    speech_emb = reference_llm.speech_embedding(prompt_speech_token)

    lm_input_ref = torch.cat([sos_emb.squeeze(1), text_emb, task_id_emb.squeeze(1), speech_emb], dim=1)

    with torch.no_grad():
        ref_output = reference_llm.llm(inputs_embeds=lm_input_ref, use_cache=False)
        ref_hidden_states = ref_output.last_hidden_state
        # logit for the last token in the sequence
        ref_logits = reference_llm.llm_decoder(ref_hidden_states[:, -1])
        ref_logp = ref_logits.log_softmax(dim=-1)

    # 3. Run TTNN
    # cosyvoice_llm_ttnn.format_prompt_embeddings expects 2D tensors
    lm_input_ttnn = cosyvoice_llm_ttnn.format_prompt_embeddings(prompt_text, prompt_speech_token)

    # Verify embeddings match before running forward pass
    pcc_passed, pcc_message = comp_pcc(lm_input_ref, lm_input_ttnn)
    assert pcc_passed, f"Embeddings PCC failed: {pcc_message}"

    # Let's run just the prefill
    device = cosyvoice_llm_ttnn.mesh_device
    batch_size = 1
    seq_len = lm_input_ttnn.shape[1]

    # Setup cache
    cosyvoice_llm_ttnn.model.args.max_batch_size = batch_size
    kv_cache = cosyvoice_llm_ttnn.model.setup_kv_cache(batch_size, cosyvoice_llm_ttnn.config.max_seq_len)

    (
        tt_tokens_embd,
        tt_rot_mats_prefill_global,
        tt_rot_mats_prefill_local,
        tt_page_table,
        tt_chunk_page_table,
    ) = cosyvoice_llm_ttnn.model.prepare_inputs_prefill_embeddings(
        lm_input_ttnn,
        start_pos=0,
        batch_size=batch_size,
    )

    tt_hidden_states = cosyvoice_llm_ttnn.model.ttnn_prefill_forward(
        tt_tokens_embd,
        rot_mats_global=tt_rot_mats_prefill_global,
        rot_mats_local=tt_rot_mats_prefill_local,
        user_id=0,
        page_table=tt_page_table,
        chunk_page_table=tt_chunk_page_table,
        get_last_token=seq_len - 1,
        kv_cache=kv_cache,
        batch_size=batch_size,
    )

    hidden_states_torch = ttnn.to_torch(tt_hidden_states)
    ttnn_logits = hidden_states_torch[0, 0, 0, :]
    ttnn_logp = torch.log_softmax(ttnn_logits, dim=-1)

    # 4. Compare
    pcc_passed, pcc_message = comp_pcc(ref_logp.squeeze(), ttnn_logp)
    assert pcc_passed, f"Prefill logp PCC failed: {pcc_message}"
