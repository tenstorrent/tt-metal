# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase-5 e2e integration check for VoxtralTransformer: build the full model (text decoder + audio
tower), run prepare_inputs_prefill on an audio+text input, and compare the post-scatter embeddings
against HF (get_input_embeddings + get_audio_features + masked_scatter). This validates the assembled
audio-injection path (audio tower loading from the stashed weights + the scatter); the text decoder
itself is validated separately (test_model.py)."""
import pytest
import torch
import transformers

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.multimodal.voxtral.voxtral_e2e_model import VoxtralTransformer
from tests.ttnn.utils_for_testing import comp_pcc

MODEL_NAME = "mistralai/Voxtral-Mini-3B-2507"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "fabric_config": True}], indirect=True)
def test_voxtral_e2e_injection(mesh_device):
    torch.manual_seed(0)
    full = transformers.VoxtralForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).eval()
    cfg = full.config
    n_audio = 375  # 1500 encoder frames / 4 (reshape groups 4)

    from transformers import AutoFeatureExtractor

    fe = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    sr = fe.sampling_rate
    t = torch.arange(int(sr * 30.0)) / sr
    wav = (0.5 * torch.sin(2 * torch.pi * 220 * t) + 0.3 * torch.sin(2 * torch.pi * 440 * t)).numpy()
    feat = fe([wav], sampling_rate=sr, return_tensors="pt").input_features.to(torch.float32)

    pre, post = [1, 100, 200, 300], [400, 500]
    input_ids = torch.tensor([pre + [cfg.audio_token_id] * n_audio + post])

    with torch.no_grad():
        ie = full.get_input_embeddings()(input_ids)
        af = full.get_audio_features(feat).pooler_output
        mask = (input_ids == cfg.audio_token_id).unsqueeze(-1).expand_as(ie)
        golden = ie.masked_scatter(mask, af.to(ie.dtype)).squeeze(0)  # [S, hidden]

    args = ModelArgs(mesh_device, max_batch_size=1, max_seq_len=1024)
    sd = args.load_state_dict()
    model = VoxtralTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
        state_dict=sd,
        weight_cache_path=args.weight_cache_path(ttnn.bfloat8_b),
    )
    tokens_embd, *_ = model.prepare_inputs_prefill(input_ids, input_features=feat)
    tt = ttnn.to_torch(tokens_embd, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    tt = tt.reshape(-1, tt.shape[-1])[: golden.shape[0], : golden.shape[-1]]

    passed, pcc = comp_pcc(golden, tt, 0.97)
    print(f"Voxtral e2e post-scatter embeddings PCC: {pcc}")
    assert passed, f"Voxtral e2e injection PCC too low: {pcc}"
