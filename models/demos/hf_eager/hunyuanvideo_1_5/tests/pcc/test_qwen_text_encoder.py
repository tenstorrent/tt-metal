# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight Qwen valid/padding PCC and mixed-length conditioning evidence."""

import glob
import os

import pytest
import torch

import ttnn
from models.demos.hf_eager.hunyuanvideo_1_5.tt.pipeline import HunyuanVideo15Pipeline
from models.demos.hf_eager.hunyuanvideo_1_5.tt.qwen_encoder import TTQwenTextEncoderAdapter

_MODEL = "models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v"


def _text_encoder_dir():
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    matches = glob.glob(os.path.join(hub, _MODEL, "snapshots", "*", "text_encoder"))
    return matches[0] if matches else None


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    return 0.0 if denominator == 0 else float(torch.dot(a, b) / denominator)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_qwen_valid_padding_and_mixed_length_pcc(mesh_device):
    encoder_dir = _text_encoder_dir()
    if not encoder_dir:
        pytest.skip(f"{_MODEL} text_encoder is not present in the local HF cache")

    from transformers import Qwen2_5_VLTextModel

    host = Qwen2_5_VLTextModel.from_pretrained(encoder_dir, torch_dtype=torch.bfloat16, local_files_only=True).eval()
    torch.manual_seed(0)
    ids = torch.randint(0, host.config.vocab_size, (2, 32))
    mask = torch.tensor([[1] * 13 + [0] * 19, [1] * 7 + [0] * 25])
    with torch.no_grad():
        reference = host(ids, attention_mask=mask, output_hidden_states=True).hidden_states[-3]

    adapter = TTQwenTextEncoderAdapter(host, mesh_device)
    actual = adapter(ids, attention_mask=mask).hidden_states[-3]
    valid = mask.bool().unsqueeze(-1).expand_as(reference)
    padding = ~valid
    valid_pcc = _pcc(reference[valid], actual[valid])
    padding_pcc = _pcc(reference[padding], actual[padding])
    print(f"Qwen valid PCC={valid_pcc:.6f}; padding PCC={padding_pcc:.6f}", flush=True)
    assert valid_pcc >= 0.999

    # Padding values intentionally differ (TT zeroes them), but padding isolation
    # removes them before the DiT. Compare the exact per-row tensors that reach it.
    for row in range(ids.shape[0]):
        ref_row, _ = HunyuanVideo15Pipeline._trim_to_valid(reference[row : row + 1], mask[row : row + 1])
        tt_row, _ = HunyuanVideo15Pipeline._trim_to_valid(actual[row : row + 1], mask[row : row + 1])
        assert _pcc(ref_row, tt_row) >= 0.999
