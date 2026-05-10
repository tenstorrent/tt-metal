# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN RMS norm stubs vs HF ``Mistral4RMSNorm`` with hub BF16 weights.

Parametrizes ``layer_idx`` in ``{0, 1}`` for ``input_layernorm`` and ``post_attention_layernorm``.
"""

import os

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    EXPECTED_HIDDEN_SIZE,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tt.text_backbone import (
    TtMistral4DecoderLayerInputNormStub,
    TtMistral4DecoderLayerPostAttnNormStub,
)
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


pytest.importorskip("transformers")
pytest.importorskip(
    "transformers.models.mistral4.modeling_mistral4", reason="Mistral4 RMSNorm requires recent transformers"
)


def _text_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")
    text = cfg.text_config
    if text is None:
        pytest.skip("Config has no text_config")
    return text


def _layer_state_dict(layer_idx: int):
    prefix = text_decoder_layer_state_dict_prefix(layer_idx)
    try:
        return load_hf_state_dict_filtered(HF_MODEL_ID, (prefix,))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"No checkpoint shards for layer {layer_idx} weights: {exc}")


_STUBS = (
    ("input_layernorm", TtMistral4DecoderLayerInputNormStub),
    ("post_attention_layernorm", TtMistral4DecoderLayerPostAttnNormStub),
)


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize(
    "device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8, 32))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize("layer_idx", (0, 1), ids=("layer0", "layer1"))
@pytest.mark.parametrize("weight_key,stub_cls", _STUBS, ids=[s[0] for s in _STUBS])
def test_mistral_small_4_text_layer0_rmsnorm_stub_pcc(
    seq_len, batch_size, layer_idx, weight_key, stub_cls, reset_seeds, device
):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RMSNorm

    text = _text_config()
    state_dict = _layer_state_dict(layer_idx)
    prefix = text_decoder_layer_state_dict_prefix(layer_idx)
    wkey = f"{prefix}{weight_key}.weight"
    assert wkey in state_dict, f"missing {wkey} in filtered checkpoint"

    eps = float(text.rms_norm_eps)
    weight = state_dict[wkey]
    assert weight.shape == (EXPECTED_HIDDEN_SIZE,)

    ref_norm = Mistral4RMSNorm(EXPECTED_HIDDEN_SIZE, eps=eps).eval()
    ref_norm.weight.data.copy_(weight)

    dtype = ttnn.bfloat16
    mode = "decode" if seq_len <= 32 else "prefill"

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, EXPECTED_HIDDEN_SIZE, dtype=torch.bfloat16)
    reference_output = ref_norm(x.float()).to(torch.bfloat16)

    tt_stub = stub_cls(device, state_dict, layer_idx=layer_idx, text_config=text, weight_dtype=dtype, eps=eps)

    tt_input = ttnn.from_torch(
        x,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=device),
    )
    tt_output = tt_stub(tt_input, mode=mode)
    from models.common.auto_compose import to_torch_auto_compose

    tt_output_torch = to_torch_auto_compose(tt_output, device=device)

    logger.info(f"{weight_key} tt_output_torch: {tt_output_torch.shape}")
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, 0.99)
    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"{weight_key} PCC: {pcc_message}")

    assert passing, f"{weight_key} stub PCC failed: {pcc_message}"
