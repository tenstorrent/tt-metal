# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL-8B text encoder for Ideogram 4.0. Ported from the Qwen2.5-VL encoder
# (encoders/qwen25vl) + the two Qwen3 deltas: per-head QK-RMSNorm and no qkv bias.
# Adds the multi-layer feature tap (raw hidden states after the 13 activation
# layers, no final norm) that Ideogram concatenates and feeds to the DiT.
#
# Verified against the real HF Qwen3-VL-8B language model (public weights). For
# text tokens the 3 MRoPE axes share the same position, so MRoPE collapses to 1D
# RoPE and the qwen25vl rope path is exact.
# =============================================================================

import pytest
import torch
import transformers
from loguru import logger

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder, create_rope_tensors
from ....reference.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

REPO = "Qwen/Qwen3-VL-8B-Instruct"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("seq_len", [128])
def test_qwen3vl_text_encoder(*, mesh_device: ttnn.MeshDevice, seq_len: int) -> None:
    torch.manual_seed(0)

    hf = transformers.AutoModel.from_pretrained(REPO, torch_dtype=torch.bfloat16)
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    lm.eval()
    cfg = lm.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    mrope_section = cfg.rope_scaling["mrope_section"]
    rope_theta = cfg.rope_scaling["rope_theta"]

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    # ---- golden: HF text model, capture RAW decoder-layer outputs at the tap layers ----
    caps: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            caps[i] = (out[0] if isinstance(out, tuple) else out).detach()

        return hook

    handles = [lm.layers[i].register_forward_hook(make_hook(i)) for i in QWEN3_VL_ACTIVATION_LAYERS]
    with torch.no_grad():
        lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for h in handles:
        h.remove()
    golden = [caps[i].float() for i in QWEN3_VL_ACTIVATION_LAYERS]

    # ---- tt encoder ----
    enc = Qwen3VlTextEncoder(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act="silu",
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        activation_layers=QWEN3_VL_ACTIVATION_LAYERS,
        device=mesh_device,
    )
    enc.load_torch_state_dict(lm.state_dict())

    cos, sin = create_rope_tensors(1, seq_len, None, head_dim, rope_theta, mrope_section)
    tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    tt_caps = enc.forward(
        tt_ids,
        attention_mask=None,  # full causal sequence (no padding)
        pos_embeds=(bf16_tensor(cos, device=mesh_device), bf16_tensor(sin, device=mesh_device)),
    )

    assert len(tt_caps) == len(golden)
    worst = 1.0
    for layer_idx, g, tt_t in zip(QWEN3_VL_ACTIVATION_LAYERS, golden, tt_caps):
        tt_torch = tensor.to_torch(tt_t, mesh_axes=[None, None, None])
        # assert_quality raises on failure; track the layer for logging
        logger.info(f"layer {layer_idx}:")
        assert_quality(g, tt_torch, pcc=0.99)
