# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Device PCC tests: TTNN SigLIP2 vision encoder + aligner vs ref/vision/siglip2.py.
# Weights: real HunyuanImage checkpoint (HUNYUAN_MODEL_DIR / ref/weights.MODEL_DIR).
# Inputs: random activations with fixed seed (same pattern as tests/vae/).
#
# Run (fast, 1 encoder layer):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/vision/test_siglip2_ttnn.py -v
#
# Full 27-layer vision stack:
#   HY_VIT_NUM_LAYERS=27 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/vision/test_siglip2_ttnn.py -v

import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import (
    ALIGNER_CONFIG,
    VIT_CONFIG,
    prepare_4d_attention_mask,
)
from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import (
    HunyuanTtLightProjector,
    HunyuanTtSiglip2Attention,
    HunyuanTtSiglip2EncoderLayer,
    HunyuanTtSiglip2MLP,
    HunyuanTtSiglip2Vision,
    HunyuanTtSiglip2VisionEmbeddings,
    build_siglip2_attention_mask,
    forward_vision_with_aligner,
)

from .conftest import (
    B,
    NUM_LAYERS,
    PCC_THR,
    S,
    SPATIAL_SHAPES_HW,
    upload_attention_mask,
    upload_pixel_values,
)

LAYER_IDX = 0


def assert_pcc(ref: torch.Tensor, tt: torch.Tensor, *, label: str) -> float:
    passing, pcc = comp_pcc(ref, tt, PCC_THR)
    logger.info(f"PCC [{label}]: {pcc}  (threshold {PCC_THR}, passing={passing})")
    assert passing, f"PCC [{label}] {pcc} < {PCC_THR}"
    return pcc


def to_torch_squeezed(t: ttnn.Tensor) -> torch.Tensor:
    """Drop singleton dims layer_norm may insert (e.g. [B,1,S,H] -> [B,S,H])."""
    out = ttnn.to_torch(t)
    while out.ndim > 3 and out.shape[1] == 1:
        out = out.squeeze(1)
    return out


def upload_hidden(device, hidden: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _random_hidden() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, S, VIT_CONFIG["hidden_size"], dtype=torch.float32)


def test_attention_mask_tt_matches_host(device, vision_inputs):
    """Device-built 4D mask: zeros on valid keys, large negative on padded keys."""
    _, _, pixel_attention_mask = vision_inputs
    host = prepare_4d_attention_mask(pixel_attention_mask, torch.float32)
    pad_tt = upload_attention_mask(device, pixel_attention_mask)
    dev = build_siglip2_attention_mask(pad_tt)
    dev_torch = ttnn.to_torch(dev).float()
    valid = host == 0
    padded = host < -1e4
    assert valid.any() and padded.any()
    assert torch.allclose(host[valid], dev_torch[valid], atol=1e-2)
    assert (dev_torch[padded] < -1e4).all()


def test_embeddings_pcc(device, ref_vision, vision_inputs, tt_vision_inputs, vision_state_dict):
    pixel_values, spatial_shapes, _ = vision_inputs

    with torch.no_grad():
        pt_out = ref_vision.embeddings(pixel_values, spatial_shapes)

    tt_mod = HunyuanTtSiglip2VisionEmbeddings(device, vision_state_dict)
    tt_mod.prewarm_pos_geometries([(8, 8, S)])
    tt_out = to_torch_squeezed(tt_mod(tt_vision_inputs))

    assert pt_out.shape == tt_out.shape == (B, S, VIT_CONFIG["hidden_size"])
    assert_pcc(pt_out, tt_out, label="embeddings")


def test_attention_pcc_no_mask(device, ref_vision, vision_state_dict):
    ref = ref_vision.encoder.layers[LAYER_IDX].self_attn
    hidden = _random_hidden()

    with torch.no_grad():
        pt_out = ref(hidden, None)

    tt_mod = HunyuanTtSiglip2Attention(device, vision_state_dict, layer_idx=LAYER_IDX)
    tt_out = to_torch_squeezed(tt_mod(upload_hidden(device, hidden), None))
    assert_pcc(pt_out, tt_out, label="attention_no_mask")


def test_attention_pcc_masked(device, ref_vision, vision_inputs, vision_state_dict):
    ref = ref_vision.encoder.layers[LAYER_IDX].self_attn
    _, _, pixel_attention_mask = vision_inputs
    hidden = _random_hidden()
    mask4d = prepare_4d_attention_mask(pixel_attention_mask, hidden.dtype)

    with torch.no_grad():
        pt_out = ref(hidden, mask4d)

    pad_tt = upload_attention_mask(device, pixel_attention_mask)
    dev_mask = build_siglip2_attention_mask(pad_tt)
    tt_mod = HunyuanTtSiglip2Attention(device, vision_state_dict, layer_idx=LAYER_IDX)
    tt_out = to_torch_squeezed(tt_mod(upload_hidden(device, hidden), dev_mask))
    assert_pcc(pt_out, tt_out, label="attention_masked")


def test_mlp_pcc(device, ref_vision, vision_state_dict):
    ref = ref_vision.encoder.layers[LAYER_IDX].mlp
    x = _random_hidden()

    with torch.no_grad():
        pt_out = ref(x)

    tt_mod = HunyuanTtSiglip2MLP(device, vision_state_dict, layer_idx=LAYER_IDX)
    tt_out = to_torch_squeezed(tt_mod(upload_hidden(device, x)))
    assert_pcc(pt_out, tt_out, label="mlp")


def test_encoder_layer_pcc_no_mask(device, ref_vision, vision_state_dict):
    ref = ref_vision.encoder.layers[LAYER_IDX]
    hidden = _random_hidden()

    with torch.no_grad():
        pt_out = ref(hidden, None)

    tt_mod = HunyuanTtSiglip2EncoderLayer(device, vision_state_dict, layer_idx=LAYER_IDX)
    tt_out = to_torch_squeezed(tt_mod(upload_hidden(device, hidden), None))
    assert_pcc(pt_out, tt_out, label="encoder_layer_no_mask")


def test_encoder_layer_pcc_masked(device, ref_vision, vision_inputs, vision_state_dict):
    ref = ref_vision.encoder.layers[LAYER_IDX]
    _, _, pixel_attention_mask = vision_inputs
    hidden = _random_hidden()
    mask4d = prepare_4d_attention_mask(pixel_attention_mask, hidden.dtype)

    with torch.no_grad():
        pt_out = ref(hidden, mask4d)

    pad_tt = upload_attention_mask(device, pixel_attention_mask)
    dev_mask = build_siglip2_attention_mask(pad_tt)
    tt_mod = HunyuanTtSiglip2EncoderLayer(device, vision_state_dict, layer_idx=LAYER_IDX)
    tt_out = to_torch_squeezed(tt_mod(upload_hidden(device, hidden), dev_mask))
    assert_pcc(pt_out, tt_out, label="encoder_layer_masked")


def test_vision_pcc_masked(device, ref_vision, vision_inputs, tt_vision_inputs, vision_state_dict):
    pixel_values, spatial_shapes, pixel_attention_mask = vision_inputs

    with torch.no_grad():
        pt_out = ref_vision(pixel_values, pixel_attention_mask, spatial_shapes)

    tt_mod = HunyuanTtSiglip2Vision(device, vision_state_dict, num_layers=NUM_LAYERS)
    tt_mod.prewarm_pos_geometries([(8, 8, S)])
    tt_out = to_torch_squeezed(tt_mod(tt_vision_inputs))

    assert pt_out.shape == tt_out.shape == (B, S, VIT_CONFIG["hidden_size"])
    assert_pcc(pt_out, tt_out, label=f"vision_masked_{NUM_LAYERS}L")


def test_vision_pcc_no_mask(device, ref_vision, vision_inputs, vision_state_dict):
    pixel_values, spatial_shapes, _ = vision_inputs

    with torch.no_grad():
        pt_out = ref_vision(pixel_values, None, spatial_shapes)

    from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import Siglip2VisionInputs

    tt_mod = HunyuanTtSiglip2Vision(device, vision_state_dict, num_layers=NUM_LAYERS)
    tt_mod.prewarm_pos_geometries([(8, 8, S)])
    all_valid = torch.ones(B, S, dtype=torch.long)
    inputs_nomask = Siglip2VisionInputs.create(
        upload_pixel_values(device, pixel_values),
        SPATIAL_SHAPES_HW,
        upload_attention_mask(device, all_valid),
    )
    tt_out = to_torch_squeezed(tt_mod(inputs_nomask))
    assert_pcc(pt_out, tt_out, label=f"vision_no_mask_{NUM_LAYERS}L")


def test_aligner_pcc(device, ref_aligner, aligner_state_dict):
    x = _random_hidden()

    with torch.no_grad():
        pt_out = ref_aligner(x)

    tt_mod = HunyuanTtLightProjector(device, aligner_state_dict)
    tt_out = to_torch_squeezed(tt_mod(upload_hidden(device, x)))

    assert pt_out.shape == tt_out.shape == (B, S, ALIGNER_CONFIG["n_embed"])
    assert_pcc(pt_out, tt_out, label="aligner")


def test_end_to_end_vision_aligner(
    device, ref_vision, ref_aligner, vision_inputs, tt_vision_inputs, vision_state_dict, aligner_state_dict
):
    pixel_values, spatial_shapes, pixel_attention_mask = vision_inputs

    with torch.no_grad():
        pt_out = ref_aligner(ref_vision(pixel_values, pixel_attention_mask, spatial_shapes))

    vision_tt = HunyuanTtSiglip2Vision(device, vision_state_dict, num_layers=NUM_LAYERS)
    vision_tt.prewarm_pos_geometries([(8, 8, S)])
    aligner_tt = HunyuanTtLightProjector(device, aligner_state_dict)
    tt_out = to_torch_squeezed(forward_vision_with_aligner(vision_tt, aligner_tt, tt_vision_inputs))

    assert pt_out.shape == tt_out.shape == (B, S, ALIGNER_CONFIG["n_embed"])
    assert_pcc(pt_out, tt_out, label=f"e2e_vision_aligner_{NUM_LAYERS}L")
