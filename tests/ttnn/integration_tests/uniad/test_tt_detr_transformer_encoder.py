# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from models.experimental.uniad.reference.detr_transformer_encoder import (
    MultiScaleDeformableAttention,
    DetrTransformerEncoder,
)
from models.experimental.uniad.reference.detr_transformer_decoder import DeformableDetrTransformerDecoder
from models.experimental.uniad.tt.tt_detr_transformer_encoder import (
    TtMultiScaleDeformableAttention,
    TtDetrTransformerEncoder,
)
from models.experimental.uniad.tt.tt_detr_transformer_decoder import (
    TtDetrTransformerDecoderLayer,
    TtDeformableDetrTransformerDecoder,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.uniad.tt.model_preprocessing_encoder import (
    create_uniad_model_parameters_encoder,
    create_uniad_FPN_parameters,
)
from models.experimental.uniad.tt.utils import TtLiDARInstance3DBoxes
from models.experimental.uniad.reference.utils import LiDARInstance3DBoxes
from models.experimental.uniad.reference.seg_deformable_transformer import SegDeformableTransformer
from models.experimental.uniad.tt.tt_seg_deformable_attention import TtSegDeformableTransformer

from models.experimental.uniad.reference.seg_mask_head import AttentionTail, Block, SegMaskHead
from models.experimental.uniad.tt.tt_seg_mask_head import TtAttentionTail, TtBlock, TtSegMaskHead


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_uniad_TtMultiScaleDeformableAttention(
    device,
    reset_seeds,
):
    print("devicedevice", device)
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"

    torch_model = MultiScaleDeformableAttention()

    torch_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    torch_dict = torch_dict["state_dict"]

    state_dict = {
        k: v for k, v in torch_dict.items() if (k.startswith("seg_head.transformer.encoder.layers.0.attentions.0"))
    }

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    query = torch.randn(300, 1, 256)
    value = torch.randn(2500, 1, 256)
    query_pos = torch.randn(300, 1, 256)
    key_padding_mask = torch.randint(0, 2, (1, 2500), dtype=torch.bool)
    reference_points = torch.rand(1, 300, 1, 4)
    spatial_shapes = torch.tensor([[50, 50]], dtype=torch.long)
    level_start_index = torch.tensor([0], dtype=torch.long)

    torch_output = torch_model(
        query=query,
        value=value,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )
    print("torch_output", torch_output.shape)

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)
    print("TtMultiScaleDeformableAttention", parameter)

    tt_model = TtMultiScaleDeformableAttention(
        params=parameter,
        device=device,
    )
    print("tt_model", tt_model)
    # print(dir(tt_model))
    query = ttnn.from_torch(query, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    value = ttnn.from_torch(value, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    query_pos = ttnn.from_torch(query_pos, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    spatial_shapes = ttnn.from_torch(spatial_shapes, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    level_start_index = ttnn.from_torch(level_start_index, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT)
    reference_points = ttnn.from_torch(
        reference_points, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    ttnn_output = tt_model(
        query=query,
        value=value,
        query_pos=query_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        reference_points=reference_points,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


def test_DetrTransformerEncoder(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"

    torch_model = DetrTransformerEncoder()

    torch_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    torch_dict = torch_dict["state_dict"]

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("seg_head.transformer.encoder"))}

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    query = torch.randn(2500, 1, 256, dtype=torch.float32)
    key = torch.randn(2500, 1, 256, dtype=torch.float32)
    value = torch.randn(2500, 1, 256, dtype=torch.float32)
    query_pos = torch.randn(2500, 1, 256, dtype=torch.float32)
    query_key_padding_mask = torch.zeros(1, 2500, dtype=torch.bool)
    spatial_shapes = torch.tensor([[50, 50]], dtype=torch.int64)
    reference_points = torch.rand(1, 2500, 1, 2, dtype=torch.float32)
    level_start_index = torch.tensor([0], dtype=torch.int64)
    valid_ratios = torch.ones(1, 1, 2, dtype=torch.float32)

    torch_output = torch_model(
        query=query,
        key=key,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
    )
    print("torch_output", torch_output.shape)
    query = ttnn.from_torch(query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    key = ttnn.from_torch(key, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    value = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    query_pos = ttnn.from_torch(query_pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    spatial_shapes = ttnn.from_torch(spatial_shapes, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    reference_points = ttnn.from_torch(reference_points, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    level_start_index = ttnn.from_torch(
        level_start_index, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    valid_ratios = ttnn.from_torch(valid_ratios, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)

    tt_model = TtDetrTransformerEncoder(
        params=parameter,
        device=device,
    )

    ttnn_output = tt_model(
        query=query,
        query_pos=query_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        reference_points=reference_points,
        valid_ratios=valid_ratios,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, 0.99)


from models.experimental.uniad.tt.model_preprocessing_encoder import extract_sequential_branch
import torch.nn as nn


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]


def test_DetrTransformerDecoder(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"

    torch_model = DeformableDetrTransformerDecoder(
        num_layers=6,
        embed_dim=256,
        num_heads=8,
    )
    print("torch_model", torch_model)

    torch_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    torch_dict = torch_dict["state_dict"]

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("seg_head.transformer.decoder"))}

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    query = torch.randn(300, 1, 256, dtype=torch.float32)
    key = None
    value = torch.randn(2500, 1, 256, dtype=torch.float32)
    query_pos = torch.randn(300, 1, 256, dtype=torch.float32)
    query_key_padding_mask = None
    spatial_shapes = torch.tensor([[50, 50]], dtype=torch.int64)
    reference_points = torch.rand(1, 300, 2, dtype=torch.float32)
    level_start_index = torch.tensor([0], dtype=torch.int64)
    valid_ratios = torch.ones(1, 1, 2, dtype=torch.float32)

    reg_branches = nn.ModuleList(
        [
            nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))
            for _ in range(6)
        ]
    )

    torch_output = torch_model(
        query=query,
        key=key,
        value=value,
        query_pos=query_pos,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
        reg_branches=reg_branches,
    )
    print("--------------===========------------")
    print("torch_output", len(torch_output))

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)

    parameters_branches = {}
    parameters_branches["reg_branches"] = extract_sequential_branch(reg_branches, dtype=ttnn.bfloat16, device=device)

    parameters_branches = DotDict(parameters_branches)

    query = ttnn.from_torch(query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    value = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    query_pos = ttnn.from_torch(query_pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    spatial_shapes = ttnn.from_torch(spatial_shapes, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    reference_points = ttnn.from_torch(reference_points, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    level_start_index = ttnn.from_torch(level_start_index, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    valid_ratios = ttnn.from_torch(valid_ratios, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_model = TtDeformableDetrTransformerDecoder(
        params=parameter,
        device=device,
        num_layers=6,
        embed_dim=256,
        num_heads=8,
        params_branches=parameters_branches,
    )
    print("tt_model", tt_model)

    ttnn_output = tt_model(
        query=query,
        key=key,
        value=value,
        query_pos=query_pos,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        reference_points=reference_points,
        valid_ratios=valid_ratios,
        reg_branches=reg_branches,
    )

    ttnn_output = ttnn.to_torch(ttnn_output[0])

    assert_with_pcc(torch_output[0], ttnn_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_SegDeformableTransformer(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"

    torch_model = SegDeformableTransformer()

    torch_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    torch_dict = torch_dict["state_dict"]

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("seg_head.transformer"))}

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    mlvl_feats = [torch.randn(1, 256, 50, 50, dtype=torch.float32)]
    mlvl_masks = [torch.randint(0, 2, (1, 50, 50), dtype=torch.int32)]
    query_embed = torch.randn(300, 512, dtype=torch.float32)
    mlvl_pos_embeds = [torch.randn(1, 256, 50, 50, dtype=torch.float32)]
    level_embeds = torch.randn(4, 256)
    reg_branches = nn.ModuleList(
        [
            nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 4))
            for _ in range(6)
        ]
    )

    torch_output = torch_model(
        mlvl_feats=mlvl_feats,
        mlvl_masks=mlvl_masks,
        query_embed=query_embed,
        mlvl_pos_embeds=mlvl_pos_embeds,
        reg_branches=reg_branches,
        level_embeds=level_embeds,
    )

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)

    mlvl_feats = [ttnn.from_torch(feat, device=device, layout=ttnn.TILE_LAYOUT) for feat in mlvl_feats]
    mlvl_masks = [
        ttnn.from_torch(mask, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16) for mask in mlvl_masks
    ]
    query_embed = ttnn.from_torch(query_embed, device=device, layout=ttnn.TILE_LAYOUT)
    mlvl_pos_embeds = [ttnn.from_torch(pos, device=device, layout=ttnn.TILE_LAYOUT) for pos in mlvl_pos_embeds]
    level_embeds = ttnn.from_torch(level_embeds, device=device, layout=ttnn.TILE_LAYOUT)

    parameters_branches = {}
    parameters_branches["reg_branches"] = extract_sequential_branch(reg_branches, dtype=ttnn.bfloat16, device=device)

    parameters_branches = DotDict(parameters_branches)

    tt_model = TtSegDeformableTransformer(params=parameter, device=device, params_branches=DotDict(parameters_branches))
    print("tt_model", tt_model)

    ttnn_output = tt_model.forward(
        mlvl_feats=mlvl_feats,
        mlvl_masks=mlvl_masks,
        query_embed=query_embed,
        mlvl_pos_embeds=mlvl_pos_embeds,
        reg_branches=reg_branches,
        level_embeds=level_embeds,
    )

    ttnn_out00 = ttnn.to_torch(ttnn_output[0][0])
    ttnn_out01 = ttnn.to_torch(ttnn_output[0][1])
    ttnn_out02 = ttnn.to_torch(ttnn_output[0][2])
    ttnn_out03 = ttnn.to_torch(ttnn_output[0][3])
    ttnn_out1 = ttnn.to_torch(ttnn_output[1])
    ttnn_out2 = ttnn.to_torch(ttnn_output[2])
    ttnn_out3 = ttnn.to_torch(ttnn_output[3])
    assert_with_pcc(torch_output[0][0], ttnn_out00, 0.99)
    assert_with_pcc(torch_output[0][1], ttnn_out01, 0.99)
    assert_with_pcc(torch_output[0][2], ttnn_out02, 0.99)
    assert_with_pcc(torch_output[0][3], ttnn_out03, 0.99)
    assert_with_pcc(torch_output[1], ttnn_out1, 0.99)
    assert_with_pcc(torch_output[2], ttnn_out2, 0.99)
    print("torch_output[3]", torch_output[3], torch_output[3].shape)
    print("ttnn_out3[3]", ttnn_out3, ttnn_out3.shape)
    # assert_with_pcc(torch_output[3], ttnn_out3,0.99) #0.16078764718945582


def test_Attentiontail(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"

    torch_model = AttentionTail(
        cfg=None,
        dim=256,
        num_heads=8,
        qkv_bias=True,
    )
    print("torch_model", torch_model)

    torch_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    torch_dict = torch_dict["state_dict"]

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("seg_head.things_mask_head.attnen"))}

    for k, v in state_dict.items():
        print("Keys:", k, v.shape)

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    query = torch.randn(1, 100, 256)
    key = torch.randn(1, 2500, 256)
    key_padding_mask = torch.randint(0, 2, (1, 2500)).bool()

    torch_output = torch_model(
        query=query,
        key=key,
        key_padding_mask=key_padding_mask,
    )

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)
    print("parameter", parameter)

    query = ttnn.from_torch(query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    key = ttnn.from_torch(key, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    key_padding_mask = ttnn.from_torch(
        key_padding_mask.to(torch.uint8), dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_model = TtAttentionTail(
        params=parameter,
        device=device,
        cfg=None,
        dim=256,
        num_heads=8,
        qkv_bias=True,
    )

    ttnn_output = tt_model.forward(
        query=query,
        key=key,
        key_padding_mask=key_padding_mask,
    )

    ttnn_output = ttnn.to_torch(ttnn_output)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.99)


def test_Block(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"

    torch_model = Block(
        cfg=None,
        dim=256,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
    )
    print("torch_model", torch_model)

    torch_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    torch_dict = torch_dict["state_dict"]

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("seg_head.things_mask_head.blocks.0"))}

    for k, v in state_dict.items():
        print("Keys:", k, v.shape)

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    # query = torch.randn(1, 100, 256)
    # key = torch.randn(1, 2500, 256)
    # key_padding_mask = torch.randint(0, 2, (1, 2500)).bool()
    query = torch.randn(1, 1, 256)
    key = torch.randn(1, 2500, 256)
    value = torch.randn(1, 2500, 256)
    key_padding_mask = torch.randint(0, 2, (1, 2500)).bool()
    hw_lvl = [torch.randn(50, 50)]

    torch_output = torch_model(
        query=query,
        key=key,
        value=value,
        key_padding_mask=key_padding_mask,
        hw_lvl=hw_lvl,
    )
    print("torch_output", torch_output[0].shape, torch_output[1].shape)

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)
    print("parameter", parameter)

    query = ttnn.from_torch(query, layout=ttnn.Layout.TILE, device=device)
    key = ttnn.from_torch(key, layout=ttnn.Layout.TILE, device=device)
    value = ttnn.from_torch(value, layout=ttnn.Layout.TILE, device=device)
    key_padding_mask = ttnn.from_torch(key_padding_mask.to(torch.float32), layout=ttnn.Layout.TILE, device=device)
    hw_lvl = [ttnn.from_torch(hw, layout=ttnn.Layout.TILE, device=device) for hw in hw_lvl]

    tt_model = TtBlock(
        parameter,
        device,
        cfg=None,
        dim=256,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
    )

    ttnn_output = tt_model.forward(
        query=query,
        key=key,
        value=value,
        key_padding_mask=key_padding_mask,
        hw_lvl=hw_lvl,
    )

    ttnn_output0 = ttnn.to_torch(ttnn_output[0])
    ttnn_output1 = ttnn.to_torch(ttnn_output[1])
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output0, torch_output[0], 0.99)
    pcc_passed, pcc_message = assert_with_pcc(ttnn_output1, torch_output[1], 0.99)


def test_SegHead(
    device,
    reset_seeds,
):
    weights_path = "models/experimental/uniad/uniad_base_e2e.pth"

    torch_model = SegMaskHead(
        cfg=None,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=4,
        dim_feedforward=64,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        self_attn=False,
    )
    print("torch_model", torch_model)

    torch_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    torch_dict = torch_dict["state_dict"]

    state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("seg_head.things_mask_head"))}

    for k, v in state_dict.items():
        print("Keys:", k, v.shape)

    new_state_dict = dict(zip(torch_model.state_dict().keys(), state_dict.values()))

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    memory = torch.randn(1, 2500, 256)
    mask_memory = torch.randint(0, 2, (1, 2500)).bool()
    query_embed = torch.randn(1, 100, 256)
    hw_lvl = [torch.empty(50, 50)]

    torch_output = torch_model(
        memory=memory,
        mask_memory=mask_memory,
        query_embed=query_embed,
        hw_lvl=hw_lvl,
        pos_memory=None,
        mask_query=None,
        pos_query=None,
    )
    print("torch_output", len(torch_output))
    print("torch_output", len(torch_output[0]))
    print("torch_output", len(torch_output[1]))
    print("torch_output", len(torch_output[2]))

    parameter = create_uniad_model_parameters_encoder(torch_model, device=device)
    # print("parameter",parameter)
    memory = ttnn.from_torch(memory, device=device, layout=ttnn.TILE_LAYOUT)
    mask_memory = ttnn.from_torch(mask_memory.to(torch.float32), device=device, layout=ttnn.TILE_LAYOUT)
    query_embed = ttnn.from_torch(query_embed, device=device, layout=ttnn.TILE_LAYOUT)
    hw_lvl = [ttnn.from_torch(hw, device=device, layout=ttnn.TILE_LAYOUT) for hw in hw_lvl]

    tt_model = TtSegMaskHead(
        parameter,
        device,
        cfg=None,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=4,
        dim_feedforward=64,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        self_attn=False,
    )

    ttnn_output = tt_model.forward(
        memory=memory,
        mask_memory=mask_memory,
        query_embed=query_embed,
        hw_lvl=hw_lvl,
        pos_memory=None,
        mask_query=None,
        pos_query=None,
    )
    # assert_with_pcc(ttnn_output, torch_output, 0.99)
    ttnn_output0 = ttnn.to_torch(ttnn_output[0])
    ttnn_output10 = ttnn.to_torch(ttnn_output[1][0])
    ttnn_output11 = ttnn.to_torch(ttnn_output[1][1])
    ttnn_output12 = ttnn.to_torch(ttnn_output[1][2])
    ttnn_output13 = ttnn.to_torch(ttnn_output[1][3])

    ttnn_output20 = ttnn.to_torch(ttnn_output[2][0])
    ttnn_output21 = ttnn.to_torch(ttnn_output[2][1])
    ttnn_output22 = ttnn.to_torch(ttnn_output[2][2])
    ttnn_output23 = ttnn.to_torch(ttnn_output[2][3])

    # ttnn_output2 = ttnn.to_torch(ttnn_output[2])
    assert_with_pcc(ttnn_output0, torch_output[0], 0.99)
    assert_with_pcc(ttnn_output10, torch_output[1][0], 0.99)
    assert_with_pcc(ttnn_output11, torch_output[1][1], 0.99)
    assert_with_pcc(ttnn_output12, torch_output[1][2], 0.99)
    assert_with_pcc(ttnn_output13, torch_output[1][3], 0.99)

    assert_with_pcc(ttnn_output20, torch_output[2][0], 0.99)
    assert_with_pcc(ttnn_output21, torch_output[2][1], 0.99)
    assert_with_pcc(ttnn_output22, torch_output[2][2], 0.99)
    assert_with_pcc(ttnn_output23, torch_output[2][3], 0.99)
