# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger

# Import reference MapTRHead from MapTR reference folder
from models.experimental.MapTR.reference.maptr import MapTRHead
from models.experimental.MapTR.resources.download_chkpoint import ensure_checkpoint_downloaded, MAPTR_WEIGHTS_PATH

# Import TTNN implementation
from models.experimental.MapTR.tt.ttnn_head import TtMapTRHead

# Import utilities
from models.common.utility_functions import comp_pcc
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
)

# Layer prefix for MapTRHead in MapTR checkpoint
HEAD_LAYER_PREFIX = "pts_bbox_head."


class ConfigDict(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[name] = value
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class ParamsWrapper:
    def __init__(self, d):
        self._dict = {}
        for k, v in d.items():
            if isinstance(v, dict):
                wrapped = ParamsWrapper(v)
                if str(k).isidentifier():
                    setattr(self, k, wrapped)
                self._dict[k] = wrapped
            else:
                if str(k).isidentifier():
                    setattr(self, k, v)
                self._dict[k] = v

    def __getitem__(self, key):
        key_str = str(key)
        if key_str.isidentifier() and hasattr(self, key_str):
            return getattr(self, key_str)
        return self._dict[key]

    def __contains__(self, key):
        key_str = str(key)
        if key_str.isidentifier() and hasattr(self, key_str):
            return True
        return key in self._dict


def load_maptr_head_weights(weights_path: str = MAPTR_WEIGHTS_PATH):
    ensure_checkpoint_downloaded(weights_path)

    checkpoint = torch.load(weights_path, map_location="cpu")
    full_state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    head_weights = {}
    for key, value in full_state_dict.items():
        if key.startswith(HEAD_LAYER_PREFIX):
            relative_key = key[len(HEAD_LAYER_PREFIX) :]
            head_weights[relative_key] = value

    logger.info(f"Loaded {len(head_weights)} weight tensors for MapTRHead")
    return head_weights


def custom_preprocessor(model, name):
    parameters = {}

    def extract_sequential_branch(module_list, dtype):
        branch_params = {}

        for i, mod in enumerate(module_list):
            layer_params = {}
            layer_index = 0

            if isinstance(mod, nn.Sequential):
                layers = mod
            else:
                layers = [mod]

            for layer in layers:
                if isinstance(layer, nn.Linear):
                    layer_params[str(layer_index)] = {
                        "weight": preprocess_linear_weight(layer.weight, dtype=dtype),
                        "bias": preprocess_linear_bias(layer.bias, dtype=dtype),
                    }
                    layer_index += 1
                elif isinstance(layer, nn.LayerNorm):
                    layer_params[f"{layer_index}_norm"] = {
                        "weight": preprocess_layernorm_parameter(layer.weight, dtype=dtype),
                        "bias": preprocess_layernorm_parameter(layer.bias, dtype=dtype),
                    }
                    layer_index += 1

            branch_params[str(i)] = layer_params

        return branch_params

    if isinstance(model, MapTRHead):
        parameters["head"] = {}

        if hasattr(model, "positional_encoding") and model.positional_encoding is not None:
            pos_encoding = model.positional_encoding
            parameters["head"]["positional_encoding"] = {
                "row_embed": {
                    "weight": ttnn.from_torch(
                        pos_encoding.row_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )
                },
                "col_embed": {
                    "weight": ttnn.from_torch(
                        pos_encoding.col_embed.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                    )
                },
            }

        if hasattr(model, "bev_embedding") and model.bev_embedding is not None:
            parameters["head"]["bev_embedding"] = {
                "weight": ttnn.from_torch(model.bev_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }

        if hasattr(model, "instance_embedding") and model.instance_embedding is not None:
            parameters["head"]["instance_embedding"] = {
                "weight": ttnn.from_torch(model.instance_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }

        if hasattr(model, "pts_embedding") and model.pts_embedding is not None:
            parameters["head"]["pts_embedding"] = {
                "weight": ttnn.from_torch(model.pts_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }

        if hasattr(model, "query_embedding") and model.query_embedding is not None:
            parameters["head"]["query_embedding"] = {
                "weight": ttnn.from_torch(model.query_embedding.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            }

        parameters["head"]["branches"] = {
            "cls_branches": extract_sequential_branch(model.cls_branches, dtype=ttnn.bfloat16),
            "reg_branches": extract_sequential_branch(model.reg_branches, dtype=ttnn.bfloat16),
        }

    return parameters


def create_maptr_model_parameters_head(model: MapTRHead, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    return ParamsWrapper(parameters.get("head", {}))


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_maptr_head(device, reset_seeds):
    torch.manual_seed(42)
    embed_dims = 256
    num_classes = 3
    num_vec = 50
    num_pts_per_vec = 20
    num_decoder_layers = 6
    num_reg_fcs = 2
    code_size = 2
    code_weights = [1.0, 1.0, 1.0, 1.0]
    bev_h, bev_w = 200, 100
    pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    batch_size = 1
    num_query = num_vec * num_pts_per_vec

    transformer_cfg = ConfigDict(
        type="MapTRPerceptionTransformer",
        embed_dims=embed_dims,
        encoder=ConfigDict(
            type="BEVFormerEncoder",
            num_layers=1,
            pc_range=pc_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayer=ConfigDict(
                type="BEVFormerLayer",
                attn_cfgs=[
                    ConfigDict(type="TemporalSelfAttention", embed_dims=embed_dims, num_levels=1),
                    ConfigDict(
                        type="SpatialCrossAttention",
                        pc_range=pc_range,
                        deformable_attention=ConfigDict(
                            type="MSDeformableAttention3D", embed_dims=embed_dims, num_points=8, num_levels=1
                        ),
                        embed_dims=embed_dims,
                    ),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
        decoder=ConfigDict(
            type="MapTRDecoder",
            num_layers=num_decoder_layers,
            return_intermediate=True,
            transformerlayer=ConfigDict(
                type="DetrTransformerDecoderLayer",
                attn_cfgs=[
                    ConfigDict(type="MultiheadAttention", embed_dims=embed_dims, num_heads=8, dropout=0.1),
                    ConfigDict(type="CustomMSDeformableAttention", embed_dims=embed_dims, num_levels=1),
                ],
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
        ),
    )

    bbox_coder_cfg = ConfigDict(
        type="MapTRNMSFreeCoder",
        pc_range=pc_range,
        post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
        max_num=50,
        num_classes=num_classes,
    )

    head_weights = load_maptr_head_weights()
    torch_model = MapTRHead(
        num_classes=num_classes,
        in_channels=embed_dims,
        embed_dims=embed_dims,
        num_query=num_query,
        num_reg_fcs=num_reg_fcs,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=code_size,
        code_weights=code_weights,
        bev_h=bev_h,
        bev_w=bev_w,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_pts_per_gt_vec=num_pts_per_vec,
        query_embed_type="instance_pts",
        transform_method="minmax",
        gt_shift_pts_pattern="v0",
        dir_interval=1,
        transformer=transformer_cfg,
        bbox_coder=bbox_coder_cfg,
        loss_cls=ConfigDict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=ConfigDict(type="L1Loss", loss_weight=0.0),
        loss_iou=ConfigDict(type="GIoULoss", loss_weight=0.0),
        loss_pts=None,
        loss_dir=None,
        train_cfg=None,
        test_cfg=ConfigDict(max_per_img=50),
    )

    torch_model.load_state_dict(head_weights, strict=False)
    torch_model.eval()

    # Create test inputs
    torch.manual_seed(42)

    hs = torch.randn(num_decoder_layers, num_query, batch_size, embed_dims) * 0.1
    init_reference = torch.rand(batch_size, num_query, 2) * 0.8 + 0.1
    inter_references = [torch.rand(batch_size, num_query, 2) * 0.8 + 0.1 for _ in range(num_decoder_layers - 1)]

    with torch.no_grad():
        hs_permuted = hs.permute(0, 2, 1, 3)

        outputs_classes_torch = []
        outputs_coords_torch = []
        outputs_pts_coords_torch = []

        for lvl in range(num_decoder_layers):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference_inv = torch.log(reference.clamp(1e-5, 1 - 1e-5) / (1 - reference.clamp(1e-5, 1 - 1e-5)))

            hs_lvl = hs_permuted[lvl]
            hs_reshaped = hs_lvl.view(batch_size, num_vec, num_pts_per_vec, -1)
            hs_mean = hs_reshaped.mean(dim=2)
            outputs_class = torch_model.cls_branches[lvl](hs_mean)

            tmp = torch_model.reg_branches[lvl](hs_lvl)
            tmp_xy = tmp[..., 0:2]
            ref_xy = reference_inv[..., 0:2]
            tmp_updated = (tmp_xy + ref_xy).sigmoid()
            outputs_coord, outputs_pts_coord = torch_model.transform_box(tmp_updated)

            outputs_classes_torch.append(outputs_class)
            outputs_coords_torch.append(outputs_coord)
            outputs_pts_coords_torch.append(outputs_pts_coord)

        outputs_classes_torch = torch.stack(outputs_classes_torch, dim=0)
        outputs_coords_torch = torch.stack(outputs_coords_torch, dim=0)
        outputs_pts_coords_torch = torch.stack(outputs_pts_coords_torch, dim=0)

    params = create_maptr_model_parameters_head(torch_model, device=device)
    tt_model = TtMapTRHead(
        params=params,
        device=device,
        transformer=None,
        positional_encoding=None,
        embed_dims=embed_dims,
        num_classes=num_classes,
        num_reg_fcs=num_reg_fcs,
        code_size=code_size,
        bev_h=bev_h,
        bev_w=bev_w,
        pc_range=pc_range,
        num_vec=num_vec,
        num_pts_per_vec=num_pts_per_vec,
        num_decoder_layers=num_decoder_layers,
        query_embed_type="instance_pts",
        transform_method="minmax",
    )

    hs_tt = ttnn.from_torch(hs, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    init_reference_tt = ttnn.from_torch(init_reference, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    inter_references_tt = [
        ttnn.from_torch(ref, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for ref in inter_references
    ]

    tt_outputs = tt_model(
        hs=hs_tt,
        init_reference=init_reference_tt,
        inter_references=inter_references_tt,
        bev_embed=None,
    )

    tt_cls_scores = ttnn.to_torch(tt_outputs["all_cls_scores"]).float()
    tt_bbox_preds = ttnn.to_torch(tt_outputs["all_bbox_preds"]).float()
    tt_pts_preds = ttnn.to_torch(tt_outputs["all_pts_preds"]).float()

    assert (
        outputs_classes_torch.shape == tt_cls_scores.shape
    ), f"cls_scores shapes don't match: {outputs_classes_torch.shape} vs {tt_cls_scores.shape}"
    assert (
        outputs_coords_torch.shape == tt_bbox_preds.shape
    ), f"bbox_preds shapes don't match: {outputs_coords_torch.shape} vs {tt_bbox_preds.shape}"
    assert (
        outputs_pts_coords_torch.shape == tt_pts_preds.shape
    ), f"pts_preds shapes don't match: {outputs_pts_coords_torch.shape} vs {tt_pts_preds.shape}"

    pcc_threshold = 0.97
    pcc_cls_passed, pcc_cls = comp_pcc(outputs_classes_torch, tt_cls_scores, pcc_threshold)
    pcc_bbox_passed, pcc_bbox = comp_pcc(outputs_coords_torch, tt_bbox_preds, pcc_threshold)
    pcc_pts_passed, pcc_pts = comp_pcc(outputs_pts_coords_torch, tt_pts_preds, pcc_threshold)

    assert pcc_cls_passed, f"Classification scores PCC {pcc_cls:.6f} below threshold {pcc_threshold}"
    assert pcc_bbox_passed, f"Bbox predictions PCC {pcc_bbox:.6f} below threshold {pcc_threshold}"
    assert pcc_pts_passed, f"Points predictions PCC {pcc_pts:.6f} below threshold {pcc_threshold}"
