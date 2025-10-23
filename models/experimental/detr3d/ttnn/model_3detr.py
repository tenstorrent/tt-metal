# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch
import numpy as np

from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.reference.model_utils import BoxProcessor
from models.experimental.detr3d.ttnn.masked_transformer_encoder import (
    TtnnTransformerEncoderLayer,
    TtnnMaskedTransformerEncoder,
    EncoderLayerArgs,
)
from models.experimental.detr3d.ttnn.transformer_decoder import (
    TtnnTransformerDecoder,
    TtnnTransformerDecoderLayer,
    DecoderLayerArgs,
)
from models.experimental.detr3d.ttnn.generic_mlp import TtnnGenericMLP
from models.experimental.detr3d.ttnn.pointnet_samodule_votes import TtnnPointnetSAModuleVotes
from models.experimental.detr3d.reference.torch_pointnet2_ops import FurthestPointSampling


class TtnnModel3DETR(LightweightModule):
    """
    NOTE: The Encoder and Decoder layers use batch first as compared to batch second in reference model,
          this helps remove lot of unnecessary permute operations within the network

    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        num_queries=256,
        torch_module=None,
        parameters=None,
        device=None,
    ):
        # NOTE: Layers and tensors with "torch_" in their names are on host
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.torch_module = torch_module
        self.parameters = parameters
        self.device = device
        self.encoder_to_decoder_projection = TtnnGenericMLP(
            torch_module.encoder_to_decoder_projection,
            parameters.encoder_to_decoder_projection,
            device,
        )
        # TODO: Check the feasibility
        # self.pos_embedding = TtnnPositionEmbeddingCoordsSine(
        #     d_pos=decoder_dim, pos_type=position_embedding, normalize=True, device=self.device, gauss_B=torch_module.pos_embedding.gauss_B
        # )
        self.torch_pos_embedding = torch_module.pos_embedding
        self.query_projection = TtnnGenericMLP(
            torch_module.query_projection,
            parameters.query_projection,
            device,
        )
        self.decoder = decoder
        self.build_mlp_heads()

        self.num_queries = num_queries
        self.torch_box_processor = BoxProcessor(dataset_config)
        # self.box_processor = TtnnBoxProcessor(dataset_config, device=self.device)
        self.torch_furthest_point_sample = FurthestPointSampling()

    def build_mlp_heads(self):
        self.mlp_heads = {
            "sem_cls_head": TtnnGenericMLP(
                self.torch_module.mlp_heads.sem_cls_head,
                self.parameters.mlp_heads.sem_cls_head,
                self.device,
            ),
            "center_head": TtnnGenericMLP(
                self.torch_module.mlp_heads.center_head,
                self.parameters.mlp_heads.center_head,
                self.device,
            ),
            "size_head": TtnnGenericMLP(
                self.torch_module.mlp_heads.size_head,
                self.parameters.mlp_heads.size_head,
                self.device,
            ),
            "angle_cls_head": TtnnGenericMLP(
                self.torch_module.mlp_heads.angle_cls_head,
                self.parameters.mlp_heads.angle_cls_head,
                self.device,
            ),
            "angle_residual_head": TtnnGenericMLP(
                self.torch_module.mlp_heads.angle_residual_head,
                self.parameters.mlp_heads.angle_residual_head,
                self.device,
            ),
        }

    def get_query_embeddings(self, torch_encoder_xyz, torch_point_cloud_dims):
        torch_query_inds = self.torch_furthest_point_sample(torch_encoder_xyz, self.num_queries)
        torch_query_inds = torch_query_inds.long()
        torch_query_xyz = [torch.gather(torch_encoder_xyz[..., x], 1, torch_query_inds) for x in range(3)]
        torch_query_xyz = torch.stack(torch_query_xyz)
        torch_query_xyz = torch_query_xyz.permute(1, 2, 0)

        # TODO: Either remove this completely or only use ttnn embedding
        # query_xyz = ttnn.from_torch(torch_query_xyz, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
        # pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        torch_pos_embed = self.torch_pos_embedding(torch_query_xyz, input_range=torch_point_cloud_dims)
        pos_embed = ttnn.from_torch(
            torch_pos_embed.permute(0, 2, 1), dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT
        )
        query_embed = self.query_projection(pos_embed)
        return torch_query_xyz, query_embed

    def _break_up_pc(self, torch_pc):
        # pc may contain color/normals.

        torch_xyz = torch_pc[..., 0:3].contiguous()
        torch_features = torch_pc[..., 3:].transpose(1, 2).contiguous() if torch_pc.size(-1) > 3 else None
        return torch_xyz, torch_features

    def run_encoder(self, torch_point_clouds):
        torch_xyz, torch_features = self._break_up_pc(torch_point_clouds)
        torch_pre_enc_xyz, pre_enc_features, _ = self.pre_encoder(torch_xyz, torch_features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # MultiHeadAttention in encoder expects batch x npoints x channel features
        pre_enc_features = ttnn.permute(pre_enc_features, (0, 2, 1))

        # xyz points are batch x npoint x channel order torch tensor
        torch_enc_xyz, enc_features, _ = self.encoder(pre_enc_features, xyz=torch_pre_enc_xyz)

        return torch_enc_xyz, enc_features, _

    def get_box_predictions(self, torch_query_xyz, torch_point_cloud_dims, box_features):
        """
        Parameters:
            torch_query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            torch_point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x num_queries x channel
        num_layers, batch, num_queries, channel = box_features.shape
        box_features = ttnn.reshape(box_features, (num_layers * batch, num_queries, channel))
        box_features = ttnn.to_memory_config(box_features, ttnn.DRAM_MEMORY_CONFIG)

        # mlp head outputs are (num_layers x batch) x nqueries x noutput
        cls_logits = self.mlp_heads["sem_cls_head"](box_features)
        center_offset = self.mlp_heads["center_head"](box_features)
        size_normalized = self.mlp_heads["size_head"](box_features)
        angle_logits = self.mlp_heads["angle_cls_head"](box_features)
        angle_residual_normalized = self.mlp_heads["angle_residual_head"](box_features)

        center_offset = ttnn.sigmoid(center_offset) - 0.5
        size_normalized = ttnn.sigmoid(size_normalized)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = ttnn.reshape(cls_logits, (num_layers, batch, num_queries, cls_logits.shape[-1]))
        center_offset = ttnn.reshape(center_offset, (num_layers, batch, num_queries, center_offset.shape[-1]))
        size_normalized = ttnn.reshape(size_normalized, (num_layers, batch, num_queries, size_normalized.shape[-1]))
        angle_logits = ttnn.reshape(angle_logits, (num_layers, batch, num_queries, angle_logits.shape[-1]))
        angle_residual_normalized = ttnn.reshape(
            angle_residual_normalized, (num_layers, batch, num_queries, angle_residual_normalized.shape[-1])
        )
        angle_residual = angle_residual_normalized * (np.pi / angle_residual_normalized.shape[-1])

        # TODO: Deallocate ttnn tensors here or add function for it
        # send outputs to torch for box processing
        torch_cls_logits = ttnn.to_torch(cls_logits)
        torch_center_offset = ttnn.to_torch(center_offset)
        torch_size_normalized = ttnn.to_torch(size_normalized)
        torch_angle_logits = ttnn.to_torch(angle_logits)
        torch_angle_residual_normalized = ttnn.to_torch(angle_residual_normalized)
        torch_angle_residual = ttnn.to_torch(angle_residual)

        torch_outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                torch_center_normalized,
                torch_center_unnormalized,
            ) = self.torch_box_processor.compute_predicted_center(
                torch_center_offset[l], torch_query_xyz, torch_point_cloud_dims
            )
            torch_angle_continuous = self.torch_box_processor.compute_predicted_angle(
                torch_angle_logits[l], torch_angle_residual[l]
            )
            torch_size_unnormalized = self.torch_box_processor.compute_predicted_size(
                torch_size_normalized[l], torch_point_cloud_dims
            )
            torch_box_corners = self.torch_box_processor.box_parametrization_to_corners(
                torch_center_unnormalized, torch_size_unnormalized, torch_angle_continuous
            )

            # below are used for matching/mAP eval
            (
                torch_semcls_prob,
                torch_objectness_prob,
            ) = self.torch_box_processor.compute_objectness_and_cls_prob(torch_cls_logits[l])

            torch_box_prediction = {
                "sem_cls_logits": torch_cls_logits[l],
                "center_normalized": torch_center_normalized,
                "center_unnormalized": torch_center_unnormalized,
                "size_normalized": torch_size_normalized[l],
                "size_unnormalized": torch_size_unnormalized,
                "angle_logits": torch_angle_logits[l],
                "angle_residual": torch_angle_residual[l],
                "angle_residual_normalized": torch_angle_residual_normalized[l],
                "angle_continuous": torch_angle_continuous,
                "objectness_prob": torch_objectness_prob,
                "sem_cls_prob": torch_semcls_prob,
                "box_corners": torch_box_corners,
            }
            torch_outputs.append(torch_box_prediction)

        # intermediate decoder layer outputs are only used during training
        # we use them to check for any instability in PCC
        torch_aux_outputs = torch_outputs[:-1]
        torch_outputs = torch_outputs[-1]

        return {
            "outputs": torch_outputs,  # output from last layer of decoder
            "aux_outputs": torch_aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, encoder_only=False):
        torch_point_clouds = inputs["point_clouds"]

        torch_enc_xyz, enc_features, _ = self.run_encoder(torch_point_clouds)
        enc_features = self.encoder_to_decoder_projection(enc_features)
        # encoder features: batch x npoints x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return torch_enc_xyz, enc_features

        torch_point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        point_cloud_dims = [
            ttnn.from_torch(t, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
            for t in torch_point_cloud_dims
        ]

        torch_query_xyz, query_embed = self.get_query_embeddings(torch_enc_xyz, torch_point_cloud_dims)
        # query_embed: batch x npoint x channel
        # TODO: Either remove this completely or only use ttnn embedding
        # enc_xyz_ttnn = ttnn.from_torch(torch_enc_xyz, dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT)
        # enc_pos = self.pos_embedding(enc_xyz_ttnn, input_range=ttnn_point_cloud_dims)
        torch_enc_pos = self.torch_pos_embedding(torch_enc_xyz, input_range=torch_point_cloud_dims)
        enc_pos = ttnn.from_torch(
            torch_enc_pos.permute(0, 2, 1), dtype=ttnn.bfloat16, device=self.device, layout=ttnn.TILE_LAYOUT
        )

        # decoder expects: batch x npoints x channel
        tgt = ttnn.zeros_like(query_embed, dtype=ttnn.bfloat16)
        box_features = self.decoder(tgt, enc_features, query_pos=query_embed, pos=enc_pos)[0]

        torch_box_predictions = self.get_box_predictions(torch_query_xyz, torch_point_cloud_dims, box_features)
        return torch_box_predictions


def build_ttnn_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    preencoder = TtnnPointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
        module=args.modules.pre_encoder,
        parameters=args.parameters.pre_encoder.mlp_module,
        device=args.device,
    )
    return preencoder


def build_ttnn_encoder(args):
    if args.enc_type in ["masked"]:
        encoder_layer = TtnnTransformerEncoderLayer
        interim_downsampling = TtnnPointnetSAModuleVotes(
            radius=0.4,
            nsample=32,
            npoint=args.preenc_npoints // 2,
            mlp=[args.enc_dim, 256, 256, args.enc_dim],
            normalize_xyz=True,
            module=args.modules.encoder.interim_downsampling,
            parameters=args.parameters.encoder.interim_downsampling.mlp_module,
            device=args.device,
        )

        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = TtnnMaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=args.enc_nlayers,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
            device=args.device,
            encoder_args=EncoderLayerArgs(
                d_model=args.enc_dim,
                nhead=args.enc_nhead,
                dim_feedforward=args.enc_ffn_dim,
            ),
            parameters=args.parameters.encoder,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_ttnn_decoder(args):
    decoder = TtnnTransformerDecoder(
        decoder_layer=TtnnTransformerDecoderLayer,
        num_layers=args.dec_nlayers,
        device=args.device,
        return_intermediate=True,
        decoder_args=DecoderLayerArgs(
            d_model=args.dec_dim,
            nhead=args.dec_nhead,
            dim_feedforward=args.dec_ffn_dim,
            normalize_before=True,
        ),
        parameters=args.parameters.decoder,
    )
    return decoder


def build_ttnn_3detr(args, dataset_config):
    pre_encoder = build_ttnn_preencoder(args)
    encoder = build_ttnn_encoder(args)
    decoder = build_ttnn_decoder(args)
    model = TtnnModel3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        num_queries=args.nqueries,
        torch_module=args.modules,
        parameters=args.parameters,
        device=args.device,
    )
    torch_output_processor = BoxProcessor(dataset_config)
    # output_processor = TtnnBoxProcessor(dataset_config, device=args.device)
    return model, torch_output_processor
