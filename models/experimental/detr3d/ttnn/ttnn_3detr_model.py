# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.detr3d.reference.model_utils import (
    FurthestPointSampling,
    BoxProcessor,
)
import math
import numpy as np
from models.common.lightweightmodule import LightweightModule

import torch

import ttnn
from models.experimental.detr3d.ttnn.ttnn_pointnet_samodule_votes import TtnnPointnetSAModuleVotes
from models.experimental.detr3d.ttnn.transformer import (
    TTTransformerEncoderLayer,
    TTTransformerDecoder,
)
from models.experimental.detr3d.ttnn.encoder import TtMaskedTransformerEncoder, EncoderArgs
from models.experimental.detr3d.ttnn.ttnn_generic_mlp import TttnnGenericMLP
from models.experimental.detr3d.reference.detr3d_model import PositionEmbeddingCoordsSine


def build_ttnn_preencoder(args):
    mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    # print("pre encoder is PointnetSAModuleVotes")
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
        encoder_layer = TTTransformerEncoderLayer
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
        encoder = TtMaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            interim_downsampling=interim_downsampling,
            masking_radius=masking_radius,
            device=args.device,
            encoder_args=EncoderArgs(
                d_model=args.enc_dim,
                nhead=args.enc_nhead,
                dim_feedforward=args.enc_ffn_dim,
                dropout=args.enc_dropout,
                activation=args.enc_activation,
            ),
            parameters=args.parameters.encoder,
        )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_ttnn_decoder(args):
    decoder = TTTransformerDecoder(
        device=args.device,
        decoder_layer_config={
            "d_model": args.dec_dim,
            "nhead": args.dec_nhead,
            "dim_feedforward": args.dec_ffn_dim,
            "dropout": args.dec_dropout,
            "normalize_before": True,  # Match the reference implementation
        },
        num_layers=args.dec_nlayers,
        return_intermediate=True,
        parameters=args.parameters.decoder,
    )
    return decoder


def build_ttnn_3detr(args, dataset_config):
    pre_encoder = build_ttnn_preencoder(args)
    encoder = build_ttnn_encoder(args)
    decoder = build_ttnn_decoder(args)
    model = TTModel3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
        torch_module=args.modules,
        parameters=args.parameters,
        device=args.device,
    )
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor


# 9: Detr3d model


class TTModel3DETR(LightweightModule):
    """
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
        mlp_dropout=0.3,
        num_queries=256,
        torch_module=None,
        parameters=None,
        device=None,
    ):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.encoder = encoder
        self.torch_module = torch_module
        self.parameters = parameters
        self.device = device
        if hasattr(self.encoder, "masking_radius"):
            hidden_dims = [encoder_dim]
        else:
            hidden_dims = [encoder_dim, encoder_dim]
        self.encoder_to_decoder_projection = TttnnGenericMLP(
            torch_module.encoder_to_decoder_projection,
            parameters.encoder_to_decoder_projection,
            device,
        )
        self.pos_embedding = PositionEmbeddingCoordsSine(d_pos=decoder_dim, pos_type=position_embedding, normalize=True)
        self.query_projection = TttnnGenericMLP(
            torch_module.query_projection,
            parameters.query_projection,
            device,
        )
        self.decoder = decoder
        self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

        self.num_queries = num_queries
        self.box_processor = BoxProcessor(dataset_config)
        self.furthest_point_sample = FurthestPointSampling()

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        self.mlp_heads = {
            "sem_cls_head": TttnnGenericMLP(
                self.torch_module.mlp_heads.sem_cls_head,
                self.parameters.mlp_heads.sem_cls_head,
                self.device,
            ),
            "center_head": TttnnGenericMLP(
                self.torch_module.mlp_heads.center_head,
                self.parameters.mlp_heads.center_head,
                self.device,
            ),
            "size_head": TttnnGenericMLP(
                self.torch_module.mlp_heads.size_head,
                self.parameters.mlp_heads.size_head,
                self.device,
            ),
            "angle_cls_head": TttnnGenericMLP(
                self.torch_module.mlp_heads.angle_cls_head,
                self.parameters.mlp_heads.angle_cls_head,
                self.device,
            ),
            "angle_residual_head": TttnnGenericMLP(
                self.torch_module.mlp_heads.angle_residual_head,
                self.parameters.mlp_heads.angle_residual_head,
                self.device,
            ),
        }

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_inds = self.furthest_point_sample(encoder_xyz, self.num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        pos_embed = ttnn.from_torch(pos_embed, device=self.device, dtype=ttnn.bfloat16)
        pos_embed = ttnn.permute(pos_embed, (0, 2, 1))
        pos_embed = ttnn.unsqueeze(pos_embed, 0)
        query_embed = self.query_projection(pos_embed)
        query_embed = ttnn.squeeze(query_embed, 0)
        query_embed = ttnn.permute(query_embed, (0, 2, 1))
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        xyz, features = self._break_up_pc(point_clouds)
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = ttnn.permute(pre_enc_features, (2, 0, 1))

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(pre_enc_features, xyz=pre_enc_xyz)
        # TODO: Cleanup enc_ids dont seem to be used
        # if enc_inds is None:
        #     # encoder does not perform any downsampling
        #     enc_inds = pre_enc_inds
        # else:
        #     # use gather here to ensure that it works for both FPS and random sampling
        #     enc_inds = ttnn.gather(pre_enc_inds, 1, enc_inds)
        #     # enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.type(torch.int64))
        return enc_xyz, enc_features, enc_inds

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        box_features = ttnn.permute(box_features, (0, 2, 3, 1))
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = ttnn.reshape(box_features, (num_layers * batch, channel, num_queries))

        box_features = ttnn.permute(box_features, (0, 2, 1))
        box_features = ttnn.unsqueeze(box_features, 0)
        box_features = ttnn.to_memory_config(box_features, ttnn.DRAM_MEMORY_CONFIG)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features)
        cls_logits = ttnn.squeeze(cls_logits, 0)
        cls_logits = ttnn.permute(cls_logits, (0, 2, 1))
        cls_logits = ttnn.transpose(cls_logits, 1, 2)
        cls_logits = ttnn.to_torch(cls_logits)

        center_offset = self.mlp_heads["center_head"](box_features)
        center_offset = ttnn.squeeze(center_offset, 0)
        center_offset = ttnn.permute(center_offset, (0, 2, 1))
        center_offset = ttnn.sigmoid(center_offset)
        center_offset = ttnn.transpose(center_offset, 1, 2)
        center_offset = center_offset - 0.5
        center_offset = ttnn.to_torch(center_offset)

        size_normalized = self.mlp_heads["size_head"](box_features)
        size_normalized = ttnn.squeeze(size_normalized, 0)
        size_normalized = ttnn.permute(size_normalized, (0, 2, 1))
        size_normalized = ttnn.sigmoid(size_normalized)
        size_normalized = ttnn.transpose(size_normalized, 1, 2)
        size_normalized = ttnn.to_torch(size_normalized)

        angle_logits = self.mlp_heads["angle_cls_head"](box_features)
        angle_logits = ttnn.squeeze(angle_logits, 0)
        angle_logits = ttnn.permute(angle_logits, (0, 2, 1))
        angle_logits = ttnn.transpose(angle_logits, 1, 2)
        angle_logits = ttnn.to_torch(angle_logits)

        angle_residual_normalized = self.mlp_heads["angle_residual_head"](box_features)
        angle_residual_normalized = ttnn.squeeze(angle_residual_normalized, 0)
        angle_residual_normalized = ttnn.permute(angle_residual_normalized, (0, 2, 1))
        angle_residual_normalized = ttnn.transpose(angle_residual_normalized, 1, 2)
        angle_residual_normalized = ttnn.to_torch(angle_residual_normalized)

        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
        size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
        angle_residual_normalized = angle_residual_normalized.reshape(num_layers, batch, num_queries, -1)
        angle_residual = angle_residual_normalized * (np.pi / angle_residual_normalized.shape[-1])

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            (
                center_normalized,
                center_unnormalized,
            ) = self.box_processor.compute_predicted_center(center_offset[l], query_xyz, point_cloud_dims)
            angle_continuous = self.box_processor.compute_predicted_angle(angle_logits[l], angle_residual[l])
            size_unnormalized = self.box_processor.compute_predicted_size(size_normalized[l], point_cloud_dims)
            box_corners = self.box_processor.box_parametrization_to_corners(
                center_unnormalized, size_unnormalized, angle_continuous
            )

            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])

            box_prediction = {
                "sem_cls_logits": cls_logits[l],
                "center_normalized": center_normalized.contiguous(),
                "center_unnormalized": center_unnormalized,
                "size_normalized": size_normalized[l],
                "size_unnormalized": size_unnormalized,
                "angle_logits": angle_logits[l],
                "angle_residual": angle_residual[l],
                "angle_residual_normalized": angle_residual_normalized[l],
                "angle_continuous": angle_continuous,
                "objectness_prob": objectness_prob,
                "sem_cls_prob": semcls_prob,
                "box_corners": box_corners,
            }
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, encoder_only=False):
        point_clouds = inputs["point_clouds"]

        enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
        enc_features = ttnn.permute(enc_features, (1, 0, 2))
        enc_features = ttnn.unsqueeze(enc_features, 0)
        enc_features = self.encoder_to_decoder_projection(enc_features)
        enc_features = ttnn.squeeze(enc_features, 0)
        enc_features = ttnn.permute(enc_features, (1, 0, 2))
        # encoder features: npoints x batch x channel
        # encoder xyz: npoints x batch x 3

        if encoder_only:
            # return: batch x npoints x channels
            return enc_xyz, ttnn.transpose(enc_features, 0, 1)

        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
        # query_embed: batch x channel x npoint
        enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)

        # decoder expects: npoints x batch x channel
        enc_pos = ttnn.from_torch(enc_pos, device=self.device, dtype=ttnn.bfloat16)
        enc_pos = ttnn.permute(enc_pos, (2, 0, 1))
        enc_pos = ttnn.permute(enc_pos, (1, 0, 2))
        query_embed = ttnn.permute(query_embed, (2, 0, 1))
        query_embed = ttnn.permute(query_embed, (1, 0, 2))
        enc_features = ttnn.permute(enc_features, (1, 0, 2))
        tgt = ttnn.zeros_like(query_embed, dtype=ttnn.bfloat16)
        box_features = self.decoder(tgt, enc_features, query_pos=query_embed, pos=enc_pos)[0]
        box_features = ttnn.permute(box_features, (0, 2, 1, 3))
        box_predictions = self.get_box_predictions(query_xyz, point_cloud_dims, box_features)
        return box_predictions
