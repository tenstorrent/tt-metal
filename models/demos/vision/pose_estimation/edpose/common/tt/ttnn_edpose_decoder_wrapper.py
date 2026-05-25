# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ED-Pose decoder wrapper with query expansion logic.

The ED-Pose decoder is complex — it has:
  - 6 deformable decoder layers (self-attn + cross-attn + FFN)
  - Query expansion at layer 2 boundary (900 box → 900*18 = 16200 box+kpt queries)
  - Prediction heads (class/bbox/pose MLPs) called within the loop for iterative refinement
  - Reference point update at each layer

Two implementations:
  - EDPoseDecoder: CPU-only reference (PyTorch)
  - TTEDPoseDecoder: Hybrid — decoder layers use ttnn (cross-attn, FFN, norms on device),
    query expansion and refinement heads remain on CPU
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

EDPOSE_ROOT = os.environ.get("EDPOSE_ROOT", os.path.expanduser("~/ttwork/ED-Pose"))


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, int(H_), int(W_))
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(
        N_, M_ * D_, Lq_
    )
    return output.transpose(1, 2).contiguous()


class CPUMSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=5, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attn_w = F.softmax(
            self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points), dim=-1
        ).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        if reference_points.shape[-1] == 2:
            normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1).float()
            locs = reference_points[:, :, None, :, None, :].float() + offsets / normalizer[None, None, None, :, None, :]
        else:
            locs = (
                reference_points[:, :, None, :, None, :2].float()
                + offsets / self.n_points * reference_points[:, :, None, :, None, 2:].float() * 0.5
            )
        out = ms_deform_attn_core_pytorch(value.float(), input_spatial_shapes, locs.float(), attn_w)
        return self.output_proj(out)


class CPUDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=2048, n_levels=5, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0)
        self.cross_attn = CPUMSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)

    def forward(self, tgt, tgt_query_pos, tgt_reference_points, memory,
                memory_spatial_shapes, memory_level_start_index,
                memory_key_padding_mask=None, self_attn_mask=None):
        # Official ED-Pose: norm2 after self-attn, norm1 after cross-attn, norm3 after FFN
        q = k = tgt + tgt_query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)
        tgt = self.norm2(tgt + tgt2)

        tgt2 = self.cross_attn(
            (tgt + tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1),
            memory.transpose(0, 1),
            memory_spatial_shapes,
            memory_level_start_index,
            memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = self.norm1(tgt + tgt2)

        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + tgt2)
        return tgt


class EDPoseDecoder:
    """
    Full ED-Pose decoder with query expansion and iterative refinement on CPU.
    Loads all necessary weights from ED-Pose checkpoint.
    """

    def __init__(self, state_dict, d_model=256, d_ffn=2048, n_levels=5, n_heads=8, n_points=4,
                 num_decoder_layers=6, num_queries=900, num_classes=2,
                 num_body_points=17, num_box_decoder_layers=2, num_group=100):
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers
        self.num_body_points = num_body_points
        self.num_box_decoder_layers = num_box_decoder_layers
        self.num_group = num_group

        self.layers = nn.ModuleList()
        for i in range(num_decoder_layers):
            layer = CPUDecoderLayer(d_model, d_ffn, n_levels, n_heads, n_points)
            layer_sd = {}
            prefix = f"transformer.decoder.layers.{i}."
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    layer_sd[k[len(prefix):]] = v
            layer.load_state_dict(layer_sd, strict=True)
            layer.eval()
            self.layers.append(layer)

        self.norm = nn.LayerNorm(d_model)
        self.norm.load_state_dict({
            "weight": state_dict["transformer.decoder.norm.weight"],
            "bias": state_dict["transformer.decoder.norm.bias"],
        })
        self.norm.eval()

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        rph_sd = {k.replace("transformer.decoder.ref_point_head.", ""): v
                   for k, v in state_dict.items() if k.startswith("transformer.decoder.ref_point_head.")}
        self.ref_point_head.load_state_dict(rph_sd, strict=True)
        self.ref_point_head.eval()

        self.bbox_embed = nn.ModuleList()
        for i in range(num_decoder_layers):
            mlp = MLP(d_model, d_model, 4, 3)
            be_sd = {k.replace(f"bbox_embed.{i}.", ""): v
                     for k, v in state_dict.items() if k.startswith(f"bbox_embed.{i}.")}
            mlp.load_state_dict(be_sd, strict=True)
            mlp.eval()
            self.bbox_embed.append(mlp)

        self.class_embed = nn.ModuleList()
        for i in range(num_decoder_layers):
            lin = nn.Linear(d_model, num_classes)
            ce_sd = {k.replace(f"class_embed.{i}.", ""): v
                     for k, v in state_dict.items() if k.startswith(f"class_embed.{i}.")}
            lin.load_state_dict(ce_sd, strict=True)
            lin.eval()
            self.class_embed.append(lin)

        n_pose_layers = num_decoder_layers - num_box_decoder_layers + 1
        self.pose_embed = nn.ModuleList()
        for i in range(n_pose_layers):
            mlp = MLP(d_model, d_model, 2, 3)
            pe_sd = {k.replace(f"pose_embed.{i}.", ""): v
                     for k, v in state_dict.items() if k.startswith(f"pose_embed.{i}.")}
            mlp.load_state_dict(pe_sd, strict=True)
            mlp.eval()
            self.pose_embed.append(mlp)

        n_pose_hw = num_decoder_layers - num_box_decoder_layers
        self.pose_hw_embed = nn.ModuleList()
        for i in range(n_pose_hw):
            mlp = MLP(d_model, d_model, 2, 3)
            phw_sd = {k.replace(f"pose_hw_embed.{i}.", ""): v
                      for k, v in state_dict.items() if k.startswith(f"pose_hw_embed.{i}.")}
            mlp.load_state_dict(phw_sd, strict=True)
            mlp.eval()
            self.pose_hw_embed.append(mlp)

        self.keypoint_embed = nn.Embedding(num_body_points, d_model)
        ke_sd = {k.replace("transformer.decoder.keypoint_embed.", ""): v
                  for k, v in state_dict.items() if k.startswith("transformer.decoder.keypoint_embed.")}
        self.keypoint_embed.load_state_dict(ke_sd, strict=True)

        self.hw = nn.Embedding(num_body_points, 2)
        hw_sd = {k.replace("transformer.decoder.hw.", ""): v
                  for k, v in state_dict.items() if k.startswith("transformer.decoder.hw.")}
        self.hw.load_state_dict(hw_sd, strict=True)

        self.kpt_index = [x for x in range(num_group * (num_body_points + 1))
                          if x % (num_body_points + 1) != 0]

    @torch.no_grad()
    def __call__(self, tgt, memory, refpoint_embed, spatial_shapes, level_start_index,
                 valid_ratios, memory_key_padding_mask=None, self_attn_mask=None, self_attn_mask2=None):
        """
        Args:
            tgt: (N, num_queries, d_model)
            memory: (N, sum(Hi*Wi), d_model) — encoder output
            refpoint_embed: (N, num_queries, 4) — unsigmoided initial reference points
            spatial_shapes: (num_levels, 2)
            level_start_index: (num_levels,)
            valid_ratios: (N, num_levels, 2)
            memory_key_padding_mask: (N, sum(Hi*Wi)) or None
            self_attn_mask: (N*nheads, Lq, Lq) or None  (for layers 0-1)
            self_attn_mask2: (N*nheads, Lq2, Lq2) or None  (for layers 2-5)

        Returns:
            hs: list of (N, nq, d_model) per layer
            references: list of (N, nq, 4) per layer + initial
        """
        output = tgt.transpose(0, 1)  # (Lq, N, C) sequence-first
        memory_seq = memory.transpose(0, 1)  # (Len_in, N, C)

        # refpoint_embed is (N, Lq, 4) batch-first; convert to sequence-first for decoder
        # Clamp after sigmoid to prevent saturation from bfloat16 encoder output
        reference_points = refpoint_embed.transpose(0, 1).sigmoid().clamp(1e-3, 1 - 1e-3)  # (Lq, N, 4)
        ref_points = [reference_points]

        intermediate = []
        tgt_mask = self_attn_mask

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            else:
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]

            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            query_pos = raw_query_pos

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_reference_points=reference_points_input,
                memory=memory_seq,
                memory_spatial_shapes=spatial_shapes,
                memory_level_start_index=level_start_index,
                memory_key_padding_mask=memory_key_padding_mask,
                self_attn_mask=tgt_mask,
            )
            intermediate.append(self.norm(output))

            if layer_id < self.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid().clamp(1e-3, 1 - 1e-3)

            if layer_id == self.num_box_decoder_layers - 1:
                class_unselected = self.class_embed[layer_id](output)
                topk_proposals = torch.topk(class_unselected.max(-1)[0], self.num_group, dim=0)[1]
                new_reference_points_for_box = torch.gather(
                    new_reference_points, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
                )
                new_output_for_box = torch.gather(
                    output, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
                )
                bs = new_output_for_box.shape[1]
                new_output_for_keypoint = new_output_for_box[:, None, :, :] + \
                    self.keypoint_embed.weight[None, :, None, :]

                delta_xy = self.pose_embed[-1](new_output_for_keypoint)[..., :2]
                keypoint_xy = (
                    inverse_sigmoid(new_reference_points_for_box[..., :2][:, None]) + delta_xy
                ).sigmoid().clamp(1e-3, 1 - 1e-3)

                num_queries_box, _, bs_, _ = keypoint_xy.shape
                keypoint_wh_weight = (
                    self.hw.weight.unsqueeze(0).unsqueeze(-2)
                    .repeat(num_queries_box, 1, bs_, 1).sigmoid()
                )
                keypoint_wh = keypoint_wh_weight * new_reference_points_for_box[..., 2:][:, None]
                new_reference_points_for_keypoint = torch.cat((keypoint_xy, keypoint_wh), dim=-1)

                new_reference_points = torch.cat(
                    (new_reference_points_for_box.unsqueeze(1), new_reference_points_for_keypoint), dim=1
                ).flatten(0, 1)
                output = torch.cat(
                    (new_output_for_box.unsqueeze(1), new_output_for_keypoint), dim=1
                ).flatten(0, 1)
                tgt_mask = self_attn_mask2

            if layer_id >= self.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                output_bbox_norm = output[0::(self.num_body_points + 1)]
                reference_before_sigmoid_bbox_norm = reference_before_sigmoid[0::(self.num_body_points + 1)]

                delta_unsig_norm = self.bbox_embed[layer_id](output_bbox_norm)
                outputs_unsig_norm = delta_unsig_norm + reference_before_sigmoid_bbox_norm
                new_reference_points_for_box_norm = outputs_unsig_norm.sigmoid().clamp(1e-3, 1 - 1e-3)

                output_kpt = output.index_select(
                    0, torch.tensor(self.kpt_index, device=output.device)
                )
                delta_xy_unsig = self.pose_embed[layer_id - self.num_box_decoder_layers](output_kpt)
                outputs_unsig = reference_before_sigmoid.index_select(
                    0, torch.tensor(self.kpt_index, device=output.device)
                ).clone()
                delta_hw_unsig = self.pose_hw_embed[layer_id - self.num_box_decoder_layers](output_kpt)
                outputs_unsig[..., :2] += delta_xy_unsig[..., :2]
                outputs_unsig[..., 2:] += delta_hw_unsig
                new_reference_points_for_keypoint = outputs_unsig.sigmoid().clamp(1e-3, 1 - 1e-3)

                bs_ = new_reference_points_for_box_norm.shape[1]
                new_reference_points_norm = torch.cat(
                    (
                        new_reference_points_for_box_norm.unsqueeze(1),
                        new_reference_points_for_keypoint.view(-1, self.num_body_points, bs_, 4),
                    ),
                    dim=1,
                ).flatten(0, 1)
                new_reference_points = new_reference_points_norm

            reference_points = new_reference_points.detach()
            ref_points.append(new_reference_points)

        return (
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_ref.transpose(0, 1) for itm_ref in ref_points],
        )


class TTEDPoseDecoder:
    """
    ED-Pose decoder with ttnn-accelerated layers.

    Each decoder layer runs cross-attention (MSDeformAttn), FFN, and layer norms
    on the TT device. Self-attention runs on host. Query expansion and iterative
    refinement heads remain on CPU.

    Same interface as EDPoseDecoder but takes a ttnn device and keeps encoder
    memory on device.
    """

    def __init__(self, device, state_dict, d_model=256, d_ffn=2048, n_levels=5, n_heads=8, n_points=4,
                 num_decoder_layers=6, num_queries=900, num_classes=2,
                 num_body_points=17, num_box_decoder_layers=2, num_group=100):
        from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_decoder import (
            TTDeformableDecoderLayer,
        )

        self.device = device
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers
        self.num_body_points = num_body_points
        self.num_box_decoder_layers = num_box_decoder_layers
        self.num_group = num_group

        self.layers = []
        for i in range(num_decoder_layers):
            prefix = f"transformer.decoder.layers.{i}"
            layer = TTDeformableDecoderLayer(
                device, state_dict, prefix,
                d_model, d_ffn, n_levels, n_heads, n_points,
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(d_model)
        self.norm.load_state_dict({
            "weight": state_dict["transformer.decoder.norm.weight"],
            "bias": state_dict["transformer.decoder.norm.bias"],
        })
        self.norm.eval()

        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        rph_sd = {k.replace("transformer.decoder.ref_point_head.", ""): v
                   for k, v in state_dict.items() if k.startswith("transformer.decoder.ref_point_head.")}
        self.ref_point_head.load_state_dict(rph_sd, strict=True)
        self.ref_point_head.eval()

        self.bbox_embed = nn.ModuleList()
        for i in range(num_decoder_layers):
            mlp = MLP(d_model, d_model, 4, 3)
            be_sd = {k.replace(f"bbox_embed.{i}.", ""): v
                     for k, v in state_dict.items() if k.startswith(f"bbox_embed.{i}.")}
            mlp.load_state_dict(be_sd, strict=True)
            mlp.eval()
            self.bbox_embed.append(mlp)

        self.class_embed = nn.ModuleList()
        for i in range(num_decoder_layers):
            lin = nn.Linear(d_model, num_classes)
            ce_sd = {k.replace(f"class_embed.{i}.", ""): v
                     for k, v in state_dict.items() if k.startswith(f"class_embed.{i}.")}
            lin.load_state_dict(ce_sd, strict=True)
            lin.eval()
            self.class_embed.append(lin)

        n_pose_layers = num_decoder_layers - num_box_decoder_layers + 1
        self.pose_embed = nn.ModuleList()
        for i in range(n_pose_layers):
            mlp = MLP(d_model, d_model, 2, 3)
            pe_sd = {k.replace(f"pose_embed.{i}.", ""): v
                     for k, v in state_dict.items() if k.startswith(f"pose_embed.{i}.")}
            mlp.load_state_dict(pe_sd, strict=True)
            mlp.eval()
            self.pose_embed.append(mlp)

        n_pose_hw = num_decoder_layers - num_box_decoder_layers
        self.pose_hw_embed = nn.ModuleList()
        for i in range(n_pose_hw):
            mlp = MLP(d_model, d_model, 2, 3)
            phw_sd = {k.replace(f"pose_hw_embed.{i}.", ""): v
                      for k, v in state_dict.items() if k.startswith(f"pose_hw_embed.{i}.")}
            mlp.load_state_dict(phw_sd, strict=True)
            mlp.eval()
            self.pose_hw_embed.append(mlp)

        self.keypoint_embed = nn.Embedding(num_body_points, d_model)
        ke_sd = {k.replace("transformer.decoder.keypoint_embed.", ""): v
                  for k, v in state_dict.items() if k.startswith("transformer.decoder.keypoint_embed.")}
        self.keypoint_embed.load_state_dict(ke_sd, strict=True)

        self.hw = nn.Embedding(num_body_points, 2)
        hw_sd = {k.replace("transformer.decoder.hw.", ""): v
                  for k, v in state_dict.items() if k.startswith("transformer.decoder.hw.")}
        self.hw.load_state_dict(hw_sd, strict=True)

        self.kpt_index = [x for x in range(num_group * (num_body_points + 1))
                          if x % (num_body_points + 1) != 0]

    @torch.no_grad()
    def __call__(self, tgt, memory_tt, refpoint_embed, spatial_shapes, level_start_index,
                 valid_ratios, memory_key_padding_mask=None, self_attn_mask=None, self_attn_mask2=None):
        """
        Args:
            tgt: torch (N, num_queries, d_model)
            memory_tt: ttnn (N, sum(Hi*Wi), d_model) TILE_LAYOUT on device
            refpoint_embed: torch (N, num_queries, 4) — unsigmoided initial reference points
            spatial_shapes: torch (num_levels, 2)
            level_start_index: torch (num_levels,)
            valid_ratios: torch (N, num_levels, 2)
            memory_key_padding_mask: torch (N, sum(Hi*Wi)) or None
            self_attn_mask: torch (N*nheads, Lq, Lq) or None  (for layers 0-1)
            self_attn_mask2: torch (N*nheads, Lq2, Lq2) or None  (for layers 2-5)

        Returns:
            hs: list of torch (N, nq, d_model) per layer
            references: list of torch (N, nq, 4) per layer + initial
        """
        output = tgt.transpose(0, 1)  # (Lq, N, C) sequence-first

        reference_points = refpoint_embed.transpose(0, 1).sigmoid().clamp(1e-3, 1 - 1e-3)
        ref_points = [reference_points]

        intermediate = []

        def _to_ttnn_additive_mask(mask):
            if mask is None:
                return None
            if isinstance(mask, ttnn.Tensor):
                return mask
            float_mask = torch.zeros_like(mask, dtype=torch.bfloat16)
            float_mask.masked_fill_(mask, float("-inf"))
            return ttnn.from_torch(
                float_mask, layout=ttnn.TILE_LAYOUT, device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        tgt_mask = _to_ttnn_additive_mask(self_attn_mask)
        tgt_mask2 = _to_ttnn_additive_mask(self_attn_mask2)

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            else:
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]

            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
            raw_query_pos = self.ref_point_head(query_sine_embed)
            query_pos = raw_query_pos

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_reference_points=reference_points_input,
                memory=memory_tt,
                memory_spatial_shapes=spatial_shapes,
                memory_level_start_index=level_start_index,
                memory_key_padding_mask=memory_key_padding_mask,
                self_attn_mask=tgt_mask,
            )
            intermediate.append(self.norm(output))

            if layer_id < self.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid().clamp(1e-3, 1 - 1e-3)

            if layer_id == self.num_box_decoder_layers - 1:
                class_unselected = self.class_embed[layer_id](output)
                topk_proposals = torch.topk(class_unselected.max(-1)[0], self.num_group, dim=0)[1]
                new_reference_points_for_box = torch.gather(
                    new_reference_points, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
                )
                new_output_for_box = torch.gather(
                    output, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
                )
                bs = new_output_for_box.shape[1]
                new_output_for_keypoint = new_output_for_box[:, None, :, :] + \
                    self.keypoint_embed.weight[None, :, None, :]

                delta_xy = self.pose_embed[-1](new_output_for_keypoint)[..., :2]
                keypoint_xy = (
                    inverse_sigmoid(new_reference_points_for_box[..., :2][:, None]) + delta_xy
                ).sigmoid().clamp(1e-3, 1 - 1e-3)

                num_queries_box, _, bs_, _ = keypoint_xy.shape
                keypoint_wh_weight = (
                    self.hw.weight.unsqueeze(0).unsqueeze(-2)
                    .repeat(num_queries_box, 1, bs_, 1).sigmoid()
                )
                keypoint_wh = keypoint_wh_weight * new_reference_points_for_box[..., 2:][:, None]
                new_reference_points_for_keypoint = torch.cat((keypoint_xy, keypoint_wh), dim=-1)

                new_reference_points = torch.cat(
                    (new_reference_points_for_box.unsqueeze(1), new_reference_points_for_keypoint), dim=1
                ).flatten(0, 1)
                output = torch.cat(
                    (new_output_for_box.unsqueeze(1), new_output_for_keypoint), dim=1
                ).flatten(0, 1)
                tgt_mask = tgt_mask2

            if layer_id >= self.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                output_bbox_norm = output[0::(self.num_body_points + 1)]
                reference_before_sigmoid_bbox_norm = reference_before_sigmoid[0::(self.num_body_points + 1)]

                delta_unsig_norm = self.bbox_embed[layer_id](output_bbox_norm)
                outputs_unsig_norm = delta_unsig_norm + reference_before_sigmoid_bbox_norm
                new_reference_points_for_box_norm = outputs_unsig_norm.sigmoid().clamp(1e-3, 1 - 1e-3)

                output_kpt = output.index_select(
                    0, torch.tensor(self.kpt_index, device=output.device)
                )
                delta_xy_unsig = self.pose_embed[layer_id - self.num_box_decoder_layers](output_kpt)
                outputs_unsig = reference_before_sigmoid.index_select(
                    0, torch.tensor(self.kpt_index, device=output.device)
                ).clone()
                delta_hw_unsig = self.pose_hw_embed[layer_id - self.num_box_decoder_layers](output_kpt)
                outputs_unsig[..., :2] += delta_xy_unsig[..., :2]
                outputs_unsig[..., 2:] += delta_hw_unsig
                new_reference_points_for_keypoint = outputs_unsig.sigmoid().clamp(1e-3, 1 - 1e-3)

                bs_ = new_reference_points_for_box_norm.shape[1]
                new_reference_points_norm = torch.cat(
                    (
                        new_reference_points_for_box_norm.unsqueeze(1),
                        new_reference_points_for_keypoint.view(-1, self.num_body_points, bs_, 4),
                    ),
                    dim=1,
                ).flatten(0, 1)
                new_reference_points = new_reference_points_norm

            reference_points = new_reference_points.detach()
            ref_points.append(new_reference_points)

        if tgt_mask2 is not None and isinstance(tgt_mask2, ttnn.Tensor):
            ttnn.deallocate(tgt_mask2)

        return (
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_ref.transpose(0, 1) for itm_ref in ref_points],
        )
