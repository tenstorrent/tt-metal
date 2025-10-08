# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import copy
import torch.nn as nn
from models.experimental.uniad.reference.motion_transformer_decoder import MotionTransformerDecoder
import pickle

from models.experimental.uniad.reference.utils import (
    anchor_coordinate_transform,
    norm_points,
    pos2posemb2d,
    bivariate_gaussian_activation,
)


class MotionHead(nn.Module):
    def __init__(
        self,
        *args,
        predict_steps=12,
        transformerlayers=None,
        bbox_coder=None,
        num_cls_fcs=2,
        bev_h=30,
        bev_w=30,
        embed_dims=256,
        num_anchor=6,
        det_layer_num=6,
        group_id_list=[],
        pc_range=None,
        use_nonlinear_optimizer=False,
        anchor_info_path=None,
        loss_traj=dict(),
        num_classes=0,
        vehicle_id_list=[0, 1, 2, 3, 4, 6, 7],
        **kwargs,
    ):
        super(MotionHead, self).__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_reg_fcs = num_cls_fcs - 1
        self.embed_dims = embed_dims
        self.num_anchor = num_anchor
        self.num_anchor_group = len(group_id_list)

        # we merge the classes into groups for anchor assignment
        self.cls2group = [0 for i in range(num_classes)]
        for i, grouped_ids in enumerate(group_id_list):
            for gid in grouped_ids:
                self.cls2group[gid] = i
        self.cls2group = torch.tensor(self.cls2group)
        self.pc_range = pc_range
        self.predict_steps = predict_steps
        self.vehicle_id_list = vehicle_id_list

        self.use_nonlinear_optimizer = use_nonlinear_optimizer
        self._load_anchors(anchor_info_path)
        self._build_loss(loss_traj)
        self._build_layers(transformerlayers, det_layer_num)
        self._init_layers()

    def _build_loss(self, loss_traj):
        self.unflatten_traj = nn.Unflatten(3, (self.predict_steps, 5))
        self.log_softmax = nn.LogSoftmax(dim=2)

    def _load_anchors(self, anchor_info_path):
        anchor_infos = pickle.load(open(anchor_info_path, "rb"))
        self.kmeans_anchors = torch.stack(
            [torch.from_numpy(a) for a in anchor_infos["anchors_all"]]
        )  # Nc, Pc, steps, 2

    def _build_layers(self, transformerlayers, det_layer_num):
        self.learnable_motion_query_embedding = nn.Embedding(self.num_anchor * self.num_anchor_group, self.embed_dims)
        self.motionformer = MotionTransformerDecoder(
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            embed_dims=256,
            num_layers=3,
            transformerlayers={
                "type": "MotionTransformerAttentionLayer",
                "batch_first": True,
                "attn_cfgs": [
                    {
                        "type": "MotionDeformableAttention",
                        "num_steps": 12,
                        "embed_dims": 256,
                        "num_levels": 1,
                        "num_heads": 8,
                        "num_points": 4,
                        "sample_index": -1,
                    }
                ],
                "feedforward_channels": 512,
                "operation_order": ("cross_attn", "norm", "ffn", "norm"),
            },
        )
        self.layer_track_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims * det_layer_num, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.agent_level_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.scene_level_ego_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.scene_level_offset_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )
        self.boxes_query_embedding_layer = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 2),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 2, self.embed_dims),
        )

    def _init_layers(self):
        traj_cls_branch = []
        traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
        traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
        traj_cls_branch.append(nn.ReLU(inplace=True))
        for _ in range(self.num_reg_fcs - 1):
            traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(nn.Linear(self.embed_dims, 1))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        traj_reg_branch = []
        traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
        traj_reg_branch.append(nn.ReLU())
        for _ in range(self.num_reg_fcs - 1):
            traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_reg_branch.append(nn.ReLU())
        traj_reg_branch.append(nn.Linear(self.embed_dims, self.predict_steps * 5))
        traj_reg_branch = nn.Sequential(*traj_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_pred = self.motionformer.num_layers
        self.traj_cls_branches = _get_clones(traj_cls_branch, num_pred)
        self.traj_reg_branches = _get_clones(traj_reg_branch, num_pred)

    def _extract_tracking_centers(self, bbox_results, bev_range):
        batch_size = len(bbox_results)
        det_bbox_posembed = []
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            xy = bboxes.gravity_center[:, :2]
            x_norm = (xy[:, 0] - bev_range[0]) / (bev_range[3] - bev_range[0])
            y_norm = (xy[:, 1] - bev_range[1]) / (bev_range[4] - bev_range[1])
            det_bbox_posembed.append(torch.cat([x_norm[:, None], y_norm[:, None]], dim=-1))
        return torch.stack(det_bbox_posembed)

    def forward_test(self, bev_embed, outs_track={}, outs_seg={}):
        track_query = outs_track["track_query_embeddings"][None, None, ...]
        track_boxes = outs_track["track_bbox_results"]

        track_query = torch.cat([track_query, outs_track["sdc_embedding"][None, None, None, :]], dim=2)
        sdc_track_boxes = outs_track["sdc_track_bbox_results"]

        track_boxes[0][0].tensor = torch.cat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        track_boxes[0][1] = torch.cat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        track_boxes[0][2] = torch.cat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        track_boxes[0][3] = torch.cat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)
        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg["args_tuple"]
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        traj_results = self.get_trajs(outs_motion, track_boxes)
        bboxes, scores, labels, bbox_index, mask = track_boxes[0]
        outs_motion["track_scores"] = scores[None, :]
        labels[-1] = 0

        def filter_vehicle_query(outs_motion, labels, vehicle_id_list):
            if len(labels) < 1:  # No other obj query except sdc query.
                return None

            # select vehicle query according to vehicle_id_list
            vehicle_mask = torch.zeros_like(labels)
            for veh_id in vehicle_id_list:
                vehicle_mask |= labels == veh_id
            outs_motion["traj_query"] = outs_motion["traj_query"][:, :, vehicle_mask > 0]
            outs_motion["track_query"] = outs_motion["track_query"][:, vehicle_mask > 0]
            outs_motion["track_query_pos"] = outs_motion["track_query_pos"][:, vehicle_mask > 0]
            outs_motion["track_scores"] = outs_motion["track_scores"][:, vehicle_mask > 0]
            return outs_motion

        outs_motion = filter_vehicle_query(outs_motion, labels, self.vehicle_id_list)

        # filter sdc query
        outs_motion["sdc_traj_query"] = outs_motion["traj_query"][:, :, -1]
        outs_motion["sdc_track_query"] = outs_motion["track_query"][:, -1]
        outs_motion["sdc_track_query_pos"] = outs_motion["track_query_pos"][:, -1]
        outs_motion["traj_query"] = outs_motion["traj_query"][:, :, :-1]
        outs_motion["track_query"] = outs_motion["track_query"][:, :-1]
        outs_motion["track_query_pos"] = outs_motion["track_query_pos"][:, :-1]
        outs_motion["track_scores"] = outs_motion["track_scores"][:, :-1]

        return traj_results, outs_motion

    def forward(self, bev_embed, track_query, lane_query, lane_query_pos, track_bbox_results):
        dtype = track_query.dtype
        device = track_query.device
        num_groups = self.kmeans_anchors.shape[0]

        # extract the last frame of the track query
        track_query = track_query[:, -1]

        # encode the center point of the track query
        reference_points_track = self._extract_tracking_centers(track_bbox_results, self.pc_range)
        track_query_pos = self.boxes_query_embedding_layer(pos2posemb2d(reference_points_track.to(device)))  # B, A, D

        learnable_query_pos = self.learnable_motion_query_embedding.weight.to(dtype)  # latent anchor (P*G, D)
        learnable_query_pos = torch.stack(torch.split(learnable_query_pos, self.num_anchor, dim=0))

        agent_level_anchors = (
            self.kmeans_anchors.to(dtype).to(device).view(num_groups, self.num_anchor, self.predict_steps, 2).detach()
        )
        scene_level_ego_anchors = anchor_coordinate_transform(
            agent_level_anchors, track_bbox_results, with_translation_transform=True
        )  # B, A, G, P ,12 ,2
        scene_level_offset_anchors = anchor_coordinate_transform(
            agent_level_anchors, track_bbox_results, with_translation_transform=False
        )  # B, A, G, P ,12 ,2

        agent_level_norm = norm_points(agent_level_anchors, self.pc_range)
        scene_level_ego_norm = norm_points(scene_level_ego_anchors, self.pc_range)
        scene_level_offset_norm = norm_points(scene_level_offset_anchors, self.pc_range)

        # we only use the last point of the anchor
        agent_level_embedding = self.agent_level_embedding_layer(pos2posemb2d(agent_level_norm[..., -1, :]))  # G, P, D
        scene_level_ego_embedding = self.scene_level_ego_embedding_layer(
            pos2posemb2d(scene_level_ego_norm[..., -1, :])
        )  # B, A, G, P , D
        scene_level_offset_embedding = self.scene_level_offset_embedding_layer(
            pos2posemb2d(scene_level_offset_norm[..., -1, :])
        )  # B, A, G, P , D

        batch_size, num_agents = scene_level_ego_embedding.shape[:2]
        agent_level_embedding = agent_level_embedding[None, None, ...].expand(batch_size, num_agents, -1, -1, -1)
        learnable_embed = learnable_query_pos[None, None, ...].expand(batch_size, num_agents, -1, -1, -1)

        scene_level_offset_anchors = self.group_mode_query_pos(track_bbox_results, scene_level_offset_anchors)

        agent_level_embedding = self.group_mode_query_pos(track_bbox_results, agent_level_embedding)
        scene_level_ego_embedding = self.group_mode_query_pos(
            track_bbox_results, scene_level_ego_embedding
        )  # B, A, G, P , D-> B, A, P , D

        scene_level_offset_embedding = self.group_mode_query_pos(track_bbox_results, scene_level_offset_embedding)
        learnable_embed = self.group_mode_query_pos(track_bbox_results, learnable_embed)

        init_reference = scene_level_offset_anchors.detach()

        outputs_traj_scores = []
        outputs_trajs = []

        inter_states, inter_references = self.motionformer(
            track_query,  # B, A_track, D
            lane_query,  # B, M, D
            track_query_pos=track_query_pos,
            lane_query_pos=lane_query_pos,
            track_bbox_results=track_bbox_results,
            bev_embed=bev_embed,
            reference_trajs=init_reference,
            traj_reg_branches=self.traj_reg_branches,
            traj_cls_branches=self.traj_cls_branches,
            # anchor embeddings
            agent_level_embedding=agent_level_embedding,
            scene_level_ego_embedding=scene_level_ego_embedding,
            scene_level_offset_embedding=scene_level_offset_embedding,
            learnable_embed=learnable_embed,
            # anchor positional embeddings layers
            agent_level_embedding_layer=self.agent_level_embedding_layer,
            scene_level_ego_embedding_layer=self.scene_level_ego_embedding_layer,
            scene_level_offset_embedding_layer=self.scene_level_offset_embedding_layer,
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=device),
            level_start_index=torch.tensor([0], device=device),
        )

        for lvl in range(inter_states.shape[0]):
            outputs_class = self.traj_cls_branches[lvl](inter_states[lvl])
            tmp = self.traj_reg_branches[lvl](inter_states[lvl])
            tmp = self.unflatten_traj(tmp)

            # we use cumsum trick here to get the trajectory
            tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)

            outputs_class = self.log_softmax(outputs_class.squeeze(3))
            outputs_traj_scores.append(outputs_class)

            for bs in range(tmp.shape[0]):
                tmp[bs] = bivariate_gaussian_activation(tmp[bs])
            outputs_trajs.append(tmp)
        outputs_traj_scores = torch.stack(outputs_traj_scores)
        outputs_trajs = torch.stack(outputs_trajs)

        B, A_track, D = track_query.shape
        valid_traj_masks = track_query.new_ones((B, A_track)) > 0
        outs = {
            "all_traj_scores": outputs_traj_scores,
            "all_traj_preds": outputs_trajs,
            "valid_traj_masks": valid_traj_masks,
            "traj_query": inter_states,
            "track_query": track_query,
            "track_query_pos": track_query_pos,
        }

        return outs

    def group_mode_query_pos(self, bbox_results, mode_query_pos):
        batch_size = len(bbox_results)
        agent_num = mode_query_pos.shape[1]
        batched_mode_query_pos = []
        self.cls2group = self.cls2group.to(mode_query_pos.device)
        # TODO: vectorize this
        # group the embeddings based on the class
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            label = labels.to(mode_query_pos.device)
            grouped_label = self.cls2group[label]
            grouped_mode_query_pos = []
            for j in range(agent_num):
                grouped_mode_query_pos.append(mode_query_pos[i, j, grouped_label[j]])
            batched_mode_query_pos.append(torch.stack(grouped_mode_query_pos))
        return torch.stack(batched_mode_query_pos)

    def get_trajs(self, preds_dicts, bbox_results):
        num_samples = len(bbox_results)
        num_layers = preds_dicts["all_traj_preds"].shape[0]
        ret_list = []
        for i in range(num_samples):
            preds = dict()
            for j in range(num_layers):
                subfix = "_" + str(j) if j < (num_layers - 1) else ""
                traj = preds_dicts["all_traj_preds"][j, i]
                traj_scores = preds_dicts["all_traj_scores"][j, i]

                traj_scores, traj = traj_scores.cpu(), traj.cpu()
                preds["traj" + subfix] = traj
                preds["traj_scores" + subfix] = traj_scores
            ret_list.append(preds)
        return ret_list
