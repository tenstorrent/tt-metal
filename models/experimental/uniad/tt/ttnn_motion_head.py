# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pickle
import numpy as np

import ttnn

from models.experimental.uniad.tt.ttnn_utils import (
    norm_points,
    pos2posemb2d,
    anchor_coordinate_transform,
    bivariate_gaussian_activation_motion_head,
)
from models.experimental.uniad.tt.ttnn_motion_transformer_decoder import TtMotionTransformerDecoder


class TtMotionHead:
    def __init__(
        self,
        parameters,
        device,
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
        self.parameters = parameters
        self.device = device
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
        self.cls2group = ttnn.Tensor(np.array(self.cls2group), device=device)
        self.pc_range = pc_range
        self.predict_steps = predict_steps
        self.vehicle_id_list = vehicle_id_list

        self.use_nonlinear_optimizer = use_nonlinear_optimizer
        self._load_anchors(anchor_info_path)
        self._build_layers(transformerlayers, det_layer_num)
        self._init_layers()

    def _load_anchors(self, anchor_info_path):
        anchor_infos = pickle.load(open(anchor_info_path, "rb"))
        self.kmeans_anchors = ttnn.stack(
            [
                ttnn.to_device(
                    ttnn.to_dtype(ttnn.Tensor(a.astype(np.float32), layout=ttnn.TILE_LAYOUT), dtype=ttnn.bfloat16),
                    device=self.device,
                )
                for a in anchor_infos["anchors_all"]
            ],  # changing this reduced the outputs_traj_scores pcc from 0.95 to 0.94
            dim=0,
        )  # Nc, Pc, steps, 2

    def _build_layers(self, transformerlayers, det_layer_num):
        self.learnable_motion_query_embedding = ttnn.embedding
        self.motionformer = TtMotionTransformerDecoder(
            parameters=self.parameters.motionformer,
            device=self.device,
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
        self.layer_track_query_fuser = [
            ttnn.linear,
            ttnn.layer_norm,
            ttnn.relu,
        ]

        self.agent_level_embedding_layer = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
        ]
        self.scene_level_ego_embedding_layer = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
        ]
        self.scene_level_offset_embedding_layer = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
        ]
        self.boxes_query_embedding_layer = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
        ]

    def _init_layers(self):
        traj_cls_branch = []
        traj_cls_branch.append(ttnn.linear)
        traj_cls_branch.append(ttnn.layer_norm)
        traj_cls_branch.append(ttnn.relu)
        for _ in range(self.num_reg_fcs - 1):
            traj_cls_branch.append(ttnn.linear)
            traj_cls_branch.append(ttnn.layer_norm)
            traj_cls_branch.append(ttnn.relu)
        traj_cls_branch.append(ttnn.linear)
        traj_cls_branch = [*traj_cls_branch]

        traj_reg_branch = []
        traj_reg_branch.append(ttnn.linear)
        traj_reg_branch.append(ttnn.relu)
        for _ in range(self.num_reg_fcs - 1):
            traj_reg_branch.append(ttnn.linear)
            traj_reg_branch.append(ttnn.relu)
        traj_reg_branch.append(ttnn.linear)
        traj_reg_branch = [*traj_reg_branch]

        def _get_clones(module, N):
            return [module for i in range(N)]

        num_pred = self.motionformer.num_layers
        self.traj_cls_branches = _get_clones(traj_cls_branch, num_pred)
        self.traj_reg_branches = _get_clones(traj_reg_branch, num_pred)

    def _extract_tracking_centers(self, bbox_results, bev_range):
        batch_size = len(bbox_results)
        det_bbox_posembed = []
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            xy = bboxes.gravity_center[:, :2]
            x_norm = ttnn.div((ttnn.sub(xy[:, 0], bev_range[0])), (bev_range[3] - bev_range[0]))
            y_norm = ttnn.div((ttnn.sub(xy[:, 1], bev_range[1])), (bev_range[4] - bev_range[1]))
            det_bbox_posembed.append(ttnn.concat([ttnn.unsqueeze(x_norm, -1), ttnn.unsqueeze(y_norm, -1)], dim=-1))
        return ttnn.stack(det_bbox_posembed, dim=0)

    def forward_test(self, bev_embed, outs_track={}, outs_seg={}):
        track_query = ttnn.unsqueeze(ttnn.unsqueeze(outs_track["track_query_embeddings"], 0), 0)
        track_boxes = outs_track["track_bbox_results"]

        if track_query.shape[2] == 0:
            track_query = ttnn.reshape(outs_track["sdc_embedding"], (1, 1, 1, outs_track["sdc_embedding"].shape[-1]))
        else:
            track_query = ttnn.concat(
                [
                    track_query,
                    ttnn.reshape(outs_track["sdc_embedding"], (1, 1, 1, outs_track["sdc_embedding"].shape[-1])),
                ],
                dim=2,
            )
        sdc_track_boxes = outs_track["sdc_track_bbox_results"]

        if track_boxes[0][0].tensor.shape[0] == 0:
            track_boxes[0][0].tensor = ttnn.clone(sdc_track_boxes[0][0].tensor)
        else:
            track_boxes[0][0].tensor = ttnn.concat([track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor], dim=0)
        if track_boxes[0][1].shape[0] == 0:
            track_boxes[0][1] = ttnn.clone(sdc_track_boxes[0][1])
        else:
            track_boxes[0][1] = ttnn.concat([track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
        if track_boxes[0][2].shape[0] == 0:
            track_boxes[0][2] = ttnn.clone(sdc_track_boxes[0][2])
        else:
            track_boxes[0][2] = ttnn.concat([track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
        if track_boxes[0][3].shape[0] == 0:
            track_boxes[0][3] = ttnn.clone(sdc_track_boxes[0][3])
        else:
            track_boxes[0][3] = ttnn.concat([track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)
        memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = outs_seg["args_tuple"]
        outs_motion = self(bev_embed, track_query, lane_query, lane_query_pos, track_boxes)
        traj_results = self.get_trajs(outs_motion, track_boxes)
        bboxes, scores, labels, bbox_index, mask = track_boxes[0]
        outs_motion["track_scores"] = ttnn.unsqueeze(scores, 0)
        if len(labels.shape) == 1 and labels.shape[0] == 1:
            labels = ttnn.Tensor(np.array([0]), device=self.device, layout=ttnn.TILE_LAYOUT)

        def filter_vehicle_query(outs_motion, labels, vehicle_id_list):
            if len(labels.shape) < 1:  # No other obj query except sdc query.
                return None

            # select vehicle query according to vehicle_id_list
            vehicle_mask = ttnn.zeros(labels.shape, device=self.device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)
            for veh_id in vehicle_id_list:
                vehicle_mask = ttnn.bitwise_or(vehicle_mask, labels.item() == veh_id)
            outs_motion["traj_query"] = outs_motion["traj_query"][
                :, :, :
            ]  # vehicle_mask > 0] #vechile_mask=1 os keeping as True
            outs_motion["track_query"] = outs_motion["track_query"][
                :, :
            ]  # vehicle_mask > 0]#vechile_mask=1 os keeping as True
            outs_motion["track_query_pos"] = outs_motion["track_query_pos"][
                :, :
            ]  # vehicle_mask > 0]#vechile_mask=1 os keeping as True
            outs_motion["track_scores"] = outs_motion["track_scores"][
                :, :
            ]  # vehicle_mask > 0]#vechile_mask=1 os keeping as True
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

    def __call__(self, bev_embed, track_query, lane_query, lane_query_pos, track_bbox_results):
        dtype = track_query.dtype
        device = track_query.device
        num_groups = self.kmeans_anchors.shape[0]

        track_query = track_query[:, -1]

        reference_points_track = self._extract_tracking_centers(track_bbox_results, self.pc_range)

        track_query_pos = pos2posemb2d(reference_points_track)
        for index, layer in enumerate(self.boxes_query_embedding_layer):
            if layer == ttnn.relu:
                track_query_pos = layer(track_query_pos)
            else:
                track_query_pos = layer(
                    track_query_pos,
                    self.parameters["boxes_query_embedding_layer"][index].weight,
                    bias=self.parameters["boxes_query_embedding_layer"][index].bias,
                    dtype=ttnn.bfloat16,
                )

        learnable_query_pos = self.parameters["learnable_motion_query_embedding"].weight  # latent anchor (P*G, D)
        learnable_query_pos = ttnn.stack([learnable_query_pos[i : i + 6] for i in range(0, 24, 6)], dim=0)

        agent_level_anchors = ttnn.reshape(self.kmeans_anchors, (num_groups, self.num_anchor, self.predict_steps, 2))
        scene_level_ego_anchors = anchor_coordinate_transform(
            agent_level_anchors, track_bbox_results, with_translation_transform=True
        )  # B, A, G, P ,12 ,2
        scene_level_offset_anchors = anchor_coordinate_transform(
            agent_level_anchors, track_bbox_results, with_translation_transform=False
        )  # B, A, G, P ,12 ,2

        agent_level_norm = norm_points(agent_level_anchors, self.pc_range)
        scene_level_ego_norm = norm_points(scene_level_ego_anchors, self.pc_range)
        scene_level_offset_norm = norm_points(scene_level_offset_anchors, self.pc_range)

        agent_level_embedding = pos2posemb2d(agent_level_norm[..., -1, :])
        for index, layer in enumerate(self.agent_level_embedding_layer):
            if layer == ttnn.relu:
                agent_level_embedding = layer(agent_level_embedding)
            else:
                agent_level_embedding = layer(
                    agent_level_embedding,
                    self.parameters["agent_level_embedding_layer"][index].weight,
                    bias=self.parameters["agent_level_embedding_layer"][index].bias,
                    dtype=ttnn.bfloat16,
                )

        scene_level_ego_embedding = pos2posemb2d(scene_level_ego_norm[..., -1, :])
        for index, layer in enumerate(self.scene_level_ego_embedding_layer):
            if layer == ttnn.relu:
                scene_level_ego_embedding = layer(scene_level_ego_embedding)
            else:
                scene_level_ego_embedding = layer(
                    scene_level_ego_embedding,
                    self.parameters["scene_level_ego_embedding_layer"][index].weight,
                    bias=self.parameters["scene_level_ego_embedding_layer"][index].bias,
                    dtype=ttnn.bfloat16,
                )

        scene_level_offset_embedding = pos2posemb2d(scene_level_offset_norm[..., -1, :])

        for index, layer in enumerate(self.scene_level_offset_embedding_layer):
            if layer == ttnn.relu:
                scene_level_offset_embedding = layer(scene_level_offset_embedding)
            else:
                scene_level_offset_embedding = layer(
                    scene_level_offset_embedding,
                    self.parameters["scene_level_offset_embedding_layer"][index].weight,
                    bias=self.parameters["scene_level_offset_embedding_layer"][index].bias,
                    dtype=ttnn.bfloat16,
                )

        batch_size, num_agents = scene_level_ego_embedding.shape[0], scene_level_ego_embedding.shape[1]
        agent_level_embedding = ttnn.expand(
            ttnn.unsqueeze(ttnn.unsqueeze(agent_level_embedding, 0), 0), (batch_size, num_agents, -1, -1, -1)
        )
        learnable_embed = ttnn.expand(
            ttnn.unsqueeze(ttnn.unsqueeze(learnable_query_pos, 0), 0), (batch_size, num_agents, -1, -1, -1)
        )

        scene_level_offset_anchors = self.group_mode_query_pos(track_bbox_results, scene_level_offset_anchors)

        agent_level_embedding = self.group_mode_query_pos(track_bbox_results, agent_level_embedding)
        scene_level_ego_embedding = self.group_mode_query_pos(track_bbox_results, scene_level_ego_embedding)

        scene_level_offset_embedding = self.group_mode_query_pos(track_bbox_results, scene_level_offset_embedding)
        learnable_embed = self.group_mode_query_pos(track_bbox_results, learnable_embed)

        init_reference = ttnn.clone(scene_level_offset_anchors)

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
            traj_reg_branches=self.parameters["traj_reg_branches"],
            traj_cls_branches=self.parameters["traj_cls_branches"],
            # anchor embeddings
            agent_level_embedding=agent_level_embedding,
            scene_level_ego_embedding=scene_level_ego_embedding,
            scene_level_offset_embedding=scene_level_offset_embedding,
            learnable_embed=learnable_embed,
            # anchor positional embeddings layers
            agent_level_embedding_layer=self.parameters["agent_level_embedding_layer"],
            scene_level_ego_embedding_layer=self.parameters["scene_level_ego_embedding_layer"],
            scene_level_offset_embedding_layer=self.parameters["scene_level_offset_embedding_layer"],
            spatial_shapes=ttnn.Tensor(
                np.array([[self.bev_h, self.bev_w]]), device=self.device, layout=ttnn.TILE_LAYOUT
            ),
            level_start_index=ttnn.Tensor(np.array([0]), device=self.device, layout=ttnn.TILE_LAYOUT),
        )

        for lvl in range(inter_states.shape[0]):
            outputs_class = ttnn.clone(inter_states[lvl])
            for index, layer in enumerate(self.traj_cls_branches[lvl]):
                if layer == ttnn.relu:
                    outputs_class = layer(outputs_class)
                elif layer == ttnn.linear:
                    outputs_class = layer(
                        outputs_class,
                        self.parameters["traj_cls_branches"][lvl][index].weight,
                        bias=self.parameters["traj_cls_branches"][lvl][index].bias,
                        dtype=ttnn.bfloat16,
                    )
                else:
                    outputs_class = layer(
                        outputs_class,
                        weight=self.parameters["traj_cls_branches"][lvl][index].weight,
                        bias=self.parameters["traj_cls_branches"][lvl][index].bias,
                        epsilon=1e-12,
                    )

            tmp = ttnn.clone(inter_states[lvl])
            for index, layer in enumerate(self.traj_reg_branches[lvl]):
                if layer == ttnn.relu:
                    tmp = layer(tmp)
                elif layer == ttnn.linear:
                    tmp = layer(
                        tmp,
                        self.parameters["traj_reg_branches"][lvl][index].weight,
                        bias=self.parameters["traj_reg_branches"][lvl][index].bias,
                        dtype=ttnn.bfloat16,
                    )

            tmp = ttnn.reshape(tmp, (tmp.shape[0], tmp.shape[1], tmp.shape[2], self.predict_steps, 5))

            tmp_a = ttnn.clone(tmp[..., :2])
            tmp_b = ttnn.clone(tmp[..., 2:])
            tmp_a = ttnn.cumsum(tmp_a, dim=3)
            tmp = ttnn.concat([tmp_a, tmp_b], dim=-1)
            ttnn.deallocate(tmp_a)
            ttnn.deallocate(tmp_b)

            outputs_class = ttnn.squeeze(outputs_class, 3)
            x_max = ttnn.max(outputs_class, dim=-1, keepdim=True)
            x_shifted = ttnn.sub(outputs_class, x_max)
            exp_x = ttnn.exp(x_shifted)
            sum_exp = ttnn.sum(exp_x, dim=-1, keepdim=True)
            log_sum_exp = ttnn.log(sum_exp)
            outputs_class = ttnn.sub(x_shifted, log_sum_exp)

            outputs_traj_scores.append(outputs_class)

            temp = []
            for bs in range(tmp.shape[0]):
                temp.append(ttnn.unsqueeze(bivariate_gaussian_activation_motion_head(tmp[bs]), 0))
            tmp = ttnn.concat(temp, dim=0)

            outputs_trajs.append(tmp)
        outputs_traj_scores = ttnn.stack(outputs_traj_scores, dim=0)
        outputs_trajs = ttnn.stack(outputs_trajs, dim=0)

        B, A_track, D = track_query.shape
        valid_traj_masks = ttnn.ones((B, A_track), device=self.device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)
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
        self.cls2group = self.cls2group
        for i in range(batch_size):
            bboxes, scores, labels, bbox_index, mask = bbox_results[i]
            label = ttnn.clone(labels)
            grouped_label = self.cls2group[label.item()]
            grouped_mode_query_pos = []
            for j in range(agent_num):
                grouped_mode_query_pos.append(mode_query_pos[i, j, grouped_label.item()])
            batched_mode_query_pos.append(ttnn.stack(grouped_mode_query_pos, 0))
        return ttnn.stack(batched_mode_query_pos, 0)

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
