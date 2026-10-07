# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer-base.

- ``tt_bevformer``: the detector, ``TtBEVFormer``
- ``model_preprocessing``: every part's parameters, ``create_bevformer_parameters`` for the whole
- ``tt_resnet``, ``tt_fpn``, ``tt_modulated_deform_conv``, ``tt_common``: the backbone and neck
- ``tt_perception_transformer``, ``tt_encoder``, ``tt_temporal_self_attention``,
  ``tt_spatial_cross_attention``, ``tt_ms_deformable_attention``, ``tt_point_sampling_3d_2d``
- ``tt_head``, ``tt_decoder``, ``tt_nms_free_coder``
"""
