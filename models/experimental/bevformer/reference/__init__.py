# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BEVFormer-base in PyTorch, the reference the TTNN port is checked against.

- ``bevformer``: the detector, ``build_bevformer_base`` and ``load_bevformer_checkpoint``
- ``resnet``, ``fpn``: the image backbone and neck
- ``perception_transformer``: the encoder's glue (CAN bus, embeddings, previous BEV)
- ``encoder``, ``temporal_self_attention``, ``spatial_cross_attention``, ``point_sampling_3d_2d``
- ``head``, ``decoder``, ``nms_free_coder``
- ``ms_deformable_attention``: the deformable attention the attentions build on
"""
