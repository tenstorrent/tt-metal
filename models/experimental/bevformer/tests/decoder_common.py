# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Configuration and helpers for the detection decoder tests."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.utility_functions import comp_pcc
from models.experimental.bevformer.config.decoder_config import CODE_SIZE, CODE_XY, CODE_Z
from models.experimental.bevformer.reference.decoder import DetectionTransformerDecoder, inverse_sigmoid, reg_branch
from models.experimental.bevformer.reference.ms_deformable_attention import MSDeformableAttention
from models.experimental.bevformer.tests.backbone_common import assert_pcc

# BEVFormer tiny and base share the decoder; only the BEV grid it attends over differs.
NUM_QUERY = 900
EMBED_DIMS = 256
NUM_HEADS = 8
NUM_LAYERS = 6
FEEDFORWARD_CHANNELS = 512
NUM_POINTS = 4

BEV_SHAPES = {"tiny": (50, 50), "base": (200, 200)}

# Spread of the random weights on top of the upstream init, set to what the BEVFormer-base
# checkpoint's decoder shows on random_bev_features (per layer: sampling offsets 1.0-1.7 px off
# the init pattern, cross-attention logits of std 0.45-0.9, self-attention logits of std
# 2.2-5), so the test sees trained-like sampling and softmaxes.
SAMPLING_OFFSET_STD_PX = 1.3
ATTENTION_LOGIT_STD = 0.7
SELF_ATTENTION_LOGIT_STD = 3.0

# Scale of the reg branches' last Linear over PyTorch's default init, whose outputs have std
# ~0.08. The checkpoint's branches refine the reference points by ~0.01 in logit space per
# layer, and emit the other box channels with std ~0.6. Larger refinements move every later
# layer's sampling points further and amplify the decoder's error through them; near-constant
# box channels make their PCC measure noise.
REG_REFINEMENT_SCALE = 0.125
REG_BOX_SCALE = 7.5

# Correlation length of the random BEV features, in cells. The encoder's BEV features are
# spatially smooth; white noise instead makes every sample position error an O(1) change in
# the sampled value, which the refinement feeds back into the next layer's positions. With
# white noise the reference run on bfloat16-rounded inputs and weights (fp32 compute) falls
# to PCC 0.65 against itself by the last layer on the 200x200 grid, so no bound on the TT
# port would mean anything.
BEV_FEATURE_CELLS = 4


def _init_cross_attention(msda, generator):
    """BEVFormer ``CustomMSDeformableAttention.init_weights`` (mmcv's
    ``MultiScaleDeformableAttention`` init) plus trained-like random weights."""
    heads, levels, points = msda.num_heads, msda.num_levels, msda.num_points
    in_features = msda.sampling_offsets.in_features

    thetas = torch.arange(heads, dtype=torch.float32) * (2.0 * math.pi / heads)
    grid = torch.stack([thetas.cos(), thetas.sin()], -1)
    grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(heads, 1, 1, 2).repeat(1, levels, points, 1)
    for i in range(points):
        grid[:, :, i, :] *= i + 1
    msda.sampling_offsets.bias.copy_(grid.flatten())

    # The Linears read query + query_pos, of variance 2 (see _init_self_attention), so a
    # weight of std s gives outputs of std s * sqrt(2 * in_features).
    fan_in_scale = 1.0 / math.sqrt(2 * in_features)
    msda.sampling_offsets.weight.copy_(
        torch.randn(msda.sampling_offsets.weight.shape, generator=generator) * SAMPLING_OFFSET_STD_PX * fan_in_scale
    )
    msda.attention_weights.weight.copy_(
        torch.randn(msda.attention_weights.weight.shape, generator=generator) * ATTENTION_LOGIT_STD * fan_in_scale
    )
    msda.attention_weights.bias.zero_()
    for proj in (msda.value_proj, msda.output_proj):
        bound = math.sqrt(6.0 / (proj.in_features + proj.out_features))
        proj.weight.copy_(torch.rand(proj.weight.shape, generator=generator) * 2 * bound - bound)
        proj.bias.zero_()


def _init_self_attention(mha, generator):
    """Q and K spread so the ``q . k / sqrt(head_dim)`` logits have std SELF_ATTENTION_LOGIT_STD.

    Q and K project ``query + query_pos``, of variance 2 (a LayerNorm-ed or random query plus
    a random position). A weight of std s then gives q, k of std s * sqrt(2 * embed_dims),
    and the scaled logit's std is std(q) * std(k).
    """
    embed_dims = mha.embed_dim
    std = math.sqrt(SELF_ATTENTION_LOGIT_STD / (2 * embed_dims))
    qk_rows = mha.in_proj_weight[: 2 * embed_dims]
    qk_rows.copy_(torch.randn(qk_rows.shape, generator=generator) * std)


def build_reference_decoder(seed=0):
    """BEVFormer's decoder with dummy weights: ``_init_cross_attention`` and ``_init_self_attention``
    on top of PyTorch's default init, which the rest keeps."""
    torch.manual_seed(seed)
    model = DetectionTransformerDecoder(
        num_layers=NUM_LAYERS,
        embed_dims=EMBED_DIMS,
        num_heads=NUM_HEADS,
        feedforward_channels=FEEDFORWARD_CHANNELS,
        num_points=NUM_POINTS,
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, MSDeformableAttention):
                _init_cross_attention(module, generator)
            elif isinstance(module, nn.MultiheadAttention):
                _init_self_attention(module, generator)
    return model.eval().requires_grad_(False)


def init_reg_branches(branches):
    """Scale each branch's last Linear to trained-like outputs: REG_REFINEMENT_SCALE on the
    rows that refine the reference points (cx, cy, cz), REG_BOX_SCALE on the others."""
    scale = torch.full((CODE_SIZE, 1), REG_BOX_SCALE)
    scale[CODE_XY] = scale[CODE_Z] = REG_REFINEMENT_SCALE
    with torch.no_grad():
        for branch in branches:
            branch[-1].weight.mul_(scale)
            branch[-1].bias.mul_(scale[:, 0])
    return branches


def build_reg_branches(seed=1):
    """BEVFormer's per-layer box regression head, ``Linear-ReLU-Linear-ReLU-Linear(code_size)``,
    with init_reg_branches' scale."""
    torch.manual_seed(seed)
    branches = init_reg_branches(nn.ModuleList(reg_branch(EMBED_DIMS, CODE_SIZE) for _ in range(NUM_LAYERS)))
    return branches.eval().requires_grad_(False)


def random_reference_points(batch_size, generator=None):
    """Uniform in [0, 1], with a slice of points on and just inside the grid edges.

    Edge points exercise out-of-bounds zero padding in grid_sample and the eps clamp of
    inverse_sigmoid, which trained queries near the BEV border reach.
    """
    points = torch.rand(batch_size, NUM_QUERY, 3, generator=generator)
    edges = torch.tensor([0.0, 1e-3, 1.0 - 1e-3, 1.0])
    num_edge = NUM_QUERY // 10
    points[:, :num_edge] = edges[torch.randint(len(edges), (batch_size, num_edge, 3), generator=generator)]
    return points


def random_bev_features(bev_shape, batch_size, generator=None):
    """Unit-variance ``(bev_h * bev_w, bs, C)`` features, smooth over ``BEV_FEATURE_CELLS`` cells."""
    bev_h, bev_w = bev_shape
    coarse = torch.randn(
        batch_size,
        EMBED_DIMS,
        math.ceil(bev_h / BEV_FEATURE_CELLS),
        math.ceil(bev_w / BEV_FEATURE_CELLS),
        generator=generator,
    )
    features = F.interpolate(coarse, size=(bev_h, bev_w), mode="bilinear", align_corners=False)
    features = features / features.std()
    return features.flatten(2).permute(2, 0, 1).contiguous()


def random_decoder_inputs(bev_shape, batch_size=1, seed=None):
    """Sequence-first query, query_pos and BEV value, and reference points in [0, 1]."""
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return dict(
        query=torch.randn(NUM_QUERY, batch_size, EMBED_DIMS, generator=generator),
        value=random_bev_features(bev_shape, batch_size, generator),
        query_pos=torch.randn(NUM_QUERY, batch_size, EMBED_DIMS, generator=generator),
        reference_points=random_reference_points(batch_size, generator),
    )


def layer_metrics(expected, actual, input_reference_points, bev_shape):
    """Per-layer accuracy of ``actual`` (output, reference points) against ``expected``, for logging.

    Per layer: the output PCC, the PCC of the refinement step in logit space for xy and z
    apart, and the mean xy error of the refined points in BEV pixels. The steps show the
    refinement's accuracy, which the absolute points barely reflect: they move little per layer.
    """
    (expected_output, expected_points), (actual_output, actual_points) = expected, actual
    bev_h, bev_w = bev_shape
    px_scale = torch.tensor([bev_w, bev_h], dtype=torch.float32)

    def steps(points):
        logits = inverse_sigmoid(torch.cat([input_reference_points[None], points]))
        return logits[1:] - logits[:-1]

    expected_steps, actual_steps = steps(expected_points), steps(actual_points)
    return [
        dict(
            output=comp_pcc(expected_output[layer], actual_output[layer])[1],
            refine_xy=comp_pcc(expected_steps[layer][..., :2], actual_steps[layer][..., :2])[1],
            refine_z=comp_pcc(expected_steps[layer][..., 2:], actual_steps[layer][..., 2:])[1],
            px=((actual_points[layer][..., :2] - expected_points[layer][..., :2]).abs() * px_scale).mean().item(),
        )
        for layer in range(expected_output.shape[0])
    ]


def assert_channels_close(expected, actual, pcc=0.99):
    """PCC of every last-dim channel apart, so a channel of small range cannot hide behind
    the others."""
    for channel in range(expected.shape[-1]):
        assert_pcc(expected[..., channel], actual[..., channel], pcc)
