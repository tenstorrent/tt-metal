# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""BEVFormer-base's configuration, shared by the PyTorch reference and the TTNN port.

The architecture follows upstream's ``projects/configs/bevformer/bevformer_base.py``
(fundamentalvision/BEVFormer): the reference modules take these values as their defaults. The TTNN section holds the port's precision
and memory choices, which the reference does not see.

Sections:
- Cameras and images
- Backbone and FPN
- BEV grid, shared sizes and perception transformer
- Encoder
- Decoder and head
- Box code and coder
- Deformable attention
- TTNN precision and memory
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ttnn

# --- Cameras and images -------------------------------------------------------------------------

NUM_CAMS = 6
# The cameras' images, (width, height).
RAW_IMAGE_SIZE = (1600, 900)
# Upstream's PadMultiViewImage pads the images at the bottom and right to a multiple of this, and
# the images' img_shape reports the padded size.
IMAGE_SIZE_DIVISOR = 32


def padded_image_size(width, height):
    """``(width, height)`` padded up to a multiple of ``IMAGE_SIZE_DIVISOR``."""
    return tuple(math.ceil(size / IMAGE_SIZE_DIVISOR) * IMAGE_SIZE_DIVISOR for size in (width, height))


# The backbone's input, 1600x928.
IMAGE_WIDTH, IMAGE_HEIGHT = padded_image_size(*RAW_IMAGE_SIZE)
# img_norm_cfg: the images are this BGR mean subtracted and divided by this std, so they stay in
# pixel units.
IMAGE_MEAN_BGR = (103.530, 116.280, 123.675)
IMAGE_STD_BGR = (1.0, 1.0, 1.0)

# --- Backbone and FPN ---------------------------------------------------------------------------

# ResNet101: caffe style, DCNv2 in layer3 and layer4, C3-C5 out (``reference/resnet.py``).
RESNET_KWARGS = dict(
    depth=101,
    in_channels=3,
    stem_channels=None,
    base_channels=64,
    num_stages=4,
    strides=(1, 2, 2, 2),
    dilations=(1, 1, 1, 1),
    out_indices=(1, 2, 3),
    style="caffe",
    deep_stem=False,
    avg_down=False,
    frozen_stages=4,
    conv_cfg=None,
    norm_cfg={"type": "BN2d", "requires_grad": False},
    norm_eval=True,
    dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
    stage_with_dcn=(False, False, True, True),
    plugins=None,
    with_cp=False,
    zero_init_residual=True,
    init_cfg=None,
)

# Four 256-channel levels from C3-C5, the fourth from an extra stride-2 conv (``reference/fpn.py``).
FPN_KWARGS = dict(
    in_channels=[512, 1024, 2048],
    out_channels=256,
    start_level=0,
    add_extra_convs="on_output",
    num_outs=4,
    relu_before_extra_convs=True,
)


def fpn_spatial_shapes(height, width, num_levels):
    """The FPN levels' ``(h, w)`` for a ``height x width`` image: strides 8, 16, ..., rounded up."""
    return tuple((-(-height // (8 << level)), -(-width // (8 << level))) for level in range(num_levels))


SPATIAL_SHAPES = fpn_spatial_shapes(IMAGE_HEIGHT, IMAGE_WIDTH, FPN_KWARGS["num_outs"])

# --- BEV grid, shared sizes and perception transformer ----------------------------------------

BEV_H = 200
BEV_W = 200
EMBED_DIMS = 256
# Shared by the encoder's and the decoder's attentions and FFNs.
NUM_HEADS = 8
FEEDFORWARD_CHANNELS = 512
NUM_LEVELS = 4
# (x_min, y_min, z_min, x_max, y_max, z_max) in metres: nuScenes' range, which the BEV grid, the
# encoder's pillars and the head's box centers share.
PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
CAN_BUS_DIMS = 18
# Cell the previous BEV rotates about, (x, y): upstream's default for every grid, tiny's 50x50
# included, where it lies outside the grid; it is not derived from the grid size.
ROTATE_CENTER = (100, 100)

# --- Encoder ------------------------------------------------------------------------------------

# SCA is the spatial cross-attention into the cameras, TSA the temporal self-attention over the
# previous BEV.
ENCODER_NUM_LAYERS = 6
# The SCA's sampling points per head and level, split over the pillar's heights.
SCA_NUM_POINTS = 8
NUM_POINTS_IN_PILLAR = 4
TSA_NUM_POINTS = 4

# --- Decoder and head ---------------------------------------------------------------------------

DECODER_NUM_LAYERS = 6
DECODER_NUM_POINTS = 4
NUM_QUERY = 900
NUM_CLASSES = 10
# inverse_sigmoid's clamp, which the decoder's refinement of its reference points goes through.
INVERSE_SIGMOID_EPS = 1e-5

# --- Box code and coder -------------------------------------------------------------------------

# The box code: (cx, cy, w, l, cz, h, sin, cos, vx, vy), the sizes as logs. In the box codes, the
# reg branches' raw output, cx, cy and cz are logit offsets, which the decoder adds to its reference
# points' logits; in the box predictions, the head's output, they are the refined centers in metres.
# The other channels are the same in both.
CODE_SIZE = 10
CODE_XY = slice(0, 2)
CODE_WL = slice(2, 4)
CODE_Z = slice(4, 5)
CODE_H = slice(5, 6)
CODE_SIN = slice(6, 7)
CODE_COS = slice(7, 8)
CODE_VELOCITY = slice(8, 10)

# cx, cy and cz, in the order of the reference points' (x, y, z).
CODE_CENTERS = [*range(CODE_SIZE)][CODE_XY] + [*range(CODE_SIZE)][CODE_Z]
# The TTNN port's channel order for box codes: the centers first, (cx, cy, cz, w, l, h, sin, cos,
# vx, vy), so its decoder takes the center offsets in one slice, CENTERS_FIRST_CENTERS. Its reg
# branches emit it, from the last Linear's rows in this order.
CENTERS_FIRST = CODE_CENTERS + [channel for channel in range(CODE_SIZE) if channel not in CODE_CENTERS]
CENTERS_FIRST_CENTERS = slice(0, len(CODE_CENTERS))


def _box_pred_pieces():
    """The TTNN head's box predictions, in box code order, as one concat of slices of its refined
    centers ``"centers"`` (x, y, z) and its centers-first codes ``"codes"``."""
    pieces = []
    for channel in range(CODE_SIZE):
        if channel in CODE_CENTERS:
            source, index = "centers", CODE_CENTERS.index(channel)
        else:
            source, index = "codes", CENTERS_FIRST.index(channel)
        if pieces and pieces[-1][0] == source and pieces[-1][1].stop == index:
            pieces[-1] = (source, slice(pieces[-1][1].start, index + 1))
        else:
            pieces.append((source, slice(index, index + 1)))
    return tuple(pieces)


BOX_PRED_PIECES = _box_pred_pieces()

# The coder's decoded box: (cx, cy, cz, w, l, h, yaw, vx, vy), cz at the box's gravity center.
BOX_CENTER = slice(0, 3)
BOX_Z = slice(2, 3)
BOX_SIZE = slice(3, 6)
BOX_H = slice(5, 6)
BOX_YAW = slice(6, 7)
BOX_VELOCITY = slice(7, 9)
# Box centers outside this range are dropped by the coder.
POST_CENTER_RANGE = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)
# Boxes the coder keeps per sample, by score.
MAX_NUM = 300

# --- Deformable attention -----------------------------------------------------------------------


@dataclass
class DeformableAttentionConfig:
    """Sizes of a multi-scale deformable attention (``reference/ms_deformable_attention.py``);
    ``num_points`` per head and level differs between its users, so it has no default."""

    num_points: int
    embed_dims: int = EMBED_DIMS
    num_heads: int = NUM_HEADS
    num_levels: int = NUM_LEVELS
    batch_first: bool = True

    def __post_init__(self):
        if self.embed_dims % self.num_heads != 0:
            raise ValueError(f"embed_dims ({self.embed_dims}) must be divisible by num_heads ({self.num_heads})")


# --- TTNN precision and memory ------------------------------------------------------------------

# The ttnn dtypes, by name. The PyTorch reference imports this module and must not need ttnn, so
# they are looked up in ttnn on access, by the module ``__getattr__`` at the end of this file.
WEIGHT_DTYPE: "ttnn.DataType"
CONV_WEIGHT_DTYPE: "ttnn.DataType"
GRID_DTYPE: "ttnn.DataType"
SCORE_DTYPE: "ttnn.DataType"
BOX_DTYPE: "ttnn.DataType"
_TTNN_DTYPES = {
    # The transformer's, the head's and the DCNs' weights.
    "WEIGHT_DTYPE": "bfloat16",
    # The backbone's and FPN's conv weights, as conv2d and the DRAM 1x1 convs' matmuls read them.
    "CONV_WEIGHT_DTYPE": "bfloat8_b",
    # The deformable attentions' reference points, in the encoder and the decoder, which the fused
    # MSDA op samples at: in bfloat16 a point in (0.5, 1) moves in steps of 2^-8, 0.8 px on the
    # 200x200 grid.
    "GRID_DTYPE": "float32",
    # The head's class logits, which the coder ranks: in bfloat16 many of the num_query *
    # num_classes scores tie, and the top-k order departs from the reference's.
    "SCORE_DTYPE": "float32",
}
# The head's box predictions: their centers are the decoder's refined reference points.
_TTNN_DTYPES["BOX_DTYPE"] = _TTNN_DTYPES["GRID_DTYPE"]

# Bisection steps of the coder's exact top-k (TTNMSFreeCoder.topk), which narrow the k-th logit's
# bracket around the bfloat16 pivot, 2^-5 of the pivot wide, down to one float32 value; 19
# suffice, the rest is margin.
CODER_BISECTION_STEPS = 22


# The backbone and FPN, for 6 cameras at 1600x928. Layer indices count the ResNet layers from 0
# (layer1); level indices the FPN levels from 0 (C3).

# Layers whose convs accumulate in an fp32 destination register (layer3 and layer4, the DCN
# layers; a DCN conv2 always does): with bfloat16 accumulation, activations with outlier channels
# lose precision through these layers' 26 blocks until C5 falls below PCC 0.99. The tests' dummy
# weights have no such channels and do not show it.
FP32_ACC_STAGES = (2, 3)

# Layers whose activations are kept in DRAM: all four. layer1, layer2 and layer4 do not fit in
# L1: layer1 and layer2 work on 6 x 232 x 400 x 256 tensors (285 MB in bfloat16), and layer4's
# 2048-channel 1x1 convs overflow L1 when sharded. layer3 would fit at bfloat16 accumulation,
# but with FP32_ACC_STAGES' fp32 accumulation its L1-sharded convs fail to allocate (their
# circular buffers clash with L1 buffers).
DRAM_ACTIVATION_STAGES = tuple(sorted({0, 1, 3} | set(FP32_ACC_STAGES)))

# FPN levels whose convs keep their activations in DRAM because they do not fit in L1:
# C3 is 6 x 116 x 200 x 512 bfloat16 (143 MB) and C4 is 6 x 58 x 100 x 1024 bfloat16 (71 MB).
DRAM_ACTIVATION_LEVELS = (0, 1)

# Width slices for the spatial convs of the DRAM layers and levels, lowered per conv to what
# its output width allows. The tightest is the strided 1x1 downsample that opens layer2: its
# 6 x 232 x 400 x 256 input is in DRAM, and each slice is read into L1 as bfloat16
# ROW_MAJOR for the halo, 3.25 MB per L1 bank in total against 576 KB free, so it needs at
# least 6 slices. 8 leaves margin over that. TTConv2D lowers the count per conv to its
# output width in tiles, as a width slice is whole tiles wide: 7 for this conv's 200-wide output.
DRAM_CONV_SLICES = 8

# Layers whose downsample conv is block sharded (layer3, layer4) and FPN levels whose output
# conv is (C3, C4).
BLOCK_SHARDED_DOWNSAMPLE_STAGES = (2, 3)
BLOCK_SHARDED_LEVELS = (0, 1)

# conv2d's act_block_h overrides: the stem conv's, the non-DCN 3x3 conv2's of the bottlenecks and
# the block-sharded FPN output convs'. None of these, nor the sharding above, is tuned for 928x1600.
STEM_ACT_BLOCK_H = 64
CONV2_ACT_BLOCK_H = 32
FPN_BLOCK_SHARDED_ACT_BLOCK_H = 128


def tt_resnet_kwargs():
    """TTResNet's memory and precision arguments for this configuration, which
    ``create_resnet_parameters`` records in the backbone's parameters. TTResNet's ``out_indices``
    defaults to all four layers; pass ``RESNET_KWARGS["out_indices"]`` for BEVFormer-base's C3-C5."""
    return dict(
        dram_activation_stages=DRAM_ACTIVATION_STAGES,
        dram_conv_slices=DRAM_CONV_SLICES,
        block_sharded_downsample_stages=BLOCK_SHARDED_DOWNSAMPLE_STAGES,
        fp32_acc_stages=FP32_ACC_STAGES,
        stem_act_block_h=STEM_ACT_BLOCK_H,
        conv2_act_block_h=CONV2_ACT_BLOCK_H,
        weights_dtype=_ttnn_dtype("CONV_WEIGHT_DTYPE"),
    )


def tt_fpn_kwargs():
    """TTFPN's memory and precision arguments for this configuration, which
    ``create_fpn_parameters`` records in the FPN's parameters."""
    return dict(
        dram_activation_levels=DRAM_ACTIVATION_LEVELS,
        dram_conv_slices=DRAM_CONV_SLICES,
        block_sharded_levels=BLOCK_SHARDED_LEVELS,
        block_sharded_act_block_h=FPN_BLOCK_SHARDED_ACT_BLOCK_H,
        weights_dtype=_ttnn_dtype("CONV_WEIGHT_DTYPE"),
    )


def _ttnn_dtype(name):
    """``_TTNN_DTYPES[name]`` as a ttnn dtype. Without ttnn, the ``ModuleNotFoundError`` reaches
    the caller as is, naming the missing module."""
    import ttnn

    return {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "float32": ttnn.float32}[_TTNN_DTYPES[name]]


def __getattr__(name):
    """The module attribute hook of PEP 562: resolves ``_TTNN_DTYPES`` on access, so ttnn is
    imported only by the code that uses them."""
    if name not in _TTNN_DTYPES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _ttnn_dtype(name)


def __dir__():
    return sorted(set(globals()) | set(_TTNN_DTYPES))
