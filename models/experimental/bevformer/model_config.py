# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""BEVFormer's configuration, shared by the PyTorch reference and the TTNN port.

The architecture follows upstream's ``projects/configs/bevformer/bevformer_{tiny,small,base}.py``
(fundamentalvision/BEVFormer): the reference modules take these values as their defaults. The TTNN section holds the port's precision
and memory choices, which the reference does not see.

The ``BEVFORMER_VARIANT`` environment variable picks the variant, ``base`` by default, when this
module is first imported; the values the variants do not share follow from it.

Sections:
- Variant
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
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ttnn

# --- Variant ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """What one of upstream's BEVFormer configurations sets apart from the others."""

    # RandomScaleImageMultiViewImage's scale of the CAMERA_IMAGE_SIZE camera images.
    image_scale: float
    # img_norm_cfg, per channel in the order the backbone reads them: RGB for tiny, BGR for small
    # and base.
    image_mean: tuple
    image_std: tuple
    resnet_depth: int
    # Where a bottleneck puts its stride: on the 3x3 conv ("pytorch") or on the first 1x1 ("caffe").
    resnet_style: str
    # DCNv2 in layer3 and layer4.
    with_dcn: bool
    # The ResNet layers that feed the FPN, from 0 (layer1).
    out_indices: tuple
    # The FPN's levels: one per out_indices entry, then extra stride-2 convs on the last.
    num_levels: int
    bev_size: int
    encoder_num_layers: int


_CAFFE_IMAGE_NORM = dict(image_mean=(103.530, 116.280, 123.675), image_std=(1.0, 1.0, 1.0))
VARIANTS = {
    # ResNet50 without DCN; C5 alone feeds a single-level FPN.
    "tiny": Variant(
        image_scale=0.5,
        image_mean=(123.675, 116.28, 103.53),
        image_std=(58.395, 57.12, 57.375),
        resnet_depth=50,
        resnet_style="pytorch",
        with_dcn=False,
        out_indices=(3,),
        num_levels=1,
        bev_size=50,
        encoder_num_layers=3,
    ),
    # base's backbone, but C5 alone feeds a single-level FPN.
    "small": Variant(
        image_scale=0.8,
        **_CAFFE_IMAGE_NORM,
        resnet_depth=101,
        resnet_style="caffe",
        with_dcn=True,
        out_indices=(3,),
        num_levels=1,
        bev_size=150,
        encoder_num_layers=3,
    ),
    "base": Variant(
        image_scale=1.0,
        **_CAFFE_IMAGE_NORM,
        resnet_depth=101,
        resnet_style="caffe",
        with_dcn=True,
        out_indices=(1, 2, 3),
        num_levels=4,
        bev_size=200,
        encoder_num_layers=6,
    ),
}
VARIANT = os.environ.get("BEVFORMER_VARIANT", "base")
if VARIANT not in VARIANTS:
    raise ValueError(f"BEVFORMER_VARIANT is {VARIANT!r}, expected one of {sorted(VARIANTS)}")
_VARIANT = VARIANTS[VARIANT]

# --- Cameras and images -------------------------------------------------------------------------

NUM_CAMS = 6
# nuScenes' camera images, (width, height), the size their intrinsics are calibrated at.
CAMERA_IMAGE_SIZE = (1600, 900)
# The camera images as the variant scales them, (width, height): CAMERA_IMAGE_SIZE for base.
RAW_IMAGE_SIZE = tuple(round(size * _VARIANT.image_scale) for size in CAMERA_IMAGE_SIZE)
# Upstream's PadMultiViewImage pads the images at the bottom and right to a multiple of this, and
# the images' img_shape reports the padded size.
IMAGE_SIZE_DIVISOR = 32


def padded_image_size(width, height):
    """``(width, height)`` padded up to a multiple of ``IMAGE_SIZE_DIVISOR``."""
    return tuple(math.ceil(size / IMAGE_SIZE_DIVISOR) * IMAGE_SIZE_DIVISOR for size in (width, height))


# The backbone's input: 1600x928 for base, 1280x736 for small, 800x480 for tiny.
IMAGE_WIDTH, IMAGE_HEIGHT = padded_image_size(*RAW_IMAGE_SIZE)
# img_norm_cfg: the images are this mean subtracted and divided by this std, per channel. small's
# and base's std of 1 leaves them in pixel units.
IMAGE_MEAN = _VARIANT.image_mean
IMAGE_STD = _VARIANT.image_std

# --- Backbone and FPN ---------------------------------------------------------------------------

# ResNet101, caffe style, DCNv2 in layer3 and layer4, for small and base; ResNet50, pytorch style,
# without DCN, for tiny (``reference/resnet.py``). base takes C3-C5 out, small and tiny C5.
RESNET_KWARGS = dict(
    depth=_VARIANT.resnet_depth,
    in_channels=3,
    stem_channels=None,
    base_channels=64,
    num_stages=4,
    strides=(1, 2, 2, 2),
    dilations=(1, 1, 1, 1),
    out_indices=_VARIANT.out_indices,
    style=_VARIANT.resnet_style,
    deep_stem=False,
    avg_down=False,
    frozen_stages=4,
    conv_cfg=None,
    norm_cfg={"type": "BN2d", "requires_grad": False},
    norm_eval=True,
    dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False} if _VARIANT.with_dcn else None,
    stage_with_dcn=(False, False, _VARIANT.with_dcn, _VARIANT.with_dcn),
    plugins=None,
    with_cp=False,
    zero_init_residual=True,
    init_cfg=None,
)

# 256-channel levels, one per backbone output, and for base a fourth from an extra stride-2 conv
# (``reference/fpn.py``). ResNet layer i outputs 256 << i channels at stride 4 << i.
FPN_KWARGS = dict(
    in_channels=[256 << i for i in _VARIANT.out_indices],
    out_channels=256,
    start_level=0,
    add_extra_convs="on_output",
    num_outs=_VARIANT.num_levels,
    relu_before_extra_convs=True,
)
# The stride of the FPN's first level: 8 for base's C3, 32 for small's and tiny's C5.
FPN_FIRST_STRIDE = 4 << _VARIANT.out_indices[0]


def fpn_spatial_shapes(height, width, num_levels, first_stride=8):
    """The FPN levels' ``(h, w)`` for a ``height x width`` image: strides ``first_stride``, twice
    that, ..., rounded up."""
    return tuple(
        (-(-height // (first_stride << level)), -(-width // (first_stride << level))) for level in range(num_levels)
    )


SPATIAL_SHAPES = fpn_spatial_shapes(IMAGE_HEIGHT, IMAGE_WIDTH, FPN_KWARGS["num_outs"], FPN_FIRST_STRIDE)

# --- BEV grid, shared sizes and perception transformer ----------------------------------------

BEV_H = _VARIANT.bev_size
BEV_W = _VARIANT.bev_size
EMBED_DIMS = 256
# Shared by the encoder's and the decoder's attentions and FFNs.
NUM_HEADS = 8
FEEDFORWARD_CHANNELS = 512
NUM_LEVELS = _VARIANT.num_levels
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
ENCODER_NUM_LAYERS = _VARIANT.encoder_num_layers
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


# The backbone and FPN. Their choices below are sized for base's 6 cameras at 1600x928; small and
# tiny take them unchanged, apart from the FPN levels, which follow the variant's outputs. Layer
# indices count the ResNet layers from 0 (layer1); level indices the FPN levels from 0, the first
# backbone output.

# Layers whose convs accumulate in an fp32 destination register (layer3 and layer4; a DCN conv2
# always does): with bfloat16 accumulation, activations with outlier channels lose precision
# through base's 26 blocks there until C5 falls below PCC 0.99. The tests' dummy weights have no
# such channels and do not show it.
FP32_ACC_STAGES = (2, 3)

# Layers whose activations are kept in DRAM: all four. layer1, layer2 and layer4 do not fit in
# L1 for base: layer1 and layer2 work on 6 x 232 x 400 x 256 tensors (285 MB in bfloat16), and layer4's
# 2048-channel 1x1 convs overflow L1 when sharded. layer3 would fit at bfloat16 accumulation,
# but with FP32_ACC_STAGES' fp32 accumulation its L1-sharded convs fail to allocate (their
# circular buffers clash with L1 buffers).
DRAM_ACTIVATION_STAGES = tuple(sorted({0, 1, 3} | set(FP32_ACC_STAGES)))

# The FPN levels C3 and C4 feed, which only base has: ResNet layers 1 and 2.
_C3_C4_LEVELS = tuple(level for level, layer in enumerate(_VARIANT.out_indices) if layer in (1, 2))

# FPN levels whose convs keep their activations in DRAM because they do not fit in L1: base's C3
# is 6 x 116 x 200 x 512 bfloat16 (143 MB) and its C4 6 x 58 x 100 x 1024 bfloat16 (71 MB). C5
# fits.
DRAM_ACTIVATION_LEVELS = _C3_C4_LEVELS

# Width slices for the spatial convs of the DRAM layers and levels, lowered per conv to what
# its output width allows. The tightest is the strided 1x1 downsample that opens layer2: its
# 6 x 232 x 400 x 256 input is in DRAM, and each slice is read into L1 as bfloat16
# ROW_MAJOR for the halo, 3.25 MB per L1 bank in total against 576 KB free, so it needs at
# least 6 slices. 8 leaves margin over that. TTConv2D lowers the count per conv to its
# output width in tiles, as a width slice is whole tiles wide: 7 for this conv's 200-wide output.
DRAM_CONV_SLICES = 8

# Layers whose downsample and non-DCN conv2s are block sharded (layer3, layer4) and FPN levels
# whose output conv is (C3, C4). Height sharded, each core of tiny's layer4 conv2 would hold its
# whole 512-channel 3x3 weight, more than L1.
BLOCK_SHARDED_STAGES = (2, 3)
BLOCK_SHARDED_LEVELS = _C3_C4_LEVELS

# conv2d's act_block_h overrides: the stem conv's, the non-DCN 3x3 conv2's of the bottlenecks and
# the block-sharded FPN output convs'. None of these, nor the sharding above, is tuned for 928x1600.
STEM_ACT_BLOCK_H = 64
CONV2_ACT_BLOCK_H = 32
FPN_BLOCK_SHARDED_ACT_BLOCK_H = 128


def tt_resnet_kwargs():
    """TTResNet's memory and precision arguments for this configuration, which
    ``create_resnet_parameters`` records in the backbone's parameters. TTResNet's ``out_indices``
    defaults to all four layers; pass ``RESNET_KWARGS["out_indices"]`` for the variant's outputs."""
    return dict(
        dram_activation_stages=DRAM_ACTIVATION_STAGES,
        dram_conv_slices=DRAM_CONV_SLICES,
        block_sharded_stages=BLOCK_SHARDED_STAGES,
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
