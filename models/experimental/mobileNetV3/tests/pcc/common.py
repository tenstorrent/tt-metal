from models.experimental.mobileNetV3.tt.ttnn_invertedResidual import InvertedResidualConfig
from functools import partial

reduce_divider = 1
dilation = 1

bneck_conf = partial(InvertedResidualConfig, width_mult=1.0)
adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=1.0)
# TODO explain all the parameters below

# (1, 16, 112, 112, 1),
# (1, 16, 56, 56, 2),
# (1, 24, 28, 28, 3),
# (1, 24, 28, 28, 4),
# (1, 40, 14, 14, 5),
# (1, 40, 14, 14, 6),
# (1, 40, 14, 14, 7),
# (1, 48, 14, 14, 8),
# (1, 48, 14, 14, 9),
# (1, 96, 7, 7, 10),
# (1, 96, 7, 7, 11),
inverted_residual_setting = [
    bneck_conf(16, 3, 16, 16, True, "RE", 2, 1, input_height=112, input_width=112),
    bneck_conf(16, 3, 72, 24, False, "RE", 2, 1, input_height=56, input_width=56),
    bneck_conf(24, 3, 88, 24, False, "RE", 1, 1, input_height=28, input_width=28),
    bneck_conf(24, 5, 96, 40, True, "HS", 2, 1, input_height=28, input_width=28),
    bneck_conf(40, 5, 240, 40, True, "HS", 1, 1, input_height=14, input_width=14),
    bneck_conf(40, 5, 240, 40, True, "HS", 1, 1, input_height=14, input_width=14),
    bneck_conf(40, 5, 120, 48, True, "HS", 1, 1, input_height=14, input_width=14),
    bneck_conf(48, 5, 144, 48, True, "HS", 1, 1, input_height=14, input_width=14),
    bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation, input_height=14, input_width=14),
    bneck_conf(
        96 // reduce_divider,
        5,
        576 // reduce_divider,
        96 // reduce_divider,
        True,
        "HS",
        1,
        dilation,
        input_height=7,
        input_width=7,
    ),
    bneck_conf(
        96 // reduce_divider,
        5,
        576 // reduce_divider,
        96 // reduce_divider,
        True,
        "HS",
        1,
        dilation,
        input_height=7,
        input_width=7,
    ),
]
# inverted_residual_setting = [
#     bneck_conf(16, 3, 16, 16, True, "RE", 2, 1,),
#     bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),
#     bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
#     bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),
#     bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
#     bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
#     bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
#     bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
#     bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),
#     bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
#     bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
# ]

last_channel = adjust_channels(1024 // reduce_divider)
