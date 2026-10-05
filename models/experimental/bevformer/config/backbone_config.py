# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Memory and precision configuration of the TTNN backbone and FPN for BEVFormer-base: 6 cameras
at 1600x928, ResNet101-DCN and its four-level FPN.

Layer indices count the ResNet layers from 0 (layer1); level indices the FPN levels from 0 (C3).
"""

# Layers whose convs accumulate in an fp32 destination register (layer3 and layer4, the DCN
# layers; a DCN conv2 always does). Measured with the BEVFormer-base checkpoint
# (bevformer_r101_dcn_24ep.pth, img_backbone): with bfloat16 accumulation, the error grows
# through these layers' 26 blocks until C5 falls below PCC 0.99. The dummy-weight tests do
# not show it; their weights lack the trained backbone's outlier channels.
FP32_ACC_STAGES = (2, 3)

# Layers whose activations are kept in DRAM because their convs do not fit in L1. layer1 and
# layer2 work on 6 x 232 x 400 x 256 tensors (285 MB in bfloat16), and layer4's 2048-channel
# 1x1 convs overflow L1 when sharded. The fp32 accumulation layers join them: their larger
# destination buffers overflow L1 next to sharded activations.
DRAM_ACTIVATION_STAGES = tuple(sorted({0, 1, 3} | set(FP32_ACC_STAGES)))

# FPN levels whose convs keep their activations in DRAM because they do not fit in L1:
# C3 is 6 x 116 x 200 x 512 bfloat16 (143 MB) and C4 is 6 x 58 x 100 x 1024 bfloat16 (71 MB).
DRAM_ACTIVATION_LEVELS = (0, 1)

# Width slices for the spatial convs of the DRAM layers and levels, lowered per conv to what
# its output width allows. The tightest is the strided 1x1 downsample that opens layer2: its
# 6 x 232 x 400 x 256 input is in DRAM, and each slice is read into L1 as bfloat16
# ROW_MAJOR for the halo, 3.25 MB per L1 bank in total against 576 KB free, so it needs at
# least 6 slices. 8 leaves margin over that; this conv's 200-wide output caps it at 7.
DRAM_CONV_SLICES = 8

# Layers whose downsample conv is block sharded (layer3, layer4) and FPN levels whose output
# conv is (C3, C4). Both come from the UniAD port and are not re-tuned for 928x1600.
BLOCK_SHARDED_DOWNSAMPLE_STAGES = (2, 3)
BLOCK_SHARDED_LEVELS = (0, 1)
