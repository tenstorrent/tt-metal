# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


class Detr3dArgs:
    # Model details
    model_name = "3detr"
    enc_type = "masked"  # detr3d-m variant
    enc_nlayers = 3
    enc_dim = 256
    enc_ffn_dim = 128
    enc_dropout = 0.1
    enc_nhead = 4
    enc_pos_embed = None
    enc_activation = "relu"

    dec_nlayers = 8
    dec_dim = 256
    dec_ffn_dim = 256
    dec_dropout = 0.1
    dec_nhead = 4

    mlp_dropout = 0.3
    nsemcls = -1
    preenc_npoints = 2048
    nqueries = 128  # detr3d-m sunrgbd variant
    use_color = False
