# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os


def residual_stream_long_threshold() -> int:
    return int(os.getenv("AUDIOX_TT_RESIDUAL_STREAM_LONG_THRESHOLD", "524288"))


def should_stream_decoder_block(input_length: int, stride: int, out_channels: int) -> bool:
    if out_channels > 128:
        return False
    return input_length * stride >= residual_stream_long_threshold()
