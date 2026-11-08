# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os


def residual_stream_long_threshold() -> int:
    return int(os.getenv("AUDIOX_TT_RESIDUAL_STREAM_LONG_THRESHOLD", "524288"))


def residual_stream_mid_channel_threshold() -> int:
    return int(os.getenv("AUDIOX_TT_RESIDUAL_STREAM_MID_CHANNEL_THRESHOLD", "110336"))


def residual_stream_stride4_threshold() -> int | None:
    value = os.getenv("AUDIOX_TT_RESIDUAL_STREAM_STRIDE4_THRESHOLD")
    return None if value is None else int(value)


def residual_stream_stride2_threshold() -> int | None:
    value = os.getenv("AUDIOX_TT_RESIDUAL_STREAM_STRIDE2_THRESHOLD")
    return None if value is None else int(value)


def conv_transpose_long_threshold() -> int:
    return int(os.getenv("AUDIOX_TT_CONV_TRANSPOSE_LONG_THRESHOLD", "1728"))


def conv_transpose_targeted_threshold() -> int:
    return int(os.getenv("AUDIOX_TT_CONV_TRANSPOSE_TARGETED_THRESHOLD", "24576"))


def conv_transpose_targeted_input_chunk() -> int:
    return int(os.getenv("AUDIOX_TT_CONV_TRANSPOSE_TARGETED_INPUT_CHUNK", "8192"))


def should_stream_decoder_block(input_length: int, stride: int, out_channels: int) -> bool:
    if out_channels > 256:
        return False
    if out_channels > 128:
        return stride == 4 and input_length * stride >= residual_stream_mid_channel_threshold()
    if stride == 4:
        threshold = residual_stream_stride4_threshold()
        if threshold is not None:
            return input_length * stride >= threshold
    if stride == 2:
        threshold = residual_stream_stride2_threshold()
        if threshold is not None:
            return input_length * stride >= threshold
    return input_length * stride >= residual_stream_long_threshold()


def should_use_long_transpose_profile(input_length: int, stride: int, out_channels: int) -> bool:
    if input_length >= conv_transpose_long_threshold():
        return True
    return stride == 4 and out_channels >= 256 and input_length >= conv_transpose_targeted_threshold()


def decoder_transpose_input_chunk_size(
    *, input_length: int, stride: int, out_channels: int, default_chunk_size: int
) -> int:
    if default_chunk_size <= 0:
        return 0
    if input_length >= conv_transpose_long_threshold():
        return default_chunk_size
    if stride == 4 and out_channels >= 256 and input_length >= conv_transpose_targeted_threshold():
        return min(default_chunk_size, conv_transpose_targeted_input_chunk())
    return 0
