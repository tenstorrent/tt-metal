# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os


def optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value, 0)


def tt_open_kwargs_from_env() -> dict:
    kwargs = {}
    l1_small_size = optional_env_int("AUDIOX_TT_L1_SMALL_SIZE")
    trace_region_size = optional_env_int("AUDIOX_TT_TRACE_REGION_SIZE")
    num_command_queues = optional_env_int("AUDIOX_TT_NUM_COMMAND_QUEUES")
    worker_l1_size = optional_env_int("AUDIOX_TT_WORKER_L1_SIZE")
    if l1_small_size is not None:
        kwargs["l1_small_size"] = l1_small_size
    if trace_region_size is not None:
        kwargs["trace_region_size"] = trace_region_size
    if num_command_queues is not None:
        kwargs["num_command_queues"] = num_command_queues
    if worker_l1_size is not None:
        kwargs["worker_l1_size"] = worker_l1_size
    return kwargs


def apply_tt_env_overrides(
    *,
    open_mode: str | None = None,
    local_mesh_width: int | None = None,
    l1_small_size: int | None = None,
    trace_region_size: int | None = None,
    num_command_queues: int | None = None,
    worker_l1_size: int | None = None,
    long_sequence_threshold: int | None = None,
    conv_transpose_input_chunk: int | None = None,
    conv1d_width_slices: int | None = None,
    conv1d_low_channel_width_slices: int | None = None,
    conv_transpose_height_slices: int | None = None,
    conv_transpose_long_threshold: int | None = None,
    conv_transpose_long_height_slices: int | None = None,
    conv_transpose_long_width_slices: int | None = None,
    conv_transpose_long_act_block_h: int | None = None,
    conv_transpose_stride2_act_block_h: int | None = None,
    conv_transpose_stride4_act_block_h: int | None = None,
    out_conv_stream_threshold: int | None = None,
    residual_stream_stride4_threshold: int | None = None,
    residual_stream_stride2_threshold: int | None = None,
) -> dict:
    overrides = {
        "AUDIOX_TT_OPEN_MODE": open_mode,
        "AUDIOX_TT_LOCAL_MESH_WIDTH": None if local_mesh_width is None else str(local_mesh_width),
        "AUDIOX_TT_L1_SMALL_SIZE": None if l1_small_size is None else str(l1_small_size),
        "AUDIOX_TT_TRACE_REGION_SIZE": None if trace_region_size is None else str(trace_region_size),
        "AUDIOX_TT_NUM_COMMAND_QUEUES": None if num_command_queues is None else str(num_command_queues),
        "AUDIOX_TT_WORKER_L1_SIZE": None if worker_l1_size is None else str(worker_l1_size),
        "AUDIOX_TT_LONG_SEQUENCE_THRESHOLD": None if long_sequence_threshold is None else str(long_sequence_threshold),
        "AUDIOX_TT_CONV_TRANSPOSE_INPUT_CHUNK": None
        if conv_transpose_input_chunk is None
        else str(conv_transpose_input_chunk),
        "AUDIOX_TT_CONV1D_WIDTH_SLICES": None if conv1d_width_slices is None else str(conv1d_width_slices),
        "AUDIOX_TT_CONV1D_LOW_CHANNEL_WIDTH_SLICES": None
        if conv1d_low_channel_width_slices is None
        else str(conv1d_low_channel_width_slices),
        "AUDIOX_TT_CONV_TRANSPOSE_HEIGHT_SLICES": None
        if conv_transpose_height_slices is None
        else str(conv_transpose_height_slices),
        "AUDIOX_TT_CONV_TRANSPOSE_LONG_THRESHOLD": None
        if conv_transpose_long_threshold is None
        else str(conv_transpose_long_threshold),
        "AUDIOX_TT_CONV_TRANSPOSE_LONG_HEIGHT_SLICES": None
        if conv_transpose_long_height_slices is None
        else str(conv_transpose_long_height_slices),
        "AUDIOX_TT_CONV_TRANSPOSE_LONG_WIDTH_SLICES": None
        if conv_transpose_long_width_slices is None
        else str(conv_transpose_long_width_slices),
        "AUDIOX_TT_CONV_TRANSPOSE_LONG_ACT_BLOCK_H": None
        if conv_transpose_long_act_block_h is None
        else str(conv_transpose_long_act_block_h),
        "AUDIOX_TT_CONV_TRANSPOSE_STRIDE2_ACT_BLOCK_H": None
        if conv_transpose_stride2_act_block_h is None
        else str(conv_transpose_stride2_act_block_h),
        "AUDIOX_TT_CONV_TRANSPOSE_STRIDE4_ACT_BLOCK_H": None
        if conv_transpose_stride4_act_block_h is None
        else str(conv_transpose_stride4_act_block_h),
        "AUDIOX_TT_OUT_CONV_STREAM_THRESHOLD": None
        if out_conv_stream_threshold is None
        else str(out_conv_stream_threshold),
        "AUDIOX_TT_RESIDUAL_STREAM_STRIDE4_THRESHOLD": None
        if residual_stream_stride4_threshold is None
        else str(residual_stream_stride4_threshold),
        "AUDIOX_TT_RESIDUAL_STREAM_STRIDE2_THRESHOLD": None
        if residual_stream_stride2_threshold is None
        else str(residual_stream_stride2_threshold),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        if value is None:
            continue
        os.environ[key] = value
    return previous


def restore_tt_env(previous: dict) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
