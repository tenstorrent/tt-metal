# SPDX-License-Identifier: Apache-2.0
"""Qualified decode convolution over independent four-row history/token groups.

Selected row 4*i+3 consumes only lane i's three history rows and current token.
The native BF16 convolution and original taps remain unchanged. The persistent
zero predecessor is prepared by the B32 owner before trace capture and shared
by its compact decode views.
"""

import ttnn

WIDTHS = (512, 512, 1536)


def history_convolution(model, projected):
    batch, seq, width = projected.shape
    if (
        batch not in (8, 16, 32)
        or seq != 1
        or width != 2560
        or projected.dtype != ttnn.bfloat16
        or model.conv_decode_predecessor is None
    ):
        raise ValueError("History convolution requires prepared BF16 B8/16/32 decode")
    history = ttnn.to_layout(model.conv_state, ttnn.ROW_MAJOR_LAYOUT)
    current = ttnn.to_layout(projected, ttnn.ROW_MAJOR_LAYOUT)
    joined = ttnn.concat([history, current], dim=1)
    sequence = ttnn.reshape(joined, (1, 4 * batch, width))
    outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        sequence,
        model.conv_decode_predecessor,
        *model.conv_taps,
        *WIDTHS,
        actual_start=model.conv_zero,
        predecessor_carry=model.conv_decode_predecessor,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
        compute_kernel_config=model.compute,
    )
    result = []
    for output, width in zip(outputs, WIDTHS):
        rows = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
        selected = rows[:, 3 : 4 * batch : 4, :]
        result.append(ttnn.to_layout(ttnn.reshape(selected, (batch, 1, width)), ttnn.TILE_LAYOUT))
    # Observed during eager warmup/capture; replay has no host callback/query.
    model._decode_conv_history_geometry = {
        "batch": batch,
        "sequence": 1,
        "input_shape": list(sequence.shape),
        "history_shape": list(model.conv_decode_predecessor.shape),
        "widths": list(WIDTHS),
        "selected_rows": list(range(3, 4 * batch, 4)),
        "dtype": str(sequence.dtype),
        "layout": str(sequence.layout),
        "native_output_shapes": [list(value.shape) for value in outputs],
        "restored_output_shapes": [list(value.shape) for value in result],
    }
    return tuple(result)
