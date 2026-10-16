# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight numerical correctness and independent-user packing on device."""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen38_27b_t3k.tt.decode_conv import make_actual_start, packed_decode_conv, row_selector
from models.demos.qwen38_27b_t3k.tt.model import Checkpoint, checkpoint_path


def first_tp_shard_conv_taps(widths, tp):
    """Load one TP rank's Q/K/V convolution taps from the pinned checkpoint."""
    conv = Checkpoint(checkpoint_path()).layer(0)["linear_attn.conv1d.weight"]
    projections = conv.split([tp * width for width in widths])
    local = torch.cat([projection.chunk(tp, dim=0)[0] for projection in projections])
    return tuple(local[:, 0, tap].reshape(1, 1, -1).bfloat16() for tap in range(4))


def torch_conv_reference(row, history, taps, widths):
    """Independent causal Conv1D plus SiLU reference for the retained decode row."""
    window = torch.cat([history, row[:, :1]], dim=1).float()
    convolved = sum(window[:, tap : tap + 1] * taps[tap].float() for tap in range(4))
    return F.silu(convolved).split(widths, dim=-1)


# T3K is the only supported mesh and runs at TP=8; TP=4 is the QB2 shard. The one-hot read's
# N changes with the width, so the shipped shard has to be the one under test.
@pytest.mark.parametrize("tp", [8, 4])
@pytest.mark.parametrize("batch", [8, 16, 24, 32])
@pytest.mark.parametrize("seed", [0, 17])
def test_packed_decode_conv_matches_independent_users(device, batch, seed, tp):
    torch.manual_seed(seed)
    widths = (2048 // tp, 2048 // tp, 6144 // tp)
    channels = sum(widths)

    def upload(tensor, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(tensor.bfloat16(), device=device, layout=layout)

    row_host = torch.randn(batch, 32, channels).bfloat16()
    history_host = torch.randn(batch, 3, channels).bfloat16()
    taps_host = first_tp_shard_conv_taps(widths, tp)
    row = upload(row_host)
    history = upload(history_host)
    taps = [upload(tap, ttnn.TILE_LAYOUT) for tap in taps_host]
    actual_start = make_actual_start(device)
    selector = row_selector(device, batch)
    reference = [
        ttnn.experimental.kda.qkv_causal_conv1d_silu(
            row[i : i + 1],
            history[i : i + 1],
            *taps,
            *widths,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
            actual_start=actual_start,
            predecessor_carry=history[i : i + 1],
        )
        for i in range(batch)
    ]
    result = packed_decode_conv(row, history, taps, widths, actual_start, selector)
    torch_reference = torch_conv_reference(row_host, history_host, taps_host, widths)
    for name, component, output in zip(("q", "k", "v"), range(3), result):
        expected = torch.cat([ttnn.to_torch(parts[component])[:, :1] for parts in reference])
        actual = ttnn.to_torch(output)
        assert torch.equal(actual[:, :1], expected)
        assert torch.count_nonzero(actual[:, 1:]) == 0
        passing, pcc = comp_pcc(torch_reference[component].reshape(-1), actual[:, :1].float().reshape(-1), 0.999)
        assert passing, f"real-weight {name} convolution PCC below 0.999: {pcc}"

    # Changing one user's state must not leak into any other retained row.
    changed_history = ttnn.to_torch(history).clone()
    changed_history[batch // 2] *= -3
    changed = packed_decode_conv(row, upload(changed_history), taps, widths, actual_start, selector)
    keep = [i for i in range(batch) if i != batch // 2]
    for original, modified in zip(result, changed):
        assert torch.equal(ttnn.to_torch(original)[keep], ttnn.to_torch(modified)[keep])
