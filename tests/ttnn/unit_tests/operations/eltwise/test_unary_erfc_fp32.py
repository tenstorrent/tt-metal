# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def _run(values, device):
    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded.view(-1)[: values.numel()] = values

    tensor = ttnn.from_torch(
        padded,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
    )

    return ttnn.to_torch(ttnn.erfc(tensor)).view(-1)[: values.numel()].float()


def _fp32_ulp(got, ref):
    got_bits = got.view(torch.int32)
    ref_bits = ref.view(torch.int32)
    return (got_bits - ref_bits).abs().max().item()


def test_erfc_fp32_accuracy(device):
    values = torch.linspace(-5.0, 5.0, 1024, dtype=torch.float32)

    got = _run(values, device)
    ref = torch.special.erfc(values)

    max_ulp = _fp32_ulp(got, ref)

    assert max_ulp <= 128, f"FP32 erfc MaxULP={max_ulp} exceeds 128"
