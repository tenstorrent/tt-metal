# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Architecture-dependent choices of the Quasar fork of Panoptic-DeepLab."""

import ttnn


def is_quasar(device) -> bool:
    return device.arch() == ttnn.device.Arch.QUASAR


def block_float_dtype(device):
    """The dtype the model uses for its bfloat8_b tensors on ``device``.

    Quasar has no block-float formats (``is_supported_quasar`` in tt_backend_api_types.cpp
    rejects Bfp8_b / Bfp4_b), so every bfloat8_b weight and activation runs as bfloat16
    there for now. Wormhole and Blackhole keep bfloat8_b.
    """
    return ttnn.bfloat16 if is_quasar(device) else ttnn.bfloat8_b
