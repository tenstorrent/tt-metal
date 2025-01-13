# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def random_complex_tensor(shape, real_range=(-100, 100), imag_range=(-100, 100)):
    torch.manual_seed(213919)
    real_part = (real_range[1] - real_range[0]) * torch.rand(shape) + real_range[0]
    imag_part = (imag_range[1] - imag_range[0]) * torch.rand(shape) + imag_range[0]
    return torch.complex(real_part, imag_part)


def convert_complex_to_torch_tensor(tt_dev):
    tt_dev_r = tt_dev.real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_dev_i = tt_dev.imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_to_torch = torch.complex(tt_dev_r, tt_dev_i)
    return tt_to_torch
