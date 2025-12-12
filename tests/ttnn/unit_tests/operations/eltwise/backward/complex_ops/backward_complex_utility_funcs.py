# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class Complex:
    def __init__(self, input_shape: torch.Size = None, re=None, im=None):
        if input_shape:
            val = 1.0 + torch.arange(0, input_shape.numel()).reshape(input_shape).bfloat16()
            self._cplx = val[:, :, :, : input_shape[-1] // 2] + val[:, :, :, input_shape[-1] // 2 :] * 1j
        else:
            self._cplx = re + im * 1j

    def reset(self, val: torch.Tensor):
        self._cplx = val

    def is_imag(self):
        return self.real == 0.0

    def is_real(self):
        return self.imag == 0.0

    @property
    def angle(self):
        return torch.angle(self._cplx)

    @property
    def real(self):
        return self._cplx.real

    @property
    def imag(self):
        return self._cplx.imag

    @property
    def metal(self):
        return torch.cat([self.real, self.imag], -1)

    ## operations
    def abs(self):
        return (self.real**2 + self.imag**2).sqrt()

    def conj(self) -> "Complex":
        self._cplx = self._cplx.conj()
        return self

    def recip(self):
        self._cplx = 1.0 / self._cplx
        return self

    def add(self, that: "Complex"):
        self._cplx += that._cplx
        return self

    def sub(self, that: "Complex"):
        self._cplx -= that._cplx
        return self

    def __mul__(self, scale):
        self._cplx *= scale
        return self

    def mul(self, that: "Complex"):
        self._cplx *= that._cplx
        return self

    def div(self, that: "Complex"):
        self._cplx /= that._cplx
        return self


def random_complex_tensor(shape, real_range=(-100, 100), imag_range=(-100, 100)):
    torch.manual_seed(213919)
    real_part = (real_range[1] - real_range[0]) * torch.rand(shape) + real_range[0]
    imag_part = (imag_range[1] - imag_range[0]) * torch.rand(shape) + imag_range[0]
    return torch.complex(real_part, imag_part)


def convert_to_torch_tensor(tt_dev):
    for i in range(len(tt_dev)):
        tt_dev_r = tt_dev[i].real.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        tt_dev_i = tt_dev[i].imag.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        tt_dev[i] = Complex(re=tt_dev_r, im=tt_dev_i).metal
    return tt_dev
