# SPDX-FileCopyrightText: © 2023-24 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from math import pi
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_rand_complex


def make_complex(input_shapes, low, high, device):
    torch_a = gen_rand_complex(size=input_shapes, low=low, high=high).to(torch.bfloat16)
    torch_b = gen_rand_complex(size=input_shapes, low=low, high=high).to(torch.bfloat16)

    torch_a, torch_b = torch_a + 1j * torch_b, torch_b + 1j * torch_a

    a_r = ttnn.to_device(ttnn.from_torch(torch_a.real, dtype=ttnn.bfloat16), device)
    a_i = ttnn.to_device(ttnn.from_torch(torch_a.imag, dtype=ttnn.bfloat16), device)
    a = ttnn.ComplexTensor(a_r, a_i)

    b_r = ttnn.to_device(ttnn.from_torch(torch_b.real, dtype=ttnn.bfloat16), device)
    b_i = ttnn.to_device(ttnn.from_torch(torch_b.imag, dtype=ttnn.bfloat16), device)
    b = ttnn.ComplexTensor(b_r, b_i)

    return a, b, torch_a, torch_b


def run_test_with(input_shapes, ttnn_complex_fn, ref_fn, low, high, unary=False):
    device_id = 0
    device = ttnn.open(device_id)

    torch.manual_seed(0)

    a, b, torch_a, torch_b = make_complex(input_shapes, low, high, device)
    if unary:
        output = ttnn_complex_fn(a)
    else:
        output = ttnn_complex_fn(a, b)

    if isinstance(output, ttnn.ComplexTensor):
        output_r = ttnn.to_layout(output.real, ttnn.ROW_MAJOR_LAYOUT)
        output_r = ttnn.to_torch(output_r)

        output_i = ttnn.to_layout(output.imag, ttnn.ROW_MAJOR_LAYOUT)
        output_i = ttnn.to_torch(output_i)
        output = output_r + 1j * output_i

        print(f"shape: {output_r.shape}")
        print(f"dtype: {output_r.dtype}")
        print(f"layout: {output_r.layout}")

    else:
        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.to_torch(output)

        print(f"shape: {output.shape}")
        print(f"dtype: {output.dtype}")
        print(f"layout: {output.layout}")

    if unary:
        assert torch.isclose(output, ref_fn(torch_a), 1e-1).all()
    else:
        assert torch.isclose(output, ref_fn(torch_a, torch_b), 1e-1).all()

    ttnn.close(device)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
class TestBinaryComplex:
    def test_ttnn_complex_add(self, input_shapes):
        run_test_with(input_shapes, ttnn.complex_add, lambda _a, _b: _a + _b, low=-100, high=100)

    def test_ttnn_complex_sub(self, input_shapes):
        run_test_with(input_shapes, ttnn.complex_sub, lambda _a, _b: _a - _b, low=-100, high=100)

    def test_ttnn_complex_mul(self, input_shapes):
        run_test_with(input_shapes, ttnn.complex_mul, lambda _a, _b: _a * _b, low=-100, high=100)

    def test_ttnn_complex_div(self, input_shapes):
        run_test_with(input_shapes, ttnn.complex_div, lambda _a, _b: _a / _b, low=-100, high=100)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
class TestUnaryComplex:
    def test_ttnn_complex_abs(self, input_shapes):
        run_test_with(
            input_shapes, ttnn.complex_abs, lambda _a: _a.abs().to(torch.bfloat16), low=-100, high=100, unary=True
        )

    def test_ttnn_complex_real(self, input_shapes):
        run_test_with(
            input_shapes, ttnn.complex_real, lambda _a: _a.real.to(torch.bfloat16), low=-100, high=100, unary=True
        )

    def test_ttnn_complex_imag(self, input_shapes):
        run_test_with(
            input_shapes, ttnn.complex_imag, lambda _a: _a.imag.to(torch.bfloat16), low=-100, high=100, unary=True
        )

    def test_ttnn_complex_conj(self, input_shapes):
        run_test_with(
            input_shapes, ttnn.complex_conj, lambda _a: _a.conj().to(torch.cfloat), low=-100, high=100, unary=True
        )

    def test_ttnn_complex_recip(self, input_shapes):
        run_test_with(
            input_shapes, ttnn.complex_recip, lambda _a: (1.0 / _a).to(torch.cfloat), low=-100, high=100, unary=True
        )

    def test_ttnn_complex_angle(self, input_shapes):
        run_test_with(
            input_shapes, ttnn.complex_angle, lambda _a: _a.angle().to(torch.bfloat16), low=-100, high=100, unary=True
        )

    def test_ttnn_complex_polar(self, input_shapes):
        run_test_with(
            input_shapes,
            ttnn.complex_polar,
            lambda _a: torch.polar(_a.real, _a.imag).to(torch.cfloat),
            low=0,
            high=2 * pi,
            unary=True,
        )

    def test_ttnn_complex_is_real(self, input_shapes):
        run_test_with(
            input_shapes,
            ttnn.complex_is_real,
            lambda _a: (_a.imag == 0).to(torch.bfloat16),
            low=-100,
            high=100,
            unary=True,
        )

    def test_ttnn_complex_is_imag(self, input_shapes):
        run_test_with(
            input_shapes,
            ttnn.complex_is_imag,
            lambda _a: (_a.real == 0).to(torch.bfloat16),
            low=-100,
            high=100,
            unary=True,
        )
