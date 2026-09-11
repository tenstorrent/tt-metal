# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API coverage for bounded, nonaligned FP32 Bluestein PRE/POST."""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.usefixtures("silicon_arch_name", "silicon_arch_wormhole_b0")


def _upload(value, device):
    return ttnn.from_torch(value, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def _download(pair):
    real, imag = pair
    assert real.dtype == imag.dtype == ttnn.float32
    assert real.layout == imag.layout == ttnn.ROW_MAJOR_LAYOUT
    return torch.complex(ttnn.to_torch(real).double(), ttnn.to_torch(imag).double())


def _check_result(got, reference, tolerance=1e-3):
    assert got.shape == reference.shape
    assert torch.isfinite(got).all()
    scale = torch.linalg.vector_norm(reference)
    if scale == 0:
        assert torch.count_nonzero(got) == 0
        return
    error = got - reference
    assert torch.linalg.vector_norm(error) <= tolerance * scale
    assert error.abs().max() <= tolerance * reference.abs().max()


def _run(device, real, imag, inverse):
    tt_real = _upload(real, device)
    tt_imag = _upload(imag, device) if imag is not None else None
    if inverse:
        result = ttnn.experimental.ifft(tt_real, tt_imag, precision="precise")
    elif tt_imag is None:
        result = ttnn.experimental.fft(tt_real, precision="precise")
    else:
        result = ttnn.experimental.fft(tt_real, tt_imag, precision="precise")
    source = real.double().to(torch.complex128)
    if imag is not None:
        source += 1j * imag.double()
    reference = torch.fft.ifft(source) if inverse else torch.fft.fft(source)
    _check_result(_download(result), reference)
    return (tt_real, tt_imag, result)


@pytest.mark.parametrize("inverse", [False, True], ids=["fft", "ifft"])
def test_bluestein_inner_complex_fft(device, inverse):
    """Qualify the actual complex inner transform independently of PRE/POST."""
    generator = torch.Generator().manual_seed(262144)
    real = torch.randn((1, 262144), generator=generator)
    imag = torch.randn((1, 262144), generator=generator)
    _run(device, real, imag, inverse)


@pytest.mark.parametrize("n", [16385, 65532, 65533, 65535, 65537, 65539])
@pytest.mark.parametrize("inverse", [False, True], ids=["fft", "ifft"])
def test_bluestein_streaming_complex(device, n, inverse):
    # 65532/65533 cross the planner's eight-padding-element boundary.
    generator = torch.Generator().manual_seed(n)
    real = torch.randn((1, n), generator=generator)
    imag = torch.randn((1, n), generator=generator)
    _run(device, real, imag, inverse)


@pytest.mark.parametrize("n", [16385, 65537])
@pytest.mark.parametrize("batched_shape", [False, True], ids=["vector", "single_batch"])
def test_bluestein_streaming_real(device, n, batched_shape):
    shape = (1, n) if batched_shape else (n,)
    real = torch.randn(shape, generator=torch.Generator().manual_seed(n))
    _run(device, real, None, False)


@pytest.mark.parametrize("pattern", ["zero", "constant", "first", "last"])
@pytest.mark.parametrize("inverse", [False, True], ids=["fft", "ifft"])
def test_bluestein_streaming_tail(device, pattern, inverse):
    real = torch.zeros((1, 65537))
    imag = torch.zeros_like(real)
    if pattern == "constant":
        real.fill_(0.75)
        imag.fill_(-0.25)
    elif pattern != "zero":
        index = 0 if pattern == "first" else -1
        real[0, index] = 1.0
        imag[0, index] = -0.5
    _run(device, real, imag, inverse)


def test_bluestein_streaming_round_trip(device):
    generator = torch.Generator().manual_seed(37)
    real = torch.randn((1, 65537), generator=generator)
    imag = torch.randn((1, 65537), generator=generator)
    spectrum = ttnn.experimental.fft(_upload(real, device), _upload(imag, device), precision="precise")
    recovered = ttnn.experimental.ifft(*spectrum, precision="precise")
    _check_result(_download(recovered), torch.complex(real.double(), imag.double()))


def test_bluestein_streaming_program_cache(device):
    """Cache hits must rebind new tensors and preserve real/complex and direction keys."""
    retained = []
    generator = torch.Generator().manual_seed(123)
    # Warm complex PRE with aliased halves, then use distinct halves below.
    # A cache implementation that infers roles solely from pointer identity
    # at the miss would incorrectly reuse the real half for both arguments.
    for inverse in [False, True]:
        real = torch.randn((1, 65537), generator=generator)
        shared = _upload(real, device)
        transform = ttnn.experimental.ifft if inverse else ttnn.experimental.fft
        result = transform(shared, shared, precision="precise")
        source = torch.complex(real.double(), real.double())
        reference = torch.fft.ifft(source) if inverse else torch.fft.fft(source)
        _check_result(_download(result), reference)
        retained.append((shared, result))
    entries = None
    for trial in range(3):
        for inverse, complex_input in [(False, False), (False, True), (True, True)]:
            real = torch.randn((1, 65537), generator=generator)
            imag = torch.randn(real.shape, generator=generator) if complex_input else None
            # Retain old allocations so a later call cannot accidentally pass by
            # receiving the same addresses as the cache-miss call.
            retained.append(_run(device, real, imag, inverse))
        current = device.num_program_cache_entries()
        assert current > 0
        if trial == 0:
            entries = current
        else:
            assert current == entries
