import pytest
import torch
import math
import ttnn


def generate_test_signal(length=256, batch_size=1):
    t = torch.linspace(0, 1, length)
    signal = torch.sin(2 * math.pi * 7 * t) + 0.5 * torch.sin(2 * math.pi * 23 * t)
    signal += 0.2 * torch.randn(length)
    if batch_size > 1:
        signal = signal.unsqueeze(0).expand(batch_size, -1)
    return signal


@pytest.mark.parametrize("wavelet", ["haar", "db1", "db2", "db3", "db4"])
@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("signal_length", [128, 256])
def test_dwt_1d_forward(signal_length, wavelet, level):
    signal = generate_test_signal(signal_length)
    cA, cD_list = ttnn.dwt(signal, wavelet=wavelet, level=level)
    assert cA.shape[-1] == signal_length // (2 ** level)
    assert len(cD_list) == level
    for i, cD in enumerate(cD_list):
        assert cD.shape[-1] == signal_length // (2 ** (i + 1))


@pytest.mark.parametrize("wavelet", ["haar", "db1", "db2"])
def test_dwt_idwt_roundtrip(wavelet):
    signal = generate_test_signal(128)
    cA, cD_list = ttnn.dwt(signal, wavelet=wavelet, level=1)
    reconstructed = ttnn.idwt(cA, cD_list, wavelet=wavelet)
    assert reconstructed.shape == signal.shape


@pytest.mark.parametrize("wavelet", ["haar", "db1"])
def test_dwt_wavelet_catalog(wavelet):
    signal = generate_test_signal(64)
    cA, cD_list = ttnn.dwt(signal, wavelet=wavelet, level=1)
    assert cA is not None
    assert len(cD_list) == 1


@pytest.mark.parametrize("wavelet", ["haar", "db1", "db2", "db3", "db4"])
def test_dwt_golden_match(wavelet):
    signal = generate_test_signal(64)
    cA_tt, cD_tt = ttnn.dwt(signal, wavelet=wavelet, level=1)
    coeffs = ttnn.operations.lwt.golden_dwt(signal, wavelet=wavelet, level=1)
    cA_ref, cD_ref = coeffs
    error = torch.abs(cA_tt.float() - cA_ref.float()).mean().item()
    assert error < 1e-5, f"cA error too high: {error}"


@pytest.mark.parametrize("wavelet", ["sym2", "sym3", "sym4", "coif1"])
def test_dwt_extended_wavelets(wavelet):
    signal = generate_test_signal(64)
    cA, cD_list = ttnn.dwt(signal, wavelet=wavelet, level=1)
    assert cA is not None
    assert len(cD_list) == 1


def test_dwt_batched_input():
    signal = generate_test_signal(64, batch_size=4)
    cA, cD_list = ttnn.dwt(signal, wavelet="haar", level=1)
    assert cA.shape[0] == 4
    assert cA.shape[-1] == 32


def test_invalid_wavelet():
    with pytest.raises(ValueError, match="Unsupported wavelet"):
        signal = generate_test_signal(64)
        ttnn.dwt(signal, wavelet="invalid_wavelet")


def test_dwt_2d():
    image = torch.randn(1, 1, 64, 64)
    cA, bands = ttnn.dwt_2d(image, wavelet="haar", level=1)
    assert cA.shape[-2:] == (32, 32)


def test_precision_compact_wavelets():
    signal = generate_test_signal(256)
    for wavelet in ["haar", "db1", "db2", "db3", "db4"]:
        cA, cD_list = ttnn.dwt(signal, wavelet=wavelet, level=1)
        coeffs = ttnn.operations.lwt.golden_dwt(signal, wavelet=wavelet, level=1)
        cA_ref = coeffs[0]
        diff = cA.float() - cA_ref.float()
        l2_error = torch.norm(diff) / torch.norm(cA_ref.float())
        assert l2_error < 1e-6, f"L2 error too high for {wavelet}: {l2_error}"
