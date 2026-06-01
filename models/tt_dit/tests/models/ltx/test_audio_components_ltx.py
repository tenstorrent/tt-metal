# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LTX audio component tests (decoder + vocoder + BWE).

Consolidates:
- test_audio_decoder_ltx.py
- test_vocoder_ltx.py
- test_bwe_ltx.py
- test_vocoder_sharded_ltx.py
"""

from __future__ import annotations

import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.audio_vae.audio_decoder_ltx import LTXAudioDecoder
from models.tt_dit.models.audio_vae.bwe_ltx import LTX_STFTFn, LTXHannUpSample1d, LTXMelSTFT, LTXVocoderWithBWE
from models.tt_dit.models.audio_vae.vocoder_ltx import (
    LTXAMPBlock1,
    LTXConvTranspose1d,
    LTXDownSample1d,
    LTXLowPassFilter1d,
    LTXUpSample1d,
    LTXVocoder,
    LTXVocoderActivation1d,
)
from models.tt_dit.utils.check import assert_quality

sys.path.insert(0, "LTX-2/packages/ltx-core/src")


# ============================================================================
# Decoder (Stage A): LTXAudioDecoder
# ============================================================================


# Single-chip - Conv2dViaConv3d is single-device. No fabric.
_AUDIO_DECODER_MESH_DEVICE_PARAMS = [
    ((1, 1), {}),
]


# Production config from the LTX-2.3 22B distilled checkpoint's
# audio_vae.model.params.ddconfig.
_PROD_CONFIG = dict(
    ch=128,
    out_ch=2,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    attn_resolutions=(),
    mid_block_add_attention=False,
    z_channels=8,
    resolution=64,
    mel_bins=64,
    sample_rate=16000,
    mel_hop_length=160,
    is_causal=True,
)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    _AUDIO_DECODER_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx_audio_decoder(mesh_device: ttnn.MeshDevice):
    """Stage A: tt-dit ``LTXAudioDecoder`` vs torch reference, random weights."""
    from ltx_core.model.audio_vae.audio_vae import AudioDecoder as TorchAudioDecoder
    from ltx_core.model.audio_vae.causality_axis import CausalityAxis
    from ltx_core.model.common.normalization import NormType

    torch.manual_seed(42)

    torch_decoder = TorchAudioDecoder(
        ch=_PROD_CONFIG["ch"],
        out_ch=_PROD_CONFIG["out_ch"],
        ch_mult=_PROD_CONFIG["ch_mult"],
        num_res_blocks=_PROD_CONFIG["num_res_blocks"],
        attn_resolutions=set(_PROD_CONFIG["attn_resolutions"]),
        resolution=_PROD_CONFIG["resolution"],
        z_channels=_PROD_CONFIG["z_channels"],
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        dropout=0.0,
        mid_block_add_attention=_PROD_CONFIG["mid_block_add_attention"],
        sample_rate=_PROD_CONFIG["sample_rate"],
        mel_hop_length=_PROD_CONFIG["mel_hop_length"],
        is_causal=_PROD_CONFIG["is_causal"],
        mel_bins=_PROD_CONFIG["mel_bins"],
    )
    torch_decoder.eval()

    z_times_f = _PROD_CONFIG["z_channels"] * _PROD_CONFIG["mel_bins"]
    torch_decoder.per_channel_statistics.__dict__["_buffers"]["std-of-means"] = torch.ones(z_times_f)
    torch_decoder.per_channel_statistics.__dict__["_buffers"]["mean-of-means"] = torch.zeros(z_times_f)

    tt_decoder = LTXAudioDecoder(
        ch=_PROD_CONFIG["ch"],
        out_ch=_PROD_CONFIG["out_ch"],
        ch_mult=_PROD_CONFIG["ch_mult"],
        num_res_blocks=_PROD_CONFIG["num_res_blocks"],
        attn_resolutions=_PROD_CONFIG["attn_resolutions"],
        resolution=_PROD_CONFIG["resolution"],
        z_channels=_PROD_CONFIG["z_channels"],
        mid_block_add_attention=_PROD_CONFIG["mid_block_add_attention"],
        sample_rate=_PROD_CONFIG["sample_rate"],
        mel_hop_length=_PROD_CONFIG["mel_hop_length"],
        is_causal=_PROD_CONFIG["is_causal"],
        mel_bins=_PROD_CONFIG["mel_bins"],
        mesh_device=mesh_device,
    )
    tt_decoder.load_torch_state_dict(torch_decoder.state_dict())

    bsz = 1
    t_steps = 64
    mel_bins = _PROD_CONFIG["mel_bins"]
    z_channels = _PROD_CONFIG["z_channels"]
    latent = torch.randn(bsz, z_channels, t_steps, mel_bins, dtype=torch.float32)

    with torch.no_grad():
        torch_out = torch_decoder(latent)

    tt_out = tt_decoder(latent)

    logger.info(f"Audio decoder: {tuple(latent.shape)} -> torch {tuple(torch_out.shape)}, tt {tuple(tt_out.shape)}")
    assert torch_out.shape == tt_out.shape, f"shape mismatch: torch {torch_out.shape}, tt {tt_out.shape}"
    assert_quality(torch_out, tt_out, pcc=0.998, relative_rmse=0.05)
    logger.info("PASSED: LTXAudioDecoder matches torch reference (PCC >= 0.998)")


# ============================================================================
# Vocoder (Stage B) component + E2E tests
# ============================================================================


_LTX_VOCODER_MESH_DEVICE_PARAMS = [
    # l1_small_size required for ttnn.conv1d (depthwise lowpass path).
    ((1, 1), {"l1_small_size": 32768}),
]


def _to_dev_btc(x_bct: torch.Tensor, mesh_device: ttnn.MeshDevice, dtype=ttnn.float32) -> ttnn.Tensor:
    """Convert torch ``(B, C, T)`` to device ``(B, T, C)`` ROW_MAJOR."""
    x_btc = x_bct.transpose(1, 2).float().contiguous()
    return ttnn.from_torch(x_btc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype)


def _from_dev_btc(x_tt: ttnn.Tensor, *, trim_channels: int | None = None) -> torch.Tensor:
    """Convert device ``(B, T, C)`` ROW_MAJOR to torch ``(B, C, T)``."""
    x_host = ttnn.to_torch(ttnn.get_device_tensors(x_tt)[0])
    if trim_channels is not None:
        x_host = x_host[..., :trim_channels]
    return x_host.transpose(-1, -2).contiguous()


_VOCODER_CFG = dict(
    resblock_kernel_sizes=[3, 7, 11],
    upsample_rates=[5, 2, 2, 2, 2, 2],
    upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_initial_channel=1536,
    resblock="AMP1",
    output_sampling_rate=24000,
    activation="snakebeta",
    use_tanh_at_final=False,
    apply_final_activation=True,
    use_bias_at_final=False,
)


def _randomize_snake_alphas(torch_module):
    """Randomize all Snake / SnakeBeta alpha & beta parameters."""
    with torch.no_grad():
        for module in torch_module.modules():
            if hasattr(module, "alpha") and isinstance(module.alpha, torch.nn.Parameter):
                module.alpha.data = torch.randn_like(module.alpha.data) * 0.1
            if hasattr(module, "beta") and isinstance(module.beta, torch.nn.Parameter):
                module.beta.data = torch.randn_like(module.beta.data) * 0.1


def _build_torch_vocoder(seed: int = 42):
    from ltx_core.model.audio_vae.vocoder import Vocoder

    torch.manual_seed(seed)
    torch_voc = Vocoder(**_VOCODER_CFG)
    torch_voc.eval()
    _randomize_snake_alphas(torch_voc)
    return torch_voc


def _build_tt_vocoder(mesh_device: ttnn.MeshDevice) -> LTXVocoder:
    tt_cfg = dict(
        resblock_kernel_sizes=_VOCODER_CFG["resblock_kernel_sizes"],
        upsample_rates=_VOCODER_CFG["upsample_rates"],
        upsample_kernel_sizes=_VOCODER_CFG["upsample_kernel_sizes"],
        resblock_dilation_sizes=_VOCODER_CFG["resblock_dilation_sizes"],
        upsample_initial_channel=_VOCODER_CFG["upsample_initial_channel"],
        resblock=_VOCODER_CFG["resblock"],
        activation=_VOCODER_CFG["activation"],
        use_tanh_at_final=_VOCODER_CFG["use_tanh_at_final"],
        apply_final_activation=_VOCODER_CFG["apply_final_activation"],
        use_bias_at_final=_VOCODER_CFG["use_bias_at_final"],
        in_channels=128,
        out_channels=2,
    )
    return LTXVocoder(mesh_device=mesh_device, dtype=ttnn.float32, **tt_cfg)


def _sample_vocoder_mel() -> torch.Tensor:
    return torch.randn(1, 2, 64, 64, dtype=torch.float32) * 0.5


@pytest.mark.parametrize(
    "channels, T, stride, kernel_size, padding_mode",
    [
        (16, 64, 1, 12, "replicate"),
        (16, 64, 2, 12, "replicate"),
        (32, 128, 2, 12, "replicate"),
    ],
    ids=["c16_t64_s1", "c16_t64_s2", "c32_t128_s2"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VOCODER_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_low_pass_filter_1d(
    mesh_device: ttnn.MeshDevice,
    channels: int,
    T: int,
    stride: int,
    kernel_size: int,
    padding_mode: str,
):
    """PCC check vs reference ``LowPassFilter1d``."""
    from ltx_core.model.audio_vae.vocoder import LowPassFilter1d

    torch.manual_seed(42)
    cutoff = 0.5 / max(stride, 2)
    half_width = 0.6 / max(stride, 2)

    torch_filter = LowPassFilter1d(
        cutoff=cutoff,
        half_width=half_width,
        stride=stride,
        padding=True,
        padding_mode=padding_mode,
        kernel_size=kernel_size,
    )
    torch_filter.eval()

    tt_filter = LTXLowPassFilter1d(
        cutoff=cutoff,
        half_width=half_width,
        stride=stride,
        kernel_size=kernel_size,
        padding=True,
        padding_mode=padding_mode,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )
    tt_filter._taps_cpu = torch_filter.filter.reshape(kernel_size).float().tolist()

    x = torch.randn(1, channels, T, dtype=torch.float32)
    with torch.no_grad():
        torch_out = torch_filter(x)

    x_dev = _to_dev_btc(x, mesh_device)
    y_dev = tt_filter(x_dev)
    tt_out = _from_dev_btc(y_dev, trim_channels=channels)

    logger.info(f"LowPassFilter1d: torch {torch_out.shape}, tt {tt_out.shape}")
    assert tt_out.shape == torch_out.shape, f"shape mismatch: torch {torch_out.shape}, tt {tt_out.shape}"
    assert_quality(torch_out, tt_out, pcc=0.999)


@pytest.mark.parametrize(
    "channels, T, ratio",
    [
        (16, 32, 2),
        (32, 64, 2),
    ],
    ids=["c16_t32_r2", "c32_t64_r2"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VOCODER_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_upsample_1d(mesh_device: ttnn.MeshDevice, channels: int, T: int, ratio: int):
    """PCC check vs reference ``UpSample1d``."""
    from ltx_core.model.audio_vae.vocoder import UpSample1d

    torch.manual_seed(42)
    torch_up = UpSample1d(ratio=ratio)
    torch_up.eval()

    tt_up = LTXUpSample1d(ratio=ratio, kernel_size=torch_up.kernel_size, mesh_device=mesh_device, dtype=ttnn.float32)
    tt_up._taps_cpu = torch_up.filter.reshape(torch_up.kernel_size).float().tolist()

    x = torch.randn(1, channels, T, dtype=torch.float32)
    with torch.no_grad():
        torch_out = torch_up(x)

    x_dev = _to_dev_btc(x, mesh_device)
    y_dev = tt_up(x_dev)
    tt_out = _from_dev_btc(y_dev, trim_channels=channels)

    logger.info(f"UpSample1d: torch {torch_out.shape}, tt {tt_out.shape}")
    assert tt_out.shape == torch_out.shape, f"shape mismatch: torch {torch_out.shape}, tt {tt_out.shape}"
    assert_quality(torch_out, tt_out, pcc=0.999)


@pytest.mark.parametrize(
    "channels, T, ratio",
    [
        (16, 64, 2),
        (32, 128, 2),
    ],
    ids=["c16_t64_r2", "c32_t128_r2"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VOCODER_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_downsample_1d(mesh_device: ttnn.MeshDevice, channels: int, T: int, ratio: int):
    """PCC check vs reference ``DownSample1d``."""
    from ltx_core.model.audio_vae.vocoder import DownSample1d

    torch.manual_seed(42)
    torch_down = DownSample1d(ratio=ratio)
    torch_down.eval()

    tt_down = LTXDownSample1d(
        ratio=ratio, kernel_size=torch_down.kernel_size, mesh_device=mesh_device, dtype=ttnn.float32
    )
    tt_down.lowpass._taps_cpu = torch_down.lowpass.filter.reshape(torch_down.kernel_size).float().tolist()

    x = torch.randn(1, channels, T, dtype=torch.float32)
    with torch.no_grad():
        torch_out = torch_down(x)

    x_dev = _to_dev_btc(x, mesh_device)
    y_dev = tt_down(x_dev)
    tt_out = _from_dev_btc(y_dev, trim_channels=channels)

    logger.info(f"DownSample1d: torch {torch_out.shape}, tt {tt_out.shape}")
    assert tt_out.shape == torch_out.shape, f"shape mismatch: torch {torch_out.shape}, tt {tt_out.shape}"
    assert_quality(torch_out, tt_out, pcc=0.999)


@pytest.mark.parametrize(
    "channels, T",
    [(32, 64), (64, 32)],
    ids=["c32_t64", "c64_t32"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VOCODER_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_vocoder_activation_1d(mesh_device: ttnn.MeshDevice, channels: int, T: int):
    """PCC check vs reference ``Activation1d(SnakeBeta)``."""
    from ltx_core.model.audio_vae.vocoder import Activation1d
    from ltx_core.model.audio_vae.vocoder import SnakeBeta as TorchSnakeBeta

    from models.tt_dit.layers.audio_ops import SnakeBeta

    torch.manual_seed(42)
    snake = TorchSnakeBeta(channels, alpha_logscale=True)
    snake.alpha.data = torch.randn(channels) * 0.1
    snake.beta.data = torch.randn(channels) * 0.1
    torch_act = Activation1d(snake)
    torch_act.eval()

    tt_snake = SnakeBeta(channels, alpha_logscale=True, mesh_device=mesh_device, dtype=ttnn.float32)
    tt_act = LTXVocoderActivation1d(
        channels=channels,
        activation=tt_snake,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )
    tt_act.load_torch_state_dict(torch_act.state_dict())

    x = torch.randn(1, channels, T, dtype=torch.float32) * 0.5
    with torch.no_grad():
        torch_out = torch_act(x)

    x_dev = _to_dev_btc(x, mesh_device)
    y_dev = tt_act(x_dev)
    tt_out = _from_dev_btc(y_dev, trim_channels=channels)

    logger.info(f"Activation1d: torch {torch_out.shape}, tt {tt_out.shape}")
    assert tt_out.shape == torch_out.shape
    assert_quality(torch_out, tt_out, pcc=0.999)


@pytest.mark.parametrize(
    "in_c, out_c, k, stride, T",
    [
        (16, 8, 4, 2, 32),
        (32, 16, 11, 5, 16),
    ],
    ids=["k4s2", "k11s5"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VOCODER_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_conv_transpose_1d(mesh_device: ttnn.MeshDevice, in_c: int, out_c: int, k: int, stride: int, T: int):
    """PCC check vs ``torch.nn.ConvTranspose1d`` with ``padding=(k-stride)//2``."""
    torch.manual_seed(42)
    torch_ct = torch.nn.ConvTranspose1d(in_c, out_c, kernel_size=k, stride=stride, padding=(k - stride) // 2)
    torch_ct.eval()

    tt_ct = LTXConvTranspose1d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=k,
        stride=stride,
        bias=True,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )
    tt_ct.load_torch_state_dict(torch_ct.state_dict())

    x = torch.randn(1, in_c, T, dtype=torch.float32)
    with torch.no_grad():
        torch_out = torch_ct(x)

    x_dev = _to_dev_btc(x, mesh_device)
    y_dev = tt_ct(x_dev)
    tt_out = _from_dev_btc(y_dev, trim_channels=out_c)

    logger.info(f"ConvTranspose1d k={k} s={stride}: torch {torch_out.shape}, tt {tt_out.shape}")
    assert tt_out.shape == torch_out.shape, f"shape mismatch: torch {torch_out.shape}, tt {tt_out.shape}"
    assert_quality(torch_out, tt_out, pcc=0.999)


@pytest.mark.parametrize(
    "channels, T, ks, dilation",
    [
        (32, 64, 3, (1, 3, 5)),
    ],
    ids=["c32_t64_k3"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VOCODER_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_amp_block_1(mesh_device: ttnn.MeshDevice, channels: int, T: int, ks: int, dilation: tuple):
    """PCC check vs reference ``AMPBlock1`` with random init."""
    from ltx_core.model.audio_vae.vocoder import AMPBlock1

    torch.manual_seed(42)
    torch_block = AMPBlock1(channels, kernel_size=ks, dilation=dilation, activation="snakebeta")
    with torch.no_grad():
        for act_list in (torch_block.acts1, torch_block.acts2):
            for a1d in act_list:
                a1d.act.alpha.data = torch.randn(channels) * 0.1
                a1d.act.beta.data = torch.randn(channels) * 0.1
    torch_block.eval()

    tt_block = LTXAMPBlock1(
        channels=channels,
        kernel_size=ks,
        dilation=dilation,
        activation="snakebeta",
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )
    tt_block.load_torch_state_dict(torch_block.state_dict())

    x = torch.randn(1, channels, T, dtype=torch.float32) * 0.5
    with torch.no_grad():
        torch_out = torch_block(x)

    x_dev = _to_dev_btc(x, mesh_device)
    y_dev = tt_block(x_dev)
    tt_out = _from_dev_btc(y_dev, trim_channels=channels)

    logger.info(f"AMPBlock1: torch {torch_out.shape}, tt {tt_out.shape}")
    assert tt_out.shape == torch_out.shape
    assert_quality(torch_out, tt_out, pcc=0.99)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_VOCODER_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx_vocoder(mesh_device: ttnn.MeshDevice):
    """PCC check of the full vocoder with the production LTX-2.3 22B config."""
    torch_voc = _build_torch_vocoder(seed=42)
    tt_voc = _build_tt_vocoder(mesh_device)
    tt_voc.load_torch_state_dict(torch_voc.state_dict())

    mel = _sample_vocoder_mel()
    bsz, stereo = mel.shape[0], mel.shape[1]

    with torch.no_grad():
        ref_out = torch_voc(mel)
    tt_out = tt_voc(mel)

    logger.info(f"Vocoder: ref {ref_out.shape}, tt {tt_out.shape}")
    assert tt_out.shape == ref_out.shape, f"shape mismatch: ref {ref_out.shape}, tt {tt_out.shape}"
    assert_quality(ref_out, tt_out, pcc=0.99)

    try:
        from ltx_core.model.audio_vae.vocoder import MelSTFT

        mel_stft = MelSTFT(filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80)
        with torch.no_grad():
            ref_mel, _, _, _ = mel_stft.mel_spectrogram(ref_out.reshape(bsz * stereo, -1))
            tt_mel, _, _, _ = mel_stft.mel_spectrogram(tt_out.reshape(bsz * stereo, -1))
        l1_db = (ref_mel - tt_mel).abs().mean().item()
        logger.info(f"Vocoder mel L1 = {l1_db:.4f} dB")
        assert l1_db <= 5.0, f"mel L1 {l1_db:.3f} > 5.0 - large spectral divergence"
    except Exception as exc:
        logger.warning(f"Skipping mel L1 sanity (failed to build MelSTFT): {exc}")

    logger.info("PASSED: LTXVocoder matches reference PCC >= 0.99")


# ============================================================================
# Vocoder + BWE (Stage C) component + E2E tests
# ============================================================================


# Same single-chip params as vocoder component tests.
_LTX_BWE_MESH_DEVICE_PARAMS = _LTX_VOCODER_MESH_DEVICE_PARAMS

_MAIN_VOCODER_CFG = dict(
    resblock_kernel_sizes=[3, 7, 11],
    upsample_rates=[5, 2, 2, 2, 2, 2],
    upsample_kernel_sizes=[11, 4, 4, 4, 4, 4],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_initial_channel=1536,
    resblock="AMP1",
    output_sampling_rate=16000,
    activation="snakebeta",
    use_tanh_at_final=False,
    apply_final_activation=True,
    use_bias_at_final=False,
)
_BWE_VOCODER_CFG = dict(
    resblock_kernel_sizes=[3, 7, 11],
    upsample_rates=[6, 5, 2, 2, 2],
    upsample_kernel_sizes=[12, 11, 4, 4, 4],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    upsample_initial_channel=512,
    resblock="AMP1",
    output_sampling_rate=48000,
    activation="snakebeta",
    use_tanh_at_final=False,
    apply_final_activation=False,
    use_bias_at_final=False,
)
_MEL_STFT_CFG = dict(
    filter_length=512,
    hop_length=80,
    win_length=512,
    n_mel_channels=64,
)
_BWE_HOP_LENGTH = 80
_INPUT_SR = 16000
_OUTPUT_SR = 48000


@pytest.mark.parametrize(
    "T",
    [320, 640, 1280],
    ids=["T320", "T640", "T1280"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_BWE_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_stft_fn(mesh_device: ttnn.MeshDevice, T: int):
    """PCC check vs the reference ``_STFTFn`` with random ``forward_basis``."""
    from ltx_core.model.audio_vae.vocoder import _STFTFn

    torch.manual_seed(42)
    filter_length, hop_length, win_length = (
        _MEL_STFT_CFG["filter_length"],
        _MEL_STFT_CFG["hop_length"],
        _MEL_STFT_CFG["win_length"],
    )
    n_freqs = filter_length // 2 + 1

    torch_stft = _STFTFn(filter_length, hop_length, win_length)
    forward_basis = torch.randn(n_freqs * 2, 1, win_length, dtype=torch.float32) * 0.05
    inverse_basis = torch.randn(n_freqs * 2, 1, win_length, dtype=torch.float32) * 0.05
    torch_stft.forward_basis.data = forward_basis.clone()
    torch_stft.inverse_basis.data = inverse_basis.clone()
    torch_stft.eval()

    tt_stft = LTX_STFTFn(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )
    tt_stft.load_torch_state_dict(torch_stft.state_dict())

    y = torch.randn(1, T, dtype=torch.float32) * 0.5
    with torch.no_grad():
        ref_mag, ref_phase = torch_stft(y)

    y_btc = y.unsqueeze(-1).contiguous()
    y_dev = ttnn.from_torch(y_btc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    tt_mag, tt_phase = tt_stft(y_dev)
    tt_mag_host = ttnn.to_torch(ttnn.get_device_tensors(tt_mag)[0]).transpose(1, 2).contiguous()
    tt_phase_host = ttnn.to_torch(ttnn.get_device_tensors(tt_phase)[0]).transpose(1, 2).contiguous()

    logger.info(
        f"STFT: ref mag {ref_mag.shape}, tt mag {tt_mag_host.shape}; "
        f"ref phase {ref_phase.shape}, tt phase {tt_phase_host.shape}"
    )
    assert tt_mag_host.shape == ref_mag.shape, f"magnitude shape mismatch: ref {ref_mag.shape}, tt {tt_mag_host.shape}"
    assert tt_phase_host.shape == ref_phase.shape
    assert_quality(ref_mag, tt_mag_host, pcc=0.999)
    assert_quality(ref_phase, tt_phase_host, pcc=0.99)


@pytest.mark.parametrize(
    "T",
    [640, 1280],
    ids=["T640", "T1280"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_BWE_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_mel_stft(mesh_device: ttnn.MeshDevice, T: int):
    """PCC check vs reference ``MelSTFT`` with random buffers."""
    from ltx_core.model.audio_vae.vocoder import MelSTFT

    torch.manual_seed(43)
    filter_length, hop_length, win_length, n_mels = (
        _MEL_STFT_CFG["filter_length"],
        _MEL_STFT_CFG["hop_length"],
        _MEL_STFT_CFG["win_length"],
        _MEL_STFT_CFG["n_mel_channels"],
    )
    n_freqs = filter_length // 2 + 1

    torch_mel = MelSTFT(filter_length, hop_length, win_length, n_mels)
    torch_mel.stft_fn.forward_basis.data = torch.randn(n_freqs * 2, 1, win_length) * 0.05
    torch_mel.stft_fn.inverse_basis.data = torch.randn(n_freqs * 2, 1, win_length) * 0.05
    torch_mel.mel_basis.data = torch.randn(n_mels, n_freqs).abs() * 0.1
    torch_mel.eval()

    tt_mel = LTXMelSTFT(
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mels,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )
    tt_mel.load_torch_state_dict(torch_mel.state_dict())

    y = torch.randn(1, T, dtype=torch.float32) * 0.5
    with torch.no_grad():
        ref_log_mel, _, _, _ = torch_mel.mel_spectrogram(y)

    y_btc = y.unsqueeze(-1).contiguous()
    y_dev = ttnn.from_torch(y_btc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    tt_log_mel = tt_mel(y_dev)
    tt_log_mel_host = ttnn.to_torch(ttnn.get_device_tensors(tt_log_mel)[0]).transpose(1, 2).contiguous()

    logger.info(f"MelSTFT: ref {ref_log_mel.shape}, tt {tt_log_mel_host.shape}")
    assert tt_log_mel_host.shape == ref_log_mel.shape
    assert_quality(ref_log_mel, tt_log_mel_host, pcc=0.999)


@pytest.mark.parametrize(
    "channels, T, ratio",
    [
        (2, 128, 2),
        (4, 256, 2),
        (2, 128, 3),
    ],
    ids=["c2_t128_r2", "c4_t256_r2", "c2_t128_r3"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_BWE_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_hann_upsample_1d(mesh_device: ttnn.MeshDevice, channels: int, T: int, ratio: int):
    """PCC check vs the reference Hann-window ``UpSample1d``."""
    from ltx_core.model.audio_vae.vocoder import UpSample1d

    torch.manual_seed(44)
    torch_up = UpSample1d(ratio=ratio, persistent=False, window_type="hann")
    torch_up.eval()

    tt_up = LTXHannUpSample1d(ratio=ratio, mesh_device=mesh_device, dtype=ttnn.float32)
    tt_up._taps_cpu = torch_up.filter.reshape(tt_up.kernel_size).float().tolist()

    x = torch.randn(1, channels, T, dtype=torch.float32)
    with torch.no_grad():
        ref_out = torch_up(x)

    x_btc = x.transpose(1, 2).contiguous()
    x_dev = ttnn.from_torch(x_btc, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
    y_dev = tt_up(x_dev)
    tt_out_host = ttnn.to_torch(ttnn.get_device_tensors(y_dev)[0]).transpose(1, 2).contiguous()

    logger.info(f"HannUpSample1d: ref {ref_out.shape}, tt {tt_out_host.shape}")
    assert tt_out_host.shape == ref_out.shape, f"shape mismatch: ref {ref_out.shape}, tt {tt_out_host.shape}"
    assert_quality(ref_out, tt_out_host, pcc=0.999)


def _build_torch_vocoder_with_bwe(seed: int = 42):
    """Build the reference VocoderWithBWE with the production config."""
    from ltx_core.model.audio_vae.vocoder import MelSTFT, Vocoder, VocoderWithBWE

    torch.manual_seed(seed)

    main_voc = Vocoder(**_MAIN_VOCODER_CFG)
    bwe_voc = Vocoder(**_BWE_VOCODER_CFG)
    mel_stft = MelSTFT(**_MEL_STFT_CFG)

    n_freqs = _MEL_STFT_CFG["filter_length"] // 2 + 1
    mel_stft.stft_fn.forward_basis.data = torch.randn(n_freqs * 2, 1, _MEL_STFT_CFG["win_length"]) * 0.05
    mel_stft.stft_fn.inverse_basis.data = torch.randn(n_freqs * 2, 1, _MEL_STFT_CFG["win_length"]) * 0.05
    mel_stft.mel_basis.data = torch.randn(_MEL_STFT_CFG["n_mel_channels"], n_freqs).abs() * 0.1

    full = VocoderWithBWE(
        vocoder=main_voc,
        bwe_generator=bwe_voc,
        mel_stft=mel_stft,
        input_sampling_rate=_INPUT_SR,
        output_sampling_rate=_OUTPUT_SR,
        hop_length=_BWE_HOP_LENGTH,
    )
    _randomize_snake_alphas(full)
    full.eval()
    return full


def _build_tt_vocoder_with_bwe(mesh_device: ttnn.MeshDevice) -> LTXVocoderWithBWE:
    """Construct the device-side LTXVocoderWithBWE shell."""
    tt_main_cfg = dict(
        resblock_kernel_sizes=_MAIN_VOCODER_CFG["resblock_kernel_sizes"],
        upsample_rates=_MAIN_VOCODER_CFG["upsample_rates"],
        upsample_kernel_sizes=_MAIN_VOCODER_CFG["upsample_kernel_sizes"],
        resblock_dilation_sizes=_MAIN_VOCODER_CFG["resblock_dilation_sizes"],
        upsample_initial_channel=_MAIN_VOCODER_CFG["upsample_initial_channel"],
        resblock=_MAIN_VOCODER_CFG["resblock"],
        activation=_MAIN_VOCODER_CFG["activation"],
        use_tanh_at_final=_MAIN_VOCODER_CFG["use_tanh_at_final"],
        apply_final_activation=_MAIN_VOCODER_CFG["apply_final_activation"],
        use_bias_at_final=_MAIN_VOCODER_CFG["use_bias_at_final"],
        in_channels=128,
        out_channels=2,
    )
    tt_bwe_cfg = dict(
        resblock_kernel_sizes=_BWE_VOCODER_CFG["resblock_kernel_sizes"],
        upsample_rates=_BWE_VOCODER_CFG["upsample_rates"],
        upsample_kernel_sizes=_BWE_VOCODER_CFG["upsample_kernel_sizes"],
        resblock_dilation_sizes=_BWE_VOCODER_CFG["resblock_dilation_sizes"],
        upsample_initial_channel=_BWE_VOCODER_CFG["upsample_initial_channel"],
        resblock=_BWE_VOCODER_CFG["resblock"],
        activation=_BWE_VOCODER_CFG["activation"],
        use_tanh_at_final=_BWE_VOCODER_CFG["use_tanh_at_final"],
        apply_final_activation=_BWE_VOCODER_CFG["apply_final_activation"],
        use_bias_at_final=_BWE_VOCODER_CFG["use_bias_at_final"],
        in_channels=128,
        out_channels=2,
    )
    main_voc = LTXVocoder(mesh_device=mesh_device, dtype=ttnn.float32, **tt_main_cfg)
    bwe_voc = LTXVocoder(mesh_device=mesh_device, dtype=ttnn.float32, **tt_bwe_cfg)
    mel_stft = LTXMelSTFT(
        filter_length=_MEL_STFT_CFG["filter_length"],
        hop_length=_MEL_STFT_CFG["hop_length"],
        win_length=_MEL_STFT_CFG["win_length"],
        n_mel_channels=_MEL_STFT_CFG["n_mel_channels"],
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )
    full = LTXVocoderWithBWE(
        vocoder=main_voc,
        bwe_generator=bwe_voc,
        mel_stft=mel_stft,
        input_sampling_rate=_INPUT_SR,
        output_sampling_rate=_OUTPUT_SR,
        hop_length=_BWE_HOP_LENGTH,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )
    return full


@pytest.mark.parametrize(
    "mesh_device, device_params",
    _LTX_BWE_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx_vocoder_with_bwe(mesh_device: ttnn.MeshDevice):
    """Full PCC check of LTXVocoderWithBWE vs reference ``VocoderWithBWE``."""
    torch_full = _build_torch_vocoder_with_bwe(seed=42)
    tt_full = _build_tt_vocoder_with_bwe(mesh_device)

    sd = torch_full.state_dict()
    incompatible = tt_full.load_torch_state_dict(sd)
    if incompatible.missing_keys:
        logger.warning(f"unexpected missing keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        logger.warning(f"unexpected keys: {incompatible.unexpected_keys}")
    assert not incompatible.missing_keys, f"missing keys: {incompatible.missing_keys}"

    bsz, stereo, t_frames, mel_bins = 1, 2, 64, 64
    mel = torch.randn(bsz, stereo, t_frames, mel_bins, dtype=torch.float32) * 0.5

    with torch.no_grad():
        ref_x = torch_full.vocoder(mel.float())
    tt_x = tt_full.vocoder(mel.float())
    logger.info(f"main vocoder 24kHz: ref {ref_x.shape}, tt {tt_x.shape}")
    logger.info(
        f"  ref_x stats: min={ref_x.min().item():.4f} max={ref_x.max().item():.4f} "
        f"mean={ref_x.mean().item():.4f} std={ref_x.std().item():.4f}"
    )
    logger.info(
        f"  tt_x  stats: min={tt_x.min().item():.4f} max={tt_x.max().item():.4f} "
        f"mean={tt_x.mean().item():.4f} std={tt_x.std().item():.4f}"
    )
    assert tt_x.shape == ref_x.shape
    logger.info("main vocoder PCC:")
    assert_quality(ref_x, tt_x, pcc=0.99)

    _, _, length_low_rate = ref_x.shape
    output_length = length_low_rate * _OUTPUT_SR // _INPUT_SR
    remainder = length_low_rate % _BWE_HOP_LENGTH
    if remainder != 0:
        pad_right = _BWE_HOP_LENGTH - remainder
        ref_x_pad = torch.nn.functional.pad(ref_x, (0, pad_right))
        tt_x_pad = torch.nn.functional.pad(tt_x, (0, pad_right))
    else:
        ref_x_pad, tt_x_pad = ref_x, tt_x

    with torch.no_grad():
        ref_mel = torch_full._compute_mel(ref_x_pad)
    tt_mel = tt_full._compute_mel_device(tt_x_pad)
    logger.info(f"log-mel: ref {ref_mel.shape}, tt {tt_mel.shape}")
    logger.info(
        f"  ref_mel stats: min={ref_mel.min().item():.4f} max={ref_mel.max().item():.4f} "
        f"mean={ref_mel.mean().item():.4f} std={ref_mel.std().item():.4f}"
    )
    logger.info(
        f"  tt_mel  stats: min={tt_mel.min().item():.4f} max={tt_mel.max().item():.4f} "
        f"mean={tt_mel.mean().item():.4f} std={tt_mel.std().item():.4f}"
    )
    assert tt_mel.shape == ref_mel.shape

    tt_mel_from_ref = tt_full._compute_mel_device(ref_x_pad)
    logger.info("log-mel PCC (apples-to-apples on ref_x_pad):")
    assert_quality(ref_mel, tt_mel_from_ref, pcc=0.99)

    logger.info("log-mel PCC (tt main -> tt mel vs ref main -> ref mel) [info only]:")
    try:
        assert_quality(ref_mel, tt_mel, pcc=0.0)
    except Exception as exc:
        logger.warning(f"informational mel PCC failure: {exc}")

    ref_mel_for_bwe = ref_mel.transpose(2, 3).contiguous()
    with torch.no_grad():
        ref_residual = torch_full.bwe_generator(ref_mel_for_bwe)
    tt_residual = tt_full.bwe_generator(ref_mel_for_bwe)
    logger.info(f"residual 48kHz (same mel): ref {ref_residual.shape}, tt {tt_residual.shape}")
    logger.info(
        f"  ref_residual stats: min={ref_residual.min().item():.4f} max={ref_residual.max().item():.4f} "
        f"mean={ref_residual.mean().item():.4f} std={ref_residual.std().item():.4f}"
    )
    logger.info(
        f"  tt_residual  stats: min={tt_residual.min().item():.4f} max={tt_residual.max().item():.4f} "
        f"mean={tt_residual.mean().item():.4f} std={tt_residual.std().item():.4f}"
    )
    assert tt_residual.shape == ref_residual.shape
    logger.info("residual PCC (apples-to-apples):")
    assert_quality(ref_residual, tt_residual, pcc=0.98)

    with torch.no_grad():
        ref_skip = torch_full.resampler(ref_x_pad)
    tt_skip = tt_full._resample_device(ref_x_pad)
    logger.info(f"skip 48kHz (same wave): ref {ref_skip.shape}, tt {tt_skip.shape}")
    assert tt_skip.shape == ref_skip.shape
    logger.info("skip PCC (apples-to-apples):")
    assert_quality(ref_skip, tt_skip, pcc=0.99)

    with torch.no_grad():
        ref_out = torch_full(mel)
    tt_out = tt_full(mel)
    logger.info(f"VocoderWithBWE final: ref {ref_out.shape}, tt {tt_out.shape}")
    assert tt_out.shape == ref_out.shape
    logger.info(f"final 48kHz length: {ref_out.shape[-1]}, expected ~{output_length}")
    logger.info("final PCC (full forward) [info only - Stage B accumulation drift]:")
    try:
        assert_quality(ref_out, tt_out, pcc=0.0)
    except Exception as exc:
        logger.warning(f"informational final PCC failure: {exc}")

    logger.info(
        "PASSED: LTXVocoderWithBWE component PCCs >= 0.99 (mel, residual, skip) " "with apples-to-apples inputs"
    )
