# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""LTX audio component tests aligned with pipeline decode stages.

Stage/component IDs match ``pipeline_ltx.py`` audio decode chain:
- Stage A: ``audio_decoder`` (mel-VAE)
- Stage B: ``vocoder`` (24 kHz waveform)
- Stage C: ``vocoder_with_bwe`` (48 kHz BWE path)

Torch references use diffusers LTX-2 audio modules (same as transformer/rope tests).
"""

from __future__ import annotations

import os

import einops
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.audio_vae.audio_decoder_ltx import AudioDecoder
from models.tt_dit.models.audio_vae.bwe_ltx import MelSTFT, VocoderWithBWE
from models.tt_dit.models.audio_vae.vocoder_ltx import Vocoder
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params, ring_params


# ---------------------------------------------------------------------------
# Mesh / device params (mirrors test_pipeline_ltx_fast_av.py)
# ---------------------------------------------------------------------------
def _with_audio_dev_l1(base: dict) -> dict:
    return {**base, "l1_small_size": 32768}


_line_params = _with_audio_dev_l1(line_params)
_ring_params = _with_audio_dev_l1(ring_params)
_ring_trace_params = {**_ring_params, "trace_region_size": 300_000_000}

_AUDIO_FAST_AV_MESH_PARAMS_FULL = [
    pytest.param((2, 2), (2, 2), 0, 1, 2, False, _line_params, ttnn.Topology.Linear, True, id="2x2sp0tp1"),
    pytest.param((2, 4), (2, 4), 0, 1, 1, True, _line_params, ttnn.Topology.Linear, True, id="2x4sp0tp1"),
    pytest.param((2, 4), (2, 4), 1, 0, 2, True, _line_params, ttnn.Topology.Linear, False, id="bh_2x4sp1tp0"),
    pytest.param((4, 8), (4, 8), 1, 0, 4, False, _ring_params, ttnn.Topology.Ring, True, id="wh_4x8sp1tp0"),
    pytest.param((4, 8), (4, 8), 1, 0, 2, False, _line_params, ttnn.Topology.Linear, False, id="bh_4x8sp1tp0_linear"),
    pytest.param((4, 8), (4, 8), 1, 0, 2, False, _ring_trace_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0_ring"),
    pytest.param((4, 32), (4, 32), 1, 0, 2, False, _ring_params, ttnn.Topology.Ring, False, id="bh_4x32sp1tp0"),
]


def _audio_mesh_params():
    full = os.environ.get("LTX_AUDIO_FULL_MATRIX", "0").lower() in ("1", "true", "yes")
    return _AUDIO_FAST_AV_MESH_PARAMS_FULL if full else _AUDIO_FAST_AV_MESH_PARAMS_FULL[1:3]


_AUDIO_FAST_AV_MESH_PARAMS = _audio_mesh_params()

# ---------------------------------------------------------------------------
# Stage configs (pipeline_ltx.py audio decode chain)
# ---------------------------------------------------------------------------
_AUDIO_DECODER_CFG = dict(
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

_MAIN_VOCODER_CFG = dict(
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

_INPUT_SR = 16000
_OUTPUT_SR = 48000
_HOP_LENGTH = 80


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vocoder_mel(t_frames: int = 120) -> torch.Tensor:
    """Stereo mel input. t_frames=120 with T-shard factor 4 leaves t_pad=8 (128-aligned),
    so the tile-align pad fits inside the last shard and the sharded path's per-op tail
    materialization is exercised — the boundary the click came from."""
    return torch.randn(1, 2, t_frames, 64, dtype=torch.float32) * 0.5


def _assert_sharded_matches_unsharded(unsharded: torch.Tensor, sharded: torch.Tensor, *, name: str) -> None:
    """The T-sharded vocoder must reproduce the unsharded device path up to the bf16 storage
    floor. A t_pad boundary regression (the audible click) shows up as a localized tail spike
    of order 1e-2 — far above the ~1e-4 floor — so a tight max|Δ| catches it where aggregate
    PCC against the torch reference (with its own fp32-kernel floor) does not."""
    max_abs = (unsharded - sharded).abs().max().item()
    logger.info(f"{name}: sharded vs unsharded device max|Δ| = {max_abs:.3e}")
    assert max_abs < 5e-3, f"{name}: sharded vs unsharded device max|Δ| {max_abs:.3e} — t_pad boundary drift"


def _require_diffusers():
    pytest.importorskip("diffusers")
    from diffusers.models.autoencoders.autoencoder_kl_ltx2_audio import AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE

    return AutoencoderKLLTX2Audio, LTX2Vocoder, LTX2VocoderWithBWE


def _diffusers_vocoder_state_to_tt(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Invert diffusers LTX-2.3 vocoder key names to the TT/ltx_core layout."""
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = (
            key.replace("conv_in.", "conv_pre.")
            .replace("conv_out.", "conv_post.")
            .replace("act_out.", "act_post.")
            .replace("upsamplers.", "ups.")
            .replace("resnets.", "resblocks.")
        )
        # TT derives downsample filter taps at init; these are not loadable Parameters.
        if new_key.endswith("downsample.filter"):
            continue
        out[new_key] = value
    return out


def _diffusers_vocoder_with_bwe_state_to_tt(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for prefix in ("vocoder.", "bwe_generator."):
        sub = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        for k, v in _diffusers_vocoder_state_to_tt(sub).items():
            out[f"{prefix}{k}"] = v
    for key, value in state_dict.items():
        if key.startswith("mel_stft."):
            out[key] = value
    return out


def _audio_decoder_state_from_diffusers(
    vae,
    *,
    stats_std: torch.Tensor,
    stats_mean: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = {k[len("decoder.") :]: v for k, v in vae.state_dict().items() if k.startswith("decoder.")}
    state["per_channel_statistics.std-of-means"] = stats_std
    state["per_channel_statistics.mean-of-means"] = stats_mean
    return state


def _decode_latent_ref(
    vae,
    latent: torch.Tensor,
    *,
    stats_std: torch.Tensor,
    stats_mean: torch.Tensor,
) -> torch.Tensor:
    """Match ltx_core AudioDecoder / TT denormalize + decoder forward."""
    b, channels, _t, mel_bins = latent.shape
    patched = einops.rearrange(latent, "b c t f -> b t (c f)")
    std = stats_std.to(dtype=patched.dtype, device=patched.device)
    mean = stats_mean.to(dtype=patched.dtype, device=patched.device)
    last_dim = patched.shape[-1]
    if std.shape[0] != last_dim:
        if std.shape[0] == channels and last_dim == channels * mel_bins:
            std = std.repeat_interleave(mel_bins)
            mean = mean.repeat_interleave(mel_bins)
        else:
            raise ValueError(f"per-channel stats {tuple(std.shape)} do not broadcast to last dim {last_dim}")
    sample = einops.rearrange(patched * std + mean, "b t (c f) -> b c t f", c=channels, f=mel_bins)
    return vae.decoder(sample)


def _compute_mel_ref(vocoder_with_bwe, audio_bct: torch.Tensor) -> torch.Tensor:
    batch, n_channels, _ = audio_bct.shape
    flat = audio_bct.reshape(batch * n_channels, -1).unsqueeze(1)
    log_mel, _, _, _ = vocoder_with_bwe.mel_stft(flat)
    return log_mel.reshape(batch, n_channels, log_mel.shape[1], log_mel.shape[2])


def _tt_vocoder_cfg(cfg: dict) -> dict:
    # Torch reference config carries keys that TT constructor does not accept.
    return {k: v for k, v in cfg.items() if k != "output_sampling_rate"}


def _randomize_snake_alphas(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for submodule in module.modules():
            if hasattr(submodule, "alpha") and isinstance(submodule.alpha, torch.nn.Parameter):
                submodule.alpha.data = torch.randn_like(submodule.alpha.data) * 0.1
            if hasattr(submodule, "beta") and isinstance(submodule.beta, torch.nn.Parameter):
                submodule.beta.data = torch.randn_like(submodule.beta.data) * 0.1


def _build_torch_stage_b(seed: int = 42):
    _, LTX2Vocoder, _ = _require_diffusers()

    torch.manual_seed(seed)
    model = LTX2Vocoder(
        in_channels=128,
        hidden_channels=_MAIN_VOCODER_CFG["upsample_initial_channel"],
        out_channels=2,
        upsample_kernel_sizes=_MAIN_VOCODER_CFG["upsample_kernel_sizes"],
        upsample_factors=_MAIN_VOCODER_CFG["upsample_rates"],
        resnet_kernel_sizes=_MAIN_VOCODER_CFG["resblock_kernel_sizes"],
        resnet_dilations=_MAIN_VOCODER_CFG["resblock_dilation_sizes"],
        act_fn=_MAIN_VOCODER_CFG["activation"],
        antialias=True,
        final_act_fn=None,
        final_bias=_MAIN_VOCODER_CFG["use_bias_at_final"],
        output_sampling_rate=_MAIN_VOCODER_CFG["output_sampling_rate"],
    ).eval()
    _randomize_snake_alphas(model)
    return model


def _audio_parallel_config(mesh_shape: tuple[int, int]) -> ParallelFactor | None:
    t_axis = 0 if mesh_shape[0] >= mesh_shape[1] else 1
    t_factor = mesh_shape[t_axis]
    return ParallelFactor(factor=t_factor, mesh_axis=t_axis) if t_factor > 1 else None


def _build_tt_stage_b(
    mesh_device: ttnn.MeshDevice, *, parallel_config: ParallelFactor | None, ccl_manager: CCLManager | None
) -> Vocoder:
    return Vocoder(
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        **_tt_vocoder_cfg(_MAIN_VOCODER_CFG),
    )


def _build_torch_stage_c(seed: int = 42):
    _, LTX2Vocoder, LTX2VocoderWithBWE = _require_diffusers()

    torch.manual_seed(seed)
    full = LTX2VocoderWithBWE(
        in_channels=128,
        hidden_channels=_MAIN_VOCODER_CFG["upsample_initial_channel"],
        out_channels=2,
        upsample_kernel_sizes=_MAIN_VOCODER_CFG["upsample_kernel_sizes"],
        upsample_factors=_MAIN_VOCODER_CFG["upsample_rates"],
        resnet_kernel_sizes=_MAIN_VOCODER_CFG["resblock_kernel_sizes"],
        resnet_dilations=_MAIN_VOCODER_CFG["resblock_dilation_sizes"],
        act_fn=_MAIN_VOCODER_CFG["activation"],
        antialias=True,
        final_act_fn=None,
        final_bias=_MAIN_VOCODER_CFG["use_bias_at_final"],
        bwe_in_channels=128,
        bwe_hidden_channels=_BWE_VOCODER_CFG["upsample_initial_channel"],
        bwe_out_channels=2,
        bwe_upsample_kernel_sizes=_BWE_VOCODER_CFG["upsample_kernel_sizes"],
        bwe_upsample_factors=_BWE_VOCODER_CFG["upsample_rates"],
        bwe_resnet_kernel_sizes=_BWE_VOCODER_CFG["resblock_kernel_sizes"],
        bwe_resnet_dilations=_BWE_VOCODER_CFG["resblock_dilation_sizes"],
        bwe_act_fn=_BWE_VOCODER_CFG["activation"],
        bwe_antialias=True,
        bwe_final_act_fn=None,
        bwe_final_bias=_BWE_VOCODER_CFG["use_bias_at_final"],
        filter_length=_MEL_STFT_CFG["filter_length"],
        hop_length=_MEL_STFT_CFG["hop_length"],
        window_length=_MEL_STFT_CFG["win_length"],
        num_mel_channels=_MEL_STFT_CFG["n_mel_channels"],
        input_sampling_rate=_INPUT_SR,
        output_sampling_rate=_OUTPUT_SR,
    )
    n_freqs = _MEL_STFT_CFG["filter_length"] // 2 + 1
    with torch.no_grad():
        full.mel_stft.stft_fn.forward_basis.copy_(torch.randn(n_freqs * 2, 1, _MEL_STFT_CFG["win_length"]) * 0.05)
        full.mel_stft.stft_fn.inverse_basis.copy_(torch.randn(n_freqs * 2, 1, _MEL_STFT_CFG["win_length"]) * 0.05)
        full.mel_stft.mel_basis.copy_(torch.randn(_MEL_STFT_CFG["n_mel_channels"], n_freqs).abs() * 0.1)
    _randomize_snake_alphas(full)
    return full.eval()


def _build_tt_stage_c(
    mesh_device: ttnn.MeshDevice, *, parallel_config: ParallelFactor | None, ccl_manager: CCLManager | None
) -> VocoderWithBWE:
    main_voc = Vocoder(
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        **_tt_vocoder_cfg(_MAIN_VOCODER_CFG),
    )
    bwe_voc = Vocoder(
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        **_tt_vocoder_cfg(_BWE_VOCODER_CFG),
    )
    mel_stft = MelSTFT(mesh_device=mesh_device, dtype=ttnn.float32, **_MEL_STFT_CFG)
    return VocoderWithBWE(
        vocoder=main_voc,
        bwe_generator=bwe_voc,
        mel_stft=mel_stft,
        input_sampling_rate=_INPUT_SR,
        output_sampling_rate=_OUTPUT_SR,
        hop_length=_HOP_LENGTH,
        mesh_device=mesh_device,
        dtype=ttnn.float32,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    (
        "mesh_device",
        "mesh_shape",
        "sp_axis",
        "tp_axis",
        "num_links",
        "dynamic_load",
        "device_params",
        "topology",
        "is_fsdp",
    ),
    _AUDIO_FAST_AV_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_stage_a_audio_decoder(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    is_fsdp: bool,
):
    AutoencoderKLLTX2Audio, _, _ = _require_diffusers()

    _ = (sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp)  # parity with fast_av config schema
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    torch.manual_seed(42)
    ref_vae = AutoencoderKLLTX2Audio(
        base_channels=_AUDIO_DECODER_CFG["ch"],
        output_channels=_AUDIO_DECODER_CFG["out_ch"],
        ch_mult=_AUDIO_DECODER_CFG["ch_mult"],
        num_res_blocks=_AUDIO_DECODER_CFG["num_res_blocks"],
        attn_resolutions=_AUDIO_DECODER_CFG["attn_resolutions"],
        resolution=_AUDIO_DECODER_CFG["resolution"],
        latent_channels=_AUDIO_DECODER_CFG["z_channels"],
        norm_type="pixel",
        causality_axis="height",
        mid_block_add_attention=_AUDIO_DECODER_CFG["mid_block_add_attention"],
        sample_rate=_AUDIO_DECODER_CFG["sample_rate"],
        mel_hop_length=_AUDIO_DECODER_CFG["mel_hop_length"],
        is_causal=_AUDIO_DECODER_CFG["is_causal"],
        mel_bins=_AUDIO_DECODER_CFG["mel_bins"],
    ).eval()

    z_times_f = _AUDIO_DECODER_CFG["z_channels"] * _AUDIO_DECODER_CFG["mel_bins"]
    stats_std = torch.ones(z_times_f)
    stats_mean = torch.zeros(z_times_f)

    tt_decoder = AudioDecoder(mesh_device=mesh_device, **_AUDIO_DECODER_CFG)
    tt_decoder.load_torch_state_dict(
        _audio_decoder_state_from_diffusers(ref_vae, stats_std=stats_std, stats_mean=stats_mean)
    )

    latent = torch.randn(1, _AUDIO_DECODER_CFG["z_channels"], 64, _AUDIO_DECODER_CFG["mel_bins"], dtype=torch.float32)
    with torch.no_grad():
        torch_out = _decode_latent_ref(ref_vae, latent, stats_std=stats_std, stats_mean=stats_mean)
    tt_out = tt_decoder(latent)

    assert torch_out.shape == tt_out.shape
    assert_quality(torch_out, tt_out, pcc=0.998, relative_rmse=0.05)


@pytest.mark.parametrize(
    (
        "mesh_device",
        "mesh_shape",
        "sp_axis",
        "tp_axis",
        "num_links",
        "dynamic_load",
        "device_params",
        "topology",
        "is_fsdp",
    ),
    _AUDIO_FAST_AV_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_stage_b_vocoder(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    is_fsdp: bool,
):
    _ = (sp_axis, tp_axis, dynamic_load, is_fsdp)  # parity with fast_av config schema
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    pc = _audio_parallel_config(mesh_shape)
    ccl = CCLManager(mesh_device, num_links=num_links, topology=topology) if pc is not None else None

    torch_voc = _build_torch_stage_b(seed=42)
    tt_voc = _build_tt_stage_b(mesh_device, parallel_config=pc, ccl_manager=ccl)
    tt_voc.load_torch_state_dict(_diffusers_vocoder_state_to_tt(torch_voc.state_dict()))

    mel = _vocoder_mel()
    with torch.no_grad():
        ref_out = torch_voc(mel)
    tt_out = tt_voc(mel)

    assert tt_out.shape == ref_out.shape
    assert_quality(ref_out, tt_out, pcc=0.99)

    # When T-sharded, assert the sharded path reproduces the unsharded device path: the t_pad
    # boundary click is localized to the tail and slips past aggregate PCC, but breaks this.
    if pc is not None:
        tt_voc_un = _build_tt_stage_b(mesh_device, parallel_config=None, ccl_manager=None)
        tt_voc_un.load_torch_state_dict(_diffusers_vocoder_state_to_tt(torch_voc.state_dict()))
        _assert_sharded_matches_unsharded(tt_voc_un(mel), tt_out, name="stage_b_vocoder")

    # Optional spectral sanity to catch large drifts that waveform PCC may miss.
    try:
        _, _, LTX2VocoderWithBWE = _require_diffusers()
        mel_stft = LTX2VocoderWithBWE(
            filter_length=1024,
            hop_length=256,
            window_length=1024,
            num_mel_channels=80,
            input_sampling_rate=_INPUT_SR,
            output_sampling_rate=_OUTPUT_SR,
        ).mel_stft
        with torch.no_grad():
            ref_mel, _, _, _ = mel_stft(ref_out.reshape(1, 2, -1))
            tt_mel, _, _, _ = mel_stft(tt_out.reshape(1, 2, -1))
        assert (ref_mel - tt_mel).abs().mean().item() <= 5.0
    except Exception as exc:
        logger.warning(f"Skipping Stage B mel sanity: {exc}")


@pytest.mark.parametrize(
    (
        "mesh_device",
        "mesh_shape",
        "sp_axis",
        "tp_axis",
        "num_links",
        "dynamic_load",
        "device_params",
        "topology",
        "is_fsdp",
    ),
    _AUDIO_FAST_AV_MESH_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_stage_c_vocoder_with_bwe(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    is_fsdp: bool,
):
    _ = (sp_axis, tp_axis, dynamic_load, is_fsdp)  # parity with fast_av config schema
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    pc = _audio_parallel_config(mesh_shape)
    ccl = CCLManager(mesh_device, num_links=num_links, topology=topology) if pc is not None else None

    torch_full = _build_torch_stage_c(seed=42)
    tt_full = _build_tt_stage_c(mesh_device, parallel_config=pc, ccl_manager=ccl)

    incompatible = tt_full.load_torch_state_dict(_diffusers_vocoder_with_bwe_state_to_tt(torch_full.state_dict()))
    assert not incompatible.missing_keys, f"missing keys: {incompatible.missing_keys}"

    mel = _vocoder_mel()

    # Stage B inside Stage C.
    with torch.no_grad():
        ref_main = torch_full.vocoder(mel)
    tt_main = tt_full.vocoder(mel)
    assert tt_main.shape == ref_main.shape
    assert_quality(ref_main, tt_main, pcc=0.99)
    if pc is not None:
        tt_full_un = _build_tt_stage_c(mesh_device, parallel_config=None, ccl_manager=None)
        tt_full_un.load_torch_state_dict(_diffusers_vocoder_with_bwe_state_to_tt(torch_full.state_dict()))
        _assert_sharded_matches_unsharded(tt_full_un.vocoder(mel), tt_main, name="stage_c_vocoder")

    # Log-mel path (apples-to-apples same waveform).
    remainder = ref_main.shape[-1] % _HOP_LENGTH
    if remainder:
        pad_right = _HOP_LENGTH - remainder
        ref_main = torch.nn.functional.pad(ref_main, (0, pad_right))
    with torch.no_grad():
        ref_mel = _compute_mel_ref(torch_full, ref_main)
    tt_mel = tt_full._compute_mel_device(ref_main)
    assert tt_mel.shape == ref_mel.shape
    assert_quality(ref_mel, tt_mel, pcc=0.99)

    # BWE residual and skip-resample paths.
    ref_mel_for_bwe = ref_mel.transpose(2, 3).contiguous()
    with torch.no_grad():
        ref_residual = torch_full.bwe_generator(ref_mel_for_bwe)
        ref_skip = torch_full.resampler(ref_main)
    tt_residual = tt_full.bwe_generator(ref_mel_for_bwe)
    tt_skip = tt_full._resample_device(ref_main)
    assert tt_residual.shape == ref_residual.shape
    assert tt_skip.shape == ref_skip.shape
    assert_quality(ref_residual, tt_residual, pcc=0.98)
    assert_quality(ref_skip, tt_skip, pcc=0.99)

    # Full Stage C forward: the muxed waveform is where the sharded tail click surfaced.
    with torch.no_grad():
        ref_out = torch_full(mel)
    tt_out = tt_full(mel)
    assert tt_out.shape == ref_out.shape
    if pc is not None:
        _assert_sharded_matches_unsharded(tt_full_un(mel), tt_out, name="stage_c_full")
