# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Audio tests for LTX-2: per-component stage checks + post-denoise decode e2e.

Component stage tests (random weights vs diffusers LTX-2 torch reference):
- Stage A: ``audio_decoder`` (mel-VAE)
- Stage B: ``vocoder`` (24 kHz waveform)
- Stage C: ``vocoder_with_bwe`` (48 kHz BWE path)

End-to-end decode (mel-VAE → vocoder → BWE, requires real checkpoints):
- on-device decode runs on representative meshes
- warm-path decode stays functional after cold compile/build
- on-device decode quality remains close to host reference (PSNR)

Denoise-path audio regressions live in sibling files.
"""

from __future__ import annotations

import math
import os
import time
from glob import glob

import einops
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.audio_vae.audio_decoder_ltx import LTXAudioDecoderAdapter
from models.tt_dit.models.audio_vae.bwe_ltx import MelSTFT, VocoderWithBWE
from models.tt_dit.models.audio_vae.mel_decoder_ltx import AudioUpsample, MelDecoder, ResnetBlock
from models.tt_dit.models.audio_vae.vocoder_ltx import AMPBlock1, Vocoder
from models.tt_dit.parallel.config import AudioTCParallelConfig, DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import skip_if_unsupported_num_links
from models.tt_dit.utils.video import Audio

from .ltx_mesh_params import LTX_AUDIO_MESH_PARAMS_FULL

_WARM_ITERS = 3


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
# Default checkpoint filenames used by the e2e/perf decode tests. Any of these can
# be overridden at runtime with LTX_CHECKPOINT, which is honored regardless of its
# basename (see _resolve_checkpoint). The tuples below only drive *fallback*
# discovery when LTX_CHECKPOINT is unset.
_DEV_CHECKPOINT = "ltx-2.3-22b-dev.safetensors"
_DISTILLED_CHECKPOINT = "ltx-2.3-22b-distilled-1.1.safetensors"
_LOCAL_CHECKPOINT_DIR = "~/.cache/ltx-checkpoints"
# HF cache snapshot glob for the LTX-2.3 repo; {name} is the checkpoint filename.
_HF_SNAPSHOT_GLOB = "~/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/{name}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_checkpoint(candidates: list[str], expected_filenames: tuple[str, ...]) -> str | None:
    # An explicit LTX_CHECKPOINT override wins unconditionally as long as the file
    # exists — that's the whole point of the env var. The expected_filenames tuple
    # only gates the fallback discovery below, never the override.
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit and os.path.exists(explicit):
        return explicit
    for path in candidates:
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            return expanded
    # Reuse default HF cache location if present; avoid network dependency.
    for name in expected_filenames:
        hits = glob(os.path.expanduser(_HF_SNAPSHOT_GLOB.format(name=name)))
        if hits:
            return hits[0]
    return None


def _build_pipeline(
    mesh_device: ttnn.MeshDevice,
    *,
    sp_axis: int,
    tp_axis: int,
    checkpoint: str,
    num_links: int,
    topology: ttnn.Topology,
) -> tuple[LTXPipeline, DiTParallelConfig]:
    mesh_shape = tuple(mesh_device.shape)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=mesh_shape[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=mesh_shape[tp_axis], mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device, num_links=num_links, topology=topology)
    pipeline = LTXPipeline(
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        checkpoint_name=None,
    )
    pipeline.checkpoint_name = checkpoint
    pipeline._audio_adapter = LTXAudioDecoderAdapter(
        pipeline.checkpoint_name,
        mesh_device=pipeline.mesh_device,
        vae_ccl_manager=pipeline.vae_ccl_manager,
        dit_parallel_config=pipeline.parallel_config,
        traced=pipeline._traced,
    )
    return pipeline, parallel_config


def _psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    ref = ref.float()
    test = test.float()
    mse = torch.mean((ref - test) ** 2).item()
    if mse == 0.0:
        return float("inf")
    peak = ref.abs().max().item()
    if peak == 0.0:
        return float("inf")
    return 20.0 * math.log10(peak) - 10.0 * math.log10(mse)


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
    LTX_AUDIO_MESH_PARAMS_FULL,
    indirect=["mesh_device", "device_params"],
)
def test_audio_decode_perf(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    is_fsdp: bool,
):
    ckpt = _resolve_checkpoint(
        [f"{_LOCAL_CHECKPOINT_DIR}/{_DEV_CHECKPOINT}"],
        (_DEV_CHECKPOINT,),
    )
    if ckpt is None:
        pytest.skip("checkpoint not found (set LTX_CHECKPOINT)")

    skip_if_unsupported_num_links(mesh_device, num_links)
    _ = (dynamic_load, is_fsdp)  # parity with fast_av config schema
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    torch.manual_seed(0)
    pipeline, parallel_config = _build_pipeline(
        mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        checkpoint=ckpt,
        num_links=num_links,
        topology=topology,
    )
    mesh_shape = tuple(mesh_device.shape)

    audio_latent = torch.randn(1, 256, 128, dtype=torch.float32)
    num_frames, fps = 145, 24.0

    t0 = time.time()
    dev_audio = pipeline.decode_audio(audio_latent, num_frames, fps=fps)
    t_cold = time.time() - t0
    assert dev_audio is not None

    warm_times = []
    for _ in range(_WARM_ITERS):
        t0 = time.time()
        out = pipeline.decode_audio(audio_latent, num_frames, fps=fps)
        warm_times.append(time.time() - t0)
        assert out is not None

    logger.info(
        f"Audio decode perf (mesh {mesh_shape}, T-shard={parallel_config.sequence_parallel.factor}): "
        f"cold={t_cold:.2f}s warm_min={min(warm_times):.2f}s warm_mean={sum(warm_times)/len(warm_times):.2f}s "
        f"wave={tuple(dev_audio.waveform.shape)} sr={dev_audio.sampling_rate}"
    )


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
    LTX_AUDIO_MESH_PARAMS_FULL,
    indirect=["mesh_device", "device_params"],
)
def test_audio_decode_e2e_psnr(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    dynamic_load: bool,
    topology: ttnn.Topology,
    is_fsdp: bool,
):
    ckpt = _resolve_checkpoint(
        [f"{_LOCAL_CHECKPOINT_DIR}/{_DISTILLED_CHECKPOINT}"],
        (_DISTILLED_CHECKPOINT,),
    )
    if ckpt is None:
        pytest.skip("distilled checkpoint not found (set LTX_CHECKPOINT)")

    skip_if_unsupported_num_links(mesh_device, num_links)
    _ = (dynamic_load, is_fsdp)  # parity with fast_av config schema
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    torch.manual_seed(0)
    mesh_shape = tuple(mesh_device.shape)
    pipeline, _ = _build_pipeline(
        mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        checkpoint=ckpt,
        num_links=num_links,
        topology=topology,
    )

    audio_latent = torch.randn(1, 256, 128, dtype=torch.float32)
    num_frames, fps = 145, 24.0

    ref_audio = _decode_audio_reference(ckpt, audio_latent, num_frames, fps=fps)
    assert ref_audio is not None, "host reference audio decode returned None"

    dev_audio = pipeline.decode_audio(audio_latent, num_frames, fps=fps)
    assert dev_audio is not None, "on-device audio decode returned None"
    assert dev_audio.sampling_rate == ref_audio.sampling_rate

    n = min(ref_audio.waveform.shape[-1], dev_audio.waveform.shape[-1])
    ref_w = ref_audio.waveform[..., :n].float()
    dev_w = dev_audio.waveform[..., :n].float()
    psnr = _psnr(ref_w, dev_w)

    logger.info(
        f"Audio decode PSNR (mesh {mesh_shape}): {psnr:.2f} dB "
        f"(ref {tuple(ref_audio.waveform.shape)} -> dev {tuple(dev_audio.waveform.shape)})"
    )
    assert psnr >= 28.0, f"audio decode PSNR {psnr:.2f} dB < 28 dB"


# ===========================================================================
# Component stage tests (pipeline_ltx.py audio decode chain)
#
# Torch references use diffusers LTX-2 audio modules (same as transformer/rope tests).
# Stage/component IDs match ``pipeline_ltx.py`` audio decode chain:
# - Stage A: ``audio_decoder`` (mel-VAE)
# - Stage B: ``vocoder`` (24 kHz waveform)
# - Stage C: ``vocoder_with_bwe`` (48 kHz BWE path)
# ===========================================================================
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
# Component helpers
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
    """Invert diffusers LTX-2.3 vocoder key names to the TT/Lightricks layout."""
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
    """Match the Lightricks MelDecoder / TT denormalize + decoder forward."""
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


def _audio_parallel_config(mesh_shape: tuple[int, int]) -> AudioTCParallelConfig | ParallelFactor | None:
    t_axis = 0 if mesh_shape[0] >= mesh_shape[1] else 1
    t_factor = mesh_shape[t_axis]
    c_axis = 1 - t_axis
    c_factor = mesh_shape[c_axis]
    if t_factor > 1 and c_factor > 1:
        return AudioTCParallelConfig(
            time_parallel=ParallelFactor(factor=t_factor, mesh_axis=t_axis),
            channel_parallel=ParallelFactor(factor=c_factor, mesh_axis=c_axis),
        )
    if t_factor > 1:
        return ParallelFactor(factor=t_factor, mesh_axis=t_axis)
    return None


def _build_tt_stage_b(mesh_device: ttnn.MeshDevice, *, parallel_config, ccl_manager: CCLManager | None) -> Vocoder:
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


def _build_torch_stage_c_real(checkpoint_name: str):
    """diffusers ``LTX2VocoderWithBWE`` with the *real* checkpoint weights — the torch oracle
    for the vocoder+BWE. The checkpoint uses the original BigVGAN key names; invert the
    diffusers→TT rename (see ``_diffusers_vocoder_state_to_tt``) to load them."""
    from safetensors import safe_open

    full = _build_torch_stage_c(seed=0)  # correct architecture; weights overwritten below

    def ckpt_to_diffusers(s: str) -> str:
        return (
            s.replace("conv_pre.", "conv_in.")
            .replace("conv_post.", "conv_out.")
            .replace("act_post.", "act_out.")
            .replace("ups.", "upsamplers.")
            .replace("resblocks.", "resnets.")
            .replace("downsample.lowpass.filter", "downsample.filter")
        )

    state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_name, framework="pt") as f:
        for k in f.keys():
            if k.startswith("vocoder."):
                state[ckpt_to_diffusers(k[len("vocoder.") :])] = f.get_tensor(k).float()
    res = full.load_state_dict(state, strict=False)
    assert (
        not res.missing_keys and not res.unexpected_keys
    ), f"torch oracle weight load mismatch: {len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected"
    return full.eval()


def _build_torch_audio_decoder_real(checkpoint_name: str):
    """diffusers ``AutoencoderKLLTX2Audio`` (mel-VAE decoder) with the *real* checkpoint weights —
    the host oracle for the mel-VAE stage. Architecture dims are read from the checkpoint's
    embedded config, the same source ``LTXAudioDecoderAdapter`` uses, so this tracks the
    real audio VAE rather than the toy component-test config.

    Returns ``(vae, stats_std, stats_mean, z_channels)`` ready for ``_decode_latent_ref``.
    """
    import json

    from safetensors import safe_open

    AutoencoderKLLTX2Audio, _, _ = _require_diffusers()

    state: dict[str, torch.Tensor] = {}
    stats_std = stats_mean = None
    with safe_open(checkpoint_name, framework="pt") as f:
        config = json.loads(f.metadata()["config"])
        ad = config["audio_vae"]["model"]["params"]
        ddconfig = ad["ddconfig"]
        stft_cfg = config["audio_vae"].get("preprocessing", {}).get("stft", {})
        mel_cfg = config["audio_vae"].get("preprocessing", {}).get("mel", {})
        mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels")
        for k in f.keys():
            if k.startswith("audio_vae.decoder."):
                state[k[len("audio_vae.") :]] = f.get_tensor(k).float()  # -> "decoder.*"
            elif k == "audio_vae.per_channel_statistics.std-of-means":
                stats_std = f.get_tensor(k).float()
            elif k == "audio_vae.per_channel_statistics.mean-of-means":
                stats_mean = f.get_tensor(k).float()

    z_channels = ddconfig.get("z_channels", 8)
    vae = AutoencoderKLLTX2Audio(
        base_channels=ddconfig.get("ch", 128),
        output_channels=ddconfig.get("out_ch", 2),
        ch_mult=tuple(ddconfig.get("ch_mult", (1, 2, 4))),
        num_res_blocks=ddconfig.get("num_res_blocks", 2),
        attn_resolutions=tuple(ddconfig.get("attn_resolutions", ())),
        resolution=ddconfig.get("resolution", 256),
        latent_channels=z_channels,
        norm_type="pixel",
        causality_axis="height",
        mid_block_add_attention=ddconfig.get("mid_block_add_attention", False),
        sample_rate=ad.get("sampling_rate", 16000),
        mel_hop_length=stft_cfg.get("hop_length", 160),
        is_causal=stft_cfg.get("causal", True),
        mel_bins=mel_bins,
    ).eval()

    # Decoder-only load: the checkpoint carries no encoder for this path, so encoder keys are
    # legitimately missing — only decoder keys must be satisfied, with nothing left unexpected.
    res = vae.load_state_dict(state, strict=False)
    missing_decoder = [k for k in res.missing_keys if k.startswith("decoder.")]
    assert not missing_decoder and not res.unexpected_keys, (
        f"audio decoder oracle weight load mismatch: {len(missing_decoder)} decoder keys missing, "
        f"{len(res.unexpected_keys)} unexpected"
    )

    if stats_std is None or stats_mean is None:
        # Checkpoint without denormalize stats → identity (matches the component-test default).
        stats_std = torch.ones(z_channels * mel_bins)
        stats_mean = torch.zeros(z_channels * mel_bins)
    return vae, stats_std, stats_mean, z_channels


def _decode_audio_reference(checkpoint_name: str, audio_latent: torch.Tensor, num_frames: int, fps: float) -> Audio:
    """Host (torch) reference for the full audio decode chain — mel-VAE → vocoder+BWE — using the
    real checkpoint weights. Mirrors ``LTXPipeline.decode_audio`` on CPU so the e2e PSNR test can
    compare the on-device decode against a torch oracle instead of skipping when the pipeline
    revision doesn't expose its own reference decode.

    Returns an ``Audio(waveform=(C, T), sampling_rate=...)`` matching the device output layout.
    """
    vae, stats_std, stats_mean, z = _build_torch_audio_decoder_real(checkpoint_name)
    vocoder_with_bwe = _build_torch_stage_c_real(checkpoint_name)

    # Unpatchify exactly as decode_audio does: (1, audio_N, 128) -> (1, z, audio_N, 128 // z).
    audio_N = audio_latent.shape[1]
    audio_spatial = audio_latent.reshape(1, audio_N, z, audio_latent.shape[2] // z).permute(0, 2, 1, 3).float()

    with torch.no_grad():
        mel = _decode_latent_ref(vae, audio_spatial, stats_std=stats_std, stats_mean=stats_mean)
        waveform = vocoder_with_bwe(mel).squeeze(0).float()
    sampling_rate = _OUTPUT_SR

    # Trim to the clip duration, matching decode_audio.
    target_samples = int((num_frames / fps) * sampling_rate)
    if waveform.shape[-1] > target_samples:
        waveform = waveform[..., :target_samples]
    return Audio(waveform=waveform, sampling_rate=sampling_rate)


def _build_tt_stage_c(
    mesh_device: ttnn.MeshDevice, *, parallel_config, ccl_manager: CCLManager | None
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
    # Mirror the pipeline (LTXAudioDecoderAdapter): channel-TP the BWE generator only
    # where the channel axis pays for the gather (factor > 2); single-axis on 2x4 (factor 2).
    if isinstance(parallel_config, AudioTCParallelConfig):
        bwe_pc = parallel_config if parallel_config.channel_parallel.factor > 2 else parallel_config.time_parallel
    else:
        bwe_pc = parallel_config
    bwe_voc = Vocoder(
        mesh_device=mesh_device,
        dtype=ttnn.float32,
        in_channels=128,
        out_channels=2,
        parallel_config=bwe_pc,
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
# Component stage tests
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
    LTX_AUDIO_MESH_PARAMS_FULL,
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

    skip_if_unsupported_num_links(mesh_device, num_links)
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

    tt_decoder = MelDecoder(mesh_device=mesh_device, **_AUDIO_DECODER_CFG)
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
    LTX_AUDIO_MESH_PARAMS_FULL,
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
    skip_if_unsupported_num_links(mesh_device, num_links)
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
    LTX_AUDIO_MESH_PARAMS_FULL,
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
    skip_if_unsupported_num_links(mesh_device, num_links)
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


# Localized audio regression: aggregate PCC over a full clip masks a brief burst, so this
# drives the real ~6s length and gates each 0.5s window vs the unsharded eager baseline.
_STATIC_WIN_S = 0.5


def _window_error_rows(ref: torch.Tensor, tt: torch.Tensor, *, sr: int, win_s: float):
    """Per-window (start_s, max|Δ|, rmse/σ_ref) over the time axis of (B, C, T)."""
    T = ref.shape[-1]
    n = max(1, int(win_s * sr))
    rows = []
    for start in range(0, T, n):
        r = ref[..., start : start + n]
        t = tt[..., start : start + n]
        max_abs = (r - t).abs().max().item()
        rmse = ((r - t) ** 2).mean().sqrt().item()
        sigma = ((r**2).mean().sqrt().item()) + 1e-9
        rows.append((start / sr, max_abs, rmse / sigma))
    return rows


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 32768,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 300000000,
            "require_exact_physical_num_devices": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_stage_c_audio_decode(mesh_device, device_params):
    """End-to-end audio synthesis (VocoderWithBWE): channel-TP main vocoder + conv1d depthwise,
    gated against the unsharded eager baseline per 0.5s window so a localized burst (the demo
    static) can't hide behind an aggregate PCC. Drives the real ~6s clip length (601 mel
    frames)."""
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(2, 4))
    pc = AudioTCParallelConfig(
        time_parallel=ParallelFactor(factor=4, mesh_axis=1),
        channel_parallel=ParallelFactor(factor=2, mesh_axis=0),
    )
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    torch_full = _build_torch_stage_c(seed=42)
    tt_full = _build_tt_stage_c(mesh_device, parallel_config=pc, ccl_manager=ccl)
    tt_full.load_torch_state_dict(_diffusers_vocoder_with_bwe_state_to_tt(torch_full.state_dict()))
    tt_un = _build_tt_stage_c(mesh_device, parallel_config=None, ccl_manager=None)
    tt_un.load_torch_state_dict(_diffusers_vocoder_with_bwe_state_to_tt(torch_full.state_dict()))

    torch.manual_seed(7)
    _tf = int(os.environ.get("AUDIO_T_FRAMES", "601"))  # 601 mel frames ≈ the 6s girl clip
    mel = torch.randn(1, 2, _tf, 64, dtype=torch.float32) * 0.5
    tt = tt_full(mel)
    out_un = tt_un(mel)

    sr = tt_full.output_sampling_rate
    logger.info(f"[audio] output {tuple(tt.shape)} = {tt.shape[-1]/sr:.2f}s @ {sr}Hz")
    su = _window_error_rows(out_un, tt, sr=sr, win_s=_STATIC_WIN_S)
    for s, su_max, su_rmse in su:
        logger.info(f"[audio]  t={s:4.2f}s  shard-vs-un max|Δ|={su_max:.3e} rmse/σ={su_rmse:.3e}")
    worst_su_max = max(r[1] for r in su)
    worst_su_rmse = max(r[2] for r in su)
    print(
        f"\nAUDIO_DECODE shard-vs-un worst_max|Δ|={worst_su_max:.3e} worst_rmse/σ={worst_su_rmse:.3e}",
        flush=True,
    )
    # Regression guard: channel-TP audio must match the unsharded eager TT baseline. Catches
    # broken sharding (the actual risk); torch-vs-TT on the full muxed BWE output is inherently
    # loose with random weights and is checked per-component in test_stage_c_vocoder_with_bwe.
    assert worst_su_max < 5e-3, f"audio window shard-vs-un max|Δ| {worst_su_max:.3e} — divergence"
    assert worst_su_rmse < 1e-2, f"audio window shard-vs-un rmse/σ {worst_su_rmse:.3e} — divergence"


# ===========================================================================
# Tracy per-op device-time profile of the on-device audio decode chain.
#
# Goal: emit every op in the audio decode (mel-VAE -> vocoder -> BWE) with its
# device duration, and have the summed device time land on the ~0.49s traced
# steady-state audio decode (within a small delta).
#
# Why eager (traced=False): each op dispatches as its own program, so (a) the
# on-device profiler can be flushed per-block to stay under the 125-scope/core
# buffer, and (b) every op gets a named DEVICE FW / KERNEL DURATION. The device
# kernel durations are identical to the traced replay — trace only removes host
# dispatch — so the summed device time reproduces the traced ~0.49s while the
# host wall-clock (printed) is far larger; that gap is pure host dispatch.
#
# Two entry points:
#   * test_audio_decode_profile     — the workload (run under `python -m tracy`)
#   * test_audio_decode_perf_table  — driver that runs the profiler on the above
#                                     and prints the per-op table + device total
#
# Manual one-liner (equivalent to the driver):
#   LTX_AUDIO_PROF=1 \
#   LTX_CHECKPOINT=~/.cache/ltx-checkpoints/ltx-2.3-22b-distilled-1.1.safetensors \
#   python -m tracy -p -r -v -o generated/profiler/ltx_audio_decode -m \
#     pytest 'models/tt_dit/tests/models/ltx/test_audio_ltx.py::test_audio_decode_profile' -s
# then sum "DEVICE FW DURATION [ns]" from
#   generated/profiler/ltx_audio_decode/reports/<date>/ops_perf_results_<date>.csv
# ===========================================================================

# The ~0.49s traced steady-state audio decode (4x8 ring, distilled, 145f@24fps),
# expressed in ms; the summed device time should land within _PROF_TOL_MS of it.
_PROF_TARGET_MS = 490.0
_PROF_TOL_MS = 150.0

# Compute-heavy leaf blocks; a ReadDeviceProfiler flush after each keeps every
# Tracy zone window well under the per-core 125-scope device buffer.
_PROF_FLUSH_TYPES = (ResnetBlock, AudioUpsample, AMPBlock1, MelSTFT)


def _walk_tt(module):
    """Depth-first walk over a tt_dit Module tree (root included)."""
    yield module
    for _, child in module.named_children():
        yield from _walk_tt(child)


def _flush_forward_after(module, mesh_device):
    """Wrap ``module.forward`` so the on-device profiler is drained after every call,
    keeping each Tracy zone window under the per-core scope buffer (see gemma harness)."""
    orig = module.forward

    def timed(*a, _orig=orig, **k):
        r = _orig(*a, **k)
        ttnn.ReadDeviceProfiler(mesh_device)
        return r

    module.forward = timed


@pytest.mark.skipif(
    os.environ.get("LTX_AUDIO_PROF") != "1",
    reason="Tracy profiling workload — run via test_audio_decode_perf_table, or manually with "
    "LTX_AUDIO_PROF=1 python -m tracy -p -r -o <out> -m pytest '<file>::test_audio_decode_profile' -s",
)
# Profiler-instrumented kernels compile fresh on the first profiling run and the eager decode
# runs op-by-op, so this comfortably exceeds the default 300s per-test cap.
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 32768,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 300_000_000,
            "require_exact_physical_num_devices": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_audio_decode_profile(mesh_device, device_params):
    """Eager on-device audio decode wrapped in Tracy signposts, one profiled iteration.

    Mirrors the 4x8 ring distilled pipeline's audio-decode config (T-shard=8, replicated
    channel; the pipeline default with LTX_AUDIO_CHANNEL_TP off). Emits ``signpost("start")``
    / ``signpost("stop")`` around a single decode so ``post_process_ops_log(has_signposts=True)``
    slices out warmup/compile; per-block ReadDeviceProfiler flushes keep the device buffer live.
    Pure timing — no PCC (correctness is covered by the sibling e2e/component tests)."""
    ckpt = _resolve_checkpoint(
        [f"{_LOCAL_CHECKPOINT_DIR}/{_DISTILLED_CHECKPOINT}"],
        (_DISTILLED_CHECKPOINT,),
    )
    if ckpt is None:
        pytest.skip("distilled checkpoint not found (set LTX_CHECKPOINT)")

    from tracy import signpost

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(4, 8))

    torch.manual_seed(0)
    # traced=False (LTXPipeline default): decode_audio runs eager, one program per op.
    pipeline, _ = _build_pipeline(
        mesh_device,
        sp_axis=1,
        tp_axis=0,
        checkpoint=ckpt,
        num_links=2,
        topology=ttnn.Topology.Ring,
    )
    assert not pipeline._traced, "profile expects the eager decode path (per-op programs)"

    audio_latent = torch.randn(1, 256, 128, dtype=torch.float32)
    num_frames, fps = 145, 24.0

    # Warmup: compile kernels + one-time device weight prep, unprofiled. Two iters so the
    # program cache and any lazy device state are fully warm before the measured pass.
    for _ in range(2):
        assert pipeline.decode_audio(audio_latent, num_frames, fps=fps) is not None
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)  # drain warmup zones (they precede the "start" marker)

    # Install per-block flushes only for the measured pass.
    for root in (pipeline.tt_mel_decoder, pipeline.tt_vocoder_with_bwe):
        for mod in _walk_tt(root):
            if isinstance(mod, _PROF_FLUSH_TYPES):
                _flush_forward_after(mod, mesh_device)

    # Capture the device-op-id range around the single forward. GLOBAL CALL COUNT in the C++
    # device digest (cpp_device_perf_report.csv) == this counter, so [op0, op1) slices the digest
    # to exactly this forward — no dependence on the fragile Tracy -r ops report or signpost rows.
    from ttnn import _ttnn

    op0 = _ttnn.get_device_operation_id()
    signpost("start")
    t0 = time.perf_counter()
    out = pipeline.decode_audio(audio_latent, num_frames, fps=fps)
    ttnn.synchronize_device(mesh_device)
    host_wall_ms = (time.perf_counter() - t0) * 1000.0
    ttnn.ReadDeviceProfiler(mesh_device)  # flush the tail ops while still inside the region
    signpost("stop")
    op1 = _ttnn.get_device_operation_id()

    assert out is not None
    logger.info(f"Audio decode profiled: host_wall={host_wall_ms:.1f}ms wave={tuple(out.waveform.shape)}")
    print(f"\nAUDIO_DECODE_HOST_WALL_MS={host_wall_ms:.2f}", flush=True)
    print(f"AUDIO_DECODE_OPID_RANGE={op0},{op1}", flush=True)


@pytest.mark.skipif(
    os.environ.get("LTX_AUDIO_PROF") != "1",
    reason="Tracy driver — run: LTX_AUDIO_PROF=1 LTX_CHECKPOINT=<distilled> pytest "
    "'<file>::test_audio_decode_perf_table' -s",
)
@pytest.mark.timeout(3600)
def test_audio_decode_perf_table():
    """Profile ``test_audio_decode_profile`` and print the per-op device-time table.

    Spawns ``python -m tracy`` on the workload (opening the device itself; this driver does not
    take the mesh_device fixture, matching test_ring_joint_sdpa_perf_table), then reads the
    signpost-sliced ops log. Prints one row per op (device FW/kernel duration) plus the summed
    device time, and asserts the total lands within ``_PROF_TOL_MS`` of the ~0.49s traced target.
    """
    import pandas as pd
    from tracy.process_model_log import post_process_ops_log, run_device_profiler

    subdir = "ltx_audio_decode"
    test_id = "models/tt_dit/tests/models/ltx/test_audio_ltx.py::test_audio_decode_profile"
    run_device_profiler(
        f"-m 'pytest {test_id} -s'",
        subdir,
        check_test_return_code=True,
        is_command_binary_exe=True,
    )

    df = post_process_ops_log(subdir, has_signposts=True)  # DataFrame sliced between start/stop
    fw_col, k_col = "DEVICE FW DURATION [ns]", "DEVICE KERNEL DURATION [ns]"

    def _num(series):
        return pd.to_numeric(series[series != "-"], errors="coerce").fillna(0.0)

    fw_ns = _num(df[fw_col])
    k_ns = _num(df[k_col])
    total_fw_ms = fw_ns.sum() / 1e6
    total_k_ms = k_ns.sum() / 1e6

    # Per-op table, ops in dispatch order.
    print("\n" + "=" * 78, flush=True)
    print(f"{'#':>3}  {'OP CODE':<34}{'FW [µs]':>12}{'KERNEL [µs]':>14}", flush=True)
    print("-" * 78, flush=True)
    codes = df["OP CODE"].tolist()
    fw_list = _num(df[fw_col]).tolist()
    k_list = _num(df[k_col]).tolist()
    for i, (code, fw, kd) in enumerate(zip(codes, fw_list, k_list)):
        print(f"{i:>3}  {str(code):<34}{fw / 1e3:>12.2f}{kd / 1e3:>14.2f}", flush=True)
    print("-" * 78, flush=True)

    # Per-op-code aggregate (sorted by total FW time).
    agg = (
        pd.DataFrame({"code": codes, "fw_ns": fw_list, "k_ns": k_list})
        .groupby("code")
        .agg(count=("fw_ns", "size"), fw_us=("fw_ns", lambda s: s.sum() / 1e3), k_us=("k_ns", lambda s: s.sum() / 1e3))
        .sort_values("fw_us", ascending=False)
    )
    print("\nBy op code (FW-descending):", flush=True)
    print(f"{'OP CODE':<34}{'count':>7}{'FW [µs]':>12}{'KERNEL [µs]':>14}", flush=True)
    for code, row in agg.iterrows():
        print(f"{str(code):<34}{int(row['count']):>7}{row['fw_us']:>12.1f}{row['k_us']:>14.1f}", flush=True)
    print("=" * 78, flush=True)

    print(
        f"\nAUDIO_DECODE_DEVICE_FW_MS={total_fw_ms:.2f} "
        f"AUDIO_DECODE_DEVICE_KERNEL_MS={total_k_ms:.2f} "
        f"(target ~{_PROF_TARGET_MS:.0f}ms, tol ±{_PROF_TOL_MS:.0f}ms, n_ops={len(df)})",
        flush=True,
    )
    logger.info(f"Audio decode device time: FW={total_fw_ms:.1f}ms kernel={total_k_ms:.1f}ms over {len(df)} ops")
    assert len(df) > 0, "no ops captured between signposts — check flushing / signpost naming"
    assert abs(total_fw_ms - _PROF_TARGET_MS) <= _PROF_TOL_MS, (
        f"summed device FW {total_fw_ms:.1f}ms is >{_PROF_TOL_MS:.0f}ms from the ~{_PROF_TARGET_MS:.0f}ms "
        f"traced target — device-bound audio-decode time drifted"
    )
