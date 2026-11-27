# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Masked end padding in HiFT (tt/hifigan/valid_length.py): a call padded to a bucket that reproduces upstream's call
at the real length.

- **The rules, on the host:** torch's HiFT (the real checkpoint) run padded to a bucket and masked by the rules must
  equal torch's HiFT run at the real length. In two parts, because the F0 predictor's convs round differently at
  a different length (up to 7e-3 Hz here), and SineGen2 integrates F0 into the sine phase over the whole utterance,
  so a rounding-level F0 difference grows to ~1e-3 in the waveform by the end: the F0 predictor on its own, to
  rounding; then the source and the decoder with the same F0 injected on both sides, to 1e-5.
- **Stage 1's padded call, on the device, against the reference** (`COSYVOICE2_STREAM_REF`, for its real mels;
  skipped without it): `TtHiFTGenerator.inference_padded`, the pipeline's call for a mel shorter than one chunk,
  against torch's HiFT at the real length. The last 20 ms within 3 dB, with no absolute floor (notes: D41).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from models.experimental.cosyvoice2.tt.hifigan.valid_length import (
    istft_end_gain,
    mask_1d,
    reflect_source_end,
    stft_frames,
)

LAST_20MS = 480  # samples at 24 kHz
END_DB = 3.0  # the end gate: the last 20 ms's RMS level within this of the reference's, with no floor (notes: D41)
STREAM_REF = os.environ.get("COSYVOICE2_STREAM_REF", "")
# (real frames, bucket): the streaming final calls (8 cached + 2 x the final hop) and Stage 1's short mels
HOST_CASES = [(62, 128), (34, 128), (70, 128), (138, 256), (150, 256), (190, 256), (404, 512), (426, 512)]


@pytest.fixture(scope="module")
def torch_hift():
    from models.experimental.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
    from models.experimental.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
    from models.experimental.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TorchHiFTGeneratorInferenceRef

    hift_sd = load_checkpoint_file("hift.pt")
    return TorchHiFTGeneratorInferenceRef(
        TorchHiFTDecodeRef.from_checkpoint(hift_sd),
        TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor.")),
        hift_sd["m_source.l_linear.weight"],
        hift_sd["m_source.l_linear.bias"],
    )


def _mask(length: int, valid: int) -> torch.Tensor:
    return torch.from_numpy(mask_1d(length, valid)).reshape(1, 1, -1)  # channel-first [1, 1, L]


def _masked_resblock(snake, m, x, mask):
    for i in range(len(m.convs1)):
        xt = m.convs1[i](snake(x, m.activations1[i].alpha)) * mask
        xt = m.convs2[i](snake(xt, m.activations2[i].alpha)) * mask
        x = xt + x
    return x


def _source(ref, f0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """SineGen2 + the merge, as `ref.inference` runs them: f0 `[1, T]` per frame -> source `[1, 1, T x 480]`."""
    from models.experimental.cosyvoice2.tt.hifigan.source import TtSourceModuleHnNSF

    s, _, _ = TtSourceModuleHnNSF.torch_reference(
        f0.repeat_interleave(ref.upsample_scale, dim=1).unsqueeze(-1),
        ref.source_linear_weight,
        ref.source_linear_bias,
        sampling_rate=ref.sampling_rate,
        upsample_scale=ref.upsample_scale,
        harmonic_num=ref.harmonic_num,
        sine_amp=ref.sine_amp,
        noise_std=ref.noise_std,
        voiced_threshold=ref.voiced_threshold,
        noise=noise,
    )
    return s.transpose(1, 2)


def masked_f0(f0_ref, mel: torch.Tensor, run: int) -> torch.Tensor:
    """The F0 predictor at `run` frames by the masking rules: zeros past the real frames going in, each conv's output
    zeroed there, and f0 itself. Returns `[1, run]`."""
    valid = mel.shape[1]
    mask = _mask(run, valid)
    x = torch.cat([mel, torch.zeros(1, run - valid, mel.shape[2])], 1).transpose(1, 2)
    for layer in f0_ref.condnet:
        x = layer(x)
        if isinstance(layer, torch.nn.Conv1d):
            x = x * mask
    return torch.abs(f0_ref.classifier(x.transpose(1, 2)).squeeze(-1)) * mask[:, 0]


@torch.no_grad()
def exact_torch_inference(ref, mel: torch.Tensor, noise: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    """`ref.inference` at the real length with `f0` `[1, T]` injected. Returns `[1, T x 480]`."""
    return ref.decode_ref.decode(mel.transpose(1, 2), _source(ref, f0, noise))


@torch.no_grad()
def masked_torch_inference(ref, mel: torch.Tensor, noise: torch.Tensor, run: int, f0: torch.Tensor) -> torch.Tensor:
    """The same, computed at `run` frames by the masking rules: mel `[1, T, 80]` channels-last, noise
    `[1, T x 480, 9]`, f0 `[1, T]`. Returns `[1, T x 480]`."""
    from models.experimental.cosyvoice2.tt.hifigan.snake import TtSnake

    d = ref.decode_ref
    valid, scale = mel.shape[1], ref.upsample_scale
    mel_cf = torch.cat([mel, torch.zeros(1, run - valid, mel.shape[2])], 1).transpose(1, 2)
    mel_mask = _mask(run, valid)
    f0_run = torch.cat([f0, torch.zeros(1, run - valid)], 1)
    noise_run = torch.cat([noise, torch.zeros(1, (run - valid) * scale, noise.shape[2])], 1)
    s = reflect_source_end(_source(ref, f0_run, noise_run).clone(), valid * scale, d.n_fft // 2)

    re, im = d._stft(s.squeeze(1))
    frames_mask = _mask(stft_frames(run, scale, d.hop_len), stft_frames(valid, scale, d.hop_len))
    s_stft = torch.cat([re, im], dim=1) * frames_mask
    x = d.conv_pre(mel_cf) * mel_mask
    rate = 1
    for i in range(d.num_upsamples):
        x = d.ups[i](F.leaky_relu(x, d.lrelu_slope))
        rate *= d.upsample_rates[i]
        length, valid_i = run * rate, valid * rate
        if i == d.num_upsamples - 1:
            x = F.pad(x, (1, 0), mode="reflect")
            length, valid_i = length + 1, valid_i + 1
        mask = _mask(length, valid_i)
        si = _masked_resblock(TtSnake.torch_reference, d.source_resblocks[i], d.source_downs[i](s_stft) * mask, mask)
        x = (x + si) * mask
        outs = [_masked_resblock(TtSnake.torch_reference, d.resblocks[i * d.num_kernels + j], x, mask)
                for j in range(d.num_kernels)]  # fmt: skip
        x = sum(outs) / d.num_kernels
    x = d.conv_post(F.leaky_relu(x))
    wav = d._istft(torch.exp(x[:, : d.bins]) * frames_mask, torch.sin(x[:, d.bins :]))[:, : valid * scale]
    index, gain = istft_end_gain(valid, run, scale, d.n_fft, d.hop_len, d.stft_window.numpy())
    wav[:, index] = wav[:, index] * torch.from_numpy(gain)
    return torch.clamp(wav, -d.audio_limit, d.audio_limit)


def _compare(got: torch.Tensor, want: torch.Tensor) -> dict:
    got, want = got.reshape(-1).double(), want.reshape(-1).double()
    out = {}
    for name, sl in (("whole", slice(None)), ("last 20 ms", slice(-LAST_20MS, None))):
        g, w = got[sl], want[sl]
        out[name] = (float((g - w).abs().max()), float(np.corrcoef(g.numpy(), w.numpy())[0, 1]))
    return out


def _case(ref, valid: int, run: int):
    """A mel in a realistic range (the real F0 predictor finds it voiced, 100s of Hz) and a noise draw."""
    g = torch.Generator().manual_seed(valid * 1000 + run)
    mel = torch.randn(1, valid, 80, generator=g) * 2.0 - 5.0
    return mel, torch.randn(1, valid * ref.upsample_scale, ref.harmonic_num + 1, generator=g)


@pytest.mark.parametrize("valid,run", HOST_CASES)
def test_masked_f0_predictor_matches_the_real_length_on_host(torch_hift, valid, run):
    """The F0 predictor padded and masked equals it at the real length, to its convs' rounding, and is zero past the
    real frames."""
    mel, _ = _case(torch_hift, valid, run)
    with torch.no_grad():
        want = torch_hift.f0_predictor_ref(mel.transpose(1, 2))
        got = masked_f0(torch_hift.f0_predictor_ref, mel, run)
    print(
        f"\n  {valid} of {run} frames: f0 max|diff| {(got[:, :valid] - want).abs().max():.2e} Hz, f0 up to {want.max():.0f} Hz"
    )
    assert torch.allclose(got[:, :valid], want, rtol=1e-4, atol=1e-2)
    assert (got[:, valid:] == 0).all()


@pytest.mark.parametrize("valid,run", HOST_CASES)
def test_masked_source_and_decode_match_the_real_length_on_host(torch_hift, valid, run):
    """With the same F0 on both sides, torch's HiFT padded to `run` and masked equals torch's HiFT at the real
    length: the sine phase's flat end, the source's reflected end, the zeroed conv tails and STFT frames, and the
    iSTFT's end normalization, over the whole waveform and over its last 20 ms."""
    mel, noise = _case(torch_hift, valid, run)
    with torch.no_grad():
        f0 = torch_hift.f0_predictor_ref(mel.transpose(1, 2))
    want = exact_torch_inference(torch_hift, mel, noise, f0)
    got = masked_torch_inference(torch_hift, mel, noise, run, f0)
    stats = _compare(got, want)
    print(
        f"\n  {valid} of {run} frames: "
        + "; ".join(f"{k} max|diff| {d:.2e} PCC {p:.8f}" for k, (d, p) in stats.items())
    )
    assert got.shape == want.shape
    for name, (max_diff, pcc) in stats.items():
        assert max_diff < 1e-5 and pcc > 0.99999, (name, max_diff, pcc)


def _dbfs(x) -> float:
    x = np.asarray(x, np.float64)
    return float(20 * np.log10(max(np.sqrt(np.mean(x * x)), 1e-12)))


@pytest.mark.skipif(not STREAM_REF, reason="set COSYVOICE2_STREAM_REF (scripts/streaming_reference.py's --out-dir)")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
@pytest.mark.timeout(0)  # a device job is never killed mid-op (pytest.ini sets 300 s)
def test_device_stage1_padded_hift_ends_like_the_reference(device, torch_hift):
    """Stage 1's HiFT call for a mel shorter than one chunk (512 frames), padded to its bucket as the pipeline pads it
    (`TtHiFTGenerator.inference_padded`), against torch's HiFT at the real length (the real checkpoint): the last 20
    ms within 3 dB of the reference's, with no floor. Real mels: upstream's non-streaming mel of each corpus case's
    tokens (the streaming reference's `nonstreaming_mel`), those under 512 frames, with one noise draw and torch's F0
    on both sides. PCC over the whole waveform and over its last 20 ms is printed."""
    import ttnn
    from models.experimental.cosyvoice2.tt.hifigan.conv import config_tensors_in_dram_override
    from models.experimental.cosyvoice2.tt.hifigan.generator import TtHiFTDecoder, TtHiFTGenerator
    from models.experimental.cosyvoice2.tt.pipeline import CosyVoice2Config, bucket_at_least

    buckets = CosyVoice2Config.reported().hift_frame_buckets()
    with config_tensors_in_dram_override(True):  # as the pipeline builds it
        decoder = TtHiFTDecoder(device, torch_hift.decode_ref, dtype=ttnn.float32)
        gen = TtHiFTGenerator(device, torch_hift, decoder, dtype=ttnn.float32)
    print("\n| case | real / bucket frames | last 20 ms: ours / reference, dBFS | PCC whole | PCC last 20 ms | max\\|diff\\| |"
          "\n|---|---|---|---|---|---|")  # fmt: skip
    failures, cases = [], 0
    for path in sorted(glob.glob(os.path.join(STREAM_REF, "*.npz"))):
        case = os.path.basename(path)[: -len(".npz")]
        mel = torch.from_numpy(np.load(path)["nonstreaming_mel"]).float()
        frames = int(mel.shape[1])
        if frames >= buckets[-1]:  # 512 frames or more run chunked, unpadded
            continue
        cases += 1
        run = bucket_at_least(frames, buckets)
        g = torch.Generator().manual_seed(frames)
        noise = torch.randn(1, frames * torch_hift.upsample_scale, torch_hift.harmonic_num + 1, generator=g)
        with torch.no_grad():
            f0 = torch_hift.f0_predictor_ref(mel.transpose(1, 2))
        want = exact_torch_inference(torch_hift, mel, noise, f0).reshape(-1).numpy()
        got = gen.inference_padded(mel, noise, run, f0=f0).reshape(-1).numpy()
        assert got.shape == want.shape, (case, got.shape, want.shape)
        got_db, want_db = _dbfs(got[-LAST_20MS:]), _dbfs(want[-LAST_20MS:])
        pcc_whole = float(np.corrcoef(got.astype(np.float64), want.astype(np.float64))[0, 1])
        pcc_end = float(np.corrcoef(got[-LAST_20MS:].astype(np.float64), want[-LAST_20MS:].astype(np.float64))[0, 1])
        print(
            f"| {case} | {frames} / {run} | {got_db:.1f} / {want_db:.1f} | {pcc_whole:.5f} | {pcc_end:.5f} | "
            f"{float(np.abs(got - want).max()):.2e} |"
        )
        if abs(got_db - want_db) > END_DB:
            failures.append(f"{case}: last 20 ms at {got_db:.1f} dBFS, the reference's at {want_db:.1f}")
    assert cases, f"no case under {buckets[-1]} frames in {STREAM_REF}"
    assert not failures, failures
