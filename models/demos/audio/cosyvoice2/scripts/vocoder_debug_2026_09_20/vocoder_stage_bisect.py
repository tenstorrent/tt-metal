"""Stage-by-stage bisection: our own TorchHiFTGeneratorInferenceRef vs. the
real, official, unmodified HiFTGenerator+ConvRNNF0Predictor (same real
hift.pt weights, already proven correct via the vocoder isolation test), on
the identical real mel for our zero-shot test sample. Captures f0 curve,
source excitation, conv_pre output, each upsample-stage's output, and the
pre-iSTFT magnitude/phase to find exactly where they first diverge.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad/real_cosyvoice_pkg")

SCRATCH = "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad"

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict

print("=== loading real hift.pt + our flow decoder's mel for the exact test sample ===")
hift_sd = load_checkpoint_file("hift.pt")
mel = torch.from_numpy(np.load(f"{SCRATCH}/stage1_v3_mel.npy"))  # [1, T, 80] channels-last
mel_cf = mel.transpose(1, 2).float()  # [1, 80, T] channel-first, both sides' convention

print("=== building OUR torch reference (TorchHiFTGeneratorInferenceRef) ===")
from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TorchHiFTGeneratorInferenceRef

decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
f0_ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor."))
hift_ref = TorchHiFTGeneratorInferenceRef(
    decode_ref, f0_ref, hift_sd["m_source.l_linear.weight"], hift_sd["m_source.l_linear.bias"]
)
decode_ref.eval()
f0_ref.eval()

print("=== building the REAL, unmodified HiFTGenerator + ConvRNNF0Predictor ===")
from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
from cosyvoice.hifigan.generator import HiFTGenerator

f0_predictor = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
generator = HiFTGenerator(
    in_channels=80,
    base_channels=512,
    nb_harmonics=8,
    sampling_rate=24000,
    nsf_alpha=0.1,
    nsf_sigma=0.003,
    nsf_voiced_threshold=10,
    upsample_rates=[8, 5, 3],
    upsample_kernel_sizes=[16, 11, 7],
    istft_params={"n_fft": 16, "hop_len": 4},
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    source_resblock_kernel_sizes=[7, 7, 11],
    source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    lrelu_slope=0.1,
    audio_limit=0.99,
    f0_predictor=f0_predictor,
)
missing, unexpected = generator.load_state_dict(hift_sd, strict=False)
print(f"real generator load: missing={len(missing)} unexpected={len(unexpected)}")
generator.eval()

from models.common.utility_functions import comp_pcc


def report(name, ours, theirs):
    if ours.shape != theirs.shape:
        print(f"{name:30} SHAPE MISMATCH ours={tuple(ours.shape)} theirs={tuple(theirs.shape)}")
        return
    _, pcc = comp_pcc(theirs, ours, 0.999999)
    max_diff = (theirs - ours).abs().max().item()
    print(f"{name:30} PCC={pcc:>10} max_diff={max_diff:.6e} ours_std={ours.std().item():.5f} theirs_std={theirs.std().item():.5f}")


with torch.no_grad():
    print("\n=== STAGE 1: f0 curve ===")
    f0_ours = hift_ref.f0_predictor_ref(mel_cf)  # [1, T_mel]
    f0_theirs = generator.f0_predictor(mel_cf)
    report("f0", f0_ours, f0_theirs)

    print("\n=== STAGE 2: source excitation s ===")
    upsample_scale = hift_ref.upsample_scale
    f0_audio_ours = f0_ours.repeat_interleave(upsample_scale, dim=1).unsqueeze(-1)  # [1, T_audio, 1]
    from models.demos.audio.cosyvoice2.tt.hifigan.source import TtSourceModuleHnNSF

    sine_merge_ours, _, _ = TtSourceModuleHnNSF.torch_reference(
        f0_audio_ours,
        hift_ref.source_linear_weight,
        hift_ref.source_linear_bias,
        sampling_rate=hift_ref.sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=hift_ref.harmonic_num,
        sine_amp=hift_ref.sine_amp,
        noise_std=hift_ref.noise_std,
        voiced_threshold=hift_ref.voiced_threshold,
    )
    s_ours_cf = sine_merge_ours.transpose(1, 2)  # [1, 1, T_audio]

    # real path: s = f0_upsamp(f0[:, None]).transpose(1, 2); s, _, _ = m_source(s)
    s_theirs_pre = generator.f0_upsamp(f0_theirs[:, None]).transpose(1, 2)  # [1, T_audio, 1]
    s_theirs, _, _ = generator.m_source(s_theirs_pre)
    s_theirs_cf = s_theirs.transpose(1, 2)  # [1, 1, T_audio]
    report("source excitation s", s_ours_cf, s_theirs_cf)

    print("\n=== STAGE 3: decode() step by step ===")
    # our side
    s_stft_r_o, s_stft_i_o = decode_ref._stft(s_ours_cf.squeeze(1))
    s_stft_o = torch.cat([s_stft_r_o, s_stft_i_o], dim=1)
    x_o = decode_ref.conv_pre(mel_cf)
    report("conv_pre(mel)", x_o, generator.conv_pre(mel_cf))

    s_stft_r_t, s_stft_i_t = generator._stft(s_theirs_cf.squeeze(1))
    s_stft_t = torch.cat([s_stft_r_t, s_stft_i_t], dim=1)
    report("s_stft", s_stft_o, s_stft_t)

    x_t = generator.conv_pre(mel_cf)

    for i in range(decode_ref.num_upsamples):
        x_o = F.leaky_relu(x_o, decode_ref.lrelu_slope)
        x_o = decode_ref.ups[i](x_o)
        x_t = F.leaky_relu(x_t, generator.lrelu_slope)
        x_t = generator.ups[i](x_t)
        report(f"stage{i} post-ups", x_o, x_t)

        if i == decode_ref.num_upsamples - 1:
            x_o = F.pad(x_o, (1, 0), mode="reflect")
            x_t = generator.reflection_pad(x_t)
            report(f"stage{i} post-reflectpad", x_o, x_t)

        si_o = decode_ref.source_downs[i](s_stft_o)
        si_o = decode_ref._resblock_forward(decode_ref.source_resblocks[i], si_o)
        x_o = x_o + si_o

        si_t = generator.source_downs[i](s_stft_t)
        si_t = generator.source_resblocks[i](si_t)
        x_t = x_t + si_t
        report(f"stage{i} post-source-fusion", x_o, x_t)

        xs_o = None
        for j in range(decode_ref.num_kernels):
            out = decode_ref._resblock_forward(decode_ref.resblocks[i * decode_ref.num_kernels + j], x_o)
            xs_o = out if xs_o is None else xs_o + out
        x_o = xs_o / decode_ref.num_kernels

        xs_t = None
        for j in range(generator.num_kernels):
            out = generator.resblocks[i * generator.num_kernels + j](x_t)
            xs_t = out if xs_t is None else xs_t + out
        x_t = xs_t / generator.num_kernels
        report(f"stage{i} post-resblock-avg", x_o, x_t)

    x_o = F.leaky_relu(x_o)
    x_o = decode_ref.conv_post(x_o)
    x_t = F.leaky_relu(x_t)
    x_t = generator.conv_post(x_t)
    report("conv_post", x_o, x_t)

    mag_o = torch.exp(x_o[:, : decode_ref.bins, :])
    phase_o = torch.sin(x_o[:, decode_ref.bins :, :])
    mag_t = torch.exp(x_t[:, : generator.istft_params["n_fft"] // 2 + 1, :])
    phase_t = torch.sin(x_t[:, generator.istft_params["n_fft"] // 2 + 1 :, :])
    report("magnitude", mag_o, mag_t)
    report("phase", phase_o, phase_t)

    wav_o = decode_ref._istft(mag_o, phase_o)
    wav_o = torch.clamp(wav_o, -decode_ref.audio_limit, decode_ref.audio_limit)
    wav_t = generator._istft(mag_t, phase_t)
    wav_t = torch.clamp(wav_t, -generator.audio_limit, generator.audio_limit)
    report("final waveform", wav_o, wav_t)

print("\n=== sanity: full inference() call, both sides, top-level ===")
with torch.no_grad():
    wav_full_ours = hift_ref.inference(mel)
    wav_full_theirs, _ = generator.inference(speech_feat=mel_cf)
report("inference() waveform", wav_full_ours.reshape(-1), wav_full_theirs.reshape(-1))
