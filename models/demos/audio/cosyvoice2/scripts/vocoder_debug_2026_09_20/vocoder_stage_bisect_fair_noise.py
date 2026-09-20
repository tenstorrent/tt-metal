"""Re-run the source-excitation comparison FAIRLY: capture the real
SourceModuleHnNSF's own two internal `torch.randn_like` draws (SineGen2's
per-sample sine noise, then the branch noise) via a monkeypatch, and feed
those EXACT same tensors into our own torch_reference call, instead of
letting ours default to zero noise (which is what the naive bisection did,
producing a misleading PCC 0.95 that was actually just "zero noise vs real
random noise", not a formula bug).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad/real_cosyvoice_pkg")

SCRATCH = "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad"

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict

hift_sd = load_checkpoint_file("hift.pt")
mel = torch.from_numpy(np.load(f"{SCRATCH}/stage1_v3_mel.npy"))
mel_cf = mel.transpose(1, 2).float()

from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TorchHiFTGeneratorInferenceRef

decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
f0_ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor."))
hift_ref = TorchHiFTGeneratorInferenceRef(
    decode_ref, f0_ref, hift_sd["m_source.l_linear.weight"], hift_sd["m_source.l_linear.bias"]
)
decode_ref.eval()
f0_ref.eval()

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
generator.load_state_dict(hift_sd, strict=False)
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
    f0_theirs = generator.f0_predictor(mel_cf)
    upsample_scale = hift_ref.upsample_scale
    s_theirs_pre = generator.f0_upsamp(f0_theirs[:, None]).transpose(1, 2)

    # capture the real module's own two randn_like draws, in call order
    captured = []
    real_randn_like = torch.randn_like

    def spy_randn_like(t, *a, **kw):
        v = real_randn_like(t, *a, **kw)
        captured.append(v.clone())
        return v

    torch.randn_like = spy_randn_like
    torch.manual_seed(123)
    try:
        s_theirs, uv_theirs, noise_theirs = generator.m_source(s_theirs_pre)
    finally:
        torch.randn_like = real_randn_like

    print(f"captured {len(captured)} randn_like draws: shapes {[tuple(c.shape) for c in captured]}")
    sine_noise_captured, branch_noise_captured = captured[0], captured[1]

    print("\n=== FAIR comparison: same f0, same real captured noise draws fed into ours ===")
    f0_audio_ours = f0_theirs.repeat_interleave(upsample_scale, dim=1).unsqueeze(-1)

    from models.demos.audio.cosyvoice2.tt.hifigan.source import TtSourceModuleHnNSF

    sine_merge_ours, uv_ours, noise_ours = TtSourceModuleHnNSF.torch_reference(
        f0_audio_ours,
        hift_ref.source_linear_weight,
        hift_ref.source_linear_bias,
        sampling_rate=hift_ref.sampling_rate,
        upsample_scale=upsample_scale,
        harmonic_num=hift_ref.harmonic_num,
        sine_amp=hift_ref.sine_amp,
        noise_std=hift_ref.noise_std,
        voiced_threshold=hift_ref.voiced_threshold,
        noise=sine_noise_captured,
        branch_noise=branch_noise_captured,
    )
    s_ours_cf = sine_merge_ours.transpose(1, 2)
    s_theirs_cf = s_theirs.transpose(1, 2)
    report("source excitation s (fair, same noise)", s_ours_cf, s_theirs_cf)
    report("uv", uv_ours, uv_theirs)
    report("branch noise term", noise_ours, noise_theirs)

    print("\n=== ALSO check: zero-noise-both-sides (isolates the deterministic harmonic signal only) ===")
    sine_merge_ours_z, _, _ = TtSourceModuleHnNSF.torch_reference(
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
    torch.randn_like = lambda t, *a, **kw: torch.zeros_like(t)
    torch.manual_seed(123)
    try:
        s_theirs_z, _, _ = generator.m_source(s_theirs_pre)
    finally:
        torch.randn_like = real_randn_like
    report("source excitation (zero noise both sides)", sine_merge_ours_z.transpose(1, 2), s_theirs_z.transpose(1, 2))
