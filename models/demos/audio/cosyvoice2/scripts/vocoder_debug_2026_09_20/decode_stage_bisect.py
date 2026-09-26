"""Stage-by-stage bisect of TtHiFTDecoder.decode vs a float64 torch reference, at the real 464-frame length.

Both sides get the SAME mel and the SAME excitation `s` (computed once in torch from torch f0 + fixed noise N1),
so F0 and the TT source module are out of the picture -- this isolates the decode stack.

For every TT sub-module call we record (input, output) and report
  CUM   : TT output vs torch's own output at that point           (accumulated error)
  LOCAL : TT output vs torch module run on TT's own input         (error THIS stage adds)
Metrics: pcc, gain (= <tt,ref>/<ref,ref>, regression slope; 1.0 = right level), rel (||tt-ref||/||ref||).
"""
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

import ttnn

S = os.environ.get("COSYVOICE2_DEBUG_OUT", "/tmp/cosyvoice2_debug")  # scratch outputs; never the repo
os.makedirs(S, exist_ok=True)
DBG = os.path.dirname(os.path.abspath(__file__))
from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
from models.demos.audio.cosyvoice2.tt.hifigan.generator import (
    TorchHiFTDecodeRef,
    TorchHiFTGeneratorInferenceRef,
    TtHiFTDecoder,
)
from models.demos.audio.cosyvoice2.tt.hifigan.source import TtSourceModuleHnNSF

UP = 480
mel = torch.from_numpy(np.load(f"{DBG}/stage1_v4_mel.npy")).float()  # [1,T,80]
T = mel.shape[1]
L = T * UP
N1 = torch.randn(1, L, 9, generator=torch.Generator().manual_seed(1234))

hift_sd = load_checkpoint_file("hift.pt")
ref32 = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
f0_ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor."))
gen_ref = TorchHiFTGeneratorInferenceRef(
    ref32, f0_ref, hift_sd["m_source.l_linear.weight"], hift_sd["m_source.l_linear.bias"]
)

# shared excitation s [1,1,L] (torch, fp32 -> both sides)
with torch.no_grad():
    f0 = f0_ref(mel.transpose(1, 2)).reshape(1, -1)
    f0a = f0.repeat_interleave(UP, dim=1).unsqueeze(-1)
    sm, _, _ = TtSourceModuleHnNSF.torch_reference(
        f0a,
        gen_ref.source_linear_weight,
        gen_ref.source_linear_bias,
        sampling_rate=24000,
        upsample_scale=UP,
        harmonic_num=8,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=10.0,
        noise=N1,
    )
s_cf = sm.transpose(1, 2).contiguous()  # [1,1,L]
print("s:", tuple(s_cf.shape), "rms", s_cf.pow(2).mean().sqrt().item())

# float64 torch reference
ref = copy.deepcopy(ref32).double()
ref.stft_window = ref32.stft_window.double()
mel64 = mel.double().transpose(1, 2).contiguous()
s64 = s_cf.double()

taps = {}  # torch full-pass outputs


def torch_full():
    with torch.no_grad():
        ss_r, ss_i = ref._stft(s64.squeeze(1))
        ss = torch.cat([ss_r, ss_i], dim=1)
        taps["stft"] = ss
        x = ref.conv_pre(mel64)
        taps["conv_pre"] = x
        for i in range(ref.num_upsamples):
            x = F.leaky_relu(x, ref.lrelu_slope)
            x = ref.ups[i](x)
            taps[f"ups.{i}"] = x
            if i == ref.num_upsamples - 1:
                x = F.pad(x, (1, 0), mode="reflect")
            si = ref.source_downs[i](ss)
            taps[f"source_downs.{i}"] = si
            si = ref._resblock_forward(ref.source_resblocks[i], si)
            taps[f"source_resblocks.{i}"] = si
            x = x + si
            xs = None
            for j in range(ref.num_kernels):
                k = i * ref.num_kernels + j
                out = ref._resblock_forward(ref.resblocks[k], x)
                taps[f"resblocks.{k}"] = out
                xs = out if xs is None else xs + out
            x = xs / ref.num_kernels
            taps[f"stage{i}_x"] = x
        x = F.leaky_relu(x)
        x = ref.conv_post(x)
        taps["conv_post"] = x
        mag = torch.exp(x[:, : ref.bins, :])
        pha = torch.sin(x[:, ref.bins :, :])
        taps["istft"] = ref._istft(mag, pha)
        taps["final"] = torch.clamp(taps["istft"], -ref.audio_limit, ref.audio_limit)


def local_torch(name, ins):
    """torch(float64) module applied to TT's recorded input(s) (channels-first)."""
    with torch.no_grad():
        if name == "stft":
            r, i = ref._stft(ins[0].double().reshape(1, -1))
            return torch.cat([r, i], dim=1)
        if name == "istft":
            re, im = ins[0].double(), ins[1].double()
            return torch.istft(torch.complex(re, im), ref.n_fft, ref.hop_len, ref.n_fft, window=ref.stft_window)
        x = ins[0].double()
        if name == "conv_pre":
            return ref.conv_pre(x)
        kind, idx = name.split(".") if "." in name else (name, None)
        if kind == "ups":
            return ref.ups[int(idx)](x)
        if kind == "source_downs":
            return ref.source_downs[int(idx)](x)
        if kind == "source_resblocks":
            return ref._resblock_forward(ref.source_resblocks[int(idx)], x)
        if kind == "resblocks":
            return ref._resblock_forward(ref.resblocks[int(idx)], x)
        if name == "conv_post":
            return ref.conv_post(x)
    raise KeyError(name)


def stats(a, b):
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    pcc = float(np.corrcoef(a.numpy(), b.numpy())[0, 1]) if b.std() > 0 and a.std() > 0 else float("nan")
    gain = float((a @ b) / (b @ b))
    rel = float((a - b).norm() / b.norm())
    return pcc, gain, rel


def host(t):
    return ttnn.to_torch(t).float().clone()


class Tap:
    def __init__(self, name, obj, rec, nin=1):
        self.name, self.obj, self.rec, self.nin = name, obj, rec, nin

    def __call__(self, *args, **kw):
        ins = [host(a) for a in args[: self.nin]]
        res = self.obj(*args, **kw)
        out = res[0] if isinstance(res, tuple) else res
        self.rec[self.name] = (ins, host(out))
        return res


def to_cf(t, channels):
    """TT channels-last [.., L, C] -> torch channel-first [1, C, L]."""
    return t.reshape(1, -1, channels).transpose(1, 2).contiguous()


torch_full()

dev = ttnn.open_device(device_id=0, l1_small_size=65536)
rec = {}
try:
    dec = TtHiFTDecoder(dev, ref32, dtype=ttnn.float32)
    dec.stft = Tap("stft", dec.stft, rec)
    dec.istft = Tap("istft", dec.istft, rec, nin=2)
    dec.conv_pre = Tap("conv_pre", dec.conv_pre, rec)
    dec.conv_post = Tap("conv_post", dec.conv_post, rec)
    dec.ups = [Tap(f"ups.{i}", m, rec) for i, m in enumerate(dec.ups)]
    dec.source_downs = [Tap(f"source_downs.{i}", m, rec) for i, m in enumerate(dec.source_downs)]
    dec.source_resblocks = [Tap(f"source_resblocks.{i}", m, rec) for i, m in enumerate(dec.source_resblocks)]
    dec.resblocks = [Tap(f"resblocks.{i}", m, rec) for i, m in enumerate(dec.resblocks)]

    mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    s_dev = ttnn.from_torch(s_cf.transpose(1, 2).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    wav = host(dec.decode(mel_dev, s_dev, T, 1)).reshape(-1)
finally:
    ttnn.close_device(dev)

chan = {"conv_pre": 512, "conv_post": 18}
for i in range(3):
    c = 512 // 2 ** (i + 1)
    chan[f"ups.{i}"] = c
    chan[f"source_resblocks.{i}"] = c
    chan[f"source_downs.{i}"] = c
    for j in range(3):
        chan[f"resblocks.{i*3+j}"] = c

print(
    f"\n{'stage':20s} {'shape(tt)':>18s} | {'CUM pcc':>9s} {'gain':>7s} {'rel':>7s} | {'LOCAL pcc':>9s} {'gain':>7s} {'rel':>7s}"
)
order = ["stft", "conv_pre"]
for i in range(3):
    order += [f"ups.{i}", f"source_downs.{i}", f"source_resblocks.{i}"] + [f"resblocks.{i*3+j}" for j in range(3)]
order += ["conv_post", "istft"]
for name in order:
    ins, out = rec[name]
    if name == "stft":
        tt_out = out.reshape(1, 18, -1)
        ins_cf = [ins[0]]
    elif name == "istft":
        tt_out = out.reshape(-1)
        ins_cf = [i.reshape(1, ref.bins, -1) for i in ins]
    else:
        c = chan[name]
        tt_out = to_cf(out, c)
        cin = {"conv_pre": 80, "conv_post": 64}.get(name)
        if name.startswith("source_downs"):
            cin = 18
        elif name.startswith("ups"):
            cin = 512 // 2 ** int(name.split(".")[1])
        elif cin is None:
            cin = c
        ins_cf = [to_cf(ins[0], cin)]
    ref_out = taps[name]
    cum = stats(tt_out, ref_out)
    loc_ref = local_torch(name, ins_cf)
    loc = stats(tt_out, loc_ref)
    print(
        f"{name:20s} {str(tuple(tt_out.shape)):>18s} | {cum[0]:9.5f} {cum[1]:7.4f} {cum[2]:7.4f} | {loc[0]:9.5f} {loc[1]:7.4f} {loc[2]:7.4f}"
    )

fc = stats(wav, taps["final"])
print(
    f"\nFINAL waveform (clamped): CUM pcc {fc[0]:.4f} gain {fc[1]:.4f} rel {fc[2]:.4f};  "
    f"level {20*np.log10(wav.double().pow(2).mean().sqrt()/taps['final'].pow(2).mean().sqrt()):.2f} dB"
)

# ---- dump the source-branch tensors for offline inspection -----------------------------------
torch.save(
    {
        "rec": {
            k: rec[k] for k in ("stft", "source_downs.0", "source_downs.1", "source_downs.2", "source_resblocks.0")
        },
        "taps": {k: taps[k] for k in ("stft", "source_downs.0", "source_downs.1", "source_downs.2")},
        "s": s_cf,
    },
    f"{S}/source_branch_dump.pt",
)
