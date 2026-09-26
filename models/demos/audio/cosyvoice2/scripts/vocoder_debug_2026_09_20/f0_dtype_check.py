"""F0 predictor device-vs-torch on the REAL 464-frame mel, bf16 vs fp32.

Reports PCC / max err, plus what the error means for audibility:
  * cents of pitch error per frame (100 cents = 1 semitone; ~5-10 cents is the
    audible-detuning threshold for steady tones)
  * total accumulated fundamental phase drift in cycles over the utterance
    (SineGen2 integrates f0/sr per audio sample; upsample_scale=480 samples/frame)
"""

import os

import numpy as np
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef, TtConvRNNF0Predictor

OUT = os.environ.get("COSYVOICE2_DEBUG_OUT", "/tmp/cosyvoice2_debug")  # scratch outputs; never the repo
os.makedirs(OUT, exist_ok=True)
MEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1_v4_mel.npy")
SR, UP = 24000, 480

mel = np.load(MEL)
print("mel npy shape", mel.shape, mel.dtype)
mel = torch.from_numpy(mel).float()
if mel.dim() == 2:
    mel = mel.unsqueeze(0)
if mel.shape[1] == 80 and mel.shape[2] != 80:  # channel-first -> channels-last
    mel = mel.transpose(1, 2)
T = mel.shape[1]
print("mel channels-last", tuple(mel.shape), "frames", T)

ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(load_checkpoint_file("hift.pt"), "f0_predictor."))
with torch.no_grad():
    want = ref(mel.transpose(1, 2)).reshape(-1).double()
print(f"torch f0: min {want.min():.2f} max {want.max():.2f} mean {want.mean():.2f}")

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    for name, dt, hf in [("bf16", ttnn.bfloat16, True), ("fp32", ttnn.float32, True)]:
        tt = TtConvRNNF0Predictor(dev, ref, dtype=dt, high_fidelity=hf)
        x = ttnn.from_torch(mel, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        got = ttnn.to_torch(tt(x, T, batch_size=1)).reshape(-1).double()
        err = got - want
        _, pcc = comp_pcc(want.float(), got.float(), 0.0)
        # pitch error in cents only where voiced (f0 > 50 Hz)
        v = want > 50
        cents = 1200 * torch.log2(got[v].clamp(min=1e-3) / want[v])
        # accumulated fundamental phase drift, cycles (each frame covers UP samples)
        drift_cycles = torch.cumsum(err * UP / SR, 0)
        print(f"\n[{name}] PCC {pcc}")
        print(
            f"  abs err Hz : max {err.abs().max():.4f}  mean {err.abs().mean():.5f}  p99 {err.abs().quantile(0.99):.4f}"
        )
        print(
            f"  frames with |err| > 0.1 Hz: {(err.abs() > 0.1).sum().item()} / {T};  > 1 Hz: {(err.abs() > 1).sum().item()}"
        )
        print(
            f"  cents (voiced {v.sum().item()} frames): max |{cents.abs().max():.2f}|  p99 {cents.abs().quantile(0.99):.2f}  mean {cents.abs().mean():.3f}"
        )
        print(
            f"  phase drift (cycles of fundamental): final {drift_cycles[-1]:.3f}  max |{drift_cycles.abs().max():.3f}|"
        )
        top = err.abs().topk(5)
        print("  worst frames:", [(int(i), round(float(want[i]), 1), round(float(err[i]), 3)) for i in top.indices])
        np.save(f"{OUT}/f0_dev_{name}.npy", got.numpy())
        ttnn.deallocate(x)
    np.save(f"{OUT}/f0_torch.npy", want.numpy())
finally:
    ttnn.close_device(dev)
