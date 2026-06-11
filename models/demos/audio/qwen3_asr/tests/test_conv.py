"""Validate the TT conv2d frontend vs golden conv_out (pre-positional-embedding)."""
import os, sys
import numpy as np
import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_weight
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt"))
import audio_encoder_ref as ref     # noqa
import audio_encoder as tt_enc      # noqa

GOLDEN = os.environ.get("GOLDEN_DIR", "/golden")
SNAP = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots"


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    snap = os.path.join(SNAP, os.listdir(SNAP)[0])
    w = ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32)
    mel = torch.from_numpy(np.load(f"{GOLDEN}/input_features.npy")).float()
    gold = torch.from_numpy(np.load(f"{GOLDEN}/conv_out.npy")).float().reshape(-1, 1024)
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        conv_w = tt_enc.preprocess_conv_weights(w, dev)
        conv_out_w = ttnn.to_device(preprocess_linear_weight(w["conv_out.weight"], dtype=ttnn.bfloat16), dev)
        out = tt_enc.conv_frontend_tt(mel, conv_w, conv_out_w, None, dev)
        p = pcc(out, gold)
        print(f"[shapes] tt={tuple(out.shape)} gold={tuple(gold.shape)}")
        print(f"[PCC] conv_out (TT conv2d vs golden) = {p:.6f}")
        print("RESULT:", "PASS" if p > 0.99 else "FAIL")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
