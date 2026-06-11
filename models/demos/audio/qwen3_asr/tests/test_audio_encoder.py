"""Phase-2 PCC test: ttnn AuT audio encoder vs the saved golden (audio_tower output).

Run inside the dev container (chip 3 as fake P150):
  docker exec -e TT_MESH_GRAPH_DESC_PATH=/work/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    qwen3asr-dev bash -lc 'source /opt/venv/bin/activate && cd /work && \
    python3 models/demos/audio/qwen3_asr/tests/test_audio_encoder.py'
"""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "reference"))
sys.path.insert(0, os.path.join(HERE, "..", "tt"))
import audio_encoder_ref as ref  # noqa: E402
import audio_encoder as tt_enc   # noqa: E402
import ttnn  # noqa: E402

GOLDEN = os.environ.get("GOLDEN_DIR", "/golden")
SNAP = os.environ.get("QWEN3ASR_SNAP",
                      "/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots")


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    snap = os.path.join(SNAP, os.listdir(SNAP)[0]) if os.path.isdir(SNAP) else None
    w = ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32)

    mel = torch.from_numpy(np.load(f"{GOLDEN}/input_features.npy")).float()
    # host conv frontend + positional embedding (validated PCC=1.0 in reference)
    conv = ref.conv_frontend(mel, w)                       # (n_chunks,13,1024)
    pe = ref.sinusoids(1500, 1024)[:conv.shape[1]]
    x_host = (conv + pe.unsqueeze(0)).reshape(-1, 1024)    # (S,1024)

    gold = torch.from_numpy(np.load(f"{GOLDEN}/audio_tower.npy")).float()  # (S,2048)

    dev = ttnn.open_device(device_id=0)
    try:
        params = tt_enc.preprocess_weights(w, dev)
        out = tt_enc.encode(x_host, params, dev)           # (S,2048) torch
    finally:
        ttnn.close_device(dev)

    p = pcc(out, gold)
    print(f"[shapes] tt={tuple(out.shape)} golden={tuple(gold.shape)}")
    print(f"[PCC] audio_tower (ttnn vs golden) = {p:.6f}")
    print("RESULT:", "PASS" if p > 0.99 else "FAIL", f"(threshold 0.99)")
    return 0 if p > 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())
