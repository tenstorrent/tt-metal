"""End-to-end Qwen3-ASR on Tenstorrent: TT AuT audio encoder -> splice audio embeds
into the prompt -> TT Qwen3-1.7B decoder -> transcription. Both models on one P150.

This wires the two TT components validated separately (encoder PCC 0.9934, decoder
prefill PCC 0.9895) into one chain. To avoid re-deriving the processor prompt, the
non-audio prompt embeddings are taken from the golden merged inputs_embeds and the
156 audio rows are REPLACED with this run's TT-encoder output (the reference fills
exactly those rows via masked_scatter), so the only non-golden inputs are the audio
embeds produced on device here.

Run inside the dev container (chip 3 = fake P150):
  docker exec -e TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto \
    -e HF_MODEL=/ttwork/qwen3_asr_text_decoder qwen3asr-dev bash -lc \
    'source /opt/venv/bin/activate && cd /work && \
     python3 models/demos/audio/qwen3_asr/demo/demo.py'
"""
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "tt"))
import audio_encoder as tt_enc  # noqa: E402
import audio_encoder_ref as ref  # noqa: E402
from qwen3_asr_decoder import Qwen3ASRDecoder  # noqa: E402

GOLDEN = os.environ.get("GOLDEN_DIR", "/golden")
CKPT = os.environ.get("HF_MODEL", "/ttwork/qwen3_asr_text_decoder")
SNAP = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots"
REF_TXT = "What's going on? Yako-san alone for the war? Is it? War? That's when it starts. The problem is."


def main():
    snap = os.path.join(SNAP, os.listdir(SNAP)[0])
    w = ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(CKPT)

    # host: mel -> conv frontend + PE
    mel = torch.from_numpy(np.load(f"{GOLDEN}/input_features.npy")).float()
    conv = ref.conv_frontend(mel, w)
    pe = ref.sinusoids(1500, 1024)[: conv.shape[1]]
    x_host = (conv + pe.unsqueeze(0)).reshape(-1, 1024)  # (S_aud, 1024)

    # golden merged embeds + locate audio rows (== golden audio_tower output rows)
    ie = torch.from_numpy(np.load(f"{GOLDEN}/inputs_embeds.npy")).float()  # (174, 2048)
    audio_gold = torch.from_numpy(np.load(f"{GOLDEN}/audio_tower.npy")).float()  # (156, 2048)
    # find the contiguous block of audio rows
    n_aud = audio_gold.shape[0]
    start = None
    for i in range(ie.shape[0] - n_aud + 1):
        if torch.allclose(ie[i : i + n_aud], audio_gold, atol=1e-3):
            start = i
            break
    assert start is not None, "could not locate audio block in inputs_embeds"
    print(f"[splice] audio rows [{start}:{start + n_aud}] of {ie.shape[0]}")

    dev = ttnn.open_device(device_id=0, trace_region_size=200000000)
    try:
        # --- one-time setup: encoder weight preprocessing + decoder build (NOT per-clip) ---
        t0 = time.time()
        enc_params = tt_enc.preprocess_weights(w, dev)
        args = ModelArgs(dev, max_batch_size=1, max_seq_len=1024)
        sd = args.load_state_dict()
        model = Qwen3ASRDecoder(
            args, ttnn.bfloat16, dev, sd, args.weight_cache_path(ttnn.bfloat16), use_paged_kv_cache=False
        )
        t_setup = time.time() - t0

        def run_once():
            ae = tt_enc.encode(x_host, enc_params, dev)  # (156,2048)
            merged = ie.clone()
            merged[start : start + n_aud] = ae.float()
            te = time.time()
            out = model.generate(merged.unsqueeze(0), max_new_tokens=64)
            return out, te

        # warmup pass (first run pays JIT compile), then a timed steady-state pass
        _ = run_once()
        t0 = time.time()
        ids, t_dec_start = run_once()
        t_total = time.time() - t0
        t_enc = t_dec_start - t0
        t_dec = time.time() - t_dec_start
    finally:
        ttnn.close_device(dev)

    txt = tok.decode(ids, skip_special_tokens=True).strip()
    audio_sec = mel.shape[1] / 100.0  # ~100 mel frames/sec
    rtf = t_total / audio_sec
    print(f"[setup]   {t_setup:.2f}s (one-time: weight preprocess + decoder build)")
    print(f"[encoder] {t_enc:.2f}s   [decoder] {t_dec:.2f}s  ({len(ids)} tok, {len(ids)/t_dec:.1f} tok/s)")
    print(f"[audio]   {audio_sec:.1f}s   [steady-state RTF] {rtf:.3f}  (CPU bf16 reference ~0.30)")
    print(f"[TT ]  {txt!r}")
    print(f"[REF]  {REF_TXT!r}")


if __name__ == "__main__":
    main()
