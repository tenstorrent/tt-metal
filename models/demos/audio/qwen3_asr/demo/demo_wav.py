"""Raw-wav, language-AUTO end-to-end on TT, vs the existing CPU Qwen3-ASR solution.

Consumes the npz produced by reference/prep_wav.py (input_ids built with NO forced
language + mel). Runs MY TT audio encoder + MY TT Qwen3-1.7B decoder, lets the model
auto-detect the language, and parses "language <Lang><asr_text><text>". Compares the
TT transcription + detected language against the CPU baseline in summary.json.
"""
import glob, json, os, re, sys, time
import numpy as np
import torch
import ttnn
from safetensors import safe_open
from transformers import AutoTokenizer
from models.tt_transformers.tt.model_config import ModelArgs

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "reference"))
sys.path.insert(0, os.path.join(ROOT, "tt"))
import audio_encoder_ref as ref          # noqa: E402
import audio_encoder as tt_enc           # noqa: E402
from qwen3_asr_decoder import Qwen3ASRDecoder  # noqa: E402

WAV_DIR = os.environ.get("WAV_DIR", "/ttwork/qwen3_asr_wav")
CKPT = os.environ.get("HF_MODEL", "/ttwork/qwen3_asr_text_decoder")
SNAP = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots"
AUDIO_TOKEN_ID = 151676


def load_embed_tokens():
    f = os.path.join(CKPT, "model.safetensors")
    with safe_open(f, "pt") as h:
        return h.get_tensor("model.embed_tokens.weight").float()  # (vocab, 2048)


def parse_asr(text):
    """Raw decode -> (language, transcription). Format: 'language <Lang><asr_text><text>'."""
    m = re.search(r"language\s*(.*?)<asr_text>(.*)", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", text.strip()


def main():
    snap = os.path.join(SNAP, os.listdir(SNAP)[0])
    w = ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(CKPT)
    embed = load_embed_tokens()
    summary = json.load(open(os.path.join(WAV_DIR, "summary.json")))

    dev = ttnn.open_device(device_id=0, trace_region_size=200000000, l1_small_size=32768)
    try:
        enc_params = tt_enc.preprocess_weights(w, dev)
        args = ModelArgs(dev, max_batch_size=1, max_seq_len=2048)
        sd = args.load_state_dict()
        model = Qwen3ASRDecoder(args, ttnn.bfloat16, dev, sd,
                                args.weight_cache_path(ttnn.bfloat16), use_paged_kv_cache=False)

        for npz in sorted(glob.glob(os.path.join(WAV_DIR, "*.npz"))):
            name = os.path.basename(npz)[:-4]
            d = np.load(npz)
            input_ids = torch.from_numpy(d["input_ids"]).long()
            mel = torch.from_numpy(d["mel"]).float()

            # full-TT encoder: mel -> TT conv2d frontend -> transformer -> projector
            t0 = time.time()
            audio_embeds = tt_enc.encode_mel(mel, enc_params, dev).float()  # (N,2048)
            inp = embed[input_ids].clone()                                 # (L,2048)
            mask = (input_ids == AUDIO_TOKEN_ID)
            assert int(mask.sum()) == audio_embeds.shape[0], (int(mask.sum()), audio_embeds.shape)
            inp[mask] = audio_embeds
            t_enc = time.time() - t0

            t0 = time.time()
            ids = model.generate(inp.unsqueeze(0), max_new_tokens=128)
            t_dec = time.time() - t0

            raw = tok.decode(ids, skip_special_tokens=False)
            lang, text = parse_asr(raw)
            audio_sec = mel.shape[1] / 100.0
            rtf = (t_enc + t_dec) / audio_sec
            cpu = summary.get(name, {})
            print(f"\n===== {name}  ({audio_sec:.0f}s, RTF {rtf:.3f}, {len(ids)} tok, "
                  f"{len(ids)/t_dec:.0f} tok/s) =====")
            print(f"  TT  lang={lang!r}")
            print(f"  TT  : {text!r}")
            print(f"  CPU lang={cpu.get('cpu_lang')!r}")
            print(f"  CPU : {cpu.get('cpu_text')!r}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
