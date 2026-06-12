"""Root-cause repro #3: replicate the server _infer (processor -> TT encoder -> splice ->
TT decoder) for a [6s,6s,30s,6s,6s] sequence and dump per-stage diagnostics, to locate why
the server returns empty on a 30s request preceded by 6s ones (decoder + encoder each proven
stable in isolation, so the bug is in the integration / a specific length)."""
import os, sys
import numpy as np, torch, ttnn, soundfile as sf
from transformers import AutoTokenizer
from safetensors import safe_open
from qwen_asr.core.transformers_backend import Qwen3ASRProcessor
from models.tt_transformers.tt.model_config import ModelArgs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))
import audio_encoder as tt_enc          # noqa
import audio_encoder_ref as ref         # noqa
from qwen3_asr_decoder import Qwen3ASRDecoder  # noqa

AUD = 151676


def main():
    snap = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots"
    snap = os.path.join(snap, os.listdir(snap)[0])
    w = ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32)
    dev = ttnn.open_device(device_id=0, trace_region_size=200000000, l1_small_size=65536)
    try:
        enc = tt_enc.preprocess_weights(w, dev)
        args = ModelArgs(dev, max_batch_size=1, max_seq_len=2048)
        model = Qwen3ASRDecoder(args, ttnn.bfloat16, dev, args.load_state_dict(),
                                args.weight_cache_path(ttnn.bfloat16), use_paged_kv_cache=False)
        with safe_open("/models/qwen3_asr_text_decoder/model.safetensors", "pt") as h:
            embed = h.get_tensor("model.embed_tokens.weight").float()
        tok = AutoTokenizer.from_pretrained("/models/qwen3_asr_text_decoder")
        proc = Qwen3ASRProcessor.from_pretrained(snap, fix_mistral_regex=True)
        prompt = proc.apply_chat_template(
            [{"role": "system", "content": ""},
             {"role": "user", "content": [{"type": "audio", "audio": ""}]}],
            add_generation_prompt=True, tokenize=False)
        au, sr = sf.read("/audio/yoon.wav", dtype="float32")
        au = au if au.ndim == 1 else au.mean(1)

        FIXED = int(14 * 16000)  # pad/cut every request to a fixed 14s -> constant prefill length
        def infer(start_s):
            real = au[int(start_s*16000): int(start_s*16000)+int(6*16000)]  # 6s real speech
            wav = np.pad(real, (0, FIXED-len(real)))  # pad to fixed 14s with silence
            inp = proc(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
            ids = inp["input_ids"][0].long()
            mel = inp["input_features"][0].float() if inp["input_features"].dim() == 3 else inp["input_features"].float()
            ae = tt_enc.encode_mel(mel, enc, dev).float()
            nmask = int((ids == AUD).sum())
            finite = bool(torch.isfinite(ae).all())
            emb = embed[ids].clone()
            m = (ids == AUD)
            k = min(ae.shape[0], nmask)
            emb[m.nonzero(as_tuple=True)[0][:k]] = ae[:k]
            out = model.generate(emb.unsqueeze(0), max_new_tokens=64)
            txt = tok.decode(out, skip_special_tokens=True)
            print(f"  start={start_s:>3.0f}s: mel={mel.shape[1]:5d} enc={ae.shape[0]:4d} nmask={nmask:4d} "
                  f"match={ae.shape[0]==nmask} finite={finite} first_tok={out[0] if out else None} "
                  f"ntok={len(out)} chars={len(txt)}")
        for st in [0, 14, 28, 42, 7, 21, 35]:   # varied content, all fixed 14s
            infer(st)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
