"""Host-side preprocessing for the raw-wav, language-auto TT demo (run in eval venv).

For each clip: build the Qwen3-ASR prompt with NO forced language (auto-detect), run
the processor to get input_ids + mel, and also run the full CPU Qwen3-ASR (the existing
solution) to get the baseline transcription + detected language. Saves a per-clip npz
the TT demo consumes; the TT path then runs MY encoder+decoder on the same input_ids+mel.
"""
import argparse, json, os
import numpy as np
import soundfile as sf
import torch

AUDIO_TOKEN_ID = 151676

CLIPS = [  # (name, wav, start_s, dur_s)
    ("en_jim", "/tmp/qwen3-asr-eval/audio/jim_keller_yt.wav", 60.0, 20.0),
    ("ja_patlabor", "/tmp/qwen3-asr-eval/audio/patlabor.wav", 30.0, 20.0),
]


def load_slice(path, start, dur, sr=16000):
    w, file_sr = sf.read(path, dtype="float32")
    assert file_sr == sr
    if w.ndim > 1:
        w = w.mean(1)
    a = int(start * sr); b = min(len(w), a + int(dur * sr))
    return w[a:b].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/ttuser/ttwork/qwen3_asr_wav")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from qwen_asr import Qwen3ASRModel
    wrap = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.float32,
                                         device_map="cpu", max_inference_batch_size=1,
                                         max_new_tokens=128)
    summary = {}
    for name, path, start, dur in CLIPS:
        wav = load_slice(path, start, dur)
        # processor inputs with NO forced language (auto-detect)
        prompt = wrap._build_text_prompt(context="", force_language=None)
        inputs = wrap.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"][0].cpu().numpy().astype(np.int64)
        feats = inputs["input_features"]
        mel = feats[0].float().cpu().numpy() if feats.dim() == 3 else feats.float().cpu().numpy()
        n_audio = int((input_ids == AUDIO_TOKEN_ID).sum())

        # existing solution: full CPU Qwen3-ASR, auto-detect
        res = wrap.transcribe(audio=[(wav, 16000)], language=None)[0]
        cpu_text, cpu_lang = res.text.strip(), (res.language or "")

        np.savez(os.path.join(args.out, f"{name}.npz"), input_ids=input_ids, mel=mel,
                 prompt_len=len(input_ids))
        summary[name] = {"wav": path, "start": start, "dur": dur,
                         "n_audio_tokens": n_audio, "prompt_len": int(len(input_ids)),
                         "mel_shape": list(mel.shape), "cpu_text": cpu_text, "cpu_lang": cpu_lang}
        print(f"[{name}] mel={mel.shape} ids={input_ids.shape} audio_tok={n_audio} "
              f"lang={cpu_lang!r}\n   CPU: {cpu_text!r}")
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
