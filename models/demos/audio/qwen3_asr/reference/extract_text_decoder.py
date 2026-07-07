"""Extract the Qwen3-ASR text decoder into a vanilla Qwen3ForCausalLM checkpoint
(so tt_transformers can load it like any Qwen3 model), and dump decoder-stage
golden for the Phase-3/4 PCC tests.

Strips the `thinker.` / `thinker.model.` prefixes, writes a Qwen3 config.json, and
copies the tokenizer files. Then runs one short transcription and saves the merged
`inputs_embeds` (audio+text), `input_ids`, `position_ids`, prefill logits and the
generated token ids.

Run with the qwen3-asr-eval venv.
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch

# Base Qwen3-ASR-1.7B snapshot dir. Overridable via env so this runs both on the host
# (eval venv, default below) and inside the server container (e.g. /root/.cache/...).
SNAP_BASE = os.environ.get(
    "QWEN3ASR_SNAP_BASE",
    "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots",
)
TEXT_CFG = dict(
    architectures=["Qwen3ForCausalLM"],
    model_type="qwen3",
    hidden_size=2048,
    intermediate_size=6144,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=151936,
    rope_theta=1000000.0,
    max_position_embeddings=65536,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    attention_bias=False,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
    bos_token_id=151643,
    eos_token_id=151645,
)
TOK_FILES = ["merges.txt", "vocab.json", "tokenizer_config.json", "generation_config.json", "chat_template.json"]


def snap_dir():
    return os.path.join(SNAP_BASE, os.listdir(SNAP_BASE)[0])


def extract_checkpoint(out_dir):
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    snap = snap_dir()
    sd = {}
    for f in sorted(glob.glob(snap + "/*.safetensors")):
        from safetensors import safe_open

        with safe_open(f, "pt") as h:
            for k in h.keys():
                if k.startswith("thinker.model."):
                    sd["model." + k[len("thinker.model.") :]] = h.get_tensor(k)
                elif k == "thinker.lm_head.weight":
                    sd["lm_head.weight"] = h.get_tensor(k)
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    json.dump(TEXT_CFG, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    for fn in TOK_FILES:
        src = os.path.join(snap, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fn))
    print(f"[extract] {len(sd)} tensors -> {out_dir} (config.json + tokenizer)")
    return out_dir


def dump_decoder_golden(out_dir, wav, start, dur, language):
    import soundfile as sf
    from qwen_asr import Qwen3ASRModel

    os.makedirs(out_dir, exist_ok=True)
    wrap = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B", dtype=torch.float32, device_map="cpu", max_inference_batch_size=1, max_new_tokens=64
    )
    model = wrap.model
    cap = {}
    # capture text-model inputs (inputs_embeds, position_ids, attention_mask)
    text_model = None
    for n, m in model.named_modules():
        if n.endswith("thinker.model") and m.__class__.__name__ == "Qwen3ASRThinkerTextModel":
            text_model = m
            break

    def pre(mod, a, kw):
        if "inputs_embeds" not in cap:  # first (prefill) call only
            cap["inputs_embeds"] = kw.get("inputs_embeds")
            cap["position_ids"] = kw.get("position_ids")
            cap["attention_mask"] = kw.get("attention_mask")

    h = text_model.register_forward_pre_hook(pre, with_kwargs=True)

    w, sr = sf.read(wav, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    a = int(start * sr)
    b = min(len(w), a + int(dur * sr))
    res = wrap.transcribe(audio=[(w[a:b].copy(), sr)], language=language)
    h.remove()
    txt = res[0].text.strip()

    def s(name, t):
        if t is None:
            print(f"[warn] {name} None")
            return
        t = t.detach().cpu()
        np.save(os.path.join(out_dir, name + ".npy"), t.float().numpy() if t.is_floating_point() else t.numpy())
        print(f"[save] {name} {tuple(t.shape)} {t.dtype}")

    ie = cap.get("inputs_embeds")
    s("inputs_embeds", ie.squeeze(0) if ie is not None and ie.dim() == 3 else ie)
    pid = cap.get("position_ids")
    s("position_ids", pid)
    print(f"[text] {txt!r}")
    json.dump({"text": txt}, open(os.path.join(out_dir, "decoder_meta.json"), "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-out", default="/home/ttuser/ttwork/qwen3_asr_text_decoder")
    ap.add_argument("--golden-out", default="/home/ttuser/ttwork/qwen3_asr_golden")
    ap.add_argument("--wav", default="/tmp/qwen3-asr-eval/audio/patlabor.wav")
    ap.add_argument("--start", type=float, default=30.0)
    ap.add_argument("--dur", type=float, default=12.0)
    ap.add_argument("--language", default="English")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-golden", action="store_true")
    args = ap.parse_args()
    if not args.skip_extract:
        extract_checkpoint(args.ckpt_out)
    if not args.skip_golden:
        dump_decoder_golden(args.golden_out, args.wav, args.start, args.dur, args.language)
