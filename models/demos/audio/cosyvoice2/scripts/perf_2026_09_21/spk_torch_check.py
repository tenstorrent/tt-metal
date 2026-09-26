"""Real Stage 1 measurement, v2 -- both audio-quality fixes applied:
1. generate()'s min_tokens bug fix (now the default in qwen2lm.py -- no
   change needed in this script, just re-running picks it up).
2. Flow decoder switched from bf16 to fp32 (measured PCC 0.9994 -> 0.9997,
   max abs diff 0.571 -> 0.204 against the real torch reference). The
   embedding table stays bf16 (ttnn.embedding hard-requires it, TT_FATAL
   otherwise -- a real platform constraint, not a choice).

Same reference clip / target text / real Stage 1 metrics as the v1 run, so
this is a direct, apples-to-apples before/after comparison.
"""
import os
import sys
import time

import numpy as np
import torch
import torchaudio
import ttnn

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/home/user/tt-metal/models/demos/audio/cosyvoice2/scripts/vocoder_debug_2026_09_20")

from cv2_frontend import extract_prompt_feat, extract_speech_tokens, extract_spk_embedding

from models.demos.audio.cosyvoice2.tt.checkpoint import (
    build_local_qwen2_checkpoint_dir,
    load_checkpoint_file,
    sub_state_dict,
)

SCRATCH = os.environ.get("COSYVOICE2_SCRATCH", "/tmp/cosyvoice2_stage1_eval")

print("=== loading real checkpoints ===")
llm_sd = load_checkpoint_file("llm.pt")
flow_sd = load_checkpoint_file("flow.pt")
hift_sd = load_checkpoint_file("hift.pt")

local_dir = build_local_qwen2_checkpoint_dir(llm_sd, f"{SCRATCH}/qwen2_local_ckpt")
import shutil

from huggingface_hub import hf_hub_download

for fn in ["tokenizer_config.json", "vocab.json", "merges.txt"]:
    p = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename=f"CosyVoice-BlankEN/{fn}")
    shutil.copy(p, f"{local_dir}/{fn}")
os.environ["HF_MODEL"] = local_dir

campplus_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="campplus.onnx")
st_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="speech_tokenizer_v2.onnx")

import onnxruntime as ort

campplus_session = ort.InferenceSession(campplus_path, providers=["CPUExecutionProvider"])
st_session = ort.InferenceSession(st_path, providers=["CPUExecutionProvider"])

print("=== loading real LibriSpeech test-clean sample ===")
from datasets import load_dataset

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
REF_IDX, TGT_IDX = 0, 3
ref_ex, tgt_ex = ds[REF_IDX], ds[TGT_IDX]
ref_wav16 = torch.tensor(ref_ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)
prompt_text = ref_ex["text"].capitalize() + "."
target_text = tgt_ex["text"].capitalize() + "."
print(f"reference audio: speaker {ref_ex['speaker_id']}, text={ref_ex['text']!r}, {ref_wav16.shape[1]/16000:.2f}s")
print(f"prompt text (reference's own transcript): {prompt_text!r}")
print(f"target text (to synthesize in reference's voice): {target_text!r}")

print("=== extracting real prompt features from reference audio ===")
ref_wav24 = torchaudio.functional.resample(ref_wav16, 16000, 24000)
prompt_tokens = extract_speech_tokens(st_session, ref_wav16)
prompt_feat = extract_prompt_feat(ref_wav24)
ref_embedding = extract_spk_embedding(campplus_session, ref_wav16)
print(f"prompt_tokens (raw): {prompt_tokens.shape}, prompt_feat (raw): {prompt_feat.shape}")

# Real upstream frontend.py's frontend_zero_shot, confirmed directly: for
# resample_rate==24000 (CosyVoice2) it forces prompt_feat.shape[1] == 2 *
# prompt_tokens.shape[1] EXACTLY by truncating both to the smaller of the two
# implied lengths -- "force speech_feat % speech_token = 2" in their own
# comment. Skipped in the first pass; real, not cosmetic (the flow decoder's
# conds splice uses prompt_feat.shape[1] directly as mel_len1, so a skewed
# ratio here desyncs the token/mel alignment the model was trained on).
token_len = min(prompt_feat.shape[1] // 2, prompt_tokens.shape[1])
prompt_feat = prompt_feat[:, : 2 * token_len]
prompt_tokens = prompt_tokens[:, :token_len]
print(f"prompt_tokens (truncated 2:1): {prompt_tokens.shape}, prompt_feat: {prompt_feat.shape}")
print(f"ref_embedding: {ref_embedding.shape}")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(local_dir)
# Real upstream Qwen2LM.inference, confirmed directly: `text =
# torch.concat([prompt_text, text], dim=1)` BEFORE embedding -- the LLM is
# always conditioned on the reference's own transcript followed by the target
# text, not the target text alone. Skipped in the first pass; this is almost
# certainly why WER was near 100% (an unconditioned, contextless continuation
# task the model was never trained for).
prompt_text_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long)
target_text_ids = torch.tensor([tokenizer.encode(target_text)], dtype=torch.long)
text_ids = torch.cat([prompt_text_ids, target_text_ids], dim=1)
print(f"prompt_text_ids: {prompt_text_ids.shape[1]}, target_text_ids: {target_text_ids.shape[1]}, combined: {text_ids.shape[1]}")



import json, glob
import soundfile as sf
from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef
from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
from models.demos.audio.cosyvoice2.tt.hifigan.generator import TorchHiFTDecodeRef, TorchHiFTGeneratorInferenceRef

torch.set_num_threads(32)
def spk(w16):  # w16: 1-D float tensor at 16 kHz
    return float(torch.nn.functional.cosine_similarity(ref_embedding, extract_spk_embedding(campplus_session, w16.unsqueeze(0)), dim=1))

toks = json.load(open(os.environ["OUT_DIR"] + "/tokens7.json"))
flow_ref = CausalMaskedDiffWithXvecRef.from_checkpoint(flow_sd); flow_ref.eval()
decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
f0_ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor."))
hift_ref = TorchHiFTGeneratorInferenceRef(decode_ref, f0_ref, hift_sd["m_source.l_linear.weight"], hift_sd["m_source.l_linear.bias"])
out = {}
res = lambda k, v: (out.__setitem__(k, v), print(f"{k}: {v}", flush=True))

res("sanity_ref_vs_itself", round(spk(ref_wav16.squeeze(0)), 4))
# real same-speaker speech, cropped to 4.4 s and 9 s (what a perfect synthesizer could reach on this metric)
for i in range(len(ds)):
    ex = ds[i]
    w = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
    if i == REF_IDX:
        continue
    res(f"real_ds{i}_spk{ex['speaker_id']}_{len(w)/16000:.1f}s_full", round(spk(w), 4))
    for sec in (4.4, 9.0):
        if len(w) / 16000 > sec + 0.3:
            res(f"real_ds{i}_spk{ex['speaker_id']}_crop{sec}s", round(spk(w[: int(sec * 16000)]), 4))
res("ref_speaker_id", ref_ex["speaker_id"])
res("ref_clip_len_s", round(ref_wav16.shape[1] / 16000, 2))
res("ref_clip_first4.4s_vs_ref_full", round(spk(ref_wav16.squeeze(0)[: int(4.4 * 16000)]), 4))

# fully torch flow + torch HiFT on the same tokens, three vocoder noise seeds
token = torch.tensor([toks["s0_door"]["tokens"]], dtype=torch.long)
with torch.no_grad():
    mel = flow_ref.inference(token, prompt_tokens, prompt_feat, ref_embedding)
res("torch_mel_shape", list(mel.shape))
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    with torch.no_grad():
        wav = hift_ref.inference(mel).reshape(-1).float()
    w16 = torchaudio.functional.resample(wav.unsqueeze(0), 24000, 16000).squeeze(0)
    res(f"fully_torch_s0_door_seed{seed}_{wav.shape[0]/24000:.2f}s", round(spk(w16), 4))
    sf.write(os.environ["OUT_DIR"] + f"/listening/fully_torch_s0_door_seed{seed}.wav", wav.numpy(), 24000)

# duration hypothesis on our own TT wavs: first 4.4 s of each longer sentence vs the full sentence
for arm in ("R", "C"):
    for k in ("s1_busstop", "s3_park", "s4_weather", "s5_sister"):
        p = f"/home/user/cosyvoice2_stft_fix_wavs/listening_arms/{arm}/{k}.wav"
        w, sr = sf.read(p); w = torch.tensor(w, dtype=torch.float32)
        w16 = torchaudio.functional.resample(w.unsqueeze(0), sr, 16000).squeeze(0)
        res(f"arm{arm}_{k}_full_{len(w)/sr:.1f}s", round(spk(w16), 4))
        res(f"arm{arm}_{k}_first4.4s", round(spk(w16[: int(4.4 * 16000)]), 4))
json.dump(out, open(os.environ["OUT_DIR"] + "/spk_torch_check.json", "w"), indent=1)
