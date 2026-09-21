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


import json
device = ttnn.CreateDevice(0, l1_small_size=65536, trace_region_size=50_000_000)
SENTENCES = [
    ("s0_door", "Please close the door when you leave."),
    ("s1_busstop", "Can you tell me where the nearest bus stop is?"),
    ("s2_walk", "I usually walk to work in the morning, and then I have a cup of coffee."),
    ("s3_park", "We are going to the park this weekend, and the kids want to bring their bikes and a big picnic lunch."),
    ("s4_weather", "The weather was nice yesterday, so we sat outside for a while and talked about our plans for the summer holidays."),
    ("s5_sister", "My sister called me last night to tell me about her new job. She likes her team, the office is close to her house, and she can finally take the train instead of driving every day."),
    ("s6_dinner", "After dinner we washed the dishes together, watched a short movie, and went to bed early because we had to get up before sunrise the next morning."),
]

try:
    print("=== building real LLM (TtQwen2LM) ===")
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(device, max_batch_size=1, max_seq_len=512, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()

    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    tt_llm = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd, use_decode_trace=True)

    out = {}
    for key, text in SENTENCES:
        tgt = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
        ids = torch.cat([prompt_text_ids, tgt], dim=1); tl = tgt.shape[1]
        toks = tt_llm.generate(ids, prompt_speech_ids=prompt_tokens, max_tokens=int(tl * 20), min_tokens=int(tl * 2), sampler="ras", seed=0)
        out[key] = {"text": text, "tokens": toks}
        print(key, len(toks), "tokens", flush=True)
    json.dump(out, open(os.environ["OUT_DIR"] + "/tokens7.json", "w"))
    print("saved")
finally:
    tt_llm.release_decode_trace()
    ttnn.CloseDevice(device)
