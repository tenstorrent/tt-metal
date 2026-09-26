"""Faithful reproduction of scripts/vocoder_debug_2026_09_20/stage1_eval_v4_noisefix.py's
synthesis (the run that produced the doc's reported 4.17% WER / "Leighton's" vs "Layton's"
miss) -- same dataset sample (REF_IDX=0, TGT_IDX=3), same real min/max-token formula, same
fp32 flow dtype, same eager (untraced) LLM decode. The ONLY thing this script changes is
the WER normalization: old (jiwer's plain Compose, no contraction expansion) vs new
(jiwer.ExpandCommonEnglishContractions added, inserted before lowercase/punctuation removal
since it needs the apostrophe present to recognize a contraction).

Run: PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal
     timeout -s KILL 600 /opt/venv/bin/python rescore_original_stage1.py
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

print("=== loading real LibriSpeech test-clean sample (REF_IDX=0, TGT_IDX=3 -- matches the original eval) ===")
from datasets import load_dataset

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
REF_IDX, TGT_IDX = 0, 3
ref_ex, tgt_ex = ds[REF_IDX], ds[TGT_IDX]
ref_wav16 = torch.tensor(ref_ex["audio"]["array"], dtype=torch.float32).unsqueeze(0)
prompt_text = ref_ex["text"].capitalize() + "."
target_text = tgt_ex["text"].capitalize() + "."
print(f"prompt text: {prompt_text!r}")
print(f"target text: {target_text!r}")

ref_wav24 = torchaudio.functional.resample(ref_wav16, 16000, 24000)
prompt_tokens = extract_speech_tokens(st_session, ref_wav16)
prompt_feat = extract_prompt_feat(ref_wav24)
ref_embedding = extract_spk_embedding(campplus_session, ref_wav16)
token_len = min(prompt_feat.shape[1] // 2, prompt_tokens.shape[1])
prompt_feat = prompt_feat[:, : 2 * token_len]
prompt_tokens = prompt_tokens[:, :token_len]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(local_dir)
prompt_text_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long)
target_text_ids = torch.tensor([tokenizer.encode(target_text)], dtype=torch.long)
text_ids = torch.cat([prompt_text_ids, target_text_ids], dim=1)

print("=== opening device ===")
device = ttnn.CreateDevice(0, l1_small_size=65536)

try:
    print("=== building real LLM (TtQwen2LM, eager -- matches original) ===")
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(device, max_batch_size=1, max_seq_len=512, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()

    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    tt_llm = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd)

    target_len = target_text_ids.shape[1]
    min_tokens_real = int(target_len * 2)
    max_tokens_real = int(target_len * 20)
    print(f"real formula: min_tokens={min_tokens_real}, max_tokens={max_tokens_real} (target_text_len={target_len})")
    torch.manual_seed(0)
    speech_tokens = tt_llm.generate(
        text_ids, prompt_speech_ids=prompt_tokens, max_tokens=max_tokens_real, min_tokens=min_tokens_real, sampler="ras", seed=0
    )
    print(f"generated {len(speech_tokens)} speech tokens")

    print("=== building real flow decoder (fp32 -- matches original) ===")
    import models.demos.audio.cosyvoice2.tt.flow.flow as flow_module
    from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtSmallEmbedding as _TtSmallEmbedding

    class _Bf16ForcedEmbedding(_TtSmallEmbedding):
        def __init__(self, device, weight, dtype=None):
            super().__init__(device, weight, dtype=ttnn.bfloat16)

    flow_module.TtSmallEmbedding = _Bf16ForcedEmbedding

    flow_ref = CausalMaskedDiffWithXvecRef.from_checkpoint(flow_sd)
    flow_ref.eval()
    tt_flow = TtCausalMaskedDiffWithXvec(device, flow_ref, dtype=ttnn.float32)

    token = torch.tensor([speech_tokens], dtype=torch.long)
    mel = tt_flow.inference(token, prompt_tokens, prompt_feat, ref_embedding)
    print(f"mel shape {mel.shape}")

    print("=== building real HiFT vocoder (fp32) ===")
    from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
    from models.demos.audio.cosyvoice2.tt.hifigan.generator import (
        TorchHiFTDecodeRef,
        TorchHiFTGeneratorInferenceRef,
        TtHiFTDecoder,
        TtHiFTGenerator,
    )

    decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
    f0_ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor."))
    hift_ref = TorchHiFTGeneratorInferenceRef(
        decode_ref, f0_ref, hift_sd["m_source.l_linear.weight"], hift_sd["m_source.l_linear.bias"]
    )
    dec = TtHiFTDecoder(device, decode_ref, dtype=ttnn.float32)
    tt_gen = TtHiFTGenerator(device, hift_ref, dec, dtype=ttnn.float32)
    mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    wav_dev = tt_gen.inference(mel_dev, mel.shape[1], batch_size=1)
    waveform = ttnn.to_torch(wav_dev).reshape(-1).float()
    print(f"waveform {waveform.shape}")
finally:
    ttnn.CloseDevice(device)

wav_np = waveform.numpy()
from scipy.io import wavfile

OUT_DIR = os.environ.get("OUT_DIR", "/tmp/cosyvoice2_perf_2026_09_22")
os.makedirs(OUT_DIR, exist_ok=True)
wavfile.write(f"{OUT_DIR}/original_stage1_synth.wav", 24000, (np.clip(wav_np, -1, 1) * 32767).astype(np.int16))

print("\n=== WER: old normalization vs. contraction-fixed normalization ===")
import jiwer
import whisper

asr_model = whisper.load_model("base.en")
wav16_for_asr = torchaudio.functional.resample(waveform.unsqueeze(0), 24000, 16000).squeeze(0).numpy()
hypothesis = asr_model.transcribe(wav16_for_asr, language="en", fp16=False)["text"].strip()
print(f"target text:    {target_text!r}")
print(f"ASR hypothesis: {hypothesis!r}")

OLD_NORM = jiwer.Compose(
    [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()]
)
NEW_NORM = jiwer.Compose(
    [
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)
wer_old = jiwer.wer(target_text, hypothesis, truth_transform=OLD_NORM, hypothesis_transform=OLD_NORM) * 100
wer_new = jiwer.wer(target_text, hypothesis, truth_transform=NEW_NORM, hypothesis_transform=NEW_NORM) * 100
print(f"\nWER (old normalization, matches the doc's reported figure's methodology): {wer_old:.2f}%")
print(f"WER (contraction-fixed normalization):                                     {wer_new:.2f}%")
print(f"moved: {'YES' if abs(wer_old - wer_new) > 1e-6 else 'NO'}")
