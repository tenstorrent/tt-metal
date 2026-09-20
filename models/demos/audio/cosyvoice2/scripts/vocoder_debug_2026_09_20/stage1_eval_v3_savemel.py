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
sys.path.insert(0, "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad")

from cv2_frontend import extract_prompt_feat, extract_speech_tokens, extract_spk_embedding

from models.demos.audio.cosyvoice2.tt.checkpoint import (
    build_local_qwen2_checkpoint_dir,
    load_checkpoint_file,
    sub_state_dict,
)

SCRATCH = "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad"

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

print("=== opening device ===")
device = ttnn.CreateDevice(0, l1_small_size=65536)

try:
    print("=== building real LLM (TtQwen2LM) ===")
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(device, max_batch_size=1, max_seq_len=512, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()

    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    tt_llm = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd)

    print("=== TOKEN-LEVEL ACCURACY: TT greedy vs. real HF PyTorch reference ===")
    from models.tt_transformers.tt.common import Mode

    hf_model = args.reference_transformer(wrap=False)

    torch.manual_seed(7)
    n_check_tokens = 20
    with torch.no_grad():
        # host-side reference: real Qwen2ForCausalLM forward, greedy, using the
        # SAME real cosyvoice-specific embedding/head tensors TtQwen2LM uses
        # (llm_embedding/speech_embedding/llm_decoder from llm.pt), not the
        # model's own (unused) lm_head -- an apples-to-apples comparison of the
        # SAME real computation, not two different models.
        hf_dtype = hf_model.dtype
        sos_emb = llm_sd["llm_embedding.weight"][0].reshape(1, 1, -1).to(hf_dtype)
        task_id_emb = llm_sd["llm_embedding.weight"][1].reshape(1, 1, -1).to(hf_dtype)
        text_emb_ref = hf_model.model.embed_tokens(text_ids)
        seq = torch.cat([sos_emb, text_emb_ref, task_id_emb], dim=1)
        ref_tokens = []
        past = None
        cur = seq
        for i in range(n_check_tokens):
            out = hf_model.model(inputs_embeds=cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            h = out.last_hidden_state[:, -1:, :].float()
            logits = h @ llm_sd["llm_decoder.weight"].T + llm_sd["llm_decoder.bias"]
            tok = int(logits[0, -1].argmax())
            ref_tokens.append(tok)
            if tok in (6561, 6562, 6563):
                break
            cur = llm_sd["speech_embedding.weight"][tok].reshape(1, 1, -1).to(hf_dtype)

    tt_tokens = tt_llm.generate(text_ids, prompt_speech_ids=None, max_tokens=n_check_tokens, min_tokens=0, sampler="greedy", seed=7)

    n = min(len(ref_tokens), len(tt_tokens))
    matches = sum(1 for i in range(n) if ref_tokens[i] == tt_tokens[i])
    token_accuracy = matches / max(len(ref_tokens), 1)
    print(f"reference (torch) tokens: {ref_tokens}")
    print(f"TT device tokens:        {tt_tokens}")
    print(f"token-level accuracy: {matches}/{len(ref_tokens)} = {token_accuracy*100:.1f}% (target: >95%)")

    print("\n=== ZERO-SHOT SYNTHESIS (real prompt, real target text) ===")
    # Real upstream Qwen2LM.inference, confirmed directly: min_len/max_len are
    # NOT fixed constants -- `min_len = int((text_len - prompt_text_len) *
    # min_token_text_ratio)`, `max_len = int((text_len - prompt_text_len) *
    # max_token_text_ratio)`, with min_token_text_ratio=2/max_token_text_ratio=20
    # (inference()'s own defaults) applied to the TARGET text length alone
    # (text_len is prompt+target combined by that point, prompt_text_len
    # subtracted back out). My hardcoded max_tokens=200 was well under the real
    # formula's 600 for this sentence (30 target text tokens) -- generation hit
    # that cap with no natural stop, and the one garbled ASR segment was exactly
    # at the tail, consistent with truncation, not a content/quality bug.
    target_len = target_text_ids.shape[1]
    min_tokens_real = int(target_len * 2)
    max_tokens_real = int(target_len * 20)
    print(f"real formula: min_tokens={min_tokens_real}, max_tokens={max_tokens_real} (target_text_len={target_len})")
    t0 = time.time()
    torch.manual_seed(0)
    speech_tokens = tt_llm.generate(
        text_ids, prompt_speech_ids=prompt_tokens, max_tokens=max_tokens_real, min_tokens=min_tokens_real, sampler="ras", seed=0
    )
    llm_time = time.time() - t0
    print(f"generated {len(speech_tokens)} speech tokens in {llm_time:.2f}s")
    assert len(speech_tokens) > 0

    print("=== building real flow decoder (fp32, per the fp32 experiment) ===")
    import models.demos.audio.cosyvoice2.tt.flow.flow as flow_module
    from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtSmallEmbedding as _TtSmallEmbedding

    # ttnn.embedding hard-requires bf16 weights (TT_FATAL otherwise) -- force
    # just the input_embedding table to stay bf16 while everything else in the
    # flow decoder runs fp32.
    class _Bf16ForcedEmbedding(_TtSmallEmbedding):
        def __init__(self, device, weight, dtype=None):
            super().__init__(device, weight, dtype=ttnn.bfloat16)

    flow_module.TtSmallEmbedding = _Bf16ForcedEmbedding

    flow_ref = CausalMaskedDiffWithXvecRef.from_checkpoint(flow_sd)
    flow_ref.eval()
    tt_flow = TtCausalMaskedDiffWithXvec(device, flow_ref, dtype=ttnn.float32)

    token = torch.tensor([speech_tokens], dtype=torch.long)
    t0 = time.time()
    mel = tt_flow.inference(token, prompt_tokens, prompt_feat, ref_embedding)
    flow_time = time.time() - t0
    mel_frames = mel.shape[1]
    print(f"mel shape {mel.shape} in {flow_time:.2f}s")
    np.save(f"{SCRATCH}/stage1_v3_mel.npy", mel.numpy())
    print(f"saved mel to {SCRATCH}/stage1_v3_mel.npy")

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

    t0 = time.time()
    wav_dev = tt_gen.inference(mel_dev, mel_frames, batch_size=1)
    hift_time = time.time() - t0
    waveform = ttnn.to_torch(wav_dev).reshape(-1).float()
    print(f"waveform {waveform.shape} in {hift_time:.2f}s")

finally:
    ttnn.CloseDevice(device)

total_audio_s = waveform.shape[0] / 24000
total_synth_s = llm_time + flow_time + hift_time
rtf = total_synth_s / total_audio_s
print(f"\nRTF (non-streaming, whole-utterance) = {total_synth_s:.2f}s / {total_audio_s:.2f}s = {rtf:.3f} (target: <1.0)")

wav_np = waveform.numpy()
from scipy.io import wavfile

wavfile.write(f"{SCRATCH}/stage1_synth_v3_ourvocoder.wav", 24000, (np.clip(wav_np, -1, 1) * 32767).astype(np.int16))
ref_wav16_np = ref_wav16.squeeze(0).numpy()
wavfile.write(f"{SCRATCH}/stage1_reference.wav", 16000, (np.clip(ref_wav16_np, -1, 1) * 32767).astype(np.int16))

print("\n=== WER (whisper ASR on synthesized audio vs. target text) ===")
import jiwer
import whisper

asr_model = whisper.load_model("base.en")
wav16_for_asr = torchaudio.functional.resample(waveform.unsqueeze(0), 24000, 16000).squeeze(0).numpy()
result = asr_model.transcribe(wav16_for_asr, language="en", fp16=False)
hypothesis = result["text"].strip()
print(f"target text:  {target_text!r}")
print(f"ASR hypothesis: {hypothesis!r}")

transform = jiwer.Compose(
    [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()]
)
wer = jiwer.wer(target_text, hypothesis, truth_transform=transform, hypothesis_transform=transform)
print(f"WER = {wer*100:.2f}% (target: <5.0%)")

print("\n=== SPEAKER SIMILARITY (real campplus xvec, synthesized vs. reference) ===")
wav16_for_spk = torchaudio.functional.resample(waveform.unsqueeze(0), 24000, 16000)
synth_embedding = extract_spk_embedding(campplus_session, wav16_for_spk)
cos_sim = torch.nn.functional.cosine_similarity(ref_embedding, synth_embedding, dim=1).item()
print(f"speaker cosine similarity (synth vs. reference) = {cos_sim:.4f} (target: >0.60)")

print("\n=== FINAL SUMMARY ===")
print(f"token-level accuracy: {token_accuracy*100:.1f}% (target >95%)  -> {'PASS' if token_accuracy > 0.95 else 'FAIL'}")
print(f"WER: {wer*100:.2f}% (target <5.0%)  -> {'PASS' if wer*100 < 5.0 else 'FAIL'}")
print(f"speaker similarity: {cos_sim:.4f} (target >0.60)  -> {'PASS' if cos_sim > 0.60 else 'FAIL'}")
print(f"RTF: {rtf:.3f} (target <1.0)  -> {'PASS' if rtf < 1.0 else 'FAIL'}")
