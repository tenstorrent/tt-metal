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


print("=== opening device ===")
device = ttnn.CreateDevice(0, l1_small_size=65536, trace_region_size=50_000_000)
N_RUNS = int(os.environ.get("N_RUNS", "3"))

def mem():
    l1 = ttnn.get_memory_view(device, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank / 1024
    dr = ttnn.get_memory_view(device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank / 1024
    return l1, dr


def sync():
    ttnn.synchronize_device(device)


class Timed:
    """Accumulates device-synchronised wall time of every call to `fn`."""

    def __init__(self, fn):
        self.fn, self.total, self.calls = fn, 0.0, 0

    def __call__(self, *a, **k):
        sync()
        t0 = time.perf_counter()
        r = self.fn(*a, **k)
        sync()
        self.total += time.perf_counter() - t0
        self.calls += 1
        return r

    def __getattr__(self, name):  # only called for attributes not found on the wrapper -> delegate
        return getattr(self.fn, name)

    def reset(self):
        self.total, self.calls = 0.0, 0


try:
    print("=== building real LLM (TtQwen2LM) ===")
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(device, max_batch_size=1, max_seq_len=512, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()

    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    tt_llm = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd, use_decode_trace=True)
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

    # ---- instrument the sub-stages (wrap, never modify the repo) ----
    t_enc = Timed(tt_flow.encoder)
    tt_flow.encoder = t_enc
    t_est = Timed(tt_flow.decoder.estimator)
    tt_flow.decoder.estimator = t_est
    t_f0 = Timed(tt_gen.f0_predictor)
    tt_gen.f0_predictor = t_f0
    t_src = Timed(tt_gen.source)
    tt_gen.source = t_src
    t_dec = Timed(tt_gen.decoder.decode)
    tt_gen.decoder.decode = t_dec
    timers = [t_enc, t_est, t_f0, t_src, t_dec]

    import jiwer, whisper
    asr = whisper.load_model("base.en")
    norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    from scipy.io import wavfile
    WAVDIR = os.path.join(os.environ["OUT_DIR"], "multi_wavs"); os.makedirs(WAVDIR, exist_ok=True)

    TEXTS = [
        "Please close the door when you leave.",
        "Can you tell me where the nearest bus stop is?",
        "I usually walk to work in the morning, and then I have a cup of coffee.",
        "We are going to the park this weekend, and the kids want to bring their bikes and a big picnic lunch.",
        "The weather was nice yesterday, so we sat outside for a while and talked about our plans for the summer holidays.",
        "My sister called me last night to tell me about her new job. She likes her team, the office is close to her house, and she can finally take the train instead of driving every day.",
        "After dinner we washed the dishes together, watched a short movie, and went to bed early because we had to get up before sunrise the next morning.",
    ]
    rows = []
    for ui, text in enumerate(TEXTS):
        tgt_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
        ids = torch.cat([prompt_text_ids, tgt_ids], dim=1)
        tl = tgt_ids.shape[1]
        for rep in ("new", "repeat"):
            for t in timers:
                t.reset()
            sync(); torch.manual_seed(0); t0 = time.perf_counter()
            toks = tt_llm.generate(ids, prompt_speech_ids=prompt_tokens, max_tokens=int(tl * 20), min_tokens=int(tl * 2), sampler="ras", seed=0)
            sync(); llm_t = time.perf_counter() - t0
            token = torch.tensor([toks], dtype=torch.long)
            sync(); t0 = time.perf_counter()
            mel = tt_flow.inference(token, prompt_tokens, prompt_feat, ref_embedding)
            sync(); flow_t = time.perf_counter() - t0
            enc_t, est_t = t_enc.total, t_est.total
            m_flow = mem()
            mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            sync(); t0 = time.perf_counter()
            wav = ttnn.to_torch(tt_gen.inference(mel_dev, mel.shape[1], batch_size=1)).reshape(-1).float()
            hift_t = time.perf_counter() - t0
            audio_s = wav.shape[0] / 24000
            m_h = mem()
            total = llm_t + flow_t + hift_t
            hyp = ""
            wer = float("nan")
            if True:
                w16 = torchaudio.functional.resample(wav.unsqueeze(0), 24000, 16000).squeeze(0).numpy()
                hyp = asr.transcribe(w16, language="en", fp16=False)["text"].strip()
                wer = jiwer.wer(text, hyp, truth_transform=norm, hypothesis_transform=norm) * 100
                wavfile.write(f"{WAVDIR}/utt{ui}_{rep}.wav", 24000, (np.clip(wav.numpy(), -1, 1) * 32767).astype(np.int16))
            rows.append((ui, rep, len(toks), audio_s, llm_t, flow_t, enc_t, est_t, hift_t, total, wer, hyp, text))
            print(f"[utt {ui} {rep:6s}] audio {audio_s:5.2f}s  tokens {len(toks):3d}  LLM {llm_t:5.2f}  flow {flow_t:5.2f} (enc {enc_t:.2f} + est {est_t:.2f})  HiFT {hift_t:5.2f}  TOTAL {total:6.2f}s  RTF {total/audio_s:5.2f}" + f"  WER {wer:.1f}%  | L1_SMALL {m_h[0]:6.2f} KB  DRAM {m_h[1]/1024:8.2f} MB/bank")
            sys.stdout.flush()
    print("\n=== SUMMARY ===")
    print(f"{'utt':>3s} {'audio_s':>8s} {'RTF new':>8s} {'RTF repeat':>11s} {'WER new':>8s} {'WER rep':>8s}   text")
    for ui in range(len(TEXTS)):
        n = next(r for r in rows if r[0] == ui and r[1] == "new"); rp = next(r for r in rows if r[0] == ui and r[1] == "repeat")
        print(f"{ui:3d} {rp[3]:8.2f} {n[9]/n[3]:8.2f} {rp[9]/rp[3]:11.2f} {n[10]:8.1f} {rp[10]:8.1f}   {rp[12][:50]}")
        print(f"      ASR(new): {n[11]}")
    print(f"final L1_SMALL {mem()[0]:.2f} KB/bank, DRAM {mem()[1]/1024:.2f} MB/bank")
finally:
    tt_llm.release_decode_trace()
    ttnn.CloseDevice(device)
