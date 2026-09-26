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
import jiwer, whisper
from scipy.io import wavfile
VARIANT = os.environ["VARIANT"]
OUT = os.environ["OUT_DIR"]
OUTD = f"{OUT}/est/{VARIANT}"; os.makedirs(OUTD, exist_ok=True)
BASED = f"{OUT}/est/baseline"
TRACE_MB = int(os.environ.get("TRACE_MB", "200"))
print(f"=== VARIANT {VARIANT}; trace region {TRACE_MB} MB; tracker env: "
      f"{os.environ.get('TT_METAL_TRACE_ALLOC_TRACKING')}/{os.environ.get('TT_METAL_TRACE_ALLOC_TRACEBACKS')}")
device = ttnn.CreateDevice(0, l1_small_size=65536, trace_region_size=TRACE_MB * 1024 * 1024)
def sync(): ttnn.synchronize_device(device)
def pcc(a, b):
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1]), float((a - b).norm() / b.norm())
toks = json.load(open(f"{OUT}/est_tokens.json"))
try:
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
    tt_flow = TtCausalMaskedDiffWithXvec(device, flow_ref, dtype=(ttnn.bfloat16 if os.environ.get("FLOW_DTYPE") == "bf16" else ttnn.float32))
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

    asr = whisper.load_model("base.en")
    norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    first_key = next(iter(toks))
    for k, d in ([] if os.environ.get('SKIP_QUALITY') else toks.items()):
        token = torch.tensor([d["tokens"]], dtype=torch.long)
        ts = []
        for rep in range(3):
            sync(); t0 = time.perf_counter()
            mel = tt_flow.inference(token, prompt_tokens, prompt_feat, ref_embedding)
            sync(); ts.append(time.perf_counter() - t0)
        np.save(f"{OUTD}/mel_{k}.npy", mel.numpy())
        line = f"[{k}] frames {mel.shape[1]} flow warm {ts[-1]:.2f}s (runs {', '.join(f'{t:.1f}' for t in ts)})"
        if os.path.exists(f"{BASED}/mel_{k}.npy"):
            p, r = pcc(mel, torch.from_numpy(np.load(f"{BASED}/mel_{k}.npy")))
            line += f"   mel vs baseline fp32: PCC {p:.6f} relL2 {r:.4f}"
        md = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        wav = ttnn.to_torch(tt_gen.inference(md, mel.shape[1], batch_size=1)).reshape(-1).float()
        ttnn.deallocate(md)
        wavfile.write(f"{OUTD}/{k}.wav", 24000, (np.clip(wav.numpy(), -1, 1) * 32767).astype(np.int16))
        w16 = torchaudio.functional.resample(wav.unsqueeze(0), 24000, 16000).squeeze(0).numpy()
        hyp = asr.transcribe(w16, language="en", fp16=False)["text"].strip()
        wer = jiwer.wer(d["text"], hyp, truth_transform=norm, hypothesis_transform=norm) * 100
        print(line + f"   WER {wer:.1f}%  | {hyp}"); sys.stdout.flush()
        first_key = first_key or k

    # ------------------------------------------------------------------ traced per-call estimator profile
    print("\n=== TRACED ESTIMATOR PROFILE (one estimator call, batch 2) ===")
    d = toks[first_key]; NTOK = int(os.environ["NTOK"]); token = torch.tensor([(d["tokens"] * 3)[:NTOK]], dtype=torch.long)
    est = tt_flow.decoder.estimator
    rec = {}
    def cp(t):
        return ttnn.from_torch(ttnn.to_torch(t).float().contiguous(), dtype=t.dtype, layout=t.layout, device=device)
    def recorder(x, mask, mu, t, spks, cond, length, batch_size):
        if "args" not in rec:
            rec["args"] = (cp(x), cp(mask), cp(mu), t.clone(), cp(spks), cp(cond), length, batch_size)
        return est_orig(x, mask, mu, t, spks, cond, length, batch_size)
    est_orig = est
    tt_flow.decoder.estimator = recorder
    tt_flow.inference(token, prompt_tokens, prompt_feat, ref_embedding)      # eager solve; records the first call's inputs
    tt_flow.decoder.estimator = est_orig
    args = rec["args"]; T = args[6]
    print(f"recorded estimator inputs: x {tuple(args[0].shape)} mask {tuple(args[1].shape)} T={T}")

    # (1) ALL eager reference work, BEFORE any capture (allocating under a live trace is the #54032 hazard)
    out_ref = ttnn.to_torch(est(*args)).float()
    N = 10
    sync(); t0 = time.perf_counter()
    for _ in range(N):
        o = est(*args); ttnn.deallocate(o)
    sync(); t_eager = (time.perf_counter() - t0) / N
    print(f"eager (untraced) per call: {t_eager*1000:.1f} ms")

    # (2) scratch enablers so the call contains no host upload / raw-weight transfer (NOT repo changes)
    from models.demos.audio.cosyvoice2.tt.flow import decoder as dmod
    temb_buf = est._sinusoidal_pos_emb(args[3])                              # persistent device tensor, made outside capture
    est._sinusoidal_pos_emb = lambda t: temb_buf
    def conv_call(self, x, input_length, batch_size=1):
        key = (input_length, batch_size)
        cfg = self._verified.get(key, self._accurate)
        cache = self.__dict__.setdefault("_tw", {})
        kw = dict(input_tensor=x, device=self.device, in_channels=self.in_channels, out_channels=self.out_channels,
                  batch_size=batch_size, input_length=input_length, kernel_size=self.kernel_size, stride=1,
                  padding=(self.kernel_size - 1, 0), dilation=1, groups=1, conv_config=self.conv_config,
                  compute_config=cfg, dtype=self.dtype, return_output_dim=True)
        if key not in cache:                                                  # warm-up call: op prepares weights; keep them on device
            res = ttnn.conv1d(weight_tensor=self._weight_4d, bias_tensor=self._bias, return_weights_and_bias=True, **kw)
            flat = []
            def walk(r):
                for e in (r if isinstance(r, (list, tuple)) else [r]):
                    flat.append(e) if not isinstance(e, (list, tuple)) else walk(e)
            walk(res)
            tens = [e for e in flat if isinstance(e, ttnn.Tensor)]; ints = [e for e in flat if isinstance(e, int)]
            out, w, b = tens[0], tens[1], tens[2]; out_len = ints[0]
            cache[key] = (w, b)
        else:
            w, b = cache[key]
            out, out_len = ttnn.conv1d(weight_tensor=w, bias_tensor=b, **kw)
        return ttnn.reshape(out, (batch_size, out_len, self.out_channels))
    dmod.TtCausalConv1d.__call__ = conv_call

    warm = est(*args)                                                         # compiles + prepares every conv's weights, OUTSIDE capture
    p, r = pcc(ttnn.to_torch(warm).float(), out_ref)
    print(f"patched (prepared-weight, device temb) vs original eager: PCC {p:.7f} relL2 {r:.5f}")
    ttnn.deallocate(warm)

    # (3) capture; from here on: replay only, no allocation of any kind
    sync()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_tr = est(*args)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    tv = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
    print("TRACE region view:", {a: getattr(tv, a) for a in dir(tv) if a.startswith("total") or a.startswith("largest")})
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False); sync()
    p, r = pcc(ttnn.to_torch(out_tr).float(), out_ref)
    print(f"traced replay vs eager reference: PCC {p:.7f} relL2 {r:.5f}")
    sync(); t0 = time.perf_counter()
    for _ in range(N):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    sync(); t_tr = (time.perf_counter() - t0) / N
    print(f"TRACED per call: {t_tr*1000:.1f} ms   (eager {t_eager*1000:.1f} ms, speedup {t_eager/t_tr:.2f}x)")
    ttnn.release_trace(device, tid)
    print("RESULT", json.dumps({"ntok": NTOK, "variant": VARIANT, "T": T, "eager_ms": t_eager * 1000, "traced_ms": t_tr * 1000}))
finally:
    ttnn.CloseDevice(device)
