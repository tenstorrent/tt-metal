"""Answers two more follow-up questions:

3. Re-run utt1 (the 9.5% WER sentence from the 2026-09-22 items 2-4 run) to confirm the
   WER comes from HiFT's unseeded excitation-noise draw (see TtHiFTGenerator.inference,
   which draws real torch.randn noise per call by default -- a real, intentional part of
   the NSF excitation model, not a bug) and not a regression anywhere in this round's
   plumbing changes. Method: run LLM generate() + flow.inference() TWICE, independently,
   to confirm tokens and mel are bit-reproducible (nothing upstream of HiFT has any
   randomness); then call HiFT.inference() TWICE on the SAME mel (two independent noise
   draws) and compare both the raw waveforms and their WER.

4. Fresh per-stage time breakdown (LLM / encoder / CFM / HiFT) now that the CFM solver is
   traced (encoder stays eager -- see BRINGUP_STATUS_22_sept.md's item 2 finding: it
   cannot be traced as currently built).

Run: PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal
     timeout -s KILL 1200 /opt/venv/bin/python wer_repro_and_stage_breakdown.py
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

os.environ["COSYVOICE2_FLOW_CFM_TRACE"] = "1"
os.environ["COSYVOICE2_FLOW_ENCODER_TRACE"] = "0"  # known not to help -- see item 2 finding

from cv2_frontend import extract_prompt_feat, extract_speech_tokens, extract_spk_embedding

from models.demos.audio.cosyvoice2.tt.checkpoint import (
    build_local_qwen2_checkpoint_dir,
    load_checkpoint_file,
    sub_state_dict,
)

SCRATCH = os.environ.get("COSYVOICE2_SCRATCH", "/tmp/cosyvoice2_stage1_eval")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/cosyvoice2_perf_2026_09_22")
os.makedirs(OUT_DIR, exist_ok=True)

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

TEXTS = [
    "Please close the door when you leave.",
    "We are going to the park this weekend, and the kids want to bring their bikes and a big picnic lunch.",
    "The weather was nice yesterday, so we sat outside for a while and talked about our plans for the summer holidays.",
    "My sister called me last night to tell me about her new job. She likes her team, the office is close to her house, and she can finally take the train instead of driving every day.",
]
UTT1_TEXT = TEXTS[1]

print("=== opening device ===")
device = ttnn.CreateDevice(0, l1_small_size=65536, trace_region_size=100_000_000)


def sync():
    ttnn.synchronize_device(device)


class Timed:
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

    def __getattr__(self, name):
        return getattr(self.fn, name)

    def reset(self):
        self.total, self.calls = 0.0, 0


try:
    print("=== building real LLM (TtQwen2LM, use_decode_trace=True) ===")
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(device, max_batch_size=1, max_seq_len=512, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()

    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    tt_llm = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd, use_decode_trace=True)

    print("=== building real flow decoder (bf16, CFM traced, encoder eager) ===")
    import models.demos.audio.cosyvoice2.tt.flow.flow as flow_module
    from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef, TtCausalMaskedDiffWithXvec
    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtSmallEmbedding as _TtSmallEmbedding

    class _Bf16ForcedEmbedding(_TtSmallEmbedding):
        def __init__(self, device, weight, dtype=None):
            super().__init__(device, weight, dtype=ttnn.bfloat16)

    flow_module.TtSmallEmbedding = _Bf16ForcedEmbedding

    flow_ref = CausalMaskedDiffWithXvecRef.from_checkpoint(flow_sd)
    flow_ref.eval()
    tt_flow = TtCausalMaskedDiffWithXvec(device, flow_ref, dtype=ttnn.bfloat16)
    print(f"    encoder.use_trace={tt_flow.encoder.use_trace}  decoder.use_trace={tt_flow.decoder.use_trace}")

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

    import jiwer
    import whisper
    from scipy.io import wavfile

    asr = whisper.load_model("base.en")
    norm = jiwer.Compose(
        [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()]
    )
    from models.common.utility_functions import comp_pcc

    # =====================================================================
    # ITEM 3 (Q3): WER reproducibility for utt1
    # =====================================================================
    print("\n=== Q3: utt1 WER reproducibility -- is it HiFT's unseeded noise, or a regression? ===")
    tgt_ids = torch.tensor([tokenizer.encode(UTT1_TEXT)], dtype=torch.long)
    ids = torch.cat([prompt_text_ids, tgt_ids], dim=1)
    tl = tgt_ids.shape[1]

    def run_llm_and_flow():
        sync()
        torch.manual_seed(0)
        toks = tt_llm.generate(
            ids, prompt_speech_ids=prompt_tokens, max_tokens=int(tl * 20), min_tokens=int(tl * 2), sampler="ras", seed=0
        )
        token = torch.tensor([toks], dtype=torch.long)
        mel = tt_flow.inference(token, prompt_tokens, prompt_feat, ref_embedding)
        tt_flow.release_traces()
        return toks, mel

    toks_1, mel_1 = run_llm_and_flow()
    toks_2, mel_2 = run_llm_and_flow()
    print(f"tokens identical across two independent LLM+flow runs: {toks_1 == toks_2}  (len {len(toks_1)} vs {len(toks_2)})")
    p_mel, r_mel = comp_pcc(mel_1, mel_2, 0.999999)
    print(f"mel PCC between the two independent runs: {r_mel}  (expect ~1.0 -- nothing upstream of HiFT has randomness)")

    def synth_and_transcribe(mel, label):
        mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        wav = ttnn.to_torch(tt_gen.inference(mel_dev, mel.shape[1], batch_size=1)).reshape(-1).float()
        ttnn.deallocate(mel_dev)
        w16 = torchaudio.functional.resample(wav.unsqueeze(0), 24000, 16000).squeeze(0).numpy()
        hyp = asr.transcribe(w16, language="en", fp16=False)["text"].strip()
        wer = jiwer.wer(UTT1_TEXT, hyp, truth_transform=norm, hypothesis_transform=norm) * 100
        wavfile.write(f"{OUT_DIR}/utt1_{label}.wav", 24000, (np.clip(wav.numpy(), -1, 1) * 32767).astype(np.int16))
        return wav, wer, hyp

    wav_a, wer_a, hyp_a = synth_and_transcribe(mel_1, "hift_draw_a")
    wav_b, wer_b, hyp_b = synth_and_transcribe(mel_1, "hift_draw_b")  # SAME mel, independent HiFT noise draw
    print(f"\nHiFT draw A: WER {wer_a:.2f}%  hyp: {hyp_a!r}")
    print(f"HiFT draw B: WER {wer_b:.2f}%  hyp: {hyp_b!r}")
    wav_diff = float((wav_a - wav_b[: wav_a.shape[0]]).abs().max()) if wav_a.shape == wav_b.shape else None
    print(f"waveform identical between the two HiFT draws: {torch.equal(wav_a, wav_b) if wav_a.shape == wav_b.shape else 'shape mismatch'}"
          + (f"  max abs diff: {wav_diff:.4f}" if wav_diff is not None else ""))
    print(f"reference target text: {UTT1_TEXT!r}")

    # =====================================================================
    # ITEM 4: fresh per-stage breakdown now that CFM is traced
    # =====================================================================
    print("\n=== Q4: fresh per-stage breakdown (LLM / encoder / CFM / HiFT), CFM traced, encoder eager ===")
    t_enc = Timed(tt_flow.encoder)
    tt_flow.encoder = t_enc
    t_cfm = Timed(tt_flow.decoder.forward)
    tt_flow.decoder.forward = t_cfm
    t_f0 = Timed(tt_gen.f0_predictor)
    tt_gen.f0_predictor = t_f0
    t_src = Timed(tt_gen.source)
    tt_gen.source = t_src
    t_dec = Timed(tt_gen.decoder.decode)
    tt_gen.decoder.decode = t_dec
    timers = [t_enc, t_cfm, t_f0, t_src, t_dec]

    rows = []
    for ui, text in enumerate(TEXTS):
        tgt_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
        ids = torch.cat([prompt_text_ids, tgt_ids], dim=1)
        tl = tgt_ids.shape[1]
        for rep in ("new", "r1", "r2", "r3"):
            for t in timers:
                t.reset()
            sync()
            torch.manual_seed(0)
            t0 = time.perf_counter()
            toks = tt_llm.generate(
                ids, prompt_speech_ids=prompt_tokens, max_tokens=int(tl * 20), min_tokens=int(tl * 2), sampler="ras", seed=0
            )
            sync()
            llm_t = time.perf_counter() - t0
            token = torch.tensor([toks], dtype=torch.long)
            sync()
            t0 = time.perf_counter()
            mel = tt_flow.inference(token, prompt_tokens, prompt_feat, ref_embedding)
            sync()
            flow_t = time.perf_counter() - t0
            enc_t, cfm_t = t_enc.total, t_cfm.total
            mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            sync()
            t0 = time.perf_counter()
            wav = ttnn.to_torch(tt_gen.inference(mel_dev, mel.shape[1], batch_size=1)).reshape(-1).float()
            sync()
            hift_t = time.perf_counter() - t0
            audio_s = wav.shape[0] / 24000
            total = llm_t + flow_t + hift_t
            rows.append((ui, rep, audio_s, llm_t, flow_t, enc_t, cfm_t, hift_t, total))
            print(
                f"[utt {ui} {rep:6s}] audio {audio_s:5.2f}s  LLM {llm_t:6.3f}  flow {flow_t:6.3f} "
                f"(enc {enc_t:.3f} + cfm {cfm_t:.3f})  HiFT {hift_t:6.3f}  TOTAL {total:6.3f}s  RTF {total/audio_s:.3f}"
            )
            sys.stdout.flush()
            tt_flow.release_traces()

    print("\n=== SUMMARY: warm (r1-r3 mean) per-stage share of TOTAL ===")
    for ui in range(len(TEXTS)):
        w = [r for r in rows if r[0] == ui and r[1] != "new"]
        n = len(w)
        audio_s = w[0][2]
        llm_m = sum(r[3] for r in w) / n
        flow_m = sum(r[4] for r in w) / n
        enc_m = sum(r[5] for r in w) / n
        cfm_m = sum(r[6] for r in w) / n
        hift_m = sum(r[7] for r in w) / n
        total_m = sum(r[8] for r in w) / n
        print(
            f"utt {ui} audio {audio_s:.2f}s  TOTAL {total_m:.3f}s (RTF {total_m/audio_s:.3f})  "
            f"LLM {llm_m:.3f}s ({llm_m/total_m*100:4.1f}%)  "
            f"encoder {enc_m:.3f}s ({enc_m/total_m*100:4.1f}%)  "
            f"CFM {cfm_m:.3f}s ({cfm_m/total_m*100:4.1f}%)  "
            f"HiFT {hift_m:.3f}s ({hift_m/total_m*100:4.1f}%)"
        )
finally:
    tt_llm.release_decode_trace()
    tt_flow.release_traces()
    ttnn.CloseDevice(device)
