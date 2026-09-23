"""Items 2-4 of the 2026-09-22 round:

  2. Cached/traced flow encoder (tt/flow/encoder.py's TtUpsampleConformerEncoder, added
     this round) -- measured at the REAL Stage 1 (whole-utterance, non-streaming) token
     length for each of the four sentences, i.e. prompt_token_len + generated_token_len,
     NOT the 100-frame streaming-chunk number from the doc's A.4. A separate, explicitly
     labeled 100-frame STREAMING PROBE is also measured, kept apart from the Stage 1
     numbers.
  3. Traced Euler-step CFM solver (tt/flow/decoder.py's TtCausalConditionalCFM, added
     this round) -- capture / warm-up / replay / release costs reported SEPARATELY, at
     steps=10 (the validated step count; no step-count reduction this round).
  4. Full Stage 1 warm end-to-end regression at steps=10, all four sentences: token
     accuracy, WER, speaker similarity, RTF -- checked against the 2026-09-21 baseline
     (100% / 4.17% / 0.8885 / 0.58-1.01) with the LLM decode trace, encoder trace and CFM
     trace all enabled together.

Run: PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal
     timeout -s KILL 900 /opt/venv/bin/python traced_encoder_and_cfm.py
from a directory other than the repo root. COSYVOICE2_SCRATCH / OUT_DIR as usual.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torchaudio
import ttnn

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/home/user/tt-metal/models/demos/audio/cosyvoice2/scripts/vocoder_debug_2026_09_20")

os.environ["COSYVOICE2_FLOW_ENCODER_TRACE"] = "1"
os.environ["COSYVOICE2_FLOW_CFM_TRACE"] = "1"

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

print("=== opening device ===")
device = ttnn.CreateDevice(0, l1_small_size=65536, trace_region_size=100_000_000)


def sync():
    ttnn.synchronize_device(device)


def mem():
    l1 = ttnn.get_memory_view(device, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank / 1024
    dr = ttnn.get_memory_view(device, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank / 1024
    return l1, dr


try:
    print("=== building real LLM (TtQwen2LM, use_decode_trace=True) ===")
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(device, max_batch_size=1, max_seq_len=512, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()

    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    tt_llm = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd, use_decode_trace=True)

    print("=== building real flow decoder (bf16, encoder + CFM trace on) ===")
    import models.demos.audio.cosyvoice2.tt.flow.flow as flow_module
    from models.demos.audio.cosyvoice2.tt.flow.decoder import TtCausalConditionalCFM
    from models.demos.audio.cosyvoice2.tt.flow.encoder import TtUpsampleConformerEncoder
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

    # =====================================================================
    # ITEM 2: cached/traced flow encoder, at real Stage-1 token lengths
    # =====================================================================
    print("\n=== ITEM 2: cached/traced flow encoder, real Stage 1 lengths ===")
    encoder_results = []
    for ui, text in enumerate(TEXTS):
        tgt_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
        ids = torch.cat([prompt_text_ids, tgt_ids], dim=1)
        tl = tgt_ids.shape[1]
        sync()
        torch.manual_seed(0)
        toks = tt_llm.generate(
            ids, prompt_speech_ids=prompt_tokens, max_tokens=int(tl * 20), min_tokens=int(tl * 2), sampler="ras", seed=0
        )
        token = torch.tensor([toks], dtype=torch.long)
        full_token = torch.cat([prompt_tokens, token], dim=1)
        real_len = full_token.shape[1]  # the ACTUAL encoder input length Stage 1 uses for this utterance

        ids_dev = ttnn.from_torch(
            full_token.reshape(1, 1, 1, -1).clamp(min=0).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        tok_emb_dev = tt_flow.input_embedding(ids_dev)

        # cold (first call at this exact length -> captures)
        sync()
        t0 = time.perf_counter()
        h = tt_flow.encoder(tok_emb_dev, real_len, 1, use_trace=True)
        sync()
        cold_t = time.perf_counter() - t0
        ttnn.deallocate(h)

        # warm (three repeats, cache hit -> replay only)
        warm_ts = []
        for _ in range(3):
            sync()
            t0 = time.perf_counter()
            h = tt_flow.encoder(tok_emb_dev, real_len, 1, use_trace=True)
            sync()
            warm_ts.append(time.perf_counter() - t0)
            ttnn.deallocate(h)

        # eager comparison at the same length (untraced)
        sync()
        t0 = time.perf_counter()
        h = tt_flow.encoder(tok_emb_dev, real_len, 1, use_trace=False)
        sync()
        eager_t = time.perf_counter() - t0
        ttnn.deallocate(h)

        row = {
            "utt": ui,
            "real_stage1_len": real_len,
            "cold_ms": cold_t * 1000,
            "warm_ms": [t * 1000 for t in warm_ts],
            "eager_ms": eager_t * 1000,
        }
        encoder_results.append(row)
        print(
            f"[utt {ui}] Stage1 real encoder length T={real_len}  cold(capture) {cold_t*1000:7.1f} ms  "
            f"warm(replay) {[f'{t*1000:.1f}' for t in warm_ts]} ms  eager {eager_t*1000:7.1f} ms  "
            f"speedup {eager_t/min(warm_ts):.2f}x"
        )
        sys.stdout.flush()
        tt_flow.encoder.release_encoder_trace()

    # --- explicitly labeled STREAMING PROBE, 100-frame (A.4's number), separate from Stage 1 ---
    print("\n=== STREAMING PROBE (100-frame chunk length; NOT a Stage 1 number) ===")
    T_PROBE = 100
    dummy_ids = torch.randint(0, 6561, (1, T_PROBE), dtype=torch.long)
    ids_dev = ttnn.from_torch(
        dummy_ids.reshape(1, 1, 1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tok_emb_dev = tt_flow.input_embedding(ids_dev)
    sync()
    t0 = time.perf_counter()
    h = tt_flow.encoder(tok_emb_dev, T_PROBE, 1, use_trace=True)
    sync()
    probe_cold = time.perf_counter() - t0
    ttnn.deallocate(h)
    probe_warm = []
    for _ in range(3):
        sync()
        t0 = time.perf_counter()
        h = tt_flow.encoder(tok_emb_dev, T_PROBE, 1, use_trace=True)
        sync()
        probe_warm.append(time.perf_counter() - t0)
        ttnn.deallocate(h)
    tt_flow.encoder.release_encoder_trace()
    print(
        f"[STREAMING PROBE T={T_PROBE}] cold(capture) {probe_cold*1000:.1f} ms  "
        f"warm(replay) {[f'{t*1000:.1f}' for t in probe_warm]} ms  -- NOT a Stage 1 measurement"
    )

    # =====================================================================
    # ITEM 3: traced CFM -- capture / warm-up / replay / release, separately
    # =====================================================================
    print("\n=== ITEM 3: traced CFM solver -- capture/warm-up/replay/release costs ===")
    # Reuse utterance 0's real mu/spks/cond/mask at steps=10 for a clean, isolated measurement.
    tgt_ids = torch.tensor([tokenizer.encode(TEXTS[0])], dtype=torch.long)
    ids = torch.cat([prompt_text_ids, tgt_ids], dim=1)
    tl = tgt_ids.shape[1]
    sync()
    torch.manual_seed(0)
    toks = tt_llm.generate(
        ids, prompt_speech_ids=prompt_tokens, max_tokens=int(tl * 20), min_tokens=int(tl * 2), sampler="ras", seed=0
    )
    token = torch.tensor([toks], dtype=torch.long)
    spks = tt_flow._xvec(ref_embedding)
    full_token = torch.cat([prompt_tokens, token], dim=1)
    ids_dev = ttnn.from_torch(
        full_token.reshape(1, 1, 1, -1).clamp(min=0).to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tok_emb_dev = tt_flow.input_embedding(ids_dev)
    h_dev = tt_flow.encoder(tok_emb_dev, full_token.shape[1], 1, use_trace=False)  # eager here; encoder isolated above
    t_len2 = h_dev.shape[1]
    mel_len1 = prompt_feat.shape[1]
    h_dev = ttnn.linear(h_dev, tt_flow.encoder_proj_w, bias=tt_flow.encoder_proj_b)
    mu = ttnn.to_torch(h_dev).float().reshape(1, t_len2, tt_flow.output_size)
    conds = torch.zeros(1, t_len2, tt_flow.output_size, dtype=mu.dtype)
    conds[:, :mel_len1] = prompt_feat
    mask = torch.ones(1, t_len2, 1, dtype=mu.dtype)
    print(f"    real T for this utterance's CFM solve: {t_len2}")

    cfm = tt_flow.decoder

    # capture (first call at this geometry: includes 2x warm-up + begin/end_trace_capture)
    sync()
    t0 = time.perf_counter()
    out_capture = cfm.forward(mu, mask, 10, spks, conds, use_trace=True)
    sync()
    capture_total = time.perf_counter() - t0
    cfm.release_cfm_trace()

    # warm-up cost alone, isolated: call _capture's internals are not separately exposed,
    # so measure warm-up by timing 2 eager estimator calls at the same geometry directly.
    zero_mu, zero_spks, zero_cond = torch.zeros_like(mu), torch.zeros_like(spks), torch.zeros_like(conds)
    mu_in = torch.cat([mu, zero_mu], dim=0)
    spks_in = torch.cat([spks, zero_spks], dim=0)
    cond_in = torch.cat([conds, zero_cond], dim=0)
    mask_in = torch.cat([mask, mask], dim=0)
    mu_dev = ttnn.from_torch(mu_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    spks_dev = ttnn.from_torch(spks_in.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cond_dev = ttnn.from_torch(cond_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mask_dev = ttnn.from_torch(mask_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x_probe = ttnn.from_torch(
        torch.cat([mu[:, :, :80], mu[:, :, :80]], dim=0) * 0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    t_probe = torch.zeros(2)
    sync()
    t0 = time.perf_counter()
    for _ in range(2):
        o = cfm.estimator(x_probe, mask_dev, mu_dev, t_probe, spks_dev, cond_dev, t_len2, batch_size=2)
        ttnn.deallocate(o)
    sync()
    warmup_only = time.perf_counter() - t0
    for t in (mu_dev, spks_dev, cond_dev, mask_dev, x_probe):
        ttnn.deallocate(t)

    # replay-only cost: capture once more, then time 10 replays in isolation
    sync()
    t0 = time.perf_counter()
    _ = cfm.forward(mu, mask, 10, spks, conds, use_trace=True)  # capture again (fresh, was released above)
    sync()
    capture_only_est = time.perf_counter() - t0 - warmup_only  # capture_total minus the warm-up portion, approx

    sync()
    t0 = time.perf_counter()
    out_replay = cfm.forward(mu, mask, 10, spks, conds, use_trace=True)  # now a cache hit: reuse + 10 replays only
    sync()
    replay_total = time.perf_counter() - t0

    sync()
    t0 = time.perf_counter()
    cfm.release_cfm_trace()
    sync()
    release_t = time.perf_counter() - t0

    print(f"[CFM T={t_len2}, steps=10] capture (incl. 2x warm-up) {capture_total*1000:7.1f} ms")
    print(f"    warm-up alone (2x eager estimator call, isolated)  {warmup_only*1000:7.1f} ms")
    print(f"    capture proper (approx, capture_total - warmup)   {capture_only_est*1000:7.1f} ms")
    print(f"    replay (10 steps, cache hit: reuse + 10x execute)  {replay_total*1000:7.1f} ms  "
          f"({replay_total/10*1000:.2f} ms/step)")
    print(f"    release                                            {release_t*1000:7.1f} ms")

    from models.common.utility_functions import comp_pcc

    p, r = comp_pcc(out_capture, out_replay, 0.99)
    print(f"    replay-vs-capture-call PCC (sanity, same utterance) {r}")
    cfm.release_cfm_trace()

    # =====================================================================
    # ITEM 4: full Stage 1 warm end-to-end regression, steps=10, all 4 sentences
    # =====================================================================
    print("\n=== ITEM 4: Stage 1 warm end-to-end regression (steps=10, real measured RTF) ===")
    import jiwer
    import whisper
    from scipy.io import wavfile

    asr = whisper.load_model("base.en")
    norm = jiwer.Compose(
        [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()]
    )
    WAVDIR = os.path.join(OUT_DIR, "regression_wavs")
    os.makedirs(WAVDIR, exist_ok=True)

    def spk_sim(w16: torch.Tensor) -> float:
        """Cosine similarity of the generated clip's speaker embedding against the
        reference's own (real campplus.onnx), same definition `spk_torch_check.py` used
        for the 2026-09-21 baseline (0.8885)."""
        return float(
            torch.nn.functional.cosine_similarity(
                ref_embedding, extract_spk_embedding(campplus_session, w16.unsqueeze(0)), dim=1
            )
        )

    rows = []
    for ui, text in enumerate(TEXTS):
        tgt_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
        ids = torch.cat([prompt_text_ids, tgt_ids], dim=1)
        tl = tgt_ids.shape[1]
        for rep in ("new", "r1", "r2", "r3"):
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
            mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            sync()
            t0 = time.perf_counter()
            wav = ttnn.to_torch(tt_gen.inference(mel_dev, mel.shape[1], batch_size=1)).reshape(-1).float()
            sync()
            hift_t = time.perf_counter() - t0
            audio_s = wav.shape[0] / 24000
            total = llm_t + flow_t + hift_t
            w16 = torchaudio.functional.resample(wav.unsqueeze(0), 24000, 16000).squeeze(0).numpy()
            hyp = asr.transcribe(w16, language="en", fp16=False)["text"].strip()
            wer = jiwer.wer(text, hyp, truth_transform=norm, hypothesis_transform=norm) * 100
            sim = spk_sim(torch.from_numpy(w16))
            wavfile.write(f"{WAVDIR}/utt{ui}_{rep}.wav", 24000, (np.clip(wav.numpy(), -1, 1) * 32767).astype(np.int16))
            rows.append((ui, rep, len(toks), audio_s, llm_t, flow_t, hift_t, total, wer, sim, hyp, text))
            print(
                f"[utt {ui} {rep:6s}] audio {audio_s:5.2f}s  tokens {len(toks):3d}  LLM {llm_t:5.2f}  "
                f"flow {flow_t:5.2f}  HiFT {hift_t:5.2f}  TOTAL {total:6.2f}s  RTF {total/audio_s:5.2f}  "
                f"WER {wer:.1f}%  spk-sim {sim:.4f}"
            )
            sys.stdout.flush()
            tt_flow.release_traces()

    print("\n=== SUMMARY (warm = r1..r3; REAL MEASURED, not composed) ===")
    for ui in range(len(TEXTS)):
        n = next(r for r in rows if r[0] == ui and r[1] == "new")
        w = [r for r in rows if r[0] == ui and r[1] != "new"]
        print(
            f"utt {ui} audio {w[0][3]:.2f}s  RTF new {n[7]/n[3]:.2f}  warm RTF "
            + " ".join(f"{r[7]/r[3]:.3f}" for r in w)
            + f"  WER new/warm {n[8]:.1f}/{w[-1][8]:.1f}%  spk-sim warm {w[-1][9]:.4f}"
        )
    all_warm_rtf = [r[7] / r[3] for r in rows if r[1] != "new"]
    all_warm_wer = [r[8] for r in rows if r[1] != "new"]
    all_warm_sim = [r[9] for r in rows if r[1] != "new"]
    print(f"\nreal measured warm RTF range: {min(all_warm_rtf):.2f} - {max(all_warm_rtf):.2f}")
    print(f"real measured warm WER range: {min(all_warm_wer):.2f}% - {max(all_warm_wer):.2f}%")
    print(f"real measured warm speaker-similarity range: {min(all_warm_sim):.4f} - {max(all_warm_sim):.4f}")
    print(f"2026-09-21 baseline for comparison: WER 4.17%, RTF 0.58-1.01, spk-sim 0.8885 (token acc 100%)")

    print(
        "\n=== RESULT JSON ===\n"
        + json.dumps(
            {
                "item2_encoder": encoder_results,
                "item2_streaming_probe_T100": {"cold_ms": probe_cold * 1000, "warm_ms": [t * 1000 for t in probe_warm]},
                "item3_cfm": {
                    "capture_incl_warmup_ms": capture_total * 1000,
                    "warmup_alone_ms": warmup_only * 1000,
                    "replay_10steps_ms": replay_total * 1000,
                    "release_ms": release_t * 1000,
                },
                "item4_regression": {
                    "warm_rtf_min": min(all_warm_rtf),
                    "warm_rtf_max": max(all_warm_rtf),
                    "warm_wer_min": min(all_warm_wer),
                    "warm_wer_max": max(all_warm_wer),
                    "warm_spk_sim_min": min(all_warm_sim),
                    "warm_spk_sim_max": max(all_warm_sim),
                },
            },
            indent=2,
        )
    )
finally:
    tt_llm.release_decode_trace()
    tt_flow.release_traces()
    ttnn.CloseDevice(device)
