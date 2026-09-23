"""DRAM-leak fix validation (this round's item 2): `GeometryWeightCache` (threshold-based,
LRU eviction on real free-DRAM pressure) replaces the unbounded per-geometry dicts that
`TtConv1d`/`TtConvTranspose1d` used to cache prepared conv weights in. The old, unbounded
behaviour measured 90.9 MB/bank -> 134.1 MB/bank of DRAM growth across four distinct
utterance lengths in the 2026-09-21 regression run (see BRINGUP_STATUS.md /
geometry_cache.py's module docstring) -- nothing ever evicted an old geometry's entries.

Two passes over the SAME real four-sentence regression set (LLM -> flow -> HiFT, real
checkpoints, real ASR scoring):

  PASS 1 -- production default threshold (COSYVOICE2_DRAM_FREE_THRESHOLD_MB=150, i.e. evict
  only when free DRAM/bank drops under 150 MB). Reports real DRAM growth across the four
  sentences. Expectation, stated up front: this specific four-sentence, single-session
  workload almost certainly never gets DRAM-tight enough to trip a 150 MB/bank floor on a
  ~1 GB/bank device -- so PASS 1 is expected to show growth similar in *shape* to the old
  unbounded numbers (nothing to evict from *yet*), which is the mechanism working as
  designed ("a session that never gets DRAM-tight never evicts anything"), not a failure of
  it. This is why PASS 2 exists.

  PASS 2 -- a deliberately aggressive threshold (set from the actual free-DRAM measured
  right after model construction, minus a 50 MB budget) on a fresh device, so eviction
  genuinely fires during this real four-sentence run -- not a synthetic single-tensor probe
  like the unit tests, but the real model, real weights, real conv/conv_transpose calls.
  Reports whether DRAM growth stayed bounded under this pressure AND whether WER against all
  four sentences still matches this round's confirmed baseline (0% / 4.17%-class results),
  i.e. that eviction + re-prepare + re-verify never corrupts real synthesis.

Run: PYTHONPATH=/home/user/tt-metal/ttnn:/home/user/tt-metal/tools:/home/user/tt-metal
     timeout -s KILL 1200 /opt/venv/bin/python dram_growth_check.py
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
os.environ["COSYVOICE2_FLOW_ENCODER_TRACE"] = "0"

from cv2_frontend import extract_prompt_feat, extract_speech_tokens, extract_spk_embedding

from models.demos.audio.cosyvoice2.tt.checkpoint import (
    build_local_qwen2_checkpoint_dir,
    load_checkpoint_file,
    sub_state_dict,
)

SCRATCH = os.environ.get("COSYVOICE2_SCRATCH", "/tmp/cosyvoice2_stage1_eval")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/cosyvoice2_perf_2026_09_23_dram")
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
REF_IDX = 0
ref_ex = ds[REF_IDX]
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

import jiwer
import whisper
from scipy.io import wavfile

print("=== loading whisper (base.en) for WER scoring ===")
asr = whisper.load_model("base.en")
CONTRACTION_RULES = [
    (r"\b(can)'t\b", r"\1 not"),
    (r"\b(won)'t\b", r"will not"),
    (r"\b(let)'s\b", r"\1 us"),
    (r"\bi'm\b", r"i am"),
    (r"n't\b", r" not"),
    (r"'re\b", r" are"),
    (r"'ve\b", r" have"),
    (r"'ll\b", r" will"),
    (r"'d\b", r" would"),
]  # scoped contraction expansion -- NOT jiwer.ExpandCommonEnglishContractions, whose 's rule
# false-positives on possessives ("Layton's" -> "Layton is"); see this round's earlier fix.
norm = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes(dict(CONTRACTION_RULES)),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def build_pipeline(device):
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(device, max_batch_size=1, max_seq_len=512, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()

    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    tt_llm = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd, use_decode_trace=True)

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
    return tt_llm, tt_flow, tt_gen


def dram_mb(device):
    mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return mv.total_bytes_allocated_per_bank / 1024 / 1024, mv.total_bytes_free_per_bank / 1024 / 1024


def run_pass(label, threshold_mb, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'=' * 70}\nPASS: {label}  (COSYVOICE2_DRAM_FREE_THRESHOLD_MB={threshold_mb})\n{'=' * 70}")
    os.environ["COSYVOICE2_DRAM_FREE_THRESHOLD_MB"] = str(threshold_mb)
    device = ttnn.CreateDevice(0, l1_small_size=65536, trace_region_size=100_000_000)
    try:
        print("--- building real pipeline (LLM decode-traced, flow CFM-traced, HiFT fp32) ---")
        tt_llm, tt_flow, tt_gen = build_pipeline(device)
        ttnn.synchronize_device(device)
        alloc0, free0 = dram_mb(device)
        print(f"after model construction: DRAM {alloc0:.1f} MB/bank allocated, {free0:.1f} MB/bank free")

        werrs = []
        for ui, text in enumerate(TEXTS):
            tgt_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long)
            ids = torch.cat([prompt_text_ids, tgt_ids], dim=1)
            tl = tgt_ids.shape[1]
            torch.manual_seed(0)
            toks = tt_llm.generate(
                ids, prompt_speech_ids=prompt_tokens, max_tokens=int(tl * 20), min_tokens=int(tl * 2), sampler="ras", seed=0
            )
            token = torch.tensor([toks], dtype=torch.long)
            mel = tt_flow.inference(token, prompt_tokens, prompt_feat, ref_embedding)
            tt_flow.release_traces()
            mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            wav = ttnn.to_torch(tt_gen.inference(mel_dev, mel.shape[1], batch_size=1)).reshape(-1).float()
            ttnn.deallocate(mel_dev)
            ttnn.synchronize_device(device)

            w16 = torchaudio.functional.resample(wav.unsqueeze(0), 24000, 16000).squeeze(0).numpy()
            hyp = asr.transcribe(w16, language="en", fp16=False)["text"].strip()
            wer = jiwer.wer(text, hyp, truth_transform=norm, hypothesis_transform=norm) * 100
            werrs.append(wer)
            wavfile.write(f"{out_dir}/utt{ui}.wav", 24000, (np.clip(wav.numpy(), -1, 1) * 32767).astype(np.int16))

            alloc, free = dram_mb(device)
            print(
                f"[utt {ui}] audio {wav.shape[0]/24000:5.2f}s  WER {wer:5.2f}%  "
                f"DRAM {alloc:7.1f} MB/bank allocated ({alloc - alloc0:+6.1f} vs post-construction), "
                f"{free:7.1f} MB/bank free  hyp: {hyp!r}"
            )
            sys.stdout.flush()

        alloc_final, free_final = dram_mb(device)
        print(
            f"\n{label} SUMMARY: DRAM allocated/bank {alloc0:.1f} -> {alloc_final:.1f} MB "
            f"({alloc_final - alloc0:+.1f} MB growth over 4 utterances), free {free0:.1f} -> {free_final:.1f} MB/bank"
        )
        print(f"{label} WER across all 4 sentences: {[f'{w:.2f}%' for w in werrs]}")
        return alloc0, alloc_final, werrs
    finally:
        tt_llm.release_decode_trace()
        tt_flow.release_traces()
        ttnn.CloseDevice(device)


if __name__ == "__main__":
    a0_default, a1_default, wer_default = run_pass("PASS 1 (production default, 150 MB/bank floor)", 150, f"{OUT_DIR}/pass1")

    # Pick an aggressive threshold from a quick probe: open a device, build the pipeline,
    # measure real free DRAM right after construction, then use (that minus a 50 MB budget)
    # as PASS 2's threshold -- small enough that any real per-geometry growth trips it during
    # the run, without starving the permanent model weights that are never evicted.
    device_probe = ttnn.CreateDevice(0, l1_small_size=65536, trace_region_size=100_000_000)
    try:
        os.environ["COSYVOICE2_DRAM_FREE_THRESHOLD_MB"] = "150"
        tt_llm_p, tt_flow_p, tt_gen_p = build_pipeline(device_probe)
        ttnn.synchronize_device(device_probe)
        _, free_after_construction = dram_mb(device_probe)
    finally:
        tt_llm_p.release_decode_trace()
        tt_flow_p.release_traces()
        ttnn.CloseDevice(device_probe)
    aggressive_threshold_mb = max(1, int(free_after_construction) - 50)
    print(f"\nprobed free DRAM right after construction: {free_after_construction:.1f} MB/bank -> PASS 2 threshold {aggressive_threshold_mb} MB/bank")

    a0_agg, a1_agg, wer_agg = run_pass(
        f"PASS 2 (aggressive, {aggressive_threshold_mb} MB/bank floor -- forces real eviction)",
        aggressive_threshold_mb,
        f"{OUT_DIR}/pass2",
    )

    print(f"\n{'=' * 70}\nFINAL COMPARISON\n{'=' * 70}")
    print(f"PASS 1 (default threshold):    growth {a1_default - a0_default:+.1f} MB/bank over 4 utterances, WER {[f'{w:.2f}%' for w in wer_default]}")
    print(f"PASS 2 (aggressive threshold): growth {a1_agg - a0_agg:+.1f} MB/bank over 4 utterances, WER {[f'{w:.2f}%' for w in wer_agg]}")
    print(
        "WER must match between the two passes (same seeds, same texts) for the eviction mechanism to be "
        "considered safe under real pressure -- any WER divergence here would mean eviction is corrupting synthesis."
    )
