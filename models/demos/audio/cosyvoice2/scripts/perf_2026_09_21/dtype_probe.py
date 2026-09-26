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


import json, sys
from collections import defaultdict
device = ttnn.CreateDevice(0, l1_small_size=65536)
REC = defaultdict(lambda: dict(n=0, inp=set(), wt=set(), out=set(), cc=set()))
def dt(x):
    try:
        return str(x.dtype).replace("DataType.", "")
    except Exception:
        return "-"
def classify():
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "cosyvoice2/tt/" in fn or "tt_transformers/tt/" in fn:
            for key, name in (("flow/encoder", "flow encoder"), ("flow/decoder", "flow estimator"), ("flow/flow", "flow glue"),
                              ("hifigan/f0_predictor", "F0 predictor"), ("hifigan/source", "source / SineGen2"), ("hifigan/stft", "STFT"),
                              ("hifigan/istft", "iSTFT"), ("hifigan/upsample", "HiFT transposed convs"), ("hifigan/conv", "HiFT convs"),
                              ("hifigan/resblock", "HiFT resblocks"), ("hifigan/snake", "HiFT snake"), ("hifigan/generator", "HiFT generator"),
                              ("llm/qwen2lm", "LLM head/embed"), ("tt_transformers", "LLM (tt_transformers)")):
                if key in fn:
                    return name
            return fn.split("/")[-1]
        f = f.f_back
    return "other"
def first(x):
    return x[0] if isinstance(x, (list, tuple)) else x
def patch(ns, attr, label, w_index=1, w_kw=("weight_tensor", "input_tensor_b", "weight"), in_kw=("input_tensor", "input_tensor_a")):
    orig = getattr(ns, attr)
    def f(*a, **k):
        out = orig(*a, **k)
        comp = classify()
        r = REC[(comp, label)]
        r["n"] += 1
        x = a[0] if a else next((k[n] for n in in_kw if n in k), None)
        w = a[w_index] if len(a) > w_index else next((k[n] for n in w_kw if n in k), None)
        r["inp"].add(dt(x)); r["wt"].add(dt(w) if w is not None else "-"); r["out"].add(dt(first(out)))
        cc = k.get("compute_kernel_config", k.get("compute_config"))
        r["cc"].add("explicit" if cc is not None else "default")
        return out
    setattr(ns, attr, f)
for ns, attr, label in ((ttnn, "linear", "linear"), (ttnn, "matmul", "matmul"), (ttnn, "conv1d", "conv1d"), (ttnn, "conv_transpose2d", "conv_transpose2d"),
                        (ttnn, "layer_norm", "layer_norm"), (ttnn, "rms_norm", "rms_norm"), (ttnn, "embedding", "embedding")):
    patch(ns, attr, label)
patch(ttnn.transformer, "scaled_dot_product_attention", "sdpa", w_index=1)
patch(ttnn.transformer, "scaled_dot_product_attention_decode", "sdpa_decode", w_index=1) if hasattr(ttnn.transformer, "scaled_dot_product_attention_decode") else None
import json
toks = json.load(open(os.environ["OUT_DIR"] + "/est_tokens.json"))["weather"]
try:
    print("=== building real LLM (TtQwen2LM) ===")
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(device, max_batch_size=1, max_seq_len=512, dummy_weights=False, use_hf_rope=True)
    state_dict = args.load_state_dict()

    from models.demos.audio.cosyvoice2.tt.llm.qwen2lm import TtQwen2LM

    tt_llm = TtQwen2LM(args, device, state_dict, cosyvoice_state_dict=llm_sd)
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

    tgt = torch.tensor([tokenizer.encode(toks["text"])], dtype=torch.long)
    ids = torch.cat([prompt_text_ids, tgt], dim=1)
    print("--- LLM: prefill + 3 decode steps (untraced)", flush=True)
    tt_llm.generate(ids, prompt_speech_ids=prompt_tokens, max_tokens=3, min_tokens=0, sampler="greedy", use_trace=False)
    print("--- flow", flush=True)
    token = torch.tensor([toks["tokens"]], dtype=torch.long)
    mel = tt_flow.inference(token, prompt_tokens, prompt_feat, ref_embedding)
    print("--- HiFT", flush=True)
    md = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gen.inference(md, mel.shape[1], batch_size=1)
    print("\n=== OBSERVED DTYPES PER COMPONENT AND OP (input / weight / output dtype; compute config passed?) ===")
    print(f"{'component':26s} {'op':17s} {'calls':>6s}  {'input dtype':18s} {'weight dtype':22s} {'output dtype':18s} compute_cfg")
    for (comp, label), r in sorted(REC.items()):
        print(f"{comp:26s} {label:17s} {r['n']:6d}  {','.join(sorted(r['inp'])):18s} {','.join(sorted(r['wt'])):22s} {','.join(sorted(r['out'])):18s} {','.join(sorted(r['cc']))}")
finally:
    ttnn.CloseDevice(device)
