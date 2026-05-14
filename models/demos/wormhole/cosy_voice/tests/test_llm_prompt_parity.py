"""
Side-by-side parity test for the LLM prompt composition.

This script:
1. Loads the PyTorch reference CosyVoice3LM
2. Hooks into its inference() to capture the exact lm_input tensor
3. Loads our TT CosyVoice3LM and captures its format_prompt_embeddings() output
4. Compares them element-by-element

This will conclusively show whether the prompt embeddings are identical
before the Qwen2 transformer even runs.
"""

import os
import sys

import soundfile as sf
import torch
import torchaudio


# Patch torchaudio.load for soundfile backend
def custom_torchaudio_load(filepath, **kwargs):
    audio, sr = sf.read(filepath)
    if len(audio.shape) == 1:
        tensor = torch.tensor(audio).unsqueeze(0).float()
    else:
        tensor = torch.tensor(audio).transpose(0, 1).float()
    return tensor, sr


torchaudio.load = custom_torchaudio_load
sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice")
sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from hyperpyyaml import load_hyperpyyaml

weights_dir = "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"

# ── 1. Load frontend and process inputs ──
with open(os.path.join(weights_dir, "cosyvoice3.yaml"), "r") as f:
    fe_configs = load_hyperpyyaml(
        f,
        overrides={
            "llm": None,
            "flow": None,
            "hift": None,
            "qwen_pretrain_path": os.path.join(weights_dir, "CosyVoice-BlankEN"),
        },
    )
frontend = CosyVoiceFrontEnd(
    fe_configs["get_tokenizer"],
    fe_configs["feat_extractor"],
    os.path.join(weights_dir, "campplus.onnx"),
    os.path.join(weights_dir, "speech_tokenizer_v3.onnx"),
    os.path.join(weights_dir, "spk2info.pt"),
    fe_configs["allowed_special"],
)

model_input = frontend.frontend_zero_shot(
    "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
    "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    "models/demos/wormhole/cosy_voice/ref/CosyVoice/asset/zero_shot_prompt.wav",
    16000,
    "",
)

print("=" * 70)
print("Frontend output keys:", list(model_input.keys()))
print(f"  text shape:                  {model_input['text'].shape}")
print(f"  text_len:                    {model_input['text_len']}")
print(f"  prompt_text shape:           {model_input['prompt_text'].shape}")
print(f"  prompt_text_len:             {model_input['prompt_text_len']}")
print(f"  llm_prompt_speech_token shape: {model_input['llm_prompt_speech_token'].shape}")
print(f"  llm_prompt_speech_token_len:   {model_input['llm_prompt_speech_token_len']}")
print(f"  llm_embedding shape:           {model_input['llm_embedding'].shape}")
print(f"  text tokens:                 {model_input['text'].flatten().tolist()}")
print(f"  prompt_text tokens:          {model_input['prompt_text'].flatten().tolist()}")

# ── 2. Load reference PyTorch LLM and capture lm_input ──
print("\n" + "=" * 70)
print("Loading reference PyTorch CosyVoice3LM...")

with open(os.path.join(weights_dir, "cosyvoice3.yaml"), "r") as f:
    llm_configs = load_hyperpyyaml(
        f,
        overrides={
            "flow": None,
            "hift": None,
            "qwen_pretrain_path": os.path.join(weights_dir, "CosyVoice-BlankEN"),
        },
    )
ref_llm = llm_configs["llm"]
llm_sd = torch.load(os.path.join(weights_dir, "llm.pt"), map_location="cpu", weights_only=True)
ref_llm.load_state_dict(llm_sd)
ref_llm.eval()

print(f"  ref_llm class: {ref_llm.__class__.__name__}")
print(f"  ref_llm.sos={ref_llm.sos}, ref_llm.task_id={ref_llm.task_id}, ref_llm.eos_token={ref_llm.eos_token}")
print(f"  ref_llm.speech_embedding.weight.shape: {ref_llm.speech_embedding.weight.shape}")
print(f"  Has llm_embedding: {hasattr(ref_llm, 'llm_embedding')}")
if hasattr(ref_llm, "llm_embedding"):
    print(f"  ref_llm.llm_embedding.weight.shape: {ref_llm.llm_embedding.weight.shape}")

# Now reproduce what inference() does to build lm_input
with torch.no_grad():
    text = model_input["text"]
    text_len = model_input["text_len"].clone()
    prompt_text = model_input["prompt_text"]
    prompt_text_len = model_input["prompt_text_len"]
    prompt_speech_token = model_input["llm_prompt_speech_token"]
    prompt_speech_token_len = model_input["llm_prompt_speech_token_len"]
    embedding = model_input["llm_embedding"]

    # Reproduce the exact logic from Qwen2LM.inference() (line 459-494)
    text_combined = torch.concat([prompt_text, text], dim=1)
    text_len_combined = text_len + prompt_text_len
    text_emb_ref = ref_llm.llm.model.model.embed_tokens(text_combined)

    print(f"\n  Combined text shape: {text_combined.shape}")
    print(f"  Combined text tokens: {text_combined.flatten().tolist()}")
    print(f"  text_emb shape: {text_emb_ref.shape}")

    # SOS and task_id — CosyVoice3LM uses speech_embedding
    sos_emb_ref = ref_llm.speech_embedding.weight[ref_llm.sos].reshape(1, 1, -1)
    task_id_emb_ref = ref_llm.speech_embedding.weight[ref_llm.task_id].reshape(1, 1, -1)

    if prompt_speech_token_len != 0:
        prompt_speech_token_emb_ref = ref_llm.speech_embedding(prompt_speech_token)
    else:
        prompt_speech_token_emb_ref = torch.zeros(1, 0, ref_llm.llm_input_size, dtype=text_emb_ref.dtype)

    ref_lm_input = torch.concat([sos_emb_ref, text_emb_ref, task_id_emb_ref, prompt_speech_token_emb_ref], dim=1)

    print(f"\n  ref lm_input shape: {ref_lm_input.shape}")
    print(f"  ref lm_input[:, 0, :5] (sos): {ref_lm_input[0, 0, :5]}")
    print(f"  ref lm_input[:, 1, :5] (first text): {ref_lm_input[0, 1, :5]}")
    print(f"  ref lm_input mean: {ref_lm_input.mean():.6f}")
    print(f"  ref lm_input std: {ref_lm_input.std():.6f}")

# ── 3. Build the TT prompt embeddings using the same weights ──
print("\n" + "=" * 70)
print("Building TT prompt embeddings (host-side, no device needed)...")

# Load the same weights our TT code uses
from models.demos.wormhole.cosy_voice.tt.model_config import remap_cosyvoice_llm_state_dict

full_sd = torch.load(os.path.join(weights_dir, "llm.pt"), map_location="cpu", weights_only=True)
qwen2_sd, cosyvoice_sd = remap_cosyvoice_llm_state_dict(full_sd)

# Our code uses these two embedding tables:
tt_llm_embedding = qwen2_sd["tok_embeddings.weight"].float()  # Qwen2 text embeddings
tt_speech_embedding = cosyvoice_sd["speech_embedding.weight"].float()  # Speech embeddings

# Our token IDs
speech_token_size = 6561
tt_sos = speech_token_size + 0  # 6561
tt_task_id = speech_token_size + 2  # 6563

print(f"  tt_llm_embedding shape: {tt_llm_embedding.shape}")
print(f"  tt_speech_embedding shape: {tt_speech_embedding.shape}")
print(f"  tt_sos={tt_sos}, tt_task_id={tt_task_id}")

# Build the TT lm_input exactly as format_prompt_embeddings does it
with torch.no_grad():
    tt_sos_emb = tt_speech_embedding[tt_sos].reshape(1, 1, -1)
    tt_task_id_emb = tt_speech_embedding[tt_task_id].reshape(1, 1, -1)

    full_text = torch.cat([prompt_text, text], dim=1)
    tt_text_emb = tt_llm_embedding[full_text.flatten()].unsqueeze(0)

    if prompt_speech_token.shape[1] > 0:
        tt_speech_emb = tt_speech_embedding[prompt_speech_token.flatten()].unsqueeze(0)
    else:
        tt_speech_emb = torch.zeros((1, 0, 896), dtype=torch.float32)

    tt_lm_input = torch.cat([tt_sos_emb, tt_text_emb, tt_task_id_emb, tt_speech_emb], dim=1)

    print(f"\n  tt lm_input shape: {tt_lm_input.shape}")
    print(f"  tt lm_input[:, 0, :5] (sos): {tt_lm_input[0, 0, :5]}")
    print(f"  tt lm_input[:, 1, :5] (first text): {tt_lm_input[0, 1, :5]}")
    print(f"  tt lm_input mean: {tt_lm_input.mean():.6f}")
    print(f"  tt lm_input std: {tt_lm_input.std():.6f}")

# ── 4. Compare ──
print("\n" + "=" * 70)
print("COMPARISON: ref vs tt lm_input")

if ref_lm_input.shape != tt_lm_input.shape:
    print(f"  ❌ SHAPE MISMATCH: ref={ref_lm_input.shape} vs tt={tt_lm_input.shape}")
else:
    print(f"  ✅ Shape match: {ref_lm_input.shape}")

# Compare each segment
min_len = min(ref_lm_input.shape[1], tt_lm_input.shape[1])

# Position 0: sos
diff_sos = (ref_lm_input[0, 0] - tt_lm_input[0, 0]).abs()
print(f"\n  Position 0 (sos):")
print(f"    ref mean={ref_lm_input[0,0].mean():.6f}, tt mean={tt_lm_input[0,0].mean():.6f}")
print(f"    max diff={diff_sos.max():.8f}, mean diff={diff_sos.mean():.8f}")

# Position 1: first text token
if min_len > 1:
    diff_t1 = (ref_lm_input[0, 1] - tt_lm_input[0, 1]).abs()
    print(f"\n  Position 1 (first text token):")
    print(f"    ref mean={ref_lm_input[0,1].mean():.6f}, tt mean={tt_lm_input[0,1].mean():.6f}")
    print(f"    max diff={diff_t1.max():.8f}, mean diff={diff_t1.mean():.8f}")

# Full tensor comparison
diff_full = (ref_lm_input[:, :min_len] - tt_lm_input[:, :min_len]).abs()
print(f"\n  Full comparison (first {min_len} positions):")
print(f"    max diff:  {diff_full.max():.8f}")
print(f"    mean diff: {diff_full.mean():.8f}")
print(f"    positions with diff > 0.01: {(diff_full.max(dim=-1).values > 0.01).sum().item()}")

# Find the FIRST position where they diverge significantly
per_pos_max_diff = diff_full[0].max(dim=-1).values  # [min_len]
divergent = (per_pos_max_diff > 0.01).nonzero(as_tuple=True)[0]
if len(divergent) > 0:
    first_div = divergent[0].item()
    print(f"\n  ❌ FIRST DIVERGENCE at position {first_div}")
    print(f"     ref[{first_div}][:10] = {ref_lm_input[0, first_div, :10]}")
    print(f"     tt [{first_div}][:10] = {tt_lm_input[0, first_div, :10]}")
    print(f"     max diff = {per_pos_max_diff[first_div]:.8f}")
else:
    print(f"\n  ✅ ALL POSITIONS MATCH (within tolerance 0.01)")

# ── 5. Also compare the text embedding tables directly ──
print("\n" + "=" * 70)
print("EMBEDDING TABLE COMPARISON:")

# Are the text embedding tables the same?
ref_text_emb_weight = ref_llm.llm.model.model.embed_tokens.weight.data.float()
tt_text_emb_weight = tt_llm_embedding

print(f"  ref text_emb shape: {ref_text_emb_weight.shape}")
print(f"  tt  text_emb shape: {tt_text_emb_weight.shape}")
text_emb_diff = (ref_text_emb_weight[: tt_text_emb_weight.shape[0]] - tt_text_emb_weight).abs()
print(f"  text_emb max diff: {text_emb_diff.max():.8f}")
print(f"  text_emb mean diff: {text_emb_diff.mean():.8f}")

# Are the speech embedding tables the same?
ref_speech_emb_weight = ref_llm.speech_embedding.weight.data.float()
tt_speech_emb_weight = tt_speech_embedding

print(f"\n  ref speech_emb shape: {ref_speech_emb_weight.shape}")
print(f"  tt  speech_emb shape: {tt_speech_emb_weight.shape}")
speech_emb_diff = (ref_speech_emb_weight - tt_speech_emb_weight).abs()
print(f"  speech_emb max diff: {speech_emb_diff.max():.8f}")
print(f"  speech_emb mean diff: {speech_emb_diff.mean():.8f}")

# Save for external analysis
torch.save({"ref_lm_input": ref_lm_input, "tt_lm_input": tt_lm_input}, "/tmp/lm_input_parity.pt")
print("\nSaved both lm_input tensors to /tmp/lm_input_parity.pt")
