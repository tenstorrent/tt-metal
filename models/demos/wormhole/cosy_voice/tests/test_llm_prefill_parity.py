"""
Side-by-side parity test: PyTorch reference vs TT for the PREFILL logits.

The prompt embeddings are already proven identical.
This test compares what the transformer produces AFTER processing those embeddings.

It runs:
1. PyTorch ref: forward_one_step(lm_input) -> llm_decoder(hidden[-1]) -> logits
2. Prints the top-k token predictions from both
"""

import os
import sys

import soundfile as sf
import torch
import torchaudio


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

# ── 1. Load frontend ──
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

# ── 2. Load reference PyTorch LLM ──
print("=" * 70)
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

# ── 3. Build lm_input (same as both models use) ──
with torch.no_grad():
    text = model_input["text"]
    text_len = model_input["text_len"].clone()
    prompt_text = model_input["prompt_text"]
    prompt_text_len = model_input["prompt_text_len"]
    prompt_speech_token = model_input["llm_prompt_speech_token"]
    prompt_speech_token_len = model_input["llm_prompt_speech_token_len"]

    text_combined = torch.concat([prompt_text, text], dim=1)
    text_emb_ref = ref_llm.llm.model.model.embed_tokens(text_combined)

    sos_emb = ref_llm.speech_embedding.weight[ref_llm.sos].reshape(1, 1, -1)
    task_id_emb = ref_llm.speech_embedding.weight[ref_llm.task_id].reshape(1, 1, -1)
    prompt_speech_token_emb = ref_llm.speech_embedding(prompt_speech_token)

    lm_input = torch.concat([sos_emb, text_emb_ref, task_id_emb, prompt_speech_token_emb], dim=1)
    print(f"lm_input shape: {lm_input.shape}")

# ── 4. Run PyTorch reference prefill ──
print("\n" + "=" * 70)
print("Running PyTorch reference prefill...")

with torch.no_grad():
    masks = torch.tril(torch.ones((1, lm_input.shape[1], lm_input.shape[1]))).to(torch.bool)
    y_pred_ref, cache_ref = ref_llm.llm.forward_one_step(lm_input, masks=masks, cache=None)

    # y_pred_ref is the hidden states from the last layer
    print(f"  y_pred_ref shape: {y_pred_ref.shape}")
    print(f"  y_pred_ref[:, -1, :5]: {y_pred_ref[0, -1, :5]}")

    # Apply llm_decoder to get logits
    ref_logits = ref_llm.llm_decoder(y_pred_ref[:, -1])
    ref_logp = ref_logits.log_softmax(dim=-1)

    print(f"  ref_logits shape: {ref_logits.shape}")
    print(f"  ref_logits[:5]: {ref_logits[0, :5]}")

    # Top-K predictions
    ref_topk_vals, ref_topk_ids = ref_logp.topk(10, dim=-1)
    print(f"\n  PyTorch reference top-10 tokens:")
    for i in range(10):
        print(f"    {i}: token={ref_topk_ids[0, i].item()}, logp={ref_topk_vals[0, i].item():.4f}")

# ── 5. Run first decode step with PyTorch reference ──
print("\n" + "=" * 70)
print("Running PyTorch reference first decode step...")

with torch.no_grad():
    # The reference takes the top token (greedy for comparison)
    ref_first_token = ref_topk_ids[0, 0].item()
    print(f"  First token (greedy): {ref_first_token}")

    # Feed it back as next input
    next_input = ref_llm.speech_embedding.weight[ref_first_token].reshape(1, 1, -1)
    masks_decode = torch.tril(torch.ones((1, lm_input.shape[1] + 1, lm_input.shape[1] + 1))).to(torch.bool)
    y_pred_2, cache_2 = ref_llm.llm.forward_one_step(next_input, masks=masks_decode, cache=cache_ref)

    ref_logits_2 = ref_llm.llm_decoder(y_pred_2[:, -1])
    ref_logp_2 = ref_logits_2.log_softmax(dim=-1)

    ref_topk_vals_2, ref_topk_ids_2 = ref_logp_2.topk(10, dim=-1)
    print(f"\n  PyTorch reference second step top-10 tokens:")
    for i in range(10):
        print(f"    {i}: token={ref_topk_ids_2[0, i].item()}, logp={ref_topk_vals_2[0, i].item():.4f}")

# ── 6. Save for external comparison ──
torch.save(
    {
        "lm_input": lm_input,
        "ref_hidden_last": y_pred_ref[:, -1],
        "ref_logits": ref_logits,
        "ref_first_token": ref_first_token,
        "ref_logits_step2": ref_logits_2,
    },
    "/tmp/ref_llm_prefill.pt",
)
print(f"\nSaved reference prefill data to /tmp/ref_llm_prefill.pt")
print("Now run the same prompt through the TT LLM and compare logits.")
