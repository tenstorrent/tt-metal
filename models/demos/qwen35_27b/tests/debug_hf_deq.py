"""Test: inject our dequantized weights into HF model and generate."""
import os

import torch

os.environ.setdefault("HF_MODEL", os.path.expanduser("~/models/Qwen3.5-27B-FP8"))

from transformers import AutoConfig, AutoTokenizer

model_path = os.environ["HF_MODEL"]
print(f"Loading HF config from {model_path}...")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

print("Creating empty HF model on CPU...")
# Qwen3.5 is a VL model wrapping a text model. Get the text config.
text_config = getattr(config, "text_config", config)
from transformers import Qwen3NextForCausalLM

# Patch config for Qwen3Next compatibility
for attr, default in [
    ("num_experts", 0),
    ("decoder_sparse_step", 1),
    ("router_aux_loss_coef", 0.0),
    ("num_experts_per_tok", 0),
    ("output_router_logits", False),
]:
    if not hasattr(text_config, attr):
        setattr(text_config, attr, default)
model = Qwen3NextForCausalLM(text_config).to(torch.bfloat16)
model.eval()

print("Loading our dequantized state dict...")
from models.demos.qwen35_27b.tt.model_config import load_qwen35_state_dict

sd = load_qwen35_state_dict(model_path)

# Map our keys back to HF keys
HF_PREFIX = "model."
hf_sd = {}
for key, tensor in sd.items():
    if key == "tok_embeddings.weight":
        hf_sd[HF_PREFIX + "embed_tokens.weight"] = tensor
    elif key == "norm.weight":
        hf_sd[HF_PREFIX + "norm.weight"] = tensor
    elif key == "output.weight":
        hf_sd["lm_head.weight"] = tensor
    elif key.startswith("layers."):
        parts = key.split(".", 2)
        layer_idx = int(parts[1])
        rest = parts[2]

        REV_SHARED = {
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "feed_forward.w1.weight": "mlp.gate_proj.weight",
            "feed_forward.w3.weight": "mlp.up_proj.weight",
            "feed_forward.w2.weight": "mlp.down_proj.weight",
        }
        REV_ATTN = {
            "attention.wqkv.weight": "self_attn.q_proj.weight",
            "attention.wk.weight": "self_attn.k_proj.weight",
            "attention.wv.weight": "self_attn.v_proj.weight",
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention.q_norm.weight": "self_attn.q_norm.weight",
            "attention.k_norm.weight": "self_attn.k_norm.weight",
        }

        if rest in REV_SHARED:
            hf_sd[f"{HF_PREFIX}layers.{layer_idx}.{REV_SHARED[rest]}"] = tensor
        elif rest in REV_ATTN:
            hf_sd[f"{HF_PREFIX}layers.{layer_idx}.{REV_ATTN[rest]}"] = tensor
        elif rest.startswith("linear_attn."):
            hf_sd[f"{HF_PREFIX}layers.{layer_idx}.{rest}"] = tensor
        else:
            hf_sd[f"{HF_PREFIX}layers.{layer_idx}.{rest}"] = tensor

print(f"Mapped {len(hf_sd)} keys for HF model")

# Load weights
missing, unexpected = model.load_state_dict(hf_sd, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if missing:
    print(f"  Missing keys (first 10): {missing[:10]}")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")
print(f"\nPrompt: '{prompt}'")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Output: '{text}'")
if "paris" in text.lower():
    print("PASS")
else:
    print("FAIL")
    # Print top-5 for first generated token
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    topk = torch.topk(probs, k=10)
    print("Top-10 for first token:")
    for j in range(10):
        tid = topk.indices[j].item()
        p = topk.values[j].item()
        print(f"  {j+1}. '{tokenizer.decode([tid])}' ({p:.4f})")
