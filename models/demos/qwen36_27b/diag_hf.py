"""Compare TT model with HuggingFace reference model on real weights."""
import torch
import os

SNAP = "/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

# Load HF reference model (text only, 1 layer for speed)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

print("Loading HF config...")
config = AutoConfig.from_pretrained(SNAP, trust_remote_code=True)
text_config = config.text_config if hasattr(config, 'text_config') else config
print(f"Model type: {config.model_type}")
print(f"Text layers: {text_config.num_hidden_layers}")

# Use only 4 layers for comparison
text_config.num_hidden_layers = 4
if hasattr(config, 'text_config'):
    config.text_config.num_hidden_layers = 4

print("Loading HF model (4 layers)...")
try:
    hf_model = AutoModelForCausalLM.from_pretrained(
        SNAP, config=config, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cpu"
    )
    hf_model.eval()
    print(f"HF model loaded: {type(hf_model).__name__}")
    
    # Check model structure
    if hasattr(hf_model, 'language_model'):
        lm = hf_model.language_model
    elif hasattr(hf_model, 'model'):
        lm = hf_model.model
    else:
        lm = hf_model
    print(f"Language model: {type(lm).__name__}")
    
    # Check layer types
    if hasattr(lm, 'layers'):
        for i, layer in enumerate(lm.layers):
            print(f"  Layer {i}: {type(layer).__name__} -> {[type(c).__name__ for n, c in layer.named_children()]}")
    
except Exception as e:
    print(f"Failed to load HF model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

# Run HF forward pass
prompt = "Hello, who are you?"
messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer.encode(formatted, return_tensors="pt")
print(f"\nPrompt tokens: {input_ids.shape[1]}")

with torch.no_grad():
    hf_out = hf_model(input_ids)
    hf_logits = hf_out.logits  # [B, S, V]
    print(f"HF logits shape: {hf_logits.shape}")
    last_logits = hf_logits[0, -1, :]  # Last position
    top5 = last_logits.float().topk(5)
    print(f"HF top-5 tokens: {[(tokenizer.decode([t]), t.item(), v.item()) for t, v in zip(top5.indices, top5.values)]}")

# Now run TT model with same 4 layers
import ttnn
device = ttnn.open_device(device_id=0)
try:
    from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig
    from models.demos.qwen36_27b.tt.load_weights import load_state_dict
    from models.demos.qwen36_27b.tt.model import TtQwen36Model
    from models.demos.qwen36_27b.tt.generator import Qwen36Generator
    
    tt_config = Qwen36ModelConfig()
    tt_config.num_hidden_layers = 4
    
    state_dict = load_state_dict(tt_config, max_layers=4, model_path=SNAP)
    model = TtQwen36Model(device, state_dict, tt_config)
    generator = Qwen36Generator(model, tt_config, tokenizer=tokenizer)
    del state_dict
    
    last_logits_tt = generator.prefill(input_ids)
    tt_logits = ttnn.to_torch(last_logits_tt).float().reshape(-1)[:tt_config.vocab_size]
    
    top5_tt = tt_logits.topk(5)
    print(f"\nTT top-5 tokens: {[(tokenizer.decode([t]), t.item(), v.item()) for t, v in zip(top5_tt.indices, top5_tt.values)]}")
    
    # Compare logits distributions
    print(f"\nHF logits stats: mean={last_logits.float().mean():.4f}, std={last_logits.float().std():.4f}")
    print(f"TT logits stats: mean={tt_logits.mean():.4f}, std={tt_logits.std():.4f}")
    
    # Compare full logit vectors
    hf_v = last_logits.float()[:tt_config.vocab_size]
    corr = torch.nn.functional.cosine_similarity(hf_v.unsqueeze(0), tt_logits.unsqueeze(0))
    print(f"Logit cosine similarity: {corr.item():.6f}")
    
finally:
    ttnn.close_device(device)
