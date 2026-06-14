"""BF16 CPU reference: per-layer hidden_states + per-MoE-layer top-1 expert choices.
Used to isolate device-bf16 vs torch-bf16 divergence (routing flips).
Run: HF_HOME=/home/yito/work/hf_cache /home/yito/work/zaya-ref-venv/bin/python dump_bf16_ref.py
"""
import os
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM

MID = "Zyphra/ZAYA1-8B"
OUT = Path(__file__).resolve().parent / "golden"


def main():
    cfg = AutoConfig.from_pretrained(MID, trust_remote_code=True)
    if not isinstance(cfg.zaya_mlp_expansion, (list, tuple)):
        s = int(cfg.zaya_mlp_expansion)
        cfg.zaya_mlp_expansion = [(s if isinstance(l, int) else 0) for l in cfg.zaya_layers]
    if cfg.hidden_size // cfg.num_attention_heads != cfg.head_dim:
        cfg.num_attention_heads = cfg.hidden_size // cfg.head_dim

    ids = torch.load(OUT / "inputs.pt", weights_only=False)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(
        MID, config=cfg, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager", low_cpu_mem_usage=True).eval()

    # hook every router: capture expert_choice (forward output index 1)
    choices = {}

    def mk(layer_n):
        def hook(mod, inp, out):
            choices[layer_n] = out[1].detach().reshape(-1).long().cpu()  # [S]
        return hook

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "zaya_block"):
            layer.zaya_block.router.register_forward_hook(mk(i))

    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True)

    hs = [h.detach().float().cpu() for h in out.hidden_states]
    torch.save(hs, OUT / "bf16_hidden_states.pt")
    torch.save({k: choices[k] for k in sorted(choices)}, OUT / "bf16_router_choices.pt")
    moe_layers = sorted(choices)
    print(f"saved bf16 hidden ({len(hs)}) + router choices for {len(moe_layers)} MoE layers")
    print("last-token logits top1:", out.logits[0, -1].float().argmax().item())
    # print per-MoE-layer choice for the LAST token (pos -1)
    print("last-token expert choices by MoE layer:",
          {k: int(choices[k][-1]) for k in moe_layers[:5]}, "...",
          {k: int(choices[k][-1]) for k in moe_layers[-5:]})


if __name__ == "__main__":
    main()
