"""CPU-only control: selected-query reference equals an ordinary full HF layer."""

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    model = "IFM/K2-Horizon-7B"
    revision = "036114ce8d46c32b24c15423211069abb9c5d25e"
    config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    cls = get_class_from_dynamic_module("modeling_k2_horizon.K2HorizonDecoderLayer", model, revision=revision)
    rope_cls = get_class_from_dynamic_module("modeling_k2_horizon.K2HorizonRotaryEmbedding", model, revision=revision)
    with safe_open(
        hf_hub_download(model, "pytorch_model-00001-of-00036.safetensors", revision=revision),
        framework="pt",
        device="cpu",
    ) as f:
        state = {
            k.removeprefix("model.layers.0."): f.get_tensor(k) for k in f.keys() if k.startswith("model.layers.0.")
        }
    with torch.device("meta"):
        hf = cls(config, layer_idx=0)
    hf.load_state_dict(state, assign=True)
    hf.eval()
    torch.manual_seed(9155)
    x = (torch.randn(1, 257, 4096) * 0.03).bfloat16()
    rope = rope_cls(config)(x, torch.arange(257)[None])
    ordinary = hf(x, position_embeddings=rope, past_key_values=DynamicCache(config=config))
    normalized = hf.input_layernorm(x)
    k = hf.self_attn.k_proj(normalized).reshape(1, -1, 8, 128).transpose(1, 2)
    v = hf.self_attn.v_proj(normalized).reshape(1, -1, 8, 128).transpose(1, 2)
    k = k * rope[0].unsqueeze(1) + torch.cat([-k[..., 64:], k[..., :64]], -1) * rope[1].unsqueeze(1)
    cache = DynamicCache(config=config)
    cache.update(k[:, :, :225], v[:, :, :225], 0)
    mask = torch.where(
        torch.arange(257)[None, :] <= torch.arange(225, 257)[:, None], 0.0, torch.finfo(torch.bfloat16).min
    ).bfloat16()[None, None]
    selected = hf(
        x[:, -32:], position_embeddings=tuple(r[:, -32:] for r in rope), attention_mask=mask, past_key_values=cache
    )
    score = torch.corrcoef(torch.stack([ordinary[:, -32:].float().flatten(), selected.float().flatten()]))[0, 1].item()
    assert score >= 0.99999, score
    result = {
        "completed": True,
        "model": model,
        "revision": revision,
        "sequence_length": 257,
        "selected_query_rows": 32,
        "full_vs_selected_HF_pcc": score,
        "max_abs_difference": (ordinary[:, -32:].float() - selected.float()).abs().max().item(),
    }
    path = Path(__file__).resolve().parents[1] / "doc/functional_decoder/reference_control.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
