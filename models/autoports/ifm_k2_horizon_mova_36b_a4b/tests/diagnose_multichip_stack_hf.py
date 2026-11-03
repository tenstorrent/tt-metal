# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU HF control: propagate the two observed dense outputs through identical MoE."""

import importlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.dynamic_module_utils import get_class_from_dynamic_module

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"
MODEL = "IFM/K2-Horizon-MoVA-36B-A4B"
REV = "e9ce717642b2074082875be233c02c729ba47960"
SNAP = Path.home() / ".cache/huggingface/hub/models--IFM--K2-Horizon-MoVA-36B-A4B/snapshots" / REV


def main():
    torch.set_num_threads(16)
    config = AutoConfig.from_pretrained(MODEL, revision=REV, trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "sdpa"
    cls = get_class_from_dynamic_module(
        "modeling_k2_horizon.K2HorizonDecoderLayer", MODEL, revision=REV, local_files_only=True
    )
    mod = importlib.import_module(cls.__module__)
    with torch.device("meta"):
        model = cls(config, 3)
    idx = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.layers.3."
    sd = {}
    for shard in sorted({v for k, v in idx.items() if k.startswith(prefix)}):
        with safe_open(SNAP / shard, framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    sd[k.removeprefix(prefix)] = f.get_tensor(k)
    model.load_state_dict(sd, assign=True)
    model.eval()
    rope = mod.K2HorizonRotaryEmbedding(config)
    outputs = {}
    routes = {}
    current = {}
    tt_scores = {}

    def hook(name):
        def capture(module, inputs, out):
            gate = model.self_attn.v_router if name == "value" else model.mlp.gate
            score = torch.nn.functional.linear(out, gate.weight).float().sigmoid() + gate.bias.float()
            current[name] = torch.topk(score, 4 if name == "value" else 8, dim=-1).indices.tolist()

        return capture

    model.input_layernorm.register_forward_hook(hook("value"))
    model.post_attention_layernorm.register_forward_hook(hook("mlp"))
    with torch.no_grad():
        for label in ("True", "False"):
            data = torch.load(DOC / f"local/debug_stack0_{label}_snapshots.pt", weights_only=True)
            pre = data["False_0"].reshape(2, 33, 2560)
            dec = data["True_0"].reshape(2, 1, 2560)
            out = []
            r = []
            for user, position in enumerate((33, 29)):
                cache = DynamicCache(config=config)
                x = pre[user : user + 1, :position]
                model(
                    x,
                    position_embeddings=rope(x, torch.arange(position)[None]),
                    attention_mask=torch.full((position, position), -float("inf")).triu(1)[None, None],
                    past_key_values=cache,
                )
                x = dec[user : user + 1]
                y = model(
                    x,
                    position_embeddings=rope(x, torch.tensor([[position]])),
                    attention_mask=None,
                    past_key_values=cache,
                )
                out.append(y)
                r.append(dict(current))
            outputs[label] = torch.cat(out)
            routes[label] = r
            a, b = outputs[label].float().flatten(), data["True_1"].float().flatten()
            tt_scores[label] = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    a, b = [outputs[k].float().flatten() for k in ("True", "False")]
    result = {
        "hf_same_layer_decode_pcc": torch.corrcoef(torch.stack([a, b]))[0, 1].item(),
        "routes": routes,
        "tt_stack_vs_hf_on_own_dense_inputs": tt_scores,
        "control": "Identical original HF layer3 and exact HF cache construction, changing only measured TT dense0 outputs; skipped-layer fixture0->3",
    }
    (DOC / "stack_hf_sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
