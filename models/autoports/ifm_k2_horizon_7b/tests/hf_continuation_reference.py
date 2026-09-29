"""CPU-only first-three-layer reference for continuation localization."""

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .hf_qualitative import MODEL, REVISION


def main():
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    config.num_hidden_layers = 3
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, config=config).eval()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    base = tok.encode("A careful scientist checks the evidence before drawing a conclusion. ")
    tokens = torch.tensor([(base * 30)[:257]])
    outputs = {}
    for index, layer in enumerate(model.model.layers):

        def capture(module, inputs, output, index=index):
            outputs[index] = (output[0] if isinstance(output, tuple) else output).detach().cpu().clone()

        layer.register_forward_hook(capture)
    with torch.no_grad():
        model.model(tokens, attention_mask=torch.ones_like(tokens), use_cache=False)
    path = Path("bringup/artifacts/ifm_k2_full_model_raw_20260928/hf_continuation_l3.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt_tokens": tokens, "layer_outputs": outputs}, path)
    print(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "layers": 3,
                "artifact": str(path),
                "shapes": {k: list(v.shape) for k, v in outputs.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
