# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Official HF reduced-stack control, CPU only; matches probe_full_model inputs."""

import argparse
import copy
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextModel, MuseGlimmerTextRotaryEmbedding

SNAPSHOT = Path(
    "/mnt/models/huggingface/hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
)


def main(args):
    torch.set_num_threads(8)
    original = AutoConfig.from_pretrained(SNAPSHOT).text_config
    config = copy.deepcopy(original)
    indices = [0, 3]
    config.num_hidden_layers = len(indices)
    config.layer_types = [original.layer_types[i] for i in indices]
    config.layer_rope_theta = [original.layer_rope_theta[i] for i in indices]
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = MuseGlimmerTextModel(config)
    weight_map = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]

    def read(key):
        with safe_open(SNAPSHOT / weight_map[key], framework="pt") as f:
            return f.get_tensor(key)

    state = {}
    for key in model.state_dict():
        source = key
        if key.startswith("layers."):
            parts = key.split(".")
            parts[1] = str(indices[int(parts[1])])
            source = ".".join(parts)
        state[key] = read("model.language_model." + source)
    model.load_state_dict(state, assign=True)
    model.rotary_emb = MuseGlimmerTextRotaryEmbedding(config)
    model.eval()
    head = read("lm_head.weight")

    def logits(x):
        value = torch.nn.functional.linear(x, head)
        value = value * config.output_multiplier
        return torch.tanh(value / config.final_logit_softcapping) * config.final_logit_softcapping

    with torch.no_grad():
        ids = torch.arange(32, 32 + args.length)[None]
        prefill = model(ids, use_cache=True)
        decode = model(torch.tensor([[42]]), past_key_values=prefill.past_key_values, use_cache=True)
        result = dict(
            prefill=logits(prefill.last_hidden_state[:, -1:]),
            decode=logits(decode.last_hidden_state),
            prefill_hidden=prefill.last_hidden_state,
            decode_hidden=decode.last_hidden_state,
            layer_indices=indices,
            length=args.length,
        )
        torch.save(result, args.output)
        print("HF_PROBE_OK", result["prefill"].argmax(-1).tolist(), result["decode"].argmax(-1).tolist(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
