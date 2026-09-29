# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stream independent CPU HF references without importing TTNN or candidate code."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers import __version__ as transformers_version
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer, GptOssRMSNorm, GptOssRotaryEmbedding

CHECKPOINT_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
BOUNDARIES = [63, 64, 65, 511, 512, 513, 767, 768, 769, 8191, 8192, 8193]
BATCHES = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def generate(snapshot, output, kind):
    if transformers_version != "5.12.1":
        raise ValueError(f"Reference requires Transformers 5.12.1, got {transformers_version}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Reference destination must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    assert config.num_hidden_layers == 36 and config.max_position_embeddings == 131072
    config._attn_implementation = "eager"
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]

    def read_weight(name):
        with safe_open(snapshot / weight_map[name], framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    batches = kind == "batches"
    boundaries = [63, 64, 65, 127, 128, 129] if batches else BOUNDARIES
    tokens = torch.randint(
        256, config.vocab_size, (32 if batches else 1, max(boundaries)), generator=torch.Generator().manual_seed(1234)
    )
    positions = torch.arange(tokens.shape[1]).unsqueeze(0).expand(tokens.shape[0], -1)
    hidden = F.embedding(tokens, read_weight("model.embed_tokens.weight").to(torch.bfloat16))
    rope = GptOssRotaryEmbedding(config=config)
    position_embeddings = rope(hidden, positions)
    mask_arguments = {"config": config, "inputs_embeds": hidden, "attention_mask": None, "past_key_values": None}
    masks = {
        "full_attention": create_causal_mask(**mask_arguments),
        "sliding_attention": create_sliding_window_causal_mask(**mask_arguments),
    }
    torch.save({"tokens": tokens, "boundaries": boundaries}, output / "inputs.pt")
    for index in range(2 if batches else 36):
        started = time.monotonic()
        prefix = f"model.layers.{index}."
        state = {name[len(prefix) :]: read_weight(name) for name in weight_map if name.startswith(prefix)}
        for projection in ("gate_up_proj", "down_proj"):
            key = f"mlp.experts.{projection}"
            state[key] = convert_moe_packed_tensors(
                state.pop(key + "_blocks"), state.pop(key + "_scales"), dtype=torch.bfloat16
            )
        state = {
            name: value.to(torch.float32 if name.endswith("layernorm.weight") else torch.bfloat16)
            for name, value in state.items()
        }
        with torch.device("meta"):
            layer = GptOssDecoderLayer(config, index)
        layer.load_state_dict(state, assign=True, strict=True)
        layer.eval()
        del state
        if index in (0, 1):
            torch.save({"input": hidden, "tokens": tokens, "layer_index": index}, output / f"layer{index}-input.pt")
        hidden = layer(
            hidden,
            attention_mask=masks[config.layer_types[index]],
            position_ids=positions,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        if index in (0, 1):
            torch.save({"output": hidden, "layer_index": index}, output / f"layer{index}-output.pt")
        del layer
        gc.collect()
        finite = bool(hidden.isfinite().all())
        print(json.dumps({"layer": index, "seconds": time.monotonic() - started, "finite": finite}), flush=True)
        assert finite
    if not batches:
        selected = hidden[:, torch.tensor(boundaries) - 1]
        norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight = torch.nn.Parameter(read_weight("model.norm.weight").to(torch.float32))
        logits = F.linear(norm(selected), read_weight("lm_head.weight").to(torch.bfloat16))
        torch.save(
            {"boundaries": boundaries, "tokens": tokens, "logits": logits, "top100": logits.topk(100, dim=-1).indices},
            output / "full-model-logits.pt",
        )
    manifest = {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "transformers": transformers_version,
        "torch": torch.__version__,
        "dtype": "bfloat16; RMS norm weights and accumulation float32",
        "attention": "eager",
        "seed": 1234,
        "layers": [0, 1] if batches else 36,
        "batches": BATCHES if batches else [1],
        "context_capacity": 131072,
        "boundaries": boundaries,
        "script_sha256": sha256(Path(__file__)),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(output.glob("*.pt"))
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("REFERENCE_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", required=True, type=Path, help="Verified snapshot of the pinned checkpoint revision"
    )
    parser.add_argument("--output", required=True, type=Path, help="Empty directory for reference tensors and manifest")
    parser.add_argument("--kind", choices=("boundaries", "batches"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)
    generate(args.snapshot, args.output, args.kind)
