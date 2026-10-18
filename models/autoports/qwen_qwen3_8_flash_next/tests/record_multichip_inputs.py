"""CPU-only reference fixture: checkpoint embeddings and HF decoder layers0-2.

Records actual layer0/1/3 inputs for precision experiments. No full-model
implementation, logits, generation, TTNN import, or TT hardware operation.
"""

import argparse, gc, hashlib, json
from contextlib import ExitStack
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState

CHECKPOINT = Path(
    "/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
)


class MappedPLE(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.weight = torch.empty(0, dtype=torch.bfloat16)

    def forward(self, ids):
        values = []
        with ExitStack() as stack:
            handles, slices = {}, {}
            for row in ids.flatten().tolist():
                shard, offset = divmod(row, 2500012)
                key = f"ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
                path = self.state.path / self.state.index[key]
                if path not in handles:
                    handles[path] = stack.enter_context(safe_open(path, framework="pt"))
                if key not in slices:
                    slices[key] = handles[path].get_slice(self.state.prefix + key)
                values.append(slices[key][offset : offset + 1].clone())
        return torch.cat(values).reshape(*ids.shape, 160)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=130)
    parser.add_argument("--tag", default="recorded_inputs")
    args = parser.parse_args()
    assert args.length >= 3
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    prompt = (
        "A research team measures how a transformer processes a document. Explain why request positions, cached keys and values, "
        "and selected experts must stay consistent when computation is distributed across four processors. Include a small example. "
    ) * max(8, args.length // 32 + 1)
    ids = tokenizer(prompt, return_tensors="pt").input_ids[:, : args.length]
    assert ids.shape == (1, args.length)
    tokens = [ids[:, :-2], ids[:, -2:-1], ids[:, -1:]]
    weights = CheckpointState(CHECKPOINT, "model.language_model.")
    embedding = weights["embed_tokens.weight"]
    inputs = [torch.nn.functional.embedding(t, embedding).repeat(1, 1, cfg.hc_count) for t in tokens]
    del embedding
    dest = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip") / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    artifacts = []
    with torch.no_grad():
        for index in range(4):
            if index in [0, 1, 3]:
                path = dest / f"layer{index}.pt"
                torch.save(dict(inputs=inputs, tokens=tokens), path)
                artifacts.append(
                    dict(
                        layer=index,
                        path=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        shapes=[list(x.shape) for x in inputs],
                    )
                )
            if index == 3:
                break
            state = CheckpointState(CHECKPOINT, f"model.language_model.layers.{index}.")
            with torch.device("meta"):
                layer = Qwen4ExpTextDecoderLayer(cfg, index)
            tensors = {k: state[k] for k in state if "ngram_embedding.shard_" not in k}
            missing, extra = layer.load_state_dict(tensors, strict=False, assign=True)
            assert (
                missing == (["ple.ple_embedding.ngram_embedding.weight"] if layer.ple is not None else []) and not extra
            )
            del tensors
            if layer.ple is not None:
                layer.ple.ple_embedding.ngram_embedding = MappedPLE(state)
            layer.eval()
            cache = DynamicCache(config=cfg)
            outputs = []
            for x, t in zip(inputs, tokens):
                kw = dict(past_key_values=cache, position_embeddings=(None, None))
                if layer.ple is not None:
                    kw["ple_input_ids"] = t
                outputs.append(layer(x, **kw).detach())
            inputs = outputs
            del layer, cache
            gc.collect()
            print("RECORDED_LAYER_OUTPUT", index, flush=True)
    result = dict(
        checkpoint=str(CHECKPOINT),
        args=vars(args),
        source="real checkpoint embedding -> HF decoders0-2",
        prompt=prompt,
        input_ids=ids.tolist(),
        artifacts=artifacts,
    )
    Path(f"models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder/{args.tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    print("INPUT_FIXTURES_PASS", flush=True)


if __name__ == "__main__":
    main()
