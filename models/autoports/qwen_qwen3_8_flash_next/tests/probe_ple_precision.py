"""Real-weight PLE compression experiment; not a TTNN decoder validation.

Run with OMP_NUM_THREADS=8 python .../tests/probe_ple_precision.py.
Only referenced 32-row table tiles are read, but quantization includes every
element in each such tile, exactly as a fully resident tiled table would.
HF executes all layer arithmetic. TTNN host tensors supply the actual BFP codec.
"""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from inspect_target import MODEL, REVISION, ROOT
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

import ttnn


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


class Checkpoint:
    def __init__(self):
        self.path = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def tensor(self, key):
        with safe_open(self.path / self.index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def rows(self, key, start, end):
        with safe_open(self.path / self.index[key], framework="pt", device="cpu") as f:
            return f.get_slice(key)[start:end]

    def layer(self, config, idx):
        with torch.device("meta"):
            layer = Qwen4ExpTextDecoderLayer(config, idx)
        prefix = f"model.language_model.layers.{idx}."
        state = {
            k[len(prefix) :]: self.tensor(k)
            for k in self.index
            if k.startswith(prefix) and "ngram_embedding.shard_" not in k
        }
        missing, extra = layer.load_state_dict(state, strict=False, assign=True)
        expected = ["ple.ple_embedding.ngram_embedding.weight"] if layer.ple is not None else []
        assert missing == expected and not extra, (missing, extra)
        return layer.eval()


class SelectedTable(torch.nn.Module):
    """Reference-only exact table lookup and quantized-table counterfactual."""

    def __init__(self, checkpoint):
        super().__init__()
        self.checkpoint = checkpoint
        self.weight = torch.empty(0, dtype=torch.bfloat16)  # HF uses only its device.
        self.codec = None
        self.cache = {}
        self.exact = None
        self.quantized = None

    def forward(self, ids):
        outputs, originals = [], []
        for row in ids.flatten().tolist():
            shard, offset = divmod(row, 2500012)
            block = offset // 32 * 32
            key = (shard, block)
            if key not in self.cache:
                name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
                self.cache[key] = self.checkpoint.rows(name, block, min(block + 32, 2500012)).clone()
            tile_rows = self.cache[key]
            originals.append(tile_rows[offset - block])
            if self.codec is not None:
                encoded = ttnn.from_torch(tile_rows, dtype=self.codec, layout=ttnn.TILE_LAYOUT)
                decoded = ttnn.to_torch(encoded)
                outputs.append(decoded[offset - block].to(torch.bfloat16))
            else:
                outputs.append(tile_rows[offset - block])
        self.exact = torch.stack(originals).reshape(*ids.shape, 160)
        self.quantized = torch.stack(outputs).reshape(*ids.shape, 160)
        return self.quantized


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    checkpoint = Checkpoint()
    config = AutoConfig.from_pretrained(checkpoint.path).text_config
    tokenizer = AutoTokenizer.from_pretrained(checkpoint.path)
    layer0 = checkpoint.layer(config, 0)
    layer1 = checkpoint.layer(config, 1)
    table = SelectedTable(checkpoint)
    layer1.ple.ple_embedding.ngram_embedding = table
    prompts = [
        "Explain why the sky appears blue and how this changes at sunset.",
        "Write a Python function to sort a list of integers and explain its complexity.",
        "Calculate the sum of the first twenty positive integers, showing each step.",
    ]
    results = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        emb = checkpoint.tensor("model.language_model.embed_tokens.weight")[ids]
        x = layer0(emb.repeat(1, 1, 4), position_embeddings=(None, None))
        table.codec = None
        exact = layer1(x, position_embeddings=(None, None), ple_input_ids=ids)
        exact_ple = layer1.ple(x, ids, None)
        for name in ("bfloat8_b", "bfloat4_b"):
            table.codec = getattr(ttnn, name)
            approximate = layer1(x, position_embeddings=(None, None), ple_input_ids=ids)
            approximate_ple = layer1.ple(x, ids, None)
            row = {
                "prompt": prompt,
                "tokens": ids.tolist(),
                "dtype": name,
                "table_pcc": pcc(table.exact, table.quantized),
                "ple_pcc": pcc(exact_ple, approximate_ple),
                "layer_pcc": pcc(exact, approximate),
                "layer_pass_0_995": pcc(exact, approximate) >= 0.995,
                "reference": "HF 5.19 real layer 0 outputs into real layer 1; only table codec changed",
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            (ROOT / "doc/functional_decoder/ple_precision.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
