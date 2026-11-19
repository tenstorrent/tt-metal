# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU diagnostic: selected real PLE rows, NF4 storage, unchanged HF PLE.

The embedding boundary reads only requested checkpoint rows. It does not model
resident device lookup and is never a complete decoder acceptance test.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


def pcc(a, b):
    return torch.corrcoef(torch.stack((a.float().flatten(), b.float().flatten())))[0, 1].item()


def quantize(rows, method="absmax"):
    codebook = torch.tensor(NF4)
    values = rows.float()
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-20)
    if method == "lsq":
        # Row-local fitting uses no evaluation activation or cross-row calibration.
        for _ in range(5):
            idx = ((values / scale)[..., None] - codebook).abs().argmin(-1)
            codes = codebook[idx]
            scale = ((values * codes).sum(-1, keepdim=True) / codes.square().sum(-1, keepdim=True)).clamp_min(1e-20)
    scale = scale.bfloat16().float()
    idx = ((values / scale)[..., None] - codebook).abs().argmin(-1)
    return (codebook[idx] * scale).bfloat16()


class SelectedRows(torch.nn.Module):
    def __init__(self, snapshot, index, prefix, method):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.snapshot, self.index, self.prefix, self.method = snapshot, index, prefix, method
        self.rows, self.quantized, self.ids = {}, False, set()

    def forward(self, ids):
        flat = ids.flatten().tolist()
        missing = sorted(set(flat) - self.rows.keys())
        by_shard = {}
        for row in missing:
            shard, local = divmod(row, 2500012)
            by_shard.setdefault(shard, []).append((row, local))
        for shard, locations in by_shard.items():
            key = f"{self.prefix}shard_{shard}.weight"
            with safe_open(self.snapshot / self.index[key], framework="pt") as f:
                source = f.get_slice(key)
                for row, local in locations:
                    self.rows[row] = source[local : local + 1, :].clone()[0]
        self.ids.update(flat)
        rows = torch.stack([self.rows[row] for row in flat]).reshape(*ids.shape, 160)
        return quantize(rows, self.method) if self.quantized else rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=257)
    parser.add_argument("--method", choices=["absmax", "lsq"], default="absmax")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(841)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.1.ple."
    with torch.device("meta"):
        ple = Qwen4ExpTextPLELayer(config, layer_idx=1, ple_layer_index=0)
    state = {}
    for name, shard in index.items():
        if name.startswith(prefix) and ".ngram_embedding.shard_" not in name:
            with safe_open(args.snapshot / shard, framework="pt") as f:
                state[name.removeprefix(prefix)] = f.get_tensor(name)
    ple.load_state_dict(state, strict=False, assign=True)
    rows = SelectedRows(args.snapshot, index, prefix + "ple_embedding.ngram_embedding.", args.method)
    ple.ple_embedding.ngram_embedding = rows
    ple.eval()
    tokens = torch.randint(0, config.vocab_size, (2, args.tokens))
    eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
    tokens[0, [0, 30, 31, 64]] = eos
    tokens[1, [1, 32, 63, 128]] = eos
    report = {
        "scope": "CPU real-weight PLE diagnostic; not TT or complete decoder acceptance",
        "snapshot": str(args.snapshot),
        "method": args.method,
        "tokens_shape": list(tokens.shape),
        "rows": [],
    }
    base = torch.randn(2, args.tokens, config.hidden_size * config.hc_count).bfloat16()
    with torch.inference_mode():
        for scale in (0.02, 0.2, 1.0, 4.0):
            hidden = base * scale
            rows.quantized = False
            exact = ple(hidden, tokens, None)
            rows.quantized = True
            approximate = ple(hidden, tokens, None)
            report["rows"].append(
                {
                    "activation_std": scale,
                    "mode": "prefill",
                    "ple_pcc": pcc(exact, approximate),
                    "residual_plus_ple_pcc": pcc(hidden + exact, hidden + approximate),
                    "relative_rmse": (
                        (exact.float() - approximate.float()).square().mean() / exact.float().square().mean()
                    )
                    .sqrt()
                    .item(),
                }
            )
        # Each side receives its own real HF conv/hash cache. Mix arbitrary chunks
        # and one-token decode and check against the same side's full call.
        hidden = base * 0.2
        outputs = []
        for quantized in (False, True):
            rows.quantized = quantized
            cache = DynamicCache(config=config)
            pieces = []
            for start, end in ((0, 31), (31, 32), (32, 65), (65, args.tokens - 1), (args.tokens - 1, args.tokens)):
                pieces.append(ple(hidden[:, start:end], tokens[:, start:end], cache))
            outputs.append(torch.cat(pieces, dim=1))
        report["rows"].append(
            {
                "activation_std": 0.2,
                "mode": "chunked_and_decode",
                "ple_pcc": pcc(*outputs),
                "last_token_ple_pcc": pcc(outputs[0][:, -1], outputs[1][:, -1]),
            }
        )
        rows.quantized = False
        control = ple(hidden, tokens, None)
        report["exact_continuation_pcc"] = pcc(control, outputs[0])
        selected = torch.stack(list(rows.rows.values()))
        report["embedding_pcc"] = pcc(selected, quantize(selected, args.method))
    report["selected_unique_rows"] = len(rows.ids)
    report["shards_touched"] = sorted({row // 2500012 for row in rows.ids})
    report["weight_stats"] = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "mean": value.float().mean().item(),
            "std": value.float().std().item(),
        }
        for name, value in state.items()
        if value.is_floating_point()
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("weight_stats", "shards_touched")}))


if __name__ == "__main__":
    main()
