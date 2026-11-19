# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only real layer1 PLE+GDN long-context reference preparation.

This reduced decoder state path includes literal HF token hashing, exact BF16
checkpoint lookup, PLE convolution, PLE residual addition, attention hyper-
connection read, GDN convolution/recurrent state, and attention residual write.
It excludes the token-local MoE and its hyper-connection. It is neither a full
decoder acceptance test nor evidence of device context capability. No TTNN is
imported. The reference is independent of the eventual device lookup format.

All tokenwise module-boundary outputs are persisted, together with the raw
input/token IDs and cache snapshots immediately before and after the final
single-token HF decode. Internal recurrent states are saved at those boundaries
only; saving one 48x128x128 state per token would exceed the fixture's scope.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextGatedDeltaNet,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextPLELayer,
)

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_quantization import SelectedRows


class ObservedRows(SelectedRows):
    """Exact selected-row lookup; inherited loading clones each checkpoint row."""

    def forward(self, ids):
        self.last_ids = ids
        result = super().forward(ids)
        self.last_embeddings = result.flatten(-2)
        return result


class ReducedStatePath(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ple = Qwen4ExpTextPLELayer(config, 1, 0)
        self.attn_hyper_connection = Qwen4ExpTextGatedResidual(config)
        self.linear_attn = Qwen4ExpTextGatedDeltaNet(config, 1)

    def forward(self, hidden, ids, cache):
        ple = self.ple(hidden, ids, cache)
        after_ple = hidden + ple
        mixed, residual, injection = self.attn_hyper_connection(after_ple)
        attention = self.linear_attn(mixed, cache_params=cache)
        after_attention = residual + (attention.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)
        rows = self.ple.ple_embedding.ngram_embedding
        return {
            "ngram_ids": rows.last_ids,
            "embeddings": rows.last_embeddings,
            "ple_output": ple,
            "after_ple": after_ple,
            "attn_mixed": mixed,
            "attn_injection": injection,
            "attention": attention,
            "after_attention": after_attention,
        }


def load_reference(snapshot, config):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.1."
    keys = [
        key
        for key in index
        if key.startswith(prefix)
        and key.removeprefix(prefix).startswith(("ple.", "attn_hyper_connection.", "linear_attn."))
        and ".ngram_embedding.shard_" not in key
    ]
    weights = {}
    for filename in sorted({index[key] for key in keys}):
        with safe_open(snapshot / filename, framework="pt") as source:
            for key in keys:
                if index[key] == filename:
                    weights[key.removeprefix(prefix)] = source.get_tensor(key)
    with torch.device("meta"):
        model = ReducedStatePath(config)
    loaded = model.load_state_dict(weights, strict=False, assign=True)
    assert loaded.missing_keys == ["ple.ple_embedding.ngram_embedding.weight"] and not loaded.unexpected_keys, loaded
    rows = ObservedRows(snapshot, index, prefix + "ple.ple_embedding.ngram_embedding.", "absmax")
    rows.quantized = False
    model.ple.ple_embedding.ngram_embedding = rows
    model.eval()
    return model, keys


def tokens(config, length, chunk, seed):
    result = torch.randint(config.vocab_size, (1, length), generator=torch.Generator().manual_seed(seed + 1))
    eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
    positions = sorted({i for i in (0, 30, 31, 32, 63, 64, chunk - 1, chunk, chunk + 1, length - 4) if 0 <= i < length})
    result[0, positions] = eos
    return result, positions


def state_snapshot(cache):
    return {
        "gdn_convolution": cache.get_conv_state(1, state_idx=0).clone(),
        "ple_convolution": cache.get_conv_state(1, state_idx=1).clone(),
        "token_history": cache.get_conv_state(1, state_idx=2).clone(),
        "gdn_recurrent": cache.get_recurrent_state(1, state_idx=0).clone(),
    }


def row_metric(expected, actual):
    if expected.dtype == torch.int64:
        return {"exact": torch.equal(expected, actual)}
    a, b = expected.float().reshape(-1, expected.shape[-1]), actual.float().reshape(-1, actual.shape[-1])
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    denominator = a.norm(dim=-1) * b.norm(dim=-1)
    correlation = (a * b).sum(-1) / denominator.clamp_min(1e-30)
    exact = (expected.reshape_as(a) == actual.reshape_as(b)).all(-1)
    correlation = torch.where(exact, 1.0, correlation)
    return {
        "minimum_row_pcc": correlation.min().item(),
        "max_abs_error": (expected.float() - actual.float()).abs().max().item(),
        "finite": bool(torch.isfinite(actual).all()),
    }


@torch.inference_mode()
def control(model, config, seed):
    hidden = (torch.randn(1, 65, 10240, generator=torch.Generator().manual_seed(seed + 2)) * 0.02).bfloat16()
    ids, _ = tokens(config, 65, 32, seed + 2)
    whole = model(hidden, ids, DynamicCache(config=config))
    cache = DynamicCache(config=config)
    parts = [
        model(hidden[:, begin:end], ids[:, begin:end], cache) for begin, end in ((0, 31), (31, 32), (32, 64), (64, 65))
    ]
    result = {
        name: row_metric(value, torch.cat([part[name] for part in parts], dim=1)) for name, value in whole.items()
    }
    assert result["ngram_ids"]["exact"]
    assert all(item.get("finite", True) and item.get("minimum_row_pcc", 1) >= 0.995 for item in result.values()), result
    return {"rows": 65, "chunk_ends": [31, 32, 64, 65], "outputs": result}


@torch.inference_mode()
def prepare(args):
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    assert 2 <= args.length <= config.max_position_embeddings and args.chunk >= 2
    assert config.hidden_size == 2560 and config.hc_count == 4 and config.layer_types[1] == "linear_attention"
    identity = {
        "format": "ple_gdn_reduced_context_v1",
        "snapshot": str(args.snapshot.resolve()),
        "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
        "checkpoint_index_sha256": hashlib.sha256(
            (args.snapshot / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "hf_source_sha256": hashlib.sha256(Path(inspect.getfile(Qwen4ExpTextPLELayer)).read_bytes()).hexdigest(),
        "fixture_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selected_rows_source_sha256": hashlib.sha256(inspect.getsource(SelectedRows).encode()).hexdigest(),
        "torch_version": str(torch.__version__),
        "transformers": transformers.__version__,
        "layer": 1,
        "length": args.length,
        "chunk": args.chunk,
        "seed": args.seed,
        "hidden_std": 0.02,
        "embedding_reference": "exact BF16 checkpoint rows, quantized=False",
    }
    if args.fixture_cache.exists():
        cached = torch.load(args.fixture_cache, weights_only=True, mmap=True)
        assert cached["identity"] == identity, "Existing fixture identity differs; use a different cache path"
        report = {**cached["report"], "fixture_cache_hit": True}
    else:
        started = time.monotonic()
        model, weight_keys = load_reference(args.snapshot, config)
        reference_control = control(model, config, args.seed)
        ids, eos_positions = tokens(config, args.length, args.chunk, args.seed)
        hidden = torch.empty((1, args.length, 10240), dtype=torch.bfloat16)
        generator = torch.Generator().manual_seed(args.seed)
        output = {}
        cache = DynamicCache(config=config)
        prefix_state = None
        ranges = [(start, min(start + args.chunk, args.length - 1)) for start in range(0, args.length - 1, args.chunk)]
        ranges.append((args.length - 1, args.length))
        for number, (start, end) in enumerate(ranges):
            hidden[:, start:end] = (torch.randn(1, end - start, 10240, generator=generator) * 0.02).bfloat16()
            current = model(hidden[:, start:end], ids[:, start:end], cache)
            for name, value in current.items():
                if name not in output:
                    output[name] = torch.empty((1, args.length, *value.shape[2:]), dtype=value.dtype)
                output[name][:, start:end] = value
                assert torch.isfinite(value).all(), (name, start)
            if end == args.length - 1:
                prefix_state = state_snapshot(cache)
            if (number + 1) % 16 == 0 or end == args.length:
                print(
                    json.dumps(
                        {
                            "cpu_reference_rows": end,
                            "target_rows": args.length,
                            "chunks": number + 1,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
        report = {
            "scope": __doc__,
            "device_executed": False,
            "capability_claim": False,
            "layer": 1,
            "advertised_context": config.max_position_embeddings,
            "reference_rows": args.length,
            "nonaligned_prefill": args.length - 1,
            "final_single_token_decode_position": args.length - 1,
            "reference_control": reference_control,
            "identity": identity,
            "real_weight_keys": weight_keys,
            "eos_positions": eos_positions,
            "output_shapes": {name: list(value.shape) for name, value in output.items()},
            "selected_exact_embedding_rows": len(model.ple.ple_embedding.ngram_embedding.rows),
            "reference_wall_seconds": time.monotonic() - started,
            "fixture_cache": str(args.fixture_cache),
            "fixture_cache_hit": False,
        }
        data = {
            "identity": identity,
            "input": hidden,
            "token_ids": ids,
            "outputs": output,
            "prefix_state": prefix_state,
            "final_state": state_snapshot(cache),
            "report": report,
        }
        args.fixture_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.fixture_cache.with_name(args.fixture_cache.name + f".tmp-{os.getpid()}")
        try:
            torch.save(data, temporary)
            os.replace(temporary, args.fixture_cache)
        finally:
            temporary.unlink(missing_ok=True)
    assert "ttnn" not in sys.modules, "CPU-only fixture unexpectedly imported TTNN"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "PLE_GDN_CPU_REFERENCE_READY",
                "rows": args.length,
                "output": str(args.output),
                "fixture_cache": str(args.fixture_cache),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--length", type=int, default=262144)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12031)
    parser.add_argument("--fixture-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
