# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU full-HF layer1 control for probe_batch seed1982, with actual packed rows.

Only the embedding lookup changes: exact checkpoint selected rows versus rows
decoded from the completed resident-format cache. Every other HF operation,
state update and BF16 expert weight is identical. This does not approximate
device BFP8 experts, prove TT correctness, or waive any .995 gate.
"""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer, Qwen4ExpTextRotaryEmbedding

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_quantization import SelectedRows
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_decoder import metric_passes, metrics
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding_cpu import _quantize_rows


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class CachedRows(SelectedRows):
    def __init__(self, snapshot, index, prefix, cache_dir):
        super().__init__(snapshot, index, prefix, "absmax")
        self.spec = json.loads((cache_dir / "manifest.json").read_text())
        assert self.spec["scale_encoding"] == "uint8_fixed_log_lut" and self.spec["group_size"] == 10
        assert self.spec["rows"] == 320001536
        assert (cache_dir / "words.bin").stat().st_size == self.spec["word_bytes"]
        assert (cache_dir / "scales.bin").stat().st_size == self.spec["scale_bytes"]
        self.words = np.memmap(
            cache_dir / "words.bin", mode="r", dtype=np.uint32, shape=(32, self.spec["words_columns"])
        )
        self.scales = np.memmap(
            cache_dir / "scales.bin", mode="r", dtype=np.uint32, shape=(32, self.spec["scales_columns"])
        )
        self.use_packed = False
        self.last_ids = self.last_exact = self.last_packed = None
        self.parity_checks = 0

    def forward(self, ids):
        exact = super().forward(ids)
        flat = ids.flatten().numpy().astype(np.int64)
        word_offsets = flat[:, None] * 20 + np.arange(20)
        scale_offsets = flat[:, None] * 4 + np.arange(4)
        words = torch.from_numpy(self.words[word_offsets % 32, word_offsets // 32].copy()).long()
        scales = torch.from_numpy(self.scales[scale_offsets % 32, scale_offsets // 32].copy()).long()
        expected_words, expected_scales = _quantize_rows(
            exact.reshape(-1, 160),
            **{
                key: self.spec[key]
                for key in ("quantization", "group_size", "codebook_values", "scale_codebook_values")
            },
        )
        assert torch.equal(words.flatten(), expected_words.long())
        assert torch.equal(scales.flatten(), expected_scales.long())
        self.parity_checks += 1
        codes = torch.stack([(words >> (lane * 4)) & 15 for lane in range(8)], -1).reshape(-1, 160)
        scale_codes = torch.stack([(scales >> (lane * 8)) & 255 for lane in range(4)], -1).reshape(-1, 16)
        packed = (
            torch.tensor(self.spec["codebook_values"])[codes]
            * torch.tensor(self.spec["scale_codebook_values"])[scale_codes].repeat_interleave(10, -1)
        ).bfloat16()
        self.last_ids, self.last_exact, self.last_packed = ids.clone(), exact, packed.reshape_as(exact)
        return self.last_packed if self.use_packed else exact


def load_reference(snapshot, config, cache_dir):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.1."
    keys = [key for key in index if key.startswith(prefix) and ".ngram_embedding.shard_" not in key]
    weights = {}
    for filename in sorted({index[key] for key in keys}):
        with safe_open(snapshot / filename, framework="pt") as source:
            for key in keys:
                if index[key] == filename:
                    weights[key.removeprefix(prefix)] = source.get_tensor(key)
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 1)
    missing = reference.load_state_dict(weights, strict=False, assign=True)
    assert missing.missing_keys == ["ple.ple_embedding.ngram_embedding.weight"] and not missing.unexpected_keys
    reference.ple.ple_embedding.ngram_embedding = CachedRows(
        snapshot, index, prefix + "ple.ple_embedding.ngram_embedding.", cache_dir
    )
    reference.eval()
    return reference


@torch.inference_mode()
def packed_backend_controls(config, weights, fixtures, snapshot, cache_dir):
    """Fresh independent HF model/cache; literal rerun must equal supplied fixtures.

    The packed control changes only lookup values, decoding actual complete-cache
    words/scales and verifying each selected row against the production quantizer.
    It does not force router indices or inject exact reference intermediates."""
    rng = torch.random.get_rng_state()
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 1)
    torch.random.set_rng_state(rng)
    missing = reference.load_state_dict(weights, strict=False, assign=True)
    assert missing.missing_keys == ["ple.ple_embedding.ngram_embedding.weight"] and not missing.unexpected_keys
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    lookup = CachedRows(snapshot, index, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.", cache_dir)
    reference.ple.ple_embedding.ngram_embedding = lookup
    reference.eval()
    captured = {}

    def router(_module, _inputs, output):
        captured["indices"] = output[2].clone()

    handle = reference.mlp.gate.register_forward_hook(router)
    controls = []
    try:
        for prompt, decode, ids, expected_prefill, expected_decode in fixtures:
            case = {}
            length = prompt.shape[1]
            for policy in ("literal", "packed"):
                lookup.use_packed = policy == "packed"
                cache = DynamicCache(config=config)
                for phase, hidden, tokens, expected in (
                    ("prefill", prompt, ids[:, :length], expected_prefill),
                    ("decode", decode, ids[:, length:], expected_decode),
                ):
                    output = reference(
                        hidden, position_embeddings=(None, None), past_key_values=cache, ple_input_ids=tokens
                    )
                    if policy == "literal":
                        assert torch.equal(output, expected), "Independent literal HF rerun must be bit-exact"
                    case.setdefault(phase, {})[policy] = output
                    case[phase][policy + "_router_indices"] = captured["indices"].tolist()
            for phase, hidden in (("prefill", prompt), ("decode", decode)):
                item = case[phase]
                item["sensitivity"] = metrics(item["literal"], item["packed"], hidden)
            controls.append(case)
    finally:
        handle.remove()
    metadata = {
        "manifest": lookup.spec,
        "manifest_sha256": hashlib.sha256((cache_dir / "manifest.json").read_bytes()).hexdigest(),
        "cache_dir": str(cache_dir),
        "complete_file_sizes_verified": True,
        "actual_selected_cache_vs_quantizer_exact_checks": lookup.parity_checks,
        "selected_rows": len(lookup.ids),
        "independent_literal_rerun_bit_exact": True,
        "expert_precision": "unchanged checkpoint BF16; actual TT expert precision is recorded separately",
    }
    return controls, metadata


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(1982)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    reference = load_reference(args.snapshot, config, args.cache_dir)
    # Keep constructor/RNG order identical to probe_batch, including unused rope.
    rope = Qwen4ExpTextRotaryEmbedding(config)
    del rope
    rows = reference.ple.ple_embedding.ngram_embedding
    captured = {}
    handles = []
    for name in ("ple", "attn_hyper_connection", "linear_attn", "mlp_hyper_connection", "mlp"):

        def hook(_module, _inputs, output, key=name):
            captured[key] = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()

        handles.append(getattr(reference, name).register_forward_hook(hook))

    def route(_module, _inputs, output):
        captured["router_logits"], captured["router_weights"], captured["router_indices"] = [v.clone() for v in output]

    handles.append(reference.mlp.gate.register_forward_hook(route))
    report = {
        "scope": __doc__,
        "device_executed": False,
        "seed": 1982,
        "batch": args.batch,
        "snapshot": str(args.snapshot),
        "cache_manifest_sha256": hashlib.sha256((args.cache_dir / "manifest.json").read_bytes()).hexdigest(),
        "hf_source_sha256": hashlib.sha256(Path(inspect.getfile(Qwen4ExpTextDecoderLayer)).read_bytes()).hexdigest(),
        "probe_batch_sha256": hashlib.sha256(Path(__file__).with_name("probe_batch.py").read_bytes()).hexdigest(),
        "expert_precision": "unchanged real BF16 weights; device BFP8 not modeled",
        "requests": [],
    }
    fixtures = []
    for request in range(args.batch):
        length = 1 + request % 3
        prompt = (torch.randn(1, length, 10240) * 0.02).bfloat16()
        decode = (torch.randn(1, 1, 10240) * 0.02).bfloat16()
        ids = torch.randint(0, config.vocab_size, (1, length + 1))
        case = {
            "request": request,
            "length": length,
            "prompt_sha256": tensor_sha(prompt),
            "decode_sha256": tensor_sha(decode),
            "token_ids": ids.tolist(),
            "phases": {},
        }
        values = {}
        # No extra RNG consumers occur in these eval forwards.
        for policy in ("exact", "packed", "exact_repeat"):
            rows.use_packed = policy == "packed"
            cache = DynamicCache(config=config)
            values[policy] = {}
            for phase, hidden, tokens in (("prefill", prompt, ids[:, :length]), ("decode", decode, ids[:, length:])):
                captured.clear()
                output = reference(
                    hidden, position_embeddings=(None, None), past_key_values=cache, ple_input_ids=tokens
                )
                values[policy][phase] = {
                    "output": output.clone(),
                    **captured,
                    "hash_ids": rows.last_ids.clone(),
                    "embeddings": (rows.last_packed if rows.use_packed else rows.last_exact).clone(),
                }
        for phase, hidden in (("prefill", prompt), ("decode", decode)):
            exact, packed, repeat = [values[policy][phase] for policy in ("exact", "packed", "exact_repeat")]
            assert all(torch.equal(value, repeat[key]) for key, value in exact.items())
            assert torch.equal(exact["hash_ids"], packed["hash_ids"])
            value = metrics(exact["output"], packed["output"], hidden)
            intermediate = {}
            for key in ("ple", "attn_hyper_connection", "linear_attn", "mlp_hyper_connection", "mlp"):
                intermediate[key] = metrics(exact[key], packed[key], torch.zeros_like(exact[key]))
            routes = []
            for token, (a, b) in enumerate(
                zip(
                    exact["router_indices"].reshape(-1, config.num_experts_per_tok),
                    packed["router_indices"].reshape(-1, config.num_experts_per_tok),
                )
            ):
                routes.append(
                    {
                        "row": token,
                        "exact": a.tolist(),
                        "packed": b.tolist(),
                        "symmetric_difference": sorted(set(a.tolist()) ^ set(b.tolist())),
                    }
                )
            case["phases"][phase] = {
                "full_decoder": value,
                "all_rows_pass_0_995": metric_passes(value),
                "intermediates": intermediate,
                "routes": routes,
                "literal_rerun_bit_exact": True,
                "hash_ids_identical": True,
            }
        report["requests"].append(case)
        fixtures.append({"prompt": prompt, "decode": decode, "token_ids": ids, "reference": values})
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "request": request,
                    "prefill": case["phases"]["prefill"]["full_decoder"],
                    "decode": case["phases"]["decode"]["full_decoder"],
                }
            ),
            flush=True,
        )
    for handle in handles:
        handle.remove()
    report["selected_rows"] = len(rows.ids)
    report["actual_cache_vs_quantizer_exact_checks"] = rows.parity_checks
    report["all_cpu_control_rows_pass_0_995"] = all(
        case["phases"][phase]["all_rows_pass_0_995"] for case in report["requests"] for phase in ("prefill", "decode")
    )
    if args.fixture_output:
        torch.save({"identity": report, "requests": fixtures}, args.fixture_output)
        report["fixture"] = str(args.fixture_output)
    assert "ttnn" not in sys.modules
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
