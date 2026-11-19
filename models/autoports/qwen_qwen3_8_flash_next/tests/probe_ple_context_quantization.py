# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU quantization-risk replay of the exact long-context PLE/GDN fixture.

Uses real HF arithmetic and continuous recurrent state with fixture embeddings
at a diagnostic lookup boundary. This excludes MoE and all device arithmetic;
it does not constitute functional-decoder or device-context acceptance.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_context import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding_cpu import _quantize_rows


class FixtureRows(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def forward(self, ids):
        assert torch.equal(ids, self.expected_ids), "HF hash/token history differs from exact fixture"
        self.last_ids, self.last_embeddings = ids, self.prepared
        return self.prepared.reshape(*ids.shape, 160)


def reconstruct(exact, spec):
    """Use production CPU packing and independently extract its exact codes."""
    if spec.get("scale_encoding") == "uint8_fixed_log_lut":
        from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_scale8_hypothesis import reconstruct_scale8

        result, clipped = reconstruct_scale8(exact, spec)
        reconstruct.clipped_scale_groups += clipped
        return result
    packed, scales = _quantize_rows(
        exact.reshape(-1, 160),
        **{key: spec[key] for key in ("quantization", "group_size", "codebook_values")},
    )
    codes = torch.stack([(packed.long() >> (4 * lane)) & 15 for lane in range(8)], -1).reshape(-1, 160)
    scales = scales.reshape(-1, 160 // spec["group_size"]).repeat_interleave(spec["group_size"], -1)
    return (torch.tensor(spec["codebook_values"])[codes] * scales.float()).bfloat16().reshape_as(exact)


reconstruct.clipped_scale_groups = 0


def correlation(expected, actual):
    a, b = expected.float().flatten(0, -2), actual.float().flatten(0, -2)
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    corr = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-30)
    exact = (expected == actual).flatten(0, -2).all(-1)
    return torch.where(exact, 1.0, corr)


def summaries(collected):
    result = {}
    for name, parts in collected.items():
        values = torch.cat(parts)
        result[name] = {
            "minimum_row_pcc": values.min().item(),
            "worst_absolute_position": values.argmin().item(),
            "rows_below_0_995": (values < 0.995).sum().item(),
            "rows": len(values),
            "percentiles_0_1_50_99_100": torch.quantile(values, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])).tolist(),
            "first4096_minimum": values[:4096].min().item(),
            "last4096_minimum": values[-4096:].min().item(),
            "final_row_pcc": values[-1].item(),
        }
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, help="Default replays all advertised-context fixture rows")
    args = parser.parse_args()
    torch.set_num_threads(8)
    started = time.monotonic()
    data = torch.load(args.fixture, mmap=True, weights_only=True)
    spec = json.loads(args.spec.read_text())
    config = AutoConfig.from_pretrained(data["identity"]["snapshot"], local_files_only=True).text_config
    model, _ = load_reference(Path(data["identity"]["snapshot"]), config)
    lookup = FixtureRows()
    model.ple.ple_embedding.ngram_embedding = lookup
    chunk, maximum = data["identity"]["chunk"], data["identity"]["length"]
    length = maximum if args.length is None else args.length
    assert 1 <= length <= maximum
    lookup.expected_ids = data["outputs"]["ngram_ids"][:, :chunk]
    lookup.prepared = data["outputs"]["embeddings"][:, :chunk]
    exact_control = model(data["input"][:, :chunk], data["token_ids"][:, :chunk], DynamicCache(config=config))
    control = {key: torch.equal(value, data["outputs"][key][:, :chunk]) for key, value in exact_control.items()}
    assert all(control.values()), control
    del exact_control
    cache = DynamicCache(config=config)
    collected = {}
    ranges = [(start, min(start + chunk, length)) for start in range(0, length, chunk)]
    if length == maximum:
        ranges[-1:] = [(ranges[-1][0], length - 1), (length - 1, length)]
    for number, (start, end) in enumerate(ranges):
        reference = {key: value[:, start:end] for key, value in data["outputs"].items()}
        lookup.expected_ids = reference["ngram_ids"]
        lookup.prepared = reconstruct(reference["embeddings"], spec)
        actual = model(data["input"][:, start:end], data["token_ids"][:, start:end], cache)
        for name in ("embeddings", "ple_output", "attn_mixed", "attention", "after_attention"):
            assert torch.isfinite(actual[name]).all(), (name, start)
            collected.setdefault(name, []).append(correlation(reference[name], actual[name]))
        collected.setdefault("after_attention_contribution", []).append(
            correlation(
                reference["after_attention"].float() - reference["after_ple"].float(),
                actual["after_attention"].float() - actual["after_ple"].float(),
            )
        )
        if (number + 1) % 16 == 0 or end == length:
            print(
                json.dumps(
                    {
                        "replayed_rows": end,
                        "elapsed_seconds": time.monotonic() - started,
                        "minimum_pcc": {key: min(float(v.min()) for v in vals) for key, vals in collected.items()},
                    }
                ),
                flush=True,
            )
    metrics = summaries(collected)
    report = {
        "scope": __doc__,
        "device_executed": False,
        "fixture": str(args.fixture),
        "fixture_identity": data["identity"],
        "quantization": spec,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": length,
        "continuous_state_replay": True,
        "exact_first_chunk_control": control,
        "scale_groups_clipped": reconstruct.clipped_scale_groups,
        "measured": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "gate": {
            name: metrics[name]["rows_below_0_995"] == 0
            for name in ("ple_output", "attention", "after_attention_contribution")
        },
    }
    report["passed_cpu_risk_gate"] = all(report["gate"].values())
    assert "ttnn" not in sys.modules
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
