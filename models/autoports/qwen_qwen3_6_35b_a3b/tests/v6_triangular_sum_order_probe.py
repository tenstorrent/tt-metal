# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU proof of installed Torch outer-reduction order for chunk64 row updates."""

import hashlib
import json
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm

from models.autoports.qwen_qwen3_6_35b_a3b.tests import reference

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
COMMIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"
SOURCE_URL = f"https://raw.githubusercontent.com/pytorch/pytorch/{COMMIT}/aten/src/ATen/native/cpu/SumKernel.cpp"


def row_sum4(products):
    """Installed SumKernel row_sum, K<=63: four partials, tail into partial0."""
    length = products.shape[-2]
    partials = [torch.zeros_like(products[..., 0, :]) for _ in range(4)]
    # size_ilp<=15, so multi_row_sum has no completed16-element cascade block.
    end = length // 4 * 4
    for index in range(end):
        partials[index % 4] = partials[index % 4] + products[..., index, :]
    for index in range(end, length):
        partials[0] = partials[0] + products[..., index, :]
    for index in range(1, 4):
        partials[0] = partials[0] + partials[index]
    return torch.zeros_like(partials[0]) + partials[0]  # CastStoreAccumulate


def canonical_outer_sum(products):
    """Actual contiguous [...,K,C] -> [...,C], K=C in1..63, sum-kernel AVX2 FP32."""
    length, columns = products.shape[-2:]
    assert length == columns and 1 <= length <= 63 and products.is_contiguous()
    result = row_sum4(products)
    # SumKernel deliberately excludes AVX512 dispatch; FP32 vector width is8.
    # Four-vector groups use multi_row_sum directly; remaining vectors and
    # scalar columns use row_sum4. Below vector width, groups of4 scalar columns
    # use multi_row_sum and only the final scalar columns use row_sum4.
    cascade_columns = columns // 32 * 32 if columns >= 8 else columns // 4 * 4
    if cascade_columns:
        template = result[..., :cascade_columns]
        completed = torch.zeros_like(template)
        end = length // 16 * 16
        for start in range(0, end, 16):
            partial = torch.zeros_like(template)
            for index in range(start, start + 16):
                partial = partial + products[..., index, :cascade_columns]
            completed = completed + partial
        tail = torch.zeros_like(template)
        for index in range(end, length):
            tail = tail + products[..., index, :cascade_columns]
        combined = tail + completed
        result[..., :cascade_columns] = torch.zeros_like(combined) + combined
    return result


def assess(expected, actual):
    return dict(
        elements=expected.numel(),
        fp32_code_differences=int(
            (expected.contiguous().view(torch.int32) != actual.contiguous().view(torch.int32)).sum()
        ),
        numeric_differences=int((expected != actual).sum()),
        max_abs=float((expected - actual).abs().max()),
    )


def main():
    torch.set_num_threads(4)
    assert torch.version.git_version == COMMIT
    assert torch.backends.cpu.get_cpu_capability() == "AVX512"
    meta = json.loads((DOC / "v5_combined_layer34_stream_probe.json").read_text())
    oracle = torch.load(meta["oracle"], weights_only=False)
    hf = reference.layer(reference.weights(34), 34)
    cache = DynamicCache(config=reference.config())
    captured = {}
    old = hf.linear_attn.chunk_gated_delta_rule

    def capture(q, k, v, **kw):
        captured.update(k=k.clone(), kw=kw)
        return old(q, k, v, **kw)

    hf.linear_attn.chunk_gated_delta_rule = capture
    with torch.no_grad():
        hf.linear_attn(hf.input_layernorm(oracle["prefix"]), cache_params=cache)
    k = l2norm(captured["k"], dim=-1, eps=1e-6).transpose(1, 2).float().contiguous().reshape(1, 32, 2, 64, 128)
    beta = captured["kw"]["beta"].transpose(1, 2).float().contiguous().reshape(1, 32, 2, 64, 1)
    g = captured["kw"]["g"].transpose(1, 2).float().contiguous().reshape(1, 32, 2, 64).cumsum(-1)
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    a = -(((k * beta) @ k.transpose(-1, -2)) * decay).masked_fill(torch.ones(64, 64, dtype=torch.bool).triu(), 0)
    actual_rows, random_rows = [], []
    generator = torch.Generator().manual_seed(20261113)
    for i in range(1, 64):
        row, sub = a[..., i, :i].clone(), a[..., :i, :i].clone()
        products = row.unsqueeze(-1) * sub
        expected = products.sum(-2)
        reproduced = canonical_outer_sum(products)
        actual_rows.append(dict(row=i, **assess(expected, reproduced)))
        a[..., i, :i] = row + expected  # Each next fixture remains canonical.
        for variant in ("gaussian", "mixed_scales", "signed_zeros"):
            values = torch.randn(2, 3, i, i, generator=generator)
            if variant == "mixed_scales":
                values = values * torch.pow(2.0, torch.randint(-20, 20, values.shape, generator=generator))
            elif variant == "signed_zeros":
                values = values * 0.0
            random_rows.append(dict(row=i, variant=variant, **assess(values.sum(-2), canonical_outer_sum(values))))
    all_rows = actual_rows + random_rows
    source = DOC / "v6_torch_SumKernel.cpp.snapshot"
    result = dict(
        status="PASS" if all(row["fp32_code_differences"] == 0 for row in all_rows) else "FAIL",
        classification="CPU exact reduction-order proof on actual and randomized triangular shapes; no hardware",
        torch_version=torch.__version__,
        torch_git_commit=torch.version.git_version,
        cpu_capability=torch.backends.cpu.get_cpu_capability(),
        sum_kernel_vector_width=8,
        source_url=SOURCE_URL,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prefix_sha256=hashlib.sha256(oracle["prefix"].view(torch.uint8).numpy().tobytes()).hexdigest(),
        actual_rows=actual_rows,
        randomized_rows=random_rows,
        total_elements=sum(row["elements"] for row in all_rows),
        total_fp32_code_differences=sum(row["fp32_code_differences"] for row in all_rows),
    )
    (DOC / "v6_triangular_sum_order_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "total_elements", "total_fp32_code_differences")}))
    assert result["status"] == "PASS", [row for row in all_rows if row["fp32_code_differences"]][:4]


if __name__ == "__main__":
    main()
