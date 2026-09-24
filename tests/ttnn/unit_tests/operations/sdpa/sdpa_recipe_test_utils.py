# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import torch
import ttnn

VARIANTS = ("A", "B", "C", "D", "E_bf16", "E_bfp8", "E_bfp4")
PRECISIONS = {"A": "FAST", "B": "COMPENSATED", "C": "BALANCED", "D": "ACCURATE"}


def load_baseline():
    """Expand the fixture's implicit no-preparation hashes for A-D.

    Each variant result occupies one JSON record so numerical evidence is easy
    to diff; metrics and provenance are unchanged from the frozen snapshot.
    """
    baseline = json.loads(Path(__file__).with_name("recipe_accuracy_baseline.json").read_text())
    for case in baseline["cases"]:
        for variant, result in case["variants"].items():
            if not variant.startswith("E_"):
                result["prepared_sha256"] = list(case["input_sha256"])
    return baseline


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def make_inputs(k_length, distribution, *, q_length=256, heads=1, seed=20260919):
    tensors = []
    for index, length in enumerate((q_length, k_length, k_length)):
        generator = torch.Generator().manual_seed(seed + index * 1000)
        shape = (1, heads, length, 128)
        x = torch.randn(shape, generator=generator)
        if distribution == "outliers":
            x += 10 * torch.randn(shape, generator=generator) * (torch.rand(shape, generator=generator) < 0.001)
        if distribution == "scaled_qk" and index < 2:
            x *= 2
        if distribution == ("common_q", "common_k", "common_v")[index]:
            x += 32
        if distribution == "uniform" and index == 0:
            x.zero_()
        if distribution == "constant_v" and index == 2:
            x.fill_(1)
        if distribution == "zero_v" and index == 2:
            x.zero_()
        if distribution == "changed_max" and index < 2:
            if index == 0:
                x = x.abs() * 0.25
            else:
                # Every later K chunk has a larger common component. Exercise
                # identity -> changed-max transitions and group-final flushes.
                x += torch.arange(length).div(512, rounding_mode="floor")[None, None, :, None] * 2
        tensors.append(x.bfloat16())
    if distribution == "clipped":
        tensors = [x.clamp(-2, 2) for x in tensors]
    elif distribution == "scaled_down":
        tensors[:2] = [(x.float() * 0.25).bfloat16() for x in tensors[:2]]
    return tensors


def reference(q, k, v, block=4096):
    """Independent stable FP64 attention; bounded memory at 256K context."""
    q = q.double()
    maximum = torch.full((*q.shape[:-1], 1), -torch.inf, dtype=torch.float64)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(q)
    for start in range(0, k.shape[-2], block):
        scores = q @ k[..., start : start + block, :].double().transpose(-1, -2) / q.shape[-1] ** 0.5
        new_maximum = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        correction = torch.exp(maximum - new_maximum)
        probabilities = torch.exp(scores - new_maximum)
        numerator = numerator * correction + probabilities @ v[..., start : start + block, :].double()
        denominator = denominator * correction + probabilities.sum(-1, keepdim=True)
        maximum = new_maximum
    return numerator / denominator


def prepare(inputs, variant):
    if not variant.startswith("E_"):
        return inputs
    dtype = {"E_bf16": ttnn.bfloat16, "E_bfp8": ttnn.bfloat8_b, "E_bfp4": ttnn.bfloat4_b}[variant]
    return [
        ttnn.transformer.prepare_sdpa_input(tensor, is_query=index == 0, dtype=ttnn.bfloat16 if index == 0 else dtype)
        for index, tensor in enumerate(inputs)
    ]


def run(inputs, variant, *, cores=1, q_chunk_size=256):
    return ttnn.transformer.scaled_dot_product_attention(
        *inputs,
        is_causal=False,
        precision=getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION")),
        inputs_prepared=variant.startswith("E_"),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(cores, 1), q_chunk_size=q_chunk_size, k_chunk_size=512
        ),
    )


def metrics(actual, expected):
    actual, expected = actual.double(), expected.double()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    delta = actual - expected
    norm = expected.norm().item()
    row_norm = expected.norm(dim=-1).flatten()
    nonzero = row_norm != 0
    row_errors = 100 * delta.norm(dim=-1).flatten()[nonzero] / row_norm[nonzero]
    a, r = actual.flatten() - actual.mean(), expected.flatten() - expected.mean()
    pcc = (a @ r / (a.norm() * r.norm())).item() if a.norm() > 0 and r.norm() > 1e-12 * norm else None
    return dict(
        l2_pct=100 * delta.norm().item() / norm if norm else None,
        pcc=pcc,
        max_abs=delta.abs().max().item(),
        row_l2_pct_p99=torch.quantile(row_errors, 0.99).item() if row_errors.numel() else None,
        row_l2_pct_max=row_errors.max().item() if row_errors.numel() else None,
    )
