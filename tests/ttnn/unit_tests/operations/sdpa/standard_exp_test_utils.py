# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure Torch fixture and oracle for the standard full-tile SDPA exp regression."""
import torch

FALSE_PCC_MIN = 0.9999
FALSE_NL2_MAX = 0.01
TRUE_PCC_MIN = 0.998


def make_fixture(k_chunks):
    if k_chunks not in (1, 2):
        raise ValueError("This regression covers exactly one or two K chunks")
    q = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
    rows = torch.arange(128)
    for head in range(4):
        q[0, head, :, 0] = ((rows + head * 7) % 64 + 1).float() / 4
    key = torch.arange(k_chunks * 512)
    dim = torch.arange(128)
    k = torch.zeros((1, 1, len(key), 128), dtype=torch.float32)
    k[0, 0, :, 0] = ((key % 64) - 63).float() / 16
    # Signed values vary across score groups and all destination faces. They expose
    # probability distortion that a uniform V tensor would hide completely.
    v = ((((key[:, None] % 64) * 17 + dim[None, :] * 13) % 61) - 30).float() / 32
    v = v[None, None]
    mask = torch.zeros((1, 1, 128, len(key)), dtype=torch.float32)
    mask[..., :512:8] = -torch.inf
    if k_chunks == 2:
        k[:, :, 512:, 0] += 1
        v[:, :, 512:] *= 4
        mask[:, :, :64, 512:] = -torch.inf
        mask[:, :, 64:, 768:] = -torch.inf
    return tuple(x.to(torch.bfloat16) for x in (q, k, v, mask))


def reference(q, k, v, mask, scale):
    # Promote the exact BF16 operands, then use independent FP32 math. No original
    # higher-precision fixture values or TTNN operations enter this golden output.
    q, k, v, mask = (x.float() for x in (q, k, v, mask))
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    return torch.softmax((q @ k.transpose(-2, -1)) * scale + mask, dim=-1) @ v


def metrics(expected, actual):
    if expected.shape != actual.shape:
        raise ValueError("Metric shapes differ")
    expected, actual = expected.double().flatten(), actual.double().flatten()
    if expected.numel() < 2 or not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
        raise ValueError("Metrics require finite nonempty vectors")
    a = expected - expected.mean()
    b = actual - actual.mean()
    norm = torch.linalg.vector_norm(expected)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    # PCC has no meaning for a constant vector. Refuse that fixture instead of
    # replacing undefined correlation with a passing number.
    if norm == 0 or denominator == 0:
        raise ValueError("Metrics require nonzero reference energy and nonconstant vectors")
    delta = actual - expected
    return {
        "pcc": (torch.dot(a, b) / denominator).item(),
        "nl2": (torch.linalg.vector_norm(delta) / norm).item(),
        "max_abs": delta.abs().max().item(),
        "rms_abs": delta.square().mean().sqrt().item(),
        "reference_std": expected.std(unbiased=False).item(),
        "reference_rms": expected.square().mean().sqrt().item(),
    }


def passes_false_gate(values):
    return values["pcc"] >= FALSE_PCC_MIN and values["nl2"] <= FALSE_NL2_MAX
