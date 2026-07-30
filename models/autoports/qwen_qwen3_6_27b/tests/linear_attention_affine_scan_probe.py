# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused probe for a logarithmic gated-delta affine scan.

This is deliberately independent of ``FunctionalDecoder``.  It verifies the
only non-obvious ingredient needed to replace the per-token prefill loop:
batched TTNN matmuls can compose the affine recurrent transforms with a
Hillis-Steele scan.
"""

import argparse

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import _to_device


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def torch_recurrence(decay, beta, key, value, initial):
    state = initial
    states = []
    for index in range(key.shape[1]):
        state = state * decay[:, index, :, None]
        memory_value = key[:, index : index + 1] @ state
        delta = (value[:, index : index + 1] - memory_value) * beta[:, index, :, None]
        state = state + key[:, index : index + 1].transpose(-2, -1) @ delta
        states.append(state)
    return torch.stack(states, dim=1)


def run(groups, sequence, width, dtype_name):
    torch.manual_seed(20260729)
    dtype = getattr(ttnn, dtype_name)
    key = torch.randn(groups, sequence, 1, width, dtype=torch.float32) * 0.05
    value = torch.randn_like(key)
    beta = torch.sigmoid(torch.randn(groups, sequence, 1, 1))
    decay = torch.exp(-torch.rand(groups, sequence, 1, 1) * 0.05)
    initial = torch.randn(groups, 1, width, width, dtype=torch.float32) * 0.01
    expected = torch_recurrence(decay, beta, key, value, initial)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:

        def to_tt(x):
            return _to_device(x, mesh_device=mesh, dtype=dtype)

        key_tt = to_tt(key)
        value_tt = to_tt(value)
        beta_tt = to_tt(beta)
        decay_tt = to_tt(decay)
        identity = to_tt(
            torch.eye(width, dtype=torch.float32).reshape(1, 1, width, width).repeat(groups, sequence, 1, 1)
        )
        zero = to_tt(torch.zeros(groups, sequence, width, width))

        outer = ttnn.matmul(ttnn.transpose(key_tt, -2, -1), key_tt)
        transform = ttnn.multiply(
            decay_tt,
            ttnn.subtract(identity, ttnn.multiply(beta_tt, outer)),
        )
        bias = ttnn.multiply(
            beta_tt,
            ttnn.matmul(ttnn.transpose(key_tt, -2, -1), value_tt),
        )
        distance = 1
        while distance < sequence:
            previous_transform = ttnn.concat([identity[:, :distance], transform[:, :-distance]], dim=1)
            previous_bias = ttnn.concat([zero[:, :distance], bias[:, :-distance]], dim=1)
            old_transform = transform
            transform = ttnn.matmul(old_transform, previous_transform)
            bias = ttnn.add(ttnn.matmul(old_transform, previous_bias), bias)
            distance *= 2

        # TTNN's default rank-4 batched matmul does not broadcast this batch
        # dimension, so make the sequence replication explicit.
        initial_tt = to_tt(initial.repeat(1, sequence, 1, 1))
        actual = ttnn.add(ttnn.matmul(transform, initial_tt), bias)
        actual = ttnn.to_torch(actual, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        # Replicated 1x1 mesh conversion may retain both a mesh and a device
        # singleton prefix, depending on the installed TTNN build.
        actual = actual.reshape(expected.shape)
        score = pcc(actual, expected)
        max_error = (actual.float() - expected).abs().max().item()
        print(f"groups={groups} sequence={sequence} width={width} dtype={dtype_name}")
        print(f"pcc={score:.9f} max_abs_error={max_error:.6f}")
        assert score >= 0.995
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()
    run(args.groups, args.sequence, args.width, args.dtype)
