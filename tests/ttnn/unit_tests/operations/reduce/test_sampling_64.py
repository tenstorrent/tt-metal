# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in 64-candidate draw regressions. Requires a rebuilt TTNN and hardware.

No model/plugin capability is changed by these tests. The old build rejects the
new keyword; a constants-only kernel patch must fail second-tile membership,
row identity, or the eager/trace checks below.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

import ttnn


def _inputs(device, users=32, width=256, k=40, p=1.0):
    generator = torch.Generator().manual_seed(701)
    # Rank is encoded in the global ID, independently shuffled in every row.
    # A wrong tile/face/row lookup cannot accidentally return a valid row ID.
    ranks = torch.stack([torch.randperm(width, generator=generator) for _ in range(users)])
    values = (-ranks.float() / 128).reshape(1, 1, users, width).to(torch.bfloat16)
    indices = (ranks + torch.arange(users)[:, None] * 100000).reshape(1, 1, users, width).int()

    def upload(value, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(value, device=device, dtype=dtype, layout=layout)

    return dict(
        input_values_tensor=upload(values, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        input_indices_tensor=upload(indices, ttnn.int32),
        k=upload(torch.full((users,), k, dtype=torch.int32), ttnn.int32),
        p=upload(torch.full((users,), p), ttnn.bfloat16),
        temp=upload(torch.ones(users), ttnn.bfloat16),
    )


def _stream_seed(device, users):
    # manual_seed requires UINT32 parameter tensors, independently of sampling indices.
    seeds = ttnn.from_torch(
        torch.arange(1, users + 1, dtype=torch.int32) * 7919,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    user_ids = ttnn.from_torch(
        torch.arange(users, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.manual_seed(seeds=seeds, user_ids=user_ids)


def _decode_ids(output, users):
    ids = ttnn.to_torch(output).flatten().long()
    assert ids.numel() == users
    assert torch.equal(ids // 100000, torch.arange(users)), "wrong user or global-index mapping"
    return ids % 100000


@pytest.mark.parametrize("users", [1, 17, 32])
@pytest.mark.parametrize("width", [64, 128, 256])
def test_sampling_64_greedy_cutoff(device, users, width):
    tensors = _inputs(device, users, width, k=40, p=0.0)
    output = ttnn.sampling(**tensors, max_top_k=64, seed=2024)
    assert torch.equal(_decode_ids(output, users), torch.zeros(users, dtype=torch.long))


@pytest.mark.parametrize("signed_zero", [False, True])
def test_sampling_64_stable_ties(device, signed_zero):
    tensors = _inputs(device, k=40, p=1.0)
    values = torch.zeros(1, 1, 32, 256, dtype=torch.bfloat16)
    if signed_zero:
        values[..., 1::2] = -0.0
    tensors["input_values_tensor"] = ttnn.from_torch(
        values,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    indices = (torch.arange(256)[None, :] + torch.arange(32)[:, None] * 100000).reshape(1, 1, 32, 256).int()
    tensors["input_indices_tensor"] = ttnn.from_torch(
        indices,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    _stream_seed(device, 32)
    tail_hits = 0
    for _ in range(32):
        ranks = _decode_ids(ttnn.sampling(**tensors, max_top_k=64), 32)
        assert torch.all(ranks < 40), "stable ties must keep the first 40 input positions"
        tail_hits += (ranks >= 32).sum().item()
    assert tail_hits > 0


def test_sampling_default_capacity_is_explicit_32(device):
    tensors = _inputs(device, k=20, p=0.95)
    implicit = ttnn.sampling(**tensors, seed=2024)
    explicit = ttnn.sampling(**tensors, seed=2024, max_top_k=32)
    assert torch.equal(ttnn.to_torch(implicit), ttnn.to_torch(explicit))
    # Capacity must participate in the program cache even at identical input
    # shapes. Returning to the 32 specialization must reproduce its own draw.
    ttnn.sampling(**tensors, seed=2024, max_top_k=64)
    again = ttnn.sampling(**tensors, seed=2024)
    assert torch.equal(ttnn.to_torch(implicit), ttnn.to_torch(again))


def test_sampling_legacy32_baseline(device):
    """Run unchanged against the original build, then compare the isolated build.

    SAMPLING_LEGACY32_MODE=record|compare and SAMPLING_LEGACY32_ARTIFACT=<json>
    are mandatory. No max_top_k keyword is used in this test, so the original
    extension can run exactly the same fixtures. Never overwrite on comparison.
    """
    mode = os.environ.get("SAMPLING_LEGACY32_MODE")
    if mode not in ("record", "compare"):
        pytest.skip("explicit original-build record / isolated-build compare required")
    artifact = Path(os.environ["SAMPLING_LEGACY32_ARTIFACT"])
    results = []
    for users in (1, 17, 32):
        for width in (64, 128, 256):
            for kind in ("ordered", "equal", "signed_zero"):
                tensors = _inputs(device, users=users, width=width, k=32, p=0.95)
                if kind != "ordered":
                    values = torch.zeros(1, 1, users, width, dtype=torch.bfloat16)
                    if kind == "signed_zero":
                        values[..., 1::2] = -0.0
                    tensors["input_values_tensor"] = ttnn.from_torch(
                        values,
                        device=device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                    )
                ks = torch.tensor([[32, 1, 16, 17, 20, 31][row % 6] for row in range(users)], dtype=torch.int32)
                tensors["k"] = ttnn.from_torch(ks, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
                _stream_seed(device, users)
                draws = [ttnn.to_torch(ttnn.sampling(**tensors)).flatten().tolist() for _ in range(3)]
                results.append(dict(users=users, width=width, kind=kind, k=ks.tolist(), draws=draws))
    record = dict(fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=results)
    if mode == "record":
        artifact.parent.mkdir(parents=True, exist_ok=True)
        with artifact.open("x") as file:
            json.dump(record, file, indent=2)
            file.write("\n")
    else:
        baseline = json.loads(artifact.read_text())
        assert baseline == record, "patched default32 differs from untouched-build fixture outputs"


@pytest.mark.parametrize("k,p", [(40, 1.0), (40, 0.95), (64, 1.0), (64, 0.95)])
def test_sampling_64_tail_and_distribution(device, k, p):
    users, draws = 32, 64
    tensors = _inputs(device, users=users, k=k, p=p)
    _stream_seed(device, users)
    counts = torch.zeros(k, dtype=torch.int64)
    for _ in range(draws):
        ranks = _decode_ids(ttnn.sampling(**tensors, max_top_k=64), users)
        assert torch.all(ranks < k), "masked rank beyond requested top-k was sampled"
        counts += torch.bincount(ranks, minlength=k)

    # Reference uses received BF16 logits/probabilities and the BF16 p input.
    # The strict, near-flat ordering puts a substantial nucleus beyond rank32.
    probabilities = torch.softmax((-torch.arange(k).float() / 128).bfloat16().float(), dim=-1).bfloat16().float()
    cutoff = torch.searchsorted(probabilities.cumsum(0), torch.tensor(p).bfloat16().float(), right=True).item() + 1
    cutoff = min(cutoff, k)
    assert cutoff > 33
    reference = probabilities.clone()
    reference[cutoff:] = 0
    reference /= reference.sum()
    empirical = counts.float() / (users * draws)
    # Allow one nucleus-edge rank for approximate device softmax rounding.
    assert counts[min(cutoff + 1, k) :].sum() == 0
    assert torch.all(counts[32 : cutoff - 1] > 0), "second-tile candidates were lost"
    tv = 0.5 * (empirical - reference).abs().sum().item()
    assert tv < 0.15, f"distribution mismatch: TV={tv}, counts={counts.tolist()}"


def test_sampling_64_mixed_k_and_parameter_reload(device):
    tensors = _inputs(device)
    pattern = [1, 16, 17, 31, 32, 33, 40, 48, 49, 64]
    for shift in (0, 3, 0):
        ks = torch.tensor([pattern[(row + shift) % len(pattern)] for row in range(32)], dtype=torch.int32)
        host_k = ttnn.from_torch(ks, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_k, tensors["k"])
        ranks = _decode_ids(ttnn.sampling(**tensors, max_top_k=64, seed=2024), 32)
        assert torch.all(ranks < ks)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 2000000}], indirect=True)
def test_sampling_64_trace_replay(device):
    device.enable_program_cache()
    tensors = _inputs(device)
    output = ttnn.sampling(**tensors, max_top_k=64, seed=2024)
    # A supplied output participates in the operation's program-cache key.
    # Warm the exact preallocated-output signature that capture will dispatch.
    ttnn.sampling(**tensors, max_top_k=64, seed=2024, output_tensor=output)
    expected = ttnn.to_torch(output).clone()
    warmed_programs = device.num_program_cache_entries()
    assert warmed_programs > 0
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    try:
        try:
            ttnn.sampling(**tensors, max_top_k=64, seed=2024, output_tensor=output)
        finally:
            # Leave capture mode even if dispatch fails, so fixture teardown
            # can synchronize and close the device without a latched capture.
            ttnn.end_trace_capture(device, trace, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert torch.equal(ttnn.to_torch(output), expected)
        assert device.num_program_cache_entries() == warmed_programs
    finally:
        ttnn.release_trace(device, trace)


@pytest.mark.parametrize("capacity", [0, 40, 128])
def test_sampling_rejects_invalid_capacity(device, capacity, expect_error):
    tensors = _inputs(device)
    with expect_error(RuntimeError, "max_top_k must be 32 or 64"):
        ttnn.sampling(**tensors, max_top_k=capacity)


def test_sampling_rejects_width_below_capacity(device, expect_error):
    tensors = _inputs(device, width=32)
    with expect_error(RuntimeError, "input width must be at least max_top_k"):
        ttnn.sampling(**tensors, max_top_k=64)
