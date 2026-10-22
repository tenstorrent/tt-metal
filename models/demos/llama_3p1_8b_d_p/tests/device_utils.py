# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN readback and cache helpers for device tests."""

import torch

import ttnn


def addresses(tensor):
    return tuple(int(shard.buffer_address()) for shard in ttnn.get_device_tensors(tensor))


def snapshot(cache):
    return [[ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(t)] for t in (cache.k, cache.v)]


def assert_unchanged(cache, before):
    for tensor, snapshots in zip((cache.k, cache.v), before):
        for shard, snapshot in zip(ttnn.get_device_tensors(tensor), snapshots):
            assert torch.equal(ttnn.to_torch(shard), snapshot)


def free_cache(cache):
    cache.k.deallocate(True)
    cache.v.deallocate(True)
