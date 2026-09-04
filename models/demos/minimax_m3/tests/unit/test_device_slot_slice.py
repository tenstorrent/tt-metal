# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Byte-parity: device-valued slot select vs host-int slice.

The chunked-prefill cache read (attention/prefill.py) slices one (user,layer) slot out of the packed cache.
A host-int begin index bakes the slot into any captured trace, so request-mode tracing reads the slot from
a persistent device tensor instead. ttnn.slice's device-tensor path is a partition-select (slice_dim split
into num_devices equal parts, one picked by the device begin) that only reshapes slice_dim, so it cannot
also bound the row dim -> slot via the device partition-slice on dim 0, then a host-int slice bounds the
rows. This drives the production helper (``_slot_slice_device`` over a ``MiniMaxKVCache``'s persistent
begin/end tensors) and asserts it is bit-identical to the single host-int slice for arbitrary slots and
row counts.
"""

import pytest
import torch

import ttnn
from models.demos.minimax_m3.tt.attention.kv_cache import MiniMaxKVCache
from models.demos.minimax_m3.tt.attention.prefill import _slot_slice_device

from ..test_factory import parametrize_mesh_with_fabric

# 3 users x 4 layers = 12 packed slots; slot = user * num_layers + layer (kv_cache.py).
N_USERS, N_LAYERS, MAX_ROWS, HEAD_DIM = 3, 4, 256, 128
N_SLOTS = N_USERS * N_LAYERS


def make_kv_cache(packed: ttnn.Tensor) -> MiniMaxKVCache:
    """A cache whose three tensors alias one packed device tensor — only the slot metadata is under test."""
    return MiniMaxKVCache(
        k=packed,
        v=packed,
        index_k=packed,
        num_users=N_USERS,
        num_layers=N_LAYERS,
        max_seq_len=MAX_ROWS,
        sp=1,
        device_slot=True,
    )


def to_device(host: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("slot", [0, 1, 5, N_SLOTS - 1])
@pytest.mark.parametrize("n_rows", [32, 128, MAX_ROWS])
def test_device_slot_slice_parity(mesh_device, device_params, slot, n_rows):
    torch.manual_seed(0)
    packed = to_device(torch.randn(N_SLOTS, 1, MAX_ROWS, HEAD_DIM), mesh_device)
    kv_cache = make_kv_cache(packed)

    ref = ttnn.to_torch(ttnn.slice(packed, (slot, 0, 0, 0), (slot + 1, 1, n_rows, HEAD_DIM)))
    got = ttnn.to_torch(_slot_slice_device(packed, kv_cache, slot % N_LAYERS, slot, n_rows, HEAD_DIM, mesh_device))

    assert tuple(got.shape) == tuple(ref.shape) == (1, 1, n_rows, HEAD_DIM)
    assert torch.equal(got, ref), f"slot={slot} n_rows={n_rows} max_abs_diff={(got - ref).abs().max().item():.3e}"
