# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the CSA block mask: indexer top-k picks folded into causal compressed columns.

The picks come from ``topk_large_indices`` rather than a synthesized index tensor, because the sentinel
it emits for a row with nothing to pick is the whole reason the mask needs a spare column.

Values are only 0 and -inf, both exact in bfloat16, so every check here is exact."""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.compressed_sparse_attention import block_mask

_BATCH = 1
_COMPRESS_RATE = 4
_SENTINEL = 0xFFFFFFFF

# (rows, capacity, k), all with capacity * compress_rate >= rows so the chunk is a real slice of a
# context the cache can hold:
#   - picks cannot cover the causal region, which is the sparse regime,
#   - k reaches the whole cache, which is the regime the reference's top-k collapses to plain causality
#     in and the one the PCC test runs in,
#   - a chunk late in a wide context, so most columns are causally live and the width rounds up to 1024.
_SHAPES = [(512, 128, 64), (512, 128, 128), (256, 512, 64)]


def _causal(rows, capacity, offset):
    """Entry w is visible to the query at position ``offset + i`` iff ``w < (offset + i + 1) // 4``,
    which is the reference's ``causal_threshold``."""
    threshold = ((offset + torch.arange(rows) + 1) // _COMPRESS_RATE).view(1, 1, rows, 1)
    live = torch.arange(capacity).view(1, 1, 1, capacity) < threshold
    return torch.where(live, 0.0, float("-inf")).expand(_BATCH, 1, rows, capacity).contiguous(), live


def _reference(picks, causal, width):
    """The mask by construction: zeros scattered at the picks into an all -inf row, invalid picks parked
    in a spare column, then ANDed with causality."""
    selected = torch.full((_BATCH, 1, picks.shape[2], width), float("-inf"), dtype=torch.float32)
    selected.scatter_(-1, picks & (width - 1), 0.0)
    return causal + selected[..., : causal.shape[-1]]


def _replicate(mesh_device, tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _download(mesh_device, tensor):
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 1)),
    )


def _assert_every_chip_agrees(mesh_device, actual, expected, note=""):
    """The mask build is per-chip local, so it runs SPMD and every chip must reach the host's answer."""
    for sp_rank in range(mesh_device.shape[0]):
        for tp_rank in range(mesh_device.shape[1]):
            got = actual[sp_rank : sp_rank + 1, tp_rank : tp_rank + 1].to(torch.float32)
            where = f"{note}chip ({sp_rank},{tp_rank}): "
            torch.testing.assert_close(got, expected, rtol=0, atol=0, msg=lambda m: where + m)


_MESH = [
    pytest.param(
        (2, 2),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
        id="2x2",
    )
]


@pytest.mark.parametrize("rows, capacity, k", _SHAPES, ids=[f"rows{r}-cap{c}-k{k}" for r, c, k in _SHAPES])
@pytest.mark.parametrize("mesh_device, device_params", _MESH, indirect=["mesh_device", "device_params"])
def test_csa_block_mask(mesh_device, device_params, rows, capacity, k):
    """The mask build is per-chip local, so this runs SPMD and every chip must agree with the host."""
    torch.manual_seed(42)
    width = 1 << capacity.bit_length()  # strictly above capacity, so the last column is the spare one
    offset = capacity * _COMPRESS_RATE - rows  # put the chunk at the end of the context the cache holds

    # Row 0 has nothing valid, so top-k can only return sentinels; row 1 has fewer valid entries than k,
    # so its picks come back part real and part sentinel. The rest are ordinary.
    logits = torch.rand(_BATCH, 1, rows, capacity, dtype=torch.bfloat16)
    logits[:, :, 0, :] = float("-inf")
    logits[:, :, 1, 8:] = float("-inf")

    top_k = ttnn.experimental.topk_large_indices(
        _replicate(mesh_device, logits, layout=ttnn.ROW_MAJOR_LAYOUT), k=k, valid_length=capacity
    )
    picks = _download(mesh_device, top_k)[0:1, 0:1].to(torch.int64)
    assert torch.all(picks[:, :, 0] == _SENTINEL), "a row with nothing valid must be all sentinel"
    assert (picks[:, :, 1] == _SENTINEL).any(), "a row with fewer than k valid entries must sentinel the rest"

    causal, live = _causal(rows, capacity, offset)
    actual_tt = block_mask(
        _replicate(mesh_device, causal),
        top_k,
        template=_replicate(mesh_device, torch.full((_BATCH, 1, rows, width), float("-inf"))),
        zeros=_replicate(mesh_device, torch.zeros(_BATCH, 1, rows, k)),
        dump=width - 1,
    )

    expected = _reference(picks, causal.to(torch.float32), width)
    _assert_every_chip_agrees(mesh_device, _download(mesh_device, actual_tt), expected)

    # The property the attention core relies on, stated directly rather than inferred from the compare:
    # no query may see an entry it is not causally allowed to, whatever the indexer picked.
    assert torch.all(torch.isinf(expected[~live.expand_as(expected)])), "mask leaks a causally hidden entry"
    if k >= capacity:
        # The indexer can reach every entry, so it drops nothing and the mask is plain causality --
        # the degenerate regime the reference's own top-k collapses to.
        torch.testing.assert_close(expected, causal.to(torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize("mesh_device, device_params", _MESH, indirect=["mesh_device", "device_params"])
def test_csa_block_mask_reuses_one_template(mesh_device, device_params):
    """``TtCSA`` allocates ``_sel_template`` once in ``alloc_state`` and hands the same tensor to every
    chunk's mask build, which is only correct while ``ttnn.scatter`` is out-of-place.

    Two calls, one template, and picks biased to opposite ends of the cache: were the scatter to write
    through, the first call's zeros would still be in the template and the second mask would open columns
    the indexer did not pick. The bias is what makes that visible -- those columns are causally live for
    every row here, so a leak cannot hide behind the causal AND."""
    torch.manual_seed(7)
    rows, capacity, k = 256, 128, 32
    width = 1 << capacity.bit_length()
    offset = capacity * _COMPRESS_RATE - rows

    template = _replicate(mesh_device, torch.full((_BATCH, 1, rows, width), float("-inf")))
    zeros = _replicate(mesh_device, torch.zeros(_BATCH, 1, rows, k))
    causal, live = _causal(rows, capacity, offset)
    causal_tt = _replicate(mesh_device, causal)
    # Every row's causal threshold is at least capacity/2 here, so the low half is live throughout.
    assert live[..., : capacity // 2].all(), "the low columns must be causally live for the leak to show"

    ramp = torch.arange(capacity, dtype=torch.float32) / capacity
    logit_sets = [
        torch.rand(_BATCH, 1, rows, capacity) + (1.0 - ramp),  # picks the low entries
        torch.rand(_BATCH, 1, rows, capacity) + ramp,  # picks the high ones
    ]

    pick_sets = []
    for call, logits in enumerate(logit_sets):
        top_k = ttnn.experimental.topk_large_indices(
            _replicate(mesh_device, logits.to(torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT),
            k=k,
            valid_length=capacity,
        )
        picks = _download(mesh_device, top_k)[0:1, 0:1].to(torch.int64)
        pick_sets.append(picks)

        actual_tt = block_mask(causal_tt, top_k, template=template, zeros=zeros, dump=width - 1)
        expected = _reference(picks, causal.to(torch.float32), width)
        _assert_every_chip_agrees(mesh_device, _download(mesh_device, actual_tt), expected, note=f"call {call}: ")

    # If the two calls happened to pick the same columns there would be nothing for a write-through
    # template to corrupt, and the test above would pass for the wrong reason.
    first, second = (set(p.reshape(-1).tolist()) for p in pick_sets)
    assert first - second, "the two calls picked the same columns, so this proves nothing about reuse"
