# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
import ttnn
import numpy as np


def test_dopout(device):
    t = torch.ones(
        (
            4,
            1,
            32,
            64,
        )
    )
    t_tt = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_ratios = []
    s = 124
    prob = 0.2
    for _ in range(1000):
        output = ttnn.experimental.dropout(t_tt, probability=prob, scale=1.0 / (1.0 - prob), seed=s + 1)
        s = s + 1
        output_torch = ttnn.to_torch(output)
        r = 1.0 - (torch.count_nonzero(output_torch) / torch.count_nonzero(t)).item()
        tt_ratios.append(r)

    mean = np.mean(tt_ratios)
    std = np.std(tt_ratios)
    # current dropout has pretty high variance so we just checking with some reasonable nubmers
    assert np.allclose(mean, prob, rtol=0.02)
    assert std < prob


def test_dropout_output_tensor(device):
    """Test that dropout correctly uses the output_tensor parameter for in-place operation.

    This is a regression test for https://github.com/tenstorrent/tt-metal/issues/30284
    """
    t = torch.ones((1, 1, 32, 64))
    tensor = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    # Call dropout with output_tensor=tensor (in-place)
    output = ttnn.experimental.dropout(tensor, probability=0.5, scale=2.0, seed=12345, output_tensor=tensor)

    # Both tensor and output should reference the same underlying data
    # After the operation, the original tensor should be modified
    tensor_torch = ttnn.to_torch(tensor)
    output_torch = ttnn.to_torch(output)

    # The tensors should have the same values (both should show the dropout result)
    assert torch.allclose(
        tensor_torch, output_torch
    ), "output_tensor parameter not working: input tensor was not modified in place"


def test_dropout_rejects_mismatched_output_shape(device, expect_error):
    input_tensor = ttnn.from_torch(
        torch.ones((1, 1, 256, 256)), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    output_tensor = ttnn.from_torch(
        torch.ones((1, 1, 32, 32)), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    with expect_error(RuntimeError, "shape to match the input"):
        ttnn.experimental.dropout(input_tensor, probability=0.5, scale=2.0, seed=1, output_tensor=output_tensor)


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("use_per_device_seed", [False, True])
def test_dropout_seed_distinguishes_cache_entries(device, in_place, use_per_device_seed):
    """Regression guard for dropout's seed static/dynamic contract, on both the distinct-output and
    in-place (output_tensor==input) fast paths, and both program factories.

    `use_per_device_seed` selects the factory and the cache-hit seed-derivation branch in
    override_runtime_arguments: True -> DropoutMeshWorkloadFactory (per-device offset), False ->
    DropoutProgramFactory (raw seed, the branch distributed tt-train uses).

    `seed` is excluded from compute_program_hash (so calls differing only in seed cache-hit) but
    must be re-applied to the cached program on every dispatch. Pins both halves:
      * different seed must NOT grow the cache  -> guards against re-adding seed to the hash.
      * different seed must change the dropout mask -> guards against the frozen-seed bug on the
        fast path (the in_place case is only reachable since resolve_bindings allows input==output).

    Parametrized over `use_per_device_seed` so both cache-hit override hooks are exercised:
    the mesh path (`DropoutMeshWorkloadFactory::override_runtime_arguments`, per-device seed offset)
    and the single-program path (`DropoutProgramFactory::override_runtime_arguments`).
    """
    t = torch.ones((1, 1, 32, 64))
    tensor = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    device.enable_program_cache()
    device.clear_program_cache()

    def run(seed):
        out_kwargs = {"output_tensor": tensor} if in_place else {}
        out = ttnn.experimental.dropout(
            tensor, probability=0.5, scale=2.0, seed=seed, use_per_device_seed=use_per_device_seed, **out_kwargs
        )
        return ttnn.to_torch(out).float().clone()

    out_a = run(1234)
    entries_a = device.num_program_cache_entries()
    run(1234)
    entries_b = device.num_program_cache_entries()
    out_c = run(5678)
    entries_c = device.num_program_cache_entries()

    assert entries_a == 1, "the first dispatch must create exactly one cache entry"
    assert entries_b == entries_a, "same seed must reuse the cached program"
    assert entries_c == entries_a, "a different seed must NOT add a cache entry -- seed is dynamic, not hashed"
    assert not torch.equal(out_a, out_c), "a different seed must change the dropout mask (seed re-patched on fast path)"

    device.disable_and_clear_program_cache()


def test_dropout_cache_hit_rederives_buffer_address(device):
    """On a cache hit the override must re-derive the input/output buffer addresses from the current
    tensors (not reuse the first-miss addresses). Reallocate a fresh input with different VALUES between
    two same-shape dispatches: the kept (non-dropped) outputs must reflect the second input, proving the
    program read the new buffer address."""
    device.enable_program_cache()
    device.clear_program_cache()

    scale = 2.0
    shape = (1, 1, 32, 64)

    # Hold both inputs and both outputs alive across the two dispatches: the second input/output cannot
    # reuse the first's freed slot, so their addresses are guaranteed to differ (asserted below). This
    # makes the re-derivation actually exercised -- if the allocator handed back the first-miss address a
    # frozen (never re-derived) dst_addr would still pass, a false negative.
    a = ttnn.from_torch(torch.ones(shape), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out_a = ttnn.experimental.dropout(a, probability=0.5, scale=scale, seed=1234)  # miss
    ttnn.to_torch(out_a)

    b = ttnn.from_torch(torch.full(shape, 3.0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out_b_tensor = ttnn.experimental.dropout(b, probability=0.5, scale=scale, seed=1234)
    out_b = ttnn.to_torch(out_b_tensor).float()

    assert device.num_program_cache_entries() == 1, "same-shape dispatch must reuse the cached program"
    assert a.buffer_address() != b.buffer_address(), "second input must land at a different address"
    assert (
        out_a.buffer_address() != out_b_tensor.buffer_address()
    ), "second output must land at a different address (dst_addr re-derivation)"
    nonzero = out_b[out_b != 0.0]
    assert nonzero.numel() > 0, "expected some elements to survive dropout"
    assert torch.allclose(
        nonzero, torch.full_like(nonzero, 3.0 * scale)
    ), "kept outputs must be 3.0*scale -- proves the hit re-derived the new input buffer address"

    device.disable_and_clear_program_cache()


@pytest.mark.parametrize("use_per_device_seed", [False, True])
def test_dropout_mask_differs_across_cores(device, use_per_device_seed):
    """Each tile lands on its own core; every core must draw its own mask, and a fixed seed must reproduce the
    same masks whether the program is freshly built or served from the cache."""
    num_tiles = 8
    t = torch.ones((1, 1, num_tiles * 32, 32))
    tensor = ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def run(seed):
        out = ttnn.experimental.dropout(
            tensor, probability=0.5, scale=2.0, seed=seed, use_per_device_seed=use_per_device_seed
        )
        masks = (ttnn.to_torch(out).float().reshape(num_tiles, 32, 32) == 0).reshape(num_tiles, -1)
        for i in range(num_tiles):
            for j in range(i + 1, num_tiles):
                assert not torch.equal(masks[i], masks[j]), f"tiles {i} and {j} share an identical dropout mask"
        return masks

    device.enable_program_cache()
    device.clear_program_cache()

    miss = run(1234)
    hit_same_seed = run(1234)
    hit_new_seed = run(4321)
    assert device.num_program_cache_entries() == 1
    assert torch.equal(miss, hit_same_seed), "a fixed seed must give the same masks on a cache miss and hit"
    assert not torch.equal(miss, hit_new_seed)

    device.clear_program_cache()
    assert torch.equal(hit_new_seed, run(4321)), "a cache hit with a new seed must match a fresh build"
    device.disable_and_clear_program_cache()


def _dropout_masks(output, mesh_device, num_tiles):
    """Each device's [num_tiles, 1024] zero mask, in row-major mesh order."""
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output)]
    assert len(shards) == mesh_device.get_num_devices()
    return [(shard.float().reshape(num_tiles, 32 * 32) == 0) for shard in shards]


# num_tiles = 1: one core, only compute group 1. num_tiles = grid + 3: every core, and a second compute group (the
# cores with one more tile), each with its own compute kernel.
@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4"), pytest.param((8, 4), id="8x4")], indirect=True)
@pytest.mark.parametrize("tiles", ["one", "two_groups"])
def test_dropout_per_device_seed_on_mesh(mesh_device, tiles):
    """use_per_device_seed on a mesh: one program for the mesh with the device seed offset set per device. Every device and
    every core draws its own mask; a new seed on the same cache entry changes every device's masks; a fixed seed
    reproduces them on a hit; use_per_device_seed=False draws the same mask on every device."""
    grid = mesh_device.compute_with_storage_grid_size()
    num_tiles = 1 if tiles == "one" else grid.x * grid.y + 3
    tensor = ttnn.from_torch(
        torch.ones((1, 1, num_tiles * 32, 32)),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    def run(seed, use_per_device_seed=True):
        out = ttnn.experimental.dropout(
            tensor, probability=0.5, scale=2.0, seed=seed, use_per_device_seed=use_per_device_seed
        )
        return _dropout_masks(out, mesh_device, num_tiles)

    mesh_device.clear_program_cache()
    miss = run(1234)
    hit_same_seed = run(1234)
    hit_new_seed = run(4321)
    assert mesh_device.num_program_cache_entries() == 1, "a new seed must reuse the cached program"
    num_devices = len(miss)
    for d in range(num_devices):
        assert torch.equal(miss[d], hit_same_seed[d]), f"device {d}: a fixed seed must reproduce the mask on a hit"
        assert not torch.equal(miss[d], hit_new_seed[d]), f"device {d}: a new seed must change the mask"
        ratio = miss[d].float().mean().item()
        assert 0.3 < ratio < 0.7, f"device {d}: dropped fraction {ratio}"
        for e in range(d + 1, num_devices):
            assert not torch.equal(miss[d], miss[e]), f"devices {d} and {e} share a mask"
    # Every tile's mask is distinct on every device: in particular every core's first tile (cores of both compute
    # groups), so every core draws its own mask.
    for d in range(num_devices):
        distinct = {mask.numpy().tobytes() for mask in miss[d]}
        assert len(distinct) == num_tiles, f"device {d}: {num_tiles - len(distinct)} tiles repeat another's mask"

    same = run(1234, use_per_device_seed=False)
    for d in range(1, num_devices):
        assert torch.equal(same[0], same[d]), f"use_per_device_seed=False: device {d} differs from device 0"


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4")], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
def test_dropout_per_device_seed_trace(mesh_device):
    """A traced per-device-seed dropout replays the masks of its capture, after a later call with another seed."""
    num_tiles = 4
    tensor = ttnn.from_torch(
        torch.ones((1, 1, num_tiles * 32, 32)),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    eager = ttnn.experimental.dropout(tensor, probability=0.5, scale=2.0, seed=77)
    expected = _dropout_masks(eager, mesh_device, num_tiles)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced = ttnn.experimental.dropout(tensor, probability=0.5, scale=2.0, seed=77)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ttnn.experimental.dropout(tensor, probability=0.5, scale=2.0, seed=99)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    replayed = _dropout_masks(traced, mesh_device, num_tiles)
    ttnn.release_trace(mesh_device, trace_id)
    for d, (want, got) in enumerate(zip(expected, replayed)):
        assert torch.equal(want, got), f"device {d}: the trace must replay its capture's mask"
