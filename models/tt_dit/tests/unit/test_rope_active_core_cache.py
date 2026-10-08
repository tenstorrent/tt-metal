# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in C16 same-process cache separation and cached new-address correctness."""

import os

import pytest
import torch


@pytest.mark.skip_post_commit
@pytest.mark.skipif(os.environ.get("C16_CACHE_CHECK") != "1", reason="explicit active-core RoPE cache regression")
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("rows,head_dim", [(32, 64), (1216, 128), (4864, 128)])
def test_rope_active_core_cache(mesh_device, rows, head_dim):
    import ttnn
    from models.tt_dit.utils.mochi import get_rot_transformation_mat

    generator = torch.Generator().manual_seed(17)
    shape = (1, 8, rows, head_dim)

    def upload(value):
        return ttnn.from_torch(value.bfloat16(), device=mesh_device, layout=ttnn.TILE_LAYOUT)

    # Keep two distinct input/angle allocations alive so cached runtime-address
    # binding cannot pass by accidentally reusing the first allocation.
    samples = []
    for _ in range(2):
        angles = torch.randn(shape[:-1] + (head_dim // 2,), generator=generator)
        samples.append(
            (
                upload(torch.randn(shape, generator=generator)),
                upload(angles.cos().repeat_interleave(2, -1)),
                upload(angles.sin().repeat_interleave(2, -1)),
            )
        )
    transform = upload(get_rot_transformation_mat())
    config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def run(sample, active):
        return ttnn.experimental.rotary_embedding_llama(
            *samples[sample], transform, compute_kernel_config=config, active_cores_only=active
        )

    def read(tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()

    baseline = run(0, False)
    base_entries = mesh_device.num_program_cache_entries()
    active = run(0, True)
    active_entries = mesh_device.num_program_cache_entries()
    if os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY"):
        pytest.skip("kernel recipe capture only; no cache/correctness evidence")
    assert active_entries == base_entries + 1, "explicit attribute must create a distinct cached descriptor"
    first = read(baseline)
    assert torch.isfinite(first).all() and torch.equal(first, read(active))
    for sample in (1, 0):
        expected, actual = read(run(sample, False)), read(run(sample, True))
        assert (
            mesh_device.num_program_cache_entries() == active_entries
        ), "new buffers must reuse both cached descriptors"
        assert torch.isfinite(actual).all() and torch.equal(actual, expected), "cached target/address coverage mismatch"
        assert torch.equal(actual, first) == (sample == 0), "stale input or nondeterministic restoration"
    print(f"C16_CACHE_PASS rows={rows} head_dim={head_dim} cache_entries={base_entries},{active_entries}")
