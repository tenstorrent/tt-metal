# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The SP relay must leave every rank the entry state the gather + chain gives it, and the final state on the first
and last chronological ranks."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.kda.recurrence import (
    AffineTransform,
    _sp_links,
    distributed_prefix,
    pack_transform,
)
from models.demos.deepseek_v3_d_p.tt.kda.relay_affine_prefix import RelayAffinePrefix
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, make_actual_start

pytestmark = run_for_blackhole()

_ROWS, _HEADS, _KEY = 640, 24, 128


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4",
        )
    ],
    indirect=True,
)
# Only the rank holding the first token orders the relay; "split" checks that an offset within that rank is ignored.
@pytest.mark.parametrize(
    "start", [0, 640, 672, 3232, 4480], ids=["aligned", "rotated", "split", "late-split", "last-rank"]
)
def test_relay_affine_prefix_matches_gather_and_chain(mesh_device, start):
    generator = torch.Generator().manual_seed(start)
    sp_size, tp_size = tuple(mesh_device.shape)
    # BF16-representable transforms make the BF16 transport exact for both paths.
    a = (0.1 * torch.randn(sp_size, _HEADS, _KEY, _KEY, generator=generator)).bfloat16().float()
    b = (0.1 * torch.randn(sp_size, _HEADS, _KEY, _KEY, generator=generator)).bfloat16().float()
    initial = 0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator)

    def per_rank(host):
        return ttnn.from_torch(
            host.reshape((-1,) + host.shape[2:]),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=(sp_size, tp_size)),
        )

    packed = pack_transform(AffineTransform(per_rank(a), per_rank(b)))
    initial_tt = ttnn.from_torch(
        initial,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    actual_start = make_actual_start(mesh_device, start)
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    relay = RelayAffinePrefix(
        mesh_device,
        sequence_parallel_axis=0,
        batch_heads=_HEADS,
        key_dim=_KEY,
        value_dim=_KEY,
        num_links=_sp_links(mesh_device, 0),
        local_rows=_ROWS,
    )
    entry, final = relay(packed, initial_tt, actual_start)
    chained_entry, chained_final = distributed_prefix(
        packed,
        initial_tt,
        sequence_parallel_axis=0,
        compute_config=compute_config,
        actual_start=actual_start,
        local_rows=_ROWS,
    )

    first_rank = (start // _ROWS) % sp_size
    carry = initial.double()
    entries = {}
    for step in range(sp_size):
        rank = (first_rank + step) % sp_size
        entries[rank] = carry.float()
        carry = a[rank].double() @ carry + b[rank].double()
    carry = carry.float()
    # The final state is valid where the relay leaves it: the last chronological rank and the first.
    final_ranks = {first_rank, (first_rank + sp_size - 1) % sp_size}
    devices = zip(
        ttnn.get_device_tensors(entry),
        ttnn.get_device_tensors(final),
        ttnn.get_device_tensors(chained_entry),
        ttnn.get_device_tensors(chained_final),
        strict=True,
    )

    def assert_matches_chain(got, chained, expected, name):
        # The same products at the same fidelity in another operand and accumulation order (the relay accumulates the
        # K-sum on top of B in DST, the chain adds B after the matmul): the relay must be no further from the FP64
        # reference than the gather + chain.
        expected = expected.double()
        relay_error = torch.linalg.vector_norm(got.double() - expected).item()
        chain_error = torch.linalg.vector_norm(chained.double() - expected).item()
        assert relay_error <= 1.01 * chain_error, f"{name}: relay error {relay_error} > chain error {chain_error}"

    for index, (local_entry, local_final, local_chained_entry, local_chained_final) in enumerate(devices):
        rank, tp = divmod(index, tp_size)
        got_entry = ttnn.to_torch(local_entry)
        assert_accurate(entries[rank], got_entry, name=f"entry rank={rank} tp={tp}", nrmse_threshold=0.025)
        assert_matches_chain(got_entry, ttnn.to_torch(local_chained_entry), entries[rank], f"entry rank={rank} tp={tp}")
        if rank in final_ranks:
            got_final = ttnn.to_torch(local_final)
            assert_accurate(carry, got_final, name=f"final rank={rank} tp={tp}", nrmse_threshold=0.03)
            assert_matches_chain(got_final, ttnn.to_torch(local_chained_final), carry, f"final rank={rank} tp={tp}")
