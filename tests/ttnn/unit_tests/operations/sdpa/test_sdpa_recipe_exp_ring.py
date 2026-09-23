# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Named SDPA precision recipes on exp_ring_joint_scaled_dot_product_attention (single pass).

Runs on two connected Blackholes as a 1x2 FABRIC_1D_RING mesh (exp ring needs Ring topology and two
links). The gate mirrors test_sdpa_recipe_ring.py: for B/C/D/E, each chip's output must equal dense
recipe attention bit-for-bit when dense is given that chip's KV concatenated in the order the exp ring
visits it (own shard first, then the next ring_id; joint KV after shard ring_size-1). That holds when
every visited segment except the last is a whole number of K512 chunks (the ring restarts chunking at
each shard, dense chunks the concatenation); other chips are gated on FP64-reference L2 instead.
A (FAST) runs the legacy exp compute and is gated on L2 against the legacy default call.
"""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import is_blackhole
from .sdpa_recipe_test_utils import PRECISIONS, VARIANTS, digest, metrics, prepare, reference

RING = 2
K_CHUNK = 512
WORKER_L1 = int(os.getenv("TEST_EXP_RING_WORKER_L1", "1344544"))  # the H3 pipeline's worker L1 size


@pytest.fixture(scope="module")
def exp_ring_mesh():
    if not is_blackhole() or ttnn.GetNumAvailableDevices() != 2:
        pytest.skip("This exp ring suite requires exactly two connected Blackholes")
    router = ttnn.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        router,
    )
    mesh = None
    manager = None
    try:
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, RING), worker_l1_size=WORKER_L1, trace_region_size=16777216
        )
        mesh.enable_program_cache()
        hardware = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(hardware.x - 1, hardware.y - 1))}
        )
        subdevice = ttnn.SubDeviceId(0)
        manager = mesh.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
        mesh.load_sub_device_manager(manager)
        mesh.set_sub_device_stall_group([subdevice])
        semaphores = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)]
        yield mesh, subdevice, semaphores
    finally:
        if mesh is not None:
            if manager is not None:
                mesh.reset_sub_device_stall_group()
                mesh.clear_loaded_sub_device_manager()
                mesh.remove_sub_device_manager(manager)
            ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def visit_order(chip, local, logical_n, joint):
    """Segments (source, start, rows) in the order chip `chip` visits them; equal for both row halves."""
    orders = []
    for step in (1, -1):  # top rows ring backward (ring_id increments), bottom rows forward
        segments = []
        for t in range(RING):
            ring_id = (chip + step * t) % RING
            rows = min(max(logical_n - ring_id * local, 0), local)
            if rows:
                segments.append(("primary", ring_id * local, rows))
            if ring_id == RING - 1 and joint:
                segments.append(("joint", 0, joint))
        orders.append(segments)
    assert orders[0] == orders[1], "a 2-chip ring visits shards in the same order in both directions"
    return orders[0]


def generate(heads, q_rows, k_rows, joint, seed=20260923):
    def randn(*shape, offset):
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed + offset)).bfloat16()

    q, k, v = (randn(1, heads, rows, 128, offset=i) for i, rows in enumerate((q_rows, k_rows, k_rows)))
    jq, jk, jv = (randn(1, heads, joint, 128, offset=10 + i) for i in range(3)) if joint else (None,) * 3
    return q, k, v, jq, jk, jv


def run_exp_ring(mesh, subdevice, semaphores, inputs, joints, backing, *, grid, q_chunk, logical_n, **options):
    return ttnn.transformer.exp_ring_joint_scaled_dot_product_attention(
        *inputs,
        *joints,
        persistent_output_buffer_k=backing[0],
        persistent_output_buffer_v=backing[1],
        joint_strategy="rear",
        logical_n=logical_n,
        program_config=options.pop(
            "program_config",
            ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=q_chunk, k_chunk_size=K_CHUNK),
        ),
        dim=2,
        multi_device_global_semaphore=semaphores,
        num_links=2,
        cluster_axis=1,
        mesh_device=mesh,
        topology=ttnn.Topology.Ring,
        subdevice_id=subdevice,
        num_workers_per_link=grid[1] // 2,
        num_buffers_per_channel=16,
        **options,
    )


def upload_case(mesh, host, variant):
    q, k, v, jq, jk, jv = host
    shard = ttnn.ShardTensorToMesh(mesh, dim=2)
    replicate = ttnn.ReplicateTensorToMesh(mesh)

    def up(values, mapper):
        return prepare(
            [ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for x in values], variant
        )

    inputs = up([q, k, v], shard)
    joints = up([jq, jk, jv], replicate) if jq is not None else [None] * 3
    backing = [
        ttnn.allocate_tensor_on_device(list(k.shape), x.dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG)
        for x in inputs[1:]
    ]
    return inputs, joints, backing


# (heads, local rows, joint rows, logical_n or None, grid, q_chunk). grid = (SDPA columns + 1 MUX column,
# 4 rows: the minimum at which the backward (rows 0, 3) and forward (rows 1, 2) MUX cores are distinct);
# heads x segments-per-head must equal the 4 rows for a single pass.
CASES = {
    "aligned": dict(heads=4, local=1024, joint=0, logical_n=None, grid=(5, 4)),
    "joint": dict(heads=2, local=1024, joint=512, logical_n=None, grid=(4, 4)),
    "skip-chunk": dict(heads=4, local=1024, joint=0, logical_n=1536, grid=(5, 4)),
    "subtile-pad-shard": dict(heads=4, local=1024, joint=0, logical_n=777, grid=(5, 4)),
    "joint-tail": dict(heads=4, local=1024, joint=768, logical_n=None, grid=(8, 4)),
    "padded-tails": dict(heads=2, local=768, joint=768, logical_n=1300, grid=(4, 4)),
}
Q128_GRIDS = {"aligned": (5, 4), "joint": (4, 4), "padded-tails": (4, 4)}
Q128_HEADS = {"aligned": 2, "joint": 1, "padded-tails": 1}


@pytest.mark.parametrize("q_chunk", [256, 128], ids=["q256", "q128"])
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", list(CASES))
def test_recipe_exp_ring(exp_ring_mesh, case, variant, q_chunk, record_property):
    mesh, subdevice, semaphores = exp_ring_mesh
    config = dict(CASES[case])
    if q_chunk == 128:
        if case not in Q128_GRIDS:
            pytest.skip("Q128 runs a reduced matrix")
        config.update(grid=Q128_GRIDS[case], heads=Q128_HEADS[case])
    heads, local, joint, grid = config["heads"], config["local"], config["joint"], config["grid"]
    logical_n = config["logical_n"] or RING * local
    q, k, v, jq, jk, jv = host = generate(heads, RING * local, RING * local, joint)
    # Garbage (finite) keys/values past logical_n: the recipe must mask or skip them.
    if logical_n < RING * local:
        for x in (k, v):
            x[..., logical_n:, :] = (8 * torch.randn(x[..., logical_n:, :].shape)).bfloat16()
    inputs, joints, backing = upload_case(mesh, host, variant)
    precision = getattr(ttnn.SDPAPrecision, PRECISIONS.get(variant, "LOW_PRECISION"))
    kwargs = dict(grid=grid, q_chunk=q_chunk, logical_n=logical_n)

    def invoke():
        return run_exp_ring(
            mesh,
            subdevice,
            semaphores,
            inputs,
            joints,
            backing,
            precision=precision,
            inputs_prepared=variant.startswith("E_"),
            **kwargs,
        )

    outputs = invoke()
    actual = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(outputs[0])]
    joint_actual = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(outputs[1])] if joint else []
    # Prepared host values, so the FP64 reference and dense see exactly the device inputs.
    prepared = [ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)) for x in inputs]
    prepared_joint = [ttnn.get_device_tensors(x)[0] for x in joints if x is not None]
    prepared_joint = [ttnn.to_torch(x) for x in prepared_joint]
    queries, gots, ordered = [], [], [[], []]
    for chip in range(RING):
        query = prepared[0][..., chip * local : (chip + 1) * local, :]
        got = actual[chip]
        if joint:
            query = torch.cat([query, prepared_joint[0]], dim=2)
            got = torch.cat([got, joint_actual[chip]], dim=2)
        queries.append(query)
        gots.append(got)
        segments = visit_order(chip, local, logical_n, joint)
        for index in (1, 2):
            source = lambda kind: prepared[index] if kind == "primary" else prepared_joint[index]
            ordered[index - 1].append(
                torch.cat([source(kind)[..., start : start + rows, :] for kind, start, rows in segments], dim=2)
            )
    kv_true = [prepared[i][..., :logical_n, :] for i in (1, 2)]
    if joint:
        kv_true = [torch.cat([x, j], dim=2) for x, j in zip(kv_true, prepared_joint[1:])]
    references = [reference(query, *kv_true) for query in queries]
    observed = [metrics(got, ref) for got, ref in zip(gots, references)]
    for chip, values in enumerate(observed):
        assert values["l2_pct"] is not None, "degenerate (all-zero) reference: inputs did not reach the device"
        for key, value in values.items():
            record_property(f"chip{chip}_{key}", value)
        if variant in ["C", "D"]:
            assert values["l2_pct"] < {"C": 2, "D": 0.4}[variant]

    if variant == "A":
        legacy = run_exp_ring(mesh, subdevice, semaphores, inputs, joints, backing, **kwargs)
        legacy_out = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(legacy[0])]
        legacy_joint = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(legacy[1])] if joint else []
        for chip in range(RING):
            base = torch.cat([legacy_out[chip], legacy_joint[chip]], dim=2) if joint else legacy_out[chip]
            baseline = metrics(base, references[chip])
            record_property(f"chip{chip}_legacy_l2_pct", baseline["l2_pct"])
            record_property(f"chip{chip}_legacy_equal", digest(base) == digest(gots[chip]))
            assert observed[chip]["l2_pct"] <= baseline["l2_pct"] * 1.05 + 1e-6
        return

    # Dense recipe attention over each chip's KV in visiting order; chips on the mesh batch axis.
    dense_q = ttnn.from_torch(
        torch.cat(queries, dim=0), device=mesh, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0)
    )
    dense_kv = [
        ttnn.from_torch(
            torch.cat(x, dim=0).to(prepared[1].dtype),
            dtype=inputs[1].dtype,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        for x in ordered
    ]
    for tensor, host_values in zip(dense_kv, ordered):
        # Prepared (grid-exact) values must round-trip through the host packer unchanged.
        roundtrip = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(tensor)]
        assert all(torch.equal(a.float(), b.float()) for a, b in zip(roundtrip, host_values))
    dense_output = ttnn.transformer.scaled_dot_product_attention(
        dense_q,
        *dense_kv,
        is_causal=False,
        precision=precision,
        inputs_prepared=variant.startswith("E_"),
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(heads, 1), q_chunk_size=q_chunk, k_chunk_size=K_CHUNK
        ),
    )
    dense = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(dense_output)]
    for chip in range(RING):
        segments = visit_order(chip, local, logical_n, joint)
        aligned = all(rows % K_CHUNK == 0 for _, _, rows in segments[:-1])
        equal = digest(gots[chip]) == digest(dense[chip])
        baseline = metrics(dense[chip], references[chip])
        record_property(f"chip{chip}_chunk_aligned", aligned)
        record_property(f"chip{chip}_ordered_dense_equal", equal)
        record_property(f"chip{chip}_dense_l2_pct", baseline["l2_pct"])
        if aligned:
            assert equal, f"chip {chip}: exp ring continuation must equal the dense recipe in visiting order"
        else:
            # Different K512 chunk boundaries than dense: same recipe error level, not the same bits.
            assert observed[chip]["l2_pct"] <= baseline["l2_pct"] * 1.25 + 1e-6

    # Program-cache hit and trace replay reproduce the first result exactly.
    again = invoke()
    assert [digest(ttnn.to_torch(x)) for x in ttnn.get_device_tensors(again[0])] == [digest(x) for x in actual]
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    traced = invoke()
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    try:
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        assert [digest(ttnn.to_torch(x)) for x in ttnn.get_device_tensors(traced[0])] == [digest(x) for x in actual]
        if joint:
            got = [digest(ttnn.to_torch(x)) for x in ttnn.get_device_tensors(traced[1])]
            assert got == [digest(x) for x in joint_actual]
    finally:
        ttnn.release_trace(mesh, trace)


@pytest.mark.parametrize("case", ["aligned", "padded-tails"])
def test_legacy_exp_ring_two_chip(exp_ring_mesh, case, record_property):
    """precision unset: the original exp ring path on this 2-chip geometry, against FP64."""
    mesh, subdevice, semaphores = exp_ring_mesh
    config = CASES[case]
    heads, local, joint, grid = config["heads"], config["local"], config["joint"], config["grid"]
    logical_n = config["logical_n"] or RING * local
    host = generate(heads, RING * local, RING * local, joint)
    inputs, joints, backing = upload_case(mesh, host, "A")
    outputs = run_exp_ring(
        mesh, subdevice, semaphores, inputs, joints, backing, grid=grid, q_chunk=256, logical_n=logical_n
    )
    q, k, v, jq, jk, jv = host
    kv = [k[..., :logical_n, :], v[..., :logical_n, :]]
    if joint:
        kv = [torch.cat([x, j], dim=2) for x, j in zip(kv, (jk, jv))]
    for chip, tensor in enumerate(ttnn.get_device_tensors(outputs[0])):
        query = q[..., chip * local : (chip + 1) * local, :]
        got = ttnn.to_torch(tensor)
        if joint:
            query = torch.cat([query, jq], dim=2)
            got = torch.cat([got, ttnn.to_torch(ttnn.get_device_tensors(outputs[1])[chip])], dim=2)
        observed = metrics(got, reference(query, *kv))
        record_property(f"chip{chip}_l2_pct", observed["l2_pct"])
        # Legacy HiFi2 BF16 streaming measures ~2.5% at K2048; a masking or transport bug is ~100%.
        assert observed["l2_pct"] < 5.0


@pytest.mark.parametrize(
    "case",
    ["multi_pass", "logical_n_tensor", "compute", "exp", "scale", "prepared", "unprepared", "q512", "k256", "l1_q320"],
)
def test_recipe_exp_ring_rejection(exp_ring_mesh, case):
    """Rejected on the host before dispatch. (Causal/balanced/window/cache do not exist on this op;
    stream_q only arises with several passes, so multi_pass covers it.)"""
    mesh, subdevice, semaphores = exp_ring_mesh
    heads = 8 if case == "multi_pass" else 4
    local = 1280 if case == "l1_q320" else 1024
    host = generate(heads, RING * local, RING * local, 0)
    inputs, joints, backing = upload_case(mesh, host, "B")
    grid, q_chunk = (5, 4), 320 if case == "l1_q320" else 256
    options = dict(precision=ttnn.SDPAPrecision.COMPENSATED)
    logical_n = RING * local
    if case == "logical_n_tensor":
        logical_n = ttnn.from_torch(
            torch.tensor([2048], dtype=torch.int32),
            dtype=ttnn.uint32,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
    changes = {
        "compute": dict(compute_kernel_config=ttnn.init_device_compute_kernel_config(mesh.arch())),
        "exp": dict(
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=512, exp_approx_mode=False
            )
        ),
        "scale": dict(scale=0.125),
        "prepared": dict(inputs_prepared=True),
        "unprepared": dict(precision=ttnn.SDPAPrecision.LOW_PRECISION),
        "q512": dict(
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(3, 4), q_chunk_size=512, k_chunk_size=512
            )
        ),
        "k256": dict(
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=256
            )
        ),
    }
    messages = {
        "multi_pass": "single pass",
        "logical_n_tensor": "scalar logical_n",
        "compute": "not both",
        "exp": "exp_approx_mode=false",
        "scale": "default D128 scale",
        "prepared": "LOW_PRECISION requires inputs_prepared",
        "unprepared": "LOW_PRECISION requires inputs_prepared",
        "q512": "Q chunks of 128, 192, 256 or 320",
        "k256": "K512/D128",
        "l1_q320": "needs .* B of L1 per core at Q320",
    }
    options.update(changes.get(case, {}))
    with pytest.raises(RuntimeError, match=messages[case]):
        run_exp_ring(
            mesh,
            subdevice,
            semaphores,
            inputs,
            joints,
            backing,
            grid=grid,
            q_chunk=q_chunk,
            logical_n=logical_n,
            **options,
        )
