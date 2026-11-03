# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QSA decode and verify glue programs versus their chains per row and per die, with trace reuse of the decode forms."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.helpers.tp_harness import (
    DEVICE_PARAMS,
    assert_program_cache_stable,
    assert_shards_equal,
    assert_tensors_equal,
    download_replica,
    replay_trace,
    uint32_bits,
    upload,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import fused, qsa
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE, Qwen38MeshContract
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (
    dram_sharded_matmul_configs,
    dram_sharded_weight_memory_config,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block as qb
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_rows as qr

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")

DIES = MESH_SHAPE[1]
TP_AXIS = 1
BLOCKS = 2048  # resident compressed blocks: an 8192-token cache, two 1024-column score pages
POSITION = 6000  # row 0's position; every row has more complete blocks than its top-k selects
TRACED = {**DEVICE_PARAMS, "trace_region_size": 2_000_000}
BF16, U32, RM = ttnn.bfloat16, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT


def _free(tensors):
    for tensor in tensors:
        if tensor.is_allocated():
            ttnn.deallocate(tensor)


def _position_rows(rows, name):
    """Rows P .. P + rows - 1 of ``emulate_qsa_position_inputs`` (the torch oracle of the device derivation)."""

    return torch.cat(
        [
            qsa.emulate_qsa_position_inputs(POSITION + row, allocated_compressed_blocks=BLOCKS)[name]
            for row in range(rows)
        ],
        dim=2,
    )


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TRACED], indirect=True)
@pytest.mark.parametrize("rows", [1, 5, 32])
def test_widen_partial_against_chain(mesh_device, rows):
    entry = fused.kernel("qsa_widen_partial")
    assert entry.tolerance == fused.BITWISE, f"{entry.name}: tolerance {entry.tolerance}"
    width = qsa.LOCAL_QUERY_WIDTH
    # the out-projection as _project_output_fused runs it: its 16-core L1 output shard is the program's input
    activation, program_config = dram_sharded_matmul_configs(mesh_device, width, qsa.HIDDEN_SIZE, num_cores=16)
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    cached = None
    for seed in (911, 912):
        rng = torch.Generator().manual_seed(seed)
        owned = []
        try:
            attention = upload(
                mesh_device,
                torch.randn(1, 1, rows, DIES * width, generator=rng),
                BF16,
                memory_config=activation,
                shard_dim=3,
            )
            weight = upload(
                mesh_device,
                torch.randn(1, 1, DIES * width, qsa.HIDDEN_SIZE, generator=rng) * 0.02,
                BF16,
                memory_config=dram_sharded_weight_memory_config(mesh_device, width, qsa.HIDDEN_SIZE),
                shard_dim=2,
            )
            out_ws = ttnn.linear(
                attention,
                weight,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=program_config,
                compute_kernel_config=compute,
            )
            owned += [attention, weight, out_ws]
            expected = entry.composed(out_ws)
            owned.append(expected)

            def check(outputs):
                assert_shards_equal(outputs[0], expected, f"fp32 partial, rows={rows}, seed={seed}")

            outputs = (entry.fused(out_ws),)
            owned.extend(outputs)
            check(outputs)
            cached = assert_program_cache_stable(mesh_device, cached)
            replay_trace(mesh_device, lambda: (entry.fused(out_ws),), check, owned)
        finally:
            _free(owned)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TRACED], indirect=True)
@pytest.mark.parametrize("rows", [1, 5, 32])
def test_selection_row_against_chain_per_row(mesh_device, rows):
    # The chain is the 1-row decode body: each row of the program against the chain on that row's inputs.
    entry = fused.kernel("qsa_selection_row")
    assert entry.tolerance == fused.BITWISE, f"{entry.name}: tolerance {entry.tolerance}"
    constants = qsa.qsa_row_constants()
    sentinel_host = constants["sentinel_pad"]
    # one shared offsets row at rows = 1 (the decode step); per row + lane u's KV region start (the lanes)
    offsets = constants["block_offsets"] + (torch.arange(rows) * qsa.COMPRESS_RATIO * BLOCKS).reshape(1, 1, rows, 1)
    keep, fill = (_position_rows(rows, name) for name in ("row_keep_bits", "row_fill"))
    complete = POSITION // qsa.COMPRESS_RATIO  # row 0's complete blocks: the top-k picks distinct ids among them
    cached = None
    for seed in (921, 922):
        rng = torch.Generator().manual_seed(seed)
        ids = torch.stack([torch.randperm(complete, generator=rng)[: qsa.BLOCK_TOPK] for _ in range(rows)])
        ids = ids.reshape(1, 1, rows, qsa.BLOCK_TOPK)
        owned = []
        try:
            sentinel = upload(mesh_device, sentinel_host, U32, layout=RM)
            inputs = [upload(mesh_device, host, U32, layout=RM) for host in (ids, offsets, keep, fill)]
            owned += [sentinel, *inputs]
            expected = []
            for row in range(rows):
                row_inputs = [
                    upload(mesh_device, host[:, :, row : row + 1], U32, layout=RM)
                    for host in (ids, offsets, keep, fill)
                ]
                selected = entry.composed(row_inputs[0], sentinel, *row_inputs[1:])
                owned += [*row_inputs, selected]
                expected.append(uint32_bits(download_replica(selected)).reshape(-1))
                # the chain's integer arithmetic on the host: expand, offsets, template, keep, fill
                expanded = (ids[0, 0, row] << 2).repeat_interleave(qsa.COMPRESS_RATIO) + offsets[0, 0, row]
                template = torch.cat([expanded, sentinel_host.reshape(-1)])
                assert_tensors_equal(
                    expected[-1], (template & keep[0, 0, row]) | fill[0, 0, row], f"chain row {row}, seed={seed}"
                )

            def check(outputs):
                got = uint32_bits(download_replica(outputs[0]))
                for row in range(rows):
                    assert_tensors_equal(got[0, 0, row], expected[row], f"row {row}, rows={rows}, seed={seed}")

            def run():
                return (entry.fused(inputs[0], sentinel, *inputs[1:]),)

            outputs = run()
            owned.extend(outputs)
            check(outputs)
            cached = assert_program_cache_stable(mesh_device, cached)
            replay_trace(mesh_device, run, check, owned)
        finally:
            _free(owned)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TRACED], indirect=True)
@pytest.mark.parametrize("rows", [1, 2, 5, 32])
def test_score_merge_against_chain_per_row(mesh_device, rows):
    entry = fused.kernel("qsa_score_merge")
    assert entry.tolerance == fused.BITWISE, f"{entry.name}: tolerance {entry.tolerance}"
    chunks = BLOCKS // qb.SCORE_CHUNK
    mask_host = _position_rows(rows, "indexer_neg_mask")
    cached = None
    for seed in (931, 932):
        rng = torch.Generator().manual_seed(seed)
        scores = torch.randn(DIES, rows, BLOCKS, generator=rng).to(torch.bfloat16)  # each die's local block scores
        owned = []
        try:
            # page (die * rows + row) * chunks + chunk holds that die's row's chunk: the all_gather of the rows' pages
            gathered = upload(mesh_device, scores.reshape(1, 1, DIES * rows * chunks, qb.SCORE_CHUNK), BF16, layout=RM)
            mask = upload(mesh_device, mask_host, BF16, layout=RM)
            owned += [gathered, mask]
            expected = []
            for row in range(rows):
                pages = upload(
                    mesh_device, scores[:, row].reshape(1, 1, DIES * chunks, qb.SCORE_CHUNK), BF16, layout=RM
                )
                row_mask = upload(mesh_device, mask_host[:, :, row : row + 1], BF16, layout=RM)
                merged = entry.composed(pages, row_mask)
                owned += [pages, row_mask, merged]
                expected.append(download_replica(merged)[0, 0, 0])

            def check(outputs):
                got = download_replica(outputs[0])
                for row in range(rows):
                    assert_tensors_equal(got[0, 0, row], expected[row], f"row {row}, rows={rows}, seed={seed}")

            outputs = (entry.fused(gathered, mask),)
            owned.extend(outputs)
            check(outputs)
            cached = assert_program_cache_stable(mesh_device, cached)
            replay_trace(mesh_device, lambda: (entry.fused(gathered, mask),), check, owned)
        finally:
            _free(owned)


LANES = 4
LANE_CACHE_ROWS = BLOCKS + 32  # a lane's blocks and its never-written tile in the flat compressed cache


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [TRACED], indirect=True)
@pytest.mark.parametrize(
    "lanes_of_rows",
    # the lane decode's B rows; the lane verify's 32-row tile at k = 4 (row r -> lane r // 5, the pad rows lane 0)
    [tuple(range(LANES)), tuple(row // 5 if row < LANES * 5 else 0 for row in range(32))],
    ids=["decode", "verify-k4"],
)
def test_lane_score_rows_of_against_chain(mesh_device, lanes_of_rows):
    rows, cached = len(lanes_of_rows), None
    for seed in (941, 942):
        rng = torch.Generator().manual_seed(seed)
        host = torch.randn(1, 1, 32, LANES * LANE_CACHE_ROWS, generator=rng).to(torch.bfloat16)
        windows = torch.stack(
            [
                host[0, 0, row, lane * LANE_CACHE_ROWS : lane * LANE_CACHE_ROWS + BLOCKS]
                for row, lane in enumerate(lanes_of_rows)
            ]
        ).reshape(1, 1, rows, BLOCKS)
        owned = []
        try:
            scores = upload(mesh_device, host, BF16, layout=RM)
            chain = qsa._lane_score_rows_of_composed(scores, lanes_of_rows, LANES, LANE_CACHE_ROWS, BLOCKS)
            owned += [scores, chain]
            assert_tensors_equal(download_replica(chain), windows, f"chain, rows={rows}, seed={seed}")

            def check(outputs):
                assert_shards_equal(outputs[0], chain, f"lane_score_rows_of, rows={rows}, seed={seed}")

            def run():
                return (qb.lane_score_rows_of(scores, lanes_of_rows, LANES, LANE_CACHE_ROWS, BLOCKS),)

            outputs = run()
            owned.extend(outputs)
            check(outputs)
            cached = assert_program_cache_stable(mesh_device, cached)
            replay_trace(mesh_device, run, check, owned)
        finally:
            _free(owned)


# The verify forms (the MTP rows of one tile) are compared eagerly.


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [1, 5, 32])
def test_score_blocks_rows_against_all_reduce(mesh_device, rows):
    entry = fused.kernel(qr.NAME)
    assert entry.tolerance == fused.BITWISE, f"{entry.name}: tolerance {entry.tolerance}"
    contract = Qwen38MeshContract(tuple(int(device_id) for device_id in mesh_device.get_device_ids()))
    mask_host = _position_rows(rows, "indexer_neg_mask")
    cached = None
    for seed in (951, 952):
        rng = torch.Generator().manual_seed(seed)
        scores = torch.randn(1, DIES, rows, BLOCKS, generator=rng).to(torch.bfloat16)
        owned = []
        try:
            mask = upload(mesh_device, mask_host, BF16, layout=RM)
            owned.append(mask)
            results = []
            # each form takes its own copy of the local partial rows: the program may consume its paged view of them
            for form in (entry.composed, entry.fused):
                score_rows = upload(mesh_device, scores, BF16, layout=RM, shard_dim=1)
                contract.mark_local_partial(score_rows, replicated_reference=mask, expected_shape=(1, 1, rows, BLOCKS))
                owned.append(score_rows)
                results.append(form(score_rows, mask, cluster_axis=TP_AXIS))
                owned.append(results[-1])
            assert_tensors_equal(
                download_replica(results[1]), download_replica(results[0]), f"masked scores, rows={rows}, seed={seed}"
            )
            cached = assert_program_cache_stable(mesh_device, cached)
        finally:
            _free(owned)
