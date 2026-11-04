# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MoE (router, BF4 routed experts, shared expert) on the 1x4 mesh against the torch reference: layer 0's block on
three independent tokens (the block is stateless), the top-10 expert set within one boundary swap, PCC on the output.
The routed experts are the BF4 cache's, so the PCC floor is below the BF16 blocks'.

test_moe_component.py proves ``Qwen38MoE`` against the pinned Transformers module on the CPU; this is the device side.
The slab call admission (``Qwen38TTNNMoE.admit_slab_call``) leaves the program cache and the allocators as it found
them, admitted or refused. The block-form slab (128 tokens per ``moe_compute`` call, the form a slab takes where the op
refuses its one call) matches the reference on every row, and its routed partial is the 128-row worker's per block.

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_moe_tp.py -v -s
Env:  MODEL_WEIGHTS_DIR, MESH_DEVICE="(1, 4)", QWEN38_CACHE_ROOT (skipped without it)
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.tests.helpers.tp_harness import (
    DEVICE_PARAMS,
    HIDDEN_SIZE,
    assert_tensors_equal,
    download_shards,
    pcc,
)
from models.demos.blackhole.qwen38_flash_next.tt.moe import Qwen38MoE, Qwen38MoEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import LONG_CHUNK_ROWS, MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import SUPPORTED_ROWS, TOP_K, Qwen38TTNNMoE, Qwen38TTNNRouting

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1",
    reason="a device test: set QWEN38_FUSED_DEVICE_TEST=1 on a held four-die line (the no-device sets are masked)",
)

LAYER_INDEX = 0
ROWS = 3


@run_for_blackhole()
@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_moe_tp(tp_harness, pcc_thresholds):
    harness = tp_harness
    builder, layer = harness.build_layer(LAYER_INDEX)
    reference = Qwen38MoE(Qwen38MoEWeights(harness.checkpoint, harness.placement, layer_index=LAYER_INDEX))
    generator = torch.Generator().manual_seed(3000)
    hidden = torch.randn((1, ROWS, HIDDEN_SIZE), generator=generator).to(torch.bfloat16)

    with builder.expert_streamer.layer(LAYER_INDEX) as (packed_w0_w1, packed_w2):
        for row in range(ROWS):
            token = hidden[:, row : row + 1]
            device_token = harness.upload_sharded(token.reshape(1, 1, 1, HIDDEN_SIZE))
            result = layer.mlp.forward(device_token, packed_w0_w1, packed_w2, return_routing=True)
            got = harness.download_sharded(result.hidden_sharded).reshape(1, 1, HIDDEN_SIZE)
            indices = harness.download_replicated(result.routing.indices).to(torch.int64).flatten()
            scores = harness.download_replicated(result.routing.scores).to(torch.float32).flatten()
            for tensor in (device_token, result.hidden_sharded, result.routing.scores, result.routing.indices):
                ttnn.deallocate(tensor)

            expected, routing = reference(token)
            # Router logits are bf16: a near-tie at the top-10 boundary may swap one expert on random inputs.
            reference_experts = routing.indices.flatten().tolist()
            assert (
                len(set(indices.tolist()) & set(reference_experts)) >= len(reference_experts) - 1
            ), f"row {row}: experts {indices.tolist()} vs reference {reference_experts}"
            if indices.tolist() == reference_experts:
                torch.testing.assert_close(scores, routing.scores.flatten().to(torch.float32), rtol=0.02, atol=0.01)
            score = pcc(got, expected)
            logger.info(f"layer {LAYER_INDEX} row {row}: PCC {score:.6f}, experts {indices.tolist()}")
            assert score >= pcc_thresholds["test_moe_tp"], f"row {row}: PCC {score:.6f}"


def _allocator_state(mesh) -> tuple:
    views = [ttnn.get_memory_view(mesh, buffer_type) for buffer_type in (ttnn.BufferType.L1, ttnn.BufferType.DRAM)]
    return mesh.num_program_cache_entries(), [
        (view.total_bytes_allocated_per_bank, view.largest_contiguous_bytes_free_per_bank) for view in views
    ]


@run_for_blackhole()
@pytest.mark.timeout(900)
@pytest.mark.parametrize("rows", [512, 2048])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_moe_slab_call_admission_allocates_and_caches_nothing(tp_harness, rows: int):
    harness = tp_harness
    builder, layer = harness.build_layer(LAYER_INDEX)
    mlp = layer.mlp
    slab = Qwen38TTNNMoE(
        harness.mesh_device,
        mlp.mesh_contract,
        mlp.weights,
        tt_ccl=mlp.tt_ccl,
        collective_topology=mlp.collective_topology,
        rows=rows,
        admitted_rows=SUPPORTED_ROWS + (rows,),
        prefill_dense=mlp.prefill_dense,
    )
    try:
        with builder.expert_streamer.layer(LAYER_INDEX) as (packed_w0_w1, packed_w2):
            before = _allocator_state(harness.mesh_device)
            admitted = slab.admit_slab_call(packed_w0_w1, packed_w2)
            after = _allocator_state(harness.mesh_device)
            logger.debug(f"{rows} rows in one call: {'admitted' if admitted else 'refused'}")
    finally:
        slab.release_owned_buffers()
    assert after == before, f"{rows} rows: (cache entries, [(allocated, largest free) L1, DRAM]) {before} -> {after}"


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("rows", [512, 2048])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_moe_tp_block_slab(tp_harness, pcc_thresholds, rows: int):
    harness = tp_harness
    builder, layer = harness.build_layer(LAYER_INDEX)
    # the slab's resident [gate | up] prefill weight, attached as a build with enable_prefill_slab attaches it
    builder.enable_prefill_slab(rows)
    builder._attach_prefill_dense_weights([layer])
    mlp = layer.mlp
    reference = Qwen38MoE(Qwen38MoEWeights(harness.checkpoint, harness.placement, layer_index=LAYER_INDEX))
    generator = torch.Generator().manual_seed(3000 + rows)
    hidden = torch.randn((1, 1, rows, HIDDEN_SIZE), generator=generator).to(torch.bfloat16)
    slab = Qwen38TTNNMoE(
        harness.mesh_device,
        mlp.mesh_contract,
        mlp.weights,
        tt_ccl=mlp.tt_ccl,
        collective_topology=mlp.collective_topology,
        rows=rows,
        routed_tokens_per_call=LONG_CHUNK_ROWS,
        admitted_rows=SUPPORTED_ROWS + (rows,),
        prefill_dense=mlp.prefill_dense,
    )
    dram, owned = ttnn.DRAM_MEMORY_CONFIG, []
    try:
        assert slab.block_slab and slab.block_instance.rows == LONG_CHUNK_ROWS, (slab.routed_tokens, slab.routed_calls)
        with builder.expert_streamer.layer(LAYER_INDEX) as (packed_w0_w1, packed_w2):
            device_hidden = harness.upload_sharded(hidden)
            owned.append(device_hidden)
            result = slab.forward(device_hidden, packed_w0_w1, packed_w2, return_routing=True)
            owned += [result.hidden_sharded, result.routing.scores, result.routing.indices]
            got = harness.download_sharded(result.hidden_sharded).reshape(1, rows, HIDDEN_SIZE)
            # The routed partial (the blocks concatenated) against the 128-row worker run on each block's rows.
            full_hidden = slab._all_gather_hidden(device_hidden)
            owned.append(full_hidden)
            partial = slab._routed_partial(full_hidden, result.routing, packed_w0_w1, packed_w2)
            owned.append(partial)
            partial_shards = download_shards(partial)
            for start in range(0, rows, LONG_CHUNK_ROWS):
                stop = start + LONG_CHUNK_ROWS
                block_rows = ttnn.slice(full_hidden, (0, 0, start, 0), (1, 1, stop, HIDDEN_SIZE), memory_config=dram)
                block_scores = ttnn.slice(
                    result.routing.scores, (0, 0, start, 0), (1, 1, stop, TOP_K), memory_config=dram
                )
                block_indices = ttnn.slice(
                    result.routing.indices, (0, 0, start, 0), (1, 1, stop, TOP_K), memory_config=dram
                )
                owned += [block_rows, block_scores, block_indices]
                block_partial = slab.block_instance._routed_partial(
                    block_rows, Qwen38TTNNRouting(block_scores, block_indices), packed_w0_w1, packed_w2
                )
                owned.append(block_partial)
                for device, block in enumerate(download_shards(block_partial)):
                    assert_tensors_equal(
                        partial_shards[device][:, :, start:stop],
                        block,
                        f"{rows}-row slab, rows {start}..{stop}, device {device}",
                    )
    finally:
        for tensor in owned:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
        slab.release_owned_buffers()

    expected, _routing = reference(hidden.reshape(1, rows, HIDDEN_SIZE))
    score = pcc(got, expected)
    logger.info(f"layer {LAYER_INDEX} {rows}-row block slab: PCC {score:.6f}")
    assert score >= pcc_thresholds["test_moe_tp"], f"{rows}-row block slab: PCC {score:.6f}"
