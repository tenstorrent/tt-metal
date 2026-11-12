# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``DeepSeekV41PersistentMoeDecoderLayer``: TP4 decoder layers whose MoE runs on the persistent TP12 MoE.

On the 8x4 Galaxy the MoE takes a 3x4 submesh at row 5 (TP12, every served layer's routed experts, router and
shared expert, one persistent ``fused_experts`` launch), and everything else -- Engram, attention, hyper-
connections, norms, caches -- sits on one 1x4 submesh at row 4, right above it, which is the MoE's only
requester. Each layer's FFN step sends its normed row down the columns and mixes in the reduced row it gets back.

``test_decoder_layer_persistent_moe_decode`` runs ``test_decoder_layer.py``'s cases and checks against the same
reference (the checkpoint's ``Block``), with the server holding only the tested layers.
``test_persistent_moe_decoder_stack`` builds all 40 layers on the one TP4 mesh and the server with all 40 MoEs,
then decodes a few tokens through the whole stack in order: it checks that everything fits (DRAM and L1, next to
the sockets), that the streams stay finite and replicated, and logs the per-token time. It feeds the Engram
layers zero rows, so it is a smoke and latency test, not a PCC test.

Both read the TP12 expert cache; fill it first, or the experts are converted on the test's first run:

    python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --all-layers --tp-size 12 --mesh-rows 3
    pytest models/experimental/deepseek_v41_flash/tests/decode/test_decoder_layer_persistent_moe.py
"""

import contextlib
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_attention import _rope_halves
from models.experimental.deepseek_v41_flash.tests.decode.test_decoder_layer import CASES, _check_decode, _snapshot
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    decoder_layer_weights,
    load_config,
)
from models.experimental.deepseek_v41_flash.tt.decode.attention import build_decode_caches
from models.experimental.deepseek_v41_flash.tt.decode.decoder_layer import DeepSeekV41PersistentMoeDecoderLayer
from models.experimental.deepseek_v41_flash.tt.decode.persistent_moe import PersistentMoE, build_moe_layers
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.decode.hyperconnection import _HC_COLLAPSE_CORES
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

PARENT_MESH = (8, 4)
MOE_SUBMESH = (3, 4)
MOE_OFFSET = (5, 0)
# The one TP4 mesh: the row right above the MoE, which its request multicasts reach and its responses north do.
TP_SIZE = 4
MAIN_ROW = 4
# The stack test: positions decoded through all layers, the first ones untimed.
STACK_WARMUP = int(os.environ.get("PERSISTENT_STACK_WARMUP", 2))
STACK_STEPS = int(os.environ.get("PERSISTENT_STACK_STEPS", 6))
STACK_MAX_SEQ = 256

DEVICE_PARAMS = pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True
)
MESH = pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])


def _setup(mesh_device):
    """``(moe_mesh, main_mesh)``: the 3x4 TP12 MoE submesh and the 1x4 TP4 submesh above it."""
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    moe_mesh = mesh_device.create_submesh(ttnn.MeshShape(*MOE_SUBMESH), ttnn.MeshCoordinate(*MOE_OFFSET))
    main_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(MAIN_ROW, 0))
    return moe_mesh, main_mesh


def _build_server(moe_mesh, main_mesh, config, loader, layer_ids) -> PersistentMoE:
    start = time.perf_counter()

    def log(layer):
        logger.info(f"MoE layer {layer} loaded at TP12 ({time.perf_counter() - start:.0f}s so far)")

    moes = build_moe_layers(config, loader, moe_mesh, WeightCache(CACHE_DIR), layer_ids, log=log)
    return PersistentMoE(moe_mesh, [main_mesh], moes)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@DEVICE_PARAMS
@MESH
@CASES
@torch.no_grad()
def test_decoder_layer_persistent_moe_decode(mesh_device, layers, seq_len):
    moe_mesh, main_mesh = _setup(mesh_device)
    snapshot = _snapshot()
    config = load_config(snapshot)
    server = _build_server(moe_mesh, main_mesh, config, DeepseekV4WeightLoader(snapshot), layers)

    def build_layer(i, weights, layer_cache, prefetch_buffers):
        layer = DeepSeekV41PersistentMoeDecoderLayer(
            config,
            i,
            weights,
            main_mesh,
            server,
            cache=layer_cache,
            weight_dtype=ttnn.bfloat4_b,
            tp_size=TP_SIZE,
            prefetch_buffers=prefetch_buffers,
        )
        return layer, (layer, "moe")

    try:
        _check_decode(main_mesh, layers, seq_len, tp_size=TP_SIZE, build_layer=build_layer)
    finally:
        server.stop()


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@DEVICE_PARAMS
@MESH
@torch.no_grad()
def test_persistent_moe_decoder_stack(mesh_device):
    moe_mesh, main_mesh = _setup(mesh_device)
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    layer_ids = list(range(config.num_hidden_layers))
    hc, hidden = config.hc_mult, config.hidden_size

    server = _build_server(moe_mesh, main_mesh, config, loader, layer_ids)
    try:
        cache = WeightCache(CACHE_DIR)
        # Engram's wkv rides its private prefetcher ring; the first Engram layer builds the mapping.
        prefetch_buffers = {}
        if not ttnn.experimental.is_tensor_prefetcher_supported(main_mesh):
            pytest.skip("Engram's wkv needs the tensor prefetcher (programmable DRAM cores)")
        start = time.perf_counter()
        layers = []
        for i in layer_ids:
            layers.append(
                DeepSeekV41PersistentMoeDecoderLayer(
                    config,
                    i,
                    decoder_layer_weights(loader, i, config),
                    main_mesh,
                    server,
                    cache=cache.sub(f"layers.{i}"),
                    weight_dtype=ttnn.bfloat4_b,
                    tp_size=TP_SIZE,
                    prefetch_buffers=prefetch_buffers,
                )
            )
            logger.info(f"TP4 layer {i} built ({time.perf_counter() - start:.0f}s so far)")
        caches = build_decode_caches(main_mesh, config, layer_ids, STACK_MAX_SEQ)
        rope = {i: _rope_halves(config, snapshot, i, STACK_MAX_SEQ) for i in layer_ids}
        dram = ttnn.get_memory_view(main_mesh, ttnn.BufferType.DRAM)
        l1 = ttnn.get_memory_view(main_mesh, ttnn.BufferType.L1)
        logger.info(
            f"TP4 mesh after building all {len(layers)} layers: DRAM {dram.total_bytes_allocated_per_bank / 2**20:.0f} "
            f"MiB per bank, L1 {l1.total_bytes_allocated_per_bank / 2**10:.0f} KiB per bank"
        )

        streams_config = width_sharded_l1_config(hc, hidden, main_mesh, num_cores=_HC_COLLAPSE_CORES, tile_height=1)
        one_hot = torch.zeros(1, 1, 1, hc)
        one_hot[..., 0] = 1
        torch.manual_seed(0)
        times = []
        with contextlib.ExitStack() as session:
            session.enter_context(tensor_prefetcher_session(main_mesh))
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(main_mesh, cq_id=0)
            for pos in range(STACK_WARMUP + STACK_STEPS):
                # Built ahead of the timed region: host-built per-position inputs, as the eager tests feed them.
                attn_inputs = [layer.self_attn.decode_inputs(pos, *rope[layer.layer_idx]) for layer in layers]
                streams = ttnn.from_torch(
                    torch.randn(1, 1, hc, hidden).bfloat16(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=main_mesh,
                    memory_config=streams_config,
                )
                pre_mix = ttnn.from_torch(one_hot, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=main_mesh)
                ttnn.synchronize_device(main_mesh)
                t0 = time.perf_counter()
                for layer, inputs in zip(layers, attn_inputs):
                    if layer.engram is not None:
                        layer.engram.write_rows(torch.zeros(1, layer.engram.kin))
                        layer.engram.wkv.fetch_weights()
                    streams, pre_mix = layer.decode(streams, pre_mix, caches[layer.layer_idx], **inputs)
                ttnn.synchronize_device(main_mesh)
                elapsed = time.perf_counter() - t0
                if pos >= STACK_WARMUP:
                    times.append(elapsed)

                copies = [ttnn.to_torch(c).float() for c in ttnn.get_device_tensors(streams)]
                spread = max((c - copies[0]).abs().max().item() for c in copies)
                logger.info(
                    f"pos {pos}: {elapsed * 1e3:.2f} ms through {len(layers)} layers, streams |max| "
                    f"{copies[0].abs().max().item():.3g}, rank spread {spread}"
                )
                assert torch.isfinite(copies[0]).all(), f"pos {pos}: non-finite streams out of the stack"
                assert spread == 0, f"pos {pos}: the TP4 ranks disagree on the streams (max abs {spread})"
                ttnn.deallocate(streams)
                ttnn.deallocate(pre_mix)

        us = torch.tensor(times, dtype=torch.float64) * 1e6
        logger.info(
            f"persistent-MoE stack, eager, {len(layers)} layers per token: median {us.median() / 1e3:.2f} ms "
            f"({us.median() / len(layers):.0f} us per layer), min {us.min() / 1e3:.2f} ms, n={len(times)}"
        )
    finally:
        server.stop()
