# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare 256K block-cyclic prefill with aligned TT prefill, without GPU traces."""

import os
import random
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import _text_token_stream
from models.demos.gemma4_d_p.tests.test_block_cyclic_golden import prefill_chunk
from models.demos.gemma4_d_p.tt.attention.ring_prefill import GlobalRingKVCache
from models.demos.gemma4_d_p.tt.attention.sliding_chunk import SlidingChunkMode
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.model import _cp_chunk_major_row_order


@pytest.mark.timeout(900)
@torch.no_grad()
def test_block_cyclic_matches_aligned_256k():
    hf_model_id = os.getenv("HF_MODEL", "google/gemma-4-31B-it")
    context_len, chunk_size = 262144, 8192
    token_ids = _text_token_stream(hf_model_id)[0, :context_len].tolist()
    assert len(token_ids) == context_len

    aligned_requests = [(start, start + chunk_size) for start in range(0, context_len, chunk_size)]
    rng = random.Random(42)
    rotated_requests = []
    end = 0
    while end < context_len:
        start = max(0, end - rng.randint(0, 2048)) // 32 * 32
        end = min(context_len, start + rng.randint(chunk_size // 2, chunk_size))
        rotated_requests.append((start, end))

    router_config = ttnn.FabricRouterConfig()
    router_config.max_packet_payload_size_bytes = 8192
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router_config)
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(8, 4),
        l1_small_size=16384,
        trace_region_size=int(os.getenv("GEMMA4_PREFILL_TRACE_REGION_SIZE", 256_000_000)),
    )
    trace_ids = {}
    trace_outputs = []
    try:
        mesh_config = MeshConfig(mesh_device)
        _, model, caches, _ = create_tt_model(
            mesh_config, prefill_chunk_size=chunk_size, max_seq_len=context_len, hf_model_id=hf_model_id
        )
        token_mapper = mesh_config.shard_mapper(mesh_dims=(1, None))
        device_tokens = ttnn.from_torch(
            torch.tensor(token_ids[:chunk_size], dtype=torch.int32).unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=token_mapper,
        )
        model._prefill_metadata_external = True
        for mode in SlidingChunkMode:
            model.prefill_metadata.update(slot_idx=0, actual_start=0, actual_end=chunk_size, sliding_mode=mode)
            output = model(model.transform_and_embed_prefill_inputs_device(device_tokens))
            ttnn.synchronize_device(mesh_device)
            output.deallocate(True)
            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            trace_ids[mode] = trace_id
            trace_outputs.append(model(model.transform_and_embed_prefill_inputs_device(device_tokens)))
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            logger.info("Captured {} trace", mode.value)

        row_order = _cp_chunk_major_row_order(context_len, mesh_config.cp_degree, chunk_size).argsort()
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_config.mesh_shape, dims=(2, 1))
        results = {}
        for name, requests in (("aligned", aligned_requests), ("block_cyclic", rotated_requests)):
            # Clear warmup and previous-run KV without changing trace addresses.
            for cache in caches:
                tensors = (cache.kv,) if isinstance(cache, GlobalRingKVCache) else (cache.k, cache.v)
                for tensor in tensors:
                    DRAMZeroFill.op(tensor)
            ttnn.synchronize_device(mesh_device)

            for start, end in requests:
                if name == "aligned":
                    # Stage contiguous tokens independently of block-cyclic packing.
                    host_tokens = ttnn.from_torch(
                        torch.tensor(token_ids[start:end], dtype=torch.int32).unsqueeze(0),
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=token_mapper,
                    )
                    ttnn.copy_host_to_device_tensor(host_tokens, device_tokens)
                    model.prefill_metadata.update(slot_idx=0, actual_start=start, actual_end=end)
                    ttnn.synchronize_device(mesh_device)
                    begin = time.perf_counter()
                    ttnn.execute_trace(mesh_device, trace_ids[SlidingChunkMode.ALIGNED], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                    elapsed_ms = (time.perf_counter() - begin) * 1000
                else:
                    elapsed_ms = prefill_chunk(
                        model, token_ids, start, end, device_tokens=device_tokens, trace_ids=trace_ids
                    )
                logger.info("{} [{}, {}): device={:.3f} ms", name, start, end, elapsed_ms)

            results[name] = ttnn.to_torch(caches[-1].kv, mesh_composer=composer)[0, :, row_order, :]
            assert torch.isfinite(results[name]).all(), f"Non-finite values in {name} KV cache"

        passed, pcc = comp_pcc(results["aligned"], results["block_cyclic"], pcc=0.9999)
        logger.info("Final-layer KV cache, block-cyclic vs aligned TT: PCC {:.8f}", pcc)
        assert passed, f"Block-cyclic vs aligned TT KV cache PCC {pcc:.8f} < 0.9999"
    finally:
        for trace_id in trace_ids.values():
            ttnn.release_trace(mesh_device, trace_id)
        for output in trace_outputs:
            output.deallocate(True)
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
