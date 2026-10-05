# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 MTP chunked prefill under a captured trace, bit-exact against eager.

Two modes, each comparing two KV cache sets after every chunk with ``torch.equal``:

- ``geometry``: eager with host scalars vs eager with an :class:`MTPTraceGeometry`. Proves the
  branch-free MTP program reproduces the eager one for every per-chunk case the schedule hits.
- ``traced``: the metadata + geometry forward run eagerly vs the same forward captured once and
  replayed per chunk. Proves the replay consumes the per-chunk tensors and nothing goes stale.
"""

from __future__ import annotations

import copy
import gc
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.runners.runner_utils import num_mtp_tokens
from models.demos.deepseek_v3_d_p.tests.mtp_prefill.test_mtp_transformer_chunks import (
    _MESH_PARAMS,
    CHUNK,
    DISPATCH_BUFFER_CAPACITY_FACTOR,
    SCHEDULE_AXIS,
    SP_AXIS,
    TOTAL,
    TP_AXIS,
    _mtp_cache_dir,
    _mtp_union,
    mtp_chunk_stream,
)
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata, write_chunk_metadata
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.device_windows import MTPUnionEmbedding
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.trace_geometry import MTPTraceGeometry
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.tt_mtp import TtMTPPredictor
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import MTP_CACHE_ENV, MTP_CACHE_PREFIX, enable_mtp_indexer_slot
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor, prepare_prefill_mtp_tokens
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

NUM_LAYERS = 1
TRACE_SCHEDULES = ("multiturn-split", "provided-half")
"""``multiturn-split`` runs a short ``p = 0`` chunk, a split chip with every level provided, and a split
short chunk that generates; ``provided-half`` generates only the upper half of the levels."""


def _caches(config, mesh_device, mesh_shape, num_kvpe_layers):
    kvpe = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=TOTAL,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        num_kvpe_cache_layers=num_kvpe_layers,
        num_users=1,
    )
    index = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=TOTAL,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        num_kvpe_cache_layers=num_full_indexer_layers(config),
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )
    return kvpe, index


def _host(t: ttnn.Tensor, mesh_device) -> torch.Tensor:
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))


def _assert_caches_equal(a, b, mesh_device, label: str) -> None:
    for name, ta, tb in (("kvpe", a[0].storage, b[0].storage), ("index", a[1], b[1])):
        ha, hb = _host(ta, mesh_device), _host(tb, mesh_device)
        if not torch.equal(ha, hb):
            diff = (ha.float() != hb.float()).nonzero()
            raise AssertionError(
                f"{label}: {name} cache differs in {diff.shape[0]} of {ha.numel()} elements; first at {diff[0].tolist()}"
            )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("mode", ["geometry", "traced"])
@pytest.mark.parametrize("mtp_levels", (4, 7), ids=["mtp4", "mtp7"])
@pytest.mark.parametrize("schedule", TRACE_SCHEDULES)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.parametrize("use_pretrained", [True], ids=["pretrained"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_mtp_trace(
    variant,
    config_only,
    weight_cache_path,
    mesh_device,
    device_params,
    num_links,
    mode,
    mtp_levels,
    schedule,
    use_pretrained,
    mtp_cfg,
    mtp_state_dict,
    mtp_layer_state_dict,
):
    torch.manual_seed(42)
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    K = mtp_levels
    N_MTP = num_mtp_tokens(K)
    chunks = SCHEDULE_AXIS[schedule](K)

    topology = per_axis_topology(device_params["fabric_config"])
    mesh_shape = list(mesh_device.shape)
    sp_factor, tp_factor = mesh_shape[SP_AXIS], mesh_shape[TP_AXIS]

    config = copy.copy(config_only)
    config.max_seq_len = TOTAL
    config.indexer_types = list(config_only.indexer_types)[:NUM_LAYERS]
    layer_idx = enable_mtp_indexer_slot(config, max(NUM_LAYERS, variant.model_config.NUM_DENSE_LAYERS))

    effective_cache_path = weight_cache_path / f"{sp_factor}x{tp_factor}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp_factor * tp_factor)
    mtp_cache_root = Path(os.getenv(MTP_CACHE_ENV) or weight_cache_path.parent.parent / "glm53_mtp_ttnn_cache")
    mtp_cache_path = mtp_cache_root / f"{variant.name}_bh_{ttnn.get_num_devices()}dev"
    mtp_cache_path = mtp_cache_path / f"{sp_factor}x{tp_factor}_L{NUM_LAYERS}"
    mtp_cache_path = _mtp_cache_dir(mtp_cache_path, Path(ttnn.CONFIG.cache_path) / "glm53_mtp_ttnn_cache")
    init_checker(mtp_cache_path)
    TtMTPPredictor.check_cache_complete(
        mtp_cache_path,
        layer_idx,
        cache_name_prefix=MTP_CACHE_PREFIX,
        experts_per_chip=experts_per_chip,
        model_cfg=variant.model_config,
    )
    init_checker(effective_cache_path)

    predictor = TtMTPPredictor(
        mesh_device,
        config,
        variant.model_config,
        {"mtp": mtp_state_dict, "layer": mtp_layer_state_dict},
        mtp_cfg,
        seq_len=CHUNK,
        num_levels=K,
        layer_idx=layer_idx,
        first_cache_slot=NUM_LAYERS,
        tp_axis=TP_AXIS,
        sp_axis=SP_AXIS,
        num_links=num_links,
        topology=topology,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        dispatch_buffer_capacity_factor=DISPATCH_BUFFER_CAPACITY_FACTOR,
        weight_cache_path=mtp_cache_path,
        cache_name_prefix=MTP_CACHE_PREFIX,
        is_chunked=True,
        max_seq_len=TOTAL,
        slot_num=1,
        layer_num=NUM_LAYERS + K,
    )
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        weight_cache_path=effective_cache_path,
        num_layers=NUM_LAYERS,
        seq_len=CHUNK,
        max_seq_len=TOTAL,
        dispatch_buffer_capacity_factor=DISPATCH_BUFFER_CAPACITY_FACTOR,
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        is_balanced=False,
        padding_side="right",
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        lm_head_is_column_parallel=True,
        is_chunked=True,
        slot_num=1,
        mtp_predictor=predictor,
    )
    gc.collect()
    mesh_device.enable_program_cache()

    caches_a = _caches(config, mesh_device, mesh_shape, transformer.num_kvpe_cache_layers)
    caches_b = _caches(config, mesh_device, mesh_shape, transformer.num_kvpe_cache_layers)
    geometry = MTPTraceGeometry(
        mesh_device,
        sp_factor=sp_factor,
        chunk_size=CHUNK,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_mtp_tokens=N_MTP,
        num_levels=K,
    )
    prompt = list(range(1, TOTAL + K + 1))

    def _chunk(start, actual_isl):
        stream, real_len, provided = mtp_chunk_stream(prompt, start, CHUNK, N_MTP, K, actual_isl=actual_isl)
        return stream, real_len, start + real_len, provided

    def _eager(caches, union, start, real_len, end, provided, *, with_geometry):
        transformer.forward(
            union.trunk,
            caches[0],
            actual_isl=real_len,
            actual_start=start,
            actual_end=end,
            cache_user_id=0,
            index_kv_cache=caches[1],
            mtp_union=union,
            provided_levels=provided,
            input_is_embedded=True,
            mtp_geometry=geometry if with_geometry else None,
        )

    try:
        if mode == "geometry":
            for i, (start, actual_isl) in enumerate(chunks):
                stream, real_len, end, provided = _chunk(start, actual_isl)
                geometry.write(start, end, provided)
                for caches, with_geometry in ((caches_a, False), (caches_b, True)):
                    union = _mtp_union(transformer, stream, N_MTP, K, start, end)
                    _eager(caches, union, start, real_len, end, provided, with_geometry=with_geometry)
                    union.deallocate()
                ttnn.synchronize_device(mesh_device)
                _assert_caches_equal(caches_a, caches_b, mesh_device, f"chunk {i} (start={start} provided={provided})")
                logger.info(f"[mtp trace] geometry chunk {i}: start={start} end={end} provided={provided}/{K} equal")
            return

        def _ids(stream, start, end):
            chunk_ids = prepare_prefill_input_tensor(
                list(stream[:CHUNK]),
                mesh_device,
                sp_factor,
                False,
                mesh_shape,
                SP_AXIS,
                chunk_start=start,
            )
            mtp_ids = prepare_prefill_mtp_tokens(
                list(stream),
                mesh_device,
                sp_factor,
                mesh_shape,
                SP_AXIS,
                num_mtp_tokens=N_MTP,
                num_levels=K,
                chunk_start=start,
                chunk_end=end,
            )
            return chunk_ids, mtp_ids

        def _meta_dev(val):
            return ttnn.from_torch(
                torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
                device=mesh_device,
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        stream0, _, end0, _ = _chunk(*chunks[0])
        trace_chunk_ids, trace_mtp_ids = _ids(stream0, chunks[0][0], end0)
        metadata = ChunkMetadata(_meta_dev(0), _meta_dev(chunks[0][0]), _meta_dev(end0), None)

        def _metadata_forward(caches):
            union = MTPUnionEmbedding.from_ids(trace_chunk_ids, trace_mtp_ids, transformer.mtp_embed_ids, num_levels=K)
            transformer.forward(
                union.trunk,
                caches[0],
                actual_isl=CHUNK,
                actual_start=None,
                actual_end=None,
                cache_user_id=0,
                index_kv_cache=caches[1],
                metadata=metadata,
                mtp_union=union,
                input_is_embedded=True,
                mtp_geometry=geometry,
            )
            union.deallocate()

        def _write_inputs(start, actual_isl):
            stream, _, end, provided = _chunk(start, actual_isl)
            chunk_ids, mtp_ids = _ids(stream, start, end)
            ttnn.copy(chunk_ids, trace_chunk_ids)
            ttnn.copy(mtp_ids, trace_mtp_ids)
            ttnn.deallocate(chunk_ids)
            ttnn.deallocate(mtp_ids)
            write_chunk_metadata(
                metadata, (0, start, end), hf_config=config, mesh_device=mesh_device, chunk_size_global=CHUNK
            )
            geometry.write(start, end, provided)
            return end, provided

        controller = SubDeviceTraceController(mesh_device)
        transformer.set_trace_controller(controller)
        _write_inputs(*chunks[0])
        _metadata_forward(caches_b)
        ttnn.synchronize_device(mesh_device)
        controller.begin_capture()
        _metadata_forward(caches_b)
        controller.end_capture()
        ttnn.synchronize_device(mesh_device)
        assert controller.num_segments > 0, "the capture recorded nothing to replay"
        logger.info(f"[mtp trace] {controller.num_segments} segments, {controller.trace_bytes() / 2**20:.2f} MB")

        for i, (start, actual_isl) in enumerate(chunks):
            end, provided = _write_inputs(start, actual_isl)
            _metadata_forward(caches_a)
            ttnn.synchronize_device(mesh_device)
            # The eager reference leaves the shared expert's keep-alive tensor allocated after the
            # capture; a replay would overwrite it.
            get_tt_ccl(mesh_device).set_shared_rs_input_keepalive(None)
            controller.replay()
            ttnn.synchronize_device(mesh_device)
            _assert_caches_equal(caches_a, caches_b, mesh_device, f"chunk {i} (start={start} provided={provided})")
            logger.info(f"[mtp trace] traced chunk {i}: start={start} end={end} provided={provided}/{K} equal")

        controller.release()
        transformer.set_trace_controller(None)
        for t in (trace_chunk_ids, trace_mtp_ids, *metadata.scalars):
            ttnn.deallocate(t)
    finally:
        geometry.deallocate()
        transformer.release_sub_device_managers()
