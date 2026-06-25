# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse MLA / DSA tests for DeepSeek V3.2-family variants.

Dense MLA coverage lives in test_mla.py. This file keeps the sparse reference
path separate while reusing the same TT execution helper and the production mesh
/ fabric axes from the dense MLA tests.
"""

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.dsa_reference import (
    assert_config_matches,
    build_cpu_reference,
    cpu_ref_cache_dir,
    make_hidden,
    run_cpu_reference,
    run_cpu_reference_chunked,
)
from models.demos.deepseek_v3_d_p.tests.mesh_utils import (
    skip_if_seq_too_small_for_sp,
    skip_if_tp1_dense_mla,
    skip_if_tp_exceeds_cap,
)
from models.demos.deepseek_v3_d_p.tests.test_mla import run_mla_inference
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.ttnn.utils_for_testing import assert_with_pcc

SPARSE_OUTPUT_PCC = 0.98
SPARSE_KVPE_PCC = 0.99
SPARSE_SEQ_LENS = [pytest.param(256, marks=pytest.mark.dev), 2048, 4096]
SPARSE_SEQ_IDS = ["seq256", "seq2k", "seq4k"]
SPARSE_VARIANTS = ["deepseek_v32", "glm_5_1"]

# Sparse MLA hardware mesh coverage. Includes Galaxy production (8x4),
# a smaller TP=4 case, and TP=2 so GLM is not skipped out entirely.
# (4, 2) is the full-box LoudBox (8 chips) shape with tp <= GLM's tp_cap=2.
SPARSE_MESH_PARAMS = [(8, 4), (4, 2), (2, 4), (2, 2)]
SPARSE_MESH_IDS = ["8x4", "4x2", "2x4", "2x2"]

SPARSE_DEVICE_PARAMS = [
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
    },
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
    },
    {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
    },
]
SPARSE_DEVICE_IDS = ["line", "ring", "fabric2d"]


def _topology_from_device_params(device_params):
    return (
        ttnn.Topology.Ring
        if device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_1D_RING
        else ttnn.Topology.Linear
    )


def run_sparse_mla_accuracy_case(
    variant, config, mesh_device, seq_len, topology, ds_layer=None, ds_checkpoint=None, ds_repo=None
):
    """Sparse-MLA accuracy: device output + KVPE cache vs MLACPU sparse reference."""
    skip_if_seq_too_small_for_sp(seq_len, mesh_device)
    skip_if_tp1_dense_mla(seq_len, mesh_device)
    skip_if_tp_exceeds_cap(variant, mesh_device)

    logger.info(
        f"[{variant.name}] sparse MLA accuracy start: seq_len={seq_len} "
        f"mesh={tuple(mesh_device.shape)} topology={topology}"
    )
    logger.debug(f"[{variant.name}] sparse MLA accuracy: building CPU reference and TT weights")
    args, mla_cpu, weights, src_tag = build_cpu_reference(
        variant, seq_len, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo
    )
    assert_config_matches(config, args)
    config.max_seq_len = seq_len
    logger.debug(f"[{variant.name}] sparse MLA accuracy: reference source tag={src_tag}")

    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    logger.debug(f"[{variant.name}] sparse MLA accuracy: initializing KVPE cache mesh_shape={mesh_shape}")
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    logger.info(f"[{variant.name}] sparse MLA accuracy: running TT inference")
    tt_output, hidden_states, _, shard_dims = run_mla_inference(
        config=config,
        weights=weights,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        topology=topology,
        tt_kvpe_cache=tt_kvpe_cache,
    )

    cache_dir = cpu_ref_cache_dir(variant)
    logger.info(f"[{variant.name}] sparse MLA accuracy: running CPU reference")
    ref_output, ref_kvpe = run_cpu_reference(
        args, mla_cpu, hidden_states, seq_len, cache_dir, cache_tag=f"{src_tag}_funcidx_max{args.max_seq_len}"
    )

    logger.debug(f"[{variant.name}] sparse MLA accuracy: collecting TT output shard_dims={shard_dims}")
    tt_output_cpu = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
    if seq_len > 2048:
        for name, sl in [("rows<2048", slice(0, 2048)), ("rows>=2048", slice(2048, seq_len))]:
            _, m = comp_pcc(ref_output[:, sl], tt_output_cpu[0, :, sl], 0)
            logger.info(f"[{variant.name}] band {name}: {m}")
    _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output_cpu, SPARSE_OUTPUT_PCC)
    logger.info(f"[{variant.name}] Output PCC: {pcc_message}")

    logger.debug(f"[{variant.name}] sparse MLA accuracy: collecting KVPE cache")
    tt_kvpe = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)[:1, :1]
    kv = config.kv_lora_rank
    _, kv_pcc = assert_with_pcc(ref_kvpe[..., :kv], tt_kvpe[..., :kv], SPARSE_KVPE_PCC)
    _, pe_pcc = assert_with_pcc(ref_kvpe[..., kv:], tt_kvpe[..., kv:], SPARSE_KVPE_PCC)
    logger.info(f"[{variant.name}] KVPE cache PCC: kv={kv_pcc} pe={pe_pcc}")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[{variant.name}] sparse MLA accuracy complete")


def run_sparse_mla_determinism_case(
    variant, config, mesh_device, seq_len, n_runs, topology, ds_layer, ds_checkpoint, ds_repo
):
    """Run the same sparse MLA case repeatedly and compare outputs."""
    skip_if_seq_too_small_for_sp(seq_len, mesh_device)
    skip_if_tp_exceeds_cap(variant, mesh_device)

    logger.info(
        f"[{variant.name}] sparse MLA determinism start: seq_len={seq_len} "
        f"mesh={tuple(mesh_device.shape)} topology={topology} n_runs={n_runs}"
    )
    logger.debug(f"[{variant.name}] sparse MLA determinism: building CPU reference and TT weights")
    args, _, weights, _ = build_cpu_reference(
        variant, seq_len, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo
    )
    assert_config_matches(config, args)
    config.max_seq_len = seq_len
    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1

    baseline = None
    for run_idx in range(n_runs):
        logger.info(f"[{variant.name}] sparse MLA determinism run {run_idx + 1}/{n_runs}: running TT inference")
        logger.debug(f"[{variant.name}] sparse MLA determinism run {run_idx + 1}: initializing KVPE cache")
        tt_kvpe_cache = init_kvpe_cache(
            kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=1,
        )
        tt_output, _, _, shard_dims = run_mla_inference(
            config=config,
            weights=dict(weights),
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            is_balanced=False,
            topology=topology,
            tt_kvpe_cache=tt_kvpe_cache,
        )
        logger.debug(f"[{variant.name}] sparse MLA determinism run {run_idx + 1}: collecting TT output")
        current = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
        ).to(torch.bfloat16)
        ttnn.synchronize_device(mesh_device)

        if baseline is None:
            baseline = current
            logger.debug(f"[{variant.name}] sparse MLA determinism run {run_idx + 1}: stored baseline")
            continue

        logger.debug(f"[{variant.name}] sparse MLA determinism run {run_idx + 1}: comparing with baseline")
        exact = torch.equal(baseline, current)
        _, msg = assert_with_pcc(baseline.float(), current.float(), 0.9999)
        logger.info(f"[{variant.name}] determinism run0 vs run{run_idx}: exact={exact} pcc={msg}")
    logger.info(f"[{variant.name}] sparse MLA determinism complete")


def run_sparse_mla_chunked_case(
    variant, config, mesh_device, seq_len, chunk, ds_layer, ds_checkpoint, ds_repo, ds_input
):
    """Sparse chunked prefill: compare chunked ttMLA against MLACPU sparse chunked truth."""
    skip_if_seq_too_small_for_sp(seq_len, mesh_device)
    skip_if_tp_exceeds_cap(variant, mesh_device)
    if mesh_device.shape[1] == 1:
        pytest.skip("chunked MLA epilogue exceeds L1 without TP head-sharding (TP=1)")

    seed = 42
    logger.info(
        f"[{variant.name}] sparse MLA chunked start: seq_len={seq_len} chunk={chunk} "
        f"mesh={tuple(mesh_device.shape)} ds_input={bool(ds_input)}"
    )
    logger.debug(f"[{variant.name}] sparse MLA chunked: building CPU reference and TT weights")
    args, mla_cpu, weights, src_tag = build_cpu_reference(variant, seq_len, seed, ds_layer, ds_checkpoint, ds_repo)
    assert_config_matches(config, args)
    config.max_seq_len = seq_len

    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    logger.debug(f"[{variant.name}] sparse MLA chunked: initializing KVPE cache mesh_shape={mesh_shape}")
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    logger.debug(f"[{variant.name}] sparse MLA chunked: constructing TT module and indexed RoPE tensors")
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        layer_num=1,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(seq_len, chunk)

    hidden = make_hidden(seq_len, config.hidden_size, seed, ds_input)
    if ds_input:
        src_tag = f"{src_tag}_in{abs(hash(ds_input)) % 10**8}"
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2

    outs = []
    num_chunks = (seq_len + chunk - 1) // chunk
    for chunk_idx, s in enumerate(range(0, seq_len, chunk)):
        logger.info(f"[{variant.name}] sparse MLA chunked TT chunk {chunk_idx + 1}/{num_chunks}: actual_start={s}")
        tt_x = ttnn.from_torch(
            hidden[:, s : s + chunk].unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )
        out = mla_tt.forward(tt_x, rope_tensors, tt_kvpe_cache, actual_start=s)
        outs.append(
            ttnn.to_torch(
                out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
            ).to(torch.bfloat16)
        )
    tt_output = torch.cat(outs, dim=2)

    cache_dir = cpu_ref_cache_dir(variant)
    logger.info(f"[{variant.name}] sparse MLA chunked: running CPU reference")
    ref_output, ref_kvpe = run_cpu_reference_chunked(
        args, mla_cpu, hidden, seq_len, chunk, cache_dir, cache_tag=f"{src_tag}_funcidx"
    )

    for s in range(0, seq_len, chunk):
        _, m = comp_pcc(ref_output[:, s : s + chunk], tt_output[0, :, s : s + chunk], 0)
        logger.info(f"[{variant.name}] chunk@{s}: {m}")

    sp = mesh_device.shape[0]
    logger.debug(f"[{variant.name}] sparse MLA chunked: collecting KVPE cache")
    cache_sr = ttnn.to_torch(
        tt_kvpe_cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)[:, :1]
    p = blockcyclic_positions(sp, chunk, cache_sr.shape[2])
    nat = torch.empty(cache_sr.shape[2], cache_sr.shape[-1], dtype=torch.bfloat16)
    nat[p] = cache_sr[0, 0]
    _, m = comp_pcc(ref_kvpe, nat[:seq_len].unsqueeze(0), 0)
    logger.info(f"[{variant.name}] kvpe prefix: {m}")

    _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output, SPARSE_OUTPUT_PCC)
    logger.info(f"[{variant.name}] Chunked output PCC: {pcc_message}")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[{variant.name}] sparse MLA chunked complete")


@pytest.mark.parametrize("mesh_device", SPARSE_MESH_PARAMS, ids=SPARSE_MESH_IDS, indirect=True)
@pytest.mark.parametrize("device_params", SPARSE_DEVICE_PARAMS, ids=SPARSE_DEVICE_IDS, indirect=True)
@pytest.mark.parametrize("seq_len", SPARSE_SEQ_LENS, ids=SPARSE_SEQ_IDS)
@pytest.mark.parametrize("variant", SPARSE_VARIANTS, indirect=True, ids=SPARSE_VARIANTS)
@pytest.mark.accuracy
@pytest.mark.mesh
@pytest.mark.gate
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_accuracy(
    mesh_device, seq_len, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo
):
    topology = _topology_from_device_params(device_params)
    run_sparse_mla_accuracy_case(variant, config_only, mesh_device, seq_len, topology, ds_layer, ds_checkpoint, ds_repo)


@pytest.mark.parametrize("mesh_device", SPARSE_MESH_PARAMS, ids=SPARSE_MESH_IDS, indirect=True)
@pytest.mark.parametrize("device_params", SPARSE_DEVICE_PARAMS, ids=SPARSE_DEVICE_IDS, indirect=True)
@pytest.mark.parametrize("seq_len", [4096], ids=["seq4k"])
@pytest.mark.parametrize("n_runs", [3], ids=["x3"])
@pytest.mark.parametrize("variant", SPARSE_VARIANTS, indirect=True, ids=SPARSE_VARIANTS)
@pytest.mark.determinism
@pytest.mark.dev
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_determinism(
    mesh_device, seq_len, n_runs, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo
):
    topology = _topology_from_device_params(device_params)
    run_sparse_mla_determinism_case(
        variant, config_only, mesh_device, seq_len, n_runs, topology, ds_layer, ds_checkpoint, ds_repo
    )


@pytest.mark.parametrize("mesh_device", SPARSE_MESH_PARAMS, ids=SPARSE_MESH_IDS, indirect=True)
@pytest.mark.parametrize("device_params", SPARSE_DEVICE_PARAMS, ids=SPARSE_DEVICE_IDS, indirect=True)
@pytest.mark.parametrize("seq_len,chunk", [(4096, 1024)], ids=["4k_c1k"])
@pytest.mark.parametrize("variant", SPARSE_VARIANTS, indirect=True, ids=SPARSE_VARIANTS)
@pytest.mark.feature_chunking
@pytest.mark.feature_cache
@pytest.mark.gate
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_chunked(
    mesh_device, seq_len, chunk, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo, ds_input
):
    run_sparse_mla_chunked_case(
        variant, config_only, mesh_device, seq_len, chunk, ds_layer, ds_checkpoint, ds_repo, ds_input
    )
