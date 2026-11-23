# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tier-1 GLM-5.4 weights on device: the reduced stack (tests/glm54_weights.py) builds as a TtPrefillTransformer
and its prelude and loop pass 1 (layers 0-2, run once) match the GLM-5.4 CPU reference stack on the same weights.

A load check, not the loop's acceptance: the loop, gates and pass 2 on device are q171.2.4 / q171.2.5. The bar
only has to catch weights that did not load as the reference reads them (wrong layer, role or order; see LOAD_PCC).

The CPU reference at the 8x4 chunk takes minutes, so ``test_glm54_tier1_reference_prepare`` computes it without a
device (run it with plain pytest first) and the device test reads it back; the device test computes it itself only
when the file is missing. Nothing at import or collection opens the device.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.chunk_layout import pack_forward_tokens, rotated_chunk_positions
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import glm54_stack_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_4.synthetic_gates import GLM54_SYNTHETIC_SEED, SYNTHETIC_GATES_VERSION
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.glm54_weights import (
    GLM54ReducedConfig,
    glm54_reduced_config,
    glm54_tier1_weights,
)
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import mesh_shard_dims
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache

# The check reads the first LOAD_ROWS positions: with synthetic weights attention is near uniform, so its output
# shrinks as a row averages more keys and bf16 noise takes over (device vs reference falls smoothly with position:
# layer 0 0.9986 at rows 0-63, 0.972 at rows >= 2048; layer 2 0.87 over the chunk), which says nothing about loading.
# Measured on rows < 512 (galaxy 8x4, 2026-11-23): correct load layer 0 0.996 / layer 2 0.953; faults on the CPU
# reference: loop layers 1/2 norms + router swapped -> layer 2 0.756, layer 0 with another layer's MLA -> 0.006.
LOAD_ROWS = 512
LOAD_PCC = 0.90
# Device layer -> reference checkpoint: the prelude output (loop pass-1 input) and the loop pass-1 output. The device
# runs layers 0-3 once; its layer 3 is not GLM-5.4's (that follows the gates and pass 2).
COMPARED = {0: "pass1.in", 2: "pass1.out"}
SP = 8  # the 8x4 mesh's SP factor (axis 0)
CHUNK = PREFILL_CHUNK_TOKENS_PER_CHIP * SP
TOKEN_SEED = 5
# Bump when the stored bytes change for the same key below (reference stack, layer generator, compared checkpoints).
REFERENCE_VERSION = 1
REFERENCE_CACHE = Path.home() / ".cache" / "tt-glm54"


def _tokens(config):
    return torch.randint(0, config.vocab_size, (CHUNK,), generator=torch.Generator().manual_seed(TOKEN_SEED))


def _reference_path() -> Path:
    key = f"seed{GLM54_SYNTHETIC_SEED}-gates{SYNTHETIC_GATES_VERSION}-chunk{CHUNK}-tokens{TOKEN_SEED}-v{REFERENCE_VERSION}"
    return REFERENCE_CACHE / f"tier1-reference-{key}.pt"


def _compute_reference(config, weights) -> dict:
    """The compared GLM-5.4 reference checkpoints for one chunk of ``_tokens``, saved for the device test."""
    x = weights.state_dict["embed_weight"][_tokens(config)].unsqueeze(0)
    cp = glm54_stack_reference(config, list(weights.state_dict["layers"]), weights.gates, x).checkpoints
    ref = {name: cp[name] for name in COMPARED.values()}
    path = _reference_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)
    logger.info(f"tier-1 reference saved to {path}")
    return ref


@pytest.mark.timeout(0)
def test_glm54_tier1_reference_prepare():
    """CPU only: compute the device test's reference (minutes) outside the device lock."""
    config = glm54_reduced_config(CHUNK)
    ref = _compute_reference(config, glm54_tier1_weights(config))
    assert all(torch.isfinite(t).all() for t in ref.values())


def _pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.detach().flatten().double(), b.detach().flatten().double()]))[0, 1])


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(GLM54ReducedConfig), l1_small_size=GLM53Adapter.l1_small_size
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_glm54_tier1_loads_on_device(mesh_device, device_params, num_links):
    if not is_blackhole():
        pytest.skip("GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
    sp_axis, tp_axis = 0, 1
    sp = mesh_device.shape[sp_axis]
    assert sp == SP
    chunk_local = PREFILL_CHUNK_TOKENS_PER_CHIP
    chunk = CHUNK
    config = glm54_reduced_config(chunk)
    weights = glm54_tier1_weights(config)
    logger.info(f"tier-1 provenance: {weights.provenance}")
    n = config.num_hidden_layers

    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=GLM54ReducedConfig,
        state_dict=weights.state_dict,
        num_layers=n,
        seq_len=chunk,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=per_axis_topology(device_params["fabric_config"]),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        is_chunked=True,
        slot_num=1,
        max_seq_len=chunk,
        routing_use_l1_small_for_semaphores=GLM53Adapter.routing_use_l1_small_for_semaphores,
        tp_shard_kv=True,
        sparse_kv_cache_format=MlaKvCacheFormat.BF16_RM,
    )
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=chunk,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kvpe_cache_layers=n,
        num_users=1,
    )
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=chunk,
        mesh_shape=list(mesh_device.shape),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_kvpe_cache_layers=full_indexer_rank(config, n),
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )

    tokens = _tokens(config)
    packed = pack_forward_tokens([tokens.tolist()], [0], sp)
    tt_tokens = ttnn.from_torch(
        torch.tensor(packed, dtype=torch.int64).reshape(sp, 1, chunk_local),
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=mesh_shard_dims(0, sp_axis)
        ),
    )
    logger.info("tier-1 device forward")
    out = transformer.forward(
        tt_tokens,
        kvpe_cache,
        actual_isl=chunk,
        actual_start=0,
        actual_end=chunk,
        cache_user_id=0,
        return_intermediates=True,
        index_kv_cache=index_kv_cache,
    )
    ttnn.synchronize_device(mesh_device)
    # Device rows are chip-major in the rotated chunk layout; put them in position order.
    positions = rotated_chunk_positions(0, sp, chunk_local)
    order = torch.tensor([positions[c][r] for c in range(sp) for r in range(chunk_local)])
    device = {}
    for i in range(n):
        h = out[f"layer_{i}"].to(torch.float32)[0]
        nat = torch.empty_like(h)
        nat[order] = h
        device[i] = nat
        assert torch.isfinite(nat).all(), f"layer {i} output not finite"

    path = _reference_path()
    if path.is_file():
        logger.info(f"tier-1 reference from {path}")
        ref = torch.load(path)
    else:
        logger.warning(f"{path} missing: computing the CPU reference inside the device lock (minutes)")
        ref = _compute_reference(config, weights)
    # Kept for diagnosis (exact-input substitution, per-position error) without holding the device.
    torch.save(device, path.with_name(path.stem + "-device.pt"))
    failures = []
    k = config.index_topk
    for i, name in COMPARED.items():
        r = ref[name][0].float()
        p = _pcc(device[i][:LOAD_ROWS], r[:LOAD_ROWS])
        # Information only: the whole chunk, split where the indexer starts selecting (rows >= index_topk).
        full, head, tail = _pcc(device[i], r), _pcc(device[i][:k], r[:k]), _pcc(device[i][k:], r[k:])
        logger.info(
            f"tier-1 layer {i} device vs reference {name} PCC rows < {LOAD_ROWS}: {p:.5f}; chunk {full:.5f} "
            f"(rows < {k}: {head:.5f}, >= {k}: {tail:.5f})"
        )
        if p < LOAD_PCC:
            failures.append(f"layer {i} PCC rows < {LOAD_ROWS} {p:.5f} < {LOAD_PCC}")
    assert not failures, failures
