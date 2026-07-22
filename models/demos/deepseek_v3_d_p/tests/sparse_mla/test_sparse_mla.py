# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse MLA / DSA tests for DeepSeek V3.2-family variants.

Dense MLA coverage lives in test_mla.py. This file keeps the sparse reference
path separate while reusing the same TT execution helper and the production mesh
/ fabric axes from the dense MLA tests.
"""

import os

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import KVPE_MIN_TOKENS_PER_CHIP, detect_num_devices
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import (
    build_weights,
    cpu_ref_cache_dir,
    make_hidden,
    run_cpu_reference,
    run_cpu_reference_chunked,
)
from models.demos.deepseek_v3_d_p.tests.test_mla import run_mla_inference
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup, interleaved_to_halfsplit_perm
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.ttnn.utils_for_testing import assert_with_pcc

SPARSE_OUTPUT_PCC = 0.98
SPARSE_KVPE_PCC = 0.99
# Indexer key cache is stored bf8 on device vs the bf16 CPU reference, so it carries block-float
# quantization noise. Measured ~0.99991 on 2x4 BH (both variants, chunked + rotated), tracking the
# bf16 KVPE cache; 0.999 keeps ample bf8 headroom while still catching a real write regression.
SPARSE_INDEX_PCC = 0.999
SPARSE_VARIANTS = ["deepseek_v32", "glm_5_1", "glm_5_2"]

# ---------------------------------------------------------------------------
# TEST MATRIX — single source of truth (see _sparse_cases for how it expands)
# ---------------------------------------------------------------------------
# Box-adaptive candidate meshes (sp, tp), keyed by physical device count. Each box lists ONLY shapes
# that fit it, so off-box shapes are never generated (no "needs N devices" skips). Shapes must be
# TP>=2 (the dense 128-head epilogue overflows L1 at TP=1). BOTH variants run every listed mesh: GLM's
# thin per-chip head shard at tp=4 (64/4=16 < 32) is handled by the head→sequence reshard in
# ttMLA._sparse_mla (#48727) + the head-replicated seq-sharded indexer, so GLM is no longer TP-capped.
# Coverage rationale:
#   QuietBox (4):  (2,2) TP=2 and (1,4) TP=4 — both variants at both TP.
#   LoudBox  (8):  (2,4) TP=4 and (4,2) TP=2 — both variants at both TP.
#   Galaxy   (32): (8,4) production TP=4 + (8,2) TP=2 plane.
# Mesh shape is NOT correctness-invariant, so accuracy sweeps the whole box set; determinism and
# chunked pin to each variant's anchor (highest supported TP) — see _sparse_cases(anchor_only=True).
SPARSE_MESH_BY_DEVICES = {
    4: [(2, 2), (1, 4)],
    8: [(2, 4), (4, 2)],
    32: [(8, 4), (8, 2)],
}

# seq_len is a sparsity-regime axis (not a code path): accuracy keeps one inert-top-k point (256,
# where sparse == dense) and one real-pruning point (5120). determinism/chunked use the single
# prod-closest length (5120). 2048 dropped (also inert, redundant with 256).
SPARSE_SEQS_ACCURACY = [256, 5120]
SPARSE_SEQS_ANCHOR = [5120]


def _sparse_meshes():
    """Current box's candidate (sp, tp) meshes; best-effort single TP plane on non-standard boxes."""
    n = detect_num_devices()
    return SPARSE_MESH_BY_DEVICES.get(n, [(1, max(n, 1))])


def _seq_ok_for_mesh(seq_len, mesh):
    # kvpe ND-shard cache needs >= KVPE_MIN_TOKENS_PER_CHIP tokens per SP shard.
    return seq_len // mesh[0] >= KVPE_MIN_TOKENS_PER_CHIP


def _anchor_mesh(meshes):
    """Production-closest mesh: the highest-TP mesh among the box's meshes."""
    return max(meshes, key=lambda m: m[1])


def _sparse_cases(seqs, anchor_only):
    """Generate (variant, mesh, seq_len) params for the CURRENT box — only valid combos, so the
    collected matrix equals the run matrix (validity is enforced here, not via runtime skips)."""
    meshes = _sparse_meshes()
    chosen = [_anchor_mesh(meshes)] if (anchor_only and meshes) else meshes
    cases = []
    for mesh in chosen:
        for seq_len in seqs:
            if not _seq_ok_for_mesh(seq_len, mesh):
                continue
            for variant_name in SPARSE_VARIANTS:
                cases.append(
                    pytest.param(variant_name, mesh, seq_len, id=f"{variant_name}-{mesh[0]}x{mesh[1]}-seq{seq_len}")
                )
    return cases


SPARSE_ACCURACY_CASES = _sparse_cases(SPARSE_SEQS_ACCURACY, anchor_only=False)
SPARSE_ANCHOR_CASES = _sparse_cases(SPARSE_SEQS_ANCHOR, anchor_only=True)

# All three fabric transports, keyed by name. Fabric is NOT swept: correctness is ~invariant to the
# transport, so the suite pins one fabric (PREFERRED below) and lets a dedicated fabric test cover
# multi-transport bring-up. The ids/dicts are kept so DS_SPARSE_FABRIC can select any of them.
_SPARSE_FABRICS = {
    "line": {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
    },
    "ring": {
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
    },
    "fabric2d": {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D,
        "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
    },
}
# Auto-pin the fabric: priority fabric2d > ring > line (fabric2d is the Galaxy/production bring-up).
# Override with DS_SPARSE_FABRIC=line|ring|fabric2d. TODO: replace the priority default with a real
# per-box capability probe; for now it defaults to the top-priority (production) transport.
_SPARSE_FABRIC_PRIORITY = ["fabric2d", "ring", "line"]


def _preferred_fabric_name() -> str:
    env = os.environ.get("DS_SPARSE_FABRIC")
    if env:
        assert env in _SPARSE_FABRICS, f"DS_SPARSE_FABRIC={env!r} must be one of {sorted(_SPARSE_FABRICS)}"
        return env
    return _SPARSE_FABRIC_PRIORITY[0]


PREFERRED_FABRIC = _preferred_fabric_name()
SPARSE_DEVICE_PARAMS = [_SPARSE_FABRICS[PREFERRED_FABRIC]]
SPARSE_DEVICE_IDS = [PREFERRED_FABRIC]


def _topology_from_device_params(device_params):
    return (
        ttnn.Topology.Ring
        if device_params.get("fabric_config") == ttnn.FabricConfig.FABRIC_1D_RING
        else ttnn.Topology.Linear
    )


def _init_index_kv_cache(config, mesh_device, seq_len, mesh_shape, sp_axis, slot_num=1):
    """Block-cyclic indexer key cache, allocated OUTSIDE ttMLA (mirrors tt_kvpe_cache) and passed into
    ttMLA.forward(index_kv_cache=...) every call. BF8 (matches BF16 top-k within bf16 noise, half the memory).

    Layer-slot count mirrors the serving adapter (glm_5_2.py allocate_kv_cache): the indexer strides the
    folded user-major cache by num_full_indexer_layers (only ``full`` layers own an index slot), so the
    cache must carry that many layer slots for update_padded_kv_cache's cache_batch % num_layers check to
    hold. Falls back to 1 when the config has no ``indexer_types`` (deepseek_v32 / glm_5_1: every layer full,
    single-layer standalone MLA -> stride 1)."""
    return init_kvpe_cache(
        kvpe_cache_head_dim=getattr(config, "index_head_dim", 128),
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_full_indexer_layers(config) or 1,
        num_users=slot_num,
        dtype=ttnn.bfloat8_b,
    )


def _collect_index_cache_natural(tt_index_kv_cache, mesh_device, config, chunk):
    """Read the block-cyclic indexer key cache back to a natural-order [S, index_head_dim] tensor in the
    CPU reference's RoPE frame, so it can be PCC'd against SparseMLAReference.index_cache.

    Same SP-shard concat + block-cyclic un-rotation as the KVPE cache (blockcyclic_positions). The device
    stores the RoPE half INTERLEAVED for both variants (the indexed RoPE op is interleaved-only; the DS
    path permutes half-split->interleaved before it). The CPU reference stores it interleaved for GLM
    (index_rope_interleave=True) but HALF-SPLIT for DS, so for DS we reindex the device's RoPE dims back
    to half-split (interleaved_to_halfsplit_perm) before comparing; the non-RoPE dims match directly."""
    sp = mesh_device.shape[0]
    cache_sr = ttnn.to_torch(
        tt_index_kv_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)[:, :1]
    p = blockcyclic_positions(sp, chunk, cache_sr.shape[2])
    nat = torch.empty(cache_sr.shape[2], cache_sr.shape[-1], dtype=torch.bfloat16)
    nat[p] = cache_sr[0, 0]
    if not getattr(config, "index_rope_interleave", False):  # DS: device interleaved -> reference half-split
        rope_dim = config.qk_rope_head_dim
        perm = interleaved_to_halfsplit_perm(rope_dim)
        nat = nat.clone()
        nat[:, :rope_dim] = nat[:, :rope_dim][:, perm]
    return nat


def run_sparse_mla_accuracy_case(
    variant, config, mesh_device, seq_len, topology, ds_layer=None, ds_checkpoint=None, ds_repo=None
):
    """Sparse-MLA accuracy: device output + KVPE cache vs MLACPU sparse reference."""
    # Validity (tp<=cap, seq/sp>=min tokens, off-box shapes) is enforced by _sparse_cases at
    # collection time, so there are no runtime skips here: collected == run.
    logger.info(
        f"[{variant.name}] sparse MLA accuracy start: seq_len={seq_len} "
        f"mesh={tuple(mesh_device.shape)} topology={topology}"
    )
    logger.debug(f"[{variant.name}] sparse MLA accuracy: building TT/CPU weights")
    weights, src_tag = build_weights(variant, config, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo)
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
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
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
    # accuracy runs the indexer's natural single-shot path (no block-cyclic index_kv_cache to read
    # back), so the index-cache reference is unused here — chunked/rotated cover the block-cyclic cache.
    ref_output, ref_kvpe, _ = run_cpu_reference(
        config, weights, hidden_states, seq_len, cache_dir, cache_tag=f"{src_tag}_funcidx"
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
    # Mesh is the per-variant anchor and seq fits SP — guaranteed by _sparse_cases (no runtime skips).
    logger.info(
        f"[{variant.name}] sparse MLA determinism start: seq_len={seq_len} "
        f"mesh={tuple(mesh_device.shape)} topology={topology} n_runs={n_runs}"
    )
    logger.debug(f"[{variant.name}] sparse MLA determinism: building TT weights")
    weights, _ = build_weights(variant, config, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo)
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
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
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
    # Anchor mesh (TP>=2) and seq/SP validity are guaranteed by _sparse_cases (no runtime skips).
    #
    # CACHE-QUANTIZATION NOTE: the CPU reference keeps KVPE in bf16 (SparseMLAReference defaults to
    # simulate_fp8=False — see reference.cpu_deepseek_v32), matching the bf16 single-shot device
    # cache. But the chunked DEVICE path reads the prefix back from the bf8 KVPE cache and upcasts it in
    # _gather_kvpe_prefix, so this comparison crosses a bf8 cache round-trip the reference does not model.
    # That quantization noise eats into the SPARSE_OUTPUT_PCC headroom here (vs single-shot). If chunked
    # PCC ever drifts toward the threshold, add a simulate_fp8=True reference variant to separate expected
    # cache-quantization noise from a true logic regression.
    seed = 42
    logger.info(
        f"[{variant.name}] sparse MLA chunked start: seq_len={seq_len} chunk={chunk} "
        f"mesh={tuple(mesh_device.shape)} ds_input={bool(ds_input)}"
    )
    logger.debug(f"[{variant.name}] sparse MLA chunked: building TT/CPU weights")
    weights, src_tag = build_weights(
        variant, config, seed=seed, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo
    )
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
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_index_kv_cache = _init_index_kv_cache(config, mesh_device, seq_len, mesh_shape, sp_axis)
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
        out = mla_tt.forward(tt_x, rope_tensors, tt_kvpe_cache, actual_start=s, index_kv_cache=tt_index_kv_cache)
        outs.append(
            ttnn.to_torch(
                out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
            ).to(torch.bfloat16)
        )
    tt_output = torch.cat(outs, dim=2)

    cache_dir = cpu_ref_cache_dir(variant)
    logger.info(f"[{variant.name}] sparse MLA chunked: running CPU reference")
    ref_output, ref_kvpe, ref_index = run_cpu_reference_chunked(
        config, weights, hidden, seq_len, chunk, cache_dir, cache_tag=f"{src_tag}_funcidx"
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

    logger.debug(f"[{variant.name}] sparse MLA chunked: collecting indexer key cache")
    idx_nat = _collect_index_cache_natural(tt_index_kv_cache, mesh_device, config, chunk)
    _, idx_pcc = assert_with_pcc(ref_index[0, :seq_len], idx_nat[:seq_len], SPARSE_INDEX_PCC)
    logger.info(f"[{variant.name}] Chunked indexer cache PCC: {idx_pcc}")

    _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output, SPARSE_OUTPUT_PCC)
    logger.info(f"[{variant.name}] Chunked output PCC: {pcc_message}")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[{variant.name}] sparse MLA chunked complete")


def run_sparse_mla_rotated_case(
    variant, config, mesh_device, iters_isl, chunk_size_global, ds_layer, ds_checkpoint, ds_repo
):
    """Rotation/padding scenario (sparse analogue of test_mla._run_chunked_prefill): one DENSE sequence
    chunked VARIABLY by iters_isl (per-iter valid token counts). Each iter's real tokens are rotated to
    start at the previous iter's real end (actual_start=kv_actual, tile-aligned); the fixed-width chunk's
    non-valid rows are zeroed. This exercises: partial chunks, rotated (mid-slab) chunk_start, and — the
    thing under test — that a later iter never attends the pad a partial earlier iter left in the cache
    (the next iter's write overwrites it before scoring; causal masks the tail). Ground truth is the
    single-shot sparse reference over the whole dense sequence; we compare each iter's valid region."""
    seed = 42
    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tile = ttnn.TILE_SIZE
    chunk_local = chunk_size_global // sp
    for v in iters_isl:
        assert 0 < v <= chunk_size_global and v % tile == 0, f"iter isl {v}: tile-aligned, <= {chunk_size_global}"
    total_len = sum(iters_isl)

    weights, src_tag = build_weights(
        variant, config, seed=seed, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo
    )
    max_window = max(sum(iters_isl[:i]) + chunk_size_global for i in range(len(iters_isl)))
    seq_len_cache = ((max_window + chunk_size_global - 1) // chunk_size_global) * chunk_size_global
    config.max_seq_len = seq_len_cache
    logger.info(
        f"[{variant.name}] rotated: iters={iters_isl} total={total_len} chunk={chunk_size_global} cache={seq_len_cache} mesh={tuple(mesh_device.shape)}"
    )

    hidden = make_hidden(total_len, config.hidden_size, seed)[0]  # [total_len, H]
    cache_dir = cpu_ref_cache_dir(variant)
    ref_output, ref_kvpe, ref_index = run_cpu_reference(
        config, weights, hidden.unsqueeze(0), total_len, cache_dir, cache_tag=f"{src_tag}_rot{total_len}_funcidx"
    )
    ref_output = ref_output[0]  # [total_len, out_dim]
    out_dim = ref_output.shape[-1]
    ref_kvpe = ref_kvpe.reshape(-1, ref_kvpe.shape[-1])  # [total_len, kvpe_dim]
    ref_index = ref_index[0]  # [total_len, index_head_dim]

    tt_index_kv_cache = _init_index_kv_cache(config, mesh_device, seq_len_cache, mesh_shape, sp_axis, slot_num=1)
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len_cache,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        slot_num=1,
        layer_num=1,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(
        cache_seq_len_global=seq_len_cache, chunk_size_global=chunk_size_global
    )
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2

    out_accum = torch.zeros(total_len, out_dim, dtype=torch.bfloat16)
    for i, isl in enumerate(iters_isl):
        kv_actual = sum(iters_isl[:i])
        valid_end = kv_actual + isl
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = [positions[c][r] for c in range(sp) for r in range(chunk_local)]
        gather_idx = torch.tensor([min(gp, total_len - 1) for gp in flat], dtype=torch.long)
        chunk_in = hidden[gather_idx].clone()
        chunk_in[torch.tensor([gp >= valid_end for gp in flat])] = 0.0
        tt_h = ttnn.from_torch(
            chunk_in.reshape(1, 1, chunk_size_global, config.hidden_size),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )
        tt_out = mla_tt.forward(
            tt_h, rope_tensors, tt_kvpe_cache, actual_start=kv_actual, cache_user_id=0, index_kv_cache=tt_index_kv_cache
        )
        out_flat = ttnn.to_torch(
            tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
        ).to(torch.bfloat16)[0, 0]
        valid_pairs = [(row, gp) for row, gp in enumerate(flat) if gp < valid_end]
        out_accum[torch.tensor([gp for _, gp in valid_pairs])] = out_flat[torch.tensor([row for row, _ in valid_pairs])]
        rot = "rotated" if kv_actual % chunk_size_global != 0 else "aligned"
        _, m = comp_pcc(ref_output[kv_actual:valid_end], out_accum[kv_actual:valid_end], 0)
        logger.info(f"[{variant.name}] iter {i} (kv_actual={kv_actual} isl={isl} {rot}): valid-region PCC {m}")

    # ISOLATION: is the block-cyclic WRITE correct under rotation? Un-rotate the final KVPE cache and
    # compare per-iter region + full to the reference. If cache matches but output above diverged, the
    # bug is in SCORING (indexer top-k / sparse_sdpa), not the write.
    cache_sr = ttnn.to_torch(
        tt_kvpe_cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)[:, :1]
    p = blockcyclic_positions(sp, chunk_size_global, cache_sr.shape[2])
    nat = torch.empty(cache_sr.shape[2], cache_sr.shape[-1], dtype=torch.bfloat16)
    nat[p] = cache_sr[0, 0]
    for i, isl in enumerate(iters_isl):
        s = sum(iters_isl[:i])
        _, m = comp_pcc(ref_kvpe[s : s + isl], nat[s : s + isl], 0)
        logger.info(f"[{variant.name}] KVPE cache region iter {i} [{s}:{s + isl}]: {m}")
    _, mk = comp_pcc(ref_kvpe[:total_len], nat[:total_len], 0)
    logger.info(f"[{variant.name}] KVPE cache full PCC: {mk}")

    # Same isolation for the block-cyclic INDEXER key cache — the untested-on-main tensor. Per-iter region
    # PCCs are logged for diagnosis (they localize a rotated-write bug to the offending iter); the full-cache
    # PCC is the gate, so a silent indexer-cache write regression fails here, not just via the output.
    idx_nat = _collect_index_cache_natural(tt_index_kv_cache, mesh_device, config, chunk_size_global)
    for i, isl in enumerate(iters_isl):
        s = sum(iters_isl[:i])
        _, m = comp_pcc(ref_index[s : s + isl], idx_nat[s : s + isl], 0)
        logger.info(f"[{variant.name}] indexer cache region iter {i} [{s}:{s + isl}]: {m}")
    _, imsg = assert_with_pcc(ref_index[:total_len], idx_nat[:total_len], SPARSE_INDEX_PCC)
    logger.info(f"[{variant.name}] indexer cache full PCC: {imsg}")

    _, msg = assert_with_pcc(
        ref_output.reshape(1, 1, total_len, out_dim), out_accum.reshape(1, 1, total_len, out_dim), SPARSE_OUTPUT_PCC
    )
    logger.info(f"[{variant.name}] rotated full PCC: {msg}")
    ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize("variant, mesh_device, seq_len", SPARSE_ANCHOR_CASES, indirect=["variant", "mesh_device"])
@pytest.mark.parametrize("device_params", SPARSE_DEVICE_PARAMS, ids=SPARSE_DEVICE_IDS, indirect=True)
@pytest.mark.parametrize("iters_isl", [[2560, 2592, 5120]], ids=["maxedge"])
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_rotated(
    mesh_device, seq_len, iters_isl, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo
):
    run_sparse_mla_rotated_case(variant, config_only, mesh_device, iters_isl, seq_len, ds_layer, ds_checkpoint, ds_repo)


# One combined parametrization (variant, mesh_device, seq_len) instead of three independent axes: the
# cases are generated by _sparse_cases for the current box, so the collected matrix IS the run matrix.
@pytest.mark.parametrize("variant, mesh_device, seq_len", SPARSE_ACCURACY_CASES, indirect=["variant", "mesh_device"])
@pytest.mark.parametrize("device_params", SPARSE_DEVICE_PARAMS, ids=SPARSE_DEVICE_IDS, indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_accuracy(
    mesh_device, seq_len, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo
):
    topology = _topology_from_device_params(device_params)
    run_sparse_mla_accuracy_case(variant, config_only, mesh_device, seq_len, topology, ds_layer, ds_checkpoint, ds_repo)


# GLM-5.2 indexer reuse: anchor cases for the reuse-capable variant only (others have no shared layers).
SPARSE_REUSE_CASES = [c for c in SPARSE_ANCHOR_CASES if "glm_5_2" in c.id]


@pytest.mark.parametrize("variant, mesh_device, seq_len", SPARSE_REUSE_CASES, indirect=["variant", "mesh_device"])
@pytest.mark.parametrize("device_params", SPARSE_DEVICE_PARAMS, ids=SPARSE_DEVICE_IDS, indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_indexer_reuse(
    mesh_device, seq_len, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo
):
    """GLM-5.2 indexer reuse: a layer fed a prior layer's top-k indices (indexer_indices=...) must
    produce the SAME output as computing them itself — validates the MLA return + accept path. Same
    weights + input + selection -> identical sparse attention, so the two outputs match bit-for-bit."""
    config = config_only
    config.max_seq_len = seq_len
    weights, _ = build_weights(variant, config, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo)
    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    topology = _topology_from_device_params(device_params)

    def _kvpe():
        # sparse_sdpa reads the KVPE cache natively and requires it uncompressed (bf16 ROW_MAJOR), not
        # the bfloat8_b/TILE default — mla.py asserts this. Mirror the accuracy case's cache.
        return init_kvpe_cache(
            kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=1,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    common = dict(
        config=config,
        weights=weights,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        topology=topology,
    )
    # A: compute the indexer, capture its top-k selection + output.
    out_a, _, _, shard_dims, idx = run_mla_inference(tt_kvpe_cache=_kvpe(), return_indices=True, **common)
    # B: a fresh MLA (same weights + input) fed A's indices -> skips its own indexer.
    out_b, _, _, _ = run_mla_inference(tt_kvpe_cache=_kvpe(), inject_indices=idx, **common)

    def _to_torch(t):
        return ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
        ).to(torch.bfloat16)

    _, pcc_message = assert_with_pcc(_to_torch(out_a), _to_torch(out_b), 0.9999)
    logger.info(f"[{variant.name}] indexer-reuse compute-vs-inject PCC: {pcc_message}")
    ttnn.synchronize_device(mesh_device)


# Anchor cases (per-variant prod-closest mesh, seq=4096); collected == run.
@pytest.mark.parametrize("variant, mesh_device, seq_len", SPARSE_ANCHOR_CASES, indirect=["variant", "mesh_device"])
@pytest.mark.parametrize("device_params", SPARSE_DEVICE_PARAMS, ids=SPARSE_DEVICE_IDS, indirect=True)
@pytest.mark.parametrize("n_runs", [3], ids=["x3"])
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_determinism(
    mesh_device, seq_len, n_runs, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo
):
    topology = _topology_from_device_params(device_params)
    run_sparse_mla_determinism_case(
        variant, config_only, mesh_device, seq_len, n_runs, topology, ds_layer, ds_checkpoint, ds_repo
    )


# Anchor cases (seq=5120) crossed with the single prefill chunk size; collected == run.
@pytest.mark.parametrize("variant, mesh_device, seq_len", SPARSE_ANCHOR_CASES, indirect=["variant", "mesh_device"])
@pytest.mark.parametrize("device_params", SPARSE_DEVICE_PARAMS, ids=SPARSE_DEVICE_IDS, indirect=True)
@pytest.mark.parametrize("chunk", [1024], ids=["c1k"])
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_chunked(
    mesh_device, seq_len, chunk, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo, ds_input
):
    run_sparse_mla_chunked_case(
        variant, config_only, mesh_device, seq_len, chunk, ds_layer, ds_checkpoint, ds_repo, ds_input
    )
