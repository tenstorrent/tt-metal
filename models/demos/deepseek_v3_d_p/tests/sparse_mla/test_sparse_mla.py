# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse MLA / DSA tests for the GLM-5.2 variant.

GLM-5.2, not GLM-5.3: these tests compare against a CPU reference cached under ``variant.mla_ref_cache_env``,
and only GLM-5.2 has a populated one. The two checkpoints are architecturally identical and these runs use
random weights, so GLM-5.2 covers the same sparse path.

Dense MLA coverage lives in test_mla.py. This file keeps the sparse reference
path separate while reusing the same TT execution helper and the production mesh
/ fabric axes from the dense MLA tests.
"""

import sys

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import KVPE_MIN_TOKENS_PER_CHIP, detect_num_devices
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import (
    build_weights,
    cpu_ref_cache_dir,
    make_hidden,
    run_cpu_reference,
    run_cpu_reference_chunked,
    run_cpu_reference_ragged,
)
from models.demos.deepseek_v3_d_p.tests.test_mla import run_mla_inference
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import normalized_hadamard_matrix, num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup, interleaved_to_halfsplit_perm
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from tests.ttnn.utils_for_testing import assert_with_pcc

SPARSE_OUTPUT_PCC = 0.98
SPARSE_KVPE_PCC = 0.99
# Indexer key cache is stored bf8 on device vs the bf16 CPU reference, so it carries block-float
# quantization noise. Measured ~0.99991 on 2x4 BH (chunked + rotated), tracking the
# bf16 KVPE cache; 0.999 keeps ample bf8 headroom while still catching a real write regression.
SPARSE_INDEX_PCC = 0.999
# Device top-k vs reference top-k, per row, over the rows that actually prune (more causal keys than
# index_topk). Tie-tolerant: a device-selected key also counts when its REFERENCE score is within
# SPARSE_TOPK_TIE_BAND (relative to that row's causal score range) of the reference's k-th score -- the
# device scores in bf8 (q, key cache) against a bf16 reference, so keys at the boundary can legitimately
# swap. Output PCC alone can hide a selection bug; this cannot.
SPARSE_TOPK_RECALL = 0.99
SPARSE_TOPK_TIE_BAND = 0.01
# glm_5_2: the production-closest GLM (populated reference cache; architecturally == GLM-5.3).
# kimi_k3_1: K3's gated NoPE MLA (96 heads) + a NoPE lightning indexer; its main KVPE cache is scaled FP8.
# deepseek_v32: 128 heads, so the only variant whose TP=2 and TP=4 head shards both suit sparse_sdpa unresharded.
SPARSE_VARIANTS = ["glm_5_2", "kimi_k3_1", "deepseek_v32"]
# Per-variant main KVPE cache format. Accuracy runs the main format at every sequence and the other format
# at the real-pruning sequence only, as a diagnostic that separates quantization error from logic error.
SPARSE_MAIN_KV_FORMAT = {
    "glm_5_2": MlaKvCacheFormat.BF16_RM,
    "kimi_k3_1": MlaKvCacheFormat.SCALED_FP8,
    "deepseek_v32": MlaKvCacheFormat.BF16_RM,
}
# Inert-top-k fallback where seq 256 is too short: the shortest of these whose stripe is long enough on that mesh
# (LoudBox / Galaxy 8x2: 1024; Galaxy 8x4: 2048). All are <= index_topk, so sparse == dense there. K3.1's and
# DeepSeek-V3.2's index caches have ONE layer slot (no compacted multi-layer slab like GLM's), so the SP x TP-striped
# per-chip stripe must itself hold >= KVPE_MIN_TOKENS_PER_CHIP rows; at 256 tokens on 8 chips (32 rows) the
# ND-sharded cache collapses to WIDTH_SHARDED and indexer_score rejects it.
SPARSE_INERT_FALLBACK_SEQS = (1024, 2048)
SINGLE_SLOT_INDEX_VARIANTS = ("kimi_k3_1", "deepseek_v32")


def _single_slot_seq_ok(seq_len, mesh):
    return _seq_ok_for_mesh(seq_len, mesh) and seq_len // (mesh[0] * mesh[1]) >= KVPE_MIN_TOKENS_PER_CHIP


def _collect_kvpe_cache(cache, mesh_device):
    sp, tp = mesh_device.shape[0], mesh_device.shape[1]
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    logical = cache.unpack_host(ttnn.to_torch(cache.storage, mesh_composer=composer))
    local = logical.shape[2] // sp  # per-chip rows = seq_len_cache / (sp*tp)
    flat = torch.cat(
        [logical[:, t, s * local : (s + 1) * local] for s in range(sp) for t in range(tp)], dim=1
    )  # [num_slots, sp*tp*local == seq_len_cache, kvpe]
    return flat.unsqueeze(1)


# ---------------------------------------------------------------------------
# TEST MATRIX — single source of truth (see _sparse_cases for how it expands)
# ---------------------------------------------------------------------------
# Box-adaptive candidate meshes (sp, tp), keyed by physical device count. Each box lists ONLY shapes
# that fit it, so off-box shapes are never generated (no "needs N devices" skips). Shapes must be
# TP>=2 (the dense 128-head epilogue overflows L1 at TP=1). Every variant runs every listed mesh: sparse
# attention reshards heads -> sequence at every TP > 1 (ttMLA.reshards_heads_to_seq), fed by the
# head-replicated seq-sharded indexer.
# Coverage rationale:
#   QuietBox (4):  (1,4) TorusX and (2,2) Fabric2D.
#   LoudBox  (8):  (2,4) TP=4 and (4,2) TP=2 — every variant at both TP.
#   Galaxy   (32): (8,4) production TP=4 + the existing (8,2) TP=2 diagnostic plane.
# Mesh shape is NOT correctness-invariant, so accuracy sweeps the whole box set; determinism and
# chunked pin to each variant's anchor (highest supported TP) — see _sparse_cases(anchor_only=True).
SPARSE_MESH_BY_DEVICES = {
    4: [(2, 2), (1, 4)],
    8: [(2, 4), (4, 2)],
    # Only the complete 8x4 view uses TorusXY. The existing 8x2 plane remains plain Fabric2D.
    32: [(8, 4), (8, 2)],
}

# seq_len is a sparsity-regime axis (not a code path): accuracy keeps one inert-top-k point (256,
# where sparse == dense) and one real-pruning point (5120). determinism/chunked use the single
# prod-closest length (5120). 2048 dropped (also inert, redundant with 256).
SPARSE_SEQS_ACCURACY = [256, 5120]
SPARSE_SEQS_ANCHOR = [5120]


def _sparse_meshes():
    """Current box's supported (sp, tp) meshes, or None for an unsupported device count."""
    return SPARSE_MESH_BY_DEVICES.get(detect_num_devices())


def _worker_l1_size():
    return ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE


def _device_params_for_mesh(mesh):
    """Assign one fabric to each existing mesh; topology is not a separate test axis."""
    if mesh == (1, 4):
        return torus_x_device_params(worker_l1_size=_worker_l1_size())
    if mesh == (8, 4):
        return torus_xy_device_params(worker_l1_size=_worker_l1_size())
    return fabric2d_device_params(worker_l1_size=_worker_l1_size())


def _seq_ok_for_mesh(seq_len, mesh):
    # kvpe ND-shard cache needs >= KVPE_MIN_TOKENS_PER_CHIP tokens per SP shard.
    return seq_len // mesh[0] >= KVPE_MIN_TOKENS_PER_CHIP


def _anchor_mesh(meshes):
    """Production-closest mesh: the highest-TP mesh among the box's meshes."""
    return max(meshes, key=lambda m: m[1])


def _sparse_cases(seqs, anchor_only, variants=None):
    """Generate (variant, mesh, seq_len, device_params) for the current box — only valid combos, so the
    collected matrix equals the run matrix (validity is enforced here, not via runtime skips)."""
    variants = SPARSE_VARIANTS if variants is None else variants
    meshes = _sparse_meshes()
    if meshes is None:
        num_devices = detect_num_devices()
        return [
            pytest.param(
                SPARSE_VARIANTS[0],
                (2, 2),
                seqs[-1],
                fabric2d_device_params(worker_l1_size=_worker_l1_size()),
                marks=pytest.mark.skip(reason=f"unsupported device count {num_devices}"),
                id="unsupported-fabric2d",
            )
        ]
    chosen = [_anchor_mesh(meshes)] if (anchor_only and meshes) else meshes
    cases = []
    for mesh in chosen:
        for seq_len in seqs:
            if not _seq_ok_for_mesh(seq_len, mesh):
                continue
            for variant_name in variants:
                topology_mark = (
                    []
                    if mesh == (8, 2)
                    else pytest.mark.requires_mesh_topology(
                        mesh_shape=mesh, topology="ring" if mesh == (1, 4) else f"mesh-{mesh[0]}x{mesh[1]}"
                    )
                )
                cases.append(
                    pytest.param(
                        variant_name,
                        mesh,
                        seq_len,
                        _device_params_for_mesh(mesh),
                        marks=topology_mark,
                        id=f"{variant_name}-shape-{mesh[0]}x{mesh[1]}-seq{seq_len}",
                    )
                )
    return cases


def _single_slot_seq_for_mesh(candidates, mesh):
    """The first candidate sequence that is valid for a single-slot index cache on this mesh, or None."""
    return next((seq for seq in candidates if _single_slot_seq_ok(seq, mesh)), None)


def _inert_fallback_cases():
    """Single-slot index-cache variants get the shortest valid inert-top-k point on meshes where seq 256 is invalid."""
    cases = []
    for mesh in _sparse_meshes() or []:
        if _single_slot_seq_ok(SPARSE_SEQS_ACCURACY[0], mesh):
            continue
        seq = _single_slot_seq_for_mesh(SPARSE_INERT_FALLBACK_SEQS, mesh)
        assert seq is not None, f"no single-slot inert-top-k sequence fits mesh {mesh}"
        variants = list(SINGLE_SLOT_INDEX_VARIANTS)
        cases += [c for c in _sparse_cases([seq], anchor_only=False, variants=variants) if c.values[1] == mesh]
    return cases


def _sparse_accuracy_cases():
    """Each variant's main KV format covers both sparsity regimes; the other format runs only in the
    real-pruning regime (GLM / DeepSeek-V3.2: BF16 main + FP8 at 5120; K3.1: scaled FP8 main + BF16 diagnostic at
    5120). Single-slot index-cache variants take the inert fallback sequence where 256 is too short."""
    cases = []
    for case in _sparse_cases(SPARSE_SEQS_ACCURACY, anchor_only=False) + _inert_fallback_cases():
        variant, mesh, seq_len, device_params = case.values
        if variant in SINGLE_SLOT_INDEX_VARIANTS and not _single_slot_seq_ok(seq_len, mesh):
            continue  # replaced by the SPARSE_INERT_FALLBACK_SEQS case
        main = SPARSE_MAIN_KV_FORMAT[variant]
        other = MlaKvCacheFormat.SCALED_FP8 if main == MlaKvCacheFormat.BF16_RM else MlaKvCacheFormat.BF16_RM
        formats = [main, other] if seq_len == SPARSE_SEQS_ANCHOR[0] else [main]
        for cache_format in formats:
            cases.append(
                pytest.param(
                    variant,
                    mesh,
                    seq_len,
                    _device_params_for_mesh(mesh),
                    cache_format,
                    marks=case.marks,
                    id=f"{case.id}-kv_{cache_format.value}",
                )
            )
    return cases


SPARSE_ACCURACY_CASES = _sparse_accuracy_cases()
SPARSE_ANCHOR_CASES = _sparse_cases(SPARSE_SEQS_ANCHOR, anchor_only=True)


def _only(cases, *variants):
    return [c for c in cases if c.values[0] in variants]


# Multi-chunk coverage on every mesh of the box: each variant's TP=2 early-Q path (all heads per chip after the
# head->sequence reshard) differs from its TP=4 one.
SPARSE_CHUNKED_CASES = _sparse_cases(SPARSE_SEQS_ANCHOR, anchor_only=False)


def _init_index_kv_cache(config, mesh_device, seq_len, mesh_shape, sp_axis, slot_num=1, tp_axis=1):
    """Block-cyclic indexer key cache, allocated OUTSIDE ttMLA (mirrors tt_kvpe_cache) and passed into
    ttMLA.forward(index_kv_cache=...) every call. BF8 (matches BF16 top-k within bf16 noise, half the memory).

    Layer-slot count mirrors the serving adapter (glm_5_3.py allocate_kv_cache): the indexer strides the
    folded user-major cache by num_full_indexer_layers (only ``full`` layers own an index slot), so the
    cache must carry that many layer slots for update_padded_kv_cache's cache_batch % num_layers check to
    hold. Falls back to 1 when the config has no ``indexer_types`` (every layer full,
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
        tp_axis=tp_axis,  # KV dedup: stripe over TP as well as SP -- the only sparse layout
    )


def _cache_natural(tt_cache, mesh_device, chunk_size_global, batch_slot=0):
    """Read a block-cyclic KV/index cache back to a natural-order [S, D] tensor.

    ConcatMesh2dToTensor(dims=(2, 1)) puts the SP concat on the seq dim and the TP concat on dim 1, so
    dim 1 is indexed by tp-coord. The sparse path always dedups, so every tp-coord holds a DIFFERENT
    1/tp slice rather than a replica: flatten them into LINEAR CHIP ORDER (L = sp_coord*tp + tp_coord --
    the order update_padded_kv_cache(tp_axis=) writes and the full-mesh gather reconstructs), giving a
    shard-major buffer over sp*tp stripes. blockcyclic_positions is generic in the stripe count, so the
    un-rotation is the same call with sp*tp.
    """
    sp, tp = mesh_device.shape[0], mesh_device.shape[1]
    cache_sr = ttnn.to_torch(
        tt_cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)
    local = cache_sr.shape[2] // sp  # per-chip rows = seq_len_cache / (sp*tp)
    flat = torch.cat(
        [cache_sr[batch_slot, t, s * local : (s + 1) * local] for s in range(sp) for t in range(tp)], dim=0
    )
    p = blockcyclic_positions(sp * tp, chunk_size_global, flat.shape[0])
    nat = torch.empty(flat.shape[0], flat.shape[-1], dtype=torch.bfloat16)
    nat[p] = flat
    return nat


def _collect_index_cache_natural(tt_index_kv_cache, mesh_device, config, chunk, cache_batch_idx=0):
    """Read the block-cyclic indexer key cache back to a natural-order [S, index_head_dim] tensor in the
    CPU reference's RoPE frame, so it can be PCC'd against SparseMLAReference.index_cache.

    Same SP-shard concat + block-cyclic un-rotation as the KVPE cache (blockcyclic_positions). Undo the
    indexer's orthonormal Hadamard transform before comparing with the untransformed CPU reference. The device
    stores the RoPE half INTERLEAVED for both variants (the indexed RoPE op is interleaved-only; the DS
    path permutes half-split->interleaved before it). The CPU reference stores it interleaved for GLM
    (index_rope_interleave=True) but HALF-SPLIT for DS, so for DS we reindex the device's RoPE dims back
    to half-split (interleaved_to_halfsplit_perm) before comparing; the non-RoPE dims match directly."""
    nat = _cache_natural(tt_index_kv_cache, mesh_device, chunk, batch_slot=cache_batch_idx)
    # The Sylvester Hadamard matrix is symmetric and self-inverse, so applying it again restores the
    # pre-transform key. Do this before the DS RoPE-frame permutation because the transform was applied
    # while the key was still in the device's interleaved frame.
    hadamard = normalized_hadamard_matrix(nat.shape[-1])
    nat = (nat.float() @ hadamard.float()).to(torch.bfloat16)
    if not getattr(config, "index_rope_interleave", False):  # DS: device interleaved -> reference half-split
        rope_dim = config.qk_rope_head_dim
        perm = interleaved_to_halfsplit_perm(rope_dim)
        nat = nat.clone()
        nat[:, :rope_dim] = nat[:, :rope_dim][:, perm]
    return nat


def _device_topk_rows(indices, mesh_device, seq_len):
    """Device indexer selection -> [seq_len, k] key positions in natural row order.

    The indexer emits [1, 1, rows, k] per chip: SP-sharded rows, further split over TP when the layer
    reshards heads -> sequence (every TP > 1), else TP-replicated. Concatenate as [sp, tp, rows, k] and flatten in
    linear chip order (sp-major, tp-minor) -- one full-sequence chunk at offset 0 is in natural order."""
    sp, tp = mesh_device.shape[0], mesh_device.shape[1]
    t = ttnn.to_torch(
        indices, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_device.shape)
    )
    t = t.reshape(sp, tp, -1, t.shape[-1]).to(torch.int64)
    rows = t.shape[2]
    if rows * sp * tp == seq_len:  # TP sequence shards
        return t.reshape(seq_len, -1)
    if rows * sp == seq_len:  # TP-replicated SP shards
        return t[:, 0].reshape(seq_len, -1)
    assert rows == seq_len, f"unrecognized indexer index layout {tuple(t.shape)} for seq {seq_len}"
    return t[0, 0]


def _topk_recall(dev_topk, ref_topk, ref_scores, index_topk):
    """Per-row (raw, tie-tolerant) recall of the device selection over the rows with more than index_topk
    causal keys. Raw: |device ∩ reference| / k. Tie-tolerant additionally accepts a device key whose
    reference score is >= the reference k-th score minus SPARSE_TOPK_TIE_BAND x the row's causal range."""
    seq_len = ref_topk.shape[0]
    raw, tol = [], []
    for s in range(index_topk, seq_len):
        ref = set(ref_topk[s].tolist())
        # Distinct valid keys only: a duplicated index must not count twice (a top-k merge that repeats one
        # correct key k times would otherwise score 1.0).
        dev = list(dict.fromkeys(int(i) for i in dev_topk[s].tolist() if 0 <= int(i) <= s))
        row = ref_scores[s, : s + 1].float()
        kth = row[ref_topk[s].long()].min()
        band = SPARSE_TOPK_TIE_BAND * (row.max() - row.min())
        hits = sum(1 for i in dev if i in ref)
        near = sum(1 for i in dev if i not in ref and row[i] >= kth - band)
        raw.append(hits / len(ref))
        tol.append(min(1.0, (hits + near) / len(ref)))
    return torch.tensor(raw), torch.tensor(tol)


def run_sparse_mla_accuracy_case(
    variant,
    config,
    mesh_device,
    seq_len,
    topology,
    cache_format,
    ds_layer=None,
    ds_checkpoint=None,
    ds_repo=None,
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
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        tp_axis=tp_axis,
    )

    logger.info(f"[{variant.name}] sparse MLA accuracy: running TT inference")
    # Sparse has no single-shot path: one chunk, spanning the whole sequence, at offset 0.
    prunes = seq_len > config.index_topk
    tt_output, hidden_states, _, shard_dims, *rest = run_mla_inference(
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
        is_chunked=True,
        active_seq_len=seq_len,
        actual_start=0,
        return_indices=prunes,
    )

    cache_dir = cpu_ref_cache_dir(variant)
    logger.info(f"[{variant.name}] sparse MLA accuracy: running CPU reference")
    # ref_index (the indexer key-cache truth) is dropped: one full-sequence chunk fills a single
    # block-cyclic slab, so chunked/rotated are where that cache is asserted (SPARSE_INDEX_PCC).
    ref_output, ref_kvpe, _, ref_topk, ref_scores = run_cpu_reference(
        config, weights, hidden_states, seq_len, cache_dir, cache_tag=f"{src_tag}_funcidx", with_topk=True
    )
    if prunes:
        # G4: the real-pruning regime is exercised (some row keeps < its causal keys), and the device picked
        # (almost) the same keys as the reference.
        assert ref_topk.shape[-1] == config.index_topk < seq_len, "5120 case did not prune"
        raw, recall = _topk_recall(
            _device_topk_rows(rest[0], mesh_device, seq_len), ref_topk, ref_scores, config.index_topk
        )
        logger.info(
            f"[{variant.name}] top-k recall over {recall.numel()} pruning rows: raw mean={raw.mean():.5f} "
            f"min={raw.min():.5f}; tie-tolerant mean={recall.mean():.5f} min={recall.min():.5f}"
        )
        worst = int(recall.argmin()) + config.index_topk
        assert (
            recall.min() >= SPARSE_TOPK_RECALL
        ), f"device top-k recall {recall.min():.5f} < {SPARSE_TOPK_RECALL} (tie-tolerant) at row {worst}"

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
    tt_kvpe = _collect_kvpe_cache(tt_kvpe_cache, mesh_device)[:1, :1]
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
        tt_kvpe_cache = init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=1,
            tp_axis=tp_axis,
        )
        # Sparse has no single-shot path: one chunk, spanning the whole sequence, at offset 0.
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
            is_chunked=True,
            active_seq_len=seq_len,
            actual_start=0,
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
    variant, config, mesh_device, seq_len, chunk, cache_format, ds_layer, ds_checkpoint, ds_repo, ds_input
):
    """Sparse chunked prefill on nonzero user slot: compare ttMLA against MLACPU sparse chunked truth."""
    # Anchor mesh (TP>=2) and seq/SP validity are guaranteed by _sparse_cases (no runtime skips).
    #
    # CACHE-QUANTIZATION NOTE: the canonical CPU reference keeps KVPE in BF16. The BF16 device case
    # therefore compares like-for-like, while the scaled-FP8 case intentionally measures the complete
    # device quantize/store/gather/reconstruct path against BF16 truth. Any PCC delta between the two
    # formats is expected quantization loss; the cache and final-output checks below keep it distinct from
    # a cache-layout, block-cyclic-remap, or sparse-attention regression.
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
    num_users, cache_user_id = 2, 1
    logger.debug(f"[{variant.name}] sparse MLA chunked: initializing KVPE cache mesh_shape={mesh_shape}")
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=num_users,
        tp_axis=tp_axis,
    )

    tt_index_kv_cache = _init_index_kv_cache(
        config, mesh_device, seq_len, mesh_shape, sp_axis, slot_num=num_users, tp_axis=tp_axis
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
        active_seq_len=chunk,
        slot_num=num_users,
        layer_num=1,
        sparse_kv_cache_format=cache_format,
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
        out = mla_tt.forward(
            tt_x,
            rope_tensors,
            tt_kvpe_cache,
            actual_start=s,
            cache_user_id=cache_user_id,
            index_kv_cache=tt_index_kv_cache,
        )
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
    cache_all = _collect_kvpe_cache(tt_kvpe_cache, mesh_device)
    assert torch.count_nonzero(cache_all[0:1]) == 0, "sparse MLA wrote KVPE into unselected user slot 0"
    cache_sr = cache_all[cache_user_id : cache_user_id + 1, :1]
    p = blockcyclic_positions(sp * mesh_device.shape[1], chunk, cache_sr.shape[2])
    nat = torch.empty(cache_sr.shape[2], cache_sr.shape[-1], dtype=torch.bfloat16)
    nat[p] = cache_sr[0, 0]
    _, m = comp_pcc(ref_kvpe, nat[:seq_len].unsqueeze(0), 0)
    logger.info(f"[{variant.name}] kvpe prefix: {m}")

    logger.debug(f"[{variant.name}] sparse MLA chunked: collecting indexer key cache")
    index_cache_layers = num_full_indexer_layers(config) or 1
    index_cache_batch_idx = cache_user_id * index_cache_layers
    index_cache_host = ttnn.to_torch(
        tt_index_kv_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    )
    assert (
        torch.count_nonzero(index_cache_host[:index_cache_layers]) == 0
    ), "sparse MLA wrote index keys into unselected user slot 0"
    idx_nat = _collect_index_cache_natural(
        tt_index_kv_cache, mesh_device, config, chunk, cache_batch_idx=index_cache_batch_idx
    )
    _, idx_pcc = assert_with_pcc(ref_index[0, :seq_len], idx_nat[:seq_len], SPARSE_INDEX_PCC)
    logger.info(f"[{variant.name}] Chunked indexer cache PCC: {idx_pcc}")

    _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output, SPARSE_OUTPUT_PCC)
    logger.info(f"[{variant.name}] Chunked output PCC: {pcc_message}")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[{variant.name}] sparse MLA chunked complete")


def pad_overflow_schedule(chunk, cache_len):
    """[(actual_start, actual_end)] whose LAST padded window overruns ``cache_len`` while its real tokens fit.

    A serving stream is ragged: chunks are a fixed width but carry fewer real tokens, so actual_start
    drifts off the chunk grid (one tile per chunk here) until a padded window ends past the cache while
    ceil32(actual_end) still fits. That drift is the only way to overrun -- a chunk-aligned start never can.
    """
    short = chunk - ttnn.TILE_SIZE
    bounds, s = [], 0
    for _ in range(cache_len // chunk):
        bounds.append((s, s + short))
        s += short
    bounds.append((s, s + ttnn.TILE_SIZE))  # tail: one tile of real tokens, chunk - 32 of pad
    return bounds


def run_sparse_mla_pad_overflow_case(
    variant, config, mesh_device, seq_len, chunk, cache_format, ds_layer, ds_checkpoint, ds_repo, ds_input
):
    """Chunked sparse prefill whose LAST chunk pads past the end of the cache.

    This is the case the fixed chunk size forces on a server and the one update_padded_kv_cache's
    valid_global clamp exists for: the tail chunk's padded window runs past the cache, only its real
    tokens are written, and the indexer/top-k/gather are bounded by that written prefix instead of by
    the window. Asserted three ways -- the output over the real rows, both caches over the real rows,
    and the cache tail past the last real token still being untouched.
    """
    seed = 42
    bounds = pad_overflow_schedule(chunk, seq_len)
    total_real = bounds[-1][1]
    tail_start, tail_end = bounds[-1]
    write_end = -(-total_real // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    assert tail_start + chunk > seq_len, "tail window must overrun the cache or this test proves nothing"
    assert write_end <= seq_len, "the tail's REAL tokens must still fit the cache"
    logger.info(
        f"[{variant.name}] pad-overflow: {len(bounds)} chunks of width {chunk}, {total_real} real tokens, "
        f"cache {seq_len}; tail [{tail_start}, {tail_end}) pads to {tail_start + chunk} "
        f"({tail_start + chunk - seq_len} past the cache)"
    )

    weights, src_tag = build_weights(
        variant, config, seed=seed, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo
    )
    config.max_seq_len = seq_len

    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    num_users, cache_user_id = 2, 1
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=num_users,
        tp_axis=tp_axis,
    )
    tt_index_kv_cache = _init_index_kv_cache(
        config, mesh_device, seq_len, mesh_shape, sp_axis, slot_num=num_users, tp_axis=tp_axis
    )
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        active_seq_len=chunk,
        slot_num=num_users,
        layer_num=1,
        sparse_kv_cache_format=cache_format,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    # tail_slack: the tail chunk ropes its whole padded slab, so the table needs one chunk of headroom.
    rope_tensors = rope.get_rope_tensors_indexed(seq_len, chunk, tail_slack=True)

    hidden = make_hidden(seq_len, config.hidden_size, seed, ds_input)
    if ds_input:
        src_tag = f"{src_tag}_in{abs(hash(ds_input)) % 10**8}"
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2

    # ROTATED input order. A mid-slab actual_start rotates which chip owns which block, so chunk row j is
    # NOT global position actual_start + j -- feeding natural order only happens to work when every start
    # is slab-aligned (what test_sparse_mla_chunked does). Every start here is drifted, so gather the
    # chunk in rotated chip-major order and scatter the output back by the same map, exactly as
    # test_sparse_mla_rotated_chunked does.
    sp = mesh_device.shape[0]
    chunk_local = chunk // sp
    hid = hidden[0]  # [seq, hidden]
    out_accum = None
    for chunk_idx, (a, b) in enumerate(bounds):
        positions = rotated_chip_positions(a, sp, chunk_local)
        flat = [positions[c][r] for c in range(sp) for r in range(chunk_local)]
        gather_idx = torch.tensor([min(gp, hid.shape[0] - 1) for gp in flat], dtype=torch.long)
        chunk_in = hid[gather_idx].clone()
        is_pad = torch.tensor([gp >= b for gp in flat])
        chunk_in[is_pad] = 0.0  # pad rows: scored and attended on device, KV not written, output dropped
        logger.info(
            f"[{variant.name}] pad-overflow TT chunk {chunk_idx + 1}/{len(bounds)}: "
            f"actual_start={a} actual_end={b} ({b - a} real, {int(is_pad.sum())} pad)"
        )
        tt_x = ttnn.from_torch(
            chunk_in.reshape(1, 1, chunk, config.hidden_size),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )
        out = mla_tt.forward(
            tt_x,
            rope_tensors,
            tt_kvpe_cache,
            actual_start=a,
            actual_end=b,
            cache_user_id=cache_user_id,
            index_kv_cache=tt_index_kv_cache,
        )
        out_flat = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
        ).to(torch.bfloat16)[0, 0]
        if out_accum is None:
            out_accum = torch.zeros(total_real, out_flat.shape[-1], dtype=torch.bfloat16)
        valid = [(row, gp) for row, gp in enumerate(flat) if gp < b]
        out_accum[torch.tensor([gp for _, gp in valid])] = out_flat[torch.tensor([row for row, _ in valid])]
    tt_output = out_accum.reshape(1, 1, total_real, out_accum.shape[-1])

    cache_dir = cpu_ref_cache_dir(variant)
    # seq_len (the CACHE length), not total_real: the reference's rope table is sized by max_seq_len and
    # the YaRN gate keys off it, so a reference built at a different length ropes different frequencies
    # than the device. Its caches are still returned sliced to what was actually filled (total_real).
    ref_output, ref_kvpe, ref_index = run_cpu_reference_ragged(
        config, weights, hidden, seq_len, bounds, cache_dir, cache_tag=f"{src_tag}_funcidx"
    )

    cache_all = _collect_kvpe_cache(tt_kvpe_cache, mesh_device)
    assert torch.count_nonzero(cache_all[0:1]) == 0, "sparse MLA wrote KVPE into unselected user slot 0"
    cache_sr = cache_all[cache_user_id : cache_user_id + 1, :1]
    p = blockcyclic_positions(sp * mesh_device.shape[1], chunk, cache_sr.shape[2])
    nat = torch.empty(cache_sr.shape[2], cache_sr.shape[-1], dtype=torch.bfloat16)
    nat[p] = cache_sr[0, 0]
    _, m = comp_pcc(ref_kvpe, nat[:total_real].unsqueeze(0), 0)
    logger.info(f"[{variant.name}] pad-overflow kvpe prefix: {m}")
    # The clamp's whole job: nothing was written past the last real token's 32-row block. The cache is
    # zero-filled at allocation, so an unwritten tail is still exactly zero.
    assert (
        torch.count_nonzero(nat[write_end:]) == 0
    ), f"clamped write spilled past the real tokens: KVPE rows [{write_end}, {nat.shape[0]}) are nonzero"

    index_cache_layers = num_full_indexer_layers(config) or 1
    index_cache_batch_idx = cache_user_id * index_cache_layers
    idx_nat = _collect_index_cache_natural(
        tt_index_kv_cache, mesh_device, config, chunk, cache_batch_idx=index_cache_batch_idx
    )
    _, idx_pcc = assert_with_pcc(ref_index[0, :total_real], idx_nat[:total_real], SPARSE_INDEX_PCC)
    logger.info(f"[{variant.name}] pad-overflow indexer cache PCC: {idx_pcc}")
    assert (
        torch.count_nonzero(idx_nat[write_end:]) == 0
    ), f"clamped write spilled past the real tokens: index rows [{write_end}, {idx_nat.shape[0]}) are nonzero"

    _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output, SPARSE_OUTPUT_PCC)
    logger.info(f"[{variant.name}] pad-overflow output PCC: {pcc_message}")
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[{variant.name}] sparse MLA pad-overflow complete")


def run_sparse_mla_kv_only_case(
    variant, config, mesh_device, seq_len, chunk, cache_format, ds_layer, ds_checkpoint, ds_repo
):
    """Last-layer chunked fast path must populate KVPE and tiled index caches without an output."""
    seed = 42
    weights, src_tag = build_weights(
        variant, config, seed=seed, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo
    )
    config.max_seq_len = seq_len
    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        tp_axis=tp_axis,
    )
    tt_index_kv_cache = _init_index_kv_cache(config, mesh_device, seq_len, mesh_shape, sp_axis, tp_axis=tp_axis)
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=1,
        sparse_kv_cache_format=cache_format,
        kv_only=True,
    )
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(
        seq_len, chunk
    )
    hidden = make_hidden(chunk, config.hidden_size, seed)
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    tt_x = ttnn.from_torch(
        hidden.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=shard_dims),
    )

    out = mla_tt.forward(
        tt_x,
        rope_tensors,
        tt_kvpe_cache,
        actual_start=0,
        index_kv_cache=tt_index_kv_cache,
    )
    assert out is None

    _, ref_kvpe, ref_index = run_cpu_reference(
        config,
        weights,
        hidden,
        chunk,
        cpu_ref_cache_dir(variant),
        cache_tag=f"{src_tag}_funcidx_kvonly",
    )
    cache_sr = _collect_kvpe_cache(tt_kvpe_cache, mesh_device)[:, :1]
    # sp*tp stripes -- see _collect_kvpe_cache.
    positions = blockcyclic_positions(mesh_shape[sp_axis] * mesh_shape[tp_axis], chunk, cache_sr.shape[2])
    cache_natural = torch.empty(cache_sr.shape[2], cache_sr.shape[-1], dtype=torch.bfloat16)
    cache_natural[positions] = cache_sr[0, 0]
    _, kv_msg = assert_with_pcc(ref_kvpe, cache_natural[:chunk].unsqueeze(0).unsqueeze(0), SPARSE_KVPE_PCC)

    index_natural = _collect_index_cache_natural(tt_index_kv_cache, mesh_device, config, chunk)
    _, index_msg = assert_with_pcc(ref_index[0, :chunk], index_natural[:chunk], SPARSE_INDEX_PCC)
    logger.info(f"[{variant.name}] kv_only cache PCC: kvpe={kv_msg} index={index_msg}")
    ttnn.synchronize_device(mesh_device)


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
    # GLM-5.3 KV dedup: caches striped over ALL sp*tp chips instead of TP-replicated. The QUERY sharding
    # (and hence the rotation pattern fed to the model below) is unchanged -- only the cache layout and
    # its readback stride change, which is exactly the decoupling this case is meant to stress.
    kv_tp_axis = tp_axis  # the sparse path always dedups; there is no TP-replicated variant
    stripe = chunk_size_global // (sp * mesh_shape[tp_axis])
    assert stripe % tile == 0, (
        f"the deduped cache needs a tile-aligned per-chip stripe: chunk {chunk_size_global} / "
        f"(sp {sp} * tp {mesh_shape[tp_axis]}) = {stripe}"
    )
    for v in iters_isl:
        assert 0 < v <= chunk_size_global and v % tile == 0, f"iter isl {v}: tile-aligned, <= {chunk_size_global}"
    total_len = sum(iters_isl)

    weights, src_tag = build_weights(
        variant, config, seed=seed, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo
    )
    max_window = max(sum(iters_isl[:i]) + chunk_size_global for i in range(len(iters_isl)))
    seq_len_cache = ((max_window + chunk_size_global - 1) // chunk_size_global) * chunk_size_global
    config.max_seq_len = seq_len_cache
    kv_layout = f"SPxTP-deduped (stripe={stripe})"
    logger.info(
        f"[{variant.name}] rotated: iters={iters_isl} total={total_len} chunk={chunk_size_global} "
        f"cache={seq_len_cache} mesh={tuple(mesh_device.shape)} kv={kv_layout}"
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

    tt_index_kv_cache = _init_index_kv_cache(
        config, mesh_device, seq_len_cache, mesh_shape, sp_axis, slot_num=1, tp_axis=kv_tp_axis
    )
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len_cache,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        active_seq_len=chunk_size_global,
        slot_num=1,
        layer_num=1,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(
        cache_seq_len_global=seq_len_cache, chunk_size_global=chunk_size_global
    )
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        tp_axis=kv_tp_axis,
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
    nat = _cache_natural(tt_kvpe_cache.storage, mesh_device, chunk_size_global)
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


# glm_5_2 only: writer / indexer_score rotation geometry is model-agnostic, and K3.1's NoPE removes the only
# model-specific piece (the indexed rope); K3.1's multi-chunk coverage is test_sparse_mla_chunked on every mesh.
@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params",
    _only(SPARSE_ANCHOR_CASES, "glm_5_2"),
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.parametrize("iters_isl", [[2560, 1600, 5120]], ids=["maxedge"])
# KV dedup under ROTATION is the interesting case (test_sparse_mla_cache.py only starts slab-aligned):
# the writer rotates at sp*tp stripes while indexer_score's causal geometry rotates at sp, and the two
# only coincide for an aligned start. maxedge gives starts 0 / 2560 / 4160 -- aligned, mid-slab on an SP
# boundary, and a full chunk starting mid SP slab (4160 % 640 = 320) that straddles into the next slab.
# The last one is where query ownership (SP rotation + TP window) and a per-device stripe rotation
# disagree on a fused full-mesh ring (#58339).
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_rotated_chunked(
    mesh_device, seq_len, iters_isl, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo
):
    run_sparse_mla_rotated_case(
        variant,
        config_only,
        mesh_device,
        iters_isl,
        seq_len,
        ds_layer,
        ds_checkpoint,
        ds_repo,
    )


# One combined parametrization instead of independent variant/mesh/sequence/format axes. BF16 retains
# both sparsity regimes; scaled FP8 is restricted to the real-pruning sequence to avoid redundant CI work.
@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params, cache_format",
    SPARSE_ACCURACY_CASES,
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_accuracy_chunked(
    mesh_device,
    seq_len,
    device_params,
    cache_format,
    variant,
    config_only,
    ds_layer,
    ds_checkpoint,
    ds_repo,
    monkeypatch,
):
    original = ttnn.experimental.ring_indexer_score_dsa

    def check_indexer_q_format(q, *args, **kwargs):
        assert q.dtype == ttnn.bfloat8_b
        return original(q, *args, **kwargs)

    monkeypatch.setattr(ttnn.experimental, "ring_indexer_score_dsa", check_indexer_q_format)
    topology = per_axis_topology(device_params["fabric_config"])
    run_sparse_mla_accuracy_case(
        variant,
        config_only,
        mesh_device,
        seq_len,
        topology,
        cache_format,
        ds_layer,
        ds_checkpoint,
        ds_repo,
    )


# GLM-5.2 indexer reuse, the reuse-capable variant, on every mesh: TP=2 injects TP-sequence-sharded indices too.
SPARSE_REUSE_CASES = _only(SPARSE_CHUNKED_CASES, "glm_5_2")


@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params",
    SPARSE_REUSE_CASES,
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_indexer_reuse_chunked(
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
    topology = per_axis_topology(device_params["fabric_config"])

    def _kvpe():
        return init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BF16_RM,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=1,
            tp_axis=tp_axis,  # KV dedup: the sparse path stripes the cache over SP*TP, its only layout
        )

    # Sparse has no single-shot path: one chunk, spanning the whole sequence, at offset 0.
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
        is_chunked=True,
        active_seq_len=seq_len,
        actual_start=0,
    )
    # A: compute the indexer, capture its top-k selection + output.
    out_a, _, _, shard_dims, idx = run_mla_inference(tt_kvpe_cache=_kvpe(), return_indices=True, **common)
    # B: a shared layer (same weights + input) fed A's indices -> it has no indexer of its own.
    shared_idx = next(i for i, t in enumerate(config.indexer_types) if t == "shared")
    out_b, _, _, _ = run_mla_inference(tt_kvpe_cache=_kvpe(), inject_indices=idx, layer_idx=shared_idx, **common)

    def _to_torch(t):
        return ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
        ).to(torch.bfloat16)

    _, pcc_message = assert_with_pcc(_to_torch(out_a), _to_torch(out_b), 0.9999)
    logger.info(f"[{variant.name}] indexer-reuse compute-vs-inject PCC: {pcc_message}")
    ttnn.synchronize_device(mesh_device)


# Anchor cases (per-variant prod-closest mesh, seq=4096); collected == run.
@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params",
    SPARSE_ANCHOR_CASES,
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.parametrize("n_runs", [3], ids=["x3"])
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_determinism_chunked(
    mesh_device, seq_len, n_runs, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo
):
    topology = per_axis_topology(device_params["fabric_config"])
    run_sparse_mla_determinism_case(
        variant, config_only, mesh_device, seq_len, n_runs, topology, ds_layer, ds_checkpoint, ds_repo
    )


# SPARSE_CHUNKED_CASES (every variant on every mesh, seq=5120) crossed with the prefill chunk sizes; collected == run.
def _chunked_cases():
    """(case, chunk): every case at 1024, plus the 1280 perf chunk where it tiles the SP x TP stripe (it lands in the
    tuned 640-row slot on LoudBox 2x4; Galaxy's 32-chip stripe needs 1024 multiples)."""
    out = []
    for c in SPARSE_CHUNKED_CASES:
        sp, tp = c.values[1]
        for chunk in (1024, 1280):
            if chunk == 1024 or chunk % (ttnn.TILE_SIZE * sp * tp) == 0:
                # Ids read cache_format-chunk-case.
                chunk_id = "c1k" if chunk == 1024 else f"c{chunk}"
                out.append(pytest.param(*c.values, chunk, marks=c.marks, id=f"{chunk_id}-{c.id}"))
    return out


@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params, chunk",
    _chunked_cases(),
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "cache_format",
    [MlaKvCacheFormat.BF16_RM, MlaKvCacheFormat.SCALED_FP8],
    ids=["kv_bf16", "kv_scaled_fp8"],
)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_chunked(
    mesh_device,
    seq_len,
    chunk,
    cache_format,
    device_params,
    variant,
    config_only,
    ds_layer,
    ds_checkpoint,
    ds_repo,
    ds_input,
):
    run_sparse_mla_chunked_case(
        variant, config_only, mesh_device, seq_len, chunk, cache_format, ds_layer, ds_checkpoint, ds_repo, ds_input
    )


# The clamp itself is model-agnostic, but K3.1's terminal pad overflow runs under NoPE (no indexed rope on
# the pad rows) with its scaled-FP8 main cache, so both variants run their anchor case. Cache format IS
# kept -- it changes the page layout the clamp addresses.
SPARSE_PAD_OVERFLOW_CASES = _only(SPARSE_ANCHOR_CASES, "glm_5_2", "kimi_k3_1")


@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params",
    SPARSE_PAD_OVERFLOW_CASES,
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.parametrize("chunk", [1024], ids=["c1k"])
@pytest.mark.parametrize(
    "cache_format",
    [MlaKvCacheFormat.BF16_RM, MlaKvCacheFormat.SCALED_FP8],
    ids=["kv_bf16", "kv_scaled_fp8"],
)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_pad_overflow_chunked(
    mesh_device,
    seq_len,
    chunk,
    cache_format,
    device_params,
    variant,
    config_only,
    ds_layer,
    ds_checkpoint,
    ds_repo,
    ds_input,
):
    """Ragged chunk stream whose tail chunk pads past the end of the cache (see pad_overflow_schedule)."""
    run_sparse_mla_pad_overflow_case(
        variant, config_only, mesh_device, seq_len, chunk, cache_format, ds_layer, ds_checkpoint, ds_repo, ds_input
    )


# glm_5_2 only: K3.1's NoPE KVPE write is already PCC-checked by the accuracy and chunked tests.
SPARSE_KV_ONLY_CASES = _only(SPARSE_ANCHOR_CASES, "glm_5_2")


@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params",
    SPARSE_KV_ONLY_CASES,
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.parametrize("chunk", [1024], ids=["c1k"])
@pytest.mark.parametrize(
    "cache_format",
    [MlaKvCacheFormat.BF16_RM, MlaKvCacheFormat.SCALED_FP8],
    ids=["kv_bf16", "kv_scaled_fp8"],
)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_kv_only_chunked(
    mesh_device, seq_len, chunk, cache_format, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo
):
    run_sparse_mla_kv_only_case(
        variant, config_only, mesh_device, seq_len, chunk, cache_format, ds_layer, ds_checkpoint, ds_repo
    )


# ---------------------------------------------------------------------------
# Kimi-K3.1 index / KVPE cache slots: the 24 DSA layers share ONE MLA-ordinal slot space.
# ---------------------------------------------------------------------------
# K3.1 carries no indexer_types, so TtIndexer takes its non-compact path: slot = cache_layer_idx, stride =
# layer_num -- exactly the KVPE cache's addressing. Model layers 3, 7, 91, 92 are MLA slots 0, 1, 22, 23
# (KimiK31Config.mla_kv_slot). Write one layer per slot (distinct inputs) into 24-layer x 2-user caches for
# user 1, so the flat slot is user * 24 + layer, and check each written slot holds its own index keys AND
# KVPE rows, each layer's output matches the reference, and every other slot (user 0 included) is untouched.
K31_SLOT_MODEL_LAYERS = (3, 7, 91, 92)
# <= index_topk: cheap CPU truth; the slot addressing does not depend on pruning. Shortest valid per mesh.
K31_SLOT_SEQS = SPARSE_INERT_FALLBACK_SEQS
SPARSE_SLOT_CASES = []
for _mesh in [_anchor_mesh(_sparse_meshes())] if _sparse_meshes() else []:
    _seq = _single_slot_seq_for_mesh(K31_SLOT_SEQS, _mesh)
    assert _seq is not None, f"no K3.1 slot-test sequence fits mesh {_mesh}"
    SPARSE_SLOT_CASES += [
        c for c in _sparse_cases([_seq], anchor_only=True, variants=["kimi_k3_1"]) if c.values[1] == _mesh
    ]


@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params",
    SPARSE_SLOT_CASES,
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_k31_cache_slots_isolated(mesh_device, seq_len, device_params, variant, config_only):
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config

    config = config_only
    config.max_seq_len = seq_len
    num_layers = len(KimiK31Config.mla_layer_ids())
    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    cache_format = SPARSE_MAIN_KV_FORMAT[variant.name]
    weights, src_tag = build_weights(variant, config, seed=42)
    num_users, user = 2, 1

    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
        tp_axis=tp_axis,
    )
    tt_index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=num_users,
        dtype=ttnn.bfloat8_b,
        tp_axis=tp_axis,
    )
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(
        seq_len, seq_len
    )
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2

    slots = {}
    for model_layer in K31_SLOT_MODEL_LAYERS:
        slot = KimiK31Config.mla_kv_slot(model_layer)
        mla_tt = ttMLA(
            config,
            weights,
            mesh_device,
            layer_idx=model_layer,
            seq_len=seq_len,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            is_chunked=True,
            active_seq_len=seq_len,
            layer_num=num_layers,
            slot_num=num_users,
            sparse_kv_cache_format=cache_format,
            # Four layer instances in one process: keep them on the serial schedule so no instance holds
            # the shared overlap manager while the next is built (overlap is covered by its own tests).
            sparse_mla_overlap_profile="off",
        )
        assert mla_tt._indexer.index_cache_layers == num_layers
        hidden = make_hidden(seq_len, config.hidden_size, seed=100 + slot)
        tt_x = ttnn.from_torch(
            hidden.unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )
        out = mla_tt.forward(
            tt_x,
            rope_tensors,
            tt_kvpe_cache,
            actual_start=0,
            cache_layer_idx=slot,
            cache_user_id=user,
            index_kv_cache=tt_index_kv_cache,
        )
        out = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
        ).to(torch.bfloat16)
        slots[slot] = (hidden, out)
        ttnn.synchronize_device(mesh_device)

    cache_dir = cpu_ref_cache_dir(variant)
    index_host = ttnn.to_torch(
        tt_index_kv_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    )
    kvpe_all = _collect_kvpe_cache(tt_kvpe_cache, mesh_device)  # [num_users * num_layers, 1, seq, kvpe]
    stripes = mesh_device.shape[0] * mesh_device.shape[1]
    for flat in range(num_users * num_layers):
        u, slot = divmod(flat, num_layers)
        if u != user or slot not in slots:
            assert torch.count_nonzero(index_host[flat]) == 0, f"index cache user {u} slot {slot} was written"
            assert torch.count_nonzero(kvpe_all[flat].float()) == 0, f"KVPE cache user {u} slot {slot} was written"
            continue
        hidden, out = slots[slot]
        ref_output, ref_kvpe, ref_index = run_cpu_reference(
            config, weights, hidden, seq_len, cache_dir, cache_tag=f"{src_tag}_slot{slot}_seed{100 + slot}"
        )
        _, out_msg = assert_with_pcc(ref_output.unsqueeze(0), out, SPARSE_OUTPUT_PCC)
        idx_nat = _collect_index_cache_natural(tt_index_kv_cache, mesh_device, config, seq_len, cache_batch_idx=flat)
        _, idx_msg = assert_with_pcc(ref_index[0, :seq_len], idx_nat[:seq_len], SPARSE_INDEX_PCC)
        p = blockcyclic_positions(stripes, seq_len, kvpe_all.shape[2])
        kv_nat = torch.empty(kvpe_all.shape[2], kvpe_all.shape[-1], dtype=torch.bfloat16)
        kv_nat[p] = kvpe_all[flat, 0].to(torch.bfloat16)
        _, kv_msg = assert_with_pcc(ref_kvpe.reshape(seq_len, -1), kv_nat[:seq_len], SPARSE_KVPE_PCC)
        logger.info(f"[{variant.name}] user {u} slot {slot}: output {out_msg}, index-K {idx_msg}, KVPE {kv_msg}")
    ttnn.synchronize_device(mesh_device)


# ---------------------------------------------------------------------------
# Trace replay correctness: capture ONE forward, replay it per chunk with new on-device metadata.
# ---------------------------------------------------------------------------
# The perf suite replays traces but compares nothing. Here a single captured chunked forward (metadata path:
# slot / actual_start / actual_end read on-device) is replayed for every chunk of TWO users' prefixes:
#   * user 1: a variable schedule (TRACE_ITERS_USER1) -> rotated, mid-slab starts (1600, 2624, 3648) and a
#     partial tail chunk (actual_end < start + chunk), the shapes a frozen capture-time start would break;
#   * user 0: one partial (992-token) chunk of a different input, so a trace that froze the capture-time user
#     fails and user 0's tail check covers rows its chunk touches.
# Inputs follow the rotated chip order (rotated_chip_positions), as the eager rotated test does. Checked
# against each user's single-shot CPU truth: every chunk's output, both caches over the real prefix, and
# the cache tail past each user's end, which a writer ignoring actual_end would overwrite.
TRACE_CHUNK = 1024
# Each iteration fits the captured chunk; starts 0 / 1024 / 1600 (rotated) / 2624 (rotated) / 3648 (rotated), end 4608.
TRACE_ITERS_USER1 = [1024, 576, 1024, 1024, 960]
TRACE_ITERS_USER0 = [992]  # partial too, so user 0's tail check covers rows a chunk actually touches
TRACE_REGION_SIZE = 64 * 1024 * 1024


def _trace_cases():
    """Every variant on every mesh of the box (SPARSE_CHUNKED_CASES): the TP=2 early-Q path replays too."""
    out = []
    anchor_ids = {c.id for c in SPARSE_ANCHOR_CASES}
    for c in SPARSE_ANCHOR_CASES + [c for c in SPARSE_CHUNKED_CASES if c.id not in anchor_ids]:
        variant, mesh, seq_len, params = c.values
        out.append(
            pytest.param(
                variant, mesh, seq_len, {**params, "trace_region_size": TRACE_REGION_SIZE}, marks=c.marks, id=c.id
            )
        )
    return out


@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params",
    _trace_cases(),
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_trace_replay_chunked(mesh_device, seq_len, device_params, variant, config_only):
    from models.demos.deepseek_v3_d_p.tt.mla.rope import write_chunk_metadata
    from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

    config = config_only
    config.max_seq_len = seq_len
    chunk = TRACE_CHUNK
    sp, tp = mesh_device.shape[0], mesh_device.shape[1]
    chunk_local = chunk // sp
    cache_format = SPARSE_MAIN_KV_FORMAT[variant.name]
    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    num_users = 2
    weights, src_tag = build_weights(variant, config, seed=42)
    hiddens = {
        1: make_hidden(seq_len, config.hidden_size, seed=42)[0],
        0: make_hidden(seq_len, config.hidden_size, seed=7)[0],
    }
    schedule = []  # (user, start, end) in replay order
    for user, iters in ((1, TRACE_ITERS_USER1), (0, TRACE_ITERS_USER0)):
        start = 0
        for isl in iters:
            assert isl % ttnn.TILE_SIZE == 0 and isl <= chunk and start + chunk <= seq_len
            schedule.append((user, start, start + isl))
            start += isl
    totals = {u: max(e for uu, _, e in schedule if uu == u) for u in (0, 1)}

    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=num_users,
        tp_axis=tp_axis,
    )
    tt_index_kv_cache = _init_index_kv_cache(
        config, mesh_device, seq_len, mesh_shape, sp_axis, slot_num=num_users, tp_axis=tp_axis
    )
    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        active_seq_len=chunk,
        slot_num=num_users,
        layer_num=1,
        sparse_kv_cache_format=cache_format,
    )
    rope_tensors = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(
        cache_seq_len_global=seq_len, chunk_size_global=chunk
    )
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)

    def chunk_layout(start, end):
        """Rotated chip order for a chunk starting at `start`: per-row global positions, pad rows >= end."""
        positions = rotated_chip_positions(start, sp, chunk_local)
        return [positions[c][r] for c in range(sp) for r in range(chunk_local)]

    pad_noise = torch.randn(chunk, config.hidden_size, generator=torch.Generator().manual_seed(99)).to(torch.bfloat16)

    def host_chunk(user, start, end):
        flat = chunk_layout(start, end)
        x = hiddens[user][torch.tensor([min(gp, seq_len - 1) for gp in flat])].clone()
        # Pad rows carry NONZERO noise: zero rows would project to zero KVPE / index keys, so a writer ignoring
        # actual_end would write zeros past the end and the tail-zero checks below could not see it. Causality
        # keeps pad rows out of every valid row's attention, so the noise cannot change a checked output.
        pad = torch.tensor([gp >= end for gp in flat])
        x[pad] = pad_noise[: int(pad.sum())]
        return ttnn.from_torch(
            x.reshape(1, 1, chunk, config.hidden_size), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
        )

    tt_x = ttnn.to_device(host_chunk(*schedule[0]), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    metadata = tuple(
        ttnn.from_torch(
            torch.tensor([v], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for v in schedule[0]
    )

    def forward():
        return mla_tt.forward(
            hidden_states=tt_x,
            rope_tensors=rope_tensors,
            kvpe_cache=tt_kvpe_cache,
            actual_start=None,
            cache_user_id=schedule[0][0],  # unused on the metadata path (slot comes from metadata[0])
            metadata=metadata,
            index_kv_cache=tt_index_kv_cache,
        )

    controller = SubDeviceTraceController(mesh_device)
    mla_tt.set_trace_controller(controller)
    capture_out = None
    capturing = False
    outs = {0: torch.zeros(seq_len, config.hidden_size, dtype=torch.bfloat16), 1: None}
    outs[1] = outs[0].clone()
    try:
        # Compile at the first chunk, then capture it: both rewrite the same rows with the same data.
        ttnn.deallocate(forward())
        ttnn.synchronize_device(mesh_device)
        controller.begin_capture()
        capturing = True
        capture_out = forward()
        capturing = False  # before end_capture: if it raises, finally must not end the capture a second time
        controller.end_capture()
        ttnn.synchronize_device(mesh_device)
        for user, start, end in schedule:
            write_chunk_metadata(
                metadata,
                (user, start, end),
                hf_config=config,
                mesh_device=mesh_device,
                chunk_size_global=chunk,
                sp_axis=sp_axis,
            )
            ttnn.copy_host_to_device_tensor(host_chunk(user, start, end), tt_x)
            controller.replay()
            ttnn.synchronize_device(mesh_device)
            out_flat = ttnn.to_torch(capture_out, mesh_composer=composer).to(torch.bfloat16)[0, 0]
            valid = [(row, gp) for row, gp in enumerate(chunk_layout(start, end)) if gp < end]
            outs[user][torch.tensor([gp for _, gp in valid])] = out_flat[torch.tensor([row for row, _ in valid])]
    finally:
        # Cleanup must never replace the test's own failure: a failed capture can leave the controller
        # mid-capture, so end_capture / release may raise too. Re-raise a cleanup error only on success.
        failing = sys.exc_info()[0] is not None
        cleanup_steps = [
            lambda: controller.end_capture() if capturing else None,
            controller.release,
            lambda: mla_tt.set_trace_controller(None),
            lambda: ttnn.deallocate(capture_out) if capture_out is not None else None,
            mla_tt.release_sparse_mla_overlap_manager,
        ]
        first_cleanup_error = None
        for step in cleanup_steps:  # every step runs, whatever an earlier one did
            try:
                step()
            except BaseException as cleanup_error:  # noqa: BLE001 -- incl. KeyboardInterrupt: later steps still run
                logger.warning(f"trace cleanup step failed: {cleanup_error!r}")
                if first_cleanup_error is None:
                    first_cleanup_error = cleanup_error
        if first_cleanup_error is not None and not failing:
            raise first_cleanup_error

    cache_dir = cpu_ref_cache_dir(variant)
    kvpe_all = _collect_kvpe_cache(tt_kvpe_cache, mesh_device)  # [num_users, 1, seq, kvpe]
    p = blockcyclic_positions(sp * tp, chunk, kvpe_all.shape[2])
    for user in (1, 0):
        total = totals[user]
        ref_output, ref_kvpe, ref_index = run_cpu_reference(
            config, weights, hiddens[user][:total].unsqueeze(0), total, cache_dir, cache_tag=f"{src_tag}_trace_u{user}"
        )
        for u, start, end in schedule:
            if u == user:
                _, msg = assert_with_pcc(ref_output[0, start:end], outs[user][start:end], SPARSE_OUTPUT_PCC)
                logger.info(f"[{variant.name}] traced replay user {user} [{start},{end}): output PCC {msg}")
        kv_nat = torch.empty(kvpe_all.shape[2], kvpe_all.shape[-1], dtype=torch.bfloat16)
        kv_nat[p] = kvpe_all[user, 0].to(torch.bfloat16)
        _, kv_msg = assert_with_pcc(ref_kvpe.reshape(total, -1), kv_nat[:total], SPARSE_KVPE_PCC)
        assert torch.count_nonzero(kv_nat[total:].float()) == 0, f"user {user}: KVPE written past actual_end {total}"
        idx_nat = _collect_index_cache_natural(
            tt_index_kv_cache, mesh_device, config, chunk, cache_batch_idx=user * (num_full_indexer_layers(config) or 1)
        )
        _, idx_msg = assert_with_pcc(ref_index[0, :total], idx_nat[:total], SPARSE_INDEX_PCC)
        assert (
            torch.count_nonzero(idx_nat[total:].float()) == 0
        ), f"user {user}: index-K written past actual_end {total}"
        logger.info(f"[{variant.name}] traced replay user {user} caches: KVPE PCC {kv_msg}, index-K PCC {idx_msg}")
    ttnn.synchronize_device(mesh_device)


# ---------------------------------------------------------------------------
# Kimi-K3.1 early-Q reshard: identity rope exactness and equivalence with the late-reshard path.
# ---------------------------------------------------------------------------
def _early_q_chunk(mesh) -> int:
    """The perf chunk (1280) when it tiles the SP x TP stripe and leaves the single-slot caches >= 64 rows per chip
    (KVPE_MIN_TOKENS_PER_CHIP); otherwise 2048 (Galaxy 8x4 / 8x2)."""
    sp, tp = mesh
    for chunk in (1280, 2048):
        if chunk % (ttnn.TILE_SIZE * sp * tp) == 0 and chunk // (sp * tp) >= KVPE_MIN_TOKENS_PER_CHIP:
            return chunk
    raise AssertionError(f"no early-Q test chunk fits mesh {mesh}")


def _bits_equal_modulo_flush(expected: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Per-element bf16 bit equality, allowing the two documented exceptions of the identity rope: a subnormal
    input flushed to (signed) zero, and -0 written as +0."""
    e, a = expected.contiguous().view(torch.int16), actual.contiguous().view(torch.int16)
    same = e == a
    e_abs = expected.abs()
    tiny = torch.finfo(torch.bfloat16).tiny
    flushed = ((e_abs < tiny) & (actual == 0)) | ((expected == 0) & (actual == 0))
    return same | flushed


@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params",
    [c for c in SPARSE_CHUNKED_CASES if c.values[0] == "kimi_k3_1"],
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
def test_identity_rope_is_a_fused_concat(mesh_device, seq_len, device_params, variant, config_only, expect_error):
    """rotary_embedding_indexed with identity tables writes concat(prefix, input) row-major, bit for bit except
    subnormal flush / signed zero; an input taller than the one-slab table is rejected."""
    from models.demos.deepseek_v3_d_p.tt.mla.rope import identity_rope_tensors_indexed

    sp_axis, tp_axis = 0, 1
    sp, tp = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
    chunk = _early_q_chunk((sp, tp))
    rows = chunk // (sp * tp)
    heads, rope_dim, prefix_dim = (
        config_only.num_attention_heads,
        config_only.qk_rope_head_dim,
        config_only.kv_lora_rank,
    )
    torch.manual_seed(5)
    q_pe = torch.randn(tp, heads, rows * sp, rope_dim).to(torch.bfloat16)
    special = torch.tensor(
        [0.0, -0.0, 1e-39, -1e-39, 3.0e38, -3.0e38, torch.finfo(torch.bfloat16).tiny, 1.0], dtype=torch.bfloat16
    )
    q_pe.view(-1)[: special.numel()] = special  # signed zeros, subnormals, normal extremes
    prefix = torch.randn(tp, heads, rows * sp, prefix_dim).to(torch.bfloat16)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(2, 0), mesh_shape=tuple(mesh_device.shape))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=mesh_device.shape)

    def dev(t):
        return ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper)

    tables = identity_rope_tensors_indexed(mesh_device, sp_axis=sp_axis, chunk_size_global=chunk, rope_dim=rope_dim)
    hifi4 = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4)

    def run(x, pre):
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            x,
            tables["cos_matrix"],
            tables["sin_matrix"],
            tables["trans_matrix"],
            kv_actual_global=0,
            cluster_axis=sp_axis,
            compute_kernel_config=hifi4,
            seq_subshard_axis=tp_axis,
            concat_prefix=pre,
            row_major_output=True,
        )

    out = ttnn.to_torch(run(dev(q_pe), dev(prefix)), mesh_composer=composer).to(torch.bfloat16)
    expected = torch.cat([prefix, q_pe], dim=-1)
    assert out.shape == expected.shape
    ok = _bits_equal_modulo_flush(expected, out)
    assert ok.all(), f"{int((~ok).sum())} elements differ beyond subnormal flush / signed zero"
    assert torch.equal(out[..., :prefix_dim].view(torch.int16), prefix.view(torch.int16)), "prefix not copied verbatim"

    taller = torch.randn(tp, heads, rows * sp * 2, rope_dim).to(torch.bfloat16)
    with expect_error(RuntimeError, "cos"):
        run(dev(taller), dev(torch.randn(tp, heads, rows * sp * 2, prefix_dim).to(torch.bfloat16)))


def _single_chunk_forward(mesh_device, config, weights, hidden, chunk, cache_format, early_q):
    """One sparse layer over one chunk at offset 0 on fresh caches. Returns the layer and (output, KVPE, index-K)."""
    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    mla = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=chunk,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=1,
        sparse_kv_cache_format=cache_format,
        sparse_mla_overlap_profile="off",
    )
    assert mla._early_q_reshard == early_q
    kvpe = init_mla_kv_cache(
        cache_format=cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=chunk,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        tp_axis=tp_axis,
    )
    index_kv = _init_index_kv_cache(config, mesh_device, chunk, mesh_shape, sp_axis, tp_axis=tp_axis)
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False).get_rope_tensors_indexed(chunk, chunk)
    tt_x = ttnn.from_torch(
        hidden.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    out = mla.forward(tt_x, rope, kvpe, actual_start=0, index_kv_cache=index_kv)
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
    out = ttnn.to_torch(out, mesh_composer=composer).to(torch.bfloat16)
    ttnn.synchronize_device(mesh_device)
    return mla, (out, _collect_kvpe_cache(kvpe, mesh_device), _cache_natural(index_kv, mesh_device, chunk))


def _early_vs_late_cases():
    """K3.1 meshes at the perf chunk; TP=2 also at 2560, the chunk whose 640 rows/chip hit the tuned TP=2 wkv_b1."""
    out = []
    for c in SPARSE_CHUNKED_CASES:
        if c.values[0] != "kimi_k3_1":
            continue
        mesh = c.values[1]
        chunks = [_early_q_chunk(mesh)] + ([2560] if mesh[1] == 2 and 2560 // mesh[0] == 640 else [])
        out += [pytest.param(*c.values, ch, marks=c.marks, id=f"{c.id}-c{ch}") for ch in chunks]
    return out


@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params, chunk",
    _early_vs_late_cases(),
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_k31_early_q_matches_late_q(
    mesh_device, seq_len, chunk, device_params, variant, config_only, monkeypatch
):
    """Two K3.1 layers on the same weights and input, one forced onto the late (absorbed-q) reshard. The q sparse_sdpa
    receives (sequence-sharded in both paths) must match channel-wise -- the 512 absorbed channels exactly, the 64
    identity channels up to subnormal flush / signed zero -- and so must the output, KVPE and index caches."""
    config = config_only
    config.max_seq_len = chunk
    sp_axis, tp_axis = 0, 1
    tp = mesh_device.shape[tp_axis]
    weights, _ = build_weights(variant, config, seed=42)
    hidden = make_hidden(chunk, config.hidden_size, seed=3)
    late_rule = ttMLA.replicates_wkv_b1
    sdpa = ttnn.transformer.sparse_sdpa
    captured = []

    def capture_q(q, *args, **kwargs):
        # Both paths hand sparse_sdpa a row-major, sequence-sharded q: per chip [1, H, S/(sp*tp), 576].
        captured.append(
            ttnn.to_torch(
                q, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_device.shape)
            )
        )
        return sdpa(q, *args, **kwargs)

    monkeypatch.setattr(ttnn.transformer, "sparse_sdpa", capture_q)

    def run(early):
        patch = pytest.MonkeyPatch()
        if not early:  # force the late reshard: TP-sharded wkv_b1 and the absorbed-q all-to-all for NoPE layers
            patch.setattr(
                ttMLA,
                "replicates_wkv_b1",
                staticmethod(lambda c, t, h: late_rule(c, t, h) and not getattr(c, "mla_use_nope", False)),
            )
        try:
            mla, results = _single_chunk_forward(
                mesh_device, config, weights, hidden, chunk, MlaKvCacheFormat.SCALED_FP8, early_q=early
            )
            heads_on_chip = mla.wkv_b1_weight.shape[1]
            assert heads_on_chip == (config.num_attention_heads if early else config.num_attention_heads // tp)
            if chunk // mesh_device.shape[sp_axis] == 640:  # the tuned 640-row slot this case exists to exercise
                assert mla._resolve_mm_cfg("wkv_b1", 640) is not None, "tuned wkv_b1 slot did not resolve"
            return results
        finally:
            patch.undo()

    late, early = run(early=False), run(early=True)
    assert len(captured) == 2, f"expected one sparse_sdpa call per run, got {len(captured)}"
    q_late, q_early = captured
    assert q_late.shape == q_early.shape
    kv = config.kv_lora_rank
    assert torch.equal(
        q_late[..., :kv].view(torch.int16), q_early[..., :kv].view(torch.int16)
    ), "absorbed q channels differ between the early and late reshard"
    ok = _bits_equal_modulo_flush(q_late[..., kv:], q_early[..., kv:])
    assert ok.all(), f"{int((~ok).sum())} identity-rope q channels differ beyond subnormal flush / signed zero"
    for name, a, b in zip(("output", "kvpe", "index-K"), late, early):
        a, b = a.float(), b.float()
        max_diff = float((a - b).abs().max())
        logger.info(f"[{variant.name} tp={tp}] early vs late {name}: max |diff| {max_diff}")
        assert torch.equal(a, b), f"{name}: early-Q path differs from the late path (max |diff| {max_diff})"


def _tp_reshard_cases():
    """GLM at TP=2 and DeepSeek-V3.2 at both TPs: the head shards sparse_sdpa can also run without the reshard. TP=2
    also runs 2560, whose 640 rows/chip hit the tuned wkv_b1 slot."""
    out = []
    for c in SPARSE_CHUNKED_CASES:
        variant, mesh = c.values[0], c.values[1]
        if (variant == "glm_5_2" and mesh[1] == 2) or variant == "deepseek_v32":
            chunks = [_early_q_chunk(mesh)] + ([2560] if mesh[1] == 2 and 2560 // mesh[0] == 640 else [])
            out += [pytest.param(*c.values, ch, marks=c.marks, id=f"{c.id}-c{ch}") for ch in chunks]
    return out


@pytest.mark.parametrize(
    "variant, mesh_device, seq_len, device_params, chunk",
    _tp_reshard_cases(),
    indirect=["variant", "mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_sparse_mla_tp_reshard_matches_head_sharded(
    mesh_device, seq_len, chunk, device_params, variant, config_only, monkeypatch
):
    """The head -> sequence reshard against the head-sharded layout sparse_sdpa can also run for these models (forced
    here; production always reshards at TP > 1), on the same weights and input.
    sparse_sdpa's math per (token, head) does not depend on how many heads share a chip, and the caches not on the
    q layout at all, so all three match bit for bit."""
    config = config_only
    config.max_seq_len = chunk
    weights, _ = build_weights(variant, config, seed=42)
    hidden = make_hidden(chunk, config.hidden_size, seed=3)
    cache_format = SPARSE_MAIN_KV_FORMAT[variant.name]
    _, resharded = _single_chunk_forward(mesh_device, config, weights, hidden, chunk, cache_format, early_q=True)
    monkeypatch.setattr(ttMLA, "reshards_heads_to_seq", staticmethod(lambda tp_factor: False))
    _, head_sharded = _single_chunk_forward(mesh_device, config, weights, hidden, chunk, cache_format, early_q=False)
    tp = mesh_device.shape[1]
    for name, a, b in zip(("output", "kvpe", "index-K"), head_sharded, resharded):
        a, b = a.float(), b.float()
        max_diff = float((a - b).abs().max())
        logger.info(f"[{variant.name} tp={tp}] resharded vs head-sharded {name}: max |diff| {max_diff}")
        assert torch.equal(a, b), f"{name} depends on the q layout (max |diff| {max_diff})"
