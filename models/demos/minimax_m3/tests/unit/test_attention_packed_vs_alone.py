# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed forward through the ATTENTION LAYER, SP=8 x TP=4, dense + sparse: one forward of 4 x 1024-token chunk
chunks must match running each chunk alone, in order, against an identical cache.

Each composition first fills the slots' prefixes (single-chunk calls, identically in both caches), then runs the
packed forward in one cache and the chunks one by one in the other. The chunks' output rows and the K / V /
index_k caches must agree. Compositions: mixed depths, consecutive chunks of one slot, a partial last chunk,
interleaved partial chunks, an empty tail (its slot's bytes stay unchanged) and two consecutive mid-slab chunks
of one slot. Random weights, the runtime's whole-cache indexed RoPE.
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.common.utils import block_cyclic_reorder
from models.demos.common.prefill.chunk_layout import pack_forward_tokens
from models.demos.common.prefill.prefill_chunk import PrefillChunk, validate_chunks
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.attention import Attention, AttentionConfig, allocate_kv_caches, msa, prefill
from models.demos.minimax_m3.tt.attention_configs import MiniMaxM3AttentionProgramConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.model import create_rope_setup
from models.demos.minimax_m3.utils.general_utils import get_default_num_links
from models.demos.minimax_m3.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial

from ..test_factory import compose_tp_hidden, parametrize_mesh_with_fabric

HIDDEN, NQ, NKV, HEAD_DIM, ROTARY_DIM, THETA, EPS = 6144, 64, 4, 128, 64, 5_000_000.0, 1e-6
NIDX, INDEX_DIM = 4, 128
C, F, USERS = 1024, 4, 4
CAP = 12 * C
PCC = 0.998


def prefix_fills(start, end):
    """Single-chunk prefix fills covering [start, end) in chunk-sized steps."""
    return [(a, min(a + C, end)) for a in range(start, end, C)]


# name -> (prefix fills per slot, the forward's chunks)
CASES = {
    "mixed_depths": (
        {1: prefix_fills(0, 2 * C), 2: prefix_fills(0, 8 * C), 3: prefix_fills(0, 4 * C)},
        [
            PrefillChunk(0, 0, C),
            PrefillChunk(1, 2 * C, 3 * C),
            PrefillChunk(2, 8 * C, 9 * C),
            PrefillChunk(3, 4 * C, 5 * C),
        ],
    ),
    "same_slot_run": (
        {1: prefix_fills(0, C)},
        [PrefillChunk(0, 0, C), PrefillChunk(0, C, 2 * C), PrefillChunk(0, 2 * C, 3 * C), PrefillChunk(1, C, 2 * C)],
    ),
    "partial_last": (
        {0: prefix_fills(0, 3 * C), 2: prefix_fills(0, 2 * C)},
        [
            PrefillChunk(1, 0, C),
            PrefillChunk(2, 2 * C, 3 * C),
            PrefillChunk(0, 3 * C, 4 * C),
            PrefillChunk(0, 4 * C, 4 * C + 300),
        ],
    ),
    "interleaved_partials": (
        {0: prefix_fills(0, C)},
        [PrefillChunk(0, C, C + 700), PrefillChunk(1, 0, C), PrefillChunk(1, C, C + 333), PrefillChunk(2, 0, 300)],
    ),
    "empty_tail": (
        {0: prefix_fills(0, 2 * C), 3: prefix_fills(0, 2 * C)},
        [PrefillChunk(0, 2 * C, 3 * C), PrefillChunk(1, 0, 600), PrefillChunk(3, 0, 0), PrefillChunk(3, 0, 0)],
    ),
    "mid_slab": (
        {0: prefix_fills(0, 2 * C + 96), 1: prefix_fills(0, C + 512)},
        [
            PrefillChunk(0, 2 * C + 96, 3 * C + 96),
            PrefillChunk(0, 3 * C + 96, 4 * C + 96),
            PrefillChunk(1, C + 512, 2 * C + 512),
            PrefillChunk(2, 0, C),
        ],
    ),
}


def _attention_fixture(mesh_device, is_sparse, global_layer_idx=0):
    """A random-weight attention layer on the 8x4 mesh with the runtime's whole-cache indexed RoPE, and two helpers:
    ``run(x_chunks, chunks, kvc)`` (one forward over ``chunks``, natural-order [C, H] rows per chunk -> output rows per
    chunk) and ``read_cache(kvc)`` (the three caches as host tensors, cache-row-major)."""
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, s = rows, C // rows
    torch.manual_seed(0)
    hf_state = {
        "q_proj.weight": torch.rand(NQ * HEAD_DIM, HIDDEN) * 0.02,
        "k_proj.weight": torch.rand(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "v_proj.weight": torch.rand(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "o_proj.weight": torch.rand(HIDDEN, NQ * HEAD_DIM) * 0.02,
        "q_norm.weight": torch.randn(HEAD_DIM) * 0.1,
        "k_norm.weight": torch.randn(HEAD_DIM) * 0.1,
    }
    if is_sparse:
        hf_state |= {
            "index_q_proj.weight": torch.rand(NIDX * INDEX_DIM, HIDDEN) * 0.02,
            "index_k_proj.weight": torch.rand(INDEX_DIM, HIDDEN) * 0.02,
            "index_q_norm.weight": torch.randn(INDEX_DIM) * 0.1,
            "index_k_norm.weight": torch.randn(INDEX_DIM) * 0.1,
        }
    state = convert_hf_qkv_to_meta_format_partial(hf_state, HEAD_DIM, ROTARY_DIM)

    mesh_config = MeshConfig((rows, cols), tp=cols)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    hf_config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=NQ,
        num_key_value_heads=NKV,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        rope_theta=THETA,
        rope_scaling=None,
        rms_norm_eps=EPS,
        max_position_embeddings=CAP,
        use_qk_norm=True,
        use_gemma_norm=True,
    )
    rope_setup = create_rope_setup(mesh_device=mesh_device, hf_config=hf_config, datatype=ttnn.bfloat16)
    attn = Attention(
        mesh_device=mesh_device,
        config=AttentionConfig(
            hidden_size=HIDDEN,
            num_heads=NQ,
            num_kv_heads=NKV,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            rms_norm_eps=EPS,
            use_qk_norm=True,
            use_gemma_norm=True,
            max_seq_len=CAP,
            max_local_batch_size=1,
            is_sparse=is_sparse,
            sequence_parallel=True,
        ),
        state_dict=state,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        program_config=MiniMaxM3AttentionProgramConfig(),
        global_layer_idx=global_layer_idx,
        transformation_mats=rope_setup.get_both_trans_mats(),
    )
    seq_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None))

    def indexed_rope(dev_mat):  # as TtPrefillRuntime._build_indexed_rope
        full = ttnn.to_torch(ttnn.get_device_tensors(dev_mat)[0])[:, :, :CAP, :]
        bc = block_cyclic_reorder(full, s, sp, seq_dim=2)
        return ttnn.from_torch(
            bc, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=seq_mapper
        )

    rope = [indexed_rope(rope_setup.cos_matrix_prefill), indexed_rope(rope_setup.sin_matrix_prefill)]

    def run(x_chunks, chunks, kvc):
        """One attention forward over ``chunks``; x_chunks are natural-order [C, H] rows per chunk. Returns
        the output rows per chunk, natural order."""
        idx = pack_forward_tokens(
            [[k * C + i for i in range(C)] for k in range(len(chunks))], [e.actual_start for e in chunks], sp
        )
        x = torch.cat(x_chunks)[idx].reshape(1, 1, -1, HIDDEN)
        x_tt = ttnn.from_torch(
            x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=seq_mapper
        )
        out = attn(x_tt, rope_mats=rope, kv_cache=kvc, indexed_rope=True, chunks=chunks)
        dts = ttnn.get_device_tensors(out)
        per_chip = [compose_tp_hidden(dts, r, cols).reshape(len(chunks), s, HIDDEN) for r in range(rows)]
        ttnn.deallocate(out)
        return [torch.cat([chip[k] for chip in per_chip]) for k in range(len(chunks))]  # chip-major rows

    def read_cache(kvc):
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
        out = {}
        for name in ("k", "v", "index_k"):
            t = getattr(kvc, name)
            sl = ttnn.slice(t, [0, 0, 0, 0], list(t.shape), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out[name] = ttnn.to_torch(sl, mesh_composer=composer).float()
            ttnn.deallocate(sl)
        return out

    return attn, run, read_cache


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("layer_kind", ["dense", "sparse"])
def test_attention_packed_vs_alone(mesh_device, device_params, layer_kind, reset_seeds):
    is_sparse = layer_kind == "sparse"
    attn, run, read_cache = _attention_fixture(mesh_device, is_sparse)
    gen = torch.Generator().manual_seed(1)
    rand_rows = lambda: torch.randn(C, HIDDEN, generator=gen) * 0.1
    failures, worst = [], 1.0
    for name, (prefixes, chunks) in CASES.items():
        validate_chunks(chunks, chunk_size=C, forward_chunks=F, num_users=USERS, max_seq_len=CAP)
        packed_kv = allocate_kv_caches(mesh_device, num_layers=1, max_seq_len=CAP, num_users=USERS)
        alone_kv = allocate_kv_caches(mesh_device, num_layers=1, max_seq_len=CAP, num_users=USERS)
        for slot, fills in prefixes.items():
            for a, b in fills:
                x = rand_rows()
                for kvc in (packed_kv, alone_kv):
                    run([x], [PrefillChunk(slot, a, b)], kvc)
        ttnn.synchronize_device(mesh_device)
        before = read_cache(packed_kv)

        xs = [rand_rows() for _ in chunks]
        packed = run(xs, chunks, packed_kv)
        alone = [None if e.is_empty else run([x], [e], alone_kv)[0] for x, e in zip(xs, chunks)]
        ttnn.synchronize_device(mesh_device)
        after, ref = read_cache(packed_kv), read_cache(alone_kv)

        for k, e in enumerate(chunks):
            if e.is_empty:
                ok = bool(torch.all(packed[k] == 0))
                logger.info(f"[{layer_kind}/{name}] chunk {k} empty: zero rows {ok}")
                if not ok:
                    failures.append(f"{name} chunk {k}: empty chunk rows are not zero")
                continue
            _, pcc = comp_pcc(alone[k], packed[k], PCC)
            worst = min(worst, pcc)
            logger.info(f"[{layer_kind}/{name}] chunk {k} {e}: out pcc={pcc:.6f}")
            if pcc < PCC:
                failures.append(f"{name} chunk {k}: out pcc {pcc:.6f}")
        names = ("k", "v", "index_k") if is_sparse else ("k", "v")
        for cache in names:
            _, pcc = comp_pcc(ref[cache], after[cache], PCC)
            diff = (ref[cache] - after[cache]).abs().max().item()
            worst = min(worst, pcc)
            mean = ref[cache].abs().mean().item()
            logger.info(f"[{layer_kind}/{name}] cache {cache}: pcc={pcc:.6f} max|diff|={diff:.4g} mean|ref|={mean:.4g}")
            if pcc < PCC:
                failures.append(f"{name} cache {cache}: pcc {pcc:.6f}")
        for slot in {e.slot_id for e in chunks if e.is_empty} - {e.slot_id for e in chunks if not e.is_empty}:
            for cache in names:
                if not torch.equal(before[cache][slot], after[cache][slot]):
                    failures.append(f"{name}: empty chunk wrote slot {slot} ({cache})")
        packed_kv.deallocate()
        alone_kv.deallocate()

    logger.info(f"[{layer_kind}] worst pcc={worst:.6f}")
    assert not failures, f"{layer_kind} packed vs alone: {failures}"


@pytest.mark.timeout(1800)
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("index_k_tp_shard", [False, True], ids=["index_k_replicated", "index_k_tp_deduped"])
def test_attention_packed_overlap_vs_serial(mesh_device, device_params, index_k_tp_shard, reset_seeds, monkeypatch):
    """The sparse packed forward with the MSA gather/compute overlap (msa_chunks_pipelined) against the same forward
    with the overlap off: outputs and caches bit for bit. Two cache layers and the attention on the second, so a
    slot-for-row mix-up in the pipelined gather lands on layer 0's untouched row instead of passing. With a
    TP-deduped index_k the window carries its two-leg rebuild as well."""
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    layers, layer = 2, 1
    attn, run, read_cache = _attention_fixture(mesh_device, True, global_layer_idx=layer)
    prefixes, chunks = CASES["mixed_depths"]  # three warm chunks and one cold
    gen = torch.Generator().manual_seed(1)
    rand_rows = lambda: torch.randn(C, HIDDEN, generator=gen) * 0.1
    fills = {slot: [(a, b, rand_rows()) for a, b in f] for slot, f in prefixes.items()}
    xs = [rand_rows() for _ in chunks]
    calls = []  # prefill.py binds the name at import: the counter wraps its reference, not msa's
    pipelined = prefill.msa_chunks_pipelined
    monkeypatch.setattr(prefill, "msa_chunks_pipelined", lambda cs, **kw: calls.append(len(cs)) or pipelined(cs, **kw))
    results = {}
    for overlap in (True, False):
        monkeypatch.setattr(msa, "MSA_CHUNK_OVERLAP", overlap)
        kvc = allocate_kv_caches(
            mesh_device, num_layers=layers, max_seq_len=CAP, num_users=USERS, index_k_tp_shard=index_k_tp_shard
        )
        try:
            for slot, f in fills.items():
                for a, b, x in f:
                    run([x], [PrefillChunk(slot, a, b)], kvc)
            calls.clear()
            out = run(xs, chunks, kvc)
            ttnn.synchronize_device(mesh_device)
            results[overlap] = (out, read_cache(kvc), list(calls))
        finally:
            kvc.deallocate()
    (on, on_cache, on_calls), (off, off_cache, off_calls) = results[True], results[False]
    n_warm = sum(1 for c in chunks if c.actual_start > 0)
    assert on_calls == [n_warm], f"overlap on: pipelined calls {on_calls}, expected one over {n_warm} warm chunks"
    assert off_calls == [], f"overlap off took the pipelined path: {off_calls}"
    for k, (a, b) in enumerate(zip(on, off)):
        assert torch.equal(a, b), f"chunk {k} output differs with the overlap on"
    for cache in ("k", "v", "index_k"):
        for c in chunks:
            row = layers * c.slot_id + layer
            assert torch.equal(
                on_cache[cache][row], off_cache[cache][row]
            ), f"{cache} row {row} differs with the overlap on"
