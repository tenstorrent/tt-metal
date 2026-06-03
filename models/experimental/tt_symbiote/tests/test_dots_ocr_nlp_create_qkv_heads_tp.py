# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP-support unit test for ``NlpCreateHeadsDeviceOperation`` on the dots.ocr shapes.

The perf sheet (perf_full.txt) contains two distinct ``NlpCreateHeadsDeviceOperation``
shapes. Each one is identified from the QKV-projection matmul dims (M × K × N) that
immediately precede it: M = seq_len, K = hidden = 1536, N = fused qkv width.

    subsystem        QKV matmul (M×K×N)   heads (Q, KV)  head_dim  width  seq_len  dtype  perf IDs
    vision  (MHA)    11264 × 1536 × 4608  (12, 12)       128       4608   11264    BFP8   42 rows 19953..25201
    text prefill     2816  × 1536 × 2048  (12,  2)       128       2048    2816    BF16   28 rows 25466..31082
    text decode      32    × 1536 × 2048  (12,  2)       128       2048      32    BF16   28 rows 31683..39243

(Vision is MHA — num_kv_heads == num_heads == 12, see dots_ocr_vision.py:1331-1333,1595.
The text decoder is GQA — 12 Q / 2 KV, see dots_ocr_attention.py:443. Both call the op with
``transpose_k_heads=False``.)

"Does it support TP4/TP2/TP1" depends on what tensor parallelism does to the create_heads
*input*; both schemes are tested:

1. K-parallel TP (the dots.ocr decoder's actual scheme): TP shards the hidden/contraction
   dim of the QKV matmul and all-reduces, so after the reduce every device holds the full
   head set regardless of TP degree. create_heads sees the identical input at TP1/2/4, so all
   are supported. Covered by ``test_..._kparallel_full_heads`` (head count is TP-independent).

2. Head-sharded TP (each device owns num_heads/tp Q and num_kv_heads/tp KV heads):
       vision (12,12): TP1(12,12), TP2(6,6), TP4(3,3) — all split evenly        ✓ all supported
       text   (12, 2): TP1(12, 2), TP2(6,1) ✓ ;  TP4 -> (3, 0) — only 2 KV heads ✗ KV dropped
   For the text shapes, head-sharded TP4 has no valid per-device shard (2 KV heads cannot be
   partitioned across 4 devices); the op does not reject it, it silently returns K/V with 0
   heads. Covered by ``test_..._head_sharded``.

Single device, DRAM in/out: this validates the per-device shape + correctness (PCC) each TP
degree would produce. It does NOT exercise real cross-device sharding/all-reduce, nor the L1
residency the model uses (L1 fit is a separate memory concern, not a shape/support question).

Run: pytest models/experimental/tt_symbiote/tests/test_dots_ocr_nlp_create_qkv_heads_tp.py -s
"""
from __future__ import annotations

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

HEAD_DIM = 128  # hidden_size 1536 / 12 heads, shared by vision and text

# One entry per distinct NlpCreateHeadsDeviceOperation shape in perf_full.txt.
# (name, num_q_heads, num_kv_heads, seq_len, dtype)
PERF_SHAPES = [
    ("vision", 12, 12, 11264, ttnn.bfloat8_b),
    ("text_prefill", 12, 2, 2816, ttnn.bfloat16),
    ("text_decode", 12, 2, 32, ttnn.bfloat16),
]
PERF_SHAPE_IDS = [c[0] for c in PERF_SHAPES]


def _create_heads(seq_len, q_heads, kv_heads, dtype, device):
    """Run nlp_create_qkv_heads on one (per-device) QKV shard. Returns (q, k, v, host_input)."""
    width = (q_heads + 2 * kv_heads) * HEAD_DIM
    A = torch.randn([1, 1, seq_len, width])
    in0_t = ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, ttnn.DRAM_MEMORY_CONFIG)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        None,
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return q, k, v, A


def _assert_split_matches_torch(seq_len, q_heads, kv_heads, dtype, q, k, v, A):
    """PCC-check the device Q/K/V split against the equivalent torch split (no K transpose)."""
    assert list(q.padded_shape) == [1, q_heads, seq_len, HEAD_DIM]
    assert list(k.padded_shape) == [1, kv_heads, seq_len, HEAD_DIM]
    assert list(v.padded_shape) == [1, kv_heads, seq_len, HEAD_DIM]

    ref_q, ref_k, ref_v = torch.split(A, [q_heads * HEAD_DIM, kv_heads * HEAD_DIM, kv_heads * HEAD_DIM], dim=-1)
    ref_q = torch.reshape(ref_q, [1, seq_len, q_heads, HEAD_DIM]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [1, seq_len, kv_heads, HEAD_DIM]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [1, seq_len, kv_heads, HEAD_DIM]).transpose(-3, -2)

    pcc = 0.99 if dtype == ttnn.bfloat8_b else 0.9999
    for name, got, ref in (("Q", q, ref_q), ("K", k, ref_k), ("V", v, ref_v)):
        passing, value = comp_pcc(ref, ttnn.to_torch(got), pcc)
        assert passing, f"{name} PCC below {pcc}: {value}"


@pytest.mark.parametrize("name,num_q,num_kv,seq_len,dtype", PERF_SHAPES, ids=PERF_SHAPE_IDS)
@pytest.mark.parametrize("tp", (1, 2, 4), ids=["TP1", "TP2", "TP4"])
def test_dots_ocr_create_heads_kparallel_full_heads(tp, name, num_q, num_kv, seq_len, dtype, device):
    """K-parallel TP: create_heads sees the full head set at every TP degree.

    The all-reduce restores the full QKV on each device, so the head count is independent of
    ``tp`` — the op gets the identical input. Asserting correctness for each ``tp`` documents
    that TP1/TP2/TP4 are all supported under the dots.ocr K-parallel TP scheme.
    """
    q, k, v, A = _create_heads(seq_len, num_q, num_kv, dtype, device)
    _assert_split_matches_torch(seq_len, num_q, num_kv, dtype, q, k, v, A)


@pytest.mark.parametrize("name,num_q,num_kv,seq_len,dtype", PERF_SHAPES, ids=PERF_SHAPE_IDS)
@pytest.mark.parametrize("tp", (1, 2, 4), ids=["TP1", "TP2", "TP4"])
def test_dots_ocr_create_heads_head_sharded(tp, name, num_q, num_kv, seq_len, dtype, device):
    """Head-sharded TP: each device owns num_q/tp Q heads and num_kv/tp KV heads.

    Splits that yield whole heads on every device are PCC-checked. The text shapes at TP4 have
    only 2 KV heads, so kv_heads becomes 0 and the op silently returns K/V with 0 heads (KV
    dropped) — recording that head-sharded TP4 is unsupported for the GQA text decoder, while
    the vision MHA shape (12,12) splits cleanly and is supported at TP4.
    """
    q_heads = num_q // tp
    kv_heads = num_kv // tp

    if num_q % tp == 0 and num_kv % tp == 0:
        q, k, v, A = _create_heads(seq_len, q_heads, kv_heads, dtype, device)
        _assert_split_matches_torch(seq_len, q_heads, kv_heads, dtype, q, k, v, A)
    else:
        # text @ TP4: 2 KV heads cannot be sharded 4 ways -> kv_heads == 0.
        assert kv_heads == 0
        q, k, v, _ = _create_heads(seq_len, q_heads, kv_heads, dtype, device)
        assert q.padded_shape[1] == q_heads
        assert k.padded_shape[1] == 0, "KV must be empty: 2 KV heads cannot be sharded 4 ways"
        assert v.padded_shape[1] == 0


@pytest.mark.parametrize("mesh_device", [4], indirect=True)
@pytest.mark.parametrize("name,num_q,num_kv,seq_len,dtype", PERF_SHAPES, ids=PERF_SHAPE_IDS)
def test_dots_ocr_create_heads_tp4_mesh_qshard_kvreplicated(name, num_q, num_kv, seq_len, dtype, mesh_device):
    """Real-mesh TP4 (n150x4): Q heads sharded across devices, KV heads replicated on each.

    This is the layout to adapt the model to for TP4: each device's fused QKV holds its 1/TP
    slice of Q heads plus the full (replicated) K/V heads. KV replication is what keeps the GQA
    text decoder (only 2 KV heads) viable at TP4 — it sidesteps the "2 KV heads can't split 4
    ways" problem that head-sharding hits. Vision (MHA) works the same way.

    Build a host tensor stacked on dim0 = device axis, where slab d = [Q_shard_d | K_all | V_all],
    distribute it with ShardTensorToMesh(dim=0), run create_heads on the mesh, gather with
    ConcatMeshToTensor(dim=0), and PCC each device's (Q shard, full K, full V) against the
    single-device split. Auto-skips if fewer than 4 devices are present.
    """
    tp = mesh_device.get_num_devices()
    assert num_q % tp == 0, f"{num_q} Q heads not divisible by TP={tp}"
    q_per_dev = num_q // tp

    torch.manual_seed(1234)
    q_w, kv_w = num_q * HEAD_DIM, num_kv * HEAD_DIM
    A = torch.randn([1, 1, seq_len, q_w + 2 * kv_w])  # model fused layout [Q_all | K_all | V_all]
    Q_full, K_full, V_full = torch.split(A, [q_w, kv_w, kv_w], dim=-1)

    # Per-device fused QKV = [Q_shard_d | K_all | V_all]; stack on dim0 for ShardTensorToMesh.
    slabs = [
        torch.cat(
            [Q_full[..., d * q_per_dev * HEAD_DIM : (d + 1) * q_per_dev * HEAD_DIM], K_full, V_full],
            dim=-1,
        )
        for d in range(tp)
    ]
    host = torch.cat(slabs, dim=0)  # [tp, 1, S, (q_per_dev + 2*num_kv)*HEAD_DIM]

    in0 = ttnn.from_torch(
        host,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0,
        None,
        num_heads=q_per_dev,
        num_kv_heads=num_kv,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    q_t = ttnn.to_torch(q, mesh_composer=composer)  # [tp, q_per_dev, S, HEAD_DIM]
    k_t = ttnn.to_torch(k, mesh_composer=composer)  # [tp, num_kv,   S, HEAD_DIM]
    v_t = ttnn.to_torch(v, mesh_composer=composer)

    ref_q = torch.reshape(Q_full, [1, seq_len, num_q, HEAD_DIM]).transpose(-3, -2)[0]  # [num_q,  S, HD]
    ref_k = torch.reshape(K_full, [1, seq_len, num_kv, HEAD_DIM]).transpose(-3, -2)[0]  # [num_kv, S, HD]
    ref_v = torch.reshape(V_full, [1, seq_len, num_kv, HEAD_DIM]).transpose(-3, -2)[0]

    pcc = 0.99 if dtype == ttnn.bfloat8_b else 0.9999
    for d in range(tp):
        # Q: device d owns heads [d*q_per_dev : (d+1)*q_per_dev].
        passing, val = comp_pcc(ref_q[d * q_per_dev : (d + 1) * q_per_dev], q_t[d], pcc)
        assert passing, f"device {d} Q PCC below {pcc}: {val}"
        # K / V: replicated -> every device must hold the full set.
        passing, val = comp_pcc(ref_k, k_t[d], pcc)
        assert passing, f"device {d} K PCC below {pcc}: {val}"
        passing, val = comp_pcc(ref_v, v_t[d], pcc)
        assert passing, f"device {d} V PCC below {pcc}: {val}"
