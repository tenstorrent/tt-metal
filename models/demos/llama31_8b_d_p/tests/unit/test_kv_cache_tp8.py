# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The **model -> cache** path at TP=8. Gate: `G-KV-TP8`.

**What this closes.** `G-KV` (P5.6) proved the cache *primitive* on a `(1,1)` mesh, where TP=1 and
the model emits **8** local KV heads into a slot that holds **one** — a shape the deployment mesh
never produces and the write op rejects outright. So `G-KV` drove the op one synthetic head at a
time and `07_RISKS.md` R-001 carried the gap: nothing had ever proved that the *model's* K and V,
sharded by the weight loader's `column_parallel` mapper, land on the chips the cache expects. That
is this file (`BRINGUP_RECIPE.md:1773-1777`, `:1735-1741`).

**Why `(1, 8)` and not `(4, 8)`.** At `sp = 1` the block-cyclic sequence layout is the **identity**,
so the only thing under test is the head/feature distribution and a failure can only be the mapper
(`BRINGUP_RECIPE.md:1773-1775`). The block-cyclic reorder is `G-MESH-KV`'s at SP=4. The `(1,8)` mesh
is a **submesh** of the full galaxy, never a top-level open (`G-FABRIC-MATRIX`).

**The mapping is gated on bit-equality, never PCC** (recipe §2.5). The rotated-column control below
— reading mesh column `(c+1) % 8` as KV head `c`, a completely wrong head->column map — still
scores a high PCC by construction, because half of every probe lane block carries a
head-*independent* position label. Correlation does not notice a permutation that preserves most of
the distribution. `torch.equal` does.

## The two arms

| arm | input distribution | what is exact | what it proves |
|---|---|---|---|
| **A — head->column** | a synthetic **labelled** probe: lane block `[0,64)` carries the position `s % 128`, lane block `[64,128)` carries the head id `c+1`; both exactly representable in `bfloat8_b` (ceiling **128**, `R-013`) | `torch.equal` | the loader's `column_parallel` mapper puts global KV head `c` on mesh column `c`, through the real projection -> head split -> (RoPE) -> `write_kv_chunk` path |
| **B — vs the fp32 golden** | the golden trace's own `token_ids`, real checkpoint weights, 32 layers | PCC, thresholds **carried** from `G-CHUNK` | the same path produces the *right numbers*, not merely the right addresses |

Arm A is exact and arm B is not, and **B is only meaningful given A**: B stacks mesh column `c` as
head `c` to build the `[1, 8, S, 128]` tensor it scores, which is the very claim A establishes.

**Why arm B carries `G-CHUNK`'s thresholds rather than picking fresh ones.** A threshold chosen
here would be fitted to this measurement and could not fail; carrying P7's makes the TP split's cost
readable directly (`BRINGUP_RECIPE.md:1802-1804`).

**RoPE and the labelled probe.** Arm A runs **V** with RoPE on — V is never rotated, so its lanes
survive the full path — and **K** with `transformation_mats=None`, because RoPE is
position-dependent and destroys any bit-exact label by construction (`DEC-078`). The same
`column_parallel` mapper places both weights, on the same axis with the same out-dim geometry, so
the two probes together pin the mapping for K and V; K's *post-RoPE* values are what arm B and
`G-CHUNK-ATTN` score.

Run:
    export PREFILL_TRACE_DIR=/home/mstojkovic/prefill_traces/llama31_8b_d_p/s512
    HF_MODEL=... pytest models/demos/llama31_8b_d_p/tests/unit/test_kv_cache_tp8.py -x -q
"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss_d_p.utils.general_utils import get_default_num_links
from models.demos.llama31_8b_d_p.tests.test_factory import (  # noqa: F401 — submesh_pool is a fixture
    GALAXY_MESH_SHAPE,
    err_ratio,
    galaxy_device_params,
    llama_config_dims,
    prefill_topology,
    quantize_like_device,
    requires_galaxy,
    requires_hf_reference,
    requires_ring_fabric,
    submesh_pool,
)
from models.demos.llama31_8b_d_p.tests.unit.test_attention_chunked_vs_ref import (
    _golden_layer,
    _golden_metadata,
    _layer0_floors,
    _to_meta,
    requires_golden_trace,
)
from models.demos.llama31_8b_d_p.tests.unit.test_model_vs_ref import _load_checkpoint
from models.demos.llama31_8b_d_p.tt.attention import Attention, ProgramConfig, allocate_kv_cache, write_kv_chunk
from models.demos.llama31_8b_d_p.tt.attention.operations import apply_qkv_projection, split_qkv_heads_prefill
from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
from models.demos.llama31_8b_d_p.tt.config import MeshConfig, derive_head_dim
from models.demos.llama31_8b_d_p.tt.layer import build_attention_config
from models.demos.llama31_8b_d_p.tt.model import Model
from models.demos.llama31_8b_d_p.tt.rope import build_prefill_rope, build_transformation_mat

TP_MESH_SHAPE = (1, 8)
WEIGHT_DTYPE = ttnn.bfloat8_b  # `DEC-022`
ACTIVATION_DTYPE = ttnn.bfloat16  # `DEC-022`
CACHE_DTYPE = ttnn.bfloat8_b  # `DEC-021`

# Carried from `G-CHUNK` verbatim (`BRINGUP_RECIPE.md:1802-1804`).
GOLDEN_K_THRESHOLD = 0.99
GOLDEN_V_THRESHOLD = 0.98
MAX_L0_ERR_RATIO = 3.0

# `bfloat8_b`'s exact-integer ceiling on this box, measured at P5.6: the first inexact integer is
# **129**, not the recipe's blanket 257 (`07_RISKS.md` R-013, `DEC-044`). Every probe payload below
# is `< 128`, so a failing probe cannot be the probe's own numerics.
BF8_EXACT_INTEGER_CEILING = 128

_RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bringup_log", "raw"
)


# ---------------------------------------------------------------------------------------------
# arm A — the labelled probe
# ---------------------------------------------------------------------------------------------
def _labelled_projection_weight(hf, head_dim, *, uniform: bool):
    """An HF-layout `[num_kv_heads*head_dim, hidden]` weight that labels each head.

    Paired with `_labelled_input`, it makes the projection output **exact**:

    * `uniform=False` (the V probe): out lane `c*head_dim + j` is the position label for
      `j < head_dim/2` and the head id `c + 1` for `j >= head_dim/2`. The split is at lane 64, which
      is a multiple of `bfloat8_b`'s 16-element exponent block, so every block is homogeneous and
      the shared exponent is exact.
    * `uniform=True` (the K probe): **every** lane of head `c` is `c + 1`, which is invariant under
      the Meta `reverse_permute` the loader applies to `k_proj` — the permutation moves rows only
      *within* a head (`models/tt_transformers/tt/load_checkpoints.py:891`), so a head-uniform
      weight comes out unchanged and the label survives the swizzle.
    """
    n_kv = hf["num_key_value_heads"]
    hidden = hf["hidden_size"]
    weight = torch.zeros(n_kv * head_dim, hidden)
    for c in range(n_kv):
        for j in range(head_dim):
            if uniform or j >= head_dim // 2:
                weight[c * head_dim + j, 1] = float(c + 1)  # reads input feature 1, which is 1.0
            else:
                weight[c * head_dim + j, 0] = 1.0  # reads input feature 0, the position label
    return weight


def _labelled_input(hf, seq_len, *, start_pos=0):
    """`[1, 1, S, hidden]` whose feature 0 is the position label and feature 1 is exactly 1.0."""
    x = torch.zeros(1, 1, seq_len, hf["hidden_size"])
    for s in range(seq_len):
        x[0, 0, s, 0] = float((start_pos + s) % BF8_EXACT_INTEGER_CEILING)
        x[0, 0, s, 1] = 1.0
    return x


def _expected_head(head_dim, seq_len, head, *, start_pos, uniform):
    """The exact `[seq_len, head_dim]` block mesh column `head` must hold. Host-side, no device."""
    out = torch.empty(seq_len, head_dim)
    for s in range(seq_len):
        pos = float((start_pos + s) % BF8_EXACT_INTEGER_CEILING)
        for j in range(head_dim):
            out[s, j] = float(head + 1) if (uniform or j >= head_dim // 2) else pos
    return out


def _build_probe_attention(sub, hf, weights_hf, *, seq_len, max_seq_len, with_rope, mesh_config, ccl_manager):
    head_dim = derive_head_dim(hf)
    hidden = hf["hidden_size"]
    state_dict = {
        "q_proj.weight": torch.zeros(hf["num_attention_heads"] * head_dim, hidden),
        "k_proj.weight": weights_hf["k"],
        "v_proj.weight": weights_hf["v"],
        "o_proj.weight": torch.zeros(hidden, hf["num_attention_heads"] * head_dim),
    }
    config = build_attention_config(hf, max_seq_len=max_seq_len)
    return Attention(
        sub,
        config,
        state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=ProgramConfig(),
        layer_idx=0,
        transformation_mats={"prefill": build_transformation_mat(sub)} if with_rope else None,
        weight_dtype=WEIGHT_DTYPE,
    )


def _read_cache_column(cache_tensor, column, *, layer_idx, seq_len):
    """Mesh column `column`'s own cache rows `[0, seq_len)` for `layer_idx`, fp32."""
    per_device = ttnn.get_device_tensors(cache_tensor)
    host = ttnn.to_torch(per_device[column]).float()  # [slots, 1, seq_local, head_dim]
    return host[layer_idx, 0, :seq_len, :]


@requires_galaxy
@requires_ring_fabric
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("seq_len", [128], ids=lambda s: f"s{s}")
@pytest.mark.parametrize("probe", ["v_with_rope", "k_without_rope"])
def test_head_to_mesh_column_is_bit_exact(submesh_pool, seq_len, probe, reset_seeds):
    """**Arm A.** Global KV head `c` must land on mesh column `c`, `rtol = atol = 0`.

    Driven through the real `Attention.__call__` — projection, GQA head split, optional RoPE,
    `write_kv_chunk` — at TP=8 on a `(1, 8)` submesh, so what is proved is the **model -> cache**
    path and not a re-implementation of it.

    * **Input distribution:** a synthetic labelled probe (see the module docstring), *not* a random
      one. A random input cannot make an address claim exact, and §2.5 requires this claim to be
      exact. Every payload is `< 128`, `bfloat8_b`'s measured exact-integer ceiling (`R-013`).
    * **Reference dtype policy:** none needed — the expected tensor is constructed on the host from
      integers that `bfloat16` (the activation dtype) and `bfloat8_b` (the cache dtype) both hold
      exactly, so the comparison is bit-equality rather than a floor.
    * **Noise floor:** not applicable to a bit-exact claim; the floor is 1.0 by construction and
      the assertion is `torch.equal`.
    * **Negative control:** `test_rotated_column_control_passes_pcc_and_fails_bit_equality`.
    """
    hf = llama_config_dims()
    head_dim, n_kv = derive_head_dim(hf), hf["num_key_value_heads"]
    uniform = probe.startswith("k")
    with_rope = probe.endswith("with_rope")
    max_seq_len = 512
    # Chunk 0 only. `Attention.__call__` at `cached_len > 0` is delta 3 and the dense core refuses
    # it — correctly, and at `(1, 8)` there is no SP axis to run the ring core on. The **write
    # offset** at TP=8 is the next test's, which drives the same projection -> split -> write path
    # without an attention core (`DEC-080`); `G-KV` already proved four offsets bit-exactly at (1,1).
    kv_actual = 0

    with submesh_pool.use(TP_MESH_SHAPE) as sub:
        assert tuple(sub.shape) == TP_MESH_SHAPE
        mesh_config = MeshConfig(TP_MESH_SHAPE, tp=TP_MESH_SHAPE[1])
        assert mesh_config.tp == n_kv, (
            f"TP({mesh_config.tp}) must equal num_key_value_heads({n_kv}); the packed cache holds "
            f"exactly one KV head per chip (00_MODEL_CARD.md section 4.1)"
        )
        ccl = CCLManager(sub, num_links=get_default_num_links(sub), topology=prefill_topology())
        weight = _labelled_projection_weight(hf, head_dim, uniform=uniform)
        # Both projections carry the SAME labelled weight, so whichever the probe reads back is the
        # one under test and the other cannot mask a failure.
        attn = _build_probe_attention(
            sub,
            hf,
            {"k": weight, "v": weight},
            seq_len=seq_len,
            max_seq_len=max_seq_len,
            with_rope=with_rope,
            mesh_config=mesh_config,
            ccl_manager=ccl,
        )
        kv_cache = allocate_kv_cache(
            sub, num_layers=1, max_seq_len=max_seq_len, num_users=1, head_dim=head_dim, cache_dtype=CACHE_DTYPE
        )

        x = ttnn.from_torch(
            _labelled_input(hf, seq_len, start_pos=kv_actual),
            device=sub,
            dtype=ACTIVATION_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
        )
        out = attn(x, build_prefill_rope(sub, hf, seq_len, start_pos=0), kv_cache=kv_cache, cached_len=kv_actual)
        out.deallocate(True)
        ttnn.synchronize_device(sub)

        read = "v" if probe.startswith("v") else "k"
        cache_tensor = kv_cache.v if read == "v" else kv_cache.k
        mismatched = []
        for column in range(n_kv):
            got = _read_cache_column(cache_tensor, column, layer_idx=0, seq_len=max_seq_len)[
                kv_actual : kv_actual + seq_len
            ]
            expected = _expected_head(head_dim, seq_len, column, start_pos=kv_actual, uniform=uniform)
            if not torch.equal(got, expected):
                mismatched.append((column, float((got - expected).abs().max())))
        logger.info(
            f"[G-KV-TP8] arm A {probe}: head->column at kv_actual={kv_actual}, s={seq_len}, "
            f"TP={mesh_config.tp}, topology={prefill_topology()}: "
            f"{n_kv - len(mismatched)}/{n_kv} columns bit-identical (torch.equal, rtol=atol=0)"
        )
        # The written region only: the pad tail past this chunk must be exactly zero.
        tail = _read_cache_column(cache_tensor, 0, layer_idx=0, seq_len=max_seq_len)[kv_actual + seq_len :]
        tail_max = float(tail.abs().max()) if tail.numel() else 0.0
        logger.info(
            f"[G-KV-TP8] arm A {probe}: pad tail rows [{kv_actual + seq_len}, {max_seq_len}) on column 0: "
            f"max|x| = {tail_max}"
        )

        kv_cache.k.deallocate(True)
        kv_cache.v.deallocate(True)

    assert not mismatched, (
        f"head->column is wrong on columns {mismatched} (column, max|delta|). At TP=8 the loader's "
        f"column_parallel mapper must put global KV head c on mesh column c; a PCC gate would not "
        f"have caught this (recipe section 2.5)."
    )
    assert tail_max == 0.0, (
        f"the write spilled past the chunk: rows [{kv_actual + seq_len}, {max_seq_len}) hold "
        f"max|x| = {tail_max}, expected exactly 0"
    )


@requires_galaxy
@requires_ring_fabric
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_rotated_column_control_passes_pcc_and_fails_bit_equality(submesh_pool, reset_seeds):
    """**The negative control, and it is a control against the *instrument*, not only the code.**

    Read mesh column `(c + 1) % 8` as KV head `c` — a completely wrong head->column map — and
    report both discriminators. `torch.equal` must reject it; the **PCC must not**, because half of
    every lane block carries the head-independent position label. This is recipe §2.5's measured
    0.99890 reproduced on this package's own probe: a layout gate built on PCC is not a gate.
    """
    hf = llama_config_dims()
    head_dim, n_kv = derive_head_dim(hf), hf["num_key_value_heads"]
    seq_len, max_seq_len = 128, 512

    with submesh_pool.use(TP_MESH_SHAPE) as sub:
        mesh_config = MeshConfig(TP_MESH_SHAPE, tp=TP_MESH_SHAPE[1])
        ccl = CCLManager(sub, num_links=get_default_num_links(sub), topology=prefill_topology())
        weight = _labelled_projection_weight(hf, head_dim, uniform=False)
        attn = _build_probe_attention(
            sub,
            hf,
            {"k": weight, "v": weight},
            seq_len=seq_len,
            max_seq_len=max_seq_len,
            with_rope=True,
            mesh_config=mesh_config,
            ccl_manager=ccl,
        )
        kv_cache = allocate_kv_cache(
            sub, num_layers=1, max_seq_len=max_seq_len, num_users=1, head_dim=head_dim, cache_dtype=CACHE_DTYPE
        )
        x = ttnn.from_torch(
            _labelled_input(hf, seq_len),
            device=sub,
            dtype=ACTIVATION_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
        )
        attn(x, build_prefill_rope(sub, hf, seq_len), kv_cache=kv_cache, cached_len=0).deallocate(True)
        ttnn.synchronize_device(sub)

        correct = torch.stack(
            [_read_cache_column(kv_cache.v, c, layer_idx=0, seq_len=seq_len) for c in range(n_kv)], dim=0
        )
        rotated = torch.stack(
            [_read_cache_column(kv_cache.v, (c + 1) % n_kv, layer_idx=0, seq_len=seq_len) for c in range(n_kv)], dim=0
        )
        kv_cache.k.deallocate(True)
        kv_cache.v.deallocate(True)

    _, pcc = comp_pcc(correct, rotated, 0.0)
    bit_equal = torch.equal(correct, rotated)
    logger.info(
        f"[G-KV-TP8] control (read column (c+1)%{n_kv} as head c): PCC = {float(pcc):.5f}, "
        f"torch.equal = {bit_equal}, max|delta| = {float((correct - rotated).abs().max()):.5f}. "
        f"Recipe section 2.5 measured 0.99890 for this control; PCC is not a layout gate."
    )
    assert (
        not bit_equal
    ), "the rotated-column control is bit-identical to the correct read — the probe has no head label"
    assert float(pcc) > 0.9, (
        f"the rotated-column control scored PCC {float(pcc):.5f}, low enough that a PCC gate would "
        f"have caught it. Then this probe is not the one recipe section 2.5 warns about and the "
        f"control is not demonstrating the weakness it exists to demonstrate — check that the "
        f"position label really is head-independent."
    )


# ---------------------------------------------------------------------------------------------
# arm A, second half — the advancing write offset at TP=8
# ---------------------------------------------------------------------------------------------
@requires_galaxy
@requires_ring_fabric
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_write_offset_at_tp8_is_bit_exact(submesh_pool, reset_seeds):
    """The chunk write offset at TP=8, over three offsets, `rtol = atol = 0`.

    Same labelled probe and same `write_kv_chunk` op as the test above, but driven through
    `apply_qkv_projection` -> `split_qkv_heads_prefill` -> `write_kv_chunk` **without an attention
    core** (`DEC-080`). The reason is structural, not convenience: `Attention.__call__` at
    `cached_len > 0` is **delta 3**, and on a `(1, 8)` mesh there is no SP axis for the ring core,
    so the dense core refuses it — correctly. `G-CHUNK-ATTN` on `(4, 8)` is where the cache-backed
    core runs; what remains to prove here is that the *write* lands on the right rows of the right
    chip when the heads are really sharded, which `G-KV` could not test at TP=1.

    Each chunk carries its own position labels (`(kv_actual + s) % 128`), so a chunk written at the
    wrong offset cannot coincide with the expected values.
    """
    hf = llama_config_dims()
    head_dim, n_kv = derive_head_dim(hf), hf["num_key_value_heads"]
    n_heads = hf["num_attention_heads"]
    seq_len, max_seq_len = 128, 512
    offsets = [0, 128, 256]

    with submesh_pool.use(TP_MESH_SHAPE) as sub:
        mesh_config = MeshConfig(TP_MESH_SHAPE, tp=TP_MESH_SHAPE[1])
        ccl = CCLManager(sub, num_links=get_default_num_links(sub), topology=prefill_topology())
        weight = _labelled_projection_weight(hf, head_dim, uniform=False)
        attn = _build_probe_attention(
            sub,
            hf,
            {"k": weight, "v": weight},
            seq_len=seq_len,
            max_seq_len=max_seq_len,
            with_rope=True,
            mesh_config=mesh_config,
            ccl_manager=ccl,
        )
        kv_cache = allocate_kv_cache(
            sub, num_layers=1, max_seq_len=max_seq_len, num_users=1, head_dim=head_dim, cache_dtype=CACHE_DTYPE
        )
        ckc = attn.program_config.get_compute_kernel_config(sub)
        for kv_actual in offsets:
            x = ttnn.from_torch(
                _labelled_input(hf, seq_len, start_pos=kv_actual),
                device=sub,
                dtype=ACTIVATION_DTYPE,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(sub),
            )
            q, k, v = apply_qkv_projection(x, attn.weights, ckc)
            tt_k, tt_v = split_qkv_heads_prefill(
                q, k, v, mesh_config.shard_size(n_heads), mesh_config.shard_size(n_kv)
            )[1:]
            for t in (x, q, k, v):
                t.deallocate(True)
            write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=0, layer_idx=0, kv_actual=kv_actual, sp_axis=0)
            tt_k.deallocate(True)
            tt_v.deallocate(True)
        ttnn.synchronize_device(sub)

        mismatched = []
        for kv_actual in offsets:
            for column in range(n_kv):
                got = _read_cache_column(kv_cache.v, column, layer_idx=0, seq_len=max_seq_len)[
                    kv_actual : kv_actual + seq_len
                ]
                expected = _expected_head(head_dim, seq_len, column, start_pos=kv_actual, uniform=False)
                if not torch.equal(got, expected):
                    mismatched.append((kv_actual, column, float((got - expected).abs().max())))
        tail = _read_cache_column(kv_cache.v, 0, layer_idx=0, seq_len=max_seq_len)[offsets[-1] + seq_len :]
        tail_max = float(tail.abs().max()) if tail.numel() else 0.0
        kv_cache.k.deallocate(True)
        kv_cache.v.deallocate(True)

    logger.info(
        f"[G-KV-TP8] arm A write offsets {offsets} x {n_kv} columns at TP=8: "
        f"{len(offsets) * n_kv - len(mismatched)}/{len(offsets) * n_kv} blocks bit-identical; "
        f"pad tail rows [{offsets[-1] + seq_len}, {max_seq_len}) max|x| = {tail_max}"
    )
    assert not mismatched, f"the write offset is wrong at (kv_actual, column, max|delta|): {mismatched}"
    assert tail_max == 0.0, f"the writes spilled past the last chunk: pad tail max|x| = {tail_max}"


# ---------------------------------------------------------------------------------------------
# arm B — 32 layers of model-produced K/V vs the fp32 golden
# ---------------------------------------------------------------------------------------------
@requires_galaxy
@requires_ring_fabric
@requires_hf_reference
@requires_golden_trace
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_model_produced_kv_vs_golden_at_tp8(submesh_pool, reset_seeds):
    """**Arm B.** The real 32-layer model writes its own cache at TP=8; score every layer.

    * **Input distribution:** the golden trace's own `token_ids` — real Llama-3.1-8B-Instruct
      tokens, 512 of them — and real checkpoint weights. This is the "real embedding scale" arm
      §2.2.2 says predicts model behaviour, not a synthetic one.
    * **Reference dtype policy:** the golden is **fp32** throughout (`DEC-059`), generated by
      `scripts/generate_golden_kv_cache.py` and proved bit-identical to `LlamaModel`'s own loop
      (`G-GOLDEN`). The device's K is Meta-swizzled over `head_dim`, so the golden is **permuted
      HF -> Meta before quantising**, because `bfloat8_b` shares one exponent per 16-element block
      of the last dim and the block boundaries move with the permutation (recipe §2.2.3a).
    * **Computed noise floor:** the layer-0 **complete** floor from `G-CHUNK` — every value the
      device holds rounded to the dtype it holds it in (bf16 input, bf16 norm gain, bf8_b
      projection weight, bf16 RoPE tables, bf8_b store), all remaining math fp32 (`R-021`,
      `DEC-064`). The storage-only floor is recorded beside it, unasserted.
    * **Negative control:** the rotated-column test above; a wrong head->column map is what would
      make these numbers plausible-but-wrong, and it is gated bit-exactly rather than here.
    """
    hf = llama_config_dims()
    metadata = _golden_metadata()
    n_layers, seq_len = int(metadata["num_layers"]), int(metadata["n_tokens"])
    head_dim, n_kv = derive_head_dim(hf), hf["num_key_value_heads"]
    tokens = torch.tensor(metadata["token_ids"], dtype=torch.long).reshape(1, seq_len)
    # > seq_len so the written region has a pad tail to check, and — at sp = 1 — with no effect on
    # the attention core, which stays `dense` (`tt/attention/prefill.py::select_attention_core`).
    max_seq_len = 2 * seq_len

    state_dict = _load_checkpoint(n_layers)
    k_floor_l0, v_floor_l0 = _layer0_floors(hf, state_dict, tokens, seq_len)
    golden = {i: _golden_layer(i) for i in range(n_layers)}

    with submesh_pool.use(TP_MESH_SHAPE) as sub:
        mesh_config = MeshConfig(TP_MESH_SHAPE, tp=TP_MESH_SHAPE[1])
        ccl = CCLManager(sub, num_links=get_default_num_links(sub), topology=prefill_topology())
        model = Model(
            sub,
            hf,
            state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            weight_dtype=WEIGHT_DTYPE,
            activation_dtype=ACTIVATION_DTYPE,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            with_lm_head=False,
        )
        kv_cache = allocate_kv_cache(
            sub,
            num_layers=n_layers,
            max_seq_len=max_seq_len,
            num_users=1,
            head_dim=head_dim,
            cache_dtype=CACHE_DTYPE,
        )
        x, rope_mats, _ = model.prepare_inputs_prefill(tokens)
        out = model.prefill_forward(x, rope_mats, kv_cache=kv_cache, skip_lm_head=True)
        out.deallocate(True)
        ttnn.synchronize_device(sub)

        curve = {"k": {}, "v": {}}
        tails = []
        for layer_idx in range(n_layers):
            # Mesh column c IS head c — the claim arm A gates bit-exactly.
            got_k = torch.stack(
                [_read_cache_column(kv_cache.k, c, layer_idx=layer_idx, seq_len=seq_len) for c in range(n_kv)], dim=0
            ).unsqueeze(0)
            got_v = torch.stack(
                [_read_cache_column(kv_cache.v, c, layer_idx=layer_idx, seq_len=seq_len) for c in range(n_kv)], dim=0
            ).unsqueeze(0)
            golden_k = _to_meta(golden[layer_idx][0], head_dim)
            _, pcc_k = comp_pcc(golden_k, got_k, 0.0)
            _, pcc_v = comp_pcc(golden[layer_idx][1], got_v, 0.0)
            curve["k"][layer_idx], curve["v"][layer_idx] = float(pcc_k), float(pcc_v)
            for cache in (kv_cache.k, kv_cache.v):
                tail = ttnn.to_torch(ttnn.get_device_tensors(cache)[0]).float()[layer_idx, 0, seq_len:, :]
                tails.append(float(tail.abs().max()))
            logger.info(f"[G-KV-TP8] arm B L{layer_idx:>2}: K={float(pcc_k):.7f} V={float(pcc_v):.7f}")

        kv_cache.k.deallocate(True)
        kv_cache.v.deallocate(True)

    golden_k0 = _to_meta(golden[0][0], head_dim)
    _, storage_k = comp_pcc(golden_k0, quantize_like_device(golden_k0, CACHE_DTYPE), 0.0)
    _, storage_v = comp_pcc(golden[0][1], quantize_like_device(golden[0][1], CACHE_DTYPE), 0.0)
    _, complete_k = comp_pcc(golden_k0, k_floor_l0, 0.0)
    _, complete_v = comp_pcc(golden[0][1], v_floor_l0, 0.0)

    ratios = {
        "k": {
            "measured": curve["k"][0],
            "storage": err_ratio(curve["k"][0], float(storage_k)),
            "complete": err_ratio(curve["k"][0], float(complete_k)),
        },
        "v": {
            "measured": curve["v"][0],
            "storage": err_ratio(curve["v"][0], float(storage_v)),
            "complete": err_ratio(curve["v"][0], float(complete_v)),
        },
    }
    for name in ("k", "v"):
        logger.info(
            f"[G-KV-TP8] L0 {name}: PCC={ratios[name]['measured']:.7f}  "
            f"storage floor -> {ratios[name]['storage']:.2f}x  "
            f"complete floor -> {ratios[name]['complete']:.2f}x  (budget {MAX_L0_ERR_RATIO}x)"
        )
    min_k, min_v = min(curve["k"].values()), min(curve["v"].values())
    logger.info(
        f"[G-KV-TP8] arm B over {n_layers} layers at TP=8, s={seq_len}: min K = {min_k:.7f} "
        f"(threshold {GOLDEN_K_THRESHOLD}), min V = {min_v:.7f} (threshold {GOLDEN_V_THRESHOLD}); "
        f"pad tail max|x| over {len(tails)} (layer, cache) pairs = {max(tails)}"
    )

    os.makedirs(_RAW_DIR, exist_ok=True)
    with open(os.path.join(_RAW_DIR, "G-KV-TP8_per_layer_pcc.json"), "w") as f:
        json.dump(
            {
                "mesh": TP_MESH_SHAPE,
                "tp": 8,
                "sp": 1,
                "topology": str(prefill_topology()),
                "seq_len": seq_len,
                "max_seq_len": max_seq_len,
                "n_layers": n_layers,
                "cache_dtype": "bfloat8_b",
                "layer0_floors": {
                    "storage_k": float(storage_k),
                    "storage_v": float(storage_v),
                    "complete_k": float(complete_k),
                    "complete_v": float(complete_v),
                },
                "layer0_ratios": ratios,
                "per_layer": {name: {str(k): v for k, v in sorted(vals.items())} for name, vals in curve.items()},
            },
            f,
            indent=2,
        )

    assert max(tails) == 0.0, f"the model's writes spilled past the prompt: pad tail max|x| = {max(tails)}"
    assert min_k >= GOLDEN_K_THRESHOLD, f"min K over {n_layers} layers is {min_k:.7f} < {GOLDEN_K_THRESHOLD}"
    assert min_v >= GOLDEN_V_THRESHOLD, f"min V over {n_layers} layers is {min_v:.7f} < {GOLDEN_V_THRESHOLD}"
    for name in ("k", "v"):
        assert ratios[name]["complete"] <= MAX_L0_ERR_RATIO, (
            f"L0 {name} is {ratios[name]['complete']:.2f}x its COMPLETE floor (budget "
            f"{MAX_L0_ERR_RATIO}x). The floor already carries every value the device stores, so the "
            f"remainder is this package's."
        )
