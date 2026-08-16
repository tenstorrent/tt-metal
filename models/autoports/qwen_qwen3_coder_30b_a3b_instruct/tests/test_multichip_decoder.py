# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the multichip decoder on the full 4-die P300_X2 mesh.

The reference these tests compare against is the **single-chip TTNN optimized
decoder**, not HuggingFace, and it is run *on the same mesh with every tensor
replicated*. That is the whole trick of this file: a mesh op is SPMD, so
uploading the unsharded stage-02 weights with ``ReplicateTensorToMesh`` makes
each of the four dies independently compute the exact single-chip answer, in the
same process, from the same host tensors, with the same program cache. The
comparison then isolates sharding and collective bugs from every source of
numerical difference that HF-vs-TTNN would drag in.

``test_baseline_upload_is_actually_replicated`` is what stops that reference
quietly becoming meaningless -- if ``from_torch`` ever stopped replicating, the
baseline would still produce *a* number and every PCC below would still pass.

Two tests here are load-bearing in a way their size does not suggest:

* ``test_topk_is_identical_across_dies``. The expert-parallel scheme assumes the
  four dies agree on the global top-8 from bit-identical replicated logits, so
  that the four 32-expert windows partition it. If ``ttnn.topk`` ever broke a
  tie differently on one die the layer would be **silently wrong** -- no shape
  error, no assert, just PCC drift -- so the property is asserted directly
  rather than argued from "same program, same input".

* ``test_expert_window_can_be_empty``. Under EP the locally-live expert count is
  data-dependent in 0..8, which is why decode must pass ``nnz=None``. The zero
  case is the one that never happens by accident in a random test and is exactly
  where an uninitialised output buffer would leak a NaN into the all-reduce.

Every test opens the mesh with ``fabric_config=FABRIC_1D_RING``. Without it the
collectives have no fabric to run on; with ``FABRIC_1D`` they would run but on
the linear topology the CCL sweep measured 1.2-1.8x slower.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.modules.tt_ccl import default_topology
from models.common.utility_functions import comp_pcc

from ..tt import functional_decoder as F
from ..tt import multichip_decoder as MC
from ..tt import optimized_decoder as O
from ..tt.weight_mapping import convert_layer_weights
from .reference import build_reference_layer, layer_state_dict, rotary_embeddings

LAYER_IDX = 0
# Against the replicated single-chip baseline only sharding and the collectives
# differ, so 0.99 -- the threshold used against HF, where dtype and kernel choice
# differ too -- was two orders of magnitude looser than the measured margin. The
# actuals span 0.99945 (two stacked layers, the only one below 0.9996) to
# 1.0; 0.999 sits below the worst of them and still catches anything that would
# make the sharding wrong rather than merely different.
PCC_VS_SINGLE_CHIP = 0.999
PCC_VS_HF = 0.995
MAX_SEQ = 1024
BLOCK_SIZE = 32
TRACE_REGION_SIZE = 90000000

# Ring fabric must be configured before the mesh is opened, which is what this
# indirect parametrisation does (conftest.set_fabric runs ahead of the open).
MESH_PARAMS = {"trace_region_size": TRACE_REGION_SIZE, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
mesh_4 = pytest.mark.parametrize("mesh_device", [MC.MESH_SHAPE], ids=["1x4"], indirect=True)
ring_fabric = pytest.mark.parametrize("device_params", [MESH_PARAMS], indirect=True)


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.fixture(scope="module")
def torch_weights(reference):
    _, hf_config = reference
    return convert_layer_weights(layer_state_dict(LAYER_IDX), hf_config)


def _hidden(hf_config, seq_len, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.float32) * 0.02


def _reference_layer(layer, hf_config, hidden):
    """HF layer output for a ``[1, 1, S, H]`` input, returned as ``[1, S, H]``."""
    hidden = hidden.reshape(1, -1, hf_config.hidden_size)
    seq_len = hidden.shape[1]
    cos, sin = rotary_embeddings(hf_config, seq_len)
    mask = torch.full((seq_len, seq_len), float("-inf")).triu(1).reshape(1, 1, seq_len, seq_len)
    with torch.no_grad():
        out = layer(hidden, position_embeddings=(cos, sin), attention_mask=mask)
    return out[0] if isinstance(out, tuple) else out


def _replicate(t, mesh_device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _per_die(t, mesh_device, dim: int = 0) -> torch.Tensor:
    """All four dies' copies of a tensor, concatenated along ``dim``.

    Every activation in this layer is ``[1, 1, ., .]``, so concatenating on dim 0
    puts die *d* at index *d* and reads like a stack -- but it *is* a
    concatenation, which matters for the KV cache, whose dim 0 is the block index
    and which therefore has to be reassembled on the head axis instead.
    """
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=dim))


class Fixture:
    """Both paths uploaded side by side onto the same mesh.

    ``baseline`` is the stage-02 optimized decoder with every tensor replicated,
    so each die computes the full single-chip layer; ``multichip`` is the sharded
    stage-03 path. They share the router weight tensor, the RoPE caches and the
    host weights, so the only difference between them is the parallelisation.
    """

    def __init__(self, mesh_device, hf_config, torch_weights):
        self.mesh = mesh_device
        self.hf = hf_config
        self.config = MC.MeshDecoderConfig.from_hf(hf_config)
        self.ctx = MC.mesh_context(mesh_device)
        self.torch_router = torch_weights["router"]
        self.multichip = MC.upload_multichip_weights(torch_weights, mesh_device, self.config)
        self.baseline_experts = O.upload_optimized_weights(torch_weights, mesh_device, self.config.global_config.moe)
        self.baseline = F.DecoderLayerWeights(
            input_layernorm=self.multichip.input_layernorm,
            post_attention_layernorm=self.multichip.post_attention_layernorm,
            attention=None,  # the optimized path reads OptimizedWeights.attention
            router=self.multichip.router,
            experts=None,
        )
        self.cos, self.sin = F.build_rope_cache(hf_config, MAX_SEQ, mesh_device)
        self.baseline_sparsity = F.build_expert_sparsity(mesh_device, self.config.global_config.moe.num_experts)
        self.sparsity = MC.build_local_sparsity(mesh_device, self.config.local_moe)

    def rep(self, t):
        return _replicate(t, self.mesh)

    def dies(self, t, dim: int = 0):
        return _per_die(t, self.mesh, dim)

    def baseline_prefill(self, x, kv_cache=None, user_id=0):
        return O.decoder_layer_prefill_optimized(
            self.rep(x),
            self.baseline,
            self.config.global_config,
            self.cos,
            self.sin,
            self.baseline_sparsity,
            self.baseline_experts,
            kv_cache=kv_cache,
            user_id=user_id,
        )

    def multichip_prefill(self, x, kv_cache=None, user_id=0):
        return MC.decoder_layer_prefill_multichip(
            self.rep(x),
            self.multichip,
            self.config,
            self.ctx,
            self.cos,
            self.sin,
            self.sparsity,
            kv_cache=kv_cache,
            user_id=user_id,
        )


@pytest.fixture
def fixture(mesh_device, reference, torch_weights):
    _, hf_config = reference
    return Fixture(mesh_device, hf_config, torch_weights)


# --- host-side weight transforms ---------------------------------------------


def test_wqkv_column_split_is_head_interleaved(reference, torch_weights):
    """Die *d* must get Q heads 8d..8d+7 plus K head d and V head d.

    A contiguous 4-way split of the checkpoint's ``[Wq | Wk | Wv]`` gives die 0
    nothing but Q heads and die 3 nothing but K and V, and produces **no shape
    error** -- ``nlp_create_qkv_heads_decode(num_heads=8, num_kv_heads=1)``
    accepts 1280 columns whatever is in them. This is a host-only test because
    that is where the bug would live and where it is cheapest to catch.
    """
    _, hf_config = reference
    cfg = MC.MeshDecoderConfig.from_hf(hf_config)
    a = cfg.global_config.attention
    n, hd = cfg.num_devices, a.head_dim
    full = torch_weights["wqkv"].reshape(a.hidden_size, -1)
    permuted = MC.head_interleaved_wqkv(full, a, n)

    q_end = a.num_attention_heads * hd
    k_end = q_end + a.num_key_value_heads * hd
    per_die = permuted.shape[-1] // n
    q_per = a.num_attention_heads // n

    for d in range(n):
        shard = permuted[:, d * per_die : (d + 1) * per_die]
        assert shard.shape[-1] == q_per * hd + 2 * hd, shard.shape
        expect_q = full[:, d * q_per * hd : (d + 1) * q_per * hd]
        expect_k = full[:, q_end + d * hd : q_end + (d + 1) * hd]
        expect_v = full[:, k_end + d * hd : k_end + (d + 1) * hd]
        assert torch.equal(shard[:, : q_per * hd], expect_q), f"die {d} Q heads"
        assert torch.equal(shard[:, q_per * hd : q_per * hd + hd], expect_k), f"die {d} K head"
        assert torch.equal(shard[:, q_per * hd + hd :], expect_v), f"die {d} V head"

    # And it is a permutation, not a rewrite: every column survives exactly once.
    # Compared as sorted multisets rather than row sums -- the reordering changes
    # float addition order, so ``sum`` differs in the last bits even when nothing
    # has been lost, and asserting on it fails for the wrong reason.
    assert torch.equal(permuted.sort(dim=-1).values, full.sort(dim=-1).values)
    logger.info(f"wqkv head-interleaved split verified for {n} dies, per-die N = {per_die}")


def test_repo_default_topology_is_wrong_for_this_mesh():
    """Documents *why* Ring is passed explicitly, so it cannot be "simplified" away.

    ``tt_ccl.default_topology()`` only returns ``Ring`` for 8-device T3K and
    Galaxy; for this 4-device Blackhole mesh it returns ``Linear``, which the CCL
    sweep measured at 1.21x slower at decode size and 1.79x at 2 MB. The module
    constant must therefore disagree with the helper.
    """
    assert MC.TOPOLOGY is ttnn.Topology.Ring
    assert MC.NUM_LINKS == 2
    logger.info(f"multichip_decoder overrides default_topology (callable: {default_topology.__name__})")


# --- the replicated baseline itself ------------------------------------------


@ring_fabric
@mesh_4
def test_baseline_upload_is_actually_replicated(fixture):
    """The single-chip reference is only a reference if all four dies agree.

    Everything else in this file divides by this. If ``ReplicateTensorToMesh``
    ever stopped replicating, or the mesh stopped being SPMD, the baseline would
    still return numbers and every PCC below would still pass against them.
    """
    out = fixture.dies(fixture.baseline_prefill(_hidden(fixture.hf, 128)))
    spread = (out - out[0:1]).abs().max().item()
    logger.info(f"replicated single-chip baseline: max spread across 4 dies = {spread:.3e}")
    assert spread == 0.0, f"replicated baseline differs across dies by {spread}; it is not a valid reference"


# --- the determinism assumption the whole scheme rests on --------------------


@ring_fabric
@mesh_4
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_topk_is_identical_across_dies(fixture, seed):
    """All four dies must select the same 8 experts from the replicated logits.

    Expert parallelism partitions the 128 experts into four 32-wide windows and
    each die keeps only the winners inside its own. That is the global top-8 only
    if the four dies agree; if they disagree the layer double-counts some experts
    and drops others, with no error of any kind. Random inputs are checked, and
    so is an all-zero input, where every logit is the router's bias-free
    projection of zero and the top-8 is decided **entirely by tie-breaking** --
    the degenerate case an ordinary test never reaches.
    """
    hidden = _hidden(fixture.hf, 128, seed=seed) if seed else torch.zeros(1, 1, 128, fixture.hf.hidden_size)
    logits = ttnn.linear(
        fixture.rep(hidden), fixture.multichip.router, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    _, indices = ttnn.topk(logits, k=fixture.config.global_config.moe.num_experts_per_tok, dim=-1, sorted=True)
    per_die = fixture.dies(indices)
    for d in range(1, fixture.config.num_devices):
        assert torch.equal(per_die[0], per_die[d]), (
            f"seed={seed}: die {d} selected different experts than die 0 "
            f"({int((per_die[0] != per_die[d]).sum())} of {per_die[0].numel()} slots differ). "
            "The four expert windows are no longer a partition of the global top-8."
        )
    logger.info(f"topk seed={seed}: 4 dies bit-identical over {per_die[0].numel()} selections")


@ring_fabric
@mesh_4
@pytest.mark.parametrize("seq_len", [1, 33, 128], ids=["decode", "s33", "s128"])
def test_router_windows_partition_global_routing(fixture, seq_len):
    """Concatenating the four local windows must reproduce the global dense routing.

    This is the direct statement of the EP contract: the multichip router returns
    ``[1, 1, S, 32]`` per die, and stitching them in device order must equal what
    the single-chip router returns as one ``[1, 1, S, 128]`` row -- same experts,
    same weights, normalised by the same global denominator.
    """
    x = fixture.rep(_hidden(fixture.hf, seq_len))
    moe = fixture.config.global_config.moe
    global_dense = fixture.dies(O.router_forward_optimized(x, fixture.multichip.router, moe))[0].float()
    local = fixture.dies(
        MC.router_forward_multichip(
            x, fixture.multichip.router, fixture.multichip.expert_window, moe, fixture.config.local_moe
        )
    ).float()

    n_local = fixture.config.local_moe.num_experts
    stitched = torch.cat([local[d].reshape(-1, n_local) for d in range(fixture.config.num_devices)], dim=-1)
    reference = global_dense.reshape(-1, moe.num_experts)
    delta = (stitched - reference).abs().max().item()
    logger.info(f"router windows seq={seq_len}: max |stitched - global| = {delta:.3e}")
    assert torch.equal(stitched > 0, reference > 0), "the four windows do not select the global top-8"
    assert delta == 0.0, f"routing weights differ by {delta}; the window matmul is not exact"
    assert ((stitched > 0).sum(dim=-1) == moe.num_experts_per_tok).all()


# --- prefill ------------------------------------------------------------------


@ring_fabric
@mesh_4
@pytest.mark.parametrize("seq_len", [32, 128, 512, 33, 100, 257], ids=["s32", "s128", "s512", "s33", "s100", "s257"])
def test_multichip_prefill_vs_single_chip(fixture, seq_len):
    """Prefill against the single-chip TTNN baseline, aligned and non-aligned.

    The non-aligned lengths are the point of the parametrisation: nothing in the
    multichip path may turn a decoder that accepted any prompt length into one
    that only accepts multiples of a chunk, tile, page or collective block. The
    collectives scatter on dim 3 (hidden, 2048), which is independent of S.
    """
    hidden = _hidden(fixture.hf, seq_len)
    base = fixture.dies(fixture.baseline_prefill(hidden))[0:1].float()
    multi = fixture.dies(fixture.multichip_prefill(hidden))

    spread = (multi - multi[0:1]).abs().max().item()
    assert spread == 0.0, f"S={seq_len}: layer output differs across dies by {spread}; the all-reduce is not complete"
    assert tuple(multi.shape) == (fixture.config.num_devices, 1, seq_len, fixture.hf.hidden_size), (
        f"S={seq_len}: four dies of [1,1,S,H] concatenated on dim 0 came back {tuple(multi.shape)}; "
        "the replicated layer contract is not intact"
    )

    passing, message = comp_pcc(base, multi[0:1].float(), PCC_VS_SINGLE_CHIP)
    logger.info(f"multichip prefill S={seq_len} vs single-chip TTNN: {message}")
    assert passing, f"multichip prefill S={seq_len} below {PCC_VS_SINGLE_CHIP} vs single-chip: {message}"


@ring_fabric
@mesh_4
@pytest.mark.parametrize("seq_len", [128, 33], ids=["s128", "s33"])
def test_multichip_prefill_vs_hf(fixture, reference, seq_len):
    """The end-to-end bar: the same 0.995 PCC against HF the single chip clears."""
    layer, hf_config = reference
    hidden = _hidden(hf_config, seq_len)
    ref = _reference_layer(layer, hf_config, hidden)
    multi = fixture.dies(fixture.multichip_prefill(hidden))[0].reshape(1, seq_len, hf_config.hidden_size)
    passing, message = comp_pcc(ref, multi.float(), PCC_VS_HF)
    logger.info(f"multichip prefill S={seq_len} vs HF: {message}")
    assert passing, f"multichip prefill S={seq_len} below {PCC_VS_HF} vs HF: {message}"


@ring_fabric
@mesh_4
def test_multichip_prefill_is_deterministic(fixture):
    """Bitwise repeatability, including through two collectives per layer."""
    hidden = _hidden(fixture.hf, 128)
    outs = [fixture.dies(fixture.multichip_prefill(hidden)).clone() for _ in range(3)]
    assert torch.equal(outs[0], outs[1]), "multichip prefill run 1 != run 2 (bitwise)"
    assert torch.equal(outs[0], outs[2]), "multichip prefill run 1 != run 3 (bitwise)"
    logger.info("multichip prefill: 3 runs bit-identical on all 4 dies")


# --- KV cache and decode ------------------------------------------------------


@ring_fabric
@mesh_4
def test_local_kv_cache_layout(fixture):
    """Each die owns exactly one KV head, and the four together hold the whole cache.

    This is the memory half of the TP decision: 512 B per token per layer per die
    instead of 2048. The test does not merely check the *shape* -- it prefills
    both paths from the same prompt and asserts that stacking the four dies' K
    caches on the head axis reproduces the single-chip cache, which is what
    proves the head *assignment* matches the wqkv column split rather than just
    the head count.
    """
    cfg = fixture.config
    base_kv = F.create_kv_cache(fixture.mesh, cfg.global_config.attention, 1, 128, block_size=BLOCK_SIZE)
    mc_kv = MC.create_mesh_kv_cache(fixture.mesh, cfg, 1, 128, block_size=BLOCK_SIZE)

    assert mc_kv.k.shape[1] == 1, f"per-die KV cache has {mc_kv.k.shape[1]} heads, expected 1"
    assert base_kv.k.shape[1] == cfg.global_config.attention.num_key_value_heads
    assert mc_kv.is_paged and base_kv.is_paged

    hidden = _hidden(fixture.hf, 64)
    fixture.baseline_prefill(hidden, kv_cache=base_kv)
    fixture.multichip_prefill(hidden, kv_cache=mc_kv)

    # The cache's dim 0 is the physical block index, so the four dies are
    # reassembled on the *head* axis -- which is also exactly what makes this a
    # test of head ownership rather than of head count.
    n_kv = cfg.global_config.attention.num_key_value_heads
    # The baseline cache is replicated, so any one die's copy is the reference;
    # take die 0's four heads out of the 16 the concat produces.
    base_k = fixture.dies(base_kv.k, dim=1)[:, :n_kv].float()  # [blocks, 4, block, head_dim]
    stitched = fixture.dies(mc_kv.k, dim=1).float()  # 4 x [blocks, 1, block, head_dim]
    assert stitched.shape == base_k.shape, (stitched.shape, base_k.shape)
    passing, message = comp_pcc(base_k, stitched, 0.999)
    logger.info(f"local KV head layout: 4x[.,1,.,.] stitched vs single-chip [.,4,.,.]: {message}")
    assert passing, f"per-die KV heads are not the single-chip heads in device order: {message}"

    bytes_per_token = mc_kv.k.shape[1] * cfg.local_attention.head_dim * 2 * 2
    logger.info(f"per-die KV: {bytes_per_token} B/token/layer (single chip: {bytes_per_token * 4})")
    assert bytes_per_token == 512


@ring_fabric
@mesh_4
@pytest.mark.parametrize("block_size", [None, 32], ids=["contiguous", "paged32"])
def test_multichip_decode_vs_single_chip(fixture, block_size):
    """One decode step against the single-chip baseline, both cache modes.

    Both paths are prefilled with the same prompt into their own caches, so this
    covers the paged write path (``paged_fill_cache``), the paged update
    (``paged_update_cache`` at 1 KV head, which the design phase flagged as
    unexercised), the page table, and ``cur_pos_tensor``, not just the matmuls.
    """
    cfg = fixture.config
    prompt = 32
    full = _hidden(fixture.hf, prompt + 1)
    base_kv = F.create_kv_cache(fixture.mesh, cfg.global_config.attention, 1, MAX_SEQ, block_size=block_size)
    mc_kv = MC.create_mesh_kv_cache(fixture.mesh, cfg, 1, MAX_SEQ, block_size=block_size)
    fixture.baseline_prefill(full[:, :, :prompt, :], kv_cache=base_kv)
    fixture.multichip_prefill(full[:, :, :prompt, :], kv_cache=mc_kv)

    pos = ttnn.from_torch(
        torch.tensor([prompt], dtype=torch.int32),
        dtype=ttnn.int32,
        device=fixture.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(fixture.mesh),
    )
    token = full[:, :, prompt : prompt + 1, :]

    base = fixture.dies(
        O.decoder_layer_decode_optimized(
            fixture.rep(token),
            fixture.baseline,
            cfg.global_config,
            fixture.cos,
            fixture.sin,
            base_kv,
            pos,
            prompt,
            packed_experts=fixture.baseline_experts,
        )
    )[0:1].float()
    multi = fixture.dies(
        MC.decoder_layer_decode_multichip(
            fixture.rep(token), fixture.multichip, cfg, fixture.ctx, fixture.cos, fixture.sin, mc_kv, pos, prompt
        )
    )
    spread = (multi - multi[0:1]).abs().max().item()
    assert spread == 0.0, f"decode output differs across dies by {spread}"
    kind = "contiguous" if block_size is None else f"paged({block_size})"
    passing, message = comp_pcc(base, multi[0:1].float(), PCC_VS_SINGLE_CHIP)
    logger.info(f"multichip decode [{kind}] vs single-chip TTNN: {message}")
    assert passing, f"multichip decode [{kind}] below {PCC_VS_SINGLE_CHIP}: {message}"


@ring_fabric
@mesh_4
def test_multichip_decode_contiguous_batch8(fixture):
    """The contiguous-cache SDPA workaround at batch > 1, against the single chip.

    ``_sdpa_program_config`` is the layer's one hand-written program config and
    the only place the multichip path departs from stage 02's tuning. It exists
    because at TP=4 the contiguous cache has 1 KV head per die, so at batch 1
    SDPA-decode asks for all 110 worker cores on that head and
    ``sdpa_decode_program_factory.cpp:245`` refuses anything over 64. But
    ``num_cores_per_head`` divides by the batch, so at batch 8 the op would have
    asked for 13 and been legal *without* the config -- and the config is
    supplied unconditionally on the contiguous path.

    That makes batch > 1 the case that actually tests the workaround rather than
    the failure it works around: here the cap is not rescuing anything, it is
    only constraining, and so are the ``q_chunk_size``/``k_chunk_size`` of 32
    that come with it (the default path picked its own). Every other contiguous
    test is batch 1 and every batch > 1 test is paged, so without this one both
    the cap and the chunk sizes are exercised in exactly one configuration.
    """
    cfg = fixture.config
    batch, prompt = 8, 32
    per_user = [_hidden(fixture.hf, prompt + 1, seed=u) for u in range(batch)]

    base_kv = F.create_kv_cache(fixture.mesh, cfg.global_config.attention, batch, 128, block_size=None)
    mc_kv = MC.create_mesh_kv_cache(fixture.mesh, cfg, batch, 128, block_size=None)
    assert not mc_kv.is_paged
    for user, hidden in enumerate(per_user):
        fixture.baseline_prefill(hidden[:, :, :prompt, :], kv_cache=base_kv, user_id=user)
        fixture.multichip_prefill(hidden[:, :, :prompt, :], kv_cache=mc_kv, user_id=user)

    tokens = torch.cat([h[:, :, prompt, :] for h in per_user], dim=1).reshape(1, 1, batch, fixture.hf.hidden_size)
    pos = ttnn.from_torch(
        torch.full((batch,), prompt, dtype=torch.int32),
        dtype=ttnn.int32,
        device=fixture.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(fixture.mesh),
    )

    base = (
        fixture.dies(
            O.decoder_layer_decode_optimized(
                fixture.rep(tokens),
                fixture.baseline,
                cfg.global_config,
                fixture.cos,
                fixture.sin,
                base_kv,
                pos,
                prompt,
                packed_experts=fixture.baseline_experts,
            )
        )[0]
        .reshape(-1, fixture.hf.hidden_size)[:batch]
        .float()
    )
    multi_all = fixture.dies(
        MC.decoder_layer_decode_multichip(
            fixture.rep(tokens), fixture.multichip, cfg, fixture.ctx, fixture.cos, fixture.sin, mc_kv, pos, prompt
        )
    )
    spread = (multi_all - multi_all[0:1]).abs().max().item()
    assert spread == 0.0, f"decode output differs across dies by {spread}"
    multi = multi_all[0].reshape(-1, fixture.hf.hidden_size)[:batch].float()

    assert (multi - multi[0:1]).abs().max().item() > 1e-3, "all users identical (broadcast bug)"
    for user in range(batch):
        passing, message = comp_pcc(base[user : user + 1], multi[user : user + 1], PCC_VS_SINGLE_CHIP)
        logger.info(f"multichip decode [contiguous, batch 8] user {user} vs single-chip TTNN: {message}")
        assert passing, f"contiguous batch-8 user {user} below {PCC_VS_SINGLE_CHIP}: {message}"


@ring_fabric
@mesh_4
def test_multichip_multi_step_decode_vs_hf(fixture, reference):
    """Four consecutive decode steps against HF, each at its own position."""
    layer, hf_config = reference
    cfg = fixture.config
    prompt, steps = 32, 4
    full = _hidden(hf_config, prompt + steps)
    ref = _reference_layer(layer, hf_config, full)

    kv = MC.create_mesh_kv_cache(fixture.mesh, cfg, 1, MAX_SEQ, block_size=BLOCK_SIZE)
    fixture.multichip_prefill(full[:, :, :prompt, :], kv_cache=kv)

    for step in range(steps):
        p = prompt + step
        pos = ttnn.from_torch(
            torch.tensor([p], dtype=torch.int32),
            dtype=ttnn.int32,
            device=fixture.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(fixture.mesh),
        )
        out = MC.decoder_layer_decode_multichip(
            fixture.rep(full[:, :, p : p + 1, :]),
            fixture.multichip,
            cfg,
            fixture.ctx,
            fixture.cos,
            fixture.sin,
            kv,
            pos,
            p,
        )
        got = fixture.dies(out)[0].reshape(1, -1).float()
        passing, message = comp_pcc(ref[:, p, :], got, 0.99)
        logger.info(f"multichip decode step {step} (pos {p}) vs HF: {message}")
        assert passing, f"multichip decode step {step} below 0.99: {message}"


@ring_fabric
@mesh_4
@pytest.mark.parametrize("batch", [1, 2, 8, 32], ids=["b1", "b2", "b8", "b32"])
def test_multichip_decode_batch(fixture, reference, batch):
    """Multi-user decode, each user against its own HF reference.

    32 is the ceiling and it is a TTNN op limit that TP does not move:
    ``nlp_create_qkv_heads_decode_device_operation.cpp:51`` asserts
    ``num_users <= 32``, and that op is on the per-die path too. Per-user
    references are what prove routing is per-user rather than broadcast, which
    matters more under EP than on one die -- a die whose window is empty for one
    user and full for another exercises the dynamic ``nnz`` path in both
    directions inside a single program.
    """
    layer, hf_config = reference
    cfg = fixture.config
    prompt = 32
    kv = MC.create_mesh_kv_cache(fixture.mesh, cfg, batch, 128, block_size=BLOCK_SIZE)
    per_user = [_hidden(hf_config, prompt + 1, seed=u) for u in range(batch)]
    for user, hidden in enumerate(per_user):
        fixture.multichip_prefill(hidden[:, :, :prompt, :], kv_cache=kv, user_id=user)

    tokens = torch.cat([h[:, :, prompt, :] for h in per_user], dim=1).reshape(1, 1, batch, hf_config.hidden_size)
    pos = ttnn.from_torch(
        torch.full((batch,), prompt, dtype=torch.int32),
        dtype=ttnn.int32,
        device=fixture.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(fixture.mesh),
    )
    out = MC.decoder_layer_decode_multichip(
        fixture.rep(tokens), fixture.multichip, cfg, fixture.ctx, fixture.cos, fixture.sin, kv, pos, prompt
    )
    got = fixture.dies(out)[0].reshape(-1, hf_config.hidden_size)[:batch].float()
    assert torch.isfinite(got).all(), f"batch={batch} produced non-finite values"

    for user, hidden in enumerate(per_user):
        ref_user = _reference_layer(layer, hf_config, hidden)[:, prompt, :]
        passing, message = comp_pcc(ref_user, got[user : user + 1], 0.99)
        logger.info(f"multichip decode batch={batch} user {user} vs HF: {message}")
        assert passing, f"batch={batch} user {user} below 0.99: {message}"

    if batch > 1:
        assert (got - got[0]).abs().max().item() > 1e-3, f"batch={batch}: all users identical (broadcast bug)"


# --- stage 04: the layer's own shape and layout contract ----------------------


@ring_fabric
@mesh_4
@pytest.mark.parametrize("batch", [1, 8], ids=["b1", "b8"])
def test_decode_output_layout_matches_input(fixture, batch):
    """The decode layer must return exactly the tensor contract it takes.

    Replicated ``[1, 1, B, 2048]``, bfloat16, TILE, DRAM-interleaved, logical
    shape included -- that is what lets 48 layers stack with no boundary
    conversion, and it is the *inter-layer residual layout contract* that
    ``doc/optimized_multichip_decoder/README.md`` writes down for full-model
    bringup.

    It is asserted rather than assumed because stage 04's persistent collective
    buffers can break it silently. A persistent output buffer imposes its own
    logical shape on the op's result, and the layer's two all-reduces have the
    same *padded* shape but different *logical* ones -- the attention partial is
    32 rows out of ``wo``, the expert partial is ``batch``. Keyed on the padded
    shape alone they collide and the layer returns a 32-row tensor; every test
    that compares a path against itself still passes.
    """
    cfg = fixture.config
    prompt = 32
    kv = MC.create_mesh_kv_cache(fixture.mesh, cfg, batch, 128, block_size=BLOCK_SIZE)
    hidden = _hidden(fixture.hf, prompt + 1)
    for user in range(batch):
        fixture.multichip_prefill(hidden[:, :, :prompt, :], kv_cache=kv, user_id=user)

    token = fixture.rep(hidden[:, :, prompt, :].reshape(1, 1, 1, -1).repeat(1, 1, batch, 1))
    pos = ttnn.from_torch(
        torch.full((batch,), prompt, dtype=torch.int32),
        dtype=ttnn.int32,
        device=fixture.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(fixture.mesh),
    )
    out = MC.decoder_layer_decode_multichip(
        token, fixture.multichip, cfg, fixture.ctx, fixture.cos, fixture.sin, kv, pos, prompt
    )
    assert list(out.shape) == list(token.shape), (
        f"decode layer changed the logical shape: in {list(token.shape)}, out {list(out.shape)}. "
        "48 of these stack, so the output contract must equal the input contract."
    )
    assert out.dtype == token.dtype, f"dtype changed: {token.dtype} -> {out.dtype}"
    assert out.layout == token.layout, f"layout changed: {token.layout} -> {out.layout}"
    assert out.memory_config() == token.memory_config(), (
        f"memory config changed: {token.memory_config()} -> {out.memory_config()}; "
        "there must be no inter-layer reshard"
    )
    spread = (fixture.dies(out) - fixture.dies(out)[0:1]).abs().max().item()
    assert spread == 0.0, f"decode output differs across dies by {spread}"


# --- the dynamic-nnz hazard ---------------------------------------------------


@ring_fabric
@mesh_4
def test_expert_window_can_be_empty(fixture):
    """A die holding none of the global top-8 must contribute an exact zero.

    Constructed rather than hoped for: the router weight keeps amplified real
    rows for experts 0..31 and zeroed rows for 32..127, so the routing collapses
    onto the low end of the expert range and at least two dies end up holding
    none of the global top-8. Measured, the split is **[6, 2, 0, 0]** -- die 0
    full, die 1 partial, dies 2 and 3 empty -- which is a better test than a
    clean [8,0,0,0] would have been, because it exercises both hazards at once:

    * ``E_local = 0``, where a host-computed ``nnz`` would be 8 against zero live
      sparsity entries and where an uninitialised ``sparse_matmul`` output would
      put a NaN into the all-reduce and poison the layer on *every* die;
    * ``0 < E_local < top_k``, the ordinary EP case, which is data-dependent and
      is exactly why no single ``nnz`` can be computed on the host for a program
      that runs on four dies at once.

    So the assertions are on the *properties*, not on the exact split: the live
    counts must sum to top-8 (the windows are a partition), at least one die must
    be empty (the hazard is reached), every empty die must contribute an exact
    zero, and the whole layer must still match the single-chip baseline.
    """
    cfg = fixture.config
    hf = fixture.hf
    n_local = cfg.local_moe.num_experts

    # Amplified real router rows for experts 0..31, zeroed rows for 32..127. The
    # zero rows give those experts a logit of exactly 0, so they only win a slot
    # when fewer than 8 of the first 32 come out positive -- which is what
    # produces the [6, 2, 0, 0] split rather than [8, 0, 0, 0].
    #
    # **The gain is 4x, and that number is a hazard, not a taste.** A first
    # version of this test used 10x *synthetic* rows, which -- against the
    # rms-normed activation, not the 0.02-scaled raw hidden -- produces logits
    # with a standard deviation near 450 and a top-8 spread past 1000.
    # ``exp(-1000)`` is exactly zero in bf16, so two of the eight routing weights
    # underflowed, ``count_nonzero(sparsity)`` fell below the ``nnz = top_k *
    # batch`` that the *single-chip* baseline passes, and the board deadlocked
    # exactly as ``sparse_matmul_device_operation.cpp:205-211`` says it will --
    # it had to be killed and reset. The multichip leg, on ``nnz=None``, was
    # unaffected. That is this stage's own reproduction of the hazard the design
    # phase only read about, and it is recorded in ``work_log.md``. At 4x on the
    # real rows the top-8 spread is a few units and every weight stays normal.
    forced = torch.zeros(cfg.global_config.moe.num_experts, hf.hidden_size)
    forced[:n_local] = 4.0 * fixture.torch_router[:n_local]
    forced_router = _replicate(
        forced.T.contiguous().reshape(1, 1, hf.hidden_size, cfg.global_config.moe.num_experts), fixture.mesh
    )

    x = fixture.rep(_hidden(hf, 1))
    routing = MC.router_forward_multichip(
        x, forced_router, fixture.multichip.expert_window, cfg.global_config.moe, cfg.local_moe
    )
    live = [int((fixture.dies(routing)[d].reshape(-1) > 0).sum()) for d in range(cfg.num_devices)]
    logger.info(f"forced routing: live experts per die = {live} (sum {sum(live)})")
    assert sum(live) == cfg.global_config.moe.num_experts_per_tok, (
        f"the four windows hold {sum(live)} experts, not the global top-"
        f"{cfg.global_config.moe.num_experts_per_tok}: {live}"
    )
    empty = [d for d in range(cfg.num_devices) if live[d] == 0]
    assert empty, f"the forced routing did not empty any die: {live}"

    partial = fixture.dies(MC.moe_decode_multichip(x, routing, fixture.multichip.experts, cfg.local_moe)).float()
    assert torch.isfinite(partial).all(), "an empty expert window produced non-finite output"
    empty_max = partial[empty].abs().max().item()
    logger.info(f"empty-window partials: max |value| on dies {empty} = {empty_max}")
    assert empty_max == 0.0, f"a die with no live experts contributed {empty_max}, not an exact zero"

    forced_baseline = F.DecoderLayerWeights(
        input_layernorm=fixture.multichip.input_layernorm,
        post_attention_layernorm=fixture.multichip.post_attention_layernorm,
        attention=None,
        router=forced_router,
        experts=None,
    )
    forced_multichip = MC.MultichipWeights(
        input_layernorm=fixture.multichip.input_layernorm,
        post_attention_layernorm=fixture.multichip.post_attention_layernorm,
        router=forced_router,
        expert_window=fixture.multichip.expert_window,
        experts=fixture.multichip.experts,
    )
    hidden = _hidden(hf, 128)
    base = fixture.dies(
        O.decoder_layer_prefill_optimized(
            fixture.rep(hidden),
            forced_baseline,
            cfg.global_config,
            fixture.cos,
            fixture.sin,
            fixture.baseline_sparsity,
            fixture.baseline_experts,
        )
    )[0:1].float()
    multi = fixture.dies(
        MC.decoder_layer_prefill_multichip(
            fixture.rep(hidden),
            forced_multichip,
            cfg,
            fixture.ctx,
            fixture.cos,
            fixture.sin,
            fixture.sparsity,
        )
    )[0:1].float()
    passing, message = comp_pcc(base, multi, PCC_VS_SINGLE_CHIP)
    logger.info(f"layer under maximally unbalanced routing vs single-chip: {message}")
    assert passing, message


# --- stacking and trace -------------------------------------------------------


@ring_fabric
@mesh_4
def test_stacked_layer_io_contract(fixture):
    """The layer's output must be usable as its own input, unmodified.

    Stage 04 stacks 48 of these. The contract is a replicated
    ``[1, 1, B, 2048]`` in DRAM in both directions, so feeding the output
    straight back in must work with no gather, reshard, layout change or dtype
    cast in between -- and must still match the single-chip baseline stacked the
    same way, which is what rules out a per-layer boundary conversion hiding
    inside the comparison.
    """
    hidden = _hidden(fixture.hf, 128)

    base_out = fixture.baseline_prefill(hidden)
    base_out2 = O.decoder_layer_prefill_optimized(
        base_out,
        fixture.baseline,
        fixture.config.global_config,
        fixture.cos,
        fixture.sin,
        fixture.baseline_sparsity,
        fixture.baseline_experts,
    )

    multi_out = fixture.multichip_prefill(hidden)
    assert multi_out.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert multi_out.layout == ttnn.TILE_LAYOUT and multi_out.dtype == ttnn.bfloat16
    multi_out2 = MC.decoder_layer_prefill_multichip(
        multi_out,
        fixture.multichip,
        fixture.config,
        fixture.ctx,
        fixture.cos,
        fixture.sin,
        fixture.sparsity,
    )
    assert multi_out2.shape == multi_out.shape
    assert multi_out2.memory_config() == multi_out.memory_config()
    assert multi_out2.dtype == multi_out.dtype and multi_out2.layout == multi_out.layout

    passing, message = comp_pcc(
        fixture.dies(base_out2)[0:1].float(), fixture.dies(multi_out2)[0:1].float(), PCC_VS_SINGLE_CHIP
    )
    logger.info(f"two stacked multichip layers vs two stacked single-chip layers: {message}")
    assert passing, f"stacked layers diverge: {message}"


@ring_fabric
@mesh_4
def test_multichip_decode_is_traceable(fixture):
    """Warmed trace capture and replay on the mesh, with a live input buffer.

    Trace capture and CCL interact: global semaphores and any persistent CCL
    buffer must exist before ``begin_trace_capture`` and nothing may allocate
    inside it. ``MeshContext`` allocates its semaphores at construction and the
    ``_ones_column`` constant is populated by the eager warm-up call below, which
    is what makes the capture legal.
    """
    cfg = fixture.config
    prompt = 32
    full = _hidden(fixture.hf, prompt + 1)
    kv = MC.create_mesh_kv_cache(fixture.mesh, cfg, 1, MAX_SEQ, block_size=BLOCK_SIZE)
    fixture.multichip_prefill(full[:, :, :prompt, :], kv_cache=kv)

    tt_in = fixture.rep(full[:, :, prompt : prompt + 1, :])
    pos = ttnn.from_torch(
        torch.tensor([prompt], dtype=torch.int32),
        dtype=ttnn.int32,
        device=fixture.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(fixture.mesh),
    )

    def step():
        return MC.decoder_layer_decode_multichip(
            tt_in, fixture.multichip, cfg, fixture.ctx, fixture.cos, fixture.sin, kv, pos, prompt
        )

    eager = fixture.dies(step()).clone()
    ttnn.synchronize_device(fixture.mesh)

    trace_id = ttnn.begin_trace_capture(fixture.mesh, cq_id=0)
    traced_out = step()
    ttnn.end_trace_capture(fixture.mesh, trace_id, cq_id=0)

    ttnn.execute_trace(fixture.mesh, trace_id, cq_id=0, blocking=True)
    replayed = fixture.dies(traced_out).clone()
    passing, message = comp_pcc(eager.float(), replayed.float(), 0.999)
    logger.info(f"multichip traced decode vs eager: {message}")
    assert passing, f"traced replay disagrees with eager: {message}"

    other = torch.randn(1, 1, 1, fixture.hf.hidden_size) * 0.02
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            other,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(fixture.mesh),
        ),
        tt_in,
    )
    ttnn.execute_trace(fixture.mesh, trace_id, cq_id=0, blocking=True)
    changed = fixture.dies(traced_out).clone()
    delta = (replayed.float() - changed.float()).abs().max().item()
    logger.info(f"multichip traced replay delta after input swap = {delta:.6f}")
    assert delta > 1e-3, "the mesh trace is not reading the live input buffer"

    ttnn.release_trace(fixture.mesh, trace_id)


@ring_fabric
@mesh_4
def test_multichip_decode_stress_is_deterministic(fixture):
    """Repeated decode at the same position must be bit-identical, run to run.

    The collectives are the new source of non-determinism here: two async CCLs
    per layer with cycling semaphores, on a ring where four dies race to the same
    reduction. Reduction order is fixed by the topology, so the result must be
    exactly repeatable; a drift would mean the semaphore cycling is letting two
    collectives overlap.
    """
    cfg = fixture.config
    prompt = 32
    full = _hidden(fixture.hf, prompt + 1)
    kv = MC.create_mesh_kv_cache(fixture.mesh, cfg, 1, MAX_SEQ, block_size=BLOCK_SIZE)
    fixture.multichip_prefill(full[:, :, :prompt, :], kv_cache=kv)
    pos = ttnn.from_torch(
        torch.tensor([prompt], dtype=torch.int32),
        dtype=ttnn.int32,
        device=fixture.mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(fixture.mesh),
    )
    token = fixture.rep(full[:, :, prompt : prompt + 1, :])

    outs = []
    for _ in range(20):
        outs.append(
            fixture.dies(
                MC.decoder_layer_decode_multichip(
                    token, fixture.multichip, cfg, fixture.ctx, fixture.cos, fixture.sin, kv, pos, prompt
                )
            ).clone()
        )
    for i, o in enumerate(outs[1:], start=1):
        assert torch.equal(outs[0], o), f"decode iteration {i} differs from iteration 0 (bitwise)"
    logger.info("multichip decode: 20 consecutive steps bit-identical across all 4 dies")


# --- runtime fallback audit ---------------------------------------------------


@ring_fabric
@mesh_4
@pytest.mark.parametrize("batch", [1, 32], ids=["b1", "b32"])
def test_no_runtime_fallbacks(fixture, batch):
    """None of the imported single-chip helpers may quietly take a slower path.

    All three of them see different inputs under TP/EP than they were tuned
    against, and all three fall back *silently* -- a PCC test cannot tell the
    difference. In particular ``_dram_sharded_ok`` needs both weight dims
    divisible by ``8 banks x 32 = 256``, and per-die wqkv N is 1280 = 5x256, one
    factor of two from failing; if it ever did, stage 02's 1.11x DRAM-sharded
    decode attention would disappear with no error at all.
    """
    audit = MC.fallback_audit(fixture.multichip, fixture.config, batch)
    logger.info(f"multichip fallback audit at batch {batch}: {audit}")
    assert audit["dram_sharded_taken"], "decode attention fell back to the interleaved path"
    assert audit["dram_sharded_qkv"] == (2048, 1280), audit
    assert audit["dram_sharded_wo"] == (1024, 2048), audit
    # **Literals, not the module constants.** ``EXPERT_IN0_BLOCK_W_*`` are now
    # derived from ``DEFAULT_PRECISION``, which is the same value the audit
    # resolves from -- so comparing them was an identity that could not fail and
    # could no longer catch a width regression. These are the widths stage 07
    # selected and measured; changing the default must fail here and be
    # re-measured, not silently ratified. ``test_precision_config.py`` pins the
    # same two literals through the full construction path.
    assert audit["gate_up_in0_block_w"] == 64, "gate/up block width moved off the stage-07 selection"
    assert audit["down_in0_block_w"] == 24, "down block width moved off the stage-07 selection"
    assert (O.EXPERT_IN0_BLOCK_W_GATE_UP, O.EXPERT_IN0_BLOCK_W_DOWN) == (
        64,
        24,
    ), "the module constants no longer agree with the selected widths"
    assert audit["local_heads"] == (8, 1) and audit["local_experts"] == 32, audit
    # Batch 1 is the latency target and must keep the intermediates in L1 -- the
    # traced A/B says L1 is 7.6% faster there. Batch 32 must not: the allocator
    # refuses 234.88 MB outright (bank_manager.cpp:462), so a budget that let it
    # through would be a crash, not a slow path. The inherited 40 MB constant
    # would have separated these two correctly by luck while silently changing
    # the answer for batches 2 to 16; the swept 128 MB is chosen against the
    # measured L1-vs-DRAM crossover. See probes/l1_budget_probe.py.
    assert audit["expert_intermediate_buffer"] == ("L1" if batch == 1 else "DRAM"), audit
    # Stage 04. The sharded residual norm writes exactly the L1 shard the
    # DRAM-sharded qkv projection reads, which is what lets the first norm's
    # output cross into attention with no conversion at all. If that equality
    # ever breaks, TTNN inserts a reshard between them and the layer gets slower
    # with no error -- the same failure mode as the three above.
    assert audit["norm_shard_feeds_qkv_directly"], (
        "the sharded norm's output shard no longer matches attention's qkv input shard; "
        "a silent reshard has been reintroduced between them"
    )
    assert audit["norm_shard_cores"] == 8, audit


def test_meta_rope_weights_match_hf():
    """The Meta channel permutation is a *pair*: Q/K rows and the QK-norm vectors.

    Stage 01 chose HF-style RoPE precisely so neither permutation was needed.
    Stage 04 adopts ``rotary_embedding_llama`` on the decode path, which brings
    both back, and crossing them "runs fine and silently produces garbage"
    (``weight_mapping.py``). This asserts the whole convention on the host, with
    no device, so a mismatch fails here rather than as a PCC that is merely
    lower.
    """
    import torch

    from ..tt.weight_mapping import hf_to_meta_channels, permute_head_vector_to_meta, permute_wqkv_to_meta

    hd, nh, nkv, hidden = 128, 8, 1, 2048
    perm = hf_to_meta_channels(hd)
    inv = torch.argsort(perm)

    # 1. The permutation is a permutation, and it is the interleave it claims.
    assert sorted(perm.tolist()) == list(range(hd))
    assert perm[0] == 0 and perm[1] == hd // 2 and perm[2] == 1

    # 2. HF rope on HF-ordered channels == Meta rope on Meta-ordered channels.
    torch.manual_seed(0)
    x = torch.randn(4, hd)
    c, s = torch.randn(hd // 2).abs(), torch.randn(hd // 2)
    cos_hf, sin_hf = torch.cat([c, c]), torch.cat([s, s])
    hf = x * cos_hf + torch.cat([-x[:, hd // 2 :], x[:, : hd // 2]], dim=-1) * sin_hf
    xm = x[:, perm]
    cos_m, sin_m = cos_hf[perm], sin_hf[perm]
    rot = torch.stack([-xm[:, 1::2], xm[:, 0::2]], dim=-1).reshape(xm.shape)
    meta = xm * cos_m + rot * sin_m
    assert torch.allclose(meta[:, inv], hf, atol=1e-6), (meta[:, inv] - hf).abs().max()

    # 3. permute_wqkv_to_meta touches Q and K and leaves V alone.
    wqkv = torch.randn(1, 1, hidden, (nh + 2 * nkv) * hd)
    out = permute_wqkv_to_meta(wqkv, n_heads=nh, n_kv_heads=nkv, head_dim=hd)
    assert out.shape == wqkv.shape
    v0 = (nh + nkv) * hd
    assert torch.equal(out[..., v0:], wqkv[..., v0:]), "V was permuted"
    for h in range(nh + nkv):
        lo = h * hd
        assert torch.equal(out[..., lo : lo + hd], wqkv[..., lo : lo + hd][..., perm])
    assert not torch.equal(out[..., :hd], wqkv[..., :hd]), "Q was not permuted"

    # 4. Applying it twice is not identity -- i.e. forgetting it is detectable.
    vec = torch.randn(hd)
    assert not torch.equal(permute_head_vector_to_meta(vec, head_dim=hd), vec)
    assert torch.equal(permute_head_vector_to_meta(vec, head_dim=hd)[inv], vec)
