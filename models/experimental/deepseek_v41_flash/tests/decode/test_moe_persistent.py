# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The persistent TP12 MoE: one ``fused_experts`` launch serving several layers over D2D sockets.

On the 8x4 mesh, the first five rows are left to attention and the other ops, and the MoE takes the last three
(a 3x4 submesh at row 5). Three layers of random weights at V4.1's shapes (D = 5120, I = 2304, 384 experts, top
6) are loaded on it, each split TP12 as the model splits it, and the op is launched once. It serves three
requesters, TP4 rows above it (1x4 submeshes at rows 2, 3 and 4), in strict turn: each request
(``persistent_moe_request``) sends a ``[64-byte header | bf16 row]`` page from each of the requester's chips down
its column to the three MoE chips below, the header's first word naming the layer. The MoE chips run that
layer's whole MoE (router, routed experts, shared expert) in their three-stage pipeline, so the requesters'
rows overlap, and reduce it onto MoE coordinate (0, 0), which sends the row back up to all four of the
requester's chips, so the request op's output is replicated. The layers are visited interleaved. Each step is
checked against a torch reference and against the same layer's non-persistent op on the same row, and then
every requester's stop request ends the kernels so the device closes normally.

``test_moe_persistent_tp12_all_layers`` does the same with every one of the checkpoint's layers, real weights,
checked against the checkpoint's reference MoE; fill the TP12 expert cache first (see its docstring).

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe_persistent.py
"""

import itertools
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.reference import load_reference, reference_moe
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    expert_provider,
    load_config,
    moe_weights,
)
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts, DeepSeekV41SparseMoeBlock
from models.experimental.deepseek_v41_flash.tt.decode.persistent_moe import PersistentMoE, default_socket_core
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

PCC = 0.95
# The persistent op runs the one-shot op's expert kernels, so the two agree to rounding wherever they pick the same
# experts. Its router is its own (sharded over the chips, its partial sums added in fp32), so on a near-tie -- a step
# whose torch reference offers alternative top-k sets (ROUTING_TIE) -- the two may pick different experts, and the
# step is held to the torch check alone.
PCC_VS_OP = 0.9999
PARENT_MESH = (8, 4)
MOE_SUBMESH = (3, 4)
MOE_OFFSET = (5, 0)
# The requesters: TP4 rows above the MoE, each chip's requests going down its column to the three MoE chips
# below. Row 4's multicast; rows 2 and 3 unicast to each MoE chip, past the rows in between. The MoE root's
# responses all leave it northward, its one free fabric direction, which puts rows 1 to 4 in reach.
REQUEST_SUBMESH = (1, 4)
REQUEST_ROWS = (2, 3, 4)
TP_SIZE = 12
NUM_LAYERS = 3
# The layer of each step: every layer is visited more than once, never in a fixed order.
STEPS = (0, 2, 1, 1, 0, 2, 0, 1, 2)
# Distinct base experts per layer; expert e is base e % EXPERT_POOL scaled by its own factor, which keeps every
# expert distinct without generating 384 full-size random experts per layer.
EXPERT_POOL = 4
# Biased router scores within this of the k-th one, or within one bf16 ulp of it if wider, are a tie the device's
# bf16 router may break either way; the torch reference accepts any top-k drawn from that band. The random
# layers' scores are ~1 (ulp ~ 0.008); the real layers' biases put theirs at 9 to 60 (ulp 0.06 to 0.25).
ROUTING_TIE = 0.02
# The real-weights test: rows per layer, every layer of the checkpoint visited this often in a shuffled order.
REAL_ROWS_PER_LAYER = 2


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _random_layer(config, seed):
    """``(provider, weights)`` in the shapes ``expert_provider`` / ``moe_weights`` return, random."""
    g = torch.Generator().manual_seed(seed)
    d, i, e = config.hidden_size, config.moe_intermediate_size, config.num_local_experts
    pool_gate_up = [torch.randn(2 * i, d, generator=g) * d**-0.5 for _ in range(EXPERT_POOL)]
    pool_down = [torch.randn(d, i, generator=g) * i**-0.5 for _ in range(EXPERT_POOL)]

    def provider(x: int):
        scale = 1.0 + x / e
        return pool_gate_up[x % EXPERT_POOL] * scale, pool_down[x % EXPERT_POOL] * scale

    weights = {
        "gate.weight": torch.randn(e, d, generator=g) * d**-0.5,
        "gate.e_score_correction_bias": torch.randn(e, generator=g) * 0.1,
        "shared_experts.gate_proj.weight": torch.randn(i, d, generator=g) * d**-0.5,
        "shared_experts.up_proj.weight": torch.randn(i, d, generator=g) * d**-0.5,
        "shared_experts.down_proj.weight": torch.randn(d, i, generator=g) * i**-0.5,
    }
    return provider, weights


def _reference(x, provider, weights, config, experts):
    """The MoE of one row ``x`` ``[D]`` in fp32: the ``sqrtsoftplus`` router ranked on the biased scores, the
    renormalized top-k experts, and the shared expert, all with the ``swiglu_limit`` clamp.

    Returns ``(candidates, margin)``. ``candidates[0]`` is the output for the exact top-k; the rest are the
    outputs for every other top-k the device could pick by breaking a tie (``ROUTING_TIE``) differently. ``margin``
    is the gap between the k-th and (k+1)-th ranked score."""
    k = experts.top_k
    gate_w = weights["gate.weight"] / getattr(config, "gate_temp", 1.0)
    scores = torch.sqrt(torch.nn.functional.softplus(gate_w @ x))
    biased = scores + weights["gate.e_score_correction_bias"]
    order = biased.argsort(descending=True).tolist()
    margin = (biased[order[k - 1]] - biased[order[k]]).item()
    boundary = biased[order[k - 1]].item()
    tie = max(ROUTING_TIE, 2.0 ** (math.floor(math.log2(abs(boundary))) - 7))
    band = [e for e in order if abs(biased[e].item() - boundary) <= tie]
    sure = [e for e in order[:k] if e not in band]
    limit = experts.limit
    i = config.moe_intermediate_size

    def swiglu(gate, up, down):
        act = torch.nn.functional.silu((gate @ x).clamp(max=limit)) * (up @ x).clamp(-limit, limit)
        return down @ act

    shared = swiglu(
        weights["shared_experts.gate_proj.weight"],
        weights["shared_experts.up_proj.weight"],
        weights["shared_experts.down_proj.weight"],
    )
    routed = {}
    for e in sure + band:
        gate_up, down = provider(e)
        routed[e] = swiglu(gate_up[:i], gate_up[i:], down)

    # ``band`` is in rank order, so the first combination completes the exact top-k.
    candidates = []
    for chosen in itertools.combinations(band, k - len(sure)):
        ids = sure + list(chosen)
        w = scores[ids]
        w = experts.routed_scaling_factor * w / (w.sum() + experts.routing_eps)
        candidates.append(shared + sum(wi * routed[e] for wi, e in zip(w, ids)))
    return candidates, margin


def _single_senders(rows):
    """Per requester row, whether one of its chips sends every request to all the MoE chips (PersistentMoE's
    ``single_senders``): one multicast does it only from the row right above the MoE."""
    return [row == MOE_OFFSET[0] - 1 for row in rows]


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def _setup(mesh_device):
    """``(moe_mesh, request_meshes, config, socket_core, request_core)``: the TP12 MoE submesh, the TP4 requester
    submeshes, the V4.1 config, and the cores the sockets connect."""
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    moe_mesh = mesh_device.create_submesh(ttnn.MeshShape(*MOE_SUBMESH), ttnn.MeshCoordinate(*MOE_OFFSET))
    request_meshes = [
        mesh_device.create_submesh(ttnn.MeshShape(*REQUEST_SUBMESH), ttnn.MeshCoordinate(row, 0))
        for row in REQUEST_ROWS
    ]
    core = default_socket_core(moe_mesh)
    return moe_mesh, request_meshes, load_config(_snapshot()), core, core


def _load_layers(device, config):
    """The ``NUM_LAYERS`` random layers: ``(layers, moes)``, ``layers[i]`` the torch ``(provider, weights)``."""
    layers = []
    moes = []
    for layer in range(NUM_LAYERS):
        start = time.perf_counter()
        provider, weights = _random_layer(config, seed=layer)
        # The random weights are a function of the seed alone, so their converted tensors are cached like
        # real ones: only the first run pays the bfloat4_b conversion of 384 experts per layer.
        cache = WeightCache(CACHE_DIR).sub(f"persistent_test.random{layer}.ffn")

        def loading_provider(x, provider=provider, layer=layer):
            # Called only on a weight-cache miss, once per expert being converted.
            if x % 64 == 0:
                logger.info(f"layer {layer}: converting experts {x}..{x + 63}")
            return provider(x)

        experts = DeepSeekV41PreloadedExperts(
            config, loading_provider, device, dtype=ttnn.bfloat4_b, cache=cache, tp_size=TP_SIZE
        )
        moe = DeepSeekV41SparseMoeBlock(
            config, {k: (lambda v=v: v) for k, v in weights.items()}, device, experts, cache=cache, tp_size=TP_SIZE
        )
        assert moe.fused_router and moe.fused_shared is not None, "the fused TP12 MoE must run router and shared"
        layers.append((provider, weights))
        moes.append(moe)
        logger.info(f"layer {layer} loaded in {time.perf_counter() - start:.1f}s (cache {CACHE_DIR})")
    return layers, moes


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_moe_persistent_tp12(mesh_device):
    moe_mesh, request_meshes, config, socket_core, request_core = _setup(mesh_device)
    layers, moes = _load_layers(moe_mesh, config)
    num_requesters = len(request_meshes)

    torch.manual_seed(0)
    hidden = torch.randn(len(STEPS), config.hidden_size).to(torch.bfloat16).float()
    candidates, margins = zip(
        *(_reference(hidden[s], *layers[layer], config, moes[layer].experts) for s, layer in enumerate(STEPS))
    )

    # The one-shot op on the same rows, before the persistent op takes over the cores.
    hidden_config = width_sharded_l1_config(1, config.hidden_size, moe_mesh, tile_height=1)
    one_shot = []
    for s, layer in enumerate(STEPS):
        out = moes[layer].decode_static(
            ttnn.from_torch(
                hidden[s].reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=moe_mesh,
                memory_config=hidden_config,
            )
        )
        one_shot.append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1).float())
        ttnn.deallocate(out)
    ttnn.synchronize_device(moe_mesh)

    # Step s comes from requester s % num_requesters; each round every requester's request is in flight at once.
    moe_op = PersistentMoE(moe_mesh, request_meshes, moes, socket_core, request_core, _single_senders(REQUEST_ROWS))
    outs = []
    try:
        for first in range(0, len(STEPS), num_requesters):
            round_steps = range(first, min(first + num_requesters, len(STEPS)))
            pending = []
            for s in round_steps:
                moe_op.upload(s % num_requesters, hidden[s])
                pending.append(moe_op.request(s % num_requesters, STEPS[s]))
            for s, out in zip(round_steps, pending):
                outs.append(PersistentMoE.row_of(out).float())
                ttnn.deallocate(out)
                logger.info(
                    f"step {s} layer {STEPS[s]} requester {s % num_requesters}: "
                    f"PCC {_pcc(one_shot[s], outs[-1])} vs the one-shot op"
                )
        hc_pccs = _hc_post_round(moe_op, hidden, outs, num_requesters) if moe_op.fuse_hc else [1.0]
    finally:
        moe_op.stop()

    # Every step in one report: the one-shot op against torch tells a reference / routing mismatch (both ops
    # off torch alike) apart from a persistent-op bug (persistent off the one-shot op).
    report = []
    failed = False
    for s, layer in enumerate(STEPS):
        pccs = [_pcc(c, outs[s]) for c in candidates[s]]
        best = max(range(len(pccs)), key=pccs.__getitem__)
        pcc_torch = pccs[best]
        pcc_one_shot_torch = max(_pcc(c, one_shot[s]) for c in candidates[s])
        pcc_op = _pcc(one_shot[s], outs[s])
        near_tie = len(candidates[s]) > 1
        failed |= pcc_torch < PCC or (pcc_op < PCC_VS_OP and not near_tie)
        routing = "exact top-k" if best == 0 else f"tie-broken top-k {best} of {len(pccs) - 1} alternatives"
        report.append(
            f"step {s} layer {layer}: persistent vs torch {pcc_torch:.5f} ({routing}; exact top-k {pccs[0]:.5f}), "
            f"one-shot vs torch {pcc_one_shot_torch:.5f}, persistent vs one-shot {pcc_op:.6f}"
            f"{' (near-tie, not checked)' if near_tie else ''}, top-k margin {margins[s]:.2e}"
        )
    logger.info("\n" + "\n".join(report))
    assert not failed, f"PCC below {PCC} vs torch, or {PCC_VS_OP} vs the one-shot op off a near-tie:\n" + "\n".join(
        report
    )
    assert min(hc_pccs) >= HC_PCC, f"fused HC post step: PCC {hc_pccs} vs mix_streams' math on the plain output"


HC_STREAMS = 4
HC_PCC = 0.9999


def _hc_post_round(moe_op, hidden, outs, num_requesters) -> list:
    """Steps 0 .. num_requesters - 1 again, one per requester, with the hyper-connection post step fused into the
    request: each requester's new streams against ``post[h] * out + comb.T @ streams`` on that step's plain output."""
    pccs = []
    pending = []
    for s in range(num_requesters):
        mesh = moe_op.request_meshes[s]
        streams = torch.randn(1, 1, HC_STREAMS, hidden.shape[1]).to(torch.bfloat16)
        post = (2 * torch.rand(1, 1, HC_STREAMS, 1)).to(torch.bfloat16)
        comb = torch.softmax(torch.randn(1, 1, HC_STREAMS, HC_STREAMS), dim=-1).to(torch.bfloat16)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        tt_streams = ttnn.from_torch(
            streams,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=moe_op.streams_memory_config(mesh, HC_STREAMS),
            mesh_mapper=rep,
        )
        tt_post, tt_comb = (
            ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
            for t in (post, comb)
        )
        moe_op.upload(s, hidden[s])
        out = moe_op.request(s, STEPS[s], streams=tt_streams, post=tt_post, comb=tt_comb)
        x = outs[s].reshape(1, -1)
        mix = comb[0, 0].float().T @ streams[0, 0].float()
        expected = post[0, 0].float() * x + mix
        pending.append((s, out, expected, (tt_streams, tt_post, tt_comb), (x, mix, post, comb, streams)))
    dump = []
    for s, out, expected, inputs, terms in pending:
        got = None
        for t in ttnn.get_device_tensors(out):
            got = ttnn.to_torch(t).reshape(expected.shape).float()
            pccs.append(_pcc(expected, got))
        # Which term is off: what the device's output implies for x and for the mix, given the other.
        x, mix, post, _, _ = terms
        p = post[0, 0].float()
        x_implied = (got - mix) / p
        mix_implied = got - p * x
        logger.info(
            f"step {s} with the HC post step fused: PCC {pccs[-1]:.6f} (last chip); implied x vs x per stream "
            f"{[round(_pcc(x_implied[h], x[0]), 5) for h in range(got.shape[0])]}, implied mix vs mix per stream "
            f"{[round(_pcc(mix_implied[h], mix[h]), 5) for h in range(got.shape[0])]}, |x| {x.abs().mean():.3e}, "
            f"|mix| {mix.abs().mean():.3e}, |err| {(got - expected).abs().mean():.3e}"
        )
        dump.append({"got": got, "expected": expected, "terms": terms})
        ttnn.deallocate(out)
        for t in inputs:
            ttnn.deallocate(t)
    if min(pccs) < HC_PCC:
        path = os.path.join("generated", "hc_post_dump.pt")
        os.makedirs("generated", exist_ok=True)
        torch.save(dump, path)
        logger.info(f"fused HC post step below {HC_PCC}: tensors saved to {path}")
    return pccs


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_moe_persistent_tp12_all_layers(mesh_device):
    """Every decoder layer's real MoE in the one persistent launch.

    All of the checkpoint's layers are loaded at TP12 on the MoE submesh, the routed experts K-split over every
    DRAM bank as the model reads them (~0.64 GB per chip per layer). Fill the cache first, or the experts are
    converted here one after another:

        python models/experimental/deepseek_v41_flash/tools/fill_expert_cache.py --all-layers --tp-size 12 --mesh-rows 3

    ``REAL_ROWS_PER_LAYER`` random rows per layer then go through the persistent op in a shuffled layer order,
    from the requesters in turn, each checked against the one-shot op on the same row and against the torch
    reference, which takes any top-k the bf16 router could pick from a near-tie (``_reference``) and has to match
    the checkpoint's own reference MoE on the exact top-k."""
    moe_mesh, request_meshes, config, socket_core, request_core = _setup(mesh_device)
    snapshot = _snapshot()
    loader = DeepseekV4WeightLoader(snapshot)
    num_layers = config.num_hidden_layers
    num_requesters = len(request_meshes)

    moes = []
    for layer in range(num_layers):
        start = time.perf_counter()
        cache = WeightCache(CACHE_DIR).sub(f"layers.{layer}.ffn")
        experts = DeepSeekV41PreloadedExperts(
            config,
            expert_provider(loader, layer),
            moe_mesh,
            dtype=ttnn.bfloat4_b,
            cache=cache,
            tp_size=TP_SIZE,
            balance_dram_banks=True,
        )
        moe = DeepSeekV41SparseMoeBlock(
            config, moe_weights(loader, layer), moe_mesh, experts, cache=cache, tp_size=TP_SIZE
        )
        assert (
            moe.fused_router and moe.fused_shared is not None
        ), f"layer {layer}: the fused TP12 MoE must run router and shared"
        moes.append(moe)
        logger.info(f"layer {layer} loaded in {time.perf_counter() - start:.1f}s")

    torch.manual_seed(0)
    steps = torch.cat([torch.randperm(num_layers) for _ in range(REAL_ROWS_PER_LAYER)]).tolist()
    hidden = torch.randn(len(steps), config.hidden_size).to(torch.bfloat16).float()

    # One layer's references at a time: their gates and shared experts are fp32, their experts loaded per call.
    model = load_reference(snapshot)
    expected = [None] * len(steps)
    candidates = [None] * len(steps)
    for layer in range(num_layers):
        reference = reference_moe(model, snapshot, loader, layer)
        weights = {k: v() for k, v in moe_weights(loader, layer).items()}
        provider = expert_provider(loader, layer)
        for s in (s for s, l in enumerate(steps) if l == layer):
            expected[s] = reference(hidden[s].reshape(1, 1, -1)).reshape(-1).float()
            candidates[s], _ = _reference(hidden[s], provider, weights, config, moes[layer].experts)
        del reference, weights

    hidden_config = width_sharded_l1_config(1, config.hidden_size, moe_mesh, tile_height=1)
    one_shot = []
    for s, layer in enumerate(steps):
        out = moes[layer].decode_static(
            ttnn.from_torch(
                hidden[s].reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=moe_mesh,
                memory_config=hidden_config,
            )
        )
        one_shot.append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1).float())
        ttnn.deallocate(out)
    ttnn.synchronize_device(moe_mesh)

    moe_op = PersistentMoE(moe_mesh, request_meshes, moes, socket_core, request_core, _single_senders(REQUEST_ROWS))
    outs = []
    try:
        for first in range(0, len(steps), num_requesters):
            round_steps = range(first, min(first + num_requesters, len(steps)))
            pending = []
            for s in round_steps:
                moe_op.upload(s % num_requesters, hidden[s])
                pending.append(moe_op.request(s % num_requesters, steps[s]))
            for out in pending:
                outs.append(PersistentMoE.row_of(out).float())
                ttnn.deallocate(out)
    finally:
        moe_op.stop()

    report = []
    failed = []
    for s, layer in enumerate(steps):
        pccs = [_pcc(c, outs[s]) for c in candidates[s]]
        best = max(range(len(pccs)), key=pccs.__getitem__)
        pcc_anchor = _pcc(expected[s], candidates[s][0])
        pcc_op = _pcc(one_shot[s], outs[s])
        near_tie = len(candidates[s]) > 1
        if pccs[best] < PCC or (pcc_op < PCC_VS_OP and not near_tie) or pcc_anchor < PCC_VS_OP:
            failed.append(s)
        routing = "exact top-k" if best == 0 else f"tie-broken top-k {best} of {len(pccs) - 1} alternatives"
        report.append(
            f"step {s} layer {layer} requester {s % num_requesters}: persistent vs torch {pccs[best]:.5f} ({routing}; "
            f"exact top-k {pccs[0]:.5f}), persistent vs one-shot {pcc_op:.6f}"
            f"{' (near-tie, not checked)' if near_tie else ''}, torch vs checkpoint reference {pcc_anchor:.6f}"
        )
    logger.info("\n" + "\n".join(report))
    assert not failed, (
        f"steps {failed}: PCC below {PCC} vs torch, or {PCC_VS_OP} vs the one-shot op off a near-tie or between torch "
        "and the checkpoint's reference"
    )


def _latency(name, seconds):
    us = torch.tensor(seconds, dtype=torch.float64) * 1e6
    return (
        f"{name:<44} median {us.median():8.1f} us   p10 {us.quantile(0.1):8.1f}   p90 {us.quantile(0.9):8.1f}"
        f"   (n={len(seconds)})"
    )


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 32 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_moe_persistent_tp12_perf(mesh_device):
    """Per-token MoE latency over the interleaved layers:

    - one-shot eager: upload, ``decode_static`` dispatched op by op, read chip 0;
    - one-shot traced: upload, replay that layer's trace, read chip 0 -- the model's decode today;
    - one-shot traced, device only: back-to-back replays with no host I/O, total / count;
    - persistent, with 1 requester and then with 3: rounds of one request per requester in flight at once, eager
      (uploads, ``persistent_moe_request`` dispatches, readbacks; per round), then traced and replayed back to back
      with no host I/O (per row). One requester leaves two of the MoE's three pipeline stages idle; three fill it.
      Each traced (requester, layer)'s output must match the eager loop's on the same row bit for bit.

    ``MOE_PERF_WARMUP`` / ``MOE_PERF_STEPS`` override the step counts, e.g. to keep a device-profiler capture
    (``TT_METAL_STREAMING_PROFILER=1``) small. ``MOE_PERF_PERSISTENT_ONLY=1`` skips the one-shot runs: the
    one-shot and persistent builds of a kernel share its profiler zone ids, so a capture holding both
    misnames zones.
    """
    moe_mesh, request_meshes, config, socket_core, request_core = _setup(mesh_device)
    _, moes = _load_layers(moe_mesh, config)
    hidden = config.hidden_size
    warmup = int(os.environ.get("MOE_PERF_WARMUP", 20))
    steps = int(os.environ.get("MOE_PERF_STEPS", 270))
    persistent_only = os.environ.get("MOE_PERF_PERSISTENT_ONLY") == "1"
    order = [STEPS[i % len(STEPS)] for i in range(warmup + steps)]
    torch.manual_seed(0)
    rows = torch.randn(len(order), hidden).to(torch.bfloat16)
    results = []
    if not persistent_only:
        hidden_config = width_sharded_l1_config(1, hidden, moe_mesh, tile_height=1)
        inputs = [
            ttnn.from_torch(
                torch.zeros(1, 1, 1, hidden),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=moe_mesh,
                memory_config=hidden_config,
            )
            for _ in range(NUM_LAYERS)
        ]

        def upload(i, layer):
            host = ttnn.from_torch(
                rows[i].reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(moe_mesh),
            )
            ttnn.copy_host_to_device_tensor(host, inputs[layer])

        eager = []
        for i, layer in enumerate(order):
            start = time.perf_counter()
            upload(i, layer)
            out = moes[layer].decode_static(inputs[layer])
            ttnn.to_torch(ttnn.get_device_tensors(out)[0])
            eager.append(time.perf_counter() - start)
            ttnn.deallocate(out)
        results.append(_latency("one-shot eager (upload + ops + readback)", eager[warmup:]))

        tids = []
        trace_outs = []
        for layer in range(NUM_LAYERS):
            tid = ttnn.begin_trace_capture(moe_mesh, cq_id=0)
            trace_outs.append(moes[layer].decode_static(inputs[layer]))
            ttnn.end_trace_capture(moe_mesh, tid, cq_id=0)
            tids.append(tid)
        ttnn.synchronize_device(moe_mesh)

        traced = []
        for i, layer in enumerate(order):
            start = time.perf_counter()
            upload(i, layer)
            ttnn.execute_trace(moe_mesh, tids[layer], cq_id=0, blocking=False)
            ttnn.to_torch(ttnn.get_device_tensors(trace_outs[layer])[0])
            traced.append(time.perf_counter() - start)
        results.append(_latency("one-shot traced (upload + replay + readback)", traced[warmup:]))

        for layer in order[:warmup]:
            ttnn.execute_trace(moe_mesh, tids[layer], cq_id=0, blocking=False)
        ttnn.synchronize_device(moe_mesh)
        start = time.perf_counter()
        for layer in order[warmup:]:
            ttnn.execute_trace(moe_mesh, tids[layer], cq_id=0, blocking=False)
        ttnn.synchronize_device(moe_mesh)
        per_replay = (time.perf_counter() - start) / steps
        results.append(f"{'one-shot traced, device only (back-to-back)':<44} mean {per_replay * 1e6:8.1f} us")

        for tid in tids:
            ttnn.release_trace(moe_mesh, tid)
        for t in trace_outs + inputs:
            ttnn.deallocate(t)
        ttnn.synchronize_device(moe_mesh)

    # Fewer requesters than stages leave the pipeline partly empty; each requester is still one request at a
    # time, waiting for its response.
    for num_requesters in (1, len(request_meshes)):
        meshes = request_meshes[-num_requesters:]
        moe_op = PersistentMoE(
            moe_mesh, meshes, moes, socket_core, request_core, _single_senders(REQUEST_ROWS[-num_requesters:])
        )
        mismatched = []
        name = f"persistent, {num_requesters} requester{'s' if num_requesters > 1 else ''}"
        try:
            # Row i comes from requester i % num_requesters, a round of requests in flight at once. The MoE
            # takes the requesters' turns in that order, so every run here has to be a prefix of it: the eager
            # rounds are whole, and the replays then start again at requester 0.
            eager = []
            eager_outs = []
            for first in range(0, len(order) - len(order) % num_requesters, num_requesters):
                start = time.perf_counter()
                pending = []
                for i in range(first, min(first + num_requesters, len(order))):
                    moe_op.upload(i % num_requesters, rows[i])
                    pending.append(moe_op.request(i % num_requesters, order[i]))
                for out in pending:
                    eager_outs.append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1))
                    ttnn.deallocate(out)
                eager.append(time.perf_counter() - start)
            results.append(
                _latency(f"{name}, eager (round of uploads + requests + readbacks)", eager[warmup // num_requesters :])
            )

            # One trace per (requester, layer) it serves, requesting the row of that pair's first step.
            firsts = {}
            for i, layer in enumerate(order):
                firsts.setdefault((i % num_requesters, layer), i)
            inputs = {}
            tids = {}
            trace_outs = {}
            for (j, layer), first in firsts.items():
                inputs[j, layer] = moe_op.request_input(j)
                moe_op.upload(j, rows[first], inputs[j, layer])
                tids[j, layer] = ttnn.begin_trace_capture(meshes[j], cq_id=0)
                trace_outs[j, layer] = moe_op.request(j, layer, inputs[j, layer])
                ttnn.end_trace_capture(meshes[j], tids[j, layer], cq_id=0)
            for mesh in meshes:
                ttnn.synchronize_device(mesh)

            def replay(indices):
                for i in indices:
                    j = i % num_requesters
                    ttnn.execute_trace(meshes[j], tids[j, order[i]], cq_id=0, blocking=False)
                for mesh in meshes:
                    ttnn.synchronize_device(mesh)

            replay(range(warmup))
            start = time.perf_counter()
            replay(range(warmup, len(order)))
            per_row = (time.perf_counter() - start) / steps
            results.append(f"{name + ', traced, device only (per row)':<44} mean {per_row * 1e6:8.1f} us")

            for (j, layer), first in firsts.items():
                copies = PersistentMoE.rows_of(trace_outs[j, layer])
                mismatched += [
                    (j, layer, c) for c, copy in enumerate(copies) if not torch.equal(copy, eager_outs[first])
                ]
            for key in tids:
                ttnn.release_trace(meshes[key[0]], tids[key])
            for t in list(trace_outs.values()) + list(inputs.values()):
                ttnn.deallocate(t)
        finally:
            moe_op.stop()
        assert not mismatched, f"{name}: traced (requester, layer, chip) outputs {mismatched} differ from eager"
        # The next launch's sockets and scratch take the same L1.
        del moe_op
        ttnn.synchronize_device(moe_mesh)

    logger.info("MoE TP12 per-token latency, layers interleaved " + str(STEPS) + ":\n" + "\n".join(results))
