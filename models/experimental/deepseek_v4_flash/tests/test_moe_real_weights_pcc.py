# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint PCC test for the ttnn DeepSeek-V4-Flash MoE block (prefill).

Unlike ``test_moe_pcc.py`` (reduced config + random weights, validated against
the HF reference via a subprocess), this test loads the *actual* V4-Flash
checkpoint weights for one ``moe`` decoder layer through
:class:`DeepseekV4WeightLoader`, dequantizes them on host, and compares:

* **reference**: a faithful pure-torch reimplementation of ``DeepseekV4SparseMoeBlock``
  (router -> routed experts -> ``+`` shared expert), run in fp32 on host, and
* **device**: the ttnn block (:class:`DeepSeekV4SparseMoeBlock` with a
  :class:`DeepSeekV4PreloadedExperts`, which keeps all 256 experts on device in
  BFloat4_b so they fit).

The device keeps all 256 routed experts resident in BFloat4_b (4-bit), so the
reference reads those exact quantised weights back for its routed-expert math;
the PCC gap then isolates the ttnn compute path rather than the deliberate 4-bit
storage choice. Everything runs in the ttnn venv — no cached transformers /
subprocess is needed because the reference is hand-written.

The routed experts are MXFP4 (int8-packed fp4 + e8m0 block scale); the shared
expert and the dense projections are block-FP8 (e4m3 + e8m0); the router gate is
bf16. See ``tt/quant.py`` for the dequantizers.

A second test, :func:`test_moe_real_weights_batched_fused_experts_pcc`, drives the
``fused_experts`` device op directly with B > 1 token rows in a single call (the
block's expert path still issues one op per token) to cover batched decode: per-token
correctness, equivalence with the per-token path, and the weight-fetch amortization
that comes from tokens sharing experts.
"""

from __future__ import annotations

import json
import os
import tempfile
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

from tracy import signpost
import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tt.moe import (
    DeepSeekV4PreloadedExperts,
    DeepSeekV4SparseMoeBlock,
    _ROUTING_EPS,
    _swiglu_cols_per_core,
)
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import (
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)

DEFAULT_MODEL_DIR = Path("/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731")

# Layers 0..2 are ``hash_moe`` (frozen tid2eid routing); 3+ are standard ``moe``.
MOE_LAYER = 5
PCC_THRESHOLD = 0.99

# BFloat4_b for the routed experts, as the model loads them: the cache key includes the dtype, so
# this has to match for the entries to be shared.
WEIGHT_DTYPE = ttnn.bfloat4_b

# On-disk cache for the converted ttnn weight tiles. Same root, namespace and dtype
# :class:`DeepSeekV4Model` uses (``<root>/ttnn`` + ``layers.N`` + ``mlp``), so entries written by the
# model or by ``test_decoder_layer_pcc`` for this layer are reused here and vice versa — turning the
# expensive dequantize-and-upload of the layer's experts into a straight tile read. Override the root
# with ``DEEPSEEK_V4_CACHE_DIR``, or set it empty to disable caching and always reconvert.
_CACHE_DIR = (
    os.environ.get("DEEPSEEK_V4_CACHE_DIR", os.path.join(tempfile.gettempdir(), "deepseek_v4_flash_cache")) or None
)

# One device for the whole module, so the layer's experts can be uploaded once and reused by every
# test and parametrization (see the ``moe_layer`` fixture).
pytestmark = pytest.mark.use_module_device

# (device, loader, cfg, experts) for the most recently built layer; see ``moe_layer``.
_LAYER_CACHE: tuple | None = None


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return False
    return True


def _expert_weight_cache() -> WeightCache | None:
    """The model's cache namespace for :data:`MOE_LAYER`'s routed experts, or ``None`` if disabled.

    Mirrors ``DeepSeekV4Model``, which builds each layer's experts with
    ``cache.sub(f"layers.{N}").sub("mlp")``, so the tiles land on (and are read from) exactly the
    paths the model uses.
    """
    if not _CACHE_DIR:
        return None
    return WeightCache(os.path.join(_CACHE_DIR, "ttnn")).sub(f"layers.{MOE_LAYER}").sub("mlp")


def _load_config(loader: DeepseekV4WeightLoader) -> types.SimpleNamespace:
    cfg_path = loader.snapshot_dir / "config.json"
    raw = json.loads(cfg_path.read_text())
    return types.SimpleNamespace(
        hidden_size=raw["hidden_size"],
        num_local_experts=raw["n_routed_experts"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        moe_intermediate_size=raw["moe_intermediate_size"],
        routed_scaling_factor=raw.get("routed_scaling_factor", 1.5),
        swiglu_limit=raw.get("swiglu_limit", 10.0),
        rms_norm_eps=raw.get("rms_norm_eps", 1.0e-6),
    )


def _dq(loader: DeepseekV4WeightLoader, name: str) -> torch.Tensor:
    """Dequantize an HF-named tensor to fp32 via its companion scale."""
    return dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


def _expert_provider(loader: DeepseekV4WeightLoader, layer: int):
    """Return ``provider(e) -> (gate_up [2I, H], down [H, I])`` in bf16.

    ``gate_up`` is the HF packed layout ``cat([gate_proj, up_proj])`` (rows
    ``0:I`` gate, ``I:2I`` up), matching ``DeepseekV4Experts.gate_up_proj``.
    """

    def provider(e: int):
        base = f"layers.{layer}.mlp.experts.{e}"
        gate = _dq(loader, f"{base}.gate_proj.weight")  # [I, H]
        up = _dq(loader, f"{base}.up_proj.weight")  # [I, H]
        down = _dq(loader, f"{base}.down_proj.weight")  # [H, I]
        gate_up = torch.cat([gate, up], dim=0).to(torch.bfloat16)  # [2I, H]
        return gate_up, down.to(torch.bfloat16)

    return provider


@pytest.fixture
def moe_layer(device) -> tuple[DeepseekV4WeightLoader, types.SimpleNamespace, DeepSeekV4PreloadedExperts]:
    """``(loader, cfg, experts)`` for :data:`MOE_LAYER`, built once for the whole module.

    Standing up the experts means dequantizing every routed expert of the layer on host and
    uploading it to device DRAM, which costs far more than the tests themselves. Two levels of reuse
    cut that down: the converted tiles go through the model's on-disk :class:`WeightCache` (see
    :func:`_expert_weight_cache`) so later *runs* read tiles instead of touching the checkpoint, and
    this fixture caches the result in process so the parametrized cases share one upload.

    The cache holds device tensors, hence the keying on the device *object*: the module-scoped device
    (see ``pytestmark``) hands back the same object all module long, and anything else — a
    function-scoped device, a later module — rebuilds rather than returning tensors that belong to a
    closed device. Holding the device object in the cache also keeps it alive, so its identity cannot
    be recycled by a new device while a stale entry is still live.
    """
    global _LAYER_CACHE
    if _LAYER_CACHE is None or _LAYER_CACHE[0] is not device:
        loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
        if not loader.has(f"layers.{MOE_LAYER}.mlp.gate.e_score_correction_bias"):
            pytest.skip(f"layer {MOE_LAYER} is not a standard `moe` layer")
        cfg = _load_config(loader)
        experts = DeepSeekV4PreloadedExperts(
            cfg,
            _expert_provider(loader, MOE_LAYER),
            device,
            dtype=WEIGHT_DTYPE,
            cache=_expert_weight_cache(),
        )
        _LAYER_CACHE = (device, loader, cfg, experts)
    return _LAYER_CACHE[1:]


def _bf4_expert_weights(experts: DeepSeekV4PreloadedExperts, e: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert ``e``'s on-device weights read back as fp32 ``([H, 2I], [I, H])``.

    The reference has to consume the exact BFloat4_b values the device matmuls see, so the weights
    come back off the device rather than being re-dequantized from the checkpoint; the PCC then
    reflects compute fidelity rather than the deliberate 4-bit storage choice.

    ``gate_up`` is stored in the ``fused_experts`` per-core ``[gate_block | up_block]`` interleaving
    (so each DRAM shard is one core's gate columns plus its paired up columns), which this undoes to
    recover plain ``[H, gate | up]`` column order.
    """
    gate_up_il = ttnn.to_torch(experts._gate_up_fused[e]).float()  # [H, 2I], per-core interleaved
    down = ttnn.to_torch(experts._down_fused[e]).float()  # [I, H]
    block = _swiglu_cols_per_core(experts.intermediate)
    h, two_i = gate_up_il.shape
    blocks = two_i // (2 * block)
    gate_up = gate_up_il.reshape(h, blocks, 2, block).permute(0, 2, 1, 3).reshape(h, two_i).contiguous()
    return gate_up, down


def _torch_routed_experts(
    flat: torch.Tensor, dense_w: torch.Tensor, experts: DeepSeekV4PreloadedExperts, limit: float
) -> torch.Tensor:
    """fp32 routed-expert compute for a given dense routing ``dense_w [T, E]`` -> ``[T, H]``.

    Every expert selected by *any* token is evaluated for *all* tokens and masked by that token's
    routing weight (0 where it did not select the expert), which is exactly what the batched
    ``fused_experts`` op does on device. Matches the per-expert math of ``DeepseekV4Experts``.
    """
    routed = torch.zeros_like(flat)
    for e in (dense_w.abs().sum(0) > 0).nonzero().flatten().tolist():
        gate_up, down = _bf4_expert_weights(experts, e)  # [H, 2I], [I, H] (bf4-rounded fp32)
        gate, up = (flat @ gate_up).chunk(2, dim=-1)  # [T, I] each
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        routed += (act @ down) * dense_w[:, e : e + 1]
    return routed


def _dense_from_sparse_routing(routing, cfg: types.SimpleNamespace) -> torch.Tensor:
    """Widen a router's ``(scores, indices)`` pair into the dense ``[T, E]`` weight row.

    The renormalize-and-scale tail this applies is the one ``fused_experts`` runs on device
    from the same two tensors, so the reference below and the op start from equal routing.
    """
    scores = ttnn.to_torch(routing.scores).float().reshape(-1, cfg.num_local_experts)
    ids = ttnn.to_torch(routing.indices).long().reshape(-1, cfg.num_experts_per_tok)
    selected = torch.gather(scores, -1, ids)
    weights = cfg.routed_scaling_factor * selected / (selected.sum(dim=-1, keepdim=True) + 1.0e-20)
    dense = torch.zeros_like(scores)
    dense.scatter_(-1, ids, weights)
    return dense


def _torch_router_topk(
    flat: torch.Tensor, gate_w: torch.Tensor, gate_bias: torch.Tensor, cfg: types.SimpleNamespace
) -> list[set]:
    """fp32 ``DeepseekV4TopKRouter`` top-k selection (for the routing diagnostic)."""
    logits = flat @ gate_w.float().t()
    scores = torch.sqrt(F.softplus(logits))
    idx = torch.topk(scores + gate_bias.float(), cfg.num_experts_per_tok, dim=-1, sorted=False).indices
    return [set(r.tolist()) for r in idx]


def _torch_experts_and_shared(
    flat: torch.Tensor,
    dense_w: torch.Tensor,
    experts: DeepSeekV4PreloadedExperts,
    shared: dict,
    cfg: types.SimpleNamespace,
) -> torch.Tensor:
    """fp32 routed-experts + shared-expert compute for a *given* routing.

    Takes the dense per-token/expert routing weights ``dense_w [T, E]`` (already
    softmax-normalised and ``routed_scaling_factor``-scaled) so the comparison
    isolates the expert / shared arithmetic from bf16 top-k *selection* noise,
    which is reported separately. Routed-expert weights are read back from the
    device (see :func:`_bf4_expert_weights`) so the reference consumes the same
    BFloat4-quantised values the on-device matmuls do — the PCC then reflects
    compute fidelity, not the deliberate 4-bit storage choice. Matches the
    per-expert math of ``DeepseekV4Experts`` + ``DeepseekV4MLP``.
    """
    routed = _torch_routed_experts(flat, dense_w, experts, cfg.swiglu_limit)

    sg, su, sd = shared["gate"].float(), shared["up"].float(), shared["down"].float()
    shared_out = (F.silu(flat @ sg.t()) * (flat @ su.t())) @ sd.t()
    return routed + shared_out


@pytest.mark.skipif(
    not _checkpoint_available(),
    reason=f"V4-Flash checkpoint not found under {DEFAULT_MODEL_DIR}",
)
@torch.no_grad()
@pytest.mark.parametrize("seq_len", (32,))
@pytest.mark.parametrize("batch_size", (1,))
def test_moe_real_weights_pcc(device, reset_seeds, moe_layer, batch_size: int, seq_len: int) -> None:
    loader, cfg, experts = moe_layer

    gate_w = loader.get_tensor(f"layers.{MOE_LAYER}.mlp.gate.weight")  # bf16 [E, H]
    gate_bias = loader.get_tensor(f"layers.{MOE_LAYER}.mlp.gate.e_score_correction_bias")  # f32 [E]
    shared = {
        "gate": _dq(loader, f"layers.{MOE_LAYER}.mlp.shared_experts.gate_proj.weight"),
        "up": _dq(loader, f"layers.{MOE_LAYER}.mlp.shared_experts.up_proj.weight"),
        "down": _dq(loader, f"layers.{MOE_LAYER}.mlp.shared_experts.down_proj.weight"),
    }

    torch.manual_seed(1234)
    hidden = torch.randn(batch_size, seq_len, cfg.hidden_size, dtype=torch.float32)
    flat = hidden.reshape(-1, cfg.hidden_size).float()

    # ttnn block: real router + shared expert weights, streaming routed experts.
    weights = {
        "gate.weight": gate_w,
        "gate.e_score_correction_bias": gate_bias,
        "shared_experts.gate_proj.weight": shared["gate"].to(torch.bfloat16),
        "shared_experts.up_proj.weight": shared["up"].to(torch.bfloat16),
        "shared_experts.down_proj.weight": shared["down"].to(torch.bfloat16),
    }
    block = DeepSeekV4SparseMoeBlock(cfg, weights, device, experts=experts)

    # The ttnn block takes [B, S, 1, H] (the reference's [B, S, H] with the tile row axis) and flattens
    # the tokens internally; x_flat mirrors that so the router below sees exactly what it will.
    hidden_tt = ttnn.from_torch(hidden.unsqueeze(2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x_flat = ttnn.reshape(hidden_tt, [1, 1, flat.shape[0], cfg.hidden_size])

    # Drive the fp32 reference with the *device* router's routing so the PCC
    # compares expert / shared arithmetic on identical routing. The router still
    # runs on device (here and again inside ``block.forward``, deterministically);
    # bf16 top-k *selection* divergence vs fp32 is reported separately below
    # because it is an inherent dtype effect, not a port bug.
    #
    # The router emits the selected ids and the score row, leaving the normalize to
    # ``fused_experts``; widening that here is what the reference needs and doubles as a
    # check that the op's on-device version of the same tail agrees.
    dense_w = _dense_from_sparse_routing(block.gate(x_flat), cfg)

    reference = _torch_experts_and_shared(flat, dense_w, experts, shared, cfg).reshape(hidden.shape)

    out_tt = block.forward(hidden_tt)
    out_torch = ttnn.to_torch(out_tt).reshape(reference.shape).to(torch.float32)

    # Routing-agreement diagnostic: how often the bf16 device router picks the
    # same expert set as the fp32 reference router (soft guard against a broken
    # router, while tolerating boundary flips among near-tied scores).
    ref_sets = _torch_router_topk(flat, gate_w, gate_bias, cfg)
    tt_sets = [set((dense_w[t] > 0).nonzero().flatten().tolist()) for t in range(flat.shape[0])]
    overlap = sum(len(a & b) for a, b in zip(ref_sets, tt_sets)) / len(ref_sets)
    logger.info(f"[moe real weights] router overlap {overlap:.3f}/{cfg.num_experts_per_tok} experts/token")

    passing, pcc_message = comp_pcc(reference, out_torch, pcc=PCC_THRESHOLD)
    logger.info(comp_allclose(reference, out_torch))
    logger.info(f"[moe real weights] layer={MOE_LAYER} PCC: {pcc_message}")

    assert overlap >= cfg.num_experts_per_tok - 1.0, f"router selection overlap too low: {overlap:.3f}"
    assert passing, f"real-weights moe PCC < {PCC_THRESHOLD} (batch={batch_size}, seq={seq_len}): {pcc_message}"


# --------------------------------------------------------------------------- #
# Batched fused_experts (B token rows in one op).
# --------------------------------------------------------------------------- #

# The batched op and the per-token ops run identical arithmetic in an identical order (a token that
# did not select an expert contributes an exact 0.0 to the sum), so they should agree far more
# tightly than either agrees with the fp32 reference.
BATCH_EQUIV_PCC = 0.999

# How many experts the op holds in L1 at once. The gathered [experts, B, I] activation block lives in
# L1 on EVERY core, so without blocking the *union* would bound the op: at H=4096 / I=2048 the
# circular buffers reach 1,564,096 B of the 1,572,864 B L1 at 12 experts and overflow at 13, which at
# top_k=6 would cap no-overlap batching at 2 tokens. `experts_block_size` decouples the two — the op
# runs the union in blocks of this many experts, so L1 is set by the block and the union is free to
# grow. A block costs ~136 KB per core here (2 slots * 64 tiles * 1088 B of bf8 activation), the two
# slots being what lets a block's fetches overlap the previous block's compute, so the ceiling for
# this layer is 5 -- 6 clears the unblocked budget but not the ~20 KB of extra routing state a
# 256-expert / 32-token layer carries.
EXPERTS_BLOCK_SIZE = 5


def _sharing_expert_sets(batch: int, top_k: int, num_experts: int, sharing: str) -> list[list[int]]:
    """Per-token expert selections with a controlled amount of sharing.

    * ``identical``: every token selects the same ``top_k`` experts (union == top_k).
    * ``partial``: alternating tokens swap one expert for a shared alternate (union == top_k + 1).
    * ``disjoint``: no two tokens share an expert, so the union is the full ``batch * top_k`` — the
      worst case for weight traffic, and the one that needs blocking to fit in L1 at all.
    """
    perm = torch.randperm(num_experts).tolist()
    base = sorted(perm[:top_k])
    if sharing == "identical":
        return [list(base) for _ in range(batch)]
    if sharing == "disjoint":
        assert batch * top_k <= num_experts, (
            f"a fully disjoint batch of {batch} at top_k {top_k} needs {batch * top_k} distinct "
            f"experts, more than the {num_experts} the layer has"
        )
        return [sorted(perm[b * top_k : (b + 1) * top_k]) for b in range(batch)]
    alternate = sorted(base[:-1] + [perm[top_k]])
    return [list(base) if b % 2 == 0 else list(alternate) for b in range(batch)]


def _routing(expert_sets: list[list[int]], num_experts: int, scaling: float):
    """Routing for explicit per-token expert sets: ``(ids [B, k], scores [B, E], dense [B, E])``.

    The first two are what a router hands the op. The third is the per-token weights the op derives
    from them -- each token's selected scores renormalized to sum to 1, then scaled by
    ``routed_scaling_factor``, and exactly 0 for every unselected expert -- widened here so the fp32
    reference can index them expert-major.
    """
    ids = torch.tensor(expert_sets, dtype=torch.int64)
    scores = (torch.rand(len(expert_sets), num_experts, dtype=torch.float32) + 0.5).to(torch.bfloat16).float()
    selected = torch.gather(scores, -1, ids)
    dense = torch.zeros_like(scores)
    dense.scatter_(-1, ids, scaling * selected / (selected.sum(dim=-1, keepdim=True) + _ROUTING_EPS))
    return ids, scores, dense


def _run_fused_experts(
    device,
    experts: DeepSeekV4PreloadedExperts,
    cfg: types.SimpleNamespace,
    x: torch.Tensor,
    ids: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """One ``fused_experts`` call over ``x [B, H]`` and the routing pair -> ``[B, H]`` fp32.

    ``num_experts`` is the size of the union of the tokens' selections, which is exactly the number
    of distinct experts the op fetches weights for. It is run in blocks of :data:`EXPERTS_BLOCK_SIZE`,
    which for a union that small is a single block (the op clamps the block to the union) and only
    starts to matter for the wider unions the disjoint cases produce.
    """
    batch, hidden = x.shape
    num_active = len(set(ids.flatten().tolist()))
    x_tt = ttnn.from_torch(x.reshape(1, 1, batch, hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ids_tt = ttnn.from_torch(
        ids.to(torch.int32).reshape(1, 1, batch, ids.shape[-1]),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    scores_tt = ttnn.from_torch(
        scores.reshape(1, 1, batch, cfg.num_local_experts),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    out = ttnn.experimental.deepseek.moe.fused_experts(
        x_tt,
        routing_indices=ids_tt,
        routing_scores=scores_tt,
        gate_up_weights=experts._gate_up_fused,
        down_weights=experts._down_fused,
        num_experts=num_active,
        intermediate_size=cfg.moe_intermediate_size,
        swiglu_limit=cfg.swiglu_limit,
        top_k=cfg.num_experts_per_tok,
        routed_scaling_factor=cfg.routed_scaling_factor,
        routing_eps=_ROUTING_EPS,
        experts_block_size=EXPERTS_BLOCK_SIZE,
    )  # [1, B, H]
    return ttnn.to_torch(out).float().reshape(batch, hidden)


@pytest.mark.skipif(
    not _checkpoint_available(),
    reason=f"V4-Flash checkpoint not found under {DEFAULT_MODEL_DIR}",
)
@torch.no_grad()
@pytest.mark.parametrize("sharing", ("identical", "partial", "disjoint"))
@pytest.mark.parametrize("batch_size", (1, 4, 16, 32))
def test_moe_real_weights_batched_fused_experts_pcc(
    device, reset_seeds, moe_layer, batch_size: int, sharing: str
) -> None:
    """``fused_experts`` with B > 1 token rows in a single op, on real checkpoint weights.

    Drives the op directly (rather than through :class:`DeepSeekV4SparseMoeBlock`, whose expert path
    still issues one op per token) with B tokens packed into dim -2, and checks three things:

    * **correctness per token** against the fp32 reference, which evaluates every selected expert for
      every token and masks by the per-token routing weight;
    * **equivalence to the per-token path**: the same tokens run one-at-a-time through the same op
      must give the same answers, so batching changes throughput and nothing else;
    * **weight sharing** (B > 1): with ``identical`` / ``partial`` routing the tokens overlap in their
      expert choices, so the union the op iterates is much smaller than the sum of the per-token
      selections. Since the op fetches each hit expert's weights exactly once per core (one iteration
      per union member, indexed by hit id rather than by token), that ratio is the DRAM traffic saved
      by batching — an expert two tokens share is fetched once, not twice. B == 1 is kept as a
      regression guard for the single-token path through the same code.

    ``disjoint`` is the opposite extreme, where no two tokens share an expert: there is nothing to
    amortize, and the point is that a token still gets contributions from *only* its own experts even
    though every core evaluates all of them. It is also the case that needs expert blocking — a
    disjoint batch of 32 at top_k 6 spans 192 distinct experts, whose activations would be ~13 MB per
    core if held at once — so it is what covers running a union many blocks deep against the
    single-block path the narrower unions take (see :data:`EXPERTS_BLOCK_SIZE`).
    """
    _, cfg, experts = moe_layer

    torch.manual_seed(1234)
    # Round the inputs to what the device sees (bf16 activations / routing weights) so the fp32
    # reference differs from the device only in compute, not in its operands.
    x = torch.randn(batch_size, cfg.hidden_size, dtype=torch.float32).to(torch.bfloat16).float()
    expert_sets = _sharing_expert_sets(batch_size, cfg.num_experts_per_tok, cfg.num_local_experts, sharing)
    ids, scores, dense_w = _routing(expert_sets, cfg.num_local_experts, cfg.routed_scaling_factor)

    hit_ids = (dense_w.abs().sum(0) > 0).nonzero().flatten().tolist()
    per_token_selections = sum(len(s) for s in expert_sets)

    # Each case has to actually be the case it claims to be, since the sharing is what the batched
    # fetch path is being tested on.
    if sharing == "disjoint":
        assert len(hit_ids) == per_token_selections, (
            f"the disjoint case must have no expert shared between tokens: union {len(hit_ids)} != "
            f"per-token selections {per_token_selections}"
        )
    elif batch_size > 1:
        assert len(hit_ids) < per_token_selections, (
            f"this case does not exercise expert sharing: union {len(hit_ids)} == "
            f"per-token selections {per_token_selections}"
        )

    signpost("fused_experts_START", f"batch={batch_size} sharing={sharing}")
    batched = _run_fused_experts(device, experts, cfg, x, ids, scores)
    signpost("fused_experts_END", f"batch={batch_size} sharing={sharing}")
    per_token = torch.cat(
        [
            _run_fused_experts(device, experts, cfg, x[b : b + 1], ids[b : b + 1], scores[b : b + 1])
            for b in range(batch_size)
        ]
    )
    reference = _torch_routed_experts(x, dense_w, experts, cfg.swiglu_limit)

    logger.info(
        f"[moe batched] batch={batch_size} sharing={sharing}: {len(hit_ids)} distinct experts fetched "
        f"for {per_token_selections} per-token selections "
        f"({per_token_selections / len(hit_ids):.1f}x weight-fetch amortization)"
    )
    ref_passing, ref_msg = comp_pcc(reference, batched, pcc=PCC_THRESHOLD)
    equiv_passing, equiv_msg = comp_pcc(per_token, batched, pcc=BATCH_EQUIV_PCC)
    logger.info(comp_allclose(reference, batched))
    logger.info(f"[moe batched] PCC vs fp32 reference: {ref_msg}")
    logger.info(f"[moe batched] PCC vs per-token ops:  {equiv_msg}")

    assert ref_passing, f"batched fused_experts PCC < {PCC_THRESHOLD} (batch={batch_size}, {sharing}): {ref_msg}"
    assert equiv_passing, (
        f"batched fused_experts disagrees with the per-token path (PCC < {BATCH_EQUIV_PCC}, "
        f"batch={batch_size}, {sharing}): {equiv_msg}"
    )
