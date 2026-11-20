# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: one decode step of sparse attention on the device (tt/attention_decode.py) vs the pure-torch
ground truth (reference/attention_decode.py), tenstorrent/tt-metal#65889 and the decode side of
#65888.

The device cache is prefilled from the host with the reference's latents and index keys, and the
reference attends over exactly what that storage holds (the rows as bf16 / BFP8 packs them), so a
difference is the step's, not the storage format's. Then, stage by stage, so a failure names
where it is:

    new token      q_resid | the latent and index-key rows that land in the cache | the indexer's
                   query heads and head weights
    selection      exactly min(top-k, context) distinct, visible picks | the picks against the exact
                   top-k of the device's own queries (the arithmetic) | the picks against the fp32
                   reference, next to what HF's arithmetic in bf16 moves on the same query
    locate         the storage row and the owner flag of every pick, against kv_layout
    gather+collect the selected rows, whole on every chip: bit-equal to the stored rows
    attention      q_b and gate | the logits and the core on the reference's inputs
    end to end     against the reference with its own picks, and with the device's picks

On one chip first, then on the 8x4 Galaxy mesh at a narrow width and at Hy4's own: from an empty
context, below and above the top-k, above 65,536 positions (where every chip runs a local top-k),
over block and round boundaries, on shared layers, with the selected rows spread unevenly over the
chips (all on one chip, none on most), with both forms of the core, and with the checkpoint's
weights: layers 0-5 on their real inputs against the recorded HF output, and at 70K context.
Once more on the mesh as the prefill stack opens it (torus fabric), since decode reads the cache
prefill writes. Last, the per-step latency of the pieces, with a cache of one user and of eight.

The stage tests feed every stage the reference's input for that stage; the end-to-end checks do not.
Hidden states are handed to both sides as bf16 holds them, which is how the layer's norm hands
them to attention.

Frames (tt/frames.py). The device cache is the one prefill writes: k_rope and the rope dims of the
queries in interleaved RoPE pairs, index keys as [rope interleaved | pass] in the Hadamard basis.
The reference and its mirror of the cache stay in the HF frame. Rows go to the device through
frames.tail_to_device / index_to_device and come back through the inverses; anything compared
row by row (cache rows, collected rows, q) is compared in the device frame. Logits, scores,
selections and outputs are the same numbers in both frames.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference import hf_harness
from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention
from models.experimental.hy4_preview.reference.attention_decode import DecodeCache, Hy4AttentionDecode
from models.experimental.hy4_preview.reference.indexer import (
    Hy4Indexer,
    IndexerConfig,
    compare_selection,
    full_layers,
    picks_source,
)
from models.experimental.hy4_preview.tests.common import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH,
    SC1_HIDDEN_AXIS,
    check_pcc,
    hy4_tensors,
    pcc,
)
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt.attention import TtHy4Attention
from models.experimental.hy4_preview.tt.attention_decode import (
    CORES,
    MASKED,
    STAGES,
    TtHy4AttentionDecode,
    begin_decode_step,
)
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.indexer import Picks, TtHy4IndexCache, TtHy4Indexer
from models.experimental.hy4_preview.tt.kv_cache import INDEX_KEY, LATENT, KvCacheConfig, TtHy4KvCache, selection_table
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

FULL_WIDTH = (AttentionConfig(), IndexerConfig())
# Hy4's latent (512 | 64), index key (128) and top-k from a narrow hidden state: long contexts stay
# cheap on the host, and every width still splits 4 ways (hidden) and 8 ways (heads).
NARROW_ATTENTION = AttentionConfig(
    hidden_size=768,
    num_heads=8,
    q_lora_rank=256,
    kv_lora_rank=512,
    qk_nope_head_dim=96,
    qk_rope_head_dim=64,
    v_head_dim=128,
)
NARROW = (NARROW_ATTENTION, IndexerConfig(hidden_size=768, q_lora_rank=256, num_heads=8, head_dim=128))
# The same with a top-k of 64: a context of a few hundred tokens is already cut, on one chip too.
NARROW_TOP64 = (NARROW_ATTENTION, IndexerConfig(hidden_size=768, q_lora_rank=256, num_heads=8, head_dim=128, topk=64))
CONFIGS = {"narrow": NARROW, "narrow-top64": NARROW_TOP64, "full-width": FULL_WIDTH}

# name -> (weight dtype, cache dtype, per-stage threshold, end-to-end threshold).
# The PCC thresholds are the ones tests/test_attention.py holds the prefill sublayer to.
DTYPES = {
    "bf16": (ttnn.bfloat16, "bf16", 0.999, 0.998),
    "bfp8": (ttnn.bfloat8_b, "bfp8", 0.998, 0.995),
}
TT_DTYPE = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
ROUND = 32 * 128  # positions in one round of the 32-chip layout
assert GALAXY_MESH == (8, 4) and SC1_HIDDEN_AXIS == 1, "these tests are written for the 8x4 mesh, hidden on axis 1"

# Selection. Two sets of picks are compared position by position (reference/indexer.py
# compare_selection): a position on which they disagree is a near-tie when its score lies within a
# tolerance of the cut, anything else is a real error, and real errors must be zero. A swap is two
# near-ties. Tolerances are a share of the largest sum of |terms| of a score. Every selection of a
# full layer is held to two things (_Model.check_selection):
#
# * The arithmetic: the device's picks against the exact fp32 top-k of the queries and head weights
#   the device itself made, over the keys its cache holds. This is the comparison the indexer's own
#   tests make, with their budget: 1% of the picks, at least two swaps. Measured on the Galaxy
#   (2048 picks, random and checkpoint weights): 2 to 4 near-ties at 70,000 positions and 6 to 10
#   at one million with the indexer's default head sum, 0 to 4 at 70,000 with its elementwise fp32
#   head sum (``head_sum="fp32"``); the furthest swapped score is 3.8e-4 from the cut.
# * The whole step against the fp32 reference, from the hidden state on. Here the queries differ
#   too: the device makes them in bf16, the reference in fp32, and scores next to the cut change
#   sides. That is not the device's doing. HF's own arithmetic in bf16 (reference as_bf16(), pinned
#   to HF by the host test) moves 30 of 2048 picks against the fp32 reference at 70,000 positions
#   with random weights, where the device moves 26 to 30; 16 to 22 with the checkpoint's, where the
#   device moves 10 to 16; and 20 at one million, where the device moves 30. So 1% of the picks
#   (20) is not a budget a bf16 selection can be held to against fp32, and the count swings by
#   several picks with the rounding of one input. The budget with bf16 weights is therefore what
#   HF's bf16 arithmetic moves on this very query, plus the arithmetic's 1%. Leaving the query
#   heads in fp32 instead of rounding them to bf16 as the indexer (and HF) does changes nothing:
#   26 either way (test_selection_precision_sc1).
#   With BFP8 attention weights q_resid comes out of a BFP8 q_a_proj, which HF's bf16 arithmetic
#   has no counterpart for: 40 to 46 near-ties at 70,000 positions with random weights, 28 to 34
#   with the checkpoint's and 52 at one million (1.4% to 2.5% of the picks). Their budget stays the
#   4% it was set to before any device run.
NEAR_TIE_BUDGET = 0.01
NEAR_TIE_FLOOR = 4
BLOCK_FLOAT_NEAR_TIE_BUDGET = 0.04
# The furthest a swapped score may lie from the cut, by weight dtype. Against the reference the
# furthest measured over every selection of this file is 2.0e-3 with bf16 weights and 5.7e-3 with
# BFP8. For the arithmetic it is twice the largest score error the indexer's tests measured on
# identical inputs (0.19%); the furthest measured here is 3.8e-4.
SCORE_TOLERANCE = {"bf16": 0.005, "bfp8": 0.015}
ARITHMETIC_TOLERANCE = 0.004

# A whole decoder layer per decode step: the parent issue's estimate (tenstorrent/tt-metal#65884), and
# the project's target of 75 tokens per second per user with 8 users at 1M context on one Galaxy:
# one step of the 8-user batch in 13.3 ms over 78 layers, everything included.
LAYER_ESTIMATE_MS = 0.33
TARGET_USERS = 8
LAYER_TARGET_MS = 1e3 / 75 / 78

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device", [(GALAXY_DEVICE_PARAMS, GALAXY_MESH)], indirect=True, ids=["8x4"]
)


# L1_SMALL per core: what the prefill stack opens the mesh with, and a size a decode step fits in.
PREFILL_L1_SMALL = 1536
DECODE_L1_SMALL = 8192


def _prefill_fabric_params(l1_small_size):
    """The 8x4 mesh the way the prefill stack opens it (models/demos/deepseek_v3_d_p
    tests/fabric_profiles.py torus_xy_device_params with the Hy4 adapter's payload): the torus
    fabric, relaxed init and a 12,352-byte packet payload. ``l1_small_size`` is the region the
    collectives' semaphores come out of once it exists (the fixtures of the other tests open the
    mesh without one)."""
    router = ttnn._ttnn.fabric.FabricRouterConfig()
    router.max_packet_payload_size_bytes = 12352
    return {
        "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        "fabric_router_config": router,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "l1_small_size": l1_small_size,
    }


prefill_fabric = pytest.mark.parametrize(
    "device_params, mesh_device, l1_small",
    [(_prefill_fabric_params(size), GALAXY_MESH, size) for size in (PREFILL_L1_SMALL, DECODE_L1_SMALL)],
    indirect=["device_params", "mesh_device"],
    ids=[f"8x4-torus-l1small{size}" for size in (PREFILL_L1_SMALL, DECODE_L1_SMALL)],
)


# ---------------------------------------------------------------------------------------------
# References and device modules
# ---------------------------------------------------------------------------------------------
def _bf16(t):
    return t.bfloat16().float()


def _reference(acfg, icfg, full, seed) -> Hy4AttentionDecode:
    """A reference layer with random parameters at the scale a trained layer has: linear outputs of
    order one, norm weights, biases and sinks away from their trivial init."""
    g = torch.Generator().manual_seed(seed)
    attention = Hy4Attention(acfg).eval()
    with torch.no_grad():
        for name in Hy4Attention.PARAMETER_NAMES:
            p = getattr(attention, name)
            if p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5)
            elif name == "learnable_sink_param":
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.25 * torch.randn(p.shape, generator=g))
    indexer = None
    if full:
        indexer = Hy4Indexer(icfg).eval()
        with torch.no_grad():
            for name, p in indexer.named_parameters():
                if p.dim() == 2:
                    p.copy_(_bf16(torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5))
                else:
                    p.copy_(_bf16((1.0 if name == "k_norm_weight" else 0.0) + 0.25 * torch.randn(p.shape, generator=g)))
    return Hy4AttentionDecode(attention, indexer)


def _checkpoint_reference(layer, full) -> Hy4AttentionDecode:
    """The reference with layer ``layer``'s attention (and indexer) weights from the checkpoint."""
    prefix = f"model.layers.{layer}.self_attn."
    names = [prefix + n + ("" if n == "learnable_sink_param" else ".weight") for n in Hy4Attention.PARAMETER_NAMES]
    attention = Hy4Attention(AttentionConfig()).eval()
    with torch.no_grad():
        for name, value in zip(Hy4Attention.PARAMETER_NAMES, hy4_tensors(names)):
            getattr(attention, name).copy_(value.reshape(getattr(attention, name).shape))
    indexer = None
    if full:
        keys = list(Hy4Indexer.CHECKPOINT_NAMES.values())
        indexer = Hy4Indexer(IndexerConfig()).eval()
        indexer.load_weights(dict(zip(keys, hy4_tensors([prefix + "indexer." + k for k in keys]))))
    return Hy4AttentionDecode(attention, indexer)


def _held(x, dtype):
    """What ``dtype`` storage holds of ``x`` [T, W] when the host packs it."""
    if x.shape[0] == 0:
        return x.float().clone()
    t = ttnn.from_torch(x.reshape(1, 1, *x.shape).float(), dtype=TT_DTYPE[dtype], layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float().reshape(x.shape)


def _rows_up(plan, t, dtype=ttnn.bfloat16, mesh_mapper=None):
    """[T, width] -> device [1, 1, T, width]."""
    return ttnn.from_torch(
        t.float().contiguous().reshape(1, 1, *t.shape),
        layout=ttnn.TILE_LAYOUT,
        device=plan.mesh_device,
        dtype=dtype,
        mesh_mapper=mesh_mapper,
    )


def _whole(t):
    """Chip 0's copy of a tensor that is whole on every chip, checked against every other chip."""
    shards = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(t)]
    for chip, shard in enumerate(shards[1:], start=1):
        assert torch.equal(shards[0], shard), f"chip {chip} differs from chip 0 on a tensor that should be whole"
    return shards[0]


def _picks_up(plan, positions, valid) -> Picks:
    """A hand-made selection: positions int [K], valid bool [K] -> the Picks the indexer would return."""
    up = lambda t: ttnn.from_torch(  # noqa: E731
        t.float().reshape(1, 1, 1, -1), layout=ttnn.TILE_LAYOUT, device=plan.mesh_device, dtype=ttnn.float32
    )
    return Picks(positions=up(positions), valid=up(valid), num_queries=1)


class _Model:
    """Some layers of a decoder as far as a decode step of attention goes: per layer the reference
    and the device module, one device cache and one index cache for all of them, and the host mirror
    of what the device cache holds."""

    def __init__(self, plan, configs, dtype, max_context, types=("full",), layers=None, core="absorb", seed=1, **kw):
        self.plan = plan
        self.acfg, self.icfg = configs
        self.weight_dtype, self.dtype, self.pcc_stage, self.pcc_e2e = DTYPES[dtype]
        self.types = list(types)
        self.layers = tuple(range(len(types))) if layers is None else tuple(layers)
        self.source = picks_source(self.types)
        self.user = kw.get("user", 0)
        self.layout = BlockCyclicLayout(num_chips=plan.hidden_tp * plan.head_tp)
        kv_cfg = KvCacheConfig(
            num_layers=len(types),
            indexer_layers=tuple(n for n, kind in enumerate(types) if kind == "full"),
            latent_dim=self.acfg.latent_dim,
            index_key_dim=self.icfg.head_dim,
            max_users=kw.get("max_users", 1),
            max_context=max_context,
            dtype=self.dtype,
            layout=self.layout,
        )
        self.cache = TtHy4KvCache(plan.mesh_device, kv_cfg, layers=self.layers)
        self.index_cache = TtHy4IndexCache(self.types)
        refs = kw.get("refs") or {}
        self.ref, self.tt, self.mirror = {}, {}, {}
        self.queries, self._ref16 = {}, {}
        for n in self.layers:
            ref = refs[n] if n in refs else _reference(self.acfg, self.icfg, self.types[n] == "full", seed + n)
            attention = TtHy4Attention(plan, self.acfg, ref.attention.weights(), weight_dtype=self.weight_dtype)
            indexer = None
            if ref.is_full:
                head_sum = kw.get("head_sum", "matmul")
                indexer = TtHy4Indexer(plan, self.icfg, ref.indexer.weights(), layout=self.layout, head_sum=head_sum)
            self.ref[n] = ref
            self.tt[n] = TtHy4AttentionDecode(plan, attention, self.cache, n, self.index_cache, indexer, core=core)
            if ref.is_full and kw.get("record", True):
                self._record_queries(n)
            width = (0, self.icfg.head_dim)
            self.mirror[n] = DecodeCache(
                torch.zeros(0, self.acfg.latent_dim), torch.zeros(*width) if ref.is_full else None
            )

    # ------------------------------------------------------------------------------------------
    def _record_queries(self, layer):
        """Keep a host copy of the indexer queries and head weights the device makes for every new
        token of ``layer`` (whole on every chip, in the cache's frame), for check_selection."""
        tt, inner = self.tt[layer], self.tt[layer].new_token

        def new_token(x, step):
            q_resid, index_q, index_w = inner(x, step)
            self.queries[layer] = (_whole(index_q)[0], _whole(index_w)[0, 0])  # [1, heads, dim], [1, heads]
            return q_resid, index_q, index_w

        tt.new_token = new_token

    def ref16(self, layer) -> Hy4AttentionDecode:
        """The layer's reference at the checkpoint's precision (reference as_bf16)."""
        if layer not in self._ref16:
            self._ref16[layer] = self.ref[layer].as_bf16()
        return self._ref16[layer]

    def inputs(self, count, seed):
        """Per layer, ``count`` hidden states of unit RMS, [count, hidden], as bf16 holds them: the
        device takes its input in bf16 (the layer's norm hands attention a bf16 tensor), and the
        reference is given the same values, not the fp32 numbers they were rounded from."""
        g = torch.Generator().manual_seed(seed)
        return {n: _bf16(torch.randn(count, self.acfg.hidden_size, generator=g)) for n in self.layers}

    def prefill(self, xs):
        """Fill every layer's cache with the reference's entries for ``xs[layer]`` [P, hidden]."""
        for n in self.layers:
            with torch.no_grad():
                entries = self.ref[n].prefill_cache(xs[n])
            self.prefill_rows(n, entries.latent, entries.index_keys)

    # the cache's frame (tt/frames.py) <-> the reference's
    def latent_to_device(self, latent):
        return frames.tail_to_device(latent, self.acfg.qk_rope_head_dim)

    def latent_from_device(self, latent):
        return frames.tail_from_device(latent, self.acfg.qk_rope_head_dim)

    def key_to_device(self, key):
        return frames.index_to_device(key, self.icfg.qk_rope_head_dim)

    def key_from_device(self, key):
        return frames.index_from_device(key, self.icfg.qk_rope_head_dim)

    def prefill_rows(self, layer, latent, index_keys=None, mirror=True, user=None):
        """Write reference-frame rows [P, width] at positions 0 .. P-1 of the layer's cache, moved to
        the cache's frame and packed by the host, and mirror what the storage holds of them."""
        d_latent = self.latent_to_device(latent)
        d_keys = None if index_keys is None else self.key_to_device(index_keys)
        if latent.shape[0]:
            self.cache.write_prefill_host(layer, self.user if user is None else user, latent=d_latent, index_key=d_keys)
        if mirror:
            held_keys = None if d_keys is None else self.key_from_device(_held(d_keys, self.dtype))
            self.mirror[layer] = DecodeCache(self.latent_from_device(_held(d_latent, self.dtype)), held_keys)

    def stored_row(self, layer, kind, position):
        """The row the device cache holds for ``position``, read from the chip that owns it (through
        the tile strip around its slot, so the read stays tile aligned)."""
        chip, slot = int(self.layout.chip(position)), int(self.layout.slot(position))
        first = slot // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        width = self.cache.cfg.width(kind)
        strip = ttnn.slice(
            self.cache.stripe(layer, kind), [self.user, 0, first, 0], [self.user + 1, 1, first + ttnn.TILE_SIZE, width]
        )
        return ttnn.to_torch(ttnn.get_device_tensors(strip)[chip]).float()[0, 0, slot - first]

    def sync(self, layer, position):
        """Replace the mirror's row of ``position`` (the reference's own, fp32) by what the device
        stored, brought back to the reference's frame, so that the next read of the cache compares
        like with like. Returns the stored rows as the device holds them (the cache's frame)."""
        mirror = self.mirror[layer]
        latent = self.stored_row(layer, LATENT, position)
        mirror.latent[position] = self.latent_from_device(latent)
        key = None
        if mirror.index_keys is not None:
            key = self.stored_row(layer, INDEX_KEY, position)
            mirror.index_keys[position] = self.key_from_device(key)
        return latent, key

    def x_up(self, x):
        return _rows_up(self.plan, x, mesh_mapper=self.plan.shard_hidden(3))

    def out_down(self, d_out):
        assert tuple(d_out.shape) == (1, 1, 1, self.acfg.hidden_size // self.plan.hidden_tp), tuple(d_out.shape)
        return self.plan.to_torch_hidden_concat(d_out, dim=-1)[0, 0].float()

    def picked(self, picks: Picks):
        """(positions int64 [K], valid bool [K]) of a device selection, checked whole on every chip."""
        positions, valid = _whole(picks.positions)[0, 0, 0], _whole(picks.valid)[0, 0, 0]
        return positions.round().long(), valid > 0.5

    # ------------------------------------------------------------------------------------------
    def check_selection(self, layer, x, token, positions, valid, label):
        """The device's picks of a full layer for the token ``x`` [1, hidden] (module docstring,
        "Selection"): against the exact top-k of the device's own queries (the arithmetic), and
        against the fp32 reference, next to what the checkpoint's own precision moves."""
        ref, mirror, k = self.ref[layer], self.mirror[layer], self.icfg.topk
        context = len(mirror)
        want = min(k, context)
        chosen = positions[valid]
        assert chosen.numel() == want, f"{label}: {chosen.numel()} valid picks, expected {want}"
        assert chosen.unique().numel() == want, f"{label}: a position was picked twice"
        assert int(chosen.max()) < context, f"{label}: a pick is ahead of the query"
        assert int(positions.max()) < self.cache.cfg.capacity and int(positions.min()) >= 0

        def top(scores):
            chosen_set = torch.zeros(1, context, dtype=torch.bool)
            chosen_set[0, scores[0].topk(want).indices] = True
            return chosen_set

        def distance(scores, a, b):
            """The furthest a swapped position's score lies from the cut, as a share of ``size``."""
            differ = a ^ b
            if context <= k or not differ.any():
                return 0.0
            return ((scores - scores[0].topk(want).values[-1]).abs()[differ]).max().item() / size

        dev_set = torch.zeros(1, context, dtype=torch.bool)
        dev_set[0, chosen] = True
        with torch.no_grad():
            scores = ref.scores(token, mirror)[None]  # [1, context], the fp32 reference
            size = Hy4Indexer.scores(token.index_q, token.index_w.abs(), mirror.index_keys).max().item()
            # the same token at the checkpoint's precision, over the same keys
            token16 = self.ref16(layer).new_token(x.bfloat16(), token.position)
            scores16 = Hy4Indexer.scores(token16.index_q, token16.index_w, mirror.index_keys)
            # the exact scores of the queries the device made, over the same keys (the cache's frame)
            d_q, d_w = self.queries[layer]
            own = Hy4Indexer.scores(d_q, d_w, self.key_to_device(mirror.index_keys))
        tol = SCORE_TOLERANCE[self.dtype] * size
        ref_set, own_set, set16 = top(scores), top(own), top(scores16)
        total = compare_selection(scores, ref_set, dev_set, k, tol=tol)
        arithmetic = compare_selection(own, own_set, dev_set, k, tol=ARITHMETIC_TOLERANCE * size)
        queries = compare_selection(scores, ref_set, own_set, k, tol=tol)
        released = compare_selection(scores, ref_set, set16, k, tol=tol)
        logger.info(
            f"SELECTION {label}: picks={want} of {context} | against the fp32 reference: missing={total['missing']} "
            f"extra={total['extra']} near_tie={total['near_tie']} real={total['real']} "
            f"furthest={distance(scores, ref_set, dev_set):.2e} | arithmetic (exact top-k of the device's queries): "
            f"near_tie={arithmetic['near_tie']} real={arithmetic['real']} furthest={distance(own, own_set, dev_set):.2e} "
            f"| the device's queries, exact arithmetic: near_tie={queries['near_tie']} "
            f"furthest={distance(scores, ref_set, own_set):.2e} | HF arithmetic in bf16: near_tie={released['near_tie']} "
            f"furthest={distance(scores, ref_set, set16):.2e}"
        )
        assert total["real"] == 0, f"{label}: the device's picks are not the reference's top-k: {total}"
        assert (
            arithmetic["real"] == 0
        ), f"{label}: the device's picks are not the top-k of its own queries: {arithmetic}"
        budget = max(NEAR_TIE_BUDGET * want, NEAR_TIE_FLOOR)
        assert arithmetic["near_tie"] <= budget, f"{label}: the selection arithmetic moves too many picks: {arithmetic}"
        if self.dtype == "bf16":
            allowed = released["near_tie"] + budget
        else:
            allowed = max(BLOCK_FLOAT_NEAR_TIE_BUDGET * want, NEAR_TIE_FLOOR)
        assert total["near_tie"] <= allowed, f"{label}: too many near-ties, {allowed:.0f} allowed: {total}"

    def step(self, xs, label, selection=None, profile=False):
        """One decode step through every layer, bottom-up, each layer on its own input ``xs[layer]``
        [1, hidden], compared end to end with the reference. ``selection`` = (positions, valid)
        replaces the indexer on every layer. Returns the finished DecodeStep (already ended)."""
        position = len(self.mirror[self.layers[0]])
        step = begin_decode_step(self.plan, self.cache, self.acfg, position, user=self.user, profile=profile)
        d_override = None if selection is None else _picks_up(self.plan, *selection)
        ref_picks, dev_picks = {}, {}
        for n in self.layers:
            ref, mirror, x = self.ref[n], self.mirror[n], xs[n]
            out = self.out_down(self.tt[n](self.x_up(x), step, picks=d_override))
            assert torch.isfinite(out).all(), f"{label} L{n}: the device output is not finite"

            with torch.no_grad():
                if selection is not None:
                    result = ref.step(x, mirror, picks=selection[0], valid=selection[1])
                    check_pcc(f"{label} L{n} end to end", result.out, out, self.pcc_e2e)
                else:
                    result = ref.step(x, mirror, picks=None if ref.is_full else ref_picks[self.source[n]])
                ref_picks[n] = result.picks
                if selection is None:
                    if ref.is_full:
                        dev_picks[n] = self.picked(self.index_cache.picks(n))
                        self.check_selection(n, x, result.token, *dev_picks[n], f"{label} L{n}")
                    else:
                        assert self.index_cache.picks(n) is self.index_cache.picks(self.source[n])
                        assert step.selection.layer == self.source[n]
                    # The arithmetic: the reference attending over the device's picks.
                    positions, valid = dev_picks[self.source[n]]
                    with_device_picks = ref.stages(x, result.token, ref.gather(mirror, positions, valid), valid)["out"]
                    check_pcc(f"{label} L{n} end to end", with_device_picks, out, self.pcc_e2e)
                    # And the reference with its own picks. Where the two selections differ (near-ties
                    # at the cut, bounded by check_selection) the outputs differ by those rows, which a
                    # small top-k shows; the comparison is then logged, not asserted.
                    differ = len(set(positions[valid].tolist()) ^ set(ref_picks[self.source[n]].tolist()))
                    if differ == 0:
                        check_pcc(f"{label} L{n} end to end, reference picks", result.out, out, self.pcc_e2e)
                    else:
                        logger.info(
                            f"{label} L{n} end to end, reference picks differ on {differ} positions: "
                            f"pcc={pcc(result.out, out):.6f}"
                        )
            self.sync(n, position)
        step.end()
        if d_override is not None:
            d_override.deallocate()
        return step


def _log_spread(label, layout, positions, valid):
    per_chip = torch.bincount(layout.chip(positions[valid]), minlength=layout.num_chips)
    logger.info(
        f"SPREAD {label}: {int(valid.sum())} rows on {int((per_chip > 0).sum())} of {layout.num_chips} chips, "
        f"most on one chip {int(per_chip.max())}, chips with none {int((per_chip == 0).sum())}"
    )
    return per_chip


# ---------------------------------------------------------------------------------------------
# Stage by stage
# ---------------------------------------------------------------------------------------------
def _check_stages(model: _Model, context, label, seed=3):
    """A full layer, one step at position ``context`` after a prefill of that many tokens, every
    stage against the reference on the reference's input for that stage."""
    plan, acfg, tt, ref = model.plan, model.acfg, model.tt[0], model.ref[0]
    layout, stage = model.layout, model.pcc_stage
    xs = model.inputs(context + 1, seed)[0]
    model.prefill({0: xs[:context]})
    x, mirror = xs[context:], model.mirror[0]
    with torch.no_grad():
        token = ref.new_token(x, context)
    step = begin_decode_step(plan, model.cache, acfg, context, user=model.user)
    d_x = model.x_up(x)

    # new token: q_resid, and the two rows that land in the cache on the chip that owns the position
    d_q_resid, d_iq, d_iw = tt.new_token(d_x, step)
    check_pcc(f"{label} q_resid", token.q_resid, _whole(d_q_resid), stage)
    mirror.append(token.latent, token.index_key)
    stored_latent, stored_key = model.sync(0, context)
    check_pcc(f"{label} latent row in the cache", model.latent_to_device(token.latent[0]), stored_latent, stage)
    check_pcc(f"{label} index-key row in the cache", model.key_to_device(token.index_key[0]), stored_key, stage)
    q_heads, head_weights = model.queries[0]
    check_pcc(f"{label} indexer queries", model.key_to_device(token.index_q), q_heads, stage)
    check_pcc(f"{label} indexer head weights", token.index_w, head_weights, stage)
    # the two frames differ: the same rows compared without the map must not match
    rope = acfg.qk_rope_head_dim
    assert not torch.allclose(token.latent[0, -rope:], stored_latent[-rope:], atol=0.05), "k_rope is not interleaved"

    # selection, from the device's own queries over the device's stripes
    selection = tt.selection(step, d_iq, d_iw)
    assert selection is step.selection and selection.picks is model.index_cache.picks(0)
    positions, valid = model.picked(selection.picks)
    model.check_selection(0, x, token, positions, valid, label)  # scored over the mirror, which holds the stored rows
    K = positions.numel()
    _log_spread(label, layout, positions, valid)

    # locate: every chip's storage row and owner flag for every pick
    want_slot, want_mine = selection_table(layout, positions)  # [chips, K]
    offset = model.user * model.cache.cfg.slots
    rows = torch.stack([ttnn.to_torch(t).long().reshape(K) for t in ttnn.get_device_tensors(selection.rows)])
    own = torch.stack([ttnn.to_torch(t).float().reshape(K) for t in ttnn.get_device_tensors(selection.own)])
    assert torch.equal(own > 0.5, want_mine & valid[None]), f"{label}: the owner flags are wrong"
    assert torch.equal(rows[want_mine], (want_slot + offset)[want_mine]), f"{label}: a pick is located on the wrong row"
    assert int(rows.max()) < model.cache.cfg.max_users * model.cache.cfg.slots
    mask = _whole(selection.mask)[0, 0, 0]
    assert torch.equal(mask == 0, valid) and torch.equal(mask[~valid], torch.full_like(mask[~valid], MASKED))

    # gather + collect: the selected rows, whole on every chip, are the stored rows bit for bit
    gathered = tt.gather(selection)
    per_chip = torch.stack([ttnn.to_torch(t).float()[0, 0] for t in ttnn.get_device_tensors(gathered)])  # [chips, K, W]
    expected = ref.gather(mirror, positions, valid)  # the reference's frame, what the reference attends over
    d_expected = model.latent_to_device(expected)  # the same rows as the cache holds them
    assert torch.equal(per_chip.sum(0), d_expected), f"{label}: the gathered rows are not the stored rows"
    assert not per_chip[~(want_mine & valid[None])].any(), f"{label}: a chip returned a row it does not own"
    selected = _whole(tt.collect(gathered))[0, 0]
    wrong = (selected != d_expected).any(dim=-1).nonzero().flatten()
    assert wrong.numel() == 0, f"{label}: {wrong.numel()} of {K} collected rows differ, first {wrong[:5].tolist()}"
    logger.info(f"{label}: {K} selected rows ({int(valid.sum())} valid) collected bit-equal to the cache on every chip")

    # attention on the reference's inputs, with the device's picks
    with torch.no_grad():
        r = ref.stages(x, token, expected, valid)
        ra = ref.absorbed_stages(x, token, expected, valid)
    d_q, d_gate = tt.query_gate(d_x, _rows_up(plan, token.q_resid))
    check_pcc(f"{label} gate", r["gate"], plan.to_torch_head_concat(d_gate, dim=-1)[0, 0], stage)
    d_qh = tt.attention.queries(d_q, step.cos, step.sin)
    r_q = frames.tail_to_device(r["q"], acfg.qk_rope_head_dim)  # [H, 1, qk], rope dims interleaved
    check_pcc(f"{label} q", r_q, plan.to_torch_head_concat(d_qh, dim=1)[0], stage)

    head_mapper = plan.shard(head_dim=1)
    d_ref_q = ttnn.from_torch(
        r_q[None].contiguous(),
        layout=ttnn.TILE_LAYOUT,
        device=plan.mesh_device,
        dtype=ttnn.bfloat16,
        mesh_mapper=head_mapper,
    )
    d_selected = _rows_up(plan, d_expected)
    if tt.core == "absorb":
        d_logits, _ = tt.absorbed_logits(d_ref_q, d_selected, selection.mask)
        logits = plan.to_torch_head_concat(d_logits, dim=2)[0, 0].float()  # [H, K]
        check_pcc(f"{label} logits", ra["logits"][:, valid], logits[:, valid], stage)
        assert (
            K == int(valid.sum()) or logits[:, ~valid].max() < 0.5 * MASKED
        ), f"{label}: a masked entry kept its logit"
        d_attn = tt.absorbed_core(d_ref_q, d_selected, selection.mask)
    else:
        d_attn = tt.expanded_core(d_ref_q, d_selected, selection.mask)
    check_pcc(f"{label} core", r["attn"], plan.to_torch_head_concat(d_attn, dim=1)[0], stage)

    # the sublayer from the device's own q_resid and collected rows
    d_out = tt.attend(d_x, d_q_resid, _rows_up(plan, d_expected), selection, step)
    check_pcc(f"{label} attend", r["out"], model.out_down(d_out), model.pcc_e2e)
    step.end()


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("config, context", [("narrow", 300), ("narrow-top64", 300)], ids=["all-rows", "top64"])
def test_decode_stages_one_chip(device, config, context, dtype, core):
    plan = MeshPlan(device, hidden_axis=None)
    model = _Model(plan, CONFIGS[config], dtype, max_context=512, core=core)
    _check_stages(model, context, f"one chip {config} {dtype} {core} P{context}")
    model.cache.deallocate()


@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("config", ["narrow", "narrow-top64"])
@pytest.mark.parametrize("start", [0, 126], ids=["empty", "P126"])
def test_decode_steps_one_chip(device, config, start, core):
    """Three steps in a row through a full, a full and a shared layer on one chip, from an empty
    context and over the end of a block."""
    plan = MeshPlan(device, hidden_axis=None)
    model = _Model(plan, CONFIGS[config], "bf16", max_context=256, types=full_layers(3), core=core)
    model.prefill(model.inputs(start, seed=5))
    xs = model.inputs(3, seed=6)
    for t in range(3):
        model.step({n: x[t : t + 1] for n, x in xs.items()}, f"one chip {config} {core} P{start + t}")
    model.cache.deallocate()


@galaxy
@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize(
    "config, context",
    [
        ("narrow", 40),
        ("narrow", 3000),
        ("narrow-top64", 3000),
        ("narrow", 70000),
        ("full-width", 3000),
        ("narrow", 1048576 - ROUND),
    ],
    ids=["narrow-P40", "narrow-P3000", "narrow-top64-P3000", "narrow-P70000", "full-width-P3000", "narrow-one-million"],
)
def test_decode_stages_sc1(mesh_device, config, context, dtype, core):
    """Every stage on the 8x4 mesh: a context far below the top-k (40: 2008 of the 2048 list
    entries are not picks), above it (3000), with a local top-k on every chip (top-64 at 3000, and
    the real top-k at 70,000, where a stripe is longer than 2048 rows), at Hy4's widths, and at one
    million positions (1,044,480: stripes of 32,768 rows, so those cases run alone and last)."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    assert (plan.hidden_tp, plan.head_tp) == (4, 8)
    model = _Model(plan, CONFIGS[config], dtype, max_context=context + ROUND, core=core)
    _check_stages(model, context, f"sc1 {config} {dtype} {core} P{context}")
    model.cache.deallocate()


def _queries_unrounded(indexer):
    """TtHy4Indexer.queries without its last line, the rounding of the query heads to bf16: a
    what-if for the selection's precision, not something the engine runs."""
    from models.experimental.hy4_preview.tt.indexer import _cols, _rope

    def queries(q, cos, sin):
        c = indexer.cfg
        T, rope = q.shape[1], c.qk_rope_head_dim
        cos = ttnn.reshape(ttnn.typecast(cos, ttnn.float32), [1, T, 1, rope])
        sin = ttnn.reshape(ttnn.typecast(sin, ttnn.float32), [1, T, 1, rope])
        q_rope = _rope(_cols(q, 0, rope), cos, sin, indexer.rope_swap, indexer.ckc)
        q = ttnn.concat([q_rope, _cols(q, rope, c.head_dim)], dim=-1)
        return ttnn.matmul(q, indexer.hadamard, compute_kernel_config=indexer.ckc, dtype=ttnn.float32)

    return queries


@galaxy
@pytest.mark.parametrize("queries", ["bf16-queries", "fp32-queries"])
@pytest.mark.parametrize("head_sum", ["matmul", "fp32"], ids=["matmul-sum", "fp32-sum"])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_selection_precision_sc1(mesh_device, dtype, head_sum, queries):
    """What moves the selection at 70,000 positions, where the cut runs through the densest scores
    of these tests: every stage again with the indexer summing its heads with a second matmul or in
    elementwise fp32 (TtHy4Indexer ``head_sum``), and with the query heads rounded to bf16 as the
    indexer does or left in fp32 (a what-if). check_selection splits each selection into the
    device's arithmetic and the precision of its queries, next to HF's own arithmetic in bf16."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    model = _Model(plan, NARROW, dtype, max_context=70000 + ROUND, head_sum=head_sum)
    if queries == "fp32-queries":
        model.tt[0].indexer.queries = _queries_unrounded(model.tt[0].indexer)
    _check_stages(model, 70000, f"sc1 narrow {dtype} {head_sum} sum {queries} P70000")
    model.cache.deallocate()


# ---------------------------------------------------------------------------------------------
# End to end on the 8x4 mesh
# ---------------------------------------------------------------------------------------------
@galaxy
@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize(
    "config, start",
    [
        ("narrow", 0),
        ("narrow", 126),
        ("narrow", 2046),
        ("narrow", ROUND - 2),
        ("narrow-top64", 62),
        ("full-width", 126),
    ],
    ids=["narrow-empty", "narrow-block-edge", "narrow-topk-edge", "narrow-round-edge", "top64-topk-edge", "full-width"],
)
def test_decode_steps_sc1(mesh_device, config, start, dtype, core):
    """Four steps in a row through a full, a full and a shared layer (the shared one reusing the
    picks of the full layer below it), starting where the interesting things happen: an empty
    context; the end of a block (the token after it lands on the next chip); the context that
    crosses the top-k (2046, 2047 keep everything, 2048 onwards cut); and the end of a round (the
    next token starts every chip's second block)."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    model = _Model(plan, CONFIGS[config], dtype, max_context=start + ROUND, types=full_layers(3), core=core)
    assert model.types == ["full", "full", "shared"]
    model.prefill(model.inputs(start, seed=7))
    xs = model.inputs(4, seed=8)
    chips = set()
    for t in range(4):
        model.step({n: x[t : t + 1] for n, x in xs.items()}, f"sc1 {config} {dtype} {core} P{start + t}")
        chips.add(int(model.layout.chip(start + t)))
    if start % 128 >= 126:
        assert len(chips) == 2, "the steps did not cross a block boundary"
    model.cache.deallocate()


@galaxy
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("config, prompt", [("narrow", 300), ("narrow-top64", 3000)], ids=["below-topk", "top64"])
def test_decode_over_leftover_rows_sc1(mesh_device, config, prompt, dtype):
    """The cache as a prefill leaves it for a prompt of ``prompt`` tokens (tt/attention_decode.py,
    "What a step needs from the cache"): valid rows at the positions below the one decode starts
    at, and something else in every other row of the stripes: the last prompt token's own row,
    the padding of the last chunk, whatever an earlier, longer context left. Here those rows are
    three times the size of real ones, so an index key among them would win the top-k if it were
    visible, and a latent among them would show in the output.

    Decode starts by re-running the last prompt token (position ``prompt`` - 1, a row the cache
    already holds), then generates one more. Below the top-k the list of 2048 has 300 real picks, and
    the other 1748 entries name leftover rows, which every chip reads and multiplies by zero."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    model = _Model(plan, CONFIGS[config], dtype, max_context=2 * ROUND, types=full_layers(3))
    capacity, valid = model.cache.cfg.capacity, prompt - 1
    assert capacity == 2 * ROUND
    xs = model.inputs(valid, seed=51)
    g = torch.Generator().manual_seed(52)
    for n in model.layers:
        ref = model.ref[n]
        with torch.no_grad():
            entries = ref.prefill_cache(xs[n])
        latent = 3.0 * torch.randn(capacity, model.acfg.latent_dim, generator=g)
        latent[:valid] = entries.latent
        keys = None
        if ref.is_full:
            keys = 3.0 * torch.randn(capacity, model.icfg.head_dim, generator=g)
            keys[:valid] = entries.index_keys
        model.prefill_rows(n, latent, keys)
        mirror = model.mirror[n]  # what the storage holds; the reference sees the valid rows only
        model.mirror[n] = DecodeCache(mirror.latent[:valid], None if keys is None else mirror.index_keys[:valid])
    steps = model.inputs(2, seed=53)
    for t in range(2):
        model.step({n: x[t : t + 1] for n, x in steps.items()}, f"sc1 leftovers {config} {dtype} P{valid + t}")
    # the rows above the two the steps wrote are still the leftovers
    after = model.cache.to_torch(0, 0, capacity)
    assert after[valid + 2 :].abs().mean() > 2.0, "the leftover rows are gone: the test did not exercise them"
    model.cache.deallocate()


@prefill_fabric
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_decode_steps_on_the_prefill_fabric(mesh_device, l1_small, dtype, expect_error):
    """The decode step on the mesh as the prefill stack opens it. Decode reads the cache prefill
    writes, so both have to run on one MeshDevice, and the fixtures of the other tests open it
    differently (plain FABRIC_2D, default payload, no L1_SMALL region). Every stage at Hy4's width
    above the top-k, then three steps through a full, a full and a shared layer over the end of a
    block.

    With the prefill stack's own 1536 bytes of L1_SMALL the step does not get through: every
    ttnn.all_reduce / all_gather program keeps global semaphores there, and the tenth collective of
    the first step finds the region full. That case pins the failure, so that it is noticed when
    it goes away; the other opens the same mesh with DECODE_L1_SMALL and reports what the steps
    took."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)

    def run():
        model = _Model(plan, FULL_WIDTH, dtype, max_context=3000 + ROUND)
        _check_stages(model, 3000, f"torus full-width {dtype} P3000")
        model.cache.deallocate()
        model = _Model(plan, FULL_WIDTH, dtype, max_context=ROUND, types=full_layers(3))
        model.prefill(model.inputs(126, seed=7))
        xs = model.inputs(3, seed=8)
        for t in range(3):
            model.step({n: x[t : t + 1] for n, x in xs.items()}, f"torus full-width {dtype} P{126 + t}")
        model.cache.deallocate()

    if l1_small == PREFILL_L1_SMALL:
        with expect_error(RuntimeError, "L1_SMALL"):
            run()
        return
    run()
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL)
    logger.info(
        f"MEASURED L1_SMALL on the torus mesh after the decode steps ({dtype}): "
        f"{view.total_bytes_allocated_per_bank} of {view.total_bytes_per_bank} bytes per core in use; "
        f"the prefill stack opens the mesh with {PREFILL_L1_SMALL}"
    )
    assert view.total_bytes_allocated_per_bank > PREFILL_L1_SMALL, "a decode step now fits the prefill stack's L1_SMALL"


@galaxy
@pytest.mark.parametrize("core", CORES)
def test_decode_second_user(mesh_device, core):
    """A step for user 1 of a three-user cache: its rows come from its own stripe, and the other
    users' stripes are not written."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    model = _Model(plan, NARROW_TOP64, "bf16", max_context=ROUND, max_users=3, user=1, core=core)
    # users 0 and 2 hold other contexts: different rows at the same slots
    for other, seed in ((0, 31), (2, 32)):
        with torch.no_grad():
            entries = model.ref[0].prefill_cache(model.inputs(200, seed)[0])
        model.prefill_rows(0, entries.latent, entries.index_keys, mirror=False, user=other)
    before = {u: model.cache.stripes_to_torch(0, LATENT, u) for u in (0, 2)}
    model.prefill(model.inputs(200, seed=33))
    xs = model.inputs(2, seed=34)
    for t in range(2):
        model.step({0: xs[0][t : t + 1]}, f"sc1 user 1 of 3 {core} P{200 + t}")
    for u in (0, 2):
        assert torch.equal(model.cache.stripes_to_torch(0, LATENT, u), before[u]), f"user {u}'s stripe changed"
    model.cache.deallocate()


# ---------------------------------------------------------------------------------------------
# Selected rows spread unevenly over the chips
# ---------------------------------------------------------------------------------------------
SPREAD_CONTEXT = 17 * ROUND  # every chip holds 17 blocks = 2176 rows, more than one selection
SPREADS = ["one-chip", "last-chip", "two-chips", "one-mesh-row", "one-mesh-column", "even", "few-valid", "token-only"]


def _spread(spread, layout, mesh_shape, context, k, seed):
    """A hand-made selection for the token at position ``context``: (positions [k], valid [k]).
    Chips are row-major over ``mesh_shape``: a mesh row is ``cols`` consecutive chips (one head
    coordinate, the whole hidden axis), a mesh column every ``cols``-th chip (one hidden coordinate)."""
    rows, cols = mesh_shape
    g = torch.Generator().manual_seed(seed)
    positions = torch.arange(context)
    owner = layout.chip(positions)

    def among(chips, count):
        pool = positions[torch.isin(owner, torch.tensor(chips))]
        assert pool.numel() >= count, (chips, pool.numel(), count)
        return pool[torch.randperm(pool.numel(), generator=g)[:count]]

    valid = torch.ones(k, dtype=torch.bool)
    if spread == "one-chip":
        picks = among([5], k)
    elif spread == "last-chip":
        picks = among([31], k)
    elif spread == "two-chips":
        picks = torch.cat([among([0], k - 1), among([31], 1)])
    elif spread == "one-mesh-row":
        picks = among([2 * cols + c for c in range(cols)], k)
    elif spread == "one-mesh-column":
        picks = among([r * cols + cols - 1 for r in range(rows)], k)
    elif spread == "even":
        picks = among(list(range(layout.num_chips)), k - 1)
        picks = torch.cat([picks, torch.tensor([context])])  # and the new token itself
    elif spread == "few-valid":  # 5 picks on one chip plus the token; the rest of the list is not picks
        picks = torch.cat([among([7], 5), torch.tensor([context]), among(list(range(layout.num_chips)), k - 6)])
        valid = torch.arange(k) < 6
    else:  # the token attends to itself alone: the row written in this very step
        picks = torch.cat([torch.tensor([context]), among([9], k - 1)])
        valid = torch.arange(k) < 1
    order = torch.randperm(k, generator=g)
    return picks[order], valid[order]


@galaxy
@pytest.mark.parametrize("core", CORES)
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_uneven_spread_sc1(mesh_device, dtype, core):
    """2048 selected rows that are anything but spread evenly: all on one chip (the other 31 hold
    none), on two, on one mesh row, on one mesh column; a list with only a few real picks; and the
    token alone. The collected rows must be the stored rows bit for bit, and the step the reference's."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    model = _Model(
        plan, NARROW, dtype, max_context=SPREAD_CONTEXT + ROUND, types=["full", "shared"], layers=(1,), core=core
    )
    tt, k = model.tt[1], NARROW[1].topk
    model.prefill(model.inputs(SPREAD_CONTEXT, seed=11))
    xs = model.inputs(len(SPREADS), seed=12)[1]
    for t, spread in enumerate(SPREADS):
        context = SPREAD_CONTEXT + t
        label = f"sc1 {dtype} {core} {spread}"
        positions, valid = _spread(spread, model.layout, plan.mesh_shape, context, k, seed=13 + t)
        per_chip = _log_spread(label, model.layout, positions, valid)
        if spread in ("one-chip", "last-chip"):
            assert per_chip.max() == k and (per_chip == 0).sum() == 31
        model.step({1: xs[t : t + 1]}, label, selection=(positions, valid))

        # the collected rows of that selection, read once more on their own
        step = begin_decode_step(plan, model.cache, model.acfg, context, user=model.user)
        d_picks = _picks_up(plan, positions, valid)
        selection = tt.locate(d_picks, step)
        selected = _whole(tt.selected_rows(selection))[0, 0]
        expected = model.latent_to_device(model.ref[1].gather(model.mirror[1], positions, valid))
        wrong = (selected != expected).any(dim=-1).nonzero().flatten()
        assert wrong.numel() == 0, f"{label}: {wrong.numel()} of {k} collected rows differ, first {wrong[:5].tolist()}"
        selection.deallocate()
        d_picks.deallocate()
        step.end()
    model.cache.deallocate()


# ---------------------------------------------------------------------------------------------
# Checkpoint weights
# ---------------------------------------------------------------------------------------------
@galaxy
@pytest.mark.parametrize("layer", range(6), ids=lambda layer: f"layer{layer}")
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_decode_step_sc1_golden(mesh_device, layer, dtype):
    """Layers 0-5 with the checkpoint's weights on their real inputs: the last token of the recorded
    128-token prompt as a decode step over a cache of the first 127, against the reference and
    against what HF (BF16) computed for that token. A shared layer is handed the selection its full
    layer makes at this length, which is every position."""
    golden = hf_harness.find_golden()
    if golden is None:
        pytest.skip(f"no golden in {hf_harness.GOLDEN_ROOT}")
    types = golden["meta"]["indexer_types"]
    rec = golden["layers"][layer]
    x = rec["attn_in"].float()
    last = x.shape[0] - 1
    full = types[layer] == "full"
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    model = _Model(
        plan,
        FULL_WIDTH,
        dtype,
        max_context=ROUND,
        types=types,
        layers=(layer,),
        refs={layer: _checkpoint_reference(layer, full)},
    )
    model.prefill({layer: x[:last]})
    selection = None
    if not full:
        k = FULL_WIDTH[1].topk
        selection = (torch.arange(k), torch.arange(k) <= last)
    label = f"sc1 real {dtype} L{layer} golden"
    step_x = {layer: x[last:]}
    step = begin_decode_step(plan, model.cache, model.acfg, last, user=0)
    d_picks = None if selection is None else _picks_up(plan, *selection)
    out = model.out_down(model.tt[layer](model.x_up(step_x[layer]), step, picks=d_picks))
    with torch.no_grad():
        picks, valid = selection if selection is not None else (None, None)
        result = model.ref[layer].step(step_x[layer], model.mirror[layer], picks=picks, valid=valid)
    if full:
        model.check_selection(layer, step_x[layer], result.token, *model.picked(model.index_cache.picks(layer)), label)
    check_pcc(f"{label} end to end", result.out, out, model.pcc_e2e)
    check_pcc(f"{label} vs HF", rec["attn_out"][last].float(), out[0], model.pcc_e2e)
    step.end()
    model.cache.deallocate()


@galaxy
@pytest.mark.parametrize("head_sum", ["matmul", "fp32"], ids=["matmul-sum", "fp32-sum"])
@pytest.mark.parametrize("dtype", list(DTYPES))
def test_decode_step_sc1_checkpoint_long_context(mesh_device, dtype, head_sum):
    """Layer 1 (full) and layer 2 (shared) with the checkpoint's weights at 70,000 positions: every
    chip runs its local top-2048 and the shared layer reuses the result. The hidden states are random
    rows of unit RMS (real ones at this length need the whole model). With both ways the indexer
    can sum its heads."""
    context = 70000
    types = full_layers(3)
    refs = {1: _checkpoint_reference(1, True), 2: _checkpoint_reference(2, False)}
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    model = _Model(
        plan, FULL_WIDTH, dtype, max_context=context + ROUND, types=types, layers=(1, 2), refs=refs, head_sum=head_sum
    )
    model.prefill(model.inputs(context, seed=21))
    xs = model.inputs(2, seed=22)
    for t in range(2):
        model.step({n: x[t : t + 1] for n, x in xs.items()}, f"sc1 real {dtype} {head_sum} sum P{context + t}")
    model.cache.deallocate()


# ---------------------------------------------------------------------------------------------
# Latency of the pieces
# ---------------------------------------------------------------------------------------------
# id -> (context, users the cache holds, core, the indexer's head sum). The step takes one user per
# call, so a batch of users is that many steps in a row; what a cache of 8 users changes for one
# user's step is the size of the table cache.gather reads from (users x slots rows).
# cache.gather untilizes that whole table, so the cases whose table holds 32,768 rows or more
# (128K with 8 users, and every one-million case) run alone and last.
LATENCY_CASES = {
    "4K-1user-absorb": (4096, 1, "absorb", "matmul"),
    "4K-1user-expand": (4096, 1, "expand", "matmul"),
    "4K-8users-absorb": (4096, 8, "absorb", "matmul"),
    "128K-1user-absorb": (131072, 1, "absorb", "matmul"),
    "128K-1user-expand": (131072, 1, "expand", "matmul"),
    "128K-1user-absorb-fp32sum": (131072, 1, "absorb", "fp32"),
    "128K-8users-absorb": (131072, 8, "absorb", "matmul"),
    "one-million-1user-absorb": (1048576 - ROUND, 1, "absorb", "matmul"),
    "one-million-1user-expand": (1048576 - ROUND, 1, "expand", "matmul"),
    "one-million-1user-absorb-fp32sum": (1048576 - ROUND, 1, "absorb", "fp32"),
    "one-million-8users-absorb": (1048576 - ROUND, 8, "absorb", "matmul"),
}
STAGED_STEPS = 3
QUEUED_STEPS = 16


@galaxy
@pytest.mark.parametrize("case", list(LATENCY_CASES))
def test_decode_step_latency(mesh_device, case):
    """Wall time of one user's decode step at Hy4's widths, BFP8 weights and cache, on a full layer
    and on the shared layer above it. Untraced and unoptimised. Two clocks:

    * staged: every piece closed by a device synchronize (DecodeStep ``profile=True``): the step's
      host uploads, new token (projections, cache append), selection (scores, top-k, merge), locate,
      row gather, collect, attention. The synchronizes add time of their own, so the pieces add up to
      more than a step costs.
    * queued: QUEUED_STEPS steps issued back to back with nothing read back and one synchronize at
      the end (and, as a cross-check of that synchronize, a readback of every chip after it). This
      is what a step costs as the module stands; the shared layer's share is the difference
      between running both layers and running the full layer alone.

    The first step of every user compiles the programs and is not counted. The cache holds random
    rows for the users that are stepped (the first and the last); nothing here checks values beyond
    the output being finite."""
    context, users, core, head_sum = LATENCY_CASES[case]
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    types = full_layers(3)
    full, shared = 1, 2
    model = _Model(
        plan,
        FULL_WIDTH,
        "bfp8",
        max_context=context + ROUND,
        types=types,
        layers=(full, shared),
        core=core,
        max_users=users,
        head_sum=head_sum,
        record=False,
    )
    stepped = sorted({0, users - 1})
    g = torch.Generator().manual_seed(41)
    for n in model.layers:
        latent = torch.randn(context, 576, generator=g)
        keys = torch.randn(context, 128, generator=g) if types[n] == "full" else None
        for user in stepped:
            model.prefill_rows(n, latent, keys, mirror=False, user=user)
        del latent, keys
    d_x = {n: model.x_up(x) for n, x in model.inputs(1, seed=42).items()}  # the module does not free its input
    position = {user: context for user in stepped}

    def synchronize():
        ttnn.synchronize_device(mesh_device)
        return time.perf_counter()

    def run(user, layers, profile=False, keep=False):
        """One step of ``user`` through ``layers``; returns (step, outputs or None)."""
        step = begin_decode_step(plan, model.cache, model.acfg, position[user], user=user, profile=profile)
        position[user] += 1
        outs = [model.tt[n](d_x[n], step) for n in layers]
        if keep:
            return step, outs
        for out in outs:
            out.deallocate(True)
        step.end()
        return step, None

    for user in stepped:
        # the first step compiles; its output is the one that is checked
        step, outs = run(user, model.layers, keep=True)
        for out in outs:
            assert torch.isfinite(model.out_down(out)).all()
        step.end()

        # staged
        staged = {n: {} for n in model.layers}
        upload = 0.0
        for _ in range(STAGED_STEPS):
            start = synchronize()
            step = begin_decode_step(plan, model.cache, model.acfg, position[user], user=user, profile=True)
            upload += synchronize() - start
            position[user] += 1
            for n in model.layers:
                before = dict(step.seconds)
                model.tt[n](d_x[n], step).deallocate(True)
                for stage, seconds in step.seconds.items():
                    if seconds > before.get(stage, 0.0):  # a shared layer adds nothing to "selection"
                        staged[n][stage] = staged[n].get(stage, 0.0) + seconds - before.get(stage, 0.0)
            step.end()

        # queued: the full layer alone, then both layers
        queued = {}
        for name, layers in (("full", (full,)), ("both", model.layers)):
            start = synchronize()
            for _ in range(QUEUED_STEPS):
                run(user, layers)
            done = synchronize()
            for shard in ttnn.get_device_tensors(model.cache.chip_id):  # a read of every chip, behind the work
                ttnn.to_torch(shard)
            queued[name] = (1e3 * (done - start) / QUEUED_STEPS, 1e3 * (time.perf_counter() - start) / QUEUED_STEPS)
        per_layer = {full: queued["full"], shared: tuple(b - f for b, f in zip(queued["both"], queued["full"]))}

        for n in model.layers:
            kind = types[n]
            mean = {stage: 1e3 * seconds / STAGED_STEPS for stage, seconds in staged[n].items()}
            pieces = sum(mean.get(stage, 0.0) for stage in STAGES)
            step_ms = per_layer[n][0]
            logger.info(
                f"MEASURED decode step {case}: {kind} layer, context {context}, cache of {users} user(s), "
                f"user {user}, core {core}, head sum {head_sum} | staged ms: "
                + ", ".join(f"{stage} {mean[stage]:.2f}" for stage in STAGES if stage in mean)
                + f"; pieces {pieces:.2f} | queued ms per step: {step_ms:.2f} "
                f"({per_layer[n][1]:.2f} with a readback of every chip behind the {QUEUED_STEPS} steps) | "
                f"{TARGET_USERS} users are {TARGET_USERS} such steps: {TARGET_USERS * step_ms:.1f} ms, against "
                f"{LAYER_TARGET_MS:.3f} ms per layer for the batch ({TARGET_USERS * step_ms / LAYER_TARGET_MS:.0f}x) "
                f"and the parent issue's estimate of {LAYER_ESTIMATE_MS} ms"
            )
            # a shared layer runs no selection and no locate of its own: it reads the full layer's
            expected = [stage for stage in STAGES if kind == "full" or stage not in ("selection", "locate")]
            assert pieces > 0 and all(stage in mean for stage in expected), sorted(mean)
        logger.info(
            f"MEASURED decode step {case}: user {user}, the step's host uploads (cos, sin, append indices, "
            f"query position; once per token, shared by the layers): {1e3 * upload / STAGED_STEPS:.2f} ms"
        )
    for t in d_x.values():
        t.deallocate(True)
    model.cache.deallocate()
