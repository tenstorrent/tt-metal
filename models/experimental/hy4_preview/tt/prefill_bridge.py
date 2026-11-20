# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill by the prefill stack, decode by the SC1 modules: the hand-over of the caches
(tenstorrent/tt-metal#65893).

The prompt is prefilled by models/demos/deepseek_v3_d_p (the ``hy4_preview`` adapter of
#65935 / #66245, iHC included), in process, on the mesh decode runs on. This is the one module of
hy4_preview that imports that stack; the decode modules stay free of it. It builds the stack's
runtime for a number of layers, runs its prefill of a prompt into a user slot, and copies what the
prefill left behind into the cache the decode step reads (tt/kv_cache.py TtHy4KvCache).

What a prefill hands over
-------------------------
The two caches and the context length, nothing else. The stack returns no state of the last
position (its transformer has no head, and its last layer writes its rows and stops), so decode
makes the first new token by running its own step for the LAST PROMPT TOKEN: after a prefill of N
tokens, step 0 feeds token N - 1 at position N - 1 over the rows of positions [0, N - 1), rewrites
row N - 1 with its own values and predicts token N. No change to the stack is needed for that, and
row N - 1 of the hand-over is never read.

Where the stack keeps the rows, and where decode wants them
-----------------------------------------------------------
    prefill   kvpe    one tensor, per chip [users * L, 1, max_seq / 32, 656] fp8_e4m3 ROW_MAJOR:
                      a packed row per token, 512 FP8 latent | 4 FP32 scales | 64 BF16 k_rope
                      (MlaKvCacheFormat.SCALED_FP8), or [.., 576] bf16 ROW_MAJOR (BF16_RM);
                      slot = user * L + layer
              index   one tensor, per chip [users * F, 1, max_seq / 32, 128] bfloat8_b TILE;
                      slot = user * F + rank of the full-indexer layer
    decode    latent[layer], index_key[layer]: per chip [users, 1, slots, 576 | 128] TILE, BFP8 or BF16

Both stripe a context the same way (chip = (p // 128) % 32 row-major over the 8x4 mesh,
row = (p // 4096) * 128 + p % 128) and both hold the same numbers (tt/frames.py: k_rope in
interleaved pairs, index keys as [rope | pass] in the Hadamard basis). Only the container differs.

The three ways across, and which are here
-----------------------------------------
(c) ``fill_from_host``: read both prefill tensors back once with the stack's helpers
    (read_chip_slabs, KvStripe.destripe, MlaKvCache.unpack_host) and write each (user, layer)
    with TtHy4KvCache.write_prefill_host. Works for either prefill format. It is the correctness
    reference: the rows pass through natural token order on the host, so the stack's stripe and
    ours are each used once and must agree.
(b) ``fill_on_device``: one conversion after the prefill, on the device, no fabric traffic (the
    stripes are the same, so every chip converts its own rows):
      index keys   ttnn.slice of the slot (DRAM) -> fill_cache_for_user_. The tiles are copied as
                   they are: bit-exact, for either prefill format.
      latents      from a BF16_RM prefill cache: ttnn.slice of the slot -> to_layout(TILE) ->
                   typecast to the cache dtype -> fill_cache_for_user_.
                   from the packed scaled-FP8 cache: NOT POSSIBLE with the ops ttnn has. The row
                   mixes FP8, FP32 and BF16 bytes inside one fp8_e4m3 tensor;
                   ttnn.experimental.deepseek_prefill has pack_scaled_fp8_kv_cache and
                   per_token_cast_back (FP8 latent + FP32 scales -> bf16) but no op that splits a
                   packed row back into its three fields, and nothing reinterprets bytes on the
                   device. The missing op is the inverse of pack_scaled_fp8_kv_cache; with it the
                   chain is unpack -> per_token_cast_back -> concat -> to_layout -> typecast -> fill.
                   Until then ``fill_on_device`` moves the index keys on the device and reports
                   the latents as left for ``fill_from_host``.
(a) decode reading the prefill storage in place: not built, and not possible for a decode that
    also appends. Reading exists in the stack (sparse_push_plan + sparse_kv_push +
    sparse_sdpa(SCALED_FP8), nobody has run them with a handful of query rows); writing does not:
    update_padded_kv_cache writes 32-row windows of a prefill chunk and there is no single-row
    append into a packed row. Decode would also have to move from head-parallel attention over
    gathered rows to that one fused op.

Why (b), and what it is not. Measured on the 8x4 Galaxy (tests/test_prefill_bridge.py):

* What ttnn reads out of the prefill storage where it is. A slot can be sliced out of either
  tensor (ND-sharded source, plain DRAM copy; the bytes are the stored bytes). On a packed slot:
  ttnn.embedding refuses (it takes a bf16 table); typecast fp8_e4m3 -> bfloat16 of the 512 latent
  bytes returns the FP8 values (all but a few in 100,000, those off by up to 0.014 on values up to
  448), but the scales they have to be multiplied by are 16 fp8-typed bytes per row that nothing
  reads as fp32, and k_rope is 128 more; to_layout and typecast of the whole packed row run and
  mean nothing. So what is missing is narrow: the scales and k_rope out of a packed row. On a
  BF16_RM slot: ttnn.embedding gathers rows from the slot's slice
  (2048 rows in 0.11 ms warm, the rows bit-equal), and refuses the storage itself ("does not
  support sharded weights"), so even there a decode gather needs a copy of the slot first.
* The same rows through (b) and (c), layers 0-5 of a 1022- and a 5118-token prompt: index keys
  bit-equal for both prefill formats and both decode cache dtypes; BF16_RM latents bit-equal in a
  BF16 decode cache and PCC 0.999975 or better in a BFP8 one (the device's typecast and the host's
  packing round a value differently, one BFP8 step at most).
* Time. (c): 1.1 s to read both tensors of 2 users x 6 layers x 8192 positions and 0.04 to 0.08 s
  to write six layers back; at 131,072 positions 8.5 s and 0.9 s, which scales to 16 minutes for
  one user at 1M and 78 layers. (b) at 131,072 positions (4,096 rows per chip): 1.8 to 4.3 ms for
  the six layers of latents and three of index keys (0.2 to 0.5 ms per stripe; 2.0 to 2.2 ms with a
  read of every chip queued behind it), which scales by rows to well under a second for one user at 1M
  and 78 layers. Not measured at 32,768 rows per chip.
* Memory. The second copy is what the decode cache takes: 50.6 KB per token in BFP8 next to the
  prefill storage's 57.8 KB (scaled FP8) or 92.7 KB (BF16_RM).

Against the target (8 users x 1M context on one Galaxy, #65884): the caches are about 460 GB of
the 563 GB left beside one set of weights, so there is room for ONE copy of the 8 users. (b) and
(c) hold two of whatever they convert while they run and are bring-up steps. Two ways out:
* (a), decode on the prefill storage: needs the single-row append above and a decode attention
  built on the stack's sparse ops. Nothing is copied.
* (b) as a staging step: the prefill storage sized for ONE user (57.8 GB at 1M in scaled FP8, with
  the unpack op; 92.7 GB in BF16_RM today), every prompt prefilled into that one slot and
  converted into its user's stripes of the decode cache (405 GB for 8 users in BFP8;
  ``cache_user`` is that mapping). 463 to 498 GB, inside the 563, for under a second per
  hand-over. The decode cache then holds FP8 latents repacked as BFP8 (PCC 0.99997 against the
  rows it was filled from). Not built: this bridge takes whatever ``num_users`` the prefill has.
Either way the prefill storage should be freed (``release_caches``) or reused as soon as decode
has its rows, and the two sets of weights (below) are the larger problem.

What the hand-over costs in accuracy. Against HF's own cache rows (BF16, mapped with
tt/frames.py), layers 0-5, as the decode cache returns them: from the scaled-FP8 prefill cache
k_pass 0.9982 to 0.9996, k_rope 0.99986 or better, index keys 0.9990 or better; from BF16_RM
k_pass 0.9992 to 0.9999, index keys 0.9994 or better. Every row is its own (row_report: no row
matches a displaced HF row as well as its own; the same rows moved one block fail on 980 to 5082
of them). A decode step over the scaled-FP8 hand-over gives HF's attention output of the new
token at 0.9960 or better (both prompts), over the BF16_RM one at 0.9986 or better (the long
prompt; tests/test_model_prefill_decode.py). At 78 layers the rows carry the prefill stack's own
drift with depth (scaled-FP8 cache, 1021 rows per layer against HF's cache): k_pass 0.917 at its
lowest (layer 73) and 0.978 at the median layer, k_rope 0.996 or better, index keys 0.9947 or
better; no row is misplaced by the bridge at any depth it was checked at (6 layers). Decode over
those rows scores what tt/model.py reports, and markedly better over HF's own rows written in
their place (tests/test_model_full.py test_depth). The format matters more at depth than at six
layers: 78 layers after a 5118-token prompt, lowest over six decode steps, logits 0.888 and 54%
of HF's top-2048 picks over the scaled-FP8 hand-over, 0.980 and 90.7% over the BF16_RM one
(0.982 and 92.5% over HF's own rows), the BF16_RM rows converted on the device in 0.65 s where
the scaled-FP8 latents take 7.8 s through the host.

Users. Every method takes a user: the prefill slot and the decode cache's user row. A prefill is
one user per call (the stack's chunk loop); the hand-over of one user reads only that user's
slots on the device, but ``read_host`` transfers the whole tensors (the stack's reader does).

Device. Both sides must share one MeshDevice. The stack needs the torus fabric, relaxed init and
its packet payload; the decode modules' collectives keep their semaphores in L1_SMALL once the
region exists and need more of it than the stack's 1536 bytes (tt/attention_decode.py).
``device_params`` is the mesh both run on, with 32,768 bytes of L1_SMALL: the stack's prefill of
six layers leaves 3,456 in use, decode steps of one user bring it to 10,112 and batches of 1 and 8
users to 14,848. Depth does not add to it: 10,112 after decode steps of one user at 6, 12, 24, 48
and 78 layers alike (every layer runs the same programs). Everything here ran on that mesh, the
stack's prefill of all 78 layers included (tests/test_model_full.py).

Memory. The prefill weights and the decode weights are two sets (the stack's dev layout
replicates attention over the 8 sequence chips; decode shards it 1/32). Six layers of the stack
with its caches and compile buffers take 1.72 GB of DRAM per chip and build in 3.8 s from its
weight cache, plus 3.2 s to compile; ``release_model`` gives all of it back but the caches in
0.2 s (0.007 GB left). Decode's six layers take 1.1 GB, so both fit side by side at six layers.
At 78 layers they do not: the stack takes 20.90 GB per chip with a one-user 8K storage (12.90 at
48 layers, 6.51 at 24, 3.31 at 12) and decode 15.14 GB (tt/model.py), 36.0 of a chip's 34.18 GB.
So the order is prefill, hand over, ``release_model`` (0.4 s at 78 layers, 0.03 GB left), then
load the decode weights; tt/model.py Hy4Engine does that. The stack's 78 layers build in 36 to
150 s from its weight cache (cold or warm page cache) and compile in 9 to 16 s.

Not here: a prefill that starts off a chunk boundary (a cached prefix), the stack's trace mode, a
prefill of several users in one call.
"""

from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass, field

import torch

from models.experimental.hy4_preview.tt.kv_cache import INDEX_KEY, LATENT
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

MESH = (8, 4)
CHUNK = 4096  # the stack's chunk on 32 chips: 128-token blocks, the stripe decode addresses
PREFILL_LAYOUT = "glm_prefill_sp8_tp4"  # the layout the stack can run today (its tp32 is refused at start)
PREFILL_L1_SMALL = 1536  # what the stack opens the mesh with
SHARED_L1_SMALL = 32768  # what a mesh shared with the decode modules is opened with
DEFAULT_WEIGHT_CACHE = "generated/hy4_preview/prefill-cache/weights"
KV_FORMATS = ("scaled_fp8", "bf16_rm")
LATENT_DIM, ROPE_DIM, INDEX_KEY_DIM = 576, 64, 128
PAD_ID = 0


# ---------------------------------------------------------------------------------------------
# Pure arithmetic and host checks (no device, no prefill stack)
# ---------------------------------------------------------------------------------------------
def full_layers(indexer_types) -> list:
    return [layer for layer, kind in enumerate(indexer_types) if kind == "full"]


def kvpe_slot(user: int, layer: int, num_layers: int) -> int:
    """The slot (dim 0) of a user's layer in the prefill KVPE tensor."""
    assert 0 <= layer < num_layers, (layer, num_layers)
    return user * num_layers + layer


def index_slot(user: int, layer: int, indexer_types) -> int:
    """The slot of a user's full-indexer layer in the prefill index-key tensor."""
    full = full_layers(indexer_types)
    assert layer in full, f"layer {layer} is a shared indexer layer and has no index keys"
    return user * len(full) + full.index(layer)


def row_pcc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of every row of ``a`` [N, W] with the same row of ``b``: [N]. A row that
    is constant on either side scores 0."""
    a = a.double() - a.double().mean(dim=-1, keepdim=True)
    b = b.double() - b.double().mean(dim=-1, keepdim=True)
    denom = a.norm(dim=-1) * b.norm(dim=-1)
    return torch.where(denom > 0, (a * b).sum(-1) / denom.clamp_min(1e-300), torch.zeros_like(denom))


# Displacements a wrong stripe produces: a neighbour, a tile, a block (= the next chip), a round
# (= the next block of the same chip).
ROW_OFFSETS = (1, 32, 128, 4096)


# Two reference rows this alike cannot tell a position from its displaced twin: at layer 0 the
# k_pass of a token does not depend on its position, so a repeated token repeats the row.
TWIN_PCC = 0.9999


def row_report(ref: torch.Tensor, got: torch.Tensor, offsets=ROW_OFFSETS) -> dict:
    """Is row p of ``got`` the row p of ``ref`` ([N, W] each, position order)?

    A PCC over the whole tensor answers a weaker question: rows of a cache resemble one another
    (every row is a normalised latent of the same layer), so a cache whose rows sit one block away
    from where they belong still correlates. This scores every row against its own reference row
    and against the reference rows at ``offsets`` on either side. ``misplaced`` counts the rows
    that match a displaced reference row at least as well as their own; ``min_row`` / ``mean_row``
    are the row's own PCC. A correct hand-over has no misplaced row and a ``min_row`` near the
    tensor PCC. A displaced reference row that is the row's twin (TWIN_PCC) is not a candidate,
    and ``twins`` counts the rows that had one.
    """
    assert ref.shape == got.shape and ref.dim() == 2, (tuple(ref.shape), tuple(got.shape))
    n = ref.shape[0]
    own = row_pcc(ref, got)
    best_other = torch.full((n,), -2.0, dtype=torch.float64)
    twins = torch.zeros(n, dtype=torch.bool)
    for d in offsets:
        if d >= n:
            continue
        for shift in (d, -d):
            lo, hi = max(0, -shift), min(n, n - shift)  # rows p with 0 <= p + shift < n
            other = row_pcc(ref[lo + shift : hi + shift], got[lo:hi])
            twin = row_pcc(ref[lo + shift : hi + shift], ref[lo:hi]) > TWIN_PCC
            twins[lo:hi] |= twin
            other = torch.where(twin, torch.full_like(other, -2.0), other)
            best_other[lo:hi] = torch.maximum(best_other[lo:hi], other)
    a, b = ref.flatten().double(), got.flatten().double()
    whole = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    misplaced = best_other >= own
    return {
        "pcc": whole,
        "min_row": own.min().item(),
        "mean_row": own.mean().item(),
        "worst_row": int(own.argmin()),
        "misplaced": int(misplaced.sum()),
        "first_misplaced": int(misplaced.nonzero()[0]) if misplaced.any() else None,
        "best_displaced": best_other.max().item(),
        "twins": int(twins.sum()),
    }


def format_report(r: dict) -> str:
    return (
        f"pcc={r['pcc']:.6f} | rows: min {r['min_row']:.6f} (row {r['worst_row']}), mean {r['mean_row']:.6f}, "
        f"{r['misplaced']} misplaced (best displaced match {r['best_displaced']:.4f}, {r['twins']} rows with a twin)"
    )


@dataclass
class HostCaches:
    """Both prefill tensors on the host, as the stack's reader returns them: per chip, in linear
    chip order. ``kvpe`` [chips, users * L, rows, 656 | 576] physical rows, ``index``
    [chips, users * F, rows, 128]. ``destripe`` ([chips, rows, W], length) -> [length, W] and
    ``unpack`` (physical rows -> logical [.., 576]) are the stack's own (KvStripe.destripe,
    MlaKvCache.unpack_host); a host test passes stand-ins."""

    kvpe: torch.Tensor
    index: torch.Tensor
    num_layers: int
    indexer_types: tuple
    destripe: object
    unpack: object
    seconds: float = 0.0

    def latent(self, user: int, layer: int, length: int) -> torch.Tensor:
        """[length, 576] fp32, device frame: k_pass | k_rope interleaved."""
        slot = kvpe_slot(user, layer, self.num_layers)
        return self.unpack(self.destripe(self.kvpe[:, slot], length)).float()

    def index_key(self, user: int, layer: int, length: int) -> torch.Tensor:
        """[length, 128] fp32 as stored: [rope interleaved | pass] in the Hadamard basis."""
        slot = index_slot(user, layer, self.indexer_types)
        return self.destripe(self.index[:, slot], length).float()

    def is_full(self, layer: int) -> bool:
        return self.indexer_types[layer] == "full"


# ---------------------------------------------------------------------------------------------
# The mesh both sides run on
# ---------------------------------------------------------------------------------------------
def fabric_payload_size() -> int:
    from models.demos.common.prefill.adapter import get_adapter

    return get_adapter("hy4_preview").fabric_payload_size()


def device_params(l1_small_size: int = SHARED_L1_SMALL) -> dict:
    """The device parameters of the 8x4 mesh as the prefill stack needs it (torus fabric, relaxed
    init, its packet payload), with an L1_SMALL region large enough for the decode modules'
    collectives as well. A fresh dict for a pytest ``device_params`` fixture."""
    from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params

    return torus_xy_device_params(fabric_payload_size=fabric_payload_size(), l1_small_size=l1_small_size)


def open_mesh(l1_small_size: int = SHARED_L1_SMALL):
    """Open that mesh outside pytest (the stack's own opener)."""
    from models.demos.common.prefill.runners import runner_utils

    return runner_utils.open_mesh_device(MESH, fabric_payload_size(), l1_small_size=l1_small_size, sp_axis=0)


def dram(mesh_device) -> dict:
    """DRAM per chip in GB: total, allocated, free."""
    import ttnn

    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return {
        "total_gb": v.total_bytes_per_bank * v.num_banks / 1e9,
        "allocated_gb": v.total_bytes_allocated_per_bank * v.num_banks / 1e9,
        "free_gb": v.total_bytes_free_per_bank * v.num_banks / 1e9,
    }


def l1_small(mesh_device) -> dict:
    """L1_SMALL per core in bytes: size and what is in use."""
    import ttnn

    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL)
    return {"bytes": v.total_bytes_per_bank, "allocated": v.total_bytes_allocated_per_bank}


# ---------------------------------------------------------------------------------------------
@dataclass
class PrefillBridge:
    """The prefill stack's runtime for layers ``0 .. num_layers - 1`` on ``mesh_device``, its two
    caches for ``num_users`` users of ``max_seq_len`` positions each, and the hand-over to decode.

    ``max_seq_len`` must be a multiple of 8192 (whole 4096-token chunks, and the stack's slots start
    on a DRAM bank boundary). ``kv_format`` is the stack's KVPE encoding: "scaled_fp8" (its
    default, 704 bytes per token per layer) or "bf16_rm" (1152 bytes, the one ``fill_on_device``
    can convert). ``expert_dtype`` (a ttnn dtype) overrides the stack's BFP4 routed experts for an
    A/B run; None keeps them. ``seconds`` collects the wall time of every step.
    """

    mesh_device: object
    num_layers: int = 6
    num_users: int = 1
    max_seq_len: int = 8192
    kv_format: str = "scaled_fp8"
    kv_only_last_layer: bool = True
    weight_cache: str | None = None
    expert_dtype: object = None
    seconds: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.kv_format in KV_FORMATS, self.kv_format
        assert self.max_seq_len % (2 * CHUNK) == 0, f"max_seq_len {self.max_seq_len} is not a multiple of {2 * CHUNK}"
        assert tuple(self.mesh_device.shape) == MESH, f"the prefill stack runs on {MESH}, got {self.mesh_device.shape}"
        self.runtime = None
        self.kv_caches = None
        self.lengths = [0] * self.num_users  # context length per user; the caches keep none
        self.indexer_types = None

    # ------------------------------------------------------------------------------------------
    # build, prefill
    # ------------------------------------------------------------------------------------------
    def build(self):
        """Build the runtime (weights from the stack's weight cache, converted from the checkpoint
        where missing), allocate both caches and warm one chunk up. The warm-up writes token 0's
        rows into user slot 0, positions [0, 4096) of every layer: junk until a prompt overwrites it."""
        import ttnn
        from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
        from models.demos.deepseek_v3_d_p.tt.hy4 import layout as hy4_layout
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

        assert self.runtime is None, "already built"
        os.environ.setdefault(hy4_layout.LAYOUT_ENV, PREFILL_LAYOUT)
        os.environ.setdefault("PREFILL_TTNN_CACHE", self.weight_cache or DEFAULT_WEIGHT_CACHE)
        os.environ["PREFILL_MAX_SEQ_LEN"] = str(self.max_seq_len)
        adapter = get_adapter("hy4_preview")
        hf_config = adapter.load_hf_config()
        hf_config.max_seq_len = self.max_seq_len
        self.indexer_types = tuple(hf_config.indexer_types[: self.num_layers])
        params = PrefillRunParams(
            mesh_shape=MESH,
            num_layers=self.num_layers,
            first_layer_idx=0,
            is_first_rank=True,
            is_last_rank=True,
            max_seq_len=self.max_seq_len,
            chunk_size=CHUNK,
            num_users=self.num_users,
            capacity_factor=8,
            num_links=2,
            gate_mode_name=adapter.default_gate_mode,
            kv_only_last_layer=self.kv_only_last_layer,
            weight_cache_path=adapter.weight_cache_path(MESH, sp_axis=0),
            tp_shard_kv=adapter.tp_shard_kv_default,
            sp_axis=0,
            tp_axis=1,
            sparse_kv_cache_format=MlaKvCacheFormat(self.kv_format),
        )
        restore = None
        if self.expert_dtype is not None:
            # The adapter has no knob for the routed experts' dtype: the runtime config does.
            base, dtype = adapter._runtime_cls(), self.expert_dtype

            class _ExpertDtypeRuntime(base):
                def __init__(self, mesh_device, hf_config, state_dict, config):
                    config.routed_expert_weights_dtype = dtype
                    super().__init__(mesh_device=mesh_device, hf_config=hf_config, state_dict=state_dict, config=config)

            restore = adapter.__dict__.get("_runtime_cls")
            adapter._runtime_cls = lambda: _ExpertDtypeRuntime
        t0 = time.perf_counter()
        try:
            self.runtime = adapter.build_runtime(mesh_device=self.mesh_device, hf_config=hf_config, params=params)
        finally:
            if self.expert_dtype is not None:
                if restore is None:
                    del adapter._runtime_cls
                else:
                    adapter._runtime_cls = restore
        ttnn.synchronize_device(self.mesh_device)
        self.seconds["build_runtime"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.kv_caches = adapter.allocate_kv_cache(mesh_device=self.mesh_device, hf_config=hf_config, params=params)
        self.runtime.compile(self.kv_caches)
        ttnn.synchronize_device(self.mesh_device)
        self.seconds["compile"] = time.perf_counter() - t0
        self.stripe = self.runtime.kv_stripe
        self.rows_per_chip = self.stripe.rows_per_chip(self.max_seq_len)
        layout = BlockCyclicLayout()
        assert (self.stripe.num_chips, self.stripe.block) == (layout.num_chips, layout.block), self.stripe
        return self

    def prefill(self, token_ids, user: int = 0) -> int:
        """Prefill ``token_ids`` (ints, N of them) into ``user``'s slot from position 0, in chunks of
        4096. Returns N, the user's context length. Afterwards the user's slots hold positions
        [0, N); rows [N, ceil32(N)) hold the chunk padding's rows and everything above is whatever
        was there before."""
        import ttnn

        assert self.runtime is not None, "build() first"
        assert 0 <= user < self.num_users, f"user {user} of {self.num_users}"
        ids = [int(t) for t in (token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids)]
        n = len(ids)
        assert 0 < n <= self.max_seq_len, f"{n} tokens do not fit a cache of {self.max_seq_len}"
        t0 = time.perf_counter()
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            chunk = ids[start:end] + [PAD_ID] * (CHUNK - (end - start))
            self.runtime.prefill_chunk(
                self.runtime.make_chunk_input(chunk), self.kv_caches, slot_id=user, actual_start=start, actual_end=end
            )
        ttnn.synchronize_device(self.mesh_device)
        self.seconds.setdefault("prefill", []).append(time.perf_counter() - t0)
        self.lengths[user] = n
        return n

    # ------------------------------------------------------------------------------------------
    # (c) through the host
    # ------------------------------------------------------------------------------------------
    def read_slabs(self, tensor) -> torch.Tensor:
        """A striped device tensor per chip in linear chip order, [chips, slots, rows, width], with
        the stack's reader (the only host read that takes an fp8_e4m3 tensor: it reads the mesh
        through a composer, and ttnn.to_torch of one chip's fp8 tensor is refused)."""
        from models.demos.deepseek_v3_d_p.tt.hy4.kv_cache import read_chip_slabs

        return read_chip_slabs(tensor, self.mesh_device, self.stripe)

    def unpack_fields(self, physical: torch.Tensor):
        """Packed scaled-FP8 rows on the host -> (the FP8 latent values before their scales, fp32
        [.., 512]; the scales, fp32 [.., 4]; k_rope, bf16 [.., 64])."""
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import unpack_scaled_fp8_kv_cache

        assert self.kv_format == "scaled_fp8", self.kv_format
        return unpack_scaled_fp8_kv_cache(physical, self.kv_caches.kvpe.geometry)

    def read_host(self) -> HostCaches:
        """Both caches on the host, every user and layer (the stack's reader moves whole tensors)."""
        t0 = time.perf_counter()
        kvpe = self.read_slabs(self.kv_caches.kvpe.storage)
        index = self.read_slabs(self.kv_caches.index)
        host = HostCaches(
            kvpe=kvpe,
            index=index,
            num_layers=self.num_layers,
            indexer_types=self.indexer_types,
            destripe=self.stripe.destripe,
            unpack=self.kv_caches.kvpe.unpack_host,
            seconds=time.perf_counter() - t0,
        )
        assert kvpe.shape[1] == self.num_users * self.num_layers, tuple(kvpe.shape)
        assert index.shape[1] == self.num_users * len(full_layers(self.indexer_types)), tuple(index.shape)
        self.seconds.setdefault("read_host", []).append(host.seconds)
        return host

    def fill_from_host(self, cache, user: int, host: HostCaches | None = None, layers=None, cache_user=None) -> dict:
        """Path (c): write ``user``'s positions [0, N) of ``layers`` (default: every layer the cache
        and the prefill share) into ``cache`` (TtHy4KvCache) at ``cache_user`` (default: the same
        user). Rows of the touched rounds past N are zeroed. Returns the wall times."""
        import ttnn

        n = self.lengths[user]
        assert n > 0, f"user {user} has not been prefilled"
        cache_user = user if cache_user is None else cache_user
        host = self.read_host() if host is None else host
        layers = [layer for layer in cache.layers if layer < self.num_layers] if layers is None else list(layers)
        t0 = time.perf_counter()
        for layer in layers:
            index_key = host.index_key(user, layer, n) if host.is_full(layer) else None
            cache.write_prefill_host(layer, cache_user, latent=host.latent(user, layer, n), index_key=index_key)
        ttnn.synchronize_device(self.mesh_device)
        took = {"read_host": host.seconds, "write": time.perf_counter() - t0, "layers": len(layers), "tokens": n}
        self.seconds.setdefault("fill_from_host", []).append(took)
        return took

    # ------------------------------------------------------------------------------------------
    # (b) on the device
    # ------------------------------------------------------------------------------------------
    def _slot_rows(self, storage, slot: int, rows: int):
        """One slot's first ``rows`` rows as its own tensor, per chip [1, 1, rows, width]. The source
        is ND-sharded over the DRAM banks; the copy must be plain DRAM (the stack's read_slot_kv)."""
        import ttnn

        width = storage.shape[-1]
        return ttnn.slice(storage, [slot, 0, 0, 0], [slot + 1, 1, rows, width], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def device_latents(self) -> bool:
        """Whether ``fill_on_device`` can move the latents: only out of a BF16_RM prefill cache."""
        return self.kv_format == "bf16_rm"

    def fill_on_device(self, cache, user: int, layers=None, cache_user=None) -> dict:
        """Path (b): copy ``user``'s stripes into ``cache`` without leaving the device. Index keys
        always; latents when the prefill cache is BF16_RM (see the module docstring for the op the
        packed scaled-FP8 cache lacks). Whole tile rows are copied, up to the round that holds
        position N - 1, so rows past N arrive as the prefill left them (padding rows, an earlier
        prompt): the decode step never selects them.

        Returns {"index_keys", "latents": seconds, "latent_layers_left": the layers whose latents
        still have to come through fill_from_host}."""
        import ttnn

        n = self.lengths[user]
        assert n > 0, f"user {user} has not been prefilled"
        cache_user = user if cache_user is None else cache_user
        layers = [layer for layer in cache.layers if layer < self.num_layers] if layers is None else list(layers)
        rows = cache.layout.slots(n)
        assert rows <= self.rows_per_chip and rows <= cache.cfg.slots, (rows, self.rows_per_chip, cache.cfg.slots)
        took = {"index_keys": 0.0, "latents": 0.0, "latent_layers_left": [], "rows_per_chip": rows, "tokens": n}

        t0 = time.perf_counter()
        for layer in layers:
            if self.indexer_types[layer] != "full":
                continue
            x = self._slot_rows(self.kv_caches.index, index_slot(user, layer, self.indexer_types), rows)
            if x.dtype != cache.dtype:
                x = ttnn.typecast(x, cache.dtype)
            ttnn.kv_cache.fill_cache_for_user_(cache.index_key[layer], x, cache_user, update_idx=0)
            x.deallocate(True)
        ttnn.synchronize_device(self.mesh_device)
        took["index_keys"] = time.perf_counter() - t0

        if not self.device_latents():
            took["latent_layers_left"] = layers
        else:
            t0 = time.perf_counter()
            for layer in layers:
                x = self._slot_rows(self.kv_caches.kvpe.storage, kvpe_slot(user, layer, self.num_layers), rows)
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
                if x.dtype != cache.dtype:
                    x = ttnn.typecast(x, cache.dtype)
                ttnn.kv_cache.fill_cache_for_user_(cache.latent[layer], x, cache_user, update_idx=0)
                x.deallocate(True)
            ttnn.synchronize_device(self.mesh_device)
            took["latents"] = time.perf_counter() - t0
        self.seconds.setdefault("fill_on_device", []).append(took)
        return took

    def hand_over(self, cache, user: int, host: HostCaches | None = None, on_device: bool = True, cache_user=None):
        """Everything decode needs of ``user`` in ``cache``, by the cheapest path that exists for
        each kind: on the device where ``fill_on_device`` can, through the host for the rest.
        Returns (context length, the wall times)."""
        took = {}
        left = [layer for layer in cache.layers if layer < self.num_layers]
        if on_device:
            took["device"] = self.fill_on_device(cache, user, cache_user=cache_user)
            left = took["device"]["latent_layers_left"]
            if left:
                host = self.read_host() if host is None else host
                n = self.lengths[user]
                target = user if cache_user is None else cache_user
                t0 = time.perf_counter()
                for layer in left:
                    cache.write_prefill_host(layer, target, latent=host.latent(user, layer, n))
                took["host_latents"] = {"read_host": host.seconds, "write": time.perf_counter() - t0}
        else:
            took["host"] = self.fill_from_host(cache, user, host, cache_user=cache_user)
        return self.lengths[user], took

    # ------------------------------------------------------------------------------------------
    # freeing
    # ------------------------------------------------------------------------------------------
    def release_model(self):
        """Free the prefill weights and the stack's persistent buffers; the caches stay allocated
        and readable. Must run before the mesh closes (the stack's sub-device managers)."""
        import ttnn
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import clear_tt_ccl_cache

        if self.runtime is None:
            return
        t0 = time.perf_counter()
        errors = []
        for step in (self.runtime.release_trace, clear_tt_ccl_cache):
            try:
                step()
            except Exception as error:  # keep going: a half-released model must not stop the mesh from closing
                errors.append(f"{step.__name__}: {error}")
        self.runtime.model = None
        self.runtime = None
        gc.collect()
        ttnn.synchronize_device(self.mesh_device)
        self.seconds["release_model"] = time.perf_counter() - t0
        assert not errors, f"releasing the prefill model: {errors}"

    def release_caches(self):
        """Free the prefill storage, once decode has its rows."""
        import ttnn

        if self.kv_caches is None:
            return
        ttnn.deallocate(self.kv_caches.kvpe.storage)
        if self.kv_caches.index is not None:
            ttnn.deallocate(self.kv_caches.index)
        self.kv_caches = None

    def close(self):
        """Release everything this bridge holds on the device. Safe to call twice; call it in a
        ``finally`` before the mesh closes."""
        try:
            self.release_model()
        finally:
            self.release_caches()


__all__ = [
    "CHUNK",
    "INDEX_KEY",
    "LATENT",
    "MESH",
    "HostCaches",
    "PrefillBridge",
    "device_params",
    "dram",
    "format_report",
    "full_layers",
    "index_slot",
    "kvpe_slot",
    "l1_small",
    "open_mesh",
    "row_pcc",
    "row_report",
]
