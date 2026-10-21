# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel TTNN decoder layer for ``google/gemma-4-12B-it`` on a QB2 (four Blackhole chips, ``[1, 4]`` mesh).

The single-chip baseline is ``optimized_decoder.OptimizedDecoder``: same op graph, precision policy (including the
per-layer BFP4 exceptions), program-config rules and paged-cache contract. This class shards it 4 ways over the
mesh's one axis (``doc/multichip_decoder/mesh_plan.md``, README):

Weights (one DRAM width-sharded copy per chip, as on one chip)
* QKV column-parallel by head: chip ``d`` owns Q heads ``4d..4d+3`` and, on sliding layers, KV heads ``2d, 2d+1``.
  Full layers have one KV head (``attention_k_eq_v``); its K (and V copy) columns are on every chip.
* o_proj and down row-parallel; gate and up column-parallel over the intermediate size (3840 per chip).

KV cache (owned by the layer, one per chip)
* Sliding: ``[num_blocks, 2, block, 256]`` (local KV heads). Full: ``[num_blocks, 1, block, 512]``, the same values
  on every chip. Page tables and positions are replicated: every chip uses the same physical block ids.

Decode (``[1, 1, B, 3840]``, B = 1..32): residual **replicated** in every chip's L1 (width-sharded). o_proj and down
partial sums are summed by a one-shot ring ``all_reduce_async`` (persistent scratch, two per layer); norms and
residual adds run full-width on every chip. Input and output share ``decode_input_memory_config()``. Full layers at
B >= 8 (a multiple of 4) attend by user instead of by head: chip ``d`` gathers all 16 Q heads of users
``[dB/4, (d+1)B/4)`` and reads only their pages of the replicated cache (``_decode_attention_user_split``;
``doc/optimized_multichip_decoder/README.md``).

Prefill (one request, logical ``[1, 1, S, 3840]``): residual **sharded on hidden**, ``[1, 1, S, 960]`` per chip in
DRAM (``prefill_input_mesh_mapper()``, i.e. ``ttnn.ShardTensorToMesh(dim=3)``, at the stack entry). Residual norms
are distributed RMSNorms on the local 960 columns (``dit_fused_distributed_rmsnorm``, ring stats gather). The
normalized slices are all-gathered for QKV and for gate/up. On chunks of the minimal_matmul path both gathers are fused
into their matmuls (``all_gather_minimal_matmul_async``; gate/up as one packed matmul on layers with the BFP8 MLP
branch, whose gathered input is BFP8); shorter chunks use ``all_gather_async``. ``reduce_scatter_minimal_async``
returns o_proj and down sums to 960
columns. Input and output share this layout. Prefill and decode are separate passes of a
stack, so their residual layouts differ; within each pass every layer kind has the same input and output layout.
"""

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.functional_decoder import HF_PREFIX, MAX_DECODE_BATCH, TILE, rope_tables
from models.autoports.google_gemma_4_12b_it.tt.fused_decoder import DEFAULT_PREFILL_CHUNK, interleave_perm
from models.autoports.google_gemma_4_12b_it.tt.optimized_decoder import (
    DEFAULT_POLICY,
    FIDELITY,
    OptimizedDecoder,
    layer_policy,
)

MESH_SHAPE = (1, 4)
TP = 4
CLUSTER_AXIS = 1
NUM_LINKS = 2

# Multichip geometry over the single-chip policy. Per-chip matmuls are 4x smaller; one residual grid (15 cores x 8
# tiles) and one MLP grid (60 cores x 2 tiles) serve every layer, so a layer stack never reshards between layers.
# Measured on BFP8 (0, 5) and BFP4 (13, 25, 35) layers at B=1/32 (README "Decode"): 2-tile MLP shards -2 us (BFP8) /
# -6..-13 us (BFP4) vs 5; gate/up in0_block_w 8 -8 us on BFP4 layers vs the single-chip 4 (24 / 40: -5.5 / -3.3); a
# 40-core residual -1 us on BFP4 but +8..+12 us on BFP8 layers, 24 / 30 / 60 cores slower on both.
MULTICHIP_POLICY = {
    "residual_cores": 15,
    "mlp_shard_tiles": 2,
    # decode in0_block_w: 8 everywhere (the auto value of the BFP8 layers); on BFP4 layers gate/up 8 is -8 us vs the
    # single-chip 4, QKV / o_proj 8 is -0.6..-3.3 us vs the auto 40 / 32 (README "Decode")
    # down 8 (auto: 12): -1.0 us on BFP8 layers 0 / 5, -3.1 us on BFP4-down layers 13 / 17, -4.1 us on layer 25
    # (optimized-multichip README "Decode matmul geometry")
    "in0": {"qkv": 8, "o": 8, "gate": 8, "up": 8, "down": 8},
    # 2D-multicast / minimal_matmul prefill crossover for every layer (the single chip used 1024 on BFP4-MLP layers):
    # within +-0.06 ms of 1024 on BFP4-MLP layers; after the fused gate|up change 1024 is +0.02..+0.06 ms on BFP8
    # layers 0 / 5 and -0.02..-0.05 ms on layer 13 at 1152-1280 rows (README "Prefill"; stage-5 candidate)
    "prefill_linear_max_rows": 1280,
    # decode all-reduce payload: "bf16" or "bfp8" (o_proj / down partial sums). BFP8 halves the persistent scratch
    # (2 x 34 KB instead of 2 x 64 KB per residual core) and is 5.4 us faster per layer (README "Decode").
    "ar_dtype": {"o": "bfp8", "down": "bfp8"},
    # sliding prefill SDPA with 4 local heads: 128-row Q chunks over 4096-row query blocks keep the 110 cores busy
    # (single chip: q256 over 2048-row blocks, 16 heads); -0.8 ms at 4096 tokens (README "Prefill")
    "sdpa_prefill_q_chunk": {"sliding": 128, "full": 64},
    "sliding_attn_block": 4096,
    # per-chip prefill minimal_matmul blocking: gate/up [M, 3840] x [3840, 3840] run 12 x 4 x 12 (single chip:
    # 8 x 4 x 16 for 15360 columns); -60 us each at 4096 rows, within the L1 headroom (README "Prefill")
    "prefill_blocks": {"gate": (12, 4, 12, 2, 4), "up": (12, 4, 12, 2, 4)},
    # prefill chunks on the minimal_matmul path (more than prefill_linear_max_rows rows): the input norm's all-gather
    # fused into the QKV matmul (all_gather_minimal_matmul_async on an 11 x 9 matmul grid; the ring muxes take the
    # last grid row). Blocking (M, K, N, subblock_h, subblock_w); K must divide the 30 per-chip K tiles.
    "prefill_ag_qkv_fused": True,
    # the same for gate/up on layers whose long-chunk MLP branch is BFP8: one fused all-gather + matmul over a
    # prefill-only packed [gate | up] weight copy (per chip [3840, 7680], +31 MB BFP8), split into the two BFP8 outputs
    # by the op; -0.05..-0.46 ms (2-6%) per layer at 1344-8192 tokens (README "Prefill"). Decode keeps the separate
    # gate and up weights.
    "prefill_ag_gate_up_fused": True,
    "prefill_agmm_blocks": {"sliding": (8, 5, 16, 2, 4), "full": (8, 5, 16, 2, 4), "gate_up": (8, 10, 16, 2, 4)},
    # prefill o_proj output and its reduce-scatter payload: None (BF16) or "bfp8" (cast back to BF16 for the
    # distributed norm). "bfp8" is a further -0.04..-0.16 ms at 2048-8192 tokens but costs up to 0.0011 worst-row PCC
    # vs HF on layer 5 (README "Prefill activation / CCL dtype"); kept BF16
    "prefill_o_dtype": None,
    # long chunks (fused all-gather + matmul path): dtype of the input norm output (gathered QKV activation, "qkv") and
    # of the pre-FFN norm output (gathered gate/up activation, "gate_up"); None (BF16) or "bfp8" (half the
    # all-gather bytes)
    "prefill_norm_dtype": {"qkv": None, "gate_up": "bfp8"},
    # full-attention decode at batch >= this (and a multiple of 4): each chip attends B/4 users with all 16 Q heads
    # instead of all users with its 4 heads (the one KV head is replicated, so this cuts each chip's cache reads 4x);
    # None = never. Layer 5 at position 4096: B=32 708 -> 530 us, B=16 509 -> 446, B=8 415 -> 410; at 16384: B=32
    # 1776 -> 804, B=8 695 -> 479; B=4 loses 32 us at 4096 (README "User-split full-layer attention")
    "decode_user_split_min_batch": 8,
    # SDPA cores per (local user, KV head) in the user-split path, {max local users: cores} (smallest key >= users
    # wins); the unsplit path keeps ``sdpa_decode_max_cores``
    "sdpa_decode_split_max_cores": {2: 16, 4: 8, 8: 4},
    # user-split gathers: "none" (fresh output each call, barrier semaphore), "persistent" (persistent output buffers,
    # no barrier) or "barrier" (persistent buffers and the barrier argument; B=32 layer 5: -2.7 / -3.0 us vs "none").
    # all_gather_async ignores the barrier semaphore when it gets a persistent output
    # (all_gather_async_default_program_factory.cpp), so "barrier" runs without one and cross-invocation skew is not
    # bounded by the op. Default "none": the barrier really runs, +5 us per split full layer (reduced model, B=8 / 32,
    # doc/optimized_full_model/logs/split_gather_ab.json; logits identical)
    "decode_split_ag_buffers": "none",
    # where the user-split gathers write: "dram" (interleaved, persistent buffers per decode_split_ag_buffers) or "l1"
    # (interleaved, fresh per call; persistent L1 buffers for every batch size would break the 128 KB headroom)
    "decode_split_ag_mem": "dram",
    # decode_user_split_min_position: {max batch: position} from which the split wins (use_user_split, smallest key
    # >= batch). Layer 5: B=8 split +23 us at 2048, -4 us at 4096; B=16 +36 at 1024, -4 at 2048; B=32 +13 at 1024,
    # -51 at 2048; B=4 +32 at 4096, -63 at 16384 (README "User split and context length")
    "decode_user_split_min_position": {4: 8192, 8: 4096, 16: 2048, 32: 1280},
    # chunks on the 2D-matmul path (<= prefill_linear_max_rows): all-gathered QKV / gate-up input in "dram" or "l1"
    # (interleaved)
    "prefill_short_gather_mem": "dram",
}
_AR_DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
_ACT_DTYPES = {None: None, "bf16": None, "bfp8": ttnn.bfloat8_b}


def merge_multichip_policy(*overrides):
    """``DEFAULT_POLICY`` + ``MULTICHIP_POLICY`` updated by each override in turn (same rules as ``merge_policy``)."""

    def batch_keyed(d):
        return all(isinstance(k, int) or str(k).isdigit() for k in d)

    def merge(base, over, path=""):
        out = {k: (merge(v, {}) if isinstance(v, dict) else v) for k, v in base.items()}
        for k, v in (over or {}).items():
            assert k in out or batch_keyed(base), f"unknown policy field {path}{k}"
            out[k] = merge(out[k], v, f"{path}{k}.") if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    out = merge(DEFAULT_POLICY, {})
    out.update({k: merge(v, {}) if isinstance(v, dict) else v for k, v in MULTICHIP_POLICY.items() if k not in out})
    out = merge(out, {k: v for k, v in MULTICHIP_POLICY.items() if k in DEFAULT_POLICY})
    for policy in overrides:
        out = merge(out, policy)
    return out


def multichip_layer_policy(layer_idx, layer_type="sliding_attention"):
    """Single-chip per-layer policy plus the multichip geometry (residual grid shared by every layer)."""
    out = layer_policy(layer_idx, layer_type)
    # The single-chip BFP4-MLP geometry (60-core residual, 8-tile MLP shards, gate/up in0_block_w 4) gives way to the
    # shared multichip grids and blocks; its 3 DRAM readers per bank stay (-15 us per layer vs 1 on layer 13).
    out.update({k: (dict(v) if isinstance(v, dict) else v) for k, v in MULTICHIP_POLICY.items()})
    return out


def fabric_router_config():
    r = ttnn.FabricRouterConfig()
    r.max_packet_payload_size_bytes = 8192
    return r


def open_qb2_mesh(**kwargs):
    """Set the ring fabric and open the ``[1, 4]`` QB2 mesh this decoder targets."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=fabric_router_config())
    return ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE), **kwargs)


def close_qb2_mesh(mesh):
    _MESH_CCL.pop(id(mesh), None)
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


_MESH_CCL = {}  # id(mesh) -> (mesh, MeshCCL): one set of semaphores and persistent CCL buffers per open mesh


def mesh_ccl(mesh_device):
    """The mesh's ``MeshCCL`` (created on first use). Every layer on a mesh shares it, so the persistent decode
    all-reduce scratch exists once per mesh, also for layers built without a ``shared`` dict."""
    entry = _MESH_CCL.get(id(mesh_device))
    if entry is None or entry[0] is not mesh_device:
        entry = (mesh_device, MeshCCL(mesh_device))
        _MESH_CCL[id(mesh_device)] = entry
    return entry[1]


class MeshCCL:
    """Ring CCL resources shared by every layer of a stack on one mesh.

    * Prefill all-gather / reduce-scatter / fused-norm semaphores rotate through ping-pong sets (as GPT-OSS
      ``CCLManager``), so back-to-back collectives never share a semaphore.
    * Prefill distributed norms use persistent FP32 stats buffers, two per chunk-size class, created with the first
      layer (``prepare_norm_stats``).
    * Decode all-reduces use two persistent slots ("o", "down"), each a scratch buffer ``[32, 4 * 3840]`` in the
      residual's L1 grid and one semaphore. Consecutive uses of one slot (layer L, layer L+1) are separated by the
      other slot's all-reduce, which no chip can finish before every chip has finished reading the first slot.
    """

    def __init__(self, mesh_device, num_links=NUM_LINKS, topology=ttnn.Topology.Ring):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        grid = mesh_device.compute_with_storage_grid_size()
        self.cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

        def sems(n):
            return [ttnn.create_global_semaphore(mesh_device, self.cores, 0) for _ in range(n)]

        self._ag = [sems(2), sems(2)]
        self._rs = [sems(3), sems(3)]
        self._norm = [sems(2), sems(2)]
        self._barrier = sems(2)
        self._idx = {"ag": 0, "rs": 0, "norm": 0, "barrier": 0}
        self._ar_slots = {}
        self.decode_state = {}  # persistent decode-only tensors shared by the layers (user-split attention)
        self._norm_stats = {}
        self._norm_stats_idx = {}

    def _next(self, name, pool):
        i = self._idx[name]
        self._idx[name] = (i + 1) % len(pool)
        return pool[i]

    def all_gather(self, x, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, persistent_output_buffer=None, barrier=True):
        """``all_gather_async`` over the ring; ``persistent_output_buffer`` (with ``barrier=False``) skips the barrier
        that protects a freshly allocated output buffer."""
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=persistent_output_buffer,
            dim=dim,
            cluster_axis=CLUSTER_AXIS,
            multi_device_global_semaphore=self._next("ag", self._ag),
            barrier_semaphore=self._next("barrier", self._barrier) if barrier else None,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=memory_config,
        )

    def reduce_scatter(self, x, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            dim=dim,
            cluster_axis=CLUSTER_AXIS,
            multi_device_global_semaphore=self._next("rs", self._rs),
            barrier_semaphore=self._next("barrier", self._barrier),
            num_links=self.num_links,
            topology=self.topology,
            memory_config=memory_config,
        )

    # The fused norm's FP32 stats buffer is [1, 1, 8 * ceil(rows / NORM_STATS_ROWS), 2048]: one size per 2048 rows.
    NORM_STATS_ROWS = 2048

    def prepare_norm_stats(self, max_rows, width, weight):
        """Create the persistent stats buffers of the distributed prefill norm for every chunk of up to ``max_rows``
        rows (two per size: <= 4 sizes at the 8192-row chunk, <= 256 KB each). Called at layer construction, so prefill
        never creates persistent device state (which would be live across a later decode trace replay)."""
        for k in range(1, -(-max_rows // self.NORM_STATS_ROWS) + 1):
            if k in self._norm_stats:
                continue
            probe = ttnn.empty(
                [1, 1, k * self.NORM_STATS_ROWS, width],
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            self._norm_stats[k] = [
                ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
                    probe,
                    cluster_axis=CLUSTER_AXIS,
                    mesh_device=self.mesh_device,
                    num_links=self.num_links,
                    weight=weight,
                )
                for _ in range(2)
            ]
            self._norm_stats_idx[k] = 0
            ttnn.deallocate(probe)

    def _norm_stats_buffer(self, x):
        """The next of the two stats buffers for this chunk's size (used in turn).

        Reusing a buffer two norms later is safe: the norm in between is an all-chip exchange that no chip completes
        before every chip has finished the earlier norm.
        """
        k = -(-x.padded_shape[-2] // self.NORM_STATS_ROWS)
        assert (
            k in self._norm_stats
        ), f"no prepared norm stats buffer for {x.padded_shape[-2]} rows (prepare_norm_stats)"
        i = self._norm_stats_idx[k]
        self._norm_stats_idx[k] = 1 - i
        return self._norm_stats[k][i]

    def distributed_rms_norm(self, x, weight, eps, compute_kernel_config, dtype=None):
        """RMSNorm over the hidden dim of a ``[.., S, 960]`` per-chip slice (stats gathered over the ring)."""
        return ttnn.experimental.dit_fused_distributed_rmsnorm(
            x,
            cluster_axis=CLUSTER_AXIS,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=self._next("norm", self._norm),
            topology=self.topology,
            epsilon=eps,
            weight=weight,
            dtype=dtype,
            persistent_output_buffer=self._norm_stats_buffer(x),
            num_preferred_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )

    def ar_slot(self, name, width, memory_config, dtype):
        """Persistent one-shot all-reduce scratch + semaphore for ``name``, keyed by layout and dtype."""
        key = (name, width, memory_config.shard_spec.grid.num_cores(), dtype)
        if key not in self._ar_slots:
            spec = memory_config.shard_spec
            scratch_mem = ttnn.create_sharded_memory_config(
                (TILE, spec.shape[1] * TP),
                core_grid=spec.grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                use_height_and_width_as_shard_shape=True,
            )
            scratch = ttnn.from_torch(
                torch.zeros(1, 1, TILE, width * TP),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=scratch_mem,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            self._ar_slots[key] = (scratch, ttnn.create_global_semaphore(self.mesh_device, self.cores, 0))
        return self._ar_slots[key]

    def all_reduce(self, x, name, memory_config):
        """One-shot ring all-reduce of a ``[1, 1, B, H]`` L1 width-sharded tensor; the output is BF16."""
        scratch, sem = self.ar_slot(name, x.shape[-1], memory_config, x.dtype)
        return ttnn.experimental.all_reduce_async(
            x,
            scratch,
            cluster_axis=CLUSTER_AXIS,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=sem,
            memory_config=memory_config,
            dtype=ttnn.bfloat16,
            topology=self.topology,
            num_links=self.num_links,
        )


class MultichipDecoder(OptimizedDecoder):
    """One Gemma4 text decoder layer (sliding or full attention), tensor-parallel over a ``[1, 4]`` QB2 mesh."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        num_blocks,
        block_size=64,
        prefill_chunk_size=DEFAULT_PREFILL_CHUNK,
        max_context=None,
        shared=None,
        policy=None,
    ):
        """Build the layer from an HF state dict (arguments as ``OptimizedDecoder.from_state_dict``).

        mesh_device: the ``[1, 4]`` mesh opened with ``FABRIC_1D_RING`` (``open_qb2_mesh``).
        num_blocks / block_size: paged KV cache geometry; every chip holds ``num_blocks`` blocks of its local heads
            (0: no cache is allocated; the owner sets ``kv_cache`` before the first forward).
        shared: dict reused across the layers of a stack (RoPE tables, rotation matrix); the CCL manager with the
            decode all-reduce scratch is per mesh (``mesh_ccl``).
        policy: overrides of ``DEFAULT_POLICY`` + ``MULTICHIP_POLICY``, applied after ``multichip_layer_policy``.
        """
        c = hf_config
        assert tuple(mesh_device.shape) == MESH_SHAPE, f"multichip decoder runs on a {MESH_SHAPE} mesh"
        assert not c.enable_moe_block and not c.hidden_size_per_layer_input and not c.num_kv_shared_layers
        assert not c.attention_bias and not c.use_double_wide_mlp
        assert c.hidden_activation == "gelu_pytorch_tanh"
        assert block_size % TILE == 0 and prefill_chunk_size % block_size == 0
        self = cls()
        self.policy = p = merge_multichip_policy(multichip_layer_policy(layer_idx, c.layer_types[layer_idx]), policy)
        self.device = mesh_device
        self.tp = TP
        self.layer_idx = layer_idx
        self.layer_type = c.layer_types[layer_idx]
        assert self.layer_type in ("sliding_attention", "full_attention")
        self.is_sliding = self.layer_type == "sliding_attention"
        self.hidden_size = c.hidden_size
        self.local_hidden = c.hidden_size // TP
        assert self.local_hidden % TILE == 0
        self.intermediate_size = c.intermediate_size // TP  # per chip (decode configs use the local size)
        self.num_heads_total = c.num_attention_heads
        self.num_heads = c.num_attention_heads // TP  # local Q heads
        assert self.num_heads * TP == c.num_attention_heads
        self.k_eq_v = bool(c.attention_k_eq_v) and not self.is_sliding
        self.head_dim = c.head_dim if self.is_sliding else c.global_head_dim
        kv_total = c.num_global_key_value_heads if self.k_eq_v else c.num_key_value_heads
        self.num_kv_heads_total = kv_total
        # Sliding: KV heads split by chip. Full: one KV head, replicated.
        self.kv_replicated = kv_total < TP
        assert self.kv_replicated or kv_total % TP == 0
        self.num_kv_heads = kv_total if self.kv_replicated else kv_total // TP
        self.sliding_window = c.sliding_window if self.is_sliding else None
        self.eps = c.rms_norm_eps
        self.max_context = c.max_position_embeddings if max_context is None else max_context
        assert self.max_context % block_size == 0
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.prefill_chunk_size = prefill_chunk_size
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.worker_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.grid.x - 1, self.grid.y - 1))}
        )
        self.dram_banks = mesh_device.dram_grid_size().x
        self.dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.dram_banks - 1, 0))}
        )
        shared = {} if shared is None else shared
        if "ccl" not in shared:
            shared["ccl"] = mesh_ccl(mesh_device)
        self.ccl = shared["ccl"]
        assert self.ccl.mesh_device is mesh_device
        self.ar_dtype = {r: _AR_DTYPES[d] for r, d in p["ar_dtype"].items()}
        self.prefill_o_dtype = _ACT_DTYPES[p["prefill_o_dtype"]]
        self.prefill_short_gather_mem = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}[
            p["prefill_short_gather_mem"]
        ]
        self.prefill_norm_dtype = {r: _ACT_DTYPES[d] for r, d in p["prefill_norm_dtype"].items()}

        prefix = HF_PREFIX.format(layer_idx)
        sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)} or state_dict
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        def as_tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mapper=None):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper or replicate,
            )

        def norm_weight(name):
            # rms_norm consumes ROW_MAJOR weights folded into [1, 1, dim / 32, 32] (replicated, full width).
            return as_tt(sd[name + ".weight"].reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)

        def norm_weight_local(name):
            # The distributed prefill norm takes each chip's 960-column slice as a TILE row.
            return as_tt(sd[name + ".weight"].reshape(1, 1, 1, -1), mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3))

        hd, nh, nkv = self.head_dim, self.num_heads, self.num_kv_heads
        q = sd["self_attn.q_proj.weight"]
        k = sd["self_attn.k_proj.weight"]
        v = k if self.k_eq_v else sd["self_attn.v_proj.weight"]
        assert q.shape[0] == self.num_heads_total * hd and k.shape[0] == kv_total * hd
        self.meta_rope = self.is_sliding
        perm = interleave_perm(hd) if self.meta_rope else None
        if self.meta_rope:
            q = q.reshape(self.num_heads_total, hd, -1)[:, perm, :].reshape(q.shape)
            k = k.reshape(kv_total, hd, -1)[:, perm, :].reshape(k.shape)
        qs = q.reshape(TP, nh * hd, -1)
        if self.kv_replicated:
            ks, vs = [k] * TP, [v] * TP
        else:
            ks, vs = k.reshape(TP, nkv * hd, -1), v.reshape(TP, nkv * hd, -1)
        # Chip-major columns: chip d's block is [its Q heads | its K heads | its V heads].
        wqkv = torch.cat([torch.cat([qs[d], ks[d], vs[d]], dim=0) for d in range(TP)], dim=0).T
        if self.meta_rope:
            # Prefill splits the local block into four equal chunks: two Q halves, K, V.
            assert nh * hd == 2 * nkv * hd
        self.wqkv = self._dram_weight(wqkv, "qkv", shard_dim=1)
        self.wo = self._dram_weight(sd["self_attn.o_proj.weight"].T, "o", shard_dim=0)
        self.w_gate = self._dram_weight(sd["mlp.gate_proj.weight"].T, "gate", shard_dim=1)
        self.w_up = self._dram_weight(sd["mlp.up_proj.weight"].T, "up", shard_dim=1)
        # The fused gate/up path runs only with BFP8 outputs: the op's program cache ignores the output dtype, so BF16
        # and BFP8 calls of one shape must not coexist (layers 10, 11 keep BF16 intermediates and the separate path).
        self.fused_gate_up = p["prefill_ag_gate_up_fused"] and p["prefill_mlp_act_dtype"]["gate_up"] == "bfp8"
        if self.fused_gate_up:
            g, u = sd["mlp.gate_proj.weight"].T, sd["mlp.up_proj.weight"].T
            ff = g.shape[1] // TP
            packed = torch.cat(
                [torch.cat([g[:, d * ff : (d + 1) * ff], u[:, d * ff : (d + 1) * ff]], 1) for d in range(TP)], 1
            )
            self.w_gate_up_packed = self._dram_weight(packed, "gate", shard_dim=1)
        self.w_down = self._dram_weight(sd["mlp.down_proj.weight"].T, "down", shard_dim=0)
        self.input_norm = norm_weight("input_layernorm")
        self.post_attn_norm = norm_weight("post_attention_layernorm")
        self.pre_ffn_norm = norm_weight("pre_feedforward_layernorm")
        self.post_ffn_norm = norm_weight("post_feedforward_layernorm")
        self.input_norm_local = norm_weight_local("input_layernorm")
        self.post_attn_norm_local = norm_weight_local("post_attention_layernorm")
        self.pre_ffn_norm_local = norm_weight_local("pre_feedforward_layernorm")
        self.post_ffn_norm_local = norm_weight_local("post_feedforward_layernorm")
        if self.meta_rope:
            q_w = sd["self_attn.q_norm.weight"][perm]
            k_w = sd["self_attn.k_norm.weight"][perm]
            self.q_norm = as_tt(q_w.reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)
            self.k_norm = as_tt(k_w.reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)
            self.q_norm_half = as_tt(q_w.repeat(nh // 2).reshape(1, 1, 1, -1))
            self.k_norm_heads = as_tt(k_w.repeat(nkv).reshape(1, 1, 1, -1))
        else:
            self.q_norm = norm_weight("self_attn.q_norm")
        if self.k_eq_v:
            self.k_norm_row = as_tt(sd["self_attn.k_norm.weight"].reshape(1, 1, 1, -1))
        elif not self.meta_rope:
            self.k_norm = norm_weight("self_attn.k_norm")
        self.layer_scalar = float(sd["layer_scalar"].float().item())
        self.gelu = {
            "tanh": ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH),
            "fast": ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 1.0),
        }[p["gelu"]]

        key = ("rope", self.layer_type, self.max_context, "interleaved" if self.meta_rope else "hf")
        if key not in shared:
            cos, sin = rope_tables(c, self.layer_type, self.max_context)
            if self.meta_rope:
                cos, sin = cos[:, perm], sin[:, perm]
            shared[key] = {
                "cos": as_tt(cos.reshape(1, 1, self.max_context, hd)),
                "sin": as_tt(sin.reshape(1, 1, self.max_context, hd)),
                "cos_rm": as_tt(cos, layout=ttnn.ROW_MAJOR_LAYOUT),
                "sin_rm": as_tt(sin, layout=ttnn.ROW_MAJOR_LAYOUT),
            }
        self.rope = shared[key]

        self.kv_dtype = p["kv_dtype"]
        # per-chip cache geometry [num_blocks, local KV heads, block, head_dim]; num_blocks 0 builds the layer without
        # a cache, for an owner that allocates it later (vLLM: ``Gemma4Model.attach_kv_cache``)
        self.kv_block_shape = (nkv, block_size, hd)
        self.kv_cache = None
        if num_blocks:
            self.kv_cache = [
                ttnn.zeros(
                    [num_blocks, *self.kv_block_shape],
                    dtype=self.kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            ]

        def ckc(name):
            fid, fp32, l1acc = FIDELITY[name]
            return ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=fid,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32,
                packer_l1_acc=l1acc,
            )

        self.compute_hifi4_fp32 = ckc("hifi4fp32")
        self.decode_ck = {r: ckc(f) for r, f in p["decode_fid"].items()}
        self.prefill_ck = {r: ckc(f) for r, f in p["prefill_fid"].items()}
        kind = "sliding" if self.is_sliding else "full"
        blocks = {r: p["prefill_blocks"].get(f"{r}@{kind}", b) for r, b in p["prefill_blocks"].items() if "@" not in r}
        self.prefill_mm_cfg = {
            r: (
                None
                if b is None
                else ttnn.MinimalMatmulConfig(
                    M_block_size=b[0],
                    K_block_size=b[1],
                    N_block_size=b[2],
                    subblock_h=b[3],
                    subblock_w=b[4],
                    compute_with_storage_grid_size=self.grid,
                )
            )
            for r, b in blocks.items()
        }
        self.compute_sdpa_prefill = ckc(p["sdpa_prefill_fid"][kind])
        self.compute_sdpa_prefill_short = ckc("hifi2")
        self.compute_sdpa_decode = ckc(p["sdpa_decode_fid"])
        self.sliding_attn_block = p["sliding_attn_block"]
        mid = {None: None, "bfp8": ttnn.bfloat8_b}
        self.prefill_gate_up_dtype = mid[p["prefill_mlp_act_dtype"]["gate_up"]]
        self.prefill_product_dtype = mid[p["prefill_mlp_act_dtype"]["product"]]
        if self.meta_rope:
            from models.autoports.google_gemma_4_12b_it.tt.fused_decoder import rot_transformation_mat

            if "rope_trans" not in shared:
                trans = rot_transformation_mat()
                users = ttnn.num_cores_to_corerangeset(MAX_DECODE_BATCH, self.grid, row_wise=True)
                shared["rope_trans"] = {
                    "prefill": as_tt(trans),
                    "decode": ttnn.from_torch(
                        trans.repeat(1, 1, MAX_DECODE_BATCH, 1),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        mesh_mapper=replicate,
                        memory_config=ttnn.create_sharded_memory_config(
                            (TILE, TILE),
                            core_grid=users,
                            strategy=ttnn.ShardStrategy.HEIGHT,
                            use_height_and_width_as_shard_shape=True,
                        ),
                    ),
                }
            self.rope_trans = shared["rope_trans"]
            self.compute_rope_decode = ckc("hifi4")
        self._sdpa_decode_cfgs = {}
        self.sdpa_prefill_q_cap = p["sdpa_prefill_q_chunk"][kind]
        self.sdpa_prefill_q_floor = p["sdpa_prefill_q_floor"][kind]
        self.sdpa_prefill_k_cap = p["sdpa_prefill_k_chunk"][kind] or block_size
        self.sdpa_prefill_exp_approx = p["sdpa_prefill_exp_approx"][kind]
        assert self.sdpa_prefill_k_cap % block_size == 0
        self._build_decode_configs()
        # Persistent decode all-reduce scratch, created with the first layer of a stack (before any activation)
        # so it sits at the top of L1 and never inside a decode pass or a trace capture.
        for role in ("o", "down"):
            self.ccl.ar_slot(role, self.hidden_size, self.residual_mem, self.ar_dtype[role])
        # Prefill norm stats buffers for every chunk size, also created before any activation or trace.
        self.ccl.prepare_norm_stats(prefill_chunk_size, self.local_hidden, self.input_norm_local)
        self._decode_layouts = {}
        # User-split attention tensors for every batch that takes that path, so decode allocates nothing persistent.
        # (every multiple of 4: the default path or a caller's user_split=True)
        if self.kv_replicated:
            for b in range(self.tp, MAX_DECODE_BATCH + 1, self.tp):
                self._user_split_state(b)
        return self

    # ------------------------------------------------------------------ weights and decode layouts
    def _dram_weight(self, w, role, shard_dim):
        """Full ``[K, N]`` torch -> per-chip ``[K, N/4]`` (shard_dim 1) or ``[K/4, N]`` (shard_dim 0) slices, each
        width-sharded over its chip's DRAM banks (one bank shard splits evenly over the role's readers)."""
        k, n = w.shape
        k_local, n_local = (k, n // TP) if shard_dim == 1 else (k // TP, n)
        assert k_local * (TP if shard_dim == 0 else 1) == k and n_local * (TP if shard_dim == 1 else 1) == n
        readers = self.policy["readers"][role]
        assert n_local % (TILE * self.dram_banks * readers) == 0, f"{role}: {readers} readers need padded weights"
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(self.dram_grid, (k_local, n_local // self.dram_banks), ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.from_torch(
            w.contiguous(),
            dtype=self.policy["dtype"][role],
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=mem,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=shard_dim),
        )

    def _build_decode_configs(self):
        """Per-chip decode matmul configs (local Q/KV heads, local intermediate) on the replicated residual grid."""
        p = self.policy
        h, ff = self.hidden_size, self.intermediate_size
        res = p["residual_cores"]
        assert (h // TILE) % res == 0
        res_tiles = h // TILE // res
        self.residual_mem = self._width_sharded(h, res)
        self.residual_norm_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=self.grid,
            subblock_w=max(d for d in (1, 2, 3, 4) if res_tiles % d == 0),
            block_h=1,
            block_w=res_tiles,
            inplace=False,
        )
        qkv_n = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        head_tiles = self.head_dim // TILE
        # o_proj input (head-concat output): one head per core
        self.attn_flat_mem = self._width_sharded(self.num_heads * self.head_dim, self.num_heads)
        self.mm_cfg = {
            "qkv": self._dram_mm_config("qkv", h, qkv_n, res_tiles, head_tiles),
            "gate": self._dram_mm_config("gate", h, ff, res_tiles, p["mlp_shard_tiles"]),
            "up": self._dram_mm_config("up", h, ff, res_tiles, p["mlp_shard_tiles"]),
            "down": self._dram_mm_config("down", ff, h, p["mlp_shard_tiles"], res_tiles),
            "o": self._dram_mm_config("o", self.num_heads * self.head_dim, h, head_tiles, res_tiles),
        }

    def _sdpa_decode_cfg(self, batch, split=False):
        """(K chunk, SDPA decode program config) for ``batch`` users; ``split``: the user-split path's ``batch``
        local users, whose cores per (user, KV head) come from ``sdpa_decode_split_max_cores``."""
        if not split:
            return super()._sdpa_decode_cfg(batch)
        key = ("split", batch)
        if key not in self._sdpa_decode_cfgs:
            p = self.policy
            k_chunk, _ = super()._sdpa_decode_cfg(batch)
            table = {int(b): v for b, v in p["sdpa_decode_split_max_cores"].items()}
            cores = table[min((b for b in table if b >= batch), default=max(table))]
            grid = p["sdpa_decode_grid"]
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid) if grid else self.grid,
                q_chunk_size=0,
                k_chunk_size=k_chunk,
                exp_approx_mode=p["sdpa_decode_exp_approx"],
                max_cores_per_head_batch=cores,
            )
            self._sdpa_decode_cfgs[key] = (k_chunk, cfg)
        return self._sdpa_decode_cfgs[key]

    def decode_input_memory_config(self, batch=1):
        """Layout of the decode residual stream on every chip (replicated; input and output of ``decode_forward``)."""
        return self.residual_mem

    def prefill_input_mesh_mapper(self):
        """Mesh mapper that turns a host ``[1, 1, S, 3840]`` hidden state into the prefill residual layout."""
        return ttnn.ShardTensorToMesh(self.device, dim=3)

    # ------------------------------------------------------------------ prefill
    def _dist_norm(self, x, weight, dtype=None):
        return self.ccl.distributed_rms_norm(x, weight, self.eps, self.compute_hifi4_fp32, dtype=dtype)

    def _mlp_and_residuals(self, x, attn_partial, matmul):
        """Prefill: reduce-scatter o_proj sums, post-attention norm + residual, gated MLP, post-FFN norm + residual.

        ``x`` and the result are this chip's ``[1, 1, rows, 960]`` residual slice.
        """
        attn_out = self.ccl.reduce_scatter(attn_partial)
        ttnn.deallocate(attn_partial)
        if attn_out.dtype != ttnn.bfloat16:
            attn_b = ttnn.typecast(attn_out, ttnn.bfloat16)
            ttnn.deallocate(attn_out)
            attn_out = attn_b
        h = self._dist_norm(attn_out, self.post_attn_norm_local)
        ttnn.deallocate(attn_out)
        x = ttnn.add(x, h)
        ttnn.deallocate(h)
        long_chunk = x.shape[-2] > self.policy["prefill_linear_max_rows"]
        h_local = self._dist_norm(
            x,
            self.pre_ffn_norm_local,
            dtype=self.prefill_norm_dtype["gate_up"] if long_chunk and self.fused_gate_up else None,
        )
        mid = self.prefill_gate_up_dtype if long_chunk else None
        if long_chunk and self.fused_gate_up:
            # The op builds its all-gather intermediate with the requested output dtype (all_gather_minimal_matmul_async
            # compute_output_specs), so a BFP8 output without a gather buffer would corrupt BF16 gathered rows; the
            # per-call gather buffer (_ag_matmul) keeps the gathered rows in the input's dtype (BFP8 by default,
            # prefill_norm_dtype["gate_up"]; multichip work log item 17).
            gate, up = self._ag_matmul(
                h_local, self.w_gate_up_packed, "gate", "gate_up", chunks=2, dtype=mid, gather_buffer=True
            )
            ttnn.deallocate(h_local)
            act = [self.gelu]
            h = None
        else:
            mem = ttnn.DRAM_MEMORY_CONFIG if long_chunk else self.prefill_short_gather_mem
            h = self.ccl.all_gather(h_local, memory_config=mem)
            ttnn.deallocate(h_local)
        if h is None:
            pass
        elif self.policy["prefill_gelu_in_gate"]:
            gate = matmul(
                h, self.w_gate, "gate", fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH), dtype=mid
            )
            act = []
        else:
            gate = matmul(h, self.w_gate, "gate", dtype=mid)
            act = [self.gelu]
        if h is not None:
            up = matmul(h, self.w_up, "up", dtype=mid)
            ttnn.deallocate(h)
        prod = ttnn.multiply(
            gate, up, input_tensor_a_activations=act, dtype=self.prefill_product_dtype if long_chunk else None
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        down_partial = matmul(prod, self.w_down, "down")
        ttnn.deallocate(prod)
        down = self.ccl.reduce_scatter(down_partial)
        ttnn.deallocate(down_partial)
        if down.dtype != ttnn.bfloat16:
            # Long chunks run the MLP branch in BFP8 (single-chip policy); the reduce-scatter keeps that payload
            # (30% fewer bytes) and the distributed norm takes BF16, so cast the 960-column slice.
            down_b = ttnn.typecast(down, ttnn.bfloat16)
            ttnn.deallocate(down)
            down = down_b
        h = self._dist_norm(down, self.post_ffn_norm_local)
        ttnn.deallocate(down)
        out = ttnn.add(x, h, activations=[(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.layer_scalar)])
        ttnn.deallocate(x)
        ttnn.deallocate(h)
        return out

    def _ag_qkv(self, h_local):
        """All-gather of the normalized 960-column slices fused into the QKV matmul. Sliding layers get the four
        column chunks [Q heads 0-1 | Q heads 2-3 | K | V] of ``minimal_matmul_split``; full layers one tensor."""
        kind = "sliding" if self.is_sliding else "full"
        return self._ag_matmul(h_local, self.wqkv, "qkv", kind, chunks=4 if self.meta_rope else 1)

    def _ag_matmul(self, h_local, weight, role, blocks, chunks, dtype=None, gather_buffer=False):
        """``all_gather_minimal_matmul_async``: gather the 960-column slices (dim 3) into the matmul with ``weight``;
        returns ``chunks`` equal column chunks of the output. ``gather_buffer``: pass a per-call BF16 buffer for the
        gathered rows (needed with a non-BF16 ``dtype``)."""
        b = self.policy["prefill_agmm_blocks"][blocks]
        gathered = None
        if h_local.dtype != ttnn.bfloat16 and dtype is None:
            dtype = ttnn.bfloat16  # a BFP8 gathered input still gives BF16 outputs unless asked otherwise
        if gather_buffer or h_local.dtype != (dtype or h_local.dtype):
            # the op builds its gather intermediate in the output dtype; keep the gathered rows in the input's
            shape = list(h_local.shape)
            shape[-1] *= TP
            gathered = ttnn.empty(shape, h_local.dtype, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG)
        grid = ttnn.CoreCoord(self.grid.x, self.grid.y - 1)
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=b[0],
            K_block_size=b[1],
            N_block_size=b[2],
            subblock_h=b[3],
            subblock_w=b[4],
            compute_with_storage_grid_size=grid,
        )
        outs = ttnn.experimental.all_gather_minimal_matmul_async(
            h_local,
            weight,
            config=cfg,
            compute_kernel_config=self.prefill_ck[role],
            dtype=dtype,
            # Without a persistent gather buffer the op needs its barrier semaphore: it keeps a fast chip from
            # writing gathered rows into a slow chip's freshly allocated buffer while that memory is still in use
            # (op contract; without it, prefill under the watcher was not deterministic, work log item 12).
            persistent_output_buffer=gathered,
            barrier_semaphore=self.ccl._next("barrier", self.ccl._barrier),
            multi_device_global_semaphore=self.ccl._next("ag", self.ccl._ag),
            num_links=self.ccl.num_links,
            topology=self.ccl.topology,
            cluster_axis=CLUSTER_AXIS,
            force_transpose=True,
            # the transposed grid sends in0 along its 11 columns: two link groups of 6 senders
            num_workers_per_link=-(-grid.x // self.ccl.num_links),
            chunks=chunks,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if gathered is not None:
            ttnn.deallocate(gathered)
        return outs

    def _heads_from_parts(self, parts, cos, sin):
        """Sliding: fused per-head norm + RoPE of the four QKV column chunks (as ``_prefill_heads_meta``)."""
        qa, qb, k_flat, v_flat = parts
        half = self.num_heads // 2
        q_parts = [self._norm_rope(t, half, self.q_norm_half, cos, sin) for t in (qa, qb)]
        q_r = ttnn.concat(q_parts, dim=1)
        k_r = self._norm_rope(k_flat, self.num_kv_heads, self.k_norm_heads, cos, sin)
        v_n = self._norm_rope(v_flat, self.num_kv_heads)
        for t in (qa, qb, k_flat, v_flat, *q_parts):
            ttnn.deallocate(t)
        return q_r, k_r, v_n

    def _heads_from_qkv(self, qkv, cos, sin):
        """Full: head split, per-head norms (k = v_norm * w_k), HF RoPE (as ``_prefill_heads_hf``)."""
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q_n = self._rms(q, self.q_norm)
        v_n = self._rms(v)
        k_n = self._k_from_v(v_n) if self.k_eq_v else self._rms(k, self.k_norm)
        for t in (q, k, v):
            ttnn.deallocate(t)
        q_r = ttnn.experimental.rotary_embedding_hf(q_n, cos, sin)
        k_r = ttnn.experimental.rotary_embedding_hf(k_n, cos, sin)
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)
        return q_r, k_r, v_n

    def _prefill_chunk(self, x, *, page_table, user_id, chunk_start, padded_len):
        fused_qkv = self.policy["prefill_ag_qkv_fused"] and padded_len > self.policy["prefill_linear_max_rows"]
        h_local = self._dist_norm(x, self.input_norm_local, dtype=self.prefill_norm_dtype["qkv"] if fused_qkv else None)
        end = chunk_start + padded_len
        cos = ttnn.slice(self.rope["cos"], [0, 0, chunk_start, 0], [1, 1, end, self.head_dim])
        sin = ttnn.slice(self.rope["sin"], [0, 0, chunk_start, 0], [1, 1, end, self.head_dim])
        if fused_qkv:
            parts = self._ag_qkv(h_local)
            ttnn.deallocate(h_local)
            if self.meta_rope:
                q_r, k_r, v_n = self._heads_from_parts(parts, cos, sin)
            else:
                q_r, k_r, v_n = self._heads_from_qkv(parts[0], cos, sin)
        else:
            short = padded_len <= self.policy["prefill_linear_max_rows"]
            h = self.ccl.all_gather(
                h_local, memory_config=self.prefill_short_gather_mem if short else ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.deallocate(h_local)
            heads = self._prefill_heads_meta if self.meta_rope else self._prefill_heads_hf
            q_r, k_r, v_n = heads(h, cos, sin)
            ttnn.deallocate(h)

        first_block = chunk_start // self.block_size
        num_chunk_blocks = padded_len // self.block_size
        chunk_table = ttnn.slice(page_table, [user_id, first_block], [user_id + 1, first_block + num_chunk_blocks])
        for cache, t in ((self.kv_cache[0], k_r), (self.kv_cache[1], v_n)):
            fill = t if t.dtype == self.kv_dtype else ttnn.typecast(t, self.kv_dtype)
            ttnn.experimental.paged_fill_cache(cache, fill, chunk_table, batch_idx=0)
            if fill is not t:
                ttnn.deallocate(fill)
        ttnn.deallocate(k_r)
        ttnn.deallocate(v_n)
        attn = self._prefill_attention(q_r, page_table=page_table, user_id=user_id, chunk_start=chunk_start)
        ttnn.deallocate(q_r)
        attn_flat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        o = self._prefill_matmul(attn_flat, self.wo, "o", dtype=self.prefill_o_dtype)
        ttnn.deallocate(attn_flat)
        return self._mlp_and_residuals(x, o, self._prefill_matmul)

    def prefill_forward(self, hidden_states, *, page_table, start_pos=0, user_id=0):
        """This chip's ``[1, 1, S, 960]`` residual slice -> ``[1, 1, S, 960]`` for one request.

        Same contract as ``OptimizedDecoder.prefill_forward`` (any ``S >= 1``; ``start_pos`` a multiple of
        ``block_size``; ``page_table`` ``[users, pages]`` replicated) on the hidden-sharded residual layout
        (``prefill_input_mesh_mapper``). Non-aligned tails are zero-padded to a block inside each chunk and sliced off.
        """
        seq_len = hidden_states.shape[-2]
        width = self.local_hidden
        assert hidden_states.shape[0] == 1 and hidden_states.shape[1] == 1, "prefill is one request per call"
        assert hidden_states.shape[-1] == width, f"prefill takes the {width}-column per-chip residual slice"
        assert seq_len >= 1 and start_pos >= 0 and start_pos % self.block_size == 0
        assert start_pos + seq_len <= self.max_context
        chunk = self.prefill_chunk_size
        outputs = []
        for offset in range(0, seq_len, chunk):
            valid = min(chunk, seq_len - offset)
            padded = -(-valid // self.block_size) * self.block_size
            x = ttnn.slice(hidden_states, [0, 0, offset, 0], [1, 1, offset + valid, width])
            if padded != valid:
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, padded - valid), (0, 0)], 0.0)
            outputs.append(
                self._prefill_chunk(
                    x, page_table=page_table, user_id=user_id, chunk_start=start_pos + offset, padded_len=padded
                )
            )
        out = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        if len(outputs) > 1:
            for t in outputs:
                ttnn.deallocate(t)
        if out.shape[-2] != seq_len:
            out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, seq_len, width])
        return out

    # ------------------------------------------------------------------ decode
    def _decode_mm_reduced(self, x, role):
        """Row-parallel decode matmul (o_proj, down) and the one-shot all-reduce of its partial sums."""
        dtype = self.ar_dtype[role]
        partial = ttnn.linear(
            x,
            self.wo if role == "o" else self.w_down,
            program_config=self.mm_cfg[role],
            compute_kernel_config=self.decode_ck[role],
            memory_config=self.residual_mem,
            dtype=dtype,
        )
        out = self.ccl.all_reduce(partial, role, self.residual_mem)
        ttnn.deallocate(partial)
        return out

    def use_user_split(self, batch, max_position):
        """Generator helper: whether a decode trace for ``batch`` users whose largest position is ``max_position``
        should take the user-split full-layer attention (``decode_forward(..., user_split=...)``). The split has a fixed
        cost (two gathers, head concats) and saves cache reads that grow with context, so it wins from a per-batch
        position (``decode_user_split_min_position``, measured on layer 5). A generator captures both variants for a
        batch and replays the one this returns; the persistent state of both exists from construction."""
        if batch % self.tp or batch < self.tp:
            return False
        table = {int(b): v for b, v in self.policy["decode_user_split_min_position"].items()}
        keys = [b for b in table if b >= batch]
        return max_position >= table[min(keys) if keys else max(table)]

    def decode_split_inputs(self, current_pos, page_table, user_split=None):
        """(positions, page-table rows) of this chip's users for the user-split full-layer attention: per-device slices
        of the replicated ``current_pos`` ``[B]`` and ``page_table`` ``[B, pages]``. They depend only on the step's
        inputs, so a layer stack computes them once per step and passes them to every full layer
        (``decode_forward(..., split_inputs=...)``); the caller then owns them. None when ``B`` (``current_pos``'s
        length) does not take the user-split path (``user_split`` as in ``decode_forward``)."""
        if not self._user_split(current_pos.shape[0], user_split):
            return None
        return (
            ttnn.mesh_partition(current_pos, dim=0, cluster_axis=CLUSTER_AXIS),
            ttnn.mesh_partition(page_table, dim=0, cluster_axis=CLUSTER_AXIS),
        )

    def _user_split(self, batch, user_split=None):
        """Whether ``batch`` users take the user-split path: ``user_split`` if given (needs a multiple of 4), else the
        policy default ``decode_user_split_min_batch``."""
        if user_split is not None:
            assert not user_split or batch % self.tp == 0, f"user split needs a multiple of {self.tp} users"
            return bool(user_split)
        b = self.policy["decode_user_split_min_batch"]
        return b is not None and batch >= b and batch % self.tp == 0

    def decode_forward(
        self, hidden_states, *, current_pos, page_table, rope_rows=None, split_inputs=None, user_split=None
    ):
        """``[1, 1, B, 3840]`` replicated -> ``[1, 1, B, 3840]`` replicated in ``decode_input_memory_config()``.

        Contract as ``OptimizedDecoder.decode_forward``: ``current_pos`` ``[B]`` int32 (-1 skips a user) and
        ``page_table`` ``[B, pages]`` are replicated; each chip updates its local KV heads. Sliding layers attend their
        local KV heads; full layers at ``B >= decode_user_split_min_batch`` (a multiple of 4) attend a quarter of the
        users with all heads (``_decode_attention_user_split``); ``user_split`` (True / False) overrides that choice for
        this call (a multiple of 4 users; ``use_user_split`` picks it by context length). On the split path
        ``current_pos`` and ``page_table`` must have exactly B rows. ``rope_rows`` / ``split_inputs``: optional per-step
        tensors shared across a stack (``decode_rope_rows``, ``decode_split_inputs`` with the same ``user_split``).
        """
        batch = hidden_states.shape[-2]
        assert 1 <= batch <= MAX_DECODE_BATCH
        layout = self._decode_layout(batch)
        x = hidden_states
        if x.memory_config() != self.residual_mem:
            x = ttnn.to_memory_config(x, self.residual_mem)
        h = self._rms_res(x, self.input_norm)
        qkv = self._decode_mm(h, "qkv")
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        head_shard = q.memory_config()
        kv_shard = k.memory_config()
        q_n, k_n, v_n = self._decode_head_norms(q, k, v)
        for t in (q, k, v):
            ttnn.deallocate(t)

        own_rows = rope_rows is None
        cos, sin = self.decode_rope_rows(current_pos, batch) if own_rows else rope_rows
        q_s = ttnn.to_memory_config(q_n, head_shard)
        k_s = ttnn.to_memory_config(k_n, kv_shard)
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)
        if self.meta_rope:
            rope = dict(is_decode_mode=True, compute_kernel_config=self.compute_rope_decode)
            q_r = ttnn.experimental.rotary_embedding_llama(q_s, cos, sin, self.rope_trans["decode"], **rope)
            k_r = ttnn.experimental.rotary_embedding_llama(k_s, cos, sin, self.rope_trans["decode"], **rope)
        else:
            q_r = ttnn.experimental.rotary_embedding_hf(q_s, cos, sin, is_decode_mode=True)
            k_r = ttnn.experimental.rotary_embedding_hf(k_s, cos, sin, is_decode_mode=True)
        for t in (q_s, k_s, cos, sin) if own_rows else (q_s, k_s):
            ttnn.deallocate(t)

        v_s = ttnn.to_memory_config(v_n, layout["v"])
        ttnn.deallocate(v_n)
        ttnn.experimental.paged_fused_update_cache(
            self.kv_cache[0], k_r, self.kv_cache[1], v_s, update_idxs_tensor=current_pos, page_table=page_table
        )
        ttnn.deallocate(k_r)
        ttnn.deallocate(v_s)
        if self.kv_replicated and self._user_split(batch, user_split):
            # the per-chip partitions slice rows [dB/4, (d+1)B/4): the positions and page table must have B rows
            assert current_pos.shape[0] == batch and page_table.shape[0] == batch, "user split needs B-row inputs"
            attn_flat = self._decode_attention_user_split(q_r, current_pos, page_table, batch, split_inputs)
            ttnn.deallocate(q_r)
            o = self._decode_mm_reduced(attn_flat, "o")
            ttnn.deallocate(attn_flat)
            return self._decode_mlp_and_residuals(x, o)
        direct = self.num_kv_heads == 1
        k_chunk, sdpa_cfg = self._sdpa_decode_cfg(batch)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_r,
            self.kv_cache[0],
            self.kv_cache[1],
            page_table_tensor=self._table_for_k_chunk(page_table, k_chunk),
            cur_pos_tensor=current_pos,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_sdpa_decode,
            memory_config=head_shard if direct else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_r)
        if not direct:
            attn_i = attn
            attn = ttnn.to_memory_config(attn_i, head_shard)
            ttnn.deallocate(attn_i)
        attn_flat = ttnn.experimental.nlp_concat_heads_decode(
            attn, num_heads=self.num_heads, sub_core_grids=self.worker_grid
        )
        ttnn.deallocate(attn)
        if attn_flat.shape[-2] != batch:
            width = attn_flat.shape[-1]
            attn_flat = ttnn.reshape(attn_flat, [1, 1, batch, width], [1, 1, attn_flat.padded_shape[-2], width])
        o = self._decode_mm_reduced(attn_flat, "o")
        ttnn.deallocate(attn_flat)
        return self._decode_mlp_and_residuals(x, o)

    def _decode_attention_user_split(self, q_r, current_pos, page_table, batch, split_inputs=None):
        """Full-layer decode attention split by user: every chip holds the same replicated KV cache, so chip ``d``
        attends users ``[dB/4, (d+1)B/4)`` with all 16 Q heads and reads only their cache pages (4x fewer KV bytes per
        chip than attending every user with its own 4 heads).

        Q: own heads concatenated per user, all-gathered over the heads (columns), padded with two head-widths of
        columns so ``nlp_create_qkv_heads_decode`` sees [16 Q | K | V], which then slices this chip's users with a
        per-chip ``batch_offset``. Output: gathered back by user, heads concatenated, this chip's 4 heads' columns
        kept for its row-parallel o_proj. Returns ``attn_flat`` ``[1, 1, B, 4 * hd]`` in the o_proj input layout.
        """
        loc = batch // self.tp
        us = self._user_split_state(batch)
        q_cat = ttnn.experimental.nlp_concat_heads_decode(
            q_r, num_heads=self.num_heads, sub_core_grids=self.worker_grid
        )
        mode = self.policy["decode_split_ag_buffers"]
        l1 = self.policy["decode_split_ag_mem"] == "l1"
        bufs = mode != "none" and not l1
        ag_mem = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG
        q_all = self.ccl.all_gather(
            q_cat,
            dim=3,
            memory_config=ag_mem,
            persistent_output_buffer=us["q_all"] if bufs else None,
            barrier=not bufs or mode != "persistent",
        )  # [1, 1, B, 16 * hd]
        ttnn.deallocate(q_cat)
        q_pad = ttnn.concat([q_all, us["kv_pad"]], dim=3, memory_config=us["qkv_mem"])  # [1, 1, B, 18 * hd]
        if not bufs:
            ttnn.deallocate(q_all)
        q_loc, k_junk, v_junk = ttnn.experimental.nlp_create_qkv_heads_decode(
            q_pad,
            num_heads=self.num_heads_total,
            num_kv_heads=1,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            batch_offset=us["offset"],
            slice_size=loc,
        )  # q_loc [1, B/4, 16, hd]: this chip's users
        for t in (q_pad, k_junk, v_junk):
            ttnn.deallocate(t)
        own = split_inputs is None
        pos_loc, table_loc = self.decode_split_inputs(current_pos, page_table, user_split=True) if own else split_inputs
        k_chunk, sdpa_cfg = self._sdpa_decode_cfg(loc, split=True)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_loc,
            self.kv_cache[0],
            self.kv_cache[1],
            page_table_tensor=self._table_for_k_chunk(table_loc, k_chunk),
            cur_pos_tensor=pos_loc,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_sdpa_decode,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, B/4, 16, hd]
        for t in (q_loc, pos_loc, table_loc) if own else (q_loc,):
            ttnn.deallocate(t)
        attn_all = self.ccl.all_gather(
            attn,
            dim=1,
            memory_config=ag_mem,
            persistent_output_buffer=us["attn_all"] if bufs else None,
            barrier=not bufs or mode != "persistent",
        )  # [1, B, 16, hd]
        ttnn.deallocate(attn)
        attn_s = ttnn.to_memory_config(attn_all, us["heads_mem"])
        if not bufs:
            ttnn.deallocate(attn_all)
        cat = ttnn.experimental.nlp_concat_heads_decode(
            attn_s, num_heads=self.num_heads_total, sub_core_grids=self.worker_grid
        )  # [1, 1, B, 16 * hd]
        ttnn.deallocate(attn_s)
        flat = ttnn.mesh_partition(cat, dim=3, cluster_axis=CLUSTER_AXIS)  # [1, 1, B, 4 * hd]: own heads
        ttnn.deallocate(cat)
        flat_s = ttnn.to_memory_config(flat, self.attn_flat_mem)
        ttnn.deallocate(flat)
        if flat_s.shape[-2] != batch:
            # tile-padded to 32 user rows; restore the logical batch (metadata only)
            width = flat_s.shape[-1]
            flat_s = ttnn.reshape(flat_s, [1, 1, batch, width], [1, 1, flat_s.padded_shape[-2], width])
        return flat_s

    def _user_split_state(self, batch):
        """Persistent tensors of the user-split attention for ``batch`` users (created on the first, compiling call)."""
        key = ("user_split", batch, self.head_dim)
        cache = self.ccl.decode_state  # shared by every full layer on the mesh
        if key not in cache:
            hd = self.head_dim
            heads = self.num_heads_total + 2
            cache[key] = {
                # two head-widths of zero columns standing in for K and V
                "kv_pad": ttnn.from_torch(
                    torch.zeros(1, 1, TILE, 2 * hd),  # the head-concat output keeps 32 (tile-padded) user rows
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                ),
                # [16 Q | K | V] width-sharded, one head per core (the sharded create-heads program takes batch_offset)
                "qkv_mem": self._width_sharded(heads * hd, heads),
                # chip d's first user
                "offset": ttnn.from_torch(
                    torch.tensor([[d * (batch // self.tp)] for d in range(self.tp)], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
                ),
                # every user's [32, hd] head block on its own core (head-concat input)
                "heads_mem": self._decode_layout(batch)["rope"],
            }
        if self.policy["decode_split_ag_buffers"] != "none" and "q_all" not in cache[key]:
            for name, shape in (
                ("q_all", [1, 1, TILE, self.num_heads_total * self.head_dim]),
                ("attn_all", [1, batch, self.num_heads_total, self.head_dim]),
            ):
                cache[key][name] = ttnn.empty(
                    shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
                )
        return cache[key]

    def _decode_mlp_and_residuals(self, x, o):
        """Decode: post-attention norm + residual, gated MLP, down all-reduce, post-FFN norm + scaled residual."""
        h = self._rms_res(o, self.post_attn_norm)
        ttnn.deallocate(o)
        x_res = ttnn.add(x, h, memory_config=self.residual_mem)
        ttnn.deallocate(h)
        h = self._rms_res(x_res, self.pre_ffn_norm)
        gate = self._decode_mm(h, "gate")
        up = self._decode_mm(h, "up")
        ttnn.deallocate(h)
        prod = ttnn.multiply(gate, up, input_tensor_a_activations=[self.gelu], memory_config=gate.memory_config())
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        down = self._decode_mm_reduced(prod, "down")
        ttnn.deallocate(prod)
        h = self._rms_res(down, self.post_ffn_norm)
        ttnn.deallocate(down)
        out = ttnn.add(
            x_res,
            h,
            activations=[(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.layer_scalar)],
            memory_config=self.residual_mem,
        )
        ttnn.deallocate(x_res)
        ttnn.deallocate(h)
        return out
