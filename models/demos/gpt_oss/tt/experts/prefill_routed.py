# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Routed (top-k only) prefill MoE experts for a single-row mesh (Blackhole 1xN).

The dense prefill path (experts/prefill.py) runs every token through all experts at TP width. This path computes
only the routed (token, expert) pairs, with local expert parallelism on the TP axis:

  * each of the N devices holds num_experts / N whole experts (hidden x intermediate at full width); on 1x4 the
    expert -> device assignment is a per-layer calibrated permutation that balances the devices' token-pair load
    (prefill_routed_placement.py, GPT_OSS_ROUTED_PREFILL_PLACEMENT): the weights are placed in that order and the
    MLP's routed router emits slot ids (TopKRouter.with_expert_order), so every table below stays the identity;
  * the replicated input x and the router's top-k ids stay on every device; each device keeps only the pairs whose
    expert it holds and packs their tokens into tile-aligned per-expert regions (chip-local, no fabric traffic);
  * unified_routed_expert_moe runs the local experts on their regions (SwiGluOai with biases);
  * a gather back to [token, slot] and post_combine_reduce apply the router weights and sum the top-k slots, giving
    this device's partial MoE output (zero for tokens routed elsewhere);
  * one all-reduce over the mesh row sums the partials, the same collective the dense TP path ends with.

Counts and region offsets come from masked_bincount (masked to this device's experts by the dispatch table) and
offset_cumsum along cluster_axis 1 (the row; TtMoERoutingSetup's axis 0 has one device here), where the other
devices contribute zero to this device's experts. The token packing and the gather back use stock ops (fp32 one-hot
ranks, scatter, embedding) instead of deepseek_prefill.dispatch / combine: those accept only cluster_axis 0 and need
a neighbour along it, which a 1xN mesh does not have.

Numerics policy: LoFi, bf16 dest accumulation, bf8 activations, bf4 weights except the layers in
GPT_OSS_ROUTED_PREFILL_BF8_LAYERS (default: layers 0 and 1), which hold bf8 weights. unified_routed_expert_moe hardcodes
LoFi and bf16 dest (kFp32DestAccEn = false in unified_routed_expert_ffn_program_factory.cpp) and ignores
compute_kernel_config, so this module does not pass one.

Only one sequence per call (batch 1, leading dims all 1); anything else runs the dense prefill path.
"""

import math
import os
import types
import weakref
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert, routed_expert_weight_memory_config
from models.demos.gpt_oss.config import Mode

from ..batched_prefill import batched_reduce_links, per_user_reduce
from .prefill_routed_placement import ROUTED_EXPERT_PLACEMENT, ROUTED_EXPERT_PLACEMENT_DEVICES

ROUTED_PREFILL_ENV = "GPT_OSS_ROUTED_PREFILL"
ROUTED_PREFILL_MIN_SEQ_ENV = "GPT_OSS_ROUTED_PREFILL_MIN_SEQ"
ROUTED_PREFILL_CHUNK_ENV = "GPT_OSS_ROUTED_PREFILL_CHUNK"
ROUTED_PREFILL_BF8_LAYERS_ENV = "GPT_OSS_ROUTED_PREFILL_BF8_LAYERS"
# Expert -> device placement of the routed experts: "calibrated" (default) holds each layer's experts in the order of
# ROUTED_EXPERT_PLACEMENT (prefill_routed_placement.py), "identity" keeps expert e on device e // (E / N).
ROUTED_PREFILL_PLACEMENT_ENV = "GPT_OSS_ROUTED_PREFILL_PLACEMENT"
ROUTED_PREFILL_PLACEMENT_DEFAULT = "calibrated"
# The routed path is the default where the mesh supports it (Blackhole, one mesh row: routed_prefill_mesh_supported);
# GPT_OSS_ROUTED_PREFILL=0 selects the dense all-expert prefill path instead.
ROUTED_PREFILL_DEFAULT = "1"
# Shortest prefill (tokens) that takes the routed path; shorter ones run dense.
ROUTED_PREFILL_MIN_SEQ_DEFAULT = 128
# Tokens per routed chunk: bounds the dispatch / expert-output buffers ([4C, H] bf16 per device). 16384 vs 8192
# measured 16K 583.2 -> 565.2 ms, 32K 1288.2 -> 1277.4, 64K 3027.4 -> 3004.9, 128K 8085.9 -> 8047.6 (fewer per-chunk
# fixed costs); DRAM at 128K: min free 19.4 -> 19.2 GiB per device (L10 lever c).
ROUTED_PREFILL_CHUNK_DEFAULT = 16384
# Chunks of at most this many tokens take their [C, C] prefix-sum operator from a shared [8192, 8192] constant, longer
# ones from the [chunk, chunk] one, so an 8K prompt reads the 8192 operator in place instead of a per-layer slice copy
# of the larger one (measured +6 ms TTFT at 8K with only the 16384 operator).
ROUTED_PREFIX_TRI_SMALL = 8192
# Chunks of a multiple of this many tokens (at least two blocks) take the blocked prefix sum (_SharedBlockedPrefix)
# instead of the [C, C] operator, whose read grows as C^2 (0.81 ms per 16384-token chunk on BH, a 64K-token batched
# forward has 4 per layer).
ROUTED_PREFIX_BLOCK = 512
# masked_bincount splits the token rows over a fixed 64-core grid.
ROUTED_PREFILL_SEQ_MULTIPLE = 64
# ttnn.scatter parallelizes over rows only, so the [1, k*C] row -> token scatter of _chunk runs on one core (1.5 ms per
# layer at C = 8192). From this chunk length up, the k*C pairs are split into ROUTED_SCATTER_GROUPS rows that scatter
# on separate cores into their own copy of the buffer, then a max over the copies merges them.
ROUTED_SCATTER_SPLIT_MIN_C = 1024
ROUTED_SCATTER_GROUPS = 32
# The combine gather writes k rows per token ([Cs, k, H] bf16) that post_combine_reduce reads straight back. It runs on
# token blocks of at most this many tokens so that intermediate (47 MB at 2048) and the reduced block live in L1 instead
# of a DRAM round trip; at C = 8192 this measured 2.39 -> 1.91 ms per layer for gather + reduce + bf8 cast (one device).
ROUTED_COMBINE_BLOCK = 2048
# Largest chunk-size-dependent transient this module places in L1 (per device, interleaved). Above it the tensor goes
# to DRAM, so a larger GPT_OSS_ROUTED_PREFILL_CHUNK still runs instead of overflowing L1 (the dispatch-gather input
# x_rm has its own cap, ROUTED_XRM_L1_CAP_BYTES); the combine block is bounded by ROUTED_COMBINE_BLOCK (2048 tokens:
# [Cs, k, H] bf16 = 47 MB, the reduced block 12 MB), which is also under it.
ROUTED_L1_CAP_BYTES = 64 << 20
# Cap for the dispatch-gather input x_rm alone (C*H*2 bytes): it is the only large transient live in L1 while it
# exists (tok_of_row is small; the gather output goes to DRAM), so the default 16384 chunk (90 MiB) fits. Measured on
# one BH device at C = 16384: untilize + dispatch gather 1.86 ms (x_rm in DRAM) -> 1.52 ms (L1), bit-identical.
ROUTED_XRM_L1_CAP_BYTES = 96 << 20
# Version of the cached routed-expert weights; the weight-cache completion marker records it (model_config.py).
ROUTED_PREFILL_WEIGHTS_VERSION = 1
ROUTED_CACHE_SUBDIR = "experts_routed_ep"
ROUTED_WEIGHTS_DTYPE = ttnn.bfloat4_b
# Layers whose routed expert weights are bfloat8_b instead of bfloat4_b ("all", "" or a comma list of layer ids).
# bf8 holds the MXFP4 checkpoint values almost exactly (weight rel. error ~1e-4 for gate/up); bf4 shares one exponent
# per 16 values and loses 12-16% of the weight norm on every layer. Layers 0 and 1 are the smallest set that passes the
# per-prompt accuracy guard (layer 0 alone does not); each bf8 layer costs ~0.2 ms TTFT at 1K, and "all" ~4 ms.
ROUTED_PREFILL_BF8_LAYERS_DEFAULT = "0,1"


def routed_prefill_enabled():
    return os.getenv(ROUTED_PREFILL_ENV, ROUTED_PREFILL_DEFAULT) == "1"


def routed_prefill_min_seq():
    return int(os.getenv(ROUTED_PREFILL_MIN_SEQ_ENV, str(ROUTED_PREFILL_MIN_SEQ_DEFAULT)))


def routed_prefill_chunk():
    chunk = int(os.getenv(ROUTED_PREFILL_CHUNK_ENV, str(ROUTED_PREFILL_CHUNK_DEFAULT)))
    assert chunk > 0 and chunk % ROUTED_PREFILL_SEQ_MULTIPLE == 0, f"{ROUTED_PREFILL_CHUNK_ENV}={chunk}"
    return chunk


def routed_prefill_bf8_layers():
    """Raw GPT_OSS_ROUTED_PREFILL_BF8_LAYERS spec (recorded in the weight-cache marker)."""
    return os.getenv(ROUTED_PREFILL_BF8_LAYERS_ENV, ROUTED_PREFILL_BF8_LAYERS_DEFAULT).strip()


def routed_weights_dtype(layer_idx):
    """Routed expert weight dtype of one layer: bfloat8_b for the layers in the bf8 list, else bfloat4_b."""
    spec = routed_prefill_bf8_layers()
    if spec == "all":
        return ttnn.bfloat8_b
    layers = {int(v) for v in spec.split(",") if v.strip() != ""}
    return ttnn.bfloat8_b if layer_idx is not None and int(layer_idx) in layers else ROUTED_WEIGHTS_DTYPE


def routed_expert_order(layer_idx, num_experts, num_devices):
    """Expert order of one layer's routed experts, or None for the identity order.

    order[v] is the expert held at virtual slot v: device v // (E / N), local slot v % (E / N). The calibrated orders
    (prefill_routed_placement.py) balance the per-device token-pair load of a 1xN mesh with
    N = ROUTED_EXPERT_PLACEMENT_DEVICES; any other N, an uncalibrated layer and GPT_OSS_ROUTED_PREFILL_PLACEMENT=identity
    keep the identity order."""
    mode = os.getenv(ROUTED_PREFILL_PLACEMENT_ENV, ROUTED_PREFILL_PLACEMENT_DEFAULT).strip().lower()
    assert mode in ("calibrated", "identity"), f"{ROUTED_PREFILL_PLACEMENT_ENV}={mode!r}"
    if mode == "identity" or layer_idx is None or num_devices != ROUTED_EXPERT_PLACEMENT_DEVICES:
        return None
    order = ROUTED_EXPERT_PLACEMENT.get(int(layer_idx))
    if order is None or list(order) == list(range(num_experts)):
        return None
    assert sorted(order) == list(range(num_experts)), f"layer {layer_idx}: placement is not a permutation"
    return list(order)


def _l1_if_fits(nbytes):
    """L1 for a transient of nbytes per device when it fits under ROUTED_L1_CAP_BYTES, else DRAM."""
    return ttnn.L1_MEMORY_CONFIG if nbytes <= ROUTED_L1_CAP_BYTES else ttnn.DRAM_MEMORY_CONFIG


def routed_prefill_mesh_supported(mesh_device, num_experts=32):
    """Single-row Blackhole mesh with more than one device whose column count divides the experts."""
    rows, cols = mesh_device.shape
    return rows == 1 and cols > 1 and num_experts % cols == 0 and ttnn.device.is_blackhole(mesh_device)


def routed_prefill_supported(mesh_device, mesh_config, hf_config, use_throughput_experts):
    """Whether the MLP builds the routed prefill experts (env flag, mesh, prefill parallelism)."""
    if not routed_prefill_enabled() or use_throughput_experts:
        return False
    if not routed_prefill_mesh_supported(mesh_device, hf_config.num_local_experts):
        return False
    prefill = mesh_config.get_config(Mode.PREFILL)
    return prefill.tp == mesh_device.shape[1] and prefill.sp == 1 and prefill.ep == 1 and mesh_config.tp_axis == 1


class _SharedPrefixTri:
    """Strictly lower-triangular ones [Cmax, Cmax] bfloat4_b (row t has ones at columns < t), the exclusive
    prefix-sum operator of _pair_positions. One per (mesh device, Cmax), shared by every layer; a chunk of C < Cmax
    tokens uses its top-left [C, C] block, which is the same operator for C tokens. Built on first use."""

    def __init__(self, mesh_device, cmax):
        self.mesh_device = mesh_device
        self.cmax = cmax
        self.tensor = None

    def get(self):
        if self.tensor is None:
            ones = ttnn.ones(
                (self.cmax, self.cmax),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tri = ttnn.tril(ones, diagonal=-1)
            ttnn.deallocate(ones)
            # 0/1 are exact in bfloat4_b (sign, shared exponent, 3-bit mantissa). The prefix matmul is bound by reading
            # this operand, so bf4 instead of bf8 halves its bytes (measured 0.375 -> 0.234 ms at C = 8192, one device).
            self.tensor = ttnn.typecast(tri, ttnn.bfloat4_b)
            ttnn.deallocate(tri)
        return self.tensor


# (mesh device id, Cmax) -> _SharedPrefixTri. Weak values: the layers hold the strong references, so the constant
# (36 MiB per device at Cmax = 8192) lives as long as the model that uses it.
_PREFIX_TRI_CACHE = weakref.WeakValueDictionary()


def _shared_prefix_tri(mesh_device, cmax):
    key = (mesh_device.id(), cmax)
    holder = _PREFIX_TRI_CACHE.get(key)
    if holder is None or holder.mesh_device is not mesh_device:
        holder = _SharedPrefixTri(mesh_device, cmax)
        _PREFIX_TRI_CACHE[key] = holder
    return holder


class _SharedBlockedPrefix:
    """Operators of the blocked exclusive prefix sum over C = nb * b tokens (b = ROUTED_PREFIX_BLOCK), bfloat4_b
    (0/1 values, exact): tri [nb, b, b], the strictly lower-triangular ones per block (prefix inside a block); ones
    [nb, 1, b] (block totals); tri_blocks [nb, nb], strictly lower-triangular (sum of the earlier blocks' totals).
    Their reads grow as C * b instead of C^2. One per (mesh device, C), shared by every layer; built on first use."""

    def __init__(self, mesh_device, C, block):
        assert C % block == 0, (C, block)
        self.mesh_device = mesh_device
        self.C = C
        self.block = block
        self.tensors = None

    def get(self):
        if self.tensors is None:
            nb, b = self.C // self.block, self.block
            kw = dict(
                device=self.mesh_device,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            self.tensors = (
                ttnn.from_torch(torch.tril(torch.ones(b, b), diagonal=-1).expand(nb, b, b).contiguous(), **kw),
                ttnn.from_torch(torch.ones(nb, 1, b), **kw),
                ttnn.from_torch(torch.tril(torch.ones(nb, nb), diagonal=-1), **kw),
            )
        return self.tensors


_BLOCKED_PREFIX_CACHE = weakref.WeakValueDictionary()


def _shared_blocked_prefix(mesh_device, C, block):
    key = (mesh_device.id(), C, block)
    holder = _BLOCKED_PREFIX_CACHE.get(key)
    if holder is None or holder.mesh_device is not mesh_device:
        holder = _SharedBlockedPrefix(mesh_device, C, block)
        _BLOCKED_PREFIX_CACHE[key] = holder
    return holder


class RoutedWeightsUnavailable(RuntimeError):
    """Neither HF expert weights nor a complete routed-weight cache: the routed path cannot be built."""


def _bias_dims(num_experts, hidden_size, intermediate_size, state_dict):
    gate_up_bias = state_dict["gate_up_proj_bias"].reshape(num_experts, 2 * intermediate_size)
    gate = gate_up_bias[..., ::2].reshape(num_experts, intermediate_size)
    up = gate_up_bias[..., 1::2].reshape(num_experts, intermediate_size)
    down = state_dict["down_proj_bias"].reshape(num_experts, hidden_size)
    return {"gate": gate, "up": up, "down": down}


class RoutedPrefillExperts:
    """Top-k routed prefill experts, local expert parallelism over the columns of a 1xN mesh."""

    def __init__(
        self, mesh_device, hf_config, state_dict, ccl_manager, mesh_config, tensor_cache_path=None, layer_idx=None
    ):
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.mesh_config = mesh_config
        self.num_experts = hf_config.num_local_experts
        self.top_k = hf_config.num_experts_per_tok
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size
        self.num_devices = mesh_device.shape[1]
        self.dispatch_group_size = 1
        self.num_dispatch_groups = self.num_devices
        self.experts_per_chip = self.num_experts // self.num_devices
        self.chunk_size = routed_prefill_chunk()
        self.min_seq = routed_prefill_min_seq()
        self.num_links = 1
        self.weights_dtype = routed_weights_dtype(layer_idx)
        # Slot v (device v // experts_per_chip, local slot v % experts_per_chip) holds expert expert_order[v]; None is
        # the identity order. The ids this module receives are slot ids: the MLP routes with a router whose output
        # column v is expert expert_order[v] (TopKRouter.with_expert_order), so the dispatch / global-index tables and
        # the contiguous per-device regions of offset_cumsum stay the identity; only the weights follow the order
        # (_load_weights_ordered). Mapping HF ids to slot ids on device instead (tables in HF ids) needs offset_cumsum
        # regions over all E experts, 32*(E - E/N) more dispatch rows: measured +0.6 ms TTFT at 128, +1.5 ms at 512.
        self.expert_order = routed_expert_order(layer_idx, self.num_experts, self.num_devices)

        # Device g's table row maps its own experts [g*E/N, (g+1)*E/N) to chip g and every other expert to -1: the
        # histogram mask of masked_bincount and the non-local skip of post_combine_reduce. Each pair is computed
        # once, on the device holding its expert. Width E + 1 (padding sentinel column). Per device: (1, E + 1).
        dispatch_table = torch.full((self.num_devices, self.num_experts + 1), -1, dtype=torch.int32)
        for g in range(self.num_devices):
            dispatch_table[g, g * self.experts_per_chip : (g + 1) * self.experts_per_chip] = g
        self._consts = {}
        # The [C, C] prefix-sum operators are shared across layers (per device: [chunk, chunk], plus [8192, 8192] for
        # chunks of at most ROUTED_PREFIX_TRI_SMALL tokens). Each is built on first use.
        self._prefix_tri = _shared_prefix_tri(mesh_device, self.chunk_size)
        self._prefix_tri_small = _shared_prefix_tri(mesh_device, min(self.chunk_size, ROUTED_PREFIX_TRI_SMALL))
        # Blocked prefix operators per chunk length (strong references to the shared holders).
        self._blocked_prefix = {}
        # fp32 dest + L1 packer accumulation keep the integer prefix sums of _pair_positions exact across K blocks.
        self._prefix_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # HiFi4 for the slot selector matmul: its fp32 operand holds expert ids up to E - 1, which LoFi would truncate.
        self._select_compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        k, E = self.top_k, self.num_experts
        # Slot one-hots of all k slots at once, [C, k*E] (column block j = slot j): the selector [k, k*E] copies
        # index column j into block j (one nonzero term per output, exact), compared against the expert ids repeated
        # k times; the fold [k*E, E] sums the k blocks into per-token expert counts.
        select = torch.zeros(k, k * E)
        fold = torch.zeros(k * E, E)
        for j in range(k):
            select[j, j * E : (j + 1) * E] = 1.0
            fold[j * E : (j + 1) * E, :] = torch.eye(E)
        self.slot_select = ttnn.from_torch(
            select,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.slot_fold = ttnn.from_torch(
            fold,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.slot_expert_ids_f32 = ttnn.from_torch(
            torch.arange(E, dtype=torch.float32).repeat(k).reshape(1, k * E),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.expert_ids_f32 = ttnn.from_torch(
            torch.arange(self.num_experts, dtype=torch.float32).reshape(1, self.num_experts),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.local_mask_f32 = ttnn.from_torch(
            (dispatch_table[:, : self.num_experts] >= 0).to(torch.float32),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(None, 0)),
        )
        self.dispatch_table = ttnn.from_torch(
            dispatch_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(None, 0)),
        )
        global_idx = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=self.experts_per_chip,
                dispatch_group_size=self.dispatch_group_size,
                num_dispatch_groups=self.num_dispatch_groups,
            ),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=get_ep_mesh_mapper(mesh_device),
        )
        self.global_expert_idx_table = ttnn.squeeze(ttnn.squeeze(global_idx, 0), 0)

        cache_dir = None
        if tensor_cache_path is not None:
            cache_dir = Path(tensor_cache_path) / f"{ROUTED_CACHE_SUBDIR}{mesh_device.shape[0]}x{mesh_device.shape[1]}"
        self.cache_dir = cache_dir
        prefix = f"routed_v{ROUTED_PREFILL_WEIGHTS_VERSION}"

        has_weights = bool(state_dict) and "gate_up_proj" in state_dict
        if has_weights and cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        if not has_weights and not self._cache_complete(prefix):
            raise RoutedWeightsUnavailable(
                f"Routed prefill expert weights are not cached in {cache_dir} and no HF expert weights were loaded. "
                f"Run once with GPT_OSS_FORCE_MODEL_LOAD=1 (or {ROUTED_PREFILL_ENV}=0 for the dense path)."
            )

        torch_weights = None
        if has_weights:
            from models.demos.gpt_oss_d_p.tt.moe.weights import prepare_routed_expert_weights

            torch_weights, _ = prepare_routed_expert_weights(
                state_dict, self.num_experts, self.hidden_size, self.intermediate_size
            )
        # TtRoutedExpert only loads/places the weights here (DRAM ND-sharded on Blackhole); forward calls
        # unified_routed_expert_moe directly so the per-expert token bound follows the chunk length.
        if self.expert_order is None:
            self.routed_expert = TtRoutedExpert(
                mesh_device=mesh_device,
                experts_per_chip=self.experts_per_chip,
                global_expert_idx_table=self.global_expert_idx_table,
                emb_dim=self.hidden_size,
                hidden_dim=self.intermediate_size,
                max_tokens=self.chunk_size,
                torch_weights=torch_weights,
                torch_biases=None,
                weights_dtype=self.weights_dtype,
                weight_cache_path=cache_dir,
                cache_name_prefix=prefix if cache_dir is not None else None,
                activation=ttnn.RoutedExpertActivation.SwiGluOai,
            )
        else:
            self.routed_expert = self._load_weights_ordered(torch_weights, prefix)
        del torch_weights
        self.gate_biases, self.up_biases, self.down_biases = self._load_biases(
            state_dict if has_weights else None, prefix
        )

    def _cache_complete(self, prefix):
        if self.cache_dir is None or not self.cache_dir.is_dir():
            return False
        names = {p.name for p in self.cache_dir.iterdir()}
        dtype = self.weights_dtype.name
        for e in range(self.experts_per_chip):
            for proj in ("gate", "up", "down"):
                stem = f"{prefix}.local_{e}_{proj}_dtype_{dtype}_"
                if not any(n.startswith(stem) and n.endswith(".tensorbin") for n in names):
                    return False
            for proj in ("gate", "up", "down"):
                stem = f"{prefix}.local_{e}_{proj}_bias_dtype_BFLOAT16_"
                if not any(n.startswith(stem) and n.endswith(".tensorbin") for n in names):
                    return False
        return True

    def _place_ordered(self, host_slots, memory_config):
        """Identity-order host tensors -> device tensors in self.expert_order.

        host_slots[e]: multi-device host tensor of local slot e in the identity order (device d's shard is expert
        d * epc + e, the cached layout). Device d's slot e of the result holds expert expert_order[d * epc + e]: its
        shard is moved as stored (no dequantize / requantize), so the bytes are those of the identity cache."""
        epc = self.experts_per_chip
        shards = [ttnn.get_device_tensors(t) for t in host_slots]
        out = []
        for e in range(epc):
            per_device = []
            for d in range(self.num_devices):
                o = self.expert_order[d * epc + e]
                per_device.append(shards[o % epc][o // epc])
            t = ttnn.from_host_shards(per_device, self.mesh_device.shape)
            t = ttnn.squeeze(ttnn.squeeze(t, 0), 0)
            mem = memory_config(t) if callable(memory_config) else memory_config
            out.append(ttnn.to_device(t, self.mesh_device, memory_config=mem))
        return out

    def _load_weights_ordered(self, torch_weights, prefix):
        """Routed expert weights in self.expert_order (gate_projs / up_projs / down_projs like TtRoutedExpert).

        With a cache directory the identity-order cache files are the source (written from the HF weights first if
        missing, exactly as TtRoutedExpert writes them) and their per-device shards are regrouped on host, so one
        cache serves every placement. Without one, the HF weights are permuted and placed directly."""
        rows, cols = self.mesh_device.shape
        epc = self.experts_per_chip
        if self.cache_dir is None:
            assert torch_weights is not None, "routed weights need HF weights or a cache directory"
            te = TtRoutedExpert(
                mesh_device=self.mesh_device,
                experts_per_chip=epc,
                global_expert_idx_table=self.global_expert_idx_table,
                emb_dim=self.hidden_size,
                hidden_dim=self.intermediate_size,
                max_tokens=self.chunk_size,
                torch_weights=[torch_weights[o] for o in self.expert_order],
                torch_biases=None,
                weights_dtype=self.weights_dtype,
                weight_cache_path=None,
                cache_name_prefix=None,
                activation=ttnn.RoutedExpertActivation.SwiGluOai,
            )
            return types.SimpleNamespace(gate_projs=te.gate_projs, up_projs=te.up_projs, down_projs=te.down_projs)
        mapper = ExpertMapping.get_weights_mesh_mapper(self.mesh_device)
        nd_sharded = ttnn.device.is_blackhole(self.mesh_device)  # TtRoutedExpert's default placement

        def weight_mem(t):
            return routed_expert_weight_memory_config(self.mesh_device, t.shape[-1], dram_nd_sharded=nd_sharded)

        placed = {}
        for proj in ("gate", "up", "down"):
            host_slots = []
            for e in range(epc):
                stacked = None
                if torch_weights is not None:
                    g, u, d = ExpertMapping.gather_weights_for_mesh_distribution(torch_weights, e, rows, cols, epc)
                    ws = {"gate": g, "up": u, "down": d}[proj]
                    stacked = torch.stack([w.T.contiguous() for w in ws], dim=0)
                    stacked = stacked.reshape(rows, cols, stacked.shape[1], stacked.shape[2])
                host_slots.append(
                    ttnn.as_tensor(
                        stacked,
                        mesh_mapper=mapper,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=self.weights_dtype,
                        cache_file_name=str(self.cache_dir / f"{prefix}.local_{e}_{proj}"),
                    )
                )
                del stacked
            placed[proj] = self._place_ordered(host_slots, weight_mem)
            del host_slots
        return types.SimpleNamespace(gate_projs=placed["gate"], up_projs=placed["up"], down_projs=placed["down"])

    def _load_biases(self, state_dict, prefix):
        """One (1, N) bf16 TILE bias per local expert and projection, placed like the routed weights (cached in the
        identity order; a non-identity expert_order regroups the cached shards, see _place_ordered)."""
        rows, cols = self.mesh_device.shape
        mapper = ExpertMapping.get_weights_mesh_mapper(self.mesh_device)
        biases = None
        if state_dict is not None:
            biases = _bias_dims(self.num_experts, self.hidden_size, self.intermediate_size, state_dict)
        order = self.expert_order
        # Ordered with a cache: identity host tensors first (the cache files), regrouped below.
        regroup = order is not None and self.cache_dir is not None
        host = {"gate": [], "up": [], "down": []}
        out = {"gate": [], "up": [], "down": []}
        for e in range(self.experts_per_chip):
            for proj in ("gate", "up", "down"):
                stacked = None
                if biases is not None:
                    per_device = []
                    for row in range(rows):
                        for col in range(cols):
                            g = ExpertMapping.get_global_expert_idx(
                                group=col,
                                chip=row,
                                local_expert=e,
                                experts_per_chip=self.experts_per_chip,
                                dispatch_group_size=rows,
                                num_dispatch_groups=cols,
                            )
                            if order is not None and not regroup:
                                g = order[g]  # no cache: place the permuted biases directly
                            per_device.append(biases[proj][g].reshape(1, -1))
                    stacked = torch.stack(per_device, dim=0)
                    stacked = stacked.reshape(rows, cols, 1, stacked.shape[-1]).to(torch.bfloat16)
                cache_name = None
                if self.cache_dir is not None:
                    cache_name = str(self.cache_dir / f"{prefix}.local_{e}_{proj}_bias")
                tt = ttnn.as_tensor(
                    stacked,
                    device=None if regroup else self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                    cache_file_name=cache_name,
                )
                if regroup:
                    host[proj].append(tt)
                else:
                    out[proj].append(ttnn.squeeze(ttnn.squeeze(tt, 0), 0))
        if regroup:
            for proj in ("gate", "up", "down"):
                out[proj] = self._place_ordered(host[proj], ttnn.DRAM_MEMORY_CONFIG)
        return out["gate"], out["up"], out["down"]

    def supports(self, shape):
        """Whether a prefill input of this shape ([..., S, H]) takes the routed path: one sequence (every leading
        dim 1, so the S rows are the S tokens of batch 1), S >= min_seq and S % 64 == 0. Otherwise dense."""
        dims = [shape[i] for i in range(len(shape))]
        if len(dims) < 2 or math.prod(dims[:-2]) != 1:
            return False
        seq_len = dims[-2]
        return seq_len >= self.min_seq and seq_len % ROUTED_PREFILL_SEQ_MULTIPLE == 0

    def _routing(self, indices):
        """Per-expert dispatch offsets, token counts and region offsets of this device's experts.

        indices: [C, k] uint16 TILE. masked_bincount counts only this device's experts (dispatch table mask);
        offset_cumsum gathers the histograms over the mesh row, where the other devices contribute zero to them."""
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(
            indices, self.dispatch_table, self.num_experts, self.top_k
        )
        offsets, counts, region_offsets, all_offsets = ttnn.experimental.deepseek_prefill.offset_cumsum(
            hist,
            cluster_axis=1,
            num_links=self.num_links,
            experts_per_chip=self.experts_per_chip,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(hist)
        ttnn.deallocate(all_offsets)
        return offsets, counts, region_offsets

    def _chunk_constants(self, C):
        """Per-chunk-length constant tensors (built once per C, < 0.3 MiB per device at C = 8192). The [C, C] prefix
        operator is not here: it is one shared [chunk, chunk] constant (_SharedPrefixTri)."""
        consts = self._consts.get(C)
        if consts is not None:
            return consts
        k = self.top_k
        # Buffer rows: the local experts' tile-aligned regions take at most k*C + 32*(E_local - 1) rows (the
        # alignment pads of E_local regions sum to a multiple of 32 below 32*E_local); one spare tile follows.
        spare = k * C + ttnn.TILE_SIZE * (self.experts_per_chip - 1)
        rows = spare + ttnn.TILE_SIZE
        # Scatter order is slot-major (pair r = slot * C + token), so the token of pair r is r mod C.
        pair_token = torch.arange(k * C, dtype=torch.int32).remainder(C).reshape(1, k * C)
        split = C >= ROUTED_SCATTER_SPLIT_MIN_C
        consts = {
            "rows": rows,
            "spare": spare,
            "scatter_groups": ROUTED_SCATTER_GROUPS if split else 1,
            "pair_token": ttnn.from_torch(
                pair_token,
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            ),
            "row_init": ttnn.from_torch(
                torch.zeros(1, rows, dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            ),
        }
        if split:
            G = ROUTED_SCATTER_GROUPS
            fp32_rm = dict(
                device=self.mesh_device,
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            # Token ids < 2^24 are exact in fp32; group g holds pairs [g*k*C/G, (g+1)*k*C/G) of the slot-major order.
            consts["pair_token_g"] = ttnn.from_torch(pair_token.reshape(G, k * C // G).to(torch.float32), **fp32_rm)
            consts["row_init_g"] = ttnn.from_torch(torch.zeros(G, rows, dtype=torch.float32), **fp32_rm)
        # The expert op writes a fresh bf8 buffer when fed the row-major dispatch (_chunk) and leaves rows outside the
        # occupied expert regions uninitialized. The spare tile is the one such region the combine reads (non-local
        # pairs, weighted 0 for a token with no local expert), so it is zeroed with one tile-row insert: zeros at
        # start[idx[0]] = spare, ceil_tile(counts[idx[0]]) = 32 rows. 16-wide uint32 rows keep the pages 64 B aligned.
        u32_row = dict(
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        consts["spare_start"] = ttnn.from_torch(torch.full((1, 16), spare, dtype=torch.int32), **u32_row)
        consts["spare_count"] = ttnn.from_torch(torch.full((1, 16), ttnn.TILE_SIZE, dtype=torch.int32), **u32_row)
        consts["spare_idx"] = ttnn.from_torch(torch.zeros((1, 16), dtype=torch.int32), **u32_row)
        consts["spare_zeros"] = ttnn.from_torch(
            torch.zeros((ttnn.TILE_SIZE, self.hidden_size)),
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # Region start of the non-local experts: the spare tile, [1, E] fp32 per device ((1 - local) * spare).
        nonlocal_f32 = ttnn.rsub(self.local_mask_f32, 1.0)
        consts["nonlocal_spare"] = ttnn.mul(nonlocal_f32, float(spare))
        ttnn.deallocate(nonlocal_f32)
        self._consts[C] = consts
        return consts

    def _exclusive_prefix(self, per_token, C):
        """[C, E] fp32 0/1 -> [C, E] fp32, row t = sum of rows < t (exact integers).

        C a multiple of ROUTED_PREFIX_BLOCK (b), at least 2 blocks: per block i (nb = C / b), the prefix inside the
        block (tri [nb, b, b] @ per_token as [nb, b, E]) plus the exclusive prefix of the block totals, broadcast over
        the block's rows. Measured at C = 16384 on one BH device: 0.44 ms vs 0.82 ms for the [C, C] matmul (b = 512;
        b = 256: 0.48, b = 128: 0.56). A [nb, 1, C] @ [C, E] matmul for the earlier blocks instead runs its K = C on
        one core: 4.1 ms.

        Otherwise one multi-core matmul with the strictly lower-triangular [C, C] ones matrix; a chunk shorter than its
        shared [Cmax, Cmax] operator uses a transient copy of the top-left [C, C] block."""
        b = ROUTED_PREFIX_BLOCK
        E = per_token.shape[-1]
        mm = dict(
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._prefix_compute_config,
        )
        if C % b == 0 and C >= 2 * b:
            holder = self._blocked_prefix.get(C)
            if holder is None:
                holder = self._blocked_prefix[C] = _shared_blocked_prefix(self.mesh_device, C, b)
            tri, ones, tri_blocks = holder.get()
            nb = C // b
            per_block = ttnn.reshape(per_token, (nb, b, E))
            within = ttnn.matmul(tri, per_block, **mm)  # [nb, b, E]: pairs from earlier tokens of the same block
            totals = ttnn.matmul(ones, per_block, **mm)  # [nb, 1, E]: pairs per block
            totals = ttnn.reshape(totals, (nb, E))
            # Block totals reach b, so this operand needs more than LoFi's mantissa bits (HiFi4, as the slot selector).
            earlier = ttnn.matmul(
                tri_blocks,
                totals,
                dtype=ttnn.float32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self._select_compute_config,
            )  # [nb, E]: pairs from all earlier blocks
            ttnn.deallocate(totals)
            earlier = ttnn.reshape(earlier, (nb, 1, E))
            before = ttnn.add(within, earlier, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(within)
            ttnn.deallocate(earlier)
            return ttnn.reshape(before, (C, E))
        assert C <= self._prefix_tri.cmax, f"chunk of {C} tokens exceeds the prefix operator ({self._prefix_tri.cmax})"
        holder = self._prefix_tri_small if C <= self._prefix_tri_small.cmax else self._prefix_tri
        tri = holder.get()
        tri_c = tri if C == holder.cmax else ttnn.slice(tri, (0, 0), (C, C))
        before = ttnn.matmul(tri_c, per_token, **mm)
        if tri_c is not tri:
            ttnn.deallocate(tri_c)
        return before

    def _pair_positions(self, indices, region_offsets, consts):
        """Dispatch-buffer row of every (token, slot) pair, uint32 ROW_MAJOR, in two orders: token-major [C, k] (the
        combine gather index) and slot-major [G, k*C/G] (pair r = slot * C + token in row-major order, the scatter
        index; G = consts["scatter_groups"]).

        Local pair (t, j) of expert e: region_offsets[e] + #pairs of e from tokens < t (a token's k experts are
        distinct, so this is the pair's rank within e's region). Non-local pairs: the spare tile. All arithmetic is
        fp32 on integers < 2^24, so it is exact."""
        C = indices.shape[-2]
        k = self.top_k
        E = self.num_experts
        idx_f = ttnn.typecast(indices, ttnn.float32)
        # Region start per expert: region_offsets for local experts, the spare tile for the others. A non-local pair
        # then lands on the spare tile through the same lookup below (its expert's per-token count is masked to 0).
        region = ttnn.typecast(ttnn.to_layout(region_offsets, ttnn.TILE_LAYOUT), ttnn.float32)  # [1, E]
        region = ttnn.mul(region, self.local_mask_f32, output_tensor=region)
        region = ttnn.add(region, consts["nonlocal_spare"], output_tensor=region)
        slot_idx = ttnn.matmul(
            idx_f,
            self.slot_select,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._select_compute_config,
        )  # [C, k*E]: block j holds index column j
        ttnn.deallocate(idx_f)
        onehot = ttnn.eq(slot_idx, self.slot_expert_ids_f32, output_tensor=slot_idx)  # [C, k*E] 0/1
        per_token = ttnn.matmul(
            onehot,
            self.slot_fold,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._prefix_compute_config,
        )  # [C, E]: pairs of each expert per token (0/1)
        per_token = ttnn.mul(per_token, self.local_mask_f32, output_tensor=per_token)  # local experts only
        # Exclusive prefix over tokens (pairs of e from earlier tokens). Operands are 0/1 (exact in bf4 / tf32) and the
        # sums (< C, 2^14 at the default chunk) accumulate in the fp32 dest, so the result is exact either way.
        before = self._exclusive_prefix(per_token, C)
        ttnn.deallocate(per_token)
        base = ttnn.add(before, region, output_tensor=before)
        ttnn.deallocate(region)
        cols = []
        for j in range(k):
            oh = ttnn.slice(onehot, (0, j * E), (C, (j + 1) * E))  # tile-aligned block j
            cols.append(ttnn.sum(ttnn.mul(oh, base, output_tensor=oh), dim=-1, keepdim=True))
            ttnn.deallocate(oh)
        ttnn.deallocate(onehot)
        ttnn.deallocate(base)
        pos = ttnn.concat(cols, dim=-1)  # [C, k] fp32 TILE
        for c in cols:
            ttnn.deallocate(c)
        pos = ttnn.typecast(pos, ttnn.uint32)
        # Slot-major copy via a tile transpose: [k, C] row-major is k long rows, so the flatten to [1, k*C] is cheap
        # (the token-major [C, k] -> [1, k*C] reshape copies C 16-byte rows and cost 3.3 ms per layer at 8K).
        pos_t = ttnn.transpose(pos, -2, -1)  # [k, C]
        G = consts["scatter_groups"]
        pos_sm = ttnn.reshape(ttnn.to_layout(pos_t, ttnn.ROW_MAJOR_LAYOUT), (G, k * C // G))
        ttnn.deallocate(pos_t)
        pos_tm = ttnn.to_layout(pos, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(pos)
        return pos_tm, pos_sm

    def _chunk(self, x, indices, weights):
        """Partial MoE output of this device's experts for one chunk.

        x: [1, 1, C, H] (any float dtype, TILE); indices: [C, k] uint16 TILE; weights: [C, k] bf16 TILE.
        Returns the [1, 1, Cs, H] bf8 TILE blocks of the output in token order (ROUTED_COMBINE_BLOCK tokens each, the
        last one shorter; zero rows where no top-k expert is local)."""
        C = x.shape[-2]
        H = self.hidden_size
        k = self.top_k
        # A token picks k distinct experts, so one expert holds at most C pairs.
        max_tokens_per_expert = C

        offsets, counts, region_offsets = self._routing(indices)

        ttnn.deallocate(offsets)
        idx_rm = ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (1, C, k))
        w_rm = ttnn.to_layout(weights, ttnn.ROW_MAJOR_LAYOUT)
        if w_rm.dtype != ttnn.bfloat16:
            w_rm = ttnn.typecast(w_rm, ttnn.bfloat16)
        w_5d = ttnn.reshape(w_rm, (1, 1, C, k, 1))

        # Buffer row of every (token, slot) pair: its expert's region start + its rank among that expert's pairs
        # (token order); pairs whose expert is on another device point at the spare tile past all regions.
        consts = self._chunk_constants(C)
        pos_tm, pos_sm = self._pair_positions(indices, region_offsets, consts)  # [C, k] and [1, k*C] uint32 RM

        # Dispatch: buffer row -> source token (scatter of the pair's token index to its row), then gather the rows.
        # Rows no pair writes (region padding, the spare tile) read token 0 (the split path: the spare row reads another
        # valid token): finite and never consumed.
        if consts["scatter_groups"] == 1:
            tok_of_row = ttnn.scatter(consts["row_init"], -1, pos_sm, consts["pair_token"])
            ttnn.deallocate(pos_sm)
            tok_of_row = ttnn.to_layout(tok_of_row, ttnn.ROW_MAJOR_LAYOUT)
        else:
            # One scatter row (core) per pair group, each into its own zero-initialized copy of the buffer. A local
            # pair's row is unique, so exactly one copy holds its token and the others hold 0; the max over the
            # copies is that token (exact in fp32). Rows no pair writes stay 0. The spare row gets the largest token
            # among the non-local pairs instead of the last one written, a valid row of x either way; the expert op
            # reads only the expert regions, so it never consumes it.
            tok_g = ttnn.scatter(consts["row_init_g"], -1, pos_sm, consts["pair_token_g"])  # [G, rows] fp32
            ttnn.deallocate(pos_sm)
            tok_g_tile = ttnn.to_layout(tok_g, ttnn.TILE_LAYOUT)
            ttnn.deallocate(tok_g)
            tok_max = ttnn.max(tok_g_tile, dim=0, keepdim=True)  # [1, rows] fp32
            ttnn.deallocate(tok_g_tile)
            tok_u = ttnn.typecast(tok_max, ttnn.uint32)
            ttnn.deallocate(tok_max)
            tok_of_row = ttnn.to_layout(tok_u, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(tok_u)
        # The row-major copy of x lives in L1 (47 MB at C = 8192) only for the gather: the untilize skips a DRAM write
        # and the gather reads the local pairs' rows from L1 (measured 0.94 -> 0.77 ms per layer at 8K, one device).
        # A chunk whose copy exceeds ROUTED_XRM_L1_CAP_BYTES (C > 17476) keeps it in DRAM.
        x_rm_mem = ttnn.L1_MEMORY_CONFIG if C * H * 2 <= ROUTED_XRM_L1_CAP_BYTES else ttnn.DRAM_MEMORY_CONFIG
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=x_rm_mem)
        x_rm = ttnn.reshape(x_rm, (C, H))
        # padding_idx=0: every row that reads token 0 (about 3/4 of the k*C buffer rows on each of the N devices are
        # unwritten and read token 0) is served from a per-core L1 copy of x[0] instead of DRAM; same values.
        # Row-major bf16 output: the expert op tilizes and packs its regions to bf8 itself, so the gather does no
        # tilize and the k*C-row bf8 typecast is gone.
        dispatched = ttnn.embedding(tok_of_row, x_rm, padding_idx=0, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(tok_of_row)
        ttnn.deallocate(x_rm)
        dispatched = ttnn.reshape(dispatched, (consts["rows"], H))

        # Row-major bf16 buffer in, fresh TILE bf8 buffer out: the op computes every local expert on its region.
        expert_out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            dispatched,
            region_offsets,
            counts,
            self.global_expert_idx_table,
            self.routed_expert.gate_projs,
            self.routed_expert.up_projs,
            self.routed_expert.down_projs,
            max_dispatched_tokens_per_expert=max_tokens_per_expert,
            activation=ttnn.RoutedExpertActivation.SwiGluOai,
            gate_biases=self.gate_biases,
            up_biases=self.up_biases,
            down_biases=self.down_biases,
        )
        ttnn.deallocate(dispatched)
        ttnn.deallocate(counts)
        ttnn.deallocate(region_offsets)
        expert_out = ttnn.experimental.deepseek_prefill.insert(
            expert_out,
            consts["spare_zeros"],
            consts["spare_start"],
            consts["spare_count"],
            consts["spare_idx"],
            local_expert_id=0,
        )

        # Combine: gather each pair's expert output row back to [Cs, k, H]; post_combine_reduce weights the slots and
        # skips the non-local ones (dispatch table -1), so the spare-tile rows they point at never contribute.
        out_rm = ttnn.to_layout(expert_out, ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        if out_rm is not expert_out:
            ttnn.deallocate(expert_out)
        # One token block (ROUTED_COMBINE_BLOCK) at a time: the [Cs, k, H] gather output and the reduced [Cs, H] block
        # stay in L1 (interleaved), so only the bf8 block is written to DRAM. Bit-identical to one DRAM-resident pass.
        Cs = min(C, ROUTED_COMBINE_BLOCK)
        # Bounded by ROUTED_COMBINE_BLOCK, not by C; the cap only guards a larger block constant.
        combined_mem = _l1_if_fits(Cs * k * H * 2)
        part_mem = _l1_if_fits(Cs * H * 2)
        pieces = []
        for s in range(0, C, Cs):
            e = min(s + Cs, C)
            if s == 0 and e == C:
                pos_s, w_s, idx_s = pos_tm, w_5d, idx_rm
            else:
                pos_s = ttnn.slice(pos_tm, (s, 0), (e, k))
                w_s = ttnn.slice(w_5d, (0, 0, s, 0, 0), (1, 1, e, k, 1))
                idx_s = ttnn.slice(idx_rm, (0, s, 0), (1, e, k))
            # padding_idx=spare: the non-local pairs (about (N-1)/N of them) all point at the spare row and are served
            # from a per-core L1 copy of it instead of DRAM; same values, and post_combine_reduce skips them anyway.
            combined = ttnn.embedding(
                pos_s,
                out_rm,
                padding_idx=consts["spare"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=combined_mem,
            )  # [Cs, k, H]
            combined = ttnn.reshape(combined, (1, 1, e - s, k, H))
            part = ttnn.experimental.deepseek_prefill.post_combine_reduce(
                combined,
                w_s,
                idx_s,
                self.dispatch_table,
                expert_dim=3,
                output_memory_config=part_mem,
            )
            ttnn.deallocate(combined)
            if pos_s is not pos_tm:
                ttnn.deallocate(pos_s)
                ttnn.deallocate(w_s)
                ttnn.deallocate(idx_s)
            pieces.append(ttnn.typecast(part, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            ttnn.deallocate(part)
        ttnn.deallocate(out_rm)
        ttnn.deallocate(pos_tm)
        ttnn.deallocate(w_rm)
        ttnn.deallocate(idx_rm)
        return pieces

    def __call__(self, hidden_states, topk_indices, topk_weights, scatter=False, batch_size=1):
        """hidden_states: [1, 1, S, H] replicated; topk_indices / topk_weights: [S, k] (router top-k path).
        topk_indices are slot ids: id v is expert self.expert_order[v] (the HF id itself when expert_order is None),
        as the MLP's routed router emits them (TopKRouter.with_expert_order).

        Returns the MoE output [1, 1, S, H] (bf8), all-reduced over the mesh row like the dense TP path; with scatter
        (sequence-parallel prefill, S % (32 * N) == 0) only the reduce-scatter half: [1, 1, S / N, H], device d holding
        rows d * S / N onward, bit-identical to those rows of the all-reduce.

        batch_size > 1 (batched prefill): the S rows are B users' tokens stacked by user; the experts are per token,
        and the reduction is batched_prefill.per_user_reduce (each user's rows as in that user's batch-1 call; with
        scatter, device d holds rows d * S / (B N) onward of every user)."""
        seq_len = hidden_states.shape[-2]
        assert self.supports(
            hidden_states.shape
        ), f"routed prefill needs batch 1, S >= {self.min_seq} and S % 64 == 0, got shape {hidden_states.shape}"
        x = hidden_states
        if len(x.shape) != 4:
            x = ttnn.reshape(x, (1, 1, seq_len, self.hidden_size))
        C = min(self.chunk_size, seq_len)
        outputs = []
        for start in range(0, seq_len, C):
            end = min(start + C, seq_len)
            if start == 0 and end == seq_len:
                x_c, i_c, w_c = x, topk_indices, topk_weights
            else:
                x_c = ttnn.slice(x, (0, 0, start, 0), (1, 1, end, self.hidden_size))
                i_c = ttnn.slice(topk_indices, (start, 0), (end, self.top_k))
                w_c = ttnn.slice(topk_weights, (start, 0), (end, self.top_k))
            outputs.extend(self._chunk(x_c, i_c, w_c))
            if x_c is not x:
                ttnn.deallocate(x_c)
                ttnn.deallocate(i_c)
                ttnn.deallocate(w_c)
        if len(outputs) == 1:
            out = outputs[0]
        else:
            out = ttnn.concat(outputs, dim=2)
            for t in outputs:
                ttnn.deallocate(t)
        axis = self.mesh_config.tp_axis
        links = batched_reduce_links(self.ccl_manager, axis)  # 2 on BH 1x4, batch 1 too (bit-exact vs 1 link)
        if batch_size > 1:
            reduced = per_user_reduce(out, batch_size, axis, links, scatter=scatter)
            ttnn.deallocate(out)
            return reduced
        if scatter:
            reduced = ttnn.reduce_scatter(
                out,
                2,
                cluster_axis=self.mesh_config.tp_axis,
                num_links=links,
                topology=ttnn.Topology.Ring,
            )
            ttnn.deallocate(out)
            return reduced
        reduced = ttnn.all_reduce(
            out,
            num_links=links,
            topology=ttnn.Topology.Ring,
            cluster_axis=self.mesh_config.tp_axis,
        )
        ttnn.deallocate(out)
        return ttnn.reshape(reduced, (1, 1, seq_len, self.hidden_size))
