# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN decoder for poolside/Laguna-XS-2.1 on a 1×D Blackhole mesh (D=1, 2, or 4).

Parallelizes the single-chip optimized decoder (``tt/optimized_decoder.py``,
``OptimizedDecoder``) across a 1×D Blackhole mesh with **1D tensor parallelism (TP=D)**
for the dense/attention path and **expert parallelism (EP=D)** for the routed MoE. The
optimized decoder is the single-chip baseline: this class subclasses it and reuses every
optimized helper (precision policy, packed-QKV split, RMSNorm, RoPE, SDPA config, the
DRAM-sharded matmul helper, and the sparse-expert program configs). Only weight placement
(mesh sharding at load time) and the collectives the scheme needs are added here.

Scheme:
  * Residual stream is **replicated** (BF16). Both RMSNorms see the full hidden and are
    exact locally — no distributed norm needed.
  * Attention: WQKV column-parallel (packed, reordered so device d owns Q heads
    [d·lqh:(d+1)·lqh) and KV heads [2d:2d+2)); per-head QK-norm/RoPE/SDPA local; KV cache
    holds the device-local KV heads; softplus gate g_proj column-parallel; WO row-parallel
    → partial → one all_reduce.
  * Dense MLP (layer 0): gate/up column-parallel, down row-parallel → partial → all_reduce.
  * MoE (layers 1-39): router runs replicated; expert weights EP-sharded (256/D experts/device);
    a mesh-sharded selection-matmul turns the replicated 256-wide router output into device d's
    contiguous 64-wide scores/sparsity; ``ttnn.sparse_matmul`` (nnz=None, Blackhole-required)
    over the local experts; shared expert TP; routed_local + shared_partial → one all_reduce.

Collectives: exactly **2 all_reduce / layer** for D>1 (attn out, MLP/MoE out), cluster_axis=1.
Topology and link count are selected with ``TT_LAGUNA_CCL_TOPOLOGY`` and
``TT_LAGUNA_CCL_NUM_LINKS``; D=1 bypasses collectives. Decoder-layer input and output share
the replicated ``[1,1,B,H]`` /
``[1,seq,H]`` layout, so layers stack with no inter-layer reshard.

Public API matches OptimizedDecoder/FunctionalDecoder (``from_state_dict``, ``alloc_kv_cache``,
``make_page_table``, ``prefill_forward``, ``decode_forward``) — the tensors must live on the
1×D mesh (replicated inputs). Enable a matching fabric before opening a multi-device mesh;
fabric must remain disabled for D=1.
"""
from __future__ import annotations

import math
import os
import re

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup

from .optimized_decoder import (
    TILE,
    LayerConfig,
    OptimizedDecoder,
    PrecisionPolicy,
    _cached_device_tensor,
    _dram_weight_memcfg,
    _hf_rope_tables,
    _sparse_pc,
    weight_cache_key,
)
from .prefill_page_table import single_shot_fill_page_table

TOKEN_DISPATCH_ENV = "TT_LAGUNA_MOE_TOKEN_DISPATCH"
MOE_PREFILL_TILE_SPARSE_ENV = "TT_LAGUNA_MOE_PREFILL_TILE_SPARSE"
TOKEN_DISPATCH_BUCKETS = frozenset({1024, 2048, 4096, 8192})
TOKEN_DISPATCH_MOE_LAYERS = frozenset(range(1, 40))
TOKEN_DISPATCH_CHUNK_M_TILES = 16
TOKEN_DISPATCH_METADATA_LEN = 5


def _parse_binary_env(name: str, default: bool = False) -> bool:
    """Parse an experimental runtime switch without accepting typos.

    The token-dispatch path is intentionally fail-closed: only literal ``0``
    and ``1`` are accepted, and an unset variable retains the default-off
    production path.
    """

    value = os.environ.get(name)
    if value is None:
        return default
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be exactly '0' or '1'; got {value!r}")
    return value == "1"


def _token_dispatch_eligibility(
    *,
    enabled: bool,
    layer_idx: int,
    is_moe: bool,
    mesh_devices: int,
    seq_len: int,
    sharded: bool,
    pack_gate_up: bool,
    global_experts: int,
    local_experts: int,
    hidden: int,
    intermediate: int,
    top_k: int,
    activation_dtype,
    moe_ff13_dtype,
    moe_ff2_dtype,
    ccl_dtype,
    moe_fidelity: str,
) -> tuple[bool, str]:
    """Return the narrow, measured token-dispatch serving envelope.

    Everything outside this envelope goes through the established 256-row
    sparse-MoE loop. Keeping this decision device-free makes every layer and
    bucket guard independently testable without opening hardware.
    """

    checks = (
        (enabled, "feature flag is disabled"),
        (layer_idx in TOKEN_DISPATCH_MOE_LAYERS and is_moe, "layer is not a Laguna routed-MoE layer"),
        (mesh_devices == 2, "only the qualified p150x2 mesh is supported"),
        (seq_len in TOKEN_DISPATCH_BUCKETS, "prefill bucket is not supported"),
        (not sharded, "decode/sharded activations are not supported"),
        (pack_gate_up, "stacked packed gate/up weights are required"),
        (
            (global_experts, local_experts, hidden, intermediate, top_k) == (256, 128, 2048, 512, 8),
            "model or expert-partition dimensions do not match Laguna-XS-2.1 p150x2",
        ),
        (activation_dtype == ttnn.bfloat16, "BF16 token-dispatch activations are required"),
        (
            moe_ff13_dtype == ttnn.bfloat4_b and moe_ff2_dtype == ttnn.bfloat4_b,
            "the qualified routed-expert weight policy is BFP4/BFP4",
        ),
        (ccl_dtype == ttnn.bfloat16, "the qualified routed accumulation uses BF16 CCL"),
        (moe_fidelity == "LoFi", "the qualified routed-expert fidelity is LoFi"),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason
    return True, "eligible"


def _cache_layer_identity(layer_idx: int, cache_namespace: str | None = None) -> int | str:
    """Return the cache-layer identity without changing established target keys.

    Draft models can reuse target layer numbers while owning different weights.  An
    explicit namespace keeps those converted tensor caches disjoint; ``None`` retains
    the byte-for-byte target-model key used before this option existed.
    """

    if cache_namespace is None:
        return layer_idx
    if not isinstance(cache_namespace, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cache_namespace):
        raise ValueError(
            "cache_namespace must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'"
        )
    return f"{cache_namespace}_L{layer_idx}"


def _build_expert_row_dispatch(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    expert_start: int,
    expert_count: int,
) -> dict[str, torch.Tensor]:
    """Build the stable token-to-local-expert packing contract on the host.

    This is an executable specification for a future token-sparse device kernel, not a
    runtime fallback: moving router output or activations through the host would erase the
    prefill win.  Entries are ordered by ``(local_expert, token, top-k slot)``; the returned
    offsets therefore delimit the selected input rows for every local expert without ever
    materialising ``[tokens, local_experts, hidden]``.

    Zero-weight routes are omitted because they cannot contribute to the weighted expert
    sum.  That also matches ``sparse_matmul``'s current interpretation of a zero routing
    entry as inactive.
    """
    if topk_indices.shape != topk_weights.shape:
        raise ValueError(
            f"top-k index/weight shapes must match, got {tuple(topk_indices.shape)} and " f"{tuple(topk_weights.shape)}"
        )
    if topk_indices.ndim < 2:
        raise ValueError(f"top-k tensors must have rank >= 2, got rank {topk_indices.ndim}")
    if expert_start < 0 or expert_count <= 0:
        raise ValueError(f"invalid local expert range: start={expert_start}, count={expert_count}")

    top_k = topk_indices.shape[-1]
    num_tokens = topk_indices.numel() // top_k
    indices = topk_indices.reshape(num_tokens, top_k).to(torch.int64)
    weights = topk_weights.reshape(num_tokens, top_k)
    token_indices = torch.arange(num_tokens, device=indices.device, dtype=torch.int64).unsqueeze(1).expand_as(indices)
    slot_indices = torch.arange(top_k, device=indices.device, dtype=torch.int64).unsqueeze(0).expand_as(indices)

    local_experts = indices - expert_start
    selected = (local_experts >= 0) & (local_experts < expert_count) & (weights != 0)
    packed_tokens = token_indices[selected]
    packed_slots = slot_indices[selected]
    packed_experts = local_experts[selected]
    packed_weights = weights[selected]

    # Sorting a unique composite key by expert and token is stable, leaving duplicate
    # (token, expert) routes in top-k-slot order.  Stable ordering is part of the combine
    # contract and makes a future device implementation deterministic.
    if packed_tokens.numel():
        key = (packed_experts * num_tokens + packed_tokens) * top_k + packed_slots
        order = torch.argsort(key, stable=True)
        packed_tokens = packed_tokens[order]
        packed_slots = packed_slots[order]
        packed_experts = packed_experts[order]
        packed_weights = packed_weights[order]

    counts = torch.bincount(packed_experts, minlength=expert_count)
    offsets = torch.empty(expert_count + 1, device=indices.device, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(counts, dim=0)
    return {
        "token_indices": packed_tokens,
        "slot_indices": packed_slots,
        "local_expert_indices": packed_experts,
        "weights": packed_weights,
        "expert_counts": counts,
        "expert_offsets": offsets,
    }


class MultichipDecoder(OptimizedDecoder):
    """Laguna decoder for a 1×D Blackhole mesh: TP=D attention/dense + EP=D routed MoE,
    replicated BF16 residual, and exactly two ``all_reduce``s per layer for D>1. On D=1 the same
    packed production path is used, but both reductions and the expert-selector projection are
    identities.

    ``PACK_GATE_UP`` (default True) enables the decode gate/up matmul-packing optimization: the two
    same-input gate/up projections (routed-expert MoE, dense MLP, shared expert) are concatenated at load
    into one wide weight so a single matmul replaces two dispatches (decode is dispatch/latency-bound). PCC
    is preserved exactly (SiLU is applied after the on-device split), so it is a pure device-time win
    (measured ~−5.4% layer decode). Set ``PACK_GATE_UP=False`` for the unpacked baseline (A/B evidence).
    Previously this lived in a separate ``OptimizedMultichipDecoder`` subclass; folded in here as a flag.
    """

    # Inherited runtime chunk knobs (MOE_PREFILL_CHUNK, PIPE_CHUNK, PREFILL_SDPA_CHUNK).
    PACK_GATE_UP = True
    # Stage 1 of prefill dispatch: bound routed gate/up work to a hardware tile instead
    # of taking one expert union across the full 256-token MoE chunk.  This remains
    # tile-sparse (an expert selected by one row computes the other rows in that tile),
    # not true token packing.  The down projection deliberately keeps the established
    # full-T union because TTNN elementwise ops do not preserve a group×expert batch for
    # input-A-sparse matmul. Keep this opt-in and unqualified; decode never uses it.
    MOE_PREFILL_TILE_SPARSE = _parse_binary_env(MOE_PREFILL_TILE_SPARSE_ENV)
    MOE_PREFILL_TILE_GROUP = TILE

    def __init__(self, cfg, weights, cos_table, sin_table, mesh_device, policy, meta):
        super().__init__(cfg, weights, cos_table, sin_table, mesh_device, policy, meta)
        self.D = meta["mesh_devices"]
        self.global_experts = meta["global_experts"]
        self.local_experts = meta["local_experts"]
        self._token_dispatch_requested = _parse_binary_env(TOKEN_DISPATCH_ENV)
        self._token_dispatch_state = None
        self._token_dispatch_fallback_reason = "feature flag is disabled"
        # On a 1×1 MeshDevice, TTNN's explicit parallel decode-SDPA program is inaccurate once
        # the cache crosses long/non-aligned boundaries (observed PCC ~= 0 at positions 513/2048).
        # The default decode op is accurate at the same positions. Keep the proven explicit k64
        # program on D=2/4; callers may still opt D1 back into a future-qualified program explicitly.
        if self.D == 1 and "TT_LAGUNA_DECODE_SDPA_PC" not in os.environ:
            self._decode_use_sdpa_pc = False
        # TP/EP live on cluster_axis 1 for every supported 1×D profile.
        self.tp_axis = 1
        if self.D == 1:
            self.ccl_topology = None
            self.num_links = 0
        else:
            topology_name = os.environ.get("TT_LAGUNA_CCL_TOPOLOGY", "ring").strip().lower()
            topologies = {"ring": ttnn.Topology.Ring, "linear": ttnn.Topology.Linear}
            if topology_name not in topologies:
                raise ValueError(
                    "TT_LAGUNA_CCL_TOPOLOGY must be 'ring' or 'linear' for a multi-device Laguna profile; "
                    f"got {topology_name!r} for D={self.D}"
                )
            try:
                num_links = int(os.environ.get("TT_LAGUNA_CCL_NUM_LINKS", str(meta.get("num_links", 2))))
            except ValueError as exc:
                raise ValueError("TT_LAGUNA_CCL_NUM_LINKS must be an integer") from exc
            if num_links not in (1, 2):
                raise ValueError(f"TT_LAGUNA_CCL_NUM_LINKS must be 1 or 2 for D={self.D}; got {num_links}")
            self.ccl_topology = topologies[topology_name]
            self.num_links = num_links

    # ---- collective ------------------------------------------------------- #
    def _reduce(self, x):
        """Reduce a row-parallel partial into the replicated residual layout.

        D=1 is an identity. D>1 uses the profile-selected all-reduce and is traceable.

        The all_reduce payload dtype is ``policy.ccl`` (default BF16 == the replicated
        residual, so the path is byte-identical to the BF16 baseline). A lower CCL
        dtype casts the partial before the collective and casts the reduced result back to
        BF16 for the residual add — swept as a yes/no switch in the datatype sweep."""
        if self.D == 1:
            return x
        ccl = getattr(self.policy, "ccl", ttnn.bfloat16)
        if ccl == ttnn.bfloat16:
            return ttnn.all_reduce(x, cluster_axis=self.tp_axis, topology=self.ccl_topology, num_links=self.num_links)
        xin = ttnn.typecast(x, ccl) if x.dtype != ccl else x
        out = ttnn.all_reduce(xin, cluster_axis=self.tp_axis, topology=self.ccl_topology, num_links=self.num_links)
        return ttnn.typecast(out, ttnn.bfloat16) if out.dtype != ttnn.bfloat16 else out

    # ---- construction ------------------------------------------------------ #
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, max_seq_len, policy=None, **kwargs):
        cfg = LayerConfig.from_hf(hf_config, layer_idx)
        policy = policy or PrecisionPolicy()
        dev = mesh_device
        cache_layer_identity = _cache_layer_identity(layer_idx, kwargs.get("cache_namespace"))
        D = mesh_device.get_num_devices()
        mesh_shape = tuple(mesh_device.shape)
        if D not in (1, 2, 4):
            raise ValueError(f"Laguna supports D=1, 2, or 4 devices; got D={D}")
        if mesh_shape != (1, D):
            raise ValueError(f"Laguna requires a 1×D mesh; got shape={mesh_shape} for D={D}")
        dram_cores = mesh_device.dram_grid_size().x

        replicate = ttnn.ReplicateTensorToMesh(dev)

        def shard(dim):
            return ttnn.ShardTensorToMesh(dev, dim=dim)

        def g(name):
            return state_dict[name].float()

        # ---- device-weight disk cache (plan Stage 0; see optimized_decoder) ----
        # The mesh tag encodes the mapping (replicate / shard-dim-N) AND the device
        # count D, so a multichip weight never collides with the single-chip ("sc")
        # copy or a different-D mesh. ``build`` is a thunk so a cache hit skips the
        # source build (critical for the 256-expert stack).
        def rep_tt(name, build, dtype, layout=ttnn.TILE_LAYOUT):
            return _cached_device_tensor(
                build,
                device=dev,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
                cache_key=weight_cache_key(name, cache_layer_identity, f"rep_d{D}"),
            )

        def shard_tt(name, build, dim, dtype, layout=ttnn.TILE_LAYOUT):
            return _cached_device_tensor(
                build,
                device=dev,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard(dim),
                cache_key=weight_cache_key(name, cache_layer_identity, f"sh{dim}_d{D}"),
            )

        # ---- global shapes ----
        H = cfg.hidden
        hd = cfg.head_dim
        GQ = cfg.num_heads  # 48 (full) / 64 (sliding)
        GKV = cfg.num_kv_heads  # 8
        assert GQ % D == 0 and GKV % D == 0, f"heads must divide mesh {D}: Q={GQ} KV={GKV}"
        lqh = GQ // D
        lkv = GKV // D
        local_q_w = lqh * hd
        local_kv_w = lkv * hd
        local_qkv_w = local_q_w + 2 * local_kv_w

        w = {}

        def store_shard(key, w_in_full, k_local, n_local, dtype, mesh_dim):
            """Store a mesh-sharded interleaved copy (``key``) and a mesh-sharded +
            per-device DRAM-width-sharded copy (``key+"_ds"``). ``w_in_full`` is the full
            (unsharded) weight already in ttnn [in, out] orientation; ``mesh_dim`` is the
            tensor axis sharded across the mesh; ``k_local``/``n_local`` are the resulting
            PER-DEVICE [in, out] dims for the DRAM shard spec. Only the interleaved
            copy is disk-cached; the ``_ds`` copy is derived on-device via
            to_memory_config (a width-sharded memory_config is not preserved through
            the ttnn tensor cache — see optimized_decoder._cached_device_tensor)."""
            w[key] = shard_tt(key, lambda: w_in_full, mesh_dim, dtype)
            w[key + "_ds"] = ttnn.to_memory_config(w[key], _dram_weight_memcfg(k_local, n_local, dram_cores))

        # --- packed QKV (column-parallel), reordered into per-device [Q_d|K_d|V_d] blocks --- #
        wq = g("self_attn.q_proj.weight").t().contiguous()  # [H, GQ*hd]
        wk = g("self_attn.k_proj.weight").t().contiguous()  # [H, GKV*hd]
        wv = g("self_attn.v_proj.weight").t().contiguous()  # [H, GKV*hd]
        blocks = []
        for d in range(D):
            blocks.append(wq[:, d * local_q_w : (d + 1) * local_q_w])
            blocks.append(wk[:, d * local_kv_w : (d + 1) * local_kv_w])
            blocks.append(wv[:, d * local_kv_w : (d + 1) * local_kv_w])
        wqkv = torch.cat(blocks, dim=1).contiguous()  # [H, D*local_qkv_w]; shard(dim1) -> per-dev block
        store_shard("wqkv", wqkv, H, local_qkv_w, policy.attn_qkv, mesh_dim=1)
        # WO row-parallel: device d owns contiguous Q-head rows [d*local_q_w:(d+1)*local_q_w] -> plain shard(dim0)
        wo = g("self_attn.o_proj.weight").t().contiguous()  # [GQ*hd, H]
        store_shard("wo", wo, local_q_w, H, policy.attn_o, mesh_dim=0)
        # softplus gate g_proj [H, GQ] column-parallel -> device d gate for its Q heads
        w["wg"] = shard_tt("wg", lambda: g("self_attn.g_proj.weight").t().contiguous(), 1, policy.attn_gate)
        w["q_norm"] = rep_tt("q_norm", lambda: g("self_attn.q_norm.weight").reshape(1, 1, 1, hd), policy.qk_norm)
        w["k_norm"] = rep_tt("k_norm", lambda: g("self_attn.k_norm.weight").reshape(1, 1, 1, hd), policy.qk_norm)
        w["input_ln"] = rep_tt("input_ln", lambda: g("input_layernorm.weight").reshape(1, 1, 1, H), ttnn.bfloat16)
        w["post_ln"] = rep_tt(
            "post_ln", lambda: g("post_attention_layernorm.weight").reshape(1, 1, 1, H), ttnn.bfloat16
        )

        global_experts = cfg.num_experts
        if cfg.is_moe and global_experts % D != 0:
            raise ValueError(f"experts must divide mesh {D}: experts={global_experts}")
        local_experts = global_experts // D if cfg.is_moe else 0

        if cfg.is_moe:
            E, I = global_experts, cfg.moe_intermediate
            # router replicated (full 256-wide logits/top-k on every device)
            w["gate_w"] = rep_tt("gate_w", lambda: g("mlp.gate.weight").t().contiguous(), policy.router)
            w["e_bias"] = rep_tt(
                "e_bias", lambda: g("mlp.experts.e_score_correction_bias").reshape(1, 1, 1, E), ttnn.bfloat16
            )
            # For D>1, a mesh-sharded identity selects device d's contiguous local expert scores.
            # D=1 already owns every score, so avoid the large identity weight and selector matmul.
            if D > 1:
                w["ep_sel"] = shard_tt("ep_sel", lambda: torch.eye(E).reshape(1, 1, E, E), 3, ttnn.bfloat16)

            # expert weights EP-sharded on the expert dim (dim1): device d holds experts [64d:64d+64].
            # The 256-expert torch.stack is the dominant boot cost, so build it lazily
            # inside the cache thunk — a HIT skips the stack (and the safetensors
            # upcast) entirely, not just the tilize/shard.
            def _stack(proj):
                return torch.stack([g(f"mlp.experts.{i}.{proj}.weight") for i in range(E)])

            w["exp_gate"] = shard_tt(
                "exp_gate", lambda: _stack("gate_proj").transpose(1, 2).reshape(1, E, H, I), 1, policy.moe_ff13
            )
            w["exp_up"] = shard_tt(
                "exp_up", lambda: _stack("up_proj").transpose(1, 2).reshape(1, E, H, I), 1, policy.moe_ff13
            )
            w["exp_down"] = shard_tt(
                "exp_down", lambda: _stack("down_proj").transpose(1, 2).reshape(1, E, I, H), 1, policy.moe_ff2
            )
            # shared expert TP (col/col/row); down produces a partial folded into the routed all_reduce
            gsh = cfg.shared_intermediate
            if gsh % D != 0:
                raise ValueError(f"shared intermediate must divide mesh {D}: shared_intermediate={gsh}")
            local_sh = gsh // D
            store_shard(
                "sh_gate",
                g("mlp.shared_expert.gate_proj.weight").t().contiguous(),
                H,
                local_sh,
                policy.shared_ff13,
                mesh_dim=1,
            )
            store_shard(
                "sh_up",
                g("mlp.shared_expert.up_proj.weight").t().contiguous(),
                H,
                local_sh,
                policy.shared_ff13,
                mesh_dim=1,
            )
            store_shard(
                "sh_down",
                g("mlp.shared_expert.down_proj.weight").t().contiguous(),
                local_sh,
                H,
                policy.shared_ff2,
                mesh_dim=0,
            )
            cfg.shared_intermediate = local_sh  # local for _glu_mlp
        else:
            II = cfg.intermediate
            if II % D != 0:
                raise ValueError(f"dense intermediate must divide mesh {D}: intermediate={II}")
            local_II = II // D
            store_shard(
                "mlp_gate", g("mlp.gate_proj.weight").t().contiguous(), H, local_II, policy.dense_ff13, mesh_dim=1
            )
            store_shard("mlp_up", g("mlp.up_proj.weight").t().contiguous(), H, local_II, policy.dense_ff13, mesh_dim=1)
            store_shard(
                "mlp_down", g("mlp.down_proj.weight").t().contiguous(), local_II, H, policy.dense_ff2, mesh_dim=0
            )
            cfg.intermediate = local_II  # local for _glu_mlp

        # build the (cos,sin) RoPE tables once PER ATTENTION KIND, not once per layer then
        # dedup. LagunaModel.from_pretrained threads a shared ``rope_tables`` dict; the first layer of
        # each kind builds+caches, later layers of that kind reuse the SAME device tensors. This removes
        # the ~4.7 GB transient peak (all 40 pairs alive before _dedup_rope) and cuts the per-layer
        # trust-remote-code load (get_class_from_dynamic_module runs twice — one per kind — not 40x).
        # Tables depend only on (attention_type, max_seq_len, config), so sharing is exact / bit-identical.
        rope_tables = kwargs.get("rope_tables")
        kind = cfg.attention_type
        if rope_tables is not None and kind in rope_tables:
            cos_2d, sin_2d = rope_tables[kind]
        else:
            cos, sin = _hf_rope_tables(hf_config, kind, max_seq_len)
            cos_2d = ttnn.from_torch(
                cos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=replicate
            )
            sin_2d = ttnn.from_torch(
                sin, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, mesh_mapper=replicate
            )
            if rope_tables is not None:
                rope_tables[kind] = (cos_2d, sin_2d)

        # ---- mutate cfg to LOCAL head counts so all inherited attention code runs per-device ----
        cfg.num_heads = lqh
        cfg.num_kv_heads = lkv
        cfg.num_kv_groups = lqh // lkv

        meta = {
            "layer_idx": layer_idx,
            "dram_cores": dram_cores,
            "q_w": local_q_w,
            "kv_w": local_kv_w,
            "qkv_w": local_qkv_w,
            "mesh_devices": D,
            "global_experts": global_experts,
            "local_experts": local_experts,
            "num_links": 2,
        }
        dec = cls(cfg, w, cos_2d, sin_2d, dev, policy, meta)
        if cls.PACK_GATE_UP:
            dec._pack_gate_up()
        dec._setup_token_dispatch()
        return dec

    def _pack_gate_up(self):
        """Concat the same-input gate/up weights (interleaved + DRAM-width-sharded ``_ds`` copies) into one
        packed weight and free the separate copies. Pure on-device concat/reshard of the already-loaded
        (and disk-cached) tensors, so it inherits the weight cache. The packed copies are intentionally NOT
        separately disk-cached: they are cheap to derive on device, and the ttnn tensor cache cannot
        round-trip the ``_ds`` copy's custom width-sharded memory_config anyway."""
        w = self.w
        dram_cores = self.meta["dram_cores"]
        H = self.cfg.hidden

        def pack_dram_pair(gk, uk, out, n_local):
            """Concat two DRAM-width-sharded gate/up weights (interleaved + _ds copies) into one
            [H, 2*n_local] weight; rebuild the DRAM shard spec for the doubled width; free originals."""
            il = ttnn.concat([w[gk], w[uk]], dim=-1)  # [H, 2*n_local] interleaved DRAM
            ds = ttnn.to_memory_config(il, _dram_weight_memcfg(H, 2 * n_local, dram_cores))
            w[out] = il
            w[out + "_ds"] = ds
            for k in (gk, uk):
                ttnn.deallocate(w[k])
                ttnn.deallocate(w[k + "_ds"])
                del w[k]
                del w[k + "_ds"]

        if self.cfg.is_moe:
            # routed experts: interleaved BFP4 [1,64,H,512] each -> [1,64,H,1024]
            w["exp_gate_up"] = ttnn.concat([w["exp_gate"], w["exp_up"]], dim=-1)
            ttnn.deallocate(w["exp_gate"])
            ttnn.deallocate(w["exp_up"])
            del w["exp_gate"]
            del w["exp_up"]
            pack_dram_pair("sh_gate", "sh_up", "sh_gate_up", self.cfg.shared_intermediate)
        else:
            pack_dram_pair("mlp_gate", "mlp_up", "mlp_gate_up", self.cfg.intermediate)

    def _token_dispatch_guard(self, seq_len: int, sharded: bool) -> tuple[bool, str]:
        """Check the qualified path contract before every dispatch attempt."""

        eligible, reason = _token_dispatch_eligibility(
            enabled=self._token_dispatch_requested,
            layer_idx=self.meta["layer_idx"],
            is_moe=self.cfg.is_moe,
            mesh_devices=self.D,
            seq_len=seq_len,
            sharded=sharded,
            pack_gate_up=self.PACK_GATE_UP,
            global_experts=self.global_experts,
            local_experts=self.local_experts,
            hidden=self.cfg.hidden,
            intermediate=self.cfg.moe_intermediate,
            top_k=self.cfg.top_k,
            activation_dtype=self.policy.activation,
            moe_ff13_dtype=self.policy.moe_ff13,
            moe_ff2_dtype=self.policy.moe_ff2,
            ccl_dtype=self.policy.ccl,
            moe_fidelity=self.policy.fid_moe,
        )
        if eligible and self._token_dispatch_state is None:
            return False, self._token_dispatch_fallback_reason
        return eligible, reason

    def _setup_token_dispatch(self) -> None:
        """Build the small local-only routing state for the opt-in D2 path.

        No state is allocated unless the flag and static model policy match.
        Bucket-specific dispatch/combine wrappers contain configuration only;
        their large buffers are allocated per invocation by the TTNN ops.
        """

        eligible, reason = _token_dispatch_eligibility(
            enabled=self._token_dispatch_requested,
            layer_idx=self.meta["layer_idx"],
            is_moe=self.cfg.is_moe,
            mesh_devices=self.D,
            seq_len=min(TOKEN_DISPATCH_BUCKETS),
            sharded=False,
            pack_gate_up=self.PACK_GATE_UP,
            global_experts=self.global_experts,
            local_experts=self.local_experts,
            hidden=self.cfg.hidden,
            intermediate=self.cfg.moe_intermediate,
            top_k=self.cfg.top_k,
            activation_dtype=self.policy.activation,
            moe_ff13_dtype=self.policy.moe_ff13,
            moe_ff2_dtype=self.policy.moe_ff2,
            ccl_dtype=self.policy.ccl,
            moe_fidelity=self.policy.fid_moe,
        )
        if not eligible:
            self._token_dispatch_fallback_reason = reason
            return
        if "exp_gate_up" not in self.w or "exp_down" not in self.w:
            self._token_dispatch_fallback_reason = "stacked routed-expert tensors are unavailable"
            return

        dispatch_group_size = 1
        num_dispatch_groups = self.D
        dispatch_table_host = ExpertMapping.create_dispatch_table(
            num_routed_experts=self.global_experts,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
        )
        routing_setup = TtMoERoutingSetup(
            mesh_device=self.device,
            expert_dispatch_table=dispatch_table_host,
            num_links=1,
            experts_per_chip=self.local_experts,
        )
        dispatch_table = TtDispatchModule.shard_expert_dispatch_table(self.device, dispatch_table_host, dispatch_axis=0)
        global_expert_idx = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=self.local_experts,
                dispatch_group_size=dispatch_group_size,
                num_dispatch_groups=num_dispatch_groups,
            ),
            mesh_mapper=get_ep_mesh_mapper(self.device),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            dtype=ttnn.uint32,
        )
        global_expert_idx = ttnn.squeeze(ttnn.squeeze(global_expert_idx, 0), 0)

        bucket_modules = {}
        for seq_len in sorted(TOKEN_DISPATCH_BUCKETS):
            # Worst case: every one of T*K routes is local, plus at most 31
            # padding rows for each local expert's tile-aligned region.
            max_dispatch_rows = seq_len * self.cfg.top_k + (TILE - 1) * self.local_experts
            bucket_modules[seq_len] = {
                "dispatch": TtDispatchModule(
                    mesh_device=self.device,
                    dispatch_group_size=dispatch_group_size,
                    experts_per_chip=self.local_experts,
                    num_routed_experts=self.global_experts,
                    num_experts_per_tok=self.cfg.top_k,
                    metadata_len=TOKEN_DISPATCH_METADATA_LEN,
                    max_dispatch_buffer_token_size=max_dispatch_rows,
                    seq_len_per_chip=seq_len,
                    emb_dim=self.cfg.hidden,
                    cluster_axis=0,
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                ),
                "combine": TtCombineModule(
                    mesh_device=self.device,
                    dispatch_group_size=dispatch_group_size,
                    num_dispatch_groups=num_dispatch_groups,
                    experts_per_chip=self.local_experts,
                    num_experts_per_tok=self.cfg.top_k,
                    seq_len_per_chip=seq_len,
                    cluster_axis=0,
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    init_zeros=True,
                ),
                "max_dispatch_rows": max_dispatch_rows,
            }

        self._token_dispatch_state = {
            "routing_setup": routing_setup,
            "dispatch_table": dispatch_table,
            "global_expert_idx": global_expert_idx,
            "buckets": bucket_modules,
        }
        self._token_dispatch_fallback_reason = "eligible"

    # ---- KV cache / page table (replicated init; each device holds its own local KV heads) ---- #
    def alloc_kv_cache(self, max_users, max_seq_len, block_size=32, dtype=None):
        dtype = dtype or self.policy.kv_cache
        blocks_per_user = int(math.ceil(max_seq_len / block_size))
        max_num_blocks = blocks_per_user * max_users
        shape = (max_num_blocks, self.cfg.num_kv_heads, block_size, self.cfg.head_dim)  # local KV heads
        k = ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        v = ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        return {"k": k, "v": v, "block_size": block_size, "blocks_per_user": blocks_per_user, "dtype": dtype}

    def make_page_table(self, num_users, blocks_per_user):
        pt = torch.arange(num_users * blocks_per_user, dtype=torch.int32).reshape(num_users, blocks_per_user)
        return ttnn.from_torch(
            pt,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )

    # ---- MoE (Expert Parallel) --------------------------------------------- #
    def _token_dispatch_router(self, ln_flat, seq_len: int):
        """Run the established 256-row router programs, then concatenate routes.

        The 8192-token qualification showed that one whole-M router changes
        roughly 18% of top-k slots. Keeping the existing slice boundaries is
        therefore an accuracy contract, even though dispatch/experts operate on
        the whole outer prefill bucket.
        """

        weights = []
        indices = []
        cfg = self.cfg
        for start in range(0, seq_len, self.MOE_PREFILL_CHUNK):
            end = min(start + self.MOE_PREFILL_CHUNK, seq_len)
            chunk = ttnn.slice(ln_flat, [0, 0, start, 0], [1, 1, end, cfg.hidden])
            logits = ttnn.linear(chunk, self.w["gate_w"], compute_kernel_config=self._ck_router)
            scores = ttnn.sigmoid(logits)
            selected_scores = ttnn.add(scores, self.w["e_bias"])
            _, idx = ttnn.topk(ttnn.typecast(selected_scores, ttnn.bfloat16), k=cfg.top_k, dim=-1, sorted=True)
            wsel = ttnn.gather(scores, dim=3, index=idx)
            if cfg.norm_topk_prob:
                wsel = ttnn.div(wsel, ttnn.sum(wsel, dim=3, keepdim=True))
            if cfg.routed_scaling != 1.0:
                wsel = ttnn.multiply(wsel, cfg.routed_scaling)
            weights.append(ttnn.reshape(ttnn.to_layout(wsel, ttnn.ROW_MAJOR_LAYOUT), (1, end - start, cfg.top_k)))
            indices.append(ttnn.reshape(ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT), (1, end - start, cfg.top_k)))
        return ttnn.concat(weights, dim=1), ttnn.concat(indices, dim=1)

    def _token_dispatch_shared(self, ln_flat, seq_len: int):
        """Preserve the established 256-row shared-expert numerics."""

        partials = []
        cfg = self.cfg
        for start in range(0, seq_len, self.MOE_PREFILL_CHUNK):
            end = min(start + self.MOE_PREFILL_CHUNK, seq_len)
            chunk = ttnn.slice(ln_flat, [0, 0, start, 0], [1, 1, end, cfg.hidden])
            partials.append(
                self._glu_mlp(
                    chunk,
                    "sh",
                    cfg.hidden,
                    cfg.shared_intermediate,
                    self._ck_shared,
                    False,
                )
            )
        return ttnn.concat(partials, dim=2)

    def _moe_token_dispatch(self, ln_flat, seq_len: int):
        """Exact local-only token-to-expert prefill dispatch on p150x2.

        Each column is a dispatch group of size one: it packs only routes for
        that ASIC's contiguous 128 experts, runs compact FFNs directly from the
        existing stacked BFP4 tensors, combines back to top-k slots, applies the
        original router weights in the fused reduction, then performs the one
        existing EP all-reduce.
        """

        state = self._token_dispatch_state
        bucket = state["buckets"][seq_len]
        cfg = self.cfg

        weights, indices = self._token_dispatch_router(ln_flat, seq_len)
        offsets, counts, region_offsets, histogram = state["routing_setup"](
            ttnn_top_k_experts_indices=indices,
            num_routed_experts=self.global_experts,
            seq_len_per_chip=seq_len,
            num_experts_per_tok=cfg.top_k,
        )
        ttnn.deallocate(histogram)

        dispatch_input = ttnn.reshape(ln_flat, (1, seq_len, cfg.hidden))
        dispatched, metadata = bucket["dispatch"](
            dispatch_input,
            weights,
            indices,
            offsets,
            state["dispatch_table"],
        )
        dispatched_tiled = ttnn.to_layout(
            ttnn.squeeze(ttnn.squeeze(dispatched, 0), 0),
            ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(dispatched)

        expert_outputs = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe_stacked(
            dispatched_tiled,
            region_offsets,
            counts,
            state["global_expert_idx"],
            self.w["exp_gate_up"],
            self.w["exp_down"],
            seq_len,
            compute_kernel_config=self._ck_moe,
            chunk_m_tiles_override=TOKEN_DISPATCH_CHUNK_M_TILES,
        )
        ttnn.deallocate(dispatched_tiled)

        combined_slots = bucket["combine"](
            ttnn.unsqueeze(ttnn.unsqueeze(expert_outputs, 0), 0),
            metadata,
            counts,
            region_offsets,
        )
        ttnn.deallocate(expert_outputs)
        ttnn.deallocate(metadata)
        ttnn.deallocate(offsets)
        ttnn.deallocate(counts)
        ttnn.deallocate(region_offsets)

        weights_5d = ttnn.unsqueeze(ttnn.unsqueeze(weights, dim=-1), dim=0)
        routed_local = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            combined_slots,
            weights_5d,
            indices,
            state["dispatch_table"],
            expert_dim=3,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        routed_local = ttnn.reshape(routed_local, (1, 1, seq_len, cfg.hidden))
        ttnn.deallocate(combined_slots)
        ttnn.deallocate(weights)
        ttnn.deallocate(indices)

        shared_partial = self._token_dispatch_shared(ln_flat, seq_len)
        local_output = ttnn.add(routed_local, ttnn.reshape(shared_partial, (1, 1, seq_len, cfg.hidden)))
        ttnn.deallocate(routed_local)
        ttnn.deallocate(shared_partial)
        output = self._reduce(local_output)
        ttnn.deallocate(local_output)
        return output

    def _moe(self, ln_flat, m, sharded):
        """Routed-expert MoE (EP=4). With PACK_GATE_UP, one packed gate+up sparse_matmul (split + SwiGLU on
        device); else the unpacked two-matmul baseline (``_moe_separate``). Both end in a ring all_reduce."""
        if not self.PACK_GATE_UP:
            return self._moe_separate(ln_flat, m, sharded)
        cfg = self.cfg
        LE = self.local_experts
        H, I, K = cfg.hidden, cfg.moe_intermediate, cfg.top_k
        T = ln_flat.shape[2]
        logits = ttnn.linear(ln_flat, self.w["gate_w"], compute_kernel_config=self._ck_router)
        scores = ttnn.sigmoid(logits)
        sel = ttnn.add(scores, self.w["e_bias"])
        _, idx = ttnn.topk(ttnn.typecast(sel, ttnn.bfloat16), k=K, dim=-1, sorted=True)
        wsel = ttnn.gather(scores, dim=3, index=idx)
        if cfg.norm_topk_prob:
            wsel = ttnn.div(wsel, ttnn.sum(wsel, dim=3, keepdim=True))
        if cfg.routed_scaling != 1.0:
            wsel = ttnn.multiply(wsel, cfg.routed_scaling)
        dense = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=idx, src=wsel)
        dense_local = (
            dense if self.D == 1 else ttnn.matmul(dense, self.w["ep_sel"], compute_kernel_config=self._ck_router)
        )
        # Preserve the established full-T union for the down projection.  The
        # optional grouped union applies only to gate/up below.
        full_union = ttnn.sum(dense_local, dim=2, keepdim=True)
        group = self.MOE_PREFILL_TILE_GROUP
        tile_sparse = self.MOE_PREFILL_TILE_SPARSE and not sharded and T >= group and T % group == 0
        if tile_sparse:
            # A sparse_matmul sparsity entry selects one complete M tile, so use one
            # routing mask per 32 token rows.  The old mask reduced all T rows and
            # caused an expert selected once to run over every row in the MoE chunk.
            # Keeping group == the input tile height avoids padding extra compute.
            groups = T // group
            dense_grouped = ttnn.reshape(dense_local, (1, groups, group, LE))
            union = ttnn.sum(dense_grouped, dim=2, keepdim=True)
            a = ttnn.reshape(ln_flat, (1, groups, group, H))
            matmul_m = group
        else:
            groups = 1
            union = full_union
            a = ttnn.reshape(ln_flat, (1, 1, T, H))
            matmul_m = T
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        down_sparsity = ttnn.to_layout(full_union, ttnn.ROW_MAJOR_LAYOUT) if tile_sparse else sparsity
        moe_mem = ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG
        otile = ttnn.Tile([TILE, TILE])
        gu_pc = _sparse_pc(2 * I, matmul_m, H)  # packed gate+up, N = 2*I
        gu = ttnn.sparse_matmul(
            a,
            self.w["exp_gate_up"],
            sparsity=sparsity,
            program_config=gu_pc,
            compute_kernel_config=self._ck_moe,
            memory_config=moe_mem,
            output_tile=otile,
        )
        if tile_sparse:
            # sparse_matmul orders output batches as A-batches then B-batches:
            # [group, local_expert, row, channel]. Restore expert-major token
            # order after the grouped gate/up projection. The down projection
            # then follows the established full-T union path.
            gu = ttnn.reshape(gu, (groups, LE, group, 2 * I))
            gu = ttnn.permute(gu, (1, 0, 2, 3))
            gu = ttnn.reshape(gu, (1, LE, T, 2 * I))
        else:
            gu = ttnn.reshape(gu, (1, LE, T, 2 * I))
        gate_o = ttnn.slice(gu, [0, 0, 0, 0], [1, LE, T, I])
        up_o = ttnn.slice(gu, [0, 0, 0, I], [1, LE, T, 2 * I])
        glu = ttnn.mul(ttnn.silu(gate_o), up_o)
        dn_pc = _sparse_pc(H, T, I)
        down_o = ttnn.sparse_matmul(
            glu,
            self.w["exp_down"],
            sparsity=down_sparsity,
            is_input_a_sparse=True,
            program_config=dn_pc,
            compute_kernel_config=self._ck_moe,
            memory_config=moe_mem,
            output_tile=otile,
        )
        wv = ttnn.reshape(dense_local, (1, T, LE))
        wv = ttnn.permute(wv, (0, 2, 1))
        wv = ttnn.reshape(wv, (1, LE, T, 1))
        weighted = ttnn.mul(down_o, wv)  # [1, LE, T, H]
        if self._use_fused_reduce:  # gated (TT_LAGUNA_FUSED_REDUCE=1); PCC-validate before enabling
            (reduced,) = ttnn.experimental.deepseek_moe_fast_reduce_nc(
                weighted, dim=1, split_size=H, output_memory_config=moe_mem, compute_kernel_config=self._ck_moe
            )
            routed_local = ttnn.reshape(reduced, (1, 1, T, H))
        else:
            routed_local = ttnn.reshape(ttnn.sum(weighted, dim=1), (1, 1, T, H))
        shared_partial = self._glu_mlp(ln_flat, "sh", cfg.hidden, cfg.shared_intermediate, self._ck_shared, sharded)
        combined = ttnn.add(routed_local, ttnn.reshape(shared_partial, (1, 1, T, H)))
        return self._reduce(combined)

    # ---- MoE baseline (unpacked gate/up: two separate sparse_matmuls) ------ #
    def _moe_separate(self, ln_flat, m, sharded):
        cfg = self.cfg
        GE, LE = self.global_experts, self.local_experts
        H, I, K = cfg.hidden, cfg.moe_intermediate, cfg.top_k
        T = ln_flat.shape[2]
        # --- router (replicated: identical full-256 selection on every device) ---
        logits = ttnn.linear(ln_flat, self.w["gate_w"], compute_kernel_config=self._ck_router)  # [1,1,T,GE]
        scores = ttnn.sigmoid(logits)
        sel = ttnn.add(scores, self.w["e_bias"])
        _, idx = ttnn.topk(ttnn.typecast(sel, ttnn.bfloat16), k=K, dim=-1, sorted=True)
        wsel = ttnn.gather(scores, dim=3, index=idx)
        if cfg.norm_topk_prob:
            wsum = ttnn.sum(wsel, dim=3, keepdim=True)
            wsel = ttnn.div(wsel, wsum)
        if cfg.routed_scaling != 1.0:
            wsel = ttnn.multiply(wsel, cfg.routed_scaling)
        dense = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=idx, src=wsel)  # [1,1,T,GE] replicated
        # --- EP selection: replicated 256-wide -> device-local contiguous 64-wide (SPMD-safe matmul) ---
        dense_local = (
            dense if self.D == 1 else ttnn.matmul(dense, self.w["ep_sel"], compute_kernel_config=self._ck_router)
        )  # [1,1,T,LE]
        union = ttnn.sum(dense_local, dim=2, keepdim=True)  # [1,1,1,LE]
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        a = ttnn.reshape(ln_flat, (1, 1, T, H))
        moe_mem = ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG
        otile = ttnn.Tile([TILE, TILE])
        gu_pc = _sparse_pc(I, T, H)
        gate_o = ttnn.sparse_matmul(
            a,
            self.w["exp_gate"],
            sparsity=sparsity,
            program_config=gu_pc,
            compute_kernel_config=self._ck_moe,
            memory_config=moe_mem,
            output_tile=otile,
        )
        up_o = ttnn.sparse_matmul(
            a,
            self.w["exp_up"],
            sparsity=sparsity,
            program_config=gu_pc,
            compute_kernel_config=self._ck_moe,
            memory_config=moe_mem,
            output_tile=otile,
        )
        gate_o = ttnn.reshape(gate_o, (1, LE, T, I))
        up_o = ttnn.reshape(up_o, (1, LE, T, I))
        glu = ttnn.mul(ttnn.silu(gate_o), up_o)
        dn_pc = _sparse_pc(H, T, I)
        down_o = ttnn.sparse_matmul(
            glu,
            self.w["exp_down"],
            sparsity=sparsity,
            is_input_a_sparse=True,
            program_config=dn_pc,
            compute_kernel_config=self._ck_moe,
            memory_config=moe_mem,
            output_tile=otile,
        )  # [1,LE,T,H]
        wv = ttnn.reshape(dense_local, (1, T, LE))
        wv = ttnn.permute(wv, (0, 2, 1))
        wv = ttnn.reshape(wv, (1, LE, T, 1))
        weighted = ttnn.mul(down_o, wv)
        routed_local = ttnn.reshape(ttnn.sum(weighted, dim=1), (1, 1, T, H))  # partial (local experts)
        shared_partial = self._glu_mlp(ln_flat, "sh", cfg.hidden, cfg.shared_intermediate, self._ck_shared, sharded)
        combined = ttnn.add(routed_local, ttnn.reshape(shared_partial, (1, 1, T, H)))
        return self._reduce(combined)  # all_reduce -> total routed + shared (replicated)

    # ---- dense / shared SwiGLU MLP: one packed gate+up matmul, split, SwiGLU (else unpacked base) ---- #
    def _glu_mlp(self, x, key, H, I, ck, sharded):
        if not self.PACK_GATE_UP:
            return super()._glu_mlp(x, key, H, I, ck, sharded)
        guk, dk = {"mlp": ("mlp_gate_up", "mlp_down"), "sh": ("sh_gate_up", "sh_down")}[key]
        w = self.w
        if sharded and self.use_dram_sharded:
            gu = self._dram_mm(x, w[guk], w[guk + "_ds"], H, 2 * I, ck)  # [.,.,M,2I] width-sharded
            gu = ttnn.sharded_to_interleaved(gu, ttnn.L1_MEMORY_CONFIG)
            shp = list(gu.shape)
            g = ttnn.slice(gu, [0] * len(shp), shp[:-1] + [I])
            u = ttnn.slice(gu, [0] * (len(shp) - 1) + [I], shp[:-1] + [2 * I])
            gg = ttnn.mul(ttnn.silu(g), u)
            out = self._dram_mm(gg, w[dk], w[dk + "_ds"], I, H, ck)
            return ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
        # prefill: interleaved packed gate+up linear, split, SwiGLU, down
        gu = ttnn.linear(x, w[guk], compute_kernel_config=ck)  # [.,seq,2I]
        shp = list(gu.shape)
        g = ttnn.slice(gu, [0] * len(shp), shp[:-1] + [I])
        u = ttnn.slice(gu, [0] * (len(shp) - 1) + [I], shp[:-1] + [2 * I])
        gg = ttnn.mul(ttnn.silu(g), u)
        return ttnn.linear(gg, w[dk], compute_kernel_config=ck)

    # ---- dense/shared MLP: gate/up column, down row -> caller reduces ------ #
    def _mlp(self, ln, T, sharded):
        cfg = self.cfg
        ln_flat = ttnn.reshape(ln, (1, 1, T, cfg.hidden))
        if not cfg.is_moe:
            partial = self._glu_mlp(ln_flat, "mlp", cfg.hidden, cfg.intermediate, self._ck_dense, sharded)
            return self._reduce(ttnn.reshape(partial, (1, 1, T, cfg.hidden)))
        use_token_dispatch, reason = self._token_dispatch_guard(T, sharded)
        if use_token_dispatch:
            return self._moe_token_dispatch(ln_flat, T)
        if self._token_dispatch_requested:
            self._token_dispatch_fallback_reason = reason
        if T <= self.MOE_PREFILL_CHUNK:
            return self._moe(ln_flat, T, sharded)
        outs = []
        for s in range(0, T, self.MOE_PREFILL_CHUNK):
            e = min(s + self.MOE_PREFILL_CHUNK, T)
            chunk = ttnn.slice(ln_flat, [0, 0, s, 0], [1, 1, e, cfg.hidden])
            outs.append(self._moe(chunk, e - s, sharded))
        return ttnn.concat(outs, dim=2)

    # ---- prefill (single shot): reuse optimized body + all_reduce after WO --- #
    def prefill_forward(
        self,
        x_BSH,
        kv_cache,
        page_table,
        *,
        fill_page_table=None,
        fill_page_table_base_pos=0,
        user_id=0,
        start_pos=0,
        rope_mats=None,
        runtime_offsets=None,
    ):
        fill_page_table = page_table if fill_page_table is None else fill_page_table
        seq = x_BSH.shape[-2]
        # Runtime offsets identify the qualified D2 path (cold pipeline or a resumed suffix).
        # Route even a small resumed bucket through the pipeline implementation so cached and cold
        # prefixes use the same flexible chunked-SDPA kernel family. In particular, this avoids the
        # single-shot q32/k64 program diverging from the pipeline's q32/k32 numerics after a cache hit.
        # Cold single-shot and all D1/D4 calls have runtime_offsets=None and remain byte-for-byte on
        # the established local-SDPA branch below.
        if runtime_offsets is not None or seq > self.PIPE_CHUNK:
            return self._prefill_pipelined(
                x_BSH,
                kv_cache,
                page_table,
                fill_page_table,
                user_id,
                start_pos,
                fill_page_table_base_pos=fill_page_table_base_pos,
                rope_mats=rope_mats,
                runtime_offsets=runtime_offsets,
            )
        cfg = self.cfg
        residual = x_BSH
        ln = self._rms(x_BSH, self.w["input_ln"])
        q, k, v = self._qkv_roped(ln, seq, start_pos, rope=rope_mats)
        cdt = kv_cache["dtype"]
        # paged_fill_cache writes k/v into the blocks listed in its dedicated table
        # starting at its first column. Bucket-padding entries are -1 (skip); the
        # attention table remains scratch-mapped and is never passed to the writer.
        # The fill table must be offset to the block
        # containing start_pos. Cold prefill (start_pos=0) => col0=0 (unchanged);
        # a prefix-cache suffix prefill (start_pos>0) must NOT write at position 0
        # (that would clobber the cached prefix's first blocks). Mirrors the
        # per-chunk slicing in _prefill_pipelined.
        fill_pt = single_shot_fill_page_table(
            fill_page_table,
            start_pos=start_pos,
            seq_len=seq,
            block_size=kv_cache["block_size"],
            fill_page_table_base_pos=fill_page_table_base_pos,
        )
        ttnn.experimental.paged_fill_cache(kv_cache["k"], self._cast_fill(k, cdt), fill_pt, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(kv_cache["v"], self._cast_fill(v, cdt), fill_pt, batch_idx=user_id)
        chunk_start_idx_tensor = None
        if runtime_offsets is not None:
            if tuple(runtime_offsets.chunk_lengths) != (int(seq),):
                raise ValueError(f"single-shot runtime chunks {runtime_offsets.chunk_lengths} do not match seq {seq}")
            chunk_start_idx_tensor = runtime_offsets.chunk_start_idxs[0]
        attn = self._prefill_attention(
            q,
            k,
            v,
            kv_cache,
            page_table,
            user_id,
            start_pos,
            seq,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
        )
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, (1, seq, cfg.num_heads * cfg.head_dim))
        attn = self._gate(attn, ln)
        o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._ck_o)  # row-parallel partial
        o = self._reduce(o)
        h = ttnn.add(residual, o)
        ln2 = self._rms(h, self.w["post_ln"])
        mlp_out = ttnn.reshape(self._mlp(ln2, seq, sharded=False), (1, seq, cfg.hidden))
        return ttnn.add(h, mlp_out)

    def _prefill_pipelined(
        self,
        x_BSH,
        kv_cache,
        page_table,
        fill_page_table,
        user_id,
        start_pos,
        *,
        fill_page_table_base_pos=0,
        rope_mats=None,
        runtime_offsets=None,
    ):
        cfg = self.cfg
        seq = x_BSH.shape[-2]
        bs = kv_cache["block_size"]
        cdt = kv_cache["dtype"]
        CH = (self._prefill_pipe_chunk // bs) * bs  # env-gated outer chunk (TT_LAGUNA_PREFILL_FAST)
        win = cfg.sliding_window
        outs = []
        expected_chunks = tuple(min(CH, seq - c) for c in range(0, seq, CH))
        if runtime_offsets is not None and tuple(runtime_offsets.chunk_lengths) != expected_chunks:
            raise ValueError(f"pipelined runtime chunks {runtime_offsets.chunk_lengths} do not match {expected_chunks}")
        if rope_mats is not None and len(rope_mats) != len(expected_chunks):
            raise ValueError(f"pipelined RoPE has {len(rope_mats)} chunks, expected {len(expected_chunks)}")
        fill_base = int(fill_page_table_base_pos)
        if int(start_pos) < fill_base or (int(start_pos) - fill_base) % bs:
            raise ValueError(f"prefill start {start_pos} and fill page-table base {fill_base} are not block aligned")
        for chunk_idx, c in enumerate(range(0, seq, CH)):
            ch = min(CH, seq - c)
            gpos = start_pos + c
            xc = ttnn.slice(x_BSH, [0, c, 0], [1, c + ch, cfg.hidden])
            residual = xc
            ln = self._rms(xc, self.w["input_ln"])
            q, k, v = self._qkv_roped(
                ln,
                ch,
                gpos,
                rope=(rope_mats[chunk_idx] if rope_mats is not None else None),
            )
            col0 = (gpos - fill_base) // bs
            ncol = (ch + bs - 1) // bs
            chunk_pt = ttnn.slice(
                fill_page_table,
                [0, col0],
                [fill_page_table.shape[0], col0 + ncol],
            )
            ttnn.experimental.paged_fill_cache(kv_cache["k"], self._cast_fill(k, cdt), chunk_pt, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(kv_cache["v"], self._cast_fill(v, cdt), chunk_pt, batch_idx=user_id)
            user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            start_tensor = runtime_offsets.chunk_start_idxs[chunk_idx] if runtime_offsets is not None else None
            start_kw = (
                {"chunk_start_idx_tensor": start_tensor} if start_tensor is not None else {"chunk_start_idx": gpos}
            )
            if not cfg.is_sliding:
                attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q,
                    kv_cache["k"],
                    kv_cache["v"],
                    user_pt,
                    **start_kw,
                    compute_kernel_config=self._sdpa_compute,
                )
            else:
                # Sliding layers read from the paged window cache (not a local
                # k_tail), so a cached prefix (prefix caching) is visible: the
                # window can span back into positions filled by an earlier
                # prefill. ttnn chunked SDPA now composes with sliding_window_size
                # (see memory laguna-ttnn-incremental-rebuild; PoC PCC 1.0 vs the
                # dense sliding op). page_table is full-width with null-padded
                # out-of-window blocks (vLLM SlidingWindowManager), so absolute
                # chunk_start_idx=gpos indexing stays correct against the smaller
                # physical pool.
                attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q,
                    kv_cache["k"],
                    kv_cache["v"],
                    user_pt,
                    **start_kw,
                    sliding_window_size=win,
                    compute_kernel_config=self._sdpa_compute,
                )
            attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn = ttnn.reshape(attn, (1, ch, cfg.num_heads * cfg.head_dim))
            attn = self._gate(attn, ln)
            o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._ck_o)
            o = self._reduce(o)
            h = ttnn.add(residual, o)
            ln2 = self._rms(h, self.w["post_ln"])
            mlp_out = ttnn.reshape(self._mlp(ln2, ch, sharded=False), (1, ch, cfg.hidden))
            outs.append(ttnn.add(h, mlp_out))
        return ttnn.concat(outs, dim=1)

    # ---- decode: reuse optimized body + all_reduce after WO --------------- #
    def decode_forward(self, x_1BH, cur_pos, rope_idx, page_table, kv_cache, sequential_kv_write=False, rope_mats=None):
        cfg = self.cfg
        B = x_1BH.shape[-2]
        residual = x_1BH
        ln = self._rms(x_1BH, self.w["input_ln"])
        qkv = self._dram_mm(ln, self.w["wqkv"], self.w["wqkv_ds"], cfg.hidden, self.meta["qkv_w"], self._ck_qkv)
        if self.use_dram_sharded:
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = self._split_qkv(qkv, B)
        q = self._per_head_norm(q, self.w["q_norm"])
        k = self._per_head_norm(k, self.w["k_norm"])
        # share the DRAM cos/sin gather across layers of a kind (rope_mats); shard to L1
        # PER LAYER (an L1-sharded cos_sh cannot be hoisted — scratch, clobbered by later layers).
        if rope_mats is None:  # local fallback (layer PCC tests / direct callers)
            cos = self._rope_decode(rope_idx, B)
            sin = self._rope_decode(rope_idx, B, sin=True)
        else:
            cos, sin = rope_mats
        if self._use_fused_rope:  # gated (TT_LAGUNA_FUSED_ROPE=1): fused HF rotate_half — PCC-validate first
            cos_sh = self._shard_cossin(cos, B, cfg.rotary_dim)
            sin_sh = self._shard_cossin(sin, B, cfg.rotary_dim)
            q = self._fused_rope_decode(q, cos_sh, sin_sh, cfg.num_heads, B)
            k = self._fused_rope_decode(k, cos_sh, sin_sh, cfg.num_kv_heads, B)
        else:
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
        k_sh = self._shard_kv(k, B)
        v_sh = self._shard_kv(v, B)
        if sequential_kv_write and B > 1:
            # Spec-decode VERIFY: the B candidate rows share ONE user's blocks; with BLOCK_SIZE==TILE==32
            # consecutive positions land in the same tile, so a batched paged_update_cache RMW-races and
            # corrupts KV. Serialize the tiny per-row writes (matmuls + SDPA above/below still run batched,
            # so the verify stays fast). Mirrors gemma4/tt/attention/decode.py:147.
            self._seq_kv_write(kv_cache["k"], k_sh, cur_pos, page_table, B)
            self._seq_kv_write(kv_cache["v"], v_sh, cur_pos, page_table, B)
        else:
            ttnn.experimental.paged_update_cache(kv_cache["k"], k_sh, update_idxs_tensor=cur_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(kv_cache["v"], v_sh, update_idxs_tensor=cur_pos, page_table=page_table)
        sdpa_kwargs = {
            "cur_pos_tensor": cur_pos,
            "page_table_tensor": page_table,
            "scale": cfg.scaling,
            "compute_kernel_config": self._sdpa_compute,
        }
        if cfg.is_sliding:
            sdpa_kwargs["sliding_window_size"] = cfg.sliding_window
        if sequential_kv_write and self._decode_use_sdpa_pc:
            # Spec-decode verify. k_chunk via TT_LAGUNA_VERIFY_K (default 64). D=1 uses the same
            # accurate TTNN fallback as normal decode; the explicit parallel program has the same
            # partial-context reduction hazard in verify mode.
            # ACCURACY FIX (2026-08-06): this is the LIVE served decode_forward (overrides OptimizedDecoder's);
            # it previously ran _sdpa_pc (k128) — the SAME config proven LOSSY on normal decode (teacher top1
            # 0.95→0.58). Committed spec tokens ARE this verify's argmax, so k128 made spec-decode inherit the
            # lossy trajectory. _sdpa_pc_verify (TT_LAGUNA_VERIFY_K, default 64) aligns it with the accurate
            # normal-decode SDPA. The disproven "verify REQUIRES k128" claim is dropped (k64 verify runs
            # correctly on device — standalone driver + served HumanEval). See tt/optimized_decoder.py.
            sdpa_kwargs["program_config"] = self._sdpa_pc_verify
            sdpa_kwargs["num_kv_heads"] = cfg.num_kv_heads
        elif self._decode_use_sdpa_pc:
            # Stage 2 fix: accuracy-safe fast NORMAL-decode config — k_chunk=64 (not 128) keeps the
            # max_cores=16 parallel-KV-scan long-context speed but drops the k128 last-partial-chunk masking
            # that was LOSSY (teacher top1 0.95→0.58, layer PCC -0.016 at low non-aligned cur_pos).
            # See tt/optimized_decoder.py (decode SDPA program config).
            sdpa_kwargs["program_config"] = self._sdpa_pc_decode
            sdpa_kwargs["num_kv_heads"] = cfg.num_kv_heads
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q, kv_cache["k"], kv_cache["v"], **sdpa_kwargs
        )
        attn = ttnn.reshape(attn, (1, 1, B, cfg.num_heads * cfg.head_dim))
        attn = self._gate(attn, ln)
        q_w = self.meta["q_w"]
        o = self._dram_mm(attn, self.w["wo"], self.w["wo_ds"], q_w, cfg.hidden, self._ck_o)
        if self.use_dram_sharded:
            o = ttnn.sharded_to_interleaved(o, ttnn.L1_MEMORY_CONFIG)
        o = self._reduce(o)  # row-parallel partial -> replicated
        h = ttnn.add(residual, o)
        ln2 = self._rms(h, self.w["post_ln"])
        mlp_out = self._mlp(ln2, B, sharded=True)
        return ttnn.add(h, mlp_out)
