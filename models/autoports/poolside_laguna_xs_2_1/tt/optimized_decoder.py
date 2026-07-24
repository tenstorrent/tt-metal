# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optimized TTNN decoder for poolside/Laguna-XS-2.1 (single Blackhole p300c, 1x1 mesh).

Optimizes the functional decoder (``tt/functional_decoder.py``) for on-device
performance while preserving its prefill/decode semantics, paged KV-cache behavior,
determinism, non-aligned logical sequence support, and PCC>=0.995 correctness bar
against the same layer-only HF reference.

Optimizations over the functional path (see ``doc/optimized_decoder/``):
  * Precision/fidelity policy (``PrecisionPolicy``): BF16 activations/norms; BFP8
    attention (packed-QKV / O / gate) + dense-MLP + shared-expert weights; BFP4 MoE
    expert gate/up weights (down guarded); BF16 router; BFP8 paged KV cache. Compute
    fidelity dropped from the functional HiFi4 to HiFi2/LoFi per group. SDPA stays
    precise (fp32 accumulate, ``exp_approx_mode=False``) for long-context accuracy.
  * Packed Q/K/V projection (one matmul, split on device) — OPT-001.
  * DRAM-sharded decode matmuls (packed QKV, O, dense/shared MLP) with width-sharded
    L1 activations, following the ``models/common/modules`` pattern.
  * MoE routed experts keep ``ttnn.sparse_matmul`` (active-expert path) with BFP4
    weights; optional static ``nnz``.

Public API is identical to ``FunctionalDecoder`` so the same tests/harness drive it:
``from_state_dict``, ``alloc_kv_cache``, ``make_page_table``, ``prefill_forward``,
``decode_forward``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


# --------------------------------------------------------------------------- #
# Precision / fidelity policy
# --------------------------------------------------------------------------- #
_DTYPE_TO_STR = {
    ttnn.bfloat16: "bf16",
    ttnn.bfloat8_b: "bfp8",
    ttnn.bfloat4_b: "bfp4",
    ttnn.float32: "fp32",
}
_STR_TO_DTYPE = {v: k for k, v in _DTYPE_TO_STR.items()}
# Aliases accepted on load.
_STR_TO_DTYPE.update(
    {
        "bfloat16": ttnn.bfloat16,
        "bfloat8_b": ttnn.bfloat8_b,
        "bfloat4_b": ttnn.bfloat4_b,
        "float32": ttnn.float32,
    }
)
_FIDELITY_STRS = ("LoFi", "HiFi2", "HiFi4")


@dataclass
class PrecisionPolicy:
    """Per-tensor-group weight dtypes, compute-fidelity, CCL/KV/logits dtypes.

    This is the single source of truth for the model's precision policy: the decoder
    reads BOTH the weight dtypes AND the per-group matmul compute fidelities from here
    (see ``OptimizedDecoder.__init__``), and ``LagunaModel.from_pretrained`` reads
    ``lm_head`` from here. Activations/residual stay BF16 (this model never materialises
    the residual stream in a lower dtype — recorded as an assumption, not a knob).

    Defaults are the optimized-full-model policy (stage 06): BF16 activations/norms; BFP8
    attention (packed-QKV / O / gate) + dense-MLP + shared-expert weights; BFP4 routed-MoE
    expert gate/up/down; BF16 router; BFP8 paged KV cache; BFP8 column-sharded LM head.
    Compute fidelity: LoFi for projections, HiFi2 for the router + attention gate, HiFi4
    (fp32 acc) reserved for the RMSNorms and SDPA (accuracy-sensitive, not swept).

    Tuned one group at a time with real-weight full-model top-1/top-5 (see
    ``doc/datatype_sweep/``). Serialise/deserialise via ``to_dict``/``from_dict`` so the
    selected config is a required artifact the construction path consumes by default.
    """

    # ---- weight dtypes ---- #
    attn_qkv: ttnn.DataType = ttnn.bfloat8_b
    attn_o: ttnn.DataType = ttnn.bfloat8_b
    attn_gate: ttnn.DataType = ttnn.bfloat8_b  # softplus output gate g_proj
    dense_ff13: ttnn.DataType = ttnn.bfloat8_b
    dense_ff2: ttnn.DataType = ttnn.bfloat8_b
    moe_ff13: ttnn.DataType = ttnn.bfloat4_b
    moe_ff2: ttnn.DataType = ttnn.bfloat4_b
    shared_ff13: ttnn.DataType = ttnn.bfloat8_b
    shared_ff2: ttnn.DataType = ttnn.bfloat8_b
    router: ttnn.DataType = ttnn.bfloat16
    qk_norm: ttnn.DataType = ttnn.bfloat16
    lm_head: ttnn.DataType = ttnn.bfloat8_b
    # ---- KV cache / CCL / logits dtypes ---- #
    kv_cache: ttnn.DataType = ttnn.bfloat8_b
    ccl: ttnn.DataType = ttnn.bfloat16  # all_reduce payload dtype (BF16 == replicated residual)
    activation: ttnn.DataType = ttnn.bfloat16  # residual/activation stream (fixed; documented)
    logits: ttnn.DataType = ttnn.bfloat16  # LM-head matmul output dtype
    # ---- per-group matmul compute fidelities (strings: LoFi / HiFi2 / HiFi4) ---- #
    fid_attn_qkv: str = "LoFi"
    fid_attn_o: str = "LoFi"
    fid_attn_gate: str = "HiFi2"
    fid_dense: str = "LoFi"
    fid_shared: str = "LoFi"
    fid_router: str = "HiFi2"
    fid_moe: str = "LoFi"

    # ---- (de)serialisation ---- #
    _DTYPE_FIELDS = (
        "attn_qkv",
        "attn_o",
        "attn_gate",
        "dense_ff13",
        "dense_ff2",
        "moe_ff13",
        "moe_ff2",
        "shared_ff13",
        "shared_ff2",
        "router",
        "qk_norm",
        "lm_head",
        "kv_cache",
        "ccl",
        "activation",
        "logits",
    )
    _FID_FIELDS = (
        "fid_attn_qkv",
        "fid_attn_o",
        "fid_attn_gate",
        "fid_dense",
        "fid_shared",
        "fid_router",
        "fid_moe",
    )

    def to_dict(self) -> dict:
        d = {f: _DTYPE_TO_STR[getattr(self, f)] for f in self._DTYPE_FIELDS}
        for f in self._FID_FIELDS:
            d[f] = getattr(self, f)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PrecisionPolicy":
        kw = {}
        for f in cls._DTYPE_FIELDS:
            if f in d and d[f] is not None:
                kw[f] = _STR_TO_DTYPE[str(d[f]).lower()] if not isinstance(d[f], ttnn.DataType) else d[f]
        for f in cls._FID_FIELDS:
            if f in d and d[f] is not None:
                assert d[f] in _FIDELITY_STRS, f"{f}={d[f]} not in {_FIDELITY_STRS}"
                kw[f] = d[f]
        return cls(**kw)


# --------------------------------------------------------------------------- #
# Per-layer configuration (identical semantics to functional LayerConfig)
# --------------------------------------------------------------------------- #
@dataclass
class LayerConfig:
    hidden: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    num_kv_groups: int
    scaling: float
    attention_type: str
    is_sliding: bool
    sliding_window: int | None
    rotary_dim: int
    eps: float
    is_moe: bool
    intermediate: int
    num_experts: int
    top_k: int
    moe_intermediate: int
    shared_intermediate: int
    routed_scaling: float
    norm_topk_prob: bool

    @classmethod
    def from_hf(cls, hf_config, layer_idx: int) -> "LayerConfig":
        layer_types = getattr(hf_config, "layer_types", None)
        attention_type = layer_types[layer_idx] if layer_types else "full_attention"
        is_sliding = attention_type == "sliding_attention"
        per_layer_heads = getattr(hf_config, "num_attention_heads_per_layer", None)
        num_heads = per_layer_heads[layer_idx] if per_layer_heads else hf_config.num_attention_heads
        head_dim = hf_config.head_dim
        nkv = hf_config.num_key_value_heads
        rp = hf_config.rope_parameters
        sub = rp["sliding_attention"] if is_sliding else rp["full_attention"]
        partial = sub.get("partial_rotary_factor", 1.0)
        rotary_dim = int(head_dim * partial)
        is_moe = (layer_idx not in hf_config.mlp_only_layers) and (
            hf_config.num_experts > 0 and (layer_idx + 1) % hf_config.decoder_sparse_step == 0
        )
        return cls(
            hidden=hf_config.hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=nkv,
            num_kv_groups=num_heads // nkv,
            scaling=head_dim**-0.5,
            attention_type=attention_type,
            is_sliding=is_sliding,
            sliding_window=hf_config.sliding_window if is_sliding else None,
            rotary_dim=rotary_dim,
            eps=hf_config.rms_norm_eps,
            is_moe=is_moe,
            intermediate=hf_config.intermediate_size,
            num_experts=hf_config.num_experts,
            top_k=hf_config.num_experts_per_tok,
            moe_intermediate=hf_config.moe_intermediate_size,
            shared_intermediate=hf_config.shared_expert_intermediate_size,
            routed_scaling=float(getattr(hf_config, "moe_routed_scaling_factor", 1.0)),
            norm_topk_prob=hf_config.norm_topk_prob,
        )


# --------------------------------------------------------------------------- #
# Setup-time helpers
# --------------------------------------------------------------------------- #
def _rectangular_grid(n: int) -> tuple[int, int]:
    """Rectangular (x, y) core grid whose product == number of 32-wide N-tiles."""
    nt = int(math.ceil(n / TILE))
    max_x, max_y = 8, 8
    for cores in range(min(nt, max_x * max_y), 0, -1):
        if nt % cores != 0:
            continue
        for x in range(min(cores, max_x), 0, -1):
            if cores % x == 0 and cores // x <= max_y:
                return (x, cores // x)
    return (1, 1)


def _sparse_pc(n: int, m: int, k: int, in0_block_w: int = 16):
    """1D multicast program config for the routed-expert sparse matmuls (unchanged
    topology from functional; weights are BFP4 in the optimized policy)."""
    core_x, core_y = _rectangular_grid(n)
    num_cores = core_x * core_y
    nt = int(math.ceil(n / TILE))
    per_core_N = (nt + num_cores - 1) // num_cores
    kt = int(math.ceil(k / TILE))
    if kt % in0_block_w != 0:
        divs = [d for d in range(2, in0_block_w + 1) if kt % d == 0]
        in0_block_w = max(divs) if divs else kt
    per_core_M = max(1, (m + TILE - 1) // TILE)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


# --- DRAM-sharded decode matmul helpers (adapted from models/common/modules/mlp/mlp_1d.py) --- #
def _largest_divisor(n: int, max_divisor: int = 8) -> int:
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _decode_shard_cores(k: int, n: int, max_cores: int = 32) -> int:
    """Pick a compute-core count that divides K into tiles, keeps in0_block_w>=2
    (K-tiles/cores >= 2), and prefers dividing N-tiles cleanly. Bounded to keep the
    activation shard wide enough for a good inner block."""
    kt, nt = k // TILE, int(math.ceil(n / TILE))
    best = 1
    for c in range(1, max_cores + 1):
        if kt % c != 0:
            continue
        if kt // c < 2:  # keep in0_block_w >= 2
            continue
        # prefer c that also divides n-tiles; otherwise still allow (ceil per_core_N)
        best = c
    # bias toward a divisor of nt among the legal set for balanced per_core_N
    legal = [c for c in range(1, max_cores + 1) if kt % c == 0 and kt // c >= 2]
    div_n = [c for c in legal if nt % c == 0]
    return max(div_n) if div_n else (max(legal) if legal else 1)


def _core_grid(num_cores: int) -> ttnn.CoreGrid:
    for rows in range(8, 0, -1):
        if num_cores % rows == 0 and num_cores // rows <= 8:
            return ttnn.CoreGrid(y=rows, x=num_cores // rows)
    return ttnn.CoreGrid(y=1, x=num_cores)


def _dram_weight_memcfg(k: int, n: int, dram_cores: int) -> ttnn.MemoryConfig:
    padded_n = math.ceil(n / (TILE * dram_cores)) * (TILE * dram_cores)
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_cores - 1, 0))})
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _dram_matmul_pc(m: int, k: int, n: int, num_cores: int, fused_activation=None):
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_largest_divisor(k // (TILE * num_cores)),
        per_core_M=math.ceil(m / TILE),
        per_core_N=math.ceil(n / (TILE * num_cores)),
        fused_activation=fused_activation,
    )


def _width_sharded_l1(m: int, k: int, num_cores: int) -> ttnn.MemoryConfig:
    grid = _core_grid(num_cores)
    return ttnn.create_sharded_memory_config(
        (m, k // num_cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _hf_rope_tables(hf_config, attention_type: str, max_seq_len: int):
    import copy

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_id = hf_config._name_or_path or "poolside/Laguna-XS-2.1"
    RE = get_class_from_dynamic_module("modeling_laguna.LagunaRotaryEmbedding", model_id)
    rp = hf_config.rope_parameters
    cfg = copy.deepcopy(hf_config)
    if attention_type == "sliding_attention":
        cfg.rope_parameters = dict(hf_config.swa_rope_parameters)
        cfg.partial_rotary_factor = cfg.rope_parameters.get("partial_rotary_factor")
    else:
        cfg.rope_parameters = dict(rp["full_attention"])
    re = RE(config=cfg)
    pos = torch.arange(max_seq_len).unsqueeze(0)
    dummy = torch.zeros(1, max_seq_len, 1)
    cos, sin = re(dummy, pos)
    return cos[0].float(), sin[0].float()


def _as_tt(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=None):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def _linear_w(t, device, dtype=ttnn.bfloat16, memory_config=None):
    """HF linear weight [out, in] -> ttnn [in, out]."""
    return _as_tt(t.t().contiguous(), device, dtype, memory_config=memory_config)


# --------------------------------------------------------------------------- #
# Optimized decoder module
# --------------------------------------------------------------------------- #
class OptimizedDecoder(LightweightModule):
    MOE_PREFILL_CHUNK = 256
    PIPE_CHUNK = 8192
    PREFILL_SDPA_CHUNK = 8192

    def __init__(self, cfg: LayerConfig, weights: dict, cos_table, sin_table, mesh_device, policy, meta):
        self.cfg = cfg
        self.w = weights
        self.cos_2d = cos_table
        self.sin_2d = sin_table
        self.device = mesh_device
        self.policy = policy
        self.meta = meta  # dict of precomputed shapes/configs (dram cores, grids, etc.)
        self.use_dram_sharded = True

        # Compute-kernel configs (Blackhole). LoFi/HiFi2 for projections; precise for norm/SDPA.
        arch = mesh_device.arch()
        self._ck_lofi = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._ck_hifi2 = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._ck_hifi4 = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._norm_ck = self._ck_hifi4
        # Precise SDPA (fp32 accumulate, exact exp) — required for long-context accuracy.
        grid = mesh_device.compute_with_storage_grid_size()
        self._sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._sdpa_compute = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # fidelity assignment per matmul group — READ FROM THE POLICY so the selected
        # compute-fidelity config is actually consumed by the measured runtime path
        # (defaults preserve the stage-06 policy: LoFi projections, HiFi2 gate/router).
        self._ck_by_fid = {"LoFi": self._ck_lofi, "HiFi2": self._ck_hifi2, "HiFi4": self._ck_hifi4}
        self._ck_qkv = self._ck_by_fid[policy.fid_attn_qkv]
        self._ck_o = self._ck_by_fid[policy.fid_attn_o]
        self._ck_gate = self._ck_by_fid[policy.fid_attn_gate]
        self._ck_dense = self._ck_by_fid[policy.fid_dense]
        self._ck_shared = self._ck_by_fid[policy.fid_shared]
        self._ck_router = self._ck_by_fid[policy.fid_router]
        self._ck_moe = self._ck_by_fid[policy.fid_moe]

    # ---- construction ------------------------------------------------------ #
    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, max_seq_len, policy=None, **kwargs):
        cfg = LayerConfig.from_hf(hf_config, layer_idx)
        policy = policy or PrecisionPolicy()
        dev = mesh_device
        w = {}
        dram_cores = mesh_device.dram_grid_size().x

        def g(name):
            return state_dict[name].float()

        def store(key, w_in, k, n, dtype):
            """Store an interleaved copy (``key``, for prefill 2D matmuls) and a
            DRAM-width-sharded copy (``key+"_ds"``, for decode DRAM-sharded matmuls).
            ``w_in`` is already in ttnn [in, out] orientation."""
            w[key] = _as_tt(w_in, dev, dtype)
            w[key + "_ds"] = _as_tt(w_in, dev, dtype, memory_config=_dram_weight_memcfg(k, n, dram_cores))

        q_w = cfg.num_heads * cfg.head_dim
        kv_w = cfg.num_kv_heads * cfg.head_dim
        qkv_w = q_w + 2 * kv_w
        H = cfg.hidden

        # --- packed QKV weight [H, qkv_w] (concat of q/k/v output columns) --- #
        wq = g("self_attn.q_proj.weight").t().contiguous()  # [H, q_w]
        wk = g("self_attn.k_proj.weight").t().contiguous()
        wv = g("self_attn.v_proj.weight").t().contiguous()
        wqkv = torch.cat([wq, wk, wv], dim=1).contiguous()  # [H, qkv_w]
        store("wqkv", wqkv, H, qkv_w, policy.attn_qkv)
        store("wo", g("self_attn.o_proj.weight").t().contiguous(), q_w, H, policy.attn_o)
        w["wg"] = _linear_w(g("self_attn.g_proj.weight"), dev, policy.attn_gate)
        w["q_norm"] = _as_tt(g("self_attn.q_norm.weight").reshape(1, 1, 1, cfg.head_dim), dev, policy.qk_norm)
        w["k_norm"] = _as_tt(g("self_attn.k_norm.weight").reshape(1, 1, 1, cfg.head_dim), dev, policy.qk_norm)
        w["input_ln"] = _as_tt(g("input_layernorm.weight").reshape(1, 1, 1, H), dev)
        w["post_ln"] = _as_tt(g("post_attention_layernorm.weight").reshape(1, 1, 1, H), dev)

        if cfg.is_moe:
            E, I = cfg.num_experts, cfg.moe_intermediate
            w["gate_w"] = _linear_w(g("mlp.gate.weight"), dev, policy.router)  # [H, E]
            bias = g("mlp.experts.e_score_correction_bias").reshape(1, 1, 1, E)
            w["e_bias"] = _as_tt(bias, dev)
            gate = torch.stack([g(f"mlp.experts.{i}.gate_proj.weight") for i in range(E)])  # [E,I,H]
            up = torch.stack([g(f"mlp.experts.{i}.up_proj.weight") for i in range(E)])
            down = torch.stack([g(f"mlp.experts.{i}.down_proj.weight") for i in range(E)])  # [E,H,I]
            w["exp_gate"] = _as_tt(gate.transpose(1, 2).reshape(1, E, H, I), dev, policy.moe_ff13)
            w["exp_up"] = _as_tt(up.transpose(1, 2).reshape(1, E, H, I), dev, policy.moe_ff13)
            w["exp_down"] = _as_tt(down.transpose(1, 2).reshape(1, E, I, H), dev, policy.moe_ff2)
            sh_I = cfg.shared_intermediate
            store("sh_gate", g("mlp.shared_expert.gate_proj.weight").t().contiguous(), H, sh_I, policy.shared_ff13)
            store("sh_up", g("mlp.shared_expert.up_proj.weight").t().contiguous(), H, sh_I, policy.shared_ff13)
            store("sh_down", g("mlp.shared_expert.down_proj.weight").t().contiguous(), sh_I, H, policy.shared_ff2)
        else:
            II = cfg.intermediate
            store("mlp_gate", g("mlp.gate_proj.weight").t().contiguous(), H, II, policy.dense_ff13)
            store("mlp_up", g("mlp.up_proj.weight").t().contiguous(), H, II, policy.dense_ff13)
            store("mlp_down", g("mlp.down_proj.weight").t().contiguous(), II, H, policy.dense_ff2)

        cos, sin = _hf_rope_tables(hf_config, cfg.attention_type, max_seq_len)
        cos_2d = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        sin_2d = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

        meta = {
            "dram_cores": dram_cores,
            "q_w": q_w,
            "kv_w": kv_w,
            "qkv_w": qkv_w,
        }
        return cls(cfg, w, cos_2d, sin_2d, dev, policy, meta)

    # ---- KV cache allocation ---------------------------------------------- #
    def alloc_kv_cache(self, max_users, max_seq_len, block_size=32, dtype=None):
        dtype = dtype or self.policy.kv_cache
        blocks_per_user = int(math.ceil(max_seq_len / block_size))
        max_num_blocks = blocks_per_user * max_users
        shape = (max_num_blocks, self.cfg.num_kv_heads, block_size, self.cfg.head_dim)
        k = ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = ttnn.from_torch(
            torch.zeros(shape),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return {"k": k, "v": v, "block_size": block_size, "blocks_per_user": blocks_per_user, "dtype": dtype}

    def make_page_table(self, num_users, blocks_per_user):
        pt = torch.arange(num_users * blocks_per_user, dtype=torch.int32).reshape(num_users, blocks_per_user)
        return ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    # ---- shared ops -------------------------------------------------------- #
    def _rms(self, x, weight):
        return ttnn.rms_norm(x, weight=weight, epsilon=self.cfg.eps, compute_kernel_config=self._norm_ck)

    def _per_head_norm(self, x, weight):
        b, h, s, d = x.shape
        flat = ttnn.reshape(x, (1, 1, b * h * s, d))
        normed = ttnn.rms_norm(flat, weight=weight, epsilon=self.cfg.eps, compute_kernel_config=self._norm_ck)
        return ttnn.reshape(normed, (b, h, s, d))

    def _apply_rope(self, x, cos, sin):
        rd = self.cfg.rotary_dim
        hd = self.cfg.head_dim
        if rd == hd:
            x_rot, x_pass = x, None
        else:
            x_rot = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], rd])
            x_pass = ttnn.slice(x, [0, 0, 0, rd], list(x.shape))
        half = rd // 2
        x1 = ttnn.slice(x_rot, [0, 0, 0, 0], [x_rot.shape[0], x_rot.shape[1], x_rot.shape[2], half])
        x2 = ttnn.slice(x_rot, [0, 0, 0, half], [x_rot.shape[0], x_rot.shape[1], x_rot.shape[2], rd])
        rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        embed = ttnn.add(ttnn.mul(x_rot, cos), ttnn.mul(rot, sin))
        if x_pass is None:
            return embed
        return ttnn.concat([embed, x_pass], dim=-1)

    def _gate(self, attn, ln):
        cfg = self.cfg
        g = ttnn.linear(ln, self.w["wg"], compute_kernel_config=self._ck_gate)
        g = ttnn.softplus(g)
        shp = list(attn.shape)
        attn_h = ttnn.reshape(attn, shp[:-1] + [cfg.num_heads, cfg.head_dim])
        g_h = ttnn.reshape(g, shp[:-1] + [cfg.num_heads, 1])
        gated = ttnn.mul(attn_h, g_h)
        return ttnn.reshape(gated, shp)

    # ---- DRAM-sharded matmul ---------------------------------------------- #
    def _dram_mm(self, x, w_il, w_ds, k, n, ck, fused_activation=None):
        """Width-shard x in L1, run DRAM-sharded matmul, return L1-width-sharded output.
        Falls back to a plain interleaved linear (interleaved weight) if disabled.
        The matmul M is the tile-padded physical row count of ``x``."""
        if not self.use_dram_sharded:
            return ttnn.linear(x, w_il, compute_kernel_config=ck)
        m = ((x.shape[-2] + TILE - 1) // TILE) * TILE
        num_cores = _decode_shard_cores(k, n)
        x_sh = ttnn.to_memory_config(x, _width_sharded_l1(m, k, num_cores))
        out = ttnn.linear(
            x_sh,
            w_ds,
            program_config=_dram_matmul_pc(m, k, n, num_cores, fused_activation=fused_activation),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=ck,
        )
        ttnn.deallocate(x_sh)
        return out

    def _glu_mlp(self, x, key, H, I, ck, sharded):
        """SwiGLU MLP over interleaved (prefill) or DRAM-sharded (decode) weights.
        ``key`` in {"mlp","sh"}: weight keys are ``{gate,up,down}`` variants."""
        gk, uk, dk = {
            "mlp": ("mlp_gate", "mlp_up", "mlp_down"),
            "sh": ("sh_gate", "sh_up", "sh_down"),
        }[key]
        w = self.w
        if sharded and self.use_dram_sharded:
            g = self._dram_mm(x, w[gk], w[gk + "_ds"], H, I, ck)
            u = self._dram_mm(x, w[uk], w[uk + "_ds"], H, I, ck)
            g = ttnn.sharded_to_interleaved(g, ttnn.L1_MEMORY_CONFIG)
            u = ttnn.sharded_to_interleaved(u, ttnn.L1_MEMORY_CONFIG)
            gu = ttnn.mul(ttnn.silu(g), u)
            out = self._dram_mm(gu, w[dk], w[dk + "_ds"], I, H, ck)
            return ttnn.sharded_to_interleaved(out, ttnn.L1_MEMORY_CONFIG)
        gate = ttnn.silu(ttnn.linear(x, w[gk], compute_kernel_config=ck))
        up = ttnn.linear(x, w[uk], compute_kernel_config=ck)
        return ttnn.linear(ttnn.mul(gate, up), w[dk], compute_kernel_config=ck)

    # ---- MoE --------------------------------------------------------------- #
    def _moe(self, ln_flat, m, sharded):
        cfg = self.cfg
        E, H, I, K = cfg.num_experts, cfg.hidden, cfg.moe_intermediate, cfg.top_k
        T = ln_flat.shape[2]
        logits = ttnn.linear(ln_flat, self.w["gate_w"], compute_kernel_config=self._ck_router)
        scores = ttnn.sigmoid(logits)
        sel = ttnn.add(scores, self.w["e_bias"])
        _, idx = ttnn.topk(ttnn.typecast(sel, ttnn.bfloat16), k=K, dim=-1, sorted=True)
        wsel = ttnn.gather(scores, dim=3, index=idx)
        if cfg.norm_topk_prob:
            wsum = ttnn.sum(wsel, dim=3, keepdim=True)
            wsel = ttnn.div(wsel, wsum)
        if cfg.routed_scaling != 1.0:
            wsel = ttnn.multiply(wsel, cfg.routed_scaling)
        dense = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=idx, src=wsel)
        union = ttnn.sum(dense, dim=2, keepdim=True)
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        a = ttnn.reshape(ln_flat, (1, 1, T, H))
        # Decode: keep expert outputs/intermediates in L1 (DRAM round-trips are a decode
        # perf smell for the sparse expert path — see GPT-OSS experts/decode.py). Prefill
        # chunks materialise [1,E,chunk,I] which is too large for L1 → keep DRAM there.
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
        gate_o = ttnn.reshape(gate_o, (1, E, T, I))
        up_o = ttnn.reshape(up_o, (1, E, T, I))
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
        )
        wv = ttnn.reshape(dense, (1, T, E))
        wv = ttnn.permute(wv, (0, 2, 1))
        wv = ttnn.reshape(wv, (1, E, T, 1))
        weighted = ttnn.mul(down_o, wv)
        routed = ttnn.sum(weighted, dim=1)
        routed = ttnn.reshape(routed, (1, 1, T, H))
        shared = self._glu_mlp(ln_flat, "sh", cfg.hidden, cfg.shared_intermediate, self._ck_shared, sharded)
        return ttnn.add(routed, shared)

    def _mlp(self, ln, T, sharded):
        cfg = self.cfg
        ln_flat = ttnn.reshape(ln, (1, 1, T, cfg.hidden))
        if not cfg.is_moe:
            return self._glu_mlp(ln_flat, "mlp", cfg.hidden, cfg.intermediate, self._ck_dense, sharded)
        if T <= self.MOE_PREFILL_CHUNK:
            return self._moe(ln_flat, T, sharded)
        outs = []
        for s in range(0, T, self.MOE_PREFILL_CHUNK):
            e = min(s + self.MOE_PREFILL_CHUNK, T)
            chunk = ttnn.slice(ln_flat, [0, 0, s, 0], [1, 1, e, cfg.hidden])
            outs.append(self._moe(chunk, e - s, sharded))
        return ttnn.concat(outs, dim=2)

    # ---- QKV (packed) ------------------------------------------------------ #
    def _split_qkv(self, qkv, rows):
        """qkv: [1,1,rows,qkv_w] -> q[1,rows,nh,hd], k/v[1,rows,nkv,hd]."""
        cfg = self.cfg
        q_w, kv_w = self.meta["q_w"], self.meta["kv_w"]
        q = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, rows, q_w])
        k = ttnn.slice(qkv, [0, 0, 0, q_w], [1, 1, rows, q_w + kv_w])
        v = ttnn.slice(qkv, [0, 0, 0, q_w + kv_w], [1, 1, rows, q_w + 2 * kv_w])
        q = ttnn.reshape(q, (1, rows, cfg.num_heads, cfg.head_dim))
        k = ttnn.reshape(k, (1, rows, cfg.num_kv_heads, cfg.head_dim))
        v = ttnn.reshape(v, (1, rows, cfg.num_kv_heads, cfg.head_dim))
        return q, k, v

    # ---- prefill ----------------------------------------------------------- #
    def _qkv_roped(self, ln, seq, start_pos):
        cfg = self.cfg
        qkv = ttnn.linear(ln, self.w["wqkv"], compute_kernel_config=self._ck_qkv)  # [1,seq,qkv_w] (seq in dim1)
        qkv = ttnn.reshape(qkv, (1, 1, seq, self.meta["qkv_w"]))
        q, k, v = self._split_qkv(qkv, seq)
        q = ttnn.permute(q, (0, 2, 1, 3))  # [1,nh,seq,hd]
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        q = self._per_head_norm(q, self.w["q_norm"])
        k = self._per_head_norm(k, self.w["k_norm"])
        cos = self._rope_prefill(start_pos, seq)
        sin = self._rope_prefill(start_pos, seq, sin=True)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        return q, k, v

    def _cast_fill(self, t, cache_dtype):
        return ttnn.typecast(t, cache_dtype) if t.dtype != cache_dtype else t

    def prefill_forward(self, x_BSH, kv_cache, page_table, *, user_id=0, start_pos=0):
        seq = x_BSH.shape[-2]
        if seq > self.PIPE_CHUNK:
            return self._prefill_pipelined(x_BSH, kv_cache, page_table, user_id, start_pos)
        cfg = self.cfg
        residual = x_BSH
        ln = self._rms(x_BSH, self.w["input_ln"])
        q, k, v = self._qkv_roped(ln, seq, start_pos)
        cdt = kv_cache["dtype"]
        ttnn.experimental.paged_fill_cache(kv_cache["k"], self._cast_fill(k, cdt), page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(kv_cache["v"], self._cast_fill(v, cdt), page_table, batch_idx=user_id)
        attn = self._prefill_attention(q, k, v, kv_cache, page_table, user_id, start_pos, seq)
        attn = ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), (1, seq, cfg.num_heads * cfg.head_dim))
        attn = self._gate(attn, ln)
        o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._ck_o)
        h = ttnn.add(residual, o)
        ln2 = self._rms(h, self.w["post_ln"])
        mlp_out = self._mlp(ln2, seq, sharded=False)
        mlp_out = ttnn.reshape(mlp_out, (1, seq, cfg.hidden))
        return ttnn.add(h, mlp_out)

    def _prefill_pipelined(self, x_BSH, kv_cache, page_table, user_id, start_pos):
        cfg = self.cfg
        seq = x_BSH.shape[-2]
        bs = kv_cache["block_size"]
        cdt = kv_cache["dtype"]
        CH = (self.PIPE_CHUNK // bs) * bs
        win = cfg.sliding_window
        k_tail = v_tail = None
        outs = []
        for c in range(0, seq, CH):
            ch = min(CH, seq - c)
            gpos = start_pos + c
            xc = ttnn.slice(x_BSH, [0, c, 0], [1, c + ch, cfg.hidden])
            residual = xc
            ln = self._rms(xc, self.w["input_ln"])
            q, k, v = self._qkv_roped(ln, ch, gpos)
            col0 = gpos // bs
            ncol = (ch + bs - 1) // bs
            chunk_pt = ttnn.slice(page_table, [0, col0], [page_table.shape[0], col0 + ncol])
            ttnn.experimental.paged_fill_cache(kv_cache["k"], self._cast_fill(k, cdt), chunk_pt, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(kv_cache["v"], self._cast_fill(v, cdt), chunk_pt, batch_idx=user_id)
            if not cfg.is_sliding:
                user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
                attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q,
                    kv_cache["k"],
                    kv_cache["v"],
                    user_pt,
                    chunk_start_idx=gpos,
                    compute_kernel_config=self._sdpa_compute,
                )
            else:
                if k_tail is None:
                    k_loc, v_loc, pad = k, v, 0
                else:
                    k_loc = ttnn.concat([k_tail, k], dim=2)
                    v_loc = ttnn.concat([v_tail, v], dim=2)
                    pad = k_tail.shape[2]
                    qpad = ttnn.zeros(
                        [1, cfg.num_heads, pad, cfg.head_dim],
                        dtype=q.dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                    )
                    q = ttnn.concat([qpad, q], dim=2)
                out_loc = ttnn.transformer.scaled_dot_product_attention(
                    q,
                    k_loc,
                    v_loc,
                    is_causal=True,
                    sliding_window_size=win,
                    scale=cfg.scaling,
                    compute_kernel_config=self._sdpa_compute,
                )
                attn = ttnn.slice(out_loc, [0, 0, pad, 0], [1, cfg.num_heads, pad + ch, cfg.head_dim])
                tail = min(win - 1, ch)
                k_tail = ttnn.slice(k, [0, 0, ch - tail, 0], [1, cfg.num_kv_heads, ch, cfg.head_dim])
                v_tail = ttnn.slice(v, [0, 0, ch - tail, 0], [1, cfg.num_kv_heads, ch, cfg.head_dim])
            attn = ttnn.reshape(ttnn.permute(attn, (0, 2, 1, 3)), (1, ch, cfg.num_heads * cfg.head_dim))
            attn = self._gate(attn, ln)
            o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._ck_o)
            h = ttnn.add(residual, o)
            ln2 = self._rms(h, self.w["post_ln"])
            mlp_out = ttnn.reshape(self._mlp(ln2, ch, sharded=False), (1, ch, cfg.hidden))
            outs.append(ttnn.add(h, mlp_out))
        return ttnn.concat(outs, dim=1)

    def _prefill_attention(self, q, k, v, kv_cache, page_table, user_id, start_pos, seq):
        cfg = self.cfg
        base = {"scale": cfg.scaling, "compute_kernel_config": self._sdpa_compute}
        if seq <= self.PREFILL_SDPA_CHUNK:
            kw = {"is_causal": True, "program_config": self._sdpa_pc, **base}
            if cfg.is_sliding:
                kw["sliding_window_size"] = cfg.sliding_window
            return ttnn.transformer.scaled_dot_product_attention(q, k, v, **kw)
        CH = self.PREFILL_SDPA_CHUNK
        outs = []
        if not cfg.is_sliding:
            user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            for c in range(start_pos, start_pos + seq, CH):
                ch = min(CH, start_pos + seq - c)
                q_c = ttnn.slice(q, [0, 0, c - start_pos, 0], [1, cfg.num_heads, c - start_pos + ch, cfg.head_dim])
                out_c = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_c,
                    kv_cache["k"],
                    kv_cache["v"],
                    user_pt,
                    chunk_start_idx=c,
                    compute_kernel_config=self._sdpa_compute,
                )
                outs.append(out_c)
            return ttnn.concat(outs, dim=2)
        win = cfg.sliding_window
        for c in range(0, seq, CH):
            ch = min(CH, seq - c)
            s0 = max(0, c - (win - 1))
            q_sl = ttnn.slice(q, [0, 0, s0, 0], [1, cfg.num_heads, c + ch, cfg.head_dim])
            k_sl = ttnn.slice(k, [0, 0, s0, 0], [1, cfg.num_kv_heads, c + ch, cfg.head_dim])
            v_sl = ttnn.slice(v, [0, 0, s0, 0], [1, cfg.num_kv_heads, c + ch, cfg.head_dim])
            out_sl = ttnn.transformer.scaled_dot_product_attention(
                q_sl, k_sl, v_sl, is_causal=True, sliding_window_size=win, **base
            )
            outs.append(ttnn.slice(out_sl, [0, 0, c - s0, 0], [1, cfg.num_heads, ch + (c - s0), cfg.head_dim]))
        return ttnn.concat(outs, dim=2)

    def _rope_prefill(self, start_pos, seq, sin=False):
        table = self.sin_2d if sin else self.cos_2d
        rd = self.cfg.rotary_dim
        sliced = ttnn.slice(table, [start_pos, 0], [start_pos + seq, rd])
        return ttnn.to_layout(ttnn.reshape(sliced, (1, 1, seq, rd)), ttnn.TILE_LAYOUT)

    # ---- decode ------------------------------------------------------------ #
    def decode_forward(self, x_1BH, cur_pos, rope_idx, page_table, kv_cache):
        cfg = self.cfg
        B = x_1BH.shape[-2]
        m = ((B + TILE - 1) // TILE) * TILE
        residual = x_1BH
        ln = self._rms(x_1BH, self.w["input_ln"])  # [1,1,B,H]
        # packed QKV (DRAM-sharded), split on device
        qkv = self._dram_mm(ln, self.w["wqkv"], self.w["wqkv_ds"], cfg.hidden, self.meta["qkv_w"], self._ck_qkv)
        # DRAM interleaved: paged SDPA decode (esp. sliding-window) requires Q in DRAM.
        if self.use_dram_sharded:
            qkv = ttnn.sharded_to_interleaved(qkv, ttnn.DRAM_MEMORY_CONFIG)
        q, k, v = self._split_qkv(qkv, B)
        q = self._per_head_norm(q, self.w["q_norm"])
        k = self._per_head_norm(k, self.w["k_norm"])
        cos = self._rope_decode(rope_idx, B)
        sin = self._rope_decode(rope_idx, B, sin=True)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        k_sh = self._shard_kv(k, B)
        v_sh = self._shard_kv(v, B)
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
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q, kv_cache["k"], kv_cache["v"], **sdpa_kwargs
        )  # [1,B,nh,hd]
        attn = ttnn.reshape(attn, (1, 1, B, cfg.num_heads * cfg.head_dim))
        attn = self._gate(attn, ln)
        q_w = self.meta["q_w"]
        o = self._dram_mm(attn, self.w["wo"], self.w["wo_ds"], q_w, cfg.hidden, self._ck_o)
        if self.use_dram_sharded:
            o = ttnn.sharded_to_interleaved(o, ttnn.L1_MEMORY_CONFIG)
        h = ttnn.add(residual, o)
        ln2 = self._rms(h, self.w["post_ln"])
        mlp_out = self._mlp(ln2, B, sharded=True)
        return ttnn.add(h, mlp_out)

    def _shard_kv(self, kv, B):
        nkv = self.cfg.num_kv_heads
        nkv32 = ((nkv + TILE - 1) // TILE) * TILE
        row = 8
        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord((B - 1) % row, (B - 1) // row))}
        )
        mem = ttnn.create_sharded_memory_config(
            shape=(nkv32, self.cfg.head_dim),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.to_memory_config(kv, mem)

    def _rope_decode(self, rope_idx, B, sin=False):
        table = self.sin_2d if sin else self.cos_2d
        rd = self.cfg.rotary_dim
        gathered = ttnn.embedding(rope_idx, table, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(gathered, (1, B, 1, rd))
