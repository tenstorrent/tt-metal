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
import os
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from .prefill_page_table import single_shot_fill_page_table

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

    Defaults are the optimized policy: BF16 activations/norms; BFP8
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
        # Sliding-window RoPE params. Config layout drifted across vLLM/transformers versions: the fork stack
        # (vLLM 0.16) exposed a top-level `swa_rope_parameters` dict; on stock vLLM 0.24 / transformers 5.11 that
        # attr is None and the sliding params live under `rope_parameters["sliding_attention"]` (mirroring the
        # `full_attention` branch below). Tolerate both.
        swa = getattr(hf_config, "swa_rope_parameters", None) or rp.get("sliding_attention")
        cfg.rope_parameters = dict(swa)
        cfg.partial_rotary_factor = cfg.rope_parameters.get("partial_rotary_factor", 1.0)
    else:
        cfg.rope_parameters = dict(rp["full_attention"])
        # Must force the top-level partial_rotary_factor too: on stock vLLM 0.24 the loaded config carries a
        # top-level `partial_rotary_factor` (=1.0) that HF's RotaryEmbedding prefers over the nested
        # rope_parameters value, so without this the full-attention table is built 128-wide (head_dim) while
        # LayerConfig.rotary_dim is 64 (head_dim*0.5) -> _rope_decode reshape TT_FATAL new_volume!=old_volume.
        cfg.partial_rotary_factor = cfg.rope_parameters.get("partial_rotary_factor", 1.0)
    re = RE(config=cfg)
    pos = torch.arange(max_seq_len).unsqueeze(0)
    dummy = torch.zeros(1, max_seq_len, 1)
    cos, sin = re(dummy, pos)
    return cos[0].float(), sin[0].float()


# --------------------------------------------------------------------------- #
# Device-weight disk cache (plan Stage 0)                                       #
#                                                                               #
# Boot re-converts every weight from safetensors on each launch (fp32 upcast,   #
# torch.stack of the 256 MoE experts across 39 layers, then tilize/shard to     #
# device — ~18 min). ``ttnn.as_tensor(..., cache_file_name=key)`` writes the    #
# CONVERTED ttnn tensor to disk once and mmap's it on later boots. Two ttnn     #
# semantics (verified in ttnn/ttnn/operations/core.py + core/tensor/            #
# serialization.cpp) drive the design:                                          #
#   1. On a cache HIT as_tensor only skips the *tilize* — the CALLER has already #
#      built the torch source. So the expensive source build (esp. the expert   #
#      torch.stack) is threaded as a THUNK and guarded by an os.path.exists()    #
#      check here, so a hit truly avoids the stack, not just the tilize.         #
#   2. load_tensor_flatbuffer places the tensor with the flatbuffer's STORED     #
#      (DRAM-interleaved) memory_config — a custom width-sharded memory_config   #
#      is NOT round-tripped. So only the interleaved copy is cached; the "_ds"   #
#      DRAM-width-sharded copy is derived on-device via to_memory_config (also   #
#      cheaper: it reshards the already-tilized weight instead of re-tilizing).  #
# Kill switch: TT_LAGUNA_WEIGHT_CACHE_DISABLE=1 -> plain from_torch path (A/B).  #
# --------------------------------------------------------------------------- #
_WEIGHT_CACHE_DISABLE = os.environ.get("TT_LAGUNA_WEIGHT_CACHE_DISABLE", "0") == "1"


def _weight_cache_dir() -> str:
    default = os.path.join(os.path.expanduser("~"), ".cache", "ttnn", "laguna_xs_2_1")
    d = os.environ.get("TT_LAGUNA_WEIGHT_CACHE", default)
    os.makedirs(d, exist_ok=True)
    return d


def weight_cache_key(name: str, layer_idx, mesh_tag: str):
    """Cache basename for a device weight, or ``None`` when caching is disabled.

    ``ttnn.as_tensor`` appends ``_dtype_{DTYPE}_layout_{LAYOUT}.tensorbin``, so the
    produced filename already encodes the dtype (bfp8/bfp4/bf16) and layout. This
    base adds everything else that changes the produced bytes: the weight ``name``,
    the LAYER INDEX, and ``mesh_tag`` (replicate vs shard-dim-N, plus the device
    count D). Two layers, two dtypes, or two mesh mappings can never collide."""
    if _WEIGHT_CACHE_DISABLE:
        return None
    return os.path.join(_weight_cache_dir(), f"L{layer_idx}_{name}_{mesh_tag}")


def _cached_device_tensor(build, *, device, dtype, layout, memory_config, mesh_mapper, cache_key):
    """``from_torch``-equivalent backed by an on-disk ttnn cache.

    ``build`` is a zero-arg thunk returning the source torch tensor (already in ttnn
    [in, out] orientation). On a cache HIT the thunk is NOT called — the converted
    ttnn tensor is mmap'd from disk, skipping BOTH the source build (safetensors
    upcast + expert torch.stack) and the tilize. On MISS/DISABLED the thunk runs and
    the result is tilized/sharded (and dumped, unless disabled)."""
    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    if cache_key is None:  # disabled / uncacheable -> original path
        return ttnn.from_torch(
            build(), dtype=dtype, layout=layout, device=device, memory_config=memory_config, mesh_mapper=mesh_mapper
        )
    # Mirror the exact filename as_tensor writes so the existence guard is accurate.
    full = f"{cache_key}_dtype_{dtype.name}_layout_{layout.name}.tensorbin"
    if os.path.exists(full):
        try:
            return ttnn.load_tensor(full, device=device)  # HIT: no torch build, no tilize
        except Exception:  # corrupt/stale cache -> fall through and rebuild + overwrite
            pass
    return ttnn.as_tensor(
        build(),
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=mesh_mapper,
        cache_file_name=cache_key,
    )


def _as_tt(src, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=None, cache_key=None):
    """``src`` is a torch tensor OR a zero-arg thunk building one (thunk form lets a
    cache hit skip an expensive source build). Single-chip (no mesh mapper)."""
    build = src if callable(src) else (lambda: src)
    return _cached_device_tensor(
        build,
        device=device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=None,
        cache_key=cache_key,
    )


def _linear_w(t, device, dtype=ttnn.bfloat16, memory_config=None, cache_key=None):
    """HF linear weight [out, in] -> ttnn [in, out]."""
    return _as_tt(lambda: t.t().contiguous(), device, dtype, memory_config=memory_config, cache_key=cache_key)


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

        # Prefill single-shot boundary. Prompts with seq > PIPE_CHUNK take the bounded-footprint
        # chunked path (_prefill_pipelined, chunk = PIPE_CHUNK). The class default (8192) allocates
        # full-length prefill activations under the resident decode trace, which wedges the allocator
        # (allocator.cpp:123) at seq >= 4096 regardless of chip count. Lower it (env
        # TT_LAGUNA_PIPE_CHUNK=2048) to route every prefill > 2048 onto the chunked path so its
        # per-chunk activation stays at the proven-safe 2048 size. warmup_model_prefill compiles the
        # chunk program pre-trace via its 4096/8192 buckets.
        self.PIPE_CHUNK = int(os.environ.get("TT_LAGUNA_PIPE_CHUNK", OptimizedDecoder.PIPE_CHUNK))
        self.PREFILL_SDPA_CHUNK = int(
            os.environ.get("TT_LAGUNA_PREFILL_SDPA_CHUNK", OptimizedDecoder.PREFILL_SDPA_CHUNK)
        )

        # Cold-prefill fast path (env-gated, default OFF == current bounded behavior).
        # _prefill_pipelined re-reads the growing KV prefix from the paged DRAM cache once per
        # OUTER chunk (chunked_scaled_dot_product_attention, chunk_start_idx=gpos attends to the
        # full cache [0:gpos+ch]). Total prefix DRAM traffic ~ kv_bytes * seq^2 / (2 * CH), i.e.
        # inversely proportional to the outer chunk CH. Serving forces TT_LAGUNA_PIPE_CHUNK=2048;
        # the pre-regression sweep effectively ran CH=8192 (the env knob did not exist yet), so the
        # 4x-smaller chunk quadrupled the redundant long-context prefix re-read -> ~2.4x slower cold
        # TTFT at 131072. TT_LAGUNA_PREFILL_FAST=1 restores the larger OUTER chunk
        # (TT_LAGUNA_PREFILL_FAST_CHUNK, default 8192 == the proven-safe pre-regression size) to cut
        # that re-read ~4x. It does NOT change the single-shot branch threshold (PIPE_CHUNK) or any
        # per-op numerics -- only how the >PIPE_CHUNK prefill is sub-chunked. Per-chunk activation
        # stays bounded to FAST_CHUNK (no full-length allocation), so use only when that width is
        # memory-safe for the served context (validated <=131072 on P150x4; do NOT enable >131072
        # without an OOM re-check).
        self.PREFILL_FAST = os.environ.get("TT_LAGUNA_PREFILL_FAST", "0") == "1"
        self.PREFILL_FAST_CHUNK = int(os.environ.get("TT_LAGUNA_PREFILL_FAST_CHUNK", 8192))

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
        # Accuracy-safe fast NORMAL-decode SDPA config (Stage 2 fix). Identical to _sdpa_pc EXCEPT
        # k_chunk_size=64 instead of 128. Root cause of the k128 lossiness (teacher 0.95->0.58, layer PCC
        # -0.016): k128's last-partial-chunk masking at low non-128-aligned cur_pos, NOT precision. The
        # long-context SPEED comes from the (unset -> default 16) max_cores_per_head_batch parallel KV
        # scan, which k64 keeps. _sdpa_pc (k128) is retained for from-scratch prefill; the spec-decode
        # verify now uses _sdpa_pc_verify (k64 default, TT_LAGUNA_VERIFY_K) — the old "verify REQUIRES k128"
        # claim was DISPROVEN on device (k64 verify runs correctly; k128 is only kept as an A/B option).
        # Sweep knobs (Stage A): defaults reproduce the shipped config (k64, exp off, max_cores unset ->
        # ttnn default 16). TT_LAGUNA_DECODE_K / _EXP / _MAXCORES let a config sweep pick the fastest
        # PCC-safe decode config without a per-run source edit. Leave unset in production.
        _dec_k = int(os.environ.get("TT_LAGUNA_DECODE_K", "64"))
        _dec_exp = os.environ.get("TT_LAGUNA_DECODE_EXP", "0") == "1"
        _dec_mc = os.environ.get("TT_LAGUNA_DECODE_MAXCORES", "")
        _dec_kwargs = dict(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=_dec_k,
            exp_approx_mode=_dec_exp,
        )
        if _dec_mc != "":
            _dec_kwargs["max_cores_per_head_batch"] = int(_dec_mc)
        self._sdpa_pc_decode = ttnn.SDPAProgramConfig(**_dec_kwargs)
        # Spec-decode VERIFY SDPA. ACCURACY FIX (2026-08-06): the verify historically used _sdpa_pc (k128) —
        # the SAME config proven LOSSY on normal decode (teacher top1 0.95->0.58 from last-partial-chunk
        # masking at non-128-aligned cur_pos). Since committed spec tokens ARE the verify's argmax, a k128
        # verify makes spec-decode inherit that lossy trajectory (an accuracy regression vs k64 serving).
        # TT_LAGUNA_VERIFY_K selects the verify k_chunk (A/B k64 vs k128 on device without a source edit).
        # DEFAULT 64: aligns spec-verify with the accurate normal-decode SDPA (teacher top1 0.95 vs k128's
        # 0.58), so committed spec tokens are drawn from the same numerics as non-spec decode. Validated on
        # device (2026-08-05): served HumanEval-164 k64==k128 (83/164, 0 pass/fail flips) and throughput
        # identical (1099s vs 1102s) — Pareto win (more accurate config, zero HumanEval/throughput cost).
        self._verify_k = int(os.environ.get("TT_LAGUNA_VERIFY_K", "64"))
        self._sdpa_pc_verify = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=self._verify_k,
            exp_approx_mode=False,
        )
        # Chunked prefix-cache read (suffix prefill, start_pos>0): chunk_start_idx
        # must be a multiple of q_chunk_size AND k_chunk_size. start_pos is only
        # guaranteed a multiple of block_size (64), so keep both <= 64 divisors of
        # it (128 would FATAL for odd multiples of 64, e.g. K=320).
        self._sdpa_pc_chunked = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            q_chunk_size=32,
            k_chunk_size=64,
            exp_approx_mode=False,
        )
        self._sdpa_compute = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # DEFAULT ON: apply a decode SDPA program config with the max_cores_per_head_batch=16 parallel
        # KV scan (the long-context speed: 128k 151.7→36.2 ms/tok, 4.2×). The NORMAL decode uses
        # self._sdpa_pc_decode (k_chunk=64); the spec-decode verify uses self._sdpa_pc (k_chunk=128).
        # HISTORY: k_chunk=128 on the normal decode was LOSSY (teacher top1 0.95→0.58, layer PCC -0.016 at
        # low non-aligned cur_pos — k128 last-partial-chunk masking, not precision); the earlier
        # "bit-identical greedy" claim was WRONG. Stage-2 fix (2026-07-31) drops the normal decode to k64,
        # which keeps the max_cores=16 speed without the k128 masking hazard. VALIDATED on device
        # (2026-08-03, full_model_checks teacher, weight-cache off): k64 teacher top1 = 0.95 (top5/100 = 1.00)
        # vs k128's 0.58 — see doc/vllm_integration/decode_config_sweep/.
        # Set TT_LAGUNA_DECODE_SDPA_PC=0 to fall back to the ttnn default decode (k32+exp_approx, max_cores=1,
        # accurate but slow @long-ctx).
        self._decode_use_sdpa_pc = os.environ.get("TT_LAGUNA_DECODE_SDPA_PC", "1") == "1"
        # DEFAULT ON (validated bit-identical greedy tokens vs baseline @B=1 AND @B=32, @4k+128k): fuse MoE
        # expert-combine reduction (ttnn.sum(dim=1) → deepseek_moe_fast_reduce_nc). TT_LAGUNA_FUSED_REDUCE=0 reverts.
        self._use_fused_reduce = os.environ.get("TT_LAGUNA_FUSED_REDUCE", "1") == "1"
        # DEFAULT ON (validated bit-identical @B=1 AND @B=32): fused HF rotate_half decode RoPE
        # (rotary_embedding_hf). Part of the +~6% combined decode win. TT_LAGUNA_FUSED_ROPE=0 reverts.
        self._use_fused_rope = os.environ.get("TT_LAGUNA_FUSED_ROPE", "1") == "1"
        # W3a: fused decode QKV head-split output template (HEIGHT_SHARDED, one user/core). The op
        # (nlp_create_qkv_heads_decode) derives q/k/v shard specs from this; grid must cover >= B cores.
        # Mirrors attention_1d.py Blackhole decode: 32 cores, shard (TILE, head_dim). Gated by
        # TT_LAGUNA_FUSE_QKV_DECODE (default off until fast-loop PCC + timing validate).
        self._fuse_qkv_decode = os.environ.get("TT_LAGUNA_FUSE_QKV_DECODE", "0") == "1"
        self._qkv_heads_decode_memcfg = ttnn.create_sharded_memory_config(
            shape=(TILE, self.cfg.head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # fidelity assignment per matmul group — READ FROM THE POLICY so the selected
        # compute-fidelity config is actually consumed by the measured runtime path
        # (defaults preserve the optimized policy: LoFi projections, HiFi2 gate/router).
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

        def ckey(name):  # single-chip (1x1) mesh tag
            return weight_cache_key(name, layer_idx, "sc")

        def store(key, w_in, k, n, dtype):
            """Store an interleaved copy (``key``, for prefill 2D matmuls) and a
            DRAM-width-sharded copy (``key+"_ds"``, for decode DRAM-sharded matmuls).
            ``w_in`` is already in ttnn [in, out] orientation. Only the interleaved
            copy is disk-cached; the ``_ds`` copy is derived on-device via
            to_memory_config (a custom width-sharded memory_config is not preserved
            through the ttnn tensor cache — see _cached_device_tensor)."""
            w[key] = _as_tt(w_in, dev, dtype, cache_key=ckey(key))
            w[key + "_ds"] = ttnn.to_memory_config(w[key], _dram_weight_memcfg(k, n, dram_cores))

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
        w["wg"] = _linear_w(g("self_attn.g_proj.weight"), dev, policy.attn_gate, cache_key=ckey("wg"))
        w["q_norm"] = _as_tt(
            g("self_attn.q_norm.weight").reshape(1, 1, 1, cfg.head_dim), dev, policy.qk_norm, cache_key=ckey("q_norm")
        )
        w["k_norm"] = _as_tt(
            g("self_attn.k_norm.weight").reshape(1, 1, 1, cfg.head_dim), dev, policy.qk_norm, cache_key=ckey("k_norm")
        )
        w["input_ln"] = _as_tt(g("input_layernorm.weight").reshape(1, 1, 1, H), dev, cache_key=ckey("input_ln"))
        w["post_ln"] = _as_tt(g("post_attention_layernorm.weight").reshape(1, 1, 1, H), dev, cache_key=ckey("post_ln"))

        if cfg.is_moe:
            E, I = cfg.num_experts, cfg.moe_intermediate
            w["gate_w"] = _linear_w(g("mlp.gate.weight"), dev, policy.router, cache_key=ckey("gate_w"))  # [H, E]
            w["e_bias"] = _as_tt(
                lambda: g("mlp.experts.e_score_correction_bias").reshape(1, 1, 1, E), dev, cache_key=ckey("e_bias")
            )

            # The 256-expert torch.stack is the dominant boot cost, so build it lazily
            # inside the cache thunk — a HIT skips the stack (and the safetensors
            # upcast) entirely, not just the tilize.
            def _stack(proj):
                return torch.stack([g(f"mlp.experts.{i}.{proj}.weight") for i in range(E)])

            w["exp_gate"] = _as_tt(
                lambda: _stack("gate_proj").transpose(1, 2).reshape(1, E, H, I),
                dev,
                policy.moe_ff13,
                cache_key=ckey("exp_gate"),
            )
            w["exp_up"] = _as_tt(
                lambda: _stack("up_proj").transpose(1, 2).reshape(1, E, H, I),
                dev,
                policy.moe_ff13,
                cache_key=ckey("exp_up"),
            )
            w["exp_down"] = _as_tt(
                lambda: _stack("down_proj").transpose(1, 2).reshape(1, E, I, H),
                dev,
                policy.moe_ff2,
                cache_key=ckey("exp_down"),
            )
            sh_I = cfg.shared_intermediate
            store("sh_gate", g("mlp.shared_expert.gate_proj.weight").t().contiguous(), H, sh_I, policy.shared_ff13)
            store("sh_up", g("mlp.shared_expert.up_proj.weight").t().contiguous(), H, sh_I, policy.shared_ff13)
            store("sh_down", g("mlp.shared_expert.down_proj.weight").t().contiguous(), sh_I, H, policy.shared_ff2)
        else:
            II = cfg.intermediate
            store("mlp_gate", g("mlp.gate_proj.weight").t().contiguous(), H, II, policy.dense_ff13)
            store("mlp_up", g("mlp.up_proj.weight").t().contiguous(), H, II, policy.dense_ff13)
            store("mlp_down", g("mlp.down_proj.weight").t().contiguous(), II, H, policy.dense_ff2)

        # shared per-attention-kind RoPE tables (build once, reuse) — see MultichipDecoder.
        rope_tables = kwargs.get("rope_tables")
        kind = cfg.attention_type
        if rope_tables is not None and kind in rope_tables:
            cos_2d, sin_2d = rope_tables[kind]
        else:
            cos, sin = _hf_rope_tables(hf_config, kind, max_seq_len)
            cos_2d = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            sin_2d = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            if rope_tables is not None:
                rope_tables[kind] = (cos_2d, sin_2d)

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
        # REVERTED: an on-device zeros alloc (ttnn.zeros bf16 -> typecast BFP8) breaks paged
        # DECODE reads of the persistent cache (test_decode_pcc PCC 0.5/-0.05 vs 0.995; prefill unaffected
        # because it writes+reads within one call). ttnn.zeros has no BFP8 support, and a typecast-origin
        # BFP8 tensor is not a valid in-place paged-cache buffer. Kept as host torch.zeros + from_torch
        # (the canonical BFP8 cache format) until a native device BFP8-zeros path exists.
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
    def _qkv_roped(self, ln, seq, start_pos, rope=None):
        cfg = self.cfg
        qkv = ttnn.linear(ln, self.w["wqkv"], compute_kernel_config=self._ck_qkv)  # [1,seq,qkv_w] (seq in dim1)
        qkv = ttnn.reshape(qkv, (1, 1, seq, self.meta["qkv_w"]))
        # fused prefill head-split — one op replaces 3x slice + 3x reshape + 3x permute.
        # Packed wqkv is [all-Q | all-K | all-V] exactly as this op expects; emits q[1,nh,seq,hd],
        # k/v[1,nkv,seq,hd] — same layout as the old slice+reshape+permute, so per-head RMSNorm (over
        # head_dim) and _apply_rope downstream are unchanged. Validated: multichip layer PCC 65/65.
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = self._per_head_norm(q, self.w["q_norm"])
        k = self._per_head_norm(k, self.w["k_norm"])
        if rope is None:  # local fallback (pipelined per-chunk positions, layer PCC tests, direct callers)
            cos = self._rope_prefill(start_pos, seq)
            sin = self._rope_prefill(start_pos, seq, sin=True)
        else:  # model-hoisted shared per-kind context
            cos, sin = rope
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)
        return q, k, v

    def _cast_fill(self, t, cache_dtype):
        return ttnn.typecast(t, cache_dtype) if t.dtype != cache_dtype else t

    @property
    def _prefill_pipe_chunk(self):
        """Outer chunk size for _prefill_pipelined. Default == live PIPE_CHUNK (byte-identical to the
        bounded path, and honours a test/runtime-mutated PIPE_CHUNK). TT_LAGUNA_PREFILL_FAST=1 swaps in
        the larger PREFILL_FAST_CHUNK to cut the redundant KV-prefix re-read on long cold prefills."""
        return self.PREFILL_FAST_CHUNK if self.PREFILL_FAST else self.PIPE_CHUNK

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
        if runtime_offsets is not None:
            raise ValueError("runtime-offset prefill is supported only by the P150x2 multichip decoder")
        fill_page_table = page_table if fill_page_table is None else fill_page_table
        seq = x_BSH.shape[-2]
        if seq > self.PIPE_CHUNK:
            return self._prefill_pipelined(
                x_BSH,
                kv_cache,
                page_table,
                fill_page_table,
                user_id,
                start_pos,
                fill_page_table_base_pos=fill_page_table_base_pos,
            )
        cfg = self.cfg
        residual = x_BSH
        ln = self._rms(x_BSH, self.w["input_ln"])
        q, k, v = self._qkv_roped(ln, seq, start_pos, rope=rope_mats)
        cdt = kv_cache["dtype"]
        fill_pt = single_shot_fill_page_table(
            fill_page_table,
            start_pos=start_pos,
            seq_len=seq,
            block_size=kv_cache["block_size"],
            fill_page_table_base_pos=fill_page_table_base_pos,
        )
        ttnn.experimental.paged_fill_cache(kv_cache["k"], self._cast_fill(k, cdt), fill_pt, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(kv_cache["v"], self._cast_fill(v, cdt), fill_pt, batch_idx=user_id)
        attn = self._prefill_attention(q, k, v, kv_cache, page_table, user_id, start_pos, seq)
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, (1, seq, cfg.num_heads * cfg.head_dim))
        attn = self._gate(attn, ln)
        o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._ck_o)
        h = ttnn.add(residual, o)
        ln2 = self._rms(h, self.w["post_ln"])
        mlp_out = self._mlp(ln2, seq, sharded=False)
        mlp_out = ttnn.reshape(mlp_out, (1, seq, cfg.hidden))
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
    ):
        cfg = self.cfg
        seq = x_BSH.shape[-2]
        bs = kv_cache["block_size"]
        cdt = kv_cache["dtype"]
        CH = (self._prefill_pipe_chunk // bs) * bs  # env-gated outer chunk (TT_LAGUNA_PREFILL_FAST)
        win = cfg.sliding_window
        k_tail = v_tail = None
        outs = []
        fill_base = int(fill_page_table_base_pos)
        if int(start_pos) < fill_base or (int(start_pos) - fill_base) % bs:
            raise ValueError(
                f"prefill start {start_pos} and fill page-table base {fill_base} are not block aligned"
            )
        for c in range(0, seq, CH):
            ch = min(CH, seq - c)
            gpos = start_pos + c
            xc = ttnn.slice(x_BSH, [0, c, 0], [1, c + ch, cfg.hidden])
            residual = xc
            ln = self._rms(xc, self.w["input_ln"])
            q, k, v = self._qkv_roped(ln, ch, gpos)
            col0 = (gpos - fill_base) // bs
            ncol = (ch + bs - 1) // bs
            chunk_pt = ttnn.slice(
                fill_page_table,
                [0, col0],
                [fill_page_table.shape[0], col0 + ncol],
            )
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
            attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn = ttnn.reshape(attn, (1, ch, cfg.num_heads * cfg.head_dim))
            attn = self._gate(attn, ln)
            o = ttnn.linear(attn, self.w["wo"], compute_kernel_config=self._ck_o)
            h = ttnn.add(residual, o)
            ln2 = self._rms(h, self.w["post_ln"])
            mlp_out = ttnn.reshape(self._mlp(ln2, ch, sharded=False), (1, ch, cfg.hidden))
            outs.append(ttnn.add(h, mlp_out))
        return ttnn.concat(outs, dim=1)

    def _prefill_attention(
        self,
        q,
        k,
        v,
        kv_cache,
        page_table,
        user_id,
        start_pos,
        seq,
        *,
        chunk_start_idx_tensor=None,
    ):
        cfg = self.cfg
        base = {"scale": cfg.scaling, "compute_kernel_config": self._sdpa_compute}
        if seq <= self.PREFILL_SDPA_CHUNK:
            if start_pos == 0:
                # From-scratch: the whole sequence is local, so a single
                # self-contained SDPA over q/k/v is correct and cheapest.
                kw = {"is_causal": True, "program_config": self._sdpa_pc, **base}
                if cfg.is_sliding:
                    kw["sliding_window_size"] = cfg.sliding_window
                return ttnn.transformer.scaled_dot_product_attention(q, k, v, **kw)
            # Prefix-cache suffix prefill (start_pos>0): a local SDPA over just the
            # suffix q/k/v would miss the cached prefix [0:start_pos). Read the paged
            # cache instead -- K/V for [start_pos:start_pos+seq] were just
            # paged_fill_cache'd by the caller, [0:start_pos) is the cached prefix.
            # Works for full and sliding (ttnn chunked SDPA composes with
            # sliding_window_size). Uses the 64-aligned chunked program config so
            # chunk_start_idx (a multiple of block_size) is valid.
            user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
            kw = {
                "program_config": self._sdpa_pc_chunked,
                "compute_kernel_config": self._sdpa_compute,
            }
            if chunk_start_idx_tensor is None:
                kw["chunk_start_idx"] = start_pos
            else:
                kw["chunk_start_idx_tensor"] = chunk_start_idx_tensor
            if cfg.is_sliding:
                kw["sliding_window_size"] = cfg.sliding_window
            return ttnn.transformer.chunked_scaled_dot_product_attention(q, kv_cache["k"], kv_cache["v"], user_pt, **kw)
        CH = self.PREFILL_SDPA_CHUNK
        if chunk_start_idx_tensor is not None:
            raise ValueError(
                "runtime-offset single-shot prefill cannot use the inner Q-chunk path; "
                "set PIPE_CHUNK <= PREFILL_SDPA_CHUNK"
            )
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

    def _rope_prefill_indexed(self, position_ids, sin=False, output_tensor=None):
        """Gather prefill RoPE rows from runtime uint32 IDs into a persistent TILE output."""
        table = self.sin_2d if sin else self.cos_2d
        rd = self.cfg.rotary_dim
        seq = int(position_ids.shape[-1])
        gathered = ttnn.embedding(
            position_ids,
            table,
            layout=ttnn.TILE_LAYOUT,
            output_tensor=output_tensor,
        )
        return ttnn.reshape(gathered, (1, 1, seq, rd))

    # ---- decode ------------------------------------------------------------ #
    def decode_forward(self, x_1BH, cur_pos, rope_idx, page_table, kv_cache, sequential_kv_write=False, rope_mats=None):
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
        if self._fuse_qkv_decode:  # W3a: fused head-split replaces _split_qkv (3 slice + 3 reshape).
            # s2i all three so the UNCHANGED per_head_norm / rope / _shard_kv tail runs byte-identical
            # (Stage 1: isolate "does the fused op emit the same q/k/v as _split_qkv?").
            qkv = ttnn.reshape(qkv, (1, 1, B, self.meta["qkv_w"]), (1, 1, TILE, self.meta["qkv_w"]))
            q_sh, k_sh_raw, v_sh_raw = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv,
                num_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                overlap_qk_coregrid=True,
                memory_config=self._qkv_heads_decode_memcfg,
            )
            q = ttnn.sharded_to_interleaved(q_sh, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.sharded_to_interleaved(k_sh_raw, ttnn.DRAM_MEMORY_CONFIG)
            v = ttnn.sharded_to_interleaved(v_sh_raw, ttnn.DRAM_MEMORY_CONFIG)
        else:
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
        if self._use_fused_rope:  # gated: fused HF rotate_half decode RoPE — see MultichipDecoder.decode_forward
            cos_sh = self._shard_cossin(cos, B, cfg.rotary_dim)
            sin_sh = self._shard_cossin(sin, B, cfg.rotary_dim)
            q = self._fused_rope_decode(q, cos_sh, sin_sh, cfg.num_heads, B)
            k = self._fused_rope_decode(k, cos_sh, sin_sh, cfg.num_kv_heads, B)
        else:
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
        k_sh = self._shard_kv(k, B)
        v_sh = self._shard_kv(v, B)
        if sequential_kv_write and B > 1:  # spec-decode verify: serialize shared-block writes (see _seq_kv_write)
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
        if sequential_kv_write:  # spec-decode verify. k_chunk via TT_LAGUNA_VERIFY_K (default 64 = accurate SDPA)
            sdpa_kwargs["program_config"] = self._sdpa_pc_verify
            sdpa_kwargs["num_kv_heads"] = cfg.num_kv_heads
        elif self._decode_use_sdpa_pc:  # Stage 2: accuracy-safe fast normal-decode config (k64, max_cores=16)
            sdpa_kwargs["program_config"] = self._sdpa_pc_decode
            sdpa_kwargs["num_kv_heads"] = cfg.num_kv_heads
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

    def _seq_kv_write(self, cache, kv_sh, cur_pos, page_table, B):
        """Serialized per-row paged_update_cache for the spec-decode VERIFY batch.

        The B batch rows are candidate tokens sharing ONE user's physical KV blocks (their page-table
        rows are identical). Because BLOCK_SIZE==TILE==32, consecutive candidate positions fall in the
        same block-tile, so a single batched paged_update_cache has every row read-modify-write that tile
        concurrently → last-writer-wins clobbering (KV corruption). Writing each row with its own ordered
        op serializes the RMW. ``kv_sh`` is the already-batch-sharded [1,B,nkv32,hd] tensor
        (``_shard_kv`` output — the exact layout paged_update_cache accepts); slice each row from DRAM and
        reshard onto a single core (the per-user layout the op expects). Mirrors
        ``models/demos/gemma4/tt/attention/decode.py:147-201``."""
        nkv32 = ((self.cfg.num_kv_heads + TILE - 1) // TILE) * TILE
        one_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        single_mem = ttnn.create_sharded_memory_config(
            shape=(nkv32, self.cfg.head_dim),
            core_grid=one_core,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        kv_dram = ttnn.to_memory_config(kv_sh, ttnn.DRAM_MEMORY_CONFIG)
        nk, hd = kv_dram.shape[-2], kv_dram.shape[-1]
        # Pass block_size + num_kv_heads EXPLICITLY (as gemma4/tt/attention/decode.py does). The kernel
        # defaults infer geometry from the cache tensor; under TRACE that inference mis-handled a
        # populated block (the writing row's own position read back wrong — verified via selfcheck),
        # while fresh-block writes were exact. cache shape = (num_blocks, num_kv_heads, block_size, hd).
        blk = int(cache.shape[2])
        nkvh = int(cache.shape[1])
        for b in range(B):
            kb = ttnn.slice(kv_dram, [0, b, 0, 0], [1, b + 1, nk, hd])
            kb = ttnn.to_memory_config(kb, single_mem)
            pos_b = ttnn.slice(cur_pos, [b], [b + 1])
            pt_b = ttnn.slice(page_table, [b, 0], [b + 1, page_table.shape[1]])
            ttnn.experimental.paged_update_cache(
                cache, kb, update_idxs_tensor=pos_b, page_table=pt_b, block_size=blk, num_kv_heads=nkvh
            )
            for t in (kb, pos_b, pt_b):
                t.deallocate(True)
        kv_dram.deallocate(True)

    def _shard_batch(self, x, nheads, width, B):
        """Height-shard [1,B,nheads,width] across B cores (one user's (nheads,width) block/core) — the
        HEIGHT_SHARDED layout the fused decode RoPE requires. Mirrors _shard_kv."""
        nh32 = ((nheads + TILE - 1) // TILE) * TILE
        row = 8
        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord((B - 1) % row, (B - 1) // row))}
        )
        mem = ttnn.create_sharded_memory_config(
            shape=(nh32, width),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.to_memory_config(x, mem)

    def _shard_cossin(self, t, B, rd):
        """Height-shard [1,B,1,rd] cos/sin across B cores for the fused decode RoPE."""
        row = 8
        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord((B - 1) % row, (B - 1) // row))}
        )
        mem = ttnn.create_sharded_memory_config(
            shape=(TILE, rd),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.to_memory_config(t, mem)

    def _fused_rope_decode(self, x, cos_sh, sin_sh, nheads, B):
        """Fused HF rotate_half decode RoPE for one of q/k. Partial rotary (rd<hd, full-attn layers)
        handled by slicing the rotary lanes, fusing, and concatenating the pass-through — numerically
        identical to _apply_rope (both HF rotate_half over the same cos/sin)."""
        cfg = self.cfg
        rd, hd = cfg.rotary_dim, cfg.head_dim
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        if rd == hd:
            x_rot, x_pass = x, None
        else:
            x_rot = ttnn.slice(x, [0, 0, 0, 0], [1, B, nheads, rd])
            x_pass = ttnn.slice(x, [0, 0, 0, rd], [1, B, nheads, hd])
        x_sh = self._shard_batch(x_rot, nheads, rd, B)
        out_sh = ttnn.experimental.rotary_embedding_hf(x_sh, cos_sh, sin_sh, is_decode_mode=True)
        out = ttnn.sharded_to_interleaved(out_sh, ttnn.DRAM_MEMORY_CONFIG)
        if x_pass is None:
            return out
        return ttnn.concat([out, x_pass], dim=-1)

    def _rope_decode(self, rope_idx, B, sin=False):
        table = self.sin_2d if sin else self.cos_2d
        rd = self.cfg.rotary_dim
        gathered = ttnn.embedding(rope_idx, table, layout=ttnn.TILE_LAYOUT)
        if os.environ.get("TT_LAGUNA_ROPE_DEBUG") == "1":
            import sys as _sys

            print(
                f"[ROPE_DEBUG] attn={self.cfg.attention_type} B={B} rd={rd} "
                f"rope_idx.shape={list(rope_idx.shape)} table.shape={list(table.shape)} "
                f"gathered.shape={list(gathered.shape)} target=(1,{B},1,{rd}) "
                f"gathered_vol={gathered.logical_volume() if hasattr(gathered,'logical_volume') else '?'} "
                f"target_vol={1*B*1*rd}",
                file=_sys.stderr,
                flush=True,
            )
        return ttnn.reshape(gathered, (1, B, 1, rd))
