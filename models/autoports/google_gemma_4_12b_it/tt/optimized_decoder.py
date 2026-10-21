# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized TTNN decoder layer for ``google/gemma-4-12B-it`` (text decoder, one Blackhole chip).

Same layer, constructor and runtime contract as ``fused_decoder.FusedDecoder`` (and therefore
``functional_decoder.FunctionalDecoder``):
- HF reference ``Gemma4UnifiedTextDecoderLayer``, both ``sliding_attention`` and ``full_attention`` kinds;
- paged KV cache owned by the layer;
- any logical prefill length;
- trace-safe decode of 1..32 users, where ``-1`` skips a user.
The fused op graph is kept; this stage changes precision, layouts and program configs.
``doc/optimized_decoder/README.md`` has the measurements behind every default below.

Precision (``DEFAULT_POLICY`` plus the per-layer exceptions of ``layer_policy``)
* BF16 activations, residual and norms; BFP8 weights with LoFi matmuls; BFP8 KV cache (prefill casts the
  K/V it writes, decode updates take BF16). Layers 13-26 and 35-39 hold BFP4 MLP weights, 13-20, 22, 23
  and 25 BFP4 attention weights, 24, 36 and 37 a BFP4 QKV; layer 47 keeps HiFi2 prefill matmuls.
* The gated MLP uses ttnn's fast GELU on the gate. In prefill chunks on the minimal_matmul path (more than 1280
  rows; 1024 on BFP4-MLP layers) its whole branch up to the residual add is BFP8: gate/up outputs, gated product,
  down output and post-FFN norm output (BF16 on layers 10, 11); the residual stays BF16.
* Decode SDPA and full-layer prefill SDPA keep FP32 accumulation; sliding-layer prefill SDPA runs the streaming
  (BF16-dest) kernel.

Weights
* One copy per projection, width-sharded over the DRAM banks. Decode reads it with the DRAM-sharded
  matmul (one to three reader cores per bank); prefill reads the same tensor with a 2D-multicast matmul
  for chunks of up to 1280 rows (1024 on BFP4-MLP layers) and with ``minimal_matmul`` for larger chunks.

Decode
* The residual stream is width-sharded in L1 (15 cores; 60 in the BFP4 layers). ``decode_forward`` returns
  it in ``decode_input_memory_config()`` and converts any other input layout once on entry.
* RMSNorms run sharded on the residual grid. QKV, o_proj, gate, up and down are DRAM-sharded matmuls
  whose outputs land width-sharded where the next op reads them: the head-split layout for QKV, the
  residual grid for o_proj and down, and the down-projection input grid for gate/up (so the gated
  multiply is a same-spec sharded eltwise op).
* SDPA decode uses 128-token K chunks at batch 1 (page-table rows padded to whole chunks) and one cache
  block per chunk above. ``decode_rope_rows`` lets a layer stack compute the RoPE rows once per step.

Prefill
* Chunked SDPA with larger Q/K chunks at HiFi2: full layers keep FP32 dest and FP32 output accumulation;
  sliding layers use the streaming BF16-dest kernel with approximate exp and a K-chunk-aligned window
  origin. Static circular buffers of every program leave at least 128 KB of L1 per core for persistent
  tensors (``test_l1_headroom``).
"""

import math

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.functional_decoder import HF_PREFIX, MAX_DECODE_BATCH, TILE, rope_tables
from models.autoports.google_gemma_4_12b_it.tt.fused_decoder import (
    DEFAULT_PREFILL_CHUNK,
    SLIDING_ATTN_BLOCK,
    SLIDING_Q_TILE,
    FusedDecoder,
    interleave_perm,
    rot_transformation_mat,
)

FIDELITY = {
    # name: (math fidelity, fp32 dest accumulation, packer L1 accumulation)
    "lofi": (ttnn.MathFidelity.LoFi, False, True),
    "lofifp32": (ttnn.MathFidelity.LoFi, True, True),
    "hifi2": (ttnn.MathFidelity.HiFi2, False, True),
    "hifi2fp32": (ttnn.MathFidelity.HiFi2, True, True),
    "hifi4": (ttnn.MathFidelity.HiFi4, False, True),
    "hifi4fp32": (ttnn.MathFidelity.HiFi4, True, False),
}

# Matmul roles: qkv, o, gate, up, down. Every entry is measured; see README "Chosen configuration".
# fmt: off
DEFAULT_POLICY = {
    # weight dtypes
    "dtype": {"qkv": ttnn.bfloat8_b, "o": ttnn.bfloat8_b, "gate": ttnn.bfloat8_b, "up": ttnn.bfloat8_b, "down": ttnn.bfloat8_b},
    "kv_dtype": ttnn.bfloat8_b,
    # decode matmul fidelity per role, prefill matmul fidelity per role
    "decode_fid": {"qkv": "lofi", "o": "lofi", "gate": "lofi", "up": "lofi", "down": "lofi"},
    "prefill_fid": {"qkv": "lofi", "o": "lofi", "gate": "lofi", "up": "lofi", "down": "lofi"},
    # prefill minimal_matmul blocking per role (``role@kind`` overrides a role for one layer kind):
    # (M_block, K_block, N_block, subblock_h, subblock_w); None = op default. Static CBs stay <= 1.18 MB
    # so persistent L1 tensors (decode residual, trace inputs) keep >= 128 KB per core (test_l1_headroom).
    "prefill_blocks": {
        "qkv": (8, 8, 12, 1, 4),
        "qkv@full": (8, 4, 16, 1, 4),
        "o": (12, 4, 12, 2, 4),
        "gate": (8, 4, 16, 2, 4),
        "up": (8, 4, 16, 2, 4),
        "down": (12, 4, 12, 2, 4),
    },
    # prefill chunks of at most this many rows run their matmuls (except the sliding QKV split) as ttnn.linear
    # with a 2D multicast program config (M over the grid rows, N over the columns), where it beats
    # minimal_matmul by 10-40% at <= 1024 rows and 2-5% at 1152-1280 (per_core_M <= 4); from 1312 rows
    # (per_core_M 5) minimal_matmul with BFP8 MLP intermediates is 3-10% faster (README "Prefill crossover")
    "prefill_linear_max_rows": 1280,
    # 2D-multicast prefill grid width: the most columns whose output subblock is at least this wide (4 = widest
    # subblock first); 3 puts the full-layer QKV (288 N tiles) on 11 columns with 1x3 subblocks instead of 9 with 1x4
    "prefill_2d_sub_w_target": 3,
    # 2D-multicast prefill: in0_block_w cap per role (largest legal divisor up to it within the CB budget)
    "prefill_2d_in0": {"qkv": 8, "o": 16, "gate": 8, "up": 8, "down": 24},
    # sliding-layer QKV: chunks of at most this many rows use the 2D-multicast matmul plus four slices instead of
    # minimal_matmul_split (0 = always the split)
    "prefill_sliding_qkv_linear_max_rows": 384,
    # sliding-layer prefill attention runs per block of this many query rows (FusedDecoder: 2048)
    "sliding_attn_block": SLIDING_ATTN_BLOCK,
    # prefill MLP intermediates in chunks of more than prefill_linear_max_rows rows (the minimal_matmul path),
    # None = BF16 or "bfp8": the gate/up matmul outputs and the gated product; the down matmul and the post-FFN
    # norm inherit the product's dtype, so with "bfp8" the branch is BFP8 up to the (BF16) residual add
    "prefill_mlp_act_dtype": {"gate_up": "bfp8", "product": "bfp8"},
    # prefill: GELU(tanh) as the gate matmul's fused activation instead of the multiply's input activation
    "prefill_gelu_in_gate": False,
    # gate activation applied in the gated multiply: "tanh" (exact gelu_pytorch_tanh) or "fast" (ttnn GELU
    # with its fast approximation; 0.64% vs 0.21% relative error of the gated product, README)
    "gelu": "fast",
    # DRAM reader cores per bank (1..3) per role; the bank shard width must divide by it
    "readers": {"qkv": 1, "o": 1, "gate": 1, "up": 1, "down": 1},
    # decode in0_block_w per role (None: largest legal value within the L1 budget)
    "in0": {"qkv": None, "o": None, "gate": None, "up": None, "down": None},
    "residual_cores": 15,
    # gate/up decode output width per core in tiles: the down projection's input shard
    "mlp_shard_tiles": 5,
    # attention: compute configs and chunk sizes (None: one cache block per K chunk, as in FusedDecoder).
    # K chunks larger than a block pad the page-table row (see _table_for_k_chunk).
    "sdpa_decode_fid": "hifi2fp32",
    # decode SDPA per layer kind and batch, {max batch: value} (smallest key >= batch wins): K chunk (None = one
    # cache block) and cores per (user, KV head). Full layers at batch 1 (one KV head) spread over 24 cores:
    # +3 us at position 4096, -14 us at 32768, -117 us at 262000 vs the op default of 16 (README "Attention").
    "sdpa_decode_k_chunk": {"sliding": {1: 128, MAX_DECODE_BATCH: None}, "full": {1: 128, MAX_DECODE_BATCH: None}},
    "sdpa_decode_max_cores": {"sliding": {MAX_DECODE_BATCH: 16}, "full": {1: 24, MAX_DECODE_BATCH: 16}},
    "sdpa_decode_grid": None,  # (x, y) SDPA decode core grid; None = full grid
    "sdpa_decode_exp_approx": False,
    "sdpa_prefill_fid": {"sliding": "hifi2", "full": "hifi2fp32"},
    "sdpa_prefill_q_chunk": {"sliding": 256, "full": 64},
    "sdpa_prefill_k_chunk": {"sliding": 128, "full": 256},
    "sdpa_prefill_exp_approx": {"sliding": True, "full": False},
    # full layers: prefill SDPA calls whose keys all lie within this many tokens use the streaming BF16-dest kernel
    # (no FP32 output accumulation); 0 = never
    "sdpa_prefill_full_streaming_max_kv": 0,
    # prefill Q chunk: the largest chunk up to the cap that divides the chunk start and the (64-aligned) row count
    # (FusedDecoder rule). A floor (min_rows, q) raises a smaller chunk to q when the SDPA call has more than
    # min_rows rows; the op then pads the last Q chunk internally. None = no floor. Sliding: rows that are 64 mod
    # 128 would otherwise run 64-row Q chunks; above 512 rows a padded 128-row chunk is 2-6% faster per layer
    # (README "Prefill SDPA Q chunk"); full layers already run 64-row chunks.
    "sdpa_prefill_q_floor": {"sliding": (512, 128), "full": None},
}
# fmt: on

# Per-layer weight exceptions (google/gemma-4-12B-it): layers whose real-activation worst-row PCC vs HF
# stays >= 0.998 with BFP4 MLP weights on two different real prompts (README "Per-layer BFP4 MLP"); every
# other layer keeps BFP8 (BFP4 there drops some layers below the 0.995 bar, e.g. 0.979 at layer 6).
MLP_BFP4_GATE_UP_DOWN_LAYERS = frozenset(range(13, 25))
MLP_BFP4_GATE_UP_LAYERS = frozenset({25, 26, 35, 36, 37, 38, 39})
# BFP4 MLP decode geometry: with half the weight bytes, three DRAM readers per bank win (-60 us per layer vs
# one); 24 readers need free worker cores, so the gate/up outputs use 8-tile shards on 60 cores instead of
# the BFP8 default of 5-tile shards on 96. These layers also run the residual on 60 cores (-31 us vs 15),
# while BFP8 layers are fastest on 15; decode_forward reshards once where the two groups meet in a stack.
# With BFP4 MLP weights the 2D prefill path ties minimal_matmul at 1024 rows and loses by up to 2.6% at
# 1152-1280, so these layers switch to minimal_matmul above 1024 rows.
_MLP_BFP4_GEOMETRY = {
    "mlp_shard_tiles": 8,
    "readers": {"gate": 3, "up": 3, "down": 3},
    "in0": {"gate": 4, "up": 4},
    "residual_cores": 60,
    "prefill_linear_max_rows": 1024,
}


# Within the BFP4-MLP group, layers whose attention projections also take BFP4 weights: real-activation worst
# row >= 0.998 on both prompts with BFP4 QKV + o_proj (or QKV only) on top of the BFP4 MLP. With half the
# bytes, more DRAM readers per bank win again (-23 to -44 us per layer): 3 for o_proj and for the full-layer
# QKV (36 tiles per bank), 2 for the sliding QKV (32 tiles per bank; 3 would need padded weights, and was not
# faster in the layer).
ATTN_BFP4_LAYERS = frozenset({13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25})
QKV_BFP4_LAYERS = frozenset({24, 36, 37})
_ATTN_BFP4 = {"dtype": {"qkv": ttnn.bfloat4_b, "o": ttnn.bfloat4_b}, "readers": {"qkv": 2, "o": 3}}
_QKV_BFP4 = {"dtype": {"qkv": ttnn.bfloat4_b}, "readers": {"qkv": 2}}
_ATTN_BFP4_FULL_QKV_READERS = 3

# Layers whose long-chunk prefill MLP intermediates stay BF16: with BFP8 gate/up outputs and product their
# real-activation worst row (4096-token prefill, two prompts) drops by more than 0.0005 (worst over both
# prompts: layer 10: -0.0022, layer 11: -0.0010); elsewhere the drop is <= 0.0005 for -6..-11% layer prefill
# (README "Prefill matmul").
MLP_BF16_INTERMEDIATE_LAYERS = frozenset({10, 11})

# Layers whose prefill matmuls keep HiFi2 instead of LoFi: the last layer has the narrowest margin
# (real-activation prefill worst row 0.99555 with LoFi, 0.99720 with HiFi2; README "Per-layer exceptions").
PREFILL_HIFI2_LAYERS = frozenset({47})


def layer_policy(layer_idx, layer_type="sliding_attention"):
    """Default policy overrides for one checkpoint layer (the per-layer exceptions above)."""
    out = {}
    if layer_idx in MLP_BFP4_GATE_UP_DOWN_LAYERS or layer_idx in MLP_BFP4_GATE_UP_LAYERS:
        roles = ("gate", "up", "down") if layer_idx in MLP_BFP4_GATE_UP_DOWN_LAYERS else ("gate", "up")
        out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _MLP_BFP4_GEOMETRY.items()}
        out["dtype"] = {r: ttnn.bfloat4_b for r in roles}
    if layer_idx in ATTN_BFP4_LAYERS or layer_idx in QKV_BFP4_LAYERS:
        for k, v in (_ATTN_BFP4 if layer_idx in ATTN_BFP4_LAYERS else _QKV_BFP4).items():
            out.setdefault(k, {}).update(v)
        if layer_type == "full_attention":
            out["readers"]["qkv"] = _ATTN_BFP4_FULL_QKV_READERS
    if layer_idx in MLP_BF16_INTERMEDIATE_LAYERS:
        out["prefill_mlp_act_dtype"] = {"gate_up": None, "product": None}
    if layer_idx in PREFILL_HIFI2_LAYERS:
        out["prefill_fid"] = {r: "hifi2" for r in ("qkv", "o", "gate", "up", "down")}
    return out


_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}
_PREFILL_2D_CB_BUDGET = 1_150_000  # estimated static CB bytes of a short-chunk 2D prefill matmul (test_l1_headroom)
_CB_BUDGET = 700_000  # bytes of streaming CBs per DRAM-sharded worker (the rest of L1 holds activations)


def merge_policy(*overrides):
    """DEFAULT_POLICY updated by each override dict in turn (dict fields merge key by key, recursively)."""

    def batch_keyed(d):
        return all(isinstance(k, int) or str(k).isdigit() for k in d)

    def merge(base, over, path=""):
        out = {k: (merge(v, {}) if isinstance(v, dict) else v) for k, v in base.items()}
        for k, v in (over or {}).items():
            # Named fields (roles, kinds, policy keys) must exist; {batch: value} tables accept any batch key.
            assert k in out or batch_keyed(base), f"unknown policy field {path}{k}"
            out[k] = merge(out[k], v, f"{path}{k}.") if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    out = merge(DEFAULT_POLICY, {})
    for policy in overrides:
        out = merge(out, policy)
    return out


class OptimizedDecoder(FusedDecoder):
    """One Gemma4 text decoder layer (sliding or full attention) on a 1x1 mesh, optimized for Blackhole."""

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
        """Build the layer from an HF state dict (arguments as ``FusedDecoder.from_state_dict``).

        policy: optional overrides of ``DEFAULT_POLICY`` (dtypes, fidelities, reader counts, grids), applied
            after the per-layer defaults of ``layer_policy(layer_idx, layer_type)``.
        """
        c = hf_config
        assert tuple(mesh_device.shape) == (1, 1), "optimized decoder runs on a 1x1 mesh"
        assert not c.enable_moe_block and not c.hidden_size_per_layer_input and not c.num_kv_shared_layers
        assert not c.attention_bias and not c.use_double_wide_mlp
        assert c.hidden_activation == "gelu_pytorch_tanh"
        assert block_size % TILE == 0 and prefill_chunk_size % block_size == 0
        self = cls()
        self.policy = p = merge_policy(layer_policy(layer_idx, c.layer_types[layer_idx]), policy)
        self.device = mesh_device
        self.layer_idx = layer_idx
        self.layer_type = c.layer_types[layer_idx]
        assert self.layer_type in ("sliding_attention", "full_attention")
        self.is_sliding = self.layer_type == "sliding_attention"
        self.hidden_size = c.hidden_size
        self.intermediate_size = c.intermediate_size
        self.num_heads = c.num_attention_heads
        self.k_eq_v = bool(c.attention_k_eq_v) and not self.is_sliding
        self.head_dim = c.head_dim if self.is_sliding else c.global_head_dim
        self.num_kv_heads = c.num_global_key_value_heads if self.k_eq_v else c.num_key_value_heads
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

        prefix = HF_PREFIX.format(layer_idx)
        sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)} or state_dict

        def as_tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(), dtype=dtype, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def norm_weight(name):
            # rms_norm consumes ROW_MAJOR weights folded into [1, 1, dim / 32, 32].
            return as_tt(sd[name + ".weight"].reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)

        q = sd["self_attn.q_proj.weight"]
        k = sd["self_attn.k_proj.weight"]
        v = k if self.k_eq_v else sd["self_attn.v_proj.weight"]
        assert q.shape[0] == self.num_heads * self.head_dim and k.shape[0] == self.num_kv_heads * self.head_dim
        self.meta_rope = self.is_sliding
        perm = interleave_perm(self.head_dim) if self.meta_rope else None
        if self.meta_rope:
            q = q.reshape(self.num_heads, self.head_dim, -1)[:, perm, :].reshape(q.shape)
            k = k.reshape(self.num_kv_heads, self.head_dim, -1)[:, perm, :].reshape(k.shape)
            assert q.shape[0] == 2 * k.shape[0] == 2 * v.shape[0]
        self.wqkv = self._dram_weight(torch.cat([q, k, v], dim=0).T, "qkv")
        self.wo = self._dram_weight(sd["self_attn.o_proj.weight"].T, "o")
        self.w_gate = self._dram_weight(sd["mlp.gate_proj.weight"].T, "gate")
        self.w_up = self._dram_weight(sd["mlp.up_proj.weight"].T, "up")
        self.w_down = self._dram_weight(sd["mlp.down_proj.weight"].T, "down")
        self.input_norm = norm_weight("input_layernorm")
        self.post_attn_norm = norm_weight("post_attention_layernorm")
        self.pre_ffn_norm = norm_weight("pre_feedforward_layernorm")
        self.post_ffn_norm = norm_weight("post_feedforward_layernorm")
        if self.meta_rope:
            q_w = sd["self_attn.q_norm.weight"][perm]
            k_w = sd["self_attn.k_norm.weight"][perm]
            self.q_norm = as_tt(q_w.reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)
            self.k_norm = as_tt(k_w.reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)
            self.q_norm_half = as_tt(q_w.repeat(self.num_heads // 2).reshape(1, 1, 1, -1))
            self.k_norm_heads = as_tt(k_w.repeat(self.num_kv_heads).reshape(1, 1, 1, -1))
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

        shared = {} if shared is None else shared
        key = ("rope", self.layer_type, self.max_context, "interleaved" if self.meta_rope else "hf")
        if key not in shared:
            cos, sin = rope_tables(c, self.layer_type, self.max_context)
            if self.meta_rope:
                cos, sin = cos[:, perm], sin[:, perm]
            shared[key] = {
                "cos": as_tt(cos.reshape(1, 1, self.max_context, self.head_dim)),
                "sin": as_tt(sin.reshape(1, 1, self.max_context, self.head_dim)),
                "cos_rm": as_tt(cos, layout=ttnn.ROW_MAJOR_LAYOUT),
                "sin_rm": as_tt(sin, layout=ttnn.ROW_MAJOR_LAYOUT),
            }
        self.rope = shared[key]
        self.row_ids = as_tt(
            torch.arange(prefill_chunk_size, dtype=torch.float32).reshape(1, 1, -1, 1), dtype=ttnn.float32
        )

        self.kv_dtype = p["kv_dtype"]
        cache_shape = [num_blocks, self.num_kv_heads, block_size, self.head_dim]
        self.kv_cache = [
            ttnn.zeros(
                cache_shape,
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

        self.compute_hifi4_fp32 = ckc("hifi4fp32")  # norms (and FusedDecoder helpers)
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
        self.compute_sdpa_prefill = ckc(p["sdpa_prefill_fid"]["sliding" if self.is_sliding else "full"])
        self.compute_sdpa_prefill_short = ckc("hifi2")
        self.compute_sdpa_decode = ckc(p["sdpa_decode_fid"])
        self.sliding_attn_block = p["sliding_attn_block"]
        mid = {None: None, "bfp8": ttnn.bfloat8_b}
        self.prefill_gate_up_dtype = mid[p["prefill_mlp_act_dtype"]["gate_up"]]
        self.prefill_product_dtype = mid[p["prefill_mlp_act_dtype"]["product"]]
        if self.meta_rope:
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
                        memory_config=ttnn.create_sharded_memory_config(
                            (TILE, TILE),
                            core_grid=users,
                            strategy=ttnn.ShardStrategy.HEIGHT,
                            use_height_and_width_as_shard_shape=True,
                        ),
                    ),
                }
            self.rope_trans = shared["rope_trans"]
            self.compute_rope_decode = ckc("hifi4")  # rotary_embedding_llama: BF16 dest at head_dim > 128
        self._sdpa_decode_cfgs = {}
        self.sdpa_prefill_q_cap = p["sdpa_prefill_q_chunk"][kind]
        self.sdpa_prefill_q_floor = p["sdpa_prefill_q_floor"][kind]
        self.sdpa_prefill_k_cap = p["sdpa_prefill_k_chunk"][kind] or block_size
        self.sdpa_prefill_exp_approx = p["sdpa_prefill_exp_approx"][kind]
        assert self.sdpa_prefill_k_cap % block_size == 0
        self._build_decode_configs()
        self._decode_layouts = {}
        return self

    # ------------------------------------------------------------------ weights and decode layouts
    def _dram_weight(self, w, role):
        """[K, N] torch -> width-sharded over the DRAM banks; each bank's shard splits evenly over its readers."""
        k, n = w.shape
        readers = self.policy["readers"][role]
        # Each bank's shard must split evenly over its readers without padding N: prefill reads the same
        # tensor (and the sliding QKV splits it four ways), so padded columns would be mis-split.
        assert n % (TILE * self.dram_banks * readers) == 0, f"{role}: {readers} readers need padded weights"
        n_pad = n
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(self.dram_grid, (k, n_pad // self.dram_banks), ttnn.ShardOrientation.ROW_MAJOR),
        )
        return ttnn.from_torch(
            w.contiguous(),
            dtype=self.policy["dtype"][role],
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=mem,
        )

    def _width_sharded(self, width, cores):
        crs = ttnn.num_cores_to_corerangeset(cores, self.grid, row_wise=True)
        return ttnn.create_sharded_memory_config(
            (TILE, width // cores),
            core_grid=crs,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )

    def _dram_mm_config(self, role, k, n, in_shard_tiles, out_shard_tiles):
        """DRAM-sharded decode matmul config: in0 blocks a whole number of input shards or a fraction of one."""
        p = self.policy
        kt, nt = k // TILE, n // TILE
        readers = p["readers"][role]
        in0 = p["in0"][role]
        if in0 is None:
            worker_n = -(-nt // (self.dram_banks * readers))
            tile_b = _TILE_BYTES[p["dtype"][role]]
            # Triple-buffered weight block plus double-buffered BF16 activation block per worker.
            legal = [
                d
                for d in range(1, kt + 1)
                if kt % d == 0
                and (in_shard_tiles % d == 0 or d % in_shard_tiles == 0)
                and (worker_n * 3 * tile_b + 2 * 2048) * d <= _CB_BUDGET
            ]
            in0 = max(legal)
        assert nt % out_shard_tiles == 0
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0,
            per_core_M=1,
            per_core_N=out_shard_tiles,
            fused_activation=None,
            num_workers_per_dram_bank=readers,
        )

    def _build_decode_configs(self):
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
        # nlp_create_qkv_heads_decode reads whole heads from equal-width shards: one head per shard.
        head_tiles = self.head_dim // TILE
        self.mm_cfg = {
            "qkv": self._dram_mm_config("qkv", h, qkv_n, res_tiles, head_tiles),
            "gate": self._dram_mm_config("gate", h, ff, res_tiles, p["mlp_shard_tiles"]),
            "up": self._dram_mm_config("up", h, ff, res_tiles, p["mlp_shard_tiles"]),
            "down": self._dram_mm_config("down", ff, h, p["mlp_shard_tiles"], res_tiles),
            # o_proj reads the head-concat output: one head per core.
            "o": self._dram_mm_config("o", self.num_heads * self.head_dim, h, head_tiles, res_tiles),
        }

    def _sdpa_decode_cfg(self, batch):
        """(K chunk, SDPA decode program config) for ``batch`` users."""
        if batch not in self._sdpa_decode_cfgs:
            p = self.policy
            kind = "sliding" if self.is_sliding else "full"

            def by_batch(table):
                table = {int(b): v for b, v in table.items()}  # JSON keys are strings; later entries win
                return table[min(b for b in table if b >= batch)]

            k_chunk = by_batch(p["sdpa_decode_k_chunk"][kind]) or self.block_size
            assert k_chunk % self.block_size == 0
            grid = p["sdpa_decode_grid"]
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid) if grid else self.grid,
                q_chunk_size=0,
                k_chunk_size=k_chunk,
                exp_approx_mode=p["sdpa_decode_exp_approx"],
                max_cores_per_head_batch=by_batch(p["sdpa_decode_max_cores"][kind]),
            )
            self._sdpa_decode_cfgs[batch] = (k_chunk, cfg)
        return self._sdpa_decode_cfgs[batch]

    def decode_input_memory_config(self, batch=1):
        """Layout of the decode residual stream (input and output of ``decode_forward``), any batch."""
        return self.residual_mem

    # ------------------------------------------------------------------ attention helpers
    def _table_for_k_chunk(self, table, k_chunk):
        """Pad a page-table row block so its width covers whole K chunks.

        The SDPA ops round the attended K range up to whole K chunks and index the page table without a
        width bound (FusedDecoder keeps K chunks at one block for this reason). Padding the row with block 0
        keeps those reads inside the tensor; the padded keys lie beyond every query's causal range and are
        masked. No-op (no device op) when the width already divides.
        """
        per_chunk = k_chunk // self.block_size
        width = table.shape[-1]
        if per_chunk == 1 or width % per_chunk == 0:
            return table
        return ttnn.pad(table, [(0, 0), (0, per_chunk - width % per_chunk)], 0)

    def _window_shift(self, chunk_start):
        """Sliding layers: page-aligned origin <= the first key the rows starting at ``chunk_start`` attend.

        FusedDecoder rounds the origin down to a multiple of the Q/K tile alignment, so a continuation that
        starts off that alignment (e.g. 1088) runs with 64-row K chunks. Here the origin keeps
        ``chunk_start - shift`` a multiple of the alignment instead, so every sliding chunk keeps the
        largest K chunk and the same tiling as an aligned chunk. Exact either way (masks depend on q - k).
        """
        if self.sliding_window is None:
            return 0
        align = math.lcm(SLIDING_Q_TILE, self.sdpa_prefill_k_cap, self.block_size)
        first_key = max(0, chunk_start - self.sliding_window + 1)
        shift = chunk_start - align * -(-(chunk_start - first_key) // align)
        return shift if shift >= 0 else 0

    def _window_table_blocks(self, rows):
        """Fixed page-table width per sliding query block (see FusedDecoder), with this layer's alignment."""
        align = math.lcm(SLIDING_Q_TILE, self.sdpa_prefill_k_cap, self.block_size)
        return -(-(self.sliding_window + align + max(rows, self.sliding_attn_block)) // self.block_size)

    def _sdpa_prefill_cfg(self, chunk_start, padded_len, fp32_acc=None):
        """Chunked-SDPA program config: chunk start a multiple of both chunk sizes, Q chunks divide the rows."""

        def largest(cap, *values):
            size = cap
            while size > TILE and any(v % size for v in values):
                size //= 2
            return size

        q_chunk = largest(self.sdpa_prefill_q_cap, chunk_start, padded_len)
        floor = self.sdpa_prefill_q_floor
        if floor is not None and padded_len > floor[0] and q_chunk < floor[1] and chunk_start % floor[1] == 0:
            q_chunk = floor[1]
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=q_chunk,
            k_chunk_size=largest(self.sdpa_prefill_k_cap, chunk_start),
            exp_approx_mode=self.sdpa_prefill_exp_approx,
            fp32_output_accumulation=(self.sliding_window is None) if fp32_acc is None else fp32_acc,
        )

    def _sdpa(self, q, *, page_table, user_id, q_start):
        """Chunked SDPA of query rows ``q`` over the request's paged cache, from the ``_window_shift`` origin."""
        shift = self._window_shift(q_start)
        first = shift // self.block_size
        last = page_table.shape[-1]
        if self.sliding_window is not None:
            last = min(last, first + self._window_table_blocks(q.shape[-2]))
        table = ttnn.slice(page_table, [user_id, first], [user_id + 1, last])
        # Full layers keep exact FP32 running accumulators for long prefixes; requests whose keys all lie within
        # ``sdpa_prefill_full_streaming_max_kv`` take the faster streaming kernel instead (BF16 dest).
        short = (
            self.sliding_window is None and q_start + q.shape[-2] <= self.policy["sdpa_prefill_full_streaming_max_kv"]
        )
        cfg = self._sdpa_prefill_cfg(q_start - shift, q.shape[-2], fp32_acc=False if short else None)
        table = self._table_for_k_chunk(table, cfg.k_chunk_size)
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.kv_cache[0],
            self.kv_cache[1],
            table,
            q_start - shift,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            program_config=cfg,
            compute_kernel_config=self.compute_sdpa_prefill_short if short else self.compute_sdpa_prefill,
        )

    # ------------------------------------------------------------------ prefill
    def _prefill_2d_cfg(self, m, k, n, role, fused_activation=None):
        """2D multicast matmul config for a short prefill chunk over the full compute grid.

        Grid width: the most columns whose output subblock width reaches ``prefill_2d_sub_w_target`` (else the
        widest subblock); per-core blocks cover M over the grid rows and N over its columns. ``in0_block_w`` is
        the largest divisor of the K tiles up to the role's swept preference (``prefill_2d_in0``) whose static
        CBs stay within ``_PREFILL_2D_CB_BUDGET``. M cannot go over the 11 columns instead (``transpose_mcast``
        is rejected for the DRAM-sharded weight), so e.g. 32 M tiles use 8 of the 10 rows.
        """
        kt, nt, mt = k // TILE, n // TILE, -(-m // TILE)

        def sub_w(per_core_n):
            return max(w for w in range(1, min(per_core_n, 4) + 1) if per_core_n % w == 0)

        # More columns win once the subblock is at least prefill_2d_sub_w_target wide (README "Prefill matmul")
        target = self.policy["prefill_2d_sub_w_target"]
        cols = max(range(1, self.grid.x + 1), key=lambda c: (min(sub_w(-(-nt // c)), target), c))
        per_core_n = -(-nt // cols)
        per_core_m = -(-mt // self.grid.y)
        w_tile = _TILE_BYTES[self.policy["dtype"][role]]
        fixed = per_core_m * per_core_n * 2048  # output block

        def cb_bytes(d):
            return per_core_m * d * 2 * 2048 + d * per_core_n * 2 * w_tile + fixed

        cap = self.policy["prefill_2d_in0"][role]
        in0 = max(
            d for d in range(1, min(cap, kt) + 1) if kt % d == 0 and (d == 1 or cb_bytes(d) <= _PREFILL_2D_CB_BUDGET)
        )
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(cols, self.grid.y),
            in0_block_w=in0,
            out_subblock_h=1,
            out_subblock_w=sub_w(per_core_n),
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=False,
        )

    def _prefill_matmul(self, x, w, role, fused_activation=None, dtype=None):
        m = x.shape[-2]
        if m <= self.policy["prefill_linear_max_rows"]:
            return ttnn.linear(
                x,
                w,
                program_config=self._prefill_2d_cfg(m, x.shape[-1], w.shape[-1], role, fused_activation),
                compute_kernel_config=self.prefill_ck[role],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=dtype,
            )
        return ttnn.experimental.minimal_matmul(
            x,
            w,
            fused_activation=fused_activation,
            dtype=dtype,
            config=self.prefill_mm_cfg[role],
            compute_kernel_config=self.prefill_ck[role],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _prefill_heads_hf(self, h, cos, sin):
        qkv = self._prefill_matmul(h, self.wqkv, "qkv")
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

    def _prefill_heads_meta(self, h, cos, sin):
        m = h.shape[-2]
        if m <= self.policy["prefill_sliding_qkv_linear_max_rows"]:
            # Short chunks: the 2D-multicast matmul, then four column slices [Q half | Q half | K | V].
            qkv = self._prefill_matmul(h, self.wqkv, "qkv")
            w = qkv.shape[-1] // 4
            qa, qb, k_flat, v_flat = (ttnn.slice(qkv, [0, 0, 0, i * w], [1, 1, m, (i + 1) * w]) for i in range(4))
            ttnn.deallocate(qkv)
        else:
            qa, qb, k_flat, v_flat = ttnn.experimental.minimal_matmul_split(
                h,
                self.wqkv,
                chunks=4,
                config=self.prefill_mm_cfg["qkv"],
                compute_kernel_config=self.prefill_ck["qkv"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        half = self.num_heads // 2
        q_parts = [self._norm_rope(t, half, self.q_norm_half, cos, sin) for t in (qa, qb)]
        q_r = ttnn.concat(q_parts, dim=1)
        k_r = self._norm_rope(k_flat, self.num_kv_heads, self.k_norm_heads, cos, sin)
        v_n = self._norm_rope(v_flat, self.num_kv_heads)
        for t in (qa, qb, k_flat, v_flat, *q_parts):
            ttnn.deallocate(t)
        return q_r, k_r, v_n

    def _mlp_and_residuals(self, x, attn_out, matmul):
        """Prefill: post-attention norm + residual, GELU(tanh)-gated MLP, post-FFN norm + residual."""
        h = self._rms(attn_out, self.post_attn_norm)
        ttnn.deallocate(attn_out)
        x = ttnn.add(x, h)
        ttnn.deallocate(h)
        h = self._rms(x, self.pre_ffn_norm)
        long_chunk = x.shape[-2] > self.policy["prefill_linear_max_rows"]
        mid = self.prefill_gate_up_dtype if long_chunk else None
        if self.policy["prefill_gelu_in_gate"]:
            gate = matmul(
                h, self.w_gate, "gate", fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH), dtype=mid
            )
            act = []
        else:
            gate = matmul(h, self.w_gate, "gate", dtype=mid)
            act = [self.gelu]
        up = matmul(h, self.w_up, "up", dtype=mid)
        ttnn.deallocate(h)
        prod = ttnn.multiply(
            gate, up, input_tensor_a_activations=act, dtype=self.prefill_product_dtype if long_chunk else None
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        down = matmul(prod, self.w_down, "down")
        ttnn.deallocate(prod)
        h = self._rms(down, self.post_ffn_norm)
        ttnn.deallocate(down)
        out = ttnn.add(x, h, activations=[(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.layer_scalar)])
        ttnn.deallocate(x)
        ttnn.deallocate(h)
        return out

    def prefill_forward(self, hidden_states, *, page_table, start_pos=0, user_id=0):
        """``[1, 1, S, H]`` -> ``[1, 1, S, H]`` for one request (contract as FusedDecoder.prefill_forward).

        Same chunking as FusedDecoder. A non-aligned tail is zero-padded by ``ttnn.pad`` alone instead of
        FusedDecoder's pad + row-id mask + ``where``: on device, tile ``pad`` first fills the last tile's implicit
        padding with the pad value (``fill_implicit_tile_padding`` in pad.cpp ``invoke_tile``), then writes zero
        tiles beyond it. Every pad row is zero either way, so the cache never holds NaN; outputs are bitwise equal.
        The fill writes the implicit padding of the input in place (the caller's tensor for a one-chunk request).
        """
        seq_len = hidden_states.shape[-2]
        assert hidden_states.shape[0] == 1 and hidden_states.shape[1] == 1, "prefill is one request per call"
        assert seq_len >= 1 and start_pos >= 0 and start_pos % self.block_size == 0
        assert start_pos + seq_len <= self.max_context
        chunk = self.prefill_chunk_size
        outputs = []
        for offset in range(0, seq_len, chunk):
            valid = min(chunk, seq_len - offset)
            padded = -(-valid // self.block_size) * self.block_size
            x = ttnn.slice(hidden_states, [0, 0, offset, 0], [1, 1, offset + valid, self.hidden_size])
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
            # A row-prefix slice may be a view of ``out``; let Python own the buffer's lifetime.
            out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, seq_len, self.hidden_size])
        return out

    def _prefill_chunk(self, x, *, page_table, user_id, chunk_start, padded_len):
        h = self._rms(x, self.input_norm)
        end = chunk_start + padded_len
        cos = ttnn.slice(self.rope["cos"], [0, 0, chunk_start, 0], [1, 1, end, self.head_dim])
        sin = ttnn.slice(self.rope["sin"], [0, 0, chunk_start, 0], [1, 1, end, self.head_dim])
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
        o = self._prefill_matmul(attn_flat, self.wo, "o")
        ttnn.deallocate(attn_flat)
        return self._mlp_and_residuals(x, o, self._prefill_matmul)

    # ------------------------------------------------------------------ decode
    def _decode_mm(self, x, role):
        w = {"qkv": self.wqkv, "o": self.wo, "gate": self.w_gate, "up": self.w_up, "down": self.w_down}[role]
        return ttnn.linear(
            x,
            w,
            program_config=self.mm_cfg[role],
            compute_kernel_config=self.decode_ck[role],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

    def _rms_res(self, x, weight):
        """RMSNorm of a residual-layout tensor, sharded on the residual grid."""
        return ttnn.rms_norm(
            x,
            epsilon=self.eps,
            weight=weight,
            program_config=self.residual_norm_cfg,
            memory_config=self.residual_mem,
            compute_kernel_config=self.compute_hifi4_fp32,
        )

    def decode_rope_rows(self, current_pos, batch):
        """Decode RoPE rows (cos, sin) for ``current_pos`` ``[B]``: one ``[32, head_dim]`` tile per user.

        The rows depend only on the positions and the layer kind (tables are shared per kind), so a layer
        stack can compute them once per step per kind and pass them to every layer of that kind
        (``decode_forward(..., rope_rows=...)``); the caller then owns them.
        """
        return self._decode_rope_rows(current_pos, batch, self._decode_layout(batch)["rope"])

    def decode_forward(self, hidden_states, *, current_pos, page_table, rope_rows=None):
        """``[1, 1, B, H]`` -> ``[1, 1, B, H]`` in ``decode_input_memory_config()`` (contract as FusedDecoder).

        rope_rows: optional ``decode_rope_rows(current_pos, B)`` of a layer of the same kind, shared across
        the layers of a stack; computed here when omitted.
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
            # The head-concat output is tile-padded to 32 users; restore the logical batch (metadata only).
            width = attn_flat.shape[-1]
            attn_flat = ttnn.reshape(attn_flat, [1, 1, batch, width], [1, 1, attn_flat.padded_shape[-2], width])
        o = self._decode_mm(attn_flat, "o")
        ttnn.deallocate(attn_flat)

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
        down = self._decode_mm(prod, "down")
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
