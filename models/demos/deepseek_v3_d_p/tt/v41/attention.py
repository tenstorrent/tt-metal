# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill attention on the V4-Flash ttnn modules (``tt/mla/heavily_compressed_attention.py``,
``tt/v4/attention``), with V4.1's differences:

* no per-head q RMSNorm (V4-Flash normalises every head of q after ``q_b_proj``; V4.1's ``model.py`` does not --
  tt-blaze DS41F-0015 found the same on the decode ring);
* the dimensions (hidden 5120, q_lora 1280, 64 heads x 512, rope 64, o_groups 8 x o_lora 1024) and ``norm_eps`` 1e-20;
* RoPE: the same two families as V4-Flash -- "main" (theta 10000, plain) for the sliding layers 0 / 1 and "compress"
  (theta 160000, YaRN x16 over 65536, beta 32 / 1) for every compressed layer -- so ``DeepseekV4RotaryEmbedding`` built
  from a V4.1-dimensioned config gives V4.1's tables (the interleaved-pair rotation == ``rotary_embedding_llama`` with
  the transformation matrix, V4-Flash ``tt/v4/golden.py``).

Weights arrive in ``model.py``'s names (``tt/v41/weights.py``) and are mapped onto the V4-Flash constructor arguments here.
"""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.v4.attention.swa import TtSWA


def v41_hf_config(cfg, *, max_seq: int = 8192):
    """A ``DeepseekV4Config`` carrying V4.1's dimensions and RoPE, for the V4-Flash modules that read an HF config (the
    rotary embedding, the block builders). Layer KINDS are not taken from it (V4.1's ratios 2 / 1 are outside V4-Flash's
    {0, 4, 128}); ``tt/v41/config.py`` ``V41Config.role`` is the schedule."""
    from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    hf = DeepseekV4Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.dim,
        moe_intermediate_size=cfg.moe_inter_dim,
        num_hidden_layers=4,  # the schedule fields below are V4-Flash-shaped placeholders; see the docstring
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=1,
        head_dim=cfg.head_dim,
        q_lora_rank=cfg.q_lora_rank,
        o_lora_rank=cfg.o_lora_rank,
        o_groups=cfg.o_groups,
        num_experts_per_tok=cfg.n_activated_experts,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=1,
        scoring_func=cfg.score_func,
        routed_scaling_factor=cfg.route_scale,
        max_position_embeddings=1 << 20,
        rope_theta=float(cfg.rope_theta),
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        compress_rope_theta=float(cfg.compress_rope_theta),
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        swiglu_limit=cfg.swiglu_limit,
        sliding_window=cfg.window_size,
        index_n_heads=cfg.index_n_heads,
        index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk,
        rms_norm_eps=cfg.norm_eps,
        rope_parameters={
            "rope_type": "yarn",
            "factor": cfg.rope_factor,
            "beta_fast": cfg.beta_fast,
            "beta_slow": cfg.beta_slow,
            "original_max_position_embeddings": cfg.original_seq_len,
        },
        compress_ratios=[0, 0, 4, 128],
        num_hash_layers=0,
        qk_rope_head_dim=cfg.rope_head_dim,
    )
    hf._attn_implementation = "eager"
    hf.max_seq_len = int(max_seq)
    return hf


class _V41QStem:
    """V4.1's q stem: ``q_b_proj(q_norm(q_a_proj(h)))`` split into heads, RoPE on the trailing 64 -- the V4-Flash stem minus
    its per-head RMSNorm. Mixed into the V4-Flash attention classes (they call ``self._q_stem``)."""

    def _q_stem(self, hidden_states, cos, sin, return_latent: bool = False):
        input_shape = tuple(hidden_states.shape)
        if len(input_shape) != 4 or input_shape[1] != 1:
            raise ValueError(f"Expected hidden_states shape [B, 1, S, hidden], got {input_shape}")
        batch, seq_len = input_shape[0], input_shape[2]
        num_heads_local = self.num_heads // self.tp_factor
        q = ttnn.linear(hidden_states, self.wq_a, memory_config=self.memory_config)
        if self.tp_factor > 1:
            q = ttnn.experimental.reduce_scatter_minimal_async(
                q,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )
            q = ttnn.experimental.all_gather_async(
                q,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.tp_axis,
            )
        q = ttnn.rms_norm(q, weight=self.q_a_norm_weight, epsilon=self.rms_norm_eps)
        latent = q
        q = ttnn.linear(q, self.wq_b, memory_config=self.memory_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=num_heads_local, num_kv_heads=0, transpose_k_heads=False, memory_config=self.memory_config
        )
        # (no per-head RMSNorm here: V4.1)
        nope_dim = self.head_dim - self.rope_head_dim
        nope = ttnn.slice(q, [0, 0, 0, 0], [batch, num_heads_local, seq_len, nope_dim])
        rope = ttnn.slice(q, [0, 0, 0, nope_dim], [batch, num_heads_local, seq_len, self.head_dim])
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        q = ttnn.concat([nope, rope], dim=-1)
        return (q, latent) if return_latent else q


def _common_kwargs(
    cfg, w: dict, rotary_emb, *, sp_axis, tp_axis, topology, weight_cache_path, layer: int, name: str
) -> dict:
    """``model.py`` attention weights -> the V4-Flash constructor arguments (fp32 / bf16 host tensors)."""
    return dict(
        q_a_proj_weight=w["attn.wq_a.weight"],
        q_a_norm_weight=w["attn.q_norm.weight"].float(),
        q_b_proj_weight=w["attn.wq_b.weight"],
        kv_proj_weight=w["attn.wkv.weight"],
        kv_norm_weight=w["attn.kv_norm.weight"].float(),
        sinks=w["attn.attn_sink"].float(),
        o_a_proj_weight=w["attn.wo_a.weight"],
        o_b_proj_weight=w["attn.wo_b.weight"],
        rotary_emb=rotary_emb,
        num_heads=cfg.n_heads,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.rope_head_dim,
        sliding_window=cfg.window_size,
        o_groups=cfg.o_groups,
        rms_norm_eps=cfg.norm_eps,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        topology=topology,
        weight_cache_path=weight_cache_path,
        cache_name_prefix=f"v41_layer_{layer}.{name}",
    )


class V41SWA(_V41QStem, TtSWA):
    """V4.1 layers 0 / 1: sliding-window attention (window 128, sinks, grouped o-proj), "main" RoPE."""

    @classmethod
    def from_weights(
        cls,
        mesh_device,
        cfg,
        layer: int,
        w: dict,
        rotary_emb,
        *,
        sp_axis=0,
        tp_axis=1,
        topology=ttnn.Topology.Linear,
        weight_cache_path=None,
    ) -> "V41SWA":
        assert cfg.compress_ratios[layer] == 0, f"layer {layer} is not a sliding-window layer"
        return cls(
            mesh_device,
            **_common_kwargs(
                cfg,
                w,
                rotary_emb,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                topology=topology,
                weight_cache_path=weight_cache_path,
                layer=layer,
                name="attn",
            ),
        )


def reference_attention_input(mpy, layer, x_in: torch.Tensor, pre_in: torch.Tensor) -> torch.Tensor:
    """model.py's attention input for a block: ``attn_norm(hc_pre(x_in, pre_in))`` ([S, hc, d], [S, hc] -> [S, d])."""
    with torch.inference_mode():
        x = layer.hc_pre(x_in.unsqueeze(0), pre_in.unsqueeze(0))
        return layer.attn_norm(x)[0]


# ---- compressed layers (CSA2: ratio 2 sources 2 / 8 / 14 + their consumers; layer 20's ratio-1 KV) --------------------------------

from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCACompressor  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.v4.attention.csa import _RAGGED_FULL_CUT, TtCSA, TtCSAIndexer  # noqa: E402
from models.demos.deepseek_v3_d_p.tt.v4.trace_island import copy_into  # noqa: E402


class V41Compressor(TtHCACompressor):
    """V4.1's ``Compressor`` (model.py): ``kv = wkv(x)``, ``score = wgate(x)``, a softmax over each group of ``ratio``
    consecutive tokens per channel, ``kv = sum(kv * softmax(score))``, RMSNorm -> the LATENT (pre-RoPE: the indexer keys
    are derived from it) -> RoPE on the last 64 at the group's FIRST position ``j * ratio`` -> the compressed entry. That
    is V4-Flash's HCA compressor with no position bias (V4.1 has no APE) at rate 2; ratio 1 (layer 20) is ``norm(wkv(x))``
    = the same with a single-row softmax (weight 1), so its gate weight is zeros.

    Called with TtCSA's compressor interface ``(hidden, real_len, first_window_position, prior, need_mask, mask_width) ->
    (entries, mask_block, prior)``; the latent of the last call is kept on ``self.last_latent`` for ``V41IndexKeys``. The
    non-overlap pooling carries no state across chunks while chunk boundaries are multiples of ``ratio`` (asserted), so the
    ``prior`` is an empty placeholder."""

    def __init__(
        self,
        device,
        *,
        wkv: torch.Tensor,
        wgate: torch.Tensor | None,
        norm: torch.Tensor,
        ratio: int,
        head_dim: int,
        rope_head_dim: int,
        rotary_emb,
        rms_norm_eps: float,
        **kw,
    ):
        wgate = torch.zeros_like(wkv) if wgate is None else wgate
        super().__init__(
            device,
            kv_proj_weight=wkv.float(),
            gate_proj_weight=wgate.float(),
            position_bias=torch.zeros(ratio, head_dim),
            kv_norm_weight=norm.float(),
            head_dim=head_dim,
            compress_rate=ratio,
            rope_head_dim=rope_head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=rms_norm_eps,
            **kw,
        )
        self.fp32 = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.last_latent = None

    # TtCSA's compressor protocol --------------------------------------------------------------------------------------
    def empty_prior(self, batch: int = 1):
        return ()

    def reset_prior(self, prior: tuple) -> None:
        return None

    def _tp_all_reduce(self, x):
        if self.tp_factor == 1:
            return x
        x = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.tp_axis,
        )
        return ttnn.experimental.all_gather_async(
            x,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.tp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.tp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.tp_axis,
        )

    def _sp_all_gather(self, x):
        if self.sp_factor == 1:
            return x
        return ttnn.experimental.all_gather_async(
            x,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            cluster_axis=self.sp_axis,
        )

    def __call__(self, hidden_states, seq_len_actual, first_window_position, prior, need_mask=True, mask_width=None):
        return self.forward_v41(hidden_states, seq_len_actual, first_window_position, need_mask, mask_width) + (prior,)

    def forward_v41(self, hidden_states, seq_len_actual, first_window_position, need_mask=True, mask_width=None):
        batch, seq_len = int(hidden_states.shape[0]), int(hidden_states.shape[2])
        rate, W = self.compress_rate, self.head_dim
        assert (
            first_window_position % rate == 0
        ), "V4.1's compressor carries no state: chunks must start on a group boundary"
        kv = self._tp_all_reduce(ttnn.linear(hidden_states, self.wkv, memory_config=self.memory_config))
        n_windows = seq_len // rate
        assert n_windows > 0, f"each chip needs a whole group ({seq_len} rows < ratio {rate})"
        if rate > 1:
            gate = self._tp_all_reduce(ttnn.linear(hidden_states, self.wgate, memory_config=self.memory_config))
            gate = ttnn.typecast(ttnn.reshape(gate, [batch, n_windows, rate, W]), ttnn.float32)
            weights = ttnn.softmax(gate, dim=2, numeric_stable=True)
            kv = ttnn.typecast(ttnn.reshape(kv, [batch, n_windows, rate, W]), ttnn.float32)
            pooled = ttnn.typecast(ttnn.sum(ttnn.multiply(kv, weights), dim=2), self.dtype)  # fp32 pooling, as model.py
            latent = ttnn.reshape(pooled, [batch, 1, n_windows, W])
        else:
            latent = kv
        latent = ttnn.rms_norm(latent, weight=self.kv_norm_weight, epsilon=self.rms_norm_eps)
        nope_dim = W - self.rope_head_dim
        nope = ttnn.slice(latent, [0, 0, 0, 0], [batch, 1, n_windows, nope_dim])
        rope = ttnn.slice(latent, [0, 0, 0, nope_dim], [batch, 1, n_windows, W])
        cos, sin = self._rope_gather(
            self._entry_rope, self._rope_index(self._entry_index, first_window_position // rate)
        )
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        entries = self._sp_all_gather(ttnn.concat([nope, rope], dim=-1))
        self.last_latent = self._sp_all_gather(latent)  # [1, 1, n, 512] pre-RoPE, replicated
        self.last_rope = (cos, sin)  # per-chip entry positions: the index keys reuse the rotation before their gather
        self._last_latent_local = latent
        mask_block = None
        if need_mask and seq_len_actual > 1 and seq_len_actual // rate > 0:
            mask_block = self._mask_block(seq_len, first_window_position, seq_len_actual, width=mask_width)
        return entries, mask_block


# DS41F-0048: model.py fp4-quantises the indexer's q and k (``fp4_act_quant(x, 32, inplace=True)``, e8m0 scales) before the
# score GEMM; V41_PREFILL_INDEX_FP4=1 does the same on device (fake quant: the values are snapped to the fp4 grid and kept in
# the tensor's dtype). Per 32-wide block: s = 2 ** ceil(log2(amax / 6)); x <- e2m1_rne(clamp(x / s, -6, 6)) * s.
INDEX_FP4 = os.environ.get("V41_PREFILL_INDEX_FP4", "0") == "1"
# DS41F-0067: the CSA sources run TtCSA's sparse path A once the context holds a window + top-k worth of entries (chunk 0
# stays on the dense path B) and their Reuse consumers gather with the source's top-k; 0 = path B everywhere (the default
# until the e2e A/B lands). With islands the top-k goes into one persistent buffer (precreate_shared_masks).
PATH_A = os.environ.get("PREFILL_V41_PATH_A", "0") == "1"
# DS41F-0068: path A3 -- the CSA attention of every path-A chunk as two trace islands per layer and slot (V4-Flash DS4F-0246 /
# 0300, TtCSA's prepare_chunk / forward_pre / glue_chunk / forward_attn / epilogue_chunk protocol), replayed by the block between
# its mHC / MoE islands; chunk 0 (path B) and a ragged final chunk stay eager. Needs PATH_A and V41_PREFILL_ISLANDS.
A3 = os.environ.get("PREFILL_V41_A3", "0") == "1"
# DS41F-0073: chunk 0's CSA attention on its own captured A2 island (V4-Flash DS4F-0300's A2c0: the same A1, then the sparse
# core with each index row repacked to [valid window | valid top-k | sentinels], rate-aware) instead of the eager path B; every
# prompt shorter than one chunk is chunk 0 alone. Default off until the export / e2e A/B lands. Needs A3.
A3_CHUNK0 = os.environ.get("PREFILL_V41_A3_CHUNK0", "0") == "1"
# e2m1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 as a sum of steps at the midpoints; ties go to the even code (0.25 -> 0,
# 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4): "gt" at a tie that rounds down, "ge" at one that rounds up
_FP4_STEPS = (
    (0.25, 0.5, "gt"),
    (0.75, 0.5, "ge"),
    (1.25, 0.5, "gt"),
    (1.75, 0.5, "ge"),
    (2.5, 1.0, "gt"),
    (3.5, 1.0, "ge"),
    (5.0, 2.0, "gt"),
)


def fp4_act_fake_quant(x, block: int = 32):
    """``model.py``'s ``fp4_act_quant(x, block, inplace=True)`` (e8m0 scales) on a [a, b, c, D] TILE tensor, D % block == 0.
    The scale's ceil(log2) takes a 3e-3 margin below the integer: an exact power of two (amax = 1.5 * 2^m, bf16) stays put,
    and the next bf16 amax above it is >= 0.0075 above in log2, so the SFPU log2's error cannot flip it."""
    a0, a1, a2, D = (int(v) for v in x.shape)
    xb = ttnn.typecast(ttnn.reshape(x, [a0, a1, a2 * (D // block), block]), ttnn.float32)
    amax = ttnn.clamp(ttnn.max(ttnn.abs(xb), dim=-1, keepdim=True), min=6.0 * 2.0**-126)
    t = ttnn.ceil(ttnn.subtract(ttnn.log2(ttnn.multiply(amax, 1.0 / 6.0)), 3e-3))
    y = ttnn.clamp(ttnn.multiply(xb, ttnn.exp2(ttnn.neg(t))), min=-6.0, max=6.0)
    mag_in = ttnn.abs(y)
    mag = None
    for th, w, op in _FP4_STEPS:
        step = ttnn.multiply((ttnn.gt if op == "gt" else ttnn.ge)(mag_in, th), w)
        mag = step if mag is None else ttnn.add(mag, step)
    q = ttnn.multiply(ttnn.multiply(ttnn.sign(y), mag), ttnn.exp2(t))
    return ttnn.typecast(ttnn.reshape(q, [a0, a1, a2, D]), x.dtype)


class V41IndexKeys:
    """V4.1's lightning-indexer KEYS (``Indexer.forward``, owner layers): ``k = k_norm(wk(latent))``, RoPE on the last 64 at
    the group positions -- from the main compressor's latent of the same chunk (V4-Flash ran a second compressor). Stands in
    for ``TtCSAIndexer.compressor``: called with that interface it returns ``(keys [1, 1, n, 128] replicated, None, prior)``,
    and it lends the main compressor's CCL / fp32 / rotation state."""

    def __init__(self, main: V41Compressor, *, wk: torch.Tensor, k_norm: torch.Tensor, head_dim: int = 128):
        self.main = main
        self.head_dim = int(head_dim)
        self.wk = main._from_torch(wk.float().t().contiguous().reshape(1, 1, *wk.t().shape), dtype=ttnn.bfloat16)
        self.k_norm = main._from_torch(k_norm.float().reshape(1, 1, 1, -1))
        self.tt_ccl, self.ccl_num_links, self.fp32, self.trans_mat = (
            main.tt_ccl,
            main.ccl_num_links,
            main.fp32,
            main.trans_mat,
        )
        self.compress_rate = main.compress_rate

    def _tp_all_reduce(self, x):
        return self.main._tp_all_reduce(x)

    def alloc_tables(self, *a, **k):
        return None

    def empty_prior(self, batch: int = 1):
        return ()

    def reset_prior(self, prior):
        return None

    def __call__(self, hidden_states, seq_len_actual, first_window_position, prior, need_mask=False, mask_width=None):
        m = self.main
        lat = m._last_latent_local  # this chip's groups (pre-RoPE), the rotation of the same positions below
        k = ttnn.matmul(lat, self.wk, memory_config=m.memory_config)  # [1, 1, n_l, 128]
        k = ttnn.rms_norm(k, weight=self.k_norm, epsilon=m.rms_norm_eps)
        n = int(k.shape[2])
        nope = ttnn.slice(k, [0, 0, 0, 0], [1, 1, n, self.head_dim - m.rope_head_dim])
        rope = ttnn.slice(k, [0, 0, 0, self.head_dim - m.rope_head_dim], [1, 1, n, self.head_dim])
        cos, sin = m.last_rope
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, m.trans_mat, is_decode_mode=False)
        k = ttnn.concat([nope, rope], dim=-1)
        if INDEX_FP4:  # model.py: fp4_act_quant(k) after the RoPE, before the key cache
            k = fp4_act_fake_quant(k)
        keys = m._sp_all_gather(k)
        return keys, None, prior


class V41CSA(_V41QStem, TtCSA):
    """V4.1 KV + index source (``Full`` mode, layers 2 / 8 / 14 at ratio 2): its own compressor, indexer and attention over
    [window | its top-512 entries] (V4-Flash's TtCSA path B: dense SDPA with the 0 / -inf selection mask). After a
    forward, ``state.compressed_kv`` / ``self.debug_last_selection`` are what its consumers read for the same chunk."""

    @classmethod
    def from_weights(
        cls,
        mesh_device,
        cfg,
        layer: int,
        w: dict,
        rotary_emb,
        *,
        sp_axis=0,
        tp_axis=1,
        topology=ttnn.Topology.Linear,
        weight_cache_path=None,
    ) -> "V41CSA":
        r = cfg.role(layer)
        assert r.mode == "full", f"layer {layer} ({r.mode}) is not a KV + index source"
        mesh = dict(sp_axis=sp_axis, tp_axis=tp_axis, topology=topology)
        comp = V41Compressor(
            mesh_device,
            wkv=w["attn.compressor.wkv.weight"],
            wgate=w.get("attn.compressor.wgate.weight"),
            norm=w["attn.compressor.norm.weight"],
            ratio=r.compress_ratio,
            head_dim=cfg.head_dim,
            rope_head_dim=cfg.rope_head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=cfg.norm_eps,
            **mesh,
        )
        keys = V41IndexKeys(
            comp, wk=w["attn.indexer.wk.weight"], k_norm=w["attn.indexer.k_norm.weight"], head_dim=cfg.index_head_dim
        )
        indexer = TtCSAIndexer(
            mesh_device,
            compressor=keys,
            q_b_proj_weight=w["attn.indexer.wq_b.weight"],
            weights_proj_weight=w["attn.indexer.weights_proj.weight"].float(),
            n_heads=cfg.index_n_heads,
            head_dim=cfg.index_head_dim,
            rope_head_dim=cfg.rope_head_dim,
            topk=cfg.index_topk,
            **mesh,
        )
        kw = _common_kwargs(
            cfg,
            w,
            rotary_emb,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            weight_cache_path=weight_cache_path,
            layer=layer,
            name="attn",
        )
        if INDEX_FP4:
            indexer.q_act_quant = fp4_act_fake_quant
        return cls(mesh_device, compressor=comp, indexer=indexer, sparse_path=PATH_A, **kw)


def _live_width(attn, state, entries: int) -> int:
    """TtCSA's live-extent width (DS4F-0252) for ``entries`` entries written: a tile multiple rounded up to the live quantum,
    capped at the capacity -- the width of the selection mask and of the compressed rows its attention reads."""
    cap = int(state.compressed_kv.shape[2])
    if not attn.live_extent:
        return cap
    w = -(-int(entries) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    if attn.live_quantum > 0:
        w = -(-w // attn.live_quantum) * attn.live_quantum
    return min(cap, w)


def _source_forward(self, hidden_states, seq_len_actual=None, *, state, export=None):
    out = TtCSA.forward(self, hidden_states, seq_len_actual, state=state, export=export)
    pub = getattr(self, "published", None)
    if pub is not None and pub["path"] == "A":
        # DS41F-0067 path A: the consumers gather with this chunk's top-k from THIS state's slab (rows [S, S + cap) hold
        # the entries; each consumer overwrites the chunk / carry rows with its own window KV -- the source is done with them)
        idx = pub["idx"]
        if getattr(self, "_shared_mask_bufs", None) is not None:
            # trace islands: the consumers' island replays run while the top-k lives -> a persistent buffer, allocated
            # before the capture (precreate_shared_masks), like path B's masks (DS4F-0262 class)
            buf = getattr(self, "_shared_idx_buf", None)
            assert (
                buf is not None
            ), "path A with islands needs the warm-up to have run a path-A chunk (precreate_shared_masks)"
            copy_into(buf, idx)
            ttnn.deallocate(idx)
            idx = pub["idx"] = buf
        self.shared = dict(path="A", idx=idx, state=state, entry_count=state.entry_count)
        return out
    if self.sparse_path:
        self._pb_template = self.debug_last_selection  # path A mode: the path-B chunks' mask template for the islands
    # what this chunk's consumers read (DS41F-0037): the selection mask over the live entries and those entries
    w = _live_width(self, state, state.entry_count)
    bufs = getattr(self, "_shared_mask_bufs", None)
    if bufs is not None:
        # trace islands (V41_PREFILL_ISLANDS): the consumers' island replays run while this lives, so the mask goes into a
        # persistent buffer (allocated before the capture) and the entries are re-sliced by each consumer in its own eager
        # window from the persistent state
        mask = self.debug_last_selection
        copy_into(bufs[w], mask)
        ttnn.deallocate(mask)
        self.debug_last_selection = None
        self.shared = dict(mask_sel=bufs[w], compressed=None, state=state, width=w, entry_count=state.entry_count)
        return out
    comp = (
        state.compressed_kv
        if w == int(state.compressed_kv.shape[2])
        else ttnn.slice(state.compressed_kv, [0, 0, 0, 0], [1, 1, w, self.head_dim])
    )
    # ``state``: a decoder Reindex layer scores this source's index keys (DS41F-0073)
    self.shared = dict(
        mask_sel=self.debug_last_selection, compressed=comp, state=state, width=w, entry_count=state.entry_count
    )
    return out


def _precreate_shared_masks(self, state) -> None:
    """Islands: one persistent selection-mask buffer per live width this source can take (multiples of the live quantum
    up to the dense capacity), shaped like the last eager mask (the warm-up's), padded / sliced on the width axis.
    DS41F-0067 path A: masks only for the widths a PATH-B chunk can take (the context before it < window + top-k worth of
    entries, plus its own entries) -- one or two instead of capacity / quantum (DS41F-0065: O(cap^2) DRAM) -- and ONE
    persistent top-k buffer shaped like the warm-up's last path-A top-k."""
    template = getattr(self, "_pb_template", None) if self.sparse_path else self.debug_last_selection
    assert template is not None, "run an eager chunk before enabling islands"
    cap = int(state.compressed_kv.shape[2])
    q = self.live_quantum if (self.live_extent and self.live_quantum > 0) else cap
    widths = sorted({min(cap, k * q) for k in range(1, -(-cap // q) + 1)} | {cap})
    if self.sparse_path:
        rate = self.compressor.compress_rate
        chunk = int(self._slab_chunk_tokens)
        w_max = _live_width(self, state, -(-max(self.sliding_window, self.indexer.topk * rate) // rate) + chunk // rate)
        widths = [w for w in widths if w <= w_max] or [w_max]
        pub = getattr(self, "published", None)
        if pub is not None and pub.get("idx") is not None:
            self._shared_idx_buf = ttnn.clone(pub["idx"])
    w0 = int(template.shape[3])
    bufs = {}
    for w in widths:
        if w == w0:
            bufs[w] = ttnn.clone(template)
        elif w > w0:
            bufs[w] = ttnn.pad(template, padding=[(0, 0), (0, 0), (0, 0), (0, w - w0)], value=0.0)
        else:
            bufs[w] = ttnn.slice(template, [0, 0, 0, 0], [1, 1, int(template.shape[2]), w])
    self._shared_mask_bufs = bufs


V41CSA.precreate_shared_masks = _precreate_shared_masks


V41CSA.forward = _source_forward


# ---- DS41F-0068: path A3 for the V4.1 sources ----------------------------------------------------------------------------------
# TtCSA's forward_pre / glue_chunk run unchanged: the V4.1 compressor and index keys take TtCSA's compressor interface (empty
# priors), the cut masks are rate-aware (DS41F-0067). What differs: the compressor's entry-rope base is its PER-CHIP
# ``_entry_index`` (V4.1 pools each chip's rows, then gathers), A2 returns (y, top-k) -- the consumers' A2 islands read the
# top-k from this island's persistent output -- and V4.1 exports from the states after the chunk (kv_export.py), so none of
# V4-Flash's contract export prep runs in A2.
def _a3_carry(attn, state, sliding_kv_g, next_carry, real_len: int) -> None:
    """The epilogue's carry: A1's persistent next-carry (the last 128 rows of the full chunk), or -- a ragged FINAL chunk on the
    islands (DS41F-0068, PREFILL_V41_A3_RAGGED) -- the slice of A1's gathered K rows ending at the real length, through the
    carry index tabulated at alloc (one TILE step per real length, persistent) into a transient, so no host tensor and no new
    persistent buffer is built after the captures. The export reads this carry as the window ring (kv_export.py)."""
    seq_len = int(sliding_kv_g.shape[2])
    if int(real_len) == seq_len:
        ttnn.copy(next_carry, state.sliding_carry)  # NOT _update_in_place: next_carry is A1's persistent output
        return
    assert int(real_len) % ttnn.TILE_SIZE == 0 and int(real_len) >= attn.sliding_window, real_len
    start, end = attn._carry_index[attn._carry_key(real_len)]
    c = ttnn.slice(sliding_kv_g, start, end, slice_dim=2, num_devices=seq_len // attn.sliding_window)
    ttnn.copy(c, state.sliding_carry)
    ttnn.deallocate(c)


def _source_traceable(self) -> bool:
    # DS41F-0073: layer 20 as a full decoder source (ratio 1) runs its attention eagerly for now, so its consumers do too
    return A3 and self.sparse_path and self.fused_indexer and self.compressor.compress_rate > 1


def _source_prepare_chunk(self, state, real_len: int) -> None:
    """Eager, before A1 replays: this chunk's position scalars into the persistent buffers A1 reads (value-cached: no host
    write lands inside a capture that follows a matching prepare)."""
    self._push_scalar(self._slab_index[1], int(state.kv_actual))
    self.compressor._push_scalar(self.compressor._entry_index[1], int(state.entry_count))  # == fwp // rate


# WS3 (11-19 TTFT push, lever 1.6) V41_PREFILL_INDEX_QTP=1: the traced source's index path on each chip's OWN query rows.
# The fused scorer keeps every head resident, so the four TP chips of an SP row used to score, mask and rank the same 512
# query rows; the sparse core then kept only its TP quarter of the top-k (mesh_partition, the 16-heads-a-chip transpose).
# Here q / w (computed as before, then partitioned) and the score mask's rows are partitioned over TP first, so each chip
# scores, masks and ranks its 128 rows only and publishes the partitioned top-k, which its own and its consumers' sparse
# cores take as is. Every op is per row, so the selections are the same bits (checked by the device-check digests).
INDEX_QTP = os.environ.get("V41_PREFILL_INDEX_QTP", "0") == "1"
# V41_PREFILL_INDEX_DEC_QTP=1: the same for the EAGER fused scorer of the ratio-1 (decoder) index sources -- layer 20 and the
# Reindex layers 24 / 28 / 32 / 36 on the last-chunk decoder (select_indices_fused below)
INDEX_DEC_QTP = os.environ.get("V41_PREFILL_INDEX_DEC_QTP", "0") == "1"


def _select_static_qtp(ix, q_latent, hidden_states, cos, sin, index_k, score_mask, kv_bucket=None):
    """``TtCSAIndexer.select_indices_static`` with the query rows partitioned over TP after the projections: ->
    [1, 1, S_l / tp, topk] (this chip's quarter of its SP shard's rows)."""
    S_l = q_latent.shape[2]
    if ix._wq_b_all is None:
        ix._wq_b_all = ix._to_tt_linear_weight(ix._host_q_b_proj_weight, tp_shard_dim=None, cache_name="wq_b_all")
    q = ttnn.linear(q_latent, ix._wq_b_all, memory_config=ix.memory_config)
    q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
        q, num_heads=ix.n_heads, num_kv_heads=0, transpose_k_heads=False, memory_config=ix.memory_config
    )
    nope_dim = ix.head_dim - ix.rope_head_dim
    nope = ttnn.slice(q, [0, 0, 0, 0], [1, ix.n_heads, S_l, nope_dim])
    rope = ttnn.slice(q, [0, 0, 0, nope_dim], [1, ix.n_heads, S_l, ix.head_dim])
    rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, ix.trans_mat, is_decode_mode=False)
    q = ttnn.concat([nope, rope], dim=-1)
    if getattr(ix, "q_act_quant", None) is not None:  # V4.1 (DS41F-0048)
        q = ix.q_act_quant(q)
    w = ix.compressor._tp_all_reduce(ttnn.linear(hidden_states, ix.w_proj, memory_config=ix.memory_config))
    w = ttnn.multiply(w, ix.scale)
    q = ttnn.mesh_partition(q, dim=2, cluster_axis=ix.tp_axis)
    w = ttnn.mesh_partition(w, dim=2, cluster_axis=ix.tp_axis)
    cap = int(index_k.shape[2])
    sp = ix.sp_factor
    kv_len = (cap // 64) * 64
    keys = index_k
    if kv_bucket is not None and int(kv_bucket) < kv_len:
        kv_len = int(kv_bucket)
        assert kv_len % 64 == 0 and kv_len >= sp * S_l + ix.topk, (kv_len, sp, S_l)
        keys = ttnn.slice(index_k, [0, 0, 0, 0], [1, 1, kv_len, int(index_k.shape[3])])
    # this chip's rows of the score mask's [0, kv_len) columns: two copies either way; slice the bucket first while it is
    # under a quarter of the cache (512 x kv_len < 128 x cap), else partition the rows first (MEASURED, ws3_qtp_bench.py:
    # 1.80 vs 2.12 ms at the 65,536 bucket, 6.47 vs 7.36 at 294,912, 11.20 vs 13.50 at 524,288 -- whole index path)
    if int(score_mask.shape[2]) * ix.tp_factor == S_l:  # INDEX_QTP_MASK: the mask holds this chip's rows already
        mask_p = (
            score_mask if kv_len >= cap else ttnn.slice(score_mask, [0, 0, 0, 0], [1, 1, S_l // ix.tp_factor, kv_len])
        )
    elif kv_len >= cap:
        mask_p = ttnn.mesh_partition(score_mask, dim=2, cluster_axis=ix.tp_axis)
    elif 4 * kv_len > cap:
        part = ttnn.mesh_partition(score_mask, dim=2, cluster_axis=ix.tp_axis)
        mask_p = ttnn.slice(part, [0, 0, 0, 0], [1, 1, S_l // ix.tp_factor, kv_len])
        ttnn.deallocate(part)
    else:
        sl = ttnn.slice(score_mask, [0, 0, 0, 0], [1, 1, S_l, kv_len])
        mask_p = ttnn.mesh_partition(sl, dim=2, cluster_axis=ix.tp_axis)
        ttnn.deallocate(sl)
    # flat device rank (sp_row * tp + tp_col) x S_l / tp = the same query offset as [SP] x S_l + tp_col x S_l / tp: the
    # phantom window [kv_len - sp * S_l, kv_len) and every real column visible, as in the unpartitioned call
    logits = ttnn.experimental.indexer_score_dsa(
        q,
        keys,
        w,
        chunk_start_idx=kv_len - sp * S_l,
        kv_len=kv_len,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=64, head_group_size=0),
        seq_shard_axes=[],
    )  # [1, 1, S_l / tp, kv_len] bf16 ROW_MAJOR
    if keys is not index_k:
        ttnn.deallocate(keys)
    mask_rm = ttnn.to_layout(mask_p, ttnn.ROW_MAJOR_LAYOUT)
    if mask_p is not score_mask:
        ttnn.deallocate(mask_p)
    logits = ttnn.add(logits, mask_rm)
    return ttnn.experimental.topk_large_indices(logits, k=ix.topk, valid_length=kv_len)


# V41_PREFILL_INDEX_QTP_MASK=1 (with INDEX_QTP): the slot's score mask itself holds only each chip's TP-quarter rows
# ([1, 1, S_l / tp, cap] instead of every TP chip holding the same S_l rows), so the scorer's per-chunk slice + partition of
# the 512-row mask becomes one slice of the 128-row mask. The glue writes the same zeros / cut values into those rows (the cut
# constant is the TP quarter of the 512-row one, built before any capture), the reset copies a -inf constant of the same
# shape, and prefix reuse opens columns with a zero block of the mask's own row count. Same mask values row for row.
INDEX_QTP_MASK = INDEX_QTP and os.environ.get("V41_PREFILL_INDEX_QTP_MASK", "0") == "1"


def _qtp_mask_on(self) -> bool:
    return INDEX_QTP_MASK and self.tp_factor > 1


_tt_csa_score_mask = TtCSA._score_mask
_tt_csa_glue_chunk = TtCSA.glue_chunk
_tt_csa_precreate_constants = TtCSA.precreate_constants


def _score_mask_v41(self, state, S_l: int):
    if state.score_mask is None and _qtp_mask_on(self):
        return _tt_csa_score_mask(self, state, int(S_l) // self.tp_factor)
    return _tt_csa_score_mask(self, state, S_l)


def _cut_tile_part(self, S_l: int, n_cols: int):
    """The TP quarter (this chip's rows) of the indexer's full-width TILE cut block, cached next to it."""
    key = ("tile_qtp", int(S_l), int(n_cols))
    cut = self.indexer._cut_masks.get(key)
    if cut is None:
        cut = self.indexer._cut_masks[key] = ttnn.mesh_partition(
            self.indexer._chunk_cut_mask_tile(S_l, n_cols), dim=2, cluster_axis=self.tp_axis
        )
    return cut


def _source_precreate_constants_v41(self, state, seq_local: int, real_len: int) -> None:
    _tt_csa_precreate_constants(self, state, seq_local, real_len)
    if _qtp_mask_on(self):
        rate = self.compressor.compress_rate
        self.indexer._zeros_block_tile(int(state.score_mask.shape[2]), real_len // rate)
        _cut_tile_part(self, seq_local, real_len // rate)


def _source_glue_chunk_v41(self, state, outs, real_len: int, export=None) -> None:
    """TtCSA.glue_chunk with the score-mask writes on the mask's own (TP-quarter) rows under INDEX_QTP_MASK; else as is."""
    if not _qtp_mask_on(self):
        return _tt_csa_glue_chunk(self, state, outs, real_len, export)
    q, q_latent, sliding_kv_g, next_carry, entries, keys, cos, sin, kv_rm, entries_rm, carry_rm = outs
    rate = self.compressor.compress_rate
    n_new = real_len // rate
    seq_pad_global = sliding_kv_g.shape[2]
    S_l = q.shape[2]
    if state.entry_count + entries.shape[2] <= state.compressed_kv.shape[2]:  # DS4F-0295
        ttnn.kv_cache.fill_cache_for_user_(state.compressed_kv, entries, 0, update_idx=state.entry_count)
    ttnn.kv_cache.fill_cache_for_user_(state.index_k, keys, 0, update_idx=state.entry_count)
    self._slab_write_rm(state, entries_rm, seq_pad_global + state.entry_count)
    self._slab_write_rm(state, kv_rm, 0)
    self._slab_write_rm(state, carry_rm, seq_pad_global + self._slab_cap)
    mask = self._score_mask(state, S_l)
    rows = int(mask.shape[2])
    if state.entry_count > state.mask_zeroed_upto:
        c0, n = state.mask_zeroed_upto, state.entry_count - state.mask_zeroed_upto
        ttnn.experimental.slice_write(
            self.indexer._zeros_block_tile(rows, n),
            mask,
            start=[0, 0, 0, c0],
            end=[1, 1, rows, c0 + n],
            step=[1, 1, 1, 1],
        )
        state.mask_zeroed_upto = state.entry_count
    if n_new > 0:
        n_cut = min(-(-n_new // ttnn.TILE_SIZE) * ttnn.TILE_SIZE, int(entries.shape[2]))
        if _RAGGED_FULL_CUT:
            n_cut = int(entries.shape[2])
        if n_cut == int(entries.shape[2]):
            cut, transient = _cut_tile_part(self, S_l, n_cut), None
        else:
            transient = self.indexer._chunk_cut_mask_tile(S_l, n_cut, cache=False)
            cut = ttnn.mesh_partition(transient, dim=2, cluster_axis=self.tp_axis)
        ttnn.experimental.slice_write(
            cut,
            mask,
            start=[0, 0, 0, state.entry_count],
            end=[1, 1, rows, state.entry_count + n_cut],
            step=[1, 1, 1, 1],
        )
        if transient is not None:
            ttnn.deallocate(cut)
            ttnn.deallocate(transient)


_tt_csa_sparse_core = TtCSA._sparse_core
_tt_csa_window_indices = TtCSA._window_indices


def _window_indices_v41(self, seq_local: int):
    """TtCSA's window indices; under INDEX_QTP also their TP-partitioned copy, built with them (an eager warm-up, before
    any capture: a persistent tensor allocated after a capture can sit on its intermediates, DS4F-0262)."""
    dev = _tt_csa_window_indices(self, seq_local)
    if (INDEX_QTP or INDEX_DEC_QTP) and self.tp_factor > 1:
        part = self.__dict__.setdefault("_window_idx_part", {})
        if seq_local not in part:
            part[seq_local] = ttnn.mesh_partition(dev, dim=2, cluster_axis=self.tp_axis)
    return dev


def _sparse_core_v41(self, q, topk_idx, cos, sin, slab_rm, *, seq_len: int, chunk0: bool = False):
    """TtCSA._sparse_core, taking an INDEX_QTP top-k (already this chip's TP quarter of the rows) as is: the window
    indices' matching quarter is concatenated and the TP partition of the index rows is skipped; the rest is unchanged.
    """
    batch, seq_local = q.shape[0], q.shape[2]
    if not (
        (INDEX_QTP or INDEX_DEC_QTP) and self.tp_factor > 1 and int(topk_idx.shape[2]) * self.tp_factor == seq_local
    ):
        return _tt_csa_sparse_core(self, q, topk_idx, cos, sin, slab_rm, seq_len=seq_len, chunk0=chunk0)
    assert not chunk0, "INDEX_QTP: chunk 0 stays on the full-row path"
    heads_local = self.num_heads // self.tp_factor
    assert heads_local % ttnn.TILE_SIZE != 0, "INDEX_QTP assumes the head->seq transpose (V4.1: 16 heads a chip)"
    ent = ttnn.typecast(topk_idx, ttnn.int32)
    ent = ttnn.add(ent, seq_len)  # entries start at slab row S
    ent = ttnn.typecast(ent, ttnn.uint32)
    if ent.layout != ttnn.ROW_MAJOR_LAYOUT:
        ent = ttnn.to_layout(ent, ttnn.ROW_MAJOR_LAYOUT)
    self._window_indices(seq_local)
    idx = ttnn.concat([self._window_idx_part[seq_local], ent], dim=3)  # [1, 1, S_l / tp, 128 + topk]
    q = ttnn.experimental.all_to_all_async_generic(
        q,
        in_dim=1,
        out_dim=2,
        num_links=self.ccl_num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=self.tp_axis,
    )
    q_rm = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.transformer.sparse_sdpa(
        q_rm,
        slab_rm,
        idx,
        self.head_dim,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
        scale=self.scaling,
        k_chunk_size=128,
        cache_batch_idx=0,
        attention_sink=self._sink_column(True),
    )
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    out = ttnn.experimental.all_to_all_async_generic(
        out,
        in_dim=2,
        out_dim=1,
        num_links=self.ccl_num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=self.tp_axis,
    )
    nope_dim = self.head_dim - self.rope_head_dim
    nope = ttnn.slice(out, [0, 0, 0, 0], [batch, heads_local, seq_local, nope_dim])
    rope = ttnn.slice(out, [0, 0, 0, nope_dim], [batch, heads_local, seq_local, self.head_dim])
    rope = ttnn.experimental.rotary_embedding_llama(rope, cos, ttnn.neg(sin), self.trans_mat, is_decode_mode=False)
    return ttnn.concat([nope, rope], dim=-1)


def _source_forward_attn(
    self, hidden_states, outs, state, real_len: int, chunk0: bool = False, _t=None, kv_bucket=None
):
    """Island A2 (chunk-invariant): the static fused scorer + top-k over the key cache (or its first ``kv_bucket`` columns),
    the sparse gather over the persistent slab, V's un-RoPE and the output projection -> (y, top-k). ``chunk0``
    (A3_CHUNK0, DS41F-0073): the A2c0 variant -- the index rows repacked for chunk 0 (TtCSA._chunk0_pack)."""
    assert A3_CHUNK0 or not chunk0, "V4.1 runs chunk 0 eagerly (path B) unless PREFILL_V41_A3_CHUNK0"
    q, q_latent, sliding_kv_g, _nc, entries, keys, cos, sin = outs[:8]
    select = _select_static_qtp if (INDEX_QTP and not chunk0 and self.tp_factor > 1) else None
    if select is not None:
        idx = select(
            self.indexer, q_latent, hidden_states, cos, sin, state.index_k, state.score_mask, kv_bucket=kv_bucket
        )
    else:
        idx = self.indexer.select_indices_static(
            q_latent, hidden_states, cos, sin, state.index_k, state.score_mask, kv_bucket=kv_bucket
        )
    attn = self._sparse_core(q, idx, cos, sin, state.slab_rm, seq_len=sliding_kv_g.shape[2], chunk0=chunk0)
    return self._o_proj(attn), idx


def _source_epilogue_chunk(self, state, outs, a2, export, real_len: int) -> None:
    """Eager, after A2: the counters, the carry (A1's persistent next-carry copied into the state's buffer), and what this
    chunk's consumers read (``shared``: the path, this slot's state, the entry count after the chunk)."""
    assert export is None, "V4.1 exports from the states after the chunk (kv_export.py)"
    state.entry_count += int(real_len) // self.compressor.compress_rate
    state.kv_actual += int(real_len)
    _a3_carry(self, state, outs[2], outs[3], real_len)
    self.last_path = "A3"
    self.shared = dict(path="A3", idx=a2[1], state=state, entry_count=state.entry_count)


def _source_alloc_state(self, max_seq_len: int, batch: int = 1, chunk_tokens: int | None = None, slot: int = 0):
    state = TtCSA.alloc_state(self, max_seq_len, batch, chunk_tokens, slot)
    if A3 and self.sparse_path:
        # the FIRST traced chunk's glue opens every entry column before it at once (entry_count - mask_zeroed_upto: one chunk's
        # entries per chunk after that) -- its zeros block must exist before any capture (DS4F-0262 class)
        chunk, rate = int(chunk_tokens or max_seq_len), self.compressor.compress_rate
        need = max(self.sliding_window, self.indexer.topk * rate)
        self.indexer._zeros_block_tile(chunk // self.sp_factor, (-(-need // chunk) * chunk) // rate)
    return state


V41CSA.alloc_state = _source_alloc_state


# ---- DS41F-0068: path A3 for the sliding-window layers 0 / 1 (PREFILL_V41_A3_SWA) -----------------------------------------------
# TtHCA's protocol with no compressor: prepare_chunk pushes the rope base, A1 = q / window-KV stems, no glue; A2 = the window
# SDPA over [carry | chunk] + o-proj (V4-Flash's A2 also builds the contract export ring, which V4.1 does not use: kv_export.py
# reads the carry); the epilogue takes the carry at the real length (a ragged final chunk) and advances the counter.
A3_SWA = (
    os.environ.get("PREFILL_V41_A3_SWA", "0") == "1"
)  # default off: no gain (127k +96, 254k +23 ms per chunk, DS41F-0068)


def _swa_traceable(self) -> bool:
    return A3 and A3_SWA


def _swa_forward_attn(self, hidden_states, outs, state, real_len: int, chunk0: bool = False, **_kw):
    assert not chunk0, "V4.1 runs chunk 0 eagerly"
    q, sliding_kv, _e, _m, cos, sin = outs
    attn, next_carry, sliding_kv_g = self._attention(
        q, sliding_kv, None, None, cos, sin, carry=state.sliding_carry, kv_actual=1, real_len=real_len
    )
    return self._o_proj(attn), next_carry, sliding_kv_g


def _swa_epilogue_chunk(self, state, outs, a2, export, real_len: int) -> None:
    assert export is None, "V4.1 exports from the states after the chunk (kv_export.py)"
    _a3_carry(self, state, a2[2], a2[1], real_len)
    state.kv_actual += int(real_len)


V41SWA.traceable = _swa_traceable
V41SWA.forward_attn = _swa_forward_attn
V41SWA.epilogue_chunk = _swa_epilogue_chunk
V41SWA.supports_chunk0_island = False
V41SWA.capacity_buckets = lambda self, state, seq_local: []
V41CSA.traceable = _source_traceable
V41CSA.prepare_chunk = _source_prepare_chunk
V41CSA.forward_attn = _source_forward_attn
V41CSA.epilogue_chunk = _source_epilogue_chunk
V41CSA.supports_chunk0_island = A3_CHUNK0


class V41CSAConsumer(_V41QStem, TtCSA):
    """V4.1 ``Reuse`` layer (encoder: 3-7, 9-13, 15-19): its own q / window KV / sinks / o-proj, attending over [window |
    the SOURCE's top-512 of the source's entries] -- model.py's ``shared_attn.compress_kv`` + ``shared_attn.topk_idxs``.
    The source runs earlier in the same chunk (layers run in order); its ``shared`` dict carries the chunk's selection mask
    (path B: 0 / -inf over the live entries) and the live entries. No compressor, no indexer, no KV write of its own."""

    def __init__(self, device, *, source: V41CSA, **kwargs):
        # TtCSA wants a compressor / indexer; the source's give the rate and the tables this layer's masks are sized by
        super().__init__(device, compressor=source.compressor, indexer=source.indexer, sparse_path=False, **kwargs)
        self.source = source

    @classmethod
    def from_weights(
        cls,
        mesh_device,
        cfg,
        layer: int,
        w: dict,
        rotary_emb,
        *,
        source: V41CSA,
        sp_axis=0,
        tp_axis=1,
        topology=ttnn.Topology.Linear,
        weight_cache_path=None,
    ) -> "V41CSAConsumer":
        r = cfg.role(layer)
        # DS41F-0073: a decoder Reuse layer's ``source`` is its INDEX source (layer 20 or the latest Reindex layer), whose
        # ``shared`` carries layer 20's entries (the KV source) with that layer's top-k
        assert r.mode == "reuse", f"layer {layer} ({r}) is not a Reuse layer"
        kw = _common_kwargs(
            cfg,
            w,
            rotary_emb,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            weight_cache_path=weight_cache_path,
            layer=layer,
            name="attn",
        )
        return cls(mesh_device, source=source, **kw)

    def alloc_state(self, max_seq_len: int, batch: int = 1, chunk_tokens: int | None = None, slot: int = 0):
        from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCAState, _rope_table_tokens

        rate = self.compressor.compress_rate
        chunk = chunk_tokens or max_seq_len
        entries = -(-int(max_seq_len) // rate)
        capacity = -(-entries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE + max(chunk // rate, chunk + 64)
        capacity = -(-capacity // 128) * 128  # == the source's (TtCSA.alloc_state)
        self._build_carry_index(chunk)
        self._build_masks(chunk, capacity)
        self._slab_rope = self._build_rope_table(_rope_table_tokens(max_seq_len, chunk), 1)
        self._slab_index = self._rope_index_base(chunk // self.sp_factor)
        # DS41F-0067 path A reads the SOURCE's slab [chunk | entries (capacity) | carry]: the same geometry; its constants are
        # built here, before any island capture (a lazily created one could sit on a replay's addresses, DS4F-0262)
        self._slab_chunk_tokens, self._slab_cap = int(chunk), int(capacity)
        if PATH_A:
            heads_local = self.num_heads // self.tp_factor
            self._window_indices(chunk // self.sp_factor)
            self._sink_column(self.tp_factor > 1 and heads_local % ttnn.TILE_SIZE != 0)
        state = TtHCAState(
            compressed_kv=None,
            sliding_carry=self._from_torch(torch.zeros(batch, 1, self.sliding_window, self.head_dim)),
            tail=None,
            max_seq_len=max_seq_len,
        )
        state.entry_count = 0
        # DS41F-0068: the block replays THIS slot's attention islands (they read its source's slot state)
        state.slot = int(slot)
        return state

    def forward(self, hidden_states, seq_len_actual=None, *, state, export=None):
        sh = self.source.shared
        assert sh.get("path") != "A3", "the source replayed its A3 islands this chunk: the consumer must too (block)"
        real_len = hidden_states.shape[2] * self.sp_factor if seq_len_actual is None else int(seq_len_actual)
        cos, sin = self._rope_gather(self._slab_rope, self._rope_index(self._slab_index, state.kv_actual))
        q = self._q_stem(hidden_states, cos, sin)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)
        if sh.get("path") == "A":
            # DS41F-0067: the source's top-k over the source's entries, this layer's own window: its chunk KV and carry go
            # into the source's slab rows [0, S) / [S + cap, S + cap + 128) (the source already attended this chunk)
            attn, next_carry, slab = self._attention_sparse(
                q,
                sliding_kv,
                None,
                sh["idx"],
                cos,
                sin,
                carry=state.sliding_carry,
                real_len=real_len,
                state=sh["state"],
            )
            if export is not None:
                self._export_ring(export, slab, state.sliding_carry, state.kv_actual, real_len)
            state.kv_actual += real_len
            state.entry_count = sh["entry_count"]
            self._update_in_place(state.sliding_carry, next_carry)
            return self._o_proj(attn)
        comp, own_comp = sh["compressed"], False
        if comp is None:  # islands: slice the source's persistent entries here (freed below, before the next replay)
            src = sh["state"].compressed_kv
            w = int(sh["width"])
            comp = src if w == int(src.shape[2]) else ttnn.slice(src, [0, 0, 0, 0], [1, 1, w, self.head_dim])
            own_comp = comp is not src
        attn, next_carry, slab = self._attention(
            q,
            sliding_kv,
            comp,
            sh["mask_sel"],
            cos,
            sin,
            carry=state.sliding_carry,
            kv_actual=state.kv_actual,
            real_len=real_len,
        )
        if own_comp:
            ttnn.deallocate(comp)
        if export is not None:
            self._export_ring(export, slab, state.sliding_carry, state.kv_actual, real_len)
        state.kv_actual += real_len
        state.entry_count = sh["entry_count"]
        self._update_in_place(state.sliding_carry, next_carry)
        return self._o_proj(attn)

    # ---- DS41F-0068: path A3 (a consumer's attention as two trace islands per slot) --------------------------------------
    # A1c: rope gather, q / window-KV stems, the SP gather of the chunk's K rows, the next carry, and ROW_MAJOR copies of the K
    # rows and of the previous carry; glue (eager): those two into the SOURCE's slab rows [0, S) and [S + cap, S + cap + 128)
    # (exactly the eager path A's _slab_write pieces); A2c: the sparse gather with the source's A2 top-k (its persistent
    # output, read in place) over the source's slab, V's un-RoPE, o-proj; epilogue (eager): counters + the carry.
    supports_chunk0_island = A3_CHUNK0

    def traceable(self) -> bool:
        return A3 and PATH_A and self.source.traceable()

    def trace_ready(self, state) -> bool:
        rate = self.compressor.compress_rate
        return int(state.kv_actual) >= max(self.sliding_window, self.indexer.topk * rate)

    def capacity_buckets(self, state, seq_local: int) -> list:
        return []  # no scorer: one A2c island

    def precreate_constants(self, state, seq_local: int, real_len: int) -> None:
        heads_local = self.num_heads // self.tp_factor
        self._window_indices(seq_local)
        self._sink_column(self.tp_factor > 1 and heads_local % ttnn.TILE_SIZE != 0)

    def prepare_chunk(self, state, real_len: int) -> None:
        self._push_scalar(self._slab_index[1], int(state.kv_actual))

    def forward_pre(self, hidden_states, state, real_len: int, _t=None):
        """-> (q, sliding_kv_gathered, next_carry, cos, sin, kv_rm, carry_rm), all persistent island outputs."""
        cos, sin = self._rope_gather(self._slab_rope, self._rope_index(self._slab_index, state.kv_actual))
        q = self._q_stem(hidden_states, cos, sin)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)
        if self.sp_factor > 1:
            sliding_kv = ttnn.experimental.all_gather_async(
                sliding_kv,
                dim=2,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
                num_links=self.ccl_num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=self.tp_ccl_topology,
                cluster_axis=self.sp_axis,
            )
        seq_len = hidden_states.shape[2] * self.sp_factor
        start, end = self._carry_index[self._carry_key(real_len)]
        next_carry = ttnn.slice(sliding_kv, start, end, slice_dim=2, num_devices=seq_len // self.sliding_window)
        kv_rm = self._rm_pieces(sliding_kv, int(sliding_kv.shape[2]), ttnn.bfloat16)[0]
        carry_rm = self._rm_pieces(state.sliding_carry, self.sliding_window, ttnn.bfloat16)[0]
        return q, sliding_kv, next_carry, cos, sin, kv_rm, carry_rm

    def glue_chunk(self, state, outs, real_len: int, export=None, src_state=None) -> None:
        assert export is None and src_state is not None
        _q, sliding_kv_g, _nc, _cos, _sin, kv_rm, carry_rm = outs
        self._slab_write_rm(src_state, kv_rm, 0)
        self._slab_write_rm(src_state, carry_rm, int(sliding_kv_g.shape[2]) + self._slab_cap)

    def forward_attn(
        self, hidden_states, outs, state, real_len: int, src_idx=None, src_state=None, chunk0: bool = False, **_kw
    ):
        q, sliding_kv_g, _nc, cos, sin = outs[:5]
        attn = self._sparse_core(
            q, src_idx, cos, sin, src_state.slab_rm, seq_len=sliding_kv_g.shape[2], chunk0=chunk0
        )  # chunk0: the source's chunk-0 top-k, repacked the same way (DS41F-0073)
        return (self._o_proj(attn),)

    def epilogue_chunk(self, state, outs, a2, export, real_len: int) -> None:
        assert export is None
        sh = self.source.shared
        assert sh.get("path") == "A3", "a traced consumer chunk needs its source's traced chunk first"
        state.kv_actual += int(real_len)
        state.entry_count = sh["entry_count"]
        _a3_carry(self, state, outs[1], outs[2], real_len)


class V41CSAReindex(_V41QStem, TtCSA):
    """V4.1 ``Reindex`` layer (decoder 24 / 28 / 32 / 36, DS41F-0073): its own q / window KV / sinks / o-proj and its OWN
    indexer (``indexer.wq_b`` / ``weights_proj``) scoring the KV source's (layer 20's) index keys -- model.py
    ``Indexer.forward`` with ``owns_k`` False -- then attention over [window | its top-512 of layer 20's entries]. No
    compressor and no key write of its own: layer 20 ran earlier in the same chunk, and its ``shared`` names the state.
    Publishes ``shared`` the way a source does, for the Reuse layers that take its top-k.

    model.py's candidate pre-filter (layer 20 keeps each query's top 2048 blocks of 8 entries, the Reindex layers score
    inside them) is NOT applied: it is a no-op up to 16384 entries, and the decode ring does not apply it either (tt-blaze
    L5), so the prefill matches the ring's decode."""

    def __init__(self, device, *, source: V41CSA, indexer, sparse_path: bool, **kwargs):
        super().__init__(device, compressor=source.compressor, indexer=indexer, sparse_path=sparse_path, **kwargs)
        self.source = source
        self._src_state = None  # the source's state of the last chunk (the islands' mask templates are sized by it)

    @classmethod
    def from_weights(
        cls,
        mesh_device,
        cfg,
        layer: int,
        w: dict,
        rotary_emb,
        *,
        source: V41CSA,
        sp_axis=0,
        tp_axis=1,
        topology=ttnn.Topology.Linear,
        weight_cache_path=None,
    ) -> "V41CSAReindex":
        r = cfg.role(layer)
        assert r.mode == "reindex", f"layer {layer} ({r.mode}) is not a Reindex layer"
        mesh = dict(sp_axis=sp_axis, tp_axis=tp_axis, topology=topology)
        # the key provider is the source's (its CCL handles / fp32 config / rotation); this layer never calls it
        indexer = TtCSAIndexer(
            mesh_device,
            compressor=source.indexer.compressor,
            q_b_proj_weight=w["attn.indexer.wq_b.weight"],
            weights_proj_weight=w["attn.indexer.weights_proj.weight"].float(),
            n_heads=cfg.index_n_heads,
            head_dim=cfg.index_head_dim,
            rope_head_dim=cfg.rope_head_dim,
            topk=cfg.index_topk,
            **mesh,
        )
        if INDEX_FP4:
            indexer.q_act_quant = fp4_act_fake_quant
        kw = _common_kwargs(
            cfg,
            w,
            rotary_emb,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            weight_cache_path=weight_cache_path,
            layer=layer,
            name="attn",
        )
        return cls(mesh_device, source=source, indexer=indexer, sparse_path=PATH_A, **kw)

    def alloc_state(self, max_seq_len: int, batch: int = 1, chunk_tokens: int | None = None, slot: int = 0):
        """The consumer's state (its own window carry) and the slab constants of the source's geometry."""
        from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import TtHCAState, _rope_table_tokens

        rate = self.compressor.compress_rate
        chunk = chunk_tokens or max_seq_len
        entries = -(-int(max_seq_len) // rate)
        capacity = -(-entries // ttnn.TILE_SIZE) * ttnn.TILE_SIZE + max(chunk // rate, chunk + 64)
        capacity = -(-capacity // 128) * 128  # == the source's (TtCSA.alloc_state)
        self._build_carry_index(chunk)
        self._build_masks(chunk, capacity)
        self._slab_rope = self._build_rope_table(_rope_table_tokens(max_seq_len, chunk), 1)
        self._slab_index = self._rope_index_base(chunk // self.sp_factor)
        self._slab_chunk_tokens, self._slab_cap = int(chunk), int(capacity)
        if self.sparse_path:  # path A's constants, before any island capture (DS4F-0262)
            heads_local = self.num_heads // self.tp_factor
            self._window_indices(chunk // self.sp_factor)
            self._sink_column(self.tp_factor > 1 and heads_local % ttnn.TILE_SIZE != 0)
        state = TtHCAState(
            compressed_kv=None,
            sliding_carry=self._from_torch(torch.zeros(batch, 1, self.sliding_window, self.head_dim)),
            tail=None,
            max_seq_len=max_seq_len,
        )
        state.entry_count = 0
        state.slot = int(slot)
        return state

    def traceable(self) -> bool:
        return False  # eager attention between the mHC / MoE islands (no A3 for the decoder yet)

    def precreate_shared_masks(self, state) -> None:
        assert self._src_state is not None, "run an eager chunk before enabling islands"
        _precreate_shared_masks(self, self._src_state)

    def forward(self, hidden_states, seq_len_actual=None, *, state, export=None):
        sh = self.source.shared
        src = sh["state"]
        self._src_state = src
        rate = self.compressor.compress_rate
        seq_pad_global = hidden_states.shape[2] * self.sp_factor
        real_len = seq_pad_global if seq_len_actual is None else int(seq_len_actual)
        n_new = real_len // rate
        E = int(sh["entry_count"])  # the source's entries AFTER this chunk (it already wrote this chunk's)
        E0 = E - n_new
        assert E0 * rate == int(state.kv_actual), (E0, rate, state.kv_actual)
        cos, sin = self._rope_gather(self._slab_rope, self._rope_index(self._slab_index, state.kv_actual))
        q, q_latent = self._q_stem(hidden_states, cos, sin, return_latent=True)
        sliding_kv = self._kv_stem(hidden_states, cos, sin)
        use_sparse = self.sparse_path and state.kv_actual >= max(self.sliding_window, self.indexer.topk * rate)
        if use_sparse:
            idx = self.indexer.select_indices_fused(
                q_latent, hidden_states, cos, sin, src.index_k, E0, n_new, seq_pad_global // rate
            )
            # the source is done with its slab's chunk / carry rows: this layer's own window KV goes there (as a consumer's)
            attn, next_carry, _slab = self._attention_sparse(
                q, sliding_kv, None, idx, cos, sin, carry=state.sliding_carry, real_len=real_len, state=src
            )
            self.last_path = "A"
            self.published = dict(path="A", idx=idx)
            if (
                getattr(self, "_shared_mask_bufs", None) is not None
            ):  # islands: the consumers' replays run while it lives
                buf = getattr(self, "_shared_idx_buf", None)
                assert buf is not None, "path A with islands needs a path-A warm-up chunk (precreate_shared_masks)"
                copy_into(buf, idx)
                ttnn.deallocate(idx)
                idx = buf
            self.shared = dict(path="A", idx=idx, state=src, entry_count=E)
        else:
            w = _live_width(self, src, E)
            mask_block = self.compressor._mask_block(int(hidden_states.shape[2]), E0 * rate, real_len, width=w)
            keys = (
                src.index_k
                if w == int(src.index_k.shape[2])
                else ttnn.slice(src.index_k, [0, 0, 0, 0], [1, 1, w, self.indexer.head_dim])
            )
            comp = (
                src.compressed_kv
                if w == int(src.compressed_kv.shape[2])
                else ttnn.slice(src.compressed_kv, [0, 0, 0, 0], [1, 1, w, self.head_dim])
            )
            mask_sel = self.indexer.select(q_latent, hidden_states, cos, sin, keys, mask_block)
            if keys is not src.index_k:
                ttnn.deallocate(keys)
            ttnn.deallocate(mask_block)
            attn, next_carry, _slab = self._attention(
                q,
                sliding_kv,
                comp,
                mask_sel,
                cos,
                sin,
                carry=state.sliding_carry,
                kv_actual=state.kv_actual,
                real_len=real_len,
            )
            self.last_path = "B"
            self.published = dict(path="B", idx=None)
            if self.sparse_path:
                self._pb_template = mask_sel  # the path-B chunks' mask template for the islands
            bufs = getattr(self, "_shared_mask_bufs", None)
            if bufs is not None:
                copy_into(bufs[w], mask_sel)
                ttnn.deallocate(mask_sel)
                if comp is not src.compressed_kv:
                    ttnn.deallocate(comp)
                self.debug_last_selection = None
                self.shared = dict(mask_sel=bufs[w], compressed=None, state=src, width=w, entry_count=E)
            else:
                self.debug_last_selection = mask_sel
                self.shared = dict(mask_sel=mask_sel, compressed=comp, state=src, width=w, entry_count=E)
        state.kv_actual += real_len
        state.entry_count = E
        self._update_in_place(state.sliding_carry, next_carry)
        return self._o_proj(attn)


def _kv_only(self, hidden_states, seq_len_actual=None, *, state, slab: bool = False):
    """Layer 20 in the prefill (DS41F-0037): the decoder's shared KV and index keys from the encoder output -- the window KV
    stem, the ratio-1 compressor (``norm(wkv(x))`` + RoPE) and the index keys -- written to ``state`` with no attention, no
    o-proj and no FFN (the ring replays the prompt's tail through layers 20..39 itself).

    ``slab`` (DS41F-0073 marked decoder chunks, V41_PREFILL_DECODER_MARKED): ``state`` is layer 20's BLOCK state, whose full
    forward later runs on the decoder chunks of the same prompt -- its path-A sparse gather and its consumers' read the
    entries from the slab ``[chunk | entries | carry]``, not from ``compressed_kv``, so the entries go there too, at the row
    ``TtCSA.forward`` writes them (chunk + entry index). The KV-only layer of the default prefill has no slab (False).
    """
    real_len = hidden_states.shape[2] * self.sp_factor if seq_len_actual is None else int(seq_len_actual)
    rate = self.compressor.compress_rate
    assert state.kv_actual % ttnn.TILE_SIZE == 0 and state.kv_actual + real_len <= state.max_seq_len
    if KVONLY_SYNC:  # DS41F-0169 diagnostic: nothing in flight around layer 20's eager CCL ops
        ttnn.synchronize_device(self.device)
    fwp = state.entry_count * rate
    cos, sin = self._rope_gather(self._slab_rope, self._rope_index(self._slab_index, state.kv_actual))
    sliding_kv = self._kv_stem(hidden_states, cos, sin)
    entries, _, _ = self.compressor(hidden_states, real_len, fwp, state.prior_c, need_mask=False)
    keys, _, _ = self.indexer.compressor(hidden_states, real_len, fwp, state.prior_i, need_mask=False)
    ttnn.kv_cache.fill_cache_for_user_(state.compressed_kv, entries, 0, update_idx=state.entry_count)
    ttnn.kv_cache.fill_cache_for_user_(state.index_k, keys, 0, update_idx=state.entry_count)
    if slab and self.sparse_path:
        self._slab_write(state, entries, hidden_states.shape[2] * self.sp_factor + state.entry_count)
    if self.sp_factor > 1:
        sliding_kv = ttnn.experimental.all_gather_async(
            sliding_kv,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.tp_ccl_topology,
            cluster_axis=self.sp_axis,
        )
    seq_len = int(sliding_kv.shape[2])
    start, end = self._carry_index[self._carry_key(real_len)]
    next_carry = ttnn.slice(sliding_kv, start, end, slice_dim=2, num_devices=seq_len // self.sliding_window)
    state.entry_count += real_len // rate
    state.kv_actual += real_len
    self._update_in_place(state.sliding_carry, next_carry)
    if KVONLY_SYNC:
        ttnn.synchronize_device(self.device)


KVONLY_SYNC = os.environ.get("V41_PREFILL_KVONLY_SYNC", "0") == "1"
V41CSA.kv_only = _kv_only


# WS3 INDEX_QTP: the sources' and their consumers' sparse cores take a TP-partitioned top-k (identity when the flag is off)
_tt_csa_attention_sparse = TtCSA._attention_sparse


def _attention_sparse_v41(self, q, sliding_kv, compressed_kv, topk_idx, cos, sin, carry, real_len: int, state=None):
    """TtCSA._attention_sparse (the eager path A: the decoder's ratio-1 sources and their consumers, every warm-up chunk),
    taking a partitioned top-k (INDEX_DEC_QTP / INDEX_QTP) as is: the window indices' TP quarter is concatenated and the
    index rows' TP partition is skipped; the slab writes, the transposes and the un-RoPE are unchanged."""
    batch, seq_local = q.shape[0], q.shape[2]
    if not (
        (INDEX_QTP or INDEX_DEC_QTP) and self.tp_factor > 1 and int(topk_idx.shape[2]) * self.tp_factor == seq_local
    ):
        return _tt_csa_attention_sparse(self, q, sliding_kv, compressed_kv, topk_idx, cos, sin, carry, real_len, state)
    seq_len = seq_local * self.sp_factor
    heads_local = self.num_heads // self.tp_factor
    assert (
        heads_local % ttnn.TILE_SIZE != 0
    ), "a partitioned top-k assumes the head->seq transpose (V4.1: 16 heads a chip)"
    if self.sp_factor > 1:
        sliding_kv = ttnn.experimental.all_gather_async(
            sliding_kv,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=self.sp_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=self.sp_axis),
            num_links=self.ccl_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.tp_ccl_topology,
            cluster_axis=self.sp_axis,
        )
    start, end = self._carry_index[self._carry_key(real_len)]
    next_carry = ttnn.slice(sliding_kv, start, end, slice_dim=2, num_devices=seq_len // self.sliding_window)
    self._slab_write(state, sliding_kv, 0)
    self._slab_write(state, carry, seq_len + self._slab_cap)
    kv_rm = state.slab_rm
    ent = ttnn.typecast(topk_idx, ttnn.int32)
    ent = ttnn.add(ent, seq_len)  # entries start at slab row S
    ent = ttnn.typecast(ent, ttnn.uint32)
    if ent.layout != ttnn.ROW_MAJOR_LAYOUT:
        ent = ttnn.to_layout(ent, ttnn.ROW_MAJOR_LAYOUT)
    self._window_indices(seq_local)
    idx = ttnn.concat([self._window_idx_part[seq_local], ent], dim=3)  # [1, 1, S_l / tp, 128 + topk]
    q = ttnn.experimental.all_to_all_async_generic(
        q,
        in_dim=1,
        out_dim=2,
        num_links=self.ccl_num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=self.tp_axis,
    )
    q_rm = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.transformer.sparse_sdpa(
        q_rm,
        kv_rm,
        idx,
        self.head_dim,
        kv_format=ttnn.transformer.SparseKVFormat.BF16,
        scale=self.scaling,
        k_chunk_size=128,
        cache_batch_idx=0,
        attention_sink=self._sink_column(True),
    )
    out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
    out = ttnn.experimental.all_to_all_async_generic(
        out,
        in_dim=2,
        out_dim=1,
        num_links=self.ccl_num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=self.tp_axis,
    )
    nope_dim = self.head_dim - self.rope_head_dim
    nope = ttnn.slice(out, [0, 0, 0, 0], [batch, heads_local, seq_local, nope_dim])
    rope = ttnn.slice(out, [0, 0, 0, nope_dim], [batch, heads_local, seq_local, self.head_dim])
    rope = ttnn.experimental.rotary_embedding_llama(rope, cos, ttnn.neg(sin), self.trans_mat, is_decode_mode=False)
    return ttnn.concat([nope, rope], dim=-1), next_carry, sliding_kv


for _cls in (V41CSA, V41CSAConsumer, V41CSAReindex):
    _cls._sparse_core = _sparse_core_v41
    _cls._window_indices = _window_indices_v41
    _cls._attention_sparse = _attention_sparse_v41
V41CSA._score_mask = _score_mask_v41
V41CSA.glue_chunk = _source_glue_chunk_v41
V41CSA.precreate_constants = _source_precreate_constants_v41


# WS3 V41_PREFILL_INDEX_FUSED_BOUND=1: the EAGER fused scorer (select_indices_fused: the last-chunk decoder's ratio-1 index
# sources, layer 20 and the Reindex layers 24 / 28 / 32 / 36) took the WHOLE index-key cache as the op's key operand, so its
# output and its work spanned the cache (1,052,800 columns at MAXSEQ 1M) whatever the context. Here the cache is sliced to a
# ladder step (V41_PREFILL_INDEX_FUSED_LADDER columns, default 32768) at or above the scored length kv_len: the scored columns
# are the same per column, the top-k still ranks the first E (valid_length) -- the same bits (ws3_fused_bench.py).
INDEX_FUSED_BOUND = os.environ.get("V41_PREFILL_INDEX_FUSED_BOUND", "0") == "1"
FUSED_LADDER = int(os.environ.get("V41_PREFILL_INDEX_FUSED_LADDER", "32768"))
_tt_select_indices_fused = TtCSAIndexer.select_indices_fused


def _select_indices_fused_bounded(
    self, q_latent, hidden_states, cos, sin, index_k, entry_count: int, n_new: int, n_new_max: int
):
    E_tiles = -(-(int(entry_count) + int(n_new)) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    kv_len = -(-(E_tiles + self.sp_factor * int(q_latent.shape[2])) // 64) * 64
    cap = int(index_k.shape[2])
    width = min(cap, -(-kv_len // FUSED_LADDER) * FUSED_LADDER)
    if not INDEX_FUSED_BOUND or width >= cap:
        return _tt_select_indices_fused(self, q_latent, hidden_states, cos, sin, index_k, entry_count, n_new, n_new_max)
    keys = ttnn.slice(index_k, [0, 0, 0, 0], [1, 1, width, int(index_k.shape[3])])
    idx = _tt_select_indices_fused(self, q_latent, hidden_states, cos, sin, keys, entry_count, n_new, n_new_max)
    ttnn.deallocate(keys)
    return idx


_tt_chunk_cut_mask = TtCSAIndexer._chunk_cut_mask


def _chunk_cut_mask_v41(self, seq_local: int, n_cols: int):
    """TtCSAIndexer._chunk_cut_mask; under INDEX_DEC_QTP its TP quarter (this chip's rows) is built with it, so a constant
    precreated before the captures has its partition precreated too (DS4F-0262 class)."""
    dev = _tt_chunk_cut_mask(self, seq_local, n_cols)
    if INDEX_DEC_QTP and self.tp_factor > 1:
        key = ("rm_qtp", int(seq_local), int(n_cols))
        if key not in self._cut_masks:
            self._cut_masks[key] = ttnn.mesh_partition(dev, dim=2, cluster_axis=self.tp_axis)
    return dev


def _select_indices_fused_qtp(
    self, q_latent, hidden_states, cos, sin, index_k, entry_count: int, n_new: int, n_new_max: int
):
    """TtCSAIndexer.select_indices_fused on this chip's TP quarter of the query rows (q / w computed on all rows first, as
    INDEX_QTP): -> [1, 1, S_l / tp, topk], taken as is by the V4.1 sparse cores."""
    S_l = q_latent.shape[2]
    if self._wq_b_all is None:
        self._wq_b_all = self._to_tt_linear_weight(self._host_q_b_proj_weight, tp_shard_dim=None, cache_name="wq_b_all")
    q = ttnn.linear(q_latent, self._wq_b_all, memory_config=self.memory_config)
    q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
        q, num_heads=self.n_heads, num_kv_heads=0, transpose_k_heads=False, memory_config=self.memory_config
    )
    nope_dim = self.head_dim - self.rope_head_dim
    nope = ttnn.slice(q, [0, 0, 0, 0], [1, self.n_heads, S_l, nope_dim])
    rope = ttnn.slice(q, [0, 0, 0, nope_dim], [1, self.n_heads, S_l, self.head_dim])
    rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
    q = ttnn.concat([nope, rope], dim=-1)
    if getattr(self, "q_act_quant", None) is not None:  # V4.1 (DS41F-0048)
        q = self.q_act_quant(q)
    w = self.compressor._tp_all_reduce(ttnn.linear(hidden_states, self.w_proj, memory_config=self.memory_config))
    w = ttnn.multiply(w, self.scale)
    q = ttnn.mesh_partition(q, dim=2, cluster_axis=self.tp_axis)
    w = ttnn.mesh_partition(w, dim=2, cluster_axis=self.tp_axis)
    rows = S_l // self.tp_factor
    E = int(entry_count) + int(n_new)
    E_tiles = -(-E // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    kv_len = -(-(E_tiles + self.sp_factor * S_l) // 64) * 64
    assert kv_len <= int(index_k.shape[2]), (kv_len, index_k.shape)
    # flat device rank x S_l / tp = the unpartitioned call's SP offset + this chip's TP quarter: every real entry visible
    logits = ttnn.experimental.indexer_score_dsa(
        q,
        index_k,
        w,
        chunk_start_idx=E_tiles,
        kv_len=kv_len,
        program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=64, head_group_size=0),
        seq_shard_axes=[],
    )
    if n_new > 0:
        self._chunk_cut_mask(S_l, n_new_max)
        mask = self._cut_masks[("rm_qtp", int(S_l), int(n_new_max))]
        if n_new < n_new_max:
            mask = ttnn.slice(mask, [0, 0, 0, 0], [1, 1, rows, n_new])
        seg = ttnn.slice(logits, [0, 0, 0, entry_count], [1, 1, rows, E])
        seg = ttnn.add(seg, mask)
        ttnn.experimental.slice_write(seg, logits, start=[0, 0, 0, entry_count], end=[1, 1, rows, E], step=[1, 1, 1, 1])
    return ttnn.experimental.topk_large_indices(logits, k=self.topk, valid_length=E)


def _select_indices_fused_v41(
    self, q_latent, hidden_states, cos, sin, index_k, entry_count: int, n_new: int, n_new_max: int
):
    """INDEX_FUSED_BOUND (the key operand sliced to the ladder step) and, for a ratio-1 (decoder) source, INDEX_DEC_QTP."""
    ratio1 = int(n_new_max) == self.sp_factor * int(q_latent.shape[2])  # rate 1: the chunk's every token is an entry
    if not (INDEX_DEC_QTP and ratio1 and self.tp_factor > 1):
        return _select_indices_fused_bounded(
            self, q_latent, hidden_states, cos, sin, index_k, entry_count, n_new, n_new_max
        )
    E_tiles = -(-(int(entry_count) + int(n_new)) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    kv_len = -(-(E_tiles + self.sp_factor * int(q_latent.shape[2])) // 64) * 64
    cap = int(index_k.shape[2])
    width = min(cap, -(-kv_len // FUSED_LADDER) * FUSED_LADDER) if INDEX_FUSED_BOUND else cap
    keys = index_k if width >= cap else ttnn.slice(index_k, [0, 0, 0, 0], [1, 1, width, int(index_k.shape[3])])
    idx = _select_indices_fused_qtp(self, q_latent, hidden_states, cos, sin, keys, entry_count, n_new, n_new_max)
    if keys is not index_k:
        ttnn.deallocate(keys)
    return idx


TtCSAIndexer._chunk_cut_mask = _chunk_cut_mask_v41
TtCSAIndexer.select_indices_fused = _select_indices_fused_v41
