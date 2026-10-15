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
from models.demos.deepseek_v3_d_p.tt.v4.attention.csa import TtCSA, TtCSAIndexer  # noqa: E402
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


def _source_forward_attn(
    self, hidden_states, outs, state, real_len: int, chunk0: bool = False, _t=None, kv_bucket=None
):
    """Island A2 (chunk-invariant): the static fused scorer + top-k over the key cache (or its first ``kv_bucket`` columns),
    the sparse gather over the persistent slab, V's un-RoPE and the output projection -> (y, top-k)."""
    assert not chunk0, "V4.1 runs chunk 0 eagerly (path B)"
    q, q_latent, sliding_kv_g, _nc, entries, keys, cos, sin = outs[:8]
    idx = self.indexer.select_indices_static(
        q_latent, hidden_states, cos, sin, state.index_k, state.score_mask, kv_bucket=kv_bucket
    )
    attn = self._sparse_core(q, idx, cos, sin, state.slab_rm, seq_len=sliding_kv_g.shape[2])
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
V41CSA.supports_chunk0_island = False


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
    supports_chunk0_island = False

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

    def forward_attn(self, hidden_states, outs, state, real_len: int, src_idx=None, src_state=None, **_kw):
        q, sliding_kv_g, _nc, cos, sin = outs[:5]
        attn = self._sparse_core(q, src_idx, cos, sin, src_state.slab_rm, seq_len=sliding_kv_g.shape[2])
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


def _kv_only(self, hidden_states, seq_len_actual=None, *, state):
    """Layer 20 in the prefill (DS41F-0037): the decoder's shared KV and index keys from the encoder output -- the window KV
    stem, the ratio-1 compressor (``norm(wkv(x))`` + RoPE) and the index keys -- written to ``state`` with no attention, no
    o-proj and no FFN (the ring replays the prompt's tail through layers 20..39 itself)."""
    real_len = hidden_states.shape[2] * self.sp_factor if seq_len_actual is None else int(seq_len_actual)
    rate = self.compressor.compress_rate
    assert state.kv_actual % ttnn.TILE_SIZE == 0 and state.kv_actual + real_len <= state.max_seq_len
    fwp = state.entry_count * rate
    cos, sin = self._rope_gather(self._slab_rope, self._rope_index(self._slab_index, state.kv_actual))
    sliding_kv = self._kv_stem(hidden_states, cos, sin)
    entries, _, _ = self.compressor(hidden_states, real_len, fwp, state.prior_c, need_mask=False)
    keys, _, _ = self.indexer.compressor(hidden_states, real_len, fwp, state.prior_i, need_mask=False)
    ttnn.kv_cache.fill_cache_for_user_(state.compressed_kv, entries, 0, update_idx=state.entry_count)
    ttnn.kv_cache.fill_cache_for_user_(state.index_k, keys, 0, update_idx=state.entry_count)
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


V41CSA.kv_only = _kv_only
