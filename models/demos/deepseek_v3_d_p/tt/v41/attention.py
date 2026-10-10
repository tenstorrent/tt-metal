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
        keys = m._sp_all_gather(ttnn.concat([nope, rope], dim=-1))
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
        return cls(mesh_device, compressor=comp, indexer=indexer, sparse_path=False, **kw)
