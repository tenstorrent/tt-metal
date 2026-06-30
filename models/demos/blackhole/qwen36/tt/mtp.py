# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device MTP (Multi-Token Prediction) drafter for Qwen3.6-27B.

Implements the K-step draft chain as sequential TTNN op dispatches with one
sync per step (unavoidable: need token ID to feed into next step's embedding).

Architecture (same as backbone full-attention layers, but one layer):
  pre_fc_norm_hidden  [dim]      — RMSNorm on backbone hidden h
  pre_fc_norm_embedding [dim]    — RMSNorm on token embedding e
  fc [dim, 2*dim]                — linear: cat(normed_h, normed_e) → h_mtp_in [dim]
  layers.0                       — gated attention + SwiGLU MLP (same as backbone)
    input_layernorm              — pre-attention RMSNorm
    self_attn                    — gated attention: q_proj 2x wide (Q + sigmoid gate)
                                   QK-norm, partial RoPE (rope_dim = head_dim * partial_rotary_factor)
    post_attention_layernorm     — pre-MLP RMSNorm
    mlp                          — SwiGLU (gate_proj, up_proj, down_proj)
  norm [dim]                     — final RMSNorm before lm_head
  lm_head [vocab, dim]           — shared with backbone (passed in, already on device)

TP strategy: ALL MTP weights are REPLICATED on every device.
  - MTP is a single layer (~500M params) vs 27B backbone — trivial DRAM cost.
  - Input h_sharded [1,1,1,dim_frac] -> all_gather -> [1,1,1,dim] once per chain.
  - K steps run identically on all devices; draft token taken from device 0.

KV cache: per-head concat tensors (max_seq_len entries). Filled from position 0
  at init time with zeros; updated per-step via ttnn.update_cache.
"""

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, rot_mats_decode


def _rep(mesh):
    return ttnn.ReplicateTensorToMesh(mesh)


def _load_replicated(t: torch.Tensor, mesh, dtype=ttnn.bfloat16, cache_path=None):
    return ttnn.as_tensor(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_rep(mesh),
        cache_file_name=str(cache_path) if cache_path else None,
    )


class Qwen36MTPTTModule:
    """On-device MTP drafter. Runs K draft steps as sequential TTNN dispatches.

    One sync per step is unavoidable: we need the draft token ID to embed it in
    the next step. All ops in a step are dispatched without CPU sync.

    Args:
        mesh_device: ttnn mesh.
        args: Qwen36ModelArgs (provides n_heads, n_kv_heads, head_dim, rope_head_dim, etc.).
        state_dict: full backbone-mapped state dict containing mtp.* keys.
        lm_head_weight: backbone's lm_head ttnn.Tensor (shared, already on device,
                        layout [vocab_size, dim] after transpose, DRAM replicated).
        embed_module: callable (tok_ids_tt) -> ttnn.Tensor [1,1,1,dim].
                      Must produce a replicated (not TP-sharded) embedding.
        tensor_cache_path: optional weight cache directory (pathlib.Path or None).
        max_seq_len: maximum sequence length for KV cache allocation.
    """

    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        lm_head_weight,
        embed_module,
        tensor_cache_path=None,
        max_seq_len: int = None,
    ):
        self.mesh = mesh_device
        self.args = args
        self.num_devices = mesh_device.get_num_devices()
        self.lm_head_weight = lm_head_weight
        self.embed = embed_module
        self.max_seq_len = max_seq_len or args.max_seq_len

        cache = (lambda n: tensor_cache_path / f"mtp.{n}") if tensor_cache_path else (lambda n: None)

        self._ckc = tpc.COMPUTE_HIFI2

        def _load_norm(key, name):
            w = state_dict[key].to(torch.float32) + 1.0
            return _load_replicated(w, mesh_device, dtype=ttnn.bfloat16, cache_path=cache(name))

        def _load_norm_flat(key, name):
            w = state_dict[key].to(torch.float32)
            return _load_replicated(w, mesh_device, dtype=ttnn.bfloat16, cache_path=cache(name + "_flat"))

        def _load_w(key, name, dtype=ttnn.bfloat8_b):
            t = state_dict[key].T.contiguous().to(torch.float32)
            return _load_replicated(t, mesh_device, dtype=dtype, cache_path=cache(name))

        # Pre-FC norms
        self.pre_fc_norm_h = _load_norm("mtp.pre_fc_norm_hidden.weight", "pre_fc_norm_hidden")
        self.pre_fc_norm_e = _load_norm("mtp.pre_fc_norm_embedding.weight", "pre_fc_norm_embedding")

        # FC weight: checkpoint stores [dim, 2*dim], transpose -> [2*dim, dim]
        self.fc_w = _load_w("mtp.fc.weight", "fc")

        # Transformer layer norms
        self.input_norm_w = _load_norm("mtp.layers.0.input_layernorm.weight", "input_norm")
        self.post_attn_norm_w = _load_norm("mtp.layers.0.post_attention_layernorm.weight", "post_attn_norm")

        # Attention weights. q_proj is 2x wide: [dim, 2*NH*HD] -> .T = [2*NH*HD, dim]
        self.wqg = _load_w("mtp.layers.0.self_attn.q_proj.weight", "attn.wqg")
        self.wk = _load_w("mtp.layers.0.self_attn.k_proj.weight", "attn.wk")
        self.wv = _load_w("mtp.layers.0.self_attn.v_proj.weight", "attn.wv")
        self.wo = _load_w("mtp.layers.0.self_attn.o_proj.weight", "attn.wo")
        # Flat QK norms for decode (no +1, matching TPAttention.forward_decode hybrid)
        self.q_norm_flat = _load_norm_flat("mtp.layers.0.self_attn.q_norm.weight", "attn.q_norm")
        self.k_norm_flat = _load_norm_flat("mtp.layers.0.self_attn.k_norm.weight", "attn.k_norm")

        # MLP
        self.mlp_gate_w = _load_w("mtp.layers.0.mlp.gate_proj.weight", "mlp.gate")
        self.mlp_up_w = _load_w("mtp.layers.0.mlp.up_proj.weight", "mlp.up")
        self.mlp_down_w = _load_w("mtp.layers.0.mlp.down_proj.weight", "mlp.down")

        # Final norm
        self.final_norm_w = _load_norm("mtp.norm.weight", "final_norm")

        # Attention shape params: detect from MTP weight shapes (MTP has NH=24, not backbone's 48)
        HD = args.head_dim
        q_shape = state_dict["mtp.layers.0.self_attn.q_proj.weight"].shape  # [2*NH*HD, dim]
        k_shape = state_dict["mtp.layers.0.self_attn.k_proj.weight"].shape  # [NKV*HD, dim]
        self.NH = q_shape[0] // (2 * HD)  # 24 for the 27B model
        self.NKV = k_shape[0] // HD
        self.HD = HD
        self.rope_dim = args.rope_head_dim
        self.scale = self.HD**-0.5

        # Per-head KV caches, allocated lazily on first chain() call
        self.k_caches = None
        self.v_caches = None

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def _ensure_kv_cache(self):
        if self.k_caches is not None:
            return
        zeros = torch.zeros(1, 1, self.max_seq_len, self.HD, dtype=torch.bfloat16)
        self.k_caches = [
            ttnn.from_torch(
                zeros,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_rep(self.mesh),
            )
            for _ in range(self.NKV)
        ]
        self.v_caches = [
            ttnn.from_torch(
                zeros,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_rep(self.mesh),
            )
            for _ in range(self.NKV)
        ]

    def reset_kv_cache(self):
        """Drop the MTP KV cache (call before each new prompt to free + reallocate)."""
        self.k_caches = None
        self.v_caches = None

    # ------------------------------------------------------------------
    # RMSNorm with pre-offset weight: y = (x / rms(x)) * w
    # ------------------------------------------------------------------

    def _norm(self, x, w):
        n = ttnn.rms_norm(x, epsilon=self.args.norm_eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.multiply(n, w, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ------------------------------------------------------------------
    # Single MTP decode step
    # ------------------------------------------------------------------

    def _mtp_step(self, h: ttnn.Tensor, tok_tt: ttnn.Tensor, pos: int):
        """Dispatch one MTP decode step. Returns (h_out, logits) both on device.

        h:      [1,1,1,dim] replicated.
        tok_tt: [1,1,1,1] uint32 token id, replicated.
        pos:    KV cache position to write at.
        """
        mc = ttnn.DRAM_MEMORY_CONFIG
        ckc = self._ckc
        NH, NKV, HD = self.NH, self.NKV, self.HD
        B = 1

        # Embed + pre-FC
        e = self.embed(tok_tt)
        normed_h = self._norm(h, self.pre_fc_norm_h)
        normed_e = self._norm(e, self.pre_fc_norm_e)
        ttnn.deallocate(e)
        combined = ttnn.concat([normed_h, normed_e], dim=-1, memory_config=mc)
        ttnn.deallocate(normed_h)
        ttnn.deallocate(normed_e)
        h_mtp = ttnn.linear(combined, self.fc_w, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(combined)

        # Gated attention
        h_norm = self._norm(h_mtp, self.input_norm_w)
        qg = ttnn.linear(h_norm, self.wqg, compute_kernel_config=ckc, memory_config=mc)
        kp = ttnn.linear(h_norm, self.wk, compute_kernel_config=ckc, memory_config=mc)
        vp = ttnn.linear(h_norm, self.wv, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(h_norm)

        qg = ttnn.reshape(qg, (1, B, NH, HD * 2))
        q = ttnn.slice(qg, (0, 0, 0, 0), (1, B, NH, HD))
        gate = ttnn.slice(qg, (0, 0, 0, HD), (1, B, NH, HD * 2))
        ttnn.deallocate(qg)
        k = ttnn.reshape(kp, (1, B, NKV, HD))
        ttnn.deallocate(kp)
        v = ttnn.reshape(vp, (1, B, NKV, HD))
        ttnn.deallocate(vp)

        # Flat QK norms (no +1) for decode
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=self.args.norm_eps), self.q_norm_flat, memory_config=mc)
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=self.args.norm_eps), self.k_norm_flat, memory_config=mc)

        # Partial RoPE
        cos_tt, sin_tt = rot_mats_decode(
            self.mesh,
            self.rope_dim,
            self.max_seq_len,
            self.args.rope_theta,
            torch.tensor([pos], dtype=torch.int32),
        )
        q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, self.rope_dim)
        k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, self.rope_dim)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        # Write K/V into per-head caches at position pos.
        # paged_update_cache requires padding to tile size 32 and a sharded memory config.
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=mc,
            mesh_mapper=_rep(self.mesh),
        )
        for h_idx in range(NKV):
            k_slice = ttnn.slice(k, (0, 0, h_idx, 0), (1, B, h_idx + 1, HD))
            v_slice = ttnn.slice(v, (0, 0, h_idx, 0), (1, B, h_idx + 1, HD))
            k_pad = ttnn.pad(k_slice, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
            v_pad = ttnn.pad(v_slice, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
            ttnn.deallocate(k_slice)
            ttnn.deallocate(v_slice)
            k_sh = ttnn.to_memory_config(k_pad, self.args.kv_update_shard_cfg)
            v_sh = ttnn.to_memory_config(v_pad, self.args.kv_update_shard_cfg)
            ttnn.deallocate(k_pad)
            ttnn.deallocate(v_pad)
            ttnn.experimental.paged_update_cache(self.k_caches[h_idx], k_sh, update_idxs_tensor=cur_pos_tt)
            ttnn.experimental.paged_update_cache(self.v_caches[h_idx], v_sh, update_idxs_tensor=cur_pos_tt)
            ttnn.deallocate(k_sh)
            ttnn.deallocate(v_sh)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # SDPA decode: q [1,B,NH,HD]; k/v caches [1,1,max_seq,HD] per head.
        # scaled_dot_product_attention_decode takes q [1,B,NH,HD] directly.
        if NKV == 1:
            k_full = self.k_caches[0]
            v_full = self.v_caches[0]
        else:
            k_full = ttnn.concat(self.k_caches, dim=1, memory_config=mc)
            v_full = ttnn.concat(self.v_caches, dim=1, memory_config=mc)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(4, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )
        attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k_full,
            v_full,
            cur_pos_tensor=cur_pos_tt,
            scale=self.scale,
            program_config=sdpa_cfg,
            memory_config=mc,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(cur_pos_tt)
        if NKV > 1:
            ttnn.deallocate(k_full)
            ttnn.deallocate(v_full)

        # attn_out [1,B,NH,HD] -> gate [1,B,NH,HD] -> gated -> reshape -> o_proj
        gated = ttnn.multiply(attn_out, ttnn.sigmoid(gate), memory_config=mc)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(gate)
        gated = ttnn.reshape(gated, (1, B, NH * HD))
        attn_proj = ttnn.linear(gated, self.wo, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(gated)
        attn_proj = ttnn.reshape(attn_proj, (1, 1, 1, attn_proj.shape[-1]))

        h_out = ttnn.add(h_mtp, attn_proj, memory_config=mc)
        ttnn.deallocate(attn_proj)

        # SwiGLU MLP
        h_norm2 = self._norm(h_out, self.post_attn_norm_w)
        g = ttnn.linear(h_norm2, self.mlp_gate_w, compute_kernel_config=ckc, memory_config=mc)
        u = ttnn.linear(h_norm2, self.mlp_up_w, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(h_norm2)
        hidden = ttnn.multiply(ttnn.silu(g, memory_config=mc), u, memory_config=mc)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        mlp_out = ttnn.linear(hidden, self.mlp_down_w, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(hidden)
        h_out = ttnn.add(h_out, mlp_out, memory_config=mc)
        ttnn.deallocate(mlp_out)

        # Final norm + lm_head
        h_normed = self._norm(h_out, self.final_norm_w)
        logits = ttnn.linear(h_normed, self.lm_head_weight, compute_kernel_config=ckc, memory_config=mc)
        ttnn.deallocate(h_normed)

        return h_out, logits

    # ------------------------------------------------------------------
    # KV warmup: populate MTP KV cache for all prompt positions
    # ------------------------------------------------------------------

    def prefill(self, prompt_token_ids: list, pos_start: int = 0):
        """Populate MTP KV cache for positions [pos_start .. pos_start+T-1].

        Uses zeros as the backbone hidden approximation (we don't store backbone hiddens
        for all T prompt positions). This warms up attention context so draft quality at
        spec-decode time is far better than a cold-start cache.

        prompt_token_ids: CPU list[int] — prompt token IDs (length T).
        pos_start: absolute position of the first token (usually 0).
        """
        self._ensure_kv_cache()
        mc = ttnn.DRAM_MEMORY_CONFIG
        dim = self.args.dim
        h = ttnn.from_torch(
            torch.zeros(1, 1, 1, dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=mc,
            mesh_mapper=_rep(self.mesh),
        )
        for i, tok in enumerate(prompt_token_ids):
            tok_tt = ttnn.from_torch(
                torch.tensor([[[[tok]]]], dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=mc,
                mesh_mapper=_rep(self.mesh),
            )
            h_next, logits_k = self._mtp_step(h, tok_tt, pos_start + i)
            ttnn.deallocate(tok_tt)
            ttnn.deallocate(logits_k)
            ttnn.deallocate(h)
            h = h_next
        ttnn.deallocate(h)

    # ------------------------------------------------------------------
    # Public chain API
    # ------------------------------------------------------------------

    def chain(
        self,
        h_sharded: ttnn.Tensor,
        last_tok: int,
        K: int,
        pos_start: int,
    ) -> tuple:
        """Generate K draft tokens on-device.

        One sync per step (unavoidable: need token ID to feed next step's embed).

        h_sharded: [1,1,1,dim_frac] TP-sharded backbone hidden from prefill_tp.
        last_tok:  CPU int — the last accepted token (input to step 0).
        K:         number of draft tokens to generate.
        pos_start: KV cache position for step 0.

        Returns:
            (draft_token_ids: List[int], h_last: ttnn.Tensor [1,1,1,dim] on device)
        """
        self._ensure_kv_cache()
        mc = ttnn.DRAM_MEMORY_CONFIG

        # All-gather TP-sharded hidden -> full [1,1,1,dim] replicated
        if self.num_devices > 1:
            h_full = ttnn.all_gather(
                h_sharded,
                dim=3,
                num_links=1,
                cluster_axis=1,
                memory_config=mc,
            )
        else:
            h_full = ttnn.to_memory_config(h_sharded, mc)

        draft_tokens = []
        h = h_full
        tok = last_tok

        for k in range(K):
            tok_tt = ttnn.from_torch(
                torch.tensor([[[[tok]]]], dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=mc,
                mesh_mapper=_rep(self.mesh),
            )
            h_next, logits_k = self._mtp_step(h, tok_tt, pos_start + k)
            ttnn.deallocate(tok_tt)
            if k > 0:
                ttnn.deallocate(h)
            h = h_next

            # Sync to read the draft token for the next embedding.
            ttnn.synchronize_device(self.mesh)
            lt = ttnn.to_torch(logits_k, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))
            tok = int(lt[0].reshape(-1)[: self.args.vocab_size].argmax().item())
            ttnn.deallocate(logits_k)
            draft_tokens.append(tok)

        return draft_tokens, h  # h: [1,1,1,dim] replicated, on device
