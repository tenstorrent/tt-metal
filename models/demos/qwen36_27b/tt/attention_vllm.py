# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Gated GQA Attention for Qwen3.6-27B — vLLM + tensor-parallel (TP=8) path.

Adapted from the coder_next attention template + p300x2/tp_model.py's TP scheme.

Architecture (per HF config, attn_output_gate=True):
  q_proj outputs 2*head_dim per head -> split into query + sigmoid gate.
  Q/K have per-head RMSNorm. Partial RoPE over the first rotary_dim (25%).
  output = o_proj(attn_output * sigmoid(gate))

TP sharding (TP=8, nh=24, nkv=4, hd=256):
  * Fuse q|k|v into ONE column-parallel matmul Wqkv. Heads are contiguous so each
    chip gets [q_heads | kv_heads] for its slice. nhp = 24/8 = 3 query heads/chip.
  * nkv=4 < TP=8: replicate each KV head x2 so kv_slots = TP = 8, nkvp = 1 KV
    head/chip. The k/v projection weights are tiled x2 on the head axis BEFORE
    sharding so every chip holds exactly one (replicated) KV head — this keeps
    GQA correct (each query head still attends its own KV group's K/V).
  * o_proj is row(input)-parallel: weight [nh*hd, H] sharded on the input dim;
    each chip matmuls its [.., nhp*hd] attn output, all_reduce sums to full H.

Decode paths:
  * _decode_trace: contiguous fixed [1,nkvp,cache_len,hd] per-chip KV cache,
    paged_update_cache + scaled_dot_product_attention_decode (device cur_pos).
  * prefill: CPU fallback for the math, writes the prompt K/V into the contiguous
    cache via fill_cache at the request's row (batch_idx).
"""

import torch
import ttnn
from models.demos.qwen36_27b.tt.mesh_utils import to_torch as mesh_to_torch
from models.common.lightweightmodule import LightweightModule


class TtGatedAttention(LightweightModule):
    def __init__(self, device, state_dict, layer_idx, config, dtype=ttnn.bfloat16, weights_dtype=None):
        super().__init__()
        self.device = device
        self.config = config
        if weights_dtype is None:
            weights_dtype = config.get_dense_dtype(getattr(config, "weights_dtype", ttnn.bfloat8_b))
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.rotary_dim = config.rotary_dim
        self.hidden_size = config.hidden_size
        self.max_seq_len = config.max_seq_len
        self.ondevice_attn = getattr(config, "ondevice_attn", True)
        self.cache_len = min(getattr(config, "kv_cache_len", 2048), config.max_seq_len)

        self.dense_tp = getattr(config, "dense_tp", False)
        self.tp_size = getattr(config, "tp_size", 8)
        # Per-chip head counts. KV heads are replicated up to kv_slots = TP so each
        # chip holds exactly one KV head when nkv < TP.
        if self.dense_tp:
            assert self.num_heads % self.tp_size == 0, "nh must be divisible by TP"
            self.nhp = self.num_heads // self.tp_size  # query heads / chip
            self.kv_slots = max(self.num_kv_heads, self.tp_size)
            assert self.kv_slots % self.tp_size == 0
            self.nkvp = self.kv_slots // self.tp_size  # KV heads / chip (1 when nkv<=TP)
            self.kv_rep = self.kv_slots // self.num_kv_heads  # x2 for nkv=4, TP=8
        else:
            self.nhp = self.num_heads
            self.nkvp = self.num_kv_heads
            self.kv_rep = 1
            self.kv_slots = self.num_kv_heads

        self._kcache = None
        self._vcache = None
        self._cache_batch = None

        prefix = f"model.layers.{layer_idx}.self_attn"
        q_w = state_dict[f"{prefix}.q_proj.weight"].T.contiguous()  # [H, nh*hd*2]
        k_w = state_dict[f"{prefix}.k_proj.weight"].T.contiguous()  # [H, nkv*hd]
        v_w = state_dict[f"{prefix}.v_proj.weight"].T.contiguous()  # [H, nkv*hd]
        o_w = state_dict[f"{prefix}.o_proj.weight"].T.contiguous()  # [nh*hd, H]

        if self.dense_tp:
            # Replicate KV heads x2 (nkv=4 -> kv_slots=8) BEFORE sharding so each
            # chip gets one KV head. The GQA group order is preserved because each
            # KV head is duplicated in-place along the head axis. The query heads
            # are arranged so chip c sees query heads [c*nhp:(c+1)*nhp]; pairing
            # those with the replicated KV head means chip c attends KV head
            # (c*nhp)//num_kv_groups — see TODO in VLLM_PORT_NOTES (HW-verify GQA map).
            def rep_kv(w):
                if self.kv_rep == 1:
                    return w
                hd = self.head_dim
                wv = w.view(self.hidden_size, self.num_kv_heads, hd)
                wv = wv.repeat_interleave(self.kv_rep, dim=1)  # [H, kv_slots, hd]
                return wv.reshape(self.hidden_size, self.kv_slots * hd).contiguous()

            k_w = rep_kv(k_w)  # [H, kv_slots*hd]
            v_w = rep_kv(v_w)  # [H, kv_slots*hd]
            # Fuse q|k|v in PER-CHIP INTERLEAVED order: [q_c | k_c | v_c] for each
            # chip c, concatenated. A contiguous dim-3 shard then gives chip c
            # exactly its nhp query heads (incl gate) + its 1 replicated K/V head —
            # the layout EVERY consumer expects (_qkv_split, _prefill reassembly,
            # _decode_trace). A plain [all-q|all-k|all-v] cat would instead put
            # whole-q on chips 0..5 and all-K/all-V on chips 6..7 (broken).
            # KV replication [0,0,1,1,2,2,3,3] aligns chip c with KV group c//2,
            # which equals query-head (nhp*c)//num_kv_groups → GQA stays correct.
            hd = self.head_dim
            nhp, nkvp = self.nhp, self.nkvp
            qv = q_w.reshape(self.hidden_size, self.num_heads, hd * 2)
            kvw = k_w.reshape(self.hidden_size, self.kv_slots, hd)
            vvw = v_w.reshape(self.hidden_size, self.kv_slots, hd)
            blocks = []
            for c in range(self.tp_size):
                qb = qv[:, c * nhp:(c + 1) * nhp, :].reshape(self.hidden_size, nhp * hd * 2)
                kb = kvw[:, c * nkvp:(c + 1) * nkvp, :].reshape(self.hidden_size, nkvp * hd)
                vb = vvw[:, c * nkvp:(c + 1) * nkvp, :].reshape(self.hidden_size, nkvp * hd)
                blocks.append(torch.cat([qb, kb, vb], dim=1))
            qkv = torch.cat(blocks, dim=1)  # [H, tp_size * (nhp*hd*2 + 2*nkvp*hd)]
            self.qkv_proj_w = ttnn.from_torch(
                qkv.unsqueeze(0).unsqueeze(0),
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=3),
            )
            # o_proj row(input)-parallel: shard input dim (dim 2 of [1,1,nh*hd,H]).
            self.o_proj_w = ttnn.from_torch(
                o_w.unsqueeze(0).unsqueeze(0),
                dtype=weights_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ShardTensorToMesh(device, dim=2),
            )
        else:
            self.q_proj_w = ttnn.from_torch(
                q_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            self.k_proj_w = ttnn.from_torch(
                k_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            self.v_proj_w = ttnn.from_torch(
                v_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )
            self.o_proj_w = ttnn.from_torch(
                o_w.unsqueeze(0).unsqueeze(0), dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, device=device
            )

        self.q_norm_w_cpu = state_dict[f"{prefix}.q_norm.weight"][: self.head_dim].float()
        self.k_norm_w_cpu = state_dict[f"{prefix}.k_norm.weight"][: self.head_dim].float()

        TILE = 32
        q_norm_w = (self.q_norm_w_cpu + 1.0).unsqueeze(0).view(1, 1, self.head_dim).reshape(1, 1, self.head_dim // TILE, TILE)
        k_norm_w = (self.k_norm_w_cpu + 1.0).unsqueeze(0).view(1, 1, self.head_dim).reshape(1, 1, self.head_dim // TILE, TILE)
        self.q_norm_w_tt = ttnn.from_torch(q_norm_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        self.k_norm_w_tt = ttnn.from_torch(k_norm_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    def forward(self, hidden_states, cos, sin, kv_cache=None, mode="decode", position=0,
                page_table=None, cur_pos=None, batch_idx=0, cache_batch=None):
        if mode == "decode":
            if isinstance(position, ttnn.Tensor):  # trace: device position tensor
                return self._decode_trace(hidden_states, cos, sin, position)
            return self._decode_trace_host(hidden_states, cos, sin, position)
        return self._prefill(hidden_states, cos, sin, kv_cache, batch_idx=batch_idx, cache_batch=cache_batch)

    # ---- helpers ----
    def _ensure_cache(self, batch=1):
        """Fixed per-chip [batch, nkvp, cache_len, hd] KV cache (replicated across
        the mesh; each chip stores its own nkvp KV head[s])."""
        if self._kcache is not None and self._cache_batch == batch:
            return
        if self._kcache is not None:
            try:
                ttnn.deallocate(self._kcache)
                ttnn.deallocate(self._vcache)
            except Exception:
                pass
        self._cache_batch = batch
        rep = ttnn.ReplicateTensorToMesh(self.device)
        shp = [batch, self.nkvp, self.cache_len, self.head_dim]
        z = torch.zeros(shp, dtype=torch.bfloat16)
        self._kcache = ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        self._vcache = ttnn.from_torch(z.clone(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        try:
            grid = self.device.compute_with_storage_grid_size()
        except Exception:
            grid = ttnn.CoreCoord(8, 7)
        self._sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=128, max_cores_per_head_batch=16
        )

    def _rope_partial_lastdim(self, x, cos_tt, sin_tt):
        """Partial RoPE on the last dim; rotate first rotary_dim, pass-through rest.
        cos_tt/sin_tt: [1,1,1,rotary_dim] broadcast over leading dims."""
        d = self.rotary_dim
        nd = len(x.shape)

        def sl(t, lo, hi):
            start = [0] * nd
            end = list(t.shape)
            start[-1] = lo
            end[-1] = hi
            return ttnn.slice(t, start, end)

        x_rot = sl(x, 0, d)
        x_pass = sl(x, d, self.head_dim)
        half = d // 2
        x1 = sl(x_rot, 0, half)
        x2 = sl(x_rot, half, d)
        rot_half = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        out_rot = ttnn.add(ttnn.mul(x_rot, cos_tt), ttnn.mul(rot_half, sin_tt))
        return ttnn.concat([out_rot, x_pass], dim=-1)

    def _qkv_split(self, hidden_states, B):
        """Run the (fused, sharded) qkv linear and return per-chip query/gate/k/v
        reshaped to [1,B,nhp,hd] / [1,B,nkvp,hd]."""
        nhp, nkvp, hd = self.nhp, self.nkvp, self.head_dim
        qsz = nhp * hd * 2
        ksz = nkvp * hd
        qkv = ttnn.linear(hidden_states, self.qkv_proj_w, compute_kernel_config=self.config.matmul_kcfg())
        qp = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, B, qsz])
        kp = ttnn.slice(qkv, [0, 0, 0, qsz], [1, 1, B, qsz + ksz])
        vp = ttnn.slice(qkv, [0, 0, 0, qsz + ksz], [1, 1, B, qsz + 2 * ksz])
        q2 = ttnn.reshape(qp, [1, B, nhp, hd * 2])
        query = ttnn.reshape(ttnn.slice(q2, [0, 0, 0, 0], [1, B, nhp, hd]), [1, 1, B, nhp * hd])
        gate = ttnn.reshape(ttnn.slice(q2, [0, 0, 0, hd], [1, B, nhp, hd * 2]), [1, 1, B, nhp * hd])
        return query, gate, kp, vp

    def _decode_trace(self, hidden_states, cos, sin, position_tensor):
        """Trace-safe batched on-device decode. hidden_states [1,1,B,H], device cur_pos."""
        nhp, nkvp, hd = self.nhp, self.nkvp, self.head_dim
        B = hidden_states.shape[2]
        assert B <= 32, f"trace decode supports batch<=32, got {B}"
        self._ensure_cache(B)
        cos_tt = cos if not isinstance(cos, torch.Tensor) else ttnn.from_torch(
            cos.reshape(1, 1, 1, self.rotary_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        sin_tt = sin if not isinstance(sin, torch.Tensor) else ttnn.from_torch(
            sin.reshape(1, 1, 1, self.rotary_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        query, gate, kp, vp = self._qkv_split(hidden_states, B)

        # nlp_create_qkv_heads_decode shards batch one-per-core; emits height-sharded q/k/v.
        fused = ttnn.concat([query, kp, vp], dim=-1)  # [1,1,B,(nhp+2*nkvp)*hd]
        fused = ttnn.to_memory_config(fused, ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused, num_heads=nhp, num_kv_heads=nkvp, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
        kv_sharded_mem = k.memory_config()

        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.rms_norm(q, epsilon=1e-6, weight=self.q_norm_w_tt)
        k = ttnn.rms_norm(k, epsilon=1e-6, weight=self.k_norm_w_tt)
        q = self._rope_partial_lastdim(q, cos_tt, sin_tt)
        k = self._rope_partial_lastdim(k, cos_tt, sin_tt)

        k = ttnn.to_memory_config(k, kv_sharded_mem)
        v = ttnn.to_memory_config(v, kv_sharded_mem)
        ttnn.experimental.paged_update_cache(self._kcache, k, update_idxs_tensor=position_tensor)
        ttnn.experimental.paged_update_cache(self._vcache, v, update_idxs_tensor=position_tensor)

        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q, self._kcache, self._vcache, cur_pos_tensor=position_tensor,
            scale=self.scaling, program_config=self._sdpa_cfg)
        attn = ttnn.reshape(attn, [1, 1, B, nhp * hd])
        attn = ttnn.mul(attn, ttnn.sigmoid(gate))
        return self._o_proj(attn), None

    def _decode_trace_host(self, hidden_states, cos, sin, position):
        """Eager (non-trace) decode: build a device cur_pos tensor from the host
        position(s) and dispatch the same on-device path as _decode_trace."""
        B = hidden_states.shape[2]
        rep = ttnn.ReplicateTensorToMesh(self.device)
        if isinstance(position, int):
            pos = torch.full((B,), position, dtype=torch.int32)
        else:
            pos = torch.as_tensor(position, dtype=torch.int32).reshape(-1)
        cur_pos = ttnn.from_torch(pos, device=self.device, mesh_mapper=rep)
        return self._decode_trace(hidden_states, cos, sin, cur_pos)

    def _o_proj(self, attn):
        """o_proj: row(input)-parallel under TP (local matmul + all_reduce)."""
        out = ttnn.linear(attn, self.o_proj_w, compute_kernel_config=self.config.matmul_kcfg())
        if self.dense_tp:
            out = ttnn.all_reduce(out, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)
        return out

    def _prefill(self, hidden_states, cos, sin, kv_cache, batch_idx=0, cache_batch=None):
        """CPU-fallback prefill (S>1, B=1). Writes the prompt K/V into the
        contiguous per-chip cache at the request's row so the trace decode reads it.

        Under TP the qkv linear is sharded, so the per-chip output is read back and
        composed (ConcatMeshToTensor on the head/feature dim) to reconstruct the
        full q/k/v before the (host) attention math; o_proj is sharded so its input
        is sharded back across the mesh. This is the un-optimized correctness path.
        """
        B = 1
        S = hidden_states.shape[2]
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        cos_cpu = cos if isinstance(cos, torch.Tensor) else mesh_to_torch(cos)
        sin_cpu = sin if isinstance(sin, torch.Tensor) else mesh_to_torch(sin)

        if self.dense_tp:
            qkv = ttnn.linear(hidden_states, self.qkv_proj_w, compute_kernel_config=self.config.matmul_kcfg())
            # Compose the column-sharded qkv back to full width on the feature dim.
            qkv_cpu = mesh_to_torch(qkv, dim=3).reshape(B, S, -1)
            nhp, nkvp = self.nhp, self.nkvp
            # Per-chip layout is [q(nhp*hd*2) | k(nkvp*hd) | v(nkvp*hd)] repeated TP
            # times along the feature dim. Reassemble full q/k/v (KV heads de-replicated).
            per = nhp * hd * 2 + 2 * nkvp * hd
            qkv_chips = qkv_cpu.reshape(B, S, self.tp_size, per)
            q_list, k_list, v_list = [], [], []
            qsz = nhp * hd * 2
            ksz = nkvp * hd
            for c in range(self.tp_size):
                blk = qkv_chips[:, :, c, :]
                q_list.append(blk[..., :qsz])
                k_list.append(blk[..., qsz:qsz + ksz])
                v_list.append(blk[..., qsz + ksz:qsz + 2 * ksz])
            q_proj_cpu = torch.cat(q_list, dim=-1)  # [B,S,nh*hd*2]
            # KV heads were replicated x kv_rep across consecutive chips; take every
            # kv_rep-th chip's KV head to recover the nkv unique heads.
            k_full = torch.cat(k_list, dim=-1).reshape(B, S, self.kv_slots, hd)
            v_full = torch.cat(v_list, dim=-1).reshape(B, S, self.kv_slots, hd)
            k_proj_cpu = k_full[:, :, :: self.kv_rep, :].reshape(B, S, nkv * hd)
            v_proj_cpu = v_full[:, :, :: self.kv_rep, :].reshape(B, S, nkv * hd)
        else:
            q_proj_cpu = mesh_to_torch(ttnn.linear(hidden_states, self.q_proj_w)).reshape(B, S, -1)
            k_proj_cpu = mesh_to_torch(ttnn.linear(hidden_states, self.k_proj_w)).reshape(B, S, -1)
            v_proj_cpu = mesh_to_torch(ttnn.linear(hidden_states, self.v_proj_w)).reshape(B, S, -1)

        q_gate = q_proj_cpu.view(B, S, nh, hd * 2)
        query, gate = q_gate.chunk(2, dim=-1)
        gate = gate.reshape(B, S, -1)

        query = self._rms_norm(query, self.q_norm_w_cpu + 1.0).transpose(1, 2)
        key = self._rms_norm(k_proj_cpu.view(B, S, nkv, hd), self.k_norm_w_cpu + 1.0).transpose(1, 2)
        value = v_proj_cpu.view(B, S, nkv, hd).transpose(1, 2)

        query, key = self._apply_partial_rotary(query, key, cos_cpu, sin_cpu)

        # Write this request's prompt K/V into the contiguous per-chip cache. Under
        # TP each chip holds the KV head[s] for its slice; the cache is replicated,
        # so write the full nkvp-per-chip K/V (= the de-replicated nkv heads tiled
        # back to kv_slots, then sliced per chip is implicit via the replicate map).
        self._ensure_cache(cache_batch or 1)
        rep = ttnn.ReplicateTensorToMesh(self.device)
        if self.dense_tp:
            # Re-replicate KV heads to kv_slots so a per-chip [1,nkvp,S,hd] cache fill
            # places the right head on each chip. We tile then take the chip-c slice
            # by relying on ReplicateTensorToMesh + the fact that nkvp may be 1: build
            # a [1, kv_slots, S, hd] tensor and shard it across the mesh on the head dim.
            k_slots = key.repeat_interleave(self.kv_rep, dim=1)  # [1,kv_slots,S,hd]
            v_slots = value.repeat_interleave(self.kv_rep, dim=1)
            k_dev = ttnn.from_torch(
                k_slots.detach().to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=self.device, mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=1))
            v_dev = ttnn.from_torch(
                v_slots.detach().to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=self.device, mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=1))
        else:
            k_dev = ttnn.from_torch(key.detach().to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
            v_dev = ttnn.from_torch(value.detach().to(torch.bfloat16), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        ttnn.fill_cache(self._kcache, k_dev, batch_idx)
        ttnn.fill_cache(self._vcache, v_dev, batch_idx)
        ttnn.deallocate(k_dev)
        ttnn.deallocate(v_dev)

        # Host attention math over the full prompt (GQA-expanded).
        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)
        return self._prefill_attn_out(query, key, value, gate, S, B)

    def _prefill_attn_out(self, query, key_exp, value_exp, gate, S, B):
        attn_weights = torch.matmul(query.float(), key_exp.float().transpose(-1, -2)) * self.scaling
        causal_mask = torch.triu(
            torch.full((S, key_exp.shape[2]), float("-inf"), device=query.device),
            diagonal=key_exp.shape[2] - S + 1,
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value_exp.float()).to(query.dtype)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1)  # [B,S,nh*hd]
        attn_output = attn_output * torch.sigmoid(gate.float()).to(attn_output.dtype)

        nh, hd = self.num_heads, self.head_dim
        if self.dense_tp:
            # o_proj is row(input)-parallel: o_proj_w is sharded on its input dim
            # (nh*hd). Shard the full attn output [1,1,S,nh*hd] across the mesh on
            # the last dim so each chip holds the [.., nh*hd/TP] rows matching its
            # weight shard, matmul local -> partial [.., H], all_reduce sums.
            attn_tt = ttnn.from_torch(
                attn_output.reshape(1, 1, B * S, nh * hd).contiguous(),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=3))
            out = ttnn.linear(attn_tt, self.o_proj_w, compute_kernel_config=self.config.matmul_kcfg())
            out = ttnn.all_reduce(out, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear)
            return out, None

        attn_output_tt = ttnn.from_torch(
            attn_output.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
        )
        return ttnn.linear(attn_output_tt, self.o_proj_w), None

    @staticmethod
    def _rms_norm(x, weight, eps=1e-6):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return (weight * x).to(x.dtype)

    def _apply_partial_rotary(self, q, k, cos, sin):
        d = self.rotary_dim
        q_rot, q_pass = q[..., :d], q[..., d:]
        k_rot, k_pass = k[..., :d], k[..., d:]
        cos = cos[:, :, : q_rot.shape[2], :]
        sin = sin[:, :, : q_rot.shape[2], :]
        q_rot = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot = (k_rot * cos) + (self._rotate_half(k_rot) * sin)
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
