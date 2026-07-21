# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter prefill module (Kimi-K2.6-DFlash).

Front-loads the DFlash drafter's *context* KV cache during the verifier (DeepSeek/Kimi MLA)
prefill. Runs ONLY the drafter's KV-processing path — "MM, norm, ROPE, kv-update" — and skips
q_proj / SDPA / o_proj / MLP (attention + feedforward), which are decode-only.

The FC context projection is decomposed across target layers because
``Linear(concat[h_1..h_6]) == sum_i fc_slice_i @ h_i`` — so it accumulates as the verifier streams
its layers.

SHARDING (sequence-parallel):
  * hidden is TP-sharded on the verifier residual stream -> the FC tap is row-parallel + a TP reduce_scatter.
  * k_proj/v_proj are column-parallel: KV heads are split across the TP axis (kv_heads/tp per device) —
    num_kv_heads=8, head_dim=128.
  * the sequence is SP-sharded — each SP chip builds + holds ONLY its cache_seq/sp tokens (separate GQA
    K/V caches, SP-sharded on seq + TP-sharded on kv-head); the caller feeds SP-sharded seq (NO SP-gather).
    The drafter KV-build is token-parallel (no cross-seq op), so only the RoPE table is SP-sharded
    (absolute positions). Decode/migration-aligned, ~sp_factor× less work.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.dflash.dflash_drafter_config import (
    DFlashDrafterConfig,
    build_drafter_rope_hf_config,
)
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_cos_sin_matrix
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm


class TtDFlashDrafter:
    # safetensors key templates for the 20-tensor prefill subset.
    _K_PROJ = "layers.{i}.self_attn.k_proj.weight"
    _V_PROJ = "layers.{i}.self_attn.v_proj.weight"
    _K_NORM = "layers.{i}.self_attn.k_norm.weight"

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: DFlashDrafterConfig,
        state_dict: dict | None = None,
        *,
        sp_axis: int = 0,
        tp_axis: int = 1,
        max_seq_len: Optional[int] = None,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        fc_mode: str = "sliced",
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.tp_factor = mesh_device.shape[tp_axis]
        self.sp_factor = mesh_device.shape[sp_axis]
        self.num_links = num_links
        self.topology = topology
        # FC context projection mode:
        #   "sliced" — stream fc_slice_i @ h_i and accumulate at tap time
        #   "concat" — store the raw tapped hiddens, then at write time concat them and do one fc matmul
        #              (fc(concat) == Σ fc_slice_i @ h_i).
        assert fc_mode in ("sliced", "concat"), f"fc_mode must be 'sliced' or 'concat', got {fc_mode!r}"
        self.fc_mode = fc_mode
        # Prefill builds drafter KV for the FULL chunk the verifier hands it (e.g. 5120 tokens), so the
        # cache is sized to max_seq_len — NOT capped at 4k.
        self.cache_seq = max_seq_len if max_seq_len is not None else config.context_len

        assert (
            self.cache_seq % self.sp_factor == 0
        ), f"seq-parallel needs cache_seq {self.cache_seq} divisible by sp {self.sp_factor}"

        assert (
            config.num_key_value_heads % self.tp_factor == 0
        ), f"num_kv_heads {config.num_key_value_heads} must be divisible by tp {self.tp_factor}"
        self.kv_heads_local = config.num_key_value_heads // self.tp_factor

        self.default_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.hifi4_fp32_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self._load_weights(state_dict)
        self._rope_cos = self._rope_sin = None
        self._rope_end = 0
        self._ensure_rope(self.cache_seq)
        self._alloc_caches()
        self._reduced_accum: Optional[ttnn.Tensor] = None  # "sliced": running TP-partial FC sum
        self._taps: list = [None] * len(config.target_layer_ids)  # "concat": stored raw taps

    # ------------------------------------------------------------------ setup
    def _mesh_mappers(self):
        """Row-parallel (shard tensor dim 0 on TP) and column-parallel (shard tensor dim 1 on TP)
        2D-weight mappers, replicating on the SP axis.
        """
        row = [None, None]
        row[self.tp_axis] = 0  # shard the contraction (input) dim across TP
        col = [None, None]
        col[self.tp_axis] = 1  # shard the output dim across TP
        col[self.sp_axis] = None
        mapper_row = ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=row)
        mapper_col = ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=col)
        return mapper_row, mapper_col

    def _load_weights(self, state_dict: dict | None):
        cfg = self.config
        H, kv_dim, D = cfg.hidden_size, cfg.kv_dim, cfg.head_dim
        mapper_row, mapper_col = self._mesh_mappers()
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        have = state_dict is not None and "fc.weight" in state_dict

        def _linear_w(torch_w, mapper):
            # torch_w is the HF Linear weight [out, in]; ttnn.linear wants [in, out].
            t = torch_w.transpose(-2, -1).contiguous() if torch_w is not None else None
            return ttnn.as_tensor(
                t,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,  # TODO: bfloat8_b for perf
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        def _norm_w(torch_w):
            # RMSNorm weight [dim] -> [1, 1, dim/32, 32] ROW_MAJOR bf16, replicated (matches ttMLA).
            t = torch_w.reshape(1, 1, -1, ttnn.TILE_SIZE) if torch_w is not None else None
            return ttnn.as_tensor(
                t,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        # FC context projection. Two layouts (see fc_mode):
        #   "sliced": n [H(out), H(in)] blocks, each row-parallel (input H sharded on TP), accumulated at
        #             tap time. fc(concat) == Σ_i fc_slice_i @ h_i.
        #   "concat": the FULL fc [H, n*H], pre-PERMUTED on the input (n*H) axis so a contiguous
        #             row-parallel TP shard lines up with the on-device grouped concat of the taps. On
        #             device tap i is [.., H/tp] (its hidden TP-shard); concatenating the n taps on the
        #             feature dim yields, per device, [h_0[shard_d]|...|h_{n-1}[shard_d]] — a GROUPED (not
        #             contiguous) slice of the n*H input. Permuting fc's columns into shard-major-then-layer
        #             order makes the standard contiguous row-parallel shard match.
        n = len(cfg.target_layer_ids)
        self.fc_slices = None
        self.fc_perm = None
        if self.fc_mode == "sliced":
            self.fc_slices = []
            fc_full = state_dict["fc.weight"] if have else None  # [H, n*H]
            for idx in range(n):
                sl = fc_full[:, idx * H : (idx + 1) * H] if have else None  # [H(out), H(in)]
                self.fc_slices.append(_linear_w(sl, mapper_row))
        else:  # "concat"
            assert H % self.tp_factor == 0, f"hidden {H} must be divisible by tp {self.tp_factor} for concat fc"
            fc_perm_w = None
            if have:
                W = state_dict["fc.weight"]  # [H(out), n*H(in)]; in = concat over target layers of H
                hs = H // self.tp_factor
                # Column order so a contiguous TP block d == {layer*H + [d*hs:(d+1)*hs] for all layers}.
                perm = [
                    layer * H + col
                    for d in range(self.tp_factor)
                    for layer in range(n)
                    for col in range(d * hs, (d + 1) * hs)
                ]
                fc_perm_w = W[:, perm].contiguous()  # [H(out), n*H(in)] permuted
            self.fc_perm = _linear_w(fc_perm_w, mapper_row)

        # hidden_norm spans the full H=7168 → it MUST be the DISTRIBUTED (TP-sharded) norm, exactly
        # like the model's attn_norm/ffn_norm. A plain ttnn.rms_norm over the replicated 7168 forces
        # one core to hold 7168-wide (224-tile) CBs and overflows L1. cluster_axis=tp_axis matches the
        # H/tp shard it consumes (see write_kv_cache: reduce_scatter -> norm -> all_gather).
        self.hidden_norm = TtDistributedRmsNorm(
            mesh_device=self.mesh_device,
            emb_dim=cfg.hidden_size,
            epsilon=cfg.rms_norm_eps,
            torch_weight=state_dict["hidden_norm.weight"] if have else None,
            cluster_axis=self.tp_axis,
            num_links=self.num_links,
            topology=self.topology,
        )

        # Per draft layer: k/v proj column-parallel (KV heads split across TP), per-head k_norm replicated.
        self.k_proj, self.v_proj, self.k_norm = [], [], []
        for i in range(cfg.num_hidden_layers):
            kw = state_dict[self._K_PROJ.format(i=i)] if have else None  # [kv_dim, H]
            vw = state_dict[self._V_PROJ.format(i=i)] if have else None
            kn = state_dict[self._K_NORM.format(i=i)] if have else None  # [head_dim]
            self.k_proj.append(_linear_w(kw, mapper_col))
            self.v_proj.append(_linear_w(vw, mapper_col))
            self.k_norm.append(_norm_w(kn))
        assert kv_dim == cfg.num_key_value_heads * D

    def _ensure_rope(self, end: int) -> None:
        """Build (memoized) drafter deepseek_yarn cos/sin covering positions [0, end). Called ONCE from
        __init__ with end=cache_seq so it stays OUT of the write_kv_cache hot path; memoized, so it does
        NOT rebuild per call (would only rebuild if a longer range were later requested). yarn inv_freq
        is position-independent, so growing the table never changes the
        values at existing positions. HALF-SPLIT (interleave=False) to match Qwen3 rotate_half +
        ttnn.experimental.rotary_embedding_hf; full head_dim (128) rotated (unlike the MLA 64-dim pe).
        ``end`` is the GLOBAL sequence length and the table is SP-sharded on seq, so each SP chip gets
        exactly its absolute-position window [r*end/sp, (r+1)*end/sp) — no on-device offset/slice."""
        if self._rope_cos is not None and self._rope_end >= end:
            return
        hf = build_drafter_rope_hf_config(self.config, max_seq_len=end)
        cos, sin = get_cos_sin_matrix(hf, interleave=False)  # [1, 1, end, head_dim]
        cos, sin = cos[..., :end, :], sin[..., :end, :]
        shard = [None, None]
        shard[self.sp_axis] = 2  # SP-shard the global table on seq → per-chip absolute-position window
        mapper = ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=shard)
        if self._rope_cos is not None:
            ttnn.deallocate(self._rope_cos)
            ttnn.deallocate(self._rope_sin)
        self._rope_cos = ttnn.from_torch(
            cos, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
        )
        self._rope_sin = ttnn.from_torch(
            sin, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
        )
        self._rope_end = end

    def _alloc_caches(self):
        """Separate K and V drafter caches: host [num_layers, num_kv_heads, cache_seq, head_dim],
        TP-sharded on kv-head (dim 1) and SP-sharded on seq (dim 2) so each SP chip owns cache_seq/sp
        tokens (decode/migration layout, no redundant per-SP copies).
        """
        cfg = self.config
        shape = (cfg.num_hidden_layers, cfg.num_key_value_heads, self.cache_seq, cfg.head_dim)
        shard = [None, None]
        shard[self.tp_axis] = 1  # kv-head dim across TP
        shard[self.sp_axis] = 2  # seq dim across SP → per-chip [.., cache_seq/sp, ..]
        mapper = ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=shard)
        zeros = torch.zeros(*shape, dtype=torch.bfloat16)
        self.k_cache = ttnn.from_torch(
            zeros,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self.v_cache = ttnn.from_torch(
            zeros.clone(),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def reset(self):
        """Clear the FC accumulator (sliced) / stored taps (concat) — call at the start of each prefill
        sequence/chunk."""
        if self._reduced_accum is not None:
            ttnn.deallocate(self._reduced_accum)
        self._reduced_accum = None
        for i, t in enumerate(self._taps):
            if t is not None:
                ttnn.deallocate(t)
                self._taps[i] = None

    def is_target_layer(self, global_layer_idx: int) -> bool:
        return global_layer_idx in self.config.target_layer_ids

    def tap(self, hidden_states: ttnn.Tensor, global_layer_idx: int) -> None:
        """FC context tap at a verifier target layer. ``hidden_states`` is the residual-stream output
        [1, 1, seq, hidden/tp], TP-sharded on hidden and SP-sharded on seq (each chip taps only its own
        slice — NO SP-gather).

        fc_mode="sliced": stream the FC-slice matmul and accumulate the (still TP-partial) sum; the TP
            cross-TP combine is deferred to write_kv_cache's reduce_scatter (local sum-then-scatter == scatter-then-sum).
        fc_mode="concat": store the raw tap for a single fc(concat) at write time."""
        if not self.is_target_layer(global_layer_idx):
            return
        idx = self.config.target_layer_ids.index(global_layer_idx)
        if self.fc_mode == "concat":
            # The drafter TAKES OWNERSHIP of the tensor (deallocated in write_kv_cache()/reset()); the
            # caller must not free it. The integration hook hands over a fresh (SP-gathered) tensor, so no
            # clone is needed. TODO: if a caller passes a tensor it will mutate/reuse (e.g. the
            # raw loop `h`), clone before storing.
            if self._taps[idx] is not None:
                ttnn.deallocate(self._taps[idx])
            self._taps[idx] = hidden_states
            return
        partial = ttnn.linear(
            hidden_states,
            self.fc_slices[idx],
            compute_kernel_config=self.default_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # TODO: add a tuned program_config.
        )
        if self._reduced_accum is None:
            self._reduced_accum = partial
        else:
            summed = ttnn.add(self._reduced_accum, partial)
            ttnn.deallocate(self._reduced_accum)
            ttnn.deallocate(partial)
            self._reduced_accum = summed

    def _split_heads(self, proj: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, seq, kv_heads_local*head_dim] -> [1, kv_heads_local, seq, head_dim].

        num_kv_heads=0 selects the single-tensor "Q-path" reshape (take the first output);
        transpose_k_heads=False keeps [.., heads, seq, head_dim] (NOT the QKᵀ transpose).
        head_dim + seq are inferred from the tensor width/shape.
        """
        heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            proj,
            num_heads=self.kv_heads_local,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return heads

    def write_kv_cache(self, positions_start: int = 0) -> None:
        """Finalize: build the TP-partial FC output (mode-dependent), TP-reduce it, hidden_norm, then per
        draft layer project/norm/rope K and project V, writing each into its cache slot. ``positions_start``
        offsets rope for the last-4k window (Phase 3); Phase 1 uses 0. Caller must have supplied
        seq-contiguous taps."""
        cfg = self.config
        if self.fc_mode == "concat":
            missing = [cfg.target_layer_ids[i] for i, t in enumerate(self._taps) if t is None]
            assert not missing, f"write_kv_cache: missing taps for target layers {missing}"
            # ONE fc over the concatenated taps. Per device each tap is [1,1,seq,H/tp] (its hidden
            # TP-shard); concat on the feature dim gives [1,1,seq,n*H/tp] laid out [h_0[shard_d]|...] per
            # device. fc_perm was pre-permuted so a contiguous row-parallel shard matches → [1,1,seq,H]
            # TP-partial (== Σ fc_slice_i @ h_i).
            cat = ttnn.concat(list(self._taps), dim=3)
            for i, t in enumerate(self._taps):
                ttnn.deallocate(t)
                self._taps[i] = None
            reduced_partial = ttnn.linear(
                cat,
                self.fc_perm,
                compute_kernel_config=self.default_compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(cat)
        else:  # "sliced"
            assert self._reduced_accum is not None, "write_kv_cache called before any tap()"
            reduced_partial = self._reduced_accum
            self._reduced_accum = None
        # Combine the row-parallel FC partials across TP: reduce_scatter SUMS them AND scatters on hidden
        # in ONE op → exactly the [1,1,seq,H/tp] shard the distributed hidden_norm wants (no
        # all_gather→replicate→mesh_partition round-trip). Matches MLA o_proj/q_a_proj + MoE tt_reduce.
        if self.tp_factor > 1:
            reduced = ttnn.reduce_scatter(
                reduced_partial,
                dim=-1,
                cluster_axis=self.tp_axis,
                num_links=self.num_links,
                topology=self.topology,
            )  # [1,1,seq,H/tp] — summed FC output, TP-sharded on hidden
            ttnn.deallocate(reduced_partial)
        else:
            reduced = reduced_partial  # tp==1: the partial IS the full sum; hidden_norm handles full H
        seq = reduced.shape[2]  # PER-CHIP seq (dim2, unchanged by the hidden scatter)
        per_chip_cap = self.cache_seq // self.sp_factor
        assert positions_start + seq <= per_chip_cap, (
            f"positions_start+seq ({positions_start + seq}) exceeds per-chip cache depth ({per_chip_cap}); "
            f"construct with max_seq_len >= the chunk length (the cache is sized to the full chunk; "
            f"the 4k window is applied at migration, not here)"
        )

        # Distributed hidden_norm on the [1,1,seq,H/tp] shard (stats all-gathered internally → correct
        # full-H norm), then all-gather back to replicated so the column-parallel k/v_proj sees full H.
        target_hidden = self.hidden_norm(reduced)  # [1,1,seq,H/tp] (or full H when tp==1)
        ttnn.deallocate(reduced)
        if self.tp_factor > 1:
            gathered = ttnn.all_gather(
                target_hidden, dim=3, cluster_axis=self.tp_axis, num_links=self.num_links, topology=self.topology
            )
            ttnn.deallocate(target_hidden)
            target_hidden = gathered  # [1,1,seq,H] replicated on TP

        # RoPE cos/sin: the __init__-built table is SP-sharded over the GLOBAL seq, so this chip already
        # holds its absolute-position window [r*cache_seq/sp, …); use it directly — no per-call build/slice.
        # TODO: chunked prefill at arbitrary offsets (positions_start>0) needs an indexed rope op
        # that takes the offset on-device (like MLA's rotary_embedding_indexed).
        assert positions_start == 0, "chunked offset not yet supported"
        cos, sin = self._rope_cos, self._rope_sin

        for i in range(cfg.num_hidden_layers):
            k = ttnn.linear(
                target_hidden,
                self.k_proj[i],
                compute_kernel_config=self.default_compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            v = ttnn.linear(
                target_hidden,
                self.v_proj[i],
                compute_kernel_config=self.default_compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            k = self._split_heads(k)  # [1, kvh_local, seq, head_dim]
            v = self._split_heads(v)
            # per-head RMSNorm over head_dim, then Qwen3 half-split rope. V untouched.
            k = ttnn.rms_norm(
                k,
                weight=self.k_norm[i],
                epsilon=cfg.rms_norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.default_compute_kernel_config,
            )
            k = ttnn.experimental.rotary_embedding_hf(
                k, cos, sin, is_decode_mode=False, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
            )
            # Write into cache slot `i` (layer as the fill "user" dim). TODO: the migration
            # writer + SP-sharded seq + bf8 cache replace this seq-replicated fill_cache_for_user_.
            ttnn.kv_cache.fill_cache_for_user_(self.k_cache, k, i)
            ttnn.kv_cache.fill_cache_for_user_(self.v_cache, v, i)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        ttnn.deallocate(target_hidden)
