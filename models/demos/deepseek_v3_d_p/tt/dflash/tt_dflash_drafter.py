# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter *context-KV* prefill module (Kimi-K2.6-DFlash) — issue #49586, Phase 1.

Front-loads the DFlash drafter's *context* KV cache during the verifier (DeepSeek/Kimi MLA)
prefill. Runs ONLY the drafter's KV-processing path — "MM, norm, ROPE, kv-update" — and skips
q_proj / SDPA / o_proj / MLP (attention + feedforward), which are decode-only. Mirrors the
verifier's own kv-only fast path (``ttMLA._forward_kv_only`` / ``_kv_stem``), swapping MLA's
fused-latent geometry for the drafter's Qwen3 GQA:

    per verifier target layer L in target_layer_ids:                       # streamed during the layer loop
        reduced += fc_slice_L( h_L )                                       # row-parallel matmul, TP-reduced
    target_hidden = hidden_norm(reduced)                                   # once, after the loop
    per draft layer D in 0..num_hidden_layers-1:
        k = rope( k_norm( k_proj_D(target_hidden) ) )  -> K_cache[D]        # k_norm + rope (Qwen3 half-split)
        v =        v_proj_D(target_hidden)             -> V_cache[D]        # V: no norm, no rope

The FC context projection is decomposed across target layers because
``Linear(concat[h_1..h_6]) == sum_i fc_slice_i @ h_i`` — so it accumulates as the verifier streams
its layers, matching the tt-blaze decode-side ``FCMatmulForward`` accumulation.

PHASE-1 SHARDING (correctness-first; the migration/distributed format is Phase 2):
  * hidden is TP-sharded on the verifier residual stream -> the FC tap is row-parallel + a TP all-reduce.
  * ``target_hidden`` is TP-replicated (after the all-reduce); the caller supplies it seq-contiguous /
    seq-replicated (the standalone unit test does exactly this).
  * k_proj/v_proj are column-parallel: KV heads are split across the TP axis (kv_heads/tp per device) —
    the natural drafter layout, matching the tt-blaze decode drafter (num_kv_heads=8, head_dim=128).
  * The drafter K and V caches are SEPARATE tensors (GQA), TP-sharded on kv-head, seq NOT SP-sharded.
    Phase 2 changes this to the tt-blaze-consumable HEIGHT_SHARDED/bf8 migration format.

Nothing in this file has run on hardware yet — program configs / memory configs are left to ttnn
defaults with `TODO(bring-up)` markers; expect on-hardware tuning while PCC'ing against
``tests/speculative_decoding/dflash/test_dflash.py``.
"""

from __future__ import annotations

from pathlib import Path
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
        weight_cache_path: Optional[Path] = None,
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
        #   "sliced" — stream fc_slice_i @ h_i and accumulate at tap time (smallest memory; the
        #              hardware-validated path used by test_dflash.py). Nothing raw is stored.
        #   "concat" — store the raw tapped hiddens, then at write time concat them and do ONE fc matmul
        #              (fc(concat) == Σ fc_slice_i @ h_i). Needs the full fc pre-permuted so a contiguous
        #              TP shard aligns with the on-device grouped concat (see _load_weights).
        assert fc_mode in ("sliced", "concat"), f"fc_mode must be 'sliced' or 'concat', got {fc_mode!r}"
        self.fc_mode = fc_mode
        # Prefill builds drafter KV for the FULL chunk the verifier hands it (e.g. 5120 tokens), so the
        # cache is sized to max_seq_len — NOT capped at 4k. The 4k (config.context_len) is the drafter's
        # DECODE context window, applied at MIGRATION (send only the last 4k to decode), per #49586's
        # "cache max_seq_len, migrate last 4k". Capping here would break integration with the verifier's
        # 5k-token prefill chunks.
        self.cache_seq = max_seq_len if max_seq_len is not None else config.context_len

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

        self._load_weights(state_dict, weight_cache_path)
        # RoPE tables are built lazily (sized to the sequence actually roped, not to the cache) so the
        # single-shot path needs no slice and no oversized table — see _ensure_rope.
        self._rope_cos = self._rope_sin = None
        self._rope_end = 0
        self._alloc_caches()
        self._reduced_accum: Optional[ttnn.Tensor] = None  # "sliced": running TP-partial FC sum
        self._taps: list = [None] * len(config.target_layer_ids)  # "concat": stored raw taps

    # ------------------------------------------------------------------ setup
    def _mesh_mappers(self):
        """Row-parallel (shard tensor dim 0 on TP) and column-parallel (shard tensor dim 1 on TP)
        2D-weight mappers, replicating on the SP axis. Mirrors ttMLA._convert_and_cache_weights."""
        row = [None, None]
        row[self.tp_axis] = 0  # shard the contraction (input) dim across TP
        col = [None, None]
        col[self.tp_axis] = 1  # shard the output dim across TP
        col[self.sp_axis] = None
        mapper_row = ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=row)
        mapper_col = ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=col)
        return mapper_row, mapper_col

    def _load_weights(self, state_dict: dict | None, cache_path: Optional[Path]):
        cfg = self.config
        H, kv_dim, D = cfg.hidden_size, cfg.kv_dim, cfg.head_dim
        mapper_row, mapper_col = self._mesh_mappers()
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        have = state_dict is not None and "fc.weight" in state_dict

        def _cache(name):
            return str(cache_path / f"dflash.{name}") if cache_path else None

        def _linear_w(torch_w, mapper, name):
            # torch_w is the HF Linear weight [out, in]; ttnn.linear wants [in, out].
            t = torch_w.transpose(-2, -1).contiguous() if torch_w is not None else None
            return ttnn.as_tensor(
                t,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,  # TODO(bring-up): bfloat8_b for perf / to match decode caches.
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
                cache_file_name=_cache(name),
            )

        def _norm_w(torch_w, name):
            # RMSNorm weight [dim] -> [1, 1, dim/32, 32] ROW_MAJOR bf16, replicated (matches ttMLA).
            t = torch_w.reshape(1, 1, -1, ttnn.TILE_SIZE) if torch_w is not None else None
            return ttnn.as_tensor(
                t,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
                cache_file_name=_cache(name),
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
                self.fc_slices.append(_linear_w(sl, mapper_row, f"fc_slice_{idx}"))
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
            self.fc_perm = _linear_w(fc_perm_w, mapper_row, "fc_perm")

        # hidden_norm spans the full H=7168 → it MUST be the DISTRIBUTED (TP-sharded) norm, exactly
        # like the model's attn_norm/ffn_norm. A plain ttnn.rms_norm over the replicated 7168 forces
        # one core to hold 7168-wide (224-tile) CBs and overflows L1. cluster_axis=tp_axis matches the
        # H/tp shard it consumes (see write_kv_cache: mesh_partition -> norm -> all_gather).
        self.hidden_norm = TtDistributedRmsNorm(
            mesh_device=self.mesh_device,
            emb_dim=cfg.hidden_size,
            epsilon=cfg.rms_norm_eps,
            torch_weight=state_dict["hidden_norm.weight"] if have else None,
            cluster_axis=self.tp_axis,
            num_links=self.num_links,
            topology=self.topology,
            weight_cache_path=cache_path,
            cache_name_prefix="dflash.hidden_norm",
        )

        # Per draft layer: k/v proj column-parallel (KV heads split across TP), per-head k_norm replicated.
        self.k_proj, self.v_proj, self.k_norm = [], [], []
        for i in range(cfg.num_hidden_layers):
            kw = state_dict[self._K_PROJ.format(i=i)] if have else None  # [kv_dim, H]
            vw = state_dict[self._V_PROJ.format(i=i)] if have else None
            kn = state_dict[self._K_NORM.format(i=i)] if have else None  # [head_dim]
            self.k_proj.append(_linear_w(kw, mapper_col, f"l{i}.k_proj"))
            self.v_proj.append(_linear_w(vw, mapper_col, f"l{i}.v_proj"))
            self.k_norm.append(_norm_w(kn, f"l{i}.k_norm"))
        assert kv_dim == cfg.num_key_value_heads * D

    def _ensure_rope(self, end: int) -> None:
        """Build (memoized) drafter deepseek_yarn cos/sin covering positions [0, end), sized to what's
        actually roped in a call — for single-shot prefill that's ``seq``, so ``write_kv_cache`` can use
        the table directly with NO slice. Only (re)builds when a longer range is later requested (a
        further chunk); yarn inv_freq is position-independent, so growing the table never changes the
        values at existing positions. HALF-SPLIT (interleave=False) to match Qwen3 rotate_half +
        ttnn.experimental.rotary_embedding_hf; full head_dim (128) rotated (unlike the MLA 64-dim pe).
        Replicated across the mesh (seq is unsharded in Phase 1)."""
        if self._rope_cos is not None and self._rope_end >= end:
            return
        hf = build_drafter_rope_hf_config(self.config, max_seq_len=end)
        cos, sin = get_cos_sin_matrix(hf, interleave=False)  # [1, 1, end, head_dim]
        cos, sin = cos[..., :end, :], sin[..., :end, :]
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        if self._rope_cos is not None:
            ttnn.deallocate(self._rope_cos)
            ttnn.deallocate(self._rope_sin)
        self._rope_cos = ttnn.from_torch(
            cos, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
        self._rope_sin = ttnn.from_torch(
            sin, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
        self._rope_end = end

    def _alloc_caches(self):
        """Separate K and V drafter caches: host [num_layers, num_kv_heads, cache_seq, head_dim],
        TP-sharded on kv-head (dim 1), SP-replicated. Phase 2 will switch to the tt-blaze
        HEIGHT_SHARDED / bf8 migration layout."""
        cfg = self.config
        shape = (cfg.num_hidden_layers, cfg.num_key_value_heads, self.cache_seq, cfg.head_dim)
        shard = [None, None]
        shard[self.tp_axis] = 1  # kv-head dim across TP
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

    # ----------------------------------------------------------------- runtime
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
        [1, 1, seq, hidden/tp] (TP-sharded on hidden, seq contiguous/replicated — SP-gather before tapping
        if the caller's seq is SP-sharded).

        fc_mode="sliced": stream the FC-slice matmul and accumulate the (still TP-partial) sum; the TP
            all-reduce is deferred to write_kv_cache (sum-then-reduce == reduce-then-sum).
        fc_mode="concat": store the raw tap for a single fc(concat) at write time."""
        if not self.is_target_layer(global_layer_idx):
            return
        idx = self.config.target_layer_ids.index(global_layer_idx)
        if self.fc_mode == "concat":
            # The drafter TAKES OWNERSHIP of the tensor (deallocated in write_kv_cache()/reset()); the
            # caller must not free it. The integration hook hands over a fresh (SP-gathered) tensor, so no
            # clone is needed. TODO(bring-up): if a caller passes a tensor it will mutate/reuse (e.g. the
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
            # TODO(bring-up): add a tuned program_config (see ttMLA._get_mm_kwargs).
        )
        if self._reduced_accum is None:
            self._reduced_accum = partial
        else:
            summed = ttnn.add(self._reduced_accum, partial)
            ttnn.deallocate(self._reduced_accum)
            ttnn.deallocate(partial)
            self._reduced_accum = summed

    def _tp_all_reduce(self, t: ttnn.Tensor) -> ttnn.Tensor:
        """Sum a TP-partial [1,1,seq,H] across the TP axis -> replicated [1,1,seq,H]. Mirrors
        ttMLA._kv_stem (all_gather on dim 1 + fast_reduce_nc)."""
        if self.tp_factor == 1:
            return t
        gathered = ttnn.all_gather(
            t, dim=1, cluster_axis=self.tp_axis, num_links=self.num_links, topology=self.topology
        )
        ttnn.deallocate(t)
        reduced = ttnn.experimental.fast_reduce_nc(
            gathered, dims=[1], output=None, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
        )
        ttnn.deallocate(gathered)
        return reduced

    def _split_heads(self, proj: ttnn.Tensor, seq: int) -> ttnn.Tensor:
        """[1, 1, seq, kv_heads_local*head_dim] -> [1, kv_heads_local, seq, head_dim]."""
        cfg = self.config
        x = ttnn.reshape(proj, (1, seq, self.kv_heads_local, cfg.head_dim))
        # TODO(bring-up): prefer ttnn.experimental.nlp_create_qkv_heads if TILE reshape/permute misbehaves.
        return ttnn.permute(x, (0, 2, 1, 3))

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
        reduced = self._tp_all_reduce(reduced_partial)  # [1,1,seq,H] replicated on TP
        seq = reduced.shape[2]
        assert positions_start + seq <= self.cache_seq, (
            f"positions_start+seq ({positions_start + seq}) exceeds cache_seq ({self.cache_seq}); "
            f"construct with max_seq_len >= the chunk length (the cache is sized to the full chunk; "
            f"the 4k window is applied at migration, not here)"
        )

        # Distributed hidden_norm: partition the replicated full-H `reduced` across TP so each core
        # norms only H/tp, run TtDistributedRmsNorm (stats all-gathered → correct full-H norm), then
        # all-gather back to replicated so the column-parallel k/v_proj can consume it.
        if self.tp_factor > 1:
            part = ttnn.mesh_partition(reduced, dim=3, cluster_axis=self.tp_axis)  # [1,1,seq,H/tp]
            ttnn.deallocate(reduced)
            reduced = part
        target_hidden = self.hidden_norm(reduced)  # [1,1,seq,H/tp] (or full H when tp==1)
        ttnn.deallocate(reduced)
        if self.tp_factor > 1:
            gathered = ttnn.all_gather(
                target_hidden, dim=3, cluster_axis=self.tp_axis, num_links=self.num_links, topology=self.topology
            )
            ttnn.deallocate(target_hidden)
            target_hidden = gathered  # [1,1,seq,H] replicated on TP

        # RoPE cos/sin for this chunk's absolute positions [positions_start, positions_start+seq).
        # This is LOOP-INVARIANT (rope is identical for every draft layer), so slice ONCE here rather
        # than per layer — ttnn.slice allocates+copies, and rotary_embedding_hf (half-split variant) has
        # no on-device position-offset arg, so a slice is the accepted pattern (cf. the DS indexer). When
        # the prebuilt table already matches the request, skip the slice entirely.
        # TODO(Phase 3): for chunked prefill at arbitrary offsets, prefer an indexed rope op that takes
        # the offset on-device (like MLA's rotary_embedding_indexed) to drop the per-chunk slice too.
        end = positions_start + seq
        self._ensure_rope(end)  # table sized to the roped positions; single-shot from 0 => no slice
        if positions_start == 0 and self._rope_end == seq:
            cos, sin, rope_owned = self._rope_cos, self._rope_sin, False
        else:
            cos = ttnn.slice(self._rope_cos, [0, 0, positions_start, 0], [1, 1, end, cfg.head_dim])
            sin = ttnn.slice(self._rope_sin, [0, 0, positions_start, 0], [1, 1, end, cfg.head_dim])
            rope_owned = True

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
            k = self._split_heads(k, seq)  # [1, kvh_local, seq, head_dim]
            v = self._split_heads(v, seq)
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
            # Write into cache slot `i` (layer as the fill "user" dim). TODO(bring-up/Phase2): the migration
            # writer + SP-sharded seq + bf8 cache replace this seq-replicated fill_cache_for_user_.
            ttnn.kv_cache.fill_cache_for_user_(self.k_cache, k, i)
            ttnn.kv_cache.fill_cache_for_user_(self.v_cache, v, i)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        if rope_owned:
            ttnn.deallocate(cos)
            ttnn.deallocate(sin)
        ttnn.deallocate(target_hidden)
