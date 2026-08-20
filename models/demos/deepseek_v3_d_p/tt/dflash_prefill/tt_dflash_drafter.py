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

from typing import Optional, Tuple, Union

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.utils import build_drafter_rope_hf_config
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup, interleaved_to_halfsplit_perm
from models.demos.deepseek_v3_d_p.tt.tt_distributed_rms_norm import TtDistributedRmsNorm

WEIGHT_DTYPE = ttnn.bfloat8_b  # fc / k_proj / v_proj projection weights
NORM_WEIGHT_DTYPE = ttnn.bfloat16  # k_norm RMSNorm weight


class TtDFlashDrafter:
    # safetensors key templates for the 20-tensor prefill subset.
    _K_PROJ = "layers.{i}.self_attn.k_proj.weight"
    _V_PROJ = "layers.{i}.self_attn.v_proj.weight"
    _K_NORM = "layers.{i}.self_attn.k_norm.weight"

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: DFlashDrafterConfig,
        state_dict: dict,
        *,
        sp_axis: int = 0,
        tp_axis: int = 1,
        max_seq_len: Optional[int] = None,
        chunk_size: Optional[int] = None,
        num_links: int = 1,
        topology: Union[ttnn.Topology, Tuple[ttnn.Topology, ttnn.Topology]] = ttnn.Topology.Linear,
        owned_target_layer_ids: Optional[tuple] = None,
        build_kv_tail: bool = True,
    ):
        self.mesh_device = mesh_device
        self.config = config
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self.tp_factor = mesh_device.shape[tp_axis]
        self.sp_factor = mesh_device.shape[sp_axis]
        self.num_links = num_links
        # `topology` may arrive per-axis (2-tuple) from the runner; all of this drafter's ccl ops run along
        # tp_axis, so collapse to that axis's scalar. Tests pass a scalar directly (isinstance guard).
        self.topology = topology[tp_axis] if isinstance(topology, tuple) else topology
        # Pipeline distribution: each rank taps only the target layers it owns (default = ALL, for single-rank),
        # and only the KV-tail rank (build_kv_tail) builds hidden_norm + the per-draft-layer k/v tail.
        self.owned_target_layer_ids = (
            tuple(owned_target_layer_ids) if owned_target_layer_ids is not None else tuple(config.target_layer_ids)
        )
        self._owned_set = set(self.owned_target_layer_ids)
        assert self._owned_set <= set(config.target_layer_ids), (
            f"owned_target_layer_ids {self.owned_target_layer_ids} must be a subset of "
            f"config.target_layer_ids {tuple(config.target_layer_ids)}"
        )
        self.build_kv_tail = build_kv_tail
        # Prefill builds drafter KV for the FULL chunk the verifier hands it (e.g. 5120 tokens), so the
        # cache is sized to max_seq_len — NOT capped at 4k.
        self.cache_seq = max_seq_len if max_seq_len is not None else config.context_len
        self.chunk_size = chunk_size

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

        if self.build_kv_tail and config.rope_convention != "interleaved":
            raise NotImplementedError(
                f"rope_convention={config.rope_convention!r} is not supported for DFlash: use 'interleaved'"
            )
        self._load_weights(state_dict)
        self._rope: Optional[dict] = None
        if self.build_kv_tail:
            assert self.chunk_size is not None, "chunk_size is required to build the drafter rope table (KV-tail rank)"
            hf = build_drafter_rope_hf_config(self.config, max_seq_len=self.cache_seq)
            self._rope = RotarySetup(hf, self.mesh_device, sp_axis=self.sp_axis).get_rope_tensors_indexed(
                cache_seq_len_global=self.cache_seq, chunk_size_global=self.chunk_size
            )
        # K/V caches are owned by the CALLER (see allocate_dflash_kv_cache) and passed into
        # forward() — the drafter does not hold them, mirroring the MLA prefill model's kvpe_cache.
        self._reduced_accum: Optional[ttnn.Tensor] = None  # running TP-partial FC sum (Σ fc_slice_i @ h_i)
        # Partial forwarded from upstream pipeline ranks (import_partial), already reduce_scattered to
        # [1,1,seq,H/tp]; summed into this rank's finalized partial. None on rank 0 / single-rank.
        self._running_sharded: Optional[ttnn.Tensor] = None

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

    def _load_weights(self, state_dict: dict):
        cfg = self.config
        H, kv_dim, D = cfg.hidden_size, cfg.kv_dim, cfg.head_dim
        mapper_row, mapper_col = self._mesh_mappers()
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)

        def _linear_w(torch_w, mapper):
            # torch_w is the HF Linear weight [out, in]; ttnn.linear wants [in, out].
            t = torch_w.transpose(-2, -1).contiguous()
            return ttnn.as_tensor(
                t,
                device=self.mesh_device,
                dtype=WEIGHT_DTYPE,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        def _norm_w(torch_w):
            # RMSNorm weight [dim] -> [1, 1, dim/32, 32] ROW_MAJOR bf16, replicated (matches ttMLA).
            t = torch_w.reshape(1, 1, -1, ttnn.TILE_SIZE)
            return ttnn.as_tensor(
                t,
                device=self.mesh_device,
                dtype=NORM_WEIGHT_DTYPE,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        # FC context projection, one row-parallel block per target layer: fc(concat[h_1..h_n]) == Σ_i fc_slice_i
        # @ h_i, so it accumulates at tap time. Build only this rank's owned blocks (columns keyed by target id).
        all_targets = list(cfg.target_layer_ids)
        fc_full = state_dict["fc.weight"]  # [H, n*H]
        self.fc_slices = {}  # global_layer_idx -> device tensor [H(in), H(out)]
        for global_idx in self.owned_target_layer_ids:
            pos = all_targets.index(global_idx)  # this layer's column block within the full fc
            sl = fc_full[:, pos * H : (pos + 1) * H]  # [H(out), H(in)]
            self.fc_slices[global_idx] = _linear_w(sl, mapper_row)

        # The tail (hidden_norm + per-draft-layer k/v/norm/rope) runs ONLY on the build_kv_tail rank; non-tail
        # ranks accumulate + forward the FC partial and skip materializing these tensors.
        if not self.build_kv_tail:
            self.hidden_norm = None
            self.k_proj, self.v_proj, self.k_norm = [], [], []
            return

        # hidden_norm spans the full H=7168 → it MUST be the DISTRIBUTED (TP-sharded) norm, exactly
        # like the model's attn_norm/ffn_norm. A plain ttnn.rms_norm over the replicated 7168 forces
        # one core to hold 7168-wide (224-tile) CBs and overflows L1. cluster_axis=tp_axis matches the
        # H/tp shard it consumes (see forward(): reduce_scatter -> norm -> all_gather).
        self.hidden_norm = TtDistributedRmsNorm(
            mesh_device=self.mesh_device,
            emb_dim=cfg.hidden_size,
            epsilon=cfg.rms_norm_eps,
            torch_weight=state_dict["hidden_norm.weight"],
            cluster_axis=self.tp_axis,
            num_links=self.num_links,
            topology=self.topology,
        )

        src = torch.argsort(interleaved_to_halfsplit_perm(D))  # [0, 64, 1, 65, ...] for head_dim=128

        # Per draft layer: k/v proj column-parallel (KV heads split across TP), per-head k_norm replicated.
        self.k_proj, self.v_proj, self.k_norm = [], [], []
        for i in range(cfg.num_hidden_layers):
            kw = state_dict[self._K_PROJ.format(i=i)]  # [kv_dim, H]; output rows are per-head head_dim blocks
            vw = state_dict[self._V_PROJ.format(i=i)]
            kn = state_dict[self._K_NORM.format(i=i)]  # [head_dim], shared across heads, applied before rope
            # kw'[h*D + j, :] = kw[h*D + src[j], :] for every head h; kn'[j] = kn[src[j]].
            kw = kw.view(cfg.num_key_value_heads, D, H)[:, src, :].reshape(kv_dim, H).contiguous()
            kn = kn[src].contiguous()
            self.k_proj.append(_linear_w(kw, mapper_col))
            self.v_proj.append(_linear_w(vw, mapper_col))
            self.k_norm.append(_norm_w(kn))
        assert kv_dim == cfg.num_key_value_heads * D

    def reset(self):
        """Clear the FC accumulator + any imported upstream partial — call at the start of each prefill
        sequence/chunk."""
        if self._reduced_accum is not None:
            ttnn.deallocate(self._reduced_accum)
        self._reduced_accum = None
        if self._running_sharded is not None:
            ttnn.deallocate(self._running_sharded)
        self._running_sharded = None

    def is_target_layer(self, global_layer_idx: int) -> bool:
        return global_layer_idx in self.config.target_layer_ids

    def tap(self, hidden_states: ttnn.Tensor, global_layer_idx: int) -> None:
        """FC context tap at a verifier target layer. ``hidden_states`` is the residual-stream output
        [1, 1, seq, hidden/tp], TP-sharded on hidden and SP-sharded on seq (each chip taps only its own
        slice — NO SP-gather).

        Streams the FC-slice matmul and accumulates the (still TP-partial) sum; the cross-TP combine is
        deferred to _finalize_sharded_partial's reduce_scatter (local sum-then-scatter == scatter-then-sum).
        The caller RETAINS ownership of ``hidden_states`` — this only reads it for the matmul (does not
        store or free it). Silently skips a target layer this rank does not own (its fc block lives on
        another pipeline rank)."""
        if global_layer_idx not in self._owned_set:
            return
        partial = ttnn.linear(
            hidden_states,
            self.fc_slices[global_layer_idx],
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

    def import_partial(self, sharded: ttnn.Tensor) -> None:
        """Seed the running partial with the upstream rank's finalized sharded partial [1,1,seq,H/tp]
        (from its export_partial). The drafter takes ownership (freed in _finalize_sharded_partial/reset).
        Non-first pipeline ranks call this once per chunk, before the tap phase."""
        assert self._running_sharded is None, "import_partial called twice without reset()"
        self._running_sharded = sharded

    def _finalize_sharded_partial(self) -> ttnn.Tensor:
        """Combine this rank's accumulated row-parallel FC partials across TP (reduce_scatter → the thin
        [1,1,seq,H/tp] shard, matching the hidden layout) and add any partial forwarded from upstream ranks.
        Correct by linearity: Σ_r reduce_scatter(A_r) == reduce_scatter(Σ_r A_r). Consumes _reduced_accum
        and _running_sharded; returns the summed sharded partial (full H when tp==1)."""
        if self._reduced_accum is None:
            # This rank owns no target layer this chunk — pass the upstream partial straight through.
            assert self._running_sharded is not None, "_finalize: this rank neither tapped nor imported a partial"
            out = self._running_sharded
            self._running_sharded = None
            return out
        reduced_partial = self._reduced_accum
        self._reduced_accum = None
        # reduce_scatter SUMS the row-parallel partials across TP AND scatters on hidden in ONE op → the
        # [1,1,seq,H/tp] shard the distributed hidden_norm wants. Matches MLA o_proj/q_a_proj + MoE tt_reduce.
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
        if self._running_sharded is not None:
            summed = ttnn.add(reduced, self._running_sharded)  # += upstream ranks' partial (same layout)
            ttnn.deallocate(reduced)
            ttnn.deallocate(self._running_sharded)
            self._running_sharded = None
            reduced = summed
        return reduced

    def export_partial(self) -> ttnn.Tensor:
        """Non-tail ranks: return this rank's finalized sharded partial [1,1,seq,H/tp] to forward to the
        next pipeline rank (packed alongside the hidden over the D2D handoff). Consumes the FC accumulator
        + imported partial; the CALLER owns the returned tensor."""
        return self._finalize_sharded_partial()

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

    def forward(
        self,
        k_cache: ttnn.Tensor,
        v_cache: ttnn.Tensor,
        kv_actual_global: int = 0,
        *,
        slot_idx: int = 0,
    ) -> None:
        """Finalize into the caller-owned ``k_cache``/``v_cache`` (allocate via
        ``allocate_dflash_kv_cache``): consume the accumulated TP-partial FC output, TP-reduce it,
        hidden_norm, then per draft layer project/norm/rope K and project V, writing each into its cache
        slot. The caches are passed in (not owned by the drafter) so the runner drives their lifecycle +
        dtype, exactly like the MLA prefill model takes ``kvpe_cache`` in ``forward()``.

        ``kv_actual_global`` is the chunk's absolute KV offset in **GLOBAL** tokens (the same unit and the
        same argument the MLA path passes to ``update_padded_kv_cache``; cf. ``mla.py``'s chunked-prefill
        write), so chunk c of a sequence passes ``c * chunk_global`` and a multi-turn resume passes the
        reused prefix length. It must be tile-aligned — align DOWN to the previous 32 and replay the ≤31
        dropped tokens rather than rounding up, since sub-tile offsets are rejected outright and the kernel's
        tile-granular staircase and the host mirror disagree off-tile.

        ``slot_idx`` selects which user's cache slot to fill; the cache is user-major
        (``slot_idx * num_hidden_layers + layer_idx``), as ``allocate_dflash_kv_cache`` lays it out.

        The taps for this chunk need NOT be seq-contiguous: token ids entering the transformer are already
        block-cyclic-gathered, so each chip's tap slice is exactly the rows its cache shard will hold, and
        the interleaved indexed rope op derives each chip's shard offset on-device from the whole-cache table
        built once in ``__init__``."""
        assert self.build_kv_tail, (
            "forward() on a non-tail drafter (build_kv_tail=False); non-tail ranks forward the partial "
            "via export_partial instead"
        )
        cfg = self.config
        # Sanity-check the un-sharded cache dims (layer/head_dim are not seq/SP-sharded, so .shape is
        # unambiguous here); the seq (dim 2) capacity is checked in GLOBAL tokens below.
        # dim0 is num_users * num_hidden_layers (user-major), so derive the slot count from it.
        for name, cache in (("k", k_cache), ("v", v_cache)):
            assert cache.shape[0] % cfg.num_hidden_layers == 0, (
                f"{name}_cache batch dim {cache.shape[0]} is not a multiple of num_hidden_layers "
                f"{cfg.num_hidden_layers} (allocate with allocate_dflash_kv_cache)"
            )
        num_slots = k_cache.shape[0] // cfg.num_hidden_layers
        assert (
            v_cache.shape[0] // cfg.num_hidden_layers == num_slots
        ), f"k/v caches disagree on slot count ({num_slots} vs {v_cache.shape[0] // cfg.num_hidden_layers})"
        assert 0 <= slot_idx < num_slots, f"slot_idx {slot_idx} out of range [0, {num_slots})"
        assert (
            k_cache.shape[-1] == cfg.head_dim and v_cache.shape[-1] == cfg.head_dim
        ), f"kv cache head_dim {k_cache.shape[-1]}/{v_cache.shape[-1]} != {cfg.head_dim}"
        # Combine this rank's accumulated FC partials across TP and add any upstream partial (import_partial).
        reduced = self._finalize_sharded_partial()  # [1,1,seq,H/tp] (or full H when tp==1)
        seq = reduced.shape[2]  # PER-CHIP seq (dim2, unchanged by the hidden scatter) == chunk_local
        chunk_global = seq * self.sp_factor
        # Mirror the MLA chunked-prefill write's guards (cf. mla.py). NOTE the units: kv_actual_global and
        # cache_seq are GLOBAL token counts while `seq` is per-chip — comparing a global offset against a
        # per-chip capacity is an sp_factor-sized error that shows up as a spurious capacity failure.
        assert chunk_global % (ttnn.TILE_SIZE * self.sp_factor) == 0, (
            f"chunk_global ({chunk_global}) must be a multiple of TILE_SIZE * sp "
            f"({ttnn.TILE_SIZE * self.sp_factor})"
        )
        assert (
            kv_actual_global % ttnn.TILE_SIZE == 0
        ), f"kv_actual_global ({kv_actual_global}) must be tile-aligned (a multiple of {ttnn.TILE_SIZE})"
        assert kv_actual_global + chunk_global <= self.cache_seq, (
            f"kv_actual_global ({kv_actual_global}) + chunk_global ({chunk_global}) exceeds the global cache "
            f"depth ({self.cache_seq}); construct with a larger max_seq_len (windowing happens at migration)"
        )
        assert self.cache_seq % chunk_global == 0, (
            f"cache_seq ({self.cache_seq}) must be a whole number of chunk_global ({chunk_global}) blocks; "
            "update_padded_kv_cache tiles the per-user cache block-cyclically in chunk_global-sized blocks, "
            "so a depth that is not a multiple would corrupt the layout"
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

        assert (
            chunk_global == self.chunk_size
        ), f"chunk_global ({chunk_global}) != chunk_size the rope table was built for ({self.chunk_size})"

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
            k = ttnn.rms_norm(
                k,
                weight=self.k_norm[i],
                epsilon=cfg.rms_norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.default_compute_kernel_config,
            )
            k = ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                k,
                self._rope["cos_matrix"],
                self._rope["sin_matrix"],
                self._rope["trans_matrix"],
                kv_actual_global=kv_actual_global,
                cluster_axis=self.sp_axis,
            )
            # The cache is bf8 (align w/ the decode KV cache) while k/v leave the projections in bf16;
            # update_padded_kv_cache FATALs unless cache and input dtypes match exactly, so typecast down
            # first — TILE both sides, so no relayout (mirrors MLA _to_cache_format). Keyed off *_cache.dtype
            # so a bf16-cache caller (override) still works.
            if k.dtype != k_cache.dtype:
                k_cast = ttnn.typecast(k, k_cache.dtype)
                ttnn.deallocate(k)
                k = k_cast
            if v.dtype != v_cache.dtype:
                v_cast = ttnn.typecast(v, v_cache.dtype)
                ttnn.deallocate(v)
                v = v_cast
            # Write this chunk into slot `slot_idx`, draft layer `i`. update_padded_kv_cache derives each
            # chip's local write offset on-device from kv_actual_global (the same call the MLA chunked-prefill
            # path makes) and is mesh-aware, unlike fill_cache_for_user_, whose program factory names out
            # mesh_dispatch_coordinate and so cannot express the per-chip staircase at all.
            # ALWAYS keyword args: the two nanobind overloads order the scalars differently.
            for cache, tensor in ((k_cache, k), (v_cache, v)):
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    cache,
                    tensor,
                    slot_idx=slot_idx,
                    layer_idx=i,
                    num_layers=cfg.num_hidden_layers,
                    kv_actual_global=kv_actual_global,
                    cluster_axis=self.sp_axis,
                )
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        ttnn.deallocate(target_hidden)
