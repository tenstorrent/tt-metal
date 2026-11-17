# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 (MLA) drafter prefill module: Kimi-K3 ``lightseekorg/kimi-k3-dflash2`` (and Inferact's MLA DSpark).

Builds the MLA drafter's *context* latent cache during the verifier prefill, the cache tt-blaze's MLA
drafter decoder reads (blaze commit f9ac6439eb, ``ops/dflash_mla_kv_injection``). The context path is

    target_hidden = hidden_norm(fc(concat(tapped hiddens)))                       (shared with GQA)
    [c_kv | k_pe] = target_hidden @ kv_a_proj_with_mqa                            (512 + 64 = 576)
    latent_row    = [ kv_a_layernorm(c_kv) | rope(k_pe) ]                          (one 576-wide row)

which is blaze ``tests/blaze/dflash/mla_golden.latent_kv`` (a port of TorchSpec's ``K3DSparkMLAAttention``).
Everything else the drafter runs -- the grouped dynamic causal conv, q_a/q_b, SDPA, o_proj, the MLP -- wraps
only the draft BLOCK's rows and is decode-only, so prefill skips it exactly as the GQA drafter skips q_proj.

Compared to :class:`TtDFlashDrafter` (GQA), only the per-draft-layer tail differs:

  * no K/V split and no head axis: one latent row per token serves as K and V (absorbed MLA), so there is
    ONE cache, ``[users * layers, 1, seq/(sp[*tp]), 576]``, laid out exactly like the verifier's KVPE cache;
  * RoPE rotates only k_pe (64 dims) in the Meta-interleaved pairing TorchSpec uses -- the same pairing the
    device op implements -- so unlike GQA no weight permutation is needed;
  * kv_a_proj_with_mqa is replicated over TP: the context feature is all-gathered to full H (as for GQA), so
    every TP chip computes the full 576-wide row and, under KV dedup (``tp_shard_kv``), keeps only its 1/tp
    window of the sequence -- the verifier KVPE's own write (``update_padded_kv_cache(tp_axis=...)``).

Taps, FC accumulation, pipeline partial export/import and the distributed hidden_norm are inherited unchanged.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import ttnn
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import DFlashDrafterConfig
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.tt_dflash_drafter import TtDFlashDrafter

# kv_a feeds the cached latent directly (no further projection before SDPA), so keep it bf16 like the
# verifier's own kv_a_proj_with_mqa (mla.py's dtype table). 5 x 7168 x 576 x 2 B = ~41 MB per chip on the
# KV-tail rank; bf8 halves that if DRAM gets tight.
KV_A_WEIGHT_DTYPE = ttnn.bfloat16


class TtDFlashMlaDrafter(TtDFlashDrafter):
    # safetensors key templates for the MLA prefill subset (TorchSpec / HF names, no ``model.`` prefix).
    _KV_A_PROJ = "layers.{i}.self_attn.kv_a_proj_with_mqa.weight"
    _KV_A_NORM = "layers.{i}.self_attn.kv_a_layernorm.weight"

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
        tp_shard_kv: bool = False,
    ):
        """``tp_shard_kv``: write the latent cache TP-deduped (each chip holds seq/(sp*tp) rows), matching a
        cache allocated by ``allocate_dflash_mla_kv_cache(..., tp_shard_kv=True)``. Follows the verifier's
        KVPE setting in the runtime so both caches share one migration-table layout."""
        tp_factor = mesh_device.shape[tp_axis]
        self.kv_tp_axis = tp_axis if (tp_shard_kv and tp_factor > 1) else None
        super().__init__(
            mesh_device,
            config,
            state_dict,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            max_seq_len=max_seq_len,
            chunk_size=chunk_size,
            num_links=num_links,
            topology=topology,
            owned_target_layer_ids=owned_target_layer_ids,
            build_kv_tail=build_kv_tail,
        )
        # kv_a feeds the cache with no later projection to absorb rounding, so accumulate in fp32 (the verifier's
        # K3 kv_a runs fp32-dest HiFi3; HiFi4 here is the conservative choice for bring-up).
        self.kv_a_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------ setup
    def _validate_tail_geometry(self) -> None:
        cfg = self.config
        assert cfg.is_mla, "TtDFlashMlaDrafter needs an MLA drafter config (attention_mode='mla')"
        assert not cfg.is_dspark, "is_dspark is the GQA DSpark gain fold; MLA folds rope_yarn_amplitude instead"
        for name, width in (("kv_lora_rank", cfg.kv_lora_rank), ("qk_rope_head_dim", cfg.qk_rope_head_dim)):
            # Both halves are sliced out of one TILE matmul output and the cache row is bf8/TILE.
            assert width % ttnn.TILE_SIZE == 0, f"{name} ({width}) must be a multiple of {ttnn.TILE_SIZE}"

    def _clear_tail_weights(self) -> None:
        self.kv_a_proj, self.kv_a_norm = [], []

    def _load_tail_weights(self, state_dict: dict) -> None:
        """Per draft layer: kv_a_proj_with_mqa [576, H] (replicated) and kv_a_layernorm [512] (replicated).

        Raw checkpoint weights. blaze folds input_layernorm into kv_a for DECODE (the block rows pass through
        it), but the context rows do not, which is why its KV-injection arm divides that gamma back out
        (``InjectNormGammaForCache``); prefill never folds it in the first place."""
        cfg = self.config
        r, rope = cfg.kv_lora_rank, cfg.qk_rope_head_dim
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        amplitude = cfg.rope_yarn_amplitude
        self.kv_a_proj, self.kv_a_norm = [], []
        for i in range(cfg.num_hidden_layers):
            kv_a = state_dict[self._KV_A_PROJ.format(i=i)]  # [576, H]: rows 0..511 c_kv, 512..575 k_pe
            assert tuple(kv_a.shape) == (
                r + rope,
                cfg.hidden_size,
            ), f"{self._KV_A_PROJ.format(i=i)} shape {tuple(kv_a.shape)} != ({r + rope}, {cfg.hidden_size})"
            if amplitude != 1.0:
                # TorchSpec's YaRN scales cos/sin by m(mscale)/m(mscale_all_dim); mla/rope.py's tables are
                # pure rotations. RoPE is linear, so the amplitude folds exactly into k_pe's projection rows.
                kv_a = kv_a.float().clone()
                kv_a[r:] *= amplitude
            self.kv_a_proj.append(
                ttnn.as_tensor(
                    kv_a.transpose(-2, -1).contiguous(),  # [H, 576]
                    device=self.mesh_device,
                    dtype=KV_A_WEIGHT_DTYPE,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate,
                )
            )
            kn = state_dict[self._KV_A_NORM.format(i=i)]
            assert kn.numel() == r, f"{self._KV_A_NORM.format(i=i)} has {kn.numel()} elements, expected {r}"
            self.kv_a_norm.append(self._norm_w(kn))

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        latent_cache: ttnn.Tensor,
        kv_actual_global: int = 0,
        *,
        slot_idx: int = 0,
        actual_end: Optional[int] = None,
        d2h_service=None,
        metadata_msg: Optional[ttnn.Tensor] = None,
        on_layer_complete: Optional[Callable[[int], None]] = None,
        layer_ack_base: int = 0,
        metadata: Optional[tuple] = None,
        trace_controller=None,
    ) -> None:
        """Finalize into the caller-owned ``latent_cache`` (``allocate_dflash_mla_kv_cache``): consume the FC
        accumulator, hidden_norm it, then per draft layer write ``[kv_a_layernorm(c_kv) | rope(k_pe)]`` into
        the slot. Every argument means what it means for :meth:`TtDFlashDrafter.forward`; the only change is
        one latent cache instead of a K and a V cache, so one ack per draft layer covers one write."""
        self._check_forward_args(d2h_service, metadata_msg)
        cfg = self.config
        L, r, rope = cfg.num_hidden_layers, cfg.kv_lora_rank, cfg.qk_rope_head_dim
        assert latent_cache.shape[0] % L == 0, (
            f"latent cache batch dim {latent_cache.shape[0]} is not a multiple of num_hidden_layers {L} "
            "(allocate with allocate_dflash_mla_kv_cache)"
        )
        num_slots = latent_cache.shape[0] // L
        assert metadata is not None or 0 <= slot_idx < num_slots, f"slot_idx {slot_idx} out of range [0, {num_slots})"
        assert (
            latent_cache.shape[-1] == cfg.latent_dim
        ), f"latent cache row width {latent_cache.shape[-1]} != kv_lora_rank + qk_rope_head_dim = {cfg.latent_dim}"
        self._check_cache_dedup(latent_cache)

        target_hidden = self._context_feature(kv_actual_global, actual_end=actual_end, metadata=metadata)
        seq = target_hidden.shape[2]

        for i in range(L):
            kv = ttnn.linear(
                target_hidden,
                self.kv_a_proj[i],
                compute_kernel_config=self.kv_a_compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, 1, seq, 576]
            c_kv = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, seq, r])
            k_pe = ttnn.slice(kv, [0, 0, 0, r], [1, 1, seq, r + rope])
            ttnn.deallocate(kv)
            # 512 wide: a plain (non-distributed) rms_norm fits, as the verifier's kv_a_layernorm.
            c_kv_normed = ttnn.rms_norm(
                c_kv,
                weight=self.kv_a_norm[i],
                epsilon=cfg.rms_norm_eps,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.default_compute_kernel_config,
            )
            ttnn.deallocate(c_kv)
            # k_pe is ONE rope head [1, 1, seq, 64], exactly the verifier MLA's tt_kv_rope operand.
            k_pe_roped = self._rope_indexed(k_pe, kv_actual_global, metadata)
            ttnn.deallocate(k_pe)
            row = ttnn.concat([c_kv_normed, k_pe_roped], dim=-1)  # [1, 1, seq, 576]
            ttnn.deallocate(c_kv_normed)
            ttnn.deallocate(k_pe_roped)
            # update_padded_kv_cache FATALs on a dtype mismatch; the cache is bf8 (as the decode cache).
            if row.dtype != latent_cache.dtype:
                cast = ttnn.typecast(row, latent_cache.dtype)
                ttnn.deallocate(row)
                row = cast
            self._write_cache(
                latent_cache, row, i, slot_idx, kv_actual_global, actual_end, metadata, tp_axis=self.kv_tp_axis
            )
            ttnn.deallocate(row)
            self._ack_layer(i, d2h_service, metadata_msg, on_layer_complete, layer_ack_base, trace_controller)
        ttnn.deallocate(target_hidden)

    def _check_cache_dedup(self, latent_cache: ttnn.Tensor) -> None:
        """The write's tp_axis must match how the cache was allocated: a deduped write into a replicated cache
        leaves (tp-1)/tp of every column's rows unwritten, and the reverse writes past each chip's shard."""
        from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import is_tp_deduped_kv_cache

        deduped = is_tp_deduped_kv_cache(latent_cache, tuple(self.mesh_device.shape))
        assert deduped == (self.kv_tp_axis is not None), (
            f"drafter latent cache is {'' if deduped else 'not '}TP-deduped but the drafter was built with "
            f"tp_shard_kv={self.kv_tp_axis is not None}; allocate and build with the same setting"
        )
