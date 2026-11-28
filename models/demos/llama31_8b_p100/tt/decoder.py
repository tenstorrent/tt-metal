# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.ttt_compat.tt.decoder import *  # noqa: F401,F403  # isort: skip
import models.ttt_compat.tt.decoder as _tt
import ttnn
from models.demos.llama31_8b_p100.tt import wiring
from models.demos.llama31_8b_p100.tt.mlp import MLP
from models.ttt_compat.tt.common import Mode
from models.ttt_compat.tt.model_config import TensorGroup


def fuse_norms(args, mode):
    if mode in (Mode.PREFILL, "prefill"):
        return wiring.on("rms_norm_residual", args) is not None
    if mode in (Mode.DECODE, "decode"):
        return (
            wiring.on("rms_norm_residual_decode", args) is not None and wiring.on("decode_cross_norm", args) is not None
        )
    return False


class TransformerBlock(_tt.TransformerBlock):
    def __init__(self, *args, **kwargs):
        saved = _tt.MLP
        _tt.MLP = MLP
        try:
            super().__init__(*args, **kwargs)
        finally:
            _tt.MLP = saved
        self._rms_norm_residual = wiring.on("rms_norm_residual", self.args)
        self._rms_norm_residual_decode = wiring.on("rms_norm_residual_decode", self.args)
        self._decode_cross_norm = wiring.on("decode_cross_norm", self.args)

    def _llama31_8b_decode_tail_ok(self, mode, attn_out, residual, ff_norm_config):
        return (
            mode in (Mode.DECODE, "decode")
            and self._rms_norm_residual_decode is not None
            and self._rms_norm_residual_decode.covers(self, attn_out, residual, ff_norm_config)
        )

    def _llama31_8b_decode_tail(self, attn_out, residual, skip_mem_cfg, ff_norm_config, next_attention_norm=None):
        if not (attn_out.is_sharded() and attn_out.memory_config().buffer_type == ttnn.BufferType.L1):
            attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
        hidden_states = self._rms_norm_residual_decode.add_norm(attn_out, residual, self.ff_norm.norm, ff_norm_config)
        ttnn.deallocate(attn_out)
        hidden_states = self.feed_forward.forward(hidden_states, Mode.DECODE)
        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )
        next_norm = getattr(next_attention_norm, "norm", None)
        attn_norm_config = self.args.get_norm_config("attn", Mode.DECODE, self.prefetcher)
        if self._decode_cross_norm is not None and self._decode_cross_norm.covers(
            next_norm, activation_dtype, attn_norm_config, hidden_states, residual
        ):
            normed = self._decode_cross_norm.add_next_norm(hidden_states, residual, next_norm, attn_norm_config)
            ttnn.deallocate(hidden_states)
            self.next_normed = normed
            return residual
        return ttnn.add(residual, hidden_states, memory_config=skip_mem_cfg, dtype=activation_dtype or ttnn.bfloat16)

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
        normed_x=None,
        next_attention_norm=None,
    ) -> ttnn.Tensor:
        TG = self.args.is_galaxy
        self.next_normed = None
        residual = x

        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)

        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        rot_mats = (
            rot_mats_local if (hasattr(self.attention, "is_sliding") and self.attention.is_sliding) else rot_mats_global
        )

        attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
        attn_in = normed_x if normed_x is not None else self.attention_norm(x, mode, norm_config=attn_norm_config)

        if batch_size > 1:
            attn_in = ttnn.reshape(attn_in, [batch_size, 1, attn_in.shape[-2] // batch_size, -1])

        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )
        ff_norm_config_decode = self.args.get_norm_config("ff", mode, self.prefetcher)
        if self._llama31_8b_decode_tail_ok(mode, attn_out, residual, ff_norm_config_decode):
            return self._llama31_8b_decode_tail(
                attn_out, residual, skip_mem_cfg, ff_norm_config_decode, next_attention_norm
            )
        if mode == Mode.PREFILL and batch_size > 1:
            residual = ttnn.reshape(residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1])
        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)

        ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
        fused = None
        if self.pre_ff_norm is None and mode in (Mode.PREFILL, "prefill") and self._rms_norm_residual is not None:
            fused = self._rms_norm_residual.add_norm(residual, attn_out, self.ff_norm, ff_norm_config)
        if fused is not None:
            hidden_states, residual = fused
            x.deallocate(True)
        else:
            if self.pre_ff_norm is None:
                hidden_states = ttnn.add(
                    residual, attn_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None
                )
                residual = hidden_states
                if mode == "prefill":
                    x.deallocate(True)
            else:
                hidden_states = attn_out

            hidden_states = self.ff_norm(hidden_states, mode, norm_config=ff_norm_config)

        if self.pre_ff_norm is not None:
            if self.num_devices > 1 and not self.args.is_distributed_norm(mode):
                hidden_states = ttnn.mesh_partition(
                    hidden_states,
                    memory_config=hidden_states.memory_config(),
                    dim=3,
                    cluster_axis=1,
                )

            hidden_states = ttnn.add(
                residual, hidden_states, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None
            )
            residual = hidden_states
            pre_ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
            hidden_states = self.pre_ff_norm(hidden_states, mode, norm_config=pre_ff_norm_config)

        ttnn.deallocate(attn_out)

        if TG and mode == "decode":
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=self.args.get_mlp_act_mem_config(mode))

        hidden_states = self.feed_forward.forward(hidden_states, mode)

        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        if self.post_ff_norm is not None:
            post_ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
            hidden_states = self.post_ff_norm(hidden_states, mode, norm_config=post_ff_norm_config)
            if self.num_devices > 1 and not self.args.is_distributed_norm(mode):
                hidden_states = ttnn.mesh_partition(
                    hidden_states,
                    memory_config=hidden_states.memory_config(),
                    dim=3,
                    cluster_axis=1,
                )

        if (
            next_attention_norm is not None
            and not TG
            and (activation_dtype or ttnn.bfloat16) == ttnn.bfloat16
            and mode in (Mode.PREFILL, "prefill")
            and self._rms_norm_residual is not None
        ):
            fused = self._rms_norm_residual.add_norm(
                residual,
                hidden_states,
                next_attention_norm,
                self.args.get_norm_config("attn", mode, self.prefetcher),
            )
            if fused is not None:
                self.next_normed, out = fused
                return out

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )

        return out
