# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import ttnn
import torch
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import TensorGroup

from models.experimental.voxtraltts.tt.text_decoder_layer import remap_voxtral_text_state_dict
from models.experimental.voxtraltts.tt.voxtral_tt_args import get_VoxtralTTArgs


class VoxtralTTTextModel:
    """Direct tt_transformers Transformer wrapper for Voxtral text stack."""

    def __init__(self, inner_transformer: Transformer) -> None:
        self.inner = inner_transformer

    @classmethod
    def create(
        cls,
        *,
        args,
        dtype: ttnn.DataType,
        mesh_device,
        state_dict: dict[str, object],
        weight_cache_path: Path | None,
        paged_attention_config=None,
        use_paged_kv_cache: bool = False,
        attention_class=None,
        rope_setup_class=None,
        prefetcher=None,
    ) -> "VoxtralTTTextModel":
        inner = Transformer(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=remap_voxtral_text_state_dict(state_dict),
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            attention_class=attention_class,
            rope_setup_class=rope_setup_class,
            prefetcher=prefetcher,
        )
        return cls(inner)

    @classmethod
    def create_from_model_name(
        cls,
        *,
        mesh_device,
        model_name_or_path: str,
        dtype: ttnn.DataType = ttnn.bfloat8_b,
        max_batch_size: int = 1,
        max_seq_len: int = 4096,
        optimizations=None,
        preloaded_state_dict: dict[str, torch.Tensor] | None = None,
        paged_attention_config=None,
        use_paged_kv_cache: bool = False,
        attention_class=None,
        rope_setup_class=None,
        prefetcher=None,
    ) -> "VoxtralTTTextModel":
        VoxtralTTArgs = get_VoxtralTTArgs(preloaded_state_dict=preloaded_state_dict)
        args = VoxtralTTArgs(
            mesh_device,
            model_name_or_path=model_name_or_path,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            prefetcher=prefetcher,
        )
        state_dict = args.load_state_dict()
        return cls.create(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=args.weight_cache_path(dtype) / "qk_hf_rope",
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            attention_class=attention_class,
            rope_setup_class=rope_setup_class,
            prefetcher=prefetcher,
        )

    def prepare_inputs_prefill(self, *args, **kwargs):
        return self.inner.prepare_inputs_prefill(*args, **kwargs)

    def prepare_inputs_decode(self, *args, **kwargs):
        return self.inner.prepare_inputs_decode(*args, **kwargs)

    def switch_mode(self, mode):
        return self.inner.switch_mode(mode)

    def forward(self, *args, **kwargs):
        return self.inner.forward(*args, **kwargs)

    def prefill_from_embeds(
        self,
        inputs_embeds: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """TT prompt prefill with pre-built embeddings (voice-injected).

        Returns a ``[dim]`` bfloat16 CPU tensor — the final-normalized hidden state
        of the last valid prompt token (after norm, before LM head).

        Implementation: runs each token through ``decode_step_from_embeds`` one at
        a time.  This fills the KV cache correctly at every prompt position and
        avoids the prefill attention kernel whose circular-buffer allocation
        overflows L1 on P150 Blackhole for this model+grid combination.
        For a typical TTS prompt (≤ a few hundred tokens) the overhead is small.
        """
        S = inputs_embeds.shape[0]
        embeds_bf16 = inputs_embeds.to(dtype=torch.bfloat16)

        last_hidden: torch.Tensor | None = None
        for i in range(S):
            last_hidden = self.decode_step_from_embeds(embeds_bf16[i], start_pos + i)

        return last_hidden  # [dim]

    def decode_step_from_embeds(
        self,
        x_embed: torch.Tensor,
        current_pos_idx: int,
    ) -> torch.Tensor:
        """One DECODE step with pre-computed embedding; returns ``[dim]`` hidden state (post-norm, pre-LM-head).

        ``x_embed``: ``[dim]`` or ``[1, dim]`` bfloat16 CPU tensor (the next-step input embedding,
        e.g. multimodal audio codebook embedding).  Runs all transformer layers in DECODE mode so
        the KV cache is correctly updated at ``current_pos_idx``.
        """
        dim = self.inner.args.dim
        x_4d = x_embed.reshape(1, 1, 1, dim).to(dtype=torch.bfloat16).contiguous()
        emb_tt = ttnn.from_torch(
            x_4d,
            device=self.inner.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.inner.mesh_device),
        )

        dummy_token = torch.zeros(1, dtype=torch.int64)
        current_pos_t = torch.tensor([current_pos_idx], dtype=torch.int64)
        _, current_pos_tt, rope_idxs, page_table = self.prepare_inputs_decode(dummy_token, current_pos_t)

        rot_mats_global = self.inner.rope_setup.get_rot_mats(rope_idxs)
        rot_mats_local = (
            self.inner.rope_local_setup.get_rot_mats(rope_idxs) if hasattr(self.inner, "rope_local_setup") else None
        )

        decode_mem_cfg = self.inner.args.get_residual_mem_config(Mode.DECODE, self.inner.prefetcher)
        x_tt = ttnn.to_memory_config(emb_tt, decode_mem_cfg)
        ttnn.deallocate(emb_tt)

        for i, layer in enumerate(self.inner.layers):
            activation_dtype = self.inner.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=i, tensor=TensorGroup.ACTIVATION
            )
            if not self.inner.args.is_galaxy:
                x_tt = ttnn.to_memory_config(x_tt, decode_mem_cfg, activation_dtype)
            elif activation_dtype is not None and x_tt.dtype != activation_dtype:
                x_tt = ttnn.typecast(x_tt, activation_dtype)
            x_tt = layer(
                x_tt,
                current_pos_tt,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=0,
                mode=Mode.DECODE,
                page_table=page_table,
                kv_cache=None,
            )

        # Match HuggingFace ``outputs.hidden_states[-1]`` used by the CPU reference:
        # final decoder hidden after the model-level RMSNorm, before LM head.
        x_norm = self.inner.norm(
            x_tt,
            mode=Mode.DECODE,
            norm_config=self.inner.args.get_norm_config("lm_head", Mode.DECODE, self.inner.prefetcher),
        )
        ttnn.deallocate(x_tt)
        try:
            is_sharded = x_norm.memory_config().is_sharded()
        except RuntimeError:
            is_sharded = False
        if is_sharded:
            x_norm_interleaved = ttnn.sharded_to_interleaved(x_norm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x_norm)
            x_norm = x_norm_interleaved
        x_rm = ttnn.to_layout(x_norm, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_norm)
        host = self.inner.concat_host_output(x_rm)
        ttnn.deallocate(x_rm)
        return host[0, 0, 0, :dim].to(dtype=torch.bfloat16)
