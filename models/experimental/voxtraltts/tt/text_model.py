# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import ttnn
import torch
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import TensorGroup

from models.experimental.voxtraltts.tt.rmsnorm import VoxtralTextRMSNorm
from models.experimental.voxtraltts.tt.text_decoder_layer import remap_voxtral_text_state_dict
from models.experimental.voxtraltts.tt.voxtral_tt_args import (
    get_VoxtralTTArgs,
    voxtral_text_default_optimizations,
)


def patch_text_model_fp32_rms_norms(
    text: "VoxtralTTTextModel",
    *,
    mesh_device,
    state_dict: dict[str, object],
    dim: int,
    norm_eps: float,
) -> None:
    """Swap tt_transformers DistributedNorm/RMSNorm with HF-faithful fp32-promoting norms."""
    remapped = remap_voxtral_text_state_dict(state_dict)
    for layer_idx, layer_block in enumerate(text.inner.layers):
        layer_block.attention_norm = VoxtralTextRMSNorm(
            device=mesh_device,
            dim=dim,
            state_dict=remapped,
            weight_key=f"layers.{layer_idx}.attention_norm",
            eps=norm_eps,
        )
        layer_block.ff_norm = VoxtralTextRMSNorm(
            device=mesh_device,
            dim=dim,
            state_dict=remapped,
            weight_key=f"layers.{layer_idx}.ffn_norm",
            eps=norm_eps,
        )
    text.inner.norm = VoxtralTextRMSNorm(
        device=mesh_device,
        dim=dim,
        state_dict=remapped,
        weight_key="norm",
        eps=norm_eps,
    )


def _decode_activation_dtype(args) -> ttnn.DataType | None:
    return args.decoders_optimizations.get_tensor_dtype(decoder_id=0, tensor=TensorGroup.ACTIVATION)


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
        dtype: ttnn.DataType = ttnn.bfloat16,
        max_batch_size: int = 1,
        max_seq_len: int = 4096,
        optimizations=voxtral_text_default_optimizations,
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
        *,
        collect_layer_hiddens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Prefill via per-token decode (KV-safe; avoids P150 prefill L1 overflow). Returns ``[dim]`` hidden."""
        S = inputs_embeds.shape[0]
        embeds = inputs_embeds.to(dtype=torch.bfloat16)

        last_hidden: torch.Tensor | None = None
        layer_hiddens: dict[str, torch.Tensor] = {}
        for i in range(S):
            is_last = i == S - 1
            if collect_layer_hiddens and is_last:
                last_hidden, layer_hiddens = self.decode_step_from_embeds(
                    embeds[i], start_pos + i, collect_layer_hiddens=True
                )
            else:
                last_hidden = self.decode_step_from_embeds(embeds[i], start_pos + i)

        assert last_hidden is not None
        if collect_layer_hiddens:
            return last_hidden, layer_hiddens
        return last_hidden

    def decode_step_from_embeds(
        self,
        x_embed: torch.Tensor,
        current_pos_idx: int,
        *,
        collect_layer_hiddens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One decode step from a CPU embedding; returns post-norm ``[dim]`` hidden (pre-LM-head)."""
        dim = self.inner.args.dim
        args = self.inner.args
        activation_dtype = _decode_activation_dtype(args) or ttnn.bfloat16
        x_4d = x_embed.reshape(1, 1, 1, dim).to(dtype=torch.bfloat16).contiguous()

        dummy_token = torch.zeros(1, dtype=torch.int64)
        current_pos_t = torch.tensor([current_pos_idx], dtype=torch.int64)
        _, current_pos_tt, rope_idxs, page_table = self.prepare_inputs_decode(dummy_token, current_pos_t)

        rot_mats_global = self.inner.rope_setup.get_rot_mats(rope_idxs)
        rot_mats_local = (
            self.inner.rope_local_setup.get_rot_mats(rope_idxs) if hasattr(self.inner, "rope_local_setup") else None
        )

        decode_mem_cfg = args.get_residual_mem_config(Mode.DECODE, self.inner.prefetcher)
        x_tt = ttnn.from_torch(
            x_4d,
            device=self.inner.mesh_device,
            dtype=activation_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=decode_mem_cfg,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.inner.mesh_device),
        )

        layer_hiddens: dict[str, torch.Tensor] = {}
        for i, layer in enumerate(self.inner.layers):
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
            if collect_layer_hiddens:
                host_layer = self.inner.concat_host_output(x_tt)
                layer_hiddens[f"layer.{i}"] = host_layer[0, 0, 0, :dim].to(dtype=torch.bfloat16)

        lm_norm_cfg = args.get_norm_config("lm_head", Mode.DECODE, self.inner.prefetcher)
        x_norm = self.inner.norm(x_tt, mode=Mode.DECODE, norm_config=lm_norm_cfg)
        ttnn.deallocate(x_tt)
        host = self.inner.concat_host_output(x_norm)
        ttnn.deallocate(x_norm)
        hidden = host[0, 0, 0, :dim].to(dtype=torch.bfloat16)
        if collect_layer_hiddens:
            layer_hiddens["layer.final_norm"] = hidden
            return hidden, layer_hiddens
        return hidden

    def decode_step_from_embeds_tt(
        self,
        x_embed_tt: ttnn.Tensor,
        current_pos_tt: ttnn.Tensor,
        rot_mats_global,
        rot_mats_local,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """One DECODE step for trace replay; returns device hidden (post-norm) without host readback."""
        decode_mem_cfg = self.inner.args.get_residual_mem_config(Mode.DECODE, self.inner.prefetcher)
        activation_dtype = _decode_activation_dtype(self.inner.args)
        if activation_dtype is not None and x_embed_tt.dtype != activation_dtype:
            x_tt = ttnn.to_memory_config(x_embed_tt, decode_mem_cfg, activation_dtype)
        else:
            x_tt = ttnn.to_memory_config(x_embed_tt, decode_mem_cfg)

        for i, layer in enumerate(self.inner.layers):
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

        x_norm = self.inner.norm(
            x_tt,
            mode=Mode.DECODE,
            norm_config=self.inner.args.get_norm_config("lm_head", Mode.DECODE, self.inner.prefetcher),
        )
        ttnn.deallocate(x_tt)
        return x_norm
