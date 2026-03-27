# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Callable, Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding, ScaledEmbedding
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup

from .reference import load_higgs_config, load_higgs_model_state_dict, remap_higgs_state_dict_to_tt

HIGGS_TT_CACHE_ENV = "HIGGS_AUDIO_TT_CACHE"
HIGGS_TT_CACHE_VERSION = "v4"


def resolve_higgs_tt_cache_root(model_name_or_path: str) -> Path:
    override = os.environ.get(HIGGS_TT_CACHE_ENV)
    if override:
        return Path(override)
    sanitized_model_name = model_name_or_path.strip("/").replace("/", "__")
    return Path.home() / ".cache" / "tt-metal-model-cache" / sanitized_model_name / HIGGS_TT_CACHE_VERSION


def resolve_higgs_optimizations(
    optimizations: str | Callable[[ModelArgs], DecodersPrecision] | None,
) -> Callable[[ModelArgs], DecodersPrecision]:
    if optimizations is None:
        optimizations = "accuracy"
    if callable(optimizations):
        return optimizations
    if isinstance(optimizations, str):
        if optimizations == "accuracy":
            return lambda model_args: build_higgs_accuracy_optimizations(model_args.n_layers, model_args.model_name)
        optimization_factory = DecodersPrecision.from_string(optimizations)
        return lambda model_args, factory=optimization_factory: factory(model_args.n_layers, model_args.model_name)
    raise TypeError(f"Unsupported Higgs optimization config: {type(optimizations)!r}")


def build_higgs_accuracy_optimizations(num_decoders: int, model_name: str) -> DecodersPrecision:
    """Higgs audio decode is more sensitive to attention/audio-MLP drift than plain Llama text decode."""
    strict_accuracy = ModelOptimizations(
        {
            "TensorPrecision": {
                TensorGroup.FF1_FF3: PrecisionSetting.BF16,
                TensorGroup.FF2: PrecisionSetting.BF16,
                TensorGroup.WQKV: PrecisionSetting.BF16,
                TensorGroup.WO: PrecisionSetting.BF16,
                TensorGroup.KV_CACHE: PrecisionSetting.BF16,
            },
            "OpFidelity": {
                OpGroup.LI_FF1_FF3: MathFidelitySetting.HIFI4,
                OpGroup.LI_FF2: MathFidelitySetting.HIFI4,
                OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI4,
                OpGroup.LI_QKV_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.SDPA_PREFILL: MathFidelitySetting.HIFI4,
                OpGroup.LI_O_PREFILL: MathFidelitySetting.HIFI4,
            },
        }
    )
    precision = DecodersPrecision(num_decoders, model_name, strict_accuracy)
    precision.__name__ = "accuracy"
    return precision


class HiggsModelArgs(ModelArgs):
    def __init__(
        self,
        mesh_device,
        model_name_or_path: str,
        dummy_weights: bool = False,
        max_batch_size: int = 1,
        max_seq_len: int = 8192,
        optimizations=None,
        cache_hf: bool = False,
        prefetcher=None,
        use_hf_rope: bool = False,
    ):
        self.higgs_model_name_or_path = model_name_or_path
        self.higgs_config = load_higgs_config(model_name_or_path)
        self.audio_vocab_size = self.higgs_config.audio_num_codebooks * (self.higgs_config.audio_codebook_size + 2)

        old_hf_model = os.environ.get("HF_MODEL")
        old_tt_cache = os.environ.get("TT_CACHE_PATH")
        default_tt_cache = resolve_higgs_tt_cache_root(model_name_or_path)
        default_tt_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_MODEL"] = model_name_or_path
        # Higgs checkpoints are large enough that the default tmpfs cache root is fragile on many hosts.
        if old_tt_cache is None:
            os.environ["TT_CACHE_PATH"] = str(default_tt_cache)
        try:
            super().__init__(
                mesh_device=mesh_device,
                instruct=True,
                dummy_weights=dummy_weights,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                optimizations=optimizations,
                cache_hf=cache_hf,
                prefetcher=prefetcher,
                use_hf_rope=use_hf_rope,
            )
        finally:
            if old_hf_model is None:
                os.environ.pop("HF_MODEL", None)
            else:
                os.environ["HF_MODEL"] = old_hf_model
            if old_tt_cache is None:
                os.environ.pop("TT_CACHE_PATH", None)
            else:
                os.environ["TT_CACHE_PATH"] = old_tt_cache
        self.model_name = "Llama-3.2-3B-Instruct"
        self.higgs_tt_cache_root = default_tt_cache if old_tt_cache is None else Path(old_tt_cache)

    def _set_hf_params(self, checkpoint_dir):
        text_config = self.higgs_config.text_config.to_dict()
        self.hf_config = self.higgs_config.text_config
        self.model_name = "Llama-3.2-3B-Instruct"
        self._set_params_from_dict(text_config)
        self.is_multimodal = False

    def load_state_dict(self):
        if self.dummy_weights:
            original_model_name = self.model_name
            self.model_name = "Llama-3.2-3B-Instruct"
            try:
                base_state_dict = super().load_state_dict()
            finally:
                self.model_name = original_model_name
            for layer_idx in range(self.n_layers):
                base_prefix = f"layers.{layer_idx}"
                base_state_dict[f"{base_prefix}.audio_attention_norm.weight"] = base_state_dict[
                    f"{base_prefix}.attention_norm.weight"
                ].clone()
                base_state_dict[f"{base_prefix}.audio_ffn_norm.weight"] = base_state_dict[
                    f"{base_prefix}.ffn_norm.weight"
                ].clone()
                for weight_name in ("w1", "w2", "w3"):
                    base_state_dict[f"{base_prefix}.audio_feed_forward.{weight_name}.weight"] = base_state_dict[
                        f"{base_prefix}.feed_forward.{weight_name}.weight"
                    ].clone()
            base_state_dict["audio_output.weight"] = torch.randn(
                self.audio_vocab_size,
                self.dim,
                dtype=base_state_dict["output.weight"].dtype,
            )
            base_state_dict["audio_codebook_embeddings.weight"] = torch.randn(
                self.higgs_config.audio_num_codebooks * (self.higgs_config.audio_codebook_size + 2),
                self.dim,
                dtype=base_state_dict["output.weight"].dtype,
            )
            return base_state_dict

        raw_state_dict = load_higgs_model_state_dict(self.higgs_model_name_or_path)
        return remap_higgs_state_dict_to_tt(
            raw_state_dict,
            self.n_layers,
            self.audio_vocab_size,
        )


class HiggsDualFFNBlock(LightweightModule):
    def __init__(
        self,
        args: HiggsModelArgs,
        mesh_device,
        tt_ccl: TT_CCL,
        dtype,
        state_dict: dict[str, torch.Tensor],
        layer_num: int,
        weight_cache_path: Path,
        transformation_mats,
        paged_attention_config=None,
    ):
        super().__init__()
        self.args = args
        self.tt_ccl = tt_ccl
        self.mesh_device = mesh_device
        self.layer_num = layer_num
        self.attention = Attention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
        )
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=args.get_model_config(),
        )
        self.audio_feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=args.get_model_config(),
            state_dict_prefix=f"layers.{layer_num}.audio_feed_forward",
        )
        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,
                weight_key=f"layers.{layer_num}.attention_norm",
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            ),
            args,
            tt_ccl=tt_ccl,
            TG=args.is_galaxy,
            ag_config_key="ATTN_LN_AG_CONFIG",
        )
        self.audio_attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,
                weight_key=f"layers.{layer_num}.audio_attention_norm",
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            ),
            args,
            tt_ccl=tt_ccl,
            TG=args.is_galaxy,
            ag_config_key="ATTN_LN_AG_CONFIG",
        )
        self.ffn_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,
                weight_key=f"layers.{layer_num}.ffn_norm",
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            ),
            args,
            tt_ccl=tt_ccl,
            TG=args.is_galaxy,
            ag_config_key="FFN_LN_AG_CONFIG",
        )
        self.audio_ffn_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,
                weight_key=f"layers.{layer_num}.audio_ffn_norm",
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            ),
            args,
            tt_ccl=tt_ccl,
            TG=args.is_galaxy,
            ag_config_key="FFN_LN_AG_CONFIG",
        )

    @staticmethod
    def _select_by_audio_mask(
        audio_token_mask: ttnn.Tensor | None,
        inverse_audio_token_mask: ttnn.Tensor | None,
        text_tensor: ttnn.Tensor,
        audio_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        if audio_token_mask is None:
            ttnn.deallocate(audio_tensor)
            return text_tensor
        if inverse_audio_token_mask is None:
            raise ValueError("Mixed-prefill mask selection requires the inverse audio token mask.")
        # N300 mixed-prefill width-shards the full-width token mask. `ttnn.where` lowers that path to a broadcasted
        # binary kernel that rejects the current subtile pattern, so keep the selection as a same-shape masked blend.
        audio_selected = ttnn.multiply(audio_tensor, audio_token_mask, dtype=ttnn.bfloat16)
        text_selected = ttnn.multiply(text_tensor, inverse_audio_token_mask, dtype=ttnn.bfloat16)
        selected = ttnn.add(text_selected, audio_selected, dtype=ttnn.bfloat16)
        ttnn.deallocate(audio_selected)
        ttnn.deallocate(text_selected)
        ttnn.deallocate(text_tensor)
        ttnn.deallocate(audio_tensor)
        return selected

    def _prepare_mixed_prefill_ffn_outputs(
        self,
        ff_text: ttnn.Tensor,
        ff_audio: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return ff_text, ff_audio

    def _forward_branch(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global,
        rot_mats_local,
        mode: Mode,
        use_audio_ffn: bool,
        audio_token_mask: ttnn.Tensor | None = None,
        inverse_audio_token_mask: ttnn.Tensor | None = None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
    ) -> ttnn.Tensor:
        TG = self.args.is_galaxy
        skip_mem_cfg = self.args.get_residual_mem_config(mode, None)
        mixed_prefill = mode == Mode.PREFILL and audio_token_mask is not None
        if mixed_prefill:
            residual = x
            attn_norm_config = self.args.get_norm_config("attn", mode, None)
            attn_text = self.attention_norm(x, mode, norm_config=attn_norm_config)
            attn_audio = self.audio_attention_norm(x, mode, norm_config=attn_norm_config)
            attn_in = self._select_by_audio_mask(audio_token_mask, inverse_audio_token_mask, attn_text, attn_audio)
            attn_out = self.attention.forward(
                attn_in,
                current_pos,
                rot_mats_local
                if (hasattr(self.attention, "is_sliding") and self.attention.is_sliding)
                else rot_mats_global,
                0,
                mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=None,
            )
            if attn_out.memory_config() != skip_mem_cfg:
                attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
            hidden_states = ttnn.add(
                residual, attn_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None
            )
            ttnn.deallocate(attn_out)

            residual = hidden_states
            ff_norm_config = self.args.get_norm_config("ff", mode, None)
            ff_text = self.ffn_norm(hidden_states, mode, norm_config=ff_norm_config)
            ff_audio = self.audio_ffn_norm(hidden_states, mode, norm_config=ff_norm_config)
            if TG and mode == Mode.DECODE:
                ff_text = ttnn.to_memory_config(ff_text, memory_config=self.args.get_mlp_act_mem_config(mode))
                ff_audio = ttnn.to_memory_config(ff_audio, memory_config=self.args.get_mlp_act_mem_config(mode))
            ff_text = self.feed_forward.forward(ff_text, mode)
            ff_audio = self.audio_feed_forward.forward(ff_audio, mode)
            ff_text, ff_audio = self._prepare_mixed_prefill_ffn_outputs(ff_text, ff_audio)
            hidden_states = self._select_by_audio_mask(audio_token_mask, inverse_audio_token_mask, ff_text, ff_audio)
            activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=self.layer_num,
                tensor=TensorGroup.ACTIVATION,
            )
            return ttnn.add(
                residual,
                hidden_states,
                memory_config=skip_mem_cfg,
                dtype=self.args.ccl_dtype
                if TG and not self.args.is_distributed_norm(mode)
                else activation_dtype or ttnn.bfloat16,
            )

        residual = x
        norm_module = self.audio_attention_norm if use_audio_ffn else self.attention_norm
        ff_norm_module = self.audio_ffn_norm if use_audio_ffn else self.ffn_norm
        mlp_module = self.audio_feed_forward if use_audio_ffn else self.feed_forward
        attn_norm_config = self.args.get_norm_config("attn", mode, None)
        attn_in = norm_module(x, mode, norm_config=attn_norm_config)
        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mats_local
            if (hasattr(self.attention, "is_sliding") and self.attention.is_sliding)
            else rot_mats_global,
            0,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=None,
        )
        if attn_out.memory_config() != skip_mem_cfg:
            attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
        hidden_states = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None)
        ttnn.deallocate(attn_out)
        residual = hidden_states
        ff_norm_config = self.args.get_norm_config("ff", mode, None)
        hidden_states = ff_norm_module(hidden_states, mode, norm_config=ff_norm_config)
        if TG and mode == Mode.DECODE:
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=self.args.get_mlp_act_mem_config(mode))
        hidden_states = mlp_module.forward(hidden_states, mode)
        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num,
            tensor=TensorGroup.ACTIVATION,
        )
        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )
        return out

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global,
        rot_mats_local,
        mode: Mode,
        is_audio_token: bool,
        audio_token_mask: ttnn.Tensor | None = None,
        inverse_audio_token_mask: ttnn.Tensor | None = None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
    ) -> ttnn.Tensor:
        return self._forward_branch(
            x=x,
            current_pos=current_pos,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            mode=mode,
            use_audio_ffn=is_audio_token,
            audio_token_mask=audio_token_mask,
            inverse_audio_token_mask=inverse_audio_token_mask,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )


class HiggsAudioTTModel(LightweightModule):
    def __init__(
        self,
        args: HiggsModelArgs,
        mesh_device,
        state_dict: dict[str, torch.Tensor],
        dtype=ttnn.bfloat8_b,
        paged_attention_config=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.state_dict = state_dict
        self.config = args.higgs_config
        self.dtype = dtype
        self.head_dtype = ttnn.bfloat16 if args.decoders_optimizations.__name__ == "accuracy" else dtype
        self.args.lm_head_dtype = self.head_dtype
        self.tt_ccl = TT_CCL(mesh_device)
        self.weight_cache_path = args.weight_cache_path(dtype)
        self.paged_attention_config = paged_attention_config
        self.last_prefill_used_fallback = False

        default_rope_setup = HfRotarySetup if args.use_hf_rope else RotarySetup
        self.rope_setup = default_rope_setup(
            device=mesh_device,
            batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            max_seq_len=args.max_seq_len,
            rope_theta=args.rope_theta,
            rope_scaling=args.rope_scaling,
            use_qk_fused=args.use_qk_fused,
            prefetcher=None,
        )
        self.rope_local_setup = None
        self.transformation_mats = self.rope_setup.get_both_trans_mats()
        self.cached_decode_rot_mats = (
            (self.rope_setup.get_rot_mats(torch.tensor([0], dtype=torch.int64)), None) if args.use_hf_rope else None
        )
        self.layers = [
            HiggsDualFFNBlock(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                layer_num=layer_idx,
                weight_cache_path=self.weight_cache_path,
                transformation_mats=self.transformation_mats,
                paged_attention_config=paged_attention_config,
            )
            for layer_idx in range(args.n_layers)
        ]
        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,
                weight_key="norm",
                weight_cache_path=None if args.dummy_weights else self.weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            TG=args.is_galaxy,
        )
        self.text_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            dtype=self.head_dtype,
            state_dict=state_dict,
            state_dict_prefix="",
            weight_cache_path=self.weight_cache_path,
            max_columns_per_device=args.max_columns_per_device_lm_head,
        )
        audio_head_args = copy.copy(args)
        audio_head_args.vocab_size = args.audio_vocab_size
        # Keep the auxiliary audio head tile-aligned so LMHead does not create a trailing non-tile shard.
        audio_head_args.padded_vocab_size = ((args.audio_vocab_size + 31) // 32) * 32
        audio_state_dict = {"output.weight": state_dict["audio_output.weight"]}
        self.audio_head = LMHead(
            args=audio_head_args,
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            dtype=self.head_dtype,
            state_dict=audio_state_dict,
            state_dict_prefix="",
            weight_cache_path=self.weight_cache_path / "audio_head",
            max_columns_per_device=audio_head_args.padded_vocab_size,
        )
        embd_kwargs = {
            "mesh_device": mesh_device,
            "args": args,
            "weight_cache_path": self.weight_cache_path,
            "state_dict": state_dict,
            "dtype": ttnn.bfloat16,
        }
        if self.args.embed_scale is not None:
            embd_cls = ScaledEmbedding
            embd_kwargs["embed_scale"] = self.args.embed_scale
        else:
            embd_cls = Embedding
        self.text_embedding_tt = embd_cls(**embd_kwargs)
        self.text_embedding = torch.nn.Embedding.from_pretrained(state_dict["tok_embeddings.weight"], freeze=True)
        self.audio_embedding = torch.nn.Embedding.from_pretrained(
            state_dict["audio_codebook_embeddings.weight"], freeze=True
        )
        self.audio_embedding_tt_weights = None
        self.audio_codebook_shift_tt = None
        self.audio_delay_postprocess_tensors = None
        self.audio_codebook_size = self.config.audio_codebook_size + 2
        self.audio_num_codebooks = self.config.audio_num_codebooks
        self.audio_embed_avg = self.config.audio_embed_avg
        self.text_decode_warmed_up = False
        self.audio_codebook_shift_host = (
            torch.arange(self.audio_num_codebooks, dtype=torch.int32).view(1, 1, self.audio_num_codebooks)
            * self.audio_codebook_size
        )
        self.audio_embedding_seq_len = max(ttnn.TILE_SIZE, self.audio_num_codebooks)

    def _torch_embed_audio_tokens(self, audio_ids: torch.Tensor) -> torch.Tensor:
        codebook_shift = torch.arange(self.audio_num_codebooks, device=audio_ids.device) * self.audio_codebook_size
        embeddings = self.audio_embedding(audio_ids + codebook_shift.unsqueeze(-1))
        if self.audio_embed_avg:
            return embeddings.mean(dim=0)
        return embeddings.sum(dim=0)

    def embed_audio_tokens(self, audio_ids: torch.Tensor) -> torch.Tensor:
        return self._torch_embed_audio_tokens(audio_ids)

    @staticmethod
    def _bucket_prefill_seq_len(effective_seq_len: int, has_audio_tokens: bool) -> int:
        if has_audio_tokens:
            if effective_seq_len <= 512:
                return 512
            if effective_seq_len <= 768:
                return 768
        else:
            if effective_seq_len <= 128:
                return 128
            if effective_seq_len <= 256:
                return 256
        return ((effective_seq_len + 255) // 256) * 256

    @classmethod
    def _bucket_prefill_inputs(
        cls,
        merged_embeddings: torch.Tensor,
        audio_token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
        effective_seq_len = merged_embeddings.shape[1]
        has_audio_tokens = audio_token_mask is not None and bool(audio_token_mask.any())
        padded_seq_len = cls._bucket_prefill_seq_len(effective_seq_len, has_audio_tokens)
        pad_tokens = padded_seq_len - effective_seq_len
        if pad_tokens <= 0:
            return merged_embeddings, audio_token_mask, effective_seq_len
        padded_embeddings = torch.nn.functional.pad(merged_embeddings, (0, 0, 0, pad_tokens))
        if audio_token_mask is None:
            padded_audio_mask = None
        else:
            padded_audio_mask = torch.nn.functional.pad(audio_token_mask, (0, pad_tokens), value=False)
        return padded_embeddings, padded_audio_mask, effective_seq_len

    @classmethod
    def _bucket_prefill_token_inputs(
        cls,
        merged_text_ids: torch.Tensor,
        merged_audio_ids: torch.Tensor | None,
        audio_token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Optional[torch.Tensor], int]:
        effective_seq_len = merged_text_ids.shape[1]
        has_audio_tokens = audio_token_mask is not None and bool(audio_token_mask.any())
        padded_seq_len = cls._bucket_prefill_seq_len(effective_seq_len, has_audio_tokens)
        pad_tokens = padded_seq_len - effective_seq_len
        if pad_tokens <= 0:
            return merged_text_ids, merged_audio_ids, audio_token_mask, effective_seq_len

        padded_text_ids = torch.nn.functional.pad(merged_text_ids, (0, pad_tokens), value=0)
        if merged_audio_ids is None:
            padded_audio_ids = None
        else:
            padded_audio_ids = torch.nn.functional.pad(merged_audio_ids, (0, 0, 0, pad_tokens), value=0)
        if audio_token_mask is None:
            padded_audio_mask = None
        else:
            padded_audio_mask = torch.nn.functional.pad(audio_token_mask, (0, pad_tokens), value=False)
        return padded_text_ids, padded_audio_ids, padded_audio_mask, effective_seq_len

    def _expand_audio_token_mask(
        self,
        audio_token_mask: Optional[torch.Tensor],
        seq_len: int,
        invert: bool = False,
    ) -> torch.Tensor | None:
        if audio_token_mask is None:
            return None
        prompt_audio_mask = audio_token_mask[0] if audio_token_mask.dim() == 2 else audio_token_mask
        if invert:
            prompt_audio_mask = ~prompt_audio_mask
        prompt_audio_mask = prompt_audio_mask[:seq_len].view(1, 1, seq_len, 1)
        return prompt_audio_mask.expand(1, 1, seq_len, self.args.dim).to(torch.bfloat16).contiguous()

    def _to_device_prefill_audio_masks(
        self,
        audio_token_mask: Optional[torch.Tensor],
        seq_len: int,
    ) -> tuple[ttnn.Tensor | None, ttnn.Tensor | None]:
        expanded_mask = self._expand_audio_token_mask(audio_token_mask, seq_len)
        if expanded_mask is None:
            return None, None
        expanded_inverse_mask = self._expand_audio_token_mask(audio_token_mask, seq_len, invert=True)
        # Mixed-prefill branch selection multiplies against full-width interleaved DRAM activations on every device.
        # Replicate the mask so each device sees the full width; width-sharding it to 1536 on N300 triggers the
        # unsupported subtile-broadcast path in binary_ng during the per-branch masked blend.
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        tt_audio_token_mask = ttnn.from_torch(
            expanded_mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )
        tt_inverse_audio_token_mask = ttnn.from_torch(
            expanded_inverse_mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )
        return tt_audio_token_mask, tt_inverse_audio_token_mask

    def prepare_prefill_embeddings_host(self, merged_embeddings: torch.Tensor) -> ttnn.Tensor:
        if merged_embeddings.dim() == 3:
            merged_embeddings = merged_embeddings.unsqueeze(1)
        return ttnn.from_torch(
            merged_embeddings,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        )

    def prepare_prefill_audio_mask_host(
        self,
        audio_token_mask: Optional[torch.Tensor],
        seq_len: int,
        invert: bool = False,
    ) -> ttnn.Tensor | None:
        expanded_mask = self._expand_audio_token_mask(audio_token_mask, seq_len, invert=invert)
        if expanded_mask is None:
            return None
        return ttnn.from_torch(
            expanded_mask,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def prepare_prefill_inputs_trace(
        self,
        input_ids: torch.Tensor,
        audio_in_ids: Optional[torch.Tensor] = None,
        audio_in_ids_start: Optional[torch.Tensor] = None,
        audio_out_ids: Optional[torch.Tensor] = None,
        audio_out_ids_start: Optional[torch.Tensor] = None,
        page_table: torch.Tensor | None = None,
        chunk_page_table: torch.Tensor | None = None,
    ) -> dict:
        audio_in_segment_count = 0 if audio_in_ids_start is None else int(audio_in_ids_start.numel())
        audio_out_segment_count = 0 if audio_out_ids_start is None else int(audio_out_ids_start.numel())
        total_audio_segment_count = audio_in_segment_count + audio_out_segment_count
        if total_audio_segment_count == 0:
            prompt_family = "text_only"
        elif total_audio_segment_count == 1:
            prompt_family = "single_audio_segment"
        else:
            prompt_family = "multi_audio_segment"
        merged_text_ids, merged_audio_ids, audio_token_mask = self._build_prompt_token_inputs(
            input_ids=input_ids,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
        )
        merged_text_ids, merged_audio_ids, audio_token_mask, effective_seq_len = self._bucket_prefill_token_inputs(
            merged_text_ids,
            merged_audio_ids,
            audio_token_mask,
        )
        has_audio_tokens = bool(audio_token_mask.any())
        text_tokens = merged_text_ids.reshape(1, 1, 1, -1).to(torch.int32)
        tt_text_tokens = ttnn.from_torch(
            text_tokens,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        tt_audio_tokens = None
        if has_audio_tokens and merged_audio_ids is not None:
            audio_tokens = merged_audio_ids.to(torch.int32) + self.audio_codebook_shift_host
            if self.audio_embedding_seq_len > self.audio_num_codebooks:
                audio_tokens = torch.nn.functional.pad(
                    audio_tokens,
                    (0, self.audio_embedding_seq_len - self.audio_num_codebooks),
                    value=0,
                )
            tt_audio_tokens = ttnn.from_torch(
                audio_tokens,
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        seq_len = merged_text_ids.shape[1]
        if has_audio_tokens:
            tt_audio_token_mask = self.prepare_prefill_audio_mask_host(audio_token_mask, seq_len)
            tt_inverse_audio_token_mask = self.prepare_prefill_audio_mask_host(audio_token_mask, seq_len, invert=True)
        else:
            tt_audio_token_mask = None
            tt_inverse_audio_token_mask = None
        tt_page_table = (
            ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            if page_table is not None
            else None
        )
        tt_chunk_page_table = (
            ttnn.from_torch(
                chunk_page_table,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            if chunk_page_table is not None
            else None
        )
        return {
            "host_inputs": (
                tt_text_tokens,
                tt_audio_tokens,
                tt_audio_token_mask,
                tt_inverse_audio_token_mask,
                tt_page_table,
                tt_chunk_page_table,
            ),
            "effective_seq_len": effective_seq_len,
            "padded_seq_len": seq_len,
            "has_audio_tokens": has_audio_tokens,
            "prompt_family": prompt_family,
        }

    @staticmethod
    def _split_audio_segments(
        audio_ids: Optional[torch.Tensor], audio_ids_start: Optional[torch.Tensor]
    ) -> list[torch.Tensor]:
        if audio_ids is None or audio_ids_start is None or audio_ids_start.numel() == 0:
            return []
        segments = []
        starts = audio_ids_start.tolist()
        for idx, start in enumerate(starts):
            end = starts[idx + 1] if idx + 1 < len(starts) else audio_ids.shape[1]
            segments.append(audio_ids[:, start:end])
        return segments

    @staticmethod
    def _pad_text_only_prefill_inputs(
        merged_embeddings: torch.Tensor,
        audio_token_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
        effective_seq_len = merged_embeddings.shape[1]
        if audio_token_mask is not None and bool(audio_token_mask.any()):
            return merged_embeddings, audio_token_mask, effective_seq_len
        pad_tokens = (-effective_seq_len) % 128
        if pad_tokens == 0:
            return merged_embeddings, audio_token_mask, effective_seq_len
        # Text-only prompts can be padded safely because causal prefill never lets earlier real tokens attend to the
        # padded suffix, and decode starts from the original prompt length so the bogus padded cache rows are ignored.
        padded_embeddings = torch.nn.functional.pad(merged_embeddings, (0, 0, 0, pad_tokens))
        if audio_token_mask is None:
            padded_audio_mask = None
        else:
            padded_audio_mask = torch.nn.functional.pad(audio_token_mask, (0, pad_tokens), value=False)
        return padded_embeddings, padded_audio_mask, effective_seq_len

    def _build_prompt_token_inputs(
        self,
        input_ids: torch.Tensor,
        audio_in_ids: Optional[torch.Tensor] = None,
        audio_in_ids_start: Optional[torch.Tensor] = None,
        audio_out_ids: Optional[torch.Tensor] = None,
        audio_out_ids_start: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError("HiggsAudioTTModel only supports batch size 1 for prompt token preparation.")
            tokens = input_ids[0]
        else:
            tokens = input_ids

        audio_in_segments = self._split_audio_segments(audio_in_ids, audio_in_ids_start)
        audio_out_segments = self._split_audio_segments(audio_out_ids, audio_out_ids_start)
        if not audio_in_segments and not audio_out_segments:
            return tokens.unsqueeze(0).long(), None, torch.zeros((1, tokens.shape[0]), dtype=torch.bool)

        zero_audio_row = torch.zeros((self.audio_num_codebooks,), dtype=torch.long)
        merged_text_ids: list[int] = []
        merged_audio_rows: list[torch.Tensor] = []
        audio_token_mask: list[bool] = []
        audio_in_index = 0
        audio_out_index = 0

        for token_id in tokens.tolist():
            if token_id == self.config.audio_in_token_idx:
                if audio_in_index >= len(audio_in_segments):
                    raise ValueError("Missing encoded audio-in segment for prompt placeholder.")
                segment = audio_in_segments[audio_in_index]
                audio_in_index += 1
                for frame_idx in range(segment.shape[1]):
                    merged_text_ids.append(0)
                    merged_audio_rows.append(segment[:, frame_idx].long())
                    audio_token_mask.append(True)
                continue

            if token_id == self.config.audio_out_token_idx:
                if audio_out_index >= len(audio_out_segments):
                    raise ValueError("Missing encoded audio-out segment for prompt placeholder.")
                segment = audio_out_segments[audio_out_index]
                audio_out_index += 1
                for frame_idx in range(segment.shape[1]):
                    merged_text_ids.append(0)
                    merged_audio_rows.append(segment[:, frame_idx].long())
                    audio_token_mask.append(True)
                continue

            merged_text_ids.append(token_id)
            merged_audio_rows.append(zero_audio_row)
            audio_token_mask.append(False)

        if audio_in_index != len(audio_in_segments) or audio_out_index != len(audio_out_segments):
            raise ValueError("Unused encoded audio segments remain after prompt token merge.")

        merged_text_ids_tensor = torch.tensor(merged_text_ids, dtype=torch.long).unsqueeze(0)
        merged_audio_ids_tensor = torch.stack(merged_audio_rows, dim=0).unsqueeze(0).long()
        audio_token_mask_tensor = torch.tensor(audio_token_mask, dtype=torch.bool).unsqueeze(0)
        return merged_text_ids_tensor, merged_audio_ids_tensor, audio_token_mask_tensor

    def embed_prompt_inputs(
        self,
        input_ids: torch.Tensor,
        audio_in_ids: Optional[torch.Tensor] = None,
        audio_in_ids_start: Optional[torch.Tensor] = None,
        audio_out_ids: Optional[torch.Tensor] = None,
        audio_out_ids_start: Optional[torch.Tensor] = None,
    ):
        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError("HiggsAudioTTModel only supports batch size 1 for prompt embedding.")
            tokens = input_ids[0]
        else:
            tokens = input_ids

        audio_in_segments = self._split_audio_segments(audio_in_ids, audio_in_ids_start)
        audio_out_segments = self._split_audio_segments(audio_out_ids, audio_out_ids_start)
        if not audio_in_segments and not audio_out_segments:
            text_embeddings = self.text_embedding(tokens.unsqueeze(0) if tokens.dim() == 1 else tokens)
            audio_mask = torch.zeros((1, text_embeddings.shape[1]), dtype=torch.bool)
            return text_embeddings, audio_mask

        text_embeddings = self.text_embedding(tokens)
        merged_embeddings = []
        audio_token_mask = []
        audio_in_index = 0
        audio_out_index = 0

        for token_index, token_id in enumerate(tokens.tolist()):
            if token_id == self.config.audio_in_token_idx:
                if audio_in_index >= len(audio_in_segments):
                    raise ValueError("Missing encoded audio-in segment for prompt placeholder.")
                segment_embedding = self._torch_embed_audio_tokens(audio_in_segments[audio_in_index])
                audio_in_index += 1
                merged_embeddings.append(segment_embedding)
                audio_token_mask.extend([True] * segment_embedding.shape[0])
            elif token_id == self.config.audio_out_token_idx:
                if audio_out_index >= len(audio_out_segments):
                    raise ValueError("Missing encoded audio-out segment for prompt placeholder.")
                segment_embedding = self._torch_embed_audio_tokens(audio_out_segments[audio_out_index])
                audio_out_index += 1
                merged_embeddings.append(segment_embedding)
                audio_token_mask.extend([True] * segment_embedding.shape[0])
            else:
                merged_embeddings.append(text_embeddings[token_index : token_index + 1])
                audio_token_mask.append(False)

        if audio_in_index != len(audio_in_segments) or audio_out_index != len(audio_out_segments):
            raise ValueError("Unused encoded audio segments remain after prompt embedding merge.")

        return torch.cat(merged_embeddings, dim=0).unsqueeze(0), torch.tensor(
            audio_token_mask, dtype=torch.bool
        ).unsqueeze(0)

    def _to_device_embeddings(self, embeddings: torch.Tensor, mode: Mode) -> ttnn.Tensor:
        if mode == Mode.DECODE:
            # Line-mesh decode attention expects the same tile-padded residual geometry as the TT model-args helper.
            # A direct width-sharded upload of a logical batch-1 embedding can wedge the first 1x2 QKV matmul even
            # though standalone Attention1D decode tests use the padded helper path successfully.
            return self.args.prepare_residual_tensor_decode(
                embeddings,
                self.args.get_residual_mem_config(mode, None),
                on_host=False,
            )
        if embeddings.dim() == 3:
            embeddings = embeddings.unsqueeze(1)
        return ttnn.from_torch(
            embeddings,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.args.get_residual_mem_config(mode, None),
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        )

    def _slice_prefill_rot_mats(self, seq_len: int):
        cos_slice = self.rope_setup.cos_matrix_prefill[:, :, :seq_len, :]
        sin_slice = self.rope_setup.sin_matrix_prefill[:, :, :seq_len, :]
        return [cos_slice, sin_slice], None

    def _decode_rot_mats(self, current_pos: int):
        if self.cached_decode_rot_mats is not None:
            return self.cached_decode_rot_mats
        pos = self._decode_rope_positions_host(current_pos)
        return self.rope_setup.get_rot_mats(pos), None

    def _decode_positions_host(self, current_pos: int, inactive_value: int) -> torch.Tensor:
        positions = torch.full((self.args.max_batch_size,), inactive_value, dtype=torch.int32)
        positions[0] = current_pos
        return positions

    def _decode_rope_positions_host(self, current_pos: int) -> torch.Tensor:
        return self._decode_positions_host(current_pos, inactive_value=0).to(torch.int64)

    def create_current_pos_tensor(self, current_pos: int):
        return ttnn.from_torch(
            self._decode_positions_host(current_pos, inactive_value=-1),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def increment_current_pos_tensor(self, current_pos_tt: ttnn.Tensor):
        ttnn.plus_one(current_pos_tt, skip_negative_entries=True)

    def create_rot_mat_idx_tensor(self, current_pos: int):
        return self.rope_setup.get_rot_idxs(self._decode_rope_positions_host(current_pos))

    def increment_rot_mat_idx_tensor(self, rot_mat_idxs: ttnn.Tensor):
        ttnn.plus_one(rot_mat_idxs)

    def reset_kv_cache(self):
        for layer in self.layers:
            attention = getattr(layer, "attention", None)
            if attention is None or not hasattr(attention, "layer_past"):
                continue
            if not attention.layer_past:
                attention.init_kv_cache(self.args, self.weight_cache_path)
                continue
            # Traced perf keeps capture state alive across cases, so tearing down and reallocating KV buffers between
            # benchmark cases is trace-unsafe. Clear the existing cache tensors in place and preserve their addresses.
            for tensor in attention.layer_past:
                ttnn.fill(
                    tensor,
                    0,
                    memory_config=ttnn.get_memory_config(tensor),
                    output_tensor=tensor,
                )

    def prepare_audio_decode_embedding_host(self, current_embedding: torch.Tensor):
        embedding = current_embedding.view(1, 1, 1, -1)
        tt_embedding = ttnn.from_torch(
            embedding,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        )
        return tt_embedding

    def prepare_audio_decode_tokens_host(self, current_audio_tokens: torch.Tensor):
        # Trace replay keeps a fixed token-id buffer shape, so pad the codebook ids to one tile.
        tokens = current_audio_tokens.reshape(1, 1, 1, self.audio_num_codebooks).to(torch.int32)
        tokens = tokens + self.audio_codebook_shift_host
        if self.audio_embedding_seq_len > self.audio_num_codebooks:
            tokens = torch.nn.functional.pad(
                tokens, (0, self.audio_embedding_seq_len - self.audio_num_codebooks), value=0
            )
        return ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def prepare_audio_decode_raw_tokens_host(self, current_audio_tokens: torch.Tensor):
        tokens = current_audio_tokens.reshape(1, 1, 1, self.audio_num_codebooks).to(torch.int32)
        return ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def prepare_audio_decode_position_inputs_host(self, current_pos: int):
        current_pos_tt = ttnn.from_torch(
            self._decode_positions_host(current_pos, inactive_value=-1),
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        rot_mat_idxs = self.rope_setup.get_rot_idxs(self._decode_rope_positions_host(current_pos), on_host=True)
        return current_pos_tt, rot_mat_idxs

    def prepare_audio_decode_inputs_host(self, current_embedding: torch.Tensor, current_pos: int):
        return (
            self.prepare_audio_decode_embedding_host(current_embedding),
            *self.prepare_audio_decode_position_inputs_host(current_pos),
        )

    def prepare_audio_decode_token_inputs_host(self, current_audio_tokens: torch.Tensor, current_pos: int):
        return (
            self.prepare_audio_decode_tokens_host(current_audio_tokens),
            *self.prepare_audio_decode_position_inputs_host(current_pos),
        )

    def prepare_audio_decode_raw_token_inputs_host(self, current_audio_tokens: torch.Tensor, current_pos: int):
        return (
            self.prepare_audio_decode_raw_tokens_host(current_audio_tokens),
            *self.prepare_audio_decode_position_inputs_host(current_pos),
        )

    def prepare_audio_decode_bootstrap_inputs_host(self, current_pos: int):
        initial_audio_tokens = torch.full(
            (self.audio_num_codebooks, 1),
            self.config.audio_stream_bos_id,
            dtype=torch.long,
        )
        return (
            *self.prepare_audio_decode_raw_token_inputs_host(initial_audio_tokens, current_pos=current_pos),
            *self.prepare_audio_decode_delay_state_host(num_delay=0, num_remaining_delays=None),
            self.prepare_audio_decode_finished_state_host(False),
        )

    def _ensure_audio_embedding_tt_weights(self):
        if self.audio_embedding_tt_weights is not None:
            return self.audio_embedding_tt_weights
        audio_embedding_weights = self.state_dict["audio_codebook_embeddings.weight"].unsqueeze(0).unsqueeze(0)
        self.audio_embedding_tt_weights = ttnn.as_tensor(
            audio_embedding_weights,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device,
                dims=(None, 3),
                mesh_shape=self.args.cluster_shape,
            ),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=self.args.get_model_config()["EMB_WEIGHTS_MEMCFG"],
            cache_file_name=None
            if self.args.dummy_weights
            else self.weight_cache_path / "audio_codebook_embeddings.weight",
        )
        return self.audio_embedding_tt_weights

    def _ensure_audio_codebook_shift_tt(self):
        if self.audio_codebook_shift_tt is not None:
            return self.audio_codebook_shift_tt
        self.audio_codebook_shift_tt = ttnn.as_tensor(
            self.audio_codebook_shift_host,
            dtype=ttnn.uint32,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=None,
        )
        return self.audio_codebook_shift_tt

    def _ensure_audio_delay_postprocess_tensors(self):
        if self.audio_delay_postprocess_tensors is not None:
            return self.audio_delay_postprocess_tensors

        pad_len = self.audio_embedding_seq_len - self.audio_num_codebooks
        codebook_indices = torch.arange(self.audio_embedding_seq_len, dtype=torch.float32).view(1, 1, 1, -1)
        codebook_shift_uint = torch.zeros((1, 1, 1, self.audio_embedding_seq_len), dtype=torch.int32)
        codebook_shift_uint[0, 0, 0, : self.audio_num_codebooks] = self.audio_codebook_shift_host.reshape(-1)
        bos_fill_uint = torch.zeros((1, 1, 1, self.audio_embedding_seq_len), dtype=torch.int32)
        bos_fill_uint[0, 0, 0, : self.audio_num_codebooks] = int(self.config.audio_stream_bos_id)
        eos_fill_uint = torch.zeros((1, 1, 1, self.audio_embedding_seq_len), dtype=torch.int32)
        eos_fill_uint[0, 0, 0, : self.audio_num_codebooks] = int(self.config.audio_stream_eos_id)
        eos_compare_uint = torch.full(
            (1, 1, 1, self.audio_embedding_seq_len),
            fill_value=int(self.config.audio_stream_eos_id) + 1,
            dtype=torch.int32,
        )
        eos_compare_uint[0, 0, 0, : self.audio_num_codebooks] = int(self.config.audio_stream_eos_id)
        zero_fill = torch.zeros((1, 1, 1, self.audio_embedding_seq_len), dtype=torch.float32)
        one_fill = torch.ones((1, 1, 1, self.audio_embedding_seq_len), dtype=torch.float32)

        scalar = lambda value: torch.tensor([[[[value]]]], dtype=torch.float32)
        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        device_tensor = lambda tensor: ttnn.from_torch(
            tensor,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        device_uint_tensor = lambda tensor: ttnn.from_torch(
            tensor,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.audio_delay_postprocess_tensors = {
            "codebook_indices": device_tensor(codebook_indices),
            "codebook_shift_uint": device_uint_tensor(codebook_shift_uint),
            "bos_fill_uint": device_uint_tensor(bos_fill_uint),
            "eos_fill_uint": device_uint_tensor(eos_fill_uint),
            "eos_compare_uint": device_uint_tensor(eos_compare_uint),
            "zero_fill": device_tensor(zero_fill),
            "one_fill": device_tensor(one_fill),
            "last_codebook_idx": device_tensor(scalar(float(self.audio_num_codebooks - 1))),
            "num_codebooks": device_tensor(scalar(float(self.audio_num_codebooks))),
            "zero_scalar": device_tensor(scalar(0.0)),
            "one_scalar": device_tensor(scalar(1.0)),
            "none_scalar": device_tensor(scalar(-1.0)),
            "pad_len": pad_len,
        }
        return self.audio_delay_postprocess_tensors

    def prime_trace_runtime_assets(self) -> None:
        # Trace replay assumes the participating tensor addresses stay stable. Prime every lazily-created TT tensor
        # used by traced prefill/decode before any trace exists so later replays do not trigger allocator warnings.
        self._ensure_audio_embedding_tt_weights()
        self._ensure_audio_codebook_shift_tt()
        self._ensure_audio_delay_postprocess_tensors()

    def create_audio_decode_delay_state_tensors(self, num_delay: int, num_remaining_delays: int | None):
        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        scalar_tensor = lambda value: ttnn.from_torch(
            torch.tensor([[[[value]]]], dtype=torch.float32),
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return scalar_tensor(float(num_delay)), scalar_tensor(
            -1.0 if num_remaining_delays is None else float(num_remaining_delays)
        )

    def prepare_audio_decode_delay_state_host(self, num_delay: int, num_remaining_delays: int | None):
        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        scalar_tensor = lambda value: ttnn.from_torch(
            torch.tensor([[[[value]]]], dtype=torch.float32),
            device=None,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )
        return scalar_tensor(float(num_delay)), scalar_tensor(
            -1.0 if num_remaining_delays is None else float(num_remaining_delays)
        )

    def prepare_audio_decode_finished_state_host(self, finished: bool = False):
        rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return ttnn.from_torch(
            torch.tensor([[[[1.0 if finished else 0.0]]]], dtype=torch.float32),
            device=None,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )

    def audio_ttnn_initialize_decode_state_inplace(
        self,
        tt_bootstrap_raw_audio_tokens: ttnn.Tensor,
        tt_bootstrap_current_pos: ttnn.Tensor,
        tt_bootstrap_rot_mat_idxs: ttnn.Tensor,
        tt_bootstrap_num_delay: ttnn.Tensor,
        tt_bootstrap_num_remaining_delays: ttnn.Tensor,
        tt_bootstrap_finished: ttnn.Tensor,
        tt_decode_raw_audio_tokens: ttnn.Tensor,
        tt_decode_current_pos: ttnn.Tensor,
        tt_decode_rot_mat_idxs: ttnn.Tensor,
        tt_decode_num_delay: ttnn.Tensor,
        tt_decode_num_remaining_delays: ttnn.Tensor,
        tt_decode_finished: ttnn.Tensor,
    ) -> ttnn.Tensor:
        # Prefill trace reuses the persistent decode buffers captured in the decode trace. Seed every decode-state tensor
        # on device so the traced performance path has no host bootstrap boundary after prefill completes.
        ttnn.copy(tt_bootstrap_raw_audio_tokens, tt_decode_raw_audio_tokens)
        ttnn.copy(tt_bootstrap_current_pos, tt_decode_current_pos)
        ttnn.copy(tt_bootstrap_rot_mat_idxs, tt_decode_rot_mat_idxs)
        ttnn.copy(tt_bootstrap_num_delay, tt_decode_num_delay)
        ttnn.copy(tt_bootstrap_num_remaining_delays, tt_decode_num_remaining_delays)
        ttnn.copy(tt_bootstrap_finished, tt_decode_finished)
        return tt_decode_finished

    def audio_ttnn_delay_pattern_postprocess_inplace(
        self,
        tt_raw_audio_tokens: ttnn.Tensor,
        tt_next_input_tokens: ttnn.Tensor,
        tt_num_delay: ttnn.Tensor,
        tt_num_remaining_delays: ttnn.Tensor,
        shift_output_tokens: bool = True,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        constants = self._ensure_audio_delay_postprocess_tensors()

        # Keep the raw argmax ids in uint32. `typecast` would reinterpret bits rather than numerically cast them,
        # which corrupts BOS/EOS comparisons and makes the device delay state terminate immediately.
        tt_tokens_rm = ttnn.reshape(tt_raw_audio_tokens, [1, 1, 1, self.audio_num_codebooks])
        if constants["pad_len"] > 0:
            tt_tokens_rm = ttnn.pad(tt_tokens_rm, [(0, 0), (0, 0), (0, 0), (0, constants["pad_len"])], value=0)

        tt_active_mask = ttnn.le(constants["codebook_indices"], tt_num_delay)
        tt_tokens_after_delay = ttnn.where(
            tt_active_mask,
            tt_tokens_rm,
            constants["bos_fill_uint"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(tt_active_mask)

        tt_can_open_more = ttnn.lt(tt_num_delay, constants["last_codebook_idx"])
        tt_num_delay_inc = ttnn.add(tt_num_delay, constants["one_scalar"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_num_delay_next = ttnn.where(
            tt_can_open_more,
            tt_num_delay_inc,
            tt_num_delay,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(tt_can_open_more)
        ttnn.deallocate(tt_num_delay_inc)

        tt_has_remaining = ttnn.ne(tt_num_remaining_delays, constants["none_scalar"])
        tt_eos_prefix = ttnn.sub(
            constants["num_codebooks"], tt_num_remaining_delays, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_existing_prefix_mask = ttnn.lt(constants["codebook_indices"], tt_eos_prefix)
        tt_tokens_with_existing_eos = ttnn.where(
            tt_existing_prefix_mask,
            constants["eos_fill_uint"],
            tt_tokens_after_delay,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_num_remaining_existing = ttnn.sub(
            tt_num_remaining_delays,
            constants["one_scalar"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(tt_eos_prefix)
        ttnn.deallocate(tt_existing_prefix_mask)

        # Keep padded lanes impossible to match so EOS detection does not depend on an extra valid-lane mask.
        # The row-major uint32 `eq` is correct here, but combining it with a separate valid mask via
        # `logical_and` produced incorrect alternating positives on device during traced decode.
        tt_eos_mask = ttnn.eq(tt_tokens_after_delay, constants["eos_compare_uint"])
        tt_eos_indices = ttnn.where(
            tt_eos_mask,
            constants["codebook_indices"],
            constants["zero_fill"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_eos_counts = ttnn.sum(
            ttnn.where(
                tt_eos_mask, constants["one_fill"], constants["zero_fill"], memory_config=ttnn.DRAM_MEMORY_CONFIG
            ),
            dim=3,
            keepdim=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_has_eos = ttnn.gt(tt_eos_counts, constants["zero_scalar"])
        tt_last_eos_idx = ttnn.argmax(tt_eos_indices, dim=3, keepdim=True)
        tt_prefix_mask_from_eos = ttnn.logical_and(
            tt_has_eos,
            ttnn.lt(constants["codebook_indices"], tt_last_eos_idx),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_tokens_from_eos = ttnn.where(
            tt_prefix_mask_from_eos,
            constants["eos_fill_uint"],
            tt_tokens_after_delay,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_num_remaining_from_eos = ttnn.sub(
            ttnn.sub(constants["num_codebooks"], tt_last_eos_idx, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            constants["one_scalar"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_num_remaining_else = ttnn.where(
            tt_has_eos,
            tt_num_remaining_from_eos,
            constants["none_scalar"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(tt_eos_mask)
        ttnn.deallocate(tt_eos_indices)
        ttnn.deallocate(tt_eos_counts)
        ttnn.deallocate(tt_prefix_mask_from_eos)
        ttnn.deallocate(tt_last_eos_idx)
        ttnn.deallocate(tt_num_remaining_from_eos)

        tt_tokens_after_eos = ttnn.where(
            tt_has_remaining,
            tt_tokens_with_existing_eos,
            tt_tokens_from_eos,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_num_remaining_next = ttnn.where(
            tt_has_remaining,
            tt_num_remaining_existing,
            tt_num_remaining_else,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(tt_has_remaining)
        ttnn.deallocate(tt_tokens_with_existing_eos)
        ttnn.deallocate(tt_num_remaining_existing)
        ttnn.deallocate(tt_has_eos)
        ttnn.deallocate(tt_tokens_from_eos)
        ttnn.deallocate(tt_num_remaining_else)
        ttnn.deallocate(tt_tokens_after_delay)

        tt_finished = ttnn.logical_and(
            ttnn.ne(tt_num_remaining_next, constants["none_scalar"]),
            ttnn.le(tt_num_remaining_next, constants["zero_scalar"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_num_delay_final = ttnn.where(
            tt_finished,
            constants["zero_scalar"],
            tt_num_delay_next,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_num_remaining_final = ttnn.where(
            tt_finished,
            constants["none_scalar"],
            tt_num_remaining_next,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(tt_num_delay_next)
        ttnn.deallocate(tt_num_remaining_next)

        # Decode replay keeps a fixed-width token-id buffer, but only the first `audio_embedding_seq_len` entries are
        # logical. Slice back to that compact row-major view before the in-place copy so replay reuses a stable spec.
        tt_next_tokens = tt_tokens_after_eos
        if shift_output_tokens:
            tt_next_tokens = ttnn.add(
                tt_next_tokens,
                constants["codebook_shift_uint"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        next_token_width = self.audio_embedding_seq_len if shift_output_tokens else self.audio_num_codebooks
        tt_next_tokens = ttnn.slice(
            tt_next_tokens,
            (0, 0, 0, 0),
            (1, 1, 1, next_token_width),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_next_tokens = ttnn.to_layout(
            tt_next_tokens,
            ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Row-major slice can collapse singleton leading dimensions into a compact 3D view; normalize back to the
        # fixed 4D token-id spec used by the persistent decode input buffer before the in-place replay copy.
        tt_next_tokens = ttnn.unsqueeze_to_4D(tt_next_tokens)
        ttnn.copy(tt_next_tokens, tt_next_input_tokens)
        ttnn.deallocate(tt_tokens_rm)
        ttnn.deallocate(tt_tokens_after_eos)
        ttnn.deallocate(tt_next_tokens)

        return tt_num_delay_final, tt_num_remaining_final, tt_finished

    def audio_ttnn_delay_pattern_postprocess_step_inplace(
        self,
        tt_raw_audio_tokens: ttnn.Tensor,
        tt_next_input_tokens: ttnn.Tensor,
        tt_num_delay: ttnn.Tensor,
        tt_num_remaining_delays: ttnn.Tensor,
        tt_finished_output: ttnn.Tensor | None = None,
        shift_output_tokens: bool = True,
    ) -> ttnn.Tensor:
        tt_num_delay_final, tt_num_remaining_final, tt_finished = self.audio_ttnn_delay_pattern_postprocess_inplace(
            tt_raw_audio_tokens,
            tt_next_input_tokens,
            tt_num_delay,
            tt_num_remaining_delays,
            shift_output_tokens=shift_output_tokens,
        )
        ttnn.copy(tt_num_delay_final, tt_num_delay)
        ttnn.copy(tt_num_remaining_final, tt_num_remaining_delays)
        ttnn.deallocate(tt_num_delay_final)
        ttnn.deallocate(tt_num_remaining_final)
        if tt_finished_output is not None:
            ttnn.copy(tt_finished, tt_finished_output)
            ttnn.deallocate(tt_finished)
            return tt_finished_output
        return tt_finished

    def transform_and_embed_prefill_inputs_device(
        self,
        tt_text_tokens: ttnn.Tensor,
        tt_audio_tokens: ttnn.Tensor | None,
        tt_audio_token_mask: ttnn.Tensor | None,
        tt_inverse_audio_token_mask: ttnn.Tensor | None,
        tt_page_table: ttnn.Tensor | None,
        tt_chunk_page_table: ttnn.Tensor | None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None]:
        tt_text_embeddings = self.text_embedding_tt(tt_text_tokens, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_text_embeddings = ttnn.unsqueeze_to_4D(tt_text_embeddings)
        if tt_audio_tokens is None:
            return (
                ttnn.to_memory_config(tt_text_embeddings, self.args.get_residual_mem_config(Mode.PREFILL, None)),
                tt_audio_token_mask,
                tt_inverse_audio_token_mask,
                tt_page_table,
                tt_chunk_page_table,
            )

        audio_embedding_tt_weights = self._ensure_audio_embedding_tt_weights()
        tt_audio_embeddings = ttnn.embedding(
            tt_audio_tokens,
            audio_embedding_tt_weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        if tt_audio_embeddings.shape[-2] != self.audio_num_codebooks:
            tt_audio_embeddings = ttnn.slice(
                tt_audio_embeddings,
                (0, 0, 0, 0),
                (1, tt_audio_embeddings.shape[1], self.audio_num_codebooks, tt_audio_embeddings.shape[-1]),
            )
        if self.audio_embed_avg:
            tt_audio_embeddings = ttnn.mean(
                tt_audio_embeddings,
                dim=2,
                keepdim=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            tt_audio_embeddings = ttnn.sum(
                tt_audio_embeddings,
                dim=2,
                keepdim=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        tt_audio_embeddings = ttnn.permute(tt_audio_embeddings, (0, 2, 1, 3))
        tt_audio_embeddings = ttnn.to_memory_config(tt_audio_embeddings, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        tt_x = HiggsDualFFNBlock._select_by_audio_mask(
            tt_audio_token_mask,
            tt_inverse_audio_token_mask,
            tt_text_embeddings,
            tt_audio_embeddings,
        )
        tt_x = ttnn.to_memory_config(tt_x, self.args.get_residual_mem_config(Mode.PREFILL, None), dtype=ttnn.bfloat16)
        return tt_x, tt_audio_token_mask, tt_inverse_audio_token_mask, tt_page_table, tt_chunk_page_table

    def ttnn_prefill_forward(
        self,
        x: ttnn.Tensor,
        effective_seq_len: int,
        audio_token_mask: ttnn.Tensor | None = None,
        inverse_audio_token_mask: ttnn.Tensor | None = None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        return_logits: bool = False,
        release_audio_masks: bool = True,
    ) -> ttnn.Tensor | None:
        seq_len = x.shape[2]
        rot_mats_global, rot_mats_local = self._slice_prefill_rot_mats(seq_len)
        tt_x = self._run_layers(
            x,
            None,
            rot_mats_global,
            rot_mats_local,
            Mode.PREFILL,
            is_audio_token=False,
            audio_token_mask=audio_token_mask,
            inverse_audio_token_mask=inverse_audio_token_mask,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        # Trace-captured prefill reuses persistent mask inputs across compile, capture, and replay,
        # so only eager one-shot prefill is allowed to consume these mask tensors here.
        if audio_token_mask is not None and release_audio_masks:
            ttnn.deallocate(audio_token_mask)
            ttnn.deallocate(inverse_audio_token_mask)
        if not return_logits:
            return tt_x

        last_block_start = (effective_seq_len - 1) // 32 * 32
        tt_x = ttnn.slice(tt_x, (0, 0, last_block_start, 0), (1, 1, last_block_start + 32, tt_x.shape[-1]))
        tt_x = self.norm(tt_x, mode=Mode.PREFILL, norm_config=self.args.get_norm_config("lm_head", Mode.PREFILL, None))
        if self.args.get_lm_head_input_mem_config(Mode.PREFILL, None).is_sharded():
            tt_x = ttnn.interleaved_to_sharded(tt_x, self.args.get_lm_head_input_mem_config(Mode.PREFILL, None))
        return self.text_head(tt_x)

    def audio_ttnn_decode_from_token_ids_forward(
        self,
        tt_audio_tokens: ttnn.Tensor,
        current_pos_tt: ttnn.Tensor,
        rot_mat_idxs: ttnn.Tensor,
    ):
        audio_embedding_tt_weights = self._ensure_audio_embedding_tt_weights()
        tt_audio_embeddings = ttnn.embedding(
            tt_audio_tokens,
            audio_embedding_tt_weights,
            layout=ttnn.TILE_LAYOUT,
            # Higgs only embeds 8 codebooks per decode step, so width-sharded decode residual output does not fit.
            # Keep the per-codebook lookup interleaved, then reduce to a single token embedding and shard that result.
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        tt_audio_embeddings = ttnn.unsqueeze_to_4D(tt_audio_embeddings)
        if tt_audio_embeddings.shape[-2] != self.audio_num_codebooks:
            tt_audio_embeddings = ttnn.slice(
                tt_audio_embeddings,
                (0, 0, 0, 0),
                (1, 1, self.audio_num_codebooks, tt_audio_embeddings.shape[-1]),
            )
        if self.audio_embed_avg:
            tt_embedding = ttnn.mean(
                tt_audio_embeddings,
                dim=2,
                keepdim=True,
                memory_config=self.args.get_residual_mem_config(Mode.DECODE, None),
            )
        else:
            tt_embedding = ttnn.sum(
                tt_audio_embeddings,
                dim=2,
                keepdim=True,
                memory_config=self.args.get_residual_mem_config(Mode.DECODE, None),
            )
        ttnn.deallocate(tt_audio_embeddings)
        return self.audio_ttnn_decode_forward(tt_embedding, current_pos_tt, rot_mat_idxs)

    def audio_ttnn_decode_from_raw_token_ids_forward(
        self,
        tt_raw_audio_tokens: ttnn.Tensor,
        current_pos_tt: ttnn.Tensor,
        rot_mat_idxs: ttnn.Tensor,
        shifted_tokens_output: ttnn.Tensor | None = None,
    ):
        tt_audio_tokens = ttnn.add(
            tt_raw_audio_tokens,
            self._ensure_audio_codebook_shift_tt(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_tensor=shifted_tokens_output,
        )
        return self.audio_ttnn_decode_from_token_ids_forward(tt_audio_tokens, current_pos_tt, rot_mat_idxs)

    def audio_ttnn_decode_forward(
        self,
        tt_embedding: ttnn.Tensor,
        current_pos_tt: ttnn.Tensor,
        rot_mat_idxs: ttnn.Tensor,
    ):
        tt_embedding = ttnn.to_memory_config(
            tt_embedding,
            self.args.get_residual_mem_config(Mode.DECODE, None),
            dtype=ttnn.bfloat16,
        )
        rot_mats_global = self.rope_setup.get_rot_mats(rot_mat_idxs)
        rot_mats_local = self.rope_local_setup.get_rot_mats(rot_mat_idxs) if self.rope_local_setup is not None else None
        tt_x = self._run_layers(
            tt_embedding,
            current_pos_tt,
            rot_mats_global,
            rot_mats_local,
            Mode.DECODE,
            is_audio_token=True,
        )
        tt_x = self.norm(tt_x, mode=Mode.DECODE, norm_config=self._get_decode_lm_head_norm_config())
        tt_x = self._prepare_decode_lm_head_input(tt_x)
        return self.audio_head(tt_x)

    def audio_ttnn_decode_greedy_tokens_forward(
        self,
        tt_embedding: ttnn.Tensor,
        current_pos_tt: ttnn.Tensor,
        rot_mat_idxs: ttnn.Tensor,
        output_tensor: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        tt_audio_logits = self.audio_ttnn_decode_forward(tt_embedding, current_pos_tt, rot_mat_idxs)
        return self._audio_logits_to_greedy_tokens(tt_audio_logits, output_tensor)

    def audio_ttnn_decode_greedy_tokens_from_token_ids_forward(
        self,
        tt_audio_tokens: ttnn.Tensor,
        current_pos_tt: ttnn.Tensor,
        rot_mat_idxs: ttnn.Tensor,
        output_tensor: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        tt_audio_logits = self.audio_ttnn_decode_from_token_ids_forward(tt_audio_tokens, current_pos_tt, rot_mat_idxs)
        return self._audio_logits_to_greedy_tokens(tt_audio_logits, output_tensor)

    def audio_ttnn_decode_greedy_tokens_from_raw_token_ids_forward(
        self,
        tt_raw_audio_tokens: ttnn.Tensor,
        current_pos_tt: ttnn.Tensor,
        rot_mat_idxs: ttnn.Tensor,
        shifted_tokens_output: ttnn.Tensor | None = None,
        output_tensor: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        tt_audio_logits = self.audio_ttnn_decode_from_raw_token_ids_forward(
            tt_raw_audio_tokens,
            current_pos_tt,
            rot_mat_idxs,
            shifted_tokens_output=shifted_tokens_output,
        )
        return self._audio_logits_to_greedy_tokens(tt_audio_logits, output_tensor)

    def audio_ttnn_decode_microstep_from_token_ids_inplace(
        self,
        tt_audio_tokens: ttnn.Tensor,
        current_pos_tt: ttnn.Tensor,
        rot_mat_idxs: ttnn.Tensor,
        tt_num_delay: ttnn.Tensor,
        tt_num_remaining_delays: ttnn.Tensor,
        tt_finished_output: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        tt_raw_audio_tokens = self.audio_ttnn_decode_greedy_tokens_from_token_ids_forward(
            tt_audio_tokens,
            current_pos_tt,
            rot_mat_idxs,
        )
        tt_finished = self.audio_ttnn_delay_pattern_postprocess_step_inplace(
            tt_raw_audio_tokens,
            tt_audio_tokens,
            tt_num_delay,
            tt_num_remaining_delays,
            tt_finished_output=tt_finished_output,
        )
        ttnn.deallocate(tt_raw_audio_tokens)
        self.increment_current_pos_tensor(current_pos_tt)
        self.increment_rot_mat_idx_tensor(rot_mat_idxs)
        return tt_finished

    def audio_ttnn_decode_microstep_from_raw_token_ids_inplace(
        self,
        tt_raw_audio_tokens: ttnn.Tensor,
        current_pos_tt: ttnn.Tensor,
        rot_mat_idxs: ttnn.Tensor,
        tt_num_delay: ttnn.Tensor,
        tt_num_remaining_delays: ttnn.Tensor,
        tt_finished_output: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        tt_raw_next_audio_tokens = self.audio_ttnn_decode_greedy_tokens_from_raw_token_ids_forward(
            tt_raw_audio_tokens,
            current_pos_tt,
            rot_mat_idxs,
        )
        tt_finished = self.audio_ttnn_delay_pattern_postprocess_step_inplace(
            tt_raw_next_audio_tokens,
            tt_raw_audio_tokens,
            tt_num_delay,
            tt_num_remaining_delays,
            tt_finished_output=tt_finished_output,
            shift_output_tokens=False,
        )
        ttnn.deallocate(tt_raw_next_audio_tokens)
        self.increment_current_pos_tensor(current_pos_tt)
        self.increment_rot_mat_idx_tensor(rot_mat_idxs)
        return tt_finished

    def audio_ttnn_decode_block_from_raw_token_ids_inplace(
        self,
        tt_raw_audio_tokens: ttnn.Tensor,
        current_pos_tt: ttnn.Tensor,
        rot_mat_idxs: ttnn.Tensor,
        tt_num_delay: ttnn.Tensor,
        tt_num_remaining_delays: ttnn.Tensor,
        *,
        block_steps: int,
        tt_finished_output: ttnn.Tensor,
    ) -> ttnn.Tensor:
        if block_steps < 1:
            raise ValueError("block_steps must be >= 1")
        tt_finished = tt_finished_output
        # The official perf suites use fixed 128/256-step horizons, so capture multiple decode microsteps into a single
        # trace block to amortize replay boundaries without changing the per-step device state semantics.
        for _ in range(block_steps):
            tt_finished = self.audio_ttnn_decode_microstep_from_raw_token_ids_inplace(
                tt_raw_audio_tokens,
                current_pos_tt,
                rot_mat_idxs,
                tt_num_delay,
                tt_num_remaining_delays,
                tt_finished_output=tt_finished_output,
            )
        return tt_finished

    def _audio_logits_to_greedy_tokens(
        self,
        tt_audio_logits: ttnn.Tensor,
        output_tensor: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        tt_audio_logits = ttnn.untilize_with_unpadding(
            tt_audio_logits,
            [0, 0, 0, self.args.audio_vocab_size - 1],
        )
        tt_audio_logits = ttnn.reshape(
            tt_audio_logits,
            [1, 1, self.audio_num_codebooks, self.audio_codebook_size],
        )
        tt_audio_tokens = ttnn.argmax(
            tt_audio_logits,
            dim=3,
            keepdim=True,
            use_multicore=True,
            output_tensor=output_tensor,
        )
        ttnn.deallocate(tt_audio_logits)
        return tt_audio_tokens

    def read_audio_decode_output(self, tt_audio_logits: ttnn.Tensor) -> torch.Tensor:
        return self._process_replicated_decode_output(tt_audio_logits, self.args.audio_vocab_size)

    def _process_replicated_decode_output(self, tt_out: ttnn.Tensor, vocab_size: int) -> torch.Tensor:
        out = ttnn.to_torch(tt_out).float()
        return out[0, 0, 0, :vocab_size]

    def _run_layers(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global,
        rot_mats_local,
        mode: Mode,
        is_audio_token: bool,
        audio_token_mask: ttnn.Tensor | None = None,
        inverse_audio_token_mask: ttnn.Tensor | None = None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
    ):
        for layer_idx, layer in enumerate(self.layers):
            activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
                decoder_id=layer_idx,
                tensor=TensorGroup.ACTIVATION,
            )
            if mode == Mode.DECODE and not self.args.is_galaxy:
                x = ttnn.to_memory_config(
                    x,
                    self.args.get_residual_mem_config(mode, None),
                    activation_dtype,
                )
            elif activation_dtype is not None and x.dtype != activation_dtype:
                x = ttnn.typecast(x, activation_dtype)
            x = layer(
                x=x,
                current_pos=current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                mode=mode,
                is_audio_token=is_audio_token,
                audio_token_mask=audio_token_mask,
                inverse_audio_token_mask=inverse_audio_token_mask,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
            )
        return x

    def prefill(
        self,
        merged_embeddings: torch.Tensor,
        audio_token_mask: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        force_decode_fallback: bool = False,
    ) -> torch.Tensor | None:
        merged_embeddings, audio_token_mask, effective_seq_len = self._bucket_prefill_inputs(
            merged_embeddings,
            audio_token_mask,
        )
        seq_len = merged_embeddings.shape[1]
        self.last_prefill_used_fallback = force_decode_fallback
        if force_decode_fallback:
            if audio_token_mask is None:
                prompt_audio_mask = torch.zeros((seq_len,), dtype=torch.bool)
            else:
                prompt_audio_mask = audio_token_mask[0] if audio_token_mask.dim() == 2 else audio_token_mask
            next_token_logits = None
            current_pos_tt = self.create_current_pos_tensor(0)
            try:
                for current_pos in range(seq_len):
                    next_token_logits, _ = self.decode_step(
                        current_embedding=merged_embeddings[0, current_pos],
                        current_pos=current_pos,
                        is_audio_token=bool(prompt_audio_mask[current_pos].item()),
                        return_text_logits=return_logits,
                        current_pos_tt=current_pos_tt,
                    )
                    self.increment_current_pos_tensor(current_pos_tt)
            finally:
                ttnn.deallocate(current_pos_tt)
            return next_token_logits if return_logits else None

        tt_x = self._to_device_embeddings(merged_embeddings, Mode.PREFILL)
        tt_audio_token_mask, tt_inverse_audio_token_mask = self._to_device_prefill_audio_masks(
            audio_token_mask, seq_len
        )
        tt_out = self.ttnn_prefill_forward(
            x=tt_x,
            effective_seq_len=effective_seq_len,
            audio_token_mask=tt_audio_token_mask,
            inverse_audio_token_mask=tt_inverse_audio_token_mask,
            return_logits=return_logits,
        )
        if not return_logits:
            # Performance prefill only needs the prompt KV state; skip lm-head work and host readback entirely.
            if tt_out is not None:
                ttnn.deallocate(tt_out)
            return None
        logits = ttnn.to_torch(tt_out).float()
        last_block_start = (effective_seq_len - 1) // 32 * 32
        return logits[0, 0, effective_seq_len - 1 - last_block_start, : self.args.vocab_size]

    def decode_step(
        self,
        current_embedding: torch.Tensor,
        current_pos: int,
        is_audio_token: bool,
        return_text_logits: bool = True,
        current_pos_tt: ttnn.Tensor | None = None,
    ):
        tt_x = self._to_device_embeddings(current_embedding.view(1, 1, -1), Mode.DECODE)
        rot_mats_global, rot_mats_local = self._decode_rot_mats(current_pos)
        if current_pos_tt is None:
            current_pos_tt = self.create_current_pos_tensor(current_pos)
        tt_x = self._run_layers(
            tt_x,
            current_pos_tt,
            rot_mats_global,
            rot_mats_local,
            Mode.DECODE,
            is_audio_token,
        )
        tt_x = self.norm(tt_x, mode=Mode.DECODE, norm_config=self._get_decode_lm_head_norm_config())
        if is_audio_token or return_text_logits:
            tt_x = self._prepare_decode_lm_head_input(tt_x)
        text_logits = None
        audio_logits = None
        if is_audio_token:
            if return_text_logits:
                # LMHead deallocates its input tensor, so dual-head decode needs a second view before text logits.
                tt_x_for_text = ttnn.clone(tt_x, memory_config=tt_x.memory_config())
                text_logits = self._process_replicated_decode_output(
                    self.text_head(tt_x_for_text), self.args.vocab_size
                )
            audio_logits = self._process_replicated_decode_output(self.audio_head(tt_x), self.args.audio_vocab_size)
        elif return_text_logits:
            text_logits = self._process_replicated_decode_output(self.text_head(tt_x), self.args.vocab_size)
        else:
            ttnn.deallocate(tt_x)
        return text_logits, audio_logits

    def _get_decode_lm_head_norm_config(self):
        return self.args.get_norm_config("lm_head", Mode.DECODE, None)

    def _prepare_decode_lm_head_input(self, tt_x: ttnn.Tensor) -> ttnn.Tensor:
        lm_head_input_mem_cfg = self.args.get_lm_head_input_mem_config(Mode.DECODE, None)
        if tt_x.memory_config() != lm_head_input_mem_cfg:
            tt_x = ttnn.to_memory_config(tt_x, lm_head_input_mem_cfg)
        return tt_x


def create_higgs_tt_model(
    mesh_device,
    model_name_or_path: str,
    max_seq_len: int = 8192,
    dummy_weights: bool = False,
    optimizations: str | Callable[[ModelArgs], DecodersPrecision] | None = None,
    use_hf_rope: bool = True,
    dtype=ttnn.bfloat8_b,
    paged_attention_config=None,
):
    # Higgs prompt/audio parity depends on HF RoPE and avoiding the BFP4 MLP fast path; keep that as the default.
    optimizations = resolve_higgs_optimizations(optimizations)
    args = HiggsModelArgs(
        mesh_device=mesh_device,
        model_name_or_path=model_name_or_path,
        dummy_weights=dummy_weights,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        optimizations=optimizations,
        cache_hf=False,
        use_hf_rope=use_hf_rope,
    )
    state_dict = args.load_state_dict()
    logger.info("Loaded Higgs state dict with {} tensors", len(state_dict))
    model = HiggsAudioTTModel(
        args=args,
        mesh_device=mesh_device,
        state_dict=state_dict,
        dtype=dtype,
        paged_attention_config=paged_attention_config,
    )
    return args, model, state_dict
