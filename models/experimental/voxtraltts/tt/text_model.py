# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import ttnn
import torch
from models.tt_transformers.tt.model import Transformer

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
            weight_cache_path=args.weight_cache_path(dtype),
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
