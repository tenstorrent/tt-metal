# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal Mistral with **HF Mistral4 hybrid text** (mesh matmuls + optional device SDPA) and the
existing TT **vision tower + MMP** from the 24B-style path.

Enabled when :func:`models.tt_transformers.demo.simple_vision_demo.create_multimodal_model` builds this
class (see env ``TT_METAL_MISTRAL4_HYBRID_TEXT``). ``Generator`` recognizes
``_mistral4_hybrid_generator`` for torch logits on prefill/decode.
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.mistral_small_4.text_backbone import Mistral4HybridTextBackbone
from models.tt_transformers.tt.multimodal.mistral_24b.vision_model import TtMistralVisionTransformer


class Mistral4HybridMultimodalTransformer(LightweightModule):
    """Vision on TT + Mistral4 hybrid causal LM; duck-typed for ``Generator`` text prefill/decode."""

    _mistral4_hybrid_generator = True

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        hf_causal_lm: torch.nn.Module,
        paged_attention_config=None,  # unused; accepted for parity with ``MistralTransformer``
        use_paged_kv_cache=False,
        *,
        use_device_sdpa_attention: bool = False,
    ):
        super().__init__()
        self.args = args
        self.configuration = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.vocab_size = int(args.vocab_size)
        self.prefetcher = None
        self.sampling = None
        self._supports_on_device_sampling = False
        self._mistral4_past_kv = None
        self.tt_ccl = TT_CCL(mesh_device)
        self.vision_model = TtMistralVisionTransformer(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix="vision_tower.",
            dtype=dtype,
            model_args=args,
            tt_ccl=self.tt_ccl,
        )
        self._hybrid = Mistral4HybridTextBackbone(
            mesh_device,
            hf_causal_lm,
            use_device_sdpa_attention=use_device_sdpa_attention,
        )

    def switch_mode(self, mode: Mode):
        return

    def compute_vision_token(self, pixel_values=None, image_sizes=None, **kwargs):
        if pixel_values is None:
            return None
        if image_sizes is not None and not isinstance(image_sizes[0], (list, tuple)):
            image_sizes = [image_sizes]
        return self.vision_model(pixel_values, image_sizes)

    def prepare_inputs_prefill(
        self, pt_tokens, start_pos=0, page_table=None, chunk_page_table=None, trace_enabled=False, **kwargs
    ):
        if trace_enabled:
            raise NotImplementedError("Mistral4HybridMultimodalTransformer does not support traced prefill")
        if start_pos != 0:
            raise NotImplementedError("Hybrid multimodal prefill with cached prefix (start_pos>0) is not supported")
        if page_table is not None or chunk_page_table is not None:
            raise NotImplementedError("Paged / chunked prefill is not supported for Mistral4 hybrid multimodal")

        self._mistral4_past_kv = None

        device = self.mesh_device
        pt_tokens = pt_tokens.long()
        s = int(pt_tokens.shape[-1])
        tokens = ttnn.from_torch(
            pt_tokens.reshape(1, 1, 1, -1),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        embed = self._hybrid.hf_model.model.embed_tokens(pt_tokens)
        vision_output = self.compute_vision_token(**kwargs)
        if vision_output is not None:
            comp_vision_output = ttnn.to_torch(
                vision_output, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            )[: vision_output.shape[0], :]
            image_features = comp_vision_output.squeeze(0)
            special_image_mask = (pt_tokens == self.args.image_token_index).unsqueeze(-1).expand_as(embed)
            image_features = image_features.to(embed.device, embed.dtype)
            embed = embed.masked_scatter(special_image_mask, image_features)

        self._prefill_fused_bsh = embed.to(torch.bfloat16).contiguous()
        self._prefill_input_ids = pt_tokens

        return tokens, None, None, None, None

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        **kwargs,
    ):
        del x, rot_mats_global, rot_mats_local, user_id, page_table, chunk_page_table, chunk_start_idx
        del get_last_token, kv_cache, kwargs
        if batch_size != 1:
            raise NotImplementedError("Mistral4HybridMultimodalTransformer currently supports batch_size=1")

        logger.info(
            "Mistral4 hybrid multimodal: running full-language prefill (first silicon run may compile many minutes; "
            "no per-op progress)."
        )
        logits, past = self._hybrid.incremental_logits(
            self._prefill_input_ids,
            past_key_values=None,
            position_ids=None,
            attention_mask_2d=None,
            prefused_hidden_states_bsh=self._prefill_fused_bsh,
        )
        self._mistral4_past_kv = past
        return logits

    def mistral4_hybrid_decode_forward(self, next_token_ids: torch.Tensor, position_ids_1d: torch.Tensor):
        """Decode one step; ``position_ids_1d`` is shape ``[B]`` (absolute positions)."""
        pos = position_ids_1d.long().view(-1, 1)
        logits, past = self._hybrid.incremental_logits(
            next_token_ids.long(),
            past_key_values=self._mistral4_past_kv,
            position_ids=pos,
            attention_mask_2d=None,
            prefused_hidden_states_bsh=None,
        )
        self._mistral4_past_kv = past
        return logits, None
