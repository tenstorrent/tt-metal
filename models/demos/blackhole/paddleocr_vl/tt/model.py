# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Text decoder fed pre-spliced embeddings and host M-RoPE, on qwen3_vl's generator contract."""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen3_vl.tt.rope import RotarySetup
from models.tt_transformers.tt.model import Transformer as TTTransformer


class Transformer(TTTransformer):
    # Inert with host sampling; kept as a prerequisite for any future attempt
    # at on-device sampling. See supports_sample_on_device in generator_vllm.py.
    _tt_vllm_always_refresh_decode_trace_inputs = True
    _tt_disable_sampling_trace = True

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        # Route greedy through the single-gather force-argmax path, not the
        # heavy top-k/top-p pipeline; must be set before super().__init__.
        ag_cfg = dict(args.model_config.get("SAMPLING_AG_CONFIG", {}) or {})
        ag_cfg["allow_force_argmax"] = True
        args.model_config["SAMPLING_AG_CONFIG"] = ag_cfg

        # qwen3_vl's RotarySetup carries the per-user rope_deltas that M-RoPE needs
        # at decode time: after an image, a user's text positions are offset by how
        # much the image's 3D positions advanced, and that offset differs per user.
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            rope_setup_class=RotarySetup,
        )

    def _prepare_cos_sin(self, rot_mats):
        """Upload host cos/sin ``[1, 1, S, head_dim]`` tables, replicated."""
        cos_matrix, sin_matrix = rot_mats
        assert cos_matrix.shape[0] == sin_matrix.shape[0], "cos and sin must agree on batch"
        out = []
        for mat in (cos_matrix, sin_matrix):
            out.append(
                ttnn.from_torch(
                    mat.expand(cos_matrix.shape[0], -1, -1, -1),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=self.rope_setup.datatype,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            )
        return out

    def prepare_inputs_prefill(
        self,
        tokens,
        rot_mats=None,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        """Prefill inputs from [1, S, dim] embeddings; rot_mats is sliced here so chunked prefill can advance start_pos."""
        assert rot_mats is not None, "PaddleOCR-VL prefill needs host M-RoPE tables; see tt/common.py"
        assert isinstance(rot_mats[0], torch.Tensor) and isinstance(rot_mats[1], torch.Tensor)
        assert len(tokens.shape) == 3, f"expected [batch, seq, dim] embeddings, got {tokens.shape}"

        S = tokens.shape[-2]
        tokens_embd = ttnn.unsqueeze(tokens, 1)

        cos_matrix, sin_matrix = self._prepare_cos_sin(rot_mats)
        assert (
            cos_matrix.shape[2] >= start_pos + S
        ), f"rope tables cover {cos_matrix.shape[2]} positions, need {start_pos + S}"
        tt_rot_mats_prefill = [
            cos_matrix[:, :, start_pos : start_pos + S, :],
            sin_matrix[:, :, start_pos : start_pos + S, :],
        ]

        device = self.mesh_device

        def _int32(t):
            return ttnn.from_torch(
                t,
                device=device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )

        tt_page_table = _int32(page_table) if page_table is not None else None
        tt_chunk_page_table = _int32(chunk_page_table) if chunk_page_table is not None else None

        # qwen3_vl's order has no chunk_start_idx slot; the chunked path passes start_pos instead.
        return (
            tokens_embd,
            tt_rot_mats_prefill,
            tt_page_table,
            tt_chunk_page_table,
            deepstack_visual_embeds,
        )

    def ttnn_prefill_forward(self, x, *args, deepstack_visual_embeds=None, **kwargs):
        """Drop the deepstack argument qwen3_vl's generator always passes; this model has none."""
        assert deepstack_visual_embeds is None, "PaddleOCR-VL has no deepstack path"
        return super().ttnn_prefill_forward(x, *args, **kwargs)
