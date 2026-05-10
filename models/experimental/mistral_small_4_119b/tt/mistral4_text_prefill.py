# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end **prefill logits** path (bring-up): ``embed_tokens`` → ``N`` × decoder → ``norm`` → ``lm_head``.

Composes :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_embed_tokens.TtMistral4EmbedTokensPrefill`,
:class:`~models.experimental.mistral_small_4_119b.tt.text_backbone.TtMistral4DecoderSequence`, and
:class:`~models.experimental.mistral_small_4_119b.tt.mistral4_output_head.TtMistral4FinalNormLmHeadPrefill`.
``forward`` returns **torch** ``[batch, seq, vocab]`` logits (host ``lm_head``). With the default
host MoE path, keep **N** small (~6 GiB bf16 routed weights **per layer** on CPU). Set
``use_ttnn_moe=True`` on :class:`TtMistral4TextPrefillLogits` to run routed + shared experts on TTNN
(see :class:`~models.experimental.mistral_small_4_119b.tt.text_backbone.TtMistral4DecoderSequence`).

By default the decoder stack uses a **device RoPE cos/sin table** (``use_device_rotary_embedding_table=True``);
``position_embeddings`` in ``forward`` is then ignored for TTNN (still fine to pass for parity with a
host reference). Set ``use_device_rotary_embedding_table=False`` to upload HF cos/sin once per forward instead.

For **decode** after prefill, create ``stack_kv = model.make_stack_kv_state()``, run ``forward`` with
``mode="prefill"`` and ``stack_kv=stack_kv``, then ``mode="decode"`` with a single-token ``input_ids``
and matching ``position_ids`` / rotary (PCC vs HF for ``N`` in ``{1, 2}`` in
``tests/test_text_prefill_e2e_decode_logits_pcc.py``).
"""

from __future__ import annotations

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_small_4_119b.tt.mistral4_embed_tokens import TtMistral4EmbedTokensPrefill
from models.experimental.mistral_small_4_119b.tt.mistral4_output_head import TtMistral4FinalNormLmHeadPrefill
from models.experimental.mistral_small_4_119b.tt.mistral4_kv_state import Mistral4DecoderStackKvState
from models.experimental.mistral_small_4_119b.tt.text_backbone import TtMistral4DecoderSequence, _validate_layer_idx


class TtMistral4TextPrefillLogits(LightweightModule):
    """
    Prefill logits: ``input_ids`` → TTNN hidden stack → host norm + ``lm_head`` → logits.

    ``state_dict`` must contain embed, ``layers.{0..N-1}.*``, ``model.norm``, and ``lm_head`` keys
    (MoE weights live under each layer's ``mlp.*`` keys).

    Keyword args:
        ``use_device_rotary_embedding_table`` (default ``True``): build persistent cos/sin rows on
        mesh and gather by ``position_ids`` (no host rotary tensor required on the TT path).
        ``rotary_table_num_positions``: number of rows in that table (HF config caps the maximum).
        ``use_ttnn_moe`` (default ``False``): use TTNN MoE in each decoder layer instead of host HF MoE.
        ``tt_ccl``: optional :class:`~models.tt_transformers.tt.ccl.TT_CCL` for multi-device MoE; if
        omitted and ``use_ttnn_moe`` with more than one mesh device, the decoder sequence constructs one.
        ``moe_hf_torch_routing``: ``False`` for demo-style **device** gate + ``topk`` (no host routing in MoE forward);
            ``True`` for HF-identical host ``topk`` during PCC bring-up (see :class:`TtMistral4DecoderLayer`).
        ``moe_shard_routed_experts``: optional override for routed expert sharding (see :class:`TtMistral4DecoderSequence`).
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        text_config,
        *,
        num_decoder_layers: int,
        weight_dtype=ttnn.bfloat16,
        eps: float | None = None,
        use_device_rotary_embedding_table: bool = True,
        rotary_table_num_positions: int = 8192,
        tt_ccl=None,
        use_ttnn_moe: bool = False,
        moe_hf_torch_routing: bool = False,
        moe_shard_routed_experts: bool | None = None,
    ):
        super().__init__()
        if num_decoder_layers < 1:
            raise ValueError("num_decoder_layers must be >= 1")
        _validate_layer_idx(num_decoder_layers - 1, text_config)
        self.device = device
        self.num_decoder_layers = int(num_decoder_layers)
        self._use_rotary_table = bool(use_device_rotary_embedding_table)
        self.embed = TtMistral4EmbedTokensPrefill(device, state_dict)
        self.stack = TtMistral4DecoderSequence(
            device,
            state_dict,
            text_config,
            layer_indices=tuple(range(self.num_decoder_layers)),
            weight_dtype=weight_dtype,
            eps=eps,
            use_device_rotary_embedding_table=use_device_rotary_embedding_table,
            rotary_table_num_positions=rotary_table_num_positions,
            tt_ccl=tt_ccl,
            use_ttnn_moe=use_ttnn_moe,
            moe_hf_torch_routing=moe_hf_torch_routing,
            moe_shard_routed_experts=moe_shard_routed_experts,
        )
        self.head = TtMistral4FinalNormLmHeadPrefill(state_dict, text_config)

    def make_stack_kv_state(self) -> Mistral4DecoderStackKvState:
        """Allocate a :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_kv_state.Mistral4DecoderStackKvState` for this depth (prefill then ``mode='decode'``)."""
        return Mistral4DecoderStackKvState(self.num_decoder_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        mode: str = "prefill",
        stack_kv: Mistral4DecoderStackKvState | None = None,
    ) -> torch.Tensor:
        hidden = self.embed(input_ids)
        hidden = self.stack(
            hidden,
            position_ids=position_ids,
            position_embeddings=None if self._use_rotary_table else position_embeddings,
            mode=mode,
            stack_kv=stack_kv,
        )
        return self.head(hidden, mesh_device=self.device, logical_batch=int(input_ids.shape[0]))
