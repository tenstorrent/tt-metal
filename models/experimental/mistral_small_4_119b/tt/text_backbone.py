# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Text backbone (Mistral4Model) — implementation placeholder.

HF wraps the language model as ``Mistral4Model`` with ``Mistral4DecoderLayer``:

- **Attention**: Mistral4Attention with compressed / MLA-style projections (see
  ``mistral-small-4-119B_max_depth.txt``: q/k/v and latents differ from the dense
  Mistral3 path used by ``models.tt_transformers.tt.attention.Attention``).
- **FFN**: ``Mistral4MoE`` = ``Mistral4NaiveMoe`` (fused gate_up + down per expert),
  ``Mistral4TopkRouter`` (``[num_experts, hidden]``), plus a shared ``Mistral4MLP``.

:class:`TtMistral4DecoderLayer` / :class:`TtMistral4DecoderLayerAttnPrefillBlock` take
``layer_idx`` and read hub keys under ``constants.text_decoder_layer_state_dict_prefix(layer_idx)``.
Pass :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_kv_state.Mistral4AttentionKvState`
into ``forward_split`` / ``forward`` with ``mode="prefill"`` to snapshot K/V, then
``mode="decode"`` with the same object for one-token extensions (see
:class:`~models.experimental.mistral_small_4_119b.tt.mistral4_kv_state.Mistral4DecoderStackKvState`).
Layer-0-only names (``TtMistral4DecoderLayer0`` etc.) remain as thin aliases for call sites
and tests that still target layer 0.

Bring-up stubs (RMS norms, BF16 hub weights)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:class:`TtMistral4DecoderLayerRmsNormStub` runs one ``Mistral4RMSNorm`` on device. See
``tests/test_text_decoder_layer0_input_norm_stub_pcc.py``. For MLA self-attention prefill,
see ``tt/mistral4_self_attention.py`` and ``tests/test_text_decoder_layer0_self_attn_prefill_pcc.py``.
"""

from __future__ import annotations

import torch
import ttnn

from models.common.auto_compose import to_torch_auto_compose, trim_torch_compose_to_reference_shape
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_24b.tt.rmsnorm import RMSNorm
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_LAYERS,
    EXPECTED_RMS_NORM_EPS,
    strip_fp8_aux_tensors_from_decoder_inner,
    text_decoder_layer_inner_state_dict,
    text_decoder_layer_state_dict_prefix,
    text_decoder_self_attn_weight_slice,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_kv_state import (
    Mistral4AttentionKvState,
    Mistral4DecoderStackKvState,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh import (
    TtMistral4MoeRoutedExpertParallelSkeleton,
    TtMistral4SharedExpertsMlpTtnn,
    mistral4_mlp_state_dict_bf16_match_hf,
    route_tokens_to_experts_torch,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import (
    TtMistral4SelfAttentionPrefill,
    upload_mistral4_rotary_cos_sin_to_mesh,
)


def _validate_layer_idx(layer_idx: int, text_config=None) -> int:
    n = int(getattr(text_config, "num_hidden_layers", 0) or 0) if text_config is not None else 0
    if n <= 0:
        n = int(EXPECTED_NUM_LAYERS)
    if layer_idx < 0 or layer_idx >= n:
        raise ValueError(f"layer_idx must be in [0, {n}), got {layer_idx}")
    return layer_idx


class TtMistral4DecoderLayerRmsNormStub(LightweightModule):
    """
    TTNN stub for one ``Mistral4RMSNorm`` in decoder layer ``layer_idx`` (hub BF16 weights).

    ``weight_key`` is the HF submodule name without ``.weight``, e.g.
    ``"input_layernorm"`` or ``"post_attention_layernorm"``.

    ``state_dict`` must include ``{prefix}{weight_key}.weight`` with
    ``prefix = text_decoder_layer_state_dict_prefix(layer_idx)``.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        *,
        layer_idx: int,
        weight_key: str,
        weight_dtype=ttnn.bfloat16,
        eps: float | None = None,
        text_config=None,
    ):
        super().__init__()
        layer_idx = _validate_layer_idx(layer_idx, text_config)
        prefix = text_decoder_layer_state_dict_prefix(layer_idx)
        if eps is None:
            eps = float(EXPECTED_RMS_NORM_EPS)
        self._norm = RMSNorm(
            device=device,
            dim=EXPECTED_HIDDEN_SIZE,
            state_dict=state_dict,
            state_dict_prefix=prefix,
            weight_key=weight_key,
            weight_dtype=weight_dtype,
            is_distributed=False,
            simplified_rms=True,
            eps=eps,
        )

    def forward(self, x: ttnn.Tensor, *, mode: str = "prefill") -> ttnn.Tensor:
        return self._norm(x, mode=mode)


class TtMistral4DecoderLayerInputNormStub(TtMistral4DecoderLayerRmsNormStub):
    """``input_layernorm`` for decoder layer ``layer_idx``."""

    def __init__(self, device, state_dict: dict, *, layer_idx: int, text_config=None, **kwargs):
        super().__init__(
            device, state_dict, layer_idx=layer_idx, weight_key="input_layernorm", text_config=text_config, **kwargs
        )


class TtMistral4DecoderLayerPostAttnNormStub(TtMistral4DecoderLayerRmsNormStub):
    """``post_attention_layernorm`` for decoder layer ``layer_idx``."""

    def __init__(self, device, state_dict: dict, *, layer_idx: int, text_config=None, **kwargs):
        super().__init__(
            device,
            state_dict,
            layer_idx=layer_idx,
            weight_key="post_attention_layernorm",
            text_config=text_config,
            **kwargs,
        )


class TtMistral4DecoderLayerAttnPrefillBlock(LightweightModule):
    """
    Prefill subgraph through ``post_attention_layernorm`` (no MoE) for layer ``layer_idx``.

    Pipeline: residual branch → ``input_layernorm`` → :class:`TtMistral4SelfAttentionPrefill`
    → add residual → ``post_attention_layernorm``.

    ``state_dict`` must use hub keys under ``language_model.model.layers.{layer_idx}.*``.
    For the full decoder layer (MoE + final residual), see :class:`TtMistral4DecoderLayer`.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        text_config,
        *,
        layer_idx: int,
        weight_dtype=ttnn.bfloat16,
        eps: float | None = None,
    ):
        super().__init__()
        layer_idx = _validate_layer_idx(layer_idx, text_config)
        if eps is None:
            eps = float(EXPECTED_RMS_NORM_EPS)
        self.device = device
        self.layer_idx = layer_idx
        self.input_norm = TtMistral4DecoderLayerInputNormStub(
            device, state_dict, layer_idx=layer_idx, text_config=text_config, weight_dtype=weight_dtype, eps=eps
        )
        attn_sd = text_decoder_self_attn_weight_slice(state_dict, layer_idx)
        self.self_attn = TtMistral4SelfAttentionPrefill(device, text_config, attn_sd)
        self.post_norm = TtMistral4DecoderLayerPostAttnNormStub(
            device, state_dict, layer_idx=layer_idx, text_config=text_config, weight_dtype=weight_dtype, eps=eps
        )

    def forward_split(
        self,
        hidden_11SH: ttnn.Tensor,
        *,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_embeddings_tt: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        mode: str = "prefill",
        kv_state: Mistral4AttentionKvState | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Returns ``(hidden_mid, post_norm_mid)`` matching HF tensors after attention and
        before ``Mistral4MoE`` (``hidden_mid`` = residual + attention output).

        When ``kv_state`` is set: ``mode="prefill"`` fills it via
        :meth:`TtMistral4SelfAttentionPrefill.forward_prefill_with_kv`; ``mode="decode"``
        (seq ``1``) extends cache via :meth:`TtMistral4SelfAttentionPrefill.forward_decode_extend_kv`.
        """
        if not isinstance(position_ids, torch.Tensor):
            raise TypeError("position_ids must be a torch.Tensor (rotary + HF parity).")
        if (position_embeddings is None) == (position_embeddings_tt is None):
            raise ValueError("Provide exactly one of position_embeddings or position_embeddings_tt.")
        if mode not in ("prefill", "decode"):
            raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
        if mode == "decode":
            if kv_state is None:
                raise ValueError("decode requires kv_state (per-layer K/V).")
            if not kv_state.populated:
                raise ValueError("decode requires a populated kv_state; run prefill with the same kv_state first.")

        residual = ttnn.clone(hidden_11SH)
        normed = self.input_norm(hidden_11SH, mode=mode)
        if kv_state is None:
            attn_out = self.self_attn(
                normed,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                position_embeddings_tt=position_embeddings_tt,
            )
        elif mode == "prefill":
            attn_out, k, v = self.self_attn.forward_prefill_with_kv(
                normed,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                position_embeddings_tt=position_embeddings_tt,
            )
            kv_state.replace(k, v)
        else:
            assert kv_state.key_states is not None and kv_state.value_states is not None
            attn_out, k_full, v_full = self.self_attn.forward_decode_extend_kv(
                normed,
                position_ids=position_ids,
                past_key_states=kv_state.key_states,
                past_value_states=kv_state.value_states,
                position_embeddings=position_embeddings,
                position_embeddings_tt=position_embeddings_tt,
            )
            kv_state.replace(k_full, v_full)

        hidden_mid = ttnn.add(residual, attn_out)
        return hidden_mid, self.post_norm(hidden_mid, mode=mode)

    def forward(
        self,
        hidden_11SH: ttnn.Tensor,
        *,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_embeddings_tt: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        mode: str = "prefill",
        kv_state: Mistral4AttentionKvState | None = None,
    ) -> ttnn.Tensor:
        return self.forward_split(
            hidden_11SH,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            position_embeddings_tt=position_embeddings_tt,
            mode=mode,
            kv_state=kv_state,
        )[1]


class TtMistral4DecoderLayer(LightweightModule):
    """
    Decoder layer ``layer_idx``: TTNN attention + MoE.

    * ``use_ttnn_moe=False`` (default): **host** ``Mistral4MoE`` (HF) for exact parity; activations
      round-trip host ↔ mesh. Host RAM scales with stacked layers (~6 GiB bf16 routed weights per layer).
    * ``use_ttnn_moe=True``: routed experts + ``shared_experts`` on TTNN
      (:class:`~models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh.TtMistral4MoeRoutedExpertParallelSkeleton`
      and :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh.TtMistral4SharedExpertsMlpTtnn`);
      ``tt_ccl`` is required when the mesh has more than one device (see skeleton). No torch matmul
      in the MoE ``forward``. Set ``moe_hf_torch_routing=True`` to use HF ``F.linear`` + ``topk`` on
      host (same activations as the host-MoE path) and TTNN expert matmuls (for full-layer PCC).
      For a **TTNN-only MoE forward** (no host ``topk`` / gate matmul), use ``moe_hf_torch_routing=False``; the
      skeleton uses FP32 softmax then bf16 ``ttnn.topk`` (see :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh.TtMistral4MoeRoutedExpertParallelSkeleton`).
      ``moe_shard_routed_experts``: passed to :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh.TtMistral4MoeRoutedExpertParallelSkeleton`
      (``None`` = auto-shard when ``n_routed_experts`` divides mesh device count).
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        text_config,
        *,
        layer_idx: int,
        weight_dtype=ttnn.bfloat16,
        eps: float | None = None,
        tt_ccl=None,
        use_ttnn_moe: bool = False,
        moe_hf_torch_routing: bool = False,
        moe_shard_routed_experts: bool | None = None,
    ):
        super().__init__()
        layer_idx = _validate_layer_idx(layer_idx, text_config)
        if eps is None:
            eps = float(EXPECTED_RMS_NORM_EPS)
        self.device = device
        self.layer_idx = layer_idx
        self.text_config = text_config
        self._use_ttnn_moe = bool(use_ttnn_moe)
        self._moe_hf_torch_routing = bool(moe_hf_torch_routing)
        self._attn = TtMistral4DecoderLayerAttnPrefillBlock(
            device, state_dict, text_config, layer_idx=layer_idx, weight_dtype=weight_dtype, eps=eps
        )

        inner = strip_fp8_aux_tensors_from_decoder_inner(text_decoder_layer_inner_state_dict(state_dict, layer_idx))
        mlp_sd = {k[len("mlp.") :]: v for k, v in inner.items() if k.startswith("mlp.")}
        if self._use_ttnn_moe:
            mlp_sd = mistral4_mlp_state_dict_bf16_match_hf(text_config, mlp_sd)

        self._hf_moe = None
        self._tt_routed_moe = None
        self._tt_shared_mlp = None
        self._mlp_gate_weight_bf16: torch.Tensor | None = None

        if self._use_ttnn_moe:
            if self._moe_hf_torch_routing:
                if "gate.weight" not in mlp_sd:
                    raise KeyError("moe_hf_torch_routing requires mlp_sd['gate.weight']")
                self._mlp_gate_weight_bf16 = mlp_sd["gate.weight"].detach().to(torch.bfloat16).contiguous()
            self._tt_routed_moe = TtMistral4MoeRoutedExpertParallelSkeleton(
                device,
                text_config,
                mlp_sd,
                tt_ccl=tt_ccl,
                weight_dtype=weight_dtype,
                use_fp32_router_softmax=not self._moe_hf_torch_routing,
                shard_routed_experts=moe_shard_routed_experts,
            )
            self._tt_shared_mlp = TtMistral4SharedExpertsMlpTtnn(device, text_config, mlp_sd, weight_dtype=weight_dtype)
        else:
            try:
                from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE
            except ImportError as exc:
                raise ImportError(
                    "TtMistral4DecoderLayer requires ``transformers`` with ``Mistral4MoE`` (recent mistral4)."
                ) from exc
            self._hf_moe = Mistral4MoE(text_config).eval()
            self._hf_moe.load_state_dict(mlp_sd, strict=True)
            self._hf_moe = self._hf_moe.to(torch.bfloat16)

    def forward(
        self,
        hidden_11SH: ttnn.Tensor,
        *,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_embeddings_tt: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        mode: str = "prefill",
        kv_state: Mistral4AttentionKvState | None = None,
    ) -> ttnn.Tensor:
        if (position_embeddings is None) == (position_embeddings_tt is None):
            raise ValueError("Provide exactly one of position_embeddings or position_embeddings_tt.")
        hidden_mid, normed = self._attn.forward_split(
            hidden_11SH,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            position_embeddings_tt=position_embeddings_tt,
            mode=mode,
            kv_state=kv_state,
        )
        if self._use_ttnn_moe:
            if self._moe_hf_torch_routing:
                # Same MoE activations and gate matmul as host ``Mistral4MoE`` (``to_torch`` + ``F.linear``);
                # routed + shared matmuls stay on TTNN for PCC vs full HF layer.
                nt = to_torch_auto_compose(normed, device=self.device)
                seq_len = int(hidden_mid.shape[2])
                hidden = int(nt.shape[3])
                logical_batch = int(position_ids.shape[0])
                nt = trim_torch_compose_to_reference_shape(nt, (logical_batch, 1, seq_len, hidden))
                x_hf = nt[:, 0, :seq_len, :].contiguous()
                h = int(x_hf.shape[-1])
                logits_flat = torch.nn.functional.linear(x_hf.reshape(-1, h), self._mlp_gate_weight_bf16)
                topk_idx_th, topk_w_th = route_tokens_to_experts_torch(logits_flat, self.text_config)
                mlp_in = x_hf.unsqueeze(1)
                n_tile = ttnn.from_torch(
                    mlp_in,
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=self.device),
                )
                routed = self._tt_routed_moe(
                    n_tile,
                    topk_indices_torch=topk_idx_th,
                    topk_weights_torch=topk_w_th,
                )
                shared = self._tt_shared_mlp(n_tile)
                mlp_tt = ttnn.add(routed, shared)
                ttnn.deallocate(routed)
                ttnn.deallocate(shared)
                ttnn.deallocate(n_tile)
            else:
                # Post-attention RMSNorm may be ROW-major; MoE linears use TILE weights/activations.
                n_tile = ttnn.to_layout(normed, ttnn.TILE_LAYOUT)
                routed = self._tt_routed_moe(n_tile)
                shared = self._tt_shared_mlp(n_tile)
                mlp_tt = ttnn.add(routed, shared)
                ttnn.deallocate(routed)
                ttnn.deallocate(shared)
                if id(n_tile) != id(normed):
                    ttnn.deallocate(n_tile)
        else:
            nt = to_torch_auto_compose(normed, device=self.device)
            seq_len = int(hidden_mid.shape[2])
            hidden = int(nt.shape[3])
            logical_batch = int(position_ids.shape[0])
            nt = trim_torch_compose_to_reference_shape(nt, (logical_batch, 1, seq_len, hidden))
            x_bf16 = nt[:, 0, :seq_len, :].contiguous()
            mlp_out = self._hf_moe(x_bf16)
            mlp_11sh = mlp_out.unsqueeze(1)
            mapper = ttnn.ReplicateTensorToMesh(mesh_device=self.device)
            mlp_tt = ttnn.from_torch(
                mlp_11sh,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
        return ttnn.add(hidden_mid, mlp_tt)


class TtMistral4DecoderSequence(LightweightModule):
    """
    Sequentially run multiple :class:`TtMistral4DecoderLayer` blocks (prefill).

    Uses one ``position_ids`` for all layers, matching HF ``Mistral4Model``.

    **Rotary**

    * If ``use_device_rotary_embedding_table`` is ``False`` (default): pass host
      ``position_embeddings``; cos/sin are uploaded **once** per forward and shared
      across layers (``position_embeddings_tt``).
    * If ``True``: host ``position_embeddings`` must be omitted; cos/sin come from a
      persistent mesh table + :func:`ttnn.embedding` (see
      :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_rotary_mesh_table.TtMistral4RotaryEmbeddingMeshTable`).

    **MoE**

    * Default (``use_ttnn_moe=False``): each layer holds a **host** ``Mistral4MoE``; stacking **N**
      layers multiplies host RAM (~6 GiB routed experts per layer in bf16); keep **N** small on
      bring-up machines (tests use ``N <= 2``).
    * ``use_ttnn_moe=True``: device MoE per layer (routed experts sharded across mesh when divisible,
      else replicated; shared MLP replicated). If ``tt_ccl`` is omitted and ``device.get_num_devices() > 1``,
      a :class:`~models.tt_transformers.tt.ccl.TT_CCL` is constructed once and shared across layers.
      ``moe_hf_torch_routing`` and ``moe_shard_routed_experts`` are passed to each :class:`TtMistral4DecoderLayer`
      when ``use_ttnn_moe`` is enabled.

    **KV / decode**

    Optional :class:`~models.experimental.mistral_small_4_119b.tt.mistral4_kv_state.Mistral4DecoderStackKvState`
    ``stack_kv`` (length ``len(blocks)``): with ``mode="prefill"`` each layer snapshots attention K/V;
    with ``mode="decode"`` and hidden ``[1,1,1,H]``, each layer extends its cache (single new token).
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        text_config,
        *,
        layer_indices: tuple[int, ...],
        weight_dtype=ttnn.bfloat16,
        eps: float | None = None,
        use_device_rotary_embedding_table: bool = False,
        rotary_table_num_positions: int = 8192,
        tt_ccl=None,
        use_ttnn_moe: bool = False,
        moe_hf_torch_routing: bool = False,
        moe_shard_routed_experts: bool | None = None,
    ):
        super().__init__()
        if not layer_indices:
            raise ValueError("layer_indices must be non-empty")
        for li in layer_indices:
            _validate_layer_idx(li, text_config)
        self.device = device
        self.layer_indices = tuple(layer_indices)
        resolved_ccl = tt_ccl
        if use_ttnn_moe:
            nd = int(device.get_num_devices())
            if nd > 1 and resolved_ccl is None:
                from models.tt_transformers.tt.ccl import TT_CCL

                resolved_ccl = TT_CCL(device)
        self._tt_ccl_for_moe = resolved_ccl
        self.blocks = tuple(
            TtMistral4DecoderLayer(
                device,
                state_dict,
                text_config,
                layer_idx=li,
                weight_dtype=weight_dtype,
                eps=eps,
                tt_ccl=self._tt_ccl_for_moe,
                use_ttnn_moe=use_ttnn_moe,
                moe_hf_torch_routing=moe_hf_torch_routing,
                moe_shard_routed_experts=moe_shard_routed_experts,
            )
            for li in self.layer_indices
        )
        self._rotary_table = None
        if use_device_rotary_embedding_table:
            from models.experimental.mistral_small_4_119b.tt.mistral4_rotary_mesh_table import (
                TtMistral4RotaryEmbeddingMeshTable,
            )

            self._rotary_table = TtMistral4RotaryEmbeddingMeshTable(
                device, text_config, num_positions=rotary_table_num_positions
            )

    def forward(
        self,
        hidden_11SH: ttnn.Tensor,
        *,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        mode: str = "prefill",
        stack_kv: Mistral4DecoderStackKvState | None = None,
    ) -> ttnn.Tensor:
        if mode == "decode" and stack_kv is None:
            raise ValueError("mode='decode' requires stack_kv (per-layer Mistral4AttentionKvState).")
        if stack_kv is not None and len(stack_kv) != len(self.blocks):
            raise ValueError(f"stack_kv must have {len(self.blocks)} layers, got {len(stack_kv)}")

        if self._rotary_table is not None:
            if position_embeddings is not None:
                raise ValueError(
                    "When use_device_rotary_embedding_table=True, pass position_embeddings=None "
                    "(RoPE cos/sin are gathered on device)."
                )
            cos_tt, sin_tt = self._rotary_table.gather(position_ids)
            try:
                h = hidden_11SH
                for i, block in enumerate(self.blocks):
                    h = block(
                        h,
                        position_ids=position_ids,
                        position_embeddings_tt=(cos_tt, sin_tt),
                        mode=mode,
                        kv_state=None if stack_kv is None else stack_kv[i],
                    )
                return h
            finally:
                ttnn.deallocate(cos_tt)
                ttnn.deallocate(sin_tt)

        if position_embeddings is None:
            raise ValueError("position_embeddings is required when use_device_rotary_embedding_table=False.")
        cos_tt, sin_tt = upload_mistral4_rotary_cos_sin_to_mesh(self.device, position_embeddings)
        try:
            h = hidden_11SH
            for i, block in enumerate(self.blocks):
                h = block(
                    h,
                    position_ids=position_ids,
                    position_embeddings_tt=(cos_tt, sin_tt),
                    mode=mode,
                    kv_state=None if stack_kv is None else stack_kv[i],
                )
            return h
        finally:
            ttnn.deallocate(cos_tt)
            ttnn.deallocate(sin_tt)


# --- Layer 0 aliases (backward compatible imports / docs) ---


class TtMistral4DecoderLayer0RmsNormStub(TtMistral4DecoderLayerRmsNormStub):
    """Same as :class:`TtMistral4DecoderLayerRmsNormStub` with ``layer_idx=0``."""

    def __init__(
        self,
        device,
        state_dict: dict,
        *,
        weight_key: str,
        weight_dtype=ttnn.bfloat16,
        eps: float | None = None,
    ):
        super().__init__(device, state_dict, layer_idx=0, weight_key=weight_key, weight_dtype=weight_dtype, eps=eps)


class TtMistral4DecoderLayer0InputNormStub(TtMistral4DecoderLayerInputNormStub):
    """Same as :class:`TtMistral4DecoderLayerInputNormStub` with ``layer_idx=0``."""

    def __init__(self, device, state_dict: dict, **kwargs):
        super().__init__(device, state_dict, layer_idx=0, **kwargs)


class TtMistral4DecoderLayer0PostAttnNormStub(TtMistral4DecoderLayerPostAttnNormStub):
    """Same as :class:`TtMistral4DecoderLayerPostAttnNormStub` with ``layer_idx=0``."""

    def __init__(self, device, state_dict: dict, **kwargs):
        super().__init__(device, state_dict, layer_idx=0, **kwargs)


class TtMistral4DecoderLayer0AttnPrefillBlock(TtMistral4DecoderLayerAttnPrefillBlock):
    """Same as :class:`TtMistral4DecoderLayerAttnPrefillBlock` with ``layer_idx=0``."""

    def __init__(self, device, state_dict: dict, text_config, *, weight_dtype=ttnn.bfloat16, eps: float | None = None):
        super().__init__(device, state_dict, text_config, layer_idx=0, weight_dtype=weight_dtype, eps=eps)


class TtMistral4DecoderLayer0(TtMistral4DecoderLayer):
    """Same as :class:`TtMistral4DecoderLayer` with ``layer_idx=0``."""

    def __init__(
        self,
        device,
        state_dict: dict,
        text_config,
        *,
        weight_dtype=ttnn.bfloat16,
        eps: float | None = None,
        tt_ccl=None,
        use_ttnn_moe: bool = False,
        moe_hf_torch_routing: bool = False,
        moe_shard_routed_experts: bool | None = None,
    ):
        super().__init__(
            device,
            state_dict,
            text_config,
            layer_idx=0,
            weight_dtype=weight_dtype,
            eps=eps,
            tt_ccl=tt_ccl,
            use_ttnn_moe=use_ttnn_moe,
            moe_hf_torch_routing=moe_hf_torch_routing,
            moe_shard_routed_experts=moe_shard_routed_experts,
        )
