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
from models.experimental.voxtraltts.utils.mesh import (
    voxtral_replicate_mesh_mapper,
    voxtral_tp_shard_last_dim_mapper,
)
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
    args = text.inner.args
    norm_kwargs = {
        "args": args,
        "tt_ccl": text.inner.tt_ccl,
        "prefetcher": text.inner.prefetcher,
    }
    remapped = remap_voxtral_text_state_dict(state_dict)
    for layer_idx, layer_block in enumerate(text.inner.layers):
        layer_block.attention_norm = VoxtralTextRMSNorm(
            device=mesh_device,
            dim=dim,
            state_dict=remapped,
            weight_key=f"layers.{layer_idx}.attention_norm",
            eps=norm_eps,
            ag_config_key="ATTN_LN_AG_CONFIG",
            **norm_kwargs,
        )
        layer_block.ff_norm = VoxtralTextRMSNorm(
            device=mesh_device,
            dim=dim,
            state_dict=remapped,
            weight_key=f"layers.{layer_idx}.ffn_norm",
            eps=norm_eps,
            ag_config_key="FFN_LN_AG_CONFIG",
            **norm_kwargs,
        )
    text.inner.norm = VoxtralTextRMSNorm(
        device=mesh_device,
        dim=dim,
        state_dict=remapped,
        weight_key="norm",
        eps=norm_eps,
        **norm_kwargs,
    )


def _decode_activation_dtype(args) -> ttnn.DataType | None:
    return args.decoders_optimizations.get_tensor_dtype(decoder_id=0, tensor=TensorGroup.ACTIVATION)


def _decode_replicated_embed_mem_cfg(args) -> ttnn.MemoryConfig:
    """Width-sharded decode input layout for replicated ``[1, 1, 1, dim]`` embeddings.

    ``get_residual_mem_config(DECODE)`` assumes tensor-parallel activations with width
    ``dim // num_devices`` per chip. Prompt/MM embeds are replicated with the full ``dim``
    on every device and must shard as ``dim // num_cores`` instead.
    """
    grid = args.dram_shard_core_grid_for_k(args.dim)
    return ttnn.create_sharded_memory_config(
        (args.tile_padded_batch_rows, args.dim // grid.num_cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


class VoxtralTTTextModel:
    """Direct tt_transformers Transformer wrapper for Voxtral text stack."""

    def __init__(self, inner_transformer: Transformer) -> None:
        self.inner = inner_transformer
        # Pre-compute static decode configs once — args and prefetcher are fixed after init.
        self._decode_mem_cfg = inner_transformer.args.get_residual_mem_config(Mode.DECODE, inner_transformer.prefetcher)
        # TP text keeps column-sharded ``[*, local_dim]`` activations; only 1×1 uses replicated full dim.
        self._decode_embed_input_mem_cfg = (
            self._decode_mem_cfg
            if inner_transformer.args.num_devices > 1
            else _decode_replicated_embed_mem_cfg(inner_transformer.args)
        )
        self._lm_norm_cfg = inner_transformer.args.get_norm_config("lm_head", Mode.DECODE, inner_transformer.prefetcher)

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
        tt_embd, *rest = self.inner.prepare_inputs_prefill(*args, **kwargs)
        if getattr(self.inner.args, "prefill_activations_l1", False):
            target = self.inner.args.get_residual_mem_config(Mode.PREFILL, self.inner.prefetcher)
            if tt_embd.memory_config() != target:
                tt_embd = ttnn.to_memory_config(tt_embd, target)
        return (tt_embd, *rest)

    def prepare_inputs_decode(self, *args, **kwargs):
        return self.inner.prepare_inputs_decode(*args, **kwargs)

    def switch_mode(self, mode):
        return self.inner.switch_mode(mode)

    def forward(self, *args, **kwargs):
        return self.inner.forward(*args, **kwargs)

    def _decode_single_token_to_tt(
        self,
        x_embed: "torch.Tensor | ttnn.Tensor",
        pos_idx: int,
        *,
        collect_layer_hiddens: bool = False,
        page_table: "ttnn.Tensor | None" = None,
    ) -> "ttnn.Tensor | tuple[ttnn.Tensor, dict[str, torch.Tensor]]":
        """Core TT decode step: embed → device x_norm (post-norm, no host readback).

        Accepts either a CPU ``torch.Tensor`` (AR loop from audio codes) or an already
        on-device ``ttnn.Tensor`` (e.g. from ``_audio_codes_to_mm_embed_device``).

        Shared by ``decode_step_from_embeds`` (wraps + host gather) and ``forward_device_resident``
        (keeps hidden on device for acoustic model via ``forward_from_tt``).
        ``collect_layer_hiddens`` is a debug flag that causes per-layer host readbacks.
        """
        dim = self.inner.args.dim
        activation_dtype = _decode_activation_dtype(self.inner.args) or ttnn.bfloat16

        current_pos_t = torch.tensor([pos_idx], dtype=torch.int64)
        rot_mats_global = self.inner.rope_setup.get_rot_mats(current_pos_t)
        rot_mats_local = (
            self.inner.rope_local_setup.get_rot_mats(current_pos_t) if hasattr(self.inner, "rope_local_setup") else None
        )
        current_pos_tt = ttnn.from_torch(
            current_pos_t,
            device=self.inner.mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.inner.mesh_device,
                dims=(None, None),
                mesh_shape=self.inner.args.cluster_shape,
            ),
        )

        if isinstance(x_embed, ttnn.Tensor):
            # Embed already on device (from mm_audio_encode_tokens_summed_forward).
            # Convert layout + memory config in one call — no host round-trip.
            embed_mem_cfg = self._decode_embed_input_mem_cfg
            if x_embed.layout != ttnn.TILE_LAYOUT:
                x_tt = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT, memory_config=embed_mem_cfg)
            else:
                x_tt = ttnn.to_memory_config(x_embed, embed_mem_cfg)
        else:
            x_4d = x_embed.reshape(1, 1, 1, dim).to(dtype=torch.bfloat16).contiguous()
            mesh_mapper = voxtral_tp_shard_last_dim_mapper(self.inner.mesh_device, self.inner.args.cluster_shape)
            if mesh_mapper is None:
                mesh_mapper = voxtral_replicate_mesh_mapper(self.inner.mesh_device)
            x_tt = ttnn.from_torch(
                x_4d,
                device=self.inner.mesh_device,
                dtype=activation_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self._decode_embed_input_mem_cfg,
                mesh_mapper=mesh_mapper,
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

        x_norm = self.inner.norm(x_tt, mode=Mode.DECODE, norm_config=self._lm_norm_cfg)
        ttnn.deallocate(x_tt)

        # Free per-step rope/position tensors explicitly. These are function-locals that
        # GC would eventually reclaim, but during multi-thousand-frame generation runs
        # delayed GC can let device memory creep; deallocate deterministically instead.
        for rm in (rot_mats_global, rot_mats_local):
            if rm is None:
                continue
            for t in rm:
                if isinstance(t, ttnn.Tensor) and t.is_allocated():
                    ttnn.deallocate(t)
        if current_pos_tt.is_allocated():
            ttnn.deallocate(current_pos_tt)

        if collect_layer_hiddens:
            return x_norm, layer_hiddens
        return x_norm

    def hidden_tt_to_torch(self, x_norm_tt: "ttnn.Tensor") -> torch.Tensor:
        """Gather device hidden ``[1,1,1,dim]`` → CPU ``[dim]`` bfloat16.

        Use this whenever a caller needs a torch tensor from a TT hidden returned by
        ``prefill_from_embeds`` — e.g. for PCC comparison or feeding the acoustic model.
        """
        host = self.inner.concat_host_output(x_norm_tt)
        return host[0, 0, 0, : self.inner.args.dim].to(dtype=torch.bfloat16)

    # Flow: torch input → ONE ttnn.from_torch at the top → TT slicing inside the loop
    # → TT throughout all layer computation → returned as ttnn.Tensor.
    # Convert to torch only when the caller explicitly needs it (PCC, acoustic model):
    #   hidden_torch = model.hidden_tt_to_torch(hidden_tt); ttnn.deallocate(hidden_tt)
    def prefill_from_embeds(
        self,
        inputs_embeds: "torch.Tensor | ttnn.Tensor",
        start_pos: int = 0,
        *,
        collect_layer_hiddens: bool = False,
        page_table: "ttnn.Tensor | None" = None,
    ) -> "ttnn.Tensor | tuple[ttnn.Tensor, dict[str, torch.Tensor]]":
        """Prefill via per-token decode (KV-safe). Returns device ``x_norm`` as ``ttnn.Tensor``.

        Uploads the full ``[S, dim]`` embedding sequence to TT in ONE ``ttnn.from_torch``
        call, then slices per token on device — no per-step host→device DMA inside the loop.

        The voice-injection scatter (building the ``[S, dim]`` tensor before calling this) is
        currently done on CPU; porting it to TT (``ttnn.embedding`` + ``ttnn.where``) is a
        follow-on task.
        """
        dim = self.inner.args.dim
        # Per-device hidden width: full ``dim`` on 1x1, ``dim // num_devices`` when the embeds are
        # column-sharded for tensor-parallel text. Slicing/reshaping must use this local width, not
        # the full ``dim`` (otherwise a 1x4 mesh slices [..., 768] tensors to [..., 3072] and TT_FATALs).
        embed_dim = int(inputs_embeds.shape[-1])
        activation_dtype = _decode_activation_dtype(self.inner.args) or ttnn.bfloat16

        # ── SINGLE TT UPLOAD ─────────────────────────────────────────────────
        # One ttnn.from_torch for the full sequence — no per-step DMA.
        if isinstance(inputs_embeds, ttnn.Tensor):
            S = int(inputs_embeds.shape[0])
            owns_embeds_tt = False  # caller owns the original; updated below if we create new tensors
            # Need ROW_MAJOR for non-tile-aligned per-token slicing.
            if inputs_embeds.layout != ttnn.ROW_MAJOR_LAYOUT:
                embeds_tt = ttnn.to_layout(inputs_embeds, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                owns_embeds_tt = True  # new tensor created; we own it
            else:
                embeds_tt = inputs_embeds
            # Ensure shape is [S, 1, 1, dim] for uniform slice coordinates.
            if len(embeds_tt.shape) != 4:
                reshaped = ttnn.reshape(embeds_tt, (S, 1, 1, embed_dim))
                if owns_embeds_tt and embeds_tt.is_allocated():
                    ttnn.deallocate(embeds_tt)  # free intermediate layout-converted tensor
                embeds_tt = reshaped
                owns_embeds_tt = True  # new tensor created; we own it
        else:
            S = int(inputs_embeds.shape[0])
            owns_embeds_tt = True
            embeds_4d = inputs_embeds.reshape(S, 1, 1, embed_dim).to(dtype=torch.bfloat16).contiguous()
            embeds_tt = ttnn.from_torch(
                embeds_4d,
                device=self.inner.mesh_device,
                dtype=activation_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,  # ROW_MAJOR allows non-tile-aligned token slicing
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.inner.mesh_device),
            )
        # embeds_tt: [S, 1, 1, dim] ROW_MAJOR DRAM

        # ── TT-ONLY LOOP ─────────────────────────────────────────────────────
        last_hidden_tt: ttnn.Tensor | None = None
        layer_hiddens: dict[str, torch.Tensor] = {}

        for i in range(S):
            is_last = i == S - 1
            collect_this = collect_layer_hiddens and is_last

            # Slice token i on device — zero host round-trips for the embedding data.
            embed_i_rm = ttnn.slice(embeds_tt, (i, 0, 0, 0), (i + 1, 1, 1, embed_dim))
            # Convert to TILE_LAYOUT + decode shard in one call.
            embed_i_tt = ttnn.to_layout(embed_i_rm, ttnn.TILE_LAYOUT, memory_config=self._decode_embed_input_mem_cfg)
            if embed_i_rm.is_allocated():
                ttnn.deallocate(embed_i_rm)

            # Position/rope: pos_idx is a Python int — a 1-element scalar index (4 bytes),
            # not embedding data. This tiny transfer is unavoidable.
            current_pos_t = torch.tensor([start_pos + i], dtype=torch.int64)
            rot_mats_global = self.inner.rope_setup.get_rot_mats(current_pos_t)
            rot_mats_local = (
                self.inner.rope_local_setup.get_rot_mats(current_pos_t)
                if hasattr(self.inner, "rope_local_setup")
                else None
            )
            current_pos_tt = ttnn.from_torch(
                current_pos_t,
                device=self.inner.mesh_device,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.inner.mesh_device,
                    dims=(None, None),
                    mesh_shape=self.inner.args.cluster_shape,
                ),
            )

            # Layer loop — pure TT. No host readback except the debug collect path.
            x_tt = embed_i_tt
            layer_hiddens_step: dict[str, torch.Tensor] = {}
            for j, layer in enumerate(self.inner.layers):
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
                if collect_this:
                    host_layer = self.inner.concat_host_output(x_tt)
                    layer_hiddens_step[f"layer.{j}"] = host_layer[0, 0, 0, :dim].to(dtype=torch.bfloat16)

            x_norm_tt = self.inner.norm(x_tt, mode=Mode.DECODE, norm_config=self._lm_norm_cfg)
            ttnn.deallocate(x_tt)

            if collect_this:
                layer_hiddens = layer_hiddens_step
                layer_hiddens["layer.final_norm"] = self.hidden_tt_to_torch(x_norm_tt)

            if last_hidden_tt is not None and last_hidden_tt.is_allocated():
                ttnn.deallocate(last_hidden_tt)
            last_hidden_tt = x_norm_tt

            # Free per-token rope/position tensors so they do not accumulate across
            # the (potentially thousands of) prefill steps for a long prompt.
            for rm in (rot_mats_global, rot_mats_local):
                if rm is None:
                    continue
                for t in rm:
                    if isinstance(t, ttnn.Tensor) and t.is_allocated():
                        ttnn.deallocate(t)
            if current_pos_tt.is_allocated():
                ttnn.deallocate(current_pos_tt)

        if owns_embeds_tt and embeds_tt.is_allocated():
            ttnn.deallocate(embeds_tt)

        assert last_hidden_tt is not None
        if collect_layer_hiddens:
            return last_hidden_tt, layer_hiddens
        return last_hidden_tt

    # Torch-return wrapper: callers that feed the hidden directly to the acoustic model
    # (which takes a torch tensor) should use this. TT-input callers use decode_step_from_embeds_tt.
    def decode_step_from_embeds(
        self,
        x_embed: torch.Tensor,
        current_pos_idx: int,
        *,
        collect_layer_hiddens: bool = False,
    ) -> "torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]":
        """One decode step from a CPU embedding; returns post-norm ``[dim]`` hidden (pre-LM-head)."""
        if collect_layer_hiddens:
            x_norm_tt, layer_hiddens = self._decode_single_token_to_tt(
                x_embed, current_pos_idx, collect_layer_hiddens=True
            )
        else:
            x_norm_tt = self._decode_single_token_to_tt(x_embed, current_pos_idx)
            layer_hiddens = {}

        # concat_host_output is the required multi-device gather (sharded → host tensor).
        # Acoustic model takes torch; TT-resident callers use decode_step_from_embeds_tt.
        hidden = self.hidden_tt_to_torch(x_norm_tt)
        ttnn.deallocate(x_norm_tt)
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
        activation_dtype = _decode_activation_dtype(self.inner.args)
        embed_mem_cfg = self._decode_embed_input_mem_cfg
        if activation_dtype is not None and x_embed_tt.dtype != activation_dtype:
            x_tt = ttnn.to_memory_config(x_embed_tt, embed_mem_cfg, activation_dtype)
        else:
            x_tt = ttnn.to_memory_config(x_embed_tt, embed_mem_cfg)

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

        x_norm = self.inner.norm(x_tt, mode=Mode.DECODE, norm_config=self._lm_norm_cfg)
        ttnn.deallocate(x_tt)
        return x_norm
