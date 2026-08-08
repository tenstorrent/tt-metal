# SPDX-FileCopyrightText: (c) 2026 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
"""
TTNN modules for Higgs Audio v2.

Higgs Audio v2 = Llama-3.2-3B-Instruct fine-tuned with an extra audio-side
MLP and norms per layer (DualFFN), plus a separate audio embedding table and
audio LM head. Attention itself is shared between text and audio tokens.

Layer (HiggsDualFFNBlock):
    x_text = x_audio = input
    x = attention(attention_norm(x_text), audio_attention_norm(x_audio))   # shared
    x_text_ff  = feed_forward(ffn_norm(x))
    x_audio_ff = audio_feed_forward(audio_ffn_norm(x))
    out = x + (text_branch * (1-mask) + audio_branch * mask)

For Spike 3+4 we only verify construction. Forward pass arrives in Spike 6.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.attention import Attention
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.mlp import MLP


class HiggsDualFFNBlock(LightweightModule):
    """Single transformer block with shared attention + dual norms + dual FFNs."""

    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        state_dict,
        layer_num: int,
        dtype,
        transformation_mats,
        weight_cache_path=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.layer_num = layer_num
        self.dim = args.dim

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
        )

        self.attention_norm = self._make_norm(state_dict, "attention_norm", layer_num, "ATTN_LN_AG_CONFIG")
        self.audio_attention_norm = self._make_norm(state_dict, "audio_attention_norm", layer_num, "ATTN_LN_AG_CONFIG")
        self.ffn_norm = self._make_norm(state_dict, "ffn_norm", layer_num, "FFN_LN_AG_CONFIG")
        self.audio_ffn_norm = self._make_norm(state_dict, "audio_ffn_norm", layer_num, "FFN_LN_AG_CONFIG")

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

    def forward(
        self,
        x,
        current_pos,
        rot_mats,
        user_id: int = 0,
        mode: Mode = Mode.DECODE,
        is_audio_token: bool = False,
        kv_cache=None,
        audio_mask=None,
        text_mask=None,
    ):
        """DualFFN block forward.

        Decode (single token of known type): ``is_audio_token`` picks the branch.
        Prefill with mixed text+audio tokens (voice cloning): pass per-position
        ``audio_mask``/``text_mask`` ([1,1,S,1], audio=1/text=1 resp.) to blend
        both branches — text tokens take the text norm/FFN, audio tokens the audio
        norm/FFN: ``out = x + (text_branch*text_mask + audio_branch*audio_mask)``.
        Attention is shared. Text-only prefill leaves audio_mask=None (single
        text branch, unchanged).
        """
        attn_norm_config = self.args.get_norm_config("attn", mode, None)
        ff_norm_config = self.args.get_norm_config("ff", mode, None)
        skip_mem_cfg = self.args.get_residual_mem_config(mode, None)
        residual = x

        if audio_mask is not None:
            t_in = self.attention_norm(x, mode, norm_config=attn_norm_config)
            a_in = self.audio_attention_norm(x, mode, norm_config=attn_norm_config)
            attn_in = ttnn.add(ttnn.mul(t_in, text_mask), ttnn.mul(a_in, audio_mask))
            ttnn.deallocate(t_in)
            ttnn.deallocate(a_in)
        else:
            norm_in = self.audio_attention_norm if is_audio_token else self.attention_norm
            attn_in = norm_in(x, mode, norm_config=attn_norm_config)

        attn_out = self.attention.forward(attn_in, current_pos, rot_mats, user_id, mode, kv_cache=kv_cache)
        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
        h = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg)
        residual = h

        if audio_mask is not None:
            t_ff = self.feed_forward.forward(self.ffn_norm(h, mode, norm_config=ff_norm_config), mode)
            a_ff = self.audio_feed_forward.forward(self.audio_ffn_norm(h, mode, norm_config=ff_norm_config), mode)
            ff_out = ttnn.add(ttnn.mul(t_ff, text_mask), ttnn.mul(a_ff, audio_mask))
            ttnn.deallocate(t_ff)
            ttnn.deallocate(a_ff)
        else:
            norm_ff = self.audio_ffn_norm if is_audio_token else self.ffn_norm
            ff = self.audio_feed_forward if is_audio_token else self.feed_forward
            ff_out = ff.forward(norm_ff(h, mode, norm_config=ff_norm_config), mode)

        out = ttnn.add(residual, ttnn.to_memory_config(ff_out, skip_mem_cfg), memory_config=skip_mem_cfg)
        return out

    def _make_norm(self, state_dict, weight_key: str, layer_num: int, ag_config_key: str):
        rms = RMSNorm(
            device=self.mesh_device,
            dim=self.args.dim,
            eps=self.args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix=None,
            weight_cache_path=None,
            weight_dtype=ttnn.bfloat16,
            weight_key=weight_key,
            layer_num=layer_num,
            is_distributed=self.args.is_distributed_norm,
            add_unit_offset=self.args.rms_norm_add_unit_offset,
            ccl_topology=self.args.ccl_topology(),
            tt_ccl=self.tt_ccl,
        )
        return DistributedNorm(
            rms,
            self.args,
            tt_ccl=self.tt_ccl,
            TG=self.args.is_galaxy,
            ag_config_key=ag_config_key,
        )


class HiggsAudioEmbedding(LightweightModule):
    """Audio codebook embedding table [num_codebooks * codebook_size, hidden_size].

    Indexed by stacked codebook IDs. Forward stays host-friendly: pure
    ``ttnn.embedding`` lookup. Per-step embed-and-sum-across-codebooks is the
    caller's job (top-level model).
    """

    def __init__(self, mesh_device, args, state_dict, dtype):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        torch_w = state_dict["audio_tok_embeddings.weight"].unsqueeze(0).unsqueeze(0)
        self.weights = ttnn.as_tensor(
            torch_w,
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device=mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=args.get_model_config()["EMB_WEIGHTS_MEMCFG"],
            cache_file_name=None,
        )

    def forward(self, x: ttnn.Tensor, memory_config=None) -> ttnn.Tensor:
        return ttnn.embedding(x, self.weights, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)


class HiggsAudioLMHead(LightweightModule):
    """Audio LM head: linear map from hidden (=dim) to audio_vocab_size (=num_codebooks * codebook_size).

    Kept deliberately simple — single bf16 matmul against the audio_output
    weight from the remapped state dict. Per-codebook reshape is done at the
    call site (top-level model decode path).
    """

    def __init__(self, mesh_device, args, state_dict, dtype=ttnn.bfloat8_b):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.dim = args.dim
        self.audio_vocab_size = args.audio_vocab_size

        # Keep audio_lm_head weights in bf16 + HiFi4 compute for accuracy.
        # 8208-vocab matmul is small enough that bf16 fits comfortably.
        torch_w = torch.transpose(state_dict["audio_output.weight"], -2, -1).contiguous()  # [dim, audio_vocab]
        self.weights = ttnn.as_tensor(
            torch_w.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=None,
        )
        # Reuse the HiFi4 kernel config from tt_transformers args.
        self.compute_kernel_config = getattr(args, "compute_kernel_config_hifi4", None)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.matmul(
            x,
            self.weights,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )


class HiggsAudioTTModel(LightweightModule):
    """Top-level Higgs Audio v2 model on TTNN.

    Holds text + audio embeddings, 28 dual-FFN blocks, final norm, dual LM
    heads. Spike 5 verifies construction only; the forward pass arrives in
    Spike 6.
    """

    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        state_dict,
        transformation_mats,
        dtype=ttnn.bfloat8_b,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl

        self.text_embedding = Embedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=None,
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # required by ROW_MAJOR embedding
        )
        self.audio_embedding = HiggsAudioEmbedding(
            mesh_device=mesh_device, args=args, state_dict=state_dict, dtype=ttnn.bfloat16
        )

        self.layers = [
            HiggsDualFFNBlock(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                state_dict=state_dict,
                layer_num=i,
                dtype=dtype,
                transformation_mats=transformation_mats,
                weight_cache_path=None,
            )
            for i in tqdm(range(args.n_layers), desc="Higgs blocks")
        ]

        # Final norm uses 'norm' weight key.
        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,
                weight_cache_path=None,
                weight_dtype=ttnn.bfloat16,
                weight_key="norm",
                layer_num=None,
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

        # Text LM head (vocab_size=128256). Reuses stock LMHead, which keys
        # off ``args.vocab_size`` and ``args.padded_vocab_size``.
        self.text_lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix="text_",  # LMHead appends "output.weight"
            weight_cache_path=None,
            max_columns_per_device=args.max_columns_per_device_lm_head,
        )

        # Audio LM head (vocab_size=8208). Custom minimal linear to avoid
        # threading audio_vocab_size through args.padded_vocab_size.
        self.audio_lm_head = HiggsAudioLMHead(mesh_device=mesh_device, args=args, state_dict=state_dict, dtype=dtype)

    def _audio_embed_host_table(self):
        """Lazy host copy of the audio codebook embedding table [audio_vocab, dim]."""
        if not hasattr(self, "_audio_embed_host"):
            self._audio_embed_host = (
                ttnn.to_torch(
                    self.audio_embedding.weights,
                    mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
                )
                .reshape(-1, self.args.dim)[: self.args.audio_vocab_size]
                .float()
            )
        return self._audio_embed_host

    def prefill_text(
        self,
        input_ids,
        rope_setup,
        start_pos: int = 0,
        audio_input_ids=None,
        audio_input_ids_mask=None,
        audio_token_id: int = 128016,
        user_id: int = 0,
    ):
        """Run prefill over a 1-D sequence of input ids.

        Args:
            input_ids: torch.LongTensor of shape [S]. Will be padded to a
                multiple of 128 internally.
            rope_setup: HfRotarySetup instance used to slice prefill cos/sin.
            start_pos: starting position offset (0 for fresh context).
            user_id: KV-cache batch row to fill. Batched serving prefills each
                stream into its own row (user_id=0..B-1); the per-row cache is
                then read back independently during batched decode.
            audio_input_ids: optional [1, F, K] reference-audio codes (voice
                cloning). Embedded (sum over codebooks) and scattered into the
                ``audio_token_id`` placeholder positions of ``input_ids`` — the
                same merge HF's HiggsAudioV2Model.forward does via masked_scatter.
            audio_input_ids_mask: optional [1, F] bool marking valid audio frames.

        Returns:
            (text_logits, last_audio_logits). KV cache inside each block is
            populated as a side-effect.
        """
        S = input_ids.shape[0]
        # Prefill paths in tt_transformers want seqlen multiple of 128 (4 tiles).
        chunk = 128
        pad = (chunk - (S % chunk)) % chunk
        S_padded = S + pad
        # The KV-cache prefill fill shards seq_len/8 tiles over the 8x8 grid; once
        # the shard height (seq_len/8) exceeds one tile it must be 32-aligned
        # (e.g. 384 -> 48 fatals). Bump such cases to the next multiple of 256.
        # Short prompts (S_padded <= 256, height <= 32) keep their validated
        # 128-padding so the accuracy gate is unchanged.
        if S_padded >= 256 and (S_padded // 8) % 32 != 0:
            S_padded = ((S_padded + 255) // 256) * 256
            pad = S_padded - S
        if pad:
            input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=0)

        # Embed on device (ROW_MAJOR → TILE).
        tokens = ttnn.from_torch(
            input_ids.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        h = self.text_embedding(tokens)  # [S_padded, dim] ROW_MAJOR

        # Voice cloning: splice reference-audio embeddings into the placeholder
        # positions (host-side masked_scatter, matching HF semantics 1:1).
        if audio_input_ids is not None:
            K = self.args.audio_num_codebooks
            cb = self.args.audio_codebook_size
            table = self._audio_embed_host_table()  # [audio_vocab, dim]
            frames = audio_input_ids[0]  # [F, K]
            if audio_input_ids_mask is not None:
                frames = frames[audio_input_ids_mask[0].bool()]  # [n_valid, K]
            offsets = torch.arange(K, dtype=torch.long) * cb
            audio_embeds = table[(frames.long() + offsets)].sum(dim=1)  # [n_valid, dim]
            ph_pos = (input_ids == audio_token_id).nonzero(as_tuple=True)[0]
            # HF merges via masked_scatter: the placeholder positions are filled
            # with the leading audio_embeds in order (the delay-pattern overhang
            # tail, n_valid - n_placeholders frames, is simply not consumed).
            n_ph = ph_pos.numel()
            assert audio_embeds.shape[0] >= n_ph, f"placeholders {n_ph} > audio frames {audio_embeds.shape[0]}"
            audio_embeds = audio_embeds[:n_ph]
            h_host = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[:S_padded].float()
            h_host[ph_pos] = audio_embeds.to(h_host.dtype)
            h = ttnn.from_torch(
                h_host.to(torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        h = ttnn.unsqueeze_to_4D(h)  # [1, 1, S_padded, dim]
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)

        # Prefill rope mats: slice cos/sin for [start_pos:start_pos+S_padded]
        mat_len = rope_setup.cos_matrix_prefill.shape[2]
        slice_end = min(mat_len, start_pos + S_padded)
        cos_slice = rope_setup.cos_matrix_prefill[:, :, start_pos:slice_end, :]
        sin_slice = rope_setup.sin_matrix_prefill[:, :, start_pos:slice_end, :]
        rot_mats = [cos_slice, sin_slice]

        # DualFFN routing mask for prefill: audio (placeholder) positions take the
        # audio norm/FFN branch, text positions the text branch. Only needed when
        # the prompt contains audio tokens (voice cloning); text-only prompts keep
        # the single text-branch path (audio_mask=None).
        audio_mask_dev = text_mask_dev = None
        if audio_input_ids is not None:
            am = (input_ids == audio_token_id).to(torch.float32).view(1, 1, S_padded, 1)
            audio_mask_dev = ttnn.from_torch(
                am.to(torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            text_mask_dev = ttnn.from_torch(
                (1.0 - am).to(torch.bfloat16),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        for blk in self.layers:
            h = blk(
                h,
                current_pos=None,
                rot_mats=rot_mats,
                user_id=user_id,
                mode=Mode.PREFILL,
                is_audio_token=False,
                audio_mask=audio_mask_dev,
                text_mask=text_mask_dev,
            )

        # Final norm + LM head, sliced to last (non-padding) token.
        last_idx = S - 1  # logical last position
        h_last = ttnn.slice(h, (0, 0, last_idx, 0), (1, 1, last_idx + 1, h.shape[-1]))
        # Pad the slice to tile height (32) so subsequent tile-layout ops are happy.
        h_last = ttnn.pad(h_last, padding=[(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)

        final_norm_config = self.args.get_norm_config("lm_head", Mode.PREFILL, None)
        h_last = self.norm(h_last, Mode.PREFILL, norm_config=final_norm_config)

        # Audio LM head on last hidden — this is the first audio prediction.
        # Pull to host so the caller can argmax/delay-process it directly.
        audio_logits = self.audio_lm_head(h_last)
        audio_logits_torch = ttnn.to_torch(audio_logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        # Take row 0 (real last-token position), slice to real audio vocab,
        # reshape into per-codebook logits.
        K = self.args.audio_num_codebooks
        cb = self.args.audio_codebook_size
        last_audio_logits = (
            audio_logits_torch.reshape(audio_logits_torch.shape[-2], -1)[0, : K * cb].reshape(K, cb).float()
        )

        # Bridge: LMHead in PREFILL wants its input in args.get_lm_head_input_mem_config.
        # Match tt_transformers.tt.model.ttnn_decode_forward's pattern (line ~944).
        lm_head_input_mem_cfg = self.args.get_lm_head_input_mem_config(Mode.PREFILL, None)
        if lm_head_input_mem_cfg.is_sharded():
            h_last = ttnn.interleaved_to_sharded(h_last, lm_head_input_mem_cfg)
        logits = self.text_lm_head(h_last)
        return logits, last_audio_logits
        # logits shape: [1, 1, 32, padded_vocab_size]; row 0 is the real last token.

    def decode_step_audio(self, audio_token_ids: torch.Tensor, current_pos, rot_mats):
        """One audio decode step.

        Args:
            audio_token_ids: torch.LongTensor of shape ``[num_codebooks]`` — the
                eight codebook IDs from the previous step (each in
                ``[0, codebook_size)``). Stacked-table offset is applied here.
            current_pos: ttnn.Tensor int32, sharded across mesh.
            rot_mats: result of ``rope_setup.get_rot_mats(...)``.

        Returns:
            Per-codebook audio logits as a ``torch.Tensor`` of shape
            ``[num_codebooks, codebook_size]`` (already moved to host).
        """
        K = self.args.audio_num_codebooks
        cb_size = self.args.audio_codebook_size

        # Pre-add codebook offsets on host (matches PR #40907 / HF native
        # HiggsAudioV2Embeddings.forward exactly): codebook c maps to slots
        # [c*cb_size, (c+1)*cb_size). Sum-over-codebooks is then done on host
        # at lookup time by *averaging* via a single embedding lookup of K
        # rows tiled into one tensor.
        offsets = torch.arange(K, dtype=torch.long) * cb_size
        stacked_ids = (audio_token_ids.long() + offsets).to(torch.int32)  # [K]

        # Lookup K embeddings on host via gather, then sum on host, then
        # send the [1, 1, 1, dim] tensor to device. This bypasses every
        # ttnn.add/slice tile dance and matches HF semantics 1:1.
        embed_weight = self.audio_embedding.weights  # ttnn tensor on device, [1, 1, V, dim]
        # Pull weight to host once — small (~25MB). For Phase 1 accuracy this
        # is fine; Phase 2 traces this back to device.
        if not hasattr(self, "_audio_embed_host"):
            self._audio_embed_host = ttnn.to_torch(
                embed_weight, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
            ).reshape(-1, self.args.dim)[: self.args.audio_vocab_size]
        host_table = self._audio_embed_host
        host_vec = host_table[stacked_ids.long()].sum(dim=0)  # [dim], fp16/bf16
        host_vec = host_vec.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1,1,1,dim]

        h = ttnn.from_torch(
            host_vec.to(torch.bfloat16),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # h shape: [1, 1, 1, 3072]. Move to the residual memcfg used by blocks.
        skip_mem_cfg = self.args.get_residual_mem_config(Mode.DECODE, None)
        h = ttnn.to_memory_config(h, skip_mem_cfg)

        # Forward through 28 blocks on the audio branch.
        for blk in self.layers:
            h = blk(h, current_pos, rot_mats, mode=Mode.DECODE, is_audio_token=True)

        final_norm_config = self.args.get_norm_config("lm_head", Mode.DECODE, None)
        h = self.norm(h, Mode.DECODE, norm_config=final_norm_config)
        logits = self.audio_lm_head(h)  # [1, 1, 1, 8208] or padded equivalent
        # Pull to host, reshape per codebook, return.
        logits_torch = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        logits_torch = logits_torch.reshape(-1)[: K * cb_size].reshape(K, cb_size).float()
        return logits_torch

    def decode_step_text(self, token_id, current_pos, rot_mats):
        """One decode step on the text branch.

        Returns: text-vocab logits as ttnn.Tensor of shape
            [1, 1, 1, padded_vocab_size].
        """
        h = self.text_embedding(token_id)
        skip_mem_cfg = self.args.get_residual_mem_config(Mode.DECODE, None)
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
        h = ttnn.to_memory_config(h, skip_mem_cfg)
        for blk in self.layers:
            h = blk(h, current_pos, rot_mats, mode=Mode.DECODE, is_audio_token=False)
        # Final norm + LM head must use the *lm_head* norm config so the
        # norm output is sharded the way LMHead expects (matches the bridge
        # in tt_transformers.tt.model.ttnn_decode_forward).
        final_norm_config = self.args.get_norm_config("lm_head", Mode.DECODE, None)
        h = self.norm(h, Mode.DECODE, norm_config=final_norm_config)
        return self.text_lm_head(h, Mode.DECODE)


__all__ = [
    "HiggsDualFFNBlock",
    "HiggsAudioEmbedding",
    "HiggsAudioLMHead",
    "HiggsAudioTTModel",
]
