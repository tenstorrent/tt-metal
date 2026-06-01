# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS Code Predictor: 5-layer Transformer that generates CB1-CB15 from CB0.

Architecture (from HF source):
    Per audio frame, the Code Predictor receives:
        1. Talker hidden state [B, 1, 2048] for this frame
        2. CB0 token (embedded via Talker's codec_embedding → [B, 1, 2048])
    These are concatenated → [B, 2, 2048], projected via small_to_mtp_projection → [B, 2, 1024],
    then fed through a 5-layer Transformer to predict CB1.

    For CB2..CB15, each step:
        a. Embed the previous token via codec_embedding[step-1] → [B, 1, 2048]
        b. Project to 1024-dim via small_to_mtp_projection
        c. Decode step through Transformer (with KV cache)
        d. Predict next token via lm_head[step]

    Key: codec_embeddings are 2048-dim (Talker space), projected to 1024 before the Transformer.

Subclasses the shared Transformer infrastructure from models/tt_transformers/.
The base Transformer provides embedding + LM head + layers. We override to:
    - Skip the base embedding (we provide pre-embedded inputs)
    - Use our own 15 LM heads instead of the base LM head
    - Add input projection and codec embedding handling
"""

import torch

import ttnn
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model import Transformer


class CodePredictorTransformer(Transformer):
    """
    5-layer Transformer that predicts codebooks 1-15 from codebook 0 hidden states.

    Inherits the full prefill/decode infrastructure from the base Transformer.
    Adds:
      - small_to_mtp_projection: Linear(2048→1024, bias=True) input projection
      - codec_embeddings: 15 embedding tables (each vocab=2048, dim=2048) on host
      - lm_heads: 15 output heads (each 1024→2048) on device
    """

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
    ):
        # Inject dummy weights for base Transformer components the CP doesn't use
        prefix = args.get_state_dict_prefix("", None)
        if state_dict is not None:
            if prefix + "tok_embeddings.weight" not in state_dict:
                state_dict[prefix + "tok_embeddings.weight"] = torch.zeros(args.vocab_size, args.dim)
            if prefix + "output.weight" not in state_dict:
                state_dict[prefix + "output.weight"] = torch.zeros(args.vocab_size, args.dim)

        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
        )

        self.talker_hidden_size = args.talker_hidden_size
        self.num_cb_predict = args.num_code_groups - 1  # 15

        # small_to_mtp_projection: Linear(2048→1024, bias=True)
        proj_w, proj_b = self._load_linear(
            state_dict, "talker.code_predictor.small_to_mtp_projection",
            args.talker_hidden_size, args.dim, dtype, mesh_device, weight_cache_path,
        )
        self.proj_w = proj_w
        self.proj_b = proj_b

        # Host-side projection weights for decode-path CPU embedding+projection
        proj_key = "talker.code_predictor.small_to_mtp_projection"
        self.proj_w_torch = state_dict[f"{proj_key}.weight"].float()
        self.proj_b_torch = state_dict[f"{proj_key}.bias"].float()

        # 15 codec embeddings (host-side, in Talker's 2048-dim space)
        self.codec_embeddings = []
        for i in range(self.num_cb_predict):
            key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
            if state_dict is not None and key in state_dict:
                self.codec_embeddings.append(state_dict[key].clone())
            else:
                self.codec_embeddings.append(
                    torch.randn(args.vocab_size, args.talker_hidden_size)
                )

        # 15 LM heads on device (each Linear(1024→2048, no bias))
        self.cp_lm_heads = []
        for i in range(self.num_cb_predict):
            w_key = f"talker.code_predictor.lm_head.{i}.weight"
            if state_dict is not None and w_key in state_dict:
                w = state_dict[w_key].T.contiguous().unsqueeze(0).unsqueeze(0)
            else:
                w = torch.randn(1, 1, args.dim, args.vocab_size)
            tt_w = ttnn.as_tensor(
                w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=weight_cache_path / w_key if weight_cache_path else None,
            )
            self.cp_lm_heads.append(tt_w)

    @staticmethod
    def _load_linear(state_dict, key_prefix, in_dim, out_dim, dtype, mesh_device, weight_cache_path, has_bias=True):
        """Load a linear layer as ttnn tensors."""
        w_key = f"{key_prefix}.weight"
        b_key = f"{key_prefix}.bias"

        if state_dict is not None and w_key in state_dict:
            w = state_dict[w_key].T.contiguous().unsqueeze(0).unsqueeze(0)
        else:
            w = torch.randn(1, 1, in_dim, out_dim)

        tt_w = ttnn.as_tensor(
            w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=weight_cache_path / w_key if weight_cache_path else None,
        )

        tt_b = None
        if has_bias:
            if state_dict is not None and b_key in state_dict:
                b = state_dict[b_key].unsqueeze(0).unsqueeze(0).unsqueeze(0)
            else:
                b = torch.zeros(1, 1, 1, out_dim)
            tt_b = ttnn.as_tensor(
                b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=weight_cache_path / b_key if weight_cache_path else None,
            )
        return tt_w, tt_b

    def input_projection(self, x):
        """Project from Talker's 2048-dim space to CP's 1024-dim space.

        Applies: Linear(2048→1024) + bias
        """
        h = ttnn.matmul(x, self.proj_w)
        if self.proj_b is not None:
            h = ttnn.add(h, self.proj_b)
        return h

    def predict_codebooks(
        self,
        talker_hidden,
        cb0_token,
        talker_codec_embedding_weight,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    ):
        """Generate CB1-CB15 for a single frame on device using KV cache.

        Prefills [talker_hidden, CB0_embed] (2 tokens) through 5 layers, then
        autoregressively decodes CB1-CB15 with KV cache (14 decode steps).

        Args:
            talker_hidden: torch.Tensor [B, 1, 2048] — Talker's post-norm hidden
            cb0_token: torch.Tensor [B] — CB0 token ID
            talker_codec_embedding_weight: torch.Tensor [vocab, 2048] — Talker's codec embedding
            temperature: sampling temperature (0=greedy)
            top_k: top-k sampling
            top_p: nucleus sampling

        Returns:
            all_tokens: torch.Tensor [B, 16] — CB0 + CB1..CB15
        """
        B = talker_hidden.shape[0]

        cb0_emb = torch.nn.functional.embedding(
            cb0_token.unsqueeze(-1), talker_codec_embedding_weight
        )  # [B, 1, 2048]

        context_2048 = torch.cat([talker_hidden, cb0_emb], dim=1)  # [B, 2, 2048]

        # --- Prefill: 2 tokens through 5 layers (padded to 128 for attention) ---
        PREFILL_PAD = 128
        context_padded = torch.nn.functional.pad(
            context_2048, (0, 0, 0, PREFILL_PAD - 2)
        )  # [B, 128, 2048]
        tt_input = ttnn.from_torch(
            context_padded.unsqueeze(1),  # [B, 1, 128, 2048]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tt_x = self.input_projection(tt_input)  # [B, 1, 128, 1024]

        cos_slice = self.rope_setup.cos_matrix_prefill[:, :, :PREFILL_PAD, :]
        sin_slice = self.rope_setup.sin_matrix_prefill[:, :, :PREFILL_PAD, :]
        rot_mats = [cos_slice, sin_slice]

        for layer in self.layers:
            tt_x = layer(
                tt_x,
                current_pos=None,
                rot_mats_global=rot_mats,
                mode=Mode.PREFILL,
                batch_size=B,
            )

        tt_x = self.norm(tt_x, mode=Mode.PREFILL)

        # CB1: logits from last real position (pos 1, within first 32-tile block)
        tt_last = ttnn.slice(tt_x, (0, 0, 1, 0), (1, 1, 2, self.args.dim))
        logits_tt = ttnn.matmul(tt_last, self.cp_lm_heads[0])
        logits_torch = self._logits_to_torch(logits_tt)

        generated = [cb0_token.unsqueeze(-1)]
        next_token = self._sample_token(logits_torch, temperature, top_k, top_p)
        generated.append(next_token.unsqueeze(-1))

        # --- Decode: CB2-CB15, one token at a time with KV cache ---
        for step in range(1, self.num_cb_predict):
            tok_emb = torch.nn.functional.embedding(
                next_token.unsqueeze(-1), self.codec_embeddings[step - 1]
            ).float()  # [B, 1, 2048]

            projected = torch.nn.functional.linear(
                tok_emb, self.proj_w_torch, self.proj_b_torch
            )  # [B, 1, 1024]
            decode_padded = torch.zeros(1, 1, 32, self.args.dim)
            decode_padded[0, 0, 0, :] = projected[0, 0, :]

            tt_decode = ttnn.from_torch(
                decode_padded,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            decode_mem = self.args.get_residual_mem_config(Mode.DECODE, self.prefetcher)
            tt_decode = ttnn.to_memory_config(tt_decode, decode_mem)

            current_pos = torch.tensor([step + 1], dtype=torch.int64)
            padded_pos = torch.nn.functional.pad(
                current_pos, (0, self.args.max_batch_size - 1), value=0
            )
            rot_idxs = self.rope_setup.get_rot_idxs(padded_pos)
            rot_mats_decode = self.rope_setup.get_rot_mats(rot_idxs)

            current_pos_tt = ttnn.from_torch(
                padded_pos.unsqueeze(0),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            x = tt_decode
            for layer in self.layers:
                x = layer(
                    x,
                    current_pos_tt,
                    rot_mats_global=rot_mats_decode,
                    mode=Mode.DECODE,
                )

            x = self.norm(
                x, mode=Mode.DECODE,
                norm_config=self.args.get_norm_config("lm_head", Mode.DECODE, self.prefetcher),
            )

            logits_tt = ttnn.matmul(
                x, self.cp_lm_heads[step],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            logits_tt = ttnn.untilize(logits_tt, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            logits_torch = self._logits_to_torch(logits_tt)
            logits_torch = logits_torch[:1, :]  # first position only

            next_token = self._sample_token(logits_torch, temperature, top_k, top_p)
            generated.append(next_token.unsqueeze(-1))

        return torch.cat(generated, dim=-1)  # [B, 16]

    def _logits_to_torch(self, tt_logits):
        """Convert ttnn logits to torch tensor [B, vocab]."""
        logits = ttnn.to_torch(
            tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)
        )
        return logits[..., : self.args.vocab_size].view(-1, self.args.vocab_size)

    @staticmethod
    def _sample_token(logits, temperature, top_k, top_p):
        """Sample from logits [B, vocab]. Returns [B]."""
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
