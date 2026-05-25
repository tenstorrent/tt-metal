# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone PyTorch reference for the Qwen3-TTS Code Predictor.
No HuggingFace / transformers dependency — loads weights from safetensors directly.
Used for PCC comparison against the TT implementation.

Architecture:
    Code Predictor is a 5-layer Transformer (hidden=1024, GQA 16Q/8KV, head_dim=128, ffn=3072)
    that autoregressively generates CB1-CB15 given the Talker's hidden state and CB0 token.

    Per frame:
        1. Embed CB0 via Talker's codec_embedding → [B, 1, 2048]
        2. Concat with Talker hidden → [B, 2, 2048]
        3. Project via small_to_mtp_projection → [B, 2, 1024]
        4. Run 5-layer Transformer (prefill)
        5. Predict CB1 via lm_head[0]
        6. For CB_i (i=1..14):
           a. Embed CB_i via code_predictor.codec_embedding[i-1] → [2048]
           b. Project → [1024]
           c. Run Transformer (decode step with KV cache)
           d. Predict CB_{i+1} via lm_head[i]
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.qwen3_tts.reference.talker_ref import (
    RMSNorm,
    TalkerAttention,
    TalkerBlock,
    TalkerMLP,
    apply_rotary_emb,
    build_rope_cache,
)


class CodePredictorReference(nn.Module):
    """Standalone PyTorch Code Predictor reference model."""

    def __init__(
        self,
        hidden_size=1024,
        talker_hidden_size=2048,
        n_layers=5,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        ffn_dim=3072,
        vocab_size=2048,
        num_code_groups=16,
        norm_eps=1e-6,
        rope_theta=1000000.0,
        max_seq_len=128,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.talker_hidden_size = talker_hidden_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.num_code_groups = num_code_groups
        self.num_cb_predict = num_code_groups - 1  # 15 codebooks to predict

        # Input projection: Talker hidden space (2048) → CP hidden space (1024)
        self.small_to_mtp_projection = nn.Linear(talker_hidden_size, hidden_size, bias=True)

        # 15 codec embeddings (one per CB0..CB14), in Talker's 2048-dim space
        self.codec_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, talker_hidden_size) for _ in range(self.num_cb_predict)]
        )

        # 5-layer Transformer (same architecture as Talker but smaller)
        self.layers = nn.ModuleList(
            [TalkerBlock(hidden_size, n_heads, n_kv_heads, head_dim, ffn_dim, norm_eps) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(hidden_size, norm_eps)

        # 15 output heads (one per CB1..CB15)
        self.lm_heads = nn.ModuleList(
            [nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(self.num_cb_predict)]
        )

        cos, sin = build_rope_cache(max_seq_len, head_dim, rope_theta)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def _causal_mask(self, seq_len, dtype, device):
        mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def predict_codebooks(
        self,
        talker_hidden,
        cb0_token,
        talker_codec_embedding,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
    ):
        """Generate CB1-CB15 for a single frame.

        Args:
            talker_hidden: [B, 1, talker_hidden_size] hidden state from Talker for this frame
            cb0_token: [B] CB0 token ID
            talker_codec_embedding: nn.Embedding — the Talker's CB0 embedding table (vocab=2048, dim=2048)
            temperature: sampling temperature (0 = greedy)
            top_k: top-k sampling
            top_p: nucleus sampling

        Returns:
            all_tokens: [B, 16] — CB0 + CB1..CB15
        """
        B = talker_hidden.shape[0]
        device = talker_hidden.device

        # Step 1: Embed CB0 via Talker's codec embedding → [B, 1, 2048]
        cb0_emb = talker_codec_embedding(cb0_token).unsqueeze(1)  # [B, 1, 2048]

        # Step 2: Concat talker_hidden + cb0_emb → [B, 2, 2048]
        prefill_input = torch.cat([talker_hidden, cb0_emb], dim=1)

        # Step 3: Project to CP hidden space → [B, 2, 1024]
        x = self.small_to_mtp_projection(prefill_input)

        # Step 4: Prefill through Transformer
        S = x.shape[1]
        cos = self.rope_cos[:S].unsqueeze(0).expand(B, -1, -1).to(device)
        sin = self.rope_sin[:S].unsqueeze(0).expand(B, -1, -1).to(device)
        mask = self._causal_mask(S, x.dtype, device)

        # Init KV caches
        max_seq = 2 + self.num_cb_predict
        kv_caches = []
        for _ in range(self.n_layers):
            k = torch.zeros(B, self.layers[0].attention.n_kv_heads, max_seq, self.layers[0].attention.head_dim, device=device, dtype=x.dtype)
            v = torch.zeros_like(k)
            kv_caches.append((k, v))

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, mask=mask, kv_cache=kv_caches[i], start_pos=0)

        hidden = self.norm(x)
        # Predict CB1 from the last position
        logits_cb1 = self.lm_heads[0](hidden[:, -1:, :])  # [B, 1, vocab]
        cb1_token = self._sample(logits_cb1, temperature, top_k, top_p)

        generated = [cb0_token.unsqueeze(-1), cb1_token.unsqueeze(-1)]

        # Step 5: Decode loop for CB2..CB15
        prev_token = cb1_token
        for step in range(1, self.num_cb_predict):
            # Embed previous token via codec_embedding[step-1]
            tok_emb = self.codec_embeddings[step - 1](prev_token).unsqueeze(1)  # [B, 1, 2048]
            tok_proj = self.small_to_mtp_projection(tok_emb)  # [B, 1, 1024]

            pos = S + step - 1
            cos_step = self.rope_cos[pos : pos + 1].unsqueeze(0).expand(B, -1, -1).to(device)
            sin_step = self.rope_sin[pos : pos + 1].unsqueeze(0).expand(B, -1, -1).to(device)

            x = tok_proj
            for i, layer in enumerate(self.layers):
                x = layer(x, cos_step, sin_step, kv_cache=kv_caches[i], start_pos=pos)

            hidden = self.norm(x)
            logits = self.lm_heads[step](hidden[:, -1:, :])
            next_token = self._sample(logits, temperature, top_k, top_p)
            generated.append(next_token.unsqueeze(-1))
            prev_token = next_token

        return torch.cat(generated, dim=-1)  # [B, 16]

    @staticmethod
    def _sample(logits, temperature, top_k, top_p):
        """Sample from logits [B, 1, vocab]. Returns [B]."""
        logits = logits[:, -1, :]
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @classmethod
    def from_hf_state_dict(cls, state_dict, talker_hidden_size=2048, **kwargs):
        """Load from HF state dict (talker.code_predictor.* keys).

        HF key structure:
            talker.code_predictor.small_to_mtp_projection.weight/bias
            talker.code_predictor.model.codec_embedding.{0-14}.weight
            talker.code_predictor.model.layers.{0-4}.self_attn.*
            talker.code_predictor.model.layers.{0-4}.mlp.*
            talker.code_predictor.model.layers.{0-4}.input_layernorm.weight
            talker.code_predictor.model.layers.{0-4}.post_attention_layernorm.weight
            talker.code_predictor.model.norm.weight
            talker.code_predictor.lm_head.{0-14}.weight
        """
        model = cls(talker_hidden_size=talker_hidden_size, **kwargs)

        key_map = {}
        prefix = "talker.code_predictor."
        for k in state_dict:
            if not k.startswith(prefix):
                continue
            new_k = k[len(prefix) :]

            # small_to_mtp_projection stays as-is
            if new_k.startswith("small_to_mtp_projection"):
                key_map[new_k] = state_dict[k]
                continue

            # Strip "model." prefix
            if new_k.startswith("model."):
                new_k = new_k[len("model.") :]

            # codec_embedding.{i}.weight → codec_embeddings.{i}.weight
            new_k = new_k.replace("codec_embedding.", "codec_embeddings.")

            # Transformer layer key remapping
            new_k = new_k.replace("self_attn.q_proj", "attention.q_proj")
            new_k = new_k.replace("self_attn.k_proj", "attention.k_proj")
            new_k = new_k.replace("self_attn.v_proj", "attention.v_proj")
            new_k = new_k.replace("self_attn.o_proj", "attention.o_proj")
            new_k = new_k.replace("self_attn.q_norm", "attention.q_norm")
            new_k = new_k.replace("self_attn.k_norm", "attention.k_norm")
            new_k = new_k.replace("mlp.gate_proj", "feed_forward.gate_proj")
            new_k = new_k.replace("mlp.up_proj", "feed_forward.up_proj")
            new_k = new_k.replace("mlp.down_proj", "feed_forward.down_proj")
            new_k = new_k.replace("input_layernorm", "attention_norm")
            new_k = new_k.replace("post_attention_layernorm", "ffn_norm")

            # lm_head.{i} → lm_heads.{i}
            new_k = new_k.replace("lm_head.", "lm_heads.")

            key_map[new_k] = state_dict[k]

        missing, unexpected = model.load_state_dict(key_map, strict=False)
        if missing:
            # rope_cos/rope_sin are computed, not loaded
            real_missing = [k for k in missing if "rope_" not in k]
            if real_missing:
                import warnings

                warnings.warn(f"Missing keys: {real_missing}")
        return model

    @classmethod
    def from_safetensors(cls, checkpoint_dir, talker_hidden_size=2048, **kwargs):
        """Load from local safetensors files."""
        from pathlib import Path

        from safetensors import safe_open

        checkpoint_dir = Path(checkpoint_dir)
        st_files = sorted(checkpoint_dir.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No safetensors files in {checkpoint_dir}")

        state_dict = {}
        for f in st_files:
            with safe_open(str(f), framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if key.startswith("talker.code_predictor."):
                        state_dict[key] = sf.get_tensor(key)

        return cls.from_hf_state_dict(state_dict, talker_hidden_size=talker_hidden_size, **kwargs)

    @classmethod
    def from_pretrained(cls, model_id, talker_hidden_size=2048, **kwargs):
        """Load from HuggingFace Hub."""
        from pathlib import Path

        from safetensors import safe_open

        local_dir = Path(model_id)
        if local_dir.is_dir():
            return cls.from_safetensors(model_id, talker_hidden_size=talker_hidden_size, **kwargs)

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_id, "model.safetensors")
        state_dict = {}
        with safe_open(path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if key.startswith("talker.code_predictor."):
                    state_dict[key] = sf.get_tensor(key)

        return cls.from_hf_state_dict(state_dict, talker_hidden_size=talker_hidden_size, **kwargs)
