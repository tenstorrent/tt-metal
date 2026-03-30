# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# References:
# https://huggingface.co/docs/transformers/model_doc/roberta

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, hidden_size, pad_id):
        super().__init__(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=pad_id,
        )


class PositionEmbedding(nn.Embedding):
    def __init__(self, max_len, hidden_size):
        super().__init__(num_embeddings=max_len, embedding_dim=hidden_size)


class RoBERTaEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, pad_id, hidden_size, drop_prob=0.1):
        super().__init__()

        self.token_embed = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            pad_id=pad_id,
        )
        self.pos_embed = PositionEmbedding(max_len=max_len, hidden_size=hidden_size)

        self.pos = torch.arange(max_len, dtype=torch.long).unsqueeze(0)

        self.norm = nn.LayerNorm(hidden_size)
        self.embed_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        b, seq_len = x.shape

        x = self.token_embed(x)
        x += self.pos_embed(self.pos[:, :seq_len].repeat(b, 1).to(x.device))

        x = self.norm(x)
        x = self.embed_drop(x)
        return x


class ResidualConnection(nn.Module):
    def __init__(self, hidden_size, drop_prob=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_size)
        self.resid_drop = nn.Dropout(drop_prob)

    def forward(self, x, sublayer):
        skip = x.clone()
        x = self.norm(x)
        x = sublayer(x)
        x += skip
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, drop_prob=0.1):
        super().__init__()

        self.n_heads = n_heads

        self.head_size = hidden_size // n_heads

        self.qkv_proj = nn.Linear(hidden_size, 3 * n_heads * self.head_size, bias=False)
        self.attn_drop = nn.Dropout(drop_prob)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _get_attention_score(self, q, k):
        attn_score = torch.einsum("bhnd,bhmd->bhnm", q, k)
        attn_score /= self.head_size**0.5
        return attn_score

    def forward(self, x, mask=None):
        q, k, v = torch.split(
            self.qkv_proj(x),
            split_size_or_sections=self.n_heads * self.head_size,
            dim=2,
        )
        q = rearrange(q, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        k = rearrange(k, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        v = rearrange(v, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        attn_score = self._get_attention_score(q=q, k=k)
        if mask is not None:
            attn_score.masked_fill_(mask=mask, value=-1e9)
        attn_weight = F.softmax(attn_score, dim=3)
        x = torch.einsum("bhnm,bhmd->bhnd", attn_weight, v)
        x = rearrange(x, pattern="b h n d -> b n (h d)")
        x = self.attn_drop(x)
        x = self.out_proj(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, mlp_size, drop_prob=0.1):
        super().__init__()

        self.proj1 = nn.Linear(hidden_size, mlp_size)
        self.proj2 = nn.Linear(mlp_size, hidden_size)
        self.mlp_drop2 = nn.Dropout(drop_prob)
        self.mlp_drop1 = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj1(x)
        # "We use a gelu activation rather than the standard relu, following OpenAI GPT."
        x = F.gelu(x)
        x = self.mlp_drop1(x)
        x = self.proj2(x)
        x = self.mlp_drop2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, n_heads, mlp_size, drop_prob=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            n_heads=n_heads,
            drop_prob=drop_prob,
        )
        self.attn_resid_conn = ResidualConnection(
            hidden_size=hidden_size,
            drop_prob=drop_prob,
        )
        self.feed_forward = PositionwiseFeedForward(
            hidden_size=hidden_size,
            mlp_size=mlp_size,
        )
        self.ff_resid_conn = ResidualConnection(
            hidden_size=hidden_size,
            drop_prob=drop_prob,
        )

    def forward(self, x, mask=None):
        x = self.attn_resid_conn(x=x, sublayer=lambda x: self.self_attn(x, mask=mask))
        x = self.ff_resid_conn(x=x, sublayer=self.feed_forward)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size, mlp_size, drop_prob):
        super().__init__()

        self.enc_stack = nn.ModuleList(
            [
                TransformerLayer(
                    n_heads=n_heads,
                    hidden_size=hidden_size,
                    mlp_size=mlp_size,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask):
        for enc_layer in self.enc_stack:
            x = enc_layer(x, mask=mask)
        return x


class RoBERTa(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        pad_id,
        n_layers=12,
        n_heads=12,
        hidden_size=768,
        mlp_size=768 * 4,
        drop_prob=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.pad_id = pad_id

        self.embed = RoBERTaEmbedding(
            vocab_size=vocab_size,
            max_len=max_len,
            pad_id=pad_id,
            hidden_size=hidden_size,
            drop_prob=drop_prob,
        )
        self.tf_block = TransformerBlock(
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
            drop_prob=drop_prob,
        )

    def _get_pad_mask(self, token_ids):
        mask = (token_ids == self.pad_id).unsqueeze(1).unsqueeze(2)
        mask.requires_grad = False
        return mask

    def forward(self, x):
        pad_mask = self._get_pad_mask(x)
        x = self.embed(x)
        x = self.tf_block(x, mask=pad_mask)
        return x


class MLMHead(nn.Module):
    def __init__(self, vocab_size, hidden_size=768):
        super().__init__()

        self.head_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.head_proj(x)
        return x


class RoBERTaForPretraining(nn.Module):
    def __init__(self, vocab_size, max_len, pad_id, n_layers, n_heads, hidden_size, mlp_size):
        super().__init__()

        self.roberta = RoBERTa(
            vocab_size=vocab_size,
            max_len=max_len,
            pad_id=pad_id,
            n_layers=n_layers,
            n_heads=n_heads,
            hidden_size=hidden_size,
            mlp_size=mlp_size,
        )

        self.mlm_head = MLMHead(vocab_size=vocab_size, hidden_size=hidden_size)

    def forward(self, x):
        x = self.roberta(x)
        x = self.mlm_head(x)
        return x
