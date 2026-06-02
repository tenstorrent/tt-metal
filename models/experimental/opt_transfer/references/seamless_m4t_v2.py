import torch
import torch.nn as nn


class SeamlessBlock(nn.Module):
    """Faithful-enough reference for the SeamlessM4T-v2 self-attn + FFN sub-block:
    BART-style 4-projection MHA (bias=True) + post-attn FFN. Used as the trace target."""

    def __init__(self, embed=1024, num_heads=16, ffn=4096):
        super().__init__()
        self.h, self.d = num_heads, embed // num_heads
        self.q_proj = nn.Linear(embed, embed)
        self.k_proj = nn.Linear(embed, embed)
        self.v_proj = nn.Linear(embed, embed)
        self.out_proj = nn.Linear(embed, embed)
        self.attn_norm = nn.LayerNorm(embed)
        self.fc1 = nn.Linear(embed, ffn)
        self.fc2 = nn.Linear(ffn, embed)
        self.ffn_norm = nn.LayerNorm(embed)
        self.scale = self.d**-0.5

    def _split(self, t):
        b, s, _ = t.shape
        return t.view(b, s, self.h, self.d).transpose(1, 2)

    def forward(self, x):
        h = self.attn_norm(x)
        q, k, v = self._split(self.q_proj(h)), self._split(self.k_proj(h)), self._split(self.v_proj(h))
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1) @ v
        b, _, s, _ = attn.shape
        attn = attn.transpose(1, 2).reshape(b, s, -1)
        x = x + self.out_proj(attn)
        h2 = self.ffn_norm(x)
        return x + self.fc2(torch.relu(self.fc1(h2)))
