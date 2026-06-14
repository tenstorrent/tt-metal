"""Phase 1 standard blocks for ZAYA1-8B on tt-metal (correctness-first, direct ttnn).

These mirror the HF reference math exactly so each can be PCC-validated against
the dumped golden tensors. Perf-oriented sharded variants come in Phase 6.
"""
import torch
import ttnn

from .model_args import ZayaConfig


# ----------------------------------------------------------------------------
# host-side helpers
# ----------------------------------------------------------------------------
def compute_cos_sin(seq_len, rotary_dim=ZayaConfig.rotary_dim, theta=ZayaConfig.rope_theta,
                    dtype=torch.float32):
    """Reference ZayaRotaryEmbedding: inv_freq over rotary_dim/2, emb=cat(freqs,freqs).
    Returns cos, sin of shape [seq_len, rotary_dim]."""
    half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))  # [half]
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)            # [seq, half]
    emb = torch.cat((freqs, freqs), dim=-1)       # [seq, rotary_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def to_dev(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memcfg=None):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=device,
        memory_config=memcfg or ttnn.DRAM_MEMORY_CONFIG,
    )


# ----------------------------------------------------------------------------
# modules
# ----------------------------------------------------------------------------
class Embedding:
    def __init__(self, device, weight_torch, dtype=ttnn.bfloat16):
        self.device = device
        self.weight = ttnn.from_torch(
            weight_torch, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, ids_torch):
        # ids_torch: [B, S] int. ttnn.embedding wants row-major uint32.
        ids = ttnn.from_torch(
            ids_torch.to(torch.int32), dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.embedding(ids, self.weight, layout=ttnn.TILE_LAYOUT)  # [B, S, dim]
        return out


class RMSNorm:
    def __init__(self, device, weight_torch, eps=ZayaConfig.norm_eps):
        # weight stored as a row vector [1, dim] in TILE layout
        self.weight = to_dev(weight_torch.reshape(1, -1), device, dtype=ttnn.bfloat16)
        self.eps = eps

    def __call__(self, x):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)


class LMHead:
    """Tied to embedding: logits = hidden @ embed^T. weight_torch is embed [vocab, dim]."""

    def __init__(self, device, weight_torch, dtype=ttnn.bfloat16):
        wT = weight_torch.t().contiguous()  # [dim, vocab]
        self.weight = to_dev(wT, device, dtype=dtype)
        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True,
        )

    def __call__(self, x):
        return ttnn.matmul(
            x, self.weight, compute_kernel_config=self.compute_kernel,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
