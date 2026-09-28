# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Temporary host fallbacks for the DeepSeek-V4.1 prototype (§4, graph nodes B5 QDQ, B6-B13).

The device prototype runs the q/kv stems, sparse attention, output projection, mHC and MoE on device.
The compressed-KV path -- compressor, index keys, compressed KV write, indexer, candidate selection,
top-k -- and the window-KV FP8 quantize-dequantize run here, on the vendored reference's own modules,
fed with the device's activations. Each call is a disposition item of the prototype; production
replaces them with device operations (beads F2-F4, F6, F7).
"""

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.kernel_cpu import act_quant

MASKED_INDEX = -1


class HostCompressedAttention:
    """Owns a reference ``Transformer`` whose layer ``i`` corresponds to device layer ``i``.

    Calls must follow execution order, as the reference's ``shared_attn`` runtime expects: every KV/index
    source runs before the layers that read its state. Single-shot prefill only (start_pos 0).
    """

    def __init__(self, reference: v41.Transformer):
        self.reference = reference

    @torch.no_grad()
    def window_kv(self, kv: torch.Tensor) -> torch.Tensor:
        """[S, head_dim] bf16 post-RoPE window KV -> the reference's FP8 (block 32, ue8m0) QDQ values."""
        out = kv.to(torch.bfloat16).clone().unsqueeze(0)
        act_quant(out, v41.fp8_block_size, v41.scale_fmt, v41.scale_dtype, True)
        return out[0]

    @torch.no_grad()
    def compressed(self, layer: int, x: torch.Tensor, qr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compressed KV rows visible to the chunk and each query's selected rows.

        ``x`` [S, dim] is the attention input (after attn_norm), ``qr`` [S, q_lora] the normalized q
        latent. Returns (kv [T, head_dim] bf16, idxs [S, k] int64 into kv, -1 = none).
        """
        attn = self.reference.layers[layer].attn
        assert attn.compress_ratio, f"layer {layer} has no compressed KV"
        seq = x.shape[0]
        with v41.set_dtype(torch.bfloat16):
            kv, idxs = attn._compress_kv(x.to(torch.bfloat16)[None], qr.to(torch.bfloat16)[None], 0, seq)
        idxs = idxs[0].long()
        return kv[0], torch.where(idxs >= 0, idxs - seq, MASKED_INDEX)


def attention_index_rows(seq: int, window: int, compressed_idxs: torch.Tensor | None) -> torch.Tensor:
    """[S, window + k] rows into ``[window KV (S rows) | compressed KV]``, valid entries first.

    The reference concatenates the causal window slots (``get_window_topk_idxs``) with the compressed
    selection offset by S; sparse_sdpa wants every masked slot in one tail, so rows are compacted.
    """
    rows = v41.get_window_topk_idxs(window, 1, seq, 0)[0].long()
    if compressed_idxs is not None:
        comp = torch.where(compressed_idxs >= 0, compressed_idxs + seq, MASKED_INDEX)
        rows = torch.cat([rows, comp], dim=-1)
    order = torch.sort((rows < 0).to(torch.int8), dim=-1, stable=True).indices
    return torch.gather(rows, -1, order)
