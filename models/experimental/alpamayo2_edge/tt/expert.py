# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The action expert on device: a 512-wide Cosmos3 Edge text stack over the LM's KV cache.

Expert layer i writes the K/V of its N action tokens (64 in the checkpoint) into rows `[p_pad, p_pad + N)` of LM
layer i's cache, with `p_pad` the CoT prefix rounded up to a tile, and attends non-causally over `[prefix ; own N]`
under an additive mask. The next call writes the same rows, which is the reference's crop back to the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.experimental.alpamayo2_edge.reference.action import expert_mask, expert_position_ids
from models.experimental.alpamayo2_edge.reference.config import TextConfig
from models.experimental.alpamayo2_edge.reference.inputs import ModelInputs
from models.experimental.alpamayo2_edge.tt.generate import CotResult
from models.experimental.alpamayo2_edge.tt.text import STATE_CONVERSION, AlpamayoCache, ReluSquaredFeedForward
from models.tt_dit.blocks.rope import RopeConfig, RotaryEmbedding
from models.tt_dit.encoders.transformer import (
    TransformerContext,
    TransformerRmsNorm,
    _apply_rope,
)
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.utils import tensor

# The reference masks with the fp32 minimum, which rounds to -inf in bf16; any finite value this far below the
# scores gives the same zero weight without risking inf - inf in the softmax.
MASKED = -1e9

_MAX_CHUNK = 128


def _ceil(x: int, multiple: int) -> int:
    return -(-x // multiple) * multiple


@dataclass
class ExpertFrame:
    """Per-frame expert conditioning, shared by every layer and every flow step; N is the number of action tokens."""

    p_pad: int  # cache row of the first action token: the CoT prefix rounded up to a tile
    mask: ttnn.Tensor  # [B, 1, N, p_pad + N] additive bf16
    cos: ttnn.Tensor  # [B, N, head_dim]
    sin: ttnn.Tensor  # [B, N, head_dim]


def make_expert_frame(
    cot: CotResult, inp: ModelInputs, *, ecfg: TextConfig, n_waypoints: int, mesh_device: ttnn.MeshDevice
) -> ExpertFrame:
    """Mask and rotary tables of the `n_waypoints` action tokens for the B samples of `cot`.

    `inp` is the single prompt the CoT was generated from. The action tokens continue each sample's text right after
    its anchor, at `rope_deltas + offset + arange(n_waypoints)` on every M-RoPE axis, so the interleaved M-RoPE
    reduces to 1D RoPE over the full frequency table.
    """
    batch = cot.tokens.shape[0]
    if inp.input_ids.shape[0] != 1 or not bool(inp.attention_mask.all()):
        msg = "the expert frame takes the single unpadded prompt the CoT was generated from"
        raise ValueError(msg)
    if n_waypoints % ttnn.TILE_SIZE != 0:
        msg = f"fill_cache writes whole tiles, so the action token count must be a multiple of 32, got {n_waypoints}"
        raise ValueError(msg)
    p_pad = _ceil(cot.prefix_length, ttnn.TILE_SIZE)
    if p_pad + n_waypoints > cot.cache.size:
        msg = f"a {cot.cache.size}-row cache cannot hold the {n_waypoints} action rows after row {p_pad}"
        raise ValueError(msg)

    rope_deltas = inp.rope_deltas.repeat_interleave(batch, dim=0)
    positions = expert_position_ids(cot.offsets, rope_deltas, n_waypoints)[0]
    rope = RotaryEmbedding(head_size=ecfg.head_dim, config=RopeConfig(theta=ecfg.rope_theta))
    cos, sin = rope.forward(
        tensor.from_torch(positions.float(), device=mesh_device, dtype=ttnn.float32), dtype=ttnn.bfloat16
    )

    # Columns from a sample's offset to p_pad are text after its anchor, or rows no sample wrote: both masked.
    prompt_mask = inp.attention_mask.repeat_interleave(batch, dim=0)
    mask = expert_mask(cot.offsets, prompt_mask, p_pad, n_waypoints).clamp(min=MASKED)
    return ExpertFrame(p_pad=p_pad, mask=tensor.from_torch(mask, device=mesh_device), cos=cos, sin=sin)


def _sdpa_program_config(device: ttnn.MeshDevice, q_len: int, kv_len: int) -> ttnn.SDPAProgramConfig:
    k_chunk = next(c for c in (_MAX_CHUNK, 64, ttnn.TILE_SIZE) if kv_len % c == 0)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=ttnn.TILE_SIZE if q_len % ttnn.TILE_SIZE == 0 else q_len,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )


class TtExpertAttention(Module):
    """GQA attention of the action tokens over the LM prefix cache and themselves, non-causal."""

    def __init__(self, ecfg: TextConfig, *, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        self.num_heads = ecfg.heads
        self.num_kv_heads = ecfg.kv_heads
        self.head_dim = ecfg.head_dim
        self.qkv = Linear(
            ecfg.hidden,
            (ecfg.heads + 2 * ecfg.kv_heads) * ecfg.head_dim,
            bias=False,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
        )
        self.o_proj = Linear(
            ecfg.heads * ecfg.head_dim, ecfg.hidden, bias=False, dtype=ttnn.bfloat16, mesh_device=mesh_device
        )
        self._device = mesh_device
        self._compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if all(f"{p}_proj.weight" in state for p in "qkv"):
            # nlp_create_qkv_heads splits the fused width as [Q heads | K heads | V heads].
            state["qkv.weight"] = torch.cat([state.pop(f"{p}_proj.weight") for p in "qkv"], dim=0)

    def forward(self, x: ttnn.Tensor, *, layer_kv: tuple[ttnn.Tensor, ttnn.Tensor], frame: ExpertFrame) -> ttnn.Tensor:
        """`x` `[B, N, hidden]` -> `[B, N, hidden]`; writes this call's K/V into `layer_kv` at `frame.p_pad`."""
        batch, seq, _ = x.shape
        k_cache, v_cache = layer_kv
        kv_len = frame.p_pad + seq
        if (frame.mask.shape[2], frame.mask.shape[3]) != (seq, kv_len):
            msg = f"the frame mask {tuple(frame.mask.shape)} is not for {seq} action tokens after row {frame.p_pad}"
            raise ValueError(msg)
        if k_cache.shape[0] != batch or k_cache.shape[1] != self.num_kv_heads or k_cache.shape[3] != self.head_dim:
            msg = f"cache {tuple(k_cache.shape)} does not match {batch} sequences of {self.num_kv_heads} KV heads"
            raise ValueError(msg)
        if kv_len > k_cache.shape[2]:
            msg = f"a {k_cache.shape[2]}-row cache cannot hold {seq} rows after row {frame.p_pad}"
            raise ValueError(msg)

        qkv = self.qkv.forward(x)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.unsqueeze(qkv, 1),
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = _apply_rope(q, frame.cos, frame.sin)
        k = _apply_rope(k, frame.cos, frame.sin)
        if k.dtype != k_cache.dtype:
            k = ttnn.typecast(k, k_cache.dtype)
        if v.dtype != v_cache.dtype:
            v = ttnn.typecast(v, v_cache.dtype)

        for b in range(batch):
            ttnn.fill_cache(k_cache, k[b : b + 1], b, update_idx=frame.p_pad)
            ttnn.fill_cache(v_cache, v[b : b + 1], b, update_idx=frame.p_pad)
        ends = [batch, self.num_kv_heads, kv_len, self.head_dim]
        k = ttnn.slice(k_cache, [0, 0, 0, 0], ends, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.slice(v_cache, [0, 0, 0, 0], ends, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=frame.mask,
            scale=self.head_dim**-0.5,
            program_config=_sdpa_program_config(self._device, seq, kv_len),
            compute_kernel_config=self._compute_kernel_config,
        )
        out = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, [batch, seq, self.num_heads * self.head_dim])
        return self.o_proj.forward(out)


class TtExpertLayer(Module):
    def __init__(self, ecfg: TextConfig, *, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        ctx = TransformerContext(device=mesh_device, tp_axis=None, ccl_manager=None, sp_axis=None, fsdp_axis=None)
        self.attn_norm = TransformerRmsNorm(ecfg.hidden, eps=ecfg.eps, ctx=ctx)
        self.attn = TtExpertAttention(ecfg, mesh_device=mesh_device)
        self.ff_norm = TransformerRmsNorm(ecfg.hidden, eps=ecfg.eps, ctx=ctx)
        self.ff = ReluSquaredFeedForward(ecfg.hidden, ecfg.mlp, mesh_device=mesh_device, dtype=ttnn.bfloat16)

    def forward(self, x: ttnn.Tensor, *, layer_kv: tuple[ttnn.Tensor, ttnn.Tensor], frame: ExpertFrame) -> ttnn.Tensor:
        x = x + self.attn.forward(self.attn_norm.forward(x), layer_kv=layer_kv, frame=frame)
        return x + self.ff.forward(self.ff_norm.forward(x))


class TtExpert(Module):
    def __init__(self, ecfg: TextConfig, *, mesh_device: ttnn.MeshDevice, lm_num_layers: int) -> None:
        super().__init__()
        # Expert layer i attends over LM layer i's cache, so the depths must agree.
        if ecfg.layers != lm_num_layers:
            msg = (
                f"expert has {ecfg.layers} layers but the LM has {lm_num_layers}; each expert layer needs its LM cache"
            )
            raise ValueError(msg)
        ctx = TransformerContext(device=mesh_device, tp_axis=None, ccl_manager=None, sp_axis=None, fsdp_axis=None)
        self.layers = ModuleList(TtExpertLayer(ecfg, mesh_device=mesh_device) for _ in range(ecfg.layers))
        self.final_norm = TransformerRmsNorm(ecfg.hidden, eps=ecfg.eps, ctx=ctx)

    @staticmethod
    def convert_state(expert: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Checkpoint `expert.expert.*` tensors, prefix stripped -> our keys."""
        return STATE_CONVERSION.convert(expert)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # A depth override builds fewer layers than the checkpoint holds; their tensors are not loaded.
        for key in list(state):
            if key.startswith("layers.") and int(key.split(".")[1]) >= len(self.layers):
                del state[key]

    def forward(self, x: ttnn.Tensor, *, cache: AlpamayoCache, frame: ExpertFrame) -> ttnn.Tensor:
        """`x` `[B, N, hidden]` action embeddings -> `[B, N, hidden]` after the final norm."""
        for i, layer in enumerate(self.layers):
            x = layer.forward(x, layer_kv=cache.layer_kv(i), frame=frame)
        return self.final_norm.forward(x)
