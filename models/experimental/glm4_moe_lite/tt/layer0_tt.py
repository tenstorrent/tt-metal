# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

import ttnn

from models.common.rmsnorm import RMSNorm
from models.experimental.glm4_moe_lite.tt.tt_embedding import convert_embedding_weight_to_tt, run_tt_embedding
from models.experimental.glm4_moe_lite.tt.weights import LazyStateDict, load_glm_lazy_state_dict


def _round_up(x: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("multiple must be > 0")
    return ((x + multiple - 1) // multiple) * multiple


def _mla_scale() -> float:
    """Return the MLA scaling factor.

    DeepseekV3Attention scales by 1/sqrt(qk_head_dim)=1/sqrt(256). Keep a kvpe
    mode for experiments.
    """
    scale_mode = os.environ.get("GLM4_MOE_LITE_MLA_SCALE_MODE", "qk").strip().lower()
    if scale_mode == "kvpe":
        return float((512 + 64) ** -0.5)
    return float(256**-0.5)


def _rot_transformation_mat_torch() -> torch.Tensor:
    # Match ttnn.experimental.rotary_embedding_llama expectations.
    dhead = 32
    rot = torch.zeros(1, 1, dhead, dhead, dtype=torch.float32)
    rot[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1.0
    rot[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1.0
    return rot


def _rope_cos_sin_torch(*, seq_len: int, dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return cos/sin matrices in TT "meta-style" interleaved form:
    - output shape: [1, 1, seq_len, dim]
    - last dim layout: [t0, t0, t1, t1, ...]
    """
    if dim % 2 != 0:
        raise ValueError(f"rope dim must be even, got dim={dim}")
    half = dim // 2

    # Standard RoPE inv_freq.
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * (2.0 / dim)))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [S, half]

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # Convert [t0..t(half-1)] -> [t0, t0, t1, t1, ...] to match meta-style rotate.
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
    return cos, sin


def make_rope_tensors(
    *,
    device,
    seq_len: int,
    rope_dim: int,
    rope_theta: float,
) -> dict[str, object]:
    cos_t, sin_t = _rope_cos_sin_torch(seq_len=seq_len, dim=rope_dim, base=rope_theta)
    trans_t = _rot_transformation_mat_torch().to(dtype=torch.bfloat16)

    is_mesh_device = device.__class__.__name__ == "MeshDevice"

    # Keep a host-side copy for trace-mode decode. `ttnn.gather` is currently not
    # trace-capture safe on mesh devices, so the decode trace updates RoPE
    # cos/sin by copying small per-step slices from host.
    cos_host = cos_t.to(dtype=torch.bfloat16).cpu()
    sin_host = sin_t.to(dtype=torch.bfloat16).cpu()

    cos = ttnn.from_torch(
        cos_t.to(dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    sin = ttnn.from_torch(
        sin_t.to(dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    trans = ttnn.from_torch(
        trans_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    return {
        "cos_matrix": cos,
        "sin_matrix": sin,
        "trans_matrix": trans,
        "cos_matrix_host": cos_host,
        "sin_matrix_host": sin_host,
    }


def _linear_weight_tt(
    *,
    device,
    torch_weight_out_in: torch.Tensor,
    cache_file: Optional[Path] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """
    Convert a torch Linear weight in HF layout [out, in] into TT layout [1, 1, in, out].
    """
    if torch_weight_out_in.ndim != 2:
        raise ValueError(f"expected 2D weight, got shape={tuple(torch_weight_out_in.shape)}")
    w = torch_weight_out_in.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)
    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    return ttnn.as_tensor(
        w,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        cache_file_name=None if cache_file is None else Path(cache_file),
    )


def _per_head_weight_tt(
    *,
    device,
    torch_weight: torch.Tensor,
    cache_file: Optional[Path] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """
    Convert a per-head torch weight [H, in, out] into TT weight [1, H, in, out].
    """
    if torch_weight.ndim != 3:
        raise ValueError(f"expected [H,in,out] weight, got shape={tuple(torch_weight.shape)}")
    w = torch_weight.unsqueeze(0).contiguous()
    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    return ttnn.as_tensor(
        w,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        cache_file_name=None if cache_file is None else Path(cache_file),
    )


@dataclass(frozen=True)
class Layer0TTWeights:
    embed_w: ttnn.Tensor

    # Layer norms
    input_layernorm: Any
    q_a_layernorm: Any
    kv_a_layernorm: Any
    post_attention_layernorm: Any

    # Attention projections
    w_q_a: ttnn.Tensor
    w_q_b: ttnn.Tensor
    w_kv_a: ttnn.Tensor
    w_kv_b1: ttnn.Tensor
    w_kv_b2: ttnn.Tensor
    w_o: ttnn.Tensor

    # MLP projections
    w_mlp_gate: ttnn.Tensor
    w_mlp_up: ttnn.Tensor
    w_mlp_down: ttnn.Tensor


def convert_layer0_weights(
    *,
    device,
    state: LazyStateDict,
    cache_dir: Optional[Path] = None,
) -> Layer0TTWeights:
    """
    Convert the (minimal) subset of GLM-4.7-Flash weights required for layer-0 prefill.
    """
    cache_dir = None if cache_dir is None else Path(cache_dir)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def c(name: str) -> Optional[Path]:
        return None if cache_dir is None else cache_dir / name

    embed_w = convert_embedding_weight_to_tt(
        device=device, embed_weight=state["model.embed_tokens.weight"], cache_file_name=c("embed_w")
    )

    # Norms (RMSNorm weight layout is handled by RMSNorm class).
    input_layernorm = RMSNorm(
        device=device,
        dim=2048,
        eps=1e-5,
        state_dict=state,
        state_dict_prefix="model.layers.0.",
        weight_key="input_layernorm",
        weight_cache_path=cache_dir,
        weight_dtype=ttnn.bfloat16,
        is_distributed=False,
    )
    q_a_layernorm = RMSNorm(
        device=device,
        dim=768,
        eps=1e-5,
        state_dict=state,
        state_dict_prefix="model.layers.0.self_attn.",
        weight_key="q_a_layernorm",
        weight_cache_path=cache_dir,
        weight_dtype=ttnn.bfloat16,
        is_distributed=False,
    )
    kv_a_layernorm = RMSNorm(
        device=device,
        dim=512,
        eps=1e-5,
        state_dict=state,
        state_dict_prefix="model.layers.0.self_attn.",
        weight_key="kv_a_layernorm",
        weight_cache_path=cache_dir,
        weight_dtype=ttnn.bfloat16,
        is_distributed=False,
    )
    post_attention_layernorm = RMSNorm(
        device=device,
        dim=2048,
        eps=1e-5,
        state_dict=state,
        state_dict_prefix="model.layers.0.",
        weight_key="post_attention_layernorm",
        weight_cache_path=cache_dir,
        weight_dtype=ttnn.bfloat16,
        is_distributed=False,
    )

    w_q_a = _linear_weight_tt(
        device=device, torch_weight_out_in=state["model.layers.0.self_attn.q_a_proj.weight"], cache_file=c("w_q_a")
    )
    w_q_b = _linear_weight_tt(
        device=device, torch_weight_out_in=state["model.layers.0.self_attn.q_b_proj.weight"], cache_file=c("w_q_b")
    )
    w_kv_a = _linear_weight_tt(
        device=device,
        torch_weight_out_in=state["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"],
        cache_file=c("w_kv_a"),
    )

    # KV b1/b2 weights are derived per-head from kv_b_proj.weight.
    kv_b = state["model.layers.0.self_attn.kv_b_proj.weight"]  # [num_heads*(qk_nope+v), kv_lora]
    num_heads = 20
    qk_nope = 192
    kv_lora = 512
    v_dim = 256

    kv_b = kv_b.view(num_heads, qk_nope + v_dim, kv_lora)
    w_kv_b1_torch = kv_b[:, :qk_nope, :].contiguous()  # [H, qk_nope, kv_lora]
    w_kv_b2_torch = kv_b[:, -v_dim:, :].transpose(1, 2).contiguous()  # [H, kv_lora, v]

    w_kv_b1 = _per_head_weight_tt(device=device, torch_weight=w_kv_b1_torch, cache_file=c("w_kv_b1"))
    w_kv_b2 = _per_head_weight_tt(device=device, torch_weight=w_kv_b2_torch, cache_file=c("w_kv_b2"))

    w_o = _linear_weight_tt(
        device=device, torch_weight_out_in=state["model.layers.0.self_attn.o_proj.weight"], cache_file=c("w_o")
    )

    w_mlp_gate = _linear_weight_tt(
        device=device, torch_weight_out_in=state["model.layers.0.mlp.gate_proj.weight"], cache_file=c("w_mlp_gate")
    )
    w_mlp_up = _linear_weight_tt(
        device=device, torch_weight_out_in=state["model.layers.0.mlp.up_proj.weight"], cache_file=c("w_mlp_up")
    )
    w_mlp_down = _linear_weight_tt(
        device=device, torch_weight_out_in=state["model.layers.0.mlp.down_proj.weight"], cache_file=c("w_mlp_down")
    )

    return Layer0TTWeights(
        embed_w=embed_w,
        input_layernorm=input_layernorm,
        q_a_layernorm=q_a_layernorm,
        kv_a_layernorm=kv_a_layernorm,
        post_attention_layernorm=post_attention_layernorm,
        w_q_a=w_q_a,
        w_q_b=w_q_b,
        w_kv_a=w_kv_a,
        w_kv_b1=w_kv_b1,
        w_kv_b2=w_kv_b2,
        w_o=w_o,
        w_mlp_gate=w_mlp_gate,
        w_mlp_up=w_mlp_up,
        w_mlp_down=w_mlp_down,
    )


@dataclass(frozen=True)
class Layer0TTOutputs:
    x_embed: torch.Tensor
    x_attn_out: torch.Tensor
    x_mlp_out: torch.Tensor


def run_layer0_prefill_tt(
    *,
    device,
    snapshot_dir: Path,
    input_ids: torch.Tensor,
    cache_dir: Optional[Path] = None,
    seq_pad_multiple: int = 128,
) -> Layer0TTOutputs:
    """
    Run GLM layer-0 prefill on TT and return intermediates on CPU.

    This is intended as a bring-up correctness harness (not yet integrated into vLLM).
    """
    if input_ids.ndim != 2:
        raise ValueError(f"expected input_ids [B,S], got shape={tuple(input_ids.shape)}")
    batch, seq_len = input_ids.shape
    if batch != 1:
        raise NotImplementedError("layer0 TT harness currently supports batch=1 only")

    padded_len = _round_up(int(seq_len), int(seq_pad_multiple))
    if padded_len != int(seq_len):
        pad = torch.zeros((batch, padded_len - int(seq_len)), dtype=input_ids.dtype)
        input_ids_padded = torch.cat([input_ids.cpu(), pad], dim=1)
    else:
        input_ids_padded = input_ids.cpu()

    # Load weights lazily and convert only the layer-0 subset to TT.
    state = load_glm_lazy_state_dict(Path(snapshot_dir), num_layers=47)
    w = convert_layer0_weights(device=device, state=state, cache_dir=cache_dir)

    # Embedding
    x_embed = run_tt_embedding(device=device, token_ids=input_ids_padded.to(torch.int32), tt_weight=w.embed_w)
    if x_embed.layout != ttnn.TILE_LAYOUT:
        x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
    x_embed = ttnn.reshape(x_embed, (1, 1, padded_len, 2048))

    # ---- Attention (MLA prefill) ----
    residual = x_embed
    x = w.input_layernorm(x_embed, mode="prefill")

    # Q
    q_a = ttnn.linear(x, w.w_q_a)
    q_a = w.q_a_layernorm(q_a, mode="prefill")
    q = ttnn.linear(q_a, w.w_q_b)

    q = ttnn.reshape(q, (1, padded_len, 20, 256))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1, H, S, 256]
    q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, 20, padded_len, 192])
    q_rope = ttnn.slice(q, [0, 0, 0, 192], [1, 20, padded_len, 256])
    ttnn.deallocate(q)

    # Project q_nope into kv_lora_rank space (per-head).
    q_nope = ttnn.linear(q_nope, w.w_kv_b1)

    # KV
    kv = ttnn.linear(x, w.w_kv_a)  # [1, 1, S, 576]
    kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, padded_len, 512])
    kv_rope = ttnn.slice(kv, [0, 0, 0, 512], [1, 1, padded_len, 576])
    ttnn.deallocate(kv)
    kv_nope = w.kv_a_layernorm(kv_nope, mode="prefill")

    rope = make_rope_tensors(device=device, seq_len=padded_len, rope_dim=64, rope_theta=1.0e6)

    # RoPE ops require bfloat16.
    if q_rope.dtype != ttnn.bfloat16:
        q_rope = ttnn.typecast(q_rope, dtype=ttnn.bfloat16)
    if kv_rope.dtype != ttnn.bfloat16:
        kv_rope = ttnn.typecast(kv_rope, dtype=ttnn.bfloat16)

    q_rope = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        rope["cos_matrix"],
        rope["sin_matrix"],
        rope["trans_matrix"],
        is_decode_mode=False,
    )
    kv_rope = ttnn.experimental.rotary_embedding_llama(
        kv_rope,
        rope["cos_matrix"],
        rope["sin_matrix"],
        rope["trans_matrix"],
        is_decode_mode=False,
    )

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [1, 20, S, 576]
    kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1, 1, S, 576]
    ttnn.deallocate(q_nope)
    ttnn.deallocate(q_rope)
    ttnn.deallocate(kv_nope)
    ttnn.deallocate(kv_rope)

    scale = _mla_scale()
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,  # heads padded up to 32
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    attn_latent = ttnn.transformer.flash_mla_prefill(
        q_kvpe,
        kvpe,
        head_dim_v=512,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        attn_mask=None,
        is_causal=True,
    )
    ttnn.deallocate(q_kvpe)
    ttnn.deallocate(kvpe)

    # flash_mla_prefill pads head dim up to q_chunk_size (32). Slice back to 20 heads.
    attn_latent = ttnn.slice(attn_latent, [0, 0, 0, 0], [1, 20, padded_len, 512])

    v = ttnn.linear(attn_latent, w.w_kv_b2)  # [1, 20, S, 256]
    ttnn.deallocate(attn_latent)

    v = ttnn.permute(v, (0, 2, 1, 3))  # [1, S, 20, 256]
    v = ttnn.reshape(v, (1, 1, padded_len, 20 * 256))  # [1, 1, S, 5120]
    attn_out = ttnn.linear(v, w.w_o)  # [1, 1, S, 2048]
    ttnn.deallocate(v)

    x_attn_out = residual + attn_out
    ttnn.deallocate(attn_out)

    # ---- MLP (dense for layer 0) ----
    residual = x_attn_out
    x = w.post_attention_layernorm(x_attn_out, mode="prefill")

    gate = ttnn.linear(x, w.w_mlp_gate)
    up = ttnn.linear(x, w.w_mlp_up)
    ttnn.deallocate(x)

    x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(gate)
    ttnn.deallocate(up)

    mlp_out = ttnn.linear(x_ff, w.w_mlp_down)
    ttnn.deallocate(x_ff)

    x_mlp_out = residual + mlp_out
    ttnn.deallocate(mlp_out)

    # Slice off padding and return to torch.
    x_embed = ttnn.slice(x_embed, [0, 0, 0, 0], [1, 1, seq_len, 2048])
    x_attn_out = ttnn.slice(x_attn_out, [0, 0, 0, 0], [1, 1, seq_len, 2048])
    x_mlp_out = ttnn.slice(x_mlp_out, [0, 0, 0, 0], [1, 1, seq_len, 2048])

    out = Layer0TTOutputs(
        x_embed=ttnn.to_torch(x_embed).reshape(batch, seq_len, 2048),
        x_attn_out=ttnn.to_torch(x_attn_out).reshape(batch, seq_len, 2048),
        x_mlp_out=ttnn.to_torch(x_mlp_out).reshape(batch, seq_len, 2048),
    )

    # Deallocate final TT tensors to keep tests from accumulating buffers.
    ttnn.deallocate(x_embed)
    ttnn.deallocate(x_attn_out)
    ttnn.deallocate(x_mlp_out)

    return out


@torch.no_grad()
def run_layer0_decode_one_step_unpaged_tt(
    *,
    device,
    snapshot_dir: Path,
    prefix_input_ids: torch.Tensor,
    next_token_id: int,
    block_size: int = 128,
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    Layer-0 decode (one step) using the *unpaged* FlashMLA decode op.

    This is an intermediate bring-up harness to validate decode math (Q + decode kernel + output projections)
    before wiring paged cache update semantics.

    Returns:
      torch.Tensor [1, hidden] for the decoded token output (post-MLP residual) for batch=1.
    """
    prefix_input_ids = prefix_input_ids.cpu()
    if prefix_input_ids.ndim != 2 or prefix_input_ids.shape[0] != 1:
        raise ValueError(f"expected prefix_input_ids shape [1,S], got {tuple(prefix_input_ids.shape)}")

    seq_len = int(prefix_input_ids.shape[1])
    if seq_len <= 0:
        raise ValueError("prefix_input_ids must be non-empty")

    # Decode kernels want a reasonably large k_chunk_size; align K to `block_size` so K_seq_len % k_chunk_size == 0.
    full_len = seq_len + 1
    padded_len = _round_up(full_len, int(block_size))

    full_padded = torch.zeros((1, padded_len), dtype=prefix_input_ids.dtype)
    full_padded[:, :seq_len] = prefix_input_ids
    full_padded[:, seq_len] = int(next_token_id)

    # Load + convert layer0 weights.
    state = load_glm_lazy_state_dict(Path(snapshot_dir), num_layers=47)
    w = convert_layer0_weights(device=device, state=state, cache_dir=cache_dir)

    # Embedding for the whole (padded) sequence.
    x_embed = run_tt_embedding(device=device, token_ids=full_padded.to(torch.int32), tt_weight=w.embed_w)
    if x_embed.layout != ttnn.TILE_LAYOUT:
        x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
    x_embed = ttnn.reshape(x_embed, (1, 1, padded_len, 2048))

    # Keep just the decode token's embedding for residual.
    x_embed_tok = ttnn.slice(x_embed, [0, 0, seq_len, 0], [1, 1, seq_len + 1, 2048])  # [1,1,1,2048]

    # ---- Build Q/KVPE for the full sequence (same math as prefill path) ----
    x = w.input_layernorm(x_embed, mode="prefill")

    # Q
    q_a = ttnn.linear(x, w.w_q_a)
    q_a = w.q_a_layernorm(q_a, mode="prefill")
    q = ttnn.linear(q_a, w.w_q_b)
    ttnn.deallocate(q_a)

    q = ttnn.reshape(q, (1, padded_len, 20, 256))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1, H, S, 256]
    q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, 20, padded_len, 192])
    q_rope = ttnn.slice(q, [0, 0, 0, 192], [1, 20, padded_len, 256])
    ttnn.deallocate(q)

    q_nope = ttnn.linear(q_nope, w.w_kv_b1)  # [1,20,S,512]

    # KV
    kv = ttnn.linear(x, w.w_kv_a)  # [1, 1, S, 576]
    ttnn.deallocate(x)
    kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, padded_len, 512])
    kv_rope = ttnn.slice(kv, [0, 0, 0, 512], [1, 1, padded_len, 576])
    ttnn.deallocate(kv)
    kv_nope = w.kv_a_layernorm(kv_nope, mode="prefill")

    rope = make_rope_tensors(device=device, seq_len=padded_len, rope_dim=64, rope_theta=1.0e6)

    if q_rope.dtype != ttnn.bfloat16:
        q_rope = ttnn.typecast(q_rope, dtype=ttnn.bfloat16)
    if kv_rope.dtype != ttnn.bfloat16:
        kv_rope = ttnn.typecast(kv_rope, dtype=ttnn.bfloat16)

    q_rope = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        rope["cos_matrix"],
        rope["sin_matrix"],
        rope["trans_matrix"],
        is_decode_mode=False,
    )
    kv_rope = ttnn.experimental.rotary_embedding_llama(
        kv_rope,
        rope["cos_matrix"],
        rope["sin_matrix"],
        rope["trans_matrix"],
        is_decode_mode=False,
    )

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [1,20,S,576]
    kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1,1,S,576]
    ttnn.deallocate(q_nope)
    ttnn.deallocate(q_rope)
    ttnn.deallocate(kv_nope)
    ttnn.deallocate(kv_rope)

    # Slice Q for the decode token and transpose to [1, B=1, H, D].
    q_last = ttnn.slice(q_kvpe, [0, 0, seq_len, 0], [1, 20, seq_len + 1, 576])  # [1,20,1,576]
    ttnn.deallocate(q_kvpe)
    q_for_decode = ttnn.permute(q_last, (0, 2, 1, 3))  # [1,1,20,576]
    ttnn.deallocate(q_last)

    pos = ttnn.from_torch(
        torch.tensor([seq_len], dtype=torch.int32), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    scale = _mla_scale()
    total_k_len = int(blocks_per_seq * block_size)
    # Pick a chunk size <= total K length to avoid edge cases on very short prompts.
    k_chunk_size = 128
    if total_k_len < k_chunk_size:
        k_chunk_size = max(32, 1 << int(math.floor(math.log2(total_k_len))))
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,  # not used in decode
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    attn_latent = ttnn.transformer.flash_multi_latent_attention_decode(
        q_for_decode,
        kvpe,
        head_dim_v=512,
        cur_pos_tensor=pos,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # [1,1,H_padded,512]
    ttnn.deallocate(q_for_decode)
    ttnn.deallocate(pos)

    attn_latent = ttnn.slice(attn_latent, [0, 0, 0, 0], [1, 1, 20, 512])  # [1,1,20,512]
    attn_latent = ttnn.permute(attn_latent, (0, 2, 1, 3))  # [1,20,1,512]

    v = ttnn.linear(attn_latent, w.w_kv_b2)  # [1,20,1,256]
    ttnn.deallocate(attn_latent)

    v = ttnn.permute(v, (0, 2, 1, 3))  # [1,1,20,256]
    v = ttnn.reshape(v, (1, 1, 1, 20 * 256))  # [1,1,1,5120]

    attn_out = ttnn.linear(v, w.w_o)  # [1,1,1,2048]
    ttnn.deallocate(v)

    x_attn_out = x_embed_tok + attn_out
    ttnn.deallocate(attn_out)
    ttnn.deallocate(x_embed_tok)

    # MLP (dense) for layer 0.
    residual = x_attn_out
    x = w.post_attention_layernorm(x_attn_out, mode="prefill")
    gate = ttnn.linear(x, w.w_mlp_gate)
    up = ttnn.linear(x, w.w_mlp_up)
    ttnn.deallocate(x)
    x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(gate)
    ttnn.deallocate(up)
    mlp_out = ttnn.linear(x_ff, w.w_mlp_down)
    ttnn.deallocate(x_ff)

    x_mlp_out = residual + mlp_out
    ttnn.deallocate(mlp_out)
    ttnn.deallocate(residual)

    out = ttnn.to_torch(x_mlp_out).reshape(1, 2048).cpu()
    ttnn.deallocate(x_mlp_out)

    # Deallocate cached rope tensors.
    ttnn.deallocate(rope["cos_matrix"])
    ttnn.deallocate(rope["sin_matrix"])
    ttnn.deallocate(rope["trans_matrix"])

    # Cleanup.
    ttnn.deallocate(x_embed)
    ttnn.deallocate(kvpe)

    return out


def _alloc_paged_kvpe_cache(
    *,
    device,
    max_num_blocks: int,
    block_size: int,
    kvpe_dim: int = 576,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    return ttnn.zeros(
        shape=(int(max_num_blocks), 1, int(block_size), int(kvpe_dim)),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _alloc_contiguous_page_table(*, batch: int, blocks_per_seq: int) -> torch.Tensor:
    """
    Produce a simple page table where user i maps to blocks [i*BPS, ..., i*BPS+BPS-1].
    """
    batch = int(batch)
    blocks_per_seq = int(blocks_per_seq)
    max_num_blocks = batch * blocks_per_seq
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(batch, blocks_per_seq)
    return page_table


@torch.no_grad()
def debug_fill_kvpe_cache_for_one_step_tt(
    *,
    device,
    snapshot_dir: Path,
    prefix_input_ids: torch.Tensor,
    next_token_id: int,
    batch: int = 1,
    block_size: int = 64,
    cache_dir: Optional[Path] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Debug helper: fill a paged KVPE cache for (prefix + next token) and return:
      1) The KVPE tensor used as the source (unpadded view): torch [batch=1, nkv=1, page_len, kvpe_dim]
      2) The KVPE reconstructed from the cache using the page table: same shape
    """
    prefix_input_ids = prefix_input_ids.cpu()
    if prefix_input_ids.ndim != 2 or prefix_input_ids.shape[0] != 1:
        raise ValueError(f"expected prefix_input_ids shape [1,S], got {tuple(prefix_input_ids.shape)}")

    seq_len = int(prefix_input_ids.shape[1])
    blocks_per_seq = _round_up(seq_len + 1, block_size) // block_size
    # Match vLLM-style behavior: allocate at least 128 tokens worth of blocks to avoid
    # short-sequence edge cases in paged-decode kernels.
    min_blocks_per_seq = max(1, (128 + int(block_size) - 1) // int(block_size))
    blocks_per_seq = max(blocks_per_seq, min_blocks_per_seq)
    page_len = int(blocks_per_seq * block_size)

    kvpe_cache = _alloc_paged_kvpe_cache(
        device=device,
        max_num_blocks=int(batch * blocks_per_seq),
        block_size=block_size,
        kvpe_dim=576,
        dtype=ttnn.bfloat8_b,
    )
    page_table = _alloc_contiguous_page_table(batch=batch, blocks_per_seq=blocks_per_seq)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load + convert layer0 weights.
    state = load_glm_lazy_state_dict(Path(snapshot_dir), num_layers=47)
    w = convert_layer0_weights(device=device, state=state, cache_dir=cache_dir)

    full_padded = torch.zeros((1, page_len), dtype=prefix_input_ids.dtype)
    full_padded[:, :seq_len] = prefix_input_ids
    full_padded[:, seq_len] = int(next_token_id)

    rope = make_rope_tensors(device=device, seq_len=page_len, rope_dim=64, rope_theta=1.0e6)

    x_embed = run_tt_embedding(device=device, token_ids=full_padded.to(torch.int32), tt_weight=w.embed_w)
    if x_embed.layout != ttnn.TILE_LAYOUT:
        x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
    x_embed = ttnn.reshape(x_embed, (1, 1, page_len, 2048))

    x = w.input_layernorm(x_embed, mode="prefill")
    kv = ttnn.linear(x, w.w_kv_a)  # [1, 1, page_len, 576]
    ttnn.deallocate(x)
    kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, page_len, 512])
    kv_rope = ttnn.slice(kv, [0, 0, 0, 512], [1, 1, page_len, 576])
    ttnn.deallocate(kv)

    kv_nope = w.kv_a_layernorm(kv_nope, mode="prefill")
    kv_rope = ttnn.experimental.rotary_embedding_llama(
        kv_rope,
        rope["cos_matrix"],
        rope["sin_matrix"],
        rope["trans_matrix"],
        is_decode_mode=False,
    )
    kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1, 1, page_len, 576]
    ttnn.deallocate(kv_nope)
    ttnn.deallocate(kv_rope)

    # Ensure the fill tensor matches the cache dtype. `paged_fill_cache` does not
    # currently support all dtype conversion combinations (e.g. BF16 -> BF8) and
    # may produce incorrect results if we rely on implicit conversion.
    if kvpe.dtype != kvpe_cache.dtype:
        kvpe_cast = ttnn.typecast(kvpe, dtype=kvpe_cache.dtype)
        ttnn.deallocate(kvpe)
        kvpe = kvpe_cast

    kvpe_ref = ttnn.to_torch(kvpe).reshape(1, 1, page_len, 576).cpu()

    # Fill cache for slot 0.
    ttnn.experimental.paged_fill_cache(kvpe_cache, kvpe, page_table=page_table_tt, batch_idx=0)

    cache_torch = ttnn.to_torch(kvpe_cache).cpu()  # [max_blocks, nkv, block_size, 576]

    # Reconstruct sequence order for each batch element using the page table.
    blocks = []
    for b in range(int(batch)):
        phys = page_table[b].to(torch.int64)  # [blocks_per_seq]
        blk = cache_torch.index_select(0, phys)  # [blocks_per_seq, nkv, block_size, 576]
        blk = blk.permute(1, 0, 2, 3).reshape(1, page_len, 576)  # [nkv=1, page_len, 576]
        blocks.append(blk)
    kvpe_from_cache = torch.stack(blocks, dim=0)  # [batch, nkv, page_len, 576]

    # Cleanup.
    ttnn.deallocate(kvpe)
    ttnn.deallocate(kvpe_cache)
    ttnn.deallocate(page_table_tt)
    ttnn.deallocate(x_embed)
    ttnn.deallocate(rope["cos_matrix"])
    ttnn.deallocate(rope["sin_matrix"])
    ttnn.deallocate(rope["trans_matrix"])

    return kvpe_ref, kvpe_from_cache


@torch.no_grad()
def debug_q_for_decode_prefill_vs_single_token_tt(
    *,
    device,
    snapshot_dir: Path,
    prefix_input_ids: torch.Tensor,
    next_token_id: int,
    block_size: int = 64,
    cache_dir: Optional[Path] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Debug helper: compute Q (KVPE space) for the decode token in two ways:
      1) "Prefill-style": build Q for the whole (padded) sequence and slice the last token.
      2) "Single-token": build Q from the new token alone and apply RoPE for its absolute position.

    Returns:
      (q_prefill, q_single_token) as torch tensors of shape [1, 1, num_heads=20, kvpe_dim=576].
    """
    prefix_input_ids = prefix_input_ids.cpu()
    if prefix_input_ids.ndim != 2 or prefix_input_ids.shape[0] != 1:
        raise ValueError(f"expected prefix_input_ids shape [1,S], got {tuple(prefix_input_ids.shape)}")

    seq_len = int(prefix_input_ids.shape[1])
    blocks_per_seq = _round_up(seq_len + 1, block_size) // block_size
    min_blocks_per_seq = max(1, (128 + int(block_size) - 1) // int(block_size))
    blocks_per_seq = max(blocks_per_seq, min_blocks_per_seq)
    page_len = int(blocks_per_seq * block_size)

    state = load_glm_lazy_state_dict(Path(snapshot_dir), num_layers=47)
    w = convert_layer0_weights(device=device, state=state, cache_dir=cache_dir)

    rope = make_rope_tensors(device=device, seq_len=page_len, rope_dim=64, rope_theta=1.0e6)
    cos_row = ttnn.slice(rope["cos_matrix"], [0, 0, seq_len, 0], [1, 1, seq_len + 1, 64])  # [1,1,1,64]
    sin_row = ttnn.slice(rope["sin_matrix"], [0, 0, seq_len, 0], [1, 1, seq_len + 1, 64])  # [1,1,1,64]

    # -------------------------
    # Q via prefill-style path
    # -------------------------
    full_padded = torch.zeros((1, page_len), dtype=prefix_input_ids.dtype)
    full_padded[:, :seq_len] = prefix_input_ids
    full_padded[:, seq_len] = int(next_token_id)

    x_embed = run_tt_embedding(device=device, token_ids=full_padded.to(torch.int32), tt_weight=w.embed_w)
    if x_embed.layout != ttnn.TILE_LAYOUT:
        x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
    x_embed = ttnn.reshape(x_embed, (1, 1, page_len, 2048))

    x = w.input_layernorm(x_embed, mode="prefill")
    q_a = ttnn.linear(x, w.w_q_a)
    q_a = w.q_a_layernorm(q_a, mode="prefill")
    q = ttnn.linear(q_a, w.w_q_b)
    ttnn.deallocate(q_a)
    ttnn.deallocate(x)

    q = ttnn.reshape(q, (1, page_len, 20, 256))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1,20,S,256]
    q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, 20, page_len, 192])
    q_rope = ttnn.slice(q, [0, 0, 0, 192], [1, 20, page_len, 256])
    ttnn.deallocate(q)

    q_nope = ttnn.linear(q_nope, w.w_kv_b1)  # [1,20,S,512]
    q_rope = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        rope["cos_matrix"],
        rope["sin_matrix"],
        rope["trans_matrix"],
        is_decode_mode=False,
    )  # [1,20,S,64]
    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [1,20,S,576]
    ttnn.deallocate(q_nope)
    ttnn.deallocate(q_rope)

    q_last = ttnn.slice(q_kvpe, [0, 0, seq_len, 0], [1, 20, seq_len + 1, 576])  # [1,20,1,576]
    ttnn.deallocate(q_kvpe)
    q_prefill = ttnn.permute(q_last, (0, 2, 1, 3))  # [1,1,20,576]
    ttnn.deallocate(q_last)

    # -------------------------
    # Q via single-token path
    # -------------------------
    token_ids = torch.tensor([[int(next_token_id)]], dtype=torch.int32)
    x_tok = run_tt_embedding(device=device, token_ids=token_ids, tt_weight=w.embed_w)
    if x_tok.layout != ttnn.TILE_LAYOUT:
        x_tok = ttnn.to_layout(x_tok, ttnn.TILE_LAYOUT)
    x_tok = ttnn.reshape(x_tok, (1, 1, 1, 2048))

    x_tok = w.input_layernorm(x_tok, mode="prefill")
    q_a = ttnn.linear(x_tok, w.w_q_a)
    q_a = w.q_a_layernorm(q_a, mode="prefill")
    q = ttnn.linear(q_a, w.w_q_b)
    ttnn.deallocate(q_a)
    ttnn.deallocate(x_tok)

    q = ttnn.reshape(q, (1, 1, 20, 256))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1,20,1,256]
    q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, 20, 1, 192])
    q_rope = ttnn.slice(q, [0, 0, 0, 192], [1, 20, 1, 256])
    ttnn.deallocate(q)

    q_nope = ttnn.linear(q_nope, w.w_kv_b1)  # [1,20,1,512]
    q_rope = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        cos_row,
        sin_row,
        rope["trans_matrix"],
        is_decode_mode=False,
    )  # [1,20,1,64]
    q_single = ttnn.concat([q_nope, q_rope], dim=-1)  # [1,20,1,576]
    ttnn.deallocate(q_nope)
    ttnn.deallocate(q_rope)

    q_single = ttnn.permute(q_single, (0, 2, 1, 3))  # [1,1,20,576]

    q_prefill_torch = ttnn.to_torch(q_prefill).reshape(1, 1, 20, 576).cpu()
    q_single_torch = ttnn.to_torch(q_single).reshape(1, 1, 20, 576).cpu()

    # Cleanup.
    ttnn.deallocate(q_prefill)
    ttnn.deallocate(q_single)
    ttnn.deallocate(x_embed)
    ttnn.deallocate(cos_row)
    ttnn.deallocate(sin_row)
    ttnn.deallocate(rope["cos_matrix"])
    ttnn.deallocate(rope["sin_matrix"])
    ttnn.deallocate(rope["trans_matrix"])

    return q_prefill_torch, q_single_torch


@torch.no_grad()
def debug_flash_mla_decode_attention_unpaged_vs_paged_tt(
    *,
    device,
    snapshot_dir: Path,
    prefix_input_ids: torch.Tensor,
    next_token_id: int,
    block_size: int = 64,
    cache_dir: Optional[Path] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Debug helper: run FlashMLA decode attention for the same Q/KVPE using:
      1) unpaged decode (flash_multi_latent_attention_decode)
      2) paged decode   (paged_flash_multi_latent_attention_decode)

    Returns:
      (attn_unpaged, attn_paged) as torch tensors of shape [1, 1, num_heads=20, head_dim_v=512].
    """
    prefix_input_ids = prefix_input_ids.cpu()
    if prefix_input_ids.ndim != 2 or prefix_input_ids.shape[0] != 1:
        raise ValueError(f"expected prefix_input_ids shape [1,S], got {tuple(prefix_input_ids.shape)}")

    seq_len = int(prefix_input_ids.shape[1])
    blocks_per_seq = _round_up(seq_len + 1, block_size) // block_size
    min_blocks_per_seq = max(1, (128 + int(block_size) - 1) // int(block_size))
    blocks_per_seq = max(blocks_per_seq, min_blocks_per_seq)
    page_len = int(blocks_per_seq * block_size)

    # Load + convert weights.
    state = load_glm_lazy_state_dict(Path(snapshot_dir), num_layers=47)
    w = convert_layer0_weights(device=device, state=state, cache_dir=cache_dir)

    # Build Q/KVPE for the padded window.
    full_padded = torch.zeros((1, page_len), dtype=prefix_input_ids.dtype)
    full_padded[:, :seq_len] = prefix_input_ids
    full_padded[:, seq_len] = int(next_token_id)

    x_embed = run_tt_embedding(device=device, token_ids=full_padded.to(torch.int32), tt_weight=w.embed_w)
    if x_embed.layout != ttnn.TILE_LAYOUT:
        x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
    x_embed = ttnn.reshape(x_embed, (1, 1, page_len, 2048))

    x = w.input_layernorm(x_embed, mode="prefill")

    q_a = ttnn.linear(x, w.w_q_a)
    q_a = w.q_a_layernorm(q_a, mode="prefill")
    q = ttnn.linear(q_a, w.w_q_b)
    ttnn.deallocate(q_a)

    q = ttnn.reshape(q, (1, page_len, 20, 256))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1,20,S,256]
    q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, 20, page_len, 192])
    q_rope = ttnn.slice(q, [0, 0, 0, 192], [1, 20, page_len, 256])
    ttnn.deallocate(q)
    q_nope = ttnn.linear(q_nope, w.w_kv_b1)  # [1,20,S,512]

    kv = ttnn.linear(x, w.w_kv_a)  # [1,1,S,576]
    ttnn.deallocate(x)
    kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, page_len, 512])
    kv_rope = ttnn.slice(kv, [0, 0, 0, 512], [1, 1, page_len, 576])
    ttnn.deallocate(kv)
    kv_nope = w.kv_a_layernorm(kv_nope, mode="prefill")

    rope = make_rope_tensors(device=device, seq_len=page_len, rope_dim=64, rope_theta=1.0e6)
    q_rope = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        rope["cos_matrix"],
        rope["sin_matrix"],
        rope["trans_matrix"],
        is_decode_mode=False,
    )  # [1,20,S,64]
    kv_rope = ttnn.experimental.rotary_embedding_llama(
        kv_rope,
        rope["cos_matrix"],
        rope["sin_matrix"],
        rope["trans_matrix"],
        is_decode_mode=False,
    )  # [1,1,S,64]

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [1,20,S,576]
    kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1,1,S,576]
    ttnn.deallocate(q_nope)
    ttnn.deallocate(q_rope)
    ttnn.deallocate(kv_nope)
    ttnn.deallocate(kv_rope)

    q_last = ttnn.slice(q_kvpe, [0, 0, seq_len, 0], [1, 20, seq_len + 1, 576])  # [1,20,1,576]
    ttnn.deallocate(q_kvpe)
    q_for_decode = ttnn.permute(q_last, (0, 2, 1, 3))  # [1,1,20,576]
    ttnn.deallocate(q_last)

    pos = ttnn.from_torch(
        torch.tensor([seq_len], dtype=torch.int32), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    scale = _mla_scale()
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Unpaged decode.
    attn_unpaged = ttnn.transformer.flash_multi_latent_attention_decode(
        q_for_decode,
        kvpe,
        head_dim_v=512,
        cur_pos_tensor=pos,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # [1,1,H_padded,512]

    # Paged decode.
    kvpe_cache = _alloc_paged_kvpe_cache(
        device=device,
        max_num_blocks=int(1 * blocks_per_seq),
        block_size=block_size,
        kvpe_dim=576,
        dtype=ttnn.bfloat16,
    )
    page_table = _alloc_contiguous_page_table(batch=1, blocks_per_seq=blocks_per_seq)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.experimental.paged_fill_cache(kvpe_cache, kvpe, page_table=page_table_tt, batch_idx=0)

    attn_paged = ttnn.transformer.paged_flash_multi_latent_attention_decode(
        q_for_decode,
        kvpe_cache,
        head_dim_v=512,
        page_table_tensor=page_table_tt,
        cur_pos_tensor=pos,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # [1,1,H_padded,512]

    # Slice padded heads back to 20.
    attn_unpaged = ttnn.slice(attn_unpaged, [0, 0, 0, 0], [1, 1, 20, 512])
    attn_paged = ttnn.slice(attn_paged, [0, 0, 0, 0], [1, 1, 20, 512])

    out_unpaged = ttnn.to_torch(attn_unpaged).reshape(1, 1, 20, 512).cpu()
    out_paged = ttnn.to_torch(attn_paged).reshape(1, 1, 20, 512).cpu()

    # Cleanup.
    ttnn.deallocate(attn_unpaged)
    ttnn.deallocate(attn_paged)
    ttnn.deallocate(q_for_decode)
    ttnn.deallocate(pos)
    ttnn.deallocate(kvpe)
    ttnn.deallocate(kvpe_cache)
    ttnn.deallocate(page_table_tt)
    ttnn.deallocate(x_embed)
    ttnn.deallocate(rope["cos_matrix"])
    ttnn.deallocate(rope["sin_matrix"])
    ttnn.deallocate(rope["trans_matrix"])

    return out_unpaged, out_paged


@torch.no_grad()
def run_layer0_decode_one_step_tt(
    *,
    device,
    snapshot_dir: Path,
    prefix_input_ids: torch.Tensor,
    next_token_id: int,
    batch: int = 32,
    block_size: int = 64,
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    Layer-0 decode (one step) using a paged KVPE cache.

    This is a bring-up harness:
    - Uses a padded decode batch (default 32) where only slot 0 is active.
    - Uses RoPE "prefill-mode" over the padded batch dimension to apply per-user rotations
      without relying on the decode-mode RoPE pipeline yet.

    Returns:
      torch.Tensor [1, hidden] for the decoded token output (post-MLP residual) for batch slot 0.
    """
    prefix_input_ids = prefix_input_ids.cpu()
    if prefix_input_ids.ndim != 2 or prefix_input_ids.shape[0] != 1:
        raise ValueError(f"expected prefix_input_ids shape [1,S], got {tuple(prefix_input_ids.shape)}")

    seq_len = int(prefix_input_ids.shape[1])
    if seq_len <= 0:
        raise ValueError("prefix_input_ids must be non-empty")

    # Allocate enough blocks for (seq_len + 1) tokens.
    blocks_per_seq = _round_up(seq_len + 1, block_size) // block_size
    min_blocks_per_seq = max(1, (128 + int(block_size) - 1) // int(block_size))
    blocks_per_seq = max(blocks_per_seq, min_blocks_per_seq)
    kvpe_cache = _alloc_paged_kvpe_cache(
        device=device,
        max_num_blocks=int(batch * blocks_per_seq),
        block_size=block_size,
        kvpe_dim=576,
        dtype=ttnn.bfloat16,
    )
    page_table = _alloc_contiguous_page_table(batch=batch, blocks_per_seq=blocks_per_seq)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load + convert layer0 weights.
    state = load_glm_lazy_state_dict(Path(snapshot_dir), num_layers=47)
    w = convert_layer0_weights(device=device, state=state, cache_dir=cache_dir)

    # -----------------------------
    # Prefill KVPE cache for slot 0 (prefix + next token)
    # -----------------------------
    # Fill cache up to a whole number of blocks.
    page_len = blocks_per_seq * block_size
    full_padded = torch.zeros((1, page_len), dtype=prefix_input_ids.dtype)
    full_padded[:, :seq_len] = prefix_input_ids
    full_padded[:, seq_len] = int(next_token_id)

    rope_prefill = make_rope_tensors(device=device, seq_len=page_len, rope_dim=64, rope_theta=1.0e6)

    # KVPE for the full sequence (slot 0 only).
    x_embed = run_tt_embedding(device=device, token_ids=full_padded.to(torch.int32), tt_weight=w.embed_w)
    if x_embed.layout != ttnn.TILE_LAYOUT:
        x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
    x_embed = ttnn.reshape(x_embed, (1, 1, page_len, 2048))

    x = w.input_layernorm(x_embed, mode="prefill")
    kv = ttnn.linear(x, w.w_kv_a)  # [1, 1, page_len, 576]
    kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, page_len, 512])
    kv_rope = ttnn.slice(kv, [0, 0, 0, 512], [1, 1, page_len, 576])
    ttnn.deallocate(kv)

    kv_nope = w.kv_a_layernorm(kv_nope, mode="prefill")
    kv_rope = ttnn.experimental.rotary_embedding_llama(
        kv_rope,
        rope_prefill["cos_matrix"],
        rope_prefill["sin_matrix"],
        rope_prefill["trans_matrix"],
        is_decode_mode=False,
    )
    kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1, 1, page_len, 576]
    ttnn.deallocate(kv_nope)
    ttnn.deallocate(kv_rope)

    # Fill slot 0's cache using its page table row.
    ttnn.experimental.paged_fill_cache(kvpe_cache, kvpe, page_table=page_table_tt, batch_idx=0)
    ttnn.deallocate(kvpe)
    ttnn.deallocate(x_embed)

    # -----------------------------
    # Decode step (slot 0 active)
    # -----------------------------
    # Build padded decode batch.
    tokens = torch.zeros((batch, 1), dtype=torch.int32)
    tokens[0, 0] = int(next_token_id)

    positions = torch.zeros((batch,), dtype=torch.int32)
    positions[0] = seq_len  # position of the new token
    tt_positions = ttnn.from_torch(
        positions,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # RoPE matrices for decode.
    #
    # We need cos/sin rows shaped as [1, 1, B, 64] to rotate the RoPE slice of Q
    # for each active slot's absolute position.
    #
    # NOTE: Avoid `ttnn.embedding` gather for these RoPE caches: during bring-up
    # it was observed to select incorrect cos/sin rows. Use slice+concat for
    # correctness (optimize later if needed).
    cos_rows = []
    sin_rows = []
    for b in range(batch):
        pos_b = int(positions[b])
        # vLLM uses -1 for inactive slots; rotate them with position 0.
        if pos_b < 0:
            pos_b = 0
        cos_rows.append(ttnn.slice(rope_prefill["cos_matrix"], [0, 0, pos_b, 0], [1, 1, pos_b + 1, 64]))  # [1,1,1,64]
        sin_rows.append(ttnn.slice(rope_prefill["sin_matrix"], [0, 0, pos_b, 0], [1, 1, pos_b + 1, 64]))  # [1,1,1,64]

    if batch == 1:
        cos_batch = cos_rows[0]
        sin_batch = sin_rows[0]
    else:
        cos_batch = ttnn.concat(cos_rows, dim=2)  # [1,1,B,64]
        sin_batch = ttnn.concat(sin_rows, dim=2)  # [1,1,B,64]
        for t in cos_rows:
            ttnn.deallocate(t)
        for t in sin_rows:
            ttnn.deallocate(t)

    # Decode token embedding: shape to [1,1,B,2048] so batch is in tile height.
    x_embed_tok = run_tt_embedding(device=device, token_ids=tokens, tt_weight=w.embed_w)
    if x_embed_tok.layout != ttnn.TILE_LAYOUT:
        x_embed_tok = ttnn.to_layout(x_embed_tok, ttnn.TILE_LAYOUT)
    x_embed_tok = ttnn.reshape(x_embed_tok, (1, batch, 1, 2048))
    x_embed_tok = ttnn.permute(x_embed_tok, (0, 2, 1, 3))  # [1,1,B,2048]

    residual = x_embed_tok
    x = w.input_layernorm(x_embed_tok, mode="decode")

    # Q path.
    q_a = ttnn.linear(x, w.w_q_a)  # [1,1,B,768]
    q_a = w.q_a_layernorm(q_a, mode="decode")
    q = ttnn.linear(q_a, w.w_q_b)  # [1,1,B,5120]
    ttnn.deallocate(q_a)

    q = ttnn.reshape(q, (1, batch, 20, 256))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1,20,B,256]
    q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, 20, batch, 192])
    q_rope = ttnn.slice(q, [0, 0, 0, 192], [1, 20, batch, 256])
    ttnn.deallocate(q)

    q_nope = ttnn.linear(q_nope, w.w_kv_b1)  # [1,20,B,512]
    q_rope = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        cos_batch,
        sin_batch,
        rope_prefill["trans_matrix"],
        is_decode_mode=False,
    )  # [1,20,B,64]

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [1,20,B,576]
    ttnn.deallocate(q_nope)
    ttnn.deallocate(q_rope)

    # We prefilled the cache (including the new token) above. For bring-up correctness,
    # avoid `paged_update_cache` here since it requires a sharded input tensor.
    ttnn.deallocate(x)

    # FlashMLA decode.
    q_for_decode = ttnn.permute(q_kvpe, (0, 2, 1, 3))  # [1,B,20,576]
    ttnn.deallocate(q_kvpe)

    scale = _mla_scale()
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,  # not used in decode
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Workaround: explicitly provide the V cache slice instead of relying on the "V is a subset of K" path.
    # This matches how vLLM traditionally stores KV caches (separate K/V).
    v_cache = ttnn.slice(kvpe_cache, [0, 0, 0, 0], [int(batch * blocks_per_seq), 1, int(block_size), 512])
    attn_latent = ttnn.transformer.paged_flash_multi_latent_attention_decode(
        q_for_decode,
        kvpe_cache,
        v_cache,
        head_dim_v=512,
        page_table_tensor=page_table_tt,
        cur_pos_tensor=tt_positions,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # [1,B,H_padded,512]
    ttnn.deallocate(v_cache)
    ttnn.deallocate(q_for_decode)

    # Slice padded heads back to 20.
    attn_latent = ttnn.slice(attn_latent, [0, 0, 0, 0], [1, batch, 20, 512])
    attn_latent = ttnn.permute(attn_latent, (0, 2, 1, 3))  # [1,20,B,512]

    v = ttnn.linear(attn_latent, w.w_kv_b2)  # [1,20,B,256]
    ttnn.deallocate(attn_latent)

    v = ttnn.permute(v, (0, 2, 1, 3))  # [1,B,20,256]
    v = ttnn.reshape(v, (1, batch, 1, 20 * 256))
    v = ttnn.permute(v, (0, 2, 1, 3))  # [1,1,B,5120]

    attn_out = ttnn.linear(v, w.w_o)  # [1,1,B,2048]
    ttnn.deallocate(v)

    x_attn_out = residual + attn_out
    ttnn.deallocate(attn_out)

    # MLP (dense) for layer 0.
    residual = x_attn_out
    x = w.post_attention_layernorm(x_attn_out, mode="decode")
    gate = ttnn.linear(x, w.w_mlp_gate)
    up = ttnn.linear(x, w.w_mlp_up)
    ttnn.deallocate(x)
    x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(gate)
    ttnn.deallocate(up)
    mlp_out = ttnn.linear(x_ff, w.w_mlp_down)
    ttnn.deallocate(x_ff)

    x_mlp_out = residual + mlp_out
    ttnn.deallocate(mlp_out)

    # Return only slot 0.
    x0 = ttnn.slice(x_mlp_out, [0, 0, 0, 0], [1, 1, 1, 2048])
    out = ttnn.to_torch(x0).reshape(1, 2048).cpu()

    # Cleanup.
    ttnn.deallocate(x0)
    ttnn.deallocate(x_mlp_out)
    ttnn.deallocate(x_attn_out)
    ttnn.deallocate(x_embed_tok)
    ttnn.deallocate(tt_positions)
    ttnn.deallocate(page_table_tt)
    ttnn.deallocate(kvpe_cache)
    ttnn.deallocate(cos_batch)
    ttnn.deallocate(sin_batch)
    ttnn.deallocate(rope_prefill["cos_matrix"])
    ttnn.deallocate(rope_prefill["sin_matrix"])
    ttnn.deallocate(rope_prefill["trans_matrix"])

    return out


@torch.no_grad()
def run_layer0_decode_one_step_update_cache_tt(
    *,
    device,
    snapshot_dir: Path,
    prefix_input_ids: torch.Tensor,
    next_token_id: int,
    batch: int = 32,
    block_size: int = 64,
    cache_dir: Optional[Path] = None,
) -> torch.Tensor:
    """Layer-0 decode (one step) using paged_update_cache for the new token.

    This is a bring-up harness for vLLM-style semantics:
    - Prefill fills the KVPE cache for the prompt only.
    - Decode computes KVPE/Q for the new token, updates the cache at cur_pos,
      then runs paged FlashMLA decode attention.

    Returns:
      torch.Tensor [1, hidden] for batch slot 0.
    """
    prefix_input_ids = prefix_input_ids.cpu()
    if prefix_input_ids.ndim != 2 or prefix_input_ids.shape[0] != 1:
        raise ValueError(f"expected prefix_input_ids shape [1,S], got {tuple(prefix_input_ids.shape)}")

    seq_len = int(prefix_input_ids.shape[1])
    if seq_len <= 0:
        raise ValueError("prefix_input_ids must be non-empty")

    # Allocate enough blocks for (seq_len + 1) tokens.
    blocks_per_seq = _round_up(seq_len + 1, block_size) // block_size
    min_blocks_per_seq = max(1, (128 + int(block_size) - 1) // int(block_size))
    blocks_per_seq = max(blocks_per_seq, min_blocks_per_seq)

    page_len = int(blocks_per_seq * block_size)
    kvpe_cache = _alloc_paged_kvpe_cache(
        device=device,
        max_num_blocks=int(batch * blocks_per_seq),
        block_size=block_size,
        kvpe_dim=576,
        dtype=ttnn.bfloat8_b,
    )
    page_table = _alloc_contiguous_page_table(batch=batch, blocks_per_seq=blocks_per_seq)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Load + convert layer0 weights.
    state = load_glm_lazy_state_dict(Path(snapshot_dir), num_layers=47)
    w = convert_layer0_weights(device=device, state=state, cache_dir=cache_dir)

    # RoPE cache for the padded page length.
    rope = make_rope_tensors(device=device, seq_len=page_len, rope_dim=64, rope_theta=1.0e6)

    # -----------------------------
    # Prefill KVPE cache for slot 0 (prompt only)
    # -----------------------------
    full_padded = torch.zeros((1, page_len), dtype=prefix_input_ids.dtype)
    full_padded[:, :seq_len] = prefix_input_ids

    x_embed = run_tt_embedding(device=device, token_ids=full_padded.to(torch.int32), tt_weight=w.embed_w)
    if x_embed.layout != ttnn.TILE_LAYOUT:
        x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
    x_embed = ttnn.reshape(x_embed, (1, 1, page_len, 2048))

    x = w.input_layernorm(x_embed, mode="prefill")
    kv = ttnn.linear(x, w.w_kv_a)  # [1, 1, page_len, 576]
    kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, page_len, 512])
    kv_rope = ttnn.slice(kv, [0, 0, 0, 512], [1, 1, page_len, 576])
    ttnn.deallocate(kv)

    kv_nope = w.kv_a_layernorm(kv_nope, mode="prefill")
    kv_rope = ttnn.experimental.rotary_embedding_llama(
        kv_rope,
        rope["cos_matrix"],
        rope["sin_matrix"],
        rope["trans_matrix"],
        is_decode_mode=False,
    )
    kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1, 1, page_len, 576]
    ttnn.deallocate(kv_nope)
    ttnn.deallocate(kv_rope)

    if kvpe.dtype != ttnn.bfloat8_b:
        kvpe = ttnn.typecast(kvpe, dtype=ttnn.bfloat8_b)

    ttnn.experimental.paged_fill_cache(kvpe_cache, kvpe, page_table=page_table_tt, batch_idx=0)
    ttnn.deallocate(kvpe)
    ttnn.deallocate(x_embed)

    # -----------------------------
    # Decode step (slot 0 active)
    # -----------------------------
    tokens = torch.zeros((batch, 1), dtype=torch.int32)
    tokens[0, 0] = int(next_token_id)

    # Padded batch; only slot 0 is active in this harness.
    # Use 0 for inactive slots to avoid negative update indices.
    positions = torch.zeros((batch,), dtype=torch.int32)
    positions[0] = seq_len
    tt_positions = ttnn.from_torch(
        positions,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # RoPE cos/sin rows for each batch slot.
    cos_rows = []
    sin_rows = []
    for b in range(batch):
        pos_b = int(positions[b])
        if pos_b < 0:
            pos_b = 0
        cos_rows.append(ttnn.slice(rope["cos_matrix"], [0, 0, pos_b, 0], [1, 1, pos_b + 1, 64]))
        sin_rows.append(ttnn.slice(rope["sin_matrix"], [0, 0, pos_b, 0], [1, 1, pos_b + 1, 64]))

    if batch == 1:
        cos_batch = cos_rows[0]
        sin_batch = sin_rows[0]
    else:
        cos_batch = ttnn.concat(cos_rows, dim=2)  # [1,1,B,64]
        sin_batch = ttnn.concat(sin_rows, dim=2)  # [1,1,B,64]
        for t in cos_rows:
            ttnn.deallocate(t)
        for t in sin_rows:
            ttnn.deallocate(t)

    # Decode token embedding: shape to [1,1,B,2048] so batch is in tile height.
    x_embed_tok = run_tt_embedding(device=device, token_ids=tokens, tt_weight=w.embed_w)
    if x_embed_tok.layout != ttnn.TILE_LAYOUT:
        x_embed_tok = ttnn.to_layout(x_embed_tok, ttnn.TILE_LAYOUT)
    x_embed_tok = ttnn.reshape(x_embed_tok, (1, batch, 1, 2048))
    x_embed_tok = ttnn.permute(x_embed_tok, (0, 2, 1, 3))  # [1,1,B,2048]

    residual = x_embed_tok
    x = w.input_layernorm(x_embed_tok, mode="decode")

    # KVPE for the new token -> update cache at cur_pos.
    kv = ttnn.linear(x, w.w_kv_a)  # [1,1,B,576]
    kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, batch, 512])
    kv_rope = ttnn.slice(kv, [0, 0, 0, 512], [1, 1, batch, 576])
    ttnn.deallocate(kv)

    kv_nope = w.kv_a_layernorm(kv_nope, mode="decode")
    kv_rope = ttnn.experimental.rotary_embedding_llama(
        kv_rope,
        cos_batch,
        sin_batch,
        rope["trans_matrix"],
        is_decode_mode=False,
    )  # [1,1,B,64]
    kvpe_new = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1,1,B,576]
    ttnn.deallocate(kv_nope)
    ttnn.deallocate(kv_rope)

    # paged_update_cache currently requires a sharded input tensor.
    #
    # Prepare a height-sharded tensor with per-(user, token) KVPE vectors laid
    # out as a 2D matrix of shape [TILE_SIZE, kvpe_dim] per user.
    kvpe_new = ttnn.pad(kvpe_new, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)  # [1,32,B,576]
    kvpe_new = ttnn.permute(kvpe_new, (0, 2, 1, 3))  # [1,B,32,576]

    # Create an explicit shard spec: shard across the (B*32) height dimension
    # so each user gets one 32x576 shard.
    grid_size = device.compute_with_storage_grid_size()
    user_grid = ttnn.num_cores_to_corerangeset(int(batch), grid_size, row_wise=True)
    kvpe_sharded_cfg = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, 576),
        core_grid=user_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    kvpe_new = ttnn.to_memory_config(kvpe_new, kvpe_sharded_cfg)
    ttnn.experimental.paged_update_cache(
        kvpe_cache,
        kvpe_new,
        update_idxs_tensor=tt_positions,
        page_table=page_table_tt,
    )
    ttnn.deallocate(kvpe_new)

    # Q path.
    q_a = ttnn.linear(x, w.w_q_a)  # [1,1,B,768]
    q_a = w.q_a_layernorm(q_a, mode="decode")
    q = ttnn.linear(q_a, w.w_q_b)  # [1,1,B,5120]
    ttnn.deallocate(q_a)

    q = ttnn.reshape(q, (1, batch, 20, 256))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [1,20,B,256]
    q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, 20, batch, 192])
    q_rope = ttnn.slice(q, [0, 0, 0, 192], [1, 20, batch, 256])
    ttnn.deallocate(q)

    q_nope = ttnn.linear(q_nope, w.w_kv_b1)  # [1,20,B,512]
    q_rope = ttnn.experimental.rotary_embedding_llama(
        q_rope,
        cos_batch,
        sin_batch,
        rope["trans_matrix"],
        is_decode_mode=False,
    )  # [1,20,B,64]

    q_kvpe = ttnn.concat([q_nope, q_rope], dim=-1)  # [1,20,B,576]
    ttnn.deallocate(q_nope)
    ttnn.deallocate(q_rope)

    ttnn.deallocate(x)

    # FlashMLA decode.
    q_for_decode = ttnn.permute(q_kvpe, (0, 2, 1, 3))  # [1,B,20,576]
    ttnn.deallocate(q_kvpe)

    scale = _mla_scale()
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,  # not used in decode
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Workaround: explicitly provide the V cache slice instead of relying on the "V is a subset of K" path.
    v_cache = ttnn.slice(kvpe_cache, [0, 0, 0, 0], [int(batch * blocks_per_seq), 1, int(block_size), 512])
    attn_latent = ttnn.transformer.paged_flash_multi_latent_attention_decode(
        q_for_decode,
        kvpe_cache,
        v_cache,
        head_dim_v=512,
        page_table_tensor=page_table_tt,
        cur_pos_tensor=tt_positions,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )  # [1,B,H_padded,512]
    ttnn.deallocate(v_cache)
    ttnn.deallocate(q_for_decode)

    attn_latent = ttnn.slice(attn_latent, [0, 0, 0, 0], [1, batch, 20, 512])
    attn_latent = ttnn.permute(attn_latent, (0, 2, 1, 3))  # [1,20,B,512]

    v = ttnn.linear(attn_latent, w.w_kv_b2)  # [1,20,B,256]
    ttnn.deallocate(attn_latent)

    v = ttnn.permute(v, (0, 2, 1, 3))  # [1,B,20,256]
    v = ttnn.reshape(v, (1, batch, 1, 20 * 256))
    v = ttnn.permute(v, (0, 2, 1, 3))  # [1,1,B,5120]

    attn_out = ttnn.linear(v, w.w_o)  # [1,1,B,2048]
    ttnn.deallocate(v)

    x_attn_out = residual + attn_out
    ttnn.deallocate(attn_out)

    # MLP (dense) for layer 0.
    residual = x_attn_out
    x = w.post_attention_layernorm(x_attn_out, mode="decode")
    gate = ttnn.linear(x, w.w_mlp_gate)
    up = ttnn.linear(x, w.w_mlp_up)
    ttnn.deallocate(x)
    x_ff = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
    ttnn.deallocate(gate)
    ttnn.deallocate(up)
    mlp_out = ttnn.linear(x_ff, w.w_mlp_down)
    ttnn.deallocate(x_ff)

    x_mlp_out = residual + mlp_out
    ttnn.deallocate(mlp_out)

    x0 = ttnn.slice(x_mlp_out, [0, 0, 0, 0], [1, 1, 1, 2048])
    out = ttnn.to_torch(x0).reshape(1, 2048).cpu()

    # Cleanup.
    ttnn.deallocate(x0)
    ttnn.deallocate(x_mlp_out)
    ttnn.deallocate(x_attn_out)
    ttnn.deallocate(x_embed_tok)
    ttnn.deallocate(tt_positions)
    ttnn.deallocate(page_table_tt)
    ttnn.deallocate(kvpe_cache)
    ttnn.deallocate(cos_batch)
    ttnn.deallocate(sin_batch)
    ttnn.deallocate(rope["cos_matrix"])
    ttnn.deallocate(rope["sin_matrix"])
    ttnn.deallocate(rope["trans_matrix"])

    return out
