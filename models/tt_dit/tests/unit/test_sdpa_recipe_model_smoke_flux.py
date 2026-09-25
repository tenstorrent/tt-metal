# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests: opt-in SDPA recipes in the FLUX.1 shared joint Attention (blocks/attention.py)
and the FLUX.2 Attention (blocks/attention_opt.py).

Random weights, small shapes (4 heads, D128). Each case runs legacy (the
module's legacy SDPA config, via tests/unit/sdpa_legacy.py), the default recipe (sdpa_precision=None),
FAST,
ACCURATE and LOW_PRECISION (bfp8 KV) on the same torch weights and inputs, and compares each output
against the torch reference of the module (and against the legacy tt output).

Cases:
- FLUX.1 joint (added KV projection, prompt stream) and no-prompt (empty joint segment), on 1x1
  (joint SDPA) and 1x2 (ring joint SDPA, sp=2); plus an unaligned spatial length on 1x2 (padded).
- FLUX.2 dual-stream joint attention and the single-stream concatenated sequence (empty joint),
  on 1x1 and 1x2.
"""

from __future__ import annotations

import diffusers.models.attention_processor
import pytest
import torch

import ttnn

from ...blocks import attention as flux1_attention
from ...blocks import attention_opt as flux2_attention
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils import tensor
from ...utils.tensor import bf16_tensor
from .sdpa_legacy import LEGACY, sdpa_variant

HEADS = 4
HEAD_DIM = 128
DIM = HEADS * HEAD_DIM

VARIANTS = {
    "legacy": (LEGACY, None),  # the module's legacy SDPA config (tests/unit/sdpa_legacy.py)
    "default": (None, None),  # the module's default recipe (sdpa_precision_default)
    "FAST": (ttnn.SDPAPrecision.FAST, None),
    "ACCURATE": (ttnn.SDPAPrecision.ACCURATE, None),
    "LOW_PRECISION": (ttnn.SDPAPrecision.LOW_PRECISION, ttnn.bfloat8_b),
}
# The default recipe is FAST (sdpa_precision_default), so "default" uses FAST's gates.
ABS_BOUND = {"default": 3.0, "FAST": 3.0, "ACCURATE": 1.0, "LOW_PRECISION": 3.0}
MARGIN = 1.0  # allowed excess over the legacy tt L2 vs torch (percentage points)

# 1x1 runs without fabric (a 1x1 submesh with FABRIC_1D fails the router handshake on a 2-chip host).
MESHES = pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params"),
    [
        pytest.param((1, 1), 0, 1, {}, id="1x1"),
        pytest.param((1, 2), 1, 0, {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="1x2sp1"),
    ],
    indirect=["mesh_device", "device_params"],
)


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return (100.0 * torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def _parallel(mesh_device, sp_axis, tp_axis):
    sp = tuple(mesh_device.shape)[sp_axis]
    tp = tuple(mesh_device.shape)[tp_axis]
    assert tp == 1
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=0, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp, mesh_axis=sp_axis),
    )
    return sp, parallel_config


def _rope(seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved (pairwise) RoPE tables [seq, HEAD_DIM], as FLUX uses."""
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    theta = torch.arange(seq).float()[:, None] * inv[None, :]
    return theta.cos().repeat_interleave(2, dim=-1), theta.sin().repeat_interleave(2, dim=-1)


def _gate(results: dict, record_property, prefix: str) -> None:
    """results: variant -> list of (name, tt_output, torch_reference); legacy must be present."""
    failures = []
    legacy = results["legacy"]
    legacy_l2 = {name: _l2(tt, ref) for name, tt, ref in legacy}
    for name, value in legacy_l2.items():
        record_property(f"{prefix}.legacy.{name}.l2_vs_torch", round(value, 4))
    for variant, outputs in results.items():
        if variant == "legacy":
            continue
        for (name, tt, ref), (_, legacy_tt, _) in zip(outputs, legacy):
            vs_torch = _l2(tt, ref)
            vs_legacy = _l2(tt, legacy_tt)
            record_property(f"{prefix}.{variant}.{name}.l2_vs_torch", round(vs_torch, 4))
            record_property(f"{prefix}.{variant}.{name}.l2_vs_legacy", round(vs_legacy, 4))
            bound = ABS_BOUND[variant]
            base = legacy_l2[name]
            # Absolute bound vs torch. If the legacy tt output itself exceeds it (bf16 weights and the
            # legacy HiFi2 SDPA config), the recipe must instead be no worse than legacy vs torch.
            # LOW_PRECISION also rounds its inputs (RNE7 Q, RNE5+BFP8 KV), so it may scale with the
            # legacy error: bias-free FLUX.2 has a small reference norm (legacy ~2.5%, E ~1.2x that; allow 1.3x).
            relative_cap = 1.3 * base if variant == "LOW_PRECISION" else base if base > bound else bound
            absolute_ok = vs_torch <= max(bound, relative_cap)
            if not absolute_ok or vs_torch > base + MARGIN:
                failures.append(
                    f"{variant}/{name}: l2 vs torch {vs_torch:.3f}% (legacy {base:.3f}%), "
                    f"vs legacy {vs_legacy:.3f}%, bound {bound}%"
                )
            print(
                f"{prefix} {variant:14s} {name:8s} vs_torch={vs_torch:.4f}% legacy={base:.4f}% vs_legacy={vs_legacy:.4f}%"
            )
    assert not failures, "; ".join(failures)


# ---------------------------------------------------------------------------------------------
# FLUX.1 shared joint Attention (blocks/attention.py)
# ---------------------------------------------------------------------------------------------


@MESHES
@pytest.mark.parametrize(
    ("spatial_seq_len", "prompt_seq_len"),
    [
        pytest.param(1024, 128, id="s1024_p128"),
        pytest.param(1024, 0, id="s1024_noprompt"),
        pytest.param(1000, 256, id="s1000_p256"),
    ],
)
def test_flux1_attention_sdpa_recipes(
    mesh_device, sp_axis, tp_axis, spatial_seq_len, prompt_seq_len, record_property
) -> None:
    torch.manual_seed(0)
    sp_factor, parallel_config = _parallel(mesh_device, sp_axis, tp_axis)
    if sp_factor == 1 and spatial_seq_len % 32:
        pytest.skip("FLUX.1 Attention pads the spatial sequence only with sequence parallelism")

    joint = prompt_seq_len > 0
    torch_model = diffusers.models.attention_processor.Attention(
        query_dim=DIM,
        added_kv_proj_dim=DIM if joint else None,
        dim_head=HEAD_DIM,
        heads=HEADS,
        out_dim=DIM,
        context_pre_only=not joint,
        pre_only=not joint,
        bias=True,
        qk_norm="rms_norm",
        eps=1e-6,
        processor=diffusers.models.attention_processor.FluxAttnProcessor2_0(),
    )
    torch_model.eval()
    state = torch_model.state_dict()

    spatial = torch.randn((1, spatial_seq_len, DIM))
    prompt = torch.randn((1, prompt_seq_len, DIM)) if joint else None
    rope_cos, rope_sin = _rope(prompt_seq_len + spatial_seq_len)
    with torch.no_grad():
        out = torch_model.forward(spatial, prompt, image_rotary_emb=(rope_cos, rope_sin))
    torch_spatial, torch_prompt = out if joint else (out, None)

    Attention = flux1_attention.Attention
    pad = lambda x: Attention.pad_spatial_sequence(x, sp_factor=sp_factor)  # noqa: E731

    results = {}
    for variant, (precision, kv_dtype) in VARIANTS.items():
        with sdpa_variant(precision) as precision:
            ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
            tt_model = Attention(
                query_dim=DIM,
                head_dim=HEAD_DIM,
                heads=HEADS,
                out_dim=DIM,
                added_kv_proj_dim=DIM if joint else 0,
                context_pre_only=not joint,
                pre_only=not joint,
                eps=1e-6,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=None,
                sdpa_precision=precision,
                sdpa_kv_dtype=kv_dtype,
            )
            tt_model.load_torch_state_dict(dict(state))

            tt_spatial = bf16_tensor(pad(spatial), device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
            tt_prompt = bf16_tensor(prompt, device=mesh_device) if joint else None
            spatial_rope = tuple(
                bf16_tensor(pad(t[prompt_seq_len:]), device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
                for t in (rope_cos, rope_sin)
            )
            prompt_rope = (
                tuple(bf16_tensor(t[:prompt_seq_len], device=mesh_device) for t in (rope_cos, rope_sin))
                if joint
                else None
            )

            tt_spatial_out, tt_prompt_out = tt_model.forward(
                spatial=tt_spatial,
                prompt=tt_prompt,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_seq_len,
            )
            outputs = [
                (
                    "spatial",
                    tensor.to_torch(tt_spatial_out, mesh_axes=[..., sp_axis, tp_axis])
                    .reshape(-1, DIM)[:spatial_seq_len]
                    .float(),
                    torch_spatial.reshape(-1, DIM),
                )
            ]
            if joint:
                outputs.append(
                    (
                        "prompt",
                        tensor.to_torch(tt_prompt_out, mesh_axes=[..., None, tp_axis]).reshape(-1, DIM).float(),
                        torch_prompt.reshape(-1, DIM),
                    )
                )
            results[variant] = outputs
            del tt_model, ccl_manager

    _gate(results, record_property, f"flux1.{tuple(mesh_device.shape)}.s{spatial_seq_len}.p{prompt_seq_len}")


# ---------------------------------------------------------------------------------------------
# FLUX.2 Attention (blocks/attention_opt.py)
# ---------------------------------------------------------------------------------------------


def _torch_rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _torch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos + rot * sin


def _torch_flux2_attention(state, seq1, seq2, rope1, rope2):
    """Reference: per-head QK RMSNorm + interleaved RoPE, joint SDPA over [seq1 | seq2], out projections.

    seq2=None is the single-stream (pre_only) case: the output is the attention result before to_out.
    """

    def heads(x):
        return x.reshape(x.shape[0], x.shape[1], HEADS, HEAD_DIM).transpose(1, 2)

    def qkv(x, prefix_q, prefix_k, prefix_v, norm_q, norm_k, rope):
        q = heads(x @ state[f"{prefix_q}.weight"].T)
        k = heads(x @ state[f"{prefix_k}.weight"].T)
        v = heads(x @ state[f"{prefix_v}.weight"].T)
        q = _torch_rope(_torch_rms(q, state[norm_q]), *rope)
        k = _torch_rope(_torch_rms(k, state[norm_k]), *rope)
        return q, k, v

    q, k, v = qkv(seq1, "to_q", "to_k", "to_v", "norm_q.weight", "norm_k.weight", rope1)
    n1 = seq1.shape[1]
    if seq2 is not None:
        aq, ak, av = qkv(
            seq2, "add_q_proj", "add_k_proj", "add_v_proj", "norm_added_q.weight", "norm_added_k.weight", rope2
        )
        q, k, v = (torch.cat([a, b], dim=2) for a, b in ((q, aq), (k, ak), (v, av)))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).reshape(seq1.shape[0], -1, DIM)
    if seq2 is None:
        return out, None
    o1, o2 = out[:, :n1], out[:, n1:]
    return o1 @ state["to_out.0.weight"].T, o2 @ state["to_add_out.weight"].T


@MESHES
@pytest.mark.parametrize(
    ("seq1_len", "seq2_len"),
    [
        pytest.param(1024, 128, id="joint_s1024_p128"),
        pytest.param(1024, 0, id="single_stream_s1024"),
    ],
)
def test_flux2_attention_sdpa_recipes(mesh_device, sp_axis, tp_axis, seq1_len, seq2_len, record_property) -> None:
    torch.manual_seed(0)
    sp_factor, parallel_config = _parallel(mesh_device, sp_axis, tp_axis)
    joint = seq2_len > 0

    def w(*shape):
        return torch.randn(shape) / shape[-1] ** 0.5

    state = {
        "to_q.weight": w(DIM, DIM),
        "to_k.weight": w(DIM, DIM),
        "to_v.weight": w(DIM, DIM),
        "norm_q.weight": torch.ones(HEAD_DIM),
        "norm_k.weight": torch.ones(HEAD_DIM),
    }
    if joint:
        state |= {
            "add_q_proj.weight": w(DIM, DIM),
            "add_k_proj.weight": w(DIM, DIM),
            "add_v_proj.weight": w(DIM, DIM),
            "norm_added_q.weight": torch.ones(HEAD_DIM),
            "norm_added_k.weight": torch.ones(HEAD_DIM),
            "to_out.0.weight": w(DIM, DIM),
            "to_add_out.weight": w(DIM, DIM),
        }

    seq1 = torch.randn((1, seq1_len, DIM))
    seq2 = torch.randn((1, seq2_len, DIM)) if joint else None
    cos, sin = _rope(seq1_len + seq2_len)
    rope1 = (cos[:seq1_len], sin[:seq1_len])
    rope2 = (cos[seq1_len:], sin[seq1_len:])
    torch_out1, torch_out2 = _torch_flux2_attention(state, seq1, seq2, rope1, rope2 if joint else None)

    results = {}
    for variant, (precision, kv_dtype) in VARIANTS.items():
        with sdpa_variant(precision) as precision:
            ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
            if joint:
                # As in transformer_block_opt.TransformerBlock.
                kwargs = dict(added_kv_proj_dim=DIM, context_pre_only=False)
            else:
                # As in transformer_flux2.Flux2SingleTransformerBlock.
                kwargs = dict(added_kv_proj_dim=0, pre_only=True, use_spatial_weights_for_prompt=True)
            tt_model = flux2_attention.Attention(
                query_dim=DIM,
                head_dim=HEAD_DIM,
                heads=HEADS,
                out_dim=DIM,
                proj_bias=False,
                eps=1e-6,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=None,
                per_head_norm=True,
                sdpa_precision=precision,
                sdpa_kv_dtype=kv_dtype,
                **kwargs,
            )
            tt_model.load_torch_state_dict({k: v.clone() for k, v in state.items()})

            tt_seq1 = tensor.from_torch(seq1, device=mesh_device, mesh_axes=[None, sp_axis, None])
            tt_rope1 = tuple(
                tensor.from_torch(t[None, None], device=mesh_device, mesh_axes=[None, None, sp_axis, None])
                for t in rope1
            )
            tt_seq2 = tt_rope2 = None
            if joint:
                tt_seq2 = tensor.from_torch(seq2, device=mesh_device)
                tt_rope2 = tuple(tensor.from_torch(t[None, None], device=mesh_device) for t in rope2)

            out1, out2 = tt_model.forward(
                sequence_1=tt_seq1,
                sequence_1_length=seq1_len,
                sequence_2=tt_seq2,
                sequence_2_length=seq2_len,
                sequence_1_rope=tt_rope1,
                sequence_2_rope=tt_rope2,
            )
            outputs = [
                (
                    "seq1",
                    tensor.to_torch(out1, mesh_axes=[..., sp_axis, None]).reshape(-1, DIM)[:seq1_len].float(),
                    torch_out1.reshape(-1, DIM),
                )
            ]
            if joint:
                outputs.append(
                    (
                        "seq2",
                        tensor.to_torch(out2, mesh_axes=[..., None, None]).reshape(-1, DIM).float(),
                        torch_out2.reshape(-1, DIM),
                    )
                )
            results[variant] = outputs
            del tt_model, ccl_manager

    _gate(results, record_property, f"flux2.{tuple(mesh_device.shape)}.s{seq1_len}.p{seq2_len}")
