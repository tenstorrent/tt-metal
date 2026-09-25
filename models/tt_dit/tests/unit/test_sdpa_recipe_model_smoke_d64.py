# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests: opt-in SDPA recipes on the D64 joint attentions of SD3.5 (SD35JointAttention)
and Motif (blocks/attention.py with context head scaling).

Random weights, small shapes (4 heads, D64). Each case runs legacy (the
module's legacy SDPA config, via tests/unit/sdpa_legacy.py), the default recipe (sdpa_precision=None),
FAST,
ACCURATE and LOW_PRECISION (bfp8 KV) on the same torch weights and inputs, and compares each output
against the torch reference of the module (and against the legacy tt output).

Cases: 1x1 (joint SDPA) and 1x2 (ring joint SDPA, sp=2), with the non-tile-aligned SD3.5/Motif
prompt length 333; Motif also with an unaligned (padded) spatial length on 1x2.
"""

from __future__ import annotations

import diffusers.models.attention_processor
import pytest
import torch

import ttnn

from ...blocks.attention import Attention
from ...models.transformers.attention_sd35 import SD35JointAttention
from ...models.transformers.transformer_motif import MotifTransformer, convert_motif_attention_state
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...reference.motif.modeling_dit import JointAttn as MotifAttentionReference
from ...utils import tensor
from ...utils.tensor import bf16_tensor
from .sdpa_legacy import LEGACY, sdpa_variant

HEADS = 4
HEAD_DIM = 64
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


def _gate(results: dict, record_property, prefix: str) -> None:
    """results: variant -> list of (name, tt_output, torch_reference); legacy must be present."""
    failures = []
    legacy = results["legacy"]
    legacy_l2 = {name: _l2(tt, ref) for name, tt, ref in legacy}
    for name, value in legacy_l2.items():
        record_property(f"{prefix}.legacy.{name}.l2_vs_torch", round(value, 4))
        print(f"{prefix} {'legacy':14s} {name:8s} vs_torch={value:.4f}%")
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
            # Absolute bound vs torch; if the legacy tt output itself exceeds it (bf16 weights, HiFi2
            # legacy SDPA), the recipe must instead be no worse than legacy vs torch (+ margin).
            absolute_ok = vs_torch <= max(bound, base)
            if not absolute_ok or vs_torch > base + MARGIN:
                failures.append(
                    f"{variant}/{name}: l2 vs torch {vs_torch:.3f}% (legacy {base:.3f}%), "
                    f"vs legacy {vs_legacy:.3f}%, bound {bound}%"
                )
            print(f"{prefix} {variant:14s} {name:8s} vs_torch={vs_torch:.4f}% legacy={base:.4f}% vs_legacy={vs_legacy:.4f}%")
    assert not failures, "; ".join(failures)


# ---------------------------------------------------------------------------------------------
# SD3.5 joint attention (models/transformers/attention_sd35.py)
# ---------------------------------------------------------------------------------------------


@MESHES
@pytest.mark.parametrize(
    ("spatial_seq_len", "prompt_seq_len"),
    [
        pytest.param(1024, 333, id="s1024_p333"),
        pytest.param(1024, 128, id="s1024_p128"),
    ],
)
def test_sd35_attention_sdpa_recipes(
    mesh_device, sp_axis, tp_axis, spatial_seq_len, prompt_seq_len, record_property
) -> None:
    torch.manual_seed(0)
    _, parallel_config = _parallel(mesh_device, sp_axis, tp_axis)

    # As in diffusers' SD3 JointTransformerBlock (not the last block: prompt output is produced).
    torch_model = diffusers.models.attention_processor.Attention(
        query_dim=DIM,
        cross_attention_dim=None,
        added_kv_proj_dim=DIM,
        dim_head=HEAD_DIM,
        heads=HEADS,
        out_dim=DIM,
        context_pre_only=False,
        bias=True,
        processor=diffusers.models.attention_processor.JointAttnProcessor2_0(),
        qk_norm="rms_norm",
        eps=1e-6,
    )
    torch_model.eval()
    state = torch_model.state_dict()

    spatial = torch.randn((1, spatial_seq_len, DIM))
    prompt = torch.randn((1, prompt_seq_len, DIM))
    with torch.no_grad():
        torch_spatial, torch_prompt = torch_model(hidden_states=spatial, encoder_hidden_states=prompt)

    results = {}
    for variant, (precision, kv_dtype) in VARIANTS.items():
        with sdpa_variant(precision) as precision:
            ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
            tt_model = SD35JointAttention(
                query_dim=DIM,
                head_dim=HEAD_DIM,
                heads=HEADS,
                out_dim=DIM,
                bias=True,
                out_bias=True,
                context_pre_only=False,
                eps=1e-6,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=None,
                sdpa_precision=precision,
                sdpa_kv_dtype=kv_dtype,
            )
            tt_model.load_torch_state_dict({k: v.clone() for k, v in state.items()})

            tt_spatial = bf16_tensor(spatial.unsqueeze(0), device=mesh_device, mesh_axis=sp_axis, shard_dim=2)
            tt_prompt = bf16_tensor(prompt.unsqueeze(0), device=mesh_device)
            tt_spatial_out, tt_prompt_out = tt_model(tt_spatial, tt_prompt, N=spatial_seq_len)

            results[variant] = [
                (
                    "spatial",
                    tensor.to_torch(tt_spatial_out, mesh_axes=[..., sp_axis, tp_axis]).reshape(-1, DIM).float(),
                    torch_spatial.reshape(-1, DIM),
                ),
                (
                    "prompt",
                    tensor.to_torch(tt_prompt_out, mesh_axes=[..., None, tp_axis]).reshape(-1, DIM).float(),
                    torch_prompt.reshape(-1, DIM),
                ),
            ]
            del tt_model, ccl_manager

    _gate(results, record_property, f"sd35.{tuple(mesh_device.shape)}.s{spatial_seq_len}.p{prompt_seq_len}")


# ---------------------------------------------------------------------------------------------
# Motif joint attention (blocks/attention.py, context head scaling)
# ---------------------------------------------------------------------------------------------


@MESHES
@pytest.mark.parametrize(
    ("spatial_seq_len", "prompt_seq_len"),
    [
        pytest.param(1024, 333, id="s1024_p333"),
        pytest.param(1000, 333, id="s1000_p333"),
    ],
)
def test_motif_attention_sdpa_recipes(
    mesh_device, sp_axis, tp_axis, spatial_seq_len, prompt_seq_len, record_property
) -> None:
    torch.manual_seed(0)
    sp_factor, parallel_config = _parallel(mesh_device, sp_axis, tp_axis)
    if sp_factor == 1 and spatial_seq_len % 32:
        pytest.skip("Motif pads the spatial sequence only with sequence parallelism")
    k_chunk_size = MotifTransformer.get_k_chunk_size(sp_factor)

    class ReferenceAttnConfig:
        hidden_dim = DIM
        num_attention_heads = HEADS
        attn_mode = "flash"

    torch_model = MotifAttentionReference(ReferenceAttnConfig())
    with torch.no_grad():
        torch_model.q_scale.uniform_(0.5, 1.5)  # exercise the context head scaling
    torch_model.eval()

    state = dict(torch_model.state_dict())
    convert_motif_attention_state(
        state,
        x_weight=torch.eye(DIM),
        x_bias=torch.zeros([DIM]),
        c_weight=torch.eye(DIM),
        c_bias=torch.zeros([DIM]),
        is_last_block=False,
    )

    spatial = torch.randn((1, spatial_seq_len, DIM))
    prompt = torch.randn((1, prompt_seq_len, DIM))
    with torch.no_grad():
        torch_spatial, torch_prompt = torch_model.forward(spatial, prompt)

    results = {}
    for variant, (precision, kv_dtype) in VARIANTS.items():
        with sdpa_variant(precision) as precision:
            ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
            # As in MotifTransformer's TransformerBlock (tuned chunks: Q128, K1024 // sp).
            tt_model = Attention(
                query_dim=DIM,
                head_dim=HEAD_DIM,
                heads=HEADS,
                out_dim=DIM,
                added_kv_proj_dim=DIM,
                context_pre_only=False,
                context_head_scaling=True,
                eps=1e-6,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=None,
                k_chunk_size=k_chunk_size,
                q_chunk_size=MotifTransformer.Q_CHUNK_SIZE,
                sdpa_precision=precision,
                sdpa_kv_dtype=kv_dtype,
            )
            tt_model.load_torch_state_dict({k: v.clone() for k, v in state.items()})

            spatial_padded = Attention.pad_spatial_sequence(spatial, sp_factor=sp_factor, k_chunk_size=k_chunk_size)
            tt_spatial = bf16_tensor(spatial_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=-2)
            tt_prompt = bf16_tensor(prompt, device=mesh_device)
            tt_spatial_out, tt_prompt_out = tt_model.forward(
                spatial=tt_spatial, prompt=tt_prompt, spatial_sequence_length=spatial_seq_len
            )

            results[variant] = [
                (
                    "spatial",
                    tensor.to_torch(tt_spatial_out, mesh_axes=[None, sp_axis, tp_axis])[:, :spatial_seq_len]
                    .reshape(-1, DIM)
                    .float(),
                    torch_spatial.reshape(-1, DIM),
                ),
                (
                    "prompt",
                    tensor.to_torch(tt_prompt_out, mesh_axes=[None, None, tp_axis]).reshape(-1, DIM).float(),
                    torch_prompt.reshape(-1, DIM),
                ),
            ]
            del tt_model, ccl_manager

    _gate(results, record_property, f"motif.{tuple(mesh_device.shape)}.s{spatial_seq_len}.p{prompt_seq_len}")
