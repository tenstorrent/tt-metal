# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared test scaffolding: mesh parametrization, PCC thresholds from the spec, tensor compare."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama_3_1_8b.conftest import FABRIC_CONFIG
from models.demos.llama_3_1_8b.reference.config import LlamaConfig
from models.demos.llama_3_1_8b.tt.ccl import CCLManager
from models.demos.llama_3_1_8b.tt.mesh import SPEC_MESH_SHAPE, SPEC_TP, MeshConfig
from models.demos.llama_3_1_8b.utils.general import default_num_links

SPEC_JSON = Path(__file__).resolve().parents[1] / "configs" / "prefill_spec.json"


def load_spec() -> dict:
    """The prefill spec. ``PREFILL_SPEC`` (the snapshot the verifier passes) wins over the vendored
    copy, so the graded run reads the same file the pipeline prepared."""
    path = os.getenv("PREFILL_SPEC") or SPEC_JSON
    with open(path) as f:
        return json.load(f)


def pcc_bounds() -> tuple:
    """``(pcc_lower_bound, pcc_target)`` — the assert and the aim. Both come from the spec; neither
    is ever hardcoded in a test."""
    acc = load_spec()["acceptance"]
    return float(acc["pcc_lower_bound"]), float(acc["pcc_target"])


PCC_LOWER, PCC_TARGET = pcc_bounds()


def assert_pcc(name: str, got: float, *, lower=None, target=None):
    """Assert at the spec's lower bound and report anything between the bound and the target.

    Recipe §4: below ``pcc_lower_bound`` the component is not accepted and the stage is red; between
    the two it passes but the measured value and reason go in the README's PCC table.
    """
    from loguru import logger

    lower = PCC_LOWER if lower is None else lower
    target = PCC_TARGET if target is None else target
    assert got == got, f"{name}: PCC is NaN"
    assert got >= lower, f"{name}: PCC {got:.6f} < pcc_lower_bound {lower}"
    if got < target:
        logger.warning(f"{name}: PCC {got:.6f} is below pcc_target {target} — record it in README.md")
    else:
        logger.info(f"{name}: PCC {got:.6f}")
    return got


def pcc(golden: torch.Tensor, got: torch.Tensor) -> float:
    return float(comp_pcc(golden, got, 0.0)[1])


def galaxy_mesh(extra_params=None):
    """Parametrize a test onto the spec's (8, 4) mesh with the fabric the descriptor implies.

    Skips (rather than fails) on a machine that cannot hold 32 chips, so the host-only parts of the
    suite still run on a laptop.
    """
    params = {"fabric_config": FABRIC_CONFIG, "trace_region_size": 0}
    params.update(extra_params or {})
    return pytest.mark.parametrize(
        "mesh_device, device_params", [pytest.param(SPEC_MESH_SHAPE, params, id="8x4")], indirect=True
    )


def single_device(extra_params=None):
    """Parametrize onto a single chip (sp=1, tp=1): the non-ring path for op-level math checks."""
    params = {"fabric_config": None, "trace_region_size": 0}
    params.update(extra_params or {})
    return pytest.mark.parametrize(
        "mesh_device, device_params", [pytest.param((1, 1), params, id="1x1")], indirect=True
    )


def spec_mesh_config(mesh_device) -> MeshConfig:
    return MeshConfig(tuple(mesh_device.shape), tp=min(SPEC_TP, mesh_device.shape[1]))


def make_ccl(mesh_device, topology=None) -> CCLManager:
    from models.demos.llama_3_1_8b.conftest import CCL_TOPOLOGY

    return CCLManager(mesh_device, num_links=default_num_links(mesh_device), topology=topology or CCL_TOPOLOGY)


def cfg_full() -> LlamaConfig:
    return LlamaConfig.from_json()


def to_torch_replicated(tensor, mesh_device):
    """Read back a tensor that is REPLICATED across the mesh: take device 0's copy."""
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()


def to_torch_sp_concat(tensor, mesh_device, mesh_config, *, seq_dim=2):
    """Read back an SP-sharded / TP-replicated activation: concatenate the SP rows on ``seq_dim``.

    ``ConcatMesh2dToTensor`` needs a dim per mesh axis, so the TP axis is concatenated on dim 0 and
    only the first replica is kept — taking the mean would hide a TP column that disagreed.
    """
    dims = [None, None]
    dims[mesh_config.sp_axis] = seq_dim
    dims[mesh_config.tp_axis] = 0
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=mesh_device.shape)
    out = ttnn.to_torch(tensor, mesh_composer=composer).float()
    return out[:1]


def to_torch_tp_concat(tensor, mesh_device, mesh_config, *, shard_dim=-1, seq_dim=2, sp_sharded=True):
    """Read back a TP-sharded tensor, concatenating TP on ``shard_dim`` and SP on ``seq_dim``."""
    dims = [None, None]
    dims[mesh_config.tp_axis] = shard_dim
    dims[mesh_config.sp_axis] = seq_dim if sp_sharded else 0
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=mesh_device.shape)
    out = ttnn.to_torch(tensor, mesh_composer=composer).float()
    return out if sp_sharded else out[:1]


def sp_shard_activation(torch_tensor, mesh_device, mesh_config, dtype=ttnn.bfloat16, seq_dim=2):
    """Push a ``[1, 1, s, hidden]`` host activation onto the mesh: seq across SP, replicated on TP."""
    return ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.sequence_parallel(mesh_device, seq_dim=seq_dim),
    )


# ---------------------------------------------------------------------------------------------
# Random weights, built ONCE and handed to both sides
# ---------------------------------------------------------------------------------------------
# Every module PCC test up to P1 runs on random weights that are identical on the device and in the
# torch reference (recipe §4). These builders are the single source of those tensors, so the two
# sides cannot be handed different ones — the failure mode that makes a PCC test meaningless.


def random_layer_weights(cfg, seed: int = 0, dtype=torch.float16) -> dict:
    """One decoder layer's weights, keyed exactly as the checkpoint keys them under a layer prefix."""
    g = torch.Generator().manual_seed(seed)
    d, h, i = cfg.head_dim, cfg.hidden_size, cfg.intermediate_size

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(dtype)

    return {
        "input_layernorm.weight": rnd(h),
        "post_attention_layernorm.weight": rnd(h),
        "self_attn.q_proj.weight": rnd(cfg.num_attention_heads * d, h),
        "self_attn.k_proj.weight": rnd(cfg.num_key_value_heads * d, h),
        "self_attn.v_proj.weight": rnd(cfg.num_key_value_heads * d, h),
        "self_attn.o_proj.weight": rnd(h, cfg.num_attention_heads * d),
        "mlp.gate_proj.weight": rnd(i, h),
        "mlp.up_proj.weight": rnd(i, h),
        "mlp.down_proj.weight": rnd(h, i),
    }


def reference_layer(cfg, weights: dict):
    """A torch reference DecoderLayer loaded with exactly ``weights``."""
    from models.demos.llama_3_1_8b.reference.model import REF_DTYPE, DecoderLayer

    layer = DecoderLayer(cfg)
    layer.load_state_dict({k: v.to(REF_DTYPE) for k, v in weights.items()}, strict=True)
    layer.eval()
    return layer


def reference_attention(cfg, weights: dict):
    """A torch reference Attention loaded with the ``self_attn.*`` subset of ``weights``."""
    from models.demos.llama_3_1_8b.reference.model import REF_DTYPE, Attention

    attn = Attention(cfg)
    attn.load_state_dict(
        {k[len("self_attn.") :]: v.to(REF_DTYPE) for k, v in weights.items() if k.startswith("self_attn.")},
        strict=True,
    )
    attn.eval()
    return attn


def reference_mlp(cfg, weights: dict):
    from models.demos.llama_3_1_8b.reference.model import REF_DTYPE, MLP

    mlp = MLP(cfg)
    mlp.load_state_dict(
        {k[len("mlp.") :]: v.to(REF_DTYPE) for k, v in weights.items() if k.startswith("mlp.")}, strict=True
    )
    mlp.eval()
    return mlp


def assert_tp_replicas_agree(tensor, mesh_device, mesh_config, name="tensor", tol=0.05):
    """Every TP column must hold the same values for a replicated activation.

    Taking replica 0 and moving on would hide a column that silently diverged (a half-completed
    all-reduce looks exactly like this), so the read-back helpers are paired with this check
    wherever the layout claims to be replicated.
    """
    dims = [None, None]
    dims[mesh_config.tp_axis] = 0
    dims[mesh_config.sp_axis] = 2
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=mesh_device.shape)
    stacked = ttnn.to_torch(tensor, mesh_composer=composer).float()
    spread = (stacked - stacked[:1]).abs().max().item()
    assert spread < tol, f"{name}: TP replicas disagree by {spread:.4f}"
    return stacked[:1]
