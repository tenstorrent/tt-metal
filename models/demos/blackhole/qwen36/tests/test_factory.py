# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for the Qwen3.5/Qwen3.6 demo test suite.

Centralizes what used to be copy-pasted across the per-test files:

* ``model_path()``                 — resolve the HF model id/dir from ``HF_MODEL``
* ``load_attn_layer`` / ``load_gdn_layer`` / ``load_mlp_layer``
                                     — dequantize one layer's weights from the
                                       FP8 (or bf16) safetensors checkpoint
* ``compute_pcc`` / ``compare_tensors`` — single PCC implementation
* ``get_pcc_threshold(request)``   — per-test threshold from ``pcc_thresholds.json``
* ``parametrize_mesh_tp()``        — the env-driven (1,4)/(1,1) mesh + FABRIC_1D idiom
* ``tp_composer`` / ``replicate_to_device`` — shared TP tensor helpers

Heavy imports (ttnn weight loaders, ``Qwen36ModelArgs``) are kept lazy / local so
pure-CPU tests (``test_weight_mapping``, ``test_substate``) stay fast at collection.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch

import ttnn

# TP tests default to the 27B variant; single-device unit tests setdefault 9B.
_DEFAULT_HF_MODEL = "Qwen/Qwen3.6-27B"
_PCC_THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "pcc_thresholds.json")

# Prefill length buckets gated by --max-prefill (see conftest). Lengths above the
# cap are auto-skipped so the routine loop stays fast.
PREFILL_BUCKETS = [128, 1024, 2048]


def model_path():
    """Resolve the HF model id/dir. Replaces the per-file ``_mp()``/``_model_path()``."""
    return os.path.expanduser(os.environ.get("HF_MODEL", _DEFAULT_HF_MODEL))


# --------------------------------------------------------------------------- #
# Checkpoint weight loaders (FP8-block dequant)
# --------------------------------------------------------------------------- #
def load_layer_weights(ckpt_dir, layer_idx, names, search_prefix, out_prefix=""):
    """Dequantize selected ``layers.<i>.<search_prefix><name>`` weights into a dict.

    ``names`` entries are leaf names, or ``(name, has_weight_suffix)`` tuples for
    params stored without a ``.weight`` suffix (GDN's ``A_log``/``dt_bias``).
    Output keys are ``<out_prefix><name>[.weight]`` — matching exactly what the
    original ``_load_*`` helpers produced.
    """
    from safetensors import safe_open

    from ..tt.tp_common import dequant_fp8_block

    ckpt_dir = Path(ckpt_dir)
    wm = json.load(open(ckpt_dir / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for entry in names:
        name, has_w = entry if isinstance(entry, tuple) else (entry, True)
        leaf = f"{search_prefix}{name}" + (".weight" if has_w else "")
        base = next(k for k in wm if k.endswith(f"layers.{layer_idx}.{leaf}"))
        with safe_open(str(ckpt_dir / wm[base]), framework="pt") as sf:
            w = sf.get_tensor(base)
            sk = base + "_scale_inv"
            if wm.get(sk):
                with safe_open(str(ckpt_dir / wm[sk]), framework="pt") as sf2:
                    w = dequant_fp8_block(w, sf2.get_tensor(sk))
            else:
                w = w.to(torch.bfloat16)
        out[f"{out_prefix}{name}" + (".weight" if has_w else "")] = w
    return out


def load_attn_layer(ckpt_dir, layer_idx):
    """Full-attention layer weights — keys ``q_proj.weight`` … ``k_norm.weight``."""
    return load_layer_weights(
        ckpt_dir,
        layer_idx,
        ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"],
        search_prefix="self_attn.",
    )


def load_gdn_layer(ckpt_dir, layer_idx):
    """Gated-DeltaNet layer weights — keys ``linear_attn.in_proj_qkv.weight`` …"""
    return load_layer_weights(
        ckpt_dir,
        layer_idx,
        [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
            "conv1d",
            ("A_log", False),
            ("dt_bias", False),
            "norm",
        ],
        search_prefix="linear_attn.",
        out_prefix="linear_attn.",
    )


def load_mlp_layer(ckpt_dir, layer_idx):
    """SwiGLU MLP layer weights — keys ``gate_proj.weight``/``up_proj.weight``/``down_proj.weight``."""
    return load_layer_weights(
        ckpt_dir,
        layer_idx,
        ["gate_proj", "up_proj", "down_proj"],
        search_prefix="mlp.",
    )


# --------------------------------------------------------------------------- #
# PCC helpers
# --------------------------------------------------------------------------- #
def compute_pcc(a, b):
    """Pearson correlation between two tensors (single canonical implementation)."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_c = a_flat - a_flat.mean()
    b_c = b_flat - b_flat.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-8)).item()


def compare_tensors(tt_tensor, torch_tensor, pcc_threshold=0.99):
    """Compare a TT/torch tensor against a torch reference; logs and returns (passing, pcc)."""
    from loguru import logger

    from models.common.utility_functions import comp_pcc

    tt = tt_tensor if isinstance(tt_tensor, torch.Tensor) else ttnn.to_torch(tt_tensor)
    passing, pcc = comp_pcc(torch_tensor, tt, pcc_threshold)
    logger.info(f"PCC={pcc} (threshold={pcc_threshold}) [{'PASS' if passing else 'FAIL'}]")
    return passing, pcc


@lru_cache(maxsize=1)
def _load_pcc_thresholds():
    with open(_PCC_THRESHOLDS_PATH) as f:
        return json.load(f)


def get_pcc_threshold(request, default=0.99):
    """Per-test PCC threshold from ``pcc_thresholds.json``.

    The table is a flat ``{test_function_name: threshold}`` map — qwen's test
    names are unique across the 9B (single-device) and 27B (TP) suites, so a
    single function-keyed table is unambiguous and, unlike a model-keyed lookup,
    is robust to ``HF_MODEL`` being a resolved local snapshot path (whose basename
    is an opaque hash, not "Qwen3.5-9B"). Unlisted tests fall back to ``default``.
    """
    table = _load_pcc_thresholds()
    func = getattr(request.node, "originalname", None) or request.node.name.split("[")[0]
    return table.get(func, default)


# --------------------------------------------------------------------------- #
# Mesh / device helpers
# --------------------------------------------------------------------------- #
def _resolve_mesh_shape(max_tp=4):
    return {"P150": (1, 1), "P150x4": (1, 4)}.get(
        os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), max_tp))
    )


def parametrize_mesh_tp(max_tp=4):
    """Parametrize a TP test over the env-selected mesh shape + FABRIC_1D.

    Mirrors the idiom the qwen TP tests used inline: ``MESH_DEVICE=P150`` -> (1,1),
    ``P150x4`` -> (1,4); otherwise (1, min(num_devices, max_tp)). The mesh shape
    gets an explicit ``RxC`` id so node names (and ``pcc_thresholds.json`` mesh
    keys) are readable.
    """
    shape = _resolve_mesh_shape(max_tp)
    # Local import to keep this test helper's module load light (see module docstring).
    from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

    def decorator(fn):
        fn = pytest.mark.parametrize(
            "device_params",
            # l1_small_size required by ttnn.conv1d in the GDN prefill path
            [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": GDN_CONV1D_L1_SMALL_SIZE}],
            indirect=True,
        )(fn)
        fn = pytest.mark.parametrize("mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True)(
            fn
        )
        return fn

    return decorator


def parametrize_batch(batches=(1, 8, 32)):
    """Parametrize a decode test over batch sizes (the ``B`` fixture argument).

    B must be a power of two <= 32 so the ``kv_update_shard_cfg`` core grid in
    ``Qwen35ModelArgs._init_tp_config`` factors cleanly (one user per core, grid
    sized to ``max_batch_size``). Ids are ``B1``/``B8``/``B32`` so node names and
    ``pcc_thresholds.json`` stay readable.
    """
    return pytest.mark.parametrize("B", [pytest.param(b, id=f"B{b}") for b in batches])


def parametrize_mesh_only(max_tp=4):
    """Parametrize over just the env-selected mesh shape (no device_params / fabric).

    For mesh tests that don't run fabric CCL ops (e.g. pure weight-loading checks),
    matching the bare ``@parametrize("mesh_device", ...)`` decorator they used inline.
    """
    shape = _resolve_mesh_shape(max_tp)

    def decorator(fn):
        return pytest.mark.parametrize(
            "mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True
        )(fn)

    return decorator


def tp_composer(mesh_device):
    """ConcatMeshToTensor composer for TP outputs (dim=3 multi-device, dim=0 single)."""
    nd = mesh_device.get_num_devices()
    return ttnn.ConcatMeshToTensor(mesh_device, dim=3 if nd > 1 else 0)


def replicate_to_device(mesh_device, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Replicate a torch tensor to every device in the mesh (TILE/DRAM by default)."""
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def shard_to_device(mesh_device, t, dim=-1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Shard a torch tensor across the mesh on ``dim`` (last dim by default).

    Use this for the PREFILL activation fed to ``forward_prefill``: the fused in-proj
    all-gather-matmul (all_gather_minimal_matmul_async) expects a K-sharded input, since
    the model's prefill RMSNorm skips its post-norm all-gather (layer.py ``_fuse_norm_agmm``)
    and hands attention/GDN a ``[.,S,dim/tp]`` activation. The fused op gathers it back to
    full ``dim`` before the matmul, so a host reference over the full tensor still matches.
    """
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
