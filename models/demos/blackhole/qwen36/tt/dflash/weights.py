# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight loading and TP fracturing for the DFlash drafter.

Reads the drafter's 58 tensors and places them on the mesh. Three things here are easy to
get wrong and expensive to debug later, so each is stated explicitly:

**1. No zero-centered "+1" fold.** qwen36's own norms are zero-centered RMSNorm and its
loaders pre-add 1.0 to every gain (``tt/attention/weights.py:14``). The drafter is
``model_type: qwen3`` with plain ``Qwen3RMSNorm``, so its gains are used as-is. Folding
would be a silent ~1.0 offset on every norm.

**2. bfloat16 everywhere.** The target ships bfp8/bfp4 weights for speed, but this is a
correctness milestone: quantization error would be indistinguishable from a port bug in the
PCC numbers. Narrowing dtypes is a later, separately-measured step.

**3. The ``fc`` input permutation.** See :func:`fc_input_permutation` -- this is the one
piece of Milestone 1 whose correctness is only observable in Milestone 2.

TP=8 layout. Every projection fractures and the residual stream stays replicated, so the
per-layer norms need no distributed statistics and the only collectives are the two
all-reduces behind ``o_proj`` and ``down_proj``:

===================  ==============  ==================================================
weight               parallelism     per-device shard
===================  ==============  ==================================================
``q_proj``           column          512 out = 4 heads x 128
``k_proj``/``v_proj``column          128 out = exactly 1 KV head (no replication needed)
``o_proj``           row             4096/8 in, then all-reduce
``gate``/``up``      column          2176 out
``down``             row             17408/8 in, then all-reduce
``fc``               row             3200 in (permuted, see below), then all-reduce
norms                replicated      full [5120] or [128]
===================  ==============  ==================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig


def fc_input_permutation(tp: int, hidden_size: int, n_taps: int) -> torch.Tensor:
    """Row order that makes ``fc``'s 25600-wide input shard match the on-device tap layout.

    ``fc`` consumes ``concat(tap_0, ..., tap_4)``, each ``hidden_size`` wide. But the target
    carries its residual stream **TP-sharded on the last dim** (``tt/model.py:1060-1062``
    reshapes the decode embedding to ``[1, 1, B, dim/tp]``, and prefill RMSNorm hands modules
    a K-sharded ``[., S, dim/tp]``). So a tap captured on device gives chip ``d`` the slice
    ``tap_i[d*640 : (d+1)*640]`` -- for *every* tap -- not a contiguous 3200-wide block of the
    concatenation.

    Sharding the natural concatenation contiguously would therefore hand chip 0 rows
    ``tap_0[0:3200]``, which is not what the device has. This permutation regroups the input
    axis as ``for d: for i: tap_i[d*640:(d+1)*640]`` so that a plain contiguous shard lands
    each chip's rows on the chip that owns them.

    Milestone 1 feeds ``target_hidden`` from a host fixture, where either ordering "works" as
    long as weight and activation agree -- which is exactly why this must be right now:
    getting it wrong passes every drafter test and silently produces garbage the moment
    Milestone 2 wires up real taps.

    Returns an index tensor of length ``n_taps * hidden_size`` to apply to the input axis.
    """
    assert hidden_size % tp == 0, f"hidden_size {hidden_size} not divisible by tp {tp}"
    per = hidden_size // tp
    idx = [i * hidden_size + d * per + j for d in range(tp) for i in range(n_taps) for j in range(per)]
    return torch.tensor(idx, dtype=torch.long)


def permute_fc_input_activation(x: torch.Tensor, tp: int, hidden_size: int, n_taps: int) -> torch.Tensor:
    """Apply :func:`fc_input_permutation` to a ``[..., n_taps * hidden_size]`` activation.

    Only needed when feeding ``target_hidden`` from a host fixture. On device the tap layout
    already produces this order, so nothing is permuted at inference time.
    """
    return x.index_select(-1, fc_input_permutation(tp, hidden_size, n_taps))


@dataclass(frozen=True)
class DFlashLayerWeights:
    """One decoder layer's weights, already fractured across the mesh."""

    q_proj: ttnn.Tensor
    k_proj: ttnn.Tensor
    v_proj: ttnn.Tensor
    o_proj: ttnn.Tensor
    q_norm: ttnn.Tensor  # [head_dim], replicated -- head_dim is never sharded
    k_norm: ttnn.Tensor
    input_layernorm: ttnn.Tensor  # [hidden], replicated
    post_attention_layernorm: ttnn.Tensor
    gate_proj: ttnn.Tensor
    up_proj: ttnn.Tensor
    down_proj: ttnn.Tensor


@dataclass(frozen=True)
class DFlashWeights:
    """The whole drafter: context encoder, per-layer stack, final norm."""

    fc: ttnn.Tensor  # [25600/tp, 5120] per device, input-permuted
    hidden_norm: ttnn.Tensor
    norm: ttnn.Tensor
    layers: tuple[DFlashLayerWeights, ...]


def checkpoint_path(path: str | None = None) -> str:
    """Local dir or Hub snapshot for the drafter checkpoint."""
    path = path or os.environ.get("DFLASH_HF_MODEL") or "z-lab/Qwen3.6-27B-DFlash"
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(path, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")


def read_state_dict(path: str | None = None, keys: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
    """Read the drafter's tensors, or just ``keys``.

    Reads lazily through ``safe_open`` so a single-module unit test pulls only the few
    tensors it grades, rather than materialising all 3.46 GB.
    """
    from safetensors.torch import safe_open

    ckpt = os.path.join(checkpoint_path(path), "model.safetensors")
    with safe_open(ckpt, framework="pt") as f:
        wanted = list(keys) if keys is not None else list(f.keys())
        missing = [k for k in wanted if k not in f.keys()]
        assert not missing, f"checkpoint {ckpt} is missing {missing}"
        return {k: f.get_tensor(k) for k in wanted}


def layer_keys(layer_idx: int) -> list[str]:
    """Checkpoint keys for one decoder layer."""
    p = f"layers.{layer_idx}."
    return [
        f"{p}self_attn.q_proj.weight",
        f"{p}self_attn.k_proj.weight",
        f"{p}self_attn.v_proj.weight",
        f"{p}self_attn.o_proj.weight",
        f"{p}self_attn.q_norm.weight",
        f"{p}self_attn.k_norm.weight",
        f"{p}input_layernorm.weight",
        f"{p}post_attention_layernorm.weight",
        f"{p}mlp.gate_proj.weight",
        f"{p}mlp.up_proj.weight",
        f"{p}mlp.down_proj.weight",
    ]


# Column-parallel: shard the OUT dim. shard_w transposes [out,in] -> [in,out], so out is -1.
_COLUMN, _ROW = -1, 0


def _shard(w: torch.Tensor, mesh, dim: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return tpc.shard_w(w, mesh, dim=dim, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=None, dtype=dtype)


def _replicate(w: torch.Tensor, mesh, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    # NOTE: no +1 fold. The drafter uses plain RMSNorm; see the module docstring.
    return tpc.replicate(w, mesh, cache_path=None, dtype=dtype)


def load_layer_weights(
    mesh_device,
    cfg: DFlashDrafterConfig,
    layer_idx: int,
    state_dict: dict[str, torch.Tensor] | None = None,
    path: str | None = None,
    dtype=ttnn.bfloat16,
) -> DFlashLayerWeights:
    """Load and fracture one decoder layer. Reads only that layer's tensors if needed."""
    sd = state_dict if state_dict is not None else read_state_dict(path, layer_keys(layer_idx))
    p = f"layers.{layer_idx}."

    tp = _tp(mesh_device)
    _assert_shardable(cfg, tp)

    return DFlashLayerWeights(
        q_proj=_shard(sd[f"{p}self_attn.q_proj.weight"], mesh_device, _COLUMN, dtype),
        k_proj=_shard(sd[f"{p}self_attn.k_proj.weight"], mesh_device, _COLUMN, dtype),
        v_proj=_shard(sd[f"{p}self_attn.v_proj.weight"], mesh_device, _COLUMN, dtype),
        o_proj=_shard(sd[f"{p}self_attn.o_proj.weight"], mesh_device, _ROW, dtype),
        q_norm=_replicate(sd[f"{p}self_attn.q_norm.weight"], mesh_device, dtype),
        k_norm=_replicate(sd[f"{p}self_attn.k_norm.weight"], mesh_device, dtype),
        input_layernorm=_replicate(sd[f"{p}input_layernorm.weight"], mesh_device, dtype),
        post_attention_layernorm=_replicate(sd[f"{p}post_attention_layernorm.weight"], mesh_device, dtype),
        gate_proj=_shard(sd[f"{p}mlp.gate_proj.weight"], mesh_device, _COLUMN, dtype),
        up_proj=_shard(sd[f"{p}mlp.up_proj.weight"], mesh_device, _COLUMN, dtype),
        down_proj=_shard(sd[f"{p}mlp.down_proj.weight"], mesh_device, _ROW, dtype),
    )


def load_fc(
    mesh_device,
    cfg: DFlashDrafterConfig,
    state_dict: dict[str, torch.Tensor] | None = None,
    path: str | None = None,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    """Load ``fc`` row-parallel, with its input axis permuted to the device tap layout."""
    sd = state_dict if state_dict is not None else read_state_dict(path, ["fc.weight"])
    w = sd["fc.weight"]  # [out=5120, in=25600]
    assert w.shape == (cfg.hidden_size, cfg.target_feature_size), w.shape

    tp = _tp(mesh_device)
    perm = fc_input_permutation(tp, cfg.hidden_size, len(cfg.target_layer_ids))
    w = w.index_select(1, perm)  # reorder the INPUT axis
    # Row-parallel: shard the input axis. shard_w transposes to [in, out], so that is dim 0.
    return _shard(w, mesh_device, _ROW, dtype)


def load_weights(
    mesh_device,
    cfg: DFlashDrafterConfig,
    path: str | None = None,
    dtype=ttnn.bfloat16,
) -> DFlashWeights:
    """Load and fracture the whole drafter."""
    sd = read_state_dict(path)
    return DFlashWeights(
        fc=load_fc(mesh_device, cfg, sd, dtype=dtype),
        hidden_norm=_replicate(sd["hidden_norm.weight"], mesh_device, dtype),
        norm=_replicate(sd["norm.weight"], mesh_device, dtype),
        layers=tuple(load_layer_weights(mesh_device, cfg, i, sd, dtype=dtype) for i in range(cfg.num_hidden_layers)),
    )


def _tp(mesh_device) -> int:
    return mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1


def _assert_shardable(cfg: DFlashDrafterConfig, tp: int) -> None:
    """Fail with the offending dim rather than a downstream shape mismatch."""
    for name, dim in (
        ("q_dim", cfg.q_dim),
        ("kv_dim", cfg.kv_dim),
        ("intermediate_size", cfg.intermediate_size),
        ("num_key_value_heads", cfg.num_key_value_heads),
    ):
        assert dim % tp == 0, f"{name}={dim} is not divisible by tp={tp}"
