# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared test scaffolding for the Llama 3.1 8B prefill bring-up.

Copied from ``minimax_m3/tests/test_factory.py``. The load-bearing part is the fabric choice: a
plain-MESH single Galaxy descriptor has **no wrap-around links**, so ``FABRIC_1D_RING`` cannot be
opened on it — multi-device meshes use ``FABRIC_1D`` and the CCLManager must use
``ttnn.Topology.Linear`` to match. Getting this pair wrong shows up as a hang or a fabric
bring-up error, not a wrong number.

Every test here runs on **random weights, identical on both sides**, and needs no checkpoint and no
network.
"""

import os

import pytest
import torch

import ttnn
from models.demos.llama3_1_8b_d_p.reference.config import LlamaConfig
from models.demos.llama3_1_8b_d_p.tt.ccl import CCLManager
from models.demos.llama3_1_8b_d_p.tt.config import MeshConfig
from models.demos.llama3_1_8b_d_p.utils.general_utils import get_default_num_links

# The production target from the prefill spec: mesh (8, 4), SP=8 on rows, TP=4 on cols.
TARGET_MESH_SHAPE = (8, 4)
TARGET_TP = 4
TARGET_SP = 8


def llama_config() -> LlamaConfig:
    """Full-dims config from the vendored config.json (no network, no checkpoint)."""
    return LlamaConfig.from_json()


def parametrize_mesh_with_fabric(mesh_shapes=None, linear_fabric=True):
    """Paired ``(mesh_device, device_params)`` parametrization.

    Defaults to ``linear_fabric=True``: FABRIC_1D, which is what a plain-MESH Galaxy supports.
    ``(1, 1)`` disables fabric entirely (no inter-chip topology).
    """
    num_devices = ttnn.get_num_devices()
    if mesh_shapes is None:
        mesh_shapes = [(1, 1), (1, 4), TARGET_MESH_SHAPE]
    mesh_shapes = [s for s in mesh_shapes if s[0] * s[1] <= num_devices]

    if not mesh_shapes:
        params = [
            pytest.param(
                (1, 1),
                {"fabric_config": None, "trace_region_size": 100000000},
                id="1x1",
                marks=pytest.mark.skip(reason="no supported mesh shape fits on this system"),
            )
        ]
    else:
        multidev_fabric = ttnn.FabricConfig.FABRIC_1D if linear_fabric else ttnn.FabricConfig.FABRIC_1D_RING
        params = [
            pytest.param(
                shape,
                {
                    "fabric_config": (None if shape == (1, 1) else multidev_fabric),
                    "trace_region_size": 100000000,
                },
                id=f"{shape[0]}x{shape[1]}",
            )
            for shape in mesh_shapes
        ]

    def decorator(func):
        return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)(func)

    return decorator


def make_mesh_config(mesh_device) -> MeshConfig:
    """MeshConfig for the opened mesh: TP spans the whole column axis, SP is the row axis."""
    shape = tuple(mesh_device.shape)
    return MeshConfig(shape, tp=shape[1], tp_axis=1)


def make_ccl(mesh_device) -> CCLManager:
    """CCLManager on the Linear topology that pairs with FABRIC_1D (see the module docstring)."""
    return CCLManager(
        mesh_device,
        num_links=get_default_num_links(mesh_device),
        topology=ttnn.Topology.Linear,
    )


def replicate(mesh_device, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def shard_seq_on_sp(mesh_device, t, mesh_config, dtype=ttnn.bfloat16, seq_dim=-2):
    """Shard the sequence dim across the SP axis, replicate across TP."""
    dims = [None, None]
    dims[mesh_config.sp_axis] = seq_dim
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims)),
    )


def block_cyclic_shard_seq(mesh_device, t, mesh_config, chunk_size, dtype=ttnn.bfloat16, seq_dim=-2):
    """Block-cyclic SP shard of the sequence dim — the layout the KV cache and indexed rope use.

    The chunk is split into ``sp`` contiguous blocks of ``chunk_size // sp`` tokens, and block ``i``
    goes to SP row ``i``. Because prefill pushes one chunk at a time, that is the same as a plain
    contiguous split of THIS chunk; the "cyclic" part appears across chunks, where row ``i`` collects
    block ``i`` of every chunk into its contiguous local cache rows.
    """
    sp = mesh_config.sp
    assert chunk_size % sp == 0, f"chunk_size {chunk_size} must be divisible by sp {sp}"
    return shard_seq_on_sp(mesh_device, t, mesh_config, dtype=dtype, seq_dim=seq_dim)


def concat_sp(mesh_device, tt, mesh_config, seq_dim=-2):
    """Undo an SP shard: concatenate the per-row shards back along the sequence dim."""
    dims = [None, None]
    dims[mesh_config.sp_axis] = seq_dim
    rows, cols = tuple(mesh_device.shape)
    devs = ttnn.get_device_tensors(tt)
    # Device tensors are row-major over the mesh: index = row * cols + col. Take TP column 0.
    per_row = [ttnn.to_torch(devs[r * cols]) for r in range(rows)]
    return torch.cat(per_row, dim=seq_dim)


def concat_tp_heads(mesh_device, tt, head_dim_axis=1):
    """Concatenate the per-TP-column head shards of one SP row back into the global head order."""
    rows, cols = tuple(mesh_device.shape)
    devs = ttnn.get_device_tensors(tt)
    return torch.cat([ttnn.to_torch(devs[c]) for c in range(cols)], dim=head_dim_axis)


def dev0(tt):
    """The first device's shard as a torch tensor (for replicated / TP-invariant outputs)."""
    return ttnn.to_torch(ttnn.get_device_tensors(tt)[0])


def hf_to_meta_qk(state_dict, head_dim):
    """Reverse-permute q_proj / k_proj rows into Meta (interleaved-RoPE) format.

    The device applies RoPE with interleaved cos/sin, which only reproduces HF's half-split rotation
    when the q/k projection ROWS are swizzled to match. Production does this once at checkpoint load
    (``tt/model_config.py``); tests that build weights from the torch reference must do the same, or
    every attention PCC measures the missing permutation instead of the math.
    """
    from models.tt_transformers.tt.load_checkpoints import reverse_permute

    out = {}
    for k, v in state_dict.items():
        if k.endswith("q_proj.weight") or k.endswith("k_proj.weight"):
            n_heads = v.shape[0] // head_dim
            out[k] = reverse_permute(v, n_heads, v.shape[0], v.shape[1])
        else:
            out[k] = v
    return out


def block_cyclic_to_natural(x, sp, chunk_local, seq_dim=2):
    """Un-permute a row-concatenated block-cyclic cache read-back into natural token order.

    The writer kernel places, on chip ``c``, local cache row ``lr`` at global position::

        g(c, lr) = (lr // chunk_local) * (sp * chunk_local) + c * chunk_local + (lr % chunk_local)

    (see ``deepseek_v3_d_p/tt/mla/utils.rotated_chip_positions``). Concatenating the per-chip shards
    in chip order therefore yields an array indexed by ``c * seq_local + lr``, which is natural order
    ONLY when the cache holds a single chunk — then ``lr // chunk_local == 0`` and ``g`` reduces to
    ``c * chunk_local + lr``.

    With several chunks it does not: chip ``c``'s contiguous rows hold block ``c`` of chunk 0, then
    block ``c`` of chunk 1, and so on. Reading it back without this inverse silently compares
    interleaved tokens, which is a PCC near zero rather than an error — and a test that only ever
    writes one chunk cannot see the difference.
    """
    import torch

    seq_local = x.shape[seq_dim] // sp
    assert seq_local % chunk_local == 0, f"seq_local {seq_local} must be a multiple of chunk_local {chunk_local}"
    chunk_global = sp * chunk_local
    positions = torch.empty(sp * seq_local, dtype=torch.long)
    for c in range(sp):
        for lr in range(seq_local):
            positions[c * seq_local + lr] = (lr // chunk_local) * chunk_global + c * chunk_local + (lr % chunk_local)
    inverse = torch.empty_like(positions)
    inverse[positions] = torch.arange(positions.numel())
    return x.index_select(seq_dim, inverse)
