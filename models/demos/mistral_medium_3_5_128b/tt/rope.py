# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""YaRN RoPE for the device path.

Values come from the torch reference (``reference/model.py::rope_cos_sin``, pinned bit-exact to HF and
to the golden trace's layer-0 K): fp32 angles, attention_factor folded into cos/sin, cast to bf16 last.
The device rotates with ``rotary_embedding_indexed``, which uses the Meta interleaved pair layout, so
q_proj / k_proj rows are permuted per head HF-half-split -> Meta at load (``permute_qk_rows``) and the
K cache holds Meta-layout K (``hf_to_meta_perm`` maps a golden HF K into the same layout).

Whole-cache tables (plumbing from minimax_m3 ``TtPrefillRuntime._build_indexed_rope``): cos/sin for every
cache position, block-cyclic reordered by ``chunk_local`` and SP-sharded, built once; the indexed op
derives each chunk's per-chip rows on device from ``kv_actual_global``.
"""

import torch

import ttnn
from models.common.tensor_utils import get_rot_transformation_mat
from models.common.utils import block_cyclic_reorder

from ..reference.model import rope_cos_sin


def hf_to_meta_perm(head_dim: int) -> torch.Tensor:
    """Column permutation p with ``meta = hf[..., p]``: Meta pairs (2j, 2j+1) are HF (j, j + d/2)."""
    half = head_dim // 2
    return torch.tensor([half * (m % 2) + m // 2 for m in range(head_dim)], dtype=torch.long)


def permute_qk_rows(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    """HF ``[n_heads * head_dim, in]`` q/k projection -> rows in Meta interleaved order per head."""
    n_heads = weight.shape[0] // head_dim
    assert n_heads * head_dim == weight.shape[0], f"{tuple(weight.shape)} is not whole heads of {head_dim}"
    perm = hf_to_meta_perm(head_dim)
    return weight.reshape(n_heads, head_dim, -1)[:, perm].reshape(weight.shape)


def meta_cos_sin(cfg, seq_len: int):
    """Meta-layout cos/sin ``[1, 1, seq_len, head_dim]`` bf16 for positions [0, seq_len): the reference's
    fp32 values permuted, then rounded to bf16 once (identical to the HF / golden bf16 cos/sin)."""
    cos, sin = rope_cos_sin(cfg, torch.arange(seq_len), dtype=torch.float32)
    perm = hf_to_meta_perm(cfg.head_dim)
    return cos[:, perm].to(torch.bfloat16)[None, None], sin[:, perm].to(torch.bfloat16)[None, None]


class RopeSetup:
    """Device cos/sin tables for every KV-cache position plus the RoPE transformation matrix.

    ``chunk_size`` is the block-cyclic period of the KV cache these positions address (one-shot: the
    whole sequence). Both tables are SP-sharded on the sequence and replicated across TP.
    """

    def __init__(self, mesh_device, mesh_config, cfg, *, max_seq_len: int, chunk_size: int):
        sp = mesh_config.sp
        assert max_seq_len % chunk_size == 0, f"max_seq_len {max_seq_len} not a multiple of chunk {chunk_size}"
        assert chunk_size % (ttnn.TILE_SIZE * sp) == 0, f"chunk {chunk_size} must be a multiple of 32 * sp"
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.cfg = cfg
        self.max_seq_len = max_seq_len
        self.chunk_size = chunk_size
        chunk_local = chunk_size // sp
        mapper = mesh_config.mapper(mesh_device, sp_dim=2)

        def upload(table):
            return ttnn.from_torch(
                block_cyclic_reorder(table, chunk_local, sp, seq_dim=2),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        cos, sin = meta_cos_sin(cfg, max_seq_len)
        self.cos, self.sin = upload(cos), upload(sin)
        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x, kv_actual: int):
        """Rotate per-chip Q or K ``[1, heads, chunk_local, head_dim]`` of the chunk starting at ``kv_actual``."""
        assert kv_actual % self.chunk_size == 0, f"kv_actual {kv_actual} is not chunk-aligned ({self.chunk_size})"
        return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
            x,
            self.cos,
            self.sin,
            self.trans_mat,
            kv_actual_global=kv_actual,
            cluster_axis=self.mesh_config.sp_axis,
            compute_kernel_config=self.compute_kernel_config,
        )
