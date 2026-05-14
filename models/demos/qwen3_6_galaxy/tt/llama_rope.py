# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MRoPE / partial-RoPE setup for Qwen3.6-27B on BH GLX 8×4 mesh.

Forked from models/demos/llama3_70b_galaxy/tt/llama_rope.py and adapted for:
- Partial rotary: rotary_dim = 64 of head_dim = 256 (partial_rotary_factor = 0.25)
- MRoPE sections [11, 11, 10]: per-channel selection from 3 position axes
- rope_theta = 10_000_000 (vs the default 10000 used in earlier Llama variants)
- Text-only inference: all 3 position axes are the same (token index), so MRoPE
  collapses to standard partial RoPE with MRoPE-section channel grouping.

Key design decision — option (b) for partial RoPE application:
    We implement the rotation using ttnn primitive ops (slice, neg, concat, mul, add)
    rather than the fused rotary_embedding_llama_fused_qk kernel. The fused kernel
    expects full head_dim rotary coverage; slicing to 64-dim and calling the fused
    kernel may produce incorrect results due to program config assumptions.

    Instead, apply_partial_rope:
        1. Slice x[..., :rotary_dim]  → x_rot
        2. Slice x[..., rotary_dim:]  → x_pass (unchanged)
        3. Compute rotate_half(x_rot) = cat([-x2, x1]) with x1, x2 = chunk(2, last)
        4. Output = x_rot * cos + rotate_half * sin
        5. Concat [rotated, x_pass]

API
---
    rope = Qwen36RopeSetup(mesh_device, args, batch_size, max_seq_len)
    cos_tt, sin_tt = rope.get_cos_sin_for_decode(cur_pos)   # [1,1,1,64]
    cos_tt, sin_tt = rope.get_cos_sin_for_prefill(seq_len)  # [1,1,seq_len,64]
    q_rotated = rope.apply_partial_rope(q_tt, cos_tt, sin_tt)

The cos/sin tables are stored as TTNN tensors replicated across all 32 chips.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin
from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs


class Qwen36RopeSetup(LightweightModule):
    """MRoPE / partial-RoPE tables and application for Qwen3.6-27B.

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        The full 8×4 BH GLX mesh.
    args : TtQwen36ModelArgs
        Model configuration.  Must have ``rope_dim``, ``head_dim``,
        ``mrope_section``, ``mrope_theta`` attributes.
    batch_size : int
        Decode batch size (not used internally yet, kept for API parity).
    max_seq_len : int
        Maximum sequence length for which tables are precomputed.
    datatype : ttnn.DataType
        On-device dtype for cos/sin tables.  Defaults to bfloat16.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        args: TtQwen36ModelArgs,
        batch_size: int = 1,
        max_seq_len: int = 128 * 1024,
        datatype=ttnn.bfloat16,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.rotary_dim = args.rope_dim  # 64
        self.head_dim = args.head_dim  # 256
        self.datatype = datatype

        # Verify tile alignment: rotary_dim=64 is 2×tile; max_seq_len must be ≥ 32
        assert self.rotary_dim % 32 == 0, f"rotary_dim={self.rotary_dim} must be tile-aligned (multiple of 32)"
        assert max_seq_len % 32 == 0 or max_seq_len >= 1, "max_seq_len should be ≥ 1"

        # ------------------------------------------------------------------
        # Precompute cos/sin tables on CPU using the reference oracle
        # ------------------------------------------------------------------
        # For text-only inference: all three position axes = token index
        # This collapses MRoPE → standard partial RoPE with mrope_section grouping.
        positions = torch.arange(max_seq_len, dtype=torch.long)
        positions_3d = torch.stack([positions, positions, positions], dim=0)  # [3, max_seq_len]

        # build_mrope_cos_sin returns (cos, sin) both of shape [1, max_seq_len, rotary_dim]
        cos_table, sin_table = build_mrope_cos_sin(
            positions_3d=positions_3d,
            head_dim=args.head_dim,
            partial_rotary_factor=args.partial_rotary_factor,
            mrope_section=args.mrope_section,
            theta=args.mrope_theta,
        )
        # cos_table, sin_table: [1, max_seq_len, rotary_dim] float32

        # Reshape to 4D: [1, 1, max_seq_len, rotary_dim] for TTNN tile layout
        cos_4d = cos_table.unsqueeze(0)  # [1, 1, max_seq_len, rotary_dim]
        sin_4d = sin_table.unsqueeze(0)

        # ------------------------------------------------------------------
        # Store on device — replicated across all chips
        # ------------------------------------------------------------------
        # cos/sin tables are small and shared by all chips; full replication is
        # the simplest and correct approach.
        self.cos_table_tt = ttnn.from_torch(
            cos_4d,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.sin_table_tt = ttnn.from_torch(
            sin_4d,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Keep CPU tables for cheap slicing in get_cos_sin_for_decode
        self._cos_table_cpu = cos_4d  # [1, 1, max_seq_len, rotary_dim] float32
        self._sin_table_cpu = sin_4d

        # ------------------------------------------------------------------
        # T14b.9: 2D cos/sin matrices for on-device embedding lookup.
        # ttnn.embedding requires a 2-D weight [vocab_size, embedding_dim],
        # so we store cos/sin as [max_seq_len, rotary_dim] ROW_MAJOR
        # replicated. The trace-friendly decode RoPE path uses these via
        # ``get_rm_rot_mats(rope_idxs)`` — purely on-device, no host writes.
        # Matches the llama3_70b_galaxy pattern (llama_rope.py:63-76).
        # ------------------------------------------------------------------
        cos_2d = cos_table.squeeze(0)  # [max_seq_len, rotary_dim]
        sin_2d = sin_table.squeeze(0)
        # ROW_MAJOR for ttnn.embedding (the op requires ROW_MAJOR weight).
        self.cos_matrix = ttnn.from_torch(
            cos_2d,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.sin_matrix = ttnn.from_torch(
            sin_2d,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # ------------------------------------------------------------------
        # T14b.7: persistent device buffers for the per-step decode slice.
        # Pre-allocated once so get_cos_sin_for_decode can refresh values via
        # ttnn.copy_host_to_device_tensor (trace-safe) instead of issuing a
        # fresh ttnn.from_torch with device= every call (which is forbidden
        # inside a captured trace).
        # ------------------------------------------------------------------
        decode_init = torch.zeros(1, 1, 1, self.rotary_dim, dtype=torch.bfloat16)
        self._cos_decode_buf = ttnn.from_torch(
            decode_init,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self._sin_decode_buf = ttnn.from_torch(
            decode_init,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cos_sin_for_decode(self, cur_pos: int):
        """Return cos/sin for a single decode step at position cur_pos.

        T14b.7: refreshes the persistent device buffers via
        ``ttnn.copy_host_to_device_tensor`` (trace-capture-safe; the buffer
        addresses stay fixed across decode calls so a captured trace can
        replay reads from them after each refresh).

        Returns
        -------
        (cos_tt, sin_tt) : each TTNN tensor of shape [1, 1, 1, rotary_dim]
            Replicated across all mesh devices. These are the persistent
            buffers — callers must NOT deallocate them.
        """
        assert 0 <= cur_pos < self.max_seq_len, f"cur_pos={cur_pos} out of range [0, {self.max_seq_len})"

        # Slice CPU table: [1, 1, max_seq_len, rd] → [1, 1, 1, rd]
        cos_slice = self._cos_table_cpu[:, :, cur_pos : cur_pos + 1, :]  # [1,1,1,64]
        sin_slice = self._sin_table_cpu[:, :, cur_pos : cur_pos + 1, :]

        # Build HOST tensors (no device=) and copy into the persistent device
        # buffers in-place.  ttnn.from_torch without device= is a host-only
        # allocation and is allowed inside trace capture; the device write
        # happens via copy_host_to_device_tensor, which targets pre-existing
        # buffers and is also trace-safe.
        cos_host = ttnn.from_torch(
            cos_slice,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        sin_host = ttnn.from_torch(
            sin_slice,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(cos_host, self._cos_decode_buf)
        ttnn.copy_host_to_device_tensor(sin_host, self._sin_decode_buf)
        return self._cos_decode_buf, self._sin_decode_buf

    def get_cos_sin_for_prefill(self, seq_len: int):
        """Return cos/sin for prefill over positions [0, seq_len).

        T14b.7: returns on-device slices of the pre-uploaded full-length
        tables (``self.cos_table_tt`` / ``self.sin_table_tt`` are built once
        at __init__).  Trace-capture-safe — the slice op is recorded with
        literal indices, which is valid for fixed-seq_len trace replay
        (the only safe shape regime for prefill trace anyway, per the
        SDPA-program-config constraint documented in the optimization skill).

        Returns
        -------
        (cos_tt, sin_tt) : each TTNN tensor of shape [1, 1, seq_len, rotary_dim]
            Replicated across all mesh devices.
        """
        assert 0 < seq_len <= self.max_seq_len, f"seq_len={seq_len} out of range (0, {self.max_seq_len}]"

        cos_tt = ttnn.slice(
            self.cos_table_tt,
            [0, 0, 0, 0],
            [1, 1, seq_len, self.rotary_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin_tt = ttnn.slice(
            self.sin_table_tt,
            [0, 0, 0, 0],
            [1, 1, seq_len, self.rotary_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return cos_tt, sin_tt

    # ------------------------------------------------------------------
    # T14b.9 — trace-friendly decode RoPE (pure device)
    # ------------------------------------------------------------------

    def get_rm_rot_idxs(self, position_idxs: torch.Tensor, on_host: bool = False) -> ttnn.Tensor:
        """Build the ``rot_idxs`` lookup tensor for a decode batch.

        T14b.9 step 6: pad to ``[TILE, TILE] = [32, 32]`` so the downstream
        ``ttnn.embedding`` output is tile-aligned in both dims and the op
        doesn't fire an internal tile-padding host-write inside trace
        capture. The actual position lives at index ``[0, 0]`` (single
        user, replicated across the rest of the tile). Mirrors the
        tile-padding approach in olmo3's ``get_rm_rot_idxs`` and
        llama3_70b_galaxy's ``get_rm_rot_idxs``.

        Parameters
        ----------
        position_idxs : torch.Tensor
            ``[1]`` int64 CPU tensor holding the current position for the
            single user in the batch (qwen3.6 max_batch_size = 1).
        on_host : bool
            When True, returns a HOST ttnn tensor that the caller will push
            to device via ``ttnn.copy_host_to_device_tensor``. When False
            (eager path), the tensor is uploaded to device immediately.

        Returns
        -------
        ttnn.Tensor of shape ``[32, 32]`` uint32, replicated across the
            mesh. All entries hold the same position value; downstream
            ``get_rm_rot_mats`` slices ``[0, 0]`` to recover the single
            cos/sin row.
        """
        assert isinstance(position_idxs, torch.Tensor), "position_idxs must be a torch tensor"
        assert position_idxs.numel() == 1, f"qwen3.6 max_batch_size=1; got {tuple(position_idxs.shape)}"
        # Broadcast the single position across a TILE×TILE grid so the
        # embedding output is naturally tile-aligned.
        pos = int(position_idxs.view(-1)[0].item())
        position_idxs_padded = torch.full((32, 32), pos, dtype=torch.int32)

        rot_idxs = ttnn.from_torch(
            position_idxs_padded,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None if on_host else self.mesh_device,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return rot_idxs

    def get_rm_rot_mats(self, rot_idxs: ttnn.Tensor):
        """Gather cos/sin for a decode position via on-device ``ttnn.embedding``.

        Parameters
        ----------
        rot_idxs : ttnn.Tensor
            ``[1, 1]`` uint32 device tensor produced by ``get_rm_rot_idxs``.

        Returns
        -------
        (cos_tt, sin_tt) : each ``[1, 1, 1, rotary_dim]`` device tensors,
            replicated across the mesh. Trace-safe: pure device ops, no
            host writes — the cos/sin tables are pre-uploaded in
            ``self.cos_matrix`` / ``self.sin_matrix`` once at __init__.
        """
        # rot_idxs has shape [32, 32] (T14b.9 step 6 — tile-aligned to avoid
        # internal ttnn.embedding padding write inside trace capture).
        # Embedding output: [32, 32, rotary_dim]. We slice down to
        # [1, 1, rotary_dim] for the single position we actually care about.
        cos_padded = ttnn.embedding(
            rot_idxs,
            self.cos_matrix,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [32, 32, rotary_dim]
        sin_padded = ttnn.embedding(
            rot_idxs,
            self.sin_matrix,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Slice to [1, 1, rotary_dim] — a real (non-no-op) slice that
        # allocates a new device tensor; pure device op (no host write).
        cos = ttnn.slice(cos_padded, [0, 0, 0], [1, 1, self.rotary_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin = ttnn.slice(sin_padded, [0, 0, 0], [1, 1, self.rotary_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos_padded.deallocate(True)
        sin_padded.deallocate(True)
        # Unsqueeze to [1, 1, 1, rotary_dim] to match apply_partial_rope's expected
        # cos/sin shape (4-D, broadcast over heads).
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        return cos, sin

    def apply_partial_rope(
        self,
        x_tt: ttnn.Tensor,
        cos_tt: ttnn.Tensor,
        sin_tt: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Apply partial RoPE to query or key tensor.

        Only the first ``rotary_dim`` channels of the last dimension are rotated;
        the remaining ``head_dim - rotary_dim`` channels are passed through unchanged.

        Args:
            x_tt:   TTNN tensor of shape [..., head_dim] (any leading dims).
                    Typically [B, n_heads, T, head_dim] or [B, 1, T, head_dim].
            cos_tt: TTNN tensor of shape [1, 1, T, rotary_dim]  (or [1, 1, 1, rotary_dim] for decode).
            sin_tt: TTNN tensor matching cos_tt shape.

        Returns:
            TTNN tensor of same shape as x_tt with first rotary_dim dims rotated.
        """
        rd = self.rotary_dim  # 64
        hd = self.head_dim  # 256

        # Get the last dimension (head_dim)
        shape = list(x_tt.shape)
        last_dim = shape[-1]
        assert last_dim == hd, f"Expected last dim={hd}, got {last_dim}"

        # ------------------------------------------------------------------
        # Slice: x_rot = x[..., :rd],  x_pass = x[..., rd:]
        # ------------------------------------------------------------------
        # Use ttnn.slice with begin/end on the last dimension
        ndim = len(shape)

        # Build slice ranges: all dims full, last dim sliced
        begins_rot = [0] * ndim
        ends_rot = shape[:]
        ends_rot[-1] = rd

        begins_pass = [0] * ndim
        ends_pass = shape[:]
        begins_pass[-1] = rd

        x_rot = ttnn.slice(x_tt, begins_rot, ends_rot)  # [..., rd]
        x_pass = ttnn.slice(x_tt, begins_pass, ends_pass)  # [..., hd-rd]

        # ------------------------------------------------------------------
        # rotate_half(x_rot) = cat([-x2, x1]) where x1 = x_rot[..., :rd//2], x2 = x_rot[..., rd//2:]
        # ------------------------------------------------------------------
        half = rd // 2  # 32

        shape_rot = list(x_rot.shape)
        begins_x1 = [0] * ndim
        ends_x1 = shape_rot[:]
        ends_x1[-1] = half

        begins_x2 = [0] * ndim
        ends_x2 = shape_rot[:]
        begins_x2[-1] = half

        x1 = ttnn.slice(x_rot, begins_x1, ends_x1)  # [..., half]
        x2 = ttnn.slice(x_rot, begins_x2, ends_x2)  # [..., half]

        neg_x2 = ttnn.neg(x2)  # [-x2]
        rotate_half = ttnn.concat([neg_x2, x1], dim=-1)  # [..., rd]  = [-x2, x1]

        # Deallocate intermediates
        x1.deallocate(True)
        x2.deallocate(True)
        neg_x2.deallocate(True)

        # ------------------------------------------------------------------
        # Rotated = x_rot * cos + rotate_half * sin
        # ------------------------------------------------------------------
        x_rot_cos = ttnn.multiply(x_rot, cos_tt)
        rh_sin = ttnn.multiply(rotate_half, sin_tt)
        x_rotated = ttnn.add(x_rot_cos, rh_sin)

        x_rot.deallocate(True)
        rotate_half.deallocate(True)
        x_rot_cos.deallocate(True)
        rh_sin.deallocate(True)

        # ------------------------------------------------------------------
        # Concat rotated + pass-through
        # ------------------------------------------------------------------
        out = ttnn.concat([x_rotated, x_pass], dim=-1)  # [..., hd]

        x_rotated.deallocate(True)
        x_pass.deallocate(True)

        return out
