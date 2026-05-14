# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from models.demos.llama3_70b_galaxy.tt.llama_common import gather_cos_sin, get_rot_transformation_mat, precompute_freqs


def compute_gather_cos_sin(dhead, end, theta, position_ids, use_scaled_rope, scale_factor):
    cos, sin = precompute_freqs(dhead, end, theta, use_scaled_rope, scale_factor)
    return gather_cos_sin(position_ids, cos, sin)


# ---------------------------------------------------------------------------
# Qwen3.6 partial-RoPE helpers (CPU, used for unit-test + on-device table prep)
# ---------------------------------------------------------------------------


def build_qwen36_partial_rope_tables(
    max_seq_len: int,
    rope_dim: int,
    rope_theta: float,
) -> tuple:
    """Build cos/sin tables for Qwen3.6 partial RoPE (text-only path).

    Qwen3.6 uses MRoPE with sections [11, 11, 10], but in text-only inference
    all three position axes equal the token index, which makes the MRoPE
    section grouping numerically identical to standard partial RoPE.

    The HF Qwen3NextRotaryEmbedding builds frequencies via::

        inv_freq = 1 / theta ** (torch.arange(0, rope_dim, 2) / rope_dim)
        freqs    = outer(positions, inv_freq)        # [T, rope_dim/2]
        emb      = cat([freqs, freqs], dim=-1)       # [T, rope_dim]
        cos, sin = emb.cos(), emb.sin()

    Returns
    -------
    (cos, sin) : torch.Tensor each of shape [max_seq_len, rope_dim] (float32).
    """
    assert rope_dim % 2 == 0, f"rope_dim must be even, got {rope_dim}"
    half = rope_dim // 2
    # inv_freq[i] = 1 / theta^(2i/rope_dim) for i in [0, half)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, rope_dim]
    return emb.cos(), emb.sin()


def apply_qwen36_partial_rope_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Apply Qwen3.6 partial RoPE to a [..., head_dim] tensor in pure pytorch.

    Math (matches HF reference ``apply_rotary_pos_emb`` + ``_rotate_half`` in
    ``models/demos/qwen3_6_galaxy/reference/qwen36.py`` lines 180-280)::

        x_rot, x_pass = x[..., :rope_dim], x[..., rope_dim:]
        x1, x2        = x_rot[..., :rope_dim//2], x_rot[..., rope_dim//2:]
        rotate_half   = cat([-x2, x1], dim=-1)
        x_rot_out     = x_rot * cos + rotate_half * sin
        out           = cat([x_rot_out, x_pass], dim=-1)

    Parameters
    ----------
    x : torch.Tensor [..., head_dim]
    cos, sin : torch.Tensor broadcastable to [..., rope_dim]
        Typically [1, 1, T, rope_dim] or [1, 1, 1, rope_dim] for decode.

    Returns
    -------
    torch.Tensor of same shape as x.
    """
    head_dim = x.shape[-1]
    assert rope_dim <= head_dim, f"rope_dim={rope_dim} > head_dim={head_dim}"
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    half = rope_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotate_half = torch.cat([-x2, x1], dim=-1)
    x_rot_out = x_rot * cos + rotate_half * sin
    return torch.cat([x_rot_out, x_pass], dim=-1)


class TtLlamaRotarySetup(LightweightModule):
    """RoPE setup for Llama-class models (70B / Qwen3-32B) and Qwen3.6 partial RoPE.

    Backward-compatible with the original llama3_70b_galaxy positional signature:
    ``TtLlamaRotarySetup(device, batch_size, head_dim, max_seq_len, rope_theta, use_scaled_rope, scale_factor)``.

    Qwen3.6 partial-RoPE behaviour is gated by ``getattr(args, "is_qwen36", False)``
    on an optional ``args`` keyword. When set:

    * cos/sin tables are computed at ``args.rope_dim`` (= 64) width, NOT
      ``args.head_dim`` (= 256). This means ``self.head_dim`` (the width of
      gathered cos/sin shards used by downstream RoPE math) is set to
      ``rope_dim``, while the original head_dim is stored as ``self.full_head_dim``.
    * ``rope_theta`` defaults to ``args.rope_theta`` (10_000_000 for qwen3.6).
    * ``self.sub_core_grids`` is taken directly from ``args.sub_core_grids``
      instead of the hard-coded prefetcher-aware split (cols 1-3 + 5-6).
    * A ``partial_rope_apply(x, cos, sin)`` helper exposes the slice→rotate→concat
      math against a pre-fetched cos/sin pair (works for both prefill and decode
      shapes; mirrors v1's ``apply_partial_rope`` in
      ``models/demos/qwen3_6_galaxy/tt/llama_rope.py``).

    The original 70B / qwen3-32B paths are untouched: they pass no ``args``
    and continue to compute full-width cos/sin tables exactly as before.
    """

    def __init__(
        self,
        device,
        batch_size: int,
        head_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000,
        use_scaled_rope: bool = False,
        scale_factor: float = 8,
        datatype=ttnn.bfloat16,
        args=None,
    ):
        super().__init__()

        # --- Qwen3.6 partial-RoPE branch detection ---
        self.is_qwen36 = bool(getattr(args, "is_qwen36", False))
        if self.is_qwen36:
            # Take RoPE-specific config straight from args.
            self.rope_dim = int(args.rope_dim)  # 64
            self.full_head_dim = int(args.head_dim)  # 256
            # rope_theta argument is ignored when args.is_qwen36 — use args.rope_theta.
            rope_theta = float(args.rope_theta)
            use_scaled_rope = False
            scale_factor = 1.0
            # The cos/sin tables we precompute have width = rope_dim.
            # Downstream code (ttnn.embedding output, sharding shape) uses
            # self.head_dim as the cos/sin width, so override it here.
            self.head_dim = self.rope_dim
        else:
            self.rope_dim = head_dim
            self.full_head_dim = head_dim
            self.head_dim = head_dim

        self.batch_size = batch_size
        self.n_kv_heads = 8
        self.device = device
        self.is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)
        self.num_devices = device.get_num_devices() if self.is_mesh_device else 1
        if self.num_devices == 32:
            self.batch_size_per_device_group = (
                max(self.batch_size // list(device.shape)[1], 1) * 2
            )  # TODO: fix for batch=1
        else:
            self.batch_size_per_device_group = self.batch_size
        self.core_grid = device.compute_with_storage_grid_size()
        num_cores = self.core_grid.x * self.core_grid.y

        # --- sub_core_grids: prefer args.sub_core_grids when provided ---
        # Qwen3.6 v2 widens this to cols 1-6 (prefetcher off, col-4 reclaimed).
        if args is not None and getattr(args, "sub_core_grids", None) is not None:
            self.sub_core_grids = args.sub_core_grids
            self.start_core = getattr(args, "start_core", ttnn.CoreCoord(1, 0))
        else:
            # Legacy 70B / qwen3-32B prefetcher-aware split.
            self.sub_core_grids = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                ]
            )
            self.start_core = ttnn.CoreCoord(1, 0)

        # ------------------------------------------------------------------
        # Generate the cos/sin matrices needed for ttnn.embedding op.
        # For qwen3.6 we build at rope_dim width via the partial-RoPE oracle;
        # for other models we keep the original full-head_dim path.
        # ------------------------------------------------------------------
        if self.is_qwen36:
            cos_2d, sin_2d = build_qwen36_partial_rope_tables(
                max_seq_len=max_seq_len * 2,
                rope_dim=self.rope_dim,
                rope_theta=rope_theta,
            )  # each [max_seq_len*2, rope_dim] float32
            # Match the gather_cos_sin output layout: [1, 1, T, rope_dim].
            # gather_cos_sin produces a [1, 1, T, head_dim] tensor by
            # interleaving (stack-then-flatten); for qwen3.6 partial RoPE the
            # canonical layout is cat([f, f], dim=-1) which we already built
            # directly above, so just expand dims.
            positions = torch.arange(max_seq_len)
            cos_matrix = cos_2d[positions].unsqueeze(0).unsqueeze(0)  # [1, 1, max_seq_len, rope_dim]
            sin_matrix = sin_2d[positions].unsqueeze(0).unsqueeze(0)
        else:
            cos_matrix, sin_matrix = compute_gather_cos_sin(
                dhead=head_dim,
                end=max_seq_len * 2,
                theta=rope_theta,
                position_ids=torch.arange(max_seq_len),
                use_scaled_rope=use_scaled_rope,
                scale_factor=scale_factor,
            )

        self.cos_matrix = ttnn.from_torch(
            cos_matrix,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )
        self.sin_matrix = ttnn.from_torch(
            sin_matrix,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=datatype,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

        # ------------------------------------------------------------------
        # V2-10: 2D cos/sin matrices for in-trace ttnn.embedding lookup.
        # The default ``self.cos_matrix`` / ``self.sin_matrix`` are stored as
        # 4-D ``[1, 1, max_seq_len, rope_dim]`` ROW_MAJOR replicated, which is
        # what the eager path consumes (``ttnn.embedding`` accepts both).
        # For the qwen3.6 in-trace decode path we want a clean
        # ``[max_seq_len, rope_dim]`` 2-D lookup table that emits
        # ``[T, T, rope_dim]`` output for a tile-aligned ``[T, T]`` rot_idxs
        # tensor (T = 32). Mirrors v1's ``get_rm_rot_mats`` precedent:
        # ``models/demos/qwen3_6_galaxy/tt/llama_rope.py:308``.
        # ------------------------------------------------------------------
        if self.is_qwen36:
            # cos_2d / sin_2d already computed above. Replicated across mesh.
            self.cos_matrix_2d_q36 = ttnn.from_torch(
                cos_2d,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=datatype,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
            )
            self.sin_matrix_2d_q36 = ttnn.from_torch(
                sin_2d,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=datatype,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
            )

        self.core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
            self.start_core, self.batch_size_per_device_group, self.sub_core_grids, row_wise=True
        )
        # Generate the transformation matrix
        trans_mat = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(
            1,
            1,
            self.batch_size_per_device_group,
            1
            # 1, 1, num_cores, 1
        )  # Repeat across all cores on device
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=self.core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.transformation_mat = ttnn.from_torch(
            trans_mat,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=trans_mat_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

        # TODO: Colman, should this be TILE_SIZE or head_dim? Why should it be different for prefill and decode?
        prefill_trans_mat_torch = get_rot_transformation_mat(dhead=head_dim)
        self.transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat_torch,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=datatype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if self.is_mesh_device else None,
        )

    def get_both_trans_mats(self):
        assert self.transformation_mat is not None, "Transformation matrix not initialized"
        assert self.transformation_mat_prefill is not None, "Prefill Transformation matrix not initialized"
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    def get_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"
        assert position_idxs.shape[0] == 32, "position idxs must be a [32] tensor"
        # repeating twice at every 8th position for fused kv rope
        position_idxs = position_idxs.view(-1, 8)  # [4, 8]
        position_idxs = position_idxs.repeat(1, 2)  # [4, 16]
        position_idxs = position_idxs.view(-1)  # [64]
        batch = position_idxs.shape[0]
        position_idxs = position_idxs.reshape(1, batch)  # [1, 1, 1, batch]
        assert position_idxs.shape == (1, batch), "position idxs must be a [1, batch] tensor"
        assert torch.min(position_idxs) >= 0, "position idxs must be non-negative"

        # Add padding if needed
        pad_size = nearest_32(batch) - batch
        position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)
        position_idxs = position_idxs.transpose(1, 0)
        position_idxs = torch.nn.functional.pad(position_idxs, (0, 31), "constant", 0)

        if on_host:  # If tensor is on host, don't pass a mesh mapper if single-device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 0) if (self.num_devices == 32 and batch > 1) else (None, None),
                    mesh_shape=list(self.device.shape),
                )
                if self.is_mesh_device
                else None,
            )
        else:  # On device
            rot_idxs = ttnn.as_tensor(
                position_idxs,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 0) if (self.num_devices == 32 and batch > 1) else (None, None),
                    mesh_shape=list(self.device.shape),
                )
                if self.is_mesh_device
                else None,
            )

        return rot_idxs

    def get_rm_rot_idxs(self, position_idxs, on_host=False):
        assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
        assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"
        assert position_idxs.shape[0] == 32, "position idxs must be a [32] tensor"
        position_idxs = position_idxs.view(-1, 8)  # [4, 8]
        position_idxs = position_idxs.repeat(1, 2)  # [4, 16]
        position_idxs = position_idxs.view(-1, 1)  # [64, 1]
        position_idxs = position_idxs.repeat(1, self.n_kv_heads)  # [64, 8]

        batch = position_idxs.shape[0]

        rot_idxs = ttnn.as_tensor(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None if on_host else self.device,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.device,
                dims=(None, 0) if (self.num_devices == 32 and batch > 1) else (None, None),
                mesh_shape=list(self.device.shape),
            )
            if self.is_mesh_device
            else None,
        )

        return rot_idxs

    def get_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs
            # assert len(rot_idxs.shape) == 2 and rot_idxs.shape[0] == 1, "rot_idxs must be a [1, batch] tensor"

        # Send the idxs to device
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=self.core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        embedding_layout = ttnn.TILE_LAYOUT
        cos = ttnn.embedding(
            rot_idxs, self.cos_matrix, layout=embedding_layout, memory_config=mem_config
        )  # [1, batch, head_dim]
        sin = ttnn.embedding(
            rot_idxs, self.sin_matrix, layout=embedding_layout, memory_config=mem_config
        )  # [1, batch, head_dim]

        cos = ttnn.reshape(
            cos,
            ttnn.Shape(
                [self.batch_size_per_device_group, 1, cos.shape[-1]]
                # (self.batch_size_per_device_group, 32, cos.shape[-1]),
            ),
            ttnn.Shape(
                # [self.batch_size_per_device_group, 1, sin.shape[-1]]
                (self.batch_size_per_device_group, 32, sin.shape[-1]),
            ),
        )
        sin = ttnn.reshape(
            sin,
            ttnn.Shape(
                [self.batch_size_per_device_group, 1, sin.shape[-1]]
                # (self.batch_size_per_device_group, 32, sin.shape[-1]),
            ),
            ttnn.Shape(
                # [self.batch_size_per_device_group, 1, sin.shape[-1]]
                (self.batch_size_per_device_group, 32, sin.shape[-1]),
            ),
        )

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, batch, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, 1, batch, head_dim]

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]

    def get_rm_rot_mats(self, position_idxs, return_rot_idxs=False):
        device = self.device

        # If position_idxs is a torch tensor, get the TTNN version of it
        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rm_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs

        # Send the idxs to device
        if rot_idxs.device != device:
            rot_idxs = ttnn.to_device(rot_idxs, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        mem_config = ttnn.create_sharded_memory_config(
            shape=(self.n_kv_heads, self.head_dim),
            core_grid=self.core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        embedding_layout = ttnn.ROW_MAJOR_LAYOUT
        cos = ttnn.embedding(
            rot_idxs, self.cos_matrix, layout=embedding_layout, memory_config=mem_config
        )  # [batch, n_kv_heads, head_dim]
        sin = ttnn.embedding(
            rot_idxs, self.sin_matrix, layout=embedding_layout, memory_config=mem_config
        )  # [batch, n_kv_heads, head_dim]

        cos = ttnn.unsqueeze_to_4D(cos)  # [1, batch, n_kv_heads, head_dim]
        sin = ttnn.unsqueeze_to_4D(sin)  # [1, batch, n_kv_heads, head_dim]

        if return_rot_idxs:
            return [cos, sin], rot_idxs
        return [cos, sin]

    def get_prefill_rot_mats(self, position_ids, seq_len):
        """
        Compute prefill rotary matrices from position indices using embedding lookup.

        Args:
            position_ids: ttnn tensor of shape [1, seq_len] with position indices (on device)
            seq_len: sequence length

        Returns:
            [cos, sin] list where each has shape [1, 1, seq_len, head_dim]
        """

        # Reshape position_ids for embedding lookup: [1, seq_len] -> [seq_len, 1]
        rot_idxs = ttnn.reshape(position_ids, [seq_len, 1])

        # Look up cos/sin values from the pre-computed embedding tables
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)

        # Reshape to [1, 1, seq_len, head_dim] to match expected prefill format
        cos = ttnn.reshape(cos, [1, 1, seq_len, self.head_dim])
        sin = ttnn.reshape(sin, [1, 1, seq_len, self.head_dim])

        return [cos, sin]

    # ------------------------------------------------------------------
    # Qwen3.6 partial-RoPE application (TTNN device path, option (a))
    # ------------------------------------------------------------------

    def partial_rope_apply(
        self,
        x_tt,
        cos_tt,
        sin_tt,
    ):
        """Apply Qwen3.6 partial RoPE to a ``[..., head_dim]`` device tensor.

        Only the first ``self.rope_dim`` channels of the last dim are rotated;
        the remaining ``self.full_head_dim - self.rope_dim`` channels pass
        through unchanged. Ported verbatim from v1
        ``models/demos/qwen3_6_galaxy/tt/llama_rope.py::apply_partial_rope``
        which is the known-good math (PCC > 0.99 in v1 single-block tests).

        Parameters
        ----------
        x_tt : ttnn.Tensor
            Shape ``[..., full_head_dim]`` (e.g. ``[B, n_heads, T, 256]`` for
            prefill, ``[1, 1, B, 256]`` for decode after concat).
        cos_tt, sin_tt : ttnn.Tensor
            Shape broadcastable to ``[..., rope_dim]`` — typically
            ``[1, 1, T, rope_dim]`` for prefill or ``[1, 1, 1, rope_dim]`` for
            decode.

        Returns
        -------
        ttnn.Tensor of same shape as ``x_tt`` with the first ``rope_dim``
        channels rotated.

        Notes
        -----
        Must be called only when ``self.is_qwen36`` is True (the 70B / qwen3-32B
        paths use the fused ``ttnn.experimental.rotary_embedding_llama_fused_qk``
        kernel and do not need this helper).
        """
        assert self.is_qwen36, "partial_rope_apply is only valid when is_qwen36=True"
        rd = self.rope_dim  # 64
        hd = self.full_head_dim  # 256
        shape = list(x_tt.shape)
        last_dim = shape[-1]
        assert last_dim == hd, f"Expected last dim={hd}, got {last_dim}"

        ndim = len(shape)
        # Slice into rotated / pass-through halves.
        begins_rot = [0] * ndim
        ends_rot = shape[:]
        ends_rot[-1] = rd
        begins_pass = [0] * ndim
        ends_pass = shape[:]
        begins_pass[-1] = rd
        x_rot = ttnn.slice(x_tt, begins_rot, ends_rot)  # [..., rd]
        x_pass = ttnn.slice(x_tt, begins_pass, ends_pass)  # [..., hd-rd]

        # rotate_half(x_rot) = cat([-x2, x1]) with x1, x2 halves of x_rot.
        half = rd // 2
        shape_rot = list(x_rot.shape)
        begins_x1 = [0] * ndim
        ends_x1 = shape_rot[:]
        ends_x1[-1] = half
        begins_x2 = [0] * ndim
        ends_x2 = shape_rot[:]
        begins_x2[-1] = half
        x1 = ttnn.slice(x_rot, begins_x1, ends_x1)
        x2 = ttnn.slice(x_rot, begins_x2, ends_x2)
        neg_x2 = ttnn.neg(x2)
        rotate_half = ttnn.concat([neg_x2, x1], dim=-1)  # [..., rd]
        x1.deallocate(True)
        x2.deallocate(True)
        neg_x2.deallocate(True)

        # Rotated = x_rot * cos + rotate_half * sin
        x_rot_cos = ttnn.multiply(x_rot, cos_tt)
        rh_sin = ttnn.multiply(rotate_half, sin_tt)
        x_rotated = ttnn.add(x_rot_cos, rh_sin)
        x_rot.deallocate(True)
        rotate_half.deallocate(True)
        x_rot_cos.deallocate(True)
        rh_sin.deallocate(True)

        # Concat rotated + pass-through
        out = ttnn.concat([x_rotated, x_pass], dim=-1)  # [..., hd]
        x_rotated.deallocate(True)
        x_pass.deallocate(True)
        return out

    # ------------------------------------------------------------------
    # V2-10: in-trace qwen3.6 partial-RoPE lookup helpers.
    # Mirror v1 ``get_rm_rot_idxs`` / ``get_rm_rot_mats`` semantics: a
    # tile-aligned [32, 32] uint32 rot_idxs tensor + on-device
    # ttnn.embedding lookup that returns ``[1, 1, 1, rope_dim]`` cos/sin.
    # Pure device ops (no host writes); safe inside trace capture.
    # ------------------------------------------------------------------

    def get_qwen36_rm_rot_idxs(self, cur_pos: int, on_host: bool = False):
        """Build a tile-aligned ``[32, 32]`` rot_idxs tensor for qwen3.6 decode.

        All 1024 entries hold the same position value; ``get_qwen36_rm_rot_mats``
        slices ``[0, 0]`` to recover the single cos/sin row.

        Parameters
        ----------
        cur_pos : int
            Current decode position.
        on_host : bool
            When True returns a HOST ttnn tensor for ``copy_host_to_device_tensor``.

        Returns
        -------
        ttnn.Tensor ``[32, 32]`` uint32, replicated.
        """
        assert self.is_qwen36, "get_qwen36_rm_rot_idxs is only valid when is_qwen36=True"
        position_idxs_padded = torch.full((32, 32), int(cur_pos), dtype=torch.int32)
        rot_idxs = ttnn.from_torch(
            position_idxs_padded,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=None if on_host else self.device,
            memory_config=None if on_host else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.is_mesh_device else None,
        )
        return rot_idxs

    def get_qwen36_rm_rot_mats(self, rot_idxs):
        """Gather cos/sin via on-device ttnn.embedding (qwen3.6 partial RoPE).

        Pure device op — safe inside trace capture. Mirrors v1
        ``models/demos/qwen3_6_galaxy/tt/llama_rope.py:get_rm_rot_mats``.

        Parameters
        ----------
        rot_idxs : ttnn.Tensor
            ``[32, 32]`` uint32 device tensor (from ``get_qwen36_rm_rot_idxs``).

        Returns
        -------
        (cos, sin) : each ``[1, 1, 1, rope_dim]`` device tensors, replicated.
        """
        assert self.is_qwen36, "get_qwen36_rm_rot_mats is only valid when is_qwen36=True"
        rd = self.rope_dim  # 64
        cos_padded = ttnn.embedding(
            rot_idxs,
            self.cos_matrix_2d_q36,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [32, 32, rope_dim]
        sin_padded = ttnn.embedding(
            rot_idxs,
            self.sin_matrix_2d_q36,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.slice(cos_padded, [0, 0, 0], [1, 1, rd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin = ttnn.slice(sin_padded, [0, 0, 0], [1, 1, rd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos_padded.deallocate(True)
        sin_padded.deallocate(True)
        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, 1, rope_dim]
        sin = ttnn.unsqueeze_to_4D(sin)
        return cos, sin
