# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RoPE setup for Qwen3.5-9B Gated Attention layers.

Qwen3.5 uses partial rotary embeddings: only 25% of the head dimensions
(64 out of 256) receive rotary position encoding. The remaining 192 dimensions
pass through unchanged. The gated attention TTNN op handles the partial
application internally — we just need to generate cos/sin for the rotary
portion (head_dim=64).
"""
import torch

import ttnn


def compute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10_000_000.0):
    """Compute RoPE frequency tensors (cos, sin) for given head_dim.

    Args:
        head_dim: Dimension of the rotary portion (64 for Qwen3.5).
        max_seq_len: Maximum sequence length to precompute.
        theta: RoPE base frequency.

    Returns:
        cos: torch.Tensor [max_seq_len, head_dim]
        sin: torch.Tensor [max_seq_len, head_dim]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [max_seq_len, head_dim // 2]
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)  # [max_seq_len, head_dim]
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)  # [max_seq_len, head_dim]
    return cos, sin


class Qwen36RoPESetup:
    """Precomputes and stores RoPE cos/sin tensors for Qwen3.5.

    Usage:
        rope = Qwen36RoPESetup(device, args)
        cos, sin = rope.get_rot_mats(position_ids)
    """

    def __init__(self, device, args):
        self.device = device
        self.head_dim = args.rope_head_dim  # 64
        self.max_seq_len = args.max_seq_len
        self.theta = args.rope_theta

        self.cos_cpu, self.sin_cpu = compute_rope_freqs(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            theta=args.rope_theta,
        )

        # --- M-RoPE (multimodal rotary) per-request state -------------------------------------
        # build_request_rope() stages a per-SEQUENCE cos/sin table for a multimodal prompt (the
        # token's cos/sin depends on its 3D (t,h,w) position, not a single index, so the absolute
        # cos_cpu/cos_device tables below cannot represent the image region). prefill_cos_sin_torch
        # / get_prefill_rot_mats slice that table by sequence position. When it is None (text-only)
        # the prefill helpers fall back to ordinary 1D RoPE — byte-identical to the pre-M-RoPE path.
        # Decode stays on the absolute tables, offset by rope_delta (post-image text has t==h==w).
        self.mrope_section = list(args.mrope_section)
        self.attention_scaling = args.rope_attention_scaling
        self.spatial_merge_size = args.spatial_merge_size
        self.image_token_id = args.image_token_id
        self.video_token_id = args.video_token_id
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self._req_cos = None  # [S, head_dim] bf16, sequence-indexed M-RoPE cos (None => text)
        self._req_sin = None
        self.rope_delta = 0  # mrope_position_delta: decode rope_pos = kv_pos + rope_delta

        # Pre-compute full RoPE table on device for fast decode lookups
        # Shape: [1, max_seq_len, head_dim] on device
        # mesh_mapper replicates to all devices; on a 1-device mesh this is a no-op.
        self.cos_device = ttnn.from_torch(
            self.cos_cpu.unsqueeze(0),  # [1, max_seq_len, head_dim]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        self.sin_device = ttnn.from_torch(
            self.sin_cpu.unsqueeze(0),  # [1, max_seq_len, head_dim]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    def get_rot_mats(self, position_ids: torch.Tensor):
        """Get cos/sin matrices for given positions.

        Args:
            position_ids: torch.Tensor [B, T] or [T] — position indices.

        Returns:
            cos_ttnn: ttnn.Tensor [B, T, head_dim] on device
            sin_ttnn: ttnn.Tensor [B, T, head_dim] on device
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        B, T = position_ids.shape

        # Fast path for single-position decode: slice from pre-computed device table
        if T == 1 and B == 1:
            pos = position_ids.item()
            cos = self.cos_device[:, pos : pos + 1, :]
            sin = self.sin_device[:, pos : pos + 1, :]
            return cos, sin

        # General path for prefill (variable positions)
        flat_pos = position_ids.reshape(-1)
        cos = self.cos_cpu[flat_pos].reshape(B, T, self.head_dim)
        sin = self.sin_cpu[flat_pos].reshape(B, T, self.head_dim)

        cos_ttnn = ttnn.from_torch(
            cos,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        sin_ttnn = ttnn.from_torch(
            sin,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        return cos_ttnn, sin_ttnn

    def get_cos_sin_host(self, pos):
        """Return cos/sin at position as host ttnn tensors for copy_host_to_device_tensor.

        Returns tensors on HOST (no device= arg) for fast DMA to pre-allocated device buffers.
        Shape: [1, 1, rope_head_dim] — must match _trace_cos/_trace_sin device buffer shapes.
        Layout: TILE_LAYOUT — must match device buffer layout for copy compatibility.

        `pos` is the ROPE position (= KV position + rope_delta for a multimodal request); the
        caller is responsible for the offset so decode reads the absolute 1D table correctly.
        """
        cos = self.cos_cpu[pos : pos + 1].unsqueeze(0).contiguous()  # [1, 1, 64]
        sin = self.sin_cpu[pos : pos + 1].unsqueeze(0).contiguous()  # [1, 1, 64]
        cos_host = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        sin_host = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return cos_host, sin_host

    # -------------------------------------------------------------------------
    # M-RoPE (multimodal) — per-request 3D position handling
    # -------------------------------------------------------------------------
    def build_request_rope(self, input_ids, image_grid_thw=None, video_grid_thw=None):
        """Stage the per-request M-RoPE cos/sin table + rope_delta for the upcoming prefill.

        Text-only (no grids) CLEARS the table (prefill falls back to 1D RoPE, decode delta 0 —
        unchanged). Multimodal derives the 3D position ids on host from input_ids (image/video
        placeholders located via the token ids, so the caller need not pass mm_token_type_ids)
        and the grid(s), builds a SEQUENCE-indexed cos/sin table via interleaved M-RoPE, and
        stores mrope_position_delta. Returns rope_delta (int)."""
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import get_rope_index, get_rot_mats

        if image_grid_thw is None and video_grid_thw is None:
            self._req_cos = None
            self._req_sin = None
            self.rope_delta = 0
            return 0

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(torch.long)
        # mm_token_type_ids: text=0, image=1, video=2 — the processor's convention, here derived
        # from the placeholder token ids (the only multimodal signal the model needs from input_ids).
        mm = torch.zeros_like(input_ids)
        if self.image_token_id is not None:
            mm[input_ids == int(self.image_token_id)] = 1
        if self.video_token_id is not None:
            mm[input_ids == int(self.video_token_id)] = 2

        position_ids, deltas = get_rope_index(
            input_ids,
            mm,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            spatial_merge_size=self.spatial_merge_size,
        )  # position_ids [3, B, S], deltas [B, 1]
        cos, sin = get_rot_mats(self.inv_freq, position_ids, self.mrope_section, self.attention_scaling)
        # get_rot_mats returns [B, S, head_dim]; B==1 here (single-sequence prefill).
        self._req_cos = cos[0].to(torch.bfloat16)  # [S, head_dim]
        self._req_sin = sin[0].to(torch.bfloat16)
        self.rope_delta = int(deltas[0, 0].item())
        return self.rope_delta

    def _extend_req_table(self, length):
        """Grow the per-request M-RoPE table to >= `length` positions with text-continuation rows
        (post-prompt positions have t==h==w, advancing as rope_pos = seq_idx + rope_delta). Used so
        the masked-bucket padding past the real prompt still has cos/sin."""
        cur = self._req_cos.shape[0]
        if length <= cur:
            return
        pos = torch.arange(cur, length, dtype=torch.float32) + self.rope_delta
        emb = torch.cat([torch.outer(pos, self.inv_freq)] * 2, dim=-1)
        self._req_cos = torch.cat([self._req_cos, emb.cos().to(torch.bfloat16)], dim=0)
        self._req_sin = torch.cat([self._req_sin, emb.sin().to(torch.bfloat16)], dim=0)

    def prefill_cos_sin_torch(self, start, length):
        """Torch bf16 cos/sin [length, head_dim] for SEQUENCE positions [start, start+length).

        Uses the per-request M-RoPE table when staged (build_request_rope); otherwise ordinary 1D
        RoPE at absolute positions [start, start+length) — byte-identical to the pre-M-RoPE path."""
        if self._req_cos is not None:
            end = start + length
            if end > self._req_cos.shape[0]:
                self._extend_req_table(end)
            return self._req_cos[start:end], self._req_sin[start:end]
        t = torch.arange(start, start + length, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, self.inv_freq)] * 2, dim=-1)
        return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)

    def get_prefill_rot_mats(self, start, length):
        """ttnn cos/sin [1, length, head_dim] (replicated) for SEQUENCE positions [start, start+length),
        M-RoPE-aware. Drop-in for the prefill sites that previously called get_rot_mats(arange(...))."""
        cos_t, sin_t = self.prefill_cos_sin_torch(start, length)
        cos = ttnn.from_torch(
            cos_t.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        sin = ttnn.from_torch(
            sin_t.unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        return cos, sin
