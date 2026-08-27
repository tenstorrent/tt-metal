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
from models.common.utility_functions import is_blackhole
from models.demos.blackhole.qwen36.tt.attention.rope_tp import to_full_width_rot_mats


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
        self.head_dim = args.rope_head_dim  # 64 -- the FREQUENCY dim (drives inv_freq), always
        self.max_seq_len = args.max_seq_len
        self.theta = args.rope_theta
        # Permuted-head_dim full-width RoPE (attention/rope_tp.py's rope_channel_perm): every
        # cos/sin this class hands out becomes head_dim wide in permuted channel order instead of
        # rope_head_dim wide in HF order. The frequencies are unchanged -- the extra channels are
        # cos=1/sin=0 identity slots -- so only the WIDTH of what leaves here differs. Consumers use
        # self.rope_width rather than self.head_dim wherever they mean "width of a cos/sin row".
        self.full_head_dim = args.head_dim if getattr(args, "rope_permuted_enabled", False) else None
        self.rope_width = self.full_head_dim or self.head_dim

        self.cos_cpu, self.sin_cpu = compute_rope_freqs(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            theta=args.rope_theta,
        )
        if self.full_head_dim:
            self.cos_cpu, self.sin_cpu = to_full_width_rot_mats(
                self.cos_cpu, self.sin_cpu, self.full_head_dim, self.head_dim, self.device
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
        cos = self.cos_cpu[flat_pos].reshape(B, T, self.rope_width)
        sin = self.sin_cpu[flat_pos].reshape(B, T, self.rope_width)

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
        Shape: [1, 1, rope_width] — must match _trace_cos/_trace_sin device buffer shapes.
        (rope_width is rope_head_dim, or head_dim under permuted full-width RoPE; the only caller is
        the Blackhole branch of prepare_decode_inputs_host, where the permutation is always off.)
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
        # get_rot_mats returns [B, S, rope_head_dim]; B==1 here (single-sequence prefill).
        if self.full_head_dim:
            cos, sin = to_full_width_rot_mats(cos, sin, self.full_head_dim, self.head_dim, self.device)
        self._req_cos = cos[0].to(torch.bfloat16)  # [S, rope_width]
        self._req_sin = sin[0].to(torch.bfloat16)
        self.rope_delta = int(deltas[0, 0].item())
        return self.rope_delta

    def _extend_req_table(self, length):
        """Grow the per-request M-RoPE table to >= `length` positions with text-continuation rows
        (post-prompt positions have t==h==w, advancing as rope_pos = seq_idx + rope_delta). Used so
        the masked-bucket padding past the real prompt still has cos/sin.

        Grows GEOMETRICALLY (at least double) rather than to exactly `length`. Every call
        reallocates and copies the whole table, and the callers extend by a chunk at a time, so
        exact growth makes a long generation quadratic in host memcpy; doubling amortizes it to
        linear. Capped at max_seq_len, past which no position can be requested anyway.
        """
        cur = self._req_cos.shape[0]
        if length <= cur:
            return
        target = min(max(length, 2 * cur), max(length, self.max_seq_len))
        pos = torch.arange(cur, target, dtype=torch.float32) + self.rope_delta
        emb = torch.cat([torch.outer(pos, self.inv_freq)] * 2, dim=-1)
        new_cos, new_sin = emb.cos(), emb.sin()
        if self.full_head_dim:
            new_cos, new_sin = to_full_width_rot_mats(new_cos, new_sin, self.full_head_dim, self.head_dim, self.device)
        self._req_cos = torch.cat([self._req_cos, new_cos.to(torch.bfloat16)], dim=0)
        self._req_sin = torch.cat([self._req_sin, new_sin.to(torch.bfloat16)], dim=0)

    @property
    def mrope_staged(self):
        """True when build_request_rope staged a per-sequence M-RoPE table (multimodal request),
        so the absolute 1D tables cannot serve this prefill."""
        return self._req_cos is not None

    def prefill_cos_sin_torch(self, start, length):
        """Torch bf16 cos/sin [length, rope_width] for SEQUENCE positions [start, start+length).

        Uses the per-request M-RoPE table when staged (build_request_rope); otherwise ordinary 1D
        RoPE at absolute positions [start, start+length) — byte-identical to the pre-M-RoPE path."""
        if self._req_cos is not None:
            end = start + length
            if end > self._req_cos.shape[0]:
                self._extend_req_table(end)
            return self._req_cos[start:end], self._req_sin[start:end]
        t = torch.arange(start, start + length, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, self.inv_freq)] * 2, dim=-1)
        cos_t, sin_t = emb.cos(), emb.sin()
        if self.full_head_dim:
            cos_t, sin_t = to_full_width_rot_mats(cos_t, sin_t, self.full_head_dim, self.head_dim, self.device)
        return cos_t.to(torch.bfloat16), sin_t.to(torch.bfloat16)

    def ensure_prefill_tables(self, n_rows):
        """Grow the resident cos/sin tables to n_rows NOW, outside any traced region.

        _rope_dev_tables grows by from_torch -- a host write. A caller that slices per chunk inside
        a trace-replay loop must warm past its last position first, or that write lands mid-loop.
        Mirrors how the decode rope path sizes to max_seq_len up front. Same args as the slice
        below, so both hit one cache entry. Blackhole never builds them (get_prefill_rot_mats keeps
        its host path there). Not gated on a staged M-RoPE table: these are the absolute tables, so
        skipping the warm-up would only leave the growth to land inside a later capture.
        """
        if is_blackhole():
            return
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import _rope_dev_tables

        _rope_dev_tables(self.device, self.head_dim, int(n_rows), self.theta, full_head_dim=self.full_head_dim)

    def get_prefill_rot_mats(self, start, length):
        """ttnn cos/sin [1, length, head_dim] (replicated) for SEQUENCE positions [start, start+length).

        Text-only: the positions are the contiguous range [start, start+length), a slice of the
        cos/sin tables already resident on device, so the rotation never touches host trig. The
        tables are ROW_MAJOR, so the slice has no tile-alignment constraint (any start/length
        works) and they grow on demand.

        The M-RoPE branch below is a DIFFERENT ALGORITHM, not a fallback for this one: a
        multimodal token's rotation comes from its 3D (t,h,w) position, which an absolute 1D table
        structurally cannot represent, so it uses the per-request table staged by
        build_request_rope(). It is unreachable for text-only inference.
        """
        if self._req_cos is None and not is_blackhole():
            from models.demos.blackhole.qwen36.tt.attention.rope_tp import _rope_dev_tables

            tbl_cos, tbl_sin = _rope_dev_tables(
                self.device, self.head_dim, start + length, self.theta, full_head_dim=self.full_head_dim
            )

            def _slice(tbl):
                r = ttnn.slice(tbl, [start, 0], [start + length, self.rope_width])  # ROW_MAJOR
                r = ttnn.reshape(r, (1, length, self.rope_width))  # metadata-only while ROW_MAJOR
                return ttnn.to_layout(r, ttnn.TILE_LAYOUT)

            return _slice(tbl_cos), _slice(tbl_sin)

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
