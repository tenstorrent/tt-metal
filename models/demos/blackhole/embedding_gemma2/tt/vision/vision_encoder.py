# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionEncoder: host-side orchestration around VisionModel (input prep, upload, trace, readback).

Host work lives here and only here. From the processor outputs (`pixel_values` [B, N, 768],
`image_position_ids` / `video_position_ids` [B, N, 2], -1 = padding; video frames are batch rows)
`prepare_inputs` builds, per forward:
    pixel_values  zero padded to the patch bucket S                        [B,1,S,768] fp32
    pos_x, pos_y  position-table rows; padding -> the appended zero row     [B,S] uint32
    cos, sin      axial RoPE tables of the patch positions                 [B,1,S,64] bf16
    keep          1 for valid patches, 0 for padding                       [B,1,1,S] bf16
    pool          one-hot 3x3 block-membership matrix                      [B,1,S_out,S] bf16
    windows       SDPA cu_window_seqlens of the valid prefixes (E5b)       [2B+1] uint32
and the per-image soft-token counts used to strip the padded pooled rows after readback.
The device forward (`VisionModel.forward`) issues no host ops.

Trace policy (same as TextEncoder): one trace per (batch, bucket), all captured by one
`capture_all(shapes)` call that allocates every persistent input and compiles every shape before
the first capture. `encode(..., traced=True)` refreshes the persistent inputs with
`copy_host_to_device_tensor` and replays. Any image (any aspect ratio / padding amount) with the
same batch and bucket reuses the trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import default_weight_cache_dir
from models.demos.blackhole.embedding_gemma2.tt.vision.attention import padding_windows, use_windowed_attention
from models.demos.blackhole.embedding_gemma2.tt.vision.config import VisionConfig, patch_bucket_for
from models.demos.blackhole.embedding_gemma2.tt.vision.pooler import pooling_matrix
from models.demos.blackhole.embedding_gemma2.tt.vision.rope import axial_cos_sin
from models.demos.blackhole.embedding_gemma2.tt.vision.vision_model import VisionDeviceInputs, VisionModel
from models.demos.blackhole.embedding_gemma2.tt.vision.weights import load_vision_config_dicts, load_vision_state_dict
from models.demos.blackhole.embedding_gemma2.tt.weights import resolve_snapshot


@dataclass
class VisionHostInputs:
    tensors: VisionDeviceInputs  # host (not on device) ttnn tensors
    batch: int
    seq_len: int  # physical patch bucket S
    num_patches: int  # logical N from the processor
    num_valid: list[int]  # valid patches per row
    num_soft_tokens: list[int]  # soft tokens per row (= num_valid / 9 on processor output)


@dataclass
class _Trace:
    trace_id: int
    inputs: VisionDeviceInputs
    soft: ttnn.Tensor
    pooled: ttnn.Tensor


class VisionEncoder(LightweightModule):
    def __init__(self, model: VisionModel, device):
        super().__init__()
        self.model = model
        self.device = device
        self.config: VisionConfig = model.config
        self._traces: dict[tuple[int, int], _Trace] = {}
        self._compiled: set[tuple[int, int]] = set()

    @classmethod
    def from_pretrained(
        cls,
        device,
        model_id_or_path: str = "google/embeddinggemma-2",
        *,
        cache_dir: Path | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
        precision=None,
    ) -> "VisionEncoder":
        """`precision`: None loads the selected precision config (tt/precision.py, its `vision`
        section); a path, dict or PrecisionConfig otherwise."""
        snapshot = resolve_snapshot(model_id_or_path)
        vcfg, text_hidden = load_vision_config_dicts(snapshot)
        config = VisionConfig.from_hf(vcfg, text_hidden)
        if state_dict is None:
            state_dict = load_vision_state_dict(snapshot)
        cache_dir = Path(cache_dir) if cache_dir is not None else default_weight_cache_dir() / snapshot.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        model = VisionModel.from_state_dict(state_dict, config, device, cache_dir=cache_dir, precision=precision)
        return cls(model, device)

    # ---------------------------------------------------------------- host side
    def prepare_inputs(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> VisionHostInputs:
        """Processor outputs -> host ttnn tensors for one forward (no device work)."""
        cfg = self.config
        if pixel_values.dim() != 3 or pixel_values.shape[-1] != cfg.patch_dim:
            raise ValueError(f"pixel_values must be [B, N, {cfg.patch_dim}], got {tuple(pixel_values.shape)}")
        B, N, _ = pixel_values.shape
        if tuple(position_ids.shape) != (B, N, 2):
            raise ValueError(f"position_ids must be [{B}, {N}, 2], got {tuple(position_ids.shape)}")
        S = patch_bucket_for(N)
        S_out = cfg.pooled_rows(S)
        pos = torch.full((B, S, 2), -1, dtype=torch.int64)
        pos[:, :N] = position_ids.long()
        valid = (pos != -1).all(dim=-1)  # [B, S]
        if ((pos >= cfg.position_embedding_size) & valid[..., None]).any():
            raise ValueError(f"patch position >= position_embedding_size {cfg.position_embedding_size}")

        px = torch.zeros((B, 1, S, cfg.patch_dim), dtype=torch.float32)
        px[:, 0, :N] = pixel_values.float()
        table_idx = torch.where(valid[..., None], pos, torch.full_like(pos, cfg.position_embedding_size))
        cos, sin = axial_cos_sin(pos, cfg.rope_theta, cfg.head_dim)
        keep = valid.float().reshape(B, 1, 1, S)
        pool, n_out = pooling_matrix(position_ids.long(), cfg.pooling_kernel_size, S_out, S)
        num_valid = valid.sum(-1).tolist()
        prefix = all(bool(valid[b, :n].all()) for b, n in enumerate(num_valid))
        if use_windowed_attention(B, S):
            if not prefix:
                raise ValueError("windowed attention (batch >= 8, bucket 1280) needs the valid patches to be a prefix")
            windows = padding_windows(num_valid, S)
        else:
            windows = torch.zeros(2 * B + 1, dtype=torch.int32)  # unused by the masked path

        def tile(t, dt=ttnn.bfloat16):
            return ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT)

        def rm_u32(t):
            return ttnn.from_torch(t.to(torch.int32).contiguous(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        tensors = VisionDeviceInputs(
            pixel_values=tile(px, ttnn.float32),
            pos_x=rm_u32(table_idx[..., 0]),
            pos_y=rm_u32(table_idx[..., 1]),
            cos=tile(cos.reshape(B, 1, S, cfg.head_dim)),
            sin=tile(sin.reshape(B, 1, S, cfg.head_dim)),
            keep=tile(keep),
            pool=tile(pool),
            windows=ttnn.from_torch(windows, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        )
        return VisionHostInputs(
            tensors=tensors,
            batch=B,
            seq_len=S,
            num_patches=N,
            num_valid=num_valid,
            num_soft_tokens=n_out,
        )

    def to_device(self, host: VisionHostInputs) -> VisionDeviceInputs:
        dram = ttnn.DRAM_MEMORY_CONFIG
        return VisionDeviceInputs(*(ttnn.to_device(t, self.device, memory_config=dram) for t in host.tensors.tensors()))

    @staticmethod
    def read_rows(t: ttnn.Tensor, counts: list[int]) -> list[torch.Tensor]:
        """[B,1,S_out,D] device -> per-image [counts[b], D] float32 (padded pooled rows stripped)."""
        x = ttnn.to_torch(t).float()
        x = x.reshape(x.shape[0], x.shape[-2], x.shape[-1])
        return [x[b, :n] for b, n in enumerate(counts)]

    # ---------------------------------------------------------------- eager
    def forward(self, x: VisionDeviceInputs) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Device-only forward on already uploaded inputs (the measured unit)."""
        return self.model(x)

    def encode(
        self,
        pixel_values: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        traced: bool = False,
        return_pooled: bool = False,
    ):
        """Soft tokens per image/frame: list of [n_b, 512] float32 (n_b = valid patches / 9)."""
        host = self.prepare_inputs(pixel_values, position_ids)
        if traced:
            soft, pooled = self.replay(host)
        else:
            key = (host.batch, host.seq_len)
            if self._traces and key not in self._compiled:
                raise RuntimeError(
                    f"eager shape {key} was not compiled before the traces were captured; its new "
                    "program-cache buffers could overlap trace buffers. Include it in capture_all()."
                )
            soft, pooled = self.forward(self.to_device(host))
            self._compiled.add(key)
        out = self.read_rows(soft, host.num_soft_tokens)
        if return_pooled:
            return out, self.read_rows(pooled, host.num_soft_tokens)
        return out

    # ---------------------------------------------------------------- trace
    def _dummy_host(self, batch: int, seq_len: int) -> VisionHostInputs:
        """A valid full-grid input of shape (batch, seq_len) for allocation / compile / capture."""
        k = self.config.pooling_kernel_size
        side_x = k * 10  # a 30-wide grid with a multiple-of-3 row count: 2520 (S 2560) / 1260 (S 1280) patches
        n = seq_len // (side_x * k) * (side_x * k)
        idx = torch.arange(n)
        pos = torch.stack([idx % side_x, idx // side_x], dim=-1).expand(batch, n, 2).clone()
        return self.prepare_inputs(torch.full((batch, n, self.config.patch_dim), 0.5), pos)

    def capture_all(self, shapes) -> None:
        """Capture one trace per (batch, bucket) in `shapes`.

        As in TextEncoder.capture_all: allocate every persistent input and compile every shape with
        one eager pass before the first capture, so no allocation made after a capture can land in
        that trace's buffers. Call once per session; eager runs afterwards are allowed only for
        shapes listed here. Trace outputs are rewritten by their own replay; read them before
        replaying another trace.
        """
        if self._traces:
            raise RuntimeError("capture_all() may run once, before any other trace is captured")
        shapes = sorted(set((int(b), int(s)) for b, s in shapes))
        inputs = {key: self.to_device(self._dummy_host(*key)) for key in shapes}
        for key in shapes:
            soft, pooled = self.forward(inputs[key])
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(soft)
            ttnn.deallocate(pooled)
            self._compiled.add(key)
        from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible

        for key in shapes:
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            soft, pooled = self.forward(inputs[key])
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.device)
            acknowledge_corruptible(soft)
            acknowledge_corruptible(pooled)
            self._traces[key] = _Trace(trace_id, inputs[key], soft, pooled)

    def replay(self, host: VisionHostInputs, *, blocking: bool = False) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        key = (host.batch, host.seq_len)
        if key not in self._traces:
            raise KeyError(f"no trace for (batch, bucket) {key}; captured: {sorted(self._traces)}")
        tr = self._traces[key]
        for src, dst in zip(host.tensors.tensors(), tr.inputs.tensors()):
            ttnn.copy_host_to_device_tensor(src, dst)
        ttnn.execute_trace(self.device, tr.trace_id, cq_id=0, blocking=blocking)
        return tr.soft, tr.pooled

    def execute_captured(self, batch: int, seq_len: int, *, blocking: bool = False):
        """Replay a captured trace on whatever its persistent inputs hold (perf loop helper)."""
        tr = self._traces[(batch, seq_len)]
        ttnn.execute_trace(self.device, tr.trace_id, cq_id=0, blocking=blocking)
        return tr.soft, tr.pooled

    def release_traces(self) -> None:
        for tr in self._traces.values():
            ttnn.release_trace(self.device, tr.trace_id)
        self._traces.clear()
