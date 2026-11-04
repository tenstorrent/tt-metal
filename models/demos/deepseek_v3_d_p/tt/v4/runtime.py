# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The prefill runtime for DeepSeek-V4.

``TtPrefillRuntime`` driving ``TtV4Transformer``, with what V4's attention state imposes:

* the state lives in the model, not in an engine-owned KV cache, one per user slot, so the runtime
  resets a slot's state at the head of each request in that slot and leaves the other slots alone;
* the state advances on host counters, which a trace replay would not update, so the runtime is
  eager-only;
* the pipeline boundary carries the ``hc_mult`` residual streams, one bf16 D2D plane each, where other
  models send one hidden-state plane; the model itself keeps them packed in fp32.

What migrates is the model's compressed caches and CSA index keys (``tt/v4/kv_caches.py``); the per-slot
carries do not.
"""

from __future__ import annotations

import inspect

import ttnn
from models.demos.common.prefill.runners.migration import allgather_kv_stage_layouts
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime
from models.demos.deepseek_v3_d_p.tt.v4.kv_caches import build_v4_kv_chunk_table
from models.demos.deepseek_v3_d_p.tt.v4.transformer import TtV4Transformer


def streams_to_planes(streams: ttnn.Tensor, hc_mult: int) -> ttnn.Tensor:
    """Packed residual streams ``[1, 1, S/sp, hc_mult * hidden/tp]`` fp32 -> the D2D activation
    ``[1, hc_mult, S/sp, hidden/tp]`` bf16, one plane per stream. Each chip's columns already hold its hidden
    slice of every stream, stream-major, so this is a per-chip reshape; the cast to bf16 is the boundary's
    only loss."""
    _, _, rows, width = streams.shape
    split = ttnn.reshape(streams, (1, rows, hc_mult, width // hc_mult))
    planes = ttnn.permute(split, (0, 2, 1, 3))
    ttnn.deallocate(split)
    out = ttnn.typecast(planes, ttnn.bfloat16)
    ttnn.deallocate(planes)
    return out


def planes_to_streams(planes: ttnn.Tensor) -> ttnn.Tensor:
    """The inverse of ``streams_to_planes``, back to fp32 for the model."""
    _, hc_mult, rows, width = planes.shape
    wide = ttnn.typecast(planes, ttnn.float32)
    split = ttnn.permute(wide, (0, 2, 1, 3))
    ttnn.deallocate(wide)
    out = ttnn.reshape(split, (1, 1, rows, hc_mult * width))
    ttnn.deallocate(split)
    return out


class TtV4Runtime(TtPrefillRuntime):
    MODEL_CLS = TtV4Transformer

    def _build_model(self, state_dict: dict) -> None:
        # The shared build starts an MTP predictor before MODEL_CLS runs, so reject here.
        if self.config.mtp_levels:
            raise ValueError(f"DeepSeek-V4 prefill has no MTP predictor; got mtp_levels={self.config.mtp_levels}")
        if self.config.use_trace:
            raise ValueError("DeepSeek-V4 prefill is eager-only: its attention state advances on host counters")
        if self.config.dflash_enabled:
            raise ValueError("DeepSeek-V4 prefill has no DFlash drafter")
        super()._build_model(state_dict)

    @property
    def activation_planes(self) -> int:
        """The residual streams cross a pipeline boundary as one D2D plane each (``streams_to_planes``)."""
        return self.hf_config.hc_mult

    def prefill_chunk(self, *args, **kwargs):
        """Reset the slot's attention state at the head of a request, then defer to the shared runtime,
        converting the pipeline handoff between the D2D planes and the model's packed streams."""
        bound = inspect.signature(TtPrefillRuntime.prefill_chunk).bind_partial(self, *args, **kwargs)
        if bound.arguments.get("actual_start") == 0:
            # None is the traced path's "the slot is on device", which an eager-only runtime never sees;
            # forward reads it as slot 0, so the reset does too.
            slot_id = bound.arguments.get("slot_id")
            self.model.reset_streams(0 if slot_id is None else slot_id)
        if not self.config.is_first_rank:
            planes = bound.arguments["input_tensor"]
            bound.arguments["input_tensor"] = planes_to_streams(planes)
            ttnn.deallocate(planes)
        out = super().prefill_chunk(*bound.args[1:], **bound.kwargs)
        if self.config.is_last_rank:
            return out
        handoff = streams_to_planes(out, self.hf_config.hc_mult)
        ttnn.deallocate(out)
        return handoff

    def kv_migration_stages(self, kv_caches, first_layer_idx=None, num_my_layers=None):
        """The model's own caches, three stages in table-config order (``tt/v4/kv_caches.py``). The engine's
        ``kv_caches`` is the adapter's empty handle: the model allocates what migrates."""
        return self.model.kv_caches.migration_stages(self.hf_config)

    def build_kv_chunk_table(self, kv_caches, path, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        """The three-config table over the model's caches; ``stage_layouts`` is gathered here when the
        caller (a single-rank test) passes none."""
        if stage_layouts is None:
            stages = self.kv_migration_stages(kv_caches)
            stage_layouts = allgather_kv_stage_layouts(self.mesh_device, stages, self.config.mesh_shape)
        return build_v4_kv_chunk_table(
            self.hf_config,
            stage_layouts=stage_layouts,
            mesh_shape=self.config.mesh_shape,
            sp_axis=self.config.sp_axis,
            num_users=self.config.num_users,
            max_seq_len=self.config.max_seq_len,
            chunk=self.config.chunk_size,
            path=path,
        )
