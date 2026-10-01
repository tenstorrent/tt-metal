# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The prefill runtime for DeepSeek-V4.

``TtPrefillRuntime`` driving ``TtV4Transformer``, with what V4's attention state imposes:

* the state lives in the model, not in an engine-owned KV cache, and is one user's, so the runtime
  runs one slot and resets the state at the head of every request;
* the state advances on host counters, which a trace replay would not update, so the runtime is
  eager-only;
* a non-first rank receives the packed fp32 residual streams, not a bf16 hidden state.

There is no KV migration: nothing reads the attention state out of a V4 rank yet.
"""

from __future__ import annotations

import inspect

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime
from models.demos.deepseek_v3_d_p.tt.v4.transformer import TtV4Transformer


class TtV4Runtime(TtPrefillRuntime):
    MODEL_CLS = TtV4Transformer

    def _build_model(self, state_dict: dict) -> None:
        # The shared build starts an MTP predictor before MODEL_CLS runs, so reject here.
        if self.config.mtp_levels:
            raise ValueError(f"DeepSeek-V4 prefill has no MTP predictor; got mtp_levels={self.config.mtp_levels}")
        if self.config.use_trace:
            raise ValueError("DeepSeek-V4 prefill is eager-only: its attention state advances on host counters")
        if self.config.num_users != 1:
            raise ValueError(f"DeepSeek-V4 attention state is single-user; got num_users={self.config.num_users}")
        if self.config.dflash_enabled:
            raise ValueError("DeepSeek-V4 prefill has no DFlash drafter")
        super()._build_model(state_dict)

    def prefill_chunk(self, *args, **kwargs):
        """Reset the attention state at the head of a request, then defer to the shared runtime."""
        bound = inspect.signature(TtPrefillRuntime.prefill_chunk).bind_partial(self, *args, **kwargs)
        if bound.arguments.get("actual_start") == 0:
            self.model.reset_streams()
        return super().prefill_chunk(*args, **kwargs)

    def make_placeholder_activation(self, dflash_packed: bool = True) -> ttnn.Tensor:
        """A zero ``[1, 1, chunk/sp, hc_mult * hidden/tp]`` fp32 activation, the shape a non-first rank receives."""
        rows = self.config.chunk_size // self.config.sp_factor
        width = self.hf_config.hc_mult * self.hf_config.hidden_size // self.config.tp_factor
        return ttnn.from_torch(
            torch.zeros(1, 1, rows, width, dtype=torch.float32),
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def kv_migration_stages(self, kv_caches, first_layer_idx=None, num_my_layers=None):
        return []
