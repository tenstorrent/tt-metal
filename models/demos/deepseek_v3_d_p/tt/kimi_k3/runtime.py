# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The prefill runtime for Kimi-K3.

`TtPrefillRuntime` is reused whole. Only two things differ, and both are consequences of Kimi-K3
carrying recurrent state that no other model in this package has:

* the transformer it drives is `TtKimiK3Transformer` (the `MODEL_CLS` seam), because only 24 of
  Kimi-K3's 93 layers write a KV slab and its residual is block-structured, so it cannot reuse the
  shared block; and
* the KDA carries must be zeroed at the head of a request. A carry summarises the whole prefix
  behind it, so leaving the previous request's carry in place is not a small error — it conditions
  every token of the new one on text it never saw.
"""

from __future__ import annotations

import inspect

from loguru import logger

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime


class TtKimiK3Runtime(TtPrefillRuntime):
    MODEL_CLS = TtKimiK3Transformer

    @property
    def activation_planes(self) -> int:
        """The live stream plus every AttnRes snapshot sealed before this rank's first layer.

        A read scores the live sum against the whole sealed set, and the snapshots for upstream
        layers exist only on the rank that produced them, so they arrive in the payload. Mirrors
        `KimiK3Adapter.pipeline_activation_planes`, which sizes the socket — the two are read at the
        same boundary and disagreeing shows up as a rendezvous failure, not a wrong answer.
        """
        return 1 + self.config.first_layer_idx // KimiK3Config.ATTN_RES_BLOCK_SIZE

    def prefill_chunk(self, *args, **kwargs):
        """Reset the KDA carries at the start of a request, then defer to the shared runtime.

        `actual_start == 0` is the head of a request, and it is the only safe moment to zero: the
        reset is a device copy from a held zero tensor rather than a reallocation, so it must not
        land inside a captured region, which would re-zero on every replay and destroy the carry
        the trace exists to advance.
        """
        # Bound from the real signature rather than by counting positions: `actual_start` is the
        # fourth positional parameter, and reading the third instead would take `slot_id` — which is
        # 0 on every chunk of a single-user run, so the carries would be zeroed at every chunk
        # boundary and multi-chunk prefill would silently lose its recurrence.
        bound = inspect.signature(TtPrefillRuntime.prefill_chunk).bind_partial(self, *args, **kwargs)
        actual_start = bound.arguments.get("actual_start")
        if actual_start == 0:
            states = getattr(self.model, "kda_states", None)
            if states is not None:
                for slot in range(states.num_slots):
                    states.reset(slot)
                logger.debug(f"KDA carries reset for {states.num_slots} slot(s) at request head")
        return super().prefill_chunk(*args, **kwargs)
