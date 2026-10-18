# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 prefill in pure ttnn.

Two block implementations live in ``block.py``:

  * ``TtV4Block`` -- main's public DeepSeek-V4 decoder block (``TtMHCWrap`` residual, ``tt/mla`` SWA / HCA attention,
    ``TtPrefillBlock._build_moe``), exported here as the package's public API.
  * ``TtV4PrefillBlock`` -- the DeepSeek-V4-Flash production prefill block (layer-kind schedule, KV geometry, mHC,
    trace islands, transformer, runtime), built on the V4 pieces in this package (``tt/mla/heavily_compressed_attention.py``,
    the V4 MoE gate, ``reference/deepseek_v4``) and driven by the model-agnostic engine in
    ``models/demos/common/prefill``. Plan: tt-blaze ``docs/plans/deepseek_v4_flash_prefill_ttnn_plan.md``.

``TtV4Block`` is resolved lazily (PEP 562): host-only users of this package (``kv_contract``, ``kv_table``,
``layer_kinds``, ``pp_pack``, ``golden``, ``mhc_math`` -- the prefill producer and the CPU tests import them) must not
import ttnn, which opens every chip.
"""

__all__ = ["TtV4Block"]


def __getattr__(name):
    if name == "TtV4Block":
        from models.demos.deepseek_v3_d_p.tt.v4.block import TtV4Block

        return TtV4Block
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
