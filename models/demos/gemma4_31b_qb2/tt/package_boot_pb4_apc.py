# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package's `single-user` class for update 5: the S1 prefix-resume single-user class
(generator_resident_spec_vllm_pb3_apc, on PB3; bench/resident/s1prefix/S1_DESIGN.md §5 + addenda) on update 4's packaged
PB4 class (package_boot_pb4: PB3Packaged's package_boot.ensure_resident_cache before construction, plus the 64-pass
continuation pre-warm keyed to resident_m3_2). The S1 class sits on PB3, so the shipped profile composes PB4's warm here
(the critic's host-build review R2).

Cooperative inheritance, S1 class first (bench/packaging/update5/UPDATE5_PLAN_v3.md §1.1; LEAD_DECISIONS_PKG P144.5):

    Gemma4ResidentSpecPB4APCPackaged
      -> Gemma4ResidentSpecPB3APCForCausalLM
      -> Gemma4ResidentSpecPB4Packaged -> Gemma4ResidentSpecPB3Packaged -> Gemma4ResidentSpecPB3ForCausalLM -> ...

Everything is inherited; nothing is copied or overridden here:
- from the S1 class: model_capabilities (supports_prefix_caching, prefix_resume with GEMMA4_S1_TAIL_SLOTS slots,
  default 2, no request ids), get_kv_cache_spec (the hybrid spec without the refusal), allocate_pools (the tail pool),
  prefill_forward (admission past 0, restore and store orders), release_request, and the `live` flag;
- from the packaged PB4 / PB3 classes: initialize_vllm_model (ensure_resident_cache, then PB3's construction),
  ENGINE_PACKAGE, warmup_model_prefill and _cont_warm (the marker-gated pre-warm, refused once a PB3 trace is live).
  The packaged methods' zero-argument super() continues to PB3 in this MRO, so their code runs unchanged.
The plugin allocates the KV cache and pools (allocate_pools, so the tail pool) before warmup_model. The pre-warm calls
the PB generator directly, never the adapter's prefill_forward, so no tail order is involved and the tail pool stays
empty."""

from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm_pb3_apc import Gemma4ResidentSpecPB3APCForCausalLM
from models.demos.gemma4_31b_qb2.tt.package_boot_pb4 import Gemma4ResidentSpecPB4Packaged


class Gemma4ResidentSpecPB4APCPackaged(Gemma4ResidentSpecPB3APCForCausalLM, Gemma4ResidentSpecPB4Packaged):
    """The S1 single-user class with update 4's package boot and PB4 continuation pre-warm, by inheritance only."""
