# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package's `batch` class for update 5: the S1 prefix-resume batch class v2
(generator_resident_multi_s3_2_pb_ring_apc_v2_vllm, bench/resident/s1prefix/S1_DESIGN.md + addenda, S1_B4_ROOT_CAUSE.md)
on update 4's packaged ring class (package_boot_s3pb_ring: package_boot.ensure_resident_cache before construction, and
the 640-pass continuation pre-warm keyed to resident_m5_11).

Cooperative inheritance, S1 class first (bench/packaging/update5/UPDATE5_PLAN_v3.md §1.1; LEAD_DECISIONS_PKG P144.5):

    Gemma4ResidentMultiS32PBRingAPCPackaged
      -> Gemma4ResidentMultiS32PBRingAPCV2ForCausalLM -> Gemma4ResidentMultiS32PBRingAPCForCausalLM
      -> Gemma4ResidentMultiS32PBRingPackaged -> Gemma4ResidentMultiS32PBRingForCausalLM -> ...

Everything is inherited; nothing is copied or overridden here:
- from the S1 classes: model_capabilities (supports_prefix_caching, prefix_resume with GEMMA4_S1_TAIL_SLOTS slots,
  default 4), get_kv_cache_spec, allocate_pools (the tail pool), prefill_forward / decode_forward (request ids, tail
  orders, the rebind and admission order), release_request;
- from the packaged ring class: initialize_vllm_model (ensure_resident_cache, then the ring construction),
  ENGINE_PACKAGE, warmup_model_prefill and _cont_warm (the marker-gated pre-warm). The packaged methods' zero-argument
  super() continues to the ring class in this MRO, so their code runs unchanged.
The plugin allocates the KV cache and pools (allocate_pools, so the tail pool) before warmup_model. The pre-warm calls
the PB generator directly, never the adapter's prefill_forward, so no request id or tail order is involved and the tail
pool stays empty; the bucket warmups reach the adapter without request ids and run the ring class unchanged."""

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_ring_apc_v2_vllm import (
    Gemma4ResidentMultiS32PBRingAPCV2ForCausalLM,
)
from models.demos.gemma4_31b_qb2.tt.package_boot_s3pb_ring import Gemma4ResidentMultiS32PBRingPackaged


class Gemma4ResidentMultiS32PBRingAPCPackaged(Gemma4ResidentMultiS32PBRingAPCV2ForCausalLM,
                                              Gemma4ResidentMultiS32PBRingPackaged):
    """The S1 batch class v2 with update 4's package boot and continuation pre-warm, by inheritance only."""
