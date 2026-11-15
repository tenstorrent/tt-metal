# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package's `single-user` class for update 2: Gemma4ResidentSpecPB3ForCausalLM (generator_resident_spec_vllm_pb3:
the paged-bucket prefill, with the drafter-seed pre-warm and the seed-path guard; served re-gate p34a / chs at PB3,
bench/packaging/dram_fix/PB3_REGATE_PLAN.md, audited) that first makes sure the resident weight cache exists, as
package_boot.Gemma4ResidentSpecPackaged does for update 1's class. package_boot.py is unchanged: this module only
reuses its ensure_resident_cache()."""

from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm_pb3 import Gemma4ResidentSpecPB3ForCausalLM
from models.demos.gemma4_31b_qb2.tt.package_boot import ensure_resident_cache


class Gemma4ResidentSpecPB3Packaged(Gemma4ResidentSpecPB3ForCausalLM):
    """Gemma4ResidentSpecPB3ForCausalLM that first makes sure the resident weight cache exists (the package's class)."""

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        ensure_resident_cache()
        return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, **kwargs)
