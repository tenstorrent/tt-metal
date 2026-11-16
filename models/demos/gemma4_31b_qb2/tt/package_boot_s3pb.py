# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package's `batch` class for update 3: Gemma4ResidentMultiS32PBForCausalLM (generator_resident_multi_s3_2_pb_vllm:
S3.2 with the paged-bucket prefill on the resident engine package named by GEMMA4_RESIDENT_ENGINE_PACKAGE, resident_m5_11
in the package; S3-PB P3 subset v5 and the L3-only re-run, bench/packaging/dram_fix/s3pb/) that first makes sure the
resident weight cache exists, as package_boot_pb3.Gemma4ResidentSpecPB3Packaged does for `single-user`. The cache is
keyed by engine package, so this profile's GRES_CACHE is its own directory (update3/UPDATE3_PLAN.md section 1).
package_boot.py is unchanged: this module only reuses its ensure_resident_cache()."""

from models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_vllm import Gemma4ResidentMultiS32PBForCausalLM
from models.demos.gemma4_31b_qb2.tt.package_boot import ensure_resident_cache


class Gemma4ResidentMultiS32PBPackaged(Gemma4ResidentMultiS32PBForCausalLM):
    """Gemma4ResidentMultiS32PBForCausalLM that first makes sure the resident weight cache exists (the package's class)."""

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        ensure_resident_cache()
        return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, **kwargs)
