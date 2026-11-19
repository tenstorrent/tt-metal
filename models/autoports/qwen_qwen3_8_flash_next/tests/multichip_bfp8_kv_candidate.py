# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in BFP8 KV policy, composed with the normal optimization candidates.

Run this module with the same argv as multichip_optimization_candidates. No
cache conversion occurs in the decoder forward; setup casts only newly created
zero caches. Native BF16 paged_update_cache inputs remain BF16 in both phases.
"""


def apply():
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt import multichip_decoder as m
    from models.autoports.qwen_qwen3_8_flash_next.tt.bfp8_selected_paged_kv import local_selected_paged_kv_bfp8

    original_state = m.MultichipDecoder._create_base_state
    original_build = m.MeshQSA.build.__func__

    def state(self, **kwargs):
        result = original_state(self, **kwargs)
        if self.kind == "indexed_attention":
            result.attention.k_cache = ttnn.typecast(result.attention.k_cache, ttnn.bfloat8_b)
            result.attention.v_cache = ttnn.typecast(result.attention.v_cache, ttnn.bfloat8_b)
        return result

    def build(cls, *args, **kwargs):
        result = original_build(cls, *args, **kwargs)
        result.gather = local_selected_paged_kv_bfp8
        return result

    m.MultichipDecoder._create_base_state = state
    m.MeshQSA.build = classmethod(build)


if __name__ == "__main__":
    import runpy

    apply()
    runpy.run_module(
        "models.autoports.qwen_qwen3_8_flash_next.tests.multichip_optimization_candidates", run_name="__main__"
    )
