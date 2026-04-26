# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import ttnn
from models.tt_transformers.tt.lm_head import LMHead


def test_lm_head_converts_bias_to_output_memory_config_before_add(monkeypatch):
    lm_head = object.__new__(LMHead)
    lm_head.prefetcher = None
    lm_head.split_sizes_dram_sharded = [128]
    lm_head.split_sizes_ring_mm = []
    lm_head.output_weights_dram_sharded = [MagicMock(name="weight")]
    lm_head.output_weights_ring_mm = []
    lm_head.output_biases_dram_sharded = [MagicMock(name="bias")]
    lm_head.output_biases_ring_mm = None
    lm_head.compute_kernel_config = MagicMock(name="compute_kernel_config")
    lm_head.mesh_device = MagicMock(name="mesh_device")
    lm_head.tt_ccl = MagicMock(name="tt_ccl")
    lm_head.args = SimpleNamespace(
        lm_head_dtype=ttnn.bfloat8_b,
        ccl_dtype=ttnn.bfloat16,
        is_galaxy=False,
        get_lm_head_program_config=lambda split_size, prefetcher: MagicMock(name="program_config"),
        get_lm_head_output_mem_config=lambda mode, prefetcher: MagicMock(name="linear_output_memcfg"),
        get_lm_head_sharded_output_mem_config=lambda prefetcher: MagicMock(name="interleaved_output_memcfg"),
    )

    x = MagicMock(name="x")
    linear_out = MagicMock(name="linear_out")
    interleaved_out = MagicMock(name="interleaved_out")
    interleaved_memcfg = MagicMock(name="interleaved_memcfg")
    interleaved_out.memory_config.return_value = interleaved_memcfg
    bias_in_output_mem = MagicMock(name="bias_in_output_mem")
    biased_out = MagicMock(name="biased_out")
    concat_out = MagicMock(name="concat_out")
    concat_out.memory_config.return_value = MagicMock(name="concat_memcfg")
    reduced_out = MagicMock(name="reduced_out")

    calls = {}

    monkeypatch.setattr(ttnn, "linear", lambda *args, **kwargs: linear_out)
    monkeypatch.setattr(
        ttnn,
        "sharded_to_interleaved",
        lambda tensor, memory_config=None: interleaved_out,
    )

    def fake_to_memory_config(tensor, memory_config=None, dtype=None):
        calls["to_memory_config"] = (tensor, memory_config, dtype)
        return bias_in_output_mem

    def fake_add(lhs, rhs, memory_config=None, dtype=None):
        calls["add"] = (lhs, rhs, memory_config, dtype)
        return biased_out

    monkeypatch.setattr(ttnn, "to_memory_config", fake_to_memory_config)
    monkeypatch.setattr(ttnn, "add", fake_add)
    monkeypatch.setattr(ttnn, "concat", lambda outputs, **kwargs: concat_out)
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: None)
    monkeypatch.setattr("models.tt_transformers.tt.lm_head.tt_all_reduce", lambda *args, **kwargs: reduced_out)

    output = lm_head.forward(x)

    assert output is reduced_out
    assert calls["to_memory_config"] == (lm_head.output_biases_dram_sharded[0], interleaved_memcfg, None)
    assert calls["add"] == (interleaved_out, bias_in_output_mem, interleaved_memcfg, ttnn.bfloat16)
