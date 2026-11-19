# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tracked reduced-model regression for penalty history across B1 prefill capture."""

import argparse
import json
import os
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import QwenGenerator
from models.autoports.qwen_qwen3_6_27b.tt.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        gen = QwenGenerator(Model(mesh, layer_indices=[0, 3]), max_seq_len=2048)
        gen.prefill_single.prepare(_diagnostic_lengths=[128])
        gen.prepare(prefill_lengths=(128,))
        assert set(gen.traces) == {"decode", "sample", "sample_penalties", "reset", "prefill_128"}
        traces = dict(gen.traces)
        programs = int(mesh.num_program_cache_entries())

        def count():
            value = gen.sampling.tt_penalties.output_counts_gathered
            return int(ttnn.to_torch(ttnn.get_device_tensors(value)[0]).sum())

        assert not gen.penalty_history_slots and count() == 0
        neutral = gen.generate([1] * 128, 3, ignore_eos=True)
        penalized = gen.generate([2] * 128, 3, presence_penalty=0.5, ignore_eos=True)
        assert count() == 3 and gen.penalty_history_slots == {0}
        gen.prefill_trace.upload([1] * 128)
        assert count() == 0 and not gen.penalty_history_slots
        gen._execute("prefill_128")
        gen.configure_sampling()
        gen._execute("sample")
        assert gen._read_tokens()[0] == neutral[0]
        assert gen.generate([2] * 128, 3, presence_penalty=0.5, ignore_eos=True) == penalized
        assert gen.generate([1] * 128, 3, ignore_eos=True) == neutral
        assert count() == 0 and not gen.penalty_history_slots
        assert dict(gen.traces) == traces and int(mesh.num_program_cache_entries()) == programs

        unsafe = ttnn.empty_like(gen.tokens)
        try:
            gen._execute("decode")
        except RuntimeError as error:
            positive_control = "corrupted on replay" in str(error).lower()
        else:
            positive_control = False
        del unsafe
        assert positive_control, "Trace allocation tracker positive control did not fire"
        gen._execute("decode")
        result = dict(
            passed=True,
            scope="Representative layers0/3, B1, 128-token trace, 2048 context; diagnostic prefill warmup subset",
            captures=gen.counters["captures"],
            retirements=gen.counters["retirements"],
            trace_ids={key: int(value) for key, value in traces.items()},
            program_cache_entries=programs,
            dirty_history_cleared_before_prefill=True,
            repeated_neutral_and_penalty_tokens_equal=True,
            neutral_tokens=neutral,
            penalty_tokens=penalized,
            allocation_positive_control=positive_control,
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
