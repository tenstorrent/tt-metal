# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device seed-offset checks through the adapter, without scheduler histories."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.vllm_adapter_decode_probe import build_adapter
from models.common.sampling.generator import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    report = {"status": "running", "layers": [0, 3], "batch": 3, "context": 256, "cases": []}

    def record(name, actual, expected):
        passed = actual == expected
        report["cases"].append({"name": name, "actual": actual, "expected": expected, "passed": passed})
        output.write_text(json.dumps(report, indent=2) + "\n")
        if not passed:
            raise AssertionError(name)

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_adapter(None, mesh, max_batch_size=3, max_seq_len=256)
        state = gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table, slots=(0, 1, 2))
        gen.capture(state)
        lengths = [33, 35, 37]
        prompts = torch.zeros((3, 37), dtype=torch.int64)
        for row, length in enumerate(lengths):
            prompts[row, :length] = torch.arange(100 + row * 100, 100 + row * 100 + length)
        params = SamplingParams(temperature=[0.8] * 3, top_k=32, top_p=0.9, seed=[42, 43, 44])

        def seeds():
            return (
                ttnn.to_torch(ttnn.get_device_tensors(gen.sampler.tt_sampling.seeds_tt_tensor)[0])
                .reshape(-1)[:3]
                .long()
                .tolist()
            )

        def decode(tokens, positions, *, reset=True, remap=None, host=False):
            handle = gen.decode_forward(
                tokens,
                positions,
                page_table=gen.page_table if reset else None,
                kv_cache=gen.cache,
                sampling_params=None if host else params,
                reload_inputs=reset,
                reload_sampling_params=reset and not host,
                reset_sampling_state=reset and not host,
                slot_remap=remap,
                read_from_device=False,
            )
            read, events = gen.read_decode_output(handle, async_read=True)
            for event in events:
                ttnn.event_synchronize(event)
            return gen.process_decode_output_host(read, is_tokens=not host)

        first = gen.prefill_forward(
            prompts,
            page_table=gen.page_table,
            kv_cache=gen.cache,
            prompt_lens=lengths,
            original_prompt_lens=list(lengths),
            sampling_params=params,
        )
        first = first[0] if isinstance(first, tuple) else first
        before = seeds()
        second = decode(first, lengths)
        record("first_decode_preserves_prefill_draw", seeds(), [x + 1 for x in before])
        third = decode(None, None, reset=False)
        record("resident_step_advances_once", seeds(), [x + 2 for x in before])

        order = [2, 0, 1]
        params = replace(params, seed=[44, 42, 43])
        gen.page_table = gen.page_table[order].clone()
        lengths = [lengths[i] for i in order]
        tokens = decode(third[order], [x + 2 for x in lengths], remap=order)
        record("remap_reset_uses_request_origin", seeds(), [before[i] + 3 for i in order])

        # Host mode consumes a logical output while the device sampler is idle.
        # This test supplies a fixed valid next token; it does not implement host sampling.
        logits = decode(tokens, [x + 3 for x in lengths], host=True)
        record("host_interval_keeps_device_seed_idle", seeds(), [before[i] + 3 for i in order])
        record("host_logits_geometry", list(logits.shape), [3, 1, gen.model.vocab_size])
        decode(torch.full((3, 1), 100, dtype=torch.int64), [x + 4 for x in lengths])
        record("return_from_host_uses_logical_offset", seeds(), [before[i] + 5 for i in order])
        report["counters"] = dict(gen.counters)
        report["status"] = "pass"
        output.write_text(json.dumps(report, indent=2) + "\n")
        print("ADAPTER_SEED_OFFSET_PASS", flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
