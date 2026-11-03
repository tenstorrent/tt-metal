# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace/eager equality and cross-request reuse at physical bucket boundaries."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import build_generator

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--context", type=int, default=131072)
    args = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    rows = []
    try:
        gen = build_generator(ROOT, mesh, override_num_layers=2, max_seq_len=args.context)
        gen.capture()
        initial = (str(gen.model_trace), {str(k): str(v[0]) for k, v in gen.sampling_traces.items()})
        # The first real request is traced before its eager oracle and maps to
        # pages not written by warmup. This catches capture-without-execute.
        first_prompt = (torch.arange(97) * 7 + 101)[None]
        first_pages = gen.page_table_host.flip(-1).contiguous()
        first_outputs = []
        for traced in (True, False):
            first_outputs.append(
                gen.prefill_forward(
                    first_prompt,
                    page_table=first_pages,
                    kv_cache=gen.kv_cache,
                    prompt_lens=[97],
                    enable_trace=traced,
                )
            )
        torch.testing.assert_close(first_outputs[0], first_outputs[1], rtol=0, atol=0)
        first_capture = dict(
            traced_before_eager=True,
            reverse_mapped_fresh_pages=True,
            length=97,
            bitwise_equal=True,
            trace_ids={str(k): str(v) for k, v in gen.prefill_traces.items()},
        )
        known = dict(gen.prefill_traces)
        # Actual branch boundaries: tile/page, chunk and attention grid threshold.
        for length in (31, 32, 33, 127, 128, 129, 4095, 4096, 4097, 65535, 65536, 65537, 129, 33):
            tokens = (torch.arange(length) % 1000 + 10)[None]
            outputs = []
            elapsed = []
            for traced in (False, True, True):
                start = time.perf_counter()
                outputs.append(
                    gen.prefill_forward(
                        tokens,
                        page_table=gen.page_table_host,
                        kv_cache=gen.kv_cache,
                        prompt_lens=[length],
                        enable_trace=traced,
                    )
                )
                elapsed.append(time.perf_counter() - start)
            for got in outputs[1:]:
                torch.testing.assert_close(got, outputs[0], rtol=0, atol=0)
            for key, tid in known.items():
                assert gen.prefill_traces[key] == tid
            known = dict(gen.prefill_traces)
            assert str(gen.model_trace) == initial[0]
            assert {str(k): str(v[0]) for k, v in gen.sampling_traces.items()} == initial[1]
            # Interleave sampling/model replay before the next request.
            gen.generate(tokens[0].tolist(), 3, stop_on_eos=False)
            rows.append(
                dict(
                    length=length,
                    bitwise_equal=True,
                    seconds=elapsed,
                    prefill_trace_ids={str(k): str(v) for k, v in known.items()},
                    counters=dict(gen.counters),
                )
            )
            args.output.write_text(json.dumps(dict(status="running", rows=rows), indent=2) + "\n")
            print(rows[-1], flush=True)
        changed_prompt = (torch.arange(129) * 11 + 3001)[None]
        changed_pages = torch.roll(gen.page_table_host, gen.pages_per_slot // 2, dims=-1)
        gen.prefill_forward(
            changed_prompt, page_table=changed_pages, kv_cache=gen.kv_cache, prompt_lens=[33], enable_trace=True
        )
        continuation = gen.prefill_forward(
            changed_prompt,
            page_table=changed_pages,
            kv_cache=gen.kv_cache,
            start_pos=[33],
            prompt_lens=[129],
            enable_trace=True,
        )
        expected = gen.prefill_forward(
            changed_prompt, page_table=changed_pages, kv_cache=gen.kv_cache, prompt_lens=[129], enable_trace=False
        )
        # Different segmentation changes the quantized physical graph. Record
        # that numerical comparison, but compare trace state against an eager
        # oracle with the same prefix and continuation boundaries. Full-model
        # reference accuracy after split prefill is checked separately.
        cross_segmentation = dict(
            pcc=float(torch.corrcoef(torch.stack([continuation.flatten(), expected.flatten()]))[0, 1]),
            top1_equal=torch.equal(continuation.argmax(-1), expected.argmax(-1)),
            continuation_top1=continuation.argmax(-1).tolist(),
            monolithic_top1=expected.argmax(-1).tolist(),
        )
        gen.prefill_forward(
            changed_prompt, page_table=changed_pages, kv_cache=gen.kv_cache, prompt_lens=[33], enable_trace=False
        )
        segmented_eager = gen.prefill_forward(
            changed_prompt,
            page_table=changed_pages,
            kv_cache=gen.kv_cache,
            start_pos=[33],
            prompt_lens=[129],
            enable_trace=False,
        )
        torch.testing.assert_close(continuation, segmented_eager, rtol=0, atol=0)
        pcc = float(torch.corrcoef(torch.stack([continuation.flatten(), segmented_eager.flatten()]))[0, 1])
        assert not torch.equal(continuation, first_outputs[0])
        for key, tid in known.items():
            assert gen.prefill_traces[key] == tid
        changed_inputs = dict(
            traced_before_matching_eager=True,
            changed_tokens=True,
            changed_page_map=True,
            continuation=[33, 129],
            pcc=pcc,
            top1_equal=True,
            bitwise_equal=True,
            oracle="eager with identical prefix33 and continuation33to129",
            cross_segmentation_diagnostic=cross_segmentation,
        )
        assert gen.counters["prefill_captures"] == len(gen.prefill_traces)
        args.output.write_text(
            json.dumps(
                dict(
                    status="passed",
                    rows=rows,
                    first_capture=first_capture,
                    changed_inputs=changed_inputs,
                    model_trace=initial[0],
                    sampling_traces=initial[1],
                    program_cache_allocations_tracked=True,
                ),
                indent=2,
            )
            + "\n"
        )
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
