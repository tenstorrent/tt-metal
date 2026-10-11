# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Admission must preserve a live request's pending feedback and RNG stream."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator

ROOT = Path(__file__).parents[1]


def check_live_admission(generator):
    assert generator.batch_size >= 2
    batch = generator.batch_size
    table = torch.stack([torch.arange(4096, dtype=torch.int32) % 128 + slot * 128 for slot in range(batch)])

    def read(tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()

    rows = []
    for top_k in (1, 32):
        for boundary in ("sample", "logits", "all_logits"):
            continuations = {}
            admission_checks = []
            for admit in (False, True):
                generator.reset()
                # Large integer seeds catch accidental BF16/FP32 conversion.
                generator.set_sampling_params(top_k=top_k, top_p=0.9 if top_k > 1 else 0.0, seeds=2147483000)
                generator.prefill_forward(
                    torch.arange(32, 65)[None],
                    page_table=table,
                    kv_cache=generator.kv_cache,
                    prompt_lens=[33],
                    slots=[0],
                )
                generator.decode_steps(3)
                if admit:
                    live = dict(
                        tokens=read(generator.tokens).flatten()[0],
                        seeds=read(generator.sampler._seeds).flatten()[0],
                        logits=read(generator.logits)[0, 0, 0],
                        position=read(generator.positions).flatten()[0],
                    )
                    # Admission at zero, then an unaligned append using decode
                    # and all three prepared continuation-prefill alignments.
                    for start, length in ((0, 31), (31, 98)):
                        generator.prefill_forward(
                            torch.arange(100 + start, 100 + start + length)[None],
                            page_table=table,
                            kv_cache=generator.kv_cache,
                            prompt_lens=[length],
                            start_pos=[start],
                            slots=[1],
                            sample=boundary == "sample",
                            return_all_logits=boundary == "all_logits",
                        )
                        actual = dict(
                            tokens=read(generator.tokens).flatten()[0],
                            seeds=read(generator.sampler._seeds).flatten()[0],
                            logits=read(generator.logits)[0, 0, 0],
                            position=read(generator.positions).flatten()[0],
                        )
                        for name in live:
                            assert torch.equal(live[name], actual[name]), (top_k, boundary, start, name)
                        admission_checks.append(dict(start=start, length=length, live_state_bit_exact=True))
                continuation = []
                for _ in range(4):
                    generator.decode_steps(1)
                    continuation.append(generator._read_tokens()[0])
                continuations[str(admit)] = dict(tokens=continuation, seed=int(read(generator.sampler._seeds)[0]))
            assert continuations["False"] == continuations["True"], (top_k, boundary, continuations)
            rows.append(
                dict(
                    top_k=top_k,
                    boundary=boundary,
                    continuations=continuations,
                    admission_checks=admission_checks,
                    exact_match=True,
                )
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    generator = None
    try:
        generator = build_generator(
            ROOT, mesh, batch_size=args.batch, **({} if args.full else {"layer_indices": [0, 3]})
        )
        rows = check_live_admission(generator)
        args.output.write_text(
            json.dumps(
                dict(batch=args.batch, layers=52 if args.full else 2, cases=rows, counters=dict(generator.counters)),
                indent=2,
            )
            + "\n"
        )
        print("LIVE_ADMISSION_PASS", flush=True)
    finally:
        if generator:
            generator.close()
        ttnn.close_mesh_device(mesh)
