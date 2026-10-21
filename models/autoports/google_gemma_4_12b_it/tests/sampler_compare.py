# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare the two common on-device samplers on this model's real logits buffer (vocab-sharded [1,1,32,65536] per
chip, QB2 ring), traced, profiler off:

* A ``models/common/sampling/tt_sampling.TTSampling`` (the generator's choice): per-chip top-32, gather 128
  candidates (values, indices), global offsets, greedy tie-break, ``ttnn.sampling``; ``tt_out_tok`` = token buffer;
* B ``models/common/modules/sampling/sampling_1d.Sampling1D``: (B1) the same candidate path, (B2) its force-argmax
  path (full-logits all-gather, untilize, argmax), the semantically greedy alternative.

The logits come from a real decode step of the reduced model (layers 0 and 5; the terminal path is complete).
Every variant runs as its own captured trace; tokens of the greedy variants must agree.

  python models/autoports/google_gemma_4_12b_it/tests/sampler_compare.py --out doc/full_model/logs/sampler_compare.json
"""

import argparse
import json
import statistics
import time

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_12b_it.tt.model import Gemma4Model
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh
from models.common.modules.sampling.sampling_1d import Sampling1D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(16)
    mesh = open_qb2_mesh(trace_region_size=0)
    res = {}
    try:
        model = Gemma4Model(mesh, layer_indices=[0, 5], max_context=16384, num_blocks=300, log=lambda *_: None)
        gen = Gemma4Generator(model, decode_buckets=(1, 32), prepare=False)
        V = model.vocab_size
        s1d = Sampling1D(V, mesh, max_batch_size=32, max_top_k=32, allow_force_argmax=True, num_gather_links=1)
        s1d.load_device_buffers()

        def dev(t, dtype):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        k = dev(torch.ones(32, dtype=torch.int32), ttnn.uint32)
        p = dev(torch.zeros(32), ttnn.bfloat16)
        temp = dev(torch.ones(32), ttnn.bfloat16)
        seeds = dev(torch.arange(32, dtype=torch.int32), ttnn.uint32)
        out_b1 = dev(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
        out_b2 = dev(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
        variants = {
            "A_ttsampling_top32": lambda: gen._sample_step(),
            "B1_sampling1d_top32": lambda: s1d.decode_forward(
                gen.logits, k=k, p=p, temp=temp, seeds=seeds, tt_out_tok=out_b1
            ),
            "B2_sampling1d_force_argmax": lambda: s1d.decode_forward(gen.logits, tt_out_tok=out_b2),
        }
        for fn in variants.values():  # compile every variant before any capture
            fn()
        gen.prepare()  # generator warmup + its traces (decode, sampling)
        traces = {}
        for name, fn in variants.items():
            if name.startswith("A_"):
                traces[name] = gen.traces[("sample",)]
                continue
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            traces[name] = tid
        # real logits: prefill a chat prompt, then one traced decode step fills gen.logits
        ids = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain the three laws of thermodynamics."}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        row = gen._alloc_row(len(ids) + 8)
        first = gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[len(ids)])[0, 0]
        logits = gen.decode_forward(
            torch.tensor([int(first.argmax())]), torch.tensor([len(ids)]), page_table=row[None], return_logits=True
        )
        host_argmax = int(logits[0].argmax())
        outs = {"A_ttsampling_top32": gen.tokens, "B1_sampling1d_top32": out_b1, "B2_sampling1d_force_argmax": out_b2}
        for name, tid in traces.items():
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            tok = int(ttnn.to_torch(ttnn.get_device_tensors(outs[name])[0]).reshape(-1)[0])
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(a.reps):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                times.append((time.perf_counter() - t0) / a.reps * 1e3)
            res[name] = {"ms_per_call": round(statistics.median(times), 4), "row0_token": tok}
            print(name, res[name], flush=True)
        res["host_argmax_row0"] = host_argmax
        res["greedy_agree"] = all(res[n]["row0_token"] == host_argmax for n in variants)
        for name, tid in traces.items():
            if not name.startswith("A_"):
                ttnn.release_trace(mesh, tid)
        gen.teardown()
    finally:
        close_qb2_mesh(mesh)
    print("SAMPLERS", json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
