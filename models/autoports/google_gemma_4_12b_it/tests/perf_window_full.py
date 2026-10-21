# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tracy windows of the reduced full model: one real layer of each kind (0 sliding, 5 full) with the complete
embedding, final norm, LM head, soft-cap and sampler, real shapes / caches / page tables.

Windows (signposts): DECODE (B=1 token-out step: embedding + gather, 2 layers, terminal, sampling), DECODE_B32,
PREFILL (128-token prompt bucket, last-token terminal and first-token sampling). The device ops are those of the
captured traces; they run eagerly here so the profiler attributes each op.

  python -m tracy -r -p -v models/autoports/google_gemma_4_12b_it/tests/perf_window_full.py

Environment: ``GEMMA4_TRACY_LAYERS`` (default ``0,5``) picks the real layers; the precision config is the model
default (``GEMMA4_PRECISION_CONFIG`` overrides it, ``tt/precision.py``).
"""

import os

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_12b_it.tt.model import SCRATCH_BLOCK, Gemma4Model
from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh

try:
    from tracy import signpost
except ImportError:

    def signpost(*args, **kwargs):
        pass


def main():
    torch.set_num_threads(16)
    mesh = open_qb2_mesh(trace_region_size=100_000_000)
    try:
        layers = [int(v) for v in os.environ.get("GEMMA4_TRACY_LAYERS", "0,5").split(",")]
        model = Gemma4Model(mesh, layer_indices=layers, max_context=16384, num_blocks=600, log=lambda *_: None)
        gen = Gemma4Generator(model, decode_buckets=(1, 32), prepare=False)
        ids = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Explain the three laws of thermodynamics."}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )

        def prefill():
            row = gen._alloc_row(len(ids) + 8)
            m = gen.model
            plan = m.chunk_plan(0, len(ids))
            table = m.device_ints(m.prefill_page_table(row, len(ids), plan[-1][2]))
            chunk = torch.zeros(plan[-1][2], dtype=torch.int32)
            chunk[: len(ids)] = torch.tensor(ids)
            h = m.prefill_chunk_hidden(chunk, table, 0)
            gen._write(gen.row_ids, torch.full((1, 32), len(ids) - 1, dtype=torch.int32))
            x = m.select_rows(h, gen.row_ids)
            m.decode_logits(x, gen.logits)
            gen._sample_step()
            gen._release(row)

        def decode(b):
            gen._write(gen.positions[b], torch.arange(200, 200 + b, dtype=torch.int32))
            gen._write(gen.tables[b][0], torch.full((b, gen.pages), SCRATCH_BLOCK, dtype=torch.int32))
            model.decode_step(gen.tokens, gen.positions[b], gen.tables[b][0], gen.logits, user_split=False)
            gen._sample_step()

        for _ in range(2):  # compile + warm
            prefill()
            decode(1)
            decode(32)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)  # flush the build / warmup ops (the device buffer holds a limited op count)
        for name, fn in (("PREFILL", prefill), ("DECODE", lambda: decode(1)), ("DECODE_B32", lambda: decode(32))):
            ttnn.synchronize_device(mesh)
            signpost(f"PERF_{name}")
            fn()
            ttnn.synchronize_device(mesh)
            signpost(f"PERF_{name}_END")
            ttnn.ReadDeviceProfiler(mesh)
    finally:
        close_qb2_mesh(mesh)


if __name__ == "__main__":
    main()
