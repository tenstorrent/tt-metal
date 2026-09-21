# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exclusive synchronized phase attribution, not an HTTP performance benchmark.

The selected norm-preserving layout/head policies are held fixed. Synchronization
perturbs execution; uninstrumented warm timings bracket the diagnostic. Nested
inclusive times must not be summed. The exclusive partition subtracts children.
"""

import argparse
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lengths", default="4096,32768")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    gen = None
    report = dict(batch=args.batch, full=args.full, rows=[])
    selected = dict(prefill_sharded_residual=True, prefill_replicated_norm=True)
    try:
        with patch(
            "models.autoports.qwen_qwen3_8_27b.tt.model.decoder_policy",
            side_effect=lambda precision, layer: {**decoder_policy(precision, layer), **selected},
        ):
            gen = build_generator(
                "models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=None if args.full else [0, 3]
            )
        gen.batched_prefill = True
        gen.model.prefill_sharded_residual = True
        gen.model.prefill_batched_head = True
        gen.skip_intermediate_prefill_head = True
        cache = gen._ensure_cache(args.batch, ((max(lengths) + 31) // 32) * 32)
        for length in lengths:
            tokens = (torch.arange(length) % 256 + 100).repeat(args.batch, 1)
            tokens += torch.arange(args.batch)[:, None] * 13
            reference = None
            for mode in ("first", "warm", "warm", "instrumented", "instrumented", "warm_after"):
                gen.reset()
                inclusive, exclusive, counts = defaultdict(float), defaultdict(float), defaultdict(int)
                active = []

                def wrap(stack, owner, name, label, named=False):
                    original = getattr(owner, name)

                    def measured(*a, **kw):
                        key = label + ("." + (a[1] if len(a) > 1 else kw["name"]) if named else "")
                        ttnn.synchronize_device(mesh)
                        frame = [time.perf_counter(), 0.0]
                        active.append(frame)
                        try:
                            return original(*a, **kw)
                        finally:
                            ttnn.synchronize_device(mesh)
                            elapsed = time.perf_counter() - frame[0]
                            active.pop()
                            inclusive[key] += elapsed
                            exclusive[key] += elapsed - frame[1]
                            counts[key] += 1
                            if active:
                                active[-1][1] += elapsed

                    stack.enter_context(patch.object(owner, name, measured))

                with ExitStack() as stack:
                    if mode == "instrumented":
                        for layer in gen.model.layers:
                            for name in ("prefill_forward", "_delta", "_full_prefill", "_finish", "_gather"):
                                wrap(stack, layer, name, layer.kind + "." + name)
                            for name in ("_linear", "_norm"):
                                wrap(stack, layer, name, layer.kind + "." + name, named=True)
                        for name in ("embed", "rope", "logits"):
                            wrap(stack, gen.model, name, "model." + name)
                        for owner, name in (
                            (ttnn.transformer, "chunk_gated_delta_rule"),
                            (ttnn.transformer, "chunked_scaled_dot_product_attention"),
                            (ttnn.experimental.kda, "qkv_causal_conv1d_silu"),
                        ):
                            wrap(stack, owner, name, "native." + name)
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    outputs = gen.prefill_forward(
                        tokens, page_table=gen.page_table, kv_cache=cache, prompt_lens=[length] * args.batch
                    )
                    ttnn.synchronize_device(mesh)
                    elapsed = time.perf_counter() - begin
                actual = torch.cat([gen._host_logits(output) for output in outputs], dim=2)
                if reference is None:
                    reference = actual.clone()
                row = dict(
                    length=length,
                    mode=mode,
                    seconds=elapsed,
                    logits_exact=torch.equal(actual, reference),
                    inclusive_seconds=dict(inclusive),
                    exclusive_seconds=dict(exclusive),
                    unattributed_seconds=elapsed - sum(exclusive.values()) if exclusive else None,
                    counts=dict(counts),
                )
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("SELECTED_PREFILL_PHASES", json.dumps(row), flush=True)
                del outputs, actual
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
