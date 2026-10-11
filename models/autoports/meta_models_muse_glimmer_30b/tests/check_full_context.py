# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Logical nonaligned context and final cache-page boundary acceptance."""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator

ROOT = Path(__file__).parents[1]


def main(args):
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    generator = None
    try:
        generator = build_generator(ROOT, mesh, **({"layer_indices": [0, 3]} if args.reduced else {}))
        context = generator.model.config.max_position_embeddings
        # Reverse physical pages ensures final-page behavior follows the table.
        table = torch.arange(4095, -1, -1, dtype=torch.int32)[None]
        started = time.perf_counter()
        generator.prefill_forward(
            torch.full((1, context - 1), 42),
            page_table=table,
            kv_cache=generator.kv_cache,
            prompt_lens=[context - 1],
            sample=False,
        )
        final_decode = generator.decode_forward(
            [42], [context - 1], page_table=table, kv_cache=generator.kv_cache, sample=False
        )
        nonaligned_seconds = time.perf_counter() - started
        positions = ttnn.to_torch(ttnn.get_device_tensors(generator.positions)[0]).tolist()
        assert positions == [context]
        generator.reset()
        started = time.perf_counter()
        final_prefill = generator.prefill_forward(
            torch.full((1, context), 42),
            page_table=table,
            kv_cache=generator.kv_cache,
            prompt_lens=[context],
            sample=False,
        ).reshape(1, -1)
        aligned_seconds = time.perf_counter() - started
        cosine = torch.nn.functional.cosine_similarity(final_prefill.float(), final_decode.float(), dim=-1).item()
        finite = bool(torch.isfinite(final_prefill).all() and torch.isfinite(final_decode).all())
        top5_match = final_decode.argmax().item() in final_prefill.flatten().topk(5).indices.tolist()
        # Preserve numeric failure evidence before enforcing the unchanged gates.
        torch.save(dict(prefill=final_prefill, decode=final_decode), args.output.with_suffix(".pt"))
        result = dict(
            command=[sys.executable, *sys.argv],
            precision_config=generator.model.precision_config,
            precision_config_environment_override=os.environ.get("MUSE_PRECISION_CONFIG"),
            status="pass" if finite and cosine > 0.99 and top5_match else "capability_gate_fail",
            finite_logits=finite,
            decode_token_in_prefill_top5=top5_match,
            logit_control_sha256=hashlib.sha256(args.output.with_suffix(".pt").read_bytes()).hexdigest(),
            terminal_policy=generator.model.terminal_policy,
            implementation_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "tt").glob("*.py")
            },
            layers=2 if args.reduced else 52,
            context=context,
            nonaligned_prefill=context - 1,
            aligned_prefill=context,
            decode_position=context - 1,
            physical_last_page=0,
            nonaligned_seconds=nonaligned_seconds,
            aligned_seconds=aligned_seconds,
            prefill_decode_cosine=cosine,
            prefill_token=final_prefill.argmax().item(),
            decode_token=final_decode.argmax().item(),
            counters=dict(generator.counters),
            traces={str(key): str(value) for key, value in generator.traces.items()},
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        assert finite
        assert cosine > 0.99, cosine
        assert top5_match
        assert generator.counters["captures"] == generator.TRACE_COUNT
        print("CONTEXT_CONTRACT_PASS", result, flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if generator:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
