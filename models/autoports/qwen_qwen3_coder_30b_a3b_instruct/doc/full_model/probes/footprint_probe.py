# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-stack DRAM footprint, measured on the model that actually runs.

Stage 03's ``doc/multichip_decoder/probes/footprint_probe.py`` allocated the
per-die *shapes* and never ran through them. This builds the real
``Qwen3CoderModel`` -- real weights, real embedding, real LM head, the real
paged KV cache at the requested context, the real RoPE tables -- captures the
real decode traces, and then runs a token through it, reporting the allocator
between each stage.

    python .../probes/footprint_probe.py --context 262144
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import Qwen3CoderGenerator  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.generator import _resolve_snapshot  # noqa: E402
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import (  # noqa: E402
    DEFAULT_TRACE_REGION_SIZE,
    Qwen3CoderModel,
)

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--prompt-len", type=int, default=128)
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), trace_region_size=DEFAULT_TRACE_REGION_SIZE)
    report = {"context": args.context, "layers": args.layers, "stages_gb_per_die": {}}
    try:

        def used_gb():
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            return (
                view.total_bytes_allocated_per_bank * view.num_banks / 1e9,
                view.total_bytes_per_bank * view.num_banks / 1e9,
            )

        base, total = used_gb()
        report["dram_per_die_gb"] = total
        report["baseline_gb_per_die"] = base
        print(f"P|DRAM per die: {total:.3f} GB total, {base:.3f} GB in use before the model", flush=True)

        t0 = time.perf_counter()
        model = Qwen3CoderModel.from_checkpoint(
            _resolve_snapshot(),
            mesh_device=mesh,
            max_batch_size=1,
            max_cache_len=args.context,
            num_layers=args.layers,
        )
        report["weight_load_s"] = time.perf_counter() - t0
        weights, _ = used_gb()
        report["stages_gb_per_die"]["weights_embed_lm_head_rope"] = weights - base
        print(f"P|weights + embed + lm_head + rope tables: {weights - base:8.3f} GB/die", flush=True)

        gen = Qwen3CoderGenerator(model, tokenizer=None)
        gen._ensure_kv_cache()
        with_kv, _ = used_gb()
        report["stages_gb_per_die"]["kv_cache"] = with_kv - weights
        print(f"P|+ paged KV cache at context {args.context}: {with_kv - weights:8.3f} GB/die", flush=True)

        prompt = list(range(1000, 1000 + args.prompt_len))
        page_table = gen.make_page_table([args.prompt_len + 8])
        gen.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, active_batch=1)
        t0 = time.perf_counter()
        gen.prefill_forward(
            torch.tensor([prompt]),
            page_table=page_table,
            kv_cache=gen._kv_cache,
            prompt_lens=[args.prompt_len],
            sampling_mode="device",
        )
        report["prefill_s"] = time.perf_counter() - t0
        gen.decode_forward(
            None,
            torch.tensor([args.prompt_len]),
            page_table=page_table,
            kv_cache=gen._kv_cache,
            sampling_mode="device",
            enable_trace=True,
            active_batch=1,
        )
        after, _ = used_gb()
        report["stages_gb_per_die"]["traces_and_persistent_buffers"] = after - with_kv
        report["total_gb_per_die"] = after - base
        report["headroom_gb_per_die"] = total - after
        print(f"P|+ captured traces and persistent buffers:  {after - with_kv:8.3f} GB/die", flush=True)
        print(f"P|TOTAL: {after - base:8.3f} GB/die, {total - after:8.3f} GB/die free", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    (HERE / f"footprint_{args.context}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
