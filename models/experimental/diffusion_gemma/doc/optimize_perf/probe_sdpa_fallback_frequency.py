# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Count denoise native-SDPA fallbacks per layer on QB2."""

from __future__ import annotations

import argparse
import json
import os

import torch
import ttnn

from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    make_generation_logits_fn_builder_from_checkpoint_state,
)
from models.experimental.diffusion_gemma.tt.diffusion_attention import (
    get_sdpa_fallback_counts,
    reset_sdpa_fallback_counts,
)
from models.experimental.diffusion_gemma.tt.generate import (
    host_canvas_to_device,
    prefill_prompt_tokens,
    tokenize_prompt,
)


def run(args):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    adapter = None
    try:
        bundle = build_tt_model_from_checkpoint_dir(
            mesh,
            args.checkpoint,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
            num_layers=args.num_layers,
            create_kv_cache=True,
        )
        model = bundle.tt_model
        prompt = args.prompt if args.prompt_repeat == 1 else (args.prompt + " ") * args.prompt_repeat
        prompt_tokens = tokenize_prompt(bundle.tokenizer, prompt)
        prefill = prefill_prompt_tokens(model, prompt_tokens)
        builder = make_generation_logits_fn_builder_from_checkpoint_state(bundle.state_dict, config=model.hf_config)
        adapter = builder(model, prompt_tokens=prompt_tokens, prompt_len=prefill.cache_len)
        adapter.prepare_trace_safe_self_conditioning(canvas_len=args.canvas_length)
        adapter.reset_signal_buffer()
        canvas = host_canvas_to_device(
            mesh,
            torch.randint(
                0,
                int(getattr(bundle.tokenizer, "vocab_size", 262144)),
                (1, args.canvas_length),
                generator=torch.Generator().manual_seed(args.seed),
            ),
        )
        reset_sdpa_fallback_counts()
        logits = adapter._trace_safe_call(canvas, 1)
        ttnn.synchronize_device(mesh)
        counts = get_sdpa_fallback_counts()
        result = {
            "prompt_tokens": len(prompt_tokens[0]),
            "cache_len": prefill.cache_len,
            "num_layers": len(model.layers),
            "fallback_layers": len(counts),
            "fallback_calls": sum(counts.values()),
            "counts": [
                {
                    "layer": layer,
                    "type": model.hf_config.layer_types[layer],
                    "calls": counts[layer],
                }
                for layer in sorted(counts)
            ],
            "native_layers": [layer for layer in range(len(model.layers)) if layer not in counts],
        }
        print("SDPA_FALLBACK_RESULT " + json.dumps(result), flush=True)
        logits.deallocate(True)
        canvas.deallocate(True)
    finally:
        if adapter is not None:
            adapter.reset()
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"),
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
