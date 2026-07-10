# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end causal-prefill A/B for the exact dense-MoE geometry.

Builds one DiffusionGemma model, warms both stock and tuned program configs, then
alternates them over identical token IDs. The measured span is the synchronized
model forward (embedding and host readback excluded). Final logits must be
elementwise identical.

*** DEVICE-OWNERSHIP: run only when QB2 is free. ***
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt.generate import embed_host_tokens
from models.experimental.diffusion_gemma.tt.prefill_moe import FLAG, use_tuned_prefill_moe

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


def _to_host(tensor):
    if tensor.device().get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()
    return ttnn.to_torch(tensor).float()


def _forward(model, tokens, *, tuned):
    os.environ[FLAG] = "1" if tuned else "0"
    embeds = embed_host_tokens(model, tokens)
    ttnn.synchronize_device(model.mesh_device)
    start = time.perf_counter()
    with use_tuned_prefill_moe(model):
        logits = model(
            embeds,
            is_decode=False,
            input_ids_torch=tokens,
            get_last_token=((tokens.shape[1] - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE,
        )
    ttnn.synchronize_device(model.mesh_device)
    elapsed = time.perf_counter() - start
    host = _to_host(logits)
    logits.deallocate(True)
    return elapsed, host


def run(*, seq_len, num_layers, max_seq_len, iters):
    if seq_len % ttnn.TILE_SIZE != 0:
        raise ValueError(f"seq_len must be a multiple of {ttnn.TILE_SIZE}")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        model_inputs = build_tt_model_from_checkpoint_dir(
            mesh,
            CKPT,
            max_batch_size=1,
            max_seq_len=max(max_seq_len, seq_len),
            num_layers=num_layers,
            create_kv_cache=True,
        )
        model = model_inputs.tt_model
        generator = torch.Generator().manual_seed(0)
        tokens = torch.randint(
            0,
            model.hf_config.vocab_size,
            (1, seq_len),
            dtype=torch.long,
            generator=generator,
        )

        _forward(model, tokens, tuned=False)
        _forward(model, tokens, tuned=True)

        baseline_times = []
        tuned_times = []
        baseline_host = None
        tuned_host = None
        for _ in range(iters):
            baseline_s, baseline_host = _forward(model, tokens, tuned=False)
            tuned_s, tuned_host = _forward(model, tokens, tuned=True)
            baseline_times.append(baseline_s)
            tuned_times.append(tuned_s)

        baseline_s = sum(baseline_times) / len(baseline_times)
        tuned_s = sum(tuned_times) / len(tuned_times)
        exact = torch.equal(baseline_host, tuned_host)
        max_abs = float((baseline_host - tuned_host).abs().max())
        print(
            f"RESULT_PREFILL_E2E seq_len={seq_len} layers={num_layers} "
            f"baseline_s={baseline_s:.6f} tuned_s={tuned_s:.6f} speedup={baseline_s/tuned_s:.3f} "
            f"exact={int(exact)} max_abs={max_abs:.6g}",
            flush=True,
        )
        if not exact:
            raise AssertionError(f"tuned prefill logits differ from baseline: max_abs={max_abs}")
    finally:
        os.environ.pop(FLAG, None)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=2)
    args = parser.parse_args()
    run(
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
