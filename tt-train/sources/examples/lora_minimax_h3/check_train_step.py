# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One LoRA forward/backward on the GPU-dumped sample, against `parity/train_step` (dump_parity.py stage `train`).

    python check_train_step.py --dump /localdev/bmijanovic/parity/transformer/0_t0.5 \
        --train-dump /localdev/bmijanovic/parity/train_step --mesh 1,8

Loads the GPU's initial adapter (gaussian A, zero B), computes the masked MSE at the dumped (x_t, noise, t) and
compares loss, global grad norm and the per-tensor grad norms of block 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import ttml

from ttml.models.minimax_h3 import (
    H3Config,
    H3Inputs,
    H3Transformer,
    build_packed_sequence,
    build_row_timesteps,
    load_transformer,
    read_time_embedder,
    upload,
)
from check_transformer import DEFAULT_WEIGHTS, open_mesh
from lora_utils import gather, grad_norms, inject, load_adapter, lora_params, masked_mse, to_diffusers_key

log = logging.getLogger("minimax_h3.check_train")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True)
    p.add_argument("--train-dump", required=True)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--mesh", default="1,8")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument(
        "--layers", type=int, default=None, help="truncate to the first N blocks (tp=1 fits N=1 on one chip)"
    )
    p.add_argument("--save-grads", default=None, help="write the per-tensor grad norms to this json")
    p.add_argument("--sdpa", choices=["fused", "composite"], default="composite")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    dump, tdump = Path(args.dump), Path(args.train_dump)
    meta = json.loads((dump / "meta.json").read_text())
    summary = json.loads((tdump / "summary.json").read_text())
    arr = {n: np.load(dump / f"{n}.npy") for n in ["x_t", "audio", "prompt_embeds", "target"]}
    L = arr["prompt_embeds"].shape[0]
    layout = build_packed_sequence(L, 1, meta["latent_height"], meta["latent_width"])
    timesteps, timestep_indices = build_row_timesteps(layout, meta["t"])

    dp, tp = (int(v) for v in args.mesh.split(","))
    open_mesh(dp, tp)
    try:
        config = H3Config.from_hf(
            Path(args.weights) / "config.json", num_layers=args.layers, use_tp=tp > 1, sdpa=args.sdpa
        )
        t0 = time.perf_counter()
        model = H3Transformer(config)
        only = None
        if args.layers is not None:
            only = ("proj_in", "audio_proj", "context_embedder", "refiner_", "norm_out", "proj_out") + tuple(
                f"blocks/{i}/" for i in range(args.layers)
            )
        load_transformer(model, Path(args.weights), only=only)
        lora = inject(model, args.rank, args.rank)
        n = load_adapter(lora, config, tdump / "lora_init")
        trainable = lora_params(lora)
        log.info(
            "loaded base + %d adapter tensors in %.0f s; %d trainable tensors (%.1f M per device)",
            n,
            time.perf_counter() - t0,
            len(trainable),
            sum(np.prod(t.shape()) for t in trainable.values()) / 1e6,
        )
        inputs = H3Inputs.build(
            config,
            read_time_embedder(Path(args.weights)),
            arr["x_t"],
            arr["audio"],
            arr["prompt_embeds"],
            layout,
            timesteps,
            timestep_indices,
        )
        target = np.zeros((1, 1, inputs.S_pad, config.video_in_channels), dtype=np.float32)
        target[0, 0, layout.video_indices] = arr["target"]
        row_mask = np.zeros_like(target)
        row_mask[0, 0, layout.video_indices] = 1.0
        target_t, mask_t = upload(target), upload(row_mask)

        lora.train()
        t0 = time.perf_counter()
        pred, _ = lora(inputs, with_audio=False)
        loss = masked_mse(pred, target_t, mask_t, inputs)
        loss_value = float(gather(loss).reshape(-1)[0])
        loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        log.info("forward+backward %.1f s", time.perf_counter() - t0)
        total, per = grad_norms(lora)
        log.info(
            "loss %.5f (gpu %.5f, rel %.3f) | grad_norm %.5f (gpu %.5f, rel %.3f)",
            loss_value,
            summary["loss"],
            loss_value / summary["loss"] - 1,
            total,
            summary["grad_norm"],
            total / summary["grad_norm"] - 1,
        )
        gpu = summary["per_tensor_grad_norm"]
        for name, value in per.items():
            if ".blocks.0." in f".{name}" or name.startswith("blocks.0."):
                key = to_diffusers_key(name).replace(".weight", ".default.weight")
                ref = gpu.get(key)
                log.info("  %-28s tt %.5f gpu %s", name, value, f"{ref:.5f}" if ref is not None else "missing " + key)
        zero_a = sum(1 for k, v in per.items() if k.endswith("lora_A") and v > 0)
        log.info("lora_A tensors with non-zero grad: %d (expected 0 while B = 0)", zero_a)
        if args.save_grads:
            Path(args.save_grads).write_text(
                json.dumps(
                    {
                        "loss": loss_value,
                        "grad_norm": total,
                        "mesh": args.mesh,
                        "layers": config.num_layers,
                        "per_tensor": per,
                    },
                    indent=1,
                )
            )
            log.info("per-tensor norms -> %s", args.save_grads)
    finally:
        ttml.close_device_mesh()
    return 0


if __name__ == "__main__":
    sys.exit(main())
