# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Staged parity of the ttml MiniMax-H3 transformer against one GPU dump (dump_parity.py, stage `transformer`).

    python check_transformer.py --dump /localdev/bmijanovic/parity/transformer/0_t0.5 --mesh 1,8 --stage refiner

Stages, each loading only the weights it needs:
  refiner  context_embedder + token refiner on the text rows      -> text_refined
  block0   transformer block 0 on the dumped packed input          -> h_block0
  full     all 50 blocks + heads                                   -> h_block_last, video_out, audio_out
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import ttml
from ttml.common.utils import no_grad

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

log = logging.getLogger("minimax_h3.check")

MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DEFAULT_WEIGHTS = (
    Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots" / MODEL_REVISION / "transformer"
)


def open_mesh(dp: int, tp: int):
    if tp * dp > 1 and "TT_MESH_GRAPH_DESC_PATH" not in os.environ:
        log.warning("TT_MESH_GRAPH_DESC_PATH is not set")
    ttml.open_device_mesh(ttml.Mesh((dp, tp), ("dp", "tp")))
    ctx = ttml.autograd.AutoContext.get_instance()
    if dp * tp > 1 and not ctx.is_parallelism_context_initialized():
        ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=dp > 1, enable_tp=tp > 1))
    ctx.set_seed(0)


def to_host(t) -> np.ndarray:
    return np.asarray(ttml.Sharding.from_tensor(t).gather(t)).astype(np.float32)


def report(name: str, got: np.ndarray, ref: np.ndarray) -> None:
    got, ref = got.astype(np.float64), ref.astype(np.float64)
    if got.shape != ref.shape:
        raise RuntimeError(f"{name}: shape {got.shape} vs reference {ref.shape}")
    a, b = got.ravel() - got.mean(), ref.ravel() - ref.mean()
    pcc = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-30)
    if got.ndim == 2:
        rc = (got * ref).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-30)
        rows = f" | row-cos min {rc.min():.5f} mean {rc.mean():.5f}"
    else:
        rows = ""
    log.info(
        "%s: pcc %.6f | rel L2 %.4f | max|d| %.3g (ref max %.3g)%s",
        name,
        pcc,
        rel,
        np.abs(got - ref).max(),
        np.abs(ref).max(),
        rows,
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="one parity/transformer/<key>_t<t> folder")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--mesh", default="1,8", help="dp,tp")
    p.add_argument("--stage", choices=["refiner", "block0", "full"], default="refiner")
    p.add_argument("--save-taps", default=None, help="folder to save the device-side taps as npy (full stage)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    dump = Path(args.dump)
    meta = json.loads((dump / "meta.json").read_text())
    arr = {
        n: np.load(dump / f"{n}.npy")
        for n in [
            "x_t",
            "audio",
            "prompt_embeds",
            "timestep",
            "timestep_indices",
            "text_refined",
            "h_in",
            "h_block0",
            "h_block_last",
            "video_out",
            "audio_out",
        ]
    }
    L = arr["prompt_embeds"].shape[0]
    layout = build_packed_sequence(L, 1, meta["latent_height"], meta["latent_width"])
    timesteps, timestep_indices = build_row_timesteps(layout, meta["t"])
    assert np.array_equal(timesteps, arr["timestep"]) and np.array_equal(timestep_indices, arr["timestep_indices"])

    dp, tp = (int(v) for v in args.mesh.split(","))
    open_mesh(dp, tp)
    try:
        num_layers = {"refiner": 0, "block0": 1, "full": None}[args.stage]
        config = H3Config.from_hf(Path(args.weights) / "config.json", num_layers=num_layers, use_tp=tp > 1)
        log.info("stage %s: %d blocks, tp %d, S=%d", args.stage, config.num_layers, tp, layout.sequence_length)
        t0 = time.perf_counter()
        model = H3Transformer(config)
        only = {"refiner": ("context_embedder", "refiner_"), "block0": ("blocks/0/",), "full": None}[args.stage]
        n = load_transformer(model, Path(args.weights), only=only)
        log.info("loaded %d params in %.0f s", n, time.perf_counter() - t0)
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
        S = inputs.S

        t0 = time.perf_counter()
        with no_grad():
            if args.stage == "refiner":
                out = to_host(model.embed_text(inputs)).reshape(inputs.L_pad, -1)[:L]
                report("text_refined", out, arr["text_refined"])
            elif args.stage == "block0":
                h = upload(np.pad(arr["h_in"], ((0, inputs.S_pad - S), (0, 0)))[None, None])
                out = to_host(model.run_blocks(h, inputs)).reshape(inputs.S_pad, -1)[:S]
                report("h_block0", out, arr["h_block0"])
            else:
                text = model.embed_text(inputs)
                h = model.pack(text, model.audio_proj_in(inputs.audio), model.proj_in(inputs.x_t), inputs)
                taps = {"h_in": to_host(h).reshape(inputs.S_pad, -1)[:S]}
                report("h_in", taps["h_in"], arr["h_in"])
                taps["h_block0"] = None
                for i, block in enumerate(model.blocks):
                    mods = block.modulation(block.adaln_table(inputs.silu_temb), inputs.sel_adaln)
                    h = block(h, mods, inputs.mask, inputs.rope_params)
                    if i == 0:
                        taps["h_block0"] = to_host(h).reshape(inputs.S_pad, -1)[:S]
                        report("h_block0", taps["h_block0"], arr["h_block0"])
                taps["h_block_last"] = to_host(h).reshape(inputs.S_pad, -1)[:S]
                report("h_block_last", taps["h_block_last"], arr["h_block_last"])
                video_out, audio_out = model.head(h, inputs)
                taps["video_out"] = to_host(video_out).reshape(inputs.S_pad, -1)[layout.video_indices]
                taps["audio_out"] = to_host(audio_out).reshape(inputs.S_pad, -1)[layout.audio_indices]
                report("video_out", taps["video_out"], arr["video_out"])
                report("audio_out", taps["audio_out"], arr["audio_out"])
                if args.save_taps:
                    folder = Path(args.save_taps)
                    folder.mkdir(parents=True, exist_ok=True)
                    for name, value in taps.items():
                        np.save(folder / f"{name}.npy", value)
                    log.info("taps -> %s", folder)
        log.info("forward %.1f s", time.perf_counter() - t0)
    finally:
        ttml.close_device_mesh()
    return 0


if __name__ == "__main__":
    sys.exit(main())
