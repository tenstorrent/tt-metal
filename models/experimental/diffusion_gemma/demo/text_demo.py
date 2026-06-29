# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Text-only DiffusionGemma prompt-to-text demo for #47464 bring-up."""

from __future__ import annotations

import argparse
import os

from models.experimental.diffusion_gemma.checkpoint import build_and_generate_text_from_checkpoint_dir


_MESH_SHAPES = {
    "N150": (1, 1),
    "N300": (1, 2),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "T3K": (1, 8),
}


def _parse_mesh_shape(mesh: str) -> tuple[int, int]:
    if mesh in _MESH_SHAPES:
        return _MESH_SHAPES[mesh]
    try:
        rows, cols = mesh.lower().split("x", maxsplit=1)
        return int(rows), int(cols)
    except ValueError as exc:
        raise ValueError(f"unknown mesh shape {mesh!r}; use a known label or ROWSxCOLS") from exc


def _open_mesh_device(mesh: str):
    import ttnn

    rows, cols = _parse_mesh_shape(mesh)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DiffusionGemma text generation on a TT mesh.")
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("DG_CKPT", "google/diffusiongemma-26B-A4B-it"),
        help="HF checkpoint directory or model id (default: DG_CKPT or google/diffusiongemma-26B-A4B-it)",
    )
    parser.add_argument("--prompt", default=os.getenv("DG_PROMPT", "The capital of France is"))
    parser.add_argument("--mesh", default=os.getenv("MESH_DEVICE", "P150x4"), help="Mesh label or ROWSxCOLS")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=123, help="Base seed for canvas init and derived noise hooks")
    parser.add_argument(
        "--num-blocks", type=int, default=None, help="Override block count; otherwise derived from max tokens"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=None, help="Optional smoke-test layer limit")
    parser.add_argument("--local-files-only", action="store_true", help="Do not fetch tokenizer files from HF hub")
    parser.add_argument("--bounded-sliding-kv-cache", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mesh_device = _open_mesh_device(args.mesh)
    try:
        generation = build_and_generate_text_from_checkpoint_dir(
            mesh_device,
            args.checkpoint,
            args.prompt,
            tokenizer_kwargs={"local_files_only": args.local_files_only},
            model_kwargs={
                "max_batch_size": args.batch,
                "max_seq_len": args.max_seq_len,
                "num_layers": args.num_layers,
                "bounded_sliding_kv_cache": args.bounded_sliding_kv_cache,
            },
            max_new_tokens=args.max_new_tokens,
            num_blocks=args.num_blocks,
            seed=args.seed,
            batch=args.batch,
        )
    finally:
        import ttnn

        ttnn.close_mesh_device(mesh_device)

    for text in generation.text:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
