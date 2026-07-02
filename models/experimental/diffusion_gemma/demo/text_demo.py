# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Text-only DiffusionGemma prompt-to-text demo for #47464 bring-up."""

from __future__ import annotations

import argparse
import os

from loguru import logger
import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.checkpoint import (
    build_tt_model_from_checkpoint_inputs,
    generate_text_from_checkpoint_model_inputs,
    load_checkpoint_inputs,
    load_tokenizer,
    text_generation_prefixes_for_layers,
)


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
    fabric_enabled = rows * cols > 1
    if fabric_enabled:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    try:
        return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
    except Exception:
        if fabric_enabled:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise


def _close_mesh_device(mesh_device) -> None:
    import ttnn

    fabric_enabled = mesh_device.get_num_devices() > 1
    try:
        ttnn.close_mesh_device(mesh_device)
    finally:
        if fabric_enabled:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _log_mesh_dram(mesh_device, label: str) -> None:
    import ttnn

    ttnn.synchronize_device(mesh_device)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    gib = 2**30
    used = view.num_banks * view.total_bytes_allocated_per_bank / gib
    free = view.num_banks * view.total_bytes_free_per_bank / gib
    total = view.num_banks * view.total_bytes_per_bank / gib
    logger.info(
        f"[{label}] per-chip DRAM: used={used:.3f} GiB free={free:.3f} GiB "
        f"usable_total={total:.3f} GiB banks={view.num_banks}"
    )


def _prefill_prompt(checkpoint_model_inputs, prompt):
    from models.experimental.diffusion_gemma.tt.generate import prefill_prompt_tokens, tokenize_prompt

    prompt_tokens = tokenize_prompt(checkpoint_model_inputs.tokenizer, prompt)
    prefill = prefill_prompt_tokens(checkpoint_model_inputs.tt_model, prompt_tokens)
    logger.info(f"[prefill] prompt_len={prefill.prompt_len} cache_len={prefill.cache_len}")
    return prefill


def _infer_vocab_size(tokenizer, tt_model) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    try:
        return len(tokenizer)
    except TypeError:
        pass
    vocab_size = getattr(tt_model, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    hf_config = getattr(tt_model, "hf_config", None)
    vocab_size = getattr(hf_config, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("could not infer vocab size for adapter-only canvas")
    return int(vocab_size)


def _adapter_logits_once(checkpoint_model_inputs, prompt, *, canvas_length: int, seed: int):
    from models.experimental.diffusion_gemma.tt.denoise_forward import (
        make_generation_logits_fn_builder_from_checkpoint_state,
    )
    from models.experimental.diffusion_gemma.tt.generate import (
        host_canvas_to_device,
        prefill_prompt_tokens,
        tokenize_prompt,
    )

    prompt_tokens = tokenize_prompt(checkpoint_model_inputs.tokenizer, prompt)
    prefill = prefill_prompt_tokens(checkpoint_model_inputs.tt_model, prompt_tokens)
    logger.info(f"[prefill] prompt_len={prefill.prompt_len} cache_len={prefill.cache_len}")

    adapter_kwargs = {}
    adapter_config = getattr(checkpoint_model_inputs.tt_model, "hf_config", None)
    if adapter_config is not None:
        adapter_kwargs["config"] = adapter_config
    logits_builder = make_generation_logits_fn_builder_from_checkpoint_state(
        checkpoint_model_inputs.state_dict,
        **adapter_kwargs,
    )
    adapter = logits_builder(
        checkpoint_model_inputs.tt_model,
        prompt_tokens=prompt_tokens,
        prompt_len=prefill.cache_len,
    )

    vocab_size = _infer_vocab_size(checkpoint_model_inputs.tokenizer, checkpoint_model_inputs.tt_model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    host_canvas = torch.randint(0, vocab_size, (1, canvas_length), dtype=torch.long, generator=generator)
    canvas = host_canvas_to_device(checkpoint_model_inputs.tt_model.mesh_device, host_canvas)
    logits = None
    try:
        logits = adapter(canvas, step=0)
        logger.info(f"[adapter] logits_shape={tuple(logits.shape)}")
        return tuple(logits.shape)
    finally:
        if logits is not None:
            adapter.reset()
        canvas.deallocate(True)


def _generation_success_summary(generation) -> str:
    device_generation = generation.generation
    generated_tokens = int(device_generation.generated.shape[-1])
    sequence_len = int(generation.sequences.shape[-1])
    text_chars = sum(len(text) for text in generation.text)
    return (
        "DG_TEXT_DEMO_SUCCESS "
        f"generated_tokens={generated_tokens} "
        f"blocks={len(device_generation.trajectories)} "
        f"prompt_len={device_generation.prompt_len} "
        f"next_pos={device_generation.next_pos} "
        f"sequence_len={sequence_len} "
        f"text_count={len(generation.text)} "
        f"text_chars={text_chars}"
    )


def _parse_success_summary(summary: str) -> dict[str, int]:
    prefix = "DG_TEXT_DEMO_SUCCESS "
    if not summary.startswith(prefix):
        raise ValueError(f"success summary must start with {prefix!r}")
    fields: dict[str, int] = {}
    for item in summary[len(prefix) :].split():
        key, sep, value = item.partition("=")
        if sep != "=" or not key:
            raise ValueError(f"malformed success summary field {item!r}")
        fields[key] = int(value)
    return fields


def _run_mode(args) -> str:
    if args.build_only:
        return "build-only"
    if args.prefill_only:
        return "prefill-only"
    if args.adapter_only:
        return "adapter-only"
    return "generate"


def _failure_summary(args, exc: BaseException) -> str:
    """Single-line, greppable failure marker mirroring DG_TEXT_DEMO_SUCCESS.

    The RUN-first path emits noisy denoise SDPA ``TT_THROW`` fallback lines even
    on success, so a crash is otherwise hard to distinguish from that noise. This
    marker lets a loop/CI grep one line for the run outcome.
    """
    return "DG_TEXT_DEMO_FAILURE " f"mode={_run_mode(args)} " f"mesh={args.mesh} " f"error_type={type(exc).__name__}"


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
    parser.add_argument("--canvas-length", type=int, default=256, help="Diffusion canvas length per generated block")
    parser.add_argument("--max-denoising-steps", type=int, default=48, help="Denoise steps per generated block")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for canvas init and derived noise hooks")
    parser.add_argument(
        "--num-blocks", type=int, default=None, help="Override block count; otherwise derived from max tokens"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=None, help="Optional smoke-test layer limit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true", help="Build the TT model, then exit before generation")
    mode.add_argument("--prefill-only", action="store_true", help="Build the TT model, prefill prompt KV, then exit")
    mode.add_argument(
        "--adapter-only",
        action="store_true",
        help="Build, prefill, run one denoise logits adapter call, then exit",
    )
    parser.add_argument("--local-files-only", action="store_true", help="Do not fetch tokenizer files from HF hub")
    parser.add_argument("--bounded-sliding-kv-cache", action="store_true")
    parser.add_argument(
        "--disable-eos-stop",
        action="store_true",
        help="Do not stop generation or trim decoded output at tokenizer EOS (useful for fixed-block smoke tests)",
    )
    parser.add_argument(
        "--argmax-sampling",
        action="store_true",
        help="Use clean argmax instead of Gumbel-max sampling to avoid full-vocab Gumbel allocation in fit smokes",
    )
    parser.add_argument(
        "--host-gumbel-sampling",
        action="store_true",
        help="Generate Gumbel noise on host and inject it, avoiding device ttnn.rand allocation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return _run(args)
    except BaseException as exc:
        logger.error(_failure_summary(args, exc))
        raise


def _run(args) -> int:
    if args.argmax_sampling and args.host_gumbel_sampling:
        raise ValueError("choose at most one of --argmax-sampling and --host-gumbel-sampling")

    zero_block_run = not args.build_only and (
        args.num_blocks == 0 or (args.num_blocks is None and args.max_new_tokens == 0)
    )
    if zero_block_run:
        from models.experimental.diffusion_gemma.tt.generate import generate_text_from_checkpoint_state

        tokenizer = load_tokenizer(args.checkpoint, local_files_only=args.local_files_only)
        generation = generate_text_from_checkpoint_state(
            object(),
            tokenizer,
            args.prompt,
            dg_state_dict={},
            num_blocks=0 if args.num_blocks == 0 else None,
            max_new_tokens=0,
            batch=args.batch,
        )
        logger.info(_generation_success_summary(generation))
        for text in generation.text:
            print(text)
        return 0

    checkpoint_inputs = load_checkpoint_inputs(
        args.checkpoint,
        tokenizer_kwargs={"local_files_only": args.local_files_only},
        state_prefixes=text_generation_prefixes_for_layers(args.num_layers),
    )

    mesh_device = _open_mesh_device(args.mesh)
    try:
        _log_mesh_dram(mesh_device, "baseline")
        checkpoint_model_inputs = build_tt_model_from_checkpoint_inputs(
            mesh_device,
            checkpoint_inputs,
            max_batch_size=args.batch,
            max_seq_len=args.max_seq_len,
            num_layers=args.num_layers,
            bounded_sliding_kv_cache=args.bounded_sliding_kv_cache,
        )
        _log_mesh_dram(mesh_device, "post-build")
        if args.build_only:
            return 0
        if args.prefill_only:
            _prefill_prompt(checkpoint_model_inputs, args.prompt)
            _log_mesh_dram(mesh_device, "post-prefill")
            return 0
        if args.adapter_only:
            _adapter_logits_once(
                checkpoint_model_inputs,
                args.prompt,
                canvas_length=args.canvas_length,
                seed=args.seed,
            )
            _log_mesh_dram(mesh_device, "post-adapter")
            return 0
        config = DiffusionConfig(canvas_length=args.canvas_length, max_denoise_steps=args.max_denoising_steps)
        generate_kwargs = {}
        if args.disable_eos_stop:
            generate_kwargs.update(eos_token_id=None, stop_token_ids=[], decode_kwargs={"skip_special_tokens": True})
        if args.argmax_sampling:
            generate_kwargs["gumbel_noise_fn"] = lambda block_idx: (lambda step: None)
        if args.host_gumbel_sampling:
            generate_kwargs["use_host_gumbel_noise"] = True
        generation = generate_text_from_checkpoint_model_inputs(
            checkpoint_model_inputs,
            args.prompt,
            config=config,
            max_new_tokens=args.max_new_tokens,
            num_blocks=args.num_blocks,
            seed=args.seed,
            batch=args.batch,
            **generate_kwargs,
        )
    finally:
        _close_mesh_device(mesh_device)

    logger.info(_generation_success_summary(generation))
    for text in generation.text:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
