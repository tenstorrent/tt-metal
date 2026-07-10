# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Literal

from loguru import logger
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-0528"
SLOW_DISPATCH_ENV = "TT_METAL_SLOW_DISPATCH_MODE"
HYBRID_ALLOCATOR_ENV = "TT_METAL_ALLOCATOR_MODE_HYBRID"


def configure_runtime_env(*, enable_sram_hot_experts: bool) -> None:
    """Set demo-required TT-Metal environment before TTNN/device initialization."""
    if os.environ.get(SLOW_DISPATCH_ENV) != "1":
        os.environ[SLOW_DISPATCH_ENV] = "1"
        logger.info("Enabled {}=1 for the DeepSeek demo", SLOW_DISPATCH_ENV)

    if enable_sram_hot_experts and os.environ.get(HYBRID_ALLOCATOR_ENV) != "1":
        os.environ[HYBRID_ALLOCATOR_ENV] = "1"
        logger.info("Enabled {}=1 for --enable-sram-hot-experts", HYBRID_ALLOCATOR_ENV)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DeepSeek-V3-B1 Demo on TT-NN (pod pipeline)")
    parser.add_argument("--prompt", type=str, default="Hello, world!", help="Prompt text (for future real decode loop)")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of pipeline token iterations",
    )
    parser.add_argument(
        "--repeat-generations",
        type=int,
        default=1,
        help="Number of complete prompt+generation runs to execute sequentially on the same pipeline.",
    )
    parser.add_argument(
        "--stop-at-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop each generation at tokenizer EOS. Use --no-stop-at-eos for fixed-length stress runs.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="HF tokenizer id or local tokenizer path",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Path to the weight cache directory (required for --weights real)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local HuggingFace model dir with model.safetensors.index.json (required for --weights real/state_dict)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=("synthetic", "real", "state_dict"),
        default="real",
        help="synthetic: random prepare path; real: TensorCache + HF safetensors; state_dict: HF safetensors + prepare path (no cache)",
    )
    parser.add_argument(
        "--fp32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use FP32 destination accumulator for LMHead sampling",
    )
    parser.add_argument(
        "--persistent-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use persistent mode for LMHead sampling kernel",
    )
    parser.add_argument(
        "--dense-layer-id-override",
        type=int,
        default=None,
        metavar="ID",
        help="Force all dense stages to use this layer id (e.g. 0); default: use 0,1,2",
    )
    parser.add_argument(
        "--moe-layer-id-override",
        type=int,
        default=None,
        metavar="ID",
        help="Force all MoE stages to use this layer id (e.g. 3); default: use stage-dependent layer ids",
    )
    parser.add_argument(
        "--bspm-dir",
        type=Path,
        default=None,
        help=(
            "Model-specific BitSculpt BSPM directory, e.g. results/deepseek-r1-0528. "
            "MoE layers look up layer_<id>/precision_eval/precision_map_<variant>_<budget>.bspm under this path."
        ),
    )
    parser.add_argument(
        "--bspm-budget",
        type=float,
        default=3.5,
        help="BitSculpt bit budget per expert used in the BSPM filename (default: 3.5)",
    )
    parser.add_argument(
        "--num-slots",
        type=int,
        default=64,
        help="Number of users/slots (KV cache batch size) for the decoder stages",
    )
    parser.add_argument(
        "--relaxed-acceptance-delta",
        type=float,
        default=0.6,
        help="Relaxed acceptance delta for the MTP verification stage",
    )
    parser.add_argument(
        "--enable-sram-hot-experts",
        action="store_true",
        help=(
            "Pin the highest-frequency routed experts to per-core L1 via "
            "prepare_compressed_sram_slots. Automatically enables TT_METAL_ALLOCATOR_MODE_HYBRID=1."
        ),
    )
    parser.add_argument(
        "--sram-hot-experts-ceiling",
        type=int,
        default=64,
        help="Maximum number of SRAM-pinned hot experts per MoE layer (top-N by routing frequency).",
    )
    parser.add_argument(
        "--enable-sram-bspm",
        action="store_true",
        help=(
            "Use the BSPM precision map for SRAM hot expert weights too "
            "(default: uniform BFP4). Requires --bspm-dir to be set."
        ),
    )
    parser.add_argument(
        "--launch-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only launch the pipeline, export H2D/D2H socket descriptors on mesh id 0, and keep the pipeline alive.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Top-k sampling for the LM head weights (only for real weights)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling for the LM head weights (only for real weights)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for softmax in probablistic sampling",
    )
    parser.add_argument(
        "--num-mtp-levels",
        type=int,
        default=1,
        help="Number of MTP stages to use for the pipeline",
    )
    parser.add_argument(
        "--io-socket-descriptor-prefix",
        type=str,
        default=None,
        help=(
            "If set, export H2D/D2H socket descriptors on mesh 0 after pipeline setup "
            "(files named <prefix>_h2d / <prefix>_d2h). When --launch-only is used and "
            "this is omitted, defaults to deepseek."
        ),
    )
    parser.add_argument(
        "--enable-speculative-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable speculative decode; use --no-enable-speculative-decode for base decode",
    )

    return parser


def load_tokenizer(tokenizer_name_or_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)


def run_demo(
    *,
    prompt: str,
    max_new_tokens: int,
    repeat_generations: int,
    stop_at_eos: bool,
    tokenizer_name_or_path: str,
    weights_mode: Literal["synthetic", "real", "state_dict"] = "real",
    cache_path: Path | None = None,
    model_path: Path | None = None,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
    dense_layer_id_override: int | None = None,
    moe_layer_id_override: int | None = None,
    launch_only: bool = False,
    io_socket_descriptor_prefix: str | None = None,
    num_slots: int = 64,
    relaxed_acceptance_delta: float = 0.6,
    top_k: int = 1,
    top_p: float = 1.0,
    temperature: float = 0.6,
    num_mtp_levels: int = 1,
    enable_sram_hot_experts: bool = False,
    sram_hot_experts_ceiling: int = 64,
    bspm_dir: Path | None = None,
    bspm_budget: float = 3.5,
    enable_sram_bspm: bool = False,
) -> None:
    """Run the pod pipeline. Requires 4, 16, or 64 distributed processes."""
    configure_runtime_env(enable_sram_hot_experts=enable_sram_hot_experts)

    from models.demos.deepseek_v3_b1.demo.mesh_device_context import open_mesh_device
    from models.demos.deepseek_v3_b1.demo.model_pipeline import ModelPipeline

    iterations = max_new_tokens
    logger.info(
        "Starting DeepSeek V3 B1 demo (iterations={}, repeat_generations={}, stop_at_eos={})",
        iterations,
        repeat_generations,
        stop_at_eos,
    )

    with open_mesh_device(num_mtp_levels=num_mtp_levels) as mesh_device:
        model_pipeline = ModelPipeline(
            mesh_device=mesh_device,
            weights_mode=weights_mode,
            cache_path=cache_path,
            model_path=model_path,
            lm_head_fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            lm_head_persistent_mode=lm_head_persistent_mode,
            dense_layer_id_override=dense_layer_id_override,
            moe_layer_id_override=moe_layer_id_override,
            io_socket_descriptor_prefix=io_socket_descriptor_prefix,
            num_slots=num_slots,
            relaxed_acceptance_delta=relaxed_acceptance_delta,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_mtp_levels=num_mtp_levels,
            enable_sram_hot_experts=enable_sram_hot_experts,
            sram_hot_experts_ceiling=sram_hot_experts_ceiling,
            bspm_dir=bspm_dir,
            bspm_budget=bspm_budget,
            enable_sram_bspm=enable_sram_bspm,
        )

        my_mesh_id = mesh_device.get_system_mesh_id()
        if my_mesh_id == 0 and not launch_only:
            tokenizer = load_tokenizer(tokenizer_name_or_path)
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            logger.debug("Prompt with chat template: {}", prompt)

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            think_open_id = tokenizer.encode("<think>", add_special_tokens=False)
            think_close_id = tokenizer.encode("</think>", add_special_tokens=False)
            if len(think_open_id) != 1 or len(think_close_id) != 1:
                raise RuntimeError("Thinking token IDs must be single tokens")
            if not prompt_ids:
                raise RuntimeError("Chat template produced an empty prompt")
            logger.debug(f"Encoded prompt: {prompt_ids}")

            eos_token_id = tokenizer.eos_token_id if stop_at_eos else None
            for repeat_idx in range(repeat_generations):
                logger.info(
                    "Running generation {}/{} on prompt with {} tokens",
                    repeat_idx + 1,
                    repeat_generations,
                    len(prompt_ids),
                )
                generated_tokens = model_pipeline.run_inference(
                    prompt_token_ids=prompt_ids,
                    max_new_tokens=iterations,
                    eos_token_id=eos_token_id,
                    think_token_ids=[think_open_id[0], think_close_id[0]],
                    return_generated_tokens=True,
                )
                assert generated_tokens is not None
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                logger.info(
                    "Generation {}/{} output ({} tokens): {}",
                    repeat_idx + 1,
                    repeat_generations,
                    len(generated_tokens),
                    generated_text,
                )

        if launch_only and my_mesh_id == 0:
            # Keep process/pipeline alive until user interrupts
            # Only runs on mesh 0, all other processes wait for a barrier
            logger.info("Pipeline launched; keeping sockets alive until interrupted.")
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                logger.info("Shutting down launch-only pipeline after interrupt.")

        model_pipeline.barrier()

        logger.info("Pod pipeline complete - terminating now...")
        model_pipeline.terminate()


def main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.repeat_generations < 1:
        parser.error("--repeat-generations must be >= 1")
    if args.weights == "real":
        if args.cache_path is None:
            parser.error("--cache-path is required when --weights real")
        if args.model_path is None:
            parser.error("--model-path is required when --weights real")
    if args.weights in ("real", "state_dict"):
        if args.model_path is None:
            parser.error(f"--model-path is required when --weights {args.weights}")
        index_path = args.model_path / "model.safetensors.index.json"
        if not index_path.is_file():
            parser.error(f"--model-path must contain model.safetensors.index.json (missing {index_path})")

    configure_runtime_env(enable_sram_hot_experts=args.enable_sram_hot_experts)

    import ttnn

    ttnn.init_distributed_context()

    io_socket_descriptor_prefix = args.io_socket_descriptor_prefix
    if args.launch_only and io_socket_descriptor_prefix is None:
        io_socket_descriptor_prefix = "deepseek"

    run_demo(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        repeat_generations=args.repeat_generations,
        stop_at_eos=args.stop_at_eos,
        tokenizer_name_or_path=args.tokenizer,
        weights_mode=args.weights,
        cache_path=args.cache_path,
        model_path=args.model_path,
        lm_head_fp32_dest_acc_en=args.fp32,
        lm_head_persistent_mode=args.persistent_mode,
        dense_layer_id_override=args.dense_layer_id_override,
        moe_layer_id_override=args.moe_layer_id_override,
        launch_only=args.launch_only,
        io_socket_descriptor_prefix=io_socket_descriptor_prefix,
        num_slots=args.num_slots,
        relaxed_acceptance_delta=args.relaxed_acceptance_delta,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        num_mtp_levels=args.num_mtp_levels,
        enable_sram_hot_experts=args.enable_sram_hot_experts,
        sram_hot_experts_ceiling=args.sram_hot_experts_ceiling,
        bspm_dir=args.bspm_dir,
        bspm_budget=args.bspm_budget,
        enable_sram_bspm=args.enable_sram_bspm,
    )
    print(end="", file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
