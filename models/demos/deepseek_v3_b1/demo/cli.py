# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Literal

from loguru import logger
from transformers import AutoTokenizer

import ttnn
from conftest import bh_2d_mesh_device_context
from models.demos.deepseek_v3_b1.demo.model_pipeline import ModelPipeline
from models.demos.deepseek_v3_b1.demo.pipeline import create_fabric_router_config

DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-R1-0528"


def _fabric_config_for_num_procs(num_procs: int):
    """Infer fabric config from process count: 4 → FABRIC_2D, 16 → FABRIC_2D_TORUS_Y."""
    if num_procs == 4:
        return ttnn.FabricConfig.FABRIC_2D
    if num_procs == 16:
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    if num_procs == 64:
        return ttnn.FabricConfig.FABRIC_2D_TORUS_Y
    raise ValueError(f"Unsupported num_procs for fabric config: {num_procs} (expected 4, 16, or 64)")


@contextlib.contextmanager
def open_mesh_device():
    my_rank = int(ttnn.distributed_context_get_rank())
    num_procs = int(ttnn.distributed_context_get_size())
    worker_l1_size = 1431568
    if num_procs == 64:
        if my_rank == 62:
            worker_l1_size = 1499000
    elif num_procs == 16:
        if my_rank == 14:
            worker_l1_size = 1499000
    """Open mesh device using bh_2d_mesh_device_context (pod pipeline settings)."""
    if not os.environ.get("TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"):
        os.environ["TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS"] = "30000"
    num_procs = int(ttnn.distributed_context_get_size())
    device_params = {
        "fabric_config": _fabric_config_for_num_procs(num_procs),
        "fabric_router_config": create_fabric_router_config(15232),
        "worker_l1_size": worker_l1_size,
    }
    logger.info("Opening mesh device...")
    with bh_2d_mesh_device_context(device_params) as mesh_device:
        logger.info(
            "Mesh device opened (id={}, shape={})",
            mesh_device.get_system_mesh_id(),
            mesh_device.shape,
        )
        yield mesh_device


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
        "--bspm-variant",
        type=str,
        choices=("B",),
        default="B",
        help="BitSculpt allocation variant letter (default: B)",
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
        "--launch-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only launch the pipeline, export H2D/D2H socket descriptors on mesh id 0, and keep the pipeline alive.",
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
        "--kv-cache-dump-dir",
        type=Path,
        default=None,
        help=(
            "If set, after the pipeline shuts down each rank dumps its on-device KV cache "
            "(decoder stages only) as torch tensor binaries into this directory."
        ),
    )
    return parser


def load_tokenizer(tokenizer_name_or_path: str):
    return AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)


def run_demo(
    *,
    prompt: str,
    max_new_tokens: int,
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
    bspm_dir: Path | None = None,
    bspm_variant: str = "B",
    bspm_budget: float = 3.5,
    kv_cache_dump_dir: Path | None = None,
) -> None:
    """Run the pod pipeline. Requires 4, 16, or 64 distributed processes."""
    iterations = max_new_tokens
    logger.info(f"Starting DeepSeek V3 B1 demo (iterations={iterations})")

    with open_mesh_device() as mesh_device:
        # Initialize model pipeline
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
            bspm_dir=bspm_dir,
            bspm_variant=bspm_variant,
            bspm_budget=bspm_budget,
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
            if not prompt_ids:
                raise RuntimeError("Chat template produced an empty prompt")
            logger.debug(f"Encoded prompt: {prompt_ids}")

            logger.info("Running inference on prompt with {} tokens", len(prompt_ids))
            generated_tokens = model_pipeline.run_inference(
                prompt_token_ids=prompt_ids,
                max_new_tokens=iterations,
                eos_token_id=tokenizer.eos_token_id,
                return_generated_tokens=True,
            )
            assert generated_tokens is not None
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logger.info("Output ({} tokens): {}", len(generated_tokens), generated_text)

        if launch_only and my_mesh_id == 0:
            # Only runs on mesh 0, all other processes wait for a barrier
            if kv_cache_dump_dir is not None:
                # Sentinel-file trigger: robust to mpirun signal-stealing.
                kv_cache_dump_dir.mkdir(parents=True, exist_ok=True)
                sentinel = (kv_cache_dump_dir / ".dump_now").resolve()
                logger.info(f"Pipeline launched. To dump KV cache and shut down: touch {sentinel}")
                while not sentinel.exists():
                    time.sleep(1)
                try:
                    sentinel.unlink()
                except OSError:
                    pass
                logger.info("Sentinel detected; shutting down launch-only pipeline.")
            else:
                logger.info("Pipeline launched; keeping sockets alive until interrupted.")
                try:
                    while True:
                        time.sleep(3600)
                except KeyboardInterrupt:
                    logger.info("Shutting down launch-only pipeline after interrupt.")
        model_pipeline.barrier()

        logger.info("Pod pipeline complete - terminating now...")
        model_pipeline.terminate()

        print("Done running inference")
        if kv_cache_dump_dir is not None:
            try:
                print("Dumping KV cache")
                with ttnn.device.setup_fast_dispatch(mesh_device):
                    model_pipeline.dump_kv_cache(kv_cache_dump_dir)
                print("KV cache dumped")
            except Exception:
                logger.exception("KV cache dump failed on this rank")


def main(argv: list[str] | None = None) -> int:
    ttnn.init_distributed_context()
    parser = create_parser()
    args = parser.parse_args(argv)

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

    io_socket_descriptor_prefix = args.io_socket_descriptor_prefix
    if args.launch_only and io_socket_descriptor_prefix is None:
        io_socket_descriptor_prefix = "deepseek"

    run_demo(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
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
        bspm_dir=args.bspm_dir,
        bspm_variant=args.bspm_variant,
        bspm_budget=args.bspm_budget,
        kv_cache_dump_dir=args.kv_cache_dump_dir,
    )
    print(file=sys.stdout, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
