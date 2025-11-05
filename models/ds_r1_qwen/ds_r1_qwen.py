"""
Pure TTNN implementation hook for DeepSeek-R1-Distill-Qwen-1.5B.

This module wires the existing TT-Transformers TTNN kernels/blocks to
instantiate a Qwen (DeepSeek-R1-Distill-Qwen-1.5B) model on a Tenstorrent mesh.

It provides a small, focused wrapper to:
- Build ModelArgs from an HF repo ID or a local checkpoint directory
- Load and standardize weights using TT-Transformers helpers
- Construct a TTNN Transformer ready for prefill/decode with KV cache
- Optionally create a Generator compatible with TT-Transformers sampling flows

Notes
- This is pure TTNN end-to-end: embeddings, attention, MLPs, RMSNorm, KV cache
  and matmuls all run through TTNN via models.tt_transformers.tt.* modules.
- Network access is not required if the weights are locally available via
  a local HF snapshot directory (or set in HF cache). Pass a local path to
  `checkpoint_dir` to avoid network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

from loguru import logger

import ttnn
from models.tt_transformers.tt.generator import Generator as TTGenerator
from models.tt_transformers.tt.model import Transformer as TTTransformer
from models.tt_transformers.tt.model_config import ModelArgs, ModelOptimizations

DEFAULT_HF_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@dataclass
class DSQwenBuildConfig:
    mesh_device: ttnn.MeshDevice
    checkpoint_dir: Optional[str] = None  # local path to HF snapshot or meta-style weights
    hf_model_id: str = DEFAULT_HF_ID
    max_batch_size: int = 1
    max_seq_len: int = 32768
    dtype: ttnn.DataType = ttnn.bfloat16
    accuracy_mode: bool = False
    use_paged_kv_cache: bool = False
    cache_hf_model: bool = False  # cache HF model object in memory when loading from HF


def _select_optimizations(accuracy_mode: bool):
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    return ModelOptimizations.accuracy(model_name) if accuracy_mode else ModelOptimizations.performance(model_name)


def _set_checkpoint_env(checkpoint_dir: Optional[str], hf_model_id: str) -> None:
    """Configure environment for ModelArgs to discover model location.

    ModelArgs reads either LLAMA_DIR (Meta-format) or HF_MODEL (HF-format) to
    locate config/weights. For DS-R1-Qwen we use HF format.
    """
    if checkpoint_dir:
        # Local HF snapshot path or directory containing config.json/model.safetensors
        os.environ["HF_MODEL"] = checkpoint_dir
        logger.info(f"Using local HF path for DS-R1-Qwen: {checkpoint_dir}")
    else:
        # HF repo ID (requires files to be locally cached or available offline)
        os.environ["HF_MODEL"] = hf_model_id
        logger.info(f"Using HF repo for DS-R1-Qwen: {hf_model_id}")


def build_model(cfg: DSQwenBuildConfig) -> Tuple[ModelArgs, TTTransformer]:
    """Build DS-R1-Distill-Qwen-1.5B TTNN model using TT-Transformers blocks.

    Returns
    - (args, model): ModelArgs and constructed TTTransformer
    """
    _set_checkpoint_env(cfg.checkpoint_dir, cfg.hf_model_id)
    optim = _select_optimizations(cfg.accuracy_mode)

    # Initialize model arguments (reads HF config, sets dims/heads/rope/etc.)
    args = ModelArgs(
        mesh_device=cfg.mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=cfg.max_batch_size,
        max_seq_len=cfg.max_seq_len,
        optimizations=optim,
        cache_hf=cfg.cache_hf_model,
    )

    # Load and standardize weights (supports local HF snapshot path or HF cache)
    state_dict = args.load_state_dict()

    # Construct the TTNN Transformer
    model = TTTransformer(
        args=args,
        dtype=cfg.dtype,
        mesh_device=cfg.mesh_device,
        state_dict=state_dict,
        weight_cache_path=args.weight_cache_path(cfg.dtype),
        use_paged_kv_cache=cfg.use_paged_kv_cache,
    )

    return args, model


def create_generator(
    model: TTTransformer,
    args: ModelArgs,
    mesh_device: ttnn.MeshDevice,
    processor=None,
    tokenizer=None,
) -> TTGenerator:
    """Create a TT-Transformers Generator for device-side sampling/decoding."""
    return TTGenerator([model], [args], mesh_device, processor=processor, tokenizer=tokenizer)


# Convenience one-liner for typical use
def init_ds_r1_qwen_1_5b(
    mesh_device: ttnn.MeshDevice,
    checkpoint_dir: Optional[str] = None,
    hf_model_id: str = DEFAULT_HF_ID,
    max_batch_size: int = 1,
    max_seq_len: int = 32768,
    dtype: ttnn.DataType = ttnn.bfloat16,
    accuracy_mode: bool = False,
    use_paged_kv_cache: bool = False,
    cache_hf_model: bool = False,
) -> Tuple[ModelArgs, TTTransformer]:
    """Initialize DS-R1-Distill-Qwen-1.5B model+args.

    - If `checkpoint_dir` is provided, uses it as a local HF snapshot path.
    - Otherwise uses the HF repo ID and relies on local cache availability.
    - Adjust `accuracy_mode` to trade performance for fidelity.
    - `dtype` controls activations/weights precision where applicable.
    """
    cfg = DSQwenBuildConfig(
        mesh_device=mesh_device,
        checkpoint_dir=checkpoint_dir,
        hf_model_id=hf_model_id,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype=dtype,
        accuracy_mode=accuracy_mode,
        use_paged_kv_cache=use_paged_kv_cache,
        cache_hf_model=cache_hf_model,
    )
    return build_model(cfg)


__all__ = [
    "DSQwenBuildConfig",
    "build_model",
    "create_generator",
    "init_ds_r1_qwen_1_5b",
]


def set_fabric(fabric_config, reliability_mode=None, fabric_tensix_config=None):
    # If fabric_config is not None, set it to fabric_config
    if fabric_config:
        if reliability_mode is None:
            reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT

        # Apply default logic for fabric_tensix_config,
        # fabric_tensix_config is used for enabling tensix extensions for the fabric router,
        # some sender channels in the fabric router are moved to the fabric tensix extension
        # (currently the extension is mux kernel, can have other kernels in future as well).
        if fabric_tensix_config is None:
            fabric_tensix_config = get_default_fabric_tensix_config()

        ttnn.set_fabric_config(fabric_config, reliability_mode, None, fabric_tensix_config)  # num_planes


def get_default_fabric_tensix_config():
    # Default to MUX for Blackhole when fabric is enabled, DISABLED otherwise
    if ttnn.device.is_blackhole():
        return ttnn.FabricTensixConfig.MUX
    else:
        return ttnn.FabricTensixConfig.DISABLED


def main():
    """Simple CLI to run a prompt on DS-R1-Distill-Qwen-1.5B using TTNN.

    Example
    - python -m models.ds_r1_qwen.ds_r1_qwen \
        --checkpoint-dir /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
        --prompt "Explain backpropagation briefly." --max-gen-len 128
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser("DeepSeek-R1-Distill-Qwen-1.5B (TTNN)")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Local HF snapshot directory with config+weights"
    )
    parser.add_argument("--hf-model-id", type=str, default=DEFAULT_HF_ID, help="HF repo id (use only if cached)")
    parser.add_argument("--mesh-rows", type=int, default=1)
    parser.add_argument("--mesh-cols", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="Hello! Explain what TTNN is.")
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1, help="Decode batch (must equal B for decode)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "bf8"],
        help="Model compute/weight dtype preference",
    )
    parser.add_argument("--accuracy-mode", action="store_true", help="Use accuracy-optimized settings")
    parser.add_argument("--paged-kv", action="store_true", help="Use paged KV cache (for external schedulers)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--stream", action="store_true", help="Stream token outputs instead of single message")

    args_cli = parser.parse_args()

    # Map dtype string to TTNN dtype
    dtype_map = {
        "bf16": ttnn.bfloat16,
        "bf8": ttnn.bfloat8_b,
    }
    dtype = dtype_map[args_cli.dtype]

    # Open mesh and build model
    mesh_shape = ttnn.MeshShape([1, 1])
    mesh = None
    try:
        set_fabric(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(mesh_shape=mesh_shape)

        ds_args, model = init_ds_r1_qwen_1_5b(
            mesh_device=mesh,
            checkpoint_dir=args_cli.checkpoint_dir,
            hf_model_id=args_cli.hf_model_id,
            max_batch_size=args_cli.batch_size,
            max_seq_len=args_cli.max_seq_len,
            dtype=dtype,
            accuracy_mode=args_cli.accuracy_mode,
            use_paged_kv_cache=args_cli.paged_kv,
        )

        # Create generator (uses tokenizer from ModelArgs)
        gen = create_generator(model, ds_args, mesh, tokenizer=ds_args.tokenizer)

        if args_cli.stream:
            # Stream tokens using generate()
            prompt_tokens = ds_args.encode_prompt(
                args_cli.prompt, system_prompt_text=args_cli.system_prompt, instruct=True
            )
            text_out = []
            for tok in gen.generate(
                vision_images=None,
                vision_mask=None,
                prompt_tokens=prompt_tokens,
                max_gen_len=args_cli.max_gen_len,
                temperature=args_cli.temperature,
                top_p=args_cli.top_p,
            ):
                sys.stdout.write(tok.text)
                sys.stdout.flush()
                text_out.append(tok.text)
            sys.stdout.write("\n")
        else:
            # Single message via chat template
            messages = []
            if args_cli.system_prompt:
                messages.append({"role": "system", "content": args_cli.system_prompt})
            messages.append({"role": "user", "content": args_cli.prompt})
            msg = gen.chat_completion(
                messages,
                temperature=args_cli.temperature,
                top_p=args_cli.top_p,
                max_gen_len=args_cli.max_gen_len,
            )
            print(msg.message)
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
