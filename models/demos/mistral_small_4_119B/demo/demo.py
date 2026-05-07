#!/usr/bin/env python3
"""
Text-only inference demo for Mistral-Small-4-119B-2603.

Use `--reference-torch` to load the vendored PyTorch reference in
`reference/model.py` (same Mistral3 weights as AutoModelForImageTextToText).

Designed for bifrost server:
  - GPU: RTX 3080, 10 GB VRAM
  - RAM: ~64 GB

Quantization options (--quant flag):
  --quant 4   INT4/NF4   ~60 GB   ← DEFAULT  (10 GB GPU + 50 GB CPU)
  --quant 8   INT8       ~119 GB  (10 GB GPU + ~60 GB CPU + disk offload)
  --quant none BF16/FP16 ~238 GB  ← WARNING: will NOT fit on bifrost RAM

Weights live under --model-dir after download; reruns load from disk (no weight re-download).
Optional: MISTRAL_LOAD_LOCAL_ONLY=1 to forbid Hugging Face Hub calls during load.

Usage (from tt-metal repo root, PYTHONPATH implicit if you `cd tt-metal`):
    python3 models/demos/mistral_small_4_119B/demo.py
    python3 models/demos/mistral_small_4_119B/demo.py --quant 4 --reference-torch
    python3 models/demos/mistral_small_4_119B/demo.py --model-dir models/mistral_small_4
"""

import argparse
import ctypes
import gc
import os
import sys
import time


def _setup_cuda_libs() -> None:
    """Pre-load libnvJitLink.so.13 with RTLD_GLOBAL before bitsandbytes imports.

    Setting os.environ["LD_LIBRARY_PATH"] after process start is ignored by
    glibc's dlopen.  The correct in-process fix is to load the library with
    ctypes.RTLD_GLOBAL first: its symbols are then resident in the process
    namespace and dlopen resolves libbitsandbytes_cuda130.so's DT_NEEDED
    dependency on libnvJitLink.so.13 without needing LD_LIBRARY_PATH.
    """
    for search_path in sys.path:
        lib_path = os.path.join(search_path, "nvidia", "cu13", "lib", "libnvJitLink.so.13")
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                print(f"  [setup] Warning: could not pre-load {lib_path}: {exc}")
            cu13_dir = os.path.dirname(lib_path)
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{cu13_dir}:{existing}" if existing else cu13_dir
            break


_setup_cuda_libs()


def _patch_bnb_and_accelerate() -> None:
    """Fix two bugs that crash INT8 + CPU-offload model loading.

    Bug 1 — BNB 0.49.x Linear8bitLt._save_to_state_dict:
      getattr(self.weight, 'SCB') raises AttributeError on CPU-offloaded
      Int8Params because PyTorch tensor-subclass operations can create
      instances that bypass Python __new__, leaving SCB unset.
      Fix: use getattr(..., None) so missing SCB is treated as unquantized.

    Bug 2 — accelerate check_device_map:
      Calls model.state_dict() before dispatch to enumerate all tensors,
      which triggers Bug 1.  The device_map was auto-computed by accelerate,
      so this defensive check is safe to skip entirely.
    """
    try:
        from bitsandbytes.nn.modules import Linear8bitLt

        _orig_save = (
            Linear8bitLt._save_to_state_dict.__func__
            if hasattr(Linear8bitLt._save_to_state_dict, "__func__")
            else Linear8bitLt._save_to_state_dict
        )

        def _patched_save(self, destination, prefix, keep_vars):
            if not hasattr(self.weight, "SCB") or self.weight.SCB is None:
                import torch.nn as nn

                nn.Module._save_to_state_dict(self, destination, prefix, keep_vars)
                return
            _orig_save(self, destination, prefix, keep_vars)

        Linear8bitLt._save_to_state_dict = _patched_save
    except Exception as exc:
        print(f"  [setup] Warning: could not patch Linear8bitLt: {exc}")

    try:
        import accelerate.big_modeling as _abm
        import accelerate.utils.modeling as _aum

        def _noop_check_device_map(model, device_map):
            pass

        _abm.check_device_map = _noop_check_device_map
        _aum.check_device_map = _noop_check_device_map
    except Exception as exc:
        print(f"  [setup] Warning: could not patch check_device_map: {exc}")


_patch_bnb_and_accelerate()

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, TextStreamer

from models.demos.mistral_small_4_119B.bnb_dispatch import (
    accelerate_max_memory,
    build_bnb_config_and_labels,
    cli_quant_label,
    cuda_available,
    resolve_effective_quant,
)
from models.demos.mistral_small_4_119B.config_patches import (
    patch_model_config_for_quantization,
    patch_tokenizer_config,
    restore_fp8_quantization_config_if_missing,
    sync_generation_config_with_text_config,
)
from models.demos.mistral_small_4_119B.env_check import ensure_transformers_has_mistral4
from models.demos.mistral_small_4_119B.mistral_moe_fp8_materialize import materialize_mistral_moe_fp8_experts_from_disk
from models.demos.mistral_small_4_119B.moe_grouped_mm_patch import apply_moe_grouped_mm_fp8_fallback_fix
from models.demos.mistral_small_4_119B.paths import DEFAULT_MODEL_DIR, from_pretrained_local_kw, resolve_model_dir
from models.demos.mistral_small_4_119B.processor_layout import ensure_preprocessor_config_json

apply_moe_grouped_mm_fp8_fallback_fix()

# Memory footprint per mode
# NOTE: BNB INT4 does NOT support CPU/disk offloading — all quantised layers
# must fit in GPU VRAM.  On bifrost (10 GB VRAM) the ~60 GB INT4 model will
# never fit, so --quant 4 falls back to INT8 with CPU offload (the only BNB
# mode that allows dispatching modules to CPU/disk).
_QUANT_INFO = {
    "4": "INT4 NF4 on GPU; **without CUDA** → auto BF16 on CPU (likely OOM for 119B)",
    "8": "INT8 on GPU; **without CUDA** → auto BF16 on CPU (likely OOM for 119B)",
    "none": "BF16 on GPU with offload, or BF16 on CPU if no CUDA (huge RAM)",
}


def load_model(model_dir: str, quant: str, reference_torch: bool = False):
    assert quant in _QUANT_INFO, f"--quant must be one of: {list(_QUANT_INFO)}"

    ensure_transformers_has_mistral4()

    model_dir = str(resolve_model_dir(model_dir))

    if cuda_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"  [cuda] device=0  name={props.name}  "
            f"total_mem={props.total_memory / (1024**3):.1f} GiB  "
            f"multi_processor_count={props.multi_processor_count}"
        )
    else:
        print("  [cuda] No CUDA GPU visible — INT4/INT8 will fall back to BF16 on CPU (see [quantize] lines below).")

    _local_kw = from_pretrained_local_kw()
    if _local_kw:
        print("  [cache] MISTRAL_LOAD_LOCAL_ONLY — load from disk only (no Hub); needs a complete snapshot.")

    patch_tokenizer_config(model_dir)
    ensure_preprocessor_config_json(model_dir)

    print(f"Loading processor from {model_dir} ...")
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, **_local_kw)
    tokenizer = processor.tokenizer

    if reference_torch:
        from models.demos.mistral_small_4_119B.reference.model import Mistral3ForConditionalGeneration

        model_cls = Mistral3ForConditionalGeneration
        print("  [reference-torch] Using vendored Mistral3ForConditionalGeneration (see reference/model.py)")
    else:
        model_cls = AutoModelForImageTextToText

    print(f"Loading model  (--quant {quant})  {_QUANT_INFO[quant]}")
    print("  Weights are read from the local --model-dir (no re-download once snapshot is complete).")

    effective_quant, quant_fallback_msg = resolve_effective_quant(quant)
    if quant_fallback_msg:
        print(quant_fallback_msg)

    bnb_cfg, _, label_effective = build_bnb_config_and_labels(effective_quant)
    print(f"  [quantize] CLI request : {cli_quant_label(quant)}")
    print(f"  [quantize] Effective : --quant {effective_quant} → {label_effective}")

    patch_model_config_for_quantization(model_dir, strip_fp8_for_bnb=bnb_cfg is not None)
    if bnb_cfg is None:
        restore_fp8_quantization_config_if_missing(
            model_dir,
            local_files_only=_local_kw.get("local_files_only", False),
        )

    load_torch_dtype = torch.float32 if (bnb_cfg is None and not cuda_available()) else torch.bfloat16

    if bnb_cfg is None:
        print("  WARNING: --quant none needs ~238 GiB RAM/VRAM for this checkpoint.")
        if cuda_available():
            model = model_cls.from_pretrained(
                model_dir,
                dtype=load_torch_dtype,
                device_map="auto",
                max_memory=accelerate_max_memory(),
                offload_folder="./offload",
                offload_state_dict=True,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                **_local_kw,
            )
        else:
            print("  [device] float32 on CPU only (no GPU); expect OOM unless you have hundreds of GiB RAM.")
            model = model_cls.from_pretrained(
                model_dir,
                dtype=load_torch_dtype,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                **_local_kw,
            )
    else:
        print("  First INT8/INT4 load can take a long time while bitsandbytes prepares weights.")
        model = model_cls.from_pretrained(
            model_dir,
            quantization_config=bnb_cfg,
            device_map="auto",
            max_memory=accelerate_max_memory(),
            offload_folder="./offload",
            offload_state_dict=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **_local_kw,
        )

    model.eval()
    try:
        if materialize_mistral_moe_fp8_experts_from_disk(model, model_dir):
            print("  [patched] MoE routed experts dequantized from safetensors (fixes CPU garbage logits).")
    except Exception as exc:
        print(f"  [warn] MoE FP8 expert materialize skipped: {exc}")
    sync_generation_config_with_text_config(model)
    print("Model loaded.\n")
    return tokenizer, model


def run_inference(tokenizer, model, prompt: str, max_new_tokens: int = 256):
    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt")

    # Move input to same device as first model parameter
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")
    print("Response:")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    n_new = output_ids.shape[-1] - inputs["input_ids"].shape[-1]
    print(f"\n[Generated {n_new} tokens in {elapsed:.1f}s — {n_new/elapsed:.1f} tok/s]")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


DEMO_PROMPTS = [
    "What is Mixture of Experts in large language models? Explain in 3 bullet points.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the key differences between attention mechanisms in transformers?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help=f"Path to downloaded weights (default: {DEFAULT_MODEL_DIR}; "
        "also accepts legacy tt-metal/models/mistral_small_4 if present)",
    )
    parser.add_argument("--prompt", default=None, help="Custom prompt (default: runs 3 demo prompts)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--quant",
        choices=["4", "8", "none"],
        default="4",
        help="Quantization: 4=INT4/NF4 (~60 GB, default), 8=INT8 (~119 GB), none=BF16 (~238 GB)",
    )
    parser.add_argument(
        "--reference-torch",
        action="store_true",
        help="Load Mistral3ForConditionalGeneration from reference/model.py instead of AutoModel",
    )
    args = parser.parse_args()

    tokenizer, model = load_model(
        model_dir=args.model_dir,
        quant=args.quant,
        reference_torch=args.reference_torch,
    )

    prompts = [args.prompt] if args.prompt else DEMO_PROMPTS
    for p in prompts:
        run_inference(tokenizer, model, p, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
