# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmark for Dots OCR (HF vs TTNN).

TTNN timing matches ``models.demos.dots_ocr.demo.demo.run_ttnn_backend``:
same ``DotsModelArgs`` / weight cache keys, device fusion (default), host RoPE
(``text_rope_from_hf``), ``generator.prefill_forward_text``, and ``_decode_loop``
(greedy TTNN argmax decode).

Measures:
  - ``ttft``          — prefill only (first-token logits on host).
  - ``decode_tok_s``  — generated token count / decode-phase wall time (same definition as demo).
  - ``avg_latency``   — total wall time per iteration (prefill + decode).

Uses the same timing columns as other VLM/decoder TT demos (ttft, decode_tok_s, etc.).

When ``--image`` is omitted, the benchmark loads ``demo/benchmark_image.png`` (shipped
under ``models/demos/dots_ocr/demo/``) so HF and TTNN paths exercise vision like the demo.
Use ``--text-only`` to skip that default and run text-only.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from loguru import logger
from PIL import Image

from models.demos.dots_ocr.logging_utils import configure_dots_ocr_console_logging
from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec, get_hf_model_id
from models.demos.dots_ocr.reference.model import DotsOCRReference


def _dots_ocr_default_benchmark_image() -> Path | None:
    """First existing file among shipped demo assets (prefer ``benchmark_image.png``)."""
    demo_dir = Path(__file__).resolve().parents[1] / "demo"
    for name in ("benchmark_image.png", "image.png", "test12.png"):
        p = demo_dir / name
        if p.is_file():
            return p
    return None


def _summarize(backend: str, metrics: list[dict]) -> dict:
    avg = sum(m["total"] for m in metrics) / len(metrics)
    ttft = sum(m["ttft"] for m in metrics) / len(metrics)
    dec_s = sum(m.get("decode_s", 0.0) for m in metrics) / len(metrics)
    dec_tok = sum(m.get("decode_tokens", 0) for m in metrics) / len(metrics)
    decode_tok_s = dec_tok / dec_s if dec_s > 0 else float("nan")
    fps = 1.0 / avg if avg > 0 else float("inf")
    print(f"\n=== {backend.upper()} ===")
    print(f"Avg total latency: {avg:.4f}s")
    print(f"FPS:               {fps:.2f}")
    print(f"TTFT (prefill):    {ttft:.4f}s")
    print(f"Decode:            {decode_tok_s:.1f} tok/s (avg {dec_tok:.0f} tokens in {dec_s:.3f}s)")
    print(f"Iters:             {len(metrics)}")
    return {
        "backend": backend,
        "avg_latency": avg,
        "fps": fps,
        "ttft": ttft,
        "decode_tok_s": decode_tok_s,
        "decode_tokens": dec_tok,
    }


# ---------------------------------------------------------------------------
# HF backend — single generate call
# ---------------------------------------------------------------------------


def benchmark_hf(model_id: str, image_path: str, prompt: str, max_new_tokens: int, warmup: int, iters: int) -> dict:
    logger.info("Running HF reference benchmark...")
    ref = DotsOCRReference(HFLoadSpec(model_id=model_id))
    image = None
    if image_path and os.path.isfile(image_path):
        image = Image.open(image_path).convert("RGB")
    inputs = ref.preprocess_image_and_prompt(image, prompt)

    for _ in range(warmup):
        _ = ref.forward(inputs, max_new_tokens=max_new_tokens)

    metrics: list[dict] = []
    for i in range(iters):
        t0 = time.perf_counter()
        out = ref.forward(inputs, max_new_tokens=max_new_tokens)
        elapsed = time.perf_counter() - t0
        n_new = int(out["generated_ids"].shape[-1]) - int(inputs.input_ids.shape[-1])
        metrics.append(
            {
                "total": elapsed,
                # HF path lumps everything into `generate`; approximate TTFT as total / (new_tokens+1).
                "ttft": elapsed / max(n_new + 1, 1),
                "decode_s": elapsed,
                "decode_tokens": n_new,
            }
        )
        logger.info(f"HF iter {i + 1}/{iters}: {elapsed:.3f}s (new tokens: {n_new})")
    return _summarize("hf", metrics)


# ---------------------------------------------------------------------------
# TTNN backend — aligned with demo.run_ttnn_backend (prefill + _decode_loop)
# ---------------------------------------------------------------------------


def benchmark_ttnn(
    model_id: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    warmup: int,
    iters: int,
    *,
    max_seq_len_user: int,
    device_fusion: bool,
    vision_backend: str,
    text_qkv_permute: bool,
    ttnn_dtype: str,
    stop_at_eos: bool,
) -> dict:
    logger.info(
        f"Running TTNN benchmark (aligned with demo: device_fusion={device_fusion}, "
        f"vision_backend={vision_backend}, text_qkv_permute={text_qkv_permute})..."
    )

    from models.demos.dots_ocr.demo.demo import _decode_loop, _ensure_local_ttnn_import, _load_dots_ttnn_state_dict

    _ensure_local_ttnn_import()
    import ttnn

    try:
        import ttnn._ttnn  # noqa: F401
    except Exception as exc:
        return {"backend": "ttnn", "error": f"TTNN not usable: {exc}"}

    from models.demos.dots_ocr.tt.common import (
        fused_ttnn_embeddings_to_torch,
        merge_vision_tokens,
        merge_vision_tokens_ttnn,
        pad_embedding_ttnn,
        pad_embedding_ttnn_tensor,
        preprocess_inputs_prefill,
        preprocess_inputs_prefill_ttnn,
        text_embeds_from_ttnn_embedding,
        text_embeds_from_ttnn_embedding_ttnn,
        text_rope_from_hf,
        ttnn_fused_batch_to_user_list,
    )
    from models.demos.dots_ocr.tt.generator import Generator
    from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, get_max_seq_len_cap, open_mesh_device
    from models.demos.dots_ocr.tt.model import DotsTransformer, DropInVisionTransformer
    from models.demos.dots_ocr.tt.model_config import DotsModelArgs
    from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs

    tt_dtype = ttnn.bfloat16 if ttnn_dtype == "bf16" else ttnn.bfloat8_b
    use_host_rope = True

    cap = get_max_seq_len_cap()
    max_seq_len = min(max_seq_len_user, cap) if cap else max_seq_len_user
    if max_seq_len < 128:
        logger.warning(f"Raising max_seq_len {max_seq_len} -> 128 (prefill/KV cache requirement)")
        max_seq_len = 128

    mesh_device = open_mesh_device()
    model_args = None
    tt_model = None
    generator = None
    visual = None
    ref = None

    try:
        os.environ.setdefault("HF_MODEL", model_id)
        ref = DotsOCRReference(HFLoadSpec(model_id=model_id))

        def _build_text_stack(*, qkv_permute: bool):
            nonlocal model_args, tt_model, generator
            model_args = DotsModelArgs(
                mesh_device=mesh_device,
                hf_config=ref.model.config,
                max_batch_size=1,
                max_seq_len=max_seq_len,
            )
            model_args.dots_text_qkv_permute = bool(qkv_permute)
            model_args.dots_use_host_rope = bool(use_host_rope)
            model_args.lm_head_dtype = ttnn.bfloat16
            state_dict = _load_dots_ttnn_state_dict(model_args, text_qkv_permute=qkv_permute)
            tt_model = DotsTransformer(
                args=model_args,
                dtype=tt_dtype,
                mesh_device=mesh_device,
                state_dict=state_dict,
                weight_cache_path=model_args.weight_cache_path(tt_dtype),
                paged_attention_config=None,
            )
            generator = Generator(tt_model, model_args, mesh_device, processor=ref.processor, tokenizer=ref.tokenizer)

        _build_text_stack(qkv_permute=bool(text_qkv_permute))

        if vision_backend == "ttnn" and (hasattr(ref.model, "vision_tower") or hasattr(ref.model, "visual")):
            vision_model_args = DotsVisionModelArgs(mesh_device=mesh_device, hf_config=ref.model.config)
            visual = DropInVisionTransformer(ref.model, vision_model_args, debug=False)
        else:
            visual = None

        def _one_iter() -> dict:
            img_for_preprocess = None
            if image_path and os.path.isfile(image_path):
                img_for_preprocess = Image.open(image_path).convert("RGB")
            inputs = ref.preprocess_image_and_prompt(img_for_preprocess, prompt)

            pad_token_id = ref.tokenizer.pad_token_id or 0

            image_embeds = torch.tensor([], dtype=torch.bfloat16)
            if getattr(inputs, "pixel_values", None) is not None:
                if vision_backend == "hf":
                    image_embeds = ref.vision_forward(inputs.pixel_values, getattr(inputs, "image_grid_thw", None))
                elif visual is not None:
                    image_embeds = visual(inputs.pixel_values, getattr(inputs, "image_grid_thw", None))
                image_embeds = image_embeds.to(torch.bfloat16)

            prefill_for_rope: torch.Tensor | None = None
            if device_fusion:
                text_ttnn = text_embeds_from_ttnn_embedding_ttnn(tt_model, inputs.input_ids)
                if image_embeds.numel() > 0:
                    fused_ttnn = merge_vision_tokens_ttnn(
                        inputs.input_ids, text_ttnn, image_embeds, ref.model.config, mesh_device=mesh_device
                    )
                else:
                    fused_ttnn = text_ttnn
                pad_tt = pad_embedding_ttnn_tensor(tt_model, int(pad_token_id))
                per_user = ttnn_fused_batch_to_user_list(fused_ttnn)
                input_prefill, decoding_pos, prefill_lens = preprocess_inputs_prefill_ttnn(
                    per_user, model_args, inputs.attention_mask, pad_tt
                )
                if use_host_rope:
                    prefill_for_rope = fused_ttnn_embeddings_to_torch(fused_ttnn, mesh_device)
            else:
                text_embeds = text_embeds_from_ttnn_embedding(tt_model, inputs.input_ids)
                input_embeds = (
                    merge_vision_tokens(inputs.input_ids, text_embeds, image_embeds, ref.model.config)
                    if image_embeds.numel() > 0
                    else text_embeds
                )
                pad_embedding = pad_embedding_ttnn(tt_model, int(pad_token_id))
                input_prefill, decoding_pos, prefill_lens = preprocess_inputs_prefill(
                    [input_embeds[0]],
                    model_args,
                    inputs.attention_mask,
                    pad_embedding,
                )
                prefill_for_rope = input_embeds

            rot_mats = None
            if use_host_rope:
                if prefill_for_rope is None:
                    raise RuntimeError("host RoPE: missing embeddings for rope")
                cos, sin = text_rope_from_hf(inputs, prefill_for_rope, ref.model, model_args, pad_token_id)
                rot_mats = (cos, sin)

            t_prefill = time.perf_counter()
            logits = generator.prefill_forward_text(
                input_prefill,
                rot_mats=rot_mats,
                kv_cache=None,
                prompt_lens=torch.tensor(decoding_pos),
            )
            ttft = time.perf_counter() - t_prefill

            t_decode = time.perf_counter()
            all_outputs = _decode_loop(
                generator=generator,
                tokenizer=ref.tokenizer,
                ref_model=ref.model,
                prefilled_logits=logits,
                decoding_pos=decoding_pos,
                max_new_tokens=max_new_tokens,
                stop_at_eos=stop_at_eos,
                repetition_penalty=None,
                fixed_steps=False,
            )
            decode_s = time.perf_counter() - t_decode

            n_gen = len(all_outputs[0]) if all_outputs else 0

            return {
                "total": ttft + decode_s,
                "ttft": ttft,
                "decode_s": decode_s,
                "decode_tokens": float(n_gen),
                "prefill_tokens": float(prefill_lens[0]),
            }

        for _ in range(warmup):
            _one_iter()

        metrics: list[dict] = []
        for i in range(iters):
            m = _one_iter()
            metrics.append(m)
            logger.info(
                f"TTNN iter {i + 1}/{iters}: total={m['total']:.3f}s ttft={m['ttft']:.3f}s "
                f"decode={int(m['decode_tokens'])} tok in {m['decode_s']:.3f}s "
                f"(prefill_len={int(m.get('prefill_tokens', 0))})"
            )
        return _summarize("ttnn", metrics)
    finally:
        try:
            if generator is not None:
                del generator
            if visual is not None:
                del visual
            if tt_model is not None:
                del tt_model
            if ref is not None:
                del ref
        except Exception:
            pass
        close_dots_mesh_device(mesh_device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    configure_dots_ocr_console_logging()
    parser = argparse.ArgumentParser(description="Dots OCR perf benchmark (HF vs TTNN)")
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Image file. If omitted, uses demo/benchmark_image.png when present (same multimodal path as demo).",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Do not load the default demo image; benchmark text-only (no pixel_values).",
    )
    parser.add_argument("--prompt", type=str, default="Extract all text from this document.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--backend", type=str, default="both", choices=["hf", "ttnn", "both"])
    parser.add_argument("--hf-model", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument(
        "--no-device-fusion",
        dest="device_fusion",
        action="store_false",
        default=True,
        help="Host-side merge + preprocess_inputs_prefill (default is device fusion, matching demo)",
    )
    parser.add_argument("--vision-backend", type=str, default="ttnn", choices=["hf", "ttnn"])
    parser.add_argument(
        "--no-text-qkv-permute",
        dest="text_qkv_permute",
        action="store_false",
        default=True,
        help="HF Q/K layout without Meta permute (default: permute on, matching demo)",
    )
    parser.add_argument("--ttnn-dtype", type=str, default="bf16", choices=["bf16", "bf8"])
    parser.add_argument(
        "--no-eos",
        dest="stop_at_eos",
        action="store_false",
        default=True,
        help="Decode fixed steps without stopping at EOS (matches demo --no-eos)",
    )
    args = parser.parse_args()

    model_id = args.hf_model or get_hf_model_id()
    # Empty `--image` must stay empty: Path("") resolves to "." (cwd), which exists() but is not a file.
    image_path_str = ""
    raw_img = (args.image or "").strip()
    if not raw_img and not args.text_only:
        default_p = _dots_ocr_default_benchmark_image()
        if default_p is not None:
            raw_img = str(default_p.resolve())
    if raw_img:
        ip = Path(raw_img)
        if ip.is_file():
            image_path_str = str(ip.resolve())
        elif ip.exists():
            logger.warning(f"Image path {ip} is not a regular file — proceeding text-only.")
        else:
            logger.warning(f"Image {ip} not found — proceeding text-only.")

    results: dict = {}
    if args.backend in ("hf", "both"):
        results["hf"] = benchmark_hf(
            model_id, image_path_str, args.prompt, args.max_new_tokens, args.warmup, args.iters
        )
    if args.backend in ("ttnn", "both"):
        results["ttnn"] = benchmark_ttnn(
            model_id,
            image_path_str,
            args.prompt,
            args.max_new_tokens,
            args.warmup,
            args.iters,
            max_seq_len_user=args.max_seq_len,
            device_fusion=args.device_fusion,
            vision_backend=args.vision_backend,
            text_qkv_permute=args.text_qkv_permute,
            ttnn_dtype=args.ttnn_dtype,
            stop_at_eos=args.stop_at_eos,
        )

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for backend, result in results.items():
        if "error" in result:
            print(f"{backend}: ERROR {result['error']}")
            continue
        print(f"\n{backend.upper()}:")
        print(f"  Avg Latency:  {result['avg_latency']:.4f}s")
        print(f"  FPS:          {result['fps']:.2f}")
        print(f"  TTFT:         {result['ttft']:.4f}s")
        print(f"  Decode:       {result['decode_tok_s']:.1f} tok/s")


if __name__ == "__main__":
    main()
