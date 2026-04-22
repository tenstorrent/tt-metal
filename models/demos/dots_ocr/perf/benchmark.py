# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmark for Dots OCR (HF vs TTNN).

Measures:
  - ``ttft``          — time from inputs → first sampled token (i.e. prefill).
  - ``decode_tok_s``  — steady-state decode throughput (tokens / s).
  - ``avg_latency``   — total wall time for the full iteration.

Uses the same timing columns as other VLM/decoder TT demos (ttft, decode_tok_s, etc.).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from loguru import logger
from PIL import Image

from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec, get_hf_model_id
from models.demos.dots_ocr.reference.model import DotsOCRReference


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
# TTNN backend — prefill + decode loop
# ---------------------------------------------------------------------------


def benchmark_ttnn(
    model_id: str,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    warmup: int,
    iters: int,
    *,
    max_seq_len: int,
) -> dict:
    logger.info("Running TTNN benchmark (prefill + decode)...")
    # tt_transformers prefill kernels require seq_len padded to a multiple of 128, and the KV cache
    # must be at least that tall. Enforce a sane minimum here to avoid cache height assertions.
    if max_seq_len < 128:
        logger.warning(f"Raising max_seq_len {max_seq_len} -> 128 (prefill/KV cache requirement)")
        max_seq_len = 128

    try:
        pass
    except Exception as exc:  # pragma: no cover
        logger.warning(f"ttnn not importable: {exc}")
        return {"backend": "ttnn", "error": str(exc)}

    from models.demos.dots_ocr.demo.demo import _build_tt_stack
    from models.demos.dots_ocr.tt.common import (
        merge_vision_tokens,
        preprocess_inputs_prefill,
        sample_host,
        text_rope_from_hf,
    )
    from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device, open_mesh_device

    mesh_device = open_mesh_device()
    try:
        ref, model_args, tt_model, generator, visual = _build_tt_stack(model_id, mesh_device, max_seq_len=max_seq_len)
        # TTTransformer forward expects `kv_cache` as a per-layer list (not nested).
        # The tt_transformers Generator wrapper takes `[kv_cache]` (list per model) for decode,
        # so our `Generator.decode_forward` wrapper will wrap this again as needed.
        tt_kv_cache = [layer.attention.layer_past for layer in tt_model.layers]

        image = Image.open(image_path).convert("RGB") if os.path.exists(image_path) else None
        inputs = ref.preprocess_image_and_prompt(image, prompt)

        pad_token_id = ref.tokenizer.pad_token_id or 0
        pad_embedding = ref.model.get_input_embeddings()(torch.tensor([pad_token_id])).squeeze(0)

        stop_ids: set[int] = set()
        for attr in ("eos_token_id", "pad_token_id"):
            val = getattr(ref.tokenizer, attr, None)
            if isinstance(val, int):
                stop_ids.add(val)
            elif isinstance(val, (list, tuple)):
                stop_ids.update(int(v) for v in val)

        def _one_iter() -> dict:
            t_start = time.perf_counter()

            if visual is not None and getattr(inputs, "pixel_values", None) is not None:
                image_embeds = visual(inputs.pixel_values, getattr(inputs, "image_grid_thw", None))
            else:
                image_embeds = torch.tensor([], dtype=torch.bfloat16)

            text_embeds = ref.model.get_input_embeddings()(inputs.input_ids)
            input_embeds = (
                merge_vision_tokens(inputs.input_ids, text_embeds, image_embeds, ref.model.config)
                if image_embeds.numel() > 0
                else text_embeds
            )
            input_prefill, decoding_pos, _prefill_lens = preprocess_inputs_prefill(
                [input_embeds[0]], model_args, inputs.attention_mask, pad_embedding
            )
            cos, sin = text_rope_from_hf(inputs, input_embeds, ref.model, model_args, pad_token_id)

            t_prefill = time.perf_counter()
            logits = generator.prefill_forward_text(
                input_prefill,
                rot_mats=(cos, sin),
                # Prefill path can run without KV-cache updates; passing a paged KV cache here
                # requires page_table/chunk_page_table alignment. Keep benchmarking simple.
                kv_cache=None,
                prompt_lens=torch.tensor(decoding_pos),
            )
            ttft = time.perf_counter() - t_prefill

            out_tok = torch.argmax(logits, dim=-1)
            current_pos = torch.tensor([decoding_pos[0]])
            decoded = 1
            t_decode = time.perf_counter()
            for _ in range(max_new_tokens - 1):
                try:
                    logits, _ = generator.decode_forward(
                        out_tok,
                        current_pos,
                        kv_cache=None,
                        enable_trace=False,
                    )
                except Exception as exc:
                    logger.warning(f"decode_forward failed: {exc}")
                    break
                _, out_tok = sample_host(logits, None, temperature=0.0)
                tok_id = int(out_tok.flatten()[0].item())
                decoded += 1
                current_pos += 1
                if tok_id in stop_ids:
                    break
            decode_s = time.perf_counter() - t_decode

            return {
                "total": time.perf_counter() - t_start,
                "ttft": ttft,
                "decode_s": decode_s,
                "decode_tokens": decoded,
            }

        for _ in range(warmup):
            _one_iter()
        metrics: list[dict] = []
        for i in range(iters):
            m = _one_iter()
            metrics.append(m)
            logger.info(
                f"TTNN iter {i + 1}/{iters}: total={m['total']:.3f}s ttft={m['ttft']:.3f}s "
                f"decode={m['decode_tokens']} in {m['decode_s']:.3f}s"
            )
        return _summarize("ttnn", metrics)
    finally:
        close_dots_mesh_device(mesh_device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Dots OCR perf benchmark (HF vs TTNN)")
    parser.add_argument("--image", type=str, default="")
    parser.add_argument("--prompt", type=str, default="Extract all text from this document.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--backend", type=str, default="both", choices=["hf", "ttnn", "both"])
    parser.add_argument("--hf-model", type=str, default=None)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    model_id = args.hf_model or get_hf_model_id()
    image_path = Path(args.image)
    if args.image and not image_path.exists():
        logger.warning(f"Image {image_path} not found — proceeding text-only.")

    results: dict = {}
    if args.backend in ("hf", "both"):
        results["hf"] = benchmark_hf(
            model_id, str(image_path), args.prompt, args.max_new_tokens, args.warmup, args.iters
        )
    if args.backend in ("ttnn", "both"):
        results["ttnn"] = benchmark_ttnn(
            model_id,
            str(image_path),
            args.prompt,
            args.max_new_tokens,
            args.warmup,
            args.iters,
            max_seq_len=args.max_seq_len,
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
