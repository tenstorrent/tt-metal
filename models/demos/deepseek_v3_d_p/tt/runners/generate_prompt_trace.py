# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Generate a prefill golden "trace dir" from a user prompt, host-only (no device).

The prefill producer reads its input tokens and golden KV cache from one trace dir
(``metadata.json["token_ids"]`` + ``kv_cache/layer_N.safetensors``). Recorded traces
come from vLLM; this builds an equivalent one from an arbitrary prompt by running the
torch/HF reference forward, so ``PREFILL_TRACE_DIR`` can point at it and the runner +
producer validate device KV against a reference generated for that exact prompt.

MLA models only (DeepSeek / Kimi): the golden is the compressed KVPE
``[seq, KV_LORA_RANK + QK_ROPE_HEAD_DIM]`` per layer.
"""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

from models.demos.common.prefill.adapter import DEFAULT_MODEL, get_adapter
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    load_and_compute_layer_by_layer,
    tokenize_prompt_to_isl,
)


def _resolve_model_path(variant) -> Path:
    """First of the variant's env var / local / shared path that holds an HF safetensors dir."""
    candidates = [os.environ.get(variant.env_var), variant.default_local_path, variant.shared_path]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise SystemExit(
        f"No model dir found for {variant.name}: set {variant.env_var} to the HF safetensors dir "
        f"(tried {variant.env_var}={os.environ.get(variant.env_var)!r}, "
        f"{variant.default_local_path!r}, {variant.shared_path!r})"
    )


def _load_config(model_path: Path, isl: int):
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    # Kimi ships a multimodal wrapper; the MLA reference wants the text sub-config.
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        config = config.text_config
    config = deepcopy(config)
    # AutoConfig does not populate this; the reference forward path expects it set.
    config.max_seq_len = isl
    return config


def _load_prompt_text(prompt: str | None, prompt_file: str | None) -> str:
    if prompt is not None:
        return prompt
    if prompt_file is not None:
        data = json.loads(Path(prompt_file).read_text())
        if isinstance(data, dict):
            data = data.get("prompts", data)
        item = data[0] if isinstance(data, list) else data
        return item["prompt"] if isinstance(item, dict) else item
    raise SystemExit("provide --prompt or --prompt-file")


def _meta_pe_to_hf(pe: torch.Tensor) -> torch.Tensor:
    """De-interleave rope "pe" from the reference's Meta frame to HF half-split.

    The producer re-interleaves HF->Meta on load (stack(halves).reshape); writing the
    reference (Meta) directly would double-apply the swap. This is that transform's inverse:
    Meta ``[a0,b0,a1,b1,...]`` -> HF ``[a0,a1,...,b0,b1,...]``.
    """
    return torch.cat([pe[:, 0::2], pe[:, 1::2]], dim=-1)


def write_trace_dir(out_dir: Path, token_ids: torch.Tensor, ref_kvpe_list, kv_lora_rank: int) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "kv_cache").mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps({"token_ids": token_ids[0].tolist()}))

    for i, kvpe in enumerate(ref_kvpe_list):
        # ref_kvpe_list[i] is the HF DynamicCache key cache for the layer, [1, 1, seq, head_dim].
        t = kvpe
        while t.dim() > 2:
            t = t[0]
        t = t.to(torch.float32)
        nope = t[:, :kv_lora_rank]  # compared directly by the producer, written as-is
        pe = _meta_pe_to_hf(t[:, kv_lora_rank:])
        row = torch.cat([nope, pe], dim=-1).contiguous()
        save_file({f"kv_post_transform_layer_{i}": row}, str(out_dir / "kv_cache" / f"layer_{i}.safetensors"))
    return out_dir


def generate(model: str, prompt_text: str, isl: int, num_layers: int, out_dir: Path) -> Path:
    variant = get_adapter(model)
    model_path = _resolve_model_path(variant)
    config = _load_config(model_path, isl)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=True, trust_remote_code=variant.tokenizer_trust_remote_code
    )
    tokenizer.padding_side = "right"

    token_ids, attention_mask, _ = tokenize_prompt_to_isl(tokenizer, max_isl=isl, prompt_text=prompt_text)
    logger.info(f"[gen-trace] model={model} isl={isl} num_layers={num_layers} tokens={token_ids.shape}")

    result = load_and_compute_layer_by_layer(
        variant=variant,
        model_path=model_path,
        config=config,
        num_layers=num_layers,
        token_ids=token_ids,
        attention_mask=attention_mask,
        compute_reference=True,
        build_ttnn_cache=False,  # host-only; no mesh_device / weight_cache_path needed
        seq_len=isl,
    )

    kv_lora_rank = variant.model_config.KV_LORA_RANK
    out = write_trace_dir(out_dir, token_ids, result.ref_kvpe_list, kv_lora_rank)
    logger.success(f"[gen-trace] wrote {num_layers}-layer golden trace to {out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))
    p.add_argument("--prompt", default=None, help="raw prompt text")
    p.add_argument("--prompt-file", default=None, help='JSON prompt file ([{"prompt": ...}] or {"prompts": [...]})')
    p.add_argument("--isl", type=int, default=int(os.environ.get("PREFILL_MAX_SEQ_LEN", "1024")))
    p.add_argument("--num-layers", type=int, default=int(os.environ.get("PREFILL_NUM_LAYERS", "2")))
    p.add_argument("--out", required=True, help="output trace dir")
    args = p.parse_args()

    prompt_text = _load_prompt_text(args.prompt, args.prompt_file)
    out = generate(args.model, prompt_text, args.isl, args.num_layers, Path(args.out))
    # last stdout line: the trace dir, for a caller to capture
    print(str(out))


if __name__ == "__main__":
    main()
