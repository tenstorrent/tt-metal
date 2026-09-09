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

It also writes ``hidden_states/layer_N.safetensors`` and ``n_layers``, which is what the per-layer
PCC tests read (``load_debug_trace``'s "per-layer format"); ``--no-hidden-states`` writes the
producer's subset alone. One caveat this cannot fix: the reference here is the same torch/HF path
those tests would otherwise compare against, so a golden from this script gives plumbing and
per-layer localisation, NOT independence from the reference implementation. Only a trace recorded
from a different implementation (vLLM) does that.
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
    """Resolve the same checkpoint the runner loads: ``PREFILL_HF_MODEL`` then the adapter's
    ``hf_model_default`` (mirrors ``MLAPrefillAdapter.load_hf_config``), so the golden is generated
    from the exact weights the device run validates against.

    The runner reads only the config from this dir and pulls weights from the TTNN cache, so its
    ``hf_model_default`` may be a config-only in-tree dir (kimi_k2_6, deepseek_v3). The golden runs
    the torch/HF reference forward and needs real weights — require safetensors here and point the
    user at PREFILL_HF_MODEL instead of failing deep in the weight load.
    """
    env = os.environ.get("PREFILL_HF_MODEL")
    model_path = env or variant.hf_model_default
    if model_path and Path(model_path).is_dir() and any(Path(model_path).glob("*.safetensors")):
        return Path(model_path)
    src = "PREFILL_HF_MODEL" if env else "hf_model_default"
    raise SystemExit(
        f"No HF safetensors checkpoint for {variant.name} (tried {src}={model_path!r}): the "
        f"prompt-trace golden needs real weights. Set PREFILL_HF_MODEL to an HF safetensors dir "
        f"(hf_model_default may be a config-only in-tree dir whose weights live in the TTNN cache)."
    )


def _load_config(variant, model_path: Path, isl: int):
    """The config the reference forward runs on.

    Prefers the variant's ``config_builder`` -- the same hook ``conftest._resolve_config_only`` uses
    -- so a variant needing normalization gets it here too. Without this, the golden is generated
    from a raw AutoConfig: Mistral Small 4 then has no ``rope_theta`` (AttributeError in rope setup)
    and, worse, no ``mla_disable_yarn_mscale``, so the reference would bake DeepSeek's mscale**2 =
    2.2058 softmax scale into the golden -- a wrong golden that every downstream comparison agrees
    with. Variants without a builder (DeepSeek, Kimi) take the AutoConfig path exactly as before.
    """
    builder = variant.config_builder
    if builder is not None:
        # Zero-arg by protocol (same as conftest); the builder resolves the checkpoint from
        # PREFILL_HF_MODEL, i.e. the dir _resolve_model_path already validated.
        config = builder()
    else:
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
        text = prompt
    elif prompt_file is not None:
        data = json.loads(Path(prompt_file).read_text())
        if isinstance(data, dict):
            data = data.get("prompts", data)
        if isinstance(data, list):
            if not data:
                raise SystemExit(f"no prompts in {prompt_file}")
            # The reference forward validates one prompt; a multi-prompt file would silently drop the rest.
            if len(data) > 1:
                logger.warning(f"[gen-trace] {prompt_file} holds {len(data)} prompts; using index 0 only")
            item = data[0]
        else:
            item = data
        if isinstance(item, dict):
            if "prompt" not in item:
                raise SystemExit(f'prompt entry missing "prompt" key: {item!r}')
            text = item["prompt"]
        else:
            text = item
    else:
        raise SystemExit("provide --prompt or --prompt-file")
    if not isinstance(text, str) or not text.strip():
        raise SystemExit("prompt is empty; provide non-empty prompt text")
    return text


def _meta_pe_to_hf(pe: torch.Tensor) -> torch.Tensor:
    """De-interleave rope "pe" from the reference's Meta frame to HF half-split.

    The producer re-interleaves HF->Meta on load (stack(halves).reshape); writing the
    reference (Meta) directly would double-apply the swap. This is that transform's inverse:
    Meta ``[a0,b0,a1,b1,...]`` -> HF ``[a0,a1,...,b0,b1,...]``.
    """
    return torch.cat([pe[:, 0::2], pe[:, 1::2]], dim=-1)


def write_trace_dir(
    out_dir: Path,
    token_ids: torch.Tensor,
    ref_kvpe_list,
    kv_lora_rank: int,
    qk_rope_head_dim: int | None = None,
    ref_snapshots=None,
    num_layers: int | None = None,
    num_real_tokens: int | None = None,
) -> Path:
    """Write ``kv_cache/`` always, and ``hidden_states/`` when ``ref_snapshots`` is given.

    The producer validates device KV against ``kv_cache/`` alone. The per-layer PCC tests read
    decoder outputs from ``hidden_states/layer_i.safetensors`` and take the layer count from
    ``metadata["n_layers"]``, so a dir written without snapshots serves the producer and makes
    ``load_debug_trace`` raise on the missing file -- loud, which is the point.
    """
    out_dir = Path(out_dir)
    (out_dir / "kv_cache").mkdir(parents=True, exist_ok=True)

    for i, kvpe in enumerate(ref_kvpe_list):
        # ref_kvpe_list[i] is the layer's compressed MLA line, [1, 1, seq, kv_lora_rank + rope].
        t = kvpe
        while t.dim() > 2:
            t = t[0]
        t = t.to(torch.float32)
        # A reference that cached expanded per-head keys arrives here as [seq, head_dim]; the nope
        # slice would clamp, `pe` would come out empty, and the golden would be quietly the wrong
        # width. reference_kvpe_for_layer derives the compressed line, so this only fires if that
        # stops happening -- but it fails silently if unchecked.
        if qk_rope_head_dim is not None and t.shape[-1] != kv_lora_rank + qk_rope_head_dim:
            raise SystemExit(
                f"layer {i}: reference KVPE width {t.shape[-1]} != {kv_lora_rank} + {qk_rope_head_dim}; "
                "this is not the compressed MLA line the device caches"
            )
        nope = t[:, :kv_lora_rank]  # compared directly by the producer, written as-is
        pe = _meta_pe_to_hf(t[:, kv_lora_rank:])
        row = torch.cat([nope, pe], dim=-1).contiguous()
        save_file({f"kv_post_transform_layer_{i}": row}, str(out_dir / "kv_cache" / f"layer_{i}.safetensors"))

    meta = {"token_ids": token_ids[0].tolist()}
    if num_real_tokens is not None:
        meta["num_real_tokens"] = int(num_real_tokens)

    if ref_snapshots is not None:
        # ref_snapshots is [embed, layer_0..layer_{n-1}, norm, lm_head]; layer i is at 1 + i.
        # fp32 like the kv rows above: the snapshots are fp32 and a bf16 golden would put its own
        # rounding into every PCC (visible on the near-1.0 early layers). Costs ~3 GB at 36L/isl 5120.
        n = num_layers if num_layers is not None else len(ref_snapshots) - 3
        (out_dir / "hidden_states").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            t = ref_snapshots[1 + i]
            while t.dim() > 2:
                t = t[0]
            save_file(
                {f"decoder_output_layer_{i}": t.to(torch.float32).contiguous()},
                str(out_dir / "hidden_states" / f"layer_{i}.safetensors"),
            )
        meta["n_layers"] = n

        # The reference's own next token, for the consumer's first-token check. Taken at the last
        # REAL position: this generator forces right padding, so the final rows are pad tokens whose
        # argmax is not the model's next token.
        logits = ref_snapshots[-1]
        while logits.dim() > 2:
            logits = logits[0]
        last_real = (num_real_tokens or logits.shape[0]) - 1
        meta["next_token_id"] = int(logits[last_real].float().argmax().item())
        meta["next_token_position"] = last_real

    (out_dir / "metadata.json").write_text(json.dumps(meta))
    return out_dir


def generate(
    model: str, prompt_text: str, isl: int, num_layers: int, out_dir: Path, hidden_states: bool = True
) -> Path:
    variant = get_adapter(model)
    # Sparse/DSA models also keep an index-key cache (config 1); the producer's validation reads its
    # golden from the trace dir, but this generator writes only the KVPE cache — so reject them loudly
    # rather than emitting an incomplete golden that crashes downstream on missing index shards.
    if hasattr(variant.model_config, "INDEX_HEAD_DIM"):
        raise SystemExit(f"{model} is a sparse/DSA model; prompt-trace generation supports dense-KVPE MLA only")
    model_path = _resolve_model_path(variant)
    config = _load_config(variant, model_path, isl)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), use_fast=True, trust_remote_code=variant.tokenizer_trust_remote_code
    )
    tokenizer.padding_side = "right"

    token_ids, attention_mask, _ = tokenize_prompt_to_isl(tokenizer, max_isl=isl, prompt_text=prompt_text)
    # A prompt that tokenizes to all padding produces a meaningless all-pad golden that still "passes" PCC.
    if int(attention_mask.sum()) == 0:
        raise SystemExit("prompt tokenized to zero real tokens (all padding)")
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

    out = write_trace_dir(
        out_dir,
        token_ids,
        result.ref_kvpe_list,
        variant.model_config.KV_LORA_RANK,
        qk_rope_head_dim=variant.model_config.QK_ROPE_HEAD_DIM,
        ref_snapshots=result.ref_snapshots if hidden_states else None,
        num_layers=num_layers,
        num_real_tokens=int(attention_mask.sum()),
    )
    logger.success(
        f"[gen-trace] wrote {num_layers}-layer golden trace to {out} "
        f"(hidden_states={'yes' if hidden_states else 'no'})"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))
    p.add_argument("--prompt", default=None, help="raw prompt text")
    p.add_argument("--prompt-file", default=None, help='JSON prompt file ([{"prompt": ...}] or {"prompts": [...]})')
    p.add_argument("--isl", type=int, default=int(os.environ.get("PREFILL_MAX_SEQ_LEN", "1024")))
    p.add_argument("--num-layers", type=int, default=int(os.environ.get("PREFILL_NUM_LAYERS", "2")))
    p.add_argument("--out", required=True, help="output trace dir")
    p.add_argument(
        "--no-hidden-states",
        action="store_true",
        help="write kv_cache/ only (enough for the producer; the per-layer PCC tests need the rest)",
    )
    args = p.parse_args()

    prompt_text = _load_prompt_text(args.prompt, args.prompt_file)
    out = generate(args.model, prompt_text, args.isl, args.num_layers, Path(args.out), not args.no_hidden_states)
    # last stdout line: the trace dir, for a caller to capture
    print(str(out))


if __name__ == "__main__":
    main()
