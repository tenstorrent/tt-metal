#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the golden KV cache for ``llama32_8b_d_p`` prefill validation (fp32 torch reference).

Runs the package's chosen reference — ``transformers`` itself (``01_REFERENCE.md`` §1, ``DEC-004``)
— **one decoder layer at a time**, in fp32, and writes each layer's **post-RoPE K** and **raw V** as
it is produced. ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache`` stores exactly those
two tensors (``tt/attention/kv_cache.py:172`` "post-RoPE K and raw V"), so this is the reference the
device cache is compared against.

**Why a per-layer loop and not ``LlamaForCausalLM.from_pretrained``** (``DEC-053``): the recipe
requires streaming weights per layer via mmap and writing per layer, so neither 32 layers of fp32
weight (32 GiB) nor 32 layers of KV is ever resident. One ``LlamaDecoderLayer`` is built, filled
from the mmapped shard, run, saved, and dropped. The math is HF's own — the same
``LlamaAttention``/``LlamaMLP``/``LlamaRMSNorm`` code path ``G-LAYER`` and ``G-MODEL`` already gate
against — so this script adds a *driver*, not a second reference implementation.

**Reference dtype policy** (Appendix E.1/E.2, stated because a gate must state it): checkpoint
weights are ``bfloat16`` on disk and are upcast to fp32 **exactly** (no rounding introduced, none
removed); every operation runs in fp32; K/V are saved at ``--dtype`` (default ``float32``, so the
stored golden carries no dtype loss of its own). The device further rounds weights to ``bfloat8_b``
and the cache to ``bfloat8_b``, so this reference does **not** share the device's rounding — which
is the trap Appendix E.1 documents for ``models/tt_transformers``' bf16-weight references.

**Causal mask is passed explicitly.** ``cfg._attn_implementation = "eager"`` plus
``create_causal_mask(...)``: ``eager_attention_forward`` applies only the mask handed to it, so a
``None`` mask yields non-causal attention *silently* (Appendix F.2). K/V for layer 0 would still be
right and every later layer subtly wrong — the worst possible failure for a golden trace.

Output format (copied exactly from ``models/demos/minimax_m3/scripts/generate_golden_kv_cache.py:27``
minus M3's MSA-only ``index_k_cache_layer_<i>``; the engine's producer read-back and
``tt/tt_prefill_runtime.py::kv_cache_pcc_check`` both expect it)::

    {trace_dir}/
        metadata.json                        # prompt, token_ids, model info
        kv_cache/
            layer_0.safetensors              # key_cache_layer_0   (post-RoPE K, HF layout)
            ...                              # value_cache_layer_0 (raw V)
            layer_31.safetensors             # each [1, num_kv_heads, seq_len, head_dim]

``key_cache_layer_<i>`` is in the **HF half-split rotary convention**. The device stores the
**Meta/interleaved** convention (``tt/rope.py`` module docstring), so a consumer must permute the
golden K's lanes before comparing — ``scripts/verify_golden_kv.py::hf_to_meta_lane_permutation``
is the one place that permutation is written, and
``tests/unit/test_attention_chunked_vs_ref.py`` uses it.

Usage::

    export HF_MODEL=/home/mstojkovic/models/Llama-3.1-8B-Instruct
    python3 models/demos/llama32_8b_d_p/scripts/generate_golden_kv_cache.py \\
        --prompt "The capital of France is" --pad-to 512 \\
        --out /home/mstojkovic/llama32_8b_golden/p7
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Make `models.demos.llama32_8b_d_p...` importable when run as a script (sys.path[0] is scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def _raise_cpu_time_limit() -> None:
    """Raise ``RLIMIT_CPU`` soft -> hard.

    Copied in spirit from ``models/demos/minimax_m3/scripts/generate_golden_kv_cache.py:68``:
    ``RLIMIT_CPU`` counts CPU-seconds summed over every thread, so with 64 threads a default soft
    cap drains ~64x wall-clock and the process dies mid-run with ``SIGXCPU``, which looks like a
    crash but is only the limit.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
            print(f"[limit] raised RLIMIT_CPU soft {soft}s -> {hard}")
        except (ValueError, OSError) as exc:  # pragma: no cover - environment dependent
            print(f"[limit] WARNING: could not raise RLIMIT_CPU (soft={soft}s): {exc}", file=sys.stderr)


class _ShardReader:
    """Lazy, mmapped, per-tensor reader over a sharded safetensors checkpoint.

    ``safe_open`` mmaps the shard, so ``get_tensor`` materialises **one** tensor at a time and the
    32 GiB fp32 model is never resident. Handles both the sharded
    (``model.safetensors.index.json``) and the single-file (``model.safetensors``) layouts, the same
    two cases ``models/tt_transformers/tt/load_checkpoints.py:18`` ``load_hf_state_dict`` handles.
    """

    def __init__(self, ckpt_dir: Path):
        self.dir = Path(ckpt_dir)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            with open(index) as fh:
                self.weight_map = json.load(fh)["weight_map"]
        else:
            single = self.dir / "model.safetensors"
            assert single.exists(), f"neither {index} nor {single} exists"
            with safe_open(str(single), framework="pt") as handle:
                self.weight_map = {k: single.name for k in handle.keys()}
        self._handles: dict[str, object] = {}

    def keys(self):
        return self.weight_map.keys()

    def get(self, key: str) -> torch.Tensor:
        assert key in self.weight_map, f"{key!r} not in the checkpoint ({len(self.weight_map)} keys)"
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.dir / shard), framework="pt")
        # .float() on a bf16 checkpoint tensor is exact: every bf16 value is an fp32 value.
        return self._handles[shard].get_tensor(key).float()

    def substate(self, prefix: str) -> dict:
        """``{suffix: tensor}`` for every key under ``prefix``, e.g. ``model.layers.7.``."""
        return {k[len(prefix) :]: self.get(k) for k in self.weight_map if k.startswith(prefix)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate the fp32 golden KV cache for llama32_8b_d_p",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", type=str, help="Direct prompt text")
    src.add_argument("--prompt-json", type=Path, help='JSON file: {"prompt": "..."} or a bare string')
    src.add_argument("--prompt-file", type=Path, help="Plain-text prompt file")

    ap.add_argument("--out", type=Path, required=True, help="Output trace directory (kv_cache/ is created inside)")
    ap.add_argument("--model-path", type=str, default=None, help="HF model dir (default: $HF_MODEL)")
    ap.add_argument("--max-tokens", type=int, default=None, help="Truncate the prompt to this many tokens")
    ap.add_argument(
        "--pad-to",
        type=int,
        default=None,
        help="Pad the token sequence with the tokenizer pad/eos id up to this length. The device "
        "prefill runs whole tiles/chunks, so a golden whose length is not a multiple of the chunk "
        "size can only be compared over a prefix; padding here keeps the two lengths equal. "
        "`n_real_tokens` in metadata.json records how many tokens were real.",
    )
    ap.add_argument("--no-chat-template", action="store_true", help="Use the raw prompt, no chat template")
    ap.add_argument("--num-layers", type=int, default=None, help="Only emit the first N layers (debug)")
    ap.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="float32",
        help="Stored K/V dtype (default float32). Compute is ALWAYS fp32; this only sets what is "
        "written, and float32 keeps the stored golden free of any rounding of its own.",
    )
    return ap.parse_args(argv)


def load_prompt(args) -> str:
    if args.prompt_json:
        with open(args.prompt_json) as fh:
            data = json.load(fh)
        if isinstance(data, str):
            return data
        assert isinstance(data, dict) and "prompt" in data, f"{args.prompt_json}: want a string or {{'prompt': ...}}"
        return data["prompt"]
    if args.prompt_file:
        return Path(args.prompt_file).read_text()
    return args.prompt


def tokenize_prompt(tokenizer, prompt, *, max_tokens, use_chat_template, pad_to):
    """Tokenize, optionally truncate, optionally pad. Returns ``(token_ids, n_real)``."""
    if use_chat_template:
        # BOTH keywords are load-bearing on transformers 5.12.1 (measured, see R-026):
        #   * without tokenize=True the call returns the rendered chat STRING;
        #   * WITH tokenize=True it returns a `BatchEncoding`, because `return_dict` now defaults
        #     to True — so `list(...)` yields the dict KEYS `['input_ids', 'attention_mask']`, i.e.
        #     a plausible-looking 2-element "token list" of strings. `return_dict=False` is what
        #     actually returns `list[int]`. `models/demos/minimax_m3/scripts/
        #     generate_golden_kv_cache.py:180` passes only `tokenize=True` and would hit this.
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
    else:
        ids = tokenizer(prompt)["input_ids"]
    ids = list(ids)
    assert all(isinstance(i, int) for i in ids), f"tokenizer returned non-int ids: {type(ids[0]).__name__}"
    if max_tokens and len(ids) > max_tokens:
        print(f"[tokenize] truncating {len(ids)} -> {max_tokens} tokens")
        ids = ids[:max_tokens]
    n_real = len(ids)
    if pad_to is not None:
        assert pad_to >= n_real, f"--pad-to {pad_to} is shorter than the prompt ({n_real} tokens)"
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        assert pad_id is not None, "tokenizer has neither pad_token_id nor eos_token_id; cannot pad"
        ids = ids + [int(pad_id)] * (pad_to - n_real)
        print(f"[tokenize] padded {n_real} -> {len(ids)} tokens with id {pad_id}")
    return ids, n_real


@torch.no_grad()
def generate(  # noqa: C901 - a linear driver; splitting it would hide the streaming order
    *,
    ckpt_dir,
    token_ids,
    out_dir,
    store_dtype=torch.float32,
    num_layers=None,
    progress=print,
):
    """Stream the fp32 reference layer by layer, writing ``layer_<i>.safetensors`` as it goes.

    Returns a dict of the facts ``metadata.json`` needs (shapes, layer count, timings).
    """
    from transformers import LlamaConfig
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama import modeling_llama as ml

    cfg = LlamaConfig.from_pretrained(str(ckpt_dir))
    # Appendix F.2: eager + an EXPLICIT causal mask. eager_attention_forward applies only the mask
    # it is handed, so attention_mask=None is silently non-causal.
    cfg._attn_implementation = "eager"
    total_layers = cfg.num_hidden_layers
    n_layers = total_layers if num_layers is None else min(num_layers, total_layers)
    head_dim = cfg.head_dim
    num_kv_heads = cfg.num_key_value_heads
    seq_len = len(token_ids)

    kv_dir = Path(out_dir) / "kv_cache"
    kv_dir.mkdir(parents=True, exist_ok=True)

    reader = _ShardReader(ckpt_dir)
    progress(f"[load] checkpoint has {len(reader.weight_map)} tensors; streaming per layer via mmap")

    # --- embeddings: one index_select, then the [vocab, hidden] table is dropped immediately ---
    embed = reader.get("model.embed_tokens.weight")
    ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    hidden = embed.index_select(0, ids.reshape(-1)).reshape(1, seq_len, cfg.hidden_size).contiguous()
    del embed

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    rotary = ml.LlamaRotaryEmbedding(cfg).float()
    position_embeddings = rotary(hidden, position_ids)
    # One fresh (empty) cache purely to size the mask; the per-layer caches below are separate.
    causal_mask = ml.create_causal_mask(
        config=cfg,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=DynamicCache(config=cfg),
        position_ids=position_ids,
    )
    assert causal_mask is not None, (
        "create_causal_mask returned None: eager attention would then be NON-CAUSAL and this golden "
        "would be silently wrong from layer 1 onward (BRINGUP_RECIPE.md Appendix F.2)"
    )

    expected = (1, num_kv_heads, seq_len, head_dim)
    t0 = time.time()
    for layer_idx in range(n_layers):
        layer = ml.LlamaDecoderLayer(cfg, 0).float().eval()
        state = reader.substate(f"model.layers.{layer_idx}.")
        missing, unexpected = layer.load_state_dict(state, strict=False)
        # Loud on either side: a renamed key makes load_state_dict a no-op for that weight and the
        # layer then runs on its random init, which looks like a model bug three phases later.
        assert not missing, f"layer {layer_idx}: checkpoint is missing {sorted(missing)}"
        assert not unexpected, f"layer {layer_idx}: checkpoint has unused keys {sorted(unexpected)}"

        cache = DynamicCache(config=cfg)
        hidden = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
        if not torch.is_tensor(hidden):  # older/newer transformers may return a tuple
            hidden = hidden[0]

        key_cache = cache.layers[0].keys  # post-RoPE K, HF half-split convention
        value_cache = cache.layers[0].values  # raw V
        assert tuple(key_cache.shape) == expected, f"layer {layer_idx}: K {tuple(key_cache.shape)} != {expected}"
        assert tuple(value_cache.shape) == expected, f"layer {layer_idx}: V {tuple(value_cache.shape)} != {expected}"
        assert torch.isfinite(key_cache).all() and torch.isfinite(value_cache).all(), f"layer {layer_idx}: non-finite"

        save_file(
            {
                f"key_cache_layer_{layer_idx}": key_cache.to(store_dtype).contiguous(),
                f"value_cache_layer_{layer_idx}": value_cache.to(store_dtype).contiguous(),
            },
            str(kv_dir / f"layer_{layer_idx}.safetensors"),
        )
        # Drop the layer, its weights and its cache before the next layer allocates.
        del layer, state, cache, key_cache, value_cache
        progress(f"[forward] layer {layer_idx:>2}/{n_layers - 1} saved  ({time.time() - t0:.1f}s elapsed)")

    return {
        "num_layers": n_layers,
        "model_num_layers": total_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": cfg.hidden_size,
        "n_tokens": seq_len,
        "key_cache_shape": list(expected),
        "value_cache_shape": list(expected),
        "forward_time_seconds": time.time() - t0,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    _raise_cpu_time_limit()

    model_path = args.model_path or os.environ.get("HF_MODEL")
    if not model_path:
        print("ERROR: pass --model-path or set $HF_MODEL", file=sys.stderr)
        return 1
    ckpt_dir = Path(model_path)
    if not ckpt_dir.is_dir():
        print(f"ERROR: {ckpt_dir} is not a directory", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    prompt = load_prompt(args)
    print(f"[load] prompt: {len(prompt)} characters")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
    use_chat_template = not args.no_chat_template
    token_ids, n_real = tokenize_prompt(
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        use_chat_template=use_chat_template,
        pad_to=args.pad_to,
    )
    print(f"[load] {len(token_ids)} tokens ({n_real} real)")

    torch.set_num_threads(os.cpu_count() or 8)
    print(f"[forward] fp32 reference, {torch.get_num_threads()} threads, streaming per layer")

    facts = generate(
        ckpt_dir=ckpt_dir,
        token_ids=token_ids,
        out_dir=out_dir,
        store_dtype=_DTYPES[args.dtype],
        num_layers=args.num_layers,
        progress=lambda msg: print(msg, flush=True),
    )

    metadata = {
        "model_path": str(ckpt_dir),
        "model_config": "Llama-3.1-8B-Instruct",
        "reference": (
            "transformers.models.llama.modeling_llama.LlamaDecoderLayer, one layer at a time, "
            "fp32 compute, weights streamed per layer via mmap (DEC-053)"
        ),
        "reference_dtype_policy": (
            "checkpoint bfloat16 upcast exactly to fp32; all arithmetic fp32; K/V stored as " f"{args.dtype}"
        ),
        "attn_implementation": "eager",
        "explicit_causal_mask": True,
        "kv_convention": "post-RoPE K (HF half-split rotary), raw V, HF layout",
        "prompt_source": (
            str(args.prompt_json) if args.prompt_json else (str(args.prompt_file) if args.prompt_file else "direct")
        ),
        "prompt": prompt if len(prompt) <= 500 else prompt[:500] + "...",
        "prompt_length_chars": len(prompt),
        "chat_template": use_chat_template,
        "token_ids": token_ids,
        "n_real_tokens": n_real,
        "padded": args.pad_to is not None,
        "dtype": args.dtype,
        "kv_cache_format": "separate_k_v",
        **facts,
    }
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    total_gb = sum(f.stat().st_size for f in (out_dir / "kv_cache").glob("*.safetensors")) / (1024**3)
    print(f"[save] {facts['num_layers']} layers -> {out_dir / 'kv_cache'} ({total_gb:.2f} GB)")
    print(f"[save] metadata -> {out_dir / 'metadata.json'}")
    print(f"[done] {facts['forward_time_seconds']:.1f}s for {facts['n_tokens']} tokens")
    print(f"[next] python3 models/demos/llama32_8b_d_p/scripts/verify_golden_kv.py {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
