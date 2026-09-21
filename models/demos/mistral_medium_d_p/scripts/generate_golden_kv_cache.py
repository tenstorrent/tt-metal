#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the golden KV cache for Mistral-Medium-3.5-128B prefill validation.

Standalone: no tt-metal imports, so it can be copied to a GPU box as a single file.

**Dependencies.** ``torch`` (>=2.1, for ``float8_e4m3fn``), ``safetensors``, and ``transformers>=5``
-- the model implementation is NOT in this file: every layer is transformers' own
``transformers.models.ministral3.modeling_ministral3`` (``Ministral3DecoderLayer``,
``Ministral3RotaryEmbedding``), which is the reference the TT bring-up is pinned to. Verified with
``transformers==5.12.1`` / ``torch==2.11``. ``transformers`` is also used for the tokenizer.

**Reference.** ``transformers.models.ministral3.Ministral3DecoderLayer`` -- HF's own math for the
text backbone of ``Mistral3ForConditionalGeneration`` -- run **one layer at a time**. One layer is
built, filled from the mmapped checkpoint shards, run, its K/V saved, and dropped; neither the
88-layer weight set (~256 GB at bf16) nor the whole KV cache is ever resident. The rotary table
(YaRN, ``attention_scaling`` included) comes from ``Ministral3RotaryEmbedding``, so the golden
follows whatever transformers does for this config.

**fp8.** The checkpoint ships q/k/v/o and gate/up/down as ``F8_E4M3`` with a **scalar**
``weight_scale_inv`` sidecar (``quantization_config.weight_block_size == null``). Dequant is
``w = fp8.float() * weight_scale_inv``, done on the compute device. ``activation_scale`` sidecars
are static activation-quant metadata and are dropped. Embeddings, norms and ``lm_head`` are bf16.

**Dtype policy.** ``--compute-dtype`` (default ``bfloat16``) is the dtype the layer runs in;
``--dtype`` (default ``bfloat16``) is the dtype K/V are stored in. HF's RMSNorm and eager softmax
already accumulate in fp32 internally. Use ``--compute-dtype float32`` for a higher-precision
golden (2x memory, much slower on GPU).

**Attention backend.** ``--attn sdpa`` (default) lets PyTorch pick flash / mem-efficient kernels on
CUDA, so the ``[96, S, S]`` score matrix is never materialised; HF's sdpa path sets
``is_causal=True`` when no mask is passed and ``q_len > 1``. ``--attn eager`` builds an explicit
additive causal mask and materialises scores (only viable for short sequences). Either way, layer 0
is re-run on a short prefix and its output compared position-by-position with the full-sequence
run: a causal layer gives the same prefix regardless of what follows, a non-causal one does not.
That check does not depend on the backend, so a silently non-causal attention cannot slip
through. Disable with ``--no-causal-check``.

**Output** (the MiniMax-style golden layout every prefill package's PCC check reads)::

    {--out}/
        metadata.json                    # prompt, token_ids, model info, dtype policy
        kv_cache/
            layer_0.safetensors          # key_cache_layer_0   [1, num_kv_heads, seq_len, head_dim]
            ...                          # value_cache_layer_0 [1, num_kv_heads, seq_len, head_dim]
            layer_87.safetensors

``key_cache_layer_<i>`` is **post-RoPE K in the HF half-split rotary convention** (what the layer's
``DynamicCache`` holds); ``value_cache_layer_<i>`` is raw V. The TT stack swizzles q/k to Meta
interleaved order (``mistral_medium_d_p/tt/checkpoint.py::load_state_dict``), so a device-side
comparison must permute the golden K's lanes first -- see
``models/demos/llama31_8b_d_p/scripts/verify_golden_kv.py::hf_to_meta_lane_permutation``.

Usage (from the tt-metal root, or with the file copied anywhere)::

    export HF_MODEL=/path/to/Mistral-Medium-3.5-128B
    S=models/demos/mistral_medium_d_p/scripts

    # exactly 5120 real tokens: the prompt is tiled until it tokenizes to >= 5120, then cut
    python3 $S/generate_golden_kv_cache.py --prompt-file $S/prompts/bringup_prompt.txt \\
        --repeat-prompt-to 5120 --out /path/to/goldens/mistral_medium_s5120

    # exact token ids from a device test instead of a prompt
    python3 $S/generate_golden_kv_cache.py --token-ids-json ids.json --out /path/to/goldens/case_a

    # first 4 layers only, fp32 compute, on CPU
    python3 $S/generate_golden_kv_cache.py --prompt "hi" --num-layers 4 \\
        --compute-dtype float32 --device cpu --out /tmp/mm_smoke
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

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
_FP8_DTYPES = tuple(
    d
    for d in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
    )
    if d is not None
)
# fp8 sidecars: consumed by the dequant (weight_scale*) or dropped (activation/input scales).
_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale", ".activation_scale", ".input_scale")
_LAYER0_Q = "layers.0.self_attn.q_proj.weight"


def _raise_cpu_time_limit() -> None:
    """RLIMIT_CPU counts CPU-seconds over all threads; a CPU run with N threads drains a 24h soft
    cap in 24h/N wall-clock and dies with SIGXCPU. Raise soft -> hard (allowed unprivileged)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
            print(f"[limit] raised RLIMIT_CPU soft {soft}s -> {hard}")
        except (ValueError, OSError) as exc:  # pragma: no cover - environment dependent
            print(f"[limit] WARNING: could not raise RLIMIT_CPU (soft={soft}s): {exc}", file=sys.stderr)


class _ShardReader:
    """Lazy, mmapped, per-tensor reader over a (sharded) safetensors checkpoint with fp8 dequant."""

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

    def raw(self, key: str) -> torch.Tensor:
        """The stored tensor, dtype untouched (fp8 stays fp8)."""
        assert key in self.weight_map, f"{key!r} not in the checkpoint ({len(self.weight_map)} keys)"
        shard = self.weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(str(self.dir / shard), framework="pt")
        return self._handles[shard].get_tensor(key)

    def get(self, key: str, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Dequantized (if fp8) tensor in ``dtype`` on ``device``."""
        t = self.raw(key)
        if t.dtype in _FP8_DTYPES:
            scale_key = next((key + s for s in ("_scale_inv", "_scale") if key + s in self.weight_map), None)
            assert scale_key is not None, f"{key} is fp8 but has no sibling weight_scale_inv / weight_scale"
            scale = self.raw(scale_key)
            assert scale.numel() == 1, (
                f"{key}: expected a per-tensor (scalar) fp8 scale, got {tuple(scale.shape)}; "
                "block-wise fp8 needs a block dequant, which this script does not do"
            )
            s = scale.reshape(()).to(torch.float32)
            assert torch.isfinite(s) and s > 0, f"{key}: bad fp8 scale {s.item()}"
            # `weight_scale_inv` is the DEQUANT multiplier (DeepSeek convention, scalar here).
            return (t.to(device).to(torch.float32) * s.to(device)).to(dtype)
        return t.to(device=device, dtype=dtype)

    def layer_state(self, prefix: str, *, dtype: torch.dtype, device: torch.device) -> dict:
        """``{suffix: tensor}`` for every non-sidecar key under ``prefix`` (e.g. ``...layers.7.``)."""
        return {
            k[len(prefix) :]: self.get(k, dtype=dtype, device=device)
            for k in self.weight_map
            if k.startswith(prefix) and not k.endswith(_SCALE_SUFFIXES)
        }


def detect_layout(reader: _ShardReader) -> tuple[str, str, int]:
    """Find the text-backbone key prefix without assuming a wrapper style.

    ``Mistral3ForConditionalGeneration`` checkpoints put the text tower under
    ``model.language_model.layers.N.``; other wrappers use ``language_model.model.layers.N.``; a
    bare causal-LM export uses ``model.layers.N.``. Anchor on layer 0's q_proj and derive the rest.
    Returns ``(layers_prefix, embed_key, num_layers_in_checkpoint)``.
    """
    q_keys = [k for k in reader.keys() if k.endswith(_LAYER0_Q)]
    assert len(q_keys) == 1, f"expected exactly one key ending in {_LAYER0_Q!r}, found {q_keys}"
    layers_prefix = q_keys[0][: -len("0.self_attn.q_proj.weight")]  # e.g. "model.language_model.layers."
    stem = layers_prefix[: -len("layers.")]
    embed_key = stem + "embed_tokens.weight"
    assert embed_key in reader.weight_map, f"{embed_key!r} not found next to {layers_prefix!r}"
    n_layers = 1 + max(
        int(k[len(layers_prefix) :].split(".", 1)[0]) for k in reader.keys() if k.startswith(layers_prefix)
    )
    return layers_prefix, embed_key, n_layers


def load_text_config(model_path: Path):
    """The ``Ministral3Config`` for the text backbone, unwrapping ``text_config`` if present."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(model_path))
    text = getattr(cfg, "text_config", None) or cfg
    if getattr(text, "model_type", None) != "ministral3":
        print(
            f"[config] WARNING: model_type={getattr(text, 'model_type', None)!r}, expected 'ministral3'; "
            "this script uses the Ministral3 decoder layer regardless",
            file=sys.stderr,
        )
    return text


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate the golden KV cache for Mistral-Medium-3.5-128B prefill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", type=str, help="Direct prompt text")
    src.add_argument("--prompt-json", type=Path, help='JSON file: {"prompt": "..."} or a bare string')
    src.add_argument("--prompt-file", type=Path, help="Plain-text prompt file")
    src.add_argument("--token-ids-json", type=Path, help="JSON list of token ids (bypasses the tokenizer)")

    ap.add_argument("--out", type=Path, required=True, help="Output trace directory (kv_cache/ is created inside)")
    ap.add_argument("--model-path", type=str, default=None, help="HF checkpoint dir (default: $HF_MODEL)")
    ap.add_argument("--tokenizer-path", type=str, default=None, help="Tokenizer dir (default: --model-path)")
    ap.add_argument("--max-tokens", type=int, default=None, help="Truncate the prompt to this many tokens")
    ap.add_argument(
        "--repeat-prompt-to",
        type=int,
        default=None,
        help="Tile the prompt text (separated by blank lines) until it tokenizes to at least N tokens, then "
        "truncate to exactly N real tokens. Use this instead of --pad-to when the golden should be all real tokens",
    )
    ap.add_argument(
        "--pad-to",
        type=int,
        default=None,
        help="Right-pad with pad_token_id (else eos) to exactly this many tokens; metadata records "
        "n_real_tokens so consumers can PCC only the real prefix",
    )
    ap.add_argument("--no-chat-template", action="store_true", help="Use the raw prompt, no chat template")
    ap.add_argument("--num-layers", type=int, default=None, help="Only emit the first N layers (debug)")
    ap.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16", help="Stored K/V dtype")
    ap.add_argument(
        "--compute-dtype", choices=sorted(_DTYPES), default="bfloat16", help="Dtype the decoder layers run in"
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device for the forward (default: cuda if available)",
    )
    ap.add_argument(
        "--attn",
        choices=["sdpa", "eager"],
        default="sdpa",
        help="HF attention backend: sdpa (flash/mem-efficient, is_causal) or eager (explicit mask, [H,S,S] scores)",
    )
    ap.add_argument(
        "--no-causal-check",
        action="store_true",
        help="Skip the layer-0 prefix-vs-full causality self-check",
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


def tokenize_prompt(tokenizer, prompt, *, max_tokens, use_chat_template, repeat_to=None):
    if repeat_to is not None:
        base = prompt.strip()
        n_copies = 1
        while len(tokenizer(base if n_copies == 1 else "\n\n".join([base] * n_copies))["input_ids"]) < repeat_to:
            n_copies += 1
        prompt = "\n\n".join([base] * n_copies)
        print(f"[tokenize] tiled the prompt x{n_copies} ({len(prompt)} chars) to reach {repeat_to} tokens")
        max_tokens = repeat_to
    if use_chat_template:
        # transformers 5.x: tokenize=True returns a BatchEncoding unless return_dict=False.
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
    else:
        ids = tokenizer(prompt)["input_ids"]
    ids = [int(i) for i in ids]
    if max_tokens and len(ids) > max_tokens:
        print(f"[tokenize] truncating {len(ids)} -> {max_tokens} tokens")
        ids = ids[:max_tokens]
    return ids


def pad_ids(ids, *, pad_to, pad_id):
    n_real = len(ids)
    if pad_to is None:
        return ids, n_real
    assert pad_to >= n_real, f"--pad-to {pad_to} is shorter than the prompt ({n_real} tokens)"
    assert pad_id is not None, "no pad_token_id / eos_token_id available; cannot pad"
    print(f"[tokenize] padded {n_real} -> {pad_to} tokens with id {pad_id}")
    return ids + [int(pad_id)] * (pad_to - n_real), n_real


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / a.norm().clamp_min(1e-12)).item()


@torch.no_grad()
def generate(  # noqa: C901 - a linear driver; splitting it would hide the streaming order
    *,
    ckpt_dir: Path,
    cfg,
    token_ids,
    out_dir: Path,
    compute_dtype=torch.bfloat16,
    store_dtype=torch.bfloat16,
    device="cuda",
    attn="sdpa",
    num_layers=None,
    causal_check=True,
    progress=print,
):
    """Stream the reference layer by layer, writing ``layer_<i>.safetensors`` as it goes."""
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.models.ministral3 import modeling_ministral3 as mm

    device = torch.device(device)
    cfg._attn_implementation = attn
    total_layers = cfg.num_hidden_layers
    n_layers = total_layers if num_layers is None else min(num_layers, total_layers)
    head_dim = cfg.head_dim
    num_kv_heads = cfg.num_key_value_heads
    seq_len = len(token_ids)

    kv_dir = Path(out_dir) / "kv_cache"
    kv_dir.mkdir(parents=True, exist_ok=True)

    reader = _ShardReader(ckpt_dir)
    layers_prefix, embed_key, ckpt_layers = detect_layout(reader)
    assert ckpt_layers >= n_layers, f"checkpoint has {ckpt_layers} layers, config says {total_layers}"
    progress(
        f"[load] {len(reader.weight_map)} tensors; text tower at {layers_prefix!r} "
        f"({ckpt_layers} layers); streaming per layer via mmap -> {device}"
    )

    # --- embeddings: one index_select on the mmapped table, which is then dropped ---
    ids = torch.tensor(token_ids, dtype=torch.long)
    embed = reader.raw(embed_key)
    hidden = embed.index_select(0, ids).reshape(1, seq_len, cfg.hidden_size).to(device=device, dtype=compute_dtype)
    del embed

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    rotary = mm.Ministral3RotaryEmbedding(cfg, device=device)
    position_embeddings = rotary(hidden, position_ids)  # YaRN cos/sin incl. attention_scaling, in compute dtype
    progress(
        f"[rope] type={cfg.rope_parameters.get('rope_type')} theta={cfg.rope_parameters.get('rope_theta')} "
        f"factor={cfg.rope_parameters.get('factor')} attention_scaling={rotary.attention_scaling:.6f}"
    )

    def make_mask(x, pos):
        if attn != "eager":
            return None  # HF sdpa path: mask None + q_len > 1 + module.is_causal -> is_causal=True
        mask = create_causal_mask(
            config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=None, position_ids=pos
        )
        assert mask is not None, "eager attention with a None mask is silently NON-causal; refusing"
        return mask

    causal_mask = make_mask(hidden, position_ids)
    expected = (1, num_kv_heads, seq_len, head_dim)
    facts = {"causal_check_rel_err": None}
    t0 = time.time()
    for layer_idx in range(n_layers):
        # Build on `meta` (no 1.4B-param random init), then bind the checkpoint tensors directly.
        with torch.device("meta"):
            layer = mm.Ministral3DecoderLayer(cfg, layer_idx)
        state = reader.layer_state(f"{layers_prefix}{layer_idx}.", dtype=compute_dtype, device=device)
        missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
        assert not missing, f"layer {layer_idx}: checkpoint is missing {sorted(missing)}"
        assert not unexpected, f"layer {layer_idx}: checkpoint has unused keys {sorted(unexpected)}"
        layer.eval()
        if attn != "eager":
            assert getattr(layer.self_attn, "is_causal", False) is True, "sdpa path relies on self_attn.is_causal"

        x_in = hidden
        cache = DynamicCache(config=cfg)
        out = layer(
            x_in,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
        hidden = out if torch.is_tensor(out) else out[0]
        key_cache = cache.layers[layer_idx].keys  # post-RoPE K, HF half-split convention
        value_cache = cache.layers[layer_idx].values  # raw V
        assert tuple(key_cache.shape) == expected, f"layer {layer_idx}: K {tuple(key_cache.shape)} != {expected}"
        assert tuple(value_cache.shape) == expected, f"layer {layer_idx}: V {tuple(value_cache.shape)} != {expected}"
        assert (
            torch.isfinite(key_cache).all() and torch.isfinite(value_cache).all()
        ), f"layer {layer_idx}: non-finite K/V"
        assert torch.isfinite(hidden).all(), f"layer {layer_idx}: non-finite hidden state"

        if causal_check and layer_idx == 0 and seq_len > 1:
            # Causality: the first n outputs must not depend on tokens n.. .
            n = min(64, seq_len - 1)
            pe = (position_embeddings[0][:, :n], position_embeddings[1][:, :n])
            c2 = DynamicCache(config=cfg)
            o2 = layer(
                x_in[:, :n],
                attention_mask=make_mask(x_in[:, :n], position_ids[:, :n]),
                position_ids=position_ids[:, :n],
                past_key_values=c2,
                use_cache=True,
                position_embeddings=pe,
            )
            o2 = o2 if torch.is_tensor(o2) else o2[0]
            err = _rel_err(hidden[:, :n], o2)
            facts["causal_check_rel_err"] = err
            facts["causal_check_prefix"] = n
            assert err < 2e-2, (
                f"layer 0 prefix output differs from the full-sequence run (rel err {err:.3e}): "
                "attention is leaking future tokens -- this golden would be wrong from layer 1 onward"
            )
            progress(f"[check] layer 0 causality OK: {n}-token prefix rel err {err:.2e}")
            del c2, o2

        save_file(
            {
                f"key_cache_layer_{layer_idx}": key_cache.to("cpu", store_dtype).contiguous(),
                f"value_cache_layer_{layer_idx}": value_cache.to("cpu", store_dtype).contiguous(),
            },
            str(kv_dir / f"layer_{layer_idx}.safetensors"),
        )
        del layer, state, cache, key_cache, value_cache, x_in, out
        if device.type == "cuda":
            torch.cuda.empty_cache()
        progress(f"[forward] layer {layer_idx:>2}/{n_layers - 1} saved  ({time.time() - t0:.1f}s elapsed)")

    facts.update(
        {
            "num_layers": n_layers,
            "model_num_layers": total_layers,
            "num_kv_heads": num_kv_heads,
            "num_attention_heads": cfg.num_attention_heads,
            "head_dim": head_dim,
            "hidden_size": cfg.hidden_size,
            "n_tokens": seq_len,
            "key_cache_shape": list(expected),
            "value_cache_shape": list(expected),
            "forward_time_seconds": time.time() - t0,
            "checkpoint_layers_prefix": layers_prefix,
            "rope_parameters": dict(cfg.rope_parameters),
            "rope_attention_scaling": float(rotary.attention_scaling),
        }
    )
    return facts


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

    cfg = load_text_config(ckpt_dir)
    print(
        f"[config] {getattr(cfg, 'model_type', '?')}: {cfg.num_hidden_layers} layers, "
        f"{cfg.num_attention_heads} Q heads, {cfg.num_key_value_heads} KV heads, head_dim {cfg.head_dim}, "
        f"hidden {cfg.hidden_size}, llama_4_scaling_beta={cfg.rope_parameters.get('llama_4_scaling_beta')}"
    )

    from transformers import AutoTokenizer

    prompt = None
    use_chat_template = not args.no_chat_template
    if args.token_ids_json:
        with open(args.token_ids_json) as fh:
            token_ids = [int(i) for i in json.load(fh)]
        pad_id = getattr(cfg, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(cfg, "eos_token_id", None)
        print(f"[load] {len(token_ids)} token ids from {args.token_ids_json}")
        use_chat_template = False
    else:
        prompt = load_prompt(args)
        print(f"[load] prompt: {len(prompt)} characters")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or str(ckpt_dir))
        token_ids = tokenize_prompt(
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            use_chat_template=use_chat_template,
            repeat_to=args.repeat_prompt_to,
        )
        if args.repeat_prompt_to is not None:
            assert (
                len(token_ids) == args.repeat_prompt_to
            ), f"got {len(token_ids)} tokens, wanted {args.repeat_prompt_to}"
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    token_ids, n_real = pad_ids(token_ids, pad_to=args.pad_to, pad_id=pad_id)
    print(f"[load] {len(token_ids)} tokens ({n_real} real)")

    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 8)
    print(
        f"[forward] Ministral3DecoderLayer x{args.num_layers or cfg.num_hidden_layers} on {args.device}, "
        f"compute {args.compute_dtype}, store {args.dtype}, attn={args.attn}"
    )

    facts = generate(
        ckpt_dir=ckpt_dir,
        cfg=cfg,
        token_ids=token_ids,
        out_dir=out_dir,
        compute_dtype=_DTYPES[args.compute_dtype],
        store_dtype=_DTYPES[args.dtype],
        device=args.device,
        attn=args.attn,
        num_layers=args.num_layers,
        causal_check=not args.no_causal_check,
        progress=lambda msg: print(msg, flush=True),
    )

    metadata = {
        "model_path": str(ckpt_dir),
        "model_config": ckpt_dir.name,
        "model_type": getattr(cfg, "model_type", None),
        "reference": (
            "transformers.models.ministral3.modeling_ministral3.Ministral3DecoderLayer, one layer at a time, "
            "weights streamed per layer via mmap, per-tensor fp8 dequantized (w = fp8 * weight_scale_inv)"
        ),
        "reference_dtype_policy": (
            f"fp8 weights dequantized to {args.compute_dtype}; bf16 norms/embeddings cast to {args.compute_dtype}; "
            f"layers run in {args.compute_dtype} (HF RMSNorm / eager softmax accumulate fp32 internally); "
            f"K/V stored as {args.dtype}"
        ),
        "compute_dtype": args.compute_dtype,
        "device": args.device,
        "attn_implementation": args.attn,
        "explicit_causal_mask": args.attn == "eager",
        "kv_convention": "post-RoPE K (HF half-split rotary), raw V, HF layout [1, num_kv_heads, seq, head_dim]",
        "prompt_source": (
            str(args.token_ids_json)
            if args.token_ids_json
            else str(args.prompt_json)
            if args.prompt_json
            else str(args.prompt_file)
            if args.prompt_file
            else "direct"
        ),
        "prompt": None if prompt is None else (prompt if len(prompt) <= 500 else prompt[:500] + "..."),
        "prompt_length_chars": None if prompt is None else len(prompt),
        "prompt_repeated_to_tokens": args.repeat_prompt_to,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
