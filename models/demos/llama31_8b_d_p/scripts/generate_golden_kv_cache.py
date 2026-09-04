#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate the **fp32** golden KV-cache trace for Llama-3.1-8B prefill. Gate: `G-GOLDEN`.

Runs the HF reference one `transformers.models.llama.modeling_llama.LlamaDecoderLayer` at a time,
streaming that layer's weights out of the safetensors shards and writing its post-RoPE **K** and raw
**V** before moving on, so 32 layers of KV never sit in RAM together.

**Imports no ttnn.** This is the reference side; the device-vs-golden scoring lives in `G-CHUNK`
(`tests/unit/test_attention_chunked_vs_ref.py`) and the structural check in `verify_golden_kv.py`.

Output layout — copied exactly from `models/demos/minimax_m3/scripts/generate_golden_kv_cache.py`'s
header (`:29-30`) because the engine's producer read-back expects it
(`models/demos/common/prefill/runners/prefill_producer.py:511`):

    {trace_dir}/metadata.json                     # prompt, token_ids, model info
    {trace_dir}/kv_cache/layer_<i>.safetensors    # key_cache_layer_<i>, value_cache_layer_<i>
                                                  # [1, num_kv_heads, seq_len, head_dim], HF layout

**fp32, not the template's bf16** (`DEC-059`). Both in-repo generators store bf16
(`models/demos/gpt_oss_d_p/scripts/generate_golden_kv_cache.py:111`,
`models/demos/minimax_m3/scripts/generate_golden_kv_cache.py:143`), which is recipe §2.1(a)'s trap
verbatim: a reference held at the device's own storage dtype shares the device's rounding and
reports a flattered PCC.

**Three things this script does that the templates do not, each closing a named trap:**

1. **An explicit causal mask.** `eager_attention_forward` applies only the mask it is handed, so
   `attention_mask=None` is a silently **non-causal** reference (`LANDMINES.md`, recipe P1 trap 3).
   The mask comes from `transformers.masking_utils.create_causal_mask` — the same call
   `LlamaModel.forward` makes
   (`python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:399`) — and
   is asserted non-`None`.
2. **The streamed driver is proved against `LlamaModel`'s own loop at `rtol=atol=0`** (`--verify-loop`,
   on by default), so the streaming is not itself the thing under test. This is `G-GOLDEN`'s
   "streamed driver == HF's own loop bit-exactly".
3. **Weights are read straight from the shards**, never through `from_pretrained`, which would load
   at the checkpoint's `torch_dtype` (bf16) and hand back a bf16-rounded reference
   (`LANDMINES.md` "`from_pretrained` for a torch reference", `DEC-006`).

The trace directory is `$PREFILL_TRACE_DIR` (the engine already owns that variable —
`models/demos/common/prefill/runners/prefill_runner.py:397`), or `--out`.

Run:
    export HF_MODEL=/home/mstojkovic/models/Llama-3.1-8B-Instruct
    export PREFILL_TRACE_DIR=/home/mstojkovic/prefill_traces/llama31_8b_d_p/s512
    python3 models/demos/llama31_8b_d_p/scripts/generate_golden_kv_cache.py --tokens 512
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_PKG_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_CONFIG_DIR = _PKG_ROOT / "configs" / "Llama-3.1-8B-Instruct"

# The bring-up prompt. Identical to `tests/unit/test_model_vs_ref.py::PROMPT` so a golden trace and
# `G-MODEL`'s top-1 run see the same tokens (`DEC-056` recorded the distribution; this keeps it).
DEFAULT_PROMPT = (
    "The capital of France is Paris. The capital of Italy is Rome. The capital of Japan is Tokyo. "
    "Large language models are trained on text and predict the next token in a sequence. "
)


# ------------------------------------------------------------------------------------------------
# checkpoint streaming
# ------------------------------------------------------------------------------------------------
class ShardReader:
    """Read individual checkpoint tensors out of the safetensors shards, one at a time.

    `safe_open` memory-maps the shard, so a layer's weights are materialised only when
    `get_tensor` is called and are freed with the returned tensor. That is what keeps peak RSS at
    one fp32 layer (~0.9 GB) instead of the whole 15 GB checkpoint.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        index = Path(model_path) / "model.safetensors.index.json"
        if not index.is_file():
            raise FileNotFoundError(f"no safetensors index at {index}; is HF_MODEL a checkpoint directory?")
        with open(index) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._open: dict[str, object] = {}

    def _handle(self, shard: str):
        if shard not in self._open:
            self._open[shard] = safe_open(str(Path(self.model_path) / shard), framework="pt", device="cpu")
        return self._open[shard]

    def get(self, key: str) -> torch.Tensor:
        if key not in self.weight_map:
            raise KeyError(f"{key!r} is not in the checkpoint index")
        return self._handle(self.weight_map[key]).get_tensor(key)

    def substate_fp32(self, prefix: str) -> dict:
        """Every key under `prefix`, with the prefix stripped, cast to **fp32**."""
        out = {}
        for key in self.weight_map:
            if key.startswith(prefix):
                out[key[len(prefix) :]] = self.get(key).float()
        if not out:
            raise KeyError(f"no checkpoint keys under {prefix!r}")
        return out

    def close(self):
        self._open.clear()


# ------------------------------------------------------------------------------------------------
# the reference driver
# ------------------------------------------------------------------------------------------------
def load_config(num_layers: int | None):
    """The bundled `config.json` as a `transformers` config, `eager` attention, fp32 math.

    The bundled copy is byte-identical to the staged checkpoint's (`DEC-001`, asserted by
    `tests/unit/test_reference_model.py::test_bundled_config_matches_checkpoint`), so this needs
    neither the network nor the checkpoint to describe the architecture.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(BUNDLED_CONFIG_DIR))
    # `eager` **and** an explicit mask; see the module docstring, item 1.
    cfg._attn_implementation = "eager"
    if num_layers is not None:
        assert 0 < num_layers <= cfg.num_hidden_layers, f"num_layers must be in (0, {cfg.num_hidden_layers}]"
        cfg.num_hidden_layers = num_layers
    return cfg


def build_prefill_inputs(cfg, reader: ShardReader, token_ids: list[int]):
    """`(hidden_states, position_ids, position_embeddings, causal_mask)` — fp32, one-shot prefill.

    Built with `LlamaModel.forward`'s own calls so the streamed driver and the real loop cannot
    disagree about the RoPE tables or the mask
    (`python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:399`,
    `:408`).
    """
    from transformers.masking_utils import create_causal_mask
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    embed = reader.get("model.embed_tokens.weight").float()
    tokens = torch.tensor(token_ids, dtype=torch.long).reshape(1, -1)
    hidden = torch.nn.functional.embedding(tokens, embed)
    del embed

    position_ids = torch.arange(hidden.shape[1]).unsqueeze(0)
    rotary = LlamaRotaryEmbedding(config=cfg).float()
    position_embeddings = rotary(hidden, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=cfg,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    # A `None` mask here would make the reference non-causal with no exception anywhere — the
    # single most expensive silent failure available to this script (`LANDMINES.md`, P1 trap 3).
    assert causal_mask is not None, (
        "create_causal_mask returned None: eager attention applies only the mask it is handed, so "
        "the reference would be silently NON-CAUSAL (recipe P1 trap 3)"
    )
    return hidden, position_ids, position_embeddings, causal_mask


@torch.no_grad()
def run_streamed(cfg, reader: ShardReader, token_ids: list[int], *, on_layer=None):
    """Drive the layers one at a time; call `on_layer(idx, k, v)` with fp32 post-RoPE K / raw V.

    Returns the **pre**-final-norm hidden state (the last layer's output) and the post-norm one, in
    that order. Both are returned rather than only one because the post-norm stream is what HF calls
    `last_hidden_state` and the pre-norm one is what a per-layer comparison needs — and confusing the
    two costs a debugging pass (`R-019`, `DEC-052`).

    K and V come out of a `transformers.cache_utils.DynamicCache`: `LlamaAttention.forward` calls
    `past_key_values.update(key_states, value_states, layer_idx)` **after** RoPE on K and with V
    untouched
    (`python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:270`),
    which is exactly the pair the device cache holds (`tt/attention/kv_cache.py`).
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

    hidden, position_ids, position_embeddings, causal_mask = build_prefill_inputs(cfg, reader, token_ids)

    for layer_idx in range(cfg.num_hidden_layers):
        layer = LlamaDecoderLayer(cfg, layer_idx).float().eval()
        state = reader.substate_fp32(f"model.layers.{layer_idx}.")
        # `strict=True` raises on any mismatch, so a renamed or missing checkpoint key cannot reach
        # the forward pass as a randomly-initialised weight.
        layer.load_state_dict(state, strict=True)
        del state

        cache = DynamicCache(config=cfg)
        hidden = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
        if on_layer is not None:
            on_layer(layer_idx, cache.layers[layer_idx].keys.float(), cache.layers[layer_idx].values.float())
        del layer, cache

    pre_norm = hidden
    norm = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).float().eval()
    norm.load_state_dict({"weight": reader.get("model.norm.weight").float()}, strict=True)
    return pre_norm, norm(pre_norm)


@torch.no_grad()
def run_llamamodel_loop(cfg, reader: ShardReader, token_ids: list[int]):
    """The same forward through **`LlamaModel` itself**, returning `(per_layer_kv, post_norm)`.

    This is the control on the streaming: `G-GOLDEN` requires the streamed driver to equal HF's own
    loop at `rtol=atol=0`, so that a bug in *this file's* layer plumbing cannot be mistaken for the
    reference. It needs the whole checkpoint in fp32 (~32 GB), which is why it is a flag.
    """
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import LlamaModel

    model = LlamaModel(cfg).float().eval()
    weights = {"embed_tokens.weight": reader.get("model.embed_tokens.weight").float()}
    weights["norm.weight"] = reader.get("model.norm.weight").float()
    for layer_idx in range(cfg.num_hidden_layers):
        for name, tensor in reader.substate_fp32(f"model.layers.{layer_idx}.").items():
            weights[f"layers.{layer_idx}.{name}"] = tensor
    incompatible = model.load_state_dict(weights, strict=False)
    assert not [k for k in incompatible.missing_keys if "rotary" not in k], incompatible.missing_keys
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys
    del weights

    tokens = torch.tensor(token_ids, dtype=torch.long).reshape(1, -1)
    cache = DynamicCache(config=cfg)
    out = model(input_ids=tokens, past_key_values=cache, use_cache=True)
    kv = [(cache.layers[i].keys.float(), cache.layers[i].values.float()) for i in range(cfg.num_hidden_layers)]
    post_norm = out.last_hidden_state.float()
    del model, cache, out
    return kv, post_norm


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def tokenize(model_path: str, prompt: str, n_tokens: int | None, use_chat_template: bool):
    """Tokenize, then **tile** to exactly `n_tokens` (or leave as-is when `n_tokens` is `None`).

    Tiling rather than padding, and the same tiling `tests/unit/test_model_vs_ref.py::_prompt_tokens`
    does: the gate needs a tile-aligned length whose activations are still in-distribution for the
    real weights, and a pad-token tail is not (`DEC-056`).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if use_chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
        )
    else:
        ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if n_tokens is None:
        return list(ids)
    assert n_tokens > 0, f"--tokens must be positive, got {n_tokens}"
    return (list(ids) * (n_tokens // len(ids) + 1))[:n_tokens]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate the fp32 golden KV-cache trace for Llama-3.1-8B prefill (G-GOLDEN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--prompt", type=str, default=None, help="prompt text (default: the bring-up prompt)")
    group.add_argument("--prompt-json", type=Path, default=None, help='JSON file with {"prompt": "..."}')
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="trace directory (default: $PREFILL_TRACE_DIR — the engine's own variable)",
    )
    ap.add_argument("--model-path", type=str, default=None, help="checkpoint directory (default: $HF_MODEL)")
    ap.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="tile/truncate the prompt to exactly this many tokens (default: the prompt's own length)",
    )
    ap.add_argument("--num-layers", type=int, default=None, help="layers to capture (default: all 32)")
    ap.add_argument("--chat-template", action="store_true", help="apply the chat template (off by default)")
    ap.add_argument(
        "--verify-loop",
        dest="verify_loop",
        action="store_true",
        default=True,
        help="also run LlamaModel's own loop and require rtol=atol=0 agreement (default: on; G-GOLDEN)",
    )
    ap.add_argument(
        "--no-verify-loop",
        dest="verify_loop",
        action="store_false",
        help="skip the LlamaModel cross-check (needs ~32 GB of RAM at 32 layers)",
    )
    ap.add_argument(
        "--zero-layer",
        type=int,
        default=None,
        help="NEGATIVE CONTROL: write layer N's K/V as zeros. verify_golden_kv.py must reject the trace.",
    )
    return ap.parse_args(argv)


def resolve_prompt(args) -> tuple[str, str]:
    if args.prompt_json is not None:
        with open(args.prompt_json) as f:
            data = json.load(f)
        prompt = data["prompt"] if isinstance(data, dict) else data
        return prompt, str(args.prompt_json)
    if args.prompt is not None:
        return args.prompt, "direct"
    return DEFAULT_PROMPT, "package default (bring-up prompt)"


def main(argv=None) -> int:
    args = parse_args(argv)

    model_path = args.model_path or os.environ.get("HF_MODEL")
    if not model_path:
        print("ERROR: set $HF_MODEL or pass --model-path", file=sys.stderr)
        return 1
    out_dir = args.out or (Path(os.environ["PREFILL_TRACE_DIR"]) if os.environ.get("PREFILL_TRACE_DIR") else None)
    if out_dir is None:
        print("ERROR: set $PREFILL_TRACE_DIR or pass --out", file=sys.stderr)
        return 1
    out_dir = Path(out_dir)

    torch.set_num_threads(os.cpu_count() or 32)
    cfg = load_config(args.num_layers)
    prompt, prompt_source = resolve_prompt(args)
    token_ids = tokenize(model_path, prompt, args.tokens, args.chat_template)
    seq_len = len(token_ids)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    expected = (1, cfg.num_key_value_heads, seq_len, head_dim)

    kv_dir = out_dir / "kv_cache"
    kv_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[golden] {cfg.num_hidden_layers} layers, {seq_len} tokens, "
        f"{cfg.num_key_value_heads} KV heads, head_dim={head_dim}, store dtype=float32",
        flush=True,
    )
    print(f"[golden] checkpoint {model_path} -> trace {out_dir}", flush=True)

    saved = []

    def save_layer(layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        for name, tensor in (("K", k), ("V", v)):
            assert tuple(tensor.shape) == expected, f"layer {layer_idx} {name} {tuple(tensor.shape)} != {expected}"
            assert tensor.dtype == torch.float32, f"layer {layer_idx} {name} is {tensor.dtype}, not float32"
            assert torch.isfinite(tensor).all(), f"layer {layer_idx} {name} holds a NaN or an Inf"
        if args.zero_layer == layer_idx:
            # The negative control: a structurally valid file whose contents are wrong.
            print(f"[golden] NEGATIVE CONTROL: zeroing layer {layer_idx}", flush=True)
            k, v = torch.zeros_like(k), torch.zeros_like(v)
        save_file(
            {f"key_cache_layer_{layer_idx}": k.contiguous(), f"value_cache_layer_{layer_idx}": v.contiguous()},
            str(kv_dir / f"layer_{layer_idx}.safetensors"),
        )
        saved.append(layer_idx)
        print(f"[golden] layer {layer_idx:>2}: K {tuple(k.shape)} V {tuple(v.shape)} written", flush=True)

    reader = ShardReader(model_path)
    t0 = time.time()
    _, post_norm = run_streamed(cfg, reader, token_ids, on_layer=save_layer)
    stream_seconds = time.time() - t0
    assert saved == list(range(cfg.num_hidden_layers)), f"layers written out of order or missing: {saved}"
    print(f"[golden] streamed driver: {cfg.num_hidden_layers} layers in {stream_seconds:.1f}s", flush=True)

    loop_check = None
    if args.verify_loop:
        t1 = time.time()
        ref_kv, ref_post_norm = run_llamamodel_loop(cfg, reader, token_ids)
        worst_k = worst_v = 0.0
        for layer_idx in range(cfg.num_hidden_layers):
            with safe_open(str(kv_dir / f"layer_{layer_idx}.safetensors"), framework="pt") as h:
                got_k = h.get_tensor(f"key_cache_layer_{layer_idx}")
                got_v = h.get_tensor(f"value_cache_layer_{layer_idx}")
            ref_k, ref_v = ref_kv[layer_idx]
            worst_k = max(worst_k, float((ref_k - got_k).abs().max()))
            worst_v = max(worst_v, float((ref_v - got_v).abs().max()))
            assert torch.equal(ref_k, got_k), f"layer {layer_idx} K differs from LlamaModel's own loop"
            assert torch.equal(ref_v, got_v), f"layer {layer_idx} V differs from LlamaModel's own loop"
        hidden_delta = float((ref_post_norm - post_norm).abs().max())
        assert torch.equal(ref_post_norm, post_norm), f"post-norm hidden differs, max|delta| = {hidden_delta:.3e}"
        loop_check = {
            "layers_compared": cfg.num_hidden_layers,
            "max_abs_delta_k": worst_k,
            "max_abs_delta_v": worst_v,
            "max_abs_delta_post_norm_hidden": hidden_delta,
            "seconds": time.time() - t1,
        }
        print(
            f"[golden] LlamaModel cross-check: {cfg.num_hidden_layers} layers bit-identical "
            f"(max|delta| K={worst_k:.1e} V={worst_v:.1e} hidden={hidden_delta:.1e}) "
            f"in {loop_check['seconds']:.1f}s",
            flush=True,
        )
        del ref_kv, ref_post_norm
    reader.close()

    metadata = {
        "model_path": str(model_path),
        "reference": "transformers.models.llama.modeling_llama.LlamaDecoderLayer, streamed one layer at a time",
        "reference_dtype_policy": "fp32 throughout: weights cast to float32 on load, all math in float32",
        "attn_implementation": cfg._attn_implementation,
        "causal_mask": "transformers.masking_utils.create_causal_mask, asserted non-None",
        "prompt_source": prompt_source,
        "prompt": prompt if len(prompt) <= 500 else prompt[:500] + "...",
        "prompt_length_chars": len(prompt),
        "prompt_tiled_to_tokens": args.tokens,
        "chat_template": args.chat_template,
        "token_ids": list(token_ids),
        "n_tokens": seq_len,
        "n_layers": cfg.num_hidden_layers,
        "num_layers": cfg.num_hidden_layers,
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": head_dim,
        "dtype": "float32",
        "kv_cache_format": "separate_k_v",
        "kv_layout": "HF: K post-RoPE (half-split rotary), V raw, [1, num_kv_heads, seq_len, head_dim]",
        "key_cache_shape": list(expected),
        "value_cache_shape": list(expected),
        "prefill_mode": "one_shot_streamed_layers",
        "stream_seconds": stream_seconds,
        "llamamodel_loop_check": loop_check,
        "zeroed_layer": args.zero_layer,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_gb = sum(f.stat().st_size for f in kv_dir.glob("*.safetensors")) / (1024**3)
    print(f"[golden] wrote {len(saved)} layer files ({total_gb:.2f} GB) + metadata.json to {out_dir}", flush=True)
    print(f"[golden] next: python3 models/demos/llama31_8b_d_p/scripts/verify_golden_kv.py {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
