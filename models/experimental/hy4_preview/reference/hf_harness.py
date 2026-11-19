# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Golden data from the real HF transformers ``hy_v4`` model (tenstorrent/tt-metal#65893).

Loads HYV4Model from the BF16 checkpoint truncated to its first N layers, runs a prompt through
it in prefill (no cache) and records, per layer, the state of the residual streams after each
sublayer and each sublayer's input and output. The record is cached on disk; the device tests
and the pure-torch reference (decoder.py) are compared against it.

Truncating is exact for what is recorded: a layer's output depends only on the layers below it.
The stream-collapsing head and the final norm that HYV4Model applies after its last layer are not
recorded, since after a truncated stack they are not the model's.

Loading. ``from_pretrained`` with a config cut to N layers memory-maps the tensors it needs out
of the 131 shards and nothing else, so the model costs page cache rather than process memory:
1.2 GB of weights for layer 0, 20 GB for each MoE layer, 1.5 GB for the embedding (about 102 GB
for layers 0-5). It keeps the checkpoint's BF16; HF holds the iHC parameters, the sinks, the router
bias and the indexer's k_norm / weights_proj in fp32 whatever the model dtype.

Dense attention and the indexer. HYV4Indexer.forward ends with
``topk = min(index_topk, T); index_scores.topk(topk)`` and HYV4Attention.forward masks every key
that is not among the picks. For a prompt of T <= index_topk (2048) tokens each query picks all
T keys, the DSA mask is empty and the layer computes plain causal attention; that also holds for
the "shared" layers, which reuse the picks of the last "full" layer. The recorder does not take
this on trust: for every layer it counts the keys a query may causally see that its picks leave
out (``dsa_dropped``). Zero means a dense causal core reproduces the layer.

This module imports transformers only inside the functions that run the HF model, so the device
tests, whose environment has no ``hy_v4``, can still read cached goldens through it.

A second kind of golden, for decode after a prefill (record_decode_golden, ``--decode-steps``):
the cut model with its head, final norm and LM head, run over a long prompt and its own greedy
continuation, with HF's cache rows of every position and the per-layer record of the decode
positions. Its layout is in that function's docstring.

Golden file (``torch.save``), all tensors without the batch dim, in the dtype HF produced them:
    meta         dict: checkpoint dir, num_layers, model dtype, prompt, seq_len, layer types, versions
    input_ids    [S] int64
    positions    [S] int64
    embedding    [S, hidden]            embed_tokens output, before it is expanded to n streams
    layers       list of N dicts with the decoder.STAGES tensors plus
                   q_resid [S, q_lora]            q_a_layernorm output (also the indexer's input)
                   kv_a    [S, kv_lora + rope]    kv_a_proj_with_mqa output: (kv_pass | k_rope before RoPE)
                   k_pass  [S, kv_lora]           kv_a_layernorm output: the cached latent minus k_rope
                   router_weights, router_indices [S, top_k]   MoE layers only
                   topk_indices [S, min(index_topk, S)] int32  the DSA picks this layer attends with
                   dsa_dropped  int               causally visible keys the picks leave out
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch

from models.experimental.hy4_preview.reference.decoder import STAGES

GOLDEN_ROOT = Path(os.environ.get("HY4_GOLDEN_DIR", "/mnt/data/zni/hy4/golden"))
HY4_HF_MODEL_ENV = "HY4_HF_MODEL"

# The default prompt: a little over 128 tokens through the chat template, cut to the first 128.
DEFAULT_PROMPT = (
    "A Blackhole Galaxy holds thirty-two accelerator chips wired as a four by eight mesh. Explain, "
    "step by step, how a large mixture-of-experts language model can be laid out on such a mesh: "
    "which weights are split across the rows, which across the columns, where the key-value cache "
    "lives for a very long context, and what has to travel between chips for every generated token. "
    "Then list the three numbers you would measure first, and say why each of them matters. Keep the "
    "answer short enough to read in two minutes, use plain words, and end with one open question."
)
DEFAULT_SEQ_LEN = 128
DEFAULT_NUM_LAYERS = 6

# HF module name -> checkpoint name, the inverse of transformers' conversion mapping for hy_v4.
_HF_TO_CHECKPOINT = (
    (r"\.attn_hc\.(fn|base|scale)$", r".hc_attn_layer.hc_pre.hc_\1"),
    (r"\.ffn_hc\.(fn|base|scale)$", r".hc_mlp_layer.hc_pre.hc_\1"),
    (r"\.self_attn\.sinks$", ".self_attn.learnable_sink_param"),
    (r"\.self_attn\.gate_proj\.", ".self_attn.linear_gate."),
    (r"(^|\.)hc_head\.hc_(fn|base|scale)$", r"\1hc_head.hc_head_\2"),
)


def hf_to_checkpoint_name(name: str) -> str:
    """An HF hy_v4 state-dict key as the checkpoint spells it (without any ``model.`` prefix change)."""
    for pattern, replacement in _HF_TO_CHECKPOINT:
        name = re.sub(pattern, replacement, name)
    return name


def layer_tensors(state_dict: dict, layer: int, prefix: str = "layers.") -> dict:
    """One layer's tensors out of an HF state dict, keyed the way reference.decoder.Hy4DecoderLayer.load
    and the device layer take them: by checkpoint name under ``model.layers.{L}.``."""
    head = f"{prefix}{layer}."
    return {hf_to_checkpoint_name(k)[len(head) :]: v for k, v in state_dict.items() if k.startswith(head)}


# ---------------------------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------------------------
def _first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def _unbatch(t: torch.Tensor) -> torch.Tensor:
    assert t.shape[0] == 1, f"the recorder takes one prompt at a time, got a batch of {t.shape[0]}"
    return t[0].detach().clone()


@contextmanager
def record_layer(layer, first_row: int = 0):
    """Record one HYV4DecoderLayer's intermediates with forward hooks while it runs.

    Yields the dict being filled; it is complete once the layer's forward has returned. Every
    entry is what an HF module received or returned, never something recomputed here.
    ``first_row`` keeps only the positions from that one on (a decode golden wants the last few
    rows of a long sequence: all 78 layers of a 5K-token pass would hold some 150 GB otherwise);
    ``dsa_dropped`` is then not computed.
    """
    rec, handles = {}, []

    def keep(t):
        t = _unbatch(t)
        return t[first_row:].clone() if first_row else t

    def output(module, key, pick=_first):
        handles.append(module.register_forward_hook(lambda m, args, out: rec.__setitem__(key, keep(pick(out)))))

    def first_input(module, key):
        handles.append(module.register_forward_pre_hook(lambda m, args: rec.__setitem__(key, keep(args[0]))))

    attn = layer.self_attn
    # HYV4HyperConnection returns (post, collapsed stream) and is called with the streams.
    first_input(layer.attn_hc, "streams_in")
    output(layer.attn_hc, "attn_post", lambda out: out[0])
    output(layer.attn_hc, "attn_hc", lambda out: out[1])
    output(layer.input_layernorm, "attn_in")
    output(attn, "attn_out")
    output(attn, "topk_indices", lambda out: out[2])
    first_input(layer.ffn_hc, "streams_mid")
    output(layer.ffn_hc, "mlp_post", lambda out: out[0])
    output(layer.ffn_hc, "mlp_hc", lambda out: out[1])
    output(layer.post_attention_layernorm, "mlp_in")
    output(layer.mlp, "mlp_out")
    output(layer, "streams_out")
    output(attn.q_a_layernorm, "q_resid")
    output(attn.kv_a_proj_with_mqa, "kv_a")
    output(attn.kv_a_layernorm, "k_pass")
    if hasattr(layer.mlp, "gate"):  # HYV4TopkRouter returns (logits, weights, indices), tokens flattened
        handles.append(
            layer.mlp.gate.register_forward_hook(
                lambda m, args, out: rec.update(
                    router_weights=out[1][first_row:].detach().clone(),
                    router_indices=out[2][first_row:].detach().clone(),
                )
            )
        )
    try:
        yield rec
    finally:
        for handle in handles:
            handle.remove()
        if "topk_indices" in rec and not first_row:
            rec["dsa_dropped"] = dsa_dropped(rec["topk_indices"])


def dsa_dropped(topk_indices: torch.Tensor) -> int:
    """How many (query, key) pairs with key <= query the DSA picks leave out. topk_indices: [S, k]
    for S queries over S keys. Zero means the layer's attention was dense causal attention."""
    S = topk_indices.shape[0]
    picked = torch.zeros(S, S, dtype=torch.bool).scatter(-1, topk_indices.long(), True)
    visible = torch.ones(S, S, dtype=torch.bool).tril()
    return int((visible & ~picked).sum())


def record_model(model, input_ids=None, inputs_embeds=None) -> dict:
    """Run an HF HYV4Model on one prompt in prefill and record every layer.

    Give ``input_ids`` [S], or ``inputs_embeds`` [S, hidden] to skip the embedding. Returns
    {"input_ids", "positions", "embedding", "layers"} as the golden file holds them.
    """
    assert (input_ids is None) != (inputs_embeds is None), "give input_ids or inputs_embeds, not both"
    captured = {}
    hook = model.embed_tokens.register_forward_hook(lambda m, args, out: captured.__setitem__("e", _unbatch(out)))
    try:
        with torch.no_grad(), ExitStack() as stack:
            layers = model.layers[: model.config.num_hidden_layers]
            records = [stack.enter_context(record_layer(layer)) for layer in layers]
            if input_ids is not None:
                model(input_ids=input_ids[None], use_cache=False)
            else:
                model(inputs_embeds=inputs_embeds[None], use_cache=False)
    finally:
        hook.remove()
    seq_len = (input_ids if input_ids is not None else inputs_embeds).shape[0]
    for index, rec in enumerate(records):
        missing = [key for key in STAGES if key not in rec]
        assert not missing, f"layer {index}: the hooks did not record {missing}"
    return {
        "input_ids": input_ids,
        "positions": torch.arange(seq_len),
        "embedding": captured.get("e", inputs_embeds),
        "layers": records,
    }


# ---------------------------------------------------------------------------------------------
# The real model
# ---------------------------------------------------------------------------------------------
def model_dir(path=None) -> Path:
    path = path or os.environ.get(HY4_HF_MODEL_ENV)
    assert path, f"give the checkpoint dir or set {HY4_HF_MODEL_ENV}"
    return Path(path)


def truncated_config(path=None, num_layers: int = DEFAULT_NUM_LAYERS):
    """The checkpoint's HYV4Config cut to its first ``num_layers`` layers, eager attention."""
    from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config

    config = HYV4Config.from_pretrained(str(model_dir(path)))
    assert 1 <= num_layers <= config.num_hidden_layers, num_layers
    config.num_hidden_layers = num_layers
    for per_layer in ("mlp_layer_types", "indexer_types", "layer_types"):
        setattr(config, per_layer, list(getattr(config, per_layer))[:num_layers])
    return config


def load_truncated_model(path=None, num_layers: int = DEFAULT_NUM_LAYERS, dtype=torch.bfloat16):
    """HYV4Model with the checkpoint's weights for layers 0 .. num_layers-1 (and the embedding).

    The tensors are memory-mapped, so this returns at once; the pages are read when a layer runs.
    The checkpoint's keys for the layers above are reported as unexpected and ignored.
    """
    from transformers.models.hy_v4.modeling_hy_v4 import HYV4Model

    config = truncated_config(path, num_layers)
    model = HYV4Model.from_pretrained(str(model_dir(path)), config=config, dtype=dtype).eval()
    assert model.config._attn_implementation == "eager", model.config._attn_implementation
    assert len(model.layers) == num_layers
    return model


def checkpoint_tensors(names, path=None, dtype=torch.float32) -> list:
    """Read tensors by checkpoint name from the shards, cast to ``dtype`` (None keeps the stored one)."""
    import json

    import safetensors

    root = model_dir(path)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    out = []
    for name in names:
        with safetensors.safe_open(str(root / weight_map[name]), framework="pt") as f:
            tensor = f.get_tensor(name)
        out.append(tensor if dtype is None else tensor.to(dtype))
    return out


def tokenize(prompt: str, path=None, seq_len: int | None = None) -> torch.Tensor:
    """The prompt as one user turn through the checkpoint's chat template -> token ids [S], cut to
    ``seq_len`` tokens when given (and required to be at least that long)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir(path)))
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )
    ids = torch.tensor(encoded if isinstance(encoded, list) else encoded["input_ids"], dtype=torch.long).flatten()
    if seq_len is not None:
        assert ids.numel() >= seq_len, f"the prompt is {ids.numel()} tokens, shorter than seq_len = {seq_len}"
        ids = ids[:seq_len]
    return ids


def golden_path(num_layers: int, input_ids: torch.Tensor, dtype=torch.bfloat16, root=None) -> Path:
    """Where the golden for these tokens lives: the name carries the depth, length, dtype and a hash of the ids."""
    digest = hashlib.sha256(input_ids.to(torch.int64).numpy().tobytes()).hexdigest()[:12]
    name = f"hy4_prefill_L{num_layers}_S{input_ids.numel()}_{str(dtype).removeprefix('torch.')}_{digest}.pt"
    return Path(root or GOLDEN_ROOT) / name


def record_golden(
    prompt: str = DEFAULT_PROMPT,
    num_layers: int = DEFAULT_NUM_LAYERS,
    seq_len: int | None = DEFAULT_SEQ_LEN,
    path=None,
    dtype=torch.bfloat16,
    root=None,
    recompute: bool = False,
) -> dict:
    """The golden record for ``prompt`` through layers 0 .. num_layers-1, from disk if it is there,
    otherwise computed with the HF model and saved. Needs transformers with ``hy_v4``."""
    import transformers

    input_ids = tokenize(prompt, path, seq_len)
    target = golden_path(num_layers, input_ids, dtype, root)
    if target.is_file() and not recompute:
        return torch.load(target, weights_only=False)

    model = load_truncated_model(path, num_layers, dtype)
    golden = record_model(model, input_ids=input_ids)
    golden["meta"] = {
        "checkpoint": str(model_dir(path)),
        "num_layers": num_layers,
        "dtype": str(dtype),
        "prompt": prompt,
        "seq_len": int(input_ids.numel()),
        "mlp_layer_types": list(model.config.mlp_layer_types),
        "indexer_types": list(model.config.indexer_types),
        "index_topk": int(model.config.index_topk),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    del model
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_suffix(".tmp")
    torch.save(golden, scratch)
    scratch.replace(target)
    return golden


def find_golden(num_layers: int = DEFAULT_NUM_LAYERS, seq_len: int = DEFAULT_SEQ_LEN, dtype=torch.bfloat16, root=None):
    """The newest cached golden of this depth, length and dtype, or None. Needs torch only: this is
    how a device test, which cannot run the HF model, gets its ground truth."""
    pattern = f"hy4_prefill_L{num_layers}_S{seq_len}_{str(dtype).removeprefix('torch.')}_*.pt"
    found = sorted(Path(root or GOLDEN_ROOT).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not found:
        return None
    golden = torch.load(found[-1], weights_only=False)
    golden["meta"]["file"] = str(found[-1])
    return golden


# ---------------------------------------------------------------------------------------------
# Prefill followed by decode: the golden of a prompt and its greedy continuation
# ---------------------------------------------------------------------------------------------
DECODE_STAGES = STAGES + ("q_resid", "topk_indices")


def decode_golden_path(
    num_layers: int, prompt_len: int, steps: int, input_ids, dtype=torch.bfloat16, root=None
) -> Path:
    digest = hashlib.sha256(input_ids[:prompt_len].to(torch.int64).numpy().tobytes()).hexdigest()[:12]
    name = f"hy4_decode_L{num_layers}_P{prompt_len}_T{steps}_{str(dtype).removeprefix('torch.')}_{digest}.pt"
    return Path(root or GOLDEN_ROOT) / name


def find_decode_golden(num_layers: int, prompt_len: int, steps: int | None = None, dtype=torch.bfloat16, root=None):
    """The newest cached decode golden of this depth and prompt length (and step count when given),
    or None. Needs torch only."""
    steps = "*" if steps is None else steps
    pattern = f"hy4_decode_L{num_layers}_P{prompt_len}_T{steps}_{str(dtype).removeprefix('torch.')}_*.pt"
    found = sorted(Path(root or GOLDEN_ROOT).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not found:
        return None
    golden = torch.load(found[-1], weights_only=False)
    golden["meta"]["file"] = str(found[-1])
    return golden


def record_decode_golden(
    prompt_ids: torch.Tensor,
    steps: int,
    num_layers: int = DEFAULT_NUM_LAYERS,
    path=None,
    dtype=torch.bfloat16,
    root=None,
    recompute: bool = False,
) -> dict:
    """What HF computes for a prompt of N tokens and ``steps`` decode steps after it, with the model
    cut to ``num_layers`` layers (the head, the final norm and the LM head applied to the cut stack).

    A decode engine that takes over after a prefill of the N prompt tokens runs step ``j`` for the
    token at position ``N - 1 + j``: step 0 re-runs the last prompt token (a prefill hands over
    caches, not that token's state) and predicts token N; step j feeds token ``N - 1 + j`` and
    predicts the next. The golden holds, for exactly those positions:

        input_ids    [N + steps] int64: the prompt, then HF's greedy continuation. Step j's input is
                     ``input_ids[N - 1 + j]`` and its prediction ``input_ids[N + j]``.
        cache        per layer {"latent" [S, kv_lora + rope], "index_key" [S, 128] (full layers)}, the
                     rows of HF's own DynamicCache after it generated the continuation: positions
                     [0, N) from its prefill pass, [N, S) from its cached decode steps; S = N + steps - 1.
                     HF frame: (k_pass | k_rope after RoPE, half-split), index key [pass | rope].
        layers       per layer, the DECODE_STAGES tensors of positions N - 1 .. S - 1 ([steps, ...]) from
                     ONE full-sequence pass over the S tokens (no cache), which is what a cached step
                     must reproduce; ``topk_indices`` are the DSA picks of those rows.
        hc_head, hidden, logits   [steps, hidden] x 2 and [steps, vocab] fp32: the stream-collapsing
                     head, the final norm, and lm_head (fp32, as HYV4ForCausalLM applies it).
        cached_logits_argmax      [steps] what HF's cached generation picked (= input_ids[N:]).
    """
    import time

    import transformers

    prompt_ids = prompt_ids.to(torch.int64).flatten()
    N = prompt_ids.numel()
    assert steps >= 1 and N >= 2, (steps, N)
    target = decode_golden_path(num_layers, N, steps, prompt_ids, dtype, root)
    if target.is_file() and not recompute:
        return torch.load(target, weights_only=False)

    t0 = time.perf_counter()
    model = load_truncated_model(path, num_layers, dtype)
    (head,) = checkpoint_tensors(["lm_head.weight"], path, torch.float32)

    def logits_of(hidden):
        return hidden.float() @ head.T

    # Pass 1: HF generates greedily through its own cache. This fixes the continuation and leaves
    # the cache HF itself would decode from.
    with torch.no_grad():
        out = model(input_ids=prompt_ids[None], use_cache=True)
        # transformers renamed the cache argument and output field ("past_key_values" -> "cache")
        name = "cache" if hasattr(out, "cache") else "past_key_values"
        cache = getattr(out, name)
        ids = prompt_ids.tolist()
        ids.append(int(logits_of(out.last_hidden_state[0, -1]).argmax()))
        for _ in range(steps - 1):
            out = model(input_ids=torch.tensor([[ids[-1]]]), use_cache=True, **{name: cache})
            ids.append(int(logits_of(out.last_hidden_state[0, -1]).argmax()))
    ids = torch.tensor(ids, dtype=torch.int64)
    S = N + steps - 1
    cached = []
    for index, layer in enumerate(cache.layers[:num_layers]):
        k_pass, k_rope = layer.keys, layer.values  # cache.update(k_pass, k_rot): [1, 1, S, width] each
        assert k_pass.shape[-2] == S and k_rope.shape[-2] == S, (index, tuple(k_pass.shape), tuple(k_rope.shape))
        entry = {"latent": torch.cat([k_pass[0, 0], k_rope[0, 0]], dim=-1).detach().clone()}
        if getattr(layer, "indexer_keys", None) is not None and layer.indexer_keys.numel():
            entry["index_key"] = layer.indexer_keys[0].detach().clone()
        cached.append(entry)
    t1 = time.perf_counter()

    # Pass 2: the S tokens in one pass without a cache, recorded.
    tail = {}
    hooks = [
        model.hc_head.register_forward_hook(
            lambda m, a, o: tail.__setitem__("hc_head", o[0, N - 1 :].detach().clone())
        ),
        model.norm.register_forward_hook(lambda m, a, o: tail.__setitem__("hidden", o[0, N - 1 :].detach().clone())),
    ]
    try:
        with torch.no_grad(), ExitStack() as stack:
            # Only the decode positions are kept as the pass runs (record_layer first_row).
            records = [stack.enter_context(record_layer(layer, N - 1)) for layer in model.layers[:num_layers]]
            model(input_ids=ids[None, :S], use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    layers = []
    for rec in records:
        row = {key: rec[key] for key in DECODE_STAGES}
        row["k_pass"] = rec["k_pass"]
        picks = rec["topk_indices"]  # rows N - 1 .. S - 1
        # per step: the keys the query may see that its picks leave out (0 while the context fits the top-k)
        row["dsa_dropped_rows"] = [
            p + 1 - int(torch.unique(picks[p - (N - 1)][picks[p - (N - 1)] <= p]).numel()) for p in range(N - 1, S)
        ]
        if "router_indices" in rec:
            row["router_indices"] = rec["router_indices"].reshape(steps, -1)
            row["router_weights"] = rec["router_weights"].reshape(steps, -1)
        layers.append(row)
    logits = logits_of(tail["hidden"])
    golden = {
        "meta": {
            "checkpoint": str(model_dir(path)),
            "num_layers": num_layers,
            "dtype": str(dtype),
            "prompt_len": N,
            "steps": steps,
            "positions": list(range(N - 1, S)),
            "mlp_layer_types": list(model.config.mlp_layer_types),
            "indexer_types": list(model.config.indexer_types),
            "index_topk": int(model.config.index_topk),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "seconds": {"generate": t1 - t0, "record": time.perf_counter() - t1},
        },
        "input_ids": ids,
        "cache": cached,
        "layers": layers,
        "hc_head": tail["hc_head"],
        "hidden": tail["hidden"],
        "logits": logits,
        "cached_logits_argmax": ids[N:].clone(),
    }
    del model, records, cache
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_suffix(".tmp")
    torch.save(golden, scratch)
    scratch.replace(target)
    return golden


def prompt_ids_from_text(text_path, num_tokens: int, path=None) -> torch.Tensor:
    """BOS + the tokens of a text file, cut to ``num_tokens`` (a long, plain prompt for prefill)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir(path)))
    config = truncated_config(path, 1)
    ids = tokenizer(Path(text_path).read_text(), add_special_tokens=False)["input_ids"]
    ids = torch.tensor([config.bos_token_id] + list(ids), dtype=torch.int64)
    assert ids.numel() >= num_tokens, f"{text_path} is {ids.numel()} tokens, shorter than {num_tokens}"
    return ids[:num_tokens]


def main():
    parser = argparse.ArgumentParser(description="Record Hy4-preview prefill goldens with the HF hy_v4 model.")
    parser.add_argument("--model", default=None, help=f"checkpoint dir (default ${HY4_HF_MODEL_ENV})")
    parser.add_argument("--layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--root", default=None, help=f"golden dir (default {GOLDEN_ROOT})")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--decode-steps", type=int, default=0, help="record a prefill + decode golden instead")
    parser.add_argument(
        "--prompt-file", default=None, help="text file the long prompt is cut from (with --decode-steps)"
    )
    args = parser.parse_args()
    if args.decode_steps:
        ids = prompt_ids_from_text(args.prompt_file, args.seq_len, args.model)
        golden = record_decode_golden(
            ids, args.decode_steps, args.layers, args.model, root=args.root, recompute=args.recompute
        )
        print(decode_golden_path(args.layers, args.seq_len, args.decode_steps, ids, root=args.root))
        print(f"seconds: {golden['meta']['seconds']}")
        print(f"continuation: {golden['input_ids'][args.seq_len:].tolist()}")
        print(f"full-pass argmax: {golden['logits'].argmax(-1).tolist()}")
        for index, rec in enumerate(golden["layers"]):
            print(f"layer {index}: visible keys the picks leave out, per step: {rec['dsa_dropped_rows']}")
        return
    golden = record_golden(args.prompt, args.layers, args.seq_len, args.model, root=args.root, recompute=args.recompute)
    print(golden_path(args.layers, golden["input_ids"], root=args.root))
    for index, rec in enumerate(golden["layers"]):
        grow = rec["streams_out"].float().norm() / rec["streams_in"].float().norm()
        print(f"layer {index}: dsa_dropped={rec['dsa_dropped']} |streams_out| / |streams_in| = {grow:.4f}")


if __name__ == "__main__":
    main()
