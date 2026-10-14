# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The layer-by-layer golden engine every generate_golden_<checkpoint>.py subclasses.

The model is built on the meta device from its config (transformers, or the repo's remote code pinned to a commit),
so nothing is allocated. Then each stage (embedding, layer i, final norm, lm_head) is materialized from just the
checkpoint shards holding it, run, recorded, and sent back to meta. All model code is the upstream implementation,
called as-is; a subclass only says what the upstream text model's forward would pass each layer, how checkpoint
tensors become HF parameters (dequantization), and which tensors are worth storing.

Trace layout (<root>/<hf-repo>/isl<N>/):
    hidden/embed.safetensors      model.embed_tokens.output                  [1, N, hidden]
    hidden/layer_XX.safetensors   model.layers.<i>.output                    [1, N, hidden]  decoder output
                                  + Recipe.sublayers outputs with --sublayers
    hidden/final.safetensors      model.last_hidden_state                    [1, N, hidden]  post final norm
                                  logits                                     [1, 1, vocab]   last position, fp32
    cache/layer_XX.safetensors    Recipe.modules / Recipe.routing captures, past_key_values.layers.<i>.<attr>
    metadata.json                 token-cache pointer, revision, versions, dtypes, every stored tensor

Tensor names are relative to the text backbone ("model." + its module path, whatever the wrapper calls it), so a
reader finds the producing module in the upstream code by name.

    python generate_golden_<checkpoint>.py --isl 10240                     # the trace
    python generate_golden_<checkpoint>.py --isl 1024 --num-layers 6 --root /tmp/smoke   # a diagnostic
    python generate_golden_<checkpoint>.py --selftest [--device cuda]      # layer-by-layer == one HF forward
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from common import (
    COMMON_NOTES,
    GOLDEN_SCHEMA,
    ROOT,
    TEXT,
    TMP,
    Inventory,
    Loader,
    Shards,
    check_isl,
    hub_kwargs,
    raise_cpu_time_limit,
    resolve_revision,
    share,
    store,
    token_cache,
)
from safetensors import safe_open
from safetensors.torch import load_file, save_file


class ChunkedAttention:
    """Run the model's own attention on ``chunk`` query rows at a time, so no [heads, N, N] score matrix exists whole.

    Each query row's softmax reads only that row's scores, so the result is the same function; memory drops from
    heads x N x N to heads x chunk x N. Nothing is renamed, because model code dispatches on both the name and the
    function: GLM-5.2 and MiniMax-M3 build their sparse masks only when the name is "eager" / "sdpa", and MiMo passes
    its sinks only when the function ``is`` its own eager_attention_forward. So transformers' "sdpa" is wrapped in the
    registry, and each model file's own eager_attention_forward is wrapped in that file's namespace (``patch``, once
    the model exists); both are put back on exit. chunk = 0 changes nothing.
    """

    def __init__(self, chunk: int):
        self.chunk, self.restore = chunk, []

    def __enter__(self):
        if self.chunk > 0:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            local = ALL_ATTENTION_FUNCTIONS._local_mapping
            self.restore.append(
                lambda had="sdpa" in local, old=local.get("sdpa"): local.update(sdpa=old) if had else local.pop("sdpa")
            )
            ALL_ATTENTION_FUNCTIONS["sdpa"] = self.wrap(ALL_ATTENTION_FUNCTIONS["sdpa"], additive_mask=False)
        return self

    def patch(self, model: torch.nn.Module) -> None:
        """Wrap eager_attention_forward in every model file this model's modules come from."""
        if self.chunk <= 0:
            return
        for name in {type(m).__module__ for m in model.modules()}:
            module = sys.modules[name]
            eager = getattr(module, "eager_attention_forward", None)
            if eager is not None and not getattr(eager, "chunked", False):
                module.eager_attention_forward = self.wrap(eager, additive_mask=True)
                self.restore.append(
                    lambda module=module, eager=eager: setattr(module, "eager_attention_forward", eager)
                )

    def __exit__(self, *exc):
        while self.restore:
            self.restore.pop()()

    def wrap(self, fn, additive_mask: bool):
        chunk = self.chunk

        def chunked(module, query, key, value, attention_mask, **kwargs):
            q_len, k_len = query.shape[-2], key.shape[-2]
            if q_len <= chunk:
                return fn(module, query, key, value, attention_mask, **kwargs)
            # sdpa gets no mask for a plain causal prefill (is_causal); a chunk of queries needs its offset rows.
            causal = attention_mask is None and kwargs.get("is_causal", getattr(module, "is_causal", True))
            outputs = []
            for s in range(0, q_len, chunk):
                e = min(s + chunk, q_len)
                mask = attention_mask
                if mask is not None and mask.shape[-2] != 1:
                    mask = mask[..., s:e, :]
                elif causal:
                    rows = torch.arange(s, e, device=query.device)[:, None] + (k_len - q_len)
                    mask = (torch.arange(k_len, device=query.device)[None, :] <= rows)[None, None]
                    if additive_mask:  # eager adds its mask to the scores
                        mask = torch.zeros(mask.shape, dtype=query.dtype, device=query.device).masked_fill(
                            ~mask, torch.finfo(query.dtype).min
                        )
                out, _ = fn(module, query[..., s:e, :], key, value, mask, **kwargs)
                outputs.append(out)  # [batch, rows, heads, head_dim], as every attention function returns it
            return torch.cat(outputs, dim=1), None  # no attention weights: they are what chunking avoids

        chunked.chunked = True
        return chunked


@dataclass
class Recipe:
    """What to store beyond the hidden states, per architecture.

    modules     submodule outputs hooked on every layer that has them, relative to the decoder layer -> cache/
    sublayers   submodule outputs hooked only with --sublayers (e.g. attention / MLP before post-norms) -> hidden/
    routing     the routed-experts module, whose (hidden, top_k_index, top_k_weights) inputs are the MoE decision
    cache       attributes of the HF cache layer that make up the state carried to the next token
    keep_fp32   tensor-name suffixes stored in fp32 whatever --store-dtype is (router logits, top-k weights, ...)
    config      text-config keys copied into metadata.json
    notes       what each stored tensor is ({i} = layer index), for metadata.json and the reader
    """

    modules: tuple[str, ...] = ()
    sublayers: tuple[str, ...] = ("self_attn", "mlp")
    routing: str | None = None
    cache: tuple[str, ...] = ("keys", "values")
    keep_fp32: tuple[str, ...] = ()
    config: tuple[str, ...] = ("layer_types",)
    notes: dict[str, str] = field(default_factory=dict)


@dataclass
class RunState:
    """What the upstream text model computes once before its layer loop, plus anything carried between layers."""

    position_ids: torch.Tensor
    cache: Any
    masks: dict[str, Any]  # layer type -> attention mask
    rope: dict[str, Any]  # layer type -> position embeddings
    carry: dict[str, Any] = field(default_factory=dict)  # extra kwargs the next layer receives
    input_ids: torch.Tensor | None = None  # set by the engine, for layers that read token ids (hash routing)
    extra: dict[str, Any] = field(default_factory=dict)


def recorded_config(config, keys: tuple[str, ...]) -> dict:
    """The keys the config has. Not hasattr: newer transformers configs can raise other errors for absent keys."""
    out = {}
    for k in keys:
        try:
            out[k] = getattr(config, k)
        except Exception:
            continue
    return out


class LayerwiseGolden:
    hf_repo: str = ""
    recipe: Recipe = Recipe()
    trust_remote_code: bool = False
    checkpoint_prefix: str | None = None  # backbone prefix in the checkpoint; default: its module path in the model
    lm_head_prefix: str | None = None  # lm_head prefix in the checkpoint; default: its module path in the model
    # Attention kernels the upstream model computes correctly. A model whose sinks (or other terms) only the eager
    # kernel applies sets both to "eager": transformers' sdpa path can drop them without an error.
    attn: str = "sdpa"
    attn_choices: tuple[str, ...] = ("sdpa", "eager")

    # ---- what a subclass overrides ------------------------------------------------------------------------------

    def build(self, config, dtype: torch.dtype) -> torch.nn.Module:
        """The full causal-LM model, called under ``torch.device("meta")``."""
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_config(config, dtype=dtype, trust_remote_code=self.trust_remote_code)

    def reduce_config(self, text_config, n: int) -> None:
        """--num-layers: cut the config to the first ``n`` layers, per-layer lists included."""
        old = text_config.num_hidden_layers
        text_config.num_hidden_layers = n
        for k, v in vars(text_config).items():
            if isinstance(v, list) and len(v) == old:
                setattr(text_config, k, v[:n])

    def convert(self, sd: dict[str, torch.Tensor], prefix: str, module: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Checkpoint tensors (keys relative to ``prefix``) -> the module's parameter names. Dequantize here."""
        return sd

    def layer_types(self, text_config) -> list[str]:
        n = text_config.num_hidden_layers
        return list(getattr(text_config, "layer_types", None) or ["full_attention"] * n)[:n]

    def make_cache(self, text_config):
        from transformers import DynamicCache

        return DynamicCache(config=text_config)

    def prepare(self, backbone, text_config, hidden: torch.Tensor, position_ids: torch.Tensor) -> RunState:
        """What the upstream text model computes before its layer loop: the cache, a mask and RoPE per layer type."""
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        cache = self.make_cache(text_config)
        kinds = set(self.layer_types(text_config))
        mask_kwargs = dict(
            config=text_config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=cache,
            position_ids=position_ids,
        )
        masks = {
            t: (create_sliding_window_causal_mask if t == "sliding_attention" else create_causal_mask)(**mask_kwargs)
            for t in kinds
        }
        rotary = backbone.rotary_emb
        if "layer_type" in inspect.signature(rotary.forward).parameters:
            rope = {t: rotary(hidden, position_ids, layer_type=t) for t in kinds}
        else:
            rope = dict.fromkeys(kinds, rotary(hidden, position_ids=position_ids))
        return RunState(position_ids=position_ids, cache=cache, masks=masks, rope=rope)

    def layer_kwargs(self, i: int, kind: str, state: RunState) -> dict:
        """Everything but the hidden states that the upstream text model passes layer ``i``."""
        return dict(
            attention_mask=state.masks[kind],
            position_ids=state.position_ids,
            past_key_values=state.cache,
            use_cache=True,
            position_embeddings=state.rope[kind],
            **state.carry,
        )

    def after_layer(self, i: int, output, state: RunState) -> torch.Tensor:
        """The hidden states out of layer ``i``; anything else it returns for the next layer goes in state.carry."""
        return output[0] if isinstance(output, tuple) else output

    def logits(self, text_config, lm_head: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        logits = lm_head(hidden[:, -1:])
        if softcap := getattr(text_config, "final_logit_softcapping", None):
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def tiny_init(self, model: torch.nn.Module) -> None:
        """Random weights for --selftest's tiny checkpoint. Override to randomize buffers the checkpoint stores too
        (router biases, ...), so the test checks they are loaded."""
        for p in model.parameters():
            p.normal_(0, 0.05)

    def tiny_config(self):
        """A small random-weight config with every layer kind of the real one, for --selftest."""
        raise NotImplementedError(f"{type(self).__name__} has no tiny_config(): --selftest unavailable")

    def compare(self, name: str, got: torch.Tensor, ref: torch.Tensor, exact: bool) -> str | None:
        """None if ``got`` (layer by layer) agrees with ``ref`` (one HF forward), else what is wrong."""
        if got.shape != ref.shape:
            return f"shape {list(got.shape)} vs {list(ref.shape)}"
        if not got.is_floating_point():
            agree = (got == ref.to(got.dtype)).float().mean().item()
            return None if agree == 1.0 or (not exact and agree >= 0.97) else f"agreement {agree:.4f}"
        diff = (got.float() - ref.to(got.dtype).float()).abs().max().item()
        scale = ref.float().abs().max().item() or 1.0
        return (
            None if (diff == 0 or (not exact and diff <= 2e-2 * scale)) else f"max|diff| {diff:.3g} (scale {scale:.3g})"
        )

    # ---- the engine ---------------------------------------------------------------------------------------------

    def parse_args(self, argv=None) -> argparse.Namespace:
        ap = argparse.ArgumentParser(
            description=sys.modules[type(self).__module__].__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        ap.add_argument("--hf-repo", default=self.hf_repo, help=f"default {self.hf_repo}")
        ap.add_argument("--revision", default="main", help="branch, tag or commit; pinned to a commit at start")
        ap.add_argument("--isl", type=int, default=None, help="prefill the first ISL tokens of the token cache")
        ap.add_argument("--root", type=Path, default=ROOT, help="trace store (default $HF_GOLDEN_ROOT or %(default)s)")
        ap.add_argument("--text", type=Path, default=None, help=f"default: <root>/{TEXT}")
        ap.add_argument("--num-layers", type=int, default=None, help="REDUCED depth: a diagnostic, never a grade")
        ap.add_argument("--sublayers", action="store_true", help="also store the recipe's sub-layer outputs")
        ap.add_argument("--attn", default=self.attn, choices=self.attn_choices)
        ap.add_argument(
            "--attn-chunk", type=int, default=0, help="attention on this many query rows at a time (0: all at once)"
        )
        ap.add_argument("--device", default="auto", help="auto (cuda if visible, else cpu), cpu, cuda, cuda:N")
        ap.add_argument("--compute-dtype", default="bfloat16", choices=("bfloat16", "float32"))
        ap.add_argument("--store-dtype", default="bfloat16", choices=("bfloat16", "float32"))
        ap.add_argument("--threads", type=int, default=None)
        ap.add_argument("--tmp", type=Path, default=TMP, help="where shards not in an HF cache are downloaded")
        ap.add_argument("--selftest", action="store_true", help="tiny random checkpoint: layerwise == one HF forward")
        args = ap.parse_args(argv)
        if not args.selftest and args.isl is None:
            ap.error("--isl is required")
        return args

    def main(self, argv=None) -> int:
        args = self.parse_args(argv)
        raise_cpu_time_limit()
        if args.threads:
            torch.set_num_threads(args.threads)
        if args.selftest:
            return 0 if all(self.selftest(args, c) for c in ("bfloat16", "float32")) else 1
        with ChunkedAttention(args.attn_chunk) as self.attn_chunker:
            self.generate(args)
        return 0

    @torch.no_grad()  # else every layer output holds a graph back to all weights so far, and release frees nothing
    def generate(self, args: argparse.Namespace) -> Path:
        import transformers
        from transformers import AutoConfig

        device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        compute, store_dtype = getattr(torch, args.compute_dtype), getattr(torch, args.store_dtype)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            # TF32 silently drops fp32 matmuls to a 10-bit mantissa; a --compute-dtype float32 run means fp32.
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        print(f"[device] {device}, compute {args.compute_dtype}", flush=True)
        check_isl(args.isl, strict=not (args.num_layers or args.selftest))  # a diagnostic need not be gradable

        revision = resolve_revision(args.hf_repo, args.revision)
        print(f"[weights] {args.hf_repo} @ {revision}", flush=True)
        pinned = None if revision == "local" else revision
        hub = hub_kwargs(args.hf_repo, pinned)  # config, tokenizer, remote code: from the copy on disk if there is one
        tok_path = token_cache(args.hf_repo, args.root, args.text or args.root / TEXT, **hub)
        token_ids = load_file(str(tok_path))["token_ids"]
        assert args.isl <= token_ids.numel(), f"isl {args.isl} > {token_ids.numel()} tokens in {tok_path}"
        input_ids = token_ids[: args.isl].long().unsqueeze(0).to(device)
        position_ids = torch.arange(args.isl, device=device).unsqueeze(0)
        with safe_open(str(tok_path), "pt") as f:
            tok_meta = f.metadata()

        config = AutoConfig.from_pretrained(args.hf_repo, trust_remote_code=self.trust_remote_code, **hub)
        text_config = config.get_text_config()
        for c in {id(config): config, id(text_config): text_config}.values():
            c._attn_implementation = args.attn
        if args.num_layers:
            self.reduce_config(text_config, args.num_layers)
            print(f"[reduced] depth {args.num_layers}: this trace is a DIAGNOSTIC, not a grade", flush=True)
        if args.isl > (context := getattr(text_config, "max_position_embeddings", args.isl)):
            # Past the trained context the RoPE positions are out of distribution: the result is a golden of nothing.
            raise SystemExit(f"isl {args.isl} exceeds the model's context window of {context}")
        n_layers = text_config.num_hidden_layers
        kinds = self.layer_types(text_config)
        with torch.device("meta"):
            model = self.build(config, compute)
        model.eval()
        if (chunker := getattr(self, "attn_chunker", None)) is not None:
            chunker.patch(model)
        backbone = model.get_decoder()
        stages = self.stages(model, backbone, text_config)
        materialize = Loader(model, compute, device)

        out = args.root / args.hf_repo / f"isl{args.isl}"
        if (out / "metadata.json").exists():
            raise SystemExit(f"{out} exists; delete it to regenerate")
        (out / "hidden").mkdir(parents=True, exist_ok=True)
        (out / "cache").mkdir(parents=True, exist_ok=True)
        inventory = Inventory({**COMMON_NOTES, **self.recipe.notes})

        def save(tensors: dict[str, torch.Tensor], file: str, layer: int | None = None) -> None:
            tensors = {
                k: store(v, torch.float32 if k.endswith(self.recipe.keep_fp32) or k == "logits" else store_dtype)
                for k, v in tensors.items()
            }
            save_file(tensors, str(out / (file.format(i=layer) if layer is not None else file)))
            inventory.add(tensors, file, layer, kinds[layer] if layer is not None else None)

        t0 = time.time()
        layer_seconds = []
        args.tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.tmp, prefix="hf_golden_") as tmp:
            shards = Shards(args.hf_repo, pinned, Path(tmp), [prefix for _, _, prefix in stages])

            def load(s: int) -> torch.nn.Module:
                _, module, prefix = stages[s]
                materialize(module, self.convert(shards.load(s, prefix), prefix, module), prefix)
                return module

            for module in self.stateless_modules(backbone, stages):  # RoPE etc.: buffers only, nothing to load
                materialize(module, {}, "")

            embed = load(0)
            hidden = embed(input_ids)
            Loader.release(embed)
            save({"model.embed_tokens.output": hidden}, "hidden/embed.safetensors")
            state = self.prepare(backbone, text_config, hidden, position_ids)
            state.input_ids = input_ids

            for i in range(n_layers):
                ts = time.time()
                layer = load(1 + i)
                captured: dict[str, torch.Tensor] = {}
                handles = self.register_hooks(i, layer, captured, args.sublayers)
                output = layer(hidden, **self.layer_kwargs(i, kinds[i], state))
                hidden = self.after_layer(i, output, state)
                for h in handles:
                    h.remove()

                cache_state = {k: v for k, v in captured.items() if not self.is_sublayer(k)}
                cache_state.update(self.cache_tensors(i, state.cache))
                save(cache_state, "cache/layer_{i:02d}.safetensors", i)
                sub = {k: v for k, v in captured.items() if self.is_sublayer(k)}
                save({f"model.layers.{i}.output": hidden, **sub}, "hidden/layer_{i:02d}.safetensors", i)

                # Nothing reads this layer again: drop its weights and its cache entry so memory stays one layer deep.
                Loader.release(layer)
                self.free_cache(i, state.cache)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                layer_seconds.append(round(time.time() - ts, 1))
                print(
                    f"[layer {i + 1}/{n_layers}] {kinds[i]} {layer_seconds[-1]:.0f}s  stored {sorted(cache_state)}",
                    flush=True,
                )

            norm = load(n_layers + 1)
            hidden = norm(hidden)
            Loader.release(norm)
            head = load(n_layers + 2)
            logits = self.logits(text_config, head, hidden)
            Loader.release(head)
            save({"model.last_hidden_state": hidden, "logits": logits}, "hidden/final.safetensors")

        elapsed = time.time() - t0
        metadata = {
            "golden_schema": GOLDEN_SCHEMA,
            "token_cache": f"../{tok_path.name}",
            "n_tokens": args.isl,
            "source": tok_meta["source"],
            "source_sha256": tok_meta["source_sha256"],
            "hf_repo": args.hf_repo,
            "revision": revision,
            "trust_remote_code": self.trust_remote_code,
            "model_type": text_config.model_type,
            "num_layers": n_layers,
            "reduced": bool(args.num_layers),
            "layer_kinds": kinds,
            "hidden_size": text_config.hidden_size,
            "config": recorded_config(text_config, self.recipe.config),
            "compute_dtype": args.compute_dtype,
            "store_dtype": args.store_dtype,
            "fp32_tensors": ["logits", *self.recipe.keep_fp32],
            "attn_implementation": args.attn,
            "attn_chunk": args.attn_chunk or None,
            "experts_implementation": getattr(text_config, "_experts_implementation", None),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "threads": torch.get_num_threads() if device.type == "cpu" else None,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "tensors": inventory.finish(),
            "layer_seconds": layer_seconds,
            "seconds": round(elapsed, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        share(out.parent, args.root)  # the token cache and the trace
        print(f"[done] {out}  ({elapsed / 60:.1f} min)", flush=True)
        return out

    # ---- pieces of the engine a subclass may need to adjust ----------------------------------------------------

    def stages(self, model, backbone, text_config) -> list[tuple[str, torch.nn.Module, str]]:
        """(name, module, checkpoint prefix) in run order: embedding, layers, final norm, lm_head."""
        path = {m: n for n, m in model.named_modules()}
        prefix = self.checkpoint_prefix if self.checkpoint_prefix is not None else path[backbone] + "."
        sub = {m: n for n, m in backbone.named_modules()}
        embed = backbone.get_input_embeddings()
        head = model.get_output_embeddings()
        stages = [("embed", embed, prefix + sub[embed] + ".")]
        stages += [
            (f"layer {i}", backbone.layers[i], f"{prefix}layers.{i}.") for i in range(text_config.num_hidden_layers)
        ]
        stages += [("norm", backbone.norm, prefix + sub[backbone.norm] + ".")]
        tied = getattr(text_config, "tie_word_embeddings", False) or getattr(model.config, "tie_word_embeddings", False)
        stages += [("lm_head", head, stages[0][2] if tied else self.lm_head_prefix or path[head] + ".")]
        return stages

    def stateless_modules(self, backbone, stages) -> list[torch.nn.Module]:
        """Direct children of the backbone with buffers but no parameters (RoPE), which no stage covers."""
        covered = {m for _, m, _ in stages} | {backbone.layers}
        return [
            m
            for m in backbone.children()
            if m not in covered and not any(True for _ in m.parameters()) and any(True for _ in m.buffers())
        ]

    def register_hooks(self, i: int, layer: torch.nn.Module, captured: dict, sublayers: bool) -> list:
        """Forward hooks storing the recipe's captures of layer ``i`` into ``captured`` under their HF names."""
        handles = []
        for name, mod in layer.named_modules():
            key = f"model.layers.{i}.{name}"
            if name in self.recipe.modules or (sublayers and name in self.recipe.sublayers):

                def put(_m, _a, o, key=f"{key}.output"):
                    captured[key] = o[0] if isinstance(o, tuple) else o

                handles.append(mod.register_forward_hook(put))
            if name == self.recipe.routing:

                def routing(_m, inputs, _o, key=key):
                    # (hidden_states, top_k_index, top_k_weights); HF's top-k is unsorted, so sort by expert id.
                    index, order = inputs[1].sort(dim=-1)
                    captured[f"{key}.top_k_index"] = index.to(torch.int32)
                    captured[f"{key}.top_k_weights"] = inputs[2].gather(-1, order)

                handles.append(mod.register_forward_hook(routing))
        return handles

    def is_sublayer(self, key: str) -> bool:
        return any(key.endswith(f".{s}.output") for s in self.recipe.sublayers) and not any(
            key.endswith(f".{m}.output") for m in self.recipe.modules
        )

    def cache_tensors(self, i: int, cache) -> dict[str, torch.Tensor]:
        if cache is None or i >= len(cache.layers):
            return {}
        layer = cache.layers[i]
        return {
            f"past_key_values.layers.{i}.{a}": getattr(layer, a)
            for a in self.recipe.cache
            if getattr(layer, a, None) is not None
        }

    def free_cache(self, i: int, cache) -> None:
        if cache is not None and i < len(cache.layers):
            for a in self.recipe.cache:
                if getattr(cache.layers[i], a, None) is not None:
                    setattr(cache.layers[i], a, None)

    # ---- --selftest ----------------------------------------------------------------------------------------------

    def selftest(self, args: argparse.Namespace, compute: str, isl: int = 64) -> bool:
        """A tiny random checkpoint of this architecture, streamed layer by layer, must give what one upstream
        forward of the whole model gives on the same device and dtype: exactly on CPU, to a tolerance on GPU (kernels
        need not be bit-reproducible between the two module arrangements)."""
        from transformers import AutoModelForCausalLM

        device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        with tempfile.TemporaryDirectory(prefix="hf_golden_selftest_") as tmp:
            ckpt = Path(tmp) / "ckpt"
            config = self.tiny_config()
            torch.manual_seed(0)
            tiny = AutoModelForCausalLM.from_config(config, trust_remote_code=self.trust_remote_code).to(torch.bfloat16)
            with torch.no_grad():
                self.tiny_init(tiny)
            tiny.save_pretrained(str(ckpt), max_shard_size="300KB")  # several shards, like the real checkpoint
            vocab = config.get_text_config().vocab_size
            ids = torch.randint(0, vocab, (4 * isl,), generator=torch.Generator().manual_seed(1), dtype=torch.int32)
            meta = {"source": "selftest", "source_sha256": "0" * 64}
            save_file({"token_ids": ids}, str(ckpt / f"{Path(TEXT).stem}.tokens.safetensors"), metadata=meta)

            run = argparse.Namespace(**{**vars(args), "hf_repo": str(ckpt), "isl": isl, "root": Path(tmp)})
            run.tmp, run.num_layers, run.sublayers, run.compute_dtype = Path(tmp), None, True, compute
            # Streamed with chunking, compared with an unchunked whole-model forward: this tests chunking too.
            with ChunkedAttention(args.attn_chunk) as self.attn_chunker:
                trace = self.generate(run)
            got = {}
            for f in trace.rglob("*.safetensors"):
                got.update(load_file(str(f)))
            ref = self.reference(ckpt, ids[:isl], device, getattr(torch, compute), args.attn)

        errors = [f"tensor names differ: {sorted(set(got) ^ set(ref))}"] if set(got) != set(ref) else []
        for k in sorted(set(got) & set(ref)):
            if (e := self.compare(k, got[k], ref[k], exact=device.type == "cpu")) is not None:
                errors.append(f"{k}: {e}")
        print(f"[selftest] {'PASS' if not errors else 'FAIL'} {device} {compute}: {len(got)} tensors", flush=True)
        for e in errors:
            print(f"    {e}", flush=True)
        return not errors

    def reference(self, ckpt: Path, ids: torch.Tensor, device: torch.device, dtype: torch.dtype, attn: str) -> dict:
        """One forward of the whole upstream model from from_pretrained, captured under the same names."""
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(str(ckpt), trust_remote_code=self.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), config=config, dtype=dtype, attn_implementation=attn, trust_remote_code=self.trust_remote_code
        )
        model = model.to(device).eval()
        backbone = model.get_decoder()
        text_config = config.get_text_config()
        captured: dict[str, torch.Tensor] = {}

        def put(key):
            return lambda _m, _a, o: captured.__setitem__(key, o[0] if isinstance(o, tuple) else o)

        backbone.get_input_embeddings().register_forward_hook(put("model.embed_tokens.output"))
        backbone.norm.register_forward_hook(put("model.last_hidden_state"))
        for i, layer in enumerate(backbone.layers):
            layer.register_forward_hook(put(f"model.layers.{i}.output"))
            self.register_hooks(i, layer, captured, sublayers=True)
        cache = self.make_cache(text_config)
        with torch.no_grad():
            out = model(input_ids=ids[None].long().to(device), past_key_values=cache, use_cache=True, logits_to_keep=1)
        captured["logits"] = out.logits
        for i in range(text_config.num_hidden_layers):
            captured.update(self.cache_tensors(i, cache))
        return {k: v.cpu() for k, v in captured.items()}
