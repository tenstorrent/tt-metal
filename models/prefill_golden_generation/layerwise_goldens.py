# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Golden trace generation for models too large to hold in memory — one decoder layer at a time.

``whole_model_goldens.py`` loads the whole model and runs one forward. That stops working when the
weights do not fit: GLM-5.2 is ~742 B parameters, which is ~3.0 TB at fp32 against 566 GB of host
RAM. Nothing about the arithmetic is hard, only the footprint.

So this walks the model instead. Load layer 0's weights, run the sequence through it, write what
that layer carries, free it, load layer 1, feed it layer 0's output, and so on. Peak memory is one
layer plus the activations rather than the whole checkpoint, and the shards for a layer are fetched
and dropped as it goes.

Built in stages. **Stage 1 (here)** reads the checkpoint index and works out, per layer, which
tensors it owns and which shards they live in — the plan the later stages execute.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from safetensors.torch import save_file

if TYPE_CHECKING:
    from transformers import PretrainedConfig

# --------------------------------------------------------------------------------------------
# where things live
# --------------------------------------------------------------------------------------------

DEFAULT_STORE: Path = Path("/mnt/models/agentic-prefill-goldens")

# Names the weights to fetch AND the store subdirectory.
DEFAULT_HF_REPO: str = "zai-org/GLM-5.2"

# The text staged in the store, tokenized once per model by whole_model_goldens.py.
DEFAULT_TEXT: str = "wikitext-2-raw-v1-train.txt"

DEFAULT_ISL: int = 5120

# --------------------------------------------------------------------------------------------
# checkpoint layout
# --------------------------------------------------------------------------------------------

WEIGHT_INDEX_NAME: str = "model.safetensors.index.json"
CONFIG_NAME: str = "config.json"

# Weight-name prefixes. A decoder layer owns everything under its own prefix; the rest is global.
LAYER_PREFIX: str = "model.layers.{index}."
EMBED_PREFIX: str = "model.embed_tokens."
NORM_PREFIX: str = "model.norm."

# lm_head maps the final hidden state to logits. A golden stops at the hidden state, so this is
# listed only to be skipped — it is 1.9 GB that nothing downstream reads.
LM_HEAD_PREFIX: str = "lm_head."

# --------------------------------------------------------------------------------------------
# canonical tensor keys
# --------------------------------------------------------------------------------------------
#
# Every key is a dotted path, the way a PyTorch state_dict is keyed, so a reader can grep
# transformers for the name and find the code that produced it. Three rules:
#
#   1. taken verbatim from the HF cache  ->  past_key_values.layers.<i>.<attr>
#   2. a module's forward output         ->  <module path>.output
#   3. the model's own output            ->  model.last_hidden_state, the name transformers
#                                            itself uses on BaseModelOutputWithPast
#
# The layer index stays in the key even though the filename carries it, so several files can be
# loaded and merged into one dict without collision — exactly like a sharded state_dict.

KEY_EMBED: str = "model.embed_tokens.output"
KEY_LAYER_OUTPUT: str = "model.layers.{index}.output"
KEY_KVPE: str = "model.layers.{index}.self_attn.kv_a_proj_with_mqa.output"
KEY_INDEXER_KEYS: str = "past_key_values.layers.{index}.indexer_keys"
KEY_INDEXER_TOPK: str = "model.layers.{index}.self_attn.indexer.output"
KEY_FINAL_HIDDEN: str = "model.last_hidden_state"

# --------------------------------------------------------------------------------------------
# trace layout
# --------------------------------------------------------------------------------------------
#
# hidden/ is the activation trail and cache/ is per-layer state. Split because the trail is most
# of the bytes — 51 of 65 GiB for GLM-5.2 at ISL 56320 — so anyone who only wants the graded cache
# does not have to pull the activations.

HIDDEN_DIR_NAME: str = "hidden"
CACHE_DIR_NAME: str = "cache"
METADATA_NAME: str = "metadata.json"
EMBED_FILE_NAME: str = "embed.safetensors"
FINAL_FILE_NAME: str = "final.safetensors"

# Zero-padded to the model's depth so lexical order is layer order; plain layer_10 sorts before
# layer_2, which makes a directory listing lie about the forward order.
LAYER_FILE_TEMPLATE: str = "layer_{index:0{width}d}.safetensors"

# Bumped when the layout or key names change, so a reader can fail loudly on a format it does not
# know rather than silently grading the wrong tensor.
GOLDEN_SCHEMA: str = "1"

# --------------------------------------------------------------------------------------------
# numerics
# --------------------------------------------------------------------------------------------

DTYPES: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float32": torch.float32}

# Compute fp32, store bf16 — see whole_model_goldens.py for the reasoning. The short version: the
# device computes in bf16, so a bf16 reference shares its rounding and inflates measured PCC.
DEFAULT_COMPUTE_DTYPE: str = "float32"
DEFAULT_STORE_DTYPE: str = "bfloat16"

# ISL has to suit the device even though nothing here talks to one.
ISL_ALIGNMENT: int = 32 * 8  # ttnn.TILE_SIZE * sp
ISL_CHUNK_SIZE: int = 5120  # spec chunk_size; a chunked grade additionally needs this


# --------------------------------------------------------------------------------------------
# stage 1 — read the checkpoint index and plan the walk
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerPlan:
    """What one decoder layer owns, and where it lives on disk.

    ``shards`` is what has to be present to build this layer. They are not contiguous and not in
    layer order — in GLM-5.2 layer 2 sits in shard 38 while layer 3 spans 75-79 — so the walk
    fetches by this plan rather than by stepping through shard numbers.
    """

    index: int
    kind: str
    tensors: tuple[str, ...]
    shards: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointPlan:
    """The whole walk: the global tensors, then one plan per decoder layer."""

    source: str
    num_layers: int
    globals_: tuple[str, ...]
    global_shards: tuple[str, ...]
    layers: tuple[LayerPlan, ...]

    @property
    def shard_count(self) -> int:
        """Distinct shards the full walk touches — every shard holding a weight we actually load."""
        return len({shard for layer in self.layers for shard in layer.shards} | set(self.global_shards))


def fetch_file(source: str, name: str) -> Path:
    """One file from a local checkpoint directory, or from the Hub when ``source`` is a repo id.

    Call inside ``temporary_hf_cache()`` for a repo id: ``env.sh`` points HF_HOME at the shared
    /mnt/models cache, whose ``refs/main`` is read-only and owned by another user, so a resolve
    against it dies with PermissionError.
    """
    local: Path = Path(source) / name
    if local.is_file():
        return local

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(source, name))


def load_weight_map(source: str) -> dict[str, str]:
    """``tensor name -> shard filename`` for the whole checkpoint.

    The index is a few MB against the checkpoint's ~1.5 TB, so this is the one file worth reading
    up front: it is what makes fetching a single layer possible at all.
    """
    index: dict = json.loads(fetch_file(source, WEIGHT_INDEX_NAME).read_text())
    return index["weight_map"]


def layer_kind(config: PretrainedConfig, index: int) -> str:
    """A short label for what this layer is, from the per-layer lists in the config.

    GLM-5.2 has two of them, both length ``num_hidden_layers``: ``mlp_layer_types`` (3 dense then
    75 sparse) and ``indexer_types`` (21 full, 57 shared). There is no ``layer_types`` — that one
    is computed by the model class, not stored. A model with neither list reports "decoder".
    """
    parts: list[str] = []
    mlp: list[str] | None = getattr(config, "mlp_layer_types", None)
    indexer: list[str] | None = getattr(config, "indexer_types", None)
    if mlp is not None:
        parts.append(f"{mlp[index]} MLP")
    if indexer is not None:
        parts.append(f"{indexer[index]} indexer")
    return " / ".join(parts) if parts else "decoder"


def plan_checkpoint(source: str, config: PretrainedConfig) -> CheckpointPlan:
    """Group every weight in the index by the layer that owns it.

    ``lm_head`` is dropped here rather than later: a trace stops at the final hidden state, so its
    1.9 GB is never needed. It shares a shard with the embedding, so this saves memory, not
    download.
    """
    weight_map: dict[str, str] = load_weight_map(source)
    num_layers: int = config.num_hidden_layers

    def pick(*prefixes: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        names: tuple[str, ...] = tuple(sorted(n for n in weight_map if n.startswith(prefixes)))
        return names, tuple(sorted({weight_map[n] for n in names}))

    global_names, global_shards = pick(EMBED_PREFIX, NORM_PREFIX)
    layers: list[LayerPlan] = []
    for index in range(num_layers):
        names, shards = pick(LAYER_PREFIX.format(index=index))
        layers.append(LayerPlan(index=index, kind=layer_kind(config, index), tensors=names, shards=shards))

    return CheckpointPlan(
        source=source,
        num_layers=num_layers,
        globals_=global_names,
        global_shards=global_shards,
        layers=tuple(layers),
    )


def summarize_plan(plan: CheckpointPlan, *, show: int = 8) -> None:
    """The plan as a table — the first ``show`` layers, then one line per distinct layer kind."""
    print(f"[plan] {plan.source}: {plan.num_layers} layers, {plan.shard_count} shards to walk")
    print(f"[plan] globals: {len(plan.globals_)} tensors in {len(plan.global_shards)} shards")
    for name in plan.globals_:
        print(f"[plan]     {name}")

    print(f"[plan] {'layer':>5s}  {'tensors':>7s}  {'shards':>6s}  kind")
    for layer in plan.layers[:show]:
        span: str = f"{layer.shards[0]}..{layer.shards[-1]}" if len(layer.shards) > 1 else layer.shards[0]
        print(f"[plan] {layer.index:5d}  {len(layer.tensors):7d}  {len(layer.shards):6d}  {layer.kind}   {span}")
    if plan.num_layers > show:
        print(f"[plan] ... {plan.num_layers - show} more")

    kinds: dict[str, int] = {}
    for layer in plan.layers:
        kinds[layer.kind] = kinds.get(layer.kind, 0) + 1
    for kind, count in sorted(kinds.items()):
        print(f"[plan] {count:3d} layers: {kind}")


# --------------------------------------------------------------------------------------------
# stage 2 — walk the layers
# --------------------------------------------------------------------------------------------


@dataclass
class LayerTiming:
    """Where one layer's wall time went, so a long walk can be read after the fact."""

    index: int
    kind: str
    fetch_s: float
    load_s: float
    compute_s: float
    write_s: float
    fetched_bytes: int

    @property
    def total_s(self) -> float:
        return self.fetch_s + self.load_s + self.compute_s + self.write_s


def fetch_shards(source: str, shards: tuple[str, ...], dest: Path) -> tuple[list[Path], int]:
    """Put this slice's shards in ``dest``, so it is a directory ``from_pretrained`` can load.

    ``local_dir`` rather than the Hub cache, because these are deleted as soon as the slice is
    loaded: a cached download leaves the blob behind and the walk would fill the disk by layer ten.
    A local checkpoint is symlinked instead — same directory layout, nothing copied, and unlinking
    the symlink later cannot touch the original.
    """
    local_source: Path = Path(source)
    paths: list[Path] = []

    if local_source.is_dir():
        for name in shards:
            link: Path = dest / name
            if not link.exists():
                link.symlink_to((local_source / name).resolve())
            paths.append(link)
        return paths, 0

    from huggingface_hub import hf_hub_download

    for name in shards:
        paths.append(Path(hf_hub_download(source, name, local_dir=str(dest))))
    return paths, sum(p.stat().st_size for p in paths)


def drop_shards(paths: list[Path], workdir: Path) -> None:
    """Delete the fetched shards. Option 1 of the fetch policy: free after every layer.

    Costs 1.28x in re-download on GLM-5.2 — 279 distinct shards against 357 fetches, because 75 of
    them serve more than one layer and one pair sits 63 layers apart. That is the cheap trade: the
    walk is compute-bound at any useful ISL, and holding shards for a span that long is what fills
    a disk. Only files under ``workdir`` are touched, so a local checkpoint is never harmed.
    """
    for path in paths:
        if workdir in path.parents:
            path.unlink(missing_ok=True)


def write_micro_checkpoint(
    dest: Path,
    raw_config: dict,
    weight_map: dict[str, str],
    names: tuple[str, ...],
    *,
    layer_index: int | None,
) -> Path:
    """A real, self-contained checkpoint holding one slice of a much larger one.

    This exists so ``from_pretrained`` can do the loading. Transformers owns a great deal of
    machinery we would otherwise have to reimplement and keep in step: the expert fusion that turns
    768 per-expert matrices into the two stacked tensors ``GlmMoeDsaNaiveMoe`` declares, the dtype
    policy, the modules pinned to fp32 because they decide a top-k or an expert, and any
    quantizer. Rather than drive those internals, hand it a checkpoint it understands.

    Three files: ``config.json`` cut to this slice's depth, a ``model.safetensors.index.json``
    listing only the tensors we want, and the shard files themselves (already in ``dest``). The
    tensor names stay as they are in the shards — the rename to layer 0 is done by
    ``from_pretrained``'s own ``key_mapping``, not by rewriting anything.

    ``layer_index=None`` builds the 0-layer shell that carries the embedding and the final norm.
    """
    depth: int = raw_config["num_hidden_layers"]
    config: dict = dict(raw_config)
    config["num_hidden_layers"] = 0 if layer_index is None else 1
    config["num_nextn_predict_layers"] = 0

    # Any per-layer list is cut to this slice. Generic rather than naming GLM's indexer_types and
    # mlp_layer_types, so a different model with its own per-layer lists needs no change here.
    for key, value in raw_config.items():
        if isinstance(value, list) and len(value) == depth:
            config[key] = [] if layer_index is None else [value[layer_index]]

    # The shell owns the real embedding; a layer slice never uses one, so shrink it to nothing
    # rather than pay for a 154880 x 6144 random tensor 78 times. pad_token_id has to come along
    # or nn.Embedding asserts its padding_idx is out of range.
    if layer_index is not None:
        config["vocab_size"], config["pad_token_id"] = 16, 0

    keep: dict[str, str] = {name: weight_map[name] for name in names}
    total: int = 0
    for shard in sorted(set(keep.values())):
        with (dest / shard).open("rb") as fh:
            header: dict = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        total += sum(t["data_offsets"][1] - t["data_offsets"][0] for k, t in header.items() if k in keep)

    (dest / CONFIG_NAME).write_text(json.dumps(config, indent=2))
    (dest / WEIGHT_INDEX_NAME).write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": keep}, indent=2))
    return dest


PARK_PREFIX: str = "__not_this_slice__"


def rename_rules(dest: Path, layer_index: int) -> dict[str, str]:
    """Map this layer onto layer 0, and park every other layer sharing its shards.

    The rename alone is not enough. transformers picks FILES from the index and then loads every
    tensor it finds in them, and shards hold several layers — layers 0 and 1 of GLM-5.2 share shard
    1. So ``model.layers.1.* -> model.layers.0.*`` collides with the shard's own
    ``model.layers.0.*``, two sources land on one target, and the wrong weights win. Measured: with
    only the rename, layer 0 is exact and layer 1 comes out at PCC -0.096, which then propagates
    down the rest of the walk.

    So the neighbours are renamed out of the way first, into a prefix no module answers to. They
    are reported unexpected and skipped, which is what we want — they belong to other layers.
    """
    pattern = re.compile(r"^model\.layers\.(\d+)\.")
    shards: set[str] = {path.name for path in dest.glob("*.safetensors")}
    neighbours: set[int] = set()
    for shard in shards:
        with safe_open(str(dest / shard), framework="pt") as fh:
            for key in fh.keys():
                found = pattern.match(key)
                if found and int(found.group(1)) != layer_index:
                    neighbours.add(int(found.group(1)))

    rules: dict[str, str] = {rf"^model\.layers\.{other}\.": f"{PARK_PREFIX}.layers.{other}." for other in neighbours}
    rules[rf"^model\.layers\.{layer_index}\."] = "model.layers.0."
    return rules


def load_slice(dest: Path, *, layer_index: int | None, dtype: torch.dtype):
    """``from_pretrained`` on a micro-checkpoint — the model, loaded by transformers end to end.

    ``key_mapping`` is what lets layer N of the real checkpoint land in layer 0 of a one-layer
    model: it is transformers' own rename hook, applied before the weight converters run, so the
    expert fusion still fires on the remapped names.

    The shell reports ``layers`` as absent and a layer slice reports ``embed_tokens``/``norm`` as
    absent. Both are expected — each slice deliberately carries only its own part.
    """
    from transformers import AutoModel

    mapping: dict[str, str] | None = None if layer_index is None else rename_rules(dest, layer_index)
    model, info = AutoModel.from_pretrained(
        dest,
        dtype=dtype,
        attn_implementation="sdpa",
        key_mapping=mapping,
        # The index names only this slice's tensors, but transformers uses it to choose FILES and
        # then loads whatever those files hold — so a layer sharing a shard with the embedding sees
        # a [154880, 6144] checkpoint tensor against the [16, 6144] stub this slice declares. That
        # one mismatch is intended; the assert below is what keeps it from covering a real one.
        ignore_mismatched_sizes=layer_index is not None,
        output_loading_info=True,
    )

    allowed: tuple[str, ...] = ("embed_tokens", "norm.weight", "layers")
    suspect: list[str] = [
        key
        for key in list(info.get("missing_keys", [])) + [str(k) for k in info.get("mismatched_keys", [])]
        if not any(token in key for token in allowed)
    ]
    assert not suspect, f"{dest.name}: transformers reinitialised weights it should have loaded: {suspect[:5]}"
    return model.eval()


@torch.no_grad()
def run_layer(layer, hidden: torch.Tensor, topk, cache, mask, position_embeddings, position_ids):
    """Run the sequence through one layer, returning its output, its top-k, and what it carried.

    ``prev_topk_indices`` is threaded exactly as ``GlmMoeDsaModel.forward`` threads it: a full
    layer computes its own and a shared layer reuses what it was handed. Nothing caches it, so the
    walk holds it as a live tensor between iterations.

    ``kvpe`` has to be hooked rather than read back: the forward computes it at
    ``modeling_glm_moe_dsa.py:427`` and then stores the DECOMPRESSED key/value into the cache, so
    the compressed latent the device produces exists only for the duration of the call.
    """
    captured: dict[str, torch.Tensor] = {}
    handle = layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(
        lambda module, args, output: captured.__setitem__("kvpe", output.detach())
    )
    try:
        hidden, topk = layer(
            hidden,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
            prev_topk_indices=topk,
        )
    finally:
        handle.remove()
    return hidden, topk, captured


def layer_file(index: int, width: int) -> str:
    return LAYER_FILE_TEMPLATE.format(index=index, width=width)


def write_layer(
    trace: Path,
    index: int,
    width: int,
    hidden: torch.Tensor,
    captured: dict,
    cache,
    topk,
    *,
    is_full_indexer: bool,
    store_dtype: torch.dtype,
) -> None:
    """``hidden/layer_NN.safetensors`` and ``cache/layer_NN.safetensors`` for one layer."""
    name: str = layer_file(index, width)
    cast = lambda t: t.to(store_dtype).cpu().contiguous()  # noqa: E731

    save_file({KEY_LAYER_OUTPUT.format(index=index): cast(hidden)}, str(trace / HIDDEN_DIR_NAME / name))

    tensors: dict[str, torch.Tensor] = {KEY_KVPE.format(index=index): cast(captured["kvpe"])}
    if is_full_indexer:
        # int32 selections, not activations: stored as-is, and graded as a set rather than by PCC.
        # Slot 0, not `index`: key_mapping loaded this layer as layer 0 of a one-layer model, so
        # its `layer_idx` is 0 and that is where it wrote its cache. `index` is only the name we
        # give it in the trace.
        tensors[KEY_INDEXER_KEYS.format(index=index)] = cast(cache.layers[0].indexer_keys)
        tensors[KEY_INDEXER_TOPK.format(index=index)] = topk.cpu().contiguous()
    save_file(tensors, str(trace / CACHE_DIR_NAME / name))


def summarize_timings(timings: list[LayerTiming], *, wall_s: float) -> None:
    """Totals and the per-kind breakdown — what a long walk actually spent its time on."""
    if not timings:
        return
    fetched: int = sum(t.fetched_bytes for t in timings)
    columns: tuple[str, ...] = ("fetch", "load", "compute", "write")
    totals: dict[str, float] = {c: sum(getattr(t, f"{c}_s") for t in timings) for c in columns}

    print(f"\n[time] {len(timings)} layers in {wall_s / 60:.1f} min   ({fetched / 2**30:.1f} GiB fetched)")
    print(f"[time] {'stage':>9s} {'seconds':>10s} {'share':>7s}")
    for column in columns:
        print(f"[time] {column:>9s} {totals[column]:10.1f} {100 * totals[column] / wall_s:6.1f}%")

    by_kind: dict[str, list[LayerTiming]] = {}
    for timing in timings:
        by_kind.setdefault(timing.kind, []).append(timing)
    print(f"[time] {'layers':>6s}  {'kind':<30s} {'compute/layer':>14s} {'total/layer':>12s}")
    for kind, group in sorted(by_kind.items()):
        compute: float = sum(t.compute_s for t in group) / len(group)
        total: float = sum(t.total_s for t in group) / len(group)
        print(f"[time] {len(group):6d}  {kind:<30s} {compute:13.2f}s {total:11.2f}s")


def walk(
    plan: CheckpointPlan,
    config: PretrainedConfig,
    input_ids: torch.Tensor,
    trace: Path,
    workdir: Path,
    *,
    dtype: torch.dtype,
    store_dtype: torch.dtype,
) -> list[LayerTiming]:
    """Prefill the whole model one layer at a time, writing each layer's tensors as it goes.

    Peak memory is one layer plus the activations, not the checkpoint — which is the entire point:
    GLM-5.2 is ~3.0 TB of fp32 weights against 566 GB of host RAM.
    """
    from transformers import DynamicCache
    from transformers.masking_utils import create_causal_mask

    raw_config: dict = json.loads(fetch_file(plan.source, CONFIG_NAME).read_text())
    width: int = len(str(plan.num_layers - 1))
    for name in (HIDDEN_DIR_NAME, CACHE_DIR_NAME):
        (trace / name).mkdir(parents=True, exist_ok=True)

    # The shell: a 0-layer model carrying the real embedding and the final norm, so both ends of
    # the walk run transformers' own modules rather than a reimplementation of them.
    started: float = time.time()
    shell_dir: Path = workdir / "shell"
    shell_dir.mkdir(parents=True, exist_ok=True)
    global_paths, global_bytes = fetch_shards(plan.source, plan.global_shards, shell_dir)
    write_micro_checkpoint(shell_dir, raw_config, load_weight_map(plan.source), plan.globals_, layer_index=None)
    shell = load_slice(shell_dir, layer_index=None, dtype=dtype)
    fetch_s: float = time.time() - started

    with torch.no_grad():
        hidden: torch.Tensor = shell.embed_tokens(input_ids)
    save_file(
        {KEY_EMBED: hidden.to(store_dtype).cpu().contiguous()},
        str(trace / HIDDEN_DIR_NAME / EMBED_FILE_NAME),
    )
    print(f"[walk] embedded {tuple(hidden.shape)} in {fetch_s:.1f}s", flush=True)

    position_ids: torch.Tensor = torch.arange(input_ids.shape[1]).unsqueeze(0)
    position_embeddings = shell.rotary_emb(hidden, position_ids=position_ids)

    # Built once and reused: it depends only on the sequence, and the attention masks into a COPY
    # (masked_fill is out-of-place), so no layer can corrupt it for the next.
    mask = create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    topk = None
    timings: list[LayerTiming] = []
    indexer_types: list[str] | None = getattr(config, "indexer_types", None)
    weight_map: dict[str, str] = load_weight_map(plan.source)

    for layer_plan in plan.layers:
        index: int = layer_plan.index
        slice_dir: Path = workdir / f"layer_{index}"
        slice_dir.mkdir(parents=True, exist_ok=True)
        mark: float = time.time()
        paths, fetched = fetch_shards(plan.source, layer_plan.shards, slice_dir)
        fetch_s = time.time() - mark

        mark = time.time()
        write_micro_checkpoint(slice_dir, raw_config, weight_map, layer_plan.tensors, layer_index=index)
        sliced = load_slice(slice_dir, layer_index=index, dtype=dtype)
        layer = sliced.layers[0]
        drop_shards(paths, slice_dir)
        load_s: float = time.time() - mark

        # A fresh cache per layer: one shared cache would accumulate every layer's K/V, which is
        # the footprint this whole file exists to avoid.
        # Built from the SLICE's config, not the full one: it has exactly one layer, declared with
        # this layer's own indexer type, so the cache allocates the matching layer class.
        cache = DynamicCache(config=sliced.config)
        mark = time.time()
        hidden, topk, captured = run_layer(layer, hidden, topk, cache, mask, position_embeddings, position_ids)
        compute_s: float = time.time() - mark

        mark = time.time()
        is_full: bool = indexer_types is None or indexer_types[index] == "full"
        write_layer(
            trace,
            index,
            width,
            hidden,
            captured,
            cache,
            topk,
            is_full_indexer=is_full,
            store_dtype=store_dtype,
        )
        write_s: float = time.time() - mark

        del layer, sliced, cache, captured
        timings.append(
            LayerTiming(
                index,
                layer_plan.kind,
                fetch_s,
                load_s,
                compute_s,
                write_s,
                fetched + (global_bytes if not index else 0),
            )
        )
        print(
            f"[walk] layer {index:3d}/{plan.num_layers}  {layer_plan.kind:<30s} "
            f"fetch {fetch_s:6.1f}s  load {load_s:6.1f}s  compute {compute_s:6.2f}s  write {write_s:5.2f}s",
            flush=True,
        )

    with torch.no_grad():
        final_hidden: torch.Tensor = shell.norm(hidden)
    save_file(
        {KEY_FINAL_HIDDEN: final_hidden.to(store_dtype).cpu().contiguous()},
        str(trace / HIDDEN_DIR_NAME / FINAL_FILE_NAME),
    )
    print(f"[walk] final_hidden {tuple(final_hidden.shape)}", flush=True)
    return timings


# --------------------------------------------------------------------------------------------


def main() -> int:
    import argparse
    import tempfile

    from models.prefill_golden_generation.whole_model_goldens import (
        check_isl,
        create_token_cache,
        load_text_config,
        read_token_cache,
        select_prefix,
        temporary_hf_cache,
        token_cache_exists,
    )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--text", type=Path, default=Path(DEFAULT_TEXT), help=f"the .txt (default {DEFAULT_TEXT})")
    ap.add_argument("--isl", type=int, default=DEFAULT_ISL, help=f"tokens to prefill (default {DEFAULT_ISL})")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help=f"weights + store subdir (default {DEFAULT_HF_REPO})")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE, help=f"shared store (default {DEFAULT_STORE})")
    ap.add_argument("--dtype", default=DEFAULT_COMPUTE_DTYPE, choices=tuple(DTYPES), help="compute dtype")
    ap.add_argument("--store-dtype", default=DEFAULT_STORE_DTYPE, choices=tuple(DTYPES), help="trace dtype")
    ap.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="local checkpoint dir to read instead of fetching --hf-repo; --hf-repo still names the store subdirectory",
    )
    ap.add_argument("--workdir", type=Path, default=None, help="where shards land in transit (default: a temp dir)")
    ap.add_argument("--layers", type=int, default=None, help="stop after this many layers (a diagnostic, not a grade)")
    ap.add_argument("--plan-only", action="store_true", help="print the plan and exit without fetching weights")
    ap.add_argument("--trust-remote-code", action="store_true", help="allow model code from the repo to execute")
    args: argparse.Namespace = ap.parse_args()

    source: str = str(args.model_path) if args.model_path else args.hf_repo
    if args.model_path:
        assert args.model_path.is_dir(), f"no checkpoint directory at {args.model_path}"
    text: Path = args.text if args.text.is_absolute() else args.store / args.text
    check_isl(args.isl)

    with temporary_hf_cache():
        config = load_text_config(source, trust_remote_code=args.trust_remote_code)
        config._attn_implementation = "sdpa"
        plan: CheckpointPlan = plan_checkpoint(source, config)
        if args.layers:
            plan = CheckpointPlan(
                plan.source, args.layers, plan.globals_, plan.global_shards, plan.layers[: args.layers]
            )
        summarize_plan(plan)
        if args.plan_only:
            return 0

        assert text.is_file(), f"no text file at {text}"
        if not token_cache_exists(text, args.hf_repo, args.store):
            create_token_cache(text, args.hf_repo, args.store, source=source, trust_remote_code=args.trust_remote_code)
        all_ids: torch.Tensor = read_token_cache(text, args.hf_repo, args.store)
        input_ids: torch.Tensor = select_prefix(all_ids, args.isl, config.max_position_embeddings)
        print(f"[tokens] prefilling {input_ids.numel()} of {all_ids.numel()} tokens", flush=True)

        trace: Path = args.store / args.hf_repo / f"isl{args.isl}"
        started: float = time.time()
        with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
            timings: list[LayerTiming] = walk(
                plan,
                config,
                input_ids,
                trace,
                Path(tmp),
                dtype=DTYPES[args.dtype],
                store_dtype=DTYPES[args.store_dtype],
            )
        wall_s: float = time.time() - started

    summarize_timings(timings, wall_s=wall_s)
    print(f"[trace] wrote {trace}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
