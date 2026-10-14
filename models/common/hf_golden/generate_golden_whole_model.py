# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Golden trace generation from a text file — HuggingFace only, CPU or GPU.

Built in stages:

* **Stage 1** — tokenize the text once, cache the ids::

      /mnt/models/agentic-prefill-goldens/
        wikitext-2-raw-v1-train.txt                       the text, model-independent
        Qwen/Qwen3.8-27B/
          wikitext-2-raw-v1-train.tokens.safetensors      that text under THIS tokenizer

* **Stage 2** — take the first ``--isl`` ids and run one HF forward, capturing each layer's
  carried state. Stage 3 will write those to a trace directory; for now they are summarized.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_STORE: Path = Path("/mnt/models/agentic-prefill-goldens")

# Names the tokenizer/weights to fetch AND the store subdirectory.
DEFAULT_HF_REPO: str = "Qwen/Qwen3.8-27B"

# The text staged in the store. 2504636 tokens under the Qwen tokenizer — ~9.5x the 262144
# context ceiling, so it covers every ISL with room to spare.
DEFAULT_TEXT: str = "wikitext-2-raw-v1-train.txt"

DEFAULT_ISL: int = 5120

TOKENS_KEY: str = "token_ids"

# Text per tokenizer call, so peak memory stays flat on an input of any size.
BLOCK_BYTES: int = 1 << 20

DTYPES: dict[str, torch.dtype] = {"bfloat16": torch.bfloat16, "float32": torch.float32}

# Compute in fp32, store in bf16 — deliberately different.
#
# fp32 is not the fast choice: measured on this host (EPYC 9354P, which has avx512_bf16) at
# [4096,5120]x[5120,17408], bf16 runs 164 ms against fp32's 371 ms, so fp32 costs ~2.3x. It is
# the default anyway because a golden has to be an INDEPENDENT oracle: the device computes in
# bf16, so a bf16 reference shares its rounding and raises measured PCC without anything having
# improved. Storage is bf16 regardless — that costs ~1.6e-6 PCC, far below the arithmetic itself.
DEFAULT_COMPUTE_DTYPE: str = "float32"
DEFAULT_STORE_DTYPE: str = "bfloat16"

# Filenames the device-side grader reads; see galaxy_prefill_kv_pcc.py.
METADATA_NAME: str = "metadata.json"
KV_DIR_NAME: str = "kv_cache"
FINAL_HIDDEN_NAME: str = "final_hidden.safetensors"
FINAL_HIDDEN_KEY: str = "final_hidden"

# ISL has to suit the device even though nothing here talks to one: galaxy_prefill_kv_pcc.py
# builds the runtime with max_seq_len=isl, and tt_prefill_runtime.py:40 asserts the sequence
# divides into 32-row tiles across the sp ranks. A trace at a misaligned ISL cannot be graded, so
# this is checked before the hours of work rather than after. Literals, not a spec import — this
# file stays clear of the device stack.
ISL_ALIGNMENT: int = 32 * 8  # ttnn.TILE_SIZE * sp
ISL_CHUNK_SIZE: int = 5120  # spec chunk_size; a chunked grade additionally needs this

# Config fields copied into metadata.json when the model has them, so a trace describes its own
# shapes. The GDN ones are Qwen3.5-only, hence "when the model has them".
RECORDED_CONFIG_FIELDS: tuple[str, ...] = (
    "layer_types",
    "num_key_value_heads",
    "head_dim",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
)


# --------------------------------------------------------------------------------------------
# shared store hygiene
# --------------------------------------------------------------------------------------------


@contextmanager
def temporary_hf_cache() -> Iterator[str]:
    """Point every Hub operation at a temp dir for the duration, then delete it.

    Set through the environment rather than ``cache_dir=``, because transformers does not forward
    ``cache_dir`` to all of its internal Hub calls: ``AutoModel.from_pretrained`` probes for a PEFT
    adapter via ``find_adapter_config_file()``, which ignores it and falls back to ``HF_HOME``.
    ``env.sh`` points that at the shared /mnt/models cache, whose ``refs/main`` is read-only and
    owned by another user, so the probe dies with ``PermissionError``.

    Must be entered **before** transformers is first imported — huggingface_hub reads these into
    module constants at import time, so a later change has no effect. Every transformers import in
    this file is function-local, which is what makes that work.
    """
    with tempfile.TemporaryDirectory() as download_dir:
        previous: dict[str, str | None] = {key: os.environ.get(key) for key in ("HF_HOME", "HF_HUB_CACHE")}
        os.environ["HF_HOME"] = download_dir
        os.environ["HF_HUB_CACHE"] = str(Path(download_dir) / "hub")
        try:
            yield download_dir
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _share(path: Path) -> None:
    """``a+rX`` — the store is shared, but a 0077 umask writes files nobody else can read.

    OR-ed onto the current mode to keep the store's setgid bit. Warns rather than raises: a path
    owned by someone else is no reason to lose an otherwise correct artifact.
    """
    try:
        mode: int = path.stat().st_mode
        extra: int = 0o055 if path.is_dir() else 0o044
        if mode & extra != extra:
            path.chmod((mode & 0o7777) | extra)
    except OSError as e:
        print(f"[perm] WARNING: could not chmod {path}: {e}", file=sys.stderr, flush=True)


def _share_under(path: Path, store: Path) -> None:
    """``a+rX`` on ``path`` and each directory down from ``store``, never above it."""
    try:
        relative: Path = path.relative_to(store)
    except ValueError:
        _share(path)
        return
    current: Path = store
    for part in relative.parts[:-1]:
        current = current / part
        _share(current)
    _share(path)


# --------------------------------------------------------------------------------------------
# stage 1 — tokenize and cache
# --------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Content hash of the text, so an edited file invalidates its cache."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_blocks(text_path: Path, block_bytes: int = BLOCK_BYTES) -> Iterator[str]:
    """The text in line-aligned blocks, so seams fall where the tokenizer would break anyway.

    Measured: on wikitext-2-raw-v1-train under the Qwen3.8-27B tokenizer this is bit-identical to
    tokenizing the file as one string — 0 of 2504636 positions differ.
    """
    buffer: list[str] = []
    size: int = 0
    with text_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            buffer.append(line)
            size += len(line)
            if size >= block_bytes:
                yield "".join(buffer)
                buffer, size = [], 0
    if buffer:
        yield "".join(buffer)


def token_cache_path(text_path: Path, hf_repo: str = DEFAULT_HF_REPO, store: Path = DEFAULT_STORE) -> Path:
    """``<store>/<hf_repo>/<text stem>.tokens.safetensors`` — model the directory, text the name."""
    return store / hf_repo / f"{text_path.stem}.tokens.safetensors"


def token_cache_exists(text_path: Path, hf_repo: str = DEFAULT_HF_REPO, store: Path = DEFAULT_STORE) -> bool:
    """True when a cache is present **and** still matches the text.

    Not just a file check: an edited text file leaves a cache that exists but describes something
    else. The hash lives in the safetensors header, which ``safe_open`` reads without the tensor.
    """
    cache: Path = token_cache_path(text_path, hf_repo, store)
    if not cache.is_file():
        return False
    with safe_open(str(cache), framework="pt") as fh:
        meta: dict[str, str] = fh.metadata() or {}
    return meta.get("source_sha256") == _sha256(text_path)


def create_token_cache(
    text_path: Path,
    hf_repo: str = DEFAULT_HF_REPO,
    store: Path = DEFAULT_STORE,
    *,
    source: str | None = None,
    trust_remote_code: bool = False,
) -> Path:
    """Tokenize the whole text and write the cache, overwriting any existing one.

    ``int32`` because the vocabulary is ~248k: lossless, and half the file.

    ``add_special_tokens=False``: a trace is a contiguous prefix, and a BOS at every block seam
    would corrupt it. Qwen has none; Llama does, so it belongs at trace time instead.
    """
    from transformers import AutoTokenizer

    cache: Path = token_cache_path(text_path, hf_repo, store)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        source or hf_repo, trust_remote_code=trust_remote_code
    )

    total_bytes: int = text_path.stat().st_size
    chunks: list[torch.Tensor] = []
    with tqdm(total=total_bytes, desc=f"tokenizing {text_path.name}", unit="B", unit_scale=True) as progress:
        for block in _text_blocks(text_path):
            chunks.append(torch.tensor(tokenizer(block, add_special_tokens=False)["input_ids"], dtype=torch.int32))
            progress.update(len(block))
    ids: torch.Tensor = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.int32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {TOKENS_KEY: ids},
        str(cache),
        metadata={
            "source": text_path.name,
            "source_sha256": _sha256(text_path),
            "source_bytes": str(total_bytes),
            "hf_repo": hf_repo,
            "tokenizer_class": type(tokenizer).__name__,
            "n_tokens": str(ids.numel()),
            "add_special_tokens": "false",
            "block_bytes": str(BLOCK_BYTES),
        },
    )
    _share_under(cache, store)
    print(f"[tokens] wrote {ids.numel()} tokens to {cache}", flush=True)
    return cache


def read_token_cache(text_path: Path, hf_repo: str = DEFAULT_HF_REPO, store: Path = DEFAULT_STORE) -> torch.Tensor:
    """The cached token ids as one ``int32`` tensor. Call ``create_token_cache`` first if absent."""
    cache: Path = token_cache_path(text_path, hf_repo, store)
    if not cache.is_file():
        raise FileNotFoundError(f"no token cache at {cache}; run create_token_cache() first")
    ids: torch.Tensor = load_file(str(cache))[TOKENS_KEY]
    _share_under(cache, store)
    print(f"[tokens] read {ids.numel()} tokens from {cache}", flush=True)
    return ids


# --------------------------------------------------------------------------------------------
# stage 2 — slice to ISL and run one forward
# --------------------------------------------------------------------------------------------


@dataclass
class ForwardResult:
    """One prefill forward: the post-final-norm hidden states, and each layer's carried state.

    ``layer_states`` is per layer and its keys depend on the layer kind — ``key``/``value`` for
    attention, ``conv_state``/``recurrent_state`` for a recurrent (Gated DeltaNet) layer.
    """

    final_hidden: torch.Tensor
    layer_states: list[dict[str, torch.Tensor]]


def resolve_device(requested: str) -> str:
    """``auto`` picks CUDA when it is there, else CPU."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_text_config(source: str, *, trust_remote_code: bool = False) -> PretrainedConfig:
    """The text tower's config. Multimodal checkpoints (Qwen3.5 is one) nest it under text_config."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(source, trust_remote_code=trust_remote_code)
    return getattr(config, "text_config", config)


def select_prefix(token_ids: torch.Tensor, isl: int, max_positions: int) -> torch.Tensor:
    """The first ``isl`` ids as an ``int64`` ``[1, isl]`` batch.

    Refuses to exceed the model's trained context: past ``max_position_embeddings`` the RoPE
    positions are out of distribution, so the result would not be a golden of anything.
    """
    if isl > max_positions:
        raise ValueError(f"isl {isl} exceeds the model's context window of {max_positions}")
    if isl > token_ids.numel():
        raise ValueError(f"isl {isl} exceeds the {token_ids.numel()} tokens in the cache")
    return token_ids[:isl].to(torch.int64).unsqueeze(0)


def load_text_model(
    source: str,
    *,
    dtype: torch.dtype,
    device: str,
    trust_remote_code: bool = False,
) -> PreTrainedModel:
    """The text backbone on ``device``, from a repo id or a local checkpoint directory.

    Call inside ``temporary_hf_cache()`` for a repo id; a local directory bypasses the Hub
    entirely and needs neither the cache nor a token.

    ``sdpa``, not the eager attention the old script pinned: eager materialises the whole
    ``[1, heads, isl, isl]`` score matrix, which is 10 GB at isl 10240 and 6.6 TB at the 262144
    ceiling. SDPA's softmax is exact, so this costs no fidelity.

    ``.language_model`` is the text tower of a multimodal wrapper — Qwen3.8-27B ships as
    ``Qwen3_5ForConditionalGeneration``. Models without one (Llama, Mistral) fall through unchanged.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        source,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=trust_remote_code,
    )
    backbone = getattr(model, "language_model", model)
    return backbone.to(device).eval()


def capture_layer_state(layer: object) -> dict[str, torch.Tensor]:
    """One cache layer's carried state, whatever kind of layer it is.

    Duck-typed on the cache object rather than dispatched on the config, so attention, recurrent
    and hybrid models all work without a per-model branch.
    """
    keys = getattr(layer, "keys", None)
    if keys is not None:
        return {"key": keys, "value": layer.values}
    return {"conv_state": layer.conv_states, "recurrent_state": layer.recurrent_states}


@torch.no_grad()
def run_forward(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    *,
    store_dtype: torch.dtype = torch.bfloat16,
) -> ForwardResult:
    """One full-sequence prefill, returning the final hidden states and every layer's state.

    Everything comes back on the host in ``store_dtype``, so a GPU run hands back tensors that are
    ready to write and frees the device memory with the model.
    """
    output = model(input_ids=input_ids, use_cache=True)
    states: list[dict[str, torch.Tensor]] = [
        {name: tensor.to(store_dtype).cpu() for name, tensor in capture_layer_state(layer).items()}
        for layer in output.past_key_values.layers
    ]
    return ForwardResult(final_hidden=output.last_hidden_state.to(store_dtype).cpu(), layer_states=states)


def summarize(result: ForwardResult) -> None:
    """Shapes of what the forward produced, one line per distinct layer kind."""
    print(f"[forward] final_hidden {tuple(result.final_hidden.shape)} {result.final_hidden.dtype}")
    seen: set[tuple[str, ...]] = set()
    for index, state in enumerate(result.layer_states):
        kind: tuple[str, ...] = tuple(sorted(state))
        if kind not in seen:
            seen.add(kind)
            shapes = "  ".join(f"{name} {tuple(tensor.shape)}" for name, tensor in sorted(state.items()))
            print(f"[forward] layer {index:2d} ({'/'.join(kind)}): {shapes}")
    print(f"[forward] {len(result.layer_states)} layers captured")


# --------------------------------------------------------------------------------------------
# stage 3 — write the trace
# --------------------------------------------------------------------------------------------


def check_isl(isl: int) -> None:
    """Reject an ISL no device run could grade. See ISL_ALIGNMENT."""
    if isl % ISL_ALIGNMENT:
        raise ValueError(
            f"isl {isl} is not a multiple of {ISL_ALIGNMENT} (TILE_SIZE*sp); the device shards the "
            f"sequence over sp ranks in 32-row tiles, so this trace could never be graded. "
            f"Nearest valid: {isl // ISL_ALIGNMENT * ISL_ALIGNMENT} or "
            f"{(isl // ISL_ALIGNMENT + 1) * ISL_ALIGNMENT}"
        )
    if isl % ISL_CHUNK_SIZE:
        print(
            f"[isl] note: {isl} is not a multiple of chunk_size {ISL_CHUNK_SIZE}, so this trace "
            f"grades one-shot runs only, not chunked ones",
            flush=True,
        )


def trace_path(text_path: Path, isl: int, hf_repo: str = DEFAULT_HF_REPO, store: Path = DEFAULT_STORE) -> Path:
    """``<store>/<hf_repo>/isl<N>/`` — beside that model's token cache.

    The text is not in the name, so two texts at one ISL share a directory. That is safe rather
    than silently wrong: ``trace_exists`` compares the recorded ``source_sha256``, so a trace from
    a different text reads as stale and is regenerated instead of being graded against.
    """
    del text_path  # named for symmetry with the other path helpers
    return store / hf_repo / f"isl{isl}"


def trace_exists(text_path: Path, isl: int, hf_repo: str = DEFAULT_HF_REPO, store: Path = DEFAULT_STORE) -> bool:
    """True when a complete trace for this (text, isl) is already written.

    Checks every per-layer file, not just ``metadata.json``: an interrupted run leaves the
    metadata behind, and a half-written trace that looks present is worse than none.
    """
    trace: Path = trace_path(text_path, isl, hf_repo, store)
    metadata_file: Path = trace / METADATA_NAME
    if not metadata_file.is_file() or not (trace / FINAL_HIDDEN_NAME).is_file():
        return False
    metadata: dict = json.loads(metadata_file.read_text())
    if metadata.get("n_tokens") != isl or metadata.get("source_sha256") != _sha256(text_path):
        return False
    return all(
        (trace / KV_DIR_NAME / f"layer_{index}.safetensors").is_file() for index in range(metadata.get("num_layers", 0))
    )


def layer_tensors(index: int, state: dict[str, torch.Tensor], conv_kernel: int) -> dict[str, torch.Tensor]:
    """One layer's captured state under the key names the grader reads.

    The conv history is trimmed to the last ``kernel-1`` columns. Upstream caches ``kernel`` of
    them; the device carries the minimal ``kernel-1``, and ``device_conv_state_to_hf`` hands back
    ``[1, conv_dim, kernel-1]`` — so an untrimmed golden fails the grader's shape assert.
    """
    if "key" in state:
        return {
            f"key_cache_layer_{index}": state["key"],
            f"value_cache_layer_{index}": state["value"],
        }
    return {
        f"recurrent_state_layer_{index}": state["recurrent_state"],
        f"conv_state_layer_{index}": state["conv_state"][..., -(conv_kernel - 1) :].contiguous(),
    }


def read_trace_tokens(trace: Path) -> torch.Tensor:
    """The token ids a trace was generated from, as ``int64 [1, n_tokens]``.

    Follows the ``token_cache`` pointer in ``metadata.json`` rather than reading ids out of it —
    the trace is the first ``n_tokens`` of that file, so the ids live in exactly one place.
    """
    metadata: dict = json.loads((trace / METADATA_NAME).read_text())
    cache: Path = (trace / metadata["token_cache"]).resolve()
    ids: torch.Tensor = load_file(str(cache))[TOKENS_KEY][: metadata["n_tokens"]]
    return ids.to(torch.int64).reshape(1, -1)


def write_trace(
    trace: Path,
    result: ForwardResult,
    config: PretrainedConfig,
    *,
    text_path: Path,
    n_tokens: int,
    hf_repo: str = DEFAULT_HF_REPO,
    store: Path = DEFAULT_STORE,
    extra_metadata: dict | None = None,
) -> Path:
    """Write ``metadata.json``, ``kv_cache/layer_N.safetensors`` and ``final_hidden.safetensors``.

    Layout and key names are the ones ``galaxy_prefill_kv_pcc.py`` reads out of
    ``PREFILL_TRACE_DIR``, so a trace from here is interchangeable with the existing goldens and
    with ``compare_traces.py``.
    """
    kv_dir: Path = trace / KV_DIR_NAME
    kv_dir.mkdir(parents=True, exist_ok=True)

    shapes: dict[str, list[int]] = {}
    for index, state in enumerate(tqdm(result.layer_states, desc="writing layers", unit="layer")):
        tensors: dict[str, torch.Tensor] = layer_tensors(index, state, getattr(config, "linear_conv_kernel_dim", 0))
        for name, tensor in tensors.items():
            shapes.setdefault(name.rsplit("_layer_", 1)[0], list(tensor.shape))
        save_file(tensors, str(kv_dir / f"layer_{index}.safetensors"))

    save_file({FINAL_HIDDEN_KEY: result.final_hidden.contiguous()}, str(trace / FINAL_HIDDEN_NAME))

    # A pointer, not a copy: the ids already exist in the token cache and the trace is its first
    # n_tokens. Relative, so moving or copying the model's directory keeps it resolvable. Use
    # read_trace_tokens() to follow it.
    token_cache: Path = token_cache_path(text_path, hf_repo, store)
    metadata: dict = {
        "token_cache": os.path.relpath(token_cache, trace),
        "n_tokens": n_tokens,
        "source": text_path.name,
        "source_sha256": _sha256(text_path),
        "num_layers": len(result.layer_states),
        "reduced": False,
        # getattr, not attribute access: the GDN fields exist on Qwen3.5 and not on a plain
        # attention model, and a trace generator should not die on the model it does not know.
        **{
            name: value
            for name, value in ((field, getattr(config, field, None)) for field in RECORDED_CONFIG_FIELDS)
            if value is not None
        },
        "final_hidden_shape": list(result.final_hidden.shape),
        **{f"{name}_shape": shape for name, shape in shapes.items()},
        **(extra_metadata or {}),
    }
    (trace / METADATA_NAME).write_text(json.dumps(metadata, indent=2))

    for path in (trace, kv_dir, trace / METADATA_NAME, trace / FINAL_HIDDEN_NAME, *kv_dir.iterdir()):
        _share_under(path, store)
    print(f"[trace] wrote {trace}", flush=True)
    return trace


# --------------------------------------------------------------------------------------------


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Generate a golden trace from a text file.")
    ap.add_argument(
        "--text",
        type=Path,
        default=Path(DEFAULT_TEXT),
        help=f"the .txt, absolute or a name under --store (default {DEFAULT_TEXT})",
    )
    ap.add_argument("--isl", type=int, default=DEFAULT_ISL, help=f"tokens to prefill (default {DEFAULT_ISL})")
    ap.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help=f"tokenizer + store subdir (default {DEFAULT_HF_REPO})")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE, help=f"shared store (default {DEFAULT_STORE})")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="default: cuda when available")
    ap.add_argument(
        "--dtype",
        default=DEFAULT_COMPUTE_DTYPE,
        choices=tuple(DTYPES),
        help=f"compute dtype (default {DEFAULT_COMPUTE_DTYPE}); bfloat16 is ~2.3x faster but "
        f"shares the device's rounding, so it is a weaker oracle",
    )
    ap.add_argument(
        "--store-dtype",
        default=DEFAULT_STORE_DTYPE,
        choices=tuple(DTYPES),
        help=f"dtype the trace is written in (default {DEFAULT_STORE_DTYPE})",
    )
    ap.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="local checkpoint dir to load from instead of fetching --hf-repo; --hf-repo still "
        "names the store subdirectory",
    )
    ap.add_argument("--trust-remote-code", action="store_true", help="allow model code from the repo to execute")
    ap.add_argument("--force", action="store_true", help="regenerate even when the token cache and trace are valid")
    args: argparse.Namespace = ap.parse_args()

    text: Path = args.text if args.text.is_absolute() else args.store / args.text
    assert text.is_file(), f"no text file at {text}"
    check_isl(args.isl)

    trace: Path = trace_path(text, args.isl, args.hf_repo, args.store)
    if not args.force and trace_exists(text, args.isl, args.hf_repo, args.store):
        print(f"[trace] already complete: {trace}", flush=True)
        return 0

    device: str = resolve_device(args.device)
    dtype: torch.dtype = DTYPES[args.dtype]
    store_dtype: torch.dtype = DTYPES[args.store_dtype]

    # What tokenizer, config and weights are loaded FROM. --hf-repo keeps naming the store
    # subdirectory either way, so a trace lands in the same place whether the weights came off the
    # Hub or off local disk.
    source: str = str(args.model_path) if args.model_path else args.hf_repo
    if args.model_path:
        assert args.model_path.is_dir(), f"no checkpoint directory at {args.model_path}"
        print(f"[model] loading from local checkpoint {args.model_path}", flush=True)

    # Wraps the whole run — tokenizer, config and weights all land in one temp dir that is
    # discarded on exit. A --model-path run touches the Hub for none of it.
    with temporary_hf_cache():
        if args.force or not token_cache_exists(text, args.hf_repo, args.store):
            create_token_cache(text, args.hf_repo, args.store, source=source, trust_remote_code=args.trust_remote_code)
        all_ids: torch.Tensor = read_token_cache(text, args.hf_repo, args.store)

        config: PretrainedConfig = load_text_config(source, trust_remote_code=args.trust_remote_code)
        input_ids: torch.Tensor = select_prefix(all_ids, args.isl, config.max_position_embeddings).to(device)
        print(f"[tokens] prefilling {input_ids.numel()} of {all_ids.numel()} tokens", flush=True)

        print(f"[model] {args.hf_repo} on {device}, compute {dtype}, store {store_dtype}", flush=True)
        model: PreTrainedModel = load_text_model(
            source, dtype=dtype, device=device, trust_remote_code=args.trust_remote_code
        )
        started: float = time.time()
        result: ForwardResult = run_forward(model, input_ids, store_dtype=store_dtype)
        elapsed: float = time.time() - started
        del model  # ~111 GB at fp32; the result is already host-side, so free it before writing

    print(f"[forward] {args.isl} tokens in {elapsed / 60:.1f} min ({args.isl / elapsed:.1f} tok/s)", flush=True)
    summarize(result)
    write_trace(
        trace,
        result,
        config,
        text_path=text,
        n_tokens=args.isl,
        hf_repo=args.hf_repo,
        store=args.store,
        extra_metadata={
            "hf_repo": args.hf_repo,
            "loaded_from": source,
            "compute_dtype": args.dtype,
            "store_dtype": args.store_dtype,
            "device": device,
            "attn_implementation": "sdpa",
            "forward_seconds": elapsed,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
