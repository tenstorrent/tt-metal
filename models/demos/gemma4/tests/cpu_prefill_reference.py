# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side reference for the whole 60-layer Gemma4 prefill graph.

``test_prefill_layers`` / ``test_prefill_full`` only assert shape, finiteness and
a plausible argmax, because there was no host reference for the full stack. That
makes them smoke tests: a context-parallel bug that shifted token positions would
still produce finite, plausible-looking output. This module supplies the missing
reference so the 60-layer graph can be checked by PCC, the same way
``test_prefill_layer`` checks a single layer.

The reference runs HuggingFace on CPU in fp32 with the checkpoint's bf16 weights —
the same "bf16 values in an fp32 module" setup the single-layer test uses, so the
PCC numbers are comparable. It is slow (tens of minutes at 4k on 64 cores) and
needs ~130 GB of RAM for a 31B model, so it is generated **once** and cached to
disk; the device test then just loads it.

Only the 4096-token bucket is generated: it is the largest trace-eligible chunk
and the one the CP work targets.

Generate::

    export HF_HUB_OFFLINE=1 HF_HOME=... HF_MODEL=google/gemma-4-31B-it \\
           TT_CACHE_PATH=...
    python -m models.demos.gemma4.tests.cpu_prefill_reference

The dump lands next to the tensor cache and holds the post-final-norm hidden
states, the last-token logit tile, the exact input tokens, and a fingerprint so a
stale dump is detected rather than silently compared against.
"""

from __future__ import annotations

import os
import pathlib
import time
from pathlib import Path

import torch

REFERENCE_CHUNK = 4096

# Long-context references, one per device target. Every one of these is reachable
# on CPU; an earlier note here claimed 256k was not, on a pure-quadratic
# extrapolation. That was wrong twice over. 50 of the 60 layers are sliding-window
# and therefore linear in n, so the true cost fits a*n + b*n^2 and lands at ~1.5 h
# for 64k, ~4.8 h for 128k and ~17 h for 256k. And the ~8.8 TB score matrix is an
# artifact of eager attention, not of the model: sdpa computes the same values in
# O(n) memory, verified bit-identical to eager at 8k (max abs diff 0.0).
LONG_REFERENCE_CONTEXTS = [8192, 16384, 32768, 65536, 131072, 262144]

# Bumped when the dump's contents or semantics change, so an old file on disk is
# rejected instead of being compared against.
_FORMAT_VERSION = 2


def _model_path() -> str:
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "google/gemma-4-31B-it")


def _cached_tokens(model_path, num_tokens):
    """``num_tokens`` tokens from any long reference already on disk, or None.

    Exists to keep the generator off the ttnn import path. The tokenizer is reached
    through the demo module, and importing that opens the UMD cluster and takes the PCIe
    chip lock -- for the whole run, which is pure CPU work. At 30 minutes that was merely
    untidy; at ~5 h for 128k it would block every device test on the machine for most of a
    day. Every dump stores its tokens and the sequences are prefix-consistent, so once any
    reference at least this long exists the tokenizer is not needed again.

    Only reads dumps at the current _FORMAT_VERSION. v1 dumps carry the old
    prompt-plus-random-filler sequence; slicing those here would silently rebuild the old
    sequence under the new scheme.
    """
    for ctx in sorted(LONG_REFERENCE_CONTEXTS):
        if ctx < num_tokens:
            continue
        path = long_reference_path(model_path, ctx)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["tokens"].shape[-1] >= num_tokens:
            return payload["tokens"][:, :num_tokens].clone()
    return None


def _long_context_text():
    """The long-context source text, read without importing the demo module.

    Deliberately does NOT go through ``text_demo.load_demo_prompt``. Importing that module
    opens the UMD cluster and takes the PCIe chip lock for the lifetime of the process --
    which for a reference generation is hours of pure CPU work with a device held hostage,
    blocking every device test on the machine. The demo's loader is 15 lines of JSON read
    plus an md5-named cache file, so it is reproduced here instead, reading the same cache
    the demo populates.
    """
    import hashlib
    import json

    prompts_dir = pathlib.Path("models/tt_transformers/demo/sample_prompts")
    cache_dir = pathlib.Path("models/tt_transformers/demo/context_cache")
    # The largest source file, unclipped: every smaller bucket clips the same text to fewer
    # characters, so this is the stream all of them are prefixes of.
    entry = json.loads((prompts_dir / "input_data_long_256k.json").read_text())[0]
    url = entry["context"]
    cache_file = cache_dir / hashlib.md5(url.encode()).hexdigest()
    if cache_file.exists():
        return cache_file.read_text()

    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cache_file.write_text(resp.text)
    return resp.text


def _real_token_stream(model_path, num_tokens):
    """``num_tokens`` tokens of real text, tiling the source when it runs short.

    No chat template: it would append the question after the context, so a 32k and a 128k
    sequence would diverge at their tails and stop being prefixes of one another.

    pg84 tokenizes to 100,680 tokens, so anything past ~100k repeats it. Repetition is still
    language -- the point is that attention sees real token statistics rather than
    uniform-random ids, not that the text is novel.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = torch.tensor(tokenizer.encode(_long_context_text()), dtype=torch.int32).unsqueeze(0)
    if ids.shape[-1] < num_tokens:
        ids = ids.repeat(1, -(-num_tokens // ids.shape[-1]))
    return ids[:, :num_tokens].clone(), tokenizer


def build_token_sequence(model_path, chunk, context_len, vocab_size=None):
    """The exact token sequence a chunked prefill run consumes.

    Real text end to end. Every chunk is language, not uniform-random ids, so a per-chunk
    PCC means the same thing for chunk 0 as for chunk 7 -- under the old scheme chunk 0
    held the real prompt and the rest was seeded filler, which sat ~0.045 higher and made
    chunk 0 the only chunk the accuracy gate ever tripped on.

    Shared by the reference generator and the device test so the two cannot drift; a PCC
    comparison against a differently-tokenized reference would be worse than none.

    Prefix-consistent by construction, since the sequence is a slice of one token stream:
    the first N tokens of a 256k sequence are the whole of an N-token sequence.
    ``test_prefill_long_context_prefix_pcc`` depends on that and verifies it by hash rather
    than trusting this comment.

    ``vocab_size`` is unused now that nothing is sampled; kept so existing callers work.
    """
    cached = _cached_tokens(model_path, context_len)
    if cached is not None:
        return cached, None, context_len
    tokens, tokenizer = _real_token_stream(model_path, context_len)
    return tokens, tokenizer, context_len


def reference_path(model_path: str | None = None, chunk: int = REFERENCE_CHUNK) -> Path:
    """Where the dump lives: alongside the tensor cache, keyed by model and chunk.

    Uses TT_CACHE_PATH when set (the prefill harness already points there),
    otherwise a local directory so the file is never written somewhere surprising.
    """
    model_path = model_path or _model_path()
    root = os.environ.get("TT_CACHE_PATH")
    base = Path(root) if root else Path("generated") / "gemma4_cpu_reference"
    slug = os.path.basename(model_path.rstrip("/")).replace("/", "--")
    return base / f"cpu_prefill_reference_{slug}_chunk{chunk}_v{_FORMAT_VERSION}.pt"


def _fingerprint(tokens: torch.Tensor, model_path: str, chunk: int) -> dict:
    """Identity of the run this dump describes.

    The token hash is the important part: the device test must feed byte-identical
    input, and ``_prompt_tokens`` depends on the tokenizer and the prompt source.
    """
    return {
        "format_version": _FORMAT_VERSION,
        "model": os.path.basename(model_path.rstrip("/")),
        "chunk": chunk,
        "num_tokens": int(tokens.numel()),
        "token_sha": hash_tokens(tokens),
    }


def hash_tokens(tokens: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(tokens.to(torch.int64).cpu().numpy().tobytes()).hexdigest()[:16]


def load(model_path: str | None = None, chunk: int = REFERENCE_CHUNK):
    """Load the dump, or return None when it has not been generated yet."""
    path = reference_path(model_path, chunk)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def generate(model_path: str | None = None, chunk: int = REFERENCE_CHUNK, out_path: Path | None = None) -> Path:
    """Run the full prefill graph on CPU and dump the reference.

    Uses the *same* ``_prompt_tokens`` the device test uses, so the input is
    identical by construction rather than by convention.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    # Imported here (not at module scope) so `python -m` does not pay for it when
    # only inspecting, and so this module stays importable without pytest present.
    from models.demos.gemma4.demo.text_demo_prefill import _prompt_tokens

    model_path = model_path or _model_path()
    out_path = out_path or reference_path(model_path, chunk)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokens, _tokenizer, prompt_len = _prompt_tokens(model_path, chunk)
    print(f"[cpu-ref] tokens={tuple(tokens.shape)} prompt_len={prompt_len} sha={hash_tokens(tokens)}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # eager attention so the reference does not depend on an SDPA backend's
    # numerics, matching what the single-layer test does.
    text_config = getattr(config, "text_config", config)
    text_config._attn_implementation = "eager"
    config._attn_implementation = "eager"

    print("[cpu-ref] loading weights in fp32 (this needs ~130 GB and a few minutes)...")
    t0 = time.time()
    # from_pretrained handles the checkpoint's model.language_model.* layout, the
    # tied lm_head, and buffer init — all of which are easy to get subtly wrong
    # when hand-loading a bare Gemma4TextModel.
    full = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    full.eval()
    text_model = full.model.language_model
    print(f"[cpu-ref] weights loaded in {time.time() - t0:.0f}s; {len(text_model.layers)} decoder layers")

    print(f"[cpu-ref] running prefill over {chunk} tokens...")
    t0 = time.time()
    with torch.no_grad():
        out = text_model(input_ids=tokens.to(torch.long), use_cache=False)
    hidden = out.last_hidden_state.to(torch.float32)  # [1, chunk, hidden], post final norm
    forward_s = time.time() - t0
    print(f"[cpu-ref] prefill done in {forward_s:.0f}s; hidden={tuple(hidden.shape)}")

    # Last-token logit tile, matching the device path: slice the 32-row tile that
    # contains the last real token, then lm_head, then softcap. Computing logits
    # for all `chunk` rows would be ~4.3 GB at a 262k vocab.
    last_token_idx = prompt_len - 1
    tile_start = (last_token_idx // 32) * 32
    with torch.no_grad():
        tile = hidden[:, tile_start : tile_start + 32, :]
        logits_tile = torch.nn.functional.linear(tile, full.lm_head.weight.to(torch.float32))
        cap = getattr(text_config, "final_logit_softcapping", None)
        if cap:
            logits_tile = torch.tanh(logits_tile / cap) * cap
    print(f"[cpu-ref] logits tile={tuple(logits_tile.shape)} (tile_start={tile_start}, softcap={cap})")

    payload = {
        "fingerprint": _fingerprint(tokens, model_path, chunk),
        "tokens": tokens.cpu(),
        "prompt_len": int(prompt_len),
        "last_token_idx": int(last_token_idx),
        "tile_start": int(tile_start),
        "hidden": hidden.cpu(),
        "logits_tile": logits_tile.cpu(),
        "forward_seconds": forward_s,
    }
    torch.save(payload, out_path)
    size_gb = out_path.stat().st_size / 1e9
    print(f"[cpu-ref] wrote {out_path} ({size_gb:.2f} GB)")
    return out_path


if __name__ == "__main__" and not os.environ.get("GEMMA4_CPU_REF_CONTEXT"):
    generate()


def long_reference_path(model_path: str | None = None, context_len: int = 8192) -> Path:
    """Dump location for a long-context reference, keyed by context length."""
    model_path = model_path or _model_path()
    root = os.environ.get("TT_CACHE_PATH")
    base = Path(root) if root else Path("generated") / "gemma4_cpu_reference"
    slug = os.path.basename(model_path.rstrip("/")).replace("/", "--")
    return base / f"cpu_prefill_reference_{slug}_ctx{context_len}_v{_FORMAT_VERSION}.pt"


def load_long(model_path: str | None = None, context_len: int = 8192):
    path = long_reference_path(model_path, context_len)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def generate_long(model_path: str | None = None, context_len: int = 8192, chunk: int = REFERENCE_CHUNK) -> Path:
    """Whole-sequence CPU reference for a ``context_len``-token prefill.

    The device runs this as ``context_len / chunk`` chunks with each chunk after
    the first reading history through the ring; the reference is one flat forward
    over the same tokens. Comparing per-chunk slices is what actually tests the
    history read — a chunk that ignored the prefix would diverge here while still
    looking finite and stable.

    Cost: measured 308 s at 8k, 710 s at 16k, 1837 s at 32k. Fitting those to
    ``a*n + b*n^2`` (50 of the 60 layers are sliding-window and therefore linear in n;
    only the 10 full-attention layers are quadratic) gives ~1.5 h at 64k, ~4.8 h at
    128k and ~17 h at 256k. A pure-quadratic extrapolation overstates this badly —
    the linear half of the model dominates until well past 32k.

    Attention implementation matters more than the arithmetic. Eager materializes a
    ``[heads, n, n]`` score matrix — 8.8 TB per layer at 256k, which is what puts
    eager out of reach there rather than the runtime. ``sdpa`` computes the same thing
    in O(n) memory, so it is the default; ``GEMMA4_CPU_REF_ATTN=eager`` forces the
    original path, which is how the two were checked against each other.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    model_path = model_path or _model_path()
    out_path = long_reference_path(model_path, context_len)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    attn = os.environ.get("GEMMA4_CPU_REF_ATTN", "sdpa")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    text_config._attn_implementation = attn
    config._attn_implementation = attn
    print(f"[cpu-ref] attn_implementation={attn}", flush=True)

    tokens, _tokenizer, prompt_len = build_token_sequence(model_path, chunk, context_len, text_config.vocab_size)
    assert tokens.shape[-1] == context_len, f"token sequence is {tokens.shape[-1]}, expected {context_len}"
    print(f"[cpu-ref] ctx={context_len} tokens={tuple(tokens.shape)} sha={hash_tokens(tokens)}", flush=True)

    print("[cpu-ref] loading weights in fp32...", flush=True)
    t0 = time.time()
    full = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True
    )
    full.eval()
    text_model = full.model.language_model
    print(f"[cpu-ref] weights loaded in {time.time() - t0:.0f}s", flush=True)

    print(f"[cpu-ref] running {context_len}-token forward (this is the slow part)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = text_model(input_ids=tokens.to(torch.long), use_cache=False)
    hidden = out.last_hidden_state.to(torch.float32)
    forward_s = time.time() - t0
    print(f"[cpu-ref] done in {forward_s:.0f}s; hidden={tuple(hidden.shape)}", flush=True)

    payload = {
        "fingerprint": _fingerprint(tokens, model_path, context_len),
        "tokens": tokens.cpu(),
        "context_len": int(context_len),
        "chunk": int(chunk),
        "prompt_len": int(prompt_len),
        "hidden": hidden.cpu(),
        "forward_seconds": forward_s,
        "attn_implementation": attn,
    }
    torch.save(payload, out_path)
    print(f"[cpu-ref] wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)
    return out_path


if __name__ == "__main__" and os.environ.get("GEMMA4_CPU_REF_CONTEXT"):
    generate_long(context_len=int(os.environ["GEMMA4_CPU_REF_CONTEXT"]))
