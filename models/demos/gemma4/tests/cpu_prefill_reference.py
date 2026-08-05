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
import time
from pathlib import Path

import torch

REFERENCE_CHUNK = 4096

# Long-context references. A full 256k reference is not reachable on CPU: the
# forward is ~51 h by extrapolation from the measured 4k run, and eager attention
# would need an ~8.8 TB score matrix per layer. 32768 is the largest of the device
# targets that fits (~1 h, ~137 GB transient), so that is where the numerical
# evidence for the ring path tops out.
LONG_REFERENCE_CONTEXTS = [8192, 16384, 32768]
FILLER_SEED = 1234

# Bumped when the dump's contents or semantics change, so an old file on disk is
# rejected instead of being compared against.
_FORMAT_VERSION = 1


def _model_path() -> str:
    return os.getenv("HF_MODEL") or os.getenv("GEMMA4_MODEL_PATH", "google/gemma-4-31B-it")


def build_token_sequence(model_path, chunk, context_len, vocab_size):
    """The exact token sequence a chunked prefill run consumes.

    Chunk 0 carries the real prompt; later chunks are deterministic filler. Shared
    by the reference generator and the device test so the two cannot drift — a PCC
    comparison against a differently-tokenized reference would be worse than none.
    """
    import torch as _torch

    from models.demos.gemma4.demo.text_demo_prefill import _prompt_tokens

    tokens_first, tokenizer, prompt_len = _prompt_tokens(model_path, chunk)
    parts = [tokens_first]
    _torch.manual_seed(FILLER_SEED)
    for _ in range(context_len // chunk - 1):
        parts.append(_torch.randint(0, vocab_size, (1, chunk), dtype=_torch.int32))
    return _torch.cat(parts, dim=-1), tokenizer, prompt_len


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

    Cost grows quadratically in the full-attention layers: ~6 min at 8k, ~18 min at
    16k, ~1 h at 32k, and out of reach beyond that (256k extrapolates to ~51 h with
    an ~8.8 TB eager attention matrix per layer).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    model_path = model_path or _model_path()
    out_path = long_reference_path(model_path, context_len)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    text_config._attn_implementation = "eager"
    config._attn_implementation = "eager"

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
    }
    torch.save(payload, out_path)
    print(f"[cpu-ref] wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)
    return out_path


if __name__ == "__main__" and os.environ.get("GEMMA4_CPU_REF_CONTEXT"):
    generate_long(context_len=int(os.environ["GEMMA4_CPU_REF_CONTEXT"]))
