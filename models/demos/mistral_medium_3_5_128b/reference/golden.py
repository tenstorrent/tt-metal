# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Golden-trace access and the package-local reference cache.

Two separate things live here, and it matters that they stay separate:

1. :class:`GoldenTrace` — a **read-only** loader for the prepared CPU golden trace under
   ``<weights>/golden/synthetic_10240``. That trace is the ground truth acceptance is measured
   against. It is a preparation artefact: never written, never regenerated from here.

2. :class:`GoldenCacheKey` / :func:`save_golden` / :func:`load_golden` — a small cache for
   references *we* compute (random-weight decoder goldens, whole-model stacks), so an 88-layer CPU
   run is paid once rather than once per test.

The shared helper in ``deepseek_v3_d_p/utils/transformer_helpers.py`` was not reusable: its key
carries ``n_routed_experts`` and its payload is ``ref_kvpe_list``, an MLA latent. This is a dense
GQA model with two caches and no experts, so the cache is re-authored here — but it carries the
same two rules that make such a cache safe:

* **The key is frozen over every field that can change the output.** Shapes, dtype, rope
  parameters, weight source, input source, layer count. Add a knob that moves the numbers and it
  goes in the key, or a stale entry silently becomes the "reference".
* **A miss asserts; it does not recompute.** :func:`load_golden` raises. Only an explicit
  regeneration entry point writes. A cache that quietly recomputes on a key change turns a
  mismatched reference into a passing test.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from models.demos.mistral_medium_3_5_128b.reference.model_config import MistralMediumConfig
from models.demos.mistral_medium_3_5_128b.reference.modeling import (
    REF_DTYPE,
    LayerWeights,
    MistralYarnRotaryEmbedding,
    build_layer,
    causal_mask,
    hf_to_meta,
)

DEFAULT_GOLDEN_TRACE = Path("/mnt/models/mistralai/Mistral-Medium-3.5-128B/golden/synthetic_10240")
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / ".golden_cache"


# ---------------------------------------------------------------------------------------------
# The prepared CPU golden trace (read-only)
# ---------------------------------------------------------------------------------------------
class GoldenTrace:
    """The prepared per-layer KV trace: ``metadata.json`` + ``kv_cache/layer_N.safetensors``.

    Per its own metadata the cached K is **post-RoPE in HF half-split layout** and V is **raw**.
    :meth:`layer_kv` hands back exactly what is on disk; :meth:`layer_kv_meta` applies the
    half-split -> interleaved permutation for comparison against the device cache. Callers should
    say which one they want rather than permuting ad hoc — getting this backwards produces a
    plausible-looking PCC around 0.3 that is easy to misread as a rope bug.
    """

    def __init__(self, path: Path | str = DEFAULT_GOLDEN_TRACE):
        self.path = Path(path)
        meta_file = self.path / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"golden trace metadata not found at {meta_file}")
        self.metadata = json.loads(meta_file.read_text())

        self.n_tokens: int = self.metadata["n_tokens"]
        self.num_layers: int = self.metadata["num_layers"]
        self.num_kv_heads: int = self.metadata["num_kv_heads"]
        self.head_dim: int = self.metadata["head_dim"]
        self.k_layout: str = self.metadata["k_layout"]
        self.k_is_post_rope: bool = self.metadata["k_is_post_rope"]
        self.v_is_raw: bool = self.metadata["v_is_raw"]
        self.attention_scaling: float = self.metadata["rope"]["attention_scaling"]
        self.next_token_id: int = self.metadata["next_token_id"]

        assert self.k_layout == "hf_half_split", f"unexpected golden k_layout {self.k_layout!r}"
        assert self.k_is_post_rope and self.v_is_raw, "reference assumes post-rope K and raw V"
        assert not self.metadata.get("reduced_depth", False), "golden trace is reduced-depth; acceptance needs full"

    #: Environment variables naming the trace directory, in precedence order. ``PREFILL_TRACE_DIR``
    #: is the acceptance interface's name (``ACCEPTANCE.md``); ``PREFILL_GOLDEN_TRACE`` is this
    #: package's own and wins, mirroring :data:`~.checkpoint.CheckpointLoader.PATH_ENV_VARS`.
    PATH_ENV_VARS = ("PREFILL_GOLDEN_TRACE", "PREFILL_TRACE_DIR")

    @classmethod
    def from_env(cls) -> "GoldenTrace":
        """Read the first of :data:`PATH_ENV_VARS` that is set, else the default."""
        path = next((os.environ[v] for v in cls.PATH_ENV_VARS if os.environ.get(v)), None)
        return cls(path or DEFAULT_GOLDEN_TRACE)

    def token_ids(self, n: int | None = None) -> torch.Tensor:
        """``[1, n]`` int64 prompt tokens (the first ``n``, default all)."""
        ids = self.metadata["token_ids"]
        return torch.tensor(ids[: n or len(ids)], dtype=torch.int64)[None]

    def layer_kv(self, layer_idx: int, n_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(k, v)`` as stored: ``[1, num_kv_heads, n_tokens, head_dim]``, K in **HF half-split**."""
        from safetensors import safe_open

        f = self.path / "kv_cache" / f"layer_{layer_idx}.safetensors"
        with safe_open(f, framework="pt") as h:
            k = h.get_tensor(f"key_cache_layer_{layer_idx}")
            v = h.get_tensor(f"value_cache_layer_{layer_idx}")
        n = n_tokens or self.n_tokens
        return k[:, :, :n, :], v[:, :, :n, :]

    def layer_kv_meta(self, layer_idx: int, n_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(k, v)`` with K permuted HF half-split -> Meta interleaved, i.e. the device's layout.

        V is unrotated, so it is layout-independent and passes through untouched.
        """
        k, v = self.layer_kv(layer_idx, n_tokens)
        return hf_to_meta(k), v


# ---------------------------------------------------------------------------------------------
# Package-local reference cache
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GoldenCacheKey:
    """Every field that can change the cached numbers. Frozen, and hashed in sorted-key order."""

    kind: str  # "decoder_layer" | "model"
    model: str
    weight_source: str  # "random:<seed>" | "checkpoint"
    input_source: str  # "golden_tokens" | "randn:<seed>"
    n_tokens: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_signature: str
    dtype: str
    k_layout: str = "hf_half_split"

    @classmethod
    def for_config(cls, cfg: MistralMediumConfig, **fields) -> "GoldenCacheKey":
        """Build a key from a config, so no caller has to restate the shape fields (and forget one)."""
        rope = (
            f"{cfg.rope_type}:theta={cfg.rope_theta}:factor={cfg.rope_factor}"
            f":fast={cfg.rope_beta_fast}:slow={cfg.rope_beta_slow}"
            f":orig={cfg.rope_original_max_position_embeddings}:maxpos={cfg.max_position_embeddings}"
            f":attn_scale={cfg.attention_scaling:.17g}"
        )
        defaults = dict(
            model="mistral_medium_3_5_128b",
            num_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_signature=rope,
            dtype=str(REF_DTYPE).replace("torch.", ""),
        )
        defaults.update(fields)
        return cls(**defaults)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def filename(self) -> str:
        return f"{self.kind}-{self.num_layers}L-{self.n_tokens}t-{self.digest()}.safetensors"


def cache_dir() -> Path:
    return Path(os.getenv("MISTRAL_GOLDEN_CACHE") or DEFAULT_CACHE_DIR)


def save_golden(key: GoldenCacheKey, tensors: dict[str, torch.Tensor]) -> Path:
    """Write a computed reference under ``key``. The key is stored alongside as safetensors metadata
    so a cache file is self-describing and a digest collision cannot pass unnoticed."""
    from safetensors.torch import save_file

    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    out = d / key.filename()
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(out),
        metadata={"golden_cache_key": json.dumps(asdict(key), sort_keys=True)},
    )
    return out


def load_golden(key: GoldenCacheKey) -> dict[str, torch.Tensor]:
    """Load the reference for ``key``, or **raise**.

    Deliberately not a get-or-compute: on a key change the right outcome is a loud miss naming the
    regeneration command, not a silent recompute that could be measuring something else.
    """
    from safetensors import safe_open

    path = cache_dir() / key.filename()
    if not path.exists():
        raise FileNotFoundError(
            f"no cached reference for {key.kind} key {key.digest()} at {path}\n"
            f"key = {json.dumps(asdict(key), sort_keys=True, indent=2)}\n"
            f"regenerate with: python -m models.demos.mistral_medium_3_5_128b.reference.golden --regenerate"
        )
    with safe_open(path, framework="pt") as h:
        stored = json.loads(h.metadata()["golden_cache_key"])
        if stored != asdict(key):
            raise ValueError(f"cache digest collision at {path}: stored key {stored} != requested {asdict(key)}")
        return {k: h.get_tensor(k) for k in h.keys()}


# ---------------------------------------------------------------------------------------------
# Reference drivers
# ---------------------------------------------------------------------------------------------
def run_reference_layer(
    cfg: MistralMediumConfig,
    weights: LayerWeights,
    hidden_states: torch.Tensor,
    *,
    position_offset: int = 0,
    past_k: torch.Tensor | None = None,
    past_v: torch.Tensor | None = None,
    dtype: torch.dtype = REF_DTYPE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One decoder layer on ``[b, s, hidden]``, returning ``(out, k_post_rope, v_raw)``.

    ``position_offset`` is the absolute position of this chunk's first token; with ``past_k`` it
    makes a chunked call numerically identical to the matching slice of a one-shot call, which is
    the property P2 has to demonstrate on device.
    """
    b, s, _ = hidden_states.shape
    past_len = 0 if past_k is None else past_k.shape[2]
    positions = torch.arange(position_offset, position_offset + s, dtype=torch.int64)[None].expand(b, -1)
    cos, sin = MistralYarnRotaryEmbedding(cfg, dtype)(positions)
    mask = causal_mask(s, past_len + s, dtype=dtype)
    layer = build_layer(cfg, weights, dtype)
    with torch.no_grad():
        return layer(hidden_states.to(dtype), cos, sin, mask, past_k, past_v)


def run_reference_model(
    cfg: MistralMediumConfig,
    weights: "ModelWeights",
    input_ids: torch.Tensor,
    *,
    chunk_size: int | None = None,
    want_logits: bool = True,
    dtype: torch.dtype = REF_DTYPE,
) -> dict[str, torch.Tensor]:
    """The whole model over ``input_ids`` ``[b, s]``, one-shot or chunked.

    ``chunk_size`` None runs the sequence in a single call; an int runs it in chunks, carrying each
    layer's K/V forward, and concatenates the per-chunk results. Both return the same keys, so a
    test can assert the two agree — that equivalence is the host-side statement of what P2 has to
    show on device.

    Returns:
        ``{"logits": [b, s, vocab] (only if want_logits), "k_<i>": ..., "v_<i>": ...}`` with K/V
        ``[b, num_kv_heads, s, head_dim]`` covering the whole sequence, K post-RoPE in HF
        half-split layout — the golden trace's own convention, so these are directly comparable to
        :meth:`GoldenTrace.layer_kv`.
    """
    from models.demos.mistral_medium_3_5_128b.reference.modeling import build_model

    model = build_model(cfg, weights, dtype)
    s = input_ids.shape[1]
    step = chunk_size or s
    assert s % step == 0, f"sequence {s} is not a whole number of {step}-token chunks"

    logits, past, per_chunk = [], None, []
    with torch.no_grad():
        for start in range(0, s, step):
            lg, kv = model(
                input_ids[:, start : start + step],
                position_offset=start,
                past=past,
                want_logits=want_logits,
            )
            if want_logits:
                logits.append(lg)
            per_chunk.append(kv)
            past = (
                kv
                if past is None
                else [(torch.cat([pk, k], 2), torch.cat([pv, v], 2)) for (pk, pv), (k, v) in zip(past, kv)]
            )

    out: dict[str, torch.Tensor] = {}
    if want_logits:
        out["logits"] = torch.cat(logits, dim=1)
    for i in range(cfg.num_hidden_layers):
        out[f"k_{i}"] = torch.cat([c[i][0] for c in per_chunk], dim=2)
        out[f"v_{i}"] = torch.cat([c[i][1] for c in per_chunk], dim=2)
    return out


def _regenerate_cli() -> None:  # pragma: no cover - operator entry point
    """``python -m ...reference.golden --regenerate`` — recompute the cached references.

    Kept as an explicit command so recomputation is always a deliberate act; see :func:`load_golden`.
    """
    import argparse

    from models.demos.mistral_medium_3_5_128b.reference.regenerate import regenerate_all

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regenerate", action="store_true", required=True)
    p.parse_args()
    for path in regenerate_all():
        print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    _regenerate_cli()
