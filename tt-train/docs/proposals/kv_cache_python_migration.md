# KvCache: move Python models off the C++ binding

## Problem

`ttml.models.KvCache` is a C++ class (`tt-train/sources/ttml/models/common/transformer_common.{hpp,cpp}`) bound via nanobind. Its entire job on the device is two `ttnn::experimental::slice_write` calls plus a `cache_position` counter — every primitive it needs is already available to Python via `ttnn.*`.

Living in C++ costs us two things. First, iteration friction: a signature or behavior change means hpp + cpp + nb_models.cpp + rebuild. Second, Qwen3 needed things the C++ class doesn't provide (lazy init once `(B, H, D)` are known, autograd-tensor I/O at the call site, prefill-vs-decode return convention, causal-mask slicing) and wrapped the C++ class in Python to get them (`tt-train/sources/examples/qwen3/utils/kv_cache.py`). Llama calls the C++ class directly (`tt-train/sources/ttml/ttml/models/llama/gqattn.py:90-109`) and pays for the missing ergonomics at the call site.

C++ models still consume the C++ KvCache (`models/llama.cpp`, `modules/grouped_query_attention.cpp`, `modules/llama_block.cpp`), so a hard replacement isn't on the table yet.

## Desired end state

A Python `ttml.models.kv_cache` used by Python models (Qwen3, Llama). The C++ class stays in place until C++ models migrate off it. Expected design moves (not a spec — just the direction the Qwen3 wrapper already validated):

- Autograd tensors in and out. No `.get_value()` at every attention call site.
- Lazy allocation on first `update` so construction doesn't need `(B, num_groups, head_dim)` up front.
- `update` infers `new_tokens` from the K tensor's seq dim rather than taking it as a separate arg.
- Prefill returns the freshly-written K/V; decode returns the full cached K/V. Same convention Qwen3's wrapper uses today.
- Mask helper stays a free function, **not** a method on the cache. Putting it on the cache couples attention and training paths to the cache's lifetime, and the Llama style ("caller owns the mask") is the right default.

## Open design calls for the follow-up

- Whether to reuse `ttml.models.KvCacheConfig` (C++ dataclass-ish) or take a flat Python constructor.
- Whether the Python cache holds raw `ttnn.Tensor` or `ttml.autograd.Tensor` internally. Autograd-tensor is friendlier at call sites; raw ttnn is closer to what `slice_write` expects and matches the C++ class.
- Whether to accept the `new_tokens != K.shape[-2]` case at all. The C++ class supports it (writing a prefix of K into the cache); nothing in Qwen3 or Llama uses that capability today. Dropping it would simplify `update`.

## Consumers to update when this lands

Python side: `ttml/models/llama/gqattn.py:90-109`, plus the new `ttml/models/qwen3/attention.py`, plus the Python generation/inference paths that allocate a cache today (`examples/qwen3/generate.py`, Llama's callers).

C++ side: unchanged in this work.

## Constraint worth naming

The follow-up isn't a performance win — every meaningful op still calls into ttnn. The case for doing it is iteration velocity and getting the Qwen3 wrapper's improvements into the default API. If that's not a pain point anyone's feeling when this comes up, it's fine to defer.
