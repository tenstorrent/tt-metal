# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the model ACTUALLY put on the chip, measured from the built model rather than inferred.

Every byte figure the roofline has ever used was a prediction about the model:

    params x 1.0 B/param      a placeholder -- 1 byte per parameter regardless of dtype. On a bf16
                              model that halves the bytes and DOUBLES the ceiling: voxtral published
                              141.8 tok/s/u against a true 54.7.
    checkpoint weight_bytes   the file on disk. An UPPER bound, and only equal to what is streamed
                              when the model is served at checkpoint precision. gemma-3-12b's 24.37 GB
                              bf16 checkpoint implies a 21.0 tok/s/u ceiling for a model that MEASURES
                              30.8 -- a ceiling it has already passed, which cannot be a ceiling.
    profile op bytes          a whole profiling window summed: prefill plus many decode steps, so it
                              reads ~89 GB for one gemma-3 token and yields 5.8 tok/s/u.

None of the three can be right for every model, because each predicts the served width instead of
observing it, and the served width is a per-model decision the checkpoint does not record.

This observes it. Given the built model it walks the tensors that are on the device and records each
one's element count and its REAL dtype, which is the only place that fact exists -- after the loader
has decided it and before anything has to guess. perf_target.active_bytes already sums exactly this
shape (`weight_tensors: [{numel, dtype}, ...]`) and has since it was written; nothing ever filled it
in, so every caller fell through to one of the three predictions above.

Mixed precision comes out right for free: each tensor carries its own dtype, so a model serving
attention at bf8 and MLP at bf4 is summed as it actually is rather than as one blended guess.

WHAT THIS DOES NOT ANSWER. A multimodal model loads a vision tower a decode token never reads. The
census counts what is RESIDENT, so scoping it to the tensors a given stage touches is a separate
question -- one the pipeline can answer and this module deliberately does not guess at. `scope`
records which subtree was walked so a caller can tell a whole-model census from a decoder-only one
instead of assuming.
"""

from __future__ import annotations

# ttnn dtype -> bytes per element. bfloat8_b and bfloat4_b are BLOCK formats: a shared exponent per
# 16-element tile adds 1/16 of a byte to each element, which is why they are 1.0625 and 0.5625 rather
# than 1 and 0.5. Getting that wrong understates a bf8 model's bytes by 6%.
_DTYPE_BYTES = {
    "bfloat16": 2.0,
    "float32": 4.0,
    "float16": 2.0,
    "bfloat8_b": 1.0625,
    "bfloat4_b": 0.5625,
    "uint32": 4.0,
    "int32": 4.0,
    "uint16": 2.0,
    "uint8": 1.0,
    # INDEX AND ID TENSORS ARE RESIDENT TOO. gemma-3 keeps a 1024-element int64 (token ids or
    # positions) on device: 8 KB, irrelevant to a 12 GB total -- but an UNKNOWN dtype forces
    # complete=False, and an incomplete census is refused whole. So one 8 KB tensor with no entry
    # here was discarding the entire measurement and sending the ceiling back to params x 1.0.
    # Knowing a width is not the same as it mattering; these are here so the refusal is reserved for
    # dtypes nobody has accounted for.
    "int64": 8.0,
    "float64": 8.0,
    "int16": 2.0,
    "int8": 1.0,
    "uint64": 8.0,
    "bfloat4": 0.5625,
    "bfloat8": 1.0625,
}


def dtype_name(dt) -> str:
    """A ttnn/torch dtype as the plain name active_bytes keys on. "" when unrecognisable.

    ttnn dtypes stringify as `DataType.BFLOAT8_B`, torch's as `torch.bfloat16`; both reduce to the
    last dotted part, lowercased. An empty string is returned rather than a default, because a dtype
    guessed here would silently become a byte width nobody chose."""
    if dt is None:
        return ""
    name = getattr(dt, "name", None) or str(dt)
    return str(name).rsplit(".", 1)[-1].strip().lower()


def bytes_per_elem(dt) -> float:
    """Bytes per element for a dtype name, or 0.0 when unknown.

    0.0, never a default: an unknown dtype must make the census INCOMPLETE (and say so) rather than
    contribute a plausible number. A byte total that silently absorbed a guess is the thing this
    module exists to replace."""
    return _DTYPE_BYTES.get(dtype_name(dt), 0.0)


def _tensor_entry(t):
    """(numel, dtype_name) for one device tensor, or None if it is not one."""
    try:
        shape = getattr(t, "shape", None) or getattr(t, "padded_shape", None)
        dt = getattr(t, "dtype", None) or getattr(t, "get_dtype", lambda: None)()
        if shape is None or dt is None:
            return None
        n = 1
        for d in tuple(shape):
            n *= int(d)
        if n <= 0:
            return None
        name = dtype_name(dt)
        return (n, name) if name else None
    except Exception:  # noqa: BLE001 -- a non-tensor attribute must not end the walk
        return None


# A device tensor cannot live inside any of these, so walking them is pure cost. This is the whole
# fix for a walk that reached 99 tensors: on gemma-3 it exhausted a 200k node budget with 450k
# objects still queued, and 94% of what it spent was the TOKENIZER -- 116,892 strings, 64,441 tuples
# of strings and 6,419 AddedToken entries for a 131k vocabulary. It wandered into the vocabulary and
# never came back to the weights.
#
# Filtered at ENQUEUE rather than at dequeue: the previous version did skip strings, but only after
# each one had already cost a node from the budget. Skipping late is not skipping.
_NEVER_WALK = (str, bytes, bytearray, int, float, bool, complex, type(None))

# RESIDENT IS NOT THE SAME AS READ-PER-UNIT, and the difference is most of the number. gemma-3 keeps
# 96 KV caches on device -- 48 layers x K and V, each 1024x8x32x256 -- which is 6.85 GB against about
# 8.6 GB of weights: the census reported 15.49 GB for a model whose weight read set is a little over
# half that. Handing that to the ceiling would DOUBLE-COUNT, because active_bytes already prices KV
# from seq_len; the whole reason the two terms are separate is that weights are flat in the token
# count and KV is linear in it.
#
# Skipped by ATTRIBUTE NAME, not by shape. The pipeline attaches them as named attributes --
# `generator.tt_kv_cache`, `generator.page_table` (pipeline.py:206) -- so the structure already says
# what these are. Matching on shape instead would mean re-deriving max_seq x kv_heads x layers x
# head_dim per model and would silently mis-file any weight that happened to share it.
_CACHE_ATTRS = ("kv_cache", "kvcache", "page_table", "paged_cache", "cache")


def _is_cache_attr(name: str) -> bool:
    n = str(name).lower().lstrip("_")
    return any(c in n for c in _CACHE_ATTRS)


def _walkable(obj) -> bool:
    return not isinstance(obj, _NEVER_WALK)


def census(root, scope: str = "model", max_nodes: int = 2_000_000) -> dict:
    """Walk `root` and record every device tensor reachable from it.

    Returns {"weight_tensors": [{numel, dtype}], "weight_bytes": int, "scope": str,
             "unknown_dtype_tensors": int, "complete": bool}.

    `complete` is False when any tensor carried a dtype this module does not know a width for. A
    caller must not treat an incomplete census as the model's byte count -- it is a LOWER bound, and
    a lower bound on bytes is an UPPER bound on the ceiling, which is the direction that makes a run
    stop early believing it is near the wall.

    Traversal is breadth-first over attributes, lists, dicts and tuples, with an identity-keyed seen
    set: models share tensors (tied embeddings, a weight referenced from two modules) and counting
    one twice inflates the byte total exactly as much as missing one deflates it.
    """
    seen: set = set()
    tensors: list = []
    unknown = 0
    queue = [root]
    nodes = 0
    while queue and nodes < max_nodes:
        obj = queue.pop()
        nodes += 1
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        entry = _tensor_entry(obj)
        if entry is not None:
            n, name = entry
            if bytes_per_elem(name) > 0:
                tensors.append({"numel": int(n), "dtype": name})
            else:
                unknown += 1
            continue
        try:
            if isinstance(obj, dict):
                queue.extend([v for v in obj.values() if _walkable(v)])
                continue
            if isinstance(obj, (list, tuple, set)):
                queue.extend([v for v in obj if _walkable(v)])
                continue
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                queue.extend([v for k, v in d.items() if _walkable(v) and not _is_cache_attr(k)])
        except Exception:  # noqa: BLE001 -- one unwalkable node must not lose the whole census
            continue
    total = sum(t["numel"] * bytes_per_elem(t["dtype"]) for t in tensors)
    # BYTES PER PARAMETER, WHICH IS WHAT THE CEILING ACTUALLY LACKS.
    #
    # The ceiling divides by bytes-streamed-per-unit and had no way to get it, so it used a
    # placeholder of 1.0 B/param. On gemma-3 that lands within 6% (served bf8 = 1.0625) and looked
    # right; on voxtral (bf16 = 2.0) it is exactly half, publishing 141.8 tok/s/u against a true ~75.
    #
    # The TOTAL is the harder number: it counts everything resident, and on gemma-3 that is 15.49 GB
    # of which ~6.85 GB is KV cache -- scratch the ceiling must not divide by, because active_bytes
    # already prices KV from seq_len. Separating those needs a weight-vs-cache rule that holds for
    # paged KV, Mamba state and architectures nobody here has seen.
    #
    # The RATIO needs none of that. Σ(numel × width) / Σ(numel) is the average width of what is
    # resident, and it is barely moved by which tensors are cache -- the cache is stored at the same
    # widths as the weights. So it survives the very ambiguity that blocks the total, and multiplied
    # by a param count the tool already trusts it gives the real byte figure for a mixed-precision
    # model: a checkpoint served part bf8 and part bf4 comes out at neither, but at what it is.
    elems = sum(t["numel"] for t in tensors)
    per_param = (total / elems) if elems else 0.0
    return {
        "weight_tensors": tensors,
        "weight_bytes": int(round(total)),
        "scope": scope,
        "bytes_per_param": round(per_param, 6),
        "resident_elems": int(elems),
        "unknown_dtype_tensors": unknown,
        "complete": unknown == 0 and bool(tensors),
        "source": "device census (built model)",
    }


def marker(c: dict) -> str:
    """The line the harness parses back, in the same shape as every other TRACE_* marker.

    One line, not the tensor list: the list is thousands of entries and the harness only needs the
    total plus enough to judge whether to trust it."""
    return "TRACE_WEIGHT_BYTES=%d scope=%s tensors=%d unknown_dtype=%d complete=%s bytes_per_param=%.4f dtypes=%s" % (
        int(c.get("weight_bytes") or 0),
        c.get("scope") or "model",
        len(c.get("weight_tensors") or []),
        int(c.get("unknown_dtype_tensors") or 0),
        "1" if c.get("complete") else "0",
        float(c.get("bytes_per_param") or 0.0),
        # The mix itself, so a reader can see WHY the average is what it is rather than trust it.
        ",".join(
            "%s:%d" % (d, n)
            for d, n in sorted(
                __import__("collections").Counter(t["dtype"] for t in (c.get("weight_tensors") or [])).items()
            )
        )
        or "none",
    )
