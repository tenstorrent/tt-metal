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

A multimodal model loads a vision tower a decode token never reads, so the resident TOTAL is not any
one stage's read set. That split used to be left to the caller, which had only the checkpoint to
apportion it by -- disk proportions, measured at disk precision. The loader does not quantise every
tower alike, so the ratio does not transfer, and the whole error lands on the stage's ceiling: on
voxtral the language tower is 85.8% of the bf16 FILE, and nothing says it is 85.8% of the 1.72 GB
actually resident. `sections` answers it from the same walk, in TWO vocabularies:

    by CHECKPOINT SECTION   language_model, audio_tower -- each resident tensor attributed to the
                            section whose file recorded its element count. This is the one every
                            consumer asks in, because it is what stage_roots establishes.
    by ATTRIBUTE            _inner, enc_a, lm_layers -- where the walk FOUND it. The only view
                            available when there is no checkpoint to match against.

The first version had only the second, and was inert for it: 19 subtrees recorded on voxtral and not
one name in common with the two the caller wanted, so the split was measured and never read. Names
cannot bridge that -- a tower is renamed, re-nested and wrapped on its way into a TT model, and
`encode`/`prefill` exist as device attributes while being 1 MB state objects, so a name match would
price the audio stage at a megabyte. An element count survives transposition, tile-padding, sharding
and re-quantisation, which is why the join is on numels, as checkpoint_numels already is.

`scope` still records which subtree was walked.
"""

from __future__ import annotations

import glob
import json
import struct
from pathlib import Path

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


def _on_device(t) -> bool:
    """Is this tensor resident on the CHIP, or a host copy that merely looks like one?

    THE CEILING DIVIDES BY WHAT THE CHIP READS. A torch tensor has .shape and .dtype exactly as a
    ttnn one does, so "anything with a shape is a tensor" counted host memory as device memory.
    voxtral loads its weights with dtype=torch.float32 and keeps that copy alive on the HOST while
    the device holds bf16 -- so the census reported 29.96 GB for a model whose device footprint is
    about 11.3, and a width of 2.9076 B/param that no device tensor has. The fp32 half is real
    waste, but it is host RAM: the chip never streams it and it cannot slow a token down.

    Asked of the tensor rather than inferred from its dtype: a device tensor answers storage_type()
    or device(), a torch one does not. Inferring from dtype would be the same class of mistake --
    fp32 is legal on device, and bf16 is legal on the host.
    """
    # storage_type() FIRST, and its VALUE, not its existence. torch.Tensor carries a `.device`
    # attribute too -- it just says `cpu` -- so "has a device attribute" passed every host tensor and
    # the filter changed nothing: voxtral still reported 29.96 GB with 18.7 GB of host fp32 in it.
    # Presence of a name is not residency; the answer has to be read.
    st = getattr(t, "storage_type", None)
    if st is not None:
        try:
            return "DEVICE" in str(st() if callable(st) else st).upper()
        except Exception:  # noqa: BLE001
            return False
    dev = getattr(t, "device", None)
    if dev is not None:
        try:
            return "CPU" not in str(dev() if callable(dev) else dev).upper()
        except Exception:  # noqa: BLE001 -- a host tensor may define the name and raise
            return False
    return False


def _tensor_entry(t):
    """(numel, dtype_name) for one device tensor, or None if it is not one."""
    try:
        if not _on_device(t):
            return None
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
# THE CHECKPOINT DECIDES, NOT A LIST OF NAMES. This skipped any attribute whose name contained one
# of ("kv_cache", "kvcache", "page_table", "paged_cache", "cache") -- a substring match, defended on
# the grounds that the pipeline attaches caches as named attributes. It does, in the models that
# list was written against. Voxtral calls its cache `kv`, which matches none of them, and 83.9 MB of
# KV cache was counted as model weights.
#
# The test that does not care what anything is called is two lines below and always was: a tensor
# whose element count appears in the CHECKPOINT was loaded from the file, and one whose count
# appears nowhere in it was made at runtime. That is true of a KV cache, an accumulator and a
# staging buffer alike, in any naming convention, for a model nobody has written yet.
#
# It needed a checkpoint to compare against, and nothing passed one until 2026-08-19 -- so the name
# list was the only filter that ever ran. With the checkpoint wired, it is redundant; without one,
# the census now reports itself INCOMPLETE rather than quietly counting scratch as weights, and
# every consumer already refuses an incomplete census.


def _walkable(obj) -> bool:
    return not isinstance(obj, _NEVER_WALK)


_FAILED_CHECKPOINT: list = []


_AMBIGUOUS = "\x00ambiguous"


def checkpoint_section_numels(model_id_or_dir) -> dict:
    """{element count: the checkpoint section that tensor belongs to}. Empty when unreadable.

    THE JOIN THE SPLIT WAS MISSING, AND WHY IT IS NUMELS. The census names each resident subtree by
    the ATTRIBUTE it was reached through -- `enc_a`, `lm_layers`, `embed` -- because that is all the
    built model tells it. Every consumer asks by CHECKPOINT SECTION -- `audio_tower`,
    `language_model` -- because that is what stage_roots establishes and what the roofline divides
    by. Measured 2026-08-16 on voxtral: 19 subtrees recorded, and not one name in common with the
    two the caller wanted, so a correct measurement was written and never read.

    Translating names afterwards cannot work: a tower is renamed, re-nested and wrapped on its way
    into a TT model, and `encode`/`prefill` DO appear as device attributes while being 1 MB state
    objects rather than towers -- so a name match would have priced the audio stage at a megabyte.

    An element count survives all of it. The tensor is transposed, tile-padded, sharded and
    re-quantised, and its numel is still the numel the file recorded -- the same property
    checkpoint_numels already relies on to tell a weight from a runtime buffer.

    AMBIGUITY IS DROPPED, NOT GUESSED. A count appearing in two sections cannot attribute a tensor,
    and picking one would silently move bytes between towers. Those tensors land in "unmatched",
    which the caller can see and weigh, rather than in whichever section came first in the file.
    """
    seen: dict = {}
    for numel, section, _name in _checkpoint_tensor_sections(model_id_or_dir):
        prev = seen.get(numel)
        if prev is None:
            seen[numel] = section
        elif prev != section:
            seen[numel] = _AMBIGUOUS
    return {n: s for n, s in seen.items() if s != _AMBIGUOUS}


def _checkpoint_tensor_sections(model_id_or_dir):
    """(numel, top-level section) for every tensor in the checkpoint. Silent on any failure.

    The section is the first dotted component of the tensor's name, which is what declared_sections
    and stage_roots both key on -- one definition of "which tower", not a second one that can drift.
    """
    try:
        _croot = _checkpoint_glob_root(model_id_or_dir)
        pats = [str(_croot)]
        if not str(_croot).endswith(".safetensors"):
            pats = [
                str(Path(_croot) / "*.safetensors"),
                str(Path(_croot) / "snapshots" / "*" / "*.safetensors"),
            ]
        for f in sorted({f for p in pats for f in glob.glob(p)}):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                if n <= 0 or n > 200_000_000:
                    continue
                hdr = json.loads(fh.read(n))
            for name, meta in hdr.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                shape = meta.get("shape") or []
                if not shape:
                    continue
                numel = 1
                for d in shape:
                    numel *= int(d)
                if numel > 0 and "." in str(name):
                    # THE FULL NAME TOO. A consumer that must tell a lookup table from a weight
                    # matches on the name (model_bytes._LOOKUP_ONLY), and the section alone cannot
                    # say which tensors inside a tower are multiplied.
                    yield numel, str(name).split(".", 1)[0], str(name)
    except Exception as exc:  # noqa: BLE001
        _FAILED_CHECKPOINT.append("%s: %s" % (type(exc).__name__, str(exc)[:100]))
        return


def _checkpoint_glob_root(model_id_or_dir):
    """The directory to look for .safetensors in, resolving a HUB ID to its cache snapshot.

    Both readers below glob `<arg>/*.safetensors`. A hub id -- "mistralai/Voxtral-Mini-3B-2507" --
    is a relative path that does not exist, so the glob found nothing and the reader returned empty:
    not "I cannot read that", just no tensors, indistinguishable from a checkpoint with none. The
    census's whole checkpoint-name vocabulary depends on these, so an id handed in silently cost the
    per-tower split, and the ceiling fell back to apportioning by disk ratio.

    Same defect shape as _model_id_for_facts handing a Path to a Source parser: a reader that
    answers "nothing" for an argument it cannot use. Resolved here rather than at each call site, so
    a caller with either form gets the same answer.
    """
    raw = str(model_id_or_dir or "")
    if not raw:
        return raw
    if Path(raw).exists() or raw.endswith(".safetensors"):
        return raw
    # An unqualified "org/name" is a hub id; the weights are in the shared cache, never beside it.
    if raw.count("/") == 1 and not raw.startswith((".", "/", "~")):
        try:
            from .checkpoint_sections import hf_cache_dir

            snap = hf_cache_dir(raw)
            if snap:
                return str(snap)
        except Exception:  # noqa: BLE001 -- unresolvable id: glob finds nothing, exactly as before
            pass
    return raw


def checkpoint_numels(model_id_or_dir) -> set:
    """Element counts of every tensor in the model's checkpoint. Empty when it cannot be read.

    THE ARCHITECTURE-INDEPENDENT WAY TO TELL A WEIGHT FROM SCRATCH. A weight was loaded from the
    file on disk; a KV cache, an accumulator, a staging buffer is allocated at runtime and exists
    nowhere in it. That distinction needs no knowledge of attention, Mamba state or paged blocks --
    it is true of every model, including ones not written yet.

    The alternatives were tried and are worse. Matching by SHAPE encodes what a KV cache looks like,
    which is an assumption: paged KV has different dimensions and Mamba has no KV at all. Skipping by
    ATTRIBUTE NAME failed outright -- the caches are reachable from the layers as well as from
    `generator.tt_kv_cache`, so cutting one attribute left them in the walk.

    Matched on element COUNT rather than name or shape, because a tensor is transposed, tile-padded
    and sharded on its way to the chip -- its name and shape need not survive, but its size usually
    does. gemma-3's checkpoint has 1065 tensors in only 13 distinct sizes, and its KV cache size
    (67,108,864) appears in none of them.
    """
    out: set = set()
    try:
        _croot = _checkpoint_glob_root(model_id_or_dir)
        pats = [str(_croot)]
        if not str(_croot).endswith(".safetensors"):
            pats = [
                str(Path(_croot) / "*.safetensors"),
                str(Path(_croot) / "snapshots" / "*" / "*.safetensors"),
            ]
        files = sorted({f for p in pats for f in glob.glob(p)})
        for f in files:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                if n <= 0 or n > 200_000_000:
                    continue
                hdr = json.loads(fh.read(n))
            for name, meta in hdr.items():
                if name == "__metadata__" or not isinstance(meta, dict):
                    continue
                shape = meta.get("shape") or []
                if not shape:
                    continue
                numel = 1
                for d in shape:
                    numel *= int(d)
                if numel > 0:
                    out.add(numel)
    except Exception as exc:  # noqa: BLE001
        # LOUD, NOT EMPTY. Returning set() here means "no checkpoint", which the caller reads as
        # "cannot split" and every tensor is then counted as a weight -- the exact answer a crash
        # should not be able to produce. This swallowed a NameError (Path had been stripped as an
        # unused import) and reported a clean split of 0 scratch tensors on a model with 764 fp32
        # buffers.
        _FAILED_CHECKPOINT.append("%s: %s" % (type(exc).__name__, str(exc)[:100]))
        return set()
    return out


def census(root, scope: str = "model", max_nodes: int = 2_000_000, checkpoint=None) -> dict:
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
    # A tensor whose element count matches nothing in the checkpoint was made at runtime: KV cache,
    # accumulator, staging copy. Only weights belong in the width the ceiling multiplies by a PARAM
    # count -- mixing the two is averaging apples and oranges then multiplying by the apples.
    ckpt = checkpoint_numels(checkpoint) if checkpoint else set()
    # WHICH TOWER, in the vocabulary the caller asks in. The attribute names below say where a tensor
    # was FOUND; this says which checkpoint section it CAME FROM, and only the second one can be
    # looked up by a stage_roots entry. Both are recorded: the attribute view is the only one
    # available when there is no checkpoint to match against.
    _ckpt_sec = checkpoint_section_numels(checkpoint) if checkpoint else {}
    # GUARDED ON THE CHECKPOINT, NOT ON THE MAP. `_ckpt_sec` is empty both when no checkpoint was
    # read and when every size in it was ambiguous -- and those need opposite answers. Keying the
    # attribution on the map skipped it entirely in the second case, so a model whose sizes all
    # collide reported no unmatched bytes at all, which reads as "fully attributed".
    _have_ckpt = bool(ckpt)
    seen: set = set()
    tensors: list = []
    scratch: list = []
    unknown = 0
    # WHERE each tensor was reached from, carried alongside it. The walk already knows -- it arrived
    # by following a named attribute -- and threw the name away, which is the whole reason a caller
    # wanting one tower's bytes had to apportion the total by the CHECKPOINT's proportions. Those are
    # disk proportions: on a mixed-precision load the towers are not quantised alike, so the ratio
    # does not transfer, and the error lands directly on the stage's ceiling.
    #
    # A tuple of attribute names, not a full path string: only the names are ever matched against,
    # list indices carry no meaning, and a tuple of interned strings costs nothing to copy.
    sections: dict = {}
    ckpt_sections: dict = {}
    queue = [(root, ())]
    nodes = 0
    while queue and nodes < max_nodes:
        obj, path = queue.pop()
        nodes += 1
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        entry = _tensor_entry(obj)
        if entry is not None:
            n, name = entry
            if bytes_per_elem(name) > 0:
                entry = {"numel": int(n), "dtype": name}
                if not ckpt or int(n) in ckpt:
                    tensors.append(entry)
                    # CREDITED TO EVERY NAME ON THE WAY IN, so a lookup works whatever depth the
                    # tower sits at -- `language_model` reached via `.model.language_model` is
                    # credited to both, and the caller asks for the one the model declared. The
                    # groups therefore OVERLAP and must never be summed against each other; the
                    # denominator is the census total, which is counted once.
                    #
                    # path[:-1] drops the attribute the TENSOR ITSELF was bound to. A subtree is
                    # never named for one of its own weights, so `weight`/`w`/`bias` as groups are
                    # noise -- and noise that would rank near the top by bytes, displacing a real
                    # tower from the bounded marker.
                    _b = int(n) * bytes_per_elem(name)
                    for seg in set(path[:-1]):
                        if not _is_section_name(seg):
                            continue
                        sections[seg] = sections.get(seg, 0.0) + _b
                    if _have_ckpt:
                        # KEPT APART, because the two vocabularies can collide -- a checkpoint
                        # section and an attribute can both be called `layers`, and adding both
                        # rules into one key would produce a number that is neither.
                        #
                        # "unmatched" is a real answer, not a gap to hide: it is the share of the
                        # model whose tower could not be established, and a reader comparing a
                        # stage's bytes against the total deserves to know how much is unaccounted.
                        _sec = _ckpt_sec.get(int(n)) or "unmatched"
                        ckpt_sections[_sec] = ckpt_sections.get(_sec, 0.0) + _b
                else:
                    scratch.append(entry)
            else:
                unknown += 1
            continue
        try:
            if isinstance(obj, dict):
                # A str key names its value as surely as an attribute does (ModuleDict, state dicts).
                queue.extend([(v, path + (k,) if isinstance(k, str) else path) for k, v in obj.items() if _walkable(v)])
                continue
            if isinstance(obj, (list, tuple, set)):
                queue.extend([(v, path) for v in obj if _walkable(v)])
                continue
            d = getattr(obj, "__dict__", None)
            if isinstance(d, dict):
                queue.extend([(v, path + (k,)) for k, v in d.items() if _walkable(v)])
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
    scratch_bytes = sum(t["numel"] * bytes_per_elem(t["dtype"]) for t in scratch)
    per_param = (total / elems) if elems else 0.0
    return {
        "weight_tensors": tensors,
        "weight_bytes": int(round(total)),
        # CHECKPOINT SECTIONS WIN on a name collision: they are the vocabulary every consumer asks
        # in, and an attribute that happens to share the name is the weaker claim.
        "sections": {
            k: int(round(v))
            for k, v in {
                **{a: b for a, b in sections.items() if b > 0},
                **{a: b for a, b in ckpt_sections.items() if b > 0},
            }.items()
        },
        "scope": scope,
        "bytes_per_param": round(per_param, 6),
        "resident_elems": int(elems),
        "scratch_tensors": len(scratch),
        "scratch_bytes": int(round(scratch_bytes)),
        "checkpoint_matched": bool(ckpt),
        "unknown_dtype_tensors": unknown,
        # AND A CHECKPOINT TO TELL A WEIGHT FROM SCRATCH. Without one, every resident tensor is
        # counted -- KV caches, accumulators, staging buffers -- and the total is not a weight
        # figure at all. gemma-3 reported 15.49 GB that way for about 8.6 GB of weights. That is an
        # OVERCOUNT, so it reads as too LOW a ceiling, which is the direction that lets a run
        # believe it has headroom it does not have.
        #
        # Incomplete is the honest answer, and it already has consequences: perf_target refuses an
        # incomplete census outright rather than using it as a bound, and falls back to the
        # checkpoint's own byte count -- which is exactly the right answer when no checkpoint was
        # available to classify against.
        # THE DEPTH IT WAS TAKEN AT, recorded here because this is where the walked object is: the
        # stage names that spell the per-stage caps come from that model's PIPELINE_STAGES.
        "depth": census_depth(root),
        "complete": unknown == 0 and bool(tensors) and _have_ckpt,
        "source": "device census (built model)",
    }


_SECTIONS_IN_MARKER = 48


def _is_section_name(seg) -> bool:
    """Whether a path segment names a SUBTREE a stage could run, or is walk bookkeeping.

    Two kinds of segment reach here that are not towers, and both rank high by bytes -- so both
    displace real names from the bounded marker, and one of them is wrong outright:

      _parameters / _modules / _buffers -- torch's own containers. Every tensor is inside one, so
      each is credited the WHOLE model. A share taken against them is 1.0 by construction.

      0 / 1 / 2 ... -- ModuleList indices, and they are not scoped to a parent: layer 3 of the audio
      tower and layer 3 of the language backbone are credited to the SAME key "3". The number that
      comes out is a sum across towers, describing no subtree that exists.

    `weight`/`bias` are already excluded by path[:-1], which drops the tensor's own attribute.
    """
    t = str(seg or "")
    return bool(t) and not t.startswith("_") and not t.isdigit()


def sections_marker(c: dict) -> str:
    """The per-subtree bytes, as its own line. Empty string when the walk recorded none.

    SEPARATE FROM `marker` because the two are consumed by different questions -- the total feeds the
    width, these feed a stage's share -- and because an older harness parsing the census line must
    not have to cope with a field that did not exist when it was written.

    Bounded: a deep model has hundreds of distinct attribute names and the tail of that list is
    LayerNorm weights. Sorted by bytes and truncated, so what survives is every name big enough for a
    stage to be made of. A name that falls off the end could not have been a tower.
    """
    secs = c.get("sections") or {}
    if not secs:
        return ""
    top = sorted(secs.items(), key=lambda kv: -int(kv[1]))[:_SECTIONS_IN_MARKER]
    # A name with a separator in it would be unparseable on the other side; attribute names cannot
    # contain these, but a dict key reached during the walk can be anything at all.
    safe = [(k, v) for k, v in top if "," not in str(k) and ":" not in str(k) and str(k).strip()]
    if not safe:
        return ""
    return "TRACE_WEIGHT_SECTIONS=%s" % ",".join("%s:%d" % (k, int(v)) for k, v in safe)


def _model_root_of(root):
    """The model directory the object being censused came from, or None.

    Its class's module file sits inside the model's own tree, which is where PIPELINE_STAGES is --
    the same derivation _checkpoint_for_census uses to find the weights.
    """
    import sys as _sys

    try:
        mod = _sys.modules.get(type(root).__module__)
        f = getattr(mod, "__file__", None) if mod else None
        if not f:
            return None
        for par in list(Path(f).resolve().parents)[:4]:
            if (par / "tt" / "pipeline.py").is_file():
                return par
            if (par / ".git").exists():
                break
    except Exception:  # noqa: BLE001
        return None
    return None


def census_depth(root=None) -> str:
    """The layer cap this process is building at: a positive int as a string, or "all".

    A CENSUS OF A CAPPED BUILD IS NOT THE MODEL'S RESIDENT BYTES, and it looks exactly like one.
    Voxtral, run 10: the census reported 1.299 B parameters against the checkpoint's 4.676 B -- 28%
    -- because it ran inside a profiling run built at the coverage window of 2 layers. Its own
    section figures say so: lm_layers at 227.3 M is two layers of a stack whose config puts each at
    100.7 M, and a 2-layer build works out at 1.235 B against the 1.299 B measured.

    Nothing downstream could tell. device_weight_bytes is pinned by the FIRST complete census, the
    capped profiling run reaches it before the uncapped full-pipeline gate, and every ceiling in the
    report then divides a number missing three quarters of the model.

    read_depth() already owns this question -- a cap is the presence of the depth variable, "all
    layers" its absence -- so the census reports that answer rather than deriving a second one.
    "unknown" is not a claim of full depth: the reader treats it as untrustworthy, like a cap.
    """
    try:
        from .layer_depth import active_depth_caps, read_depth

        # The stage names come from the model, so the census hands over the pipeline it just walked;
        # stack_knob_repair reads PIPELINE_STAGES from its source without building anything.
        caps = active_depth_caps(model_root=_model_root_of(root))
        if caps:
            # Name the tightest cap in force, so the log says which knob shrank the build.
            k = min(caps, key=lambda x: caps[x])
            return "%s=%d" % (k, caps[k])
        d = read_depth()
    except Exception:  # noqa: BLE001
        return "unknown"
    return "all" if d is None else str(int(d))


def marker(c: dict) -> str:
    """The line the harness parses back, in the same shape as every other TRACE_* marker.

    One line, not the tensor list: the list is thousands of entries and the harness only needs the
    total plus enough to judge whether to trust it."""
    return (
        "TRACE_WEIGHT_BYTES=%d scope=%s tensors=%d unknown_dtype=%d complete=%s bytes_per_param=%.4f "
        "depth=%s dtypes=%s"
    ) % (
        int(c.get("weight_bytes") or 0),
        c.get("scope") or "model",
        len(c.get("weight_tensors") or []),
        int(c.get("unknown_dtype_tensors") or 0),
        "1" if c.get("complete") else "0",
        float(c.get("bytes_per_param") or 0.0),
        # THE DEPTH IT WAS TAKEN AT. A census of a capped build is not the model's resident bytes and
        # looks exactly like one; see census_depth().
        str(c.get("depth") or census_depth()),
        # The mix itself, so a reader can see WHY the average is what it is rather than trust it.
        ",".join(
            "%s:%d" % (d, n)
            for d, n in sorted(
                __import__("collections").Counter(t["dtype"] for t in (c.get("weight_tensors") or [])).items()
            )
        )
        or "none",
    )
