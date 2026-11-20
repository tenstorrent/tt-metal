# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vision-path timing and host-side speedups for the image request (r16 VISION-FAST). Host-only: no ttnn import, so the
logic here is tested without a device (tests/test_vision_fast_host.py). Device work stays in vision/model.py, tt/model.py
and tt/qwen36_vllm.py, which call these helpers.

Every flag defaults to today's behaviour (unset = the same host and device calls as before this module existed):

  QWEN36_VISION_TIMING=1        one ``[vision_timing]`` log line per image request: ms per stage, device stages
                                synchronised before the clock is read. Adds device waits, changes no result.
  QWEN36_VISION_SPLICE_SEG=1    a splice buffer pair per prefill trace length (128..2048 and the chunk size) next to the
                                8192-row one, so an image segment uploads its own bucket only. Read at start-up (the
                                buffers are allocated before trace capture); results unchanged.
  QWEN36_VISION_GRID_CACHE=N    LRU of N grids: position-embedding interpolation and the rotary cos / sin tables (as
                                tilized host tensors) are built once per (grid, bucket). Results unchanged.
  QWEN36_VISION_RESIDENT_GRIDS  "1x44x44;1x42x46": rotary tables of these grids are uploaded once in warmup_vision
                                (before trace capture) and stay in DRAM. Results unchanged.
  QWEN36_VISION_BUCKET_EXACT=1  bucket = ceil(patches / 2048) * 2048 (an exact multiple stays in its bucket). Changes the
                                tower output of exact-multiple images (fewer padding rows are attended to).
  QWEN36_VISION_BUCKET_1024=1   adds a 1,024-row bucket. Changes the output of images that land in it; needs its own
                                warm-up grid (added automatically by warmup_vision).
  QWEN36_VISION_PIPELINE=1      several images in one request: every image is dispatched before the first is read back,
                                so the host pre-work of image k+1 overlaps the device work of image k. Results unchanged.
  QWEN36_VISION_PATCH_LINEAR=1  patch embedding as one F.linear instead of the reference Conv3d. Float32 rounding may
                                differ in the last bit before the bf16 cast.
  QWEN36_VISION_LEAN=1 | list   per-call overheads, all result-preserving: "rotpad" (rotary tables padded 72 -> 96 on
                                the host once instead of 54 device pads per image), "hostrows" (plain multimodal path:
                                no upload + read-back of the tower rows), "maskskip" (no zero-mask upload when the mask
                                is already zero), "hostref" (the splice's host tensors stay alive until the next call).
"""

import collections
import os
import time

import torch

_LEAN_ALL = ("rotpad", "hostrows", "maskskip", "hostref")


def _on(name):
    return os.environ.get(name, "0") == "1"


def timing_enabled():
    return _on("QWEN36_VISION_TIMING")


def splice_seg_enabled():
    return _on("QWEN36_VISION_SPLICE_SEG")


def bucket_exact_enabled():
    return _on("QWEN36_VISION_BUCKET_EXACT")


def bucket_1024_enabled():
    return _on("QWEN36_VISION_BUCKET_1024")


def bucket_flags_on():
    return bucket_exact_enabled() or bucket_1024_enabled()


def pipeline_enabled():
    return _on("QWEN36_VISION_PIPELINE")


def patch_linear_enabled():
    return _on("QWEN36_VISION_PATCH_LINEAR")


def grid_cache_cap():
    return max(0, int(os.environ.get("QWEN36_VISION_GRID_CACHE", "0") or 0))


def lean(name):
    """QWEN36_VISION_LEAN: "1" / "all" = every item, or a comma list of rotpad, hostrows, maskskip, hostref."""
    assert name in _LEAN_ALL, name
    v = os.environ.get("QWEN36_VISION_LEAN", "0").strip()
    if v in ("", "0"):
        return False
    if v in ("1", "all"):
        return True
    items = {x.strip() for x in v.split(",") if x.strip()}
    unknown = items - set(_LEAN_ALL)
    if unknown:
        raise ValueError(f"QWEN36_VISION_LEAN: unknown item(s) {sorted(unknown)}; known: {_LEAN_ALL}")
    return name in items


def parse_grids(spec):
    """"1x32x32;1x48x64" -> [(1, 32, 32), (1, 48, 64)]."""
    return [tuple(int(x) for x in g.split("x")) for g in (spec or "").split(";") if g.strip()]


def flags_summary():
    """The vision flags that are set, for the timing line and the start-up log."""
    names = (
        "QWEN36_VISION_TIMING",
        "QWEN36_VISION_SPLICE_SEG",
        "QWEN36_VISION_GRID_CACHE",
        "QWEN36_VISION_RESIDENT_GRIDS",
        "QWEN36_VISION_BUCKET_EXACT",
        "QWEN36_VISION_BUCKET_1024",
        "QWEN36_VISION_PIPELINE",
        "QWEN36_VISION_PATCH_LINEAR",
        "QWEN36_VISION_LEAN",
    )
    out = [f"{n[len('QWEN36_VISION_'):]}={os.environ[n]}" for n in names if os.environ.get(n, "0") not in ("", "0")]
    return ",".join(out).replace(" ", "") or "none"


# --------------------------------------------------------------------------------------------------------------------
# Buckets
# --------------------------------------------------------------------------------------------------------------------
BUCKET_STEP = 2048


def vision_bucket(num_patches, exact=None, b1024=None):
    """Padded patch rows the tower runs on. Default (both flags off): (n // 2048 + 1) * 2048, the rule of the release.
    ``exact``: ceil(n / 2048) * 2048. ``b1024``: counts up to 1,023 (1,024 with ``exact``) run on 1,024 rows.
    The tower needs a multiple of 1,024 rows up to 2,048 and a multiple of 2,048 above (vision_mlp.py reshape to
    1,024-row batches, vision_attention.py QKV split), so 1,024 is the only bucket that can be added below 2,048."""
    n = int(num_patches)
    exact = bucket_exact_enabled() if exact is None else exact
    b1024 = bucket_1024_enabled() if b1024 is None else b1024
    if b1024 and n <= (1024 if exact else 1023):
        return 1024
    if exact:
        return max(1, -(-n // BUCKET_STEP)) * BUCKET_STEP
    return (n // BUCKET_STEP + 1) * BUCKET_STEP


def grid_for_bucket(bucket, exact=None, b1024=None):
    """A (1, h, w) grid (h, w even) whose patch count lands in ``bucket`` under the given rule, close to its top."""
    w = 2 * (int((bucket - 1) ** 0.5) // 2)
    h = 2 * ((bucket - 1) // w // 2)
    assert vision_bucket(h * w, exact, b1024) == bucket, (bucket, h, w)
    return (1, h, w)


def warm_grids(grids, exact=None, b1024=None):
    """Warm-up grids under a changed bucket rule. The buckets the given grids warm under the RELEASE rule stay warmed
    (a grid that now lands in a lower bucket would leave its old bucket uncompiled), and ``b1024`` adds bucket 1,024.
    Returns the grids plus one generated grid per bucket that is no longer covered. Flags off: the list unchanged."""
    exact = bucket_exact_enabled() if exact is None else exact
    b1024 = bucket_1024_enabled() if b1024 is None else b1024
    grids = [tuple(int(x) for x in g) for g in grids]
    if not (exact or b1024) or not grids:
        return grids
    want = {vision_bucket(t * h * w, False, False) for t, h, w in grids}
    if b1024:
        want.add(1024)
    have = {vision_bucket(t * h * w, exact, b1024) for t, h, w in grids}
    out = list(grids)
    for b in sorted(want - have):
        out.append(grid_for_bucket(b, exact, b1024))
    return out


def warmed_buckets(grids, exact=None, b1024=None):
    return sorted({vision_bucket(t * h * w, exact, b1024) for t, h, w in grids})


# --------------------------------------------------------------------------------------------------------------------
# Per-grid host constants
# --------------------------------------------------------------------------------------------------------------------
class LRU:
    """Small LRU with hit / miss counters. cap <= 0: nothing is kept."""

    def __init__(self, cap):
        self.cap = int(cap)
        self._d = collections.OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self._d)

    def get(self, key):
        v = self._d.get(key)
        if v is None:
            self.misses += 1
            return None
        self._d.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key, value):
        if self.cap <= 0:
            return
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.cap:
            self._d.popitem(last=False)


def grid_key(grid_thw, seq_len, rot_width):
    """Cache key of the per-grid constants: the grid, the padded rows (the tables are padded to them) and the table
    width (72, or 96 with the host pad)."""
    t, h, w = (int(x) for x in torch.as_tensor(grid_thw).reshape(-1)[:3])
    return (t, h, w, int(seq_len), int(rot_width))


def pad_rot_tables(cos, sin, unpadded_seq_len, seq_len, width=None):
    """[n, d] cos / sin -> [1, 1, seq_len, width or d]: rows past n and columns past d are the identity rotation
    (cos 1, sin 0). The row pad is the release's; the column pad is what vision_attention pads on device per block
    (ttnn.pad to padded_head_dim with 1.0 / 0.0), done once here."""
    d = int(cos.shape[-1])
    extra = 0 if width is None else int(width) - d
    assert extra >= 0, (width, d)
    rows = int(seq_len) - int(unpadded_seq_len)
    assert rows >= 0, (seq_len, unpadded_seq_len)
    cos = torch.nn.functional.pad(cos, (0, extra, 0, rows), value=1).unsqueeze(0).unsqueeze(0)
    sin = torch.nn.functional.pad(sin, (0, extra, 0, rows), value=0).unsqueeze(0).unsqueeze(0)
    return cos, sin


def patch_embed_linear(patch_embed, pixel_values):
    """The reference patch embedding (Conv3d with kernel == stride over [N, C, T, P, P] views of the rows) as one
    matmul over the flat rows. Same weights and bias; the summation order inside the kernel may differ."""
    proj = patch_embed.proj
    w = proj.weight
    x = pixel_values.to(dtype=w.dtype)
    return torch.nn.functional.linear(x.reshape(x.shape[0], -1), w.reshape(w.shape[0], -1), proj.bias)


# --------------------------------------------------------------------------------------------------------------------
# Splice buffers
# --------------------------------------------------------------------------------------------------------------------
def splice_lengths(buckets, chunk_size, largest):
    """Trace lengths that get their own splice buffer pair: the short-trace buckets and the chunk size, below the
    size of the release's single buffer (``largest``), which keeps serving every other length."""
    cand = {int(b) for b in buckets}
    if chunk_size:
        cand.add(int(chunk_size))
    return sorted(x for x in cand if 0 < x < int(largest))


class SpliceSet:
    """Book-keeping of the vision splice buffer pairs. Key None = the release's single pair; an int key = the pair of
    that trace length (QWEN36_VISION_SPLICE_SEG). ``pick`` is used by BOTH the trace body (_apply_vision_merge) and the
    staging (_set_vision_merge), so a replay always reads the pair that was staged for it. ``dirty`` = keys whose
    device mask may be non-zero."""

    def __init__(self):
        self.pairs = {}
        self.dirty = set()

    def add(self, key, vis, mask, zero_host):
        assert key not in self.pairs, key
        self.pairs[key] = (vis, mask, zero_host)

    def pick(self, length):
        return length if (length is not None and length in self.pairs) else None

    def get(self, key):
        return self.pairs[key]

    def staged(self, key):
        self.dirty.add(key)

    def to_clear(self, key, skip_clean=False):
        """Keys whose mask a clear aimed at ``key`` must zero: every dirty one, plus ``key`` itself unless
        ``skip_clean`` and it is known zero."""
        keys = [k for k in self.pairs if k in self.dirty and k != key]
        if not (skip_clean and key not in self.dirty):
            keys.append(key)
        return keys

    def cleared(self, keys):
        self.dirty.difference_update(keys)


# --------------------------------------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------------------------------------
_IMAGE_KEYS = (
    "host_rot_ms",
    "host_patch_ms",
    "host_pos_ms",
    "up_rot_ms",
    "up_rot_bytes",
    "up_patch_ms",
    "up_patch_bytes",
    "tower_disp_ms",
    "tower_ms",
    "merger_disp_ms",
    "merger_ms",
    "readback_ms",
    "trim_ms",
)
_REQ_KEYS = (
    "cache_hits",
    "cache_misses",
    "pre_sync_ms",
    "tower_call_ms",
    "rows_upload_ms",
    "rows_upload_bytes",
    "splice_n",
    "splice_rows",
    "splice_len",
    "splice_up_bytes",
    "splice_host_ms",
    "splice_up_ms",
    "mask_clear_n",
    "mask_clear_ms",
    "end_sync_ms",
)


class VisionTiming:
    """Stage clock of one image request. ``sync`` waits for the device queue (ttnn.synchronize_device); a stage that
    ends on device work is closed with ``lap(..., sync=True)`` so its time includes the device part, and the host part
    before the wait is kept as ``<stage>_disp_ms`` for the two eager device stages. No image content is recorded."""

    def __init__(self, sync=None, standalone=False, clock=time.perf_counter):
        self._sync = sync
        self._clock = clock
        self.standalone = standalone
        self.images = []
        self.req = collections.OrderedDict()
        self.meta = collections.OrderedDict()
        self.t0 = self._t = clock()

    # -- clock -------------------------------------------------------------------------------------------------------
    def start(self):
        self._t = self._clock()

    def sync(self):
        if self._sync is not None:
            self._sync()

    def lap(self, name, sync=False, image=True, nbytes=None, disp=False):
        """Close stage ``name``: ms since the previous lap / start (after an optional device wait)."""
        t_host = self._clock()
        if sync:
            self.sync()
        t = self._clock()
        d = self.images[-1] if (image and self.images) else self.req
        d[name + "_ms"] = d.get(name + "_ms", 0.0) + 1e3 * (t - self._t)
        if disp:
            d[name + "_disp_ms"] = d.get(name + "_disp_ms", 0.0) + 1e3 * (t_host - self._t)
        if nbytes is not None:
            d[name + "_bytes"] = d.get(name + "_bytes", 0) + int(nbytes)
        self._t = t

    def add(self, name, value):
        self.req[name] = self.req.get(name, 0) + value

    def note(self, name, value):
        self.meta[name] = value

    def image(self, grid, patches, bucket):
        t, h, w = (int(x) for x in grid)
        self.images.append(collections.OrderedDict(grid=f"{t}x{h}x{w}", patches=int(patches), bucket=int(bucket)))
        self.start()

    # -- output ------------------------------------------------------------------------------------------------------
    def line(self):
        total = 1e3 * (self._clock() - self.t0)
        parts = ["[vision_timing]", f"standalone={int(self.standalone)}"]
        parts += [f"{k}={v}" for k, v in self.meta.items()]
        parts.append(f"tower_images={len(self.images)}")
        if self.images:
            for k, label in (("grid", "grids"), ("patches", "patches"), ("bucket", "buckets")):
                parts.append(f"{label}=" + ",".join(str(im[k]) for im in self.images))
            for k in _IMAGE_KEYS:
                if any(k in im for im in self.images):
                    parts.append(f"{k}=" + ",".join(_fmt(im.get(k, 0)) for im in self.images))
        for k in _REQ_KEYS:
            if k in self.req:
                parts.append(f"{k}={_fmt(self.req[k])}")
        known = sum(self.req.get(k, 0.0) for k in _REQ_KEYS if k.endswith("_ms"))
        if not self.standalone:
            # everything of the prefill call that is not the tower, the row upload, the splice or a timing wait: request
            # RoPE, snapshot restore / saves, trace replays, drafter warm-up, slot write, logits read
            parts.append(f"token_prefill_ms={_fmt(total - known)}")
        parts.append(f"total_ms={_fmt(total)}")
        parts.append(f"flags={flags_summary()}")
        return " ".join(parts)


def _fmt(v):
    return str(v) if isinstance(v, int) else f"{v:.2f}"


_current = None


def timing_current():
    """The open record of the image request being prefilled (None when timing is off or no request is open)."""
    return _current


def timing_begin(sync=None, standalone=False):
    """Open the record of one image request (the engine prefills one request at a time, so one record is open at
    most). The device queue is drained first so earlier work (the previous decode step) is not charged to a stage."""
    global _current
    rec = VisionTiming(sync=sync, standalone=standalone)
    rec.sync()
    rec.lap("pre_sync", image=False)
    _current = rec
    return rec


def timing_end(rec, log=None):
    """Close the record: wait for the device (so in-flight replays / writes of this request are inside its total),
    then emit the line through ``log`` (a callable taking the string)."""
    global _current
    if rec is None:
        return None
    if _current is rec:
        _current = None
    rec.start()
    rec.sync()
    rec.lap("end_sync", image=False)
    line = rec.line()
    if log is not None:
        log(line)
    return line


def timing_abort(rec):
    """Drop the record of a request that raised (no line)."""
    global _current
    if rec is not None and _current is rec:
        _current = None
