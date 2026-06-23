import os
from contextlib import contextmanager
from typing import Any, Optional

import ttnn
import torch

from .quant import dequantize_weight
from .weight_loader import DeepseekV4WeightLoader
from loguru import logger


# ``ttnn.ReadDeviceProfiler`` is a host call that syncs the device; it must never
# run inside a ``ttnn`` trace capture (which records device ops only and forbids
# host round-trips / syncs mid-capture). The traced decode path reuses several of
# the eager ``forward`` helpers below, so route every profiler read through this
# guard and silence it while a trace is being captured.
_IN_TRACE_CAPTURE = False


def _profile(device) -> None:
    if not _IN_TRACE_CAPTURE:
        ttnn.ReadDeviceProfiler(device)


# Tracy signposts let the (flat) device-op profile be sliced per decoder-layer
# sub-module (attention, MoE router/experts/shared, hyper-connection, norms): each
# ``_region`` emits a ``<NAME>_START`` / ``<NAME>_END`` host marker around the ops
# it issues. ``tracy`` only imports on a profiler-enabled build, so degrade to a
# no-op otherwise, and stay silent during a ttnn trace capture (no host calls).
try:
    from tracy import signpost as _tracy_signpost
except Exception:  # pragma: no cover - tracy missing on non-profiling builds
    _tracy_signpost = None

# Master switch for the per-module signposts. Defaults on (they are a no-op unless
# the run is captured under the Tracy profiler), but can be disabled to drop even
# the host-side call overhead: set ``DEEPSEEK_V4_SIGNPOSTS=0`` or call
# :func:`set_signposts_enabled(False)` at runtime.
_SIGNPOSTS_ENABLED = os.environ.get("DEEPSEEK_V4_SIGNPOSTS", "1") not in ("0", "", "false", "False")


def set_signposts_enabled(enabled: bool) -> None:
    """Enable/disable the per-module Tracy signposts at runtime."""
    global _SIGNPOSTS_ENABLED
    _SIGNPOSTS_ENABLED = bool(enabled)


def _signpost(header: str) -> None:
    if _SIGNPOSTS_ENABLED and _tracy_signpost is not None and not _IN_TRACE_CAPTURE:
        _tracy_signpost(header=header)


@contextmanager
def _region(name: str):
    """Wrap the enclosed ttnn ops in a Tracy ``<name>_START`` / ``<name>_END`` pair."""
    _signpost(f"{name}_START")
    try:
        yield
    finally:
        _signpost(f"{name}_END")


@contextmanager
def _trace_capture_guard():
    """Silence :func:`_profile` for the duration of a trace capture."""
    global _IN_TRACE_CAPTURE
    prev = _IN_TRACE_CAPTURE
    _IN_TRACE_CAPTURE = True
    try:
        yield
    finally:
        _IN_TRACE_CAPTURE = prev


class _CachePath(str):
    """A ``str`` cache path that also carries the cache's ``require_cache`` flag.

    Lets the low-level loaders (:func:`_materialize`) see -- from the
    ``cache_file_name`` alone -- whether a cache miss should hard-fail instead of
    falling back to the (expensive) HF checkpoint read, without threading the
    flag through every ``Linear`` / ``RMSNorm`` constructor. It is a plain
    ``str`` everywhere else (``ttnn.as_tensor``, ``os.path``, f-strings).
    """

    __slots__ = ("require_cache",)

    def __new__(cls, value: str, require_cache: bool = False):
        s = super().__new__(cls, value)
        s.require_cache = require_cache
        return s


class WeightCache:
    """On-disk cache namespace for converted ttnn weight tensors.

    Wraps a base directory plus a dotted name prefix. :meth:`file` builds a
    ``cache_file_name`` for :func:`ttnn.as_tensor` (the first load tilizes and
    dumps the tensor; later runs read it straight back, skipping re-conversion).
    A ``None`` path disables caching, so :meth:`file` returns ``None`` and
    ``as_tensor`` simply converts every time -- this is the default, keeping the
    modules' behaviour unchanged unless a caller opts in.

    Sub-modules get a namespaced child via :meth:`sub` so every weight maps to a
    unique, stable path mirroring the checkpoint hierarchy (e.g.
    ``layers.5.attn.q_a_proj``).

    ``require_cache`` turns a cache *miss* (for a tile-cached weight) into a hard
    error instead of a fallback HF-checkpoint read -- used to assert a populated
    cache is actually being consumed (see :class:`DeepSeekV4Model`). It rides
    along on the :class:`_CachePath` returned by :meth:`file` and is propagated to
    every child via :meth:`sub`.
    """

    __slots__ = ("path", "prefix", "require_cache")

    def __init__(self, path: Optional[str] = None, prefix: str = "", require_cache: bool = False):
        self.path = path
        self.prefix = prefix
        self.require_cache = require_cache

    def sub(self, name: str) -> "WeightCache":
        prefix = f"{self.prefix}.{name}" if self.prefix else name
        return WeightCache(self.path, prefix, self.require_cache)

    def require(self, flag: bool = True) -> "WeightCache":
        """A sibling cache (same path / prefix) with ``require_cache`` set."""
        return WeightCache(self.path, self.prefix, flag)

    def _name(self, name: str) -> str:
        return f"{self.prefix}.{name}" if self.prefix else name

    def file(self, name: str) -> Optional[str]:
        if not self.path:
            return None
        return _CachePath(os.path.join(self.path, self._name(name).replace("/", "_")), self.require_cache)

    def hit(self, name: str, dtype: ttnn.DataType, layout: ttnn.Layout = ttnn.TILE_LAYOUT) -> bool:
        """Whether a cached file already exists for ``name`` (matching the exact
        ``ttnn.as_tensor`` suffix), so callers can skip producing the torch
        tensor entirely on a cache hit."""
        base = self.file(name)
        if not base:
            return False
        return os.path.isfile(f"{base}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


def _as_cache(cache: Optional[WeightCache]) -> WeightCache:
    """Normalise ``None`` to a disabled cache so call sites stay branch-free."""
    return cache if cache is not None else WeightCache()


def _load_weight(
    tensor: Optional[torch.Tensor],
    device: ttnn.MeshDevice,
    *,
    cache_file_name: Optional[str] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    """``ttnn.as_tensor`` for a (static) weight, with optional disk caching.

    Equivalent to ``ttnn.from_torch(...)`` when ``cache_file_name`` is ``None``;
    otherwise the tilized tensor is dumped on first use and loaded back on later
    runs. ``tensor`` may be ``None`` only on a verified cache hit (see
    :meth:`WeightCache.hit`).
    """
    return ttnn.as_tensor(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_file_name,
    )


def to_ttnn_device(
    tensor: torch.Tensor,
    device: ttnn.MeshDevice,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    cache_file_name: Optional[str] = None,
) -> ttnn.Tensor:
    return _load_weight(tensor, device, cache_file_name=cache_file_name, layout=layout)


# ---------------------------------------------------------------------------- #
# Lazy weight resolution
#
# A "weight source" handed to a module may be either an eager ``torch.Tensor``
# or a zero-arg callable (a *thunk*) that produces one on demand. The thunk lets
# a populated on-disk cache short-circuit the (expensive) checkpoint read /
# dequant entirely: when the converted tile already exists for a weight, the
# source is never touched, so no tensor is pulled from the safetensors shards.
# ---------------------------------------------------------------------------- #
def _cache_hit_file(
    cache_file_name: Optional[str],
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> bool:
    """Whether the converted-tile cache file for ``cache_file_name`` exists.

    Mirrors the suffix :func:`ttnn.as_tensor` appends, so a leaf weight loader
    can tell -- from the cache path alone -- that a populated cache will serve
    the tensor and the (lazy) checkpoint read can be skipped entirely.
    """
    if not cache_file_name:
        return False
    return os.path.isfile(f"{cache_file_name}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


def _materialize(
    weight,
    cache_file_name: Optional[str],
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
) -> Optional[torch.Tensor]:
    """Resolve a (possibly lazy) weight source to a torch tensor, or ``None``.

    ``weight`` is either a ``torch.Tensor`` or a zero-arg callable returning one.
    On a cache hit the source is never touched (so a populated cache reads
    nothing from the checkpoint); otherwise the thunk is called -- or the tensor
    returned as-is for the eager path.

    When the cache path carries ``require_cache`` (a :class:`_CachePath`), a miss
    is a hard error instead of an HF-checkpoint fallback read.
    """
    if _cache_hit_file(cache_file_name, dtype, layout):
        return None
    if getattr(cache_file_name, "require_cache", False):
        raise RuntimeError(
            f"weight cache miss for {str(cache_file_name)!r} "
            f"(dtype={dtype.name}, layout={layout.name}) with require_cache=True; "
            "refusing to load from the HF checkpoint"
        )
    return weight() if callable(weight) else weight


def _memo(weight):
    """Wrap a tensor-or-thunk as a memoized zero-arg thunk (loads at most once).

    Used where one checkpoint tensor is sliced into several device weights (e.g.
    the packed hyper-connection projection) so the underlying source is read a
    single time even across multiple cache-miss slices.
    """
    if not callable(weight):
        return lambda: weight
    box: dict = {}

    def get():
        if "v" not in box:
            box["v"] = weight()
        return box["v"]

    return get


# ---------------------------------------------------------------------------- #
# DeepSeek-V4-Flash attention (decode, running KV cache)
#
# ttnn port of ``DeepseekV4Attention`` (and its CSA / HCA compressors) from
# ``modular_deepseek_v4.py``. Scope is *decode only*: each step appends the new
# token's K=V (and compressor projections) to the running cache and attends the
# tokens-so-far, via the fused ``scaled_dot_product_attention_decode`` op.
#
# Layout conventions, matching the reference:
#   B = batch, S = query/seq length, H = num_attention_heads, Dh = head_dim,
#   Rd = qk_rope_head_dim (the trailing RoPE slice of each head).
# V4 is shared-KV MQA (one KV head broadcast to all query heads) and lays each
# head out as ``[nope | rope]`` with interleaved RoPE on the trailing ``Rd``.
# ---------------------------------------------------------------------------- #

# fp32 accumulation everywhere keeps the long (Dh=512) reductions and the
# softmax/RoPE chains from drifting under bf16; the per-layer PCC test needs it.
_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# The fused ``scaled_dot_product_attention_decode`` op must NOT run with
# ``fp32_dest_acc_en=True``: for this attention shape (head_dim=256, MQA with a
# single shared K==V head) that flag makes the kernel emit garbage (PCC ~0.36 vs
# the manual softmax). HiFi4 with bf16 dest accumulation matches the manual path
# at PCC ~0.9999. ``packer_l1_acc`` is safe to keep on.
_HIFI4_SDPA = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

# Additive-mask "-inf": a finite bf16-representable floor. Masked logits feed
# ``exp(x - max)`` which underflows to 0 for both this and a true ``-inf``, but
# the finite value avoids ``inf - inf -> NaN`` if a whole row were masked.
_MASK_NEG = -1.0e9


class DeepSeekV4Module:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


# ---------------------------------------------------------------------------- #
# KV / compressor cache (decode)
#
# The only cross-token state in the V4-Flash stack lives in attention: the
# hyper-connection streams, RMSNorms, the routed/shared MoE and the MLP are all
# strictly per-token. So a single-token decode step only needs to remember, per
# decoder layer:
#
#   * the rotated sliding K=V entries (shared-KV MQA, K==V), capped to the
#     ``sliding_window`` most recent tokens, and
#   * for CSA / HCA layers, every source token's compressor projections
#     (``kv`` / ``gate``); the compressed long-range entries are re-pooled from
#     these each step with the exact prefill pooling, so decode is bit-for-bit
#     the same function of the tokens-so-far as a full prefill over them (no
#     separate rolling-window / overlap / entry-count bookkeeping needed).
#
# Re-pooling the (small, ``head_dim``-wide) compressor each step is cheap next
# to the per-token MoE / projection matmuls, which now run at ``S = 1`` instead
# of over the whole growing context as in the repeated-prefill demo.
# ---------------------------------------------------------------------------- #
class _SlidingKVCache:
    """Append-only rotated K=V cache, capped to the last ``sliding_window`` rows."""

    def __init__(self, sliding_window: int):
        self.window = sliding_window
        self.kv: Optional[ttnn.Tensor] = None  # [B, 1, L, Dh]

    def append(self, kv_new: ttnn.Tensor) -> ttnn.Tensor:
        """Append ``kv_new`` ``[B, 1, n, Dh]`` and return the (capped) cache."""
        self.kv = kv_new if self.kv is None else ttnn.concat([self.kv, kv_new], dim=2)
        b, _, length, dh = self.kv.shape
        if length > self.window:
            self.kv = ttnn.slice(self.kv, [0, 0, length - self.window, 0], [b, 1, length, dh])
        return self.kv


class _CompressorCache:
    """All source tokens' compressor ``kv`` / ``gate`` projections ``[B, T, c*Dh]``."""

    def __init__(self):
        self.kv: Optional[ttnn.Tensor] = None
        self.gate: Optional[ttnn.Tensor] = None

    def append(self, kv_new: ttnn.Tensor, gate_new: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        self.kv = kv_new if self.kv is None else ttnn.concat([self.kv, kv_new], dim=1)
        self.gate = gate_new if self.gate is None else ttnn.concat([self.gate, gate_new], dim=1)
        return self.kv, self.gate


class _LayerKVCache:
    """Per-decoder-layer decode state: a sliding K=V cache + optional compressor."""

    def __init__(self, sliding_window: int, has_compressor: bool):
        self.sliding = _SlidingKVCache(sliding_window)
        self.compressor = _CompressorCache() if has_compressor else None


class _StaticLayerCache:
    """Fixed-size, in-place per-layer decode caches for the *traced* decode path.

    Unlike :class:`_LayerKVCache` (which grows via ``concat`` each step), these are
    DRAM tensors of a fixed capacity written in place at the new token's position
    by ``paged_update_cache`` (a device-tensor index), so the captured trace's
    shapes / addresses are step-invariant:

      * ``sliding`` ``[1, 1, window, Dh]`` -- a ring buffer (slot ``pos % window``);
        attention masks unwritten / out-of-window slots.
      * ``compressor_kv`` / ``compressor_gate`` ``[1, 1, cap, feat]`` -- every
        source token's compressor projection at its absolute position; the pool
        runs over the whole buffer and the block-bias mask drops the windows past
        the current position. ``None`` for sliding-only layers.

    Built empty (all-zero) by :meth:`DeepSeekV4Model.prepare_static_decode`; the
    prompt is written in by replaying :meth:`decode_traced` per prompt token.
    """

    __slots__ = ("sliding", "compressor_kv", "compressor_gate")

    def __init__(
        self, sliding: ttnn.Tensor, compressor_kv: Optional[ttnn.Tensor], compressor_gate: Optional[ttnn.Tensor]
    ):
        self.sliding = sliding
        self.compressor_kv = compressor_kv
        self.compressor_gate = compressor_gate


def _interleaved_rotate_matrix(rope_dim: int) -> torch.Tensor:
    """Fixed ``[Rd, Rd]`` matrix ``R`` s.t. ``x @ R == rotate_half(x)``.

    GLM/V4 interleaved ``rotate_half`` maps each consecutive pair
    ``(x_{2p}, x_{2p+1}) -> (-x_{2p+1}, x_{2p})``. As a right-multiply that is a
    block-diagonal matrix of ``[[0, 1], [-1, 0]]`` blocks, which lets us express
    the rotation as a single on-device matmul instead of strided gathers.
    """
    r = torch.zeros(rope_dim, rope_dim, dtype=torch.float32)
    for p in range(rope_dim // 2):
        r[2 * p, 2 * p + 1] = 1.0
        r[2 * p + 1, 2 * p] = -1.0
    return r


def make_rope_table(cos_half: torch.Tensor, sin_half: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand half-sized rotary ``(cos, sin)`` to full ``Rd`` and shape ``[1,1,L,Rd]``.

    ``DeepseekV4RotaryEmbedding`` emits one entry per interleaved pair; the
    reference ``apply_rotary_pos_emb`` does ``repeat_interleave(2)`` before the
    rotation. We bake that into the host-side table (broadcast over batch/heads).
    """
    cos = cos_half.repeat_interleave(2, dim=-1)
    sin = sin_half.repeat_interleave(2, dim=-1)
    cos = cos.reshape(1, 1, cos.shape[-2], cos.shape[-1]).float()
    sin = sin.reshape(1, 1, sin.shape[-2], sin.shape[-1]).float()
    return cos, sin


class Linear(DeepSeekV4Module):
    """``nn.Linear`` (bias-free) as ``x @ Wᵀ`` for ttnn.

    ttnn ``linear`` computes ``a @ b`` with ``b`` shaped ``[in, out]``, so we
    store the torch ``[out, in]`` weight transposed.
    """

    def __init__(
        self,
        weight,
        device: ttnn.MeshDevice,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        w = _materialize(weight, cache_file_name, dtype)
        self.weight = _load_weight(
            w.t().contiguous() if w is not None else None,
            device,
            cache_file_name=cache_file_name,
            dtype=dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(x, self.weight, compute_kernel_config=_HIFI4)


class DeepSeekV4RMSNorm(DeepSeekV4Module):
    """Weighted RMSNorm over the last dim (matches ``DeepseekV4RMSNorm``)."""

    def __init__(self, weight, eps: float, device: ttnn.MeshDevice, cache_file_name: Optional[str] = None):
        w = _materialize(weight, cache_file_name, ttnn.bfloat16)
        self.weight = _load_weight(
            w.reshape(1, 1, 1, -1) if w is not None else None, device, cache_file_name=cache_file_name
        )
        self.eps = eps

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(x, weight=self.weight, epsilon=self.eps)


def _rms_norm_unweighted(x: ttnn.Tensor, eps: float) -> ttnn.Tensor:
    """Unweighted RMSNorm over the last dim (matches ``DeepseekV4UnweightedRMSNorm``)."""
    return ttnn.rms_norm(x, epsilon=eps)


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, rot: ttnn.Tensor, rope_dim: int) -> ttnn.Tensor:
    """Interleaved RoPE on the trailing ``rope_dim`` channels of ``x`` ([.., D]).

    ``cos`` / ``sin`` are ``[1,1,L,rope_dim]`` tables (broadcast over batch/heads);
    ``rot`` is the ``[rope_dim, rope_dim]`` ``rotate_half`` matrix. Leading "nope"
    channels pass through untouched.
    """
    shape = list(x.shape)
    d = shape[-1]
    if d == rope_dim:
        nope = None
        rope = x
    else:
        nope = ttnn.slice(x, [0, 0, 0, 0], [shape[0], shape[1], shape[2], d - rope_dim])
        rope = ttnn.slice(x, [0, 0, 0, d - rope_dim], shape)
    rotated = ttnn.add(
        ttnn.multiply(rope, cos), ttnn.multiply(ttnn.matmul(rope, rot, compute_kernel_config=_HIFI4), sin)
    )
    if nope is None:
        return rotated
    return ttnn.concat([nope, rotated], dim=-1)


# ---------------------------------------------------------------------------- #
# Traced-decode helpers (fixed-size, in-place KV cache via ``paged_update_cache``)
#
# A reusable ``ttnn`` trace requires fixed tensor shapes / addresses and no host
# round-trips inside the captured region, so the traced decode swaps the eager
# concat-grown caches for fixed-size DRAM buffers that are written *in place*
# every step at the new token's position (a device-tensor index, so the same
# trace serves every step). ``paged_update_cache`` is the canonical trace-safe
# in-place KV writer (it mutates the persistent cache buffer during capture,
# unlike ``ttnn.copy`` which is rejected mid-capture).
# ---------------------------------------------------------------------------- #
def _height_sharded_l1_config(width: int) -> ttnn.MemoryConfig:
    """Single-core height-sharded L1 config for a ``[1, 1, 1, width]`` decode row.

    ``paged_update_cache`` requires its (single-token) input to be height-sharded
    with one core per batch user (B == 1 here -> one core), shard width == the
    last dim, ROW_MAJOR orientation.
    """
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    shard_spec = ttnn.ShardSpec(grid, [ttnn.TILE_SIZE, width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _update_cache_at(cache: ttnn.Tensor, row: ttnn.Tensor, pos_tensor: ttnn.Tensor) -> None:
    """In-place write ``row`` ``[1, 1, 1, F]`` into ``cache`` ``[1, 1, L, F]`` at the
    sequence index held (on device) by ``pos_tensor`` ``[1]`` (INT32). Trace-safe."""
    width = row.shape[-1]
    row_sharded = ttnn.interleaved_to_sharded(row, _height_sharded_l1_config(width))
    ttnn.experimental.paged_update_cache(cache, row_sharded, update_idxs_tensor=pos_tensor)
    ttnn.deallocate(row_sharded)


class DeepSeekV4Embedding(DeepSeekV4Module):
    def __init__(
        self,
        weight_loader: DeepseekV4WeightLoader,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
    ):
        self.weight_loader = weight_loader
        self.device = device
        cache = _as_cache(cache)
        # ``ttnn.embedding`` expects a row-major weight table.
        cfn = cache.file("embed_tokens")
        hit = cache.hit("embed_tokens", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        if cache.require_cache and not hit:
            raise RuntimeError("weight cache miss for 'embed_tokens' with require_cache=True")
        embed = None if hit else weight_loader.get_tensor("embed_tokens.weight")
        self.embedding_weight = to_ttnn_device(embed, device, layout=ttnn.ROW_MAJOR_LAYOUT, cache_file_name=cfn)

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.embedding(input_ids, self.embedding_weight, layout=ttnn.TILE_LAYOUT)


class DeepSeekV4Flash(DeepSeekV4Module):
    """Stub V4-Flash model wired up to the safetensors weight loader.

    Only the embedding table is materialised today; the rest of the model is
    a placeholder. The loader, however, can already serve every parameter in
    the checkpoint (see ``load_state_dict_torch``), so as more submodules are
    fleshed out they can be populated with one ``loader.get_tensor(...)`` per
    parameter.
    """

    def __init__(
        self,
        config: dict,
        weights_dir: str,
        device: ttnn.MeshDevice,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.weights_dir = weights_dir
        self.device = device
        self.weight_loader = DeepseekV4WeightLoader(weights_dir)
        # Converted-weight cache (opt-in): pass ``cache_dir`` to dump/reuse the
        # tilized ttnn tensors across runs (skipping re-conversion / dequant).
        # ``None`` keeps caching off, so weights are converted every time.
        self.cache = WeightCache(cache_dir)
        self.embed_tokens = DeepSeekV4Embedding(self.weight_loader, device, cache=self.cache)

    def load_weights(self) -> None:
        """Populate the model's submodules from the safetensors checkpoint.

        Currently fills only ``embed_tokens`` (the rest of the model is a
        stub); extend this as new submodules land.
        """
        embed = self.weight_loader.get_tensor("embed_tokens.weight")
        self.embed_tokens.weight = ttnn.from_torch(embed)

    def load_state_dict_torch(self, hf_names: list[str]) -> dict[str, torch.Tensor]:
        """Return a ``{hf_name: torch.Tensor}`` dict for the given names.

        Convenience for tests / reference comparisons that want the raw
        torch tensors keyed by HF-style names.
        """
        return {name: self.weight_loader.get_tensor(name) for name in hf_names}

    def forward(self, input_ids: ttnn.Tensor, attention_mask: ttnn.Tensor) -> ttnn.Tensor:
        input_embeddings = self.embed_tokens(input_ids)
        return input_embeddings


def _softmax_weighted_sum(kv: ttnn.Tensor, gate: ttnn.Tensor, window_axis: int) -> ttnn.Tensor:
    """``sum_w softmax(gate, axis=w) * kv`` over the window axis.

    Shared compressor pooling (``DeepseekV4*Compressor``): the gate logits are
    softmaxed over the per-window token axis and used to convex-combine the kv
    rows into one compressed entry per window.
    """
    weights = ttnn.softmax(gate, dim=window_axis)
    return ttnn.sum(ttnn.multiply(kv, weights), dim=window_axis)


class DeepSeekV4HCACompressor:
    """Heavily-Compressed-Attention compressor (decode, running KV cache).

    Compresses every complete window of ``compress_rate`` (m'=128) source tokens
    into a single softmax-gated KV entry, then RoPEs each entry at its window's
    absolute position. Returns ``compressed_kv`` shaped ``[B, 1, n_windows, Dh]``
    ready to concat onto the sliding KV axis.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device,
        rot,
        rope_dim: int,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.rope_dim = rope_dim
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        cache = _as_cache(cache)
        self.kv_proj = Linear(
            weights["compressor.kv_proj.weight"], device, cache.file("compressor.kv_proj"), dtype=weight_dtype
        )
        self.gate_proj = Linear(
            weights["compressor.gate_proj.weight"], device, cache.file("compressor.gate_proj"), dtype=weight_dtype
        )
        self.kv_norm = DeepSeekV4RMSNorm(
            weights["compressor.kv_norm.weight"], self.eps, device, cache.file("compressor.kv_norm")
        )
        # position_bias: [compress_rate, head_dim] -> broadcast over [B, n_win].
        pb = _materialize(weights["compressor.position_bias"], cache.file("compressor.position_bias"), ttnn.bfloat16)
        self.position_bias = _load_weight(
            pb.reshape(1, 1, self.compress_rate, self.head_dim) if pb is not None else None,
            device,
            cache_file_name=cache.file("compressor.position_bias"),
        )

    def _project(self, hidden: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``hidden`` ``[B, S, D]`` -> per-token ``(kv, gate)`` ``[B, S, Dh]`` each."""
        return self.kv_proj(hidden), self.gate_proj(hidden)

    def _pool(
        self, kv: ttnn.Tensor, gate: ttnn.Tensor, cos_win: ttnn.Tensor, sin_win: ttnn.Tensor
    ) -> ttnn.Tensor | None:
        """Pool the projected ``(kv, gate)`` ``[B, T, Dh]`` into compressed entries.

        Compresses every complete window of ``compress_rate`` tokens into one
        softmax-gated entry and RoPEs it at its window position. Shared by the
        :meth:`decode` and :meth:`decode_static` paths.
        """
        b = kv.shape[0]
        t = kv.shape[1]
        cr = self.compress_rate
        n_win = t // cr
        if n_win == 0:
            return None
        usable = n_win * cr
        if usable != t:
            kv = ttnn.slice(kv, [0, 0, 0], [b, usable, self.head_dim])
            gate = ttnn.slice(gate, [0, 0, 0], [b, usable, self.head_dim])
        kv = ttnn.reshape(kv, [b, n_win, cr, self.head_dim])
        gate = ttnn.reshape(gate, [b, n_win, cr, self.head_dim])
        gate = ttnn.add(gate, self.position_bias)
        compressed = _softmax_weighted_sum(kv, gate, window_axis=2)  # [B, n_win, Dh]
        compressed = self.kv_norm(compressed)
        compressed = ttnn.reshape(compressed, [b, 1, n_win, self.head_dim])
        return _apply_rope(compressed, cos_win, sin_win, self.rot, self.rope_dim)

    def decode(
        self, hidden: ttnn.Tensor, cos_win: ttnn.Tensor, sin_win: ttnn.Tensor, cache: "_CompressorCache"
    ) -> ttnn.Tensor | None:
        """Project the new token(s), append to ``cache``, re-pool all entries.

        ``cos_win`` / ``sin_win`` must cover every window emittable from the full
        cached projection length (``n_win`` rows); the caller slices them.
        """
        kv, gate = self._project(hidden)
        kv_all, gate_all = cache.append(kv, gate)
        return self._pool(kv_all, gate_all, cos_win, sin_win)

    def decode_static(
        self,
        hidden: ttnn.Tensor,
        cos_win: ttnn.Tensor,
        sin_win: ttnn.Tensor,
        kv_cache: ttnn.Tensor,
        gate_cache: ttnn.Tensor,
        pos_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Trace-safe decode: write this token's projection in place at ``pos_tensor``
        into the fixed ``[1, 1, cap, Dh]`` caches, then pool over the *whole* buffer.

        ``cos_win`` / ``sin_win`` cover every window of the fixed capacity
        (``cap // compress_rate`` rows); windows past the current position are
        pooled from zero-filled (unwritten) projections and dropped by the caller's
        additive block-bias mask.
        """
        kv, gate = self._project(hidden)  # [1, 1, Dh]
        kv = ttnn.reshape(kv, [1, 1, 1, self.head_dim])
        gate = ttnn.reshape(gate, [1, 1, 1, self.head_dim])
        _update_cache_at(kv_cache, kv, pos_tensor)
        _update_cache_at(gate_cache, gate, pos_tensor)
        cap = kv_cache.shape[2]
        kv_all = ttnn.reshape(kv_cache, [1, cap, self.head_dim])
        gate_all = ttnn.reshape(gate_cache, [1, cap, self.head_dim])
        return self._pool(kv_all, gate_all, cos_win, sin_win)


class DeepSeekV4CSACompressor:
    """Compressed-Sparse-Attention compressor (decode, running KV cache).

    Like HCA but with the two-series Ca/Cb overlap scheme: each token projects to
    ``2*Dh`` (Ca = its contribution to the *next* window, Cb = to the *current*
    window). Compressed entry ``w`` pools window ``w-1``'s Ca slice with window
    ``w``'s Cb slice over a width-``2*compress_rate`` window. Window 0's Ca half
    is zero-kv / ``-inf``-gate (softmax weight 0), since there is no prior window.

    The CSA Lightning Indexer only affects *which* compressed entries each query
    may see (the ``block_bias``); for ``seq_len <= index_topk * compress_rate``
    its top-k selects every entry, so the block_bias reduces to plain causal
    masking over windows, which the caller builds on host. The compressed KV
    values themselves (this module's output) do not depend on the indexer.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device,
        rot,
        rope_dim: int,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.rope_dim = rope_dim
        self.rot = rot
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        cache = _as_cache(cache)
        self.kv_proj = Linear(
            weights["compressor.kv_proj.weight"], device, cache.file("compressor.kv_proj"), dtype=weight_dtype
        )
        self.gate_proj = Linear(
            weights["compressor.gate_proj.weight"], device, cache.file("compressor.gate_proj"), dtype=weight_dtype
        )
        self.kv_norm = DeepSeekV4RMSNorm(
            weights["compressor.kv_norm.weight"], self.eps, device, cache.file("compressor.kv_norm")
        )
        pb = _materialize(weights["compressor.position_bias"], cache.file("compressor.position_bias"), ttnn.bfloat16)
        self.position_bias = _load_weight(
            pb.reshape(1, 1, self.compress_rate, 2 * self.head_dim) if pb is not None else None,
            device,
            cache_file_name=cache.file("compressor.position_bias"),
        )
        # Persistent window-0 Ca filler (zero kv / ``-inf`` gate, softmax weight 0).
        # ``_pool`` re-uploaded these from host on every call -- a host transfer
        # that is illegal inside a trace -- so keep them resident (B == 1).
        self._zeros_filler = ttnn.from_torch(
            torch.zeros(1, 1, self.compress_rate, self.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._neg_filler = ttnn.from_torch(
            torch.full((1, 1, self.compress_rate, self.head_dim), _MASK_NEG),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def _project(self, hidden: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``hidden`` ``[B, S, D]`` -> per-token ``(kv, gate)`` ``[B, S, 2*Dh]`` each."""
        return self.kv_proj(hidden), self.gate_proj(hidden)

    def _pool(
        self, kv: ttnn.Tensor, gate: ttnn.Tensor, cos_win: ttnn.Tensor, sin_win: ttnn.Tensor
    ) -> ttnn.Tensor | None:
        """Pool the projected ``(kv, gate)`` ``[B, T, 2*Dh]`` into compressed entries
        (two-series Ca/Cb overlap). Shared by the decode paths."""
        b = kv.shape[0]
        t = kv.shape[1]
        cr = self.compress_rate
        dh = self.head_dim
        n_win = t // cr
        if n_win == 0:
            return None
        usable = n_win * cr
        if usable != t:
            kv = ttnn.slice(kv, [0, 0, 0], [b, usable, 2 * dh])
            gate = ttnn.slice(gate, [0, 0, 0], [b, usable, 2 * dh])
        kv = ttnn.reshape(kv, [b, n_win, cr, 2 * dh])
        gate = ttnn.reshape(gate, [b, n_win, cr, 2 * dh])
        gate = ttnn.add(gate, self.position_bias)

        ca, cb = ttnn.split(kv, dh, dim=3)
        ca_g, cb_g = ttnn.split(gate, dh, dim=3)
        _profile(self.device)

        # Shift Ca down one window: entry w sees window w-1's Ca; entry 0 sees a
        # zero-kv / -inf-gate filler (softmax weight 0). Fillers are resident
        # (B == 1) so the pool stays trace-safe (no per-call host transfer).
        zeros = self._zeros_filler
        neg = self._neg_filler
        ca_prev_src = ttnn.slice(ca, [0, 0, 0, 0], [b, n_win - 1, cr, dh])
        cag_prev_src = ttnn.slice(ca_g, [0, 0, 0, 0], [b, n_win - 1, cr, dh])
        ca_prev = ttnn.concat([zeros, ca_prev_src], dim=1)
        cag_prev = ttnn.concat([neg, cag_prev_src], dim=1)

        new_kv = ttnn.concat([ca_prev, cb], dim=2)  # [B, n_win, 2*cr, Dh]
        new_gate = ttnn.concat([cag_prev, cb_g], dim=2)
        compressed = _softmax_weighted_sum(new_kv, new_gate, window_axis=2)  # [B, n_win, Dh]
        compressed = self.kv_norm(compressed)
        compressed = ttnn.reshape(compressed, [b, 1, n_win, dh])
        return _apply_rope(compressed, cos_win, sin_win, self.rot, self.rope_dim)

    def decode(
        self, hidden: ttnn.Tensor, cos_win: ttnn.Tensor, sin_win: ttnn.Tensor, cache: "_CompressorCache"
    ) -> ttnn.Tensor | None:
        kv, gate = self._project(hidden)
        kv_all, gate_all = cache.append(kv, gate)
        return self._pool(kv_all, gate_all, cos_win, sin_win)

    def decode_static(
        self,
        hidden: ttnn.Tensor,
        cos_win: ttnn.Tensor,
        sin_win: ttnn.Tensor,
        kv_cache: ttnn.Tensor,
        gate_cache: ttnn.Tensor,
        pos_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Trace-safe decode: write this token's ``2*Dh`` projection in place at
        ``pos_tensor`` into the fixed ``[1, 1, cap, 2*Dh]`` caches, then pool the
        whole buffer (Ca/Cb overlap). See :meth:`DeepSeekV4HCACompressor.decode_static`."""
        feat = 2 * self.head_dim
        kv, gate = self._project(hidden)  # [1, 1, 2*Dh]
        kv = ttnn.reshape(kv, [1, 1, 1, feat])
        gate = ttnn.reshape(gate, [1, 1, 1, feat])
        _update_cache_at(kv_cache, kv, pos_tensor)
        _update_cache_at(gate_cache, gate, pos_tensor)
        cap = kv_cache.shape[2]
        kv_all = ttnn.reshape(kv_cache, [1, cap, feat])
        gate_all = ttnn.reshape(gate_cache, [1, cap, feat])
        return self._pool(kv_all, gate_all, cos_win, sin_win)


_COMPRESSORS = {
    "compressed_sparse_attention": DeepSeekV4CSACompressor,
    "heavily_compressed_attention": DeepSeekV4HCACompressor,
}


class DeepSeekV4Attention(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4Attention`` (decode only, running KV cache).

    Construct from a ``config`` (the HF ``DeepseekV4Config`` or any object
    exposing the same attributes), the layer's torch ``weights`` (HF-named
    ``state_dict`` entries), and a device. :meth:`decode` / :meth:`decode_static`
    consume pre-built RoPE tables (see :func:`make_rope_table`); these are inputs
    because the rotary embedding is owned by the surrounding model in the
    reference, not by the attention block.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.layer_type = config.layer_types[layer_idx]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.o_groups = config.o_groups
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        self.scaling = self.head_dim**-0.5
        cache = _as_cache(cache)

        self.q_a_proj = Linear(weights["q_a_proj.weight"], device, cache.file("q_a_proj"), dtype=weight_dtype)
        self.q_a_norm = DeepSeekV4RMSNorm(weights["q_a_norm.weight"], self.eps, device, cache.file("q_a_norm"))
        self.q_b_proj = Linear(weights["q_b_proj.weight"], device, cache.file("q_b_proj"), dtype=weight_dtype)
        self.kv_proj = Linear(weights["kv_proj.weight"], device, cache.file("kv_proj"), dtype=weight_dtype)
        self.kv_norm = DeepSeekV4RMSNorm(weights["kv_norm.weight"], self.eps, device, cache.file("kv_norm"))
        self.o_b_proj = Linear(weights["o_b_proj.weight"], device, cache.file("o_b_proj"), dtype=weight_dtype)

        # Grouped output projection (``DeepseekV4GroupedLinear``): block-diagonal
        # over o_groups. Store the per-group weight as [g, in_per_group, out_per_group]
        # so a single batched matmul (batch axis = group) does all groups at once.
        # oa: [g*o_lora_rank, (H*Dh)//g] -> per-group [g, in_per_group, o_lora_rank].
        oa = _materialize(weights["o_a_proj.weight"], cache.file("o_a_proj"), weight_dtype)
        in_per_group = (self.num_heads * self.head_dim) // self.o_groups
        if oa is not None:
            oa = oa.reshape(self.o_groups, self.o_lora_rank, in_per_group).transpose(1, 2).contiguous()
        self.o_a_weight = _load_weight(oa, device, cache_file_name=cache.file("o_a_proj"), dtype=weight_dtype)

        # sinks live on host (folded into the softmax denominator), so there is
        # no tile cache for them -- always materialise.
        sinks = weights["sinks"]
        sinks = sinks() if callable(sinks) else sinks
        self.sinks_torch = sinks.reshape(1, self.num_heads, 1, 1).float()
        # Sink for the fused SDPA-decode op (:meth:`_sdpa_decode`). That kernel
        # multiplies ``scale`` into BOTH the QK logits and the sink before the
        # exp, but the reference leaves the sink un-scaled, so we pre-divide by
        # ``scaling`` to cancel it. Shape ``[H, TILE]``
        # (per-head, tile-padded width), resident so the call stays trace-safe.
        sdpa_sink = self.sinks_torch.reshape(self.num_heads, 1) / self.scaling
        sdpa_sink = torch.nn.functional.pad(sdpa_sink, (0, ttnn.TILE_SIZE - 1), "constant", value=0.0)
        self.sdpa_sinks_tt = ttnn.from_torch(sdpa_sink, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        # SDPA-decode needs an explicit program config (k_chunk_size) when given an
        # attn_mask. The K=V sequence (sliding window + compressor windows) is a
        # multiple of the tile size, so a 32-wide chunk divides it cleanly.
        #
        # ``max_cores_per_head_batch`` (NOT the grid) is the L1 lever here: this is
        # MQA (one shared KV head) at batch 1, so there is a single reduction group
        # and the op assigns ``min(grid, max_cores_per_head_batch)`` cores to reduce
        # that one head. Its per-core reduction-scratch CB grows as
        # ``(out_tiles + 2*PNHt) * (cores_per_head - 1)``; with the default 16 and
        # ``head_dim == 256`` that overflows L1 (~1.8 MB > 1.5 MB), independent of
        # the grid. Capping it to 4 shrinks that CB ~5x while still parallelising
        # the KV reduction 4 ways.
        self._sdpa_pcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=0,
            k_chunk_size=32,
            exp_approx_mode=False,
            max_cores_per_head_batch=4,
        )

        # The rotate-half matrix must stay precise (a bf4 rotation would corrupt RoPE).
        self.rot = _load_weight(_interleaved_rotate_matrix(self.rope_dim), device, cache_file_name=cache.file("rot"))

        compressor_cls = _COMPRESSORS.get(self.layer_type)
        self.compressor = (
            compressor_cls(config, weights, device, self.rot, self.rope_dim, cache=cache, weight_dtype=weight_dtype)
            if compressor_cls is not None
            else None
        )

    def _sdpa_decode(self, q: ttnn.Tensor, kv: ttnn.Tensor, mask: ttnn.Tensor) -> ttnn.Tensor:
        """Single-token (``S == 1``) attention via the fused SDPA-decode op.

        Drop-in for :meth:`_attention` on the decode paths: fuses the scale, the
        additive ``mask``, the per-head sink, and both matmuls into one device op.

        ``q`` ``[1, H, 1, Dh]``; ``kv`` is the shared K==V ``[1, 1, Skv, Dh]``
        (MQA, one KV head); ``mask`` ``[1, 1, 1, Skv]`` additive (``0`` valid /
        ``_MASK_NEG`` masked). The op consumes Q as ``[1, B, H, Dh]`` (one row per
        batch) and emits the same, so we swap the head/seq axes around the call.

        The op requires the mask to carry the same (padded) head count as Q, so the
        head-independent ``mask`` is broadcast across the ``H`` head axis first.
        """
        q_in = ttnn.transpose(q, 1, 2)  # [1, H, 1, Dh] -> [1, 1, H, Dh]
        mask_h = ttnn.repeat(mask, ttnn.Shape([1, 1, self.num_heads, 1]))  # [1, 1, H, Skv]
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_in,
            kv,
            kv,  # K == V (shared single KV head)
            is_causal=False,
            attn_mask=mask_h,
            attention_sink=self.sdpa_sinks_tt,
            scale=self.scaling,
            program_config=self._sdpa_pcfg,
            compute_kernel_config=_HIFI4_SDPA,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, H, Dh]
        return ttnn.transpose(attn, 1, 2)  # -> [1, H, 1, Dh]

    def _grouped_output(self, attn: ttnn.Tensor) -> ttnn.Tensor:
        """``DeepseekV4GroupedLinear`` (o_a) + ``o_b_proj``.

        ``attn`` is ``[B, S, H, Dh]``. Reshape to per-group feature blocks, run a
        batched matmul over the group axis, then mix groups back to hidden.
        """
        b, s, h, dh = attn.shape
        in_per_group = (h * dh) // self.o_groups
        x = ttnn.reshape(attn, [b * s, self.o_groups, in_per_group])
        x = ttnn.permute(x, [1, 0, 2])  # [g, B*S, in_per_group]
        y = ttnn.matmul(x, self.o_a_weight, compute_kernel_config=_HIFI4)  # [g, B*S, o_lora_rank]
        y = ttnn.permute(y, [1, 0, 2])  # [B*S, g, o_lora_rank]
        y = ttnn.reshape(y, [b, s, self.o_groups * self.o_lora_rank])
        return self.o_b_proj(y)

    def _qkv(self, hidden: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Project + RoPE the query and (shared) K=V for ``hidden`` ``[B, S, D]``.

        Returns ``q`` ``[B, H, S, Dh]`` and the rotated ``kv`` ``[B, 1, S, Dh]``
        (pre-compressor, pre-cache). Shared by the decode paths.
        """
        b, s, _ = hidden.shape
        h, dh = self.num_heads, self.head_dim
        _profile(self.device)

        q_residual = self.q_a_norm(self.q_a_proj(hidden))  # [B, S, q_lora_rank]
        q = self.q_b_proj(q_residual)  # [B, S, H*Dh]
        q = ttnn.reshape(q, [b, s, h, dh])
        q = ttnn.transpose(q, 1, 2)  # [B, H, S, Dh]
        q = _rms_norm_unweighted(q, self.eps)
        q = _apply_rope(q, cos, sin, self.rot, self.rope_dim)

        kv = self.kv_norm(self.kv_proj(hidden))  # [B, S, Dh]
        kv = ttnn.reshape(kv, [b, s, 1, dh])
        kv = ttnn.transpose(kv, 1, 2)  # [B, 1, S, Dh]
        kv = _apply_rope(kv, cos, sin, self.rot, self.rope_dim)
        return q, kv

    def decode(
        self,
        hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        kv_cache: "_LayerKVCache",
    ) -> ttnn.Tensor:
        """Single-token decode attention against the running ``kv_cache``.

        ``hidden`` is ``[B, 1, D]`` (the new token); ``cos`` / ``sin`` / ``neg_sin``
        are the single RoPE rows ``[1,1,1,Rd]`` at this token's absolute position;
        ``cos_win`` / ``sin_win`` cover every currently-emittable compressor window.

        The mask is implicitly zero: a single causal query at the front of its
        (window-capped) cache sees every cached sliding key and every existing
        compressed entry, so no additive masking is needed (valid while the
        sequence stays within ``index_topk * compress_rate``, matching the
        prefill port's degenerate-indexer assumption).
        """
        b, s, _ = hidden.shape  # s == 1

        q, kv_new = self._qkv(hidden, cos, sin)  # q [B,H,1,Dh], kv_new [B,1,1,Dh]
        kv = kv_cache.sliding.append(kv_new)  # [B, 1, L_sld, Dh]

        if self.compressor is not None:
            compressed = self.compressor.decode(hidden, cos_win, sin_win, kv_cache.compressor)
            if compressed is not None:
                kv = ttnn.concat([kv, compressed], dim=2)  # [B, 1, L_sld + n_win, Dh]
        _profile(self.device)

        mask = ttnn.zeros([1, 1, s, kv.shape[2]], ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device)
        attn = self._sdpa_decode(q, kv, mask)  # [B, H, 1, Dh]

        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim)
        attn = ttnn.transpose(attn, 1, 2)  # [B, 1, H, Dh]
        return self._grouped_output(attn)

    def decode_static(
        self,
        hidden: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        mask: ttnn.Tensor,
        scache: "_StaticLayerCache",
        sliding_pos: ttnn.Tensor,
        compress_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Trace-safe single-token decode against fixed-size in-place caches.

        Writes the new token's rotated K=V into the ``sliding_window``-sized ring
        buffer (slot = ``pos % window`` carried by ``sliding_pos``) and, for
        CSA/HCA layers, its compressor projections at absolute ``compress_pos``;
        attends the *whole* fixed cache (sliding slots ++ all compressor windows)
        under the supplied additive ``mask`` (zeros for valid slots / windows,
        ``_MASK_NEG`` for unwritten slots and not-yet-emittable windows). Equivalent
        to :meth:`decode` but with static shapes / addresses for a reusable trace.
        """
        q, kv_new = self._qkv(hidden, cos, sin)  # q [1,H,1,Dh], kv_new [1,1,1,Dh]
        _update_cache_at(scache.sliding, kv_new, sliding_pos)
        kv = scache.sliding  # [1, 1, window, Dh] (updated in place)

        if self.compressor is not None:
            compressed = self.compressor.decode_static(
                hidden, cos_win, sin_win, scache.compressor_kv, scache.compressor_gate, compress_pos
            )
            kv = ttnn.concat([kv, compressed], dim=2)  # [1, 1, window + n_win, Dh]

        attn = self._sdpa_decode(q, kv, mask)  # [1, H, 1, Dh]
        attn = _apply_rope(attn, cos, neg_sin, self.rot, self.rope_dim)
        attn = ttnn.transpose(attn, 1, 2)  # [1, 1, H, Dh]
        return self._grouped_output(attn)


# ---------------------------------------------------------------------------- #
# DeepSeek-V4-Flash Mixture-of-Experts (prefill)
#
# ttnn port of ``DeepseekV4SparseMoeBlock`` (and its ``DeepseekV4TopKRouter`` /
# ``DeepseekV4Experts`` / ``DeepseekV4MLP`` shared expert) from
# ``modular_deepseek_v4.py``. Scope is the standard top-k routed MoE block (the
# ``mlp_layer_types == "moe"`` path); the static ``hash_moe`` router is out of
# scope here (it only swaps the *which-experts* selection for a frozen
# ``tid2eid[input_ids]`` lookup, leaving the expert / shared-expert compute
# identical).
#
# Layout conventions, matching the reference:
#   B = batch, S = seq length, T = B*S flattened tokens, H = hidden_size,
#   E = num routed experts, I = moe_intermediate_size, k = num_experts_per_tok.
#
# The reference dispatches each token to its top-k experts and loops over the
# *hit* experts. We instead run a *dense* batched compute: every expert is
# evaluated for every token, then masked by the per-token routing weight (0 for
# unselected experts) and summed across the expert axis. This is the standard
# small-mesh ttnn MoE shape (cf. ``models/demos/gpt_oss``); it is mathematically
# identical to the gather/scatter reference because unselected experts get a
# routing weight of exactly 0.
# ---------------------------------------------------------------------------- #


class DeepSeekV4MLP(DeepSeekV4Module):
    """Dense SwiGLU MLP (matches ``DeepseekV4MLP`` / ``LlamaMLP``).

    Used as the always-on *shared expert*: ``down(silu(gate(x)) * up(x))`` with
    no clamp (the routed experts clamp; the shared expert does not).
    """

    def __init__(
        self,
        weights: dict,
        prefix: str,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        cache = _as_cache(cache)
        self.gate_proj = Linear(
            weights[f"{prefix}.gate_proj.weight"], device, cache.file(f"{prefix}.gate_proj"), dtype=weight_dtype
        )
        self.up_proj = Linear(
            weights[f"{prefix}.up_proj.weight"], device, cache.file(f"{prefix}.up_proj"), dtype=weight_dtype
        )
        self.down_proj = Linear(
            weights[f"{prefix}.down_proj.weight"], device, cache.file(f"{prefix}.down_proj"), dtype=weight_dtype
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.silu(self.gate_proj(x))
        return self.down_proj(ttnn.multiply(gate, self.up_proj(x)))


class DeepSeekV4TopKRouter(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4TopKRouter``.

    Produces a *dense* ``[1, 1, T, E]`` routing-weight tensor: ``sqrtsoftplus``
    of the gate logits gives per-expert scores; the top-k experts (by
    ``scores + e_score_correction_bias``) are selected via ``ttnn.topk`` +
    ``ttnn.scatter`` into a one-hot mask; the masked scores are renormalised to
    sum to 1 per token and scaled by ``routed_scaling_factor``. Unselected
    experts carry weight 0, which lets the dense expert compute drop them.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)
        self.gate = Linear(weights["gate.weight"], device, cache.file("gate"))
        bias = _materialize(
            weights["gate.e_score_correction_bias"], cache.file("gate.e_score_correction_bias"), ttnn.bfloat16
        )
        self.e_score_correction_bias = _load_weight(
            bias.reshape(1, 1, 1, self.num_experts) if bias is not None else None,
            device,
            cache_file_name=cache.file("gate.e_score_correction_bias"),
        )
        # Persistent scatter operands for the trace-safe decode path (T == 1).
        # ``ttnn.zeros`` / ``ttnn.ones`` host-init their buffers (a host->device
        # write that is illegal mid-capture), so the static router reuses these
        # pre-built constants instead of allocating + writing them each call.
        self._scatter_zeros = ttnn.zeros([1, 1, 1, self.num_experts], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
        self._scatter_ones = ttnn.ones([1, 1, 1, self.top_k], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)

    def forward(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """``x_flat`` is ``[1, 1, T, H]``; returns routing weights ``[1, 1, T, E]``."""
        logits = self.gate(x_flat)  # [1, 1, T, E]
        scores = ttnn.sqrt(ttnn.softplus(logits))
        biased = ttnn.add(scores, self.e_score_correction_bias)
        _profile(self.device)

        # Top-k selection -> one-hot mask. Scatter (rather than a >= threshold
        # compare) selects exactly k experts even if two scores collide under
        # bf16 rounding. Scatter wants ROW_MAJOR + a matching-rank index tensor.
        _, top_idx = ttnn.topk(biased, self.top_k, dim=-1)  # [1, 1, T, k]
        t = x_flat.shape[2]
        top_idx = ttnn.to_layout(top_idx, ttnn.ROW_MAJOR_LAYOUT)
        mask = ttnn.zeros([1, 1, t, self.num_experts], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.device)
        src = ttnn.ones([1, 1, t, self.top_k], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.device)
        mask = ttnn.scatter(mask, -1, top_idx, src)
        mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)

        # Weights are the *unbiased* scores gathered at the selected experts,
        # normalised per token, then scaled. Masking before the sum makes the
        # dense [1,1,T,E] tensor equal the reference's gathered/normalised one.
        selected = ttnn.multiply(scores, mask)
        denom = ttnn.add(ttnn.sum(selected, dim=-1, keepdim=True), 1.0e-20)
        return ttnn.multiply(ttnn.div(selected, denom), self.routed_scaling_factor)

    def forward_static(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """Trace-safe single-token (``T == 1``) top-k routing -> ``[1,1,1,E]``.

        Identical math to :meth:`forward`, but the scatter's zeros / ones operands
        are the persistent constants built at init rather than freshly
        ``ttnn.zeros`` / ``ttnn.ones`` tensors (whose host-init write is rejected
        during trace capture). Scatter allocates its own output, which is allowed.
        """
        logits = self.gate(x_flat)  # [1, 1, 1, E]
        scores = ttnn.sqrt(ttnn.softplus(logits))
        biased = ttnn.add(scores, self.e_score_correction_bias)

        _, top_idx = ttnn.topk(biased, self.top_k, dim=-1)  # [1, 1, 1, k]
        top_idx = ttnn.to_layout(top_idx, ttnn.ROW_MAJOR_LAYOUT)
        mask = ttnn.scatter(self._scatter_zeros, -1, top_idx, self._scatter_ones)
        mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)

        selected = ttnn.multiply(scores, mask)
        denom = ttnn.add(ttnn.sum(selected, dim=-1, keepdim=True), 1.0e-20)
        return ttnn.multiply(ttnn.div(selected, denom), self.routed_scaling_factor)


class DeepSeekV4HashRouter(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HashRouter`` (the first ``num_hash_layers`` MoE
    layers, paper §2.1).

    Expert *selection* is a frozen ``tid2eid[input_ids]`` lookup — a fixed
    token-id -> expert-id table — rather than a learned top-k argmax. The learned
    gate still produces the per-expert ``sqrtsoftplus`` scores that weight the
    selected experts; only the *which-experts* decision is static. As with
    :class:`DeepSeekV4TopKRouter` we emit a dense ``[1,1,T,E]`` weight tensor
    (selected experts carry their renormalised score, the rest are 0) so the same
    dense / preloaded expert compute consumes it.

    The selection mask is built host-side from ``input_ids`` (known at call time)
    — no on-device gather/scatter is needed since the table is fixed.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        cache = _as_cache(cache)
        self.gate = Linear(weights["gate.weight"], device, cache.file("gate"))
        # tid2eid [vocab, top_k]: frozen token-id -> expert-id table (host-side,
        # no tile cache) -- always materialise.
        tid = weights["gate.tid2eid"]
        tid = tid() if callable(tid) else tid
        self.tid2eid = tid.long()

    def forward(self, x_flat: ttnn.Tensor, input_ids: torch.Tensor) -> ttnn.Tensor:
        """``x_flat`` ``[1,1,T,H]`` and ``input_ids`` torch ``[..]`` (T tokens);
        returns dense routing weights ``[1,1,T,E]``."""
        logits = self.gate(x_flat)  # [1, 1, T, E]
        scores = ttnn.sqrt(ttnn.softplus(logits))
        t = x_flat.shape[2]
        _profile(self.device)

        # Static per-token expert selection -> host one-hot mask [1,1,T,E].
        eids = self.tid2eid[input_ids.reshape(-1).long()]  # [T, top_k]
        mask = torch.zeros(t, self.num_experts, dtype=torch.float32)
        mask.scatter_(1, eids, 1.0)
        mask_tt = ttnn.from_torch(
            mask.reshape(1, 1, t, self.num_experts), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        selected = ttnn.multiply(scores, mask_tt)
        denom = ttnn.add(ttnn.sum(selected, dim=-1, keepdim=True), 1.0e-20)
        return ttnn.multiply(ttnn.div(selected, denom), self.routed_scaling_factor)

    def selection_mask(self, token_id: int) -> torch.Tensor:
        """Host one-hot expert-selection mask ``[1, 1, 1, E]`` for ``token_id`` —
        the frozen ``tid2eid`` lookup, built on host (the traced decode writes it
        into a persistent device input each step)."""
        eids = self.tid2eid[int(token_id)].reshape(-1).long()
        mask = torch.zeros(self.num_experts, dtype=torch.float32)
        mask.scatter_(0, eids, 1.0)
        return mask.reshape(1, 1, 1, self.num_experts)

    def forward_static(self, x_flat: ttnn.Tensor, mask_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Trace-safe hash routing: ``mask_tt`` ``[1,1,1,E]`` is the (persistent,
        per-step) device selection mask from :meth:`selection_mask`; the gate score
        path stays on device. Returns dense routing weights ``[1,1,1,E]``."""
        logits = self.gate(x_flat)
        scores = ttnn.sqrt(ttnn.softplus(logits))
        selected = ttnn.multiply(scores, mask_tt)
        denom = ttnn.add(ttnn.sum(selected, dim=-1, keepdim=True), 1.0e-20)
        return ttnn.multiply(ttnn.div(selected, denom), self.routed_scaling_factor)


# --------------------------------------------------------------------------- #
# fused_experts (single-op decode path)
#
# ``ttnn.experimental.deepseek.moe.fused_experts`` runs the whole routed-expert
# FFN (gate_up + SwiGLU + down + routing-weighted accumulation) for one token in
# a single device op. It is hard-wired to the real V4-Flash sizes -- an 8x8 (64)
# compute grid where each core owns 2 output tiles, so the hidden size must be
# exactly ``_FUSED_HIDDEN`` (64 * 2 * 32) -- and is decode-only (``T == 1``). The
# weights must be DRAM ND-sharded with one shard per core (see below), a layout
# distinct from the plain matmul weights used by the prefill loop, so the decode
# path keeps its own copy.
# --------------------------------------------------------------------------- #
_FUSED_HIDDEN = 4096  # op requires H == 64 cores * 2 tiles * 32 = 4096
_FUSED_COLS_PER_CORE = 64  # SwiGLU output columns per core (2 tiles)
_FUSED_NUM_CORES = 64  # 8x8 compute grid
_FUSED_DRAM_BANKS = 8  # Blackhole DRAM banks (round-robin shard target)


def _interleave_gate_up(w: torch.Tensor, block: int = _FUSED_COLS_PER_CORE) -> torch.Tensor:
    """Permute a ``[K, 2I]`` gate_up weight into per-core ``[gate_block | up_block]``
    order so each ``[K, 2*block]`` DRAM shard holds a core's gate columns followed
    by its paired up columns (what ``fused_experts`` reads in a single NoC read).

    ``gate = w[:, :I]``, ``up = w[:, I:]``; output column ``c*2*block + h*block + t``
    maps to ``w[:, h*I + c*block + t]``.
    """
    k, two_i = w.shape
    intermediate = two_i // 2
    blocks = intermediate // block
    return w.reshape(k, 2, blocks, block).permute(0, 2, 1, 3).reshape(k, two_i).contiguous()


def _fused_nd_dram_config(rows: int, cols: int, shard_width: int) -> ttnn.MemoryConfig:
    """DRAM ND-shard config: ``rows x shard_width`` shards round-robined over the
    DRAM banks (one shard per compute core), as ``fused_experts`` expects."""
    assert cols % shard_width == 0, f"last dim {cols} must divide into shards of {shard_width}"
    dram_core_range_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0)) for bank in range(_FUSED_DRAM_BANKS)]
    )
    return ttnn.MemoryConfig(
        ttnn.BufferType.DRAM,
        ttnn.NdShardSpec(
            shard_shape=[rows, shard_width],
            grid=dram_core_range_set,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def _load_fused_weight(
    tensor: Optional[torch.Tensor],
    device: ttnn.MeshDevice,
    nd_config: ttnn.MemoryConfig,
    *,
    cache_file_name: Optional[str] = None,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
) -> ttnn.Tensor:
    """Load a ``fused_experts`` weight as a DRAM ND-sharded tensor.

    The tile cache cannot round-trip an ND-shard memory config (a cache *hit*
    reloads the tensor with its plain serialized spec), so the (interleaved)
    weight is cached in standard interleaved DRAM under its own cache entry and
    then resharded to the ND-shard layout on device.
    """
    sharded = ttnn.as_tensor(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=nd_config,
        cache_file_name=cache_file_name,
    )
    return sharded


class DeepSeekV4PreloadedExperts(DeepSeekV4Module):
    """Routed-experts compute via the single-op ``fused_experts`` kernel.

    The whole routed-expert FFN for one token (gate_up + SwiGLU + down +
    routing-weighted accumulation) runs in a single ``fused_experts`` device op.
    The op is hard-wired to the real V4-Flash sizes -- an 8x8 (64) compute grid
    where each core owns 2 output tiles, so ``H`` must be exactly ``_FUSED_HIDDEN``
    (``64 * 2 * 32 == 4096``) and ``I`` a multiple of the 64-column per-core
    slice. Both prefill and decode go through the op: it is natively single-token
    (``T == 1``), so **prefill is computed by decode** -- each of the ``T`` tokens
    runs as its own op and the per-token outputs are concatenated.

    Every expert is kept resident on device as DRAM ND-sharded weights (one shard
    per compute core), in low precision (``BFloat4_b`` by default; ~3.5 GB for the
    256 experts, a natural match for the MXFP4 checkpoint). At init it pulls each
    expert's dequantized weights from the host ``provider`` once, permutes the
    gate_up into the op's interleaved per-core layout, and uploads the ND-sharded
    tensors; ``forward`` then runs purely on device with no per-step host
    transfers beyond reading the (tiny) routing weights to pick the hit experts.

    ``provider(expert_idx) -> (gate_up [2I, H], down [H, I])`` returns host
    torch tensors (the HF packed layout: ``gate_up`` is ``cat([w_gate, w_up])``).
    Experts with zero total routing weight are skipped (matching the reference's
    ``hit`` set), so only the experts some token actually selected are computed.
    """

    def __init__(
        self,
        config,
        provider,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat4_b,
        cache: Optional[WeightCache] = None,
    ):
        self.device = device
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.intermediate = config.moe_intermediate_size
        self.hidden = config.hidden_size
        self.limit = config.swiglu_limit
        cache = _as_cache(cache)

        # ``fused_experts`` is hard-wired to the real V4-Flash sizes: ``H == 4096``
        # on the 64-core grid and ``I`` a multiple of the 64-column per-core slice.
        # There is no fallback path -- this class is for that config only.
        if self.hidden != _FUSED_HIDDEN or self.intermediate % _FUSED_COLS_PER_CORE != 0:
            raise ValueError(
                f"DeepSeekV4PreloadedExperts requires the fused_experts layout "
                f"(H == {_FUSED_HIDDEN}, I % {_FUSED_COLS_PER_CORE} == 0); "
                f"got H={self.hidden}, I={self.intermediate}"
            )
        gate_up_nd = _fused_nd_dram_config(self.hidden, 2 * self.intermediate, 2 * _FUSED_COLS_PER_CORE)
        down_nd = _fused_nd_dram_config(self.intermediate, self.hidden, self.hidden // _FUSED_NUM_CORES)

        # Upload every expert once as the op's DRAM ND-sharded weights (gate_up
        # interleaved per core, down ND-sharded), stored in low precision. With
        # caching enabled and a hit, the provider (and its expensive dequant) is
        # skipped entirely; the ND-shard layout can't round-trip the tile cache,
        # so the interleaved weight is cached in standard DRAM and resharded on
        # device (see :func:`_load_fused_weight`).
        self._gate_up_fused: list[ttnn.Tensor] = []
        self._down_fused: list[ttnn.Tensor] = []
        for e in range(self.num_experts):
            gu_f_name, dn_f_name = f"experts.{e}.gate_up_fused", f"experts.{e}.down_fused"
            need_torch = not (cache.hit(gu_f_name, dtype) and cache.hit(dn_f_name, dtype))
            if cache.require_cache and need_torch:
                raise RuntimeError(f"weight cache miss for routed expert {e} (gate_up/down) with require_cache=True")
            gate_up_w, down_w = provider(e) if need_torch else (None, None)
            # Provider gives gate_up [2I, H] / down [H, I]; transpose to matmul-ready
            # [H, 2I] / [I, H] (memoized so each is materialized at most once).
            gate_up_t = _memo((lambda gw=gate_up_w: gw.t().contiguous()) if gate_up_w is not None else (lambda: None))
            down_t = _memo((lambda dw=down_w: dw.t().contiguous()) if down_w is not None else (lambda: None))
            gu_il = _materialize(lambda: _interleave_gate_up(gate_up_t()), cache.file(gu_f_name), dtype)
            self._gate_up_fused.append(
                _load_fused_weight(gu_il, device, gate_up_nd, cache_file_name=cache.file(gu_f_name), dtype=dtype)
            )
            self._down_fused.append(
                _load_fused_weight(down_t(), device, down_nd, cache_file_name=cache.file(dn_f_name), dtype=dtype)
            )

    def _run_fused(self, x_tok: ttnn.Tensor, routing_row: ttnn.Tensor, num_experts: int) -> ttnn.Tensor:
        """Run ``fused_experts`` for one token. ``x_tok`` ``[1,1,1,H]`` (TILE) and
        ``routing_row`` a ROW_MAJOR routing slice; returns ``[1,1,1,H]``."""
        routing_row = ttnn.reshape(routing_row, [1, 1, 1, self.num_experts])
        out = ttnn.experimental.deepseek.moe.fused_experts(
            x_tok,
            routing_weights=routing_row,
            gate_up_weights=self._gate_up_fused,
            down_weights=self._down_fused,
            num_experts=num_experts,
            intermediate_size=self.intermediate,
            swiglu_limit=self.limit,
        )  # [1, 1, H]
        return ttnn.reshape(out, [1, 1, 1, self.hidden])

    def _decode_token(self, x_tok: ttnn.Tensor, rw_tok: ttnn.Tensor) -> ttnn.Tensor:
        """Run one token's routed FFN through ``fused_experts``.

        ``x_tok`` ``[1,1,1,H]`` and ``rw_tok`` the host routing-weight row ``[E]``;
        returns ``[1,1,1,H]``. The op finds the active (non-zero) experts from the
        routing row itself, so we only pass ``num_experts`` = the hit count.
        """
        routing_row = ttnn.to_layout(rw_tok, ttnn.ROW_MAJOR_LAYOUT)
        out = self._run_fused(x_tok, routing_row, 6)
        _profile(self.device)
        return out

    def forward(self, x_flat: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """``x_flat`` ``[1,1,T,H]`` and ``routing_weights`` ``[1,1,T,E]``; returns ``[1,1,T,H]``.

        Every token runs as its own single-token ``fused_experts`` op (the op is
        natively ``T == 1``), so prefill is computed by decode: the ``T`` per-token
        outputs are concatenated back into ``[1,1,T,H]``.
        """
        t = x_flat.shape[2]
        _profile(self.device)

        if t == 1:
            return self._decode_token(x_flat, routing_weights)

        # Prefill loops the single-token op over ``T`` tokens. Slicing a single,
        # non-tile-aligned row out of a TILE tensor forces an untilize/unpad +
        # re-tilize per token (and the routing row needs a per-token untilize for
        # the ROW_MAJOR op input). Hoist both layout conversions out of the loop:
        # untilize ``x_flat`` / ``routing_weights`` once, slice rows cheaply in
        # ROW_MAJOR, and tilize only the small ``[1,1,1,H]`` activation row the op
        # actually consumes.
        x_rm = ttnn.to_layout(x_flat, ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, T, H]
        rw_rm = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)  # [1, 1, T, E]
        outs = []
        for ti in range(t):
            x_tok_rm = ttnn.slice(x_rm, [0, 0, ti, 0], [1, 1, ti + 1, self.hidden])
            x_tok = ttnn.to_layout(x_tok_rm, ttnn.TILE_LAYOUT)
            routing_row = ttnn.slice(rw_rm, [0, 0, ti, 0], [1, 1, ti + 1, self.num_experts])
            outs.append(self._run_fused(x_tok, routing_row, 6))
        return ttnn.concat(outs, dim=2)  # [1, 1, T, H]

    def decode_static(self, x_tok: ttnn.Tensor, routing_weights: ttnn.Tensor) -> ttnn.Tensor:
        """Trace-safe single-token routed FFN. ``x_tok`` ``[1,1,1,H]`` and
        ``routing_weights`` ``[1,1,1,E]`` (a device tensor); returns ``[1,1,1,H]``.

        Unlike :meth:`forward`, the active experts are *not* read back to host:
        ``fused_experts`` finds the non-zero experts from the routing row on device
        and ``num_experts`` is fixed to ``num_experts_per_tok`` (the router always
        selects exactly ``top_k``), so the op's program — and hence the trace — is
        invariant across steps.
        """
        routing_row = ttnn.to_layout(routing_weights, ttnn.ROW_MAJOR_LAYOUT)
        routing_row = ttnn.reshape(routing_row, [1, 1, 1, self.num_experts])
        out = ttnn.experimental.deepseek.moe.fused_experts(
            x_tok,
            routing_weights=routing_row,
            gate_up_weights=self._gate_up_fused,
            down_weights=self._down_fused,
            num_experts=self.top_k,
            intermediate_size=self.intermediate,
            swiglu_limit=self.limit,
        )  # [1, 1, H]
        return ttnn.reshape(out, [1, 1, 1, self.hidden])


class DeepSeekV4SparseMoeBlock(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4SparseMoeBlock`` (standard ``moe`` layer).

    ``routed = experts(router(x)) ; return routed + shared_experts(x)``.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        experts,
        gate=None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.hidden = config.hidden_size
        cache = _as_cache(cache)
        # ``gate`` may be injected (e.g. a :class:`DeepSeekV4HashRouter` for the
        # first ``num_hash_layers`` layers); otherwise the learned top-k router.
        self.gate = gate if gate is not None else DeepSeekV4TopKRouter(config, weights, device, cache=cache)
        self.is_hash = isinstance(self.gate, DeepSeekV4HashRouter)
        # The routed-expert compute (a :class:`DeepSeekV4PreloadedExperts` keeping
        # all 256 experts resident on device in BFloat4_b) is always injected.
        self.experts = experts
        self.shared_experts = DeepSeekV4MLP(weights, "shared_experts", device, cache=cache, weight_dtype=weight_dtype)

    def forward(self, hidden: ttnn.Tensor, input_ids: Optional[torch.Tensor] = None) -> ttnn.Tensor:
        """``hidden`` ``[B, S, H]`` -> ``[B, S, H]``. ``input_ids`` is required
        only for hash-routed layers (frozen ``tid2eid`` selection)."""
        b, s, h = hidden.shape
        x_flat = ttnn.reshape(hidden, [1, 1, b * s, h])
        _profile(self.device)

        with _region("MOE_ROUTER"):
            if self.is_hash:
                routing_weights = self.gate(x_flat, input_ids)  # [1, 1, T, E]
            else:
                routing_weights = self.gate(x_flat)  # [1, 1, T, E]
        _profile(self.device)

        with _region("MOE_EXPERTS"):
            routed = self.experts(x_flat, routing_weights)  # [1, 1, T, H]
            routed = ttnn.reshape(routed, [b, s, h])
        _profile(self.device)

        with _region("MOE_SHARED"):
            shared = self.shared_experts(hidden)  # [B, S, H]

        _profile(self.device)

        return ttnn.add(routed, shared)

    def decode_static(self, hidden: ttnn.Tensor, hash_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """Trace-safe single-token MoE. ``hidden`` ``[1, 1, H]`` -> ``[1, 1, H]``.

        Routing stays entirely on device: the learned top-k router is already
        host-sync-free, and hash layers consume the persistent ``hash_mask``
        ``[1,1,1,E]`` device input (see :meth:`DeepSeekV4HashRouter.forward_static`).
        The routed FFN runs through the no-host-readback fused-experts decode path.
        """
        h = hidden.shape[-1]
        x_flat = ttnn.reshape(hidden, [1, 1, 1, h])
        if self.is_hash:
            routing_weights = self.gate.forward_static(x_flat, hash_mask)
        else:
            routing_weights = self.gate.forward_static(x_flat)
        routed = self.experts.decode_static(x_flat, routing_weights)  # [1, 1, 1, H]
        routed = ttnn.reshape(routed, [1, 1, h])
        shared = self.shared_experts(hidden)
        return ttnn.add(routed, shared)


class DeepSeekV4HyperConnection(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HyperConnection`` (Manifold-Constrained Hyper-
    Connections / mHC).

    Given the residual stream stack ``hidden_streams [B, S, H, D]`` (``H`` =
    ``hc_mult`` parallel streams, ``D`` = ``hidden_size``) it returns the triple
    ``(post, comb, collapsed)``:
      * ``collapsed [B, S, D]`` -- the ``pre``-weighted collapse of the streams
        into a single sequence (the sublayer input),
      * ``post [B, S, H]`` -- the sublayer-output placement weights
        (``2 * sigmoid(.)``),
      * ``comb [B, S, H, H]`` -- the stream-mixing matrix projected onto the
        doubly-stochastic manifold by ``hc_sinkhorn_iters`` Sinkhorn-Knopp steps.

    The learned ``fn`` / ``base`` / ``scale`` parameters are split host-side into
    their ``pre`` / ``post`` / ``comb`` parts (and ``fn`` run as three separate
    linears) so we never sub-tile-slice the packed ``(2+H)*H``-wide projection.
    See ``modular_deepseek_v4.py`` for the reference math.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        hc = self.hc
        # ``fn`` / ``base`` are one packed checkpoint tensor each, sliced into
        # pre [H] / post [H] / comb [H*H] parts; memoize so a cache miss reads
        # each source once across its slices. ``scale`` is 3 host scalars (no
        # tile cache), so it is always materialised.
        fn = _memo(weights["fn"])  # [(2+H)*H, H*D]
        base = _memo(weights["base"])  # [(2+H)*H]
        scale_src = weights["scale"]
        scale = (scale_src() if callable(scale_src) else scale_src).flatten().tolist()  # 3 learned scalars

        self.fn_pre = Linear(lambda: fn()[:hc], device, cache.file("fn_pre"))
        self.fn_post = Linear(lambda: fn()[hc : 2 * hc], device, cache.file("fn_post"))
        self.fn_comb = Linear(lambda: fn()[2 * hc : 2 * hc + hc * hc], device, cache.file("fn_comb"))
        self.pre_b = _load_weight(
            _materialize(lambda: base()[:hc].reshape(1, 1, 1, hc), cache.file("pre_b"), ttnn.bfloat16),
            device,
            cache_file_name=cache.file("pre_b"),
        )
        self.post_b = _load_weight(
            _materialize(lambda: base()[hc : 2 * hc].reshape(1, 1, 1, hc), cache.file("post_b"), ttnn.bfloat16),
            device,
            cache_file_name=cache.file("post_b"),
        )
        self.comb_b = _load_weight(
            _materialize(
                lambda: base()[2 * hc : 2 * hc + hc * hc].reshape(1, 1, 1, hc * hc),
                cache.file("comb_b"),
                ttnn.bfloat16,
            ),
            device,
            cache_file_name=cache.file("comb_b"),
        )
        self.pre_scale, self.post_scale, self.comb_scale = (float(scale[0]), float(scale[1]), float(scale[2]))

    def forward(self, hidden_streams: ttnn.Tensor):
        """``hidden_streams`` ``[B, S, H, D]`` -> ``(post [B,S,H], comb [B,S,H,H], collapsed [B,S,D])``."""
        b, s, hc, d = hidden_streams.shape
        t = b * s

        # Flatten streams to [1,1,T,H*D] and unweighted-RMSNorm over H*D.
        flat = ttnn.reshape(hidden_streams, [1, 1, t, hc * d])
        flat = _rms_norm_unweighted(flat, self.norm_eps)

        pre_w = self.fn_pre(flat)  # [1,1,T,H]
        post_w = self.fn_post(flat)  # [1,1,T,H]
        comb_w = self.fn_comb(flat)  # [1,1,T,H*H]
        _profile(self.device)

        # pre = sigmoid(w*scale + b) + eps ; post = 2*sigmoid(w*scale + b).
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(pre_w, self.pre_scale), self.pre_b)), self.eps)
        post = ttnn.multiply(ttnn.sigmoid(ttnn.add(ttnn.multiply(post_w, self.post_scale), self.post_b)), 2.0)

        # comb logits -> [1,T,H,H]; softmax over last dim, then Sinkhorn (alternate
        # row/col normalisation) onto the doubly-stochastic manifold.
        comb_logits = ttnn.add(ttnn.multiply(comb_w, self.comb_scale), self.comb_b)  # [1,1,T,H*H]
        comb_logits = ttnn.reshape(comb_logits, [1, t, hc, hc])
        comb = ttnn.add(ttnn.softmax(comb_logits, dim=-1), self.eps)
        comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), self.eps))  # column
        for _ in range(self.iters - 1):
            comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-1, keepdim=True), self.eps))  # row
            comb = ttnn.div(comb, ttnn.add(ttnn.sum(comb, dim=-2, keepdim=True), self.eps))  # column

        # collapsed = sum_h pre[..,h] * hidden_streams[..,h,:]  (weighted stream sum).
        hs = ttnn.reshape(hidden_streams, [1, t, hc, d])
        pre_col = ttnn.reshape(pre, [1, t, hc, 1])
        collapsed = ttnn.sum(ttnn.multiply(hs, pre_col), dim=-2, keepdim=True)  # [1,T,1,D]

        post = ttnn.reshape(post, [b, s, hc])
        comb = ttnn.reshape(comb, [b, s, hc, hc])
        collapsed = ttnn.reshape(collapsed, [b, s, d])
        return post, comb, collapsed


class DeepSeekV4HyperHead(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4HyperHead`` (final HC-stream collapse).

    Collapses the ``hc_mult`` residual streams ``[B, S, H, D]`` into a single
    ``[B, S, D]`` sequence before the model's shared RMSNorm + ``lm_head``::

        flat  = unweighted_rmsnorm(streams.flatten(2))
        pre   = sigmoid(hc_fn @ flat * hc_scale + hc_base) + eps
        out   = (pre[..,None] * streams).sum(dim=2)

    ``weights`` keys: ``hc_fn`` ``[H, H*D]``, ``hc_base`` ``[H]``, ``hc_scale``
    (scalar). Unlike :class:`DeepSeekV4HyperConnection` there is no ``post`` /
    ``comb`` placement: the head only produces the collapsed sequence.
    """

    def __init__(self, config, weights: dict, device: ttnn.MeshDevice, cache: Optional[WeightCache] = None):
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        self.fn = Linear(weights["hc_fn"], device, cache.file("hc_fn"))  # [H, H*D]
        base_src = weights["hc_base"]
        self.base = _load_weight(
            _materialize(
                lambda: (base_src() if callable(base_src) else base_src).reshape(1, 1, 1, self.hc),
                cache.file("hc_base"),
                ttnn.bfloat16,
            ),
            device,
            cache_file_name=cache.file("hc_base"),
        )
        # hc_scale is a host scalar (no tile cache) -- always materialise.
        scale_src = weights["hc_scale"]
        self.scale = float((scale_src() if callable(scale_src) else scale_src).flatten().tolist()[0])

    def forward(self, hidden_streams: ttnn.Tensor) -> ttnn.Tensor:
        """``hidden_streams`` ``[B, S, H, D]`` -> ``[B, S, D]``."""
        b, s, hc, d = hidden_streams.shape
        t = b * s

        flat = ttnn.reshape(hidden_streams, [1, 1, t, hc * d])
        flat = _rms_norm_unweighted(flat, self.norm_eps)

        mixes = self.fn(flat)  # [1,1,T,H]
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(mixes, self.scale), self.base)), self.eps)

        hs = ttnn.reshape(hidden_streams, [1, t, hc, d])
        pre_col = ttnn.reshape(pre, [1, t, hc, 1])
        out = ttnn.sum(ttnn.multiply(hs, pre_col), dim=-2, keepdim=True)  # [1,T,1,D]
        return ttnn.reshape(out, [b, s, d])


def _strip_prefix(weights: dict, prefix: str) -> dict:
    """Sub-dict of ``weights`` whose keys start with ``prefix.`` (prefix stripped)."""
    p = f"{prefix}."
    return {k[len(p) :]: v for k, v in weights.items() if k.startswith(p)}


class DeepSeekV4DecoderLayer(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4DecoderLayer`` (decode).

    The residual is a stack of ``hc_mult`` parallel streams kept in
    ``[B, S, H, D]`` (``H`` = ``hc_mult``, ``D`` = ``hidden_size``) throughout the
    block, mixed in/out by two :class:`DeepSeekV4HyperConnection` modules. For
    each sublayer (attention, then MoE) the matching HC collapses the streams
    into the sublayer input, and the sublayer output is folded back into the
    streams via the learned ``post`` placement weights plus the Sinkhorn
    ``comb`` stream-mixing matrix::

        post, comb, collapsed = hc(streams)
        out = sublayer(norm(collapsed))
        streams = post[..,None] * out[..,None,:] + (comb.T @ streams)

    ``comb`` is consumed *transposed* (mix over the first hc axis), matching the
    reference ``torch.matmul(comb.transpose(-1, -2), streams)``.

    ``weights`` keys mirror the HF decoder-layer param names: ``self_attn.*``,
    ``mlp.*``, ``attn_hc.{fn,base,scale}``, ``ffn_hc.{fn,base,scale}``,
    ``input_layernorm.weight``, ``post_attention_layernorm.weight``. RoPE tables
    and the additive mask are inputs (built by the surrounding model / test, per
    :func:`make_rope_table`), exactly as for :class:`DeepSeekV4Attention`.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device: ttnn.MeshDevice,
        experts=None,
        gate=None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        eps = config.rms_norm_eps
        cache = _as_cache(cache)

        self.self_attn = DeepSeekV4Attention(
            config,
            layer_idx,
            _strip_prefix(weights, "self_attn"),
            device,
            cache=cache.sub("self_attn"),
            weight_dtype=weight_dtype,
        )
        self.mlp = DeepSeekV4SparseMoeBlock(
            config,
            _strip_prefix(weights, "mlp"),
            device,
            experts=experts,
            gate=gate,
            cache=cache.sub("mlp"),
            weight_dtype=weight_dtype,
        )
        self.input_layernorm = DeepSeekV4RMSNorm(
            weights["input_layernorm.weight"], eps, device, cache.file("input_layernorm")
        )
        self.post_attention_layernorm = DeepSeekV4RMSNorm(
            weights["post_attention_layernorm.weight"], eps, device, cache.file("post_attention_layernorm")
        )
        self.attn_hc = DeepSeekV4HyperConnection(
            config, _strip_prefix(weights, "attn_hc"), device, cache=cache.sub("attn_hc")
        )
        self.ffn_hc = DeepSeekV4HyperConnection(
            config, _strip_prefix(weights, "ffn_hc"), device, cache=cache.sub("ffn_hc")
        )
        _profile(self.device)

    def _mix(
        self, post: ttnn.Tensor, comb: ttnn.Tensor, sublayer_out: ttnn.Tensor, streams: ttnn.Tensor
    ) -> ttnn.Tensor:
        """``post[..,None] * out[..,None,:] + comb.T @ streams`` -> new streams.

        ``post`` ``[B,S,H]``, ``comb`` ``[B,S,H,H]``, ``sublayer_out`` ``[B,S,D]``,
        ``streams`` ``[B,S,H,D]``; returns ``[B,S,H,D]``.
        """
        b, s, hc, d = streams.shape
        t = b * s
        _profile(self.device)

        # placement = post.unsqueeze(-1) * sublayer_out.unsqueeze(-2) -> [1,T,H,D].
        out = ttnn.reshape(sublayer_out, [1, t, 1, d])
        out = ttnn.repeat(out, ttnn.Shape([1, 1, hc, 1]))  # broadcast over the stream axis
        placement = ttnn.multiply(out, ttnn.reshape(post, [1, t, hc, 1]))

        # mix = matmul(comb.transpose(-1, -2), streams): sum over the FIRST hc axis.
        comb_t = ttnn.transpose(ttnn.reshape(comb, [1, t, hc, hc]), -2, -1)
        mixed = ttnn.matmul(comb_t, ttnn.reshape(streams, [1, t, hc, d]), compute_kernel_config=_HIFI4)

        return ttnn.reshape(ttnn.add(placement, mixed), [b, s, hc, d])

    def decode(
        self,
        hidden_streams: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        kv_cache: "_LayerKVCache",
        input_ids: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Single-token decode: ``hidden_streams`` ``[B, 1, hc_mult, D]`` -> same.

        Everything outside attention (hyper-connections, norms, MoE / MLP) is
        per-token; attention runs against the running ``kv_cache``.
        """
        with _region("ATTN_HC"):
            post, comb, collapsed = self.attn_hc(hidden_streams)
        with _region("INPUT_NORM"):
            normed = self.input_layernorm(collapsed)
        with _region("ATTENTION"):
            attn_out = self.self_attn.decode(normed, cos, sin, neg_sin, cos_win, sin_win, kv_cache)
        with _region("ATTN_MIX"):
            hidden_streams = self._mix(post, comb, attn_out, hidden_streams)
        _profile(self.device)
        with _region("FFN_HC"):
            post, comb, collapsed = self.ffn_hc(hidden_streams)
        with _region("POST_NORM"):
            normed = self.post_attention_layernorm(collapsed)
        with _region("MOE"):
            mlp_out = self.mlp(normed, input_ids=input_ids)
        with _region("FFN_MIX"):
            return self._mix(post, comb, mlp_out, hidden_streams)

    def decode_static(
        self,
        hidden_streams: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        neg_sin: ttnn.Tensor,
        cos_win: ttnn.Tensor | None,
        sin_win: ttnn.Tensor | None,
        mask: ttnn.Tensor,
        scache: "_StaticLayerCache",
        sliding_pos: ttnn.Tensor,
        compress_pos: ttnn.Tensor,
        hash_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Trace-safe single-token decode (see :meth:`decode`). Uses the fixed-size
        in-place attention cache + the host-sync-free MoE so the whole block can be
        captured into a reusable ``ttnn`` trace."""
        post, comb, collapsed = self.attn_hc(hidden_streams)
        attn_out = self.self_attn.decode_static(
            self.input_layernorm(collapsed),
            cos,
            sin,
            neg_sin,
            cos_win,
            sin_win,
            mask,
            scache,
            sliding_pos,
            compress_pos,
        )
        hidden_streams = self._mix(post, comb, attn_out, hidden_streams)
        post, comb, collapsed = self.ffn_hc(hidden_streams)
        mlp_out = self.mlp.decode_static(self.post_attention_layernorm(collapsed), hash_mask=hash_mask)
        return self._mix(post, comb, mlp_out, hidden_streams)


# ---------------------------------------------------------------------------- #
# DeepSeek-V4-Flash full model (prefill, ``past_key_values is None``)
#
# ttnn port of ``DeepseekV4Model`` from ``modular_deepseek_v4.py``. Wires the
# embedding, the stack of :class:`DeepSeekV4DecoderLayer`s, the final
# :class:`DeepSeekV4HyperHead` stream-collapse and the model's shared RMSNorm
# into one module driven straight off the safetensors checkpoint (via
# :class:`DeepseekV4WeightLoader` + the ``quant`` dequantizers).
#
# Differences from the reference, all forced by the prefill-only / on-device
# scope already established by the sub-modules in this file:
#   * The rotary tables are *inputs* (built host-side, e.g. by the YaRN rotary
#     in the system interpreter — see ``test_bf4_decode_demo.py``) rather than
#     produced by an owned ``DeepseekV4RotaryEmbedding``; ttnn has no rope-init.
#   * The additive sliding-window / compressed-window masks are built here on
#     host (mirroring ``create_sliding_window_causal_mask`` + the compressors'
#     ``block_bias``), since the device attention consumes a plain additive mask.
#   * Every layer's weights are resident at once (the reference also holds the
#     whole stack); on the real 43-layer checkpoint cap with ``max_layers`` /
#     a populated ``cache`` or run the per-layer load/free loop in the demo.
# ---------------------------------------------------------------------------- #


def _sliding_causal_mask(seq_len: int, sliding_window: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive ``[1, 1, S, S]`` sliding-window causal mask (0 keep / ``_MASK_NEG``)."""
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    keep = (j <= i) & (i - j < sliding_window)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype).masked_fill(~keep, _MASK_NEG)
    return mask.view(1, 1, seq_len, seq_len)


def _block_bias(seq_len: int, n_windows: int, compress_rate: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Additive ``[1, 1, S, n_windows]`` causal block bias over compressed windows.

    Query ``t`` may attend compressed entry ``w`` iff ``w < (t + 1) // compress_rate``
    — the degenerate CSA/HCA top-k for ``seq_len <= index_topk * compress_rate``.
    """
    position_ids = torch.arange(seq_len).unsqueeze(0)
    entry = torch.arange(n_windows).view(1, 1, 1, n_windows)
    threshold = ((position_ids + 1) // compress_rate).view(1, 1, seq_len, 1)
    bias = torch.zeros(1, 1, seq_len, n_windows, dtype=dtype)
    return bias.masked_fill(entry >= threshold, _MASK_NEG)


class DeepSeekV4Model(DeepSeekV4Module):
    """ttnn port of ``DeepseekV4Model`` (prefill).

    Builds the embedding, the ``num_hidden_layers`` decoder stack, the final
    :class:`DeepSeekV4HyperHead` and the shared RMSNorm from the checkpoint, then
    runs the V4 forward: embed the ids, expand to the ``hc_mult`` residual-stream
    stack, run every decoder layer (building each layer's RoPE tables + additive
    mask from the supplied ``rope`` bundle), collapse the streams and normalise.

    ``rope`` matches the bundle emitted by the reference rotary (see
    ``test_bf4_decode_demo.py``)::

        rope["main"]    = (cos_half, sin_half)          # sliding layers
        rope["compress"]= (cos_half, sin_half)          # CSA / HCA layers
        rope["win"][cr] = (cos_half, sin_half)          # per compress-rate windows

    ``forward`` returns the model's ``last_hidden_state`` ``[B, S, hidden_size]``
    (the reference's pre-``lm_head`` output); apply an external ``lm_head``
    :class:`Linear` for logits.
    """

    def __init__(
        self,
        config,
        loader: DeepseekV4WeightLoader,
        full_device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        cache_dir: Optional[str] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat4_b,
        max_layers: Optional[int] = None,
        use_submeshes: bool = False,
        require_cache: bool = False,
    ):
        """Build the V4-Flash model off the checkpoint.

        Caching: pass either a pre-built ``cache`` :class:`WeightCache` or a
        ``cache_dir`` (the model builds ``WeightCache(cache_dir)`` and owns the
        per-layer ``layers.N`` / head namespacing internally, so callers no longer
        repeat the ``WeightCache(...).sub("layers.N")`` dance). ``None`` for both
        disables caching (every weight is converted from the checkpoint).

        ``require_cache=True`` asserts the converted-tile cache is fully populated:
        any tile-cached weight that would otherwise be (re)loaded from the HF
        checkpoint raises instead. The small host-side scalars (attention sinks,
        the HC ``scale`` triplets, the hash router's ``tid2eid`` table) and the
        locally-computed RoPE rotate matrix have no tile cache by design and are
        always materialised, so they are exempt.
        """
        self.config = config
        self.loader = loader
        self.weight_dtype = weight_dtype
        if cache is None and cache_dir is not None:
            cache = WeightCache(cache_dir)
        cache = _as_cache(cache)
        if require_cache:
            if not cache.path:
                raise ValueError(
                    "require_cache=True needs a populated cache; pass cache=WeightCache(dir) or cache_dir=..."
                )
            cache = cache.require(True)
        self.cache = cache
        self.require_cache = require_cache

        self.use_submeshes = use_submeshes
        self.num_submeshes = full_device.get_num_devices()
        if use_submeshes:
            logger.info(f"Using submeshes: {self.num_submeshes}")
            full_device.reshape(ttnn.MeshShape(1, full_device.get_num_devices()))
            self.submeshes = []
            for i in range(self.num_submeshes):
                self.submeshes.append(full_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, i)))
            self.first_device = self.submeshes[0]
            self.last_device = self.submeshes[-1]

            # Create socket pairs between submeshes for copying hidden_states .
            # One pair per (from_id, to_id) with from_id != to_id; reused for all forward passes.
            num_submeshes = full_device.get_num_devices()
            self.submesh_socket_pairs = {}
            socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16 * 1024)
            for from_id in range(num_submeshes - 1):
                to_id = from_id + 1
                from_submesh = self.submeshes[from_id]
                to_submesh = self.submeshes[to_id]
                socket_connections = []
                for coord in ttnn.MeshCoordinateRange(from_submesh.shape):
                    socket_connections.append(
                        ttnn.SocketConnection(
                            ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
                            ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
                        )
                    )
                socket_config = ttnn.SocketConfig(socket_connections, socket_memconfig)
                sender_socket, receiver_socket = ttnn.create_socket_pair(from_submesh, to_submesh, socket_config)
                self.submesh_socket_pairs[(from_id, to_id)] = (sender_socket, receiver_socket)
        else:
            self.first_device = full_device
            self.last_device = full_device

        n = config.num_hidden_layers if max_layers is None else min(max_layers, config.num_hidden_layers)
        self.num_layers = n

        self.embed_tokens = DeepSeekV4Embedding(loader, self.first_device, cache=cache)

        self.layers: list[DeepSeekV4DecoderLayer] = []
        self.layer_devices: list[ttnn.MeshDevice] = []
        self.layers_per_device = 6  # 43 layers across 8 devices.
        for li in range(n):
            if self.use_submeshes:
                layer_device_id = li // self.layers_per_device
                current_device = self.submeshes[layer_device_id]
                logger.info(f"Layer {li} is on device {layer_device_id}")
            else:
                current_device = self.device
            self.layer_devices.append(current_device)
            layer_type = config.layer_types[li]
            is_hash = config.mlp_layer_types[li] == "hash_moe"
            layer_cache = cache.sub(f"layers.{li}")
            weights = self._build_layer_weights(li, layer_type, is_hash)
            gate = self._hash_gate(li) if is_hash else None
            experts = DeepSeekV4PreloadedExperts(
                config,
                self._expert_provider(li),
                current_device,
                dtype=weight_dtype,
                cache=layer_cache.sub("mlp"),
            )
            self.layers.append(
                DeepSeekV4DecoderLayer(
                    config,
                    li,
                    weights,
                    current_device,
                    experts=experts,
                    gate=gate,
                    cache=layer_cache,
                    weight_dtype=weight_dtype,
                )
            )
            _profile(current_device)

        # The head (hc_head / norm / external lm_head) must live where the *last*
        # decoder layer's output lands, not unconditionally on the final submesh —
        # otherwise a capped (``max_layers``) stack would end on a lower submesh
        # than the head and mismatch devices.
        if self.layer_devices:
            self.last_device = self.layer_devices[-1]

        # Per-layer decode state (sliding K=V + optional compressor projections).
        self.sliding_window = config.sliding_window
        self.kv_caches: list[_LayerKVCache] = self._new_caches()

        self.hc_head = DeepSeekV4HyperHead(
            config,
            {
                "hc_fn": self._thunk("hc_head.hc_fn"),
                "hc_base": self._thunk("hc_head.hc_base"),
                "hc_scale": self._thunk("hc_head.hc_scale"),
            },
            self.last_device,
            cache=cache.sub("hc_head"),
        )
        self.norm = DeepSeekV4RMSNorm(
            self._thunk("norm.weight"), config.rms_norm_eps, self.last_device, cache.file("norm")
        )

    # -- weight plumbing (lazy dequant; a populated tile cache skips the read) -- #
    def _thunk(self, name: str):
        loader = self.loader
        return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))

    @staticmethod
    def _attn_keys(layer_type: str) -> list[str]:
        keys = [
            "q_a_proj.weight",
            "q_a_norm.weight",
            "q_b_proj.weight",
            "kv_proj.weight",
            "kv_norm.weight",
            "o_a_proj.weight",
            "o_b_proj.weight",
            "sinks",
        ]
        if layer_type != "sliding_attention":
            keys += [
                "compressor.kv_proj.weight",
                "compressor.gate_proj.weight",
                "compressor.kv_norm.weight",
                "compressor.position_bias",
            ]
        return keys

    def _build_layer_weights(self, layer_idx: int, layer_type: str, is_hash: bool) -> dict:
        weights: dict = {}
        for k in self._attn_keys(layer_type):
            weights[f"self_attn.{k}"] = self._thunk(f"layers.{layer_idx}.self_attn.{k}")
        weights["mlp.gate.weight"] = self._thunk(f"layers.{layer_idx}.mlp.gate.weight")
        if not is_hash:
            weights["mlp.gate.e_score_correction_bias"] = self._thunk(
                f"layers.{layer_idx}.mlp.gate.e_score_correction_bias"
            )
        for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            weights[f"mlp.shared_experts.{k}"] = self._thunk(f"layers.{layer_idx}.mlp.shared_experts.{k}")
        for hc in ("attn_hc", "ffn_hc"):
            for p in ("fn", "base", "scale"):
                weights[f"{hc}.{p}"] = self._thunk(f"layers.{layer_idx}.{hc}.{p}")
        for k in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            weights[k] = self._thunk(f"layers.{layer_idx}.{k}")
        return weights

    def _expert_provider(self, layer_idx: int):
        def provider(e: int):
            base = f"layers.{layer_idx}.mlp.experts.{e}"
            gate = self._thunk(f"{base}.gate_proj.weight")()
            up = self._thunk(f"{base}.up_proj.weight")()
            down = self._thunk(f"{base}.down_proj.weight")()
            return torch.cat([gate, up], dim=0).float(), down.float()

        return provider

    def _hash_gate(self, layer_idx: int) -> DeepSeekV4HashRouter:
        weights = {
            "gate.weight": self._thunk(f"layers.{layer_idx}.mlp.gate.weight"),
            "gate.tid2eid": self.loader.get_tensor(f"layers.{layer_idx}.mlp.gate.tid2eid").long(),
        }
        if self.use_submeshes:
            this_device = self.submeshes[layer_idx // self.layers_per_device]
        else:
            this_device = self.first_device
        return DeepSeekV4HashRouter(self.config, weights, this_device)

    # -- decode KV-cache state -------------------------------------------------- #
    def _new_caches(self) -> list["_LayerKVCache"]:
        return [
            _LayerKVCache(self.sliding_window, self.config.layer_types[li] != "sliding_attention")
            for li in range(self.num_layers)
        ]

    def reset_caches(self) -> None:
        """Drop all per-layer decode state (call before decoding a fresh sequence)."""
        self.kv_caches = self._new_caches()

    # -- per-layer RoPE tables / masks ------------------------------------------ #
    def _to_tt(self, t: torch.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
        _profile(device)

        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _rope_rows_decode(
        self, rope: dict, pos: int, layer_type: str, compress_rate: Optional[int], cache: dict, device: ttnn.MeshDevice
    ):
        """Single-position RoPE rows for a decode step.

        Returns ``(cos, sin, neg_sin, cos_win, sin_win)`` where ``cos/sin/neg_sin``
        are the one ``[1,1,1,Rd]`` rows at absolute position ``pos`` and the window
        tables cover the ``(pos + 1) // compress_rate`` currently-emittable windows
        (``None`` when none / a sliding layer).
        """
        key = f'{"sliding" if layer_type == "sliding_attention" else compress_rate}_{device.id()}'
        if key in cache:
            return cache[key]
        cos_h, sin_h = rope["main"] if layer_type == "sliding_attention" else rope["compress"]
        cos_row, sin_row = cos_h[pos : pos + 1], sin_h[pos : pos + 1]
        cos_full, sin_full = make_rope_table(cos_row, sin_row)
        cos_tt = self._to_tt(cos_full, device)
        sin_tt = self._to_tt(sin_full, device)
        neg_sin_tt = self._to_tt(-sin_full, device)

        cos_win_tt = sin_win_tt = None
        if layer_type != "sliding_attention":
            n_win = (pos + 1) // compress_rate
            if n_win > 0:
                cw_h, sw_h = rope["win"][compress_rate]
                cw, sw = make_rope_table(cw_h[:n_win], sw_h[:n_win])
                cos_win_tt = self._to_tt(cw, device)
                sin_win_tt = self._to_tt(sw, device)
        out = (cos_tt, sin_tt, neg_sin_tt, cos_win_tt, sin_win_tt)
        cache[key] = out
        return out

    def _copy_streams_between_submeshes(self, streams, from_submesh_id: int, to_submesh_id: int):
        """Move the decode residual streams between two adjacent submeshes over the
        pre-created socket pair — device-to-device, with no host round-trip.

        Used by the eager :meth:`decode` path: allocate a fresh tensor on the target
        submesh, receive into it, and return it (the loop reassigns ``streams``). The
        traced path instead folds the send/recv directly into each submesh's trace
        (see :meth:`_decode_submesh_static`).
        """
        to_submesh = self.submeshes[to_submesh_id]
        sender_socket, receiver_socket = self.submesh_socket_pairs[(from_submesh_id, to_submesh_id)]
        output_tensor = ttnn.allocate_tensor_on_device(streams.spec, to_submesh)
        ttnn.experimental.send_async(streams, sender_socket)
        ttnn.experimental.recv_async(output_tensor, receiver_socket)
        streams.deallocate(True)
        return output_tensor

    def decode(self, token_id: int, pos: int, rope: dict) -> ttnn.Tensor:
        """Generate one step: feed ``token_id`` at absolute position ``pos`` against
        the running KV cache; returns ``[B, 1, hidden]`` (apply ``lm_head`` for logits).

        ``rope`` is the *full* (max-length) host bundle; the needed rows are sliced
        per layer. The prompt is prefilled by calling this once per prompt token at
        ascending positions, so the cache holds positions ``0 .. pos - 1``."""
        ids = torch.tensor([[token_id]], dtype=torch.long)
        ids_tt = ttnn.from_torch(
            ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.first_device
        )
        inputs_embeds = self.embed_tokens(ids_tt)  # [B, 1, D]
        b, s, d = inputs_embeds.shape
        streams = ttnn.reshape(inputs_embeds, [b, s, 1, d])
        streams = ttnn.repeat(streams, ttnn.Shape([1, 1, self.config.hc_mult, 1]))  # [B, 1, hc_mult, D]

        rope_cache: dict = {}
        last_submesh_id = 0
        for li, layer in enumerate(self.layers):
            if self.use_submeshes:
                current_submesh_id = li // self.layers_per_device
                if current_submesh_id != last_submesh_id:
                    streams = self._copy_streams_between_submeshes(streams, last_submesh_id, current_submesh_id)
                this_device = self.submeshes[current_submesh_id]
            else:
                this_device = self.first_device
            layer_type = self.config.layer_types[li]
            compress_rate = None if layer_type == "sliding_attention" else self.config.compress_rates[layer_type]
            cos_tt, sin_tt, neg_sin_tt, cos_win_tt, sin_win_tt = self._rope_rows_decode(
                rope, pos, layer_type, compress_rate, rope_cache, this_device
            )
            streams = layer.decode(
                streams,
                cos_tt,
                sin_tt,
                neg_sin_tt,
                cos_win_tt,
                sin_win_tt,
                self.kv_caches[li],
                input_ids=ids,
            )
            last_submesh_id = current_submesh_id
            _profile(this_device)
        return self.norm(self.hc_head(streams))

    # ------------------------------------------------------------------ #
    # Traced decode (one reusable trace per submesh / device)
    #
    # The eager :meth:`decode` is host-bound: every step re-dispatches ~43
    # layers' worth of ops, rebuilds the RoPE rows / masks from host, reads the
    # MoE routing weights back to host, and host-copies the residual streams
    # across submeshes. The traced path captures one ``ttnn`` trace per submesh
    # (so each device replays its own slice of the stack) and, between replays,
    # only writes the tiny per-step inputs (token id, RoPE rows, masks, cache
    # positions, hash-router masks) into persistent device tensors; the streams
    # are socket-copied between submeshes from inside the traces themselves (no
    # per-step host op dispatch). All cross-token state lives in fixed-size
    # in-place caches (:class:`_StaticLayerCache`) so a single capture serves
    # every step. See :meth:`prepare_static_decode` / :meth:`decode_traced`.
    # ------------------------------------------------------------------ #
    def _rope_row_host(self, rope: dict, pos: int, rope_type: str):
        """Host ``(cos, sin, neg_sin)`` ``[1,1,1,Rd]`` RoPE rows at ``pos`` for the
        ``"main"`` (sliding) or ``"compress"`` (CSA/HCA) family."""
        cos_h, sin_h = rope["main"] if rope_type == "main" else rope["compress"]
        cos_full, sin_full = make_rope_table(cos_h[pos : pos + 1], sin_h[pos : pos + 1])
        mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return mk(cos_full), mk(sin_full), mk(-sin_full)

    def _decode_mask_host(self, pos: int, layer_type: str, compress_rate: Optional[int], n_win_cap: int) -> ttnn.Tensor:
        """Host additive decode mask ``[1,1,1,W(+n_win_cap)]`` for ``layer_type`` at ``pos``.

        Sliding cols: a ring slot is valid iff its token has been written and is
        within the window (``pos+1 >= W`` -> all valid, else slots ``0..pos``).
        Compressor cols: window ``w`` is valid iff ``w < (pos+1)//compress_rate``
        (the degenerate-indexer causal block bias). Invalid -> ``_MASK_NEG``.
        """
        w = self.sliding_window
        slots = torch.arange(w)
        sld_valid = torch.ones(w, dtype=torch.bool) if pos + 1 >= w else (slots <= pos)
        sld = torch.zeros(w).masked_fill(~sld_valid, _MASK_NEG)
        if layer_type == "sliding_attention":
            row = sld
        else:
            entries = torch.arange(n_win_cap)
            win = torch.zeros(n_win_cap).masked_fill(entries >= ((pos + 1) // compress_rate), _MASK_NEG)
            row = torch.cat([sld, win], dim=0)
        return ttnn.from_torch(row.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def _build_static_layer_cache(self, li: int, device: ttnn.MeshDevice) -> "_StaticLayerCache":
        """Allocate a layer's fixed-size in-place caches *empty* (all-zero).

        There is no prefill: the prompt is fed one token at a time through
        :meth:`decode_traced`, which writes each token's K=V / compressor
        projection into these buffers in place at its absolute position. Unwritten
        ring slots / windows stay zero and are dropped by the per-step decode mask.
        """
        dh = self.config.head_dim
        w = self.sliding_window
        sliding = ttnn.from_torch(
            torch.zeros(1, 1, w, dh),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ckv = cgate = None
        layer_type = self.config.layer_types[li]
        if layer_type != "sliding_attention":
            cr = self.config.compress_rates[layer_type]
            cap = self._cr_caps[cr][0]
            feat = (2 if layer_type == "compressed_sparse_attention" else 1) * dh
            ckv = ttnn.from_torch(
                torch.zeros(1, 1, cap, feat),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            cgate = ttnn.from_torch(
                torch.zeros(1, 1, cap, feat),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return _StaticLayerCache(sliding, ckv, cgate)

    def prepare_static_decode(self, rope: dict, max_seq: int, lm_head=None) -> None:
        """Allocate the traced-decode state (the prompt is prefilled by replaying
        :meth:`decode_traced` once per prompt token into these empty caches).

        Builds, per submesh: the fixed-size in-place caches (empty / all-zero),
        the persistent per-step input tensors (token id / streams,
        RoPE rows, masks, cache positions, hash masks) and the constant window-RoPE
        tables. ``max_seq`` must be a multiple of every compress-rate (the caller
        pads it) so each compressor's fixed capacity tiles cleanly into windows.
        ``lm_head`` (optional) is folded into the last submesh's trace so a step
        returns logits directly.
        """
        if not self.use_submeshes:
            raise NotImplementedError("traced decode requires use_submeshes=True")
        cfg = self.config
        for cr in {cfg.compress_rates[t] for t in cfg.layer_types[: self.num_layers] if t != "sliding_attention"}:
            assert max_seq % cr == 0, f"max_seq ({max_seq}) must be a multiple of compress_rate {cr}"
        self._traced_rope = rope
        self._lm_head_traced = lm_head
        self._cr_caps = {
            cr: (max_seq, max_seq // cr)
            for cr in {cfg.compress_rates[t] for t in cfg.layer_types[: self.num_layers] if t != "sliding_attention"}
        }

        rd = cfg.qk_rope_head_dim
        hc, d, e, w = cfg.hc_mult, cfg.hidden_size, cfg.num_local_experts, self.sliding_window
        num_sm = (self.num_layers + self.layers_per_device - 1) // self.layers_per_device

        def _dev_zeros(shape, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            tt_dtype = {ttnn.bfloat16: torch.float32, ttnn.uint32: torch.int32, ttnn.int32: torch.int32}[dtype]
            return ttnn.from_torch(torch.zeros(shape, dtype=tt_dtype), dtype=dtype, layout=layout, device=device)

        self.submeshes_io = []
        for k in range(num_sm):
            device = self.submeshes[k]
            layers_k = [li for li in range(self.num_layers) if li // self.layers_per_device == k]
            types = {cfg.layer_types[li] for li in layers_k}
            crs = {cfg.compress_rates[t] for t in types if t != "sliding_attention"}
            sm = {
                "device": device,
                "index": k,
                "layers": layers_k,
                "first": k == 0,
                "last": layers_k and layers_k[-1] == self.num_layers - 1,
                "pos_sliding": _dev_zeros([1], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                "pos_compress": _dev_zeros([1], device, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                "rope_in": {},
                "mask_in": {},
                "win_rope": {},
                "hash_masks": {},
                "scaches": {li: self._build_static_layer_cache(li, device) for li in layers_k},
                "tid": None,
                "output": None,
            }
            for rt in ({"main"} if "sliding_attention" in types else set()) | ({"compress"} if crs else set()):
                sm["rope_in"][rt] = tuple(_dev_zeros([1, 1, 1, rd], device) for _ in range(3))
            for lt in types:
                width = w if lt == "sliding_attention" else w + self._cr_caps[cfg.compress_rates[lt]][1]
                sm["mask_in"][lt] = _dev_zeros([1, 1, 1, width], device)
            for cr in crs:
                n_win_cap = self._cr_caps[cr][1]
                cw, sw = make_rope_table(rope["win"][cr][0][:n_win_cap], rope["win"][cr][1][:n_win_cap])
                sm["win_rope"][cr] = (
                    ttnn.from_torch(cw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                    ttnn.from_torch(sw, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                )
            for li in layers_k:
                if self.layers[li].mlp.is_hash:
                    sm["hash_masks"][li] = _dev_zeros([1, 1, 1, e], device)
            if k == 0:
                sm["token_in"] = _dev_zeros([1, 1], device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            else:
                sm["streams_in"] = _dev_zeros([1, 1, hc, d], device)
            self.submeshes_io.append(sm)
        self._traced_captured = False

    def _decode_submesh_static(self, sm: dict) -> ttnn.Tensor:
        """Run one submesh's slice of the decode stack over its persistent inputs /
        in-place caches (shared by the compile run and the trace capture)."""
        cfg = self.config
        k = sm["index"]
        if sm["first"]:
            inputs_embeds = self.embed_tokens(sm["token_in"])  # [1, 1, D]
            b, s, d = inputs_embeds.shape
            streams = ttnn.repeat(ttnn.reshape(inputs_embeds, [b, s, 1, d]), ttnn.Shape([1, 1, cfg.hc_mult, 1]))
        else:
            # Receive the residual streams from the previous submesh directly into
            # the persistent input buffer. Captured inside this submesh's trace so
            # the cross-submesh copy needs no host-side op dispatch at replay time.
            _, receiver_socket = self.submesh_socket_pairs[(k - 1, k)]
            ttnn.experimental.recv_async(sm["streams_in"], receiver_socket)
            streams = sm["streams_in"]
        for li in sm["layers"]:
            layer = self.layers[li]
            lt = cfg.layer_types[li]
            rope_type = "main" if lt == "sliding_attention" else "compress"
            cos, sin, neg_sin = sm["rope_in"][rope_type]
            if lt == "sliding_attention":
                cos_win = sin_win = None
            else:
                cos_win, sin_win = sm["win_rope"][cfg.compress_rates[lt]]
            streams = layer.decode_static(
                streams,
                cos,
                sin,
                neg_sin,
                cos_win,
                sin_win,
                sm["mask_in"][lt],
                sm["scaches"][li],
                sm["pos_sliding"],
                sm["pos_compress"],
                hash_mask=sm["hash_masks"].get(li),
            )
        if sm["last"]:
            streams = self.norm(self.hc_head(streams))
            if self._lm_head_traced is not None:
                streams = self._lm_head_traced(streams)
        else:
            # Send the residual streams to the next submesh over the socket pair.
            # Captured inside this submesh's trace, so the cross-submesh copy is
            # dispatched on device at replay time (no host round-trip).
            sender_socket, _ = self.submesh_socket_pairs[(k, k + 1)]
            ttnn.experimental.send_async(streams, sender_socket)
        return streams

    def _set_step_inputs(self, token_id: int, pos: int) -> None:
        """Write the per-step inputs (token id, RoPE rows, masks, cache positions,
        hash masks) into every submesh's persistent device tensors (allocation-free
        on device, so it is safe to interleave with ``execute_trace``)."""
        cfg = self.config
        w = self.sliding_window
        ps = ttnn.from_torch(torch.tensor([pos % w], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pc = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        rope_host = {rt: self._rope_row_host(self._traced_rope, pos, rt) for rt in ("main", "compress")}
        mask_host: dict = {}
        for lt in {cfg.layer_types[li] for li in range(self.num_layers)}:
            cr = None if lt == "sliding_attention" else cfg.compress_rates[lt]
            n_win_cap = self._cr_caps[cr][1] if cr is not None else 0
            mask_host[lt] = self._decode_mask_host(pos, lt, cr, n_win_cap)

        for sm in self.submeshes_io:
            ttnn.copy_host_to_device_tensor(ps, sm["pos_sliding"])
            ttnn.copy_host_to_device_tensor(pc, sm["pos_compress"])
            for rt, tensors in sm["rope_in"].items():
                for src, dst in zip(rope_host[rt], tensors):
                    ttnn.copy_host_to_device_tensor(src, dst)
            for lt, dst in sm["mask_in"].items():
                ttnn.copy_host_to_device_tensor(mask_host[lt], dst)
            for li, dst in sm["hash_masks"].items():
                mh = ttnn.from_torch(
                    self.layers[li].mlp.gate.selection_mask(token_id), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                )
                ttnn.copy_host_to_device_tensor(mh, dst)
        if self.submeshes_io:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor([[token_id]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                ),
                self.submeshes_io[0]["token_in"],
            )

    def _capture_traces(self) -> None:
        """Capture one trace per submesh: a compile run (to JIT the programs, which
        trace capture itself cannot do), then the recorded capture.

        Each submesh is captured independently — capture only fixes program shapes
        / buffer addresses, so the (stale) compile-run inputs are immaterial: any
        cache rows the compile run writes are at the *same* device-indexed slots a
        later replay overwrites with real values. The real per-step results always
        come from the :meth:`decode_traced` replay loop, never the capture run.

        The compile runs are issued for *all* submeshes before synchronizing,
        because each submesh's slice now contains the cross-submesh socket
        send/recv: a lone ``send_async`` followed by a blocking per-submesh
        ``synchronize_device`` would deadlock (the residual streams exceed the
        socket's L1 buffer, so the send cannot drain until the next submesh posts
        its matching ``recv_async``). Issuing every submesh first lets the sends
        and receives pair up across devices, after which a single sync drains
        them. Trace capture only records ops (it does not execute them), so the
        capture loop is free of this hazard.
        """
        compile_outs = []
        for sm in self.submeshes_io:
            logger.info(f"[traced-decode] compiling submesh {sm['index']} ({len(sm['layers'])} layers)")
            compile_outs.append(self._decode_submesh_static(sm))  # compile run (JITs the programs)
        for out in compile_outs:
            out.deallocate(True)
        for sm in self.submeshes_io:
            device = sm["device"]
            logger.info(f"[traced-decode] capturing submesh {sm['index']} ({len(sm['layers'])} layers)")
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            with _trace_capture_guard():
                out = self._decode_submesh_static(sm)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            sm["tid"] = tid
            sm["output"] = out  # persistent; overwritten in place by every execute_trace
        self._traced_captured = True

    def decode_traced(self, token_id: int, pos: int) -> ttnn.Tensor:
        """One traced decode step: feed ``token_id`` at absolute position ``pos``.

        Requires a prior :meth:`prepare_static_decode`. Captures
        the per-submesh traces lazily on the first call, then (every call) refreshes
        the per-step inputs and replays each submesh's trace in order. The residual
        streams are socket-copied between submeshes from *inside* each trace
        (device-to-device, no host hop and no per-step host op dispatch).
        Returns the last submesh's persistent output tensor — logits ``[1,1,vocab]``
        if an ``lm_head`` was passed to :meth:`prepare_static_decode`, else the
        pre-head hidden ``[1,1,hidden]``.

        The returned tensor is overwritten by the next call, so consume it (e.g.
        ``ttnn.to_torch``) before decoding the following token.
        """
        self._set_step_inputs(token_id, pos)
        if not self._traced_captured:
            self._capture_traces()
        for sm in self.submeshes_io:
            ttnn.execute_trace(sm["device"], sm["tid"], cq_id=0, blocking=False)
        return self.submeshes_io[-1]["output"]
