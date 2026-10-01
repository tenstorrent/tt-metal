import os
from typing import Optional

import ttnn
import torch


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
        """The plain string ``value`` with the cache's ``require_cache`` flag attached to the
        instance (a ``str`` is immutable, so the flag rides on the subclass)."""
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
    error instead of a fallback HF-checkpoint read, so a populated cache is
    actually consumed (see :class:`DeepSeekV4Model`). It rides
    along on the :class:`_CachePath` returned by :meth:`file` and is propagated to
    every child via :meth:`sub`.
    """

    __slots__ = ("path", "prefix", "require_cache")

    def __init__(self, path: Optional[str] = None, prefix: str = "", require_cache: bool = False):
        """``path`` is the cache directory (``None`` disables caching), ``prefix`` this cache's
        dotted checkpoint path, ``require_cache`` turns a cache miss into a hard error."""
        self.path = path
        self.prefix = prefix
        self.require_cache = require_cache

    def sub(self, name: str) -> "WeightCache":
        """A child cache for the sub-module ``name``: this one's path and ``require_cache``,
        with ``prefix`` extended to ``prefix.name``."""
        prefix = f"{self.prefix}.{name}" if self.prefix else name
        return WeightCache(self.path, prefix, self.require_cache)

    def require(self, flag: bool = True) -> "WeightCache":
        """A sibling cache (same path / prefix) with ``require_cache`` set."""
        return WeightCache(self.path, self.prefix, flag)

    def _name(self, name: str) -> str:
        """This cache's dotted name for ``name``: ``prefix.name``, or ``name`` at the root."""
        return f"{self.prefix}.{name}" if self.prefix else name

    def file(self, name: str) -> Optional[str]:
        """The ``cache_file_name`` to hand :func:`ttnn.as_tensor` for the weight ``name``, or
        ``None`` when caching is off (which makes ``as_tensor`` convert every time).

        ``/`` in the dotted name becomes ``_`` so one flat directory holds the whole model's
        converted tensors, and the returned :class:`_CachePath` carries ``require_cache``. The
        key is the name plus the dtype/layout suffix ``as_tensor`` appends -- *not* the weight's
        shape -- so a caller whose weight shape varies with a parameter puts that parameter in
        the name (e.g. ``gate_proj.tp4.decode``).
        """
        if not self.path:
            return None
        return _CachePath(os.path.join(self.path, self._name(name).replace("/", "_")), self.require_cache)

    def hit(self, name: str, dtype: ttnn.DataType, layout: ttnn.Layout = ttnn.TILE_LAYOUT) -> bool:
        """Whether a cached file already exists for ``name`` (matching the exact
        ``ttnn.as_tensor`` suffix), so callers can skip producing the torch
        tensor entirely on a cache hit. See :meth:`file` for what the key does not include."""
        base = self.file(name)
        if not base:
            return False
        return os.path.isfile(f"{base}_dtype_{dtype.name}_layout_{layout.name}.tensorbin")


def _as_cache(cache: Optional[WeightCache]) -> WeightCache:
    """Normalise ``None`` to a disabled cache (``path=None``, so :meth:`WeightCache.file`
    returns ``None``) so call sites stay branch-free."""
    return cache if cache is not None else WeightCache()


def _load_weight(
    tensor: Optional[torch.Tensor],
    device: ttnn.MeshDevice,
    *,
    cache_file_name: Optional[str] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=None,
) -> ttnn.Tensor:
    """``ttnn.as_tensor`` for a (static) weight, with optional disk caching.

    ``tensor`` is the host weight (``[K, N]`` for a projection, whatever the layer's layout
    wants) and the result is that weight on ``device`` as a ``dtype``/``layout`` tensor in
    ``memory_config``. Equivalent to ``ttnn.from_torch(...)`` when ``cache_file_name`` is
    ``None``; otherwise the tilized tensor is dumped on first use and loaded back on later
    runs. ``tensor`` may be ``None`` only on a verified cache hit (see
    :meth:`WeightCache.hit`). ``mesh_mapper`` shards or replicates across a mesh.
    """
    return ttnn.as_tensor(
        tensor,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
        cache_file_name=cache_file_name,
        mesh_mapper=mesh_mapper,
    )


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

    ``weight`` is either a ``torch.Tensor`` or a zero-arg callable returning one (any shape the
    layer wants, e.g. ``[K, N]``). On a cache hit the source is never touched (so a populated
    cache reads nothing from the checkpoint); otherwise the thunk is called -- or the tensor
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

    Returns a callable: a tensor argument is boxed as-is, a callable one is evaluated on the
    first call and its result kept. Used where one checkpoint tensor is sliced into several
    device weights (e.g. the hyper-connections' fused ``fn`` weight ``[(2+hc)*hc, hc*D]``,
    whose pre/post/comb parts are contiguous slices of it) so the underlying source is read a
    single time even across multiple cache-miss slices.
    """
    if not callable(weight):
        return lambda: weight
    box: dict = {}

    def get():
        """The memoized source weight (every shape the caller slices out of it)."""
        if "v" not in box:
            box["v"] = weight()
        return box["v"]

    return get
