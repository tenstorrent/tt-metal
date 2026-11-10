"""Per-layer routed-expert weight format (default off): mixed precision by depth.

  DSV41_BFP4_LAYERS=<list / ranges, e.g. 0-9,12>   MoE layers whose routed experts are bfp4
  DSV41_BFP4_MAX_LAYER=N                           bfp4 for MoE layers < N (union with the list)
  DSV41_BFP4_RING=<8 | 24 | 40>                    decode matmul ring of the bfp4 layers (default 8); the other layers stay bfp8 on the ring of 8
Nothing set = the global switches (MOE_COMPUTE_BFP8_WEIGHTS / MOE_COMPUTE_RING) decide as before. With a selection set the global switches stay at the production values (bfp8, ring 8);
the MTP / drafter experts keep them (they do not go through ``load_moe_layer``).

How a layer picks its format: ``load_moe_layer`` and ``DSV41MoEBlock.__init__`` run under ``env_ctx(fmt)``, which sets MOE_COMPUTE_BFP8_WEIGHTS / MOE_COMPUTE_RING for the duration of the
weight packing, the cache directory / file names (layer_<L>_r<N>, moe_*_bfp4|bfp8) and the device upload. The ops derive the dtype and the ring from the weight tensors themselves
(program factory: w0_w1 dtype, w2 shape[0]), so no env is read at call time; the unified in-place prefill op reads each layer's tensors as they are.
"""
import contextlib
import functools
import os

_KEYS = ("MOE_COMPUTE_BFP8_WEIGHTS", "MOE_COMPUTE_RING")


def _parse(spec):
    out = set()
    for part in (spec or "").replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def mixed_active():
    return bool(os.environ.get("DSV41_BFP4_LAYERS") or os.environ.get("DSV41_BFP4_MAX_LAYER"))


def bfp4_layer(layer_id):
    mx = int(os.environ.get("DSV41_BFP4_MAX_LAYER") or 0)
    return layer_id < mx or layer_id in _parse(os.environ.get("DSV41_BFP4_LAYERS"))


def bfp4_ring():
    return int(os.environ.get("DSV41_BFP4_RING") or 8)


def any_bfp4():
    return mixed_active() and (
        int(os.environ.get("DSV41_BFP4_MAX_LAYER") or 0) > 0 or bool(_parse(os.environ.get("DSV41_BFP4_LAYERS")))
    )


def fmt(layer_id):
    """None when no selection is set, else {"bfp4": bool, "ring": int} of this MoE layer."""
    if not mixed_active():
        return None
    if bfp4_layer(layer_id):
        return {"bfp4": True, "ring": bfp4_ring()}
    return {"bfp4": False, "ring": 8}


def describe():
    if not mixed_active():
        return "global"
    return f"bfp4 layers < {os.environ.get('DSV41_BFP4_MAX_LAYER') or 0} + [{os.environ.get('DSV41_BFP4_LAYERS') or ''}] on ring {bfp4_ring()}, the other layers bfp8 ring 8"


@contextlib.contextmanager
def env_ctx(f):
    if not f:
        yield
        return
    old = {k: os.environ.get(k) for k in _KEYS}
    os.environ["MOE_COMPUTE_BFP8_WEIGHTS"] = "0" if f["bfp4"] else "1"
    if f["ring"] != 8:
        os.environ["MOE_COMPUTE_RING"] = str(f["ring"])
    else:
        os.environ.pop("MOE_COMPUTE_RING", None)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def layer_ctx(fn):
    """decorator for load_moe_layer(layer_id, ...): the whole call runs under the layer's format"""

    @functools.wraps(fn)
    def w(layer_id, *a, **k):
        with env_ctx(fmt(layer_id)):
            return fn(layer_id, *a, **k)

    return w


def ctor_ctx(fn):
    """decorator for DSV41MoEBlock.__init__(self, mesh_device, weights, ...): runs under weights["fmt"] (set by load_moe_layer)"""

    @functools.wraps(fn)
    def w(self, mesh_device, weights, *a, **k):
        with env_ctx(weights.get("fmt")):
            return fn(self, mesh_device, weights, *a, **k)

    return w
