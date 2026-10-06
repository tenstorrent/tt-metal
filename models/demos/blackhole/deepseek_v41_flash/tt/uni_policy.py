"""Policy for the unified (deepseek_prefill) routed-expert prefill MoE reading the decode ring weights in place (ONE weight copy).

Default: ON (``DSV41_PREFILL_MOE`` unset / ``auto`` / ``unified``); ``DSV41_PREFILL_MOE=off`` (or 0 / none / baseline / moe_compute) forces the old
``moe_compute`` prefill path. ``DSV41_UNI_RING`` (default 1) = read the decode weights in place; ``DSV41_UNI_RING=0`` + ``DSV41_UNI_NODECODE=1`` is the old
prefill-only measurement mode with its own weight copy. With the default ON the model falls back to the old path (one log line saying why) when
  * the loaded ttnn library has no RING_WEIGHTS mode (`ring_supported`),
  * the chip does not have 8 live DRAM banks (the ring size of the decode weights; the op needs one N column per ring core),
  * U = 1 user per row and ``DSV41_UNI_U1`` is not ``1`` (see `U1_DEFAULT_ON`; the unified path's compile pass was seen to hang there),
  * (per chunk, in `colsplit_active`) the chunk shape does not allow the column split: the layers then use their moe_compute path.
The decode weights / decode path are never touched."""
import ctypes
import os

OFF_WORDS = ("off", "0", "none", "false", "baseline", "moe_compute", "no")
# U = 1 (B = 4): set from the hang evidence, see PREFILL_UNIFIED_MOE_NOTES.md
U1_DEFAULT_ON = True

_state = {"in_use": False}


def requested():
    """True unless the user forced the old path."""
    return os.environ.get("DSV41_PREFILL_MOE", "").strip().lower() not in OFF_WORDS


def ring_requested():
    return os.environ.get("DSV41_UNI_RING", "1") == "1" and os.environ.get("DSV41_UNI_NODECODE") != "1"


def in_use():
    """Is the unified MoE what the prefill layers use right now (set by the model builder / the per-scenario toggle of the demo)?"""
    return _state["in_use"]


def set_in_use(v):
    _state["in_use"] = bool(v)


_RING_MARK = b"ring weights mode needs 8 live DRAM banks"
_ring_ok = {}


def ring_supported():
    """Does the _ttnncpp library MAPPED in this process contain the RING_WEIGHTS validation string? (scans the mapped read-only segments, so a library
    that was replaced on disk after this process started is judged by what is actually loaded)"""
    if "v" in _ring_ok:
        return _ring_ok["v"]
    ok = False
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "_ttnncpp.so" in line and line.split()[1].startswith("r"):
                    a, b = (int(x, 16) for x in line.split()[0].split("-"))
                    if _RING_MARK in ctypes.string_at(a, b - a):
                        ok = True
                        break
    except Exception:
        ok = False
    _ring_ok["v"] = ok
    return ok


def banks_ok(md):
    try:
        from ttnn.experimental.moe_compute_utils import effective_matmul_ring_size

        return int(effective_matmul_ring_size(md)) == 8
    except Exception:
        return False


_decided = {}


def decide(md, U, log=print):
    """-> True if the unified MoE (+ ring weights when requested) should be built for this model; logs one line when it falls back (memoised per U)."""
    if U in _decided:
        set_in_use(_decided[U])
        return _decided[U]
    r = _decide(md, U, log)
    _decided[U] = r
    return r


def _decide(md, U, log):
    if not requested():
        set_in_use(False)
        return False
    why = None
    if ring_requested() and not ring_supported():
        why = "the loaded ttnn library has no ring-weights mode for unified_routed_expert_moe (rebuild: PREFILL_UNIFIED_MOE_NOTES.md)"
    elif ring_requested() and not banks_ok(md):
        why = "the chip does not have 8 live DRAM banks (ring size != 8)"
    elif U == 1 and not U1_DEFAULT_ON and os.environ.get("DSV41_UNI_U1") != "1":
        why = "1 user per mesh row (U=1): the unified path is off by default there (DSV41_UNI_U1=1 enables it)"
    if why:
        log(f"prefill MoE: falling back to the moe_compute prefill path: {why}")
        set_in_use(False)
        return False
    set_in_use(True)
    return True
