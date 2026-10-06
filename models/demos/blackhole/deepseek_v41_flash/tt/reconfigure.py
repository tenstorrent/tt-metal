# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Helpers of ``Model.reconfigure`` (tt/dsv41_model.py): reset the module level caches of device constants that depend on the batch / chunk
configuration, and free nested tensor containers. See RECONFIGURE_NOTES.md."""

import ttnn


def free_tensors(obj):
    """ttnn.deallocate every ttnn tensor in a (nested) tuple / list / dict container."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            free_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            free_tensors(v)
    elif isinstance(obj, ttnn.Tensor):
        try:
            if obj.is_allocated():
                ttnn.deallocate(obj)
        except Exception:  # noqa: BLE001  (already freed / not a device tensor)
            pass


def reset_module_caches():
    """Free and clear every module level cache that holds device tensors or batch specific state. Must run after all traces were released (a trace
    may hold constants of these caches) and before the model is rebuilt."""
    from models.demos.blackhole.deepseek_v41_flash.tt import (
        pf_tune,
        prefill_attention,
        prefill_layer,
        prefill_sparse,
        prefill_unified_moe,
    )

    for d in (
        prefill_attention._MASKS,
        prefill_attention._TABS,
        prefill_attention._ZEROS,
        prefill_sparse._CONST,
        prefill_sparse._PERSIST,
        pf_tune._P64,
    ):
        free_tensors(list(d.values()))
        d.clear()
    for sh in prefill_unified_moe._SHARED.values():  # per-(mesh, tokens per row) dispatch tables
        free_tensors([getattr(sh, "dispatch_table", None), getattr(sh, "gidx", None)])
    prefill_unified_moe._SHARED.clear()
    prefill_layer._G1_BUFFERS.clear()  # shared T=32 moe_compute scratch buffers (freed with their last reference)
    pf_tune._CACHE.clear()
    try:
        from models.demos.blackhole.deepseek_v41_flash.tt import mhc_mixes2

        mhc_mixes2._PLANS.clear()
    except Exception:  # noqa: BLE001
        pass


def debug_l1_referrers(log, md, depth=6):
    """DSV41_RECONFIG_DEBUG=1: after the release, list every object of the old model that is still alive and the chain of python objects that refer to
    it, to find what keeps device state alive."""
    import gc
    import types

    objs = gc.get_objects()
    skip = {id(objs)}

    def label(r, o):
        if isinstance(r, dict):
            return f"dict{[k for k, v in r.items() if v is o][:2]}"
        if isinstance(r, types.FunctionType):
            return f"function {r.__module__}.{r.__qualname__}"
        return type(r).__name__ + (f"[{len(r)}]" if isinstance(r, (list, tuple)) else "")

    def owners(o, d):
        out = []
        for r in gc.get_referrers(o):
            if id(r) in skip or isinstance(r, types.FrameType) or len(out) >= 4:
                continue
            skip.add(id(r))
            if isinstance(r, dict):
                own = [x for x in gc.get_referrers(r) if getattr(x, "__dict__", None) is r and id(x) not in skip]
                out.append(
                    (
                        label(r, o) + (f" of {type(own[0]).__name__}" if own else ""),
                        owners(own[0], d - 1) if own and d > 0 else [],
                    )
                )
            else:
                out.append((label(r, o), owners(r, d - 1) if d > 0 else []))
        return out

    def fmt(ch, ind=2):
        return "".join(f"\n{' ' * ind}{lab}{fmt(sub, ind + 2)}" for lab, sub in ch)

    names = (
        "_TTMoEDecodeBuffers",
        "DSV41PrefillMoE",
        "DSV41Layer",
        "PagedKVPool",
        "DSV41PrefillLayer",
        "DSV41MoEBlock",
        "CCLManager",
    )
    for o in objs:
        if type(o).__name__ in names:
            log(f"RC_DEBUG surviving {type(o).__name__} referrers:{fmt(owners(o, depth))}")
    log("RC_DEBUG done")
