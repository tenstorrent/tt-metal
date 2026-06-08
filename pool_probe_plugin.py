import ttnn

_orig = ttnn.max_pool2d


def _wrap(*args, **kw):
    t = kw.get("input_tensor", args[0] if args else None)
    try:
        mc = t.memory_config()
        print(
            f"POOLPROBE: max_pool2d input buffer_type={mc.buffer_type} layout={mc.memory_layout} shape={list(t.shape)}",
            flush=True,
        )
    except Exception as e:
        print(f"POOLPROBE: err {e!r}", flush=True)
    return _orig(*args, **kw)


ttnn.max_pool2d = _wrap
