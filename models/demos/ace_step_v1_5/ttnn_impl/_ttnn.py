from __future__ import annotations


def get_ttnn():
    try:
        import ttnn  # type: ignore

        return ttnn
    except Exception:
        return None
