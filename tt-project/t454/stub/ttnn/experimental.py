def __getattr__(n):
    from ttnn import _Any

    return _Any()
