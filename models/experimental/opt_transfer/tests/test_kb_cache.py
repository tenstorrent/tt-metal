from models.experimental.opt_transfer.kb.cache import ContentCache


def test_cache_hit_avoids_recompute(tmp_path):
    cache = ContentCache(tmp_path)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"entries": [1, 2, 3]}

    a = cache.get_or_compute(key="f.py", content="abc", compute=compute)
    b = cache.get_or_compute(key="f.py", content="abc", compute=compute)
    assert a == b == {"entries": [1, 2, 3]}
    assert calls["n"] == 1


def test_cache_invalidates_on_content_change(tmp_path):
    cache = ContentCache(tmp_path)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    cache.get_or_compute(key="f.py", content="v1", compute=compute)
    cache.get_or_compute(key="f.py", content="v2", compute=compute)
    assert calls["n"] == 2
