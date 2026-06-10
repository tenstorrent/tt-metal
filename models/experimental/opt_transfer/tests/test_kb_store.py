from models.experimental.opt_transfer.kb.store import KBStore
from models.experimental.opt_transfer.schema import KBEntry, PatternKind


def _entry(id, category="uncategorized", torch_pattern=None):
    return KBEntry(
        id=id,
        fused_op="op",
        category=category,
        pattern_kind=PatternKind.CHAIN,
        torch_pattern=torch_pattern or ["linear"],
        signature={},
        config_template={},
        weight_transform=None,
        source="x",
    )


def test_save_then_load(tmp_path):
    store = KBStore(tmp_path)
    store.save([_entry("a", "attention.qkv"), _entry("b", "norm")])
    loaded = KBStore(tmp_path).load()
    assert {e.id for e in loaded} == {"a", "b"}


def test_retrieve_by_category(tmp_path):
    store = KBStore(tmp_path)
    store.save([_entry("a", "attention.qkv"), _entry("b", "norm")])
    hits = store.retrieve(categories=["attention.qkv"])
    assert [e.id for e in hits] == ["a"]


def test_load_excludes_index_file(tmp_path):
    store = KBStore(tmp_path)
    store.save([_entry("a_fuse")])
    assert (tmp_path / "_index.json").exists()
    loaded = store.load()
    assert [e.id for e in loaded] == ["a_fuse"]


def test_lookup_by_op_overlap(tmp_path):
    store = KBStore(tmp_path)
    store.save(
        [
            _entry("qkv_merge", torch_pattern=["linear", "linear", "linear"]),
            _entry("softmax_fuse", torch_pattern=["matmul", "softmax"]),
        ]
    )
    out = store.lookup(["Linear", "layer_norm"])
    assert [e.id for e in out] == ["qkv_merge"]


def test_lookup_top_k_and_empty_query(tmp_path):
    store = KBStore(tmp_path)
    store.save([_entry(f"lin_{i}", torch_pattern=["linear"]) for i in range(5)])
    assert len(store.lookup(["linear"], top_k=2)) == 2
    assert store.lookup([]) == []
