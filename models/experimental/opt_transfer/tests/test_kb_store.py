from models.experimental.opt_transfer.kb.store import KBStore
from models.experimental.opt_transfer.schema import KBEntry, PatternKind


def _entry(id, category):
    return KBEntry(
        id=id,
        fused_op="op",
        category=category,
        pattern_kind=PatternKind.CHAIN,
        torch_pattern=["linear"],
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
