import json
from models.experimental.opt_transfer.handoff import dump_bundle


def test_dump_bundle_writes_readable_artifacts(tmp_path):
    state = {
        "model": "seamless_m4t_v2",
        "iteration": 3,
        "diagnosis": {"node": "ffn_fuse", "axis": "per_block_pcc", "measured": 0.8},
        "proposals": [{"entry_id": "qkv_merge"}],
    }
    path = dump_bundle(state, run_dir=tmp_path)
    bundle = json.loads((path / "diagnosis_bundle.json").read_text())
    assert bundle["model"] == "seamless_m4t_v2"
    assert (path / "README_FOR_CLAUDE.md").exists()
