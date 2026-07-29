"""Promote / learning loop — should_promote, prompt, provisional-lever write + graduate (no hw)."""

from pathlib import Path

from agent import states
from agent.promote import (
    graduate_lever,
    maybe_graduate,
    write_provisional_lever,
)


class _Ctx:
    def __init__(
        self, lever, decision, *, top_ops=None, bucket="datamove", last_diff="@@ -1 +1 @@\n-x\n+y", root="nemotron"
    ):
        self.state = {
            "selected_lever": lever,
            "last_decision": decision,
            "current_bucket": bucket,
            "top_ops": top_ops or [{"op_code": "Tilize", "shape": "1024x6144"}],
            "last_diff": last_diff,
        }
        self.deps = {}
        self._root = root

    def model_root(self):
        return Path(self._root)

    def current_profile(self):
        return {"buckets": [{"id": "datamove", "tags": {"op_class": "datamove"}}]}


_WIN = {"result": "keep", "before": 90.7, "after": 88.0, "pcc": 0.999}


def test_write_provisional_lever_is_router_indexable(tmp_path):
    section = "## Learned: datamove coherence {#learned-datamove-coherence-x}\n<!-- route\nop_class: datamove\nlever_type: structural\n-->\n\n**Fires when:** redundant tilize.\nEmit the producer's tile layout."
    path = write_provisional_lever(section, "datamove-coherence-x", tmp_path, "nemotron")
    assert path.exists() and path.name == "LEARNED_datamove-coherence-x.md"
    text = path.read_text()
    assert "provisional: true" in text and "learned_on: nemotron" in text and "{#learned-datamove-coherence-x}" in text
    # the router's build_index globs *.md -> the learned lever is picked up
    from agent.router import build_index

    idx = build_index(tmp_path)
    assert any(e["id"] == "learned-datamove-coherence-x" for e in idx)


def test_graduate_flips_provisional_and_renames(tmp_path):
    p = write_provisional_lever(
        "## x {#x}\n<!-- route\nop_class: datamove\nlever_type: structural\n-->\nbody", "x", tmp_path, "nemotron"
    )
    assert p.name == "LEARNED_x.md"
    np = graduate_lever(p, "seamless")
    # renamed LEARNED_ -> GRADUATED_ so it leaves the gitignored provisional set
    assert np.name == "GRADUATED_x.md" and np.exists() and not p.exists()
    text = np.read_text()
    assert "provisional: false" in text and "graduated_on: seamless" in text


def test_maybe_graduate_only_cross_model(tmp_path):
    write_provisional_lever(
        "## y {#y}\n<!-- route\nop_class: datamove\nlever_type: structural\n-->\nbody", "y", tmp_path, "nemotron"
    )
    # same model that learned it -> NOT graduated (no cross-model evidence)
    assert maybe_graduate(_Ctx(states.FROM_PRINCIPLES, _WIN, root="nemotron"), "y", guidelines_dir=tmp_path) is None
    assert (tmp_path / "LEARNED_y.md").exists()
    # a DIFFERENT model re-using and keeping it -> graduates (rename + trusted)
    gp = maybe_graduate(_Ctx(states.FROM_PRINCIPLES, _WIN, root="seamless"), "y", guidelines_dir=tmp_path)
    assert gp and gp.name == "GRADUATED_y.md" and not (tmp_path / "LEARNED_y.md").exists()
    # unrelated lever id -> nothing graduates
    assert (
        maybe_graduate(_Ctx(states.FROM_PRINCIPLES, _WIN, root="qwen"), "nonexistent", guidelines_dir=tmp_path) is None
    )
