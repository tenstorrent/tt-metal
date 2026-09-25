# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F3: reference interface, HF parity, chunked + graph-replay check, golden generator and reader. CPU only.

Runs the generic scripts on the synthetic fixture (selftest/fixture_model.py), and reads ERNIE's existing 2k
golden (read-only) to check the reader against a real golden when it is present on this machine.
"""


import pytest

from models.demos.common.bringup.core.spec import CODE_ROOT, Spec
from models.demos.common.bringup.reference import check_hf, check_reference, generate_golden
from models.demos.common.bringup.reference.golden import Golden, content_hash
from models.demos.common.bringup.reference.interface import Step, validate_graph
from models.demos.common.bringup.selftest.conftest import got

ERNIE_2K = CODE_ROOT / "generated/ernie45_d_p/golden/s4096_c2048"


def test_fixture_spec_is_valid(fx):
    assert Spec.load(fx()).validate() == []


def test_hf_parity(fx):
    check_hf.main(["--spec", fx(), "--seq", "128"])
    m = got()
    assert [k for k in m if k.startswith("pcc_hidden_L")] == ["pcc_hidden_L00", "pcc_hidden_L01", "pcc_hidden_L02"]
    assert min(v for k, v in m.items() if k.startswith("pcc_")) > 0.99999
    assert m["top1_match_frac"] == 1.0


def test_hf_parity_on_truncated_layers(fx):
    check_hf.main(["--spec", fx(), "--seq", "64", "--num-layers", "2"])
    assert got()["parity_layers"] == 2 and "pcc_hidden_L02" not in got()


def test_chunked_and_graph_replay(fx):
    check_reference.main(["--spec", fx(), "--seq", "256", "--chunk", "64"])
    m = got()
    assert m["pcc_hidden"] > 0.999999 and m["pcc_state_min"] > 0.999999
    assert m["graph_errors"] == 0 and m["boundaries_missing"] == 0 and m["graph_replay_maxabs"] == 0.0


def test_graph_replay_catches_a_graph_that_is_not_the_code(fx):
    check_reference.main(["--spec", fx(fixture={"broken_graph": True}), "--seq", "256", "--chunk", "64"])
    assert got()["graph_replay_maxabs"] > 0


def test_validate_graph():
    ok = [Step("a", ("in",), "x"), Step("b", ("in", "x"), "out")]
    assert validate_graph(ok) == []
    errs = validate_graph([Step("a", ("y",), "x"), Step("a", ("x",), "x")])
    assert any("before any step" in e for e in errs) and any("duplicate" in e for e in errs)
    assert any("overwrites" in e for e in errs) and any("'out'" in e for e in errs)


def test_golden_full_layers(fx):
    spec_path = fx()
    generate_golden.main(["--spec", spec_path, "--rung", "s256"])
    assert got()["golden_hash_ok"] == 1 and got()["golden_chunks"] == 4
    g = Golden.for_rung(Spec.load(spec_path), "s256")
    assert g.verify() and g.dumped_chunks == [0, 1, 2, 3] and g.layers == [0, 1, 2]
    assert set(g.layer(2, 1)) == {"in", "attn_norm", "attn_out", "h_mid", "ffn_norm", "mlp_out", "out"}
    assert g.state(0)["key"].shape == (256, 64) and g.tokens().shape == (256,)
    assert set(g.model(3)) >= {"embed", "final_norm", "top32_ids", "top32_values", "logits_tail", "tokens"}
    with pytest.raises(SystemExit, match="generated once"):
        generate_golden.main(["--spec", spec_path, "--rung", "s256"])
    (g.dir / "chunk_00/model.safetensors").write_bytes(b"tampered")
    assert not Golden(g.dir).verify()


def test_golden_layer_subset_stores_run_starts_on_every_chunk(fx):
    spec_path = fx(layers=[0, 2])
    generate_golden.main(["--spec", spec_path, "--rung", "s512"])
    g = Golden.for_rung(Spec.load(spec_path), "s512")
    assert g.manifest["subset"] and g.manifest["run_starts"] == [0, 2] and g.dumped_chunks == [3]
    assert set(g.layer(0, 0)) == {"in"} and set(g.layer(0, 2)) == {"in"} and not g.has_layer(0, 1)
    assert "out" in g.layer(3, 2) and not g.has_layer(3, 1)
    assert sorted(p.name for p in (g.dir / "kv_cache").iterdir()) == ["layer_0.safetensors", "layer_2.safetensors"]
    # the "last" rung reads the s512 golden
    assert Golden.for_rung(Spec.load(spec_path), "last").dir == g.dir
    generate_golden.main(["--spec", spec_path, "--rung", "last"])  # no-op


def test_content_hash_ignores_the_manifest(tmp_path):
    (tmp_path / "a").write_text("1")
    h = content_hash(tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    assert content_hash(tmp_path) == h


@pytest.mark.skipif(not ERNIE_2K.exists(), reason="ERNIE 2k golden not on this machine")
def test_reader_reads_the_existing_ernie_golden():
    g = Golden(ERNIE_2K)
    assert g.seq == 4096 and g.n_chunks == 2 and len(g.layers) == 28
    assert {"in", "out", "attn_out", "mlp_out"} <= set(g.layer(1, 1))
    st = g.state(0)
    assert set(st) == {"key", "value"} and st["key"].shape[-2:] == (4096, 128)
    assert len(g.pinned_hash()) == 64 and not g.verify()  # made before content hashes existed
