# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-list -> rank binding derivation in run_pipeline_prefill.sh.

Driven through PP_DRY_RUN=1, so these need bash and nothing else -- no MPI, no devices.
"""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

RUNNERS = Path(__file__).resolve().parents[1] / "runners"
LAUNCHER = RUNNERS / "run_pipeline_prefill.sh"
TOPO = RUNNERS / "topology_configuration"
COMBINED = TOPO / "pipeline_prefill_request.yaml"
INTRAGALAXY = TOPO / "pipeline_prefill_request_intragalaxy_2rank.yaml"


def dry_run(binding, hosts, tmp_path, **env):
    """Run the launcher in dry-run mode; return (exit code, stdout, stderr)."""
    proc = subprocess.run(
        [str(LAUNCHER), str(binding), hosts],
        capture_output=True,
        text=True,
        env={**os.environ, "PP_DRY_RUN": "1", "PP_BINDING_OUT_DIR": str(tmp_path), **env},
    )
    return proc.returncode, proc.stdout, proc.stderr


def derived(binding, hosts, tmp_path, **env):
    code, out, err = dry_run(binding, hosts, tmp_path, **env)
    assert code == 0, f"launcher failed: {err}"
    _, _, body = out.partition("\n")
    return yaml.safe_load(body)


@pytest.mark.parametrize(
    "hosts, num_ranks, descriptor",
    [
        ("h0", 1, "single_bh_galaxy_torus_xy_graph_descriptor.textproto"),
        ("h0,h1", 2, "pipeline_prefill_2galaxy_connected_mesh_graph_descriptor.textproto"),
        ("h0,h1,h2,h3", 4, "pipeline_prefill_4galaxy_connected_mesh_graph_descriptor.textproto"),
        (
            "h0:1,h1:1,h2:1,h3:1,h4:1,h5:1,h6:1,h7:1",
            8,
            "pipeline_prefill_8galaxy_connected_mesh_graph_descriptor.textproto",
        ),
    ],
    ids=["1rank", "2rank", "4rank", "8rank_explicit_slots"],
)
def test_rank_count_comes_from_host_list(hosts, num_ranks, descriptor, tmp_path):
    config = derived(COMBINED, hosts, tmp_path)

    assert config["rank_bindings"] == [
        {"rank": r, "mesh_id": r, "mesh_host_rank": 0, "env_overrides": {}} for r in range(num_ranks)
    ]
    assert config["mesh_graph_desc_path"].endswith(descriptor)


def test_derived_binding_keeps_the_source_env(tmp_path):
    config = derived(COMBINED, "h0,h1", tmp_path)
    expected = yaml.safe_load(COMBINED.read_text())["global_env"]

    assert config["global_env"] == expected


@pytest.mark.parametrize("num_ranks", [1, 2, 4, 8], ids=lambda n: f"{n}rank")
def test_derived_matches_the_pinned_per_rank_binding(num_ranks, tmp_path):
    """The per-rank yamls stay supported; derivation must agree with them."""
    pinned = yaml.safe_load((TOPO / f"pipeline_prefill_request_{num_ranks}rank.yaml").read_text())
    config = derived(COMBINED, ",".join(f"h{r}" for r in range(num_ranks)), tmp_path)

    assert config["rank_bindings"] == pinned["rank_bindings"]
    assert config["mesh_graph_desc_path"] == pinned["mesh_graph_desc_path"]


def test_pinned_binding_is_passed_through_untouched(tmp_path):
    code, out, err = dry_run(INTRAGALAXY, "h0,h1", tmp_path)

    assert code == 0, err
    assert out.splitlines()[0] == f"rank-binding: {INTRAGALAXY}"
    assert list(tmp_path.iterdir()) == [], "a pinned binding must not generate a file"
    # the per-rank TT_VISIBLE_DEVICES masks survive
    config = yaml.safe_load(out.partition("\n")[2])
    assert config["rank_bindings"][1]["env_overrides"]["TT_VISIBLE_DEVICES"]


def test_a_pinned_binding_may_still_put_several_ranks_on_one_host(tmp_path):
    """Derivation refuses hostA:2, but pinning is exactly how intragalaxy splits one galaxy."""
    code, out, err = dry_run(INTRAGALAXY, "h0:2", tmp_path)

    assert code == 0, err
    assert out.splitlines()[0] == f"rank-binding: {INTRAGALAXY}"


def test_the_two_keys_are_derived_independently(tmp_path):
    """A binding may pin one key and let the other be derived (e.g. an odd galaxy count)."""
    own_mgd = TOPO / "pipeline_prefill_4galaxy_connected_fabric2d_mesh_graph_descriptor.textproto"
    binding = tmp_path / "mgd_pinned.yaml"
    binding.write_text(f'mesh_graph_desc_path: {own_mgd}\nglobal_env:\n  LOGURU_LEVEL: "INFO"\n')

    config = derived(binding, "h0,h1,h2", tmp_path)

    assert [b["rank"] for b in config["rank_bindings"]] == [0, 1, 2]
    assert config["mesh_graph_desc_path"] == str(own_mgd)


def test_prefill_mgd_overrides_the_rank_count_default(tmp_path):
    own_mgd = TOPO / "pipeline_prefill_4galaxy_connected_fabric2d_mesh_graph_descriptor.textproto"

    config = derived(COMBINED, "h0,h1,h2", tmp_path, PREFILL_MGD=str(own_mgd))

    assert len(config["rank_bindings"]) == 3
    assert config["mesh_graph_desc_path"] == str(own_mgd)


def test_a_pinned_descriptor_wins_over_prefill_mgd(tmp_path):
    """Priority is binding first, then PREFILL_MGD, then the rank-count default."""
    code, out, err = dry_run(
        INTRAGALAXY,
        "h0,h1",
        tmp_path,
        PREFILL_MGD=str(TOPO / "pipeline_prefill_8galaxy_connected_mesh_graph_descriptor.textproto"),
    )

    assert code == 0, err
    config = yaml.safe_load(out.partition("\n")[2])
    assert config["mesh_graph_desc_path"] == yaml.safe_load(INTRAGALAXY.read_text())["mesh_graph_desc_path"]


@pytest.mark.parametrize(
    "hosts, env, message",
    [
        ("h0:2,h1:2", {}, "puts 2 ranks on one host"),
        ("h0:x", {}, "non-numeric slot count"),
        ("h0,h1,h2", {}, "no default mesh graph descriptor for 3 rank(s)"),
        ("h0,h1", {"PREFILL_MGD": "does_not_exist.textproto"}, "mesh graph descriptor not found"),
    ],
    ids=["multi_slot_host", "bad_slot_count", "unmapped_rank_count", "missing_descriptor"],
)
def test_ambiguous_or_impossible_derivation_fails_loudly(hosts, env, message, tmp_path):
    code, _, err = dry_run(COMBINED, hosts, tmp_path, **env)

    assert code != 0, "expected a non-zero exit"
    assert message in err


def test_concurrent_derivations_do_not_share_a_file(tmp_path):
    """Same binding, same rank count, two launches -- each keeps its own artifact."""
    procs = [
        subprocess.Popen(
            [str(LAUNCHER), str(COMBINED), "h0,h1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PP_DRY_RUN": "1", "PP_BINDING_OUT_DIR": str(tmp_path)},
        )
        for _ in range(4)
    ]
    for proc in procs:
        assert proc.wait() == 0, proc.stderr.read()

    assert len(list(tmp_path.iterdir())) == 4
