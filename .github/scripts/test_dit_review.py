from collections import Counter

from dit_review import dir_weights, dit_test_legs, ineligibility_reasons, pick_reviewers, run_covers_dit

MEMBERS = {"alice", "bob", "carol", "dave"}


def changed(filename, status="modified", lines=10, **extra):
    return {"filename": filename, "status": status, "additions": lines, "deletions": 0, "patch": "@@", **extra}


def test_small_fix_is_eligible():
    files = [changed("models/tt_dit/pipelines/wan/pipeline_wan.py", lines=12)]
    assert ineligibility_reasons(files, "alice", MEMBERS, 0, True, True) == []


def test_new_tests_allowed_and_not_counted_toward_size():
    files = [
        changed("models/tt_dit/pipelines/wan/pipeline_wan.py", lines=30),
        changed("models/tt_dit/tests/models/wan/test_new.py", status="added", lines=500),
        changed("models/tt_dit/tests/models/wan/conftest.py", status="added", lines=50),
    ]
    assert ineligibility_reasons(files, "alice", MEMBERS, 0, True, True) == []


def test_test_files_outside_tests_dir_count_toward_size():
    files = [
        changed("models/tt_dit/pipelines/wan/conftest.py", status="added", lines=30),
        changed("models/tt_dit/pipelines/wan/test_helper.py", status="added", lines=30),
    ]
    assert ineligibility_reasons(files, "alice", MEMBERS, 0, True, True) == ["60 non-test lines (max 40)"]


def test_renames_check_the_old_path():
    moved_in = changed("models/tt_dit/pipelines/wan/x.py", status="renamed", previous_filename="ttnn/x.py")
    assert ineligibility_reasons([moved_in], "alice", MEMBERS, 0, True, True) == [
        "changes files outside models/tt_dit/"
    ]
    moved_out_of_core = changed(
        "models/tt_dit/pipelines/wan/x.py", status="renamed", previous_filename="models/tt_dit/layers/x.py"
    )
    assert ineligibility_reasons([moved_out_of_core], "alice", MEMBERS, 0, True, True) == [
        "changes core code (layers, parallel, utils)"
    ]
    moved_test = changed(
        "models/tt_dit/pipelines/wan/x.py", status="renamed", previous_filename="models/tt_dit/tests/x.py"
    )
    assert ineligibility_reasons([moved_test], "alice", MEMBERS, 0, True, True) == ["changes existing tests"]


def test_binary_files_and_base_branch():
    binary = changed("models/tt_dit/pipelines/wan/ref.bin")
    del binary["patch"]
    assert ineligibility_reasons([binary], "alice", MEMBERS, 0, True, True, base_ok=False) == [
        "base branch is not the default branch",
        "changes binary or very large files",
    ]


def test_ineligibility_reasons():
    files = [
        changed("models/tt_dit/tests/models/wan/test_pipeline_wan.py"),
        changed("models/tt_dit/layers/linear.py"),
        changed("models/tt_dit/pipelines/a.py", lines=20),
        changed("models/tt_dit/pipelines/b.py", lines=20),
        changed("models/tt_dit/pipelines/c.py"),
        changed("ttnn/foo.py"),
    ]
    assert ineligibility_reasons(files, "outsider", MEMBERS, 2, False, False) == [
        "author has not added the dit-auto-merge label",
        "author not on DiT team",
        "changes files outside models/tt_dit/",
        "changes existing tests",
        "changes core code (layers, parallel, utils)",
        "5 non-test files (max 3)",
        "70 non-test lines (max 40)",
        "2 unresolved review threads",
        "no passing All Model Tests run covering DiT on the latest commit",
    ]


def test_unchecked_tests_are_not_a_reason():
    files = [changed("models/tt_dit/pipelines/wan/pipeline_wan.py")]
    assert ineligibility_reasons(files, "alice", MEMBERS, 0, False, None) == [
        "author has not added the dit-auto-merge label"
    ]


def test_dit_test_legs(tmp_path):
    unit = tmp_path / "unit.yaml"
    unit.write_text(
        "- name: DiT\n  cmd: pytest models/tt_dit/tests/unit/\n  model: dit-common\n"
        "  skus:\n    wh_n150:\n      tier: 1\n    bh_quietbox_2:\n      tier: 2\n"
        "- name: Llama\n  cmd: pytest models/tt_transformers/\n  model: llama3\n  skus:\n    wh_n150:\n      tier: 1\n"
    )
    e2e = tmp_path / "e2e.yaml"
    e2e.write_text(
        "- name: Wan\n  cmd: |\n    pytest models/tt_dit/tests/models/wan/\n  model: wan-2.2-t2v\n"
        "  skus:\n    wh_galaxy:\n      tier: 1\n"
    )
    assert sorted(dit_test_legs({"unit": str(unit), "e2e": str(e2e)})) == [
        ("e2e", "wan-2.2-t2v", "wh_galaxy", 1),
        ("unit", "dit-common", "bh_quietbox_2", 2),
        ("unit", "dit-common", "wh_n150", 1),
    ]


def test_run_covers_dit():
    legs = [("unit", "dit-common", "wh_n150", 1), ("e2e", "wan-2.2-t2v", "bh_galaxy", 2)]

    def title(model="all", skus="all", tests="unit,e2e", tiers="1,2"):
        return f"All Model Tests — Tiers[{tiers}] Tests[{tests}] Model[{model}] SKUs[{skus}]"

    assert run_covers_dit(title(model="dit-common"), legs)
    assert run_covers_dit(title(model="llama,WAN"), legs)
    assert run_covers_dit(title(), legs)
    assert run_covers_dit(title(skus="bh_all"), legs)
    assert run_covers_dit(title(skus=""), legs)
    assert not run_covers_dit(title(model="llama"), legs)
    # Model[all] on SKUs, test types or tiers without DiT legs.
    assert not run_covers_dit(title(skus="wh_galaxy,bh_p150"), legs)
    assert not run_covers_dit(title(tests="sweep"), legs)
    assert not run_covers_dit(title(tiers="3"), legs)
    assert not run_covers_dit(title(tests="e2e", skus="wh_all"), legs)
    # The exabox SKUs are in no group.
    assert not run_covers_dit(title(), [("unit", "dit-common", "bh_sc4", 1)])
    assert run_covers_dit(title(skus="bh_sc4"), [("unit", "dit-common", "bh_sc4", 1)])


def test_dir_weights_ignores_non_dit_files():
    files = [
        "models/tt_dit/pipelines/wan/a.py",
        "models/tt_dit/pipelines/wan/b.py",
        "models/tt_dit/layers/c.py",
        "ttnn/d.py",
    ]
    assert dir_weights(files) == Counter({"models/tt_dit/pipelines/wan": 2, "models/tt_dit/layers": 1})


def test_picks_highest_scores_and_skips_author_and_non_members():
    scores = Counter({"alice": 5, "bob": 3, "carol": 1, "outsider": 10})
    assert pick_reviewers(scores, MEMBERS, "alice", set(), 2, seed=1) == ["bob", "carol"]


def test_tops_up_when_dit_member_already_requested():
    scores = Counter({"alice": 5, "bob": 3})
    assert pick_reviewers(scores, MEMBERS, "dave", {"carol"}, 2, seed=1) == ["alice"]
    assert pick_reviewers(scores, MEMBERS, "dave", {"carol", "bob"}, 2, seed=1) == []


def test_falls_back_to_random_member_without_history():
    picks = pick_reviewers(Counter(), MEMBERS, "alice", set(), 2, seed=7)
    assert len(picks) == 1 and picks[0] in MEMBERS - {"alice"}
    assert picks == pick_reviewers(Counter(), MEMBERS, "alice", set(), 2, seed=7)


def test_no_fallback_when_dit_member_already_present():
    assert pick_reviewers(Counter(), MEMBERS, "alice", {"bob"}, 2, seed=7) == []


def test_non_member_requests_do_not_count_toward_guarantee():
    picks = pick_reviewers(Counter(), MEMBERS, "alice", {"outsider"}, 2, seed=7)
    assert len(picks) == 1 and picks[0] in MEMBERS
