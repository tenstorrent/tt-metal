from collections import Counter

from dit_review import dir_weights, dit_approvers, ineligibility_reasons, pick_reviewers

MEMBERS = {"alice", "bob", "carol", "dave"}


def review(login, state):
    return {"user": {"login": login}, "state": state}


def changed(filename, status="modified", lines=10):
    return {"filename": filename, "status": status, "additions": lines, "deletions": 0}


def test_small_fix_is_eligible():
    files = [changed("models/tt_dit/pipelines/wan/pipeline_wan.py", lines=12)]
    assert ineligibility_reasons(files, "alice", MEMBERS, 0) == []


def test_new_tests_allowed_and_not_counted_toward_size():
    files = [
        changed("models/tt_dit/pipelines/wan/pipeline_wan.py", lines=30),
        changed("models/tt_dit/tests/models/wan/test_new.py", status="added", lines=500),
    ]
    assert ineligibility_reasons(files, "alice", MEMBERS, 0) == []


def test_ineligibility_reasons():
    files = [
        changed("models/tt_dit/tests/models/wan/test_pipeline_wan.py"),
        changed("models/tt_dit/layers/linear.py"),
        changed("models/tt_dit/pipelines/a.py", lines=20),
        changed("models/tt_dit/pipelines/b.py", lines=20),
        changed("models/tt_dit/pipelines/c.py"),
        changed("ttnn/foo.py"),
    ]
    assert ineligibility_reasons(files, "outsider", MEMBERS, 2) == [
        "author not on DiT team",
        "changes files outside models/tt_dit/",
        "changes existing tests",
        "changes core code (layers, parallel, utils)",
        "5 non-test files (max 3)",
        "70 non-test lines (max 40)",
        "2 unresolved review threads",
    ]


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


def test_approvers_uses_latest_decisive_review():
    reviews = [
        review("alice", "APPROVED"),
        review("alice", "COMMENTED"),
        review("bob", "APPROVED"),
        review("bob", "CHANGES_REQUESTED"),
        review("carol", "APPROVED"),
        review("carol", "DISMISSED"),
        review("outsider", "APPROVED"),
        {"user": None, "state": "APPROVED"},
    ]
    assert dit_approvers(reviews, MEMBERS) == ["alice"]
