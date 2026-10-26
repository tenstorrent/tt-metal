import pytest

from codeowner_bypass import Config, GhError, Gh, bless, enqueue, evaluate_enqueue, not_green

BOT = "tenstorrent-github-bot"
REQUIRED = ["PR Gate Status", "Merge Gate Status", "check-doc"]
SHA = "a" * 40
CFG = Config()


class FakeGh(Gh):
    """In-memory GitHub. Label changes made through the bot append `labeled` events by the bot."""

    def __init__(self):
        self.members = {"infra-dev"}
        self.state = "OPEN"
        self.draft = False
        self.in_queue = False
        self.auto_merge = True
        self.labels = []
        self.events = []  # actor logins of `labeled` events, oldest first
        self.head = SHA
        self.required = list(REQUIRED)
        self.runs = [
            {"name": n, "id": i, "status": "completed", "conclusion": "success"} for i, n in enumerate(REQUIRED)
        ]
        self.statuses = []
        self.enqueue_error = None
        self.comments = []
        self.enqueued = []

    def hand_label(self, actor="some-writer"):
        """Someone with write access adds the label by hand."""
        if CFG.label not in self.labels:
            self.labels.append(CFG.label)
        self.events.append(actor)

    def set_run(self, name, status, conclusion=None, run_id=100):
        self.runs = [r for r in self.runs if r["name"] != name]
        self.runs.append({"name": name, "id": run_id, "status": status, "conclusion": conclusion})

    def viewer_login(self):
        return BOT

    def team_state(self, org, team, user):
        return "active" if user in self.members else "none"

    def pull_request(self, number):
        return {
            "id": "PR_node",
            "state": self.state,
            "isDraft": self.draft,
            "headRefOid": self.head,
            "baseRefName": "main",
            "isInMergeQueue": self.in_queue,
            "autoMergeRequest": {"enabledAt": "t"} if self.auto_merge else None,
            "labels": list(self.labels),
        }

    def open_pr_for_sha(self, sha):
        return 7 if sha == self.head else None

    def label_events(self, number, label):
        return list(self.events)

    def required_contexts(self, base_ref):
        return self.required

    def check_runs(self, sha):
        return self.runs

    def commit_statuses(self, sha):
        return self.statuses

    def enqueue(self, node_id):
        if self.enqueue_error:
            raise GhError(self.enqueue_error)
        self.enqueued.append(node_id)
        return "3"

    def comment(self, number, body):
        self.comments.append(body)

    def react(self, comment_id, content):
        pass

    def ensure_label(self, label, description):
        pass

    def remove_label(self, number, label):
        if label in self.labels:
            self.labels.remove(label)

    def add_label(self, number, label):
        self.labels.append(label)
        self.events.append(BOT)


@pytest.fixture
def gh():
    fake = FakeGh()
    bless(fake, CFG, 7, "1", "infra-dev", "/codeowners bypass")
    fake.comments.clear()
    return fake


def decide(gh, **ids):
    ids = ids or {"event_pr_number": 7}
    return evaluate_enqueue(gh, CFG, **ids)


# ---- bless ----------------------------------------------------------------------------------


def test_bless_by_team_member_labels_as_bot():
    gh = FakeGh()
    bless(gh, CFG, 7, "1", "infra-dev", "/codeowners bypass")
    assert gh.labels == [CFG.label] and gh.events == [BOT]
    assert "Blessed" in gh.comments[0]


def test_bless_by_non_member_is_refused():
    gh = FakeGh()
    bless(gh, CFG, 7, "1", "random-dev", "/codeowners bypass")
    assert gh.labels == [] and "only members" in gh.comments[0]


@pytest.mark.parametrize("body", ["/codeowners ping", "please /codeowners bypass this", "/codeowners bypass now"])
def test_bless_ignores_other_comments(body):
    gh = FakeGh()
    bless(gh, CFG, 7, "1", "infra-dev", body)
    assert gh.labels == [] and gh.comments == []


def test_bless_closed_pr_is_refused():
    gh = FakeGh()
    gh.state = "CLOSED"
    bless(gh, CFG, 7, "1", "infra-dev", "/codeowners bypass")
    assert gh.labels == [] and "not open" in gh.comments[0]


# ---- label provenance -----------------------------------------------------------------------


def test_blessed_pr_with_green_checks_is_enqueued(gh):
    assert enqueue(gh, CFG, event_pr_number=7) == 0
    assert gh.enqueued == ["PR_node"] and "queue position 3" in gh.comments[0]


def test_hand_added_label_does_not_count():
    gh = FakeGh()
    gh.hand_label()
    target, reason = decide(gh)
    assert target is None and "some-writer" in reason


def test_hand_label_after_blessing_does_not_count(gh):
    gh.hand_label()
    target, reason = decide(gh)
    assert target is None and "not the bot" in reason


def test_rebless_after_hand_label_restores_eligibility(gh):
    gh.hand_label()
    bless(gh, CFG, 7, "1", "infra-dev", "/codeowners bypass")
    assert gh.events[-1] == BOT
    assert decide(gh)[1] is None


def test_unblessed_pr_is_not_enqueued():
    gh = FakeGh()
    assert "no codeowner-bypass label" in decide(gh)[1]


# ---- the rest of the condition --------------------------------------------------------------


@pytest.mark.parametrize(
    "attr, value, fragment",
    [
        ("auto_merge", False, "auto-merge is not enabled"),
        ("draft", True, "draft"),
        ("state", "CLOSED", "not open"),
        ("in_queue", True, "already in the merge queue"),
    ],
)
def test_pr_state_gates(gh, attr, value, fragment):
    setattr(gh, attr, value)
    target, reason = decide(gh)
    assert target is None and fragment in reason


def test_pending_required_check_blocks(gh):
    gh.set_run("check-doc", "in_progress")
    target, reason = decide(gh)
    assert target is None and "check-doc: pending" in reason


def test_failed_required_check_blocks(gh):
    gh.set_run("Merge Gate Status", "completed", "failure")
    target, reason = decide(gh)
    assert target is None and "Merge Gate Status: failure" in reason


def test_missing_required_check_blocks(gh):
    gh.runs = [r for r in gh.runs if r["name"] != "check-doc"]
    assert "check-doc: missing" in decide(gh)[1]


def test_pr_gate_alone_is_not_enough(gh):
    gh.runs = [r for r in gh.runs if r["name"] == "PR Gate Status"]
    target, reason = decide(gh)
    assert target is None and "Merge Gate Status: missing" in reason


def test_rerun_supersedes_earlier_failure(gh):
    gh.set_run("check-doc", "completed", "failure", run_id=50)
    gh.runs.append({"name": "check-doc", "id": 60, "status": "completed", "conclusion": "success"})
    assert decide(gh)[1] is None


def test_non_required_failure_is_ignored(gh):
    gh.runs.append({"name": "some-optional-job", "id": 999, "status": "completed", "conclusion": "failure"})
    assert decide(gh)[1] is None


def test_skipped_required_check_passes(gh):
    gh.set_run("check-doc", "completed", "skipped")
    assert decide(gh)[1] is None


def test_legacy_commit_status_can_satisfy_a_context(gh):
    gh.runs = [r for r in gh.runs if r["name"] != "check-doc"]
    gh.statuses = [{"context": "check-doc", "state": "success"}]
    assert decide(gh)[1] is None


def test_no_required_checks_configured_is_an_error(gh):
    gh.required = []
    with pytest.raises(GhError):
        decide(gh)


# ---- completion order: whichever event lands last must act -----------------------------------


def test_workflow_that_finishes_last_triggers_enqueue_whatever_its_conclusion(gh):
    # The last workflow to finish was skipped / had a non-required job fail. The decision does
    # not look at the triggering run's conclusion, only at the required checks.
    gh.set_run("check-doc", "in_progress")
    assert decide(gh, run_head_sha=SHA)[1] is not None  # not yet
    gh.set_run("check-doc", "completed", "success")
    gh.runs.append({"name": "optional", "id": 500, "status": "completed", "conclusion": "failure"})
    target, reason = decide(gh, run_head_sha=SHA)  # fork PR: found by SHA
    assert reason is None and target == (7, "PR_node")


def test_auto_merge_enabled_last_triggers_enqueue(gh):
    gh.auto_merge = False
    assert decide(gh)[1] is not None
    gh.auto_merge = True
    assert decide(gh)[1] is None


def test_blessing_last_triggers_enqueue():
    gh = FakeGh()
    assert decide(gh)[1] is not None
    bless(gh, CFG, 7, "1", "infra-dev", "/codeowners bypass")
    assert decide(gh)[1] is None


def test_run_for_stale_commit_is_ignored(gh):
    gh.head = "b" * 40
    assert decide(gh, run_pr_number=7, run_head_sha=SHA)[0] is None


def test_run_with_no_open_pr_is_ignored(gh):
    assert decide(gh, run_head_sha="c" * 40) == (None, "no open PR for this event")


# ---- enqueue outcome ------------------------------------------------------------------------


def test_enqueue_failure_is_reported_and_fails_the_job(gh):
    gh.enqueue_error = "Pull request is missing a code owner review"
    assert enqueue(gh, CFG, event_pr_number=7) == 1
    assert gh.enqueued == [] and "code owner review" in gh.comments[0]


def test_skip_is_quiet(gh):
    gh.auto_merge = False
    assert enqueue(gh, CFG, event_pr_number=7) == 0
    assert gh.comments == [] and gh.enqueued == []


# ---- not_green ------------------------------------------------------------------------------


def test_not_green_all_passing():
    runs = [{"name": "a", "id": 1, "status": "completed", "conclusion": "success"}]
    assert not_green(["a"], runs, []) == {}
