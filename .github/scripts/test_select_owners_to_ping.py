"""Unit tests for the availability ranking in `select_owners_to_ping.py`.

Run from this directory: `python3 -m pytest test_select_owners_to_ping.py`.

These cover the ordering only. Which rules count as pending, and who is eligible
for them, is exercised by the workflow itself.
"""

from datetime import datetime, timezone

from select_owners_to_ping import (
    TIER_OOO,
    TIER_REACHABLE,
    TIER_WORKING,
    Selector,
    availability_tier,
    rotation_key,
)


def _utc(year: int, month: int, day: int, hour: int) -> int:
    return int(datetime(year, month, day, hour, tzinfo=timezone.utc).timestamp())


# A Wednesday, 12:00 UTC.
NOW = _utc(2026, 9, 9, 12)

IST = 5 * 3600 + 1800  # 17:30 local, inside working hours
PST = -7 * 3600  # 05:00 local, outside working hours


def _slack_user(tz_offset: int = 0, status_text: str = "", status_emoji: str = "", expiry: int = 0) -> dict:
    return {
        "id": "U1",
        "tz_offset": tz_offset,
        "profile": {
            "status_text": status_text,
            "status_emoji": status_emoji,
            "status_expiration": expiry,
        },
    }


class _FakeDirectory:
    """Stands in for Slack: fixed tiers and presence, no network."""

    def __init__(self, tiers: dict[str, int], active: set[str] | None = None) -> None:
        self.tiers = tiers
        self._active = active or set()

    def tier(self, login: str, _now_utc: int) -> int:
        return self.tiers.get(login, TIER_REACHABLE)

    def active(self, login: str) -> bool:
        return login in self._active


def _selector(tiers: dict[str, int], active: set[str] | None = None, seed: str = "1") -> Selector:
    selector = Selector()
    selector.directory = _FakeDirectory(tiers, active)
    selector.now_utc = NOW
    selector.seed = seed
    return selector


# --- availability_tier --------------------------------------------------------
def test_weekday_inside_local_working_hours_is_the_top_tier() -> None:
    assert availability_tier(_slack_user(tz_offset=IST), NOW) == TIER_WORKING


def test_weekday_outside_local_working_hours_is_reachable_not_working() -> None:
    assert availability_tier(_slack_user(tz_offset=PST), NOW) == TIER_REACHABLE


def test_working_hours_are_local_so_a_weekend_never_qualifies() -> None:
    saturday_noon = _utc(2026, 9, 12, 12)
    assert availability_tier(_slack_user(tz_offset=IST), saturday_noon) == TIER_REACHABLE


def test_out_of_office_status_text_outranks_working_hours() -> None:
    user = _slack_user(tz_offset=IST, status_text="On leave until Monday")
    assert availability_tier(user, NOW) == TIER_OOO


def test_out_of_office_emoji_is_recognised_without_status_text() -> None:
    assert availability_tier(_slack_user(tz_offset=IST, status_emoji=":palm_tree:"), NOW) == TIER_OOO


def test_expired_status_slack_has_not_cleared_is_ignored() -> None:
    stale = _slack_user(tz_offset=IST, status_text="PTO", expiry=NOW - 3600)
    assert availability_tier(stale, NOW) == TIER_WORKING


def test_a_login_with_no_slack_profile_is_reachable_not_out_of_office() -> None:
    # Being unmatchable in Slack is not a reason to never ask someone for review.
    assert availability_tier(None, NOW) == TIER_REACHABLE


# --- pick_two -----------------------------------------------------------------
def test_working_hours_candidates_are_asked_before_off_hours_ones() -> None:
    selector = _selector({"awake_a": TIER_WORKING, "awake_b": TIER_WORKING, "asleep": TIER_REACHABLE})
    assert selector.pick_two(["asleep", "awake_a", "awake_b"]) == sorted(["awake_a", "awake_b"])


def test_the_second_slot_drops_a_tier_rather_than_pinging_one_person() -> None:
    # Stopping after the only in-hours owner would halve the chance of a reply.
    selector = _selector({"awake": TIER_WORKING, "asleep": TIER_REACHABLE})
    assert selector.pick_two(["asleep", "awake"]) == ["awake", "asleep"]


def test_presence_orders_within_a_tier_but_never_across_one() -> None:
    tiers = {"active_off_hours": TIER_REACHABLE, "away_in_hours": TIER_WORKING, "active_in_hours": TIER_WORKING}
    selector = _selector(tiers, active={"active_off_hours", "active_in_hours"})
    assert selector.pick_two(list(tiers)) == ["active_in_hours", "away_in_hours"]


def test_out_of_office_owners_are_skipped_when_anyone_else_is_available() -> None:
    selector = _selector({"away_on_pto": TIER_OOO, "here": TIER_REACHABLE})
    assert selector.pick_two(["away_on_pto", "here"]) == ["here"]


def test_out_of_office_owners_are_pinged_when_they_are_the_only_owners() -> None:
    # An unanswered ping still beats a review request that reaches nobody.
    selector = _selector({"a": TIER_OOO, "b": TIER_OOO})
    assert sorted(selector.pick_two(["a", "b"])) == ["a", "b"]


def test_selection_is_stable_across_reruns_on_the_same_pr() -> None:
    # `/codeowners ping` twice should nudge the same people, not a fresh pair.
    candidates = ["u1", "u2", "u3", "u4", "u5"]
    first = _selector({}, seed="55687").pick_two(candidates)
    second = _selector({}, seed="55687").pick_two(list(reversed(candidates)))
    assert first == second


def test_rotation_differs_between_prs_so_load_is_spread() -> None:
    candidates = ["u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8"]
    picks = {tuple(_selector({}, seed=str(pr)).pick_two(candidates)) for pr in range(30)}
    assert len(picks) > 1


def test_rotation_key_depends_on_both_seed_and_login() -> None:
    assert rotation_key("1", "alice") != rotation_key("2", "alice")
    assert rotation_key("1", "alice") != rotation_key("1", "bob")


def test_no_candidates_selects_nobody() -> None:
    assert _selector({}).pick_two([]) == []
