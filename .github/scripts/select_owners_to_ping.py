#!/usr/bin/env python3
"""Select which CodeOwners to ping for a pending PR review.

Extracted from the inline `Select owners for notification` step of
`.github/workflows/codeowners-group-analysis.yaml`. Which rules are pending, and
which owners are eligible for them, is unchanged from the original bash. Only the
final pick within a rule differs: the two slots used to be filled at random and
are now filled by availability, so a review request lands on someone who can act
on it today.

Inputs (environment variables):
  TEAM_MEMBERS        Contents of ${RUNNER_TEMP}/team_members.txt, i.e. a
                      '|'-separated list of "@org/team:member1,member2,..."
                      entries. A team's members value may instead be one of the
                      sentinel error strings (insufficient-permissions,
                      team-not-found, unauthorized, api-error, no-members), in
                      which case that team contributes no members.
  TEAMS               '§'-separated list of "@org/team:file1,file2,..." entries
                      (from the analyze-codeowners job).
  INDIVIDUALS         '§'-separated list of
                      "pattern:owner1|team1,owner2|team2,...:file1,file2,..."
                      entries (from the analyze-codeowners job).
  APPROVED_REVIEWERS  Newline- or comma-separated logins that already approved.
  MOREH_TEAM_MEMBERS  thirdparty-moreh team members (excluded from pinging).
  PR_AUTHOR_LOGIN     PR author login (excluded from pinging).
  GITHUB_OUTPUT       Path to the step output file (GitHub Actions).

Optional inputs, used to rank candidates by availability. Each one is a pure
improvement: with none of them set the selection still works, it just treats
every candidate as equally reachable and falls back to the rotation order.

  SLACK_USERS_FILE    Path to the `users.list` dump the notify job already
                      writes (${RUNNER_TEMP}/slack_users.json). Supplies each
                      member's timezone offset and status, i.e. working hours
                      and out-of-office.
  SLACK_BOT_TOKEN     Slack bot token, used only for `users.getPresence` to
                      break ties within an availability tier.
  GITHUB_TOKEN        Used to resolve a login's full name when it does not match
                      a Slack handle directly, mirroring the workflow's own
                      lookup so the tier is computed for the person who will
                      actually be mentioned.
  PR_NUMBER           Rotation seed. Keeps re-runs of `/codeowners ping` on one
                      PR reaching the same people instead of a fresh pair.

Outputs (appended to $GITHUB_OUTPUT):
  selected-owners        comma-separated, sorted+deduped individual logins
  selected-slack-groups  comma-separated Slack group IDs
  no-owners-available    "true" iff both of the above are empty
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

# --- team -> Slack group mapping (ID + handle) --------------------------------
# Kept identical to the original workflow's get_slack_group_id / _handle.
SLACK_GROUPS: dict[str, tuple[str, str]] = {
    "@tenstorrent/metalium-developers-infra": (
        "S0985AN7TC5",
        "metalium-developers-infra",
    ),
    "@tenstorrent/metalium-developers-ttnn-core": (
        "S0988UJEW8K",
        "metalium-developers-ttnn-core",
    ),
    "@tenstorrent/metalium-ttnn-core-team": (
        "S0988UJEW8K",
        "metalium-developers-ttnn-core",
    ),
    "@tenstorrent/metalium-developers-convolutions": (
        "S09DNR6NAG4",
        "metalium-developers-convolutions",
    ),
    "@tenstorrent/metalium-developers-ops-data-movement": (
        "S09QQRK1CF8",
        "metalium-developers-ops-data-movement",
    ),
    "@tenstorrent/metalium-developers-eltwise": (
        "S0ABKSS1D3R",
        "metalium-developers-eltwise",
    ),
}

TEAM_MEMBER_ERROR_SENTINELS = {
    "insufficient-permissions",
    "team-not-found",
    "unauthorized",
    "api-error",
    "no-members",
}

BYPASS_TEAM = "@tenstorrent/codeowner-bypass"
API_OWNERS_TEAM = "@tenstorrent/metalium-api-owners"
API_REQUIRED_REVIEWER = "akerteszTT"

# --- availability ladder ------------------------------------------------------
# Owners of a pending rule are ranked before the two ping slots are filled, so a
# review request lands on someone who can act on it today. Out-of-office is the
# only tier that is skipped rather than deprioritised, and even that is undone
# when skipping would leave the rule with nobody: an unanswered ping still beats
# no ping.
TIER_WORKING = 0  # inside local working hours, no out-of-office status
TIER_REACHABLE = 1  # no out-of-office status, but outside working hours
TIER_OOO = 2  # marked out of office

WORK_START_HOUR = 9
WORK_END_HOUR = 18  # exclusive, so 09:00-17:59 local

OOO_EMOJI = {
    ":palm_tree:",
    ":ooo:",
    ":airplane:",
    ":no_entry:",
    ":face_with_thermometer:",
    ":hospital:",
}
OOO_TEXT = re.compile(
    r"\b(ooo|out of office|off today|pto|vacation|holiday|on leave|annual leave|sick|parental)\b",
    re.IGNORECASE,
)

# Logins whose Slack profile the workflow's fuzzy matcher cannot find. Kept in
# sync with the `find_slack_user_id` case list in
# `.github/workflows/codeowners-group-analysis.yaml` so that the tier is computed
# for the same person the ping step goes on to mention.
SLACK_ID_OVERRIDES: dict[str, str] = {
    "mradosavljevicTT": "U0837MYG788",
    "nsextonTT": "U08TVGQGGAE",
    "ncvetkovicTT": "U07AUABTEP6",
    "jvegaTT": "U07M7QZ0BQA",
}

# users.getPresence is rate limited and only ever breaks a tie, so cap the calls
# and treat anyone past the cap as away rather than slowing the job down.
PRESENCE_LOOKUP_LIMIT = 40


def log(msg: str) -> None:
    print(msg, flush=True)


def slack_group_id(team: str) -> str:
    entry = SLACK_GROUPS.get(team)
    return entry[0] if entry else ""


def split_nonempty(value: str, sep: str) -> list[str]:
    return [p for p in value.split(sep) if p != ""]


def sorted_files_key(files_csv: str) -> str:
    """Reproduce: tr ',' '\\n' | sort | tr '\\n' ',' | sed 's/,$//'."""
    parts = [f for f in files_csv.split(",") if f != ""]
    return ",".join(sorted(parts))


def approved(login: str, approved_reviewers: str) -> bool:
    """Substring match, matching the original `grep -q "$member"`."""
    return login != "" and login in approved_reviewers


def approved_exact(login: str, approved_reviewers: str) -> bool:
    """Exact match, matching the original per-file `... | tr ',' '\\n' | grep -qx`."""
    if login == "":
        return False
    return login in approved_reviewers.replace(",", "\n").split("\n")


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def is_ooo(user: dict, now_utc: int) -> bool:
    """Whether a Slack profile is flagged out of office right now."""
    profile = user.get("profile") or {}
    expiry = profile.get("status_expiration") or 0
    if expiry and expiry <= now_utc:
        return False  # an expired status Slack has not cleared yet
    if (profile.get("status_emoji") or "") in OOO_EMOJI:
        return True
    return bool(OOO_TEXT.search(profile.get("status_text") or ""))


def local_time(user: dict, now_utc: int) -> tuple[int, int]:
    """(hour, weekday) where the member is, Monday = 0.

    Slack reports `tz_offset` as the member's *current* offset from UTC, DST
    already applied, so this needs no timezone database.
    """
    moment = datetime.fromtimestamp(now_utc + int(user.get("tz_offset") or 0), tz=timezone.utc)
    return moment.hour, moment.weekday()


def availability_tier(user: dict | None, now_utc: int) -> int:
    """Rank a candidate for pinging. Lower is better."""
    if user is None:
        # No Slack match. We cannot tell what their day looks like, so treat them
        # as reachable rather than penalising them into the out-of-office tier.
        return TIER_REACHABLE
    if is_ooo(user, now_utc):
        return TIER_OOO
    hour, weekday = local_time(user, now_utc)
    if weekday < 5 and WORK_START_HOUR <= hour < WORK_END_HOUR:
        return TIER_WORKING
    return TIER_REACHABLE


def rotation_key(seed: str, login: str) -> str:
    """Stable, evenly-spread tiebreak between equally good candidates.

    Sorting on a digest replaces the previous `random` pick with something that
    reaches the same people when `/codeowners ping` is re-run on a PR rather than
    spraying a new pair each time, while still spreading the load evenly across
    PRs. It is a rotation order, not a secret.
    """
    return hashlib.sha256(f"{seed}:{login}".encode()).hexdigest()


def _name_fields(user: dict) -> list[str]:
    profile = user.get("profile") or {}
    return [
        user.get("real_name") or "",
        profile.get("real_name") or "",
        profile.get("display_name") or "",
    ]


class SlackDirectory:
    """Resolve a GitHub login to the Slack profile the ping step will mention.

    The match order mirrors `find_slack_user_id` in
    `.github/workflows/codeowners-group-analysis.yaml`: hardcoded overrides, then
    the GitHub full name, then the login, then a word-by-word match accepted only
    when exactly one user matches. Matching the workflow matters — ranking a
    candidate by someone else's timezone would be worse than not ranking at all.

    Every failure path returns None, which `availability_tier` treats as merely
    reachable, so a missing token or an unreachable API degrades the ordering
    instead of breaking the selection.
    """

    def __init__(self, users: list[dict], github_token: str = "", slack_token: str = "") -> None:
        self.users = users
        self.github_token = github_token
        self.slack_token = slack_token
        self.by_id = {u.get("id"): u for u in users if u.get("id")}
        self._user_cache: dict[str, dict | None] = {}
        self._full_names: dict[str, str] = {}
        self._presence: dict[str, bool] = {}
        self._presence_calls = 0

    @classmethod
    def from_env(cls) -> "SlackDirectory":
        users: list[dict] = []
        path = os.environ.get("SLACK_USERS_FILE", "")
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, list):
                    users = [u for u in loaded if isinstance(u, dict)]
                log(f"DEBUG select-owners: loaded {len(users)} Slack users for availability ranking")
            except (OSError, ValueError) as exc:
                log(f"WARNING select-owners: could not read {path} ({exc}); ranking by rotation only")
        else:
            log("DEBUG select-owners: no Slack user list; ranking by rotation only")
        return cls(
            users,
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            slack_token=os.environ.get("SLACK_BOT_TOKEN", ""),
        )

    # -- GitHub full name (only consulted when the login does not match) -------
    def full_name(self, login: str) -> str:
        if login in self._full_names:
            return self._full_names[login]
        name = ""
        if self.github_token:
            request = urllib.request.Request(f"https://api.github.com/users/{login}")
            request.add_header("Authorization", f"Bearer {self.github_token}")
            request.add_header("Accept", "application/vnd.github+json")
            request.add_header("User-Agent", "tt-metal-codeowners-ping")
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    name = (json.load(response).get("name") or "").strip()
            except (urllib.error.URLError, ValueError, TimeoutError) as exc:
                log(f"DEBUG select-owners: no GitHub profile for {login} ({exc})")
        self._full_names[login] = name
        return name

    def _match(self, login: str) -> dict | None:
        override = SLACK_ID_OVERRIDES.get(login)
        if override:
            return self.by_id.get(override)

        lowered = login.lower()
        for user in self.users:
            profile = user.get("profile") or {}
            if (user.get("name") or "").lower() == lowered or (profile.get("display_name") or "").lower() == lowered:
                return user

        full_name = self.full_name(login)
        if not full_name:
            return None
        for user in self.users:
            if full_name in _name_fields(user):
                return user

        # Word-by-word, and only when a word identifies exactly one person. A
        # surname shared by two colleagues has to fall through rather than guess.
        for word in full_name.split():
            if len(word) < 3:
                continue
            needle = word.lower()
            hits = [u for u in self.users if any(needle in field.lower() for field in _name_fields(u))]
            if len(hits) == 1:
                return hits[0]
        return None

    def user(self, login: str) -> dict | None:
        if login not in self._user_cache:
            self._user_cache[login] = self._match(login)
        return self._user_cache[login]

    def tier(self, login: str, now_utc: int) -> int:
        return availability_tier(self.user(login), now_utc)

    def active(self, login: str) -> bool:
        """Slack presence, used only to order people within a tier.

        Never a filter: `away` means no recent keyboard input, so letting it
        exclude anyone would risk a PR that pings nobody because a screensaver
        came on.
        """
        if login in self._presence:
            return self._presence[login]
        result = False
        user = self.user(login)
        if user and self.slack_token and self._presence_calls < PRESENCE_LOOKUP_LIMIT:
            self._presence_calls += 1
            request = urllib.request.Request(f"https://slack.com/api/users.getPresence?user={user.get('id')}")
            request.add_header("Authorization", f"Bearer {self.slack_token}")
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    payload = json.load(response)
                result = bool(payload.get("ok")) and payload.get("presence") == "active"
            except (urllib.error.URLError, ValueError, TimeoutError) as exc:
                log(f"DEBUG select-owners: presence lookup failed for {login} ({exc})")
        self._presence[login] = result
        return result


class Selector:
    def __init__(self) -> None:
        # TEAM_MEMBERS may be passed inline or via a file path (the original
        # step read ${RUNNER_TEMP}/team_members.txt). Prefer the file when set.
        members_file = os.environ.get("TEAM_MEMBERS_FILE", "")
        if members_file and os.path.isfile(members_file):
            with open(members_file, encoding="utf-8") as fh:
                self.team_members_raw = fh.read().strip()
        else:
            self.team_members_raw = os.environ.get("TEAM_MEMBERS", "")
        self.teams = os.environ.get("TEAMS", "")
        self.individuals = os.environ.get("INDIVIDUALS", "")
        self.approved_reviewers = os.environ.get("APPROVED_REVIEWERS", "")
        self.moreh_members = os.environ.get("MOREH_TEAM_MEMBERS", "")
        self.pr_author = os.environ.get("PR_AUTHOR_LOGIN", "")

        # Rotation seed and availability data. Both are optional: without them
        # every candidate ranks the same and the pick is the rotation order.
        self.seed = os.environ.get("PR_NUMBER", "")
        self.now_utc = int(datetime.now(tz=timezone.utc).timestamp())
        self.directory = SlackDirectory.from_env()

        # Parse "team:members" file once into a lookup (first entry wins, as the
        # original `grep "^$team:" | head -1` did).
        self._team_to_members: dict[str, str] = {}
        for entry in split_nonempty(self.team_members_raw, "|"):
            team, _, members = entry.partition(":")
            if team and team not in self._team_to_members:
                self._team_to_members[team] = members

    # -- exclusion predicates (identical semantics to the bash helpers) --------
    def is_moreh_member(self, username: str) -> bool:
        return bool(self.moreh_members) and username in self.moreh_members

    def is_pr_author(self, username: str) -> bool:
        return username == self.pr_author

    def team_owners(self, team: str) -> list[str]:
        """Return usable members for a team, or [] for missing/sentinel."""
        members = self._team_to_members.get(team)
        if members is None or members in TEAM_MEMBER_ERROR_SENTINELS:
            return []
        return [m for m in members.split(",") if m != ""]

    # -- STEP 1: file-set -> combined members / teams --------------------------
    def build_file_maps(self):
        files_to_members: dict[str, str] = {}
        files_to_teams: dict[str, str] = {}

        for team_entry in split_nonempty(self.teams, "§"):
            team = team_entry.split(":", 1)[0]
            if not team or team == BYPASS_TEAM:
                continue
            team_files = team_entry.split(":", 1)[1] if ":" in team_entry else ""
            key = sorted_files_key(team_files)

            if files_to_teams.get(key):
                files_to_teams[key] += f"|{team}"
            else:
                files_to_teams[key] = team

            owners = self.team_owners(team)
            if owners:
                joined = ",".join(owners)
                if files_to_members.get(key):
                    files_to_members[key] += f",{joined}"
                else:
                    files_to_members[key] = joined

        for pattern_group in split_nonempty(self.individuals, "§"):
            # Format: pattern:owner1|team1,owner2|team2,...:files
            # `files` is the last colon field; `owners` is everything between
            # the first and last colon (matching the rev/cut logic).
            head, _, files = pattern_group.rpartition(":")
            owners = head.split(":", 1)[1] if ":" in head else ""
            key = sorted_files_key(files)

            usernames = [pair.split("|", 1)[0] for pair in owners.split(",") if pair != ""]
            joined = ",".join(usernames)
            if joined:
                if files_to_members.get(key):
                    files_to_members[key] += f",{joined}"
                else:
                    files_to_members[key] = joined

        return files_to_members, files_to_teams

    # -- STEP 2: combined approval per file set --------------------------------
    def compute_approval(self, files_to_members: dict[str, str]) -> dict[str, bool]:
        def has_overlap(a: str, b: str) -> bool:
            if not a or not b:
                return False
            return bool(set(a.split(",")) & set(b.split(",")))

        result: dict[str, bool] = {}
        for key, combined in files_to_members.items():
            has_approval = any(approved(m, self.approved_reviewers) for m in combined.split(",") if m)
            if has_approval:
                log(f"DEBUG select-owners: Files [{key}] approved (combined)")
            else:
                for other_key, other_members in files_to_members.items():
                    if other_key == key or not has_overlap(key, other_key):
                        continue
                    if any(approved(m, self.approved_reviewers) for m in other_members.split(",") if m):
                        has_approval = True
                        log(f"DEBUG select-owners: Files [{key}] approved via " f"overlapping set [{other_key}]")
                        break
            result[key] = has_approval
        return result

    # -- pick up to 2 owners, most available first ------------------------------
    def pick_two(self, candidates: list[str]) -> list[str]:
        """Pick the two owners most likely to act on the request.

        Out-of-office candidates are dropped first, but only if that leaves
        someone: a rule where everyone is away still gets pinged. The two slots
        are then filled in tier order rather than stopping after one in-hours
        reviewer, because asking a single person halves the chance of a reply.
        Slack presence orders people inside a tier, and `rotation_key` breaks
        what is left so the choice is stable across re-runs on the same PR.
        """
        if not candidates:
            return []
        pool = [c for c in candidates if self.directory.tier(c, self.now_utc) < TIER_OOO]
        if not pool:
            log("All candidates are marked out of office; pinging them anyway")
            pool = list(candidates)
        ordered = sorted(
            pool,
            key=lambda c: (
                self.directory.tier(c, self.now_utc),
                0 if self.directory.active(c) else 1,
                rotation_key(self.seed, c),
            ),
        )
        chosen = ordered[:2]
        for login in chosen:
            tier = self.directory.tier(login, self.now_utc)
            log(f"DEBUG select-owners: {login} tier={tier} active={self.directory.active(login)}")
        return chosen

    def unapproved_filtered(self, candidates: list[str]) -> list[str]:
        """Drop approved reviewers, moreh members, and the PR author."""
        out: list[str] = []
        for username in candidates:
            if username == "" or approved(username, self.approved_reviewers):
                continue
            if self.is_moreh_member(username):
                log(f"Excluding {username} (thirdparty-moreh team member)")
            elif self.is_pr_author(username):
                log(f"Excluding {username} (PR author)")
            else:
                out.append(username)
        return out

    # -- STEP 3: selection -----------------------------------------------------
    def select(self):
        files_to_members, _ = self.build_file_maps()
        files_has_approval = self.compute_approval(files_to_members)

        selected_owners: list[str] = []
        selected_slack_groups: list[str] = []

        # Teams
        for team_entry in split_nonempty(self.teams, "§"):
            team = team_entry.split(":", 1)[0]
            if not team or team == BYPASS_TEAM:
                continue
            team_files = team_entry.split(":", 1)[1] if ":" in team_entry else ""

            # Skip team only if every file it owns is covered by an approval.
            all_files_approved = True
            for tfile in (f for f in team_files.split(",") if f):
                file_approved = False
                for fset, members in files_to_members.items():
                    if tfile in fset.split(","):
                        if any(approved_exact(m, self.approved_reviewers) for m in members.split(",") if m):
                            file_approved = True
                            break
                if not file_approved:
                    all_files_approved = False
                    break
            if team_files and all_files_approved:
                log(f"Team {team}: all files have an approved owner, skipping")
                continue

            gid = slack_group_id(team)
            if gid:
                selected_slack_groups.append(gid)
                log(f"Added pending Slack group: {team} -> {gid}")
                continue

            # No Slack group -> individual selection from team members.
            log(f"Team {team} has no Slack group, using individual selection")
            unapproved = self.unapproved_filtered(self.team_owners(team))

            # metalium-api-owners: always include akerteszTT for tt_metal/api/ files.
            if team == API_OWNERS_TEAM:
                files_under_api = any(f.lstrip("./").startswith("tt_metal/api/") for f in team_files.split(",") if f)
                if files_under_api and API_REQUIRED_REVIEWER in unapproved:
                    selected_owners.append(API_REQUIRED_REVIEWER)
                    unapproved = [u for u in unapproved if u != API_REQUIRED_REVIEWER]
                    log(f"Added {API_REQUIRED_REVIEWER} as required reviewer (tt_metal/api/)")

            selected_owners.extend(self.pick_two(unapproved))

        # Individual patterns
        for pattern_group in split_nonempty(self.individuals, "§"):
            head, _, files = pattern_group.rpartition(":")
            pattern = head.split(":", 1)[0] if ":" in head else head
            owners = head.split(":", 1)[1] if ":" in head else ""
            key = sorted_files_key(files)

            if files_has_approval.get(key):
                log(f"Pattern {pattern} already approved (combined), skipping")
                continue

            usernames = [pair.split("|", 1)[0] for pair in owners.split(",") if pair != ""]
            unapproved = self.unapproved_filtered(usernames)
            selected_owners.extend(self.pick_two(unapproved))

        # Sort + dedupe individual owners (parity with sort | uniq).
        final_owners = sorted(set(o for o in selected_owners if o))
        final_groups = [g for g in selected_slack_groups if g]
        no_owners = not final_owners and not final_groups

        return final_owners, final_groups, no_owners


def write_output(owners: list[str], groups: list[str], no_owners: bool) -> None:
    selected_owners = ",".join(owners)
    selected_groups = ",".join(groups)
    log(f"DEBUG: Final SELECTED_OWNERS='{selected_owners}'")
    log(f"DEBUG: Final SELECTED_SLACK_GROUPS='{selected_groups}'")
    log(f"DEBUG: NO_OWNERS_AVAILABLE='{'true' if no_owners else ''}'")

    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        log("GITHUB_OUTPUT not set; printing outputs to stdout")
        out = sys.stdout
        close = False
    else:
        out = open(out_path, "a", encoding="utf-8")
        close = True
    try:
        out.write(f"selected-owners={selected_owners}\n")
        out.write(f"selected-slack-groups={selected_groups}\n")
        out.write(f"no-owners-available={'true' if no_owners else ''}\n")
    finally:
        if close:
            out.close()


def main() -> int:
    owners, groups, no_owners = Selector().select()
    write_output(owners, groups, no_owners)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
