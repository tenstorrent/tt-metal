#!/usr/bin/env python3
"""Select which CodeOwners to ping for a pending PR review.

Extracted from the inline `Select owners for notification` step of
`.github/workflows/codeowners-group-analysis.yaml`. The behaviour is a faithful
port of the original bash: same inputs, same selection rules, same outputs.

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

Outputs (appended to $GITHUB_OUTPUT):
  selected-owners        comma-separated, sorted+deduped individual logins
  selected-slack-groups  comma-separated Slack group IDs
  no-owners-available    "true" iff both of the above are empty
"""

from __future__ import annotations

import os
import random
import sys

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

    # -- pick up to 2 owners at random (parity with RANDOM % n twice) ----------
    @staticmethod
    def pick_two(candidates: list[str]) -> list[str]:
        n = len(candidates)
        if n == 0:
            return []
        if n == 1:
            return [candidates[0]]
        r1 = random.randrange(n)
        r2 = random.randrange(n)
        while r2 == r1:
            r2 = random.randrange(n)
        return [candidates[r1], candidates[r2]]

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
