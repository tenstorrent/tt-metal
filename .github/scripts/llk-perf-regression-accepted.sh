#!/usr/bin/env bash
# Usage: llk-perf-regression-accepted.sh <pr-number>
# Writes accepted=true to $GITHUB_OUTPUT when the PR has the llk-perf-accept-regression label, the
# last person who added it is in tenstorrent/regression-approve, and that person is not the PR author.
# Anything else, a failed API call included, writes accepted=false.
# GH_TOKEN reads the PR; TEAM_TOKEN needs org read to see the team.
set -uo pipefail

LABEL=llk-perf-accept-regression
TEAM=regression-approve
PR=${1:-}
REPO=${GITHUB_REPOSITORY:-tenstorrent/tt-metal}
OUT=${GITHUB_OUTPUT:-/dev/stdout}

result() {
  [ -n "${2:-}" ] && echo "::notice::#${PR}: $2"
  echo "accepted=$1" >> "$OUT"
  exit 0
}

[ -n "$PR" ] || result false
LABELS=$(gh api "repos/${REPO}/issues/${PR}/labels" --paginate --jq '.[].name') \
  || result false "could not read the labels, so no LLK perf regression is accepted."
grep -qxF "$LABEL" <<< "$LABELS" || result false

ACTOR=$(gh api "repos/${REPO}/issues/${PR}/events" --paginate \
  --jq ".[] | select(.event == \"labeled\" and .label.name == \"${LABEL}\") | .actor.login" | tail -n 1)
AUTHOR=$(gh api "repos/${REPO}/pulls/${PR}" --jq .user.login)
[ -n "$ACTOR" ] && [ -n "$AUTHOR" ] \
  || result false "could not find who added the ${LABEL} label, so the regression is not accepted."
[ "$ACTOR" != "$AUTHOR" ] \
  || result false "@${ACTOR} added the ${LABEL} label to their own PR. The author cannot accept their own regression."

CODE=$(curl -s -g -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer ${TEAM_TOKEN}" -H "Accept: application/vnd.github+json" \
  "https://api.github.com/orgs/tenstorrent/teams/${TEAM}/members/${ACTOR}")
[ "$CODE" = 204 ] \
  || result false "@${ACTOR} added the ${LABEL} label but is not in tenstorrent/${TEAM} (HTTP ${CODE}), so the regression is not accepted."

result true "@${ACTOR} (tenstorrent/${TEAM}) accepted the LLK perf regression with the ${LABEL} label."
