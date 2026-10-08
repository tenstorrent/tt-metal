#!/bin/bash
# usage: main_job_status.sh "<workflow name>" "<job name substring>" [branch=main] [n_runs=5]
# Conclusion of the matching job(s) in the last n completed runs of <workflow> on <branch>, newest first.
# "unrelated, main also broken" = the same job fails here too. A run that prints "no matching job"
# ran a different matrix — it is NOT evidence that main is green. Repo: $GH_REPO, default tenstorrent/tt-metal.
wf=$1; job=$2; br=${3:-main}; n=${4:-5}; repo=${GH_REPO:-tenstorrent/tt-metal}
wid=$(gh api "repos/$repo/actions/workflows?per_page=100" --paginate -q ".workflows[] | select(.name == \"$wf\") | .id" | head -1)
[ -z "$wid" ] && { echo "no workflow named '$wf'" >&2; exit 2; }
gh api "repos/$repo/actions/workflows/$wid/runs?branch=$br&status=completed&per_page=$n" \
    -q '.workflow_runs[] | "\(.id)\t\(.created_at)\t\(.head_sha[0:11])"' |
while IFS=$'\t' read -r rid t sha; do
    out=$(gh api "repos/$repo/actions/runs/$rid/jobs?per_page=100" --paginate \
        -q ".jobs[] | select(.name | contains(\"$job\")) | \"\(.conclusion)\t\(.name)\"")
    echo "${out:-no matching job}" | sed "s#^#$t $sha run=$rid  #"
done
