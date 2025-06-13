import git
from codeowners import CodeOwners

repo = git.Repo(".")

print(repo.active_branch)

final_reviewers = set()
candidate_reviewers = []

main_branch = "origin/main"
diff_files = repo.git.diff("--name-only", f"{main_branch}...HEAD").splitlines()
codeowners_file = repo.git.show(f"{repo.active_branch}:.github/CODEOWNERS")
codeowners = CodeOwners(codeowners_file)
for file in diff_files:
    print("\t", file)
    owners = codeowners.of(file)
    for owner in owners:
        print("\t\t", owner[1])

    parsed_owners = set()
    for owner in owners:
        parsed_owners.add(owner[1])
    if parsed_owners:
        candidate_reviewers.append(parsed_owners)

while candidate_reviewers:
    candidate_reviewer_counts = {}
    for reviewers in candidate_reviewers:
        for reviewer in reviewers:
            if reviewer not in candidate_reviewer_counts:
                candidate_reviewer_counts[reviewer] = 0
            candidate_reviewer_counts[reviewer] += 1

    most_common_candidate_reviewer = max(candidate_reviewer_counts, key=candidate_reviewer_counts.get)
    final_reviewers.add(most_common_candidate_reviewer)

    for reviewers in candidate_reviewers:
        if most_common_candidate_reviewer in reviewers:
            candidate_reviewers.remove(reviewers)

print()

for final_reviewer in final_reviewers:
    print(final_reviewer)
