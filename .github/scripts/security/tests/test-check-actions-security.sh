#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
# Smoke tests for the GitHub Actions security linter.

set -uo pipefail

passed=0
failed=0
case_index=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECURITY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKER="${SECURITY_DIR}/check-actions-security.sh"
PARALLEL_CHECKER="${SECURITY_DIR}/check-actions-security-parallel.sh"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

test_pass() {
    printf '%s\n' "  PASS: $1"
    ((passed++)) || true
}

test_fail() {
    printf '%s\n' "  FAIL: $1"
    ((failed++)) || true
}

next_case_file() {
    ((case_index++)) || true
    printf '%s\n' "${TMP_DIR}/case-${case_index}.yaml"
}

assert_detects() {
    local description="$1"
    local checks="$2"
    local expected="$3"
    local file
    local output

    file="$(next_case_file)"
    cat > "${file}"

    output=$(bash "${CHECKER}" --strict -c "${checks}" "${file}" 2>&1 || true)
    if [[ "${output}" == *"${expected}"* ]]; then
        test_pass "${description}"
    else
        test_fail "${description} (expected output to contain '${expected}')"
        printf '%s\n' "${output}"
    fi
}

assert_clean() {
    local description="$1"
    local checks="$2"
    local file
    local output
    local status

    file="$(next_case_file)"
    cat > "${file}"

    output=$(bash "${CHECKER}" --strict -c "${checks}" "${file}" 2>&1)
    status=$?
    if [[ ${status} -eq 0 && "${output}" == *"No security issues found!"* ]]; then
        test_pass "${description}"
    else
        test_fail "${description} (expected clean run)"
        printf '%s\n' "${output}"
    fi
}

assert_parallel_detects() {
    local description="$1"
    local expected="$2"
    local bad_file="${TMP_DIR}/parallel-bad.yaml"
    local good_file="${TMP_DIR}/parallel-good.yaml"
    local output

    cat > "${bad_file}"
    cat > "${good_file}" <<'EOF'
name: good
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
EOF

    output=$(bash "${PARALLEL_CHECKER}" --strict "${bad_file}" "${good_file}" 2>&1 || true)
    if [[ "${output}" == *"${expected}"* ]]; then
        test_pass "${description}"
    else
        test_fail "${description} (expected output to contain '${expected}')"
        printf '%s\n' "${output}"
    fi
}

echo "============================================"
echo "GitHub Actions Security Linter Smoke Tests"
echo "============================================"
echo ""

# --- Check 1: Direct event data interpolation in run blocks ---

assert_detects "check 1 flags event data interpolation in run block" "1" "dangerous event data interpolation" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Echo title
        run: echo "${{ github.event.pull_request.title }}"
EOF

assert_clean "check 1 accepts event data via env var" "1" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Echo title
        env:
          PR_TITLE: ${{ github.event.pull_request.title }}
        run: echo "$PR_TITLE"
EOF

# --- Check 2: eval usage ---

assert_detects "check 2 flags eval statement" "2" "eval" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run eval
        run: eval "$CMD"
EOF

assert_clean "check 2 accepts case statement alternative" "2" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run command
        run: |
          case "$CMD" in
            build) make build ;;
            test) make test ;;
          esac
EOF

# --- Check 3: bash -c with direct interpolation ---

assert_detects "check 3 flags bash -c with interpolation" "3" "bash -c" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run bash
        run: bash -c "echo ${{ inputs.command }}"
EOF

assert_clean "check 3 accepts bash -c with env var" "3" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run bash
        env:
          CMD: ${{ inputs.command }}
        run: bash -c "echo \"$CMD\""
EOF

# --- Check 4: pull_request_target with checkout ---

assert_detects "check 4 flags pull_request_target with checkout" "4" "pull_request_target with checkout" <<'EOF'
name: test
on: pull_request_target
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
      - name: Build
        run: make build
EOF

assert_clean "check 4 accepts pull_request with checkout" "4" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
      - name: Build
        run: make build
EOF

# --- Check 5: Unpinned external action references ---

assert_detects "check 5 flags external actions not pinned to full SHA" "5" "not pinned to a full commit SHA" <<'EOF'
name: test
on: push
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@1234567
EOF

assert_clean "check 5 accepts full SHA action pins" "5" <<'EOF'
name: test
on: push
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
EOF

# --- Check 6: secrets: inherit usage (aggregate, threshold 50) ---
# Detect test omitted: check 6 requires 51+ files with 'secrets: inherit' to trigger.

assert_clean "check 6 accepts explicit secret passing" "6" <<'EOF'
name: test
on: push
jobs:
  test:
    uses: ./.github/workflows/build.yaml
    secrets:
      DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
EOF

# --- Check 7: Token exposure in logs ---

assert_detects "check 7 flags token exposure in echo" "7" "token exposure" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Debug
        run: echo "Token is $GITHUB_TOKEN"
EOF

assert_clean "check 7 accepts token in non-echo context" "7" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Use token
        run: curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos
EOF

assert_detects "check 7 flags printf token exposure" "7" "token exposure" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Debug
        run: printf '%s\n' "$GITHUB_TOKEN"
EOF

assert_detects "check 7 flags Python print token exposure" "7" "token exposure" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Debug
        run: python -c "import os; print(os.environ.get('GITHUB_TOKEN', 'none'))"
EOF

# --- Check 8: Overly broad permissions ---

assert_detects "check 8 flags write-all permissions" "8" "write-all" <<'EOF'
name: test
on: push
permissions: write-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Build
        run: make build
EOF

assert_clean "check 8 accepts scoped permissions" "8" <<'EOF'
name: test
on: push
permissions:
  contents: read
  pull-requests: write
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Build
        run: make build
EOF

# --- Check 9: Ref-based injection patterns ---

assert_detects "check 9 flags head.ref interpolation in run block" "9" "head.ref/base.ref interpolation" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Echo ref
        run: echo "${{ github.event.pull_request.head.ref }}"
EOF

assert_detects "check 9 flags github.head_ref (underscore) in run block" "9" "head.ref/base.ref interpolation" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Echo ref
        run: echo "${{ github.head_ref }}"
EOF

assert_clean "check 9 accepts head.ref via env var" "9" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Echo ref
        env:
          HEAD_REF: ${{ github.event.pull_request.head.ref }}
        run: echo "$HEAD_REF"
EOF

# --- Check 10: curl | bash patterns ---

assert_detects "check 10 flags curl pipe to bash" "10" "curl | bash" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Install
        run: curl -sSL https://example.com/install.sh | bash
EOF

assert_clean "check 10 accepts download then verify pattern" "10" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Install
        run: |
          curl -sSL -o install.sh https://example.com/install.sh
          sha256sum -c <<< "expected_hash  install.sh"
          bash install.sh
EOF

# --- Check 11: inputs/outputs interpolation in run blocks ---

assert_detects "check 11 flags inputs interpolation in run block" "11" "inputs" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      name:
        required: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Greet
        run: echo "Hello ${{ inputs.name }}"
EOF

assert_clean "check 11 accepts inputs via env var" "11" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      name:
        required: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Greet
        env:
          NAME: ${{ inputs.name }}
        run: echo "Hello $NAME"
EOF

# --- Check 12: Deprecated ::set-output and ::save-state ---

assert_detects "check 12 flags deprecated set-output command" "12" "deprecated" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set output
        run: echo "::set-output name=result::$VALUE"
EOF

assert_clean "check 12 accepts GITHUB_OUTPUT file" "12" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set output
        run: echo "result=$VALUE" >> "$GITHUB_OUTPUT"
EOF

# --- Check 13: ACTIONS_ALLOW_UNSECURE_COMMANDS ---

assert_detects "check 13 flags ACTIONS_ALLOW_UNSECURE_COMMANDS" "13" "ACTIONS_ALLOW_UNSECURE_COMMANDS" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      ACTIONS_ALLOW_UNSECURE_COMMANDS: true
    steps:
      - name: Set env
        run: echo "::set-env name=PATH::$NEW_PATH"
EOF

assert_clean "check 13 accepts GITHUB_ENV file usage" "13" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set env
        run: echo "PATH=$NEW_PATH" >> "$GITHUB_ENV"
EOF

# --- Check 14: toJSON() in run blocks ---

assert_detects "check 14 flags toJSON in run block" "14" "toJSON()" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        config: [a, b]
    steps:
      - name: Show config
        run: |
          data='${{ toJSON(matrix.config) }}'
          echo "$data" | jq .
EOF

assert_clean "check 14 accepts toJSON via env var" "14" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        config: [a, b]
    steps:
      - name: Show config
        env:
          CONFIG_JSON: ${{ toJSON(matrix.config) }}
        run: echo "$CONFIG_JSON" | jq .
EOF

# --- Check 15: Cross-repo reusable workflow mutable refs ---

assert_detects "check 15 flags reusable workflows on mutable refs" "15" "mutable ref instead of commit SHA" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  reuse:
    uses: octo-org/example/.github/workflows/build.yaml@release-2026.03
EOF

# --- Check 16: Additional attacker-controlled event fields ---

assert_detects "check 16 flags attacker-controlled event fields in run block" "16" "attacker-controlled github.event field" <<'EOF'
name: test
on: discussion
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Echo title
        run: echo "${{ github.event.discussion.title }}"
EOF

assert_clean "check 16 accepts attacker-controlled fields via env var" "16" <<'EOF'
name: test
on: discussion
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Echo title
        env:
          TITLE: ${{ github.event.discussion.title }}
        run: echo "$TITLE"
EOF

# --- Check 17: actions/github-script with expression interpolation ---

assert_detects "check 17 flags github-script with event interpolation" "17" "JavaScript injection" <<'EOF'
name: test
on: issues
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@v7
        with:
          script: |
            const title = "${{ github.event.issue.title }}";
            console.log(title);
EOF

assert_clean "check 17 accepts github-script with process.env" "17" <<'EOF'
name: test
on: issues
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@v7
        env:
          ISSUE_TITLE: ${{ github.event.issue.title }}
        with:
          script: |
            const title = process.env.ISSUE_TITLE;
            console.log(title);
EOF

# --- Check 18: secrets.* interpolation in run blocks ---

assert_detects "check 18 flags secrets interpolation in run block" "18" "secrets" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set remote
        run: git remote set-url origin https://${{ secrets.MY_TOKEN }}@github.com/org/repo.git
EOF

assert_clean "check 18 accepts secrets via env var" "18" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set remote
        env:
          MY_TOKEN: ${{ secrets.MY_TOKEN }}
        run: git remote set-url origin "https://${MY_TOKEN}@github.com/org/repo.git"
EOF

# --- Check 19: matrix.* interpolation in run blocks ---

assert_detects "check 19 flags matrix interpolation in run block" "19" "matrix" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        os: [ubuntu, macos]
    steps:
      - name: Show OS
        run: echo "Testing on ${{ matrix.os }}"
EOF

assert_clean "check 19 accepts matrix via env var" "19" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        os: [ubuntu, macos]
    steps:
      - name: Show OS
        env:
          MATRIX_OS: ${{ matrix.os }}
        run: echo "Testing on $MATRIX_OS"
EOF

# --- Check 20: github.repository/github.event_name in run blocks ---

assert_detects "check 20 flags github.repository in run block" "20" "github.repository" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Show repo
        run: gh pr view 123 --repo "${{ github.repository }}"
EOF

assert_clean "check 20 accepts github.repository via env var" "20" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Show repo
        env:
          REPO: ${{ github.repository }}
        run: gh pr view 123 --repo "$REPO"
EOF

# --- Check 21: pull_request_target privilege boundary issues ---

assert_detects "check 21 flags pull_request_target head checkout" "21" "checks out the PR head ref/repository" <<'EOF'
name: test
on:
  pull_request_target:
jobs:
  privileged:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - uses: ./.github/actions/local-action
EOF

assert_detects "check 22 flags privileged workflow_run artifact use" "22" "downloads or references upstream run artifacts" <<'EOF'
name: test
on:
  workflow_run:
    workflows: ["Build"]
    types: [completed]
permissions:
  contents: write
jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - run: gh run download "${{ github.event.workflow_run.id }}"
EOF

assert_detects "check 23 flags self-hosted runners on pull_request" "23" "self-hosted runner" <<'EOF'
name: test
on:
  pull_request:
jobs:
  test:
    runs-on: [self-hosted, linux]
    steps:
      - run: echo safe
EOF

assert_detects "check 24 flags write permissions on untrusted triggers" "24" "write-capable token scopes" <<'EOF'
name: test
on:
  issue_comment:
permissions:
  contents: write
jobs:
  comment:
    runs-on: ubuntu-latest
    steps:
      - run: echo hello
EOF

assert_detects "check 25 flags mutable container image references" "25" "without immutable @sha256 digests" <<'EOF'
name: test
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/example/build-env:latest
    steps:
      - uses: docker://alpine:3.20
EOF

assert_clean "check 25 accepts digest-pinned container images" "25" <<'EOF'
name: test
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/example/build-env@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    steps:
      - uses: docker://alpine@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
EOF

assert_parallel_detects "parallel wrapper preserves new check output" "self-hosted runner" <<'EOF'
name: bad
on:
  pull_request:
jobs:
  test:
    runs-on: [self-hosted, linux]
    steps:
      - run: echo hi
EOF

assert_detects "check 26 flags fromJSON with attacker-controlled input" "26" "fromJSON() used with potentially attacker-controlled input" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  build:
    runs-on: ${{ fromJSON(inputs.runner-label) }}
    steps:
      - run: echo hi
EOF

assert_clean "check 26 accepts fromJSON with safe context" "26" <<'EOF'
name: test
on: push
jobs:
  build:
    runs-on: ${{ fromJSON(vars.RUNNER_LABELS) }}
    steps:
      - run: echo hi
EOF

assert_detects "check 26 flags fromJSON with inputs on single-line on: workflow_dispatch" "26" "fromJSON() used with potentially attacker-controlled input" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  build:
    runs-on: ${{ fromJSON(inputs.runner-label) }}
    steps:
      - run: echo hi
EOF

assert_detects "check 27 flags dynamic expressions in uses:" "27" "Uses statement contains expression interpolation" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  build:
    uses: ./.github/workflows/${{ inputs.workflow }}.yaml
EOF

assert_clean "check 27 accepts static uses: references" "27" <<'EOF'
name: test
on: push
jobs:
  build:
    uses: ./.github/workflows/build.yaml
EOF

assert_detects "check 28 flags cache keys with attacker-controlled input" "28" "Cache key contains potentially attacker-controlled input" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/cache@v4
        with:
          key: ${{ inputs.cache-key }}-deps
EOF

assert_clean "check 28 accepts cache keys with safe context" "28" <<'EOF'
name: test
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/cache@v4
        with:
          key: ${{ github.sha }}-deps
EOF

assert_detects "check 29 flags artifact paths with expressions" "29" "Artifact path contains expression interpolation" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/upload-artifact@v4
        with:
          path: ${{ inputs.artifact-path }}
EOF

assert_clean "check 29 accepts hardcoded artifact paths" "29" <<'EOF'
name: test
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/upload-artifact@v4
        with:
          path: ./build/
EOF

assert_detects "check 29 flags artifact paths with single-line on: workflow_dispatch" "29" "Artifact path contains expression interpolation" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/upload-artifact@v4
        with:
          path: ${{ inputs.artifact-path }}
EOF

assert_detects "check 30 flags curl with expression URLs" "30" "curl/wget command contains expression interpolation" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: curl -sSL "${{ inputs.download-url }}" -o file.tar.gz
EOF

assert_clean "check 30 accepts curl with hardcoded URLs" "30" <<'EOF'
name: test
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: curl -sSL "https://example.com/file.tar.gz" -o file.tar.gz
EOF

assert_detects "check 31 flags container volume expressions" "31" "Container volume mount contains expression interpolation" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: ubuntu:latest
      volumes:
        - ${{ inputs.mount-path }}:/workspace
    steps:
      - run: echo hi
EOF

assert_clean "check 31 accepts hardcoded container volumes" "31" <<'EOF'
name: test
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: ubuntu:latest
      volumes:
        - /workspace:/workspace
    steps:
      - run: echo hi
EOF

# --- Check 32: Expression interpolation in GITHUB_ENV/PATH/OUTPUT writes ---

assert_detects "check 32 flags expression in GITHUB_ENV write" "32" "GITHUB_ENV/PATH/OUTPUT" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set env
        run: echo "BRANCH=${{ github.head_ref }}" >> "$GITHUB_ENV"
EOF

assert_detects "check 32 flags expression in GITHUB_PATH write" "32" "GITHUB_ENV/PATH/OUTPUT" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      extra-path:
        required: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set path
        run: echo "${{ inputs.extra-path }}" >> "$GITHUB_PATH"
EOF

assert_clean "check 32 accepts shell variable in GITHUB_ENV write" "32" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set env
        env:
          BRANCH: ${{ github.head_ref }}
        run: echo "BRANCH=$BRANCH" >> "$GITHUB_ENV"
EOF

# --- Check 33: Missing explicit permissions key (aggregate) ---

assert_detects "check 33 flags workflow missing permissions key" "33" "without an explicit top-level" <<'EOF'
name: test
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Build
        run: make build
EOF

assert_clean "check 33 accepts workflow with permissions key" "33" <<'EOF'
name: test
on: push
permissions:
  contents: read
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Build
        run: make build
EOF

# --- Check 34: Concurrency group with attacker-controlled data ---

assert_detects "check 34 flags concurrency with head_ref" "34" "Concurrency group contains attacker-controlled expression" <<'EOF'
name: test
on: pull_request
concurrency:
  group: deploy-${{ github.head_ref }}
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: echo hello
EOF

assert_detects "check 34 flags concurrency with inputs" "34" "Concurrency group contains attacker-controlled expression" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      env-name:
        required: true
concurrency:
  group: deploy-${{ inputs.env-name }}
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: echo hello
EOF

assert_clean "check 34 accepts concurrency with safe context" "34" <<'EOF'
name: test
on: push
concurrency:
  group: ci-${{ github.sha }}
  cancel-in-progress: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Build
        run: echo hello
EOF

# --- Check 35: issue_comment trigger without authorization gate ---

assert_detects "check 35 flags issue_comment without auth gate" "35" "no author_association" <<'EOF'
name: test
on: issue_comment
jobs:
  deploy:
    if: startsWith(github.event.comment.body, '/deploy')
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: echo deploying
EOF

assert_clean "check 35 accepts issue_comment with author_association" "35" <<'EOF'
name: test
on: issue_comment
jobs:
  deploy:
    if: |
      startsWith(github.event.comment.body, '/deploy') &&
      github.event.comment.author_association == 'MEMBER'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        run: echo deploying
EOF

# --- Check 36: working-directory with attacker-controlled expressions ---

assert_detects "check 36 flags working-directory with inputs expression" "36" "working-directory contains attacker-controlled expression" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      build-dir:
        required: true
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
      - name: Build
        run: make build
        working-directory: ${{ inputs.build-dir }}
EOF

assert_clean "check 36 accepts working-directory with github.workspace" "36" <<'EOF'
name: test
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
      - name: Build
        run: make build
        working-directory: ${{ github.workspace }}/src
EOF

assert_detects "check 36 flags working-directory with inputs on single-line on: workflow_dispatch" "36" "working-directory contains attacker-controlled expression" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
      - name: Build
        run: make build
        working-directory: ${{ inputs.build-dir }}
EOF

# --- Check 37: github-script with non-event attacker-controlled interpolation ---

assert_detects "check 37 flags github-script with inputs interpolation" "37" "JavaScript injection risk" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      user_script:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        with:
          script: |
            const body = "${{ inputs.user_script }}";
            console.log(body);
EOF

assert_clean "check 37 accepts github-script with process.env for inputs" "37" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      user_script:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        env:
          USER_SCRIPT: ${{ inputs.user_script }}
        with:
          script: |
            const body = process.env.USER_SCRIPT;
            console.log(body);
EOF

# --- Check 38: github-script dynamic code execution primitives ---

assert_detects "check 38 flags github-script dynamic code execution" "38" "dynamic code execution" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        with:
          script: |
            const dynamic = core.getInput('payload');
            return new Function(dynamic)();
EOF

assert_detects "check 38 flags github-script eval with process.env" "38" "dynamic code execution" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        env:
          USER_SCRIPT: ${{ inputs.payload }}
        with:
          script: |
            eval(process.env.USER_SCRIPT)
EOF

assert_clean "check 38 accepts github-script treating input as data" "38" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        with:
          script: |
            const payload = core.getInput('payload');
            console.log(payload);
EOF

# --- Check 39: Inline interpreter eval/exec patterns in run blocks ---

assert_detects "check 39 flags node -e eval with env data" "39" "Inline interpreter command uses eval/exec-style dynamic code execution" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          USER_SCRIPT: ${{ inputs.payload }}
        run: node -e "eval(process.env.USER_SCRIPT)"
EOF

assert_clean "check 39 accepts inline interpreter without eval" "39" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          USER_SCRIPT: ${{ inputs.payload }}
        run: node -e "console.log(process.env.USER_SCRIPT)"
EOF

# --- Check 40: github-script command execution APIs ---

assert_detects "check 40 flags github-script child_process execution with input" "40" "command injection risk" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        with:
          script: |
            const { execSync } = require('child_process');
            const cmd = core.getInput('payload');
            execSync(cmd, { shell: true });
EOF

assert_clean "check 40 accepts github-script without command execution" "40" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        with:
          script: |
            const payload = core.getInput('payload');
            console.log(payload);
EOF

# --- Check 41: Inline interpreter command execution APIs ---

assert_detects "check 41 flags python subprocess shell execution" "41" "shell/process execution APIs" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          USER_CMD: ${{ inputs.payload }}
        run: python -c "import os, subprocess; subprocess.run(os.environ['USER_CMD'], shell=True, check=True)"
EOF

assert_clean "check 41 accepts inline interpreter without shell execution" "41" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          USER_CMD: ${{ inputs.payload }}
        run: python -c "import os; print(os.environ['USER_CMD'])"
EOF

# --- Check 42: Shell command execution from variables ---

assert_detects "check 42 flags bash -c executing variable contents" "42" "executed from a variable" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          USER_CMD: ${{ inputs.payload }}
        run: bash -c "$USER_CMD"
EOF

assert_clean "check 42 accepts shell variable treated as data" "42" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      payload:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          USER_CMD: ${{ inputs.payload }}
        run: printf '%s\n' "$USER_CMD"
EOF

# --- Check 43: Remote script execution variants beyond pipes ---

assert_detects "check 43 flags bash process substitution from curl" "43" "Remote content is executed directly" <<'EOF'
name: test
on: push
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: bash <(curl -sSL https://example.com/install.sh)
EOF

assert_detects "check 43 flags PowerShell iex irm pattern" "43" "Remote content is executed directly" <<'EOF'
name: test
on: push
permissions:
  contents: read
jobs:
  test:
    runs-on: windows-latest
    steps:
      - run: pwsh -Command "iex (irm https://example.com/install.ps1)"
EOF

assert_clean "check 43 accepts download then verify pattern" "43" <<'EOF'
name: test
on: push
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: |
          curl -sSL -o install.sh https://example.com/install.sh
          sha256sum -c <<< "expected_hash  install.sh"
          bash install.sh
EOF

# --- Check 44: Attacker-controlled env vars written to GITHUB_ENV/PATH/OUTPUT ---

assert_detects "check 44 flags attacker-controlled env var written to GITHUB_PATH" "44" "Attacker-controlled env var is written to GITHUB_ENV/PATH/OUTPUT" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      extra-path:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          EXTRA_PATH: ${{ inputs.extra-path }}
        run: echo "$EXTRA_PATH" >> "$GITHUB_PATH"
EOF

assert_clean "check 44 accepts validated path before GITHUB_PATH write" "44" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      extra-path:
        required: true
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          EXTRA_PATH: ${{ inputs.extra-path }}
        run: |
          [[ "$EXTRA_PATH" == /opt/tools/* ]] || exit 1
          printf '%s\n' "/opt/tools/bin" >> "$GITHUB_PATH"
EOF

assert_detects "check 44 flags env var written to GITHUB_PATH with single-line on: workflow_dispatch" "44" "Attacker-controlled env var is written to GITHUB_ENV/PATH/OUTPUT" <<'EOF'
name: test
on: workflow_dispatch
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          EXTRA_PATH: ${{ inputs.extra-path }}
        run: echo "$EXTRA_PATH" >> "$GITHUB_PATH"
EOF

# --- Check 45: Expression injection in 'if:' conditions ---

assert_detects "check 45 flags if condition with event data" "45" "if:' condition contains" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Conditional step
        if: contains('${{ github.event.pull_request.title }}', 'build')
        run: echo "This step runs conditionally"
EOF

assert_clean "check 45 accepts safe if condition" "45" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Safe conditional
        env:
          PR_TITLE: ${{ github.event.pull_request.title }}
        if: contains(env.PR_TITLE, 'build')
        run: echo "This step runs conditionally"
EOF

assert_detects "check 45 flags if condition with inputs on single-line on: workflow_dispatch" "45" "if:' condition contains" <<'EOF'
name: test
on: workflow_dispatch
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Conditional step
        if: contains('${{ inputs.deploy-target }}', 'prod')
        run: echo "deploying"
EOF

# --- Check 46: Bracket notation bypass in expressions ---

assert_detects "check 46 flags bracket notation in run block" "46" "bracket notation" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Bracket notation
        run: echo "${{ inputs['user-command'] }}"
EOF

assert_detects "check 46 flags bracket notation for github.event" "46" "bracket notation" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Bracket notation event
        run: echo "${{ github.event['pull_request']['title'] }}"
EOF

assert_clean "check 46 accepts bracket notation via env var" "46" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Safe bracket notation
        env:
          USER_CMD: ${{ inputs['user-command'] }}
        run: echo "$USER_CMD"
EOF

# --- Check 47: Additional attacker-controlled event contexts ---

assert_detects "check 47 flags release body in run block" "47" "additional attacker-controlled github.event" <<'EOF'
name: test
on: release
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Release body
        run: echo "${{ github.event.release.body }}"
EOF

assert_detects "check 47 flags workflow_dispatch inputs in run block" "47" "additional attacker-controlled github.event" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      command:
        required: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Input command
        run: ${{ github.event.inputs.command }}
EOF

assert_clean "check 47 accepts safe event fields via env" "47" <<'EOF'
name: test
on: release
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Safe release body
        env:
          BODY: ${{ github.event.release.body }}
        run: echo "$BODY"
EOF

# --- Check 48: Export statements with direct interpolation ---

assert_detects "check 48 flags export with input interpolation" "48" "Export statement contains" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      build-dir:
        required: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Export build dir
        run: |
          export BUILD_DIR="${{ inputs.build-dir }}"
          cd "$BUILD_DIR" && make
EOF

assert_detects "check 48 flags export with event data" "48" "Export statement contains" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Export PR title
        run: |
          export PR_TITLE="${{ github.event.pull_request.title }}"
          echo "$PR_TITLE"
EOF

assert_clean "check 48 accepts export of env var" "48" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      build-dir:
        required: true
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      BUILD_DIR: ${{ inputs.build-dir }}
    steps:
      - name: Export from env
        run: |
          export DIR="$BUILD_DIR"
          echo "$DIR"
EOF

# --- Check 49: Dynamic 'shell:' property expressions ---

assert_detects "check 49 flags shell property with inputs" "49" "'shell:' property contains" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      custom_shell:
        required: true
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Dynamic shell
        run: echo "hello"
        shell: ${{ inputs.custom_shell }}
EOF

assert_detects "check 49 flags shell property with event data" "49" "'shell:' property contains" <<'EOF'
name: test
on: issue_comment
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Shell from comment
        run: echo "hello"
        shell: ${{ github.event.comment.body }}
EOF

assert_clean "check 49 accepts hardcoded shell" "49" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Hardcoded shell
        run: echo "hello"
        shell: bash
EOF

# --- Check 50: env.* context interpolation in run blocks ---

assert_detects "check 50 flags env interpolation in run block" "50" "Contains \${{ env.* }} directly in run: block" <<'EOF'
name: test
on: push
env:
  MY_VAR: malicious
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Dynamic env
        run: echo "${{ env.MY_VAR }}"
EOF

assert_clean "check 50 accepts env var mapped to shell var" "50" <<'EOF'
name: test
on: push
env:
  MY_VAR: malicious
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Safe env
        run: echo "$MY_VAR"
EOF

# --- Check 51: github.ref/github.ref_name interpolation in run blocks ---

assert_detects "check 51 flags github.ref_name interpolation in run block" "51" "Contains github.ref or github.ref_name interpolation directly in run: block" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Dynamic ref
        run: echo "${{ github.ref_name }}"
EOF

assert_detects "check 51 flags github.ref interpolation in run block" "51" "Contains github.ref or github.ref_name interpolation directly in run: block" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Dynamic ref
        run: echo "${{ github.ref }}"
EOF

assert_clean "check 51 accepts github.ref_name via env block" "51" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Safe ref
        env:
          REF_NAME: ${{ github.ref_name }}
        run: echo "$REF_NAME"
EOF

# --- Check 52: Dynamic environment variable names ---

assert_detects "check 52 flags dynamic env var names" "52" "dynamic environment variable names" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      env_name:
        required: true
      env_value:
        required: true
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      ${{ inputs.env_name }}: ${{ inputs.env_value }}
    steps:
      - run: echo test
EOF

assert_clean "check 52 accepts static env var names" "52" <<'EOF'
name: test
on:
  workflow_dispatch:
    inputs:
      env_value:
        required: true
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      MY_STATIC_VAR: ${{ inputs.env_value }}
    steps:
      - run: echo test
EOF

# =============================================================================
# Checks 53-66: New gap checks
# =============================================================================

printf '\n'
printf '%s\n' "--- Checks 53-66: New gap checks ---"

# Check 53: github.actor in run blocks
assert_detects "check 53 flags github.actor in run block" "53" "github.actor/github.triggering_actor" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "PR by ${{ github.actor }}"
EOF

assert_clean "check 53 accepts github.actor via env var" "53" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          ACTOR: ${{ github.actor }}
        run: echo "PR by ${ACTOR}"
EOF

# Check 54: github.event.issue.title in run blocks
assert_detects "check 54 flags issue.title in run block" "54" "github.event.issue.title" << 'EOF'
name: test
on: issues
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "${{ github.event.issue.title }}"
EOF

assert_clean "check 54 accepts issue.title via env var" "54" << 'EOF'
name: test
on: issues
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          TITLE: ${{ github.event.issue.title }}
        run: echo "${TITLE}"
EOF

# Check 55: head.repo.full_name / head.label in run blocks
assert_detects "check 55 flags head.repo.full_name in run block" "55" "head.repo/label" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "Fork ${{ github.event.pull_request.head.repo.full_name }}"
EOF

assert_detects "check 55 flags head.label in run block" "55" "head.repo/label" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "${{ github.event.pull_request.head.label }}"
EOF

assert_clean "check 55 accepts head.repo via env var" "55" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          FORK: ${{ github.event.pull_request.head.repo.full_name }}
        run: echo "${FORK}"
EOF

# Check 56: user.login / sender.login in run blocks
assert_detects "check 56 flags review.user.login in run block" "56" "user.login/sender.login" << 'EOF'
name: test
on: pull_request_review
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "${{ github.event.review.user.login }}"
EOF

assert_detects "check 56 flags sender.login in run block" "56" "user.login/sender.login" << 'EOF'
name: test
on: issue_comment
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "${{ github.event.sender.login }}"
EOF

# Check 57: workflow_run.head_commit.message
assert_detects "check 57 flags workflow_run.head_commit.message in run block" "57" "workflow_run.head_commit" << 'EOF'
name: test
on: workflow_run
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "${{ github.event.workflow_run.head_commit.message }}"
EOF

assert_clean "check 57 accepts workflow_run.head_sha (not commit message)" "57" << 'EOF'
name: test
on: workflow_run
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          SHA: ${{ github.event.workflow_run.head_sha }}
        run: echo "${SHA}"
EOF

# Check 58: deployment.payload in run blocks
assert_detects "check 58 flags deployment.payload in run block" "58" "deployment" << 'EOF'
name: test
on: deployment
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "${{ github.event.deployment.payload }}"
EOF

# Check 59: discussion.category.name in run blocks
assert_detects "check 59 flags discussion.category.name in run block" "59" "discussion sub-field" << 'EOF'
name: test
on: discussion
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "${{ github.event.discussion.category.name }}"
EOF

# Check 60: github.workflow in run blocks
assert_detects "check 60 flags github.workflow in run block" "60" "github.workflow" << 'EOF'
name: "test; echo pwned"
on: workflow_dispatch
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "Workflow ${{ github.workflow }}"
EOF

assert_clean "check 60 accepts github.workflow_ref (different field)" "60" << 'EOF'
name: test
on: workflow_dispatch
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - env:
          WF: ${{ github.workflow }}
        run: echo "${WF}"
EOF

# Check 61: run-name with attacker-controlled expressions
assert_detects "check 61 flags run-name with pull_request.title" "61" "run-name" << 'EOF'
name: test
on: pull_request
run-name: "PR ${{ github.event.pull_request.title }}"
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo test
EOF

assert_clean "check 61 accepts run-name with safe fields" "61" << 'EOF'
name: test
on: pull_request
run-name: "PR #${{ github.event.pull_request.number }}"
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo test
EOF

# Check 62: environment with attacker-controlled expressions
assert_detects "check 62 flags environment with head.ref expression" "62" "environment" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ github.event.pull_request.head.ref }}
    steps:
      - run: echo deploy
EOF

assert_clean "check 62 accepts hardcoded environment" "62" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - run: echo deploy
EOF

# Check 63: GITHUB_STEP_SUMMARY with attacker-controlled expressions
assert_detects "check 63 flags attacker data written to GITHUB_STEP_SUMMARY" "63" "GITHUB_STEP_SUMMARY" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "${{ github.event.pull_request.title }}" >> "${GITHUB_STEP_SUMMARY}"
EOF

assert_clean "check 63 accepts safe GITHUB_STEP_SUMMARY usage" "63" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "## Build Results" >> "${GITHUB_STEP_SUMMARY}"
EOF

# Check 64: continue-on-error: true
assert_detects "check 64 flags continue-on-error: true" "64" "continue-on-error" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: ./security-check.sh
        continue-on-error: true
      - run: ./deploy.sh
EOF

assert_clean "check 64 accepts continue-on-error: false" "64" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: ./security-check.sh
        continue-on-error: false
      - run: ./deploy.sh
EOF

# Check 65: merge_group.head_commit.message
assert_detects "check 65 flags merge_group.head_commit.message in run block" "65" "merge_group.head_commit" << 'EOF'
name: test
on: merge_group
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo "${{ github.event.merge_group.head_commit.message }}"
EOF

# Check 66: concurrency group with github.actor
assert_detects "check 66 flags github.actor in concurrency group" "66" "concurrency group" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    concurrency:
      group: deploy-${{ github.actor }}
    steps:
      - run: echo test
EOF

assert_clean "check 66 accepts github.sha in concurrency group" "66" << 'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    concurrency:
      group: deploy-${{ github.sha }}
    steps:
      - run: echo test
EOF

# =============================================================================
# Checks 67-69: Supply chain, OIDC, and timeout checks
# =============================================================================

printf '\n'
printf '%s\n' "--- Checks 67-69: Supply chain and resource checks ---"

# Check 67: actions/cache with sensitive credential directories
assert_detects "check 67 flags cache with ~/.aws" "67" "sensitive credential directory" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/cache@0c907fcceea5f9bb3ddf7a8f0ee3e5b7b6f5b9c2
        with:
          path: ~/.aws
          key: aws-creds-${{ runner.os }}
EOF

assert_detects "check 67 flags cache with ~/.kube" "67" "sensitive credential directory" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/cache@0c907fcceea5f9bb3ddf7a8f0ee3e5b7b6f5b9c2
        with:
          path: ~/.kube
          key: kube-${{ runner.os }}
EOF

assert_detects "check 67 flags cache with ~/.ssh" "67" "sensitive credential directory" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/cache@0c907fcceea5f9bb3ddf7a8f0ee3e5b7b6f5b9c2
        with:
          path: ~/.ssh
          key: ssh-keys-${{ runner.os }}
EOF

assert_clean "check 67 accepts cache with ~/.npm" "67" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/cache@0c907fcceea5f9bb3ddf7a8f0ee3e5b7b6f5b9c2
        with:
          path: ~/.npm
          key: npm-${{ runner.os }}-${{ hashFiles('**/package-lock.json') }}
EOF

assert_clean "check 67 ignores files without actions/cache" "67" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: echo "~/.aws credentials not used here"
EOF

# Check 68: id-token: write on untrusted triggers
assert_detects "check 68 flags id-token: write on pull_request" "68" "id-token: write permission on untrusted trigger" << 'EOF'
name: test
on: pull_request
permissions:
  id-token: write
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: echo "hello"
EOF

assert_detects "check 68 flags id-token: write on issue_comment" "68" "id-token: write permission on untrusted trigger" << 'EOF'
name: test
on: issue_comment
permissions:
  id-token: write
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: echo "hello"
EOF

assert_clean "check 68 accepts id-token: write on push" "68" << 'EOF'
name: test
on: push
permissions:
  id-token: write
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: echo "hello"
EOF

assert_clean "check 68 accepts id-token: none on pull_request" "68" << 'EOF'
name: test
on: pull_request
permissions:
  id-token: none
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: echo "hello"
EOF

# Check 69: missing timeout-minutes at job level
assert_detects "check 69 flags missing timeout-minutes" "69" "missing timeout-minutes" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: make build
EOF

assert_clean "check 69 accepts job with timeout-minutes" "69" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - run: make build
EOF

printf '\n'
printf '%s\n' "Results: ${passed} passed, ${failed} failed"

if [[ ${failed} -ne 0 ]]; then
    exit 1
fi
