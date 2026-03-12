#!/usr/bin/env bash
# Smoke tests for the GitHub Actions security linter.

set -uo pipefail

passed=0
failed=0
case_index=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECURITY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKER="$SECURITY_DIR/check-actions-security.sh"
PARALLEL_CHECKER="$SECURITY_DIR/check-actions-security-parallel.sh"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

test_pass() {
    echo "  PASS: $1"
    ((passed++)) || true
}

test_fail() {
    echo "  FAIL: $1"
    ((failed++)) || true
}

next_case_file() {
    ((case_index++)) || true
    echo "$TMP_DIR/case-${case_index}.yaml"
}

assert_detects() {
    local description="$1"
    local checks="$2"
    local expected="$3"
    local file
    local output

    file="$(next_case_file)"
    cat > "$file"

    output=$(bash "$CHECKER" --strict -c "$checks" "$file" 2>&1 || true)
    if [[ "$output" == *"$expected"* ]]; then
        test_pass "$description"
    else
        test_fail "$description (expected output to contain '$expected')"
        echo "$output"
    fi
}

assert_clean() {
    local description="$1"
    local checks="$2"
    local file
    local output
    local status

    file="$(next_case_file)"
    cat > "$file"

    output=$(bash "$CHECKER" --strict -c "$checks" "$file" 2>&1)
    status=$?
    if [[ $status -eq 0 && "$output" == *"No security issues found!"* ]]; then
        test_pass "$description"
    else
        test_fail "$description (expected clean run)"
        echo "$output"
    fi
}

assert_parallel_detects() {
    local description="$1"
    local expected="$2"
    local bad_file="$TMP_DIR/parallel-bad.yaml"
    local good_file="$TMP_DIR/parallel-good.yaml"
    local output

    cat > "$bad_file"
    cat > "$good_file" <<'EOF'
name: good
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
EOF

    output=$(bash "$PARALLEL_CHECKER" --strict "$bad_file" "$good_file" 2>&1 || true)
    if [[ "$output" == *"$expected"* ]]; then
        test_pass "$description"
    else
        test_fail "$description (expected output to contain '$expected')"
        echo "$output"
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

echo ""
echo "Results: $passed passed, $failed failed"

if [[ $failed -ne 0 ]]; then
    exit 1
fi
