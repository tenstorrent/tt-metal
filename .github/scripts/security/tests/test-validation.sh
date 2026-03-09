#!/usr/bin/env bash
# Comprehensive Unit Tests for Input Validation Library

passed=0
failed=0

test_pass() {
    echo "  ✓ PASS: $1"
    ((passed++)) || true
}

test_fail() {
    echo "  ✗ FAIL: $1"
    ((failed++)) || true
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../input-validation.sh"

set +e
set +u
set +o pipefail

assert_returns_0() {
    local description="$1"
    shift
    (
        set +e
        "$@" 2>/dev/null
        exit $?
    )
    local ret=$?
    if [[ $ret -eq 0 ]]; then
        test_pass "$description"
    else
        test_fail "$description"
    fi
}

assert_returns_1() {
    local description="$1"
    shift
    (
        set +e
        "$@" 2>/dev/null
        exit $?
    )
    local ret=$?
    if [[ $ret -ne 0 ]]; then
        test_pass "$description"
    else
        test_fail "$description"
    fi
}

assert_output_equals() {
    local expected="$1"
    local description="$2"
    shift 2
    local output
    output=$(
        set +e
        "$@" 2>/dev/null
    )
    if [[ "$output" == "$expected" ]]; then
        test_pass "$description"
    else
        test_fail "$description (expected '$expected', got '$output')"
    fi
}

assert_output_contains() {
    local pattern="$1"
    local description="$2"
    shift 2
    local output
    output=$(
        set +e
        "$@" 2>&1
    )
    if [[ "$output" == *"$pattern"* ]]; then
        test_pass "$description"
    else
        test_fail "$description (output didn't contain '$pattern')"
    fi
}

echo "============================================"
echo "Input Validation Library Unit Tests"
echo "============================================"
echo ""

# ============================================
# validate_hostname tests
# ============================================
echo "Testing validate_hostname..."

assert_returns_0 "hostname: simple hostname" validate_hostname "localhost"
assert_returns_0 "hostname: with subdomain" validate_hostname "sub.example.com"
assert_returns_0 "hostname: with hyphens" validate_hostname "my-server.example-domain.com"
assert_returns_0 "hostname: single label" validate_hostname "server"
assert_returns_0 "hostname: numeric label" validate_hostname "server-123.example"
assert_returns_0 "hostname: max label length (63 chars)" validate_hostname "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.example.com"
assert_returns_0 "hostname: starts with number" validate_hostname "1server.example.com"
assert_returns_0 "hostname: uppercase allowed (RFC 1123)" validate_hostname "EXAMPLE.COM"
assert_returns_0 "hostname: single char label" validate_hostname "a.com"

assert_returns_1 "hostname: empty string" validate_hostname ""
assert_returns_1 "hostname: starts with hyphen" validate_hostname "-server.example.com"
assert_returns_1 "hostname: ends with hyphen" validate_hostname "server-.example.com"
assert_returns_1 "hostname: ends with dot" validate_hostname "example.com."
assert_returns_1 "hostname: consecutive dots" validate_hostname "example..com"
assert_returns_1 "hostname: label starts with hyphen" validate_hostname "-sub.example.com"
assert_returns_1 "hostname: label ends with hyphen" validate_hostname "sub-.example.com"
assert_returns_1 "hostname: label too long (64 chars)" validate_hostname "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.example.com"
assert_returns_1 "hostname: contains underscore" validate_hostname "server_name.example.com"
assert_returns_1 "hostname: contains space" validate_hostname "server name.example.com"
assert_returns_1 "hostname: starts with dot" validate_hostname ".example.com"

# Newline/CRLF injection
assert_returns_1 "hostname: embedded newline" validate_hostname $'host\nname.com'
assert_returns_1 "hostname: embedded carriage return" validate_hostname $'host\rname.com'
assert_returns_1 "hostname: embedded CRLF" validate_hostname $'host\r\nname.com'

# Length boundaries
long_hostname="$(printf 'a%.0s' {1..63}).$(printf 'a%.0s' {1..63}).$(printf 'a%.0s' {1..63}).$(printf 'a%.0s' {1..60})"
assert_returns_0 "hostname: max total length (252 chars with dots)" validate_hostname "$long_hostname"
long_hostname_253="$(printf 'a%.0s' {1..63}).$(printf 'a%.0s' {1..63}).$(printf 'a%.0s' {1..63}).$(printf 'a%.0s' {1..61})"
assert_returns_0 "hostname: boundary length (253 chars with dots)" validate_hostname "$long_hostname_253"
long_hostname_plus="$(printf 'a%.0s' {1..63}).$(printf 'a%.0s' {1..63}).$(printf 'a%.0s' {1..63}).$(printf 'a%.0s' {1..62})"
assert_returns_1 "hostname: exceeds max length (254 chars with dots)" validate_hostname "$long_hostname_plus"

echo ""

# ============================================
# validate_username tests
# ============================================
echo "Testing validate_username..."

assert_returns_0 "username: simple alphanumeric" validate_username "user123"
assert_returns_0 "username: starts with underscore" validate_username "_user"
assert_returns_0 "username: contains hyphen" validate_username "user-name"
assert_returns_0 "username: contains underscore" validate_username "user_name"
assert_returns_0 "username: numbers only" validate_username "user12345"
assert_returns_0 "username: max length (39 chars)" validate_username "$(printf 'a%.0s' {1..39})"

assert_returns_1 "username: empty string" validate_username ""
assert_returns_1 "username: starts with hyphen" validate_username "-user"
assert_returns_1 "username: ends with hyphen" validate_username "user-"
assert_returns_1 "username: contains space" validate_username "user name"
assert_returns_1 "username: contains special chars" validate_username "user@name"
assert_returns_1 "username: contains dot" validate_username "user.name"
assert_returns_1 "username: too long (40 chars)" validate_username "$(printf 'a%.0s' {1..40})"

# Newline injection
assert_returns_1 "username: embedded newline" validate_username $'user\nname'
assert_returns_1 "username: embedded CR" validate_username $'user\rname'

# Boundaries
assert_returns_0 "username: exactly 39 chars" validate_username "$(printf 'a%.0s' {1..39})"
assert_returns_1 "username: 40 chars" validate_username "$(printf 'a%.0s' {1..40})"

echo ""

# ============================================
# validate_allocation_name tests
# ============================================
echo "Testing validate_allocation_name..."

assert_returns_0 "allocation_name: simple lowercase" validate_allocation_name "my-allocation"
assert_returns_0 "allocation_name: numbers" validate_allocation_name "allocation-123"
assert_returns_0 "allocation_name: starts with letter" validate_allocation_name "allocation"
assert_returns_0 "allocation_name: starts with number" validate_allocation_name "123allocation"

assert_returns_1 "allocation_name: empty string" validate_allocation_name ""
assert_returns_1 "allocation_name: contains uppercase" validate_allocation_name "My-Allocation"
assert_returns_1 "allocation_name: starts with hyphen" validate_allocation_name "-my-allocation"
assert_returns_1 "allocation_name: ends with hyphen" validate_allocation_name "my-allocation-"
assert_returns_1 "allocation_name: consecutive hyphens" validate_allocation_name "my--allocation"
assert_returns_1 "allocation_name: contains underscore" validate_allocation_name "my_allocation"
assert_returns_1 "allocation_name: contains dot" validate_allocation_name "my.allocation"
assert_returns_1 "allocation_name: contains space" validate_allocation_name "my allocation"

# Newline injection
assert_returns_1 "allocation_name: embedded newline" validate_allocation_name $'my\nallocation'

assert_returns_0 "allocation_name: max length (253 chars)" validate_allocation_name "$(printf 'a%.0s' {1..253})"
assert_returns_1 "allocation_name: exceeds max length (254 chars)" validate_allocation_name "$(printf 'a%.0s' {1..254})"

echo ""

# ============================================
# validate_branch_name tests
# ============================================
echo "Testing validate_branch_name..."

assert_returns_0 "branch_name: simple name" validate_branch_name "main"
assert_returns_0 "branch_name: with slash" validate_branch_name "feature/new-feature"
assert_returns_0 "branch_name: with hyphen" validate_branch_name "feature-branch"
assert_returns_0 "branch_name: with underscore" validate_branch_name "feature_branch"
assert_returns_0 "branch_name: with dot" validate_branch_name "release.v1.2"
assert_returns_0 "branch_name: numeric" validate_branch_name "123"
assert_returns_0 "branch_name: complex path" validate_branch_name "feature/user123/new-feature"

assert_returns_1 "branch_name: empty string" validate_branch_name ""
assert_returns_1 "branch_name: single @" validate_branch_name "@"
assert_returns_1 "branch_name: starts with hyphen" validate_branch_name "-feature"
assert_returns_1 "branch_name: starts with dot" validate_branch_name ".hidden"
assert_returns_1 "branch_name: ends with dot" validate_branch_name "feature."
assert_returns_1 "branch_name: ends with slash" validate_branch_name "feature/"
assert_returns_1 "branch_name: consecutive dots" validate_branch_name "feature..name"
assert_returns_1 "branch_name: consecutive slashes" validate_branch_name "feature//name"
assert_returns_1 "branch_name: contains @{ sequence" validate_branch_name "feature@{ref"
assert_returns_1 "branch_name: contains tilde" validate_branch_name "feature~name"
assert_returns_1 "branch_name: contains caret" validate_branch_name "feature^name"
assert_returns_1 "branch_name: contains colon" validate_branch_name "feature:name"
assert_returns_1 "branch_name: contains backslash" validate_branch_name "feature\\name"
assert_returns_1 "branch_name: contains space" validate_branch_name "feature name"
assert_returns_1 "branch_name: control character" validate_branch_name $'feature\x01name'

# Shell metacharacter injection (H1 fix)
assert_returns_1 "branch_name: dollar sign (command sub)" validate_branch_name 'feature/$(whoami)'
assert_returns_1 "branch_name: backtick (command sub)" validate_branch_name 'feature/`whoami`'
assert_returns_1 "branch_name: glob star" validate_branch_name "feature/*"
assert_returns_1 "branch_name: glob question" validate_branch_name "feature/file?"
assert_returns_1 "branch_name: single quote" validate_branch_name "feature/'name"
assert_returns_1 "branch_name: double quote" validate_branch_name 'feature/"name'
assert_returns_1 "branch_name: curly braces (brace expansion)" validate_branch_name "feature/file{a,b}"
assert_returns_1 "branch_name: exclamation mark (history)" validate_branch_name "feature/name!"
assert_returns_1 "branch_name: hash (comment)" validate_branch_name "feature/#comment"
assert_returns_1 "branch_name: square bracket (glob)" validate_branch_name "feature/[abc]"
assert_returns_1 "branch_name: parentheses (subshell)" validate_branch_name "feature/(sub)"
assert_returns_1 "branch_name: angle brackets (redirect)" validate_branch_name "feature/<file"

# GitHub Actions expression injection
assert_returns_1 "branch_name: GHA expression \${{ }}" validate_branch_name '${{ github.event.comment.body }}'

# Newline injection
assert_returns_1 "branch_name: embedded newline" validate_branch_name $'feature\nname'
assert_returns_1 "branch_name: embedded CR" validate_branch_name $'feature\rname'

assert_returns_0 "branch_name: max length (255 chars)" validate_branch_name "$(printf 'a%.0s' {1..255})"
assert_returns_1 "branch_name: exceeds max length (256 chars)" validate_branch_name "$(printf 'a%.0s' {1..256})"

echo ""

# ============================================
# validate_path tests
# ============================================
echo "Testing validate_path..."

# Valid paths (re-enabled after C1 null byte fix)
assert_returns_0 "path: absolute path" validate_path "/home/user/file.txt"
assert_returns_0 "path: root path" validate_path "/"
assert_returns_0 "path: nested path" validate_path "/home/user/projects/myapp/config.yaml"
assert_returns_0 "path: path with dots in filename" validate_path "/home/user/file.tar.gz"
assert_returns_0 "path: path with hyphen" validate_path "/home/user/my-project/file.txt"
assert_returns_0 "path: path with underscore" validate_path "/home/user/my_project/file.txt"
assert_returns_0 "path: path with spaces" validate_path "/home/user/my project/file.txt"
assert_returns_0 "path: relative with flag" validate_path "file.txt" --allow-relative
assert_returns_0 "path: relative dir with flag" validate_path "src/main.c" --allow-relative
assert_returns_0 "path: path with at sign" validate_path "/home/user/@scope/package"
assert_returns_0 "path: path with plus" validate_path "/opt/g++/bin"
assert_returns_0 "path: path with equals" validate_path "/tmp/key=value"
assert_returns_0 "path: path with comma" validate_path "/tmp/a,b"

# Invalid paths
assert_returns_1 "path: empty string" validate_path ""
assert_returns_1 "path: relative without flag" validate_path "./file.txt"
assert_returns_1 "path: relative no slash without flag" validate_path "file.txt"
assert_returns_1 "path: contains .. traversal" validate_path "/home/../etc/passwd"
assert_returns_1 "path: ends with .. traversal" validate_path "/home/user/.."
assert_returns_1 "path: contains ; (shell meta)" validate_path "/home/user; rm -rf /"
assert_returns_1 "path: contains | (shell meta)" validate_path "/home/user|cat /etc/passwd"
assert_returns_1 "path: contains & (shell meta)" validate_path "/home/user&malicious"
assert_returns_1 "path: contains \$ (shell meta)" validate_path '/home/user/$HOME'
assert_returns_1 "path: contains backtick (shell meta)" validate_path '/home/user/`whoami`'
assert_returns_1 "path: contains newline" validate_path $'/home/user\n/etc/passwd'
assert_returns_1 "path: contains carriage return" validate_path $'/home/user\r/etc/passwd'

# Allowlist enforcement (H3 fix)
assert_returns_1 "path: contains glob star" validate_path "/home/user/*.txt"
assert_returns_1 "path: contains glob question mark" validate_path "/home/user/file?.txt"
assert_returns_1 "path: contains square bracket (glob)" validate_path "/home/user/file[0].txt"
assert_returns_1 "path: contains parentheses" validate_path "/home/user/(subshell)"
assert_returns_1 "path: contains angle bracket" validate_path "/home/user/<redirect"
assert_returns_1 "path: contains single quote" validate_path "/home/user/it's"
assert_returns_1 "path: contains double quote" validate_path '/home/user/"file"'
assert_returns_1 "path: contains curly braces" validate_path "/home/user/{a,b}"
assert_returns_1 "path: contains exclamation" validate_path "/home/user/file!"
assert_returns_1 "path: contains hash" validate_path "/home/user/#comment"
assert_returns_1 "path: contains backslash" validate_path '/home/user/file\name'

# GHA expression injection
assert_returns_1 "path: GHA expression" validate_path '/tmp/${{ github.event.comment.body }}'

# Length boundary
assert_returns_0 "path: max length (4096 chars)" validate_path "/$(printf 'a%.0s' {1..4095})"
long_path_plus="/$(printf 'a%.0s' {1..4096})"
assert_returns_1 "path: exceeds max length" validate_path "$long_path_plus"

echo ""

# ============================================
# validate_workflow_id tests
# ============================================
echo "Testing validate_workflow_id..."

assert_returns_0 "workflow_id: simple number" validate_workflow_id "12345"
assert_returns_0 "workflow_id: large number" validate_workflow_id "123456789012345"
assert_returns_0 "workflow_id: starts with zero" validate_workflow_id "0123"

assert_returns_1 "workflow_id: empty string" validate_workflow_id ""
assert_returns_1 "workflow_id: contains letters" validate_workflow_id "123abc"
assert_returns_1 "workflow_id: contains special" validate_workflow_id "123-456"
assert_returns_1 "workflow_id: decimal" validate_workflow_id "123.456"
assert_returns_1 "workflow_id: negative" validate_workflow_id "-123"
assert_returns_1 "workflow_id: hex" validate_workflow_id "0x123"

# Newline injection
assert_returns_1 "workflow_id: embedded newline" validate_workflow_id $'123\n456'

assert_returns_0 "workflow_id: max length (19 digits, int64 max)" validate_workflow_id "9223372036854775807"
assert_returns_1 "workflow_id: exceeds max length (20 digits)" validate_workflow_id "92233720368547758071"

echo ""

# ============================================
# validate_docker_image tests
# ============================================
echo "Testing validate_docker_image..."

# Valid docker images (re-enabled after C2 tag parsing fix)
assert_returns_0 "docker_image: simple image" validate_docker_image "alpine"
assert_returns_0 "docker_image: with registry" validate_docker_image "docker.io/library/alpine"
assert_returns_0 "docker_image: with tag" validate_docker_image "alpine:latest"
assert_returns_0 "docker_image: with registry and port" validate_docker_image "localhost:5000/myimage"
assert_returns_0 "docker_image: with namespace and tag" validate_docker_image "myrepo/myimage:v1.0"
assert_returns_0 "docker_image: with digest" validate_docker_image "alpine@sha256:abc1230000000000000000000000000000000000000000000000000000000000"
assert_returns_0 "docker_image: full ghcr path" validate_docker_image "ghcr.io/tenstorrent/tt-metal/ubuntu-22-04-amd64:latest"
assert_returns_0 "docker_image: strict mode with allowed prefix" validate_docker_image "ghcr.io/tenstorrent/tt-metal/ubuntu-22-04-amd64:latest" --strict
assert_returns_0 "docker_image: strict mode with docker.io" validate_docker_image "docker.io/library/alpine:latest" --strict
assert_returns_0 "docker_image: strict mode with gcr.io" validate_docker_image "gcr.io/project/image:latest" --strict
assert_returns_0 "docker_image: registry with port and tag" validate_docker_image "registry:5000/image:v1.0"
assert_returns_0 "docker_image: deep namespace" validate_docker_image "ghcr.io/org/team/project/image:sha-abc1234"

# Invalid docker images
assert_returns_1 "docker_image: empty string" validate_docker_image ""
assert_returns_1 "docker_image: contains ; (shell meta)" validate_docker_image "alpine;rm -rf /"
assert_returns_1 "docker_image: contains | (shell meta)" validate_docker_image "alpine|cat /etc/passwd"
assert_returns_1 "docker_image: contains & (shell meta)" validate_docker_image "alpine&malicious"
assert_returns_1 "docker_image: contains backtick" validate_docker_image 'alpine`whoami`'
assert_returns_1 "docker_image: contains single quote" validate_docker_image "alpine' malicious"
assert_returns_1 "docker_image: contains double quote" validate_docker_image 'alpine" malicious'
assert_returns_1 "docker_image: contains parentheses" validate_docker_image "alpine(malicious)"
assert_returns_1 "docker_image: strict mode with unknown registry" validate_docker_image "evil.com/malicious:latest" --strict
assert_returns_1 "docker_image: invalid digest format" validate_docker_image "alpine@sha256:invalid"
assert_returns_1 "docker_image: digest too short" validate_docker_image "alpine@sha256:abc123"

# Newline injection
assert_returns_1 "docker_image: embedded newline" validate_docker_image $'alpine\n:latest'

echo ""

# ============================================
# validate_commit_sha tests
# ============================================
echo "Testing validate_commit_sha..."

assert_returns_0 "commit_sha: full SHA (40 chars)" validate_commit_sha "aabbccdd11223344556677889900aabbccdd1122"
assert_returns_0 "commit_sha: short SHA (7 chars)" validate_commit_sha "aabbccd"
assert_returns_0 "commit_sha: medium SHA (20 chars)" validate_commit_sha "aabbccdd112233445566"
assert_returns_0 "commit_sha: uppercase" validate_commit_sha "AABBCCDD11223344556677889900AABBCCDD1122"
assert_returns_0 "commit_sha: mixed case" validate_commit_sha "AaBbCcDd11223344556677889900AaBbCcDd1122"

assert_returns_1 "commit_sha: empty string" validate_commit_sha ""
assert_returns_1 "commit_sha: too short (6 chars)" validate_commit_sha "aabbcc"
assert_returns_1 "commit_sha: too long (41 chars)" validate_commit_sha "aabbccdd11223344556677889900aabbccdd11220"
assert_returns_1 "commit_sha: contains non-hex" validate_commit_sha "aabbccdd11223344556677889900ggggccdd1122"
assert_returns_1 "commit_sha: contains special" validate_commit_sha "aabbccdd11223344556677889900aabbccdd112!"

# Newline injection
assert_returns_1 "commit_sha: embedded newline" validate_commit_sha $'aabbcc\ndd112233'

echo ""

# ============================================
# validate_semver tests
# ============================================
echo "Testing validate_semver..."

assert_returns_0 "semver: simple version" validate_semver "1.2.3"
assert_returns_0 "semver: with v prefix" validate_semver "v1.2.3"
assert_returns_0 "semver: with prerelease" validate_semver "1.2.3-alpha"
assert_returns_0 "semver: with prerelease and dot" validate_semver "1.2.3-alpha.1"
assert_returns_0 "semver: with prerelease numbers" validate_semver "1.2.3-0.3.7"
assert_returns_0 "semver: with build metadata" validate_semver "1.2.3+build.123"
assert_returns_0 "semver: prerelease + build metadata" validate_semver "1.2.3-beta+exp.sha.5114f85"
assert_returns_0 "semver: major zero" validate_semver "0.0.0"
assert_returns_0 "semver: large numbers" validate_semver "999.999.999"

assert_returns_1 "semver: empty string" validate_semver ""
assert_returns_1 "semver: only two numbers" validate_semver "1.2"
assert_returns_1 "semver: four numbers (C3 fix)" validate_semver "1.2.3.4"
assert_returns_1 "semver: leading zeros in major" validate_semver "01.2.3"
assert_returns_1 "semver: leading zeros in minor" validate_semver "1.02.3"
assert_returns_1 "semver: leading zeros in patch (C3 fix)" validate_semver "1.2.03"
assert_returns_1 "semver: negative number" validate_semver "-1.2.3"
assert_returns_1 "semver: missing patch" validate_semver "1.2."
assert_returns_1 "semver: missing minor and patch" validate_semver "1.."
assert_returns_1 "semver: letters in core version" validate_semver "1.a.3"
assert_returns_1 "semver: invalid prerelease chars" validate_semver "1.2.3-alpha_beta"
assert_returns_1 "semver: leading zero in prerelease num" validate_semver "1.2.3-alpha.01"

# Trailing content injection (C3 fix)
assert_returns_1 "semver: trailing semicolon (injection)" validate_semver "1.2.3; rm -rf /"
assert_returns_1 "semver: trailing pipe" validate_semver "1.2.3|cat /etc/passwd"
assert_returns_1 "semver: trailing text" validate_semver "1.2.3extra"

echo ""

# ============================================
# validate_job_name tests
# ============================================
echo "Testing validate_job_name..."

assert_returns_0 "job_name: simple name" validate_job_name "build"
assert_returns_0 "job_name: with hyphen" validate_job_name "build-and-test"
assert_returns_0 "job_name: with underscore" validate_job_name "build_and_test"
assert_returns_0 "job_name: with numbers" validate_job_name "build-123"
assert_returns_0 "job_name: starts with number" validate_job_name "123-build"
assert_returns_0 "job_name: max length (100 chars)" validate_job_name "$(printf 'a%.0s' {1..100})"

assert_returns_1 "job_name: empty string" validate_job_name ""
assert_returns_1 "job_name: contains space" validate_job_name "build job"
assert_returns_1 "job_name: contains dot" validate_job_name "build.job"
assert_returns_1 "job_name: contains slash" validate_job_name "build/job"
assert_returns_1 "job_name: contains colon" validate_job_name "build:job"
assert_returns_1 "job_name: contains special" validate_job_name "build@job"
assert_returns_1 "job_name: exceeds max length (101 chars)" validate_job_name "$(printf 'a%.0s' {1..101})"

# Newline injection
assert_returns_1 "job_name: embedded newline" validate_job_name $'build\njob'

echo ""

# ============================================
# Sanitization Function Tests
# ============================================
echo "Testing sanitize_path..."

# sanitize_path uses realpath under the hood; results vary by OS.
# Test with paths that exist to avoid realpath failures.
sanitized=$(sanitize_path "/tmp" 2>/dev/null)
if [[ -n "$sanitized" && "$sanitized" == *"tmp"* ]]; then
    test_pass "sanitize_path: resolves existing absolute path"
else
    test_fail "sanitize_path: resolves existing absolute path (got: $sanitized)"
fi

# Traversal rejection
sanitized=$(sanitize_path "/home/../etc/passwd" 2>/dev/null)
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "sanitize_path: rejects path traversal"
else
    test_fail "sanitize_path: rejects path traversal"
fi

# Shell metacharacter rejection
sanitized=$(sanitize_path '/tmp/$(whoami)' 2>/dev/null)
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "sanitize_path: rejects shell metacharacters"
else
    test_fail "sanitize_path: rejects shell metacharacters"
fi

# Base directory containment (use existing dirs to avoid realpath issues)
sanitized=$(sanitize_path "/etc/passwd" "/tmp" 2>/dev/null)
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "sanitize_path: rejects path outside base dir"
else
    test_fail "sanitize_path: rejects path outside base dir"
fi

echo ""
echo "Testing sanitize_shell_arg..."

assert_output_equals "'hello'" "sanitize_shell_arg: simple string" sanitize_shell_arg "hello"
assert_output_equals "'hello world'" "sanitize_shell_arg: with space" sanitize_shell_arg "hello world"
assert_output_equals "'hello-world'" "sanitize_shell_arg: with hyphen" sanitize_shell_arg "hello-world"
assert_output_equals "'hello_world'" "sanitize_shell_arg: with underscore" sanitize_shell_arg "hello_world"
assert_output_contains "hello'world" "sanitize_shell_arg: with single quote (uses double quotes)" sanitize_shell_arg "hello'world"

# Adversarial inputs
assert_output_equals "''" "sanitize_shell_arg: empty string" sanitize_shell_arg ""
assert_output_equals "'\$(whoami)'" "sanitize_shell_arg: command substitution" sanitize_shell_arg '$(whoami)'
assert_output_equals "'\`whoami\`'" "sanitize_shell_arg: backtick command sub" sanitize_shell_arg '`whoami`'

# Verify quotes protect the value
quoted=$(sanitize_shell_arg "hello world")
if [[ "$quoted" == "'hello world'" ]]; then
    test_pass "sanitize_shell_arg: single quotes prevent word splitting"
else
    test_fail "sanitize_shell_arg: single quotes prevent word splitting"
fi

# Combined single and double quotes
result=$(sanitize_shell_arg "it's a \"test\"" 2>/dev/null)
if [[ -n "$result" ]]; then
    test_pass "sanitize_shell_arg: handles mixed quotes"
else
    test_fail "sanitize_shell_arg: handles mixed quotes"
fi

echo ""
echo "Testing escape_quotes..."

assert_output_equals "hello" "escape_quotes: no special chars" escape_quotes "hello"
assert_output_equals 'hello\"world' "escape_quotes: escapes double quotes" escape_quotes 'hello"world'
assert_output_equals 'hello\"world\"test' "escape_quotes: escapes multiple quotes" escape_quotes 'hello"world"test'

# H2 fix: now also escapes $, backtick, backslash
assert_output_equals 'hello\$world' "escape_quotes: escapes dollar sign" escape_quotes 'hello$world'
assert_output_equals 'hello\`world' "escape_quotes: escapes backtick" escape_quotes 'hello`world'
assert_output_equals 'hello\\world' "escape_quotes: escapes backslash" escape_quotes 'hello\world'

# Combined dangerous sequence
result=$(escape_quotes '$(rm -rf /)')
if [[ "$result" == '\$(rm -rf /)' ]]; then
    test_pass "escape_quotes: neutralizes command substitution"
else
    test_fail "escape_quotes: neutralizes command substitution (got: $result)"
fi

echo ""
echo "Testing sanitize_sed_replacement..."

assert_output_equals "hello" "sanitize_sed_replacement: no special chars" sanitize_sed_replacement "hello"
assert_output_equals "hello\\/world" "sanitize_sed_replacement: escapes slash" sanitize_sed_replacement "hello/world"
assert_output_equals "hello\\&world" "sanitize_sed_replacement: escapes ampersand" sanitize_sed_replacement "hello&world"
assert_output_equals "hello\\\\world" "sanitize_sed_replacement: escapes backslash" sanitize_sed_replacement "hello\\world"

test_string="hello/world&test"
sanitized=$(sanitize_sed_replacement "$test_string")
if [[ "$sanitized" == "hello\\/world\\&test" ]]; then
    test_pass "sanitize_sed_replacement: correctly escapes sed metacharacters"
else
    test_fail "sanitize_sed_replacement: correctly escapes sed metacharacters"
fi

echo ""

# ============================================
# Execution Helper Tests
# ============================================
echo "Testing safe_exec_script..."

# Test with invalid path (relative path)
safe_exec_script "echo test" "../invalid/path.sh" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_exec_script: rejects invalid path"
else
    test_fail "safe_exec_script: rejects invalid path"
fi

# Test requires .sh extension
safe_exec_script "echo test" "/tmp/test.txt" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_exec_script: requires .sh extension"
else
    test_fail "safe_exec_script: requires .sh extension"
fi

# Test successful execution with temp file
result=$(safe_exec_script "echo hello_world" 2>/dev/null)
ret=$?
if [[ $ret -eq 0 && "$result" == "hello_world" ]]; then
    test_pass "safe_exec_script: executes command successfully"
else
    test_fail "safe_exec_script: executes command successfully (ret=$ret, output=$result)"
fi

# Test cleanup: temp files should not accumulate
before_count=$(ls /tmp/safe_exec.*.sh 2>/dev/null | wc -l || echo 0)
safe_exec_script "echo cleanup_test" 2>/dev/null
after_count=$(ls /tmp/safe_exec.*.sh 2>/dev/null | wc -l || echo 0)
if [[ "$after_count" -le "$before_count" ]]; then
    test_pass "safe_exec_script: cleans up temp files"
else
    test_fail "safe_exec_script: cleans up temp files"
fi

echo ""
echo "Testing safe_docker_exec (validation only, no docker daemon required)..."

# Invalid image
safe_docker_exec "" "echo test" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_docker_exec: rejects empty image"
else
    test_fail "safe_docker_exec: rejects empty image"
fi

safe_docker_exec "alpine;rm -rf /" "echo test" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_docker_exec: rejects image with shell injection"
else
    test_fail "safe_docker_exec: rejects image with shell injection"
fi

# Dangerous docker options (H5 fix)
safe_docker_exec "alpine" "echo test" "--privileged" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_docker_exec: blocks --privileged flag"
else
    test_fail "safe_docker_exec: blocks --privileged flag"
fi

safe_docker_exec "alpine" "echo test" "--pid=host" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_docker_exec: blocks --pid=host flag"
else
    test_fail "safe_docker_exec: blocks --pid=host flag"
fi

safe_docker_exec "alpine" "echo test" "--cap-add" "ALL" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_docker_exec: blocks --cap-add flag"
else
    test_fail "safe_docker_exec: blocks --cap-add flag"
fi

safe_docker_exec "alpine" "echo test" "--network=host" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_docker_exec: blocks --network=host flag"
else
    test_fail "safe_docker_exec: blocks --network=host flag"
fi

safe_docker_exec "alpine" "echo test" "--security-opt" "apparmor=unconfined" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_docker_exec: blocks --security-opt flag"
else
    test_fail "safe_docker_exec: blocks --security-opt flag"
fi

echo ""
echo "Testing safe_ssh_exec (validation only, no SSH required)..."

# Invalid hostname
safe_ssh_exec "" "echo test" "" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_ssh_exec: rejects empty hostname"
else
    test_fail "safe_ssh_exec: rejects empty hostname"
fi

safe_ssh_exec "-invalid.host" "echo test" "" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_ssh_exec: rejects invalid hostname"
else
    test_fail "safe_ssh_exec: rejects invalid hostname"
fi

# Invalid username
safe_ssh_exec "valid.host" "echo test" "-invalid" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_ssh_exec: rejects invalid username"
else
    test_fail "safe_ssh_exec: rejects invalid username"
fi

# Dangerous SSH options (H6 fix)
safe_ssh_exec "valid.host" "echo test" "user" "-o" "ProxyCommand=evil" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_ssh_exec: blocks ProxyCommand"
else
    test_fail "safe_ssh_exec: blocks ProxyCommand"
fi

safe_ssh_exec "valid.host" "echo test" "user" "-o" "LocalCommand=evil" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_ssh_exec: blocks LocalCommand"
else
    test_fail "safe_ssh_exec: blocks LocalCommand"
fi

safe_ssh_exec "valid.host" "echo test" "user" "-o" "PermitLocalCommand=yes" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "safe_ssh_exec: blocks PermitLocalCommand"
else
    test_fail "safe_ssh_exec: blocks PermitLocalCommand"
fi

echo ""

# ============================================
# Error / Output Helper Tests
# ============================================
echo "Testing validation_error..."

error_output=$(validation_error "test error message" 2>&1)
if [[ "$error_output" == "::error::test error message" ]]; then
    test_pass "validation_error: outputs correct GitHub Actions format"
else
    test_fail "validation_error: outputs correct GitHub Actions format (got: $error_output)"
fi

error_output=$(validation_error "test error" "test.sh" 2>&1)
if [[ "$error_output" == "::error file=test.sh::test error" ]]; then
    test_pass "validation_error: includes file reference"
else
    test_fail "validation_error: includes file reference (got: $error_output)"
fi

error_output=$(validation_error "test error" "test.sh" "42" 2>&1)
if [[ "$error_output" == "::error file=test.sh,line=42::test error" ]]; then
    test_pass "validation_error: includes file and line"
else
    test_fail "validation_error: includes file and line (got: $error_output)"
fi

# Newline injection prevention (M1 fix)
error_output=$(validation_error $'line1\nline2' 2>&1)
line_count=$(echo "$error_output" | wc -l | tr -d ' ')
if [[ "$line_count" -eq 1 ]]; then
    test_pass "validation_error: strips newlines from message"
else
    test_fail "validation_error: strips newlines from message (got $line_count lines)"
fi

error_output=$(validation_error "msg" $'file\n::warning::injected' 2>&1)
if [[ "$error_output" != *"::warning::"* ]]; then
    test_pass "validation_error: prevents workflow command injection via file param"
else
    test_fail "validation_error: prevents workflow command injection via file param"
fi

echo ""
echo "Testing validation_warning..."

warn_output=$(validation_warning "test warning" 2>&1)
if [[ "$warn_output" == "::warning::test warning" ]]; then
    test_pass "validation_warning: outputs correct format"
else
    test_fail "validation_warning: outputs correct format (got: $warn_output)"
fi

warn_output=$(validation_warning "test warning" "file.sh" "10" 2>&1)
if [[ "$warn_output" == "::warning file=file.sh,line=10::test warning" ]]; then
    test_pass "validation_warning: includes file and line"
else
    test_fail "validation_warning: includes file and line (got: $warn_output)"
fi

# Newline injection prevention
warn_output=$(validation_warning $'injected\n::error::boom' 2>&1)
if [[ "$warn_output" != *"::error::"* ]]; then
    test_pass "validation_warning: prevents newline injection"
else
    test_fail "validation_warning: prevents newline injection"
fi

echo ""
echo "Testing validation_notice..."

notice_output=$(validation_notice "test notice" 2>&1)
if [[ "$notice_output" == "::notice::test notice" ]]; then
    test_pass "validation_notice: outputs correct format"
else
    test_fail "validation_notice: outputs correct format (got: $notice_output)"
fi

notice_output=$(validation_notice "test notice" "file.sh" "5" 2>&1)
if [[ "$notice_output" == "::notice file=file.sh,line=5::test notice" ]]; then
    test_pass "validation_notice: includes file and line"
else
    test_fail "validation_notice: includes file and line (got: $notice_output)"
fi

# Newline injection prevention
notice_output=$(validation_notice $'injected\n::error::boom' 2>&1)
if [[ "$notice_output" != *"::error::"* ]]; then
    test_pass "validation_notice: prevents newline injection"
else
    test_fail "validation_notice: prevents newline injection"
fi

echo ""
echo "Testing set_github_output..."

# Test with GITHUB_OUTPUT not set (should use deprecated format)
unset GITHUB_OUTPUT || true
output=$(set_github_output "test_name" "test_value" 2>/dev/null)
if [[ "$output" == "::set-output name=test_name::test_value" ]]; then
    test_pass "set_github_output: uses deprecated format when GITHUB_OUTPUT unset"
else
    test_fail "set_github_output: uses deprecated format when GITHUB_OUTPUT unset (got: $output)"
fi

# Test with GITHUB_OUTPUT set
temp_output=$(mktemp)
export GITHUB_OUTPUT="$temp_output"
set_github_output "output_name" "output_value" 2>/dev/null
if grep -q "output_name=output_value" "$temp_output"; then
    test_pass "set_github_output: writes to GITHUB_OUTPUT file"
else
    test_fail "set_github_output: writes to GITHUB_OUTPUT file"
fi

# Test with multi-line value
: > "$temp_output"
set_github_output "multi" $'line1\nline2' 2>/dev/null
if grep -q "line1" "$temp_output" && grep -q "line2" "$temp_output"; then
    test_pass "set_github_output: handles multi-line values"
else
    test_fail "set_github_output: handles multi-line values"
fi

# Test invalid output name
set_github_output "invalid@name" "value" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "set_github_output: rejects invalid output name"
else
    test_fail "set_github_output: rejects invalid output name"
fi

# CRLF injection test (M2 fix)
: > "$temp_output"
set_github_output "crlf_test" $'value\r\nother=injected' 2>/dev/null
if grep -q "<<OUTPUT_" "$temp_output"; then
    test_pass "set_github_output: uses delimiter for values with CR"
else
    test_fail "set_github_output: uses delimiter for values with CR"
fi

# Deprecated fallback newline stripping (M3 fix)
# Verify the output is a single line (newlines stripped) to prevent
# multi-command injection. The value may still contain "::error::" text
# but it's harmless as part of a single set-output command value.
unset GITHUB_OUTPUT
output=$(set_github_output "fallback_test" $'value\n::error::injected' 2>/dev/null)
line_count=$(printf '%s' "$output" | wc -l | tr -d ' ')
if [[ "$line_count" -le 1 ]]; then
    test_pass "set_github_output: deprecated fallback produces single line"
else
    test_fail "set_github_output: deprecated fallback produces single line (got $line_count lines)"
fi

rm -f "$temp_output"

echo ""
echo "Testing set_github_env..."

# Test name validation
set_github_env "VALID_NAME" "value" 2>/dev/null
ret=$?
if [[ $ret -eq 0 ]]; then
    test_pass "set_github_env: accepts valid name"
else
    test_fail "set_github_env: accepts valid name"
fi

set_github_env "123invalid" "value" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "set_github_env: rejects name starting with digit"
else
    test_fail "set_github_env: rejects name starting with digit"
fi

set_github_env "invalid-name" "value" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "set_github_env: rejects name with hyphen"
else
    test_fail "set_github_env: rejects name with hyphen"
fi

# Test with GITHUB_ENV file
temp_env=$(mktemp)
export GITHUB_ENV="$temp_env"
set_github_env "MY_VAR" "my_value" 2>/dev/null
if grep -q "MY_VAR=my_value" "$temp_env"; then
    test_pass "set_github_env: writes to GITHUB_ENV file"
else
    test_fail "set_github_env: writes to GITHUB_ENV file"
fi

# Multi-line value
: > "$temp_env"
set_github_env "MULTI_VAR" $'line1\nline2' 2>/dev/null
if grep -q "line1" "$temp_env" && grep -q "line2" "$temp_env"; then
    test_pass "set_github_env: handles multi-line values"
else
    test_fail "set_github_env: handles multi-line values"
fi

# CRLF uses delimiter (M2 fix)
: > "$temp_env"
set_github_env "CRLF_VAR" $'value\r\ninjected' 2>/dev/null
if grep -q "<<ENV_" "$temp_env"; then
    test_pass "set_github_env: uses delimiter for values with CR"
else
    test_fail "set_github_env: uses delimiter for values with CR"
fi

unset GITHUB_ENV
rm -f "$temp_env"

# Dangerous env var name warning (M4 fix)
warn_output=$(set_github_env "PATH" "/usr/bin" 2>&1)
if [[ "$warn_output" == *"security-sensitive"* ]]; then
    test_pass "set_github_env: warns on dangerous env var name (PATH)"
else
    test_fail "set_github_env: warns on dangerous env var name (PATH)"
fi

warn_output=$(set_github_env "LD_PRELOAD" "/evil.so" 2>&1)
if [[ "$warn_output" == *"security-sensitive"* ]]; then
    test_pass "set_github_env: warns on dangerous env var name (LD_PRELOAD)"
else
    test_fail "set_github_env: warns on dangerous env var name (LD_PRELOAD)"
fi

warn_output=$(set_github_env "BASH_ENV" "/evil.sh" 2>&1)
if [[ "$warn_output" == *"security-sensitive"* ]]; then
    test_pass "set_github_env: warns on dangerous env var name (BASH_ENV)"
else
    test_fail "set_github_env: warns on dangerous env var name (BASH_ENV)"
fi

echo ""

# ============================================
# Batch Validation Helper Tests
# ============================================
echo "Testing validate_required..."

TEST_VAR1="value1"
TEST_VAR2="value2"
TEST_VAR3=""
export TEST_VAR1 TEST_VAR2 TEST_VAR3

validate_required "TEST_VAR1" "TEST_VAR2" 2>/dev/null
ret=$?
if [[ $ret -eq 0 ]]; then
    test_pass "validate_required: passes when all vars are set"
else
    test_fail "validate_required: passes when all vars are set"
fi

validate_required "TEST_VAR1" "TEST_VAR3" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "validate_required: fails when a var is empty"
else
    test_fail "validate_required: fails when a var is empty"
fi

validate_required "TEST_VAR1" "NONEXISTENT_VAR_XYZ" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "validate_required: fails when a var is unset"
else
    test_fail "validate_required: fails when a var is unset"
fi

echo ""
echo "Testing validate_array..."

validate_array "validate_hostname" "host1.local" "host2.local" "host3.local" 2>/dev/null
ret=$?
if [[ $ret -eq 0 ]]; then
    test_pass "validate_array: passes with all valid elements"
else
    test_fail "validate_array: passes with all valid elements"
fi

validate_array "validate_hostname" "host1.local" "-invalid" "host3.local" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "validate_array: fails with one invalid element"
else
    test_fail "validate_array: fails with one invalid element"
fi

validate_array "validate_hostname" 2>/dev/null
ret=$?
if [[ $ret -eq 0 ]]; then
    test_pass "validate_array: passes with empty array"
else
    test_fail "validate_array: passes with empty array"
fi

# Validator whitelist test (cleanup-dead-code fix)
validate_array "rm" "test" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "validate_array: rejects non-function validator (rm)"
else
    test_fail "validate_array: rejects non-function validator (rm)"
fi

validate_array "cat" "/etc/passwd" 2>/dev/null
ret=$?
if [[ $ret -ne 0 ]]; then
    test_pass "validate_array: rejects non-function validator (cat)"
else
    test_fail "validate_array: rejects non-function validator (cat)"
fi

echo ""

# ============================================
# validate_positive_integer
# ============================================
echo "Testing validate_positive_integer..."

assert_returns_0 "positive_integer: valid 1" validate_positive_integer "1"
assert_returns_0 "positive_integer: valid 42" validate_positive_integer "42"
assert_returns_0 "positive_integer: valid 999" validate_positive_integer "999"
assert_returns_0 "positive_integer: with label" validate_positive_integer "30" "Timeout"
assert_returns_0 "positive_integer: at max" validate_positive_integer "100" "Value" "100"
assert_returns_1 "positive_integer: empty" validate_positive_integer ""
assert_returns_1 "positive_integer: zero" validate_positive_integer "0"
assert_returns_1 "positive_integer: negative" validate_positive_integer "-1"
assert_returns_1 "positive_integer: float" validate_positive_integer "3.14"
assert_returns_1 "positive_integer: text" validate_positive_integer "abc"
assert_returns_1 "positive_integer: mixed" validate_positive_integer "12abc"
assert_returns_1 "positive_integer: exceeds max" validate_positive_integer "101" "Value" "100"
assert_returns_1 "positive_integer: injection attempt" validate_positive_integer '$(whoami)'
assert_returns_1 "positive_integer: newline injection" validate_positive_integer $'5\n6'

echo ""

# ============================================
# Unicode / Encoding Edge Cases
# ============================================
echo "Testing unicode/encoding edge cases..."

# Fullwidth characters should be rejected by strict ASCII patterns
assert_returns_1 "hostname: fullwidth chars" validate_hostname "ｅｘａｍｐｌｅ.com"
assert_returns_1 "username: fullwidth chars" validate_username "ｕｓｅｒ"
assert_returns_1 "path: fullwidth slash traversal" validate_path "/tmp/ａ／ｂ"
# Git allows UTF-8 in branch names; fullwidth chars are not shell-dangerous
assert_returns_0 "branch_name: fullwidth chars (UTF-8 allowed)" validate_branch_name "ｆｅａｔｕｒｅ"

# Unicode homoglyph attacks
assert_returns_1 "hostname: cyrillic 'a' in hostname" validate_hostname "exаmple.com"  # 'а' is U+0430
assert_returns_1 "path: unicode division slash" validate_path "/tmp/a∕b"

echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "Test Summary"
echo "============================================"
echo "Results: $passed passed, $failed failed"

if [[ $failed -gt 0 ]]; then
    echo ""
    echo "✗ Some tests failed!"
    exit 1
else
    echo ""
    echo "✓ All tests passed!"
    exit 0
fi
