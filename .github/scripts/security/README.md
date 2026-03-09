# Security Input Validation Library

A centralized bash library providing validation, sanitization, and safe execution functions for GitHub Actions workflows and composite actions. This library helps prevent command injection, path traversal, and other common security vulnerabilities in CI/CD pipelines.

## Overview

The Security Input Validation Library provides:

- **Input Validation**: Strict validation functions for common input types (hostnames, usernames, paths, branch names, Docker images, commit SHAs, semantic versions, job names)
- **Sanitization Functions**: Safe string manipulation for shell arguments, paths, and sed replacements
- **Safe Execution Helpers**: Secure wrappers for SSH, Docker, and script execution
- **Output Helpers**: GitHub Actions compatible output formatting and environment variable handling

### Key Features

- RFC 1123 compliant hostname validation
- Path traversal prevention with base directory containment
- Shell metacharacter filtering
- Git ref-format compatible branch name validation
- Docker image reference validation with strict mode
- GitHub Actions workflow command integration (::error, ::warning, ::notice)
- Multi-line value support for GitHub outputs and environment variables

## Quick Start

### Installation

Source the library in your bash scripts:

```bash
source .github/scripts/security/input-validation.sh
```

### Basic Usage Example

```bash
#!/usr/bin/env bash
set -euo pipefail

# Source the validation library
source .github/scripts/security/input-validation.sh

# Validate inputs
validate_hostname "$HOST" || exit 1
validate_path "$CONFIG_FILE" --base-dir /etc/myapp || exit 1
validate_branch_name "$GIT_REF" || exit 1

# Sanitize for safe use
sanitized_path=$(sanitize_path "$USER_INPUT")

# Set GitHub Actions output safely
set_github_output "config_path" "$sanitized_path"
```

## Validation Functions

All validation functions return exit code 0 on success, 1 on failure, and output error messages to stderr in GitHub Actions format.

### validate_hostname()

Validates RFC 1123 compliant hostnames.

**Rules:**
- Maximum 253 characters total
- Each label (dot-separated) maximum 63 characters
- Must start and end with alphanumeric characters
- Labels can contain alphanumerics and hyphens only
- Labels cannot start or end with hyphens
- No consecutive dots

**Usage:**
```bash
validate_hostname "$hostname"
```

**Returns:** 0 if valid, 1 otherwise

**Example:**
```bash
# Valid hostnames
validate_hostname "worker-01.cluster.local"    # Success
validate_hostname "github.com"                  # Success

# Invalid hostnames
validate_hostname "-invalid.com"                  # Fails: starts with hyphen
validate_hostname "worker..01.local"              # Fails: consecutive dots
validate_hostname "host_name.com"                 # Fails: underscore not allowed
```

### validate_username()

Validates usernames (GitHub username format).

**Rules:**
- Maximum 39 characters
- Must start with alphanumeric or underscore
- Can contain alphanumerics, hyphens, and underscores
- Cannot end with hyphen

**Usage:**
```bash
validate_username "$username"
```

**Example:**
```bash
validate_username "octocat"        # Success
validate_username "user_123"       # Success
validate_username "-invalid"       # Fails: starts with hyphen
validate_username "user-"          # Fails: ends with hyphen
```

### validate_allocation_name()

Validates allocation/environment names (Kubernetes resource naming).

**Rules:**
- Must be lowercase alphanumeric
- Can contain hyphens (no consecutive)
- Must start and end with alphanumeric
- Maximum 253 characters

**Usage:**
```bash
validate_allocation_name "$name"
```

**Example:**
```bash
validate_allocation_name "my-env-123"      # Success
validate_allocation_name "MyEnv"             # Fails: uppercase
validate_allocation_name "env--name"         # Fails: consecutive hyphens
```

### validate_branch_name()

Validates git branch/tag names following `git check-ref-format` rules.

**Rules:**
- No consecutive dots (`..`)
- No forbidden sequences: `@{`, `~`, `^`, `:`, `\`, spaces
- Cannot start with `-` or `.`
- Cannot end with `.` or `/`
- Cannot be `@` (reserved in git)
- Maximum 255 characters

**Usage:**
```bash
validate_branch_name "$branch"
```

**Example:**
```bash
validate_branch_name "feature/new-thing"     # Success
validate_branch_name "main"                     # Success
validate_branch_name "../path/traversal"        # Fails: double dots
validate_branch_name "feature~name"             # Fails: contains ~
```

### validate_path()

Validates file paths with path traversal and shell metacharacter protection.

**Usage:**
```bash
validate_path "$path" [options]
```

**Options:**
- `--allow-relative`: Allow relative paths (default requires absolute paths)
- `--base-dir DIR`: Require path to be within specified base directory

**Rules:**
- Maximum 4096 characters
- No null bytes
- No directory traversal (`..`)
- No shell metacharacters (`;`, `|`, `&`, `$`, `` ` ``, `\`)
- No line ending characters

**Example:**
```bash
# Absolute path (default)
validate_path "/etc/config/myapp.yaml"          # Success

# Relative path with flag
validate_path "../config" --allow-relative      # Success

# Path containment check
validate_path "/etc/myapp/config.yaml" --base-dir /etc/myapp  # Success
validate_path "/etc/other/file.yaml" --base-dir /etc/myapp  # Fails

# Invalid paths
validate_path "/etc/../../../etc/passwd"        # Fails: path traversal
validate_path "/etc/config; rm -rf /"           # Fails: shell metacharacters
```

### validate_workflow_id()

Validates GitHub workflow run IDs.

**Rules:**
- Numeric only
- Maximum 19 characters (int64 max)

**Usage:**
```bash
validate_workflow_id "$run_id"
```

**Example:**
```bash
validate_workflow_id "123456789"     # Success
validate_workflow_id "abc123"         # Fails: non-numeric
```

### validate_docker_image()

Validates Docker image references with optional strict mode.

**Rules:**
- Maximum 2048 characters
- No dangerous characters (`;`, `|`, `&`, `$`, `` ` ``, `'`, `"`, `(`, `)`)
- Format: `[registry/][namespace/]name[:tag|@digest]`
- Digest format must be `sha256:[64 hex chars]`

**Usage:**
```bash
validate_docker_image "$image" [--strict]
```

**Options:**
- `--strict`: Only allow images from configured safe prefixes

**Example:**
```bash
# Standard validation
validate_docker_image "ghcr.io/tenstorrent/myapp:v1.0"      # Success
validate_docker_image "ubuntu:latest"                        # Success
validate_docker_image "myapp; rm -rf /"                      # Fails: shell injection

# Strict mode with allowed prefixes
validate_docker_image "ghcr.io/tenstorrent/myapp:v1.0" --strict  # Success
validate_docker_image "docker.io/evil/image:latest" --strict      # Fails: not in allowlist
```

**Configurable Safe Prefixes:**
```bash
# Edit in input-validation.sh
readonly SAFE_CONTAINER_PREFIXES=(
    "ghcr.io/tenstorrent/"
    "docker.io/library/"
    "gcr.io/"
)
```

### validate_commit_sha()

Validates commit SHA format.

**Rules:**
- 7-40 characters (full SHA: 40, short SHA: 7+)
- Hexadecimal characters only (0-9, a-f, A-F)

**Usage:**
```bash
validate_commit_sha "$sha"
```

**Example:**
```bash
validate_commit_sha "abc1234"                                    # Success (short SHA)
validate_commit_sha "a1b2c3d4e5f6789012345678901234567890abcd"  # Success (full SHA)
validate_commit_sha "abc"                                         # Fails: too short
validate_commit_sha "not-hex"                                     # Fails: invalid characters
```

### validate_semver()

Validates semantic version strings.

**Rules:**
- Format: `MAJOR.MINOR.PATCH[-prerelease][+build]`
- Leading `v` is allowed (loose mode)
- Major, minor, patch are numeric without leading zeros
- Prerelease identifiers: alphanumeric or numeric (no leading zeros)

**Usage:**
```bash
validate_semver "$version"
```

**Example:**
```bash
validate_semver "1.2.3"              # Success
validate_semver "v1.2.3-alpha.1"      # Success
validate_semver "2.0.0+build.123"     # Success
validate_semver "01.2.3"              # Fails: leading zero
validate_semver "1.2"                  # Fails: missing patch
```

### validate_job_name()

Validates GitHub Actions job identifiers.

**Rules:**
- Maximum 100 characters
- Alphanumerics, underscores, and hyphens only

**Usage:**
```bash
validate_job_name "$job_name"
```

**Example:**
```bash
validate_job_name "build-and-test"    # Success
validate_job_name "deploy_prod"         # Success
validate_job_name "deploy prod"         # Fails: contains space
```

## Sanitization Functions

### sanitize_path()

Sanitizes a path by resolving it and ensuring it's safe.

**Usage:**
```bash
sanitized=$(sanitize_path "$path" ["$base_dir"])
```

**Returns:** Sanitized absolute path on stdout, or empty on failure

**Example:**
```bash
# Get absolute path
abs_path=$(sanitize_path "./config.yaml")
# Returns: /current/working/dir/config.yaml

# With base directory containment
safe_path=$(sanitize_path "/etc/myapp/config.yaml" "/etc/myapp")
# Returns: /etc/myapp/config.yaml (or fails if outside base)
```

### sanitize_shell_arg()

Sanitizes a string for safe use as a shell argument with proper quoting.

**Usage:**
```bash
quoted=$(sanitize_shell_arg "$argument")
```

**Returns:** Properly quoted string on stdout

**Example:**
```bash
# Safe argument with single quotes
arg=$(sanitize_shell_arg "hello world")
# Returns: 'hello world'

# Argument with single quotes (uses double quotes with escaping)
arg=$(sanitize_shell_arg "it's done")
# Returns: "it\'s done"

cmd="echo $arg"  # Safe to use in command construction
```

### escape_quotes()

Escapes double quotes in a string for safe variable expansion.

**Usage:**
```bash
escaped=$(escape_quotes "$string")
```

**Example:**
```bash
message='Hello "World"'
escaped=$(escape_quotes "$message")
# Returns: Hello \"World\"
```

### sanitize_sed_replacement()

Escapes special characters for use in sed replacement strings.

**Usage:**
```bash
safe=$(sanitize_sed_replacement "$string")
```

**Example:**
```bash
replacement="/path/to/file"
safe_replacement=$(sanitize_sed_replacement "$replacement")
sed "s|PLACEHOLDER|$safe_replacement|" input.txt
```

## Safe Execution Helpers

### safe_exec_script()

Creates and executes a script file from a command string. Safer than `bash -c` because it avoids shell parsing issues.

**Usage:**
```bash
safe_exec_script "$command" ["$script_file"] ["$work_dir"]
```

**Parameters:**
- `$1`: Command string or script content
- `$2`: Optional path for script file (default: temp file)
- `$3`: Optional working directory to execute in

**Example:**
```bash
# Execute command in temp script
safe_exec_script "echo 'Hello World' && ls -la"

# Execute with specific script path
safe_exec_script "complex_command" "/tmp/myscript.sh" "/workspace"
```

### safe_docker_exec()

Safely executes commands in Docker containers with validated inputs.

**Usage:**
```bash
safe_docker_exec "$image" "$command" ["$docker_opts"...]
```

**Parameters:**
- `$1`: Docker image name (validated)
- `$2`: Command to execute
- `$3+`: Additional docker run options

**Example:**
```bash
# Basic execution
safe_docker_exec "ubuntu:latest" "echo 'Hello from container'"

# With additional options
safe_docker_exec "ghcr.io/tenstorrent/myapp:v1.0" \
    "run_tests.sh" \
    "-v" "/workspace:/workspace" \
    "-e" "DEBUG=1"
```

### safe_ssh_exec()

Safely executes commands via SSH with validated hostname and username.

**Usage:**
```bash
safe_ssh_exec "$hostname" "$command" ["$username"] ["$ssh_opts"...]
```

**Parameters:**
- `$1`: Target hostname (validated)
- `$2`: Command to execute
- `$3`: Optional username (validated if provided)
- `$4+`: Additional SSH options

**Default Options:**
- `StrictHostKeyChecking=yes`
- `BatchMode=yes`

**Example:**
```bash
# Execute command on remote host
safe_ssh_exec "worker-01.cluster.local" "uptime"

# With username and custom options
safe_ssh_exec "server.example.com" \
    "deploy.sh" \
    "deployuser" \
    "-p" "2222" \
    "-i" "/path/to/key"
```

## Output Helpers

### set_github_output()

Safely sets GitHub Actions output with proper escaping for multi-line values.

**Usage:**
```bash
set_github_output "$name" "$value"
```

**Features:**
- Validates output name (alphanumerics, underscores, hyphens only)
- Handles multi-line values using delimiter syntax
- Uses `GITHUB_OUTPUT` environment file (recommended) or falls back to deprecated `::set-output`

**Example:**
```bash
# Single-line output
set_github_output "status" "success"

# Multi-line output
json_output='{"key": "value", "data": "complex"}'
set_github_output "json_data" "$json_output"
```

### set_github_env()

Safely sets GitHub Actions environment variables with multi-line support.

**Usage:**
```bash
set_github_env "$name" "$value"
```

**Features:**
- Validates variable name (shell variable naming rules)
- Handles multi-line values
- Uses `GITHUB_ENV` environment file or exports in current shell

**Example:**
```bash
set_github_env "MY_VAR" "my_value"
set_github_env "PATH_APPEND" "/extra/path"
```

### validation_error()

Outputs validation error in GitHub Actions format.

**Usage:**
```bash
validation_error "$message" ["$file"] ["$line"]
```

**Example:**
```bash
validation_error "Invalid input provided"
validation_error "Invalid input provided" "action.yml" "42"
```

**Output:**
```
::error::Invalid input provided
::error file=action.yml,line=42::Invalid input provided
```

### validation_warning()

Outputs validation warning in GitHub Actions format.

**Usage:**
```bash
validation_warning "$message" ["$file"] ["$line"]
```

### validation_notice()

Outputs validation notice in GitHub Actions format.

**Usage:**
```bash
validation_notice "$message" ["$file"] ["$line"]
```

## Batch Validation Helpers

### validate_required()

Validates that multiple required variables exist and are non-empty.

**Usage:**
```bash
validate_required "$var1" "$var2" "$var3"
```

**Example:**
```bash
validate_required "HOSTNAME" "USERNAME" "API_KEY" || exit 1
# Fails if any variable is empty or unset
```

### validate_array()

Runs a validation function on each element of an array.

**Usage:**
```bash
validate_array "$validator_function" "${array[@]}"
```

**Example:**
```bash
hostnames=("host1.local" "host2.local" "host3.local")
validate_array "validate_hostname" "${hostnames[@]}" || exit 1
```

## Usage Examples

### Before (Inline Validation)

```yaml
# action.yml
name: 'Configure Application'
inputs:
  directory:
    description: 'Directory for configuration'
    required: true
  hostname:
    description: 'Target hostname'
    required: true

runs:
  using: 'composite'
  steps:
    - name: Validate directory input
      shell: bash
      env:
        DIRECTORY: ${{ inputs.directory }}
      run: |
        if [[ "$DIRECTORY" == *".."* ]]; then
          echo "Error: directory cannot contain path traversal sequences"
          exit 1
        fi
        if [[ "$DIRECTORY" =~ [;\|\&\$\`] ]]; then
          echo "Error: directory contains shell metacharacters"
          exit 1
        fi
        if [[ ! "$DIRECTORY" =~ ^[A-Za-z0-9._/-]+$ ]]; then
          echo "Error: directory contains invalid characters"
          exit 1
        fi

    - name: Validate hostname
      shell: bash
      env:
        HOSTNAME: ${{ inputs.hostname }}
      run: |
        if [[ ! "$HOSTNAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*[a-zA-Z0-9]$ ]]; then
          echo "Error: invalid hostname format"
          exit 1
        fi

    - name: Deploy configuration
      shell: bash
      env:
        DIRECTORY: ${{ inputs.directory }}
        HOSTNAME: ${{ inputs.hostname }}
      run: |
        scp "$DIRECTORY/config.yaml" "$HOSTNAME":/etc/myapp/
```

### After (Using Library)

```yaml
# action.yml
name: 'Configure Application'
inputs:
  directory:
    description: 'Directory for configuration'
    required: true
  hostname:
    description: 'Target hostname'
    required: true

runs:
  using: 'composite'
  steps:
    - name: Validate inputs
      id: validate
      shell: bash
      env:
        DIRECTORY: ${{ inputs.directory }}
        HOSTNAME: ${{ inputs.hostname }}
      run: |
        source .github/scripts/security/input-validation.sh

        # Validate all inputs with consistent error messages
        validate_path "$DIRECTORY" --allow-relative || exit 1
        validate_hostname "$HOSTNAME" || exit 1

        # Sanitize for safe use
        SAFE_DIR=$(sanitize_path "$DIRECTORY")
        set_github_output "safe_directory" "$SAFE_DIR"

    - name: Deploy configuration
      shell: bash
      env:
        DIRECTORY: ${{ steps.validate.outputs.safe_directory }}
        HOSTNAME: ${{ inputs.hostname }}
      run: |
        source .github/scripts/security/input-validation.sh

        # Use safe SSH execution
        safe_ssh_exec "$HOSTNAME" "sudo mkdir -p /etc/myapp"
        scp "$DIRECTORY/config.yaml" "$HOSTNAME":/tmp/config.yaml
        safe_ssh_exec "$HOSTNAME" "sudo mv /tmp/config.yaml /etc/myapp/"
```

## Migration Guide

### Step 1: Source the Library

Add the source line at the beginning of your script steps:

```bash
source .github/scripts/security/input-validation.sh
```

### Step 2: Replace Inline Validation

Replace ad-hoc validation with library functions:

| Before (Inline) | After (Library) |
|----------------|-----------------|
| `[[ "$host" =~ ^[a-zA-Z0-9.-]+$ ]]` | `validate_hostname "$host"` |
| `[[ "$path" != *".."* ]]` | `validate_path "$path"` |
| `[[ "$id" =~ ^[0-9]+$ ]]` | `validate_workflow_id "$id"` |
| `echo "::set-output name=$name::$value"` | `set_github_output "$name" "$value"` |

### Step 3: Add Input Sanitization

Use sanitization functions when passing inputs to commands:

```bash
# Before
cmd="echo $USER_INPUT"  # Dangerous!
eval "$cmd"

# After
source .github/scripts/security/input-validation.sh
safe_input=$(sanitize_shell_arg "$USER_INPUT")
cmd="echo $safe_input"  # Safe to execute
```

### Step 4: Use Safe Execution Helpers

Replace direct SSH/Docker execution with safe wrappers:

```bash
# Before
ssh "$HOSTNAME" "$COMMAND"  # No validation

# After
safe_ssh_exec "$HOSTNAME" "$COMMAND"  # Validated inputs
```

### Step 5: Batch Validation

For multiple required inputs, use batch validation:

```bash
# Before
[[ -z "$VAR1" ]] && { echo "Missing VAR1"; exit 1; }
[[ -z "$VAR2" ]] && { echo "Missing VAR2"; exit 1; }
[[ -z "$VAR3" ]] && { echo "Missing VAR3"; exit 1; }

# After
validate_required "VAR1" "VAR2" "VAR3" || exit 1
```

## Testing

### Manual Testing

Test individual functions by sourcing the library and running validation:

```bash
source .github/scripts/security/input-validation.sh

# Test valid input
validate_hostname "valid.host.local" && echo "PASS" || echo "FAIL"

# Test invalid input
validate_hostname "-invalid" && echo "FAIL (should have rejected)" || echo "PASS (correctly rejected)"

# Test edge cases
validate_path "/etc/../../../etc/passwd" && echo "FAIL" || echo "PASS"
validate_commit_sha "abc1234" && echo "PASS" || echo "FAIL"
```

### Automated Testing Script

Create a test script to validate all functions:

```bash
#!/usr/bin/env bash
set -euo pipefail

source .github/scripts/security/input-validation.sh

# Test data
valid_hostnames=("host.local" "worker-01.cluster" "github.com")
invalid_hostnames=("-bad.com" "bad-.com" "double..dots" "a.b.c.d.e.f" "a" "host.c")

# Run tests
pass_count=0
fail_count=0

for host in "${valid_hostnames[@]}"; do
    if validate_hostname "$host" 2>/dev/null; then
        ((pass_count++))
        echo "PASS: validate_hostname '$host'"
    else
        ((fail_count++))
        echo "FAIL: validate_hostname '$host' (expected valid)"
    fi
done

for host in "${invalid_hostnames[@]}"; do
    if ! validate_hostname "$host" 2>/dev/null; then
        ((pass_count++))
        echo "PASS: validate_hostname '$host' (correctly rejected)"
    else
        ((fail_count++))
        echo "FAIL: validate_hostname '$host' (expected invalid)"
    fi
done

echo ""
echo "Results: $pass_count passed, $fail_count failed"
```

### Testing in CI

Add a workflow step to validate the library itself:

```yaml
- name: Test validation library
  run: bash .github/scripts/security/tests/test-validation.sh
```

## Security Considerations

### Threat Model

This library protects against the following attack vectors in GitHub Actions workflows:

1. **Command Injection**: Malicious inputs containing shell metacharacters (`;`, `|`, `&`, `` ` ``, `$`)
2. **Path Traversal**: Inputs containing `..` to access files outside intended directories
3. **Host Injection**: Malicious hostnames that could redirect SSH/SCP operations
4. **Docker Image Injection**: Malicious image references that could execute arbitrary code

### Defense in Depth

The library implements multiple layers of defense:

1. **Input Validation**: Strict validation of all inputs before use
2. **Sanitization**: Proper quoting and escaping for safe use in commands
3. **Safe Execution**: Validated wrappers for dangerous operations (SSH, Docker)
4. **Output Encoding**: Proper handling of multi-line values and special characters

### Configuration Limits

The following limits are enforced (configurable in the library):

| Input Type | Maximum Length | Pattern |
|------------|----------------|---------|
| Hostname | 253 chars (total), 63 chars (label) | RFC 1123 |
| Username | 39 chars | `[a-zA-Z0-9_][a-zA-Z0-9_-]*` |
| Allocation Name | 253 chars | lowercase alphanumeric + hyphens |
| Branch Name | 255 chars | git check-ref-format |
| Path | 4096 chars | No `..`, shell metacharacters |
| Workflow ID | 19 chars | Numeric only |
| Docker Image | 2048 chars | Valid image reference |
| Commit SHA | 40 chars | Hexadecimal only |
| Job Name | 100 chars | Alphanumeric + underscore + hyphen |

### Safe Container Prefixes

When using `validate_docker_image` with `--strict`, only images from configured prefixes are allowed:

```bash
readonly SAFE_CONTAINER_PREFIXES=(
    "ghcr.io/tenstorrent/"
    "docker.io/library/"
    "gcr.io/"
)
```

Modify this list in `input-validation.sh` to match your organization's trusted registries.

### Limitations

1. **Context-Specific Validation**: The library provides generic validation. Some inputs may require additional context-specific checks.
2. **Unicode Handling**: Some functions may not fully handle all Unicode edge cases. Stick to ASCII where possible.
3. **False Positives**: Strict validation may reject valid but unusual inputs. Use `--allow-relative` and other flags appropriately.

### Best Practices

1. **Always validate before use**: Call validation functions at the entry point of your action/script
2. **Fail fast**: Exit immediately on validation failure to prevent partial execution
3. **Use safe execution helpers**: Prefer `safe_ssh_exec` and `safe_docker_exec` over direct command construction
4. **Sanitize outputs**: Use `set_github_output` and `set_github_env` for all workflow outputs
5. **Log validation failures**: The library outputs GitHub Actions formatted errors that appear in workflow logs
6. **Test edge cases**: Add test cases for your specific input patterns

### Related CWE References

This library mitigates the following Common Weakness Enumerations:

- **[CWE-78](https://cwe.mitre.org/data/definitions/78.html)**: Improper Neutralization of Special Elements used in an OS Command (OS Command Injection)
- **[CWE-22](https://cwe.mitre.org/data/definitions/22.html)**: Improper Limitation of a Pathname to a Restricted Directory (Path Traversal)
- **[CWE-77](https://cwe.mitre.org/data/definitions/77.html)**: Improper Neutralization of Special Elements used in a Command (Command Injection)
- **[CWE-116](https://cwe.mitre.org/data/definitions/116.html)**: Improper Encoding or Escaping of Output

### Out of Scope

The following concerns are **not** addressed by this library and require additional mitigations:

1. **TOCTOU (Time-of-Check to Time-of-Use)**: A validated path could be changed between validation and use (e.g., symlink races). This is inherent to filesystem validation in shell.
2. **Secrets exposure**: The library does not prevent secrets from being logged or leaked. Use GitHub Actions `add-mask` and avoid `echo`-ing sensitive values.
3. **Supply chain attacks**: Beyond action pinning checks in `check-actions-security.sh`, the library does not verify dependency integrity.
4. **Network-level attacks**: SSRF via manipulated hostnames is checked only at the format level, not at the DNS resolution level.

## Security Linting Script

### `check-actions-security.sh`

A standalone linting script that scans GitHub Actions workflows and composite actions for common security anti-patterns.

**Usage:**

```bash
.github/scripts/security/check-actions-security.sh [--strict]
```

**Options:**
- `--strict`: Exit with non-zero code if any issues are found (useful for CI gating)

**Checks performed:**

| # | Check | Severity | Description |
|---|-------|----------|-------------|
| 1 | Event data interpolation | HIGH | Direct use of `github.event.comment.body`, `issue.body`, `pull_request.body/title` in `run:` blocks |
| 2 | `eval` usage | HIGH | Any `eval` statement in workflows or actions |
| 3 | `bash -c` with `${{ }}` | MEDIUM | Direct expression interpolation in `bash -c` commands |
| 4 | `pull_request_target` + checkout | HIGH | Workflows using `pull_request_target` that also checkout PR code |
| 5 | Unpinned external actions | LOW | Actions using mutable refs (`@v*`, `@main`) instead of SHA pins |
| 6 | `secrets: inherit` | LOW | Excessive use of `secrets: inherit` (threshold: 50) |
| 7 | Token exposure in logs | HIGH | `echo`/`print` statements containing `GITHUB_TOKEN` or runtime tokens |
| 8 | Broad permissions | MEDIUM | Workflows using `write-all` permissions |
| 9 | Ref-based injection | MEDIUM | `head.ref`/`base.ref` in `${{ }}` expressions |
| 10 | `curl \| bash` patterns | HIGH | Downloading and piping scripts directly to a shell |

### Reporting Security Issues

If you discover a security vulnerability in this library or its usage in workflows, please report it to the security team following your organization's responsible disclosure process.
