#!/usr/bin/env bash
# Centralized Input Validation Library for GitHub Actions
# Usage: source .github/scripts/security/input-validation.sh
#
# This library provides validation, sanitization, and safe execution
# functions for GitHub Actions workflows and composite actions.
#
# Example:
#   source .github/scripts/security/input-validation.sh
#   validate_hostname "$HOST" || exit 1
#   validate_path "$FILE" || exit 1

set -euo pipefail

# ============================================
# Configuration / Whitelists
# ============================================

# Known safe container prefixes
readonly SAFE_CONTAINER_PREFIXES=(
    "ghcr.io/tenstorrent/"
    "docker.io/library/"
    "gcr.io/"
)

# Known safe GitHub users/orgs for forks (optional)
readonly SAFE_GITHUB_USERS=()

# Maximum lengths for various inputs
readonly MAX_HOSTNAME_LENGTH=253
readonly MAX_HOSTNAME_LABEL_LENGTH=63
readonly MAX_USERNAME_LENGTH=39
readonly MAX_ALLOCATION_NAME_LENGTH=253
readonly MAX_BRANCH_NAME_LENGTH=255
readonly MAX_PATH_LENGTH=4096
readonly MAX_WORKFLOW_ID_LENGTH=19  # Max int64
readonly MAX_DOCKER_IMAGE_LENGTH=2048
readonly MAX_COMMIT_SHA_LENGTH=40

# ============================================
# Internal Helpers
# ============================================

# ============================================
# Validation Functions
# ============================================

# Validates RFC 1123 compliant hostnames
# Returns 0 if valid, 1 otherwise
# Outputs error message to stderr on failure
validate_hostname() {
    local hostname="$1"

    # Check empty
    if [[ -z "$hostname" ]]; then
        validation_error "Hostname cannot be empty"
        return 1
    fi

    # Check max length (253 chars)
    if [[ ${#hostname} -gt $MAX_HOSTNAME_LENGTH ]]; then
        validation_error "Hostname exceeds maximum length of $MAX_HOSTNAME_LENGTH characters"
        return 1
    fi

    # Check overall pattern (alphanumeric, dots, hyphens)
    if [[ ! "$hostname" =~ ^[a-zA-Z0-9] ]]; then
        validation_error "Hostname must start with alphanumeric character"
        return 1
    fi

    if [[ ! "$hostname" =~ [a-zA-Z0-9]$ ]]; then
        validation_error "Hostname must end with alphanumeric character"
        return 1
    fi

    # Check for consecutive dots
    if [[ "$hostname" =~ \.\. ]]; then
        validation_error "Hostname cannot contain consecutive dots"
        return 1
    fi

    # Reject control characters (including newlines) that could bypass label checks
    if [[ "$hostname" =~ [[:cntrl:]] ]]; then
        validation_error "Hostname cannot contain control characters"
        return 1
    fi

    # Validate each label (parts between dots)
    local labels
    IFS='.' read -ra labels <<< "$hostname"
    local label
    for label in "${labels[@]}"; do
        # Check label length (max 63 chars)
        if [[ ${#label} -gt $MAX_HOSTNAME_LABEL_LENGTH ]]; then
            validation_error "Hostname label '$label' exceeds maximum length of $MAX_HOSTNAME_LABEL_LENGTH characters"
            return 1
        fi

        # Check label content
        if [[ ! "$label" =~ ^[a-zA-Z0-9-]+$ ]]; then
            validation_error "Hostname label '$label' contains invalid characters (only alphanumerics and hyphens allowed)"
            return 1
        fi

        # Check label doesn't start/end with hyphen
        if [[ "$label" =~ ^- || "$label" =~ -$ ]]; then
            validation_error "Hostname label '$label' cannot start or end with hyphen"
            return 1
        fi
    done

    return 0
}

# Validates usernames (alphanumeric, underscore, hyphen)
# GitHub usernames follow these rules:
# - Start with alphanumeric or underscore
# - Contain alphanumerics, hyphens, underscores
# - Max 39 characters
validate_username() {
    local username="$1"

    # Check empty
    if [[ -z "$username" ]]; then
        validation_error "Username cannot be empty"
        return 1
    fi

    # Check max length
    if [[ ${#username} -gt $MAX_USERNAME_LENGTH ]]; then
        validation_error "Username exceeds maximum length of $MAX_USERNAME_LENGTH characters"
        return 1
    fi

    # Check valid characters and format
    if [[ ! "$username" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_-]*$ ]]; then
        validation_error "Username contains invalid characters (only alphanumerics, underscores, and hyphens allowed, must start with alphanumeric or underscore)"
        return 1
    fi

    # Check doesn't end with hyphen
    if [[ "$username" =~ -$ ]]; then
        validation_error "Username cannot end with hyphen"
        return 1
    fi

    return 0
}

# Validates allocation/environment names (Kubernetes resource naming)
# Rules:
# - Must be lowercase alphanumeric
# - Can contain hyphens
# - Must start and end with alphanumeric
# - Max 253 characters
# - No consecutive hyphens
validate_allocation_name() {
    local name="$1"

    # Check empty
    if [[ -z "$name" ]]; then
        validation_error "Allocation name cannot be empty"
        return 1
    fi

    # Check max length
    if [[ ${#name} -gt $MAX_ALLOCATION_NAME_LENGTH ]]; then
        validation_error "Allocation name exceeds maximum length of $MAX_ALLOCATION_NAME_LENGTH characters"
        return 1
    fi

    # Check lowercase only
    if [[ "$name" =~ [A-Z] ]]; then
        validation_error "Allocation name must be lowercase only"
        return 1
    fi

    # Check start/end with alphanumeric
    if [[ ! "$name" =~ ^[a-z0-9] ]]; then
        validation_error "Allocation name must start with lowercase alphanumeric character"
        return 1
    fi

    if [[ ! "$name" =~ [a-z0-9]$ ]]; then
        validation_error "Allocation name must end with lowercase alphanumeric character"
        return 1
    fi

    # Check no consecutive hyphens
    if [[ "$name" =~ -- ]]; then
        validation_error "Allocation name cannot contain consecutive hyphens"
        return 1
    fi

    # Check valid characters
    if [[ ! "$name" =~ ^[a-z0-9-]+$ ]]; then
        validation_error "Allocation name contains invalid characters (only lowercase alphanumerics and hyphens allowed)"
        return 1
    fi

    return 0
}

# Validates git branch/tag names
# Rules based on git check-ref-format:
# - Cannot contain consecutive dots (..)
# - Cannot contain sequences like @{, space, ~, ^, :, \
# - Cannot start with -, end with ., or end with /
# - Cannot be @
validate_branch_name() {
    local branch="$1"

    # Check empty
    if [[ -z "$branch" ]]; then
        validation_error "Branch name cannot be empty"
        return 1
    fi

    # Check max length
    if [[ ${#branch} -gt $MAX_BRANCH_NAME_LENGTH ]]; then
        validation_error "Branch name exceeds maximum length of $MAX_BRANCH_NAME_LENGTH characters"
        return 1
    fi

    # Check for @ (reserved in git)
    if [[ "$branch" == "@" ]]; then
        validation_error "Branch name cannot be '@'"
        return 1
    fi

    # Check for control characters
    if [[ "$branch" =~ [[:cntrl:]] ]]; then
        validation_error "Branch name cannot contain control characters"
        return 1
    fi

    # Check doesn't start with hyphen
    if [[ "$branch" =~ ^- ]]; then
        validation_error "Branch name cannot start with hyphen"
        return 1
    fi

    # Check doesn't start with dot
    if [[ "$branch" =~ ^\. ]]; then
        validation_error "Branch name cannot start with dot"
        return 1
    fi

    # Check doesn't end with dot
    if [[ "$branch" =~ \.$ ]]; then
        validation_error "Branch name cannot end with dot"
        return 1
    fi

    # Check doesn't end with slash
    if [[ "$branch" =~ /$ ]]; then
        validation_error "Branch name cannot end with slash"
        return 1
    fi

    # Check for double dots (path traversal or parent reference)
    if [[ "$branch" =~ \.\. ]]; then
        validation_error "Branch name cannot contain consecutive dots"
        return 1
    fi

    # Check for forbidden sequences
    if [[ "$branch" =~ @\{ ]]; then
        validation_error "Branch name cannot contain '@{' sequence"
        return 1
    fi

    if [[ "$branch" =~ [~\^:] ]]; then
        validation_error "Branch name cannot contain '~', '^', or ':' characters"
        return 1
    fi

    # Check for backslash
    if [[ "$branch" =~ \\ ]]; then
        validation_error "Branch name cannot contain backslash"
        return 1
    fi

    # Check for space characters
    if [[ "$branch" =~ [[:space:]] ]]; then
        validation_error "Branch name cannot contain spaces"
        return 1
    fi

    # Check for consecutive slashes
    if [[ "$branch" =~ // ]]; then
        validation_error "Branch name cannot contain consecutive slashes"
        return 1
    fi

    # Reject shell metacharacters that could enable injection when branch names
    # are used in shell contexts (e.g., $() for command substitution, glob chars).
    # Pattern stored in variable to prevent bash from consuming escape sequences.
    # The ] must be first in the bracket expression to be treated as literal.
    local _shell_metachar_pattern='[][$*?{}"'"'"'!#()<>`]'
    if [[ "$branch" =~ $_shell_metachar_pattern ]]; then
        validation_error "Branch name cannot contain shell metacharacters"
        return 1
    fi

    return 0
}

# Validates file paths
# Prevents path traversal and shell metacharacters
# $1: path to validate
# $2: optional flag --allow-relative (allows relative paths, otherwise requires absolute)
# $3: optional base directory (requires path to be within this directory)
validate_path() {
    local path="$1"
    local allow_relative=false
    local base_dir=""

    # Parse optional arguments
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --allow-relative)
                allow_relative=true
                ;;
            --base-dir)
                base_dir="$2"
                shift
                ;;
        esac
        shift
    done

    # Check empty
    if [[ -z "$path" ]]; then
        validation_error "Path cannot be empty"
        return 1
    fi

    # Check max length
    if [[ ${#path} -gt $MAX_PATH_LENGTH ]]; then
        validation_error "Path exceeds maximum length of $MAX_PATH_LENGTH characters"
        return 1
    fi

    # Null byte check intentionally omitted: bash variables cannot contain
    # null bytes (\0), so any null byte injection is already neutralized
    # by the shell itself before reaching this function.

    # Check for path traversal sequences
    if [[ "$path" =~ \.\.(/|$) ]]; then
        validation_error "Path contains directory traversal sequence"
        return 1
    fi

    # Allowlist: only permit safe characters in paths.
    # Letters, digits, dot, underscore, slash, hyphen, plus, at, space, comma, equals.
    if [[ ! "$path" =~ ^[a-zA-Z0-9._/@+[:space:],:=-]+$ ]]; then
        validation_error "Path contains forbidden characters (only alphanumerics, ._/@+,-= and spaces allowed)"
        return 1
    fi

    # Check for carriage return/newline (command injection risk)
    if [[ "$path" =~ [$'\r\n'] ]]; then
        validation_error "Path cannot contain line ending characters"
        return 1
    fi

    # If absolute path required
    if [[ "$allow_relative" == false && ! "$path" =~ ^/ ]]; then
        validation_error "Path must be absolute (start with /)"
        return 1
    fi

    # If base directory specified, verify path is within it
    if [[ -n "$base_dir" ]]; then
        # Normalize paths for comparison
        local normalized_path
        local normalized_base

        if command -v realpath &>/dev/null; then
            normalized_path=$(realpath -m "$path" 2>/dev/null) || \
                normalized_path=$(realpath "$path" 2>/dev/null) || \
                normalized_path="$path"
            normalized_base=$(realpath -m "$base_dir" 2>/dev/null) || \
                normalized_base=$(realpath "$base_dir" 2>/dev/null) || \
                normalized_base="$base_dir"
        else
            normalized_path="$path"
            normalized_base="$base_dir"
        fi

        # Ensure base_dir ends with /
        [[ "$normalized_base" =~ /$ ]] || normalized_base="${normalized_base}/"

        if [[ ! "$normalized_path" =~ ^"$normalized_base" ]]; then
            validation_error "Path must be within base directory: $base_dir"
            return 1
        fi
    fi

    return 0
}

# Validates GitHub workflow run IDs
# Must be numeric only
validate_workflow_id() {
    local id="$1"

    # Check empty
    if [[ -z "$id" ]]; then
        validation_error "Workflow ID cannot be empty"
        return 1
    fi

    # Check max length (reasonable limit for int64)
    if [[ ${#id} -gt $MAX_WORKFLOW_ID_LENGTH ]]; then
        validation_error "Workflow ID exceeds maximum length"
        return 1
    fi

    # Check numeric only
    if [[ ! "$id" =~ ^[0-9]+$ ]]; then
        validation_error "Workflow ID must be numeric only"
        return 1
    fi

    return 0
}

# Validates Docker image references
# Checks format: [registry/][namespace/]name[:tag|@digest]
validate_docker_image() {
    local image="$1"
    local strict=false

    # Check for strict mode (check against safe prefixes)
    if [[ "${2:-}" == "--strict" ]]; then
        strict=true
    fi

    # Check empty
    if [[ -z "$image" ]]; then
        validation_error "Docker image cannot be empty"
        return 1
    fi

    # Check max length
    if [[ ${#image} -gt $MAX_DOCKER_IMAGE_LENGTH ]]; then
        validation_error "Docker image reference exceeds maximum length of $MAX_DOCKER_IMAGE_LENGTH characters"
        return 1
    fi

    # Check for dangerous characters
    if [[ "$image" =~ [\;\|\&\$\`\'\"\(\)] ]]; then
        validation_error "Docker image contains forbidden characters"
        return 1
    fi

    # Split into name and digest/tag components
    local tag=""
    local digest=""
    local name="$image"

    # Extract digest first (takes precedence over tag)
    if [[ "$image" == *@sha256:* ]]; then
        digest="${image#*@}"
        name="${image%%@*}"

        if [[ ! "$digest" =~ ^sha256:[a-f0-9]{64}$ ]]; then
            validation_error "Invalid image digest format"
            return 1
        fi
    elif [[ "$image" == *:* ]]; then
        # Find the tag: it's the part after the LAST colon that appears
        # AFTER the last slash. This distinguishes port numbers
        # (registry:5000/image) from tags (image:latest).
        local after_last_slash="${image##*/}"
        if [[ "$after_last_slash" == *:* ]]; then
            tag="${after_last_slash##*:}"
            name="${image%:*}"
        fi
    fi

    # Validate tag if extracted
    if [[ -n "$tag" ]]; then
        if [[ ! "$tag" =~ ^[a-zA-Z0-9._-]+$ ]]; then
            validation_error "Docker image tag contains invalid characters"
            return 1
        fi
    fi

    # Validate image name characters (registry/namespace/name, may include port numbers)
    if [[ ! "$name" =~ ^[a-zA-Z0-9][a-zA-Z0-9._/:-]*$ ]]; then
        validation_error "Docker image name contains invalid characters"
        return 1
    fi

    # Strict mode: check against allowlist
    if [[ "$strict" == true ]]; then
        local found=false
        for prefix in "${SAFE_CONTAINER_PREFIXES[@]}"; do
            if [[ "$image" =~ ^"$prefix" ]]; then
                found=true
                break
            fi
        done

        if [[ "$found" == false ]]; then
            validation_error "Docker image not in allowed registry list (strict mode enabled)"
            return 1
        fi
    fi

    return 0
}

# Validates commit SHA format
# Accepts full (40) or short (7-39) SHA
validate_commit_sha() {
    local sha="$1"

    # Check empty
    if [[ -z "$sha" ]]; then
        validation_error "Commit SHA cannot be empty"
        return 1
    fi

    # Check max length
    if [[ ${#sha} -gt $MAX_COMMIT_SHA_LENGTH ]]; then
        validation_error "Commit SHA exceeds maximum length of $MAX_COMMIT_SHA_LENGTH characters"
        return 1
    fi

    # Check minimum length (short SHA must be at least 7)
    if [[ ${#sha} -lt 7 ]]; then
        validation_error "Commit SHA must be at least 7 characters"
        return 1
    fi

    # Check hex only
    if [[ ! "$sha" =~ ^[a-fA-F0-9]+$ ]]; then
        validation_error "Commit SHA must contain only hexadecimal characters (0-9, a-f, A-F)"
        return 1
    fi

    return 0
}

# Validates semantic version strings
# Format: MAJOR.MINOR.PATCH[-prerelease][+build]
# MAJOR, MINOR, PATCH are numeric without leading zeros
validate_semver() {
    local version="$1"

    # Check empty
    if [[ -z "$version" ]]; then
        validation_error "Semantic version cannot be empty"
        return 1
    fi

    # Remove leading 'v' if present (loose mode)
    if [[ "$version" =~ ^v ]]; then
        version="${version#v}"
    fi

    # Core semver pattern (without build metadata for simplicity)
    # MAJOR.MINOR.PATCH
    local core_pattern='^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)'

    # Check core version
    if [[ ! "$version" =~ $core_pattern ]]; then
        validation_error "Invalid semantic version format (expected: MAJOR.MINOR.PATCH)"
        return 1
    fi

    # Extract remainder after core MAJOR.MINOR.PATCH
    local remainder="${version#${BASH_REMATCH[0]}}"

    # Remainder must be empty, start with '-' (prerelease), or '+' (build metadata)
    if [[ -n "$remainder" && ! "$remainder" =~ ^[-+] ]]; then
        validation_error "Invalid trailing content after version core: $remainder"
        return 1
    fi

    # Validate prerelease if present (after -)
    if [[ "$remainder" =~ ^- ]]; then
        local prerelease="${remainder#-}"
        prerelease="${prerelease%%+*}"

        if [[ -z "$prerelease" ]]; then
            validation_error "Empty prerelease identifier"
            return 1
        fi

        local IFS='.'
        read -ra identifiers <<< "$prerelease"

        for id in "${identifiers[@]}"; do
            if [[ ! "$id" =~ ^[a-zA-Z0-9-]+$ ]]; then
                validation_error "Invalid prerelease identifier: $id"
                return 1
            fi

            if [[ "$id" =~ ^[0-9]+$ && "$id" != "0" && "$id" =~ ^0 ]]; then
                validation_error "Numeric prerelease identifier cannot have leading zeros: $id"
                return 1
            fi
        done
    fi

    return 0
}

# Validates job names (for GitHub Actions job identifiers)
# Must be valid for use in shell variables and paths
validate_job_name() {
    local name="$1"

    # Check empty
    if [[ -z "$name" ]]; then
        validation_error "Job name cannot be empty"
        return 1
    fi

    # Check max length
    if [[ ${#name} -gt 100 ]]; then
        validation_error "Job name exceeds maximum length of 100 characters"
        return 1
    fi

    # Valid characters: alphanumerics, underscores, hyphens, spaces
    # We'll be more restrictive and not allow spaces for safety
    if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        validation_error "Job name contains invalid characters (only alphanumerics, underscores, and hyphens allowed)"
        return 1
    fi

    return 0
}

# ============================================
# Sanitization Functions
# ============================================

# Sanitizes a path by resolving it and ensuring it's safe
# Returns sanitized path on stdout, or empty on failure
# $1: path to sanitize
# $2: optional base directory for containment check
sanitize_path() {
    local path="$1"
    local base_dir="${2:-}"
    local sanitized=""

    # Basic validation first
    if ! validate_path "$path" --allow-relative; then
        return 1
    fi

    # Try to get absolute path (GNU realpath -m, then BSD realpath, then manual)
    if command -v realpath &>/dev/null; then
        sanitized=$(realpath -m "$path" 2>/dev/null) || \
            sanitized=$(realpath "$path" 2>/dev/null) || {
            # Fallback: manual resolution if realpath can't resolve
            sanitized="${path//\/\//\/}"
            if [[ ! "$sanitized" =~ ^/ ]]; then
                sanitized="${PWD}/${sanitized}"
            fi
        }
    else
        sanitized="${path//\/\//\/}"
        if [[ ! "$sanitized" =~ ^/ ]]; then
            sanitized="${PWD}/${sanitized}"
        fi
    fi

    # Check against base directory if specified
    if [[ -n "$base_dir" ]]; then
        local base_abs
        if command -v realpath &>/dev/null; then
            base_abs=$(realpath -m "$base_dir" 2>/dev/null) || \
                base_abs=$(realpath "$base_dir" 2>/dev/null) || \
                base_abs="$base_dir"
        else
            base_abs="$base_dir"
        fi

        # Ensure base ends with slash
        [[ "$base_abs" =~ /$ ]] || base_abs="${base_abs}/"

        if [[ ! "$sanitized" =~ ^"$base_abs" ]]; then
            validation_error "Sanitized path is outside base directory"
            return 1
        fi
    fi

    echo "$sanitized"
}

# Sanitizes a string for safe use as a shell argument
# Adds proper quoting to prevent word splitting and injection
# Returns sanitized string on stdout
sanitize_shell_arg() {
    local arg="$1"

    # Check for single quotes in the argument
    if [[ "$arg" == *"'"* ]]; then
        # Contains single quotes, use double quotes with escaping
        # Escape: $ ` \ and ! (in some shells)
        local escaped="${arg//\\/\\\\}"
        escaped="${escaped//\$/\\$}"
        escaped="${escaped//\`/\\\`}"
        escaped="${escaped//\"/\\\"}"
        echo "\"$escaped\""
    else
        # Safe to use single quotes (stronger protection)
        echo "'$arg'"
    fi
}

# Escapes a string for safe use inside double quotes.
# Escapes: backslash, dollar sign, backtick, and double quote.
# Returns escaped string on stdout.
escape_quotes() {
    local str="$1"
    str="${str//\\/\\\\}"
    str="${str//\$/\\$}"
    str="${str//\`/\\\`}"
    str="${str//\"/\\\"}"
    printf '%s\n' "$str"
}

# Sanitizes a string for use in sed replacement
# Escapes special characters used by sed
sanitize_sed_replacement() {
    local str="$1"
    # Escape: & \ /
    str="${str//\\/\\\\}"
    str="${str//\&/\\&}"
    str="${str//\//\\/}"
    echo "$str"
}

# ============================================
# Safe Execution Helpers
# ============================================

# Creates and executes a script file from command string
# This is safer than bash -c because it avoids shell parsing issues
# $1: command string or script content
# $2: optional path for script file (default: temp file)
# $3: optional working directory
# Returns exit code of the executed command
safe_exec_script() {
    local cmd="$1"
    local script_file="${2:-}"
    local work_dir="${3:-}"
    local cleanup=false

    # Create temp file if no path specified
    if [[ -z "$script_file" ]]; then
        script_file=$(mktemp /tmp/safe_exec.XXXXXX.sh)
        cleanup=true
    fi

    # Validate the script path
    if ! validate_path "$script_file"; then
        validation_error "Invalid script file path: $script_file"
        return 1
    fi

    # Ensure script file has .sh extension for safety
    if [[ ! "$script_file" =~ \.sh$ ]]; then
        validation_error "Script file must have .sh extension"
        return 1
    fi

    # Write script content
    cat > "$script_file" << 'SAFEEXEC_EOF'
#!/usr/bin/env bash
set -euo pipefail
SAFEEXEC_EOF

    printf '%s\n' "$cmd" >> "$script_file"

    # Set executable
    chmod +x "$script_file"

    # Change directory if specified
    if [[ -n "$work_dir" ]]; then
        if ! validate_path "$work_dir"; then
            validation_error "Invalid working directory: $work_dir"
            [[ "$cleanup" == true ]] && rm -f "$script_file"
            return 1
        fi
        pushd "$work_dir" > /dev/null || return 1
    fi

    # Execute the script
    local exit_code=0
    "$script_file" || exit_code=$?

    # Restore directory
    if [[ -n "$work_dir" ]]; then
        popd > /dev/null || true
    fi

    # Clean up temp file only (never delete user-provided script files)
    if [[ "$cleanup" == true ]]; then
        rm -f "$script_file"
    fi

    return $exit_code
}

# Safely executes commands in Docker containers
# Validates inputs before passing to docker
# $1: image name
# $2: command to execute
# $3+: additional docker run options
safe_docker_exec() {
    local image="$1"
    local cmd="$2"
    shift 2
    local docker_opts=("$@")

    # Validate image
    if ! validate_docker_image "$image"; then
        return 1
    fi

    # Blocklist of dangerous Docker flags that could escape container isolation
    local -a _dangerous_docker_patterns=(
        "--privileged"
        "--pid=host"
        "--network=host"
        "--cap-add"
        "--security-opt"
        "--device"
        "--userns=host"
        "--ipc=host"
    )

    # Validate docker options against blocklist
    for opt in "${docker_opts[@]}"; do
        for pattern in "${_dangerous_docker_patterns[@]}"; do
            if [[ "$opt" == "$pattern" || "$opt" == "$pattern"=* ]]; then
                validation_error "Dangerous Docker option blocked: $opt"
                return 1
            fi
        done
        # Block volume mounts of host root or sensitive directories
        if [[ "$opt" =~ ^-v$ ]] || [[ "$opt" == --volume=* ]]; then
            local mount_src="${opt#*=}"
            if [[ "$mount_src" == "/" || "$mount_src" == /etc* || "$mount_src" == /var/run/docker* ]]; then
                validation_error "Dangerous Docker volume mount blocked: $opt"
                return 1
            fi
        fi
    done

    # Sanitize command for logging
    local safe_cmd
    safe_cmd=$(sanitize_shell_arg "$cmd")

    # Build docker command array to avoid shell injection
    local docker_cmd=("docker" "run" "--rm")

    # Add user options
    for opt in "${docker_opts[@]}"; do
        docker_cmd+=("$opt")
    done

    docker_cmd+=("$image")

    # Use safe_exec_script approach for the command
    local script_file
    script_file=$(mktemp /tmp/docker_exec.XXXXXX.sh)

    echo "#!/usr/bin/env bash" > "$script_file"
    echo "set -euo pipefail" >> "$script_file"
    echo "$cmd" >> "$script_file"
    chmod +x "$script_file"

    docker_cmd+=("/bin/bash" "-c" "$(cat "$script_file")")

    # Cleanup temp file
    rm -f "$script_file"

    # Execute docker command
    "${docker_cmd[@]}"
}

# Safely executes commands via SSH
# Validates hostname and other parameters
# $1: hostname
# $2: command to execute
# $3: optional username
# $4+: additional ssh options
safe_ssh_exec() {
    local hostname="$1"
    local cmd="$2"
    local username="${3:-}"
    shift 3
    local ssh_opts=("$@")

    # Validate hostname
    if ! validate_hostname "$hostname"; then
        return 1
    fi

    # Validate username if provided
    if [[ -n "$username" ]]; then
        if ! validate_username "$username"; then
            return 1
        fi
    fi

    # Blocklist SSH options that enable arbitrary command execution
    local -a _dangerous_ssh_patterns=(
        "ProxyCommand"
        "LocalCommand"
        "PermitLocalCommand"
        "ProxyUseFdpass"
        "LocalForward"
        "RemoteForward"
        "DynamicForward"
    )
    for opt in "${ssh_opts[@]}"; do
        for pattern in "${_dangerous_ssh_patterns[@]}"; do
            if [[ "$opt" == *"$pattern"* ]]; then
                validation_error "Dangerous SSH option blocked: $opt"
                return 1
            fi
        done
    done

    # Build ssh command
    local ssh_cmd=("ssh")

    # Add safety options by default
    ssh_cmd+=("-o" "StrictHostKeyChecking=yes")
    ssh_cmd+=("-o" "BatchMode=yes")

    # Add user options
    for opt in "${ssh_opts[@]}"; do
        ssh_cmd+=("$opt")
    done

    # Add target
    if [[ -n "$username" ]]; then
        ssh_cmd+=("${username}@${hostname}")
    else
        ssh_cmd+=("$hostname")
    fi

    # Sanitize and add command
    local safe_cmd
    safe_cmd=$(sanitize_shell_arg "$cmd")
    ssh_cmd+=("$safe_cmd")

    # Execute
    "${ssh_cmd[@]}"
}

# ============================================
# Output / Error Helpers
# ============================================

# Sanitizes a string for safe use in GitHub Actions workflow commands.
# Strips newlines/CR to prevent command injection and escapes :: delimiters.
_sanitize_workflow_cmd_param() {
    local str="$1"
    str="${str//$'\n'/ }"
    str="${str//$'\r'/}"
    str="${str//::/ }"
    printf '%s' "$str"
}

# Outputs validation error in GitHub Actions format
# $1: error message
# $2: optional file reference
# $3: optional line number
validation_error() {
    local msg
    msg=$(_sanitize_workflow_cmd_param "$1")
    local file="${2:-}"
    local line="${3:-}"

    if [[ -n "$file" ]]; then
        file=$(_sanitize_workflow_cmd_param "$file")
        if [[ -n "$line" ]]; then
            line=$(_sanitize_workflow_cmd_param "$line")
            echo "::error file=$file,line=$line::$msg" >&2
        else
            echo "::error file=$file::$msg" >&2
        fi
    else
        echo "::error::$msg" >&2
    fi
}

# Outputs validation warning in GitHub Actions format
# $1: warning message
# $2: optional file reference
# $3: optional line number
validation_warning() {
    local msg
    msg=$(_sanitize_workflow_cmd_param "$1")
    local file="${2:-}"
    local line="${3:-}"

    if [[ -n "$file" ]]; then
        file=$(_sanitize_workflow_cmd_param "$file")
        if [[ -n "$line" ]]; then
            line=$(_sanitize_workflow_cmd_param "$line")
            echo "::warning file=$file,line=$line::$msg" >&2
        else
            echo "::warning file=$file::$msg" >&2
        fi
    else
        echo "::warning::$msg" >&2
    fi
}

# Outputs validation notice in GitHub Actions format
# $1: notice message
# $2: optional file reference
# $3: optional line number
validation_notice() {
    local msg
    msg=$(_sanitize_workflow_cmd_param "$1")
    local file="${2:-}"
    local line="${3:-}"

    if [[ -n "$file" ]]; then
        file=$(_sanitize_workflow_cmd_param "$file")
        if [[ -n "$line" ]]; then
            line=$(_sanitize_workflow_cmd_param "$line")
            echo "::notice file=$file,line=$line::$msg"
        else
            echo "::notice file=$file::$msg"
        fi
    else
        echo "::notice::$msg"
    fi
}

# Safely sets GitHub Actions output
# Handles proper escaping for multi-line values
# $1: output name
# $2: output value
set_github_output() {
    local name="$1"
    local value="$2"

    # Validate output name
    if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        validation_error "Invalid output name: $name"
        return 1
    fi

    # Use environment file if available (GitHub Actions recommended way)
    if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
        # Handle multi-line values or values with \r using delimiter
        if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
            local delimiter="OUTPUT_$(date +%s)_$(head -c 16 /dev/urandom | xxd -p | head -c 16)"
            {
                echo "${name}<<${delimiter}"
                echo "$value"
                echo "${delimiter}"
            } >> "$GITHUB_OUTPUT"
        else
            echo "${name}=${value}" >> "$GITHUB_OUTPUT"
        fi
    else
        # Fallback: use deprecated ::set-output command
        # Strip newlines and CR to prevent workflow command injection
        local safe_value="${value//$'\n'/}"
        safe_value="${safe_value//$'\r'/}"
        echo "::set-output name=${name}::${safe_value}"
    fi
}

# Safely sets a GitHub Actions environment variable
# $1: variable name
# $2: variable value
set_github_env() {
    local name="$1"
    local value="$2"

    # Validate variable name (shell variable naming rules)
    if [[ ! "$name" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
        validation_error "Invalid environment variable name: $name"
        return 1
    fi

    # Warn when setting security-sensitive environment variable names
    local -a _dangerous_env_names=(
        "PATH" "LD_PRELOAD" "LD_LIBRARY_PATH" "BASH_ENV" "ENV"
        "GITHUB_TOKEN" "ACTIONS_RUNTIME_TOKEN" "ACTIONS_ID_TOKEN_REQUEST_TOKEN"
    )
    for _dname in "${_dangerous_env_names[@]}"; do
        if [[ "$name" == "$_dname" ]]; then
            validation_warning "Setting security-sensitive environment variable: $name"
            break
        fi
    done

    # Use environment file if available
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        # Handle multi-line values or values with \r
        if [[ "$value" == *$'\n'* || "$value" == *$'\r'* ]]; then
            local delimiter="ENV_$(date +%s)_$(head -c 16 /dev/urandom | xxd -p | head -c 16)"
            {
                echo "${name}<<${delimiter}"
                echo "$value"
                echo "${delimiter}"
            } >> "$GITHUB_ENV"
        else
            echo "${name}=${value}" >> "$GITHUB_ENV"
        fi
    else
        # Export in current shell
        export "${name}=${value}"
    fi
}

# ============================================
# Validation Batch Helpers
# ============================================

# Validates multiple required variables exist and are non-empty
# $1+: variable names to check
validate_required() {
    local var_name
    local missing=()

    for var_name in "$@"; do
        local value="${!var_name:-}"
        if [[ -z "$value" ]]; then
            missing+=("$var_name")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        validation_error "Missing required variables: ${missing[*]}"
        return 1
    fi

    return 0
}

# Runs a validation function on each element of an array
# $1: validation function name (must be a declared function)
# $2+: array elements
validate_array() {
    local validator="$1"
    shift

    # Ensure the validator is actually a declared function (not an arbitrary command)
    if ! declare -f "$validator" > /dev/null 2>&1; then
        validation_error "Unknown validator function: $validator"
        return 1
    fi

    local item
    for item in "$@"; do
        if ! "$validator" "$item"; then
            validation_error "Array validation failed for item: $item"
            return 1
        fi
    done

    return 0
}
