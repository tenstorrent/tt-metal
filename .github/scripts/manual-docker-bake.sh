#!/bin/bash
# Implementation of the "Manual Docker Bake" composite action
# (.github/actions/manual-docker-bake/action.yml). The action's run: step is a
# thin wrapper that exports the same INPUT_* environment variables it always has
# and invokes this script; all logic lives here so it stays reviewable.
#
# Runs `docker buildx bake` with shared options: builder selection, attestation
# --set construction (provenance + SBOM), retry logic, a Harbor pull-through
# GHCR-direct fallback, per-target validation (registry vs local), and verbose
# best-effort logging (real content sizes, SLSA v1.0 provenance summary, SBOM
# summary) plus optional SBOM enrichment from OCI labels via enrich-sbom.sh.
#
# Configured entirely through INPUT_* environment variables (see the action's
# inputs: block for the authoritative descriptions):
#   INPUT_BAKE_FILE, INPUT_TARGETS, INPUT_SET_LINES, INPUT_UBUNTU_VERSION,
#   INPUT_PYTHON_VERSION, INPUT_RETRIES, INPUT_RETRY_DELAY_SECONDS,
#   INPUT_USE_DEFAULT_BUILDER, INPUT_VALIDATE_IMAGES, INPUT_ENABLE_ATTESTATIONS,
#   INPUT_ENRICH_SBOM, INPUT_HARBOR_PREFIX
#
# Requires: docker (buildx), jq, numfmt; oras + syft are installed on demand for
# SBOM enrichment. Registry auth is taken from the ambient docker credential
# store (callers docker-login before invoking).

set -euo pipefail

# Directory containing this script - used to locate the sibling enrich-sbom.sh
# regardless of the caller's working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Install syft (https://github.com/anchore/syft, Apache-2.0 license) + oras
# once per job when SBOM enrichment is requested (see enrich-sbom input).
# Caller-settable via env vars (e.g. to bump a version), defaulting to the
# current pinned values below - do not change the defaults to float to
# latest. Checksums verified against each project's published
# *_checksums.txt release asset; if you override the version, override the
# matching sha256 too.
SYFT_VERSION="${SYFT_VERSION:-v1.46.0}"
SYFT_SHA256="${SYFT_SHA256:-d654f678b709eb53c393d38519d5ed7d2e57205529404018614cfefa0fb2b5ca}"
ORAS_VERSION="${ORAS_VERSION:-1.3.3}"
ORAS_SHA256="${ORAS_SHA256:-9ce999f8d2de03fc03968b29d743077a58783e545e5eaa53917ca177352d0e59}"

# Populated by parse_inputs; referenced by later functions.
CLEAN_TARGETS=()
SET_LINES=()
BAKE_SETS=()

# --- select the buildx default builder if the caller asked for it ---
resolve_builder() {
  if [ "${INPUT_USE_DEFAULT_BUILDER}" = "true" ]; then
    docker buildx use default || true
  fi
}

# --- parse inputs: env passthrough, target list, --set overrides ---
parse_inputs() {
  if [ -n "${INPUT_UBUNTU_VERSION}" ]; then
    export UBUNTU_VERSION="${INPUT_UBUNTU_VERSION}"
  fi
  if [ -n "${INPUT_PYTHON_VERSION}" ]; then
    export PYTHON_VERSION="${INPUT_PYTHON_VERSION}"
  fi

  mapfile -t SET_LINES < <(printf '%s\n' "${INPUT_SET_LINES}")

  local TARGETS trimmed t
  IFS=',' read -r -a TARGETS <<< "${INPUT_TARGETS}"
  CLEAN_TARGETS=()
  for t in "${TARGETS[@]}"; do
    trimmed="${t#"${t%%[![:space:]]*}"}"
    trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
    [ -n "$trimmed" ] && CLEAN_TARGETS+=("$trimmed")
  done
  if [ "${#CLEAN_TARGETS[@]}" -eq 0 ]; then
    echo "No bake targets provided."
    exit 1
  fi
}

# --- append attestation --set entries for each target to the named array ---
# Usage: append_attestation_sets <array-name>
# Emits the provenance/SBOM attest overrides (or the disabled variants) so both
# the primary bake and the Harbor fallback share identical attestation shaping.
append_attestation_sets() {
  local -n _sets="$1"
  local t
  if [ "${INPUT_ENABLE_ATTESTATIONS}" = "true" ]; then
    for t in "${CLEAN_TARGETS[@]}"; do
      _sets+=(--set "${t}.attest=type=provenance,mode=max")
      _sets+=(--set "${t}.attest=type=sbom")
    done
  else
    for t in "${CLEAN_TARGETS[@]}"; do
      _sets+=(--set "${t}.attest=type=provenance,disabled=true")
      _sets+=(--set "${t}.attest=type=sbom,disabled=true")
    done
  fi
}

# --- build the primary --set array from set-lines + attestation overrides ---
build_bake_sets() {
  local s
  BAKE_SETS=()
  for s in "${SET_LINES[@]}"; do
    [ -n "$s" ] && BAKE_SETS+=(--set "$s")
  done
  append_attestation_sets BAKE_SETS
}

# --- download a release tarball, verify its sha256, install one named binary ---
# No black-box install scripts: pull the tarball directly and verify it against
# a pinned checksum before extracting anything from it.
# Usage: install_pinned_binary <url> <expected_sha256> <binary_name> <dest_dir>
install_pinned_binary() {
  local url="$1" expected_sha256="$2" binary_name="$3" dest_dir="$4"
  local tmp_dir tarball actual_sha256
  tmp_dir="$(mktemp -d)"
  tarball="${tmp_dir}/${binary_name}.tar.gz"

  curl -sSfL "$url" -o "$tarball"
  actual_sha256="$(sha256sum "$tarball" | cut -d' ' -f1)"
  if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "ERROR: sha256 mismatch for ${binary_name} tarball from ${url}" >&2
    echo "  expected: ${expected_sha256}" >&2
    echo "  actual:   ${actual_sha256}" >&2
    rm -rf "$tmp_dir"
    exit 1
  fi

  tar -xzf "$tarball" -C "$tmp_dir" "$binary_name"
  install "${tmp_dir}/${binary_name}" "${dest_dir}/${binary_name}"
  rm -rf "$tmp_dir"
}

# --- install syft + oras into a writable temp dir when enrichment is on ---
install_bake_tools() {
  if [ "${INPUT_ENRICH_SBOM}" = "true" ] && [ "${INPUT_ENABLE_ATTESTATIONS}" = "true" ]; then
    # /usr/local/bin is not guaranteed to be writable without sudo on every
    # runner (observed: "install: cannot create regular file
    # '/usr/local/bin/syft': Permission denied" on a real run). Install into a
    # directory we know is writable and put it on PATH for the rest of this
    # script instead of assuming elevated privileges.
    local TOOL_BIN_DIR="${RUNNER_TEMP:-/tmp}/manual-docker-bake-bin"
    mkdir -p "$TOOL_BIN_DIR"
    export PATH="${TOOL_BIN_DIR}:${PATH}"

    if ! command -v syft >/dev/null 2>&1; then
      echo "Installing syft ${SYFT_VERSION} (sha256-verified)..."
      install_pinned_binary \
        "https://github.com/anchore/syft/releases/download/${SYFT_VERSION}/syft_${SYFT_VERSION#v}_linux_amd64.tar.gz" \
        "$SYFT_SHA256" "syft" "$TOOL_BIN_DIR"
    fi
    if ! command -v oras >/dev/null 2>&1; then
      echo "Installing oras v${ORAS_VERSION} (sha256-verified)..."
      install_pinned_binary \
        "https://github.com/oras-project/oras/releases/download/v${ORAS_VERSION}/oras_${ORAS_VERSION}_linux_amd64.tar.gz" \
        "$ORAS_SHA256" "oras" "$TOOL_BIN_DIR"
    fi
  fi
}

# --- run the primary bake with retries, then Harbor GHCR-direct fallback ---
# Exits non-zero if the build cannot be completed.
run_bake_with_retries() {
  local retries="${INPUT_RETRIES}"
  local delay="${INPUT_RETRY_DELAY_SECONDS}"
  local attempt=1
  local bake_success=false
  local last_bake_exit=0
  until [ "$attempt" -gt "$retries" ]; do
    echo "docker buildx bake targets=${CLEAN_TARGETS[*]} attempt ${attempt}/${retries}"
    docker buildx bake -f "${INPUT_BAKE_FILE}" "${BAKE_SETS[@]}" "${CLEAN_TARGETS[@]}" \
      && last_bake_exit=0 || last_bake_exit=$?
    if [ "$last_bake_exit" -eq 0 ]; then
      bake_success=true
      break
    fi
    if [ "$attempt" -lt "$retries" ]; then
      echo "Bake failed, retrying in ${delay}s..."
      sleep "$delay"
    fi
    attempt=$((attempt + 1))
  done

  if [ "$bake_success" != "true" ]; then
    if [ -n "${INPUT_HARBOR_PREFIX}" ]; then
      echo "WARNING: Bake failed after ${retries} attempt(s) with Harbor prefix." \
        "Retrying without Harbor prefix (GHCR direct fallback)..."
      local FALLBACK_SETS=()
      local s stripped
      for s in "${SET_LINES[@]}"; do
        if [ -n "$s" ]; then
          # Strip harbor prefix from docker-image:// context URIs so bake pulls from GHCR directly
          stripped="${s//"docker-image://${INPUT_HARBOR_PREFIX}"/"docker-image://"}"
          FALLBACK_SETS+=(--set "$stripped")
        fi
      done
      append_attestation_sets FALLBACK_SETS
      docker buildx bake -f "${INPUT_BAKE_FILE}" "${FALLBACK_SETS[@]}" "${CLEAN_TARGETS[@]}" \
        && bake_success=true || last_bake_exit=$?
      if [ "$bake_success" = "true" ]; then
        echo "Bake succeeded after GHCR direct fallback."
      else
        echo "ERROR: Bake failed even without Harbor prefix (exit code: ${last_bake_exit})."
        exit 1
      fi
    else
      echo "ERROR: Bake failed after ${retries} attempts (last exit code: ${last_bake_exit})."
      exit 1
    fi
  fi
}

# --- verbose: print real content sizes (sum of layer sizes, not manifest-document
# byte count) for a registry image. Best-effort/informational; never fails. ---
# Usage: log_manifest_sizes <target_tag>
log_manifest_sizes() {
  local target_tag="$1"
  local IMAGETOOLS_JSON image_repo entry child_digest child_platform child_kind
  local child_manifest content_bytes layer_count content_human
  IMAGETOOLS_JSON=$(docker buildx imagetools inspect "$target_tag" --format '{{json .}}' 2>/dev/null || echo "")
  if [ -n "$IMAGETOOLS_JSON" ]; then
    echo "  Manifest: $(echo "$IMAGETOOLS_JSON" | jq -r '.manifest.mediaType // "unknown"') digest=$(echo "$IMAGETOOLS_JSON" | jq -r '.manifest.digest // "?"')"
    image_repo="${target_tag%:*}"
    if echo "$IMAGETOOLS_JSON" | jq -e '.manifest.manifests' > /dev/null 2>&1; then
      while IFS= read -r entry; do
        child_digest=$(echo "$entry" | jq -r '.digest')
        child_platform=$(echo "$entry" | jq -r '(.platform.architecture // "?") + "/" + (.platform.os // "?")')
        child_kind=$(echo "$entry" | jq -r '.annotations["vnd.docker.reference.type"] // ""')
        child_manifest=$(docker manifest inspect "${image_repo}@${child_digest}" 2>/dev/null || echo "")
        if [ -n "$child_manifest" ]; then
          content_bytes=$(echo "$child_manifest" | jq -r '[.layers[]?.size] | add // 0')
          layer_count=$(echo "$child_manifest" | jq -r '.layers | length')
          content_human=$(numfmt --to=iec --suffix=B "$content_bytes" 2>/dev/null || echo "${content_bytes} bytes")
          if [ -n "$child_kind" ]; then
            echo "    - ${child_platform} [${child_kind}]: ${layer_count} layer(s), ${content_human}"
          else
            echo "    - ${child_platform}: ${layer_count} layer(s), real content size = ${content_human}"
          fi
        else
          echo "    - ${child_platform}: (could not fetch child manifest ${child_digest})"
        fi
      done < <(echo "$IMAGETOOLS_JSON" | jq -c '.manifest.manifests[]')
    else
      content_bytes=$(echo "$IMAGETOOLS_JSON" | jq -r '[.manifest.layers[]?.size] | add // 0')
      layer_count=$(echo "$IMAGETOOLS_JSON" | jq -r '.manifest.layers | length')
      content_human=$(numfmt --to=iec --suffix=B "$content_bytes" 2>/dev/null || echo "${content_bytes} bytes")
      echo "    (single manifest, ${layer_count} layer(s), real content size = ${content_human})"
    fi
  else
    echo "  WARNING: docker buildx imagetools inspect failed for $target_tag - skipping size/attestation details"
  fi
}

# --- verbose: summarize the SLSA v1.0 provenance attestation. Best-effort. ---
# Usage: log_provenance_summary <target_tag>
log_provenance_summary() {
  local target_tag="$1"
  local PROVENANCE_JSON
  PROVENANCE_JSON=$(docker buildx imagetools inspect "$target_tag" --format '{{json .Provenance}}' 2>/dev/null || echo "")
  if [ -n "$PROVENANCE_JSON" ] && [ "$PROVENANCE_JSON" != "null" ]; then
    echo "  Provenance (SLSA v1.0 predicate: https://slsa.dev/provenance/v1):"
    echo "$PROVENANCE_JSON" | jq -r '
      .SLSA as $p |
      "    buildType: " + ($p.buildDefinition.buildType // "unknown"),
      "    builder.id: " + (if ($p.runDetails.builder.id // "") == "" then "(not set by BuildKit)" else $p.runDetails.builder.id end),
      "    invocation: " + ($p.runDetails.metadata.startedOn // "?") + " -> " + ($p.runDetails.metadata.finishedOn // "?"),
      (if $p.runDetails.metadata.buildkit_metadata.vcs then
        "    vcs: " + ($p.runDetails.metadata.buildkit_metadata.vcs.source // "?") + "@" + ($p.runDetails.metadata.buildkit_metadata.vcs.revision // "?")
      else empty end),
      "    resolvedDependencies (" + (($p.buildDefinition.resolvedDependencies // []) | length | tostring) + "):",
      (($p.buildDefinition.resolvedDependencies // [])[] | "      - " + .uri)
    ' 2>/dev/null || echo "    (could not parse provenance JSON - see raw with 'docker buildx imagetools inspect ${target_tag} --format \"{{json .Provenance}}\"')"
  else
    echo "  WARNING: no provenance attestation found for $target_tag"
  fi
}

# --- verbose: summarize the SBOM attestation (package count + byte size). ---
# Usage: log_sbom_summary <target_tag>
log_sbom_summary() {
  local target_tag="$1"
  local SBOM_JSON SBOM_PKG_COUNT SBOM_BYTES
  SBOM_JSON=$(docker buildx imagetools inspect "$target_tag" --format '{{json .SBOM}}' 2>/dev/null || echo "")
  if [ -n "$SBOM_JSON" ] && [ "$SBOM_JSON" != "null" ]; then
    SBOM_PKG_COUNT=$(echo "$SBOM_JSON" | jq -r '.SPDX.packages // [] | length' 2>/dev/null || echo "?")
    SBOM_BYTES=$(echo -n "$SBOM_JSON" | wc -c)
    echo "  SBOM (SPDX): ${SBOM_PKG_COUNT} package(s) cataloged, ${SBOM_BYTES} bytes" \
      "(full document not printed here - a real SBOM can run to hundreds of KB; fetch with" \
      "'docker buildx imagetools inspect ${target_tag} --format \"{{json .SBOM}}\"')"
  else
    echo "  WARNING: no SBOM attestation found for $target_tag"
  fi
}

# --- replace BuildKit's NOASSERTION SBOM with one enriched from OCI labels ---
# A target with no attestation-manifest is a safe no-op (handled inside the
# script); a real failure hard-fails the job. Re-logs the enriched describing
# package afterward.
# Usage: enrich_target_sbom <target_tag>
enrich_target_sbom() {
  local target_tag="$1"
  local ENRICHED_SBOM_JSON
  if [ "${INPUT_ENRICH_SBOM}" = "true" ] && [ "${INPUT_ENABLE_ATTESTATIONS}" = "true" ]; then
    echo "  Enriching SBOM attestation for $target_tag from OCI labels..."
    "${SCRIPT_DIR}/enrich-sbom.sh" "$target_tag"
    ENRICHED_SBOM_JSON=$(docker buildx imagetools inspect "$target_tag" --format '{{json .SBOM}}' 2>/dev/null || echo "")
    if [ -n "$ENRICHED_SBOM_JSON" ] && [ "$ENRICHED_SBOM_JSON" != "null" ]; then
      echo "  SBOM attestation enriched - describing package now reflects OCI labels:"
      # Resolve the actual describing/root package the same way enrich-sbom.sh
      # does (documentDescribes[0], falling back to a DocumentRoot-matching
      # SPDXID) rather than blindly taking packages[0] - for a real image with
      # many detected packages (e.g. a Python venv full of pip packages),
      # packages[0] is just whichever unrelated package syft listed first, not
      # the one we patched. (Bug found by Neil Sexton on a real ci-test venv
      # image: packages[0] was "ipywidgets"/BSD-3-Clause/Project Jupyter, not
      # the image's own Apache-2.0/tenstorrent describing package.)
      echo "$ENRICHED_SBOM_JSON" | jq -r '
        .SPDX as $s |
        (
          ($s.documentDescribes[0]? // null)
          // ([$s.packages[]? | select(.SPDXID | test("DocumentRoot"))][0].SPDXID // null)
          // null
        ) as $root_id |
        ([$s.packages[]? | select(.SPDXID == $root_id)][0] // {}) as $root |
        "    documentNamespace: " + ($s.documentNamespace // "?"),
        "    licenseDeclared: " + ($root.licenseDeclared // "NOASSERTION"),
        "    downloadLocation: " + ($root.downloadLocation // "NOASSERTION"),
        "    supplier: " + ($root.supplier // "NOASSERTION")
      ' 2>/dev/null || echo "    (could not parse enriched SBOM JSON)"
    else
      echo "  WARNING: could not re-fetch SBOM after enrichment for $target_tag"
    fi
  fi
}

# --- validate a single built target (registry manifest vs local image) ---
# Sets the global VALIDATION_FAILED=true on validation failure.
# Usage: validate_target <target>
validate_target() {
  local target="$1"
  local target_tag="" target_output="" s local_tag local_size
  for s in "${SET_LINES[@]}"; do
    # Extract tag override: target.tags=xxx
    if [[ "$s" =~ ^${target}\.tags=(.+)$ ]]; then
      target_tag="${BASH_REMATCH[1]}"
    fi
    # Extract output override: target.output=xxx
    if [[ "$s" =~ ^${target}\.output=(.+)$ ]]; then
      target_output="${BASH_REMATCH[1]}"
    fi
  done

  # Determine how to validate based on output configuration
  # If output contains 'push=true' without 'type=docker', image is registry-only
  if [[ "$target_output" =~ push=true ]] && [[ ! "$target_output" =~ type=docker ]]; then
    # Registry-only image - validate by checking manifest
    if [ -n "$target_tag" ]; then
      echo "Checking registry image for target '$target' (tag: $target_tag)..."
      if ! docker manifest inspect "$target_tag" > /dev/null 2>&1; then
        echo "ERROR: Registry image validation failed for target '$target' (tag: $target_tag)"
        echo "The build reported success but the image is not available in the registry."
        VALIDATION_FAILED=true
      else
        echo "  Registry image validated: $target_tag"

        # Verbose best-effort details (never fail validation).
        log_manifest_sizes "$target_tag"
        log_provenance_summary "$target_tag"
        log_sbom_summary "$target_tag"
        enrich_target_sbom "$target_tag"
      fi
    else
      echo "  Skipping validation for target '$target' (no tag specified for registry check)"
    fi
  else
    # Local image validation
    local_tag="$target_tag"
    if [ -z "$local_tag" ]; then
      # Get the default image name from the bake file
      local_tag=$(docker buildx bake -f "${INPUT_BAKE_FILE}" --print "$target" 2>/dev/null \
        | jq -r --arg t "$target" '.target[$t].tags[0]' 2>/dev/null || echo "")
      if [ -z "$local_tag" ] || [ "$local_tag" = "null" ]; then
        local_tag="tt-metalium-${target}:local"
      fi
    fi
    echo "Checking local image for target '$target' (tag: $local_tag)..."
    if ! docker image inspect "$local_tag" > /dev/null 2>&1; then
      echo "ERROR: Image validation failed for target '$target' (tag: $local_tag)"
      echo "The build reported success but the image is not inspectable."
      VALIDATION_FAILED=true
    else
      local_size=$(docker image inspect "$local_tag" --format '{{.Size}}' 2>/dev/null || echo "?")
      echo "  Local image validated: $local_tag (size=${local_size} bytes)"
      echo "  (local docker-loaded images don't carry attestation manifests - no provenance/SBOM to report)"
    fi
  fi
}

# --- validate that images were actually built and are inspectable ---
validate_images() {
  if [ "${INPUT_VALIDATE_IMAGES}" != "true" ]; then
    return 0
  fi
  VALIDATION_FAILED=false
  echo ""
  echo "Validating built images..."
  local target
  for target in "${CLEAN_TARGETS[@]}"; do
    validate_target "$target"
  done

  if [ "$VALIDATION_FAILED" = "true" ]; then
    echo "ERROR: One or more image validations failed."
    exit 1
  fi
  echo "All images validated successfully."
}

main() {
  resolve_builder
  parse_inputs
  build_bake_sets
  install_bake_tools
  run_bake_with_retries
  validate_images
}

main "$@"
