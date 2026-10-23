#!/bin/bash
# Replace BuildKit's directory-scanned SBOM attestation with one enriched from the
# image's own OCI labels (org.opencontainers.image.source/description/licenses/version).
#
# WHY: BuildKit generates its SBOM attestation by mounting the exported image's
# rootfs as a plain DIRECTORY to the scanner (see moby/buildkit
# docs/attestations/sbom-protocol.md) - the scanner never receives the image
# Config/Labels, so the SBOM's describing package always has
# licenseDeclared/downloadLocation/supplier = NOASSERTION even when the correct
# values are present in the image's config Labels one manifest over. syft cannot
# auto-map OCI labels into SPDX fields either (anchore/syft#3098). This script
# rescans the pushed image from the registry with syft (seeded from the labels),
# jq-patches the describing package, and rebuilds the attestation-manifest in
# place - preserving BuildKit's exact OCI shape (see tenstorrent/tt-metal#49882).
#
# Usage: enrich-sbom.sh <image-ref>
#   e.g. enrich-sbom.sh ghcr.io/tenstorrent/tt-metal/tt-metalium/tools/sfpi:<tag>
#
# Requires: docker (buildx), oras, syft, jq. Registry auth is taken from the
# ambient docker credential store (callers docker-login before invoking).

set -euo pipefail

REF="${1:?Usage: $0 <image-ref>}"

log() { echo "[enrich-sbom] $*" >&2; }

REPO="${REF%:*}"          # ghcr.io/owner/.../sfpi
TAG="${REF##*:}"          # <tag>
REPO_NOSCHEME="$REF"      # full ref used in the PURL subject name (matches BuildKit)
REPO_NOSCHEME="${REPO_NOSCHEME%%@*}"
REPO_NOSCHEME="${REPO_NOSCHEME%:*}"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# --- (a) resolve the OCI index; find image-manifest + attestation-manifest ---
INDEX_JSON="$(docker buildx imagetools inspect "$REF" --raw)"
INDEX_MT="$(printf '%s' "$INDEX_JSON" | jq -r '.mediaType // ""')"

if ! printf '%s' "$INDEX_JSON" | jq -e '.manifests' >/dev/null 2>&1; then
  log "No manifest list / OCI index at $REF (mediaType=$INDEX_MT) - no attestation to enrich. Skipping."
  exit 0
fi

IMG_DIGEST="$(printf '%s' "$INDEX_JSON" | jq -r '
  [.manifests[] | select((.annotations["vnd.docker.reference.type"] // "") == "")][0].digest // ""')"
ATT_DIGEST="$(printf '%s' "$INDEX_JSON" | jq -r '
  [.manifests[] | select((.annotations["vnd.docker.reference.type"] // "") == "attestation-manifest")][0].digest // ""')"

if [ -z "$IMG_DIGEST" ]; then
  log "Could not identify the real image manifest in the index for $REF."
  exit 1
fi
if [ -z "$ATT_DIGEST" ]; then
  log "No attestation-manifest child in the index for $REF - nothing to enrich. Skipping."
  exit 0
fi
log "image manifest: $IMG_DIGEST"
log "attestation manifest: $ATT_DIGEST"

# --- (b) fetch the real OCI Labels from the image config ---
IMG_CONFIG_JSON="$(docker buildx imagetools inspect "${REPO}@${IMG_DIGEST}" --format '{{json .Image}}')"
# --format '{{json .Image}}' may be keyed by platform; normalize to a single config object.
if printf '%s' "$IMG_CONFIG_JSON" | jq -e 'has("config")' >/dev/null 2>&1; then
  LABELS_JSON="$(printf '%s' "$IMG_CONFIG_JSON" | jq -c '.config.Labels // {}')"
else
  LABELS_JSON="$(printf '%s' "$IMG_CONFIG_JSON" | jq -c '[.[].config.Labels // {}] | add // {}')"
fi
LABEL_SOURCE="$(printf '%s' "$LABELS_JSON" | jq -r '."org.opencontainers.image.source" // ""')"
LABEL_DESC="$(printf '%s' "$LABELS_JSON" | jq -r '."org.opencontainers.image.description" // ""')"
LABEL_LICENSES="$(printf '%s' "$LABELS_JSON" | jq -r '."org.opencontainers.image.licenses" // ""')"
LABEL_VERSION="$(printf '%s' "$LABELS_JSON" | jq -r '."org.opencontainers.image.version" // ""')"
log "labels: source='${LABEL_SOURCE}' licenses='${LABEL_LICENSES}' version='${LABEL_VERSION}'"

# --- (c) rescan the image from the registry with syft, seeded from labels ---
# file.metadata.selection defaults to "owned-by-package": syft only computes real
# file digests for files it can attribute to a detected package (dpkg/rpm/language
# manifests/etc). These tool images are raw binaries COPYed from `FROM scratch` with
# no package-manager metadata at all, so every file is "unowned" and syft falls back
# to a placeholder all-zero SHA1 checksum instead of a real one (a known syft
# behavior for unowned files - see anchore/syft#2307/#1226). Force "all" so real
# digests get computed for every file regardless of package ownership.
SYFT_SPDX="$WORKDIR/syft.spdx.json"
SYFT_ARGS=(scan "registry:${REF}" -o "spdx-json=${SYFT_SPDX}")
[ -n "$LABEL_VERSION" ] && SYFT_ARGS+=(--source-version "$LABEL_VERSION")
SYFT_FILE_METADATA_SELECTION=all syft "${SYFT_ARGS[@]}"

# --- (d) jq-patch ONLY the describing/root package ---
PATCHED_SPDX="$WORKDIR/spdx.patched.json"
ROOT_ID="$(jq -r '(.documentDescribes[0]) // ""' "$SYFT_SPDX")"
if [ -z "$ROOT_ID" ]; then
  # fall back to the DocumentRoot-* package syft always emits for a scan root
  ROOT_ID="$(jq -r '[.packages[] | select(.SPDXID | test("DocumentRoot"))][0].SPDXID // ""' "$SYFT_SPDX")"
fi
if [ -z "$ROOT_ID" ]; then
  log "Could not find a describing/root package in the syft SBOM; leaving packages unpatched."
  cp "$SYFT_SPDX" "$PATCHED_SPDX"
else
  log "patching describing package: $ROOT_ID"
  jq \
    --arg root "$ROOT_ID" \
    --arg lic "$LABEL_LICENSES" \
    --arg src "$LABEL_SOURCE" \
    --arg sup "Organization: Tenstorrent" '
    .packages |= map(
      if .SPDXID == $root then
        (if $lic != "" then .licenseDeclared = $lic else . end)
        | (if $src != "" then .downloadLocation = $src else . end)
        | .supplier = $sup
      else . end)
  ' "$SYFT_SPDX" > "$PATCHED_SPDX"
fi

# --- (e) wrap corrected SPDX in the in-toto Statement (subject = IMAGE manifest) ---
SPDX_LAYER="$WORKDIR/spdx.intoto.json"
jq -c -n \
  --slurpfile spdx "$PATCHED_SPDX" \
  --arg name "pkg:docker/${REPO_NOSCHEME}@${TAG}?platform=linux%2Famd64" \
  --arg dig "${IMG_DIGEST#sha256:}" '
  {
    "_type": "https://in-toto.io/Statement/v0.1",
    "predicateType": "https://spdx.dev/Document",
    "subject": [{"name": $name, "digest": {"sha256": $dig}}],
    "predicate": $spdx[0]
  }' > "$SPDX_LAYER"

# --- (f) copy the existing provenance layer bytes UNCHANGED ---
ATT_MANIFEST_JSON="$(docker buildx imagetools inspect "${REPO}@${ATT_DIGEST}" --raw)"
PROV_LAYER_DIGEST="$(printf '%s' "$ATT_MANIFEST_JSON" | jq -r '
  [.layers[] | select(.annotations["in-toto.io/predicate-type"] == "https://slsa.dev/provenance/v1")][0].digest // ""')"
if [ -z "$PROV_LAYER_DIGEST" ]; then
  log "No provenance layer found in the attestation-manifest for $REF."
  exit 1
fi
PROV_LAYER="$WORKDIR/provenance.intoto.json"
oras blob fetch "${REPO}@${PROV_LAYER_DIGEST}" --output "$PROV_LAYER"

# --- (g) digests + new config blob with updated diff_ids ---
sha() { sha256sum "$1" | cut -d' ' -f1; }
NEW_SPDX_DIGEST="sha256:$(sha "$SPDX_LAYER")"
NEW_SPDX_SIZE="$(wc -c < "$SPDX_LAYER")"
PROV_SIZE="$(wc -c < "$PROV_LAYER")"

CONFIG_BLOB="$WORKDIR/config.json"
jq -c -n --arg spdx "$NEW_SPDX_DIGEST" --arg prov "$PROV_LAYER_DIGEST" '
  {"architecture":"unknown","os":"unknown","config":{},
   "rootfs":{"type":"layers","diff_ids":[$spdx,$prov]}}' > "$CONFIG_BLOB"
NEW_CONFIG_DIGEST="sha256:$(sha "$CONFIG_BLOB")"
NEW_CONFIG_SIZE="$(wc -c < "$CONFIG_BLOB")"

# --- (h) push blobs + the new attestation-manifest ---
oras blob push "${REPO}@${NEW_SPDX_DIGEST}" --media-type application/vnd.in-toto+json "$SPDX_LAYER"
oras blob push "${REPO}@${PROV_LAYER_DIGEST}" --media-type application/vnd.in-toto+json "$PROV_LAYER"
oras blob push "${REPO}@${NEW_CONFIG_DIGEST}" --media-type application/vnd.oci.image.config.v1+json "$CONFIG_BLOB"

NEW_ATT_MANIFEST="$WORKDIR/attestation-manifest.json"
jq -c -n \
  --arg cfg "$NEW_CONFIG_DIGEST" --argjson cfgsize "$NEW_CONFIG_SIZE" \
  --arg spdx "$NEW_SPDX_DIGEST" --argjson spdxsize "$NEW_SPDX_SIZE" \
  --arg prov "$PROV_LAYER_DIGEST" --argjson provsize "$PROV_SIZE" '
  {
    "schemaVersion": 2,
    "mediaType": "application/vnd.oci.image.manifest.v1+json",
    "config": {"mediaType":"application/vnd.oci.image.config.v1+json","digest":$cfg,"size":$cfgsize},
    "layers": [
      {"mediaType":"application/vnd.in-toto+json","digest":$spdx,"size":$spdxsize,
       "annotations":{"in-toto.io/predicate-type":"https://spdx.dev/Document"}},
      {"mediaType":"application/vnd.in-toto+json","digest":$prov,"size":$provsize,
       "annotations":{"in-toto.io/predicate-type":"https://slsa.dev/provenance/v1"}}
    ]
  }' > "$NEW_ATT_MANIFEST"

NEW_ATT_DIGEST="sha256:$(sha "$NEW_ATT_MANIFEST")"
oras manifest push "${REPO}@${NEW_ATT_DIGEST}" \
  --media-type application/vnd.oci.image.manifest.v1+json "$NEW_ATT_MANIFEST"
log "pushed new attestation-manifest: $NEW_ATT_DIGEST"

# --- (i) rebuild the OCI index: unchanged image entry + new attestation entry ---
# imagetools create -f takes one source descriptor (JSON) per line/entry.
IMG_DESC="$WORKDIR/img.desc.json"
ATT_DESC="$WORKDIR/att.desc.json"

printf '%s' "$INDEX_JSON" | jq -c '
  [.manifests[] | select((.annotations["vnd.docker.reference.type"] // "") == "")][0]' > "$IMG_DESC"

# new attestation descriptor: preserve the ORIGINAL annotations
# (vnd.docker.reference.type + vnd.docker.reference.digest -> image manifest),
# only swap in the new digest/size.
printf '%s' "$INDEX_JSON" | jq -c \
  --arg dig "$NEW_ATT_DIGEST" --argjson size "$(wc -c < "$NEW_ATT_MANIFEST")" '
  ([.manifests[] | select((.annotations["vnd.docker.reference.type"] // "") == "attestation-manifest")][0])
  | .digest = $dig
  | .size = $size
  | .mediaType = "application/vnd.oci.image.manifest.v1+json"
' > "$ATT_DESC"

docker buildx imagetools create \
  -f "$IMG_DESC" \
  -f "$ATT_DESC" \
  --tag "$REF"

log "retagged $REF with enriched SBOM (license='${LABEL_LICENSES}', source='${LABEL_SOURCE}')"
