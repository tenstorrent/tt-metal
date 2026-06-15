#!/usr/bin/env bash
#
# cpmcache-snapshot.sh
# -----------------------------------------------------------------------------
# RUN THIS: after the Build Cache restore, BEFORE `cmake` configure.
#
# It records a fingerprint (mode + size + mtime + content hash) of every file
# in the CPM source cache, and prints a space-consumption breakdown so you can
# see whether .git dirs dominate. The fingerprint is written outside the cache
# so it does not become part of what gets re-published.
#
# Companion: cpmcache-report.sh (run at end of job) diffs against this baseline.
# -----------------------------------------------------------------------------
set -euo pipefail
export LC_ALL=C

# --- config (override via env) ----------------------------------------------
CACHE_DIR="${CACHE_DIR:-${CPM_SOURCE_CACHE:-.cpmcache}}"   # what to fingerprint
SNAPSHOT="${SNAPSHOT:-${TMPDIR:-/tmp}/cpmcache.baseline.tsv}"  # MUST be outside CACHE_DIR
HASH="${HASH:-1}"          # 1 = content hashes (distinguishes content vs mtime-only churn); 0 = metadata only
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

# --- manifest builder (KEEP IN SYNC with cpmcache-report.sh) -----------------
build_manifest() {  # $1 = source dir, $2 = output manifest path
  local src="$1" out="$2" td
  td="$(mktemp -d)"
  ( cd "$src" && find . -type f -printf '%m\t%s\t%T@\t%p\n' ) \
    | sort -t "$(printf '\t')" -k4 > "$td/meta.tsv"
  if [[ "$HASH" == "1" ]]; then
    ( cd "$src" && find . -type f -print0 | xargs -0 -P"$JOBS" sha1sum 2>/dev/null ) \
      | sed -E 's/^([0-9a-fA-F]{40}) [ *](.*)$/\2\t\1/' > "$td/hash.tsv"
  else
    : > "$td/hash.tsv"
  fi
  awk -F'\t' -v hf="$td/hash.tsv" '
    FILENAME==hf { h[$1]=$2; next }
    { print $1"\t"$2"\t"$3"\t"(($4 in h)?h[$4]:"-")"\t"$4 }
  ' "$td/hash.tsv" "$td/meta.tsv" | sort -t "$(printf '\t')" -k5 > "$out"
  rm -rf "$td"
}

# --- space report ------------------------------------------------------------
space_report() {  # $1 = dir
  local dir="$1" total git nongit
  echo "=== SPACE in $dir ==="
  total=$(du -sb --apparent-size "$dir" 2>/dev/null | awk '{print $1+0}')
  git=$(find "$dir" -type d -name .git -prune -print0 2>/dev/null \
        | xargs -0 -r du -sb --apparent-size 2>/dev/null | awk '{s+=$1} END{print s+0}')
  nongit=$(( total - git ))
  pct() { awk -v a="$1" -v b="$2" 'BEGIN{printf (b>0)?"%.1f%%":"-", a*100/b}'; }
  printf '  total      %12s\n' "$(numfmt --to=iec "$total" 2>/dev/null || echo "${total}B")"
  printf '  .git dirs  %12s   (%s of total)\n' "$(numfmt --to=iec "$git" 2>/dev/null || echo "${git}B")" "$(pct "$git" "$total")"
  printf '  non-.git   %12s   (%s of total)\n' "$(numfmt --to=iec "$nongit" 2>/dev/null || echo "${nongit}B")" "$(pct "$nongit" "$total")"
  echo
  echo "Largest entries (depth<=2):"
  du -b --apparent-size --max-depth=2 "$dir" 2>/dev/null | sort -rn | head -15 \
    | awk '{ $1=sprintf("%-10s", ($1>=1073741824)?sprintf("%.1fG",$1/1073741824):($1>=1048576)?sprintf("%.1fM",$1/1048576):sprintf("%.0fK",$1/1024)); print "  " $0 }'
  echo
  echo "Largest .git dirs:"
  find "$dir" -type d -name .git -prune -print0 2>/dev/null \
    | xargs -0 -r du -sb --apparent-size 2>/dev/null | sort -rn | head -10 \
    | awk '{ $1=sprintf("%-10s", ($1>=1073741824)?sprintf("%.1fG",$1/1073741824):($1>=1048576)?sprintf("%.1fM",$1/1048576):sprintf("%.0fK",$1/1024)); print "  " $0 }'
  echo "============================="
}

# --- main --------------------------------------------------------------------
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "WARNING: cache dir '$CACHE_DIR' does not exist." >&2
  echo "         The Build Cache was probably NOT restored (or CACHE_DIR is wrong)." >&2
  : > "$SNAPSHOT"   # empty baseline -> everything will show up as ADDED in the report
  exit 0
fi

echo "Fingerprinting '$CACHE_DIR' -> '$SNAPSHOT'  (hash=$HASH, jobs=$JOBS)"
build_manifest "$CACHE_DIR" "$SNAPSHOT"
echo "Baseline: $(wc -l < "$SNAPSHOT") files recorded."
echo
space_report "$CACHE_DIR"
