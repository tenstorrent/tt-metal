#!/usr/bin/env bash
#
# cpmcache-report.sh
# -----------------------------------------------------------------------------
# RUN THIS: at the END of the job (after build, before / instead of relying on
# "Publish only if changed").
#
# It re-fingerprints the CPM source cache and diffs it against the baseline
# captured by cpmcache-snapshot.sh, then tells you EXACTLY what changed during
# the build and whether the churn lives in .git dirs or in real source files.
#
# Categories:
#   ADDED     - file present now, absent at baseline
#   REMOVED   - file present at baseline, gone now
#   CONTENT   - same path, different content hash  (a "real" change)
#   METADATA  - same path & content, but mode/mtime changed (defeats publish-if-changed
#               only if TeamCity's detector is metadata-sensitive)
# -----------------------------------------------------------------------------
set -euo pipefail
export LC_ALL=C

# --- config (must match the snapshot step) ----------------------------------
CACHE_DIR="${CACHE_DIR:-${CPM_SOURCE_CACHE:-.cpmcache}}"
SNAPSHOT="${SNAPSHOT:-${TMPDIR:-/tmp}/cpmcache.baseline.tsv}"
HASH="${HASH:-1}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
TOPN="${TOPN:-15}"            # how many top churning dirs to list
VERBOSE="${VERBOSE:-0}"       # 1 = also list every CONTENT/ADDED file path
TEAMCITY="${TEAMCITY:-1}"     # 1 = emit ##teamcity statistic lines (harmless elsewhere)

# --- manifest builder (KEEP IN SYNC with cpmcache-snapshot.sh) ---------------
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

# --- main --------------------------------------------------------------------
if [[ ! -f "$SNAPSHOT" ]]; then
  echo "ERROR: baseline '$SNAPSHOT' not found. Did cpmcache-snapshot.sh run first in this job?" >&2
  exit 1
fi
if [[ ! -d "$CACHE_DIR" ]]; then
  echo "ERROR: cache dir '$CACHE_DIR' missing at end of job." >&2
  exit 1
fi

CUR="$(mktemp)"
trap 'rm -f "$CUR"' EXIT
echo "Re-fingerprinting '$CACHE_DIR' and diffing vs baseline..."
build_manifest "$CACHE_DIR" "$CUR"

awk -F'\t' \
  -v topn="$TOPN" -v verbose="$VERBOSE" -v tc="$TEAMCITY" '
  function hum(b,  u,i){ split("B KB MB GB TB",u," "); i=1; while(b>=1024 && i<5){b/=1024;i++} return sprintf("%.1f%s",b,u[i]) }
  function isgit(p){ return (p ~ /(^|\/)\.git(\/|$)/) }
  function topdir(p,  s,a,n,d,i,depth){
    s=p; sub(/^\.\//,"",s)                       # strip leading ./
    if (s ~ /\.git(\/|$)/) { sub(/\.git\/.*/,".git",s); return s }   # collapse to the .git dir
    n=split(s,a,"/"); depth=(n-1<3?n-1:3); if(depth<1)depth=1
    d=a[1]; for(i=2;i<=depth;i++) d=d"/"a[i]; return d               # else group ~3 levels deep
  }
  # baseline first
  NR==FNR { bmode[$5]=$1; bsize[$5]=$2; bmtime[$5]=$3; bhash[$5]=$4; base[$5]=1; nb++; next }
  {
    p=$5; cur[p]=1; nc++
    if (!(p in base)) { cat="ADDED"; bytes=$2+0 }
    else {
      if ($4!="-" && bhash[p]!="-" && $4!=bhash[p]) { cat="CONTENT"; bytes=$2+0 }
      else if ($1!=bmode[p] || $3!=bmtime[p])       { cat="METADATA"; bytes=$2+0 }
      else                                           { cat="SAME"; bytes=0 }
    }
    if (cat=="SAME") next
    g = isgit(p) ? "git" : "src"
    cnt[cat]++; byt[cat]+=bytes
    cnt2[cat":"g]++; byt2[cat":"g]+=bytes
    gcnt[g]++; gbyt[g]+=bytes
    d=topdir(p); dcnt[d]++; dbyt[d]+=bytes
    if (verbose=="1" && (cat=="CONTENT"||cat=="ADDED")) print "  ["cat"] "hum(bytes)"\t"p > "/dev/stderr"
  }
  END {
    for (p in base) if (!(p in cur)) {
      cat="REMOVED"; bytes=bsize[p]+0
      g = isgit(p)?"git":"src"
      cnt[cat]++; byt[cat]+=bytes; cnt2[cat":"g]++; byt2[cat":"g]+=bytes
      gcnt[g]++; gbyt[g]+=bytes
      d=topdir(p); dcnt[d]++; dbyt[d]+=bytes
    }
    split("ADDED REMOVED CONTENT METADATA", order, " ")
    print ""
    print "================ CPM CACHE CHURN ================"
    printf "baseline files: %d    current files: %d\n\n", nb, nc
    printf "%-9s %8s %12s   %8s/%-10s %8s/%-10s\n","category","files","bytes","git f","git bytes","src f","src bytes"
    tch=0; tcb=0
    for (i=1;i<=4;i++){ c=order[i];
      printf "%-9s %8d %12s   %8d/%-10s %8d/%-10s\n", c, cnt[c]+0, hum(byt[c]+0), \
        cnt2[c":git"]+0, hum(byt2[c":git"]+0), cnt2[c":src"]+0, hum(byt2[c":src"]+0)
      tch+=cnt[c]+0; tcb+=byt[c]+0
    }
    printf "%-9s %8d %12s\n", "TOTAL", tch, hum(tcb)
    print ""
    print "---- churn by location ----"
    printf "  .git : %d files, %s\n", gcnt["git"]+0, hum(gbyt["git"]+0)
    printf "  src  : %d files, %s\n", gcnt["src"]+0, hum(gbyt["src"]+0)
    if (tcb>0) printf "  => %.0f%% of churned bytes is inside .git dirs\n", (gbyt["git"]+0)*100/tcb
    print ""
    printf "---- top %d churning dirs (by bytes) ----\n", topn
    n=0; for (d in dbyt) printf "%020d\t  %-10s %6d files  %s\n", dbyt[d], hum(dbyt[d]), dcnt[d], d | "sort -rn | head -"topn" | cut -f2-"
    close("sort -rn | head -"topn" | cut -f2-")
    print "================================================"
    if (tc=="1") {
      print "##teamcity[buildStatisticValue key='\''cpmCacheChurnBytes'\'' value='\''" tcb "'\'']"
      print "##teamcity[buildStatisticValue key='\''cpmCacheChurnGitBytes'\'' value='\''" (gbyt["git"]+0) "'\'']"
    }
  }
' "$SNAPSHOT" "$CUR"
