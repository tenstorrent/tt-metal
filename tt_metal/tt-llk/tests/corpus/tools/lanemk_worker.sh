#!/usr/bin/env bash
# laneMK exabox node-side worker — runs on ONE glx host, work-steals ops, streams each
# op's full 2^32 sem-vs-hand equivalence in resume-safe bands, object-identity-gated.
#
#   lanemk_worker.sh <ops_tsv> <claims_dir> <out_dir> <build_dir> <venv> <llk_home> <pydir> <band_bits> <idmap>
#
# ops_tsv rows: op<TAB>sem_node<TAB>hand_node   (consume-mode nodes; ELFs prebuilt in build_dir)
# idmap rows:   op<TAB>sem_variant<TAB>sem_elf_sha256<TAB>hand_variant<TAB>hand_elf_sha256<TAB>status
#   OBJECT-IDENTITY GATE (in-farm, toolchain-free): before running an op, sha256sum the shipped
#   math.elf for sem_variant & hand_variant and REFUSE unless both match the pinned reference AND
#   sem!=hand. A verdict is never run on an ELF whose identity as the certified pin-59 object fails.
# Claims are NFS-atomic (mkdir); a freed host steals the next unclaimed op -> zero idle.
# Per op: for each band, run sem+hand in ONE device session each (the LANEMK_STREAM hook),
# compare per-leg SHA; capture each leg's ELF dir (object identity: assert sem!=hand). A band
# that differs is a witness band (flagged for bisection). Verdict written atomically.
set -uo pipefail
OPS="${1:?ops_tsv}"; CLAIMS="${2:?claims_dir}"; OUTD="${3:?out_dir}"; BUILD="${4:?build_dir}"
VENV="${5:?venv}"; LLKH="${6:?llk_home}"; PYDIR="${7:?pydir}"; BB="${8:-28}"; IDMAP="${9:?idmap}"
TILE="${LANEMK_TILE_DIM:-256,256}"
SELFDIR="$(cd "$(dirname "$0")" && pwd)"
EXTRACT="${LANEMK_EXTRACT:-$SELFDIR/elf_text_sha.py}"   # dependency-free .text sha256 (no objcopy)

identity_gate(){ # <op> -> prints "OK <sem_text_sha> <hand_text_sha>" or "FAIL <reason>"
  local op="$1" row; row=$(awk -F'\t' -v o="$op" '$1==o{print;exit}' "$IDMAP")
  [ -n "$row" ] || { echo "FAIL no-idmap-row"; return; }
  local sv ss hv hs; sv=$(echo "$row"|cut -f2); ss=$(echo "$row"|cut -f3); hv=$(echo "$row"|cut -f4); hs=$(echo "$row"|cut -f5)
  local se="$ELFROOT/$sv/elf/math.elf" he="$ELFROOT/$hv/elf/math.elf"
  [ -f "$se" ] && [ -f "$he" ] || { echo "FAIL missing-elf"; return; }
  # object identity = the certified pin-59 kernel .text (stable anchor; ELF file is not).
  local sa ha; sa=$("$VENV" "$EXTRACT" "$se" 2>/dev/null); ha=$("$VENV" "$EXTRACT" "$he" 2>/dev/null)
  [ "$sa" == "$ss" ] || { echo "FAIL sem-text-mismatch($sa!=$ss)"; return; }
  [ "$ha" == "$hs" ] || { echo "FAIL hand-text-mismatch($ha!=$hs)"; return; }
  [ "$sa" != "$ha" ] || { echo "FAIL sem==hand"; return; }
  echo "OK $sa $ha"
}
mkdir -p "$CLAIMS" "$OUTD"
HOST="$(hostname -s)"
BAND=$(( 1 << BB )); TOTAL=$(( 1 << 32 )); NB=$(( (TOTAL + BAND - 1) / BAND ))
sha_key(){ printf '%s' "$HOST|$1" | sha256sum | cut -c1-16; }

# Per-host node-local RUNNER_TEMP holding a private copy of the prebuilt ELFs. Sharing one
# RUNNER_TEMP on /data races on conftest's order_records mkdir (laneKC gotcha) AND is slow NFS;
# a node-local copy is race-free and faster. ELFs are consume-only (never rebuilt).
LOCALRT="${LANEMK_LOCAL_RT:-/tmp/lanemk-rt-$HOST}"
if [ ! -d "$LOCALRT/tt-llk-build/sources" ]; then
  mkdir -p "$LOCALRT"; cp -a "$BUILD/tt-llk-build" "$LOCALRT/" 2>/dev/null || true
fi
RT="$LOCALRT"; [ -d "$RT/tt-llk-build/sources" ] || RT="$BUILD"   # fall back to shared if copy failed
ELFROOT="$RT/tt-llk-build/sources/eltwise_unary_sfpu_test.cpp"

run_leg(){ # <node> <start> <count> <shafile> <logfile>  -> prints the output_sha256 hex
  local node="$1" start="$2" count="$3" shaf="$4" logf="$5"
  [ -s "$shaf" ] && { cat "$shaf"; return 0; }   # resume
  local sha
  for _ in 1 2 3; do   # cold-start / transient-race tolerance (retry a failed dispatch)
    ( cd "$PYDIR" && CHIP_ARCH=blackhole SHORT_ARCH=bh LLK_HOME="$LLKH" PYTHONUNBUFFERED=1 \
        RUNNER_TEMP="$RT" LANEMK_TILE_DIM="$TILE" LANEMK_WAIT_TIMEOUT="${LANEMK_WAIT_TIMEOUT:-120}" \
        LANEMK_STREAM="$start,$count,$OUTD/.tmp.$node.$start" \
        "$VENV" -m pytest -o addopts= -q -s --compile-consumer "$node" > "$logf" 2>&1 )
    sha=$(grep -aoE 'output_sha256=[0-9a-f]{64}' "$logf" | head -1 | cut -d= -f2)
    [ -n "$sha" ] && { printf '%s\n' "$sha" | tee "$shaf"; return 0; }
  done
  return 1
}

do_op(){ # <op> <sem_node> <hand_node>
  local op="$1" sem="$2" hand="$3"
  local od="$OUTD/$op"; mkdir -p "$od/bands"
  [ -s "$od/VERDICT" ] && { echo "[$HOST] $op resume-done"; return 0; }
  # OBJECT-IDENTITY GATE first — never stream a verdict on an unverified ELF.
  local idg; idg=$(identity_gate "$op")
  if [ "${idg%% *}" != "OK" ]; then
    echo "op=$op verdict=REFUSED-IDENTITY($idg) host=$HOST" > "$od/VERDICT"
    echo "[$HOST] $op -> REFUSED-IDENTITY ($idg)"; return 0
  fi
  local sem_elf_sha hand_elf_sha; sem_elf_sha=$(echo "$idg"|awk '{print $2}'); hand_elf_sha=$(echo "$idg"|awk '{print $3}')
  local all_eq=1 witness="" covered=0
  local led="$od/LEDGER.tsv"; echo -e "band\tstart\tcount\tsem_sha\thand_sha\tverdict" > "$led"
  local k s c
  for (( k=0; k<NB; k++ )); do
    s=$(( k * BAND )); c=$(( BAND )); (( s + c > TOTAL )) && c=$(( TOTAL - s ))
    local ssha hsha
    ssha=$(run_leg "$sem"  "$s" "$c" "$od/bands/b$k-sem.sha"  "$od/bands/b$k-sem.log")  || { echo "$op band $k sem FAIL" > "$od/ERROR"; return 1; }
    hsha=$(run_leg "$hand" "$s" "$c" "$od/bands/b$k-hand.sha" "$od/bands/b$k-hand.log") || { echo "$op band $k hand FAIL" > "$od/ERROR"; return 1; }
    local v="EQ"; [ "$ssha" == "$hsha" ] || { v="DIFF"; all_eq=0; witness="$witness $k"; }
    echo -e "$k\t$s\t$c\t$ssha\t$hsha\t$v" >> "$led"
    covered=$(( covered + c ))
  done
  local verdict
  if [ "$covered" -ne "$TOTAL" ]; then verdict="REFUSED-coverage($covered!=$TOTAL)"
  elif [ "$all_eq" -eq 1 ]; then verdict="BIT-EXACT-ALL-INPUTS"
  else verdict="DIVERGENT witness_bands=$witness"; fi
  echo "op=$op verdict=$verdict covered=$covered full2^32=$([ $covered -eq $TOTAL ] && echo true || echo false) host=$HOST sem_elf_sha=$sem_elf_sha hand_elf_sha=$hand_elf_sha" > "$od/VERDICT"
  echo "[$HOST] $op -> $verdict"
}

echo "[$HOST] worker start $(date -u +%H:%M:%SZ) bands=$NB tile=$TILE"
# dispersed claim order (per-host sha salt) so hosts rarely race the same op
while IFS=$'\t' read -r op sem hand; do [ -n "$op" ] && printf '%s\t%s\t%s\t%s\n' "$(sha_key "$op")" "$op" "$sem" "$hand"; done < "$OPS" \
 | sort | cut -f2- | while IFS=$'\t' read -r op sem hand; do
    [ -n "$op" ] || continue
    if mkdir "$CLAIMS/$op" 2>/dev/null; then echo "$HOST" > "$CLAIMS/$op/owner"; do_op "$op" "$sem" "$hand" || echo "$HOST FAILED $op" >> "$CLAIMS/$op/fail"; fi
done
echo "[$HOST] worker DONE $(date -u +%H:%M:%SZ)"
