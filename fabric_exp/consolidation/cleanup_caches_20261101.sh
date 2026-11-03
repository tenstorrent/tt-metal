#!/bin/bash
# Deletes the kernel caches listed in cleanup_caches_20261101.txt (user, 2026-11-01: caches of variants tried and
# rejected; regenerable from source). Run inside a broker job so no device job uses a cache meanwhile. Only
# /home/aagarwal/.tt-metal-cache-* paths, never the Phase 0 caches (p0, p0-O2, p0-Os) or main's.
L=/home/aagarwal/tt-metal/fabric_exp/consolidation/cleanup_caches_20261101.txt
n=0
while read -r d; do
    case "$d" in
        /home/aagarwal/.tt-metal-cache-p0|/home/aagarwal/.tt-metal-cache-p0-*|*main*) echo "kept $d" ;;
        /home/aagarwal/.tt-metal-cache-?*) rm -rf -- "$d"; n=$((n + 1)) ;;
        *) echo "skipped $d" ;;
    esac
done < "$L"
echo "deleted $n caches"; df -h /home | tail -1
