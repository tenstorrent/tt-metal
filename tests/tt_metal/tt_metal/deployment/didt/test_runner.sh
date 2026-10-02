#!/bin/sh
# Wraps tests/didt/test_minimal_matmul.py (already present in tt-metal): runs the
# minimal_matmul DIDT compute stress test on all visible chips together, as a single
# synchronized mesh workload. Must be run from the repository root with the tests already built.

set -u

LOGDIR="."
PYTHON="$(command -v python3 || command -v python)"
MGD="tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"

usage() {
	cat << EOF
Usage: $0 [--output <logdir>] [--mgd <path>]

Optional:
    --output <logdir>    Directory to write the log to (default: current directory)
    --mgd <path>         Path to the mesh graph descriptor (.textproto) file to use (default: $MGD)
    -h                   Display this help message and exit
EOF
}

while [ -n "${1:-}" ]
do
	case "$1" in
	--output)
		if [ -z "${2:-}" ]; then echo "Missing argument to $1"; exit 1; fi
		LOGDIR="$2"
		shift
		;;
	--mgd)
		if [ -z "${2:-}" ]; then echo "Missing argument to $1"; exit 1; fi
		MGD="$2"
		shift
		;;
	-h)
		usage
		exit
		;;
	*)
		echo "Unknown option: $1"
		usage
		exit 1
		;;
	esac
	shift
done

NUM_DEVICES="$(tt-smi -s 2>&1 | jq '.device_info | length' 2>/dev/null)"
case "$NUM_DEVICES" in
''|*[!0-9]*) echo "Could not detect number of visible devices via tt-smi -s"; exit 1 ;;
esac
echo "Detected $NUM_DEVICES visible chip(s), running combined DIDT test across all of them"

if [ -n "$MGD" ]; then
	case "$MGD" in
	/*) ;;
	*) MGD="$PWD/$MGD" ;;
	esac
	if [ ! -f "$MGD" ]; then echo "Mesh graph descriptor not found: $MGD"; exit 1; fi
	TT_MESH_GRAPH_DESC_PATH="$MGD"
	export TT_MESH_GRAPH_DESC_PATH
fi

mkdir -p "$LOGDIR"

log="$LOGDIR/didt_all_chips.log"
junit="$LOGDIR/didt_all_chips_junit.xml"

# Carries pytest's exit status out of the tee pipeline
RCFILE="$(mktemp)"
echo 0 > "$RCFILE"
{ $PYTHON -m pytest tests/didt/test_minimal_matmul.py::test_minimal_matmul \
	-k "all and bf16_HiFi2" \
	--didt-workload-iterations 500 \
	--determinism-check-interval 50 \
	--timeout 400 -q --junitxml="$junit" 2>&1 || echo "$?" > "$RCFILE"; } | tee "$log"
rc="$(cat "$RCFILE")"
rm -f "$RCFILE"

if [ "$rc" -eq 0 ]
then
	if [ ! -f "$junit" ]
	then
		echo "DIDT (all chips): FAILED, missing JUnit report $junit"
		rc=1
	else
		counts="$($PYTHON - "$junit" << 'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
tests = sum(int(s.get("tests", 0)) for s in suites)
failures = sum(int(s.get("failures", 0)) for s in suites)
errors = sum(int(s.get("errors", 0)) for s in suites)
skipped = sum(int(s.get("skipped", 0)) for s in suites)
print(tests, failures, errors, skipped)
PY
)"
		set -- $counts
		tests="$1"
		failures="$2"
		errors="$3"
		skipped="$4"
		if [ "$tests" -ne 1 ] || [ "$failures" -ne 0 ] || [ "$errors" -ne 0 ] || [ "$skipped" -ne 0 ]
		then
			echo "DIDT (all chips): FAILED, expected exactly 1 passed test with zero skips (tests=$tests failures=$failures errors=$errors skipped=$skipped)"
			rc=1
		fi
	fi
fi

if [ "$rc" -eq 0 ]
then
	echo "DIDT (all chips): PASSED (log: $log)"
else
	echo "DIDT (all chips): FAILED, exit code $rc (log: $log)"
fi

exit "$rc"
