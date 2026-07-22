#!/bin/sh

set -e

LOGDIR="."
LOGFILE="deployment_$(hostname)_$(date +%4Y-%m-%d-%H-%M-%S).log"
ITERS=5

usage() {
	echo "Usage: $0 [-l logdir]"
	echo "	-l <logdir>		The directory where to save the log file"
}

while [ -n "$1" ]
do
	case "$1" in
	-l)
		if [ -z "$2" ]; then echo "Missing argument to $1"; exit 1; fi
		LOGDIR="$2"
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

mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/$LOGFILE"

GREEN="$(printf '\033[32m')"
RED="$(printf '\033[31m')"
RESET="$(printf '\033[m')"

FAIL=failed
PASS=passed

if [ -t 1 ]
then
	FAIL="$RED$FAIL$RESET"
	PASS="$GREEN$PASS$RESET"
fi

truncate -s0 "$LOGFILE"

failures=0
passes=0

run_test() {
	for i in $(seq "$ITERS")
	do
		printf "$MESSAGE loop $i "
		if ! "$@" >> "$LOGFILE" 2>&1
		then
			failures=$((failures + 1))
			echo "$FAIL"
		else
			passes=$((passes + 1))
			echo "$PASS"
		fi
	done
}

MESSAGE='Ethernet tests\t'
run_test python tests/tt_metal/tt_metal/deployment/eth/test_runner.py

MESSAGE='DRAM tests\t'
run_test python tests/tt_metal/tt_metal/deployment/dram/test_runner.py

MESSAGE='PCIe read test\t'
run_test ./build/tools/mem_bench --benchmark_filter='Device Reading Host/1073741824/32768/1/0/0/iterations:5/manual_time' --device-id=0

MESSAGE='PCIe write test\t'
run_test ./build/tools/mem_bench --benchmark_filter='Device Writing Host/1073741824/32768/0/1/0/iterations:5/manual_time' --device-id=0

if [ "$passes" -gt 0 ]
then
	echo "$passes tests $PASS"
fi

if [ "$failures" -gt 0 ]
then
	echo "$failures tests $FAIL"
	exit 1
fi
