#!/bin/sh

set -e

LOGFILE=log
ITERS=5

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

run_test() {
	for i in $(seq "$ITERS")
	do
		printf "$MESSAGE"
		if ! "$@" >> "$LOGFILE" 2>&1
		then
			failures=$((failures + 1))
			echo "$FAIL"
		else
			echo "$PASS"
		fi
	done
}

MESSAGE='Ethernet tests\t'
run_test python tests/tt_metal/tt_metal/deployment/eth/test_runner.py

MESSAGE='DRAM tests\t'
run_test python tests/tt_metal/tt_metal/deployment/dram/test_runner.py

if [ "$failures" -gt 0 ]
then
	echo "$failures tests failed"
	exit 1
fi
