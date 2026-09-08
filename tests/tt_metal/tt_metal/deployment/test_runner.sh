#!/bin/sh

set -e

LOGDIR="."
ITERATIONS=5
PYTHON="$(command -v python3 || command -v python)"
SKIP_RESET=0
CONTINUE_ON_FAILURE=0
RESET_CMD="tt-smi -glx_reset"

usage() {
	cat << EOF
Usage: $0 [--output <logdir>] [--iterations <n>] [--skip-reset] [--continue-on-failure] [--no-eth-links]

Run the deployment test suite (Ethernet, DRAM, PCIe read/write).
Must be run from the repository root with the tests already built.
Everything printed to the console is also written to a single log file per run.

An iteration is a board reset followed by one run of each test.

Optional:
    --output <logdir>                       Directory where the log file is written
                                            (default: current directory)
    --iterations <n>                        Number of iterations to run (default: $ITERATIONS).
                                            Stops after an iteration fails.
    --skip-reset                            Do not reset the boards before each iteration.
                                            Use to stress the tests without resets in between.
    --continue-on-failure                   Run all iterations even if one fails
    --no-eth-links                          Do not require a specific number of Ethernet links per
                                            chip. Use on partially cabled systems, otherwise
                                            10 links per chip are expected.
    -h                                      Display this help message and exit

Examples:
    Deployment qualification ($ITERATIONS reset-and-test iterations), logs in ./logs:
        $0 --output ./logs

    Quick health check (one reset-and-test iteration):
        $0 --output ./logs --iterations 1

    Stress run of 20 iterations without resets, collecting every failure:
        $0 --output ./logs --iterations 20 --skip-reset --continue-on-failure

    Run on a partially cabled system:
        $0 --output ./logs --no-eth-links
EOF
}

while [ -n "$1" ]
do
	case "$1" in
	--output)
		if [ -z "$2" ]; then echo "Missing argument to $1"; exit 1; fi
		LOGDIR="$2"
		shift
		;;
	--iterations)
		if [ -z "$2" ]; then echo "Missing argument to $1"; exit 1; fi
		case "$2" in
		''|*[!0-9]*|0) echo "$1 must be a positive integer, got '$2'"; exit 1 ;;
		esac
		ITERATIONS="$2"
		shift
		;;
	--skip-reset)
		SKIP_RESET=1
		;;
	--continue-on-failure)
		CONTINUE_ON_FAILURE=1
		;;
	--no-eth-links)
		ETH_TEST_EXPECTED_LINKS=0
		export ETH_TEST_EXPECTED_LINKS
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

RUN_LOG="$LOGDIR/deployment_$(hostname)_$(date +%4Y-%m-%d-%H-%M-%S).log"
: > "$RUN_LOG"

# Carries a command's exit status out of the tee pipeline
RCFILE="$(mktemp)"
trap 'rm -f "$RCFILE"' EXIT HUP INT TERM

RULE_HEAVY='=============================================================================='
RULE_LIGHT='------------------------------------------------------------------------------'

BOLD=""
NORM=""
FAIL=failed
PASS=passed

if [ -t 1 ]
then
	BOLD="$(printf '\033[1m')"
	NORM="$(printf '\033[m')"
	FAIL="$(printf '\033[31m')$FAIL$NORM"
	PASS="$(printf '\033[32m')$PASS$NORM"
fi

# emit: print a line.
emit() {
	printf '%s\n' "$*"
	printf '%s\n' "$*" >> "$RUN_LOG"
}

# emit_bold: print a highlighted line.
emit_bold() {
	printf '%s%s%s\n' "$BOLD" "$*" "$NORM"
	printf '%s\n' "$*" >> "$RUN_LOG"
}

# emit_banner: print a heading delimited by heavy rules.
emit_banner() {
	emit ""
	emit "$RULE_HEAVY"
	emit_bold "$*"
	emit "$RULE_HEAVY"
}

# emit_section: print a heading delimited by light rules.
emit_section() {
	emit ""
	emit "$RULE_LIGHT"
	emit_bold "$*"
	emit "$RULE_LIGHT"
}

# emit_status <text> <passed|failed>: print a line ending in a pass/fail verdict.
emit_status() {
	case "$2" in
	passed) colored="$PASS" ;;
	*) colored="$FAIL" ;;
	esac
	printf '%s %s\n' "$1" "$colored"
	printf '%s %s\n' "$1" "$2" >> "$RUN_LOG"
}

# run_logged: run a command and set $rc to its exit status.
run_logged() {
	echo 0 > "$RCFILE"
	{ "$@" 2>&1 || echo "$?" > "$RCFILE"; } | tee -a "$RUN_LOG"
	rc="$(cat "$RCFILE")"
}

# run_reset <message>: reset the boards and return its exit status.
run_reset() {
	emit_section "$1"
	run_logged $RESET_CMD
	if [ "$rc" -ne 0 ]
	then
		emit "Reset failed (exit code $rc)"
		return "$rc"
	fi
	return 0
}

# run_test <label> <command...>: runs a test once and records its result.
run_test() {
	label="$1"
	shift
	emit_section "$label"
	run_logged "$@"
	if [ "$rc" -ne 0 ]
	then
		failures=$((failures + 1))
		emit_status "$label:" failed
		return 1
	fi
	passes=$((passes + 1))
	emit_status "$label:" passed
	return 0
}

# run_tests: runs one round of every test.
# Sets last_eth_ok, last_dram_ok, last_pcie_read_ok, last_pcie_write_ok (1=pass, 0=fail).
# Returns 1 if any test failed, 0 otherwise.
run_tests() {
	failures=0
	passes=0

	run_test 'Ethernet tests' $PYTHON tests/tt_metal/tt_metal/deployment/eth/test_runner.py &&
		last_eth_ok=1 || last_eth_ok=0

	run_test 'DRAM tests' $PYTHON tests/tt_metal/tt_metal/deployment/dram/test_runner.py &&
		last_dram_ok=1 || last_dram_ok=0

	run_test 'PCIe read test' ./build/tools/mem_bench --benchmark_filter='Device Reading Host/1073741824/32768/1/0/0/iterations:5/manual_time' --device-id=0 &&
		last_pcie_read_ok=1 || last_pcie_read_ok=0

	run_test 'PCIe write test' ./build/tools/mem_bench --benchmark_filter='Device Writing Host/1073741824/32768/0/1/0/iterations:5/manual_time' --device-id=0 &&
		last_pcie_write_ok=1 || last_pcie_write_ok=0

	emit_section 'Test results'

	if [ "$passes" -gt 0 ]
	then
		emit_status "$passes tests" passed
	fi

	if [ "$failures" -gt 0 ]
	then
		emit_status "$failures tests" failed
		return 1
	fi

	return 0
}

if [ "$SKIP_RESET" -eq 1 ]
then
	MODE="$ITERATIONS iteration(s), no board reset"
else
	MODE="$ITERATIONS iteration(s), $RESET_CMD before each"
fi

emit_banner "DEPLOYMENT TESTS RUN"
emit "$(printf '%-12s %s' 'Date:' "$(date)")"
emit "$(printf '%-12s %s' 'Host:' "$(hostname)")"
emit "$(printf '%-12s %s' 'Tests:' 'Ethernet, DRAM, PCIe read, PCIe write')"
emit "$(printf '%-12s %s' 'Mode:' "$MODE")"
emit "$(printf '%-12s %s' 'Run log:' "$RUN_LOG")"
emit "$RULE_HEAVY"

iteration_failures=0
reset_failures=0
iterations_run=0
eth_pass=0
dram_pass=0
pcie_read_pass=0
pcie_write_pass=0

for iteration in $(seq 1 "$ITERATIONS")
do
	emit_banner "ITERATION $iteration/$ITERATIONS"
	reset_ok=1
	if [ "$SKIP_RESET" -eq 0 ]
	then
		if ! run_reset "Resetting boards ($RESET_CMD)..."
		then
			reset_ok=0
			reset_failures=$((reset_failures + 1))
		fi
	fi
	if [ "$reset_ok" -eq 0 ]
	then
		last_eth_ok=0
		last_dram_ok=0
		last_pcie_read_ok=0
		last_pcie_write_ok=0
		iteration_failures=$((iteration_failures + 1))
		emit_banner "ITERATION $iteration FAILED (reset failed)"
	elif run_tests
	then
		emit_banner "ITERATION $iteration PASSED"
	else
		iteration_failures=$((iteration_failures + 1))
		emit_banner "ITERATION $iteration FAILED"
	fi
	iterations_run=$((iterations_run + 1))
	eth_pass=$((eth_pass + last_eth_ok))
	dram_pass=$((dram_pass + last_dram_ok))
	pcie_read_pass=$((pcie_read_pass + last_pcie_read_ok))
	pcie_write_pass=$((pcie_write_pass + last_pcie_write_ok))
	if [ "$CONTINUE_ON_FAILURE" -eq 0 ] && [ "$iteration_failures" -gt 0 ]
	then
		emit "Stopping: iteration $iteration failed."
		break
	fi
done

emit_banner "DEPLOYMENT TEST SUITE - RESULTS SUMMARY (${iterations_run}/${ITERATIONS} iterations ran)"
emit "$(printf '%-20s %s' 'Host:'            "$(hostname)")"
emit "$RULE_LIGHT"
emit "$(printf '%-20s %s' 'Reset failures:'   "$reset_failures/$iterations_run iterations failed")"
emit "$(printf '%-20s %s' 'Ethernet tests:'  "$eth_pass/$iterations_run iterations passed")"
emit "$(printf '%-20s %s' 'DRAM tests:'      "$dram_pass/$iterations_run iterations passed")"
emit "$(printf '%-20s %s' 'PCIe read test:'  "$pcie_read_pass/$iterations_run iterations passed")"
emit "$(printf '%-20s %s' 'PCIe write test:' "$pcie_write_pass/$iterations_run iterations passed")"
emit "$RULE_LIGHT"
if [ "$iteration_failures" -gt 0 ]
then
	emit_bold "$(printf '%-20s %s' 'Overall:' "$((iterations_run - iteration_failures))/$iterations_run iterations passed")"
	emit "$RULE_HEAVY"
	emit "Run log: $RUN_LOG"
	exit 1
fi
emit_bold "$(printf '%-20s %s' 'Overall:' "All $iterations_run iterations passed")"
emit "$RULE_HEAVY"
emit "Run log: $RUN_LOG"
