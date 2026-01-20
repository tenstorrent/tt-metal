#!/bin/bash
# Quick launcher for telemetry benchmark suite

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

show_usage() {
    cat << EOF
Telemetry Benchmark Suite Launcher

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    reduced         Run Phase 1 (reduced) benchmark suite (~2-3 hours)
    full            Run Phase 2 (full) benchmark suite (~9-12 hours)
    hypothesis      Run only core hypothesis validation test (~30 min)
    single          Run only single-device benchmark
    multi           Run only multi-device benchmark
    sustained       Run only sustained workload test
    analyze         Run post-analysis on existing results
    help            Show this help message

Options:
    --phase PHASE   Phase for individual tests: 'reduced' or 'full' (default: reduced)
    --output DIR    Output directory (default: /tmp)

Examples:
    # Run reduced suite (recommended first run)
    $0 reduced

    # Run full suite
    $0 full

    # Run only hypothesis test
    $0 hypothesis

    # Run single-device benchmark (reduced)
    $0 single --phase reduced

    # Analyze results
    $0 analyze --phase reduced

EOF
}

# Default values
PHASE="reduced"
OUTPUT_DIR="/tmp"

# Parse command
COMMAND="${1:-help}"
shift || true

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Execute command
case $COMMAND in
    reduced|full)
        echo "Running $COMMAND benchmark suite..."
        echo "This will take approximately $([ "$COMMAND" = "reduced" ] && echo "2-3" || echo "9-12") hours"
        echo ""
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 "$SCRIPT_DIR/run_telemetry_benchmark_suite.py" --phase "$COMMAND" --output "$OUTPUT_DIR"
        else
            echo "Cancelled"
            exit 0
        fi
        ;;

    hypothesis)
        echo "Running core hypothesis validation test..."
        python3 "$SCRIPT_DIR/validate_mmio_only.py"
        ;;

    single)
        echo "Running single-device benchmark ($PHASE phase)..."
        python3 "$SCRIPT_DIR/comprehensive_single_device_benchmark.py" "$PHASE"
        ;;

    multi)
        echo "Running multi-device benchmark ($PHASE phase)..."
        python3 "$SCRIPT_DIR/comprehensive_multi_device_benchmark.py" "$PHASE"
        ;;

    sustained)
        echo "Running sustained workload test..."
        python3 "$SCRIPT_DIR/sustained_workload_test.py" 1000
        ;;

    analyze)
        echo "Running post-analysis ($PHASE phase)..."
        python3 "$SCRIPT_DIR/analyze_benchmark_results.py" --phase "$PHASE" --output "$OUTPUT_DIR"
        ;;

    help|--help|-h)
        show_usage
        exit 0
        ;;

    *)
        echo "Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
echo "Complete!"
