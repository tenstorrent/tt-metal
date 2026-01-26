#!/usr/bin/env bash
# Simple wrapper for nd_bisect_n300.sh
# This script is kept for backward compatibility but delegates to the generic script
#
# Usage: ./bert_nd_bisect_command.sh [same args as nd_bisect_n300.sh]
#
# Example:
#   ./bert_nd_bisect_command.sh run_bert_func
#   ./bert_nd_bisect_command.sh run_bert_func --target-commit 51fc518f284972f46f32bb1ad77c1e6f535c6a2e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/nd_bisect_n300.sh" "$@"
