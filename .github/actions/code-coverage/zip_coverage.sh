#!/bin/bash
#
# Script to create a zip file of the coverage report for easy transfer
#
# Usage:
#   .github/actions/code-coverage/zip_coverage.sh [output_zip_path]
#
# Example:
#   .github/actions/code-coverage/zip_coverage.sh
#   # Creates: .github/coverage_report.zip
#
#   .github/actions/code-coverage/zip_coverage.sh ~/coverage.zip
#   # Creates: ~/coverage.zip

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

# Default coverage directory
COVERAGE_DIR="${COVERAGE_DIR:-.github/coverage}"
HTML_DIR="$COVERAGE_DIR/html"

# Output archive file path (default: coverage_report.tar.gz in coverage directory)
if [ -n "$1" ]; then
    OUTPUT_ZIP="$1"
else
    OUTPUT_ZIP="$COVERAGE_DIR/coverage_report.tar.gz"
fi

# Check if coverage HTML directory exists
if [ ! -d "$HTML_DIR" ]; then
    echo "Error: Coverage HTML directory not found: $HTML_DIR"
    echo "  Run the coverage script first to generate the report."
    exit 1
fi

echo "Creating compressed archive of coverage report..."
echo "  Source: $HTML_DIR"
echo "  Output: $OUTPUT_ZIP"

# Determine output format and command
# Remove .zip extension if present, we'll add the appropriate one
OUTPUT_BASE="${OUTPUT_ZIP%.zip}"
OUTPUT_BASE="${OUTPUT_BASE%.tar.gz}"
OUTPUT_BASE="${OUTPUT_BASE%.tgz}"

# Make output path absolute
if [[ "$OUTPUT_BASE" != /* ]]; then
    OUTPUT_BASE="$REPO_ROOT/$OUTPUT_BASE"
fi

# Try zip first, fall back to tar.gz
cd "$COVERAGE_DIR"
if command -v zip &> /dev/null; then
    OUTPUT_FILE="${OUTPUT_BASE}.zip"
    zip -r "$OUTPUT_FILE" html/ -q
    EXTRACT_CMD="unzip"
elif command -v tar &> /dev/null; then
    OUTPUT_FILE="${OUTPUT_BASE}.tar.gz"
    tar -czf "$OUTPUT_FILE" html/
    EXTRACT_CMD="tar -xzf"
else
    echo "Error: Neither 'zip' nor 'tar' command found."
    echo "  Install one of them:"
    echo "    apt-get install zip"
    echo "    # or tar should already be installed"
    exit 1
fi

# Get file size
ARCHIVE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)

echo "✓ Created archive: $OUTPUT_FILE"
echo "  Size: $ARCHIVE_SIZE"
echo ""
echo "To copy to your MacBook, run from your MacBook terminal:"
ARCHIVE_REL_PATH=$(realpath --relative-to="$REPO_ROOT" "$OUTPUT_FILE" 2>/dev/null || echo "$OUTPUT_FILE")
ARCHIVE_EXT="${OUTPUT_FILE##*.}"
if [[ "$OUTPUT_FILE" == *.tar.gz ]]; then
    ARCHIVE_EXT="tar.gz"
fi
echo "  scp <vm_username>@<vm_hostname_or_ip>:$ARCHIVE_REL_PATH ~/coverage_report.$ARCHIVE_EXT"
echo ""
echo "Then extract and open:"
if [[ "$OUTPUT_FILE" == *.zip ]]; then
    echo "  unzip ~/coverage_report.zip -d ~/coverage_report"
else
    echo "  tar -xzf ~/coverage_report.tar.gz -C ~/coverage_report"
fi
echo "  open ~/coverage_report/html/index.html"
