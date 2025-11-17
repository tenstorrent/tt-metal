#!/bin/bash
#
# Setup environment variables for code coverage collection
# Also installs missing dependencies if needed
#
# Usage:
#   source .github/actions/code-coverage/setup_coverage_env.sh
#   # or
#   . .github/actions/code-coverage/setup_coverage_env.sh
#
# Then run your tests normally:
#   coverage run -m pytest tests/...
#   ./build/test/tt_metal/unit_tests
#

# Don't use set -e here because we want to continue even if some installations fail
# set -e

# Get the repository root directory
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
else
    # Fallback if not sourced from bash
    REPO_ROOT="$(pwd)"
fi

cd "$REPO_ROOT"

echo "=========================================="
echo "Setting up coverage environment..."
echo "=========================================="

# Check and install missing dependencies
MISSING_TOOLS=()

# Check for llvm-profdata
if ! command -v llvm-profdata &> /dev/null; then
    MISSING_TOOLS+=("llvm-profdata")
fi

# Check for llvm-cov
if ! command -v llvm-cov &> /dev/null; then
    MISSING_TOOLS+=("llvm-cov")
fi

# Check for genhtml (from lcov package)
if ! command -v genhtml &> /dev/null; then
    MISSING_TOOLS+=("genhtml")
fi

# Check for Python coverage module
if ! python3 -c "import coverage" 2>/dev/null; then
    MISSING_TOOLS+=("python3-coverage")
fi

# Install missing tools
if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo "Installing missing dependencies..."

    # Check if we can use apt-get (Ubuntu/Debian)
    INSTALL_CMD=""
    SUDO_CMD=""

    # Check if we need sudo (if not root)
    if [ "$EUID" -ne 0 ] && command -v sudo &> /dev/null; then
        SUDO_CMD="sudo"
    fi

    if command -v apt-get &> /dev/null; then
        INSTALL_CMD="$SUDO_CMD apt-get install -y"
    elif command -v yum &> /dev/null; then
        INSTALL_CMD="$SUDO_CMD yum install -y"
    else
        echo "⚠ Warning: Cannot auto-install dependencies. Please install manually:"
        for tool in "${MISSING_TOOLS[@]}"; do
            echo "  - $tool"
        done
    fi

    if [ -n "$INSTALL_CMD" ]; then
        # Try to install LLVM tools
        if [[ " ${MISSING_TOOLS[@]} " =~ " llvm-profdata " ]] || [[ " ${MISSING_TOOLS[@]} " =~ " llvm-cov " ]]; then
            echo "  Installing LLVM tools..."
            # Try different LLVM versions
            INSTALLED=false
            for version in 17 18 19; do
                if $INSTALL_CMD "llvm-${version}-tools" 2>&1 | grep -qE "Setting up|already the newest|Nothing to do|is already"; then
                    echo "    ✓ LLVM tools (llvm-${version}-tools) installed"
                    INSTALLED=true
                    break
                fi
            done
            if [ "$INSTALLED" = false ]; then
                # Try generic package
                if $INSTALL_CMD llvm-tools 2>&1 | grep -qE "Setting up|already the newest|Nothing to do|is already"; then
                    echo "    ✓ LLVM tools (llvm-tools) installed"
                    INSTALLED=true
                fi
            fi
            if [ "$INSTALLED" = false ]; then
                echo "    ⚠ Could not install LLVM tools via package manager"
                echo "      Try manually: $INSTALL_CMD llvm-17-tools"
                echo "      Or check if tools are installed with version suffix: llvm-profdata-17, llvm-cov-17"
            fi
        fi

        # Install lcov for genhtml
        if [[ " ${MISSING_TOOLS[@]} " =~ " genhtml " ]]; then
            echo "  Installing lcov..."
            if $INSTALL_CMD lcov >/dev/null 2>&1; then
                echo "    ✓ lcov installed"
            else
                echo "    ⚠ Could not install lcov"
                echo "      Try manually: $INSTALL_CMD lcov"
            fi
        fi
    fi

    # Install Python coverage (assumes we're in a virtual environment)
    if [[ " ${MISSING_TOOLS[@]} " =~ " python3-coverage " ]]; then
        echo "  Installing Python coverage module..."
        if pip install coverage 2>/dev/null || pip3 install coverage 2>/dev/null || python3 -m pip install coverage 2>/dev/null; then
            echo "    ✓ Python coverage installed"
        else
            echo "    ⚠ Could not install Python coverage"
            echo "      Try: pip install coverage"
        fi
    fi

    # Re-check what's still missing
    STILL_MISSING=()
    if ! command -v llvm-profdata &> /dev/null; then
        STILL_MISSING+=("llvm-profdata")
    fi
    if ! command -v llvm-cov &> /dev/null; then
        STILL_MISSING+=("llvm-cov")
    fi
    if ! command -v genhtml &> /dev/null; then
        STILL_MISSING+=("genhtml")
    fi
    if ! python3 -c "import coverage" 2>/dev/null; then
        STILL_MISSING+=("python3-coverage")
    fi

    if [ ${#STILL_MISSING[@]} -gt 0 ]; then
        echo ""
        echo "⚠ Warning: Some tools are still missing:"
        for tool in "${STILL_MISSING[@]}"; do
            echo "  - $tool"
        done
        echo ""
        echo "Please install them manually:"
        echo "  - LLVM tools: apt-get install llvm-17-tools (or llvm-18-tools)"
        echo "  - lcov: apt-get install lcov"
        echo "  - Python coverage: pip install coverage"
    else
        echo "✓ All dependencies installed successfully"
    fi
    echo ""
fi

# Set coverage output directory (use .github/coverage to keep files organized)
# If COVERAGE_DIR is not set, default to .github/coverage
if [ -z "$COVERAGE_DIR" ]; then
    COVERAGE_DIR=".github/coverage"
fi
mkdir -p "$COVERAGE_DIR"
export LLVM_PROFILE_FILE="$COVERAGE_DIR/%p.profraw"

# Enable watcher to track kernel usage
export TT_METAL_WATCHER_APPEND=1

# IMPORTANT: Preload Clang ASan runtime (REQUIRED for Docker containers and ASanCoverage builds)
# Find Clang ASan runtime
CLANG_ASAN_RUNTIME=""
if [ -f "/usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.asan-x86_64.so" ]; then
    CLANG_ASAN_RUNTIME="/usr/lib/llvm-17/lib/clang/17/lib/linux/libclang_rt.asan-x86_64.so"
elif [ -f "/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so" ]; then
    CLANG_ASAN_RUNTIME="/usr/lib/llvm-18/lib/clang/18/lib/linux/libclang_rt.asan-x86_64.so"
else
    # Try to find it
    CLANG_ASAN_RUNTIME=$(find /usr/lib/llvm-* -name "libclang_rt.asan-x86_64.so" 2>/dev/null | head -1)
fi

# Check if binaries are statically linked with ASan (common for executables)
# If so, we don't need LD_PRELOAD - it will cause "incompatible ASan runtimes" errors
NEEDS_LD_PRELOAD=true
if [ -d "$REPO_ROOT/build_ASanCoverage" ] || [ -d "$REPO_ROOT/build" ]; then
    # Check a sample binary to see if ASan is statically linked
    SAMPLE_BINARY=""
    if [ -f "$REPO_ROOT/build_ASanCoverage/test/tt_metal/test_add_two_ints" ]; then
        SAMPLE_BINARY="$REPO_ROOT/build_ASanCoverage/test/tt_metal/test_add_two_ints"
    elif [ -f "$REPO_ROOT/build/test/tt_metal/test_add_two_ints" ]; then
        SAMPLE_BINARY="$REPO_ROOT/build/test/tt_metal/test_add_two_ints"
    fi

    if [ -n "$SAMPLE_BINARY" ] && command -v nm &> /dev/null; then
        # Check if __asan_init is in the binary (indicates static linking)
        if nm -D "$SAMPLE_BINARY" 2>/dev/null | grep -q "__asan_init"; then
            NEEDS_LD_PRELOAD=false
            echo "✓ Detected statically-linked ASan in binaries - LD_PRELOAD not needed"
        fi
    fi
fi

if [ "$NEEDS_LD_PRELOAD" = true ] && [ -n "$CLANG_ASAN_RUNTIME" ] && [ -f "$CLANG_ASAN_RUNTIME" ]; then
    # Only set LD_PRELOAD if not already set (to avoid overriding user settings)
    # LD_PRELOAD is needed for Python tests that load shared libraries with ASan
    if [ -z "$LD_PRELOAD" ]; then
        export LD_PRELOAD="$CLANG_ASAN_RUNTIME"
        echo "✓ Preloading Clang ASan runtime: $CLANG_ASAN_RUNTIME"
        echo "  (Note: Only needed for Python tests. C++ binaries with static ASan don't need this.)"
    else
        echo "✓ LD_PRELOAD already set: $LD_PRELOAD"
        echo "  (Clang ASan runtime found at: $CLANG_ASAN_RUNTIME, but not overriding existing LD_PRELOAD)"
    fi
elif [ "$NEEDS_LD_PRELOAD" = false ]; then
    # Don't set LD_PRELOAD for statically-linked binaries
    if [ -n "$LD_PRELOAD" ]; then
        echo "⚠ Note: LD_PRELOAD is set but binaries have static ASan - you may want to unset it for C++ binaries"
        echo "  Current LD_PRELOAD: $LD_PRELOAD"
        echo "  For C++ binaries, run: unset LD_PRELOAD"
    fi
else
    echo "⚠ Warning: Clang ASan runtime not found. If you're using ASanCoverage build, you may encounter errors."
    echo "  Searched in: /usr/lib/llvm-*/lib/clang/*/lib/linux/libclang_rt.asan-x86_64.so"
fi

# Set library path to include build directory and Clang runtime directory
if [ -n "$CLANG_ASAN_RUNTIME" ]; then
    CLANG_RUNTIME_DIR="$(dirname "$CLANG_ASAN_RUNTIME")"
    export LD_LIBRARY_PATH="$REPO_ROOT/build/lib:$CLANG_RUNTIME_DIR:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$REPO_ROOT/build/lib:${LD_LIBRARY_PATH}"
fi

# Create coverage directory
mkdir -p "$REPO_ROOT/coverage"

echo "=========================================="
echo "Coverage Environment Setup Complete"
echo "=========================================="
echo "LLVM_PROFILE_FILE: $LLVM_PROFILE_FILE"
echo "TT_METAL_WATCHER_APPEND: $TT_METAL_WATCHER_APPEND"
echo "LD_PRELOAD: ${LD_PRELOAD:-not set}"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Coverage directory: $REPO_ROOT/coverage"
echo ""
echo "Installed tools:"
command -v llvm-profdata &> /dev/null && echo "  ✓ llvm-profdata: $(which llvm-profdata)" || echo "  ✗ llvm-profdata: not found"
command -v llvm-cov &> /dev/null && echo "  ✓ llvm-cov: $(which llvm-cov)" || echo "  ✗ llvm-cov: not found"
command -v genhtml &> /dev/null && echo "  ✓ genhtml: $(which genhtml)" || echo "  ✗ genhtml: not found"
python3 -c "import coverage; print('  ✓ Python coverage:', coverage.__version__)" 2>/dev/null || echo "  ✗ Python coverage: not found"
echo ""
echo "You can now run tests with coverage:"
echo "  coverage run -m pytest tests/..."
echo "  ./build/test/tt_metal/unit_tests"
echo "=========================================="
