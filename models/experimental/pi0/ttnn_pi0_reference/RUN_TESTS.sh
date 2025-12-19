#!/bin/bash
# Quick test runner for ttnn_pi0_reference

echo "======================================================================"
echo "  TTNN PI0 Reference - Quick Test"
echo "======================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "pcc_test_standalone.py" ]; then
    echo "❌ Error: Must run from ttnn_pi0_reference directory"
    exit 1
fi

# Run the standalone PCC test
echo "Running PCC tests..."
echo ""
python3 pcc_test_standalone.py

exit_code=$?

echo ""
echo "======================================================================"
if [ $exit_code -eq 0 ]; then
    echo "  ✅ Tests completed successfully!"
else
    echo "  ❌ Tests failed with exit code $exit_code"
fi
echo "======================================================================"
echo ""
echo "For more tests, see TESTING_GUIDE.md"
echo ""

exit $exit_code
