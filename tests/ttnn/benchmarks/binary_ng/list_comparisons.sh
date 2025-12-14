#!/bin/bash
# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Helper script to list and manage comparison directories

COMPARISONS_DIR="comparisons"

echo "════════════════════════════════════════════════════════════════════"
echo "  COMPARISON HISTORY"
echo "════════════════════════════════════════════════════════════════════"
echo ""

if [ ! -d "$COMPARISONS_DIR" ]; then
    echo "❌ No comparisons directory found."
    echo "   Run: python compare_multi_strategy.py max_ab max_abc"
    exit 0
fi

# Count comparisons
TOTAL=$(find "$COMPARISONS_DIR" -maxdepth 1 -type d -name "comparison_*" | wc -l)

if [ "$TOTAL" -eq 0 ]; then
    echo "❌ No comparisons found."
    echo "   Run: python compare_multi_strategy.py max_ab max_abc"
    exit 0
fi

echo "📊 Total Comparisons: $TOTAL"
echo ""
echo "Latest comparisons (most recent first):"
echo "────────────────────────────────────────────────────────────────────"

# List comparisons with details
find "$COMPARISONS_DIR" -maxdepth 1 -type d -name "comparison_*" | \
    sort -r | \
    head -20 | \
    while read -r dir; do
        # Extract strategies from directory name
        basename_dir=$(basename "$dir")

        # Get timestamp
        timestamp=$(echo "$basename_dir" | grep -oE '[0-9]{8}_[0-9]{6}' || echo "custom")

        # Format timestamp
        if [ "$timestamp" != "custom" ]; then
            date_part=$(echo "$timestamp" | cut -d_ -f1)
            time_part=$(echo "$timestamp" | cut -d_ -f2)
            formatted_date="${date_part:0:4}-${date_part:4:2}-${date_part:6:2}"
            formatted_time="${time_part:0:2}:${time_part:2:2}:${time_part:4:2}"
            when="$formatted_date $formatted_time"
        else
            when="Custom name"
        fi

        # Get strategies
        strategies=$(echo "$basename_dir" | sed 's/comparison_//; s/_vs_/ vs /g; s/_[0-9]\{8\}_[0-9]\{6\}$//')

        # Count files
        file_count=$(find "$dir" -type f | wc -l)

        # Get directory size
        size=$(du -sh "$dir" | cut -f1)

        # Check if summary exists and extract key info
        summary_file="$dir/comparison_summary.txt"
        if [ -f "$summary_file" ]; then
            # Extract winner if it's a 3-way comparison
            winner=$(grep "🏆 WINNER DISTRIBUTION" -A 2 "$summary_file" | tail -1 | awk '{print $1}' || echo "")
        else
            winner=""
        fi

        echo ""
        echo "📁 $basename_dir"
        echo "   Strategies: $strategies"
        echo "   When:       $when"
        echo "   Files:      $file_count files ($size)"
        if [ -n "$winner" ] && [ "$winner" != "🏆" ]; then
            echo "   Winner:     $winner ⭐"
        fi
        echo "   Path:       $dir"
    done

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "💡 Quick Commands:"
echo ""
echo "  # View latest comparison summary"
echo "  cat \$(ls -t comparisons/comparison_*/comparison_summary.txt | head -1) | head -50"
echo ""
echo "  # Open latest comparison directory"
echo "  cd \$(ls -td comparisons/comparison_*/ | head -1)"
echo ""
echo "  # List all PNG charts from latest"
echo "  ls -lh \$(ls -td comparisons/comparison_*/ | head -1)*.png"
echo ""
echo "  # Compare specific strategies again"
echo "  python compare_multi_strategy.py max_ab max_abc full_grid"
echo ""
echo "════════════════════════════════════════════════════════════════════"
