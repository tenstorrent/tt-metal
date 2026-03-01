#!/bin/bash

echo "========================================="
echo "Verifying MoE Configuration Parity"
echo "========================================="

echo ""
echo "1. CCL max_links Configuration:"
echo "---------------------------------"
echo "Reference (models/demos/deepseek_v3/tt/ccl.py):"
grep -A1 "def get_max_links" ../../demos/deepseek_v3/tt/ccl.py | tail -20 | head -10

echo ""
echo "Copied (models/tt-moe/deepseek_reference/ccl.py):"
grep -A1 "def get_max_links" deepseek_reference/ccl.py | tail -20 | head -10

echo ""
echo "2. MoE num_links Configuration:"
echo "---------------------------------"
echo "Reference (models/demos/deepseek_v3/tt/moe.py):"
grep -B2 -A2 'num_links' ../../demos/deepseek_v3/tt/moe.py | grep -E "(all_to_all|num_links)" | head -4

echo ""
echo "Copied (models/tt-moe/deepseek_reference/moe.py):"
grep -B2 -A2 'num_links' deepseek_reference/moe.py | grep -E "(all_to_all|num_links)" | head -4

echo ""
echo "3. Topology Configuration:"
echo "---------------------------------"
echo "Reference topology lines:"
grep -n "topology=ttnn.Topology.Linear" ../../demos/deepseek_v3/tt/moe.py | wc -l
echo "Copied topology lines:"
grep -n "topology=ttnn.Topology.Linear" deepseek_reference/moe.py | wc -l

echo ""
echo "========================================="
echo "Summary:"
echo "========================================="

# Check if configurations match
ref_links=$(grep -A5 "def get_max_links" ../../demos/deepseek_v3/tt/ccl.py | grep "return" | head -1 | sed 's/[^0-9]*//g')
copy_links=$(grep -A5 "def get_max_links" deepseek_reference/ccl.py | grep "return" | head -1 | sed 's/[^0-9]*//g')

if [ "$ref_links" = "$copy_links" ]; then
    echo "✓ CCL max_links match: both return $ref_links"
else
    echo "✗ CCL max_links differ: reference=$ref_links, copied=$copy_links"
fi

# Check num_links in MoE
ref_num_links=$(grep '"num_links":' ../../demos/deepseek_v3/tt/moe.py | head -1 | sed 's/.*: //;s/,//')
copy_num_links=$(grep '"num_links":' deepseek_reference/moe.py | head -1 | sed 's/.*: //;s/,//')

if [ "$ref_num_links" = "$copy_num_links" ]; then
    echo "✓ MoE num_links match: both use $ref_num_links"
else
    echo "✗ MoE num_links differ: reference=$ref_num_links, copied=$copy_num_links"
fi

echo "========================================="
