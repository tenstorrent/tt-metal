#!/bin/bash

# Test runner for all generated operator tests
# Generated automatically - do not edit manually

export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export ARCH_NAME=wormhole_b0
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

echo "Running comprehensive operator tests..."
echo "Total tests: 42"

# Activate virtual environment
source python_env/bin/activate

# Run all tests and collect results
mkdir -p ssinghal/test_results

echo "Starting test execution..."

echo "(1/42) Testing softplus..."
python_env/bin/python -m pytest ssinghal/tests/test_softplus.py -v --tb=short > ssinghal/test_results/softplus_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/softplus_results.txt"

echo "(2/42) Testing tanh..."
python_env/bin/python -m pytest ssinghal/tests/test_tanh.py -v --tb=short > ssinghal/test_results/tanh_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/tanh_results.txt"

echo "(3/42) Testing Mish..."
python_env/bin/python -m pytest ssinghal/tests/test_mish.py -v --tb=short > ssinghal/test_results/mish_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/mish_results.txt"

echo "(4/42) Testing add..."
python_env/bin/python -m pytest ssinghal/tests/test_add.py -v --tb=short > ssinghal/test_results/add_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/add_results.txt"

echo "(5/42) Testing cat..."
python_env/bin/python -m pytest ssinghal/tests/test_cat.py -v --tb=short > ssinghal/test_results/cat_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/cat_results.txt"

echo "(6/42) Testing leakyrelu..."
python_env/bin/python -m pytest ssinghal/tests/test_leakyrelu.py -v --tb=short > ssinghal/test_results/leakyrelu_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/leakyrelu_results.txt"

echo "(7/42) Testing MaxPool2d..."
python_env/bin/python -m pytest ssinghal/tests/test_maxpool2d.py -v --tb=short > ssinghal/test_results/maxpool2d_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/maxpool2d_results.txt"

echo "(8/42) Testing SiLU..."
python_env/bin/python -m pytest ssinghal/tests/test_silu.py -v --tb=short > ssinghal/test_results/silu_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/silu_results.txt"

echo "(9/42) Testing Concat..."
python_env/bin/python -m pytest ssinghal/tests/test_concat.py -v --tb=short > ssinghal/test_results/concat_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/concat_results.txt"

echo "(10/42) Testing view..."
python_env/bin/python -m pytest ssinghal/tests/test_view.py -v --tb=short > ssinghal/test_results/view_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/view_results.txt"

echo "(11/42) Testing permute..."
python_env/bin/python -m pytest ssinghal/tests/test_permute.py -v --tb=short > ssinghal/test_results/permute_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/permute_results.txt"

echo "(12/42) Testing clone..."
python_env/bin/python -m pytest ssinghal/tests/test_clone.py -v --tb=short > ssinghal/test_results/clone_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/clone_results.txt"

echo "(13/42) Testing sigmoid..."
python_env/bin/python -m pytest ssinghal/tests/test_sigmoid.py -v --tb=short > ssinghal/test_results/sigmoid_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/sigmoid_results.txt"

echo "(14/42) Testing copy..."
python_env/bin/python -m pytest ssinghal/tests/test_copy.py -v --tb=short > ssinghal/test_results/copy_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/copy_results.txt"

echo "(15/42) Testing silu..."
python_env/bin/python -m pytest ssinghal/tests/test_silu.py -v --tb=short > ssinghal/test_results/silu_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/silu_results.txt"

echo "(16/42) Testing splitwithsizes..."
python_env/bin/python -m pytest ssinghal/tests/test_splitwithsizes.py -v --tb=short > ssinghal/test_results/splitwithsizes_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/splitwithsizes_results.txt"

echo "(17/42) Testing softmax..."
python_env/bin/python -m pytest ssinghal/tests/test_softmax.py -v --tb=short > ssinghal/test_results/softmax_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/softmax_results.txt"

echo "(18/42) Testing unsqueeze..."
python_env/bin/python -m pytest ssinghal/tests/test_unsqueeze.py -v --tb=short > ssinghal/test_results/unsqueeze_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/unsqueeze_results.txt"

echo "(19/42) Testing bmm..."
python_env/bin/python -m pytest ssinghal/tests/test_bmm.py -v --tb=short > ssinghal/test_results/bmm_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/bmm_results.txt"

echo "(20/42) Testing addmm..."
python_env/bin/python -m pytest ssinghal/tests/test_addmm.py -v --tb=short > ssinghal/test_results/addmm_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/addmm_results.txt"

echo "(21/42) Testing Linear..."
python_env/bin/python -m pytest ssinghal/tests/test_linear.py -v --tb=short > ssinghal/test_results/linear_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/linear_results.txt"

echo "(22/42) Testing linalgvectornorm..."
python_env/bin/python -m pytest ssinghal/tests/test_linalgvectornorm.py -v --tb=short > ssinghal/test_results/linalgvectornorm_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/linalgvectornorm_results.txt"

echo "(23/42) Testing clampmin..."
python_env/bin/python -m pytest ssinghal/tests/test_clampmin.py -v --tb=short > ssinghal/test_results/clampmin_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/clampmin_results.txt"

echo "(24/42) Testing expand..."
python_env/bin/python -m pytest ssinghal/tests/test_expand.py -v --tb=short > ssinghal/test_results/expand_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/expand_results.txt"

echo "(25/42) Testing unsafeview..."
python_env/bin/python -m pytest ssinghal/tests/test_unsafeview.py -v --tb=short > ssinghal/test_results/unsafeview_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/unsafeview_results.txt"

echo "(26/42) Testing relu..."
python_env/bin/python -m pytest ssinghal/tests/test_relu.py -v --tb=short > ssinghal/test_results/relu_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/relu_results.txt"

echo "(27/42) Testing Dropout..."
python_env/bin/python -m pytest ssinghal/tests/test_dropout.py -v --tb=short > ssinghal/test_results/dropout_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/dropout_results.txt"

echo "(28/42) Testing GELU..."
python_env/bin/python -m pytest ssinghal/tests/test_gelu.py -v --tb=short > ssinghal/test_results/gelu_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/gelu_results.txt"

echo "(29/42) Testing Identity..."
python_env/bin/python -m pytest ssinghal/tests/test_identity.py -v --tb=short > ssinghal/test_results/identity_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/identity_results.txt"

echo "(30/42) Testing topk..."
python_env/bin/python -m pytest ssinghal/tests/test_topk.py -v --tb=short > ssinghal/test_results/topk_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/topk_results.txt"

echo "(31/42) Testing transpose..."
python_env/bin/python -m pytest ssinghal/tests/test_transpose.py -v --tb=short > ssinghal/test_results/transpose_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/transpose_results.txt"

echo "(32/42) Testing div..."
python_env/bin/python -m pytest ssinghal/tests/test_div.py -v --tb=short > ssinghal/test_results/div_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/div_results.txt"

echo "(33/42) Testing mul..."
python_env/bin/python -m pytest ssinghal/tests/test_mul.py -v --tb=short > ssinghal/test_results/mul_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/mul_results.txt"

echo "(34/42) Testing clamp..."
python_env/bin/python -m pytest ssinghal/tests/test_clamp.py -v --tb=short > ssinghal/test_results/clamp_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/clamp_results.txt"

echo "(35/42) Testing log..."
python_env/bin/python -m pytest ssinghal/tests/test_log.py -v --tb=short > ssinghal/test_results/log_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/log_results.txt"

echo "(36/42) Testing stack..."
python_env/bin/python -m pytest ssinghal/tests/test_stack.py -v --tb=short > ssinghal/test_results/stack_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/stack_results.txt"

echo "(37/42) Testing mean..."
python_env/bin/python -m pytest ssinghal/tests/test_mean.py -v --tb=short > ssinghal/test_results/mean_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/mean_results.txt"

echo "(38/42) Testing hardtanh..."
python_env/bin/python -m pytest ssinghal/tests/test_hardtanh.py -v --tb=short > ssinghal/test_results/hardtanh_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/hardtanh_results.txt"

echo "(39/42) Testing Sigmoid..."
python_env/bin/python -m pytest ssinghal/tests/test_sigmoid.py -v --tb=short > ssinghal/test_results/sigmoid_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/sigmoid_results.txt"

echo "(40/42) Testing ReLU..."
python_env/bin/python -m pytest ssinghal/tests/test_relu.py -v --tb=short > ssinghal/test_results/relu_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/relu_results.txt"

echo "(41/42) Testing GELUActivation..."
python_env/bin/python -m pytest ssinghal/tests/test_geluactivation.py -v --tb=short > ssinghal/test_results/geluactivation_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/geluactivation_results.txt"

echo "(42/42) Testing mm..."
python_env/bin/python -m pytest ssinghal/tests/test_mm.py -v --tb=short > ssinghal/test_results/mm_results.txt 2>&1
echo "  Results saved to ssinghal/test_results/mm_results.txt"

echo "All tests completed!"
echo "Results are in ssinghal/test_results/"

# Generate summary
python3 ssinghal/analyze_test_results.py
