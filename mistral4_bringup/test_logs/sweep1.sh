#!/usr/bin/env bash
# Shakeout sweep: newly-added variant-parametrized entries. Cheap ones first.
R=/data/ssalice/temp/tt-metal/mistral4_bringup/test_logs/run.sh
T=models/demos/deepseek_v3_d_p/tests
$R vp_mla_cache            $T/cache/test_mla_cache.py -k mistral4 -vvv --tb=short
$R vp_parallel_embedding   $T/pcc/test_parallel_embedding.py -k mistral4 -vvv --tb=short
$R vp_reduce               $T/op_unit_tests/test_reduce.py -k mistral4 -vvv --tb=short
$R vp_dispatch_combine     $T/op_unit_tests/test_ttnn_dispatch_combine.py -k "mistral4-640-avg" -vvv --tb=short
$R vp_prefill_combine      $T/op_unit_tests/test_prefill_combine.py -k "mistral4-mesh-8x4-pcc" -vvv --tb=short
$R vp_prefill_dispatch     $T/op_unit_tests/test_prefill_dispatch.py -k "mistral4-pcc and mesh-8x4" -vvv --tb=short
echo "SWEEP1 DONE"
