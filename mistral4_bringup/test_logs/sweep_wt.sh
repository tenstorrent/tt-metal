#!/usr/bin/env bash
# Authoritative sweep on ssalice/mistral4-tests. Cheapest first so partial results are useful.
R=/data/ssalice/temp/tt-metal/mistral4_bringup/test_logs/run_wt.sh
T=models/demos/deepseek_v3_d_p/tests
$R kv_cache_table  $T/test_kv_cache_table.py::test_mistral4_kv_cache_table -vvv --tb=short
$R mla_8x4         $T/test_mla.py::test_mistral4_mla -k "8x4" -vvv --tb=short
$R moe_pcc         $T/pcc/test_ttnn_moe.py::test_mistral4_moe -vvv --tb=short
$R prefill_block   $T/test_prefill_block.py::test_mistral4_prefill_block -vvv --tb=short
$R mla_chunked     $T/test_mla.py::test_mla_chunked_prefill -k "mistral4 and 8x4 and scalar and no_determinism and cpu and (plain-5k or rot-aligned_min or rot-midchip_straddle)" -vvv --tb=short
$R transformer_2L  $T/test_prefill_transformer.py::test_mistral4_prefill_transformer -k "2_layers" -vvv --tb=short
echo "SWEEP_WT DONE"
