#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

cd $TT_METAL_HOME
export PYTHONPATH=$TT_METAL_HOME

# Run one of each component tests for prefill
pytest models/demos/t3000/falcon40b/tests/test_falcon_mlp.py::test_FalconMLP_inference[BFLOAT8_B-DRAM-falcon_40b-prefill_seq128-8chips]
pytest models/demos/t3000/falcon40b/tests/test_falcon_attention.py::test_FalconAttention_inference[BFLOAT8_B-DRAM-falcon_40b-prefill_seq128-8chips]
pytest models/demos/t3000/falcon40b/tests/test_falcon_decoder.py::test_FalconDecoder_inference[BFLOAT8_B-DRAM-falcon_40b-layer_0-prefill_seq128-8chips-disable_program_cache]
pytest models/demos/t3000/falcon40b/tests/test_falcon_model.py::test_FalconModel_inference[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq128-8chips]
pytest models/demos/t3000/falcon40b/tests/test_falcon_causallm.py::test_FalconCausalLM_inference[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq128-8chips]
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq128-8chips-disable_program_cache]

# Run prefill perf tests
pytest models/demos/t3000/falcon40b/tests/test_perf_e2e_falcon.py::test_perf_bare_metal[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq128-8chips]
pytest models/demos/t3000/falcon40b/tests/test_perf_falcon.py::test_perf_bare_metal[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq128-8chips]


# Run one of each component tests for decode
pytest models/demos/t3000/falcon40b/tests/test_falcon_mlp.py::test_FalconMLP_inference[BFLOAT8_B-SHARDED-falcon_40b-decode_batch32-8chips]
pytest models/demos/t3000/falcon40b/tests/test_falcon_attention.py::test_FalconAttention_inference[BFLOAT8_B-SHARDED-falcon_40b-decode_batch32-8chips]
pytest models/demos/t3000/falcon40b/tests/test_falcon_decoder.py::test_FalconDecoder_inference[BFLOAT8_B-SHARDED-falcon_40b-layer_0-decode_batch32-8chips-disable_program_cache]
pytest models/demos/t3000/falcon40b/tests/test_falcon_model.py::test_FalconModel_inference[BFLOAT8_B-SHARDED-falcon_40b-layers_1-decode_batch32-8chips]
pytest models/demos/t3000/falcon40b/tests/test_falcon_causallm.py::test_FalconCausalLM_inference[BFLOAT8_B-SHARDED-falcon_40b-layers_1-decode_batch32-8chips]
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-SHARDED-falcon_40b-layers_1-decode_batch32-8chips-disable_program_cache]

# Run a 4 chip test for decode
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-SHARDED-falcon_40b-layers_1-decode_batch32-4chips-disable_program_cache]

# Run prefill with sequence lengths = {32, 64, 128 (done above), 256, 512, 1024, 2048}
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq32-8chips-disable_program_cache]
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq64-8chips-disable_program_cache]
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq256-8chips-disable_program_cache]
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq512-8chips-disable_program_cache]
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq1024-8chips-disable_program_cache]
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-DRAM-falcon_40b-layers_1-prefill_seq2048-8chips-disable_program_cache]

# Run prefill S=128 with 60L
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-DRAM-falcon_40b-layers_60-prefill_seq128-8chips-disable_program_cache]

# Run prefill S=2048 with 60L
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-DRAM-falcon_40b-layers_60-prefill_seq2048-8chips-disable_program_cache]
