## Distilbert Model Data Parallelisation

# Platforms:
    WH N300

# Details
The data parallel implementation of the DistilBERT model is provided in the file `models/demos/distilbert/tt/ttnn_optimized_distilbert.py`. The entry point for this implementation is the `distilbert_for_question_answering` function.
In this implementation, the model's inputs and weights are processed, converted to TTNN , and placed on the mesh.

# Perf files
The end-to-end performance and device performance pipelines are enabled in `models/demos/distilbert/tests/test_perf_distilbert.py`.

To address the `Cannot mix single and multi-device tensors when calling launch op!` issue, the following changes were made to `ttnn/cpp/ttnn/operations/matmul/matmul.hpp`:

converted
`constexpr auto matmul = ttnn::register_operation<"ttnn::matmul", operations::matmul::MatmulOperation>();`
`constexpr auto linear = ttnn::register_operation<"ttnn::linear", operations::matmul::LinearOperation>();`
to

`constexpr auto matmul = ttnn::register_operation_with_auto_launch_op<"ttnn::matmul", operations::matmul::MatmulOperation>();`
`constexpr auto linear = ttnn::register_operation_with_auto_launch_op<"ttnn::linear", operations::matmul::LinearOperation>();`

# Steps to generate the perf file
Fetch `sudharsan/ttnn_distilbert_data_parallel`
Active the environment variables
Build using `./scripts/build_scripts/build_with_profiler_opt.sh`
pytest models/demos/distilbert/tests/test_perf_distilbert.py::test_distilbert_perf_device

# Steps to reproduce the error
Fetch `sudharsan/ttnn_distilbert_data_parallel`
pytest `/home/ubuntu/sudharsan/tt-metal/tests/ttnn/integration_tests/distilbert/test_ttnn_distilbert.py`

# Expected Result

`Cannot mix single and multi-device tensors when calling launch op!`

Note: The above mentioned error is thrown from both linear and matmul op.
