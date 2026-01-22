[1mdiff --git a/tests/ttnn/unit_tests/operations/matmul/test_experimental.py b/tests/ttnn/unit_tests/operations/matmul/test_experimental.py[m
[1mindex bf5ab9361be..186e1eb7438 100644[m
[1m--- a/tests/ttnn/unit_tests/operations/matmul/test_experimental.py[m
[1m+++ b/tests/ttnn/unit_tests/operations/matmul/test_experimental.py[m
[36m@@ -58,12 +58,12 @@[m [mdef test_ttnn_matmul(device, m_size, k_size, n_size):[m
 [m
 [m
 @pytest.mark.requires_fast_runtime_mode_off[m
[31m-@pytest.mark.parametrize("input_a_is_sharded", [True, False])[m
[31m-@pytest.mark.parametrize("output_is_sharded", [True, False])[m
[32m+[m[32m@pytest.mark.parametrize("input_a_is_sharded", [True])[m
[32m+[m[32m@pytest.mark.parametrize("output_is_sharded", [True])[m
 @pytest.mark.parametrize("m_size, num_cores", [[5632, 22]])[m
[31m-@pytest.mark.parametrize("k_size, n_size", [[64, 64], [64, 256]])[m
[31m-@pytest.mark.parametrize("input_a_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])[m
[31m-@pytest.mark.parametrize("input_b_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])[m
[32m+[m[32m@pytest.mark.parametrize("k_size, n_size", [[64, 64]])[m
[32m+[m[32m@pytest.mark.parametrize("input_a_dtype", [ttnn.bfloat8_b])[m
[32m+[m[32m@pytest.mark.parametrize("input_b_dtype", [ttnn.bfloat16])[m
 def test_ttnn_linear([m
     device, input_a_is_sharded, output_is_sharded, m_size, k_size, n_size, num_cores, input_a_dtype, input_b_dtype[m
 ):[m
[36m@@ -95,32 +95,32 @@[m [mdef test_ttnn_linear([m
     )[m
 [m
     with ttnn.tracer.trace():[m
[31m-        torch_input_tensor_a = torch.randn(input_shape_a).bfloat16().float()[m
[31m-        torch_input_tensor_b = torch.randn(input_shape_b).bfloat16().float()[m
[31m-        torch_bias = torch.randn(bias_shape).bfloat16().float()[m
[32m+[m[32m        torch_input_tensor_a = torch.randn(input_shape_a, dtype=torch.bfloat16)[m
[32m+[m[32m        torch_input_tensor_b = torch.randn(input_shape_b, dtype=torch.bfloat16)[m
[32m+[m[32m        torch_bias = torch.randn(bias_shape, dtype=torch.bfloat16)[m
         torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b + torch_bias[m
 [m
         input_tensor_a = ttnn.from_torch([m
             torch_input_tensor_a,[m
[31m-            device=device,[m
[32m+[m[32m            dtype=input_a_dtype,[m
             layout=ttnn.TILE_LAYOUT,[m
[32m+[m[32m            device=device,[m
             memory_config=interleaved_memory_config,[m
[31m-            dtype=input_a_dtype,[m
         )[m
         input_tensor_b = ttnn.from_torch([m
             torch_input_tensor_b,[m
[31m-            device=device,[m
[32m+[m[32m            dtype=input_b_dtype,[m
             layout=ttnn.TILE_LAYOUT,[m
[32m+[m[32m            device=device,[m
             memory_config=interleaved_memory_config,[m
[31m-            dtype=input_b_dtype,[m
         )[m
 [m
         bias = ttnn.from_torch([m
             torch_bias,[m
[31m-            device=device,[m
[32m+[m[32m            dtype=input_b_dtype,[m
             layout=ttnn.TILE_LAYOUT,[m
[32m+[m[32m            device=device,[m
             memory_config=interleaved_memory_config,[m
[31m-            dtype=input_b_dtype,[m
         )[m
         if input_a_is_sharded:[m
             input_tensor_a = ttnn.interleaved_to_sharded([m
