#loc1 = loc("p0.1")
#loc4 = loc("reduce-window.10")
module @SyncTensorsGraph.12 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["x"=1, "y"=1]> loc(#loc)
  func.func @main(%arg0: tensor<1168640xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1168640xi64>>, ttir.name = "args_0"} loc("p0.1")) -> (tensor<1168640xi64> {ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1168640xi64>>}) {
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc)
    %0 = stablehlo.reshape %arg0 : (tensor<1168640xi64>) -> tensor<1x1x1168640xi64> loc(#loc2)
    %1 = stablehlo.reshape %0 : (tensor<1x1x1168640xi64>) -> tensor<1168640xi64> loc(#loc3)
    %2 = "stablehlo.reduce_window"(%1, %c) <{padding = dense<[[1168639, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 1168640>}> ({
    ^bb0(%arg1: tensor<i64> loc("reduce-window.10"), %arg2: tensor<i64> loc("reduce-window.10")):
      %3 = stablehlo.add %arg1, %arg2 : tensor<i64> loc(#loc5)
      stablehlo.return %3 : tensor<i64> loc(#loc)
    }) : (tensor<1168640xi64>, tensor<i64>) -> tensor<1168640xi64> loc(#loc4)
    return %2 : tensor<1168640xi64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("reshape.2")
#loc3 = loc("reshape.4")
#loc5 = loc("add.9")
