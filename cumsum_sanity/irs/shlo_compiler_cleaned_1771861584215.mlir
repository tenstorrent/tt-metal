#loc = loc(unknown)
module @SyncTensorsGraph.12 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1168640xi64> loc(unknown)) -> tensor<1168640xi64> {
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc)
    %0 = stablehlo.reshape %arg0 : (tensor<1168640xi64>) -> tensor<1x1x1168640xi64> loc(#loc)
    %1 = stablehlo.reshape %0 : (tensor<1x1x1168640xi64>) -> tensor<1168640xi64> loc(#loc)
    %2 = "stablehlo.reduce_window"(%1, %c) <{padding = dense<[[1168639, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 1168640>}> ({
    ^bb0(%arg1: tensor<i64> loc(unknown), %arg2: tensor<i64> loc(unknown)):
      %3 = stablehlo.add %arg1, %arg2 : tensor<i64> loc(#loc)
      stablehlo.return %3 : tensor<i64> loc(#loc)
    }) : (tensor<1168640xi64>, tensor<i64>) -> tensor<1168640xi64> loc(#loc)
    return %2 : tensor<1168640xi64> loc(#loc)
  } loc(#loc)
} loc(#loc)
