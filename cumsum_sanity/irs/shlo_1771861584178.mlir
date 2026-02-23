#loc1 = loc("p0.1")
#loc5 = loc("reduce-window.10")
module @SyncTensorsGraph.12 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<1168640xi64> loc("p0.1")) -> tensor<1168640xi64> {
    %c = stablehlo.constant dense<0> : tensor<i64> loc(#loc)
    %0 = stablehlo.reshape %arg0 : (tensor<1168640xi64>) -> tensor<1x1x1168640xi64> loc(#loc2)
    %1 = stablehlo.custom_call @tt.mark_argument(%0) {api_version = 0 : i32, mhlo.frontend_attributes = {ttcore.argument_type = "input", ttir.name = "args_0"}} : (tensor<1x1x1168640xi64>) -> tensor<1x1x1168640xi64> loc(#loc3)
    %2 = stablehlo.reshape %1 : (tensor<1x1x1168640xi64>) -> tensor<1168640xi64> loc(#loc4)
    %3 = "stablehlo.reduce_window"(%2, %c) <{padding = dense<[[1168639, 0]]> : tensor<1x2xi64>, window_dimensions = array<i64: 1168640>}> ({
    ^bb0(%arg1: tensor<i64> loc("reduce-window.10"), %arg2: tensor<i64> loc("reduce-window.10")):
      %4 = stablehlo.add %arg1, %arg2 : tensor<i64> loc(#loc6)
      stablehlo.return %4 : tensor<i64> loc(#loc)
    }) : (tensor<1168640xi64>, tensor<i64>) -> tensor<1168640xi64> loc(#loc5)
    return %3 : tensor<1168640xi64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("reshape.2")
#loc3 = loc("custom-call.3")
#loc4 = loc("reshape.4")
#loc6 = loc("add.9")
