#loc1 = loc("p0.1")
#loc2 = loc("p1.3")
module @SyncTensorsGraph.7 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<256x128x3x3xbf16> loc("p0.1"), %arg1: tensor<1x128x256x256xbf16> loc("p1.3")) -> tensor<1x256x128x128xbf16> {
    %0 = stablehlo.custom_call @tt.mark_argument(%arg1) {api_version = 0 : i32, mhlo.frontend_attributes = {ttcore.argument_type = "input", ttir.name = "args_0"}} : (tensor<1x128x256x256xbf16>) -> tensor<1x128x256x256xbf16> loc(#loc3)
    %1 = stablehlo.custom_call @tt.mark_argument(%arg0) {api_version = 0 : i32, mhlo.frontend_attributes = {ttcore.argument_type = "parameter", ttir.name = "l__self___b_conv_weight"}} : (tensor<256x128x3x3xbf16>) -> tensor<256x128x3x3xbf16> loc(#loc4)
    %2 = stablehlo.convolution(%0, %1) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x128x256x256xbf16>, tensor<256x128x3x3xbf16>) -> tensor<1x256x128x128xbf16> loc(#loc5)
    return %2 : tensor<1x256x128x128xbf16> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc3 = loc("custom-call.4")
#loc4 = loc("custom-call.2")
#loc5 = loc("convolution.5")
