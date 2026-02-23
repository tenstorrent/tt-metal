#loc1 = loc("p0.1")
#loc5 = loc("reduce-window.10")
module @SyncTensorsGraph.12 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  vhlo.func_v1 @main(%arg0: !vhlo.tensor_v1<1168640x!vhlo.i64_v1> loc("p0.1")) -> (!vhlo.tensor_v1<1168640x!vhlo.i64_v1>) {
    %0 = "vhlo.constant_v1"() <{value = #vhlo.tensor_v1<dense<0> : tensor<i64>>}> : () -> !vhlo.tensor_v1<!vhlo.i64_v1> loc(#loc)
    %1 = "vhlo.reshape_v1"(%arg0) : (!vhlo.tensor_v1<1168640x!vhlo.i64_v1>) -> !vhlo.tensor_v1<1x1x1168640x!vhlo.i64_v1> loc(#loc2)
    %2 = "vhlo.custom_call_v1"(%1) <{api_version = #vhlo<api_version_v1 API_VERSION_UNSPECIFIED>, backend_config = #vhlo.string_v1<"">, call_target_name = #vhlo.string_v1<"tt.mark_argument">, called_computations = #vhlo.array_v1<[]>, has_side_effect = #vhlo.bool_v1<false>, operand_layouts = #vhlo.array_v1<[]>, output_operand_aliases = #vhlo.array_v1<[]>, result_layouts = #vhlo.array_v1<[]>}> {mhlo.frontend_attributes = #vhlo.dict_v1<{#vhlo.string_v1<"ttcore.argument_type"> = #vhlo.string_v1<"input">, #vhlo.string_v1<"ttir.name"> = #vhlo.string_v1<"args_0">}>} : (!vhlo.tensor_v1<1x1x1168640x!vhlo.i64_v1>) -> !vhlo.tensor_v1<1x1x1168640x!vhlo.i64_v1> loc(#loc3)
    %3 = "vhlo.reshape_v1"(%2) : (!vhlo.tensor_v1<1x1x1168640x!vhlo.i64_v1>) -> !vhlo.tensor_v1<1168640x!vhlo.i64_v1> loc(#loc4)
    %4 = "vhlo.reduce_window_v1"(%3, %0) <{base_dilations = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>, padding = #vhlo.tensor_v1<dense<[[1168639, 0]]> : tensor<1x2xi64>>, window_dilations = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>, window_dimensions = #vhlo.tensor_v1<dense<1168640> : tensor<1xi64>>, window_strides = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>}> ({
    ^bb0(%arg1: !vhlo.tensor_v1<!vhlo.i64_v1> loc("reduce-window.10"), %arg2: !vhlo.tensor_v1<!vhlo.i64_v1> loc("reduce-window.10")):
      %5 = "vhlo.add_v1"(%arg1, %arg2) : (!vhlo.tensor_v1<!vhlo.i64_v1>, !vhlo.tensor_v1<!vhlo.i64_v1>) -> !vhlo.tensor_v1<!vhlo.i64_v1> loc(#loc6)
      "vhlo.return_v1"(%5) : (!vhlo.tensor_v1<!vhlo.i64_v1>) -> () loc(#loc)
    }) : (!vhlo.tensor_v1<1168640x!vhlo.i64_v1>, !vhlo.tensor_v1<!vhlo.i64_v1>) -> !vhlo.tensor_v1<1168640x!vhlo.i64_v1> loc(#loc5)
    "vhlo.return_v1"(%4) : (!vhlo.tensor_v1<1168640x!vhlo.i64_v1>) -> () loc(#loc)
  } {arg_attrs = #vhlo.array_v1<[]>, res_attrs = #vhlo.array_v1<[]>, sym_visibility = #vhlo.string_v1<"">} loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("reshape.2")
#loc3 = loc("custom-call.3")
#loc4 = loc("reshape.4")
#loc6 = loc("add.9")
