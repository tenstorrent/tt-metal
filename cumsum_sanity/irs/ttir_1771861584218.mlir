#loc1 = loc("p0.1")
module @SyncTensorsGraph.12 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.12 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @main(%arg0: tensor<1168640xi64> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1168640xi64>>, ttir.name = "args_0"} loc("p0.1")) -> (tensor<1168640xi64> {ttcore.runtime_tensor_sharding = #ttcore<runtime_tensor_sharding shard_status = <unsharded>, local_shape = tensor<1168640xi64>>}) {
        %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1168640 : i32]}> : (tensor<1168640xi64>) -> tensor<1x1x1168640xi64> loc(#loc2)
        %1 = "ttir.reshape"(%0) <{shape = [1168640 : i32]}> : (tensor<1x1x1168640xi64>) -> tensor<1168640xi64> loc(#loc3)
        %2 = "ttir.cumsum"(%1) <{dim = 0 : i64}> : (tensor<1168640xi64>) -> tensor<1168640xi64> loc(#loc4)
        return %2 : tensor<1168640xi64> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("reshape.2")
#loc3 = loc("reshape.4")
#loc4 = loc("reduce-window.10")
