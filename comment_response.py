Comment = r"""

So right now we have a model that is composed of layers, each layer has it's own forward function:
it boils down to this in principle:

class model
def forward(metadata_arg):
  for i in range(num_layer):
  layers[i].forward(metadata_arg)

class layer:
def forward (metadata_arg):
  ttnn.matmul()
  ttnn.add()
  ttnn.op_with_metadata_arg(metadata_arg)

metadata_arg is noted here as an example of a runtime arg that need trace patching

and then in the path that calls model, we do:
  model.forward(metadata_arg)

Rignt now (traced runtime args aside), we do:

model.forward() // compile pass
ttnn.begin_trace_capture()
model.forward() // capture trace
trace_id = ttnn.end_trace_capture()

for i in range (num_model_invocations):
  ttnn.execute_trace(trace_id)

how would the trace capture look like under new APIs?
would we have to add builder.build/add to all of our ops ?
there are also cases where we want to run model untraced, for debug purposes, current trace API
allows us to do so by simply not calling trace capture and just calling model_fwd() ->
are we going to be able to do the same here?
"""

# mesh_device is supplied by the application or test fixture.
# Omit the builder for an untraced call.
model.forward(input_tensor, metadata_arg=2)

# Pass the builder to enable tracing.
builder = ttnn.MeshTraceBuilder(mesh_device)
model.forward(input_tensor, metadata_arg=2, builder=builder)
trace_cq0 = builder.build(cq_id=0)

for i in range(num_model_invocations):
    new_input_tensor = ttnn.from_torch(torch.randn(x, y, z), device=mesh_device, layout=ttnn.TILE_LAYOUT)
    trace_cq0.update_args(
        {
            "input_tensor": new_input_tensor,
            "metadata_arg_layer_0": i,
            "metadata_arg_layer_1": i + 1,
        }
    )
    trace_cq0.replay(blocking=False)

ttnn.synchronize_device(mesh_device)
trace_cq0.deallocate()


# The model and layer code would look like this to support trace:
class model:
    def forward(self, input_tensor, metadata_arg, builder=None):
        input_tensor_param = ttnn.TraceParam(default=input_tensor, name="input_tensor")
        # The first input tensor becomes a trace parameter
        output = input_tensor_param
        for i in range(num_layer):
            metadata_param = ttnn.TraceParam(default=metadata_arg, name=f"metadata_arg_layer_{i}")
            output = layers[i].forward(output, metadata_param, builder)
        return output


class layer:
    def forward(self, input_tensor, metadata_param, builder=None):
        # A call wrapper would be needed to support both traced and untraced calls
        output = call_op(builder, ttnn.matmul, input_tensor, self.weight)
        output = call_op(builder, ttnn.add, output, self.bias)
        # call_op unwraps TraceParam to its default value when no builder is provided.
        return call_op(builder, ttnn.op_with_metadata_arg, output, metadata_param)


# Helper function to support both traced and untraced calls
def call_op(builder, op, *args, **kwargs):
    if builder is None:
        # If no builder is provided, we unwrap the trace params to their default value
        unwrap = lambda value: (value.default if isinstance(value, ttnn.TraceParam) else value)
        args = tuple(unwrap(value) for value in args)
        kwargs = {name: unwrap(value) for name, value in kwargs.items()}
        # call the op directly
        return op(*args, **kwargs)

    return builder.add(op, *args, **kwargs)


# -------- With transparrent trace --------
# No active builder: TraceParam arguments are unwrapped to their default values.
model.forward(input_tensor, metadata_arg=2)

# The active builder is owned by TTNN, so it does not need to be threaded through
# the model. TTNN operations forward TraceParam bindings to the enqueue path.
ttnn.begin_trace_build(mesh_device)
model.forward(input_tensor, metadata_arg=2)
ttnn.pause_trace_build()  # Switch to untraced mode; the builder remains alive.
trace_cq0 = ttnn.build_trace(cq_id=0)

for i in range(num_model_invocations):
    new_input_tensor = ttnn.from_torch(
        torch.randn(x, y, z),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )
    trace_cq0.update_args(
        {
            "input_tensor": new_input_tensor,
            "metadata_arg_layer_0": i,
            "metadata_arg_layer_1": i + 1,
        }
    )
    trace_cq0.replay(blocking=False)

ttnn.synchronize_device(mesh_device)
trace_cq0.deallocate()


class model:
    def forward(self, input_tensor, metadata_arg):
        input_tensor_param = ttnn.TraceParam(default=input_tensor, name="input_tensor")
        output = input_tensor_param
        for i in range(num_layer):
            metadata_param = ttnn.TraceParam(default=metadata_arg, name=f"metadata_arg_layer_{i}")
            output = layers[i].forward(output, metadata_param)
        return output


class layer:
    def forward(self, input_tensor, metadata_arg):
        # TTNN ops accept TraceParam for supported arguments. Without an active
        # builder they use its default value. With an active builder they also
        # pass the parameter binding to the C++ enqueue path.
        output = ttnn.matmul(input_tensor, self.weight)
        output = ttnn.add(output, self.bias)
        return ttnn.op_with_metadata_arg(output, metadata_arg)


# -------- Current matmul call path --------
MATMUL_RELEVANT_CODE = r"""
// ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp

ttnn::bind_function<"matmul">(
    mod,
    /* docstring omitted */,
    ttnn::overload_t(
        &matmul,
        nb::arg("input_tensor_a"),
        nb::arg("input_tensor_b"),
        nb::kw_only(),
        nb::arg("transpose_a") = false,
        nb::arg("transpose_b") = false,
        nb::arg("memory_config") = nb::none(),
        nb::arg("dtype") = nb::none(),
        nb::arg("program_config") = nb::none(),
        nb::arg("activation") = nb::none(),
        nb::arg("compute_kernel_config") = nb::none(),
        nb::arg("core_grid") = nb::none(),
        nb::arg("output_tile") = nb::none(),
        nb::arg("optional_output_tensor") = nb::none(),
        nb::arg("global_cb") = nb::none(),
        nb::arg("sub_device_id") = nb::none()));


// ttnn/cpp/ttnn/operations/matmul/matmul.hpp

Tensor matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    bool transpose_a = false,
    bool transpose_b = false,
    const std::optional<const MemoryConfig>& memory_config = std::nullopt,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const MatmulProgramConfig>& program_config = std::nullopt,
    const std::optional<const Activation>& activation = std::nullopt,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    std::optional<const CoreGrid> core_grid = std::nullopt,
    const std::optional<const tt::tt_metal::Tile>& output_tile = std::nullopt,
    std::optional<Tensor> optional_output_tensor = std::nullopt,
    const std::optional<const GlobalCircularBuffer>& global_cb = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt);


// ttnn/cpp/ttnn/operations/matmul/matmul.cpp

Tensor matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const MemoryConfig>& memory_config,
    const std::optional<const DataType> dtype,
    const std::optional<const MatmulProgramConfig>& program_config,
    const std::optional<const Activation>& activation,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<const CoreGrid> core_grid,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    std::optional<Tensor> optional_output_tensor,
    const std::optional<const GlobalCircularBuffer>& global_cb,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    std::optional<CoreCoord> user_core_coord;
    if (core_grid.has_value()) {
        user_core_coord = CoreCoord(core_grid->x, core_grid->y);
    }
    bool user_run_batched = detail::is_input_batched(input_tensor_b.logical_shape());
    const bool untilize_out =
        program_config.has_value() &&
                std::holds_alternative<MatmulMultiCoreReuseMultiCast1DProgramConfig>(
                    program_config.value())
            ? std::get<MatmulMultiCoreReuseMultiCast1DProgramConfig>(
                  program_config.value())
                  .untilize_out
            : false;
    auto matmul_params = ttnn::prim::MatmulParams{
        program_config,
        /*bcast_batch=*/std::nullopt,
        memory_config.has_value() ? memory_config.value() : ttnn::DRAM_MEMORY_CONFIG,
        dtype,
        compute_kernel_config,
        untilize_out,
        user_core_coord,
        get_fused_activation(activation),
        user_run_batched,
        transpose_a,
        transpose_b,
        output_tile,
        global_cb,
        sub_device_id};

    return bound_matmul(
        input_tensor_a,
        input_tensor_b,
        /*bias=*/std::nullopt,
        matmul_params,
        optional_output_tensor);
}


// ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation_types.hpp

struct MatmulInputs {
    std::vector<Tensor> input_tensors;                                // a, b, weights
    std::vector<std::optional<const Tensor>> optional_input_tensors;  // bias
    std::vector<std::optional<Tensor>> optional_output_tensors;       // output
};


// ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.hpp

struct MatmulDeviceOperation {
    using operation_attributes_t = MatmulParams;
    using tensor_args_t = MatmulInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    using program_factory_t = std::variant<
        MatmulMultiCoreProgramFactory,
        MatmulMultiCoreReuseOptimizedProgramFactory,
        MatmulMultiCoreReuseMcast1DProgramFactory,
        MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory,
        MatmulMultiCoreReuseMcast2DProgramFactory,
        MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory,
        MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args);
};


// ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp

MatmulDeviceOperation::tensor_return_value_t matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& optional_output_tensor,
    const MatmulParams& attributes) {
    MatmulParams normalized_attributes = attributes;
    if (!normalized_attributes.program_config.has_value()) {
        uint32_t bias_single_tile_size = 0;
        if (bias.has_value()) {
            auto bias_data_format =
                tt::tt_metal::datatype_to_dataformat_converter(bias.value().dtype());
            bias_single_tile_size = tt::tile_size(bias_data_format);
        }

        normalized_attributes.program_config = operations::matmul::get_program_config(
            input_tensor_a,
            input_tensor_b,
            normalized_attributes.transpose_a,
            normalized_attributes.transpose_b,
            bias_single_tile_size,
            normalized_attributes);
    }
    operations::matmul::normalize_program_config(
        normalized_attributes.program_config.value(),
        input_tensor_a.device()->compute_with_storage_grid_size());
    return ttnn::device_operation::launch<MatmulDeviceOperation>(
        normalized_attributes,
        {{input_tensor_a, input_tensor_b}, {bias}, {optional_output_tensor}});
}
"""
