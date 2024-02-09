Using ttnn
##########


Basic Examples
**************


1. Converting from and to torch tensor
--------------------------------------

.. code-block:: python

    import torch
    import ttnn

    torch_input_tensor = torch.zeros(2, 4, dtype=torch.float32)
    tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16)
    torch_output_tensor = ttnn.to_torch(tensor)

2. Running an operation on the device
--------------------------------------

.. code-block:: python

    import torch
    import ttnn

    device_id = 0
    device = ttnn.open(device_id)

    torch_input_tensor = torch.rand(2, 4, dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.exp(input_tensor)
    torch_output_tensor = ttnn.to_torch(output_tensor)

    ttnn.close(device)

3. Enabling program cache
--------------------------------------

.. code-block:: python

    import torch
    import ttnn
    import time

    ttnn.enable_program_cache()

    device_id = 0
    device = ttnn.open(device_id)

    torch_input_tensor = torch.rand(2, 4, dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Running the first time will compile the program and cache it
    start_time = time.time()
    output_tensor = ttnn.exp(input_tensor)
    torch_output_tensor = ttnn.to_torch(output_tensor)
    end_time = time.time()
    duration = end_time - start_time
    print(f"duration of the first run: {duration}")
    # stdout: duration of the first run: 0.6391518115997314

    # Running the subsequent time will use the cached program
    start_time = time.time()
    output_tensor = ttnn.exp(input_tensor)
    torch_output_tensor = ttnn.to_torch(output_tensor)
    end_time = time.time()
    duration = end_time - start_time
    print(f"duration of the second run: {duration}")
    # stdout: duration of the subsequent run: 0.0007393360137939453

    ttnn.close(device)


4. Debugging intermediate tensors
---------------------------------

.. code-block:: python

    import torch
    import ttnn

    device_id = 0
    device = ttnn.open(device_id)

    torch_input_tensor = torch.rand(32, 32, dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with ttnn.enable_debug_decorator():
        with ttnn.override_pcc_of_debug_decorator(0.9998): # This is optional in case default value of 0.9999 is too high
            output_tensor = ttnn.exp(input_tensor)
    torch_output_tensor = ttnn.to_torch(output_tensor)

    ttnn.close(device)




5. Tracing the graph of operations
----------------------------------

.. code-block:: python

    import torch
    import ttnn

    device_id = 0
    device = ttnn.open(device_id)

    with ttnn.tracer.trace():
        torch_input_tensor = torch.rand(32, 32, dtype=torch.float32)
        input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.exp(input_tensor)
        torch_output_tensor = ttnn.to_torch(output_tensor)
    ttnn.tracer.visualize(torch_output_tensor, file_name="exp_trace.svg")

    ttnn.close(device)

6. Registering a function as ttnn operation
-------------------------------------------

.. code-block:: python

    import torch
    import ttnn
    import tt_lib as ttl

    # Do not pass in or return "tt_lib.tensor.Tensor" from ttnn-registered functions

    def _new_operation_validate_input_tensors(operation_name, input_tensor: ttnn.Tensor):
        ttnn.validate_input_tensor(
            operation_name,
            input_tensor,
            ranks=(2, 3, 4),
            dtypes=(ttnn.bfloat16, ttnn.bfloat8_b),
            layouts=(ttnn.TILE_LAYOUT,),
            can_be_on_device=True,
            can_be_on_cpu=False,
        )

    @ttnn.register_operation(name="new_operation", validate_input_tensors=_new_operation_validate_input_tensors)
    def new_operation(input_tensor: ttnn.Tensor):
        original_shape = input_tensor.shape
        input_tensor = ttnn.unsqueeze_to_4D(input_tensor)
        ttl_input_tensor = input_tensor.value
        ttl_output_tensor = ttl.tensor.exp(ttl_input_tensor)
        output_tensor = ttnn.Tensor(ttl_output_tensor)
        return ttnn.reshape(output_tensor, original_shape)

    device_id = 0
    device = ttnn.open(device_id)

    torch_input_tensor = torch.rand(2, 4, dtype=torch.float32)
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = new_operation(input_tensor)
    torch_output_tensor = ttnn.to_torch(output_tensor)

    ttnn.close(device)
