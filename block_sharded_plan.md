now i want to support depthwise conv2d block sharded case.
you need to do 2 things:
- prepare weights
- handle weights in reader kernel (with depthwise factory support)

weights preparation is similar to width sharded preparation (please look at ttnn/cpp/ttnn/operations/conv/conv2d/DEPTHWISE_CONV2D_WIDTH_SHARDED_PLAN.md) - only thing you need to change here is to divide stick by number of cores for channels, not all cores

weights handling is similar to height sharded case, but not fully same - now we need to read weights from DRAM at just first cores in columns, and mcast it accross all columns - you can look at case in classic conv2d: ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks.cpp and ttnn/cpp/ttnn/operations/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks.cpp

please make a deatiled plan similar to ttnn/cpp/ttnn/operations/conv/conv2d/DEPTHWISE_CONV2D_WIDTH_SHARDED_PLAN.md. build, recover, and test commands are same - put them at the beggining of file with other important stuff. you won't copy that plan, just look on important things there.

when you make plan, we will execute it in new session.
