import ttnn


class ConvConfigs:
    def __init__(self):
        self.frontend = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.layer1_0_conv1 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.layer1_1_conv1 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.layer2_0_conv1 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.layer2_1_conv1 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.layer3_0_conv1 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.layer3_1_conv1 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.layer4_0_conv1 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.layer4_1_conv1 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.lat8 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.lat16 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.lat32 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.topdown_0_conv1 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.topdown_1_conv1 = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )

        self.head = ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_split_reader=True,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=256,
        )


# Instanciranje
conv_configs = ConvConfigs()
