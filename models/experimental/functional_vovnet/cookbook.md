#  Effective Squeeze-Excitation Variety of View Network (ESE-VoVNet) model
VoVNet is a convolutional neural network that seeks to make DenseNet more efficient by concatenating all features only once in the last feature map, which makes input size constant and enables enlarging new output channel.


To solve the inefficiency of DenseNet, VoVNet model is proposed with energy and computation-efficient architecture comprised of One-Shot Aggregation (OSA). The OSA not only adopts the strength of DenseNet that represents diversified features with multi-receptive fields but also overcomes the inefficiency of dense connection by aggregating all features only once in the last feature maps.
https://arxiv.org/abs/1904.09730


## Vovnet submodules:
Vovnet has the following modules,

1. ConvNormAct
2. SeparableConvNormAct
3. OsaStage
4. OsaBlock
5. SequentialAppendList
6. EffectiveSEModule
7. ClassifierHead

Vovnet has a sequential implementation of ConvNormAct and two SeparableConvNormAct followed by OsaStage submodule and ends with ClassifierHead.

```py
def vovnet(
   device,
   x,
   torch_model,
   parameters,
   model,
   batch_size,
   layer_per_block=3,
   residual=False,
   depthwise=True,
   debug=False,
   bias=False,
):
   x1, x_h, x_w = conv_norm_act(
       device=device,
       x=x,
       torch_model=torch_model,
       path=f"stem.0",
       input_params=x.shape,
       conv_params=[2, 2, 1, 1],
       bias=bias,
       batch_size=batch_size,
       act_block_h=32,
   )
   x = ttnn_reshape_rm(x1, x_h, x_w, batch_size)
   x1.deallocate()
   x2, x_h, x_w = seperable_conv_norm_act(
       device=device,
       x=x,
       torch_model=torch_model,
       path=f"stem.1",
       conv_params1=[1, 1, 1, 1],
       conv_params2=[1, 1, 0, 0],
       debug=debug,
       groups=64,
       bias=bias,
       batch_size=batch_size,
       act_block_h=32,
   )
   x = ttnn_reshape_rm(x2, x_h, x_w, batch_size)
   x2.deallocate()
   x3, x_h, x_w = seperable_conv_norm_act(
       device=device,
       x=x,
       torch_model=torch_model,
       path=f"stem.2",
       conv_params1=[2, 2, 1, 1],
       conv_params2=[1, 1, 0, 0],
       debug=debug,
       groups=64,
       bias=bias,
       batch_size=batch_size,
   )
   x = ttnn_reshape_rm(x3, x_h, x_w, batch_size)
   x3.deallocate()


   groups = [128, 160, 192, 224]
   for i in range(4):
       x = osa_stage(
           device=device,
           x=x,
           torch_model=torch_model,
           path=f"stages.{i}",
           parameters=parameters.stages[i],
           model=model.stages[i],
           groups=groups[i],
           residual=residual,
           depthwise=depthwise,
           debug=debug,
           bias=bias,
           downsample=False if i == 0 else True,
           layer_per_block=layer_per_block,
           batch_size=batch_size,
       )


   x = ttnn_permute(x, device, (0, 2, 3, 1))
   x = classifier_head(
       device=device,
       x=x,
       torch_model=torch_model,
       path=f"head",
   )


   return x
```
Since the model has many convolution layers, a method for convolution2d is implemented,

```py
def conv(
   device,
   input_tensor,
   model,
   path,
   input_params,
   conv_params,
   *,
   act_block_h=None,
   reshard=False,
   deallocate=False,
   height_sharding=True,
   activation="",
   fused_op=False,
   use_shallow_conv_variant=False,
   fp32_accum=False,
   packer_l1_acc=True,
   enable_act_double_buffer=False,
   enable_split_reader=False,
   enable_subblock_padding=False,
   reallocate_halo_output=False,
   debug=False,
   groups=1,
   bias=False,
   conv=f"conv",
   math_approx=False,
   split_conv=False,
):
   if fused_op:
       weights, bias = fold_bn_to_conv_weights_bias(model, path, device, conv)
   else:
       weight = model[path + ".weight"]
       weights = ttnn.from_torch(weight, dtype=ttnn.bfloat16)
       if bias:
           bias = model[path + ".bias"]
           bias = bias.reshape(1, 1, 1, -1)
           bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)


   if not split_conv:
       kernel_size = (weights.shape[2], weights.shape[3])
       out_channels = weights.shape[0]
       reader_patterns_cache = {}
       conv_config = ttnn.Conv2dConfig(
           dtype=ttnn.bfloat16,
           weights_dtype=ttnn.bfloat16,
           math_fidelity=ttnn.MathFidelity.HiFi4,
           activation=activation,
           shard_layout=(
               ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.WIDTH_SHARDED
           ),
           input_channels_alignment=(
               16 if use_shallow_conv_variant or (input_params[-1] == 16 and input_params[1] == 115) else 32
           ),
           transpose_shards=True if deallocate else False,
           reshard_if_not_optimal=reshard,
           deallocate_activation=deallocate,
           fp32_dest_acc_enabled=fp32_accum,
           packer_l1_accum_enabled=packer_l1_acc,
           enable_act_double_buffer=enable_act_double_buffer,
           enable_split_reader=enable_split_reader,
           enable_subblock_padding=enable_subblock_padding,
           reallocate_halo_output=reallocate_halo_output,
           math_approx_mode_enabled=math_approx,
       )


       if act_block_h is not None:
           conv_config.act_block_h_override = act_block_h

       [output_tensor, _out_height, _out_width, weights, bias] = ttnn.conv2d(
           input_tensor=input_tensor,
           weight_tensor=weights,
           in_channels=input_params[-1],
           out_channels=out_channels,
           device=device,
           bias_tensor=bias if bias else None,
           kernel_size=kernel_size,
           stride=(conv_params[0], conv_params[1]),
           padding=(conv_params[2], conv_params[-1]),
           batch_size=input_params[0],
           input_height=input_params[1],
           input_width=input_params[2],
           conv_config=conv_config,
           conv_op_cache=reader_patterns_cache,
           debug=debug,
           groups=groups,
       )


   else:

       split_factor = 4 if input_params[-1] == 1024 else 2
       input_channels = input_params[-1]
       split_input_channels = input_channels // split_factor
       torch_input_tensor_nhwc = ttnn.to_torch(input_tensor)
       torch_input_tensor_nchw = torch.permute(torch_input_tensor_nhwc, (0, 3, 1, 2))
       weights = ttnn.to_torch(weights)
       torch_bias_tensor = ttnn.to_torch(bias)
       split_input_tensors = torch.split(torch_input_tensor_nchw, split_input_channels, 1)
       split_weight_tensors = torch.split(weights, weights.shape[1] // split_factor, 1)
       split_conv_config = ttnn.Conv2dConfig(
           dtype=ttnn.bfloat16,
           weights_dtype=ttnn.bfloat16,
           math_fidelity=ttnn.MathFidelity.HiFi4,
           activation=activation,
           shard_layout=(
               ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
           ),
           input_channels_alignment=(
               16 if use_shallow_conv_variant or (input_params[-1] == 16 and input_params[1] == 115) else 32
           ),
           transpose_shards=True if deallocate else False,
           reshard_if_not_optimal=reshard,
           deallocate_activation=deallocate,
           fp32_dest_acc_enabled=fp32_accum,
           packer_l1_accum_enabled=packer_l1_acc,
           reallocate_halo_output=reallocate_halo_output,
           math_approx_mode_enabled=math_approx,
       )


       for i in range(split_factor):
           tt_weight_tensor = ttnn.from_torch(split_weight_tensors[i], dtype=ttnn.bfloat16)
           tt_bias_tensor = ttnn.from_torch(torch_bias_tensor, dtype=ttnn.bfloat16)


           torch_input_tensor = torch.permute(split_input_tensors[i], (0, 2, 3, 1))
           tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)


           [tt_output_tensor_on_device, _out_height, _out_width, weights_device, bias_device] = ttnn.conv2d(
               input_tensor=tt_input_tensor,
               weight_tensor=tt_weight_tensor,
               in_channels=split_input_channels,
               out_channels=weights.shape[0],
               device=device,
               bias_tensor=tt_bias_tensor,
               kernel_size=(weights.shape[2], weights.shape[3]),
               stride=(conv_params[0], conv_params[1]),
               padding=(conv_params[2], conv_params[-1]),
               batch_size=input_params[0],
               input_height=input_params[1],
               input_width=input_params[2],
               conv_config=split_conv_config,
               conv_op_cache={},
           )
           output_tensor = ttnn.from_device(tt_output_tensor_on_device)


   return output_tensor, _out_height, _out_width
```
## ConvNormAct:

The ConvNormAct module consists of a convolution layer followed by a BatchNorm layer with ReLU activation.

```py
def conv_norm_act(
   device,
   x,
   torch_model,
   path,
   input_params,
   conv_params,
   batch_size,
   groups=1,
   debug=False,
   bias=False,
   reshard=False,
   act_block_h=None,
   height_sharding=True,
   activation="relu",
   fused_op=True,
   reallocate_halo_output=False,
   deallocate=True,
   math_approx=True,
   split_conv=False,
):
   if x.layout == ttnn.TILE_LAYOUT:
       x = ttnn_reshape_rm(x, x.shape[1], x.shape[2], batch_size, reshape=False)


   x, _x_h, _x_w = conv(
       device,
       x,
       torch_model,
       f"{path}",
       Input_params,
       conv_params,
       act_block_h=act_block_h,
       reshard=reshard,
       deallocate=deallocate,
       height_sharding=height_sharding,
       activation=activation,
       fused_op=fused_op,
       debug=debug,
       bias=bias,
       reallocate_halo_output=reallocate_halo_output,
       math_approx=math_approx,
       groups=groups,
       split_conv=split_conv,
   )

   return x, _x_h, _x_w

```

## SeperableConvNormAct
The SeperableConvNormAct module consists of Depthwise separable conv modules, a pointwise convolution layer followed by a Batchnorm with ReLU activation.

```py
def seperable_conv_norm_act(
   device,
   x,
   torch_model,
   path,
   conv_params1,
   conv_params2,
   groups,
   batch_size,
   debug=False,
   bias=False,
   reshard=False,
   act_block_h=None,
   height_sharding=True,
   activation="relu",
   fused_op=True,
   deallocate=True,
):
   if x.layout == ttnn.TILE_LAYOUT:
       x = ttnn_reshape_rm(x, x.shape[1], x.shape[2], batch_size, reshape=False)


   x, _x_h, _x_w = conv(
       device,
       x,
       torch_model,
       f"{path}.conv_dw",
       x.shape,
       conv_params1,
       act_block_h=act_block_h,
       reshard=reshard,
       deallocate=deallocate,
       height_sharding=height_sharding,
       debug=debug,
       groups=groups,
       bias=bias,
   )


   x, _x1_h, _x1_w = conv(
       device,
       x,
       torch_model,
       f"{path}",
       x.shape,
       conv_params2,
       act_block_h=act_block_h,
       reshard=reshard,
       deallocate=deallocate,
       height_sharding=height_sharding,
       activation=activation,
       fused_op=fused_op,
       debug=debug,
       conv=f"conv_pw",
       bias=bias,
       groups=1,
   )
   x = ttnn_reshape_rm(x, _x_h, _x_w, batch_size)


   return x, _x_h, _x_w
```

# OSAStage
The following is the structure of OSAStage modules,

```
	OsaStage(
		OsaBlock(
			SequentialAppendList(
				[3] SeparableConvNormAct
				ConvNormAct
				EffectiveSEModule
			)
		)
	)
```

## EffectiveSEModule
EffectiveSEModule (Squeeze-and-Excitation Networks), consists of a convolution layer and a hard sigmoid activation layer.

```py
def effective_se_module(
   device,
   torch_model,
   path,
   input_tensor,
   conv_params,
   batch_size,
   debug=False,
   fused_op=False,
   bias=False,
   split_conv=False,
):
   torch_input_tensor = ttnn.to_torch(input_tensor)
   x_se = torch_input_tensor.mean((2, 3), keepdim=True)
   x_se = ttnn.from_torch(x_se, dtype=ttnn.bfloat16)


   x_se_permute = ttnn_permute(x_se, device, (0, 2, 3, 1))
   x_se.deallocate()


   x_se_permuter = ttnn_reshape_rm(
       x_se_permute, x_se_permute.shape[2], x_se_permute.shape[-1], batch_size, reshape=False
   )
   x_se_permute.deallocate()
   if(x_se_permuter.shape[3]>512):
       x_se, _x_se_h, _x_se_w = conv(
           device,
           x_se_permuter,
           torch_model,
           f"{path}.fc",
           x_se_permuter.shape,
           conv_params,
           fused_op=fused_op,
           deallocate=False,
           height_sharding=True,
           bias=bias,
           debug=debug,
           split_conv=True,
       )
   else:
       x_se, _x_se_h, _x_se_w = conv(
           device,
           x_se_permuter,
           torch_model,
           f"{path}.fc",
           x_se_permuter.shape,
           conv_params,
           fused_op=fused_op,
           deallocate=False,
           height_sharding=True,
           bias=bias,
           debug=debug,
           split_conv=False,
       )


   x_se_permute = ttnn_permute(x_se, device, (0, 3, 1, 2))

   return input_tensor * ttnn.hardsigmoid(x_se_permute)
```

## ClassifierHead
The ClassifierHead module consists of Global average pool followed by linear layer.

```py
def classifier_head(
   device,
   x,
   torch_model,
   path,
):
   x = ttnn.global_avg_pool2d(x)


   weights_tensor = torch_model[path + ".fc.weight"]
   weights_tensor = weights_tensor.permute(1, 0)
   weights_tensor = ttnn.from_torch(weights_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


   bias_tensor = torch_model[path + ".fc.bias"]
   bias_tensor = ttnn.from_torch(bias_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
   x = ttnn.linear(
       x,
       weights_tensor,
       bias=bias_tensor,
       memory_config=ttnn.L1_MEMORY_CONFIG,
   )

   return x
```

To be added - optimisation techniques
