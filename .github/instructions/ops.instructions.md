---
applyTo: "ttnn/cpp/ttnn/operations/**"
---

For TT-NN Operations code:

I'd like you to comment on the PR after you compile the following instructions into a single comment:

- Identify the list of operations that is affected by the code changes.
- Use the following list to identify which models are affected by the OP change.

# 📊 Device Operations Report

### `BcastDeviceOperation`
- bert

### `BinaryDeviceOperation`
- efficientnet_b0
- mobilenetv2
- resnet
- segformer
- ufld_v2

### `BinaryNgDeviceOperation`
- bert
- bert_tiny
- distilbert
- mnist
- segformer
- sentence_bert
- swin_s
- ufld_v2

### `ConcatDeviceOperation`
- distilbert
- segformer
- swin_s
- vanilla_unet
- vgg_unet

### `Conv2dDeviceOperation`
- efficientnet_b0
- mobilenetv2
- resnet
- segformer
- swin_s
- ufld_v2
- vanilla_unet
- vgg
- vgg_unet

### `ConvertToCHWDeviceOperation`
- vanilla_unet

### `ConvertToHWCDeviceOperation`
- vanilla_unet

### `EmbeddingsDeviceOperation`
- bert
- bert_tiny
- distilbert
- sentence_bert

### `FillPadDeviceOperation`
- swin_s

### `Fold`
- segformer

### `HaloDeviceOperation`
- efficientnet_b0
- mobilenetv2
- resnet
- segformer
- swin_s
- ufld_v2
- vanilla_unet
- vgg
- vgg_unet

### `InterleavedToShardedDeviceOperation`
- bert
- efficientnet_b0
- segformer
- sentence_bert
- swin_s
- ufld_v2
- vanilla_unet
- vgg
- vgg_unet

### `LayerNormDeviceOperation`
- bert
- bert_tiny
- distilbert
- segformer
- sentence_bert
- swin_s

### `Matmul`
- bert
- bert_tiny
- distilbert
- efficientnet_b0
- mnist
- mobilenetv2
- resnet
- segformer
- sentence_bert
- swin_s
- ufld_v2
- vanilla_unet
- vgg
- vgg_unet

### `MoveDeviceOperation`
- efficientnet_b0
- mobilenetv2
- segformer
- swin_s
- ufld_v2
- vanilla_unet
- vgg
- vgg_unet

### `NLPConcatHeadsDeviceOperation`
- bert
- segformer
- sentence_bert

### `NlpCreateHeadsDeviceOperation`
- distilbert

### `NlpCreateHeadsSegformerDeviceOperation`
- segformer

### `PadDeviceOperation`
- efficientnet_b0
- resnet
- segformer
- swin_s
- ufld_v2
- vgg_unet

### `PermuteDeviceOperation`
- bert_tiny
- distilbert
- swin_s
- vgg

### `Pool2D`
- mobilenetv2
- resnet
- swin_s
- ufld_v2
- vanilla_unet
- vgg
- vgg_unet

### `ReduceDeviceOperation`
- efficientnet_b0
- sentence_bert

### `RepeatDeviceOperation`
- sentence_bert

### `ReshapeDeviceOperation`
- bert_tiny
- mnist
- sentence_bert
- swin_s
- ufld_v2
- vgg

### `ReshardDeviceOperation`
- efficientnet_b0
- mobilenetv2
- resnet
- segformer
- sentence_bert
- swin_s
- ufld_v2
- vgg_unet

### `ShardedToInterleavedDeviceOperation`
- bert
- efficientnet_b0
- segformer
- sentence_bert
- swin_s
- ufld_v2
- vanilla_unet
- vgg_unet

### `SliceDeviceOperation`
- distilbert
- resnet
- swin_s

### `SoftmaxDeviceOperation`
- bert
- bert_tiny
- distilbert
- mnist
- segformer
- sentence_bert
- swin_s

### `SplitFusedQKVAndSplitHeadsDeviceOperation`
- bert
- sentence_bert

### `Tilize`
- bert_tiny
- distilbert
- resnet
- segformer
- sentence_bert
- swin_s
- ufld_v2
- vanilla_unet

### `TilizeWithValPadding`
- efficientnet_b0
- mnist
- sentence_bert
- swin_s
- vgg

### `TransposeDeviceOperation`
- bert_tiny
- distilbert
- efficientnet_b0
- resnet
- segformer
- swin_s
- ufld_v2
- vgg_unet

### `TypecastDeviceOperation`
- sentence_bert

### `UnaryDeviceOperation`
- distilbert
- efficientnet_b0
- mnist
- sentence_bert
- ufld_v2
- vgg
- vgg_unet

### `Untilize`
- bert_tiny
- distilbert
- segformer
- swin_s
- vanilla_unet

### `UntilizeWithUnpadding`
- bert_tiny
- distilbert
- efficientnet_b0
- resnet
- sentence_bert
- swin_s

### `UpsampleOperation`
- segformer



- Identify Action Workflows that run the models that were impacted by the changes. Add the links to these workflows using url prefix 
https://github.com/tenstorrent/tt-metal/actions/workflows/