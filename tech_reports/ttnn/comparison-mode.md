# TT-NN Comparison Mode
Sometimes when debugging a long sequence of operations (such as a model forward pass) it is useful to be able to automatically check the correctness of each individual operation against a known reference.

TT-NN provides a mechanism for doing this called "comparison mode". In this mode, the runtime will automatically insert correctness checks after each operation and report to the user if the output does not match the expected value.

Comparisons are currently done by computing PCC between the operation output and the reference operation output. The sensitivity of this comparison can be modified using the TTNN configuration (see below).

## How to Use it?

To enable comparison mode, the user must update their TTNN configuration. This can be done easily through the `TTNN_CONFIG_OVERRIDES` environment variable:

```sh
export TTNN_CONFIG_OVERRIDES='{
    "enable_fast_runtime_mode": false,
    "enable_comparison_mode": true,
    "comparison_mode_should_raise_exception": true,
    "comparison_mode_pcc": 0.999
}'
```

- `enable_fast_runtime_mode`: This option controls whether fast runtime mode is enabled or not. Comparison mode currently requires that this feature is disabled. Defaults to `true`.
- `enable_comparison_mode`: When set to `true`, comparison mode is activated. Defaults to `false`.
- `comparison_mode_should_raise_exception`: When set to `true`, comparison mode will raise an exception, ending op execution, when encountering an op that fails comparison checks. This is useful for localizing a failing operations when you are running a full model, or want to integrate comparison mode checks into a unit test. Defaults to `false`.
- `comparison_mode_pcc`: Sets the PCC threshold used in comparison mode.

## ttnn.conv2d and comparison mode

`ttnn.conv2d` operates with weights in the channel-last folded format: `[1, 1, NHW, C]`.
If a tensor passed to the operation resides on the host or is not already in this format, the operator internally prepares the weights tensor.

When implementing CNN models on TT hardware, the standard practice is to retrieve and reuse these internally prepared tensors in subsequent runs to improve performance.

However, weights reuse can cause comparison mode to fail when applied to such models. This happens because the conv2d golden function expects weights in the standard channel-first unfolded format.

To enable debugging models in comparison mode, weights reuse must be disabled.
The method for disabling reuse is left to the model writer. For example, in the `SDXL` model, this is controlled via a pytest fixture `debug_mode`, which—when enabled—prevents reuse of convolution weights.
