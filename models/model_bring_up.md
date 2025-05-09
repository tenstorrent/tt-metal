# Model Bring-Up

This guide covers basic steps for model bring-up on Tenstorrent devices.

## Basic Requirements

- Access to TT-Hardware | [Buy TT-Hardware](https://tenstorrent.com/hardware/wormhole)
- Knowledge of PyTorch and transformers.
- Familiarity with [TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/index.html) and [TT-NN](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html).
- See: [TT-Metalium README.md](https://github.com/tenstorrent/tt-metal/blob/main/README.md) for the latest updates to Tenstorrent models.
- See: [TT-Metalium Tech Reports](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#tt-metalium-tech-reports) for information on TT-Metalium.

## Run a Demo

After setting up the environment correctly, run a demo to test the environment.

- Determine which model you are configuring. Here is a list of [Tenstorrent Models](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#llms).
- Model details are available in the TT-Metalium GitHub repository: [TT-Metalium Model Demos](https://github.com/tenstorrent/tt-metal/tree/main/models/demos).

## LLM Implementation

### Baseline Validation

- Follow standard instructions to run the reference model on CPU/GPU for baseline validation.
- Ensure the model is set up correctly with proper weights, attributes, etc. before adapting it for Tenstorrent devices.

### Implementation

- For transformer based models, use [models/tt_transformers](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers) codebase as reference implementation.

- For other models, choose the model from [/models](https://github.com/tenstorrent/tt-metal/tree/main/models) that is the most similar:
  - Most transformer based models can be run by changing the tensor dimensions of llama3 and can be added as a new model configuration to the existing codebase. For other models, make a copy of the model codebase for advanced changes.
  - Modify modules with model dimensions as needed.
  - First use a single device for simpler bring-up if models can fit on that single device; Wormhole has 12 GB DRAM storage and can support models of up to roughly 12B parameters in BFP8. If possible, use a smaller version of the model that fits on a single device. The model can be scaled up in size and on more devices.

> [!NOTE]
> In the llama3 demo implementation the decode stage supports batch=32. Each row is a separate user in 32x32 tiles used by the TT-Metalium stack. The prefill stage supports batch=1 where rows map to different iput tokens. Because prefill is compute-bound, multiple batches do not benefit performance. See [Converting Torch Model to TT-NN](https://docs.tenstorrent.com/docs-test/ttnn/latest/ttnn/converting_torch_model_to_ttnn.html) for model conversion.

## Systematic Component-wise Large Language Model (LLM) Bring-Up

1. Bring-up decode stage modules first.
2. Bring-up each individual decode module separately.
   - Implement the module in TT-NN then pass the same inputs to the reference and TT-NN modules to check for correctness.
   - Create a unit test with model dimensions, feed random data activations and real weights.
   - Verify that output PCC matches the reference output, use reference implementation for validation.
   - Unit tests are useful for the accuracy/precision analysis layer.

> [!NOTE]
> Examples of standard modules used are: Layernorm/RMSNorm, RotaryEmbedding, Attention, or Multilayer Perceptron (MLP).

4. Compose all modules into higher level modules like single layer decoder or full decoder.
5. Implement decode mode then use decode to run prefill.
6. Test the model configuration without a dedicated prefill implementation.
7. Create a full model test. Use real inputs to produce real outputs; for LLMs, input text to output decoded tokens.
8. Run the same inputs through the reference and TT-NN models to check the accuracy of your implementation. Teacher forcing is the ideal method to use with LLMs.
9. Generate tokens from the reference and TT-NN models. Input the reference tokens into both models in the next interation. Depending on differences in the outputs, you can check accuracy metrics.
10. Verify the output tokens are:
    - Meaningful and coherent.
    - Similar to reference model tokens.
    - Measure the top1/top5 accuracy of the generated tokens w.r.t. to the reference tokens.

> [!NOTE]
> Due to differences in floating point arithmetic and non-linear approximations, tokens may not be exact matches.

11. Prefill implementation:
    - Bring-up layer-by-layer similar to decode.
    - Run the entire model including prefill and decode.
12. See: [LLMs Bring-up in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md) or [ViT in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md) for more information on these steps.

## CNN Bring-up

1. Bring-up the model module by module.
2. See: [CNN Bring-up & Optimization in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/cnn_optimizations.md) for more information on CNN bring-up.

## Model Performance Optimization

Optimization techniques like Metal Trace, async mode, and multiple command queues improve model performance. See [Advanced Performance Optimizations for Models](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md#1-metal-trace) for information on performance optimization.

## Data Parallel Implementation

Determine how many copies of a model can be run by dividing the model size by the available memory on device. For example:
  - Wormhole n150 has 12GB of storage supporting models up to roughly 12B parameters in BFP8.
  - Wormhole n300 has 24GB of storage supporting models up to roughly 12B parameters in BFP8.
  - Each Wormhole n150 can run a copy of the llama3.1 8B model using BFP8 weights (~8GB of model weights).
  - A TT-LoudBox (TW-02001) has four Wormhole n300s. Using data parallel scaling, eight independent instances of the llama3.1 8B model can be run.
  - Large models like Falcon 40B do not fit on a single device. At least two Wormhole n300s (24GB each) are required to run in tensor parallel scaling where single operations are distributed across devices.
  - TT-LoudBox and the TT-QuietBox (Wormhole) Systems have four Wormhole n300s; each system can run two copies of Falcon 40B with each copy running on two Wormhole n300 cards.
  - The TT-QuietBox (Blackhole) System has four p150cs; this system can run two copies of Falcon 40B with each copy running on two Blackhole p150c cards.
  - How to Run a Model Data Parallel:
    - Weights must be replicated on different devices.
    - Different inputs must be sent to different devices.
    - Use device mesh APIs in TT-NN.
    - We recommend adding data parallel support to each module separately and unit test each module before running the entire model.
  - See [Multi-Device Reference](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md#33-multi-device) for information on data parallel implementation.

## Tensor Parallel

See the [Multi-Device](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md#33-multi-device) section of LLMs in TT-NN for information on tensor parallel scaling.
