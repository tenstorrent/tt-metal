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

> [!TIP]
> Use the llama3 codebase for transformer based models. Select a model most similar to the model being brought up.

- Determine which model you are configuring. Here is a list of [Tenstorrent Models](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#llms).
- Model details are available in the TT-Metalium GitHub repository: [TT-Metalium Model Demos](https://github.com/tenstorrent/tt-metal/tree/main/models/demos).

## LLM Implementation

### Baseline Validation

- Follow standard instructions to run the reference model on CPU/GPU for baseline validation.
- Ensure the model is set up correctly with proper weights, attributes, etc. before adapting it for Tenstorrent devices.

### Implementation

- For transformer based models, use [models/demos/llama3](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/llama3) codebase as reference implementation. For other model types choose a model in [/models](https://github.com/tenstorrent/tt-metal/tree/main/models) that is the most similar:
  - Make a copy of the model codebase.
  - Modify modules with model dimensions as needed.
  - Use a single device first for simpler bring-up if the models fit on a single device; Wormhole has a 12 GB DRAM storage and can support models up to 12B parameters in BFP8. If possible, use smaller version of the model that fit on a single device. The model can be scaled up in size and on more devices from here.
 
> [!NOTE]
> In the llama3 demo implementation the decode layer support batch=32. Each row is a separate user in 32x32 tiles used by the TT-Metalium stack.
> In the llama3 demo implementation, in prefill, rows map to different input tokens. Implement prefill with batch=1; prefill is compute-bound and multiple batches do not benefit performance.
> See [Converting Torch Model to TT-NN](https://docs.tenstorrent.com/docs-test/ttnn/latest/ttnn/converting_torch_model_to_ttnn.html) for model conversion.

## Systematic Component-wise Model Bring-Up

1. Bring-up decode stage modules first.
2. Bring-up each individual decode module separately.
   - Implement the module in TT-NN then pass the same inputs to the reference and the TT-NN modules to check for correctness.
   - Create a unit test with model dimensions, feed random data activations and real weights.
   - Verify that output PCC matches the reference output, use reference implementation for validation.
   - Unit tests are useful for the accuarcy/precision analysis layer.
3. Test each module by verifying the PCC.

> [!NOTE]
> Examples of standard modules used are: Layernorm/RMSNorm, RotaryEmbedding, Attention, or Multilayer Perceptron (MLP).

4. Compose all modules into higher level nodules like single layer decoder or full decoder.
5. Implement decode mode then use decode to run prefill.
6. Test the model configuration without a dedicated prefill implementation.
7. Create a full model test. Use real inputs to produce real outputs; for LLMs, input text to output decoded tokens.
8. Run the same inputs through the reference and TT-NN models to check the accuarcy of your implementation. Teacher forcing is the ideal method to use with LLMs.
9. Generate tokens from the reference and TT-NN models. Input the reference tokens into both models in the next interation. Depending on differeneces in the outputs, you can check accuarcy metrics.
10. Verify the output tokens are:
    - Meaningful and coherent.
    - Similar to reference model tokens.
   
> [!NOTE]
> Due to differences in floating point arithmetic and non-linear approximations, tokens may not be exact matches.

11. Prefill implementation:
    - Bring-up layer-by-layer similar to decode.
    - Run the entire model including prefill an decode.
12. See: [LLMs Bring-up in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md) or [ViT in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md) for more information on these steps.

## CNN Bring-up

1. Bring-up decode stage modules first.
2. See: [CNN Bring-up & Optimization in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/cnn_optimizations.md) for more information on CNN bring-up.

## Model Performance Optimization

Optimization tools like Metal Trace, async mode, and multiple command queues improve model performance. See [Advanced Performance Optimizations for Models](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md#1-metal-trace) for information on performance optimization.

## Data Parallel Implementation

Determine how many copies of a model can be run by dividing the model size by the available memory on device. For example:
  - Wormhole n150 has 12GB of storage supporting models up to roughly 12B parameters in BFP8.
  - For example, llama 3.1 model size is 8B, each Wormhole n150 can run a copy of it.
  - A TT-LoudBox (TW-02001) has four Wormhole n300s. Using data parallel scaling, it can run eight independent instances of llama 3.1 to increase throughput.
  - Large models like Falcon 40B do not fit on a single device. At lease two Wormhole n300s (24GB each) are required to run in tensor parallel scaling where single operations are distributed across devices.
  - TT-QuiteBox and TT-LouBox Systems have four Wormhole n300s; it can run two copies of Falcon 40B with each copy running on two Worhmhole n300 cards.
  - How to Run a Model Data Parallel:
    - Weights must be replicated on different devices.
    - Different inputs must be sent to different devices.
    - Can be done using the device mesh APIs in TT-NN.
    - We recommend adding data parallel support to each module separately and unit test each module before running the entire model.
  - See [Multi-Device Reference](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md#33-multi-device) for information on data parallel implementation.

## Tensor Parallel

See the [Multi-Device](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md#33-multi-device) section of LLMs in TT-NN for information on tensor parallel scaling.
