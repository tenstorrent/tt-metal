# Adding a Model

## Basic Requirements

- Access to TT-Hardware
- Knowledge of PyTorch and Transformers
- Familiarity with TT-Metalium and TTNN
- See [TT-Metal README.md](https://github.com/tenstorrent/tt-metal/blob/main/README.md) for the latest updates to Tenstorrent models.

## Initial Model Bring-up

1. Run a reference model to ensure you have correctly set up your model with correct weights, attributes, etc. See the [HuggingFace PyTorch GitHub](https://github.com/huggingface/pytorch-image-models) for reference models.
2. Decompose the model into modules for function implementation. Here are examples of standard modules used in LLMs: Layernorm/RMSNorm, RotaryEmbedding, Attention, or Multilayer Perceptron (MLP).
3. Compose all modules into higher level modules. Decoder Layer and Full Model are examples of higher level modules.
4. Implement decode and prefill modes. Both must be included for the model to function.
5. Unit test each module. Start with the smallest module working up to composite modules.
6. Implement the module in TTNN, then pass the same inputs to the reference module and the TTNN module to check for correctness.
7. Create a full model test. Use real inputs to produce real outputs; for LLMs, input text to output decoded tokens.
8. Run the same inputs through the reference model and TTNN model to check the accuracy of your implementation. Teacher forcing is the ideal method to use with LLMs.
9. Generate a token from the reference model and TTNN model. Input these reference tokens into both models in the next iteration. Depending on the differences in the outputs, you can check accuracy metrics.
10. See: [LLMs Bring up in TT-NN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/LLMs/llms.md) or [ViT in TTNN](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ViT-TTNN/vit.md) for more information on these steps.

## Model Performance Optimization

Optimization tools like Metal Trace, async mode, and multiple command queues improve the performance of your model.

- Metal Trace - Metal Trace is a performance optimization tool that removes host overhead of constructing and dispatching operations.
- Async Mode - Async mode allows the host to continuously send commands without blocking util data is read back from the device.
- Multiple Command Queues - Metalium can support two command queues. These command queues are independent from each other and allow for parallel dispatches on the same device.
- See [Advanced Performance Optimizations for Models](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/AdvancedPerformanceOptimizationsForModels/AdvancedPerformanceOptimizationsForModels.md#1-metal-trace) for more information on performance optimization.

## Run the Demo

1. Download weights.
2. Setup environment variables.
3. Cache weights.
4. Execute the demo.
