<div align="center">
  <img src="docs/images/icon.png">
</div>


# ByteMLPerf Benchmark Tool
ByteMLPerf是字节使用的一个基准套件，用于测量推理系统在各种部署场景中运行模型的速度。相比MLPerf，ByteMLPerf有如下特点：
- 模型和运行环境会更贴近真实业务；
- 对于新硬件，除了评估性能和精度之外，同时也会评估图编译的易用性、覆盖率等指标；
- 在开放Model Zoo上测试所得的性能和精度，会作为新硬件引入评估的参考；

## 类别
ByteMLPerf 基准分为三个主要类别：推理（Inference）、训练（Training）和微观性能（Micro），每个类别针对 AI 加速器性能的不同方面：

- Inference：此类别进一步细分为两个部分，以适应不同类型的模型：
  - General Perf：此部分致力于使用常见模型（如 ResNet-50 和 BERT）评估加速器的推理能力。其目的是提供加速器在一系列典型任务中性能的广泛理解。想要接入General Perf的厂商可以参考该文档接入测试：[ByteMLPerf Inference General Perf厂商接入指南](https://bytedance.feishu.cn/docs/doccno9eLS3OseTA5aMBeeQf2cf)
  - LLM Perf：专门设计用于评估加速器处理大型语言模型的能力，此部分解决了这些模型的大小和复杂性带来的独特挑战。想要接入LLM Perf的厂商可以参考该文档接入测试：[ByteMLPerf Inference LLM Perf厂商接入指南](https://bytedance.larkoffice.com/docx/ZoU7dkPXYoKtJtxlrRMcNGMwnTc)

- Micro：Micro Perf侧重于评估特定操作或“运算”（如 Gemm、Softmax 和各种通信操作）的性能，这些操作是 AI 计算的基础。这种详细级别的测试对于了解加速器在更细致的操作层面的能力和限制至关重要。想要接入Micro Perf的厂商可以参考该文档接入测试：[ByteMLPerf Micro Perf厂商接入指南](https://bytedance.us.larkoffice.com/docx/LJWvdGVAzoxXkTxF9h9uIETbsWc)

- Training：目前正在开发中的此类别旨在评估 AI 加速器在训练场景中的性能。它将提供关于加速器如何处理训练 AI 模型的计算密集过程的见解，这对于开发新的和更先进的 AI 系统至关重要。

希望评估和改进其 AI 加速器的供应商可以使用 ByteMLPerf 基准作为全面的指南。该基准不仅提供了性能和准确性评估的详细框架，还包括了 ASIC 硬件的编译器可用性和覆盖范围的考虑，确保了全面的评估方法。

更多细节您可以访问我们的官方网站:[bytemlperf.ai](https://bytemlperf.ai/)

## Vendor List
目前支持的厂商Backend如下:

| Vendor | SKU | Key Parameters | Inference(General Perf) | Inference(LLM Perf) |
| :---- | :----| :---- | :---- | :---- |
| Intel | Xeon | - | - | - |
| Stream Computing | STC P920 | <li>Computation Power:128 TFLOPS@FP16 <li> Last Level Buffer: 8MB, 256GB/s <li>Level 1 Buffer: 1.25MB, 512GB/s   <li> Memory: 16GB, 119.4GB/S <li> Host Interface：PCIe 4, 16x, 32GB/s <li> TDP: 160W | [STC Introduction](byte_infer_perf/general_perf/backends/STC/README.md) | - |
| Graphcore | Graphcore® C600 | <li>Compute: 280 TFLOPS@FP16, 560 TFLOPS@FP8 <li> In Processor Memory: 900 MB, 52 TB/s <li> Host Interface: Dual PCIe Gen4 8-lane interfaces, 32GB/s <li> TDP: 185W | [IPU Introduction](byte_infer_perf/general_perf/backends/IPU/README.md) | - |
| Moffett-AI | Moffett-AI S30 | <li>Compute: 1440 (32x-Sparse) TFLOPS@BF16, 2880 (32x-Sparse) TOPS@INT8, <li> Memory: 60 GB,  <li> Host Interface: Dual PCIe Gen4 8-lane interfaces, 32GB/s <li> TDP: 250W | [SPU Introduction](byte_infer_perf/general_perf/backends/SPU/README.md) | - |
| Habana | Gaudi2 | <li>24 Tensor Processor Cores, Dual matrix multiplication engines <li> Memory: 96 GB HBM2E, 48MB SRAM | [HPU Introduction](byte_infer_perf/general_perf/backends/HPU/README.md) | - |

## Statement
[ASF Statement on Compliance with US Export Regulations and Entity List](https://news.apache.org/foundation/entry/statement-by-the-apache-software)
