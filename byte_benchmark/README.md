<div align="center">
  <img src="docs/images/icon.png">
</div>


# ByteMLPerf Benchmark Tool
ByteMLPerf is an AI Accelerator Benchmark that focuses on evaluating AI Accelerators from practical production perspective, including the ease of use and versatility of software and hardware. Byte MLPerf has the following characteristics:
- Models and runtime environments are more closely aligned with practical business use cases.
- For ASIC hardware evaluation, besides evaluate performance and accuracy, it also measure metrics like compiler usability and coverage.
- Performance and accuracy results obtained from testing on the open Model Zoo serve as reference metrics for evaluating ASIC hardware integration.

## Category
The ByteMLPerf benchmark is structured into three main categories: Inference, Training, and Micro, each targeting different aspects of AI accelerator performance:

- Inference: This category is subdivided into two distinct sections to cater to different types of models:

  - General Performance: This section is dedicated to evaluating the inference capabilities of accelerators using common models such as ResNet-50 and BERT. It aims to provide a broad understanding of the accelerator's performance across a range of typical tasks. Vendors can refer to this document for guidance on building general perf backend: [ByteMLPerf General Perf Guide](https://bytedance.us.feishu.cn/docx/L98Mdw3J6obMtJxeRBzuHeRbsof) [[中文版](https://bytedance.feishu.cn/docs/doccno9eLS3OseTA5aMBeeQf2cf#TDK8of)]

  - Large Language Model (LLM) Performance: Specifically designed to assess the capabilities of accelerators in handling large language models, this section addresses the unique challenges posed by the size and complexity of these models. Vendors can refer to this document for guidance on building llm perf backend: [ByteMLPerf LLM Perf Guide](https://bytedance.larkoffice.com/docx/ZoU7dkPXYoKtJtxlrRMcNGMwnTc) [[中文版](https://bytedance.larkoffice.com/docx/ZoU7dkPXYoKtJtxlrRMcNGMwnTc)]

- Micro: The Micro category focuses on the performance of specific operations or "ops" that are fundamental to AI computations, such as Gemm, Softmax, and various communication operations. This granular level of testing is crucial for understanding the capabilities and limitations of accelerators at a more detailed operational level. Vendors can refer to this document for guidance on building micro perf backend: [ByteMLPerf Micro Perf Guide](https://bytedance.us.larkoffice.com/docx/EpjFdSpRsoOIHWxtKgjuRsMPsFB)[[中文版](https://bytedance.us.larkoffice.com/docx/LJWvdGVAzoxXkTxF9h9uIETbsWc)]

- Training: Currently under development, this category aims to evaluate the performance of AI accelerators in training scenarios. It will provide insights into how well accelerators can handle the computationally intensive process of training AI models, which is vital for the development of new and more advanced AI systems.

Vendors looking to evaluate and improve their AI accelerators can utilize the ByteMLPerf benchmark as a comprehensive guide. The benchmark not only offers a detailed framework for performance and accuracy evaluation but also includes considerations for compiler usability and coverage for ASIC hardware, ensuring a holistic assessment approach.

For more details, you can visit our offical website here: [bytemlperf.ai](https://bytemlperf.ai/)

## Vendor List
ByteMLPerf Vendor Backend List will be shown below

| Vendor | SKU | Key Parameters | Inference(General Perf) | Inference(LLM Perf) |
| :---- | :----| :---- | :---- | :---- |
| Intel | Xeon | - | - | - |
| Stream Computing | STC P920 | <li>Computation Power:128 TFLOPS@FP16 <li> Last Level Buffer: 8MB, 256GB/s <li>Level 1 Buffer: 1.25MB, 512GB/s   <li> Memory: 16GB, 119.4GB/S <li> Host Interface：PCIe 4, 16x, 32GB/s <li> TDP: 160W | [STC Introduction](byte_infer_perf/general_perf/backends/STC/README.md) | - |
| Graphcore | Graphcore® C600 | <li>Compute: 280 TFLOPS@FP16, 560 TFLOPS@FP8 <li> In Processor Memory: 900 MB, 52 TB/s <li> Host Interface: Dual PCIe Gen4 8-lane interfaces, 32GB/s <li> TDP: 185W | [IPU Introduction](byte_infer_perf/general_perf/backends/IPU/README.md) | - |
| Moffett-AI | Moffett-AI S30 | <li>Compute: 1440 (32x-Sparse) TFLOPS@BF16, 2880 (32x-Sparse) TOPS@INT8, <li> Memory: 60 GB,  <li> Host Interface: Dual PCIe Gen4 8-lane interfaces, 32GB/s <li> TDP: 250W | [SPU Introduction](byte_infer_perf/general_perf/backends/SPU/README.md) | - |
| Habana | Gaudi2 | <li>24 Tensor Processor Cores, Dual matrix multiplication engines <li> Memory: 96 GB HBM2E, 48MB SRAM | [HPU Introduction](byte_infer_perf/general_perf/backends/HPU/README.md) | - |

## Statement
[ASF Statement on Compliance with US Export Regulations and Entity List](https://news.apache.org/foundation/entry/statement-by-the-apache-software)
